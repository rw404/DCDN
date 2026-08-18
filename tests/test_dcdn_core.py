from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dcdn.core import DCDNCore, DCDNCoreBlock  # noqa: E402
from dcdn.validation import (  # noqa: E402
    OperandValidationError,
    canonical_denominator_mask,
    canonical_schema_sha256,
    validate_materialized_operands,
    validate_schema,
)


def _core(
    *,
    features: int = 3,
    layers: int = 1,
    mask: list[bool] | None = None,
    eps: float = 1e-2,
    runtime_checks: bool = True,
    schema_hash: str | None = None,
) -> DCDNCore:
    if mask is None:
        mask = [True] + [False] * (features - 1)
    return DCDNCore(
        features_count=features,
        num_layers=layers,
        denominator_mask=mask,
        div_eps=eps,
        runtime_value_checks=runtime_checks,
        feature_schema_sha256=schema_hash,
    )


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"features_count": 0}, ValueError),
        ({"features_count": -1}, ValueError),
        ({"features_count": True}, ValueError),
        ({"num_layers": 0}, ValueError),
        ({"num_layers": -1}, ValueError),
        ({"div_eps": 0.0}, ValueError),
        ({"div_eps": float("inf")}, ValueError),
        ({"product_init_scale": -0.1}, ValueError),
        ({"product_init_scale": float("nan")}, ValueError),
        ({"div_init_scale": -0.1}, ValueError),
        ({"runtime_value_checks": 1}, TypeError),
    ],
)
def test_constructor_rejects_invalid_scalar_config(kwargs, exception):
    config = {
        "features_count": 2,
        "num_layers": 1,
        "denominator_mask": [True, False],
    }
    config.update(kwargs)
    with pytest.raises(exception):
        DCDNCore(**config)


def test_constructor_rejects_invalid_mask_and_schema_hash():
    with pytest.raises(TypeError, match="booleans"):
        _core(features=2, mask=[1, 0])
    with pytest.raises(ValueError, match="2 entries"):
        _core(features=2, mask=[True])
    with pytest.raises(ValueError, match="SHA-256"):
        _core(features=2, mask=[True, False], schema_hash="not-a-hash")


def test_seeded_initialization_is_repeatable_and_biases_are_zero():
    torch.manual_seed(1234)
    first = _core(features=4, layers=2, mask=[True, False, True, False])
    torch.manual_seed(1234)
    second = _core(features=4, layers=2, mask=[True, False, True, False])
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter)
    for layer in first.layers:
        torch.testing.assert_close(
            layer.product_block.bias, torch.zeros_like(layer.product_block.bias)
        )
        torch.testing.assert_close(
            layer.division_block.bias, torch.zeros_like(layer.division_block.bias)
        )


def test_exact_one_layer_manual_recurrence_and_masked_output_coordinate():
    model = _core(features=2, mask=[True, False], eps=0.1)
    layer = model.layers[0]
    with torch.no_grad():
        layer.product_block.weight.copy_(torch.tensor([[0.5, -0.25], [0.1, 0.3]]))
        layer.product_block.bias.copy_(torch.tensor([0.2, -0.4]))
        layer.division_block.weight.copy_(torch.tensor([[0.7, -0.2], [-0.6, 0.9]]))
        layer.division_block.bias.copy_(torch.tensor([0.3, 0.8]))

    x_0 = torch.tensor([[2.0, -3.0], [0.05, 4.0]])
    inv = torch.tensor([[0.5, 0.0], [10.0, 0.0]])
    expected = (
        x_0
        + x_0 * F.linear(x_0, layer.product_block.weight, layer.product_block.bias)
        + inv * F.linear(x_0, layer.division_block.weight, layer.division_block.bias)
    )
    actual = model(x_0)
    torch.testing.assert_close(model.reciprocal_carrier(x_0), inv)
    torch.testing.assert_close(actual, expected)


def test_depth_uses_one_fixed_anchor_and_one_fixed_carrier():
    model = _core(features=2, layers=2, mask=[True, True], eps=1e-4)
    with torch.no_grad():
        model.layers[0].product_block.weight.copy_(
            torch.tensor([[0.2, 0.1], [-0.1, 0.3]])
        )
        model.layers[0].product_block.bias.copy_(torch.tensor([0.05, -0.02]))
        model.layers[0].division_block.weight.copy_(
            torch.tensor([[0.4, -0.2], [0.25, 0.1]])
        )
        model.layers[0].division_block.bias.copy_(torch.tensor([0.1, 0.2]))
        model.layers[1].product_block.weight.copy_(
            torch.tensor([[0.3, -0.15], [0.2, 0.05]])
        )
        model.layers[1].product_block.bias.copy_(torch.tensor([-0.03, 0.04]))
        model.layers[1].division_block.weight.copy_(
            torch.tensor([[0.2, 0.35], [-0.4, 0.15]])
        )
        model.layers[1].division_block.bias.copy_(torch.tensor([0.07, -0.08]))

    x_0 = torch.tensor([[1.5, 2.5], [0.75, 4.0]])
    fixed_inv = x_0.reciprocal()
    manual_states = [x_0]
    state = x_0
    for layer in model.layers:
        state = (
            state
            + x_0
            * F.linear(state, layer.product_block.weight, layer.product_block.bias)
            + fixed_inv
            * F.linear(state, layer.division_block.weight, layer.division_block.bias)
        )
        manual_states.append(state)
    torch.testing.assert_close(model(x_0), manual_states[-1])

    first_state = manual_states[1]
    wrong_dynamic_inv = first_state.clamp_min(model.div_eps).reciprocal()
    last = model.layers[1]
    wrong_second_state = (
        first_state
        + x_0
        * F.linear(first_state, last.product_block.weight, last.product_block.bias)
        + wrong_dynamic_inv
        * F.linear(first_state, last.division_block.weight, last.division_block.bias)
    )
    assert not torch.allclose(model(x_0), wrong_second_state)


def test_reciprocal_floor_boundaries_and_negative_non_denominator():
    model = _core(
        features=4,
        mask=[True, True, True, False],
        eps=1e-2,
    )
    x_0 = torch.tensor([[0.005, 0.01, 2.0, -3.0]])
    expected = torch.tensor([[100.0, 100.0, 0.5, 0.0]])
    torch.testing.assert_close(model.reciprocal_carrier(x_0), expected)
    assert torch.isfinite(model(x_0)).all()
    large = torch.tensor([[1.0, 1.0, 1e20, -3.0]])
    large_carrier = model.reciprocal_carrier(large)
    assert large_carrier[0, 2] == pytest.approx(1e-20)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_active_invalid_denominator_debug_error_or_optimized_nan_sentinel(bad_value):
    debug_model = _core(features=2, mask=[True, False], runtime_checks=True)
    x_0 = torch.tensor([[bad_value, -2.0]])
    with pytest.raises(ValueError, match="finite and positive"):
        debug_model(x_0)

    optimized_model = _core(
        features=2,
        mask=[True, False],
        runtime_checks=False,
    )
    carrier = optimized_model.reciprocal_carrier(x_0)
    assert torch.isnan(carrier[0, 0])
    assert carrier[0, 1] == 0
    assert torch.isnan(optimized_model(x_0)).any()


def test_optimized_path_does_not_invoke_synchronous_torch_any(monkeypatch):
    model = _core(features=2, mask=[True, False], runtime_checks=False)

    def forbidden_torch_any(*_args, **_kwargs):
        raise AssertionError("optimized DCDN path invoked torch.any")

    monkeypatch.setattr(torch, "any", forbidden_torch_any)
    valid_output = model(torch.tensor([[2.0, -3.0]]))
    assert torch.isfinite(valid_output).all()
    invalid_carrier = model.reciprocal_carrier(torch.tensor([[-1.0, -3.0]]))
    assert torch.isnan(invalid_carrier[0, 0])


def test_missing_denominator_requires_finite_fallback_but_can_be_nonpositive():
    model = _core(features=2, mask=[False, True])
    validity = torch.tensor([[True, False]])
    finite_fallback = torch.tensor([[1.0, -9.0]])
    carrier = model.reciprocal_carrier(finite_fallback, validity)
    torch.testing.assert_close(carrier, torch.zeros_like(carrier))

    nonfinite_fallback = torch.tensor([[1.0, float("nan")]])
    with pytest.raises(ValueError, match="finite fallback"):
        model.reciprocal_carrier(nonfinite_fallback, validity)


def test_missing_fallback_isolated_only_from_reciprocal_not_residual_product():
    model = _core(features=2, mask=[False, True])
    layer = model.layers[0]
    with torch.no_grad():
        layer.product_block.weight.copy_(torch.eye(2))
        layer.product_block.bias.zero_()
        layer.division_block.weight.fill_(2.0)
        layer.division_block.bias.fill_(1.0)
    validity = torch.tensor([[True, False]])
    first = torch.tensor([[1.0, 2.0]])
    second = torch.tensor([[1.0, 5.0]])
    torch.testing.assert_close(
        model.reciprocal_carrier(first, validity),
        model.reciprocal_carrier(second, validity),
    )
    assert not torch.allclose(model(first, validity), model(second, validity))


def test_mask_and_validity_isolate_reciprocal_gradients():
    model = _core(features=3, mask=[True, False, True], eps=1e-3)
    x_0 = torch.tensor([[2.0, 3.0, 4.0]], requires_grad=True)
    validity = torch.tensor([[True, True, False]])
    carrier = model.reciprocal_carrier(x_0, validity)
    carrier.sum().backward()
    expected_gradient = torch.tensor([[-0.25, 0.0, 0.0]])
    torch.testing.assert_close(x_0.grad, expected_gradient)
    torch.testing.assert_close(carrier, torch.tensor([[0.5, 0.0, 0.0]]))


def test_mask_and_row_validity_isolate_division_parameter_gradients():
    torch.manual_seed(17)
    model = _core(features=2, mask=[True, False])
    x_0 = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    validity = torch.tensor([[True, True], [False, True]])
    model(x_0, validity).sum().backward()
    batch_weight_gradient = model.layers[0].division_block.weight.grad.clone()
    batch_bias_gradient = model.layers[0].division_block.bias.grad.clone()
    torch.testing.assert_close(
        batch_weight_gradient[1], torch.zeros_like(batch_weight_gradient[1])
    )
    torch.testing.assert_close(
        batch_bias_gradient[1], torch.zeros_like(batch_bias_gradient[1])
    )

    comparison = deepcopy(model)
    comparison.zero_grad(set_to_none=True)
    comparison(x_0[:1], validity[:1]).sum().backward()
    torch.testing.assert_close(
        batch_weight_gradient[0],
        comparison.layers[0].division_block.weight.grad[0],
    )
    torch.testing.assert_close(
        batch_bias_gradient[0],
        comparison.layers[0].division_block.bias.grad[0],
    )


def test_shape_dtype_precision_and_validity_contracts_fail_loud():
    model = _core(features=2, mask=[True, False])
    with pytest.raises(ValueError, match="shape"):
        model(torch.ones(2))
    with pytest.raises(ValueError, match="shape"):
        model(torch.ones(2, 3))
    with pytest.raises(TypeError, match="FP16"):
        model(torch.ones(2, 2, dtype=torch.float16))
    with pytest.raises(TypeError, match="FP32 or BF16"):
        model(torch.ones(2, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="match x_0.shape"):
        model(torch.ones(2, 2), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        model(torch.ones(2, 2), torch.ones(2, 2))
    meta_validity = torch.ones(2, 2, dtype=torch.bool, device="meta")
    with pytest.raises(ValueError, match="x_0.device"):
        model(torch.ones(2, 2), meta_validity)

    half_model = _core(features=2, mask=[True, False]).half()
    with pytest.raises(TypeError, match="FP16"):
        half_model(torch.ones(2, 2, dtype=torch.float16))


def test_block_rejects_broadcastable_mismatches():
    block = DCDNCoreBlock(features_count=2)
    x_0 = torch.ones(3, 2)
    with pytest.raises(ValueError, match="identical shapes"):
        block(x_0, torch.ones(1, 2), torch.ones(3, 2))
    with pytest.raises(TypeError, match="identical dtypes"):
        block(x_0, x_0.to(torch.bfloat16), x_0)


def test_single_canonical_mask_buffer_and_core_has_no_hybrid_side_path():
    model = _core(features=3, layers=4, mask=[True, False, True])
    mask_keys = [key for key in model.state_dict() if "denominator_mask" in key]
    assert mask_keys == ["_denominator_mask"]
    assert not any(
        isinstance(module, nn.BatchNorm1d | nn.Dropout) for module in model.modules()
    )
    for forbidden_name in ("fc_block", "projector", "concat"):
        assert not hasattr(model, forbidden_name)

    returned_mask = model.denominator_mask
    returned_mask[0] = False
    assert model.denominator_mask.tolist() == [True, False, True]


def test_core_parameter_formula_matches_trainable_parameters():
    features = 7
    layers = 3
    model = _core(
        features=features,
        layers=layers,
        mask=[True, False, True, False, True, False, False],
    )
    expected = DCDNCore.parameter_count_formula(features, layers)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert actual == expected == layers * 2 * features * (features + 1)


def test_serialization_metadata_round_trip_and_contract_rejections():
    names = ("count", "shows", "duration")
    schema_hash = canonical_schema_sha256(names)
    model = _core(
        features=3,
        layers=2,
        mask=[False, True, True],
        eps=0.02,
        schema_hash=schema_hash,
    )
    metadata = model.serialization_metadata()
    assert metadata["architecture"] == "DCDN-Core-L2"
    assert metadata["fixed_anchor"] is True
    assert metadata["precision"]["parameter_dtype"] == "float32"
    assert metadata["precision"]["reciprocal_compute_dtype"] == "float32"
    assert len(model.serialization_metadata_sha256) == 64

    payload = model.state_dict_with_metadata()
    restored = DCDNCore.from_serialization_metadata(metadata)
    restored.load_state_dict_with_metadata(payload)
    x_0 = torch.tensor([[2.0, 4.0, 5.0]])
    torch.testing.assert_close(model(x_0), restored(x_0))

    tamper_cases = []
    wrong_order = deepcopy(metadata)
    wrong_order["feature_schema_sha256"] = canonical_schema_sha256(
        ("shows", "count", "duration")
    )
    tamper_cases.append(wrong_order)
    wrong_mask = deepcopy(metadata)
    wrong_mask["denominator_mask"] = [True, False, True]
    tamper_cases.append(wrong_mask)
    wrong_eps = deepcopy(metadata)
    wrong_eps["div_eps"] = 0.03
    tamper_cases.append(wrong_eps)
    wrong_depth = deepcopy(metadata)
    wrong_depth["num_layers"] = 3
    tamper_cases.append(wrong_depth)
    wrong_precision = deepcopy(metadata)
    wrong_precision["precision"]["allowed_input_dtypes"] = ["float32"]
    tamper_cases.append(wrong_precision)
    for tampered in tamper_cases:
        with pytest.raises(ValueError, match="metadata mismatch"):
            model.validate_serialization_metadata(tampered)

    corrupt_digest_payload = dict(payload)
    corrupt_digest_payload["metadata_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        restored.load_state_dict_with_metadata(corrupt_digest_payload)

    bf16_model = _core(
        features=3,
        layers=2,
        mask=[False, True, True],
        eps=0.02,
        schema_hash=schema_hash,
    ).to(dtype=torch.bfloat16)
    bf16_payload = bf16_model.state_dict_with_metadata()
    assert bf16_payload["metadata"]["precision"]["parameter_dtype"] == "bfloat16"
    with pytest.raises(ValueError, match="metadata mismatch"):
        model.validate_serialization_metadata(bf16_payload["metadata"])
    restored_bf16 = DCDNCore.from_serialization_metadata(bf16_payload["metadata"])
    restored_bf16.load_state_dict_with_metadata(bf16_payload)
    assert {parameter.dtype for parameter in restored_bf16.parameters()} == {
        torch.bfloat16
    }


def test_finite_forward_backward_and_optimizer_update():
    torch.manual_seed(19)
    model = _core(features=3, layers=2, mask=[False, True, True])
    x_0 = torch.rand(16, 3) + 0.2
    target = torch.randn(16, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x_0)
    loss = F.mse_loss(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_bfloat16_carrier_and_cpu_amp_path_are_finite():
    model_bf16 = _core(features=2, mask=[True, False], eps=1e-2).to(
        dtype=torch.bfloat16
    )
    x_bf16 = torch.tensor([[0.003, -2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    carrier = model_bf16.reciprocal_carrier(x_bf16)
    expected = torch.tensor([[100.0, 0.0], [1.0 / 3.0, 0.0]]).to(dtype=torch.bfloat16)
    torch.testing.assert_close(carrier, expected, rtol=2e-2, atol=2e-2)
    assert torch.isfinite(model_bf16(x_bf16)).all()

    model_fp32 = _core(features=2, mask=[True, False])
    x_fp32 = torch.tensor([[2.0, -1.0], [4.0, 3.0]])
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model_fp32(x_fp32)
    except RuntimeError as exc:  # pragma: no cover - platform capability gate.
        pytest.skip(f"CPU BF16 autocast unavailable: {exc}")
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_deterministic_one_batch_overfit_threshold():
    torch.manual_seed(2027)
    features = torch.rand(24, 2) + torch.tensor([0.2, 1.0])
    target = (
        0.35 * features[:, :1]
        - 0.15 * features[:, 1:]
        + 0.8 * features[:, :1] / features[:, 1:]
    )
    core = _core(features=2, layers=1, mask=[False, True], eps=1e-4)
    head = nn.Linear(2, 1)
    model = nn.Sequential(core, head)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    with torch.no_grad():
        initial_loss = F.mse_loss(model(features), target).item()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(350):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(features), target)
            loss.backward()
            optimizer.step()
        final_loss = F.mse_loss(model(features), target).item()
    finally:
        torch.set_num_threads(previous_threads)
    assert final_loss < 1e-3
    assert final_loss < initial_loss * 1e-3


def test_schema_order_hash_and_named_mask_contract():
    names = ("context", "shows", "duration")
    digest = canonical_schema_sha256(names)
    assert validate_schema(names, expected_schema_sha256=digest) == digest
    with pytest.raises(ValueError, match="order mismatch"):
        validate_schema(names, expected_feature_names=("shows", "context", "duration"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_schema(names, expected_schema_sha256="0" * 64)
    mask = canonical_denominator_mask(names, ("shows", "duration"))
    assert mask.tolist() == [False, True, True]
    with pytest.raises(ValueError, match="unknown denominator"):
        canonical_denominator_mask(names, ("unknown",))


def test_materialized_validation_reports_coverage_floor_and_row_ids():
    names = ("numerator", "denom_a", "denom_b")
    values = torch.tensor([[1.0, 0.005, 2.0], [2.0, 0.01, 3.0], [3.0, 7.0, 4.0]])
    validity = torch.tensor(
        [[True, True, True], [True, True, True], [True, False, True]]
    )
    report = validate_materialized_operands(
        values,
        row_ids=("r0", "r1", "r2"),
        feature_names=names,
        denominator_mask=[False, True, True],
        denominator_validity=validity,
        div_eps=0.01,
        expected_feature_names=names,
        expected_schema_sha256=canonical_schema_sha256(names),
    )
    assert report.rows == 3
    assert report.denominator_features == 2
    assert report.active_denominator_cells == 5
    assert report.overall_coverage == pytest.approx(5 / 6)
    assert report.floored_active_cells == 1
    assert report.overall_floor_fraction == pytest.approx(1 / 5)
    denom_a = report.field_stats[0]
    assert denom_a.name == "denom_a"
    assert denom_a.valid_rows == 2
    assert denom_a.missing_rows == 1
    assert denom_a.floored_rows == 1
    assert denom_a.missing_row_ids == ("str:r2",)
    assert denom_a.floored_row_ids == ("str:r0",)
    assert len(report.schema_sha256) == len(report.row_ids_sha256) == 64


def test_materialized_validation_errors_include_stable_row_ids():
    names = ("count", "shows")
    nonpositive = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
    with pytest.raises(OperandValidationError) as nonpositive_error:
        validate_materialized_operands(
            nonpositive,
            row_ids=("row-a", "row-b"),
            feature_names=names,
            denominator_mask=[False, True],
        )
    assert nonpositive_error.value.kind == "nonpositive_active_denominator"
    assert nonpositive_error.value.row_ids == ("str:row-b",)
    assert nonpositive_error.value.feature_names == ("shows",)

    nonfinite_fallback = torch.tensor([[1.0, float("nan")]])
    validity = torch.tensor([[True, False]])
    with pytest.raises(OperandValidationError) as nonfinite_error:
        validate_materialized_operands(
            nonfinite_fallback,
            row_ids=("missing-row",),
            feature_names=names,
            denominator_mask=[False, True],
            denominator_validity=validity,
        )
    assert nonfinite_error.value.kind == "nonfinite_fallback"
    assert nonfinite_error.value.row_ids == ("str:missing-row",)
