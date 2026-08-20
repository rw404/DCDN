from __future__ import annotations

import pytest
import torch

from dcdn.models.cross import DCNv2FullRankConfig
from dcdn.models.dcdn import DCDNCoreConfig
from dcdn.models.field import MLPConfig
from dcdn.multitask_capacity import (
    common_counts,
    dcdn_counts,
    ordinary_dcnv2_counts,
    ratios,
    resolve_capacity_control,
)
from dcdn.synthetic_multiquotient import COLUMNS, SPEC, build_benchmark
from dcdn.synthetic_runtime import (
    FULL_SCHEMA_SHA256,
    OPERAND_SCHEMA_SHA256,
    MultiQuotientModel,
    SyntheticFieldEncoder,
    encoder_spec,
    ordered_schema_sha256,
    slice_tensors,
)

EMBEDDING_DIM = 8
REPRESENTATION_DIM = 64
HEAD_HIDDEN_DIMS = (128, 64)

DCDN_CONFIG = DCDNCoreConfig(
    num_layers=1,
    div_eps=1e-2,
    product_init_scale=0.1,
    div_init_scale=0.1,
    runtime_value_checks=True,
)
DCNV2_CONFIG = DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1)
MLP_CONFIG = MLPConfig(hidden_dims=(512, 256, 128), dropout=0.0)


@pytest.fixture(scope="module")
def spec():
    return encoder_spec(EMBEDDING_DIM, REPRESENTATION_DIM)


def build(spec, config, seed: int = 42, afn: bool = False) -> MultiQuotientModel:
    return MultiQuotientModel(
        spec,
        config,
        num_targets=len(SPEC),
        head_hidden_dims=HEAD_HIDDEN_DIMS,
        init_seed=seed,
        with_afn_view=afn,
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def test_schema_hash_is_order_sensitive() -> None:
    assert ordered_schema_sha256(COLUMNS) != ordered_schema_sha256(
        tuple(reversed(COLUMNS))
    )


def test_full_schema_equals_operand_schema_without_a_bypass() -> None:
    assert FULL_SCHEMA_SHA256 == OPERAND_SCHEMA_SHA256


def test_encoder_lift_matches_its_declared_formula() -> None:
    encoder = SyntheticFieldEncoder(num_fields=3, embedding_dim=4, with_afn_view=False)
    operands = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    fields, auxiliary = encoder(operands)
    assert auxiliary is None
    expected = operands.unsqueeze(-1) * encoder.weight + encoder.bias
    assert torch.allclose(fields, expected)


def test_encoder_emits_an_independent_afn_view_when_requested() -> None:
    encoder = SyntheticFieldEncoder(num_fields=3, embedding_dim=4, with_afn_view=True)
    fields, auxiliary = encoder(torch.rand(5, 3) + 1.0)
    assert auxiliary is not None
    assert auxiliary.shape == fields.shape
    assert not torch.allclose(auxiliary, fields)


def test_encoder_rejects_a_wrong_field_count() -> None:
    encoder = SyntheticFieldEncoder(num_fields=3, embedding_dim=4, with_afn_view=False)
    with pytest.raises(ValueError, match="shape"):
        encoder(torch.rand(2, 4))


def test_encoder_spec_binds_the_denominator_mask(spec) -> None:
    assert len(spec.denominator_mask) == spec.operand_dim == len(COLUMNS)
    active = {c for c, m in zip(COLUMNS, spec.denominator_mask, strict=True) if m}
    assert active == {"d1", "d2", "d3"}
    assert spec.bypass_dim == 0


# ---------------------------------------------------------------------------
# Model wiring
# ---------------------------------------------------------------------------


def test_every_arm_emits_one_column_per_target(spec) -> None:
    probe = torch.rand(7, len(COLUMNS)) + 1.0
    for config in (MLP_CONFIG, DCNV2_CONFIG, DCDN_CONFIG):
        model = build(spec, config)
        assert model(probe).shape == (7, len(SPEC))


def test_batch_of_one_is_preserved(spec) -> None:
    model = build(spec, DCDN_CONFIG)
    assert model(torch.rand(1, len(COLUMNS)) + 1.0).shape == (1, len(SPEC))


def test_encoder_parameters_are_identical_across_arms(spec) -> None:
    """Fairness: the shared encoder must not vary with the interaction block."""

    left = build(spec, DCNV2_CONFIG)
    right = build(spec, DCDN_CONFIG)
    for a, b in zip(left.encoder.parameters(), right.encoder.parameters(), strict=True):
        assert torch.equal(a, b)


def test_head_parameters_are_identical_across_arms(spec) -> None:
    """The audited factory gives every arm the same head for one base seed."""

    left = build(spec, DCNV2_CONFIG)
    right = build(spec, DCDN_CONFIG)
    for a, b in zip(
        left.model.head.parameters(), right.model.head.parameters(), strict=True
    ):
        assert torch.equal(a, b)


def test_different_seeds_give_different_parameters(spec) -> None:
    left = build(spec, DCDN_CONFIG, seed=42)
    right = build(spec, DCDN_CONFIG, seed=43)
    assert not torch.equal(
        next(iter(left.model.head.parameters())),
        next(iter(right.model.head.parameters())),
    )


def test_construction_is_deterministic_for_one_seed(spec) -> None:
    left = build(spec, DCDN_CONFIG, seed=42)
    right = build(spec, DCDN_CONFIG, seed=42)
    for a, b in zip(left.parameters(), right.parameters(), strict=True):
        assert torch.equal(a, b)


def test_fused_head_keeps_targets_gradient_independent(spec) -> None:
    """``Linear(d, T)`` is T separate heads: no gradient crosses between them."""

    model = build(spec, DCDN_CONFIG)
    logits = model(torch.rand(16, len(COLUMNS)) + 1.0)
    logits[:, 0].sum().backward()
    head_grad = model.model.head.output.weight.grad
    assert head_grad is not None
    assert float(head_grad[0].abs().sum()) > 0.0
    # Rows 1 and 2 belong to the other two targets and must be untouched.
    assert float(head_grad[1].abs().sum()) == 0.0
    assert float(head_grad[2].abs().sum()) == 0.0


def test_dcdn_receives_raw_positive_operands(spec) -> None:
    """Runtime value checks pass only if the core sees the raw positive input."""

    model = build(spec, DCDN_CONFIG)  # runtime_value_checks=True
    benchmark = build_benchmark()
    features, _, _, validity = slice_tensors(benchmark["id"], "cpu")
    out = model(features[:64], validity[:64])
    assert torch.isfinite(out).all()


def test_invalid_denominator_is_gated_not_amplified(spec) -> None:
    model = build(spec, DCDN_CONFIG)
    core = model.model.interaction.core
    x0 = torch.rand(4, len(COLUMNS)) + 1.0
    validity = torch.ones_like(x0, dtype=torch.bool)
    validity[:, COLUMNS.index("d1")] = False
    carrier = core.reciprocal_carrier(x0, validity)
    assert float(carrier[:, COLUMNS.index("d1")].abs().max()) == 0.0
    assert float(carrier[:, COLUMNS.index("d2")].min()) > 0.0


def test_reciprocal_carrier_is_zero_outside_the_mask(spec) -> None:
    model = build(spec, DCDN_CONFIG)
    core = model.model.interaction.core
    x0 = torch.rand(4, len(COLUMNS)) + 1.0
    carrier = core.reciprocal_carrier(x0, torch.ones_like(x0, dtype=torch.bool))
    for index, column in enumerate(COLUMNS):
        if column in ("d1", "d2", "d3"):
            assert float(carrier[:, index].min()) > 0.0
        else:
            assert float(carrier[:, index].abs().max()) == 0.0


def test_slice_tensors_declare_full_validity() -> None:
    benchmark = build_benchmark()
    _, _, masks, validity = slice_tensors(benchmark["id"], "cpu")
    assert bool(masks.all())
    assert bool(validity.all())


# ---------------------------------------------------------------------------
# Capacity accounting
# ---------------------------------------------------------------------------


def test_common_counts_scale_with_target_count() -> None:
    one = common_counts(8, 64, HEAD_HIDDEN_DIMS, 1)
    three = common_counts(8, 64, HEAD_HIDDEN_DIMS, 3)
    assert three["linear_head_parameters"] == 3 * one["linear_head_parameters"]
    assert three["tower_parameters"] == one["tower_parameters"]


def test_analytic_parameter_counts_match_the_constructed_model(spec) -> None:
    common = common_counts(
        len(COLUMNS), REPRESENTATION_DIM, HEAD_HIDDEN_DIMS, len(SPEC)
    )
    for config, counts in (
        (DCDN_CONFIG, dcdn_counts(len(COLUMNS), 1, common)),
        (DCNV2_CONFIG, ordinary_dcnv2_counts(len(COLUMNS), 2, common)),
    ):
        model = build(spec, config)
        assert model.parameter_counts()["matched"] == counts["parameters"]


def test_dcdn_l1_and_dcnv2_l2_have_equal_interaction_parameters() -> None:
    common = common_counts(
        len(COLUMNS), REPRESENTATION_DIM, HEAD_HIDDEN_DIMS, len(SPEC)
    )
    dcdn = dcdn_counts(len(COLUMNS), 1, common)
    dcnv2 = ordinary_dcnv2_counts(len(COLUMNS), 2, common)
    # 1 * 2 * 8 * 9 == 2 * (8*8 + 8) == 144.
    assert dcdn["interaction_parameters"] == dcnv2["interaction_parameters"] == 144


def test_capacity_resolution_lands_inside_tolerance() -> None:
    common = common_counts(
        len(COLUMNS), REPRESENTATION_DIM, HEAD_HIDDEN_DIMS, len(SPEC)
    )
    target = dcdn_counts(len(COLUMNS), 1, common)
    resolution = resolve_capacity_control(
        target=target,
        operand_width=len(COLUMNS),
        common=common,
        ranks=tuple(range(1, 17)),
        layer_options=(1, 2),
        expert_options=(1, 2, 4),
    )
    assert resolution.matched
    assert resolution.selected_ratios["within_tolerance"]
    assert resolution.payload()["selection_used_validation_quality"] is False


def test_capacity_resolution_is_deterministic() -> None:
    common = common_counts(
        len(COLUMNS), REPRESENTATION_DIM, HEAD_HIDDEN_DIMS, len(SPEC)
    )
    target = dcdn_counts(len(COLUMNS), 1, common)
    kwargs = dict(
        target=target,
        operand_width=len(COLUMNS),
        common=common,
        ranks=tuple(range(1, 17)),
        layer_options=(1, 2),
        expert_options=(1, 2, 4),
    )
    first = resolve_capacity_control(**kwargs).selected
    second = resolve_capacity_control(**kwargs).selected
    assert first == second


def test_unreachable_target_returns_two_bracketing_controls() -> None:
    common = common_counts(
        len(COLUMNS), REPRESENTATION_DIM, HEAD_HIDDEN_DIMS, len(SPEC)
    )
    target = dcdn_counts(len(COLUMNS), 1, common)
    # Only rank 16 is offered, so nothing can land within tolerance of a target
    # that has been shrunk far below every candidate.
    impossible = {**target, "parameters": 1, "flops": 1}
    resolution = resolve_capacity_control(
        target=impossible,
        operand_width=len(COLUMNS),
        common=common,
        ranks=(16,),
        layer_options=(2,),
        expert_options=(4,),
    )
    assert not resolution.matched
    assert resolution.upper_bracket is not None
    assert "two_bracketing_controls" in resolution.selection_rule


def test_ratio_helper_flags_out_of_tolerance() -> None:
    target = {"parameters": 1000, "flops": 1000}
    assert ratios({"parameters": 1000, "flops": 1000}, target)["within_tolerance"]
    assert not ratios({"parameters": 1200, "flops": 1000}, target)["within_tolerance"]
    assert not ratios({"parameters": 1000, "flops": 1200}, target)["within_tolerance"]
