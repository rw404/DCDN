from __future__ import annotations

import inspect
from dataclasses import MISSING, fields, replace

import pytest
import torch
from torch import nn

from dcdn.models import (
    AFNPlusConfig,
    AFNPlusInteraction,
    AutoIntPlusConfig,
    DCDNCoreConfig,
    DCDNCoreInteraction,
    DCNConfig,
    DCNv2CapacityCoreConfig,
    DCNv2CapacityCoreInteraction,
    DCNv2FullRankConfig,
    DCNv2FullRankInteraction,
    DeepFMConfig,
    DeepFMInteraction,
    EncodedInteractionBatch,
    EncoderSpec,
    HeadConfig,
    MLPConfig,
    OriginalDCNInteraction,
    WukongLayer,
    WukongSmallConfig,
    build_controlled_model,
)

FULL_SCHEMA_HASH = "1" * 64
OPERAND_SCHEMA_HASH = "2" * 64


def _spec(*, representation_dim: int = 64) -> EncoderSpec:
    return EncoderSpec(
        bypass_dim=2,
        operand_dim=3,
        num_fields=4,
        embedding_dim=4,
        representation_dim=representation_dim,
        denominator_mask=(False, True, False),
        full_feature_schema_sha256=FULL_SCHEMA_HASH,
        operand_schema_sha256=OPERAND_SCHEMA_HASH,
    )


def _batch(
    batch_size: int = 2,
    *,
    dtype: torch.dtype = torch.float32,
    include_afn_aux: bool = True,
) -> EncodedInteractionBatch:
    generator = torch.Generator().manual_seed(1234 + batch_size)
    bypass = torch.randn(batch_size, 2, generator=generator).to(dtype)
    operands = (torch.rand(batch_size, 3, generator=generator) + 0.2).to(dtype)
    fields_view = torch.randn(batch_size, 4, 4, generator=generator).to(dtype)
    auxiliary = (
        torch.randn(batch_size, 4, 4, generator=generator).to(dtype)
        if include_afn_aux
        else None
    )
    return EncodedInteractionBatch(
        bypass_features=bypass,
        operand_features=operands,
        field_embeddings=fields_view,
        denominator_validity=torch.ones(batch_size, 3, dtype=torch.bool),
        afn_dnn_embeddings=auxiliary,
    )


def _configs():  # type: ignore[no-untyped-def]
    return [
        MLPConfig(hidden_dims=(8,), dropout=0.0),
        DeepFMConfig(deep_hidden_dims=(8,), dropout=0.0),
        DCNConfig(num_layers=2, product_init_scale=0.1),
        AutoIntPlusConfig(
            attention_layers=2,
            attention_dim=4,
            num_heads=2,
            dnn_hidden_dims=(8,),
            dropout=0.0,
            scale_attention=True,
            layer_norm=True,
        ),
        DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1),
        DCNv2CapacityCoreConfig(
            num_layers=2,
            low_rank=2,
            num_experts=2,
            product_init_scale=0.1,
        ),
        AFNPlusConfig(
            logarithmic_neurons=2,
            afn_hidden_dims=(8,),
            dnn_hidden_dims=(8,),
            log_floor=1e-5,
            exp_clip=None,
            dropout=0.0,
            batch_norm=False,
        ),
        WukongSmallConfig(
            num_layers=1,
            lcb_features=2,
            fmb_features=2,
            fmb_rank=2,
            fmb_hidden_dims=(8,),
            dropout=0.0,
            layer_norm=True,
        ),
        DCDNCoreConfig(
            num_layers=2,
            div_eps=1e-2,
            product_init_scale=0.1,
            div_init_scale=0.1,
            runtime_value_checks=True,
        ),
    ]


def _head() -> HeadConfig:
    return HeadConfig(hidden_dims=(), dropout=0.0, output_dim=1)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("config", _configs(), ids=lambda value: type(value).__name__)
def test_every_model_has_finite_common_forward_backward(config, batch_size):  # type: ignore[no-untyped-def]
    model = build_controlled_model(
        encoder_spec=_spec(),
        interaction_config=config,
        head_config=_head(),
        init_seed=42,
    )
    output = model(_batch(batch_size))

    assert output.representation.values.shape == (batch_size, 64)
    assert output.logits.shape == (batch_size, 1)
    assert torch.isfinite(output.representation.values).all()
    assert torch.isfinite(output.logits).all()

    output.logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_original_dcn_matches_manual_fixed_anchor_recurrence() -> None:
    block = OriginalDCNInteraction(
        _spec(), DCNConfig(num_layers=2, product_init_scale=0.1)
    )
    with torch.no_grad():
        block.weights[0].copy_(torch.tensor([0.2, -0.1, 0.3, 0.4, -0.2]))
        block.weights[1].copy_(torch.tensor([-0.3, 0.5, 0.1, -0.2, 0.4]))
        block.biases[0].copy_(torch.tensor([0.1, 0.2, 0.0, -0.1, 0.3]))
        block.biases[1].copy_(torch.tensor([-0.2, 0.0, 0.1, 0.2, -0.1]))

    batch = _batch(2)
    anchor = batch.flat_features
    expected = anchor
    for weight, bias in zip(block.weights, block.biases, strict=True):
        expected = expected + anchor * (expected * weight).sum(-1, keepdim=True) + bias
    torch.testing.assert_close(block(batch), expected)


def test_dcnv2_full_rank_matches_manual_fixed_anchor_recurrence() -> None:
    block = DCNv2FullRankInteraction(
        _spec(), DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1)
    )
    with torch.no_grad():
        for index, layer in enumerate(block.layers, start=1):
            layer.weight.copy_(torch.eye(5) * (0.1 * index))
            layer.bias.fill_(0.05 * index)

    batch = _batch(2)
    anchor = batch.flat_features
    expected = anchor
    for layer in block.layers:
        expected = expected + anchor * (expected @ layer.weight.T + layer.bias)
    torch.testing.assert_close(block(batch), expected)


def test_capacity_core_matches_manual_moe_and_preserves_bypass_for_b1() -> None:
    block = DCNv2CapacityCoreInteraction(
        _spec(),
        DCNv2CapacityCoreConfig(1, low_rank=2, num_experts=2, product_init_scale=0.1),
    )
    batch = _batch(1)
    layer = block.layers[0]
    anchor = batch.operand_features
    crosses = []
    gate_logits = []
    for expert, gate in zip(layer.experts, layer.gates, strict=True):
        low = torch.tanh(anchor @ expert.v)
        low = torch.tanh(low @ expert.c)
        projected = low @ expert.u.T
        crosses.append(anchor * (projected + layer.bias))
        gate_logits.append(gate(anchor))
    weights = torch.softmax(torch.cat(gate_logits, dim=1), dim=1)
    expected_operands = anchor + torch.sum(
        torch.stack(crosses, dim=1) * weights.unsqueeze(-1), dim=1
    )
    actual = block(batch)

    assert actual.shape == (1, 5)
    torch.testing.assert_close(actual[:, :2], batch.bypass_features)
    torch.testing.assert_close(actual[:, 2:], expected_operands)
    assert block.display_label == "DCNv2-capacity-matched-core"


def test_dcdn_adapter_uses_operand_hash_fixed_anchor_mask_and_context_bypass() -> None:
    block = DCDNCoreInteraction(
        _spec(),
        DCDNCoreConfig(2, 0.1, 0.1, 0.1, runtime_value_checks=True),
    )
    with torch.no_grad():
        for index, layer in enumerate(block.core.layers, start=1):
            layer.product_block.weight.copy_(torch.eye(3) * (0.1 * index))
            layer.product_block.bias.fill_(0.02 * index)
            layer.division_block.weight.copy_(torch.eye(3) * (0.03 * index))
            layer.division_block.bias.fill_(0.01 * index)

    batch = _batch(2)
    anchor = batch.operand_features
    reciprocal = torch.zeros_like(anchor)
    reciprocal[:, 1] = anchor[:, 1].clamp_min(0.1).reciprocal()
    expected = anchor
    for layer in block.core.layers:
        expected = (
            expected
            + anchor * layer.product_block(expected)
            + reciprocal * layer.division_block(expected)
        )
    actual = block(batch)

    torch.testing.assert_close(actual[:, :2], batch.bypass_features)
    torch.testing.assert_close(actual[:, 2:], expected)
    metadata = block.core.serialization_metadata()
    assert metadata["feature_schema_sha256"] == OPERAND_SCHEMA_HASH
    assert metadata["feature_schema_sha256"] != FULL_SCHEMA_HASH
    assert block.display_label == "DCDN-Core-L2"


def test_deepfm_second_order_term_matches_pairwise_oracle() -> None:
    fields_view = torch.tensor([[[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]]])
    expected = torch.zeros(1, 1)
    for left in range(fields_view.shape[1]):
        for right in range(left + 1, fields_view.shape[1]):
            expected += (fields_view[:, left] * fields_view[:, right]).sum(
                dim=-1, keepdim=True
            )
    torch.testing.assert_close(
        DeepFMInteraction.factorization_term(fields_view), expected
    )


def test_autoint_plus_raw_output_contains_parallel_dnn_branch() -> None:
    config = AutoIntPlusConfig(1, 4, 2, (7,), 0.0, True, True)
    block = build_controlled_model(
        encoder_spec=_spec(),
        interaction_config=config,
        head_config=_head(),
        init_seed=4,
    ).interaction
    batch = _batch(2)
    raw = block(batch)
    expected_dnn = block.deep(batch.field_embeddings.flatten(start_dim=1))
    torch.testing.assert_close(raw[:, block.attention_output_dim :], expected_dnn)

    raw[:, block.attention_output_dim :].sum().backward()
    assert all(parameter.grad is not None for parameter in block.deep.parameters())
    with pytest.raises(ValueError, match="must not be empty"):
        AutoIntPlusConfig(1, 4, 2, (), 0.0, True, True)


def test_afn_logarithmic_oracle_abs_floor_fp32_exp_and_two_branches() -> None:
    config = AFNPlusConfig(2, (4,), (4,), 0.1, None, 0.0, False)
    block = AFNPlusInteraction(_spec(), config)
    with torch.no_grad():
        block.coefficients.weight.copy_(
            torch.tensor([[0.5, -0.5, 0.25, 1.0], [-0.2, 0.3, 0.4, -0.1]])
        )
    source = torch.tensor(
        [
            [
                [-2.0, 0.0, 0.05, 3.0],
                [4.0, -5.0, 0.2, 0.1],
                [-0.5, 2.0, -3.0, 1.0],
                [1.5, -0.2, 0.4, -2.5],
            ]
        ]
    )
    logged = source.float().abs().clamp_min(config.log_floor).log()
    exponents = block.coefficients(logged.transpose(1, 2)).transpose(1, 2)
    expected = exponents.float().exp().flatten(start_dim=1)
    actual = block.logarithmic_features(source)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, block.logarithmic_features(-source))

    batch = _batch(2)
    raw = block(batch)
    assert raw.shape == (2, 2)
    raw[:, 0].sum().backward(retain_graph=True)
    assert block.coefficients.weight.grad is not None
    assert block.dnn_output.weight.grad is not None
    assert torch.count_nonzero(block.dnn_output.weight.grad) == 0
    block.zero_grad(set_to_none=True)
    block(batch)[:, 1].sum().backward()
    assert block.dnn_output.weight.grad is not None
    assert block.coefficients.weight.grad is not None
    assert torch.count_nonzero(block.coefficients.weight.grad) == 0


def test_afn_optional_exp_clamp_is_explicit_and_b1_batch_norm_is_finite() -> None:
    config = AFNPlusConfig(2, (4,), (4,), 1e-5, 8.0, 0.0, True)
    model = build_controlled_model(
        encoder_spec=_spec(),
        interaction_config=config,
        head_config=_head(),
        init_seed=9,
    )
    output = model(_batch(1))
    output.logits.sum().backward()
    assert torch.isfinite(output.logits).all()
    assert model.contract_state()["interaction_extra"]["exp_policy"] == (
        "explicit_symmetric_pre_exp_clamp"
    )


def test_wukong_fmb_lcb_residual_layer_oracles_and_no_scale_defaults() -> None:
    config = WukongSmallConfig(1, 2, 2, 2, (8,), 0.0, True)
    layer = WukongLayer(input_fields=4, embedding_dim=4, config=config)
    values = _batch(1).field_embeddings

    projected = values.transpose(1, 2) @ layer.fmb.projection
    expected_interactions = torch.bmm(values, projected)
    torch.testing.assert_close(
        layer.fmb.interaction_matrix(values), expected_interactions
    )
    expected_lcb = torch.einsum("bne,no->boe", values, layer.lcb.weight)
    torch.testing.assert_close(layer.lcb(values), expected_lcb)
    expected_layer = layer.normalization(
        torch.cat((layer.fmb(values), layer.lcb(values)), dim=1)
        + layer.residual(values)
    )
    torch.testing.assert_close(layer(values), expected_layer)
    assert layer(values).shape == (1, 4, 4)

    assert all(
        field.default is MISSING and field.default_factory is MISSING
        for field in fields(WukongSmallConfig)
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in inspect.signature(WukongSmallConfig).parameters.values()
    )


def test_factory_is_deterministic_preserves_global_rng_and_components_are_seeded() -> (
    None
):
    kwargs = {
        "encoder_spec": _spec(),
        "interaction_config": MLPConfig((8,), 0.0),
        "head_config": _head(),
        "init_seed": 42,
    }
    first = build_controlled_model(**kwargs)
    second = build_controlled_model(**kwargs)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])
    assert first.contract_sha256 == second.contract_sha256

    torch.manual_seed(888)
    expected_next = torch.rand(5)
    torch.manual_seed(888)
    build_controlled_model(**kwargs)
    actual_next = torch.rand(5)
    torch.testing.assert_close(actual_next, expected_next)

    other = build_controlled_model(**(kwargs | {"init_seed": 43}))
    assert any(
        not torch.equal(value, other.state_dict()[name])
        for name, value in first.state_dict().items()
    )
    initialization = first.contract_state()["initialization"]
    assert set(initialization["component_seeds"]) == {
        "interaction",
        "adapter",
        "head",
    }


def test_common_head_weights_are_identical_across_model_variants() -> None:
    models = [
        build_controlled_model(
            encoder_spec=_spec(),
            interaction_config=config,
            head_config=_head(),
            init_seed=42,
        )
        for config in _configs()
    ]
    reference = models[0].head.state_dict()
    for model in models[1:]:
        for name, value in reference.items():
            torch.testing.assert_close(value, model.head.state_dict()[name])
        assert model.head.contract_state() == models[0].head.contract_state()


def test_all_linear_biases_are_explicitly_zero_at_initialization() -> None:
    for config in _configs():
        model = build_controlled_model(
            encoder_spec=_spec(),
            interaction_config=config,
            head_config=_head(),
            init_seed=42,
        )
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))


def test_contract_hashes_full_state_and_pinned_provenance() -> None:
    model = build_controlled_model(
        encoder_spec=_spec(),
        interaction_config=DeepFMConfig((8,), 0.0),
        head_config=_head(),
        init_seed=42,
    )
    state = model.contract_state()
    assert state["model_key"] == "deepfm"
    assert state["display_label"] == "DeepFM"
    assert state["source_provenance"]["commit"] == (
        "0245987fdc1d43c36164ea496a02324885a8557a"
    )
    assert state["source_provenance"]["license"] == "Apache-2.0"
    assert len(model.contract_sha256) == 64

    wukong = build_controlled_model(
        encoder_spec=_spec(),
        interaction_config=WukongSmallConfig(1, 2, 2, 2, (8,), 0.0, True),
        head_config=_head(),
        init_seed=42,
    )
    paper = wukong.contract_state()["source_provenance"]
    assert paper["paper_record"] == (
        "https://proceedings.mlr.press/v235/zhang24ao.html"
    )
    assert paper["paper_pdf_origin"] == (
        "https://raw.githubusercontent.com/mlresearch/v235/main/assets/"
        "zhang24ao/zhang24ao.pdf"
    )
    assert paper["paper_arxiv_identifier"] == "arXiv:2403.02545"
    assert paper["paper_local_pdf_sha256"] == (
        "8f4f539a174d3ff12653694d7e3d16f2c3b0c0651770a97445ac4826dcb5fabd"
    )

    changed = build_controlled_model(
        encoder_spec=_spec(),
        interaction_config=DeepFMConfig((9,), 0.0),
        head_config=_head(),
        init_seed=42,
    )
    assert changed.contract_sha256 != model.contract_sha256


def test_input_and_parameter_dtype_contracts_fail_clearly_and_allow_bf16() -> None:
    model = build_controlled_model(
        encoder_spec=_spec(),
        interaction_config=MLPConfig((8,), 0.0),
        head_config=_head(),
        init_seed=42,
    )
    with pytest.raises(TypeError, match="FP16 is forbidden"):
        model(_batch(2, dtype=torch.float16))
    with pytest.raises(TypeError, match="must match outside autocast"):
        model(_batch(2, dtype=torch.bfloat16))

    bf16_model = build_controlled_model(
        encoder_spec=_spec(),
        interaction_config=MLPConfig((8,), 0.0),
        head_config=_head(),
        init_seed=42,
    ).to(dtype=torch.bfloat16)
    bf16_output = bf16_model(_batch(2, dtype=torch.bfloat16))
    assert bf16_output.logits.dtype == torch.bfloat16
    bf16_output.logits.float().sum().backward()

    with pytest.raises(TypeError, match="FP16 parameters"):
        model.half()(_batch(2))


def test_bf16_inputs_with_fp32_parameters_require_and_work_under_autocast() -> None:
    model = build_controlled_model(
        encoder_spec=_spec(),
        interaction_config=MLPConfig((8,), 0.0),
        head_config=_head(),
        init_seed=42,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(_batch(2, dtype=torch.bfloat16))
    assert output.logits.dtype == torch.bfloat16


def test_invalid_shapes_validity_auxiliary_and_configs_are_rejected() -> None:
    spec = _spec()
    batch = _batch(2)
    bad_shape = replace(batch, operand_features=torch.ones(2, 4))
    with pytest.raises(ValueError, match="operand_features must have shape"):
        bad_shape.validate(spec)
    bad_validity = replace(
        batch, denominator_validity=torch.ones(2, 3, dtype=torch.float32)
    )
    with pytest.raises(TypeError, match="must be boolean"):
        bad_validity.validate(spec)

    afn = build_controlled_model(
        encoder_spec=spec,
        interaction_config=AFNPlusConfig(2, (4,), (4,), 1e-5, None, 0.0, False),
        head_config=_head(),
        init_seed=1,
    )
    with pytest.raises(ValueError, match="requires afn_dnn_embeddings"):
        afn(_batch(2, include_afn_aux=False))

    with pytest.raises(ValueError, match="length must equal"):
        replace(spec, denominator_mask=(True, False))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(spec, operand_schema_sha256="not-a-hash")
    with pytest.raises(ValueError, match="divisible"):
        AutoIntPlusConfig(1, 5, 2, (4,), 0.0, True, True)
    with pytest.raises(ValueError, match="must be finite and positive"):
        AFNPlusConfig(2, (4,), (4,), 0.0, None, 0.0, False)


def test_capacity_and_dcdn_contracts_bind_identical_operand_core_boundary() -> None:
    capacity = build_controlled_model(
        encoder_spec=_spec(),
        interaction_config=DCNv2CapacityCoreConfig(1, 2, 2, 0.1),
        head_config=_head(),
        init_seed=2,
    ).contract_state()
    dcdn = build_controlled_model(
        encoder_spec=_spec(),
        interaction_config=DCDNCoreConfig(1, 1e-2, 0.1, 0.1, True),
        head_config=_head(),
        init_seed=2,
    ).contract_state()
    assert capacity["interaction_extra"]["crossed_view"] == ("operand_features_P_only")
    assert dcdn["interaction_extra"]["crossed_view"] == ("operand_features_P_only")
    assert capacity["interaction_extra"]["bypass_view"] == ("context_C_unchanged")
    assert dcdn["interaction_extra"]["bypass_view"] == "context_C_unchanged"
