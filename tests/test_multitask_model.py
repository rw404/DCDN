from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest
import torch

from dcdn.encoder import DatasetEncoderSpec
from dcdn.models.contracts import canonical_sha256
from dcdn.models.dcdn import DCDNCoreConfig
from dcdn.models.field import MLPConfig
from dcdn.multitask import MultiTargetSpec, TargetSpec
from dcdn.multitask_model import build_multitarget_encoded_model


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _target_spec() -> MultiTargetSpec:
    return MultiTargetSpec(
        dataset_id="fixture",
        targets=(
            TargetSpec("watch_time", "regression"),
            TargetSpec("like", "binary"),
            TargetSpec("share", "binary"),
        ),
        primary_confirmatory_target="watch_time",
    )


def _encoder_spec(*, model_keys: tuple[str, ...]) -> DatasetEncoderSpec:
    operands = ("numerator", "denominator", "shared_cosine")
    mask = (False, True, False)
    return DatasetEncoderSpec(
        dataset_id="fixture",
        feature_regime="C+P",
        encoder_contract_sha256=_digest("encoder"),
        data_manifest_sha256=_digest("data"),
        views_manifest_sha256=_digest("views"),
        feature_contract_sha256=_digest("features"),
        physical_schema_sha256=_digest("physical"),
        full_feature_schema_sha256=_digest("expanded"),
        operand_schema_sha256=canonical_sha256(operands),
        denominator_mask_sha256=canonical_sha256(mask),
        full_materialized_scan_sha256=_digest("scan"),
        cache_content_sha256=_digest("cache"),
        ordered_context_fields=("country",),
        categorical_cardinalities=(8,),
        ordered_operand_fields=operands,
        denominator_mask=mask,
        allowed_model_keys=model_keys,
    )


@dataclass(frozen=True)
class _Raw:
    context_tokens: torch.Tensor
    operand_features: torch.Tensor
    denominator_validity: torch.Tensor
    targets: torch.Tensor
    target_masks: torch.Tensor


def _raw() -> _Raw:
    return _Raw(
        context_tokens=torch.tensor([[2], [3], [4], [5]], dtype=torch.int64),
        operand_features=torch.tensor(
            [
                [2.0, 1.0, 0.1],
                [3.0, 2.0, -0.2],
                [4.0, 0.5, 0.3],
                [5.0, 4.0, -0.4],
            ],
            dtype=torch.float32,
        ),
        denominator_validity=torch.tensor([[True, True, True]] * 4, dtype=torch.bool),
        targets=torch.zeros((4, 3), dtype=torch.float32),
        target_masks=torch.ones((4, 3), dtype=torch.bool),
    )


@pytest.mark.parametrize(
    "interaction",
    [
        MLPConfig(hidden_dims=(8,), dropout=0.0),
        DCDNCoreConfig(
            num_layers=1,
            div_eps=1e-2,
            product_init_scale=0.1,
            div_init_scale=0.1,
            runtime_value_checks=True,
        ),
    ],
)
def test_multitarget_model_emits_one_logit_per_frozen_head(interaction) -> None:
    model = build_multitarget_encoded_model(
        encoder_spec=_encoder_spec(model_keys=("dnn", "dcdn_core")),
        interaction_config=interaction,
        target_spec=_target_spec(),
        hidden_dims=(16,),
        dropout=0.0,
        run_seed=42,
        runtime_value_checks=True,
    )
    output = model(_raw())
    assert output.logits.shape == (4, 3)
    assert model.target_spec.names == ("watch_time", "like", "share")
    assert model.controlled_model.head.contract_state()["heads"] == (
        "one_identical_linear_head_per_target"
    )
    assert "targets" not in str(model.encoder.forward.__annotations__)


def test_multitarget_initialization_is_reproducible_and_model_independent_head() -> (
    None
):
    spec = _encoder_spec(model_keys=("dnn", "dcdn_core"))
    target_spec = _target_spec()
    first = build_multitarget_encoded_model(
        encoder_spec=spec,
        interaction_config=MLPConfig(hidden_dims=(8,), dropout=0.0),
        target_spec=target_spec,
        hidden_dims=(16,),
        dropout=0.0,
        run_seed=42,
        runtime_value_checks=True,
    )
    repeat = build_multitarget_encoded_model(
        encoder_spec=spec,
        interaction_config=MLPConfig(hidden_dims=(8,), dropout=0.0),
        target_spec=target_spec,
        hidden_dims=(16,),
        dropout=0.0,
        run_seed=42,
        runtime_value_checks=True,
    )
    dcdn = build_multitarget_encoded_model(
        encoder_spec=spec,
        interaction_config=DCDNCoreConfig(
            num_layers=1,
            div_eps=1e-2,
            product_init_scale=0.1,
            div_init_scale=0.1,
            runtime_value_checks=True,
        ),
        target_spec=target_spec,
        hidden_dims=(16,),
        dropout=0.0,
        run_seed=42,
        runtime_value_checks=True,
    )
    for left, right in zip(
        first.controlled_model.head.parameters(),
        repeat.controlled_model.head.parameters(),
        strict=True,
    ):
        assert torch.equal(left, right)
    for left, right in zip(
        first.controlled_model.head.parameters(),
        dcdn.controlled_model.head.parameters(),
        strict=True,
    ):
        assert torch.equal(left, right)
    assert first.contract_sha256 == repeat.contract_sha256
    assert first.contract_sha256 != dcdn.contract_sha256


def test_multitarget_model_rejects_wrong_head_count() -> None:
    with pytest.raises(ValueError, match="output_dim"):
        # The public builder derives the correct count; this checks the frozen
        # HeadConfig itself remains exact rather than allowing a float/bool.
        from dcdn.models.common import HeadConfig

        HeadConfig(hidden_dims=(), dropout=0.0, output_dim=True)
