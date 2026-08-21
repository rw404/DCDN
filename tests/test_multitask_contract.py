from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dcdn.data.vk_lsvd_multitask_runtime import _load_views_manifest, regime_spec
from dcdn.multitask_capacity import (
    common_counts,
    dcdn_counts,
    ordinary_dcnv2_counts,
    resolve_capacity_control,
)
from dcdn.multitask_contract import MultiTargetContractError, MultiTargetEncoderLock
from dcdn.protocol import read_verified_artifact

ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "configs/encoder_multitask_v2.yaml"
CAPACITY = ROOT / "configs/capacity/vk_lsvd_multitask_c_plus_p_v2.json"
VIEWS = ROOT / "data/manifests/vk_lsvd_multitask_views_v2.yaml"


def test_multitarget_encoder_lock_resolves_both_verified_regimes() -> None:
    lock = MultiTargetEncoderLock.load(ENCODER)
    views, _ = _load_views_manifest(VIEWS)
    bindings = lock.payload["bindings"]
    for name in ("C+P", "C+P+E"):
        frozen = lock.payload["regimes"][name]
        regime = regime_spec(views, name)
        identity = SimpleNamespace(
            dataset_id="vk_lsvd_up0.01_ir0.01",
            feature_regime=name,
            data_manifest_sha256=bindings["data_manifest_sha256"],
            views_manifest_sha256=bindings["views_manifest_sha256"],
            feature_contract_sha256=bindings["feature_contract_sha256"],
            full_materialized_scan_sha256=bindings["full_materialized_scan_sha256"],
            targets_multitask_v2_sha256=bindings["targets_multitask_v2_sha256"],
            feature_policy_sha256=bindings["feature_policy_sha256"],
            matrix_multitask_sha256=bindings["matrix_multitask_sha256"],
            task_constants_sha256=bindings["task_constants_sha256"],
            physical_schema_sha256=frozen["physical_schema_sha256"],
            expanded_schema_sha256=frozen["expanded_schema_sha256"],
            operand_schema_sha256=frozen["operand_schema_sha256"],
            denominator_mask_sha256=frozen["denominator_mask_sha256"],
            cache_content_sha256=frozen["cache_content_sha256"],
            ordered_context_fields=(
                "age",
                "gender",
                "geo",
                "place",
                "platform",
                "agent",
            ),
            categorical_cardinalities=(55, 4, 82, 24, 13, 22),
            ordered_operand_fields=regime.ordered_operands,
            denominator_mask=regime.denominator_mask,
        )
        resolved = lock.resolve(identity)
        assert resolved.model_spec.operand_dim == frozen["operand_width"]
        assert resolved.model_spec.flat_dim == frozen["flat_width"]
        assert resolved.allowed_model_keys == tuple(frozen["allowed_models"])


def test_multitarget_encoder_lock_rejects_type_confusion() -> None:
    lock = MultiTargetEncoderLock.load(ENCODER)
    payload = copy.deepcopy(lock.payload)
    payload["schema_version"] = 2.0
    with pytest.raises(MultiTargetContractError, match="exact int 2"):
        MultiTargetEncoderLock._validate(payload)


def test_capacity_artifact_recomputes_without_metrics() -> None:
    verified = read_verified_artifact(CAPACITY)
    payload = json.loads(verified.data)
    common = common_counts(151, 64, (), 7)
    target = dcdn_counts(55, 2, common)
    ordinary = ordinary_dcnv2_counts(151, 2, common)
    decision = resolve_capacity_control(
        target=target,
        operand_width=55,
        common=common,
        ranks=tuple(range(1, 56)),
        layer_options=(2,),
        expert_options=(4,),
    ).payload()
    assert payload["common_counts"] == common
    assert payload["dcdn_target"] == target
    assert payload["ordinary_dcnv2"] == ordinary
    assert payload["decision"] == decision
    assert payload["decision"]["selection_used_validation_quality"] is False
    assert payload["decision"]["selected"]["low_rank"] == 12
    assert payload["test_unlocked"] is False
