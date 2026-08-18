from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from torch import nn

from dcdn.capacity_artifact import (
    CapacityArtifactError,
    capacity_resolution_projection,
    load_verified_capacity_artifact,
    validate_capacity_artifact,
    verify_repository_bindings,
)
from dcdn.encoder import DatasetEncoderSpec, SharedFeatureEncoder, TrainableEncodedModel
from dcdn.models import DCDNCoreConfig, HeadConfig, build_controlled_model
from dcdn.search_space import SearchSpaceLock, validate_capacity_resolution
from dcdn.training import (
    count_trainable_non_embedding_parameters,
    deterministic_forward_flops_per_example,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "configs/capacity/vk_lsvd_c_plus_p_v1.json"


def _payload() -> dict[str, object]:
    loaded = load_verified_capacity_artifact(ARTIFACT)
    return json.loads(json.dumps(loaded))


def _dataset_spec(payload: dict[str, object]) -> DatasetEncoderSpec:
    geometry = payload["geometry"]
    bindings = payload["frozen_bindings"]
    assert isinstance(geometry, dict)
    assert isinstance(bindings, dict)
    context_fields = ("age", "gender", "geo", "place", "platform", "agent")
    # The wrapper audit concerns parameter ownership and width. DatasetEncoderSpec
    # also verifies its schema hashes, so use the frozen real names via the lock.
    encoder_lock = yaml.safe_load((ROOT / "configs/encoder_v1.yaml").read_text())
    dataset = encoder_lock["datasets"]["vk_lsvd_up0.01_ir0.01"]
    regime = dataset["regimes"]["C+P"]
    operand_fields = tuple(regime["ordered_operand_fields"])
    mask = tuple(regime["denominator_mask"])
    return DatasetEncoderSpec(
        dataset_id="vk_lsvd_up0.01_ir0.01",
        feature_regime="C+P",
        encoder_contract_sha256=bindings["encoder_v1_sha256"],
        data_manifest_sha256=dataset["source_manifest_sha256"],
        views_manifest_sha256=dataset["views_manifest_sha256"],
        feature_contract_sha256=dataset["feature_contract_sha256"],
        physical_schema_sha256=regime["physical_schema_sha256"],
        full_feature_schema_sha256=regime["expanded_schema_sha256"],
        operand_schema_sha256=bindings["vk_operand_schema_sha256"],
        denominator_mask_sha256=bindings["vk_denominator_mask_sha256"],
        full_materialized_scan_sha256=dataset["full_materialized_scan_sha256"],
        cache_content_sha256=regime["cache_content_sha256"],
        ordered_context_fields=context_fields,
        categorical_cardinalities=tuple(dataset["categorical_cardinalities"]),
        ordered_operand_fields=operand_fields,
        denominator_mask=mask,
        allowed_model_keys=(),
    )


def test_frozen_vk_capacity_artifact_and_repository_bindings_pass() -> None:
    payload = _payload()
    validate_capacity_artifact(payload)
    verify_repository_bindings(payload, ROOT)
    assert payload["decision"]["selected_low_rank"] == 12
    assert payload["naming"]["reporting_label"] == "DCNv2-capacity-matched"


def test_joint_wrapper_adds_no_non_embedding_encoder_capacity() -> None:
    payload = _payload()
    dataset_spec = _dataset_spec(payload)
    controlled = build_controlled_model(
        encoder_spec=dataset_spec.model_spec,
        interaction_config=DCDNCoreConfig(2, 0.01, 0.1, 0.1, False),
        head_config=HeadConfig(hidden_dims=(), dropout=0.0, output_dim=1),
        init_seed=42,
    )
    encoder = SharedFeatureEncoder(
        dataset_spec,
        run_seed=42,
        include_afn_secondary=False,
        runtime_value_checks=False,
    )
    joint = TrainableEncodedModel(encoder=encoder, controlled_model=controlled)
    owning_types = sorted(
        {
            type(module).__name__
            for module in encoder.modules()
            if tuple(module.parameters(recurse=False))
        }
    )
    assert owning_types == ["Embedding"]
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 4128
    assert count_trainable_non_embedding_parameters(joint) == 23673
    assert count_trainable_non_embedding_parameters(controlled) == 23673
    assert deterministic_forward_flops_per_example(controlled) == 47274
    assert all(
        isinstance(module, nn.Embedding)
        for module in encoder.modules()
        if tuple(module.parameters(recurse=False))
    )


def test_projection_is_accepted_by_existing_search_space_validator() -> None:
    payload = _payload()
    projection = capacity_resolution_projection(payload)
    search_lock = SearchSpaceLock.load(ROOT / "configs/search_space_v1.yaml")
    validate_capacity_resolution(search_lock.payload, projection)
    assert projection["target"] == {"parameters": 23673, "flops": 47274}
    assert projection["ordinary"] == {"parameters": 57725, "flops": 115320}
    assert projection["decision"] == {
        "mode": "fallback_bracket",
        "selected_candidate": "capacity_rank_lower",
    }


def test_validator_rejects_count_or_decision_tampering() -> None:
    payload = _payload()
    bad_count = copy.deepcopy(payload)
    bad_count["target"]["parameters"] += 1
    with pytest.raises(CapacityArtifactError, match="target.parameters"):
        validate_capacity_artifact(bad_count)

    bad_decision = copy.deepcopy(payload)
    bad_decision["decision"]["selected_candidate"] = "capacity_rank_upper"
    with pytest.raises(CapacityArtifactError, match="decision"):
        validate_capacity_artifact(bad_decision)


def test_exact_sidecar_format_is_enforced(tmp_path: Path) -> None:
    copied = tmp_path / ARTIFACT.name
    copied.write_bytes(ARTIFACT.read_bytes())
    digest = __import__("hashlib").sha256(copied.read_bytes()).hexdigest()
    copied.with_name(f"{copied.name}.sha256").write_text(
        f"{digest} {copied.name}\n", encoding="ascii"
    )
    with pytest.raises(CapacityArtifactError, match="sidecar"):
        load_verified_capacity_artifact(copied)
