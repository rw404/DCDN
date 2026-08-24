"""Strict validator for metric-free public capacity-resolution artifacts.

The validator is intentionally independent of model execution and dataset rows.
It recomputes parameter/FLOP totals from frozen integer geometry and verifies
the adjacent-rank decision mechanically.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CAPACITY_ARTIFACT_CONTRACT = "dcdn-vk-capacity-resolution-v1"
CANONICAL_DISPLAY_LABEL = "DCNv2-capacity-matched"
INTERNAL_MODEL_KEY = "dcnv2_capacity_matched_core"
HISTORICAL_DISPLAY_ALIAS = "DCNv2-capacity-matched-core"
_UTC_RE = re.compile(r"^2026-08-18T\d{2}:\d{2}:\d{2}Z$")
_EXPECTED_BINDINGS = {
    "search_space_v1_sha256": (
        "b263b55da768ec33656ebdb1506d58f2809fd5826aa8b4976aedaecee53bb344"
    ),
    "encoder_v1_sha256": (
        "16fa99db8e92ce4dd55c0bd5e7d7bfd8af8d3d59251cc769b41988e61d414449"
    ),
    "model_sources_v1_sha256": (
        "0844a838e5a68029299a52aaa31b5f74205fe57fd486b51544061b32c571a3f4"
    ),
    "experiment_matrix_v1_sha256": (
        "867118db089bbbffb4abf02ff7b3c01e2c9f43e804a188a27a3886f66db4e000"
    ),
    "vk_features_v1_sha256": (
        "380d105c847a73afb4927740e40401124688f895142792309b4fc8267a2a5f2b"
    ),
    "vk_operand_schema_sha256": (
        "bb2f6c47fa12c175a3820830f90ff577acb293c56b107c7d1c40e4c682f2ff4d"
    ),
    "vk_denominator_mask_sha256": (
        "9e168fae4014f4763008e89f61324858f17d376f56fb5de46f2775911650bcaa"
    ),
}
_EXPECTED_GEOMETRY = {
    "ordered_context_field_count": 6,
    "context_embedding_dim": 16,
    "context_bypass_width": 96,
    "operand_width": 58,
    "flat_width": 154,
    "representation_width": 64,
    "head_output_width": 1,
    "dcdn_num_layers": 2,
    "ordinary_dcnv2_num_layers": 2,
    "capacity_num_layers": 2,
    "capacity_num_experts": 4,
}


class CapacityArtifactError(RuntimeError):
    """A capacity artifact or one of its frozen bindings is invalid."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Any, expected: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CapacityArtifactError(f"{field} fields drifted")
    return value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise CapacityArtifactError(f"{field} must be a positive integer")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CapacityArtifactError(f"{field} must be a lowercase SHA-256")
    return value


def _biased_linear_flops(input_width: int, output_width: int) -> int:
    return 2 * input_width * output_width


def _unbiased_linear_flops(input_width: int, output_width: int) -> int:
    return output_width * (2 * input_width - 1)


def _common_counts(flat_width: int, representation_width: int) -> dict[str, int]:
    adapter_parameters = (flat_width + 1) * representation_width
    head_parameters = representation_width + 1
    return {
        "adapter_parameters": adapter_parameters,
        "head_parameters": head_parameters,
        "adapter_flops": _biased_linear_flops(flat_width, representation_width),
        "head_flops": _biased_linear_flops(representation_width, 1),
    }


def _dcdn_counts(
    operand_width: int,
    num_layers: int,
    common: Mapping[str, int],
) -> dict[str, Any]:
    interaction_parameters_per_layer = 2 * (
        operand_width * operand_width + operand_width
    )
    interaction_parameters = num_layers * interaction_parameters_per_layer
    reciprocal_flops = operand_width
    interaction_flops_per_layer = 4 * operand_width**2 + 4 * operand_width
    interaction_flops = num_layers * interaction_flops_per_layer + reciprocal_flops
    return {
        "architecture": "DCDN-Core-L2",
        "interaction_parameters_per_layer": interaction_parameters_per_layer,
        "interaction_parameters": interaction_parameters,
        "adapter_parameters": common["adapter_parameters"],
        "head_parameters": common["head_parameters"],
        "parameters": (
            interaction_parameters
            + common["adapter_parameters"]
            + common["head_parameters"]
        ),
        "reciprocal_flops": reciprocal_flops,
        "interaction_flops_per_layer": interaction_flops_per_layer,
        "interaction_flops": interaction_flops,
        "adapter_flops": common["adapter_flops"],
        "head_flops": common["head_flops"],
        "flops": interaction_flops + common["adapter_flops"] + common["head_flops"],
    }


def _ordinary_counts(
    flat_width: int,
    num_layers: int,
    common: Mapping[str, int],
) -> dict[str, Any]:
    interaction_parameters_per_layer = flat_width * flat_width + flat_width
    interaction_parameters = num_layers * interaction_parameters_per_layer
    interaction_flops_per_layer = (
        _biased_linear_flops(flat_width, flat_width) + 2 * flat_width
    )
    interaction_flops = num_layers * interaction_flops_per_layer
    return {
        "architecture": "DCNv2-full-rank-L2-flat-C-plus-P",
        "interaction_parameters_per_layer": interaction_parameters_per_layer,
        "interaction_parameters": interaction_parameters,
        "adapter_parameters": common["adapter_parameters"],
        "head_parameters": common["head_parameters"],
        "parameters": (
            interaction_parameters
            + common["adapter_parameters"]
            + common["head_parameters"]
        ),
        "interaction_flops_per_layer": interaction_flops_per_layer,
        "interaction_flops": interaction_flops,
        "adapter_flops": common["adapter_flops"],
        "head_flops": common["head_flops"],
        "flops": interaction_flops + common["adapter_flops"] + common["head_flops"],
    }


def _candidate_counts(
    operand_width: int,
    rank: int,
    num_layers: int,
    num_experts: int,
    common: Mapping[str, int],
) -> dict[str, Any]:
    expert_parameters = 2 * operand_width * rank + rank * rank
    layer_parameters = (
        num_experts * expert_parameters + num_experts * operand_width + operand_width
    )
    interaction_parameters = num_layers * layer_parameters
    expert_flops = (
        _unbiased_linear_flops(operand_width, rank)
        + rank
        + _unbiased_linear_flops(rank, rank)
        + rank
        + _unbiased_linear_flops(rank, operand_width)
        + 2 * operand_width
        + _unbiased_linear_flops(operand_width, 1)
    )
    softmax_flops = 3 * num_experts - 1
    weighted_residual_flops = 2 * num_experts * operand_width
    layer_flops = num_experts * expert_flops + softmax_flops + weighted_residual_flops
    interaction_flops = num_layers * layer_flops
    return {
        "low_rank": rank,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_parameters": expert_parameters,
        "interaction_parameters_per_layer": layer_parameters,
        "interaction_parameters": interaction_parameters,
        "adapter_parameters": common["adapter_parameters"],
        "head_parameters": common["head_parameters"],
        "parameters": (
            interaction_parameters
            + common["adapter_parameters"]
            + common["head_parameters"]
        ),
        "expert_flops": expert_flops,
        "interaction_flops_per_layer": layer_flops,
        "interaction_flops": interaction_flops,
        "adapter_flops": common["adapter_flops"],
        "head_flops": common["head_flops"],
        "flops": interaction_flops + common["adapter_flops"] + common["head_flops"],
    }


def _ratios(counts: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    parameter_ratio = counts["parameters"] / target["parameters"]
    flop_ratio = counts["flops"] / target["flops"]
    return {
        "parameter_ratio": parameter_ratio,
        "flop_ratio": flop_ratio,
        "eligible_within_plus_or_minus_5_percent": (
            0.95 <= parameter_ratio <= 1.05 and 0.95 <= flop_ratio <= 1.05
        ),
        "distance": abs(math.log(parameter_ratio)) + abs(math.log(flop_ratio)),
    }


def _require_exact_mapping(
    actual: Any,
    expected: Mapping[str, Any],
    field: str,
) -> None:
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        raise CapacityArtifactError(f"{field} fields drifted")
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            if type(actual_value) not in (int, float) or not math.isclose(
                float(actual_value), expected_value, rel_tol=1e-15, abs_tol=0.0
            ):
                raise CapacityArtifactError(f"{field}.{key} drifted")
        elif actual_value != expected_value:
            raise CapacityArtifactError(f"{field}.{key} drifted")


def validate_capacity_artifact(payload: Mapping[str, Any]) -> None:
    """Validate a VK C+P capacity artifact without data, labels, or metrics."""

    root = _exact_keys(
        payload,
        {
            "schema_version",
            "contract",
            "status",
            "resolved_utc",
            "metric_firewall",
            "identity",
            "naming",
            "frozen_bindings",
            "geometry",
            "encoder_parameter_audit",
            "counting_contract",
            "target",
            "ordinary_dcnv2",
            "candidates",
            "decision",
        },
        "artifact",
    )
    if root["schema_version"] != 1 or root["contract"] != CAPACITY_ARTIFACT_CONTRACT:
        raise CapacityArtifactError("artifact contract/version drifted")
    if root["status"] != "resolved_before_first_model_metric":
        raise CapacityArtifactError("artifact status drifted")
    if (
        not isinstance(root["resolved_utc"], str)
        or _UTC_RE.fullmatch(root["resolved_utc"]) is None
    ):
        raise CapacityArtifactError("artifact resolved_utc drifted")
    firewall = _exact_keys(
        root["metric_firewall"],
        {
            "data_rows_opened",
            "labels_opened",
            "validation_metrics_consulted",
            "test_metrics_consulted",
            "gpu_used",
        },
        "metric_firewall",
    )
    if any(value is not False for value in firewall.values()):
        raise CapacityArtifactError(
            "capacity resolution must remain metric/data/GPU free"
        )
    identity = _exact_keys(
        root["identity"], {"dataset_id", "feature_regime"}, "identity"
    )
    if identity != {"dataset_id": "vk_lsvd_up0.01_ir0.01", "feature_regime": "C+P"}:
        raise CapacityArtifactError("capacity identity drifted")
    naming = _exact_keys(
        root["naming"],
        {
            "reporting_label",
            "internal_model_key",
            "historical_frozen_display_alias",
            "alias_policy",
        },
        "naming",
    )
    if naming["reporting_label"] != CANONICAL_DISPLAY_LABEL:
        raise CapacityArtifactError("canonical capacity label drifted")
    if naming["internal_model_key"] != INTERNAL_MODEL_KEY:
        raise CapacityArtifactError("internal capacity key drifted")
    if naming["historical_frozen_display_alias"] != HISTORICAL_DISPLAY_ALIAS:
        raise CapacityArtifactError("historical capacity alias drifted")
    if naming["alias_policy"] != (
        "historical_internal_alias_retained_in_frozen_locks_"
        "reporting_label_authoritative"
    ):
        raise CapacityArtifactError("capacity alias policy drifted")

    bindings = _exact_keys(
        root["frozen_bindings"],
        {
            "search_space_v1_sha256",
            "encoder_v1_sha256",
            "model_sources_v1_sha256",
            "experiment_matrix_v1_sha256",
            "vk_features_v1_sha256",
            "vk_operand_schema_sha256",
            "vk_denominator_mask_sha256",
        },
        "frozen_bindings",
    )
    for key, value in bindings.items():
        _sha256(value, f"frozen_bindings.{key}")
    if dict(bindings) != _EXPECTED_BINDINGS:
        raise CapacityArtifactError("frozen capacity bindings drifted")

    geometry = _exact_keys(
        root["geometry"],
        {
            "ordered_context_field_count",
            "context_embedding_dim",
            "context_bypass_width",
            "operand_width",
            "flat_width",
            "representation_width",
            "head_output_width",
            "dcdn_num_layers",
            "ordinary_dcnv2_num_layers",
            "capacity_num_layers",
            "capacity_num_experts",
        },
        "geometry",
    )
    for key, value in geometry.items():
        _positive_int(value, f"geometry.{key}")
    if dict(geometry) != _EXPECTED_GEOMETRY:
        raise CapacityArtifactError("frozen VK capacity geometry drifted")
    if geometry["context_bypass_width"] != (
        geometry["ordered_context_field_count"] * geometry["context_embedding_dim"]
    ):
        raise CapacityArtifactError("context bypass width is inconsistent")
    if (
        geometry["flat_width"]
        != geometry["context_bypass_width"] + geometry["operand_width"]
    ):
        raise CapacityArtifactError("flat width is inconsistent")
    if geometry["head_output_width"] != 1:
        raise CapacityArtifactError("capacity artifact requires the frozen scalar head")

    encoder_audit = _exact_keys(
        root["encoder_parameter_audit"],
        {
            "trainable_encoder_parameters_total",
            "trainable_encoder_non_embedding_parameters",
            "parameter_owning_module_types",
            "wrapper_non_embedding_delta",
            "conclusion",
        },
        "encoder_parameter_audit",
    )
    if encoder_audit["trainable_encoder_parameters_total"] != 4128:
        raise CapacityArtifactError("VK encoder parameter total drifted")
    if encoder_audit["trainable_encoder_non_embedding_parameters"] != 0:
        raise CapacityArtifactError("non-embedding encoder parameters must be counted")
    if encoder_audit["parameter_owning_module_types"] != ["Embedding"]:
        raise CapacityArtifactError("encoder parameter ownership drifted")
    if encoder_audit["wrapper_non_embedding_delta"] != 0:
        raise CapacityArtifactError("joint wrapper changed non-embedding capacity")
    if (
        encoder_audit["conclusion"]
        != "all_trainable_encoder_parameters_are_nn_Embedding_weights"
    ):
        raise CapacityArtifactError("encoder audit conclusion drifted")

    counting = _exact_keys(
        root["counting_contract"],
        {
            "parameter_scope",
            "embedding_policy",
            "included_components",
            "flop_contract",
            "flop_scalar_operations",
            "flop_zero_cost_operations",
        },
        "counting_contract",
    )
    if (
        counting["parameter_scope"]
        != "all_unique_trainable_non_embedding_parameters_in_joint_wrapper"
    ):
        raise CapacityArtifactError("parameter scope drifted")
    if (
        counting["embedding_policy"]
        != "exclude_only_parameters_owned_directly_by_nn_Embedding_modules"
    ):
        raise CapacityArtifactError("embedding policy drifted")
    if counting["included_components"] != [
        "interaction",
        "visible_adapter",
        "common_linear_head",
    ]:
        raise CapacityArtifactError("included capacity components drifted")
    if counting["flop_contract"] != "dcdn-forward-floating-arithmetic-flops-v1":
        raise CapacityArtifactError("FLOP contract drifted")
    if counting["flop_scalar_operations"] != [
        "multiply",
        "add",
        "reciprocal",
        "tanh",
        "exp",
        "divide",
    ]:
        raise CapacityArtifactError("counted FLOP scalar operations drifted")
    if counting["flop_zero_cost_operations"] != [
        "comparison",
        "mask",
        "concatenation",
        "memory_movement",
        "embedding_lookup",
    ]:
        raise CapacityArtifactError("zero-cost FLOP operations drifted")

    common = _common_counts(geometry["flat_width"], geometry["representation_width"])
    target = _dcdn_counts(
        geometry["operand_width"], geometry["dcdn_num_layers"], common
    )
    ordinary = _ordinary_counts(
        geometry["flat_width"], geometry["ordinary_dcnv2_num_layers"], common
    )
    _require_exact_mapping(root["target"], target, "target")

    ordinary_payload = _exact_keys(
        root["ordinary_dcnv2"],
        set(ordinary) | {"ratios_to_target"},
        "ordinary_dcnv2",
    )
    _require_exact_mapping(
        {key: ordinary_payload[key] for key in ordinary}, ordinary, "ordinary_dcnv2"
    )
    ordinary_ratios = _ratios(ordinary, target)
    _require_exact_mapping(
        ordinary_payload["ratios_to_target"],
        ordinary_ratios,
        "ordinary_dcnv2.ratios_to_target",
    )

    candidates = _exact_keys(
        root["candidates"], {"capacity_rank_lower", "capacity_rank_upper"}, "candidates"
    )
    expected_candidates: dict[str, dict[str, Any]] = {}
    for candidate_id, rank in (
        ("capacity_rank_lower", 12),
        ("capacity_rank_upper", 13),
    ):
        counts = _candidate_counts(
            geometry["operand_width"],
            rank,
            geometry["capacity_num_layers"],
            geometry["capacity_num_experts"],
            common,
        )
        expected_candidates[candidate_id] = {
            **counts,
            "ratios_to_target": _ratios(counts, target),
        }
        _require_exact_mapping(
            candidates[candidate_id], expected_candidates[candidate_id], candidate_id
        )
    if expected_candidates["capacity_rank_lower"]["parameters"] > target["parameters"]:
        raise CapacityArtifactError("lower capacity rank no longer lies below target")
    if expected_candidates["capacity_rank_upper"]["parameters"] < target["parameters"]:
        raise CapacityArtifactError("upper capacity rank no longer brackets target")

    eligible = [
        (candidate_id, candidate)
        for candidate_id, candidate in expected_candidates.items()
        if candidate["ratios_to_target"]["eligible_within_plus_or_minus_5_percent"]
    ]
    selected = min(
        eligible,
        key=lambda item: (
            item[1]["ratios_to_target"]["distance"],
            0 if item[0] == "capacity_rank_lower" else 1,
        ),
    )[0]
    decision = _exact_keys(
        root["decision"],
        {
            "ordinary_reused",
            "mode",
            "selected_candidate",
            "selected_low_rank",
            "selection_rule",
            "optimizer_learning_rate",
        },
        "decision",
    )
    expected_decision = {
        "ordinary_reused": False,
        "mode": "fallback_bracket",
        "selected_candidate": selected,
        "selected_low_rank": expected_candidates[selected]["low_rank"],
        "selection_rule": (
            "minimum_abs_log_parameter_ratio_plus_abs_log_flop_ratio_then_lower_rank"
        ),
        "optimizer_learning_rate": (
            "inherit_selected_same_unit_DCNv2_learning_rate_after_validation_only_tuning"
        ),
    }
    if decision != expected_decision:
        raise CapacityArtifactError("capacity decision drifted")


def capacity_resolution_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project the full artifact to the existing mechanical resolver schema."""

    validate_capacity_artifact(payload)
    target = payload["target"]
    ordinary = payload["ordinary_dcnv2"]
    candidates: dict[str, dict[str, int]] = {}
    for candidate_id in ("capacity_rank_lower", "capacity_rank_upper"):
        candidate = payload["candidates"][candidate_id]
        candidates[candidate_id] = {
            "low_rank": candidate["low_rank"],
            "parameters": candidate["parameters"],
            "flops": candidate["flops"],
        }
    return {
        "metrics_consulted": False,
        "resolved_before_first_model_metric": True,
        "flop_counting_contract": payload["counting_contract"]["flop_contract"],
        "embedding_lookup_excluded": True,
        "adapter_and_common_head_included": True,
        "target": {
            "parameters": target["parameters"],
            "flops": target["flops"],
        },
        "ordinary": {
            "parameters": ordinary["parameters"],
            "flops": ordinary["flops"],
        },
        "candidates": candidates,
        "decision": {
            "mode": payload["decision"]["mode"],
            "selected_candidate": payload["decision"]["selected_candidate"],
        },
    }


def load_verified_capacity_artifact(path: str | Path) -> Mapping[str, Any]:
    """Load a JSON artifact after an exact adjacent SHA-256 sidecar check."""

    artifact_path = Path(path)
    sidecar_path = artifact_path.with_name(f"{artifact_path.name}.sha256")
    if not artifact_path.is_file() or not sidecar_path.is_file():
        raise CapacityArtifactError(
            "capacity JSON and adjacent SHA sidecar are required"
        )
    sidecar_text = sidecar_path.read_text(encoding="ascii")
    expected_line_suffix = f"  {artifact_path.name}\n"
    if not sidecar_text.endswith(expected_line_suffix) or sidecar_text.count("\n") != 1:
        raise CapacityArtifactError("capacity SHA sidecar has an invalid exact format")
    expected_digest = sidecar_text[: -len(expected_line_suffix)]
    _sha256(expected_digest, "capacity sidecar digest")
    actual_digest = sha256_file(artifact_path)
    if actual_digest != expected_digest:
        raise CapacityArtifactError("capacity JSON does not match its SHA sidecar")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapacityArtifactError("capacity JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise CapacityArtifactError("capacity JSON root must be a mapping")
    validate_capacity_artifact(payload)
    return payload


def verify_repository_bindings(
    payload: Mapping[str, Any], repository_root: str | Path
) -> None:
    """Verify that every named frozen input still has the recorded byte hash."""

    validate_capacity_artifact(payload)
    root = Path(repository_root)
    bindings = payload["frozen_bindings"]
    paths = {
        "search_space_v1_sha256": root / "configs/search_space_v1.yaml",
        "encoder_v1_sha256": root / "configs/encoder_v1.yaml",
        "model_sources_v1_sha256": root / "configs/model_sources_v1.yaml",
        "experiment_matrix_v1_sha256": root / "configs/experiment_matrix_v1.yaml",
        "vk_features_v1_sha256": root / "configs/data/vk_lsvd/features_v1.yaml",
    }
    for key, path in paths.items():
        if not path.is_file() or sha256_file(path) != bindings[key]:
            raise CapacityArtifactError(f"frozen repository binding drifted: {key}")
