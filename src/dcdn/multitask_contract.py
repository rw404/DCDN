"""Frozen encoder/capacity contracts for the VK seven-head experiment.

The runtime cache owns materialized field order and row content.  This module
binds that verified identity to an immutable repository lock and derives the
shared encoder specification without opening labels or model metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dcdn.data.vk_lsvd_multitask_runtime import MultiTargetSourceIdentity
from dcdn.encoder import DatasetEncoderSpec
from dcdn.protocol import SHA256_RE, read_verified_artifact


class MultiTargetContractError(RuntimeError):
    """A seven-head encoder or capacity lock drifted."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise MultiTargetContractError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise MultiTargetContractError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise MultiTargetContractError(f"{name} fields drifted")
    return value


@dataclass(frozen=True)
class MultiTargetEncoderLock:
    path: Path
    sha256: str
    payload: dict[str, Any]

    @classmethod
    def load(
        cls, path: str | Path = "configs/encoder_multitask_v2.yaml"
    ) -> MultiTargetEncoderLock:
        lock_path = Path(path)
        try:
            verified = read_verified_artifact(lock_path)
        except Exception as exc:
            raise MultiTargetContractError("encoder lock verification failed") from exc
        if len(verified.data) > 1024 * 1024:
            raise MultiTargetContractError("encoder lock exceeds size limit")
        try:
            payload = yaml.load(verified.data.decode("utf-8"), Loader=_UniqueKeyLoader)
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise MultiTargetContractError("encoder lock is not strict YAML") from exc
        if not isinstance(payload, dict):
            raise MultiTargetContractError("encoder lock must be a mapping")
        cls._validate(payload)
        return cls(lock_path, verified.digest, payload)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        root = _exact_keys(
            payload,
            {
                "schema_version",
                "contract",
                "status",
                "test_unlocked",
                "metric_firewall",
                "bindings",
                "architecture",
                "dataset",
                "regimes",
                "capacity_resolution",
            },
            "encoder lock",
        )
        if type(root["schema_version"]) is not int or root["schema_version"] != 2:
            raise MultiTargetContractError("encoder schema_version must be exact int 2")
        if root["contract"] != "dcdn-vk-seven-head-encoder-capacity-v2":
            raise MultiTargetContractError("encoder contract drifted")
        if root["status"] != "frozen_before_first_real_dataset_model_metric":
            raise MultiTargetContractError("encoder status drifted")
        if root["test_unlocked"] is not False:
            raise MultiTargetContractError("test must remain locked")
        firewall = _exact_keys(
            root["metric_firewall"],
            {"validation_metrics_opened", "test_metrics_opened", "gpu_used"},
            "metric firewall",
        )
        if any(value is not False for value in firewall.values()):
            raise MultiTargetContractError("encoder lock is not metric/GPU free")
        bindings = _exact_keys(
            root["bindings"],
            {
                "data_manifest_sha256",
                "views_manifest_sha256",
                "feature_contract_sha256",
                "feature_policy_sha256",
                "matrix_multitask_sha256",
                "targets_multitask_v2_sha256",
                "target_spec_sha256",
                "task_constants_sha256",
                "full_materialized_scan_sha256",
            },
            "bindings",
        )
        for key, value in bindings.items():
            _sha(value, f"bindings.{key}")
        architecture = _exact_keys(
            root["architecture"],
            {
                "context_embedding_dim",
                "representation_dim",
                "shared_head_hidden_dims",
                "shared_head_dropout",
                "num_targets",
            },
            "architecture",
        )
        if architecture != {
            "context_embedding_dim": 16,
            "representation_dim": 64,
            "shared_head_hidden_dims": [],
            "shared_head_dropout": 0.0,
            "num_targets": 7,
        }:
            raise MultiTargetContractError("common architecture drifted")
        dataset = _exact_keys(
            root["dataset"],
            {"dataset_id", "ordered_context_fields", "categorical_cardinalities"},
            "dataset",
        )
        if dataset != {
            "dataset_id": "vk_lsvd_up0.01_ir0.01",
            "ordered_context_fields": [
                "age",
                "gender",
                "geo",
                "place",
                "platform",
                "agent",
            ],
            "categorical_cardinalities": [55, 4, 82, 24, 13, 22],
        }:
            raise MultiTargetContractError("dataset encoder identity drifted")
        regimes = root["regimes"]
        if not isinstance(regimes, dict) or set(regimes) != {"C+P", "C+P+E"}:
            raise MultiTargetContractError("feature regimes drifted")
        expected_models = {
            "C+P": [
                "dcnv2_full_rank",
                "dcdn_core",
                "afn_plus",
                "dcnv2_capacity_matched_core",
            ],
            "C+P+E": ["dcnv2_full_rank", "dcdn_core", "afn_plus"],
        }
        for name, regime in regimes.items():
            item = _exact_keys(
                regime,
                {
                    "operand_width",
                    "flat_width",
                    "physical_schema_sha256",
                    "expanded_schema_sha256",
                    "operand_schema_sha256",
                    "denominator_mask_sha256",
                    "cache_content_sha256",
                    "cache_metadata_sha256",
                    "allowed_models",
                },
                f"regimes.{name}",
            )
            for key in (
                "physical_schema_sha256",
                "expanded_schema_sha256",
                "operand_schema_sha256",
                "denominator_mask_sha256",
                "cache_content_sha256",
                "cache_metadata_sha256",
            ):
                _sha(item[key], f"regimes.{name}.{key}")
            if (
                type(item["operand_width"]) is not int
                or type(item["flat_width"]) is not int
            ):
                raise MultiTargetContractError("regime widths must be exact integers")
            if item["flat_width"] != 96 + item["operand_width"]:
                raise MultiTargetContractError("regime flat width drifted")
            if item["allowed_models"] != expected_models[name]:
                raise MultiTargetContractError("regime model allowlist drifted")
        capacity = _exact_keys(
            root["capacity_resolution"],
            {
                "target_model",
                "control_model",
                "selection_used_validation_quality",
                "tolerance_fraction",
                "target_parameters",
                "target_flops",
                "selected_rank",
                "selected_parameters",
                "selected_flops",
                "parameter_ratio",
                "flop_ratio",
            },
            "capacity_resolution",
        )
        if (
            capacity["target_model"] != "dcdn_core"
            or capacity["control_model"] != "dcnv2_capacity_matched_core"
            or capacity["selection_used_validation_quality"] is not False
            or capacity["tolerance_fraction"] != 0.05
            or capacity["target_parameters"] != 22503
            or capacity["target_flops"] != 44919
            or capacity["selected_rank"] != 12
            or capacity["selected_parameters"] != 22445
            or capacity["selected_flops"] != 45862
        ):
            raise MultiTargetContractError("capacity resolution drifted")
        if not (0.95 <= capacity["parameter_ratio"] <= 1.05):
            raise MultiTargetContractError(
                "capacity parameter ratio is outside tolerance"
            )
        if not (0.95 <= capacity["flop_ratio"] <= 1.05):
            raise MultiTargetContractError("capacity FLOP ratio is outside tolerance")

    def resolve(self, identity: MultiTargetSourceIdentity) -> DatasetEncoderSpec:
        self._validate(self.payload)
        bindings = self.payload["bindings"]
        regime = self.payload["regimes"].get(identity.feature_regime)
        if not isinstance(regime, dict):
            raise MultiTargetContractError("source regime is not frozen")
        expected = {
            "dataset_id": self.payload["dataset"]["dataset_id"],
            "data_manifest_sha256": bindings["data_manifest_sha256"],
            "views_manifest_sha256": bindings["views_manifest_sha256"],
            "feature_contract_sha256": bindings["feature_contract_sha256"],
            "full_materialized_scan_sha256": bindings["full_materialized_scan_sha256"],
            "targets_multitask_v2_sha256": bindings["targets_multitask_v2_sha256"],
            "feature_policy_sha256": bindings["feature_policy_sha256"],
            "matrix_multitask_sha256": bindings["matrix_multitask_sha256"],
            "task_constants_sha256": bindings["task_constants_sha256"],
            "physical_schema_sha256": regime["physical_schema_sha256"],
            "expanded_schema_sha256": regime["expanded_schema_sha256"],
            "operand_schema_sha256": regime["operand_schema_sha256"],
            "denominator_mask_sha256": regime["denominator_mask_sha256"],
            "cache_content_sha256": regime["cache_content_sha256"],
            "ordered_context_fields": tuple(
                self.payload["dataset"]["ordered_context_fields"]
            ),
            "categorical_cardinalities": tuple(
                self.payload["dataset"]["categorical_cardinalities"]
            ),
        }
        for key, value in expected.items():
            if getattr(identity, key) != value:
                raise MultiTargetContractError(f"source identity mismatch: {key}")
        if len(identity.ordered_operand_fields) != regime["operand_width"]:
            raise MultiTargetContractError("source operand width drifted")
        return DatasetEncoderSpec(
            dataset_id=identity.dataset_id,
            feature_regime=identity.feature_regime,
            encoder_contract_sha256=self.sha256,
            data_manifest_sha256=identity.data_manifest_sha256,
            views_manifest_sha256=identity.views_manifest_sha256,
            feature_contract_sha256=identity.feature_contract_sha256,
            physical_schema_sha256=identity.physical_schema_sha256,
            full_feature_schema_sha256=identity.expanded_schema_sha256,
            operand_schema_sha256=identity.operand_schema_sha256,
            denominator_mask_sha256=identity.denominator_mask_sha256,
            full_materialized_scan_sha256=identity.full_materialized_scan_sha256,
            cache_content_sha256=identity.cache_content_sha256,
            ordered_context_fields=identity.ordered_context_fields,
            categorical_cardinalities=identity.categorical_cardinalities,
            ordered_operand_fields=identity.ordered_operand_fields,
            denominator_mask=identity.denominator_mask,
            allowed_model_keys=tuple(regime["allowed_models"]),
        )


__all__ = ["MultiTargetContractError", "MultiTargetEncoderLock"]
