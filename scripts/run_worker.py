#!/usr/bin/env python3
"""Registry-owned one-process/one-GPU worker with local safety guards."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import FrameType
from typing import Any

from cost_guard import evaluate_cost
from storage_guard import inspect_storage

from dcdn.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    CheckpointMetadata,
    _load_verified_payload,
    canonical_json_bytes,
    verify_checkpoint_sidecar,
)
from dcdn.protocol import (
    load_verified_yaml_artifact,
    read_verified_artifact,
    verify_sha256_sidecar,
)
from dcdn.registry import (
    CHECKPOINT_SIZE_VERIFIER_FAILURE_REASON,
    RunRegistry,
)
from dcdn.selection_artifact import (
    CAPACITY_CANDIDATE_ID,
    CAPACITY_LOW_RANK,
    CAPACITY_RESOLUTION_SHA256,
    EMPTY_DIFF_SHA256,
    ORDINARY_SELECTION_SHA256,
    PREMETRIC_PROTOCOL_SHA256,
    SELECTED_LEARNING_RATE,
    CapacityBindingEvidence,
    load_capacity_lr_binding,
    validate_capacity_training_binding_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_model.py"
_TELEMETRY_ENV = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_DATASETS_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
}


class WorkerError(RuntimeError):
    """A worker ownership or local safety invariant failed."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BEST_CHECKPOINT_RE = re.compile(r"^best-step-[0-9]{12}-generation-[0-9]{4}\.pt$")
_REGISTRY_HELPER_MODULE = "dcdn.registry_helper"
_HEARTBEAT_INTERVAL_SECONDS = 30.0
_REGISTRY_OPERATION_TIMEOUT_SECONDS = 10.0
_REGISTRY_DEGRADED_DEADLINE_SECONDS = 180.0
_RUNNING_PUBLISH_DEADLINE_SECONDS = 30.0
_TERMINAL_PUBLISH_DEADLINE_SECONDS = 300.0
_REGISTRY_RETRY_DELAYS_SECONDS = (1.0, 2.0, 5.0, 10.0, 15.0, 30.0)
_REGISTRY_RESPONSE_LIMIT_BYTES = 64 * 1024


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkerError(f"{field} must be a mapping")
    return value


def _resolved_path(value: Any, *, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise WorkerError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _inside(path: Path, root: Path, field: str) -> Path:
    if not path.is_relative_to(root):
        raise WorkerError(f"{field} must remain inside project_root")
    return path


def _validate_resolved_execution_contract(
    resolved: Mapping[str, Any],
) -> str:
    training = _mapping(resolved.get("training"), "resolved.training")
    is_capacity = training.get("model_key") == "dcnv2_capacity_matched_core"
    schema_version = resolved.get("schema_version")
    if type(schema_version) is not int or (
        (schema_version == 3 and is_capacity)
        or (schema_version == 4 and not is_capacity)
        or schema_version not in {3, 4}
    ):
        raise WorkerError("resolved schema must be v3 non-capacity or v4 capacity")
    execution_kind = resolved.get("execution_kind")
    if type(execution_kind) is not str or execution_kind not in {
        "implementation_smoke",
        "scientific_tuning",
    }:
        raise WorkerError("resolved execution_kind is invalid")
    runtime = _mapping(resolved.get("runtime"), "resolved.runtime")
    runtime_value_checks = training.get("runtime_value_checks")
    if type(runtime_value_checks) is not bool:
        raise WorkerError("resolved runtime_value_checks must be boolean")
    if execution_kind == "scientific_tuning":
        if runtime_value_checks:
            raise WorkerError("scientific tuning requires runtime_value_checks=false")
        return execution_kind

    effective_batch = training.get("effective_batch_size")
    validation_cadence = training.get("validate_every_examples")
    if (
        isinstance(effective_batch, bool)
        or not isinstance(effective_batch, int)
        or effective_batch != 4096
        or isinstance(validation_cadence, bool)
        or not isinstance(validation_cadence, int)
        or validation_cadence <= effective_batch
    ):
        raise WorkerError("implementation smoke training budget is invalid")
    if runtime_value_checks is not True:
        raise WorkerError("implementation smoke requires runtime_value_checks=true")
    if runtime.get("resume") is not False:
        raise WorkerError("implementation smoke must be a fresh run")
    if (
        type(runtime.get("stop_after_examples")) is not int
        or runtime["stop_after_examples"] != effective_batch
    ):
        raise WorkerError(
            "implementation smoke must stop after exactly one effective batch"
        )
    if "latency" not in runtime or runtime["latency"] is not None:
        raise WorkerError("implementation smoke forbids latency evaluation")
    return execution_kind


def _verify_capacity_binding(
    resolved: Mapping[str, Any],
    project_root: Path,
    *,
    record: Mapping[str, Any] | None = None,
) -> CapacityBindingEvidence | None:
    """Re-open and compare every capacity authorization boundary."""

    training = _mapping(resolved.get("training"), "resolved.training")
    binding_fields = {
        "ordinary_dcnv2_selection_sha256",
        "capacity_lr_binding_sha256",
    }
    is_capacity = training.get("model_key") == "dcnv2_capacity_matched_core"
    schema_version = resolved.get("schema_version")
    if not is_capacity:
        if (
            type(schema_version) is not int
            or schema_version != 3
            or "capacity_lr_binding_path" in resolved
            or set(training) & binding_fields
        ):
            raise WorkerError("non-capacity resolved config contains capacity binding")
        return None
    if type(schema_version) is not int or schema_version != 4:
        raise WorkerError("capacity resolved config requires schema v4")
    if record is not None and record.get("code_diff_sha256") != EMPTY_DIFF_SHA256:
        raise WorkerError("capacity registry requires an empty tracked code diff")
    binding_path = _inside(
        _resolved_path(
            resolved.get("capacity_lr_binding_path"),
            root=project_root,
            field="resolved.capacity_lr_binding_path",
        ),
        project_root,
        "resolved.capacity_lr_binding_path",
    )
    try:
        evidence = load_capacity_lr_binding(binding_path, project_root)
    except Exception as exc:
        raise WorkerError(f"capacity LR binding verification failed: {exc}") from exc
    protocol_path = _inside(
        _resolved_path(
            resolved.get("protocol_path"),
            root=project_root,
            field="resolved.protocol_path",
        ),
        project_root,
        "resolved.protocol_path",
    )
    capacity_path = _inside(
        _resolved_path(
            resolved.get("capacity_artifact_path"),
            root=project_root,
            field="resolved.capacity_artifact_path",
        ),
        project_root,
        "resolved.capacity_artifact_path",
    )
    model = _mapping(resolved.get("model"), "resolved.model")
    interaction = _mapping(
        model.get("interaction_config"),
        "resolved.model.interaction_config",
    )
    head = _mapping(model.get("head_config"), "resolved.model.head_config")
    expected_head = {"hidden_dims": [], "dropout": 0.0, "output_dim": 1}
    expected_interaction = {
        "model_key": "dcnv2_capacity_matched_core",
        "num_layers": 2,
        "low_rank": CAPACITY_LOW_RANK,
        "num_experts": 4,
        "product_init_scale": 0.1,
    }
    if (
        type(head.get("hidden_dims")) is not list
        or head.get("hidden_dims") != []
        or type(head.get("dropout")) is not float
        or type(head.get("output_dim")) is not int
        or dict(head) != expected_head
        or type(interaction.get("num_layers")) is not int
        or type(interaction.get("low_rank")) is not int
        or type(interaction.get("num_experts")) is not int
        or type(interaction.get("product_init_scale")) is not float
        or dict(interaction) != expected_interaction
        or training.get("candidate_id") != CAPACITY_CANDIDATE_ID
        or type(training.get("learning_rate")) is not float
        or training.get("learning_rate") != SELECTED_LEARNING_RATE
        or training.get("capacity_resolution_sha256") != CAPACITY_RESOLUTION_SHA256
        or training.get("ordinary_dcnv2_selection_sha256") != ORDINARY_SELECTION_SHA256
        or training.get("capacity_lr_binding_sha256") != evidence.sha256
        or evidence.ordinary.learning_rate != SELECTED_LEARNING_RATE
        or verify_sha256_sidecar(protocol_path) != PREMETRIC_PROTOCOL_SHA256
        or verify_sha256_sidecar(capacity_path) != CAPACITY_RESOLUTION_SHA256
    ):
        raise WorkerError("capacity resolved binding values drifted")
    if record is not None:
        if (
            type(record.get("learning_rate")) is not float
            or type(record.get("capacity_low_rank")) is not int
        ):
            raise WorkerError("registry capacity binding scalar types mismatch")
        expected_registry = {
            "model_key": "dcnv2_capacity_matched_core",
            "candidate_id": CAPACITY_CANDIDATE_ID,
            "learning_rate": SELECTED_LEARNING_RATE,
            "capacity_low_rank": CAPACITY_LOW_RANK,
            "capacity_resolution_sha256": CAPACITY_RESOLUTION_SHA256,
            "ordinary_dcnv2_selection_sha256": ORDINARY_SELECTION_SHA256,
            "capacity_lr_binding_sha256": evidence.sha256,
        }
        for name, expected in expected_registry.items():
            if record.get(name) != expected:
                raise WorkerError(f"registry capacity binding mismatch: {name}")
    return evidence


def _verify_capacity_checkpoint_binding(
    path: Path,
    *,
    expected_digest: str,
    expected_role: str,
    resolved_training: Mapping[str, Any],
    expected_joint_model_contract_sha256: str,
    expected_best_sha256: str | None,
    expected_best_filename: str | None,
) -> None:
    """Safely inspect capacity checkpoint metadata without restoring state."""

    try:
        payload, digest = _load_verified_payload(path, map_location="cpu")
    except Exception as exc:
        raise WorkerError(
            f"capacity checkpoint safe inspection failed: {path.name}: {exc}"
        ) from exc
    if digest != expected_digest:
        raise WorkerError("capacity checkpoint digest differs from result")
    expected_root = {
        "format",
        "version",
        "metadata",
        "progress",
        "components",
        "rng_state",
        "extra_state",
    }
    if set(payload) != expected_root:
        raise WorkerError("capacity checkpoint payload keys mismatch")
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or type(payload.get("version")) is not int
        or payload.get("version") != CHECKPOINT_VERSION
    ):
        raise WorkerError("capacity checkpoint format/version mismatch")
    try:
        metadata = CheckpointMetadata.from_dict(payload["metadata"])
    except Exception as exc:
        raise WorkerError(f"capacity checkpoint metadata is invalid: {exc}") from exc
    checkpoint_config = _mapping(
        metadata.resolved_config,
        "checkpoint.metadata.resolved_config",
    )
    checkpoint_training = _mapping(
        checkpoint_config.get("training"),
        "checkpoint.metadata.resolved_config.training",
    )
    try:
        validate_capacity_training_binding_payload(
            checkpoint_config.get("capacity_training_binding"),
            capacity_resolution_sha256=resolved_training["capacity_resolution_sha256"],
            ordinary_dcnv2_selection_sha256=resolved_training[
                "ordinary_dcnv2_selection_sha256"
            ],
            capacity_lr_binding_sha256=resolved_training["capacity_lr_binding_sha256"],
        )
    except Exception as exc:
        raise WorkerError("capacity checkpoint training binding is invalid") from exc
    if (
        canonical_json_bytes(checkpoint_training)
        != canonical_json_bytes(resolved_training)
        or checkpoint_config.get("capacity_resolution_sha256")
        != CAPACITY_RESOLUTION_SHA256
        or checkpoint_config.get("checkpoint_protocol_sha256")
        != PREMETRIC_PROTOCOL_SHA256
        or metadata.protocol_hash != PREMETRIC_PROTOCOL_SHA256
        or checkpoint_config.get("joint_model_contract_sha256")
        != expected_joint_model_contract_sha256
    ):
        raise WorkerError("capacity checkpoint metadata binding mismatch")
    extra_state = _mapping(payload.get("extra_state"), "checkpoint.extra_state")
    if extra_state.get("checkpoint_role") != expected_role:
        raise WorkerError("capacity checkpoint role mismatch")
    if expected_role == "last":
        if (
            extra_state.get("selected_best_checkpoint_sha256") != expected_best_sha256
            or extra_state.get("selected_best_checkpoint_filename")
            != expected_best_filename
        ):
            raise WorkerError("last checkpoint best-generation lineage mismatch")
    elif expected_role == "best":
        if (
            extra_state.get("selected_best_checkpoint_sha256") is not None
            or extra_state.get("selected_best_checkpoint_filename") is not None
        ):
            raise WorkerError("best checkpoint recursively binds a generation")
    else:
        raise WorkerError("capacity checkpoint expected role is invalid")


def _require_registry_execution_kind(
    record: Mapping[str, Any],
    expected: str,
) -> None:
    value = record.get("execution_kind")
    if type(value) is not str or value != expected:
        raise WorkerError("registry execution_kind differs from resolved config")


def _completed_reason(execution_kind: str, result_digest: str) -> str:
    if _SHA256_RE.fullmatch(result_digest) is None:
        raise WorkerError("completed result digest is invalid")
    if execution_kind == "implementation_smoke":
        scope = "NON-SCIENTIFIC implementation smoke"
    elif execution_kind == "scientific_tuning":
        scope = "scientific tuning"
    else:
        raise WorkerError("completed execution_kind is invalid")
    return (
        f"{scope}; trainer exited successfully; exact immutable result "
        f"re-opened and verified: {result_digest}"
    )


def _verify_completed_result(
    *,
    project_root: Path,
    resolved_config: Path,
    expected_resolved_digest: str,
    expected_resolved: Mapping[str, Any],
    run_id: str,
) -> tuple[str, bool]:
    """Re-open exact config/result bytes before publishing completed state."""

    resolved_artifact = load_verified_yaml_artifact(resolved_config)
    if resolved_artifact.digest != expected_resolved_digest:
        raise WorkerError("resolved config digest drifted while worker was running")
    reloaded = resolved_artifact.payload
    if reloaded != expected_resolved:
        raise WorkerError("resolved config payload drifted while worker was running")
    execution_kind = _validate_resolved_execution_contract(reloaded)
    capacity_evidence = _verify_capacity_binding(reloaded, project_root)
    is_capacity = capacity_evidence is not None
    result_path = _inside(
        _resolved_path(
            reloaded.get("result_path"),
            root=project_root,
            field="resolved.result_path",
        ),
        project_root,
        "resolved.result_path",
    )
    artifact = read_verified_artifact(result_path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if not isinstance(key, str) or key in result:
                raise ValueError(f"duplicate or invalid JSON key: {key!r}")
            result[key] = value
        return result

    try:
        receipt = json.loads(
            artifact.data.decode("ascii"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkerError("result receipt is not strict finite ASCII JSON") from exc
    expected_keys = {
        "schema_version",
        "execution_kind",
        "scientific_result_eligible",
        "run_id",
        "resolved_config_sha256",
        "protocol_sha256",
        "training",
        "selected_validation_metric",
        "test_metrics",
        "end_to_end_raw_to_logit_validation_latency",
        "capacity_resolution_sha256",
        "joint_model_contract_sha256",
        "final_cost_snapshot",
        "final_storage_snapshot",
        "telemetry",
    }
    if is_capacity:
        expected_keys.add("capacity_training_binding")
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise WorkerError("result receipt keys mismatch")
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != reloaded["schema_version"]
        or receipt["run_id"] != run_id
    ):
        raise WorkerError("result receipt identity mismatch")
    if receipt["execution_kind"] != execution_kind:
        raise WorkerError("result receipt execution_kind mismatch")
    if type(receipt["scientific_result_eligible"]) is not bool:
        raise WorkerError("result scientific eligibility must be boolean")
    if receipt["resolved_config_sha256"] != expected_resolved_digest:
        raise WorkerError("result receipt resolved-config digest mismatch")
    protocol_path = _inside(
        _resolved_path(
            reloaded.get("protocol_path"),
            root=project_root,
            field="resolved.protocol_path",
        ),
        project_root,
        "resolved.protocol_path",
    )
    if receipt["protocol_sha256"] != verify_sha256_sidecar(protocol_path):
        raise WorkerError("result receipt protocol digest mismatch")
    training = _mapping(receipt["training"], "result.training")
    resolved_training = _mapping(reloaded.get("training"), "resolved.training")
    expected_training_keys = {
        "examples_seen",
        "optimizer_steps",
        "validation_checks",
        "best_validation_metric",
        "stopped_reason",
        "elapsed_seconds",
        "throughput_examples_per_second",
        "peak_vram_bytes",
        "oom_count",
        "last_checkpoint_sha256",
        "best_checkpoint_sha256",
        "best_checkpoint_filename",
        "numerical_summary",
        "reproduction_command",
        "declared_maximum_train_examples",
        "realized_maximum_train_examples",
    }
    if set(training) != expected_training_keys:
        raise WorkerError("result TrainingResult keys mismatch")
    effective_batch = resolved_training.get("effective_batch_size")
    declared_maximum = resolved_training.get("maximum_train_examples")
    if (
        isinstance(effective_batch, bool)
        or not isinstance(effective_batch, int)
        or effective_batch <= 0
        or isinstance(declared_maximum, bool)
        or not isinstance(declared_maximum, int)
        or declared_maximum <= 0
    ):
        raise WorkerError("resolved training example budget is invalid")
    realized_maximum = (declared_maximum // effective_batch) * effective_batch
    examples_seen = training["examples_seen"]
    optimizer_steps = training["optimizer_steps"]
    if (
        isinstance(examples_seen, bool)
        or not isinstance(examples_seen, int)
        or examples_seen <= 0
        or isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps <= 0
        or examples_seen != optimizer_steps * effective_batch
        or examples_seen > realized_maximum
    ):
        raise WorkerError("result examples/effective-batch equality is invalid")
    if (
        type(training["declared_maximum_train_examples"]) is not int
        or type(training["realized_maximum_train_examples"]) is not int
        or training["declared_maximum_train_examples"] != declared_maximum
        or training["realized_maximum_train_examples"] != realized_maximum
    ):
        raise WorkerError("result declared/realized example budget mismatch")
    for name in ("validation_checks", "peak_vram_bytes", "oom_count"):
        value = training[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WorkerError(f"result {name} is invalid")
    if training["oom_count"] > 1:
        raise WorkerError("result OOM count exceeds the frozen limit")
    for name in (
        "elapsed_seconds",
        "throughput_examples_per_second",
    ):
        value = training[name]
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise WorkerError(f"result {name} must be finite and positive")
    if (
        not isinstance(training["stopped_reason"], str)
        or not training["stopped_reason"]
        or training["reproduction_command"]
        != resolved_training.get("reproduction_command")
    ):
        raise WorkerError("result stop/reproduction metadata mismatch")
    numerical = _mapping(training["numerical_summary"], "numerical_summary")
    if set(numerical) != {
        "submitted",
        "completed",
        "pending",
        "nonfinite_total",
        "nonfinite_by_kind",
        "asynchronous_cuda",
        "activation_hook_count",
        "activation_hook_observations",
    }:
        raise WorkerError("result numerical summary keys mismatch")
    for name in (
        "submitted",
        "completed",
        "pending",
        "nonfinite_total",
        "activation_hook_count",
        "activation_hook_observations",
    ):
        value = numerical[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WorkerError(f"result numerical counter {name} is invalid")
    if (
        numerical["pending"] != 0
        or numerical["submitted"] != numerical["completed"]
        or numerical["submitted"] <= 0
        or numerical["nonfinite_total"] != 0
    ):
        raise WorkerError("result numerical monitor was not durably drained")
    if numerical["nonfinite_by_kind"] != {}:
        raise WorkerError("result contains non-finite numerical observations")
    if (
        numerical["activation_hook_count"] != 3
        or numerical["activation_hook_observations"] <= 0
        or numerical["activation_hook_observations"] > numerical["submitted"]
    ):
        raise WorkerError("result activation-hook evidence is incomplete")
    resolved_runtime = _mapping(reloaded.get("runtime"), "resolved.runtime")
    expected_async = resolved_runtime.get("device") == "cuda:0"
    if numerical["asynchronous_cuda"] is not expected_async:
        raise WorkerError("result numerical monitor device mode mismatch")
    run_directory = _inside(
        _resolved_path(
            reloaded.get("run_directory"),
            root=project_root,
            field="resolved.run_directory",
        ),
        project_root,
        "resolved.run_directory",
    )
    last_digest = training["last_checkpoint_sha256"]
    if (
        not isinstance(last_digest, str)
        or _SHA256_RE.fullmatch(last_digest) is None
        or verify_checkpoint_sidecar(run_directory / "last.pt") != last_digest
    ):
        raise WorkerError("result last checkpoint is missing, stale, or corrupt")
    best_metric = training["best_validation_metric"]
    best_digest = training["best_checkpoint_sha256"]
    best_filename = training["best_checkpoint_filename"]
    if best_metric is None:
        if best_digest is not None or best_filename is not None:
            raise WorkerError("result has a best generation without a metric")
    elif (
        type(best_metric) is not float
        or not math.isfinite(best_metric)
        or training["validation_checks"] <= 0
        or not isinstance(best_digest, str)
        or _SHA256_RE.fullmatch(best_digest) is None
        or not isinstance(best_filename, str)
        or _BEST_CHECKPOINT_RE.fullmatch(best_filename) is None
        or verify_checkpoint_sidecar(run_directory / best_filename) != best_digest
    ):
        raise WorkerError("result best generation is missing, stale, or corrupt")
    if is_capacity:
        _verify_capacity_checkpoint_binding(
            run_directory / "last.pt",
            expected_digest=last_digest,
            expected_role="last",
            resolved_training=resolved_training,
            expected_joint_model_contract_sha256=receipt["joint_model_contract_sha256"],
            expected_best_sha256=best_digest,
            expected_best_filename=best_filename,
        )
        if best_metric is not None:
            assert isinstance(best_filename, str)
            assert isinstance(best_digest, str)
            _verify_capacity_checkpoint_binding(
                run_directory / best_filename,
                expected_digest=best_digest,
                expected_role="best",
                resolved_training=resolved_training,
                expected_joint_model_contract_sha256=receipt[
                    "joint_model_contract_sha256"
                ],
                expected_best_sha256=None,
                expected_best_filename=None,
            )
    selected_metric = receipt["selected_validation_metric"]
    if (
        (best_metric is None and selected_metric is not None)
        or (best_metric is not None and type(selected_metric) is not float)
        or selected_metric != best_metric
    ):
        raise WorkerError("selected validation metric differs from training result")
    if receipt["test_metrics"] is not None:
        raise WorkerError("premetric worker result must not contain test metrics")
    latency = receipt["end_to_end_raw_to_logit_validation_latency"]
    if latency is not None:
        latency = _mapping(latency, "validation latency")
        if set(latency) != {
            "warmup_repetitions",
            "measured_repetitions",
            "batch_size",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "scope",
        }:
            raise WorkerError("validation latency keys mismatch")
        for name in (
            "warmup_repetitions",
            "measured_repetitions",
            "batch_size",
        ):
            value = latency[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise WorkerError(f"validation latency {name} is invalid")
        percentiles: list[float] = []
        for name in ("p50_ms", "p95_ms", "p99_ms"):
            value = latency[name]
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise WorkerError(f"validation latency {name} is invalid")
            percentiles.append(value)
        if percentiles != sorted(percentiles):
            raise WorkerError("validation latency percentiles are not ordered")
        if latency["scope"] != "end_to_end_raw_to_logit_model":
            raise WorkerError("validation latency scope is invalid")

    scientific_result_eligible = receipt["scientific_result_eligible"]
    if execution_kind == "implementation_smoke":
        if scientific_result_eligible is not False:
            raise WorkerError("implementation smoke must be explicitly non-scientific")
        if (
            examples_seen != effective_batch
            or optimizer_steps != 1
            or training["validation_checks"] != 0
            or best_metric is not None
            or best_digest is not None
            or best_filename is not None
            or receipt["selected_validation_metric"] is not None
            or latency is not None
            or training["stopped_reason"] != "requested_checkpoint_boundary"
        ):
            raise WorkerError(
                "implementation smoke contains validation/scientific result state"
            )
    else:
        expected_eligibility = (
            training["validation_checks"] > 0
            and best_metric is not None
            and best_digest is not None
            and best_filename is not None
        )
        if scientific_result_eligible is not expected_eligibility:
            raise WorkerError("scientific result eligibility is inconsistent")
    if receipt["capacity_resolution_sha256"] != resolved_training.get(
        "capacity_resolution_sha256"
    ):
        raise WorkerError("result capacity artifact digest mismatch")
    if is_capacity:
        assert capacity_evidence is not None
        try:
            validate_capacity_training_binding_payload(
                receipt.get("capacity_training_binding"),
                capacity_resolution_sha256=resolved_training[
                    "capacity_resolution_sha256"
                ],
                ordinary_dcnv2_selection_sha256=resolved_training[
                    "ordinary_dcnv2_selection_sha256"
                ],
                capacity_lr_binding_sha256=resolved_training[
                    "capacity_lr_binding_sha256"
                ],
            )
        except Exception as exc:
            raise WorkerError("result capacity training binding mismatch") from exc
    if (
        not isinstance(receipt["joint_model_contract_sha256"], str)
        or _SHA256_RE.fullmatch(receipt["joint_model_contract_sha256"]) is None
    ):
        raise WorkerError("result joint model contract digest is invalid")
    if (
        _mapping(receipt["final_cost_snapshot"], "final_cost_snapshot").get(
            "launch_allowed"
        )
        is not True
    ):
        raise WorkerError("result final cost guard did not pass")
    if (
        _mapping(receipt["final_storage_snapshot"], "final_storage_snapshot").get("ok")
        is not True
    ):
        raise WorkerError("result final storage guard did not pass")
    if receipt["telemetry"] != "disabled":
        raise WorkerError("result telemetry state mismatch")
    return artifact.digest, scientific_result_eligible


@dataclass(frozen=True)
class _RegistryOperationResult:
    ok: bool
    result: Mapping[str, Any] | None = None
    transient_candidate: bool = False
    error_type: str | None = None
    message: str | None = None
    stage: str | None = None
    errno: int | None = None
    retain_pending: bool = False
    failure_already_reported: bool = False


@dataclass
class _PendingRegistryOperation:
    process: subprocess.Popen[bytes]
    request: Mapping[str, Any]
    started_at: float
    kill_sent: bool = False
    timeout_reported: bool = False


def _registry_helper_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "RANK",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
    ):
        environment.pop(name, None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["WORLD_SIZE"] = "1"
    for name, value in _TELEMETRY_ENV.items():
        environment[name] = value
    source_path = str(PROJECT_ROOT / "src")
    inherited_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + inherited_python_path if inherited_python_path else ""
    )
    return environment


def _signal_pending_registry_helper(
    pending: _PendingRegistryOperation,
) -> None:
    if pending.kill_sent:
        return
    with suppress(ProcessLookupError):
        os.killpg(pending.process.pid, signal.SIGKILL)
    pending.kill_sent = True


@dataclass
class _RegistryHelperOwner:
    pending: _PendingRegistryOperation | None = None

    def require_idle(self) -> None:
        if self.pending is not None:
            raise WorkerError(
                "cannot start a registry helper while the previous helper "
                "remains unreaped"
            )

    def adopt(self, pending: _PendingRegistryOperation) -> None:
        self.require_idle()
        self.pending = pending

    def release(self, pending: _PendingRegistryOperation) -> None:
        if self.pending is not pending:
            raise WorkerError("registry helper owner lost process identity")
        if pending.process.poll() is None:
            raise WorkerError("cannot release an unreaped registry helper")
        self.pending = None

    def close(
        self,
        *,
        deadline_monotonic: float,
        now_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        pending = self.pending
        if pending is None:
            return
        _signal_pending_registry_helper(pending)
        while True:
            if pending.process.poll() is not None:
                self.release(pending)
                return
            try:
                pending.process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
            else:
                self.release(pending)
                return
            now = now_fn()
            if now >= deadline_monotonic:
                raise WorkerError(
                    "registry helper remained unreaped at the cleanup deadline"
                )
            sleep_fn(min(0.1, max(0.0, deadline_monotonic - now)))


def _registry_helper_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "dcdn.supervised_exec",
        "--expected-parent-pid",
        str(os.getpid()),
        "--",
        sys.executable,
        "-m",
        _REGISTRY_HELPER_MODULE,
    ]


def _start_registry_operation(
    request: Mapping[str, Any],
    now: float,
    owner: _RegistryHelperOwner,
) -> _PendingRegistryOperation:
    owner.require_idle()
    body = (
        json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    if len(body) > _REGISTRY_RESPONSE_LIMIT_BYTES:
        raise WorkerError("registry helper request exceeds the bounded limit")
    try:
        process = subprocess.Popen(
            _registry_helper_command(),
            cwd="/tmp",
            env=_registry_helper_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise WorkerError(f"could not start registry helper: {exc}") from exc
    pending = _PendingRegistryOperation(process, dict(request), now)
    owner.adopt(pending)
    try:
        if process.stdin is None:
            raise WorkerError("registry helper stdin was not created")
        process.stdin.write(body)
        process.stdin.close()
    except BaseException:
        _signal_pending_registry_helper(pending)
        raise
    return pending


def _bounded_pipe_read(pipe: Any, *, field: str) -> bytes:
    if pipe is None:
        raise WorkerError(f"registry helper {field} pipe was not created")
    data = pipe.read(_REGISTRY_RESPONSE_LIMIT_BYTES + 1)
    if len(data) > _REGISTRY_RESPONSE_LIMIT_BYTES:
        raise WorkerError(f"registry helper {field} exceeded the bounded limit")
    return data


def _poll_registry_operation(
    pending: _PendingRegistryOperation,
    now: float,
    *,
    timeout_seconds: float = _REGISTRY_OPERATION_TIMEOUT_SECONDS,
) -> _RegistryOperationResult | None:
    return_code = pending.process.poll()
    if pending.timeout_reported:
        if return_code is None:
            try:
                pending.process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                return None
        return _RegistryOperationResult(
            ok=False,
            transient_candidate=True,
            error_type="RegistryHelperReapedAfterTimeout",
            message="timed-out registry helper was reaped",
            stage="helper_reaped_after_timeout",
            failure_already_reported=True,
        )
    if return_code is None:
        if now - pending.started_at < timeout_seconds:
            return None
        _signal_pending_registry_helper(pending)
        pending.timeout_reported = True
        return _RegistryOperationResult(
            ok=False,
            transient_candidate=True,
            error_type="RegistryHelperTimeout",
            message=f"registry helper exceeded {timeout_seconds:.3f}s",
            stage="helper_timeout",
            retain_pending=True,
        )
    stdout = _bounded_pipe_read(pending.process.stdout, field="stdout")
    stderr = _bounded_pipe_read(pending.process.stderr, field="stderr")
    try:
        response = json.loads(stdout.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        detail = stderr.decode("utf-8", errors="replace")[:500]
        raise WorkerError(
            "registry helper returned malformed JSON"
            + (f": {detail}" if detail else "")
        ) from exc
    if not isinstance(response, Mapping) or type(response.get("ok")) is not bool:
        raise WorkerError("registry helper response schema is invalid")
    if response["ok"]:
        if return_code != 0 or not isinstance(response.get("result"), Mapping):
            raise WorkerError("registry helper success response is inconsistent")
        return _RegistryOperationResult(ok=True, result=dict(response["result"]))
    if return_code == 0 or type(response.get("transient_candidate")) is not bool:
        raise WorkerError("registry helper failure response is inconsistent")
    error_number = response.get("errno")
    if error_number is not None and (
        isinstance(error_number, bool) or not isinstance(error_number, int)
    ):
        raise WorkerError("registry helper errno is invalid")
    values: dict[str, str | None] = {}
    for field in ("error_type", "message", "stage"):
        value = response.get(field)
        if value is not None and not isinstance(value, str):
            raise WorkerError(f"registry helper {field} is invalid")
        values[field] = value
    return _RegistryOperationResult(
        ok=False,
        transient_candidate=response["transient_candidate"],
        error_type=values["error_type"],
        message=values["message"],
        stage=values["stage"],
        errno=error_number,
    )


def _registry_failure_text(result: _RegistryOperationResult) -> str:
    return (
        f"{result.error_type or 'RegistryOperationError'} at "
        f"{result.stage or 'unknown'} (errno={result.errno}): "
        f"{result.message or 'no detail'}"
    )


def _run_registry_operation_until(
    request: Mapping[str, Any],
    *,
    deadline_monotonic: float,
    helper_owner: _RegistryHelperOwner,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    start_fn: Callable[
        [Mapping[str, Any], float, _RegistryHelperOwner],
        _PendingRegistryOperation,
    ] = _start_registry_operation,
    poll_fn: Callable[
        [_PendingRegistryOperation, float], _RegistryOperationResult | None
    ] = _poll_registry_operation,
) -> Mapping[str, Any]:
    retry_index = 0
    while True:
        helper_owner.require_idle()
        now = now_fn()
        if now >= deadline_monotonic:
            raise WorkerError("registry operation exhausted its absolute deadline")
        pending = start_fn(request, now, helper_owner)
        if helper_owner.pending is not pending:
            raise WorkerError("registry helper was not adopted by its outer owner")
        result: _RegistryOperationResult | None = None
        reported_timeout: _RegistryOperationResult | None = None
        try:
            while result is None:
                now = now_fn()
                if now >= deadline_monotonic:
                    _signal_pending_registry_helper(pending)
                    raise WorkerError(
                        "registry operation exhausted its absolute deadline"
                    )
                result = poll_fn(pending, now)
                if result is not None and result.retain_pending:
                    reported_timeout = result
                    result = None
                elif result is not None and result.failure_already_reported:
                    result = reported_timeout or result
                if result is None:
                    sleep_fn(min(0.1, max(0.0, deadline_monotonic - now)))
        except BaseException:
            if pending.process.poll() is None:
                _signal_pending_registry_helper(pending)
            else:
                helper_owner.release(pending)
            raise
        helper_owner.release(pending)
        if result.ok:
            if result.result is None:
                raise WorkerError("registry helper omitted its result")
            return result.result
        if not result.transient_candidate:
            raise WorkerError(
                "hard registry failure: " + _registry_failure_text(result)
            )
        delay = _REGISTRY_RETRY_DELAYS_SECONDS[
            min(retry_index, len(_REGISTRY_RETRY_DELAYS_SECONDS) - 1)
        ]
        retry_index += 1
        remaining = deadline_monotonic - now_fn()
        if remaining <= 0:
            raise WorkerError(
                "transient registry failure exhausted its absolute deadline: "
                + _registry_failure_text(result)
            )
        sleep_fn(min(delay, remaining))


def _require_independent_incident_overlay(
    overlay_path: Path,
    registry_root: Path,
) -> None:
    try:
        overlay_device = os.stat(overlay_path).st_dev
        registry_device = os.stat(registry_root).st_dev
    except OSError as exc:
        raise WorkerError(f"could not preflight incident filesystems: {exc}") from exc
    if overlay_device == registry_device:
        raise WorkerError(
            "emergency incident overlay is not independent of the registry filesystem"
        )


def _write_emergency_incident(
    overlay_path: Path,
    registry_root: Path,
    payload: Mapping[str, Any],
) -> Mapping[str, str]:
    del registry_root
    temporary: Path | None = None
    descriptor = -1
    try:
        directory = overlay_path / "dcdn-supervisor-incidents"
        directory.mkdir(mode=0o750, parents=True, exist_ok=True)
        data = (
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
        ).encode("ascii")
        digest = hashlib.sha256(data).hexdigest()
        target = directory / f"{payload['run_id']}-{digest}.json"
        if target.exists():
            if target.read_bytes() != data:
                raise WorkerError("emergency incident hash-path collision")
        else:
            descriptor, raw_temporary = tempfile.mkstemp(
                prefix=f".{target.name}.{os.getpid()}-",
                suffix=".tmp",
                dir=directory,
            )
            temporary = Path(raw_temporary)
            os.fchmod(descriptor, 0o640)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, target)
            directory_descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        if target.read_bytes() != data:
            raise WorkerError("emergency incident reopen verification failed")
    except OSError as exc:
        raise WorkerError(f"could not persist emergency incident: {exc}") from exc
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
    return {"path": str(target), "sha256": digest}


class _HeartbeatController:
    def __init__(
        self,
        *,
        registry_root: Path,
        overlay_path: Path,
        run_id: str,
        pid: int,
        now: float,
        helper_owner: _RegistryHelperOwner | None = None,
        start_fn: Callable[
            [Mapping[str, Any], float, _RegistryHelperOwner],
            _PendingRegistryOperation,
        ] = _start_registry_operation,
        poll_fn: Callable[
            [_PendingRegistryOperation, float], _RegistryOperationResult | None
        ] = _poll_registry_operation,
        emergency_writer: Callable[
            [Path, Path, Mapping[str, Any]], Mapping[str, str]
        ] = _write_emergency_incident,
    ) -> None:
        self.registry_root = registry_root
        self.overlay_path = overlay_path
        self.run_id = run_id
        self.pid = pid
        self._helper_owner = helper_owner or _RegistryHelperOwner()
        self._start_fn = start_fn
        self._poll_fn = poll_fn
        self._emergency_writer = emergency_writer
        self._pending: _PendingRegistryOperation | None = None
        self._next_due = now + _HEARTBEAT_INTERVAL_SECONDS
        self._degraded_since: float | None = None
        self._retry_index = 0
        self._incident: Mapping[str, Any] | None = None
        self._local_receipt: Mapping[str, str] | None = None
        self._persistent_receipts: list[Mapping[str, str]] = []
        self._incident_sequence = 0

    @property
    def pending_incident(self) -> Mapping[str, Any] | None:
        return self._incident

    @property
    def persistent_receipts(self) -> tuple[Mapping[str, str], ...]:
        return tuple(self._persistent_receipts)

    @property
    def local_receipt(self) -> Mapping[str, str] | None:
        return self._local_receipt

    def _heartbeat_request(self) -> Mapping[str, Any]:
        request: dict[str, Any] = {
            "operation": "heartbeat",
            "registry_root": str(self.registry_root),
            "run_id": self.run_id,
            "pid": self.pid,
        }
        if self._incident is not None:
            request["incident"] = dict(self._incident)
        return request

    def _begin_incident(
        self,
        result: _RegistryOperationResult,
        *,
        started_at: float,
    ) -> None:
        self._incident_sequence += 1
        incident_id = (
            f"registry-io-{self.pid}-{time.time_ns()}-{self._incident_sequence}"
        )
        self._incident = {
            "schema_version": 1,
            "artifact_role": "run_registry_io_incident",
            "incident_id": incident_id,
            "run_id": self.run_id,
            "supervisor_pid": os.getpid(),
            "trainer_pid": self.pid,
            "first_failure_utc": datetime.now(UTC)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "first_failure_monotonic": started_at,
            "operation": "heartbeat",
            "error_type": result.error_type,
            "error_stage": result.stage,
            "errno": result.errno,
            "detail": (result.message or "no detail")[:500],
            "heartbeat_interval_seconds": _HEARTBEAT_INTERVAL_SECONDS,
            "operation_timeout_seconds": _REGISTRY_OPERATION_TIMEOUT_SECONDS,
            "degraded_deadline_seconds": _REGISTRY_DEGRADED_DEADLINE_SECONDS,
            "receipt_scope": "supervisor_io_evidence_not_registry_state",
        }
        self._local_receipt = self._emergency_writer(
            self.overlay_path,
            self.registry_root,
            self._incident,
        )

    def _record_success(
        self,
        result: _RegistryOperationResult,
        *,
        now: float,
    ) -> str | None:
        if self._incident is not None:
            incident_receipt = (
                result.result.get("incident") if result.result is not None else None
            )
            if not isinstance(incident_receipt, Mapping):
                return "registry recovered without a persistent incident receipt"
            digest = incident_receipt.get("sha256")
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                return "persistent incident receipt digest is invalid"
            if (
                self._local_receipt is None
                or self._local_receipt.get("sha256") != digest
            ):
                return (
                    "persistent incident receipt does not match the fixed "
                    "local incident digest"
                )
            self._persistent_receipts.append(dict(incident_receipt))
        self._degraded_since = None
        self._retry_index = 0
        self._incident = None
        self._local_receipt = None
        self._next_due = now + _HEARTBEAT_INTERVAL_SECONDS
        return None

    def _record_failure(
        self,
        result: _RegistryOperationResult,
        *,
        started_at: float,
        now: float,
    ) -> str | None:
        if not result.transient_candidate:
            return "hard registry heartbeat failure: " + _registry_failure_text(result)
        if self._degraded_since is None:
            self._degraded_since = started_at
            try:
                self._begin_incident(result, started_at=started_at)
            except WorkerError as exc:
                return str(exc)
        assert self._degraded_since is not None
        elapsed = now - self._degraded_since
        if elapsed >= _REGISTRY_DEGRADED_DEADLINE_SECONDS:
            return (
                "transient registry heartbeat failures exceeded the 180-second "
                "degraded deadline: " + _registry_failure_text(result)
            )
        delay = _REGISTRY_RETRY_DELAYS_SECONDS[
            min(self._retry_index, len(_REGISTRY_RETRY_DELAYS_SECONDS) - 1)
        ]
        self._retry_index += 1
        self._next_due = min(
            now + delay,
            self._degraded_since + _REGISTRY_DEGRADED_DEADLINE_SECONDS,
        )
        return None

    def tick(self, now: float) -> str | None:
        if self._pending is not None:
            pending = self._pending
            if self._degraded_since is not None and (
                now - self._degraded_since >= _REGISTRY_DEGRADED_DEADLINE_SECONDS
            ):
                return (
                    "registry heartbeat degraded deadline elapsed with an "
                    "unreaped timed-out helper"
                )
            try:
                result = self._poll_fn(pending, now)
            except WorkerError as exc:
                if pending.process.poll() is not None:
                    self._helper_owner.release(pending)
                    self._pending = None
                return f"registry helper protocol failure: {exc}"
            if result is None:
                return None
            if result.retain_pending:
                return self._record_failure(
                    result,
                    started_at=pending.started_at,
                    now=now,
                )
            self._helper_owner.release(pending)
            if result.failure_already_reported:
                self._pending = None
                self._next_due = now
                return None
            self._pending = None
            if result.ok:
                return self._record_success(result, now=now)
            return self._record_failure(
                result,
                started_at=pending.started_at,
                now=now,
            )
        if self._degraded_since is not None and (
            now - self._degraded_since >= _REGISTRY_DEGRADED_DEADLINE_SECONDS
        ):
            return "registry heartbeat degraded deadline elapsed before retry"
        if now >= self._next_due:
            try:
                pending = self._start_fn(
                    self._heartbeat_request(),
                    now,
                    self._helper_owner,
                )
                if self._helper_owner.pending is not pending:
                    raise WorkerError(
                        "registry helper was not adopted by its outer owner"
                    )
                self._pending = pending
            except WorkerError as exc:
                return f"could not start registry heartbeat helper: {exc}"
        return None

    def close(
        self,
        *,
        deadline_monotonic: float | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if self._pending is None:
            return
        pending = self._pending
        deadline = now_fn() + 2.0 if deadline_monotonic is None else deadline_monotonic
        if self._helper_owner.pending is not pending:
            raise WorkerError("heartbeat lost registry helper ownership")
        self._helper_owner.close(
            deadline_monotonic=deadline,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        self._pending = None


def _stop_child(
    process: subprocess.Popen[bytes], *, grace_seconds: float = 20.0
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if process.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _guard(
    *,
    project_root: Path,
    overlay_path: Path,
    eta_hours: Decimal,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    cost = evaluate_cost(
        datetime.now(UTC),
        eta_hours=eta_hours,
        phase="compute",
        work_class="mandatory",
    ).as_dict()
    storage = inspect_storage(project_root, overlay_path)
    if cost["launch_allowed"] is not True:
        raise WorkerError(
            "cost/time guard blocked worker: " + "; ".join(cost["reasons"])
        )
    if storage["ok"] is not True:
        raise WorkerError(
            "storage guard blocked worker: " + "; ".join(storage["violations"])
        )
    return cost, storage


def _worker_environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["WORLD_SIZE"] = "1"
    for name, value in _TELEMETRY_ENV.items():
        environment[name] = value
    return environment


def _trainer_command(resolved_config: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dcdn.supervised_exec",
        "--expected-parent-pid",
        str(os.getpid()),
        "--",
        sys.executable,
        str(_TRAIN_SCRIPT),
        "run",
        "--resolved",
        str(resolved_config),
    ]


def _require_code_identity(project_root: Path, record: Mapping[str, Any]) -> None:
    """Bind execution to HEAD and the canonical tracked binary diff bytes."""

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise WorkerError(
                "git provenance command failed: "
                + completed.stderr.decode("utf-8", errors="replace")[:500]
            )
        return completed.stdout

    try:
        commit = git("rev-parse", "--verify", "HEAD").decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise WorkerError("git HEAD is not ASCII") from exc
    if commit != record["code_commit"]:
        raise WorkerError("registry code_commit differs from checkout HEAD")
    diff = git("diff", "--binary", "HEAD", "--")
    if hashlib.sha256(diff).hexdigest() != record["code_diff_sha256"]:
        raise WorkerError("registry code_diff_sha256 differs from checkout")
    untracked = git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "src",
        "scripts",
    )
    for raw_path in untracked.split(b"\0"):
        if not raw_path:
            continue
        try:
            relative = Path(raw_path.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise WorkerError("untracked code path is not UTF-8") from exc
        candidate = project_root / relative
        if candidate.suffix in {".py", ".pyi", ".pyc", ".so", ".sh"} or (
            candidate.exists() and os.access(candidate, os.X_OK)
        ):
            raise WorkerError(
                f"untracked executable/importable source is forbidden: {relative}"
            )


def reconcile_checkpoint_size_failure(
    *,
    project_root: Path,
    registry_root: Path,
    run_id: str,
    gpu: int,
    resolved_config: Path,
) -> int:
    """Re-verify one immutable result rejected only by the old 64 MiB gate."""

    if project_root != PROJECT_ROOT:
        raise WorkerError("project_root does not match this checkout")
    if gpu not in (0, 1):
        raise WorkerError("gpu must be physical index zero or one")
    registry_root = _inside(registry_root, project_root, "registry_root")
    resolved_config = _inside(resolved_config, project_root, "resolved_config")
    registry = RunRegistry(registry_root)
    record = registry.read(run_id)
    if (
        record.get("status") != "failed"
        or record.get("scientific_result_eligible") is not None
        or record.get("completion_or_failure_reason")
        != CHECKPOINT_SIZE_VERIFIER_FAILURE_REASON
        or record.get("execution_kind") != "scientific_tuning"
        or record.get("gpu") != gpu
    ):
        raise WorkerError("run is not the known checkpoint-size verifier failure")
    expected_config_path = resolved_config.relative_to(project_root).as_posix()
    if record.get("resolved_config_path") != expected_config_path:
        raise WorkerError("reconciliation resolved-config path mismatch")
    recorded_pid = record.get("pid")
    if not isinstance(recorded_pid, int) or recorded_pid <= 0:
        raise WorkerError("failed record does not preserve the trainer PID")
    try:
        os.kill(recorded_pid, 0)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise WorkerError("failed trainer PID ownership cannot be verified") from exc
    else:
        raise WorkerError("failed trainer PID is still alive or was reused")

    resolved_artifact = load_verified_yaml_artifact(resolved_config)
    if resolved_artifact.digest != record.get("resolved_config_sha256"):
        raise WorkerError("reconciliation resolved-config digest mismatch")
    resolved = resolved_artifact.payload
    if (
        resolved.get("run_id") != run_id
        or _validate_resolved_execution_contract(resolved) != "scientific_tuning"
    ):
        raise WorkerError("reconciliation resolved execution identity mismatch")

    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if head.returncode != 0:
        raise WorkerError("reconciliation could not resolve verifier HEAD")
    try:
        verifier_commit = head.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise WorkerError("reconciliation verifier HEAD is not ASCII") from exc
    _require_code_identity(
        project_root,
        {
            "code_commit": verifier_commit,
            "code_diff_sha256": EMPTY_DIFF_SHA256,
        },
    )

    result_digest, scientific_result_eligible = _verify_completed_result(
        project_root=project_root,
        resolved_config=resolved_config,
        expected_resolved_digest=resolved_artifact.digest,
        expected_resolved=resolved,
        run_id=run_id,
    )
    if scientific_result_eligible is not True:
        raise WorkerError("reconciliation result is not scientifically eligible")
    run_directory = _inside(
        _resolved_path(
            resolved.get("run_directory"),
            root=project_root,
            field="resolved.run_directory",
        ),
        project_root,
        "resolved.run_directory",
    )
    try:
        last_size = (run_directory / "last.pt").stat().st_size
    except OSError as exc:
        raise WorkerError("reconciliation last checkpoint is unavailable") from exc
    if last_size <= 64 * 1024 * 1024:
        raise WorkerError("failed run does not exhibit the repaired size condition")

    _require_code_identity(
        project_root,
        {
            "code_commit": verifier_commit,
            "code_diff_sha256": EMPTY_DIFF_SHA256,
        },
    )
    failed_record_path = registry.records / f"{run_id}.json"
    try:
        failed_record_digest = hashlib.sha256(
            failed_record_path.read_bytes()
        ).hexdigest()
    except OSError as exc:
        raise WorkerError("failed registry record could not be hashed") from exc
    reconciled = registry.reconcile_checkpoint_size_verifier_failure(
        run_id,
        expected_failed_record_sha256=failed_record_digest,
        verifier_code_commit=verifier_commit,
        verifier_code_diff_sha256=EMPTY_DIFF_SHA256,
        result_sha256=result_digest,
    )
    if (
        reconciled.get("status") != "completed"
        or reconciled.get("scientific_result_eligible") is not True
        or reconciled.get("terminal_reconciliation", {}).get("result_sha256")
        != result_digest
    ):
        raise WorkerError("reconciled registry record failed exact reopen")
    verified_again, eligible_again = _verify_completed_result(
        project_root=project_root,
        resolved_config=resolved_config,
        expected_resolved_digest=resolved_artifact.digest,
        expected_resolved=resolved,
        run_id=run_id,
    )
    if verified_again != result_digest or eligible_again is not True:
        raise WorkerError("post-reconciliation result verification drifted")
    return 0


def run_worker(
    *,
    project_root: Path,
    registry_root: Path,
    run_id: str,
    gpu: int,
    resolved_config: Path,
    poll_seconds: float,
) -> int:
    if project_root != PROJECT_ROOT:
        raise WorkerError("project_root does not match this checkout")
    if gpu not in (0, 1):
        raise WorkerError("gpu must be physical index zero or one")
    if not poll_seconds > 0:
        raise WorkerError("poll_seconds must be positive")
    registry_root = _inside(registry_root, project_root, "registry_root")
    resolved_config = _inside(resolved_config, project_root, "resolved_config")
    resolved_artifact = load_verified_yaml_artifact(resolved_config)
    resolved_config_digest = resolved_artifact.digest
    resolved = resolved_artifact.payload
    if resolved.get("run_id") != run_id:
        raise WorkerError("resolved run_id does not match worker run_id")
    if (
        _resolved_path(
            resolved.get("project_root"),
            root=project_root,
            field="resolved.project_root",
        )
        != project_root
    ):
        raise WorkerError("resolved project_root mismatch")
    execution_kind = _validate_resolved_execution_contract(resolved)
    runtime = _mapping(resolved.get("runtime"), "resolved.runtime")
    eta_hours = Decimal(str(runtime.get("eta_hours")))
    if eta_hours < 0:
        raise WorkerError("runtime eta_hours must be non-negative")
    overlay_path = _resolved_path(
        resolved.get("overlay_path"), root=project_root, field="overlay_path"
    )

    registry = RunRegistry(registry_root)
    record = registry.read(run_id)
    if record["status"] != "pending":
        raise WorkerError("worker may claim only a pending run")
    _require_registry_execution_kind(record, execution_kind)
    if record["resolved_config_sha256"] != resolved_config_digest:
        raise WorkerError("registry resolved_config_sha256 mismatch")
    resolved_training = _mapping(resolved.get("training"), "resolved.training")
    if record["seed"] != resolved_training.get("seed"):
        raise WorkerError("registry seed differs from resolved training seed")
    protocol_path = _inside(
        _resolved_path(
            resolved.get("protocol_path"),
            root=project_root,
            field="resolved.protocol_path",
        ),
        project_root,
        "resolved.protocol_path",
    )
    protocol_artifact = load_verified_yaml_artifact(protocol_path)
    protocol_digest = protocol_artifact.digest
    protocol_payload = protocol_artifact.payload
    protocol_hashes = _mapping(protocol_payload.get("hashes"), "protocol.hashes")
    if record["protocol_sha256"] != protocol_digest:
        raise WorkerError("registry protocol_sha256 mismatch")
    if record["data_manifest_sha256"] != protocol_hashes.get("data_manifest_sha256"):
        raise WorkerError("registry data manifest digest mismatch")
    if record["features_sha256"] != protocol_hashes.get("features_sha256"):
        raise WorkerError("registry features digest mismatch")
    recorded_resolved = _inside(
        _resolved_path(
            record["resolved_config_path"],
            root=project_root,
            field="record.resolved_config_path",
        ),
        project_root,
        "record.resolved_config_path",
    )
    if recorded_resolved != resolved_config:
        raise WorkerError("registry resolved_config_path mismatch")
    _verify_capacity_binding(resolved, project_root, record=record)
    log_path = _inside(
        _resolved_path(record["log_path"], root=project_root, field="record.log_path"),
        project_root,
        "record.log_path",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    active_on_gpu = [
        item["run_id"]
        for item in registry.active_runs()
        if item.get("gpu") == gpu and item["run_id"] != run_id
    ]
    if active_on_gpu:
        raise WorkerError(f"GPU {gpu} already owns active runs: {active_on_gpu}")

    lock_path = registry.locks / f"physical-gpu-{gpu}.worker.lock"
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o640)
    helper_owner = _RegistryHelperOwner()
    process: subprocess.Popen[bytes] | None = None
    heartbeat: _HeartbeatController | None = None
    terminal_intent: dict[str, Any] | None = None
    terminal_attempted = False
    terminal_published = False
    terminal_deadline: float | None = None
    stop_signal: int | None = None
    previous_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        nonlocal stop_signal
        stop_signal = signum

    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkerError(f"GPU {gpu} worker lock is already held") from exc

        previous_handlers = {
            signum: signal.signal(signum, request_stop)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }

        _guard(
            project_root=project_root,
            overlay_path=overlay_path,
            eta_hours=eta_hours,
        )
        _require_independent_incident_overlay(
            overlay_path,
            registry_root,
        )
        _require_code_identity(project_root, record)
        command = _trainer_command(resolved_config)
        environment = _worker_environment(gpu)
        with log_path.open("ab", buffering=0) as log_handle:
            if stop_signal is not None:
                raise WorkerError(
                    f"worker received signal {stop_signal} before trainer spawn"
                )
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                _run_registry_operation_until(
                    {
                        "operation": "ensure_running",
                        "registry_root": str(registry_root),
                        "run_id": run_id,
                        "pid": process.pid,
                        "gpu": gpu,
                    },
                    deadline_monotonic=(
                        time.monotonic() + _RUNNING_PUBLISH_DEADLINE_SECONDS
                    ),
                    helper_owner=helper_owner,
                )
            except BaseException:
                _stop_child(process)
                raise
            heartbeat = _HeartbeatController(
                registry_root=registry_root,
                overlay_path=overlay_path,
                run_id=run_id,
                pid=process.pid,
                now=time.monotonic(),
                helper_owner=helper_owner,
            )

            failure_reason: str | None = None
            try:
                last_guard = time.monotonic()
                while process.poll() is None:
                    if stop_signal is not None:
                        failure_reason = f"worker received signal {stop_signal}"
                        _stop_child(process)
                        break
                    now = time.monotonic()
                    heartbeat_failure = heartbeat.tick(now)
                    if heartbeat_failure is not None:
                        failure_reason = heartbeat_failure
                        _stop_child(process)
                        break
                    if now - last_guard >= 30.0:
                        try:
                            _guard(
                                project_root=project_root,
                                overlay_path=overlay_path,
                                eta_hours=Decimal(0),
                            )
                        except WorkerError as exc:
                            failure_reason = str(exc)
                            _stop_child(process)
                            break
                        last_guard = now
                    time.sleep(min(poll_seconds, 5.0))
            except BaseException as exc:
                failure_reason = (
                    f"worker supervisor failure: {type(exc).__name__}: {exc}"
                )
                _stop_child(process)
            try:
                if terminal_deadline is None:
                    terminal_deadline = (
                        time.monotonic() + _TERMINAL_PUBLISH_DEADLINE_SECONDS
                    )
                heartbeat.close(deadline_monotonic=terminal_deadline)
            except BaseException as exc:
                if failure_reason is None:
                    failure_reason = (
                        f"registry helper cleanup failure: {type(exc).__name__}: {exc}"
                    )
                _stop_child(process)
            return_code = process.wait()
            if failure_reason is None and return_code == 0:
                try:
                    post_run_record = registry.read(run_id)
                    if post_run_record.get("status") != "running":
                        raise WorkerError(
                            "registry status drifted before terminal verification"
                        )
                    for identity_field in (
                        "resolved_config_sha256",
                        "protocol_sha256",
                        "code_commit",
                        "code_diff_sha256",
                    ):
                        if post_run_record.get(identity_field) != record.get(
                            identity_field
                        ):
                            raise WorkerError(
                                f"registry identity drifted: {identity_field}"
                            )
                    _require_code_identity(project_root, post_run_record)
                    _verify_capacity_binding(
                        resolved, project_root, record=post_run_record
                    )
                    (
                        result_digest,
                        scientific_result_eligible,
                    ) = _verify_completed_result(
                        project_root=project_root,
                        resolved_config=resolved_config,
                        expected_resolved_digest=resolved_config_digest,
                        expected_resolved=resolved,
                        run_id=run_id,
                    )
                except BaseException as exc:
                    failure_reason = (
                        "trainer result verification failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    terminal_intent = {
                        "operation": "ensure_terminal",
                        "registry_root": str(registry_root),
                        "run_id": run_id,
                        "status": "completed",
                        "pid": process.pid,
                        "gpu": gpu,
                        "reason": _completed_reason(
                            execution_kind,
                            result_digest,
                        ),
                        "scientific_result_eligible": scientific_result_eligible,
                    }
                    if heartbeat.pending_incident is not None:
                        terminal_intent["incident"] = dict(heartbeat.pending_incident)
                    if terminal_deadline is None:
                        terminal_deadline = (
                            time.monotonic() + _TERMINAL_PUBLISH_DEADLINE_SECONDS
                        )
                    terminal_attempted = True
                    terminal_result = _run_registry_operation_until(
                        terminal_intent,
                        deadline_monotonic=terminal_deadline,
                        helper_owner=helper_owner,
                    )
                    if terminal_result.get("status") != "completed":
                        raise WorkerError(
                            "terminal helper did not reconcile completed state"
                        )
                    completed_record = registry.read(run_id)
                    if completed_record.get("status") != "completed":
                        raise WorkerError("fresh registry record is not completed")
                    for identity_field in (
                        "resolved_config_sha256",
                        "protocol_sha256",
                        "code_commit",
                        "code_diff_sha256",
                    ):
                        if completed_record.get(identity_field) != record.get(
                            identity_field
                        ):
                            raise WorkerError(
                                f"completed registry identity drifted: {identity_field}"
                            )
                    _require_code_identity(project_root, completed_record)
                    _verify_capacity_binding(
                        resolved,
                        project_root,
                        record=completed_record,
                    )
                    terminal_published = True
                    return 0
            reason = failure_reason or f"trainer exited with return code {return_code}"
            terminal_intent = {
                "operation": "ensure_terminal",
                "registry_root": str(registry_root),
                "run_id": run_id,
                "status": "failed",
                "pid": process.pid,
                "gpu": gpu,
                "reason": reason[:1000],
                "scientific_result_eligible": None,
            }
            if heartbeat.pending_incident is not None:
                terminal_intent["incident"] = dict(heartbeat.pending_incident)
            if terminal_deadline is None:
                terminal_deadline = (
                    time.monotonic() + _TERMINAL_PUBLISH_DEADLINE_SECONDS
                )
            terminal_attempted = True
            terminal_result = _run_registry_operation_until(
                terminal_intent,
                deadline_monotonic=terminal_deadline,
                helper_owner=helper_owner,
            )
            if terminal_result.get("status") != "failed":
                raise WorkerError("terminal helper did not reconcile failed state")
            terminal_published = True
            return 2
    finally:
        trainer_cleanup_error: BaseException | None = None
        helper_cleanup_error: BaseException | None = None
        try:
            if process is not None and process.poll() is None:
                try:
                    _stop_child(process)
                except BaseException as exc:
                    trainer_cleanup_error = exc
            if heartbeat is not None:
                try:
                    if terminal_deadline is None:
                        terminal_deadline = (
                            time.monotonic() + _TERMINAL_PUBLISH_DEADLINE_SECONDS
                        )
                    heartbeat.close(deadline_monotonic=terminal_deadline)
                except BaseException as exc:
                    helper_cleanup_error = exc
            if (
                process is not None
                and process.poll() is not None
                and terminal_intent is None
            ):
                terminal_intent = {
                    "operation": "ensure_terminal",
                    "registry_root": str(registry_root),
                    "run_id": run_id,
                    "status": "failed",
                    "pid": process.pid,
                    "gpu": gpu,
                    "reason": ("worker owner exited before a terminal transition"),
                    "scientific_result_eligible": None,
                }
                if heartbeat is not None and heartbeat.pending_incident is not None:
                    terminal_intent["incident"] = dict(heartbeat.pending_incident)
            if (
                terminal_intent is not None
                and not terminal_published
                and not terminal_attempted
            ):
                if terminal_deadline is None:
                    terminal_deadline = (
                        time.monotonic() + _TERMINAL_PUBLISH_DEADLINE_SECONDS
                    )
                terminal_attempted = True
                terminal_result = _run_registry_operation_until(
                    terminal_intent,
                    deadline_monotonic=terminal_deadline,
                    helper_owner=helper_owner,
                )
                if terminal_result.get("status") != terminal_intent["status"]:
                    raise WorkerError("terminal helper did not reconcile exact intent")
                terminal_published = True
            if helper_cleanup_error is not None:
                raise WorkerError(
                    "registry helper could not be reaped cleanly: "
                    f"{type(helper_cleanup_error).__name__}: "
                    f"{helper_cleanup_error}"
                ) from helper_cleanup_error
            if trainer_cleanup_error is not None:
                raise WorkerError(
                    "trainer could not be reaped cleanly; terminal state was "
                    "not published: "
                    f"{type(trainer_cleanup_error).__name__}: "
                    f"{trainer_cleanup_error}"
                ) from trainer_cleanup_error
        finally:
            try:
                if helper_owner.pending is not None:
                    cleanup_deadline = max(
                        terminal_deadline or 0.0,
                        time.monotonic() + 2.0,
                    )
                    helper_owner.close(
                        deadline_monotonic=cleanup_deadline,
                    )
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
                try:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(lock_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--resolved", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--reconcile-checkpoint-size-failure",
        action="store_true",
        help="re-verify and reconcile only the known legacy 64 MiB verifier failure",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve(strict=True)
    registry_root = args.registry_root.resolve(strict=True)
    resolved_config = args.resolved.resolve(strict=True)
    if args.reconcile_checkpoint_size_failure:
        result = reconcile_checkpoint_size_failure(
            project_root=project_root,
            registry_root=registry_root,
            run_id=args.run_id,
            gpu=args.gpu,
            resolved_config=resolved_config,
        )
        mode = "checkpoint_size_failure_reconciliation"
    else:
        result = run_worker(
            project_root=project_root,
            registry_root=registry_root,
            run_id=args.run_id,
            gpu=args.gpu,
            resolved_config=resolved_config,
            poll_seconds=args.poll_seconds,
        )
        mode = "worker"
    print(
        json.dumps(
            {
                "ok": result == 0,
                "run_id": args.run_id,
                "gpu": args.gpu,
                "mode": mode,
            }
        )
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
