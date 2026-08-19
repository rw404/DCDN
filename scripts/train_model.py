#!/usr/bin/env python3
"""Fail-closed single-process VK trainer for frozen DCDN experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch
from cost_guard import evaluate_cost
from storage_guard import inspect_storage

from dcdn.capacity_artifact import (
    load_verified_capacity_artifact,
)
from dcdn.capacity_artifact import (
    sha256_file as capacity_sha256_file,
)
from dcdn.data.vk_lsvd_runtime import (
    C_PLUS_P,
    C_PLUS_P_PLUS_E,
    build_vk_lsvd_raw_source,
)
from dcdn.encoder import EncoderContractLock, build_trainable_encoded_model
from dcdn.models import DCNv2CapacityCoreConfig
from dcdn.protocol import (
    ProtocolGate,
    load_verified_yaml,
    read_verified_artifact,
    verify_sha256_sidecar,
)
from dcdn.raw import RawSourceIdentity
from dcdn.search_space import SearchSpaceLock
from dcdn.selection_artifact import (
    CAPACITY_CANDIDATE_ID,
    CAPACITY_LOW_RANK,
    CAPACITY_RESOLUTION_SHA256,
    ORDINARY_SELECTION_SHA256,
    PREMETRIC_PROTOCOL_SHA256,
    SELECTED_LEARNING_RATE,
    CapacityBindingEvidence,
    capacity_phase_resolutions,
    capacity_resolved_model_config_sha256,
    capacity_selection_binding_payload,
    capacity_training_binding_payload,
    load_capacity_lr_binding,
)
from dcdn.training import (
    CompactEventLog,
    SafetyAttestation,
    TrainingConfig,
    TrainingContractError,
    TrainingHarness,
    TrainingResult,
    resolved_model_components,
    runtime_identity_from_raw,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATASET_ID = "vk_lsvd_up0.01_ir0.01"
_TELEMETRY_ENV = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_DATASETS_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingContractError(f"{field} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise TrainingContractError(f"{field} keys mismatch")


def _path(value: Any, *, root: Path, field: str, inside: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TrainingContractError(f"{field} must be a non-empty path")
    result = Path(value)
    if not result.is_absolute():
        result = root / result
    result = result.resolve()
    if inside and not result.is_relative_to(root):
        raise TrainingContractError(f"{field} must remain inside project_root")
    return result


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Publish an immutable JSON/sidecar pair or leave neither new path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("ascii")
    digest = hashlib.sha256(data).hexdigest()
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar_data = f"{digest}  {path.name}\n".encode("ascii")
    result_fd = -1
    sidecar_fd = -1
    result_temporary: Path | None = None
    sidecar_temporary: Path | None = None
    result_published = False
    sidecar_published = False
    publication_complete = False

    def fsync_directory() -> None:
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    try:
        result_fd, result_temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        result_temporary = Path(result_temporary_name)
        sidecar_fd, sidecar_temporary_name = tempfile.mkstemp(
            prefix=f".{sidecar.name}.", suffix=".tmp", dir=path.parent
        )
        sidecar_temporary = Path(sidecar_temporary_name)
        os.fchmod(result_fd, 0o640)
        os.fchmod(sidecar_fd, 0o640)
        with os.fdopen(result_fd, "wb", closefd=True) as handle:
            result_fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        with os.fdopen(sidecar_fd, "wb", closefd=True) as handle:
            sidecar_fd = -1
            handle.write(sidecar_data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(result_temporary, path)
        result_published = True
        os.link(sidecar_temporary, sidecar)
        sidecar_published = True
        fsync_directory()
        publication_complete = True
    except FileExistsError as exc:
        raise TrainingContractError(f"immutable output exists: {path}") from exc
    except OSError as exc:
        raise TrainingContractError(
            f"immutable output pair publication failed: {path}"
        ) from exc
    finally:
        if sidecar_published and not publication_complete:
            with suppress(FileNotFoundError):
                sidecar.unlink()
        if result_published and not publication_complete:
            with suppress(FileNotFoundError):
                path.unlink()
        if result_fd >= 0:
            os.close(result_fd)
        if sidecar_fd >= 0:
            os.close(sidecar_fd)
        for temporary in (result_temporary, sidecar_temporary):
            if temporary is None:
                continue
            with suppress(FileNotFoundError):
                temporary.unlink()
        try:
            fsync_directory()
        except OSError:
            if publication_complete:
                for target in (sidecar, path):
                    with suppress(FileNotFoundError):
                        target.unlink()
            raise
    try:
        artifact = read_verified_artifact(path)
        if artifact.digest != digest or artifact.data != data:
            raise TrainingContractError(
                "published result bytes differ from the immutable receipt"
            )
    except BaseException as exc:
        for target in (sidecar, path):
            with suppress(FileNotFoundError):
                target.unlink()
        fsync_directory()
        if isinstance(exc, TrainingContractError):
            raise
        raise TrainingContractError(
            "published result pair could not be re-opened and verified"
        ) from exc
    return digest


def _disable_telemetry() -> None:
    for name, value in _TELEMETRY_ENV.items():
        os.environ[name] = value


def _reject_distributed_runtime() -> None:
    if os.environ.get("WORLD_SIZE") not in (None, "", "1"):
        raise TrainingContractError("WORLD_SIZE must be one; DDP is forbidden")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise TrainingContractError("initialized distributed runtime is forbidden")


def _device(value: Any) -> torch.device:
    if not isinstance(value, str):
        raise TrainingContractError("runtime.device must be a string")
    device = torch.device(value)
    if device.type == "cuda":
        if device != torch.device("cuda:0"):
            raise TrainingContractError("worker GPU must be addressed as cuda:0")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise TrainingContractError("exactly one GPU must be visible per process")
    elif device.type != "cpu":
        raise TrainingContractError("runtime.device must be cpu or cuda:0")
    return device


def _training_config(payload: Mapping[str, Any]) -> TrainingConfig:
    values = dict(payload)
    if "adam_betas" in values:
        values["adam_betas"] = tuple(values["adam_betas"])
    try:
        return TrainingConfig(**values)
    except (TypeError, ValueError) as exc:
        raise TrainingContractError(f"invalid training config: {exc}") from exc


def _validate_execution_contract(
    execution_kind: Any,
    config: TrainingConfig,
    runtime: Mapping[str, Any],
) -> str:
    """Validate scientific versus non-scientific execution before model/source use."""

    if type(execution_kind) is not str or execution_kind not in {
        "implementation_smoke",
        "scientific_tuning",
    }:
        raise TrainingContractError("execution_kind is invalid")
    if execution_kind == "scientific_tuning":
        if config.runtime_value_checks:
            raise TrainingContractError(
                "scientific tuning requires runtime_value_checks=false"
            )
        return execution_kind

    if config.runtime_value_checks is not True:
        raise TrainingContractError(
            "implementation smoke requires runtime_value_checks=true"
        )
    if config.effective_batch_size != 4096:
        raise TrainingContractError(
            "implementation smoke requires frozen effective_batch_size=4096"
        )
    if runtime.get("resume") is not False:
        raise TrainingContractError("implementation smoke must be a fresh run")
    stop_after_examples = runtime.get("stop_after_examples")
    if (
        type(stop_after_examples) is not int
        or stop_after_examples != config.effective_batch_size
    ):
        raise TrainingContractError(
            "implementation smoke must stop after exactly one effective batch"
        )
    if runtime.get("latency") is not None:
        raise TrainingContractError("implementation smoke forbids latency evaluation")
    if config.validate_every_examples <= config.effective_batch_size:
        raise TrainingContractError(
            "implementation smoke must stop before the first validation threshold"
        )
    return execution_kind


def _validate_result_scope(
    execution_kind: str,
    config: TrainingConfig,
    result: TrainingResult,
) -> bool:
    """Return scientific eligibility after enforcing kind-specific result shape."""

    if execution_kind == "implementation_smoke":
        if (
            result.examples_seen != config.effective_batch_size
            or result.optimizer_steps != 1
            or result.validation_checks != 0
            or result.best_validation_metric is not None
            or result.best_checkpoint_sha256 is not None
            or result.best_checkpoint_filename is not None
            or result.stopped_reason != "requested_checkpoint_boundary"
        ):
            raise TrainingContractError(
                "implementation smoke produced scientific/validation result state"
            )
        return False
    if execution_kind != "scientific_tuning":
        raise TrainingContractError("execution_kind is invalid at result boundary")
    return (
        result.validation_checks > 0
        and result.best_validation_metric is not None
        and result.best_checkpoint_sha256 is not None
        and result.best_checkpoint_filename is not None
    )


class _RuntimeGuard:
    def __init__(
        self,
        root: Path,
        overlay: Path,
        log: CompactEventLog,
        interval_seconds: float,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise TrainingContractError("guard interval must be positive")
        self.root = root
        self.overlay = overlay
        self.log = log
        self.interval = interval_seconds
        self.last = -math.inf
        self.cost: Mapping[str, Any] | None = None
        self.storage: Mapping[str, Any] | None = None

    def check(self, *, eta_hours: Decimal = Decimal(0), force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last < self.interval:
            return
        self.cost = evaluate_cost(
            datetime.now(UTC),
            eta_hours=eta_hours,
            phase="compute",
            work_class="mandatory",
        ).as_dict()
        self.storage = inspect_storage(self.root, self.overlay)
        self.last = now
        self.log.emit(
            "local_guard",
            launch_allowed=self.cost["launch_allowed"],
            storage_ok=self.storage["ok"],
            projected_spend_usd=self.cost["projected_spend_usd"],
        )
        if self.cost["launch_allowed"] is not True:
            raise TrainingContractError("cost/time guard blocked run")
        if self.storage["ok"] is not True:
            raise TrainingContractError("storage guard blocked run")


def _expected_identity(
    gate: ProtocolGate,
    lock: EncoderContractLock,
    split: str,
    *,
    model_key: str,
) -> RawSourceIdentity:
    if gate.dataset_id != _DATASET_ID:
        raise TrainingContractError("CLI supports only the frozen VK dataset")
    if gate.feature_regime not in {C_PLUS_P, C_PLUS_P_PLUS_E}:
        raise TrainingContractError("CLI supports only frozen VK C+P or C+P+E")
    if gate.feature_regime == C_PLUS_P_PLUS_E and (
        model_key != "dcnv2_full_rank"
        or gate.required_model_keys != ("dcnv2_full_rank",)
    ):
        raise TrainingContractError(
            "VK C+P+E is restricted to the isolated dcnv2_full_rank protocol"
        )
    dataset = _mapping(lock.payload["datasets"][_DATASET_ID], "encoder dataset")
    regime = _mapping(
        dataset["regimes"][gate.feature_regime],
        "encoder regime",
    )
    allowed_models = regime.get("allowed_models")
    if not isinstance(allowed_models, list) or any(
        not isinstance(item, str) for item in allowed_models
    ):
        raise TrainingContractError("encoder allowed_models is invalid")
    if allowed_models and model_key not in allowed_models:
        raise TrainingContractError("model is forbidden for the frozen feature regime")
    hashes = gate.payload["hashes"]
    return RawSourceIdentity(
        dataset_id=gate.dataset_id,
        split=split,  # type: ignore[arg-type]
        feature_regime=gate.feature_regime,
        row_count=gate.expected_row_count(split),
        data_manifest_sha256=hashes["data_manifest_sha256"],
        views_manifest_sha256=hashes["views_manifest_sha256"],
        feature_contract_sha256=hashes["features_sha256"],
        physical_schema_sha256=hashes["physical_schema_sha256"],
        expanded_schema_sha256=hashes["expanded_schema_sha256"],
        operand_schema_sha256=hashes["operand_schema_sha256"],
        denominator_mask_sha256=hashes["denominator_mask_sha256"],
        full_materialized_scan_sha256=hashes["full_materialized_scan_sha256"],
        cache_content_sha256=hashes["cache_content_sha256"],
        ordered_context_fields=tuple(dataset["ordered_context_fields"]),
        ordered_operand_fields=tuple(regime["ordered_operand_fields"]),
        categorical_cardinalities=tuple(dataset["categorical_cardinalities"]),
    )


def _load_exact_views_manifest(
    gate: ProtocolGate,
    path: Path,
) -> Mapping[str, Any]:
    expected = gate.payload["hashes"]["views_manifest_sha256"]
    actual = verify_sha256_sidecar(path)
    if actual != expected:
        raise TrainingContractError(
            "views manifest digest differs from the active protocol"
        )
    views = load_verified_yaml(path)
    gate.require_views_split(views)
    return views


def _run(resolved_path: Path) -> int:
    payload = load_verified_yaml(resolved_path)
    base_keys = {
        "schema_version",
        "execution_kind",
        "run_id",
        "project_root",
        "overlay_path",
        "protocol_path",
        "active_protocol_path",
        "views_manifest_path",
        "search_space_path",
        "encoder_contract_path",
        "capacity_artifact_path",
        "full_scan_manifest_path",
        "runtime_cache_root",
        "model",
        "training",
        "leakage_checks",
        "registry_root",
        "run_directory",
        "result_path",
        "runtime",
    }
    training_payload = _mapping(payload.get("training"), "training")
    is_capacity = training_payload.get("model_key") == "dcnv2_capacity_matched_core"
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int:
        raise TrainingContractError("resolved schema_version must be an exact integer")
    if schema_version == 3:
        if is_capacity:
            raise TrainingContractError(
                "schema v3 capacity runs are forbidden; use binding-aware v4"
            )
        keys = base_keys
    elif schema_version == 4:
        if not is_capacity:
            raise TrainingContractError("schema v4 is capacity-control only")
        keys = base_keys | {"capacity_lr_binding_path"}
    else:
        raise TrainingContractError("resolved schema_version must be three or four")
    _exact_keys(payload, keys, "resolved config")
    root = _path(payload["project_root"], root=PROJECT_ROOT, field="root", inside=True)
    if root != PROJECT_ROOT:
        raise TrainingContractError("project_root does not match this checkout")
    overlay = _path(payload["overlay_path"], root=root, field="overlay", inside=False)

    def local(name: str) -> Path:
        return _path(payload[name], root=root, field=name, inside=True)

    training = _training_config(training_payload)
    capacity_binding: CapacityBindingEvidence | None = None
    capacity_binding_path: Path | None = None
    if is_capacity:
        capacity_binding_path = local("capacity_lr_binding_path")
        try:
            capacity_binding = load_capacity_lr_binding(capacity_binding_path, root)
        except Exception as exc:
            raise TrainingContractError(
                f"capacity LR binding failed before model/source creation: {exc}"
            ) from exc
        if (
            training.candidate_id != CAPACITY_CANDIDATE_ID
            or training.learning_rate != SELECTED_LEARNING_RATE
            or training.capacity_resolution_sha256 != CAPACITY_RESOLUTION_SHA256
            or training.ordinary_dcnv2_selection_sha256 != ORDINARY_SELECTION_SHA256
            or training.capacity_lr_binding_sha256 != capacity_binding.sha256
            or capacity_binding.ordinary.learning_rate != SELECTED_LEARNING_RATE
        ):
            raise TrainingContractError(
                "capacity resolved config differs from the compact authorization"
            )

    run_directory = local("run_directory")
    result_path = local("result_path")
    registry_root = local("registry_root")
    registry_root.mkdir(parents=True, exist_ok=True)
    if os.stat(registry_root).st_dev != os.stat(root).st_dev:
        raise TrainingContractError("registry is not on the repository filesystem")

    gate = ProtocolGate.load(
        local("protocol_path"),
        active_registry_path=local("active_protocol_path"),
    )
    if gate.test_unlocked:
        raise TrainingContractError("train CLI refuses final/test protocols")
    gate.require_active()
    views = _load_exact_views_manifest(gate, local("views_manifest_path"))
    search = SearchSpaceLock.load(local("search_space_path"))
    encoder_lock = EncoderContractLock.load(local("encoder_contract_path"))
    capacity_path = local("capacity_artifact_path")
    capacity_digest = capacity_sha256_file(capacity_path)
    verify_sha256_sidecar(capacity_path)

    capacity_payload: Mapping[str, Any] | None = None
    if is_capacity:
        assert capacity_binding is not None
        if gate.digest != PREMETRIC_PROTOCOL_SHA256:
            raise TrainingContractError(
                "capacity control requires the frozen premetric protocol"
            )
        try:
            capacity_payload = load_verified_capacity_artifact(capacity_path)
            resolved_capacity_phase = search.require_phase(
                "capacity_control_training",
                resolutions=capacity_phase_resolutions(
                    capacity_payload,
                    selected_learning_rate=capacity_binding.ordinary.learning_rate,
                ),
            )
        except Exception as exc:
            raise TrainingContractError(
                f"capacity search-phase gate failed before model creation: {exc}"
            ) from exc
        phase_lr = resolved_capacity_phase["models"]["dcnv2_low_rank_mix_capacity"][
            "base_config"
        ]["optimizer"]["learning_rate"]
        if phase_lr != SELECTED_LEARNING_RATE:
            raise TrainingContractError(
                "capacity search phase did not inherit the selected LR"
            )

    if (
        training.dataset_id != _DATASET_ID
        or training.feature_regime != gate.feature_regime
        or training.feature_regime not in {C_PLUS_P, C_PLUS_P_PLUS_E}
        or training.phase != "tuning"
    ):
        raise TrainingContractError("CLI permits only active premetric VK tuning")
    if training.feature_regime == C_PLUS_P_PLUS_E and (
        training.model_key != "dcnv2_full_rank"
        or gate.required_model_keys != ("dcnv2_full_rank",)
    ):
        raise TrainingContractError(
            "VK C+P+E is restricted to the isolated dcnv2_full_rank protocol"
        )
    runtime = _mapping(payload["runtime"], "runtime")
    _exact_keys(
        runtime,
        {
            "device",
            "eta_hours",
            "resume",
            "stop_after_examples",
            "latency",
            "guard_interval_seconds",
        },
        "runtime",
    )
    if type(runtime["resume"]) is not bool:
        raise TrainingContractError("runtime.resume must be boolean")
    execution_kind = _validate_execution_contract(
        payload["execution_kind"],
        training,
        runtime,
    )
    eta = Decimal(str(runtime["eta_hours"]))
    if eta < 0:
        raise TrainingContractError("runtime.eta_hours must be non-negative")
    device = _device(runtime["device"])

    model_payload = _mapping(payload["model"], "model")
    _exact_keys(
        model_payload,
        {"interaction_config", "head_config", "init_seed"},
        "model",
    )
    head_payload = _mapping(model_payload["head_config"], "model.head_config")
    _exact_keys(
        head_payload, {"hidden_dims", "dropout", "output_dim"}, "model.head_config"
    )
    if (
        type(head_payload["hidden_dims"]) is not list
        or head_payload["hidden_dims"] != []
        or type(head_payload["dropout"]) is not float
        or head_payload["dropout"] != 0.0
        or type(head_payload["output_dim"]) is not int
        or head_payload["output_dim"] != 1
    ):
        raise TrainingContractError("model does not use the exact frozen common head")
    # selected_configs.lock has one cell per model, not per final seed.  Its
    # digest therefore covers only seed-invariant architecture/head choices;
    # init_seed remains exact in TrainingConfig and the joint checkpoint.
    selected_model_payload = {
        "interaction_config": model_payload["interaction_config"],
        "head_config": model_payload["head_config"],
    }
    if is_capacity:
        assert capacity_binding is not None
        selection_binding = capacity_selection_binding_payload(
            capacity_candidate_id=training.candidate_id,
            learning_rate=training.learning_rate,
            capacity_resolution_sha256=training.capacity_resolution_sha256,
            ordinary_dcnv2_selection_sha256=(training.ordinary_dcnv2_selection_sha256),
            capacity_lr_binding_sha256=training.capacity_lr_binding_sha256,
            premetric_protocol_sha256=gate.digest,
        )
        model_digest = capacity_resolved_model_config_sha256(
            interaction_config=_mapping(
                selected_model_payload["interaction_config"],
                "model.interaction_config",
            ),
            head_config=_mapping(
                selected_model_payload["head_config"], "model.head_config"
            ),
            capacity_selection_binding=selection_binding,
        )
    else:
        model_digest = hashlib.sha256(
            json.dumps(
                selected_model_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
    if model_digest != training.resolved_model_config_sha256:
        raise TrainingContractError("resolved model digest mismatch")
    interaction, head, init_seed = resolved_model_components(model_payload)
    if is_capacity and (
        not isinstance(interaction, DCNv2CapacityCoreConfig)
        or interaction.low_rank != CAPACITY_LOW_RANK
        or interaction.num_layers != 2
        or interaction.num_experts != 4
        or interaction.product_init_scale != 0.1
    ):
        raise TrainingContractError("capacity model geometry is not authorized")
    if init_seed != training.seed:
        raise TrainingContractError("model init_seed differs from run seed")
    if training.encoder_contract_sha256 != encoder_lock.sha256:
        raise TrainingContractError("encoder digest mismatch")
    if training.capacity_resolution_sha256 != capacity_digest:
        raise TrainingContractError("capacity artifact digest mismatch")

    train_id = _expected_identity(
        gate,
        encoder_lock,
        "train",
        model_key=training.model_key,
    )
    validation_id = _expected_identity(
        gate,
        encoder_lock,
        "validation",
        model_key=training.model_key,
    )
    model = build_trainable_encoded_model(
        encoder_contract=encoder_lock,
        dataset_id=training.dataset_id,
        source_identity=train_id,
        interaction_config=interaction,
        head_config=head,
        run_seed=training.seed,
        runtime_value_checks=training.runtime_value_checks,
    )
    log = CompactEventLog(run_directory / "history.jsonl")
    guard = _RuntimeGuard(
        root,
        overlay,
        log,
        float(runtime["guard_interval_seconds"]),
    )
    guard.check(eta_hours=eta, force=True)
    assert guard.cost is not None and guard.storage is not None
    safety = SafetyAttestation(
        protocol_gate=gate,
        cost_snapshot=guard.cost,
        storage_snapshot=guard.storage,
        leakage_checks=_mapping(payload["leakage_checks"], "leakage"),
        views_payload=views,
        views_payload_sha256=hashlib.sha256(
            json.dumps(
                views,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        views_split_sha256=gate.payload["hashes"]["split_sha256"],
        search_space_sha256=search.digest,
        experiment_matrix_sha256=search.experiment_matrix_digest,
        encoder_contract_sha256=encoder_lock.sha256,
        capacity_resolution_sha256=capacity_digest,
        train_runtime_identity=runtime_identity_from_raw(
            train_id,
            encoder_contract_sha256=encoder_lock.sha256,
            capacity_resolution_sha256=capacity_digest,
        ),
        validation_runtime_identity=runtime_identity_from_raw(
            validation_id,
            encoder_contract_sha256=encoder_lock.sha256,
            capacity_resolution_sha256=capacity_digest,
        ),
        train_source_identity=train_id,
        validation_source_identity=validation_id,
        registry_on_persistent_storage=True,
    )
    cache_root = _path(
        payload["runtime_cache_root"],
        root=root,
        field="runtime_cache_root",
        inside=False,
    )
    scan_path = local("full_scan_manifest_path")

    def source_factory(split: str):
        def build():
            return build_vk_lsvd_raw_source(
                cache_root,
                split=split,  # type: ignore[arg-type]
                full_scan_manifest_path=scan_path,
                feature_regime=training.feature_regime,
                expected_scan_sha256=train_id.full_materialized_scan_sha256,
                expected_cache_content_sha256=train_id.cache_content_sha256,
                verify_hashes=True,
                enforce_production_locks=True,
                runtime_value_checks=training.runtime_value_checks,
            )

        return build

    train_factory = source_factory("train")

    def forbidden_smoke_validation_factory():
        raise TrainingContractError(
            "implementation smoke must not open or evaluate validation"
        )

    validation_factory = (
        forbidden_smoke_validation_factory
        if execution_kind == "implementation_smoke"
        else source_factory("validation")
    )
    harness = TrainingHarness(
        model=model,
        config=training,
        safety=safety,
        search_lock=search,
        capacity_resolution=None,
        capacity_artifact_path=capacity_path,
        capacity_lr_binding_path=capacity_binding_path,
        repository_root=root,
        operand_feature_order=train_id.ordered_operand_fields,
        run_directory=run_directory,
        device=device,
        event_log=log,
        heartbeat=guard.check,
    )
    try:
        log.emit(
            "run_started",
            run_id=payload["run_id"],
            execution_kind=execution_kind,
            resolved_config_sha256=verify_sha256_sidecar(resolved_path),
            protocol_sha256=gate.digest,
            joint_model_contract_sha256=model.contract_sha256,
            reproduction_command=training.reproduction_command,
        )
        result = harness.fit(
            train_factory,
            validation_factory,
            resume=runtime["resume"],
            stop_after_examples=runtime["stop_after_examples"],
        )
        scientific_result_eligible = _validate_result_scope(
            execution_kind,
            training,
            result,
        )
        latency_result = None
        if runtime["latency"] is not None:
            latency = _mapping(runtime["latency"], "latency")
            _exact_keys(
                latency,
                {"warmup_repetitions", "measured_repetitions", "batch_size"},
                "latency",
            )
            harness.restore_best_for_evaluation()
            latency_result = asdict(
                harness.measure_model_latency(
                    validation_factory,
                    warmup_repetitions=latency["warmup_repetitions"],
                    measured_repetitions=latency["measured_repetitions"],
                    batch_size=latency["batch_size"],
                )
            )
        guard.check(force=True)
        receipt = {
            "schema_version": schema_version,
            "execution_kind": execution_kind,
            "scientific_result_eligible": scientific_result_eligible,
            "run_id": payload["run_id"],
            "resolved_config_sha256": verify_sha256_sidecar(resolved_path),
            "protocol_sha256": gate.digest,
            "training": asdict(result),
            "selected_validation_metric": result.best_validation_metric,
            "test_metrics": None,
            "end_to_end_raw_to_logit_validation_latency": latency_result,
            "capacity_resolution_sha256": capacity_digest,
            "joint_model_contract_sha256": model.contract_sha256,
            "final_cost_snapshot": guard.cost,
            "final_storage_snapshot": guard.storage,
            "telemetry": "disabled",
        }
        if is_capacity:
            receipt["capacity_training_binding"] = capacity_training_binding_payload(
                capacity_resolution_sha256=training.capacity_resolution_sha256,
                ordinary_dcnv2_selection_sha256=(
                    training.ordinary_dcnv2_selection_sha256
                ),
                capacity_lr_binding_sha256=(training.capacity_lr_binding_sha256),
            )
        digest = _write_immutable_json(result_path, receipt)
        log.emit(
            "result_written",
            path=result_path.name,
            sha256=digest,
            execution_kind=execution_kind,
            scientific_result_eligible=scientific_result_eligible,
        )
        print(json.dumps({"ok": True, "result": str(result_path), "sha256": digest}))
        return 0
    except BaseException as exc:
        log.emit("run_failed", error_type=type(exc).__name__, message=str(exc)[:1000])
        raise
    finally:
        harness.close()


def main(argv: list[str] | None = None) -> int:
    _disable_telemetry()
    _reject_distributed_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run",))
    parser.add_argument("--resolved", type=Path, required=True)
    args = parser.parse_args(argv)
    return _run(args.resolved.resolve(strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
