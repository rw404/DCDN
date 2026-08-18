#!/usr/bin/env python3
"""Fail-closed single-process VK C+P trainer for frozen DCDN experiments."""

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

from dcdn.capacity_artifact import sha256_file as capacity_sha256_file
from dcdn.data.vk_lsvd_runtime import C_PLUS_P, build_vk_lsvd_raw_source
from dcdn.encoder import EncoderContractLock, build_trainable_encoded_model
from dcdn.protocol import (
    ProtocolGate,
    load_verified_yaml,
    read_verified_artifact,
    verify_sha256_sidecar,
)
from dcdn.raw import RawSourceIdentity
from dcdn.search_space import SearchSpaceLock
from dcdn.training import (
    CompactEventLog,
    SafetyAttestation,
    TrainingConfig,
    TrainingContractError,
    TrainingHarness,
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


def _require_scientific_runtime_check_mode(config: TrainingConfig) -> None:
    if config.runtime_value_checks:
        raise TrainingContractError(
            "scientific tuning requires runtime_value_checks=false; true is reserved "
            "for a dedicated one-batch implementation-smoke path"
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
) -> RawSourceIdentity:
    if gate.dataset_id != _DATASET_ID or gate.feature_regime != C_PLUS_P:
        raise TrainingContractError("CLI supports only frozen VK C+P v2")
    dataset = _mapping(lock.payload["datasets"][_DATASET_ID], "encoder dataset")
    regime = _mapping(dataset["regimes"][C_PLUS_P], "encoder regime")
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
    keys = {
        "schema_version",
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
    _exact_keys(payload, keys, "resolved config")
    if payload["schema_version"] != 2:
        raise TrainingContractError("resolved schema_version must equal two")
    root = _path(payload["project_root"], root=PROJECT_ROOT, field="root", inside=True)
    if root != PROJECT_ROOT:
        raise TrainingContractError("project_root does not match this checkout")
    overlay = _path(payload["overlay_path"], root=root, field="overlay", inside=False)

    def local(name: str) -> Path:
        return _path(payload[name], root=root, field=name, inside=True)

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

    training = _training_config(_mapping(payload["training"], "training"))
    if (
        training.dataset_id != _DATASET_ID
        or training.feature_regime != C_PLUS_P
        or training.phase != "tuning"
    ):
        raise TrainingContractError("CLI permits only premetric VK C+P tuning")
    _require_scientific_runtime_check_mode(training)
    model_payload = _mapping(payload["model"], "model")
    # selected_configs.lock has one cell per model, not per final seed.  Its
    # digest therefore covers only seed-invariant architecture/head choices;
    # init_seed remains exact in TrainingConfig and the joint checkpoint.
    selected_model_payload = {
        "interaction_config": model_payload["interaction_config"],
        "head_config": model_payload["head_config"],
    }
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
    if init_seed != training.seed:
        raise TrainingContractError("model init_seed differs from run seed")
    if training.encoder_contract_sha256 != encoder_lock.sha256:
        raise TrainingContractError("encoder digest mismatch")
    if training.capacity_resolution_sha256 != capacity_digest:
        raise TrainingContractError("capacity artifact digest mismatch")

    train_id = _expected_identity(gate, encoder_lock, "train")
    validation_id = _expected_identity(gate, encoder_lock, "validation")
    model = build_trainable_encoded_model(
        encoder_contract=encoder_lock,
        dataset_id=training.dataset_id,
        source_identity=train_id,
        interaction_config=interaction,
        head_config=head,
        run_seed=training.seed,
        runtime_value_checks=training.runtime_value_checks,
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
    eta = Decimal(str(runtime["eta_hours"]))
    if eta < 0:
        raise TrainingContractError("runtime.eta_hours must be non-negative")
    device = _device(runtime["device"])
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
                feature_regime=C_PLUS_P,
                expected_scan_sha256=train_id.full_materialized_scan_sha256,
                expected_cache_content_sha256=train_id.cache_content_sha256,
                verify_hashes=True,
                enforce_production_locks=True,
                runtime_value_checks=training.runtime_value_checks,
            )

        return build

    train_factory = source_factory("train")
    validation_factory = source_factory("validation")
    harness = TrainingHarness(
        model=model,
        config=training,
        safety=safety,
        search_lock=search,
        capacity_resolution=None,
        capacity_artifact_path=capacity_path,
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
            "schema_version": 2,
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
        digest = _write_immutable_json(result_path, receipt)
        log.emit("result_written", path=result_path.name, sha256=digest)
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
