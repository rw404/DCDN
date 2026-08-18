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
import time
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import FrameType
from typing import Any

from cost_guard import evaluate_cost
from storage_guard import inspect_storage

from dcdn.protocol import (
    load_verified_yaml,
    read_verified_artifact,
    verify_sha256_sidecar,
)
from dcdn.registry import RunRegistry

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


def _verify_completed_result(
    *,
    project_root: Path,
    resolved_config: Path,
    expected_resolved_digest: str,
    expected_resolved: Mapping[str, Any],
    run_id: str,
) -> str:
    """Re-open exact config/result bytes before publishing completed state."""

    if verify_sha256_sidecar(resolved_config) != expected_resolved_digest:
        raise WorkerError("resolved config digest drifted while worker was running")
    reloaded = load_verified_yaml(resolved_config)
    if reloaded != expected_resolved:
        raise WorkerError("resolved config payload drifted while worker was running")
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

    try:
        receipt = json.loads(
            artifact.data.decode("ascii"), parse_constant=reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkerError("result receipt is not strict finite ASCII JSON") from exc
    expected_keys = {
        "schema_version",
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
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise WorkerError("result receipt keys mismatch")
    if receipt["schema_version"] != 2 or receipt["run_id"] != run_id:
        raise WorkerError("result receipt identity mismatch")
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
        training["declared_maximum_train_examples"] != declared_maximum
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
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
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
        or verify_sha256_sidecar(run_directory / "last.pt") != last_digest
    ):
        raise WorkerError("result last checkpoint is missing, stale, or corrupt")
    best_metric = training["best_validation_metric"]
    best_digest = training["best_checkpoint_sha256"]
    best_filename = training["best_checkpoint_filename"]
    if best_metric is None:
        if best_digest is not None or best_filename is not None:
            raise WorkerError("result has a best generation without a metric")
    elif (
        isinstance(best_metric, bool)
        or not isinstance(best_metric, int | float)
        or not math.isfinite(float(best_metric))
        or training["validation_checks"] <= 0
        or not isinstance(best_digest, str)
        or _SHA256_RE.fullmatch(best_digest) is None
        or not isinstance(best_filename, str)
        or _BEST_CHECKPOINT_RE.fullmatch(best_filename) is None
        or verify_sha256_sidecar(run_directory / best_filename) != best_digest
    ):
        raise WorkerError("result best generation is missing, stale, or corrupt")
    if receipt["selected_validation_metric"] != best_metric:
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
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise WorkerError(f"validation latency {name} is invalid")
            percentiles.append(float(value))
        if percentiles != sorted(percentiles):
            raise WorkerError("validation latency percentiles are not ordered")
        if latency["scope"] != "end_to_end_raw_to_logit_model":
            raise WorkerError("validation latency scope is invalid")
    if receipt["capacity_resolution_sha256"] != resolved_training.get(
        "capacity_resolution_sha256"
    ):
        raise WorkerError("result capacity artifact digest mismatch")
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
    return artifact.digest


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
    resolved_config_digest = verify_sha256_sidecar(resolved_config)
    resolved = load_verified_yaml(resolved_config)
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
    protocol_digest = verify_sha256_sidecar(protocol_path)
    protocol_payload = load_verified_yaml(protocol_path)
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
    process: subprocess.Popen[bytes] | None = None
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
        _require_code_identity(project_root, record)
        command = [
            sys.executable,
            str(_TRAIN_SCRIPT),
            "run",
            "--resolved",
            str(resolved_config),
        ]
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
                registry.transition(
                    run_id,
                    "running",
                    pid=process.pid,
                    gpu=gpu,
                )
            except BaseException:
                _stop_child(process)
                raise

            failure_reason: str | None = None
            try:
                last_guard = time.monotonic()
                while process.poll() is None:
                    if stop_signal is not None:
                        failure_reason = f"worker received signal {stop_signal}"
                        _stop_child(process)
                        break
                    registry.heartbeat(run_id, pid=process.pid)
                    now = time.monotonic()
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
                    time.sleep(min(poll_seconds, 15.0))
            except BaseException as exc:
                failure_reason = (
                    f"worker supervisor failure: {type(exc).__name__}: {exc}"
                )
                _stop_child(process)
            return_code = process.wait()
            if failure_reason is None and return_code == 0:
                try:
                    _require_code_identity(project_root, record)
                    result_digest = _verify_completed_result(
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
                    registry.transition(
                        run_id,
                        "completed",
                        pid=process.pid,
                        gpu=gpu,
                        reason=(
                            "trainer exited successfully; exact immutable result "
                            f"re-opened and verified: {result_digest}"
                        ),
                    )
                    return 0
            reason = failure_reason or f"trainer exited with return code {return_code}"
            registry.transition(
                run_id,
                "failed",
                pid=process.pid,
                gpu=gpu,
                reason=reason[:1000],
            )
            return 2
    finally:
        try:
            if process is not None and process.poll() is None:
                _stop_child(process)
            try:
                current = registry.read(run_id)
                if current["status"] == "running":
                    registry.transition(
                        run_id,
                        "failed",
                        pid=process.pid if process is not None else None,
                        gpu=gpu,
                        reason="worker owner exited before a terminal transition",
                    )
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
        finally:
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_worker(
        project_root=args.project_root.resolve(strict=True),
        registry_root=args.registry_root.resolve(strict=True),
        run_id=args.run_id,
        gpu=args.gpu,
        resolved_config=args.resolved.resolve(strict=True),
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps({"ok": result == 0, "run_id": args.run_id, "gpu": args.gpu}))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
