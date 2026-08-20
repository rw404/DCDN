"""At-most-once final evaluation and immutable metric bundle publication."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from dcdn.metrics import watch_time_metrics
from dcdn.protocol import (
    SHA256_RE,
    ProtocolGate,
    RuntimeIdentity,
    TestClaimRegistry,
    TestEvaluationClaim,
    verify_sha256_sidecar,
)
from dcdn.raw import FinalRawBatchSource, RawAuditBatch, RawFeatureBatch

FINAL_BUNDLE_SCHEMA_VERSION = 1


class FinalEvaluationError(RuntimeError):
    """The final one-pass evaluation contract failed closed."""


@dataclass(frozen=True)
class FinalBundleRecord:
    path: Path
    sidecar_path: Path
    sha256: str
    size_bytes: int
    payload: Mapping[str, Any]
    claim: TestEvaluationClaim


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise FinalEvaluationError(
            f"final bundle is not canonical JSON: {exc}"
        ) from exc


def _relative_output(path: str | Path) -> PurePosixPath:
    value = str(path)
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise FinalEvaluationError("final bundle path must be repository-relative")
    if len(relative.parts) < 3 or relative.parts[:2] != ("results", "final"):
        raise FinalEvaluationError("final bundle must remain under results/final")
    if relative.suffix != ".json" or relative.name.startswith("."):
        raise FinalEvaluationError("final bundle must be a visible JSON file")
    return relative


def _open_root(root: Path) -> int:
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise FinalEvaluationError("repository root is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise FinalEvaluationError("repository root must be a real directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(root, flags)
    except OSError as exc:
        raise FinalEvaluationError(
            "repository root could not be opened safely"
        ) from exc


def _open_parent(root: Path, relative: PurePosixPath, *, create: bool) -> int:
    descriptor = _open_root(root)
    try:
        for component in relative.parts[:-1]:
            if component in {"", ".", ".."}:
                raise FinalEvaluationError("final bundle parent is unsafe")
            if create:
                try:
                    os.mkdir(component, mode=0o750, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise FinalEvaluationError(
                        "final bundle parent could not be created safely"
                    ) from exc
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise FinalEvaluationError(
                    "final bundle parent contains a symlink or non-directory"
                ) from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def preflight_final_bundle_path(
    repository_root: str | Path,
    result_path: str | Path,
) -> PurePosixPath:
    """Create only safe parent directories and prove both outputs are absent."""

    root = Path(repository_root)
    relative = _relative_output(result_path)
    parent_fd = _open_parent(root, relative, create=True)
    try:
        for name in (relative.name, f"{relative.name}.sha256"):
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise FinalEvaluationError(
                    "final bundle output could not be inspected"
                ) from exc
            raise FinalEvaluationError(f"immutable final bundle output exists: {name}")
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return relative


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise FinalEvaluationError("short write while publishing final bundle")
        offset += written
    os.fsync(descriptor)


def publish_final_bundle(
    repository_root: str | Path,
    result_path: str | Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str, int]:
    """Publish a fail-closed immutable JSON/strict-sidecar pair."""

    root = Path(repository_root)
    relative = _relative_output(result_path)
    data = _canonical_json(payload)
    digest = hashlib.sha256(data).hexdigest()
    sidecar_name = f"{relative.name}.sha256"
    sidecar_data = f"{digest}  {relative.name}\n".encode("ascii")
    parent_fd = _open_parent(root, relative, create=False)
    published: list[str] = []
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for name, content in ((relative.name, data), (sidecar_name, sidecar_data)):
            descriptor = os.open(name, flags, 0o640, dir_fd=parent_fd)
            try:
                _write_all(descriptor, content)
            finally:
                os.close(descriptor)
            published.append(name)
        os.fsync(parent_fd)
    except BaseException:
        for name in reversed(published):
            with suppress(FileNotFoundError):
                os.unlink(name, dir_fd=parent_fd)
        with suppress(OSError):
            os.fsync(parent_fd)
        raise
    finally:
        os.close(parent_fd)

    absolute = root.joinpath(*relative.parts)
    try:
        reopened = absolute.read_bytes()
    except OSError as exc:
        raise FinalEvaluationError(
            "published final bundle could not be reopened"
        ) from exc
    if reopened != data or verify_sha256_sidecar(absolute) != digest:
        raise FinalEvaluationError("published final bundle failed exact read-back")
    return absolute, digest, len(data)


def _digest_strings(values: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        hasher.update(struct.pack("<Q", len(encoded)))
        hasher.update(encoded)
    return hasher.hexdigest()


def _digest_u64(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values.astype("<u8", copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _digest_f64(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values.astype("<f8", copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _user_cluster_sha256(user_id: int) -> str:
    payload = b"dcdn-user-cluster-v1\0" + struct.pack("<Q", user_id)
    return hashlib.sha256(payload).hexdigest()


class _MetricAccumulator:
    def __init__(self, row_count: int) -> None:
        if type(row_count) is not int or row_count <= 1:
            raise FinalEvaluationError("final evaluation needs at least two rows")
        self.row_count = row_count
        self.cursor = 0
        self.targets = np.empty(row_count, dtype=np.float64)
        self.predictions = np.empty(row_count, dtype=np.float64)
        self.user_ids = np.empty(row_count, dtype=np.uint64)
        self.stable_row_ids: list[str] = []
        self._stable_seen: set[str] = set()

    def add(
        self,
        indices: Sequence[int],
        raw_batch: RawFeatureBatch,
        audit_batch: RawAuditBatch,
        predictions: torch.Tensor | np.ndarray,
    ) -> None:
        requested = tuple(indices)
        expected = tuple(range(self.cursor, self.cursor + len(requested)))
        if requested != expected:
            raise FinalEvaluationError(
                "final rows must be consumed once in exact order"
            )
        target = (
            raw_batch.targets.detach().to(device="cpu", dtype=torch.float64).numpy()
        )
        target = target.reshape(-1)
        if isinstance(predictions, torch.Tensor):
            prediction = (
                predictions.detach().to(device="cpu", dtype=torch.float64).numpy()
            )
        else:
            prediction = np.asarray(predictions, dtype=np.float64)
        prediction = prediction.reshape(-1)
        if target.shape != (len(requested),) or prediction.shape != target.shape:
            raise FinalEvaluationError("final target/prediction batch shape mismatch")
        if not np.isfinite(target).all() or not np.isfinite(prediction).all():
            raise FinalEvaluationError(
                "final target/prediction contains non-finite values"
            )
        if np.any((target < 0.0) | (target > 1.0)):
            raise FinalEvaluationError("final normalized target lies outside [0, 1]")
        duplicated = self._stable_seen.intersection(audit_batch.stable_row_ids)
        if duplicated:
            raise FinalEvaluationError("stable_row_id repeats across final batches")
        self._stable_seen.update(audit_batch.stable_row_ids)
        stop = self.cursor + len(requested)
        self.targets[self.cursor : stop] = target
        self.predictions[self.cursor : stop] = prediction
        self.user_ids[self.cursor : stop] = np.asarray(
            audit_batch.user_ids, dtype=np.uint64
        )
        self.stable_row_ids.extend(audit_batch.stable_row_ids)
        self.cursor = stop

    def finalize(self, *, duration_constant_seconds: float) -> dict[str, Any]:
        if self.cursor != self.row_count:
            raise FinalEvaluationError("final source was not consumed completely")
        metrics = watch_time_metrics(
            self.targets,
            self.predictions,
            duration_constant_seconds=duration_constant_seconds,
        )
        squared_error = np.square(self.predictions - self.targets)
        cluster: dict[int, list[float | int]] = {}
        for user_id, error in zip(
            self.user_ids.tolist(), squared_error.tolist(), strict=True
        ):
            bucket = cluster.setdefault(int(user_id), [0, 0.0])
            bucket[0] = int(bucket[0]) + 1
            bucket[1] = float(bucket[1]) + float(error)
        return {
            "metrics": asdict(metrics),
            "pairing": {
                "stable_row_ids_sha256": _digest_strings(self.stable_row_ids),
                "user_ids_u64le_sha256": _digest_u64(self.user_ids),
                "normalized_targets_f64le_sha256": _digest_f64(self.targets),
                "normalized_predictions_f64le_sha256": _digest_f64(self.predictions),
                "squared_errors_f64le_sha256": _digest_f64(squared_error),
                "unique_stable_row_ids": len(self._stable_seen),
                "user_cluster_count": len(cluster),
                "user_cluster_sufficient_statistics": [
                    {
                        "user_cluster_sha256": _user_cluster_sha256(user_id),
                        "row_count": int(values[0]),
                        "normalized_squared_error_sum": float(values[1]),
                    }
                    for user_id, values in sorted(cluster.items())
                ],
            },
        }


def _runtime_matches_source(
    identity: RuntimeIdentity, source: FinalRawBatchSource
) -> None:
    raw = source.identity
    expected = {
        "dataset_id": identity.dataset_id,
        "split": "test",
        "feature_regime": identity.feature_regime,
        "row_count": identity.row_count,
        "data_manifest_sha256": identity.data_manifest_sha256,
        "views_manifest_sha256": identity.views_manifest_sha256,
        "feature_contract_sha256": identity.features_sha256,
        "physical_schema_sha256": identity.physical_schema_sha256,
        "expanded_schema_sha256": identity.expanded_schema_sha256,
        "operand_schema_sha256": identity.operand_schema_sha256,
        "denominator_mask_sha256": identity.denominator_mask_sha256,
        "full_materialized_scan_sha256": identity.full_materialized_scan_sha256,
        "cache_content_sha256": identity.cache_content_sha256,
    }
    for name, value in expected.items():
        if getattr(raw, name) != value:
            raise FinalEvaluationError(f"final raw/runtime identity mismatch: {name}")
    if len(source) != identity.row_count:
        raise FinalEvaluationError("final source length differs from runtime identity")


def _read_bounded_at(parent_fd: int, name: str, *, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise FinalEvaluationError(f"final bundle file is unavailable: {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise FinalEvaluationError(f"final bundle entry is not regular: {name}")
        if metadata.st_size > limit:
            raise FinalEvaluationError(f"final bundle entry exceeds size limit: {name}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise FinalEvaluationError(
                    f"final bundle entry exceeds size limit: {name}"
                )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _strict_json(data: bytes) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise FinalEvaluationError(f"duplicate final bundle key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(data.decode("ascii"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalEvaluationError("final bundle is not strict ASCII JSON") from exc
    if not isinstance(payload, Mapping):
        raise FinalEvaluationError("final bundle root must be a mapping")
    if _canonical_json(payload) != data:
        raise FinalEvaluationError("final bundle bytes are not canonical")
    return payload


def load_verified_final_bundle(
    repository_root: str | Path,
    result_path: str | Path,
) -> tuple[Path, Mapping[str, Any], str, int]:
    """Open the immutable bundle and strict sidecar from held no-follow parents."""

    root = Path(repository_root)
    relative = _relative_output(result_path)
    parent_fd = _open_parent(root, relative, create=False)
    try:
        data = _read_bounded_at(parent_fd, relative.name, limit=64 * 1024 * 1024)
        sidecar_name = f"{relative.name}.sha256"
        sidecar = _read_bounded_at(parent_fd, sidecar_name, limit=4096)
    finally:
        os.close(parent_fd)
    digest = hashlib.sha256(data).hexdigest()
    expected_sidecar = f"{digest}  {relative.name}\n".encode("ascii")
    if sidecar != expected_sidecar:
        raise FinalEvaluationError("final bundle sidecar grammar or digest mismatch")
    payload = _strict_json(data)
    return root.joinpath(*relative.parts), payload, digest, len(data)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FinalEvaluationError(f"{field} must be numeric")
    converted = float(value)
    if not np.isfinite(converted):
        raise FinalEvaluationError(f"{field} must be finite")
    return converted


def _validate_bundle_payload(
    payload: Mapping[str, Any],
    claim: TestEvaluationClaim,
    runtime_identity: RuntimeIdentity,
) -> None:
    expected_root = {
        "schema_version",
        "artifact_role",
        "claim",
        "runtime_identity",
        "evaluation",
        "metrics",
        "pairing",
    }
    if set(payload) != expected_root:
        raise FinalEvaluationError("final bundle root keys mismatch")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise FinalEvaluationError("final bundle schema_version must equal 1")
    if payload["artifact_role"] != "one_pass_test_metric_bundle":
        raise FinalEvaluationError("final bundle artifact_role mismatch")

    expected_claim = {
        "claim_id": claim.claim_id,
        "model_key": claim.model_key,
        "seed": claim.seed,
        "final_protocol_sha256": claim.final_protocol_sha256,
        "parent_premetric_protocol_sha256": claim.parent_premetric_protocol_sha256,
        "selected_configs_sha256": claim.selected_configs_sha256,
        "selection_receipt_sha256": claim.selection_receipt_sha256,
        "candidate_id": claim.candidate_id,
        "resolved_config_sha256": claim.resolved_config_sha256,
        "resolved_run_config_sha256": claim.resolved_run_config_sha256,
        "validation_result_sha256": claim.validation_result_sha256,
        "checkpoint_sha256": claim.checkpoint_sha256,
        "selected_validation_metric": claim.selected_validation_metric,
    }
    if payload["claim"] != expected_claim:
        raise FinalEvaluationError("final bundle claim binding mismatch")
    if payload["runtime_identity"] != asdict(runtime_identity):
        raise FinalEvaluationError("final bundle runtime identity mismatch")

    evaluation = payload["evaluation"]
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "row_count",
        "batch_size",
        "source_passes",
        "source_factory_calls",
        "test_metrics_read",
        "validation_selection_changed",
        "duration_constant_seconds",
    }:
        raise FinalEvaluationError("final bundle evaluation keys mismatch")
    if (
        type(evaluation["row_count"]) is not int
        or evaluation["row_count"] != runtime_identity.row_count
        or type(evaluation["batch_size"]) is not int
        or evaluation["batch_size"] <= 0
        or evaluation["source_passes"] != 1
        or type(evaluation["source_passes"]) is not int
        or evaluation["source_factory_calls"] != 1
        or type(evaluation["source_factory_calls"]) is not int
        or evaluation["test_metrics_read"] is not True
        or evaluation["validation_selection_changed"] is not False
    ):
        raise FinalEvaluationError("final bundle one-pass evaluation contract failed")
    if (
        _finite_number(
            evaluation["duration_constant_seconds"], "duration_constant_seconds"
        )
        <= 0
    ):
        raise FinalEvaluationError("duration_constant_seconds must be positive")

    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "normalized_mse",
        "mae_seconds",
        "mse_seconds2",
        "concordance",
    }:
        raise FinalEvaluationError("final bundle metric keys mismatch")
    for name in ("normalized_mse", "mae_seconds", "mse_seconds2"):
        if _finite_number(metrics[name], f"metrics.{name}") < 0:
            raise FinalEvaluationError(f"metrics.{name} must be non-negative")
    concordance = metrics["concordance"]
    if not isinstance(concordance, Mapping) or set(concordance) != {
        "value",
        "comparable_pairs",
        "total_pairs",
        "comparable_pair_coverage",
        "concordant_pairs",
        "discordant_pairs",
        "prediction_tied_pairs",
    }:
        raise FinalEvaluationError("final bundle concordance keys mismatch")
    value = _finite_number(concordance["value"], "concordance.value")
    coverage = _finite_number(
        concordance["comparable_pair_coverage"], "concordance.coverage"
    )
    if not 0 <= value <= 1 or not 0 <= coverage <= 1:
        raise FinalEvaluationError("final concordance value/coverage is invalid")
    for name in (
        "comparable_pairs",
        "total_pairs",
        "concordant_pairs",
        "discordant_pairs",
        "prediction_tied_pairs",
    ):
        if type(concordance[name]) is not int or concordance[name] < 0:
            raise FinalEvaluationError(f"concordance.{name} must be non-negative int")
    if (
        concordance["total_pairs"]
        != runtime_identity.row_count * (runtime_identity.row_count - 1) // 2
    ):
        raise FinalEvaluationError("final concordance total_pairs mismatch")
    if concordance["comparable_pairs"] != (
        concordance["concordant_pairs"]
        + concordance["discordant_pairs"]
        + concordance["prediction_tied_pairs"]
    ):
        raise FinalEvaluationError("final concordance pair accounting mismatch")

    pairing = payload["pairing"]
    if not isinstance(pairing, Mapping) or set(pairing) != {
        "stable_row_ids_sha256",
        "user_ids_u64le_sha256",
        "normalized_targets_f64le_sha256",
        "normalized_predictions_f64le_sha256",
        "squared_errors_f64le_sha256",
        "unique_stable_row_ids",
        "user_cluster_count",
        "user_cluster_sufficient_statistics",
    }:
        raise FinalEvaluationError("final bundle pairing keys mismatch")
    for name in (
        "stable_row_ids_sha256",
        "user_ids_u64le_sha256",
        "normalized_targets_f64le_sha256",
        "normalized_predictions_f64le_sha256",
        "squared_errors_f64le_sha256",
    ):
        if (
            not isinstance(pairing[name], str)
            or SHA256_RE.fullmatch(pairing[name]) is None
        ):
            raise FinalEvaluationError(f"pairing.{name} must be a SHA-256")
    if (
        type(pairing["unique_stable_row_ids"]) is not int
        or pairing["unique_stable_row_ids"] != runtime_identity.row_count
        or type(pairing["user_cluster_count"]) is not int
        or pairing["user_cluster_count"] <= 0
    ):
        raise FinalEvaluationError("final pairing row/user counts mismatch")
    statistics = pairing["user_cluster_sufficient_statistics"]
    if (
        not isinstance(statistics, list)
        or len(statistics) != pairing["user_cluster_count"]
    ):
        raise FinalEvaluationError("final user-cluster statistics count mismatch")
    seen_users: set[str] = set()
    total_rows = 0
    total_squared_error = 0.0
    for item in statistics:
        if not isinstance(item, Mapping) or set(item) != {
            "user_cluster_sha256",
            "row_count",
            "normalized_squared_error_sum",
        }:
            raise FinalEvaluationError("final user-cluster statistic keys mismatch")
        user_cluster = item["user_cluster_sha256"]
        rows = item["row_count"]
        if (
            not isinstance(user_cluster, str)
            or SHA256_RE.fullmatch(user_cluster) is None
            or user_cluster in seen_users
            or type(rows) is not int
            or rows <= 0
        ):
            raise FinalEvaluationError("final user-cluster identifier/count is invalid")
        seen_users.add(user_cluster)
        total_rows += rows
        error_sum = _finite_number(
            item["normalized_squared_error_sum"], "cluster squared-error sum"
        )
        if error_sum < 0:
            raise FinalEvaluationError("cluster squared-error sum must be non-negative")
        total_squared_error += error_sum
    if total_rows != runtime_identity.row_count:
        raise FinalEvaluationError("final user-cluster rows are incomplete")
    recomputed_mse = total_squared_error / total_rows
    if not np.isclose(
        recomputed_mse,
        float(metrics["normalized_mse"]),
        rtol=1e-12,
        atol=1e-15,
    ):
        raise FinalEvaluationError("final normalized_mse differs from cluster sums")


def recover_published_test_bundle(
    *,
    gate: ProtocolGate,
    selected_configs_path: str | Path,
    runtime_identity: RuntimeIdentity,
    claim_id: str,
    repository_root: str | Path,
    result_path: str | Path,
) -> FinalBundleRecord:
    """Complete an opened claim from its bundle without reopening test data."""

    active = gate.require_active()
    claims_path = active.payload["test_claim_registry_path"]
    if not isinstance(claims_path, str):
        raise FinalEvaluationError("active final protocol lacks a claim registry")
    claim = TestClaimRegistry(claims_path).load_claim(
        claim_id,
        gate=gate,
        selected_configs_path=selected_configs_path,
        runtime_identity=runtime_identity,
    )
    if claim.state not in {"opened", "completed"}:
        raise FinalEvaluationError(
            "only an opened/completed claim can recover a bundle"
        )
    path, payload, digest, size_bytes = load_verified_final_bundle(
        repository_root,
        result_path,
    )
    _validate_bundle_payload(payload, claim, runtime_identity)
    completed = claim.mark_completed(
        gate=gate,
        selected_configs_path=selected_configs_path,
        runtime_identity=runtime_identity,
        result_receipt_sha256=digest,
        result_row_count=runtime_identity.row_count,
    )
    return FinalBundleRecord(
        path=path,
        sidecar_path=path.with_name(f"{path.name}.sha256"),
        sha256=digest,
        size_bytes=size_bytes,
        payload=payload,
        claim=completed,
    )


def run_one_pass_test_bundle(
    *,
    gate: ProtocolGate,
    selected_configs_path: str | Path,
    runtime_identity: RuntimeIdentity,
    model_key: str,
    seed: int,
    candidate_id: str,
    resolved_config_sha256: str,
    resolved_run_config_sha256: str,
    checkpoint_protocol_sha256: str,
    checkpoint_sha256: str,
    inspect_checkpoint: Callable[[], str],
    restore_checkpoint: Callable[[], str],
    source_factory: Callable[[], FinalRawBatchSource],
    predict_batch: Callable[[RawFeatureBatch], torch.Tensor | np.ndarray],
    repository_root: str | Path,
    result_path: str | Path,
    device: torch.device | str,
    batch_size: int,
    duration_constant_seconds: float,
    finalize_predictions: Callable[[], None] | None = None,
) -> FinalBundleRecord:
    """Run one exact final cell without permitting automatic re-evaluation."""

    if type(batch_size) is not int or batch_size <= 0:
        raise FinalEvaluationError("final batch_size must be a positive integer")
    if not callable(inspect_checkpoint) or not callable(restore_checkpoint):
        raise FinalEvaluationError("checkpoint inspect/restore callbacks are required")
    if not callable(source_factory) or not callable(predict_batch):
        raise FinalEvaluationError("final source and predictor callbacks are required")
    if finalize_predictions is not None and not callable(finalize_predictions):
        raise FinalEvaluationError(
            "finalize_predictions must be callable when provided"
        )
    evaluation_device = torch.device(device)
    if evaluation_device.type not in {"cpu", "cuda"}:
        raise FinalEvaluationError("final evaluation device must be CPU or CUDA")
    relative_result = preflight_final_bundle_path(repository_root, result_path)

    claim = gate.require_split(
        "test",
        runtime_identity=runtime_identity,
        selected_configs_path=selected_configs_path,
        model_key=model_key,
        candidate_id=candidate_id,
        resolved_config_sha256=resolved_config_sha256,
        resolved_run_config_sha256=resolved_run_config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        seed=seed,
        checkpoint_protocol_sha256=checkpoint_protocol_sha256,
    )
    if claim is None or claim.state != "reserved":
        raise FinalEvaluationError("final claim was not durably reserved")
    if inspect_checkpoint() != checkpoint_sha256:
        raise FinalEvaluationError("checkpoint inspection digest mismatch")
    if restore_checkpoint() != checkpoint_sha256:
        raise FinalEvaluationError("checkpoint restore digest mismatch")

    opened = claim.mark_opened(
        gate=gate,
        selected_configs_path=selected_configs_path,
        runtime_identity=runtime_identity,
    )
    if opened.state != "opened":
        raise FinalEvaluationError("final claim did not enter opened state")

    source: FinalRawBatchSource | None = None
    accumulator = _MetricAccumulator(runtime_identity.row_count)
    try:
        source = source_factory()
        if not isinstance(source, FinalRawBatchSource):
            raise FinalEvaluationError("final source lacks batch_with_audit contract")
        _runtime_matches_source(runtime_identity, source)
        for start in range(0, len(source), batch_size):
            indices = tuple(range(start, min(start + batch_size, len(source))))
            raw_batch, audit_batch = source.batch_with_audit(
                indices,
                device=evaluation_device,
            )
            if not isinstance(raw_batch, RawFeatureBatch) or not isinstance(
                audit_batch, RawAuditBatch
            ):
                raise FinalEvaluationError("final source returned invalid batch types")
            raw_batch.validate(
                source.identity,
                indices,
                runtime_value_checks=source.runtime_value_checks,
            )
            audit_batch.validate(source.identity, indices)
            if not torch.equal(
                raw_batch.global_row_indices, audit_batch.global_row_indices
            ):
                raise FinalEvaluationError("model and audit row indices differ")
            accumulator.add(indices, raw_batch, audit_batch, predict_batch(raw_batch))
    finally:
        if source is not None:
            source.close()

    if finalize_predictions is not None:
        finalize_predictions()
    summaries = accumulator.finalize(
        duration_constant_seconds=duration_constant_seconds
    )
    payload: dict[str, Any] = {
        "schema_version": FINAL_BUNDLE_SCHEMA_VERSION,
        "artifact_role": "one_pass_test_metric_bundle",
        "claim": {
            "claim_id": opened.claim_id,
            "model_key": opened.model_key,
            "seed": opened.seed,
            "final_protocol_sha256": opened.final_protocol_sha256,
            "parent_premetric_protocol_sha256": (
                opened.parent_premetric_protocol_sha256
            ),
            "selected_configs_sha256": opened.selected_configs_sha256,
            "selection_receipt_sha256": opened.selection_receipt_sha256,
            "candidate_id": opened.candidate_id,
            "resolved_config_sha256": opened.resolved_config_sha256,
            "resolved_run_config_sha256": opened.resolved_run_config_sha256,
            "validation_result_sha256": opened.validation_result_sha256,
            "checkpoint_sha256": opened.checkpoint_sha256,
            "selected_validation_metric": opened.selected_validation_metric,
        },
        "runtime_identity": asdict(runtime_identity),
        "evaluation": {
            "row_count": runtime_identity.row_count,
            "batch_size": batch_size,
            "source_passes": 1,
            "source_factory_calls": 1,
            "test_metrics_read": True,
            "validation_selection_changed": False,
            "duration_constant_seconds": duration_constant_seconds,
        },
        **summaries,
    }
    _validate_bundle_payload(payload, opened, runtime_identity)
    absolute, digest, size_bytes = publish_final_bundle(
        repository_root,
        relative_result,
        payload,
    )
    completed = opened.mark_completed(
        gate=gate,
        selected_configs_path=selected_configs_path,
        runtime_identity=runtime_identity,
        result_receipt_sha256=digest,
        result_row_count=runtime_identity.row_count,
    )
    return FinalBundleRecord(
        path=absolute,
        sidecar_path=absolute.with_name(f"{absolute.name}.sha256"),
        sha256=digest,
        size_bytes=size_bytes,
        payload=payload,
        claim=completed,
    )
