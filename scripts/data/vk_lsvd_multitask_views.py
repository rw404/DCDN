#!/usr/bin/env python3
"""Materialize the frozen VK-LSVD multi-target REDUCED/FULL view pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq
import yaml

from dcdn.data.vk_lsvd_multitask import (
    ALLOWED_WEEKS,
    DATASET_ID,
    DENOMINATOR_ELIGIBLE_FIELDS,
    ORDERED_DENOMINATOR_MASK,
    ORDERED_OPERAND_FIELDS,
    R_DIV_FIELDS,
    SHARED_SIMILARITY_FIELDS,
    TARGET_COLUMNS,
    TARGET_KINDS,
    TARGET_MASK_COLUMNS,
    TARGET_NAMES,
    VALIDATION_WEEK,
    contract_state,
    physical_schema_sha256,
    transform_verified_shard,
    view_schema,
)
from dcdn.protocol import load_verified_yaml_artifact, read_verified_artifact

MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 32 * 1024**3
PARQUET_ROW_GROUP_ROWS = 262_144
EXPECTED_LEGACY_MANIFEST_SHA256 = (
    "738d3610fd83a93297d94ccad4c4316dd3ad8b241f8409e84daf5b2322040dbe"
)
EXPECTED_TARGETS_V2_SHA256 = (
    "d03704fb856551cb5d2eee7810992f0e8fa23909860a9c28c626b5362456c6a6"
)
EXPECTED_FEATURE_POLICY_SHA256 = (
    "fd8baeeaa82ba8e7d825762f30207c7875449a45a625d173bde419869c79b0a3"
)
EXPECTED_MATRIX_SHA256 = (
    "1b73dcb34a565f192cb66669163f71bac244884401e9c958a2cc836b2de36946"
)


class MaterializationError(RuntimeError):
    """The immutable view publication contract failed."""


def _require_safe_directory_chain(path: Path, *, field: str) -> Path:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as error:
            raise MaterializationError(f"{field} is unavailable: {current}") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MaterializationError(f"{field} has unsafe ancestor: {current}")
    return absolute


def _open_safe_regular(path: Path, *, field: str) -> tuple[int, os.stat_result]:
    absolute = (
        _require_safe_directory_chain(path.parent, field=f"{field} parent") / path.name
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise MaterializationError(f"cannot safely open {field}: {path}") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise MaterializationError(f"{field} must be one regular file: {path}")
    if metadata.st_size <= 0 or metadata.st_size > MAX_SOURCE_BYTES:
        os.close(descriptor)
        raise MaterializationError(f"{field} size is outside bounds: {path}")
    return descriptor, metadata


def _sha256_file(path: Path) -> str:
    descriptor, before = _open_safe_regular(path, field="source")
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise MaterializationError(f"source changed while hashing: {path}")
    return digest.hexdigest()


def _read_verified_parquet(
    path: Path, *, expected_sha256: Any, columns: Sequence[str] | None = None
) -> Any:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise MaterializationError("expected Parquet SHA-256 is invalid")
    descriptor, before = _open_safe_regular(path, field="Parquet source")
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise MaterializationError(f"Parquet source hash changed: {path}")
        handle.seek(0)
        table = pq.read_table(handle, columns=columns)
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise MaterializationError(f"Parquet source changed while reading: {path}")
    _require_safe_directory_chain(path.parent, field="Parquet source parent")
    return table


def _safe_relative(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MaterializationError(f"{field} must be a non-empty path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise MaterializationError(f"{field} is unsafe")
    return Path(*pure.parts)


def _resolve_under(root: Path, relative: Path, *, field: str) -> Path:
    resolved_root = _require_safe_directory_chain(root, field=f"{field} root")
    candidate = resolved_root / relative
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise MaterializationError(f"{field} escapes its root") from error
    _require_safe_directory_chain(candidate.parent, field=f"{field} parent")
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise MaterializationError(f"{field} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MaterializationError(f"{field} must be one regular file")
    return candidate


def _require_clean_code_identity(repository_root: Path) -> Mapping[str, str]:
    commands = (
        ("git", "diff", "--quiet"),
        ("git", "diff", "--cached", "--quiet"),
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise MaterializationError("tracked or staged diff must be empty")
    untracked = subprocess.run(
        ("git", "ls-files", "--others", "--exclude-standard", "src", "scripts"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if untracked:
        raise MaterializationError(
            "untracked importable/executable source is forbidden"
        )
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise MaterializationError("HEAD is not a full lowercase Git object ID")
    return {
        "code_commit": commit,
        "code_diff_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _verified_frozen_inputs(repository_root: Path) -> Mapping[str, str]:
    inputs = {
        "targets_multitask_v2_sha256": (
            "configs/targets_multitask_v2.yaml",
            EXPECTED_TARGETS_V2_SHA256,
        ),
        "feature_policy_sha256": (
            "configs/features_manual_hist_divisions_only_v1.yaml",
            EXPECTED_FEATURE_POLICY_SHA256,
        ),
        "matrix_multitask_sha256": (
            "configs/matrix_multitask_v1.yaml",
            EXPECTED_MATRIX_SHA256,
        ),
    }
    result: dict[str, str] = {}
    for field, (relative, expected) in inputs.items():
        verified = load_verified_yaml_artifact(repository_root / relative)
        if verified.digest != expected:
            raise MaterializationError(f"frozen digest changed: {relative}")
        if verified.payload.get("test_unlocked") is not False:
            raise MaterializationError(f"frozen input is not test-locked: {relative}")
        result[field] = verified.digest
    return result


def _source_object_map(payload: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    records = payload.get("source_objects_verified")
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise MaterializationError("legacy source-object inventory is invalid")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise MaterializationError("legacy source-object record is invalid")
        path = record.get("path")
        if not isinstance(path, str) or path in result:
            raise MaterializationError("legacy source-object path is invalid")
        result[path] = record
    return result


def _load_legacy_manifest(path: Path) -> tuple[Mapping[str, Any], str]:
    verified = load_verified_yaml_artifact(path)
    payload = verified.payload
    if verified.digest != EXPECTED_LEGACY_MANIFEST_SHA256:
        raise MaterializationError("legacy views manifest digest changed")
    expected = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "artifact_role": "train_validation_feature_views",
        "materialization_state": "complete_and_hash_verified",
        "test_unlocked": False,
        "test_labels_opened": False,
        "test_view_materialized": False,
        "week26_file_open_attempted": False,
        "test_metrics_computed": False,
    }
    for key, value in expected.items():
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise MaterializationError(f"legacy manifest gate changed: {key}")
    shards = payload.get("shards")
    if not isinstance(shards, Sequence) or len(shards) != len(ALLOWED_WEEKS):
        raise MaterializationError("legacy manifest must contain six allowed shards")
    if tuple(record.get("week") for record in shards) != ALLOWED_WEEKS:
        raise MaterializationError("legacy manifest shard order changed")
    return payload, verified.digest


def _atomic_parquet(path: Path, table: Any) -> Mapping[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise MaterializationError(f"output collision: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp-"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            row_group_size=PARQUET_ROW_GROUP_ROWS,
            use_dictionary=False,
            write_statistics=True,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != table.num_rows:
            raise MaterializationError("reopened Parquet row count changed")
        if not parquet.schema_arrow.equals(table.schema, check_metadata=True):
            raise MaterializationError("reopened Parquet schema changed")
        digest = _sha256_file(path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "rows": table.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
        }
    except BaseException:
        with suppress(OSError):
            path.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    data = yaml.safe_dump(
        dict(payload), sort_keys=True, allow_unicode=False, width=100
    ).encode("ascii")
    if not data or len(data) > MAX_MANIFEST_BYTES:
        raise MaterializationError("manifest size is outside bounds")
    return data


def _publish_manifest(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists() or path.with_name(f"{path.name}.sha256").exists():
        raise MaterializationError("manifest or sidecar already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _manifest_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar_data = f"{digest}  {path.name}\n".encode("ascii")
    published: list[Path] = []
    try:
        for target, content in ((path, data), (sidecar, sidecar_data)):
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                with suppress(OSError):
                    os.close(descriptor)
                raise
            published.append(target)
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        reopened = read_verified_artifact(path)
        if reopened.digest != digest or reopened.data != data:
            raise MaterializationError("manifest pair did not reopen exactly")
    except BaseException:
        for target in reversed(published):
            with suppress(OSError):
                target.unlink()
        raise
    return digest


def _remove_published_output(output_root: Path) -> None:
    """Remove only the fresh output directory published by this invocation."""

    if output_root.is_symlink() or not output_root.is_dir():
        raise MaterializationError("published output root changed during cleanup")
    shutil.rmtree(output_root)
    parent_fd = os.open(output_root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def materialize(
    *,
    repository_root: Path,
    legacy_manifest_path: Path,
    legacy_views_root: Path,
    raw_dataset_root: Path,
    output_root: Path,
    manifest_path: Path,
) -> str:
    """Execute the complete premetric transformation and immutable publication."""

    repository_root = repository_root.resolve(strict=True)
    code_identity = _require_clean_code_identity(repository_root)
    frozen = _verified_frozen_inputs(repository_root)
    legacy, legacy_digest = _load_legacy_manifest(legacy_manifest_path)
    if (
        output_root.exists()
        or manifest_path.exists()
        or manifest_path.with_name(f"{manifest_path.name}.sha256").exists()
    ):
        raise MaterializationError("fresh output root and manifest are required")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(dir=output_root.parent, prefix=f".{output_root.name}.tmp-")
    )
    output_published = False
    source_objects = _source_object_map(legacy)
    shard_records: list[Mapping[str, Any]] = []
    total_rows = 0
    try:
        for legacy_record in legacy["shards"]:
            week = legacy_record["week"]
            role = "train" if week < VALIDATION_WEEK else "validation"
            feature_record = legacy_record["C_plus_P_plus_E"]
            feature_relative = _safe_relative(
                feature_record.get("path"), field="legacy feature shard path"
            )
            feature_path = _resolve_under(
                legacy_views_root, feature_relative, field="legacy feature shard"
            )
            raw_relative_text = (
                f"subsamples/up0.01_ir0.01/{role}/week_{week:02d}.parquet"
            )
            raw_record = source_objects.get(raw_relative_text)
            if raw_record is None:
                raise MaterializationError("raw label source is absent from inventory")
            raw_path = _resolve_under(
                raw_dataset_root,
                _safe_relative(raw_relative_text, field="raw label shard path"),
                field="raw label shard",
            )
            feature_table = _read_verified_parquet(
                feature_path, expected_sha256=feature_record.get("sha256")
            )
            interactions = _read_verified_parquet(
                raw_path,
                expected_sha256=raw_record.get("sha256"),
                columns=[
                    "user_id",
                    "item_id",
                    "like",
                    "dislike",
                    "share",
                    "bookmark",
                    "click_on_author",
                    "open_comments",
                ],
            )
            transformed = transform_verified_shard(
                week=week,
                legacy_full=feature_table,
                interactions=interactions,
            )
            reduced_relative = Path("REDUCED") / role / f"week_{week:02d}.parquet"
            full_relative = Path("FULL") / role / f"week_{week:02d}.parquet"
            if staging is None:
                raise AssertionError("staging directory disappeared")
            reduced_record = dict(
                _atomic_parquet(staging / reduced_relative, transformed.reduced)
            )
            full_record = dict(
                _atomic_parquet(staging / full_relative, transformed.full)
            )
            reduced_record["path"] = reduced_relative.as_posix()
            full_record["path"] = full_relative.as_posix()
            shard_records.append(
                {
                    "week": week,
                    "role": role,
                    "rows": transformed.reduced.num_rows,
                    "legacy_feature_sha256": feature_record["sha256"],
                    "raw_label_source_sha256": raw_record["sha256"],
                    "REDUCED": reduced_record,
                    "FULL": full_record,
                    "shared_C_plus_P_values_sha256": (transformed.shared_values_sha256),
                    "labels_and_masks_sha256": transformed.labels_and_masks_sha256,
                    "stable_row_ids_sha256": transformed.stable_row_ids_sha256,
                }
            )
            total_rows += transformed.reduced.num_rows
        os.rename(staging, output_root)
        output_published = True
        parent_fd = os.open(
            output_root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        staging = None
        state = contract_state()
        state_bytes = json.dumps(
            state, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
        manifest = {
            "schema_version": 2,
            "dataset_id": DATASET_ID,
            "artifact_role": "multi_target_train_validation_feature_views",
            "materialization_state": "complete_and_hash_verified",
            "materialized_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "algorithm_version": "vk-lsvd-multitask-view-transform-v2",
            "algorithm_source_sha256": _sha256_file(Path(__file__).resolve()),
            "code_identity": dict(code_identity),
            "legacy_views_manifest_sha256": legacy_digest,
            **dict(frozen),
            "contract_state_sha256": hashlib.sha256(state_bytes).hexdigest(),
            "contract": state,
            "test_unlocked": False,
            "test_labels_opened": False,
            "test_view_materialized": False,
            "week26_file_open_attempted": False,
            "validation_model_metrics_computed": False,
            "test_metrics_computed": False,
            "split": {
                "train_weeks": list(range(20, 25)),
                "validation_week": 25,
                "sealed_test_week": 26,
            },
            "targets": {
                "names": list(TARGET_NAMES),
                "kinds": list(TARGET_KINDS),
                "columns": list(TARGET_COLUMNS),
                "mask_columns": list(TARGET_MASK_COLUMNS),
                "all_labels_observed": True,
            },
            "feature_algebra": {
                "ordered_P_shared": list(ORDERED_OPERAND_FIELDS),
                "ordered_R_div": list(R_DIV_FIELDS),
                "ordered_D": list(DENOMINATOR_ELIGIBLE_FIELDS),
                "ordered_denominator_mask": list(ORDERED_DENOMINATOR_MASK),
                "shared_similarity_fields": list(SHARED_SIMILARITY_FIELDS),
                "REDUCED_physical_schema_sha256": physical_schema_sha256(
                    view_schema("REDUCED")
                ),
                "FULL_physical_schema_sha256": physical_schema_sha256(
                    view_schema("FULL")
                ),
            },
            "row_counts": {
                "train": sum(
                    record["rows"]
                    for record in shard_records
                    if record["role"] == "train"
                ),
                "validation": sum(
                    record["rows"]
                    for record in shard_records
                    if record["role"] == "validation"
                ),
                "total": total_rows,
            },
            "shards": shard_records,
        }
        digest = _publish_manifest(manifest_path, manifest)
        output_published = False
        return digest
    except BaseException:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if output_published:
            _remove_published_output(output_root)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--legacy-views-root", type=Path, required=True)
    parser.add_argument("--raw-dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    digest = materialize(
        repository_root=args.repository_root,
        legacy_manifest_path=args.legacy_manifest,
        legacy_views_root=args.legacy_views_root,
        raw_dataset_root=args.raw_dataset_root,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
    )
    print(f"multi-target views manifest sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
