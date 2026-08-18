"""Verified VK-LSVD full-scan attestation and bounded raw batch source.

Only the already materialized train (weeks 20--24) and validation (week 25)
views are accepted.  This module deliberately has no code path for sealed week
26, does not compute a model metric, and never exposes audit identifiers as
model inputs.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from dcdn.data.vk_lsvd import (
    DENOMINATOR_ELIGIBLE_FIELDS,
    FEATURE_SCHEMA_VERSION,
    MODEL_FORBIDDEN_INPUT_FIELDS,
    ORDERED_C_FIELDS,
    ORDERED_C_PLUS_P_FLAT_SCHEMA,
    ORDERED_C_PLUS_P_PLUS_E_FLAT_SCHEMA,
    ORDERED_DENOMINATOR_MASK,
    ORDERED_E_FIELDS,
    ORDERED_NUMERIC_OPERANDS,
    ORDERED_P0_FIELDS,
    RECIPROCAL_EPSILON,
    TRAIN_WEEKS,
    VALIDATION_WEEK,
    denominator_mask_sha256,
    floor_indicator_name,
    ordered_schema_sha256,
    stable_row_id,
)
from dcdn.data.vk_lsvd_materialize import _validate_manifest_payload
from dcdn.protocol import SHA256_RE
from dcdn.raw import RawFeatureBatch, RawSourceIdentity

DATASET_ID = "vk_lsvd_up0.01_ir0.01"
FEATURE_CONTRACT_SHA256 = (
    "380d105c847a73afb4927740e40401124688f895142792309b4fc8267a2a5f2b"
)
SOURCE_MANIFEST_SHA256 = (
    "5dbbfeb23442a6b1ebdb829dc239c6a3c6c655d71446cde4551b83cc6af456ac"
)
VIEWS_MANIFEST_SHA256 = (
    "738d3610fd83a93297d94ccad4c4316dd3ad8b241f8409e84daf5b2322040dbe"
)
FULL_SCAN_ALGORITHM_VERSION = "vk-lsvd-full-materialized-scan-v2"
CACHE_ALGORITHM_VERSION = "vk-lsvd-raw-memmap-v2"
FROZEN_V2_GENERATOR_SOURCE_SHA256 = (
    "40edf35afc0cddac1ae93c924337985579df96c13270bc2cb6a973397961f5b1"
)
PARQUET_BATCH_ROWS = 262_144
MAX_BATCH_ROWS = 262_144
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_CACHE_METADATA_BYTES = 2 * 1024 * 1024
MAX_SIDECAR_BYTES = 256
MIN_AVAILABLE_BYTES = 20 * 1024**3
MAX_USED_FRACTION = 0.85
MAX_CACHE_BYTES = 8 * 1024**3
LABEL_C_SECONDS = np.float32(157.0)
EXPECTED_CARDINALITIES = (55, 4, 82, 24, 13, 22)
AUDIT_COLUMNS = (
    "user_id",
    "item_id",
    "week",
    "source_row_index0",
    "stable_row_id",
)
TARGET_COLUMNS = ("label_normalized_watch_time", "label_timespent_seconds")
VALIDITY_COLUMNS = tuple(
    f"denominator_validity__{name}" for name in ORDERED_NUMERIC_OPERANDS
)
EXPECTED_WEEKS = (*TRAIN_WEEKS, VALIDATION_WEEK)
SHA256_LINE_RE = re.compile(r"([0-9a-f]{64})  ([^\n]+)\n\Z")
FeatureRegime = Literal["C+P", "C+P+E"]
C_PLUS_P: FeatureRegime = "C+P"
C_PLUS_P_PLUS_E: FeatureRegime = "C+P+E"
FEATURE_REGIMES = (C_PLUS_P, C_PLUS_P_PLUS_E)
C_PLUS_P_PLUS_E_OPERANDS = (*ORDERED_NUMERIC_OPERANDS, *ORDERED_E_FIELDS)
C_PLUS_P_PLUS_E_MASK = (*ORDERED_DENOMINATOR_MASK, *(False for _ in ORDERED_E_FIELDS))


class VKLSVDRuntimeError(RuntimeError):
    """A frozen view, scan, cache, or raw-batch invariant was violated."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise VKLSVDRuntimeError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class RuntimeGuards:
    checked_path: str
    total_bytes: int
    available_bytes: int
    used_fraction: float
    estimated_output_bytes: int


@dataclass(frozen=True)
class _ResolvedShard:
    week: int
    role: Literal["train", "validation"]
    rows: int
    cp_path: Path
    cpe_path: Path
    record: Mapping[str, Any]


def _directory_fd(path: Path, label: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise VKLSVDRuntimeError(f"cannot securely open {label}: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise VKLSVDRuntimeError(f"{label} must be a non-symlink directory")
    return descriptor


def _regular_fd_at(
    directory_fd: int, name: str, label: str
) -> tuple[int, os.stat_result]:
    if not name or Path(name).name != name:
        raise VKLSVDRuntimeError(f"{label} name is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise VKLSVDRuntimeError(f"cannot securely open {label}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise VKLSVDRuntimeError(
            f"{label} must be a single-link non-symlink regular file"
        )
    return descriptor, metadata


def _stable_stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_fd_bytes(descriptor: int, *, maximum_bytes: int, label: str) -> bytes:
    before = os.fstat(descriptor)
    if before.st_size > maximum_bytes:
        raise VKLSVDRuntimeError(f"{label} exceeds size limit")
    chunks: list[bytes] = []
    position = 0
    while position < before.st_size:
        chunk = os.pread(
            descriptor,
            min(1024 * 1024, before.st_size - position),
            position,
        )
        if not chunk:
            raise VKLSVDRuntimeError(f"{label} changed during verified read")
        chunks.append(chunk)
        position += len(chunk)
    after = os.fstat(descriptor)
    if position != before.st_size or _stable_stat_identity(
        before
    ) != _stable_stat_identity(after):
        raise VKLSVDRuntimeError(f"{label} changed during verified read")
    return b"".join(chunks)


def _read_regular_path(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    parent_fd = _directory_fd(path.parent, f"{label} parent")
    try:
        descriptor, _ = _regular_fd_at(parent_fd, path.name, label)
        try:
            return _read_fd_bytes(descriptor, maximum_bytes=maximum_bytes, label=label)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _sha256_file(path: Path) -> str:
    parent_fd = _directory_fd(path.parent, "hashed-file parent")
    try:
        descriptor, before = _regular_fd_at(parent_fd, path.name, "hashed file")
        try:
            digest = hashlib.sha256()
            position = 0
            while position < before.st_size:
                chunk = os.pread(
                    descriptor,
                    min(8 * 1024 * 1024, before.st_size - position),
                    position,
                )
                if not chunk:
                    raise VKLSVDRuntimeError("hashed file changed during read")
                digest.update(chunk)
                position += len(chunk)
            after = os.fstat(descriptor)
            if position != before.st_size or _stable_stat_identity(
                before
            ) != _stable_stat_identity(after):
                raise VKLSVDRuntimeError("hashed file changed during read")
            return digest.hexdigest()
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _sha256_array(digest: Any, label: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes(order="C"))


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise VKLSVDRuntimeError(f"{label} is not a lowercase SHA-256 digest")
    if len(set(value)) == 1:
        raise VKLSVDRuntimeError(f"{label} is a placeholder digest")
    return value


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise VKLSVDRuntimeError(f"cannot inspect {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise VKLSVDRuntimeError(f"{label} must be a non-symlink regular file")
    if metadata.st_nlink != 1:
        raise VKLSVDRuntimeError(f"{label} must have exactly one hard link")
    return metadata


def _safe_relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise VKLSVDRuntimeError(f"{label} path is missing")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise VKLSVDRuntimeError(f"{label} path is unsafe")
    forbidden_components = tuple(part.casefold() for part in relative.parts)
    if any(
        "week_26" in component or "test" in component
        for component in forbidden_components
    ):
        raise VKLSVDRuntimeError(f"{label} attempts to reference sealed test data")
    return relative


def _resolve_beneath(root: Path, relative: PurePosixPath, label: str) -> Path:
    root_metadata = root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise VKLSVDRuntimeError("views root must be a non-symlink directory")
    root_resolved = root.resolve(strict=True)
    candidate = root.joinpath(*relative.parts)
    _require_regular_file(candidate, label)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise VKLSVDRuntimeError(f"{label} escapes views root")
    return resolved


def _parse_sidecar_bytes(data: bytes, *, expected_name: str) -> str:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise VKLSVDRuntimeError("SHA-256 sidecar must be ASCII") from error
    match = SHA256_LINE_RE.fullmatch(text)
    if match is None or match.group(2) != expected_name:
        raise VKLSVDRuntimeError("SHA-256 sidecar is malformed")
    return _require_sha256(match.group(1), "sidecar digest")


def _read_sidecar(path: Path, *, expected_name: str) -> str:
    data = _read_regular_path(
        path, maximum_bytes=MAX_SIDECAR_BYTES, label="SHA-256 sidecar"
    )
    return _parse_sidecar_bytes(data, expected_name=expected_name)


def _load_yaml_with_sidecar(
    path: Path,
    *,
    maximum_bytes: int = MAX_MANIFEST_BYTES,
) -> tuple[dict[str, Any], str]:
    parent_fd = _directory_fd(path.parent, "manifest parent")
    try:
        manifest_fd, _ = _regular_fd_at(parent_fd, path.name, "manifest")
        sidecar_name = f"{path.name}.sha256"
        sidecar_fd, _ = _regular_fd_at(parent_fd, sidecar_name, "SHA-256 sidecar")
        try:
            data = _read_fd_bytes(
                manifest_fd, maximum_bytes=maximum_bytes, label="manifest"
            )
            sidecar_data = _read_fd_bytes(
                sidecar_fd,
                maximum_bytes=MAX_SIDECAR_BYTES,
                label="SHA-256 sidecar",
            )
        finally:
            os.close(sidecar_fd)
            os.close(manifest_fd)
    finally:
        os.close(parent_fd)
    digest = _parse_sidecar_bytes(sidecar_data, expected_name=path.name)
    if hashlib.sha256(data).hexdigest() != digest:
        raise VKLSVDRuntimeError("manifest and sidecar digests differ")
    try:
        payload = yaml.load(data.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise VKLSVDRuntimeError("manifest is not valid unique-key YAML") from error
    if not isinstance(payload, Mapping):
        raise VKLSVDRuntimeError("manifest root must be a mapping")
    return dict(payload), digest


def _dump_yaml(payload: Mapping[str, Any]) -> bytes:
    data = yaml.safe_dump(
        dict(payload), sort_keys=False, allow_unicode=False, default_flow_style=False
    ).encode("utf-8")
    if len(data) > MAX_MANIFEST_BYTES:
        raise VKLSVDRuntimeError("full-scan manifest exceeds size limit")
    return data


def _atomic_write_pair(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    if stat.S_ISLNK(path.parent.lstat().st_mode):
        raise VKLSVDRuntimeError("manifest parent must not be a symlink")
    target = parent / path.name
    sidecar = parent / f"{path.name}.sha256"
    digest = hashlib.sha256(data).hexdigest()
    sidecar_data = f"{digest}  {path.name}\n".encode("ascii")
    temporaries: list[Path] = []
    try:
        for prefix, content in (
            (f".{path.name}.manifest-", data),
            (f".{path.name}.sidecar-", sidecar_data),
        ):
            descriptor, name = tempfile.mkstemp(dir=parent, prefix=prefix)
            temporary = Path(name)
            temporaries.append(temporary)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporaries[0], target)
        os.replace(temporaries[1], sidecar)
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
    return digest


def _physical_schema_sha256(schema: pa.Schema) -> str:
    payload = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _numpy(batch: pa.RecordBatch, field: str) -> np.ndarray:
    index = batch.schema.get_field_index(field)
    if index < 0:
        raise VKLSVDRuntimeError(f"required column is missing: {field}")
    column = batch.column(index)
    if column.null_count:
        raise VKLSVDRuntimeError(f"required column contains nulls: {field}")
    return column.to_numpy(zero_copy_only=False)


def _manifest_digest(path: Path) -> str:
    _require_regular_file(path, "locked manifest")
    return _sha256_file(path)


def _load_views_manifest_verified(path: Path) -> tuple[dict[str, Any], str]:
    payload, digest = _load_yaml_with_sidecar(path)
    try:
        _validate_manifest_payload(payload)
    except Exception as error:
        if isinstance(error, VKLSVDRuntimeError):
            raise
        raise VKLSVDRuntimeError("views manifest payload is invalid") from error
    return payload, digest


def _validate_frozen_manifest_identity(
    payload: Mapping[str, Any],
    views_digest: str,
    *,
    enforce_production_locks: bool,
) -> None:
    if payload.get("dataset_id") != DATASET_ID:
        raise VKLSVDRuntimeError("views dataset identity changed")
    flags = (
        "test_unlocked",
        "test_labels_opened",
        "test_view_materialized",
        "week26_file_open_attempted",
        "validation_model_metrics_computed",
        "test_metrics_computed",
    )
    if any(type(payload.get(name)) is not bool or payload[name] for name in flags):
        raise VKLSVDRuntimeError(
            "views manifest reports forbidden test/metric activity"
        )
    if payload.get("feature_contract_sha256") != FEATURE_CONTRACT_SHA256:
        raise VKLSVDRuntimeError("views feature-contract lock changed")
    if payload.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise VKLSVDRuntimeError("views source-manifest lock changed")
    if enforce_production_locks and views_digest != VIEWS_MANIFEST_SHA256:
        raise VKLSVDRuntimeError("views manifest digest changed from frozen value")

    split = payload.get("split")
    if not isinstance(split, Mapping):
        raise VKLSVDRuntimeError("views split contract is missing")
    if tuple(split.get("train_weeks", ())) != TRAIN_WEEKS:
        raise VKLSVDRuntimeError("views train weeks changed")
    if split.get("validation_week") != VALIDATION_WEEK:
        raise VKLSVDRuntimeError("views validation week changed")
    if split.get("sealed_test_week") != 26:
        raise VKLSVDRuntimeError("sealed test-week declaration changed")

    algebra = payload.get("feature_algebra")
    if not isinstance(algebra, Mapping):
        raise VKLSVDRuntimeError("views feature algebra is missing")
    expected_sequences = {
        "ordered_C": ORDERED_C_FIELDS,
        "ordered_P": ORDERED_NUMERIC_OPERANDS,
        "ordered_E": ORDERED_E_FIELDS,
        "ordered_denominator_mask": ORDERED_DENOMINATOR_MASK,
        "denominator_eligible_fields": DENOMINATOR_ELIGIBLE_FIELDS,
    }
    for key, expected in expected_sequences.items():
        if tuple(algebra.get(key, ())) != tuple(expected):
            raise VKLSVDRuntimeError(f"views {key} changed")
    if algebra.get("dimensions") != {"C": 6, "P0": 31, "V": 27, "P": 58, "E": 20}:
        raise VKLSVDRuntimeError("views C/P/E dimensions changed")
    if float(algebra.get("denominator_epsilon", -1.0)) != RECIPROCAL_EPSILON:
        raise VKLSVDRuntimeError("views reciprocal epsilon changed")

    mappings = payload.get("train_only_fit", {}).get("categorical_mappings", {})
    cardinalities = tuple(
        mappings.get(name, {}).get("cardinality_with_reserved_tokens")
        for name in ORDERED_C_FIELDS
    )
    if cardinalities != EXPECTED_CARDINALITIES:
        raise VKLSVDRuntimeError("views categorical cardinalities changed")


def _resolve_shards(
    payload: Mapping[str, Any], views_root: Path
) -> list[_ResolvedShard]:
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != 6:
        raise VKLSVDRuntimeError(
            "exactly six paired train/validation shards are required"
        )
    resolved: list[_ResolvedShard] = []
    for position, raw in enumerate(raw_shards):
        if not isinstance(raw, Mapping):
            raise VKLSVDRuntimeError("shard record must be a mapping")
        week = raw.get("week")
        role = raw.get("role")
        expected_week = EXPECTED_WEEKS[position]
        expected_role = "train" if position < len(TRAIN_WEEKS) else "validation"
        if week != expected_week or role != expected_role:
            raise VKLSVDRuntimeError("shards are not in frozen week/role order")
        if week == 26 or role == "test":
            raise VKLSVDRuntimeError("sealed test shard is forbidden")
        rows = raw.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise VKLSVDRuntimeError("shard rows must be positive integers")
        cp = raw.get("C_plus_P")
        cpe = raw.get("C_plus_P_plus_E")
        if not isinstance(cp, Mapping) or not isinstance(cpe, Mapping):
            raise VKLSVDRuntimeError("paired shard entries are missing")
        cp_path = _resolve_beneath(
            views_root, _safe_relative_path(cp.get("path"), "C+P"), "C+P shard"
        )
        cpe_path = _resolve_beneath(
            views_root,
            _safe_relative_path(cpe.get("path"), "C+P+E"),
            "C+P+E shard",
        )
        if cp_path == cpe_path:
            raise VKLSVDRuntimeError("paired regimes cannot reuse one physical file")
        resolved.append(
            _ResolvedShard(
                week=int(week),
                role=role,
                rows=rows,
                cp_path=cp_path,
                cpe_path=cpe_path,
                record=raw,
            )
        )
    expected_counts = payload.get("row_counts")
    train_rows = sum(item.rows for item in resolved if item.role == "train")
    validation_rows = sum(item.rows for item in resolved if item.role == "validation")
    if expected_counts != {
        "train": train_rows,
        "validation": validation_rows,
        "total": train_rows + validation_rows,
    }:
        raise VKLSVDRuntimeError("views aggregate row counts changed")
    return resolved


def _expected_schema(*, include_engineered: bool) -> pa.Schema:
    fields = [
        pa.field("user_id", pa.uint32(), nullable=False),
        pa.field("item_id", pa.uint32(), nullable=False),
        pa.field("week", pa.uint8(), nullable=False),
        pa.field("source_row_index0", pa.uint64(), nullable=False),
        pa.field("stable_row_id", pa.string(), nullable=False),
        pa.field("label_normalized_watch_time", pa.float32(), nullable=False),
        pa.field("label_timespent_seconds", pa.float32(), nullable=False),
        *(pa.field(name, pa.uint32(), nullable=False) for name in ORDERED_C_FIELDS),
        *(
            pa.field(name, pa.float32(), nullable=False)
            for name in ORDERED_NUMERIC_OPERANDS
        ),
        *(
            pa.field(name, pa.float32(), nullable=False)
            for name in ORDERED_E_FIELDS
            if include_engineered
        ),
        *(pa.field(name, pa.bool_(), nullable=False) for name in VALIDITY_COLUMNS),
    ]
    regime = "C+P+E" if include_engineered else "C+P"
    return pa.schema(
        fields,
        metadata={
            b"dcdn.dataset": DATASET_ID.encode("ascii"),
            b"dcdn.feature_contract_sha256": FEATURE_CONTRACT_SHA256.encode("ascii"),
            b"dcdn.regime": regime.encode("ascii"),
            b"dcdn.test_unlocked": b"false",
        },
    )


def _initial_denominator_scan() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "rows": 0,
            "valid_rows": 0,
            "invalid_or_zero_rows": 0,
            "reciprocal_floor_rows": 0,
            "nonfinite_rows": 0,
            "minimum_positive_normalized": None,
            "maximum_normalized": None,
        }
        for name in DENOMINATOR_ELIGIBLE_FIELDS
    }


def _update_denominator_scan(
    scan: dict[str, dict[str, Any]],
    p: Mapping[str, np.ndarray],
    validity: Mapping[str, np.ndarray],
) -> None:
    for name in DENOMINATOR_ELIGIBLE_FIELDS:
        values = p[name]
        valid = validity[name]
        nonfinite = ~np.isfinite(values)
        if np.any(nonfinite):
            raise VKLSVDRuntimeError(f"non-finite denominator operand: {name}")
        if np.any(values < 0.0):
            raise VKLSVDRuntimeError(f"negative denominator operand: {name}")
        if np.any(valid & (values <= 0.0)):
            raise VKLSVDRuntimeError(f"active denominator is non-positive: {name}")
        if np.any((~valid) & (values != 0.0)):
            raise VKLSVDRuntimeError(
                f"inactive denominator is not canonical zero: {name}"
            )
        indicator = p[floor_indicator_name(name)]
        expected_indicator = (valid & (values < RECIPROCAL_EPSILON)).astype(np.float32)
        if not np.array_equal(indicator, expected_indicator):
            raise VKLSVDRuntimeError(f"reciprocal-floor indicator mismatch: {name}")
        item = scan[name]
        item["rows"] += len(values)
        item["valid_rows"] += int(valid.sum())
        item["invalid_or_zero_rows"] += int((~valid).sum())
        item["reciprocal_floor_rows"] += int(
            (valid & (values < RECIPROCAL_EPSILON)).sum()
        )
        item["nonfinite_rows"] += int(nonfinite.sum())
        if np.any(valid):
            minimum = float(values[valid].min())
            maximum = float(values[valid].max())
            old_min = item["minimum_positive_normalized"]
            old_max = item["maximum_normalized"]
            item["minimum_positive_normalized"] = (
                minimum if old_min is None else min(old_min, minimum)
            )
            item["maximum_normalized"] = (
                maximum if old_max is None else max(old_max, maximum)
            )


def _empty_numeric_range() -> dict[str, float | None]:
    return {"minimum": None, "maximum": None}


def _update_range(record: dict[str, float | None], values: np.ndarray) -> None:
    minimum = float(values.min())
    maximum = float(values.max())
    current_min = record["minimum"]
    current_max = record["maximum"]
    record["minimum"] = minimum if current_min is None else min(current_min, minimum)
    record["maximum"] = maximum if current_max is None else max(current_max, maximum)


def _scan_pair(shard: _ResolvedShard, algebra: Mapping[str, Any]) -> dict[str, Any]:
    cp_record = shard.record["C_plus_P"]
    cpe_record = shard.record["C_plus_P_plus_E"]
    cp_meta = _require_regular_file(shard.cp_path, "C+P shard")
    cpe_meta = _require_regular_file(shard.cpe_path, "C+P+E shard")
    cp_digest = _sha256_file(shard.cp_path)
    cpe_digest = _sha256_file(shard.cpe_path)
    for name, actual, metadata, record in (
        ("C+P", cp_digest, cp_meta, cp_record),
        ("C+P+E", cpe_digest, cpe_meta, cpe_record),
    ):
        if actual != _require_sha256(record.get("sha256"), f"{name} shard SHA"):
            raise VKLSVDRuntimeError(f"{name} shard hash drift")
        if metadata.st_size != record.get("bytes"):
            raise VKLSVDRuntimeError(f"{name} shard byte-size drift")

    expected_cp_schema = _expected_schema(include_engineered=False)
    expected_cpe_schema = _expected_schema(include_engineered=True)
    cp_file = pq.ParquetFile(shard.cp_path)
    cpe_file = pq.ParquetFile(shard.cpe_path)
    for name, parquet, expected, record in (
        ("C+P", cp_file, expected_cp_schema, cp_record),
        ("C+P+E", cpe_file, expected_cpe_schema, cpe_record),
    ):
        if parquet.metadata.num_rows != shard.rows:
            raise VKLSVDRuntimeError(f"{name} row-count drift")
        if parquet.metadata.num_row_groups != record.get("row_groups"):
            raise VKLSVDRuntimeError(f"{name} row-group drift")
        if not parquet.schema_arrow.equals(expected, check_metadata=True):
            raise VKLSVDRuntimeError(f"{name} physical schema drift")
        schema_key = f"{name.replace('+', '_plus_')}_physical_schema_sha256"
        if _physical_schema_sha256(parquet.schema_arrow) != algebra.get(schema_key):
            raise VKLSVDRuntimeError(f"{name} physical schema hash drift")

    shared_fields = (
        *AUDIT_COLUMNS,
        *TARGET_COLUMNS,
        *ORDERED_C_FIELDS,
        *ORDERED_NUMERIC_OPERANDS,
        *VALIDITY_COLUMNS,
    )
    cpe_fields = (*shared_fields, *ORDERED_E_FIELDS)
    cp_batches = cp_file.iter_batches(
        batch_size=PARQUET_BATCH_ROWS, columns=list(shared_fields), use_threads=False
    )
    cpe_batches = cpe_file.iter_batches(
        batch_size=PARQUET_BATCH_ROWS, columns=list(cpe_fields), use_threads=False
    )
    shared_digest = hashlib.sha256()
    paired_shared_digest = hashlib.sha256()
    row_id_digest = hashlib.sha256()
    paired_row_id_digest = hashlib.sha256()
    denominator_scan = _initial_denominator_scan()
    token_min = {name: None for name in ORDERED_C_FIELDS}
    token_max = {name: None for name in ORDERED_C_FIELDS}
    p_ranges = {name: _empty_numeric_range() for name in ORDERED_NUMERIC_OPERANDS}
    e_ranges = {name: _empty_numeric_range() for name in ORDERED_E_FIELDS}
    target_ranges = {name: _empty_numeric_range() for name in TARGET_COLUMNS}
    rows = 0
    while True:
        try:
            cp_batch = next(cp_batches)
        except StopIteration:
            cp_batch = None
        try:
            cpe_batch = next(cpe_batches)
        except StopIteration:
            cpe_batch = None
        if cp_batch is None or cpe_batch is None:
            if cp_batch is not None or cpe_batch is not None:
                raise VKLSVDRuntimeError("paired shards have different batch counts")
            break
        if cp_batch.num_rows != cpe_batch.num_rows:
            raise VKLSVDRuntimeError("paired shards have different batch lengths")
        count = cp_batch.num_rows
        if count <= 0:
            raise VKLSVDRuntimeError("empty Parquet record batch")

        cp_values = {name: _numpy(cp_batch, name) for name in shared_fields}
        cpe_values = {name: _numpy(cpe_batch, name) for name in cpe_fields}
        for name in shared_fields:
            equal_nan = np.issubdtype(cp_values[name].dtype, np.floating)
            if not np.array_equal(
                cp_values[name], cpe_values[name], equal_nan=equal_nan
            ):
                raise VKLSVDRuntimeError(f"paired C+P shared-value mismatch: {name}")

        weeks = cp_values["week"]
        indices = cp_values["source_row_index0"]
        if not np.all(weeks == shard.week):
            raise VKLSVDRuntimeError("physical rows do not match declared week")
        expected_indices = np.arange(rows, rows + count, dtype=np.uint64)
        if not np.array_equal(indices, expected_indices):
            raise VKLSVDRuntimeError("source_row_index0 is not contiguous from zero")
        stable_ids = cp_values["stable_row_id"]
        paired_ids = cpe_values["stable_row_id"]
        for offset, (value, paired) in enumerate(
            zip(stable_ids, paired_ids, strict=True)
        ):
            expected = stable_row_id(week=shard.week, source_row_index0=rows + offset)
            if value != expected or paired != expected:
                raise VKLSVDRuntimeError("stable row identity mismatch")
            row_id_digest.update(str(value).encode("ascii") + b"\n")
            paired_row_id_digest.update(str(paired).encode("ascii") + b"\n")

        p = {}
        validity = {}
        for name in ORDERED_C_FIELDS:
            values = cp_values[name]
            if values.dtype != np.dtype("uint32"):
                raise VKLSVDRuntimeError(f"context token dtype drift: {name}")
            minimum = int(values.min())
            maximum = int(values.max())
            position = ORDERED_C_FIELDS.index(name)
            if minimum < 0 or maximum >= EXPECTED_CARDINALITIES[position]:
                raise VKLSVDRuntimeError(f"context token range violation: {name}")
            token_min[name] = (
                minimum if token_min[name] is None else min(token_min[name], minimum)
            )
            token_max[name] = (
                maximum if token_max[name] is None else max(token_max[name], maximum)
            )
        for name in ORDERED_NUMERIC_OPERANDS:
            values = cp_values[name]
            if values.dtype != np.dtype("float32") or not np.isfinite(values).all():
                raise VKLSVDRuntimeError(f"operand is not finite FP32: {name}")
            p[name] = values
            _update_range(p_ranges[name], values)
            valid = cp_values[f"denominator_validity__{name}"]
            if valid.dtype != np.dtype("bool"):
                raise VKLSVDRuntimeError(f"validity dtype drift: {name}")
            validity[name] = valid
            if name not in DENOMINATOR_ELIGIBLE_FIELDS and not bool(valid.all()):
                raise VKLSVDRuntimeError(
                    f"non-denominator validity is not all true: {name}"
                )
        _update_denominator_scan(denominator_scan, p, validity)

        for name in ORDERED_E_FIELDS:
            values = cpe_values[name]
            if values.dtype != np.dtype("float32") or not np.isfinite(values).all():
                raise VKLSVDRuntimeError(
                    f"engineered feature is not finite FP32: {name}"
                )
            _update_range(e_ranges[name], values)
        normalized = cp_values["label_normalized_watch_time"]
        seconds = cp_values["label_timespent_seconds"]
        if (
            normalized.dtype != np.dtype("float32")
            or seconds.dtype != np.dtype("float32")
            or not np.isfinite(normalized).all()
            or not np.isfinite(seconds).all()
        ):
            raise VKLSVDRuntimeError("targets are not finite FP32")
        if np.any((normalized < 0.0) | (normalized > 1.0)):
            raise VKLSVDRuntimeError("normalized target is outside [0,1]")
        if np.any((seconds < 0.0) | (seconds > LABEL_C_SECONDS)):
            raise VKLSVDRuntimeError("seconds target is outside [0,157]")
        expected_normalized = (seconds / LABEL_C_SECONDS).astype(np.float32)
        if not np.array_equal(normalized, expected_normalized):
            raise VKLSVDRuntimeError("normalized target transform mismatch")
        _update_range(target_ranges[TARGET_COLUMNS[0]], normalized)
        _update_range(target_ranges[TARGET_COLUMNS[1]], seconds)

        for name in ORDERED_C_FIELDS:
            _sha256_array(shared_digest, name, cp_values[name])
            _sha256_array(paired_shared_digest, name, cpe_values[name])
        for name in ORDERED_NUMERIC_OPERANDS:
            _sha256_array(shared_digest, name, cp_values[name])
            _sha256_array(paired_shared_digest, name, cpe_values[name])
        rows += count

    if rows != shard.rows:
        raise VKLSVDRuntimeError("streamed row count differs from shard manifest")
    expected_shared = _require_sha256(
        shard.record.get("shared_C_plus_P_values_sha256"), "shared-value SHA"
    )
    if (
        shared_digest.hexdigest() != expected_shared
        or paired_shared_digest.hexdigest() != expected_shared
    ):
        raise VKLSVDRuntimeError("canonical shared C+P value hash drift")
    expected_ids = _require_sha256(
        shard.record.get("stable_row_ids_sha256"), "stable-row-ID SHA"
    )
    if (
        row_id_digest.hexdigest() != expected_ids
        or paired_row_id_digest.hexdigest() != expected_ids
    ):
        raise VKLSVDRuntimeError("stable-row-ID hash drift")

    return {
        "week": shard.week,
        "role": shard.role,
        "rows": rows,
        "C_plus_P": {
            "path": shard.record["C_plus_P"]["path"],
            "bytes": cp_meta.st_size,
            "sha256": cp_digest,
            "row_groups": cp_file.metadata.num_row_groups,
        },
        "C_plus_P_plus_E": {
            "path": shard.record["C_plus_P_plus_E"]["path"],
            "bytes": cpe_meta.st_size,
            "sha256": cpe_digest,
            "row_groups": cpe_file.metadata.num_row_groups,
        },
        "stable_row_ids_sha256": row_id_digest.hexdigest(),
        "paired_stable_row_ids_sha256": paired_row_id_digest.hexdigest(),
        "shared_C_plus_P_values_sha256": shared_digest.hexdigest(),
        "paired_shared_C_plus_P_values_sha256": paired_shared_digest.hexdigest(),
        "token_bounds": {
            name: {
                "minimum": token_min[name],
                "maximum": token_max[name],
                "cardinality": EXPECTED_CARDINALITIES[position],
            }
            for position, name in enumerate(ORDERED_C_FIELDS)
        },
        "P_ranges": p_ranges,
        "E_ranges": e_ranges,
        "target_ranges": target_ranges,
        "denominator_scan": denominator_scan,
    }


def _merge_scan_ranges(
    shard_records: Sequence[Mapping[str, Any]], section: str, fields: Sequence[str]
) -> dict[str, dict[str, float | None]]:
    result = {name: _empty_numeric_range() for name in fields}
    for shard in shard_records:
        for name in fields:
            values = shard[section][name]
            for key in ("minimum", "maximum"):
                value = values[key]
                current = result[name][key]
                if current is None:
                    result[name][key] = value
                elif key == "minimum":
                    result[name][key] = min(current, value)
                else:
                    result[name][key] = max(current, value)
    return result


def _merge_denominator_scan(
    shard_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = _initial_denominator_scan()
    for shard in shard_records:
        for name in DENOMINATOR_ELIGIBLE_FIELDS:
            source = shard["denominator_scan"][name]
            target = result[name]
            for key in (
                "rows",
                "valid_rows",
                "invalid_or_zero_rows",
                "reciprocal_floor_rows",
                "nonfinite_rows",
            ):
                target[key] += source[key]
            for key, operation in (
                ("minimum_positive_normalized", min),
                ("maximum_normalized", max),
            ):
                value = source[key]
                current = target[key]
                if value is not None:
                    target[key] = (
                        value if current is None else operation(current, value)
                    )
    return result


def build_full_scan_attestation(
    *,
    views_manifest_path: Path,
    views_root: Path,
    output_manifest_path: Path,
    source_manifest_path: Path | None = None,
    feature_contract_path: Path | None = None,
    enforce_production_locks: bool = True,
) -> tuple[dict[str, Any], str]:
    """Reopen and stream every explicit train/validation view and attest it."""

    views_payload, views_digest = _load_views_manifest_verified(views_manifest_path)
    _validate_frozen_manifest_identity(
        views_payload, views_digest, enforce_production_locks=enforce_production_locks
    )
    if enforce_production_locks:
        if source_manifest_path is None or feature_contract_path is None:
            raise VKLSVDRuntimeError(
                "production scan requires source and feature manifests"
            )
        if _manifest_digest(source_manifest_path) != SOURCE_MANIFEST_SHA256:
            raise VKLSVDRuntimeError("source manifest drift")
        if _manifest_digest(feature_contract_path) != FEATURE_CONTRACT_SHA256:
            raise VKLSVDRuntimeError("feature contract drift")
    algebra = views_payload["feature_algebra"]
    resolved = _resolve_shards(views_payload, views_root)
    scanned = [_scan_pair(shard, algebra) for shard in resolved]
    denominator_scan = _merge_denominator_scan(scanned)
    if denominator_scan != views_payload.get("denominator_scan"):
        raise VKLSVDRuntimeError("full denominator scan differs from views manifest")

    token_bounds = {}
    for position, name in enumerate(ORDERED_C_FIELDS):
        token_bounds[name] = {
            "minimum": min(
                record["token_bounds"][name]["minimum"] for record in scanned
            ),
            "maximum": max(
                record["token_bounds"][name]["maximum"] for record in scanned
            ),
            "cardinality": EXPECTED_CARDINALITIES[position],
        }
    row_counts = {
        "train": sum(record["rows"] for record in scanned if record["role"] == "train"),
        "validation": sum(
            record["rows"] for record in scanned if record["role"] == "validation"
        ),
    }
    row_counts["total"] = row_counts["train"] + row_counts["validation"]
    if row_counts != views_payload["row_counts"]:
        raise VKLSVDRuntimeError("full scan row totals changed")

    forbidden_present = sorted(
        (set(MODEL_FORBIDDEN_INPUT_FIELDS) | set(AUDIT_COLUMNS) | set(TARGET_COLUMNS))
        & (
            set(ORDERED_C_FIELDS)
            | set(ORDERED_NUMERIC_OPERANDS)
            | set(ORDERED_E_FIELDS)
        )
    )
    if forbidden_present:
        raise VKLSVDRuntimeError("audit/outcome column appears in model inputs")
    physical_hashes = {
        "C_plus_P": algebra["C_plus_P_physical_schema_sha256"],
        "C_plus_P_plus_E": algebra["C_plus_P_plus_E_physical_schema_sha256"],
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "artifact_role": "canonical_full_materialized_train_validation_scan",
        "scan_state": "complete_and_hash_verified",
        "algorithm_version": FULL_SCAN_ALGORITHM_VERSION,
        "algorithm_source_sha256": _sha256_file(Path(__file__).resolve()),
        "source_manifest_sha256": views_payload["source_manifest_sha256"],
        "views_manifest_sha256": views_digest,
        "feature_contract_sha256": views_payload["feature_contract_sha256"],
        "test_unlocked": False,
        "test_labels_opened": False,
        "test_view_opened": False,
        "week26_path_constructed": False,
        "model_metrics_computed": False,
        "scan_scope": {
            "roles": ["train", "validation"],
            "weeks": list(EXPECTED_WEEKS),
            "paired_regimes": ["C+P", "C+P+E"],
            "parquet_batch_rows": PARQUET_BATCH_ROWS,
        },
        "schema_identity": {
            "ordered_C": list(ORDERED_C_FIELDS),
            "ordered_P": list(ORDERED_NUMERIC_OPERANDS),
            "ordered_E": list(ORDERED_E_FIELDS),
            "dimensions": {"C": 6, "P0": 31, "V": 27, "P": 58, "E": 20},
            "physical_schema_sha256": physical_hashes,
            "expanded_schema_sha256": {
                "C_plus_P": algebra["C_plus_P_expanded_schema_sha256"],
                "C_plus_P_plus_E": algebra["C_plus_P_plus_E_expanded_schema_sha256"],
            },
            "operand_schema_sha256": {
                "C_plus_P": algebra["ordered_operand_schema_sha256"],
                "C_plus_P_plus_E": ordered_schema_sha256(C_PLUS_P_PLUS_E_OPERANDS),
            },
            "denominator_mask_sha256": {
                "C_plus_P": denominator_mask_sha256(ORDERED_DENOMINATOR_MASK),
                "C_plus_P_plus_E": denominator_mask_sha256(C_PLUS_P_PLUS_E_MASK),
            },
            "ordered_denominator_mask": {
                "C_plus_P": list(ORDERED_DENOMINATOR_MASK),
                "C_plus_P_plus_E": list(C_PLUS_P_PLUS_E_MASK),
            },
        },
        "input_boundary": {
            "model_C_plus_P_fields": [*ORDERED_C_FIELDS, *ORDERED_NUMERIC_OPERANDS],
            "model_C_plus_P_plus_E_fields": [
                *ORDERED_C_FIELDS,
                *ORDERED_NUMERIC_OPERANDS,
                *ORDERED_E_FIELDS,
            ],
            "audit_only_fields": list(AUDIT_COLUMNS),
            "target_only_fields": list(TARGET_COLUMNS),
            "forbidden_model_input_intersection": [],
            "current_outcomes_consumed_as_features": False,
        },
        "row_counts": row_counts,
        "token_bounds": token_bounds,
        "numeric_ranges": {
            "P": _merge_scan_ranges(scanned, "P_ranges", ORDERED_NUMERIC_OPERANDS),
            "E": _merge_scan_ranges(scanned, "E_ranges", ORDERED_E_FIELDS),
            "targets": _merge_scan_ranges(scanned, "target_ranges", TARGET_COLUMNS),
            "all_P_finite": True,
            "all_E_finite": True,
            "all_targets_finite": True,
        },
        "normalization_provenance": {
            "fit_split": "train_weeks_20_24",
            "method": views_payload["train_only_fit"]["normalization"]["method"],
            "quantile": views_payload["train_only_fit"]["normalization"]["quantile"],
            "centering": views_payload["train_only_fit"]["normalization"]["centering"],
            "clipping": views_payload["train_only_fit"]["normalization"]["clipping"],
            "P0_scales": views_payload["train_only_fit"]["normalization"]["P0_scales"],
            "E_scales": views_payload["train_only_fit"]["normalization"]["E_scales"],
            "label_c_seconds": float(LABEL_C_SECONDS),
        },
        "denominator_semantics": {
            "eligible_fields": list(DENOMINATOR_ELIGIBLE_FIELDS),
            "epsilon": RECIPROCAL_EPSILON,
            "noneligible_validity_always_true": True,
            "active_values_strictly_positive": True,
            "inactive_values_canonical_zero": True,
            "floor_indicators_exact": True,
            "scan": denominator_scan,
        },
        "shards": scanned,
    }
    digest = _atomic_write_pair(output_manifest_path, _dump_yaml(payload))
    loaded, loaded_digest = load_full_scan_attestation(
        output_manifest_path,
        expected_scan_sha256=digest,
        enforce_production_locks=enforce_production_locks,
    )
    if loaded_digest != digest or loaded["row_counts"] != row_counts:
        raise VKLSVDRuntimeError("full-scan manifest reopen verification failed")
    return payload, digest


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VKLSVDRuntimeError(f"{label} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        extra = sorted(actual - expected_set)
        raise VKLSVDRuntimeError(
            f"{label} keys changed (missing={missing}, extra={extra})"
        )


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VKLSVDRuntimeError(f"{label} must be a positive integer")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VKLSVDRuntimeError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise VKLSVDRuntimeError(f"{label} must be finite")
    return result


def _validate_range_map(
    value: Any, fields: Sequence[str], label: str
) -> Mapping[str, Any]:
    result = _require_mapping(value, label)
    _require_exact_keys(result, fields, label)
    for field in fields:
        record = _require_mapping(result[field], f"{label}.{field}")
        _require_exact_keys(record, ("minimum", "maximum"), f"{label}.{field}")
        minimum = _require_finite_number(record["minimum"], f"{label}.{field}.minimum")
        maximum = _require_finite_number(record["maximum"], f"{label}.{field}.maximum")
        if minimum > maximum:
            raise VKLSVDRuntimeError(f"{label}.{field} has reversed bounds")
    return result


def _validate_token_bounds(value: Any, label: str) -> Mapping[str, Any]:
    result = _require_mapping(value, label)
    _require_exact_keys(result, ORDERED_C_FIELDS, label)
    for position, field in enumerate(ORDERED_C_FIELDS):
        record = _require_mapping(result[field], f"{label}.{field}")
        _require_exact_keys(
            record, ("minimum", "maximum", "cardinality"), f"{label}.{field}"
        )
        minimum = record["minimum"]
        maximum = record["maximum"]
        cardinality = record["cardinality"]
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (minimum, maximum, cardinality)
        ):
            raise VKLSVDRuntimeError(f"{label}.{field} bounds must be integers")
        if cardinality != EXPECTED_CARDINALITIES[position]:
            raise VKLSVDRuntimeError(f"{label}.{field} cardinality changed")
        if not 0 <= minimum <= maximum < cardinality:
            raise VKLSVDRuntimeError(f"{label}.{field} token bounds are invalid")
    return result


_DENOMINATOR_RECORD_KEYS = (
    "rows",
    "valid_rows",
    "invalid_or_zero_rows",
    "reciprocal_floor_rows",
    "nonfinite_rows",
    "minimum_positive_normalized",
    "maximum_normalized",
)


def _validate_denominator_scan(
    value: Any, *, expected_rows: int, label: str
) -> Mapping[str, Any]:
    result = _require_mapping(value, label)
    _require_exact_keys(result, DENOMINATOR_ELIGIBLE_FIELDS, label)
    for field in DENOMINATOR_ELIGIBLE_FIELDS:
        record = _require_mapping(result[field], f"{label}.{field}")
        _require_exact_keys(record, _DENOMINATOR_RECORD_KEYS, f"{label}.{field}")
        counts: dict[str, int] = {}
        for key in _DENOMINATOR_RECORD_KEYS[:5]:
            item = record[key]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise VKLSVDRuntimeError(
                    f"{label}.{field}.{key} must be a non-negative integer"
                )
            counts[key] = item
        if counts["rows"] != expected_rows:
            raise VKLSVDRuntimeError(f"{label}.{field}.rows changed")
        if counts["valid_rows"] + counts["invalid_or_zero_rows"] != expected_rows:
            raise VKLSVDRuntimeError(f"{label}.{field} validity counts do not sum")
        if counts["reciprocal_floor_rows"] > counts["valid_rows"]:
            raise VKLSVDRuntimeError(f"{label}.{field} floor count exceeds valid")
        if counts["nonfinite_rows"] != 0:
            raise VKLSVDRuntimeError(f"{label}.{field} has non-finite evidence")
        minimum = record["minimum_positive_normalized"]
        maximum = record["maximum_normalized"]
        if counts["valid_rows"]:
            minimum_value = _require_finite_number(
                minimum, f"{label}.{field}.minimum_positive_normalized"
            )
            maximum_value = _require_finite_number(
                maximum, f"{label}.{field}.maximum_normalized"
            )
            if minimum_value <= 0 or minimum_value > maximum_value:
                raise VKLSVDRuntimeError(
                    f"{label}.{field} active-positive evidence is invalid"
                )
        elif minimum is not None or maximum is not None:
            raise VKLSVDRuntimeError(
                f"{label}.{field} empty-valid extrema must be null"
            )
    return result


def _validate_scan_schema_identity(value: Any) -> Mapping[str, Any]:
    schema = _require_mapping(value, "full-scan schema_identity")
    _require_exact_keys(
        schema,
        (
            "ordered_C",
            "ordered_P",
            "ordered_E",
            "dimensions",
            "physical_schema_sha256",
            "expanded_schema_sha256",
            "operand_schema_sha256",
            "denominator_mask_sha256",
            "ordered_denominator_mask",
        ),
        "full-scan schema_identity",
    )
    for key, expected in {
        "ordered_C": ORDERED_C_FIELDS,
        "ordered_P": ORDERED_NUMERIC_OPERANDS,
        "ordered_E": ORDERED_E_FIELDS,
    }.items():
        if tuple(schema[key]) != tuple(expected):
            raise VKLSVDRuntimeError(f"full-scan {key} changed")
    if schema["dimensions"] != {"C": 6, "P0": 31, "V": 27, "P": 58, "E": 20}:
        raise VKLSVDRuntimeError("full-scan dimensions changed")
    expected_maps = {
        "physical_schema_sha256": {
            "C_plus_P": _physical_schema_sha256(
                _expected_schema(include_engineered=False)
            ),
            "C_plus_P_plus_E": _physical_schema_sha256(
                _expected_schema(include_engineered=True)
            ),
        },
        "expanded_schema_sha256": {
            "C_plus_P": ordered_schema_sha256(ORDERED_C_PLUS_P_FLAT_SCHEMA),
            "C_plus_P_plus_E": ordered_schema_sha256(
                ORDERED_C_PLUS_P_PLUS_E_FLAT_SCHEMA
            ),
        },
        "operand_schema_sha256": {
            "C_plus_P": ordered_schema_sha256(ORDERED_NUMERIC_OPERANDS),
            "C_plus_P_plus_E": ordered_schema_sha256(C_PLUS_P_PLUS_E_OPERANDS),
        },
        "denominator_mask_sha256": {
            "C_plus_P": denominator_mask_sha256(ORDERED_DENOMINATOR_MASK),
            "C_plus_P_plus_E": denominator_mask_sha256(C_PLUS_P_PLUS_E_MASK),
        },
        "ordered_denominator_mask": {
            "C_plus_P": list(ORDERED_DENOMINATOR_MASK),
            "C_plus_P_plus_E": list(C_PLUS_P_PLUS_E_MASK),
        },
    }
    for key, expected in expected_maps.items():
        if schema[key] != expected:
            raise VKLSVDRuntimeError(f"full-scan {key} changed")
    return schema


def _validate_scan_shards(
    value: Any,
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    if not isinstance(value, list) or len(value) != 6:
        raise VKLSVDRuntimeError("full-scan must contain exactly six paired shards")
    shards: list[Mapping[str, Any]] = []
    seen_paths: set[str] = set()
    file_hash_count = 0
    counts = {"train": 0, "validation": 0}
    shard_keys = (
        "week",
        "role",
        "rows",
        "C_plus_P",
        "C_plus_P_plus_E",
        "stable_row_ids_sha256",
        "paired_stable_row_ids_sha256",
        "shared_C_plus_P_values_sha256",
        "paired_shared_C_plus_P_values_sha256",
        "token_bounds",
        "P_ranges",
        "E_ranges",
        "target_ranges",
        "denominator_scan",
    )
    for position, item in enumerate(value):
        shard = _require_mapping(item, f"full-scan shard {position}")
        _require_exact_keys(shard, shard_keys, f"full-scan shard {position}")
        week = EXPECTED_WEEKS[position]
        role = "train" if position < len(TRAIN_WEEKS) else "validation"
        if shard["week"] != week or shard["role"] != role:
            raise VKLSVDRuntimeError("full-scan shard week/role order changed")
        rows = _require_positive_int(shard["rows"], f"week {week} rows")
        counts[role] += rows
        for regime_key in ("C_plus_P", "C_plus_P_plus_E"):
            record = _require_mapping(shard[regime_key], f"week {week} {regime_key}")
            _require_exact_keys(
                record,
                ("path", "bytes", "sha256", "row_groups"),
                f"week {week} {regime_key}",
            )
            relative = _safe_relative_path(record["path"], f"week {week} {regime_key}")
            expected_path = PurePosixPath(regime_key, role, f"week_{week}.parquet")
            if relative != expected_path:
                raise VKLSVDRuntimeError(f"week {week} {regime_key} path changed")
            if relative.as_posix() in seen_paths:
                raise VKLSVDRuntimeError("full-scan shard path is duplicated")
            seen_paths.add(relative.as_posix())
            _require_positive_int(record["bytes"], f"week {week} {regime_key} bytes")
            _require_positive_int(
                record["row_groups"], f"week {week} {regime_key} row_groups"
            )
            _require_sha256(record["sha256"], f"week {week} {regime_key} SHA")
            file_hash_count += 1
        for key in (
            "stable_row_ids_sha256",
            "paired_stable_row_ids_sha256",
            "shared_C_plus_P_values_sha256",
            "paired_shared_C_plus_P_values_sha256",
        ):
            _require_sha256(shard[key], f"week {week} {key}")
        if shard["stable_row_ids_sha256"] != shard["paired_stable_row_ids_sha256"]:
            raise VKLSVDRuntimeError("paired stable-row-ID evidence differs")
        if (
            shard["shared_C_plus_P_values_sha256"]
            != shard["paired_shared_C_plus_P_values_sha256"]
        ):
            raise VKLSVDRuntimeError("paired shared-value evidence differs")
        _validate_token_bounds(shard["token_bounds"], f"week {week} token_bounds")
        _validate_range_map(
            shard["P_ranges"], ORDERED_NUMERIC_OPERANDS, f"week {week} P"
        )
        _validate_range_map(shard["E_ranges"], ORDERED_E_FIELDS, f"week {week} E")
        _validate_range_map(
            shard["target_ranges"], TARGET_COLUMNS, f"week {week} targets"
        )
        _validate_denominator_scan(
            shard["denominator_scan"],
            expected_rows=rows,
            label=f"week {week} denominator",
        )
        shards.append(shard)
    if len(seen_paths) != 12 or file_hash_count != 12:
        raise VKLSVDRuntimeError("full-scan does not contain exactly 12 file hashes")
    return shards, counts


def _validate_full_scan_provenance_and_denominators(
    payload: Mapping[str, Any],
    *,
    shards: Sequence[Mapping[str, Any]],
    total_rows: int,
) -> None:
    normalization = _require_mapping(
        payload["normalization_provenance"], "normalization_provenance"
    )
    _require_exact_keys(
        normalization,
        (
            "fit_split",
            "method",
            "quantile",
            "centering",
            "clipping",
            "P0_scales",
            "E_scales",
            "label_c_seconds",
        ),
        "normalization_provenance",
    )
    expected_policy = {
        "fit_split": "train_weeks_20_24",
        "method": "divide_by_train_abs_q99",
        "quantile": 0.99,
        "centering": "none",
        "clipping": "none",
        "label_c_seconds": float(LABEL_C_SECONDS),
    }
    if any(normalization[key] != expected for key, expected in expected_policy.items()):
        raise VKLSVDRuntimeError("full-scan normalization policy changed")
    for key, fields in (
        ("P0_scales", ORDERED_P0_FIELDS),
        ("E_scales", ORDERED_E_FIELDS),
    ):
        scales = _require_mapping(normalization[key], f"normalization.{key}")
        _require_exact_keys(scales, fields, f"normalization.{key}")
        for field, value in scales.items():
            if _require_finite_number(value, f"normalization.{key}.{field}") <= 0:
                raise VKLSVDRuntimeError(f"normalization.{key}.{field} is not positive")

    semantics = _require_mapping(payload["denominator_semantics"], "denominator")
    _require_exact_keys(
        semantics,
        (
            "eligible_fields",
            "epsilon",
            "noneligible_validity_always_true",
            "active_values_strictly_positive",
            "inactive_values_canonical_zero",
            "floor_indicators_exact",
            "scan",
        ),
        "denominator_semantics",
    )
    if tuple(semantics["eligible_fields"]) != DENOMINATOR_ELIGIBLE_FIELDS:
        raise VKLSVDRuntimeError("full-scan denominator fields changed")
    if semantics["epsilon"] != RECIPROCAL_EPSILON:
        raise VKLSVDRuntimeError("full-scan denominator epsilon changed")
    for key in (
        "noneligible_validity_always_true",
        "active_values_strictly_positive",
        "inactive_values_canonical_zero",
        "floor_indicators_exact",
    ):
        if type(semantics[key]) is not bool or not semantics[key]:
            raise VKLSVDRuntimeError(f"full-scan denominator {key} must be true")
    aggregate = _validate_denominator_scan(
        semantics["scan"], expected_rows=total_rows, label="denominator.scan"
    )
    if dict(aggregate) != _merge_denominator_scan(shards):
        raise VKLSVDRuntimeError("full-scan denominator evidence does not match shards")


def _validate_full_scan_payload(
    payload: Mapping[str, Any], *, enforce_production_locks: bool
) -> None:
    _require_exact_keys(
        payload,
        (
            "schema_version",
            "dataset_id",
            "artifact_role",
            "scan_state",
            "algorithm_version",
            "algorithm_source_sha256",
            "source_manifest_sha256",
            "views_manifest_sha256",
            "feature_contract_sha256",
            "test_unlocked",
            "test_labels_opened",
            "test_view_opened",
            "week26_path_constructed",
            "model_metrics_computed",
            "scan_scope",
            "schema_identity",
            "input_boundary",
            "row_counts",
            "token_bounds",
            "numeric_ranges",
            "normalization_provenance",
            "denominator_semantics",
            "shards",
        ),
        "full-scan manifest",
    )
    for key, expected in {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "artifact_role": "canonical_full_materialized_train_validation_scan",
        "scan_state": "complete_and_hash_verified",
        "algorithm_version": FULL_SCAN_ALGORITHM_VERSION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
    }.items():
        if payload[key] != expected:
            raise VKLSVDRuntimeError(f"full-scan {key} changed")
    generator_source = _require_sha256(
        payload["algorithm_source_sha256"], "full-scan algorithm source SHA"
    )
    expected_generator = (
        FROZEN_V2_GENERATOR_SOURCE_SHA256
        if enforce_production_locks
        else _sha256_file(Path(__file__).resolve())
    )
    if generator_source != expected_generator:
        raise VKLSVDRuntimeError("full-scan generator source is not pinned")
    views_digest = _require_sha256(payload["views_manifest_sha256"], "views SHA")
    if enforce_production_locks and views_digest != VIEWS_MANIFEST_SHA256:
        raise VKLSVDRuntimeError("full-scan views manifest is not frozen")
    for key in (
        "test_unlocked",
        "test_labels_opened",
        "test_view_opened",
        "week26_path_constructed",
        "model_metrics_computed",
    ):
        if type(payload[key]) is not bool or payload[key]:
            raise VKLSVDRuntimeError(f"full-scan {key} must be false")

    scope = _require_mapping(payload["scan_scope"], "full-scan scan_scope")
    _require_exact_keys(
        scope,
        ("roles", "weeks", "paired_regimes", "parquet_batch_rows"),
        "full-scan scan_scope",
    )
    if dict(scope) != {
        "roles": ["train", "validation"],
        "weeks": list(EXPECTED_WEEKS),
        "paired_regimes": ["C+P", "C+P+E"],
        "parquet_batch_rows": PARQUET_BATCH_ROWS,
    }:
        raise VKLSVDRuntimeError("full-scan scope changed")
    _validate_scan_schema_identity(payload["schema_identity"])

    boundary = _require_mapping(payload["input_boundary"], "full-scan input_boundary")
    _require_exact_keys(
        boundary,
        (
            "model_C_plus_P_fields",
            "model_C_plus_P_plus_E_fields",
            "audit_only_fields",
            "target_only_fields",
            "forbidden_model_input_intersection",
            "current_outcomes_consumed_as_features",
        ),
        "full-scan input_boundary",
    )
    expected_boundary = {
        "model_C_plus_P_fields": [*ORDERED_C_FIELDS, *ORDERED_NUMERIC_OPERANDS],
        "model_C_plus_P_plus_E_fields": [
            *ORDERED_C_FIELDS,
            *ORDERED_NUMERIC_OPERANDS,
            *ORDERED_E_FIELDS,
        ],
        "audit_only_fields": list(AUDIT_COLUMNS),
        "target_only_fields": list(TARGET_COLUMNS),
        "forbidden_model_input_intersection": [],
        "current_outcomes_consumed_as_features": False,
    }
    if dict(boundary) != expected_boundary:
        raise VKLSVDRuntimeError("full-scan model-input boundary changed")

    shards, split_counts = _validate_scan_shards(payload["shards"])
    row_counts = _require_mapping(payload["row_counts"], "full-scan row_counts")
    _require_exact_keys(row_counts, ("train", "validation", "total"), "row_counts")
    expected_counts = {
        "train": split_counts["train"],
        "validation": split_counts["validation"],
        "total": split_counts["train"] + split_counts["validation"],
    }
    if dict(row_counts) != expected_counts:
        raise VKLSVDRuntimeError("full-scan row counts do not match exact shards")
    if enforce_production_locks and expected_counts != {
        "train": 7_505_820,
        "validation": 1_516_831,
        "total": 9_022_651,
    }:
        raise VKLSVDRuntimeError("full-scan production row counts changed")

    token_bounds = _validate_token_bounds(payload["token_bounds"], "token_bounds")
    merged_tokens = {
        field: {
            "minimum": min(shard["token_bounds"][field]["minimum"] for shard in shards),
            "maximum": max(shard["token_bounds"][field]["maximum"] for shard in shards),
            "cardinality": EXPECTED_CARDINALITIES[position],
        }
        for position, field in enumerate(ORDERED_C_FIELDS)
    }
    if dict(token_bounds) != merged_tokens:
        raise VKLSVDRuntimeError("full-scan token bounds do not match shards")
    if enforce_production_locks:
        production_tokens = {
            field: {
                "minimum": 2,
                "maximum": cardinality - 1,
                "cardinality": cardinality,
            }
            for field, cardinality in zip(
                ORDERED_C_FIELDS, EXPECTED_CARDINALITIES, strict=True
            )
        }
        if dict(token_bounds) != production_tokens:
            raise VKLSVDRuntimeError("full-scan production token bounds changed")

    numeric = _require_mapping(payload["numeric_ranges"], "numeric_ranges")
    _require_exact_keys(
        numeric,
        ("P", "E", "targets", "all_P_finite", "all_E_finite", "all_targets_finite"),
        "numeric_ranges",
    )
    for key in ("all_P_finite", "all_E_finite", "all_targets_finite"):
        if type(numeric[key]) is not bool or not numeric[key]:
            raise VKLSVDRuntimeError(f"full-scan {key} must be true")
    _validate_range_map(numeric["P"], ORDERED_NUMERIC_OPERANDS, "numeric_ranges.P")
    _validate_range_map(numeric["E"], ORDERED_E_FIELDS, "numeric_ranges.E")
    targets = _validate_range_map(numeric["targets"], TARGET_COLUMNS, "targets")
    if numeric["P"] != _merge_scan_ranges(shards, "P_ranges", ORDERED_NUMERIC_OPERANDS):
        raise VKLSVDRuntimeError("full-scan P ranges do not match shards")
    if numeric["E"] != _merge_scan_ranges(shards, "E_ranges", ORDERED_E_FIELDS):
        raise VKLSVDRuntimeError("full-scan E ranges do not match shards")
    if numeric["targets"] != _merge_scan_ranges(
        shards, "target_ranges", TARGET_COLUMNS
    ):
        raise VKLSVDRuntimeError("full-scan target ranges do not match shards")
    normalized = targets["label_normalized_watch_time"]
    seconds = targets["label_timespent_seconds"]
    if normalized["minimum"] < 0 or normalized["maximum"] > 1:
        raise VKLSVDRuntimeError("full-scan normalized target bounds are invalid")
    if seconds["minimum"] < 0 or seconds["maximum"] > float(LABEL_C_SECONDS):
        raise VKLSVDRuntimeError("full-scan seconds target bounds are invalid")

    _validate_full_scan_provenance_and_denominators(
        payload, shards=shards, total_rows=expected_counts["total"]
    )


def load_full_scan_attestation(
    path: Path,
    *,
    expected_scan_sha256: str | None = None,
    require_expected_digest: bool = False,
    enforce_production_locks: bool = True,
) -> tuple[dict[str, Any], str]:
    payload, digest = _load_yaml_with_sidecar(path)
    if expected_scan_sha256 is not None:
        expected = _require_sha256(expected_scan_sha256, "expected full-scan SHA")
        if digest != expected:
            raise VKLSVDRuntimeError(
                "full-scan digest differs from independently frozen pin"
            )
    elif require_expected_digest:
        raise VKLSVDRuntimeError(
            "independently frozen expected_scan_sha256 is required"
        )
    _validate_full_scan_payload(
        payload, enforce_production_locks=enforce_production_locks
    )
    return payload, digest


def runtime_storage_guard(path: Path, *, estimated_output_bytes: int) -> RuntimeGuards:
    if estimated_output_bytes <= 0 or estimated_output_bytes > MAX_CACHE_BYTES:
        raise VKLSVDRuntimeError(
            "estimated runtime cache size is outside safety budget"
        )
    path.mkdir(parents=True, exist_ok=True)
    nearest = path.resolve(strict=True)
    usage = shutil.disk_usage(nearest)
    used_fraction = (usage.total - usage.free) / usage.total
    if usage.free - estimated_output_bytes < MIN_AVAILABLE_BYTES:
        raise VKLSVDRuntimeError("runtime cache would leave less than 20 GiB free")
    if (
        usage.total - usage.free + estimated_output_bytes
    ) / usage.total >= MAX_USED_FRACTION:
        raise VKLSVDRuntimeError("runtime cache would cross 85% filesystem use")
    return RuntimeGuards(
        checked_path=str(nearest),
        total_bytes=usage.total,
        available_bytes=usage.free,
        used_fraction=used_fraction,
        estimated_output_bytes=estimated_output_bytes,
    )


@dataclass(frozen=True)
class _RegimeSpec:
    name: FeatureRegime
    manifest_key: str
    slug: str
    ordered_operands: tuple[str, ...]
    denominator_mask: tuple[bool, ...]


def _regime_spec(feature_regime: str) -> _RegimeSpec:
    if feature_regime == C_PLUS_P:
        return _RegimeSpec(
            name=C_PLUS_P,
            manifest_key="C_plus_P",
            slug="c_plus_p",
            ordered_operands=ORDERED_NUMERIC_OPERANDS,
            denominator_mask=ORDERED_DENOMINATOR_MASK,
        )
    if feature_regime == C_PLUS_P_PLUS_E:
        return _RegimeSpec(
            name=C_PLUS_P_PLUS_E,
            manifest_key="C_plus_P_plus_E",
            slug="c_plus_p_plus_e",
            ordered_operands=C_PLUS_P_PLUS_E_OPERANDS,
            denominator_mask=C_PLUS_P_PLUS_E_MASK,
        )
    raise VKLSVDRuntimeError("feature_regime must be exactly C+P or C+P+E")


CACHE_ARRAY_ORDER = (
    "context_tokens",
    "operand_features",
    "denominator_validity",
    "targets",
)


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(data).hexdigest()


@dataclass
class _VerifiedNpy:
    """Hash-verified NPY inode retained as the consumed read-only array."""

    fd: int
    mapping: mmap.mmap
    array: np.ndarray
    evidence: dict[str, Any]

    def close(self) -> None:
        self.array = np.empty((0,), dtype=np.uint8)
        self.mapping.close()
        os.close(self.fd)


def _regular_fd_beneath(
    root_fd: int, relative: PurePosixPath, label: str
) -> tuple[int, os.stat_result]:
    current = os.dup(root_fd)
    try:
        for component in relative.parts[:-1]:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                child = os.open(component, flags, dir_fd=current)
            except OSError as error:
                raise VKLSVDRuntimeError(
                    f"cannot securely open directory for {label}"
                ) from error
            os.close(current)
            current = child
        return _regular_fd_at(current, relative.parts[-1], label)
    finally:
        os.close(current)


def _open_verified_npy_at(
    root_fd: int,
    relative: PurePosixPath,
    *,
    expected_dtype: str,
    expected_shape: Sequence[int],
) -> _VerifiedNpy:
    descriptor, before = _regular_fd_beneath(root_fd, relative, "runtime tensor memmap")
    mapping: mmap.mmap | None = None
    try:
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            version = np.lib.format.read_magic(handle)
            shape, fortran_order, dtype = np.lib.format._read_array_header(
                handle, version
            )
            offset = handle.tell()
        if (
            dtype.str != expected_dtype
            or list(shape) != list(expected_shape)
            or fortran_order
            or dtype.hasobject
        ):
            raise VKLSVDRuntimeError("runtime tensor dtype/shape drift")
        tensor_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if before.st_size != offset + tensor_nbytes:
            raise VKLSVDRuntimeError(
                "runtime tensor file has trailing or missing bytes"
            )
        file_digest = hashlib.sha256()
        tensor_digest = hashlib.sha256()
        position = 0
        while position < before.st_size:
            chunk = os.pread(
                descriptor,
                min(8 * 1024 * 1024, before.st_size - position),
                position,
            )
            if not chunk:
                raise VKLSVDRuntimeError("runtime tensor changed during verified read")
            file_digest.update(chunk)
            end = position + len(chunk)
            if end > offset:
                tensor_digest.update(chunk[max(0, offset - position) :])
            position = end
        after = os.fstat(descriptor)
        if position != before.st_size or _stable_stat_identity(
            before
        ) != _stable_stat_identity(after):
            raise VKLSVDRuntimeError("runtime tensor changed during verified read")
        mapping = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
        array = np.ndarray(shape, dtype=dtype, buffer=mapping, offset=offset, order="C")
        if array.flags.writeable or not array.flags.c_contiguous:
            raise VKLSVDRuntimeError("runtime tensor must be read-only C-contiguous")
        evidence = {
            "bytes": before.st_size,
            "sha256": file_digest.hexdigest(),
            "dtype": expected_dtype,
            "shape": list(expected_shape),
            "tensor_nbytes": tensor_nbytes,
            "tensor_bytes_sha256": tensor_digest.hexdigest(),
        }
        return _VerifiedNpy(descriptor, mapping, array, evidence)
    except BaseException:
        if mapping is not None:
            mapping.close()
        os.close(descriptor)
        raise


def _inspect_npy_tensor(
    path: Path,
    *,
    expected_dtype: str,
    expected_shape: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    root_fd = _directory_fd(path.parent, "runtime tensor parent")
    try:
        verified = _open_verified_npy_at(
            root_fd,
            PurePosixPath(path.name),
            expected_dtype=expected_dtype,
            expected_shape=expected_shape,
        )
    finally:
        os.close(root_fd)
    os.close(verified.fd)
    return verified.array, verified.evidence


def _cache_content_payload(
    metadata: Mapping[str, Any], spec: _RegimeSpec
) -> dict[str, Any]:
    splits = _require_mapping(metadata.get("splits"), "cache splits")
    content_splits = {}
    for split in ("train", "validation"):
        split_record = _require_mapping(splits.get(split), f"cache {split}")
        arrays = _require_mapping(split_record.get("arrays"), f"cache {split} arrays")
        records = []
        for name in CACHE_ARRAY_ORDER:
            record = _require_mapping(arrays.get(name), f"cache {split} {name}")
            records.append(
                {
                    "name": name,
                    "dtype": record.get("dtype"),
                    "shape": record.get("shape"),
                    "tensor_nbytes": record.get("tensor_nbytes"),
                    "tensor_bytes_sha256": record.get("tensor_bytes_sha256"),
                }
            )
        content_splits[split] = {
            "row_count": split_record.get("rows"),
            "arrays": records,
        }
    return {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "feature_regime": spec.name,
        "upstream": {
            "source_manifest_sha256": metadata.get("source_manifest_sha256"),
            "views_manifest_sha256": metadata.get("views_manifest_sha256"),
            "feature_contract_sha256": metadata.get("feature_contract_sha256"),
            "full_materialized_scan_sha256": metadata.get(
                "full_materialized_scan_sha256"
            ),
        },
        "schema": {
            "physical_schema_sha256": metadata.get("physical_schema_sha256"),
            "expanded_schema_sha256": metadata.get("expanded_schema_sha256"),
            "operand_schema_sha256": metadata.get("operand_schema_sha256"),
            "denominator_mask_sha256": metadata.get("denominator_mask_sha256"),
            "ordered_context_fields": metadata.get("ordered_context_fields"),
            "ordered_operand_fields": metadata.get("ordered_operand_fields"),
            "categorical_cardinalities": metadata.get("categorical_cardinalities"),
        },
        "splits": content_splits,
    }


def _cache_expected_bytes(
    row_counts: Mapping[str, Any], feature_regime: FeatureRegime = C_PLUS_P
) -> int:
    rows = int(row_counts["train"]) + int(row_counts["validation"])
    width = len(_regime_spec(feature_regime).ordered_operands)
    # NPY header overhead is bounded separately with one page per array.
    return rows * (6 * 4 + width * 4 + width + 4) + 8 * 4096


def _write_cache_metadata(path: Path, payload: Mapping[str, Any]) -> str:
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        + b"\n"
    )
    if len(data) > MAX_CACHE_METADATA_BYTES:
        raise VKLSVDRuntimeError("cache metadata exceeds size limit")
    digest = hashlib.sha256(data).hexdigest()
    path.write_bytes(data)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    with sidecar.open("rb") as handle:
        os.fsync(handle.fileno())
    return digest


def _load_cache_metadata(path: Path) -> tuple[dict[str, Any], str]:
    parent_fd = _directory_fd(path.parent, "runtime cache metadata parent")
    try:
        return _load_cache_metadata_at(parent_fd, path.name)
    finally:
        os.close(parent_fd)


def _load_cache_metadata_at(
    directory_fd: int, name: str = "cache_manifest.json"
) -> tuple[dict[str, Any], str]:
    metadata_fd, _ = _regular_fd_at(directory_fd, name, "runtime cache metadata")
    sidecar_name = f"{name}.sha256"
    sidecar_fd, _ = _regular_fd_at(directory_fd, sidecar_name, "cache sidecar")
    try:
        data = _read_fd_bytes(
            metadata_fd,
            maximum_bytes=MAX_CACHE_METADATA_BYTES,
            label="runtime cache metadata",
        )
        sidecar_data = _read_fd_bytes(
            sidecar_fd,
            maximum_bytes=MAX_SIDECAR_BYTES,
            label="cache sidecar",
        )
    finally:
        os.close(sidecar_fd)
        os.close(metadata_fd)
    digest = _parse_sidecar_bytes(sidecar_data, expected_name=name)
    if hashlib.sha256(data).hexdigest() != digest:
        raise VKLSVDRuntimeError("runtime cache metadata sidecar mismatch")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VKLSVDRuntimeError("runtime cache metadata is invalid JSON") from error
    if not isinstance(payload, dict):
        raise VKLSVDRuntimeError("runtime cache metadata root must be an object")
    return payload, digest


_CACHE_METADATA_KEYS = (
    "schema_version",
    "dataset_id",
    "artifact_role",
    "cache_state",
    "built_utc",
    "algorithm_version",
    "algorithm_source_sha256",
    "source_manifest_sha256",
    "views_manifest_sha256",
    "feature_contract_sha256",
    "full_materialized_scan_sha256",
    "cache_content_sha256",
    "content_identity",
    "feature_regime",
    "test_split_present",
    "model_metrics_computed",
    "ordered_context_fields",
    "ordered_operand_fields",
    "categorical_cardinalities",
    "physical_schema_sha256",
    "expanded_schema_sha256",
    "operand_schema_sha256",
    "denominator_mask_sha256",
    "storage_guard",
    "splits",
    "elapsed_seconds",
)
_CACHE_ARRAY_FILENAMES = {
    "context_tokens": "context_tokens.i32.npy",
    "operand_features": "operand_features.f32.npy",
    "denominator_validity": "denominator_validity.u8.npy",
    "targets": "targets.f32.npy",
}
_CACHE_ARRAY_DTYPES = {
    "context_tokens": "<i4",
    "operand_features": "<f4",
    "denominator_validity": "|u1",
    "targets": "<f4",
}


def _cache_array_shape(name: str, rows: int, operand_width: int) -> list[int]:
    if name == "context_tokens":
        return [rows, len(ORDERED_C_FIELDS)]
    if name in {"operand_features", "denominator_validity"}:
        return [rows, operand_width]
    if name == "targets":
        return [rows, 1]
    raise VKLSVDRuntimeError(f"unknown cache tensor: {name}")


def _validate_cache_metadata(
    metadata: Mapping[str, Any],
    *,
    cache_root: Path,
    cache_root_fd: int,
    scan: Mapping[str, Any],
    scan_digest: str,
    spec: _RegimeSpec,
    expected_cache_content_sha256: str | None,
    require_expected_cache_digest: bool,
    enforce_production_locks: bool,
    retain_split: str | None = None,
    retained_arrays: dict[str, _VerifiedNpy] | None = None,
) -> str:
    _require_exact_keys(metadata, _CACHE_METADATA_KEYS, "runtime cache metadata")
    expected_scalars = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "artifact_role": "ephemeral_raw_memmap_cache",
        "cache_state": "complete_and_hash_verified",
        "algorithm_version": CACHE_ALGORITHM_VERSION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "full_materialized_scan_sha256": scan_digest,
        "feature_regime": spec.name,
        "test_split_present": False,
        "model_metrics_computed": False,
    }
    for key, expected in expected_scalars.items():
        if metadata[key] != expected:
            raise VKLSVDRuntimeError(f"runtime cache {key} changed")
    generator_source = _require_sha256(
        metadata["algorithm_source_sha256"],
        "runtime cache algorithm source SHA",
    )
    expected_generator = (
        FROZEN_V2_GENERATOR_SOURCE_SHA256
        if enforce_production_locks
        else _sha256_file(Path(__file__).resolve())
    )
    if generator_source != expected_generator:
        raise VKLSVDRuntimeError("runtime cache generator source is not pinned")
    if metadata["views_manifest_sha256"] != scan["views_manifest_sha256"]:
        raise VKLSVDRuntimeError("runtime cache views identity differs from full scan")
    if (
        enforce_production_locks
        and metadata["views_manifest_sha256"] != VIEWS_MANIFEST_SHA256
    ):
        raise VKLSVDRuntimeError("runtime cache views manifest is not frozen")
    built_utc = metadata["built_utc"]
    if not isinstance(built_utc, str) or not built_utc.endswith("Z"):
        raise VKLSVDRuntimeError("runtime cache built_utc is invalid")
    try:
        datetime.fromisoformat(built_utc.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise VKLSVDRuntimeError("runtime cache built_utc is invalid") from error
    elapsed = _require_finite_number(
        metadata["elapsed_seconds"], "runtime cache elapsed_seconds"
    )
    if elapsed < 0:
        raise VKLSVDRuntimeError("runtime cache elapsed_seconds is negative")

    schema = scan["schema_identity"]
    regime_key = spec.manifest_key
    expected_schema = {
        "physical_schema_sha256": schema["physical_schema_sha256"][regime_key],
        "expanded_schema_sha256": schema["expanded_schema_sha256"][regime_key],
        "operand_schema_sha256": schema["operand_schema_sha256"][regime_key],
        "denominator_mask_sha256": schema["denominator_mask_sha256"][regime_key],
    }
    for key, expected in expected_schema.items():
        if metadata[key] != expected:
            raise VKLSVDRuntimeError(f"runtime cache {key} changed")
        _require_sha256(metadata[key], f"runtime cache {key}")
    if tuple(metadata["ordered_context_fields"]) != ORDERED_C_FIELDS:
        raise VKLSVDRuntimeError("runtime cache context order changed")
    if tuple(metadata["ordered_operand_fields"]) != spec.ordered_operands:
        raise VKLSVDRuntimeError("runtime cache operand order changed")
    if tuple(metadata["categorical_cardinalities"]) != EXPECTED_CARDINALITIES:
        raise VKLSVDRuntimeError("runtime cache cardinalities changed")

    guard = _require_mapping(metadata["storage_guard"], "runtime cache storage_guard")
    _require_exact_keys(
        guard,
        (
            "checked_path",
            "total_bytes",
            "available_bytes_before",
            "used_fraction_before",
            "estimated_output_bytes",
        ),
        "runtime cache storage_guard",
    )
    if not isinstance(guard["checked_path"], str) or not guard["checked_path"]:
        raise VKLSVDRuntimeError("runtime cache guard path is missing")
    for key in ("total_bytes", "available_bytes_before", "estimated_output_bytes"):
        _require_positive_int(guard[key], f"runtime cache guard {key}")
    used_fraction = _require_finite_number(
        guard["used_fraction_before"], "runtime cache guard used_fraction_before"
    )
    if not 0 <= used_fraction < 1:
        raise VKLSVDRuntimeError("runtime cache guard fraction is invalid")

    content_digest = _require_sha256(
        metadata["cache_content_sha256"], "runtime cache content SHA"
    )
    if expected_cache_content_sha256 is not None:
        expected_content = _require_sha256(
            expected_cache_content_sha256, "expected cache-content SHA"
        )
        if content_digest != expected_content:
            raise VKLSVDRuntimeError(
                "cache content differs from independently frozen pin"
            )
    elif require_expected_cache_digest:
        raise VKLSVDRuntimeError(
            "independently frozen expected_cache_content_sha256 is required"
        )
    distinct_from = {
        scan_digest,
        metadata["source_manifest_sha256"],
        metadata["views_manifest_sha256"],
        metadata["feature_contract_sha256"],
        *expected_schema.values(),
    }
    if content_digest in distinct_from:
        raise VKLSVDRuntimeError("cache-content digest is not independently distinct")

    splits = _require_mapping(metadata["splits"], "runtime cache splits")
    _require_exact_keys(splits, ("train", "validation"), "runtime cache splits")
    scan_shards = {
        (record["role"], record["week"]): record for record in scan["shards"]
    }
    operand_width = len(spec.ordered_operands)
    for split in ("train", "validation"):
        record = _require_mapping(splits[split], f"runtime cache {split}")
        _require_exact_keys(
            record,
            ("rows", "implicit_global_indices", "source_shard_ranges", "arrays"),
            f"runtime cache {split}",
        )
        rows = _require_positive_int(record["rows"], f"runtime cache {split} rows")
        if rows != scan["row_counts"][split]:
            raise VKLSVDRuntimeError("runtime cache split row count changed")
        if record["implicit_global_indices"] != [0, rows]:
            raise VKLSVDRuntimeError("runtime cache implicit index range changed")
        source_ranges = record["source_shard_ranges"]
        expected_weeks = TRAIN_WEEKS if split == "train" else (VALIDATION_WEEK,)
        if not isinstance(source_ranges, list) or len(source_ranges) != len(
            expected_weeks
        ):
            raise VKLSVDRuntimeError("runtime cache source ranges changed")
        offset = 0
        for position, week in enumerate(expected_weeks):
            source = _require_mapping(
                source_ranges[position], f"runtime cache {split} source range"
            )
            _require_exact_keys(
                source,
                ("week", "start", "stop", "rows", "source_path", "source_sha256"),
                f"runtime cache {split} source range",
            )
            scanned = scan_shards[(split, week)]
            expected_source = scanned[regime_key]
            expected_rows = scanned["rows"]
            if source != {
                "week": week,
                "start": offset,
                "stop": offset + expected_rows,
                "rows": expected_rows,
                "source_path": expected_source["path"],
                "source_sha256": expected_source["sha256"],
            }:
                raise VKLSVDRuntimeError(
                    "runtime cache source range or source identity changed"
                )
            _safe_relative_path(source["source_path"], "cache source shard")
            _require_sha256(source["source_sha256"], "cache source shard SHA")
            offset += expected_rows
        if offset != rows:
            raise VKLSVDRuntimeError("runtime cache source ranges do not cover split")

        arrays = _require_mapping(record["arrays"], f"runtime cache {split} arrays")
        _require_exact_keys(arrays, CACHE_ARRAY_ORDER, f"runtime cache {split} arrays")
        for name in CACHE_ARRAY_ORDER:
            array_record = _require_mapping(
                arrays[name], f"runtime cache {split} {name}"
            )
            _require_exact_keys(
                array_record,
                (
                    "path",
                    "dtype",
                    "shape",
                    "bytes",
                    "sha256",
                    "tensor_nbytes",
                    "tensor_bytes_sha256",
                ),
                f"runtime cache {split} {name}",
            )
            expected_path = PurePosixPath(split, _CACHE_ARRAY_FILENAMES[name])
            relative = _safe_relative_path(
                array_record["path"], f"runtime cache {split} {name}"
            )
            if relative != expected_path:
                raise VKLSVDRuntimeError("runtime cache tensor path changed")
            expected_shape = _cache_array_shape(name, rows, operand_width)
            verified = _open_verified_npy_at(
                cache_root_fd,
                relative,
                expected_dtype=_CACHE_ARRAY_DTYPES[name],
                expected_shape=expected_shape,
            )
            expected_record = {
                "path": relative.as_posix(),
                **verified.evidence,
            }
            if dict(array_record) != expected_record:
                verified.close()
                raise VKLSVDRuntimeError(
                    f"runtime cache tensor evidence changed: {split}/{name}"
                )
            if retained_arrays is not None and split == retain_split:
                retained_arrays[name] = verified
            else:
                verified.close()

    content_identity = _require_mapping(
        metadata["content_identity"], "runtime cache content_identity"
    )
    computed_identity = _cache_content_payload(metadata, spec)
    if dict(content_identity) != computed_identity:
        raise VKLSVDRuntimeError("runtime cache content identity changed")
    if _canonical_json_sha256(computed_identity) != content_digest:
        raise VKLSVDRuntimeError("runtime cache content digest is invalid")
    return content_digest


def build_vk_lsvd_runtime_cache(
    *,
    views_manifest_path: Path,
    views_root: Path,
    full_scan_manifest_path: Path,
    cache_root: Path,
    feature_regime: FeatureRegime = C_PLUS_P,
    expected_scan_sha256: str | None = None,
    enforce_production_locks: bool = True,
) -> tuple[dict[str, Any], str]:
    """Build atomic raw NPY memmaps for one exact train/validation regime."""

    spec = _regime_spec(feature_regime)
    scan, scan_digest = load_full_scan_attestation(
        full_scan_manifest_path,
        expected_scan_sha256=expected_scan_sha256,
        require_expected_digest=enforce_production_locks,
        enforce_production_locks=enforce_production_locks,
    )
    views, views_digest = _load_views_manifest_verified(views_manifest_path)
    _validate_frozen_manifest_identity(
        views, views_digest, enforce_production_locks=enforce_production_locks
    )
    if scan.get("views_manifest_sha256") != views_digest:
        raise VKLSVDRuntimeError(
            "full scan does not attest the selected views manifest"
        )
    if scan_digest == views_digest:
        raise VKLSVDRuntimeError("full scan digest must be distinct from views digest")
    resolved = _resolve_shards(views, views_root)
    estimated = _cache_expected_bytes(views["row_counts"], spec.name)
    guard = runtime_storage_guard(cache_root.parent, estimated_output_bytes=estimated)
    if cache_root.exists() or cache_root.is_symlink():
        raise VKLSVDRuntimeError(
            "runtime cache output already exists; reuse is forbidden"
        )
    if stat.S_ISLNK(cache_root.parent.lstat().st_mode):
        raise VKLSVDRuntimeError("runtime cache parent must not be a symlink")
    temporary = cache_root.parent / f".{cache_root.name}.incomplete-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise VKLSVDRuntimeError("runtime cache temporary output already exists")
    temporary.mkdir(mode=0o700)
    started = time.monotonic()
    split_records: dict[str, Any] = {}
    scan_shards = {
        (record["role"], record["week"]): record for record in scan["shards"]
    }
    operand_width = len(spec.ordered_operands)
    try:
        for split in ("train", "validation"):
            split_shards = [item for item in resolved if item.role == split]
            rows = sum(item.rows for item in split_shards)
            split_dir = temporary / split
            split_dir.mkdir(mode=0o700)
            arrays = {
                "context_tokens": np.lib.format.open_memmap(
                    split_dir / "context_tokens.i32.npy",
                    mode="w+",
                    dtype="<i4",
                    shape=(rows, 6),
                ),
                "operand_features": np.lib.format.open_memmap(
                    split_dir / "operand_features.f32.npy",
                    mode="w+",
                    dtype="<f4",
                    shape=(rows, operand_width),
                ),
                "denominator_validity": np.lib.format.open_memmap(
                    split_dir / "denominator_validity.u8.npy",
                    mode="w+",
                    dtype="|u1",
                    shape=(rows, operand_width),
                ),
                "targets": np.lib.format.open_memmap(
                    split_dir / "targets.f32.npy",
                    mode="w+",
                    dtype="<f4",
                    shape=(rows, 1),
                ),
            }
            offset = 0
            ranges = []
            columns = [
                *ORDERED_C_FIELDS,
                *spec.ordered_operands,
                *VALIDITY_COLUMNS,
                "label_normalized_watch_time",
            ]
            for shard in split_shards:
                source_path = shard.cp_path if spec.name == C_PLUS_P else shard.cpe_path
                views_source = shard.record[spec.manifest_key]
                scan_source = scan_shards[(split, shard.week)][spec.manifest_key]
                if dict(views_source) != {
                    "path": scan_source["path"],
                    "bytes": scan_source["bytes"],
                    "sha256": scan_source["sha256"],
                    "row_groups": scan_source["row_groups"],
                }:
                    raise VKLSVDRuntimeError(
                        "selected source shard differs between views and full scan"
                    )
                if _sha256_file(source_path) != scan_source["sha256"]:
                    raise VKLSVDRuntimeError(
                        f"{spec.name} shard drift before cache build"
                    )
                parquet = pq.ParquetFile(source_path)
                shard_start = offset
                for batch in parquet.iter_batches(
                    batch_size=PARQUET_BATCH_ROWS, columns=columns, use_threads=False
                ):
                    count = batch.num_rows
                    end = offset + count
                    context = np.column_stack(
                        [_numpy(batch, name) for name in ORDERED_C_FIELDS]
                    )
                    if context.dtype != np.dtype("uint32"):
                        raise VKLSVDRuntimeError("cache context source dtype drift")
                    for column, cardinality in enumerate(EXPECTED_CARDINALITIES):
                        values = context[:, column]
                        if int(values.min()) < 0 or int(values.max()) >= cardinality:
                            raise VKLSVDRuntimeError(
                                "cache context token exceeds frozen cardinality"
                            )
                    operands = np.column_stack(
                        [_numpy(batch, name) for name in spec.ordered_operands]
                    )
                    p_validity = np.column_stack(
                        [_numpy(batch, name) for name in VALIDITY_COLUMNS]
                    )
                    targets = _numpy(batch, "label_normalized_watch_time")[:, None]
                    if (
                        operands.dtype != np.dtype("float32")
                        or p_validity.dtype != np.dtype("bool")
                        or targets.dtype != np.dtype("float32")
                        or not np.isfinite(operands).all()
                        or not np.isfinite(targets).all()
                    ):
                        raise VKLSVDRuntimeError(
                            "cache input dtype/nonfinite violation"
                        )
                    if np.any((targets < 0.0) | (targets > 1.0)):
                        raise VKLSVDRuntimeError(
                            "cache normalized target is outside [0,1]"
                        )
                    p_operands = operands[:, : len(ORDERED_NUMERIC_OPERANDS)]
                    for column, eligible in enumerate(ORDERED_DENOMINATOR_MASK):
                        column_validity = p_validity[:, column]
                        column_values = p_operands[:, column]
                        if not eligible and not bool(column_validity.all()):
                            raise VKLSVDRuntimeError(
                                "non-denominator cache validity is not all true"
                            )
                        if eligible and (
                            np.any(column_validity & (column_values <= 0.0))
                            or np.any((~column_validity) & (column_values != 0.0))
                        ):
                            raise VKLSVDRuntimeError(
                                "cache denominator validity/value mismatch"
                            )
                    validity = p_validity
                    if spec.name == C_PLUS_P_PLUS_E:
                        validity = np.column_stack(
                            (
                                p_validity,
                                np.zeros(
                                    (count, len(ORDERED_E_FIELDS)), dtype=np.bool_
                                ),
                            )
                        )
                    arrays["context_tokens"][offset:end] = context.astype(np.int32)
                    arrays["operand_features"][offset:end] = operands
                    arrays["denominator_validity"][offset:end] = validity.astype(
                        np.uint8
                    )
                    arrays["targets"][offset:end] = targets
                    offset = end
                ranges.append(
                    {
                        "week": shard.week,
                        "start": shard_start,
                        "stop": offset,
                        "rows": shard.rows,
                        "source_path": scan_source["path"],
                        "source_sha256": scan_source["sha256"],
                    }
                )
                if offset - shard_start != shard.rows:
                    raise VKLSVDRuntimeError(
                        "cache shard streamed row count differs from manifest"
                    )
            if offset != rows:
                raise VKLSVDRuntimeError("cache row total differs from source shards")
            for array in arrays.values():
                array.flush()
            del arrays
            array_records: dict[str, Any] = {}
            for name in CACHE_ARRAY_ORDER:
                filename = _CACHE_ARRAY_FILENAMES[name]
                dtype = _CACHE_ARRAY_DTYPES[name]
                shape = _cache_array_shape(name, rows, operand_width)
                path = split_dir / filename
                reopened, inspected = _inspect_npy_tensor(
                    path, expected_dtype=dtype, expected_shape=shape
                )
                del reopened
                array_records[name] = {
                    "path": f"{split}/{filename}",
                    **inspected,
                }
            split_records[split] = {
                "rows": rows,
                "implicit_global_indices": [0, rows],
                "source_shard_ranges": ranges,
                "arrays": array_records,
            }
        metadata_payload = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "artifact_role": "ephemeral_raw_memmap_cache",
            "cache_state": "complete_and_hash_verified",
            "built_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "algorithm_version": CACHE_ALGORITHM_VERSION,
            "algorithm_source_sha256": _sha256_file(Path(__file__).resolve()),
            "source_manifest_sha256": views["source_manifest_sha256"],
            "views_manifest_sha256": views_digest,
            "feature_contract_sha256": views["feature_contract_sha256"],
            "full_materialized_scan_sha256": scan_digest,
            "cache_content_sha256": "",
            "content_identity": {},
            "feature_regime": spec.name,
            "test_split_present": False,
            "model_metrics_computed": False,
            "ordered_context_fields": list(ORDERED_C_FIELDS),
            "ordered_operand_fields": list(spec.ordered_operands),
            "categorical_cardinalities": list(EXPECTED_CARDINALITIES),
            "physical_schema_sha256": scan["schema_identity"]["physical_schema_sha256"][
                spec.manifest_key
            ],
            "expanded_schema_sha256": scan["schema_identity"]["expanded_schema_sha256"][
                spec.manifest_key
            ],
            "operand_schema_sha256": scan["schema_identity"]["operand_schema_sha256"][
                spec.manifest_key
            ],
            "denominator_mask_sha256": scan["schema_identity"][
                "denominator_mask_sha256"
            ][spec.manifest_key],
            "storage_guard": {
                "checked_path": guard.checked_path,
                "total_bytes": guard.total_bytes,
                "available_bytes_before": guard.available_bytes,
                "used_fraction_before": guard.used_fraction,
                "estimated_output_bytes": guard.estimated_output_bytes,
            },
            "splits": split_records,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        content_identity = _cache_content_payload(metadata_payload, spec)
        content_digest = _canonical_json_sha256(content_identity)
        metadata_payload["content_identity"] = content_identity
        metadata_payload["cache_content_sha256"] = content_digest
        metadata_digest = _write_cache_metadata(
            temporary / "cache_manifest.json", metadata_payload
        )
        os.replace(temporary, cache_root)
        descriptor = os.open(
            cache_root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    cache_root_fd = _directory_fd(cache_root, "runtime cache root")
    try:
        loaded, loaded_digest = _load_cache_metadata_at(cache_root_fd)
        if loaded_digest != metadata_digest:
            raise VKLSVDRuntimeError(
                "runtime cache metadata reopen verification failed"
            )
        reopened_content = _validate_cache_metadata(
            loaded,
            cache_root=cache_root.resolve(strict=True),
            cache_root_fd=cache_root_fd,
            scan=scan,
            scan_digest=scan_digest,
            spec=spec,
            expected_cache_content_sha256=content_digest,
            require_expected_cache_digest=True,
            enforce_production_locks=enforce_production_locks,
        )
    finally:
        os.close(cache_root_fd)
    if reopened_content != content_digest:
        raise VKLSVDRuntimeError("runtime cache content reopen verification failed")
    return metadata_payload, metadata_digest


class VKLSVDRawBatchSource:
    """Read-only process-local VK memmap implementation of ``RawBatchSource``."""

    def __init__(
        self,
        cache_root: Path,
        *,
        split: Literal["train", "validation"],
        full_scan_manifest_path: Path,
        feature_regime: FeatureRegime = C_PLUS_P,
        expected_scan_sha256: str | None = None,
        expected_cache_content_sha256: str | None = None,
        verify_hashes: bool = True,
        enforce_production_locks: bool = True,
        runtime_value_checks: bool = True,
    ) -> None:
        if split not in {"train", "validation"}:
            raise VKLSVDRuntimeError(
                "only train and validation raw sources are allowed"
            )
        if enforce_production_locks and not verify_hashes:
            raise VKLSVDRuntimeError(
                "production source cannot disable cache tensor hashing"
            )
        if enforce_production_locks and expected_scan_sha256 is None:
            raise VKLSVDRuntimeError("production source requires expected_scan_sha256")
        if enforce_production_locks and expected_cache_content_sha256 is None:
            raise VKLSVDRuntimeError(
                "production source requires expected_cache_content_sha256"
            )
        if not isinstance(runtime_value_checks, bool):
            raise TypeError("runtime_value_checks must be boolean")
        spec = _regime_spec(feature_regime)
        scan, scan_digest = load_full_scan_attestation(
            full_scan_manifest_path,
            expected_scan_sha256=expected_scan_sha256,
            require_expected_digest=enforce_production_locks,
            enforce_production_locks=enforce_production_locks,
        )
        root_fd = _directory_fd(cache_root, "runtime cache root")
        retained: dict[str, _VerifiedNpy] = {}
        try:
            metadata, _ = _load_cache_metadata_at(root_fd)
            content_digest = _validate_cache_metadata(
                metadata,
                cache_root=cache_root,
                cache_root_fd=root_fd,
                scan=scan,
                scan_digest=scan_digest,
                spec=spec,
                expected_cache_content_sha256=expected_cache_content_sha256,
                require_expected_cache_digest=enforce_production_locks,
                enforce_production_locks=enforce_production_locks,
                retain_split=split,
                retained_arrays=retained,
            )
        except BaseException:
            for verified in retained.values():
                verified.close()
            os.close(root_fd)
            raise
        split_record = metadata["splits"][split]
        rows = split_record["rows"]
        self._pid = os.getpid()
        self._root_fd = root_fd
        self._verified_arrays = retained
        self._arrays = {name: verified.array for name, verified in retained.items()}
        self._runtime_value_checks = runtime_value_checks
        self._identity = RawSourceIdentity(
            dataset_id=DATASET_ID,
            split=split,
            feature_regime=spec.name,
            row_count=rows,
            data_manifest_sha256=metadata["source_manifest_sha256"],
            views_manifest_sha256=metadata["views_manifest_sha256"],
            feature_contract_sha256=metadata["feature_contract_sha256"],
            physical_schema_sha256=metadata["physical_schema_sha256"],
            expanded_schema_sha256=metadata["expanded_schema_sha256"],
            operand_schema_sha256=metadata["operand_schema_sha256"],
            denominator_mask_sha256=metadata["denominator_mask_sha256"],
            full_materialized_scan_sha256=scan_digest,
            cache_content_sha256=content_digest,
            ordered_context_fields=ORDERED_C_FIELDS,
            ordered_operand_fields=spec.ordered_operands,
            categorical_cardinalities=EXPECTED_CARDINALITIES,
        )

    @property
    def identity(self) -> RawSourceIdentity:
        return self._identity

    @property
    def runtime_value_checks(self) -> bool:
        return self._runtime_value_checks

    def close(self) -> None:
        root_fd = getattr(self, "_root_fd", -1)
        if root_fd < 0:
            return
        self._arrays = {}
        verified_arrays = self._verified_arrays
        self._verified_arrays = {}
        self._root_fd = -1
        for verified in verified_arrays.values():
            verified.close()
        os.close(root_fd)

    def __del__(self) -> None:
        with suppress(AttributeError, OSError, BufferError):
            self.close()

    def __len__(self) -> int:
        return self._identity.row_count

    def batch(
        self,
        indices: Sequence[int],
        *,
        device: torch.device,
    ) -> RawFeatureBatch:
        if os.getpid() != self._pid:
            raise VKLSVDRuntimeError("raw memmaps cannot be inherited across processes")
        if isinstance(indices, str | bytes) or not isinstance(indices, Sequence):
            raise VKLSVDRuntimeError("raw batch indices must be a bounded Sequence")
        try:
            batch_rows = len(indices)
        except (OverflowError, TypeError) as error:
            raise VKLSVDRuntimeError(
                "raw batch indices must expose a cheap finite length"
            ) from error
        if batch_rows <= 0:
            raise VKLSVDRuntimeError("raw batch indices must be non-empty")
        if batch_rows > MAX_BATCH_ROWS:
            raise VKLSVDRuntimeError(
                f"raw batch exceeds MAX_BATCH_ROWS={MAX_BATCH_ROWS}"
            )
        requested = tuple(indices)
        if len(requested) != batch_rows:
            raise VKLSVDRuntimeError("raw batch Sequence length changed during read")
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in requested
        ):
            raise VKLSVDRuntimeError("raw batch indices must be Python integers")
        if any(index < 0 or index >= len(self) for index in requested):
            raise VKLSVDRuntimeError("raw batch index is out of range")
        target_device = torch.device(device)
        selection = np.asarray(requested, dtype=np.int64)
        context = np.asarray(self._arrays["context_tokens"][selection], dtype=np.int32)
        operands = np.asarray(
            self._arrays["operand_features"][selection], dtype=np.float32
        )
        validity = np.asarray(
            self._arrays["denominator_validity"][selection], dtype=np.uint8
        )
        targets = np.asarray(self._arrays["targets"][selection], dtype=np.float32)
        result = RawFeatureBatch(
            context_tokens=torch.from_numpy(context).to(
                device=target_device, dtype=torch.int64
            ),
            operand_features=torch.from_numpy(operands).to(device=target_device),
            denominator_validity=torch.from_numpy(validity).to(
                device=target_device, dtype=torch.bool
            ),
            targets=torch.from_numpy(targets).to(device=target_device),
            global_row_indices=torch.tensor(requested, dtype=torch.int64),
        )
        result.validate(
            self._identity,
            requested,
            runtime_value_checks=self._runtime_value_checks,
        )
        return result


def build_vk_lsvd_raw_source(
    cache_root: Path,
    *,
    split: Literal["train", "validation"],
    full_scan_manifest_path: Path,
    feature_regime: FeatureRegime = C_PLUS_P,
    expected_scan_sha256: str | None = None,
    expected_cache_content_sha256: str | None = None,
    verify_hashes: bool = True,
    enforce_production_locks: bool = True,
    runtime_value_checks: bool = True,
) -> VKLSVDRawBatchSource:
    return VKLSVDRawBatchSource(
        cache_root,
        split=split,
        full_scan_manifest_path=full_scan_manifest_path,
        feature_regime=feature_regime,
        expected_scan_sha256=expected_scan_sha256,
        expected_cache_content_sha256=expected_cache_content_sha256,
        verify_hashes=verify_hashes,
        enforce_production_locks=enforce_production_locks,
        runtime_value_checks=runtime_value_checks,
    )
