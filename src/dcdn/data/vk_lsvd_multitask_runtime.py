"""Verified bounded-memory runtime for VK-LSVD seven-head views.

Only weeks 20--24 (train) and 25 (validation) exist in this path graph.  Week
26 is rejected by both manifest validation and path resolution.  Raw labels,
masks, and audit user IDs remain outside the model encoder signature.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from dcdn.data.vk_lsvd_runtime import (
    EXPECTED_CARDINALITIES,
    MAX_BATCH_ROWS,
    PARQUET_BATCH_ROWS,
    _resolve_beneath,
    _safe_relative_path,
    _sha256_file,
)
from dcdn.models.contracts import canonical_sha256
from dcdn.multitask import MultiTargetSpec, TargetSpec, TaskConstants
from dcdn.protocol import SHA256_RE, read_verified_artifact
from dcdn.raw import RawSourceIdentity

DATASET_ID = "vk_lsvd_up0.01_ir0.01"
DATA_MANIFEST_SHA256 = (
    "5dbbfeb23442a6b1ebdb829dc239c6a3c6c655d71446cde4551b83cc6af456ac"
)
FEATURE_CONTRACT_SHA256 = (
    "ad4d912439e438a345119adfb52de4384ac7d71d4584b79c59fd3f24ed01aa22"
)
VIEWS_MANIFEST_SHA256 = (
    "e880130ae672e97257343e7c1e57c91c5087dd9e303231dd0b738fe6c3b88bf5"
)
TARGETS_V2_SHA256 = "d03704fb856551cb5d2eee7810992f0e8fa23909860a9c28c626b5362456c6a6"
FEATURE_POLICY_SHA256 = (
    "fd8baeeaa82ba8e7d825762f30207c7875449a45a625d173bde419869c79b0a3"
)
MATRIX_MULTITASK_SHA256 = (
    "1b73dcb34a565f192cb66669163f71bac244884401e9c958a2cc836b2de36946"
)
EXPECTED_CONTEXT_FIELDS = ("age", "gender", "geo", "place", "platform", "agent")
EXPECTED_TARGET_NAMES = (
    "watch_time",
    "like",
    "dislike",
    "share",
    "bookmark",
    "click_on_author",
    "open_comments",
)
EXPECTED_TARGET_KINDS = (
    "regression",
    "binary",
    "binary",
    "binary",
    "binary",
    "binary",
    "binary",
)
EXPECTED_TARGET_COLUMNS = tuple(f"label_{name}" for name in EXPECTED_TARGET_NAMES)
EXPECTED_MASK_COLUMNS = tuple(f"label_mask_{name}" for name in EXPECTED_TARGET_NAMES)
EXPECTED_TRAIN_WEEKS = (20, 21, 22, 23, 24)
EXPECTED_VALIDATION_WEEK = 25
EXPECTED_TEST_WEEK = 26
VALIDITY_PREFIX = "denominator_validity__"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_SCAN_BYTES = 2 * 1024 * 1024
MAX_CACHE_METADATA_BYTES = 4 * 1024 * 1024
MIN_OVERLAY_AVAILABLE_BYTES = 10 * 1024**3
MAX_OVERLAY_USED_FRACTION = 0.85
FeatureRegime = Literal["C+P", "C+P+E"]
Split = Literal["train", "validation"]


class VKLSVDMultiTargetRuntimeError(RuntimeError):
    """A manifest, cache, batch, or test-firewall invariant was violated."""


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
            raise VKLSVDMultiTargetRuntimeError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise VKLSVDMultiTargetRuntimeError(f"{label} is not a SHA-256 digest")
    return value


def _load_views_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        verified = read_verified_artifact(path)
    except Exception as exc:
        raise VKLSVDMultiTargetRuntimeError(
            "seven-head views manifest or sidecar failed verification"
        ) from exc
    if verified.digest != VIEWS_MANIFEST_SHA256:
        raise VKLSVDMultiTargetRuntimeError("seven-head views manifest SHA drifted")
    if len(verified.data) > MAX_MANIFEST_BYTES:
        raise VKLSVDMultiTargetRuntimeError("seven-head views manifest is too large")
    try:
        payload = yaml.load(verified.data.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise VKLSVDMultiTargetRuntimeError(
            "seven-head views manifest is not strict YAML"
        ) from exc
    if not isinstance(payload, dict):
        raise VKLSVDMultiTargetRuntimeError("seven-head views manifest must be a map")
    _validate_views_manifest(payload)
    return payload, verified.digest


def _exact_string_tuple(value: Any, expected: tuple[str, ...], label: str) -> None:
    if not isinstance(value, list) or tuple(value) != expected:
        raise VKLSVDMultiTargetRuntimeError(f"{label} drifted")


def _validate_views_manifest(payload: Mapping[str, Any]) -> None:
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 2:
        raise VKLSVDMultiTargetRuntimeError("views schema_version must be exact int 2")
    if payload.get("dataset_id") != DATASET_ID:
        raise VKLSVDMultiTargetRuntimeError("views dataset_id drifted")
    if payload.get("materialization_state") != "complete_and_hash_verified":
        raise VKLSVDMultiTargetRuntimeError("views are not complete/hash verified")
    if payload.get("targets_multitask_v2_sha256") != TARGETS_V2_SHA256:
        raise VKLSVDMultiTargetRuntimeError("target-v2 binding drifted")
    if payload.get("feature_policy_sha256") != FEATURE_POLICY_SHA256:
        raise VKLSVDMultiTargetRuntimeError("feature-policy binding drifted")
    if payload.get("matrix_multitask_sha256") != MATRIX_MULTITASK_SHA256:
        raise VKLSVDMultiTargetRuntimeError("multi-target matrix binding drifted")
    for flag in (
        "test_unlocked",
        "test_labels_opened",
        "test_metrics_computed",
        "test_view_materialized",
        "validation_model_metrics_computed",
        "week26_file_open_attempted",
    ):
        if payload.get(flag) is not False:
            raise VKLSVDMultiTargetRuntimeError(f"test firewall flag drifted: {flag}")
    split = payload.get("split")
    if not isinstance(split, dict) or split != {
        "sealed_test_week": EXPECTED_TEST_WEEK,
        "train_weeks": list(EXPECTED_TRAIN_WEEKS),
        "validation_week": EXPECTED_VALIDATION_WEEK,
    }:
        raise VKLSVDMultiTargetRuntimeError("train/validation/test split drifted")
    targets = payload.get("targets")
    if not isinstance(targets, dict) or targets.get("all_labels_observed") is not True:
        raise VKLSVDMultiTargetRuntimeError("target observation contract drifted")
    _exact_string_tuple(targets.get("names"), EXPECTED_TARGET_NAMES, "targets.names")
    _exact_string_tuple(targets.get("kinds"), EXPECTED_TARGET_KINDS, "targets.kinds")
    _exact_string_tuple(
        targets.get("columns"), EXPECTED_TARGET_COLUMNS, "targets.columns"
    )
    _exact_string_tuple(
        targets.get("mask_columns"), EXPECTED_MASK_COLUMNS, "targets.mask_columns"
    )
    rows = payload.get("row_counts")
    if not isinstance(rows, dict) or rows != {
        "total": 9_022_651,
        "train": 7_505_820,
        "validation": 1_516_831,
    }:
        raise VKLSVDMultiTargetRuntimeError("row counts drifted")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise VKLSVDMultiTargetRuntimeError("views contract is missing")
    _exact_string_tuple(contract.get("ordered_C"), EXPECTED_CONTEXT_FIELDS, "ordered_C")
    _exact_string_tuple(
        contract.get("target_names"), EXPECTED_TARGET_NAMES, "contract target names"
    )
    shards = payload.get("shards")
    if not isinstance(shards, list) or len(shards) != 6:
        raise VKLSVDMultiTargetRuntimeError("exactly six week records are required")
    expected_roles = ["train"] * 5 + ["validation"]
    expected_weeks = [*EXPECTED_TRAIN_WEEKS, EXPECTED_VALIDATION_WEEK]
    for record, week, role in zip(shards, expected_weeks, expected_roles, strict=True):
        if not isinstance(record, dict) or record.get("week") != week:
            raise VKLSVDMultiTargetRuntimeError("shard week order drifted")
        if record.get("role") != role or type(record.get("rows")) is not int:
            raise VKLSVDMultiTargetRuntimeError("shard role/rows drifted")
        for manifest_key in ("REDUCED", "FULL"):
            source = record.get(manifest_key)
            if not isinstance(source, dict):
                raise VKLSVDMultiTargetRuntimeError("shard regime record missing")
            relative = _safe_relative_path(source.get("path"), "multi-target shard")
            expected_prefix = "REDUCED" if manifest_key == "REDUCED" else "FULL"
            if relative.parts[0] != expected_prefix or str(week) not in relative.name:
                raise VKLSVDMultiTargetRuntimeError("shard path/regime/week drifted")
            if type(source.get("rows")) is not int or source["rows"] != record["rows"]:
                raise VKLSVDMultiTargetRuntimeError("shard row count drifted")
            _require_sha256(source.get("sha256"), "shard SHA")


@dataclass(frozen=True)
class RegimeSpec:
    name: FeatureRegime
    manifest_key: Literal["REDUCED", "FULL"]
    ordered_operands: tuple[str, ...]
    denominator_mask: tuple[bool, ...]
    physical_schema_sha256: str

    @property
    def operand_schema_sha256(self) -> str:
        return canonical_sha256(self.ordered_operands)

    @property
    def denominator_mask_sha256(self) -> str:
        return canonical_sha256(self.denominator_mask)

    @property
    def expanded_schema_sha256(self) -> str:
        return canonical_sha256((*EXPECTED_CONTEXT_FIELDS, *self.ordered_operands))


def regime_spec(
    payload: Mapping[str, Any], feature_regime: FeatureRegime
) -> RegimeSpec:
    contract = payload["contract"]
    shared = tuple(contract["ordered_P_shared"])
    manual = tuple(contract["ordered_R_div"])
    mask = tuple(contract["ordered_denominator_mask"])
    if len(shared) != len(mask) or any(type(value) is not bool for value in mask):
        raise VKLSVDMultiTargetRuntimeError("shared operand mask drifted")
    hashes = contract["schema_hashes"]
    if feature_regime == "C+P":
        return RegimeSpec(
            name="C+P",
            manifest_key="REDUCED",
            ordered_operands=shared,
            denominator_mask=mask,
            physical_schema_sha256=hashes["REDUCED_physical"],
        )
    if feature_regime == "C+P+E":
        return RegimeSpec(
            name="C+P+E",
            manifest_key="FULL",
            ordered_operands=(*shared, *manual),
            denominator_mask=(*mask, *(False for _ in manual)),
            physical_schema_sha256=hashes["FULL_physical"],
        )
    raise VKLSVDMultiTargetRuntimeError("unknown feature regime")


def frozen_target_spec() -> MultiTargetSpec:
    return MultiTargetSpec(
        dataset_id=DATASET_ID,
        targets=tuple(
            TargetSpec(name=name, kind=kind)
            for name, kind in zip(
                EXPECTED_TARGET_NAMES, EXPECTED_TARGET_KINDS, strict=True
            )
        ),
        primary_confirmatory_target="watch_time",
    )


def task_constants_from_cache_metadata(
    metadata: Mapping[str, Any],
) -> TaskConstants:
    content = metadata.get("content_identity")
    if not isinstance(content, Mapping):
        raise VKLSVDMultiTargetRuntimeError("cache content identity is missing")
    state = content.get("task_constants")
    if not isinstance(state, Mapping) or set(state) != {
        "contract",
        "spec_sha256",
        "estimated_on_split",
        "constants",
    }:
        raise VKLSVDMultiTargetRuntimeError("task constants schema drifted")
    spec = frozen_target_spec()
    if (
        state.get("contract") != TaskConstants.CONTRACT
        or state.get("spec_sha256") != spec.contract_sha256
        or state.get("estimated_on_split") != "train"
    ):
        raise VKLSVDMultiTargetRuntimeError("task constants binding drifted")
    records = state.get("constants")
    if not isinstance(records, Mapping) or set(records) != set(spec.names):
        raise VKLSVDMultiTargetRuntimeError("task constants target set drifted")
    values: list[float] = []
    counts: list[int] = []
    for target in spec.targets:
        record = records[target.name]
        if not isinstance(record, Mapping) or set(record) != {
            "kind",
            "constant_name",
            "value",
            "observed_labels",
        }:
            raise VKLSVDMultiTargetRuntimeError("task constant record drifted")
        if (
            record.get("kind") != target.kind
            or record.get("constant_name") != target.constant_name
            or not isinstance(record.get("value"), str)
            or type(record.get("observed_labels")) is not int
        ):
            raise VKLSVDMultiTargetRuntimeError("task constant fields drifted")
        try:
            value = float(record["value"])
        except ValueError as exc:
            raise VKLSVDMultiTargetRuntimeError(
                "task constant value is not a float"
            ) from exc
        if repr(value) != record["value"]:
            raise VKLSVDMultiTargetRuntimeError("task constant float is not canonical")
        values.append(value)
        counts.append(record["observed_labels"])
    constants = TaskConstants(
        spec=spec,
        values=tuple(values),
        observed_counts=tuple(counts),
    )
    if constants.contract_sha256 != content.get("task_constants_sha256"):
        raise VKLSVDMultiTargetRuntimeError("task constants digest drifted")
    return constants


def _schema_sha256(parquet: pq.ParquetFile) -> str:
    return hashlib.sha256(str(parquet.schema_arrow).encode("utf-8")).hexdigest()


def scan_multitask_views(
    *, manifest_path: Path, views_root: Path
) -> tuple[dict[str, Any], str]:
    """Stream and attest all twelve train/validation shard files."""

    payload, manifest_digest = _load_views_manifest(manifest_path)
    records: list[dict[str, Any]] = []
    for week_record in payload["shards"]:
        scanned: dict[str, Any] = {
            "week": week_record["week"],
            "role": week_record["role"],
            "rows": week_record["rows"],
            "stable_row_ids_sha256": week_record["stable_row_ids_sha256"],
            "labels_and_masks_sha256": week_record["labels_and_masks_sha256"],
        }
        for key in ("REDUCED", "FULL"):
            source = week_record[key]
            relative = _safe_relative_path(source["path"], "multi-target shard")
            path = _resolve_beneath(views_root, relative, "multi-target shard")
            metadata = path.stat()
            digest = _sha256_file(path)
            parquet = pq.ParquetFile(path)
            if (
                digest != source["sha256"]
                or metadata.st_size != source["bytes"]
                or parquet.metadata.num_rows != source["rows"]
                or parquet.num_row_groups != source["row_groups"]
            ):
                raise VKLSVDMultiTargetRuntimeError(
                    f"{key} week {week_record['week']} differs from manifest"
                )
            names = set(parquet.schema_arrow.names)
            required = {
                *EXPECTED_CONTEXT_FIELDS,
                *EXPECTED_TARGET_COLUMNS,
                *EXPECTED_MASK_COLUMNS,
                "user_id",
                "stable_row_id",
            }
            if not required.issubset(names):
                raise VKLSVDMultiTargetRuntimeError("shard required schema drifted")
            scanned[key] = {
                "path": source["path"],
                "bytes": metadata.st_size,
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.num_row_groups,
                "sha256": digest,
                "arrow_schema_sha256": _schema_sha256(parquet),
            }
        records.append(scanned)
    scan = {
        "schema_version": 1,
        "artifact_role": "vk_lsvd_multitask_full_scan_v2",
        "dataset_id": DATASET_ID,
        "views_manifest_sha256": manifest_digest,
        "targets_multitask_v2_sha256": TARGETS_V2_SHA256,
        "feature_policy_sha256": FEATURE_POLICY_SHA256,
        "matrix_multitask_sha256": MATRIX_MULTITASK_SHA256,
        "scanned_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "shards": records,
        "row_counts": payload["row_counts"],
        "test_split_present": False,
        "week26_path_constructed": False,
        "model_metrics_computed": False,
    }
    return scan, hashlib.sha256(_canonical_json_bytes(scan)).hexdigest()


def _load_scan(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    try:
        verified = read_verified_artifact(path)
    except Exception as exc:
        raise VKLSVDMultiTargetRuntimeError(
            "scan artifact verification failed"
        ) from exc
    if verified.digest != expected_sha256:
        raise VKLSVDMultiTargetRuntimeError("scan artifact SHA drifted")
    if len(verified.data) > MAX_SCAN_BYTES:
        raise VKLSVDMultiTargetRuntimeError("scan artifact is too large")
    try:
        payload = json.loads(
            verified.data,
            object_pairs_hook=lambda pairs: _unique_json_pairs(pairs),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VKLSVDMultiTargetRuntimeError("scan artifact is not strict JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("views_manifest_sha256") != VIEWS_MANIFEST_SHA256
        or payload.get("test_split_present") is not False
        or payload.get("week26_path_constructed") is not False
        or payload.get("model_metrics_computed") is not False
    ):
        raise VKLSVDMultiTargetRuntimeError("scan artifact contract drifted")
    return payload, verified.digest


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VKLSVDMultiTargetRuntimeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _numpy(batch: Any, name: str) -> np.ndarray:
    column = batch.column(batch.schema.get_field_index(name))
    array = column.to_numpy(zero_copy_only=False)
    if array.dtype.kind == "O":
        raise VKLSVDMultiTargetRuntimeError(f"column {name} became object dtype")
    return array


_ARRAY_FILES = {
    "context_tokens": "context_tokens.i32.npy",
    "operand_features": "operand_features.f32.npy",
    "denominator_validity": "denominator_validity.u8.npy",
    "targets": "targets.f32.npy",
    "target_masks": "target_masks.u8.npy",
    "user_ids": "user_ids.i64.npy",
}


def _array_shape(name: str, rows: int, operands: int) -> tuple[int, ...]:
    if name == "context_tokens":
        return (rows, len(EXPECTED_CONTEXT_FIELDS))
    if name in {"operand_features", "denominator_validity"}:
        return (rows, operands)
    if name in {"targets", "target_masks"}:
        return (rows, len(EXPECTED_TARGET_NAMES))
    if name == "user_ids":
        return (rows,)
    raise AssertionError(name)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_immutable_json_artifact(path: Path, payload: Mapping[str, Any]) -> str:
    """Publish a canonical JSON/sidecar pair without following target links."""

    path = Path(path)
    if not path.name or not path.parent.is_dir() or path.parent.is_symlink():
        raise VKLSVDMultiTargetRuntimeError(
            "artifact parent must be an existing real directory"
        )
    data = _canonical_json_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar_data = f"{digest}  {path.name}\n".encode("ascii")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise VKLSVDMultiTargetRuntimeError("O_NOFOLLOW is required for publication")
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow,
    )
    created: list[str] = []
    try:
        for name, body in ((path.name, data), (sidecar.name, sidecar_data)):
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | nofollow,
                0o640,
                dir_fd=directory_fd,
            )
            try:
                view = memoryview(body)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short artifact write")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            created.append(name)
        os.fsync(directory_fd)
    except BaseException as exc:
        for name in reversed(created):
            with suppress(FileNotFoundError):
                os.unlink(name, dir_fd=directory_fd)
        with suppress(OSError):
            os.fsync(directory_fd)
        raise VKLSVDMultiTargetRuntimeError(
            f"immutable artifact publication failed: {path}"
        ) from exc
    finally:
        os.close(directory_fd)
    try:
        verified = read_verified_artifact(path)
    except Exception as exc:
        raise VKLSVDMultiTargetRuntimeError(
            "published artifact pair failed exact reopen"
        ) from exc
    if verified.digest != digest or verified.data != data:
        raise VKLSVDMultiTargetRuntimeError("published artifact bytes drifted")
    return digest


def _write_cache_metadata(directory: Path, payload: Mapping[str, Any]) -> str:
    data = _canonical_json_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    path = directory / "cache.json"
    sidecar = directory / "cache.json.sha256"
    path.write_bytes(data)
    sidecar.write_text(f"{digest}  cache.json\n", encoding="ascii")
    _fsync_file(path)
    _fsync_file(sidecar)
    return digest


def build_multitask_runtime_cache(
    *,
    manifest_path: Path,
    views_root: Path,
    scan_path: Path,
    expected_scan_sha256: str,
    cache_root: Path,
    feature_regime: FeatureRegime,
) -> tuple[dict[str, Any], str]:
    """Build one immutable train/validation NPY cache without test paths."""

    views, views_digest = _load_views_manifest(manifest_path)
    scan, scan_digest = _load_scan(scan_path, expected_scan_sha256)
    if scan_digest == views_digest:
        raise VKLSVDMultiTargetRuntimeError("scan and views digests must differ")
    spec = regime_spec(views, feature_regime)
    if cache_root.exists() or cache_root.is_symlink():
        raise VKLSVDMultiTargetRuntimeError("runtime cache already exists")
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    if stat.S_ISLNK(cache_root.parent.lstat().st_mode):
        raise VKLSVDMultiTargetRuntimeError("cache parent must not be a symlink")
    disk = shutil.disk_usage(cache_root.parent)
    estimated = views["row_counts"]["total"] * (
        4 * len(EXPECTED_CONTEXT_FIELDS)
        + 5 * len(spec.ordered_operands)
        + 5 * len(EXPECTED_TARGET_NAMES)
        + 8
    )
    if (
        disk.free - estimated < MIN_OVERLAY_AVAILABLE_BYTES
        or (disk.used + estimated) / disk.total > MAX_OVERLAY_USED_FRACTION
    ):
        raise VKLSVDMultiTargetRuntimeError("runtime cache storage guard failed")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{cache_root.name}.incomplete-", dir=cache_root.parent
        )
    )
    started = time.monotonic()
    split_records: dict[str, Any] = {}
    train_counts = np.zeros(len(EXPECTED_TARGET_NAMES), dtype=np.int64)
    train_sums = np.zeros(len(EXPECTED_TARGET_NAMES), dtype=np.float64)
    train_squares = np.zeros(len(EXPECTED_TARGET_NAMES), dtype=np.float64)
    try:
        scan_by_week = {(item["role"], item["week"]): item for item in scan["shards"]}
        for split in ("train", "validation"):
            week_records = [item for item in views["shards"] if item["role"] == split]
            rows = sum(item["rows"] for item in week_records)
            split_dir = temporary / split
            split_dir.mkdir(mode=0o700)
            operands = len(spec.ordered_operands)
            arrays = {
                "context_tokens": np.lib.format.open_memmap(
                    split_dir / _ARRAY_FILES["context_tokens"],
                    mode="w+",
                    dtype="<i4",
                    shape=_array_shape("context_tokens", rows, operands),
                ),
                "operand_features": np.lib.format.open_memmap(
                    split_dir / _ARRAY_FILES["operand_features"],
                    mode="w+",
                    dtype="<f4",
                    shape=_array_shape("operand_features", rows, operands),
                ),
                "denominator_validity": np.lib.format.open_memmap(
                    split_dir / _ARRAY_FILES["denominator_validity"],
                    mode="w+",
                    dtype="|u1",
                    shape=_array_shape("denominator_validity", rows, operands),
                ),
                "targets": np.lib.format.open_memmap(
                    split_dir / _ARRAY_FILES["targets"],
                    mode="w+",
                    dtype="<f4",
                    shape=_array_shape("targets", rows, operands),
                ),
                "target_masks": np.lib.format.open_memmap(
                    split_dir / _ARRAY_FILES["target_masks"],
                    mode="w+",
                    dtype="|u1",
                    shape=_array_shape("target_masks", rows, operands),
                ),
                "user_ids": np.lib.format.open_memmap(
                    split_dir / _ARRAY_FILES["user_ids"],
                    mode="w+",
                    dtype="<i8",
                    shape=_array_shape("user_ids", rows, operands),
                ),
            }
            offset = 0
            ranges: list[dict[str, Any]] = []
            p_shared = tuple(views["contract"]["ordered_P_shared"])
            validity_columns = tuple(f"{VALIDITY_PREFIX}{name}" for name in p_shared)
            columns = [
                *EXPECTED_CONTEXT_FIELDS,
                *spec.ordered_operands,
                *validity_columns,
                *EXPECTED_TARGET_COLUMNS,
                *EXPECTED_MASK_COLUMNS,
                "user_id",
            ]
            for record in week_records:
                source = record[spec.manifest_key]
                scanned = scan_by_week[(split, record["week"])][spec.manifest_key]
                if source["sha256"] != scanned["sha256"]:
                    raise VKLSVDMultiTargetRuntimeError("scan/source shard mismatch")
                relative = _safe_relative_path(source["path"], "runtime source shard")
                source_path = _resolve_beneath(
                    views_root, relative, "runtime source shard"
                )
                if _sha256_file(source_path) != source["sha256"]:
                    raise VKLSVDMultiTargetRuntimeError("runtime source shard drifted")
                parquet = pq.ParquetFile(source_path)
                start = offset
                for batch in parquet.iter_batches(
                    batch_size=PARQUET_BATCH_ROWS,
                    columns=columns,
                    use_threads=False,
                ):
                    count = batch.num_rows
                    end = offset + count
                    context = np.column_stack(
                        [_numpy(batch, name) for name in EXPECTED_CONTEXT_FIELDS]
                    )
                    for column, cardinality in enumerate(EXPECTED_CARDINALITIES):
                        values = context[:, column]
                        if int(values.min()) < 0 or int(values.max()) >= cardinality:
                            raise VKLSVDMultiTargetRuntimeError(
                                "context token exceeds frozen cardinality"
                            )
                    operand_values = np.column_stack(
                        [_numpy(batch, name) for name in spec.ordered_operands]
                    ).astype(np.float32, copy=False)
                    shared_validity = np.column_stack(
                        [_numpy(batch, name) for name in validity_columns]
                    )
                    validity = shared_validity
                    if spec.name == "C+P+E":
                        extra = len(spec.ordered_operands) - len(p_shared)
                        validity = np.column_stack(
                            (shared_validity, np.zeros((count, extra), dtype=np.bool_))
                        )
                    targets = np.column_stack(
                        [_numpy(batch, name) for name in EXPECTED_TARGET_COLUMNS]
                    ).astype(np.float32, copy=False)
                    masks = np.column_stack(
                        [_numpy(batch, name) for name in EXPECTED_MASK_COLUMNS]
                    )
                    user_ids = _numpy(batch, "user_id")
                    if (
                        not np.isfinite(operand_values).all()
                        or not np.isfinite(targets[masks]).all()
                        or masks.dtype != np.dtype("bool")
                    ):
                        raise VKLSVDMultiTargetRuntimeError(
                            "runtime source dtype/nonfinite violation"
                        )
                    if not bool(masks.all()):
                        raise VKLSVDMultiTargetRuntimeError(
                            "VK v2 declares every target always observed"
                        )
                    if np.any(
                        masks[:, 0] & ((targets[:, 0] < 0) | (targets[:, 0] > 1))
                    ):
                        raise VKLSVDMultiTargetRuntimeError(
                            "normalized watch-time target is outside [0,1]"
                        )
                    for target_index in range(1, len(EXPECTED_TARGET_NAMES)):
                        observed = targets[masks[:, target_index], target_index]
                        if np.any((observed != 0.0) & (observed != 1.0)):
                            raise VKLSVDMultiTargetRuntimeError(
                                "binary target contains a value outside {0,1}"
                            )
                    for column, eligible in enumerate(spec.denominator_mask):
                        if not eligible:
                            continue
                        column_validity = validity[:, column]
                        values = operand_values[:, column]
                        if np.any(column_validity & (values <= 0.0)):
                            raise VKLSVDMultiTargetRuntimeError(
                                "active denominator is not positive"
                            )
                    arrays["context_tokens"][offset:end] = context.astype(np.int32)
                    arrays["operand_features"][offset:end] = operand_values
                    arrays["denominator_validity"][offset:end] = validity.astype(
                        np.uint8
                    )
                    arrays["targets"][offset:end] = targets
                    arrays["target_masks"][offset:end] = masks.astype(np.uint8)
                    if user_ids.dtype.kind not in {"i", "u"}:
                        raise VKLSVDMultiTargetRuntimeError(
                            "user_id must be an integer identifier"
                        )
                    if user_ids.dtype.kind == "u" and np.any(
                        user_ids > np.iinfo(np.int64).max
                    ):
                        raise VKLSVDMultiTargetRuntimeError(
                            "user_id exceeds signed int64"
                        )
                    arrays["user_ids"][offset:end] = user_ids.astype(np.int64)
                    if split == "train":
                        targets64 = targets.astype(np.float64)
                        train_counts += masks.sum(axis=0, dtype=np.int64)
                        train_sums += targets64.sum(axis=0, dtype=np.float64)
                        train_squares += np.square(targets64).sum(
                            axis=0, dtype=np.float64
                        )
                    offset = end
                if offset - start != record["rows"]:
                    raise VKLSVDMultiTargetRuntimeError("streamed shard rows drifted")
                ranges.append(
                    {
                        "week": record["week"],
                        "start": start,
                        "stop": offset,
                        "rows": record["rows"],
                        "source_path": source["path"],
                        "source_sha256": source["sha256"],
                        "stable_row_ids_sha256": record["stable_row_ids_sha256"],
                    }
                )
            if offset != rows:
                raise VKLSVDMultiTargetRuntimeError("split row total drifted")
            for array in arrays.values():
                array.flush()
            del arrays
            array_records: dict[str, Any] = {}
            for name, filename in _ARRAY_FILES.items():
                path = split_dir / filename
                opened = np.load(path, mmap_mode="r", allow_pickle=False)
                shape = _array_shape(name, rows, len(spec.ordered_operands))
                if tuple(opened.shape) != shape:
                    raise VKLSVDMultiTargetRuntimeError("cache array shape drifted")
                del opened
                _fsync_file(path)
                array_records[name] = {
                    "path": f"{split}/{filename}",
                    "bytes": path.stat().st_size,
                    "shape": list(shape),
                    "sha256": _sha256_file(path),
                }
            split_records[split] = {
                "rows": rows,
                "source_shard_ranges": ranges,
                "arrays": array_records,
            }
        constant_values: list[float] = []
        for index, kind in enumerate(EXPECTED_TARGET_KINDS):
            count = int(train_counts[index])
            if count != views["row_counts"]["train"]:
                raise VKLSVDMultiTargetRuntimeError(
                    "train constant count differs from always-observed contract"
                )
            mean = float(train_sums[index] / count)
            if kind == "binary":
                if not 0.0 < mean < 1.0:
                    raise VKLSVDMultiTargetRuntimeError(
                        "binary train entropy is undefined"
                    )
                constant_values.append(
                    float(-(mean * math.log(mean) + (1.0 - mean) * math.log1p(-mean)))
                )
            else:
                variance = float(train_squares[index] / count - mean * mean)
                if not math.isfinite(variance) or variance <= 0.0:
                    raise VKLSVDMultiTargetRuntimeError(
                        "regression train variance is undefined"
                    )
                constant_values.append(variance)
        task_constants = TaskConstants(
            spec=frozen_target_spec(),
            values=tuple(constant_values),
            observed_counts=tuple(int(value) for value in train_counts),
        )
        content_identity = {
            "views_manifest_sha256": views_digest,
            "full_materialized_scan_sha256": scan_digest,
            "targets_multitask_v2_sha256": TARGETS_V2_SHA256,
            "feature_policy_sha256": FEATURE_POLICY_SHA256,
            "matrix_multitask_sha256": MATRIX_MULTITASK_SHA256,
            "feature_regime": spec.name,
            "ordered_context_fields": list(EXPECTED_CONTEXT_FIELDS),
            "ordered_operand_fields": list(spec.ordered_operands),
            "denominator_mask": list(spec.denominator_mask),
            "target_names": list(EXPECTED_TARGET_NAMES),
            "target_kinds": list(EXPECTED_TARGET_KINDS),
            "task_constants": task_constants.contract_state(),
            "task_constants_sha256": task_constants.contract_sha256,
            "splits": split_records,
        }
        content_digest = canonical_sha256(content_identity)
        metadata = {
            "schema_version": 1,
            "artifact_role": "vk_lsvd_multitask_raw_memmap_cache_v1",
            "cache_state": "complete_and_hash_verified",
            "built_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "dataset_id": DATASET_ID,
            "cache_content_sha256": content_digest,
            "content_identity": content_identity,
            "physical_schema_sha256": spec.physical_schema_sha256,
            "expanded_schema_sha256": spec.expanded_schema_sha256,
            "operand_schema_sha256": spec.operand_schema_sha256,
            "denominator_mask_sha256": spec.denominator_mask_sha256,
            "categorical_cardinalities": list(EXPECTED_CARDINALITIES),
            "test_split_present": False,
            "model_metrics_computed": False,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        metadata_digest = _write_cache_metadata(temporary, metadata)
        directory_fd = os.open(temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, cache_root)
        parent_fd = os.open(
            cache_root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return metadata, metadata_digest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@dataclass(frozen=True)
class MultiTargetSourceIdentity(RawSourceIdentity):
    targets_multitask_v2_sha256: str
    feature_policy_sha256: str
    matrix_multitask_sha256: str
    task_constants_sha256: str
    denominator_mask: tuple[bool, ...]
    target_names: tuple[str, ...]
    target_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except (TypeError, ValueError) as exc:
            raise VKLSVDMultiTargetRuntimeError(
                f"base raw source identity drifted: {exc}"
            ) from exc
        if self.dataset_id != DATASET_ID or self.split not in {"train", "validation"}:
            raise VKLSVDMultiTargetRuntimeError("source dataset/split drifted")
        for name in (
            "targets_multitask_v2_sha256",
            "feature_policy_sha256",
            "matrix_multitask_sha256",
            "task_constants_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            self.target_names != EXPECTED_TARGET_NAMES
            or self.target_kinds != EXPECTED_TARGET_KINDS
        ):
            raise VKLSVDMultiTargetRuntimeError("source target contract drifted")
        if (
            self.targets_multitask_v2_sha256 != TARGETS_V2_SHA256
            or self.feature_policy_sha256 != FEATURE_POLICY_SHA256
            or self.matrix_multitask_sha256 != MATRIX_MULTITASK_SHA256
            or self.ordered_context_fields != EXPECTED_CONTEXT_FIELDS
            or self.categorical_cardinalities != EXPECTED_CARDINALITIES
            or self.feature_regime not in {"C+P", "C+P+E"}
            or not self.ordered_operand_fields
            or len(self.ordered_operand_fields) != len(self.denominator_mask)
            or any(type(value) is not bool for value in self.denominator_mask)
        ):
            raise VKLSVDMultiTargetRuntimeError("source frozen identity drifted")


@dataclass(frozen=True)
class MultiTargetRawBatch:
    context_tokens: torch.Tensor
    operand_features: torch.Tensor
    denominator_validity: torch.Tensor
    targets: torch.Tensor
    target_masks: torch.Tensor
    user_ids: torch.Tensor
    global_row_indices: torch.Tensor

    def validate(
        self,
        identity: MultiTargetSourceIdentity,
        requested_indices: Sequence[int],
        *,
        runtime_value_checks: bool,
    ) -> None:
        requested = tuple(requested_indices)
        batch_size = len(requested)
        expected = {
            "context_tokens": (batch_size, len(identity.ordered_context_fields)),
            "operand_features": (batch_size, len(identity.ordered_operand_fields)),
            "denominator_validity": (batch_size, len(identity.ordered_operand_fields)),
            "targets": (batch_size, len(identity.target_names)),
            "target_masks": (batch_size, len(identity.target_names)),
            "user_ids": (batch_size,),
            "global_row_indices": (batch_size,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise VKLSVDMultiTargetRuntimeError(f"batch {name} shape drifted")
        if (
            self.context_tokens.dtype != torch.int64
            or self.operand_features.dtype != torch.float32
            or self.denominator_validity.dtype != torch.bool
            or self.targets.dtype != torch.float32
            or self.target_masks.dtype != torch.bool
            or self.user_ids.dtype != torch.int64
            or self.global_row_indices.dtype != torch.int64
        ):
            raise VKLSVDMultiTargetRuntimeError("batch dtype contract drifted")
        feature_devices = {
            self.context_tokens.device,
            self.operand_features.device,
            self.denominator_validity.device,
            self.targets.device,
            self.target_masks.device,
        }
        if len(feature_devices) != 1:
            raise VKLSVDMultiTargetRuntimeError("batch model tensors differ in device")
        if (
            self.user_ids.device.type != "cpu"
            or self.global_row_indices.device.type != "cpu"
        ):
            raise VKLSVDMultiTargetRuntimeError("audit IDs must remain CPU resident")
        if not torch.equal(
            self.global_row_indices, torch.tensor(requested, dtype=torch.int64)
        ):
            raise VKLSVDMultiTargetRuntimeError("batch row order drifted")
        if runtime_value_checks:
            if not bool(torch.isfinite(self.operand_features).all()):
                raise VKLSVDMultiTargetRuntimeError("batch operands are non-finite")
            if not bool(torch.isfinite(self.targets[self.target_masks]).all()):
                raise VKLSVDMultiTargetRuntimeError("observed targets are non-finite")
            if not bool(self.target_masks.all()):
                raise VKLSVDMultiTargetRuntimeError(
                    "VK v2 batch violates always-observed target masks"
                )


class VKLSVDMultiTargetSource:
    """Process-local read-only source backed by verified NPY memmaps."""

    CONTRACT: ClassVar[str] = "vk-lsvd-multitarget-memmap-source-v1"

    def __init__(
        self,
        cache_root: Path,
        *,
        split: Split,
        expected_cache_metadata_sha256: str,
        expected_cache_content_sha256: str,
        expected_scan_sha256: str,
        runtime_value_checks: bool,
        verify_array_hashes: bool = True,
    ) -> None:
        if split not in {"train", "validation"}:
            raise VKLSVDMultiTargetRuntimeError("only train/validation are allowed")
        if not isinstance(runtime_value_checks, bool) or not isinstance(
            verify_array_hashes, bool
        ):
            raise TypeError("runtime/hash flags must be boolean")
        verified = read_verified_artifact(cache_root / "cache.json")
        if verified.digest != expected_cache_metadata_sha256:
            raise VKLSVDMultiTargetRuntimeError("cache metadata SHA drifted")
        if len(verified.data) > MAX_CACHE_METADATA_BYTES:
            raise VKLSVDMultiTargetRuntimeError("cache metadata is too large")
        metadata = json.loads(
            verified.data,
            object_pairs_hook=lambda pairs: _unique_json_pairs(pairs),
        )
        if (
            metadata.get("cache_state") != "complete_and_hash_verified"
            or metadata.get("test_split_present") is not False
            or metadata.get("model_metrics_computed") is not False
        ):
            raise VKLSVDMultiTargetRuntimeError("cache firewall/state drifted")
        content = metadata.get("content_identity")
        if not isinstance(content, dict):
            raise VKLSVDMultiTargetRuntimeError("cache content identity is missing")
        content_digest = canonical_sha256(content)
        if (
            content_digest != metadata.get("cache_content_sha256")
            or content_digest != expected_cache_content_sha256
            or content.get("full_materialized_scan_sha256") != expected_scan_sha256
            or content.get("views_manifest_sha256") != VIEWS_MANIFEST_SHA256
        ):
            raise VKLSVDMultiTargetRuntimeError("cache content binding drifted")
        constants = task_constants_from_cache_metadata(metadata)
        split_record = content["splits"][split]
        arrays: dict[str, np.ndarray] = {}
        for name, record in split_record["arrays"].items():
            relative = _safe_relative_path(record["path"], "cache array")
            path = _resolve_beneath(cache_root, relative, "cache array")
            if verify_array_hashes and _sha256_file(path) != record["sha256"]:
                raise VKLSVDMultiTargetRuntimeError("cache array SHA drifted")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                list(array.shape) != record["shape"]
                or path.stat().st_size != record["bytes"]
            ):
                raise VKLSVDMultiTargetRuntimeError("cache array metadata drifted")
            arrays[name] = array
        operands = tuple(content["ordered_operand_fields"])
        mask = tuple(content["denominator_mask"])
        self._pid = os.getpid()
        self._arrays = arrays
        self._runtime_value_checks = runtime_value_checks
        self._identity = MultiTargetSourceIdentity(
            dataset_id=DATASET_ID,
            split=split,
            feature_regime=content["feature_regime"],
            row_count=split_record["rows"],
            data_manifest_sha256=DATA_MANIFEST_SHA256,
            views_manifest_sha256=content["views_manifest_sha256"],
            feature_contract_sha256=FEATURE_CONTRACT_SHA256,
            full_materialized_scan_sha256=content["full_materialized_scan_sha256"],
            cache_content_sha256=content_digest,
            targets_multitask_v2_sha256=content["targets_multitask_v2_sha256"],
            feature_policy_sha256=content["feature_policy_sha256"],
            matrix_multitask_sha256=content["matrix_multitask_sha256"],
            physical_schema_sha256=metadata["physical_schema_sha256"],
            expanded_schema_sha256=metadata["expanded_schema_sha256"],
            operand_schema_sha256=metadata["operand_schema_sha256"],
            denominator_mask_sha256=metadata["denominator_mask_sha256"],
            task_constants_sha256=constants.contract_sha256,
            ordered_context_fields=tuple(content["ordered_context_fields"]),
            categorical_cardinalities=tuple(metadata["categorical_cardinalities"]),
            ordered_operand_fields=operands,
            denominator_mask=mask,
            target_names=tuple(content["target_names"]),
            target_kinds=tuple(content["target_kinds"]),
        )

    @property
    def identity(self) -> MultiTargetSourceIdentity:
        return self._identity

    @property
    def runtime_value_checks(self) -> bool:
        return self._runtime_value_checks

    def __len__(self) -> int:
        return self._identity.row_count

    def close(self) -> None:
        self._arrays = {}

    def __del__(self) -> None:
        with suppress(AttributeError, BufferError):
            self.close()

    def batch(
        self, indices: Sequence[int], *, device: torch.device
    ) -> MultiTargetRawBatch:
        if os.getpid() != self._pid:
            raise VKLSVDMultiTargetRuntimeError("memmaps cannot cross processes")
        if isinstance(indices, str | bytes) or not isinstance(indices, Sequence):
            raise VKLSVDMultiTargetRuntimeError("indices must be a bounded Sequence")
        requested = tuple(indices)
        if not requested or len(requested) > MAX_BATCH_ROWS:
            raise VKLSVDMultiTargetRuntimeError("batch size is outside bounds")
        if any(
            type(index) is not int or index < 0 or index >= len(self)
            for index in requested
        ):
            raise VKLSVDMultiTargetRuntimeError("batch index is invalid")
        selection = np.asarray(requested, dtype=np.int64)
        target_device = torch.device(device)
        result = MultiTargetRawBatch(
            context_tokens=torch.from_numpy(
                np.asarray(self._arrays["context_tokens"][selection], dtype=np.int32)
            ).to(device=target_device, dtype=torch.int64),
            operand_features=torch.from_numpy(
                np.asarray(
                    self._arrays["operand_features"][selection], dtype=np.float32
                )
            ).to(device=target_device),
            denominator_validity=torch.from_numpy(
                np.asarray(
                    self._arrays["denominator_validity"][selection], dtype=np.uint8
                )
            ).to(device=target_device, dtype=torch.bool),
            targets=torch.from_numpy(
                np.asarray(self._arrays["targets"][selection], dtype=np.float32)
            ).to(device=target_device),
            target_masks=torch.from_numpy(
                np.asarray(self._arrays["target_masks"][selection], dtype=np.uint8)
            ).to(device=target_device, dtype=torch.bool),
            user_ids=torch.from_numpy(
                np.asarray(self._arrays["user_ids"][selection], dtype=np.int64)
            ),
            global_row_indices=torch.tensor(requested, dtype=torch.int64),
        )
        result.validate(
            self._identity,
            requested,
            runtime_value_checks=self._runtime_value_checks,
        )
        return result


__all__ = [
    "DATASET_ID",
    "DATA_MANIFEST_SHA256",
    "EXPECTED_CONTEXT_FIELDS",
    "EXPECTED_TARGET_KINDS",
    "EXPECTED_TARGET_NAMES",
    "FEATURE_POLICY_SHA256",
    "FEATURE_CONTRACT_SHA256",
    "MATRIX_MULTITASK_SHA256",
    "MultiTargetRawBatch",
    "MultiTargetSourceIdentity",
    "TARGETS_V2_SHA256",
    "VIEWS_MANIFEST_SHA256",
    "VKLSVDMultiTargetRuntimeError",
    "VKLSVDMultiTargetSource",
    "build_multitask_runtime_cache",
    "frozen_target_spec",
    "publish_immutable_json_artifact",
    "regime_spec",
    "scan_multitask_views",
    "task_constants_from_cache_metadata",
]
