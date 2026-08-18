"""Deterministic train/validation materialization for frozen VK-LSVD views.

Week 26 is deliberately absent from every path constructor and execution
branch in this module.  The public entry point verifies the frozen source and
feature contracts, fits all history state on weeks 0--19, fits categorical
mappings and scale-only normalizers on weeks 20--24, and emits only train and
validation shards.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import implicit
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from dcdn.data.vk_lsvd import (
    ALPHA_SMOOTH,
    BETA_SMOOTH,
    DENOMINATOR_ELIGIBLE_FIELDS,
    HISTORY_WEEKS,
    IALS_SIGNALS,
    ITEM_RATE_SOURCES,
    LABEL_DURATION_Q99_SECONDS,
    ORDERED_C_FIELDS,
    ORDERED_C_PLUS_P_FLAT_SCHEMA,
    ORDERED_C_PLUS_P_PLUS_E_FLAT_SCHEMA,
    ORDERED_DENOMINATOR_MASK,
    ORDERED_E_FIELDS,
    ORDERED_NUMERIC_OPERANDS,
    ORDERED_P0_FIELDS,
    ORDERED_V_FIELDS,
    RECIPROCAL_EPSILON,
    SOURCE_MANIFEST_PATH,
    TRAIN_WEEKS,
    USER_RATE_SOURCES,
    VALIDATION_WEEK,
    VKLSVDContractError,
    floor_indicator_name,
    load_feature_contract,
    ordered_schema_sha256,
    semantic_week_relative_path,
    stable_row_id,
    verify_source_manifest_reference,
)
from dcdn.source_manifest import load_source_manifest

ALGORITHM_VERSION = "vk-lsvd-materialize-v1"
FEATURE_CONTRACT_SHA256 = (
    "380d105c847a73afb4927740e40401124688f895142792309b4fc8267a2a5f2b"
)
SOURCE_MANIFEST_SHA256 = (
    "5dbbfeb23442a6b1ebdb829dc239c6a3c6c655d71446cde4551b83cc6af456ac"
)
MODEL_SOURCES_SHA256 = (
    "0844a838e5a68029299a52aaa31b5f74205fe57fd486b51544061b32c571a3f4"
)
SEARCH_SPACE_SHA256 = "b263b55da768ec33656ebdb1506d58f2809fd5826aa8b4976aedaecee53bb344"
EXPERIMENT_MATRIX_SHA256 = (
    "867118db089bbbffb4abf02ff7b3c01e2c9f43e804a188a27a3886f66db4e000"
)

DEFAULT_RAW_ROOT = Path("/home/codex/.cache/dcdn-public/raw/vk_lsvd")
DEFAULT_OUTPUT_ROOT = Path("/home/codex/.cache/dcdn-public/processed/vk_lsvd")
DEFAULT_FEATURE_CONTRACT = Path("configs/data/vk_lsvd/features_v1.yaml")
DEFAULT_VIEWS_MANIFEST = Path("data/manifests/vk_lsvd_views_v1.yaml")
DEFAULT_COST_LEDGER = Path("docs/cost_ledger.md")

MIN_OVERLAY_AVAILABLE_BYTES = 20 * 1024**3
MAX_OVERLAY_USAGE_FRACTION = 0.85
ASSUMED_POD_RATE_USD_PER_HOUR = 3.29
STORAGE_RESERVE_USD = 25.0
WARNING_THRESHOLD_USD = 370.0
HARD_CAP_USD = 460.0
FINALIZATION_RESERVE_HOURS = 8.0
MAX_ALLOWED_ETA_HOURS = 2.0
PARQUET_BATCH_ROWS = 262_144
PARQUET_ROW_GROUP_ROWS = 131_072
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_SIDECAR_BYTES = 256
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

HISTORY_VALUE_COLUMNS = (
    "user_id",
    "item_id",
    "timespent",
    "like",
    "dislike",
    "share",
    "bookmark",
    "click_on_author",
    "open_comments",
)
TARGET_VALUE_COLUMNS = (
    "user_id",
    "item_id",
    "place",
    "platform",
    "agent",
    "timespent",
)
BASE_V_FIELDS = tuple(
    field for field in ORDERED_V_FIELDS if "_reciprocal_floor_active" not in field
)
DENOMINATOR_VALIDITY_COLUMNS = tuple(
    f"denominator_validity__{field}" for field in ORDERED_NUMERIC_OPERANDS
)


class VKLSVDMaterializationError(RuntimeError):
    """Raised when a materialization gate or invariant fails closed."""


@dataclass(frozen=True)
class IALSFitConfig:
    factors: int = 32
    iterations: int = 15
    regularization: float = 0.01
    random_state: int = 42
    threads: int = 1

    def validate(self) -> None:
        if self.factors <= 0 or self.iterations <= 0 or self.threads != 1:
            raise VKLSVDMaterializationError(
                "iALS geometry must be positive and single-threaded"
            )
        if not math.isfinite(self.regularization) or self.regularization <= 0:
            raise VKLSVDMaterializationError("iALS regularization must be positive")


@dataclass(frozen=True)
class ExplicitInputPaths:
    history: tuple[tuple[int, Path], ...]
    train: tuple[tuple[int, Path], ...]
    validation: tuple[tuple[int, Path], ...]
    users_metadata: Path
    items_metadata: Path

    def all_interactions(self) -> tuple[tuple[int, Path], ...]:
        return self.history + self.train + self.validation


@dataclass(frozen=True)
class GroupLookup:
    keys: np.ndarray
    show_count: np.ndarray
    timespent_sum: np.ndarray
    event_counts: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class IALSLookup:
    user_keys: np.ndarray
    item_keys: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_norms: np.ndarray
    item_norms: np.ndarray
    fingerprint_sha256: str


@dataclass(frozen=True)
class HistoryState:
    users: GroupLookup
    items: GroupLookup
    ials: Mapping[str, IALSLookup]
    history_rows: int
    fingerprint_sha256: str


@dataclass(frozen=True)
class MetadataLookup:
    user_keys: np.ndarray
    user_age: np.ndarray
    user_gender: np.ndarray
    user_geo: np.ndarray
    item_keys: np.ndarray
    item_duration: np.ndarray
    fingerprint_sha256: str


@dataclass(frozen=True)
class StageResult:
    week: int
    role: str
    path: Path
    rows: int
    categorical_values: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class FinalShard:
    week: int
    role: str
    rows: int
    c_plus_p_path: Path
    c_plus_p_bytes: int
    c_plus_p_sha256: str
    c_plus_p_plus_e_path: Path
    c_plus_p_plus_e_bytes: int
    c_plus_p_plus_e_sha256: str
    shared_values_sha256: str
    stable_row_ids_sha256: str
    c_plus_p_row_groups: int
    c_plus_p_plus_e_row_groups: int


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(digest: Any, label: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes(order="C"))


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise VKLSVDMaterializationError(f"cannot inspect {label}: {path}") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or path.is_symlink()
    ):
        raise VKLSVDMaterializationError(f"{label} must be a single-link regular file")
    return metadata


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise VKLSVDMaterializationError("cannot resolve an existing output parent")
        candidate = candidate.parent
    if candidate.is_symlink() or not candidate.is_dir():
        raise VKLSVDMaterializationError("output parent must be a real directory")
    return candidate


def _filesystem_guard(path: Path) -> dict[str, Any]:
    parent = _nearest_existing_parent(path)
    stats = os.statvfs(parent)
    total = stats.f_blocks * stats.f_frsize
    available = stats.f_bavail * stats.f_frsize
    used_fraction = 1.0 - available / total
    if available < MIN_OVERLAY_AVAILABLE_BYTES:
        raise VKLSVDMaterializationError(
            "less than 20 GiB remains on the output filesystem"
        )
    if used_fraction >= MAX_OVERLAY_USAGE_FRACTION:
        raise VKLSVDMaterializationError("output filesystem usage is at or above 85%")
    return {
        "checked_path": str(parent.resolve()),
        "total_bytes": total,
        "available_bytes": available,
        "used_fraction": used_fraction,
        "minimum_available_bytes": MIN_OVERLAY_AVAILABLE_BYTES,
        "maximum_used_fraction": MAX_OVERLAY_USAGE_FRACTION,
    }


def _parse_ledger_epoch(ledger_path: Path) -> datetime:
    text = ledger_path.read_text(encoding="utf-8")
    match = re.search(r"ledger_epoch_utc:\s*([0-9T:\-]+Z)", text)
    if match is None:
        raise VKLSVDMaterializationError("ledger_epoch_utc is missing")
    return datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))


def resource_and_cost_guard(
    *,
    output_root: Path,
    ledger_path: Path,
    eta_hours: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fail before work that would cross local storage, time, or cost gates."""

    if not math.isfinite(eta_hours) or not 0 < eta_hours <= MAX_ALLOWED_ETA_HOURS:
        raise VKLSVDMaterializationError("eta_hours must be in (0, 2]")
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise VKLSVDMaterializationError("guard time must be timezone-aware")
    epoch = _parse_ledger_epoch(ledger_path)
    elapsed_hours = max(0.0, (current - epoch).total_seconds() / 3600.0)
    current_estimate = (
        elapsed_hours * ASSUMED_POD_RATE_USD_PER_HOUR + STORAGE_RESERVE_USD
    )
    projected_estimate = (
        elapsed_hours + eta_hours
    ) * ASSUMED_POD_RATE_USD_PER_HOUR + STORAGE_RESERVE_USD
    epoch_stop = epoch + timedelta(hours=120)
    absolute_stop = datetime(2026, 8, 24, 11, 0, tzinfo=UTC)
    effective_stop = min(epoch_stop, absolute_stop)
    reserve_start = effective_stop - timedelta(hours=FINALIZATION_RESERVE_HOURS)
    if current + timedelta(hours=eta_hours) > reserve_start:
        raise VKLSVDMaterializationError("ETA crosses the finalization reserve")
    if projected_estimate >= HARD_CAP_USD:
        raise VKLSVDMaterializationError(
            "projected local-accounting estimate reaches hard cap"
        )
    return {
        "ledger_epoch_utc": epoch.isoformat().replace("+00:00", "Z"),
        "checked_utc": current.isoformat().replace("+00:00", "Z"),
        "eta_hours": eta_hours,
        "assumed_pod_rate_usd_per_hour": ASSUMED_POD_RATE_USD_PER_HOUR,
        "storage_reserve_usd": STORAGE_RESERVE_USD,
        "current_estimated_incremental_spend_usd": round(current_estimate, 4),
        "projected_estimated_incremental_spend_usd": round(projected_estimate, 4),
        "warning_threshold_usd": WARNING_THRESHOLD_USD,
        "warning_threshold_crossed": projected_estimate >= WARNING_THRESHOLD_USD,
        "hard_cap_usd": HARD_CAP_USD,
        "effective_stop_utc": effective_stop.isoformat().replace("+00:00", "Z"),
        "finalization_reserve_start_utc": reserve_start.isoformat().replace(
            "+00:00", "Z"
        ),
        "control_plane_values_used": False,
        "filesystem": _filesystem_guard(output_root),
    }


def frozen_input_paths(raw_root: Path) -> ExplicitInputPaths:
    """Construct only the explicit unsealed paths; week 26 is unreachable."""

    history = tuple(
        (week, raw_root.joinpath(*semantic_week_relative_path(week).parts))
        for week in HISTORY_WEEKS
    )
    train = tuple(
        (week, raw_root.joinpath(*semantic_week_relative_path(week).parts))
        for week in TRAIN_WEEKS
    )
    validation = (
        (
            VALIDATION_WEEK,
            raw_root.joinpath(*semantic_week_relative_path(VALIDATION_WEEK).parts),
        ),
    )
    return ExplicitInputPaths(
        history=history,
        train=train,
        validation=validation,
        users_metadata=raw_root / "metadata/users_metadata.parquet",
        items_metadata=raw_root / "metadata/items_metadata.parquet",
    )


def _verify_lock(path: Path, expected: str, label: str) -> None:
    _require_regular_file(path, label)
    if _sha256_file(path) != expected:
        raise VKLSVDMaterializationError(f"{label} SHA-256 drifted")


def verify_frozen_inputs(
    *, repository_root: Path, raw_root: Path, contract_path: Path
) -> tuple[dict[str, Any], ExplicitInputPaths, list[dict[str, Any]]]:
    """Verify all allowed source bytes without touching the sealed test object."""

    _verify_lock(contract_path, FEATURE_CONTRACT_SHA256, "feature contract")
    _verify_lock(
        repository_root / "configs/model_sources_v1.yaml",
        MODEL_SOURCES_SHA256,
        "model-source lock",
    )
    _verify_lock(
        repository_root / "configs/search_space_v1.yaml",
        SEARCH_SPACE_SHA256,
        "search-space lock",
    )
    _verify_lock(
        repository_root / "configs/experiment_matrix_v1.yaml",
        EXPERIMENT_MATRIX_SHA256,
        "experiment-matrix lock",
    )
    try:
        contract = load_feature_contract(contract_path)
        digest = verify_source_manifest_reference(
            contract, repository_root=repository_root
        )
    except VKLSVDContractError as error:
        raise VKLSVDMaterializationError(str(error)) from error
    if digest != SOURCE_MANIFEST_SHA256:
        raise VKLSVDMaterializationError("source manifest SHA-256 drifted")
    source = load_source_manifest(repository_root / SOURCE_MANIFEST_PATH)
    object_by_path = {record["path"]: record for record in source["objects"]}
    paths = frozen_input_paths(raw_root)
    allowed = [
        (semantic_week_relative_path(week).as_posix(), path)
        for week, path in paths.all_interactions()
    ]
    allowed.extend(
        [
            ("metadata/users_metadata.parquet", paths.users_metadata),
            ("metadata/items_metadata.parquet", paths.items_metadata),
        ]
    )
    verified: list[dict[str, Any]] = []
    for relative, local in allowed:
        expected = object_by_path.get(relative)
        if not isinstance(expected, Mapping):
            raise VKLSVDMaterializationError(
                f"source object is not allowlisted: {relative}"
            )
        metadata = _require_regular_file(local, f"source object {relative}")
        if metadata.st_size != expected.get("bytes"):
            raise VKLSVDMaterializationError(f"source size drifted: {relative}")
        digest = _sha256_file(local)
        if digest != expected.get("sha256"):
            raise VKLSVDMaterializationError(f"source SHA-256 drifted: {relative}")
        verified.append({"path": relative, "bytes": metadata.st_size, "sha256": digest})
    return contract, paths, verified


def _read_batches(path: Path, columns: Sequence[str]) -> Iterable[pa.RecordBatch]:
    _require_regular_file(path, "Parquet source")
    try:
        parquet = pq.ParquetFile(path)
        yield from parquet.iter_batches(
            batch_size=PARQUET_BATCH_ROWS,
            columns=list(columns),
            use_threads=False,
        )
    except Exception as error:
        raise VKLSVDMaterializationError(
            f"cannot stream Parquet source: {path}"
        ) from error


def _batch_numpy(batch: pa.RecordBatch, field: str) -> np.ndarray:
    index = batch.schema.get_field_index(field)
    if index < 0:
        raise VKLSVDMaterializationError(f"required source column is missing: {field}")
    column = batch.column(index)
    if column.null_count:
        raise VKLSVDMaterializationError(
            f"required source column contains nulls: {field}"
        )
    return column.to_numpy(zero_copy_only=False)


def _load_history_arrays(
    paths: Sequence[tuple[int, Path]],
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {field: [] for field in HISTORY_VALUE_COLUMNS}
    for _, path in paths:
        for batch in _read_batches(path, HISTORY_VALUE_COLUMNS):
            for field in HISTORY_VALUE_COLUMNS:
                chunks[field].append(_batch_numpy(batch, field))
    arrays = {field: np.concatenate(parts) for field, parts in chunks.items()}
    rows = len(arrays["user_id"])
    if rows == 0 or any(len(value) != rows for value in arrays.values()):
        raise VKLSVDMaterializationError("history arrays are empty or misaligned")
    return arrays


def _group_lookup(
    *,
    keys: np.ndarray,
    timespent: np.ndarray,
    events: Mapping[str, np.ndarray],
) -> GroupLookup:
    frame = pl.DataFrame(
        {
            "key": keys,
            "timespent": timespent.astype(np.uint64, copy=False),
            **{
                name: value.astype(np.uint8, copy=False)
                for name, value in events.items()
            },
        }
    )
    grouped = (
        frame.group_by("key")
        .agg(
            pl.len().cast(pl.UInt64).alias("show_count"),
            pl.col("timespent").sum().cast(pl.Float64).alias("timespent_sum"),
            *(pl.col(name).sum().cast(pl.UInt64).alias(name) for name in events),
        )
        .sort("key")
    )
    result = GroupLookup(
        keys=grouped["key"].to_numpy(),
        show_count=grouped["show_count"].to_numpy(),
        timespent_sum=grouped["timespent_sum"].to_numpy(),
        event_counts={name: grouped[name].to_numpy() for name in events},
    )
    if len(result.keys) == 0 or np.any(result.keys[1:] <= result.keys[:-1]):
        raise VKLSVDMaterializationError("history group keys are not unique ascending")
    return result


def _fingerprint_ials(
    user_keys: np.ndarray,
    item_keys: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_norms: np.ndarray,
    item_norms: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for label, value in (
        ("user_keys", user_keys),
        ("item_keys", item_keys),
        ("user_factors", user_factors),
        ("item_factors", item_factors),
        ("user_norms", user_norms),
        ("item_norms", item_norms),
    ):
        _sha256_array(digest, label, value)
    return digest.hexdigest()


def _fit_one_ials(
    *,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    values: np.ndarray,
    user_keys: np.ndarray,
    item_keys: np.ndarray,
    config: IALSFitConfig,
) -> IALSLookup:
    matrix = csr_matrix(
        (
            values.astype(np.float32, copy=False),
            (row_indices, column_indices),
        ),
        shape=(len(user_keys), len(item_keys)),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.nnz == 0:
        raise VKLSVDMaterializationError("an iALS history signal has no positive mass")
    with threadpool_limits(limits=1):
        model = implicit.als.AlternatingLeastSquares(
            factors=config.factors,
            iterations=config.iterations,
            regularization=config.regularization,
            random_state=config.random_state,
            num_threads=config.threads,
        )
        model.fit(matrix, show_progress=False)
    user_factors = np.asarray(model.user_factors, dtype=np.float32, order="C")
    item_factors = np.asarray(model.item_factors, dtype=np.float32, order="C")
    user_norms = np.linalg.norm(user_factors, axis=1).astype(np.float32)
    item_norms = np.linalg.norm(item_factors, axis=1).astype(np.float32)
    for value in (user_factors, item_factors, user_norms, item_norms):
        if not np.isfinite(value).all():
            raise VKLSVDMaterializationError("iALS produced a non-finite factor")
    return IALSLookup(
        user_keys=user_keys,
        item_keys=item_keys,
        user_factors=user_factors,
        item_factors=item_factors,
        user_norms=user_norms,
        item_norms=item_norms,
        fingerprint_sha256=_fingerprint_ials(
            user_keys,
            item_keys,
            user_factors,
            item_factors,
            user_norms,
            item_norms,
        ),
    )


def build_history_state(
    paths: Sequence[tuple[int, Path]], *, config: IALSFitConfig
) -> HistoryState:
    """Fit aggregate and iALS state using only explicitly supplied history paths."""

    config.validate()
    arrays = _load_history_arrays(paths)
    timespent = arrays["timespent"]
    survival15 = timespent >= 15
    survival30 = timespent >= 30
    feedback = (
        arrays["like"]
        | arrays["share"]
        | arrays["bookmark"]
        | arrays["click_on_author"]
        | arrays["open_comments"]
    )
    user_events = {
        "like": arrays["like"],
        "share": arrays["share"],
        "bookmark": arrays["bookmark"],
        "open_comments": arrays["open_comments"],
        "dislike": arrays["dislike"],
        "survival_15": survival15,
        "survival_30": survival30,
    }
    item_events = {
        "like": arrays["like"],
        "share": arrays["share"],
        "bookmark": arrays["bookmark"],
        "open_comments": arrays["open_comments"],
        "dislike": arrays["dislike"],
        "click_on_author": arrays["click_on_author"],
        "survival_15": survival15,
        "survival_30": survival30,
    }
    users = _group_lookup(
        keys=arrays["user_id"], timespent=timespent, events=user_events
    )
    items = _group_lookup(
        keys=arrays["item_id"], timespent=timespent, events=item_events
    )
    row_indices = np.searchsorted(users.keys, arrays["user_id"]).astype(np.int32)
    column_indices = np.searchsorted(items.keys, arrays["item_id"]).astype(np.int32)
    signals = {
        "viewtime": timespent.astype(np.float32),
        "feedback": feedback.astype(np.float32),
        "survival": survival15.astype(np.float32),
    }
    ials = {
        signal: _fit_one_ials(
            row_indices=row_indices,
            column_indices=column_indices,
            values=signals[signal],
            user_keys=users.keys,
            item_keys=items.keys,
            config=config,
        )
        for signal in IALS_SIGNALS
    }
    digest = hashlib.sha256()
    for label, value in (
        ("user_keys", users.keys),
        ("user_shows", users.show_count),
        ("user_timespent", users.timespent_sum),
        ("item_keys", items.keys),
        ("item_shows", items.show_count),
        ("item_timespent", items.timespent_sum),
    ):
        _sha256_array(digest, label, value)
    for prefix, lookup in (("user", users), ("item", items)):
        for name in sorted(lookup.event_counts):
            _sha256_array(digest, f"{prefix}_{name}", lookup.event_counts[name])
    for signal in IALS_SIGNALS:
        digest.update(signal.encode("ascii"))
        digest.update(bytes.fromhex(ials[signal].fingerprint_sha256))
    return HistoryState(
        users=users,
        items=items,
        ials=ials,
        history_rows=len(arrays["user_id"]),
        fingerprint_sha256=digest.hexdigest(),
    )


def _sorted_unique_metadata(
    keys: np.ndarray, values: Sequence[np.ndarray], label: str
) -> tuple[np.ndarray, ...]:
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if len(sorted_keys) == 0 or np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise VKLSVDMaterializationError(f"{label} keys must be non-empty and unique")
    return (sorted_keys, *(value[order] for value in values))


def load_metadata(paths: ExplicitInputPaths) -> MetadataLookup:
    users = pq.read_table(
        paths.users_metadata,
        columns=["user_id", "age", "gender", "geo"],
        use_threads=False,
    )
    items = pq.read_table(
        paths.items_metadata,
        columns=["item_id", "duration"],
        use_threads=False,
    )
    if any(column.null_count for column in [*users.columns, *items.columns]):
        raise VKLSVDMaterializationError("required metadata contains nulls")
    user_values = _sorted_unique_metadata(
        users["user_id"].to_numpy(),
        [
            users["age"].to_numpy(),
            users["gender"].to_numpy(),
            users["geo"].to_numpy(),
        ],
        "user metadata",
    )
    item_values = _sorted_unique_metadata(
        items["item_id"].to_numpy(),
        [items["duration"].to_numpy()],
        "item metadata",
    )
    if np.any(item_values[1] <= 0):
        raise VKLSVDMaterializationError("item duration must be positive")
    digest = hashlib.sha256()
    for label, value in zip(
        ("user_id", "age", "gender", "geo", "item_id", "duration"),
        (*user_values, *item_values),
        strict=True,
    ):
        _sha256_array(digest, label, value)
    return MetadataLookup(
        user_keys=user_values[0],
        user_age=user_values[1],
        user_gender=user_values[2],
        user_geo=user_values[3],
        item_keys=item_values[0],
        item_duration=item_values[1],
        fingerprint_sha256=digest.hexdigest(),
    )


def _lookup_positions(
    keys: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(keys, query)
    in_range = positions < len(keys)
    clipped = np.minimum(positions, len(keys) - 1)
    valid = in_range & (keys[clipped] == query)
    return clipped, valid


def _lookup_numeric(
    keys: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    *,
    fallback: int | float,
) -> tuple[np.ndarray, np.ndarray]:
    positions, valid = _lookup_positions(keys, query)
    output = np.full(len(query), fallback, dtype=values.dtype)
    output[valid] = values[positions[valid]]
    return output, valid


def _group_batch(
    lookup: GroupLookup, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    shows, present = _lookup_numeric(lookup.keys, lookup.show_count, query, fallback=0)
    timespent, _ = _lookup_numeric(
        lookup.keys, lookup.timespent_sum, query, fallback=0.0
    )
    events = {
        name: _lookup_numeric(lookup.keys, value, query, fallback=0)[0]
        for name, value in lookup.event_counts.items()
    }
    return shows, timespent, events, present


def _ials_batch(
    lookup: IALSLookup, user_ids: np.ndarray, item_ids: np.ndarray
) -> dict[str, np.ndarray]:
    user_positions, user_seen = _lookup_positions(lookup.user_keys, user_ids)
    item_positions, item_seen = _lookup_positions(lookup.item_keys, item_ids)
    both = user_seen & item_seen
    dot = np.zeros(len(user_ids), dtype=np.float32)
    if np.any(both):
        dot[both] = np.einsum(
            "ij,ij->i",
            lookup.user_factors[user_positions[both]],
            lookup.item_factors[item_positions[both]],
            optimize=False,
        )
    user_norm = np.zeros(len(user_ids), dtype=np.float32)
    item_norm = np.zeros(len(item_ids), dtype=np.float32)
    user_norm[user_seen] = lookup.user_norms[user_positions[user_seen]]
    item_norm[item_seen] = lookup.item_norms[item_positions[item_seen]]
    return {
        "dot": dot,
        "user_norm": user_norm,
        "item_norm": item_norm,
        "user_seen": user_seen,
        "item_seen": item_seen,
    }


def _raw_stage_schema() -> pa.Schema:
    fields = [
        pa.field("user_id", pa.uint32(), nullable=False),
        pa.field("item_id", pa.uint32(), nullable=False),
        pa.field("week", pa.uint8(), nullable=False),
        pa.field("source_row_index0", pa.uint64(), nullable=False),
        pa.field("label_timespent_seconds_raw", pa.uint8(), nullable=False),
    ]
    fields.extend(
        pa.field(f"raw_c__{name}", pa.int16(), nullable=False)
        for name in ORDERED_C_FIELDS
    )
    fields.extend(
        pa.field(name, pa.float64(), nullable=False) for name in ORDERED_P0_FIELDS
    )
    fields.extend(pa.field(name, pa.bool_(), nullable=False) for name in BASE_V_FIELDS)
    fields.extend(
        pa.field(name, pa.float64(), nullable=False) for name in ORDERED_E_FIELDS
    )
    fields.extend(
        pa.field(f"raw_dvalid__{name}", pa.bool_(), nullable=False)
        for name in DENOMINATOR_ELIGIBLE_FIELDS
    )
    return pa.schema(fields)


def _raw_feature_arrays(
    batch: pa.RecordBatch,
    *,
    week: int,
    row_offset: int,
    history: HistoryState,
    metadata: MetadataLookup,
) -> dict[str, np.ndarray]:
    user_ids = _batch_numpy(batch, "user_id").astype(np.uint32, copy=False)
    item_ids = _batch_numpy(batch, "item_id").astype(np.uint32, copy=False)
    nrows = len(user_ids)
    if len(item_ids) != nrows:
        raise VKLSVDMaterializationError("target user/item columns are misaligned")
    (
        user_shows,
        user_timespent,
        user_events,
        user_history_present,
    ) = _group_batch(history.users, user_ids)
    (
        item_shows,
        item_timespent,
        item_events,
        item_history_present,
    ) = _group_batch(history.items, item_ids)

    user_positions, user_metadata_present = _lookup_positions(
        metadata.user_keys, user_ids
    )
    item_positions, item_metadata_present = _lookup_positions(
        metadata.item_keys, item_ids
    )
    age = np.full(nrows, -1, dtype=np.int16)
    gender = np.full(nrows, -1, dtype=np.int16)
    geo = np.full(nrows, -1, dtype=np.int16)
    age[user_metadata_present] = metadata.user_age[
        user_positions[user_metadata_present]
    ]
    gender[user_metadata_present] = metadata.user_gender[
        user_positions[user_metadata_present]
    ]
    geo[user_metadata_present] = metadata.user_geo[
        user_positions[user_metadata_present]
    ]
    duration = np.zeros(nrows, dtype=np.float64)
    duration[item_metadata_present] = metadata.item_duration[
        item_positions[item_metadata_present]
    ]

    p0: dict[str, np.ndarray] = {"p_duration_seconds": duration}
    for group, shows, timespent_sum, events, rates in (
        ("user", user_shows, user_timespent, user_events, USER_RATE_SOURCES),
        ("item", item_shows, item_timespent, item_events, ITEM_RATE_SOURCES),
    ):
        p0[f"p_{group}_show_count"] = shows.astype(np.float64)
        p0[f"p_{group}_show_count_plus_beta"] = shows.astype(np.float64) + BETA_SMOOTH
        p0[f"p_{group}_timespent_sum_seconds"] = timespent_sum.astype(np.float64)
        for name, source in rates:
            p0[f"p_{group}_{name}_count_plus_alpha"] = (
                events[source].astype(np.float64) + ALPHA_SMOOTH
            )

    ials_values: dict[str, dict[str, np.ndarray]] = {}
    for signal in IALS_SIGNALS:
        values = _ials_batch(history.ials[signal], user_ids, item_ids)
        ials_values[signal] = values
        p0[f"p_ials_{signal}_dot"] = values["dot"].astype(np.float64)
        p0[f"p_ials_{signal}_user_norm"] = values["user_norm"].astype(np.float64)
        p0[f"p_ials_{signal}_item_norm"] = values["item_norm"].astype(np.float64)
    if tuple(p0) != ORDERED_P0_FIELDS:
        raise AssertionError("materialized P0 order drifted")

    base_v: dict[str, np.ndarray] = {
        "v_user_history_present": user_history_present,
        "v_item_history_present": item_history_present,
        "v_item_metadata_present": item_metadata_present,
        "v_duration_positive": item_metadata_present & (duration > 0),
    }
    for signal in IALS_SIGNALS:
        values = ials_values[signal]
        base_v[f"v_ials_{signal}_user_seen"] = values["user_seen"]
        base_v[f"v_ials_{signal}_item_seen"] = values["item_seen"]
        base_v[f"v_ials_{signal}_user_norm_positive"] = values["user_seen"] & (
            values["user_norm"] > 0
        )
        base_v[f"v_ials_{signal}_item_norm_positive"] = values["item_seen"] & (
            values["item_norm"] > 0
        )
    if tuple(base_v) != BASE_V_FIELDS:
        raise AssertionError("materialized base V order drifted")

    engineered: dict[str, np.ndarray] = {}
    for group, shows, timespent_sum, rates in (
        ("user", user_shows, user_timespent, USER_RATE_SOURCES),
        ("item", item_shows, item_timespent, ITEM_RATE_SOURCES),
    ):
        average = np.zeros(nrows, dtype=np.float64)
        np.divide(timespent_sum, shows, out=average, where=shows > 0)
        engineered[f"e_{group}_avg_timespent_seconds"] = average
        for name, _ in rates:
            engineered[f"e_{group}_{name}_rate"] = (
                p0[f"p_{group}_{name}_count_plus_alpha"]
                / p0[f"p_{group}_show_count_plus_beta"]
            )
    for signal in IALS_SIGNALS:
        values = ials_values[signal]
        denominator = values["user_norm"] * values["item_norm"]
        valid = (
            values["user_seen"]
            & values["item_seen"]
            & (values["user_norm"] > 0)
            & (values["item_norm"] > 0)
        )
        cosine = np.zeros(nrows, dtype=np.float64)
        np.divide(values["dot"], denominator, out=cosine, where=valid)
        engineered[f"e_ials_{signal}_cosine"] = cosine
    if tuple(engineered) != ORDERED_E_FIELDS:
        raise AssertionError("materialized E order drifted")

    dvalid: dict[str, np.ndarray] = {
        "p_duration_seconds": item_metadata_present & (duration > 0),
        "p_user_show_count": user_history_present,
        "p_user_show_count_plus_beta": np.ones(nrows, dtype=np.bool_),
        "p_item_show_count": item_history_present,
        "p_item_show_count_plus_beta": np.ones(nrows, dtype=np.bool_),
    }
    for signal in IALS_SIGNALS:
        values = ials_values[signal]
        dvalid[f"p_ials_{signal}_user_norm"] = values["user_seen"] & (
            values["user_norm"] > 0
        )
        dvalid[f"p_ials_{signal}_item_norm"] = values["item_seen"] & (
            values["item_norm"] > 0
        )
    if tuple(dvalid) != DENOMINATOR_ELIGIBLE_FIELDS:
        raise AssertionError("materialized denominator validity order drifted")

    result: dict[str, np.ndarray] = {
        "user_id": user_ids,
        "item_id": item_ids,
        "week": np.full(nrows, week, dtype=np.uint8),
        "source_row_index0": np.arange(row_offset, row_offset + nrows, dtype=np.uint64),
        "label_timespent_seconds_raw": _batch_numpy(batch, "timespent").astype(
            np.uint8, copy=False
        ),
        "raw_c__age": age,
        "raw_c__gender": gender,
        "raw_c__geo": geo,
        "raw_c__place": _batch_numpy(batch, "place").astype(np.int16),
        "raw_c__platform": _batch_numpy(batch, "platform").astype(np.int16),
        "raw_c__agent": _batch_numpy(batch, "agent").astype(np.int16),
        **p0,
        **base_v,
        **engineered,
        **{f"raw_dvalid__{name}": value for name, value in dvalid.items()},
    }
    if any(
        np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()
        for value in result.values()
    ):
        raise VKLSVDMaterializationError(
            "raw materialization produced non-finite values"
        )
    return result


def _arrays_to_table(arrays: Mapping[str, np.ndarray], schema: pa.Schema) -> pa.Table:
    if tuple(arrays) != tuple(schema.names):
        raise VKLSVDMaterializationError("Arrow array order differs from schema")
    return pa.Table.from_arrays(
        [pa.array(arrays[field.name], type=field.type) for field in schema],
        schema=schema,
    )


def _stage_week(
    *,
    week: int,
    role: str,
    source_path: Path,
    stage_path: Path,
    history: HistoryState,
    metadata: MetadataLookup,
) -> StageResult:
    schema = _raw_stage_schema()
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    categorical: dict[str, set[int]] = {name: set() for name in ORDERED_C_FIELDS}
    rows = 0
    with pq.ParquetWriter(
        stage_path,
        schema,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
    ) as writer:
        for batch in _read_batches(source_path, TARGET_VALUE_COLUMNS):
            arrays = _raw_feature_arrays(
                batch,
                week=week,
                row_offset=rows,
                history=history,
                metadata=metadata,
            )
            table = _arrays_to_table(arrays, schema)
            writer.write_table(table, row_group_size=PARQUET_ROW_GROUP_ROWS)
            for name in ORDERED_C_FIELDS:
                values = arrays[f"raw_c__{name}"]
                categorical[name].update(
                    int(value) for value in np.unique(values) if value >= 0
                )
            rows += len(table)
    if rows <= 0:
        raise VKLSVDMaterializationError(f"week {week} produced no staged rows")
    return StageResult(
        week=week,
        role=role,
        path=stage_path,
        rows=rows,
        categorical_values={
            name: tuple(sorted(values)) for name, values in categorical.items()
        },
    )


def _merge_train_categories(
    stages: Sequence[StageResult],
) -> dict[str, tuple[int, ...]]:
    merged: dict[str, set[int]] = {name: set() for name in ORDERED_C_FIELDS}
    for stage in stages:
        if stage.role != "train":
            continue
        for name in ORDERED_C_FIELDS:
            merged[name].update(stage.categorical_values[name])
    if any(not values for values in merged.values()):
        raise VKLSVDMaterializationError("a train-only categorical mapping is empty")
    return {name: tuple(sorted(values)) for name, values in merged.items()}


def fit_train_scales(
    stage_paths: Sequence[Path],
) -> tuple[dict[str, float], dict[str, float]]:
    if not stage_paths:
        raise VKLSVDMaterializationError("scale fitting requires train stages")
    fields = ORDERED_P0_FIELDS + ORDERED_E_FIELDS
    lazy = pl.scan_parquet([str(path) for path in stage_paths])
    row = lazy.select(
        *(
            pl.col(name).abs().quantile(0.99, interpolation="nearest").alias(name)
            for name in fields
        )
    ).collect(engine="streaming")
    values = row.to_dicts()[0]
    scales: dict[str, float] = {}
    for name in fields:
        value = float(values[name])
        if not math.isfinite(value) or value < 0:
            raise VKLSVDMaterializationError(f"invalid train q99 scale for {name}")
        scales[name] = max(value, 1e-12)
    return (
        {name: scales[name] for name in ORDERED_P0_FIELDS},
        {name: scales[name] for name in ORDERED_E_FIELDS},
    )


def _tokenize(values: np.ndarray, known_values: Sequence[int]) -> np.ndarray:
    known = np.asarray(known_values, dtype=np.int16)
    positions = np.searchsorted(known, values)
    in_range = positions < len(known)
    clipped = np.minimum(positions, len(known) - 1)
    matched = in_range & (known[clipped] == values)
    tokens = np.zeros(len(values), dtype=np.uint32)
    tokens[values < 0] = 1
    known_mask = (values >= 0) & matched
    tokens[known_mask] = positions[known_mask].astype(np.uint32) + 2
    return tokens


def _final_schema(*, include_engineered: bool) -> pa.Schema:
    fields = [
        pa.field("user_id", pa.uint32(), nullable=False),
        pa.field("item_id", pa.uint32(), nullable=False),
        pa.field("week", pa.uint8(), nullable=False),
        pa.field("source_row_index0", pa.uint64(), nullable=False),
        pa.field("stable_row_id", pa.string(), nullable=False),
        pa.field("label_normalized_watch_time", pa.float32(), nullable=False),
        pa.field("label_timespent_seconds", pa.float32(), nullable=False),
    ]
    fields.extend(
        pa.field(name, pa.uint32(), nullable=False) for name in ORDERED_C_FIELDS
    )
    fields.extend(
        pa.field(name, pa.float32(), nullable=False)
        for name in ORDERED_NUMERIC_OPERANDS
    )
    if include_engineered:
        fields.extend(
            pa.field(name, pa.float32(), nullable=False) for name in ORDERED_E_FIELDS
        )
    fields.extend(
        pa.field(name, pa.bool_(), nullable=False)
        for name in DENOMINATOR_VALIDITY_COLUMNS
    )
    regime = "C+P+E" if include_engineered else "C+P"
    metadata = {
        b"dcdn.dataset": b"vk_lsvd_up0.01_ir0.01",
        b"dcdn.feature_contract_sha256": FEATURE_CONTRACT_SHA256.encode("ascii"),
        b"dcdn.regime": regime.encode("ascii"),
        b"dcdn.test_unlocked": b"false",
    }
    return pa.schema(fields, metadata=metadata)


def _physical_schema_sha256(schema: pa.Schema) -> str:
    payload = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _empty_denominator_audit() -> dict[str, dict[str, Any]]:
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


def _update_denominator_audit(
    audit: dict[str, dict[str, Any]],
    normalized_p0: Mapping[str, np.ndarray],
    dvalid: Mapping[str, np.ndarray],
) -> None:
    for name in DENOMINATOR_ELIGIBLE_FIELDS:
        values = normalized_p0[name]
        valid = dvalid[name]
        nonfinite = ~np.isfinite(values)
        if np.any(nonfinite):
            raise VKLSVDMaterializationError(
                f"non-finite denominator candidate: {name}"
            )
        if np.any(values < 0):
            raise VKLSVDMaterializationError(f"negative denominator candidate: {name}")
        if np.any(valid & (values <= 0)):
            raise VKLSVDMaterializationError(
                f"active denominator is non-positive: {name}"
            )
        record = audit[name]
        record["rows"] += len(values)
        record["valid_rows"] += int(valid.sum())
        record["invalid_or_zero_rows"] += int((~valid).sum())
        record["reciprocal_floor_rows"] += int(
            (valid & (values < RECIPROCAL_EPSILON)).sum()
        )
        record["nonfinite_rows"] += int(nonfinite.sum())
        if np.any(valid):
            minimum = float(values[valid].min())
            maximum = float(values[valid].max())
            current_min = record["minimum_positive_normalized"]
            current_max = record["maximum_normalized"]
            record["minimum_positive_normalized"] = (
                minimum if current_min is None else min(current_min, minimum)
            )
            record["maximum_normalized"] = (
                maximum if current_max is None else max(current_max, maximum)
            )


def _canonical_shared_update(
    digest: Any,
    c_values: Mapping[str, np.ndarray],
    p_values: Mapping[str, np.ndarray],
) -> None:
    for name in ORDERED_C_FIELDS:
        _sha256_array(digest, name, c_values[name])
    for name in ORDERED_NUMERIC_OPERANDS:
        _sha256_array(digest, name, p_values[name])


def _final_arrays(
    batch: pa.RecordBatch,
    *,
    categories: Mapping[str, Sequence[int]],
    p0_scales: Mapping[str, float],
    e_scales: Mapping[str, float],
) -> tuple[
    dict[str, np.ndarray | list[str]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    nrows = batch.num_rows
    week_values = _batch_numpy(batch, "week").astype(np.uint8, copy=False)
    if nrows == 0 or not np.all(week_values == week_values[0]):
        raise VKLSVDMaterializationError(
            "a staged shard is empty or contains multiple weeks"
        )
    week = int(week_values[0])
    row_indices = _batch_numpy(batch, "source_row_index0").astype(np.uint64, copy=False)
    expected = np.arange(
        int(row_indices[0]), int(row_indices[0]) + nrows, dtype=np.uint64
    )
    if not np.array_equal(row_indices, expected):
        raise VKLSVDMaterializationError("staged source row order is not contiguous")

    c_values = {
        name: _tokenize(_batch_numpy(batch, f"raw_c__{name}"), categories[name])
        for name in ORDERED_C_FIELDS
    }
    normalized_p0 = {
        name: (
            _batch_numpy(batch, name).astype(np.float64, copy=False) / p0_scales[name]
        ).astype(np.float32)
        for name in ORDERED_P0_FIELDS
    }
    normalized_e = {
        name: (
            _batch_numpy(batch, name).astype(np.float64, copy=False) / e_scales[name]
        ).astype(np.float32)
        for name in ORDERED_E_FIELDS
    }
    if any(
        not np.isfinite(value).all()
        for value in (*normalized_p0.values(), *normalized_e.values())
    ):
        raise VKLSVDMaterializationError("normalized features contain NaN or Inf")
    dvalid_eligible = {
        name: _batch_numpy(batch, f"raw_dvalid__{name}").astype(np.bool_, copy=False)
        for name in DENOMINATOR_ELIGIBLE_FIELDS
    }
    v_values: dict[str, np.ndarray] = {
        name: _batch_numpy(batch, name).astype(np.float32) for name in BASE_V_FIELDS
    }
    for name in DENOMINATOR_ELIGIBLE_FIELDS:
        v_values[floor_indicator_name(name)] = (
            dvalid_eligible[name] & (normalized_p0[name] < RECIPROCAL_EPSILON)
        ).astype(np.float32)
    if tuple(v_values) != ORDERED_V_FIELDS:
        raise AssertionError("final V order drifted")
    p_values = {
        **normalized_p0,
        **{name: v_values[name] for name in ORDERED_V_FIELDS},
    }
    if tuple(p_values) != ORDERED_NUMERIC_OPERANDS:
        raise AssertionError("final P order drifted")
    dvalid_full = {
        name: dvalid_eligible.get(name, np.ones(nrows, dtype=np.bool_))
        for name in ORDERED_NUMERIC_OPERANDS
    }
    raw_label = _batch_numpy(batch, "label_timespent_seconds_raw").astype(np.float32)
    target_seconds = np.clip(raw_label, 0.0, LABEL_DURATION_Q99_SECONDS).astype(
        np.float32
    )
    normalized_label = (target_seconds / LABEL_DURATION_Q99_SECONDS).astype(np.float32)
    stable_ids = [
        stable_row_id(week=week, source_row_index0=int(index)) for index in row_indices
    ]
    common: dict[str, np.ndarray | list[str]] = {
        "user_id": _batch_numpy(batch, "user_id").astype(np.uint32, copy=False),
        "item_id": _batch_numpy(batch, "item_id").astype(np.uint32, copy=False),
        "week": week_values,
        "source_row_index0": row_indices,
        "stable_row_id": stable_ids,
        "label_normalized_watch_time": normalized_label,
        "label_timespent_seconds": target_seconds,
        **c_values,
        **p_values,
        **{
            f"denominator_validity__{name}": value
            for name, value in dvalid_full.items()
        },
    }
    return common, c_values, p_values, normalized_e


def _table_from_final_arrays(
    arrays: Mapping[str, np.ndarray | list[str]], schema: pa.Schema
) -> pa.Table:
    if set(arrays) != set(schema.names):
        missing = set(schema.names) - set(arrays)
        extra = set(arrays) - set(schema.names)
        raise VKLSVDMaterializationError(
            "final array/schema mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return pa.Table.from_arrays(
        [pa.array(arrays[field.name], type=field.type) for field in schema],
        schema=schema,
    )


def _verify_reopened_parquet(path: Path, schema: pa.Schema, expected_rows: int) -> int:
    _require_regular_file(path, "materialized shard")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise VKLSVDMaterializationError("reopened shard row count changed")
    if not parquet.schema_arrow.equals(schema, check_metadata=True):
        raise VKLSVDMaterializationError("reopened shard schema changed")
    return parquet.metadata.num_row_groups


def finalize_stage(
    *,
    stage: StageResult,
    c_plus_p_path: Path,
    c_plus_p_plus_e_path: Path,
    categories: Mapping[str, Sequence[int]],
    p0_scales: Mapping[str, float],
    e_scales: Mapping[str, float],
    denominator_audit: dict[str, dict[str, Any]],
) -> FinalShard:
    cp_schema = _final_schema(include_engineered=False)
    cpe_schema = _final_schema(include_engineered=True)
    c_plus_p_path.parent.mkdir(parents=True, exist_ok=True)
    c_plus_p_plus_e_path.parent.mkdir(parents=True, exist_ok=True)
    shared_digest = hashlib.sha256()
    row_id_digest = hashlib.sha256()
    rows = 0
    with (
        pq.ParquetWriter(
            c_plus_p_path,
            cp_schema,
            compression="zstd",
            use_dictionary=list(ORDERED_C_FIELDS),
            write_statistics=True,
        ) as cp_writer,
        pq.ParquetWriter(
            c_plus_p_plus_e_path,
            cpe_schema,
            compression="zstd",
            use_dictionary=list(ORDERED_C_FIELDS),
            write_statistics=True,
        ) as cpe_writer,
    ):
        for batch in _read_batches(stage.path, _raw_stage_schema().names):
            common, c_values, p_values, e_values = _final_arrays(
                batch,
                categories=categories,
                p0_scales=p0_scales,
                e_scales=e_scales,
            )
            row_indices = common["source_row_index0"]
            if not isinstance(row_indices, np.ndarray) or int(row_indices[0]) != rows:
                raise VKLSVDMaterializationError(
                    "staged source row order is not contiguous across batches"
                )
            dvalid = {
                name: _batch_numpy(batch, f"raw_dvalid__{name}").astype(
                    np.bool_, copy=False
                )
                for name in DENOMINATOR_ELIGIBLE_FIELDS
            }
            normalized_p0 = {name: p_values[name] for name in ORDERED_P0_FIELDS}
            _update_denominator_audit(denominator_audit, normalized_p0, dvalid)
            _canonical_shared_update(shared_digest, c_values, p_values)
            for value in common["stable_row_id"]:
                row_id_digest.update(str(value).encode("ascii"))
                row_id_digest.update(b"\n")
            cp_table = _table_from_final_arrays(common, cp_schema)
            insertion = len(cp_schema.names) - len(DENOMINATOR_VALIDITY_COLUMNS)
            ordered_cpe: dict[str, np.ndarray | list[str]] = {}
            for name in cp_schema.names[:insertion]:
                ordered_cpe[name] = common[name]
            ordered_cpe.update(e_values)
            for name in DENOMINATOR_VALIDITY_COLUMNS:
                ordered_cpe[name] = common[name]
            cpe_table = _table_from_final_arrays(ordered_cpe, cpe_schema)
            cp_writer.write_table(cp_table, row_group_size=PARQUET_ROW_GROUP_ROWS)
            cpe_writer.write_table(cpe_table, row_group_size=PARQUET_ROW_GROUP_ROWS)
            rows += len(cp_table)
    if rows != stage.rows:
        raise VKLSVDMaterializationError("final shard row count differs from stage")
    cp_row_groups = _verify_reopened_parquet(c_plus_p_path, cp_schema, rows)
    cpe_row_groups = _verify_reopened_parquet(c_plus_p_plus_e_path, cpe_schema, rows)
    return FinalShard(
        week=stage.week,
        role=stage.role,
        rows=rows,
        c_plus_p_path=c_plus_p_path,
        c_plus_p_bytes=c_plus_p_path.stat().st_size,
        c_plus_p_sha256=_sha256_file(c_plus_p_path),
        c_plus_p_plus_e_path=c_plus_p_plus_e_path,
        c_plus_p_plus_e_bytes=c_plus_p_plus_e_path.stat().st_size,
        c_plus_p_plus_e_sha256=_sha256_file(c_plus_p_plus_e_path),
        shared_values_sha256=shared_digest.hexdigest(),
        stable_row_ids_sha256=row_id_digest.hexdigest(),
        c_plus_p_row_groups=cp_row_groups,
        c_plus_p_plus_e_row_groups=cpe_row_groups,
    )


def _category_manifest(
    categories: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    return {
        name: {
            "OOV_token": 0,
            "MISSING_token": 1,
            "known_raw_values_ascending": list(values),
            "known_token_start": 2,
            "cardinality_with_reserved_tokens": len(values) + 2,
        }
        for name, values in categories.items()
    }


def _shard_manifest(record: FinalShard, final_root: Path) -> dict[str, Any]:
    return {
        "week": record.week,
        "role": record.role,
        "rows": record.rows,
        "C_plus_P": {
            "path": record.c_plus_p_path.relative_to(final_root).as_posix(),
            "bytes": record.c_plus_p_bytes,
            "sha256": record.c_plus_p_sha256,
            "row_groups": record.c_plus_p_row_groups,
        },
        "C_plus_P_plus_E": {
            "path": record.c_plus_p_plus_e_path.relative_to(final_root).as_posix(),
            "bytes": record.c_plus_p_plus_e_bytes,
            "sha256": record.c_plus_p_plus_e_sha256,
            "row_groups": record.c_plus_p_plus_e_row_groups,
        },
        "shared_C_plus_P_values_sha256": record.shared_values_sha256,
        "stable_row_ids_sha256": record.stable_row_ids_sha256,
    }


def _validate_manifest_payload(payload: Mapping[str, Any]) -> None:
    expected_flags = {
        "test_unlocked": False,
        "test_labels_opened": False,
        "test_view_materialized": False,
        "week26_file_open_attempted": False,
        "validation_model_metrics_computed": False,
        "test_metrics_computed": False,
    }
    if payload.get("schema_version") != 1:
        raise VKLSVDMaterializationError("views manifest schema_version must equal 1")
    if payload.get("dataset_id") != "vk_lsvd_up0.01_ir0.01":
        raise VKLSVDMaterializationError("views manifest dataset changed")
    if payload.get("artifact_role") != "train_validation_feature_views":
        raise VKLSVDMaterializationError("views manifest artifact role changed")
    for key, expected in expected_flags.items():
        if type(payload.get(key)) is not bool or payload[key] is not expected:
            raise VKLSVDMaterializationError(f"views manifest {key} must be false")
    if payload.get("feature_contract_sha256") != FEATURE_CONTRACT_SHA256:
        raise VKLSVDMaterializationError("views manifest feature contract changed")
    if payload.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise VKLSVDMaterializationError("views manifest source changed")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise VKLSVDMaterializationError("views manifest shards are missing")
    if any(
        record.get("week") == 26 for record in shards if isinstance(record, Mapping)
    ):
        raise VKLSVDMaterializationError("views manifest must never contain week 26")


def _dump_manifest(payload: Mapping[str, Any]) -> bytes:
    _validate_manifest_payload(payload)
    data = yaml.safe_dump(
        dict(payload),
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    ).encode("utf-8")
    if len(data) > MAX_MANIFEST_BYTES:
        raise VKLSVDMaterializationError("views manifest exceeds size limit")
    return data


def _write_fsynced_temp(parent: Path, prefix: str, data: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(dir=parent, prefix=prefix)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_views_manifest(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    sidecar = parent / f"{path.name}.sha256"
    data = _dump_manifest(payload)
    digest = hashlib.sha256(data).hexdigest()
    sidecar_data = f"{digest}  {path.name}\n".encode("ascii")
    manifest_temp = _write_fsynced_temp(parent, f".{path.name}.manifest-", data)
    sidecar_temp = _write_fsynced_temp(parent, f".{path.name}.sidecar-", sidecar_data)
    try:
        os.replace(manifest_temp, target)
        os.replace(sidecar_temp, sidecar)
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        manifest_temp.unlink(missing_ok=True)
        sidecar_temp.unlink(missing_ok=True)
    return digest


def load_views_manifest(path: Path) -> dict[str, Any]:
    _require_regular_file(path, "views manifest")
    sidecar_path = path.with_name(f"{path.name}.sha256")
    _require_regular_file(sidecar_path, "views manifest sidecar")
    data = path.read_bytes()
    sidecar = sidecar_path.read_bytes()
    if len(data) > MAX_MANIFEST_BYTES or len(sidecar) > MAX_SIDECAR_BYTES:
        raise VKLSVDMaterializationError("views manifest or sidecar exceeds size limit")
    try:
        text = sidecar.decode("ascii")
    except UnicodeDecodeError as error:
        raise VKLSVDMaterializationError("views sidecar must be ASCII") from error
    if not text.endswith("\n") or text.count("\n") != 1:
        raise VKLSVDMaterializationError("views sidecar must contain one line")
    parts = text.rstrip("\n").split("  ")
    if (
        len(parts) != 2
        or parts[1] != path.name
        or SHA256_RE.fullmatch(parts[0]) is None
    ):
        raise VKLSVDMaterializationError("views sidecar is malformed")
    if hashlib.sha256(data).hexdigest() != parts[0]:
        raise VKLSVDMaterializationError("views manifest sidecar mismatch")
    try:
        payload = yaml.load(data.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise VKLSVDMaterializationError("invalid views manifest YAML") from error
    if not isinstance(payload, Mapping):
        raise VKLSVDMaterializationError("views manifest root must be a mapping")
    result = dict(payload)
    _validate_manifest_payload(result)
    return result


def materialize_explicit_paths(
    *,
    paths: ExplicitInputPaths,
    output_root: Path,
    fit_config: IALSFitConfig,
    guard_callback: Any | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Shared deterministic implementation for synthetic and frozen runs."""

    if output_root.exists():
        raise VKLSVDMaterializationError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True, mode=0o700)
    stage_root = output_root / "_raw_stage"
    stage_root.mkdir(mode=0o700)
    if progress_callback is not None:
        progress_callback("history_fit_started", {})
    history = build_history_state(paths.history, config=fit_config)
    if progress_callback is not None:
        progress_callback(
            "history_fit_complete", {"history_rows": history.history_rows}
        )
    metadata = load_metadata(paths)
    if progress_callback is not None:
        progress_callback(
            "metadata_loaded",
            {
                "users": len(metadata.user_keys),
                "items": len(metadata.item_keys),
            },
        )
    if guard_callback is not None:
        guard_callback()

    stages: list[StageResult] = []
    for week, path in paths.train:
        stages.append(
            _stage_week(
                week=week,
                role="train",
                source_path=path,
                stage_path=stage_root / f"train_week_{week:02d}.parquet",
                history=history,
                metadata=metadata,
            )
        )
        if progress_callback is not None:
            progress_callback(
                "stage_complete",
                {"week": week, "role": "train", "rows": stages[-1].rows},
            )
        if guard_callback is not None:
            guard_callback()
    for week, path in paths.validation:
        stages.append(
            _stage_week(
                week=week,
                role="validation",
                source_path=path,
                stage_path=(stage_root / f"validation_week_{week:02d}.parquet"),
                history=history,
                metadata=metadata,
            )
        )
        if progress_callback is not None:
            progress_callback(
                "stage_complete",
                {
                    "week": week,
                    "role": "validation",
                    "rows": stages[-1].rows,
                },
            )
        if guard_callback is not None:
            guard_callback()

    categories = _merge_train_categories(stages)
    train_stages = [stage.path for stage in stages if stage.role == "train"]
    p0_scales, e_scales = fit_train_scales(train_stages)
    if progress_callback is not None:
        progress_callback("train_only_fit_complete", {})
    denominator_audit = _empty_denominator_audit()
    final_shards: list[FinalShard] = []
    for stage in stages:
        final_shards.append(
            finalize_stage(
                stage=stage,
                c_plus_p_path=(
                    output_root
                    / "C_plus_P"
                    / stage.role
                    / f"week_{stage.week:02d}.parquet"
                ),
                c_plus_p_plus_e_path=(
                    output_root
                    / "C_plus_P_plus_E"
                    / stage.role
                    / f"week_{stage.week:02d}.parquet"
                ),
                categories=categories,
                p0_scales=p0_scales,
                e_scales=e_scales,
                denominator_audit=denominator_audit,
            )
        )
        if progress_callback is not None:
            progress_callback(
                "final_shard_complete",
                {
                    "week": stage.week,
                    "role": stage.role,
                    "rows": final_shards[-1].rows,
                },
            )
        if guard_callback is not None:
            guard_callback()

    shutil.rmtree(stage_root)
    total_rows = sum(record.rows for record in final_shards)
    train_rows = sum(record.rows for record in final_shards if record.role == "train")
    validation_rows = total_rows - train_rows
    return {
        "history": history,
        "metadata": metadata,
        "categories": categories,
        "p0_scales": p0_scales,
        "e_scales": e_scales,
        "denominator_audit": denominator_audit,
        "shards": final_shards,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "total_rows": total_rows,
        "C_plus_P_physical_schema_sha256": _physical_schema_sha256(
            _final_schema(include_engineered=False)
        ),
        "C_plus_P_plus_E_physical_schema_sha256": _physical_schema_sha256(
            _final_schema(include_engineered=True)
        ),
    }


def _source_code_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _build_manifest(
    *,
    result: Mapping[str, Any],
    final_root: Path,
    verified_sources: Sequence[Mapping[str, Any]],
    resource_guard: Mapping[str, Any],
    fit_config: IALSFitConfig,
) -> dict[str, Any]:
    history: HistoryState = result["history"]
    metadata: MetadataLookup = result["metadata"]
    shards: Sequence[FinalShard] = result["shards"]
    return {
        "schema_version": 1,
        "dataset_id": "vk_lsvd_up0.01_ir0.01",
        "artifact_role": "train_validation_feature_views",
        "materialization_state": "complete_and_hash_verified",
        "materialized_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm_source_sha256": _source_code_sha256(),
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "model_sources_sha256": MODEL_SOURCES_SHA256,
        "search_space_sha256": SEARCH_SPACE_SHA256,
        "experiment_matrix_sha256": EXPERIMENT_MATRIX_SHA256,
        "test_unlocked": False,
        "test_labels_opened": False,
        "test_view_materialized": False,
        "week26_file_open_attempted": False,
        "validation_model_metrics_computed": False,
        "test_metrics_computed": False,
        "split": {
            "history_weeks": list(HISTORY_WEEKS),
            "train_weeks": list(TRAIN_WEEKS),
            "validation_week": VALIDATION_WEEK,
            "sealed_test_week": 26,
            "state_policy": "fit_on_history_0_19_once_then_freeze",
            "equal_time_policy": (
                "no_post_history_updates_so_equal_time_rows_never_update_each_other"
            ),
        },
        "source_objects_verified": list(verified_sources),
        "history_state": {
            "rows": history.history_rows,
            "users": len(history.users.keys),
            "items": len(history.items.keys),
            "fingerprint_sha256": history.fingerprint_sha256,
            "metadata_fingerprint_sha256": metadata.fingerprint_sha256,
            "ials": {
                signal: {
                    "factors": fit_config.factors,
                    "iterations": fit_config.iterations,
                    "regularization": fit_config.regularization,
                    "random_state": fit_config.random_state,
                    "threads": fit_config.threads,
                    "csr_duplicate_rule": "sum_duplicates",
                    "user_id_order": "ascending_integer",
                    "item_id_order": "ascending_integer",
                    "fingerprint_sha256": history.ials[signal].fingerprint_sha256,
                }
                for signal in IALS_SIGNALS
            },
        },
        "train_only_fit": {
            "categorical_mappings": _category_manifest(result["categories"]),
            "normalization": {
                "method": "divide_by_train_abs_q99",
                "quantile": 0.99,
                "interpolation": "nearest",
                "minimum_scale": 1e-12,
                "centering": "none",
                "clipping": "none",
                "P0_scales": dict(result["p0_scales"]),
                "E_scales": dict(result["e_scales"]),
            },
        },
        "feature_algebra": {
            "ordered_C": list(ORDERED_C_FIELDS),
            "ordered_P0": list(ORDERED_P0_FIELDS),
            "ordered_V": list(ORDERED_V_FIELDS),
            "ordered_P": list(ORDERED_NUMERIC_OPERANDS),
            "ordered_E": list(ORDERED_E_FIELDS),
            "dimensions": {
                "C": 6,
                "P0": 31,
                "V": 27,
                "P": 58,
                "E": 20,
            },
            "ordered_operand_schema_sha256": ordered_schema_sha256(
                ORDERED_NUMERIC_OPERANDS
            ),
            "C_plus_P_expanded_schema_sha256": ordered_schema_sha256(
                ORDERED_C_PLUS_P_FLAT_SCHEMA
            ),
            "C_plus_P_plus_E_expanded_schema_sha256": ordered_schema_sha256(
                ORDERED_C_PLUS_P_PLUS_E_FLAT_SCHEMA
            ),
            "ordered_denominator_mask": list(ORDERED_DENOMINATOR_MASK),
            "denominator_eligible_fields": list(DENOMINATOR_ELIGIBLE_FIELDS),
            "denominator_epsilon": RECIPROCAL_EPSILON,
            "C_plus_P_physical_schema_sha256": result[
                "C_plus_P_physical_schema_sha256"
            ],
            "C_plus_P_plus_E_physical_schema_sha256": result[
                "C_plus_P_plus_E_physical_schema_sha256"
            ],
        },
        "label": {
            "source": "current_row.timespent",
            "c_seconds": LABEL_DURATION_Q99_SECONDS,
            "transform": "clip(timespent,0,c)/c",
            "model_metrics_computed": False,
        },
        "row_counts": {
            "train": result["train_rows"],
            "validation": result["validation_rows"],
            "total": result["total_rows"],
        },
        "denominator_scan": result["denominator_audit"],
        "storage": {
            "root": str(final_root),
            "class": "ephemeral_regenerable_public_processed_data",
            "repository_contains_processed_shards": False,
            "shards_byte_hashed": True,
            "shard_schemas_reopened": True,
            "shard_values_reopened_after_verification": False,
        },
        "resource_guard": dict(resource_guard),
        "shards": [_shard_manifest(record, final_root) for record in shards],
    }


def materialize_vk_lsvd_views(
    *,
    repository_root: Path,
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    contract_path: Path = DEFAULT_FEATURE_CONTRACT,
    manifest_path: Path = DEFAULT_VIEWS_MANIFEST,
    ledger_path: Path = DEFAULT_COST_LEDGER,
    eta_hours: float = MAX_ALLOWED_ETA_HOURS,
    progress_callback: Any | None = None,
) -> tuple[dict[str, Any], str]:
    """Run the frozen materialization and atomically publish its manifest."""

    repository_root = repository_root.resolve(strict=True)
    contract_path = (
        contract_path
        if contract_path.is_absolute()
        else repository_root / contract_path
    )
    manifest_path = (
        manifest_path
        if manifest_path.is_absolute()
        else repository_root / manifest_path
    )
    ledger_path = (
        ledger_path if ledger_path.is_absolute() else repository_root / ledger_path
    )
    guard = resource_and_cost_guard(
        output_root=output_root,
        ledger_path=ledger_path,
        eta_hours=eta_hours,
    )
    _, paths, verified_sources = verify_frozen_inputs(
        repository_root=repository_root,
        raw_root=raw_root,
        contract_path=contract_path,
    )
    if output_root.is_symlink():
        raise VKLSVDMaterializationError("output_root must not be a symlink")
    final_root = output_root / "views_v1"
    temporary_root = output_root / f".views_v1.incomplete-{os.getpid()}"
    if final_root.exists() or temporary_root.exists():
        raise VKLSVDMaterializationError("materialization output already exists")
    output_root.mkdir(parents=True, exist_ok=True)

    def check_storage() -> None:
        _filesystem_guard(output_root)

    fit_config = IALSFitConfig()
    result = materialize_explicit_paths(
        paths=paths,
        output_root=temporary_root,
        fit_config=fit_config,
        guard_callback=check_storage,
        progress_callback=progress_callback,
    )
    os.replace(temporary_root, final_root)
    # The returned paths still point at the pre-rename root; rewrite them only
    # after the atomic directory rename and before serializing provenance.
    rewritten_shards: list[FinalShard] = []
    for record in result["shards"]:
        rewritten_shards.append(
            replace(
                record,
                c_plus_p_path=final_root
                / record.c_plus_p_path.relative_to(temporary_root),
                c_plus_p_plus_e_path=final_root
                / record.c_plus_p_plus_e_path.relative_to(temporary_root),
            )
        )
    result = {**result, "shards": rewritten_shards}
    manifest = _build_manifest(
        result=result,
        final_root=final_root,
        verified_sources=verified_sources,
        resource_guard=guard,
        fit_config=fit_config,
    )
    digest = write_views_manifest(manifest_path, manifest)
    reopened = load_views_manifest(manifest_path)
    if reopened != manifest:
        raise VKLSVDMaterializationError("reopened views manifest differs from memory")
    return manifest, digest
