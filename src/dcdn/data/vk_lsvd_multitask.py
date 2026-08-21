"""Frozen VK-LSVD multi-target view contract and row-preserving transformer.

The owner override recorded on 2026-08-20 supersedes the legacy single-target
VK view without rewriting it.  This module derives the new headline view from
two already verified inputs:

* the legacy ``C+P+E`` feature shard, whose history-frozen feature values are
  retained byte-for-byte; and
* the row-aligned released interaction shard, read only for current-row labels.

Only seventeen manually materialized divisions of historical counters/sums
belong to ``R_div``.  The three iALS cosine fields move into the shared block,
and iALS norms are never reciprocal denominators.  Week 26 is rejected here;
this premetric transformer can open only train weeks 20--24 and validation
week 25.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import pyarrow as pa

from dcdn.data.vk_lsvd import (
    IALS_SIGNALS,
    ORDERED_C_FIELDS,
    ORDERED_P0_FIELDS,
    stable_row_id,
)
from dcdn.data.vk_lsvd import (
    ORDERED_E_FIELDS as LEGACY_ORDERED_E_FIELDS,
)
from dcdn.data.vk_lsvd import (
    ORDERED_V_FIELDS as LEGACY_ORDERED_V_FIELDS,
)

DATASET_ID: Final = "vk_lsvd_up0.01_ir0.01"
TRAIN_WEEKS: Final = tuple(range(20, 25))
VALIDATION_WEEK: Final = 25
ALLOWED_WEEKS: Final = (*TRAIN_WEEKS, VALIDATION_WEEK)

TARGET_NAMES: Final = (
    "watch_time",
    "like",
    "dislike",
    "share",
    "bookmark",
    "click_on_author",
    "open_comments",
)
TARGET_KINDS: Final = ("regression", *("binary",) * 6)
TARGET_COLUMNS: Final = tuple(f"label_{name}" for name in TARGET_NAMES)
TARGET_MASK_COLUMNS: Final = tuple(f"label_mask_{name}" for name in TARGET_NAMES)

SHARED_SIMILARITY_FIELDS: Final = tuple(
    f"e_ials_{signal}_cosine" for signal in IALS_SIGNALS
)
R_DIV_FIELDS: Final = tuple(
    field for field in LEGACY_ORDERED_E_FIELDS if field not in SHARED_SIMILARITY_FIELDS
)
DENOMINATOR_ELIGIBLE_FIELDS: Final = (
    "p_duration_seconds",
    "p_user_show_count",
    "p_user_show_count_plus_beta",
    "p_item_show_count",
    "p_item_show_count_plus_beta",
)
_REMOVED_NORM_FLOOR_FIELDS: Final = tuple(
    f"v_ials_{signal}_{entity}_norm_reciprocal_floor_active"
    for signal in IALS_SIGNALS
    for entity in ("user", "item")
)
ORDERED_V_FIELDS: Final = tuple(
    field
    for field in LEGACY_ORDERED_V_FIELDS
    if field not in _REMOVED_NORM_FLOOR_FIELDS
)
ORDERED_OPERAND_FIELDS: Final = (
    *ORDERED_P0_FIELDS,
    *SHARED_SIMILARITY_FIELDS,
    *ORDERED_V_FIELDS,
)
ORDERED_DENOMINATOR_MASK: Final = tuple(
    field in DENOMINATOR_ELIGIBLE_FIELDS for field in ORDERED_OPERAND_FIELDS
)
VALIDITY_COLUMNS: Final = tuple(
    f"denominator_validity__{field}" for field in ORDERED_OPERAND_FIELDS
)
AUDIT_COLUMNS: Final = (
    "user_id",
    "item_id",
    "week",
    "source_row_index0",
    "stable_row_id",
    "label_timespent_seconds",
)

if len(R_DIV_FIELDS) != 17:
    raise AssertionError("VK R_div must contain exactly seventeen manual divisions")
if len(SHARED_SIMILARITY_FIELDS) != 3:
    raise AssertionError("VK shared similarity block must contain three cosines")
if len(ORDERED_OPERAND_FIELDS) != 55:
    raise AssertionError("VK multi-target P_shared width must be 55")
if len(ORDERED_V_FIELDS) != 21:
    raise AssertionError("VK multi-target validity width must be 21")
if sum(ORDERED_DENOMINATOR_MASK) != 5:
    raise AssertionError("VK multi-target reciprocal mask must activate five fields")


class VKLSVDMultiTargetError(RuntimeError):
    """The frozen multi-target view contract was violated."""


FeatureRegime = Literal["REDUCED", "FULL"]


@dataclass(frozen=True)
class TransformedPair:
    """One bitwise-matched REDUCED/FULL pair for a single allowed week."""

    week: int
    reduced: pa.Table
    full: pa.Table
    shared_values_sha256: str
    labels_and_masks_sha256: str
    stable_row_ids_sha256: str

    def __post_init__(self) -> None:
        if self.week not in ALLOWED_WEEKS:
            raise ValueError("transformed week is outside train/validation")
        if self.reduced.num_rows <= 0 or self.full.num_rows != self.reduced.num_rows:
            raise ValueError("transformed pair must be non-empty and row matched")


def ordered_schema_sha256(values: Sequence[str | bool]) -> str:
    payload = (
        "[" + ",".join(_canonical_scalar(value) for value in values) + "]"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _canonical_scalar(value: str | bool) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if not isinstance(value, str):
        raise TypeError("schema values must be exact strings or booleans")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def view_schema(regime: FeatureRegime) -> pa.Schema:
    """Return the exact physical schema for one multi-target feature regime."""

    if regime not in {"REDUCED", "FULL"}:
        raise ValueError("feature regime must be REDUCED or FULL")
    fields = [
        pa.field("user_id", pa.uint32(), nullable=False),
        pa.field("item_id", pa.uint32(), nullable=False),
        pa.field("week", pa.uint8(), nullable=False),
        pa.field("source_row_index0", pa.uint64(), nullable=False),
        pa.field("stable_row_id", pa.string(), nullable=False),
        pa.field("label_timespent_seconds", pa.float32(), nullable=False),
    ]
    fields.extend(
        pa.field(name, pa.float32(), nullable=False) for name in TARGET_COLUMNS
    )
    fields.extend(
        pa.field(name, pa.bool_(), nullable=False) for name in TARGET_MASK_COLUMNS
    )
    fields.extend(
        pa.field(name, pa.uint32(), nullable=False) for name in ORDERED_C_FIELDS
    )
    fields.extend(
        pa.field(name, pa.float32(), nullable=False) for name in ORDERED_OPERAND_FIELDS
    )
    if regime == "FULL":
        fields.extend(
            pa.field(name, pa.float32(), nullable=False) for name in R_DIV_FIELDS
        )
    fields.extend(
        pa.field(name, pa.bool_(), nullable=False) for name in VALIDITY_COLUMNS
    )
    metadata = {
        b"dcdn.dataset": DATASET_ID.encode("ascii"),
        b"dcdn.feature_regime": regime.encode("ascii"),
        b"dcdn.objective": b"macro_nloss",
        b"dcdn.target_names": ",".join(TARGET_NAMES).encode("ascii"),
        b"dcdn.test_unlocked": b"false",
    }
    return pa.schema(fields, metadata=metadata)


def physical_schema_sha256(schema: pa.Schema) -> str:
    records = "\n".join(
        f"{field.name}\t{field.type}\t{str(field.nullable).lower()}" for field in schema
    ).encode("ascii")
    return hashlib.sha256(records).hexdigest()


def _numpy(table: pa.Table, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise VKLSVDMultiTargetError(f"required column is absent: {name}")
    column = table[name].combine_chunks()
    try:
        return column.to_numpy(zero_copy_only=False)
    except pa.ArrowInvalid as error:
        raise VKLSVDMultiTargetError(f"cannot read required column: {name}") from error


def _sha256_array(digest: Any, name: str, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    header = f"{name}\0{contiguous.dtype.str}\0{contiguous.shape}\0".encode("ascii")
    digest.update(header)
    digest.update(memoryview(contiguous).cast("B"))


def _sha256_strings(digest: Any, name: str, values: Sequence[str]) -> None:
    digest.update(f"{name}\0{len(values)}\0".encode("ascii"))
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)


def _require_input_columns(feature_table: pa.Table, interactions: pa.Table) -> None:
    feature_required = {
        "user_id",
        "item_id",
        "week",
        "source_row_index0",
        "stable_row_id",
        "label_normalized_watch_time",
        "label_timespent_seconds",
        *ORDERED_C_FIELDS,
        *ORDERED_P0_FIELDS,
        *SHARED_SIMILARITY_FIELDS,
        *ORDERED_V_FIELDS,
        *R_DIV_FIELDS,
        *(f"denominator_validity__{name}" for name in DENOMINATOR_ELIGIBLE_FIELDS),
    }
    interaction_required = {
        "user_id",
        "item_id",
        "like",
        "dislike",
        "share",
        "bookmark",
        "click_on_author",
        "open_comments",
    }
    missing_features = feature_required - set(feature_table.column_names)
    missing_interactions = interaction_required - set(interactions.column_names)
    if missing_features:
        raise VKLSVDMultiTargetError(
            f"legacy feature shard misses columns: {sorted(missing_features)}"
        )
    if missing_interactions:
        raise VKLSVDMultiTargetError(
            f"interaction shard misses columns: {sorted(missing_interactions)}"
        )


def transform_verified_shard(
    *,
    week: int,
    legacy_full: pa.Table,
    interactions: pa.Table,
) -> TransformedPair:
    """Transform one verified legacy FULL shard into the frozen headline pair.

    The function deliberately accepts in-memory Arrow tables so that secure
    file opening, sidecar verification, atomic publication, and manifest
    binding remain the responsibility of the caller.  It performs all semantic
    and row-alignment checks before returning any output table.
    """

    if isinstance(week, bool) or not isinstance(week, int) or week not in ALLOWED_WEEKS:
        raise VKLSVDMultiTargetError("only VK train weeks 20--24 and validation 25")
    if not isinstance(legacy_full, pa.Table) or not isinstance(interactions, pa.Table):
        raise TypeError("legacy_full and interactions must be Arrow tables")
    if legacy_full.num_rows <= 0 or interactions.num_rows != legacy_full.num_rows:
        raise VKLSVDMultiTargetError("feature and interaction rows must match")
    _require_input_columns(legacy_full, interactions)

    rows = legacy_full.num_rows
    user_ids = _numpy(legacy_full, "user_id").astype(np.uint32, copy=False)
    item_ids = _numpy(legacy_full, "item_id").astype(np.uint32, copy=False)
    raw_user_ids = _numpy(interactions, "user_id").astype(np.uint32, copy=False)
    raw_item_ids = _numpy(interactions, "item_id").astype(np.uint32, copy=False)
    if not np.array_equal(user_ids, raw_user_ids) or not np.array_equal(
        item_ids, raw_item_ids
    ):
        raise VKLSVDMultiTargetError("raw labels are not row-aligned by user/item")

    week_values = _numpy(legacy_full, "week")
    if week_values.dtype != np.dtype("uint8") or not np.all(week_values == week):
        raise VKLSVDMultiTargetError("feature rows do not match the declared week")
    row_indices = _numpy(legacy_full, "source_row_index0").astype(np.uint64, copy=False)
    if not np.array_equal(row_indices, np.arange(rows, dtype=np.uint64)):
        raise VKLSVDMultiTargetError("source_row_index0 must be contiguous from zero")
    stable_ids = tuple(str(value) for value in legacy_full["stable_row_id"].to_pylist())
    expected_ids = tuple(
        stable_row_id(week=week, source_row_index0=index) for index in range(rows)
    )
    if stable_ids != expected_ids:
        raise VKLSVDMultiTargetError("stable row identity drift")

    watch = _numpy(legacy_full, "label_normalized_watch_time").astype(
        np.float32, copy=False
    )
    seconds = _numpy(legacy_full, "label_timespent_seconds").astype(
        np.float32, copy=False
    )
    if not np.isfinite(watch).all() or np.any((watch < 0.0) | (watch > 1.0)):
        raise VKLSVDMultiTargetError("watch-time target is not finite in [0,1]")
    labels = {
        "label_watch_time": watch,
        "label_like": _numpy(interactions, "like").astype(np.float32),
        "label_dislike": _numpy(interactions, "dislike").astype(np.float32),
        "label_share": _numpy(interactions, "share").astype(np.float32),
        "label_bookmark": _numpy(interactions, "bookmark").astype(np.float32),
        "label_click_on_author": _numpy(interactions, "click_on_author").astype(
            np.float32
        ),
        "label_open_comments": _numpy(interactions, "open_comments").astype(np.float32),
    }
    for name, values in labels.items():
        if values.dtype != np.dtype("float32") or not np.isfinite(values).all():
            raise VKLSVDMultiTargetError(f"target is not finite FP32: {name}")
        if name != "label_watch_time" and np.any(
            (values != np.float32(0.0)) & (values != np.float32(1.0))
        ):
            raise VKLSVDMultiTargetError(f"binary target is outside zero/one: {name}")
    masks = {name: np.ones(rows, dtype=np.bool_) for name in TARGET_MASK_COLUMNS}

    context = {
        name: _numpy(legacy_full, name).astype(np.uint32, copy=False)
        for name in ORDERED_C_FIELDS
    }
    operands = {
        name: _numpy(legacy_full, name).astype(np.float32, copy=False)
        for name in ORDERED_OPERAND_FIELDS
    }
    if any(not np.isfinite(values).all() for values in operands.values()):
        raise VKLSVDMultiTargetError("shared operands contain NaN or Inf")
    r_div = {
        name: _numpy(legacy_full, name).astype(np.float32, copy=False)
        for name in R_DIV_FIELDS
    }
    if any(not np.isfinite(values).all() for values in r_div.values()):
        raise VKLSVDMultiTargetError("R_div contains NaN or Inf")
    validity = {
        name: (
            _numpy(legacy_full, f"denominator_validity__{name}").astype(
                np.bool_, copy=False
            )
            if name in DENOMINATOR_ELIGIBLE_FIELDS
            else np.ones(rows, dtype=np.bool_)
        )
        for name in ORDERED_OPERAND_FIELDS
    }

    shared: dict[str, Any] = {
        "user_id": user_ids,
        "item_id": item_ids,
        "week": week_values,
        "source_row_index0": row_indices,
        "stable_row_id": stable_ids,
        "label_timespent_seconds": seconds,
        **labels,
        **masks,
        **context,
        **operands,
        **{
            f"denominator_validity__{name}": values for name, values in validity.items()
        },
    }
    reduced_schema = view_schema("REDUCED")
    full_schema = view_schema("FULL")
    reduced = pa.Table.from_arrays(
        [pa.array(shared[field.name], type=field.type) for field in reduced_schema],
        schema=reduced_schema,
    )
    full_values = {**shared, **r_div}
    full = pa.Table.from_arrays(
        [pa.array(full_values[field.name], type=field.type) for field in full_schema],
        schema=full_schema,
    )

    shared_names = tuple(
        name for name in reduced.column_names if name in full.column_names
    )
    if not reduced.select(shared_names).equals(full.select(shared_names)):
        raise AssertionError("FULL and REDUCED shared blocks are not bitwise equal")
    if set(full.column_names) - set(reduced.column_names) != set(R_DIV_FIELDS):
        raise AssertionError("FULL minus REDUCED is not exactly R_div")

    shared_digest = hashlib.sha256()
    for name in (*ORDERED_C_FIELDS, *ORDERED_OPERAND_FIELDS):
        _sha256_array(shared_digest, name, _numpy(reduced, name))
    labels_digest = hashlib.sha256()
    for name in (*TARGET_COLUMNS, *TARGET_MASK_COLUMNS):
        _sha256_array(labels_digest, name, _numpy(reduced, name))
    ids_digest = hashlib.sha256()
    _sha256_strings(ids_digest, "stable_row_id", stable_ids)
    return TransformedPair(
        week=week,
        reduced=reduced,
        full=full,
        shared_values_sha256=shared_digest.hexdigest(),
        labels_and_masks_sha256=labels_digest.hexdigest(),
        stable_row_ids_sha256=ids_digest.hexdigest(),
    )


def contract_state() -> Mapping[str, Any]:
    """Canonical JSON-ready structural state bound by the future manifest."""

    return {
        "dataset_id": DATASET_ID,
        "target_names": list(TARGET_NAMES),
        "target_kinds": list(TARGET_KINDS),
        "target_columns": list(TARGET_COLUMNS),
        "target_mask_columns": list(TARGET_MASK_COLUMNS),
        "ordered_C": list(ORDERED_C_FIELDS),
        "ordered_P_shared": list(ORDERED_OPERAND_FIELDS),
        "ordered_R_div": list(R_DIV_FIELDS),
        "ordered_D": list(DENOMINATOR_ELIGIBLE_FIELDS),
        "ordered_denominator_mask": list(ORDERED_DENOMINATOR_MASK),
        "similarity_fields_shared": list(SHARED_SIMILARITY_FIELDS),
        "test_unlocked": False,
        "allowed_weeks": list(ALLOWED_WEEKS),
        "schema_hashes": {
            "P_shared": ordered_schema_sha256(ORDERED_OPERAND_FIELDS),
            "R_div": ordered_schema_sha256(R_DIV_FIELDS),
            "denominator_mask": ordered_schema_sha256(ORDERED_DENOMINATOR_MASK),
            "REDUCED_physical": physical_schema_sha256(view_schema("REDUCED")),
            "FULL_physical": physical_schema_sha256(view_schema("FULL")),
        },
    }
