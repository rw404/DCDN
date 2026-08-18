"""Fail-closed KuaiRec feature, split, and leakage contract.

The module is deliberately pre-metric.  It freezes the HEGM-comparable
positive-watch estimand, global 80/10/10 split, train-only mappings, strict
``date < target_date`` daily-history rule, and the ordered ``C/P/V/E``
schemas.  It never opens raw data itself and never evaluates a model.

``watch_ratio`` and the current interaction's ``play_duration`` are outcomes,
not inputs.  A companion command may apply the frozen positivity predicate to
``play_duration`` without exposing values, but test outcomes remain sealed.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import stat
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from dcdn.source_manifest import load_source_manifest

DATASET_ID = "kuairec_big_matrix"
SOURCE_DATASET_ID = "kuairec-big-matrix"
FEATURE_SCHEMA_VERSION = 1
SOURCE_MANIFEST_PATH = "data/manifests/kuairec_source_v1.yaml"
SOURCE_MANIFEST_SHA256 = (
    "963019bcb4c8dbd4663f23c24a472fe0332f6494fdcbbaa74ca158a73488c5b4"
)
RAW_ARCHIVE_SHA256 = "c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c"
BIG_MATRIX_SHA256 = "4ee1a72ff9ce4e86ef50525da9dfd7aefcefa717b37c9d570010f689c42f4ef9"
DAILY_FEATURES_SHA256 = (
    "45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b"
)
HEGM_REVISION = "d16c097d57f867f438f906949d9cc1efd0f6ddc4"
HEGM_PREPROCESS_SHA256 = (
    "184729b5dd200a3d8f8e13ae83179286242b586fb27162bb7b713e0988ae2807"
)
HEGM_LICENSE_SHA256 = "eb74dde2903b7a84802e5907b35717441530b25738fc00c0b6639bb8287878c3"

MAX_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_SIDECAR_BYTES = 256
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MILLISECONDS_PER_SECOND = 1000.0

BIG_MATRIX_COLUMNS = (
    "user_id",
    "video_id",
    "play_duration",
    "video_duration",
    "time",
    "date",
    "timestamp",
    "watch_ratio",
)
DAILY_COLUMNS = (
    "video_id",
    "date",
    "author_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "visible_status",
    "video_duration",
    "video_width",
    "video_height",
    "music_id",
    "video_tag_id",
    "video_tag_name",
    "show_cnt",
    "show_user_num",
    "play_cnt",
    "play_user_num",
    "play_duration",
    "complete_play_cnt",
    "complete_play_user_num",
    "valid_play_cnt",
    "valid_play_user_num",
    "long_time_play_cnt",
    "long_time_play_user_num",
    "short_time_play_cnt",
    "short_time_play_user_num",
    "play_progress",
    "comment_stay_duration",
    "like_cnt",
    "like_user_num",
    "click_like_cnt",
    "double_click_cnt",
    "cancel_like_cnt",
    "cancel_like_user_num",
    "comment_cnt",
    "comment_user_num",
    "direct_comment_cnt",
    "reply_comment_cnt",
    "delete_comment_cnt",
    "delete_comment_user_num",
    "comment_like_cnt",
    "comment_like_user_num",
    "follow_cnt",
    "follow_user_num",
    "cancel_follow_cnt",
    "cancel_follow_user_num",
    "share_cnt",
    "share_user_num",
    "download_cnt",
    "download_user_num",
    "report_cnt",
    "report_user_num",
    "reduce_similar_cnt",
    "reduce_similar_user_num",
    "collect_cnt",
    "collect_user_num",
    "cancel_collect_cnt",
    "cancel_collect_user_num",
)
EXPECTED_DAILY_ROWS = 343_341
EXPECTED_PARTIAL_NULL_ROWS = 69_683
PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES = (
    "collect_cnt",
    "collect_user_num",
    "cancel_collect_cnt",
    "cancel_collect_user_num",
)


ORDERED_C_FIELDS = (
    "c_user_id",
    "c_video_id",
    "c_first_level_category_id",
    "c_second_level_category_id",
    "c_third_level_category_id",
)

CORE_DAILY_SOURCES = (
    "show_cnt",
    "show_user_num",
    "play_cnt",
    "play_user_num",
    "play_duration",
    "complete_play_cnt",
    "complete_play_user_num",
    "valid_play_cnt",
    "valid_play_user_num",
    "long_time_play_cnt",
    "long_time_play_user_num",
    "short_time_play_cnt",
    "short_time_play_user_num",
)
ENGAGEMENT_DAILY_SOURCES = (
    "like_cnt",
    "like_user_num",
    "click_like_cnt",
    "double_click_cnt",
    "cancel_like_cnt",
    "cancel_like_user_num",
    "comment_cnt",
    "comment_user_num",
    "direct_comment_cnt",
    "reply_comment_cnt",
    "delete_comment_cnt",
    "delete_comment_user_num",
    "comment_like_cnt",
    "comment_like_user_num",
    "follow_cnt",
    "follow_user_num",
    "cancel_follow_cnt",
    "cancel_follow_user_num",
    "share_cnt",
    "share_user_num",
    "download_cnt",
    "download_user_num",
    "report_cnt",
    "report_user_num",
    "reduce_similar_cnt",
    "reduce_similar_user_num",
)
DAILY_BASE_SOURCES = CORE_DAILY_SOURCES + ENGAGEMENT_DAILY_SOURCES


def history_field(source: str) -> str:
    """Return the unambiguous model-facing name for one lagged daily sum."""

    if source not in DAILY_BASE_SOURCES:
        raise ValueError(f"unknown daily source: {source}")
    if source == "play_duration":
        return "p_hist_play_duration_sum"
    return f"p_hist_{source}"


ORDERED_HISTORY_P_FIELDS = tuple(history_field(source) for source in DAILY_BASE_SOURCES)
SHIFTED_DENOMINATOR_FIELDS = (
    "p_hist_show_cnt_plus_1",
    "p_hist_show_user_num_plus_1",
    "p_hist_play_cnt_plus_1",
    "p_hist_play_user_num_plus_1",
)
ORDERED_P_FIELDS = (
    "p_video_duration_seconds",
    *ORDERED_HISTORY_P_FIELDS,
    *SHIFTED_DENOMINATOR_FIELDS,
)
ORDERED_V_FIELDS = (
    "v_prior_daily_history_present",
    "v_hist_show_cnt_nonzero",
    "v_hist_show_user_num_nonzero",
    "v_hist_play_cnt_nonzero",
    "v_hist_play_user_num_nonzero",
    "v_video_duration_positive",
)

RATE_FAMILIES = (
    "complete_play",
    "valid_play",
    "long_time_play",
    "short_time_play",
)
ORDERED_CORE_E_FIELDS = (
    "e_play_per_show",
    "e_unique_play_per_unique_show",
    *(
        f"e_{family}_{denominator}"
        for family in RATE_FAMILIES
        for denominator in ("per_play", "per_show")
    ),
    *(
        f"e_{family}_user_{denominator}"
        for family in RATE_FAMILIES
        for denominator in ("per_play_user", "per_show_user")
    ),
    "e_average_play_duration_seconds",
    "e_play_duration_per_show_seconds",
)
ORDERED_ENGAGEMENT_E_FIELDS = tuple(
    f"e_{source.removesuffix('_cnt').removesuffix('_user_num')}_per_"
    f"{'show_user' if source.endswith('_user_num') else 'show'}"
    for source in ENGAGEMENT_DAILY_SOURCES
)
ORDERED_E_FIELDS = ORDERED_CORE_E_FIELDS + ORDERED_ENGAGEMENT_E_FIELDS

DENOMINATOR_ELIGIBLE_FIELDS = (
    "p_video_duration_seconds",
    *SHIFTED_DENOMINATOR_FIELDS,
)
ORDERED_NUMERIC_OPERANDS = ORDERED_P_FIELDS + ORDERED_V_FIELDS
_DENOMINATOR_SET = frozenset(DENOMINATOR_ELIGIBLE_FIELDS)
ORDERED_DENOMINATOR_MASK = tuple(
    field in _DENOMINATOR_SET for field in ORDERED_NUMERIC_OPERANDS
)

MODEL_FORBIDDEN_RAW_FIELDS = frozenset(
    {
        "play_duration",
        "watch_ratio",
        "time",
        "date",
        "timestamp",
        "source_row_index0",
        "stable_row_id",
        "video_type",
        "is_ad",
    }
)


class KuaiRecContractError(RuntimeError):
    """Raised when the frozen KuaiRec contract cannot be trusted."""


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
        try:
            duplicate = key in result
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ordered_schema_sha256(fields: Sequence[str]) -> str:
    encoded = json.dumps(list(fields), ensure_ascii=True, separators=(",", ":")).encode(
        "ascii"
    )
    return sha256_bytes(encoded)


def denominator_mask_sha256(mask: Sequence[bool]) -> str:
    encoded = json.dumps(list(mask), ensure_ascii=True, separators=(",", ":")).encode(
        "ascii"
    )
    return sha256_bytes(encoded)


def stable_row_id(source_row_index0: int) -> str:
    """Identify a raw big-matrix row without hashing any outcome value."""

    if type(source_row_index0) is not int or source_row_index0 < 0:
        raise KuaiRecContractError("source_row_index0 must be a non-negative integer")
    payload = (f"{DATASET_ID}\0{BIG_MATRIX_SHA256}\0{source_row_index0}").encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SplitBoundaries:
    total_filtered_rows: int
    train_end_exclusive: int
    validation_end_exclusive: int

    @property
    def train_rows(self) -> int:
        return self.train_end_exclusive

    @property
    def validation_rows_before_cold_drop(self) -> int:
        return self.validation_end_exclusive - self.train_end_exclusive

    @property
    def test_rows_before_cold_drop(self) -> int:
        return self.total_filtered_rows - self.validation_end_exclusive


def split_boundaries(total_filtered_rows: int) -> SplitBoundaries:
    """Reproduce HEGM's floor(0.8N), floor(0.1N), remainder split."""

    if type(total_filtered_rows) is not int or total_filtered_rows <= 0:
        raise KuaiRecContractError("total_filtered_rows must be positive")
    train = int(0.8 * total_filtered_rows)
    validation = int(0.1 * total_filtered_rows)
    if train <= 0 or validation <= 0 or train + validation >= total_filtered_rows:
        raise KuaiRecContractError("dataset is too small for a nonempty 80/10/10 split")
    return SplitBoundaries(total_filtered_rows, train, train + validation)


def split_role(sorted_rank0: int, boundaries: SplitBoundaries) -> str:
    if (
        type(sorted_rank0) is not int
        or not 0 <= sorted_rank0 < boundaries.total_filtered_rows
    ):
        raise KuaiRecContractError("sorted_rank0 falls outside the filtered population")
    if sorted_rank0 < boundaries.train_end_exclusive:
        return "train"
    if sorted_rank0 < boundaries.validation_end_exclusive:
        return "validation"
    return "sealed_test"


def fit_sorted_mapping(
    values: Iterable[int | str], *, first_index: int = 0
) -> dict[int | str, int]:
    """Fit a deterministic train-only categorical mapping."""

    if type(first_index) is not int or first_index < 0:
        raise KuaiRecContractError("first_index must be a non-negative integer")
    unique = set(values)
    try:
        ordered = sorted(unique)
    except TypeError as error:
        raise KuaiRecContractError(
            "mapping values must have one comparable type"
        ) from error
    return {value: first_index + index for index, value in enumerate(ordered)}


def linear_quantile(values: Iterable[float], q: float = 0.99) -> float:
    """Exact R-7/NumPy-linear quantile used for train-only constants."""

    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise KuaiRecContractError("q must be finite and lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise KuaiRecContractError("cannot fit a quantile from an empty training set")
    if any(not math.isfinite(value) or value <= 0.0 for value in ordered):
        raise KuaiRecContractError(
            "duration quantile inputs must be finite and positive"
        )
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_released_date(value: str | int) -> date:
    text = str(value)
    if len(text) != 8 or not text.isdigit():
        raise KuaiRecContractError("released date must use YYYYMMDD")
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError as error:
        raise KuaiRecContractError(f"invalid released date: {text}") from error


@dataclass(frozen=True)
class DailyPrefixRow:
    video_id: int
    day: date
    values: Mapping[str, float]


@dataclass(frozen=True)
class DailyPrefixIndex:
    days_by_video: Mapping[int, tuple[date, ...]]
    values_by_video: Mapping[int, tuple[Mapping[str, float], ...]]


def _validated_daily_values(values: Mapping[str, float]) -> dict[str, float]:
    unexpected = set(values) - set(DAILY_BASE_SOURCES)
    missing = set(DAILY_BASE_SOURCES) - set(values)
    if unexpected or missing:
        detail = f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        raise KuaiRecContractError(f"daily values mismatch; {detail}")
    validated: dict[str, float] = {}
    for source in DAILY_BASE_SOURCES:
        value = float(values[source])
        if not math.isfinite(value) or value < 0.0:
            raise KuaiRecContractError(
                f"daily base {source} must be finite and non-negative"
            )
        validated[source] = value
    return validated


def build_daily_prefix_index(rows: Iterable[DailyPrefixRow]) -> DailyPrefixIndex:
    """Convert raw daily milliseconds, aggregate video-days, and prefix-sum."""

    by_video_day: dict[tuple[int, date], dict[str, float]] = {}
    for row in rows:
        if type(row.video_id) is not int or row.video_id < 0:
            raise KuaiRecContractError("video_id must be a non-negative integer")
        values = _validated_daily_values(row.values)
        values["play_duration"] /= MILLISECONDS_PER_SECOND
        key = (row.video_id, row.day)
        aggregate = by_video_day.setdefault(
            key, {source: 0.0 for source in DAILY_BASE_SOURCES}
        )
        for source, value in values.items():
            aggregate[source] += value
            if not math.isfinite(aggregate[source]):
                raise KuaiRecContractError("daily duplicate aggregation overflowed")

    grouped: dict[int, list[tuple[date, Mapping[str, float]]]] = defaultdict(list)
    for (video_id, day), values in by_video_day.items():
        grouped[video_id].append((day, values))

    days_by_video: dict[int, tuple[date, ...]] = {}
    values_by_video: dict[int, tuple[Mapping[str, float], ...]] = {}
    for video_id, entries in grouped.items():
        entries.sort(key=lambda item: item[0])
        cumulative = {source: 0.0 for source in DAILY_BASE_SOURCES}
        days: list[date] = []
        snapshots: list[Mapping[str, float]] = []
        for day, values in entries:
            for source, value in values.items():
                cumulative[source] += value
            days.append(day)
            snapshots.append(dict(cumulative))
        days_by_video[video_id] = tuple(days)
        values_by_video[video_id] = tuple(snapshots)
    return DailyPrefixIndex(days_by_video, values_by_video)


def daily_history_before(
    index: DailyPrefixIndex,
    *,
    video_id: int,
    target_day: date,
) -> tuple[dict[str, float], bool]:
    """Return cumulative values through the latest day strictly before target."""

    days = index.days_by_video.get(video_id, ())
    position = bisect.bisect_left(days, target_day) - 1
    if position < 0:
        return ({source: 0.0 for source in DAILY_BASE_SOURCES}, False)
    values = index.values_by_video[video_id][position]
    return (dict(values), True)


def _safe_rate(numerator: float, shifted_denominator: float) -> float:
    if not math.isfinite(numerator) or numerator < 0.0:
        raise KuaiRecContractError("rate numerator must be finite and non-negative")
    if not math.isfinite(shifted_denominator) or shifted_denominator <= 0.0:
        raise KuaiRecContractError("rate denominator must be finite and positive")
    return numerator / shifted_denominator


def build_feature_values(
    *,
    video_duration_seconds: float,
    history: Mapping[str, float],
    prior_history_present: bool,
) -> dict[str, dict[str, float]]:
    """Evaluate the frozen P/V/E algebra for one row without current outcomes."""

    values = _validated_daily_values(history)
    duration = float(video_duration_seconds)
    if not math.isfinite(duration) or duration <= 0.0:
        raise KuaiRecContractError("video_duration_seconds must be finite and positive")

    p: dict[str, float] = {"p_video_duration_seconds": duration}
    for source in DAILY_BASE_SOURCES:
        p[history_field(source)] = values[source]
    p.update(
        {
            "p_hist_show_cnt_plus_1": values["show_cnt"] + 1.0,
            "p_hist_show_user_num_plus_1": values["show_user_num"] + 1.0,
            "p_hist_play_cnt_plus_1": values["play_cnt"] + 1.0,
            "p_hist_play_user_num_plus_1": values["play_user_num"] + 1.0,
        }
    )

    v = {
        "v_prior_daily_history_present": float(bool(prior_history_present)),
        "v_hist_show_cnt_nonzero": float(values["show_cnt"] > 0.0),
        "v_hist_show_user_num_nonzero": float(values["show_user_num"] > 0.0),
        "v_hist_play_cnt_nonzero": float(values["play_cnt"] > 0.0),
        "v_hist_play_user_num_nonzero": float(values["play_user_num"] > 0.0),
        "v_video_duration_positive": 1.0,
    }

    show = p["p_hist_show_cnt_plus_1"]
    show_users = p["p_hist_show_user_num_plus_1"]
    play = p["p_hist_play_cnt_plus_1"]
    play_users = p["p_hist_play_user_num_plus_1"]
    e: dict[str, float] = {
        "e_play_per_show": _safe_rate(values["play_cnt"], show),
        "e_unique_play_per_unique_show": _safe_rate(
            values["play_user_num"], show_users
        ),
    }
    for family in RATE_FAMILIES:
        e[f"e_{family}_per_play"] = _safe_rate(values[f"{family}_cnt"], play)
        e[f"e_{family}_per_show"] = _safe_rate(values[f"{family}_cnt"], show)
    for family in RATE_FAMILIES:
        e[f"e_{family}_user_per_play_user"] = _safe_rate(
            values[f"{family}_user_num"], play_users
        )
        e[f"e_{family}_user_per_show_user"] = _safe_rate(
            values[f"{family}_user_num"], show_users
        )
    e["e_average_play_duration_seconds"] = _safe_rate(values["play_duration"], play)
    e["e_play_duration_per_show_seconds"] = _safe_rate(values["play_duration"], show)
    for source in ENGAGEMENT_DAILY_SOURCES:
        denominator = show_users if source.endswith("_user_num") else show
        name = (
            f"e_{source.removesuffix('_cnt').removesuffix('_user_num')}_per_"
            f"{'show_user' if source.endswith('_user_num') else 'show'}"
        )
        e[name] = _safe_rate(values[source], denominator)

    if tuple(p) != ORDERED_P_FIELDS:
        raise KuaiRecContractError("P feature construction drifted from frozen order")
    if tuple(v) != ORDERED_V_FIELDS:
        raise KuaiRecContractError("V feature construction drifted from frozen order")
    if tuple(e) != ORDERED_E_FIELDS:
        raise KuaiRecContractError("E feature construction drifted from frozen order")
    return {"P": p, "V": v, "E": e}


def _read_bounded_regular(path: Path, *, limit: int, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise KuaiRecContractError(f"{label} must be a regular file: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise KuaiRecContractError(f"{label} must be a private regular file")
        if metadata.st_size > limit:
            raise KuaiRecContractError(f"{label} exceeds size limit")
        with os.fdopen(descriptor, "rb") as handle:
            data = handle.read(limit + 1)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise
    if len(data) > limit:
        raise KuaiRecContractError(f"{label} exceeds size limit")
    return data


def _require_exact_bool(payload: Mapping[str, Any], key: str, expected: bool) -> None:
    if type(payload.get(key)) is not bool or payload[key] is not expected:
        raise KuaiRecContractError(f"{key} must be exactly {expected}")


def validate_feature_contract(payload: Mapping[str, Any]) -> None:
    """Reject any drift in the predeclared KuaiRec feature/split protocol."""

    if payload.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise KuaiRecContractError("schema_version must equal 1")
    if payload.get("dataset_id") != DATASET_ID:
        raise KuaiRecContractError(f"dataset_id must equal {DATASET_ID}")
    if payload.get("artifact_role") != "pre_metric_feature_contract":
        raise KuaiRecContractError("unexpected artifact_role")
    for key in (
        "test_unlocked",
        "test_labels_opened",
        "model_metrics_read",
        "final_test_view_materialized",
    ):
        _require_exact_bool(payload, key, False)

    initial_revision = payload.get("schema_only_revision")
    if not isinstance(initial_revision, Mapping):
        raise KuaiRecContractError("schema_only_revision must be a mapping")
    if initial_revision.get("initial_contract_sha256") != (
        "c6f539d228dce75644b62cb445bf393e141944c9e66bf84545db77e3b75a606d"
    ):
        raise KuaiRecContractError("initial rejected contract hash drifted")
    if initial_revision.get("failure_reason") != (
        "four_declared_daily_bases_are_partially_null"
    ):
        raise KuaiRecContractError("initial failure gate drifted")
    if initial_revision.get("daily_rows_checked") != EXPECTED_DAILY_ROWS:
        raise KuaiRecContractError("corrected daily row count drifted")
    if (
        initial_revision.get("null_rows_per_excluded_source")
        != EXPECTED_PARTIAL_NULL_ROWS
    ):
        raise KuaiRecContractError("partial-null count drifted")
    if initial_revision.get("partially_null_sources") != list(
        PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES
    ):
        raise KuaiRecContractError("partially-null source order drifted")
    _require_exact_bool(initial_revision, "source_wide_all_null", False)
    _require_exact_bool(
        initial_revision, "owner_approved_full_exclusion_all_regimes", True
    )
    _require_exact_bool(initial_revision, "zero_imputation_used", False)
    _require_exact_bool(initial_revision, "row_or_population_filter_added", False)
    _require_exact_bool(initial_revision, "filtering_and_split_changed", False)

    unit_revision = payload.get("unit_normalization_revision")
    if not isinstance(unit_revision, Mapping):
        raise KuaiRecContractError("unit_normalization_revision must be a mapping")
    if unit_revision.get("rejected_contract_sha256") != (
        "c5aadbeb7da5f05a281681d30a7ef98e8c660019f6c0bc679e430c1014e5a23c"
    ):
        raise KuaiRecContractError("unit-rejected contract hash drifted")
    if unit_revision.get("failure_gate") != (
        "train_only_video_duration_q99_seconds_sanity_interval"
    ):
        raise KuaiRecContractError("unit failure gate drifted")
    _require_exact_bool(unit_revision, "filtering_and_split_changed", False)
    _require_exact_bool(unit_revision, "ordered_schema_names_or_hashes_changed", False)
    _require_exact_bool(unit_revision, "watch_ratio_read", False)
    _require_exact_bool(
        unit_revision, "model_or_validation_or_test_metrics_read", False
    )
    _require_exact_bool(unit_revision, "audit_artifact_written", False)

    incident = payload.get("inspection_incident")
    if not isinstance(incident, Mapping):
        raise KuaiRecContractError("inspection_incident must be a mapping")
    if incident.get("retained_schema_fact") != (
        "official_duration_fields_are_milliseconds"
    ):
        raise KuaiRecContractError("duration provenance incident drifted")
    _require_exact_bool(incident, "quarantined_values_persisted_or_used", False)
    _require_exact_bool(incident, "local_test_label_rows_or_vectors_read", False)
    _require_exact_bool(incident, "DCDN_outputs_or_model_metrics_read", False)
    _require_exact_bool(incident, "test_unlocked", False)

    invalid_artifacts = payload.get("invalid_intermediate_artifacts")
    if not isinstance(invalid_artifacts, Mapping):
        raise KuaiRecContractError("invalid_intermediate_artifacts must be a mapping")
    false_claim = invalid_artifacts.get("false_source_wide_claim_contract")
    prior_scan = invalid_artifacts.get("prior_blind_scan")
    if not isinstance(false_claim, Mapping) or false_claim.get("sha256") != (
        "f0883ced4c8fb3f0945cacb1affa0b4891f748c4294e3a2d1016100191011b8d"
    ):
        raise KuaiRecContractError("invalid intermediate contract evidence drifted")
    if not isinstance(prior_scan, Mapping) or prior_scan.get("sha256") != (
        "3b38391097c9294b7622bc91c7b836283247b7ff41bbf41a0a7a4e84c35357ab"
    ):
        raise KuaiRecContractError("invalid prior blind scan evidence drifted")
    if false_claim.get("status") != "invalid_not_usable":
        raise KuaiRecContractError("invalid intermediate contract status drifted")
    if prior_scan.get("status") != "invalid_not_usable":
        raise KuaiRecContractError("invalid prior blind scan status drifted")

    inspection = payload.get("inspection_boundary")
    if not isinstance(inspection, Mapping):
        raise KuaiRecContractError("inspection_boundary must be a mapping")
    if inspection.get("blind_scan_daily_audit_only_excluded_columns") != list(
        PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES
    ):
        raise KuaiRecContractError("partial-null audit-only columns drifted")
    _require_exact_bool(
        inspection, "audit_only_excluded_values_exposed_or_persisted", False
    )

    source = payload.get("source_manifest")
    if not isinstance(source, Mapping):
        raise KuaiRecContractError("source_manifest must be a mapping")
    expected_source = {
        "path": SOURCE_MANIFEST_PATH,
        "sha256": SOURCE_MANIFEST_SHA256,
        "raw_archive_sha256": RAW_ARCHIVE_SHA256,
        "big_matrix_sha256": BIG_MATRIX_SHA256,
        "item_daily_features_sha256": DAILY_FEATURES_SHA256,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise KuaiRecContractError(f"source_manifest.{key} drifted")

    provenance = payload.get("hegm_provenance")
    if not isinstance(provenance, Mapping):
        raise KuaiRecContractError("hegm_provenance must be a mapping")
    if provenance.get("revision") != HEGM_REVISION:
        raise KuaiRecContractError("HEGM revision drifted")
    if provenance.get("preprocess_kuairec_sha256") != HEGM_PREPROCESS_SHA256:
        raise KuaiRecContractError("HEGM preprocessing hash drifted")
    if provenance.get("license_sha256") != HEGM_LICENSE_SHA256:
        raise KuaiRecContractError("HEGM license hash drifted")

    units = payload.get("unit_normalization")
    if not isinstance(units, Mapping):
        raise KuaiRecContractError("unit_normalization must be a mapping")
    expected_raw_units = {
        "big_matrix.play_duration": "milliseconds",
        "big_matrix.video_duration": "milliseconds",
        "item_daily_features.play_duration": "milliseconds",
    }
    if units.get("raw_units") != expected_raw_units:
        raise KuaiRecContractError("raw duration units drifted")
    if units.get("milliseconds_per_second") != MILLISECONDS_PER_SECOND:
        raise KuaiRecContractError("duration divisor drifted")
    _require_exact_bool(units, "positivity_filters_run_on_raw_milliseconds", True)
    _require_exact_bool(units, "conversion_changes_filter_split_or_order", False)

    ordered = payload.get("ordered_fields")
    if not isinstance(ordered, Mapping):
        raise KuaiRecContractError("ordered_fields must be a mapping")
    exact_orders = {
        "C": ORDERED_C_FIELDS,
        "P": ORDERED_P_FIELDS,
        "V": ORDERED_V_FIELDS,
        "E": ORDERED_E_FIELDS,
        "numeric_interaction_operands": ORDERED_NUMERIC_OPERANDS,
    }
    for key, expected in exact_orders.items():
        if ordered.get(key) != list(expected):
            raise KuaiRecContractError(
                f"ordered_fields.{key} differs from frozen order"
            )

    all_model_fields = set().union(*exact_orders.values())
    overlap = MODEL_FORBIDDEN_RAW_FIELDS & all_model_fields
    if overlap:
        raise KuaiRecContractError(
            f"outcome/audit fields entered model schema: {overlap}"
        )
    if len(all_model_fields) != sum(
        len(exact_orders[key]) for key in ("C", "P", "V", "E")
    ):
        raise KuaiRecContractError("C/P/V/E fields must be pairwise disjoint")

    mask = payload.get("denominator_mask")
    if not isinstance(mask, Mapping):
        raise KuaiRecContractError("denominator_mask must be a mapping")
    if mask.get("eligible_fields") != list(DENOMINATOR_ELIGIBLE_FIELDS):
        raise KuaiRecContractError("denominator eligible fields drifted")
    if mask.get("ordered_boolean_mask") != list(ORDERED_DENOMINATOR_MASK):
        raise KuaiRecContractError("ordered denominator mask drifted")

    hashes = payload.get("schema_hashes")
    if not isinstance(hashes, Mapping):
        raise KuaiRecContractError("schema_hashes must be a mapping")
    expected_hashes = {
        "C": ordered_schema_sha256(ORDERED_C_FIELDS),
        "P": ordered_schema_sha256(ORDERED_P_FIELDS),
        "V": ordered_schema_sha256(ORDERED_V_FIELDS),
        "E": ordered_schema_sha256(ORDERED_E_FIELDS),
        "numeric_interaction_operands": ordered_schema_sha256(ORDERED_NUMERIC_OPERANDS),
        "denominator_mask": denominator_mask_sha256(ORDERED_DENOMINATOR_MASK),
    }
    for key, expected in expected_hashes.items():
        if hashes.get(key) != expected:
            raise KuaiRecContractError(f"schema_hashes.{key} drifted")

    filtering = payload.get("filtering")
    if not isinstance(filtering, Mapping):
        raise KuaiRecContractError("filtering must be a mapping")
    if filtering.get("ordered_predicates") != [
        "static_video_type_is_not_AD",
        "current_play_duration_gt_0",
        "current_video_duration_gt_0",
    ]:
        raise KuaiRecContractError("HEGM-comparable filtering drifted")
    if filtering.get("duration_predicate_units") != "raw_milliseconds":
        raise KuaiRecContractError("duration filter units drifted")
    _require_exact_bool(filtering, "watch_ratio_read", False)
    _require_exact_bool(filtering, "positive_watch_estimand_disclosed", True)

    split = payload.get("split")
    if not isinstance(split, Mapping):
        raise KuaiRecContractError("split must be a mapping")
    expected_split = {
        "sort_keys": ["timestamp", "user_id", "video_id", "source_row_index0"],
        "train_fraction": 0.8,
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        "rounding": "train=floor(0.8*N); validation=floor(0.1*N); test=remainder",
        "mapping_fit_split": "train_only",
        "validation_test_cold_user_item_policy": "inner_drop_and_report_counts",
    }
    for key, expected in expected_split.items():
        if split.get(key) != expected:
            raise KuaiRecContractError(f"split.{key} drifted")

    history = payload.get("daily_history")
    if not isinstance(history, Mapping):
        raise KuaiRecContractError("daily_history must be a mapping")
    if history.get("availability_rule") != "daily_date_strictly_less_than_event_date":
        raise KuaiRecContractError("daily availability must remain date < d")
    _require_exact_bool(history, "same_day_allowed", False)
    _require_exact_bool(history, "unique_video_shortcut_allowed", False)
    if history.get("play_duration_output_name") != "hist_play_duration_sum":
        raise KuaiRecContractError("historical play-duration name drifted")
    expected_history_units = {
        "play_duration_raw_units": "milliseconds",
        "play_duration_canonical_units": "seconds",
        "play_duration_divisor": MILLISECONDS_PER_SECOND,
    }
    for key, expected in expected_history_units.items():
        if history.get(key) != expected:
            raise KuaiRecContractError(f"daily_history.{key} drifted")
    _require_exact_bool(history, "play_duration_conversion_before_P_and_E", True)

    label = payload.get("label")
    if not isinstance(label, Mapping):
        raise KuaiRecContractError("label must be a mapping")
    if label.get("source") != "big_matrix.current_play_duration":
        raise KuaiRecContractError("label source drifted")
    if (
        label.get("transform")
        != "clip(current_play_duration_ms/1000,0,c_seconds)/c_seconds"
    ):
        raise KuaiRecContractError("label transform drifted")
    if label.get("c_fit") != (
        "filtered_train_video_duration_ms_div_1000_q99_linear_r7"
    ):
        raise KuaiRecContractError("c must remain a train-only video-duration q99")
    _require_exact_bool(label, "watch_ratio_as_input", False)
    if label.get("raw_source_field") != "big_matrix.play_duration":
        raise KuaiRecContractError("raw label source drifted")
    if label.get("raw_source_units") != "milliseconds":
        raise KuaiRecContractError("raw label units drifted")
    if label.get("sealed_seconds_conversion") != "raw_play_duration / 1000.0":
        raise KuaiRecContractError("sealed label conversion drifted")
    _require_exact_bool(label, "current_play_duration_as_input", False)

    routing = payload.get("architecture_routing")
    if not isinstance(routing, Mapping):
        raise KuaiRecContractError("architecture_routing must be a mapping")
    if routing.get("dcdn_core_operands") != "P_plus_V_only":
        raise KuaiRecContractError("DCDN operand routing drifted")
    if routing.get("C") != "bypass_reciprocal_and_join_common_encoder":
        raise KuaiRecContractError("C must bypass DCDN reciprocal")
    if routing.get("E") != "DCNv2_C_plus_P_plus_E_arm_only":
        raise KuaiRecContractError("E routing drifted")


def load_feature_contract(path: str | Path) -> dict[str, Any]:
    """Verify sidecar before strict YAML parsing and semantic validation."""

    contract = Path(path)
    sidecar = contract.with_name(f"{contract.name}.sha256")
    data = _read_bounded_regular(contract, limit=MAX_CONTRACT_BYTES, label="contract")
    sidecar_data = _read_bounded_regular(
        sidecar, limit=MAX_SIDECAR_BYTES, label="contract sidecar"
    )
    try:
        sidecar_text = sidecar_data.decode("ascii")
    except UnicodeDecodeError as error:
        raise KuaiRecContractError("contract sidecar must be ASCII") from error
    expected = f"{sha256_bytes(data)}  {contract.name}\n"
    if sidecar_text != expected:
        raise KuaiRecContractError("contract sidecar mismatch")
    try:
        payload = yaml.load(data, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise KuaiRecContractError(f"invalid or duplicate-key YAML: {error}") from error
    if not isinstance(payload, Mapping):
        raise KuaiRecContractError("contract root must be a mapping")
    result = dict(payload)
    validate_feature_contract(result)
    return result


def verify_source_manifest_reference(
    payload: Mapping[str, Any], *, repository_root: str | Path
) -> str:
    source = payload.get("source_manifest")
    if not isinstance(source, Mapping) or source.get("path") != SOURCE_MANIFEST_PATH:
        raise KuaiRecContractError("source manifest reference drifted")
    path = Path(repository_root) / SOURCE_MANIFEST_PATH
    data = _read_bounded_regular(
        path, limit=MAX_CONTRACT_BYTES, label="source manifest"
    )
    digest = sha256_bytes(data)
    if digest != SOURCE_MANIFEST_SHA256 or source.get("sha256") != digest:
        raise KuaiRecContractError("source manifest digest mismatch")
    try:
        manifest = load_source_manifest(path)
    except Exception as error:
        raise KuaiRecContractError(
            f"source manifest validation failed: {error}"
        ) from error
    if manifest.get("dataset_id") != SOURCE_DATASET_ID:
        raise KuaiRecContractError("source dataset_id mismatch")
    if manifest.get("test_unlocked") is not False:
        raise KuaiRecContractError("source manifest unexpectedly unlocks test")
    return digest
