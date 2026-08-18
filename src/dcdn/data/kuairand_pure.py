"""Fail-closed KuaiRand-Pure feature and row-identity contract.

This module contains only pre-metric protocol machinery.  It does not open raw
data, materialize features, or expose validation/test labels.  The companion
audit command is deliberately limited to source row offsets plus ``date``,
``time_ms``, and ``is_rand``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO

import yaml

DATASET_ID = "kuairand_pure"
FEATURE_SCHEMA_VERSION = 1
MAX_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_SIDECAR_BYTES = 256
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

EVENT_COLUMNS = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
    "tab",
)

STANDARD_HISTORY_FILE = "log_standard_4_08_to_4_21_pure.csv"
STANDARD_MAIN_FILE = "log_standard_4_22_to_5_08_pure.csv"
RANDOM_MAIN_FILE = "log_random_4_22_to_5_08_pure.csv"
EVENT_FILES = (STANDARD_HISTORY_FILE, STANDARD_MAIN_FILE, RANDOM_MAIN_FILE)
FILE_POLICY = {
    STANDARD_HISTORY_FILE: "standard",
    STANDARD_MAIN_FILE: "standard",
    RANDOM_MAIN_FILE: "randomized",
}
POLICY_PRIORITY = {"standard": 0, "randomized": 1}

GROUPS = ("user", "item", "author", "tab")
WINDOWS = ("1d", "7d", "all")
P_STATISTICS = (
    "exposure_count",
    "long_view_count",
    "like_count",
    "forward_count",
    "hate_count",
    "follow_count",
    "play_time_ms_sum",
    "duration_ms_sum",
    "dwell_time_ms_sum",
)
RATE_STATISTICS = (
    "long_view_per_exposure",
    "like_per_exposure",
    "forward_per_exposure",
    "hate_per_exposure",
    "follow_per_exposure",
    "mean_play_time_seconds",
    "play_over_duration",
    "mean_dwell_time_seconds",
)
SHORT_LONG_STATISTICS = (
    "exposure_1d_over_7d",
    "exposure_7d_over_all",
    "long_view_1d_over_7d",
    "long_view_7d_over_all",
)

ORDERED_C_FIELDS = (
    "c_user_id",
    "c_video_id",
    "c_author_id",
    "c_tab",
    "c_video_type",
    "c_upload_type",
    "c_music_id",
    "c_music_type",
)
ORDERED_P_FIELDS = tuple(
    f"p_{group}_{window}_{statistic}"
    for group in GROUPS
    for window in WINDOWS
    for statistic in P_STATISTICS
)
ORDERED_V_FIELDS = tuple(
    f"v_{group}_{window}_{kind}_valid"
    for group in GROUPS
    for window in WINDOWS
    for kind in ("exposure", "duration")
)
ORDERED_R_FIELDS = tuple(
    [
        f"r_{group}_{window}_{statistic}"
        for group in GROUPS
        for window in WINDOWS
        for statistic in RATE_STATISTICS
    ]
    + [
        f"r_{group}_{statistic}"
        for group in GROUPS
        for statistic in SHORT_LONG_STATISTICS
    ]
)
ORDERED_NUMERIC_OPERANDS = ORDERED_P_FIELDS + ORDERED_V_FIELDS
DENOMINATOR_ELIGIBLE_FIELDS = tuple(
    f"p_{group}_{window}_{statistic}"
    for group in GROUPS
    for window in WINDOWS
    for statistic in ("exposure_count", "duration_ms_sum")
)
ORDERED_DENOMINATOR_MASK = tuple(
    field in set(DENOMINATOR_ELIGIBLE_FIELDS) for field in ORDERED_NUMERIC_OPERANDS
)

MODEL_FORBIDDEN_RAW_FIELDS = frozenset(
    {
        "date",
        "hourmin",
        "time_ms",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "play_time_ms",
        "duration_ms",
        "profile_stay_time",
        "comment_stay_time",
        "is_profile_enter",
        "is_rand",
    }
)

EXPECTED_SPLITS = {
    "history_burn_in": {
        "policy": "standard",
        "start": "2022-04-09",
        "end": "2022-04-21",
        "role": "state_update_only",
    },
    "train": {
        "policy": "standard",
        "start": "2022-04-22",
        "end": "2022-05-02",
        "role": "label_and_state_update",
    },
    "validation": {
        "policy": "randomized",
        "start": "2022-05-03",
        "end": "2022-05-05",
        "role": "validation_label_and_state_update",
    },
    "headline_test": {
        "policy": "randomized",
        "start": "2022-05-06",
        "end": "2022-05-08",
        "role": "locked_test_label_and_state_update",
    },
    "in_policy_test": {
        "policy": "standard",
        "start": "2022-05-06",
        "end": "2022-05-08",
        "role": "locked_secondary_test_label_and_state_update",
    },
}

EXPECTED_TOTAL_SOURCE_ROWS = 2_622_668
EXPECTED_EMBARGO_CUTOFFS = {
    "history_to_train": 1_650_552_767_170,
    "train_to_validation": 1_651_503_652_252,
    "validation_to_test": 1_651_762_945_505,
}
EMBARGO_CUTOFF_NAME_BY_ROLE = {
    "history_burn_in": "history_to_train",
    "train": "train_to_validation",
    "validation": "validation_to_test",
    "validation_state_update_only": "validation_to_test",
}
EXPECTED_PRE_EMBARGO_ROLE_COUNTS = {
    "history_burn_in": 1_141_112,
    "headline_test": 322_152,
    "in_policy_test": 43_268,
    "pre_validation_randomized_excluded": 548_567,
    "train": 202_296,
    "validation": 315_340,
    "validation_state_update_only": 49_933,
}
EXPECTED_EMBARGO_EXCLUDED_COUNTS = {
    "history_burn_in": 1_204,
    "train": 976,
    "validation": 3_850,
    "validation_state_update_only": 609,
}
EXPECTED_POST_EMBARGO_ROLE_COUNTS = {
    "history_burn_in": 1_139_908,
    "headline_test": 322_152,
    "in_policy_test": 43_268,
    "pre_validation_randomized_excluded": 548_567,
    "train": 201_320,
    "validation": 311_490,
    "validation_state_update_only": 49_324,
}


class KuaiRandContractError(RuntimeError):
    """Raised when the frozen KuaiRand contract cannot be trusted."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe loader that rejects duplicate mapping keys."""


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
    """Hash an ordered schema with a documented canonical JSON encoding."""

    encoded = json.dumps(list(fields), ensure_ascii=True, separators=(",", ":")).encode(
        "ascii"
    )
    return sha256_bytes(encoded)


def denominator_mask_sha256(mask: Sequence[bool]) -> str:
    encoded = json.dumps(list(mask), ensure_ascii=True, separators=(",", ":")).encode(
        "ascii"
    )
    return sha256_bytes(encoded)


def _require_exact_bool(payload: Mapping[str, Any], key: str, value: bool) -> None:
    if type(payload.get(key)) is not bool or payload[key] is not value:
        raise KuaiRandContractError(f"{key} must be exactly {value!r}")


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise KuaiRandContractError(f"{key} must be a mapping")
    return value


def _require_exact_list(
    payload: Mapping[str, Any], key: str, expected: Sequence[Any]
) -> None:
    value = payload.get(key)
    if not isinstance(value, list) or value != list(expected):
        raise KuaiRandContractError(f"{key} does not match the frozen order")


def validate_feature_contract(payload: Mapping[str, Any]) -> None:
    """Validate exact pre-metric KuaiRand feature and split semantics."""

    if payload.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise KuaiRandContractError("schema_version must equal 1")
    if payload.get("dataset_id") != DATASET_ID:
        raise KuaiRandContractError(f"dataset_id must equal {DATASET_ID!r}")
    if payload.get("artifact_role") != "pre_metric_feature_contract":
        raise KuaiRandContractError(
            "artifact_role must equal pre_metric_feature_contract"
        )
    for key, expected in {
        "test_unlocked": False,
        "test_labels_opened": False,
        "model_metrics_read": False,
        "final_test_view_materialized": False,
        "feature_values_scanned": False,
        "phase1_gate_passed": True,
    }.items():
        _require_exact_bool(payload, key, expected)

    decision = _require_mapping(payload, "approved_temporal_policy")
    if decision.get("status") != "owner_approved_and_applied":
        raise KuaiRandContractError("temporal policy approval is not frozen")
    if decision.get("owner_approval_recorded_date") != "2026-08-18":
        raise KuaiRandContractError("temporal policy approval date changed")
    _require_exact_bool(decision, "applied", True)
    _require_exact_bool(decision, "train_validation_test_dates_changed", False)
    burn_in = _require_mapping(decision, "actual_burn_in")
    if dict(burn_in) != {
        "policy": "standard",
        "original_requested_start": "2022-04-08",
        "actual_available_start": "2022-04-09",
        "end": "2022-04-21",
        "april_08_rows_available": False,
        "synthetic_or_imputed_rows_forbidden": True,
    }:
        raise KuaiRandContractError("actual burn-in policy changed")
    embargo = _require_mapping(decision, "embargo")
    if embargo.get("role") != "owner_approved_boundary_embargo":
        raise KuaiRandContractError("approved embargo role changed")
    _require_exact_bool(embargo, "later_role_rows_retained", True)
    _require_exact_bool(embargo, "labels_or_model_features_used", False)
    if embargo.get("cutoff_inputs") != (
        "date_time_ms_policy_source_file_and_source_row_index0_only"
    ):
        raise KuaiRandContractError("embargo cutoff inputs changed")
    if embargo.get("excluded_rows_action") != (
        "exclude_from_labels_state_updates_and_every_materialized_view"
    ):
        raise KuaiRandContractError("embargo exclusion action changed")
    boundaries = _require_mapping(embargo, "boundaries")
    if dict(boundaries) != {
        "history_to_train": "history_burn_in_before_min_train_time_ms",
        "train_to_validation": "train_before_min_validation_union_time_ms",
        "validation_to_test": "validation_union_before_min_test_union_time_ms",
    }:
        raise KuaiRandContractError("approved embargo boundaries changed")
    if dict(_require_mapping(embargo, "cutoffs_time_ms")) != (EXPECTED_EMBARGO_CUTOFFS):
        raise KuaiRandContractError("approved embargo cutoffs changed")
    if dict(_require_mapping(embargo, "pre_embargo_rows_by_role")) != (
        EXPECTED_PRE_EMBARGO_ROLE_COUNTS
    ):
        raise KuaiRandContractError("pre-embargo evidence changed")
    if dict(_require_mapping(embargo, "excluded_rows_by_role")) != (
        EXPECTED_EMBARGO_EXCLUDED_COUNTS
    ):
        raise KuaiRandContractError("embargo exclusion evidence changed")
    if dict(_require_mapping(embargo, "post_embargo_rows_by_role")) != (
        EXPECTED_POST_EMBARGO_ROLE_COUNTS
    ):
        raise KuaiRandContractError("effective role evidence changed")
    if embargo.get("total_excluded_rows") != sum(
        EXPECTED_EMBARGO_EXCLUDED_COUNTS.values()
    ):
        raise KuaiRandContractError("total embargo exclusion count changed")

    source = _require_mapping(payload, "source_manifest")
    if source.get("path") != "data/manifests/kuairand_pure_source_v1.yaml":
        raise KuaiRandContractError("unexpected source manifest path")
    digest = source.get("sha256")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise KuaiRandContractError("source_manifest.sha256 is invalid")

    exclusions = _require_mapping(payload, "strict_exclusions")
    if exclusions.get("video_features_statistic_pure.csv") != "forbidden":
        raise KuaiRandContractError("monthly video statistics must be forbidden")
    if exclusions.get("is_rand") != "audit_and_stratification_only":
        raise KuaiRandContractError("is_rand must be audit-only")
    _require_exact_list(
        exclusions,
        "current_row_outcomes",
        sorted(MODEL_FORBIDDEN_RAW_FIELDS - {"date", "hourmin", "time_ms", "is_rand"}),
    )

    ordered = _require_mapping(payload, "ordered_fields")
    _require_exact_list(ordered, "C", ORDERED_C_FIELDS)
    _require_exact_list(ordered, "P", ORDERED_P_FIELDS)
    _require_exact_list(ordered, "V", ORDERED_V_FIELDS)
    _require_exact_list(ordered, "R", ORDERED_R_FIELDS)
    combined = list(ORDERED_NUMERIC_OPERANDS)
    _require_exact_list(ordered, "numeric_interaction_operands", combined)

    all_fields = (
        list(ORDERED_C_FIELDS)
        + list(ORDERED_P_FIELDS)
        + list(ORDERED_V_FIELDS)
        + list(ORDERED_R_FIELDS)
    )
    if len(all_fields) != len(set(all_fields)):
        raise KuaiRandContractError("model field names must be globally unique")
    if "is_rand" in all_fields:
        raise KuaiRandContractError("is_rand cannot be a model field")

    groups = _require_mapping(payload, "rolling_aggregates")
    _require_exact_list(groups, "group_order", GROUPS)
    _require_exact_list(groups, "window_order", WINDOWS)
    _require_exact_list(groups, "statistic_order", P_STATISTICS)
    windows = _require_mapping(groups, "windows")
    if windows != {
        "1d": {"lower_inclusive_ms": -86_400_000, "upper": "time_ms<t"},
        "7d": {"lower_inclusive_ms": -604_800_000, "upper": "time_ms<t"},
        "all": {"lower": "burn_in_start", "upper": "time_ms<t"},
    }:
        raise KuaiRandContractError("rolling windows differ from the frozen policy")

    splits = _require_mapping(payload, "temporal_splits")
    for name, expected in EXPECTED_SPLITS.items():
        if splits.get(name) != expected:
            raise KuaiRandContractError(f"temporal_splits.{name} changed")

    state = _require_mapping(payload, "rolling_state")
    required_state = {
        "feature_cutoff": "strict_time_ms_less_than_target",
        "equal_timestamp_rule": "score_complete_group_then_update_complete_group",
        "update_order": ["policy_priority", "source_file", "source_row_index0"],
        "policy_priority": {"standard": 0, "randomized": 1},
        "initial_state": "standard_only_through_2022-05-02",
        "evaluation_updates": "standard_and_randomized_union",
        "embargo_excluded_rows_never_update_state": True,
        "weights_frozen_after_train": True,
    }
    for key, expected in required_state.items():
        if state.get(key) != expected:
            raise KuaiRandContractError(f"rolling_state.{key} changed")

    dedup = _require_mapping(payload, "row_identity_and_dedup")
    if dedup.get("dedup_key") != "full_row_fingerprint_sha256":
        raise KuaiRandContractError("dedup must use the full-row fingerprint")
    if dedup.get("event_key_dedup_forbidden") is not True:
        raise KuaiRandContractError("event-key dedup must remain forbidden")
    _require_exact_list(dedup, "canonical_column_order", EVENT_COLUMNS)

    mask = _require_mapping(payload, "denominator_mask")
    if mask.get("aligned_to") != "ordered_fields.numeric_interaction_operands":
        raise KuaiRandContractError("denominator mask alignment changed")
    _require_exact_list(mask, "eligible_fields", DENOMINATOR_ELIGIBLE_FIELDS)
    if mask.get("invalid_history_behavior") != "zero_reciprocal_carrier":
        raise KuaiRandContractError("invalid denominator behavior changed")
    if mask.get("normalized_positive_floor") != 0.01:
        raise KuaiRandContractError("normalized denominator floor must be 0.01")

    label = _require_mapping(payload, "label")
    if label != {
        "name": "long_view",
        "source_column": "long_view",
        "type": "binary",
        "allowed_values": [0, 1],
        "current_row_feature_use": "forbidden",
        "state_update_timing": "after_complete_equal_timestamp_group_is_scored",
    }:
        raise KuaiRandContractError("label contract changed")

    schema_hashes = _require_mapping(payload, "schema_hashes")
    expected_hashes = {
        "C": ordered_schema_sha256(ORDERED_C_FIELDS),
        "P": ordered_schema_sha256(ORDERED_P_FIELDS),
        "V": ordered_schema_sha256(ORDERED_V_FIELDS),
        "R": ordered_schema_sha256(ORDERED_R_FIELDS),
        "C_plus_P_plus_V": ordered_schema_sha256(
            ORDERED_C_FIELDS + ORDERED_P_FIELDS + ORDERED_V_FIELDS
        ),
        "C_plus_P_plus_V_plus_R": ordered_schema_sha256(
            ORDERED_C_FIELDS + ORDERED_P_FIELDS + ORDERED_V_FIELDS + ORDERED_R_FIELDS
        ),
        "numeric_interaction_operands": ordered_schema_sha256(ORDERED_NUMERIC_OPERANDS),
        "denominator_mask": denominator_mask_sha256(ORDERED_DENOMINATOR_MASK),
    }
    if dict(schema_hashes) != expected_hashes:
        raise KuaiRandContractError("schema_hashes do not match frozen field order")


def _open_private_regular(path: Path) -> BinaryIO:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise KuaiRandContractError(f"cannot safely open {path}") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise KuaiRandContractError(
            f"expected a private regular non-hardlinked file: {path}"
        )
    return os.fdopen(descriptor, "rb")


def load_feature_contract(path: str | Path) -> dict[str, Any]:
    """Verify canonical sidecar bytes, parse unique-key YAML, and validate."""

    source = Path(path)
    sidecar = source.with_name(f"{source.name}.sha256")
    for candidate, maximum, label in (
        (source, MAX_CONTRACT_BYTES, "contract"),
        (sidecar, MAX_SIDECAR_BYTES, "sidecar"),
    ):
        if candidate.is_symlink() or not candidate.is_file():
            raise KuaiRandContractError(f"{label} must be a regular file")
        if candidate.stat().st_nlink != 1:
            raise KuaiRandContractError(f"{label} must not be hard-linked")
        if candidate.stat().st_size > maximum:
            raise KuaiRandContractError(f"{label} exceeds the size limit")
    with _open_private_regular(source) as handle:
        data = handle.read(MAX_CONTRACT_BYTES + 1)
    digest = sha256_bytes(data)
    expected = f"{digest}  {source.name}\n".encode("ascii")
    with _open_private_regular(sidecar) as handle:
        sidecar_data = handle.read(MAX_SIDECAR_BYTES + 1)
    if sidecar_data != expected:
        raise KuaiRandContractError("feature contract sidecar mismatch")
    try:
        payload = yaml.load(data, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise KuaiRandContractError(
            f"invalid feature contract YAML: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise KuaiRandContractError("feature contract root must be a mapping")
    validate_feature_contract(payload)
    return payload


def verify_source_manifest_reference(
    contract: Mapping[str, Any], *, repository_root: str | Path
) -> str:
    """Verify that the contract references the exact local source manifest."""

    source = _require_mapping(contract, "source_manifest")
    relative = source["path"]
    if not isinstance(relative, str):
        raise KuaiRandContractError("source manifest path must be a string")
    root = Path(repository_root).resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise KuaiRandContractError(
            "source manifest escapes repository root"
        ) from error
    data = candidate.read_bytes()
    digest = sha256_bytes(data)
    if digest != source["sha256"]:
        raise KuaiRandContractError("source manifest digest mismatch")
    sidecar = candidate.with_name(f"{candidate.name}.sha256")
    if sidecar.read_bytes() != f"{digest}  {candidate.name}\n".encode("ascii"):
        raise KuaiRandContractError("source manifest sidecar mismatch")
    return digest


def _canonical_field_bytes(value: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError("raw row fields must be strings")
    if "\x00" in value:
        raise KuaiRandContractError("NUL is forbidden in a raw row field")
    return unicodedata.normalize("NFC", value).encode("utf-8")


def canonical_full_row_bytes(
    row: Mapping[str, str] | Sequence[str],
) -> bytes:
    """Encode every released event field without numeric reinterpretation.

    CSV quoting and line endings are transport details.  Field text is kept
    byte-significant after UTF-8 NFC normalization; whitespace and numeric
    spellings are deliberately not stripped or coerced.
    """

    if isinstance(row, Mapping):
        if set(row) != set(EVENT_COLUMNS):
            raise KuaiRandContractError("raw row keys must match EVENT_COLUMNS")
        values = [row[column] for column in EVENT_COLUMNS]
    else:
        if isinstance(row, str | bytes) or len(row) != len(EVENT_COLUMNS):
            raise KuaiRandContractError("raw row width must match EVENT_COLUMNS")
        values = list(row)
    output = bytearray(b"dcdn:kuairand-pure:full-row:v1\x00")
    for value in values:
        encoded = _canonical_field_bytes(value)
        output.extend(len(encoded).to_bytes(8, byteorder="big", signed=False))
        output.extend(encoded)
    return bytes(output)


def full_row_fingerprint(row: Mapping[str, str] | Sequence[str]) -> str:
    """Return the sole allowed deduplication key for an event row."""

    return sha256_bytes(canonical_full_row_bytes(row))


def stable_row_id(
    *,
    fingerprint: str,
    source_file: str,
    source_row_index0: int,
) -> str:
    """Identify the deterministic canonical occurrence retained after dedup."""

    if SHA256_RE.fullmatch(fingerprint) is None:
        raise KuaiRandContractError("fingerprint must be a lowercase SHA-256")
    if source_file not in EVENT_FILES or "/" in source_file or "\x00" in source_file:
        raise KuaiRandContractError("unknown or unsafe source_file")
    if type(source_row_index0) is not int or source_row_index0 < 0:
        raise KuaiRandContractError("source_row_index0 must be non-negative")
    payload = (
        b"dcdn:kuairand-pure:stable-row:v1\x00"
        + fingerprint.encode("ascii")
        + b"\x00"
        + source_file.encode("ascii")
        + b"\x00"
        + str(source_row_index0).encode("ascii")
    )
    return f"krp1_{sha256_bytes(payload)}"


@dataclass(frozen=True)
class RowOrderingKey:
    """Audit-only row metadata used for prequential grouping and tie order."""

    time_ms: int
    policy: str
    source_file: str
    source_row_index0: int

    def __post_init__(self) -> None:
        if type(self.time_ms) is not int or self.time_ms < 0:
            raise KuaiRandContractError("time_ms must be a non-negative integer")
        if self.policy not in POLICY_PRIORITY:
            raise KuaiRandContractError("unknown policy")
        if FILE_POLICY.get(self.source_file) != self.policy:
            raise KuaiRandContractError("source_file and policy disagree")
        if type(self.source_row_index0) is not int or self.source_row_index0 < 0:
            raise KuaiRandContractError("source_row_index0 must be non-negative")

    @property
    def update_key(self) -> tuple[int, str, int]:
        return (
            POLICY_PRIORITY[self.policy],
            self.source_file,
            self.source_row_index0,
        )


def iter_prequential_groups(
    rows: Iterable[RowOrderingKey],
) -> Iterable[tuple[int, tuple[RowOrderingKey, ...]]]:
    """Yield global timestamp groups with deterministic post-score order."""

    ordered = sorted(
        rows,
        key=lambda row: (row.time_ms, *row.update_key),
    )
    index = 0
    while index < len(ordered):
        time_ms = ordered[index].time_ms
        stop = index + 1
        while stop < len(ordered) and ordered[stop].time_ms == time_ms:
            stop += 1
        yield time_ms, tuple(ordered[index:stop])
        index = stop


def _parse_iso_date(value: str) -> date:
    if DATE_RE.fullmatch(value) is None:
        raise KuaiRandContractError(f"invalid ISO date: {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise KuaiRandContractError(f"invalid ISO date: {value!r}") from error


def classify_row_role(*, iso_date: str, policy: str) -> str:
    """Classify permitted row metadata without opening any label."""

    current = _parse_iso_date(iso_date)
    if policy not in POLICY_PRIORITY:
        raise KuaiRandContractError("unknown policy")
    for name, rule in EXPECTED_SPLITS.items():
        if policy != rule["policy"]:
            continue
        if _parse_iso_date(rule["start"]) <= current <= _parse_iso_date(rule["end"]):
            return name
    if policy == "standard" and date(2022, 5, 3) <= current <= date(2022, 5, 5):
        return "validation_state_update_only"
    if policy == "randomized" and date(2022, 4, 22) <= current <= date(2022, 5, 2):
        return "pre_validation_randomized_excluded"
    return "outside_frozen_protocol"


def is_row_embargo_excluded(
    *, role: str, time_ms: int, cutoffs: Mapping[str, int]
) -> bool:
    """Apply the approved earlier-role boundary embargo to audit metadata."""

    if type(time_ms) is not int or time_ms < 0:
        raise KuaiRandContractError("time_ms must be a non-negative integer")
    cutoff_name = EMBARGO_CUTOFF_NAME_BY_ROLE.get(role)
    if cutoff_name is None:
        return False
    cutoff = cutoffs.get(cutoff_name)
    if type(cutoff) is not int or cutoff < 0:
        raise KuaiRandContractError(f"missing valid embargo cutoff: {cutoff_name}")
    return time_ms >= cutoff
