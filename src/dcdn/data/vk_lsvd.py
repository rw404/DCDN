"""Fail-closed VK-LSVD feature, split, and test-seal contract.

The mandatory public experiment uses history weeks 0--19, training weeks
20--24, validation week 25, and a sealed week-26 test.  This module contains
only pre-metric protocol machinery and deterministic row-level feature
algebra.  It never opens a Parquet file; the companion audit command owns the
strictly allowlisted schema/provenance reads.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from dcdn.source_manifest import load_source_manifest

DATASET_ID = "vk_lsvd_up0.01_ir0.01"
FEATURE_SCHEMA_VERSION = 1
SOURCE_MANIFEST_PATH = "data/manifests/vk_lsvd_source_v1.yaml"
SOURCE_DATASET_ID = "deepvk/VK-LSVD"
SOURCE_REVISION = "084a18386e149a454dd6dbe0123478e75c6b09ce"
HEGM_REVISION = "d16c097d57f867f438f906949d9cc1efd0f6ddc4"
HEGM_PREPROCESS_SHA256 = (
    "5e7137820f8d923692afc0055c60069faa27678d98f8fa29054b4777b53be92e"
)
HEGM_LICENSE_SHA256 = "eb74dde2903b7a84802e5907b35717441530b25738fc00c0b6639bb8287878c3"

HISTORY_WEEKS = tuple(range(0, 20))
TRAIN_WEEKS = tuple(range(20, 25))
VALIDATION_WEEK = 25
SEALED_TEST_WEEK = 26
PHASE1_SEMANTIC_WEEKS = HISTORY_WEEKS + TRAIN_WEEKS + (VALIDATION_WEEK,)

ALPHA_SMOOTH = 5.0
BETA_SMOOTH = 100.0
RECIPROCAL_EPSILON = 0.01
CONTEXT_EMBEDDING_DIM = 16
LABEL_DURATION_Q99_SECONDS = 157.0

MAX_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_SIDECAR_BYTES = 256
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

INTERACTION_SCHEMA = (
    "user_id:uint32",
    "item_id:uint32",
    "place:uint8",
    "platform:uint8",
    "agent:uint8",
    "timespent:uint8",
    "like:bool",
    "dislike:bool",
    "share:bool",
    "bookmark:bool",
    "click_on_author:bool",
    "open_comments:bool",
)
USERS_METADATA_SCHEMA = (
    "user_id:uint32",
    "age:uint8",
    "gender:uint8",
    "geo:uint8",
    "train_interactions_rank:uint32",
)
ITEMS_METADATA_SCHEMA = (
    "item_id:uint32",
    "author_id:uint32",
    "duration:uint8",
    "train_interactions_rank:uint32",
)

ORDERED_C_FIELDS = ("age", "gender", "geo", "place", "platform", "agent")

USER_RATE_SOURCES = (
    ("like", "like"),
    ("share", "share"),
    ("bookmark", "bookmark"),
    ("comments", "open_comments"),
    ("dislike", "dislike"),
    ("heartbeat15", "survival_15"),
    ("heartbeat30", "survival_30"),
)
ITEM_RATE_SOURCES = (
    USER_RATE_SOURCES[:5]
    + (("click_on_author", "click_on_author"),)
    + USER_RATE_SOURCES[5:]
)
IALS_SIGNALS = ("viewtime", "feedback", "survival")


def _group_p_fields(group: str, rates: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    return (
        f"p_{group}_show_count",
        f"p_{group}_show_count_plus_beta",
        f"p_{group}_timespent_sum_seconds",
        *(f"p_{group}_{name}_count_plus_alpha" for name, _ in rates),
    )


def _group_e_fields(group: str, rates: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    return (
        f"e_{group}_avg_timespent_seconds",
        *(f"e_{group}_{name}_rate" for name, _ in rates),
    )


ORDERED_P0_FIELDS = (
    "p_duration_seconds",
    *_group_p_fields("user", USER_RATE_SOURCES),
    *_group_p_fields("item", ITEM_RATE_SOURCES),
    *(
        field
        for signal in IALS_SIGNALS
        for field in (
            f"p_ials_{signal}_dot",
            f"p_ials_{signal}_user_norm",
            f"p_ials_{signal}_item_norm",
        )
    ),
)
ORDERED_E_FIELDS = (
    *_group_e_fields("user", USER_RATE_SOURCES),
    *_group_e_fields("item", ITEM_RATE_SOURCES),
    *(f"e_ials_{signal}_cosine" for signal in IALS_SIGNALS),
)

DENOMINATOR_ELIGIBLE_FIELDS = (
    "p_duration_seconds",
    "p_user_show_count",
    "p_user_show_count_plus_beta",
    "p_item_show_count",
    "p_item_show_count_plus_beta",
    *(
        field
        for signal in IALS_SIGNALS
        for field in (
            f"p_ials_{signal}_user_norm",
            f"p_ials_{signal}_item_norm",
        )
    ),
)


def floor_indicator_name(denominator_field: str) -> str:
    if denominator_field not in DENOMINATOR_ELIGIBLE_FIELDS:
        raise ValueError("floor indicators require a denominator-eligible field")
    return f"v_{denominator_field.removeprefix('p_')}_reciprocal_floor_active"


ORDERED_V_FIELDS = (
    "v_user_history_present",
    "v_item_history_present",
    "v_item_metadata_present",
    "v_duration_positive",
    *(
        field
        for signal in IALS_SIGNALS
        for field in (
            f"v_ials_{signal}_user_seen",
            f"v_ials_{signal}_item_seen",
            f"v_ials_{signal}_user_norm_positive",
            f"v_ials_{signal}_item_norm_positive",
        )
    ),
    *(floor_indicator_name(field) for field in DENOMINATOR_ELIGIBLE_FIELDS),
)
ORDERED_NUMERIC_OPERANDS = ORDERED_P0_FIELDS + ORDERED_V_FIELDS
ORDERED_DENOMINATOR_MASK = tuple(
    field in set(DENOMINATOR_ELIGIBLE_FIELDS) for field in ORDERED_NUMERIC_OPERANDS
)

EXPANDED_C_COORDINATES = tuple(
    f"{field}__embedding_{coordinate:02d}"
    for field in ORDERED_C_FIELDS
    for coordinate in range(CONTEXT_EMBEDDING_DIM)
)
ORDERED_C_PLUS_P_FLAT_SCHEMA = EXPANDED_C_COORDINATES + ORDERED_NUMERIC_OPERANDS
ORDERED_C_PLUS_P_PLUS_E_FLAT_SCHEMA = ORDERED_C_PLUS_P_FLAT_SCHEMA + ORDERED_E_FIELDS

MODEL_FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "user_id",
        "item_id",
        "timespent",
        "like",
        "dislike",
        "share",
        "bookmark",
        "click_on_author",
        "open_comments",
        "author_id",
        "train_interactions_rank",
    }
)


def _p_formulas() -> dict[str, str]:
    result = {
        "p_duration_seconds": (
            "items_metadata.duration joined by item_id; missing fallback 0"
        )
    }
    for group, rates in (
        ("user", USER_RATE_SOURCES),
        ("item", ITEM_RATE_SOURCES),
    ):
        result[f"p_{group}_show_count"] = (
            f"count(history rows grouped by {group}_id); missing fallback 0"
        )
        result[f"p_{group}_show_count_plus_beta"] = f"p_{group}_show_count + beta"
        result[f"p_{group}_timespent_sum_seconds"] = (
            f"sum(history.timespent grouped by {group}_id); missing fallback 0"
        )
        for name, source in rates:
            result[f"p_{group}_{name}_count_plus_alpha"] = (
                f"sum(history.{source} grouped by {group}_id) + alpha"
            )
    for signal in IALS_SIGNALS:
        result[f"p_ials_{signal}_dot"] = (
            f"dot(ials.{signal}.user_factor, ials.{signal}.item_factor); "
            "unseen fallback 0"
        )
        result[f"p_ials_{signal}_user_norm"] = (
            f"l2_norm(ials.{signal}.user_factor); unseen fallback 0"
        )
        result[f"p_ials_{signal}_item_norm"] = (
            f"l2_norm(ials.{signal}.item_factor); unseen fallback 0"
        )
    return result


def _e_formulas() -> dict[str, str]:
    result: dict[str, str] = {}
    for group, rates in (
        ("user", USER_RATE_SOURCES),
        ("item", ITEM_RATE_SOURCES),
    ):
        result[f"e_{group}_avg_timespent_seconds"] = (
            f"p_{group}_timespent_sum_seconds / p_{group}_show_count "
            f"when v_{group}_history_present else 0"
        )
        for name, _ in rates:
            result[f"e_{group}_{name}_rate"] = (
                f"p_{group}_{name}_count_plus_alpha / p_{group}_show_count_plus_beta"
            )
    for signal in IALS_SIGNALS:
        result[f"e_ials_{signal}_cosine"] = (
            f"p_ials_{signal}_dot / "
            f"(p_ials_{signal}_user_norm * p_ials_{signal}_item_norm) "
            "when both factors are seen and both norms are positive else 0"
        )
    return result


P0_FORMULAS = _p_formulas()
E_FORMULAS = _e_formulas()


def _v_formulas() -> dict[str, str]:
    result = {
        "v_user_history_present": "p_user_show_count > 0",
        "v_item_history_present": "p_item_show_count > 0",
        "v_item_metadata_present": "item_id matched exactly one metadata row",
        "v_duration_positive": ("v_item_metadata_present and p_duration_seconds > 0"),
    }
    for signal in IALS_SIGNALS:
        result[f"v_ials_{signal}_user_seen"] = (
            f"user_id exists in history-fitted ials.{signal}.user_map"
        )
        result[f"v_ials_{signal}_item_seen"] = (
            f"item_id exists in history-fitted ials.{signal}.item_map"
        )
        result[f"v_ials_{signal}_user_norm_positive"] = (
            f"v_ials_{signal}_user_seen and p_ials_{signal}_user_norm > 0"
        )
        result[f"v_ials_{signal}_item_norm_positive"] = (
            f"v_ials_{signal}_item_seen and p_ials_{signal}_item_norm > 0"
        )
    for field in DENOMINATOR_ELIGIBLE_FIELDS:
        result[floor_indicator_name(field)] = (
            f"denominator_valid({field}) and normalized({field}) < 0.01"
        )
    return result


V_FORMULAS = _v_formulas()

DENOMINATOR_VALIDITY_SOURCES = {
    "p_duration_seconds": "v_item_metadata_present AND v_duration_positive",
    "p_user_show_count": "v_user_history_present",
    "p_user_show_count_plus_beta": "always_true_after_beta_shift",
    "p_item_show_count": "v_item_history_present",
    "p_item_show_count_plus_beta": "always_true_after_beta_shift",
    **{
        f"p_ials_{signal}_{entity}_norm": (
            f"v_ials_{signal}_{entity}_seen AND v_ials_{signal}_{entity}_norm_positive"
        )
        for signal in IALS_SIGNALS
        for entity in ("user", "item")
    },
}


class VKLSVDContractError(RuntimeError):
    """Raised when the frozen VK-LSVD contract cannot be trusted."""


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


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise VKLSVDContractError(f"{key} must be a mapping")
    return value


def _require_exact_bool(payload: Mapping[str, Any], key: str, value: bool) -> None:
    if type(payload.get(key)) is not bool or payload[key] is not value:
        raise VKLSVDContractError(f"{key} must be exactly {value!r}")


def _require_exact_list(
    payload: Mapping[str, Any], key: str, expected: Sequence[Any]
) -> None:
    value = payload.get(key)
    if not isinstance(value, list) or value != list(expected):
        raise VKLSVDContractError(f"{key} does not match the frozen order")


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise VKLSVDContractError(f"{field} must be a lowercase SHA-256")
    return value


def validate_feature_contract(payload: Mapping[str, Any]) -> None:
    """Validate exact pre-metric VK-LSVD feature and split semantics."""

    if payload.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise VKLSVDContractError("schema_version must equal 1")
    if payload.get("dataset_id") != DATASET_ID:
        raise VKLSVDContractError(f"dataset_id must equal {DATASET_ID!r}")
    if payload.get("artifact_role") != "pre_metric_feature_contract":
        raise VKLSVDContractError(
            "artifact_role must equal pre_metric_feature_contract"
        )
    for key, expected in {
        "test_unlocked": False,
        "test_labels_opened": False,
        "validation_model_metrics_read": False,
        "test_metrics_read": False,
        "week26_schema_inspected": False,
        "week26_view_materialized": False,
    }.items():
        _require_exact_bool(payload, key, expected)

    source = _require_mapping(payload, "source_manifest")
    if source.get("path") != SOURCE_MANIFEST_PATH:
        raise VKLSVDContractError("unexpected source manifest path")
    _require_sha256(source.get("sha256"), "source_manifest.sha256")
    if source.get("resolved_revision") != SOURCE_REVISION:
        raise VKLSVDContractError("source revision changed")

    upstream = _require_mapping(payload, "upstream_preprocessing_provenance")
    if upstream.get("revision") != HEGM_REVISION:
        raise VKLSVDContractError("HEGM revision changed")
    if upstream.get("preprocess_vklsvd_sha256") != HEGM_PREPROCESS_SHA256:
        raise VKLSVDContractError("HEGM preprocessing source hash changed")
    if upstream.get("license_sha256") != HEGM_LICENSE_SHA256:
        raise VKLSVDContractError("HEGM license hash changed")
    if upstream.get("published_result_values_read") is not False:
        raise VKLSVDContractError("published HEGM result values must remain unread")

    split = _require_mapping(payload, "split_contract")
    _require_exact_list(split, "history_weeks", HISTORY_WEEKS)
    _require_exact_list(split, "train_weeks", TRAIN_WEEKS)
    if split.get("validation_week") != VALIDATION_WEEK:
        raise VKLSVDContractError("validation_week changed")
    if split.get("sealed_test_week") != SEALED_TEST_WEEK:
        raise VKLSVDContractError("sealed_test_week changed")
    if split.get("feature_state") != "fit_once_on_history_0_19_then_freeze":
        raise VKLSVDContractError("feature state must remain history-frozen")
    if split.get("row_order") != "week_then_stable_parquet_row_index0":
        raise VKLSVDContractError("row order changed")
    if split.get("deduplication") != "none_retain_every_released_row":
        raise VKLSVDContractError("deduplication changed")

    schemas = _require_mapping(payload, "raw_schemas")
    _require_exact_list(schemas, "interaction", INTERACTION_SCHEMA)
    _require_exact_list(schemas, "users_metadata", USERS_METADATA_SCHEMA)
    _require_exact_list(schemas, "items_metadata", ITEMS_METADATA_SCHEMA)

    ordered = _require_mapping(payload, "ordered_fields")
    _require_exact_list(ordered, "C", ORDERED_C_FIELDS)
    _require_exact_list(ordered, "P0", ORDERED_P0_FIELDS)
    _require_exact_list(ordered, "V", ORDERED_V_FIELDS)
    _require_exact_list(ordered, "P", ORDERED_NUMERIC_OPERANDS)
    _require_exact_list(ordered, "E", ORDERED_E_FIELDS)

    formulas = _require_mapping(payload, "feature_formulas")
    for key, expected in {
        "P0": P0_FORMULAS,
        "V": V_FORMULAS,
        "E": E_FORMULAS,
    }.items():
        value = formulas.get(key)
        if not isinstance(value, Mapping) or dict(value) != expected:
            raise VKLSVDContractError(f"feature_formulas.{key} changed")

    smoothing = _require_mapping(payload, "smoothing")
    if smoothing.get("alpha") != ALPHA_SMOOTH:
        raise VKLSVDContractError("alpha smoothing changed")
    if smoothing.get("beta") != BETA_SMOOTH:
        raise VKLSVDContractError("beta smoothing changed")

    ials = _require_mapping(payload, "ials")
    _require_exact_list(ials, "signals", IALS_SIGNALS)
    expected_ials = {
        "fit_weeks": list(HISTORY_WEEKS),
        "factors": 32,
        "iterations": 15,
        "regularization": 0.01,
        "random_state": 42,
        "id_order": "ascending_integer",
        "threads": 1,
        "sparse_duplicate_rule": "csr_sum_duplicates",
    }
    for key, expected in expected_ials.items():
        if ials.get(key) != expected:
            raise VKLSVDContractError(f"ials.{key} changed")

    normalization = _require_mapping(payload, "normalization")
    expected_normalization = {
        "fit_weeks": list(TRAIN_WEEKS),
        "method": "divide_by_train_abs_q99",
        "quantile": 0.99,
        "interpolation": "nearest",
        "minimum_scale": 1e-12,
        "centering": "none",
        "clipping": "none",
        "validity_indicator_scale": 1.0,
    }
    for key, expected in expected_normalization.items():
        if normalization.get(key) != expected:
            raise VKLSVDContractError(f"normalization.{key} changed")

    denominator = _require_mapping(payload, "denominator_contract")
    _require_exact_list(denominator, "eligible_fields", DENOMINATOR_ELIGIBLE_FIELDS)
    _require_exact_list(denominator, "ordered_mask", ORDERED_DENOMINATOR_MASK)
    if denominator.get("validity_sources") != DENOMINATOR_VALIDITY_SOURCES:
        raise VKLSVDContractError("denominator validity sources changed")
    if denominator.get("epsilon") != RECIPROCAL_EPSILON:
        raise VKLSVDContractError("reciprocal epsilon changed")
    if denominator.get("invalid_action") != "validity_gate_to_zero":
        raise VKLSVDContractError("invalid denominator action changed")
    if denominator.get("negative_action") != "reject":
        raise VKLSVDContractError("negative denominator action changed")

    label = _require_mapping(payload, "primary_label")
    if label.get("source") != "current_row.timespent":
        raise VKLSVDContractError("primary label source changed")
    if label.get("c_seconds") != LABEL_DURATION_Q99_SECONDS:
        raise VKLSVDContractError("label c changed")
    if label.get("transform") != "clip(timespent,0,c)/c":
        raise VKLSVDContractError("label transform changed")
    if label.get("c_source") != (
        "full items_metadata.duration q0.99 with Polars nearest interpolation"
    ):
        raise VKLSVDContractError("label c provenance changed")

    hashes = _require_mapping(payload, "schema_hashes")
    expected_hashes = {
        "ordered_operand_schema_sha256": ordered_schema_sha256(
            ORDERED_NUMERIC_OPERANDS
        ),
        "denominator_mask_sha256": denominator_mask_sha256(ORDERED_DENOMINATOR_MASK),
        "C_plus_P_full_ordered_schema_sha256": ordered_schema_sha256(
            ORDERED_C_PLUS_P_FLAT_SCHEMA
        ),
        "C_plus_P_plus_E_full_ordered_schema_sha256": ordered_schema_sha256(
            ORDERED_C_PLUS_P_PLUS_E_FLAT_SCHEMA
        ),
    }
    if dict(hashes) != expected_hashes:
        raise VKLSVDContractError("schema hashes do not match generated orders")

    dimensions = _require_mapping(payload, "derived_dimensions")
    expected_dimensions = {
        "C_fields": len(ORDERED_C_FIELDS),
        "P0_fields": len(ORDERED_P0_FIELDS),
        "V_fields": len(ORDERED_V_FIELDS),
        "P_fields": len(ORDERED_NUMERIC_OPERANDS),
        "E_fields": len(ORDERED_E_FIELDS),
        "encoded_C_bypass_width": len(EXPANDED_C_COORDINATES),
        "C_plus_P_model_field_count": (
            len(ORDERED_C_FIELDS) + len(ORDERED_NUMERIC_OPERANDS)
        ),
        "C_plus_P_flat_width": len(ORDERED_C_PLUS_P_FLAT_SCHEMA),
        "C_plus_P_plus_E_model_field_count": (
            len(ORDERED_C_FIELDS)
            + len(ORDERED_NUMERIC_OPERANDS)
            + len(ORDERED_E_FIELDS)
        ),
        "C_plus_P_plus_E_flat_width": len(ORDERED_C_PLUS_P_PLUS_E_FLAT_SCHEMA),
        "denominator_mask_width": len(ORDERED_DENOMINATOR_MASK),
        "denominator_eligible_count": sum(ORDERED_DENOMINATOR_MASK),
    }
    if dict(dimensions) != expected_dimensions:
        raise VKLSVDContractError("derived dimensions changed")

    resolution = _require_mapping(payload, "resolution_gate")
    if resolution.get("unresolved_fields") != []:
        raise VKLSVDContractError("unresolved fields must be an explicit list")
    if resolution.get("first_model_metric_blocked") is not False:
        raise VKLSVDContractError("feature contract is unexpectedly unresolved")


def _read_bounded_regular(path: Path, *, limit: int, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VKLSVDContractError(
            f"{label} must be a regular non-symlink file: {path}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise VKLSVDContractError(f"{label} must be a single-link regular file")
        if metadata.st_size > limit:
            raise VKLSVDContractError(f"{label} exceeds its size limit")
        handle = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise VKLSVDContractError(f"{label} exceeds its size limit")
    return data


def load_feature_contract(path: str | Path) -> dict[str, Any]:
    """Verify the sidecar before parsing unique-key YAML."""

    source = Path(path)
    sidecar = source.with_name(f"{source.name}.sha256")
    data = _read_bounded_regular(source, limit=MAX_CONTRACT_BYTES, label="contract")
    sidecar_data = _read_bounded_regular(
        sidecar, limit=MAX_SIDECAR_BYTES, label="contract sidecar"
    )
    try:
        sidecar_text = sidecar_data.decode("ascii")
    except UnicodeDecodeError as error:
        raise VKLSVDContractError("contract sidecar must be ASCII") from error
    if not sidecar_text.endswith("\n") or sidecar_text.count("\n") != 1:
        raise VKLSVDContractError("contract sidecar must contain one line")
    parts = sidecar_text.rstrip("\n").split("  ")
    if len(parts) != 2 or parts[1] != source.name:
        raise VKLSVDContractError("contract sidecar filename is invalid")
    if SHA256_RE.fullmatch(parts[0]) is None:
        raise VKLSVDContractError("contract sidecar digest is invalid")
    if sha256_bytes(data) != parts[0]:
        raise VKLSVDContractError("contract sidecar mismatch")
    try:
        payload = yaml.load(data.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise VKLSVDContractError(f"invalid contract YAML: {error}") from error
    if not isinstance(payload, Mapping):
        raise VKLSVDContractError("contract root must be a mapping")
    result = dict(payload)
    validate_feature_contract(result)
    return result


def verify_source_manifest_reference(
    contract: Mapping[str, Any], *, repository_root: str | Path
) -> str:
    """Verify that the contract references the exact local source manifest."""

    validate_feature_contract(contract)
    source = contract["source_manifest"]
    relative = PurePosixPath(source["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise VKLSVDContractError("source manifest path is unsafe")
    root = Path(repository_root).resolve(strict=True)
    path = root.joinpath(*relative.parts)
    data = _read_bounded_regular(
        path, limit=MAX_CONTRACT_BYTES, label="source manifest"
    )
    digest = sha256_bytes(data)
    if digest != source["sha256"]:
        raise VKLSVDContractError("source manifest reference hash mismatch")
    try:
        source_payload = load_source_manifest(path)
    except Exception as error:
        raise VKLSVDContractError(str(error)) from error
    if source_payload.get("dataset_id") != SOURCE_DATASET_ID:
        raise VKLSVDContractError("source manifest dataset changed")
    if source_payload.get("resolved_revision") != SOURCE_REVISION:
        raise VKLSVDContractError("source manifest revision changed")
    if source_payload.get("test_unlocked") is not False:
        raise VKLSVDContractError("source manifest test must remain locked")
    if source_payload.get("week26_sealed") is not True:
        raise VKLSVDContractError("source manifest week26 must remain sealed")
    return digest


def classify_week(week: int) -> str:
    if type(week) is not int:
        raise TypeError("week must be an integer")
    if week in HISTORY_WEEKS:
        return "history"
    if week in TRAIN_WEEKS:
        return "train"
    if week == VALIDATION_WEEK:
        return "validation"
    if week == SEALED_TEST_WEEK:
        return "sealed_test"
    raise VKLSVDContractError("week must be in [0, 26]")


def semantic_week_relative_path(week: int) -> PurePosixPath:
    """Return an unsealed semantic path; fail before touching week 26."""

    role = classify_week(week)
    if role == "sealed_test":
        raise VKLSVDContractError(
            "week 26 semantic access is forbidden while test_unlocked=false"
        )
    split = "validation" if week == VALIDATION_WEEK else "train"
    return PurePosixPath(f"subsamples/up0.01_ir0.01/{split}/week_{week:02d}.parquet")


def phase1_semantic_week_paths(raw_root: str | Path) -> tuple[Path, ...]:
    root = Path(raw_root)
    return tuple(
        root.joinpath(*semantic_week_relative_path(week).parts)
        for week in PHASE1_SEMANTIC_WEEKS
    )


def stable_row_id(*, week: int, source_row_index0: int) -> str:
    if week not in PHASE1_SEMANTIC_WEEKS and week != SEALED_TEST_WEEK:
        raise VKLSVDContractError("stable row IDs require week in [0, 26]")
    if type(source_row_index0) is not int or source_row_index0 < 0:
        raise VKLSVDContractError("source_row_index0 must be non-negative")
    payload = (
        f"{SOURCE_REVISION}\0up0.01_ir0.01\0{week:02d}\0{source_row_index0}"
    ).encode("ascii")
    return f"vklsvd-v1-{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class GroupHistory:
    """One history-only user or item aggregate before smoothing."""

    show_count: int
    timespent_sum_seconds: float
    event_counts: Mapping[str, int]

    def validate(self, required_sources: Sequence[str]) -> None:
        if type(self.show_count) is not int or self.show_count < 0:
            raise VKLSVDContractError("show_count must be a non-negative integer")
        if not math.isfinite(self.timespent_sum_seconds):
            raise VKLSVDContractError("timespent sum must be finite")
        if self.timespent_sum_seconds < 0:
            raise VKLSVDContractError("timespent sum must be non-negative")
        if set(self.event_counts) != set(required_sources):
            raise VKLSVDContractError("history event-count keys changed")
        for value in self.event_counts.values():
            if type(value) is not int or value < 0 or value > self.show_count:
                raise VKLSVDContractError(
                    "event counts must be integers in [0, show_count]"
                )


@dataclass(frozen=True)
class IALSOperands:
    dot: float
    user_norm: float
    item_norm: float
    user_seen: bool
    item_seen: bool

    def validate(self) -> None:
        if type(self.user_seen) is not bool or type(self.item_seen) is not bool:
            raise VKLSVDContractError("iALS seen flags must be booleans")
        values = (self.dot, self.user_norm, self.item_norm)
        if any(not math.isfinite(value) for value in values):
            raise VKLSVDContractError("iALS operands must be finite")
        if self.user_norm < 0 or self.item_norm < 0:
            raise VKLSVDContractError("iALS norms must be non-negative")
        if not self.user_seen and (self.dot != 0 or self.user_norm != 0):
            raise VKLSVDContractError("unseen iALS users require zero fallback")
        if not self.item_seen and (self.dot != 0 or self.item_norm != 0):
            raise VKLSVDContractError("unseen iALS items require zero fallback")


@dataclass(frozen=True)
class FeatureInputs:
    duration_seconds: float
    item_metadata_present: bool
    user_history: GroupHistory
    item_history: GroupHistory
    ials: Mapping[str, IALSOperands]

    def validate(self) -> None:
        if type(self.item_metadata_present) is not bool:
            raise VKLSVDContractError("item_metadata_present must be boolean")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise VKLSVDContractError("duration must be finite and non-negative")
        if not self.item_metadata_present and self.duration_seconds != 0:
            raise VKLSVDContractError("missing metadata requires duration fallback 0")
        self.user_history.validate(tuple(source for _, source in USER_RATE_SOURCES))
        self.item_history.validate(tuple(source for _, source in ITEM_RATE_SOURCES))
        if set(self.ials) != set(IALS_SIGNALS):
            raise VKLSVDContractError("iALS signal keys changed")
        for operands in self.ials.values():
            operands.validate()


@dataclass(frozen=True)
class DerivedFeatureRow:
    raw_p0: tuple[float, ...]
    normalized_p0: tuple[float, ...]
    validity_indicators: tuple[float, ...]
    numeric_operands: tuple[float, ...]
    raw_e: tuple[float, ...]
    normalized_e: tuple[float, ...]
    denominator_validity: tuple[bool, ...]


def _validate_scales(
    scales: Mapping[str, float], expected_fields: Sequence[str], label: str
) -> None:
    if set(scales) != set(expected_fields):
        raise VKLSVDContractError(f"{label} scale keys changed")
    if any(not math.isfinite(value) or value <= 0 for value in scales.values()):
        raise VKLSVDContractError(f"{label} scales must be finite and positive")


def positive_abs_q99_scale(values: Iterable[float]) -> float:
    """Reference nearest-q99 scale for deterministic tiny/smoke builders."""

    ordered = sorted(abs(float(value)) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise VKLSVDContractError("scale fitting requires finite non-empty values")
    index = math.floor((len(ordered) - 1) * 0.99 + 0.5)
    return max(ordered[index], 1e-12)


def _raw_p0(inputs: FeatureInputs) -> dict[str, float]:
    result: dict[str, float] = {"p_duration_seconds": inputs.duration_seconds}
    for group, aggregate, rates in (
        ("user", inputs.user_history, USER_RATE_SOURCES),
        ("item", inputs.item_history, ITEM_RATE_SOURCES),
    ):
        result[f"p_{group}_show_count"] = float(aggregate.show_count)
        result[f"p_{group}_show_count_plus_beta"] = (
            float(aggregate.show_count) + BETA_SMOOTH
        )
        result[f"p_{group}_timespent_sum_seconds"] = float(
            aggregate.timespent_sum_seconds
        )
        for name, source in rates:
            result[f"p_{group}_{name}_count_plus_alpha"] = (
                float(aggregate.event_counts[source]) + ALPHA_SMOOTH
            )
    for signal in IALS_SIGNALS:
        operands = inputs.ials[signal]
        result[f"p_ials_{signal}_dot"] = float(operands.dot)
        result[f"p_ials_{signal}_user_norm"] = float(operands.user_norm)
        result[f"p_ials_{signal}_item_norm"] = float(operands.item_norm)
    if tuple(result) != ORDERED_P0_FIELDS:
        raise AssertionError("generated P0 order drifted")
    return result


def _raw_e(inputs: FeatureInputs, p0: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for group, aggregate, rates in (
        ("user", inputs.user_history, USER_RATE_SOURCES),
        ("item", inputs.item_history, ITEM_RATE_SOURCES),
    ):
        result[f"e_{group}_avg_timespent_seconds"] = (
            aggregate.timespent_sum_seconds / aggregate.show_count
            if aggregate.show_count > 0
            else 0.0
        )
        for name, _ in rates:
            result[f"e_{group}_{name}_rate"] = (
                p0[f"p_{group}_{name}_count_plus_alpha"]
                / p0[f"p_{group}_show_count_plus_beta"]
            )
    for signal in IALS_SIGNALS:
        operands = inputs.ials[signal]
        valid = (
            operands.user_seen
            and operands.item_seen
            and operands.user_norm > 0
            and operands.item_norm > 0
        )
        denominator = operands.user_norm * operands.item_norm
        result[f"e_ials_{signal}_cosine"] = operands.dot / denominator if valid else 0.0
    if tuple(result) != ORDERED_E_FIELDS:
        raise AssertionError("generated E order drifted")
    return result


def _denominator_validity_map(
    inputs: FeatureInputs, p0: Mapping[str, float]
) -> dict[str, bool]:
    result = {
        "p_duration_seconds": (
            inputs.item_metadata_present and p0["p_duration_seconds"] > 0
        ),
        "p_user_show_count": inputs.user_history.show_count > 0,
        "p_user_show_count_plus_beta": True,
        "p_item_show_count": inputs.item_history.show_count > 0,
        "p_item_show_count_plus_beta": True,
    }
    for signal in IALS_SIGNALS:
        operands = inputs.ials[signal]
        result[f"p_ials_{signal}_user_norm"] = (
            operands.user_seen and operands.user_norm > 0
        )
        result[f"p_ials_{signal}_item_norm"] = (
            operands.item_seen and operands.item_norm > 0
        )
    if tuple(result) != DENOMINATOR_ELIGIBLE_FIELDS:
        raise AssertionError("generated denominator order drifted")
    return result


def derive_feature_row(
    inputs: FeatureInputs,
    *,
    p0_scales: Mapping[str, float],
    e_scales: Mapping[str, float],
    reciprocal_epsilon: float = RECIPROCAL_EPSILON,
) -> DerivedFeatureRow:
    """Derive one normalized C+P/C+P+E row from history-frozen inputs."""

    inputs.validate()
    _validate_scales(p0_scales, ORDERED_P0_FIELDS, "P0")
    _validate_scales(e_scales, ORDERED_E_FIELDS, "E")
    if not math.isfinite(reciprocal_epsilon) or reciprocal_epsilon <= 0:
        raise VKLSVDContractError("reciprocal epsilon must be finite and positive")

    p0 = _raw_p0(inputs)
    e = _raw_e(inputs, p0)
    normalized_p0_map = {
        field: p0[field] / p0_scales[field] for field in ORDERED_P0_FIELDS
    }
    normalized_e_map = {field: e[field] / e_scales[field] for field in ORDERED_E_FIELDS}
    if any(not math.isfinite(value) for value in normalized_p0_map.values()):
        raise VKLSVDContractError("normalized P0 must be finite")
    if any(not math.isfinite(value) for value in normalized_e_map.values()):
        raise VKLSVDContractError("normalized E must be finite")

    denominator_validity_map = _denominator_validity_map(inputs, p0)
    for field, valid in denominator_validity_map.items():
        value = normalized_p0_map[field]
        if valid and value <= 0:
            raise VKLSVDContractError(
                f"active denominator {field} must remain positive after scaling"
            )
        if value < 0:
            raise VKLSVDContractError(f"denominator candidate {field} is negative")

    v: dict[str, float] = {
        "v_user_history_present": float(inputs.user_history.show_count > 0),
        "v_item_history_present": float(inputs.item_history.show_count > 0),
        "v_item_metadata_present": float(inputs.item_metadata_present),
        "v_duration_positive": float(
            inputs.item_metadata_present and inputs.duration_seconds > 0
        ),
    }
    for signal in IALS_SIGNALS:
        operands = inputs.ials[signal]
        v[f"v_ials_{signal}_user_seen"] = float(operands.user_seen)
        v[f"v_ials_{signal}_item_seen"] = float(operands.item_seen)
        v[f"v_ials_{signal}_user_norm_positive"] = float(
            operands.user_seen and operands.user_norm > 0
        )
        v[f"v_ials_{signal}_item_norm_positive"] = float(
            operands.item_seen and operands.item_norm > 0
        )
    for field in DENOMINATOR_ELIGIBLE_FIELDS:
        v[floor_indicator_name(field)] = float(
            denominator_validity_map[field]
            and normalized_p0_map[field] < reciprocal_epsilon
        )
    if tuple(v) != ORDERED_V_FIELDS:
        raise AssertionError("generated V order drifted")

    normalized_p0 = tuple(normalized_p0_map[field] for field in ORDERED_P0_FIELDS)
    validity_indicators = tuple(v[field] for field in ORDERED_V_FIELDS)
    numeric_operands = normalized_p0 + validity_indicators
    denominator_validity = tuple(
        denominator_validity_map.get(field, True) for field in ORDERED_NUMERIC_OPERANDS
    )
    return DerivedFeatureRow(
        raw_p0=tuple(p0[field] for field in ORDERED_P0_FIELDS),
        normalized_p0=normalized_p0,
        validity_indicators=validity_indicators,
        numeric_operands=numeric_operands,
        raw_e=tuple(e[field] for field in ORDERED_E_FIELDS),
        normalized_e=tuple(normalized_e_map[field] for field in ORDERED_E_FIELDS),
        denominator_validity=denominator_validity,
    )
