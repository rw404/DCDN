from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest
import yaml

from dcdn.data.kuairec import (
    BIG_MATRIX_COLUMNS,
    DAILY_BASE_SOURCES,
    DAILY_COLUMNS,
    DENOMINATOR_ELIGIBLE_FIELDS,
    EXPECTED_DAILY_ROWS,
    EXPECTED_PARTIAL_NULL_ROWS,
    MILLISECONDS_PER_SECOND,
    ORDERED_C_FIELDS,
    ORDERED_DENOMINATOR_MASK,
    ORDERED_E_FIELDS,
    ORDERED_NUMERIC_OPERANDS,
    ORDERED_P_FIELDS,
    ORDERED_V_FIELDS,
    PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES,
    DailyPrefixRow,
    KuaiRecContractError,
    build_daily_prefix_index,
    build_feature_values,
    daily_history_before,
    denominator_mask_sha256,
    fit_sorted_mapping,
    linear_quantile,
    load_feature_contract,
    ordered_schema_sha256,
    parse_released_date,
    split_boundaries,
    split_role,
    stable_row_id,
    validate_feature_contract,
    verify_source_manifest_reference,
)
from scripts.data.kuairec_features import (
    KuaiRecBlindAuditError,
    _audit_daily_numeric,
    _audit_excluded_partial_null_daily,
    _load_projected_big_matrix,
    _membership_hash,
    _static_video_types,
    write_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/data/kuairec/features_v1.yaml"


def _write_hashed(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


def _read_hashed_yaml(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert path.with_name(f"{path.name}.sha256").read_text(encoding="ascii") == (
        f"{digest}  {path.name}\n"
    )
    payload = yaml.safe_load(data)
    assert isinstance(payload, dict)
    return payload


def _zero_daily_values(**updates: float) -> dict[str, float]:
    values = {source: 0.0 for source in DAILY_BASE_SOURCES}
    values.update(updates)
    return values


def _write_daily_csv(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [",".join(DAILY_COLUMNS)]
    for updates in rows:
        row = {column: "0" for column in DAILY_COLUMNS}
        row.update({"video_id": "1", "date": "20200101", "video_type": "NORMAL"})
        row.update(updates)
        lines.append(",".join(row[column] for column in DAILY_COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_big_csv(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [",".join(BIG_MATRIX_COLUMNS)]
    for updates in rows:
        row = {
            "user_id": "1",
            "video_id": "1",
            "play_duration": "5",
            "video_duration": "10",
            "time": "opaque",
            "date": "20200102",
            "timestamp": "2",
            "watch_ratio": "secret-outcome-derived",
        }
        row.update(updates)
        lines.append(",".join(row[column] for column in BIG_MATRIX_COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_persisted_contract_and_source_reference_are_locked() -> None:
    payload = load_feature_contract(CONTRACT)
    assert payload["test_unlocked"] is False
    assert payload["test_labels_opened"] is False
    assert payload["model_metrics_read"] is False
    assert payload["final_test_view_materialized"] is False
    assert verify_source_manifest_reference(payload, repository_root=ROOT) == (
        "963019bcb4c8dbd4663f23c24a472fe0332f6494fdcbbaa74ca158a73488c5b4"
    )
    revision = payload["schema_only_revision"]
    assert revision["initial_contract_sha256"] == (
        "c6f539d228dce75644b62cb445bf393e141944c9e66bf84545db77e3b75a606d"
    )
    assert revision["daily_rows_checked"] == EXPECTED_DAILY_ROWS
    assert revision["null_rows_per_excluded_source"] == EXPECTED_PARTIAL_NULL_ROWS
    assert revision["nonnull_rows_per_excluded_source"] == (
        EXPECTED_DAILY_ROWS - EXPECTED_PARTIAL_NULL_ROWS
    )
    assert revision["source_wide_all_null"] is False
    assert revision["owner_approved_full_exclusion_all_regimes"] is True
    assert revision["zero_imputation_used"] is False
    assert revision["big_matrix_interaction_rows_read_before_revision"] is False
    assert revision["filtering_and_split_changed"] is False
    unit_revision = payload["unit_normalization_revision"]
    assert unit_revision["rejected_contract_sha256"] == (
        "c5aadbeb7da5f05a281681d30a7ef98e8c660019f6c0bc679e430c1014e5a23c"
    )
    assert unit_revision["filtering_and_split_changed"] is False
    assert unit_revision["ordered_schema_names_or_hashes_changed"] is False
    assert unit_revision["watch_ratio_read"] is False
    incident = payload["inspection_incident"]
    assert incident["quarantined_values_persisted_or_used"] is False
    assert incident["local_test_label_rows_or_vectors_read"] is False
    units = payload["unit_normalization"]
    assert units["milliseconds_per_second"] == MILLISECONDS_PER_SECOND
    assert units["positivity_filters_run_on_raw_milliseconds"] is True
    assert units["conversion_changes_filter_split_or_order"] is False


def test_corrected_blind_audit_and_invalidated_artifacts_are_locked() -> None:
    directory = CONTRACT.parent
    audit = _read_hashed_yaml(directory / "blind_scan_v1.yaml")
    assert audit["feature_contract"]["sha256"] == (
        "ab016a7f6d300e7a90de9940891ea974695aa5f61ea85985e239588290522c36"
    )
    assert audit["test_unlocked"] is False
    assert audit["test_labels_opened"] is False
    assert audit["watch_ratio_read"] is False
    assert audit["validation_or_test_model_metrics_read"] is False
    excluded = audit["excluded_partial_null_daily_schema"]
    assert excluded["daily_rows"] == EXPECTED_DAILY_ROWS
    assert excluded["expected_null_rows_per_column"] == EXPECTED_PARTIAL_NULL_ROWS
    assert excluded["excluded_from_all_feature_regimes"] is True
    for source in PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES:
        assert excluded["columns"][source]["null"] == EXPECTED_PARTIAL_NULL_ROWS
        assert excluded["columns"][source]["nonnull"] == (
            EXPECTED_DAILY_ROWS - EXPECTED_PARTIAL_NULL_ROWS
        )
    assert audit["train_only_label_constant"]["value"] == pytest.approx(131.684)

    invalid = _read_hashed_yaml(directory / "blind_scan_v1.invalid.yaml")
    assert invalid["invalidation"]["blind_scan_status"] == "invalid_not_usable"
    assert invalid["corrected_successor"]["blind_scan_sha256"] == (
        "716e976e0afc642b354c50096ca0d12bdef07e55381b5e91b75cc7c4910f810a"
    )
    assert invalid["corrected_successor"]["test_unlocked"] is False

    archives = {
        "features_v1.rejected-f0883ced.yaml": (
            "f0883ced4c8fb3f0945cacb1affa0b4891f748c4294e3a2d1016100191011b8d"
        ),
        "blind_scan_v1.rejected-3b383910.yaml": (
            "3b38391097c9294b7622bc91c7b836283247b7ff41bbf41a0a7a4e84c35357ab"
        ),
    }
    for name, expected in archives.items():
        path = directory / name
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        assert path.with_name(f"{name}.sha256").read_text(encoding="ascii") == (
            f"{expected}  {name}\n"
        )


def test_frozen_orders_counts_hashes_and_mask_are_exact() -> None:
    payload = load_feature_contract(CONTRACT)
    ordered = payload["ordered_fields"]
    assert ordered["C"] == list(ORDERED_C_FIELDS)
    assert ordered["P"] == list(ORDERED_P_FIELDS)
    assert ordered["V"] == list(ORDERED_V_FIELDS)
    assert ordered["E"] == list(ORDERED_E_FIELDS)
    assert ordered["numeric_interaction_operands"] == list(ORDERED_NUMERIC_OPERANDS)
    assert (len(ORDERED_C_FIELDS), len(ORDERED_P_FIELDS)) == (5, 44)
    assert (len(ORDERED_V_FIELDS), len(ORDERED_E_FIELDS)) == (6, 46)
    assert sum(ORDERED_DENOMINATOR_MASK) == 5
    assert set(PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES).isdisjoint(DAILY_BASE_SOURCES)
    excluded_rates = {
        "e_collect_per_show",
        "e_collect_per_show_user",
        "e_cancel_collect_per_show",
        "e_cancel_collect_per_show_user",
    }
    assert excluded_rates.isdisjoint(ORDERED_E_FIELDS)
    assert payload["denominator_mask"]["eligible_fields"] == list(
        DENOMINATOR_ELIGIBLE_FIELDS
    )
    assert payload["schema_hashes"]["numeric_interaction_operands"] == (
        ordered_schema_sha256(ORDERED_NUMERIC_OPERANDS)
    )
    assert payload["schema_hashes"]["denominator_mask"] == (
        denominator_mask_sha256(ORDERED_DENOMINATOR_MASK)
    )


def test_contract_rejects_test_unlock_reorder_and_same_day_history() -> None:
    payload = deepcopy(load_feature_contract(CONTRACT))
    payload["test_unlocked"] = True
    with pytest.raises(KuaiRecContractError, match="test_unlocked"):
        validate_feature_contract(payload)

    payload = deepcopy(load_feature_contract(CONTRACT))
    payload["ordered_fields"]["P"][0:2] = reversed(payload["ordered_fields"]["P"][0:2])
    with pytest.raises(KuaiRecContractError, match="frozen order"):
        validate_feature_contract(payload)

    payload = deepcopy(load_feature_contract(CONTRACT))
    payload["daily_history"]["same_day_allowed"] = True
    with pytest.raises(KuaiRecContractError, match="same_day_allowed"):
        validate_feature_contract(payload)

    payload = deepcopy(load_feature_contract(CONTRACT))
    payload["unit_normalization"]["milliseconds_per_second"] = 1.0
    with pytest.raises(KuaiRecContractError, match="duration divisor"):
        validate_feature_contract(payload)


def test_contract_loader_rejects_duplicate_keys_and_tampering(tmp_path: Path) -> None:
    source = CONTRACT.read_text(encoding="utf-8")
    duplicated = source.replace(
        "dataset_id: kuairec_big_matrix\n",
        "dataset_id: kuairec_big_matrix\ndataset_id: conflict\n",
        1,
    )
    duplicate_path = tmp_path / "duplicate.yaml"
    _write_hashed(duplicate_path, duplicated)
    with pytest.raises(KuaiRecContractError, match="duplicate-key"):
        load_feature_contract(duplicate_path)

    tampered = tmp_path / "tampered.yaml"
    _write_hashed(tampered, source)
    tampered.write_text(
        source.replace("test_unlocked: false", "test_unlocked: true", 1),
        encoding="utf-8",
    )
    with pytest.raises(KuaiRecContractError, match="sidecar mismatch"):
        load_feature_contract(tampered)


def test_split_matches_hegm_floor_rule_and_test_remains_last() -> None:
    boundaries = split_boundaries(101)
    assert boundaries.train_rows == 80
    assert boundaries.validation_rows_before_cold_drop == 10
    assert boundaries.test_rows_before_cold_drop == 11
    assert split_role(0, boundaries) == "train"
    assert split_role(79, boundaries) == "train"
    assert split_role(80, boundaries) == "validation"
    assert split_role(89, boundaries) == "validation"
    assert split_role(90, boundaries) == "sealed_test"
    with pytest.raises(KuaiRecContractError):
        split_role(101, boundaries)


def test_stable_row_id_depends_only_on_source_identity_and_index() -> None:
    assert stable_row_id(7) == stable_row_id(7)
    assert stable_row_id(7) != stable_row_id(8)
    assert len(stable_row_id(7)) == 64
    with pytest.raises(KuaiRecContractError):
        stable_row_id(-1)


def test_train_mapping_and_linear_q99_are_deterministic() -> None:
    assert fit_sorted_mapping([9, 2, 9, 4]) == {2: 0, 4: 1, 9: 2}
    assert fit_sorted_mapping(["b", "a"], first_index=1) == {"a": 1, "b": 2}
    assert linear_quantile([1.0, 2.0, 3.0, 4.0], q=0.5) == 2.5
    assert linear_quantile([10.0, 20.0], q=0.99) == pytest.approx(19.9)
    with pytest.raises(KuaiRecContractError, match="positive"):
        linear_quantile([0.0, 1.0])


def test_daily_prefix_is_strictly_before_target_and_sums_duplicates() -> None:
    index = build_daily_prefix_index(
        [
            DailyPrefixRow(
                1,
                date(2020, 1, 1),
                _zero_daily_values(show_cnt=2, play_cnt=1, play_duration=5_000),
            ),
            DailyPrefixRow(
                1,
                date(2020, 1, 1),
                _zero_daily_values(show_cnt=3, play_cnt=2, play_duration=7_000),
            ),
            DailyPrefixRow(
                1,
                date(2020, 1, 2),
                _zero_daily_values(show_cnt=11, play_cnt=9, play_duration=13_000),
            ),
        ]
    )
    empty, present = daily_history_before(
        index, video_id=1, target_day=date(2020, 1, 1)
    )
    assert present is False
    assert empty["show_cnt"] == 0

    first_day, present = daily_history_before(
        index, video_id=1, target_day=date(2020, 1, 2)
    )
    assert present is True
    assert first_day["show_cnt"] == 5
    assert first_day["play_duration"] == 12

    through_second, _ = daily_history_before(
        index, video_id=1, target_day=date(2020, 1, 3)
    )
    assert through_second["show_cnt"] == 16
    assert through_second["play_cnt"] == 12
    assert through_second["play_duration"] == 25


def test_feature_algebra_uses_only_lagged_values_and_positive_denominators() -> None:
    history = _zero_daily_values(
        show_cnt=9,
        show_user_num=4,
        play_cnt=4,
        play_user_num=2,
        play_duration=30,
        complete_play_cnt=2,
        complete_play_user_num=1,
        like_cnt=3,
        like_user_num=2,
    )
    result = build_feature_values(
        video_duration_seconds=60,
        history=history,
        prior_history_present=True,
    )
    assert list(result["P"]) == list(ORDERED_P_FIELDS)
    assert list(result["V"]) == list(ORDERED_V_FIELDS)
    assert list(result["E"]) == list(ORDERED_E_FIELDS)
    assert result["P"]["p_hist_show_cnt_plus_1"] == 10
    assert result["P"]["p_hist_play_cnt_plus_1"] == 5
    assert result["E"]["e_play_per_show"] == pytest.approx(0.4)
    assert result["E"]["e_average_play_duration_seconds"] == pytest.approx(6.0)
    assert result["E"]["e_like_per_show"] == pytest.approx(0.3)
    assert result["E"]["e_like_per_show_user"] == pytest.approx(0.4)
    for field in DENOMINATOR_ELIGIBLE_FIELDS:
        assert result["P"][field] > 0
    assert "play_duration" not in result["P"]
    assert "watch_ratio" not in result["P"]


def test_released_date_parser_is_exact() -> None:
    assert parse_released_date(20200705) == date(2020, 7, 5)
    with pytest.raises(KuaiRecContractError):
        parse_released_date("2020-07-05")


def test_label_blind_csv_projection_and_daily_schema_checks(tmp_path: Path) -> None:
    daily = tmp_path / "item_daily_features.csv"
    _write_daily_csv(
        daily,
        [
            {
                "video_id": "1",
                "date": "20200101",
                "show_cnt": "3",
                **{source: "" for source in PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES},
            },
            {"video_id": "1", "date": "20200102", "show_cnt": "4"},
            {"video_id": "2", "date": "20200101", "video_type": "AD"},
        ],
    )
    static, static_evidence = _static_video_types(daily)
    assert static.height == 2
    assert static_evidence["videos_with_inconsistent_video_type"] == 0
    daily_evidence = _audit_daily_numeric(daily)
    assert daily_evidence == {
        "daily_rows": 3,
        "video_date_groups": 3,
        "invalid_numeric_or_key_cells": 0,
    }

    partial = _audit_excluded_partial_null_daily(
        daily, expected_rows=3, expected_null_rows=1
    )
    assert partial["daily_rows"] == 3
    assert partial["excluded_from_all_feature_regimes"] is True
    for source in PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES:
        assert partial["columns"][source] == {
            "total": 3,
            "null": 1,
            "nonnull": 2,
            "parse_invalid": 0,
            "nonfinite": 0,
            "negative": 0,
        }
    with pytest.raises(KuaiRecBlindAuditError, match="missingness drifted"):
        _audit_excluded_partial_null_daily(daily, expected_rows=3, expected_null_rows=0)
    big = tmp_path / "big_matrix.csv"
    _write_big_csv(
        big,
        [
            {"user_id": "2", "timestamp": "5", "watch_ratio": "not-a-number"},
            {"user_id": "1", "timestamp": "3", "watch_ratio": "still-opaque"},
        ],
    )
    projected = _load_projected_big_matrix(big)
    assert projected.columns == [
        "source_row_index0",
        "user_id",
        "video_id",
        "play_duration",
        "video_duration",
        "date",
        "timestamp",
    ]
    assert "watch_ratio" not in projected.columns


def test_video_type_invariance_gate_fails_closed(tmp_path: Path) -> None:
    daily = tmp_path / "item_daily_features.csv"
    _write_daily_csv(
        daily,
        [
            {"video_id": "1", "date": "20200101", "video_type": "NORMAL"},
            {"video_id": "1", "date": "20200102", "video_type": "AD"},
        ],
    )
    with pytest.raises(KuaiRecBlindAuditError, match="not static"):
        _static_video_types(daily)


def test_membership_hash_uses_ordered_source_indices_not_outcomes(
    tmp_path: Path,
) -> None:
    big = tmp_path / "big_matrix.csv"
    _write_big_csv(
        big,
        [
            {"timestamp": "9", "watch_ratio": "secret-a"},
            {"timestamp": "8", "watch_ratio": "secret-b"},
        ],
    )
    frame = (
        _load_projected_big_matrix(big)
        .sort("timestamp")
        .with_row_index("filtered_sorted_rank0")
    )
    digest = _membership_hash(frame, split_name="validation_after_cold_drop")
    assert len(digest) == 64
    changed = frame.with_columns((frame["video_duration"] * 2).alias("video_duration"))
    assert _membership_hash(changed, split_name="validation_after_cold_drop") == digest


def test_atomic_audit_writer_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "blind_scan_v1.yaml"
    payload = {
        "schema_version": 1,
        "test_unlocked": False,
        "validation_or_test_model_metrics_read": False,
    }
    digest = write_audit(path, payload)
    data = path.read_bytes()
    assert digest == hashlib.sha256(data).hexdigest()
    assert path.with_name(f"{path.name}.sha256").read_text(encoding="ascii") == (
        f"{digest}  blind_scan_v1.yaml\n"
    )
    assert yaml.safe_load(data)["test_unlocked"] is False
