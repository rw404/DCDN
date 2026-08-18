from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dcdn.data.kuairand_pure import (
    DENOMINATOR_ELIGIBLE_FIELDS,
    EVENT_COLUMNS,
    EXPECTED_EMBARGO_CUTOFFS,
    EXPECTED_EMBARGO_EXCLUDED_COUNTS,
    EXPECTED_POST_EMBARGO_ROLE_COUNTS,
    ORDERED_C_FIELDS,
    ORDERED_DENOMINATOR_MASK,
    ORDERED_NUMERIC_OPERANDS,
    ORDERED_P_FIELDS,
    ORDERED_R_FIELDS,
    ORDERED_V_FIELDS,
    RANDOM_MAIN_FILE,
    STANDARD_HISTORY_FILE,
    STANDARD_MAIN_FILE,
    KuaiRandContractError,
    RowOrderingKey,
    canonical_full_row_bytes,
    classify_row_role,
    denominator_mask_sha256,
    full_row_fingerprint,
    is_row_embargo_excluded,
    iter_prequential_groups,
    load_feature_contract,
    ordered_schema_sha256,
    stable_row_id,
    validate_feature_contract,
    verify_source_manifest_reference,
)
from scripts.data.kuairand_pure_features import (
    CoverageAuditError,
    run_label_blind_coverage_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/data/kuairand_pure/features_v1.yaml"


def _event_row(**updates: str) -> list[str]:
    row = {column: f"opaque-{column}" for column in EVENT_COLUMNS}
    row.update(
        {
            "user_id": "u1",
            "video_id": "v1",
            "date": "20220422",
            "hourmin": "1200",
            "time_ms": "1650628800000",
            "is_rand": "0",
            "tab": "1",
        }
    )
    row.update(updates)
    return [row[column] for column in EVENT_COLUMNS]


def _write_hashed(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


def test_persisted_contract_and_source_reference_are_verified() -> None:
    payload = load_feature_contract(CONTRACT)
    assert payload["test_unlocked"] is False
    assert payload["test_labels_opened"] is False
    assert payload["model_metrics_read"] is False
    assert payload["final_test_view_materialized"] is False
    assert payload["phase1_gate_passed"] is True
    policy = payload["approved_temporal_policy"]
    assert policy["status"] == "owner_approved_and_applied"
    assert policy["applied"] is True
    assert policy["actual_burn_in"]["actual_available_start"] == "2022-04-09"
    assert policy["actual_burn_in"]["april_08_rows_available"] is False
    assert policy["embargo"]["total_excluded_rows"] == 6_639
    assert verify_source_manifest_reference(payload, repository_root=ROOT) == (
        "bc5b441fdd3188d033e0ecbf0bdfd73f8bcf07bbc744d474d260ee220ae4d6e9"
    )


def test_exact_field_orders_hashes_and_mask_remain_aligned() -> None:
    payload = load_feature_contract(CONTRACT)
    ordered = payload["ordered_fields"]
    assert ordered["C"] == list(ORDERED_C_FIELDS)
    assert ordered["P"] == list(ORDERED_P_FIELDS)
    assert ordered["V"] == list(ORDERED_V_FIELDS)
    assert ordered["R"] == list(ORDERED_R_FIELDS)
    assert ordered["numeric_interaction_operands"] == list(ORDERED_NUMERIC_OPERANDS)
    assert len(ORDERED_C_FIELDS) == 8
    assert len(ORDERED_P_FIELDS) == 108
    assert len(ORDERED_V_FIELDS) == 24
    assert len(ORDERED_R_FIELDS) == 112
    assert sum(ORDERED_DENOMINATOR_MASK) == 24
    assert payload["denominator_mask"]["eligible_fields"] == list(
        DENOMINATOR_ELIGIBLE_FIELDS
    )
    assert payload["schema_hashes"]["numeric_interaction_operands"] == (
        ordered_schema_sha256(ORDERED_NUMERIC_OPERANDS)
    )
    assert payload["schema_hashes"]["denominator_mask"] == (
        denominator_mask_sha256(ORDERED_DENOMINATOR_MASK)
    )


def test_contract_rejects_test_unlock_and_field_reordering() -> None:
    payload = deepcopy(load_feature_contract(CONTRACT))
    payload["test_unlocked"] = True
    with pytest.raises(KuaiRandContractError, match="test_unlocked"):
        validate_feature_contract(payload)

    payload = deepcopy(load_feature_contract(CONTRACT))
    payload["approved_temporal_policy"]["applied"] = False
    with pytest.raises(KuaiRandContractError, match="must be exactly True"):
        validate_feature_contract(payload)

    payload = deepcopy(load_feature_contract(CONTRACT))
    fields = payload["ordered_fields"]["P"]
    fields[0], fields[1] = fields[1], fields[0]
    with pytest.raises(KuaiRandContractError, match="frozen order"):
        validate_feature_contract(payload)


def test_contract_loader_rejects_duplicate_keys_even_when_rehashed(
    tmp_path: Path,
) -> None:
    source = CONTRACT.read_text(encoding="utf-8")
    duplicated = source.replace(
        "dataset_id: kuairand_pure\n",
        "dataset_id: kuairand_pure\ndataset_id: conflict\n",
        1,
    )
    path = tmp_path / "duplicate.yaml"
    _write_hashed(path, duplicated)
    with pytest.raises(KuaiRandContractError, match="duplicate key"):
        load_feature_contract(path)


def test_contract_loader_rejects_tamper_before_yaml_parse(tmp_path: Path) -> None:
    path = tmp_path / "features.yaml"
    text = CONTRACT.read_text(encoding="utf-8")
    _write_hashed(path, text)
    path.write_text(text.replace("test_unlocked: false", "test_unlocked: true"))
    with pytest.raises(KuaiRandContractError, match="sidecar mismatch"):
        load_feature_contract(path)


def test_full_row_fingerprint_mapping_and_sequence_are_identical() -> None:
    sequence = _event_row()
    mapping = dict(zip(EVENT_COLUMNS, sequence, strict=True))
    assert canonical_full_row_bytes(sequence) == canonical_full_row_bytes(mapping)
    assert full_row_fingerprint(sequence) == full_row_fingerprint(mapping)


@given(
    column_index=st.integers(min_value=0, max_value=len(EVENT_COLUMNS) - 1),
    suffix=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",), blacklist_characters=("\x00",)
        ),
        min_size=1,
        max_size=8,
    ),
)
def test_every_full_row_field_is_fingerprint_significant(
    column_index: int, suffix: str
) -> None:
    baseline = _event_row()
    changed = list(baseline)
    changed[column_index] += suffix
    assert full_row_fingerprint(changed) != full_row_fingerprint(baseline)


def test_fingerprint_normalizes_unicode_but_preserves_numeric_spelling() -> None:
    decomposed = _event_row(tab="e\u0301")
    composed = _event_row(tab="\u00e9")
    assert full_row_fingerprint(decomposed) == full_row_fingerprint(composed)
    assert full_row_fingerprint(_event_row(tab="1")) != full_row_fingerprint(
        _event_row(tab="1.0")
    )


def test_stable_row_id_uses_fingerprint_and_canonical_occurrence() -> None:
    fingerprint = full_row_fingerprint(_event_row())
    first = stable_row_id(
        fingerprint=fingerprint,
        source_file=STANDARD_MAIN_FILE,
        source_row_index0=7,
    )
    assert first == stable_row_id(
        fingerprint=fingerprint,
        source_file=STANDARD_MAIN_FILE,
        source_row_index0=7,
    )
    assert first != stable_row_id(
        fingerprint=fingerprint,
        source_file=STANDARD_MAIN_FILE,
        source_row_index0=8,
    )
    with pytest.raises(KuaiRandContractError, match="source_row_index0"):
        stable_row_id(
            fingerprint=fingerprint,
            source_file=STANDARD_MAIN_FILE,
            source_row_index0=-1,
        )


def test_equal_timestamp_rows_share_pre_update_state() -> None:
    rows = [
        RowOrderingKey(101, "randomized", RANDOM_MAIN_FILE, 4),
        RowOrderingKey(100, "randomized", RANDOM_MAIN_FILE, 2),
        RowOrderingKey(100, "standard", STANDARD_MAIN_FILE, 9),
        RowOrderingKey(100, "standard", STANDARD_MAIN_FILE, 3),
    ]
    state = 0
    snapshots: list[tuple[int, int, tuple[int, ...]]] = []
    for time_ms, group in iter_prequential_groups(rows):
        snapshots.append(
            (time_ms, state, tuple(row.source_row_index0 for row in group))
        )
        state += len(group)
    assert snapshots == [(100, 0, (3, 9, 2)), (101, 3, (4,))]


def test_split_boundaries_and_update_only_roles_are_exact() -> None:
    assert (
        classify_row_role(iso_date="2022-04-08", policy="standard")
        == "outside_frozen_protocol"
    )
    assert classify_row_role(iso_date="2022-04-09", policy="standard") == (
        "history_burn_in"
    )
    assert classify_row_role(iso_date="2022-05-02", policy="standard") == "train"
    assert classify_row_role(iso_date="2022-05-03", policy="randomized") == (
        "validation"
    )
    assert classify_row_role(iso_date="2022-05-05", policy="standard") == (
        "validation_state_update_only"
    )
    assert classify_row_role(iso_date="2022-05-06", policy="randomized") == (
        "headline_test"
    )
    assert classify_row_role(iso_date="2022-05-08", policy="standard") == (
        "in_policy_test"
    )
    assert classify_row_role(iso_date="2022-05-02", policy="randomized") == (
        "pre_validation_randomized_excluded"
    )


def test_approved_embargo_rule_is_exact_at_every_cutoff() -> None:
    role_to_cutoff = {
        "history_burn_in": "history_to_train",
        "train": "train_to_validation",
        "validation": "validation_to_test",
        "validation_state_update_only": "validation_to_test",
    }
    for role, cutoff_name in role_to_cutoff.items():
        cutoff = EXPECTED_EMBARGO_CUTOFFS[cutoff_name]
        assert not is_row_embargo_excluded(
            role=role, time_ms=cutoff - 1, cutoffs=EXPECTED_EMBARGO_CUTOFFS
        )
        assert is_row_embargo_excluded(
            role=role, time_ms=cutoff, cutoffs=EXPECTED_EMBARGO_CUTOFFS
        )
    assert not is_row_embargo_excluded(
        role="headline_test",
        time_ms=EXPECTED_EMBARGO_CUTOFFS["validation_to_test"] + 1,
        cutoffs=EXPECTED_EMBARGO_CUTOFFS,
    )
    assert sum(EXPECTED_EMBARGO_EXCLUDED_COUNTS.values()) == 6_639
    assert EXPECTED_POST_EMBARGO_ROLE_COUNTS["history_burn_in"] == 1_139_908


def _date_range(start: date, end: date) -> list[date]:
    result: list[date] = []
    current = start
    while current <= end:
        result.append(current)
        current += timedelta(days=1)
    return result


def _write_opaque_log(
    path: Path, dates: list[date], *, is_rand: int, corrupt_rand: bool = False
) -> None:
    lines = [",".join(EVENT_COLUMNS)]
    for index, current in enumerate(dates):
        timestamp = int(
            datetime(
                current.year, current.month, current.day, 12, tzinfo=UTC
            ).timestamp()
            * 1000
        )
        row = _event_row(
            date=current.strftime("%Y%m%d"),
            time_ms=str(timestamp),
            is_rand=str(1 - is_rand if corrupt_rand and index == 0 else is_rand),
            # These opaque values prove that the phase-1 projection does not
            # parse or validate label/model-feature fields.
            long_view="not-decoded-test-label",
            is_like="not-decoded-feature-update",
        )
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_label_blind_audit_records_only_allowed_coverage(tmp_path: Path) -> None:
    _write_opaque_log(
        tmp_path / STANDARD_HISTORY_FILE,
        _date_range(date(2022, 4, 9), date(2022, 4, 21)),
        is_rand=0,
    )
    _write_opaque_log(
        tmp_path / STANDARD_MAIN_FILE,
        _date_range(date(2022, 4, 22), date(2022, 5, 8)),
        is_rand=0,
    )
    _write_opaque_log(
        tmp_path / RANDOM_MAIN_FILE,
        _date_range(date(2022, 4, 22), date(2022, 5, 8)),
        is_rand=1,
    )
    report = run_label_blind_coverage_audit(
        raw_data_dir=tmp_path,
        contract_path=CONTRACT,
        repository_root=ROOT,
        _allow_unpinned_fixture=True,
    )
    assert report["decoded_raw_columns"] == ["date", "time_ms", "is_rand"]
    assert report["labels_decoded_or_aggregated"] is False
    assert report["model_features_decoded_or_aggregated"] is False
    assert report["test_labels_opened"] is False
    assert report["final_test_view_materialized"] is False
    assert report["test_unlocked"] is False
    assert report["actual_burn_in_dates"] == ["2022-04-09", "2022-04-21"]
    assert report["date_coverage_gate_passed"] is True
    assert report["post_embargo_time_boundary_gate_passed"] is True
    assert report["phase1_gate_passed"] is True
    assert report["total_embargo_excluded_rows"] == 0
    assert report["post_embargo_rows_by_frozen_role"]["headline_test"] == 3
    assert report["post_embargo_rows_by_frozen_role"]["in_policy_test"] == 3


def test_label_blind_audit_rejects_policy_mismatch(tmp_path: Path) -> None:
    _write_opaque_log(
        tmp_path / STANDARD_HISTORY_FILE,
        _date_range(date(2022, 4, 9), date(2022, 4, 21)),
        is_rand=0,
        corrupt_rand=True,
    )
    _write_opaque_log(
        tmp_path / STANDARD_MAIN_FILE,
        _date_range(date(2022, 4, 22), date(2022, 5, 8)),
        is_rand=0,
    )
    _write_opaque_log(
        tmp_path / RANDOM_MAIN_FILE,
        _date_range(date(2022, 4, 22), date(2022, 5, 8)),
        is_rand=1,
    )
    with pytest.raises(CoverageAuditError, match="is_rand/file-policy mismatch"):
        run_label_blind_coverage_audit(
            raw_data_dir=tmp_path,
            contract_path=CONTRACT,
            repository_root=ROOT,
            _allow_unpinned_fixture=True,
        )
