from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dcdn.data.vk_lsvd import (
    DENOMINATOR_ELIGIBLE_FIELDS,
    E_FORMULAS,
    IALS_SIGNALS,
    INTERACTION_SCHEMA,
    ITEMS_METADATA_SCHEMA,
    MODEL_FORBIDDEN_INPUT_FIELDS,
    ORDERED_C_FIELDS,
    ORDERED_C_PLUS_P_FLAT_SCHEMA,
    ORDERED_C_PLUS_P_PLUS_E_FLAT_SCHEMA,
    ORDERED_DENOMINATOR_MASK,
    ORDERED_E_FIELDS,
    ORDERED_NUMERIC_OPERANDS,
    ORDERED_P0_FIELDS,
    ORDERED_V_FIELDS,
    P0_FORMULAS,
    PHASE1_SEMANTIC_WEEKS,
    USERS_METADATA_SCHEMA,
    V_FORMULAS,
    FeatureInputs,
    GroupHistory,
    IALSOperands,
    VKLSVDContractError,
    denominator_mask_sha256,
    derive_feature_row,
    load_feature_contract,
    ordered_schema_sha256,
    phase1_semantic_week_paths,
    positive_abs_q99_scale,
    semantic_week_relative_path,
    stable_row_id,
    validate_feature_contract,
    verify_source_manifest_reference,
)
from scripts.data.vk_lsvd_features import Phase1AuditError, run_phase1_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/data/vk_lsvd/features_v1.yaml"


def _write_hashed(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


def test_persisted_contract_and_source_reference_are_verified() -> None:
    payload = load_feature_contract(CONTRACT)
    assert payload["test_unlocked"] is False
    assert payload["week26_schema_inspected"] is False
    assert payload["resolution_gate"]["unresolved_fields"] == []
    assert payload["resolution_gate"]["first_model_metric_blocked"] is False
    assert (
        verify_source_manifest_reference(payload, repository_root=ROOT)
        == (payload["source_manifest"]["sha256"])
    )


def test_generated_orders_counts_hashes_and_mask_are_exact() -> None:
    payload = load_feature_contract(CONTRACT)
    ordered = payload["ordered_fields"]
    assert ordered["C"] == list(ORDERED_C_FIELDS)
    assert ordered["P0"] == list(ORDERED_P0_FIELDS)
    assert ordered["V"] == list(ORDERED_V_FIELDS)
    assert ordered["P"] == list(ORDERED_NUMERIC_OPERANDS)
    assert ordered["E"] == list(ORDERED_E_FIELDS)
    assert (len(ORDERED_C_FIELDS), len(ORDERED_P0_FIELDS)) == (6, 31)
    assert (len(ORDERED_V_FIELDS), len(ORDERED_E_FIELDS)) == (27, 20)
    assert len(ORDERED_NUMERIC_OPERANDS) == 58
    assert sum(ORDERED_DENOMINATOR_MASK) == 11
    assert payload["schema_hashes"] == {
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


def test_every_feature_has_one_exact_formula_and_regimes_are_disjoint() -> None:
    assert tuple(P0_FORMULAS) == ORDERED_P0_FIELDS
    assert tuple(V_FORMULAS) == ORDERED_V_FIELDS
    assert tuple(E_FORMULAS) == ORDERED_E_FIELDS
    assert not set(ORDERED_C_FIELDS) & set(ORDERED_NUMERIC_OPERANDS)
    assert not set(ORDERED_E_FIELDS) & set(ORDERED_NUMERIC_OPERANDS)
    assert not MODEL_FORBIDDEN_INPUT_FIELDS & (
        set(ORDERED_C_FIELDS) | set(ORDERED_NUMERIC_OPERANDS) | set(ORDERED_E_FIELDS)
    )
    assert all(
        "validation" not in formula and "week26" not in formula
        for formula in P0_FORMULAS.values()
    )


def test_contract_rejects_unlock_reordering_and_mask_mutation() -> None:
    payload = deepcopy(load_feature_contract(CONTRACT))
    payload["test_unlocked"] = True
    with pytest.raises(VKLSVDContractError, match="test_unlocked"):
        validate_feature_contract(payload)

    payload = deepcopy(load_feature_contract(CONTRACT))
    payload["ordered_fields"]["P0"][0:2] = reversed(
        payload["ordered_fields"]["P0"][0:2]
    )
    with pytest.raises(VKLSVDContractError, match="frozen order"):
        validate_feature_contract(payload)

    payload = deepcopy(load_feature_contract(CONTRACT))
    payload["denominator_contract"]["ordered_mask"][0] = False
    with pytest.raises(VKLSVDContractError, match="frozen order"):
        validate_feature_contract(payload)


def test_contract_loader_rejects_duplicate_keys_and_tampering(
    tmp_path: Path,
) -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    duplicate = text.replace(
        "dataset_id: vk_lsvd_up0.01_ir0.01\n",
        "dataset_id: vk_lsvd_up0.01_ir0.01\ndataset_id: conflict\n",
        1,
    )
    duplicate_path = tmp_path / "duplicate.yaml"
    _write_hashed(duplicate_path, duplicate)
    with pytest.raises(VKLSVDContractError, match="duplicate key"):
        load_feature_contract(duplicate_path)

    tampered_path = tmp_path / "tampered.yaml"
    _write_hashed(tampered_path, text)
    tampered_path.write_text(
        text.replace("test_unlocked: false", "test_unlocked: true", 1),
        encoding="utf-8",
    )
    with pytest.raises(VKLSVDContractError, match="sidecar mismatch"):
        load_feature_contract(tampered_path)


def test_week26_semantic_path_fails_before_filesystem_access(tmp_path: Path) -> None:
    sealed = tmp_path / "subsamples/up0.01_ir0.01/test/week_26.parquet"
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(b"not-a-parquet-file-and-must-never-be-opened")
    with pytest.raises(VKLSVDContractError, match="week 26 semantic access"):
        semantic_week_relative_path(26)
    paths = phase1_semantic_week_paths(tmp_path)
    assert len(paths) == 26
    assert all("week_26" not in path.name for path in paths)
    assert sealed.read_bytes() == b"not-a-parquet-file-and-must-never-be-opened"


def _write_interaction_parquet(path: Path, *, extra_column: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "user_id": pa.array([1], type=pa.uint32()),
        "item_id": pa.array([7], type=pa.uint32()),
        "place": pa.array([1], type=pa.uint8()),
        "platform": pa.array([2], type=pa.uint8()),
        "agent": pa.array([3], type=pa.uint8()),
        "timespent": pa.array([4], type=pa.uint8()),
        "like": pa.array([False], type=pa.bool_()),
        "dislike": pa.array([False], type=pa.bool_()),
        "share": pa.array([False], type=pa.bool_()),
        "bookmark": pa.array([False], type=pa.bool_()),
        "click_on_author": pa.array([False], type=pa.bool_()),
        "open_comments": pa.array([False], type=pa.bool_()),
    }
    if extra_column:
        fields["unexpected"] = pa.array([0], type=pa.uint8())
    pq.write_table(pa.table(fields), path)


def _write_synthetic_phase1_tree(
    root: Path, *, mismatch_week: int | None = None
) -> None:
    for week in PHASE1_SEMANTIC_WEEKS:
        relative = semantic_week_relative_path(week)
        _write_interaction_parquet(
            root.joinpath(*relative.parts), extra_column=week == mismatch_week
        )
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array([1], type=pa.uint32()),
                "age": pa.array([2], type=pa.uint8()),
                "gender": pa.array([1], type=pa.uint8()),
                "geo": pa.array([3], type=pa.uint8()),
                "train_interactions_rank": pa.array([1], type=pa.uint32()),
            }
        ),
        metadata / "users_metadata.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "item_id": pa.array([7], type=pa.uint32()),
                "author_id": pa.array([9], type=pa.uint32()),
                "duration": pa.array([157], type=pa.uint8()),
                "train_interactions_rank": pa.array([1], type=pa.uint32()),
            }
        ),
        metadata / "items_metadata.parquet",
    )
    sealed = root / "subsamples/up0.01_ir0.01/test/week_26.parquet"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    sealed.write_bytes(b"opaque-sealed-week-26")


def test_phase1_audit_checks_weeks_0_25_and_never_opens_week26(
    tmp_path: Path,
) -> None:
    _write_synthetic_phase1_tree(tmp_path)
    report = run_phase1_audit(
        raw_root=tmp_path,
        contract_path=CONTRACT,
        repository_root=ROOT,
    )
    assert report["semantic_weeks_opened"] == list(range(26))
    assert report["week26_file_open_attempted"] is False
    assert report["week26_schema_inspected"] is False
    assert report["week26_rows_or_labels_read"] is False
    assert report["interaction_values_decoded"] is False
    assert report["validation_or_test_model_metrics_computed"] is False
    assert report["label_c_seconds"] == 157.0


def test_phase1_audit_rejects_one_schema_mismatch(tmp_path: Path) -> None:
    _write_synthetic_phase1_tree(tmp_path, mismatch_week=25)
    with pytest.raises(Phase1AuditError, match="schema mismatch in week 25"):
        run_phase1_audit(
            raw_root=tmp_path,
            contract_path=CONTRACT,
            repository_root=ROOT,
        )


def _history(*, show_count: int, timespent: float, include_click: bool) -> GroupHistory:
    sources = {
        "like": min(2, show_count),
        "share": min(1, show_count),
        "bookmark": min(1, show_count),
        "open_comments": min(2, show_count),
        "dislike": 0,
        "survival_15": min(4, show_count),
        "survival_30": min(3, show_count),
    }
    if include_click:
        sources["click_on_author"] = min(1, show_count)
    return GroupHistory(show_count, timespent, sources)


def _scales(fields: tuple[str, ...], value: float = 1.0) -> dict[str, float]:
    return dict.fromkeys(fields, value)


def test_row_builder_reproduces_smoothed_rates_averages_and_ials_cosines() -> None:
    inputs = FeatureInputs(
        duration_seconds=60.0,
        item_metadata_present=True,
        user_history=_history(show_count=10, timespent=200.0, include_click=False),
        item_history=_history(show_count=20, timespent=300.0, include_click=True),
        ials={
            signal: IALSOperands(
                dot=6.0, user_norm=2.0, item_norm=3.0, user_seen=True, item_seen=True
            )
            for signal in IALS_SIGNALS
        },
    )
    row = derive_feature_row(
        inputs,
        p0_scales=_scales(ORDERED_P0_FIELDS),
        e_scales=_scales(ORDERED_E_FIELDS),
    )
    p0 = dict(zip(ORDERED_P0_FIELDS, row.raw_p0, strict=True))
    engineered = dict(zip(ORDERED_E_FIELDS, row.raw_e, strict=True))
    assert p0["p_user_like_count_plus_alpha"] == 7.0
    assert p0["p_user_show_count_plus_beta"] == 110.0
    assert engineered["e_user_avg_timespent_seconds"] == 20.0
    assert engineered["e_user_like_rate"] == pytest.approx(7.0 / 110.0)
    assert engineered["e_item_avg_timespent_seconds"] == 15.0
    for signal in IALS_SIGNALS:
        assert engineered[f"e_ials_{signal}_cosine"] == 1.0
    assert len(row.numeric_operands) == len(ORDERED_NUMERIC_OPERANDS)
    assert all(row.denominator_validity)


def test_cold_fallbacks_gate_invalid_denominators_and_record_flooring() -> None:
    inputs = FeatureInputs(
        duration_seconds=0.0,
        item_metadata_present=False,
        user_history=_history(show_count=0, timespent=0.0, include_click=False),
        item_history=_history(show_count=0, timespent=0.0, include_click=True),
        ials={
            signal: IALSOperands(
                dot=0.0,
                user_norm=0.0,
                item_norm=0.0,
                user_seen=False,
                item_seen=False,
            )
            for signal in IALS_SIGNALS
        },
    )
    p_scales = _scales(ORDERED_P0_FIELDS)
    p_scales["p_user_show_count_plus_beta"] = 20_000.0
    p_scales["p_item_show_count_plus_beta"] = 20_000.0
    row = derive_feature_row(
        inputs,
        p0_scales=p_scales,
        e_scales=_scales(ORDERED_E_FIELDS),
    )
    p0 = dict(zip(ORDERED_P0_FIELDS, row.raw_p0, strict=True))
    engineered = dict(zip(ORDERED_E_FIELDS, row.raw_e, strict=True))
    validity = dict(
        zip(ORDERED_NUMERIC_OPERANDS, row.denominator_validity, strict=True)
    )
    indicators = dict(zip(ORDERED_V_FIELDS, row.validity_indicators, strict=True))
    assert p0["p_user_show_count_plus_beta"] == 100.0
    assert engineered["e_user_like_rate"] == 0.05
    assert validity["p_duration_seconds"] is False
    assert validity["p_user_show_count"] is False
    assert validity["p_user_show_count_plus_beta"] is True
    assert validity["p_item_show_count_plus_beta"] is True
    for field in DENOMINATOR_ELIGIBLE_FIELDS[5:]:
        assert validity[field] is False
    assert indicators["v_user_show_count_plus_beta_reciprocal_floor_active"] == 1.0
    assert indicators["v_item_show_count_plus_beta_reciprocal_floor_active"] == 1.0


def test_row_builder_rejects_unseen_nonzero_factors_and_bad_scale() -> None:
    operands = {
        signal: IALSOperands(
            dot=0.0, user_norm=0.0, item_norm=0.0, user_seen=False, item_seen=False
        )
        for signal in IALS_SIGNALS
    }
    operands["viewtime"] = IALSOperands(
        dot=1.0, user_norm=0.0, item_norm=0.0, user_seen=False, item_seen=False
    )
    inputs = FeatureInputs(
        duration_seconds=60.0,
        item_metadata_present=True,
        user_history=_history(show_count=1, timespent=1.0, include_click=False),
        item_history=_history(show_count=1, timespent=1.0, include_click=True),
        ials=operands,
    )
    with pytest.raises(VKLSVDContractError, match="unseen iALS users"):
        derive_feature_row(
            inputs,
            p0_scales=_scales(ORDERED_P0_FIELDS),
            e_scales=_scales(ORDERED_E_FIELDS),
        )

    operands["viewtime"] = IALSOperands(
        dot=0.0, user_norm=0.0, item_norm=0.0, user_seen=False, item_seen=False
    )
    bad_scales = _scales(ORDERED_P0_FIELDS)
    bad_scales["p_duration_seconds"] = 0.0
    with pytest.raises(VKLSVDContractError, match="scales must be"):
        derive_feature_row(
            FeatureInputs(
                duration_seconds=60.0,
                item_metadata_present=True,
                user_history=inputs.user_history,
                item_history=inputs.item_history,
                ials=operands,
            ),
            p0_scales=bad_scales,
            e_scales=_scales(ORDERED_E_FIELDS),
        )


def test_reference_q99_scale_and_stable_row_identity() -> None:
    assert positive_abs_q99_scale([0.0, -1.0, 2.0, 100.0]) == 100.0
    assert positive_abs_q99_scale([0.0]) == 1e-12
    first = stable_row_id(week=20, source_row_index0=7)
    assert first == stable_row_id(week=20, source_row_index0=7)
    assert first != stable_row_id(week=20, source_row_index0=8)
    assert first != stable_row_id(week=21, source_row_index0=7)
    with pytest.raises(VKLSVDContractError, match="source_row_index0"):
        stable_row_id(week=20, source_row_index0=-1)


def test_schema_constants_match_expected_arrow_spellings() -> None:
    assert INTERACTION_SCHEMA[-1] == "open_comments:bool"
    assert USERS_METADATA_SCHEMA[0] == "user_id:uint32"
    assert ITEMS_METADATA_SCHEMA[2] == "duration:uint8"
