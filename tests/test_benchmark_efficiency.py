from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    """Load the benchmark script by path; ``scripts`` is not an import target."""

    spec = importlib.util.spec_from_file_location(
        "benchmark_efficiency", REPO_ROOT / "scripts" / "benchmark_efficiency.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_efficiency"] = module
    spec.loader.exec_module(module)
    return module


beff = _load_module()


@pytest.fixture(scope="module")
def quotient():
    return next(g for g in beff.GEOMETRIES if g.geometry_id == "quotient")


@pytest.fixture(scope="module")
def ranker():
    return next(g for g in beff.GEOMETRIES if g.geometry_id == "ranker")


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------


def test_protocol_constants_match_the_frozen_matrix() -> None:
    assert beff.BATCHES == (1, 64, 1024)
    assert beff.WARMUPS == 200
    assert beff.FORWARDS == 1000
    assert beff.ROUNDS == 5
    assert beff.RUNTIME_MATCH_LATENCY_FACTOR == 1.10
    assert beff.RUNTIME_MATCH_THROUGHPUT_FACTOR == 0.90


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_quotient_geometry_matches_the_rq1_benchmark(quotient) -> None:
    assert quotient.bypass_dim() == 0
    assert quotient.operand_dim("reduced") == 8
    assert quotient.operand_dim("full") == 11
    assert quotient.num_targets == 3


def test_ranker_geometry_matches_the_declared_feature_widths(ranker) -> None:
    # features_multitask_v1: 6 C fields, 96 bypass, 61 P_shared, 78 with R_div.
    assert ranker.bypass_dim() == 96
    assert ranker.operand_dim("reduced") == 61
    assert ranker.operand_dim("full") == 78
    assert ranker.num_fields("reduced") == 67
    assert ranker.num_fields("full") == 84
    assert ranker.num_targets == 4


def test_ranker_geometry_reads_no_real_data(ranker) -> None:
    assert ranker.reads_real_data is False


def test_only_the_trained_geometry_may_join_to_quality(quotient, ranker) -> None:
    """Cost and quality share axes only where the same models produced both."""

    assert quotient.quality_source_compatible is True
    assert ranker.quality_source_compatible is False


def test_pareto_omits_geometries_that_were_never_trained() -> None:
    quality = {"dcdn_reduced": {"id": 0.5}, "dcnv2_reduced": {"id": 0.9}}
    row = {
        1: {"batch_latency_ms_p95": 1.0, "device_latency_ms_p95": 1.0},
        1024: {"throughput_examples_per_second": 100.0},
    }
    measurements = {
        "quotient": {"dcdn_reduced": row, "dcnv2_reduced": row},
        "ranker": {"dcdn_reduced": row, "dcnv2_reduced": row},
    }
    counts = {"non_embedding": 10}
    parameters = {
        g: {"dcdn_reduced": counts, "dcnv2_reduced": counts}
        for g in ("quotient", "ranker")
    }
    flops = {
        g: {"dcdn_reduced": 1, "dcnv2_reduced": 2} for g in ("quotient", "ranker")
    }
    rows = beff._pareto_rows(
        measurements, parameters, flops, quality, {}, beff.GEOMETRIES
    )
    assert rows
    assert {r["geometry"] for r in rows} == {"quotient"}


def test_denominator_mask_width_equals_operand_width(quotient, ranker) -> None:
    for geometry in (quotient, ranker):
        for regime in ("reduced", "full"):
            mask = geometry.denominator_mask(regime)
            assert len(mask) == geometry.operand_dim(regime)


def test_denominator_count_is_constant_across_regimes(quotient, ranker) -> None:
    """R_div columns are appended in the full regime and are never denominators."""

    for geometry in (quotient, ranker):
        reduced = sum(geometry.denominator_mask("reduced"))
        full = sum(geometry.denominator_mask("full"))
        assert reduced == full == geometry.denominator_count


def test_appended_full_columns_are_not_denominators(ranker) -> None:
    mask = ranker.denominator_mask("full")
    appended = mask[ranker.operand_dim("reduced") :]
    assert appended and not any(appended)


def test_spec_binds_the_mask_to_the_operand_width(ranker) -> None:
    spec = ranker.spec("reduced")
    assert len(spec.denominator_mask) == spec.operand_dim == 61
    assert spec.bypass_dim == 96
    assert spec.num_fields == 67


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def test_matrix_arm_ids_match_the_published_rq1_order(quotient) -> None:
    """Cost rows must join to the published quality table without a mapping."""

    capacity = beff.resolve_capacity(quotient)
    arms = beff.build_arms(quotient, capacity)
    matrix = [a.arm_id for a in arms if a.tier in ("matrix", "floor")]
    assert matrix == [
        "linear",
        "dcnv2_reduced",
        "dcdn_reduced",
        "afn_reduced",
        "dcnv2_capacity_reduced",
        "dcnv2_full",
        "dcdn_full",
        "afn_full",
    ]


def test_full_regime_arms_use_the_full_operand_width(quotient) -> None:
    capacity = beff.resolve_capacity(quotient)
    arms = {a.arm_id: a for a in beff.build_arms(quotient, capacity)}
    for arm_id in ("dcnv2_full", "dcdn_full", "afn_full"):
        assert arms[arm_id].regime == "full"
    for arm_id in ("dcnv2_reduced", "dcdn_reduced", "afn_reduced"):
        assert arms[arm_id].regime == "reduced"


def test_every_arm_emits_one_column_per_target(quotient) -> None:
    capacity = beff.resolve_capacity(quotient)
    for arm in beff.build_arms(quotient, capacity):
        model = beff.instantiate(arm, quotient)
        cat, ops, val = beff._make_inputs(
            quotient, arm.regime, 4, torch.device("cpu")
        )
        assert model(cat, ops, val).shape == (4, quotient.num_targets)


def test_batch_of_one_survives_every_arm(ranker) -> None:
    capacity = beff.resolve_capacity(ranker)
    for arm in beff.build_arms(ranker, capacity):
        model = beff.instantiate(arm, ranker)
        cat, ops, val = beff._make_inputs(ranker, arm.regime, 1, torch.device("cpu"))
        assert model(cat, ops, val).shape == (1, ranker.num_targets)


def test_encoder_is_identical_across_arms_of_one_geometry(ranker) -> None:
    """Fairness: the shared front end must not vary with the interaction block."""

    capacity = beff.resolve_capacity(ranker)
    arms = {a.arm_id: a for a in beff.build_arms(ranker, capacity)}
    left = beff.instantiate(arms["dcnv2_reduced"], ranker)
    right = beff.instantiate(arms["dcdn_reduced"], ranker)
    for a, b in zip(
        left.encoder.parameters(), right.encoder.parameters(), strict=True
    ):
        assert torch.equal(a, b)


def test_head_is_identical_across_arms_of_one_geometry(ranker) -> None:
    capacity = beff.resolve_capacity(ranker)
    arms = {a.arm_id: a for a in beff.build_arms(ranker, capacity)}
    left = beff.instantiate(arms["dcnv2_reduced"], ranker)
    right = beff.instantiate(arms["dcdn_reduced"], ranker)
    for a, b in zip(
        left.model.head.parameters(), right.model.head.parameters(), strict=True
    ):
        assert torch.equal(a, b)


def test_construction_is_deterministic(quotient) -> None:
    capacity = beff.resolve_capacity(quotient)
    arm = next(
        a for a in beff.build_arms(quotient, capacity) if a.arm_id == "dcdn_reduced"
    )
    left = beff.instantiate(arm, quotient)
    right = beff.instantiate(arm, quotient)
    for a, b in zip(left.parameters(), right.parameters(), strict=True):
        assert torch.equal(a, b)


def test_non_embedding_count_excludes_embedding_tables(ranker) -> None:
    capacity = beff.resolve_capacity(ranker)
    arm = next(
        a for a in beff.build_arms(ranker, capacity) if a.arm_id == "dcdn_reduced"
    )
    counts = beff.instantiate(arm, ranker).parameter_counts()
    # Six tables of 100k x 16 dominate the total but not the non-embedding count.
    assert counts["total"] > 9_000_000
    assert counts["non_embedding"] < 100_000


def test_dcdn_core_is_smaller_than_dcnv2_at_ranker_width(ranker) -> None:
    """DCDN crosses P only; DCNv2 crosses the full flat width."""

    capacity = beff.resolve_capacity(ranker)
    arms = {a.arm_id: a for a in beff.build_arms(ranker, capacity)}
    dcdn = beff.instantiate(arms["dcdn_reduced"], ranker).parameter_counts()
    dcnv2 = beff.instantiate(arms["dcnv2_reduced"], ranker).parameter_counts()
    assert dcdn["interaction"] < dcnv2["interaction"]


# ---------------------------------------------------------------------------
# Capacity and FLOPs
# ---------------------------------------------------------------------------


def test_capacity_control_lands_inside_tolerance(quotient, ranker) -> None:
    for geometry in (quotient, ranker):
        resolved = beff.resolve_capacity(geometry)["reduced"]
        assert 0.95 <= resolved["parameter_ratio"] <= 1.05
        assert 0.95 <= resolved["flop_ratio"] <= 1.05


def test_capacity_resolution_is_deterministic(ranker) -> None:
    assert beff.resolve_capacity(ranker) == beff.resolve_capacity(ranker)


def test_flops_reported_only_where_a_frozen_formula_exists(quotient) -> None:
    capacity = beff.resolve_capacity(quotient)
    analytic = beff.analytic_counts(quotient)
    arms = {a.arm_id: a for a in beff.build_arms(quotient, capacity)}
    for arm_id in (
        "dcdn_reduced",
        "dcdn_full",
        "dcnv2_reduced",
        "dcnv2_full",
        "dcnv2_capacity_reduced",
        "dcdn_reduced_l2",
    ):
        assert isinstance(beff.arm_flops(arms[arm_id], quotient, analytic), int)
    for arm_id in ("afn_reduced", "dnn_reduced", "linear"):
        assert (
            beff.arm_flops(arms[arm_id], quotient, analytic)
            == "unavailable_no_frozen_formula"
        )


def test_deeper_dcdn_costs_more_flops(quotient) -> None:
    analytic = beff.analytic_counts(quotient)
    assert (
        analytic["reduced"]["dcdn_l2"]["flops"]
        > analytic["reduced"]["dcdn_l1"]["flops"]
    )


# ---------------------------------------------------------------------------
# Statistics and verdicts
# ---------------------------------------------------------------------------


def test_percentile_matches_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert beff._percentile(values, 0.0) == 1.0
    assert beff._percentile(values, 1.0) == 4.0
    assert beff._percentile(values, 0.5) == pytest.approx(2.5)


def _row(p50: float, p95: float, throughput: float) -> dict:
    return {
        "batch_latency_ms_p50": p50,
        "batch_latency_ms_p95": p95,
        "throughput_examples_per_second": throughput,
    }


def test_runtime_match_requires_both_thresholds() -> None:
    comparator = {1: _row(1.0, 1.0, 100.0), 1024: _row(1.0, 1.0, 100.0)}
    passing = {1: _row(1.05, 1.05, 100.0), 1024: _row(1.0, 1.0, 95.0)}
    verdict = beff.runtime_match_verdict(passing, comparator, 1024)
    assert verdict["runtime_matched"] is True

    slow = {1: _row(1.5, 1.5, 100.0), 1024: _row(1.0, 1.0, 95.0)}
    slow_verdict = beff.runtime_match_verdict(slow, comparator, 1024)
    assert slow_verdict["runtime_matched"] is False

    starved = {1: _row(1.0, 1.0, 100.0), 1024: _row(1.0, 1.0, 50.0)}
    starved_verdict = beff.runtime_match_verdict(starved, comparator, 1024)
    assert starved_verdict["runtime_matched"] is False


def test_failed_verdict_forbids_the_same_performance_phrase() -> None:
    comparator = {1: _row(1.0, 1.0, 100.0), 1024: _row(1.0, 1.0, 100.0)}
    slow = {1: _row(2.0, 2.0, 100.0), 1024: _row(1.0, 1.0, 40.0)}
    verdict = beff.runtime_match_verdict(slow, comparator, 1024)
    assert "not permitted" in verdict["reporting_requirement"]


def test_contamination_check_flags_a_dirty_wall_clock_tail() -> None:
    clean = {"g": {"a": {1: {"wall_over_device_p95": 1.1}}}}
    dirty = {"g": {"a": {1: {"wall_over_device_p95": 9.0}}}}
    assert beff.contamination_check(clean)["wall_clock_tail_is_clean"] is True
    assert beff.contamination_check(dirty)["wall_clock_tail_is_clean"] is False


def test_contamination_check_reports_unavailable_on_cpu() -> None:
    assert beff.contamination_check({"g": {"a": {1: {}}}})["available"] is False


def test_verdict_status_is_decided_by_device_agreement_not_load() -> None:
    """A busy host does not invalidate a run whose device time agrees."""

    idle = {"host_idle": True}
    busy = {"host_idle": False}
    clean = {"available": True, "wall_clock_tail_is_clean": True}
    dirty = {"available": True, "wall_clock_tail_is_clean": False}

    assert beff._verdict_status(idle, clean) == "final_idle_host"
    assert (
        beff._verdict_status(busy, clean)
        == "final_load_gate_waived_by_device_time_agreement"
    )
    for state in (idle, busy):
        assert beff._verdict_status(state, dirty).startswith("provisional")
    assert beff._verdict_status(idle, {"available": False}).startswith("provisional")


# ---------------------------------------------------------------------------
# Publication discipline
# ---------------------------------------------------------------------------


def test_publish_writes_exact_bytes_and_a_verifiable_sidecar(tmp_path) -> None:
    target = tmp_path / "artifact.json"
    digest = beff.publish_json(target, {"b": 2, "a": 1})
    data = target.read_bytes()
    # Canonical: sorted keys, indent 2, one trailing newline.
    assert data == b'{\n  "a": 1,\n  "b": 2\n}\n'
    sidecar = (tmp_path / "artifact.json.sha256").read_text()
    assert sidecar == f"{digest}  artifact.json\n"
    # The sidecar must hash the published bytes including the trailing newline.
    import hashlib

    assert hashlib.sha256(data).hexdigest() == digest


def test_publish_refuses_to_overwrite(tmp_path) -> None:
    target = tmp_path / "artifact.json"
    beff.publish_json(target, {"a": 1})
    with pytest.raises(FileExistsError):
        beff.publish_json(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 1}


def test_publish_rejects_a_sidecar_path(tmp_path) -> None:
    with pytest.raises(ValueError, match="SHA sidecar"):
        beff.publish_bytes(tmp_path / "x.sha256", b"data")


def test_publish_csv_quotes_and_terminates(tmp_path) -> None:
    target = tmp_path / "rows.csv"
    beff.publish_csv(target, [{"a": "x,y", "b": 1}, {"a": None, "b": 2}])
    assert target.read_text() == 'a,b\n"x,y",1\n,2\n'


def test_prepare_output_dir_refuses_an_existing_directory(tmp_path) -> None:
    results = REPO_ROOT / "results"
    assert results.exists()
    with pytest.raises(beff.PublicationError, match="fresh output directory"):
        beff.prepare_output_dir(results / "multitask")


def test_prepare_output_dir_refuses_outside_results(tmp_path) -> None:
    with pytest.raises(beff.PublicationError, match="below repository results/"):
        beff.prepare_output_dir(tmp_path / "elsewhere")
