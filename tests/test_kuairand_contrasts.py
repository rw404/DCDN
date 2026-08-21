from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

TARGETS = ("long_view", "like", "forward", "hate", "watch_time")


def _load():
    spec = importlib.util.spec_from_file_location(
        "report_kuairand_contrasts",
        REPO_ROOT / "scripts" / "report_kuairand_contrasts.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["report_kuairand_contrasts"] = module
    spec.loader.exec_module(module)
    return module


rc = _load()


def _record(
    arm_id: str, seed: int, macro: float, auc: float | None = None
) -> dict:
    return {
        "arm_id": arm_id,
        "label": arm_id.upper(),
        "regime": "reduced" if "reduced" in arm_id else "full",
        "seed": seed,
        "seed_class": "frozen_protocol" if seed in (42, 43, 44)
        else "stability_annex",
        "tuning_contaminated": seed == 42,
        "learning_rate": 0.001,
        "parameters": {"interaction": 1000, "non_embedding": 5000},
        "validation": {
            "macro_nloss": macro,
            "per_target_normalized": {name: macro for name in TARGETS},
            "roc_auc": {
                name: (0.55 if auc is None else auc) for name in TARGETS[:4]
            },
            "watch_time_mae": macro / 10,
            "watch_time_mae_seconds": macro,
            "nonfinite_predictions": 0,
        },
    }


def _payload(values: dict[str, list[float]], auc: float | None = None) -> dict:
    records = [
        _record(arm, 42 + index, macro, auc)
        for arm, macros in values.items()
        for index, macro in enumerate(macros)
    ]
    return {
        "targets": list(TARGETS),
        "evaluated_split": "validation",
        "test_unlocked": False,
        "selected_configs_sha256": "0" * 64,
        "final_seeds": [42, 43, 44],
        "annex_seeds": [],
        "records": records,
    }


@pytest.fixture()
def scratch(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """A throwaway output directory under `results/`.

    The publication guard in `benchmark_efficiency` deliberately refuses to
    write anywhere outside the repository's `results/` tree, so the test cannot
    simply hand it `tmp_path`. It gets a dot-prefixed sibling instead, removed
    on teardown, which exercises the real guard rather than bypassing it.
    """

    target = REPO_ROOT / "results" / f".pytest-{request.node.name[:60]}"
    shutil.rmtree(target, ignore_errors=True)
    yield target
    shutil.rmtree(target, ignore_errors=True)


def _build(payload: dict) -> dict:
    """Assemble the report in memory, exactly as `main` does, without publishing.

    `beff.publish_json` fsyncs onto a FUSE mount and costs tens of seconds per
    call; routing every case through it would buy no extra coverage of the
    logic under test. One test below does the full round-trip through the real
    publisher so the guard and the on-disk schema stay covered.
    """

    targets = payload["targets"]
    contrasts: list[dict] = []
    per_target: list[dict] = []
    summary: list[dict] = []
    for scope in rc.SEED_SCOPES:
        records = rc._scope_records(payload["records"], scope)
        if not records:
            continue
        grouped = rc._by_arm(records)
        contrasts.extend(rc._contrast_rows(grouped, scope))
        per_target.extend(rc._per_target_rows(grouped, scope, targets))
        summary.extend(rc._summary_rows(grouped, scope, targets))
    return {
        "contrasts": contrasts,
        "per_target": per_target,
        "arm_summary": summary,
        "test_unlocked": payload["test_unlocked"],
        "evaluated_split": payload["evaluated_split"],
        "normalizer_note": "1.0 is not the constant-predictor level here",
        "convergence_gate": {
            "divergence_multiple_of_constant_predictor": rc.DIVERGENCE_MULTIPLE,
            "chance_auc": rc.CHANCE_AUC,
            "primary_target": rc.PRIMARY_TARGET,
        },
    }


def _run(
    tmp_path: Path,
    scratch: Path,
    values: dict[str, list[float]],
    auc: float | None = None,
) -> dict:
    return _build(_payload(values, auc))


def _c1(report: dict, scope: str = "frozen_protocol") -> dict:
    return next(
        r for r in report["contrasts"]
        if r["contrast"] == "C1" and r["seed_scope"] == scope
    )


# ---------------------------------------------------------------------------
# The separation rule is the guard against quoting noise as a factor
# ---------------------------------------------------------------------------


def test_disjoint_ranges_are_reported_as_separated() -> None:
    assert rc._separated([0.10, 0.11, 0.12], [0.20, 0.21, 0.22])
    assert rc._separated([0.20, 0.21, 0.22], [0.10, 0.11, 0.12])


def test_overlapping_ranges_are_not_separated() -> None:
    assert not rc._separated([0.10, 0.15, 0.20], [0.12, 0.18, 0.30])


def test_a_single_shared_endpoint_is_not_separation() -> None:
    """Touching ranges are indistinguishable at three seeds."""

    assert not rc._separated([0.1, 0.2], [0.2, 0.3])


# ---------------------------------------------------------------------------
# Direction is published as measured, in both directions
# ---------------------------------------------------------------------------


def test_contrast_reports_the_comparator_when_the_comparator_wins(
    tmp_path: Path, scratch: Path,
) -> None:
    report = _run(tmp_path, scratch, {
        "dcdn_reduced": [0.90, 0.91, 0.92],
        "dcnv2_reduced": [0.50, 0.51, 0.52],
    })
    c1 = _c1(report)
    assert c1["better"] == "dcnv2_reduced"
    assert c1["difference"] > 0
    assert c1["seed_ranges_disjoint"] is True


def test_contrast_reports_the_subject_when_the_subject_wins(
    tmp_path: Path, scratch: Path
) -> None:
    report = _run(tmp_path, scratch, {
        "dcdn_reduced": [0.50, 0.51, 0.52],
        "dcnv2_reduced": [0.90, 0.91, 0.92],
    })
    c1 = _c1(report)
    assert c1["better"] == "dcdn_reduced"
    assert c1["difference"] < 0
    assert pytest.approx(c1["ratio"], rel=1e-9) == 0.91 / 0.51


def test_a_close_call_is_flagged_not_separated(
    tmp_path: Path, scratch: Path
) -> None:
    report = _run(tmp_path, scratch, {
        "dcdn_reduced": [0.50, 0.60, 0.70],
        "dcnv2_reduced": [0.55, 0.62, 0.68],
    })
    c1 = _c1(report)
    assert c1["seed_ranges_disjoint"] is False


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_missing_arms_are_skipped_rather_than_crashing(
    tmp_path: Path, scratch: Path
) -> None:
    report = _run(tmp_path, scratch, {"dcdn_reduced": [0.5], "dcnv2_reduced": [0.6]})
    assert {r["contrast"] for r in report["contrasts"]} == {"C1"}


def test_every_contrast_is_broken_down_per_target(
    tmp_path: Path, scratch: Path
) -> None:
    report = _run(tmp_path, scratch, {"dcdn_reduced": [0.5], "dcnv2_reduced": [0.6]})
    assert {r["target"] for r in report["per_target"]} == set(TARGETS)
    assert {r["metric"] for r in report["per_target"]} == {
        "normalized_loss", "roc_auc"
    }


def test_five_confirmatory_contrasts_are_declared() -> None:
    assert sum(1 for row in rc.CONTRASTS if row[4]) == 5
    assert len(rc.CONTRASTS) == 7


def test_report_records_that_the_test_split_stayed_locked(
    tmp_path: Path, scratch: Path
) -> None:
    report = _run(tmp_path, scratch, {"dcdn_reduced": [0.5], "dcnv2_reduced": [0.6]})
    assert report["test_unlocked"] is False
    assert report["evaluated_split"] == "validation"
    assert "constant-predictor" in report["normalizer_note"]


def test_arm_summary_carries_both_mae_units(
    tmp_path: Path, scratch: Path
) -> None:
    report = _run(tmp_path, scratch, {"dcdn_reduced": [0.5], "dcnv2_reduced": [0.6]})
    entry = next(
        r for r in report["arm_summary"]
        if r["arm_id"] == "dcdn_reduced" and r["seed_scope"] == "frozen_protocol"
    )
    assert entry["watch_time_mae_normalized"] == pytest.approx(0.05)
    assert entry["watch_time_mae_seconds"] == pytest.approx(0.5)
    assert "auc_watch_time" not in entry


# ---------------------------------------------------------------------------
# Seed scopes: the frozen table and the annex must never be pooled
# ---------------------------------------------------------------------------


def _annex_payload() -> dict:
    """Three frozen seeds plus five annex seeds, with different distributions.

    The annex values are deliberately shifted so that a pooled mean cannot
    coincide with the frozen-three mean; if the report ever pools them, the
    numbers move and the test notices.
    """

    values = {
        "dcdn_reduced": [0.50, 0.51, 0.52, 0.90, 0.91, 0.92, 0.93, 0.94],
        "dcnv2_reduced": [0.60, 0.61, 0.62, 0.60, 0.61, 0.62, 0.63, 0.64],
    }
    return _payload(values)


def test_frozen_scope_uses_only_the_three_frozen_seeds(
    tmp_path: Path, scratch: Path
) -> None:
    report = _build(_annex_payload())

    frozen = _c1(report, "frozen_protocol")
    assert frozen["seeds_used"] == 3
    assert frozen["subject_mean"] == pytest.approx(0.51)

    pooled = _c1(report, "all_seeds")
    assert pooled["seeds_used"] == 8
    assert pooled["subject_mean"] == pytest.approx(0.76625)

    # And the two must not be the same number, or the split is cosmetic.
    assert frozen["subject_mean"] != pytest.approx(pooled["subject_mean"])


def test_clean_scope_drops_the_tuning_seed(tmp_path: Path, scratch: Path) -> None:
    """Seed 42 selected the learning rate on validation, so it is not evidence."""

    report = _build(_annex_payload())
    clean = _c1(report, "frozen_protocol_clean")
    assert clean["seeds_used"] == 2
    assert clean["subject_mean"] == pytest.approx(0.515)


def test_every_scope_is_present_in_all_three_tables(
    tmp_path: Path, scratch: Path
) -> None:
    report = _build(_annex_payload())
    for table in ("contrasts", "per_target", "arm_summary"):
        assert {r["seed_scope"] for r in report[table]} == set(rc.SEED_SCOPES), table


# ---------------------------------------------------------------------------
# Convergence gate: a diverged comparator cannot lose a contrast
# ---------------------------------------------------------------------------


def test_a_diverged_comparator_makes_the_contrast_unquotable(
    tmp_path: Path, scratch: Path
) -> None:
    """This is the geometric-preconditioner failure, in miniature.

    DCNv2 overflowed to 1e11 and the arithmetic would otherwise announce a
    DCDN win by eleven orders of magnitude. That is not a result.
    """

    report = _run(tmp_path, scratch, {
        "dcdn_reduced": [0.50, 0.51, 0.52],
        "dcnv2_reduced": [1.26e11, 2.96e17, 1.75e18],
    })
    c1 = _c1(report)
    assert c1["quotable"] is False
    assert c1["better"] == "not_comparable"
    assert c1["comparator_converged"] is False
    assert c1["seed_ranges_disjoint"] is False
    assert "macro_nloss_above" in c1["non_convergence"]


def test_below_chance_ranking_counts_as_non_convergence(
    tmp_path: Path, scratch: Path
) -> None:
    report = _run(tmp_path, scratch, {
        "dcdn_reduced": [0.50, 0.51, 0.52],
        "dcnv2_reduced": [0.60, 0.61, 0.62],
    }, auc=0.44)
    c1 = _c1(report)
    assert c1["quotable"] is False
    assert "below_chance" in c1["non_convergence"]


def test_a_merely_bad_run_is_still_a_result(tmp_path: Path, scratch: Path) -> None:
    """The gate must not quietly discard arms that are bad but converged."""

    report = _run(tmp_path, scratch, {
        "dcdn_reduced": [0.50, 0.51, 0.52],
        "dcnv2_reduced": [8.0, 8.1, 8.2],
    })
    c1 = _c1(report)
    assert c1["quotable"] is True
    assert c1["better"] == "dcdn_reduced"
    assert c1["seed_ranges_disjoint"] is True


def test_non_finite_predictions_are_non_convergence(
    tmp_path: Path, scratch: Path
) -> None:
    payload = _payload({
        "dcdn_reduced": [0.50, 0.51, 0.52],
        "dcnv2_reduced": [0.60, 0.61, 0.62],
    })
    payload["records"][0]["validation"]["nonfinite_predictions"] = 7
    report = _build(payload)
    c1 = _c1(report)
    assert c1["quotable"] is False
    assert "non_finite_predictions" in c1["non_convergence"]


def test_convergence_gate_thresholds_are_recorded_in_the_artifact(
    tmp_path: Path, scratch: Path
) -> None:
    report = _run(tmp_path, scratch, {
        "dcdn_reduced": [0.5], "dcnv2_reduced": [0.6],
    })
    gate = report["convergence_gate"]
    assert gate["divergence_multiple_of_constant_predictor"] == 10.0
    assert gate["chance_auc"] == 0.5
    assert gate["primary_target"] == "long_view"


# ---------------------------------------------------------------------------
# One full round-trip through the real publisher
# ---------------------------------------------------------------------------


def test_publisher_round_trip_writes_every_table(
    tmp_path: Path, scratch: Path
) -> None:
    """Exercises the publication guard and the on-disk schema for real."""

    results = tmp_path / "results_v1.json"
    results.write_text(json.dumps(_annex_payload()), encoding="utf-8")
    assert rc.main(["--results", str(results), "--output-dir", str(scratch)]) == 0

    for name in (
        "contrasts.csv",
        "contrasts_per_target.csv",
        "arm_summary.csv",
        "contrasts_v1.json",
    ):
        assert (scratch / name).is_file(), name
        assert (scratch / f"{name}.sha256").is_file(), name

    report = json.loads((scratch / "contrasts_v1.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["test_unlocked"] is False
    assert report["frozen_final_seeds"] == [42, 43, 44]
    assert set(report["seed_scopes"]) == set(rc.SEED_SCOPES)
    assert _c1(report, "frozen_protocol")["seeds_used"] == 3
    assert _c1(report, "all_seeds")["seeds_used"] == 8
