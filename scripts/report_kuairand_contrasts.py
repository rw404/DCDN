#!/usr/bin/env python
"""Compute the seven frozen contrasts from the KuaiRand-Pure result bundle.

Reads a run's ``results_v1.json`` and emits the contrast table
``configs/matrix_multitask_v1.yaml`` declares, plus the per-target breakdown the
objective requires alongside every aggregate.

Four reporting rules are enforced here rather than left to prose:

* **Direction is published as measured.** Every contrast appears with its sign,
  whether or not it favours DCDN.
* **The frozen table and the stability annex never mix.** The frozen matrix
  fixes ``final_seeds: [42, 43, 44]``. Extra seeds are run to see whether a
  contrast survives more evidence, but pooling them into one mean would publish
  a post-hoc eight-seed number under the frozen protocol's label. Every row is
  therefore emitted once per seed scope, tagged.
* **A non-convergent arm cannot win or lose a contrast.** An arm whose loss is
  an order of magnitude worse than the frozen constant predictor, or that ranks
  below chance, or that produced non-finite predictions, did not produce a
  result to compare against. Without this gate the arithmetic happily reports
  "DCDN beats DCNv2 by 8.2e11" when what actually happened is that DCNv2
  overflowed - a spectacular-looking win that is pure divergence in the
  comparator.
* **A difference is only called established when the per-seed ranges are
  disjoint** *and* separation survives dropping the tuning seed. With three
  seeds a mean difference inside overlapping ranges is not evidence.

The KuaiRand validation split is the *randomized* exposure pool while train is
the standard policy, so the frozen train-fitted task constants do not normalize
validation to 1.0. Cross-arm comparison stays valid; the absolute level against
1.0 does not, and the report says so.

Usage::

    python scripts/report_kuairand_contrasts.py --results <bundle>/results_v1.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beff = _load("benchmark_efficiency", REPO_ROOT / "scripts" / "benchmark_efficiency.py")

#: The frozen contrast set. `confirmatory` marks the five that carry the
#: multiplicity correction on the primary target.
CONTRASTS = (
    ("C1", "dcdn_reduced", "dcnv2_reduced",
     "DCDN vs DCNv2 on identical reduced inputs", True),
    ("C2", "dcdn_reduced", "afn_reduced",
     "DCDN vs AFN+ on identical reduced inputs", True),
    ("C3", "dcdn_reduced", "dcnv2_capacity_reduced",
     "reciprocal structure vs matched capacity", True),
    ("C4", "dcnv2_full", "dcnv2_reduced",
     "predictive value of the removed manual divisions", True),
    ("C5", "dcdn_reduced", "dcnv2_full",
     "can reduced DCDN replace full DCNv2", True),
    ("C6", "dcdn_full", "dcnv2_full",
     "honest full-feature comparison, DCDN vs DCNv2", False),
    ("C7", "afn_full", "dcnv2_full",
     "honest full-feature comparison, AFN+ vs DCNv2", False),
)

#: A run this far above the frozen constant-predictor level has not learned the
#: task; it has diverged. Ten times is deliberately loose - the divergences this
#: catches are at 1e11 and beyond - so no merely-bad-but-real run is discarded.
DIVERGENCE_MULTIPLE = 10.0

#: Ranking below chance on the primary confirmatory target is the second
#: independent symptom of a run that did not converge.
CHANCE_AUC = 0.5

PRIMARY_TARGET = "long_view"

SEED_SCOPES = ("frozen_protocol", "frozen_protocol_clean", "all_seeds")


def _scope_records(
    records: list[dict[str, Any]], scope: str
) -> list[dict[str, Any]]:
    """Select the records a given seed scope is allowed to see.

    ``frozen_protocol``       the three seeds the matrix freezes
    ``frozen_protocol_clean`` those three minus the tuning seed, whose
                              validation number is the one the learning rate was
                              selected on and so is not an independent
                              measurement while the test split stays locked
    ``all_seeds``             the frozen three plus the stability annex
    """

    if scope == "all_seeds":
        return list(records)
    frozen = [r for r in records if r.get("seed_class", "frozen_protocol")
              == "frozen_protocol"]
    if scope == "frozen_protocol":
        return frozen
    if scope == "frozen_protocol_clean":
        return [r for r in frozen if not r.get("tuning_contaminated", False)]
    raise ValueError(scope)


def _convergence(record: dict[str, Any]) -> tuple[bool, str]:
    """Whether a single run produced a result at all, and why not if it did not."""

    validation = record["validation"]
    if validation.get("nonfinite_predictions", 0) > 0:
        return False, "non_finite_predictions"
    if not (validation["macro_nloss"] < DIVERGENCE_MULTIPLE):
        return False, f"macro_nloss_above_{DIVERGENCE_MULTIPLE:g}x_constant"
    auc = validation.get("roc_auc", {}).get(PRIMARY_TARGET)
    if auc is not None and auc < CHANCE_AUC:
        return False, "primary_target_ranks_below_chance"
    return True, ""


def _by_arm(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["arm_id"], []).append(record)
    return grouped


def _values(
    records: list[dict[str, Any]], metric: str, target: str | None
) -> list[float]:
    if metric == "macro":
        return [r["validation"]["macro_nloss"] for r in records]
    if metric == "per_target":
        return [r["validation"]["per_target_normalized"][target] for r in records]
    if metric == "auc":
        return [r["validation"]["roc_auc"][target] for r in records]
    raise ValueError(metric)


def _separated(left: list[float], right: list[float]) -> bool:
    if not left or not right:
        return False
    return max(left) < min(right) or max(right) < min(left)


def _arm_status(records: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons = sorted({reason for reason in
                      (_convergence(r)[1] for r in records) if reason})
    return (not reasons), reasons


def _contrast_rows(
    grouped: dict[str, list[dict[str, Any]]], scope: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, subject, comparator, question, confirmatory in CONTRASTS:
        if subject not in grouped or comparator not in grouped:
            continue
        subject_records, comparator_records = grouped[subject], grouped[comparator]
        if not subject_records or not comparator_records:
            continue
        subject_ok, subject_why = _arm_status(subject_records)
        comparator_ok, comparator_why = _arm_status(comparator_records)
        left = _values(subject_records, "macro", None)
        right = _values(comparator_records, "macro", None)
        subject_mean, comparator_mean = statistics.fmean(left), statistics.fmean(right)
        quotable = subject_ok and comparator_ok
        rows.append({
            "contrast": name,
            "seed_scope": scope,
            "seeds_used": len(left),
            "expression": f"{subject} - {comparator}",
            "question": question,
            "confirmatory": confirmatory,
            "metric": "macro_nloss",
            "target": "aggregate",
            "subject_mean": subject_mean,
            "comparator_mean": comparator_mean,
            "difference": subject_mean - comparator_mean,
            "ratio": (
                max(subject_mean, comparator_mean)
                / min(subject_mean, comparator_mean)
            ),
            # `better` is withheld, not guessed, when either side diverged: a
            # comparison against a run that did not converge has no winner.
            "better": (
                (subject if subject_mean < comparator_mean else comparator)
                if quotable else "not_comparable"
            ),
            "quotable": quotable,
            "subject_converged": subject_ok,
            "comparator_converged": comparator_ok,
            "non_convergence": ";".join(
                [f"{subject}:{r}" for r in subject_why]
                + [f"{comparator}:{r}" for r in comparator_why]
            ),
            "seed_ranges_disjoint": quotable and _separated(left, right),
            "subject_per_seed": ";".join(f"{v:.8g}" for v in sorted(left)),
            "comparator_per_seed": ";".join(f"{v:.8g}" for v in sorted(right)),
        })
    return rows


def _per_target_rows(
    grouped: dict[str, list[dict[str, Any]]], scope: str, targets: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, subject, comparator, _question, _confirmatory in CONTRASTS:
        if subject not in grouped or comparator not in grouped:
            continue
        subject_records, comparator_records = grouped[subject], grouped[comparator]
        if not subject_records or not comparator_records:
            continue
        quotable = (_arm_status(subject_records)[0]
                    and _arm_status(comparator_records)[0])
        for target in targets:
            for metric, lower_is_better in (("per_target", True), ("auc", False)):
                available = subject_records[0]["validation"]["roc_auc"]
                if metric == "auc" and target not in available:
                    continue
                left = _values(subject_records, metric, target)
                right = _values(comparator_records, metric, target)
                subject_mean = statistics.fmean(left)
                comparator_mean = statistics.fmean(right)
                subject_wins = (
                    subject_mean < comparator_mean if lower_is_better
                    else subject_mean > comparator_mean
                )
                rows.append({
                    "contrast": name,
                    "seed_scope": scope,
                    "expression": f"{subject} - {comparator}",
                    "target": target,
                    "metric": (
                        "normalized_loss" if metric == "per_target" else "roc_auc"
                    ),
                    "lower_is_better": lower_is_better,
                    "subject_mean": subject_mean,
                    "comparator_mean": comparator_mean,
                    "difference": subject_mean - comparator_mean,
                    "better": (
                        (subject if subject_wins else comparator)
                        if quotable else "not_comparable"
                    ),
                    "quotable": quotable,
                    "seed_ranges_disjoint": quotable and _separated(left, right),
                    "subject_per_seed": ";".join(f"{v:.6g}" for v in sorted(left)),
                    "comparator_per_seed": ";".join(f"{v:.6g}" for v in sorted(right)),
                })
    return rows


def _summary_rows(
    grouped: dict[str, list[dict[str, Any]]], scope: str, targets: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm_id, records in sorted(grouped.items()):
        if not records:
            continue
        macro = _values(records, "macro", None)
        converged, reasons = _arm_status(records)
        entry = {
            "arm_id": arm_id,
            "seed_scope": scope,
            "label": records[0]["label"],
            "regime": records[0]["regime"],
            "seeds": len(macro),
            "converged": converged,
            "non_convergence": ";".join(reasons),
            "macro_nloss_mean": statistics.fmean(macro),
            "macro_nloss_sd": statistics.stdev(macro) if len(macro) > 1 else 0.0,
            "macro_nloss_min": min(macro),
            "macro_nloss_max": max(macro),
            "interaction_parameters": records[0]["parameters"]["interaction"],
            "non_embedding_parameters": records[0]["parameters"]["non_embedding"],
            "learning_rate": records[0]["learning_rate"],
            "nonfinite_predictions": sum(
                r["validation"]["nonfinite_predictions"] for r in records
            ),
        }
        for target in targets:
            entry[f"nloss_{target}"] = statistics.fmean(
                _values(records, "per_target", target)
            )
        for target in targets:
            if target in records[0]["validation"]["roc_auc"]:
                entry[f"auc_{target}"] = statistics.fmean(
                    _values(records, "auc", target)
                )
        entry["watch_time_mae_normalized"] = statistics.fmean(
            r["validation"]["watch_time_mae"] for r in records
        )
        if "watch_time_mae_seconds" in records[0]["validation"]:
            entry["watch_time_mae_seconds"] = statistics.fmean(
                r["validation"]["watch_time_mae_seconds"] for r in records
            )
        rows.append(entry)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "results" / "kuairand_multitask" / "rq23_primary_v1"
        / "results_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "kuairand_multitask" / "contrasts_v1",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    targets = payload["targets"]
    beff.prepare_output_dir(args.output_dir)

    contrasts: list[dict[str, Any]] = []
    per_target: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for scope in SEED_SCOPES:
        records = _scope_records(payload["records"], scope)
        if not records:
            continue
        grouped = _by_arm(records)
        contrasts.extend(_contrast_rows(grouped, scope))
        per_target.extend(_per_target_rows(grouped, scope, targets))
        summary.extend(_summary_rows(grouped, scope, targets))

    beff.publish_csv(args.output_dir / "contrasts.csv", contrasts)
    beff.publish_csv(args.output_dir / "contrasts_per_target.csv", per_target)
    beff.publish_csv(args.output_dir / "arm_summary.csv", summary)
    beff.publish_json(args.output_dir / "contrasts_v1.json", {
        "artifact_role": "kuairand_pure_contrast_report",
        "schema_version": 2,
        "source_results": (
            str(args.results.relative_to(REPO_ROOT))
            if args.results.is_relative_to(REPO_ROOT) else str(args.results)
        ),
        "source_results_sha256": payload.get("selected_configs_sha256"),
        "evaluated_split": payload["evaluated_split"],
        "test_unlocked": payload["test_unlocked"],
        "preconditioner": payload.get("pre_training_manifest_sha256"),
        "seed_scopes": list(SEED_SCOPES),
        "frozen_final_seeds": payload.get("final_seeds"),
        "annex_seeds": payload.get("annex_seeds", []),
        "convergence_gate": {
            "divergence_multiple_of_constant_predictor": DIVERGENCE_MULTIPLE,
            "chance_auc": CHANCE_AUC,
            "primary_target": PRIMARY_TARGET,
            "rule": (
                "a contrast is not quotable when either arm produced non-finite "
                "predictions, exceeded the divergence multiple, or ranked below "
                "chance on the primary target"
            ),
        },
        "normalizer_note": (
            "task constants are fitted on train (standard policy) while "
            "validation is the randomized pool, so 1.0 is not the "
            "constant-predictor level on validation; cross-arm comparison "
            "remains valid, absolute level against 1.0 does not"
        ),
        "contrasts": contrasts,
        "per_target": per_target,
        "arm_summary": summary,
    })

    for scope in SEED_SCOPES:
        rows = [r for r in contrasts if r["seed_scope"] == scope]
        if not rows:
            continue
        print(f"\n=== {scope} (n={rows[0]['seeds_used']} seeds per arm) ===")
        print(f"{'':<5}{'expression':<42}{'better':<24}{'ratio':>9}  separated")
        for row in rows:
            ratio = f"{row['ratio']:>9.3f}" if row["quotable"] else f"{'--':>9}"
            print(
                f"{row['contrast']:<5}{row['expression']:<42}"
                f"{row['better']:<24}{ratio}  {row['seed_ranges_disjoint']}"
            )
    print(f"\nwrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
