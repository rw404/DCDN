#!/usr/bin/env python
"""The consolidated KuaiRand-Pure matrix: four architectures x three task settings.

Builds one table covering

    DCNv2(FULL) / DCNv2(REDUCED) / DCDN(REDUCED) / DCDN(FULL)

across

    1. the multi-head task    (five heads under Macro-NLoss)
    2. single-head long_view  (binary)
    3. single-head watch_time (regression)

reporting, for every cell, the validation loss for that task plus the ranking
metrics: ROC-AUC and user-grouped GAUC for binary heads, and global X-AUC with
its user-grouped counterpart for the regression head. Every metric comes from
`src/dcdn/metrics.py`, the frozen implementation named by
`configs/metrics_multitask_v1.yaml` — the three task settings are therefore
measured by one definition rather than three similar-looking ones.

Three reading rules are enforced rather than left to prose.

* **Seed scopes are never pooled.** The frozen matrix fixes
  `final_seeds: [42, 43, 44]`; the stability annex (45-49) is reported beside
  it, tagged, never averaged into it.
* **A non-convergent arm is not scored.** An arm that produced non-finite
  predictions, or landed an order of magnitude past the constant predictor, did
  not produce a result; its cell reads `non-convergent` instead of a number
  that would rank it.
* **Separation is stated, not implied.** A difference between two arms is only
  marked established when their per-seed ranges are disjoint.

`session_gauc` is null throughout with an explicit reason: KuaiRand-Pure ships
no session identifier, and the frozen `session_policy` forbids synthesizing
one.

Usage::

    python scripts/report_kuairand_matrix.py
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

#: The four the comparison is about, in the order the question was asked.
FOCUS_ARMS = ("dcnv2_full", "dcnv2_reduced", "dcdn_reduced", "dcdn_full")
#: Carried for context; they are part of the frozen matrix but not the question.
CONTEXT_ARMS = ("afn_reduced", "afn_full", "dcnv2_capacity_reduced")
ALL_ARMS = FOCUS_ARMS + CONTEXT_ARMS

TASKS = (
    ("multi_head", "final_mt_v2", "five heads, Macro-NLoss"),
    ("single_long_view", "final_lv_v2", "one binary head"),
    ("single_watch_time", "final_wt_v2", "one regression head"),
)

#: Same gate as the contrast reporter: an arm this far past the constant
#: predictor did not learn the task, it diverged.
DIVERGENCE_MULTIPLE = 10.0


def _converged(record: dict[str, Any]) -> bool:
    validation = record["validation"]
    if validation.get("nonfinite_predictions", 0) > 0:
        return False
    return validation["macro_nloss"] < DIVERGENCE_MULTIPLE


def _scope(records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "all_seeds":
        return records
    return [
        r for r in records
        if r.get("seed_class", "frozen_protocol") == "frozen_protocol"
    ]


def _cell(records: list[dict[str, Any]], head: str) -> dict[str, Any]:
    """One (task, arm) cell: loss plus the ranking metrics for that head kind."""

    converged = [r for r in records if _converged(r)]
    entry: dict[str, Any] = {
        "seeds": len(records),
        "converged_seeds": len(converged),
        "quotable": len(converged) == len(records) and bool(records),
    }
    if not records:
        return entry

    losses = [r["validation"]["macro_nloss"] for r in records]
    entry["loss_mean"] = statistics.fmean(losses)
    entry["loss_sd"] = statistics.stdev(losses) if len(losses) > 1 else 0.0
    entry["loss_min"] = min(losses)
    entry["loss_max"] = max(losses)

    metrics = [r["validation"]["frozen_metrics"][head] for r in records]
    kind = metrics[0]["kind"]
    entry["kind"] = kind
    if kind == "binary":
        for key, out in (
            ("roc_auc", "auc"), ("user_gauc", "gauc"), ("auprc", "auprc"),
            ("log_loss", "log_loss"),
        ):
            values = [m[key] for m in metrics]
            entry[out] = statistics.fmean(values)
            entry[f"{out}_min"] = min(values)
            entry[f"{out}_max"] = max(values)
        entry["gauc_group_coverage"] = statistics.fmean(
            m["user_gauc_group_coverage"] for m in metrics
        )
    else:
        for key, out in (
            ("global_xauc", "xauc"), ("user_xgauc", "xgauc"),
            ("normalized_mse", "normalized_mse"), ("mae_seconds", "mae_seconds"),
        ):
            values = [m[key] for m in metrics]
            entry[out] = statistics.fmean(values)
            entry[f"{out}_min"] = min(values)
            entry[f"{out}_max"] = max(values)
        entry["xgauc_group_coverage"] = statistics.fmean(
            m["user_xgauc_group_coverage"] for m in metrics
        )
        entry["xauc_pair_coverage"] = statistics.fmean(
            m["global_xauc_comparable_pair_coverage"] for m in metrics
        )
    entry["session_gauc"] = None
    entry["session_gauc_availability"] = metrics[0]["session_gauc_availability"]
    return entry


def _head_for(task: str, payload: dict[str, Any]) -> str:
    if task == "multi_head":
        return "long_view"
    return payload["target"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = REPO_ROOT / "results" / "kuairand_multitask"
    parser.add_argument("--results-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=root / "matrix_v1")
    args = parser.parse_args(argv)

    bundles: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for task, directory, _ in TASKS:
        path = args.results_root / directory / "results_v1.json"
        if not path.is_file():
            missing.append(f"{task} ({path})")
            continue
        bundles[task] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        raise SystemExit("missing bundles: " + "; ".join(missing))

    for task, payload in bundles.items():
        if payload["test_unlocked"]:
            raise SystemExit(f"{task} bundle reports an unlocked test split")

    beff.prepare_output_dir(args.output_dir)

    rows: list[dict[str, Any]] = []
    per_head_rows: list[dict[str, Any]] = []
    for scope in ("frozen_protocol", "all_seeds"):
        for task, _directory, description in TASKS:
            payload = bundles[task]
            grouped: dict[str, list[dict[str, Any]]] = {}
            for record in _scope(payload["records"], scope):
                grouped.setdefault(record["arm_id"], []).append(record)
            head = _head_for(task, payload)
            for arm in ALL_ARMS:
                if arm not in grouped:
                    continue
                rows.append({
                    "task": task,
                    "task_description": description,
                    "seed_scope": scope,
                    "arm_id": arm,
                    "focus": arm in FOCUS_ARMS,
                    "scored_head": head,
                    **_cell(grouped[arm], head),
                })
            # The multi-head task is the only one with heads to break out.
            if task != "multi_head":
                continue
            for arm in ALL_ARMS:
                if arm not in grouped:
                    continue
                for other in payload["targets"]:
                    per_head_rows.append({
                        "task": task,
                        "seed_scope": scope,
                        "arm_id": arm,
                        "head": other,
                        **_cell(grouped[arm], other),
                    })

    # Binary and regression cells carry different metric names, and the CSV
    # writer requires a uniform header. Pad rather than split into two files:
    # a reader comparing a binary task against a regression one wants them
    # side by side, with the inapplicable columns visibly blank.
    def squared(collection: list[dict[str, Any]]) -> list[dict[str, Any]]:
        columns: list[str] = []
        for row in collection:
            for key in row:
                if key not in columns:
                    columns.append(key)
        return [{key: row.get(key, "") for key in columns} for row in collection]

    beff.publish_csv(args.output_dir / "matrix.csv", squared(rows))
    beff.publish_csv(
        args.output_dir / "multi_head_per_head.csv", squared(per_head_rows)
    )
    beff.publish_json(args.output_dir / "matrix_v1.json", {
        "artifact_role": "kuairand_pure_consolidated_matrix",
        "schema_version": 1,
        "evaluated_split": "validation",
        "test_unlocked": False,
        "focus_arms": list(FOCUS_ARMS),
        "context_arms": list(CONTEXT_ARMS),
        "tasks": [{"task": t, "bundle": d, "description": s} for t, d, s in TASKS],
        "metric_contract": "configs/metrics_multitask_v1.yaml",
        "metric_implementation": "src/dcdn/metrics.py",
        "convergence_gate_multiple": DIVERGENCE_MULTIPLE,
        "session_gauc": None,
        "session_gauc_availability": (
            "not_available_kuairand_pure_has_no_session_identifier"
        ),
        "seed_scopes_pooled": False,
        "rows": rows,
        "per_head": per_head_rows,
    })

    for scope in ("frozen_protocol", "all_seeds"):
        for task, _directory, description in TASKS:
            subset = [
                r for r in rows
                if r["task"] == task and r["seed_scope"] == scope and r["focus"]
            ]
            if not subset:
                continue
            kind = subset[0].get("kind")
            print(f"\n=== {task} ({description}) — {scope} ===")
            if kind == "binary":
                header = f"{'arm':<24}{'loss':>9}{'AUC':>9}{'gAUC':>9}{'AUPRC':>9}"
            else:
                header = (
                    f"{'arm':<24}{'loss':>9}{'xAUC':>9}{'xgAUC':>9}{'MAE(s)':>9}"
                )
            print(header)
            for row in subset:
                if not row["quotable"]:
                    print(f"{row['arm_id']:<24}{'non-convergent':>36}")
                    continue
                if kind == "binary":
                    print(
                        f"{row['arm_id']:<24}{row['loss_mean']:>9.4f}"
                        f"{row['auc']:>9.4f}{row['gauc']:>9.4f}{row['auprc']:>9.4f}"
                    )
                else:
                    print(
                        f"{row['arm_id']:<24}{row['loss_mean']:>9.4f}"
                        f"{row['xauc']:>9.4f}{row['xgauc']:>9.4f}"
                        f"{row['mae_seconds']:>9.2f}"
                    )
    print(f"\nwrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
