#!/usr/bin/env python
"""User-grouped metrics for VK-LSVD, recomputed with the real grouping key.

Why this script exists
----------------------
`run_kuairand_multitask.USER_COLUMN` documents column 0 of the categorical
block as `c_user_id`, which holds on the KuaiRand and KuaiRec views. The
VK-LSVD views carry no user column in that block at all - its six fields have
cardinalities 55, 4, 82, 24, 13, 22 - and ship the grouping key as a separate
`user_ids.i64.npy`. Every user-grouped number produced by a VK run before the
`grouping_key` fix was therefore grouped by a 55-value categorical field over
53 observed groups, not by the 97,612 users actually present.

The fix is in the runner, but the four VK targets were already trained. Their
per-row predictions are cached, so `user_gauc` and NDCG are recomputed here
rather than retrained - the same reason `PredictionStore` exists.

Alignment is verified, not assumed: the cached labels must reproduce exactly
from the view's own watch-time column under the target's threshold before any
metric is computed. A mismatch aborts.

Usage::

    python scripts/report_vk_user_metrics.py --target watch_time_gt10
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dcdn import metrics as dm  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beff = _load("benchmark_efficiency", REPO_ROOT / "scripts" / "benchmark_efficiency.py")
ndcg = _load("report_ndcg", REPO_ROOT / "scripts" / "report_ndcg.py")

CLIP_SECONDS = 157.0
THRESHOLDS = {
    "watch_time": None,
    "watch_time_gt10": 10.0,
    "watch_time_gt30": 30.0,
    "watch_time_gt60": 60.0,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(os.environ.get("DCDN_DATA_CACHE") or Path.home() / ".cache" / "dcdn-public")
    parser.add_argument("--target", choices=sorted(THRESHOLDS), required=True)
    parser.add_argument("--views", type=Path, default=cache / "runtime" / "vk_lsvd" / "multitask_v2")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    study = f"{args.target}_v1"
    predictions = args.predictions or cache / "predictions" / f"{study}.npz"
    output_dir = args.output_dir or REPO_ROOT / "results" / "vk_lsvd" / study / "user_metrics_v1"

    payload = np.load(predictions)
    labels = payload["labels"][:, 0].astype(np.float64)
    split = args.views / "c_plus_p" / "validation"
    users = np.load(split / "user_ids.i64.npy")
    watch = np.load(split / "targets.f32.npy")[:, 0]

    if users.size != labels.size:
        raise SystemExit(f"row count mismatch: {users.size} vs {labels.size}")
    threshold = THRESHOLDS[args.target]
    expected = watch if threshold is None else (watch > threshold / CLIP_SECONDS).astype(np.float32)
    if not np.array_equal(expected.astype(np.float64), labels):
        raise SystemExit("cached labels do not reproduce from the view - aborting")

    stale = payload["user_ids"]
    print(f"{labels.size:,} rows | true users {np.unique(users).size:,} "
          f"| stale grouping key had {np.unique(stale).size} groups")

    binary = threshold is not None
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(payload.files):
        if not key.startswith("pred__"):
            continue
        _, arm, seed = key.split("__")
        scores = payload[key][:, 0].astype(np.float64)
        entry: dict[str, Any] = {"seed": int(seed)}
        if binary:
            # Argument order and weighting mirror `metrics.masked_binary_target_metrics`.
            gauc = dm.user_group_auc(labels, scores, users, weighting="uniform")
            entry["user_gauc"] = float(gauc.value)
            entry["gauc_group_coverage"] = float(gauc.group_coverage)
            entry["gauc_row_coverage"] = float(gauc.row_coverage)
        else:
            # The regression path mirrors `run_kuairand_multitask` exactly, clip
            # included: clipping is not monotone at the ends and manufactures
            # ties, so dropping it would not reproduce the frozen metric.
            grouped = dm.group_concordance(
                CLIP_SECONDS * labels,
                CLIP_SECONDS * np.clip(scores, 0.0, 1.0),
                users,
                weighting="uniform",
            )
            entry["user_xauc"] = float(grouped.value)
            entry["xauc_group_coverage"] = float(grouped.group_coverage)
            entry["xauc_row_coverage"] = float(grouped.row_coverage)
        entry.update(ndcg.ndcg_by_group(scores, labels, users))
        by_arm.setdefault(arm, []).append(entry)
        print(f"  {arm:<26} seed={seed} done")

    fields = (["user_gauc"] if binary else ["user_xauc"]) + [
        "ndcg_at_5", "ndcg_at_10", "ndcg_at_50", "ndcg_full"
    ]
    rows: list[dict[str, Any]] = []
    for arm, entries in sorted(by_arm.items()):
        row: dict[str, Any] = {"arm_id": arm, "seeds": len(entries)}
        for field in fields:
            values = [e[field] for e in entries]
            row[field] = statistics.fmean(values)
            row[f"{field}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        prefix = "gauc" if binary else "xauc"
        row[f"{prefix}_group_coverage"] = entries[0][f"{prefix}_group_coverage"]
        row[f"{prefix}_row_coverage"] = entries[0][f"{prefix}_row_coverage"]
        row["user_coverage"] = entries[0]["user_coverage"]
        row["eligible_users"] = entries[0]["eligible_users"]
        row["total_users"] = entries[0]["total_users"]
        row["tied_adjacent_pair_fraction"] = statistics.fmean(
            e["tied_adjacent_pair_fraction"] for e in entries
        )
        rows.append(row)

    beff.prepare_output_dir(output_dir)
    beff.publish_csv(output_dir / "user_metrics.csv", rows)
    beff.publish_json(output_dir / "user_metrics_v1.json", {
        "artifact_role": "vk_lsvd_user_grouped_metrics_recomputed_with_true_user_ids",
        "schema_version": 1,
        "target": args.target,
        "source_predictions": str(predictions),
        "grouping_key": "runtime/vk_lsvd/multitask_v2/c_plus_p/validation/user_ids.i64.npy",
        "supersedes": "the user_gauc field of results_v1.json for this target",
        "supersession_reason": "runs predated the grouping_key fix and grouped by "
                               "categorical column 0, which is not a user id on VK-LSVD",
        "label_alignment_verified": True,
        "ranking_unit": "all_of_a_users_validation_rows",
        "ranking_unit_is_a_study_construction_not_a_released_candidate_list": True,
        "tie_break": "stable_sort_by_row_position",
        "cutoffs": list(ndcg.CUTOFFS),
        "rows": rows,
    })

    lead = "gAUC" if binary else "xAUC"
    head = fields[0]
    print(f"\n{'arm':<26}{lead:>9}{'NDCG@5':>9}{'@10':>9}{'@50':>9}{'ties':>8}")
    for row in sorted(rows, key=lambda r: -r[head]):
        print(f"{row['arm_id']:<26}{row[head]:>9.4f}{row['ndcg_at_5']:>9.4f}"
              f"{row['ndcg_at_10']:>9.4f}{row['ndcg_at_50']:>9.4f}"
              f"{row['tied_adjacent_pair_fraction']:>8.4f}")
    print(f"\nuser coverage {rows[0]['user_coverage']:.4f} "
          f"({rows[0]['eligible_users']:,}/{rows[0]['total_users']:,})")
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
