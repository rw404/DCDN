#!/usr/bin/env python
"""User-grouped NDCG from cached validation predictions.

Runs on the per-row predictions the runners persist with `--save-predictions`,
so adding this metric costs seconds rather than a retrain.

Why this is computed for KuaiRec and not for KuaiRand
-----------------------------------------------------
`configs/data/kuairand_pure/features_v1.yaml` declares
``no_ndcg_reason: no_released_complete_candidate_lists_per_request`` — that
dataset ships impressions, not the candidate pool a ranker actually scored, so
an NDCG over it would be measuring a list the system never produced.
`configs/data/kuairec/features_v1.yaml` makes no such statement and forbids
nothing here.

What the "list" is, stated plainly
----------------------------------
There is still no released per-request candidate list. The ranking unit used
here is **all of a user's validation rows**, which is the standard offline
convention and is a construction of this study, not a property of the data. A
user's validation rows span the whole validation window, so NDCG@10 answers
"did the ten items this user was most likely to replay end up at the top of our
ordering", not "was one served page ordered well". Quote it with that attached.

Ties
----
NDCG is sensitive to how equal scores are ordered, and a deterministic sort
would let row order stand in for model quality. Ties are broken by row position via a
stable sort, which is arbitrary but fixed, and the fraction of tied adjacent
pairs is reported beside every value.

On KuaiRec that fraction is 24.3% and is *identical for every architecture*,
because the contract sets ``deduplication: none`` and a repeated (user, item)
pair is scored identically by any model. It matters far less than it looks:
only 0.6% of those tied pairs carry disagreeing labels, so the tie-break can
affect 0.13% of adjacent pairs.

Usage::

    python scripts/report_ndcg.py --predictions <cache>/predictions/binary_field_v1.npz
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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beff = _load("benchmark_efficiency", REPO_ROOT / "scripts" / "benchmark_efficiency.py")

CUTOFFS = (5, 10, 50)


def ndcg_by_group(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    cutoffs: tuple[int, ...] = CUTOFFS,
) -> dict[str, float]:
    """Uniform mean NDCG over users, with coverage and tie incidence.

    A user contributes only when it has at least one positive and at least two
    rows: with no positive the ideal DCG is zero and NDCG is undefined, and with
    one row every ordering is identical. Both exclusions are counted.
    """

    order = np.argsort(groups, kind="stable")
    scores, labels, groups = scores[order], labels[order], groups[order]
    boundaries = np.flatnonzero(np.diff(groups)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [groups.size]))

    per_cutoff: dict[int, list[float]] = {k: [] for k in (*cutoffs, 0)}
    eligible = 0
    tied_pairs = 0
    total_pairs = 0
    for start, end in zip(starts, ends, strict=True):
        group_labels = labels[start:end]
        if end - start < 2 or group_labels.sum() <= 0:
            continue
        eligible += 1
        group_scores = scores[start:end]
        # Deterministic tie-break by position; see the module docstring.
        ranking = np.argsort(-group_scores, kind="stable")
        ranked = group_labels[ranking]
        ideal = np.sort(group_labels)[::-1]

        ordered_scores = group_scores[ranking]
        tied_pairs += int((np.diff(ordered_scores) == 0).sum())
        total_pairs += max(0, ordered_scores.size - 1)

        discount = 1.0 / np.log2(np.arange(2, ranked.size + 2))
        for k in (*cutoffs, 0):
            limit = ranked.size if k == 0 else min(k, ranked.size)
            dcg = float((ranked[:limit] * discount[:limit]).sum())
            idcg = float((ideal[:limit] * discount[:limit]).sum())
            if idcg > 0.0:
                per_cutoff[k].append(dcg / idcg)

    total_groups = int(starts.size)
    report: dict[str, float] = {
        "eligible_users": eligible,
        "total_users": total_groups,
        "user_coverage": eligible / total_groups if total_groups else 0.0,
        "tied_adjacent_pair_fraction": (
            tied_pairs / total_pairs if total_pairs else 0.0
        ),
    }
    for k in (*cutoffs, 0):
        name = "ndcg_full" if k == 0 else f"ndcg_at_{k}"
        values = per_cutoff[k]
        report[name] = statistics.fmean(values) if values else float("nan")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(os.environ.get("DCDN_DATA_CACHE", "/tmp"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=cache / "predictions" / "binary_field_v1.npz",
    )
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "kuairec_watch_time" / "ndcg_v1",
    )
    args = parser.parse_args(argv)

    payload = np.load(args.predictions)
    labels = payload["labels"][:, args.target_index].astype(np.float64)
    users = payload["user_ids"]
    print(f"{labels.size:,} validation rows, {np.unique(users).size:,} users, "
          f"positive rate {labels.mean():.4f}")

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(payload.files):
        if not key.startswith("pred__"):
            continue
        _, arm, seed = key.split("__")
        scores = payload[key][:, args.target_index].astype(np.float64)
        by_arm.setdefault(arm, []).append(
            {"seed": int(seed), **ndcg_by_group(scores, labels, users)}
        )
        print(f"  {arm:<26} seed={seed} done")

    rows: list[dict[str, Any]] = []
    for arm, entries in sorted(by_arm.items()):
        row: dict[str, Any] = {"arm_id": arm, "seeds": len(entries)}
        for metric in ("ndcg_at_5", "ndcg_at_10", "ndcg_at_50", "ndcg_full"):
            values = [e[metric] for e in entries]
            row[metric] = statistics.fmean(values)
            row[f"{metric}_sd"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        row["user_coverage"] = entries[0]["user_coverage"]
        row["eligible_users"] = entries[0]["eligible_users"]
        row["total_users"] = entries[0]["total_users"]
        row["tied_adjacent_pair_fraction"] = statistics.fmean(
            e["tied_adjacent_pair_fraction"] for e in entries
        )
        rows.append(row)

    beff.prepare_output_dir(args.output_dir)
    beff.publish_csv(args.output_dir / "ndcg.csv", rows)
    beff.publish_json(args.output_dir / "ndcg_v1.json", {
        "artifact_role": "user_grouped_ndcg_from_cached_predictions",
        "schema_version": 1,
        "source_predictions": str(args.predictions),
        "ranking_unit": "all_of_a_users_validation_rows",
        "ranking_unit_is_a_study_construction_not_a_released_candidate_list": True,
        "tie_break": "stable_sort_by_row_position",
        "eligibility": "at_least_one_positive_and_at_least_two_rows",
        "cutoffs": list(CUTOFFS),
        "rows": rows,
    })

    print(f"\n{'arm':<26}{'NDCG@5':>9}{'@10':>9}{'@50':>9}{'full':>9}{'ties':>8}")
    for row in sorted(rows, key=lambda r: -r["ndcg_at_10"]):
        print(f"{row['arm_id']:<26}{row['ndcg_at_5']:>9.4f}{row['ndcg_at_10']:>9.4f}"
              f"{row['ndcg_at_50']:>9.4f}{row['ndcg_full']:>9.4f}"
              f"{row['tied_adjacent_pair_fraction']:>8.4f}")
    print(f"\nuser coverage {rows[0]['user_coverage']:.4f} "
          f"({rows[0]['eligible_users']:,}/{rows[0]['total_users']:,})")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
