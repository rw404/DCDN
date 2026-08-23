#!/usr/bin/env python
"""Quality against serving cost for the seven core arms on three datasets.

The question this answers is narrower than "which model is best": it is whether
DCDN buys more ranking quality per millisecond than DCNv2 and AFN+ at the same
input regime. Rank order alone cannot answer it, and neither can latency alone.

Three frozen report slices, one per dataset - Yambda `listen50`, KuaiRec
`watch_ratio_gt2`, KuaiRand-Pure `watch_time`. Quality and cost describe the
SAME configurations: the benchmark runs the exact arm configs the quality run
trained, at the exact operand widths of that dataset's geometry.

`dcnv2_capacity_reduced` is the only mechanically matched comparator - within
±5% of `dcdn_reduced` on trainable non-embedding parameters and FLOPs. AFN+ and
plain DCNv2 are published-size baselines and differ by 2x to 44x, so a
difference against them confounds architecture with capacity and is reported
as such.

Latency is the eager-mode p50 at B=8192 from each dataset's own sweep. CUDA
graph runs exist but are not used: private memory pools accumulate across
captures, which makes peak VRAM meaningless per arm, and the graph numbers are
kept for the latency shape only.

Usage::

    python scripts/report_rq3_matrix.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ARMS = (
    "dcnv2_full", "dcnv2_reduced", "dcnv2_capacity_reduced",
    "dcdn_full", "dcdn_reduced",
    "afn_full", "afn_reduced",
)

#: (dataset label, results file, target, sweep csv, ndcg json or None)
SLICES = (
    ("Yambda-50M / listen50",
     "results/yambda_listen/l50_field_v1/results_v1.json", "listen50",
     "results/yambda_listen/batch_sweep_eager_v1/batch_sweep.csv",
     "results/yambda_listen/ndcg_v1/ndcg_v1.json"),
    ("KuaiRec / watch_ratio_gt2",
     "results/kuairec_watch_time/binary_field_v1/results_v1.json", "watch_ratio_gt2",
     "results/kuairec_watch_time/batch_sweep_field_eager_v1/batch_sweep.csv",
     "results/kuairec_watch_time/ndcg_v1/ndcg_v1.json"),
    ("KuaiRand-Pure / watch_time",
     "results/kuairand_multitask/field_watch_time_rms_v1/results_v1.json", "watch_time",
     "results/kuairand_multitask/batch_sweep_field_v1/batch_sweep.csv",
     None),
    ("VK-LSVD ur0.01_ir0.01 / timespent_gt10",
     "results/vk_lsvd_ur/timespent_gt10_v1/results_v1.json", "timespent_gt10",
     "results/vk_lsvd_ur/batch_sweep_v1/batch_sweep.csv",
     "results/vk_lsvd_ur/timespent_gt10_v1/ndcg_v1/ndcg_v1.json"),
)

BATCH = 8192


def quality(path: str, target: str) -> dict[str, dict]:
    payload = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    grouped: dict[str, list] = collections.defaultdict(list)
    for record in payload["records"]:
        grouped[record["arm_id"]].append(record)
    out: dict[str, dict] = {}
    for arm, records in grouped.items():
        metrics = [r["validation"]["frozen_metrics"][target] for r in records]
        kind = metrics[0]["kind"]
        lead = "roc_auc" if kind == "binary" else "global_xauc"
        grouping = "user_gauc" if kind == "binary" else "user_xgauc"
        out[arm] = {
            "kind": kind,
            "lead": st.fmean(m[lead] for m in metrics),
            "lead_sd": st.stdev(m[lead] for m in metrics),
            "grouped": st.fmean(m[grouping] for m in metrics),
            "per_seed": {r["seed"]: m[lead] for r, m in zip(records, metrics, strict=True)},
            "params": records[0]["parameters"]["non_embedding"],
            "seeds": len(records),
        }
    return out


def latency(path: str) -> dict[str, float]:
    rows = list(csv.DictReader((REPO_ROOT / path).open()))
    return {
        r["arm_id"]: float(r["batch_latency_ms_p50"])
        for r in rows if int(r["batch"]) == BATCH
    }


def ndcg(path: str | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    return {r["arm_id"]: r["ndcg_at_5"] for r in payload["rows"]}


def paired(a: dict, b: dict) -> tuple[float, float, int, int]:
    seeds = sorted(set(a["per_seed"]) & set(b["per_seed"]))
    diff = [a["per_seed"][s] - b["per_seed"][s] for s in seeds]
    stderr = st.stdev(diff) / (len(diff) ** 0.5)
    return st.fmean(diff), stderr, sum(d > 0 for d in diff), len(seeds)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    for label, results, target, sweep, ndcg_path in SLICES:
        q, lat, nd = quality(results, target), latency(sweep), ndcg(ndcg_path)
        lead_name = "AUC" if q["dcdn_reduced"]["kind"] == "binary" else "xAUC"
        group_name = "gAUC" if q["dcdn_reduced"]["kind"] == "binary" else "user xAUC"
        print(f"\n### {label}\n")
        head = f"| arm | inputs | params | {lead_name} | sd | {group_name} |"
        if nd:
            head += " NDCG@5 |"
        head += f" p50 @ B={BATCH} | quality/ms |"
        print(head)
        print("|---|---|---|---|---|---|" + ("---|" if nd else "") + "---|---|")
        for arm in ARMS:
            if arm not in q or arm not in lat:
                print(f"| {arm} | - | absent from this slice |")
                continue
            regime = "RED" if arm.endswith("_reduced") else "FULL"
            if arm == "dcnv2_capacity_reduced":
                regime = "RED*"
            name = f"**{arm}**" if arm.startswith("dcdn") else arm
            ratio = (q[arm]["lead"] - 0.5) / lat[arm]
            cells = (f"| {name} | {regime} | {q[arm]['params']:,} | "
                     f"{q[arm]['lead']:.5f} | {q[arm]['lead_sd']:.5f} | "
                     f"{q[arm]['grouped']:.5f} |")
            if nd:
                cells += f" {nd.get(arm, float('nan')):.4f} |"
            cells += f" {lat[arm]:.3f} ms | {ratio:.4f} |"
            print(cells)
        print(f"\n`quality/ms` = ({lead_name} - 0.5) / latency. A crude ratio, "
              f"offered as an ordering aid, not a decision metric.\n")
        print("| pair | mean diff | se | seeds | verdict | latency ratio |")
        print("|---|---|---|---|---|---|")
        for other in ("dcnv2_capacity_reduced", "dcnv2_reduced", "afn_reduced",
                      "dcnv2_full", "afn_full"):
            if other not in q:
                continue
            m, se, wins, n = paired(q["dcdn_reduced"], q[other])
            verdict = "**significant**" if abs(m) > 2 * se else "not significant"
            print(f"| dcdn_reduced - {other} | {m:+.5f} | {se:.5f} | {wins}/{n} | "
                  f"{verdict} | {lat[other] / lat['dcdn_reduced']:.2f}x |")

        # The ratio combines a noisy numerator with a latency that carries no
        # seed variation, so the same paired test applies to it directly: form
        # the per-seed ratio difference and test it against its own error.
        # Without this a "win" that rests entirely on the denominator reads the
        # same as one earned on quality.
        print("\nPaired on `quality/ms`, best DCDN arm against each comparator:")
        print("\n| pair | mean diff | se | seeds | verdict |")
        print("|---|---|---|---|---|")
        best = max(("dcdn_reduced", "dcdn_full"), key=lambda a: (q[a]["lead"] - 0.5) / lat[a])
        for other in ARMS:
            if other == best or other not in q:
                continue
            seeds = sorted(set(q[best]["per_seed"]) & set(q[other]["per_seed"]))
            diff = [(q[best]["per_seed"][s] - 0.5) / lat[best]
                    - (q[other]["per_seed"][s] - 0.5) / lat[other] for s in seeds]
            mean, stderr = st.fmean(diff), st.stdev(diff) / (len(diff) ** 0.5)
            verdict = "**significant**" if abs(mean) > 2 * stderr else "not significant"
            print(f"| {best} - {other} | {mean:+.4f} | {stderr:.4f} | "
                  f"{sum(d > 0 for d in diff)}/{len(seeds)} | {verdict} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
