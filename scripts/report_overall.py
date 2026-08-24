#!/usr/bin/env python
"""One report over four datasets, fourteen architectures, quality and cost.

Assembles the whole comparison from the frozen per-dataset artifacts: the four
report slices, every metric each contract declares, trainable parameter counts,
and measured serving latency. Nothing is recomputed here that a study already
published; this is a join, so a number that appears in two places is the same
number.

`dcnv2_capacity_reduced` is the only mechanically matched comparator for
`dcdn_reduced` - the two agree within 0.2% on trainable non-embedding
parameters and FLOPs on all four datasets. Every other pairwise difference
confounds architecture with capacity, by a factor between 1.9x and 54x.

Usage::

    python scripts/report_overall.py [--section quality|cost|params|all]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH = 8192

#: label, quality dirs (base + euler addendum), target, kind, ndcg jsons, sweep
SLICES = (
    ("Yambda-50M / listen50",
     ("results/yambda_listen/l50_field_v1", "results/yambda_listen/l50_euler_v1"),
     "listen50", "binary",
     ("results/yambda_listen/ndcg_v1/ndcg_v1.json",
      "results/yambda_listen/l50_euler_v1/ndcg_v1/ndcg_v1.json"),
     "results/yambda_listen/batch_sweep_14_v1/batch_sweep.csv"),
    ("KuaiRec / watch_ratio_gt2",
     ("results/kuairec_watch_time/binary_field_v1",
      "results/kuairec_watch_time/binary_euler_v1"),
     "watch_ratio_gt2", "binary",
     ("results/kuairec_watch_time/ndcg_v1/ndcg_v1.json",
      "results/kuairec_watch_time/binary_euler_v1/ndcg_v1/ndcg_v1.json"),
     "results/kuairec_watch_time/batch_sweep_14_v1/batch_sweep.csv"),
    ("KuaiRand-Pure / watch_time",
     ("results/kuairand_multitask/field_watch_time_rms_v1",
      "results/kuairand_multitask/watch_time_euler_v1"),
     "watch_time", "regression",
     ("results/kuairand_multitask/field_watch_time_rms_v1/ndcg_v1/ndcg_v1.json",
      "results/kuairand_multitask/watch_time_euler_v1/ndcg_v1/ndcg_v1.json"),
     "results/kuairand_multitask/batch_sweep_14_v1/batch_sweep.csv"),
    ("VK-LSVD ur / timespent_gt10",
     ("results/vk_lsvd_ur/timespent_gt10_v1",
      "results/vk_lsvd_ur/timespent_gt10_euler_v1"),
     "timespent_gt10", "binary",
     ("results/vk_lsvd_ur/timespent_gt10_v1/ndcg_v1/ndcg_v1.json",
      "results/vk_lsvd_ur/timespent_gt10_euler_v1/ndcg_v1/ndcg_v1.json"),
     "results/vk_lsvd_ur/batch_sweep_14_v1/batch_sweep.csv"),
)

BINARY = (("roc_auc", "AUC"), ("user_gauc", "gAUC"), ("auprc", "AUPRC"),
          ("log_loss", "logloss"))
REGRESSION = (("global_xauc", "xAUC"), ("user_xgauc", "user xAUC"),
              ("normalized_mse", "nMSE"), ("mae_seconds", "MAE"))
HIGHLIGHT = ("dcdn", "eulernet")


def load_quality(dirs, target):
    grouped: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    parameters: dict[str, dict] = {}
    for directory in dirs:
        payload = json.loads(
            (REPO_ROOT / directory / "results_v1.json").read_text(encoding="utf-8"))
        for record in payload["records"]:
            grouped[record["arm_id"]][record["seed"]] = (
                record["validation"]["frozen_metrics"][target])
            parameters[record["arm_id"]] = record["parameters"]
    return grouped, parameters


def load_ndcg(paths):
    rows: dict[str, dict] = {}
    for path in paths:
        try:
            payload = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        for row in payload["rows"]:
            rows[row["arm_id"]] = row
    return rows


def load_latency(path):
    try:
        rows = list(csv.DictReader((REPO_ROOT / path).open()))
    except FileNotFoundError:
        return {}
    return {r["arm_id"]: float(r["batch_latency_ms_p50"])
            for r in rows if int(r["batch"]) == BATCH}


def regime(arm: str) -> str:
    if arm == "dcnv2_capacity_reduced":
        return "RED*"
    return "RED" if arm.endswith("_reduced") else "FULL"


def name(arm: str) -> str:
    return f"**{arm}**" if arm.startswith(HIGHLIGHT) else arm


def quality_tables() -> None:
    for label, dirs, target, kind, ndcg_paths, sweep in SLICES:
        grouped, _ = load_quality(dirs, target)
        ndcg = load_ndcg(ndcg_paths)
        latency = load_latency(sweep)
        fields = BINARY if kind == "binary" else REGRESSION
        lead = fields[0][0]
        print(f"\n### {label}\n")
        header = "| # | arm | in | " + " | ".join(l for _, l in fields) + " | sd |"
        if ndcg:
            header += " NDCG@5 | @10 | @50 |"
        if latency:
            header += " p50 ms |"
        print(header)
        print("|---|---|---|" + "---|" * (len(fields) + 1)
              + ("---|---|---|" if ndcg else "") + ("---|" if latency else ""))
        rows = []
        for arm, seeds in grouped.items():
            values = {k: st.fmean(m[k] for m in seeds.values()) for k, _ in fields}
            rows.append((arm, values, st.stdev(m[lead] for m in seeds.values())))
        for index, (arm, values, sd) in enumerate(
                sorted(rows, key=lambda r: -r[1][lead]), 1):
            cells = (f"| {index} | {name(arm)} | {regime(arm)} | "
                     + " | ".join(f"{values[k]:.5f}" if abs(values[k]) < 10
                                  else f"{values[k]:.3f}" for k, _ in fields)
                     + f" | {sd:.5f} |")
            if ndcg:
                entry = ndcg.get(arm)
                cells += (f" {entry['ndcg_at_5']:.4f} | {entry['ndcg_at_10']:.4f} | "
                          f"{entry['ndcg_at_50']:.4f} |") if entry else " — | — | — |"
            if latency:
                cells += f" {latency.get(arm, float('nan')):.3f} |"
            print(cells)


def parameter_table() -> None:
    columns = [label.split(" /")[0] for label, *_ in SLICES]
    table: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for (label, dirs, target, *_rest) in SLICES:
        _, parameters = load_quality(dirs, target)
        for arm, counts in parameters.items():
            table[arm][label.split(" /")[0]] = counts
    base = {c: table["dcdn_reduced"][c]["non_embedding"] for c in columns}
    print("\n### Trainable non-embedding parameters\n")
    print("| arm | in | " + " | ".join(columns) + " | vs dcdn_reduced |")
    print("|---|---|" + "---|" * (len(columns) + 1))
    rows = sorted(table, key=lambda a: st.median(
        table[a][c]["non_embedding"] / base[c] for c in columns))
    for arm in rows:
        ratio = st.median(table[arm][c]["non_embedding"] / base[c] for c in columns)
        print(f"| {name(arm)} | {regime(arm)} | "
              + " | ".join(f"{table[arm][c]['non_embedding']:,}" for c in columns)
              + f" | {ratio:.2f}x |")
    print("\n### Interaction block only\n")
    print("| arm | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for arm in rows:
        print(f"| {name(arm)} | "
              + " | ".join(f"{table[arm][c]['interaction']:,}" for c in columns) + " |")


def cost_table() -> None:
    columns = [label.split(" /")[0] for label, *_ in SLICES]
    latencies = {label.split(" /")[0]: load_latency(sweep)
                 for label, _d, _t, _k, _n, sweep in SLICES}
    arms = sorted(set().union(*(set(v) for v in latencies.values() if v)))
    if not arms:
        print("\n(no 14-arm sweeps present yet)\n")
        return
    base = {c: latencies[c].get("dcdn_reduced") for c in columns}
    print(f"\n### Serving latency, eager-mode p50 at B = {BATCH} (ms)\n")
    print("| arm | in | " + " | ".join(columns) + " | vs dcdn_reduced |")
    print("|---|---|" + "---|" * (len(columns) + 1))
    def ratio(arm):
        values = [latencies[c][arm] / base[c] for c in columns
                  if arm in latencies[c] and base.get(c)]
        return st.median(values) if values else float("inf")
    for arm in sorted(arms, key=ratio):
        cells = []
        for column in columns:
            value = latencies[column].get(arm)
            cells.append(f"{value:.3f}" if value is not None else "—")
        print(f"| {name(arm)} | {regime(arm)} | " + " | ".join(cells)
              + f" | {ratio(arm):.2f}x |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", default="all",
                        choices=("quality", "cost", "params", "all"))
    args = parser.parse_args(argv)
    if args.section in ("params", "all"):
        parameter_table()
    if args.section in ("cost", "all"):
        cost_table()
    if args.section in ("quality", "all"):
        quality_tables()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
