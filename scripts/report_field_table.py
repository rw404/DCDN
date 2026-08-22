#!/usr/bin/env python
"""One arm-by-arm table for a completed single-target field study.

Prints the twelve-arm standings plus the paired seed-by-seed comparison
against `dcdn_reduced`, which is the instrument that matters at this
resolution: the whole field usually spans less than four times the per-seed
standard deviation, so rank order alone says almost nothing. Two arms are
compared on the SAME seeds and the difference is tested against its own
standard error, never by asking whether two ranges overlap.

`dcnv2_capacity_reduced` is called out because it is the only arm mechanically
matched to `dcdn_reduced` in trainable non-embedding parameters and FLOPs.
Every other comparison confounds architecture with size.

Usage::

    python scripts/report_field_table.py --results results/<study>/results_v1.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

BINARY = ("roc_auc", "user_gauc", "auprc", "log_loss")
REGRESSION = ("global_xauc", "user_xgauc", "normalized_mse", "mae_seconds")
LOWER_IS_BETTER = {"log_loss", "normalized_mse", "mae_seconds"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", default="dcdn_reduced")
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    target = payload["target"]
    by_arm: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for record in payload["records"]:
        metrics = record["validation"]["frozen_metrics"][target]
        by_arm[record["arm_id"]][record["seed"]] = metrics
    kind = next(iter(next(iter(by_arm.values())).values()))["kind"]
    fields = BINARY if kind == "binary" else REGRESSION
    lead = fields[0]

    def mean(arm: str, field: str) -> float:
        return st.fmean(m[field] for m in by_arm[arm].values())

    print(f"{target} ({kind}), {len(next(iter(by_arm.values())))} seeds, "
          f"{len(by_arm)} arms\n")
    header = "".join(f"{f:>16}" for f in fields)
    print(f"{'arm':<26}{'inputs':<8}{header}{'sd':>10}")
    order = sorted(by_arm, key=lambda a: mean(a, lead), reverse=lead not in LOWER_IS_BETTER)
    for arm in order:
        regime = "RED" if arm.endswith("_reduced") else "FULL"
        cells = "".join(f"{mean(arm, f):>16.5f}" for f in fields)
        sd = st.stdev(m[lead] for m in by_arm[arm].values())
        mark = "  <<<" if arm.startswith("dcdn") else ""
        print(f"{arm:<26}{regime:<8}{cells}{sd:>10.5f}{mark}")

    base = by_arm[args.baseline]
    seeds = sorted(base)
    print(f"\npaired against {args.baseline}, {lead}, same seeds:")
    rows = []
    for arm in by_arm:
        if arm == args.baseline:
            continue
        diff = [by_arm[arm][s][lead] - base[s][lead] for s in seeds]
        stderr = st.stdev(diff) / (len(diff) ** 0.5)
        rows.append((arm, st.fmean(diff), stderr, sum(d > 0 for d in diff)))
    for arm, m, se, wins in sorted(rows, key=lambda r: r[1]):
        verdict = "SIGNIFICANT" if abs(m) > 2 * se else "not significant"
        note = "  <- capacity-matched" if arm == "dcnv2_capacity_reduced" else ""
        print(f"  {arm:<26}{m:+.5f}  se {se:.5f}  beats {wins}/{len(seeds)}  "
              f"{verdict}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
