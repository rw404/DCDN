#!/usr/bin/env python
"""Measure task transfer on `watch_time`: single-head against multi-task head.

Pairs, per arm and per seed, the single-target model's normalized `watch_time`
loss against the *corresponding head* of the five-head multi-task model.

The comparison is only meaningful because the normalizer is identical. Both
studies divide MSE by the same train-only variance constant
(`0.008677877`), computed from the same train rows, so the two numbers are the
same quantity and their difference is transfer rather than a change of scale.
The script verifies that constant matches before reporting anything and refuses
to continue if it does not.

Sign convention: `difference = multi_task_head - single_head`. Positive means
the single-head model did better, i.e. **negative transfer** — multi-tasking
cost `watch_time` quality.

The frozen follow-up protocol classifies a result as transfer only when a
paired interval excludes zero; a non-significant result is reported as "no
detected effect" and never as proof of equality. With eight seeds this uses a
paired bootstrap over seeds rather than assuming normality.

Usage::

    python scripts/report_kuairand_transfer.py
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

TARGET = "watch_time"
BOOTSTRAP_RESAMPLES = 20000
#: Deterministic: the same bundles must always produce the same interval.
BOOTSTRAP_SEED = 20260821


class TransferError(RuntimeError):
    """The two bundles are not comparable."""


def _paired(single: dict, multi: dict) -> dict[str, dict[int, tuple[float, float]]]:
    """`arm -> seed -> (single_head_loss, multi_task_head_loss)`."""

    left: dict[str, dict[int, float]] = {}
    for record in single["records"]:
        left.setdefault(record["arm_id"], {})[record["seed"]] = (
            record["validation"]["macro_nloss"]
        )
    right: dict[str, dict[int, float]] = {}
    for record in multi["records"]:
        right.setdefault(record["arm_id"], {})[record["seed"]] = (
            record["validation"]["per_target_normalized"][TARGET]
        )

    paired: dict[str, dict[int, tuple[float, float]]] = {}
    for arm in sorted(set(left) & set(right)):
        seeds = sorted(set(left[arm]) & set(right[arm]))
        paired[arm] = {s: (left[arm][s], right[arm][s]) for s in seeds}
    return paired


def _bootstrap_ci(diffs: list[float]) -> tuple[float, float]:
    """Percentile bootstrap over the paired per-seed differences."""

    import random

    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(diffs)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        means.append(
            statistics.fmean(diffs[rng.randrange(count)] for _ in range(count))
        )
    means.sort()
    low = means[int(0.025 * BOOTSTRAP_RESAMPLES)]
    high = means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    return (low, high)


def _classify(low: float, high: float) -> str:
    """The frozen follow-up protocol's three-way classification."""

    if low != low or high != high:
        return "insufficient_seeds"
    if low > 0.0:
        return "negative_transfer"
    if high < 0.0:
        return "positive_transfer"
    return "no_detected_effect"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = REPO_ROOT / "results" / "kuairand_multitask"
    parser.add_argument(
        "--single", type=Path, default=root / "single_target_v1" / "results_v1.json"
    )
    parser.add_argument(
        "--multi", type=Path, default=root / "rq23_final_v1" / "results_v1.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "transfer_v1"
    )
    args = parser.parse_args(argv)

    single = json.loads(args.single.read_text(encoding="utf-8"))
    multi = json.loads(args.multi.read_text(encoding="utf-8"))

    # The whole comparison rests on both studies dividing by the same train-only
    # variance, so a mismatch would turn a change of scale into an apparent
    # transfer effect. The two `task_constants_sha256` values cannot be compared
    # directly - one covers five targets and the other one, so they differ by
    # construction. What IS checkable, and what actually implies the constants
    # agree, is that both studies read the same materialized train rows: the
    # constant is the variance of `watch_time` over those rows and its
    # definition is fixed in the contract.
    view_hashes = {}
    for label, bundle in (("single", args.single), ("multi", args.multi)):
        manifest_path = bundle.parent / "pre_training_manifest_v1.json"
        if not manifest_path.is_file():
            raise TransferError(f"{label} bundle has no pre-training manifest")
        view_hashes[label] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )["view_manifest_sha256"]
    if view_hashes["single"] != view_hashes["multi"]:
        raise TransferError(
            "bundles were built from different materialized views, so their "
            f"train-only normalizers cannot be assumed equal: {view_hashes}"
        )

    if single["test_unlocked"] or multi["test_unlocked"]:
        raise TransferError("refusing to report against an unlocked test split")
    if single["evaluated_split"] != multi["evaluated_split"]:
        raise TransferError("bundles evaluate different splits")

    paired = _paired(single, multi)
    # Seed class travels on the records themselves, which is the only place it
    # is guaranteed to exist in both bundles.
    frozen_seeds = {
        r["seed"] for r in multi["records"]
        if r.get("seed_class", "frozen_protocol") == "frozen_protocol"
    }
    if not frozen_seeds:
        raise TransferError("multi-task bundle declares no frozen-protocol seeds")
    if not paired:
        raise TransferError("no arm appears in both bundles")

    beff.prepare_output_dir(args.output_dir)

    rows: list[dict[str, Any]] = []
    for scope, keep in (
        ("frozen_protocol", lambda s: s in frozen_seeds),
        ("all_seeds", lambda s: True),
    ):
        for arm, per_seed in paired.items():
            seeds = [s for s in sorted(per_seed) if keep(s)]
            if not seeds:
                continue
            single_values = [per_seed[s][0] for s in seeds]
            multi_values = [per_seed[s][1] for s in seeds]
            diffs = [m - x for x, m in zip(single_values, multi_values, strict=True)]
            low, high = _bootstrap_ci(diffs)
            rows.append({
                "arm_id": arm,
                "seed_scope": scope,
                "seeds": len(seeds),
                "target": TARGET,
                "single_head_mean": statistics.fmean(single_values),
                "multi_task_head_mean": statistics.fmean(multi_values),
                "mean_paired_difference": statistics.fmean(diffs),
                "ci95_low": low,
                "ci95_high": high,
                "classification": _classify(low, high),
                "seeds_favouring_single_head": sum(1 for d in diffs if d > 0),
                "ratio_multi_over_single": (
                    statistics.fmean(multi_values) / statistics.fmean(single_values)
                ),
                "per_seed_difference": ";".join(f"{d:.6g}" for d in diffs),
            })

    beff.publish_csv(args.output_dir / "transfer.csv", rows)
    beff.publish_json(args.output_dir / "transfer_v1.json", {
        "artifact_role": "kuairand_pure_single_target_transfer_report",
        "schema_version": 1,
        "study": "kuairand_pure_single_target_watch_time",
        "is_the_frozen_single_target_study": False,
        "target": TARGET,
        "evaluated_split": single["evaluated_split"],
        "test_unlocked": False,
        "sign_convention": "difference = multi_task_head - single_head; "
                           "positive means the single head won, i.e. negative transfer",
        "interval_method": {
            "kind": "paired_percentile_bootstrap_over_seeds",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "nonsignificant_is_not_equality": True,
        "shared_view_manifest_sha256": view_hashes["single"],
        "rows": rows,
    })

    for scope in ("frozen_protocol", "all_seeds"):
        subset = [r for r in rows if r["seed_scope"] == scope]
        if not subset:
            continue
        print(f"\n=== {scope} ===")
        print(f"{'arm':<24}{'single':>8}{'multi':>9}{'diff':>9}"
              f"{'95% CI':>22}  classification")
        for row in sorted(subset, key=lambda r: -r["mean_paired_difference"]):
            interval = f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]"
            print(
                f"{row['arm_id']:<24}{row['single_head_mean']:>8.3f}"
                f"{row['multi_task_head_mean']:>9.3f}"
                f"{row['mean_paired_difference']:>+9.3f}{interval:>22}  "
                f"{row['classification']}"
            )
    print(f"\nwrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
