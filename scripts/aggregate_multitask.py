#!/usr/bin/env python
"""Consolidate every completed multi-target cell into the published tables.

Reads each dataset's ``results_v1.json`` (and its optional
``serving_performance_v1.json``) and emits the four top-level tables required
by the frozen protocol:

``multitask_summary.csv``     one row per (dataset, arm, slice): Macro-NLoss
                              mean and SD across seeds, non-finite counts,
                              capacity, and scale consistency
``multitask_per_target.csv``  the mandatory per-target breakdown; the aggregate
                              is never published without it
``efficiency.csv``            latency percentiles, throughput, peak VRAM,
                              parameters, and FLOPs/example
``pareto.csv``                quality against cost, with the Pareto frontier
                              marked and the runtime-matched verdict carried
                              through so no pair can be described as
                              "at the same serving performance" unless it
                              actually passed both thresholds

Cells that were not executed are listed explicitly under ``coverage`` in
``multitask_index.json``.  Silent omission is forbidden: a missing cell must
read as missing, not as absent-because-unimportant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

MANDATORY_DATASETS = (
    "synthetic_multiquotient_3",
    "vk_lsvd_up0.01_ir0.01",
    "yambda_50m",
    "kuairand_pure",
    "kuairec_big_matrix",
)

SOURCES = {
    "synthetic_multiquotient_3": ("synthetic_multiquotient", "id"),
    "vk_lsvd_up0.01_ir0.01": ("vk_lsvd_multitask", "test"),
    "yambda_50m": ("yambda_multitask", "test"),
    "kuairand_pure": ("kuairand_multitask", "test"),
    "kuairec_big_matrix": ("kuairec_multitask", "test"),
}


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _group_by_arm(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in result["records"]:
        grouped.setdefault(record["arm_id"], []).append(record)
    return grouped


def summary_rows(dataset: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    slices = result["benchmark_manifest"]["evaluation_slices"]
    rows: list[dict[str, Any]] = []
    for arm_id, records in sorted(_group_by_arm(result).items()):
        for slice_name in slices:
            values = [r["slices"][slice_name]["macro_nloss"] for r in records]
            rows.append(
                {
                    "dataset": dataset,
                    "arm_id": arm_id,
                    "label": records[0]["label"],
                    "slice": slice_name,
                    "seeds": len(values),
                    "macro_nloss_mean": statistics.fmean(values),
                    "macro_nloss_sd": _sd(values),
                    "macro_nloss_per_seed": ";".join(f"{v:.8g}" for v in values),
                    "nonfinite_predictions": sum(
                        r["slices"][slice_name]["nonfinite_predictions"]
                        for r in records
                    ),
                    "reciprocal_floor_incidence_max": max(
                        max(
                            r["slices"][slice_name][
                                "reciprocal_floor_incidence"
                            ].values()
                        )
                        for r in records
                    ),
                    "parameters_total": records[0]["parameters"]["total"],
                    "parameters_matched": records[0]["parameters"]["matched"],
                    "parameters_interaction": records[0]["parameters"]["interaction"],
                    "flops_per_example": (
                        (records[0].get("analytic_counts") or {}).get("flops", "")
                    ),
                    "scale_consistency_median": statistics.fmean(
                        r["scale_consistency"]["median_relative_deviation"]
                        for r in records
                    ),
                    "learning_rate": records[0].get("learning_rate", ""),
                }
            )
    return rows


def per_target_rows(dataset: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    slices = result["benchmark_manifest"]["evaluation_slices"]
    targets = [t["name"] for t in result["benchmark_manifest"]["targets"]]
    rows: list[dict[str, Any]] = []
    for arm_id, records in sorted(_group_by_arm(result).items()):
        for slice_name in slices:
            for target in targets:
                normalized = [
                    r["slices"][slice_name]["per_target_normalized"][target]
                    for r in records
                ]
                raw = [
                    r["slices"][slice_name]["per_target_raw"][target] for r in records
                ]
                rows.append(
                    {
                        "dataset": dataset,
                        "arm_id": arm_id,
                        "label": records[0]["label"],
                        "slice": slice_name,
                        "target": target,
                        "seeds": len(normalized),
                        "normalized_loss_mean": statistics.fmean(normalized),
                        "normalized_loss_sd": _sd(normalized),
                        "raw_loss_mean": statistics.fmean(raw),
                        "observed_labels": records[0]["slices"][slice_name][
                            "observed_labels"
                        ][target],
                    }
                )
    return rows


def efficiency_rows(dataset: str, serving: dict[str, Any]) -> list[dict[str, Any]]:
    analytic = serving.get("analytic_counts", {})
    rows: list[dict[str, Any]] = []
    for arm_id, batches in sorted(serving["measurements"].items()):
        for batch, row in sorted(batches.items(), key=lambda kv: int(kv[0])):
            rows.append(
                {
                    "dataset": dataset,
                    "arm_id": arm_id,
                    "batch": int(batch),
                    "batch_latency_ms_p50": row["batch_latency_ms_p50"],
                    "batch_latency_ms_p95": row["batch_latency_ms_p95"],
                    "batch_latency_ms_p99": row["batch_latency_ms_p99"],
                    "per_example_latency_ms_p50": row["per_example_latency_ms_p50"],
                    "per_example_latency_ms_p95": row["per_example_latency_ms_p95"],
                    "per_example_latency_ms_p99": row["per_example_latency_ms_p99"],
                    "throughput_examples_per_second": row[
                        "throughput_examples_per_second"
                    ],
                    "round_to_round_p50_spread_ms": row["round_to_round_p50_spread_ms"],
                    "peak_vram_bytes": row["peak_vram_bytes"],
                    "parameters_total": serving["parameters"][arm_id]["total"],
                    "parameters_matched": serving["parameters"][arm_id]["matched"],
                    "parameters_interaction": serving["parameters"][arm_id][
                        "interaction"
                    ],
                    "flops_per_example": (analytic.get(arm_id) or {}).get("flops", ""),
                }
            )
    return rows


def pareto_rows(
    dataset: str,
    result: dict[str, Any],
    serving: dict[str, Any],
    primary_slice: str,
) -> list[dict[str, Any]]:
    """Quality against cost, with the non-dominated set marked.

    A row is on the frontier when no other row is simultaneously at least as
    good on quality and at least as cheap on p95 latency at ``B=1``.
    """

    grouped = _group_by_arm(result)
    verdicts = serving.get("runtime_match_verdicts", {})
    points: list[dict[str, Any]] = []
    for arm_id, batches in serving["measurements"].items():
        if arm_id not in grouped:
            continue
        largest = str(max(int(b) for b in batches))
        verdict = verdicts.get(f"dcdn_core_l1_vs_{arm_id}")
        points.append(
            {
                "dataset": dataset,
                "arm_id": arm_id,
                "label": grouped[arm_id][0]["label"],
                "quality_slice": primary_slice,
                "macro_nloss_mean": statistics.fmean(
                    r["slices"][primary_slice]["macro_nloss"] for r in grouped[arm_id]
                ),
                "p95_batch1_latency_ms": batches["1"]["batch_latency_ms_p95"],
                "throughput_at_max_batch": batches[largest][
                    "throughput_examples_per_second"
                ],
                "parameters_matched": serving["parameters"][arm_id]["matched"],
                "flops_per_example": (
                    (serving.get("analytic_counts", {}).get(arm_id) or {}).get(
                        "flops", ""
                    )
                ),
                "runtime_matched_vs_dcdn": (
                    "" if verdict is None else verdict["runtime_matched"]
                ),
                "runtime_match_note": (
                    "" if verdict is None else verdict["reporting_requirement"]
                ),
            }
        )

    for point in points:
        point["on_pareto_frontier"] = not any(
            other is not point
            and other["macro_nloss_mean"] <= point["macro_nloss_mean"]
            and other["p95_batch1_latency_ms"] <= point["p95_batch1_latency_ms"]
            and (
                other["macro_nloss_mean"] < point["macro_nloss_mean"]
                or other["p95_batch1_latency_ms"] < point["p95_batch1_latency_ms"]
            )
            for other in points
        )
    return sorted(points, key=lambda p: p["macro_nloss_mean"])


def registry_rows(dataset: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per executed (dataset, arm, seed) training unit.

    This is the multi-target track's run registry.  It is deliberately kept
    separate from ``runs/registry/``, which is owned by the single-target
    harness and is actively written by a concurrent session; adding foreign
    records there could break that session's validators.
    """

    rows: list[dict[str, Any]] = []
    for record in result["records"]:
        training = record.get("training") or {}
        rows.append(
            {
                "dataset": dataset,
                "arm_id": record["arm_id"],
                "label": record["label"],
                "seed": record["seed"],
                "learning_rate": record.get("learning_rate"),
                "benchmark_manifest_sha256": result.get("benchmark_manifest_sha256"),
                "selected_configs_sha256": result.get("selected_configs_sha256"),
                "validation_macro_nloss": training.get("validation_macro_nloss"),
                "best_step": training.get("best_step"),
                "nonfinite_training_batches": training.get(
                    "nonfinite_training_batches"
                ),
                "training_seconds": training.get("seconds"),
                "parameters_total": record["parameters"]["total"],
                "parameters_matched": record["parameters"]["matched"],
                "parameters_interaction": record["parameters"]["interaction"],
                "result_path": f"results/{SOURCES[dataset][0]}/results_v1.json",
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results" / "multitask"
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    per_target: list[dict[str, Any]] = []
    efficiency: list[dict[str, Any]] = []
    pareto: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}

    for dataset, (directory_name, primary_slice) in SOURCES.items():
        directory = args.results_root / directory_name
        result = _load(directory / "results_v1.json")
        serving = _load(directory / "serving_performance_v1.json")
        if result is None:
            coverage[dataset] = {
                "executed": False,
                "mandatory": dataset in MANDATORY_DATASETS,
                "reason": f"no results_v1.json under results/{directory_name}",
            }
            continue
        summary.extend(summary_rows(dataset, result))
        per_target.extend(per_target_rows(dataset, result))
        registry.extend(registry_rows(dataset, result))
        if serving is not None:
            efficiency.extend(efficiency_rows(dataset, serving))
            pareto.extend(pareto_rows(dataset, result, serving, primary_slice))
        coverage[dataset] = {
            "executed": True,
            "mandatory": dataset in MANDATORY_DATASETS,
            "arms": sorted({r["arm_id"] for r in result["records"]}),
            "seeds": sorted({r["seed"] for r in result["records"]}),
            "records": len(result["records"]),
            "serving_benchmark_present": serving is not None,
            "benchmark_manifest_sha256": result.get("benchmark_manifest_sha256"),
            "selected_configs_sha256": result.get("selected_configs_sha256"),
        }

    counts = {
        "multitask_summary.csv": write_csv(
            args.output_dir / "multitask_summary.csv", summary
        ),
        "multitask_per_target.csv": write_csv(
            args.output_dir / "multitask_per_target.csv", per_target
        ),
        "efficiency.csv": write_csv(args.output_dir / "efficiency.csv", efficiency),
        "pareto.csv": write_csv(args.output_dir / "pareto.csv", pareto),
        "run_registry.csv": write_csv(args.output_dir / "run_registry.csv", registry),
    }

    index = {
        "schema_version": 1,
        "artifact_role": "multitask_publication_index",
        "mandatory_datasets": list(MANDATORY_DATASETS),
        "coverage": coverage,
        "table_row_counts": counts,
        "aggregate_never_published_without_per_target": True,
        "unexecuted_cells_are_listed_not_omitted": True,
    }
    text = json.dumps(index, indent=2, sort_keys=True)
    (args.output_dir / "multitask_index.json").write_text(text + "\n", encoding="utf-8")
    (args.output_dir / "multitask_index.json.sha256").write_text(
        f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}  multitask_index.json\n",
        encoding="utf-8",
    )

    for name, count in counts.items():
        print(f"{name:<28} {count} rows")
    executed = [k for k, v in coverage.items() if v["executed"]]
    missing = [k for k, v in coverage.items() if not v["executed"] and v["mandatory"]]
    print(f"executed datasets : {executed}")
    print(f"MISSING mandatory : {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
