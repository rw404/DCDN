#!/usr/bin/env python
"""Matched serving-performance benchmark for the multi-target model matrix.

The measured unit is the **full multi-target forward with every head**, which
is what a ranker actually pays at serving time.  Per the frozen protocol:

* one idle A100, ``eval()`` and ``inference_mode()``;
* one precision policy for every row;
* device-resident inputs, so preprocessing, the DataLoader, and the
  host-to-device copy are excluded;
* batches ``{1, 64, 1024}``;
* 200 warmups, then 1,000 synchronized forwards, repeated for 5 independent
  timing rounds.

A comparison may be called **runtime matched** only when both hold:

* ``p95(B=1)_dcdn <= 1.10 * p95(B=1)_comparator``;
* ``throughput_dcdn >= 0.90 * throughput_comparator`` at the shared maximum
  batch.

When either fails, the pair is reported as a Pareto trade-off and the phrase
"at the same serving performance" may not be used.  This script computes and
records that verdict rather than leaving it to prose.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dcdn.models.afn import AFNPlusConfig  # noqa: E402
from dcdn.models.cross import (  # noqa: E402
    DCNv2CapacityCoreConfig,
    DCNv2FullRankConfig,
)
from dcdn.models.dcdn import DCDNCoreConfig  # noqa: E402
from dcdn.models.field import MLPConfig  # noqa: E402
from dcdn.multitask_capacity import (  # noqa: E402
    common_counts,
    dcdn_counts,
    ordinary_dcnv2_counts,
)
from dcdn.synthetic_multiquotient import COLUMNS, SPEC  # noqa: E402
from dcdn.synthetic_runtime import MultiQuotientModel, encoder_spec  # noqa: E402

EMBEDDING_DIM = 8
REPRESENTATION_DIM = 64
HEAD_HIDDEN_DIMS = (128, 64)

BATCHES = (1, 64, 1024)
WARMUPS = 200
FORWARDS = 1000
ROUNDS = 5

RUNTIME_MATCH_LATENCY_FACTOR = 1.10
RUNTIME_MATCH_THROUGHPUT_FACTOR = 0.90


#: A one-minute load average above this makes the p95/p99 tail unreliable.
#: The protocol requires an *idle* A100; a busy host inflates the tail without
#: touching the median, so the automatic runtime-match verdict is marked
#: provisional rather than silently reported as protocol-compliant.
IDLE_LOAD_AVERAGE_CEILING = 2.0


def _host_load() -> dict[str, Any]:
    one, five, fifteen = os.getloadavg()
    return {
        "load_average_1m": one,
        "load_average_5m": five,
        "load_average_15m": fifteen,
        "idle_ceiling": IDLE_LOAD_AVERAGE_CEILING,
        "host_idle": one <= IDLE_LOAD_AVERAGE_CEILING,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no samples")
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


@torch.inference_mode()
def _time_one_round(
    model: nn.Module,
    features: torch.Tensor,
    validity: torch.Tensor,
    device: torch.device,
) -> list[float]:
    samples: list[float] = []
    for _ in range(FORWARDS):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model(features, validity)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


@torch.inference_mode()
def measure(
    model: nn.Module, batch: int, device: torch.device, seed: int = 20260820
) -> dict[str, Any]:
    model = model.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    width = model.spec.operand_dim
    features = (
        torch.rand(batch, width, generator=generator) * 9.0 + 1.0
    ).to(device)
    validity = torch.ones_like(features, dtype=torch.bool)

    for _ in range(WARMUPS):
        model(features, validity)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    rounds = [_time_one_round(model, features, validity, device) for _ in range(ROUNDS)]
    pooled = [value for round_samples in rounds for value in round_samples]
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )

    p50, p95, p99 = (
        _percentile(pooled, 0.50),
        _percentile(pooled, 0.95),
        _percentile(pooled, 0.99),
    )
    return {
        "batch": batch,
        "rounds": ROUNDS,
        "forwards_per_round": FORWARDS,
        "warmups": WARMUPS,
        "batch_latency_ms_p50": p50,
        "batch_latency_ms_p95": p95,
        "batch_latency_ms_p99": p99,
        "per_example_latency_ms_p50": p50 / batch,
        "per_example_latency_ms_p95": p95 / batch,
        "per_example_latency_ms_p99": p99 / batch,
        "throughput_examples_per_second": batch / (p50 / 1e3),
        "per_round_p50_ms": [_percentile(r, 0.50) for r in rounds],
        "round_to_round_p50_spread_ms": (
            max(_percentile(r, 0.50) for r in rounds)
            - min(_percentile(r, 0.50) for r in rounds)
        ),
        "peak_vram_bytes": peak_vram,
    }


def _arms(capacity: dict[str, int]) -> dict[str, tuple[Any, bool]]:
    return {
        "mlp": (MLPConfig(hidden_dims=(512, 256, 128), dropout=0.0), False),
        "dcnv2_core": (
            DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1), False
        ),
        "dcdn_core_l1": (
            DCDNCoreConfig(
                num_layers=1, div_eps=1e-2, product_init_scale=0.1,
                div_init_scale=0.1, runtime_value_checks=False,
            ),
            False,
        ),
        "dcnv2_capacity": (
            DCNv2CapacityCoreConfig(
                num_layers=capacity["num_layers"],
                low_rank=capacity["low_rank"],
                num_experts=capacity["num_experts"],
                product_init_scale=0.1,
            ),
            False,
        ),
        "afn_plus": (
            AFNPlusConfig(
                logarithmic_neurons=128, afn_hidden_dims=(512,),
                dnn_hidden_dims=(512,), log_floor=1e-6, exp_clip=None,
                dropout=0.0, batch_norm=True,
            ),
            True,
        ),
    }


def _verdict(
    dcdn: dict[int, dict[str, Any]],
    comparator: dict[int, dict[str, Any]],
    shared_max_batch: int,
) -> dict[str, Any]:
    latency_ratio = (
        dcdn[1]["batch_latency_ms_p95"] / comparator[1]["batch_latency_ms_p95"]
    )
    # The threshold is defined on p95, but p95 is the quantity a busy host
    # corrupts.  The median ratio is reported alongside it so a contaminated
    # verdict is visible rather than merely suspected: on an idle host the two
    # ratios agree closely, and a large disagreement is itself the evidence
    # that the run is not protocol-valid.
    median_ratio = (
        dcdn[1]["batch_latency_ms_p50"] / comparator[1]["batch_latency_ms_p50"]
    )
    throughput_ratio = (
        dcdn[shared_max_batch]["throughput_examples_per_second"]
        / comparator[shared_max_batch]["throughput_examples_per_second"]
    )
    latency_ok = latency_ratio <= RUNTIME_MATCH_LATENCY_FACTOR
    throughput_ok = throughput_ratio >= RUNTIME_MATCH_THROUGHPUT_FACTOR
    matched = latency_ok and throughput_ok
    return {
        "p95_batch1_ratio": latency_ratio,
        "p95_batch1_threshold": RUNTIME_MATCH_LATENCY_FACTOR,
        "p95_batch1_passes": latency_ok,
        "p50_batch1_ratio": median_ratio,
        "p50_batch1_would_pass": median_ratio <= RUNTIME_MATCH_LATENCY_FACTOR,
        "p50_and_p95_disagree": (
            (median_ratio <= RUNTIME_MATCH_LATENCY_FACTOR) is not latency_ok
        ),
        "throughput_ratio_at_shared_max_batch": throughput_ratio,
        "throughput_threshold": RUNTIME_MATCH_THROUGHPUT_FACTOR,
        "throughput_passes": throughput_ok,
        "shared_max_batch": shared_max_batch,
        "runtime_matched": matched,
        "reporting_requirement": (
            "may be described as runtime matched"
            if matched
            else "must be reported as a Pareto trade-off; the phrase 'at the "
                 "same serving performance' is not permitted for this pair"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--capacity-from",
        type=Path,
        default=REPO_ROOT / "results" / "synthetic_multiquotient"
        / "selected_configs_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "results" / "synthetic_multiquotient",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    frozen = json.loads(args.capacity_from.read_text(encoding="utf-8"))
    capacity = frozen["capacity_resolution"]["selected"]

    spec = encoder_spec(EMBEDDING_DIM, REPRESENTATION_DIM)
    common = common_counts(
        len(COLUMNS), REPRESENTATION_DIM, HEAD_HIDDEN_DIMS, len(SPEC)
    )
    analytic = {
        "dcdn_core_l1": dcdn_counts(len(COLUMNS), 1, common),
        "dcnv2_core": ordinary_dcnv2_counts(len(COLUMNS), 2, common),
        "dcnv2_capacity": capacity,
    }

    start_load = _host_load()
    print(f"device: {device}  batches={BATCHES}  warmups={WARMUPS}  "
          f"forwards={FORWARDS}  rounds={ROUNDS}")
    print(f"host load 1m/5m/15m: {start_load['load_average_1m']:.2f}/"
          f"{start_load['load_average_5m']:.2f}/"
          f"{start_load['load_average_15m']:.2f}  "
          f"idle={start_load['host_idle']}")
    if not start_load["host_idle"]:
        print("  WARNING: host is not idle; p95/p99 tails will be inflated and "
              "the runtime-match verdict is recorded as PROVISIONAL")
    measurements: dict[str, dict[int, dict[str, Any]]] = {}
    parameters: dict[str, dict[str, int]] = {}
    for arm_id, (config, afn) in _arms(capacity).items():
        model = MultiQuotientModel(
            spec, config, num_targets=len(SPEC),
            head_hidden_dims=HEAD_HIDDEN_DIMS, init_seed=args.seed,
            with_afn_view=afn,
        )
        parameters[arm_id] = model.parameter_counts()
        measurements[arm_id] = {
            batch: measure(model, batch, device) for batch in BATCHES
        }
        row = measurements[arm_id]
        print(f"  {arm_id:<16} "
              + "  ".join(
                  f"B={b}: p50={row[b]['batch_latency_ms_p50']:.4f} "
                  f"p95={row[b]['batch_latency_ms_p95']:.4f}"
                  for b in BATCHES
              ))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    shared_max = max(BATCHES)
    verdicts = {
        f"dcdn_core_l1_vs_{comparator}": _verdict(
            measurements["dcdn_core_l1"], measurements[comparator], shared_max
        )
        for comparator in ("dcnv2_core", "dcnv2_capacity", "afn_plus", "mlp")
    }
    for name, verdict in verdicts.items():
        print(f"  {name:<32} runtime_matched={verdict['runtime_matched']}  "
              f"p95_ratio={verdict['p95_batch1_ratio']:.3f}  "
              f"throughput_ratio={verdict['throughput_ratio_at_shared_max_batch']:.3f}")

    payload = {
        "schema_version": 1,
        "artifact_role": "matched_serving_performance_report",
        "protocol": {
            "forward_scope": "full_multi_target_forward_with_every_head",
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "mode": "eval_and_inference_mode",
            "precision_policy": "float32_parameters_no_autocast_identical_across_rows",
            "inputs": "device_resident",
            "excluded": ["preprocessing", "dataloader", "host_to_device_transfer"],
            "batches": list(BATCHES),
            "warmups": WARMUPS,
            "synchronized_forwards": FORWARDS,
            "independent_rounds": ROUNDS,
            "runtime_match_latency_factor": RUNTIME_MATCH_LATENCY_FACTOR,
            "runtime_match_throughput_factor": RUNTIME_MATCH_THROUGHPUT_FACTOR,
        },
        "seed": args.seed,
        "host_load_at_start": start_load,
        "host_load_at_end": _host_load(),
        "protocol_compliant_idle_host": (
            start_load["host_idle"] and _host_load()["host_idle"]
        ),
        "verdict_status": (
            "final"
            if start_load["host_idle"]
            else "provisional_host_not_idle_p95_tail_is_contaminated"
        ),
        "parameters": parameters,
        "analytic_counts": analytic,
        "measurements": {
            arm: {str(batch): row for batch, row in rows.items()}
            for arm, rows in measurements.items()
        },
        "runtime_match_verdicts": verdicts,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    (args.output_dir / "serving_performance_v1.json").write_text(
        text + "\n", encoding="utf-8"
    )
    (args.output_dir / "serving_performance_v1.json.sha256").write_text(
        f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}  "
        "serving_performance_v1.json\n",
        encoding="utf-8",
    )

    efficiency_rows = []
    for arm_id, rows in measurements.items():
        counts = analytic.get(arm_id) or {}
        for batch, row in rows.items():
            efficiency_rows.append({
                "arm_id": arm_id,
                "batch": batch,
                "batch_latency_ms_p50": row["batch_latency_ms_p50"],
                "batch_latency_ms_p95": row["batch_latency_ms_p95"],
                "batch_latency_ms_p99": row["batch_latency_ms_p99"],
                "per_example_latency_ms_p50": row["per_example_latency_ms_p50"],
                "per_example_latency_ms_p95": row["per_example_latency_ms_p95"],
                "per_example_latency_ms_p99": row["per_example_latency_ms_p99"],
                "throughput_examples_per_second": (
                    row["throughput_examples_per_second"]
                ),
                "round_to_round_p50_spread_ms": row["round_to_round_p50_spread_ms"],
                "peak_vram_bytes": row["peak_vram_bytes"],
                "parameters_total": parameters[arm_id]["total"],
                "parameters_matched": parameters[arm_id]["matched"],
                "parameters_interaction": parameters[arm_id]["interaction"],
                "flops_per_example": counts.get("flops", ""),
            })
    _write_csv(args.output_dir / "efficiency.csv", efficiency_rows)
    print(f"\nwrote {args.output_dir / 'serving_performance_v1.json'}")
    print(f"wrote {args.output_dir / 'efficiency.csv'}")
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
