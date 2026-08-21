#!/usr/bin/env python
"""RQ2 addendum: serving cost across the production batch range.

The frozen protocol fixes its runtime-matched threshold at ``p95(B=1)``, and
`docs/rq2_fused_inference.md` reports that point. But a production ranker uses
dynamic batching over roughly ``1..16000``, and ``B=1`` is the *worst* case for
this comparison: at one row per call both architectures are launch-bound, so
the measurement is dominated by fixed overhead and DCDN's lower arithmetic
cost cannot show. A single operating point is not a serving-cost answer.

This sweep measures the whole range on a power-of-two grid to ``16384``, so the
curve — not one point — carries the claim. Two regimes are expected and the
crossover between them is the thing worth locating:

* **launch-bound** (small ``B``) — fixed per-call overhead dominates; per-example
  latency falls roughly as ``1/B``; architectures look alike;
* **work-bound** (large ``B``) — arithmetic and memory traffic dominate; DCDN's
  2.2x lower interaction FLOPs at ranker width should begin to pay.

Each arm runs under its best available implementation, as established by
`scripts/benchmark_fused_inference.py`: DCDN fused, DCNv2 reference.

Sweep protocol, deliberately lighter than the headline protocol and declared as
such: 50 warmups, 300 synchronized forwards, 3 independent rounds per point.
The headline ``{1, 64, 1024}`` numbers stay with the 200/1000/5 protocol in the
other studies; this grid exists to trace a shape across 15 batch sizes, not to
re-decide the frozen threshold.

Trains nothing; reads no dataset row, label, or test metric.

Usage::

    python scripts/benchmark_batch_sweep.py --device cuda:0 --allow-busy-host
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dcdn.fused_inference import (  # noqa: E402
    TRITON_AVAILABLE,
    FusedDCDNCore,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beff = _load("benchmark_efficiency", REPO_ROOT / "scripts" / "benchmark_efficiency.py")
bfus = _load(
    "benchmark_fused_inference", REPO_ROOT / "scripts" / "benchmark_fused_inference.py"
)

SWEEP_BATCHES = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
)
SWEEP_WARMUPS = 50
SWEEP_FORWARDS = 300
SWEEP_ROUNDS = 3

#: Arm -> best available implementation, from the fused study.
BEST_IMPLEMENTATION = {
    "dcnv2_reduced": "reference",
    "dcdn_reduced": "fused",
    "dcnv2_capacity_reduced": "reference",
    "dcnv2_full": "reference",
    "dcdn_full": "fused",
}


@torch.inference_mode()
def _measure_point(
    model: nn.Module,
    geometry,
    regime: str,
    batch: int,
    device: torch.device,
    capture_graph: bool = False,
) -> dict[str, Any]:
    categorical, operands, validity = beff._make_inputs(
        geometry, regime, batch, device
    )
    for _ in range(SWEEP_WARMUPS):
        model(categorical, operands, validity)
    torch.cuda.synchronize(device)

    graph: torch.cuda.CUDAGraph | None = None
    if capture_graph:
        # Capture on a side stream, as the CUDA graph API requires. The replay
        # reuses the captured input tensors, which is exactly the production
        # shape-specialized serving pattern: one graph per batch size.
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(3):
                model(categorical, operands, validity)
        torch.cuda.current_stream(device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model(categorical, operands, validity)
        torch.cuda.synchronize(device)

    def run() -> None:
        if graph is not None:
            graph.replay()
        else:
            model(categorical, operands, validity)

    torch.cuda.reset_peak_memory_stats(device)

    wall: list[float] = []
    dev: list[float] = []
    for _ in range(SWEEP_ROUNDS):
        for _ in range(SWEEP_FORWARDS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start_event.record()
            started = time.perf_counter()
            run()
            end_event.record()
            torch.cuda.synchronize(device)
            wall.append((time.perf_counter() - started) * 1e3)
            dev.append(start_event.elapsed_time(end_event))

    p50 = beff._percentile(wall, 0.50)
    p95 = beff._percentile(wall, 0.95)
    d50 = beff._percentile(dev, 0.50)
    return {
        "batch": batch,
        "batch_latency_ms_p50": p50,
        "batch_latency_ms_p95": p95,
        "per_example_latency_us_p50": (p50 / batch) * 1e3,
        "device_latency_ms_p50": d50,
        "throughput_examples_per_second": batch / (p50 / 1e3),
        "device_throughput_examples_per_second": batch / (d50 / 1e3),
        "wall_over_device_p50": p50 / d50 if d50 > 0 else None,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_graph": capture_graph,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--geometry", default="ranker")
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="capture each (arm, batch) into a CUDA graph and time the "
             "replay; applied identically to every arm",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "efficiency" / "rq2_batch_sweep_v1",
    )
    parser.add_argument("--allow-busy-host", action="store_true")
    args = parser.parse_args(argv)

    if not TRITON_AVAILABLE:
        raise SystemExit("Triton is required for the fused arms")
    device = torch.device(args.device)
    if device.type == "cuda":
        # Without this, CUDA events are created on the default device while the
        # work runs on `device`, and bracketing a graph replay with them fails
        # outright. It only shows on a non-default device, which is why the
        # eager path on cuda:0 never surfaced it.
        torch.cuda.set_device(device)
    state = beff.host_state(device)
    print(
        f"device: {device}  geometry: {args.geometry}  batches: "
        f"{SWEEP_BATCHES[0]}..{SWEEP_BATCHES[-1]}  "
        f"warmups={SWEEP_WARMUPS} forwards={SWEEP_FORWARDS} rounds={SWEEP_ROUNDS}"
    )
    print(
        f"host load 1m/5m/15m: {state['load_average_1m']:.2f}/"
        f"{state['load_average_5m']:.2f}/{state['load_average_15m']:.2f}  "
        f"idle={state['host_idle']}"
    )
    if not state["host_idle"] and not args.allow_busy_host:
        raise SystemExit("host is not idle; pass --allow-busy-host to proceed")

    beff.prepare_output_dir(args.output_dir)
    geometry = next(
        g for g in beff.GEOMETRIES if g.geometry_id == args.geometry
    )
    capacity = beff.resolve_capacity(geometry)
    arms = {
        a.arm_id: a
        for a in beff.build_arms(geometry, capacity)
        if a.arm_id in BEST_IMPLEMENTATION
    }

    manifest = {
        "artifact_role": "rq2_batch_sweep_pre_measurement_manifest",
        "schema_version": 1,
        "purpose": (
            "trace serving cost across the production batch range rather than "
            "at the single frozen operating point B=1, which is the worst case "
            "for this comparison because both architectures are launch-bound "
            "there"
        ),
        "geometry": geometry.geometry_id,
        "batches": list(SWEEP_BATCHES),
        "sweep_protocol": {
            "warmups": SWEEP_WARMUPS,
            "synchronized_forwards": SWEEP_FORWARDS,
            "independent_rounds": SWEEP_ROUNDS,
            "note": (
                "lighter than the headline 200/1000/5 protocol; this grid "
                "traces a shape across 15 batch sizes and does not re-decide "
                "the frozen threshold"
            ),
        },
        "best_implementation": BEST_IMPLEMENTATION,
        "trains_any_model": False,
        "reads_dataset_rows": False,
        "reads_test_metrics": False,
        "host_state_at_start": state,
    }
    manifest_digest = beff.publish_json(
        args.output_dir / "pre_measurement_manifest_v1.json", manifest
    )
    print(f"pre-measurement manifest sha256: {manifest_digest}\n")

    measurements: dict[str, dict[int, dict[str, Any]]] = {}
    for arm_id, arm in arms.items():
        implementation = BEST_IMPLEMENTATION[arm_id]
        model = beff.instantiate(arm, geometry).to(device).eval()
        if implementation == "fused":
            model = bfus._swap_to_fused(model, bfus.FUSABLE[arm_id])
            # The fused DCDN carrier used to read its validity tensor into
            # Python on every forward. That device-to-host synchronization cost
            # 4.5% of the B=1 forward and made the module impossible to capture
            # in a CUDA graph, and the comparator has no equivalent
            # synchronization anywhere - so it was charging a guard to the
            # architecture. The invariant is verified once here, on the same
            # tensors the sweep will use, and then taken off the hot path.
            _, _, validity = beff._make_inputs(geometry, arm.regime, 8, device)
            for module in model.modules():
                if isinstance(module, FusedDCDNCore):
                    module.verify_all_valid(validity)
                    module.validate_validity_per_call = False
        rows: dict[int, dict[str, Any]] = {}
        for batch in SWEEP_BATCHES:
            rows[batch] = _measure_point(
                model, geometry, arm.regime, batch, device,
                capture_graph=args.cuda_graphs,
            )
            row = rows[batch]
            print(
                f"  {arm_id:<24} B={batch:<6} p50={row['batch_latency_ms_p50']:8.4f}ms "
                f"per-ex={row['per_example_latency_us_p50']:9.3f}us "
                f"tput={row['throughput_examples_per_second']:12,.0f}/s"
            )
        measurements[arm_id] = rows
        del model
        torch.cuda.empty_cache()
        print()

    # Where does the DCDN/DCNv2 comparison change character?
    ratios = []
    for batch in SWEEP_BATCHES:
        subject = measurements["dcdn_reduced"][batch]
        comparator = measurements["dcnv2_reduced"][batch]
        ratios.append(
            {
                "batch": batch,
                "latency_ratio_p50": (
                    subject["batch_latency_ms_p50"]
                    / comparator["batch_latency_ms_p50"]
                ),
                "throughput_ratio": (
                    subject["throughput_examples_per_second"]
                    / comparator["throughput_examples_per_second"]
                ),
            }
        )

    print("=== DCDN / DCNv2 across the batch range ===")
    for entry in ratios:
        print(
            f"  B={entry['batch']:<6} latency x{entry['latency_ratio_p50']:.3f}  "
            f"throughput x{entry['throughput_ratio']:.3f}"
        )

    results = {
        "artifact_role": "rq2_batch_sweep_result",
        "schema_version": 1,
        "pre_measurement_manifest_sha256": manifest_digest,
        "geometry": geometry.geometry_id,
        "batches": list(SWEEP_BATCHES),
        "best_implementation": BEST_IMPLEMENTATION,
        "host_state_at_start": state,
        "host_state_at_end": beff.host_state(device),
        "measurements": {
            arm: {str(b): row for b, row in rows.items()}
            for arm, rows in measurements.items()
        },
        "dcdn_over_dcnv2": ratios,
    }
    digest = beff.publish_json(args.output_dir / "results_v1.json", results)

    csv_rows = []
    for arm_id, rows in measurements.items():
        for row in rows.values():
            csv_rows.append(
                {
                    "geometry": geometry.geometry_id,
                    "arm_id": arm_id,
                    "implementation": BEST_IMPLEMENTATION[arm_id],
                    **row,
                }
            )
    csv_digest = beff.publish_csv(args.output_dir / "batch_sweep.csv", csv_rows)
    print(f"\npublished results_v1.json  sha256={digest}")
    print(f"published batch_sweep.csv  sha256={csv_digest} ({len(csv_rows)} rows)")
    print(f"output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
