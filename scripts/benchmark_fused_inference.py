#!/usr/bin/env python
"""RQ2 addendum: serving cost under the best available implementation.

`scripts/benchmark_efficiency.py` measured the reference PyTorch forward. That
measurement is honest but it conflates architecture with implementation: DCDN
decomposes into more elementwise passes than DCNv2, so at serving batch sizes
it pays more kernel launches even though it carries fewer parameters and fewer
FLOPs. Attributing that gap to the architecture would be wrong.

This script removes the implementation variable. It measures every arm twice —
once with the reference implementation and once with a fused Triton
implementation — and reports the comparison under each architecture's **best
available** implementation. Fusing only DCDN would replace one unfair
comparison with its mirror image, so `dcdn.fused_inference` provides fused
kernels for the DCNv2 cross as well, and whichever implementation is faster for
a given arm is the one that represents it.

Equivalence is verified before any timing: the fused core must reproduce the
audited module's output within float32 reassociation tolerance on every
geometry and batch measured. The fused path changes the schedule, never a
parameter and never a result, and is used only for inference cost — never for
a quality number.

Usage::

    python scripts/benchmark_fused_inference.py --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dcdn.fused_inference import (  # noqa: E402
    TRITON_AVAILABLE,
    FusedCrossNet,
    FusedDCDNCore,
    FusedInferenceError,
    max_abs_difference,
)


def _load_efficiency_module():
    """Reuse the frozen geometries, arms, and publication helpers."""

    spec = importlib.util.spec_from_file_location(
        "benchmark_efficiency", REPO_ROOT / "scripts" / "benchmark_efficiency.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_efficiency"] = module
    spec.loader.exec_module(module)
    return module


beff = _load_efficiency_module()

#: Float32 GEMM reassociation moves the last couple of mantissa bits. Anything
#: beyond this is a real disagreement, not rounding.
EQUIVALENCE_TOLERANCE = 1e-3

#: Arms whose interaction block has a fused twin.
FUSABLE = {
    "dcdn_reduced": "dcdn",
    "dcdn_full": "dcdn",
    "dcdn_reduced_l2": "dcdn",
    "dcnv2_reduced": "cross",
    "dcnv2_full": "cross",
}


def _swap_to_fused(model: nn.Module, kind: str) -> nn.Module:
    """Replace the interaction block in place with its fused twin."""

    interaction = model.model.interaction
    if kind == "dcdn":
        interaction.core = FusedDCDNCore.from_core(interaction.core).to(
            next(model.parameters()).device
        )
    elif kind == "cross":
        fused = FusedCrossNet.from_interaction(interaction).to(
            next(model.parameters()).device
        )
        interaction.layers = fused._weight  # keep the module registered
        interaction.forward = lambda batch: fused(batch.flat_features)  # type: ignore[method-assign]
    else:
        raise FusedInferenceError(f"unknown fusable kind {kind!r}")
    return model


@torch.inference_mode()
def _verify_equivalence(
    arm, geometry, device: torch.device, kind: str
) -> dict[str, Any]:
    """Fused output must match the audited module before anything is timed."""

    worst = 0.0
    for batch in beff.BATCHES:
        reference = beff.instantiate(arm, geometry).to(device).eval()
        cat, ops, val = beff._make_inputs(geometry, arm.regime, batch, device)
        expected = reference(cat, ops, val)
        fused_model = _swap_to_fused(
            beff.instantiate(arm, geometry).to(device).eval(), kind
        )
        actual = fused_model(cat, ops, val)
        worst = max(worst, max_abs_difference(expected, actual))
        del reference, fused_model
    if worst > EQUIVALENCE_TOLERANCE:
        raise FusedInferenceError(
            f"{arm.arm_id}: fused output differs by {worst:.3e}, above the "
            f"{EQUIVALENCE_TOLERANCE:.0e} reassociation tolerance"
        )
    return {"max_abs_difference": worst, "tolerance": EQUIVALENCE_TOLERANCE}


@torch.inference_mode()
def _measure(model: nn.Module, geometry, regime: str, batch: int, device) -> dict:
    return beff.measure(model, geometry, regime, batch, device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "efficiency" / "rq2_fused_v1",
    )
    parser.add_argument("--allow-busy-host", action="store_true")
    args = parser.parse_args(argv)

    if not TRITON_AVAILABLE:
        raise SystemExit("Triton is unavailable; the fused study cannot run")

    device = torch.device(args.device)
    state = beff.host_state(device)
    print(
        f"device: {device}  batches={beff.BATCHES}  warmups={beff.WARMUPS}  "
        f"forwards={beff.FORWARDS}  rounds={beff.ROUNDS}"
    )
    print(
        f"host load 1m/5m/15m: {state['load_average_1m']:.2f}/"
        f"{state['load_average_5m']:.2f}/{state['load_average_15m']:.2f}  "
        f"idle={state['host_idle']}"
    )
    if not state["host_idle"] and not args.allow_busy_host:
        raise SystemExit("host is not idle; pass --allow-busy-host to proceed")

    beff.prepare_output_dir(args.output_dir)

    capacity = {g.geometry_id: beff.resolve_capacity(g) for g in beff.GEOMETRIES}
    equivalence: dict[str, dict[str, Any]] = {}
    reference: dict[str, dict[str, dict[int, dict]]] = {}
    fused: dict[str, dict[str, dict[int, dict]]] = {}
    parameters: dict[str, dict[str, dict[str, int]]] = {}

    manifest = {
        "artifact_role": "rq2_fused_inference_pre_measurement_manifest",
        "schema_version": 1,
        "purpose": (
            "separate architecture cost from implementation cost by measuring "
            "every arm under both the reference and a fused Triton "
            "implementation, and comparing under each architecture's best"
        ),
        "fused_arms": sorted(FUSABLE),
        "fuses_both_architectures": True,
        "equivalence_tolerance": EQUIVALENCE_TOLERANCE,
        "trains_any_model": False,
        "reads_dataset_rows": False,
        "reads_test_metrics": False,
        "protocol": {
            "batches": list(beff.BATCHES),
            "warmups": beff.WARMUPS,
            "synchronized_forwards": beff.FORWARDS,
            "independent_rounds": beff.ROUNDS,
            "timing": "wall_clock_synchronized_and_cuda_event_device_time",
        },
        "host_state_at_start": state,
        "triton_version": __import__("triton").__version__,
    }
    manifest_digest = beff.publish_json(
        args.output_dir / "pre_measurement_manifest_v1.json", manifest
    )
    print(f"pre-measurement manifest sha256: {manifest_digest}")

    for geometry in beff.GEOMETRIES:
        gid = geometry.geometry_id
        print(f"\n--- geometry: {gid} ---")
        reference[gid], fused[gid], parameters[gid] = {}, {}, {}
        for arm in beff.build_arms(geometry, capacity[gid]):
            if arm.arm_id not in FUSABLE:
                continue
            kind = FUSABLE[arm.arm_id]
            equivalence[f"{gid}:{arm.arm_id}"] = _verify_equivalence(
                arm, geometry, device, kind
            )

            model = beff.instantiate(arm, geometry).to(device).eval()
            parameters[gid][arm.arm_id] = model.parameter_counts()
            reference[gid][arm.arm_id] = {
                b: _measure(model, geometry, arm.regime, b, device)
                for b in beff.BATCHES
            }
            del model
            torch.cuda.empty_cache()

            fused_model = _swap_to_fused(
                beff.instantiate(arm, geometry).to(device).eval(), kind
            )
            fused[gid][arm.arm_id] = {
                b: _measure(fused_model, geometry, arm.regime, b, device)
                for b in beff.BATCHES
            }
            del fused_model
            torch.cuda.empty_cache()

            r95 = reference[gid][arm.arm_id][1]["batch_latency_ms_p95"]
            f95 = fused[gid][arm.arm_id][1]["batch_latency_ms_p95"]
            print(
                f"  {arm.arm_id:<20} ref p95={r95:.4f}  fused p95={f95:.4f}  "
                f"speedup={r95 / f95:.2f}x  "
                f"|delta|={equivalence[f'{gid}:{arm.arm_id}']['max_abs_difference']:.1e}"
            )

    # Best available implementation per arm, then the architecture comparison.
    best: dict[str, dict[str, dict[int, dict]]] = {}
    choice: dict[str, str] = {}
    for gid in reference:
        best[gid] = {}
        for arm_id in reference[gid]:
            r = reference[gid][arm_id][1]["batch_latency_ms_p95"]
            f = fused[gid][arm_id][1]["batch_latency_ms_p95"]
            use_fused = f < r
            best[gid][arm_id] = (
                fused[gid][arm_id] if use_fused else reference[gid][arm_id]
            )
            choice[f"{gid}:{arm_id}"] = "fused" if use_fused else "reference"

    shared_max = max(beff.BATCHES)
    verdicts: dict[str, dict[str, Any]] = {}
    for gid in best:
        for subject, comparator in (
            ("dcdn_reduced", "dcnv2_reduced"),
            ("dcdn_full", "dcnv2_full"),
        ):
            if subject in best[gid] and comparator in best[gid]:
                verdicts[f"{gid}:{subject}_vs_{comparator}"] = (
                    beff.runtime_match_verdict(
                        best[gid][subject], best[gid][comparator], shared_max
                    )
                )

    print("\n--- best-implementation verdicts ---")
    for key, verdict in verdicts.items():
        print(
            f"  {key:<44} matched={str(verdict['runtime_matched']):<5} "
            f"p95r={verdict['p95_batch1_ratio']:.3f} "
            f"tput={verdict['throughput_ratio_at_shared_max_batch']:.3f}"
        )

    contamination = beff.contamination_check(
        {f"{g}:ref": rows for g, rows in reference.items()}
        | {f"{g}:fused": rows for g, rows in fused.items()}
    )
    results = {
        "artifact_role": "rq2_fused_inference_result",
        "schema_version": 1,
        "pre_measurement_manifest_sha256": manifest_digest,
        "host_state_at_start": state,
        "host_state_at_end": beff.host_state(device),
        "contamination_check": contamination,
        "verdict_status": beff._verdict_status(state, contamination),
        "equivalence": equivalence,
        "implementation_choice": choice,
        "parameters": parameters,
        "measurements_reference": {
            g: {a: {str(b): r for b, r in rows.items()} for a, rows in arms.items()}
            for g, arms in reference.items()
        },
        "measurements_fused": {
            g: {a: {str(b): r for b, r in rows.items()} for a, rows in arms.items()}
            for g, arms in fused.items()
        },
        "best_implementation_verdicts": verdicts,
    }
    digest = beff.publish_json(args.output_dir / "results_v1.json", results)

    rows = []
    for gid in reference:
        for arm_id in reference[gid]:
            for impl, table in (("reference", reference), ("fused", fused)):
                for batch in beff.BATCHES:
                    row = table[gid][arm_id][batch]
                    rows.append(
                        {
                            "geometry": gid,
                            "arm_id": arm_id,
                            "implementation": impl,
                            "batch": batch,
                            "batch_latency_ms_p50": row["batch_latency_ms_p50"],
                            "batch_latency_ms_p95": row["batch_latency_ms_p95"],
                            "device_latency_ms_p95": row.get("device_latency_ms_p95"),
                            "throughput_examples_per_second": row[
                                "throughput_examples_per_second"
                            ],
                            "peak_vram_bytes": row["peak_vram_bytes"],
                            "is_best_for_arm": choice[f"{gid}:{arm_id}"] == impl,
                        }
                    )
    csv_digest = beff.publish_csv(args.output_dir / "fused_efficiency.csv", rows)
    print(f"\npublished results_v1.json      sha256={digest}")
    print(f"published fused_efficiency.csv sha256={csv_digest} ({len(rows)} rows)")
    print(f"output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
