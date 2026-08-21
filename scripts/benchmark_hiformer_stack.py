#!/usr/bin/env python
"""RQ2 addendum: serving cost with a HiFormer-style layer after the cross.

The cross-only stacks measured so far put DCDN and DCNv2 at runtime parity, and
the reduced feature view bought nothing at inference: appending 17 ``R_div``
columns changed latency by 0.2-1%. That is because a cross layer's cost is
roughly linear in width, so removing 17 of 78 operands saves almost nothing.

A production ranker usually does not stop at the cross. Once a component whose
cost grows **superlinearly** in the field count sits downstream, the feature
reduction stops being free real estate and starts being a saving. Self-attention
is the canonical such component: cost ``O(F^2 d)``. At ranker width the reduced
view carries 67 field tokens against the full view's 84, so the attention term
alone differs by ``(85/68)^2 = 1.56`` once the cross token is included.

This script measures that. Four stacks, identical except for the interaction
block and the feature regime::

    encoder -> fields [B, F, d] and flat [B, bypass+operands]
    cross(flat) -> crossed
    cross token = Linear(flat -> d)
    HiFormer-style heterogeneous attention over [cross token | fields]
    flatten -> adapter -> shared tower -> per-target heads

``dcnv2_reduced``  DCNv2 on 61 operands, 68 tokens
``dcdn_reduced``   DCDN-Core-L1 (fused) on 61 operands, 68 tokens
``dcnv2_full``     DCNv2 on 78 operands, 85 tokens  <- the comparison of interest
``dcdn_full``      DCDN-Core-L1 (fused) on 78 operands, 85 tokens

The question under test is whether **DCDN on the reduced view beats DCNv2 on
the full view** once the attention layer is present — a cost claim the cross-only
stack could not support.

Scope: this is a **cost** study. The HiFormer-style layer is outside the frozen
nine-model suite, nothing is trained, and no quality number is produced or
implied. Whether DCDN-REDUCED matches DCNv2-FULL in *accuracy* under this stack
is a separate, unmeasured question; contrast M2-M5 on the cross-only stack found
DCNv2-FULL better in-distribution.

Usage::

    python scripts/benchmark_hiformer_stack.py --device cuda:0 --allow-busy-host
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dcdn.fused_inference import TRITON_AVAILABLE, FusedDCDNCore  # noqa: E402
from dcdn.hiformer import HeterogeneousAttentionLayer  # noqa: E402
from dcdn.models.common import (  # noqa: E402
    DenseTower,
    HeadConfig,  # noqa: E402
    initialize_output_linear,
)
from dcdn.models.contracts import EncodedInteractionBatch  # noqa: E402
from dcdn.models.cross import DCNv2FullRankConfig  # noqa: E402
from dcdn.models.dcdn import DCDNCoreConfig  # noqa: E402
from dcdn.models.suite import build_controlled_model  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beff = _load("benchmark_efficiency", REPO_ROOT / "scripts" / "benchmark_efficiency.py")

NUM_HEADS = 4
FFN_RATIO = 2
INIT_SEED = 42
SWEEP_BATCHES = (1, 64, 256, 1024, 4096, 16384)
WARMUPS = 50
FORWARDS = 300
ROUNDS = 3
DIV_EPS = 1e-2
REPRESENTATION_DIM = 64
HEAD_HIDDEN_DIMS = (128, 64)


@dataclass(frozen=True)
class StackArm:
    arm_id: str
    label: str
    regime: str
    kind: str
    fused: bool


ARMS = (
    StackArm("dcnv2_reduced", "DCNv2 + HiFormer, reduced", "reduced", "dcnv2", False),
    StackArm("dcdn_reduced", "DCDN + HiFormer, reduced", "reduced", "dcdn", True),
    StackArm("dcnv2_full", "DCNv2 + HiFormer, full", "full", "dcnv2", False),
    StackArm("dcdn_full", "DCDN + HiFormer, full", "full", "dcdn", True),
)


class HiFormerStack(nn.Module):
    """Shared encoder, interaction block, one attention layer, shared heads."""

    def __init__(self, geometry, arm: StackArm) -> None:
        super().__init__()
        self.geometry = geometry
        self.arm = arm
        spec = geometry.spec(arm.regime)
        self.spec = spec
        operand_width = geometry.operand_dim(arm.regime)
        flat_width = geometry.bypass_dim() + operand_width
        embedding_dim = geometry.embedding_dim

        with torch.random.fork_rng(devices=[], enabled=True):
            torch.random.default_generator.manual_seed(INIT_SEED)
            self.encoder = beff.PerfEncoder(geometry, arm.regime)
            # The cross result enters attention as one extra token.
            self.cross_token = nn.Linear(flat_width, embedding_dim)
            # Two declared groups: the cross token joins the numeric operands.
            self.attention = HeterogeneousAttentionLayer(
                embedding_dim=embedding_dim,
                num_heads=NUM_HEADS,
                group_sizes=(
                    geometry.categorical_fields,
                    operand_width + 1,
                ),
                ffn_ratio=FFN_RATIO,
            )
            self.tower = DenseTower(
                (operand_width + 1 + geometry.categorical_fields) * embedding_dim,
                (REPRESENTATION_DIM,),
                0.0,
            )
            self.output = nn.Linear(REPRESENTATION_DIM, geometry.num_targets)
            initialize_output_linear(self.output)

        interaction_config = (
            DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1)
            if arm.kind == "dcnv2"
            else DCDNCoreConfig(
                num_layers=1,
                div_eps=DIV_EPS,
                product_init_scale=0.1,
                div_init_scale=0.1,
                runtime_value_checks=False,
            )
        )
        controlled = build_controlled_model(
            encoder_spec=spec,
            interaction_config=interaction_config,
            head_config=HeadConfig(
                hidden_dims=HEAD_HIDDEN_DIMS,
                dropout=0.0,
                output_dim=geometry.num_targets,
            ),
            init_seed=INIT_SEED,
        )
        # Only the interaction block is reused; this stack owns its own head.
        self.interaction = controlled.interaction
        self.num_tokens = self.attention.num_tokens

    def fuse(self) -> HiFormerStack:
        if self.arm.kind != "dcdn":
            return self
        self.interaction.core = FusedDCDNCore.from_core(self.interaction.core).to(
            next(self.parameters()).device
        )
        return self

    def forward(
        self,
        categorical: torch.Tensor,
        operands: torch.Tensor,
        validity: torch.Tensor,
    ) -> torch.Tensor:
        bypass, fields = self.encoder(categorical, operands)
        batch = EncodedInteractionBatch(
            bypass_features=bypass,
            operand_features=operands,
            field_embeddings=fields,
            denominator_validity=validity,
        )
        crossed = self.interaction(batch)
        token = self.cross_token(crossed).unsqueeze(1)
        # Categorical tokens first, then the cross token with the operands, so
        # the group boundaries the attention layer declares are respected.
        categorical_tokens = fields[:, : self.geometry.categorical_fields]
        operand_tokens = fields[:, self.geometry.categorical_fields :]
        tokens = torch.cat([categorical_tokens, token, operand_tokens], dim=1)
        mixed = self.attention(tokens)
        return self.output(self.tower(mixed.flatten(1)))

    def counts(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        embedding_ids = {
            id(p)
            for module in self.modules()
            if isinstance(module, nn.Embedding)
            for p in module.parameters()
        }
        return {
            "tokens": self.num_tokens,
            "interaction": count(self.interaction),
            "attention": count(self.attention),
            "tower_and_output": count(self.tower) + count(self.output),
            "non_embedding": sum(
                p.numel()
                for p in self.parameters()
                if p.requires_grad and id(p) not in embedding_ids
            ),
            "attention_flops_per_example": (
                self.attention.attention_flops_per_example()
            ),
        }


@torch.inference_mode()
def _measure(model: nn.Module, geometry, regime: str, batch: int, device) -> dict:
    categorical, operands, validity = beff._make_inputs(
        geometry, regime, batch, device
    )
    for _ in range(WARMUPS):
        model(categorical, operands, validity)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    wall: list[float] = []
    dev: list[float] = []
    import time

    for _ in range(ROUNDS):
        for _ in range(FORWARDS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start_event.record()
            started = time.perf_counter()
            model(categorical, operands, validity)
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
        "device_latency_ms_p50": d50,
        "wall_over_device_p50": p50 / d50 if d50 > 0 else None,
        "throughput_examples_per_second": batch / (p50 / 1e3),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "efficiency" / "rq2_hiformer_v1",
    )
    parser.add_argument("--allow-busy-host", action="store_true")
    args = parser.parse_args(argv)

    if not TRITON_AVAILABLE:
        raise SystemExit("Triton is required for the fused DCDN arms")
    device = torch.device(args.device)
    state = beff.host_state(device)
    print(
        f"device: {device}  batches={SWEEP_BATCHES}  "
        f"warmups={WARMUPS} forwards={FORWARDS} rounds={ROUNDS}"
    )
    print(
        f"host load 1m/5m/15m: {state['load_average_1m']:.2f}/"
        f"{state['load_average_5m']:.2f}/{state['load_average_15m']:.2f}  "
        f"idle={state['host_idle']}"
    )
    if not state["host_idle"] and not args.allow_busy_host:
        raise SystemExit("host is not idle; pass --allow-busy-host to proceed")

    beff.prepare_output_dir(args.output_dir)
    geometry = next(g for g in beff.GEOMETRIES if g.geometry_id == "ranker")

    counts: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        counts[arm.arm_id] = HiFormerStack(geometry, arm).counts()

    manifest = {
        "artifact_role": "rq2_hiformer_stack_pre_measurement_manifest",
        "schema_version": 1,
        "purpose": (
            "test whether a downstream component with O(F^2) cost turns the "
            "reduced feature view into a serving saving, which a cross-only "
            "stack cannot show"
        ),
        "layer": "hiformer_style_heterogeneous_attention",
        "faithful_hiformer_reproduction": False,
        "part_of_frozen_model_suite": False,
        "trains_any_model": False,
        "produces_quality_numbers": False,
        "reads_dataset_rows": False,
        "reads_test_metrics": False,
        "geometry": geometry.geometry_id,
        "batches": list(SWEEP_BATCHES),
        "num_heads": NUM_HEADS,
        "ffn_ratio": FFN_RATIO,
        "protocol": {
            "warmups": WARMUPS,
            "synchronized_forwards": FORWARDS,
            "independent_rounds": ROUNDS,
        },
        "counts": counts,
        "host_state_at_start": state,
    }
    manifest_digest = beff.publish_json(
        args.output_dir / "pre_measurement_manifest_v1.json", manifest
    )
    print(f"pre-measurement manifest sha256: {manifest_digest}\n")
    for arm_id, entry in counts.items():
        print(
            f"  {arm_id:<16} tokens={entry['tokens']:<4} "
            f"attention_flops/ex={entry['attention_flops_per_example']:>9,} "
            f"interaction={entry['interaction']:>7,} "
            f"non_embedding={entry['non_embedding']:>9,}"
        )
    print()

    measurements: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in ARMS:
        model = HiFormerStack(geometry, arm).to(device).eval()
        if arm.fused:
            model = model.fuse()
        rows: dict[int, dict[str, Any]] = {}
        for batch in SWEEP_BATCHES:
            rows[batch] = _measure(model, geometry, arm.regime, batch, device)
            row = rows[batch]
            print(
                f"  {arm.arm_id:<16} B={batch:<6} "
                f"p50={row['batch_latency_ms_p50']:8.4f} "
                f"tput={row['throughput_examples_per_second']:12,.0f}/s"
            )
        measurements[arm.arm_id] = rows
        del model
        torch.cuda.empty_cache()
        print()

    comparisons = []
    for batch in SWEEP_BATCHES:
        subject = measurements["dcdn_reduced"][batch]
        for comparator_id in ("dcnv2_reduced", "dcnv2_full"):
            comparator = measurements[comparator_id][batch]
            comparisons.append(
                {
                    "batch": batch,
                    "subject": "dcdn_reduced",
                    "comparator": comparator_id,
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

    print("=== DCDN-REDUCED against DCNv2, with the attention layer ===")
    for entry in comparisons:
        print(
            f"  B={entry['batch']:<6} vs {entry['comparator']:<15} "
            f"latency x{entry['latency_ratio_p50']:.3f}  "
            f"throughput x{entry['throughput_ratio']:.3f}"
        )

    contamination = beff.contamination_check(
        {"ranker": {a: {b: {} for b in r} for a, r in measurements.items()}}
    )
    results = {
        "artifact_role": "rq2_hiformer_stack_result",
        "schema_version": 1,
        "pre_measurement_manifest_sha256": manifest_digest,
        "geometry": geometry.geometry_id,
        "batches": list(SWEEP_BATCHES),
        "counts": counts,
        "host_state_at_start": state,
        "host_state_at_end": beff.host_state(device),
        "contamination_check": contamination,
        "measurements": {
            arm: {str(b): row for b, row in rows.items()}
            for arm, rows in measurements.items()
        },
        "comparisons": comparisons,
        "scope": (
            "cost only; the attention layer is outside the frozen model suite "
            "and no quality number is produced or implied"
        ),
    }
    digest = beff.publish_json(args.output_dir / "results_v1.json", results)
    csv_rows = [
        {"arm_id": arm, **row}
        for arm, rows in measurements.items()
        for row in rows.values()
    ]
    csv_digest = beff.publish_csv(args.output_dir / "hiformer_stack.csv", csv_rows)
    print(f"\npublished results_v1.json    sha256={digest}")
    print(f"published hiformer_stack.csv sha256={csv_digest} ({len(csv_rows)} rows)")
    print(f"output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
