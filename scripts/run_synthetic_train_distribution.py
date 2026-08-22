#!/usr/bin/env python
"""How wide must the training distribution be before OOD loss becomes small?

Two hypotheses about the large out-of-distribution losses in `docs/rq1_verdict.md`
have already been ruled out by measurement:

* **The reciprocal floor is not binding.** `div_eps` incidence is exactly 0.0 on
  all four slices for both DCDN arms, and there are zero non-finite predictions.
  Nothing is being clipped.
* **Task interference is not the cause.** Training each target alone made DCDN's
  OOD *worse*, not better (`results/synthetic_multiquotient/single_target_v1`).

What the diagnostic did show is that the model fits a surface that agrees with
``n/d`` on the training box and produces arbitrary values outside it — including
negatives for a strictly positive target. If that is the mechanism, then the
lever is the training distribution itself, not the architecture.

This sweeps four training distributions against the same frozen evaluation
slices:

``uniform_1_10``      the frozen baseline: operands ~ U[1, 10]
``loguniform_1_10``   same support, log-uniform shape - does shape alone help?
``loguniform_1_100``  wider by one decade
``loguniform_01_1000``  four decades

The practically motivated direction is denominator growth: historical counters
like impression counts rise over time, which is exactly what ``ood_low_ratio``
(denominators in [20, 60]) represents.

Coverage is reported, not hidden
--------------------------------
A wider training range makes some evaluation slices in-distribution, and a slice
that is covered is no longer an extrapolation test. Each row of the output
records, per slice, whether that slice's operand range lies inside the training
range. Small losses on a covered slice mean "we trained on it", and the table
says so rather than letting the number speak for itself.

Usage::

    python scripts/run_synthetic_train_distribution.py --device cuda:1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

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


rsm = _load(
    "run_synthetic_multiquotient",
    REPO_ROOT / "scripts" / "run_synthetic_multiquotient.py",
)
sta = _load(
    "run_synthetic_single_target",
    REPO_ROOT / "scripts" / "run_synthetic_single_target.py",
)

from dcdn.multitask import (  # noqa: E402
    MultiTargetSpec,
    TargetSpec,
    compute_task_constants,
    macro_nloss,
)
from dcdn.synthetic_multiquotient import COLUMNS, build_benchmark  # noqa: E402

EVALUATION_SLICES = ("id", "ood_common_scale", "ood_low_ratio", "ood_high_ratio")
#: Operand ranges of the frozen evaluation slices, used only to report coverage.
SLICE_OPERAND_RANGE = {
    "id": (1.0, 10.0),
    "ood_common_scale": (10.0, 100.0),
    "ood_low_ratio": (1.0, 60.0),
    "ood_high_ratio": (0.2, 10.0),
}

ARMS = ("dcdn_reduced", "afn_reduced", "dcnv2_capacity_reduced", "dcnv2_reduced")
TARGET = "y1"
SEEDS = (42, 43, 44)
LEARNING_RATE = 1e-3
TRAIN_ROWS = 200_000

DISTRIBUTIONS = (
    ("uniform_1_10", "uniform", 1.0, 10.0),
    ("loguniform_1_10", "loguniform", 1.0, 10.0),
    ("loguniform_1_100", "loguniform", 1.0, 100.0),
    ("loguniform_01_1000", "loguniform", 0.1, 1000.0),
)


def make_training_set(
    kind: str, low: float, high: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Operands and the y1 target, drawn from one candidate distribution.

    Every column is drawn independently from the same distribution, matching how
    the frozen benchmark draws its operands. The target stays the exact quotient
    ``n1 / d1``, so the supervision is unchanged; only the input distribution
    moves.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "uniform":
        values = (
            torch.rand(TRAIN_ROWS, len(COLUMNS), generator=generator)
            * (high - low) + low
        )
    elif kind == "loguniform":
        exponent = torch.rand(TRAIN_ROWS, len(COLUMNS), generator=generator)
        values = torch.exp(
            torch.log(torch.tensor(low))
            + exponent * (torch.log(torch.tensor(high)) - torch.log(torch.tensor(low)))
        )
    else:
        raise ValueError(kind)
    n1 = values[:, COLUMNS.index("n1")]
    d1 = values[:, COLUMNS.index("d1")]
    return values, (n1 / d1).unsqueeze(1)


def covers(low: float, high: float, slice_name: str) -> bool:
    slice_low, slice_high = SLICE_OPERAND_RANGE[slice_name]
    return low <= slice_low and slice_high <= high


def train(model, spec, constants, features, targets, arm, device, seed) -> None:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=rsm.WEIGHT_DECAY,
    )
    features = features.to(device)
    targets = targets.to(device)
    masks = torch.ones_like(targets, dtype=torch.bool)
    validity = torch.ones_like(features, dtype=torch.bool)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(rsm.MAX_STEPS):
        model.train()
        index = torch.randint(
            0, features.shape[0], (rsm.BATCH_SIZE,), generator=generator
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features[index], validity[index])
        if not bool(torch.isfinite(logits).all()):
            continue
        loss = macro_nloss(
            spec, constants, logits, targets[index], masks[index]
        ).aggregate
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), rsm.GRAD_CLIP)
        optimizer.step()


@torch.no_grad()
def evaluate(model, spec, constants, item, arm, device) -> float:
    model.eval()
    features, targets, masks, validity = rsm._slice_view(item, arm, device)
    logits = torch.nan_to_num(model(features, validity))
    return float(macro_nloss(spec, constants, logits, targets, masks).aggregate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "synthetic_multiquotient"
        / "train_distribution_v1",
    )
    args = parser.parse_args(argv)
    if not args.output_dir.is_absolute():
        args.output_dir = REPO_ROOT / args.output_dir
    device = torch.device(args.device)
    rsm._prepare_output_dir(args.output_dir)

    benchmark = build_benchmark()
    spec = MultiTargetSpec(
        dataset_id="synthetic_multiquotient_3",
        targets=(TargetSpec(name=TARGET, kind="regression"),),
        primary_confirmatory_target=TARGET,
    )
    rsm.SPEC = spec
    narrowed = {
        name: sta._NarrowedSlice(benchmark[name], 0) for name in EVALUATION_SLICES
    }

    common = rsm.common_counts(
        flat_width=len(COLUMNS),
        representation_width=rsm.REPRESENTATION_DIM,
        head_hidden_dims=rsm.HEAD_HIDDEN_DIMS,
        num_targets=1,
    )
    resolution = rsm.resolve_capacity_control(
        target=rsm.dcdn_counts(len(COLUMNS), 1, common),
        operand_width=len(COLUMNS), common=common,
        ranks=tuple(range(1, 17)), layer_options=(1, 2), expert_options=(1, 2, 4),
    )
    sel = resolution.payload()["selected"]
    arms = {a.arm_id: a for a in rsm._make_arms(len(COLUMNS))}
    arms["dcnv2_capacity_reduced"] = rsm.Arm(
        "dcnv2_capacity_reduced", "DCNv2-capacity-matched",
        lambda encoder, seed: rsm.MultiQuotientModel(
            encoder,
            rsm.DCNv2CapacityCoreConfig(
                num_layers=sel["num_layers"], low_rank=sel["low_rank"],
                num_experts=sel["num_experts"], product_init_scale=0.1,
            ),
            num_targets=1, head_hidden_dims=rsm.HEAD_HIDDEN_DIMS,
            init_seed=seed, with_afn_view=False,
        ),
    )

    records: list[dict[str, Any]] = []
    for name, kind, low, high in DISTRIBUTIONS:
        coverage = {s: covers(low, high, s) for s in EVALUATION_SLICES}
        print(f"\n=== training distribution {name} — operands in [{low}, {high}]")
        print("    slices this range covers: "
              + ", ".join(s for s, c in coverage.items() if c) + "\n")
        features, targets = make_training_set(kind, low, high, 20260901)
        # The task constant is refit per distribution, because the training
        # target variance changes with the operand distribution and the metric
        # must stay "relative to this training set's constant predictor".
        constants = compute_task_constants(
            spec, targets, torch.ones_like(targets, dtype=torch.bool)
        )
        print(f"    train target variance = {constants.values[0]:.6g}")
        for arm_id in ARMS:
            arm = arms[arm_id]
            for seed in SEEDS:
                encoder = rsm.encoder_spec(rsm.EMBEDDING_DIM, rsm.REPRESENTATION_DIM)
                model = arm.build(encoder, seed)
                train(model, spec, constants, features, targets, arm, device, seed)
                slices = {
                    s: evaluate(model, spec, constants, narrowed[s], arm, device)
                    for s in EVALUATION_SLICES
                }
                records.append({
                    "distribution": name, "operand_low": low, "operand_high": high,
                    "arm_id": arm_id, "seed": seed,
                    "slice_covered_by_training_range": coverage,
                    "train_target_variance": float(constants.values[0]),
                    "slices": slices,
                })
                print(f"  {arm_id:<24} seed={seed} " + "  ".join(
                    f"{s.replace('ood_', '')}={v:.4g}"
                    + ("*" if coverage[s] else "")
                    for s, v in slices.items()
                ))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    digest = rsm._publish_bytes(
        args.output_dir / "results_v1.json",
        json.dumps({
            "artifact_role": "synthetic_multiquotient_training_distribution_sweep",
            "schema_version": 1,
            "study": "how_wide_must_training_be_for_small_ood_loss",
            "is_the_frozen_rq1_study": False,
            "benchmark_manifest_sha256": benchmark.manifest_sha256,
            "target": TARGET,
            "seeds": list(SEEDS),
            "distributions": [
                {"name": n, "kind": k, "low": lo, "high": hi}
                for n, k, lo, hi in DISTRIBUTIONS
            ],
            "slice_operand_ranges": SLICE_OPERAND_RANGE,
            "ruled_out_before_this_sweep": [
                "reciprocal_floor_div_eps_incidence_is_exactly_zero_on_all_slices",
                "single_target_training_made_dcdn_ood_worse_not_better",
            ],
            "records": records,
        }, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n",
    )
    print(f"\n* = slice is inside the training operand range, so it is not an "
          f"extrapolation test\npublished sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
