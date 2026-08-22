#!/usr/bin/env python
"""Does forcing scale invariance during training make the quotient generalize?

The diagnostic in `docs/rq1_verdict.md` showed that on ``ood_common_scale`` —
where every operand is multiplied by 10 and the targets are bitwise unchanged —
a trained DCDN predicts between -139 and +417 for a target that lives in
[0.10, 9.76]. It does not compute ``n/d``; it fits a surface that agrees with
``n/d`` on the training box and produces arbitrary values outside it.

The architecture *can* represent the quotient exactly: one reciprocal-branch
coordinate is ``inv(x0)_d * W_d(x0)_d = n/d``. Nothing in an MSE objective on a
bounded box forces it to use that route, because infinitely many functions agree
there.

This asks whether making invariance **mandatory** changes that. Each training
row is multiplied by a random common factor before the forward pass, with its
target left untouched — which is exactly the identity ``(s*n)/(s*d) = n/d``. A
model that fits a generic surface can no longer do so; it must become invariant.

The training scale range deliberately does NOT cover the evaluation scale
-------------------------------------------------------------------------
``ood_common_scale`` uses a factor of exactly 10. Training with factors up to 10
would make that slice in-distribution and the test meaningless. The augmentation
draws from ``LogUniform(1/3, 3)`` instead, so the question stays honest: does an
invariance learned over a narrow band of scales *extrapolate* to a band it never
saw?

Prediction this makes, stated before the run
--------------------------------------------
Scale augmentation should help ``ood_common_scale`` substantially and should do
little for ``ood_low_ratio`` and ``ood_high_ratio``. A common rescale leaves the
quotient fixed, so the augmentation expands the range of operand magnitudes but
not the range of quotient VALUES; the low- and high-ratio slices move the
quotient itself outside what was trained. If those two also improve, the
explanation above is wrong.

Usage::

    python scripts/run_synthetic_scale_augment.py --device cuda:1
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
LEARNING_ARMS = (
    "linear", "dcnv2_reduced", "dcdn_reduced", "afn_reduced",
    "dcnv2_capacity_reduced",
)
#: Deliberately narrower than the evaluation factor of 10. See the module
#: docstring: covering it would make the slice in-distribution.
SCALE_LOW, SCALE_HIGH = 1.0 / 3.0, 3.0
TARGET = "y1"
SEEDS = (42, 43, 44)
LEARNING_RATE = 1e-3


def train(
    model: nn.Module, spec, constants, narrowed, arm, device, seed: int,
    *, augment: bool,
) -> None:
    """The frozen training loop, with one optional line added.

    Everything is copied from `run_synthetic_multiquotient._train` so the two
    conditions differ in exactly one thing: whether each batch's operands are
    multiplied by a random common factor.
    """

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=rsm.WEIGHT_DECAY,
    )
    train_x, train_y, train_m, train_v = rsm._slice_view(
        narrowed["train"], arm, device
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for _ in range(1, rsm.MAX_STEPS + 1):
        model.train()
        index = torch.randint(
            0, train_x.shape[0], (rsm.BATCH_SIZE,), generator=generator
        ).to(device)
        features = train_x[index]
        if augment:
            # One common factor per row, log-uniform. The target is NOT
            # touched: (s*n)/(s*d) = n/d is the whole point.
            exponent = torch.rand(
                features.shape[0], 1, generator=generator
            ).to(device)
            low, high = torch.log(torch.tensor(SCALE_LOW)), torch.log(
                torch.tensor(SCALE_HIGH)
            )
            factor = torch.exp(low.to(device) + exponent * (high - low).to(device))
            features = features * factor
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, train_v[index])
        if not bool(torch.isfinite(logits).all()):
            continue
        loss = macro_nloss(
            spec, constants, logits, train_y[index], train_m[index]
        ).aggregate
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), rsm.GRAD_CLIP)
        optimizer.step()


@torch.no_grad()
def evaluate(model, spec, constants, item, arm, device) -> float:
    model.eval()
    features, targets, masks, validity = rsm._slice_view(item, arm, device)
    logits = torch.nan_to_num(model(features, validity))
    return float(
        macro_nloss(spec, constants, logits, targets, masks).aggregate
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "synthetic_multiquotient" / "scale_augment_v1",
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
        name: sta._NarrowedSlice(benchmark[name], 0)
        for name in ("train", "validation", *EVALUATION_SLICES)
    }
    constants = compute_task_constants(
        spec, narrowed["train"].targets, narrowed["train"].masks
    )
    print(f"target={TARGET}  constant={constants.values[0]:.6g}")
    print(f"augmentation scale range: [{SCALE_LOW:.4f}, {SCALE_HIGH:.4f}]  "
          f"evaluation uses 10.0, deliberately outside it\n")

    # The capacity control is resolved mechanically and assembled outside
    # `_make_arms`, exactly as the frozen runner does it.
    common = rsm.common_counts(
        flat_width=len(COLUMNS),
        representation_width=rsm.REPRESENTATION_DIM,
        head_hidden_dims=rsm.HEAD_HIDDEN_DIMS,
        num_targets=1,
    )
    resolution = rsm.resolve_capacity_control(
        target=rsm.dcdn_counts(len(COLUMNS), 1, common),
        operand_width=len(COLUMNS),
        common=common,
        ranks=tuple(range(1, 17)),
        layer_options=(1, 2),
        expert_options=(1, 2, 4),
    )
    payload = resolution.payload()
    if not resolution.matched:
        raise SystemExit(f"no capacity control inside tolerance: {payload}")
    sel = payload["selected"]
    print(f"capacity control: rank={sel['low_rank']} layers={sel['num_layers']} "
          f"experts={sel['num_experts']}\n")
    arms = {a.arm_id: a for a in rsm._make_arms(len(COLUMNS))}
    arms["dcnv2_capacity_reduced"] = rsm.Arm(
        "dcnv2_capacity_reduced",
        "DCNv2-capacity-matched",
        lambda encoder, seed: rsm.MultiQuotientModel(
            encoder,
            rsm.DCNv2CapacityCoreConfig(
                num_layers=sel["num_layers"],
                low_rank=sel["low_rank"],
                num_experts=sel["num_experts"],
                product_init_scale=0.1,
            ),
            num_targets=1,
            head_hidden_dims=rsm.HEAD_HIDDEN_DIMS,
            init_seed=seed,
            with_afn_view=False,
        ),
    )
    records: list[dict[str, Any]] = []
    for arm_id in LEARNING_ARMS:
        arm = arms[arm_id]
        for augment in (False, True):
            for seed in SEEDS:
                encoder = rsm.encoder_spec(rsm.EMBEDDING_DIM, rsm.REPRESENTATION_DIM)
                model = arm.build(encoder, seed)
                train(model, spec, constants, narrowed, arm, device, seed,
                      augment=augment)
                slices = {
                    name: evaluate(model, spec, constants, narrowed[name], arm, device)
                    for name in EVALUATION_SLICES
                }
                records.append({
                    "arm_id": arm_id, "seed": seed, "scale_augmented": augment,
                    "slices": slices,
                })
                tag = "augmented" if augment else "baseline "
                print(
                    f"  {arm_id:<24} {tag} seed={seed} " + "  ".join(
                        f"{n.replace('ood_', '')}={v:.4g}" for n, v in slices.items()
                    )
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    digest = rsm._publish_bytes(
        args.output_dir / "results_v1.json",
        json.dumps({
            "artifact_role": "synthetic_multiquotient_scale_augmentation",
            "schema_version": 1,
            "study": "does_forced_scale_invariance_make_the_quotient_generalize",
            "is_the_frozen_rq1_study": False,
            "benchmark_manifest_sha256": benchmark.manifest_sha256,
            "target": TARGET,
            "seeds": list(SEEDS),
            "learning_rate": LEARNING_RATE,
            "augmentation": {
                "kind": "per_row_common_log_uniform_rescale_of_all_operands",
                "low": SCALE_LOW,
                "high": SCALE_HIGH,
                "targets_unchanged": True,
                "evaluation_scale_factor": 10.0,
                "training_range_covers_evaluation_scale": False,
            },
            "records": records,
        }, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n",
    )
    print(f"\npublished sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
