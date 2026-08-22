#!/usr/bin/env python
"""RQ1 addendum: three independent single-target models instead of one 3-head model.

The frozen RQ1 study trains one model with three heads for ``y1 = n1/d1``,
``y2 = n2/d2`` and ``y3 = (n1+n2)/d3`` under Macro-NLoss. This runs the same
arms, slices, seeds and grid, but as **three separate models**, each carrying a
single head for one target.

Why
---
The RQ1 verdict is bounded by two facts that this design can separate. On the
in-distribution and high-ratio slices DCDN wins by 3.6x and 5.1x, but on
``ood_common_scale`` and ``ood_low_ratio`` its absolute loss is 5975 and 2990
against a constant predictor at 1.0 — it is best-in-class on two slices and
unusable on two others. Being top-1 relatively is not the same as being right
absolutely.

One candidate explanation is task interference: three heads share a trunk under
an unweighted mean, so the representation is a compromise none of the three
targets would choose alone. That is testable, and this is the test. The
prediction it makes is specific: if interference is the cause, the absolute OOD
loss should fall substantially when each target is trained alone. If it does
not, interference is not the explanation and the local-approximation reading in
`docs/rq1_verdict.md` §6.1 stands.

The same question was raised on KuaiRand as RQ4; this is the cheaper synthetic
version of it.

Everything except the head count is held identical to the frozen study: the same
materialized and pre-hashed slices, the same arms, the same capacity control,
the same tuning seed, the same final seeds, the same learning-rate grid, the
same optimizer and step budget. The three per-target models are trained from
scratch; no multi-target checkpoint is reused.

Standing: this is an ADDENDUM to RQ1, not a replacement. It cannot restate the
RQ1 verdict, which is about the three-head study.

Usage::

    python scripts/run_synthetic_single_target.py --device cuda:1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

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

from dcdn.multitask import (  # noqa: E402
    MultiTargetSpec,
    TargetSpec,
    compute_task_constants,
)
from dcdn.multitask_capacity import (  # noqa: E402
    common_counts,
    dcdn_counts,
    resolve_capacity_control,
)
from dcdn.synthetic_multiquotient import (  # noqa: E402
    COLUMNS,
    TARGET_NAMES,
    build_benchmark,
)

EVALUATION_SLICES = ("id", "ood_common_scale", "ood_low_ratio", "ood_high_ratio")

#: Captured before any narrowing so each target rebinds from the true oracle.
_ORACLE = rsm.analytical_oracle


class _NarrowedSlice:
    """One materialized slice with its targets restricted to a single column.

    The features are untouched, so the model sees exactly what the three-head
    study's model saw; only the supervision narrows. `sha256` is carried through
    unchanged because it hashes the generated slice, and the slice itself has
    not changed.
    """

    def __init__(self, item, column: int) -> None:
        self.features = item.features
        self.targets = item.targets[:, column : column + 1]
        self.masks = item.masks[:, column : column + 1]
        self.sha256 = item.sha256
        self.config = item.config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "synthetic_multiquotient" / "single_target_v1",
    )
    args = parser.parse_args(argv)
    if not args.output_dir.is_absolute():
        args.output_dir = REPO_ROOT / args.output_dir

    device = torch.device(args.device)
    rsm._prepare_output_dir(args.output_dir)

    benchmark = build_benchmark()
    print(f"benchmark manifest sha256 : {benchmark.manifest_sha256}")
    print(f"targets                   : {TARGET_NAMES}")

    records: list[dict[str, Any]] = []
    per_target_meta: dict[str, Any] = {}

    for column, target_name in enumerate(TARGET_NAMES):
        print(f"\n{'=' * 72}\n=== {target_name} — independent single-head model\n"
              f"{'=' * 72}")
        spec = MultiTargetSpec(
            dataset_id="synthetic_multiquotient_3",
            targets=(TargetSpec(name=target_name, kind="regression"),),
            primary_confirmatory_target=target_name,
        )
        # `_make_arms` and `_train` read the module-level SPEC of the frozen
        # runner, so pointing it at the one-target spec is what makes the head
        # narrow while leaving every other decision untouched.
        rsm.SPEC = spec
        # The analytical oracle returns all three closed-form quotients. With a
        # single-column label it must return only the one under test, or the
        # loss cannot even be shaped. Narrowing it here keeps the oracle an
        # exact reference rather than dropping it from the comparison.
        rsm.analytical_oracle = (
            lambda features, col=column, base=_ORACLE: base(features)[:, col:col + 1]
        )

        narrowed = {
            name: _NarrowedSlice(benchmark[name], column)
            for name in ("train", "validation", *EVALUATION_SLICES)
        }
        constants = compute_task_constants(
            spec, narrowed["train"].targets, narrowed["train"].masks
        )
        print(f"frozen task constant      : {target_name}={constants.values[0]:.6g}")

        reduced_common = common_counts(
            flat_width=len(COLUMNS),
            representation_width=rsm.REPRESENTATION_DIM,
            head_hidden_dims=rsm.HEAD_HIDDEN_DIMS,
            num_targets=1,
        )
        target_counts = dcdn_counts(len(COLUMNS), 1, reduced_common)
        resolution = resolve_capacity_control(
            target=target_counts,
            operand_width=len(COLUMNS),
            common=reduced_common,
            ranks=tuple(range(1, 17)),
            layer_options=(1, 2),
            expert_options=(1, 2, 4),
        )
        payload = resolution.payload()
        if not resolution.matched:
            raise SystemExit(f"no capacity control inside tolerance: {payload}")
        selected = payload["selected"]
        print(
            f"capacity control          : rank={selected['low_rank']} "
            f"layers={selected['num_layers']} experts={selected['num_experts']} "
            f"ratios={payload['selected_ratios']['parameter_ratio']:.4f}/"
            f"{payload['selected_ratios']['flop_ratio']:.4f}"
        )
        capacity_arm = rsm.Arm(
            "dcnv2_capacity_reduced",
            "DCNv2-capacity-matched",
            lambda encoder, seed, sel=selected: rsm.MultiQuotientModel(
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
        base_arms = rsm._make_arms(len(COLUMNS))
        arms = (*base_arms[:4], capacity_arm, *base_arms[4:])

        def build(arm, seed: int):
            """Arms take an ENCODER spec; full arms get the wider one.

            The analytical oracle has no parameters and no `build`; it is
            evaluated in closed form.
            """

            if arm.analytical:
                return None
            encoder = (
                rsm.encoder_spec_for_full() if arm.explicit_ratios
                else rsm.encoder_spec(rsm.EMBEDDING_DIM, rsm.REPRESENTATION_DIM)
            )
            return arm.build(encoder, seed)
        per_target_meta[target_name] = {
            "task_constant": float(constants.values[0]),
            "task_constant_sha256": constants.contract_sha256,
            "capacity_resolution": payload,
        }

        print("\n--- tuning (seed 42, validation-selected) ---")
        selected_lr: dict[str, float] = {}
        for arm in arms:
            if arm.analytical:
                selected_lr[arm.arm_id] = 0.0
                continue
            trials = []
            for learning_rate in rsm.LEARNING_RATES:
                model = build(arm, rsm.TUNING_SEED)
                started = time.perf_counter()
                outcome = rsm._train(
                    model, spec, constants, narrowed, arm, device,
                    learning_rate, rsm.TUNING_SEED,
                )
                trials.append({"learning_rate": learning_rate, **outcome})
                print(
                    f"  {arm.arm_id:<26} lr={learning_rate:<7g} "
                    f"val={outcome['validation_macro_nloss']:.6g} "
                    f"({time.perf_counter() - started:.0f}s)"
                )
            best = min(trials, key=lambda t: t["validation_macro_nloss"])
            selected_lr[arm.arm_id] = best["learning_rate"]

        print("\n--- final evaluation (seeds 42, 43, 44) ---")
        for arm in arms:
            for seed in rsm.FINAL_SEEDS:
                model = build(arm, seed)
                training = (
                    {} if arm.analytical else rsm._train(
                        model, spec, constants, narrowed, arm, device,
                        selected_lr[arm.arm_id], seed,
                    )
                )
                slices: dict[str, Any] = {}
                for name in EVALUATION_SLICES:
                    report, _ = rsm._evaluate(
                        model, spec, constants, narrowed[name], arm, device
                    )
                    slices[name] = report
                records.append({
                    "target": target_name,
                    "arm_id": arm.arm_id,
                    "label": arm.label,
                    "seed": seed,
                    "learning_rate": selected_lr[arm.arm_id],
                    "training": training,
                    "slices": slices,
                })
                print(
                    f"  {arm.arm_id:<26} seed={seed} "
                    f"id={slices['id']['macro_nloss']:.6g} "
                    f"high={slices['ood_high_ratio']['macro_nloss']:.6g} "
                    f"scale={slices['ood_common_scale']['macro_nloss']:.6g} "
                    f"low={slices['ood_low_ratio']['macro_nloss']:.6g}"
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    digest = rsm._publish_bytes(
        args.output_dir / "results_v1.json",
        json.dumps({
            "artifact_role": "synthetic_multiquotient_single_target_results",
            "schema_version": 1,
            "study": "rq1_addendum_three_independent_single_head_models",
            "is_the_frozen_rq1_study": False,
            "replaces_rq1": False,
            "benchmark_manifest_sha256": benchmark.manifest_sha256,
            "targets": list(TARGET_NAMES),
            "evaluation_slices": list(EVALUATION_SLICES),
            "final_seeds": list(rsm.FINAL_SEEDS),
            "tuning_seed": rsm.TUNING_SEED,
            "learning_rates": list(rsm.LEARNING_RATES),
            "per_target": per_target_meta,
            "records": records,
        }, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n",
    )
    print(f"\npublished results_v1.json sha256={digest}")
    print(f"output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
