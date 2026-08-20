#!/usr/bin/env python
"""Execute ``Synthetic-MultiQuotient-3`` with the arm set declared in §9.

This is the exact runner that produced
``results/synthetic_multiquotient/results_v1.json`` (benchmark manifest
``659d29f5…``, selected configs ``63555e1f…``).  It is preserved under an
explicit ``_v1`` name because a concurrent session has since rewritten
``scripts/run_synthetic_multiquotient.py`` in place to drive the seven-row
recommendation matrix ``M1..M7`` instead.  Both readings are defensible; they
answer different questions, and the results committed alongside this file were
produced by this one.

Arm set, exactly as the owner override lists it for the quotient benchmark:

``linear``            a linear probe on the raw operands
``mlp``               the frozen DNN geometry
``dcnv2_core``        full-rank DCNv2
``dcnv2_capacity``    the mechanically matched low-rank mixture control
``dcdn_core_l1``      DCDN-Core-L1
``afn_plus``          AFN+
``dcnv2_ratio_pc``    DCNv2 with the explicit target ratios appended
                      (positive control: what the quotient is worth when it is
                      handed to the model)
``oracle``            the analytical identity, computed in closed form

Protocol, in order, with no step allowed to run before the previous one:

1. Materialize and hash every slice.  Nothing is trained before the hashes
   exist.
2. Resolve the capacity control mechanically in (parameter, FLOP) space.  No
   validation metric is read during resolution.
3. Tune on **seed 42, validation only**: two learning rates per arm, selected
   by validation Macro-NLoss.  The evaluation slices are not touched.
4. Freeze and hash the selected configurations.
5. Evaluate on seeds 42, 43, 44 across all four evaluation slices.

Reported per arm: Macro-NLoss and every quotient task separately on every
slice, scale consistency, non-finite counts, reciprocal-floor incidence,
parameters, and FLOPs/example.

Usage::

    python scripts/run_synthetic_multiquotient_v1.py --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
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
from dcdn.multitask import (  # noqa: E402
    MultiTargetSpec,
    TaskConstants,
    compute_task_constants,
    macro_nloss,
)
from dcdn.multitask_capacity import (  # noqa: E402
    common_counts,
    dcdn_counts,
    ordinary_dcnv2_counts,
    resolve_capacity_control,
)
from dcdn.synthetic_multiquotient import (  # noqa: E402
    COLUMNS,
    DENOMINATOR_COLUMNS,
    SPEC,
    TARGET_NAMES,
    analytical_oracle,
    build_benchmark,
    explicit_ratio_features,
)
from dcdn.synthetic_runtime import (  # noqa: E402
    MultiQuotientModel,
    encoder_spec,
    ordered_schema_sha256,
)

EMBEDDING_DIM = 8
REPRESENTATION_DIM = 64
HEAD_HIDDEN_DIMS = (128, 64)
DIV_EPS = 1e-2

TUNING_SEED = 42
FINAL_SEEDS = (42, 43, 44)
LEARNING_RATES = (3e-4, 1e-3)

MAX_STEPS = 4000
BATCH_SIZE = 4096
VALIDATE_EVERY = 200
PATIENCE = 5
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 1.0


@dataclass(frozen=True)
class Arm:
    arm_id: str
    label: str
    build: Any  # callable(spec, seed) -> nn.Module, or None
    explicit_ratios: bool = False
    analytical: bool = False
    with_afn_view: bool = False


class LinearProbe(nn.Module):
    """The floor of the comparison: one affine map from operands to targets."""

    model_key = "linear_probe"

    def __init__(self, input_width: int, num_targets: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_width, num_targets)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        operands: torch.Tensor,
        denominator_validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del denominator_validity
        return self.projection(operands)

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "encoder_shared": 0,
            "interaction": 0,
            "adapter": 0,
            "head": total,
            "matched": total,
            "total": total,
        }


def _seeded(seed: int, factory):
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.default_generator.manual_seed(seed)
        return factory()


def _make_arms() -> tuple[Arm, ...]:
    """Every arm except the mechanically resolved capacity control."""

    def controlled(config, *, afn: bool = False):
        def build(spec, seed: int) -> nn.Module:
            return MultiQuotientModel(
                spec,
                config,
                num_targets=len(SPEC),
                head_hidden_dims=HEAD_HIDDEN_DIMS,
                init_seed=seed,
                with_afn_view=afn,
            )

        return build

    return (
        Arm(
            "linear",
            "Linear probe",
            lambda spec, seed: _seeded(
                seed, lambda: LinearProbe(spec.operand_dim, len(SPEC))
            ),
        ),
        Arm(
            "mlp",
            "MLP baseline",
            controlled(MLPConfig(hidden_dims=(512, 256, 128), dropout=0.0)),
        ),
        Arm(
            "dcnv2_core",
            "DCNv2-Core",
            controlled(DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1)),
        ),
        Arm(
            "dcdn_core_l1",
            "DCDN-Core-L1",
            controlled(
                DCDNCoreConfig(
                    num_layers=1,
                    div_eps=DIV_EPS,
                    product_init_scale=0.1,
                    div_init_scale=0.1,
                    runtime_value_checks=False,
                )
            ),
        ),
        Arm(
            "afn_plus",
            "AFN+",
            controlled(
                AFNPlusConfig(
                    logarithmic_neurons=128,
                    afn_hidden_dims=(512,),
                    dnn_hidden_dims=(512,),
                    log_floor=1e-6,
                    exp_clip=None,
                    dropout=0.0,
                    batch_norm=True,
                ),
                afn=True,
            ),
            with_afn_view=True,
        ),
        Arm(
            "dcnv2_ratio_pc",
            "DCNv2 + explicit ratios (positive control)",
            controlled(DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1)),
            explicit_ratios=True,
        ),
        Arm("oracle", "Analytical identity oracle", None, analytical=True),
    )


def encoder_spec_for_ratio_control():
    """Encoder shape for the positive control, whose input carries 3 extra columns."""

    from dcdn.models.contracts import EncoderSpec
    from dcdn.synthetic_multiquotient import DENOMINATOR_MASK

    width = len(COLUMNS) + 3
    schema = ordered_schema_sha256((*COLUMNS, "r1", "r2", "r3"))
    return EncoderSpec(
        bypass_dim=0,
        operand_dim=width,
        num_fields=width,
        embedding_dim=EMBEDDING_DIM,
        representation_dim=REPRESENTATION_DIM,
        # The appended ratios are inputs, never denominators.
        denominator_mask=(*DENOMINATOR_MASK, False, False, False),
        full_feature_schema_sha256=schema,
        operand_schema_sha256=schema,
    )


def _slice_view(item, arm: Arm, device: torch.device):
    features = item.features
    if arm.explicit_ratios:
        features = explicit_ratio_features(features)
    features = features.to(device)
    validity = torch.ones_like(features, dtype=torch.bool)
    if arm.explicit_ratios:
        # The appended ratios are ordinary inputs.  They must never be promoted
        # into the reciprocal mask, which is defined on d1, d2, d3 only.
        validity[:, len(COLUMNS) :] = False
    return features, item.targets.to(device), item.masks.to(device), validity


def _floor_incidence(features: torch.Tensor) -> dict[str, float]:
    """Fraction of active denominator entries at or below the reciprocal floor."""

    report: dict[str, float] = {}
    for column in DENOMINATOR_COLUMNS:
        values = features[:, COLUMNS.index(column)]
        report[column] = float((values <= DIV_EPS).float().mean())
    return report


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    spec: MultiTargetSpec,
    constants: TaskConstants,
    item,
    arm: Arm,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    features, targets, masks, validity = _slice_view(item, arm, device)
    if arm.analytical:
        logits = analytical_oracle(item.features.to(device))
    else:
        model.eval()
        logits = model(features, validity)
    nonfinite = int((~torch.isfinite(logits)).sum())
    safe = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
    report = macro_nloss(spec, constants, safe, targets, masks).as_report(spec)
    report["nonfinite_predictions"] = nonfinite
    report["reciprocal_floor_incidence"] = _floor_incidence(item.features)
    return report, safe


def _train(
    model: nn.Module,
    spec: MultiTargetSpec,
    constants: TaskConstants,
    benchmark,
    arm: Arm,
    device: torch.device,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )
    train_x, train_y, train_m, train_v = _slice_view(benchmark["train"], arm, device)
    valid_x, valid_y, valid_m, valid_v = _slice_view(
        benchmark["validation"], arm, device
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    best: dict[str, Any] = {"macro": float("inf"), "state": None, "step": -1}
    stale = 0
    nonfinite_batches = 0
    started = time.perf_counter()

    for step in range(1, MAX_STEPS + 1):
        model.train()
        index = torch.randint(
            0, train_x.shape[0], (BATCH_SIZE,), generator=generator
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x[index], train_v[index])
        if not bool(torch.isfinite(logits).all()):
            nonfinite_batches += 1
            if nonfinite_batches > 1:
                raise FloatingPointError(
                    f"{arm.arm_id}: repeated non-finite training activations"
                )
            continue
        loss = macro_nloss(
            spec, constants, logits, train_y[index], train_m[index]
        ).aggregate
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                value = float(
                    macro_nloss(
                        spec,
                        constants,
                        torch.nan_to_num(model(valid_x, valid_v)),
                        valid_y,
                        valid_m,
                    ).aggregate
                )
            if value < best["macro"]:
                best = {
                    "macro": value,
                    "state": {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    },
                    "step": step,
                }
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

    if best["state"] is None:
        raise RuntimeError(f"{arm.arm_id}: no validation checkpoint was recorded")
    model.load_state_dict(best["state"])
    return {
        "validation_macro_nloss": best["macro"],
        "best_step": best["step"],
        "nonfinite_training_batches": nonfinite_batches,
        "seconds": time.perf_counter() - started,
    }


def _scale_consistency(
    identity: torch.Tensor, scaled: torch.Tensor
) -> dict[str, float]:
    """How far predictions move when every operand is multiplied by a constant.

    The two slices are row aligned and carry identical targets, so a model that
    truly learned the quotient produces identical predictions and scores zero
    here.  Reported as a median relative deviation, which is robust to the
    heavy tail a diverging arm produces.
    """

    denominator = identity.abs().clamp_min(1e-6)
    relative = (scaled - identity).abs() / denominator
    return {
        "median_relative_deviation": float(relative.median()),
        "p95_relative_deviation": float(relative.flatten().quantile(0.95)),
        "mean_absolute_deviation": float((scaled - identity).abs().mean()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "synthetic_multiquotient",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -- Step 1: materialize and hash before any training --------------------
    benchmark = build_benchmark()
    manifest = benchmark.manifest()
    print(f"benchmark manifest sha256 : {benchmark.manifest_sha256}")
    for name, entry in sorted(manifest["slices"].items()):
        print(
            f"  {name:<18} rows={entry['rows']:>7}  sha256={entry['sha256'][:16]}..."
        )

    train_slice = benchmark["train"]
    constants = compute_task_constants(SPEC, train_slice.targets, train_slice.masks)
    print(
        "frozen task constants     : "
        f"{dict(zip(SPEC.names, constants.values, strict=True))}"
    )
    print(f"task-constant sha256      : {constants.contract_sha256}")

    # -- Step 2: mechanical capacity resolution ------------------------------
    common = common_counts(
        flat_width=len(COLUMNS),
        representation_width=REPRESENTATION_DIM,
        head_hidden_dims=HEAD_HIDDEN_DIMS,
        num_targets=len(SPEC),
    )
    target_counts = dcdn_counts(len(COLUMNS), 1, common)
    resolution = resolve_capacity_control(
        target=target_counts,
        operand_width=len(COLUMNS),
        common=common,
        ranks=tuple(range(1, 17)),
        layer_options=(1, 2),
        expert_options=(1, 2, 4),
    )
    payload = resolution.payload()
    if not resolution.matched:
        raise SystemExit(
            "no capacity control inside tolerance; bracket controls required: "
            f"{json.dumps(payload, indent=2)}"
        )
    selected = payload["selected"]
    print(
        f"capacity control          : rank={selected['low_rank']} "
        f"layers={selected['num_layers']} experts={selected['num_experts']} "
        f"ratios={payload['selected_ratios']['parameter_ratio']:.4f}/"
        f"{payload['selected_ratios']['flop_ratio']:.4f}"
    )

    capacity_arm = Arm(
        "dcnv2_capacity",
        "DCNv2-capacity-matched",
        lambda spec, seed: MultiQuotientModel(
            spec,
            DCNv2CapacityCoreConfig(
                num_layers=selected["num_layers"],
                low_rank=selected["low_rank"],
                num_experts=selected["num_experts"],
                product_init_scale=0.1,
            ),
            num_targets=len(SPEC),
            head_hidden_dims=HEAD_HIDDEN_DIMS,
            init_seed=seed,
            with_afn_view=False,
        ),
    )
    base_arms = _make_arms()
    arms = (*base_arms[:4], capacity_arm, *base_arms[4:])

    analytic_flops = {
        "dcdn_core_l1": target_counts,
        "dcnv2_core": ordinary_dcnv2_counts(len(COLUMNS), 2, common),
        "dcnv2_capacity": selected,
    }

    # -- Step 3: validation-only tuning at seed 42 ---------------------------
    print("\n--- tuning (seed 42, validation only) ---")
    selected_lr: dict[str, float] = {}
    tuning_log: list[dict[str, Any]] = []
    for arm in arms:
        if arm.analytical:
            continue
        trials = []
        for learning_rate in LEARNING_RATES:
            spec = (
                encoder_spec_for_ratio_control()
                if arm.explicit_ratios
                else encoder_spec(EMBEDDING_DIM, REPRESENTATION_DIM)
            )
            model = arm.build(spec, TUNING_SEED)
            outcome = _train(
                model,
                SPEC,
                constants,
                benchmark,
                arm,
                device,
                learning_rate,
                TUNING_SEED,
            )
            trials.append({"learning_rate": learning_rate, **outcome})
            print(
                f"  {arm.arm_id:<18} lr={learning_rate:<8} "
                f"val Macro-NLoss={outcome['validation_macro_nloss']:.6f} "
                f"({outcome['seconds']:.1f}s)"
            )
        chosen = min(trials, key=lambda t: t["validation_macro_nloss"])
        selected_lr[arm.arm_id] = chosen["learning_rate"]
        tuning_log.append(
            {
                "arm_id": arm.arm_id,
                "trials": trials,
                "selected_learning_rate": chosen["learning_rate"],
            }
        )

    # -- Step 4: freeze the selected configuration ---------------------------
    frozen = {
        "benchmark_manifest_sha256": benchmark.manifest_sha256,
        "objective_spec_sha256": SPEC.contract_sha256,
        "task_constants_sha256": constants.contract_sha256,
        "capacity_resolution": payload,
        "selected_learning_rates": selected_lr,
        "encoder": {
            "embedding_dim": EMBEDDING_DIM,
            "representation_dim": REPRESENTATION_DIM,
            "head_hidden_dims": list(HEAD_HIDDEN_DIMS),
            "operand_schema_sha256": ordered_schema_sha256(COLUMNS),
        },
        "optimization": {
            "family": "AdamW",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_l2": GRAD_CLIP,
            "batch_size": BATCH_SIZE,
            "maximum_steps": MAX_STEPS,
            "validate_every_steps": VALIDATE_EVERY,
            "patience": PATIENCE,
        },
        "final_seeds": list(FINAL_SEEDS),
    }
    frozen_text = json.dumps(frozen, indent=2, sort_keys=True)
    frozen_sha = hashlib.sha256(frozen_text.encode("utf-8")).hexdigest()
    (args.output_dir / "selected_configs_v1.json").write_text(
        frozen_text + "\n", encoding="utf-8"
    )
    (args.output_dir / "selected_configs_v1.json.sha256").write_text(
        f"{frozen_sha}  selected_configs_v1.json\n", encoding="utf-8"
    )
    print(f"\nfrozen selected configs sha256: {frozen_sha}")

    # -- Step 5: final evaluation across seeds -------------------------------
    print("\n--- final evaluation (seeds 42, 43, 44) ---")
    records: list[dict[str, Any]] = []
    for arm in arms:
        for seed in FINAL_SEEDS:
            if arm.analytical:
                model = None
                training = {
                    "validation_macro_nloss": None,
                    "best_step": None,
                    "nonfinite_training_batches": 0,
                    "seconds": 0.0,
                }
                counts = {
                    "encoder_shared": 0,
                    "interaction": 0,
                    "adapter": 0,
                    "head": 0,
                    "matched": 0,
                    "total": 0,
                }
            else:
                spec = (
                    encoder_spec_for_ratio_control()
                    if arm.explicit_ratios
                    else encoder_spec(EMBEDDING_DIM, REPRESENTATION_DIM)
                )
                model = arm.build(spec, seed)
                counts = model.parameter_counts()
                training = _train(
                    model,
                    SPEC,
                    constants,
                    benchmark,
                    arm,
                    device,
                    selected_lr[arm.arm_id],
                    seed,
                )

            slices: dict[str, Any] = {}
            predictions: dict[str, torch.Tensor] = {}
            for name in benchmark.evaluation_slice_names:
                report, logits = _evaluate(
                    model, SPEC, constants, benchmark[name], arm, device
                )
                slices[name] = report
                predictions[name] = logits

            consistency = _scale_consistency(
                predictions["id"], predictions["ood_common_scale"]
            )
            record = {
                "arm_id": arm.arm_id,
                "label": arm.label,
                "seed": seed,
                "learning_rate": selected_lr.get(arm.arm_id),
                "parameters": counts,
                "analytic_counts": analytic_flops.get(arm.arm_id),
                "training": training,
                "slices": slices,
                "scale_consistency": consistency,
            }
            records.append(record)
            print(
                f"  {arm.arm_id:<18} seed={seed}  "
                f"ID={slices['id']['macro_nloss']:.6f}  "
                f"scale={slices['ood_common_scale']['macro_nloss']:.4f}  "
                f"low={slices['ood_low_ratio']['macro_nloss']:.4f}  "
                f"high={slices['ood_high_ratio']['macro_nloss']:.4f}  "
                f"consistency={consistency['median_relative_deviation']:.4f}"
            )

    _write_outputs(args.output_dir, benchmark, frozen, frozen_sha, records)
    print(f"\nwrote results to {args.output_dir}")
    return 0


def _write_outputs(
    output_dir: Path,
    benchmark,
    frozen: dict[str, Any],
    frozen_sha: str,
    records: list[dict[str, Any]],
) -> None:
    result = {
        "schema_version": 1,
        "artifact_role": "synthetic_multiquotient_3_result",
        "benchmark_manifest": benchmark.manifest(),
        "benchmark_manifest_sha256": benchmark.manifest_sha256,
        "selected_configs_sha256": frozen_sha,
        "selected_configs": frozen,
        "records": records,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    (output_dir / "results_v1.json").write_text(text + "\n", encoding="utf-8")
    (output_dir / "results_v1.json.sha256").write_text(
        f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}  results_v1.json\n",
        encoding="utf-8",
    )

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_arm.setdefault(record["arm_id"], []).append(record)

    summary_rows = []
    per_target_rows = []
    for arm_id, arm_records in by_arm.items():
        label = arm_records[0]["label"]
        for slice_name in benchmark.evaluation_slice_names:
            values = [r["slices"][slice_name]["macro_nloss"] for r in arm_records]
            summary_rows.append(
                {
                    "arm_id": arm_id,
                    "label": label,
                    "slice": slice_name,
                    "macro_nloss_mean": statistics.fmean(values),
                    "macro_nloss_sd": (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    ),
                    "seeds": len(values),
                    "nonfinite_predictions": sum(
                        r["slices"][slice_name]["nonfinite_predictions"]
                        for r in arm_records
                    ),
                    "parameters_matched": arm_records[0]["parameters"]["matched"],
                    "parameters_total": arm_records[0]["parameters"]["total"],
                    "flops_per_example": (
                        (arm_records[0]["analytic_counts"] or {}).get("flops", "")
                    ),
                    "scale_consistency_median": statistics.fmean(
                        r["scale_consistency"]["median_relative_deviation"]
                        for r in arm_records
                    ),
                }
            )
            for target in TARGET_NAMES:
                per_values = [
                    r["slices"][slice_name]["per_target_normalized"][target]
                    for r in arm_records
                ]
                per_target_rows.append(
                    {
                        "arm_id": arm_id,
                        "label": label,
                        "slice": slice_name,
                        "target": target,
                        "normalized_loss_mean": statistics.fmean(per_values),
                        "normalized_loss_sd": (
                            statistics.stdev(per_values) if len(per_values) > 1 else 0.0
                        ),
                        "seeds": len(per_values),
                    }
                )

    _write_csv(output_dir / "multitask_summary.csv", summary_rows)
    _write_csv(output_dir / "multitask_per_target.csv", per_target_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
