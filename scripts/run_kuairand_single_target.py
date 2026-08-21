#!/usr/bin/env python
"""Single-target `watch_time` follow-up on KuaiRand-Pure.

The same seven arms, the same splits, the same encoder, the same capacity
control, the same seeds and the same learning-rate grid as the multi-target
study in `run_kuairand_multitask.py`. **One** thing differs: the model carries a
single regression head for `watch_time` and is trained on that target's loss
alone, instead of five heads under the Macro-NLoss aggregate.

Why this target, and why separately
-----------------------------------
`watch_time` is `clip(play_time_ms / 1000, 0, c) / c` — a ratio. Of the five
heads it is the only one whose structure matches the hypothesis DCDN's
reciprocal branch encodes, so isolating it is the cleanest test of that
hypothesis available on real data here. It is also the head that dominates
Macro-NLoss on this dataset (per-target normalized losses span 0.50 to 10.76
against 0.32 to 2.90 for every other head), which means the multi-target
aggregate cannot distinguish "this arm models watch_time well" from "this arm's
shared trunk was pulled around by four binary heads". Training it alone
separates those.

Standing
--------
This is **not** the frozen single-target study. `configs/single_target_followup_v1.yaml`
requires `training_may_start_only_after: main_multi_target_table_complete`, and
the main table is not complete — VK-LSVD is in progress and Yambda-50M has no
pipeline. That frozen study remains blocked. This one is scoped to the
validation-only KuaiRand RQ2/RQ3 work, declares itself in
`configs/data/kuairand_pure/single_target_rq23_v1.yaml`, and may not be
substituted for the frozen study or used to rescue any main-table conclusion.

The test split is never opened.

Usage::

    python scripts/run_kuairand_single_target.py --device cuda:0 --annex-seeds
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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


beff = _load("benchmark_efficiency", REPO_ROOT / "scripts" / "benchmark_efficiency.py")
krm = _load(
    "run_kuairand_multitask",
    REPO_ROOT / "scripts" / "run_kuairand_multitask.py",
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

#: Which single heads this study may be run on, and what kind each is.
#: `watch_time` is the ratio target; `long_view` is the frozen primary
#: confirmatory target of the multi-target study.
SUPPORTED_TARGETS = {
    "watch_time": "regression",
    "long_view": "binary",
    # KuaiRec's canonical engagement label: the user played the video through
    # more than twice.
    "watch_ratio_gt2": "binary",
    # Yambda, both from `configs/targets_multitask_v2.yaml` verbatim.
    # `play_fraction` is the normalized listening time.
    "play_fraction": "regression",
    "listen50": "binary",
}


def krm_manifest_targets(views: Path) -> list[str]:
    """The label columns the materialized view actually carries, in order."""

    manifest = json.loads((views / "manifest_v1.json").read_text(encoding="utf-8"))
    declared = manifest.get("label_names")
    if declared:
        return list(declared)
    # KuaiRand's manifest predates `label_names`; its column order is the
    # runner's frozen TARGETS tuple.
    return list(krm.TARGETS)


def build_spec(target: str) -> MultiTargetSpec:
    """A one-target spec.

    `macro_nloss` over a single target reduces to that target's normalized
    loss, so the selection metric, the early-stopping metric and the reported
    metric are the same quantity as the corresponding head of the multi-task
    model — which is what makes the two directly comparable.
    """

    return MultiTargetSpec(
        dataset_id="kuairand_pure",
        targets=(TargetSpec(name=target, kind=SUPPORTED_TARGETS[target]),),
        primary_confirmatory_target=target,
    )


def narrow_to_target(views: krm.Views, target_index: int) -> None:
    """Restrict the loaded labels and masks to the single follow-up target.

    Everything else about `views` — the numeric blocks, the input
    preconditioner, the denominator mask, the split row order — is left exactly
    as the multi-target study built it. The follow-up protocol requires the
    splits and feature manifests to be held identical, so nothing here may
    touch them.
    """

    for split in ("train", "validation"):
        payload = views.data[split]
        payload["labels"] = payload["labels"][:, target_index : target_index + 1]
        payload["masks"] = payload["masks"][:, target_index : target_index + 1]


def _headline(report: dict[str, Any], target: str) -> str:
    """One short per-seed line, in the units that suit the head's kind."""

    metrics = report["frozen_metrics"][target]
    if metrics["kind"] == "binary":
        return (
            f"AUC={metrics['roc_auc']:.4f} gAUC={metrics['user_gauc']:.4f} "
            f"(cov {metrics['user_gauc_group_coverage']:.2f})"
        )
    return (
        f"xAUC={metrics['global_xauc']:.4f} "
        f"xgAUC={metrics['user_xgauc']:.4f} "
        f"MAE={metrics['mae_seconds']:.2f}s"
    )


@torch.no_grad()
def evaluate(model, views, arm, constants, spec, target, c_seconds):
    model.eval()
    valid = views.data["validation"]
    numeric = views.numeric("validation", arm.regime)
    validity = views.validity("validation", arm.regime)
    predictions = krm.forward_in_batches(
        model, valid["categorical"], numeric, validity
    )
    nonfinite = int((~torch.isfinite(predictions)).sum())
    safe = torch.nan_to_num(predictions)
    from dcdn.multitask import macro_nloss

    report = macro_nloss(
        spec, constants, safe, valid["labels"], valid["masks"]
    ).as_report(spec)
    report["nonfinite_predictions"] = nonfinite

    scores = safe.cpu().numpy()
    labels = valid["labels"].cpu().numpy()
    # The same frozen metric set the multi-target study reports, so the single
    # head and the corresponding multi-task head are compared on identical
    # definitions rather than on two similar-looking implementations.
    report["_predictions"] = scores
    report["frozen_metrics"] = krm.frozen_metric_set(
        spec, scores, labels,
        valid["masks"].cpu().numpy(),
        valid["categorical"][:, krm.USER_COLUMN].cpu().numpy(),
        c_seconds,
    )
    if target == "watch_time":
        normalized_mae = float(np.abs(scores[:, 0] - labels[:, 0]).mean())
        report["watch_time_mae"] = normalized_mae
        report["watch_time_mae_seconds"] = normalized_mae * c_seconds
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--views",
        type=Path,
        default=Path(__import__("os").environ.get("DCDN_DATA_CACHE", "/tmp"))
        / "runtime"
        / "kuairand_pure_multitask_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "kuairand_multitask" / "single_target_v1",
    )
    parser.add_argument(
        "--preconditioner", choices=krm.PRECONDITIONERS, default="rms"
    )
    parser.add_argument("--annex-seeds", action="store_true")
    parser.add_argument(
        "--field",
        action="store_true",
        help="also run the five architectures outside the frozen M1-M7 matrix "
             "(MLP, DeepFM, original DCN, AutoInt+, Wukong-small), reported "
             "separately so the frozen matrix is not silently widened",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="persist per-row validation predictions to the data cache so a "
             "future metric can be computed without retraining",
    )
    parser.add_argument(
        "--target", choices=sorted(SUPPORTED_TARGETS), default="watch_time"
    )
    args = parser.parse_args(argv)

    target = args.target
    spec = build_spec(target)
    # KuaiRand materializes five label columns and this study selects one;
    # KuaiRec materializes the single watch_time column already, so there is
    # nothing to select. Deciding by the manifest rather than by dataset name
    # keeps the runner dataset-agnostic.
    label_names = krm_manifest_targets(args.views)
    target_index = label_names.index(target) if target in label_names else 0

    device = torch.device(args.device)
    beff.prepare_output_dir(args.output_dir)
    views = krm.Views(args.views, device, preconditioner=args.preconditioner)
    narrow_to_target(views, target_index)
    # KuaiRand and KuaiRec normalize watch time by a single train-only duration
    # constant, so their MAE converts back to seconds. Yambda's target is a
    # fraction of a per-row track length, so no single constant exists; 1.0
    # leaves the error in normalized units, and the manifest records which case
    # this is rather than presenting normalized units as seconds.
    c_seconds = float(views.manifest.get("watch_time_c_seconds", 1.0))
    seconds_are_real = "watch_time_c_seconds" in views.manifest
    print(
        f"views: train={views.data['train']['labels'].shape[0]:,} "
        f"validation={views.data['validation']['labels'].shape[0]:,} "
        f"reduced={views.reduced_width} full={views.full_width} "
        f"target={target}"
    )

    constants = compute_task_constants(
        spec, views.data["train"]["labels"], views.data["train"]["masks"]
    )
    print(f"frozen task constant: {target}={constants.values[0]:.6g}")

    # Mechanical capacity control, recomputed for a one-wide head. No
    # validation metric is read. The head is narrower here than in the
    # multi-target study, identically for every arm, so the resolved rank is
    # matched to the DCDN of *this* study rather than inherited from another.
    common = common_counts(
        len(views.cardinalities) * krm.EMBEDDING_DIM + views.reduced_width,
        krm.REPRESENTATION_DIM, krm.HEAD_HIDDEN_DIMS, 1,
    )
    target_counts = dcdn_counts(views.reduced_width, 1, common)
    resolution = resolve_capacity_control(
        target=target_counts, operand_width=views.reduced_width, common=common,
        ranks=tuple(range(1, 129)), layer_options=(1, 2), expert_options=(1, 2, 4),
    )
    payload = resolution.payload()
    if not resolution.matched:
        raise SystemExit(f"no capacity control inside tolerance: {json.dumps(payload)}")
    capacity = payload["selected"]
    print(
        f"capacity control: rank={capacity['low_rank']} "
        f"layers={capacity['num_layers']} experts={capacity['num_experts']} "
        f"param_ratio={payload['selected_ratios']['parameter_ratio']:.4f} "
        f"flop_ratio={payload['selected_ratios']['flop_ratio']:.4f}"
    )
    arms = krm.build_arms(capacity)
    if args.field:
        arms = arms + krm.build_field_arms()
        print(f"field arms appended: {len(krm.build_field_arms())} "
              f"({', '.join(a.arm_id for a in krm.build_field_arms())})")

    def make(arm: krm.Arm, seed: int) -> krm.KuaiRandModel:
        width = views.reduced_width if arm.regime == "reduced" else views.full_width
        mask = views.mask_reduced if arm.regime == "reduced" else views.mask_full
        return krm.KuaiRandModel(
            views.cardinalities, width, mask, arm.config, seed,
            with_afn_view=arm.with_afn_view,
            output_dim=1,
        )

    # `train_one` selects on `macro_nloss` built from the module-level SPEC of
    # the multi-target runner. Point it at the one-target spec for the duration
    # of this study so selection, early stopping and reporting all use the same
    # single-target quantity.
    krm.SPEC = spec

    manifest = {
        "artifact_role": "kuairand_pure_single_target_followup_manifest",
        "schema_version": 1,
        "dataset_id": "kuairand_pure",
        "study": "kuairand_pure_single_target_watch_time",
        "is_the_frozen_single_target_study": False,
        "frozen_study_blocked_on": "main_multi_target_table_complete",
        "target": target,
        "target_kind": SUPPORTED_TARGETS[target],
        "duration_constant_seconds": c_seconds,
        "mae_seconds_is_real_seconds": seconds_are_real,
        "objective_spec_sha256": spec.contract_sha256,
        "task_constants_sha256": constants.contract_sha256,
        "view_manifest_sha256": hashlib.sha256(
            (args.views / "manifest_v1.json").read_bytes()
        ).hexdigest(),
        "held_identical_to_multitask": [
            "splits", "feature_manifests", "input_preconditioner", "encoder",
            "representation_width", "capacity_control", "optimizer",
            "batch_size", "maximum_steps", "patience", "learning_rate_grid",
            "seeds",
        ],
        "only_difference": "single regression head trained on watch_time alone",
        "capacity_resolution": payload,
        "arm_order": [a.arm_id for a in arms],
        "final_seeds": list(krm.FINAL_SEEDS),
        "annex_seeds": list(krm.ANNEX_SEEDS) if args.annex_seeds else [],
        "learning_rates": list(krm.LEARNING_RATES),
        "test_unlocked": False,
        "evaluated_split": "validation",
        "locked_test_window_read": False,
    }
    digest = beff.publish_json(
        args.output_dir / "pre_training_manifest_v1.json", manifest
    )
    print(f"pre-training manifest sha256: {digest}\n")

    store = None
    if args.save_predictions:
        store = krm.PredictionStore(
            Path(__import__("os").environ.get("DCDN_DATA_CACHE", "/tmp")),
            args.output_dir.name,
        )
        store.add_context(views, spec)

    print("--- tuning (seed 42, validation-selected) ---")
    selected: dict[str, float] = {}
    for arm in arms:
        trials = []
        for learning_rate in krm.LEARNING_RATES:
            started = time.perf_counter()
            outcome = krm.train_one(
                make(arm, krm.TUNING_SEED), views, arm, constants, device,
                learning_rate, krm.TUNING_SEED,
            )
            seconds = time.perf_counter() - started
            trials.append({"learning_rate": learning_rate, **outcome})
            print(
                f"  {arm.arm_id:<24} lr={learning_rate:<7g} "
                f"val NLoss={outcome['validation_macro_nloss']:.6f} ({seconds:.0f}s)"
            )
        best = min(trials, key=lambda t: t["validation_macro_nloss"])
        selected[arm.arm_id] = best["learning_rate"]

    frozen = {
        "artifact_role": "kuairand_pure_single_target_selected_configs",
        "schema_version": 1,
        "target": target,
        "selected_learning_rate": selected,
        "capacity_resolution": payload,
        "objective_spec_sha256": spec.contract_sha256,
        "task_constants_sha256": constants.contract_sha256,
    }
    frozen_digest = beff.publish_json(
        args.output_dir / "selected_configs_v1.json", frozen
    )
    print(f"\nfrozen selected configs sha256: {frozen_digest}\n")

    seeds = tuple(krm.FINAL_SEEDS) + (
        tuple(krm.ANNEX_SEEDS) if args.annex_seeds else ()
    )
    print(f"--- final evaluation (seeds {', '.join(map(str, seeds))}; validation) ---")
    records: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            model = make(arm, seed)
            training = krm.train_one(
                model, views, arm, constants, device, selected[arm.arm_id], seed
            )
            report = evaluate(
                model, views, arm, constants, spec, target, c_seconds
            )
            counts = model.counts()
            records.append({
                "arm_id": arm.arm_id, "label": arm.label, "regime": arm.regime,
                "seed_class": (
                    "frozen_protocol" if seed in krm.FINAL_SEEDS
                    else "stability_annex"
                ),
                "tuning_contaminated": seed == krm.TUNING_SEED,
                "seed": seed, "learning_rate": selected[arm.arm_id],
                "parameters": counts, "training": training, "validation": report,
            })
            stored = report.pop("_predictions")
            if store is not None:
                store.add(arm.arm_id, seed, stored)
            print(
                f"  {arm.arm_id:<24} seed={seed} "
                f"NLoss={report['macro_nloss']:.6f} {_headline(report, target)}"
            )
            del model
            torch.cuda.empty_cache()

    predictions_digest = store.write() if store is not None else None
    if predictions_digest:
        print(f"\npredictions cached: {store.path} sha256={predictions_digest}")

    published = beff.publish_json(args.output_dir / "results_v1.json", {
        "artifact_role": "kuairand_pure_single_target_results",
        "schema_version": 1,
        "study": "kuairand_pure_single_target_watch_time",
        "is_the_frozen_single_target_study": False,
        "pre_training_manifest_sha256": digest,
        "selected_configs_sha256": frozen_digest,
        "evaluated_split": "validation",
        "test_unlocked": False,
        "target": target,
        "targets": [target],
        "final_seeds": list(krm.FINAL_SEEDS),
        "annex_seeds": list(krm.ANNEX_SEEDS) if args.annex_seeds else [],
        "cached_predictions_sha256": predictions_digest,
        "cached_predictions_path": str(store.path) if store is not None else None,
        "records": records,
    })
    print(f"\npublished results_v1.json sha256={published}")
    print(f"output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
