#!/usr/bin/env python
"""Single-head VK-LSVD study over the multi-target runtime views.

Consumes the `runtime/vk_lsvd/multitask_v2` views built elsewhere in this
repository and builds nothing of its own. The frozen protocol is
`configs/data/vk_lsvd/single_target_rq2_v1.yaml`, which declares the whole
target ladder — `watch_time` plus three thresholds of it — **before** any of
them was run.

Why an adapter rather than the KuaiRand runner directly
-------------------------------------------------------
Those views store one `.npy` per array with a separate REDUCED (`c_plus_p`) and
FULL (`c_plus_p_plus_e`) directory, where the KuaiRand and KuaiRec views store a
single `.npz` per split holding both regimes. The arrays mean the same things;
only the layout differs, so this module presents them through the same `Views`
interface the shared runner already expects and everything downstream — arms,
capacity control, metrics, seed scopes, the convergence gate — is reused
unchanged.

The four labels all come from the SAME stored column. The binary variants are
thresholds of the normalized watch time already in the view, so no new outcome
column is read.

Usage::

    python scripts/run_vk_single_target.py --device cuda:0 --target watch_time_gt10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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


krm = _load(
    "run_kuairand_multitask", REPO_ROOT / "scripts" / "run_kuairand_multitask.py"
)
sta = _load(
    "run_kuairand_single_target",
    REPO_ROOT / "scripts" / "run_kuairand_single_target.py",
)

from dcdn.data.vk_lsvd_runtime import (  # noqa: E402
    DENOMINATOR_ELIGIBLE_FIELDS,
    denominator_mask_sha256,
)

views_builder = _load(
    "vk_lsvd_multitask_views",
    REPO_ROOT / "scripts" / "data" / "vk_lsvd_multitask_views.py",
)

CONTRACT = REPO_ROOT / "configs" / "data" / "vk_lsvd" / "single_target_rq2_v1.yaml"
CLIP_SECONDS = 157.0

#: `configs/data/vk_lsvd/features_multitask_v1.yaml` change
#: `ials_norms_leave_the_denominator_mask` drops the six iALS norms from the
#: eligible set, leaving five historical denominators. The override requires
#: reciprocal mask `false` for cosine, dot and norm, so the DCDN reciprocal
#: branch sees only semantically positive historical denominators.
ELIGIBLE_DENOMINATORS = tuple(
    field for field in DENOMINATOR_ELIGIBLE_FIELDS if "ials" not in field
)


def contract_denominator_mask(fields: tuple[str, ...], expected_sha256: str) -> tuple[bool, ...]:
    """The mask the view was built with, verified rather than inferred.

    An earlier version of this adapter derived the mask from the stored
    `denominator_validity` tensor, taking "this coordinate is gated on at
    least one row" to mean "this coordinate is a denominator". That is a
    different predicate and it produced a different set: one coordinate in
    REDUCED and eighteen in FULL, the latter including all seventeen R_div
    ratios, which the contract's pre-training checks forbid outright. The
    validity tensor flags fallback values, not eligibility.

    The mask is therefore taken from the contract's ordered field lists and
    checked against the `denominator_mask_sha256` the view itself recorded. A
    mismatch aborts, so a future change to either side cannot pass silently.
    """

    eligible = set(ELIGIBLE_DENOMINATORS)
    missing = eligible - set(fields)
    if missing:
        raise SystemExit(f"eligible denominators absent from the view: {sorted(missing)}")
    mask = tuple(field in eligible for field in fields)
    actual = denominator_mask_sha256(mask)
    if actual != expected_sha256:
        raise SystemExit(
            "denominator mask does not reproduce the view's recorded hash: "
            f"built {actual} against stored {expected_sha256}"
        )
    return mask

#: Declared in the frozen protocol before any run. `watch_time` is the frozen
#: primary; the three thresholds are derived from the same column.
TARGETS: dict[str, dict[str, Any]] = {
    "watch_time": {"kind": "regression", "threshold": None},
    "watch_time_gt10": {"kind": "binary", "threshold": 10.0},
    "watch_time_gt30": {"kind": "binary", "threshold": 30.0},
    "watch_time_gt60": {"kind": "binary", "threshold": 60.0},
}


class VkViews:
    """The `multitask_v2` arrays, presented as the shared runner's `Views`."""

    def __init__(
        self, root: Path, device: torch.device, target: str,
    ) -> None:
        reduced_cache = json.loads(
            (root / "c_plus_p" / "cache.json").read_text(encoding="utf-8")
        )
        full_cache = json.loads(
            (root / "c_plus_p_plus_e" / "cache.json").read_text(encoding="utf-8")
        )
        self.cardinalities = reduced_cache["categorical_cardinalities"]
        self.engineered_block = "E"
        self.blocks = ("P", "V", "E")
        self.preconditioner = "none"

        threshold = TARGETS[target]["threshold"]
        self.data: dict[str, dict[str, torch.Tensor]] = {}
        widths: dict[str, int] = {}
        for split in ("train", "validation"):
            reduced = np.load(
                root / "c_plus_p" / split / "operand_features.f32.npy"
            )
            full = np.load(
                root / "c_plus_p_plus_e" / split / "operand_features.f32.npy"
            )
            categorical = np.load(
                root / "c_plus_p" / split / "context_tokens.i32.npy"
            ).astype(np.int64)
            validity = np.load(
                root / "c_plus_p" / split / "denominator_validity.u8.npy"
            ).astype(bool)
            full_validity = np.load(
                root / "c_plus_p_plus_e" / split / "denominator_validity.u8.npy"
            ).astype(bool)
            watch = np.load(root / "c_plus_p" / split / "targets.f32.npy")[:, :1]
            if threshold is None:
                labels = watch
            else:
                labels = (watch > threshold / CLIP_SECONDS).astype(np.float32)
            widths["reduced"] = reduced.shape[1]
            widths["full"] = full.shape[1]
            self.data[split] = {
                "categorical": torch.from_numpy(categorical).to(device),
                "reduced": torch.from_numpy(reduced).to(device),
                "full": torch.from_numpy(full).to(device),
                "reduced_validity": torch.from_numpy(validity).to(device),
                "full_validity": torch.from_numpy(full_validity).to(device),
                "labels": torch.from_numpy(labels).to(device),
                "masks": torch.ones(
                    labels.shape, dtype=torch.bool, device=device
                ),
                "user_ids": torch.from_numpy(
                    np.load(root / "c_plus_p" / split / "user_ids.i64.npy")
                ).to(device),
            }

        self.reduced_width = widths["reduced"]
        self.full_width = widths["full"]
        reduced_fields = tuple(views_builder.ORDERED_OPERAND_FIELDS)
        full_fields = reduced_fields + tuple(views_builder.R_DIV_FIELDS)
        if len(reduced_fields) != self.reduced_width:
            raise SystemExit(
                f"REDUCED width {self.reduced_width} against "
                f"{len(reduced_fields)} contract fields"
            )
        if len(full_fields) != self.full_width:
            raise SystemExit(
                f"FULL width {self.full_width} against {len(full_fields)} contract fields"
            )
        self.mask_reduced = contract_denominator_mask(
            reduced_fields, reduced_cache["denominator_mask_sha256"]
        )
        self.mask_full = contract_denominator_mask(
            full_fields, full_cache["denominator_mask_sha256"]
        )
        self.manifest = {
            "reduced_cache_sha256": reduced_cache["cache_content_sha256"],
            "full_cache_sha256": full_cache["cache_content_sha256"],
            "denominator_mask_sha256": reduced_cache["denominator_mask_sha256"],
            "full_denominator_mask_sha256": full_cache["denominator_mask_sha256"],
            "denominator_mask_source": "contract_field_lists_verified_against_view_hash",
            "eligible_denominators": list(ELIGIBLE_DENOMINATORS),
            "label_names": [target],
            "clip_seconds": CLIP_SECONDS,
            "watch_time_c_seconds": CLIP_SECONDS,
        }

    def numeric(self, split: str, regime: str) -> torch.Tensor:
        return self.data[split][regime]

    def validity(self, split: str, regime: str) -> torch.Tensor:
        return self.data[split][f"{regime}_validity"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(__import__("os").environ.get("DCDN_DATA_CACHE", "/tmp"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target", choices=sorted(TARGETS), default="watch_time")
    parser.add_argument(
        "--views", type=Path,
        default=cache / "runtime" / "vk_lsvd" / "multitask_v2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annex-seeds", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--field", action="store_true")
    parser.add_argument("--arms", default=None)
    args = parser.parse_args(argv)

    # The shared runner reads the target kind from its own table; register the
    # VK targets there so nothing else has to change.
    sta.SUPPORTED_TARGETS[args.target] = TARGETS[args.target]["kind"]
    sta.krm_manifest_targets = lambda _views: [args.target]

    # `sta` loaded its own instance of the multitask runner, so patching the
    # copy this module imported would have no effect - the single-target runner
    # resolves `Views` through `sta.krm`, which is a different object.
    original = sta.krm.Views
    sta.krm.Views = lambda root, device, preconditioner="none": VkViews(  # noqa: ARG005
        root, device, args.target
    )
    try:
        forwarded = [
            "--device", args.device, "--target", args.target,
            "--preconditioner", "none",
            "--views", str(args.views),
            "--output-dir", str(args.output_dir),
        ]
        if args.annex_seeds:
            forwarded.append("--annex-seeds")
        if args.save_predictions:
            forwarded.append("--save-predictions")
        if args.field:
            forwarded.append("--field")
        if args.arms:
            forwarded += ["--arms", args.arms]
        return sta.main(forwarded)
    finally:
        sta.krm.Views = original


if __name__ == "__main__":
    raise SystemExit(main())
