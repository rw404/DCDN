#!/usr/bin/env python
"""Single-head study on the VK-LSVD ur0.01_ir0.01 high-density feature view.

The views this consumes were built by `scripts/data/vk_ur_materialize.py` from
`configs/data/vk_lsvd_ur/features_v1.yaml`, frozen at sha256
2bf0dfb4732efa2042ca9c53174f8a074994d8173347f4e3d188a1f1824e6f1d before any arm
was trained. They already carry the contract's train-median normalization, so
the shared runner is invoked with `--preconditioner none`.

Their layout is the KuaiRand one - `categorical`, `P`, `V`, `R`, `labels` per
split with a manifest carrying the denominator mask - so no adapter is needed
and everything downstream is the shared code path unchanged.

The registered target ladder and its thresholds come from the frozen contract,
where they were chosen from train positive rates while no model existed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

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


sta = _load("run_kuairand_single_target",
            REPO_ROOT / "scripts" / "run_kuairand_single_target.py")

TARGETS = {
    "timespent": "regression",
    "timespent_gt10": "binary",
    "timespent_gt20": "binary",
    "timespent_gt40": "binary",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(os.environ.get("DCDN_DATA_CACHE", "/tmp"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target", choices=sorted(TARGETS), default="timespent")
    parser.add_argument("--views", type=Path,
                        default=cache / "runtime" / "vk_lsvd_ur_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annex-seeds", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--field", action="store_true")
    parser.add_argument("--arms", default=None)
    args = parser.parse_args(argv)

    sta.SUPPORTED_TARGETS.update(TARGETS)
    forwarded = ["--device", args.device, "--target", args.target,
                 "--preconditioner", "none", "--views", str(args.views),
                 "--output-dir", str(args.output_dir)]
    for flag in ("annex_seeds", "save_predictions", "field"):
        if getattr(args, flag):
            forwarded.append("--" + flag.replace("_", "-"))
    if args.arms:
        forwarded += ["--arms", args.arms]
    return sta.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
