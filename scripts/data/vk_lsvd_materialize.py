#!/usr/bin/env python3
"""Materialize frozen VK-LSVD train/validation feature views.

The command verifies the source and contract locks, fits history-only state on
weeks 0--19, fits mappings/scales on train weeks 20--24, and emits week 25 as
validation.  The sealed test week is not opened or decoded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

from dcdn.data.vk_lsvd_materialize import (  # noqa: E402
    DEFAULT_COST_LEDGER,
    DEFAULT_FEATURE_CONTRACT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RAW_ROOT,
    DEFAULT_VIEWS_MANIFEST,
    MAX_ALLOWED_ETA_HOURS,
    VKLSVDMaterializationError,
    materialize_vk_lsvd_views,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_FEATURE_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_VIEWS_MANIFEST)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_COST_LEDGER)
    parser.add_argument("--eta-hours", type=float, default=MAX_ALLOWED_ETA_HOURS)
    return parser


def _progress(event: str, details: dict[str, Any]) -> None:
    print(
        json.dumps(
            {"event": event, **details},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest, digest = materialize_vk_lsvd_views(
            repository_root=args.repository_root,
            raw_root=args.raw_root,
            output_root=args.output_root,
            contract_path=args.contract,
            manifest_path=args.manifest,
            ledger_path=args.ledger,
            eta_hours=args.eta_hours,
            progress_callback=_progress,
        )
    except (OSError, VKLSVDMaterializationError) as error:
        print(
            json.dumps(
                {"event": "blocked", "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "event": "materialization_complete",
                "manifest_sha256": digest,
                "train_rows": manifest["row_counts"]["train"],
                "validation_rows": manifest["row_counts"]["validation"],
                "test_unlocked": False,
                "week26_file_open_attempted": False,
                "validation_model_metrics_computed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
