#!/usr/bin/env python3
"""Materialize the immutable VK-LSVD C+P selected_configs.lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dcdn.selected_configs_materialize import (
    OUTPUT_PATH,
    materialize_vk_selected_configs_lock,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    lock = materialize_vk_selected_configs_lock(root)
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(root / OUTPUT_PATH),
                "sha256": lock.digest,
                "test_data_accessed": False,
                "test_metrics_consulted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
