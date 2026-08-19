#!/usr/bin/env python3
"""Materialize the immutable capacity-control LR authorization once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dcdn.selection_artifact import (
    CAPACITY_LR_BINDING_PATH,
    materialize_capacity_lr_binding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="clean committed repository checkout",
    )
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    evidence = materialize_capacity_lr_binding(root)
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(root / CAPACITY_LR_BINDING_PATH),
                "sha256": evidence.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
