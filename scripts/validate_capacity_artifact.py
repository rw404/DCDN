#!/usr/bin/env python3
"""Validate a metric-free capacity artifact and its frozen repository inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from dcdn.capacity_artifact import (
    load_verified_capacity_artifact,
    verify_repository_bindings,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        default="configs/capacity/vk_lsvd_c_plus_p_v1.json",
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parent.parent
    payload = load_verified_capacity_artifact(repository_root / args.artifact)
    verify_repository_bindings(payload, repository_root)
    print(
        "capacity artifact valid: "
        f"{payload['decision']['selected_candidate']} "
        f"(rank={payload['decision']['selected_low_rank']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
