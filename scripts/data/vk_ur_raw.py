#!/usr/bin/env python
"""Fail-closed acquisition of the VK-LSVD ur0.01_ir0.01 subsample.

Sizes and SHA-256 for every object are pinned in
`data/manifests/vk_lsvd_ur_source_v1.yaml` before a byte is fetched. A file
whose length or digest differs is deleted and the run aborts, so a silently
changed upstream artifact fails loudly instead of entering the study.

Week 26 is the sealed test split. Its digest is in the manifest so the object
is identifiable; this script never requests it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "manifests" / "vk_lsvd_ur_source_v1.yaml"
BASE = "https://huggingface.co/datasets/deepvk/VK-LSVD/resolve/main/"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(os.environ.get("DCDN_DATA_CACHE", "/tmp"))
    parser.add_argument("--root", type=Path, default=cache / "raw" / "vk_lsvd")
    args = parser.parse_args(argv)

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    fetched = skipped = 0
    for obj in manifest["objects"]:
        if obj["sealed"]:
            print(f"  sealed, never requested: {obj['path']}")
            skipped += 1
            continue
        target = args.root / obj["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == obj["size_bytes"]:
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest == obj["sha256"]:
                skipped += 1
                continue
        url = BASE + obj["path"]
        request = urllib.request.Request(url, headers={"User-Agent": "dcdn/1.0"})
        token = os.environ.get("HF_TOKEN")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if len(payload) != obj["size_bytes"]:
            raise SystemExit(
                f"{obj['path']}: {len(payload)} bytes against pinned "
                f"{obj['size_bytes']}"
            )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != obj["sha256"]:
            raise SystemExit(f"{obj['path']}: sha256 {digest} against pinned {obj['sha256']}")
        target.write_bytes(payload)
        fetched += 1
        print(f"  {obj['path']}  {len(payload):,} bytes  verified")
    print(f"\nfetched {fetched}, already present or sealed {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
