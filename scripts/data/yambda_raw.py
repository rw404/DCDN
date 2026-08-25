#!/usr/bin/env python
"""Acquire the Yambda-50M listen stream and freeze its provenance.

Fail-closed, in the order `docs/public_raw_acquisition.md` requires: the exact
byte size and SHA-256 of every object are pinned **here in code** before a
single byte is fetched, and a download whose hash does not match is deleted
rather than kept. The manifest is written only after verification.

What is fetched, and what is deliberately not
---------------------------------------------
``flat/50m/listens.parquet`` is the only mandatory object. The frozen objective
in `configs/targets_multitask_v2.yaml` defines both Yambda targets from
``played_ratio_pct``, which lives in that file.

The two item-side mapping files are fetched because they are the only source of
categorical structure beyond the raw ids.

Three groups are deliberately skipped:

* ``likes/dislikes/unlikes/undislikes`` — the frozen
  ``event_stream_join_policy`` forbids joining them without a pre-registered
  attribution rule frozen before any metric, and no such rule exists;
* ``multi_event.parquet`` — the same streams in interleaved form;
* ``embeddings.parquet`` — 13.8 GB, which alone would consume half the durable
  envelope, and the protocol does not call for it.

Nothing here reads a row value. Parquet metadata is read for the schema only.

Usage::

    python scripts/data/yambda_raw.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
BASE = f"https://huggingface.co/datasets/yandex/yambda/resolve/{REVISION}"

#: Pinned before any fetch. Sizes and SHA-256 come from the Hugging Face tree
#: API at the revision above; the SHA-256 is the LFS object id.
OBJECTS: dict[str, dict[str, Any]] = {
    "flat/50m/listens.parquet": {
        "bytes": 369_110_332,
        "sha256": "eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1",
        "role": "mandatory_listen_stream",
    },
    "album_item_mapping.parquet": {
        "bytes": 54_528_894,
        "sha256": "559b6576063d96280414574b32292eee7b47ee0b9f2952cb3ae670a5d19125e4",
        "role": "item_side_categorical",
    },
    "artist_item_mapping.parquet": {
        "bytes": 38_274_621,
        "sha256": "cf1110aa2413ed70008ec9232d395660b2181ffda731002fb74da54e87b6f25d",
        "role": "item_side_categorical",
    },
}

SKIPPED = {
    "flat/50m/likes.parquet": "event_stream_join_policy_forbids_unattributed_join",
    "flat/50m/dislikes.parquet": "event_stream_join_policy_forbids_unattributed_join",
    "flat/50m/unlikes.parquet": "event_stream_join_policy_forbids_unattributed_join",
    "flat/50m/undislikes.parquet": (
        "event_stream_join_policy_forbids_unattributed_join"
    ),
    "flat/50m/multi_event.parquet": "same_streams_interleaved_not_needed",
    "embeddings.parquet": "13.8GB_would_consume_half_the_durable_envelope",
}


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class AcquisitionError(RuntimeError):
    """The bytes on the wire are not the bytes that were pinned."""


def _tree(path: str) -> list[dict[str, Any]]:
    url = (
        f"https://huggingface.co/api/datasets/yandex/yambda/tree/{REVISION}/{path}"
        if path else
        f"https://huggingface.co/api/datasets/yandex/yambda/tree/{REVISION}"
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read())


def resolve_pins() -> dict[str, dict[str, Any]]:
    """Read the authoritative size and hash for every object from the tree API.

    Called with `--pin` to fill in a blank hash, and always called to CHECK the
    pinned values against the source before downloading. A pinned value that
    disagrees with the tree is an error, not something to overwrite silently.
    """

    found: dict[str, dict[str, Any]] = {}
    for listing in ("", "flat/50m"):
        for entry in _tree(listing):
            lfs = entry.get("lfs") or {}
            if not lfs:
                continue
            found[entry["path"]] = {
                "bytes": int(lfs["size"]),
                "sha256": str(lfs["oid"]),
            }
    return found


def download(path: str, expected: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Stream one object to disk, hashing as it goes, and verify before keeping."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        request = urllib.request.Request(f"{BASE}/{path}")
        with urllib.request.urlopen(request, timeout=300) as response:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        handle.flush()
        os.fsync(handle.fileno())

    actual = digest.hexdigest()
    if written != expected["bytes"] or actual != expected["sha256"]:
        temporary.unlink(missing_ok=True)
        raise AcquisitionError(
            f"{path}: expected {expected['bytes']} bytes / {expected['sha256']}, "
            f"got {written} / {actual}; the partial file was deleted"
        )
    os.replace(temporary, destination)
    return {"bytes": written, "sha256": actual}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(os.environ.get("DCDN_DATA_CACHE", "/tmp"))
    parser.add_argument("--destination", type=Path, default=cache / "raw" / "yambda")
    parser.add_argument(
        "--pin",
        action="store_true",
        help="fill in blank pinned hashes from the tree API and print them",
    )
    args = parser.parse_args(argv)

    print(f"revision {REVISION}")
    authoritative = resolve_pins()

    for path, pinned in OBJECTS.items():
        source = authoritative.get(path)
        if source is None:
            raise AcquisitionError(f"{path} is not present at this revision")
        if not pinned["sha256"]:
            if not args.pin:
                raise AcquisitionError(
                    f"{path} has no pinned hash; re-run with --pin to record "
                    f"{source['sha256']} and {source['bytes']} bytes"
                )
            pinned["sha256"] = source["sha256"]
            pinned["bytes"] = source["bytes"]
            print(f"  pinned {path}: {source['bytes']} bytes {source['sha256']}")
            continue
        if (pinned["bytes"], pinned["sha256"]) != (
            source["bytes"], source["sha256"]
        ):
            raise AcquisitionError(
                f"{path}: the pinned size/hash disagrees with the source at this "
                f"revision. pinned={pinned['bytes']}/{pinned['sha256']} "
                f"source={source['bytes']}/{source['sha256']}"
            )
        print(f"  pin verified {path}")

    if args.pin:
        print("\nPins resolved. Re-run without --pin after copying them into "
              "OBJECTS.")
        return 0

    verified: dict[str, Any] = {}
    for path, pinned in OBJECTS.items():
        destination = args.destination / Path(path).name
        if destination.is_file():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest == pinned["sha256"]:
                print(f"  already present and verified: {Path(path).name}")
                verified[path] = {
                    "bytes": destination.stat().st_size, "sha256": digest
                }
                continue
            destination.unlink()
        print(f"  fetching {path} ({pinned['bytes'] / 2**20:.1f} MiB)")
        verified[path] = download(path, pinned, destination)
        print(f"    verified {verified[path]['sha256'][:16]}...")

    manifest_path = REPO_ROOT / "data" / "manifests" / "yambda_source_v1.yaml"
    lines = [
        "schema_version: 1",
        "dataset_id: yambda-50m",
        "artifact_role: raw_source_provenance_only",
        "acquisition_state: raw_download_verified",
        "raw_download_verified: true",
        "test_unlocked: false",
        "values_not_scanned: true",
        "test_labels_opened: false",
        "validation_or_test_metrics_read: false",
        "artifact_class: raw_public_source_manifest",
        f"verified_at_utc: '{_now()}'",
        "source:",
        "  repository: https://huggingface.co/datasets/yandex/yambda",
        f"  revision: {REVISION}",
        "  access: public_unauthenticated_read_only",
        "  credentials_used: false",
        "  telemetry_used: false",
        "  gated: false",
        "license_provenance:",
        "  declared: apache-2.0",
        "  raw_redistribution: forbidden_by_project_policy",
        "objects:",
    ]
    for path, entry in verified.items():
        lines += [
            f"  - path: {path}",
            f"    role: {OBJECTS[path]['role']}",
            f"    expected_bytes: {OBJECTS[path]['bytes']}",
            f"    verified_bytes: {entry['bytes']}",
            f"    expected_sha256: {OBJECTS[path]['sha256']}",
            f"    verified_sha256: {entry['sha256']}",
        ]
    lines.append("skipped_objects:")
    for path, reason in SKIPPED.items():
        lines += [f"  - path: {path}", f"    reason: {reason}"]
    lines += [
        "inspection:",
        "  byte_hashes_computed: true",
        "  parquet_metadata_read: false",
        "  row_values_scanned: false",
        "  test_labels_opened: false",
        "  validation_or_test_metrics_read: false",
        "storage:",
        "  class: ephemeral_regenerable_public_raw_data",
        "  persistent_raw_copy: false",
        "  committed_to_git: false",
        "",
    ]
    body = "\n".join(lines)
    manifest_path.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    Path(f"{manifest_path}.sha256").write_text(
        f"{digest}  {manifest_path.name}\n", encoding="utf-8"
    )
    print(f"\nmanifest {manifest_path.relative_to(REPO_ROOT)} sha256={digest}")
    print(f"raw destination: {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
