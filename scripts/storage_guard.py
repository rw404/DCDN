#!/usr/bin/env python3
"""Local durable-project and temporary-overlay storage safety guard."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3
EXPECTED_PERSISTENT_CAPACITY_BYTES = 50 * GIB
MAX_PROJECT_DURABLE_BYTES = 40 * GIB
MIN_PERSISTENT_HEADROOM_BYTES = 10 * GIB
MIN_SMALL_FS_HEADROOM_BYTES = 5 * GIB
MIN_SMALL_FS_HEADROOM_FRACTION = 0.20
MIN_OVERLAY_AVAILABLE_BYTES = 20 * GIB
MAX_OVERLAY_USAGE_FRACTION = 0.85


@dataclass(frozen=True)
class MountInfo:
    mount_point: str
    fs_type: str
    source: str


@dataclass(frozen=True)
class FilesystemStats:
    total_bytes: int
    available_bytes: int

    @property
    def usage_fraction(self) -> float:
        if self.total_bytes <= 0:
            raise ValueError("filesystem total_bytes must be positive")
        return 1.0 - self.available_bytes / self.total_bytes


def _unescape_mount_field(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def find_mount(
    path: Path, mountinfo_path: Path = Path("/proc/self/mountinfo")
) -> MountInfo:
    """Return the longest mount-point match from Linux mountinfo."""

    resolved = path.resolve(strict=True)
    matches: list[MountInfo] = []
    for line in mountinfo_path.read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or len(right_fields) < 2:
            continue
        mount_point = Path(_unescape_mount_field(left_fields[4]))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        matches.append(
            MountInfo(
                mount_point=str(mount_point),
                fs_type=right_fields[0],
                source=_unescape_mount_field(right_fields[1]),
            )
        )
    if not matches:
        raise RuntimeError(f"no mountinfo entry found for {resolved}")
    return max(matches, key=lambda item: len(item.mount_point))


def filesystem_stats(path: Path) -> FilesystemStats:
    values = os.statvfs(path)
    fragment_size = values.f_frsize or values.f_bsize
    return FilesystemStats(
        total_bytes=fragment_size * values.f_blocks,
        available_bytes=fragment_size * values.f_bavail,
    )


def measure_project_bytes(root: Path) -> int:
    """Conservatively count unique regular-file logical or allocated bytes."""

    total = 0
    seen: set[tuple[int, int]] = set()
    for directory, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            path = Path(directory, filename)
            try:
                stat = path.lstat()
            except FileNotFoundError:
                continue
            if path.is_symlink() or not path.is_file():
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            allocated = getattr(stat, "st_blocks", 0) * 512
            total += max(stat.st_size, allocated)
    return total


def _looks_shared_fuse(mount: MountInfo, total_bytes: int) -> bool:
    descriptor = f"{mount.fs_type} {mount.source}".lower()
    return (
        "fuse" in descriptor
        or "mfs" in descriptor
        or "runpod" in descriptor
        or total_bytes >= 4 * EXPECTED_PERSISTENT_CAPACITY_BYTES
    )


def evaluate_storage(
    *,
    project_bytes: int,
    persistent: FilesystemStats,
    persistent_mount: MountInfo,
    overlay: FilesystemStats,
) -> dict[str, object]:
    """Evaluate the frozen logical quota plus locally observed free space."""

    for name, value in (
        ("project_bytes", project_bytes),
        ("persistent total", persistent.total_bytes),
        ("persistent available", persistent.available_bytes),
        ("overlay total", overlay.total_bytes),
        ("overlay available", overlay.available_bytes),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if persistent.total_bytes == 0 or overlay.total_bytes == 0:
        raise ValueError("filesystem totals must be positive")

    shared_fuse = _looks_shared_fuse(persistent_mount, persistent.total_bytes)
    if persistent.total_bytes < EXPECTED_PERSISTENT_CAPACITY_BYTES:
        logical_capacity = persistent.total_bytes
        required_headroom = max(
            MIN_SMALL_FS_HEADROOM_BYTES,
            int(persistent.total_bytes * MIN_SMALL_FS_HEADROOM_FRACTION),
        )
        project_cap = min(
            MAX_PROJECT_DURABLE_BYTES,
            max(0, logical_capacity - required_headroom),
        )
        capacity_basis = "smaller_locally_observed_filesystem"
    else:
        # RunPod's network FUSE reports a large shared backend through statvfs,
        # not the owner-visible volume quota. Retain the frozen 50 GiB logical
        # envelope even when df/statvfs appears to expose terabytes.
        logical_capacity = EXPECTED_PERSISTENT_CAPACITY_BYTES
        required_headroom = MIN_PERSISTENT_HEADROOM_BYTES
        project_cap = MAX_PROJECT_DURABLE_BYTES
        capacity_basis = "frozen_50_gib_logical_envelope"

    logical_headroom = max(0, logical_capacity - project_bytes)
    overlay_usage = overlay.usage_fraction
    violations: list[str] = []
    if project_bytes > project_cap:
        violations.append("project-owned durable bytes exceed the dynamic cap")
    if logical_headroom < required_headroom:
        violations.append("logical persistent headroom is below the required reserve")
    if not shared_fuse and persistent.available_bytes < required_headroom:
        violations.append("locally observed persistent free space is below the reserve")
    if overlay.available_bytes < MIN_OVERLAY_AVAILABLE_BYTES:
        violations.append("temporary overlay has less than 20 GiB available")
    if overlay_usage >= MAX_OVERLAY_USAGE_FRACTION:
        violations.append("temporary overlay usage is at or above 85%")

    return {
        "ok": not violations,
        "violations": violations,
        "project_bytes": project_bytes,
        "project_gib": project_bytes / GIB,
        "project_cap_bytes": project_cap,
        "project_cap_gib": project_cap / GIB,
        "logical_capacity_bytes": logical_capacity,
        "logical_capacity_gib": logical_capacity / GIB,
        "logical_headroom_bytes": logical_headroom,
        "logical_headroom_gib": logical_headroom / GIB,
        "required_headroom_bytes": required_headroom,
        "required_headroom_gib": required_headroom / GIB,
        "capacity_basis": capacity_basis,
        "persistent_mount_point": persistent_mount.mount_point,
        "persistent_fs_type": persistent_mount.fs_type,
        "persistent_source": persistent_mount.source,
        "persistent_observed_total_gib": persistent.total_bytes / GIB,
        "persistent_observed_available_gib": persistent.available_bytes / GIB,
        "shared_fuse_df_caveat": shared_fuse,
        "persistent_df_enforcement": (
            "informational_shared_backend"
            if shared_fuse
            else "locally_observed_free_space_enforced"
        ),
        "overlay_total_gib": overlay.total_bytes / GIB,
        "overlay_available_gib": overlay.available_bytes / GIB,
        "overlay_usage_fraction": overlay_usage,
    }


def inspect_storage(project_root: Path, overlay_path: Path) -> dict[str, object]:
    project_root = project_root.resolve(strict=True)
    overlay_path = overlay_path.resolve(strict=True)
    mount = find_mount(project_root)
    return evaluate_storage(
        project_bytes=measure_project_bytes(project_root),
        persistent=filesystem_stats(project_root),
        persistent_mount=mount,
        overlay=filesystem_stats(overlay_path),
    )


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--overlay-path", type=Path, default=Path("/"))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = inspect_storage(args.project_root, args.overlay_path)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        verdict = "PASS" if result["ok"] else "BLOCK"
        print(
            f"{verdict}: durable={result['project_gib']:.3f} GiB / "
            f"{result['project_cap_gib']:.3f} GiB; "
            f"logical_headroom={result['logical_headroom_gib']:.3f} GiB; "
            f"overlay_available={result['overlay_available_gib']:.3f} GiB; "
            f"overlay_usage={100 * result['overlay_usage_fraction']:.2f}%"
        )
        if result["shared_fuse_df_caveat"]:
            print(
                "- shared FUSE df/statvfs values are informational; "
                "the frozen 50 GiB logical envelope is enforced"
            )
        for violation in result["violations"]:
            print(f"- {violation}")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
