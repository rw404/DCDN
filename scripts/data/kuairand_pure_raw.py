#!/usr/bin/env python3
"""Verify and safely materialize the frozen KuaiRand-Pure raw archive.

This command deliberately performs no semantic row scan.  It reads archive
metadata, streams bytes for integrity checks/extraction, and reads exactly the
first line of each CSV to record column names.  It never evaluates labels,
dates, distributions, validation metrics, or test metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from dcdn.source_manifest import write_source_manifest

SOURCE_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
ARCHIVE_NAME = "KuaiRand-Pure.tar.gz"
EXPECTED_BYTES = 47_432_272
EXPECTED_MD5 = "0820331067a3784d9691136f772b35a7"
EXPECTED_SHA256 = "c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0"
TOP_LEVEL = "KuaiRand-Pure"
MAX_MEMBERS = 1_000
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_HEADER_BYTES = 1024 * 1024

DEFAULT_DOWNLOAD_DIR = Path("/home/codex/.cache/dcdn-public/downloads/kuairand_pure")
DEFAULT_ARCHIVE = DEFAULT_DOWNLOAD_DIR / ARCHIVE_NAME
DEFAULT_EXTRACT_ROOT = Path("/home/codex/.cache/dcdn-public/raw/kuairand_pure")

EXPECTED_MEMBER_SIZES: dict[str, int] = {
    "KuaiRand-Pure": 0,
    "KuaiRand-Pure/LICENSE": 20_138,
    "KuaiRand-Pure/data": 0,
    "KuaiRand-Pure/load_data_pure.py": 1_608,
    "KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv": 83_961_282,
    "KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv": 87_086_116,
    "KuaiRand-Pure/data/user_features_pure.csv": 3_519_028,
    "KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv": 21_765_075,
    "KuaiRand-Pure/data/video_features_statistic_pure.csv": 6_559_217,
    "KuaiRand-Pure/data/video_features_basic_pure.csv": 626_669,
}


class RawArchiveError(RuntimeError):
    """Raised when acquisition cannot satisfy the frozen safety contract."""


@dataclass(frozen=True)
class ArchiveVerification:
    bytes: int
    md5: str
    sha256: str


@dataclass(frozen=True)
class MemberRecord:
    path: str
    kind: str
    bytes: int


@dataclass(frozen=True)
class ExtractedFileRecord:
    path: str
    bytes: int
    md5: str
    sha256: str


def _stream_hashes(handle: BinaryIO) -> tuple[int, str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    total = 0
    while chunk := handle.read(8 * 1024 * 1024):
        total += len(chunk)
        md5.update(chunk)
        sha256.update(chunk)
    return total, md5.hexdigest(), sha256.hexdigest()


def _open_regular_read(path: Path) -> BinaryIO:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RawArchiveError(f"cannot safely open regular file: {path}") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise RawArchiveError(f"expected a private regular non-symlink file: {path}")
    return os.fdopen(descriptor, "rb")


def file_hashes(path: Path) -> ArchiveVerification:
    """Return byte count plus MD5/SHA-256 without decoding file contents."""

    with _open_regular_read(path) as handle:
        size, md5, sha256 = _stream_hashes(handle)
    return ArchiveVerification(bytes=size, md5=md5, sha256=sha256)


def verify_archive(path: Path) -> ArchiveVerification:
    """Fail closed unless the archive exactly matches the frozen release."""

    observed = file_hashes(path)
    expected = ArchiveVerification(
        bytes=EXPECTED_BYTES,
        md5=EXPECTED_MD5,
        sha256=EXPECTED_SHA256,
    )
    if observed != expected:
        raise RawArchiveError(
            "archive integrity mismatch: "
            f"expected={asdict(expected)} observed={asdict(observed)}"
        )
    return observed


def _safe_member_path(name: str, *, top_level: str) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise RawArchiveError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RawArchiveError(f"unsafe archive member path: {name!r}")
    if not path.parts or path.parts[0] != top_level:
        raise RawArchiveError(f"member escapes frozen top-level directory: {name!r}")
    return path


def audit_archive(
    path: Path,
    *,
    top_level: str = TOP_LEVEL,
    expected_member_sizes: dict[str, int] | None = EXPECTED_MEMBER_SIZES,
) -> list[MemberRecord]:
    """Read tar headers only and reject traversal, links, and special files."""

    records: list[MemberRecord] = []
    seen: dict[str, str] = {}
    seen_collision_keys: dict[str, str] = {}
    total_regular_bytes = 0
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > MAX_MEMBERS:
                raise RawArchiveError("archive member count exceeds safety limit")
            for member in members:
                safe_path = _safe_member_path(member.name, top_level=top_level)
                normalized = safe_path.as_posix()
                collision_key = unicodedata.normalize("NFC", normalized).casefold()
                if normalized in seen or collision_key in seen_collision_keys:
                    raise RawArchiveError(
                        f"duplicate or case-colliding archive member: {normalized}"
                    )

                if member.issym() or member.islnk():
                    raise RawArchiveError(f"links are forbidden: {normalized}")
                if member.isdir():
                    kind = "directory"
                elif member.isfile():
                    kind = "regular_file"
                    if member.size < 0:
                        raise RawArchiveError(f"negative member size: {normalized}")
                    total_regular_bytes += member.size
                else:
                    raise RawArchiveError(f"special member is forbidden: {normalized}")

                for parent in safe_path.parents:
                    parent_text = parent.as_posix()
                    if parent_text == ".":
                        continue
                    parent_key = unicodedata.normalize("NFC", parent_text).casefold()
                    if (
                        seen.get(parent_text) == "regular_file"
                        or seen_collision_keys.get(parent_key) == "regular_file"
                    ):
                        raise RawArchiveError(
                            f"file/member parent collision: {parent_text}"
                        )

                seen[normalized] = kind
                seen_collision_keys[collision_key] = kind

                if total_regular_bytes > MAX_UNCOMPRESSED_BYTES:
                    raise RawArchiveError("uncompressed archive exceeds safety limit")
                records.append(
                    MemberRecord(path=normalized, kind=kind, bytes=member.size)
                )
    except (tarfile.TarError, OSError) as error:
        raise RawArchiveError(f"cannot audit archive: {error}") from error

    for normalized in seen:
        for parent in PurePosixPath(normalized).parents:
            parent_text = parent.as_posix()
            if parent_text == ".":
                continue
            parent_key = unicodedata.normalize("NFC", parent_text).casefold()
            if (
                seen.get(parent_text) == "regular_file"
                or seen_collision_keys.get(parent_key) == "regular_file"
            ):
                raise RawArchiveError(f"file/member parent collision: {parent_text}")

    if expected_member_sizes is not None:
        observed = {record.path: record.bytes for record in records}
        if observed != expected_member_sizes:
            raise RawArchiveError(
                "archive member set/size mismatch: "
                f"expected={expected_member_sizes!r} observed={observed!r}"
            )
    return records


def _contained_target(root: Path, relative: PurePosixPath) -> Path:
    target = root.joinpath(*relative.parts)
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise RawArchiveError(f"extraction target escapes root: {relative}") from error
    return target


def _copy_member(
    source: BinaryIO, target: Path, expected_bytes: int, *, root: Path
) -> None:
    try:
        target.parent.resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise RawArchiveError(f"extraction parent escapes root: {target}") from error
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if target.stat().st_size != expected_bytes:
        raise RawArchiveError(f"extracted size mismatch: {target}")


def safe_extract(
    archive_path: Path,
    extract_root: Path,
    records: list[MemberRecord],
) -> None:
    """Extract audited regular files into an empty private directory."""

    if extract_root.is_symlink():
        raise RawArchiveError("extraction root must not be a symlink")
    if extract_root.exists() and (
        not extract_root.is_dir() or any(extract_root.iterdir())
    ):
        raise RawArchiveError(
            f"extraction root must be empty or absent: {extract_root}"
        )
    extract_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = extract_root.parent.resolve(strict=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{extract_root.name}.extract-", dir=parent)
    )
    staging.chmod(0o700)
    root = staging.resolve(strict=True)

    expected = {record.path: record for record in records}
    extracted: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                normalized = _safe_member_path(
                    member.name,
                    top_level=TOP_LEVEL,
                ).as_posix()
                if normalized in extracted:
                    raise RawArchiveError(f"duplicate member after audit: {normalized}")
                record = expected.get(normalized)
                if record is None:
                    raise RawArchiveError(
                        f"archive changed after audit: {member.name!r}"
                    )
                extracted.add(normalized)
                observed_kind = (
                    "directory"
                    if member.isdir()
                    else "regular_file"
                    if member.isfile()
                    else "forbidden"
                )
                if (
                    member.issym()
                    or member.islnk()
                    or observed_kind != record.kind
                    or member.size != record.bytes
                ):
                    raise RawArchiveError(
                        f"member type/size changed after audit: {normalized}"
                    )
                relative = PurePosixPath(record.path)
                target = _contained_target(root, relative)
                if record.kind == "directory":
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = archive.extractfile(member)
                if source is None:
                    raise RawArchiveError(f"cannot stream member: {member.name}")
                with source:
                    _copy_member(source, target, record.bytes, root=root)
            if extracted != set(expected):
                raise RawArchiveError("archive members disappeared after audit")
        if extract_root.exists():
            extract_root.rmdir()
        os.replace(staging, extract_root)
        staging = None
    except (tarfile.TarError, OSError) as error:
        raise RawArchiveError(f"safe extraction failed: {error}") from error
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def read_csv_header(path: Path) -> list[str]:
    """Read exactly one bounded line and return CSV column names only."""

    with _open_regular_read(path) as handle:
        line = handle.readline(MAX_HEADER_BYTES + 1)
    if not line or len(line) > MAX_HEADER_BYTES:
        raise RawArchiveError(f"missing or oversized CSV header: {path}")
    try:
        header = next(csv.reader([line.decode("utf-8-sig")]))
    except (UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise RawArchiveError(f"invalid CSV header: {path}") from error
    if not header or any(not column for column in header):
        raise RawArchiveError(f"empty CSV column name: {path}")
    if len(header) != len(set(header)):
        raise RawArchiveError(f"duplicate CSV column name: {path}")
    return header


def inspect_extracted_files(
    extract_root: Path,
    records: list[MemberRecord],
) -> tuple[list[ExtractedFileRecord], dict[str, list[str]]]:
    """Hash regular members and inspect only the first line of CSV members."""

    root = extract_root.resolve(strict=True)
    files: list[ExtractedFileRecord] = []
    schemas: dict[str, list[str]] = {}
    for record in records:
        if record.kind != "regular_file":
            continue
        path = _contained_target(root, PurePosixPath(record.path))
        observed = file_hashes(path)
        if observed.bytes != record.bytes:
            raise RawArchiveError(
                f"member size changed after extraction: {record.path}"
            )
        files.append(
            ExtractedFileRecord(
                path=record.path,
                bytes=observed.bytes,
                md5=observed.md5,
                sha256=observed.sha256,
            )
        )
        if path.suffix.lower() == ".csv":
            schemas[record.path] = read_csv_header(path)
    return files, schemas


def build_manifest(
    *,
    archive_path: Path,
    extract_root: Path,
    verification: ArchiveVerification,
    records: list[MemberRecord],
    files: list[ExtractedFileRecord],
    schemas: dict[str, list[str]],
    verified_at_utc: datetime,
) -> dict[str, object]:
    """Build the source-only manifest without any row-derived statistics."""

    if verified_at_utc.tzinfo is None:
        raise ValueError("verified_at_utc must be timezone-aware")
    return {
        "schema_version": 1,
        "dataset_id": "kuairand_pure",
        "artifact_role": "raw_source_provenance_only",
        "acquisition_state": "raw_download_verified",
        "raw_download_verified": True,
        "test_unlocked": False,
        "values_not_scanned": True,
        "test_labels_opened": False,
        "validation_or_test_metrics_read": False,
        "verified_at_utc": verified_at_utc.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": {
            "url": SOURCE_URL,
            "zenodo_record": 10_439_422,
            "access": "public_unauthenticated_read_only",
            "credentials_used": False,
            "telemetry_used": False,
            "archive_name": ARCHIVE_NAME,
            "archive_path_ephemeral": str(archive_path.resolve()),
            "expected_bytes": EXPECTED_BYTES,
            "observed_bytes": verification.bytes,
            "expected_md5": EXPECTED_MD5,
            "observed_md5": verification.md5,
            "sha256": verification.sha256,
        },
        "license_provenance": {
            "zenodo_declared": "CC-BY-4.0",
            "upstream_repository_declared": "CC-BY-SA-4.0",
            "conservative_terms_applied": "CC-BY-SA-4.0",
            "license_file_content_read": False,
            "license_file_hash_recorded": True,
        },
        "storage": {
            "class": "ephemeral_regenerable_public_raw_data",
            "extract_root": str(extract_root.resolve()),
            "persistent_raw_copy_created": False,
            "raw_data_committed": False,
        },
        "inspection": {
            "archive_or_repository_metadata_read": True,
            "byte_hashes_computed": True,
            "csv_headers_read": True,
            "first_line_headers_only": True,
            "parquet_or_numpy_decoded": False,
            "row_values_scanned": False,
            "test_labels_opened": False,
            "validation_or_test_metrics_read": False,
        },
        "archive_safety": {
            "member_count": len(records),
            "regular_file_bytes": sum(
                record.bytes for record in records if record.kind == "regular_file"
            ),
            "path_traversal_detected": False,
            "absolute_path_detected": False,
            "symlink_detected": False,
            "hardlink_detected": False,
            "special_member_detected": False,
            "extraction_policy": "audited_regular_files_and_directories_only",
            "archive_members": [asdict(record) for record in records],
        },
        "extracted_files": [asdict(record) for record in files],
        "csv_schemas_header_only": schemas,
        "inspection_boundary": {
            "allowed_and_performed": [
                "archive_header_names_sizes_and_types",
                "bytewise_archive_and_file_hashing",
                "safe_raw_byte_extraction",
                "csv_first_line_column_names_only",
                "license_and_source_provenance_metadata",
            ],
            "not_performed": [
                "csv_row_value_scan",
                "date_or_split_coverage_scan",
                "label_value_or_distribution_scan",
                "is_rand_counts_or_distribution_scan",
                "README_or_results_read",
                "loader_source_read",
                "validation_metric_read",
                "test_label_read",
                "test_metric_read",
            ],
        },
        "strict_causal_exclusions": {
            "KuaiRand-Pure/data/video_features_statistic_pure.csv": (
                "excluded because monthly statistics may overlap target outcomes; "
                "only hash and header metadata were recorded"
            ),
            "is_rand": "stratification_only; forbidden as a model feature",
            "current_row_outcomes": "labels_only; forbidden as model inputs",
        },
        "next_gate": (
            "feature/split manifests remain unresolved; test_unlocked stays false"
        ),
    }


def write_manifest(path: Path, payload: dict[str, object]) -> str:
    """Atomically write YAML and its fsynced deterministic sidecar."""

    return write_source_manifest(path, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extract-root", type=Path, default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--extract",
        action="store_true",
        help="materialize into an existing-empty or new ephemeral directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    verification = verify_archive(args.archive)
    records = audit_archive(args.archive)
    if args.extract:
        safe_extract(args.archive, args.extract_root, records)
        # Detect any archive replacement during extraction.
        if verify_archive(args.archive) != verification:
            raise RawArchiveError("archive changed during extraction")
    files, schemas = inspect_extracted_files(args.extract_root, records)
    payload = build_manifest(
        archive_path=args.archive,
        extract_root=args.extract_root,
        verification=verification,
        records=records,
        files=files,
        schemas=schemas,
        verified_at_utc=datetime.now(UTC),
    )
    digest = write_manifest(args.manifest_output, payload)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest_output),
                "manifest_sha256": digest,
                "raw_download_verified": True,
                "test_unlocked": False,
                "values_not_scanned": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
