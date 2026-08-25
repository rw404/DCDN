"""Checksum-locked KuaiRec big-matrix acquisition without row inspection.

This module is intentionally limited to source acquisition and provenance.
It downloads only the frozen Zenodo object, verifies the complete archive,
validates every ZIP member before writing anything, extracts into ephemeral
storage, hashes the extracted files, and reads only CSV header records.

It does not filter examples, construct splits, inspect labels, or compute any
dataset/model statistic. Those actions belong to a later frozen protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.request import Request, urlopen

from dcdn.source_manifest import write_source_manifest

SOURCE_URL = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
EXPECTED_BYTES = 431_964_858
EXPECTED_MD5 = "261550d472c48eff4990fb13c0e5bcf7"
EXPECTED_SHA256 = "c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c"

_CACHE = Path(os.environ.get("DCDN_DATA_CACHE", "/tmp"))
DEFAULT_ARCHIVE = _CACHE / "downloads" / "kuairec" / "KuaiRec.zip"
DEFAULT_EXTRACT_ROOT = _CACHE / "raw" / "kuairec" / "source-v1-c23f211c"
DEFAULT_MANIFEST = Path("data/manifests/kuairec_source_v1.yaml")

MAX_MEMBERS = 10_000
MAX_MEMBER_BYTES = 12 * 1024**3
MAX_TOTAL_UNCOMPRESSED_BYTES = 20 * 1024**3
MAX_COMPRESSION_RATIO = 1_000
CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
MAX_CSV_HEADER_BYTES = 256 * 1024
COPY_CHUNK_BYTES = 8 * 1024**2


class KuaiRecSourceError(RuntimeError):
    """Fail-closed source, integrity, or archive-safety error."""


@dataclass(frozen=True)
class FileDigests:
    bytes: int
    md5: str
    sha256: str


@dataclass(frozen=True)
class SafeMember:
    path: str
    kind: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: str


@dataclass(frozen=True)
class ExtractedFile:
    path: str
    bytes: int
    sha256: str
    header_columns: list[str] | None


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_digests(path: Path) -> FileDigests:
    """Stream a local file once while computing size, MD5, and SHA-256."""

    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return FileDigests(bytes=size, md5=md5.hexdigest(), sha256=sha256.hexdigest())


def verify_official_archive(path: Path) -> FileDigests:
    """Verify the frozen official archive without opening its contents."""

    if path.is_symlink() or not path.is_file():
        raise KuaiRecSourceError(f"archive must be a regular file: {path}")
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise KuaiRecSourceError(
            f"archive size mismatch: expected {EXPECTED_BYTES}, observed {size}"
        )
    digests = file_digests(path)
    if digests.md5 != EXPECTED_MD5:
        raise KuaiRecSourceError(
            f"archive MD5 mismatch: expected {EXPECTED_MD5}, observed {digests.md5}"
        )
    if digests.sha256 != EXPECTED_SHA256:
        raise KuaiRecSourceError(
            "archive SHA-256 mismatch: "
            f"expected {EXPECTED_SHA256}, observed {digests.sha256}"
        )
    return digests


def _open_download_output(
    path: Path,
    *,
    existed_before: bool,
    expected_existing_bytes: int,
) -> BinaryIO:
    flags = os.O_WRONLY
    if existed_before:
        if expected_existing_bytes:
            flags |= os.O_APPEND
    else:
        flags |= os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_size != expected_existing_bytes
        ):
            raise KuaiRecSourceError(
                "download destination changed before it could be opened safely"
            )
        return os.fdopen(descriptor, "ab" if expected_existing_bytes else "wb")
    except BaseException:
        os.close(descriptor)
        raise


def download_with_resume(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    opener: Callable[[Request], Any] = urlopen,
) -> None:
    """Download one HTTPS object, appending only a validated byte range."""

    if type(expected_size) is not int or expected_size <= 0:
        raise KuaiRecSourceError("expected download size must be a positive integer")
    if url != SOURCE_URL:
        raise KuaiRecSourceError("only the frozen KuaiRec Zenodo URL is allowed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise KuaiRecSourceError("download destination must not be a symlink")
    if destination.exists() and not destination.is_file():
        raise KuaiRecSourceError("download destination must be a regular file")

    destination_existed = destination.exists()
    if destination_existed:
        metadata = destination.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise KuaiRecSourceError(
                "download destination must be a private regular file"
            )
        existing = metadata.st_size
    else:
        existing = 0
    if existing == expected_size:
        return
    if existing > expected_size:
        raise KuaiRecSourceError("partial archive is larger than the expected object")

    headers = {
        "Accept": "application/octet-stream",
        "Accept-Encoding": "identity",
        "User-Agent": "dcdn-kuairec-source-acquisition/1",
    }
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = Request(url, headers=headers, method="GET")

    try:
        response_context = opener(request)
        with response_context as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            if existing:
                content_range = response.headers.get("Content-Range", "")
                match = CONTENT_RANGE_RE.fullmatch(content_range)
                expected_range = (existing, expected_size - 1, expected_size)
                if (
                    status != 206
                    or match is None
                    or tuple(int(value) for value in match.groups()) != expected_range
                ):
                    raise KuaiRecSourceError(
                        "server did not honor the exact requested resume byte range"
                    )
            else:
                if status != 200:
                    raise KuaiRecSourceError(
                        f"unexpected initial download HTTP status: {status}"
                    )

            remaining = expected_size - existing
            with _open_download_output(
                destination,
                existed_before=destination_existed,
                expected_existing_bytes=existing,
            ) as output:
                while chunk := response.read(COPY_CHUNK_BYTES):
                    if len(chunk) > remaining:
                        raise KuaiRecSourceError(
                            "download response exceeds the expected object size"
                        )
                    output.write(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
    except OSError as exc:
        raise KuaiRecSourceError(f"download failed: {exc}") from exc

    if destination.is_symlink() or not destination.is_file():
        raise KuaiRecSourceError("download destination changed after transfer")
    observed = destination.stat().st_size
    if observed != expected_size:
        raise KuaiRecSourceError(
            f"incomplete download: expected {expected_size}, observed {observed}"
        )


def _normalized_member_path(name: str) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise KuaiRecSourceError(f"unsafe ZIP member name: {name!r}")
    if name.startswith(("/", "//")):
        raise KuaiRecSourceError(f"absolute ZIP member path: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise KuaiRecSourceError(f"traversing ZIP member path: {name!r}")
    parts = tuple(part for part in pure.parts if part not in ("", "."))
    if not parts:
        raise KuaiRecSourceError(f"empty normalized ZIP member path: {name!r}")
    if len(parts[0]) >= 2 and parts[0][1] == ":":
        raise KuaiRecSourceError(f"drive-qualified ZIP member path: {name!r}")
    return "/".join(parts)


def _member_kind(info: zipfile.ZipInfo) -> str:
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if file_type == stat.S_IFLNK:
        raise KuaiRecSourceError(f"symlink ZIP member is forbidden: {info.filename}")
    if info.is_dir():
        if file_type not in (0, stat.S_IFDIR):
            raise KuaiRecSourceError(
                f"directory member has unsafe file type: {info.filename}"
            )
        return "directory"
    if file_type not in (0, stat.S_IFREG):
        raise KuaiRecSourceError(
            f"non-regular ZIP member is forbidden: {info.filename}"
        )
    return "file"


def validate_zip_members(archive: Path) -> list[SafeMember]:
    """Validate the entire central directory before extraction begins."""

    try:
        with zipfile.ZipFile(archive, "r") as bundle:
            infos = bundle.infolist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise KuaiRecSourceError(f"invalid ZIP archive: {exc}") from exc

    if not infos or len(infos) > MAX_MEMBERS:
        raise KuaiRecSourceError(
            f"unexpected ZIP member count: {len(infos)} (limit {MAX_MEMBERS})"
        )

    members: list[SafeMember] = []
    seen: dict[str, str] = {}
    seen_collision_keys: dict[str, str] = {}
    total = 0
    for info in infos:
        path = _normalized_member_path(info.filename)
        kind = _member_kind(info)
        folded = unicodedata.normalize("NFC", path).casefold()
        if path in seen or folded in seen_collision_keys:
            raise KuaiRecSourceError(f"duplicate ZIP target path: {path}")
        if info.flag_bits & 0x1:
            raise KuaiRecSourceError(f"encrypted ZIP member is forbidden: {path}")
        if info.file_size < 0 or info.compress_size < 0:
            raise KuaiRecSourceError(f"invalid ZIP member size: {path}")
        if info.file_size > MAX_MEMBER_BYTES:
            raise KuaiRecSourceError(f"ZIP member exceeds safety limit: {path}")
        if info.file_size and (
            info.compress_size == 0
            or info.file_size > info.compress_size * MAX_COMPRESSION_RATIO
        ):
            raise KuaiRecSourceError(
                f"ZIP member compression ratio exceeds safety limit: {path}"
            )

        for parent in PurePosixPath(path).parents:
            parent_text = parent.as_posix()
            if parent_text == ".":
                continue
            parent_key = unicodedata.normalize("NFC", parent_text).casefold()
            if (
                seen.get(parent_text) == "file"
                or seen_collision_keys.get(parent_key) == "file"
            ):
                raise KuaiRecSourceError(
                    f"ZIP file/member parent collision: {parent_text}"
                )

        seen[path] = kind
        seen_collision_keys[folded] = kind
        total += info.file_size
        if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise KuaiRecSourceError(
                "ZIP total uncompressed size exceeds the 20 GiB safety limit"
            )
        members.append(
            SafeMember(
                path=path,
                kind=kind,
                compressed_bytes=info.compress_size,
                uncompressed_bytes=info.file_size,
                crc32=f"{info.CRC:08x}",
            )
        )

    for path in seen:
        for parent in PurePosixPath(path).parents:
            parent_text = parent.as_posix()
            if parent_text == ".":
                continue
            parent_key = unicodedata.normalize("NFC", parent_text).casefold()
            if (
                seen.get(parent_text) == "file"
                or seen_collision_keys.get(parent_key) == "file"
            ):
                raise KuaiRecSourceError(
                    f"ZIP file/member parent collision: {parent_text}"
                )
    return members


def _iter_file_members(
    bundle: zipfile.ZipFile, safe_members: list[SafeMember]
) -> Iterator[tuple[zipfile.ZipInfo, SafeMember]]:
    infos = bundle.infolist()
    observed = [
        SafeMember(
            path=_normalized_member_path(info.filename),
            kind=_member_kind(info),
            compressed_bytes=info.compress_size,
            uncompressed_bytes=info.file_size,
            crc32=f"{info.CRC:08x}",
        )
        for info in infos
    ]
    if observed != safe_members:
        raise KuaiRecSourceError("ZIP central directory changed after validation")
    for info, safe in zip(infos, safe_members, strict=True):
        if safe.kind == "file":
            yield info, safe


def _open_new_extracted_file(path: Path, *, extraction_root: Path) -> BinaryIO:
    try:
        path.parent.resolve(strict=True).relative_to(extraction_root)
    except ValueError as exc:
        raise KuaiRecSourceError(f"ZIP target parent escapes root: {path}") from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise KuaiRecSourceError("extracted target is not a private regular file")
        return os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise


def extract_verified_zip(
    archive: Path, extraction_root: Path, safe_members: list[SafeMember]
) -> dict[str, str]:
    """Extract regular files after validation and return per-file SHA-256."""

    if extraction_root.is_symlink():
        raise KuaiRecSourceError("extraction root must not be a symlink")
    if extraction_root.exists() and (
        not extraction_root.is_dir() or any(extraction_root.iterdir())
    ):
        raise KuaiRecSourceError("extraction root must be absent or empty")
    extraction_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = extraction_root.parent.resolve(strict=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{extraction_root.name}.extract-", dir=parent)
    )
    staging: Path | None = staging_root
    staging_root.chmod(0o700)
    resolved_root = staging_root.resolve(strict=True)

    hashes: dict[str, str] = {}
    try:
        with zipfile.ZipFile(archive, "r") as bundle:
            for info, member in _iter_file_members(bundle, safe_members):
                target = staging_root.joinpath(*PurePosixPath(member.path).parts)
                resolved_target = target.resolve(strict=False)
                try:
                    resolved_target.relative_to(resolved_root)
                except ValueError as exc:
                    raise KuaiRecSourceError(
                        f"ZIP target escapes extraction root: {member.path}"
                    ) from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                written = 0
                with (
                    bundle.open(info, "r") as source,
                    _open_new_extracted_file(
                        target, extraction_root=resolved_root
                    ) as output,
                ):
                    while chunk := source.read(COPY_CHUNK_BYTES):
                        if len(chunk) > member.uncompressed_bytes - written:
                            raise KuaiRecSourceError(
                                f"ZIP member exceeds declared size: {member.path}"
                            )
                        output.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                if written != member.uncompressed_bytes:
                    raise KuaiRecSourceError(
                        f"extracted size mismatch for {member.path}"
                    )
                hashes[member.path] = digest.hexdigest()
        if extraction_root.exists():
            extraction_root.rmdir()
        os.replace(staging_root, extraction_root)
        staging = None
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise KuaiRecSourceError(f"safe extraction failed: {exc}") from exc
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    return hashes


def read_csv_header(path: Path) -> list[str]:
    """Read and decode exactly the first physical record of a CSV file."""

    if path.is_symlink() or not path.is_file():
        raise KuaiRecSourceError(f"CSV must be a regular non-symlink file: {path}")
    with path.open("rb") as handle:
        raw = handle.readline(MAX_CSV_HEADER_BYTES + 1)
    if len(raw) > MAX_CSV_HEADER_BYTES:
        raise KuaiRecSourceError(f"CSV header exceeds safety limit: {path}")
    if not raw:
        raise KuaiRecSourceError(f"CSV file has no header: {path}")
    try:
        text = raw.decode("utf-8-sig").rstrip("\r\n")
        columns = next(csv.reader([text]))
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise KuaiRecSourceError(f"invalid CSV header in {path}") from exc
    if not columns or any(not column for column in columns):
        raise KuaiRecSourceError(f"CSV header contains an empty column: {path}")
    if len(columns) != len(set(columns)):
        raise KuaiRecSourceError(f"CSV header contains duplicate columns: {path}")
    return columns


def collect_extracted_metadata(
    extraction_root: Path,
    safe_members: list[SafeMember],
    extracted_hashes: dict[str, str],
) -> list[ExtractedFile]:
    """Collect sizes/hashes and CSV schema names, never data-row values."""

    files: list[ExtractedFile] = []
    expected_paths = {member.path for member in safe_members if member.kind == "file"}
    if set(extracted_hashes) != expected_paths:
        raise KuaiRecSourceError(
            "extracted hash set does not match the validated ZIP file set"
        )
    for member in safe_members:
        if member.kind != "file":
            continue
        path = extraction_root.joinpath(*PurePosixPath(member.path).parts)
        if path.is_symlink() or not path.is_file():
            raise KuaiRecSourceError(
                f"extracted member is not a regular non-symlink file: {member.path}"
            )
        size = path.stat().st_size
        if size != member.uncompressed_bytes:
            raise KuaiRecSourceError(f"extracted size drift: {member.path}")
        header = read_csv_header(path) if path.suffix.lower() == ".csv" else None
        files.append(
            ExtractedFile(
                path=member.path,
                bytes=size,
                sha256=extracted_hashes[member.path],
                header_columns=header,
            )
        )
    return files


def build_manifest(
    archive_digests: FileDigests,
    safe_members: list[SafeMember],
    extracted_files: list[ExtractedFile],
    *,
    verified_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_id": "kuairec-big-matrix",
        "artifact_role": "raw_source_provenance_only",
        "acquisition_state": "raw_download_verified",
        "raw_download_verified": True,
        "test_unlocked": False,
        "values_not_scanned": True,
        "test_labels_opened": False,
        "validation_or_test_metrics_read": False,
        "artifact_class": "raw_public_source_manifest",
        "verified_at_utc": verified_at_utc,
        "source": {
            "url": SOURCE_URL,
            "zenodo_record": "18164998",
            "access": "public_unauthenticated_read_only",
            "credentials_used": False,
            "telemetry_used": False,
            "expected_bytes": EXPECTED_BYTES,
            "expected_md5": EXPECTED_MD5,
            "verified_bytes": archive_digests.bytes,
            "verified_md5": archive_digests.md5,
            "verified_sha256": archive_digests.sha256,
        },
        "license_provenance": {
            "zenodo_record_declared": "CC-BY-4.0",
            "upstream_github_declared": "CC-BY-SA-4.0",
            "conservative_terms_applied": "CC-BY-SA-4.0",
            "raw_redistribution": "forbidden_by_project_policy",
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
        "inspection_boundary": {
            "archive_filenames_read": True,
            "member_sizes_and_hashes_computed": True,
            "csv_header_schema_names_read": True,
            "license_and_provenance_metadata_recorded": True,
            "row_values_scanned": False,
            "label_or_outcome_values_scanned": False,
            "readme_results_or_evaluation_opened": False,
            "non_csv_contents_opened_or_interpreted": False,
            "filtering_or_split_construction_performed": False,
            "distribution_or_metric_computed": False,
            "test_unlocked": False,
        },
        "storage": {
            "archive_and_extraction": "ephemeral_cache_only",
            "class": "ephemeral_regenerable_public_raw_data",
            "persistent_raw_copy_created": False,
            "raw_data_committed": False,
            "raw_data_tracked_by_git": False,
        },
        "archive": {
            "format": "zip",
            "member_count": len(safe_members),
            "total_uncompressed_bytes": sum(
                member.uncompressed_bytes for member in safe_members
            ),
            "safety_validation": {
                "zip_slip_rejected": True,
                "absolute_and_drive_paths_rejected": True,
                "backslash_paths_rejected": True,
                "symlinks_and_special_files_rejected": True,
                "encrypted_members_rejected": True,
                "duplicate_targets_rejected": True,
                "member_and_total_size_caps_enforced": True,
                "compression_ratio_cap_enforced": True,
            },
            "members": [asdict(member) for member in safe_members],
        },
        "extracted_files": [asdict(file) for file in extracted_files],
    }


def write_manifest_with_sidecar(path: Path, payload: dict[str, Any]) -> str:
    """Atomically write canonical YAML and its fsynced strict sidecar."""

    return write_source_manifest(path, payload)


def acquire(
    archive: Path,
    extraction_root: Path,
    manifest: Path,
    *,
    offline: bool,
) -> str:
    incomplete = not archive.exists()
    if archive.exists() and archive.is_file():
        incomplete = archive.stat().st_size < EXPECTED_BYTES
    if incomplete:
        if offline:
            raise KuaiRecSourceError(
                f"complete archive is unavailable in offline mode: {archive}"
            )
        download_with_resume(
            SOURCE_URL,
            archive,
            expected_size=EXPECTED_BYTES,
        )
    digests = verify_official_archive(archive)
    members = validate_zip_members(archive)
    extracted_hashes = extract_verified_zip(archive, extraction_root, members)
    extracted = collect_extracted_metadata(extraction_root, members, extracted_hashes)
    payload = build_manifest(
        digests,
        members,
        extracted,
        verified_at_utc=utc_now(),
    )
    return write_manifest_with_sidecar(manifest, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire and inspect KuaiRec source metadata without row access."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extract-root", type=Path, default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="require the checksum-verified archive to exist locally",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        digest = acquire(
            args.archive,
            args.extract_root,
            args.manifest,
            offline=args.offline,
        )
    except KuaiRecSourceError as exc:
        raise SystemExit(f"FAIL: {exc}") from exc
    print(f"PASS: KuaiRec source manifest sha256={digest}; test_unlocked=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
