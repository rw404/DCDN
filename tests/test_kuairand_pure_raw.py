from __future__ import annotations

import hashlib
import io
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from scripts.data.kuairand_pure_raw import (
    EXPECTED_BYTES,
    EXPECTED_MD5,
    EXPECTED_SHA256,
    ArchiveVerification,
    ExtractedFileRecord,
    MemberRecord,
    RawArchiveError,
    audit_archive,
    build_manifest,
    read_csv_header,
    safe_extract,
    verify_archive,
    write_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_tar(
    path: Path,
    entries: list[tuple[str, bytes | None, bytes | None]],
) -> None:
    """Write (name, payload, link type) entries for security tests."""

    with tarfile.open(path, "w:gz") as archive:
        for name, payload, link_type in entries:
            member = tarfile.TarInfo(name)
            if link_type is not None:
                member.type = link_type
                member.linkname = "Root/target"
                archive.addfile(member)
            elif payload is None:
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            else:
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))


def _audit_test_tar(path: Path) -> list[MemberRecord]:
    return audit_archive(
        path,
        top_level="Root",
        expected_member_sizes=None,
    )


def test_frozen_release_constants_are_exact() -> None:
    assert EXPECTED_BYTES == 47_432_272
    assert EXPECTED_MD5 == "0820331067a3784d9691136f772b35a7"
    assert EXPECTED_SHA256 == (
        "c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0"
    )


def test_verify_archive_fails_closed_on_unapproved_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "wrong.tar.gz"
    archive.write_bytes(b"not the frozen archive")
    with pytest.raises(RawArchiveError, match="integrity mismatch"):
        verify_archive(archive)


@pytest.mark.parametrize(
    "unsafe_name",
    ["../escape", "/Root/absolute", "Root/../escape", "Other/file.csv"],
)
def test_archive_audit_rejects_path_escape(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _write_tar(archive, [(unsafe_name, b"x", None)])
    with pytest.raises(RawArchiveError):
        _audit_test_tar(archive)


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_archive_audit_rejects_links(
    tmp_path: Path,
    link_type: bytes,
) -> None:
    archive = tmp_path / "link.tar.gz"
    _write_tar(archive, [("Root/link", None, link_type)])
    with pytest.raises(RawArchiveError, match="links are forbidden"):
        _audit_test_tar(archive)


def test_archive_audit_rejects_duplicate_member(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.tar.gz"
    _write_tar(
        archive,
        [("Root/data.csv", b"a\n", None), ("Root/data.csv", b"b\n", None)],
    )
    with pytest.raises(RawArchiveError, match="duplicate or case-colliding"):
        _audit_test_tar(archive)


def test_archive_audit_rejects_unicode_case_collision(tmp_path: Path) -> None:
    archive = tmp_path / "case-collision.tar.gz"
    _write_tar(
        archive,
        [
            ("Root/Caf\N{LATIN SMALL LETTER E WITH ACUTE}.csv", b"a\n", None),
            ("Root/Cafe\N{COMBINING ACUTE ACCENT}.csv", b"b\n", None),
        ],
    )
    with pytest.raises(RawArchiveError, match="case-colliding"):
        _audit_test_tar(archive)


def test_archive_audit_rejects_reverse_file_parent_collision(tmp_path: Path) -> None:
    archive = tmp_path / "parent-collision.tar.gz"
    _write_tar(
        archive,
        [("Root/parent/child.csv", b"a\n", None), ("Root/PARENT", b"x", None)],
    )
    with pytest.raises(RawArchiveError, match="parent collision"):
        _audit_test_tar(archive)


def test_safe_extract_materializes_only_audited_regular_files(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "safe.tar.gz"
    _write_tar(
        archive,
        [
            ("KuaiRand-Pure", None, None),
            ("KuaiRand-Pure/data", None, None),
            ("KuaiRand-Pure/data/schema.csv", b"left,right\n1,2\n", None),
        ],
    )
    records = audit_archive(archive, expected_member_sizes=None)
    extract_root = tmp_path / "raw"
    safe_extract(archive, extract_root, records)
    assert (extract_root / "KuaiRand-Pure/data/schema.csv").read_bytes() == (
        b"left,right\n1,2\n"
    )

    with pytest.raises(RawArchiveError, match="must be empty"):
        safe_extract(archive, extract_root, records)


def test_csv_header_reader_does_not_decode_second_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "schema.csv"
    csv_path.write_bytes(b"alpha,beta\n\xff\xfe-secret-row")
    assert read_csv_header(csv_path) == ["alpha", "beta"]


def test_manifest_keeps_test_lock_and_no_value_scan(tmp_path: Path) -> None:
    verification = ArchiveVerification(
        bytes=EXPECTED_BYTES,
        md5=EXPECTED_MD5,
        sha256=EXPECTED_SHA256,
    )
    members = [MemberRecord("KuaiRand-Pure", "directory", 0)]
    files = [
        ExtractedFileRecord(
            path="KuaiRand-Pure/data/log.csv",
            bytes=10,
            md5="0" * 32,
            sha256="1" * 64,
        )
    ]
    payload = build_manifest(
        archive_path=tmp_path / "archive.tar.gz",
        extract_root=tmp_path / "raw",
        verification=verification,
        records=members,
        files=files,
        schemas={"KuaiRand-Pure/data/log.csv": ["a", "b"]},
        verified_at_utc=datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
    )

    assert payload["acquisition_state"] == "raw_download_verified"
    assert payload["raw_download_verified"] is True
    assert payload["test_unlocked"] is False
    assert payload["values_not_scanned"] is True
    assert payload["test_labels_opened"] is False
    assert payload["validation_or_test_metrics_read"] is False
    assert "csv_row_value_scan" in payload["inspection_boundary"]["not_performed"]


def test_manifest_sidecar_hashes_exact_yaml_bytes(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    payload = build_manifest(
        archive_path=tmp_path / "archive.tar.gz",
        extract_root=tmp_path / "raw",
        verification=ArchiveVerification(
            bytes=EXPECTED_BYTES,
            md5=EXPECTED_MD5,
            sha256=EXPECTED_SHA256,
        ),
        records=[],
        files=[],
        schemas={},
        verified_at_utc=datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
    )
    digest = write_manifest(path, payload)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.with_name("manifest.yaml.sha256").read_text() == (
        f"{digest}  manifest.yaml\n"
    )
    assert yaml.safe_load(path.read_text()) == payload


def test_persisted_source_manifest_is_locked_and_hash_verified() -> None:
    path = PROJECT_ROOT / "data/manifests/kuairand_pure_source_v1.yaml"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(f"{path.name}.sha256").read_text().split()
    payload = yaml.safe_load(path.read_text())

    assert sidecar == [digest, path.name]
    assert payload["acquisition_state"] == "raw_download_verified"
    assert payload["raw_download_verified"] is True
    assert payload["test_unlocked"] is False
    assert payload["values_not_scanned"] is True
    assert payload["test_labels_opened"] is False
    assert payload["source"]["sha256"] == EXPECTED_SHA256
