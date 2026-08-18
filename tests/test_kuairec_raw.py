from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from pathlib import Path
from urllib.request import Request

import pytest
import yaml

from scripts.data.kuairec_raw import (
    EXPECTED_BYTES,
    EXPECTED_MD5,
    EXPECTED_SHA256,
    SOURCE_URL,
    FileDigests,
    KuaiRecSourceError,
    build_manifest,
    collect_extracted_metadata,
    download_with_resume,
    extract_verified_zip,
    read_csv_header,
    validate_zip_members,
    write_manifest_with_sidecar,
)


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, content in members.items():
            bundle.writestr(name, content)


def test_official_source_contract_is_pinned() -> None:
    assert SOURCE_URL == "https://zenodo.org/records/18164998/files/KuaiRec.zip"
    assert EXPECTED_BYTES == 431_964_858
    assert EXPECTED_MD5 == "261550d472c48eff4990fb13c0e5bcf7"
    assert EXPECTED_SHA256 == (
        "c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c"
    )


def test_safe_extract_hash_and_header_only_metadata(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    body = b"a,b,c\n" + b"\xffthis-row-must-not-be-decoded\n"
    write_zip(
        archive,
        {
            "KuaiRec/data/table.csv": body,
            "KuaiRec/LICENSE": b"fixture license",
        },
    )

    members = validate_zip_members(archive)
    output = tmp_path / "raw"
    hashes = extract_verified_zip(archive, output, members)
    files = collect_extracted_metadata(output, members, hashes)

    csv_file = next(file for file in files if file.path.endswith(".csv"))
    assert csv_file.header_columns == ["a", "b", "c"]
    assert csv_file.sha256 == hashlib.sha256(body).hexdigest()
    assert read_csv_header(output / csv_file.path) == ["a", "b", "c"]


def test_extract_rejects_validation_drift_and_cleans_staging(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    write_zip(archive, {"data.csv": b"a\n"})
    output = tmp_path / "raw"
    with pytest.raises(KuaiRecSourceError, match="changed after validation"):
        extract_verified_zip(archive, output, [])
    assert not output.exists()
    assert not list(tmp_path.glob(".raw.extract-*"))


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.csv",
        "/absolute.csv",
        "C:/drive.csv",
        "folder\\windows.csv",
        "folder/../../escape.csv",
    ],
)
def test_unsafe_member_paths_are_rejected(tmp_path: Path, member_name: str) -> None:
    archive = tmp_path / "unsafe.zip"
    write_zip(archive, {member_name: b"a\n1\n"})
    with pytest.raises(KuaiRecSourceError):
        validate_zip_members(archive)


def test_symlink_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(info, b"target")
    with pytest.raises(KuaiRecSourceError, match="symlink"):
        validate_zip_members(archive)


def test_duplicate_casefold_target_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.zip"
    write_zip(archive, {"A.csv": b"a\n", "a.csv": b"a\n"})
    with pytest.raises(KuaiRecSourceError, match="duplicate"):
        validate_zip_members(archive)


def test_file_parent_collision_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "collision.zip"
    write_zip(archive, {"parent": b"x", "parent/child.csv": b"a\n"})
    with pytest.raises(KuaiRecSourceError, match="parent collision"):
        validate_zip_members(archive)


def test_zip_bomb_compression_ratio_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "bomb.zip"
    write_zip(archive, {"zeros.bin": b"\0" * (4 * 1024 * 1024)})
    with pytest.raises(KuaiRecSourceError, match="compression ratio"):
        validate_zip_members(archive)


def test_resume_requires_and_appends_exact_range(tmp_path: Path) -> None:
    destination = tmp_path / "archive.part"
    destination.write_bytes(b"abc")

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 3-5/6"}

        def getcode(self) -> int:
            return self.status

    def opener(request: Request) -> Response:
        assert request.get_header("Range") == "bytes=3-"
        assert request.get_header("Accept-encoding") == "identity"
        return Response(b"def")

    download_with_resume(
        SOURCE_URL,
        destination,
        expected_size=6,
        opener=opener,
    )
    assert destination.read_bytes() == b"abcdef"


def test_zero_byte_existing_destination_starts_without_range(tmp_path: Path) -> None:
    destination = tmp_path / "archive.part"
    destination.write_bytes(b"")

    class Response(io.BytesIO):
        status = 200
        headers: dict[str, str] = {}

        def getcode(self) -> int:
            return self.status

    def opener(request: Request) -> Response:
        assert request.get_header("Range") is None
        return Response(b"abc")

    download_with_resume(
        SOURCE_URL,
        destination,
        expected_size=3,
        opener=opener,
    )
    assert destination.read_bytes() == b"abc"


def test_resume_rejects_wrong_source_and_unhonored_range(tmp_path: Path) -> None:
    destination = tmp_path / "archive.part"
    destination.write_bytes(b"abc")

    class Response(io.BytesIO):
        status = 200
        headers: dict[str, str] = {}

        def getcode(self) -> int:
            return self.status

    with pytest.raises(KuaiRecSourceError, match="only the frozen"):
        download_with_resume(
            "https://example.invalid/KuaiRec.zip",
            destination,
            expected_size=6,
            opener=lambda _request: Response(b"def"),
        )
    with pytest.raises(KuaiRecSourceError, match="did not honor"):
        download_with_resume(
            SOURCE_URL,
            destination,
            expected_size=6,
            opener=lambda _request: Response(b"def"),
        )


def test_manifest_is_locked_and_sidecar_matches(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    write_zip(archive, {"data.csv": b"x,y\n1,2\n"})
    members = validate_zip_members(archive)
    extraction = tmp_path / "raw"
    hashes = extract_verified_zip(archive, extraction, members)
    files = collect_extracted_metadata(extraction, members, hashes)
    digests = FileDigests(bytes=10, md5="a" * 32, sha256="b" * 64)
    payload = build_manifest(
        digests,
        members,
        files,
        verified_at_utc="2026-08-18T00:00:00Z",
    )
    manifest = tmp_path / "kuairec_source_v1.yaml"
    digest = write_manifest_with_sidecar(manifest, payload)

    observed = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest == observed
    assert manifest.with_name(f"{manifest.name}.sha256").read_text() == (
        f"{digest}  {manifest.name}\n"
    )
    loaded = yaml.safe_load(manifest.read_text())
    assert loaded["test_unlocked"] is False
    assert loaded["inspection_boundary"]["row_values_scanned"] is False
    assert loaded["inspection_boundary"]["readme_results_or_evaluation_opened"] is False
    assert loaded["inspection_boundary"]["distribution_or_metric_computed"] is False
