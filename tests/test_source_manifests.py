from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dcdn.source_manifest import (
    SourceManifestError,
    load_source_manifest,
    write_source_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "kuairand_pure": ROOT / "data/manifests/kuairand_pure_source_v1.yaml",
    "kuairec-big-matrix": ROOT / "data/manifests/kuairec_source_v1.yaml",
    "deepvk/VK-LSVD": ROOT / "data/manifests/vk_lsvd_source_v1.yaml",
}


def _minimal_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": "fixture/source",
        "artifact_role": "raw_source_provenance_only",
        "acquisition_state": "raw_download_verified",
        "raw_download_verified": True,
        "test_unlocked": False,
        "values_not_scanned": True,
        "test_labels_opened": False,
        "validation_or_test_metrics_read": False,
        "source": {
            "url": "https://example.invalid/frozen",
            "access": "public_unauthenticated_read_only",
            "credentials_used": False,
            "telemetry_used": False,
        },
        "storage": {
            "class": "ephemeral_regenerable_public_raw_data",
            "persistent_raw_copy_created": False,
            "raw_data_committed": False,
        },
        "inspection": {
            "archive_or_repository_metadata_read": True,
            "byte_hashes_computed": True,
            "csv_headers_read": False,
            "first_line_headers_only": False,
            "parquet_or_numpy_decoded": False,
            "row_values_scanned": False,
            "test_labels_opened": False,
            "validation_or_test_metrics_read": False,
        },
    }


def test_all_persisted_manifests_share_the_locked_source_contract() -> None:
    loaded = {
        dataset_id: load_source_manifest(path) for dataset_id, path in MANIFESTS.items()
    }
    assert set(loaded) == set(MANIFESTS)
    for expected_dataset_id, payload in loaded.items():
        assert payload["dataset_id"] == expected_dataset_id
        assert payload["artifact_role"] == "raw_source_provenance_only"
        assert payload["acquisition_state"] == "raw_download_verified"
        assert payload["raw_download_verified"] is True
        assert payload["test_unlocked"] is False
        assert payload["values_not_scanned"] is True
        assert payload["test_labels_opened"] is False
        assert payload["validation_or_test_metrics_read"] is False
        assert payload["inspection"]["row_values_scanned"] is False
        assert payload["inspection"]["parquet_or_numpy_decoded"] is False


def test_persisted_release_hashes_and_vk_seal_remain_exact() -> None:
    kuairand = load_source_manifest(MANIFESTS["kuairand_pure"])
    kuairec = load_source_manifest(MANIFESTS["kuairec-big-matrix"])
    vk = load_source_manifest(MANIFESTS["deepvk/VK-LSVD"])
    assert kuairand["source"]["sha256"] == (
        "c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0"
    )
    assert kuairec["source"]["verified_sha256"] == (
        "c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c"
    )
    week26 = next(obj for obj in vk["objects"] if obj.get("week") == 26)
    assert vk["week26_sealed"] is True
    assert week26["content_access"] == "sealed_hash_only"
    assert week26["sha256"] == (
        "e540f80917fa7e065a9905a47611955820dddb8fcd7f45385ccb53d6a961fdb7"
    )
    assert all(
        obj["content_access"] == "not_decoded"
        for obj in vk["objects"]
        if obj.get("week") != 26
    )


def test_atomic_writer_round_trip_has_exact_sidecar_and_no_temp(tmp_path: Path) -> None:
    path = tmp_path / "source.yaml"
    digest = write_source_manifest(path, _minimal_payload())
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.with_name(f"{path.name}.sha256").read_bytes() == (
        f"{digest}  {path.name}\n".encode()
    )
    assert load_source_manifest(path)["dataset_id"] == "fixture/source"
    assert not list(tmp_path.glob(".*.manifest-*"))
    assert not list(tmp_path.glob(".*.sidecar-*"))


def test_writer_rejects_integer_as_security_boolean(tmp_path: Path) -> None:
    payload = _minimal_payload()
    source = payload["source"]
    assert isinstance(source, dict)
    source["credentials_used"] = 0
    with pytest.raises(SourceManifestError, match="credentials_used"):
        write_source_manifest(tmp_path / "source.yaml", payload)


def test_loader_rejects_noncanonical_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "source.yaml"
    digest = write_source_manifest(path, _minimal_payload())
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest} *{path.name}\n", encoding="ascii"
    )
    with pytest.raises(SourceManifestError, match="sidecar mismatch"):
        load_source_manifest(path)


def test_loader_rejects_duplicate_yaml_keys_even_with_matching_hash(
    tmp_path: Path,
) -> None:
    valid = write_source_manifest(tmp_path / "valid.yaml", _minimal_payload())
    assert valid
    source = (tmp_path / "valid.yaml").read_text(encoding="utf-8")
    duplicated = source.replace(
        "dataset_id: fixture/source\n",
        "dataset_id: fixture/source\ndataset_id: conflicting/source\n",
    ).encode()
    path = tmp_path / "duplicate.yaml"
    path.write_bytes(duplicated)
    digest = hashlib.sha256(duplicated).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    with pytest.raises(SourceManifestError, match="duplicate key"):
        load_source_manifest(path)
