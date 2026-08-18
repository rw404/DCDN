from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/data/vk_lsvd_raw.py"
SPEC = importlib.util.spec_from_file_location("vk_lsvd_raw", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
vk = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = vk
SPEC.loader.exec_module(vk)
MANIFEST = ROOT / "data/manifests/vk_lsvd_source_v1.yaml"


def test_committed_manifest_is_sidecar_verified_and_exact() -> None:
    manifest = vk.load_manifest(MANIFEST)
    assert manifest.revision == "084a18386e149a454dd6dbe0123478e75c6b09ce"
    assert manifest.test_unlocked is False
    assert manifest.week26_sealed is True
    assert len(manifest.objects) == 30
    assert manifest.total_bytes == 2_843_350_925
    assert len(manifest.unsealed_objects()) == 29
    assert {obj.content_access for obj in manifest.unsealed_objects()} == {
        "not_decoded"
    }


def test_interaction_weeks_and_sealed_test_are_exact() -> None:
    manifest = vk.load_manifest(MANIFEST)
    interactions = {obj.week: obj for obj in manifest.objects if obj.week is not None}
    assert set(interactions) == set(range(27))
    for week in range(25):
        assert interactions[week].split == "train"
        assert not interactions[week].sealed
    assert interactions[25].split == "validation"
    assert not interactions[25].sealed
    sealed = interactions[26]
    assert sealed.split == "test"
    assert sealed.sealed
    assert sealed not in manifest.unsealed_objects()
    assert sealed.size_bytes == 8_330_983
    assert sealed.sha256 == (
        "e540f80917fa7e065a9905a47611955820dddb8fcd7f45385ccb53d6a961fdb7"
    )


def test_metadata_allowlist_is_exact() -> None:
    manifest = vk.load_manifest(MANIFEST)
    metadata = {
        obj.path: (obj.size_bytes, obj.sha256)
        for obj in manifest.objects
        if obj.week is None
    }
    assert metadata == {
        "metadata/item_embeddings.npz": (
            2_362_338_243,
            "6a4024ad7623f5d8150e59c5faaaf42fb7d7f48e2beec44c6917e5e57debea7c",
        ),
        "metadata/items_metadata.parquet": (
            155_747_843,
            "607ffd09d3fb6627f7eec07ed53583d45c47263a2e6538e4deee66cd8a22e51e",
        ),
        "metadata/users_metadata.parquet": (
            72_526_148,
            "51fa9528f71bb0da947311d7baa4727d325285a3e748b54816477b82492aff20",
        ),
    }


def test_resolver_url_is_official_and_revision_pinned() -> None:
    manifest = vk.load_manifest(MANIFEST)
    obj = manifest.objects[0]
    url = vk.resolver_url(manifest, obj)
    assert url.startswith(
        "https://huggingface.co/datasets/deepvk/VK-LSVD/resolve/"
        "084a18386e149a454dd6dbe0123478e75c6b09ce/"
    )
    assert "/resolve/main/" not in url


def test_remote_tree_validator_rejects_any_metadata_drift() -> None:
    manifest = vk.load_manifest(MANIFEST)
    entries = []
    for obj in manifest.objects:
        entries.append(
            {
                "type": "file",
                "path": obj.path,
                "size": obj.size_bytes,
                "oid": obj.git_blob_oid,
                "xetHash": obj.xet_hash,
                "lfs": {"oid": obj.sha256, "size": obj.size_bytes},
                "lastCommit": {"id": obj.last_commit},
            }
        )
    vk._validate_remote_entries(manifest, entries)
    entries[0]["size"] += 1
    with pytest.raises(vk.AcquisitionError, match="remote size mismatch"):
        vk._validate_remote_entries(manifest, entries)


def test_verify_object_hashes_bytes_without_decoding(tmp_path: Path) -> None:
    payload = b"not-a-parquet-decoder-test\x00\xff"
    path = tmp_path / "opaque.bin"
    path.write_bytes(payload)
    exemplar = vk.load_manifest(MANIFEST).objects[0]
    obj = replace(
        exemplar,
        path="opaque.bin",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        etag=hashlib.sha256(payload).hexdigest(),
    )
    assert vk.verify_object(path, obj) == obj.sha256
    path.write_bytes(payload + b"drift")
    with pytest.raises(vk.AcquisitionError, match="size mismatch"):
        vk.verify_object(path, obj)
    symlink = tmp_path / "opaque-link.bin"
    symlink.symlink_to(path)
    with pytest.raises(vk.AcquisitionError, match="non-symlink"):
        vk.verify_object(symlink, obj)


def test_destination_rejects_traversal_even_for_constructed_object(
    tmp_path: Path,
) -> None:
    exemplar = vk.load_manifest(MANIFEST).objects[0]
    unsafe = replace(exemplar, path="../escape")
    with pytest.raises(vk.AcquisitionError, match="escapes cache root"):
        vk.destination(tmp_path, unsafe)


def test_credential_environment_removes_hf_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in vk.TOKEN_ENV_VARS:
        monkeypatch.setenv(name, "must-not-propagate")
    env = vk._credential_free_environment()
    for name in vk.TOKEN_ENV_VARS:
        assert name not in env
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
