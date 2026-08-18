from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from dcdn.protocol import (
    ProtocolError,
    ProtocolGate,
    load_verified_yaml,
    sha256_file,
    validate_protocol,
    verify_sha256_sidecar,
)

DIGEST = "a" * 64


def protocol_payload(*, unlocked: bool = False) -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "synthetic-no-test-data",
        "test_unlocked": unlocked,
        "hashes": {
            "data_manifest_sha256": DIGEST,
            "features_sha256": "b" * 64,
            "split_sha256": "c" * 64,
            "search_space_sha256": "d" * 64,
            "experiment_matrix_sha256": "e" * 64,
            "selected_configs_sha256": "f" * 64 if unlocked else None,
        },
        "final_seeds": [42, 43, 44],
        "tuning_seed": 42,
        "tuning_splits": ["train", "validation"],
        "test_evaluations_per_final_seed": 1,
    }


def write_hashed(path: Path, payload: dict | str) -> str:
    text = payload if isinstance(payload, str) else yaml.safe_dump(payload)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def test_sha256_sidecar_round_trip_and_tamper(tmp_path: Path) -> None:
    path = tmp_path / "artifact.yaml"
    digest = write_hashed(path, {"ok": True})
    assert sha256_file(path) == digest
    assert verify_sha256_sidecar(path) == digest
    assert load_verified_yaml(path) == {"ok": True}

    path.write_text("ok: false\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="SHA-256 mismatch"):
        verify_sha256_sidecar(path)


def test_invalid_protocol_fields_fail_closed() -> None:
    valid = protocol_payload()
    validate_protocol(valid)

    mutations = [
        ("schema_version", 2),
        ("dataset_id", ""),
        ("test_unlocked", "false"),
        ("final_seeds", [42]),
        ("tuning_seed", 43),
        ("tuning_splits", ["train", "validation", "test"]),
        ("test_evaluations_per_final_seed", 2),
    ]
    for key, value in mutations:
        candidate = deepcopy(valid)
        candidate[key] = value
        with pytest.raises(ProtocolError):
            validate_protocol(candidate)

    candidate = deepcopy(valid)
    candidate["hashes"]["features_sha256"] = "UNRESOLVED"
    with pytest.raises(ProtocolError, match="features_sha256"):
        validate_protocol(candidate)


def test_locked_protocol_allows_train_validation_but_rejects_test(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "protocol_frozen.yaml"
    write_hashed(protocol_path, protocol_payload(unlocked=False))
    gate = ProtocolGate.load(protocol_path)

    gate.require_split("train")
    gate.require_split("validation")
    with pytest.raises(ProtocolError, match="test access is locked"):
        gate.require_split("test")
    with pytest.raises(ProtocolError, match="unknown split"):
        gate.require_split("holdout")


def test_unlocked_protocol_requires_matching_selected_config_lock(
    tmp_path: Path,
) -> None:
    selected_path = tmp_path / "selected_configs.lock"
    selected_digest = write_hashed(selected_path, "frozen: true\n")

    payload = protocol_payload(unlocked=True)
    payload["hashes"]["selected_configs_sha256"] = selected_digest
    protocol_path = tmp_path / "protocol_frozen.yaml"
    write_hashed(protocol_path, payload)
    gate = ProtocolGate.load(protocol_path)

    with pytest.raises(ProtocolError, match="selected_configs.lock is required"):
        gate.require_split("test")
    gate.require_split("test", selected_configs_path=selected_path)

    selected_path.write_text("frozen: false\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="SHA-256 mismatch"):
        gate.require_split("test", selected_configs_path=selected_path)


def test_unlocked_protocol_requires_selected_config_digest() -> None:
    payload = protocol_payload(unlocked=True)
    payload["hashes"]["selected_configs_sha256"] = None
    with pytest.raises(ProtocolError, match="selected_configs_sha256"):
        validate_protocol(payload)
