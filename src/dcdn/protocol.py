"""Immutable protocol manifests and fail-closed test-access gating."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
FINAL_SEEDS = (42, 43, 44)


class ProtocolError(RuntimeError):
    """Raised when a protocol artifact or access request is unsafe."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of path without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProtocolError(f"{field} must be a lowercase SHA-256 digest")
    return value


def read_sha256_sidecar(path: str | Path) -> str:
    """Read a conventional digest-and-filename SHA-256 sidecar."""

    source = Path(path)
    sidecar = source.with_name(f"{source.name}.sha256")
    try:
        tokens = sidecar.read_text(encoding="utf-8").strip().split()
    except FileNotFoundError as exc:
        raise ProtocolError(f"missing SHA-256 sidecar: {sidecar}") from exc
    if len(tokens) != 2 or tokens[1] != source.name:
        raise ProtocolError(f"invalid SHA-256 sidecar format: {sidecar}")
    return _require_sha256(tokens[0], str(sidecar))


def verify_sha256_sidecar(path: str | Path) -> str:
    """Verify a file against its adjacent SHA-256 sidecar."""

    expected = read_sha256_sidecar(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ProtocolError(
            f"SHA-256 mismatch for {Path(path)}: expected {expected}, got {actual}"
        )
    return actual


def load_verified_yaml(path: str | Path) -> Mapping[str, Any]:
    """Load a mapping only after its byte-level sidecar hash passes."""

    source = Path(path)
    verify_sha256_sidecar(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"YAML root must be a mapping: {source}")
    return payload


def validate_protocol(payload: Mapping[str, Any]) -> None:
    """Validate the immutable fields required before any split is opened."""

    if payload.get("schema_version") != 1:
        raise ProtocolError("protocol schema_version must equal 1")
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ProtocolError("dataset_id must be a non-empty string")
    if type(payload.get("test_unlocked")) is not bool:
        raise ProtocolError("test_unlocked must be a boolean")

    hashes = payload.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ProtocolError("hashes must be a mapping")
    for name in (
        "data_manifest_sha256",
        "features_sha256",
        "split_sha256",
        "search_space_sha256",
        "experiment_matrix_sha256",
    ):
        _require_sha256(hashes.get(name), f"hashes.{name}")

    seeds = payload.get("final_seeds")
    normalized_seeds = tuple(seeds) if isinstance(seeds, list) else None
    if normalized_seeds != FINAL_SEEDS:
        raise ProtocolError(f"final_seeds must be exactly {list(FINAL_SEEDS)}")
    if payload.get("tuning_seed") != 42:
        raise ProtocolError("tuning_seed must equal 42")
    if payload.get("tuning_splits") != ["train", "validation"]:
        raise ProtocolError("tuning_splits must be exactly train and validation")
    if payload.get("test_evaluations_per_final_seed") != 1:
        raise ProtocolError("each final seed may be evaluated on test exactly once")

    if payload["test_unlocked"]:
        _require_sha256(
            hashes.get("selected_configs_sha256"),
            "hashes.selected_configs_sha256",
        )


@dataclass(frozen=True)
class ProtocolGate:
    """A verified, immutable dataset protocol with fail-closed split access."""

    path: Path
    digest: str
    payload: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> ProtocolGate:
        source = Path(path)
        payload = load_verified_yaml(source)
        validate_protocol(payload)
        return cls(path=source, digest=sha256_file(source), payload=payload)

    @property
    def dataset_id(self) -> str:
        return str(self.payload["dataset_id"])

    @property
    def test_unlocked(self) -> bool:
        return bool(self.payload["test_unlocked"])

    def require_split(
        self,
        split: str,
        *,
        selected_configs_path: str | Path | None = None,
    ) -> None:
        """Reject unknown splits and test access that lacks a verified lock."""

        if split not in ALLOWED_SPLITS:
            raise ProtocolError(f"unknown split: {split!r}")
        if split != "test":
            return
        if not self.test_unlocked:
            raise ProtocolError(f"test access is locked for dataset {self.dataset_id}")
        if selected_configs_path is None:
            raise ProtocolError("selected_configs.lock is required for test access")

        selected_path = Path(selected_configs_path)
        selected_digest = verify_sha256_sidecar(selected_path)
        expected = self.payload["hashes"]["selected_configs_sha256"]
        if selected_digest != expected:
            raise ProtocolError("selected_configs.lock digest does not match protocol")
