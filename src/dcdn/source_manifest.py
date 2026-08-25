"""Fail-closed source-manifest serialization and validation.

Source manifests are durable provenance records only.  They do not unlock a
dataset test split and must never contain row-derived values or metrics.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_SIDECAR_BYTES = 256
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
MANIFEST_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SourceManifestError(RuntimeError):
    """Raised when a source manifest or its sidecar is not trustworthy."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_source_manifest_payload(payload: Mapping[str, Any]) -> None:
    """Validate the common source-only contract and its locked boundaries."""

    if payload.get("schema_version") != 1:
        raise SourceManifestError("schema_version must equal 1")
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or DATASET_ID_RE.fullmatch(dataset_id) is None:
        raise SourceManifestError("dataset_id is missing or invalid")
    if payload.get("artifact_role") != "raw_source_provenance_only":
        raise SourceManifestError("artifact_role must equal raw_source_provenance_only")
    if payload.get("acquisition_state") != "raw_download_verified":
        raise SourceManifestError("acquisition_state must equal raw_download_verified")

    exact_flags = {
        "raw_download_verified": True,
        "test_unlocked": False,
        "values_not_scanned": True,
        "test_labels_opened": False,
        "validation_or_test_metrics_read": False,
    }
    for field, expected in exact_flags.items():
        if type(payload.get(field)) is not bool or payload[field] is not expected:
            raise SourceManifestError(f"{field} must be {expected!r}")

    for field in ("source", "storage", "inspection"):
        if not isinstance(payload.get(field), Mapping):
            raise SourceManifestError(f"{field} must be a mapping")

    source = payload["source"]
    if source.get("access") != "public_unauthenticated_read_only":
        raise SourceManifestError(
            "source.access must be 'public_unauthenticated_read_only'"
        )
    for field in ("credentials_used", "telemetry_used"):
        if type(source.get(field)) is not bool or source[field] is not False:
            raise SourceManifestError(f"source.{field} must be False")

    storage = payload["storage"]
    if storage.get("class") != "ephemeral_regenerable_public_raw_data":
        raise SourceManifestError(
            "storage.class must be 'ephemeral_regenerable_public_raw_data'"
        )
    for field in ("persistent_raw_copy_created", "raw_data_committed"):
        if type(storage.get(field)) is not bool or storage[field] is not False:
            raise SourceManifestError(f"storage.{field} must be False")

    inspection = payload["inspection"]
    inspection_flags = {
        "archive_or_repository_metadata_read": True,
        "byte_hashes_computed": True,
        "row_values_scanned": False,
        "test_labels_opened": False,
        "validation_or_test_metrics_read": False,
        "parquet_or_numpy_decoded": False,
    }
    for field, expected in inspection_flags.items():
        if type(inspection.get(field)) is not bool or inspection[field] is not expected:
            raise SourceManifestError(f"inspection.{field} must be {expected!r}")
    csv_headers_read = inspection.get("csv_headers_read")
    if type(csv_headers_read) is not bool:
        raise SourceManifestError("inspection.csv_headers_read must be a boolean")
    if inspection.get("first_line_headers_only") is not csv_headers_read:
        raise SourceManifestError(
            "inspection.first_line_headers_only must match csv_headers_read"
        )


def dump_source_manifest(payload: Mapping[str, Any]) -> bytes:
    """Return deterministic YAML bytes after validating the common contract."""

    validate_source_manifest_payload(payload)
    text = yaml.safe_dump(
        dict(payload),
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    )
    data = text.encode("utf-8")
    if len(data) > MAX_MANIFEST_BYTES:
        raise SourceManifestError("serialized source manifest exceeds size limit")
    return data


def _write_fsynced_temp(parent: Path, prefix: str, data: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(dir=parent, prefix=prefix)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_source_manifest(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Atomically replace a manifest and deterministic SHA-256 sidecar.

    Both temporary files are created and fsynced in the target directory.
    A crash between the two replacements can only leave a detectable
    manifest/sidecar mismatch; readers always verify the pair before YAML
    parsing.
    """

    requested = Path(path)
    if MANIFEST_NAME_RE.fullmatch(requested.name) is None:
        raise SourceManifestError("manifest filename is missing or unsafe")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    sidecar = parent / f"{requested.name}.sha256"

    data = dump_source_manifest(payload)
    digest = sha256_bytes(data)
    sidecar_data = f"{digest}  {target.name}\n".encode("ascii")
    manifest_temp: Path | None = None
    sidecar_temp: Path | None = None
    try:
        manifest_temp = _write_fsynced_temp(parent, f".{target.name}.manifest-", data)
        sidecar_temp = _write_fsynced_temp(
            parent, f".{sidecar.name}.sidecar-", sidecar_data
        )
        os.replace(manifest_temp, target)
        manifest_temp = None
        os.replace(sidecar_temp, sidecar)
        sidecar_temp = None
        _fsync_directory(parent)
    finally:
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)
        if sidecar_temp is not None:
            sidecar_temp.unlink(missing_ok=True)
    return digest


def _read_bounded_regular(path: Path, *, limit: int, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourceManifestError(
            f"{label} must be a regular non-symlink file: {path}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SourceManifestError(f"{label} must be a private regular file")
        if metadata.st_size > limit:
            raise SourceManifestError(f"{label} exceeds size limit")
        handle = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise SourceManifestError(f"{label} exceeds size limit")
    return data


def load_source_manifest(path: str | Path) -> dict[str, Any]:
    """Verify a strict sidecar, parse unique-key YAML, and validate the lock."""

    source = Path(path)
    if MANIFEST_NAME_RE.fullmatch(source.name) is None:
        raise SourceManifestError("manifest filename is missing or unsafe")
    sidecar = source.with_name(f"{source.name}.sha256")
    data = _read_bounded_regular(source, limit=MAX_MANIFEST_BYTES, label="manifest")
    sidecar_data = _read_bounded_regular(
        sidecar, limit=MAX_SIDECAR_BYTES, label="sidecar"
    )
    digest = sha256_bytes(data)
    expected_sidecar = f"{digest}  {source.name}\n".encode("ascii")
    if sidecar_data != expected_sidecar:
        raise SourceManifestError("source manifest sidecar mismatch")
    try:
        payload = yaml.load(data, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise SourceManifestError(f"invalid source manifest YAML: {error}") from error
    if not isinstance(payload, dict):
        raise SourceManifestError("source manifest root must be a mapping")
    validate_source_manifest_payload(payload)
    return payload
