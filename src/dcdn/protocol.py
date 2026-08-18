"""Protocol v3: immutable lineage, active-state, and split-access gates.

All security-sensitive artifacts are read once through O_NOFOLLOW file
descriptors. The exact bytes that are hashed are also the bytes that are
parsed. Protocol activation and test claims are durable, sidecar-verified,
centrally pinned, and fail closed on a torn write.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from yaml.constructor import ConstructorError

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
FINAL_SEEDS = (42, 43, 44)
PROTOCOL_SCHEMA_VERSION = 3
SELECTED_CONFIGS_SCHEMA_VERSION = 1
ACTIVE_PROTOCOL_SCHEMA_VERSION = 1
TEST_CLAIMS_SCHEMA_VERSION = 2

BASE_HASH_KEYS = frozenset(
    {
        "data_manifest_sha256",
        "views_manifest_sha256",
        "features_sha256",
        "split_sha256",
        "search_space_sha256",
        "experiment_matrix_sha256",
        "physical_schema_sha256",
        "expanded_schema_sha256",
        "operand_schema_sha256",
        "denominator_mask_sha256",
        "full_materialized_scan_sha256",
        "cache_content_sha256",
        "encoder_contract_sha256",
        "capacity_resolution_sha256",
    }
)
SELECTED_CONFIGS_HASH_KEY = "selected_configs_sha256"
COMMON_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "feature_regime",
        "required_model_keys",
        "test_unlocked",
        "test_path_available",
        "hashes",
        "split_row_counts",
        "final_seeds",
        "tuning_seed",
        "tuning_splits",
        "test_evaluations_per_final_seed",
    }
)
PARENT_PROTOCOL_KEY = "parent_premetric_protocol_sha256"
RUNTIME_HASH_FIELDS = (
    "data_manifest_sha256",
    "views_manifest_sha256",
    "features_sha256",
    "physical_schema_sha256",
    "expanded_schema_sha256",
    "operand_schema_sha256",
    "denominator_mask_sha256",
    "full_materialized_scan_sha256",
    "cache_content_sha256",
    "encoder_contract_sha256",
    "capacity_resolution_sha256",
)


class ProtocolError(RuntimeError):
    """Raised when a protocol artifact or access request is unsafe."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate keys at every depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class VerifiedArtifact:
    """Bytes and digest obtained from one verified source-file descriptor."""

    path: Path
    data: bytes
    digest: str


@dataclass(frozen=True)
class VerifiedYamlArtifact:
    """Duplicate-free YAML parsed from the exact verified artifact bytes."""

    path: Path
    data: bytes
    digest: str
    payload: Mapping[str, Any]


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProtocolError(f"{field} must be a lowercase SHA-256 digest")
    if len(set(value)) == 1:
        raise ProtocolError(
            f"{field} must not be a repeated-character placeholder digest"
        )
    return value


def _require_exact_keys(
    payload: Mapping[Any, Any],
    expected: frozenset[str],
    field: str,
) -> None:
    actual = set(payload)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(str(key) for key in actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing={missing}")
    if unknown:
        details.append(f"unknown={unknown}")
    raise ProtocolError(f"{field} keys must be exact ({'; '.join(details)})")


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{field} must be a non-empty string")
    if value.strip() != value:
        raise ProtocolError(f"{field} must not have surrounding whitespace")
    if value.casefold() in {
        "unresolved",
        "pending",
        "placeholder",
        "todo",
        "tbd",
    }:
        raise ProtocolError(f"{field} must be resolved")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"{field} must be a positive integer")
    return value


def _require_string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProtocolError(f"{field} must be a non-empty list")
    normalized = tuple(
        _require_nonempty_string(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(normalized)) != len(normalized):
        raise ProtocolError(f"{field} must not contain duplicates")
    return normalized


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _assert_directory_chain(path: Path, *, field: str) -> None:
    if not path.is_absolute():
        raise ProtocolError(f"{field} must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise ProtocolError(f"missing {field}: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ProtocolError(f"unsafe symlink in {field}: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProtocolError(f"non-directory in {field}: {current}")


def _normalized_artifact_path(
    path: str | Path,
    *,
    field: str,
    expected_name: str | None = None,
) -> Path:
    raw = Path(path)
    if any(part == ".." for part in raw.parts):
        raise ProtocolError(f"unsafe {field} path traversal: {raw}")
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    if not absolute.name or any(character.isspace() for character in absolute.name):
        raise ProtocolError(f"unsafe {field} basename: {absolute}")
    if absolute.name.endswith(".sha256") and field != "SHA-256 sidecar":
        raise ProtocolError(f"{field} path must not name a SHA-256 sidecar")
    if expected_name is not None and absolute.name != expected_name:
        raise ProtocolError(f"{field} must be named {expected_name}")
    _assert_directory_chain(absolute.parent, field=f"{field} parent")
    return absolute


def _open_parent_directory(path: Path, *, field: str) -> int:
    _assert_directory_chain(path.parent, field=f"{field} parent")
    try:
        before = path.parent.lstat()
        descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except (FileNotFoundError, OSError) as exc:
        raise ProtocolError(
            f"cannot safely open {field} parent: {path.parent}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        after = path.parent.lstat()
        _assert_directory_chain(path.parent, field=f"{field} parent")
        identities = {
            (before.st_dev, before.st_ino),
            (opened.st_dev, opened.st_ino),
            (after.st_dev, after.st_ino),
        }
        if len(identities) != 1 or not stat.S_ISDIR(opened.st_mode):
            raise ProtocolError(f"{field} parent changed during safe open")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _open_regular_at(parent_fd: int, basename: str, *, field: str) -> int:
    try:
        descriptor = os.open(
            basename,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except (FileNotFoundError, OSError) as exc:
        raise ProtocolError(f"cannot safely open {field}: {basename}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ProtocolError(f"{field} must be a regular file: {basename}")
    return descriptor


def _read_fd_all(
    descriptor: int,
    *,
    field: str,
    limit: int = 64 * 1024 * 1024,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise ProtocolError(f"{field} exceeds the {limit}-byte safety limit")
        chunks.append(chunk)


def read_verified_artifact(path: str | Path) -> VerifiedArtifact:
    """Open source and sidecar once, then hash and return the same source bytes."""

    source = _normalized_artifact_path(path, field="artifact")
    parent_fd = _open_parent_directory(source, field="artifact")
    source_fd: int | None = None
    sidecar_fd: int | None = None
    try:
        source_fd = _open_regular_at(parent_fd, source.name, field="artifact")
        sidecar_name = f"{source.name}.sha256"
        sidecar_fd = _open_regular_at(
            parent_fd,
            sidecar_name,
            field="SHA-256 sidecar",
        )
        source_bytes = _read_fd_all(source_fd, field="artifact")
        sidecar_bytes = _read_fd_all(
            sidecar_fd,
            field="SHA-256 sidecar",
            limit=4096,
        )
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if sidecar_fd is not None:
            os.close(sidecar_fd)
        os.close(parent_fd)

    encoded_name = os.fsencode(source.name)
    expected_length = 64 + 2 + len(encoded_name) + 1
    if (
        len(sidecar_bytes) != expected_length
        or sidecar_bytes[64:66] != b"  "
        or sidecar_bytes[66:-1] != encoded_name
        or sidecar_bytes[-1:] != b"\n"
    ):
        raise ProtocolError(
            "invalid SHA-256 sidecar grammar; expected exactly "
            "64hex + two spaces + basename + newline"
        )
    try:
        expected_digest = sidecar_bytes[:64].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ProtocolError("SHA-256 sidecar digest must be ASCII") from exc
    _require_sha256(expected_digest, "SHA-256 sidecar digest")
    actual_digest = hashlib.sha256(source_bytes).hexdigest()
    if actual_digest != expected_digest:
        raise ProtocolError(
            f"SHA-256 mismatch for {source}: expected {expected_digest}, "
            f"got {actual_digest}"
        )
    return VerifiedArtifact(path=source, data=source_bytes, digest=actual_digest)


def verify_sha256_sidecar(path: str | Path) -> str:
    """Verify an artifact with the strict single-open reader."""

    return read_verified_artifact(path).digest


def read_sha256_sidecar(path: str | Path) -> str:
    """Return the verified digest; source bytes are opened only once."""

    return read_verified_artifact(path).digest


def sha256_file(path: str | Path) -> str:
    """Hash one safely opened regular file without following symlinks."""

    source = _normalized_artifact_path(path, field="artifact")
    parent_fd = _open_parent_directory(source, field="artifact")
    descriptor: int | None = None
    try:
        descriptor = _open_regular_at(parent_fd, source.name, field="artifact")
        data = _read_fd_all(descriptor, field="artifact")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    return hashlib.sha256(data).hexdigest()


def _parse_unique_yaml(data: bytes, source: Path) -> Mapping[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"YAML must be UTF-8: {source}") from exc
    try:
        payload = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ProtocolError(f"invalid or duplicate-key YAML: {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"YAML root must be a mapping: {source}")
    return payload


def load_verified_yaml_artifact(path: str | Path) -> VerifiedYamlArtifact:
    """Hash and parse exactly the same single-open source bytes."""

    artifact = read_verified_artifact(path)
    payload = _parse_unique_yaml(artifact.data, artifact.path)
    return VerifiedYamlArtifact(
        path=artifact.path,
        data=artifact.data,
        digest=artifact.digest,
        payload=payload,
    )


def load_verified_yaml(path: str | Path) -> Mapping[str, Any]:
    """Compatibility wrapper returning duplicate-free verified YAML."""

    return load_verified_yaml_artifact(path).payload


def _canonical_json_value(value: Any, *, field: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ProtocolError(f"{field} contains invalid Unicode") from exc
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ProtocolError(f"{field} contains NaN or Infinity")
        return value
    if isinstance(value, list):
        return [
            _canonical_json_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError(f"{field} mapping keys must be strings")
            normalized[key] = _canonical_json_value(item, field=f"{field}.{key}")
        return normalized
    raise ProtocolError(f"{field} contains unsupported type {type(value).__name__}")


def canonical_split_json_bytes(split_mapping: Mapping[str, Any]) -> bytes:
    """Canonical compact JSON for exactly the views split mapping."""

    if not isinstance(split_mapping, Mapping):
        raise ProtocolError("views split must be a mapping")
    normalized = _canonical_json_value(split_mapping, field="views split")
    try:
        text = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ProtocolError(f"views split is not canonical JSON data: {exc}") from exc


def canonical_split_sha256(split_mapping: Mapping[str, Any]) -> str:
    """Hash the recursively validated canonical views split mapping."""

    return hashlib.sha256(canonical_split_json_bytes(split_mapping)).hexdigest()


def split_sha256_from_views_payload(views_payload: Mapping[str, Any]) -> str:
    """Derive split_sha256 from the singular split views key."""

    if not isinstance(views_payload, Mapping):
        raise ProtocolError("views manifest root must be a mapping")
    split_mapping = views_payload.get("split")
    if not isinstance(split_mapping, Mapping):
        raise ProtocolError("views manifest must contain a split mapping")
    return canonical_split_sha256(split_mapping)


def _entry_stat(parent_fd: int, basename: str) -> os.stat_result | None:
    try:
        return os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _artifact_pair_exists(path: Path) -> bool:
    parent_fd = _open_parent_directory(path, field="artifact registry")
    try:
        source_stat = _entry_stat(parent_fd, path.name)
        sidecar_stat = _entry_stat(parent_fd, f"{path.name}.sha256")
    finally:
        os.close(parent_fd)
    if (source_stat is None) != (sidecar_stat is None):
        raise ProtocolError(f"torn artifact/sidecar pair: {path}")
    for metadata in (source_stat, sidecar_stat):
        if metadata is not None and not stat.S_ISREG(metadata.st_mode):
            raise ProtocolError(f"unsafe non-regular artifact registry entry: {path}")
    return source_stat is not None


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise ProtocolError("short write while persisting protocol state")
        view = view[written:]


def _create_temp_file(parent_fd: int, basename: str, data: bytes) -> str:
    temp_name = f".{basename}.tmp-{secrets.token_hex(12)}"
    descriptor = os.open(
        temp_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProtocolError("temporary protocol state is not a regular file")
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return temp_name


def _atomic_write_hashed_yaml(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically replace content and sidecar; every torn pair fails closed."""

    normalized = _normalized_artifact_path(path, field="state artifact")
    content = yaml.safe_dump(
        dict(payload),
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    sidecar = f"{digest}  {normalized.name}\n".encode()
    parent_fd = _open_parent_directory(normalized, field="state artifact")
    content_temp: str | None = None
    sidecar_temp: str | None = None
    try:
        for name in (normalized.name, f"{normalized.name}.sha256"):
            metadata = _entry_stat(parent_fd, name)
            if metadata is not None and not stat.S_ISREG(metadata.st_mode):
                raise ProtocolError(f"unsafe state target entry: {name}")
        content_temp = _create_temp_file(parent_fd, normalized.name, content)
        sidecar_temp = _create_temp_file(
            parent_fd,
            f"{normalized.name}.sha256",
            sidecar,
        )
        os.replace(
            content_temp,
            normalized.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        content_temp = None
        os.replace(
            sidecar_temp,
            f"{normalized.name}.sha256",
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        sidecar_temp = None
        os.fsync(parent_fd)
    finally:
        for temp_name in (content_temp, sidecar_temp):
            if temp_name is not None:
                with suppress(FileNotFoundError):
                    os.unlink(temp_name, dir_fd=parent_fd)
        os.close(parent_fd)
    return digest


@contextmanager
def _exclusive_registry_lock(path: Path) -> Iterator[None]:
    normalized = _normalized_artifact_path(path, field="registry target")
    parent_fd = _open_parent_directory(normalized, field="registry target")
    lock_name = f".{normalized.name}.lock"
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                lock_name,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise ProtocolError("cannot safely open registry lock") from exc
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ProtocolError("registry lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(parent_fd)


def _optional_verified_yaml(path: Path) -> VerifiedYamlArtifact | None:
    if not _artifact_pair_exists(path):
        return None
    return load_verified_yaml_artifact(path)


def validate_protocol(payload: Mapping[str, Any]) -> None:
    """Validate the exact Protocol v3 schema and lock state."""

    if not isinstance(payload, Mapping):
        raise ProtocolError("protocol root must be a mapping")
    actual_keys = set(payload)
    with_parent = COMMON_ROOT_KEYS | {PARENT_PROTOCOL_KEY}
    unlocked = payload.get("test_unlocked") is True
    if unlocked:
        _require_exact_keys(payload, with_parent, "protocol root")
    elif actual_keys != set(COMMON_ROOT_KEYS) and actual_keys != set(with_parent):
        expected = (
            with_parent if PARENT_PROTOCOL_KEY in actual_keys else COMMON_ROOT_KEYS
        )
        _require_exact_keys(payload, expected, "protocol root")

    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != PROTOCOL_SCHEMA_VERSION:
        raise ProtocolError(
            f"protocol schema_version must equal {PROTOCOL_SCHEMA_VERSION}"
        )
    _require_nonempty_string(payload.get("dataset_id"), "dataset_id")
    _require_nonempty_string(payload.get("feature_regime"), "feature_regime")
    _require_string_list(payload.get("required_model_keys"), "required_model_keys")
    if type(payload.get("test_unlocked")) is not bool:
        raise ProtocolError("test_unlocked must be a boolean")
    if type(payload.get("test_path_available")) is not bool:
        raise ProtocolError("test_path_available must be a boolean")

    unlocked = payload["test_unlocked"]
    if payload["test_path_available"] is not unlocked:
        raise ProtocolError(
            "test_path_available must be false while locked and true only at "
            "final test unlock"
        )
    parent_digest = payload.get(PARENT_PROTOCOL_KEY)
    if unlocked:
        _require_sha256(parent_digest, PARENT_PROTOCOL_KEY)
    elif parent_digest is not None:
        raise ProtocolError(
            "parent_premetric_protocol_sha256 must be null or absent while locked"
        )

    hashes = payload.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ProtocolError("hashes must be a mapping")
    expected_hash_keys = BASE_HASH_KEYS | (
        {SELECTED_CONFIGS_HASH_KEY} if unlocked else set()
    )
    _require_exact_keys(hashes, frozenset(expected_hash_keys), "hashes")
    for name in sorted(expected_hash_keys):
        _require_sha256(hashes[name], f"hashes.{name}")

    counts = payload.get("split_row_counts")
    if not isinstance(counts, Mapping):
        raise ProtocolError("split_row_counts must be a mapping")
    allowed_count_keys = frozenset({"train", "validation", "test"})
    actual_count_keys = set(counts)
    if not {"train", "validation"}.issubset(actual_count_keys):
        raise ProtocolError("split_row_counts must contain train and validation")
    if not actual_count_keys.issubset(allowed_count_keys):
        unknown = sorted(str(key) for key in actual_count_keys - allowed_count_keys)
        raise ProtocolError(f"unknown split_row_counts keys: {unknown}")
    _require_positive_int(counts["train"], "split_row_counts.train")
    _require_positive_int(counts["validation"], "split_row_counts.validation")
    if unlocked:
        if "test" not in counts:
            raise ProtocolError(
                "split_row_counts.test is required at final test unlock"
            )
        _require_positive_int(counts["test"], "split_row_counts.test")
    elif counts.get("test") is not None:
        raise ProtocolError(
            "split_row_counts.test must be null or absent while test is locked"
        )

    seeds = payload.get("final_seeds")
    if (
        not isinstance(seeds, list)
        or any(type(seed) is not int for seed in seeds)
        or tuple(seeds) != FINAL_SEEDS
    ):
        raise ProtocolError(f"final_seeds must be exactly {list(FINAL_SEEDS)}")
    tuning_seed = payload.get("tuning_seed")
    if type(tuning_seed) is not int or tuning_seed != 42:
        raise ProtocolError("tuning_seed must equal 42")
    if payload.get("tuning_splits") != ["train", "validation"]:
        raise ProtocolError("tuning_splits must be exactly train and validation")
    test_evaluations = payload.get("test_evaluations_per_final_seed")
    if type(test_evaluations) is not int or test_evaluations != 1:
        raise ProtocolError("test_evaluations_per_final_seed must equal 1")


def _validate_final_lineage(
    final_payload: Mapping[str, Any],
    parent_payload: Mapping[str, Any],
    parent_digest: str,
) -> None:
    if parent_payload["test_unlocked"] is not False:
        raise ProtocolError("final protocol parent must be a premetric locked protocol")
    if final_payload[PARENT_PROTOCOL_KEY] != parent_digest:
        raise ProtocolError(
            "final protocol parent digest does not match parent artifact"
        )
    immutable_root = (
        "schema_version",
        "dataset_id",
        "feature_regime",
        "required_model_keys",
        "final_seeds",
        "tuning_seed",
        "tuning_splits",
        "test_evaluations_per_final_seed",
    )
    for name in immutable_root:
        if final_payload[name] != parent_payload[name]:
            raise ProtocolError(f"final protocol immutable field drift: {name}")
    for name in BASE_HASH_KEYS:
        if final_payload["hashes"][name] != parent_payload["hashes"][name]:
            raise ProtocolError(f"final protocol base hash drift: {name}")
    for split in ("train", "validation"):
        if (
            final_payload["split_row_counts"][split]
            != parent_payload["split_row_counts"][split]
        ):
            raise ProtocolError(f"final protocol row-count drift: {split}")


@dataclass(frozen=True)
class RuntimeIdentity:
    """Complete independently attested runtime identity for one full split."""

    dataset_id: str
    split: str
    feature_regime: str
    row_count: int
    data_manifest_sha256: str
    views_manifest_sha256: str
    features_sha256: str
    physical_schema_sha256: str
    expanded_schema_sha256: str
    operand_schema_sha256: str
    denominator_mask_sha256: str
    full_materialized_scan_sha256: str
    cache_content_sha256: str
    encoder_contract_sha256: str
    capacity_resolution_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.dataset_id, "runtime dataset_id")
        if self.split not in ALLOWED_SPLITS:
            raise ProtocolError(f"unknown runtime split: {self.split!r}")
        _require_nonempty_string(self.feature_regime, "runtime feature_regime")
        _require_positive_int(self.row_count, "runtime row_count")
        for name in RUNTIME_HASH_FIELDS:
            _require_sha256(getattr(self, name), f"runtime {name}")


SELECTED_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "artifact_role",
        "dataset_id",
        "feature_regime",
        "parent_premetric_protocol_sha256",
        "search_space_sha256",
        "experiment_matrix_sha256",
        "encoder_contract_sha256",
        "capacity_resolution_sha256",
        "required_model_keys",
        "selected_models",
    }
)
SELECTED_MODEL_KEYS = frozenset(
    {
        "candidate_id",
        "resolved_config_sha256",
        "selected_on_validation_only",
        "test_data_accessed",
        "test_metrics_consulted",
    }
)


def _validate_selected_configs(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, SELECTED_LOCK_KEYS, "selected_configs.lock root")
    schema_version = payload.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != SELECTED_CONFIGS_SCHEMA_VERSION
    ):
        raise ProtocolError("selected_configs.lock schema_version must equal 1")
    if payload.get("artifact_role") != "final_selected_configs_lock":
        raise ProtocolError("selected_configs.lock artifact_role is invalid")
    _require_nonempty_string(payload.get("dataset_id"), "selected dataset_id")
    _require_nonempty_string(payload.get("feature_regime"), "selected feature_regime")
    for name in (
        "parent_premetric_protocol_sha256",
        "search_space_sha256",
        "experiment_matrix_sha256",
        "encoder_contract_sha256",
        "capacity_resolution_sha256",
    ):
        _require_sha256(payload.get(name), f"selected {name}")
    required = _require_string_list(
        payload.get("required_model_keys"),
        "selected required_model_keys",
    )
    models = payload.get("selected_models")
    if not isinstance(models, Mapping):
        raise ProtocolError("selected_models must be a mapping")
    if set(models) != set(required):
        raise ProtocolError(
            "selected_models keys must exactly match required_model_keys"
        )
    for model_key in required:
        selected = models[model_key]
        if not isinstance(selected, Mapping):
            raise ProtocolError(f"selected model {model_key} must be a mapping")
        _require_exact_keys(
            selected,
            SELECTED_MODEL_KEYS,
            f"selected model {model_key}",
        )
        _require_nonempty_string(
            selected.get("candidate_id"),
            f"selected model {model_key}.candidate_id",
        )
        _require_sha256(
            selected.get("resolved_config_sha256"),
            f"selected model {model_key}.resolved_config_sha256",
        )
        if selected.get("selected_on_validation_only") is not True:
            raise ProtocolError(f"selected model {model_key} must be validation-only")
        if selected.get("test_data_accessed") is not False:
            raise ProtocolError(
                f"selected model {model_key} must attest no test data access"
            )
        if selected.get("test_metrics_consulted") is not False:
            raise ProtocolError(
                f"selected model {model_key} must attest no test metrics"
            )


@dataclass(frozen=True)
class SelectedConfigsLock:
    """Strict parsed selected-configuration lock bound to a final protocol."""

    path: Path
    digest: str
    payload: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> SelectedConfigsLock:
        source = _normalized_artifact_path(
            path,
            field="selected configs lock",
            expected_name="selected_configs.lock",
        )
        artifact = load_verified_yaml_artifact(source)
        _validate_selected_configs(artifact.payload)
        return cls(
            path=artifact.path,
            digest=artifact.digest,
            payload=_freeze(artifact.payload),
        )

    def require_for_protocol(self, gate: ProtocolGate) -> None:
        if not gate.test_unlocked:
            raise ProtocolError("selected configs can bind only a final protocol")
        expected = {
            "dataset_id": gate.dataset_id,
            "feature_regime": gate.feature_regime,
            "parent_premetric_protocol_sha256": gate.parent_premetric_digest,
            "search_space_sha256": gate.payload["hashes"]["search_space_sha256"],
            "experiment_matrix_sha256": gate.payload["hashes"][
                "experiment_matrix_sha256"
            ],
            "encoder_contract_sha256": gate.payload["hashes"][
                "encoder_contract_sha256"
            ],
            "capacity_resolution_sha256": gate.payload["hashes"][
                "capacity_resolution_sha256"
            ],
        }
        for name, value in expected.items():
            if self.payload[name] != value:
                raise ProtocolError(f"selected configs binding mismatch: {name}")
        if tuple(self.payload["required_model_keys"]) != gate.required_model_keys:
            raise ProtocolError("selected configs required_model_keys mismatch")
        if self.digest != gate.payload["hashes"][SELECTED_CONFIGS_HASH_KEY]:
            raise ProtocolError("selected_configs.lock digest does not match protocol")

    def require_model(
        self,
        *,
        model_key: str,
        candidate_id: str,
        resolved_config_sha256: str,
    ) -> None:
        if model_key not in self.payload["selected_models"]:
            raise ProtocolError(f"model is not in selected_configs.lock: {model_key}")
        selected = self.payload["selected_models"][model_key]
        if selected["candidate_id"] != candidate_id:
            raise ProtocolError("candidate_id does not match selected_configs.lock")
        _require_sha256(resolved_config_sha256, "resolved_config_sha256")
        if selected["resolved_config_sha256"] != resolved_config_sha256:
            raise ProtocolError(
                "resolved_config_sha256 does not match selected_configs.lock"
            )


ACTIVE_STATE_KEYS = frozenset(
    {
        "schema_version",
        "artifact_role",
        "dataset_id",
        "feature_regime",
        "active_protocol_sha256",
        "active_lock_stage",
        "parent_premetric_protocol_sha256",
        "test_claim_registry_path",
    }
)


def _validate_active_state(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, ACTIVE_STATE_KEYS, "active protocol state")
    schema_version = payload.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != ACTIVE_PROTOCOL_SCHEMA_VERSION
    ):
        raise ProtocolError("active protocol state schema_version must equal 1")
    if payload.get("artifact_role") != "active_protocol_registry":
        raise ProtocolError("active protocol state artifact_role is invalid")
    _require_nonempty_string(payload.get("dataset_id"), "active dataset_id")
    _require_nonempty_string(payload.get("feature_regime"), "active feature_regime")
    _require_sha256(
        payload.get("active_protocol_sha256"),
        "active_protocol_sha256",
    )
    stage = payload.get("active_lock_stage")
    if stage not in {"premetric_locked", "final_selected_locked"}:
        raise ProtocolError("active_lock_stage is invalid")
    parent = payload.get("parent_premetric_protocol_sha256")
    claims_path = payload.get("test_claim_registry_path")
    if stage == "premetric_locked":
        if parent is not None or claims_path is not None:
            raise ProtocolError("premetric active state must not expose final facts")
    else:
        _require_sha256(parent, "active parent_premetric_protocol_sha256")
        if not isinstance(claims_path, str):
            raise ProtocolError(
                "final active state requires a test claim registry path"
            )
        normalized = _normalized_artifact_path(
            claims_path,
            field="test claim registry",
            expected_name="test_claim_registry.yaml",
        )
        if str(normalized) != claims_path:
            raise ProtocolError(
                "test claim registry path must be canonical and absolute"
            )


@dataclass(frozen=True)
class ActiveProtocolState:
    path: Path
    digest: str
    payload: Mapping[str, Any]


class ActiveProtocolRegistry:
    """Durable single-active-protocol state with rollback-safe activation."""

    def __init__(self, path: str | Path) -> None:
        self.path = _normalized_artifact_path(
            path,
            field="active protocol registry",
            expected_name="active_protocol.yaml",
        )

    def load(self) -> ActiveProtocolState:
        artifact = load_verified_yaml_artifact(self.path)
        _validate_active_state(artifact.payload)
        return ActiveProtocolState(
            path=artifact.path,
            digest=artifact.digest,
            payload=_freeze(artifact.payload),
        )

    def _load_optional(self) -> ActiveProtocolState | None:
        artifact = _optional_verified_yaml(self.path)
        if artifact is None:
            return None
        _validate_active_state(artifact.payload)
        return ActiveProtocolState(
            path=artifact.path,
            digest=artifact.digest,
            payload=_freeze(artifact.payload),
        )

    def activate(
        self,
        gate: ProtocolGate,
        *,
        selected_configs_path: str | Path | None = None,
        test_claim_registry_path: str | Path | None = None,
    ) -> ActiveProtocolState:
        if gate.active_registry_path != self.path:
            raise ProtocolError("gate is not bound to this active protocol registry")
        selected: SelectedConfigsLock | None = None
        claims_path: Path | None = None
        if gate.test_unlocked:
            if selected_configs_path is None or test_claim_registry_path is None:
                raise ProtocolError(
                    "final activation requires selected configs and test claim registry"
                )
            selected = SelectedConfigsLock.load(selected_configs_path)
            selected.require_for_protocol(gate)
            claims_path = _normalized_artifact_path(
                test_claim_registry_path,
                field="test claim registry",
                expected_name="test_claim_registry.yaml",
            )
        elif selected_configs_path is not None or test_claim_registry_path is not None:
            raise ProtocolError("premetric activation must not provide final artifacts")

        with _exclusive_registry_lock(self.path):
            current = self._load_optional()
            if (
                current is not None
                and current.payload["active_protocol_sha256"] == gate.digest
            ):
                expected_claims = str(claims_path) if claims_path is not None else None
                if current.payload["test_claim_registry_path"] != expected_claims:
                    raise ProtocolError("idempotent activation claim registry mismatch")
                return current
            if gate.test_unlocked:
                if current is None:
                    raise ProtocolError(
                        "final activation requires its premetric parent to be active"
                    )
                if (
                    current.payload["active_lock_stage"] != "premetric_locked"
                    or current.payload["active_protocol_sha256"]
                    != gate.parent_premetric_digest
                ):
                    raise ProtocolError(
                        "final activation parent does not match the active protocol"
                    )
            elif current is not None:
                raise ProtocolError("active protocol rollback/replacement is forbidden")

            payload = {
                "schema_version": ACTIVE_PROTOCOL_SCHEMA_VERSION,
                "artifact_role": "active_protocol_registry",
                "dataset_id": gate.dataset_id,
                "feature_regime": gate.feature_regime,
                "active_protocol_sha256": gate.digest,
                "active_lock_stage": gate.lock_stage,
                "parent_premetric_protocol_sha256": (
                    gate.parent_premetric_digest if gate.test_unlocked else None
                ),
                "test_claim_registry_path": (
                    str(claims_path) if claims_path is not None else None
                ),
            }
            _atomic_write_hashed_yaml(self.path, payload)
            return self.load()

    def require_active(self, gate: ProtocolGate) -> ActiveProtocolState:
        state = self.load()
        expected = {
            "dataset_id": gate.dataset_id,
            "feature_regime": gate.feature_regime,
            "active_protocol_sha256": gate.digest,
            "active_lock_stage": gate.lock_stage,
            "parent_premetric_protocol_sha256": (
                gate.parent_premetric_digest if gate.test_unlocked else None
            ),
        }
        for name, value in expected.items():
            if state.payload[name] != value:
                raise ProtocolError(f"gate protocol is not active: {name}")
        return state


CLAIM_REGISTRY_KEYS = frozenset(
    {"schema_version", "artifact_role", "dataset_id", "feature_regime", "claims"}
)
CLAIM_RUNTIME_FIELDS = (
    "dataset_id",
    "split",
    "feature_regime",
    "row_count",
    *RUNTIME_HASH_FIELDS,
)
CLAIM_RUNTIME_KEYS = frozenset(CLAIM_RUNTIME_FIELDS)
CLAIM_STATES = frozenset({"reserved", "opened", "completed"})
CLAIM_KEYS = frozenset(
    {
        "claim_id",
        "state",
        "model_key",
        "seed",
        "final_protocol_sha256",
        "parent_premetric_protocol_sha256",
        "selected_configs_sha256",
        "checkpoint_protocol_sha256",
        "candidate_id",
        "resolved_config_sha256",
        "runtime_identity",
        "result_receipt_sha256",
        "result_row_count",
    }
)
CLAIM_BINDING_KEYS = frozenset(
    CLAIM_KEYS
    - {
        "claim_id",
        "state",
        "result_receipt_sha256",
        "result_row_count",
    }
)


def _runtime_identity_payload(identity: RuntimeIdentity) -> dict[str, Any]:
    return {name: getattr(identity, name) for name in CLAIM_RUNTIME_FIELDS}


def _runtime_identity_from_payload(
    payload: Mapping[str, Any],
    *,
    field: str,
) -> RuntimeIdentity:
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"{field} must be a mapping")
    _require_exact_keys(payload, CLAIM_RUNTIME_KEYS, field)
    return RuntimeIdentity(**{name: payload[name] for name in CLAIM_RUNTIME_FIELDS})


def _claim_key(
    dataset_id: str,
    feature_regime: str,
    claim: Mapping[str, Any],
) -> tuple[str, str, str, int, str]:
    return (
        dataset_id,
        feature_regime,
        claim["model_key"],
        claim["seed"],
        claim["final_protocol_sha256"],
    )


def _validate_claim_registry(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, CLAIM_REGISTRY_KEYS, "test claim registry")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != TEST_CLAIMS_SCHEMA_VERSION:
        raise ProtocolError(
            f"test claim registry schema_version must equal "
            f"{TEST_CLAIMS_SCHEMA_VERSION}"
        )
    if payload.get("artifact_role") != "test_evaluation_claim_registry":
        raise ProtocolError("test claim registry artifact_role is invalid")
    dataset_id = _require_nonempty_string(
        payload.get("dataset_id"),
        "claim registry dataset_id",
    )
    regime = _require_nonempty_string(
        payload.get("feature_regime"),
        "claim registry feature_regime",
    )
    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise ProtocolError("test claim registry claims must be a list")

    seen_keys: set[tuple[str, str, str, int, str]] = set()
    seen_ids: set[str] = set()
    for index, claim in enumerate(claims):
        field = f"test claim {index}"
        if not isinstance(claim, Mapping):
            raise ProtocolError(f"{field} must be a mapping")
        _require_exact_keys(claim, CLAIM_KEYS, field)
        claim_id = _require_sha256(claim.get("claim_id"), f"{field}.claim_id")
        if claim_id in seen_ids:
            raise ProtocolError("duplicate test claim_id")
        seen_ids.add(claim_id)

        state = claim.get("state")
        if state not in CLAIM_STATES:
            raise ProtocolError(f"{field}.state is invalid")
        _require_nonempty_string(claim.get("model_key"), f"{field}.model_key")
        seed = claim.get("seed")
        if type(seed) is not int or seed not in FINAL_SEEDS:
            raise ProtocolError(f"{field}.seed must be one of {FINAL_SEEDS}")
        for name in (
            "final_protocol_sha256",
            "parent_premetric_protocol_sha256",
            "selected_configs_sha256",
            "checkpoint_protocol_sha256",
            "resolved_config_sha256",
        ):
            _require_sha256(claim.get(name), f"{field}.{name}")
        if (
            claim["parent_premetric_protocol_sha256"]
            != claim["checkpoint_protocol_sha256"]
        ):
            raise ProtocolError(f"{field} checkpoint must remain parent-bound")
        _require_nonempty_string(
            claim.get("candidate_id"),
            f"{field}.candidate_id",
        )

        identity = _runtime_identity_from_payload(
            claim.get("runtime_identity"),
            field=f"{field}.runtime_identity",
        )
        if identity.split != "test":
            raise ProtocolError(f"{field} runtime split must be test")
        if identity.dataset_id != dataset_id:
            raise ProtocolError(f"{field} runtime dataset mismatch")
        if identity.feature_regime != regime:
            raise ProtocolError(f"{field} runtime feature regime mismatch")

        receipt = claim.get("result_receipt_sha256")
        result_rows = claim.get("result_row_count")
        if state in {"reserved", "opened"}:
            if receipt is not None or result_rows is not None:
                raise ProtocolError(f"{field} has premature completion evidence")
        else:
            _require_sha256(receipt, f"{field}.result_receipt_sha256")
            _require_positive_int(result_rows, f"{field}.result_row_count")
            if result_rows != identity.row_count:
                raise ProtocolError(f"{field} result row count is incomplete")

        key = _claim_key(dataset_id, regime, claim)
        if key in seen_keys:
            raise ProtocolError("duplicate test lifecycle cell")
        seen_keys.add(key)


@dataclass(frozen=True)
class TestEvaluationClaim:
    claim_id: str
    state: str
    dataset_id: str
    feature_regime: str
    model_key: str
    seed: int
    final_protocol_sha256: str
    parent_premetric_protocol_sha256: str
    selected_configs_sha256: str
    checkpoint_protocol_sha256: str
    candidate_id: str
    resolved_config_sha256: str
    runtime_identity: RuntimeIdentity
    result_receipt_sha256: str | None
    result_row_count: int | None
    registry_path: Path

    def mark_opened(
        self,
        *,
        gate: ProtocolGate,
        selected_configs_path: str | Path,
        runtime_identity: RuntimeIdentity,
    ) -> TestEvaluationClaim:
        """Atomically consume the at-most-once test-opening permission."""

        return TestClaimRegistry(self.registry_path).mark_opened(
            self,
            gate=gate,
            selected_configs_path=selected_configs_path,
            runtime_identity=runtime_identity,
        )

    def mark_completed(
        self,
        *,
        gate: ProtocolGate,
        selected_configs_path: str | Path,
        runtime_identity: RuntimeIdentity,
        result_receipt_sha256: str,
        result_row_count: int,
    ) -> TestEvaluationClaim:
        """Atomically persist the single completion receipt."""

        return TestClaimRegistry(self.registry_path).mark_completed(
            self,
            gate=gate,
            selected_configs_path=selected_configs_path,
            runtime_identity=runtime_identity,
            result_receipt_sha256=result_receipt_sha256,
            result_row_count=result_row_count,
        )


class TestClaimRegistry:
    """Central reserved/opened/completed lifecycle for final test cells."""

    def __init__(self, path: str | Path) -> None:
        self.path = _normalized_artifact_path(
            path,
            field="test claim registry",
            expected_name="test_claim_registry.yaml",
        )

    def _require_bindings(
        self,
        *,
        gate: ProtocolGate,
        selected_configs_path: str | Path,
        runtime_identity: RuntimeIdentity,
        model_key: str,
        seed: int,
        checkpoint_protocol_sha256: str,
        candidate_id: str,
        resolved_config_sha256: str,
    ) -> SelectedConfigsLock:
        if not gate.test_unlocked:
            raise ProtocolError("test lifecycle requires a final protocol")
        active = gate.require_active()
        if active.payload["test_claim_registry_path"] != str(self.path):
            raise ProtocolError("test claim registry is not the active pinned registry")
        gate.require_runtime_identity(runtime_identity, requested_split="test")
        _require_nonempty_string(model_key, "test claim model_key")
        if type(seed) is not int or seed not in FINAL_SEEDS:
            raise ProtocolError(f"test claim seed must be one of {FINAL_SEEDS}")
        _require_sha256(checkpoint_protocol_sha256, "checkpoint_protocol_sha256")
        _require_nonempty_string(candidate_id, "test claim candidate_id")
        _require_sha256(resolved_config_sha256, "test claim resolved_config_sha256")

        selected = SelectedConfigsLock.load(selected_configs_path)
        selected.require_for_protocol(gate)
        selected.require_model(
            model_key=model_key,
            candidate_id=candidate_id,
            resolved_config_sha256=resolved_config_sha256,
        )
        if checkpoint_protocol_sha256 != gate.parent_premetric_digest:
            raise ProtocolError(
                "test checkpoint must remain bound to the parent premetric protocol"
            )
        return selected

    def _binding_record(
        self,
        *,
        gate: ProtocolGate,
        selected: SelectedConfigsLock,
        runtime_identity: RuntimeIdentity,
        model_key: str,
        seed: int,
        checkpoint_protocol_sha256: str,
        candidate_id: str,
        resolved_config_sha256: str,
    ) -> dict[str, Any]:
        parent_digest = gate.parent_premetric_digest
        if parent_digest is None:
            raise ProtocolError("final protocol is missing its parent digest")
        return {
            "model_key": model_key,
            "seed": seed,
            "final_protocol_sha256": gate.digest,
            "parent_premetric_protocol_sha256": parent_digest,
            "selected_configs_sha256": selected.digest,
            "checkpoint_protocol_sha256": checkpoint_protocol_sha256,
            "candidate_id": candidate_id,
            "resolved_config_sha256": resolved_config_sha256,
            "runtime_identity": _runtime_identity_payload(runtime_identity),
        }

    def _load_mutable_payload(self, gate: ProtocolGate) -> dict[str, Any]:
        artifact = _optional_verified_yaml(self.path)
        if artifact is None:
            return {
                "schema_version": TEST_CLAIMS_SCHEMA_VERSION,
                "artifact_role": "test_evaluation_claim_registry",
                "dataset_id": gate.dataset_id,
                "feature_regime": gate.feature_regime,
                "claims": [],
            }
        _validate_claim_registry(artifact.payload)
        return {
            "schema_version": artifact.payload["schema_version"],
            "artifact_role": artifact.payload["artifact_role"],
            "dataset_id": artifact.payload["dataset_id"],
            "feature_regime": artifact.payload["feature_regime"],
            "claims": [
                {
                    **dict(item),
                    "runtime_identity": dict(item["runtime_identity"]),
                }
                for item in artifact.payload["claims"]
            ],
        }

    def _require_registry_identity(
        self,
        payload: Mapping[str, Any],
        gate: ProtocolGate,
    ) -> None:
        if (
            payload["dataset_id"] != gate.dataset_id
            or payload["feature_regime"] != gate.feature_regime
        ):
            raise ProtocolError("test claim registry dataset/regime mismatch")

    def _to_claim(
        self,
        payload: Mapping[str, Any],
        record: Mapping[str, Any],
    ) -> TestEvaluationClaim:
        identity = _runtime_identity_from_payload(
            record["runtime_identity"],
            field="test claim runtime_identity",
        )
        return TestEvaluationClaim(
            claim_id=record["claim_id"],
            state=record["state"],
            dataset_id=payload["dataset_id"],
            feature_regime=payload["feature_regime"],
            model_key=record["model_key"],
            seed=record["seed"],
            final_protocol_sha256=record["final_protocol_sha256"],
            parent_premetric_protocol_sha256=record["parent_premetric_protocol_sha256"],
            selected_configs_sha256=record["selected_configs_sha256"],
            checkpoint_protocol_sha256=record["checkpoint_protocol_sha256"],
            candidate_id=record["candidate_id"],
            resolved_config_sha256=record["resolved_config_sha256"],
            runtime_identity=identity,
            result_receipt_sha256=record["result_receipt_sha256"],
            result_row_count=record["result_row_count"],
            registry_path=self.path,
        )

    def _write_payload(self, payload: dict[str, Any]) -> None:
        payload["claims"].sort(
            key=lambda item: (
                item["model_key"],
                item["seed"],
                item["final_protocol_sha256"],
            )
        )
        _validate_claim_registry(payload)
        _atomic_write_hashed_yaml(self.path, payload)

    def claim(
        self,
        *,
        gate: ProtocolGate,
        selected_configs_path: str | Path,
        runtime_identity: RuntimeIdentity,
        model_key: str,
        seed: int,
        checkpoint_protocol_sha256: str,
        candidate_id: str,
        resolved_config_sha256: str,
    ) -> TestEvaluationClaim:
        """Reserve or recover the same still-reserved final test cell."""

        selected = self._require_bindings(
            gate=gate,
            selected_configs_path=selected_configs_path,
            runtime_identity=runtime_identity,
            model_key=model_key,
            seed=seed,
            checkpoint_protocol_sha256=checkpoint_protocol_sha256,
            candidate_id=candidate_id,
            resolved_config_sha256=resolved_config_sha256,
        )
        binding = self._binding_record(
            gate=gate,
            selected=selected,
            runtime_identity=runtime_identity,
            model_key=model_key,
            seed=seed,
            checkpoint_protocol_sha256=checkpoint_protocol_sha256,
            candidate_id=candidate_id,
            resolved_config_sha256=resolved_config_sha256,
        )
        new_key = (
            gate.dataset_id,
            gate.feature_regime,
            model_key,
            seed,
            gate.digest,
        )
        with _exclusive_registry_lock(self.path):
            payload = self._load_mutable_payload(gate)
            self._require_registry_identity(payload, gate)
            for existing in payload["claims"]:
                if (
                    _claim_key(
                        payload["dataset_id"],
                        payload["feature_regime"],
                        existing,
                    )
                    != new_key
                ):
                    continue
                for name in CLAIM_BINDING_KEYS:
                    if existing[name] != binding[name]:
                        raise ProtocolError(
                            f"reserved test lifecycle binding mismatch: {name}"
                        )
                if existing["state"] != "reserved":
                    raise ProtocolError(
                        "test cell was already opened or completed; "
                        "automatic re-evaluation is forbidden"
                    )
                return self._to_claim(payload, existing)

            used_ids = {item["claim_id"] for item in payload["claims"]}
            claim_id = secrets.token_hex(32)
            while claim_id in used_ids or len(set(claim_id)) == 1:
                claim_id = secrets.token_hex(32)
            record = {
                "claim_id": claim_id,
                "state": "reserved",
                **binding,
                "result_receipt_sha256": None,
                "result_row_count": None,
            }
            payload["claims"].append(record)
            self._write_payload(payload)
            return self._to_claim(payload, record)

    def _transition_context(
        self,
        claim: TestEvaluationClaim,
        *,
        gate: ProtocolGate,
        selected_configs_path: str | Path,
        runtime_identity: RuntimeIdentity,
    ) -> dict[str, Any]:
        if not isinstance(claim, TestEvaluationClaim):
            raise ProtocolError("transition requires a TestEvaluationClaim")
        if claim.registry_path != self.path:
            raise ProtocolError("claim is bound to a different central registry")
        selected = self._require_bindings(
            gate=gate,
            selected_configs_path=selected_configs_path,
            runtime_identity=runtime_identity,
            model_key=claim.model_key,
            seed=claim.seed,
            checkpoint_protocol_sha256=claim.checkpoint_protocol_sha256,
            candidate_id=claim.candidate_id,
            resolved_config_sha256=claim.resolved_config_sha256,
        )
        expected = self._binding_record(
            gate=gate,
            selected=selected,
            runtime_identity=runtime_identity,
            model_key=claim.model_key,
            seed=claim.seed,
            checkpoint_protocol_sha256=claim.checkpoint_protocol_sha256,
            candidate_id=claim.candidate_id,
            resolved_config_sha256=claim.resolved_config_sha256,
        )
        claim_binding = {
            "model_key": claim.model_key,
            "seed": claim.seed,
            "final_protocol_sha256": claim.final_protocol_sha256,
            "parent_premetric_protocol_sha256": (
                claim.parent_premetric_protocol_sha256
            ),
            "selected_configs_sha256": claim.selected_configs_sha256,
            "checkpoint_protocol_sha256": claim.checkpoint_protocol_sha256,
            "candidate_id": claim.candidate_id,
            "resolved_config_sha256": claim.resolved_config_sha256,
            "runtime_identity": _runtime_identity_payload(claim.runtime_identity),
        }
        for name in CLAIM_BINDING_KEYS:
            if claim_binding[name] != expected[name]:
                raise ProtocolError(f"claim transition binding mismatch: {name}")
        if claim.runtime_identity != runtime_identity:
            raise ProtocolError("claim transition runtime identity mismatch")
        return expected

    def _find_claim_record(
        self,
        payload: Mapping[str, Any],
        claim_id: str,
    ) -> dict[str, Any]:
        matches = [item for item in payload["claims"] if item["claim_id"] == claim_id]
        if len(matches) != 1:
            raise ProtocolError("claim_id is missing or ambiguous")
        return matches[0]

    def mark_opened(
        self,
        claim: TestEvaluationClaim,
        *,
        gate: ProtocolGate,
        selected_configs_path: str | Path,
        runtime_identity: RuntimeIdentity,
    ) -> TestEvaluationClaim:
        """Move reserved to opened immediately before the source factory."""

        if claim.state != "reserved":
            raise ProtocolError("only a reserved claim can be opened")
        expected = self._transition_context(
            claim,
            gate=gate,
            selected_configs_path=selected_configs_path,
            runtime_identity=runtime_identity,
        )
        with _exclusive_registry_lock(self.path):
            payload = self._load_mutable_payload(gate)
            self._require_registry_identity(payload, gate)
            record = self._find_claim_record(payload, claim.claim_id)
            for name in CLAIM_BINDING_KEYS:
                if record[name] != expected[name]:
                    raise ProtocolError(f"opened transition binding mismatch: {name}")
            if record["state"] != "reserved":
                raise ProtocolError(
                    "test cell is no longer reserved; reopening is forbidden"
                )
            record["state"] = "opened"
            self._write_payload(payload)
            return self._to_claim(payload, record)

    def mark_completed(
        self,
        claim: TestEvaluationClaim,
        *,
        gate: ProtocolGate,
        selected_configs_path: str | Path,
        runtime_identity: RuntimeIdentity,
        result_receipt_sha256: str,
        result_row_count: int,
    ) -> TestEvaluationClaim:
        """Persist one full-row result receipt for an opened test cell."""

        if claim.state not in {"opened", "completed"}:
            raise ProtocolError("only an opened claim can be completed")
        _require_sha256(result_receipt_sha256, "result_receipt_sha256")
        _require_positive_int(result_row_count, "result_row_count")
        if result_row_count != runtime_identity.row_count:
            raise ProtocolError("result_row_count must equal the attested test rows")
        expected = self._transition_context(
            claim,
            gate=gate,
            selected_configs_path=selected_configs_path,
            runtime_identity=runtime_identity,
        )
        with _exclusive_registry_lock(self.path):
            payload = self._load_mutable_payload(gate)
            self._require_registry_identity(payload, gate)
            record = self._find_claim_record(payload, claim.claim_id)
            for name in CLAIM_BINDING_KEYS:
                if record[name] != expected[name]:
                    raise ProtocolError(
                        f"completed transition binding mismatch: {name}"
                    )
            if record["state"] == "completed":
                if (
                    record["result_receipt_sha256"] == result_receipt_sha256
                    and record["result_row_count"] == result_row_count
                ):
                    return self._to_claim(payload, record)
                raise ProtocolError("test cell already has a different completion")
            if record["state"] != "opened":
                raise ProtocolError("test cell was not opened before completion")
            record["state"] = "completed"
            record["result_receipt_sha256"] = result_receipt_sha256
            record["result_row_count"] = result_row_count
            self._write_payload(payload)
            return self._to_claim(payload, record)


@dataclass(frozen=True)
class ProtocolGate:
    """Verified protocol lineage bound to one central active-state registry."""

    path: Path
    digest: str
    payload: Mapping[str, Any]
    parent_premetric_path: Path | None
    active_registry_path: Path | None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        parent_premetric_protocol_path: str | Path | None = None,
        active_registry_path: str | Path | None = None,
    ) -> ProtocolGate:
        artifact = load_verified_yaml_artifact(path)
        validate_protocol(artifact.payload)
        unlocked = bool(artifact.payload["test_unlocked"])
        parent_path: Path | None = None
        if unlocked:
            if parent_premetric_protocol_path is None:
                raise ProtocolError(
                    "final protocol load requires verified parent premetric path"
                )
            parent_artifact = load_verified_yaml_artifact(
                parent_premetric_protocol_path
            )
            validate_protocol(parent_artifact.payload)
            _validate_final_lineage(
                artifact.payload,
                parent_artifact.payload,
                parent_artifact.digest,
            )
            parent_path = parent_artifact.path
        elif parent_premetric_protocol_path is not None:
            raise ProtocolError("premetric protocol load must not provide a parent")
        active_path = (
            _normalized_artifact_path(
                active_registry_path,
                field="active protocol registry",
                expected_name="active_protocol.yaml",
            )
            if active_registry_path is not None
            else None
        )
        return cls(
            path=artifact.path,
            digest=artifact.digest,
            payload=_freeze(artifact.payload),
            parent_premetric_path=parent_path,
            active_registry_path=active_path,
        )

    @property
    def dataset_id(self) -> str:
        return str(self.payload["dataset_id"])

    @property
    def feature_regime(self) -> str:
        return str(self.payload["feature_regime"])

    @property
    def required_model_keys(self) -> tuple[str, ...]:
        return tuple(self.payload["required_model_keys"])

    @property
    def test_unlocked(self) -> bool:
        return bool(self.payload["test_unlocked"])

    @property
    def lock_stage(self) -> str:
        return "final_selected_locked" if self.test_unlocked else "premetric_locked"

    @property
    def parent_premetric_digest(self) -> str | None:
        value = self.payload.get(PARENT_PROTOCOL_KEY)
        return str(value) if value is not None else None

    def expected_row_count(self, split: str) -> int:
        if split not in ALLOWED_SPLITS:
            raise ProtocolError(f"unknown split: {split!r}")
        value = self.payload["split_row_counts"].get(split)
        if value is None:
            raise ProtocolError(f"row count is unavailable for sealed split {split!r}")
        return int(value)

    def require_views_split(self, views_payload: Mapping[str, Any]) -> None:
        actual = split_sha256_from_views_payload(views_payload)
        if actual != self.payload["hashes"]["split_sha256"]:
            raise ProtocolError("views split mapping digest does not match protocol")

    def require_runtime_identity(
        self,
        identity: RuntimeIdentity,
        *,
        requested_split: str | None = None,
    ) -> None:
        if not isinstance(identity, RuntimeIdentity):
            raise ProtocolError("runtime identity must be a RuntimeIdentity")
        split = requested_split or identity.split
        if split not in ALLOWED_SPLITS or identity.split != split:
            raise ProtocolError("runtime identity split mismatch")
        expected_scalars = {
            "dataset_id": self.dataset_id,
            "feature_regime": self.feature_regime,
            "row_count": self.expected_row_count(split),
        }
        for name, value in expected_scalars.items():
            if getattr(identity, name) != value:
                raise ProtocolError(f"runtime identity mismatch: {name}")
        for name in RUNTIME_HASH_FIELDS:
            if getattr(identity, name) != self.payload["hashes"][name]:
                raise ProtocolError(f"runtime identity mismatch: {name}")

    def activate(
        self,
        *,
        selected_configs_path: str | Path | None = None,
        test_claim_registry_path: str | Path | None = None,
    ) -> ActiveProtocolState:
        if self.active_registry_path is None:
            raise ProtocolError("gate has no active protocol registry binding")
        return ActiveProtocolRegistry(self.active_registry_path).activate(
            self,
            selected_configs_path=selected_configs_path,
            test_claim_registry_path=test_claim_registry_path,
        )

    def require_active(self) -> ActiveProtocolState:
        if self.active_registry_path is None:
            raise ProtocolError("gate has no active protocol registry binding")
        return ActiveProtocolRegistry(self.active_registry_path).require_active(self)

    def require_split(
        self,
        split: str,
        *,
        runtime_identity: RuntimeIdentity | None = None,
        selected_configs_path: str | Path | None = None,
        model_key: str | None = None,
        candidate_id: str | None = None,
        resolved_config_sha256: str | None = None,
        seed: int | None = None,
        checkpoint_protocol_sha256: str | None = None,
    ) -> TestEvaluationClaim | None:
        """Authorize a split; test calls durably claim their cell before return."""

        if split not in ALLOWED_SPLITS:
            raise ProtocolError(f"unknown split: {split!r}")
        active = self.require_active()
        if split == "test" and not self.test_unlocked:
            raise ProtocolError(f"test access is locked for dataset {self.dataset_id}")
        if split != "test" and self.test_unlocked:
            raise ProtocolError(
                "train/validation access requires the active premetric protocol"
            )
        if runtime_identity is None:
            raise ProtocolError("complete RuntimeIdentity is required")
        self.require_runtime_identity(runtime_identity, requested_split=split)
        if split != "test":
            if any(
                value is not None
                for value in (
                    selected_configs_path,
                    model_key,
                    candidate_id,
                    resolved_config_sha256,
                    seed,
                    checkpoint_protocol_sha256,
                )
            ):
                raise ProtocolError(
                    "premetric split access must not provide test facts"
                )
            return None

        required_test_values = {
            "selected_configs_path": selected_configs_path,
            "model_key": model_key,
            "candidate_id": candidate_id,
            "resolved_config_sha256": resolved_config_sha256,
            "seed": seed,
            "checkpoint_protocol_sha256": checkpoint_protocol_sha256,
        }
        missing = sorted(
            name for name, value in required_test_values.items() if value is None
        )
        if missing:
            raise ProtocolError(f"test access is missing required facts: {missing}")
        assert selected_configs_path is not None
        assert model_key is not None
        assert candidate_id is not None
        assert resolved_config_sha256 is not None
        assert seed is not None
        assert checkpoint_protocol_sha256 is not None
        selected = SelectedConfigsLock.load(selected_configs_path)
        selected.require_for_protocol(self)
        selected.require_model(
            model_key=model_key,
            candidate_id=candidate_id,
            resolved_config_sha256=resolved_config_sha256,
        )
        if type(seed) is not int or seed not in FINAL_SEEDS:
            raise ProtocolError(f"final test seed must be one of {FINAL_SEEDS}")
        if checkpoint_protocol_sha256 != self.parent_premetric_digest:
            raise ProtocolError(
                "test checkpoint must remain bound to the parent premetric protocol"
            )
        claims_path = active.payload["test_claim_registry_path"]
        if not isinstance(claims_path, str):
            raise ProtocolError(
                "active final protocol lacks its central claim registry"
            )
        return TestClaimRegistry(claims_path).claim(
            gate=self,
            selected_configs_path=selected_configs_path,
            runtime_identity=runtime_identity,
            model_key=model_key,
            seed=seed,
            checkpoint_protocol_sha256=checkpoint_protocol_sha256,
            candidate_id=candidate_id,
            resolved_config_sha256=resolved_config_sha256,
        )
