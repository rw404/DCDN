"""Fail-closed authorization for the VK capacity-control learning rate.

The metric-free capacity artifact fixes the low-rank geometry, but deliberately
does not choose an optimizer learning rate.  This module verifies the separate
validation-only ordinary-DCNv2 selection receipt and materializes one compact,
post-validation/pre-test authorization for the capacity control.

The production authorization is intentionally not created at import time.  It
may be written only by :func:`materialize_capacity_lr_binding` from a clean,
committed implementation checkout and is always published with exclusive
creation semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from dcdn.capacity_artifact import (
    capacity_resolution_projection,
    validate_capacity_artifact,
    verify_repository_bindings,
)
from dcdn.protocol import load_verified_yaml_artifact
from dcdn.registry import validate_run_record
from dcdn.search_space import SearchSpaceLock, validate_search_space

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
BEST_CHECKPOINT_FILENAME_RE = re.compile(r"best-step-[0-9]{12}-generation-[0-9]{4}\.pt")

ORDINARY_SELECTION_PATH = PurePosixPath(
    "results/tuning/vk_lsvd_c_plus_p_dcnv2_full_rank_selection_v1.json"
)
CAPACITY_RESOLUTION_PATH = PurePosixPath("configs/capacity/vk_lsvd_c_plus_p_v1.json")
PREMETRIC_PROTOCOL_PATH = PurePosixPath(
    "configs/protocols/vk_lsvd_c_plus_p/protocol_frozen.yaml"
)
SEARCH_SPACE_PATH = PurePosixPath("configs/search_space_v1.yaml")
EXPERIMENT_MATRIX_PATH = PurePosixPath("configs/experiment_matrix_v1.yaml")
ENCODER_CONTRACT_PATH = PurePosixPath("configs/encoder_v1.yaml")
MODEL_SOURCES_PATH = PurePosixPath("configs/model_sources_v1.yaml")
CAPACITY_LR_BINDING_PATH = PurePosixPath(
    "artifacts/selection/vk_lsvd_c_plus_p_capacity_lr_binding_v1.json"
)
MATERIALIZER_MODULE_PATH = PurePosixPath("src/dcdn/selection_artifact.py")
MATERIALIZER_SCRIPT_PATH = PurePosixPath("scripts/materialize_capacity_lr_binding.py")

ORDINARY_SELECTION_SHA256 = (
    "869cbcb7a227b274700cda142f23c59040497bfdfe4da101888e2cc63f39d67a"
)
CAPACITY_RESOLUTION_SHA256 = (
    "d3e65d90fe0184ec48cf15abbe7358f8fd57b4944b5a95f3201b2b4e3fe81565"
)
PREMETRIC_PROTOCOL_SHA256 = (
    "1919ba14e9a1f881fdf9b45016e16c4d0b92fc71d45944d329822e28816a5a0b"
)
SEARCH_SPACE_SHA256 = "b263b55da768ec33656ebdb1506d58f2809fd5826aa8b4976aedaecee53bb344"
EXPERIMENT_MATRIX_SHA256 = (
    "867118db089bbbffb4abf02ff7b3c01e2c9f43e804a188a27a3886f66db4e000"
)
ENCODER_CONTRACT_SHA256 = (
    "16fa99db8e92ce4dd55c0bd5e7d7bfd8af8d3d59251cc769b41988e61d414449"
)
MODEL_SOURCES_SHA256 = (
    "0844a838e5a68029299a52aaa31b5f74205fe57fd486b51544061b32c571a3f4"
)
EMPTY_DIFF_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

SELECTED_ORDINARY_CANDIDATE_ID = "dcnv2_full_rank_lr_3e_4"
OTHER_ORDINARY_CANDIDATE_ID = "dcnv2_full_rank_lr_1e_3"
SELECTED_LEARNING_RATE = 0.0003
SELECTED_VALIDATION_METRIC = 0.020908477427111632
SELECTED_COMPLETION_RUN_ID = "vk-cp-tune-dcnv2-full-rank-l2-s42-lr3e4-v1-resume-v2"
SELECTED_RESULT_SHA256 = (
    "0e6381d016a65a48d9f12f48b6c871ae242139b2e9a2341f1ecba3efd03cf58e"
)
SELECTED_RESOLVED_CONFIG_SHA256 = (
    "1f1c3e7167394cc1726cdd0e1a9d00a1f5e01e27711684065ecee65a6adae454"
)
ORDINARY_MODEL_CONFIG_SHA256 = (
    "25a5e2b040fe5ff52981a8499d7d4f8dd453bdd91d24e0d5a2b40b49d7521dcc"
)
SELECTED_BEST_CHECKPOINT_SHA256 = (
    "f3a27d6522df474f1cea06abe302d289ff9fc331e9ae7135baa9589a3427618c"
)
SELECTED_COMPLETION_CODE_COMMIT = "d55d61b3a09c95045872f49893377c01f57c2d45"

CAPACITY_CANDIDATE_ID = "capacity_rank_lower"
CAPACITY_LOW_RANK = 12
CAPACITY_NUM_LAYERS = 2
CAPACITY_NUM_EXPERTS = 4
CAPACITY_PRODUCT_INIT_SCALE = 0.1

_MAX_TEXT_BYTES = 16 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024


class SelectionArtifactError(RuntimeError):
    """A selection receipt, compact authorization, or binding is invalid."""


@dataclass(frozen=True)
class OrdinarySelectionEvidence:
    """Recomputed facts needed to authorize capacity-control training."""

    candidate_id: str
    learning_rate: float
    validation_metric: float
    completion_run_id: str
    completed_result_sha256: str
    completed_resolved_config_sha256: str
    resolved_model_config_seed_invariant_sha256: str
    selected_best_checkpoint_sha256: str
    completion_code_commit: str
    completion_code_diff_sha256: str


@dataclass(frozen=True)
class CapacityBindingEvidence:
    """Verified compact authorization plus its exact byte digest."""

    payload: Mapping[str, Any]
    sha256: str
    ordinary: OrdinarySelectionEvidence


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Any, expected: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionArtifactError(f"{field} must be a mapping")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(str(key) for key in actual - expected)
        raise SelectionArtifactError(
            f"{field} keys mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SelectionArtifactError(f"{field} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, field: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise SelectionArtifactError(f"{field} must be a full lowercase Git commit")
    return value


def _require_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SelectionArtifactError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SelectionArtifactError(f"{field} must be finite")
    return result


def _require_exact_float(value: Any, field: str) -> float:
    if type(value) is not float:
        raise SelectionArtifactError(f"{field} must be an exact float")
    if not math.isfinite(value):
        raise SelectionArtifactError(f"{field} must be finite")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise SelectionArtifactError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise SelectionArtifactError(f"{field} must be a non-negative integer")
    return value


def _relative_path(value: Any, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise SelectionArtifactError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise SelectionArtifactError(f"{field} must be a canonical relative path")
    return path


def _repository_root(value: str | Path) -> Path:
    try:
        root = Path(value).resolve(strict=True)
    except OSError as exc:
        raise SelectionArtifactError("repository root is unavailable") from exc
    if not root.is_dir():
        raise SelectionArtifactError("repository root must be a directory")
    return root


def _secure_repository_file(
    repository_root: Path,
    relative_path: str | PurePosixPath,
    field: str,
) -> Path:
    relative = _relative_path(str(relative_path), field)
    current = repository_root
    try:
        for part in relative.parts:
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise SelectionArtifactError(f"{field} must not traverse a symlink")
    except FileNotFoundError as exc:
        raise SelectionArtifactError(f"{field} does not exist") from exc
    except OSError as exc:
        raise SelectionArtifactError(f"{field} could not be inspected") from exc
    if not stat.S_ISREG(current.stat(follow_symlinks=False).st_mode):
        raise SelectionArtifactError(f"{field} must be a regular file")
    return current


def _read_regular_file(path: Path, *, maximum_bytes: int, field: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SelectionArtifactError(f"{field} could not be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SelectionArtifactError(f"{field} must be a regular file")
        if metadata.st_size > maximum_bytes:
            raise SelectionArtifactError(f"{field} exceeds its bounded size")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > maximum_bytes:
            raise SelectionArtifactError(f"{field} exceeds its bounded size")
        return data
    except OSError as exc:
        raise SelectionArtifactError(f"{field} could not be read") from exc
    finally:
        os.close(descriptor)


def _reject_nonfinite_tree(value: Any, field: str = "JSON") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SelectionArtifactError(f"{field} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite_tree(child, f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite_tree(child, f"{field}[{index}]")


def _strict_json_bytes(data: bytes, field: str) -> Mapping[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelectionArtifactError(
                    f"{field} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise SelectionArtifactError(
            f"{field} contains non-finite JSON constant {value!r}"
        )

    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionArtifactError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise SelectionArtifactError(f"{field} root must be a mapping")
    _reject_nonfinite_tree(payload, field)
    return payload


def _strict_json_lines(data: bytes, field: str) -> list[Mapping[str, Any]]:
    try:
        lines = data.splitlines()
    except AttributeError as exc:
        raise SelectionArtifactError(f"{field} is not bytes") from exc
    records: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line:
            raise SelectionArtifactError(f"{field} contains an empty JSONL line")
        records.append(_strict_json_bytes(line, f"{field}:{index}"))
    if not records:
        raise SelectionArtifactError(f"{field} must not be empty")
    return records


def _verified_pair_bytes(
    repository_root: Path,
    relative_path: str | PurePosixPath,
    *,
    expected_sha256: str | None,
    field: str,
    maximum_bytes: int = _MAX_TEXT_BYTES,
) -> tuple[bytes, str]:
    relative = _relative_path(str(relative_path), field)
    path = _secure_repository_file(repository_root, relative, field)
    sidecar_relative = PurePosixPath(f"{relative}.sha256")
    sidecar = _secure_repository_file(
        repository_root,
        sidecar_relative,
        f"{field}.sidecar",
    )
    data = _read_regular_file(path, maximum_bytes=maximum_bytes, field=field)
    sidecar_data = _read_regular_file(
        sidecar,
        maximum_bytes=256,
        field=f"{field}.sidecar",
    )
    digest = sha256_bytes(data)
    expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar_data != expected_sidecar:
        raise SelectionArtifactError(f"{field} sidecar grammar or digest is invalid")
    if expected_sha256 is not None and digest != expected_sha256:
        raise SelectionArtifactError(f"{field} digest differs from its frozen binding")
    return data, digest


def _verified_json_pair(
    repository_root: Path,
    relative_path: str | PurePosixPath,
    *,
    expected_sha256: str | None,
    field: str,
) -> tuple[Mapping[str, Any], str]:
    data, digest = _verified_pair_bytes(
        repository_root,
        relative_path,
        expected_sha256=expected_sha256,
        field=field,
    )
    return _strict_json_bytes(data, field), digest


def _verified_unpaired_json(
    repository_root: Path,
    relative_path: str | PurePosixPath,
    *,
    expected_sha256: str,
    field: str,
) -> Mapping[str, Any]:
    path = _secure_repository_file(repository_root, relative_path, field)
    data = _read_regular_file(path, maximum_bytes=_MAX_TEXT_BYTES, field=field)
    if sha256_bytes(data) != expected_sha256:
        raise SelectionArtifactError(f"{field} digest mismatch")
    return _strict_json_bytes(data, field)


def _verified_history(
    repository_root: Path,
    record: Mapping[str, Any],
    field: str,
) -> None:
    path_value = record.get("path")
    digest = _require_sha256(record.get("sha256"), f"{field}.sha256")
    path = _secure_repository_file(repository_root, path_value, f"{field}.path")
    data = _read_regular_file(path, maximum_bytes=_MAX_TEXT_BYTES, field=field)
    if sha256_bytes(data) != digest:
        raise SelectionArtifactError(f"{field} digest mismatch")
    events = _strict_json_lines(data, field)
    for event in events:
        event_name = event.get("event")
        if not isinstance(event_name, str) or not event_name:
            raise SelectionArtifactError(f"{field} event name is invalid")
        split = event.get("split")
        if split == "test" or event_name.casefold().startswith("test"):
            raise SelectionArtifactError(f"{field} contains a test event")


def _load_verified_yaml_at(
    repository_root: Path,
    relative_path: str | PurePosixPath,
    *,
    expected_sha256: str,
    field: str,
) -> Mapping[str, Any]:
    source = _secure_repository_file(
        repository_root,
        relative_path,
        field,
    )
    try:
        artifact = load_verified_yaml_artifact(source)
    except Exception as exc:
        raise SelectionArtifactError(
            f"{field} YAML verification failed: {exc}"
        ) from exc
    if artifact.digest != expected_sha256:
        raise SelectionArtifactError(f"{field} digest differs from its frozen binding")
    return artifact.payload


def _candidate_evidence_records(
    candidate: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str, str
]:
    candidate_id = candidate.get("candidate_id")
    if candidate_id == OTHER_ORDINARY_CANDIDATE_ID:
        evidence = candidate.get("evidence_files")
        if not isinstance(evidence, Mapping):
            raise SelectionArtifactError("ordinary LR=0.001 evidence_files are missing")
        config = _exact_keys(
            evidence.get("resolved_config"),
            {
                "path",
                "sha256",
                "sidecar_path",
                "sidecar_strictly_verified",
                "size_bytes",
            },
            "ordinary LR=0.001 resolved_config evidence",
        )
        result = _exact_keys(
            evidence.get("result"),
            {
                "path",
                "sha256",
                "sidecar_path",
                "sidecar_strictly_verified",
                "size_bytes",
            },
            "ordinary LR=0.001 result evidence",
        )
        registry = _exact_keys(
            evidence.get("registry"),
            {
                "path",
                "sha256",
                "sidecar_present",
                "size_bytes",
                "status",
                "scientific_result_eligible",
            },
            "ordinary LR=0.001 registry evidence",
        )
        history = _exact_keys(
            evidence.get("history"),
            {
                "path",
                "sha256",
                "sidecar_present",
                "size_bytes",
                "json_lines_strictly_parsed",
                "test_events",
            },
            "ordinary LR=0.001 history evidence",
        )
        return (
            config,
            result,
            registry,
            history,
            _require_commit(
                candidate.get("code_commit"), "ordinary LR=0.001 code_commit"
            ),
            _require_sha256(
                candidate.get("code_diff_sha256"), "ordinary LR=0.001 code_diff_sha256"
            ),
        )

    if candidate_id != SELECTED_ORDINARY_CANDIDATE_ID:
        raise SelectionArtifactError("unknown ordinary DCNv2 candidate")
    lineage = candidate.get("continuation_lineage")
    if not isinstance(lineage, list):
        raise SelectionArtifactError(
            "selected candidate continuation_lineage is missing"
        )
    completed = [
        item
        for item in lineage
        if isinstance(item, Mapping)
        and item.get("attempt_role")
        == "second_exact_resume_completed_scientific_result"
    ]
    if len(completed) != 1:
        raise SelectionArtifactError(
            "selected candidate must have one completed lineage"
        )
    record = completed[0]
    config = {
        "path": record.get("resolved_config_path"),
        "sha256": record.get("resolved_config_sha256"),
    }
    result = {
        "path": record.get("result_path"),
        "sha256": record.get("result_sha256"),
    }
    registry = {
        "path": record.get("registry_path"),
        "sha256": record.get("registry_sha256"),
        "status": record.get("registry_status"),
        "scientific_result_eligible": record.get("registry_scientific_result_eligible"),
    }
    history = candidate.get("history")
    if not isinstance(history, Mapping):
        raise SelectionArtifactError("selected candidate history evidence is missing")
    code_diffs = _exact_keys(
        candidate.get("code_diff_sha256_by_attempt"),
        {"initial", "resume_v1", "resume_v2"},
        "selected candidate code_diff_sha256_by_attempt",
    )
    return (
        config,
        result,
        registry,
        history,
        _require_commit(record.get("code_commit"), "selected completion code_commit"),
        _require_sha256(
            code_diffs.get("resume_v2"),
            "selected completion code_diff_sha256",
        ),
    )


def _validate_candidate_evidence(
    candidate: Mapping[str, Any],
    repository_root: Path,
) -> tuple[float, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str, str]:
    candidate_id = candidate.get("candidate_id")
    expected_lr = {
        OTHER_ORDINARY_CANDIDATE_ID: 0.001,
        SELECTED_ORDINARY_CANDIDATE_ID: SELECTED_LEARNING_RATE,
    }.get(candidate_id)
    if (
        expected_lr is None
        or type(candidate.get("learning_rate")) is not float
        or candidate.get("learning_rate") != expected_lr
    ):
        raise SelectionArtifactError("ordinary candidate ID/learning-rate pair drifted")
    if candidate.get("execution_kind") != "scientific_tuning":
        raise SelectionArtifactError("ordinary candidate execution kind drifted")
    if candidate.get("scientific_result_eligible") is not True:
        raise SelectionArtifactError(
            "ordinary candidate is not scientifically eligible"
        )
    if candidate.get("test_metrics") is not None:
        raise SelectionArtifactError("ordinary candidate contains test metrics")
    if (
        candidate.get("resolved_model_config_seed_invariant_sha256")
        != ORDINARY_MODEL_CONFIG_SHA256
    ):
        raise SelectionArtifactError("ordinary seed-invariant model digest drifted")

    summary = _exact_keys(
        candidate.get("training_summary"),
        {
            "best_validation_metric",
            "declared_maximum_train_examples",
            "examples_seen",
            "nonfinite_total",
            "oom_count",
            "optimizer_steps",
            "realized_maximum_train_examples",
            "stopped_reason",
            "validation_checks",
        },
        f"{candidate_id}.training_summary",
    )
    metric = _require_exact_float(
        summary.get("best_validation_metric"),
        f"{candidate_id}.best_validation_metric",
    )
    for name in (
        "declared_maximum_train_examples",
        "examples_seen",
        "optimizer_steps",
        "realized_maximum_train_examples",
        "validation_checks",
    ):
        _require_positive_int(summary.get(name), f"{candidate_id}.{name}")
    for name in ("nonfinite_total", "oom_count"):
        if _require_nonnegative_int(summary.get(name), f"{candidate_id}.{name}") != 0:
            raise SelectionArtifactError(
                "ordinary candidate has numerical/OOM failures"
            )
    if type(summary.get("stopped_reason")) is not str or not summary["stopped_reason"]:
        raise SelectionArtifactError("ordinary candidate stopped_reason is invalid")

    (
        config_record,
        result_record,
        registry_record,
        history_record,
        code_commit,
        code_diff,
    ) = _candidate_evidence_records(candidate)
    config_digest = _require_sha256(config_record.get("sha256"), "config evidence SHA")
    config_path = _relative_path(config_record.get("path"), "config evidence path")
    config = _load_verified_yaml_at(
        repository_root,
        config_path,
        expected_sha256=config_digest,
        field=f"{candidate_id}.resolved_config",
    )
    result_digest = _require_sha256(result_record.get("sha256"), "result evidence SHA")
    result_path = _relative_path(result_record.get("path"), "result evidence path")
    result, _ = _verified_json_pair(
        repository_root,
        result_path,
        expected_sha256=result_digest,
        field=f"{candidate_id}.result",
    )
    registry_digest = _require_sha256(
        registry_record.get("sha256"), "registry evidence SHA"
    )
    registry_path = _relative_path(
        registry_record.get("path"), "registry evidence path"
    )
    registry = _verified_unpaired_json(
        repository_root,
        registry_path,
        expected_sha256=registry_digest,
        field=f"{candidate_id}.registry",
    )
    try:
        validate_run_record(registry)
    except Exception as exc:
        raise SelectionArtifactError(
            f"terminal registry validation failed: {exc}"
        ) from exc
    if not isinstance(history_record, Mapping):
        raise SelectionArtifactError("history evidence must be a mapping")
    _verified_history(repository_root, history_record, f"{candidate_id}.history")

    run_id = candidate.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise SelectionArtifactError("candidate run_id is invalid")
    training = config.get("training")
    model = config.get("model")
    if not isinstance(training, Mapping) or not isinstance(model, Mapping):
        raise SelectionArtifactError("resolved candidate config is malformed")
    interaction = model.get("interaction_config")
    if not isinstance(interaction, Mapping):
        raise SelectionArtifactError("resolved interaction config is malformed")
    if type(config.get("schema_version")) is not int:
        raise SelectionArtifactError("resolved config schema_version must be exact int")
    if type(training.get("learning_rate")) is not float:
        raise SelectionArtifactError("resolved learning_rate must be exact float")
    if type(training.get("seed")) is not int:
        raise SelectionArtifactError("resolved seed must be exact int")
    if type(interaction.get("num_layers")) is not int:
        raise SelectionArtifactError("resolved num_layers must be exact int")
    if type(interaction.get("product_init_scale")) is not float:
        raise SelectionArtifactError("resolved product_init_scale must be exact float")
    if (
        config.get("schema_version") != 3
        or config.get("execution_kind") != "scientific_tuning"
        or config.get("run_id") != run_id
        or training.get("dataset_id") != "vk_lsvd_up0.01_ir0.01"
        or training.get("feature_regime") != "C+P"
        or training.get("model_key") != "dcnv2_full_rank"
        or training.get("candidate_id") != candidate_id
        or training.get("learning_rate") != expected_lr
        or training.get("seed") != 42
        or training.get("phase") != "tuning"
        or training.get("resolved_model_config_sha256") != ORDINARY_MODEL_CONFIG_SHA256
        or training.get("capacity_resolution_sha256") != CAPACITY_RESOLUTION_SHA256
        or training.get("search_space_sha256") != SEARCH_SPACE_SHA256
        or training.get("experiment_matrix_sha256") != EXPERIMENT_MATRIX_SHA256
        or training.get("encoder_contract_sha256") != ENCODER_CONTRACT_SHA256
        or interaction
        != {
            "model_key": "dcnv2_full_rank",
            "num_layers": 2,
            "product_init_scale": 0.1,
        }
    ):
        raise SelectionArtifactError("resolved ordinary candidate config drifted")

    result_training = result.get("training")
    if not isinstance(result_training, Mapping):
        raise SelectionArtifactError("ordinary result training payload is malformed")
    if (
        type(result.get("schema_version")) is not int
        or type(result.get("selected_validation_metric")) is not float
        or type(result_training.get("best_validation_metric")) is not float
    ):
        raise SelectionArtifactError("ordinary result scalar types drifted")
    for name in ("validation_checks", "examples_seen", "optimizer_steps"):
        _require_positive_int(result_training.get(name), f"result.training.{name}")
    if (
        result.get("schema_version") != 3
        or result.get("execution_kind") != "scientific_tuning"
        or result.get("scientific_result_eligible") is not True
        or result.get("run_id") != run_id
        or result.get("resolved_config_sha256") != config_digest
        or result.get("protocol_sha256") != PREMETRIC_PROTOCOL_SHA256
        or result.get("capacity_resolution_sha256") != CAPACITY_RESOLUTION_SHA256
        or result.get("test_metrics") is not None
        or result.get("selected_validation_metric") != metric
        or result_training.get("best_validation_metric") != metric
        or result_training.get("validation_checks") != summary.get("validation_checks")
        or result_training.get("examples_seen") != summary.get("examples_seen")
        or result_training.get("optimizer_steps") != summary.get("optimizer_steps")
        or result_training.get("oom_count") != 0
    ):
        raise SelectionArtifactError("completed ordinary result drifted")
    validation_checks = result_training.get("validation_checks")
    best_metric = result_training.get("best_validation_metric")
    best_sha256 = result_training.get("best_checkpoint_sha256")
    best_filename = result_training.get("best_checkpoint_filename")
    if (
        type(validation_checks) is not int
        or validation_checks <= 0
        or type(best_metric) is not float
        or not math.isfinite(best_metric)
        or best_metric != metric
        or not isinstance(best_sha256, str)
        or SHA256_RE.fullmatch(best_sha256) is None
        or not isinstance(best_filename, str)
        or BEST_CHECKPOINT_FILENAME_RE.fullmatch(best_filename) is None
    ):
        raise SelectionArtifactError(
            "completed ordinary result is not scientifically eligible"
        )
    lineage = candidate.get("checkpoint_lineage")
    generations = (
        lineage.get("best_generations") if isinstance(lineage, Mapping) else None
    )
    matching_best = (
        [
            item
            for item in generations
            if isinstance(item, Mapping)
            and PurePosixPath(str(item.get("path"))).name == best_filename
            and item.get("sha256") == best_sha256
            and item.get("selected_within_candidate") is True
            and item.get("validation_metric_at_creation") == metric
        ]
        if isinstance(generations, list)
        else []
    )
    if len(matching_best) != 1:
        raise SelectionArtifactError(
            "result best checkpoint does not match candidate lineage"
        )

    numerical = result_training.get("numerical_summary")
    if (
        not isinstance(numerical, Mapping)
        or type(numerical.get("nonfinite_total")) is not int
        or numerical.get("nonfinite_total") != 0
    ):
        raise SelectionArtifactError(
            "completed ordinary result is numerically ineligible"
        )

    if (
        registry.get("status") != "completed"
        or registry.get("scientific_result_eligible") is not True
        or registry.get("run_id") != run_id
        or registry.get("seed") != 42
        or registry.get("resolved_config_path") != str(config_path)
        or registry.get("resolved_config_sha256") != config_digest
        or registry.get("protocol_sha256") != PREMETRIC_PROTOCOL_SHA256
        or registry.get("code_commit") != code_commit
        or registry.get("code_diff_sha256") != code_diff
    ):
        raise SelectionArtifactError("terminal ordinary registry drifted")
    if (
        registry_record.get("status") != "completed"
        or registry_record.get("scientific_result_eligible") is not True
    ):
        raise SelectionArtifactError("receipt terminal registry claim drifted")
    return metric, config, result, registry, code_commit, code_diff


def validate_ordinary_dcnv2_selection(
    payload: Mapping[str, Any],
    repository_root: str | Path,
) -> OrdinarySelectionEvidence:
    """Recompute the strict two-candidate validation-only selection."""

    root = _repository_root(repository_root)
    receipt = _exact_keys(
        payload,
        {
            "artifact_role",
            "candidates",
            "created_utc",
            "dataset_id",
            "feature_regime",
            "frozen_identities",
            "integrity_audit",
            "remaining_gates",
            "retention_inventory",
            "schema_version",
            "selected",
            "selection_comparison",
            "selection_rule",
            "selection_scope",
            "status",
            "test_firewall",
            "tuning_unit",
        },
        "ordinary selection receipt",
    )
    if (
        type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
        or receipt.get("artifact_role")
        != "per_model_validation_only_selection_evidence"
        or receipt.get("status") != "selected_on_validation_only"
        or receipt.get("dataset_id") != "vk_lsvd_up0.01_ir0.01"
        or receipt.get("feature_regime") != "C+P"
        or receipt.get("selection_scope")
        != "one_model_only_not_dataset_selected_configs_lock"
    ):
        raise SelectionArtifactError("ordinary selection receipt identity drifted")
    tuning_unit = _exact_keys(
        receipt.get("tuning_unit"),
        {"dataset", "feature_regime", "model", "primary_target"},
        "ordinary selection tuning_unit",
    )
    if dict(tuning_unit) != {
        "dataset": "vk_lsvd_up0.01_ir0.01",
        "feature_regime": "C+P",
        "model": "dcnv2_full_rank",
        "primary_target": "normalized_watch_time",
    }:
        raise SelectionArtifactError("ordinary tuning unit drifted")
    rule = _exact_keys(
        receipt.get("selection_rule"),
        {
            "candidate_budget",
            "direction",
            "independent_selection_per_tuning_unit",
            "metric",
            "source",
            "source_sha256",
            "tuning_seed",
            "tuning_splits",
        },
        "ordinary selection rule",
    )
    if (
        type(rule.get("candidate_budget")) is not int
        or type(rule.get("tuning_seed")) is not int
        or type(rule.get("independent_selection_per_tuning_unit")) is not bool
        or type(rule.get("tuning_splits")) is not list
    ):
        raise SelectionArtifactError("ordinary selection rule scalar types drifted")
    if dict(rule) != {
        "candidate_budget": 2,
        "direction": "minimize",
        "independent_selection_per_tuning_unit": True,
        "metric": "normalized_mse",
        "source": "configs/search_space_v1.yaml",
        "source_sha256": SEARCH_SPACE_SHA256,
        "tuning_seed": 42,
        "tuning_splits": ["train", "validation"],
    }:
        raise SelectionArtifactError("ordinary selection rule drifted")
    firewall = _exact_keys(
        receipt.get("test_firewall"),
        {
            "combined_history_test_event_count",
            "protocol_test_path_available",
            "protocol_test_unlocked",
            "selected_on_validation_only",
            "test_data_accessed",
            "test_metrics_consulted",
            "test_metrics_in_candidate_results",
        },
        "ordinary selection test_firewall",
    )
    firewall_booleans = (
        "protocol_test_path_available",
        "protocol_test_unlocked",
        "selected_on_validation_only",
        "test_data_accessed",
        "test_metrics_consulted",
    )
    if type(firewall.get("combined_history_test_event_count")) is not int or any(
        type(firewall.get(name)) is not bool for name in firewall_booleans
    ):
        raise SelectionArtifactError("ordinary selection firewall scalar types drifted")
    if dict(firewall) != {
        "combined_history_test_event_count": 0,
        "protocol_test_path_available": False,
        "protocol_test_unlocked": False,
        "selected_on_validation_only": True,
        "test_data_accessed": False,
        "test_metrics_consulted": False,
        "test_metrics_in_candidate_results": None,
    }:
        raise SelectionArtifactError("ordinary selection test firewall drifted")

    candidates = receipt.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise SelectionArtifactError(
            "ordinary selection requires exactly two candidates"
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise SelectionArtifactError("ordinary candidate must be a mapping")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in by_id:
            raise SelectionArtifactError("ordinary candidate IDs must be unique")
        by_id[candidate_id] = candidate
    if set(by_id) != {
        OTHER_ORDINARY_CANDIDATE_ID,
        SELECTED_ORDINARY_CANDIDATE_ID,
    }:
        raise SelectionArtifactError("ordinary candidate set drifted")

    evaluated: dict[
        str,
        tuple[float, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str, str],
    ] = {}
    for candidate_id, candidate in by_id.items():
        evaluated[candidate_id] = _validate_candidate_evidence(candidate, root)
    selected_metric = evaluated[SELECTED_ORDINARY_CANDIDATE_ID][0]
    other_metric = evaluated[OTHER_ORDINARY_CANDIDATE_ID][0]
    if selected_metric == other_metric:
        raise SelectionArtifactError("ordinary validation metrics tie without a rule")
    strict_winner = min(evaluated, key=lambda item: evaluated[item][0])
    if strict_winner != SELECTED_ORDINARY_CANDIDATE_ID:
        raise SelectionArtifactError(
            "receipt-selected candidate is not the strict minimum"
        )
    if (
        by_id[SELECTED_ORDINARY_CANDIDATE_ID].get("selected") is not True
        or by_id[OTHER_ORDINARY_CANDIDATE_ID].get("selected") is not False
    ):
        raise SelectionArtifactError("ordinary candidate selected flags drifted")

    comparison = _exact_keys(
        receipt.get("selection_comparison"),
        {
            "absolute_other_minus_selected",
            "general_equal_metric_tie_break_predeclared",
            "no_post_metric_tie_break_invented",
            "other_candidate_id",
            "other_validation_metric",
            "relative_reduction_against_other_percent",
            "selected_candidate_id",
            "selected_validation_metric",
            "tie_break_applied",
            "tie_occurred",
        },
        "ordinary selection comparison",
    )
    comparison_floats = (
        "selected_validation_metric",
        "other_validation_metric",
        "absolute_other_minus_selected",
        "relative_reduction_against_other_percent",
    )
    comparison_booleans = (
        "tie_occurred",
        "tie_break_applied",
        "general_equal_metric_tie_break_predeclared",
        "no_post_metric_tie_break_invented",
    )
    if any(
        type(comparison.get(name)) is not float for name in comparison_floats
    ) or any(type(comparison.get(name)) is not bool for name in comparison_booleans):
        raise SelectionArtifactError(
            "ordinary selection comparison scalar types drifted"
        )
    if (
        comparison.get("selected_candidate_id") != SELECTED_ORDINARY_CANDIDATE_ID
        or comparison.get("other_candidate_id") != OTHER_ORDINARY_CANDIDATE_ID
        or comparison.get("selected_validation_metric") != selected_metric
        or comparison.get("other_validation_metric") != other_metric
        or comparison.get("absolute_other_minus_selected") != 3.027318395658e-06
        or comparison.get("relative_reduction_against_other_percent")
        != 0.014476808017884993
        or comparison.get("tie_occurred") is not False
        or comparison.get("tie_break_applied") is not False
        or comparison.get("general_equal_metric_tie_break_predeclared") is not False
        or comparison.get("no_post_metric_tie_break_invented") is not True
    ):
        raise SelectionArtifactError("ordinary selection comparison drifted")

    selected = _exact_keys(
        receipt.get("selected"),
        {
            "best_checkpoint_path",
            "best_checkpoint_sha256",
            "candidate_id",
            "completion_run_id",
            "learning_rate",
            "resolved_config_path",
            "resolved_config_sha256",
            "resolved_model_config_seed_invariant_sha256",
            "result_path",
            "result_sha256",
            "validation_metric",
        },
        "ordinary selected record",
    )
    if (
        type(selected.get("learning_rate")) is not float
        or type(selected.get("validation_metric")) is not float
    ):
        raise SelectionArtifactError("ordinary selected scalar types drifted")
    selected_candidate = by_id[SELECTED_ORDINARY_CANDIDATE_ID]
    selected_evaluation = evaluated[SELECTED_ORDINARY_CANDIDATE_ID]
    selected_config = selected_evaluation[1]
    selected_result = selected_evaluation[2]
    selected_registry = selected_evaluation[3]
    code_commit = selected_evaluation[4]
    code_diff = selected_evaluation[5]
    if (
        selected.get("candidate_id") != SELECTED_ORDINARY_CANDIDATE_ID
        or selected.get("learning_rate") != SELECTED_LEARNING_RATE
        or selected.get("validation_metric") != selected_metric
        or selected.get("completion_run_id") != selected_candidate.get("run_id")
        or selected.get("resolved_config_sha256")
        != selected_result.get("resolved_config_sha256")
        or selected.get("result_sha256") != SELECTED_RESULT_SHA256
        or selected.get("resolved_model_config_seed_invariant_sha256")
        != ORDINARY_MODEL_CONFIG_SHA256
        or selected_registry.get("code_commit") != code_commit
        or selected_registry.get("code_diff_sha256") != code_diff
    ):
        raise SelectionArtifactError("ordinary selected record drifted")
    if selected.get("completion_run_id") != SELECTED_COMPLETION_RUN_ID:
        raise SelectionArtifactError("selected completion run ID drifted")
    if selected.get("resolved_config_sha256") != SELECTED_RESOLVED_CONFIG_SHA256:
        raise SelectionArtifactError("selected resolved-config digest drifted")
    if selected_config.get("run_id") != SELECTED_COMPLETION_RUN_ID:
        raise SelectionArtifactError("selected resolved config run ID drifted")
    if code_commit != SELECTED_COMPLETION_CODE_COMMIT or code_diff != EMPTY_DIFF_SHA256:
        raise SelectionArtifactError("selected completion code identity drifted")

    checkpoint_path = _relative_path(
        selected.get("best_checkpoint_path"),
        "ordinary selected checkpoint path",
    )
    checkpoint_sha = _require_sha256(
        selected.get("best_checkpoint_sha256"),
        "ordinary selected checkpoint SHA",
    )
    _, verified_checkpoint_sha = _verified_pair_bytes(
        root,
        checkpoint_path,
        expected_sha256=checkpoint_sha,
        field="ordinary selected checkpoint",
        maximum_bytes=_MAX_CHECKPOINT_BYTES,
    )
    if checkpoint_sha != SELECTED_BEST_CHECKPOINT_SHA256:
        raise SelectionArtifactError("selected checkpoint digest drifted")
    lineage = selected_candidate.get("checkpoint_lineage")
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("checkpoint_metadata_protocol_hash") != PREMETRIC_PROTOCOL_SHA256
    ):
        raise SelectionArtifactError("selected checkpoint protocol lineage drifted")
    generations = lineage.get("best_generations")
    if not isinstance(generations, list):
        raise SelectionArtifactError("selected best-generation lineage is missing")
    generation_matches = [
        item
        for item in generations
        if isinstance(item, Mapping)
        and item.get("path") == str(checkpoint_path)
        and item.get("sha256") == checkpoint_sha
        and item.get("selected_within_candidate") is True
        and item.get("selected_across_candidates") is True
        and item.get("validation_metric_at_creation") == selected_metric
    ]
    if len(generation_matches) != 1:
        raise SelectionArtifactError("selected checkpoint generation lineage drifted")

    if selected_metric != SELECTED_VALIDATION_METRIC:
        raise SelectionArtifactError("selected validation metric drifted")
    return OrdinarySelectionEvidence(
        candidate_id=SELECTED_ORDINARY_CANDIDATE_ID,
        learning_rate=SELECTED_LEARNING_RATE,
        validation_metric=selected_metric,
        completion_run_id=SELECTED_COMPLETION_RUN_ID,
        completed_result_sha256=SELECTED_RESULT_SHA256,
        completed_resolved_config_sha256=SELECTED_RESOLVED_CONFIG_SHA256,
        resolved_model_config_seed_invariant_sha256=ORDINARY_MODEL_CONFIG_SHA256,
        selected_best_checkpoint_sha256=verified_checkpoint_sha,
        completion_code_commit=code_commit,
        completion_code_diff_sha256=code_diff,
    )


def load_ordinary_selection_receipt(
    repository_root: str | Path,
) -> tuple[Mapping[str, Any], OrdinarySelectionEvidence]:
    """Load the immutable human/audit receipt and recompute its selection."""

    root = _repository_root(repository_root)
    payload, digest = _verified_json_pair(
        root,
        ORDINARY_SELECTION_PATH,
        expected_sha256=ORDINARY_SELECTION_SHA256,
        field="ordinary selection receipt",
    )
    if digest != ORDINARY_SELECTION_SHA256:
        raise SelectionArtifactError("ordinary selection receipt digest drifted")
    return payload, validate_ordinary_dcnv2_selection(payload, root)


def _verify_frozen_inputs(
    repository_root: Path,
) -> tuple[Mapping[str, Any], SearchSpaceLock]:
    matrix_payload = _load_verified_yaml_at(
        repository_root,
        EXPERIMENT_MATRIX_PATH,
        expected_sha256=EXPERIMENT_MATRIX_SHA256,
        field="experiment matrix",
    )
    encoder_payload = _load_verified_yaml_at(
        repository_root,
        ENCODER_CONTRACT_PATH,
        expected_sha256=ENCODER_CONTRACT_SHA256,
        field="encoder contract",
    )
    model_sources_payload = _load_verified_yaml_at(
        repository_root,
        MODEL_SOURCES_PATH,
        expected_sha256=MODEL_SOURCES_SHA256,
        field="model sources",
    )
    search_payload = _load_verified_yaml_at(
        repository_root,
        SEARCH_SPACE_PATH,
        expected_sha256=SEARCH_SPACE_SHA256,
        field="search space",
    )
    try:
        validate_search_space(
            search_payload,
            model_sources=model_sources_payload,
        )
    except Exception as exc:
        raise SelectionArtifactError(
            f"search-space verification failed: {exc}"
        ) from exc
    search = SearchSpaceLock(
        path=repository_root / SEARCH_SPACE_PATH,
        digest=SEARCH_SPACE_SHA256,
        payload=search_payload,
        model_sources_path=repository_root / MODEL_SOURCES_PATH,
        model_sources_digest=MODEL_SOURCES_SHA256,
        model_sources_payload=model_sources_payload,
        experiment_matrix_path=repository_root / EXPERIMENT_MATRIX_PATH,
        experiment_matrix_digest=EXPERIMENT_MATRIX_SHA256,
    )
    del matrix_payload, encoder_payload
    protocol = _load_verified_yaml_at(
        repository_root,
        PREMETRIC_PROTOCOL_PATH,
        expected_sha256=PREMETRIC_PROTOCOL_SHA256,
        field="premetric protocol",
    )
    hashes = protocol.get("hashes")
    if (
        protocol.get("schema_version") != 3
        or protocol.get("dataset_id") != "vk_lsvd_up0.01_ir0.01"
        or protocol.get("feature_regime") != "C+P"
        or protocol.get("test_unlocked") is not False
        or protocol.get("test_path_available") is not False
        or not isinstance(hashes, Mapping)
        or hashes.get("search_space_sha256") != SEARCH_SPACE_SHA256
        or hashes.get("experiment_matrix_sha256") != EXPERIMENT_MATRIX_SHA256
        or hashes.get("encoder_contract_sha256") != ENCODER_CONTRACT_SHA256
        or hashes.get("capacity_resolution_sha256") != CAPACITY_RESOLUTION_SHA256
    ):
        raise SelectionArtifactError("premetric protocol binding/firewall drifted")
    capacity_data, _ = _verified_pair_bytes(
        repository_root,
        CAPACITY_RESOLUTION_PATH,
        expected_sha256=CAPACITY_RESOLUTION_SHA256,
        field="capacity resolution",
    )
    capacity = _strict_json_bytes(capacity_data, "capacity resolution")
    try:
        validate_capacity_artifact(capacity)
        verify_repository_bindings(capacity, repository_root)
    except Exception as exc:
        raise SelectionArtifactError(
            f"capacity artifact verification failed: {exc}"
        ) from exc
    _verified_pair_bytes(
        repository_root,
        PurePosixPath("configs/data/vk_lsvd/features_v1.yaml"),
        expected_sha256=(
            "380d105c847a73afb4927740e40401124688f895142792309b4fc8267a2a5f2b"
        ),
        field="VK feature contract",
    )
    geometry = capacity.get("geometry")
    decision = capacity.get("decision")
    firewall = capacity.get("metric_firewall")
    if (
        not isinstance(geometry, Mapping)
        or not isinstance(decision, Mapping)
        or not isinstance(firewall, Mapping)
        or decision.get("selected_candidate") != CAPACITY_CANDIDATE_ID
        or decision.get("selected_low_rank") != CAPACITY_LOW_RANK
        or geometry.get("capacity_num_layers") != CAPACITY_NUM_LAYERS
        or geometry.get("capacity_num_experts") != CAPACITY_NUM_EXPERTS
        or any(value is not False for value in firewall.values())
    ):
        raise SelectionArtifactError("metric-free capacity decision drifted")
    return capacity, search


def capacity_phase_resolutions(
    capacity_payload: Mapping[str, Any],
    *,
    selected_learning_rate: float,
) -> dict[str, Any]:
    """Build the existing search-space phase context without changing its lock."""

    geometry = capacity_payload.get("geometry")
    bindings = capacity_payload.get("frozen_bindings")
    if not isinstance(geometry, Mapping) or not isinstance(bindings, Mapping):
        raise SelectionArtifactError("capacity geometry/bindings are malformed")
    return {
        "dimensions": {
            "DATASET_MODEL_FIELD_COUNT": (
                geometry["ordered_context_field_count"] + geometry["operand_width"]
            ),
            "DATASET_FLAT_WIDTH": geometry["flat_width"],
            "DATASET_CORE_OPERAND_WIDTH": geometry["operand_width"],
            "DATASET_CONTEXT_BYPASS_WIDTH": geometry["context_bypass_width"],
            "DATASET_DENOMINATOR_MASK_WIDTH": geometry["operand_width"],
        },
        "hashes": {
            "DATASET_FULL_SCHEMA_SHA256": (
                "2cfd6d12cd85346e413f476ff223e19d57be213d74fe68b75d6f0ce686c0f091"
            ),
            "DATASET_OPERAND_SCHEMA_SHA256": bindings["vk_operand_schema_sha256"],
        },
        "hyperparameters": {
            "SELECTED_DCNV2_LEARNING_RATE": selected_learning_rate,
        },
        "capacity_resolution": capacity_resolution_projection(capacity_payload),
    }


def _git(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[:500]
        raise SelectionArtifactError(f"Git identity command failed: {detail}")
    return completed.stdout


def _reject_untracked_code(repository_root: Path) -> None:
    untracked = _git(
        repository_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "src",
        "scripts",
    )
    for raw_path in untracked.split(b"\0"):
        if not raw_path:
            continue
        try:
            relative = Path(raw_path.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise SelectionArtifactError("untracked code path is not UTF-8") from exc
        candidate = repository_root / relative
        if candidate.suffix in {".py", ".pyi", ".pyc", ".so", ".sh"} or (
            candidate.exists() and os.access(candidate, os.X_OK)
        ):
            raise SelectionArtifactError(
                f"untracked executable/importable source is forbidden: {relative}"
            )


def _materializer_code_identity(repository_root: Path) -> dict[str, str]:
    try:
        commit = (
            _git(repository_root, "rev-parse", "--verify", "HEAD")
            .decode("ascii")
            .strip()
        )
    except UnicodeDecodeError as exc:
        raise SelectionArtifactError("materializer Git HEAD is not ASCII") from exc
    _require_commit(commit, "materializer code_commit")
    diff_digest = sha256_bytes(_git(repository_root, "diff", "--binary", "HEAD", "--"))
    if diff_digest != EMPTY_DIFF_SHA256:
        raise SelectionArtifactError("materializer requires a clean tracked worktree")
    _reject_untracked_code(repository_root)
    for path in (MATERIALIZER_MODULE_PATH, MATERIALIZER_SCRIPT_PATH):
        _secure_repository_file(repository_root, path, f"materializer source {path}")
        _git(repository_root, "ls-files", "--error-unmatch", str(path))
        committed = _git(repository_root, "show", f"{commit}:{path}")
        current = _read_regular_file(
            repository_root / path,
            maximum_bytes=_MAX_TEXT_BYTES,
            field=f"materializer source {path}",
        )
        if committed != current:
            raise SelectionArtifactError(
                f"materializer source differs from clean commit: {path}"
            )
    return {"code_commit": commit, "code_diff_sha256": diff_digest}


def _validate_materializer_code_identity(
    repository_root: Path,
    payload: Mapping[str, Any],
) -> None:
    identity = _exact_keys(
        payload,
        {"code_commit", "code_diff_sha256"},
        "materializer_code_identity",
    )
    commit = _require_commit(identity.get("code_commit"), "materializer code_commit")
    if identity.get("code_diff_sha256") != EMPTY_DIFF_SHA256:
        raise SelectionArtifactError("materializer code diff must be empty")
    _git(repository_root, "cat-file", "-e", f"{commit}^{{commit}}")
    for path in (MATERIALIZER_MODULE_PATH, MATERIALIZER_SCRIPT_PATH):
        _git(repository_root, "cat-file", "-e", f"{commit}:{path}")


def _capacity_binding_payload(
    ordinary: OrdinarySelectionEvidence,
    materializer_identity: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_role": "capacity_control_training_authorization",
        "status": "authorized_after_validation_only_selection",
        "identity": {
            "dataset_id": "vk_lsvd_up0.01_ir0.01",
            "feature_regime": "C+P",
            "primary_target": "normalized_watch_time",
            "ordinary_model_key": "dcnv2_full_rank",
            "capacity_model_key": "dcnv2_capacity_matched_core",
            "tuning_seed": 42,
        },
        "frozen_bindings": {
            "ordinary_selection_path": str(ORDINARY_SELECTION_PATH),
            "ordinary_selection_sha256": ORDINARY_SELECTION_SHA256,
            "capacity_resolution_path": str(CAPACITY_RESOLUTION_PATH),
            "capacity_resolution_sha256": CAPACITY_RESOLUTION_SHA256,
            "premetric_protocol_path": str(PREMETRIC_PROTOCOL_PATH),
            "premetric_protocol_sha256": PREMETRIC_PROTOCOL_SHA256,
            "search_space_path": str(SEARCH_SPACE_PATH),
            "search_space_sha256": SEARCH_SPACE_SHA256,
            "experiment_matrix_sha256": EXPERIMENT_MATRIX_SHA256,
            "encoder_contract_sha256": ENCODER_CONTRACT_SHA256,
            "model_sources_sha256": MODEL_SOURCES_SHA256,
        },
        "ordinary_dcnv2_selection": {
            "candidate_id": ordinary.candidate_id,
            "learning_rate": ordinary.learning_rate,
            "validation_metric": ordinary.validation_metric,
            "completion_run_id": ordinary.completion_run_id,
            "completed_result_sha256": ordinary.completed_result_sha256,
            "completed_resolved_config_sha256": (
                ordinary.completed_resolved_config_sha256
            ),
            "resolved_model_config_seed_invariant_sha256": (
                ordinary.resolved_model_config_seed_invariant_sha256
            ),
            "selected_best_checkpoint_sha256": (
                ordinary.selected_best_checkpoint_sha256
            ),
            "completion_code_commit": ordinary.completion_code_commit,
            "completion_code_diff_sha256": ordinary.completion_code_diff_sha256,
        },
        "capacity_control": {
            "candidate_id": CAPACITY_CANDIDATE_ID,
            "low_rank": CAPACITY_LOW_RANK,
            "num_layers": CAPACITY_NUM_LAYERS,
            "num_experts": CAPACITY_NUM_EXPERTS,
            "learning_rate": SELECTED_LEARNING_RATE,
            "rank_selection_source": "metric_free_capacity_artifact",
            "learning_rate_selection_source": (
                "ordinary_dcnv2_validation_only_selection"
            ),
        },
        "materializer_code_identity": dict(materializer_identity),
        "test_firewall": {
            "selected_on_validation_only": True,
            "test_unlocked": False,
            "test_path_available": False,
            "test_data_accessed": False,
            "test_metrics_consulted": False,
            "validation_metrics_used_for_capacity_rank": False,
        },
    }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SelectionArtifactError(
            "artifact is not canonical finite ASCII JSON"
        ) from exc


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("short exclusive artifact write")
        offset += written


def _exclusive_pair(
    repository_root: Path,
    relative_path: str | PurePosixPath,
    payload: Mapping[str, Any],
) -> str:
    """Publish through held no-follow directory descriptors with O_EXCL."""

    root = _repository_root(repository_root)
    relative = _relative_path(str(relative_path), "authorization output path")
    data = _canonical_json(payload)
    digest = sha256_bytes(data)
    filename = relative.name
    sidecar_filename = f"{filename}.sha256"
    sidecar_data = f"{digest}  {filename}\n".encode("ascii")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise SelectionArtifactError("secure no-follow publication is unavailable")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
    )
    file_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | nofollow
    )
    descriptors: list[int] = []
    created: list[str] = []
    directory_descriptor = -1
    try:
        directory_descriptor = os.open(root, directory_flags)
        descriptors.append(directory_descriptor)
        for part in relative.parent.parts:
            try:
                child_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o750, dir_fd=directory_descriptor)
                    os.fsync(directory_descriptor)
                except FileExistsError:
                    pass
                child_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            metadata = os.fstat(child_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child_descriptor)
                raise SelectionArtifactError(
                    "authorization output ancestor is not a directory"
                )
            descriptors.append(child_descriptor)
            directory_descriptor = child_descriptor
        for target, body in (
            (filename, data),
            (sidecar_filename, sidecar_data),
        ):
            descriptor = os.open(
                target,
                file_flags,
                0o640,
                dir_fd=directory_descriptor,
            )
            created.append(target)
            try:
                _write_all(descriptor, body)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(directory_descriptor)
    except FileExistsError as exc:
        for target in reversed(created):
            with suppress(OSError):
                os.unlink(target, dir_fd=directory_descriptor)
        if directory_descriptor >= 0:
            with suppress(OSError):
                os.fsync(directory_descriptor)
        raise SelectionArtifactError(
            "immutable capacity authorization already exists"
        ) from exc
    except SelectionArtifactError:
        for target in reversed(created):
            with suppress(OSError):
                os.unlink(target, dir_fd=directory_descriptor)
        if directory_descriptor >= 0:
            with suppress(OSError):
                os.fsync(directory_descriptor)
        raise
    except OSError as exc:
        for target in reversed(created):
            with suppress(OSError):
                os.unlink(target, dir_fd=directory_descriptor)
        if directory_descriptor >= 0:
            with suppress(OSError):
                os.fsync(directory_descriptor)
        raise SelectionArtifactError(
            "exclusive capacity authorization publication failed"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
    return digest


def validate_capacity_lr_binding(
    payload: Mapping[str, Any],
    repository_root: str | Path,
) -> OrdinarySelectionEvidence:
    """Validate every compact field against independently reopened evidence."""

    root = _repository_root(repository_root)
    artifact = _exact_keys(
        payload,
        {
            "schema_version",
            "artifact_role",
            "status",
            "identity",
            "frozen_bindings",
            "ordinary_dcnv2_selection",
            "capacity_control",
            "materializer_code_identity",
            "test_firewall",
        },
        "capacity LR binding",
    )
    if (
        type(artifact.get("schema_version")) is not int
        or artifact.get("schema_version") != 1
        or artifact.get("artifact_role") != "capacity_control_training_authorization"
        or artifact.get("status") != "authorized_after_validation_only_selection"
    ):
        raise SelectionArtifactError("capacity LR binding root identity drifted")
    identity = _exact_keys(
        artifact.get("identity"),
        {
            "dataset_id",
            "feature_regime",
            "primary_target",
            "ordinary_model_key",
            "capacity_model_key",
            "tuning_seed",
        },
        "capacity LR binding identity",
    )
    if type(identity.get("tuning_seed")) is not int:
        raise SelectionArtifactError("capacity LR identity scalar types drifted")

    if dict(identity) != {
        "dataset_id": "vk_lsvd_up0.01_ir0.01",
        "feature_regime": "C+P",
        "primary_target": "normalized_watch_time",
        "ordinary_model_key": "dcnv2_full_rank",
        "capacity_model_key": "dcnv2_capacity_matched_core",
        "tuning_seed": 42,
    }:
        raise SelectionArtifactError("capacity LR binding identity drifted")
    frozen = _exact_keys(
        artifact.get("frozen_bindings"),
        {
            "ordinary_selection_path",
            "ordinary_selection_sha256",
            "capacity_resolution_path",
            "capacity_resolution_sha256",
            "premetric_protocol_path",
            "premetric_protocol_sha256",
            "search_space_path",
            "search_space_sha256",
            "experiment_matrix_sha256",
            "encoder_contract_sha256",
            "model_sources_sha256",
        },
        "capacity LR frozen_bindings",
    )
    expected_frozen = {
        "ordinary_selection_path": str(ORDINARY_SELECTION_PATH),
        "ordinary_selection_sha256": ORDINARY_SELECTION_SHA256,
        "capacity_resolution_path": str(CAPACITY_RESOLUTION_PATH),
        "capacity_resolution_sha256": CAPACITY_RESOLUTION_SHA256,
        "premetric_protocol_path": str(PREMETRIC_PROTOCOL_PATH),
        "premetric_protocol_sha256": PREMETRIC_PROTOCOL_SHA256,
        "search_space_path": str(SEARCH_SPACE_PATH),
        "search_space_sha256": SEARCH_SPACE_SHA256,
        "experiment_matrix_sha256": EXPERIMENT_MATRIX_SHA256,
        "encoder_contract_sha256": ENCODER_CONTRACT_SHA256,
        "model_sources_sha256": MODEL_SOURCES_SHA256,
    }
    if dict(frozen) != expected_frozen:
        raise SelectionArtifactError("capacity LR frozen bindings drifted")
    _, ordinary = load_ordinary_selection_receipt(root)
    capacity, search = _verify_frozen_inputs(root)
    try:
        resolved_search = search.require_phase(
            "capacity_control_training",
            resolutions=capacity_phase_resolutions(
                capacity,
                selected_learning_rate=ordinary.learning_rate,
            ),
        )
    except Exception as exc:
        raise SelectionArtifactError(
            f"capacity search phase gate failed: {exc}"
        ) from exc
    inherited = resolved_search["models"]["dcnv2_low_rank_mix_capacity"]["base_config"][
        "optimizer"
    ]["learning_rate"]
    if inherited != SELECTED_LEARNING_RATE:
        raise SelectionArtifactError("search phase did not resolve the selected LR")

    ordinary_payload = _exact_keys(
        artifact.get("ordinary_dcnv2_selection"),
        {
            "candidate_id",
            "learning_rate",
            "validation_metric",
            "completion_run_id",
            "completed_result_sha256",
            "completed_resolved_config_sha256",
            "resolved_model_config_seed_invariant_sha256",
            "selected_best_checkpoint_sha256",
            "completion_code_commit",
            "completion_code_diff_sha256",
        },
        "capacity LR ordinary_dcnv2_selection",
    )
    if (
        type(ordinary_payload.get("learning_rate")) is not float
        or type(ordinary_payload.get("validation_metric")) is not float
    ):
        raise SelectionArtifactError("compact ordinary selection scalar types drifted")
    if dict(ordinary_payload) != {
        "candidate_id": ordinary.candidate_id,
        "learning_rate": ordinary.learning_rate,
        "validation_metric": ordinary.validation_metric,
        "completion_run_id": ordinary.completion_run_id,
        "completed_result_sha256": ordinary.completed_result_sha256,
        "completed_resolved_config_sha256": (ordinary.completed_resolved_config_sha256),
        "resolved_model_config_seed_invariant_sha256": (
            ordinary.resolved_model_config_seed_invariant_sha256
        ),
        "selected_best_checkpoint_sha256": (ordinary.selected_best_checkpoint_sha256),
        "completion_code_commit": ordinary.completion_code_commit,
        "completion_code_diff_sha256": ordinary.completion_code_diff_sha256,
    }:
        raise SelectionArtifactError("compact ordinary selection fields drifted")
    control = _exact_keys(
        artifact.get("capacity_control"),
        {
            "candidate_id",
            "low_rank",
            "num_layers",
            "num_experts",
            "learning_rate",
            "rank_selection_source",
            "learning_rate_selection_source",
        },
        "capacity LR capacity_control",
    )
    if (
        type(control.get("low_rank")) is not int
        or type(control.get("num_layers")) is not int
        or type(control.get("num_experts")) is not int
        or type(control.get("learning_rate")) is not float
    ):
        raise SelectionArtifactError("compact capacity-control scalar types drifted")
    if dict(control) != {
        "candidate_id": CAPACITY_CANDIDATE_ID,
        "low_rank": CAPACITY_LOW_RANK,
        "num_layers": CAPACITY_NUM_LAYERS,
        "num_experts": CAPACITY_NUM_EXPERTS,
        "learning_rate": SELECTED_LEARNING_RATE,
        "rank_selection_source": "metric_free_capacity_artifact",
        "learning_rate_selection_source": ("ordinary_dcnv2_validation_only_selection"),
    }:
        raise SelectionArtifactError("compact capacity-control fields drifted")
    firewall = _exact_keys(
        artifact.get("test_firewall"),
        {
            "selected_on_validation_only",
            "test_unlocked",
            "test_path_available",
            "test_data_accessed",
            "test_metrics_consulted",
            "validation_metrics_used_for_capacity_rank",
        },
        "capacity LR test_firewall",
    )
    if any(type(value) is not bool for value in firewall.values()):
        raise SelectionArtifactError("compact test firewall scalar types drifted")
    if dict(firewall) != {
        "selected_on_validation_only": True,
        "test_unlocked": False,
        "test_path_available": False,
        "test_data_accessed": False,
        "test_metrics_consulted": False,
        "validation_metrics_used_for_capacity_rank": False,
    }:
        raise SelectionArtifactError("compact test firewall drifted")
    _validate_materializer_code_identity(
        root,
        _exact_keys(
            artifact.get("materializer_code_identity"),
            {"code_commit", "code_diff_sha256"},
            "materializer_code_identity",
        ),
    )
    return ordinary


def materialize_capacity_lr_binding(
    repository_root: str | Path,
    output_path: str | Path | None = None,
) -> CapacityBindingEvidence:
    """Create the compact authorization once, from a clean implementation HEAD."""

    root = _repository_root(repository_root)
    target = root / CAPACITY_LR_BINDING_PATH
    if output_path is not None:
        requested = Path(output_path)
        if not requested.is_absolute():
            requested = root / requested
        try:
            requested_parent = requested.parent.resolve(strict=True)
        except OSError as exc:
            raise SelectionArtifactError(
                "authorization output parent is unavailable"
            ) from exc
        requested = requested_parent / requested.name
        if requested != target:
            raise SelectionArtifactError(
                "capacity authorization output path is fixed by the contract"
            )
    _, ordinary = load_ordinary_selection_receipt(root)
    _verify_frozen_inputs(root)
    code_identity = _materializer_code_identity(root)
    payload = _capacity_binding_payload(ordinary, code_identity)
    validate_capacity_lr_binding(payload, root)
    digest = _exclusive_pair(root, CAPACITY_LR_BINDING_PATH, payload)
    loaded = load_capacity_lr_binding(target, root)
    if loaded.sha256 != digest or loaded.payload != payload:
        raise SelectionArtifactError(
            "materialized capacity authorization failed exact reopen"
        )
    return loaded


def load_capacity_lr_binding(
    path: str | Path,
    repository_root: str | Path,
) -> CapacityBindingEvidence:
    """Load the fixed compact artifact after sidecar and evidence verification."""

    root = _repository_root(repository_root)
    supplied = Path(path)
    if not supplied.is_absolute():
        supplied = root / supplied
    try:
        parent = supplied.parent.resolve(strict=True)
    except OSError as exc:
        raise SelectionArtifactError(
            "capacity authorization parent is unavailable"
        ) from exc
    supplied = parent / supplied.name
    expected = root / CAPACITY_LR_BINDING_PATH
    if supplied != expected:
        raise SelectionArtifactError("capacity authorization path is not canonical")
    payload, digest = _verified_json_pair(
        root,
        CAPACITY_LR_BINDING_PATH,
        expected_sha256=None,
        field="capacity LR binding",
    )
    ordinary = validate_capacity_lr_binding(payload, root)
    return CapacityBindingEvidence(payload=payload, sha256=digest, ordinary=ordinary)


def capacity_selection_binding_payload(
    *,
    capacity_candidate_id: str,
    learning_rate: float,
    capacity_resolution_sha256: str,
    ordinary_dcnv2_selection_sha256: str,
    capacity_lr_binding_sha256: str,
    premetric_protocol_sha256: str,
) -> dict[str, Any]:
    """Return the exact binding nested in the capacity selected-model digest."""

    for value, field in (
        (capacity_resolution_sha256, "capacity_resolution_sha256"),
        (ordinary_dcnv2_selection_sha256, "ordinary_dcnv2_selection_sha256"),
        (capacity_lr_binding_sha256, "capacity_lr_binding_sha256"),
        (premetric_protocol_sha256, "premetric_protocol_sha256"),
    ):
        _require_sha256(value, field)
    if not isinstance(capacity_candidate_id, str) or not capacity_candidate_id:
        raise SelectionArtifactError("capacity_candidate_id must be non-empty")
    _require_exact_float(learning_rate, "learning_rate")
    return {
        "capacity_candidate_id": capacity_candidate_id,
        "learning_rate": learning_rate,
        "capacity_resolution_sha256": capacity_resolution_sha256,
        "ordinary_dcnv2_selection_sha256": ordinary_dcnv2_selection_sha256,
        "capacity_lr_binding_sha256": capacity_lr_binding_sha256,
        "premetric_protocol_sha256": premetric_protocol_sha256,
    }


def capacity_resolved_model_config_sha256(
    *,
    interaction_config: Mapping[str, Any],
    head_config: Mapping[str, Any],
    capacity_selection_binding: Mapping[str, Any],
) -> str:
    """Hash the capacity model geometry and post-validation authorization."""

    binding = _exact_keys(
        capacity_selection_binding,
        {
            "capacity_candidate_id",
            "learning_rate",
            "capacity_resolution_sha256",
            "ordinary_dcnv2_selection_sha256",
            "capacity_lr_binding_sha256",
            "premetric_protocol_sha256",
        },
        "capacity_selection_binding",
    )
    # Re-run scalar/digest validation rather than treating the mapping as trusted.
    canonical_binding = capacity_selection_binding_payload(
        capacity_candidate_id=binding["capacity_candidate_id"],
        learning_rate=binding["learning_rate"],
        capacity_resolution_sha256=binding["capacity_resolution_sha256"],
        ordinary_dcnv2_selection_sha256=binding["ordinary_dcnv2_selection_sha256"],
        capacity_lr_binding_sha256=binding["capacity_lr_binding_sha256"],
        premetric_protocol_sha256=binding["premetric_protocol_sha256"],
    )
    payload = {
        "schema_version": 1,
        "interaction_config": dict(interaction_config),
        "head_config": dict(head_config),
        "capacity_selection_binding": canonical_binding,
    }
    try:
        data = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SelectionArtifactError(
            "capacity selected-model payload is not canonical JSON"
        ) from exc
    return sha256_bytes(data)


def capacity_training_binding_payload(
    *,
    capacity_resolution_sha256: str,
    ordinary_dcnv2_selection_sha256: str,
    capacity_lr_binding_sha256: str,
) -> dict[str, Any]:
    """Return the exact capacity binding embedded in result/checkpoint evidence."""

    for value, field in (
        (capacity_resolution_sha256, "capacity_resolution_sha256"),
        (ordinary_dcnv2_selection_sha256, "ordinary_dcnv2_selection_sha256"),
        (capacity_lr_binding_sha256, "capacity_lr_binding_sha256"),
    ):
        _require_sha256(value, field)
    return {
        "capacity_candidate_id": CAPACITY_CANDIDATE_ID,
        "selected_low_rank": CAPACITY_LOW_RANK,
        "inherited_from_candidate_id": SELECTED_ORDINARY_CANDIDATE_ID,
        "learning_rate": SELECTED_LEARNING_RATE,
        "capacity_resolution_sha256": capacity_resolution_sha256,
        "ordinary_dcnv2_selection_sha256": ordinary_dcnv2_selection_sha256,
        "capacity_lr_binding_sha256": capacity_lr_binding_sha256,
    }


def validate_capacity_training_binding_payload(
    payload: Any,
    *,
    capacity_resolution_sha256: str,
    ordinary_dcnv2_selection_sha256: str,
    capacity_lr_binding_sha256: str,
) -> Mapping[str, Any]:
    """Validate exact scalar types and values in retained capacity evidence."""

    binding = _exact_keys(
        payload,
        {
            "capacity_candidate_id",
            "selected_low_rank",
            "inherited_from_candidate_id",
            "learning_rate",
            "capacity_resolution_sha256",
            "ordinary_dcnv2_selection_sha256",
            "capacity_lr_binding_sha256",
        },
        "capacity_training_binding",
    )
    expected = capacity_training_binding_payload(
        capacity_resolution_sha256=capacity_resolution_sha256,
        ordinary_dcnv2_selection_sha256=ordinary_dcnv2_selection_sha256,
        capacity_lr_binding_sha256=capacity_lr_binding_sha256,
    )
    if (
        type(binding.get("selected_low_rank")) is not int
        or type(binding.get("learning_rate")) is not float
    ):
        raise SelectionArtifactError("capacity training binding scalar types drifted")
    if dict(binding) != expected:
        raise SelectionArtifactError("capacity training binding values drifted")
    return binding


def iter_leaf_paths(
    value: Any, prefix: tuple[str, ...] = ()
) -> Iterable[tuple[str, ...]]:
    """Yield mapping leaf paths; useful for exhaustive schema mutation tests."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_leaf_paths(child, (*prefix, str(key)))
    else:
        yield prefix
