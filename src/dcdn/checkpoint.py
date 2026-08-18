"""Strict, integrity-checked training checkpoints for deterministic resume."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .reproducibility import (
    RNGStateError,
    capture_rng_state,
    restore_rng_state,
    validate_rng_state,
)
from .sampler import SamplerStateError, StatefulPermutationSampler

CHECKPOINT_FORMAT = "dcdn.training-checkpoint"
CHECKPOINT_VERSION = 1
SHA256_SUFFIX = ".sha256"


class CheckpointError(RuntimeError):
    """Base error for checkpoint operations."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when a checkpoint or its sidecar fails integrity checks."""


class CheckpointFormatError(CheckpointError):
    """Raised when a verified checkpoint has an invalid payload schema."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised before resume when runtime identity does not match the checkpoint."""


def _normalise_json(value: Any, *, path: str = "$") -> Any:
    """Return a detached JSON-safe value while rejecting lossy coercions."""

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            result[key] = _normalise_json(child, path=f"{path}.{key}")
        return result
    if isinstance(value, list | tuple):
        return [
            _normalise_json(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    raise TypeError(f"{path} contains unsupported JSON value {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value canonically for resolved-config identity hashing."""

    normalised = _normalise_json(value)
    return json.dumps(
        normalised,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_resolved_config(resolved_config: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a canonical, fully resolved configuration."""

    if not isinstance(resolved_config, Mapping):
        raise TypeError("resolved_config must be a mapping")
    return hashlib.sha256(canonical_json_bytes(resolved_config)).hexdigest()


def _validate_digest(value: str, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    if value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class CheckpointMetadata:
    """Frozen identities that must match before any training state is mutated."""

    resolved_config: Mapping[str, Any]
    data_hash: str
    feature_hash: str
    protocol_hash: str
    feature_order: Sequence[str]
    denominator_mask: Sequence[bool]
    div_eps: float
    depth: int
    precision: str
    config_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resolved_config, Mapping):
            raise TypeError("resolved_config must be a mapping")
        normalised_config = _normalise_json(self.resolved_config)
        object.__setattr__(self, "resolved_config", normalised_config)

        computed_config_hash = hash_resolved_config(normalised_config)
        if self.config_hash is None:
            object.__setattr__(self, "config_hash", computed_config_hash)
        else:
            _validate_digest(self.config_hash, field="config_hash")
            if self.config_hash != computed_config_hash:
                raise ValueError(
                    "config_hash does not match canonical resolved_config content"
                )

        for field_name in ("data_hash", "feature_hash", "protocol_hash"):
            _validate_digest(getattr(self, field_name), field=field_name)

        feature_order = tuple(self.feature_order)
        if not feature_order or any(
            not isinstance(name, str) or not name for name in feature_order
        ):
            raise ValueError("feature_order must contain non-empty strings")
        if len(set(feature_order)) != len(feature_order):
            raise ValueError("feature_order must not contain duplicates")
        object.__setattr__(self, "feature_order", feature_order)

        denominator_mask = tuple(self.denominator_mask)
        if len(denominator_mask) != len(feature_order):
            raise ValueError("denominator_mask must align exactly with feature_order")
        if any(type(value) is not bool for value in denominator_mask):
            raise TypeError("denominator_mask must contain booleans")
        object.__setattr__(self, "denominator_mask", denominator_mask)

        if type(self.div_eps) not in (float, int) or isinstance(self.div_eps, bool):
            raise TypeError("div_eps must be a finite positive number")
        div_eps = float(self.div_eps)
        if not math.isfinite(div_eps) or div_eps <= 0:
            raise ValueError("div_eps must be a finite positive number")
        object.__setattr__(self, "div_eps", div_eps)

        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise TypeError("depth must be an integer")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if not isinstance(self.precision, str) or not self.precision.strip():
            raise ValueError("precision must be a non-empty string")
        object.__setattr__(self, "precision", self.precision.strip().lower())

    def to_dict(self) -> dict[str, Any]:
        if hash_resolved_config(self.resolved_config) != self.config_hash:
            raise ValueError(
                "resolved_config was mutated after CheckpointMetadata creation"
            )
        return {
            "resolved_config": _normalise_json(self.resolved_config),
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "feature_hash": self.feature_hash,
            "protocol_hash": self.protocol_hash,
            "feature_order": list(self.feature_order),
            "denominator_mask": list(self.denominator_mask),
            "div_eps": self.div_eps,
            "depth": self.depth,
            "precision": self.precision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointMetadata:
        if not isinstance(value, Mapping):
            raise CheckpointFormatError("checkpoint metadata must be a mapping")
        expected = {
            "resolved_config",
            "config_hash",
            "data_hash",
            "feature_hash",
            "protocol_hash",
            "feature_order",
            "denominator_mask",
            "div_eps",
            "depth",
            "precision",
        }
        actual = set(value)
        if actual != expected:
            raise CheckpointFormatError(
                "checkpoint metadata keys mismatch; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        if value["config_hash"] is None:
            raise CheckpointFormatError("checkpoint metadata config_hash is missing")
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            raise CheckpointFormatError(f"invalid checkpoint metadata: {exc}") from exc

    def mismatch_fields(self, expected: CheckpointMetadata) -> list[str]:
        saved = self.to_dict()
        wanted = expected.to_dict()
        return [key for key in saved if saved[key] != wanted[key]]


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    sidecar_path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ResumeState:
    global_step: int
    epoch: int
    accumulation_step: int
    accumulation_steps: int
    mode: str
    sha256: str
    extra_state: Mapping[str, Any]


def _component_name(component: Any) -> str:
    cls = component.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}{SHA256_SUFFIX}")


def _sha256_open_file(file_object: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = file_object.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EROFS):
                raise
    finally:
        os.close(descriptor)


def _atomic_write_checkpoint(
    path: Path, payload: Mapping[str, Any]
) -> CheckpointRecord:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_path(path)
    checkpoint_temp: Path | None = None
    sidecar_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            checkpoint_temp = Path(temporary.name)
            torch.save(dict(payload), temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary.seek(0)
            digest = _sha256_open_file(temporary)
            size_bytes = temporary.tell()

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_sidecar:
            sidecar_temp = Path(temporary_sidecar.name)
            temporary_sidecar.write(f"{digest}  {path.name}\n")
            temporary_sidecar.flush()
            os.fsync(temporary_sidecar.fileno())

        os.replace(checkpoint_temp, path)
        checkpoint_temp = None
        _fsync_directory(path.parent)
        os.replace(sidecar_temp, sidecar)
        sidecar_temp = None
        _fsync_directory(path.parent)
        return CheckpointRecord(path, sidecar, digest, size_bytes)
    finally:
        for temporary_path in (checkpoint_temp, sidecar_temp):
            if temporary_path is not None:
                with suppress(FileNotFoundError):
                    temporary_path.unlink()


def _read_sidecar(path: Path) -> str:
    sidecar = _sidecar_path(path)
    try:
        text = sidecar.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise CheckpointIntegrityError(
            f"checkpoint SHA-256 sidecar is unavailable: {sidecar}"
        ) from exc
    parts = text.strip().split()
    if len(parts) != 2 or parts[1] != path.name:
        raise CheckpointIntegrityError("malformed checkpoint SHA-256 sidecar")
    try:
        return _validate_digest(parts[0], field="checkpoint sha256")
    except ValueError as exc:
        raise CheckpointIntegrityError(str(exc)) from exc


def _load_verified_payload(
    path: Path, *, map_location: str | torch.device | Mapping[str, str] = "cpu"
) -> tuple[Mapping[str, Any], str]:
    path = path.absolute()
    expected_digest = _read_sidecar(path)
    try:
        with path.open("rb") as checkpoint_file:
            actual_digest = _sha256_open_file(checkpoint_file)
            if actual_digest != expected_digest:
                raise CheckpointIntegrityError(
                    "checkpoint SHA-256 mismatch: "
                    f"expected={expected_digest}, actual={actual_digest}"
                )
            checkpoint_file.seek(0)
            try:
                payload = torch.load(
                    checkpoint_file,
                    map_location=map_location,
                    weights_only=True,
                )
            except Exception as exc:
                raise CheckpointFormatError(
                    "verified checkpoint could not be safely deserialized"
                ) from exc
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"checkpoint is unavailable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointFormatError("checkpoint payload must be a mapping")
    return payload, actual_digest


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise CheckpointFormatError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointFormatError(f"{field} must be a non-negative integer")
    return value


def _validate_module_state(model: nn.Module, state: Any) -> None:
    if not isinstance(state, Mapping):
        raise CheckpointFormatError("model state must be a mapping")
    current = model.state_dict()
    saved_keys = set(state)
    current_keys = set(current)
    if saved_keys != current_keys:
        raise CheckpointCompatibilityError(
            "model state keys mismatch; "
            f"missing={sorted(current_keys - saved_keys)}, "
            f"extra={sorted(saved_keys - current_keys)}"
        )
    for name, current_value in current.items():
        saved_value = state[name]
        if isinstance(current_value, torch.Tensor):
            if not isinstance(saved_value, torch.Tensor):
                raise CheckpointCompatibilityError(
                    f"model state {name!r} is not a tensor"
                )
            if saved_value.shape != current_value.shape:
                raise CheckpointCompatibilityError(
                    f"model state shape mismatch for {name!r}: "
                    f"checkpoint={tuple(saved_value.shape)}, "
                    f"current={tuple(current_value.shape)}"
                )
            if saved_value.dtype != current_value.dtype:
                raise CheckpointCompatibilityError(
                    f"model state dtype mismatch for {name!r}: "
                    f"checkpoint={saved_value.dtype}, current={current_value.dtype}"
                )
        elif type(saved_value) is not type(current_value):
            raise CheckpointCompatibilityError(
                f"model extra state type mismatch for {name!r}"
            )


def _validate_optimizer_state(optimizer: torch.optim.Optimizer, state: Any) -> None:
    if not isinstance(state, Mapping):
        raise CheckpointFormatError("optimizer state must be a mapping")
    if set(state) != {"state", "param_groups"}:
        raise CheckpointFormatError("optimizer state has unexpected keys")
    saved_groups = state["param_groups"]
    current_groups = optimizer.state_dict()["param_groups"]
    if not isinstance(saved_groups, list) or len(saved_groups) != len(current_groups):
        raise CheckpointCompatibilityError("optimizer parameter-group count mismatch")
    paired_groups = zip(saved_groups, current_groups, strict=False)
    for index, (saved, current) in enumerate(paired_groups):
        if not isinstance(saved, Mapping) or "params" not in saved:
            raise CheckpointFormatError(
                f"optimizer parameter group {index} is malformed"
            )
        if len(saved["params"]) != len(current["params"]):
            raise CheckpointCompatibilityError(
                f"optimizer parameter count mismatch in group {index}"
            )
    saved_state = state["state"]
    if not isinstance(saved_state, Mapping):
        raise CheckpointFormatError("optimizer per-parameter state must be a mapping")
    valid_parameter_ids = {
        parameter_id for group in saved_groups for parameter_id in group["params"]
    }
    if not set(saved_state).issubset(valid_parameter_ids):
        raise CheckpointFormatError("optimizer state references an unknown parameter")


def _validate_optional_component(
    *,
    label: str,
    saved: Any,
    current: Any,
) -> None:
    if saved is None:
        if current is not None:
            raise CheckpointCompatibilityError(
                f"checkpoint has no {label}, but the runtime supplied one"
            )
        return
    if current is None:
        raise CheckpointCompatibilityError(
            f"checkpoint contains {label}, but the runtime did not supply one"
        )
    if not isinstance(saved, Mapping) or set(saved) != {"type", "state"}:
        raise CheckpointFormatError(f"{label} component is malformed")
    if saved["type"] != _component_name(current):
        raise CheckpointCompatibilityError(
            f"{label} type mismatch: checkpoint={saved['type']!r}, "
            f"current={_component_name(current)!r}"
        )
    if not isinstance(saved["state"], Mapping):
        raise CheckpointFormatError(f"{label} state must be a mapping")
    # Scheduler keys are stable after construction and can be checked without
    # applying state. GradScaler may be intentionally lazy and have an empty
    # runtime state before its first scale(), so it cannot use the same check.
    if label == "scheduler":
        current_state = current.state_dict()
        if not isinstance(current_state, Mapping):
            raise CheckpointCompatibilityError(
                "runtime scheduler state_dict is not a mapping"
            )
        if set(saved["state"]) != set(current_state):
            raise CheckpointCompatibilityError(
                "scheduler state keys mismatch; "
                f"missing={sorted(set(current_state) - set(saved['state']))}, "
                f"extra={sorted(set(saved['state']) - set(current_state))}"
            )


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    sampler: StatefulPermutationSampler,
    scaler: Any | None,
    global_step: int,
    epoch: int,
    accumulation_step: int,
    accumulation_steps: int,
    metadata: CheckpointMetadata,
    mode: str,
    extra_state: Mapping[str, Any] | None = None,
    include_cuda_rng: bool = True,
) -> CheckpointRecord:
    """Atomically save a checkpoint at a completed accumulation boundary.

    Mid-accumulation checkpoints are intentionally rejected: silently omitting
    accumulated gradients would make the resumed optimizer trajectory differ.
    Call this function after ``optimizer.step`` and the corresponding sampler
    commit, with ``accumulation_step=0``.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    if not isinstance(sampler, StatefulPermutationSampler):
        raise TypeError("sampler must be a StatefulPermutationSampler")
    if not isinstance(metadata, CheckpointMetadata):
        raise TypeError("metadata must be CheckpointMetadata")
    if type(include_cuda_rng) is not bool:
        raise TypeError("include_cuda_rng must be boolean")
    global_step = _validate_nonnegative_integer(global_step, field="global_step")
    epoch = _validate_nonnegative_integer(epoch, field="epoch")
    accumulation_step = _validate_nonnegative_integer(
        accumulation_step, field="accumulation_step"
    )
    if (
        isinstance(accumulation_steps, bool)
        or not isinstance(accumulation_steps, int)
        or accumulation_steps <= 0
    ):
        raise ValueError("accumulation_steps must be a positive integer")
    if accumulation_step != 0:
        raise CheckpointCompatibilityError(
            "checkpoint save is allowed only at an accumulation boundary "
            "(accumulation_step must be zero)"
        )
    if epoch != sampler.epoch:
        raise CheckpointCompatibilityError(
            "epoch does not match sampler state: "
            f"epoch={epoch}, sampler={sampler.epoch}"
        )
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("mode must be a non-empty string")

    normalised_extra = _normalise_json(extra_state or {})
    scheduler_component = (
        None
        if scheduler is None
        else {"type": _component_name(scheduler), "state": scheduler.state_dict()}
    )
    scaler_component = (
        None
        if scaler is None
        else {"type": _component_name(scaler), "state": scaler.state_dict()}
    )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "metadata": metadata.to_dict(),
        "progress": {
            "global_step": global_step,
            "epoch": epoch,
            "accumulation_step": accumulation_step,
            "accumulation_steps": accumulation_steps,
            "accumulation_boundary": True,
            "mode": mode.strip(),
            "model_training": bool(model.training),
        },
        "components": {
            "model": {"type": _component_name(model), "state": model.state_dict()},
            "optimizer": {
                "type": _component_name(optimizer),
                "state": optimizer.state_dict(),
            },
            "scheduler": scheduler_component,
            "sampler": {
                "type": _component_name(sampler),
                "state": sampler.state_dict(),
            },
            "scaler": scaler_component,
        },
        "rng_state": capture_rng_state(include_cuda=include_cuda_rng),
        "extra_state": normalised_extra,
    }
    return _atomic_write_checkpoint(Path(path), payload)


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    sampler: StatefulPermutationSampler,
    scaler: Any | None,
    expected_metadata: CheckpointMetadata,
    expected_accumulation_steps: int | None = None,
    map_location: str | torch.device | Mapping[str, str] = "cpu",
) -> ResumeState:
    """Strictly validate then restore a training checkpoint.

    Integrity, format, frozen metadata, component types, state shapes, sampler,
    and RNG compatibility are all checked before model/optimizer mutation.  RNG
    restoration is deliberately the final operation.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    if not isinstance(sampler, StatefulPermutationSampler):
        raise TypeError("sampler must be a StatefulPermutationSampler")
    if not isinstance(expected_metadata, CheckpointMetadata):
        raise TypeError("expected_metadata must be CheckpointMetadata")
    if expected_accumulation_steps is not None:
        if isinstance(expected_accumulation_steps, bool) or not isinstance(
            expected_accumulation_steps, int
        ):
            raise TypeError("expected_accumulation_steps must be an integer")
        if expected_accumulation_steps <= 0:
            raise ValueError("expected_accumulation_steps must be positive")

    payload, digest = _load_verified_payload(Path(path), map_location=map_location)
    _require_exact_keys(
        payload,
        {
            "format",
            "version",
            "metadata",
            "progress",
            "components",
            "rng_state",
            "extra_state",
        },
        label="checkpoint payload",
    )
    if payload["format"] != CHECKPOINT_FORMAT:
        raise CheckpointFormatError(
            f"unsupported checkpoint format: {payload['format']!r}"
        )
    if payload["version"] != CHECKPOINT_VERSION:
        raise CheckpointFormatError(
            f"unsupported checkpoint version: {payload['version']!r}"
        )

    saved_metadata = CheckpointMetadata.from_dict(payload["metadata"])
    mismatches = saved_metadata.mismatch_fields(expected_metadata)
    if mismatches:
        raise CheckpointCompatibilityError(
            "frozen checkpoint metadata mismatch: " + ", ".join(mismatches)
        )

    progress = payload["progress"]
    if not isinstance(progress, Mapping):
        raise CheckpointFormatError("checkpoint progress must be a mapping")
    _require_exact_keys(
        progress,
        {
            "global_step",
            "epoch",
            "accumulation_step",
            "accumulation_steps",
            "accumulation_boundary",
            "mode",
            "model_training",
        },
        label="checkpoint progress",
    )
    global_step = _validate_nonnegative_integer(
        progress["global_step"], field="global_step"
    )
    epoch = _validate_nonnegative_integer(progress["epoch"], field="epoch")
    accumulation_step = _validate_nonnegative_integer(
        progress["accumulation_step"], field="accumulation_step"
    )
    accumulation_steps = progress["accumulation_steps"]
    if (
        isinstance(accumulation_steps, bool)
        or not isinstance(accumulation_steps, int)
        or accumulation_steps <= 0
    ):
        raise CheckpointFormatError("accumulation_steps must be a positive integer")
    if progress["accumulation_boundary"] is not True or accumulation_step != 0:
        raise CheckpointCompatibilityError(
            "checkpoint is not at a completed accumulation boundary"
        )
    if (
        expected_accumulation_steps is not None
        and accumulation_steps != expected_accumulation_steps
    ):
        raise CheckpointCompatibilityError(
            "gradient accumulation mismatch: "
            f"checkpoint={accumulation_steps}, current={expected_accumulation_steps}"
        )
    mode = progress["mode"]
    if not isinstance(mode, str) or not mode.strip() or mode != mode.strip():
        raise CheckpointFormatError("checkpoint mode must be a non-empty string")
    if type(progress["model_training"]) is not bool:
        raise CheckpointFormatError("model_training must be boolean")

    components = payload["components"]
    if not isinstance(components, Mapping):
        raise CheckpointFormatError("checkpoint components must be a mapping")
    _require_exact_keys(
        components,
        {"model", "optimizer", "scheduler", "sampler", "scaler"},
        label="checkpoint components",
    )
    required_components = (
        ("model", model),
        ("optimizer", optimizer),
        ("sampler", sampler),
    )
    for label, current in required_components:
        component = components[label]
        if not isinstance(component, Mapping) or set(component) != {"type", "state"}:
            raise CheckpointFormatError(f"{label} component is malformed")
        if component["type"] != _component_name(current):
            raise CheckpointCompatibilityError(
                f"{label} type mismatch: checkpoint={component['type']!r}, "
                f"current={_component_name(current)!r}"
            )

    _validate_optional_component(
        label="scheduler", saved=components["scheduler"], current=scheduler
    )
    _validate_optional_component(
        label="scaler", saved=components["scaler"], current=scaler
    )
    _validate_module_state(model, components["model"]["state"])
    _validate_optimizer_state(optimizer, components["optimizer"]["state"])
    try:
        sampler.validate_state_dict(components["sampler"]["state"])
    except SamplerStateError as exc:
        raise CheckpointCompatibilityError(f"invalid sampler state: {exc}") from exc
    if components["sampler"]["state"]["epoch"] != epoch:
        raise CheckpointCompatibilityError(
            "checkpoint epoch does not match its sampler epoch"
        )
    try:
        validate_rng_state(payload["rng_state"])
    except RNGStateError as exc:
        raise CheckpointCompatibilityError(f"invalid RNG state: {exc}") from exc
    try:
        extra_state = _normalise_json(payload["extra_state"])
    except (TypeError, ValueError) as exc:
        raise CheckpointFormatError(f"invalid checkpoint extra_state: {exc}") from exc

    # All fail-closed checks above are mutation-free. These state_dict loads are
    # expected not to fail after structural validation. RNG is restored last.
    model.load_state_dict(components["model"]["state"], strict=True)
    optimizer.load_state_dict(components["optimizer"]["state"])
    if scheduler is not None:
        scheduler.load_state_dict(components["scheduler"]["state"])
    if scaler is not None:
        scaler.load_state_dict(components["scaler"]["state"])
    sampler.load_state_dict(components["sampler"]["state"])
    model.train(progress["model_training"])
    restore_rng_state(payload["rng_state"])

    return ResumeState(
        global_step=global_step,
        epoch=epoch,
        accumulation_step=accumulation_step,
        accumulation_steps=accumulation_steps,
        mode=mode,
        sha256=digest,
        extra_state=extra_state,
    )
