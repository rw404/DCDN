"""Typed, dataset-agnostic contracts for controlled interaction models.

The dataset encoder is intentionally outside this package. It must produce
the same ordered examples and the views below from one frozen feature schema.
This keeps the unresolved categorical/numerical encoder boundary explicit.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import torch

_ALLOWED_DTYPES = (torch.float32, torch.bfloat16)


def require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def require_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def require_probability(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return value


def require_finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def require_finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def require_dims(name: str, value: tuple[int, ...], *, allow_empty: bool) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of positive integers")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    for index, width in enumerate(value):
        require_positive_int(f"{name}[{index}]", width)


def require_optional_sha256(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class EncoderSpec:
    """Shapes emitted by one future frozen shared feature encoder.

    ``bypass_features`` contains encoded C coordinates that never enter the
    reciprocal/capacity-control core. ``operand_features`` contains ordered P.
    ``field_embeddings`` is the field view required by FM/attention models.
    The full C+P and operand-only ordered schemas have separate hashes so a
    DCDN checkpoint can never serialize an ambiguous schema identity.
    """

    bypass_dim: int
    operand_dim: int
    num_fields: int
    embedding_dim: int
    representation_dim: int
    denominator_mask: tuple[bool, ...]
    full_feature_schema_sha256: str | None
    operand_schema_sha256: str | None

    def __post_init__(self) -> None:
        require_nonnegative_int("bypass_dim", self.bypass_dim)
        require_positive_int("operand_dim", self.operand_dim)
        require_positive_int("num_fields", self.num_fields)
        require_positive_int("embedding_dim", self.embedding_dim)
        require_positive_int("representation_dim", self.representation_dim)
        if not isinstance(self.denominator_mask, tuple) or any(
            not isinstance(value, bool) for value in self.denominator_mask
        ):
            raise TypeError("denominator_mask must be a tuple of booleans")
        if len(self.denominator_mask) != self.operand_dim:
            raise ValueError("denominator_mask length must equal operand_dim")
        require_optional_sha256(
            "full_feature_schema_sha256", self.full_feature_schema_sha256
        )
        require_optional_sha256("operand_schema_sha256", self.operand_schema_sha256)

    @property
    def flat_dim(self) -> int:
        return self.bypass_dim + self.operand_dim

    @property
    def flattened_field_dim(self) -> int:
        return self.num_fields * self.embedding_dim


@dataclass(frozen=True)
class EncodedInteractionBatch:
    """One batch at the explicit post-encoder boundary.

    AFN+ owns two independent embedding tables in its source architecture.
    ``afn_dnn_embeddings`` is an explicit second learned view of identical
    ordered raw fields, not an extra feature regime. Other models ignore it.
    """

    bypass_features: torch.Tensor
    operand_features: torch.Tensor
    field_embeddings: torch.Tensor
    denominator_validity: torch.Tensor | None = None
    afn_dnn_embeddings: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.operand_features.shape[0])

    @property
    def flat_features(self) -> torch.Tensor:
        return torch.cat((self.bypass_features, self.operand_features), dim=-1)

    def validate(self, spec: EncoderSpec, *, require_afn_aux: bool = False) -> None:
        tensors = {
            "bypass_features": self.bypass_features,
            "operand_features": self.operand_features,
            "field_embeddings": self.field_embeddings,
        }
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
            raise TypeError("all encoded feature views must be tensors")
        if self.operand_features.ndim < 1:
            raise ValueError("operand_features must include a batch axis")
        if self.batch_size <= 0:
            raise ValueError("empty batches are not supported")

        expected_shapes = {
            "bypass_features": (self.batch_size, spec.bypass_dim),
            "operand_features": (self.batch_size, spec.operand_dim),
            "field_embeddings": (
                self.batch_size,
                spec.num_fields,
                spec.embedding_dim,
            ),
        }
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != expected_shapes[name]:
                raise ValueError(
                    f"{name} must have shape {expected_shapes[name]}, "
                    f"received {tuple(tensor.shape)}"
                )

        dtypes = {tensor.dtype for tensor in tensors.values()}
        devices = {tensor.device for tensor in tensors.values()}
        if len(dtypes) != 1:
            raise TypeError("all encoded feature views must share one dtype")
        if len(devices) != 1:
            raise ValueError("all encoded feature views must share one device")
        dtype = next(iter(dtypes))
        device = next(iter(devices))
        if dtype == torch.float16:
            raise TypeError("FP16 is forbidden at the controlled model boundary")
        if dtype not in _ALLOWED_DTYPES:
            raise TypeError("encoded feature views must use FP32 or BF16")

        if self.denominator_validity is not None:
            validity = self.denominator_validity
            if not isinstance(validity, torch.Tensor):
                raise TypeError("denominator_validity must be a tensor or None")
            if tuple(validity.shape) != (self.batch_size, spec.operand_dim):
                raise ValueError(
                    "denominator_validity must match operand_features.shape"
                )
            if validity.dtype != torch.bool:
                raise TypeError("denominator_validity must be boolean")
            if validity.device != device:
                raise ValueError("denominator_validity must share the feature device")

        if require_afn_aux and self.afn_dnn_embeddings is None:
            raise ValueError(
                "AFN+ requires afn_dnn_embeddings from its independent DNN "
                "embedding table"
            )
        if self.afn_dnn_embeddings is not None:
            auxiliary = self.afn_dnn_embeddings
            if not isinstance(auxiliary, torch.Tensor):
                raise TypeError("afn_dnn_embeddings must be a tensor or None")
            if auxiliary.shape != self.field_embeddings.shape:
                raise ValueError("afn_dnn_embeddings must match field_embeddings.shape")
            if auxiliary.dtype != dtype:
                raise TypeError(
                    "afn_dnn_embeddings must share the encoded feature dtype"
                )
            if auxiliary.device != device:
                raise ValueError(
                    "afn_dnn_embeddings must share the encoded feature device"
                )


@dataclass(frozen=True)
class InteractionRepresentation:
    values: torch.Tensor
    model_key: str

    def validate(self, *, batch_size: int, representation_dim: int) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("representation values must be a tensor")
        if tuple(self.values.shape) != (batch_size, representation_dim):
            raise RuntimeError(
                "interaction representation violated the common shape contract"
            )
        if not isinstance(self.model_key, str) or not self.model_key:
            raise ValueError("model_key must be a non-empty string")


@dataclass(frozen=True)
class PredictionOutput:
    representation: InteractionRepresentation
    logits: torch.Tensor


def canonical_config(value: Any) -> Any:
    """Convert dataclass configuration state to JSON-safe canonical values."""

    if is_dataclass(value) and not isinstance(value, type):
        return canonical_config(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): canonical_config(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple | list):
        return [canonical_config(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise TypeError(f"unsupported configuration value: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        canonical_config(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
