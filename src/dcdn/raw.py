"""Immutable raw-batch boundary for model-owned encoder integration.

This module deliberately contains no encoder and no Parquet implementation.
It defines the exact label-bearing batch contract that a bounded-memory shard
reader must satisfy before the training harness may be connected to public
data.  In particular, it never exposes precomputed field embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch

from dcdn.protocol import SHA256_RE

RawSplit = Literal["train", "validation", "test"]


class RawContractError(RuntimeError):
    """Raw source identity or row-order invariants were violated."""


@dataclass(frozen=True)
class RawSourceIdentity:
    """Complete immutable identity of one materialized raw split."""

    dataset_id: str
    split: RawSplit
    feature_regime: str
    row_count: int
    data_manifest_sha256: str
    views_manifest_sha256: str
    feature_contract_sha256: str
    physical_schema_sha256: str
    expanded_schema_sha256: str
    operand_schema_sha256: str
    denominator_mask_sha256: str
    full_materialized_scan_sha256: str
    cache_content_sha256: str
    ordered_context_fields: tuple[str, ...]
    ordered_operand_fields: tuple[str, ...]
    categorical_cardinalities: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("raw source split is invalid")
        if not isinstance(self.feature_regime, str) or not self.feature_regime.strip():
            raise ValueError("feature_regime must be a non-empty string")
        if (
            isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or self.row_count <= 0
        ):
            raise ValueError("row_count must be a positive integer")
        for name in (
            "data_manifest_sha256",
            "views_manifest_sha256",
            "feature_contract_sha256",
            "physical_schema_sha256",
            "expanded_schema_sha256",
            "operand_schema_sha256",
            "denominator_mask_sha256",
            "full_materialized_scan_sha256",
            "cache_content_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            if (
                name
                in {
                    "full_materialized_scan_sha256",
                    "cache_content_sha256",
                }
                and len(set(value)) == 1
            ):
                raise ValueError(f"{name} must not be a placeholder digest")
        content_digests = {
            self.full_materialized_scan_sha256,
            self.cache_content_sha256,
        }
        metadata_digests = {
            self.data_manifest_sha256,
            self.views_manifest_sha256,
            self.feature_contract_sha256,
            self.physical_schema_sha256,
            self.expanded_schema_sha256,
            self.operand_schema_sha256,
            self.denominator_mask_sha256,
        }
        if len(content_digests) != 2 or content_digests & metadata_digests:
            raise ValueError(
                "full scan and cache content must be distinct content digests"
            )
        self._validate_ordered_names(
            self.ordered_context_fields, "ordered_context_fields"
        )
        self._validate_ordered_names(
            self.ordered_operand_fields, "ordered_operand_fields"
        )
        if len(self.categorical_cardinalities) != len(self.ordered_context_fields):
            raise ValueError(
                "categorical_cardinalities must align with ordered_context_fields"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 1
            for value in self.categorical_cardinalities
        ):
            raise ValueError("categorical cardinalities must be integers above one")

    @staticmethod
    def _validate_ordered_names(values: tuple[str, ...], field: str) -> None:
        if not isinstance(values, tuple) or not values:
            raise ValueError(f"{field} must be a non-empty tuple")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{field} must contain non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"{field} must not contain duplicates")


@dataclass(frozen=True)
class RawFeatureBatch:
    """Raw C/P rows and labels in a caller-requested global row order."""

    context_tokens: torch.Tensor
    operand_features: torch.Tensor
    denominator_validity: torch.Tensor
    targets: torch.Tensor
    global_row_indices: torch.Tensor

    def validate(
        self,
        identity: RawSourceIdentity,
        requested_indices: Sequence[int],
        *,
        runtime_value_checks: bool = True,
    ) -> None:
        """Fail unless structural invariants and selected value checks agree."""

        if not isinstance(runtime_value_checks, bool):
            raise TypeError("runtime_value_checks must be boolean")
        requested = tuple(requested_indices)
        if not requested:
            raise RawContractError("requested_indices must be non-empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in requested
        ):
            raise RawContractError("requested_indices must contain Python integers")
        if any(index < 0 or index >= identity.row_count for index in requested):
            raise RawContractError("requested_indices exceed the raw source bounds")

        if not isinstance(self.global_row_indices, torch.Tensor):
            raise RawContractError("global_row_indices must be a tensor")
        if (
            self.global_row_indices.device.type != "cpu"
            or self.global_row_indices.dtype != torch.int64
            or self.global_row_indices.ndim != 1
        ):
            raise RawContractError(
                "global_row_indices must be a one-dimensional CPU int64 tensor"
            )
        expected_indices = torch.tensor(requested, dtype=torch.int64)
        if not torch.equal(self.global_row_indices, expected_indices):
            raise RawContractError(
                "global_row_indices do not exactly preserve requested order"
            )

        batch_size = len(requested)
        context_width = len(identity.ordered_context_fields)
        operand_width = len(identity.ordered_operand_fields)
        expected_shapes = {
            "context_tokens": (batch_size, context_width),
            "operand_features": (batch_size, operand_width),
            "denominator_validity": (batch_size, operand_width),
            "targets": (batch_size, 1),
        }
        for name, shape in expected_shapes.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise RawContractError(f"{name} must have shape {shape}")

        if self.context_tokens.dtype != torch.int64:
            raise RawContractError("context_tokens must be int64")
        if self.operand_features.dtype != torch.float32:
            raise RawContractError("operand_features must be FP32")
        if self.denominator_validity.dtype != torch.bool:
            raise RawContractError("denominator_validity must be boolean")
        if self.targets.dtype != torch.float32:
            raise RawContractError("targets must be FP32")
        model_devices = {
            self.context_tokens.device,
            self.operand_features.device,
            self.denominator_validity.device,
            self.targets.device,
        }
        if len(model_devices) != 1:
            raise RawContractError("raw model tensors must share one device")
        if runtime_value_checks:
            if not bool(torch.isfinite(self.operand_features).all()):
                raise RawContractError("operand_features contain non-finite values")
            if not bool(torch.isfinite(self.targets).all()):
                raise RawContractError("targets contain non-finite values")
            for column, cardinality in enumerate(identity.categorical_cardinalities):
                tokens = self.context_tokens[:, column]
                if bool(((tokens < 0) | (tokens >= cardinality)).any()):
                    raise RawContractError(
                        f"context token column {column} exceeds its cardinality"
                    )


@runtime_checkable
class RawBatchSource(Protocol):
    """Bounded-memory manifest/shard source with deterministic lifecycle."""

    @property
    def identity(self) -> RawSourceIdentity: ...
    @property
    def runtime_value_checks(self) -> bool: ...

    def __len__(self) -> int: ...

    def batch(
        self,
        indices: Sequence[int],
        *,
        device: torch.device,
    ) -> RawFeatureBatch: ...

    def close(self) -> None: ...
