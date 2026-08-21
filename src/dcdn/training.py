"""Deterministic, metric-safe training primitives for the frozen model suite.

This module owns no dataset preprocessing and performs no external telemetry.
Real execution starts at a bounded-memory :class:`RawBatchSource`; the complete
optimizer/checkpoint boundary is :class:`TrainableEncodedModel`, which owns the
shared encoder, interaction, adapter, and common head.  The legacy encoded
tensor path remains available only to synthetic deterministic unit tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from dcdn.capacity_artifact import (
    capacity_resolution_projection,
    load_verified_capacity_artifact,
    validate_capacity_artifact,
    verify_repository_bindings,
)
from dcdn.capacity_artifact import (
    sha256_file as capacity_sha256_file,
)
from dcdn.checkpoint import (
    CheckpointMetadata,
    CheckpointProgress,
    CheckpointRecord,
    EvaluationCheckpointState,
    ResumeState,
    load_checkpoint,
    load_model_checkpoint_for_evaluation,
    save_checkpoint,
    verify_checkpoint_sidecar,
)
from dcdn.encoder import TrainableEncodedModel
from dcdn.final_evaluation import (
    FinalBundleRecord,
    run_one_pass_test_bundle,
)
from dcdn.models import (
    AFNPlusConfig,
    AutoIntPlusConfig,
    DCDNCoreConfig,
    DCNConfig,
    DCNv2CapacityCoreConfig,
    DCNv2FullRankConfig,
    DeepFMConfig,
    EncodedInteractionBatch,
    EncoderSpec,
    HeadConfig,
    InteractionConfig,
    MLPConfig,
    ProvenancedControlledModel,
    WukongSmallConfig,
    build_controlled_model,
)
from dcdn.monitor import AsyncNonFiniteMonitor, NumericalMonitorError
from dcdn.multitask_model import (
    MultiTargetTrainableEncodedModel,
    ProvenancedMultiTargetModel,
)
from dcdn.protocol import (
    SHA256_RE,
    ProtocolGate,
    RuntimeIdentity,
    SelectedConfigsLock,
    TestEvaluationClaim,
    split_sha256_from_views_payload,
)
from dcdn.raw import (
    FinalRawBatchSource,
    RawBatchSource,
    RawContractError,
    RawFeatureBatch,
    RawSourceIdentity,
)
from dcdn.reproducibility import seed_everything
from dcdn.sampler import StatefulPermutationSampler
from dcdn.search_space import SearchSpaceLock, validate_capacity_resolution
from dcdn.selection_artifact import (
    CAPACITY_CANDIDATE_ID,
    CAPACITY_LOW_RANK,
    CAPACITY_RESOLUTION_SHA256,
    ORDINARY_SELECTION_SHA256,
    PREMETRIC_PROTOCOL_SHA256,
    SELECTED_LEARNING_RATE,
    CapacityBindingEvidence,
    capacity_phase_resolutions,
    capacity_training_binding_payload,
    load_capacity_lr_binding,
)

Split = Literal["train", "validation", "test"]
RunPhase = Literal["tuning", "final"]
LossName = Literal[
    "binary_cross_entropy_with_logits",
    "normalized_mean_squared_error",
    "macro_nloss",
]

_SECRET_RE = re.compile(
    r"(github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
)
_FINAL_SEEDS = (42, 43, 44)
_BEST_CHECKPOINT_FILENAME_RE = re.compile(
    r"^best-step-[0-9]{12}-generation-[0-9]{4}\.pt$"
)
CAPACITY_FLOP_CONTRACT = "dcdn-forward-floating-arithmetic-flops-v1"
_SEARCH_ARCHITECTURE_KEYS = {
    "dnn": "dnn",
    "deepfm": "deepfm",
    "dcn": "dcn",
    "autoint_plus": "autoint_plus",
    "dcnv2_full_rank": "dcnv2_full_rank",
    "afn_plus": "afn_plus",
    "wukong_small": "wukong_small",
    "dcdn_core": "dcdn_core",
    "dcnv2_capacity_matched_core": "dcnv2_low_rank_mix_capacity",
}
_TUNED_CANDIDATES: dict[str, dict[str, float]] = {
    "dnn": {"dnn_lr_3e_4": 3e-4, "dnn_lr_1e_3": 1e-3},
    "deepfm": {"deepfm_lr_3e_4": 3e-4, "deepfm_lr_1e_3": 1e-3},
    "dcn": {"dcn_lr_3e_4": 3e-4, "dcn_lr_1e_3": 1e-3},
    "autoint_plus": {
        "autoint_plus_lr_3e_4": 3e-4,
        "autoint_plus_lr_1e_3": 1e-3,
    },
    "dcnv2_full_rank": {
        "dcnv2_full_rank_lr_3e_4": 3e-4,
        "dcnv2_full_rank_lr_1e_3": 1e-3,
    },
    "afn_plus": {"afn_plus_lr_3e_4": 3e-4, "afn_plus_lr_1e_3": 1e-3},
    "dcdn_core": {
        "dcdn_core_l2_lr_3e_4": 3e-4,
        "dcdn_core_l2_lr_1e_3": 1e-3,
    },
}


class TrainingContractError(RuntimeError):
    """A frozen training, safety, or architecture contract was violated."""


class SplitAccessError(TrainingContractError):
    """A split was requested before its immutable access gate passed."""


class SecondOOMError(TrainingContractError):
    """A run encountered its second OOM and must stop."""


class UnresolvedEncoderBridgeError(TrainingContractError):
    """A public shard cannot run before its model-owned encoder is frozen."""


def assert_synthetic_encoded_execution(dataset_id: str) -> None:
    """Keep the legacy pre-encoded source boundary synthetic-only."""

    if dataset_id == "synthetic" or dataset_id.startswith("synthetic-"):
        return
    raise UnresolvedEncoderBridgeError(
        "real public-dataset execution requires RawBatchSource plus a model-owned "
        "TrainableEncodedModel; precomputed field embeddings are forbidden"
    )


@dataclass(frozen=True)
class TrainingConfig:
    dataset_id: str
    model_key: str
    feature_regime: str
    candidate_id: str
    seed: int
    phase: RunPhase
    learning_rate: float
    microbatch_size: int
    effective_batch_size: int
    maximum_train_examples: int
    validate_every_examples: int
    patience_validation_checks: int
    loss: LossName
    selection_metric: str
    selection_direction: Literal["minimize"]
    search_space_sha256: str
    experiment_matrix_sha256: str
    encoder_contract_sha256: str
    capacity_resolution_sha256: str
    resolved_model_config_sha256: str
    full_schema_sha256: str
    operand_schema_sha256: str
    denominator_mask_sha256: str
    weight_decay: float = 1e-6
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    gradient_clip_l2: float = 1.0
    autocast_bfloat16: bool = True
    checkpoint_every_examples: int | None = None
    allow_one_oom_microbatch_reduction: bool = True
    numerical_nonfinite_limit: int = 1
    runtime_value_checks: bool = True
    reproduction_command: str = ""
    selection_receipt_path: str | None = None
    selection_receipt_sha256: str | None = None
    ordinary_dcnv2_selection_sha256: str | None = None
    capacity_lr_binding_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "model_key",
            "feature_regime",
            "candidate_id",
            "selection_metric",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.phase not in {"tuning", "final"}:
            raise ValueError("phase must be tuning or final")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.phase == "tuning" and self.seed != 42:
            raise ValueError("validation-only tuning must use seed 42")
        if self.phase == "final" and self.seed not in _FINAL_SEEDS:
            raise ValueError(f"final seed must be one of {_FINAL_SEEDS}")
        for name in (
            "microbatch_size",
            "effective_batch_size",
            "maximum_train_examples",
            "validate_every_examples",
            "patience_validation_checks",
            "numerical_nonfinite_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.effective_batch_size % self.microbatch_size:
            raise ValueError(
                "effective_batch_size must be divisible by microbatch_size"
            )
        if self.checkpoint_every_examples is not None and (
            isinstance(self.checkpoint_every_examples, bool)
            or not isinstance(self.checkpoint_every_examples, int)
            or self.checkpoint_every_examples <= 0
        ):
            raise ValueError("checkpoint_every_examples must be positive")
        for name in (
            "learning_rate",
            "weight_decay",
            "adam_epsilon",
            "gradient_clip_l2",
        ):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be an exact float")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            type(self.adam_betas) is not tuple
            or len(self.adam_betas) != 2
            or not all(type(value) is float for value in self.adam_betas)
        ):
            raise TypeError("adam_betas must contain two exact floats")
        if not all(
            math.isfinite(value) and 0.0 < value < 1.0 for value in self.adam_betas
        ):
            raise ValueError("adam_betas must contain two finite values in (0, 1)")
        if self.loss not in {
            "binary_cross_entropy_with_logits",
            "normalized_mean_squared_error",
            "macro_nloss",
        }:
            raise ValueError("unsupported frozen loss")
        if self.selection_direction != "minimize":
            raise ValueError("the frozen primary selection direction is minimize")
        for name in (
            "search_space_sha256",
            "experiment_matrix_sha256",
            "encoder_contract_sha256",
            "capacity_resolution_sha256",
            "resolved_model_config_sha256",
            "full_schema_sha256",
            "operand_schema_sha256",
            "denominator_mask_sha256",
        ):
            if (
                not isinstance(getattr(self, name), str)
                or SHA256_RE.fullmatch(getattr(self, name)) is None
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        receipt_bindings = (
            self.selection_receipt_path,
            self.selection_receipt_sha256,
        )
        if any(value is not None for value in receipt_bindings):
            if self.phase != "final" or any(
                value is None for value in receipt_bindings
            ):
                raise ValueError(
                    "selection receipt path and digest are a final-phase pair"
                )
            assert self.selection_receipt_path is not None
            assert self.selection_receipt_sha256 is not None
            receipt_path = Path(self.selection_receipt_path)
            if (
                receipt_path.is_absolute()
                or receipt_path.as_posix() != self.selection_receipt_path
                or any(part in {"", ".", ".."} for part in receipt_path.parts)
                or not self.selection_receipt_path.endswith(".json")
            ):
                raise ValueError("selection_receipt_path must be repository-relative")
        capacity_bindings = (
            self.ordinary_dcnv2_selection_sha256,
            self.capacity_lr_binding_sha256,
        )
        if self.model_key == "dcnv2_capacity_matched_core":
            if (
                self.candidate_id != CAPACITY_CANDIDATE_ID
                or self.learning_rate != SELECTED_LEARNING_RATE
                or self.capacity_resolution_sha256 != CAPACITY_RESOLUTION_SHA256
                or self.ordinary_dcnv2_selection_sha256 != ORDINARY_SELECTION_SHA256
            ):
                raise ValueError(
                    "capacity control requires the authorized candidate, LR, "
                    "capacity artifact, and ordinary selection receipt"
                )
            for name, value in zip(
                (
                    "ordinary_dcnv2_selection_sha256",
                    "capacity_lr_binding_sha256",
                ),
                capacity_bindings,
                strict=True,
            ):
                if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                    raise ValueError(
                        f"capacity control requires {name} as a lowercase SHA-256"
                    )
        elif any(value is not None for value in capacity_bindings):
            raise ValueError(
                "ordinary/capacity LR selection bindings are capacity-only"
            )
        if type(self.autocast_bfloat16) is not bool:
            raise TypeError("autocast_bfloat16 must be boolean")
        if type(self.allow_one_oom_microbatch_reduction) is not bool:
            raise TypeError("allow_one_oom_microbatch_reduction must be boolean")
        if type(self.runtime_value_checks) is not bool:
            raise TypeError("runtime_value_checks must be boolean")
        if _SECRET_RE.search(self.reproduction_command):
            raise ValueError("reproduction command contains credential-like material")

    @property
    def accumulation_steps(self) -> int:
        return self.effective_batch_size // self.microbatch_size

    @property
    def checkpoint_cadence(self) -> int:
        return self.checkpoint_every_examples or self.validate_every_examples

    @property
    def checkpoint_precision(self) -> str:
        """Describe forward precision without hiding disabled autocast."""

        if self.autocast_bfloat16:
            return "fp32_parameters_bf16_autocast"
        return "fp32_parameters_fp32_forward"

    @property
    def realized_maximum_examples(self) -> int:
        """Largest full effective-batch boundary not above the declared hard max."""

        value = (
            self.maximum_train_examples // self.effective_batch_size
        ) * self.effective_batch_size
        if value <= 0:
            raise ValueError(
                "declared maximum permits no complete effective optimizer batch"
            )
        return value

    def completed_batch_boundary_at_or_after(self, threshold: int) -> int:
        """Return the first completed effective batch at/after a declaration."""

        if isinstance(threshold, bool) or not isinstance(threshold, int):
            raise TypeError("threshold must be an integer")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        return (
            (threshold + self.effective_batch_size - 1) // self.effective_batch_size
        ) * self.effective_batch_size


def training_config_payload(config: TrainingConfig) -> dict[str, Any]:
    """Serialize without adding null capacity fields to legacy checkpoints."""

    payload = asdict(config)
    for name in (
        "selection_receipt_path",
        "selection_receipt_sha256",
        "ordinary_dcnv2_selection_sha256",
        "capacity_lr_binding_sha256",
    ):
        if payload[name] is None:
            del payload[name]
    return payload


def runtime_identity_from_raw(
    identity: RawSourceIdentity,
    *,
    encoder_contract_sha256: str,
    capacity_resolution_sha256: str,
) -> RuntimeIdentity:
    """Project an exact raw identity into Protocol v3's independent identity."""

    return RuntimeIdentity(
        dataset_id=identity.dataset_id,
        split=identity.split,
        feature_regime=identity.feature_regime,
        row_count=identity.row_count,
        data_manifest_sha256=identity.data_manifest_sha256,
        views_manifest_sha256=identity.views_manifest_sha256,
        features_sha256=identity.feature_contract_sha256,
        physical_schema_sha256=identity.physical_schema_sha256,
        expanded_schema_sha256=identity.expanded_schema_sha256,
        operand_schema_sha256=identity.operand_schema_sha256,
        denominator_mask_sha256=identity.denominator_mask_sha256,
        full_materialized_scan_sha256=identity.full_materialized_scan_sha256,
        cache_content_sha256=identity.cache_content_sha256,
        encoder_contract_sha256=encoder_contract_sha256,
        capacity_resolution_sha256=capacity_resolution_sha256,
    )


def _identity_without_split_and_rows(identity: object) -> dict[str, Any]:
    payload = asdict(identity)  # type: ignore[arg-type]
    payload.pop("split")
    payload.pop("row_count")
    return payload


@dataclass(frozen=True)
class SafetyAttestation:
    """Independent evidence required before any raw source factory is invoked."""

    protocol_gate: ProtocolGate
    cost_snapshot: Mapping[str, Any]
    storage_snapshot: Mapping[str, Any]
    leakage_checks: Mapping[str, bool]
    views_payload: Mapping[str, Any]
    views_payload_sha256: str
    views_split_sha256: str
    search_space_sha256: str
    experiment_matrix_sha256: str
    encoder_contract_sha256: str
    capacity_resolution_sha256: str
    train_runtime_identity: RuntimeIdentity
    validation_runtime_identity: RuntimeIdentity
    train_source_identity: RawSourceIdentity
    validation_source_identity: RawSourceIdentity
    registry_on_persistent_storage: bool
    selected_configs_path: Path | None = None
    test_runtime_identity: RuntimeIdentity | None = None
    test_source_identity: RawSourceIdentity | None = None

    def _validate_identity_pair(
        self,
        runtime_identity: RuntimeIdentity,
        source_identity: RawSourceIdentity,
    ) -> None:
        projected = runtime_identity_from_raw(
            source_identity,
            encoder_contract_sha256=self.encoder_contract_sha256,
            capacity_resolution_sha256=self.capacity_resolution_sha256,
        )
        if projected != runtime_identity:
            raise TrainingContractError(
                f"{source_identity.split} raw/runtime identity mismatch"
            )
        try:
            self.protocol_gate.require_runtime_identity(
                runtime_identity,
                requested_split=source_identity.split,
            )
        except Exception as exc:
            raise TrainingContractError(str(exc)) from exc

    def validate_common(self, config: TrainingConfig) -> None:
        if self.protocol_gate.dataset_id != config.dataset_id:
            raise TrainingContractError("protocol dataset_id mismatch")
        if self.protocol_gate.feature_regime != config.feature_regime:
            raise TrainingContractError("protocol feature_regime mismatch")
        try:
            self.protocol_gate.require_active()
        except Exception as exc:
            raise TrainingContractError(
                f"active protocol state is required: {exc}"
            ) from exc
        for name in (
            "views_payload_sha256",
            "views_split_sha256",
            "search_space_sha256",
            "experiment_matrix_sha256",
            "encoder_contract_sha256",
            "capacity_resolution_sha256",
        ):
            if SHA256_RE.fullmatch(getattr(self, name)) is None:
                raise TrainingContractError(f"{name} is not a SHA-256 digest")
        if self.search_space_sha256 != config.search_space_sha256:
            raise TrainingContractError("attested search-space digest mismatch")
        if self.experiment_matrix_sha256 != config.experiment_matrix_sha256:
            raise TrainingContractError("attested experiment-matrix digest mismatch")
        if self.encoder_contract_sha256 != config.encoder_contract_sha256:
            raise TrainingContractError("attested encoder digest mismatch")
        if self.capacity_resolution_sha256 != config.capacity_resolution_sha256:
            raise TrainingContractError("attested capacity artifact digest mismatch")

        hashes = self.protocol_gate.payload["hashes"]
        try:
            canonical_views = json.dumps(
                self.views_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise TrainingContractError(
                f"views payload is not canonical JSON data: {exc}"
            ) from exc
        if hashlib.sha256(canonical_views).hexdigest() != self.views_payload_sha256:
            raise TrainingContractError("exact views payload digest mismatch")
        expected_hashes = {
            "split_sha256": self.views_split_sha256,
            "search_space_sha256": self.search_space_sha256,
            "experiment_matrix_sha256": self.experiment_matrix_sha256,
            "encoder_contract_sha256": self.encoder_contract_sha256,
            "capacity_resolution_sha256": self.capacity_resolution_sha256,
        }
        for name, value in expected_hashes.items():
            if hashes.get(name) != value:
                raise TrainingContractError(f"protocol {name} mismatch")
        try:
            actual_split_hash = split_sha256_from_views_payload(self.views_payload)
            self.protocol_gate.require_views_split(self.views_payload)
        except Exception as exc:
            raise TrainingContractError(
                f"views split attestation failed: {exc}"
            ) from exc
        if actual_split_hash != self.views_split_sha256:
            raise TrainingContractError("views split payload digest mismatch")

        self._validate_identity_pair(
            self.train_runtime_identity, self.train_source_identity
        )
        self._validate_identity_pair(
            self.validation_runtime_identity, self.validation_source_identity
        )
        if _identity_without_split_and_rows(
            self.train_runtime_identity
        ) != _identity_without_split_and_rows(self.validation_runtime_identity):
            raise TrainingContractError(
                "train/validation RuntimeIdentity drift exceeds split and row_count"
            )
        if _identity_without_split_and_rows(
            self.train_source_identity
        ) != _identity_without_split_and_rows(self.validation_source_identity):
            raise TrainingContractError(
                "train/validation RawSourceIdentity drift exceeds split and row_count"
            )
        if self.test_runtime_identity is None:
            if self.test_source_identity is not None:
                raise TrainingContractError(
                    "test source identity lacks runtime identity"
                )
        else:
            if self.test_source_identity is None:
                raise TrainingContractError(
                    "test runtime identity lacks source identity"
                )
            self._validate_identity_pair(
                self.test_runtime_identity, self.test_source_identity
            )

        required_leakage = {
            "source_manifest_verified",
            "split_verified",
            "ordered_schema_verified",
            "row_alignment_verified",
            "temporal_leakage_check_passed",
            "reciprocal_mask_verified",
            "full_operand_scan_passed",
        }
        if set(self.leakage_checks) != required_leakage:
            raise TrainingContractError(
                "leakage checks must contain exactly the frozen safety gates"
            )
        if any(type(value) is not bool for value in self.leakage_checks.values()):
            raise TrainingContractError("leakage gate values must be booleans")
        failed = sorted(
            name for name, value in self.leakage_checks.items() if not value
        )
        if failed:
            raise TrainingContractError("leakage gates failed: " + ", ".join(failed))
        if self.cost_snapshot.get("launch_allowed") is not True:
            raise TrainingContractError("local cost guard did not allow launch")
        if self.cost_snapshot.get("provider_actions_performed") is not False:
            raise TrainingContractError(
                "cost evidence has an invalid provider-action flag"
            )
        if self.storage_snapshot.get("ok") is not True:
            raise TrainingContractError("local storage guard did not pass")
        if self.registry_on_persistent_storage is not True:
            raise TrainingContractError("run registry persistence was not verified")

    def expected_runtime_identity(self, split: Split) -> RuntimeIdentity:
        if split == "train":
            return self.train_runtime_identity
        if split == "validation":
            return self.validation_runtime_identity
        if split == "test" and self.test_runtime_identity is not None:
            return self.test_runtime_identity
        raise SplitAccessError(f"no exact runtime identity is attested for {split}")

    def expected_source_identity(self, split: Split) -> RawSourceIdentity:
        if split == "train":
            return self.train_source_identity
        if split == "validation":
            return self.validation_source_identity
        if split == "test" and self.test_source_identity is not None:
            return self.test_source_identity
        raise SplitAccessError(f"no exact raw source identity is attested for {split}")

    def require_split(
        self,
        split: Split,
        config: TrainingConfig,
        *,
        checkpoint_protocol_sha256: str | None = None,
        resolved_run_config_sha256: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> TestEvaluationClaim | None:
        """Run Protocol v3 before a factory; test returns a reserved claim only."""

        self.validate_common(config)
        if split not in {"train", "validation", "test"}:
            raise SplitAccessError(f"unknown split: {split!r}")
        if split == "test" and config.phase != "final":
            raise SplitAccessError("tuning runs may never reserve the test split")
        identity = self.expected_runtime_identity(split)
        try:
            return self.protocol_gate.require_split(
                split,
                runtime_identity=identity,
                selected_configs_path=(
                    self.selected_configs_path if split == "test" else None
                ),
                model_key=config.model_key if split == "test" else None,
                candidate_id=config.candidate_id if split == "test" else None,
                resolved_config_sha256=(
                    config.resolved_model_config_sha256 if split == "test" else None
                ),
                resolved_run_config_sha256=(
                    resolved_run_config_sha256 if split == "test" else None
                ),
                checkpoint_sha256=(checkpoint_sha256 if split == "test" else None),
                seed=config.seed if split == "test" else None,
                checkpoint_protocol_sha256=(
                    checkpoint_protocol_sha256 if split == "test" else None
                ),
            )
        except Exception as exc:
            raise SplitAccessError(str(exc)) from exc


@runtime_checkable
class EncodedBatchSource(Protocol):
    """Synthetic-only post-encoder source used by deterministic harness tests."""

    @property
    def split(self) -> Split: ...

    def __len__(self) -> int: ...

    def batch(
        self,
        indices: Sequence[int],
        *,
        device: torch.device,
    ) -> tuple[EncodedInteractionBatch, torch.Tensor]: ...


@dataclass(frozen=True)
class EncodedTensorDataset:
    """In-memory post-encoder tensors restricted to synthetic verification."""

    split: Split
    bypass_features: torch.Tensor
    operand_features: torch.Tensor
    field_embeddings: torch.Tensor
    targets: torch.Tensor
    denominator_validity: torch.Tensor | None = None
    afn_dnn_embeddings: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("dataset split is invalid")
        rows = (
            int(self.targets.shape[0]) if isinstance(self.targets, torch.Tensor) else -1
        )
        if rows <= 0:
            raise ValueError("encoded dataset must contain rows")
        for name in ("bypass_features", "operand_features", "field_embeddings"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != rows:
                raise ValueError(f"{name} must be a row-aligned tensor")
            if tensor.device.type != "cpu":
                raise ValueError("durable encoded tensors must be CPU resident")
        if self.targets.ndim == 1:
            object.__setattr__(self, "targets", self.targets.unsqueeze(-1))
        if self.targets.ndim != 2 or self.targets.device.type != "cpu":
            raise ValueError("targets must be a two-dimensional CPU tensor")
        for optional_name in ("denominator_validity", "afn_dnn_embeddings"):
            tensor = getattr(self, optional_name)
            if tensor is not None and (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape[0] != rows
                or tensor.device.type != "cpu"
            ):
                raise ValueError(f"{optional_name} must be a row-aligned CPU tensor")

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def batch(
        self,
        indices: Sequence[int],
        *,
        device: torch.device,
    ) -> tuple[EncodedInteractionBatch, torch.Tensor]:
        index = torch.as_tensor(tuple(indices), dtype=torch.int64, device="cpu")
        if index.ndim != 1 or index.numel() == 0:
            raise ValueError("batch indices must be a non-empty vector")

        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            selected = value.index_select(0, index)
            return selected.to(device=device, non_blocking=device.type == "cuda")

        encoded = EncodedInteractionBatch(
            bypass_features=move(self.bypass_features),  # type: ignore[arg-type]
            operand_features=move(self.operand_features),  # type: ignore[arg-type]
            field_embeddings=move(self.field_embeddings),  # type: ignore[arg-type]
            denominator_validity=move(self.denominator_validity),
            afn_dnn_embeddings=move(self.afn_dnn_embeddings),
        )
        targets = move(self.targets)
        assert targets is not None
        return encoded, targets


@dataclass(frozen=True)
class LatencySummary:
    warmup_repetitions: int
    measured_repetitions: int
    batch_size: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    scope: Literal["end_to_end_raw_to_logit_model"] = "end_to_end_raw_to_logit_model"


@dataclass(frozen=True)
class TrainingResult:
    examples_seen: int
    optimizer_steps: int
    validation_checks: int
    best_validation_metric: float | None
    stopped_reason: str
    elapsed_seconds: float
    throughput_examples_per_second: float
    peak_vram_bytes: int
    oom_count: int
    last_checkpoint_sha256: str
    best_checkpoint_sha256: str | None
    best_checkpoint_filename: str | None
    numerical_summary: Mapping[str, Any]
    reproduction_command: str
    declared_maximum_train_examples: int
    realized_maximum_train_examples: int


class CompactEventLog:
    """Append-only local JSONL run history with durability and secret checks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **payload: Any) -> None:
        if not isinstance(event, str) or not event:
            raise ValueError("event must be non-empty")
        record = {"event": event, "monotonic_ns": time.monotonic_ns(), **payload}
        serialized = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        if _SECRET_RE.search(serialized):
            raise TrainingContractError(
                "event payload contains credential-like material"
            )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably create one immutable JSON receipt, rejecting any overwrite."""

    serialized = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    if _SECRET_RE.search(serialized):
        raise TrainingContractError("receipt contains credential-like material")
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise SplitAccessError(
            f"immutable receipt already exists: {path.name}"
        ) from exc
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _mask_sha256(mask: Sequence[bool]) -> str:
    if any(type(value) is not bool for value in mask):
        raise TypeError("denominator mask must contain booleans")
    canonical = json.dumps(
        list(mask),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _controlled_model(
    model: (
        TrainableEncodedModel
        | MultiTargetTrainableEncodedModel
        | ProvenancedControlledModel
        | ProvenancedMultiTargetModel
    ),
) -> ProvenancedControlledModel | ProvenancedMultiTargetModel:
    if isinstance(model, TrainableEncodedModel | MultiTargetTrainableEncodedModel):
        return model.controlled_model
    if isinstance(model, ProvenancedControlledModel | ProvenancedMultiTargetModel):
        return model
    raise TypeError("model must be a frozen controlled or joint model")


def _expected_frozen_interaction_config(
    config: TrainingConfig,
    search_lock: SearchSpaceLock,
    *,
    capacity_artifact: Mapping[str, Any] | None = None,
) -> InteractionConfig | None:
    """Normalize the frozen architecture record to the runtime config dataclass."""

    try:
        search_key = _SEARCH_ARCHITECTURE_KEYS[config.model_key]
        model_record = search_lock.payload["models"][search_key]
        architecture = model_record["base_config"]["architecture"]
    except (KeyError, TypeError) as exc:
        raise TrainingContractError(
            "frozen search-space architecture record is missing"
        ) from exc
    if not isinstance(architecture, Mapping):
        raise TrainingContractError("frozen architecture must be a mapping")
    if architecture.get("model_key") != config.model_key:
        raise TrainingContractError("frozen architecture model_key mismatch")

    try:
        if config.model_key == "dnn":
            return MLPConfig(
                hidden_dims=tuple(architecture["hidden_dims"]),
                dropout=architecture["dropout"],
            )
        if config.model_key == "deepfm":
            return DeepFMConfig(
                deep_hidden_dims=tuple(architecture["dnn_hidden_dims"]),
                dropout=architecture["dropout"],
            )
        if config.model_key == "dcn":
            return DCNConfig(
                num_layers=architecture["num_cross_layers"],
                product_init_scale=0.1,
            )
        if config.model_key == "autoint_plus":
            if architecture["attention_dropout"] != architecture["dnn_dropout"]:
                raise TrainingContractError(
                    "frozen AutoInt+ branch dropouts are not representable exactly"
                )
            return AutoIntPlusConfig(
                attention_layers=architecture["attention_layers"],
                attention_dim=architecture["attention_dim"],
                num_heads=architecture["num_heads"],
                dnn_hidden_dims=tuple(architecture["dnn_hidden_dims"]),
                dropout=architecture["attention_dropout"],
                scale_attention=architecture["use_scale"],
                layer_norm=architecture["layer_norm"],
            )
        if config.model_key == "dcnv2_full_rank":
            return DCNv2FullRankConfig(
                num_layers=architecture["num_cross_layers"],
                product_init_scale=0.1,
            )
        if config.model_key == "afn_plus":
            treatment = architecture["positive_log_treatment"]
            if not isinstance(treatment, Mapping):
                raise TrainingContractError(
                    "frozen AFN+ positive-log treatment must be a mapping"
                )
            return AFNPlusConfig(
                logarithmic_neurons=architecture["logarithmic_neurons"],
                afn_hidden_dims=tuple(architecture["afn_hidden_dims"]),
                dnn_hidden_dims=tuple(architecture["dnn_hidden_dims"]),
                log_floor=treatment["clamp_min"],
                exp_clip=treatment["exp_clip"],
                dropout=architecture["dropout"],
                batch_norm=architecture["batch_norm"],
            )
        if config.model_key == "wukong_small":
            return WukongSmallConfig(
                num_layers=architecture["num_layers"],
                lcb_features=architecture["lcb_features"],
                fmb_features=architecture["fmb_features"],
                fmb_rank=architecture["fmb_rank"],
                fmb_hidden_dims=tuple(architecture["fmb_hidden_dims"]),
                dropout=architecture["dropout"],
                layer_norm=architecture["layer_norm"],
            )
        if config.model_key == "dcdn_core":
            reciprocal = architecture["reciprocal"]
            if not isinstance(reciprocal, Mapping):
                raise TrainingContractError(
                    "frozen DCDN reciprocal contract must be a mapping"
                )
            return DCDNCoreConfig(
                num_layers=architecture["num_layers"],
                div_eps=reciprocal["epsilon"],
                product_init_scale=architecture["product_init_scale"],
                div_init_scale=architecture["division_init_scale"],
                runtime_value_checks=config.runtime_value_checks,
            )
        if config.model_key == "dcnv2_capacity_matched_core":
            if capacity_artifact is None:
                return None
            candidates = capacity_artifact.get("candidates")
            if not isinstance(candidates, Mapping):
                raise TrainingContractError(
                    "full capacity artifact candidates must be a mapping"
                )
            candidate = candidates.get(config.candidate_id)
            if not isinstance(candidate, Mapping):
                raise TrainingContractError(
                    "capacity candidate is absent from the full artifact"
                )
            low_rank = candidate.get("low_rank")
            if type(low_rank) is not int or low_rank <= 0:
                raise TrainingContractError(
                    "capacity candidate low_rank must be a positive integer"
                )
            return DCNv2CapacityCoreConfig(
                num_layers=architecture["num_cross_layers"],
                low_rank=low_rank,
                num_experts=architecture["num_experts"],
                product_init_scale=0.1,
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingContractError(
            f"invalid frozen architecture for {config.model_key}: {exc}"
        ) from exc
    raise TrainingContractError("model is outside the frozen architecture map")


def _validate_frozen_interaction_contract(
    config: TrainingConfig,
    interaction_config: InteractionConfig,
    search_lock: SearchSpaceLock,
    *,
    capacity_artifact: Mapping[str, Any] | None = None,
) -> None:
    expected = _expected_frozen_interaction_config(
        config,
        search_lock,
        capacity_artifact=capacity_artifact,
    )
    if expected is None:
        return
    if type(interaction_config) is not type(expected) or interaction_config != expected:
        raise TrainingContractError(
            "runtime interaction config differs from frozen search architecture"
        )


def _validated_capacity_candidate(
    config: TrainingConfig,
    capacity_artifact: Mapping[str, Any],
) -> Mapping[str, Any]:
    if config.model_key != "dcnv2_capacity_matched_core":
        raise TrainingContractError(
            "capacity decision validation requires the capacity-control model"
        )
    decision = capacity_artifact.get("decision")
    candidates = capacity_artifact.get("candidates")
    if not isinstance(decision, Mapping) or not isinstance(candidates, Mapping):
        raise TrainingContractError(
            "capacity artifact decision/candidates must be mappings"
        )
    selected_candidate = decision.get("selected_candidate")
    if selected_candidate != config.candidate_id:
        raise TrainingContractError(
            "resolved capacity candidate differs from metric-free decision"
        )
    candidate = candidates.get(config.candidate_id)
    if not isinstance(candidate, Mapping) or (
        decision.get("selected_low_rank") != candidate.get("low_rank")
    ):
        raise TrainingContractError(
            "capacity decision rank differs from selected candidate"
        )
    return candidate


def _validate_capacity_projection_for_feature_regime(
    config: TrainingConfig,
    *,
    controlled: ProvenancedControlledModel,
    artifact: Mapping[str, Any],
    projection: Mapping[str, Any],
    search_lock: SearchSpaceLock,
) -> None:
    """Recompute C+P capacity controls without misapplying them to C+P+E."""

    if config.feature_regime == "C+P+E":
        if controlled.model_key != "dcnv2_full_rank":
            raise TrainingContractError(
                "C+P+E capacity provenance is restricted to dcnv2_full_rank"
            )
        return
    if config.feature_regime != "C+P":
        raise TrainingContractError(
            "unsupported feature regime for capacity provenance"
        )
    geometry = artifact["geometry"]
    mechanical = resolve_metric_free_capacity_preflight(
        encoder_spec=controlled.encoder_spec,
        dcdn_config=DCDNCoreConfig(
            num_layers=geometry["dcdn_num_layers"],
            div_eps=1e-2,
            product_init_scale=0.1,
            div_init_scale=0.1,
            runtime_value_checks=False,
        ),
        ordinary_dcnv2_config=DCNv2FullRankConfig(
            num_layers=geometry["ordinary_dcnv2_num_layers"],
            product_init_scale=0.1,
        ),
        capacity_num_layers=geometry["capacity_num_layers"],
        capacity_num_experts=geometry["capacity_num_experts"],
        capacity_product_init_scale=0.1,
        head_config=HeadConfig(hidden_dims=(), dropout=0.0, output_dim=1),
        init_seed=config.seed,
        search_lock=search_lock,
    )
    if projection != mechanical:
        raise TrainingContractError(
            "full capacity artifact projection differs from mechanical resolver"
        )


def _expected_capacity_parameter_count_for_feature_regime(
    config: TrainingConfig,
    *,
    controlled_model_key: str,
    artifact: Mapping[str, Any],
) -> int | None:
    """Bind C+P capacity counts without treating C+P+E as that control."""

    if config.feature_regime == "C+P+E":
        if controlled_model_key != "dcnv2_full_rank":
            raise TrainingContractError(
                "C+P+E capacity provenance is restricted to dcnv2_full_rank"
            )
        return None
    if config.feature_regime != "C+P":
        raise TrainingContractError(
            "unsupported feature regime for capacity provenance"
        )
    if controlled_model_key == "dcdn_core":
        return int(artifact["target"]["parameters"])
    if controlled_model_key == "dcnv2_full_rank":
        return int(artifact["ordinary_dcnv2"]["parameters"])
    if controlled_model_key == "dcnv2_capacity_matched_core":
        return int(_validated_capacity_candidate(config, artifact)["parameters"])
    return None


def validate_frozen_training_contract(
    config: TrainingConfig,
    *,
    model: (
        TrainableEncodedModel
        | MultiTargetTrainableEncodedModel
        | ProvenancedControlledModel
        | ProvenancedMultiTargetModel
    ),
    search_lock: SearchSpaceLock,
) -> None:
    """Bind one resolved run to the frozen search/model/common-head contract."""

    controlled = _controlled_model(model)
    if search_lock.digest != config.search_space_sha256:
        raise TrainingContractError("search-space digest mismatch")
    multi_target = isinstance(controlled, ProvenancedMultiTargetModel)
    expected_matrix_digest = (
        "1b73dcb34a565f192cb66669163f71bac244884401e9c958a2cc836b2de36946"
        if multi_target
        else search_lock.experiment_matrix_digest
    )
    if expected_matrix_digest != config.experiment_matrix_sha256:
        raise TrainingContractError("experiment-matrix digest mismatch")
    if controlled.model_key != config.model_key:
        raise TrainingContractError("resolved model_key does not match built model")
    spec = controlled.encoder_spec
    if spec.full_feature_schema_sha256 != config.full_schema_sha256:
        raise TrainingContractError("full ordered schema hash mismatch")
    if spec.operand_schema_sha256 != config.operand_schema_sha256:
        raise TrainingContractError("ordered P operand schema hash mismatch")
    if _mask_sha256(spec.denominator_mask) != config.denominator_mask_sha256:
        raise TrainingContractError("semantic denominator-mask hash mismatch")
    if spec.representation_dim != 64:
        raise TrainingContractError("common representation width must remain 64")
    expected_output_dim = len(controlled.target_spec) if multi_target else 1
    if controlled.head_config != HeadConfig(
        hidden_dims=(), dropout=0.0, output_dim=expected_output_dim
    ):
        raise TrainingContractError("model does not use the frozen common linear head")
    if multi_target and (
        config.loss != "macro_nloss"
        or config.selection_metric != "macro_nloss"
        or controlled.target_spec.contract_sha256
        != "850251f3808910ca1ffe12838c60d533631b6daba431f56c22eb77acbf9deb59"
    ):
        raise TrainingContractError("multi-target objective/spec contract drifted")
    _validate_frozen_interaction_contract(
        config,
        controlled.interaction.config,
        search_lock,
    )
    if isinstance(model, TrainableEncodedModel | MultiTargetTrainableEncodedModel):
        if model.encoder.spec.dataset_id != config.dataset_id:
            raise TrainingContractError("joint encoder dataset_id mismatch")
        if model.encoder.spec.feature_regime != config.feature_regime:
            raise TrainingContractError("joint encoder feature_regime mismatch")
        if model.encoder.spec.encoder_contract_sha256 != config.encoder_contract_sha256:
            raise TrainingContractError("joint encoder lock digest mismatch")
        if model.encoder.run_seed != config.seed:
            raise TrainingContractError("joint encoder seed mismatch")
        if model.encoder.runtime_value_checks != config.runtime_value_checks:
            raise TrainingContractError("joint encoder runtime check mode mismatch")
        interaction_config = controlled.interaction.config
        if isinstance(interaction_config, DCDNCoreConfig) and (
            interaction_config.runtime_value_checks != config.runtime_value_checks
        ):
            raise TrainingContractError("DCDN core runtime check mode mismatch")

    if config.model_key in _TUNED_CANDIDATES:
        candidates = _TUNED_CANDIDATES[config.model_key]
        if config.candidate_id not in candidates:
            raise TrainingContractError("candidate is outside the frozen two-LR grid")
        if config.learning_rate != candidates[config.candidate_id]:
            raise TrainingContractError("candidate learning rate mismatch")
    elif config.model_key == "wukong_small":
        if (
            config.candidate_id != "wukong_small_paper_explicit_min"
            or config.learning_rate != 3e-4
        ):
            raise TrainingContractError("Wukong-small is a frozen no-tune candidate")
    elif config.model_key == "dcnv2_capacity_matched_core":
        if (
            config.candidate_id != CAPACITY_CANDIDATE_ID
            or config.learning_rate != SELECTED_LEARNING_RATE
        ):
            raise TrainingContractError(
                "capacity control must use authorized lower rank and DCNv2 LR"
            )
    else:
        raise TrainingContractError("model is outside the frozen public suite")

    common = search_lock.payload["common_contract"]

    def require_exact_common_fields(
        section: Mapping[str, Any],
        section_name: str,
        expected_fields: Mapping[str, Any],
    ) -> None:
        for name, expected_value in expected_fields.items():
            actual = section.get(name)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise TrainingContractError(
                    f"frozen {section_name} field mismatch: {name}"
                )

    optimizer = common["optimizer"]
    require_exact_common_fields(
        optimizer,
        "optimizer",
        {"family": "AdamW", "decoupled_weight_decay": True},
    )
    betas = optimizer.get("betas")
    if not isinstance(betas, list) or tuple(betas) != config.adam_betas:
        raise TrainingContractError("AdamW betas differ from frozen common contract")
    if optimizer.get("epsilon") != config.adam_epsilon:
        raise TrainingContractError("AdamW epsilon differs from frozen common contract")
    if optimizer.get("weight_decay") != config.weight_decay:
        raise TrainingContractError(
            "AdamW weight decay differs from frozen common contract"
        )

    precision = common["precision"]
    require_exact_common_fields(
        precision,
        "precision",
        {
            "parameter_dtype": "float32",
            "autocast_enabled": True,
            "autocast_dtype": "bfloat16",
            "optimizer_state_dtype": "float32",
            "loss_accumulation_dtype": "float32",
            "fp16_forbidden": True,
        },
    )
    if config.autocast_bfloat16 is not precision["autocast_enabled"]:
        raise TrainingContractError("BF16 autocast differs from frozen common contract")

    stopping = common["stopping"]
    require_exact_common_fields(
        stopping,
        "stopping",
        {
            "minimum_delta": 0.0,
            "restore_best_validation_checkpoint": True,
            "maximum_examples_is_hard": True,
            "validation_cadence_is_examples_based": True,
        },
    )
    if stopping.get("patience_validation_checks") != (
        config.patience_validation_checks
    ):
        raise TrainingContractError(
            "validation patience differs from frozen common contract"
        )
    dataset_stop = stopping["datasets"].get(config.dataset_id)
    if not isinstance(dataset_stop, Mapping):
        raise TrainingContractError("dataset is outside the frozen stopping contract")
    expected = {
        "maximum_train_examples": config.maximum_train_examples,
        "validate_every_examples": config.validate_every_examples,
        "selection_direction": config.selection_direction,
    }
    if not multi_target:
        expected["loss"] = config.loss
        expected["selection_metric"] = config.selection_metric
    for name, value in expected.items():
        if dataset_stop.get(name) != value:
            raise TrainingContractError(f"frozen stopping field mismatch: {name}")

    clipping = common["gradient_clipping"]
    require_exact_common_fields(
        clipping,
        "gradient clipping",
        {"kind": "global_l2_norm", "apply_after_unscale": True},
    )
    if clipping.get("max_norm") != config.gradient_clip_l2:
        raise TrainingContractError(
            "gradient clipping norm differs from frozen common contract"
        )

    batching = common["batching"]
    require_exact_common_fields(
        batching,
        "batching",
        {
            "one_process_per_gpu": True,
            "ddp": False,
            "gradient_accumulation_preserves_effective_batch": True,
            "one_microbatch_reduction_after_first_oom": True,
            "second_oom_stops_run": True,
        },
    )
    if (
        batching.get("effective_batch_per_independent_run")
        != config.effective_batch_size
    ):
        raise TrainingContractError("effective batch differs from frozen common budget")
    if (
        config.allow_one_oom_microbatch_reduction
        is not (batching["one_microbatch_reduction_after_first_oom"])
    ):
        raise TrainingContractError(
            "one-OOM microbatch policy differs from frozen common contract"
        )


def _tuple_field(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    values = dict(payload)
    if name in values:
        if not isinstance(values[name], list | tuple):
            raise TypeError(f"{name} must be a sequence")
        values[name] = tuple(values[name])
    return values


def resolved_model_components(
    payload: Mapping[str, Any],
) -> tuple[InteractionConfig, HeadConfig, int]:
    """Parse one exact interaction/head/seed record for the joint factory."""

    if set(payload) != {"interaction_config", "head_config", "init_seed"}:
        raise TrainingContractError(
            "resolved model components must contain exactly interaction_config, "
            "head_config, and init_seed"
        )
    interaction_values = dict(payload["interaction_config"])
    model_key = interaction_values.pop("model_key", None)
    constructors: dict[str, tuple[type[Any], tuple[str, ...]]] = {
        "dnn": (MLPConfig, ("hidden_dims",)),
        "deepfm": (DeepFMConfig, ("deep_hidden_dims",)),
        "dcn": (DCNConfig, ()),
        "autoint_plus": (AutoIntPlusConfig, ("dnn_hidden_dims",)),
        "dcnv2_full_rank": (DCNv2FullRankConfig, ()),
        "dcnv2_capacity_matched_core": (DCNv2CapacityCoreConfig, ()),
        "afn_plus": (AFNPlusConfig, ("afn_hidden_dims", "dnn_hidden_dims")),
        "wukong_small": (WukongSmallConfig, ("fmb_hidden_dims",)),
        "dcdn_core": (DCDNCoreConfig, ()),
    }
    if model_key not in constructors:
        raise TrainingContractError("resolved model_key is outside the frozen suite")
    constructor, tuple_fields = constructors[model_key]
    for name in tuple_fields:
        interaction_values = _tuple_field(interaction_values, name)
    try:
        interaction_config = constructor(**interaction_values)
    except (TypeError, ValueError) as exc:
        raise TrainingContractError(f"invalid interaction config: {exc}") from exc

    head_values = _tuple_field(dict(payload["head_config"]), "hidden_dims")
    head_config = HeadConfig(**head_values)
    init_seed = payload["init_seed"]
    if isinstance(init_seed, bool) or not isinstance(init_seed, int) or init_seed < 0:
        raise TrainingContractError("resolved model init_seed is invalid")
    return interaction_config, head_config, init_seed


def build_model_from_resolved_config(
    payload: Mapping[str, Any],
) -> ProvenancedControlledModel:
    """Build only through the audited suite factory from an exact resolved map."""

    if set(payload) != {
        "encoder_spec",
        "interaction_config",
        "head_config",
        "init_seed",
    }:
        raise TrainingContractError(
            "resolved model config must contain exactly encoder_spec, "
            "interaction_config, head_config, and init_seed"
        )
    encoder_values = dict(payload["encoder_spec"])
    encoder_values["denominator_mask"] = tuple(encoder_values["denominator_mask"])
    encoder_spec = EncoderSpec(**encoder_values)
    interaction_config, head_config, init_seed = resolved_model_components(
        {
            "interaction_config": payload["interaction_config"],
            "head_config": payload["head_config"],
            "init_seed": payload["init_seed"],
        }
    )
    return build_controlled_model(
        encoder_spec=encoder_spec,
        interaction_config=interaction_config,
        head_config=head_config,
        init_seed=init_seed,
    )


def count_trainable_non_embedding_parameters(model: nn.Module) -> int:
    """Count unique trainable parameters while excluding embedding tables."""

    excluded = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        if parameter.requires_grad and identity not in excluded:
            total += parameter.numel()
    return int(total)


def _biased_linear_flops(input_width: int, output_width: int) -> int:
    # Per output: input_width multiplies, input_width-1 reductions, one bias add.
    return 2 * input_width * output_width


def _unbiased_linear_flops(input_width: int, output_width: int) -> int:
    return output_width * (2 * input_width - 1)


def deterministic_forward_flops_per_example(
    model: ProvenancedControlledModel,
) -> int:
    """Count one forward example under a frozen arithmetic convention.

    Multiplication, addition, reciprocal, and each scalar transcendental count
    as one FLOP. Comparisons, masks, concatenation, and memory movement count
    as zero. Embedding lookup is outside the controlled post-encoder model.
    """

    if not isinstance(model, ProvenancedControlledModel):
        raise TypeError("model must come from the controlled suite factory")
    spec = model.encoder_spec
    interaction = model.interaction.config
    if model.head_config != HeadConfig(hidden_dims=(), dropout=0.0, output_dim=1):
        raise TrainingContractError("capacity FLOPs require the frozen linear head")

    if isinstance(interaction, DCDNCoreConfig):
        width = spec.operand_dim
        reciprocal_flops = width
        layer_flops = 4 * width * width + 4 * width
        interaction_flops = interaction.num_layers * layer_flops + reciprocal_flops
    elif isinstance(interaction, DCNv2FullRankConfig):
        width = spec.flat_dim
        layer_flops = _biased_linear_flops(width, width) + 2 * width
        interaction_flops = interaction.num_layers * layer_flops
    elif isinstance(interaction, DCNv2CapacityCoreConfig):
        width = spec.operand_dim
        rank = interaction.low_rank
        experts = interaction.num_experts
        expert_flops = (
            _unbiased_linear_flops(width, rank)
            + rank  # first tanh
            + _unbiased_linear_flops(rank, rank)
            + rank  # second tanh
            + _unbiased_linear_flops(rank, width)
            + 2 * width  # shared bias add and anchor multiply
            + _unbiased_linear_flops(width, 1)  # gate projection
        )
        softmax_flops = 3 * experts - 1
        weighted_residual_flops = 2 * experts * width
        interaction_flops = interaction.num_layers * (
            experts * expert_flops + softmax_flops + weighted_residual_flops
        )
    else:
        raise TrainingContractError(
            "capacity FLOPs support only DCDN, ordinary DCNv2, and its brackets"
        )

    adapter_flops = _biased_linear_flops(
        model.adapter.raw_dim, model.adapter.representation_dim
    )
    head_flops = _biased_linear_flops(model.encoder_spec.representation_dim, 1)
    total = interaction_flops + adapter_flops + head_flops
    if total <= 0:
        raise TrainingContractError("capacity FLOP count must be positive")
    return int(total)


def _capacity_counts(model: ProvenancedControlledModel) -> dict[str, int]:
    return {
        "parameters": count_trainable_non_embedding_parameters(model),
        "flops": deterministic_forward_flops_per_example(model),
    }


def _within_capacity_tolerance(
    candidate: Mapping[str, int],
    target: Mapping[str, int],
    tolerance: float,
) -> bool:
    return all(
        1.0 - tolerance <= candidate[name] / target[name] <= 1.0 + tolerance
        for name in ("parameters", "flops")
    )


def _capacity_distance(
    candidate: Mapping[str, int], target: Mapping[str, int]
) -> float:
    return sum(
        abs(math.log(candidate[name] / target[name]))
        for name in ("parameters", "flops")
    )


def resolve_metric_free_capacity_preflight(
    *,
    encoder_spec: EncoderSpec,
    dcdn_config: DCDNCoreConfig,
    ordinary_dcnv2_config: DCNv2FullRankConfig,
    capacity_num_layers: int,
    capacity_num_experts: int,
    capacity_product_init_scale: float,
    head_config: HeadConfig,
    init_seed: int,
    search_lock: SearchSpaceLock,
    event_log: CompactEventLog | None = None,
) -> dict[str, Any]:
    """Resolve the public core capacity control without data, labels, or metrics."""

    target_model = build_controlled_model(
        encoder_spec=encoder_spec,
        interaction_config=dcdn_config,
        head_config=head_config,
        init_seed=init_seed,
    )
    ordinary_model = build_controlled_model(
        encoder_spec=encoder_spec,
        interaction_config=ordinary_dcnv2_config,
        head_config=head_config,
        init_seed=init_seed,
    )
    target = _capacity_counts(target_model)
    ordinary = _capacity_counts(ordinary_model)

    def bracket_model(rank: int) -> ProvenancedControlledModel:
        return build_controlled_model(
            encoder_spec=encoder_spec,
            interaction_config=DCNv2CapacityCoreConfig(
                num_layers=capacity_num_layers,
                low_rank=rank,
                num_experts=capacity_num_experts,
                product_init_scale=capacity_product_init_scale,
            ),
            head_config=head_config,
            init_seed=init_seed,
        )

    lower_rank: int | None = None
    lower_counts: dict[str, int] | None = None
    for rank in range(1, encoder_spec.operand_dim + 1):
        counts = _capacity_counts(bracket_model(rank))
        if counts["parameters"] <= target["parameters"]:
            lower_rank = rank
            lower_counts = counts
            continue
        break
    if lower_rank is None or lower_counts is None:
        raise TrainingContractError(
            "no positive low-rank capacity bracket lies below the DCDN target"
        )
    upper_rank = lower_rank + 1
    upper_counts = _capacity_counts(bracket_model(upper_rank))
    if upper_counts["parameters"] < target["parameters"]:
        raise TrainingContractError("adjacent capacity ranks do not bracket target")

    capacity_model = search_lock.payload["models"]["dcnv2_low_rank_mix_capacity"]
    tolerance = float(
        capacity_model["ordinary_full_rank_reuse_check"]["tolerance_fraction"]
    )
    candidates = {
        "capacity_rank_lower": {"low_rank": lower_rank, **lower_counts},
        "capacity_rank_upper": {"low_rank": upper_rank, **upper_counts},
    }
    if _within_capacity_tolerance(ordinary, target, tolerance):
        decision = {
            "mode": "reuse_ordinary",
            "selected_candidate": "dcnv2_full_rank",
        }
    else:
        eligible = [
            (candidate_id, counts)
            for candidate_id, counts in candidates.items()
            if _within_capacity_tolerance(counts, target, tolerance)
        ]
        if not eligible:
            decision = {"mode": "unavailable", "selected_candidate": None}
        else:
            selected = min(
                eligible,
                key=lambda item: (
                    _capacity_distance(item[1], target),
                    0 if item[0] == "capacity_rank_lower" else 1,
                ),
            )[0]
            decision = {
                "mode": "fallback_bracket",
                "selected_candidate": selected,
            }

    resolution: dict[str, Any] = {
        "metrics_consulted": False,
        "resolved_before_first_model_metric": True,
        "flop_counting_contract": CAPACITY_FLOP_CONTRACT,
        "embedding_lookup_excluded": True,
        "adapter_and_common_head_included": True,
        "target": target,
        "ordinary": ordinary,
        "candidates": candidates,
        "decision": decision,
    }
    validate_capacity_resolution(search_lock.payload, resolution)
    if event_log is not None:
        event_log.emit("capacity_resolution", **resolution)
    return resolution


class TrainingHarness:
    """One-process, one-device deterministic train/validation harness."""

    def __init__(
        self,
        *,
        model: (
            TrainableEncodedModel
            | MultiTargetTrainableEncodedModel
            | ProvenancedControlledModel
            | ProvenancedMultiTargetModel
        ),
        config: TrainingConfig,
        safety: SafetyAttestation,
        search_lock: SearchSpaceLock,
        capacity_resolution: Mapping[str, Any] | None,
        operand_feature_order: Sequence[str],
        run_directory: str | Path,
        device: torch.device | str,
        capacity_artifact_path: str | Path | None = None,
        capacity_lr_binding_path: str | Path | None = None,
        repository_root: str | Path | None = None,
        event_log: CompactEventLog | None = None,
        heartbeat: Callable[[], None] | None = None,
    ) -> None:
        controlled = _controlled_model(model)
        self.raw_execution = isinstance(
            model, TrainableEncodedModel | MultiTargetTrainableEncodedModel
        )
        if not self.raw_execution:
            assert_synthetic_encoded_execution(config.dataset_id)
        self.model = model
        self.controlled_model = controlled
        self.config = config
        self.safety = safety
        self.search_lock = search_lock
        self.device = torch.device(device)
        self.repository_root: Path | None = None
        if repository_root is not None:
            try:
                resolved_root = Path(repository_root).resolve(strict=True)
            except OSError as exc:
                raise TrainingContractError("repository root is unavailable") from exc
            if not resolved_root.is_dir():
                raise TrainingContractError("repository root must be a directory")
            self.repository_root = resolved_root
        if self.device.type not in {"cpu", "cuda"}:
            raise TrainingContractError("training device must be CPU or one CUDA GPU")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise TrainingContractError("CUDA device requested but unavailable")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise TrainingContractError("DDP/distributed process groups are forbidden")
        if isinstance(model, nn.parallel.DistributedDataParallel):
            raise TrainingContractError("DDP-wrapped models are forbidden")

        self.capacity_lr_binding: CapacityBindingEvidence | None = None
        self.capacity_training_binding: Mapping[str, Any] | None = None
        if controlled.model_key == "dcnv2_capacity_matched_core":
            if (
                capacity_lr_binding_path is None
                or capacity_artifact_path is None
                or repository_root is None
            ):
                raise TrainingContractError(
                    "capacity control requires compact LR authorization, full "
                    "capacity artifact, and repository root"
                )
            try:
                binding = load_capacity_lr_binding(
                    capacity_lr_binding_path, repository_root
                )
                phase_capacity = load_verified_capacity_artifact(
                    Path(capacity_artifact_path)
                )
                resolved_phase = search_lock.require_phase(
                    "capacity_control_training",
                    resolutions=capacity_phase_resolutions(
                        phase_capacity,
                        selected_learning_rate=binding.ordinary.learning_rate,
                    ),
                )
            except Exception as exc:
                raise TrainingContractError(
                    f"capacity LR authorization failed before optimizer: {exc}"
                ) from exc
            resolved_lr = resolved_phase["models"]["dcnv2_low_rank_mix_capacity"][
                "base_config"
            ]["optimizer"]["learning_rate"]
            interaction = controlled.interaction.config
            if (
                config.candidate_id != CAPACITY_CANDIDATE_ID
                or config.learning_rate != SELECTED_LEARNING_RATE
                or binding.ordinary.learning_rate != SELECTED_LEARNING_RATE
                or resolved_lr != SELECTED_LEARNING_RATE
                or config.capacity_resolution_sha256 != CAPACITY_RESOLUTION_SHA256
                or config.ordinary_dcnv2_selection_sha256 != ORDINARY_SELECTION_SHA256
                or config.capacity_lr_binding_sha256 != binding.sha256
                or safety.protocol_gate.digest != PREMETRIC_PROTOCOL_SHA256
                or not isinstance(interaction, DCNv2CapacityCoreConfig)
                or interaction.low_rank != CAPACITY_LOW_RANK
                or interaction.num_layers != 2
                or interaction.num_experts != 4
                or interaction.product_init_scale != 0.1
            ):
                raise TrainingContractError(
                    "capacity config, protocol, geometry, LR, or authorization drifted"
                )
            self.capacity_lr_binding = binding
            self.capacity_training_binding = capacity_training_binding_payload(
                capacity_resolution_sha256=config.capacity_resolution_sha256,
                ordinary_dcnv2_selection_sha256=(
                    config.ordinary_dcnv2_selection_sha256
                ),
                capacity_lr_binding_sha256=config.capacity_lr_binding_sha256,
            )
        elif capacity_lr_binding_path is not None:
            raise TrainingContractError(
                "compact capacity LR authorization is forbidden for non-capacity models"
            )

        validate_frozen_training_contract(config, model=model, search_lock=search_lock)
        safety.validate_common(config)
        if self.raw_execution:
            if capacity_artifact_path is None or repository_root is None:
                raise TrainingContractError(
                    "real raw execution requires the full verified capacity artifact "
                    "and repository root"
                )
            artifact_path = Path(capacity_artifact_path)
            try:
                artifact = load_verified_capacity_artifact(artifact_path)
                validate_capacity_artifact(artifact)
                verify_repository_bindings(artifact, repository_root)
                artifact_digest = capacity_sha256_file(artifact_path)
                projection = capacity_resolution_projection(artifact)
            except Exception as exc:
                raise TrainingContractError(
                    f"full capacity artifact validation failed: {exc}"
                ) from exc
            if artifact_digest != config.capacity_resolution_sha256:
                raise TrainingContractError(
                    "full capacity artifact SHA differs from resolved config"
                )
            _validate_frozen_interaction_contract(
                config,
                controlled.interaction.config,
                search_lock,
                capacity_artifact=artifact,
            )
            if controlled.model_key == "dcnv2_capacity_matched_core":
                _validated_capacity_candidate(config, artifact)
            if (
                capacity_resolution is not None
                and dict(capacity_resolution) != projection
            ):
                raise TrainingContractError(
                    "provided capacity projection differs from the full artifact"
                )
            _validate_capacity_projection_for_feature_regime(
                config,
                controlled=controlled,
                artifact=artifact,
                projection=projection,
                search_lock=search_lock,
            )
            self.capacity_artifact: Mapping[str, Any] | None = artifact
            self.capacity_resolution = projection
            self.capacity_resolution_sha256 = artifact_digest
        else:
            if capacity_artifact_path is not None:
                raise TrainingContractError(
                    "synthetic encoded tests must not impersonate a real artifact"
                )
            if capacity_resolution is None:
                raise TrainingContractError(
                    "synthetic tests require a metric-free capacity projection"
                )
            try:
                capacity_bytes = json.dumps(
                    capacity_resolution,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            except (TypeError, ValueError) as exc:
                raise TrainingContractError(
                    f"capacity resolution is not canonical JSON: {exc}"
                ) from exc
            self.capacity_artifact = None
            self.capacity_resolution = json.loads(capacity_bytes)
            validate_capacity_resolution(search_lock.payload, self.capacity_resolution)
            self.capacity_resolution_sha256 = hashlib.sha256(capacity_bytes).hexdigest()
            if self.capacity_resolution_sha256 != config.capacity_resolution_sha256:
                raise TrainingContractError(
                    "synthetic capacity projection digest mismatch"
                )

        if self.capacity_resolution_sha256 != safety.capacity_resolution_sha256:
            raise TrainingContractError("capacity digest differs from safety evidence")
        if self.raw_execution:
            assert isinstance(
                model, TrainableEncodedModel | MultiTargetTrainableEncodedModel
            )
            model.encoder.spec.assert_source_identity(safety.train_source_identity)
            model.encoder.spec.assert_source_identity(safety.validation_source_identity)
            controlled_count = count_trainable_non_embedding_parameters(controlled)
            joint_count = count_trainable_non_embedding_parameters(model)
            encoder_count = count_trainable_non_embedding_parameters(model.encoder)
            if encoder_count != 0 or joint_count != controlled_count:
                raise TrainingContractError(
                    "joint capacity must exclude all and only nn.Embedding parameters"
                )
            embedding_ids = {
                id(parameter)
                for module in model.encoder.modules()
                if isinstance(module, nn.Embedding)
                for parameter in module.parameters(recurse=False)
            }
            encoder_ids = {
                id(parameter)
                for parameter in model.encoder.parameters()
                if parameter.requires_grad
            }
            if not encoder_ids or encoder_ids != embedding_ids:
                raise TrainingContractError(
                    "frozen shared encoder has unexpected parameter ownership"
                )
            assert self.capacity_artifact is not None
            expected_capacity_count = (
                None
                if isinstance(model, MultiTargetTrainableEncodedModel)
                else _expected_capacity_parameter_count_for_feature_regime(
                    config,
                    controlled_model_key=controlled.model_key,
                    artifact=self.capacity_artifact,
                )
            )
            if (
                expected_capacity_count is not None
                and joint_count != expected_capacity_count
            ):
                raise TrainingContractError(
                    "composite non-embedding count differs from capacity artifact"
                )
        names = tuple(operand_feature_order)
        if len(names) != controlled.encoder_spec.operand_dim or len(set(names)) != len(
            names
        ):
            raise TrainingContractError(
                "ordered P feature names do not match operand width"
            )
        if names != safety.train_source_identity.ordered_operand_fields:
            raise TrainingContractError(
                "ordered P feature names differ from exact raw source identity"
            )
        self.operand_feature_order = names

        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint_path = self.run_directory / "last.pt"
        self.best_checkpoint_path: Path | None = None
        self.event_log = event_log or CompactEventLog(
            self.run_directory / "history.jsonl"
        )
        self.heartbeat = heartbeat
        self.current_microbatch = config.microbatch_size
        self.oom_count = 0
        self.global_step = 0
        self.examples_seen = 0
        self.validation_checks = 0
        self.best_metric: float | None = None
        self.bad_validation_checks = 0
        self.best_checkpoint_record: CheckpointRecord | None = None
        self.last_checkpoint_record: CheckpointRecord | None = None
        self.selected_best_checkpoint_sha256: str | None = None

        seed_everything(config.seed, seed_cuda=self.device.type == "cuda")
        self.model.to(device=self.device, dtype=torch.float32)
        parameter_dtypes = {parameter.dtype for parameter in self.model.parameters()}
        if parameter_dtypes != {torch.float32}:
            raise TrainingContractError("all model parameters must remain FP32")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=config.adam_betas,
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        optimizer_parameter_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        trainable_parameter_ids = {
            id(parameter)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        }
        if optimizer_parameter_ids != trainable_parameter_ids:
            raise TrainingContractError(
                "optimizer does not own every unique trainable composite parameter"
            )
        self.scheduler = None
        self.sampler: StatefulPermutationSampler | None = None
        self.monitor = AsyncNonFiniteMonitor(
            self.device, nonfinite_limit=config.numerical_nonfinite_limit
        )
        self._attach_monitor_hooks()

    def _attach_monitor_hooks(self) -> None:
        suffixes = (
            "controlled_model.interaction",
            "controlled_model.adapter",
            "controlled_model.head",
        )
        legacy_names = {"interaction", "adapter", "head"}
        self.monitor.attach_activation_hooks(
            self.model,
            select=lambda name, _module: (
                any(name.endswith(suffix) for suffix in suffixes)
                if self.raw_execution
                else name in legacy_names
            ),
            expected_count=3,
        )

    @property
    def accumulation_steps(self) -> int:
        return self.config.effective_batch_size // self.current_microbatch

    @property
    def checkpoint_protocol_sha256(self) -> str:
        gate = self.safety.protocol_gate
        if not gate.test_unlocked:
            return gate.digest
        parent = gate.parent_premetric_digest
        if parent is None:
            raise TrainingContractError(
                "final protocol lacks its parent premetric checkpoint digest"
            )
        return parent

    def _resolved_checkpoint_config(self) -> dict[str, Any]:
        payload = {
            "training": training_config_payload(self.config),
            "joint_model_contract_sha256": self.model.contract_sha256,
            "joint_model_contract": self.model.contract_state(),
            "controlled_model_contract_sha256": (self.controlled_model.contract_sha256),
            "encoder_contract_sha256": self.config.encoder_contract_sha256,
            "capacity_resolution_sha256": self.capacity_resolution_sha256,
            "capacity_artifact_role": (
                "full_verified_capacity_artifact"
                if self.raw_execution
                else "synthetic_projection_only"
            ),
            "checkpoint_protocol_sha256": self.checkpoint_protocol_sha256,
            "examples_budget": {
                "declared_hard_maximum": self.config.maximum_train_examples,
                "effective_batch_size": self.config.effective_batch_size,
                "realized_floor_boundary": (self.config.realized_maximum_examples),
                "rounding": "floor_never_exceed_declared_hard_maximum",
                "validation_threshold_policy": (
                    "first_completed_effective_batch_at_or_after_declared_threshold"
                ),
            },
            "train_runtime_identity": asdict(self.safety.train_runtime_identity),
            "validation_runtime_identity": asdict(
                self.safety.validation_runtime_identity
            ),
            "train_source_identity": asdict(self.safety.train_source_identity),
            "validation_source_identity": asdict(
                self.safety.validation_source_identity
            ),
            "allowed_oom_fallback": {
                "initial_microbatch": self.config.microbatch_size,
                "minimum_microbatch": max(1, self.config.microbatch_size // 2),
                "effective_batch_unchanged": self.config.effective_batch_size,
                "maximum_oom_events": 1,
            },
        }
        if self.capacity_training_binding is not None:
            payload["capacity_training_binding"] = dict(self.capacity_training_binding)
        return payload

    def checkpoint_metadata(self) -> CheckpointMetadata:
        controlled = self.controlled_model
        return CheckpointMetadata(
            resolved_config=self._resolved_checkpoint_config(),
            data_hash=self.safety.train_runtime_identity.data_manifest_sha256,
            feature_hash=self.safety.train_runtime_identity.features_sha256,
            protocol_hash=self.checkpoint_protocol_sha256,
            feature_order=self.operand_feature_order,
            denominator_mask=controlled.encoder_spec.denominator_mask,
            div_eps=(
                float(controlled.interaction.config.div_eps)
                if controlled.model_key == "dcdn_core"
                else 1e-2
            ),
            depth=int(getattr(controlled.interaction.config, "num_layers", 1)),
            precision=self.config.checkpoint_precision,
        )

    def _extra_state(self, checkpoint_role: Literal["last", "best"]) -> dict[str, Any]:
        if self.sampler is None:
            raise TrainingContractError("sampler is not initialized")
        return {
            "checkpoint_role": checkpoint_role,
            "selected_best_checkpoint_filename": (
                self.best_checkpoint_path.name
                if checkpoint_role == "last" and self.best_checkpoint_path is not None
                else None
            ),
            "selected_best_checkpoint_sha256": (
                self._best_checkpoint_sha256() if checkpoint_role == "last" else None
            ),
            "examples_seen": self.examples_seen,
            "validation_checks": self.validation_checks,
            "best_metric": self.best_metric,
            "bad_validation_checks": self.bad_validation_checks,
            "oom_count": self.oom_count,
            "current_microbatch": self.current_microbatch,
            "effective_batch_size": self.config.effective_batch_size,
            "sampler_dataset_size": self.sampler.dataset_size,
            "sampler_epoch": self.sampler.epoch,
            "sampler_cursor": self.sampler.cursor,
            "numerical_summary": asdict(self.monitor.summary()),
        }

    def _save(
        self,
        path: Path,
        *,
        checkpoint_role: Literal["last", "best"],
    ) -> CheckpointRecord:
        if self.sampler is None:
            raise TrainingContractError("sampler is not initialized")
        if checkpoint_role == "last":
            if path != self.last_checkpoint_path:
                raise TrainingContractError("last checkpoint path drifted")
        elif (
            path.parent != self.run_directory
            or _BEST_CHECKPOINT_FILENAME_RE.fullmatch(path.name) is None
            or path.exists()
            or path.with_name(f"{path.name}.sha256").exists()
        ):
            raise TrainingContractError(
                "best checkpoint generation path is unsafe or already exists"
            )
        self.monitor.drain()
        self.model.train()
        record = save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler=self.sampler,
            scaler=None,
            global_step=self.global_step,
            epoch=self.sampler.epoch,
            accumulation_step=0,
            accumulation_steps=self.accumulation_steps,
            metadata=self.checkpoint_metadata(),
            mode="train",
            extra_state=self._extra_state(checkpoint_role),
            include_cuda_rng=self.device.type == "cuda",
        )
        self.event_log.emit(
            "checkpoint",
            path=path.name,
            sha256=record.sha256,
            checkpoint_role=checkpoint_role,
            examples_seen=self.examples_seen,
            optimizer_steps=self.global_step,
        )
        return record

    def _next_best_checkpoint_path(self) -> Path:
        for generation in range(10_000):
            candidate = self.run_directory / (
                f"best-step-{self.global_step:012d}-generation-{generation:04d}.pt"
            )
            if (
                not candidate.exists()
                and not candidate.with_name(f"{candidate.name}.sha256").exists()
            ):
                return candidate
        raise TrainingContractError("best checkpoint generation namespace is exhausted")

    def _best_path_from_filename(self, filename: Any) -> Path:
        if (
            not isinstance(filename, str)
            or _BEST_CHECKPOINT_FILENAME_RE.fullmatch(filename) is None
        ):
            raise TrainingContractError("selected best checkpoint filename is invalid")
        return self.run_directory / filename

    def _best_checkpoint_sha256(self) -> str | None:
        if self.best_checkpoint_record is not None:
            return self.best_checkpoint_record.sha256
        if self.best_metric is None:
            return None
        if self.best_checkpoint_path is None:
            raise TrainingContractError(
                "validation-selected best checkpoint path is unavailable"
            )
        try:
            return verify_checkpoint_sidecar(self.best_checkpoint_path)
        except Exception as exc:
            raise TrainingContractError(
                "validation-selected best checkpoint is missing or corrupt"
            ) from exc

    @staticmethod
    def _identity_bytes(identity: RawSourceIdentity) -> bytes:
        return json.dumps(
            asdict(identity),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

    @staticmethod
    def _close_raw_source(source: RawBatchSource) -> None:
        close = getattr(source, "close", None)
        if not callable(close):
            raise TrainingContractError(
                "real RawBatchSource must expose deterministic close()"
            )
        close()

    def _assert_raw_source(self, source: RawBatchSource, split: Split) -> None:
        expected = self.safety.expected_source_identity(split)
        actual = source.identity
        if not isinstance(actual, RawSourceIdentity):
            raise SplitAccessError("raw source returned an invalid identity type")
        if self._identity_bytes(actual) != self._identity_bytes(expected):
            raise SplitAccessError(
                f"{split} raw source identity differs byte-for-byte from attestation"
            )
        if len(source) != expected.row_count:
            raise SplitAccessError("raw source length differs from attested row count")
        if type(source.runtime_value_checks) is not bool:
            raise SplitAccessError("raw source runtime_value_checks is not boolean")
        if source.runtime_value_checks != self.config.runtime_value_checks:
            raise SplitAccessError("raw source runtime check mode mismatch")
        if not isinstance(
            self.model, TrainableEncodedModel | MultiTargetTrainableEncodedModel
        ):
            raise TrainingContractError("raw source requires TrainableEncodedModel")
        if self.model.encoder.runtime_value_checks != source.runtime_value_checks:
            raise SplitAccessError("source and encoder runtime check modes differ")

    def _open_raw_source(
        self,
        split: Literal["train", "validation"],
        source_factory: Callable[[], RawBatchSource],
    ) -> RawBatchSource:
        if not callable(source_factory):
            raise TypeError("real raw execution requires a source factory")
        self.safety.require_split(split, self.config)
        source = source_factory()
        try:
            self._assert_raw_source(source, split)
            if not callable(getattr(source, "close", None)):
                raise TrainingContractError(
                    "real RawBatchSource must expose deterministic close()"
                )
            return source
        except BaseException:
            close = getattr(source, "close", None)
            if callable(close):
                close()
            raise

    def _batch_forward(
        self,
        source: RawBatchSource | EncodedBatchSource,
        indices: Sequence[int],
        *,
        split: Split,
    ) -> tuple[Any, torch.Tensor]:
        before_hooks = self.monitor.activation_hook_observations
        if self.raw_execution:
            assert isinstance(
                self.model, TrainableEncodedModel | MultiTargetTrainableEncodedModel
            )
            raw_source = source
            self._assert_raw_source(raw_source, split)  # type: ignore[arg-type]
            raw_batch = raw_source.batch(indices, device=self.device)  # type: ignore[union-attr]
            if not isinstance(raw_batch, RawFeatureBatch):
                raise RawContractError("raw source batch returned the wrong type")
            raw_batch.validate(
                raw_source.identity,  # type: ignore[union-attr]
                indices,
                runtime_value_checks=raw_source.runtime_value_checks,  # type: ignore[union-attr]
            )
            targets = raw_batch.targets
            with self._autocast():
                output = self.model(raw_batch)
        else:
            encoded, targets = source.batch(indices, device=self.device)  # type: ignore[assignment]
            with self._autocast():
                output = self.model(encoded)
        self.monitor.assert_activation_hooks_observed(before_hooks)
        return output, targets

    def resume(self, source: RawBatchSource | EncodedBatchSource) -> ResumeState:
        if self.raw_execution:
            self._assert_raw_source(source, "train")  # type: ignore[arg-type]
        elif source.split != "train":  # type: ignore[union-attr]
            raise SplitAccessError("training source must declare split=train")
        self.sampler = StatefulPermutationSampler(
            len(source), seed=self.config.seed, shuffle=True
        )
        # A failed drain must occur before any checkpoint component or RNG state
        # is allowed to mutate the live run.
        self.monitor.drain()

        def validate_last_before_mutation(
            extra: Mapping[str, Any], progress: CheckpointProgress
        ) -> None:
            self._validate_checkpoint_extra_state(
                extra,
                progress,
                expected_role="last",
                source_size=len(source),
            )
            best_metric = extra["best_metric"]
            if best_metric is None:
                return
            selected_digest = extra["selected_best_checkpoint_sha256"]
            selected_path = self._best_path_from_filename(
                extra["selected_best_checkpoint_filename"]
            )
            inspected_best = load_model_checkpoint_for_evaluation(
                selected_path,
                model=self.model,
                expected_metadata=self.checkpoint_metadata(),
                expected_mode="train",
                map_location="cpu",
                extra_state_validator=lambda best_extra, best_progress: (
                    self._validate_checkpoint_extra_state(
                        best_extra,
                        best_progress,
                        expected_role="best",
                        source_size=len(source),
                        expected_best_metric=float(best_metric),
                    )
                ),
                restore_model=False,
            )
            if inspected_best.sha256 != selected_digest:
                raise TrainingContractError(
                    "inspected best digest differs from last checkpoint lineage"
                )

        state = load_checkpoint(
            self.last_checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler=self.sampler,
            scaler=None,
            expected_metadata=self.checkpoint_metadata(),
            expected_accumulation_steps=None,
            map_location="cpu",
            extra_state_validator=validate_last_before_mutation,
        )
        self._apply_checkpoint_extra_state(state.extra_state)
        self.event_log.emit(
            "resume",
            sha256=state.sha256,
            checkpoint_role="last",
            examples_seen=self.examples_seen,
            optimizer_steps=self.global_step,
            sampler_epoch=self.sampler.epoch,
            sampler_cursor=self.sampler.cursor,
        )
        return state

    def _validate_checkpoint_extra_state(
        self,
        extra: Mapping[str, Any],
        progress: CheckpointProgress,
        *,
        expected_role: Literal["last", "best"],
        source_size: int,
        expected_best_metric: float | None = None,
    ) -> None:
        required = {
            "checkpoint_role",
            "selected_best_checkpoint_filename",
            "selected_best_checkpoint_sha256",
            "examples_seen",
            "validation_checks",
            "best_metric",
            "bad_validation_checks",
            "oom_count",
            "current_microbatch",
            "effective_batch_size",
            "sampler_dataset_size",
            "sampler_epoch",
            "sampler_cursor",
            "numerical_summary",
        }
        if set(extra) != required:
            raise TrainingContractError("checkpoint harness state keys mismatch")
        if extra["checkpoint_role"] != expected_role:
            raise TrainingContractError(
                "checkpoint role mismatch: "
                f"checkpoint={extra['checkpoint_role']!r}, "
                f"expected={expected_role!r}"
            )
        if progress.mode != "train" or progress.model_training is not True:
            raise TrainingContractError(
                "durable training checkpoint must preserve train mode"
            )
        if progress.accumulation_step != 0:
            raise TrainingContractError("checkpoint is not at an accumulation boundary")
        if extra["effective_batch_size"] != self.config.effective_batch_size:
            raise TrainingContractError("checkpoint effective batch mismatch")

        def nonnegative_integer(name: str) -> int:
            value = extra[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrainingContractError(
                    f"checkpoint {name} must be a non-negative integer"
                )
            return value

        examples_seen = nonnegative_integer("examples_seen")
        validation_checks = nonnegative_integer("validation_checks")
        bad_validation_checks = nonnegative_integer("bad_validation_checks")
        oom_count = nonnegative_integer("oom_count")
        sampler_dataset_size = nonnegative_integer("sampler_dataset_size")
        sampler_epoch = nonnegative_integer("sampler_epoch")
        sampler_cursor = nonnegative_integer("sampler_cursor")
        if source_size <= 0 or sampler_dataset_size != source_size:
            raise TrainingContractError("checkpoint sampler dataset size mismatch")
        if sampler_cursor > source_size:
            raise TrainingContractError("checkpoint sampler cursor is out of range")
        if sampler_epoch != progress.epoch:
            raise TrainingContractError("checkpoint sampler/progress epoch mismatch")
        if examples_seen != progress.global_step * self.config.effective_batch_size:
            raise TrainingContractError(
                "checkpoint examples/optimizer-step equality is violated"
            )
        if examples_seen != sampler_epoch * source_size + sampler_cursor:
            raise TrainingContractError(
                "checkpoint examples/sampler ACK equality is violated"
            )
        if validation_checks > progress.global_step:
            raise TrainingContractError(
                "checkpoint validation count exceeds optimizer steps"
            )
        if bad_validation_checks > validation_checks:
            raise TrainingContractError(
                "checkpoint bad-validation count exceeds validation count"
            )
        if oom_count > int(self.config.allow_one_oom_microbatch_reduction):
            raise TrainingContractError("checkpoint OOM count exceeds frozen limit")
        current_microbatch = extra["current_microbatch"]
        if (
            isinstance(current_microbatch, bool)
            or not isinstance(current_microbatch, int)
            or current_microbatch <= 0
            or self.config.effective_batch_size % current_microbatch
        ):
            raise TrainingContractError("checkpoint current microbatch is invalid")
        if (
            progress.accumulation_steps
            != self.config.effective_batch_size // current_microbatch
        ):
            raise TrainingContractError(
                "checkpoint accumulation contract is inconsistent"
            )
        best_metric = extra["best_metric"]
        if best_metric is not None and (
            isinstance(best_metric, bool)
            or not isinstance(best_metric, int | float)
            or not math.isfinite(float(best_metric))
        ):
            raise TrainingContractError("checkpoint best metric is invalid")
        if expected_role == "best" and best_metric is None:
            raise TrainingContractError(
                "best checkpoint lacks its validation selection metric"
            )
        if expected_best_metric is not None and (
            best_metric is None or float(best_metric) != float(expected_best_metric)
        ):
            raise TrainingContractError(
                "best checkpoint metric differs from selected validation metric"
            )
        selected_best_sha256 = extra["selected_best_checkpoint_sha256"]
        selected_best_filename = extra["selected_best_checkpoint_filename"]
        if expected_role == "best":
            if selected_best_sha256 is not None or selected_best_filename is not None:
                raise TrainingContractError(
                    "best checkpoint cannot recursively bind its own generation"
                )
        elif best_metric is None:
            if selected_best_sha256 is not None or selected_best_filename is not None:
                raise TrainingContractError(
                    "last checkpoint binds a best generation without a best metric"
                )
        else:
            if (
                not isinstance(selected_best_sha256, str)
                or SHA256_RE.fullmatch(selected_best_sha256) is None
            ):
                raise TrainingContractError(
                    "last checkpoint lacks the selected best digest"
                )
            selected_best_path = self._best_path_from_filename(selected_best_filename)
            try:
                actual_best_sha256 = verify_checkpoint_sidecar(selected_best_path)
            except Exception as exc:
                raise TrainingContractError(
                    "selected best checkpoint is missing or corrupt"
                ) from exc
            if actual_best_sha256 != selected_best_sha256:
                raise TrainingContractError(
                    "selected best checkpoint digest differs from last checkpoint"
                )
        self.monitor.validate_summary(extra["numerical_summary"])

    def _apply_checkpoint_extra_state(self, extra: Mapping[str, Any]) -> None:
        self.global_step = (
            int(extra["examples_seen"]) // self.config.effective_batch_size
        )
        self.examples_seen = int(extra["examples_seen"])
        self.validation_checks = int(extra["validation_checks"])
        self.best_metric = (
            None if extra["best_metric"] is None else float(extra["best_metric"])
        )
        self.bad_validation_checks = int(extra["bad_validation_checks"])
        self.oom_count = int(extra["oom_count"])
        self.current_microbatch = int(extra["current_microbatch"])
        self.selected_best_checkpoint_sha256 = extra["selected_best_checkpoint_sha256"]
        selected_filename = extra["selected_best_checkpoint_filename"]
        self.best_checkpoint_path = (
            None
            if selected_filename is None
            else self._best_path_from_filename(selected_filename)
        )
        self.monitor.restore_summary(extra["numerical_summary"])

    def restore_best_for_evaluation(self) -> EvaluationCheckpointState:
        """Restore a premetric validation-selected model for validation latency."""

        if self.safety.protocol_gate.test_unlocked:
            raise SplitAccessError(
                "final protocols must use restore_parent_best_for_final with "
                "exact saved parent metadata"
            )
        if self.best_metric is None:
            raise TrainingContractError(
                "no validation-selected best metric is available for restore"
            )
        expected_best_sha256 = self._best_checkpoint_sha256()
        if expected_best_sha256 is None:
            raise TrainingContractError(
                "validation-selected best checkpoint digest is unavailable"
            )
        if self.best_checkpoint_path is None:
            raise TrainingContractError("selected best generation path is unavailable")

        state = load_model_checkpoint_for_evaluation(
            self.best_checkpoint_path,
            model=self.model,
            expected_metadata=self.checkpoint_metadata(),
            expected_mode="train",
            map_location="cpu",
            extra_state_validator=lambda extra, progress: (
                self._validate_checkpoint_extra_state(
                    extra,
                    progress,
                    expected_role="best",
                    source_size=self.safety.train_source_identity.row_count,
                    expected_best_metric=self.best_metric,
                )
            ),
        )
        if state.sha256 != expected_best_sha256:
            raise TrainingContractError(
                "restored best digest differs from selected best lineage"
            )
        self.model.eval()
        self.event_log.emit(
            "best_checkpoint_restored",
            sha256=state.sha256,
            checkpoint_role="best",
            checkpoint_protocol_sha256=self.checkpoint_protocol_sha256,
            resolved_model_config_sha256=(self.config.resolved_model_config_sha256),
            selected_by=self.config.selection_metric,
        )
        return state

    def restore_parent_best_for_final(
        self,
        checkpoint_path: str | Path,
        *,
        parent_checkpoint_metadata: CheckpointMetadata,
        expected_checkpoint_sha256: str,
        expected_best_metric: float,
    ) -> EvaluationCheckpointState:
        """Restore an exact parent best under a selected-config final protocol."""

        gate = self.safety.protocol_gate
        if self.config.phase != "final" or not gate.test_unlocked:
            raise SplitAccessError(
                "parent-best final restore requires an active final protocol"
            )
        if self.safety.selected_configs_path is None:
            raise SplitAccessError("final restore lacks selected_configs.lock")
        if not isinstance(parent_checkpoint_metadata, CheckpointMetadata):
            raise TypeError("parent_checkpoint_metadata must be CheckpointMetadata")
        if (
            not isinstance(expected_checkpoint_sha256, str)
            or SHA256_RE.fullmatch(expected_checkpoint_sha256) is None
        ):
            raise ValueError("expected_checkpoint_sha256 must be a lowercase SHA-256")
        if (
            isinstance(expected_best_metric, bool)
            or not isinstance(expected_best_metric, int | float)
            or not math.isfinite(float(expected_best_metric))
        ):
            raise ValueError("expected_best_metric must be finite")
        parent_digest = gate.parent_premetric_digest
        if (
            parent_digest is None
            or parent_checkpoint_metadata.protocol_hash != parent_digest
        ):
            raise TrainingContractError(
                "parent checkpoint protocol digest differs from final lineage"
            )
        try:
            selected = SelectedConfigsLock.load(self.safety.selected_configs_path)
            selected.require_for_protocol(gate)
            selected.require_model(
                model_key=self.config.model_key,
                candidate_id=self.config.candidate_id,
                resolved_config_sha256=(self.config.resolved_model_config_sha256),
            )
        except Exception as exc:
            raise TrainingContractError(
                f"selected final configuration binding failed: {exc}"
            ) from exc

        current_metadata = self.checkpoint_metadata()
        parent_fixed = parent_checkpoint_metadata.to_dict()
        current_fixed = current_metadata.to_dict()
        for name in (
            "data_hash",
            "feature_hash",
            "protocol_hash",
            "feature_order",
            "denominator_mask",
            "div_eps",
            "depth",
            "precision",
        ):
            if parent_fixed[name] != current_fixed[name]:
                raise TrainingContractError(
                    f"parent checkpoint metadata mismatch: {name}"
                )

        parent_config = deepcopy(parent_checkpoint_metadata.resolved_config)
        current_config = deepcopy(current_metadata.resolved_config)
        if set(parent_config) != set(current_config):
            raise TrainingContractError(
                "parent/current resolved checkpoint config keys differ"
            )
        parent_training = parent_config.get("training")
        current_training = current_config.get("training")
        if not isinstance(parent_training, dict) or not isinstance(
            current_training, dict
        ):
            raise TrainingContractError(
                "parent/current training checkpoint configs are malformed"
            )
        parent_phase = parent_training.get("phase")
        if parent_phase == "tuning":
            if (
                current_training.get("phase") != "final"
                or parent_training.get("seed") != 42
                or current_training.get("seed") != 42
            ):
                raise TrainingContractError(
                    "only seed-42 tuning best may cross to final phase"
                )
            parent_training["phase"] = "final"
            parent_training["reproduction_command"] = current_training.get(
                "reproduction_command"
            )
        elif parent_phase != "final":
            raise TrainingContractError(
                "parent checkpoint phase must be tuning or final"
            )
        if parent_config != current_config:
            raise TrainingContractError(
                "parent checkpoint differs beyond the explicit phase/command bridge"
            )

        def validate_best_extra(
            extra: Mapping[str, Any], progress: CheckpointProgress
        ) -> None:
            self._validate_checkpoint_extra_state(
                extra,
                progress,
                expected_role="best",
                source_size=self.safety.train_source_identity.row_count,
                expected_best_metric=float(expected_best_metric),
            )

        inspected = load_model_checkpoint_for_evaluation(
            checkpoint_path,
            model=self.model,
            expected_metadata=parent_checkpoint_metadata,
            expected_sha256=expected_checkpoint_sha256,
            expected_mode="train",
            map_location="cpu",
            extra_state_validator=validate_best_extra,
            restore_model=False,
        )
        if inspected.sha256 != expected_checkpoint_sha256:
            raise TrainingContractError(
                "parent best digest differs from the selected result receipt"
            )
        state = load_model_checkpoint_for_evaluation(
            checkpoint_path,
            model=self.model,
            expected_metadata=parent_checkpoint_metadata,
            expected_sha256=expected_checkpoint_sha256,
            expected_mode="train",
            map_location="cpu",
            extra_state_validator=validate_best_extra,
        )
        if state.sha256 != expected_checkpoint_sha256:
            raise TrainingContractError(
                "parent best digest differs from the selected result receipt"
            )
        self.model.eval()
        self.event_log.emit(
            "parent_best_checkpoint_restored_for_final",
            sha256=state.sha256,
            checkpoint_role="best",
            parent_premetric_protocol_sha256=parent_digest,
            selected_configs_sha256=selected.digest,
            resolved_model_config_sha256=(self.config.resolved_model_config_sha256),
        )
        return state

    def evaluate_test_bundle(
        self,
        source_factory: Callable[[], FinalRawBatchSource],
        *,
        checkpoint_path: str | Path,
        parent_checkpoint_metadata: CheckpointMetadata,
        expected_checkpoint_sha256: str,
        expected_best_metric: float,
        resolved_run_config_sha256: str,
        result_path: str | Path,
        batch_size: int,
        duration_constant_seconds: float = 157.0,
    ) -> FinalBundleRecord:
        """Run one exact final test cell and publish its immutable metric bundle."""

        if not self.raw_execution or not isinstance(self.model, TrainableEncodedModel):
            raise SplitAccessError(
                "legacy scalar final bundle does not support multi-target execution"
            )
        if self.config.phase != "final" or not self.safety.protocol_gate.test_unlocked:
            raise SplitAccessError(
                "final test bundle requires an active final protocol"
            )
        if self.safety.selected_configs_path is None:
            raise SplitAccessError("final test bundle lacks selected_configs.lock")
        if self.repository_root is None:
            raise TrainingContractError("final bundle lacks a verified repository root")
        if self.config.loss != "normalized_mean_squared_error":
            raise TrainingContractError(
                "watch-time final bundle requires normalized mean squared error"
            )

        checkpoint = Path(checkpoint_path)

        def inspect_checkpoint() -> str:
            actual = verify_checkpoint_sidecar(checkpoint)
            if actual != expected_checkpoint_sha256:
                raise TrainingContractError(
                    "final checkpoint sidecar differs from selected lineage"
                )
            return actual

        def restore_checkpoint() -> str:
            return self.restore_parent_best_for_final(
                checkpoint,
                parent_checkpoint_metadata=parent_checkpoint_metadata,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                expected_best_metric=expected_best_metric,
            ).sha256

        def verified_source_factory() -> FinalRawBatchSource:
            source = source_factory()
            try:
                if not isinstance(source, FinalRawBatchSource):
                    raise SplitAccessError(
                        "final source lacks the atomic batch_with_audit interface"
                    )
                self._assert_raw_source(source, "test")
                return source
            except BaseException:
                close = getattr(source, "close", None)
                if callable(close):
                    close()
                raise

        def predict_batch(raw_batch: RawFeatureBatch) -> torch.Tensor:
            before_hooks = self.monitor.activation_hook_observations
            with torch.no_grad(), self._autocast():
                output = self.model(raw_batch)
            self.monitor.assert_activation_hooks_observed(before_hooks)
            self.monitor.submit("test.output", output.logits, kind="output")
            self.monitor.assert_healthy()
            return output.logits

        self.model.eval()
        try:
            return run_one_pass_test_bundle(
                gate=self.safety.protocol_gate,
                selected_configs_path=self.safety.selected_configs_path,
                runtime_identity=self.safety.expected_runtime_identity("test"),
                model_key=self.config.model_key,
                seed=self.config.seed,
                candidate_id=self.config.candidate_id,
                resolved_config_sha256=self.config.resolved_model_config_sha256,
                resolved_run_config_sha256=resolved_run_config_sha256,
                checkpoint_protocol_sha256=parent_checkpoint_metadata.protocol_hash,
                checkpoint_sha256=expected_checkpoint_sha256,
                inspect_checkpoint=inspect_checkpoint,
                restore_checkpoint=restore_checkpoint,
                source_factory=verified_source_factory,
                predict_batch=predict_batch,
                finalize_predictions=self.monitor.drain,
                repository_root=self.repository_root,
                result_path=result_path,
                device=self.device,
                batch_size=batch_size,
                duration_constant_seconds=duration_constant_seconds,
            )
        finally:
            self.model.train()

    def _autocast(self):  # type: ignore[no-untyped-def]
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.autocast_bfloat16,
        )

    def _loss_sum(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_fp32 = logits.float()
        targets_fp32 = targets.float()
        if logits_fp32.shape != targets_fp32.shape:
            raise TrainingContractError("prediction and target shapes differ")
        if self.config.loss == "binary_cross_entropy_with_logits":
            return functional.binary_cross_entropy_with_logits(
                logits_fp32, targets_fp32, reduction="sum"
            )
        return functional.mse_loss(logits_fp32, targets_fp32, reduction="sum")

    def _next_indices(self, requested: int) -> list[int]:
        if self.sampler is None:
            raise TrainingContractError("sampler is not initialized")
        if self.sampler.exhausted:
            self.sampler.advance_epoch()
        count = min(requested, self.sampler.remaining)
        result: list[int] = []
        iterator = iter(self.sampler)
        for _ in range(count):
            result.append(next(iterator))
        return result

    @staticmethod
    def _is_oom(error: BaseException) -> bool:
        return isinstance(error, torch.cuda.OutOfMemoryError) or (
            isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
        )

    def _restore_after_first_oom(
        self, source: RawBatchSource | EncodedBatchSource
    ) -> None:
        if (
            not self.config.allow_one_oom_microbatch_reduction
            or self.oom_count >= 1
            or self.current_microbatch <= 1
        ):
            raise SecondOOMError("second OOM or unavailable microbatch fallback")
        previous = self.current_microbatch
        reduced = previous // 2
        while reduced > 0 and self.config.effective_batch_size % reduced:
            reduced -= 1
        if reduced <= 0:
            raise SecondOOMError(
                "no valid smaller microbatch preserves effective batch"
            )
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.resume(source)
        self.current_microbatch = reduced
        self.oom_count = 1
        self.last_checkpoint_record = self._save(
            self.last_checkpoint_path, checkpoint_role="last"
        )
        self.event_log.emit(
            "oom_recovery",
            previous_microbatch=previous,
            current_microbatch=reduced,
            effective_batch_size=self.config.effective_batch_size,
        )

    def evaluate(
        self,
        split: Literal["validation", "test"],
        source_factory: Callable[[], RawBatchSource | EncodedBatchSource],
        *,
        batch_size: int | None = None,
    ) -> float:
        """Evaluate validation only; final test requires a complete bundle runner."""

        if split == "test":
            raise SplitAccessError(
                "loss-only test evaluation is forbidden; no claim was reserved or "
                "opened because a durable one-pass metric/uncertainty bundle runner "
                "is not yet implemented"
            )
        size = batch_size or self.current_microbatch
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("evaluation batch_size must be positive")
        if self.raw_execution:
            source = self._open_raw_source("validation", source_factory)  # type: ignore[arg-type]
        else:
            self.safety.require_split("validation", self.config)
            source = source_factory()
            if source.split != "validation":  # type: ignore[union-attr]
                raise SplitAccessError("encoded source split declaration mismatch")
        self.model.eval()
        total_loss = 0.0
        total_rows = 0
        try:
            with torch.no_grad():
                for start in range(0, len(source), size):
                    indices = list(range(start, min(start + size, len(source))))
                    output, targets = self._batch_forward(
                        source, indices, split="validation"
                    )
                    loss_sum = self._loss_sum(output.logits, targets)
                    self.monitor.submit("validation.loss", loss_sum, kind="loss")
                    self.monitor.submit(
                        "validation.representation",
                        output.representation.values,
                        kind="activation",
                    )
                    total_loss += float(loss_sum.detach().cpu().item())
                    total_rows += len(indices)
                    self.monitor.assert_healthy()
            self.monitor.drain()
        finally:
            self.model.train()
            if self.raw_execution:
                self._close_raw_source(source)  # type: ignore[arg-type]
        if total_rows != len(source):
            raise TrainingContractError(
                "evaluation did not consume every row exactly once"
            )
        value = total_loss / total_rows
        if not math.isfinite(value):
            raise NumericalMonitorError("evaluation metric is non-finite")
        self.event_log.emit(
            "split_evaluated",
            split=split,
            rows=total_rows,
            selection_metric=self.config.selection_metric,
            value=value,
        )
        return value

    def fit(
        self,
        train_source_factory: Callable[[], RawBatchSource] | EncodedBatchSource,
        validation_source_factory: Callable[[], RawBatchSource | EncodedBatchSource],
        *,
        resume: bool = False,
        stop_after_examples: int | None = None,
    ) -> TrainingResult:
        """Open the train source only after its gate and always close real sources."""

        checkpoint_paths = [
            self.last_checkpoint_path,
            self.last_checkpoint_path.with_name(
                f"{self.last_checkpoint_path.name}.sha256"
            ),
            *self.run_directory.glob("best-step-*.pt"),
            *self.run_directory.glob("best-step-*.pt.sha256"),
        ]
        if resume:
            last_pair = checkpoint_paths[:2]
            if not all(path.is_file() for path in last_pair):
                raise TrainingContractError(
                    "resume requires an intact last checkpoint pair"
                )
        elif any(path.exists() for path in checkpoint_paths):
            raise TrainingContractError(
                "fresh run refuses an existing last/best checkpoint or sidecar"
            )

        if self.raw_execution:
            train_source = self._open_raw_source(
                "train",
                train_source_factory,  # type: ignore[arg-type]
            )
        else:
            self.safety.require_split("train", self.config)
            train_source = (
                train_source_factory()
                if callable(train_source_factory)
                else train_source_factory
            )
            if train_source.split != "train":
                raise SplitAccessError("training source must declare split=train")
        try:
            return self._fit_open_source(
                train_source,
                validation_source_factory,
                resume=resume,
                stop_after_examples=stop_after_examples,
            )
        finally:
            if self.raw_execution:
                self._close_raw_source(train_source)  # type: ignore[arg-type]

    def _fit_open_source(
        self,
        train_source: RawBatchSource | EncodedBatchSource,
        validation_source_factory: Callable[[], RawBatchSource | EncodedBatchSource],
        *,
        resume: bool,
        stop_after_examples: int | None,
    ) -> TrainingResult:
        if stop_after_examples is None:
            stop_target = self.config.realized_maximum_examples
        else:
            if (
                isinstance(stop_after_examples, bool)
                or not isinstance(stop_after_examples, int)
                or stop_after_examples <= 0
                or stop_after_examples > self.config.realized_maximum_examples
                or stop_after_examples % self.config.effective_batch_size
            ):
                raise ValueError(
                    "stop_after_examples must be a positive effective-batch boundary"
                )
            stop_target = stop_after_examples

        if resume:
            self.resume(train_source)
        else:
            self.sampler = StatefulPermutationSampler(
                len(train_source), seed=self.config.seed, shuffle=True
            )
            self.last_checkpoint_record = self._save(
                self.last_checkpoint_path, checkpoint_role="last"
            )

        if self.examples_seen > stop_target:
            raise TrainingContractError(
                "checkpoint already exceeds requested stop target"
            )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start_examples = self.examples_seen
        start_time = time.perf_counter()
        next_validation = (
            (self.examples_seen // self.config.validate_every_examples) + 1
        ) * self.config.validate_every_examples
        next_checkpoint = (
            (self.examples_seen // self.config.checkpoint_cadence) + 1
        ) * self.config.checkpoint_cadence
        stopped_reason = (
            "realized_floor_below_declared_hard_maximum"
            if stop_target == self.config.realized_maximum_examples
            else "requested_checkpoint_boundary"
        )
        self.model.train()

        try:
            while self.examples_seen < stop_target:
                self.monitor.assert_healthy()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                try:
                    while accumulated < self.config.effective_batch_size:
                        request = min(
                            self.current_microbatch,
                            self.config.effective_batch_size - accumulated,
                        )
                        indices = self._next_indices(request)
                        output, targets = self._batch_forward(
                            train_source, indices, split="train"
                        )
                        loss_sum = self._loss_sum(output.logits, targets)
                        scaled_loss = loss_sum / self.config.effective_batch_size
                        self.monitor.submit("train.loss", loss_sum, kind="loss")
                        self.monitor.submit(
                            "train.representation",
                            output.representation.values,
                            kind="activation",
                        )
                        scaled_loss.backward()
                        self.monitor.observe_gradients(self.model)
                        self.monitor.assert_healthy()
                        assert self.sampler is not None
                        self.sampler.commit(indices)
                        accumulated += len(indices)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.gradient_clip_l2,
                        error_if_nonfinite=True,
                    )
                    self.optimizer.step()
                except BaseException as error:
                    if not self._is_oom(error):
                        raise
                    self._restore_after_first_oom(train_source)
                    continue

                self.global_step += 1
                self.examples_seen += accumulated
                if accumulated != self.config.effective_batch_size:
                    raise TrainingContractError(
                        "optimizer step violated effective batch"
                    )
                if self.heartbeat is not None:
                    self.heartbeat()

                if self.examples_seen >= next_validation:
                    declared_validation_threshold = next_validation
                    metric = self.evaluate("validation", validation_source_factory)
                    self.validation_checks += 1
                    improved = self.best_metric is None or metric < self.best_metric
                    if improved:
                        self.best_metric = metric
                        self.bad_validation_checks = 0
                        best_generation = self._next_best_checkpoint_path()
                        self.best_checkpoint_record = self._save(
                            best_generation,
                            checkpoint_role="best",
                        )
                        self.best_checkpoint_path = best_generation
                    else:
                        self.bad_validation_checks += 1
                    self.event_log.emit(
                        "validation",
                        declared_threshold_examples=declared_validation_threshold,
                        realized_examples=self.examples_seen,
                        examples_seen=self.examples_seen,
                        check=self.validation_checks,
                        metric=metric,
                        improved=improved,
                        bad_checks=self.bad_validation_checks,
                    )
                    while next_validation <= self.examples_seen:
                        next_validation += self.config.validate_every_examples
                    if (
                        self.bad_validation_checks
                        >= self.config.patience_validation_checks
                    ):
                        stopped_reason = "validation_patience"
                        break

                if self.examples_seen >= next_checkpoint:
                    self.last_checkpoint_record = self._save(
                        self.last_checkpoint_path, checkpoint_role="last"
                    )
                    while next_checkpoint <= self.examples_seen:
                        next_checkpoint += self.config.checkpoint_cadence
        finally:
            self.optimizer.zero_grad(set_to_none=True)

        self.monitor.drain()
        self.last_checkpoint_record = self._save(
            self.last_checkpoint_path, checkpoint_role="last"
        )
        elapsed = time.perf_counter() - start_time
        peak_vram = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        if self.last_checkpoint_record is None:
            raise TrainingContractError("last checkpoint was not written")
        result = TrainingResult(
            examples_seen=self.examples_seen,
            optimizer_steps=self.global_step,
            validation_checks=self.validation_checks,
            best_validation_metric=self.best_metric,
            stopped_reason=stopped_reason,
            elapsed_seconds=elapsed,
            throughput_examples_per_second=(
                (self.examples_seen - start_examples) / elapsed
                if elapsed > 0
                else float("inf")
            ),
            peak_vram_bytes=peak_vram,
            oom_count=self.oom_count,
            last_checkpoint_sha256=self.last_checkpoint_record.sha256,
            best_checkpoint_sha256=self._best_checkpoint_sha256(),
            best_checkpoint_filename=(
                self.best_checkpoint_path.name
                if self.best_checkpoint_path is not None
                else None
            ),
            numerical_summary=asdict(self.monitor.summary()),
            reproduction_command=self.config.reproduction_command,
            declared_maximum_train_examples=(self.config.maximum_train_examples),
            realized_maximum_train_examples=(self.config.realized_maximum_examples),
        )
        self.event_log.emit("training_stopped", **asdict(result))
        return result

    def measure_model_latency(
        self,
        source_factory: Callable[[], RawBatchSource | EncodedBatchSource],
        *,
        warmup_repetitions: int,
        measured_repetitions: int,
        batch_size: int,
    ) -> LatencySummary:
        """Measure the complete validation raw-to-logit model; test is forbidden."""

        for name, value in (
            ("warmup_repetitions", warmup_repetitions),
            ("measured_repetitions", measured_repetitions),
            ("batch_size", batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.raw_execution:
            source = self._open_raw_source(
                "validation",
                source_factory,  # type: ignore[arg-type]
            )
        else:
            self.safety.require_split("validation", self.config)
            source = source_factory()
            if source.split != "validation":  # type: ignore[union-attr]
                raise SplitAccessError("latency source must be validation")
        durations: list[float] = []
        hooks_detached = False
        try:
            if batch_size > len(source):
                raise ValueError("latency batch exceeds validation rows")
            indices = tuple(range(batch_size))
            if self.raw_execution:
                self._assert_raw_source(source, "validation")  # type: ignore[arg-type]
                model_input = source.batch(indices, device=self.device)  # type: ignore[union-attr]
                model_input.validate(
                    source.identity,  # type: ignore[union-attr]
                    indices,
                    runtime_value_checks=source.runtime_value_checks,  # type: ignore[union-attr]
                )
            else:
                model_input, _ = source.batch(indices, device=self.device)  # type: ignore[assignment]
            self.model.eval()

            def synchronize() -> None:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

            with torch.no_grad():
                for _ in range(warmup_repetitions):
                    previous = self.monitor.activation_hook_observations
                    with self._autocast():
                        self.model(model_input)
                    self.monitor.assert_activation_hooks_observed(previous)
                self.monitor.drain()
                self.monitor.close_hooks()
                hooks_detached = True
                synchronize()
                for _ in range(measured_repetitions):
                    synchronize()
                    started = time.perf_counter_ns()
                    with self._autocast():
                        self.model(model_input)
                    synchronize()
                    durations.append((time.perf_counter_ns() - started) / 1e6)
        finally:
            try:
                if hooks_detached:
                    self._attach_monitor_hooks()
            finally:
                if self.raw_execution:
                    self._close_raw_source(source)  # type: ignore[arg-type]
        values = np.asarray(durations, dtype=np.float64)
        summary = LatencySummary(
            warmup_repetitions=warmup_repetitions,
            measured_repetitions=measured_repetitions,
            batch_size=batch_size,
            p50_ms=float(np.quantile(values, 0.50)),
            p95_ms=float(np.quantile(values, 0.95)),
            p99_ms=float(np.quantile(values, 0.99)),
        )
        self.event_log.emit("model_latency", **asdict(summary))
        return summary

    def close(self) -> None:
        try:
            self.monitor.drain()
        finally:
            self.monitor.close_hooks()

    def __enter__(self) -> TrainingHarness:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
