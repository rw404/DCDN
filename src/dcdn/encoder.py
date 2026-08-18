"""Frozen trainable shared feature encoder for public DCDN experiments.

The materialized-data boundary contains categorical token IDs and numeric
operands, never embeddings.  This module owns the trainable embedding tables,
emits every view required by the controlled model suite, and can be wrapped
with the interaction/head as one checkpointed and optimized ``nn.Module``.

No label, row ID, timestamp, policy flag, or other audit column appears in the
encoder forward signature.  The raw numeric operands and denominator-validity
mask are passed through without copying or transformation.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
import yaml
from torch import nn

from dcdn.models.common import HeadConfig
from dcdn.models.contracts import (
    EncodedInteractionBatch,
    EncoderSpec,
    PredictionOutput,
    canonical_config,
    canonical_sha256,
    require_nonnegative_int,
)
from dcdn.models.suite import (
    InteractionConfig,
    ProvenancedControlledModel,
    build_controlled_model,
)
from dcdn.protocol import SHA256_RE
from dcdn.raw import RawFeatureBatch, RawSourceIdentity

_INPUT_DTYPES = (torch.float32, torch.bfloat16)
_EXPECTED_ARCHITECTURE: dict[str, Any] = {
    "context_embedding_dim": 16,
    "numeric_field_embedding_dim": 16,
    "representation_dim": 64,
    "categorical_tables": "one_separate_nn_Embedding_per_ordered_C_field",
    "numeric_table": "one_nn_Embedding_table_indexed_by_ordered_numeric_field",
    "categorical_initializer": "normal_mean_0_std_0.01",
    "numeric_initializer": "normal_mean_0_std_0.01",
    "categorical_reserved_tokens": {"OOV": 0, "MISSING": 1},
    "bypass_features": "flattened_ordered_C_embeddings",
    "operand_features": "unchanged_ordered_raw_or_scale_normalized_numeric_values",
    "numeric_field_view": "learned_field_vector_times_scalar_without_bias",
    "field_order": "ordered_C_then_ordered_numeric_fields",
    "denominator_validity": "unchanged_pass_through",
    "parameter_dtype": "float32",
    "accepted_input_dtypes": ["float32", "bfloat16"],
    "fp16_inputs": "forbidden",
    "autocast_dtype": "bfloat16",
    "audit_id_and_label_fields": "structurally_absent_from_encoder_forward_signature",
}
_EXPECTED_INITIALIZATION: dict[str, Any] = {
    "policy": "independent_sha256_derived_cpu_component_seeds_v1",
    "primary_seed_domain": "dcdn-shared-encoder-primary-v1",
    "afn_secondary_seed_domain": "dcdn-shared-encoder-afn-secondary-v1",
    "per_table_seed_domain": "dcdn-shared-encoder-table-v1",
    "model_key_in_primary_seed": False,
    "construction_order_independent": True,
    "cuda_rng_touched": False,
    "afn_plus": {
        "independent_secondary_view": True,
        "identical_ordered_raw_fields": True,
        "shares_parameters_with_primary": False,
    },
    "all_other_models": {"secondary_view_present": False},
}
_EXPECTED_RUNTIME_IDENTITY: dict[str, Any] = {
    "source": "dcdn.raw.RawSourceIdentity",
    "require_dataset_id": True,
    "require_data_manifest_sha256": True,
    "require_views_manifest_sha256": True,
    "require_feature_contract_sha256": True,
    "require_physical_schema_sha256": True,
    "require_expanded_schema_sha256": True,
    "require_operand_schema_sha256": True,
    "require_denominator_mask_sha256": True,
    "require_full_materialized_scan_sha256": True,
    "require_cache_content_sha256": True,
    "content_digests_distinct_from_each_other_and_metadata": True,
    "spec_provenance": "independently_frozen_expected_values_never_copied_from_input",
    "precomputed_embeddings": "forbidden",
}
_EXPECTED_RUNTIME_VALUE_CHECKS: dict[str, Any] = {
    "debug_smoke": {
        "runtime_value_checks": True,
        "finite_and_token_predicates": True,
        "gpu_sync_allowed": True,
    },
    "optimized": {
        "runtime_value_checks": False,
        "finite_and_token_predicates": False,
        "gpu_sync_allowed": False,
        "requires_bound_full_materialized_scan_sha256": True,
        "requires_bound_cache_content_sha256": True,
    },
}
_VK_DATASET_ID = "vk_lsvd_up0.01_ir0.01"
_VK_SOURCE_MANIFEST_SHA256 = (
    "5dbbfeb23442a6b1ebdb829dc239c6a3c6c655d71446cde4551b83cc6af456ac"
)
_VK_VIEWS_MANIFEST_SHA256 = (
    "738d3610fd83a93297d94ccad4c4316dd3ad8b241f8409e84daf5b2322040dbe"
)
_VK_FEATURE_CONTRACT_SHA256 = (
    "380d105c847a73afb4927740e40401124688f895142792309b4fc8267a2a5f2b"
)
_VK_FULL_SCAN_SHA256 = (
    "cc7c74b183688c1125cd3e49d32daed2b1e7560d9ee796cfe93de972792ed612"
)
_VK_CONTEXT_FIELDS = ("age", "gender", "geo", "place", "platform", "agent")
_VK_CARDINALITIES = (55, 4, 82, 24, 13, 22)
_VK_REGIME_HASHES: dict[str, dict[str, Any]] = {
    "C+P": {
        "physical_schema_sha256": (
            "0b334e587d2028e1e39760b735a71e712af5e75ec4fd204cbc76a5efafe12f68"
        ),
        "expanded_schema_sha256": (
            "2cfd6d12cd85346e413f476ff223e19d57be213d74fe68b75d6f0ce686c0f091"
        ),
        "operand_schema_sha256": (
            "bb2f6c47fa12c175a3820830f90ff577acb293c56b107c7d1c40e4c682f2ff4d"
        ),
        "denominator_mask_sha256": (
            "9e168fae4014f4763008e89f61324858f17d376f56fb5de46f2775911650bcaa"
        ),
        "cache_content_sha256": (
            "b732b95f80f9ec28f4d9dd5192bf7dba3ed16bb96093bc5add209743edfff590"
        ),
        "allowed_models": [],
    },
    "C+P+E": {
        "physical_schema_sha256": (
            "91061695b06f2a1f4c8743f4ba4c0fa66c14cd7c0dd09d15e6c40d7262b6197b"
        ),
        "expanded_schema_sha256": (
            "381beb02c21c8a56a4111b2dde50eb7c870518e9289a806f6e4ddb8a2c9dda59"
        ),
        "operand_schema_sha256": (
            "3b34b22f0779dd059e89016b388952862da074c5bfa2534cc7ba251569ba54e3"
        ),
        "denominator_mask_sha256": (
            "8bca04940fe8b7e6cf0d2f8e100db0787cc32e066c6c9eadd0ba711652b37715"
        ),
        "cache_content_sha256": (
            "34f415bde13633adc6994e22548e9aa44817a586d6d74350c93e859835c9d5d7"
        ),
        "allowed_models": ["dcnv2_full_rank"],
    },
}
_EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "frozen_utc",
    "contract",
    "metric_derived_choices",
    "test_unlocked",
    "architecture",
    "initialization",
    "runtime_identity",
    "runtime_value_checks",
    "datasets",
    "unresolved_datasets",
    "resolution_rule",
}


class EncoderContractError(RuntimeError):
    """The encoder lock, source identity, or tensor contract was violated."""


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_names(name: str, values: Any) -> tuple[str, ...]:
    if not isinstance(values, list | tuple) or not values:
        raise ValueError(f"{name} must be a non-empty sequence")
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _require_cardinalities(values: Any, expected: int) -> tuple[int, ...]:
    if not isinstance(values, list | tuple) or len(values) != expected:
        raise ValueError("categorical cardinalities must align with C fields")
    result = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 1
        for value in result
    ):
        raise ValueError("categorical cardinalities must be integers above one")
    return result


def _require_mask(values: Any, expected: int) -> tuple[bool, ...]:
    if not isinstance(values, list | tuple) or len(values) != expected:
        raise ValueError("denominator mask must align with numeric operands")
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("denominator mask must contain booleans")
    return tuple(values)


_ENCODER_LOCK_MAX_BYTES = 1024 * 1024
_ENCODER_SIDECAR_MAX_BYTES = 256


def _encoder_regular_fd_at(
    directory_fd: int, name: str, label: str
) -> tuple[int, os.stat_result]:
    if not name or Path(name).name != name:
        raise EncoderContractError(f"{label} name is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise EncoderContractError(f"cannot securely open {label}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise EncoderContractError(
            f"{label} must be a single-link non-symlink regular file"
        )
    return descriptor, metadata


def _encoder_stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_encoder_fd(descriptor: int, *, maximum_bytes: int, label: str) -> bytes:
    before = os.fstat(descriptor)
    if before.st_size > maximum_bytes:
        raise EncoderContractError(f"{label} exceeds size limit")
    chunks: list[bytes] = []
    position = 0
    while position < before.st_size:
        chunk = os.pread(
            descriptor,
            min(1024 * 1024, before.st_size - position),
            position,
        )
        if not chunk:
            raise EncoderContractError(f"{label} changed during verified read")
        chunks.append(chunk)
        position += len(chunk)
    after = os.fstat(descriptor)
    if position != before.st_size or _encoder_stat_identity(
        before
    ) != _encoder_stat_identity(after):
        raise EncoderContractError(f"{label} changed during verified read")
    return b"".join(chunks)


def _read_encoder_lock_pair(lock_path: Path, sidecar_path: Path) -> tuple[bytes, bytes]:
    if sidecar_path.name != f"{lock_path.name}.sha256":
        raise EncoderContractError("encoder sidecar must use the adjacent exact name")
    if sidecar_path.parent.absolute() != lock_path.parent.absolute():
        raise EncoderContractError("encoder sidecar must be adjacent to the YAML")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        parent_fd = os.open(lock_path.parent, flags)
        parent_metadata = os.fstat(parent_fd)
    except OSError as error:
        raise EncoderContractError(
            "cannot securely open encoder lock parent"
        ) from error
    if not stat.S_ISDIR(parent_metadata.st_mode):
        os.close(parent_fd)
        raise EncoderContractError("encoder lock parent must be a directory")
    try:
        lock_fd, _ = _encoder_regular_fd_at(parent_fd, lock_path.name, "encoder YAML")
        sidecar_fd, _ = _encoder_regular_fd_at(
            parent_fd, sidecar_path.name, "encoder SHA sidecar"
        )
        try:
            lock_bytes = _read_encoder_fd(
                lock_fd,
                maximum_bytes=_ENCODER_LOCK_MAX_BYTES,
                label="encoder YAML",
            )
            sidecar_bytes = _read_encoder_fd(
                sidecar_fd,
                maximum_bytes=_ENCODER_SIDECAR_MAX_BYTES,
                label="encoder SHA sidecar",
            )
        finally:
            os.close(sidecar_fd)
            os.close(lock_fd)
    finally:
        os.close(parent_fd)
    return lock_bytes, sidecar_bytes


def _require_exact_lock_section(
    name: str,
    value: Any,
    expected: Mapping[str, Any],
) -> None:
    if not isinstance(value, dict) or value != expected:
        raise EncoderContractError(f"encoder {name} section drifted")


@dataclass(frozen=True)
class EncoderContractLock:
    """Byte-verified YAML lock for all metric-independent encoder choices."""

    path: Path
    sha256: str
    payload: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path = "configs/encoder_v1.yaml",
        sidecar_path: str | Path | None = None,
    ) -> EncoderContractLock:
        lock_path = Path(path)
        sidecar = (
            Path(sidecar_path)
            if sidecar_path is not None
            else lock_path.with_name(f"{lock_path.name}.sha256")
        )
        lock_bytes, sidecar_bytes = _read_encoder_lock_pair(lock_path, sidecar)
        try:
            sidecar_text = sidecar_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise EncoderContractError("encoder SHA sidecar must be ASCII") from error
        if not sidecar_text.endswith("\n") or sidecar_text.count("\n") != 1:
            raise EncoderContractError("encoder SHA sidecar has an invalid format")
        parts = sidecar_text[:-1].split("  ")
        if len(parts) != 2 or parts[1] != lock_path.name:
            raise EncoderContractError("encoder SHA sidecar has an invalid format")
        expected = _require_sha256("encoder sidecar digest", parts[0])
        actual = hashlib.sha256(lock_bytes).hexdigest()
        if actual != expected:
            raise EncoderContractError("encoder YAML does not match its SHA sidecar")
        try:
            loaded = yaml.safe_load(lock_bytes.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise EncoderContractError("encoder YAML is invalid") from error
        if not isinstance(loaded, dict):
            raise EncoderContractError("encoder YAML root must be a mapping")
        cls._validate_payload(loaded)
        return cls(path=lock_path, sha256=actual, payload=loaded)

    @staticmethod
    def _validate_payload(payload: Mapping[str, Any]) -> None:
        if set(payload) != _EXPECTED_TOP_LEVEL_KEYS:
            raise EncoderContractError("encoder YAML top-level fields drifted")
        exact_top_level = {
            "schema_version": 1,
            "status": "frozen_before_metrics",
            "frozen_utc": "2026-08-18T15:16:00Z",
            "contract": "dcdn-trainable-shared-feature-encoder-v1",
            "metric_derived_choices": False,
            "test_unlocked": False,
        }
        for key, expected in exact_top_level.items():
            if payload.get(key) != expected:
                raise EncoderContractError(f"encoder {key} drifted")

        _require_exact_lock_section(
            "architecture", payload.get("architecture"), _EXPECTED_ARCHITECTURE
        )
        _require_exact_lock_section(
            "initialization", payload.get("initialization"), _EXPECTED_INITIALIZATION
        )
        _require_exact_lock_section(
            "runtime_identity",
            payload.get("runtime_identity"),
            _EXPECTED_RUNTIME_IDENTITY,
        )
        _require_exact_lock_section(
            "runtime_value_checks",
            payload.get("runtime_value_checks"),
            _EXPECTED_RUNTIME_VALUE_CHECKS,
        )

        datasets = payload.get("datasets")
        if not isinstance(datasets, dict) or set(datasets) != {_VK_DATASET_ID}:
            raise EncoderContractError("encoder datasets resolution drifted")
        dataset = datasets[_VK_DATASET_ID]
        expected_dataset_keys = {
            "resolution_status",
            "source_manifest_sha256",
            "views_manifest_sha256",
            "feature_contract_sha256",
            "full_materialized_scan_sha256",
            "ordered_context_fields",
            "categorical_cardinalities",
            "regimes",
        }
        if not isinstance(dataset, dict) or set(dataset) != expected_dataset_keys:
            raise EncoderContractError("VK encoder resolution fields drifted")
        expected_dataset_values = {
            "resolution_status": (
                "verified_train_validation_views_before_model_metrics"
            ),
            "source_manifest_sha256": _VK_SOURCE_MANIFEST_SHA256,
            "views_manifest_sha256": _VK_VIEWS_MANIFEST_SHA256,
            "feature_contract_sha256": _VK_FEATURE_CONTRACT_SHA256,
            "full_materialized_scan_sha256": _VK_FULL_SCAN_SHA256,
            "ordered_context_fields": list(_VK_CONTEXT_FIELDS),
            "categorical_cardinalities": list(_VK_CARDINALITIES),
        }
        for key, expected in expected_dataset_values.items():
            if dataset.get(key) != expected:
                raise EncoderContractError(f"VK encoder resolution {key} drifted")
        for key in (
            "source_manifest_sha256",
            "views_manifest_sha256",
            "feature_contract_sha256",
            "full_materialized_scan_sha256",
        ):
            _require_sha256(key, dataset[key])

        regimes = dataset.get("regimes")
        if not isinstance(regimes, dict) or set(regimes) != set(_VK_REGIME_HASHES):
            raise EncoderContractError("VK feature regimes drifted")
        resolved_fields: dict[str, tuple[str, ...]] = {}
        resolved_masks: dict[str, tuple[bool, ...]] = {}
        for regime_name, frozen in _VK_REGIME_HASHES.items():
            regime = regimes.get(regime_name)
            expected_regime_keys = {
                "allowed_models",
                "physical_schema_sha256",
                "expanded_schema_sha256",
                "operand_schema_sha256",
                "denominator_mask_sha256",
                "cache_content_sha256",
                "ordered_operand_fields",
                "denominator_mask",
            }
            if regime_name == "C+P+E":
                expected_regime_keys.add("operand_schema_hash_derivation")
            if not isinstance(regime, dict) or set(regime) != expected_regime_keys:
                raise EncoderContractError(
                    f"VK {regime_name} encoder resolution fields drifted"
                )
            for key, expected in frozen.items():
                if regime.get(key) != expected:
                    raise EncoderContractError(
                        f"VK {regime_name} encoder resolution {key} drifted"
                    )
            for key in (
                "physical_schema_sha256",
                "expanded_schema_sha256",
                "operand_schema_sha256",
                "denominator_mask_sha256",
                "cache_content_sha256",
            ):
                _require_sha256(key, regime[key])
            if (
                regime_name == "C+P+E"
                and regime.get("operand_schema_hash_derivation")
                != "sha256_compact_ASCII_JSON_of_ordered_P_then_E_names"
            ):
                raise EncoderContractError("VK P+E operand hash derivation drifted")

            fields = _require_names(
                f"{regime_name}.ordered_operand_fields",
                regime.get("ordered_operand_fields"),
            )
            mask = _require_mask(regime.get("denominator_mask"), len(fields))
            if canonical_sha256(fields) != regime["operand_schema_sha256"]:
                raise EncoderContractError(f"VK {regime_name} ordered operands drifted")
            if canonical_sha256(mask) != regime["denominator_mask_sha256"]:
                raise EncoderContractError(f"VK {regime_name} denominator mask drifted")
            resolved_fields[regime_name] = fields
            resolved_masks[regime_name] = mask

        base_fields = resolved_fields["C+P"]
        full_fields = resolved_fields["C+P+E"]
        base_mask = resolved_masks["C+P"]
        full_mask = resolved_masks["C+P+E"]
        if len(base_fields) != 58 or len(full_fields) != 78:
            raise EncoderContractError("VK numeric feature widths drifted")
        if full_fields[: len(base_fields)] != base_fields:
            raise EncoderContractError(
                "VK full view must preserve exact P prefix order"
            )
        if full_mask[: len(base_mask)] != base_mask or any(full_mask[len(base_mask) :]):
            raise EncoderContractError(
                "VK full-view denominator mask must be P mask plus false E entries"
            )

        content_digests = {
            dataset["full_materialized_scan_sha256"],
            regimes["C+P"]["cache_content_sha256"],
            regimes["C+P+E"]["cache_content_sha256"],
        }
        metadata_digests = {
            dataset["source_manifest_sha256"],
            dataset["views_manifest_sha256"],
            dataset["feature_contract_sha256"],
        }
        for regime in regimes.values():
            metadata_digests.update(
                {
                    regime["physical_schema_sha256"],
                    regime["expanded_schema_sha256"],
                    regime["operand_schema_sha256"],
                    regime["denominator_mask_sha256"],
                }
            )
        if len(content_digests) != 3 or content_digests & metadata_digests:
            raise EncoderContractError(
                "VK scan/cache content digests must be distinct from metadata"
            )

        if payload.get("unresolved_datasets") != [
            "kuairand_pure",
            "kuairec_big_matrix",
        ]:
            raise EncoderContractError("unresolved dataset list drifted")
        if payload.get("resolution_rule") != (
            "add_only_after_verified_view_manifest_cardinalities_and_scan_attestation"
        ):
            raise EncoderContractError("encoder dataset resolution rule drifted")


@dataclass(frozen=True)
class DatasetEncoderSpec:
    """One dataset/regime resolution bound to verified materialized identity."""

    dataset_id: str
    feature_regime: str
    encoder_contract_sha256: str
    data_manifest_sha256: str
    views_manifest_sha256: str
    feature_contract_sha256: str
    physical_schema_sha256: str
    full_feature_schema_sha256: str
    operand_schema_sha256: str
    denominator_mask_sha256: str
    full_materialized_scan_sha256: str
    cache_content_sha256: str
    ordered_context_fields: tuple[str, ...]
    categorical_cardinalities: tuple[int, ...]
    ordered_operand_fields: tuple[str, ...]
    denominator_mask: tuple[bool, ...]
    allowed_model_keys: tuple[str, ...]
    embedding_dim: int = 16
    representation_dim: int = 64
    initialization_std: float = 0.01
    oov_token: int = 0
    missing_token: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string")
        if not isinstance(self.feature_regime, str) or not self.feature_regime.strip():
            raise ValueError("feature_regime must be a non-empty string")
        for name in (
            "encoder_contract_sha256",
            "data_manifest_sha256",
            "views_manifest_sha256",
            "feature_contract_sha256",
            "physical_schema_sha256",
            "full_feature_schema_sha256",
            "operand_schema_sha256",
            "denominator_mask_sha256",
            "full_materialized_scan_sha256",
            "cache_content_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        content_digests = {
            self.full_materialized_scan_sha256,
            self.cache_content_sha256,
        }
        metadata_digests = {
            self.data_manifest_sha256,
            self.views_manifest_sha256,
            self.feature_contract_sha256,
            self.physical_schema_sha256,
            self.full_feature_schema_sha256,
            self.operand_schema_sha256,
            self.denominator_mask_sha256,
        }
        if (
            len(content_digests) != 2
            or any(len(set(value)) == 1 for value in content_digests)
            or content_digests & metadata_digests
        ):
            raise EncoderContractError(
                "full scan and cache must be distinct non-placeholder content digests"
            )
        _require_names("ordered_context_fields", self.ordered_context_fields)
        _require_names("ordered_operand_fields", self.ordered_operand_fields)
        if set(self.ordered_context_fields) & set(self.ordered_operand_fields):
            raise ValueError("C and numeric operand names must be disjoint")
        _require_cardinalities(
            self.categorical_cardinalities, len(self.ordered_context_fields)
        )
        _require_mask(self.denominator_mask, len(self.ordered_operand_fields))
        if canonical_sha256(self.denominator_mask) != self.denominator_mask_sha256:
            raise EncoderContractError("denominator mask digest mismatch")
        if canonical_sha256(self.ordered_operand_fields) != self.operand_schema_sha256:
            raise EncoderContractError("ordered numeric operand digest mismatch")
        if self.embedding_dim != 16 or self.representation_dim != 64:
            raise EncoderContractError("encoder dimensions drifted from the lock")
        if self.initialization_std != 0.01:
            raise EncoderContractError("encoder initialization std drifted")
        if (self.oov_token, self.missing_token) != (0, 1):
            raise EncoderContractError("reserved categorical token IDs drifted")
        if any(
            not isinstance(value, str) or not value for value in self.allowed_model_keys
        ):
            raise TypeError("allowed_model_keys must contain non-empty strings")
        if len(set(self.allowed_model_keys)) != len(self.allowed_model_keys):
            raise ValueError("allowed_model_keys must not contain duplicates")

    @property
    def optimized_runtime_attested(self) -> bool:
        """Whether scan/cache identity permits predicate-free optimized execution."""

        content_digests = {
            self.full_materialized_scan_sha256,
            self.cache_content_sha256,
        }
        metadata_digests = {
            self.data_manifest_sha256,
            self.views_manifest_sha256,
            self.feature_contract_sha256,
            self.physical_schema_sha256,
            self.full_feature_schema_sha256,
            self.operand_schema_sha256,
            self.denominator_mask_sha256,
        }
        return (
            len(content_digests) == 2
            and all(SHA256_RE.fullmatch(value) is not None for value in content_digests)
            and all(len(set(value)) > 1 for value in content_digests)
            and not content_digests & metadata_digests
        )

    @property
    def model_spec(self) -> EncoderSpec:
        return EncoderSpec(
            bypass_dim=len(self.ordered_context_fields) * self.embedding_dim,
            operand_dim=len(self.ordered_operand_fields),
            num_fields=(
                len(self.ordered_context_fields) + len(self.ordered_operand_fields)
            ),
            embedding_dim=self.embedding_dim,
            representation_dim=self.representation_dim,
            denominator_mask=self.denominator_mask,
            full_feature_schema_sha256=self.full_feature_schema_sha256,
            operand_schema_sha256=self.operand_schema_sha256,
        )

    def contract_state(self) -> dict[str, Any]:
        return canonical_config(self)

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.contract_state())

    def assert_source_identity(self, identity: RawSourceIdentity) -> None:
        expected: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "feature_regime": self.feature_regime,
            "data_manifest_sha256": self.data_manifest_sha256,
            "views_manifest_sha256": self.views_manifest_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "physical_schema_sha256": self.physical_schema_sha256,
            "expanded_schema_sha256": self.full_feature_schema_sha256,
            "operand_schema_sha256": self.operand_schema_sha256,
            "denominator_mask_sha256": self.denominator_mask_sha256,
            "full_materialized_scan_sha256": self.full_materialized_scan_sha256,
            "cache_content_sha256": self.cache_content_sha256,
            "ordered_context_fields": self.ordered_context_fields,
            "ordered_operand_fields": self.ordered_operand_fields,
            "categorical_cardinalities": self.categorical_cardinalities,
        }
        for name, value in expected.items():
            if getattr(identity, name) != value:
                raise EncoderContractError(f"raw source identity mismatch: {name}")


def resolve_dataset_encoder_spec(
    lock: EncoderContractLock,
    *,
    dataset_id: str,
    source_identity: RawSourceIdentity,
) -> DatasetEncoderSpec:
    """Resolve one source identity against the byte-frozen encoder contract."""

    EncoderContractLock._validate_payload(lock.payload)
    _require_sha256("encoder_contract_sha256", lock.sha256)
    datasets = lock.payload["datasets"]
    if dataset_id not in datasets:
        raise EncoderContractError(
            f"dataset {dataset_id!r} has no verified encoder resolution"
        )
    dataset = datasets[dataset_id]
    if not isinstance(dataset, dict):
        raise EncoderContractError("dataset encoder resolution must be a mapping")
    if dataset.get("resolution_status") != (
        "verified_train_validation_views_before_model_metrics"
    ):
        raise EncoderContractError("dataset encoder resolution is not verified")
    regimes = dataset.get("regimes")
    if not isinstance(regimes, dict):
        raise EncoderContractError("dataset feature regimes must be a mapping")
    regime_name = next(
        (
            frozen_name
            for frozen_name in regimes
            if frozen_name == source_identity.feature_regime
        ),
        None,
    )
    if regime_name is None:
        raise EncoderContractError("feature regime has no verified encoder resolution")
    regime = regimes[regime_name]
    if not isinstance(regime, dict):
        raise EncoderContractError("feature regime resolution must be a mapping")

    context_fields = _require_names(
        "ordered_context_fields", dataset.get("ordered_context_fields")
    )
    cardinalities = _require_cardinalities(
        dataset.get("categorical_cardinalities"), len(context_fields)
    )
    operand_fields = _require_names(
        "ordered_operand_fields", regime.get("ordered_operand_fields")
    )
    mask = _require_mask(regime.get("denominator_mask"), len(operand_fields))
    allowed = regime.get("allowed_models", [])
    if not isinstance(allowed, list) or any(
        not isinstance(value, str) or not value for value in allowed
    ):
        raise EncoderContractError("allowed_models must be a string list")

    spec = DatasetEncoderSpec(
        dataset_id=dataset_id,
        feature_regime=regime_name,
        encoder_contract_sha256=lock.sha256,
        data_manifest_sha256=_require_sha256(
            "source_manifest_sha256", dataset.get("source_manifest_sha256")
        ),
        views_manifest_sha256=_require_sha256(
            "views_manifest_sha256", dataset.get("views_manifest_sha256")
        ),
        feature_contract_sha256=_require_sha256(
            "feature_contract_sha256", dataset.get("feature_contract_sha256")
        ),
        physical_schema_sha256=_require_sha256(
            "physical_schema_sha256", regime.get("physical_schema_sha256")
        ),
        full_feature_schema_sha256=_require_sha256(
            "expanded_schema_sha256", regime.get("expanded_schema_sha256")
        ),
        operand_schema_sha256=_require_sha256(
            "operand_schema_sha256", regime.get("operand_schema_sha256")
        ),
        denominator_mask_sha256=_require_sha256(
            "denominator_mask_sha256", regime.get("denominator_mask_sha256")
        ),
        full_materialized_scan_sha256=_require_sha256(
            "full_materialized_scan_sha256",
            dataset.get("full_materialized_scan_sha256"),
        ),
        cache_content_sha256=_require_sha256(
            "cache_content_sha256", regime.get("cache_content_sha256")
        ),
        ordered_context_fields=context_fields,
        categorical_cardinalities=cardinalities,
        ordered_operand_fields=operand_fields,
        denominator_mask=mask,
        allowed_model_keys=tuple(allowed),
    )
    spec.assert_source_identity(source_identity)
    return spec


def _derived_seed(*parts: str | int) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


class _EmbeddingView(nn.Module):
    """One independent categorical and numeric learned field view."""

    def __init__(
        self,
        spec: DatasetEncoderSpec,
        *,
        view_name: str,
        view_seed: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.view_name = view_name
        self.view_seed = require_nonnegative_int("view_seed", view_seed)
        # nn.Embedding constructors initialize eagerly.  Forking the CPU RNG
        # makes those discarded values construction-order independent and
        # leaves the caller's RNG state untouched.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.random.default_generator.manual_seed(0)
            self.categorical_embeddings = nn.ModuleList(
                nn.Embedding(cardinality, spec.embedding_dim)
                for cardinality in spec.categorical_cardinalities
            )
            self.numeric_vectors = nn.Embedding(
                len(spec.ordered_operand_fields), spec.embedding_dim
            )
        self.register_buffer(
            "_numeric_field_indices",
            torch.arange(len(spec.ordered_operand_fields), dtype=torch.int64),
            persistent=False,
        )
        self._component_seeds = self._reset_parameters()

    def _table_seed(self, table: str) -> int:
        return _derived_seed(
            "dcdn-shared-encoder-table-v1",
            self.view_seed,
            self.spec.full_feature_schema_sha256,
            table,
        )

    def _normal_(self, parameter: nn.Parameter, seed: int) -> None:
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise EncoderContractError(
                "encoder embeddings must be initialized on CPU in FP32"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        nn.init.normal_(
            parameter,
            mean=0.0,
            std=self.spec.initialization_std,
            generator=generator,
        )

    def _reset_parameters(self) -> dict[str, int]:
        seeds: dict[str, int] = {}
        for index, (name, embedding) in enumerate(
            zip(
                self.spec.ordered_context_fields,
                self.categorical_embeddings,
                strict=True,
            )
        ):
            key = f"categorical:{index}:{name}"
            seed = self._table_seed(key)
            self._normal_(embedding.weight, seed)
            seeds[key] = seed
        numeric_key = "numeric:" + self.spec.operand_schema_sha256
        numeric_seed = self._table_seed(numeric_key)
        self._normal_(self.numeric_vectors.weight, numeric_seed)
        seeds[numeric_key] = numeric_seed
        return seeds

    @property
    def component_seeds(self) -> dict[str, int]:
        return dict(self._component_seeds)

    def forward(
        self,
        context_tokens: torch.Tensor,
        operand_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_dtype = operand_features.dtype
        categorical = torch.stack(
            [
                table(context_tokens[:, index])
                for index, table in enumerate(self.categorical_embeddings)
            ],
            dim=1,
        ).to(dtype=output_dtype)
        numeric_vectors = self.numeric_vectors(self._numeric_field_indices).to(
            dtype=output_dtype
        )
        numeric = operand_features.unsqueeze(-1) * numeric_vectors.unsqueeze(0)
        return categorical, torch.cat((categorical, numeric), dim=1)


class SharedFeatureEncoder(nn.Module):
    """Trainable C/P encoder shared by all controlled model families."""

    CONTRACT: ClassVar[str] = "dcdn-shared-feature-encoder-module-v1"

    def __init__(
        self,
        spec: DatasetEncoderSpec,
        *,
        run_seed: int,
        include_afn_secondary: bool,
        runtime_value_checks: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.run_seed = require_nonnegative_int("run_seed", run_seed)
        if not isinstance(include_afn_secondary, bool):
            raise TypeError("include_afn_secondary must be boolean")
        if not isinstance(runtime_value_checks, bool):
            raise TypeError("runtime_value_checks must be boolean")
        if not runtime_value_checks and not spec.optimized_runtime_attested:
            raise EncoderContractError(
                "optimized encoder execution requires bound scan/cache identity"
            )
        self.include_afn_secondary = include_afn_secondary
        self.runtime_value_checks = runtime_value_checks
        self.primary_seed = _derived_seed(
            "dcdn-shared-encoder-primary-v1",
            self.run_seed,
            spec.contract_sha256,
        )
        self.afn_secondary_seed = _derived_seed(
            "dcdn-shared-encoder-afn-secondary-v1",
            self.run_seed,
            spec.contract_sha256,
        )
        self.primary = _EmbeddingView(
            spec, view_name="primary", view_seed=self.primary_seed
        )
        self.afn_secondary = (
            _EmbeddingView(
                spec,
                view_name="afn_secondary",
                view_seed=self.afn_secondary_seed,
            )
            if include_afn_secondary
            else None
        )

    @property
    def model_spec(self) -> EncoderSpec:
        return self.spec.model_spec

    def _validate_inputs(
        self,
        context_tokens: torch.Tensor,
        operand_features: torch.Tensor,
        denominator_validity: torch.Tensor,
    ) -> None:
        if not all(
            isinstance(value, torch.Tensor)
            for value in (context_tokens, operand_features, denominator_validity)
        ):
            raise TypeError("all encoder inputs must be tensors")
        if context_tokens.ndim != 2 or tuple(context_tokens.shape[1:]) != (
            len(self.spec.ordered_context_fields),
        ):
            raise ValueError("context_tokens violate the ordered C width")
        batch_size = int(context_tokens.shape[0])
        if batch_size <= 0:
            raise ValueError("empty encoder batches are forbidden")
        expected_operands = (batch_size, len(self.spec.ordered_operand_fields))
        if tuple(operand_features.shape) != expected_operands:
            raise ValueError("operand_features violate the ordered numeric width")
        if tuple(denominator_validity.shape) != expected_operands:
            raise ValueError("denominator_validity must match operand_features")
        if context_tokens.dtype != torch.int64:
            raise TypeError("context_tokens must be int64")
        if operand_features.dtype == torch.float16:
            raise TypeError("FP16 encoder inputs are forbidden")
        if operand_features.dtype not in _INPUT_DTYPES:
            raise TypeError("operand_features must use FP32 or BF16")
        if denominator_validity.dtype != torch.bool:
            raise TypeError("denominator_validity must be boolean")
        devices = {
            context_tokens.device,
            operand_features.device,
            denominator_validity.device,
        }
        if len(devices) != 1:
            raise ValueError("all encoder inputs must share one device")
        parameter_devices = {parameter.device for parameter in self.parameters()}
        parameter_dtypes = {parameter.dtype for parameter in self.parameters()}
        if parameter_devices != {operand_features.device}:
            raise ValueError("encoder inputs and parameters must share one device")
        if parameter_dtypes != {torch.float32}:
            raise TypeError("all encoder parameters must remain FP32")
        if self.runtime_value_checks:
            if not bool(torch.isfinite(operand_features).all()):
                raise ValueError("operand_features contain non-finite values")
            for index, cardinality in enumerate(self.spec.categorical_cardinalities):
                tokens = context_tokens[:, index]
                if bool(((tokens < 0) | (tokens >= cardinality)).any()):
                    raise ValueError(f"C token column {index} exceeds its cardinality")

    def forward(
        self,
        context_tokens: torch.Tensor,
        operand_features: torch.Tensor,
        denominator_validity: torch.Tensor,
    ) -> EncodedInteractionBatch:
        self._validate_inputs(context_tokens, operand_features, denominator_validity)
        categorical, fields = self.primary(context_tokens, operand_features)
        auxiliary = None
        if self.afn_secondary is not None:
            _, auxiliary = self.afn_secondary(context_tokens, operand_features)
        encoded = EncodedInteractionBatch(
            bypass_features=categorical.flatten(start_dim=1),
            operand_features=operand_features,
            field_embeddings=fields,
            denominator_validity=denominator_validity,
            afn_dnn_embeddings=auxiliary,
        )
        encoded.validate(self.model_spec, require_afn_aux=self.include_afn_secondary)
        return encoded

    def contract_state(self) -> dict[str, Any]:
        return canonical_config(
            {
                "contract": self.CONTRACT,
                "dataset": self.spec.contract_state(),
                "run_seed": self.run_seed,
                "runtime": {
                    "runtime_value_checks": self.runtime_value_checks,
                    "mode": (
                        "debug_smoke" if self.runtime_value_checks else "optimized"
                    ),
                    "optimized_scan_cache_identity_bound": (
                        self.spec.optimized_runtime_attested
                    ),
                    "optimized_finite_and_token_predicates": False,
                },
                "initialization": {
                    "policy": "independent_sha256_derived_cpu_component_seeds_v1",
                    "primary_seed": self.primary_seed,
                    "primary_component_seeds": self.primary.component_seeds,
                    "afn_secondary_included": self.include_afn_secondary,
                    "afn_secondary_seed": (
                        self.afn_secondary_seed if self.include_afn_secondary else None
                    ),
                    "afn_secondary_component_seeds": (
                        self.afn_secondary.component_seeds
                        if self.afn_secondary is not None
                        else None
                    ),
                    "model_key_in_primary_seed": False,
                    "construction_device": "cpu",
                    "parameter_dtype": "float32",
                    "normal_std": self.spec.initialization_std,
                },
                "views": {
                    "bypass": "flattened_ordered_C_embeddings",
                    "operands": "unchanged_ordered_numeric_inputs",
                    "fields": "ordered_C_then_scalar_times_numeric_embedding",
                    "numeric_bias": False,
                    "denominator_validity": "unchanged_pass_through",
                },
            }
        )

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.contract_state())

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "contract_sha256": self.contract_sha256,
            "contract_state": self.contract_state(),
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise EncoderContractError(
                "checkpoint encoder metadata does not match the live contract"
            )


class TrainableEncodedModel(nn.Module):
    """One optimizer/checkpoint boundary owning encoder, interaction, and head."""

    CONTRACT: ClassVar[str] = "dcdn-raw-to-prediction-model-v1"

    def __init__(
        self,
        *,
        encoder: SharedFeatureEncoder,
        controlled_model: ProvenancedControlledModel,
    ) -> None:
        super().__init__()
        if encoder.model_spec != controlled_model.encoder_spec:
            raise EncoderContractError(
                "encoder output spec and controlled model input spec differ"
            )
        requires_aux = controlled_model.interaction.requires_afn_aux
        if encoder.include_afn_secondary != requires_aux:
            raise EncoderContractError(
                "AFN secondary encoder view does not match the model family"
            )
        if (
            encoder.spec.allowed_model_keys
            and controlled_model.model_key not in encoder.spec.allowed_model_keys
        ):
            raise EncoderContractError(
                "model family is forbidden for this frozen feature regime"
            )
        self.encoder = encoder
        self.controlled_model = controlled_model

    @property
    def model_key(self) -> str:
        return self.controlled_model.model_key

    @property
    def display_label(self) -> str:
        return self.controlled_model.display_label

    @property
    def encoder_spec(self) -> EncoderSpec:
        return self.controlled_model.encoder_spec

    @property
    def interaction(self) -> nn.Module:
        return self.controlled_model.interaction

    @property
    def head(self) -> nn.Module:
        return self.controlled_model.head

    def encode_raw_batch(self, batch: RawFeatureBatch) -> EncodedInteractionBatch:
        """Encode only C/P/validity; targets and audit indices are inaccessible."""

        if not isinstance(batch, RawFeatureBatch):
            raise TypeError("raw input must be RawFeatureBatch")
        return self.encoder(
            batch.context_tokens,
            batch.operand_features,
            batch.denominator_validity,
        )

    def forward(self, batch: RawFeatureBatch) -> PredictionOutput:
        return self.controlled_model(self.encode_raw_batch(batch))

    def contract_state(self) -> dict[str, Any]:
        return canonical_config(
            {
                "contract": self.CONTRACT,
                "encoder_contract_sha256": self.encoder.contract_sha256,
                "encoder_contract": self.encoder.contract_state(),
                "controlled_model_contract_sha256": (
                    self.controlled_model.contract_sha256
                ),
                "controlled_model_contract": self.controlled_model.contract_state(),
                "checkpoint_scope": "encoder_interaction_adapter_head",
                "optimizer_scope": "all_trainable_parameters_in_this_module",
            }
        )

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.contract_state())

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "contract_sha256": self.contract_sha256,
            "contract_state": self.contract_state(),
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise EncoderContractError(
                "joint checkpoint metadata does not match the live contract"
            )


def build_trainable_encoded_model(
    *,
    encoder_contract: EncoderContractLock,
    dataset_id: str,
    source_identity: RawSourceIdentity,
    interaction_config: InteractionConfig,
    head_config: HeadConfig,
    run_seed: int,
    runtime_value_checks: bool = True,
) -> TrainableEncodedModel:
    """Build the complete raw-to-logit module without touching CUDA RNG state."""

    spec = resolve_dataset_encoder_spec(
        encoder_contract,
        dataset_id=dataset_id,
        source_identity=source_identity,
    )
    model = build_controlled_model(
        encoder_spec=spec.model_spec,
        interaction_config=interaction_config,
        head_config=head_config,
        init_seed=run_seed,
    )
    encoder = SharedFeatureEncoder(
        spec,
        run_seed=run_seed,
        include_afn_secondary=model.interaction.requires_afn_aux,
        runtime_value_checks=runtime_value_checks,
    )
    return TrainableEncodedModel(encoder=encoder, controlled_model=model)


def canonical_encoder_metadata_json(model: TrainableEncodedModel) -> str:
    """Return stable checkpoint/run-registry metadata for the joint module."""

    if not isinstance(model, TrainableEncodedModel):
        raise TypeError("model must be TrainableEncodedModel")
    return json.dumps(
        model.contract_state(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
