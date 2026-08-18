"""Numerically guarded theorem-aligned DCDN core.

This module implements ``DCDN-Core-L{L}``, the public/offline architecture.
It intentionally does not contain the production-only FC, BatchNorm, Dropout,
concatenation, or projection path.  The original anchor is fixed for every
layer and the reciprocal carrier is computed once from that anchor.

The exact Laurent interpretation applies only to exact reciprocals on a valid
nonzero domain.  This implementation uses a declared positive floor and is
therefore a stabilized, piecewise-exact training operator.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import torch
from torch import nn

_ALLOWED_FLOAT_DTYPES = (torch.float32, torch.bfloat16)


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _require_finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _canonical_mask(
    features_count: int,
    denominator_mask: Sequence[bool] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(denominator_mask, torch.Tensor):
        mask = denominator_mask.detach()
    else:
        try:
            mask = torch.as_tensor(denominator_mask)
        except (TypeError, ValueError) as exc:
            raise TypeError("denominator_mask must be a boolean sequence") from exc
    if mask.dtype != torch.bool:
        raise TypeError("denominator_mask must contain booleans")
    if mask.ndim != 1 or mask.numel() != features_count:
        raise ValueError(
            f"denominator_mask must be one-dimensional with {features_count} entries"
        )
    return mask.to(device="cpu", dtype=torch.bool).clone().contiguous()


def _mask_sha256(mask: torch.Tensor) -> str:
    payload = bytes(mask.detach().to(device="cpu", dtype=torch.uint8).tolist())
    return hashlib.sha256(payload).hexdigest()


def _validate_optional_sha256(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    if any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
            return bool(torch.is_autocast_cpu_enabled())
        return bool(torch.is_autocast_enabled())


def _bad_index_sample(mask: torch.Tensor, limit: int = 8) -> list[list[int]]:
    return torch.nonzero(mask, as_tuple=False)[:limit].detach().cpu().tolist()


class DCDNCoreBlock(nn.Module):
    """One residual/product/division affine layer over ``[..., F]``.

    A block deliberately owns no denominator mask.  ``DCDNCore`` stores one
    canonical mask, constructs one fixed reciprocal carrier, and passes it to
    every block.  This prevents per-layer mask drift and duplicate mask state.
    """

    def __init__(
        self,
        features_count: int,
        product_init_scale: float = 0.1,
        div_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.features_count = _require_positive_int("features_count", features_count)
        self.product_init_scale = _require_finite_nonnegative(
            "product_init_scale", product_init_scale
        )
        self.div_init_scale = _require_finite_nonnegative(
            "div_init_scale", div_init_scale
        )

        self.product_block = nn.Linear(features_count, features_count, bias=True)
        self.division_block = nn.Linear(features_count, features_count, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.product_block.weight)
        nn.init.kaiming_normal_(self.division_block.weight)
        with torch.no_grad():
            self.product_block.weight.mul_(self.product_init_scale)
            self.division_block.weight.mul_(self.div_init_scale)
        nn.init.zeros_(self.product_block.bias)
        nn.init.zeros_(self.division_block.bias)

    def _check_pair_and_carrier(
        self,
        x_0: torch.Tensor,
        x_i: torch.Tensor,
        inv_x_0: torch.Tensor,
    ) -> None:
        if not all(isinstance(tensor, torch.Tensor) for tensor in (x_0, x_i, inv_x_0)):
            raise TypeError("x_0, x_i, and inv_x_0 must be tensors")
        if x_0.shape != x_i.shape or x_0.shape != inv_x_0.shape:
            raise ValueError("x_0, x_i, and inv_x_0 must have identical shapes")
        if x_0.ndim < 2 or x_0.shape[-1] != self.features_count:
            raise ValueError("inputs must have shape [..., features_count]")
        if x_0.dtype != x_i.dtype or x_0.dtype != inv_x_0.dtype:
            raise TypeError("x_0, x_i, and inv_x_0 must have identical dtypes")
        if x_0.device != x_i.device or x_0.device != inv_x_0.device:
            raise ValueError("x_0, x_i, and inv_x_0 must be on one device")
        if x_0.dtype == torch.float16:
            raise TypeError("FP16 is forbidden for the DCDN reciprocal path")
        if x_0.dtype not in _ALLOWED_FLOAT_DTYPES:
            raise TypeError("DCDN inputs must use FP32 or BF16")
        parameter_devices = {parameter.device for parameter in self.parameters()}
        parameter_dtypes = {parameter.dtype for parameter in self.parameters()}
        if len(parameter_devices) != 1 or len(parameter_dtypes) != 1:
            raise RuntimeError("all block parameters must share one device and dtype")
        parameter_device = next(iter(parameter_devices))
        parameter_dtype = next(iter(parameter_dtypes))
        if parameter_dtype == torch.float16:
            raise TypeError("FP16 parameters are forbidden for DCDN-Core")
        if parameter_dtype not in _ALLOWED_FLOAT_DTYPES:
            raise TypeError("DCDN parameters must use FP32 or BF16")
        if x_0.device != parameter_device:
            raise ValueError("block inputs and parameters must share a device")
        if x_0.dtype != parameter_dtype and not _autocast_enabled(x_0.device.type):
            raise TypeError(
                "block input and parameter dtypes must match outside autocast"
            )

    def forward_with_carrier(
        self,
        x_0: torch.Tensor,
        x_i: torch.Tensor,
        inv_x_0: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the frozen recurrence using a prevalidated carrier."""

        self._check_pair_and_carrier(x_0, x_i, inv_x_0)
        result = (
            x_i + x_0 * self.product_block(x_i) + inv_x_0 * self.division_block(x_i)
        )
        # Autocast policies can promote elementwise terms.  Keep the public
        # interface stable and make the state dtype explicit.
        if result.dtype != x_i.dtype:
            result = result.to(dtype=x_i.dtype)
        return result

    def forward(
        self,
        x_0: torch.Tensor,
        x_i: torch.Tensor,
        inv_x_0: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_carrier(x_0, x_i, inv_x_0)


class DCDNCore(nn.Module):
    """Fixed-anchor stack for public raw/base-operand experiments."""

    SERIALIZATION_CONTRACT: ClassVar[str] = "dcdn-core-serialization-v1"
    RECURRENCE_CONTRACT: ClassVar[str] = (
        "x_i + x_0*product_block(x_i) + masked_reciprocal(x_0)*division_block(x_i)"
    )

    def __init__(
        self,
        features_count: int,
        num_layers: int,
        denominator_mask: Sequence[bool] | torch.Tensor,
        div_eps: float = 1e-2,
        product_init_scale: float = 0.1,
        div_init_scale: float = 0.1,
        runtime_value_checks: bool = True,
        feature_schema_sha256: str | None = None,
    ) -> None:
        super().__init__()
        self.features_count = _require_positive_int("features_count", features_count)
        self.num_layers = _require_positive_int("num_layers", num_layers)
        self.div_eps = _require_finite_positive("div_eps", div_eps)
        self.product_init_scale = _require_finite_nonnegative(
            "product_init_scale", product_init_scale
        )
        self.div_init_scale = _require_finite_nonnegative(
            "div_init_scale", div_init_scale
        )
        if not isinstance(runtime_value_checks, bool):
            raise TypeError("runtime_value_checks must be a boolean")
        self.runtime_value_checks = runtime_value_checks
        self.feature_schema_sha256 = _validate_optional_sha256(
            "feature_schema_sha256", feature_schema_sha256
        )

        # This is the single canonical mask for the complete stack.  Blocks do
        # not register their own copies.
        self.register_buffer(
            "_denominator_mask",
            _canonical_mask(features_count, denominator_mask),
            persistent=True,
        )
        self.layers = nn.ModuleList(
            DCDNCoreBlock(
                features_count=features_count,
                product_init_scale=self.product_init_scale,
                div_init_scale=self.div_init_scale,
            )
            for _ in range(num_layers)
        )

    @property
    def architecture_label(self) -> str:
        return f"DCDN-Core-L{self.num_layers}"

    @property
    def denominator_mask(self) -> torch.Tensor:
        """Return a defensive copy of the canonical semantic mask."""

        return self._denominator_mask.detach().clone()

    @property
    def denominator_mask_sha256(self) -> str:
        return _mask_sha256(self._denominator_mask)

    def _parameter_dtype_name(self) -> str:
        parameter_dtypes = {parameter.dtype for parameter in self.parameters()}
        if parameter_dtypes == {torch.float32}:
            return "float32"
        if parameter_dtypes == {torch.bfloat16}:
            return "bfloat16"
        raise TypeError(
            "serialization requires all DCDN parameters to use FP32 or BF16"
        )

    @staticmethod
    def parameter_count_formula(features_count: int, num_layers: int) -> int:
        features_count = _require_positive_int("features_count", features_count)
        num_layers = _require_positive_int("num_layers", num_layers)
        return num_layers * 2 * features_count * (features_count + 1)

    def _check_model_and_input(self, x_0: torch.Tensor) -> None:
        if not isinstance(x_0, torch.Tensor):
            raise TypeError("x_0 must be a tensor")
        if x_0.ndim < 2 or x_0.shape[-1] != self.features_count:
            raise ValueError("x_0 must have shape [..., features_count]")
        if x_0.dtype == torch.float16:
            raise TypeError("FP16 is forbidden for the DCDN reciprocal path")
        if x_0.dtype not in _ALLOWED_FLOAT_DTYPES:
            raise TypeError("DCDN inputs must use FP32 or BF16")

        parameter_devices = {parameter.device for parameter in self.parameters()}
        parameter_dtypes = {parameter.dtype for parameter in self.parameters()}
        if len(parameter_devices) != 1 or len(parameter_dtypes) != 1:
            raise RuntimeError("all DCDN parameters must share one device and dtype")
        parameter_device = next(iter(parameter_devices))
        parameter_dtype = next(iter(parameter_dtypes))
        if parameter_dtype == torch.float16:
            raise TypeError("FP16 parameters are forbidden for DCDN-Core")
        if parameter_dtype not in _ALLOWED_FLOAT_DTYPES:
            raise TypeError("DCDN parameters must use FP32 or BF16")
        if (
            x_0.device != parameter_device
            or x_0.device != self._denominator_mask.device
        ):
            raise ValueError(
                "input, parameters, and denominator mask must share a device"
            )
        if x_0.dtype != parameter_dtype and not _autocast_enabled(x_0.device.type):
            raise TypeError(
                "input and parameter dtypes must match outside an autocast context"
            )

    def _check_validity(
        self,
        x_0: torch.Tensor,
        denominator_validity: torch.Tensor | None,
    ) -> torch.Tensor:
        if denominator_validity is None:
            return torch.ones_like(x_0, dtype=torch.bool)
        if not isinstance(denominator_validity, torch.Tensor):
            raise TypeError("denominator_validity must be a tensor or None")
        if denominator_validity.shape != x_0.shape:
            raise ValueError("denominator_validity must match x_0.shape")
        if denominator_validity.dtype != torch.bool:
            raise TypeError("denominator_validity must be a boolean tensor")
        if denominator_validity.device != x_0.device:
            raise ValueError("denominator_validity must be on x_0.device")
        return denominator_validity

    def reciprocal_carrier(
        self,
        x_0: torch.Tensor,
        denominator_validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the one fixed masked carrier in FP32, then cast it back.

        Debug mode raises indexed Python errors.  Optimized mode contains no
        synchronous value predicates: invalid active denominators become NaN
        sentinels for the mandatory asynchronous non-finite monitor, never a
        silently clamped positive reciprocal.  Invalid/missing denominator rows
        are expected to contain a finite fallback established by preflight.
        """

        self._check_model_and_input(x_0)
        valid = self._check_validity(x_0, denominator_validity)
        view_shape = (1,) * (x_0.ndim - 1) + (self.features_count,)
        eligible = self._denominator_mask.view(view_shape)
        active = eligible & valid
        finite = torch.isfinite(x_0)

        invalid_active = active & (~finite | (x_0 <= 0))
        if self.runtime_value_checks:
            nonfinite_fallback = eligible & ~valid & ~finite
            if torch.any(nonfinite_fallback):
                raise ValueError(
                    "invalid denominator rows require a finite fallback; indices="
                    f"{_bad_index_sample(nonfinite_fallback)}"
                )
            if torch.any(invalid_active):
                raise ValueError(
                    "active denominator coordinates must be finite and positive; "
                    f"indices={_bad_index_sample(invalid_active)}"
                )

        # Mask before division.  The conversion occurs before flooring and
        # reciprocal so FP32 is used for both FP32 and BF16 model states.
        valid_active = active & ~invalid_active
        source_fp32 = torch.where(
            valid_active,
            x_0.to(dtype=torch.float32),
            torch.ones_like(x_0, dtype=torch.float32),
        )
        reciprocal_fp32 = source_fp32.clamp_min(self.div_eps).reciprocal()
        reciprocal_fp32 = torch.where(
            valid_active, reciprocal_fp32, torch.zeros_like(reciprocal_fp32)
        )
        # In optimized mode this is the fail-loud sentinel consumed by the
        # asynchronous loss/activation monitor.  Debug mode has already raised.
        reciprocal_fp32 = torch.where(
            invalid_active,
            torch.full_like(reciprocal_fp32, float("nan")),
            reciprocal_fp32,
        )
        return reciprocal_fp32.to(dtype=x_0.dtype)

    def forward(
        self,
        x_0: torch.Tensor,
        denominator_validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_model_and_input(x_0)
        inv_x_0 = self.reciprocal_carrier(x_0, denominator_validity)
        if self.runtime_value_checks and torch.any(~torch.isfinite(x_0)):
            raise ValueError("x_0 must be finite")
        x_i = x_0
        for layer_index, layer in enumerate(self.layers):
            x_i = layer.forward_with_carrier(x_0, x_i, inv_x_0)
            if self.runtime_value_checks and torch.any(~torch.isfinite(x_i)):
                raise FloatingPointError(
                    f"non-finite DCDN activation after layer {layer_index}"
                )
        return x_i

    def serialization_metadata(self) -> dict[str, Any]:
        """Return the deterministic architecture/configuration contract."""

        mask_list = [bool(value) for value in self._denominator_mask.cpu().tolist()]
        return {
            "contract": self.SERIALIZATION_CONTRACT,
            "architecture": self.architecture_label,
            "fixed_anchor": True,
            "recurrence": self.RECURRENCE_CONTRACT,
            "features_count": self.features_count,
            "num_layers": self.num_layers,
            "div_eps": self.div_eps,
            "product_init_scale": self.product_init_scale,
            "div_init_scale": self.div_init_scale,
            "runtime_value_checks": self.runtime_value_checks,
            "feature_schema_sha256": self.feature_schema_sha256,
            "denominator_mask": mask_list,
            "denominator_mask_sha256": self.denominator_mask_sha256,
            "precision": {
                "allowed_input_dtypes": ["float32", "bfloat16"],
                "parameter_dtype": self._parameter_dtype_name(),
                "reciprocal_compute_dtype": "float32",
                "float16_rejected": True,
            },
        }

    @staticmethod
    def metadata_sha256(metadata: Mapping[str, Any]) -> str:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        canonical = json.dumps(
            dict(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @property
    def serialization_metadata_sha256(self) -> str:
        return self.metadata_sha256(self.serialization_metadata())

    def validate_serialization_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Fail if serialized state belongs to another DCDN contract."""

        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        expected = self.serialization_metadata()
        received = dict(metadata)
        if received != expected:
            differing_keys = sorted(
                key
                for key in set(expected) | set(received)
                if expected.get(key) != received.get(key)
            )
            raise ValueError(
                "DCDN serialization metadata mismatch for keys: "
                + ", ".join(differing_keys)
            )

    def state_dict_with_metadata(self) -> dict[str, Any]:
        metadata = self.serialization_metadata()
        return {
            "metadata": metadata,
            "metadata_sha256": self.metadata_sha256(metadata),
            "state_dict": self.state_dict(),
        }

    def load_state_dict_with_metadata(
        self,
        payload: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> Any:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        expected_keys = {"metadata", "metadata_sha256", "state_dict"}
        if set(payload) != expected_keys:
            raise ValueError(
                "serialized payload must contain exactly metadata, "
                "metadata_sha256, and state_dict"
            )
        metadata = payload["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("payload metadata must be a mapping")
        digest = payload["metadata_sha256"]
        if digest != self.metadata_sha256(metadata):
            raise ValueError("serialized metadata SHA-256 mismatch")
        self.validate_serialization_metadata(metadata)
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError("payload state_dict must be a mapping")
        return self.load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_serialization_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> DCDNCore:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        required = {
            "features_count",
            "num_layers",
            "denominator_mask",
            "div_eps",
            "product_init_scale",
            "div_init_scale",
            "runtime_value_checks",
            "feature_schema_sha256",
        }
        missing = sorted(required - set(metadata))
        if missing:
            raise ValueError("metadata is missing keys: " + ", ".join(missing))
        model = cls(
            features_count=metadata["features_count"],
            num_layers=metadata["num_layers"],
            denominator_mask=metadata["denominator_mask"],
            div_eps=metadata["div_eps"],
            product_init_scale=metadata["product_init_scale"],
            div_init_scale=metadata["div_init_scale"],
            runtime_value_checks=metadata["runtime_value_checks"],
            feature_schema_sha256=metadata["feature_schema_sha256"],
        )
        precision = metadata.get("precision")
        if not isinstance(precision, Mapping):
            raise ValueError("metadata precision contract must be a mapping")
        parameter_dtype_name = precision.get("parameter_dtype")
        parameter_dtype_by_name = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if parameter_dtype_name not in parameter_dtype_by_name:
            raise ValueError("metadata parameter_dtype must be float32 or bfloat16")
        model.to(dtype=parameter_dtype_by_name[parameter_dtype_name])
        model.validate_serialization_metadata(metadata)
        return model
