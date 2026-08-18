"""Guarded adapter for the preserved production-hybrid DCDN block.

``DCDNBlock_provided.py`` is immutable provenance and remains the source for
``DCDN-Hybrid-Prod-L1-PLE8``.  The adapter below reuses that instance's exact
learned modules while adding an experimental execution contract: strict
``[B, 8, F]`` validation and an FP32 reciprocal calculation for FP32/BF16
execution.  It is not a claim that the unchanged production source originally
contained these additional guards.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

PROVIDED_BLOCK_SHA256 = (
    "d58c31c03096d112a47718a643fc91bc5dd9b5a53cd7941948145b34335ec1bb"
)
PROVIDED_BLOCK_FILENAME = "DCDNBlock_provided.py"


def production_hybrid_parameter_formula(features_count: int) -> int:
    if isinstance(features_count, bool) or not isinstance(features_count, int):
        raise ValueError("features_count must be a positive integer")
    if features_count <= 0:
        raise ValueError("features_count must be a positive integer")
    return 3 * features_count * features_count + 67 * features_count + 48


def _autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
            return bool(torch.is_autocast_cpu_enabled())
        return bool(torch.is_autocast_enabled())


def _require_linear(
    module: Any,
    *,
    name: str,
    in_features: int,
    out_features: int,
) -> nn.Linear:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"provided block {name} must be nn.Linear")
    if module.in_features != in_features or module.out_features != out_features:
        raise ValueError(
            f"provided block {name} must map {in_features} -> {out_features}"
        )
    if module.bias is None:
        raise ValueError(f"provided block {name} must retain its bias")
    return module


def validate_provided_block_structure(
    provided_block: nn.Module,
    features_count: int,
) -> None:
    """Fail if the adapter is pointed at a different hybrid topology."""

    if not isinstance(provided_block, nn.Module):
        raise TypeError("provided_block must be an nn.Module")
    production_hybrid_parameter_formula(features_count)
    _require_linear(
        getattr(provided_block, "block", None),
        name="block",
        in_features=features_count,
        out_features=features_count,
    )
    _require_linear(
        getattr(provided_block, "div_block", None),
        name="div_block",
        in_features=features_count,
        out_features=features_count,
    )
    fc_block = getattr(provided_block, "fc_block", None)
    if not isinstance(fc_block, nn.Sequential) or len(fc_block) != 4:
        raise ValueError("provided block fc_block must contain exactly four modules")
    _require_linear(
        fc_block[0],
        name="fc_block[0]",
        in_features=features_count,
        out_features=32,
    )
    if not isinstance(fc_block[1], nn.LeakyReLU):
        raise TypeError("provided block fc_block[1] must be LeakyReLU")
    if not isinstance(fc_block[2], nn.BatchNorm1d) or fc_block[2].num_features != 8:
        raise ValueError("provided block must retain BatchNorm1d(8)")
    if not isinstance(fc_block[3], nn.Dropout) or fc_block[3].p != 0.1:
        raise ValueError("provided block must retain Dropout(0.1)")
    _require_linear(
        getattr(provided_block, "projector", None),
        name="projector",
        in_features=features_count + 32,
        out_features=features_count,
    )
    div_eps = float(getattr(provided_block, "div_eps", float("nan")))
    if not math.isfinite(div_eps) or div_eps <= 0:
        raise ValueError("provided block div_eps must be finite and positive")
    actual_parameters = sum(
        parameter.numel() for parameter in provided_block.parameters()
    )
    expected_parameters = production_hybrid_parameter_formula(features_count)
    if actual_parameters != expected_parameters:
        raise ValueError(
            "provided production-hybrid parameter count mismatch: "
            f"{actual_parameters} != {expected_parameters}"
        )


class ExperimentalProductionHybridAdapter(nn.Module):
    """Strict FP32-reciprocal adapter over the immutable provided block.

    The source block remains unmasked across all PLE coordinates, matching the
    confirmed production topology.  The semantic raw-denominator mask used by
    :class:`dcdn.core.DCDNCore` must never be supplied here.
    """

    ARCHITECTURE_LABEL: ClassVar[str] = "DCDN-Hybrid-Prod-L1-PLE8"

    def __init__(
        self,
        provided_block: nn.Module,
        *,
        features_count: int,
    ) -> None:
        super().__init__()
        validate_provided_block_structure(provided_block, features_count)
        self.features_count = features_count
        self.ple_channels = 8
        self.provided_block = provided_block

    @classmethod
    def from_provided_source(
        cls,
        *,
        features_count: int,
        div_eps: float = 1e-2,
        div_init_scale: float = 0.1,
    ) -> ExperimentalProductionHybridAdapter:
        """Instantiate the immutable root source lazily from the repository."""

        source_path = Path(__file__).resolve().parents[2] / PROVIDED_BLOCK_FILENAME
        if not source_path.is_file():
            raise ImportError(
                f"production provenance source is unavailable: {source_path}"
            )
        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_digest != PROVIDED_BLOCK_SHA256:
            raise ImportError(
                "production provenance source hash mismatch: "
                f"expected {PROVIDED_BLOCK_SHA256}, got {source_digest}"
            )
        module_name = "DCDNBlock_provided"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, source_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load production provenance: {source_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
        module_path = Path(getattr(module, "__file__", "")).resolve()
        if module_path != source_path.resolve():
            raise ImportError(
                f"unexpected production provenance module path: {module_path}"
            )
        block_type = getattr(module, "DCDNBlock", None)
        if block_type is None:
            raise ImportError("DCDNBlock_provided.DCDNBlock is unavailable")
        provided_block = block_type(
            features_count=features_count,
            div_eps=div_eps,
            div_init_scale=div_init_scale,
        )
        return cls(provided_block, features_count=features_count)

    def provenance_metadata(self) -> dict[str, Any]:
        return {
            "architecture": self.ARCHITECTURE_LABEL,
            "source_file": PROVIDED_BLOCK_FILENAME,
            "source_sha256": PROVIDED_BLOCK_SHA256,
            "source_unchanged": True,
            "features_count": self.features_count,
            "ple_channels": self.ple_channels,
            "depth": 1,
            "initial_state_contract": "x_i=x_0",
            "anchor_contract": "fixed x_0",
            "semantic_denominator_mask": False,
            "adapter_reciprocal_compute_dtype": "float32",
            "source_guard_delta": (
                "experimental adapter adds validation and FP32 reciprocal; "
                "the provenance source itself is unchanged"
            ),
        }

    def _check_inputs(self, x_0: torch.Tensor, x_i: torch.Tensor) -> None:
        if not isinstance(x_0, torch.Tensor) or not isinstance(x_i, torch.Tensor):
            raise TypeError("x_0 and x_i must be tensors")
        expected_suffix = (self.ple_channels, self.features_count)
        if x_0.ndim != 3 or tuple(x_0.shape[1:]) != expected_suffix:
            raise ValueError(
                "production-hybrid inputs must have shape "
                f"[B, 8, {self.features_count}]"
            )
        if x_0.shape[0] <= 0:
            raise ValueError("production-hybrid batch dimension must be positive")
        if x_i.shape != x_0.shape:
            raise ValueError("x_0 and x_i must have identical [B, 8, F] shapes")
        if x_i.dtype != x_0.dtype:
            raise TypeError("x_0 and x_i must have identical dtypes")
        if x_i.device != x_0.device:
            raise ValueError("x_0 and x_i must share a device")
        if x_0.dtype == torch.float16:
            raise TypeError("FP16 is forbidden for the hybrid reciprocal path")
        if x_0.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("hybrid inputs must use FP32 or BF16")
        if torch.any(~torch.isfinite(x_0)) or torch.any(~torch.isfinite(x_i)):
            raise ValueError("x_0 and x_i must be finite")

        parameter_devices = {
            parameter.device for parameter in self.provided_block.parameters()
        }
        parameter_dtypes = {
            parameter.dtype for parameter in self.provided_block.parameters()
        }
        buffer_devices = {buffer.device for buffer in self.provided_block.buffers()}
        if len(parameter_devices) != 1 or len(parameter_dtypes) != 1:
            raise RuntimeError("provided block parameters must share device and dtype")
        parameter_device = next(iter(parameter_devices))
        parameter_dtype = next(iter(parameter_dtypes))
        if parameter_dtype == torch.float16:
            raise TypeError("FP16 parameters are forbidden for the hybrid adapter")
        if parameter_dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("hybrid parameters must use FP32 or BF16")
        if x_0.device != parameter_device:
            raise ValueError("hybrid input and parameters must share a device")
        if buffer_devices and buffer_devices != {x_0.device}:
            raise ValueError("hybrid buffers and inputs must share a device")
        if x_0.dtype != parameter_dtype and not _autocast_enabled(x_0.device.type):
            raise TypeError(
                "hybrid input and parameter dtypes must match outside autocast"
            )

        div_eps = float(self.provided_block.div_eps)
        if not math.isfinite(div_eps) or div_eps <= 0:
            raise ValueError("provided block div_eps must remain finite and positive")

    def forward(self, x_0: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        self._check_inputs(x_0, x_i)

        # Preserve the source's sign-preserving hard-floor semantics while
        # executing thresholding and reciprocal in FP32 for FP32/BF16 inputs.
        x_0_fp32 = x_0.to(dtype=torch.float32)
        sign = torch.sign(x_0_fp32)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        eps = float(self.provided_block.div_eps)
        safe_x_0_fp32 = torch.where(
            x_0_fp32.abs() > eps,
            x_0_fp32,
            sign * eps,
        )
        inv_x_0 = safe_x_0_fp32.reciprocal().to(dtype=x_0.dtype)

        cross_x = x_i + x_0 * self.provided_block.block(x_i)
        div_x = inv_x_0 * self.provided_block.div_block(x_i)
        cross_div = cross_x + div_x
        fc_x = self.provided_block.fc_block(x_i)
        result = self.provided_block.projector(torch.cat([cross_div, fc_x], dim=2))
        if result.dtype != x_i.dtype:
            result = result.to(dtype=x_i.dtype)
        if result.shape != x_i.shape or result.device != x_i.device:
            raise RuntimeError("hybrid adapter violated its shape/device contract")
        if torch.any(~torch.isfinite(result)):
            raise FloatingPointError("hybrid adapter produced non-finite output")
        return result


class ExperimentalProductionHybridOneBlock(nn.Module):
    """Minimal confirmed one-block wrapper with ``x_i = x_0``.

    This is only the confirmed DCDN block boundary, not a reconstruction of the
    still-undocumented end-to-end production PLE preprocessing, tower, or head.
    """

    ARCHITECTURE_LABEL: ClassVar[str] = "DCDN-Hybrid-Prod-L1-PLE8"

    def __init__(self, adapter: ExperimentalProductionHybridAdapter) -> None:
        super().__init__()
        if not isinstance(adapter, ExperimentalProductionHybridAdapter):
            raise TypeError("adapter must be ExperimentalProductionHybridAdapter")
        self.adapter = adapter

    @classmethod
    def from_provided_source(
        cls,
        *,
        features_count: int,
        div_eps: float = 1e-2,
        div_init_scale: float = 0.1,
    ) -> ExperimentalProductionHybridOneBlock:
        return cls(
            ExperimentalProductionHybridAdapter.from_provided_source(
                features_count=features_count,
                div_eps=div_eps,
                div_init_scale=div_init_scale,
            )
        )

    def provenance_metadata(self) -> dict[str, Any]:
        metadata = self.adapter.provenance_metadata()
        metadata["wrapper_scope"] = (
            "confirmed one-block x_i=x_0 boundary only; end-to-end production "
            "wrapper remains incomplete"
        )
        return metadata

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        return self.adapter(x_0, x_0)
