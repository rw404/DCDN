"""Theorem-aligned DCDN core for public offline experiments.

This is deliberately separate from ``DCDNBlock_provided.py``.  It contains
only the residual, product-cross, and reciprocal-cross recurrence.  Shared
context encoders and the prediction head belong outside this module; the
production-only FC/BatchNorm/Dropout path and concatenation projector are not
part of this core.

The reciprocal mask addresses named positive raw/base-operand denominators.
Missing denominators must use a finite fallback plus ``denominator_validity``;
invalid rows receive zero reciprocal contribution.  Exact Laurent theorems
refer to the unfloored reciprocal on a nonzero domain.  This implementation
uses a positive floor for numerical training and is therefore piecewise exact.
The stack defaults to asynchronous training without per-batch Python tensor
predicates; the experiment harness must first scan every materialized operand
for finiteness/positivity and must monitor losses, activations, and gradients.
Set ``runtime_value_checks=True`` only for smoke/debug runs because it forces
GPU synchronization.
"""

from collections.abc import Sequence

import torch
from torch import nn


class DCDNCoreBlock(nn.Module):
    """One fixed-anchor DCDN cross-division layer over ``[..., F]`` inputs."""

    def __init__(
        self,
        features_count: int,
        denominator_mask: Sequence[bool] | torch.Tensor,
        div_eps: float = 1e-2,
        product_init_scale: float = 0.1,
        div_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if features_count <= 0:
            raise ValueError("features_count must be positive")
        if div_eps <= 0:
            raise ValueError("div_eps must be positive")
        if product_init_scale < 0:
            raise ValueError("product_init_scale must be non-negative")
        if div_init_scale < 0:
            raise ValueError("div_init_scale must be non-negative")

        mask = torch.as_tensor(denominator_mask)
        if mask.dtype != torch.bool:
            raise TypeError("denominator_mask must contain booleans")
        if mask.ndim != 1 or mask.numel() != features_count:
            raise ValueError(
                "denominator_mask must be one-dimensional with "
                f"{features_count} entries"
            )

        self.features_count = features_count
        self.div_eps = float(div_eps)
        self.product_init_scale = float(product_init_scale)
        self.div_init_scale = float(div_init_scale)
        self.register_buffer("denominator_mask", mask.clone())

        self.product_block = nn.Linear(features_count, features_count, bias=True)
        self.division_block = nn.Linear(features_count, features_count, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.product_block.weight)
        with torch.no_grad():
            self.product_block.weight.mul_(self.product_init_scale)
        nn.init.zeros_(self.product_block.bias)
        nn.init.kaiming_normal_(self.division_block.weight)
        with torch.no_grad():
            self.division_block.weight.mul_(self.div_init_scale)
        nn.init.zeros_(self.division_block.bias)

    def _check_shapes(self, x_0: torch.Tensor, x_i: torch.Tensor) -> None:
        if x_0.shape != x_i.shape:
            raise ValueError("x_0 and x_i must have identical shapes")
        if x_0.ndim < 2 or x_0.shape[-1] != self.features_count:
            raise ValueError(
                "x_0 and x_i must have shape [..., features_count]"
            )
        if not x_0.is_floating_point() or not x_i.is_floating_point():
            raise TypeError("DCDN inputs must be floating-point tensors")

    def reciprocal_carrier(
        self,
        x_0: torch.Tensor,
        denominator_validity: torch.Tensor | None = None,
        validate_values: bool = True,
    ) -> torch.Tensor:
        """Return masked, validity-gated, positively floored reciprocals."""

        view_shape = (1,) * (x_0.ndim - 1) + (self.features_count,)
        eligible = self.denominator_mask.view(view_shape)

        if denominator_validity is None:
            valid = torch.ones_like(x_0, dtype=torch.bool)
        else:
            if denominator_validity.shape != x_0.shape:
                raise ValueError("denominator_validity must match x_0.shape")
            if denominator_validity.dtype != torch.bool:
                raise TypeError("denominator_validity must be a boolean tensor")
            if denominator_validity.device != x_0.device:
                raise ValueError("denominator_validity must be on x_0.device")
            valid = denominator_validity

        active = eligible & valid
        if validate_values:
            invalid_active = active & ((x_0 <= 0) | ~torch.isfinite(x_0))
            if torch.any(invalid_active):
                raise ValueError(
                    "active denominator coordinates must be finite and positive"
                )

        # Mask before division so ineligible or invalid coordinates cannot
        # propagate NaN/Inf through a later multiplication by zero.
        safe_source = torch.where(active, x_0, torch.ones_like(x_0))
        safe_source = safe_source.clamp_min(self.div_eps)
        # Compute the numerically sensitive reciprocal in FP32 under AMP.
        reciprocal = safe_source.float().reciprocal().to(dtype=x_0.dtype)
        return torch.where(active, reciprocal, torch.zeros_like(reciprocal))

    def forward_with_carrier(
        self,
        x_0: torch.Tensor,
        x_i: torch.Tensor,
        inv_x_0: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the recurrence using an already validated fixed carrier."""

        return (
            x_i
            + x_0 * self.product_block(x_i)
            + inv_x_0 * self.division_block(x_i)
        )

    def forward(
        self,
        x_0: torch.Tensor,
        x_i: torch.Tensor,
        denominator_validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_shapes(x_0, x_i)
        if not torch.all(torch.isfinite(x_0)) or not torch.all(torch.isfinite(x_i)):
            raise ValueError("x_0 and x_i must be finite")

        inv_x_0 = self.reciprocal_carrier(x_0, denominator_validity)
        return self.forward_with_carrier(x_0, x_i, inv_x_0)


class DCDNCore(nn.Module):
    """Stack DCDN core layers while keeping the original anchor fixed.

    Product and division weights use small scaled-Kaiming initialization so the
    residual path starts near identity.  Apply the same product initialization
    to DCN/DCNv2 controls in a fairness-sensitive experiment.
    """

    def __init__(
        self,
        features_count: int,
        num_layers: int,
        denominator_mask: Sequence[bool] | torch.Tensor,
        div_eps: float = 1e-2,
        product_init_scale: float = 0.1,
        div_init_scale: float = 0.1,
        runtime_value_checks: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.features_count = features_count
        self.num_layers = num_layers
        self.runtime_value_checks = runtime_value_checks
        self.layers = nn.ModuleList(
            DCDNCoreBlock(
                features_count=features_count,
                denominator_mask=denominator_mask,
                div_eps=div_eps,
                product_init_scale=product_init_scale,
                div_init_scale=div_init_scale,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        x_0: torch.Tensor,
        denominator_validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first_layer = self.layers[0]
        first_layer._check_shapes(x_0, x_0)
        if self.runtime_value_checks and not torch.all(torch.isfinite(x_0)):
            raise ValueError("x_0 must be finite")
        inv_x_0 = first_layer.reciprocal_carrier(
            x_0,
            denominator_validity,
            validate_values=self.runtime_value_checks,
        )

        x_i = x_0
        for layer in self.layers:
            x_i = layer.forward_with_carrier(x_0, x_i, inv_x_0)
        return x_i
