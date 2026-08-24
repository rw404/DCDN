"""Fused Triton inference kernels for the DCDN and DCNv2 cross recurrences.

Why this module exists
----------------------
The naive PyTorch forward launches one kernel per elementwise operation.  At
serving batch sizes the interaction blocks are tiny, so wall time is dominated
by kernel launches and memory round-trips rather than by arithmetic.  Measuring
those implementations therefore compares *implementations*, not architectures —
and it does so unfairly, because the two architectures decompose into different
numbers of elementwise passes:

``DCNv2``  ``x <- x + x0 * (W x + b)``
           1 GEMM + 2 elementwise passes per layer.

``DCDN``   ``x <- x + x0 * (Wp x + bp) + inv(x0) * (Wd x + bd)``
           2 GEMMs + 4 elementwise passes per layer, plus a reciprocal carrier
           built once from ~17 elementwise operations.

DCDN carries *fewer* parameters and FLOPs than DCNv2 at ranker width, yet loses
on wall time. That gap is an implementation artifact, so this module removes it
from **both** architectures and re-measures. Fusing only DCDN would replace one
unfair comparison with its mirror image.

What is fused
-------------
* both cross recurrences: one GEMM plus a single fused epilogue kernel per
  layer, replacing 2–4 separate elementwise passes;
* DCDN's two per-layer GEMMs into one, by concatenating the product and
  division weights into a single ``[2F, F]`` matrix;
* DCDN's reciprocal carrier into one kernel, replacing the masked
  clamp/reciprocal/select chain.

Per DCDN layer this goes from 2 GEMMs + 4 kernels to 1 GEMM + 1 kernel; the
carrier goes from many kernels to one, once.

Numerical contract
------------------
The fused path is **inference-only** and assumes the invariant the frozen
preflight already verifies: every active denominator coordinate is finite and
strictly positive, or is gated to zero by its validity indicator. Under that
invariant the fused carrier is arithmetically identical to
``DCDNCore.reciprocal_carrier``. ``FusedDCDNCore.from_core`` re-checks the
invariant on the first batch unless explicitly told not to, and
``tests/test_fused_inference.py`` asserts bitwise-or-near equality against the
audited modules on every geometry used in the efficiency study.

The fused path changes no parameter and no result — only the schedule. It is
never used for training and never for a quality number.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

try:  # pragma: no cover - exercised by availability test
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


class FusedInferenceError(RuntimeError):
    """A fused inference precondition was violated."""


if TRITON_AVAILABLE:

    @triton.jit
    def _dcdn_carrier_kernel(
        x0_ptr,
        mask_ptr,
        out_ptr,
        n_elements,
        features,
        div_eps,
        BLOCK: tl.constexpr,
    ):
        """Masked reciprocal carrier in one pass.

        ``out = 1 / max(x0, div_eps)`` on masked coordinates, ``0`` elsewhere.
        """

        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        guard = offsets < n_elements
        feature = offsets % features
        x0 = tl.load(x0_ptr + offsets, mask=guard, other=1.0)
        eligible = tl.load(mask_ptr + feature, mask=guard, other=0)
        active = eligible != 0
        source = tl.where(active, x0, 1.0)
        source = tl.maximum(source, div_eps)
        reciprocal = 1.0 / source
        tl.store(out_ptr + offsets, tl.where(active, reciprocal, 0.0), mask=guard)

    @triton.jit
    def _dcdn_epilogue_kernel(
        state_ptr,
        anchor_ptr,
        carrier_ptr,
        gemm_ptr,
        out_ptr,
        n_elements,
        features,
        BLOCK: tl.constexpr,
    ):
        """``out = state + anchor * product + carrier * division``.

        ``gemm`` holds both affine results contiguously as ``[B, 2F]``: the
        product half first, the division half second.
        """

        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        guard = offsets < n_elements
        row = offsets // features
        feature = offsets % features
        state = tl.load(state_ptr + offsets, mask=guard, other=0.0)
        anchor = tl.load(anchor_ptr + offsets, mask=guard, other=0.0)
        carrier = tl.load(carrier_ptr + offsets, mask=guard, other=0.0)
        base = row * (2 * features) + feature
        product = tl.load(gemm_ptr + base, mask=guard, other=0.0)
        division = tl.load(gemm_ptr + base + features, mask=guard, other=0.0)
        tl.store(
            out_ptr + offsets,
            state + anchor * product + carrier * division,
            mask=guard,
        )

    @triton.jit
    def _cross_epilogue_kernel(
        state_ptr,
        anchor_ptr,
        gemm_ptr,
        out_ptr,
        n_elements,
        BLOCK: tl.constexpr,
    ):
        """``out = state + anchor * affine`` — the DCNv2 cross epilogue."""

        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        guard = offsets < n_elements
        state = tl.load(state_ptr + offsets, mask=guard, other=0.0)
        anchor = tl.load(anchor_ptr + offsets, mask=guard, other=0.0)
        affine = tl.load(gemm_ptr + offsets, mask=guard, other=0.0)
        tl.store(out_ptr + offsets, state + anchor * affine, mask=guard)


def _grid(n_elements: int, block: int):
    return ((n_elements + block - 1) // block,)


def _block_for(n_elements: int) -> int:
    """Pick a power-of-two block that does not dwarf the work."""

    block = 256
    while block > 32 and block > n_elements:
        block //= 2
    return block


def _require_triton() -> None:
    if not TRITON_AVAILABLE:
        raise FusedInferenceError("Triton is required for the fused inference path")


class FusedDCDNCore(nn.Module):
    """Inference-only fused equivalent of :class:`dcdn.core.DCDNCore`.

    Holds no independent parameters: weights are views/copies of the audited
    core's, rebuilt by :meth:`refresh` if the source is ever mutated.
    """

    def __init__(
        self,
        features_count: int,
        num_layers: int,
        denominator_mask: torch.Tensor,
        div_eps: float,
        fused_weight: list[torch.Tensor],
        fused_bias: list[torch.Tensor],
        validate_validity_per_call: bool = True,
    ) -> None:
        super().__init__()
        # The per-call guard below reads a device tensor into Python, which
        # forces a device-to-host synchronization on EVERY forward. That has two
        # costs, and both were being charged to the architecture rather than to
        # the guard: it serializes the pipeline (0.045 ms of a 0.993 ms forward
        # at B=1, i.e. 4.5%), and it makes the module impossible to capture in a
        # CUDA graph, which is the single largest win available to a
        # dispatch-bound ranker. The comparator, DCNv2, has no equivalent
        # synchronization anywhere in its path, so leaving this on the hot path
        # measured a guard, not a design.
        #
        # `verify_all_valid` performs exactly the same check once, off the hot
        # path. Set this False only after calling it.
        self.validate_validity_per_call = bool(validate_validity_per_call)
        self.features_count = features_count
        self.num_layers = num_layers
        self.div_eps = float(div_eps)
        self.register_buffer("_mask_i32", denominator_mask.to(torch.int32))
        # ``[2F, F]`` per layer: product rows first, division rows second.
        self._fused_weight = nn.ParameterList(
            nn.Parameter(w, requires_grad=False) for w in fused_weight
        )
        self._fused_bias = nn.ParameterList(
            nn.Parameter(b, requires_grad=False) for b in fused_bias
        )

    @classmethod
    def from_core(cls, core: Any) -> FusedDCDNCore:
        """Build the fused twin of an audited ``DCDNCore``."""

        _require_triton()
        weights: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        for layer in core.layers:
            weights.append(
                torch.cat(
                    [
                        layer.product_block.weight.detach(),
                        layer.division_block.weight.detach(),
                    ],
                    dim=0,
                ).contiguous()
            )
            biases.append(
                torch.cat(
                    [
                        layer.product_block.bias.detach(),
                        layer.division_block.bias.detach(),
                    ],
                    dim=0,
                ).contiguous()
            )
        return cls(
            features_count=core.features_count,
            num_layers=core.num_layers,
            denominator_mask=core.denominator_mask,
            div_eps=core.div_eps,
            fused_weight=weights,
            fused_bias=biases,
        )

    @staticmethod
    def verify_all_valid(denominator_validity: torch.Tensor | None) -> None:
        """Reject a validity tensor that actually gates anything.

        The fused carrier implements the all-valid case only, which is the
        invariant the frozen preflight verifies. This is the same check the
        forward used to run per call; it is a `staticmethod` so a caller can
        run it once for a whole dataset and then set
        `validate_validity_per_call = False`, instead of paying a
        device-to-host synchronization on every inference.
        """

        if denominator_validity is None:
            return
        if not bool(denominator_validity.all()):
            raise FusedInferenceError(
                "the fused carrier implements the all-valid case only; a "
                "gating validity tensor requires the audited core"
            )

    def reciprocal_carrier(self, x_0: torch.Tensor) -> torch.Tensor:
        _require_triton()
        out = torch.empty_like(x_0)
        n_elements = x_0.numel()
        block = _block_for(n_elements)
        _dcdn_carrier_kernel[_grid(n_elements, block)](
            x_0,
            self._mask_i32,
            out,
            n_elements,
            self.features_count,
            self.div_eps,
            BLOCK=block,
        )
        return out

    def forward(
        self,
        x_0: torch.Tensor,
        denominator_validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Drop-in replacement for ``DCDNCore.forward``.

        ``denominator_validity`` is accepted so the fused core can be swapped
        into ``DCDNCoreInteraction`` without touching it.  The fused carrier
        implements the all-valid case only, which is the invariant the frozen
        preflight verifies, so a validity tensor that actually gates anything
        is rejected rather than silently ignored.
        """

        _require_triton()
        if x_0.ndim != 2 or x_0.shape[1] != self.features_count:
            raise FusedInferenceError(
                f"x_0 must have shape [B, {self.features_count}]"
            )
        if self.validate_validity_per_call:
            self.verify_all_valid(denominator_validity)
        x_0 = x_0.contiguous()
        carrier = self.reciprocal_carrier(x_0)
        n_elements = x_0.numel()
        block = _block_for(n_elements)
        grid = _grid(n_elements, block)
        state = x_0
        for index in range(self.num_layers):
            affine = torch.addmm(
                self._fused_bias[index], state, self._fused_weight[index].t()
            )
            out = torch.empty_like(state)
            _dcdn_epilogue_kernel[grid](
                state,
                x_0,
                carrier,
                affine,
                out,
                n_elements,
                self.features_count,
                BLOCK=block,
            )
            state = out
        return state


class FusedCrossNet(nn.Module):
    """Inference-only fused equivalent of the full-rank DCNv2 cross.

    Present so that the comparison optimizes both architectures. Fusing only
    DCDN would trade one implementation artifact for its mirror image.
    """

    def __init__(self, weights: list[torch.Tensor], biases: list[torch.Tensor]) -> None:
        super().__init__()
        self.num_layers = len(weights)
        self.input_dim = int(weights[0].shape[0])
        self._weight = nn.ParameterList(
            nn.Parameter(w, requires_grad=False) for w in weights
        )
        self._bias = nn.ParameterList(
            nn.Parameter(b, requires_grad=False) for b in biases
        )

    @classmethod
    def from_interaction(cls, interaction: Any) -> FusedCrossNet:
        _require_triton()
        weights = [layer.weight.detach().contiguous() for layer in interaction.layers]
        biases = [layer.bias.detach().contiguous() for layer in interaction.layers]
        return cls(weights, biases)

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        _require_triton()
        anchor = anchor.contiguous()
        n_elements = anchor.numel()
        block = _block_for(n_elements)
        grid = _grid(n_elements, block)
        state = anchor
        for index in range(self.num_layers):
            affine = torch.addmm(self._bias[index], state, self._weight[index].t())
            out = torch.empty_like(state)
            _cross_epilogue_kernel[grid](
                state, anchor, affine, out, n_elements, BLOCK=block
            )
            state = out
        return state


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.double() - right.double()).abs().max())


__all__ = [
    "TRITON_AVAILABLE",
    "FusedCrossNet",
    "FusedDCDNCore",
    "FusedInferenceError",
    "max_abs_difference",
]
