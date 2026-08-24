"""Clean-room EulerNet interaction block.

Why this model is here rather than as an afterthought
-----------------------------------------------------
`DCDN_HANDOFF.md` names EulerNet-like approaches as prior art that the novelty
claim must not ignore, alongside AFN+ and NALU/NPU. The reason is algebraic and
specific: EulerNet moves each field into polar form, and in that space a
monomial with arbitrary real exponents is a *linear* map on log-amplitude and
phase. A learned coefficient of -1 is therefore reachable by ordinary gradient
descent, so division is inside its hypothesis class by construction - exactly
the property DCDN's reciprocal branch is built to provide.

AFN+ has the same property through a log-exp branch and pays for it: in this
programme's measurements it carries 1.5M-3.2M parameters and costs 3.9x-11.3x
DCDN's latency. Whether EulerNet reproduces that cost or avoids it decides how
narrow DCDN's niche actually is, which is why it is worth the arm.

Deviations from the source, declared
------------------------------------
* The source paper trains with its own embedding table and output head; here the
  block sits at the shared post-encoder boundary and its output goes through the
  same `RepresentationAdapter` every other arm uses, so parameters and FLOPs are
  counted the same way.
* Amplitude, phase and the exponential are evaluated in FP32 regardless of the
  surrounding dtype, matching the treatment AFN+ receives in this repository.
* The order matrix is initialised by a softmax over a scaled Gaussian, as in the
  reference implementation, which starts the layer near a convex combination of
  input fields rather than at an arbitrary monomial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from dcdn.models.common import InteractionBlock, initialize_output_linear
from dcdn.models.contracts import (
    EncodedInteractionBatch,
    EncoderSpec,
    require_finite_positive,
    require_positive_int,
    require_probability,
)

#: Guards the logarithm of the squared modulus. The source uses 1e-8 on the
#: squared quantity, which this matches rather than re-deriving.
MODULUS_FLOOR = 1e-8


@dataclass(frozen=True)
class EulerNetConfig:
    num_layers: int
    fields_per_layer: int
    dropout: float
    exp_clip: float | None

    def __post_init__(self) -> None:
        require_positive_int("num_layers", self.num_layers)
        require_positive_int("fields_per_layer", self.fields_per_layer)
        require_probability("dropout", self.dropout)
        if self.exp_clip is not None:
            require_finite_positive("exp_clip", self.exp_clip)


class EulerInteractionLayer(nn.Module):
    """One explicit-plus-implicit Euler layer.

    The explicit half computes ``exp(A @ log |z|)`` and ``A @ arg(z)`` over the
    field axis, which is a monomial with learned real exponents; the implicit
    half is an ordinary linear map on the rectangular coordinates. Their sum is
    layer-normalised, as in the source.
    """

    def __init__(
        self, in_fields: int, out_fields: int, embedding_dim: int,
        dropout: float, exp_clip: float | None,
    ) -> None:
        super().__init__()
        self.in_fields = in_fields
        self.out_fields = out_fields
        self.embedding_dim = embedding_dim
        self.exp_clip = exp_clip
        orders = torch.softmax(torch.randn(in_fields, out_fields) / 0.01, dim=0)
        self.orders = nn.Parameter(orders)
        self.implicit = nn.Linear(in_fields * embedding_dim, out_fields * embedding_dim)
        initialize_output_linear(self.implicit)
        self.bias_amplitude = nn.Parameter(
            torch.randn(1, embedding_dim, out_fields) * 0.01)
        self.bias_phase = nn.Parameter(
            torch.randn(1, embedding_dim, out_fields) * 0.01)
        self.norm_real = nn.LayerNorm(embedding_dim)
        self.norm_imaginary = nn.LayerNorm(embedding_dim)
        self.drop_explicit = nn.Dropout(dropout)
        self.drop_implicit = nn.Dropout(dropout)

    def forward(
        self, real: torch.Tensor, imaginary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_fp32 = real.to(torch.float32)
        imaginary_fp32 = imaginary.to(torch.float32)
        squared_modulus = real_fp32.square() + imaginary_fp32.square() + MODULUS_FLOOR
        # 0.5 * log(r^2 + p^2) is log|z| without a square root on the hot path.
        log_amplitude = 0.5 * torch.log(squared_modulus)
        # The floor protects the amplitude but not the phase, and the two fail
        # differently: at the origin `atan2(0, 0)` returns 0 with a NaN
        # gradient, so a single exactly-zero field embedding poisons the whole
        # backward pass. This matters here and not in the source implementation
        # because a numeric operand of exactly zero produces an exactly zero
        # field embedding, which is 1.02% of cells on VK-LSVD ur - dense
        # categorical embeddings never hit the singularity.
        #
        # The phase is undefined at the origin, so it is defined to be zero
        # there and the gradient is blocked from that branch rather than
        # allowed to be NaN.
        degenerate = (real_fp32 == 0.0) & (imaginary_fp32 == 0.0)
        phase = torch.atan2(
            torch.where(degenerate, torch.zeros_like(imaginary_fp32), imaginary_fp32),
            torch.where(degenerate, torch.ones_like(real_fp32), real_fp32),
        )

        orders = self.orders.to(torch.float32)
        log_amplitude = log_amplitude.transpose(1, 2) @ orders + self.bias_amplitude
        phase = phase.transpose(1, 2) @ orders + self.bias_phase
        if self.exp_clip is not None:
            log_amplitude = log_amplitude.clamp(-self.exp_clip, self.exp_clip)
        amplitude = torch.exp(log_amplitude)

        explicit_real = (amplitude * torch.cos(phase)).transpose(1, 2)
        explicit_imaginary = (amplitude * torch.sin(phase)).transpose(1, 2)
        explicit_real = self.drop_explicit(explicit_real.to(real.dtype))
        explicit_imaginary = self.drop_explicit(explicit_imaginary.to(real.dtype))

        batch = real.shape[0]
        implicit_real = self.implicit(
            self.drop_implicit(real.reshape(batch, -1))
        ).reshape(batch, self.out_fields, self.embedding_dim)
        implicit_imaginary = self.implicit(
            self.drop_implicit(imaginary.reshape(batch, -1))
        ).reshape(batch, self.out_fields, self.embedding_dim)

        return (
            self.norm_real(explicit_real + implicit_real),
            self.norm_imaginary(explicit_imaginary + implicit_imaginary),
        )


class EulerNetInteraction(InteractionBlock):
    """Stacked Euler layers over a learned complex lift of the field view."""

    model_key = "eulernet"
    display_label = "EulerNet"

    def __init__(self, spec: EncoderSpec, config: EulerNetConfig) -> None:
        width = config.fields_per_layer
        super().__init__(2 * width * spec.embedding_dim, config)
        self.num_fields = spec.num_fields
        self.embedding_dim = spec.embedding_dim
        # The lift follows the reference: the embedding supplies both parts, a
        # learnable per-field scale separating them so the initial phase is not
        # identically pi/4.
        self.mu = nn.Parameter(torch.ones(1, spec.num_fields, 1))
        layers = []
        in_fields = spec.num_fields
        for _ in range(config.num_layers):
            layers.append(EulerInteractionLayer(
                in_fields, width, spec.embedding_dim, config.dropout, config.exp_clip))
            in_fields = width
        self.layers = nn.ModuleList(layers)

    def forward(self, batch: EncodedInteractionBatch) -> torch.Tensor:
        fields = batch.field_embeddings
        if fields.ndim != 3 or fields.shape[1:] != (self.num_fields, self.embedding_dim):
            raise ValueError("EulerNet fields violate the configured [B, N, E] shape")
        real = self.mu * fields
        imaginary = fields
        for layer in self.layers:
            real, imaginary = layer(real, imaginary)
        return torch.cat(
            (real.flatten(start_dim=1), imaginary.flatten(start_dim=1)), dim=-1)

    def extra_contract_state(self) -> dict[str, Any]:
        return {
            "polar_compute_dtype": "float32",
            "modulus_floor": MODULUS_FLOOR,
            "order_initialisation": "softmax_over_gaussian_scaled_by_0.01",
            "exp_policy": ("unbounded" if self.config.exp_clip is None
                           else "explicit_symmetric_pre_exp_clamp"),
            "negative_exponents_reachable": True,
            "phase_at_origin": "defined_zero_with_gradient_blocked",
            "clean_room": True,
        }
