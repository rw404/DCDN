"""Shared encoder and runtime that drive the audited suite on the benchmark.

``Synthetic-MultiQuotient-3`` has no categorical fields, so the shared encoder
here does exactly one thing: it lifts each positive scalar operand into a field
embedding, which is what the FM/attention/logarithmic models require.  The
operand vector itself passes through unchanged, so ``DCDN-Core`` sees the raw
positive operands its reciprocal mask is defined on.

Fairness is structural:

* every arm receives the *same* encoder architecture built from the *same*
  seed, so encoder parameters are identical at initialization across arms;
* every arm receives the same examples, the same labels, and the same masks;
* the interaction block is the only architectural difference;
* the shared adapter and prediction head are built by the audited
  ``build_controlled_model`` factory, which derives independent component seeds
  and therefore gives every arm *identical* head weights for a given base seed.

The prediction head is ``CommonPredictionHead`` with ``output_dim = T``.  A
single ``Linear(d, T)`` is exactly ``T`` separate ``Linear(d, 1)`` heads: row
``t`` of the weight matrix and entry ``t`` of the bias are target ``t``'s head,
no parameter is shared between targets, and no gradient crosses between them.
``tests/test_synthetic_runtime.py`` asserts that independence directly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn

from dcdn.models.common import HeadConfig
from dcdn.models.contracts import EncodedInteractionBatch, EncoderSpec
from dcdn.models.suite import InteractionConfig, build_controlled_model
from dcdn.synthetic_multiquotient import (
    COLUMNS,
    DENOMINATOR_MASK,
    MaterializedSlice,
)


def ordered_schema_sha256(columns: Sequence[str]) -> str:
    """Hash the ordered feature schema exactly as the runtime consumes it."""

    payload = "\n".join(columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


OPERAND_SCHEMA_SHA256 = ordered_schema_sha256(COLUMNS)
#: With no categorical bypass the full schema equals the operand schema.
FULL_SCHEMA_SHA256 = OPERAND_SCHEMA_SHA256


class SyntheticFieldEncoder(nn.Module):
    """Per-field scalar lift shared verbatim by every arm.

    ``embedding[b, f, :] = weight[f] * x[b, f] + bias[f]``

    This is the standard numeric-field embedding used by FM-style models: one
    learned direction per field, modulated by the field's value.  AFN+ owns a
    second, independent view of the same ordered fields; it is produced here so
    that the auxiliary view is part of the shared encoder rather than a private
    extra feature regime.
    """

    CONTRACT: ClassVar[str] = "dcdn-synthetic-field-encoder-v1"

    def __init__(
        self, num_fields: int, embedding_dim: int, *, with_afn_view: bool
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.with_afn_view = bool(with_afn_view)
        self.weight = nn.Parameter(torch.empty(num_fields, embedding_dim))
        self.bias = nn.Parameter(torch.empty(num_fields, embedding_dim))
        if self.with_afn_view:
            self.afn_weight = nn.Parameter(torch.empty(num_fields, embedding_dim))
            self.afn_bias = nn.Parameter(torch.empty(num_fields, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name in ("weight", "afn_weight"):
            parameter = getattr(self, name, None)
            if parameter is not None:
                nn.init.normal_(parameter, mean=0.0, std=0.1)
        for name in ("bias", "afn_bias"):
            parameter = getattr(self, name, None)
            if parameter is not None:
                nn.init.zeros_(parameter)

    def forward(
        self, operands: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if operands.ndim != 2 or operands.shape[1] != self.num_fields:
            raise ValueError(f"operands must have shape [B, {self.num_fields}]")
        values = operands.unsqueeze(-1)
        fields = values * self.weight + self.bias
        auxiliary = None
        if self.with_afn_view:
            auxiliary = values * self.afn_weight + self.afn_bias
        return fields, auxiliary

    def contract_state(self) -> dict[str, Any]:
        return {
            "contract": self.CONTRACT,
            "num_fields": self.num_fields,
            "embedding_dim": self.embedding_dim,
            "with_afn_view": self.with_afn_view,
            "lift": "embedding[b,f,:] = weight[f] * x[b,f] + bias[f]",
            "initializer": "normal_std_0.1_weight_zero_bias",
        }


def encoder_spec(embedding_dim: int, representation_dim: int) -> EncoderSpec:
    """The one encoder shape every arm of the benchmark consumes."""

    return EncoderSpec(
        bypass_dim=0,
        operand_dim=len(COLUMNS),
        num_fields=len(COLUMNS),
        embedding_dim=embedding_dim,
        representation_dim=representation_dim,
        denominator_mask=DENOMINATOR_MASK,
        full_feature_schema_sha256=FULL_SCHEMA_SHA256,
        operand_schema_sha256=OPERAND_SCHEMA_SHA256,
    )


@dataclass(frozen=True)
class ArmSpec:
    """One row of the synthetic comparison matrix."""

    arm_id: str
    label: str
    interaction_config: InteractionConfig | None
    explicit_ratio_inputs: bool = False
    analytical: bool = False


class MultiQuotientModel(nn.Module):
    """Shared encoder + audited interaction model with ``T`` heads."""

    def __init__(
        self,
        spec: EncoderSpec,
        interaction_config: InteractionConfig,
        num_targets: int,
        head_hidden_dims: tuple[int, ...],
        init_seed: int,
        *,
        with_afn_view: bool,
    ) -> None:
        super().__init__()
        # Construct the encoder under its own recorded seed so that every arm
        # starts from identical encoder parameters.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.random.default_generator.manual_seed(_encoder_seed(init_seed))
            self.encoder = SyntheticFieldEncoder(
                num_fields=spec.num_fields,
                embedding_dim=spec.embedding_dim,
                with_afn_view=with_afn_view,
            )
        self.model = build_controlled_model(
            encoder_spec=spec,
            interaction_config=interaction_config,
            head_config=HeadConfig(
                hidden_dims=head_hidden_dims,
                dropout=0.0,
                output_dim=num_targets,
            ),
            init_seed=init_seed,
        )
        self.spec = spec

    @property
    def model_key(self) -> str:
        return self.model.model_key

    def forward(
        self,
        operands: torch.Tensor,
        denominator_validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fields, auxiliary = self.encoder(operands)
        batch = EncodedInteractionBatch(
            bypass_features=operands.new_zeros((operands.shape[0], 0)),
            operand_features=operands,
            field_embeddings=fields,
            denominator_validity=denominator_validity,
            afn_dnn_embeddings=auxiliary,
        )
        return self.model(batch).logits

    def parameter_counts(self) -> dict[str, int]:
        """Countable capacity, split into the shared and arm-specific parts.

        The encoder is common to every arm, so capacity matching operates on
        ``interaction + adapter + head`` — the trainable non-embedding
        parameters that actually differ between rows.
        """

        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        interaction = count(self.model.interaction)
        adapter = count(self.model.adapter)
        head = count(self.model.head)
        return {
            "encoder_shared": count(self.encoder),
            "interaction": interaction,
            "adapter": adapter,
            "head": head,
            "matched": interaction + adapter + head,
            "total": count(self),
        }


def _encoder_seed(init_seed: int) -> int:
    payload = f"dcdn-synthetic-encoder-init-v1:{init_seed}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def slice_tensors(
    item: MaterializedSlice, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move one slice to the compute device as ``(x, y, mask, validity)``.

    Validity is all-true: every synthetic denominator is generated strictly
    positive and above the reciprocal floor, so any floor activation observed
    at evaluation time is a defect rather than an expected gate.
    """

    features = item.features.to(device)
    validity = torch.ones_like(features, dtype=torch.bool)
    return (
        features,
        item.targets.to(device),
        item.masks.to(device),
        validity,
    )


__all__ = [
    "FULL_SCHEMA_SHA256",
    "OPERAND_SCHEMA_SHA256",
    "ArmSpec",
    "MultiQuotientModel",
    "SyntheticFieldEncoder",
    "encoder_spec",
    "ordered_schema_sha256",
    "slice_tensors",
]
