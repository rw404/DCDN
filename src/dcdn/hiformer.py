"""HiFormer-style heterogeneous attention layer, for serving-cost study only.

Scope and honesty
-----------------
This is a **HiFormer-style** layer, not a faithful reproduction of the published
architecture: it implements the property that matters for the cost question —
heterogeneous per-feature-group projections over field tokens with attention
cost ``O(F^2 d)`` — inside the project's fairness contract. It is **not** part of
the frozen nine-model suite, produces no quality number, and no result cell
depends on it. The repository's `divide_dcnv3_hiformer.ipynb` is a legacy
prototype and is not authoritative.

Why it is measured at all
-------------------------
The reduced feature view removes 17 manually materialised divisions, so a
ranker-width model sees 67 field tokens on ``REDUCED`` against 84 on ``FULL``.
Any downstream component whose cost grows superlinearly in the field count turns
that reduction into a serving saving. Self-attention is exactly such a
component: ``(85/68)^2 = 1.56``. A production stack that puts an attention layer
after the cross therefore changes the DCDN-versus-DCNv2 cost comparison in a way
the cross-only stack cannot show, which is what this module exists to measure.

Placement
---------
One layer, immediately after the interaction block, as::

    encoder -> field tokens [B, F, d]
            -> flat [B, bypass + operands]
    cross(flat) -> crossed [B, flat]
    cross token  = Linear(flat -> d)            [B, 1, d]
    tokens       = concat(cross token, fields)  [B, F+1, d]
    HiFormer(tokens)                            [B, F+1, d]
    flatten -> adapter -> shared tower -> per-target heads

The cross result enters attention as one additional token so the layer sees both
the interaction output and the original fields, and so the token count is driven
by the feature regime — which is the quantity under test.

Heterogeneity
-------------
Vanilla self-attention applies one projection to every token. HiFormer's central
idea is that different *kinds* of feature deserve different projections. Here the
tokens carry two declared groups — categorical fields and numeric operands (the
cross token joins the numeric group) — and each group owns its own query, key,
and value matrices. Attention is then computed over the whole sequence, so a
categorical field and an operand interact through group-specific projections
rather than a shared one.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import torch
from torch import nn


class HiFormerConfigError(ValueError):
    """A HiFormer-style layer was configured inconsistently."""


class HeterogeneousAttentionLayer(nn.Module):
    """One pre-norm heterogeneous-attention block over field tokens."""

    CONTRACT: ClassVar[str] = "dcdn-hiformer-style-heterogeneous-attention-v1"

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        group_sizes: tuple[int, ...],
        ffn_ratio: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads:
            raise HiFormerConfigError(
                "embedding_dim must be divisible by num_heads"
            )
        if not group_sizes or any(size <= 0 for size in group_sizes):
            raise HiFormerConfigError("group_sizes must be positive integers")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.group_sizes = tuple(group_sizes)
        self.num_tokens = sum(group_sizes)

        self.norm_in = nn.LayerNorm(embedding_dim)
        # One query/key/value projection per declared feature group: this is
        # what makes the layer heterogeneous rather than a plain transformer.
        self.query = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in group_sizes
        )
        self.key = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in group_sizes
        )
        self.value = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in group_sizes
        )
        self.project = nn.Linear(embedding_dim, embedding_dim)
        self.norm_ffn = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(embedding_dim * ffn_ratio, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def _grouped(self, tokens: torch.Tensor, projections: nn.ModuleList
                 ) -> torch.Tensor:
        """Apply each group's projection to its own slice, then reassemble."""

        pieces = []
        start = 0
        for size, projection in zip(self.group_sizes, projections, strict=True):
            pieces.append(projection(tokens[:, start : start + size]))
            start += size
        return torch.cat(pieces, dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != self.num_tokens:
            raise HiFormerConfigError(
                f"tokens must have shape [B, {self.num_tokens}, "
                f"{self.embedding_dim}]"
            )
        batch = tokens.shape[0]
        normed = self.norm_in(tokens)

        def heads(values: torch.Tensor) -> torch.Tensor:
            return values.view(
                batch, self.num_tokens, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query = heads(self._grouped(normed, self.query))
        key = heads(self._grouped(normed, self.key))
        value = heads(self._grouped(normed, self.value))

        # The O(F^2 d) term the whole study is about.
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        mixed = (attention @ value).transpose(1, 2).reshape(
            batch, self.num_tokens, self.embedding_dim
        )
        tokens = tokens + self.dropout(self.project(mixed))
        return tokens + self.dropout(self.ffn(self.norm_ffn(tokens)))

    def attention_flops_per_example(self) -> int:
        """Score and weighted-sum cost, the part that scales as ``F^2``."""

        return 4 * self.num_tokens * self.num_tokens * self.embedding_dim

    def contract_state(self) -> dict[str, Any]:
        return {
            "contract": self.CONTRACT,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "group_sizes": list(self.group_sizes),
            "num_tokens": self.num_tokens,
            "heterogeneous_projections_per_group": True,
            "faithful_hiformer_reproduction": False,
            "part_of_frozen_model_suite": False,
            "produces_quality_numbers": False,
        }


__all__ = [
    "HeterogeneousAttentionLayer",
    "HiFormerConfigError",
]
