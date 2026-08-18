"""Clean-room DCN and DCNv2 interaction recurrences.

All recurrences keep their declared anchor fixed across layers. The ordinary
DCN/DCNv2 baselines cross the flat C+P representation. The capacity control
crosses only P, exactly like DCDN-Core, and concatenates C unchanged.
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
    require_finite_nonnegative,
    require_positive_int,
)


@dataclass(frozen=True)
class DCNConfig:
    num_layers: int
    product_init_scale: float

    def __post_init__(self) -> None:
        require_positive_int("num_layers", self.num_layers)
        require_finite_nonnegative("product_init_scale", self.product_init_scale)


@dataclass(frozen=True)
class DCNv2FullRankConfig:
    num_layers: int
    product_init_scale: float

    def __post_init__(self) -> None:
        require_positive_int("num_layers", self.num_layers)
        require_finite_nonnegative("product_init_scale", self.product_init_scale)


@dataclass(frozen=True)
class DCNv2CapacityCoreConfig:
    """Low-rank MoE bracket for ``DCNv2-capacity-matched-core``."""

    num_layers: int
    low_rank: int
    num_experts: int
    product_init_scale: float

    def __post_init__(self) -> None:
        require_positive_int("num_layers", self.num_layers)
        require_positive_int("low_rank", self.low_rank)
        require_positive_int("num_experts", self.num_experts)
        require_finite_nonnegative("product_init_scale", self.product_init_scale)


class OriginalDCNInteraction(InteractionBlock):
    """Original vector-parameterized DCN recurrence over flat C+P.

    ``x_{l+1} = x_l + x_0 (w_l^T x_l) + b_l``.
    """

    model_key = "dcn"
    display_label = "DCN"

    def __init__(self, spec: EncoderSpec, config: DCNConfig) -> None:
        super().__init__(spec.flat_dim, config)
        self.input_dim = spec.flat_dim
        self.weights = nn.ParameterList(
            nn.Parameter(torch.empty(self.input_dim)) for _ in range(config.num_layers)
        )
        self.biases = nn.ParameterList(
            nn.Parameter(torch.zeros(self.input_dim)) for _ in range(config.num_layers)
        )
        for weight in self.weights:
            nn.init.kaiming_normal_(weight.unsqueeze(0))
            with torch.no_grad():
                weight.mul_(config.product_init_scale)

    def forward(self, batch: EncodedInteractionBatch) -> torch.Tensor:
        anchor = batch.flat_features
        state = anchor
        for weight, bias in zip(self.weights, self.biases, strict=True):
            scalar = torch.sum(state * weight, dim=-1, keepdim=True)
            state = state + anchor * scalar + bias
        return state


class DCNv2FullRankInteraction(InteractionBlock):
    """Dense-matrix DCNv2 recurrence over flat C+P."""

    model_key = "dcnv2_full_rank"
    display_label = "DCNv2"

    def __init__(self, spec: EncoderSpec, config: DCNv2FullRankConfig) -> None:
        super().__init__(spec.flat_dim, config)
        self.input_dim = spec.flat_dim
        self.layers = nn.ModuleList(
            nn.Linear(self.input_dim, self.input_dim) for _ in range(config.num_layers)
        )
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight)
            with torch.no_grad():
                layer.weight.mul_(config.product_init_scale)
            nn.init.zeros_(layer.bias)

    def forward(self, batch: EncodedInteractionBatch) -> torch.Tensor:
        anchor = batch.flat_features
        state = anchor
        for layer in self.layers:
            state = state + anchor * layer(state)
        return state


class _LowRankExpert(nn.Module):
    def __init__(self, input_dim: int, low_rank: int, init_scale: float) -> None:
        super().__init__()
        self.v = nn.Parameter(torch.empty(input_dim, low_rank))
        self.c = nn.Parameter(torch.empty(low_rank, low_rank))
        self.u = nn.Parameter(torch.empty(input_dim, low_rank))
        nn.init.xavier_normal_(self.v)
        nn.init.xavier_normal_(self.c)
        nn.init.xavier_normal_(self.u)
        # Scaling U alone makes the complete expert term residual-near while
        # retaining the V-C-U factorization.
        with torch.no_grad():
            self.u.mul_(init_scale)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        low_rank_state = torch.tanh(state @ self.v)
        low_rank_state = torch.tanh(low_rank_state @ self.c)
        return low_rank_state @ self.u.transpose(0, 1)


class _LowRankMixLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        low_rank: int,
        num_experts: int,
        init_scale: float,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            _LowRankExpert(input_dim, low_rank, init_scale) for _ in range(num_experts)
        )
        self.gates = nn.ModuleList(
            nn.Linear(input_dim, 1, bias=False) for _ in range(num_experts)
        )
        for gate in self.gates:
            initialize_output_linear(gate)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, anchor: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        expert_crosses = torch.stack(
            [anchor * (expert(state) + self.bias) for expert in self.experts],
            dim=1,
        )
        gate_logits = torch.cat([gate(state) for gate in self.gates], dim=1)
        gate_weights = torch.softmax(gate_logits, dim=1).unsqueeze(-1)
        return state + torch.sum(expert_crosses * gate_weights, dim=1)


class DCNv2CapacityCoreInteraction(InteractionBlock):
    """Operand-only low-rank Mix control with unchanged context bypass."""

    model_key = "dcnv2_capacity_matched_core"
    display_label = "DCNv2-capacity-matched-core"

    def __init__(self, spec: EncoderSpec, config: DCNv2CapacityCoreConfig) -> None:
        super().__init__(spec.flat_dim, config)
        self.layers = nn.ModuleList(
            _LowRankMixLayer(
                input_dim=spec.operand_dim,
                low_rank=config.low_rank,
                num_experts=config.num_experts,
                init_scale=config.product_init_scale,
            )
            for _ in range(config.num_layers)
        )

    def forward(self, batch: EncodedInteractionBatch) -> torch.Tensor:
        anchor = batch.operand_features
        state = anchor
        for layer in self.layers:
            state = layer(anchor, state)
        # Deliberately no unconstrained squeeze(): [1, P] stays [1, P].
        return torch.cat((batch.bypass_features, state), dim=-1)

    def extra_contract_state(self) -> dict[str, Any]:
        return {
            "cross_parameterization": "low_rank_moe",
            "crossed_view": "operand_features_P_only",
            "bypass_view": "context_C_unchanged",
            "capacity_selection": "parameter_and_flop_distance_only",
            "batch_one_safe": True,
        }
