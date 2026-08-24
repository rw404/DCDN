"""Clean-room Wukong interaction stack with explicit configuration.

No constructor default encodes a paper scale.  The experiment manifest must
name every layer/FMB/LCB/rank/MLP choice, which prevents an implementation
README default or an inferred Cartesian-grid combination from becoming the
frozen Wukong-small baseline accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from dcdn.models.common import (
    DenseTower,
    InteractionBlock,
    initialize_hidden_linear,
)
from dcdn.models.contracts import (
    EncodedInteractionBatch,
    EncoderSpec,
    require_dims,
    require_positive_int,
    require_probability,
)


@dataclass(frozen=True)
class WukongSmallConfig:
    num_layers: int
    lcb_features: int
    fmb_features: int
    fmb_rank: int
    fmb_hidden_dims: tuple[int, ...]
    dropout: float
    layer_norm: bool

    def __post_init__(self) -> None:
        require_positive_int("num_layers", self.num_layers)
        require_positive_int("lcb_features", self.lcb_features)
        require_positive_int("fmb_features", self.fmb_features)
        require_positive_int("fmb_rank", self.fmb_rank)
        require_dims("fmb_hidden_dims", self.fmb_hidden_dims, allow_empty=False)
        require_probability("dropout", self.dropout)
        if not isinstance(self.layer_norm, bool):
            raise TypeError("layer_norm must be boolean")


class LinearCompressionBlock(nn.Module):
    def __init__(self, input_fields: int, output_fields: int) -> None:
        super().__init__()
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.weight = nn.Parameter(torch.empty(input_fields, output_fields))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bne,no->boe", fields, self.weight)


class FactorizationMachineBlock(nn.Module):
    """Optimized ``X (X^T Y)`` FMB followed by an explicit MLP."""

    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        embedding_dim: int,
        rank: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.projection = nn.Parameter(torch.empty(input_fields, rank))
        nn.init.kaiming_uniform_(self.projection)
        self.normalization = nn.LayerNorm(input_fields * rank)
        self.tower = DenseTower(
            input_fields * rank,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.output = nn.Linear(self.tower.output_dim, output_fields * embedding_dim)
        initialize_hidden_linear(self.output)

    def interaction_matrix(self, fields: torch.Tensor) -> torch.Tensor:
        projected = torch.einsum("bne,nr->ber", fields, self.projection)
        return torch.bmm(fields, projected)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        interactions = self.interaction_matrix(fields).flatten(start_dim=1)
        values = torch.relu(self.output(self.tower(self.normalization(interactions))))
        return values.reshape(fields.shape[0], self.output_fields, self.embedding_dim)


class WukongLayer(nn.Module):
    """Parallel FMB/LCB followed by residual projection and optional LN."""

    def __init__(
        self,
        input_fields: int,
        embedding_dim: int,
        config: WukongSmallConfig,
    ) -> None:
        super().__init__()
        output_fields = config.fmb_features + config.lcb_features
        self.fmb = FactorizationMachineBlock(
            input_fields=input_fields,
            output_fields=config.fmb_features,
            embedding_dim=embedding_dim,
            rank=config.fmb_rank,
            hidden_dims=config.fmb_hidden_dims,
            dropout=config.dropout,
        )
        self.lcb = LinearCompressionBlock(input_fields, config.lcb_features)
        self.residual = (
            nn.Identity()
            if input_fields == output_fields
            else LinearCompressionBlock(input_fields, output_fields)
        )
        self.normalization = (
            nn.LayerNorm(embedding_dim) if config.layer_norm else nn.Identity()
        )
        self.output_fields = output_fields

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        branches = torch.cat((self.fmb(fields), self.lcb(fields)), dim=1)
        return self.normalization(branches + self.residual(fields))


class WukongInteraction(InteractionBlock):
    model_key = "wukong_small"
    display_label = "Wukong-small"

    def __init__(self, spec: EncoderSpec, config: WukongSmallConfig) -> None:
        output_fields = config.fmb_features + config.lcb_features
        super().__init__(output_fields * spec.embedding_dim, config)
        layers: list[WukongLayer] = []
        input_fields = spec.num_fields
        for _ in range(config.num_layers):
            layer = WukongLayer(input_fields, spec.embedding_dim, config)
            layers.append(layer)
            input_fields = output_fields
        self.layers = nn.ModuleList(layers)
        self.output_fields = output_fields
        self.embedding_dim = spec.embedding_dim

    def forward(self, batch: EncodedInteractionBatch) -> torch.Tensor:
        values = batch.field_embeddings
        for layer in self.layers:
            values = layer(values)
        # Explicit flatten preserves B=1; no unconstrained squeeze is used.
        return values.reshape(batch.batch_size, self.output_fields * self.embedding_dim)

    def extra_contract_state(self) -> dict[str, Any]:
        return {
            "layer_equation": "LN(concat(FMB(X),LCB(X))+residual(X))",
            "fmb_equation": "reshape(ReLU(MLP(LN(flatten(X*(X^T*Y))))))",
            "fmb_output_activation": "relu",
            "constructor_has_paper_scale_defaults": False,
            "prediction_mlp_replaced_by": (
                "visible_representation_adapter_plus_common_head"
            ),
        }
