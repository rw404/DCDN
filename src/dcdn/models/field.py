"""Clean-room MLP, DeepFM, and AutoInt+ interaction blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from dcdn.models.common import (
    DenseTower,
    InteractionBlock,
    initialize_output_linear,
)
from dcdn.models.contracts import (
    EncodedInteractionBatch,
    EncoderSpec,
    require_dims,
    require_positive_int,
    require_probability,
)


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: tuple[int, ...]
    dropout: float

    def __post_init__(self) -> None:
        require_dims("hidden_dims", self.hidden_dims, allow_empty=False)
        require_probability("dropout", self.dropout)


@dataclass(frozen=True)
class DeepFMConfig:
    deep_hidden_dims: tuple[int, ...]
    dropout: float

    def __post_init__(self) -> None:
        require_dims("deep_hidden_dims", self.deep_hidden_dims, allow_empty=False)
        require_probability("dropout", self.dropout)


@dataclass(frozen=True)
class AutoIntPlusConfig:
    attention_layers: int
    attention_dim: int
    num_heads: int
    dnn_hidden_dims: tuple[int, ...]
    dropout: float
    scale_attention: bool
    layer_norm: bool

    def __post_init__(self) -> None:
        require_positive_int("attention_layers", self.attention_layers)
        require_positive_int("attention_dim", self.attention_dim)
        require_positive_int("num_heads", self.num_heads)
        if self.attention_dim % self.num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        # Non-empty is what distinguishes AutoInt+ from attention-only AutoInt.
        require_dims("dnn_hidden_dims", self.dnn_hidden_dims, allow_empty=False)
        require_probability("dropout", self.dropout)
        if not isinstance(self.scale_attention, bool):
            raise TypeError("scale_attention must be boolean")
        if not isinstance(self.layer_norm, bool):
            raise TypeError("layer_norm must be boolean")


class MLPInteraction(InteractionBlock):
    model_key = "dnn"
    display_label = "DNN/MLP"

    def __init__(self, spec: EncoderSpec, config: MLPConfig) -> None:
        super().__init__(config.hidden_dims[-1], config)
        self.tower = DenseTower(
            spec.flat_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )

    def forward(self, batch: EncodedInteractionBatch) -> torch.Tensor:
        return self.tower(batch.flat_features)


class DeepFMInteraction(InteractionBlock):
    """DeepFM branches over a shared, already encoded field tensor.

    The first-order term is a linear map over the same flattened field view;
    the second-order term is the standard FM identity; and the deep branch is
    an MLP.  Their raw outputs are concatenated before the visible common
    representation adapter.
    """

    model_key = "deepfm"
    display_label = "DeepFM"

    def __init__(self, spec: EncoderSpec, config: DeepFMConfig) -> None:
        super().__init__(2 + config.deep_hidden_dims[-1], config)
        self.flattened_dim = spec.flattened_field_dim
        self.first_order = nn.Linear(self.flattened_dim, 1)
        initialize_output_linear(self.first_order)
        self.deep = DenseTower(
            self.flattened_dim,
            hidden_dims=config.deep_hidden_dims,
            dropout=config.dropout,
        )

    @staticmethod
    def factorization_term(fields: torch.Tensor) -> torch.Tensor:
        if fields.ndim != 3:
            raise ValueError("fields must have shape [B, N, E]")
        summed = torch.sum(fields, dim=1)
        pair_vector = 0.5 * (summed.square() - torch.sum(fields.square(), dim=1))
        return torch.sum(pair_vector, dim=-1, keepdim=True)

    def forward(self, batch: EncodedInteractionBatch) -> torch.Tensor:
        flattened = batch.field_embeddings.flatten(start_dim=1)
        first_order = self.first_order(flattened)
        second_order = self.factorization_term(batch.field_embeddings)
        deep = self.deep(flattened)
        return torch.cat((first_order, second_order, deep), dim=-1)


class _MultiHeadInteraction(nn.Module):
    def __init__(
        self,
        input_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout: float,
        scale_attention: bool,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale_attention = scale_attention
        self.query = nn.Linear(input_dim, attention_dim, bias=False)
        self.key = nn.Linear(input_dim, attention_dim, bias=False)
        self.value = nn.Linear(input_dim, attention_dim, bias=False)
        self.residual = (
            nn.Identity()
            if input_dim == attention_dim
            else nn.Linear(input_dim, attention_dim, bias=False)
        )
        for projection in (self.query, self.key, self.value):
            initialize_output_linear(projection)
        if isinstance(self.residual, nn.Linear):
            initialize_output_linear(self.residual)
        self.attention_dropout = nn.Dropout(dropout)
        self.normalization = (
            nn.LayerNorm(attention_dim) if layer_norm else nn.Identity()
        )

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, num_fields, _ = values.shape
        return values.reshape(
            batch_size, num_fields, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(fields))
        key = self._split_heads(self.key(fields))
        value = self._split_heads(self.value(fields))
        scores = query @ key.transpose(-2, -1)
        if self.scale_attention:
            scores = scores / math.sqrt(self.head_dim)
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        attended = weights @ value
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .reshape(fields.shape[0], fields.shape[1], self.attention_dim)
        )
        return torch.relu(self.normalization(attended + self.residual(fields)))


class AutoIntPlusInteraction(InteractionBlock):
    """AutoInt+ with both its attention and parallel DNN branches present."""

    model_key = "autoint_plus"
    display_label = "AutoInt+"

    def __init__(self, spec: EncoderSpec, config: AutoIntPlusConfig) -> None:
        attention_output_dim = spec.num_fields * config.attention_dim
        super().__init__(attention_output_dim + config.dnn_hidden_dims[-1], config)
        layers: list[nn.Module] = []
        current_dim = spec.embedding_dim
        for _ in range(config.attention_layers):
            layers.append(
                _MultiHeadInteraction(
                    input_dim=current_dim,
                    attention_dim=config.attention_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    scale_attention=config.scale_attention,
                    layer_norm=config.layer_norm,
                )
            )
            current_dim = config.attention_dim
        self.attention = nn.Sequential(*layers)
        self.deep = DenseTower(
            spec.flattened_field_dim,
            hidden_dims=config.dnn_hidden_dims,
            dropout=config.dropout,
        )
        self.attention_output_dim = attention_output_dim

    def forward(self, batch: EncodedInteractionBatch) -> torch.Tensor:
        fields = batch.field_embeddings
        attended = self.attention(fields).flatten(start_dim=1)
        deep = self.deep(fields.flatten(start_dim=1))
        return torch.cat((attended, deep), dim=-1)
