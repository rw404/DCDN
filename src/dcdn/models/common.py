"""Shared towers, adapter, head, and deterministic controlled-model wrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn

from dcdn.models.contracts import (
    EncodedInteractionBatch,
    EncoderSpec,
    InteractionRepresentation,
    PredictionOutput,
    canonical_config,
    canonical_sha256,
    require_dims,
    require_nonnegative_int,
    require_positive_int,
    require_probability,
)


def initialize_hidden_linear(layer: nn.Linear) -> None:
    """Kaiming-uniform/ReLU initializer with an explicit zero bias."""

    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def initialize_output_linear(layer: nn.Linear) -> None:
    """Xavier-uniform linear/output initializer with an explicit zero bias."""

    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@dataclass(frozen=True)
class HeadConfig:
    """Architecture shared verbatim by every controlled model."""

    hidden_dims: tuple[int, ...]
    dropout: float
    output_dim: int

    def __post_init__(self) -> None:
        require_dims("hidden_dims", self.hidden_dims, allow_empty=True)
        if type(self.dropout) is not float:
            raise TypeError("dropout must be an exact float")
        require_probability("dropout", self.dropout)
        require_positive_int("output_dim", self.output_dim)


class DenseTower(nn.Module):
    """A small ReLU MLP that preserves batch axes, including ``B=1``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        require_positive_int("input_dim", input_dim)
        require_dims("hidden_dims", hidden_dims, allow_empty=True)
        require_probability("dropout", dropout)
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(current_dim, hidden_dim)
            initialize_hidden_linear(linear)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = current_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError("DenseTower inputs must have shape [B, D]")
        return self.network(inputs)

    def reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                initialize_hidden_linear(module)


class CommonPredictionHead(nn.Module):
    """Identical post-representation prediction architecture for all models."""

    CONTRACT: ClassVar[str] = "dcdn-common-prediction-head-v1"

    def __init__(self, representation_dim: int, config: HeadConfig) -> None:
        super().__init__()
        require_positive_int("representation_dim", representation_dim)
        self.representation_dim = representation_dim
        self.config = config
        self.tower = DenseTower(
            representation_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        self.output = nn.Linear(self.tower.output_dim, config.output_dim)
        initialize_output_linear(self.output)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        if representation.ndim != 2:
            raise ValueError("head input must have shape [B, representation_dim]")
        if representation.shape[1] != self.representation_dim:
            raise ValueError("head input width does not match representation_dim")
        return self.output(self.tower(representation))

    def reset_parameters(self) -> None:
        self.tower.reset_parameters()
        initialize_output_linear(self.output)

    def contract_state(self) -> dict[str, Any]:
        return {
            "contract": self.CONTRACT,
            "representation_dim": self.representation_dim,
            "config": canonical_config(self.config),
            "hidden_initializer": "kaiming_uniform_relu_zero_bias",
            "output_initializer": "xavier_uniform_zero_bias",
        }


class InteractionBlock(nn.Module, ABC):
    """Post-encoder interaction block with an explicit raw output width."""

    model_key: ClassVar[str]
    display_label: ClassVar[str]
    requires_afn_aux: ClassVar[bool] = False

    def __init__(self, raw_output_dim: int, config: Any) -> None:
        super().__init__()
        self.raw_output_dim = require_positive_int("raw_output_dim", raw_output_dim)
        self.config = config

    @abstractmethod
    def forward(self, batch: EncodedInteractionBatch) -> torch.Tensor:
        raise NotImplementedError

    def extra_contract_state(self) -> dict[str, Any]:
        return {}


class RepresentationAdapter(nn.Module):
    """Declared model-specific linear map into the shared head width.

    Raw interaction widths are inherently architecture-specific. Keeping this
    adapter visible makes its parameters/FLOPs countable in comparisons.
    """

    CONTRACT: ClassVar[str] = "linear-representation-adapter-v1"

    def __init__(self, raw_dim: int, representation_dim: int) -> None:
        super().__init__()
        self.raw_dim = require_positive_int("raw_dim", raw_dim)
        self.representation_dim = require_positive_int(
            "representation_dim", representation_dim
        )
        self.projection = nn.Linear(raw_dim, representation_dim)
        initialize_output_linear(self.projection)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.raw_dim:
            raise RuntimeError(
                f"raw interaction output must have shape [B, {self.raw_dim}]"
            )
        return self.projection(values)

    def reset_parameters(self) -> None:
        initialize_output_linear(self.projection)


class ControlledInteractionModel(nn.Module):
    """Interaction block + explicit adapter + one common prediction head."""

    CONTRACT: ClassVar[str] = "controlled-interaction-model-v1"

    def __init__(
        self,
        encoder_spec: EncoderSpec,
        interaction: InteractionBlock,
        head_config: HeadConfig,
        init_seed: int,
    ) -> None:
        super().__init__()
        require_nonnegative_int("init_seed", init_seed)
        self.encoder_spec = encoder_spec
        self.interaction = interaction
        self.head_config = head_config
        self.init_seed = init_seed
        self.adapter = RepresentationAdapter(
            interaction.raw_output_dim, encoder_spec.representation_dim
        )
        self.head = CommonPredictionHead(encoder_spec.representation_dim, head_config)

    @property
    def model_key(self) -> str:
        return self.interaction.model_key

    @property
    def display_label(self) -> str:
        return self.interaction.display_label

    def forward(self, batch: EncodedInteractionBatch) -> PredictionOutput:
        if not isinstance(batch, EncodedInteractionBatch):
            raise TypeError("model input must be EncodedInteractionBatch")
        batch.validate(
            self.encoder_spec,
            require_afn_aux=self.interaction.requires_afn_aux,
        )
        raw_values = self.interaction(batch)
        representation_values = self.adapter(raw_values)
        representation = InteractionRepresentation(
            values=representation_values,
            model_key=self.model_key,
        )
        representation.validate(
            batch_size=batch.batch_size,
            representation_dim=self.encoder_spec.representation_dim,
        )
        logits = self.head(representation.values)
        expected_shape = (batch.batch_size, self.head_config.output_dim)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"prediction head returned {tuple(logits.shape)}, "
                f"expected {expected_shape}"
            )
        return PredictionOutput(representation=representation, logits=logits)

    def contract_state(self) -> dict[str, Any]:
        state = {
            "contract": self.CONTRACT,
            "model_key": self.model_key,
            "display_label": self.display_label,
            "encoder_spec": canonical_config(self.encoder_spec),
            "interaction_config": canonical_config(self.interaction.config),
            "interaction_extra": canonical_config(
                self.interaction.extra_contract_state()
            ),
            "adapter": {
                "contract": self.adapter.CONTRACT,
                "raw_dim": self.adapter.raw_dim,
                "representation_dim": self.adapter.representation_dim,
                "initializer": "xavier_uniform_zero_bias",
            },
            "head": self.head.contract_state(),
            "init_seed": self.init_seed,
        }
        return canonical_config(state)

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.contract_state())
