"""Forward-only multi-target wrapper for the frozen public model suite.

The legacy single-target harness remains byte-compatible.  This module reuses
the exact shared encoder, interaction block, and representation adapter, then
replaces only the scalar prediction head with the frozen multi-target head.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch
from torch import nn

from dcdn.encoder import DatasetEncoderSpec, SharedFeatureEncoder
from dcdn.models.common import HeadConfig
from dcdn.models.contracts import (
    EncodedInteractionBatch,
    EncoderSpec,
    InteractionRepresentation,
    PredictionOutput,
    canonical_config,
    canonical_sha256,
)
from dcdn.models.suite import InteractionConfig, build_controlled_model
from dcdn.multitask import MultiTargetPredictionHead, MultiTargetSpec


class MultiTargetModelContractError(RuntimeError):
    """The shared encoder/interaction/multi-head boundary drifted."""


@runtime_checkable
class MultiTargetRawFeatures(Protocol):
    """Structural raw input boundary; labels and audit IDs are inaccessible."""

    context_tokens: torch.Tensor
    operand_features: torch.Tensor
    denominator_validity: torch.Tensor


def _component_seed(init_seed: int, component: str) -> int:
    payload = f"dcdn-multitarget-model-init-v1:{init_seed}:{component}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


class ProvenancedMultiTargetModel(nn.Module):
    """Controlled interaction + adapter + one shared multi-target head."""

    CONTRACT: ClassVar[str] = "dcdn-controlled-multitarget-model-v1"

    def __init__(
        self,
        *,
        base_model: nn.Module,
        target_spec: MultiTargetSpec,
        head_config: HeadConfig,
        init_seed: int,
        head_seed: int,
    ) -> None:
        super().__init__()
        required = ("encoder_spec", "interaction", "adapter", "contract_state")
        if any(not hasattr(base_model, name) for name in required):
            raise TypeError("base_model must come from the controlled suite factory")
        if head_config.output_dim != len(target_spec):
            raise MultiTargetModelContractError(
                "head output_dim must equal the frozen target count"
            )
        self.encoder_spec: EncoderSpec = base_model.encoder_spec
        self.interaction = base_model.interaction
        self.adapter = base_model.adapter
        self.target_spec = target_spec
        self.head_config = head_config
        self.init_seed = init_seed
        self.head_seed = head_seed
        self._base_contract: Mapping[str, Any] = canonical_config(
            base_model.contract_state()
        )
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.random.default_generator.manual_seed(head_seed)
            self.head = MultiTargetPredictionHead(
                self.encoder_spec.representation_dim,
                target_spec,
                hidden_dims=head_config.hidden_dims,
                dropout=head_config.dropout,
            )

    @property
    def model_key(self) -> str:
        return self.interaction.model_key

    @property
    def display_label(self) -> str:
        return self.interaction.display_label

    def _validate_parameter_contract(self, batch: EncodedInteractionBatch) -> None:
        devices = {parameter.device for parameter in self.parameters()}
        dtypes = {parameter.dtype for parameter in self.parameters()}
        if len(devices) != 1 or len(dtypes) != 1:
            raise MultiTargetModelContractError(
                "all model parameters must share one device and dtype"
            )
        if dtypes != {torch.float32}:
            raise MultiTargetModelContractError(
                "multi-target model parameters must remain FP32"
            )
        if devices != {batch.operand_features.device}:
            raise MultiTargetModelContractError(
                "encoded inputs and model parameters must share one device"
            )

    def forward(self, batch: EncodedInteractionBatch) -> PredictionOutput:
        if not isinstance(batch, EncodedInteractionBatch):
            raise TypeError("model input must be EncodedInteractionBatch")
        batch.validate(
            self.encoder_spec,
            require_afn_aux=self.interaction.requires_afn_aux,
        )
        self._validate_parameter_contract(batch)
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
        expected = (batch.batch_size, len(self.target_spec))
        if tuple(logits.shape) != expected:
            raise MultiTargetModelContractError(
                f"multi-target head returned {tuple(logits.shape)}, expected {expected}"
            )
        return PredictionOutput(representation=representation, logits=logits)

    def contract_state(self) -> dict[str, Any]:
        base = dict(self._base_contract)
        base.pop("head", None)
        return canonical_config(
            {
                "contract": self.CONTRACT,
                "model_key": self.model_key,
                "display_label": self.display_label,
                "base_interaction_adapter_contract": base,
                "target_spec_sha256": self.target_spec.contract_sha256,
                "target_names": list(self.target_spec.names),
                "head": self.head.contract_state(),
                "head_config": self.head_config,
                "initialization": {
                    "base_seed": self.init_seed,
                    "head_seed": self.head_seed,
                    "head_seed_domain": "dcdn-multitarget-model-init-v1",
                    "cuda_rng_touched": False,
                },
            }
        )

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.contract_state())

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "contract_sha256": self.contract_sha256,
            "contract_state": self.contract_state(),
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise MultiTargetModelContractError(
                "checkpoint multi-target metadata differs from the live contract"
            )


class MultiTargetTrainableEncodedModel(nn.Module):
    """One optimizer/checkpoint boundary over encoder, interaction, and heads."""

    CONTRACT: ClassVar[str] = "dcdn-raw-to-multitarget-model-v1"

    def __init__(
        self,
        *,
        encoder: SharedFeatureEncoder,
        controlled_model: ProvenancedMultiTargetModel,
    ) -> None:
        super().__init__()
        if encoder.model_spec != controlled_model.encoder_spec:
            raise MultiTargetModelContractError(
                "encoder output and controlled model input specs differ"
            )
        requires_aux = controlled_model.interaction.requires_afn_aux
        if encoder.include_afn_secondary != requires_aux:
            raise MultiTargetModelContractError(
                "AFN auxiliary encoder ownership differs from the model family"
            )
        if (
            encoder.spec.allowed_model_keys
            and controlled_model.model_key not in encoder.spec.allowed_model_keys
        ):
            raise MultiTargetModelContractError(
                "model family is forbidden for this frozen feature regime"
            )
        self.encoder = encoder
        self.controlled_model = controlled_model

    @property
    def model_key(self) -> str:
        return self.controlled_model.model_key

    @property
    def display_label(self) -> str:
        return self.controlled_model.display_label

    @property
    def encoder_spec(self) -> EncoderSpec:
        return self.controlled_model.encoder_spec

    @property
    def interaction(self) -> nn.Module:
        return self.controlled_model.interaction

    @property
    def head(self) -> nn.Module:
        return self.controlled_model.head

    @property
    def target_spec(self) -> MultiTargetSpec:
        return self.controlled_model.target_spec

    def encode_raw_batch(
        self, batch: MultiTargetRawFeatures
    ) -> EncodedInteractionBatch:
        if not isinstance(batch, MultiTargetRawFeatures):
            raise TypeError("raw input violates the multi-target feature protocol")
        return self.encoder(
            batch.context_tokens,
            batch.operand_features,
            batch.denominator_validity,
        )

    def forward(self, batch: MultiTargetRawFeatures) -> PredictionOutput:
        return self.controlled_model(self.encode_raw_batch(batch))

    def contract_state(self) -> dict[str, Any]:
        return canonical_config(
            {
                "contract": self.CONTRACT,
                "encoder_contract_sha256": self.encoder.contract_sha256,
                "encoder_contract": self.encoder.contract_state(),
                "controlled_model_contract_sha256": (
                    self.controlled_model.contract_sha256
                ),
                "controlled_model_contract": self.controlled_model.contract_state(),
                "checkpoint_scope": "encoder_interaction_adapter_shared_tower_heads",
                "optimizer_scope": "all_trainable_parameters_in_this_module",
            }
        )

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.contract_state())

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "contract_sha256": self.contract_sha256,
            "contract_state": self.contract_state(),
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise MultiTargetModelContractError(
                "joint multi-target checkpoint metadata differs from the live contract"
            )


def build_multitarget_encoded_model(
    *,
    encoder_spec: DatasetEncoderSpec,
    interaction_config: InteractionConfig,
    target_spec: MultiTargetSpec,
    hidden_dims: tuple[int, ...],
    dropout: float,
    run_seed: int,
    runtime_value_checks: bool,
) -> MultiTargetTrainableEncodedModel:
    """Construct the joint model on CPU without touching CUDA RNG state."""

    head_config = HeadConfig(
        hidden_dims=hidden_dims,
        dropout=dropout,
        output_dim=len(target_spec),
    )
    base = build_controlled_model(
        encoder_spec=encoder_spec.model_spec,
        interaction_config=interaction_config,
        head_config=head_config,
        init_seed=run_seed,
    )
    controlled = ProvenancedMultiTargetModel(
        base_model=base,
        target_spec=target_spec,
        head_config=head_config,
        init_seed=run_seed,
        head_seed=_component_seed(run_seed, "head"),
    )
    encoder = SharedFeatureEncoder(
        encoder_spec,
        run_seed=run_seed,
        include_afn_secondary=controlled.interaction.requires_afn_aux,
        runtime_value_checks=runtime_value_checks,
    )
    return MultiTargetTrainableEncodedModel(
        encoder=encoder,
        controlled_model=controlled,
    )


__all__ = [
    "MultiTargetModelContractError",
    "MultiTargetRawFeatures",
    "MultiTargetTrainableEncodedModel",
    "ProvenancedMultiTargetModel",
    "build_multitarget_encoded_model",
]
