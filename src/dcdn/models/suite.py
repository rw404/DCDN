"""Deterministic factory for the controlled interaction-model suite."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from dcdn.models.afn import AFNPlusConfig, AFNPlusInteraction
from dcdn.models.common import (
    ControlledInteractionModel,
    HeadConfig,
    InteractionBlock,
)
from dcdn.models.contracts import (
    EncodedInteractionBatch,
    EncoderSpec,
    PredictionOutput,
    canonical_config,
    require_nonnegative_int,
)
from dcdn.models.cross import (
    DCNConfig,
    DCNv2CapacityCoreConfig,
    DCNv2CapacityCoreInteraction,
    DCNv2FullRankConfig,
    DCNv2FullRankInteraction,
    OriginalDCNInteraction,
)
from dcdn.models.dcdn import DCDNCoreConfig, DCDNCoreInteraction
from dcdn.models.euler import EulerNetConfig, EulerNetInteraction
from dcdn.models.field import (
    AutoIntPlusConfig,
    AutoIntPlusInteraction,
    DeepFMConfig,
    DeepFMInteraction,
    MLPConfig,
    MLPInteraction,
)
from dcdn.models.provenance import source_provenance
from dcdn.models.wukong import WukongInteraction, WukongSmallConfig

type InteractionConfig = (
    MLPConfig
    | DeepFMConfig
    | DCNConfig
    | AutoIntPlusConfig
    | DCNv2FullRankConfig
    | DCNv2CapacityCoreConfig
    | AFNPlusConfig
    | WukongSmallConfig
    | DCDNCoreConfig
)


def _build_interaction(
    spec: EncoderSpec, config: InteractionConfig
) -> InteractionBlock:
    if isinstance(config, MLPConfig):
        return MLPInteraction(spec, config)
    if isinstance(config, DeepFMConfig):
        return DeepFMInteraction(spec, config)
    if isinstance(config, DCNConfig):
        return OriginalDCNInteraction(spec, config)
    if isinstance(config, AutoIntPlusConfig):
        return AutoIntPlusInteraction(spec, config)
    if isinstance(config, DCNv2FullRankConfig):
        return DCNv2FullRankInteraction(spec, config)
    if isinstance(config, DCNv2CapacityCoreConfig):
        return DCNv2CapacityCoreInteraction(spec, config)
    if isinstance(config, AFNPlusConfig):
        return AFNPlusInteraction(spec, config)
    if isinstance(config, WukongSmallConfig):
        return WukongInteraction(spec, config)
    if isinstance(config, EulerNetConfig):
        return EulerNetInteraction(spec, config)
    if isinstance(config, DCDNCoreConfig):
        return DCDNCoreInteraction(spec, config)
    raise TypeError(f"unsupported interaction config: {type(config).__name__}")


def _autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        return bool(torch.is_autocast_enabled())


def _component_seed(init_seed: int, component: str) -> int:
    payload = f"dcdn-model-init-v1:{init_seed}:{component}".encode("ascii")
    # Torch's CPU generator accepts signed-63-bit positive seeds portably.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


class ProvenancedControlledModel(ControlledInteractionModel):
    """Factory-owned wrapper with input/parameter and provenance checks."""

    def __init__(
        self,
        encoder_spec: EncoderSpec,
        interaction: InteractionBlock,
        head_config: HeadConfig,
        init_seed: int,
        provenance: dict[str, Any],
        component_seeds: dict[str, int],
    ) -> None:
        super().__init__(
            encoder_spec=encoder_spec,
            interaction=interaction,
            head_config=head_config,
            init_seed=init_seed,
        )
        self._source_provenance = canonical_config(provenance)
        self._component_seeds = dict(component_seeds)

    def _validate_parameter_contract(self, batch: EncodedInteractionBatch) -> None:
        parameter_devices = {parameter.device for parameter in self.parameters()}
        parameter_dtypes = {parameter.dtype for parameter in self.parameters()}
        if len(parameter_devices) != 1 or len(parameter_dtypes) != 1:
            raise RuntimeError("all model parameters must share one device and dtype")
        parameter_device = next(iter(parameter_devices))
        parameter_dtype = next(iter(parameter_dtypes))
        if parameter_dtype == torch.float16:
            raise TypeError("FP16 parameters are forbidden for the model suite")
        if parameter_dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("model parameters must use FP32 or BF16")
        input_device = batch.operand_features.device
        input_dtype = batch.operand_features.dtype
        if input_device != parameter_device:
            raise ValueError("encoded inputs and model parameters must share a device")
        if input_dtype != parameter_dtype and not _autocast_enabled(input_device.type):
            raise TypeError(
                "encoded input and parameter dtypes must match outside autocast"
            )

    def forward(self, batch: EncodedInteractionBatch) -> PredictionOutput:
        if not isinstance(batch, EncodedInteractionBatch):
            raise TypeError("model input must be EncodedInteractionBatch")
        batch.validate(
            self.encoder_spec,
            require_afn_aux=self.interaction.requires_afn_aux,
        )
        self._validate_parameter_contract(batch)
        return super().forward(batch)

    def contract_state(self) -> dict[str, Any]:
        state = super().contract_state()
        state["source_provenance"] = self._source_provenance
        state["initialization"] = {
            "policy": "independent_hash_derived_cpu_component_seeds_v1",
            "base_seed": self.init_seed,
            "component_seeds": self._component_seeds,
            "head_weights_identical_across_model_variants_for_same_base_seed": True,
            "adapter_seed_shared_but_weight_shape_model_specific": True,
            "cuda_rng_touched": False,
            "construction_device": "cpu",
        }
        return canonical_config(state)


def build_controlled_model(
    *,
    encoder_spec: EncoderSpec,
    interaction_config: InteractionConfig,
    head_config: HeadConfig,
    init_seed: int,
) -> ProvenancedControlledModel:
    """Construct final parameters under isolated component-specific CPU RNGs.

    Model construction is CPU-only. The caller may move the completed model to
    one GPU afterwards. No CUDA generator is read or changed here.
    """

    require_nonnegative_int("init_seed", init_seed)
    component_seeds = {
        name: _component_seed(init_seed, name)
        for name in ("interaction", "adapter", "head")
    }
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.default_generator.manual_seed(component_seeds["interaction"])
        interaction = _build_interaction(encoder_spec, interaction_config)

        # Wrapper construction consumes transient random values. Both visible
        # shared components are reset below under independent recorded seeds.
        torch.random.default_generator.manual_seed(component_seeds["adapter"])
        model = ProvenancedControlledModel(
            encoder_spec=encoder_spec,
            interaction=interaction,
            head_config=head_config,
            init_seed=init_seed,
            provenance=source_provenance(interaction.model_key),
            component_seeds=component_seeds,
        )
        torch.random.default_generator.manual_seed(component_seeds["adapter"])
        model.adapter.reset_parameters()
        torch.random.default_generator.manual_seed(component_seeds["head"])
        model.head.reset_parameters()
    return model
