"""Controlled, dataset-agnostic interaction-model blocks."""

from dcdn.models.afn import AFNPlusConfig, AFNPlusInteraction
from dcdn.models.common import HeadConfig
from dcdn.models.contracts import (
    EncodedInteractionBatch,
    EncoderSpec,
    InteractionRepresentation,
    PredictionOutput,
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
from dcdn.models.suite import (
    InteractionConfig,
    ProvenancedControlledModel,
    build_controlled_model,
)
from dcdn.models.wukong import (
    FactorizationMachineBlock,
    LinearCompressionBlock,
    WukongInteraction,
    WukongLayer,
    WukongSmallConfig,
)

__all__ = [
    "AFNPlusConfig",
    "AFNPlusInteraction",
    "AutoIntPlusConfig",
    "AutoIntPlusInteraction",
    "DCDNCoreConfig",
    "EulerNetConfig",
    "EulerNetInteraction",
    "DCDNCoreInteraction",
    "DCNConfig",
    "DCNv2CapacityCoreConfig",
    "DCNv2CapacityCoreInteraction",
    "DCNv2FullRankConfig",
    "DCNv2FullRankInteraction",
    "DeepFMConfig",
    "DeepFMInteraction",
    "EncodedInteractionBatch",
    "EncoderSpec",
    "FactorizationMachineBlock",
    "HeadConfig",
    "InteractionConfig",
    "InteractionRepresentation",
    "LinearCompressionBlock",
    "MLPConfig",
    "MLPInteraction",
    "OriginalDCNInteraction",
    "PredictionOutput",
    "ProvenancedControlledModel",
    "WukongInteraction",
    "WukongLayer",
    "WukongSmallConfig",
    "build_controlled_model",
]
