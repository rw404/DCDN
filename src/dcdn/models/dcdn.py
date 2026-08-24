"""Adapter from the audited DCDN-Core to the controlled model contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dcdn.core import DCDNCore
from dcdn.models.common import InteractionBlock
from dcdn.models.contracts import (
    EncodedInteractionBatch,
    EncoderSpec,
    require_finite_nonnegative,
    require_finite_positive,
    require_positive_int,
)


@dataclass(frozen=True)
class DCDNCoreConfig:
    num_layers: int
    div_eps: float
    product_init_scale: float
    div_init_scale: float
    runtime_value_checks: bool

    def __post_init__(self) -> None:
        require_positive_int("num_layers", self.num_layers)
        require_finite_positive("div_eps", self.div_eps)
        require_finite_nonnegative("product_init_scale", self.product_init_scale)
        require_finite_nonnegative("div_init_scale", self.div_init_scale)
        if not isinstance(self.runtime_value_checks, bool):
            raise TypeError("runtime_value_checks must be boolean")


class DCDNCoreInteraction(InteractionBlock):
    """Transform only P operands; concatenate unchanged C bypass coordinates."""

    model_key = "dcdn_core"

    def __init__(self, spec: EncoderSpec, config: DCDNCoreConfig) -> None:
        super().__init__(spec.flat_dim, config)
        self.core = DCDNCore(
            features_count=spec.operand_dim,
            num_layers=config.num_layers,
            denominator_mask=spec.denominator_mask,
            div_eps=config.div_eps,
            product_init_scale=config.product_init_scale,
            div_init_scale=config.div_init_scale,
            runtime_value_checks=config.runtime_value_checks,
            # DCDNCore sees P only, so its checkpoint must bind P's ordered
            # schema rather than the wider C+P encoder schema.
            feature_schema_sha256=spec.operand_schema_sha256,
        )
        self.display_label = self.core.architecture_label

    def forward(self, batch: EncodedInteractionBatch) -> torch.Tensor:
        operands = self.core(
            batch.operand_features,
            denominator_validity=batch.denominator_validity,
        )
        return torch.cat((batch.bypass_features, operands), dim=-1)

    def extra_contract_state(self) -> dict[str, Any]:
        return {
            "architecture_label": self.core.architecture_label,
            "fixed_anchor": True,
            "crossed_view": "operand_features_P_only",
            "bypass_view": "context_C_unchanged",
            "local_fc_batchnorm_dropout_concat_projector": False,
            "core_serialization": self.core.serialization_metadata(),
        }
