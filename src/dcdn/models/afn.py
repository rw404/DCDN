"""Clean-room AFN+ ensemble interaction block.

The logarithm and exponential themselves are always evaluated in FP32. The
task loss is outside this block, so the controlled harness uses one common
loss rather than the auxiliary branch losses in the original TF1 script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    require_finite_positive,
    require_positive_int,
    require_probability,
)


@dataclass(frozen=True)
class AFNPlusConfig:
    logarithmic_neurons: int
    afn_hidden_dims: tuple[int, ...]
    dnn_hidden_dims: tuple[int, ...]
    log_floor: float
    exp_clip: float | None
    dropout: float
    batch_norm: bool

    def __post_init__(self) -> None:
        require_positive_int("logarithmic_neurons", self.logarithmic_neurons)
        require_dims("afn_hidden_dims", self.afn_hidden_dims, allow_empty=False)
        require_dims("dnn_hidden_dims", self.dnn_hidden_dims, allow_empty=False)
        require_finite_positive("log_floor", self.log_floor)
        if self.exp_clip is not None:
            require_finite_positive("exp_clip", self.exp_clip)
        require_probability("dropout", self.dropout)
        if not isinstance(self.batch_norm, bool):
            raise TypeError("batch_norm must be boolean")


class AFNPlusInteraction(InteractionBlock):
    """AFN logarithmic branch ensembled with a distinct DNN branch."""

    model_key = "afn_plus"
    display_label = "AFN+"
    requires_afn_aux = True

    def __init__(self, spec: EncoderSpec, config: AFNPlusConfig) -> None:
        super().__init__(2, config)
        self.num_fields = spec.num_fields
        self.embedding_dim = spec.embedding_dim
        self.coefficients = nn.Linear(
            spec.num_fields, config.logarithmic_neurons, bias=False
        )
        initialize_output_linear(self.coefficients)
        self.log_normalization: nn.Module = (
            nn.BatchNorm1d(spec.num_fields) if config.batch_norm else nn.Identity()
        )
        self.exp_normalization: nn.Module = (
            nn.BatchNorm1d(config.logarithmic_neurons)
            if config.batch_norm
            else nn.Identity()
        )
        self.afn_tower = DenseTower(
            config.logarithmic_neurons * spec.embedding_dim,
            hidden_dims=config.afn_hidden_dims,
            dropout=config.dropout,
        )
        self.afn_output = nn.Linear(self.afn_tower.output_dim, 1)
        initialize_output_linear(self.afn_output)
        self.dnn_tower = DenseTower(
            spec.flattened_field_dim,
            hidden_dims=config.dnn_hidden_dims,
            dropout=config.dropout,
        )
        self.dnn_output = nn.Linear(self.dnn_tower.output_dim, 1)
        initialize_output_linear(self.dnn_output)

    def logarithmic_features(self, fields: torch.Tensor) -> torch.Tensor:
        if fields.ndim != 3 or fields.shape[1:] != (
            self.num_fields,
            self.embedding_dim,
        ):
            raise ValueError("AFN fields violate the configured [B, N, E] shape")

        # Domain treatment: remove sign with abs, then apply a declared
        # positive floor before the FP32 logarithm.
        log_fp32 = torch.log(
            fields.to(dtype=torch.float32).abs().clamp_min(self.config.log_floor)
        )
        branch_dtype = self.coefficients.weight.dtype
        normalized_log = self.log_normalization(log_fp32.to(branch_dtype))
        exponents = self.coefficients(normalized_log.transpose(1, 2)).transpose(1, 2)

        exponent_fp32 = exponents.to(dtype=torch.float32)
        if self.config.exp_clip is not None:
            exponent_fp32 = exponent_fp32.clamp(
                min=-self.config.exp_clip,
                max=self.config.exp_clip,
            )
        powered_fp32 = torch.exp(exponent_fp32)
        normalized_exp = self.exp_normalization(powered_fp32.to(branch_dtype))
        return normalized_exp.flatten(start_dim=1)

    def forward(self, batch: EncodedInteractionBatch) -> torch.Tensor:
        auxiliary = batch.afn_dnn_embeddings
        if auxiliary is None:
            raise ValueError("AFN+ requires its independent DNN embedding view")
        afn_value = self.afn_output(
            self.afn_tower(self.logarithmic_features(batch.field_embeddings))
        )
        dnn_value = self.dnn_output(self.dnn_tower(auxiliary.flatten(start_dim=1)))
        return torch.cat((afn_value, dnn_value), dim=-1)

    def extra_contract_state(self) -> dict[str, Any]:
        exp_policy = (
            "unbounded_source_equation_fail_via_nonfinite_monitor"
            if self.config.exp_clip is None
            else "explicit_symmetric_pre_exp_clamp"
        )
        return {
            "ensemble": True,
            "independent_dnn_embedding_view_required": True,
            "domain": "abs_then_positive_floor",
            "log_compute_dtype": "float32",
            "exp_compute_dtype": "float32",
            "exp_policy": exp_policy,
            "loss_contract": "one_common_task_loss_no_auxiliary_branch_losses",
            "fusion": "visible_linear_representation_adapter_then_common_head",
        }
