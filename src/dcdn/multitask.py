"""Frozen multi-target objective, task constants, and per-target heads.

This module implements the objective declared in
``configs/targets_multitask_v1.yaml``:

.. math::

    \\text{Macro-NLoss} = \\frac{1}{|T_{obs}|} \\sum_{t \\in T_{obs}}
        \\frac{L_t}{K_t}

where ``L_t`` is averaged **only over observed labels of target ``t``**, and
``K_t`` is a constant estimated on the train split and frozen before training:

* binary heads use ``L_t = LogLoss_t`` and ``K_t = H_t``, the entropy of the
  constant predictor;
* regression heads use ``L_t = MSE_t`` and ``K_t = V_t = Var_train(y_t)``.

Three rules are enforced structurally rather than by convention:

1. **Missing labels are never zeros.** Every target carries a boolean mask.
   An observed event indicator of ``0`` is a valid negative label and is
   included; only unobserved entries are excluded.
2. **Task weights are exactly one.** There is no adaptive weighting, learned
   uncertainty weighting, GradNorm, or post-result reweighting. The aggregate
   is an unweighted mean over the targets observed in the batch.
3. **Constants are frozen.** ``TaskConstants`` is immutable and carries the
   split it was estimated on; recomputing it on validation or test raises.

The module is deliberately independent of ``dcdn.training`` so that the
frozen single-target harness can adopt it without a rewrite.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import torch
from torch import nn
from torch.nn import functional

from dcdn.models.common import DenseTower, initialize_output_linear
from dcdn.models.contracts import (
    canonical_config,
    canonical_sha256,
    require_positive_int,
    require_probability,
)

TargetKind = Literal["binary", "regression"]
_ALLOWED_KINDS: tuple[TargetKind, ...] = ("binary", "regression")
_CONSTANT_SPLIT = "train"


class MultiTargetContractError(RuntimeError):
    """A frozen multi-target objective or specification contract was violated."""


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSpec:
    """One head of the frozen multi-target objective."""

    name: str
    kind: TargetKind

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("target name must be a non-empty string")
        if self.name != self.name.strip():
            raise ValueError("target name must not carry surrounding whitespace")
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(f"target kind must be one of {_ALLOWED_KINDS}")

    @property
    def loss_name(self) -> str:
        return (
            "binary_cross_entropy_with_logits"
            if self.kind == "binary"
            else "mean_squared_error"
        )

    @property
    def constant_name(self) -> str:
        if self.kind == "binary":
            return "train_label_entropy"
        return "train_label_variance"


@dataclass(frozen=True)
class MultiTargetSpec:
    """The ordered, frozen set of targets for one dataset."""

    CONTRACT: ClassVar[str] = "dcdn-multi-target-objective-v1"

    dataset_id: str
    targets: tuple[TargetSpec, ...]
    primary_confirmatory_target: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string")
        if type(self.targets) is not tuple or not self.targets:
            raise ValueError("targets must be a non-empty tuple")
        if not all(isinstance(t, TargetSpec) for t in self.targets):
            raise TypeError("targets must contain TargetSpec entries")
        names = [t.name for t in self.targets]
        if len(set(names)) != len(names):
            raise ValueError("target names must be unique")
        if self.primary_confirmatory_target not in names:
            raise ValueError(
                "primary_confirmatory_target must name one of the declared targets"
            )

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.targets)

    @property
    def kinds(self) -> tuple[TargetKind, ...]:
        return tuple(t.kind for t in self.targets)

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown target {name!r}") from exc

    @classmethod
    def from_mapping(
        cls, dataset_id: str, payload: Mapping[str, Any]
    ) -> MultiTargetSpec:
        """Build a spec from the frozen ``targets_multitask_v1.yaml`` shape."""

        entries = payload.get("targets")
        if not isinstance(entries, Sequence) or isinstance(entries, str | bytes):
            raise MultiTargetContractError("targets must be a sequence")
        targets = tuple(
            TargetSpec(name=str(entry["name"]), kind=str(entry["kind"]))  # type: ignore[arg-type]
            for entry in entries
        )
        primary = payload.get("primary_confirmatory_target")
        if not isinstance(primary, str):
            raise MultiTargetContractError(
                "primary_confirmatory_target must be declared as a string"
            )
        return cls(
            dataset_id=dataset_id,
            targets=targets,
            primary_confirmatory_target=primary,
        )

    def contract_state(self) -> dict[str, Any]:
        return canonical_config(
            {
                "contract": self.CONTRACT,
                "dataset_id": self.dataset_id,
                "primary_confirmatory_target": self.primary_confirmatory_target,
                "targets": [
                    {
                        "name": t.name,
                        "kind": t.kind,
                        "loss": t.loss_name,
                        "constant": t.constant_name,
                    }
                    for t in self.targets
                ],
                "aggregation": "unweighted_mean_over_observed_targets",
                "per_target_reduction": "mean_over_observed_labels_only",
                "adaptive_task_weights": False,
                "learned_uncertainty_weights": False,
                "gradnorm": False,
            }
        )

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.contract_state())


# ---------------------------------------------------------------------------
# Frozen task constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskConstants:
    """Per-target normalizers ``K_t``, estimated on train and then frozen."""

    CONTRACT: ClassVar[str] = "dcdn-multi-target-task-constants-v1"

    spec: MultiTargetSpec
    values: tuple[float, ...]
    observed_counts: tuple[int, ...]
    estimated_on_split: str = _CONSTANT_SPLIT

    def __post_init__(self) -> None:
        if len(self.values) != len(self.spec):
            raise ValueError("one constant is required per declared target")
        if len(self.observed_counts) != len(self.spec):
            raise ValueError("one observed count is required per declared target")
        if self.estimated_on_split != _CONSTANT_SPLIT:
            raise MultiTargetContractError(
                "task constants may only be estimated on the train split"
            )
        for target, value in zip(self.spec.targets, self.values, strict=True):
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"task constant for {target.name!r} must be finite and positive"
                )
        for target, count in zip(self.spec.targets, self.observed_counts, strict=True):
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(
                    f"target {target.name!r} has no observed train labels; a constant "
                    "cannot be estimated and the target may not be declared"
                )

    def tensor(
        self, *, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.tensor(self.values, device=device, dtype=dtype)

    def contract_state(self) -> dict[str, Any]:
        return canonical_config(
            {
                "contract": self.CONTRACT,
                "spec_sha256": self.spec.contract_sha256,
                "estimated_on_split": self.estimated_on_split,
                "constants": {
                    target.name: {
                        "kind": target.kind,
                        "constant_name": target.constant_name,
                        # Full float64 repr keeps the hash exact and reproducible.
                        "value": repr(value),
                        "observed_labels": count,
                    }
                    for target, value, count in zip(
                        self.spec.targets,
                        self.values,
                        self.observed_counts,
                        strict=True,
                    )
                },
            }
        )

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.contract_state())


def _validate_label_block(
    spec: MultiTargetSpec,
    labels: torch.Tensor,
    masks: torch.Tensor,
) -> None:
    if not isinstance(labels, torch.Tensor) or not isinstance(masks, torch.Tensor):
        raise TypeError("labels and masks must be tensors")
    if labels.ndim != 2 or labels.shape[1] != len(spec):
        raise ValueError(f"labels must have shape [B, {len(spec)}]")
    if masks.shape != labels.shape:
        raise ValueError("masks must have exactly the shape of labels")
    if masks.dtype != torch.bool:
        raise TypeError("masks must be a boolean tensor; zeros are valid labels")
    if labels.device != masks.device:
        raise ValueError("labels and masks must share a device")
    observed = labels[masks]
    if observed.numel() and not torch.isfinite(observed).all():
        raise ValueError("observed labels must be finite")


def compute_task_constants(
    spec: MultiTargetSpec,
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    split: str = _CONSTANT_SPLIT,
) -> TaskConstants:
    """Estimate ``H_t`` / ``V_t`` from train labels and freeze them.

    Only observed labels contribute.  A binary target whose observed train
    labels are all identical, or a regression target with zero observed
    variance, raises: its normalized loss would be undefined.
    """

    if split != _CONSTANT_SPLIT:
        raise MultiTargetContractError(
            "task constants may only be estimated on "
            f"{_CONSTANT_SPLIT!r}, not {split!r}"
        )
    _validate_label_block(spec, labels, masks)

    values: list[float] = []
    counts: list[int] = []
    for index, target in enumerate(spec.targets):
        column = labels[:, index]
        observed = column[masks[:, index]].to(dtype=torch.float64)
        count = int(observed.numel())
        counts.append(count)
        if count == 0:
            raise MultiTargetContractError(
                f"target {target.name!r} has no observed train labels"
            )
        if target.kind == "binary":
            unique = torch.unique(observed)
            if bool(((unique != 0.0) & (unique != 1.0)).any()):
                raise ValueError(
                    f"binary target {target.name!r} must contain only zero and one"
                )
            rate = float(observed.mean())
            if not 0.0 < rate < 1.0:
                raise MultiTargetContractError(
                    f"binary target {target.name!r} is constant on train; its entropy "
                    "normalizer is zero and the target may not be declared"
                )
            require_probability(f"{target.name}_positive_rate", rate)
            values.append(
                float(-(rate * math.log(rate) + (1.0 - rate) * math.log1p(-rate)))
            )
        else:
            variance = float(observed.var(unbiased=False))
            if not math.isfinite(variance) or variance <= 0.0:
                raise MultiTargetContractError(
                    f"regression target {target.name!r} has zero train variance; its "
                    "normalizer is undefined and the target may not be declared"
                )
            values.append(variance)

    return TaskConstants(
        spec=spec,
        values=tuple(values),
        observed_counts=tuple(counts),
        estimated_on_split=split,
    )


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroNLoss:
    """One evaluation of the frozen aggregate plus its mandatory breakdown."""

    aggregate: torch.Tensor
    per_target: torch.Tensor
    raw_per_target: torch.Tensor
    observed_counts: torch.Tensor

    def as_report(self, spec: MultiTargetSpec) -> dict[str, Any]:
        """A JSON-ready breakdown; the aggregate is never published alone."""

        return {
            "macro_nloss": float(self.aggregate.detach()),
            "per_target_normalized": {
                name: float(value)
                for name, value in zip(
                    spec.names, self.per_target.detach().cpu(), strict=True
                )
            },
            "per_target_raw": {
                name: float(value)
                for name, value in zip(
                    spec.names, self.raw_per_target.detach().cpu(), strict=True
                )
            },
            "observed_labels": {
                name: int(value)
                for name, value in zip(
                    spec.names, self.observed_counts.detach().cpu(), strict=True
                )
            },
        }


def macro_nloss(
    spec: MultiTargetSpec,
    constants: TaskConstants,
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
) -> MacroNLoss:
    """Compute the frozen aggregate and its per-target breakdown.

    Targets with no observed label in this batch contribute nothing and are
    excluded from the mean, so the aggregate never becomes NaN because of a
    sparse target.  Their reported per-target entry is zero with an observed
    count of zero, which the caller must not read as a good loss.
    """

    if constants.spec != spec:
        raise MultiTargetContractError(
            "task constants were frozen for a different spec"
        )
    _validate_label_block(spec, labels, masks)
    if logits.shape != labels.shape:
        raise ValueError("logits must have exactly the shape of labels")
    if logits.device != labels.device:
        raise ValueError("logits and labels must share a device")

    # FP32 for the loss regardless of an enclosing autocast context.
    logits_fp32 = logits.float()
    labels_fp32 = labels.float()
    constant_values = constants.tensor(device=logits.device)

    raw: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    for index, target in enumerate(spec.targets):
        column_mask = masks[:, index]
        count = column_mask.sum()
        counts.append(count)
        selected = column_mask.nonzero(as_tuple=True)[0]
        if selected.numel() == 0:
            raw.append(torch.zeros((), device=logits.device, dtype=torch.float32))
            continue
        head_logits = logits_fp32[selected, index]
        head_labels = labels_fp32[selected, index]
        if target.kind == "binary":
            value = functional.binary_cross_entropy_with_logits(
                head_logits, head_labels, reduction="mean"
            )
        else:
            value = functional.mse_loss(head_logits, head_labels, reduction="mean")
        raw.append(value)

    raw_stack = torch.stack(raw)
    count_stack = torch.stack(counts)
    normalized = raw_stack / constant_values
    contributing = count_stack > 0
    observed_targets = int(contributing.sum())
    if observed_targets == 0:
        raise MultiTargetContractError(
            "no target has an observed label in this batch; the aggregate is undefined"
        )
    # Unweighted mean over the observed targets.  Every task weight is one.
    aggregate = (normalized * contributing).sum() / observed_targets
    return MacroNLoss(
        aggregate=aggregate,
        per_target=normalized * contributing,
        raw_per_target=raw_stack * contributing,
        observed_counts=count_stack,
    )


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class MultiTargetPredictionHead(nn.Module):
    """One shared tower plus a separate identical linear head per target.

    Every row of the comparison matrix uses this module verbatim, so the only
    architectural difference between rows is the interaction block.  Heads are
    kept as distinct ``nn.Linear`` modules rather than one fused
    ``Linear(d, T)`` so that per-target parameters are individually countable
    and per-target head state is individually inspectable.
    """

    CONTRACT: ClassVar[str] = "dcdn-multi-target-prediction-head-v1"

    def __init__(
        self,
        representation_dim: int,
        spec: MultiTargetSpec,
        hidden_dims: tuple[int, ...] = (),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.representation_dim = require_positive_int(
            "representation_dim", representation_dim
        )
        if not isinstance(spec, MultiTargetSpec):
            raise TypeError("spec must be a MultiTargetSpec")
        self.spec = spec
        self.tower = DenseTower(representation_dim, hidden_dims, dropout)
        self.heads = nn.ModuleList(
            nn.Linear(self.tower.output_dim, 1) for _ in range(len(spec))
        )
        for head in self.heads:
            initialize_output_linear(head)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        if representation.ndim != 2:
            raise ValueError("head input must have shape [B, representation_dim]")
        if representation.shape[1] != self.representation_dim:
            raise ValueError("head input width does not match representation_dim")
        shared = self.tower(representation)
        return torch.cat([head(shared) for head in self.heads], dim=1)

    def reset_parameters(self) -> None:
        self.tower.reset_parameters()
        for head in self.heads:
            initialize_output_linear(head)

    def non_embedding_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def contract_state(self) -> dict[str, Any]:
        return canonical_config(
            {
                "contract": self.CONTRACT,
                "representation_dim": self.representation_dim,
                "spec_sha256": self.spec.contract_sha256,
                "target_names": list(self.spec.names),
                "tower_output_dim": self.tower.output_dim,
                "heads": "one_identical_linear_head_per_target",
                "head_initializer": "xavier_uniform_zero_bias",
            }
        )

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.contract_state())


__all__ = [
    "MacroNLoss",
    "MultiTargetContractError",
    "MultiTargetPredictionHead",
    "MultiTargetSpec",
    "TargetKind",
    "TargetSpec",
    "TaskConstants",
    "compute_task_constants",
    "macro_nloss",
]
