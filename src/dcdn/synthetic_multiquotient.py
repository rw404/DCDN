"""``Synthetic-MultiQuotient-3`` — the frozen multi-target quotient benchmark.

This replaces the legacy MNIST-like single-target division notebooks as the
implementation-capability benchmark.  It answers RQ1 only:

    *Can DCDN learn quotients directly from the operands, without a
    ready-made ratio feature?*

It is **evidence about the implementation**, not evidence of recommendation
superiority.  The legacy notebooks are retained as follow-up material and never
substitute for this benchmark.

Task
----
Positive raw inputs ``[n1, d1, n2, d2, d3, z1, z2, z3]`` and three targets::

    y1 = n1 / d1
    y2 = n2 / d2
    y3 = (n1 + n2) / d3

``y3`` is the load-bearing head: its numerator is not an input coordinate, so
the model must assemble it inside the division branch.  ``z1..z3`` are positive
distractors that appear in no target.

The reciprocal mask is active on exactly ``d1, d2, d3``.  All denominators are
generated strictly above the reciprocal floor, so a floor activation during
evaluation is a defect signal rather than expected behaviour.

Slices
------
``id``               operands and ratios drawn from the training distribution
``ood_common_scale`` the ``id`` rows with every operand multiplied by a common
                     factor.  The *targets are unchanged row for row*, so a
                     model that learned the quotient is invariant and a
                     piecewise-linear approximator is not
``ood_low_ratio``    denominators shifted up, moving the quotient below the
                     training range
``ood_high_ratio``   denominators shifted down, moving the quotient above it

Every slice is materialized and hashed before any training step.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import torch

from dcdn.multitask import MultiTargetSpec, TargetSpec

SliceName = Literal[
    "train", "validation", "id", "ood_common_scale", "ood_low_ratio", "ood_high_ratio"
]

COLUMNS: tuple[str, ...] = ("n1", "d1", "n2", "d2", "d3", "z1", "z2", "z3")
DENOMINATOR_COLUMNS: tuple[str, ...] = ("d1", "d2", "d3")
DENOMINATOR_MASK: tuple[bool, ...] = tuple(c in DENOMINATOR_COLUMNS for c in COLUMNS)
TARGET_NAMES: tuple[str, ...] = ("y1", "y2", "y3")
TARGET_DEFINITIONS: dict[str, str] = {
    "y1": "n1 / d1",
    "y2": "n2 / d2",
    "y3": "(n1 + n2) / d3",
}

#: Denominators never fall below this value, which sits well above the
#: ``div_eps`` reciprocal floor used by ``DCDNCore``.
MINIMUM_DENOMINATOR = 0.2

SPEC = MultiTargetSpec(
    dataset_id="synthetic_multiquotient_3",
    targets=tuple(TargetSpec(name=name, kind="regression") for name in TARGET_NAMES),
    primary_confirmatory_target="y1",
)


class SyntheticContractError(RuntimeError):
    """A frozen synthetic-benchmark contract was violated."""


@dataclass(frozen=True)
class SliceConfig:
    """Generation parameters for one materialized slice."""

    name: str
    rows: int
    seed: int
    scale: float
    denominator_low: float
    denominator_span: float
    numerator_low: float = 1.0
    numerator_span: float = 9.0

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("rows must be positive")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")
        if self.denominator_low < MINIMUM_DENOMINATOR:
            raise SyntheticContractError(
                f"slice {self.name!r} would place denominators below the declared "
                f"minimum {MINIMUM_DENOMINATOR}"
            )
        if self.denominator_span <= 0.0 or self.numerator_span <= 0.0:
            raise ValueError("spans must be positive")
        if self.numerator_low <= 0.0:
            raise ValueError("numerators must be strictly positive")


#: The frozen slice grid.  ``ood_common_scale`` multiplies every operand by ten
#: while leaving the ratios untouched; the remaining OOD slices move the ratio
#: itself without changing the operand scale.
FROZEN_SLICES: tuple[SliceConfig, ...] = (
    SliceConfig(
        "train",
        rows=200_000,
        seed=20260820,
        scale=1.0,
        denominator_low=1.0,
        denominator_span=9.0,
    ),
    SliceConfig(
        "validation",
        rows=40_000,
        seed=20260821,
        scale=1.0,
        denominator_low=1.0,
        denominator_span=9.0,
    ),
    SliceConfig(
        "id",
        rows=40_000,
        seed=20260822,
        scale=1.0,
        denominator_low=1.0,
        denominator_span=9.0,
    ),
    # Deliberately the SAME seed as ``id``: only ``scale`` differs, so row i of
    # this slice and row i of ``id`` carry bitwise-identical targets.  That
    # makes scale consistency measurable per row rather than only in
    # distribution.
    SliceConfig(
        "ood_common_scale",
        rows=40_000,
        seed=20260822,
        scale=10.0,
        denominator_low=1.0,
        denominator_span=9.0,
    ),
    SliceConfig(
        "ood_low_ratio",
        rows=40_000,
        seed=20260824,
        scale=1.0,
        denominator_low=20.0,
        denominator_span=40.0,
    ),
    SliceConfig(
        "ood_high_ratio",
        rows=40_000,
        seed=20260825,
        scale=1.0,
        denominator_low=0.2,
        denominator_span=0.4,
    ),
)


@dataclass(frozen=True)
class MaterializedSlice:
    """One generated slice with its pre-training hash."""

    config: SliceConfig
    features: torch.Tensor
    targets: torch.Tensor
    masks: torch.Tensor
    sha256: str

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])


def generate_slice(config: SliceConfig) -> MaterializedSlice:
    """Materialize one slice deterministically from its frozen seed.

    Generation is CPU-only and float64-free: the tensors are float32, which is
    the dtype every model consumes, so the hash covers exactly the bytes the
    models see.
    """

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    numerators = (
        torch.rand(config.rows, 2, generator=generator) * config.numerator_span
        + config.numerator_low
    )
    denominators = (
        torch.rand(config.rows, 3, generator=generator) * config.denominator_span
        + config.denominator_low
    )
    distractors = (
        torch.rand(config.rows, 3, generator=generator) * config.numerator_span
        + config.numerator_low
    )

    n1 = numerators[:, 0:1]
    n2 = numerators[:, 1:2]
    d1 = denominators[:, 0:1]
    d2 = denominators[:, 1:2]
    d3 = denominators[:, 2:3]

    # Ratios are computed BEFORE the common scale is applied.  A common scale
    # cancels exactly in a quotient, so the OOD-common-scale slice keeps the
    # same target values with different operand magnitudes.
    targets = torch.cat([n1 / d1, n2 / d2, (n1 + n2) / d3], dim=1).float()
    features = (
        torch.cat([n1, d1, n2, d2, d3, distractors], dim=1) * config.scale
    ).float()

    finite = bool(torch.isfinite(features).all()) and bool(
        torch.isfinite(targets).all()
    )
    if not finite:
        raise SyntheticContractError(
            f"slice {config.name!r} produced non-finite values"
        )
    if bool((features <= 0).any()):
        raise SyntheticContractError(
            f"slice {config.name!r} produced a non-positive input"
        )
    active = features[:, [COLUMNS.index(c) for c in DENOMINATOR_COLUMNS]]
    if float(active.min()) < MINIMUM_DENOMINATOR * config.scale:
        raise SyntheticContractError(
            f"slice {config.name!r} placed a denominator below the declared minimum"
        )

    masks = torch.ones_like(targets, dtype=torch.bool)
    digest = hashlib.sha256()
    digest.update(features.numpy().tobytes())
    digest.update(targets.numpy().tobytes())
    return MaterializedSlice(
        config=config,
        features=features,
        targets=targets,
        masks=masks,
        sha256=digest.hexdigest(),
    )


@dataclass(frozen=True)
class Benchmark:
    """The full materialized benchmark plus its manifest."""

    CONTRACT: ClassVar[str] = "dcdn-synthetic-multiquotient-3-v1"

    slices: dict[str, MaterializedSlice]

    def __getitem__(self, name: str) -> MaterializedSlice:
        try:
            return self.slices[name]
        except KeyError as exc:
            raise KeyError(f"unknown slice {name!r}") from exc

    @property
    def evaluation_slice_names(self) -> tuple[str, ...]:
        return ("id", "ood_common_scale", "ood_low_ratio", "ood_high_ratio")

    def manifest(self) -> dict[str, Any]:
        return {
            "contract": self.CONTRACT,
            "columns": list(COLUMNS),
            "denominator_columns": list(DENOMINATOR_COLUMNS),
            "denominator_mask": list(DENOMINATOR_MASK),
            "minimum_denominator": MINIMUM_DENOMINATOR,
            "targets": [
                {
                    "name": name,
                    "kind": "regression",
                    "definition": TARGET_DEFINITIONS[name],
                }
                for name in TARGET_NAMES
            ],
            "primary_confirmatory_target": SPEC.primary_confirmatory_target,
            "objective_spec_sha256": SPEC.contract_sha256,
            "evaluation_slices": list(self.evaluation_slice_names),
            "slices": {
                name: {
                    "rows": item.rows,
                    "seed": item.config.seed,
                    "scale": item.config.scale,
                    "denominator_low": item.config.denominator_low,
                    "denominator_span": item.config.denominator_span,
                    "sha256": item.sha256,
                }
                for name, item in sorted(self.slices.items())
            },
        }

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_benchmark(configs: tuple[SliceConfig, ...] = FROZEN_SLICES) -> Benchmark:
    """Materialize and hash every slice before any training step occurs."""

    slices = {config.name: generate_slice(config) for config in configs}
    missing = {"train", "validation"} - set(slices)
    if missing:
        raise SyntheticContractError(f"missing required slices: {sorted(missing)}")
    return Benchmark(slices=slices)


def analytical_oracle(features: torch.Tensor) -> torch.Tensor:
    """The identity oracle: compute the three targets in closed form.

    This is the upper reference row of the benchmark.  Its normalized loss is
    zero up to float32 rounding; any model exceeding it has not learned the
    identity.
    """

    if features.ndim != 2 or features.shape[1] != len(COLUMNS):
        raise ValueError(f"features must have shape [B, {len(COLUMNS)}]")
    n1 = features[:, COLUMNS.index("n1")]
    d1 = features[:, COLUMNS.index("d1")]
    n2 = features[:, COLUMNS.index("n2")]
    d2 = features[:, COLUMNS.index("d2")]
    d3 = features[:, COLUMNS.index("d3")]
    return torch.stack([n1 / d1, n2 / d2, (n1 + n2) / d3], dim=1)


def explicit_ratio_features(features: torch.Tensor) -> torch.Tensor:
    """Append the three target ratios — the positive-control input.

    Only the DCNv2 positive-control arm consumes this view.  It exists to show
    what a model achieves when the quotient is handed to it, which bounds what
    learning the quotient can be worth.
    """

    return torch.cat([features, analytical_oracle(features)], dim=1)


__all__ = [
    "COLUMNS",
    "DENOMINATOR_COLUMNS",
    "DENOMINATOR_MASK",
    "FROZEN_SLICES",
    "MINIMUM_DENOMINATOR",
    "SPEC",
    "TARGET_DEFINITIONS",
    "TARGET_NAMES",
    "Benchmark",
    "MaterializedSlice",
    "SliceConfig",
    "SyntheticContractError",
    "analytical_oracle",
    "build_benchmark",
    "explicit_ratio_features",
    "generate_slice",
]
