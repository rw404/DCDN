"""Multi-target capacity and FLOP accounting.

The single-target rule in ``dcdn.capacity_artifact`` matches a control to
``DCDN-Core`` on trainable non-embedding parameters *and* FLOPs/example, within
±5% on both simultaneously, using a mechanical distance in
(parameter, FLOP) space rather than validation quality.

This module keeps exactly that arithmetic contract and only generalizes the
shared part: the adapter is followed by a shared tower and by ``T`` identical
linear heads instead of one.  The interaction formulas are imported verbatim
from ``dcdn.capacity_artifact`` so the two paths can never drift.

Counting conventions, unchanged from the frozen contract:

* a biased linear layer of shape ``(in, out)`` costs ``2 * in * out`` FLOPs;
* an unbiased linear layer costs ``out * (2 * in - 1)`` FLOPs;
* elementwise activations and products cost one FLOP per output element.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dcdn.capacity_artifact import (
    _candidate_counts,
    _dcdn_counts,
    _ordinary_counts,
)

CAPACITY_FLOP_CONTRACT = "dcdn-forward-floating-arithmetic-flops-v1"
DEFAULT_TOLERANCE = 0.05


class MultiTargetCapacityError(RuntimeError):
    """A multi-target capacity-matching contract was violated."""


def common_counts(
    flat_width: int,
    representation_width: int,
    head_hidden_dims: Sequence[int],
    num_targets: int,
) -> dict[str, int]:
    """Parameters and FLOPs of the parts every row shares verbatim.

    ``T`` identical ``Linear(d, 1)`` heads and one fused ``Linear(d, T)`` have
    the same parameter count ``T * (d + 1)`` and the same FLOP count, so this
    accounting is valid for either realization.
    """

    if num_targets <= 0:
        raise ValueError("num_targets must be positive")
    adapter_parameters = (flat_width + 1) * representation_width
    adapter_flops = 2 * flat_width * representation_width

    tower_parameters = 0
    tower_flops = 0
    width = representation_width
    for hidden in head_hidden_dims:
        if hidden <= 0:
            raise ValueError("head hidden dims must be positive")
        tower_parameters += (width + 1) * hidden
        # Linear plus one elementwise ReLU per output unit.
        tower_flops += 2 * width * hidden + hidden
        width = hidden

    head_parameters = (width + 1) * num_targets
    head_flops = 2 * width * num_targets
    return {
        "adapter_parameters": adapter_parameters,
        "adapter_flops": adapter_flops,
        "tower_parameters": tower_parameters,
        "tower_flops": tower_flops,
        # The frozen interaction formulas read these two keys; they must carry
        # the whole post-adapter cost so nothing is dropped from the match.
        "head_parameters": tower_parameters + head_parameters,
        "head_flops": tower_flops + head_flops,
        "linear_head_parameters": head_parameters,
        "linear_head_flops": head_flops,
        "num_targets": num_targets,
    }


def dcdn_counts(
    operand_width: int, num_layers: int, common: Mapping[str, int]
) -> dict[str, Any]:
    """DCDN-Core counts with a multi-target shared tower and heads."""

    counts = dict(_dcdn_counts(operand_width, num_layers, common))
    counts["architecture"] = f"DCDN-Core-L{num_layers}"
    counts["num_layers"] = num_layers
    return counts


def ordinary_dcnv2_counts(
    flat_width: int, num_layers: int, common: Mapping[str, int]
) -> dict[str, Any]:
    """Full-rank DCNv2 counts with a multi-target shared tower and heads."""

    counts = dict(_ordinary_counts(flat_width, num_layers, common))
    counts["architecture"] = f"DCNv2-full-rank-L{num_layers}"
    counts["num_layers"] = num_layers
    return counts


def capacity_candidate_counts(
    operand_width: int,
    rank: int,
    num_layers: int,
    num_experts: int,
    common: Mapping[str, int],
) -> dict[str, Any]:
    """Low-rank mixture DCNv2 counts for one candidate control."""

    return dict(_candidate_counts(operand_width, rank, num_layers, num_experts, common))


def ratios(counts: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    parameter_ratio = counts["parameters"] / target["parameters"]
    flop_ratio = counts["flops"] / target["flops"]
    return {
        "parameter_ratio": parameter_ratio,
        "flop_ratio": flop_ratio,
        "within_tolerance": (
            1.0 - DEFAULT_TOLERANCE <= parameter_ratio <= 1.0 + DEFAULT_TOLERANCE
            and 1.0 - DEFAULT_TOLERANCE <= flop_ratio <= 1.0 + DEFAULT_TOLERANCE
        ),
        "distance": abs(math.log(parameter_ratio)) + abs(math.log(flop_ratio)),
    }


@dataclass(frozen=True)
class CapacityResolution:
    """The mechanically selected control, plus its bracket when no match exists."""

    target: dict[str, Any]
    selected: dict[str, Any] | None
    selected_ratios: dict[str, Any] | None
    lower_bracket: dict[str, Any] | None
    upper_bracket: dict[str, Any] | None
    tolerance: float
    selection_rule: str
    considered: int

    @property
    def matched(self) -> bool:
        return self.selected is not None

    def payload(self) -> dict[str, Any]:
        return {
            "flop_contract": CAPACITY_FLOP_CONTRACT,
            "tolerance_fraction": self.tolerance,
            "selection_rule": self.selection_rule,
            "selection_used_validation_quality": False,
            "candidates_considered": self.considered,
            "target": self.target,
            "matched": self.matched,
            "selected": self.selected,
            "selected_ratios": self.selected_ratios,
            "lower_bracket": self.lower_bracket,
            "upper_bracket": self.upper_bracket,
        }


def resolve_capacity_control(
    *,
    target: Mapping[str, Any],
    operand_width: int,
    common: Mapping[str, int],
    ranks: Sequence[int],
    layer_options: Sequence[int],
    expert_options: Sequence[int],
    tolerance: float = DEFAULT_TOLERANCE,
) -> CapacityResolution:
    """Pick the control by distance in (parameter, FLOP) space only.

    No validation metric is read.  When nothing lands inside the tolerance on
    *both* quantities, the two nearest candidates on either side of the target
    parameter count are returned as an explicit bracket, which the caller must
    report instead of a single matched row.
    """

    scored: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for layers in layer_options:
        for experts in expert_options:
            for rank in ranks:
                counts = capacity_candidate_counts(
                    operand_width, rank, layers, experts, common
                )
                scored.append((counts, ratios(counts, target)))
    if not scored:
        raise MultiTargetCapacityError("no capacity candidates were enumerated")

    eligible = [
        (counts, ratio)
        for counts, ratio in scored
        if 1.0 - tolerance <= ratio["parameter_ratio"] <= 1.0 + tolerance
        and 1.0 - tolerance <= ratio["flop_ratio"] <= 1.0 + tolerance
    ]
    if eligible:
        counts, ratio = min(
            eligible,
            # Deterministic tie-break on the full candidate identity.
            key=lambda item: (
                item[1]["distance"],
                item[0]["num_layers"],
                item[0]["num_experts"],
                item[0]["low_rank"],
            ),
        )
        return CapacityResolution(
            target=dict(target),
            selected=counts,
            selected_ratios=ratio,
            lower_bracket=None,
            upper_bracket=None,
            tolerance=tolerance,
            selection_rule="minimum_log_distance_in_parameter_and_flop_space",
            considered=len(scored),
        )

    below = [item for item in scored if item[0]["parameters"] <= target["parameters"]]
    above = [item for item in scored if item[0]["parameters"] > target["parameters"]]
    lower = max(below, key=lambda item: item[0]["parameters"]) if below else None
    upper = min(above, key=lambda item: item[0]["parameters"]) if above else None
    return CapacityResolution(
        target=dict(target),
        selected=None,
        selected_ratios=None,
        lower_bracket=({**lower[0], "ratios": lower[1]} if lower is not None else None),
        upper_bracket=({**upper[0], "ratios": upper[1]} if upper is not None else None),
        tolerance=tolerance,
        selection_rule="no_candidate_inside_tolerance_report_two_bracketing_controls",
        considered=len(scored),
    )


__all__ = [
    "CAPACITY_FLOP_CONTRACT",
    "DEFAULT_TOLERANCE",
    "CapacityResolution",
    "MultiTargetCapacityError",
    "capacity_candidate_counts",
    "common_counts",
    "dcdn_counts",
    "ordinary_dcnv2_counts",
    "ratios",
    "resolve_capacity_control",
]
