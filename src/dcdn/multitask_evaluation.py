"""Frozen multi-target reporting with masked per-head metrics and user-GAUC."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from dcdn.metrics import (
    BinaryTargetMetrics,
    GroupConcordance,
    WatchTimeMetrics,
    group_concordance,
    masked_binary_target_metrics,
    watch_time_metrics,
)
from dcdn.multitask import MultiTargetSpec, TaskConstants, macro_nloss


@dataclass(frozen=True)
class RegressionTargetMetrics:
    metrics: WatchTimeMetrics
    user_gauc: GroupConcordance
    session_gauc: GroupConcordance | None
    session_gauc_availability: str
    observed_rows: int
    total_rows: int
    observed_row_coverage: float


def _regression_as_report(metrics: RegressionTargetMetrics) -> dict[str, Any]:
    payload = asdict(metrics)
    pointwise = payload["metrics"]
    pointwise["global_xauc"] = pointwise.pop("concordance")
    return payload


@dataclass(frozen=True)
class MultiTargetEvaluation:
    """Complete mandatory report; the aggregate is never emitted alone."""

    macro_nloss: float
    per_target_normalized: tuple[tuple[str, float], ...]
    per_target_raw: tuple[tuple[str, float], ...]
    observed_labels: tuple[tuple[str, int], ...]
    binary: tuple[tuple[str, BinaryTargetMetrics], ...]
    regression: tuple[tuple[str, RegressionTargetMetrics], ...]

    def as_report(self) -> dict[str, Any]:
        return {
            "macro_nloss": self.macro_nloss,
            "per_target_normalized": dict(self.per_target_normalized),
            "per_target_raw": dict(self.per_target_raw),
            "observed_labels": dict(self.observed_labels),
            "per_target_binary": {
                name: asdict(metrics) for name, metrics in self.binary
            },
            "per_target_regression": {
                name: _regression_as_report(metrics)
                for name, metrics in self.regression
            },
            "user_gauc_weighting": "uniform_over_eligible_users",
            "single_class_user_policy": "excluded_with_user_and_row_coverage",
            "continuous_pairwise_policy": (
                "unequal_targets_only_prediction_ties_half_credit"
            ),
            "continuous_user_gauc_weighting": (
                "uniform_over_users_with_unequal_target_pairs"
            ),
        }


def _matrix(
    values: npt.ArrayLike,
    *,
    name: str,
    rows: int | None = None,
    columns: int,
    boolean: bool = False,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape [N, {columns}]")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} row count differs from predictions")
    if boolean:
        if array.dtype != np.bool_:
            raise TypeError(f"{name} must contain exact booleans")
        return array
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    probability = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    probability[~positive] = exponent / (1.0 + exponent)
    return probability


def evaluate_multitarget_predictions(
    *,
    spec: MultiTargetSpec,
    constants: TaskConstants,
    logits: npt.ArrayLike,
    labels: npt.ArrayLike,
    masks: npt.ArrayLike,
    user_ids: npt.ArrayLike,
    regression_scales: Mapping[str, float],
    session_ids: npt.ArrayLike | None = None,
) -> MultiTargetEvaluation:
    """Evaluate one canonical pass without publishing user identifiers.

    Binary model outputs are logits and are converted with a stable sigmoid.
    Regression outputs are normalized predictions.  Every target is evaluated
    only where its boolean mask is true; an observed zero remains a label.
    """

    if not isinstance(spec, MultiTargetSpec) or not isinstance(
        constants, TaskConstants
    ):
        raise TypeError("spec and constants must use the frozen multi-target types")
    target_count = len(spec)
    prediction_matrix = _matrix(logits, name="logits", columns=target_count)
    label_matrix = _matrix(
        labels,
        name="labels",
        rows=prediction_matrix.shape[0],
        columns=target_count,
    )
    mask_matrix = _matrix(
        masks,
        name="masks",
        rows=prediction_matrix.shape[0],
        columns=target_count,
        boolean=True,
    )
    groups = np.asarray(user_ids)
    if groups.ndim != 1 or groups.shape[0] != prediction_matrix.shape[0]:
        raise ValueError("user_ids must be one-dimensional and row-aligned")
    if any(value is None for value in groups.tolist()):
        raise ValueError("user_ids must not contain missing values")
    sessions: np.ndarray | None = None
    if session_ids is not None:
        sessions = np.asarray(session_ids)
        if sessions.ndim != 1 or sessions.shape[0] != prediction_matrix.shape[0]:
            raise ValueError("session_ids must be one-dimensional and row-aligned")
        if any(value is None for value in sessions.tolist()):
            raise ValueError("session_ids must not contain missing values")

    regression_names = {
        target.name for target in spec.targets if target.kind == "regression"
    }
    if set(regression_scales) != regression_names:
        raise ValueError("regression_scales must match regression targets exactly")
    for name, scale in regression_scales.items():
        if type(scale) is not float or not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"regression scale for {name!r} must be positive float")

    objective = macro_nloss(
        spec,
        constants,
        torch.from_numpy(prediction_matrix),
        torch.from_numpy(label_matrix),
        torch.from_numpy(mask_matrix),
    ).as_report(spec)

    binary_reports: list[tuple[str, BinaryTargetMetrics]] = []
    regression_reports: list[tuple[str, RegressionTargetMetrics]] = []
    for index, target in enumerate(spec.targets):
        observed = mask_matrix[:, index]
        observed_rows = int(observed.sum())
        if observed_rows == 0:
            raise ValueError(
                f"target {target.name!r} has no observed evaluation labels"
            )
        if target.kind == "binary":
            binary_reports.append(
                (
                    target.name,
                    masked_binary_target_metrics(
                        label_matrix[:, index],
                        _sigmoid(prediction_matrix[:, index]),
                        groups,
                        observed,
                        session_group=sessions,
                    ),
                )
            )
        else:
            selected_labels = label_matrix[observed, index]
            selected_predictions = prediction_matrix[observed, index]
            session_gauc: GroupConcordance | None = None
            session_availability = "not_available_source_has_no_event_timestamp"
            if sessions is not None:
                try:
                    session_gauc = group_concordance(
                        selected_labels,
                        selected_predictions,
                        sessions[observed],
                        weighting="uniform",
                    )
                    session_availability = "available"
                except ValueError as error:
                    if "unequal-target pairs" not in str(error):
                        raise
                    session_availability = "not_available_no_comparable_sessions"
            regression_reports.append(
                (
                    target.name,
                    RegressionTargetMetrics(
                        metrics=watch_time_metrics(
                            selected_labels,
                            selected_predictions,
                            duration_constant_seconds=regression_scales[target.name],
                        ),
                        user_gauc=group_concordance(
                            selected_labels,
                            selected_predictions,
                            groups[observed],
                            weighting="uniform",
                        ),
                        session_gauc=session_gauc,
                        session_gauc_availability=session_availability,
                        observed_rows=observed_rows,
                        total_rows=prediction_matrix.shape[0],
                        observed_row_coverage=(
                            observed_rows / prediction_matrix.shape[0]
                        ),
                    ),
                )
            )

    return MultiTargetEvaluation(
        macro_nloss=float(objective["macro_nloss"]),
        per_target_normalized=tuple(objective["per_target_normalized"].items()),
        per_target_raw=tuple(objective["per_target_raw"].items()),
        observed_labels=tuple(objective["observed_labels"].items()),
        binary=tuple(binary_reports),
        regression=tuple(regression_reports),
    )


__all__ = [
    "MultiTargetEvaluation",
    "RegressionTargetMetrics",
    "evaluate_multitarget_predictions",
]
