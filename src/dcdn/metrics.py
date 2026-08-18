"""Dataset-independent metrics with explicit coverage and tie policies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GroupAUC:
    value: float
    retained_groups: int
    total_groups: int
    retained_rows: int
    total_rows: int
    group_coverage: float
    row_coverage: float
    weighting: str


@dataclass(frozen=True)
class Concordance:
    value: float
    comparable_pairs: int
    total_pairs: int
    comparable_pair_coverage: float
    concordant_pairs: int
    discordant_pairs: int
    prediction_tied_pairs: int


@dataclass(frozen=True)
class WatchTimeMetrics:
    normalized_mse: float
    mae_seconds: float
    mse_seconds2: float
    concordance: Concordance


def _one_dimensional_finite(
    values: Iterable[float] | npt.ArrayLike,
    *,
    name: str,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _paired_arrays(
    target: Iterable[float] | npt.ArrayLike,
    prediction: Iterable[float] | npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    y_true = _one_dimensional_finite(target, name="target")
    y_pred = _one_dimensional_finite(prediction, name="prediction")
    if y_true.shape != y_pred.shape:
        raise ValueError("target and prediction must have identical shapes")
    return y_true, y_pred


def binary_log_loss(
    target: Iterable[float] | npt.ArrayLike,
    probability: Iterable[float] | npt.ArrayLike,
    *,
    epsilon: float = 1e-15,
) -> float:
    """Mean binary cross entropy with an explicit clipping constant."""

    y_true, y_prob = _paired_arrays(target, probability)
    if not np.isin(y_true, (0.0, 1.0)).all():
        raise ValueError("binary target must contain only zero and one")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("probability must lie in [0, 1]")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie strictly between zero and 0.5")
    clipped = np.clip(y_prob, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(y_true * np.log(clipped) + (1.0 - y_true) * np.log1p(-clipped))
    )


def roc_auc(
    target: Iterable[float] | npt.ArrayLike,
    score: Iterable[float] | npt.ArrayLike,
) -> float:
    """Tie-aware binary ROC-AUC using average ranks."""

    y_true, y_score = _paired_arrays(target, score)
    if not np.isin(y_true, (0.0, 1.0)).all():
        raise ValueError("binary target must contain only zero and one")
    positive_count = int(y_true.sum())
    negative_count = int(y_true.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC is undefined for a single-class target")

    order = np.argsort(y_score, kind="mergesort")
    sorted_score = y_score[order]
    ranks = np.empty(y_score.size, dtype=np.float64)
    start = 0
    while start < sorted_score.size:
        stop = start + 1
        while stop < sorted_score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop

    positive_rank_sum = float(ranks[y_true == 1.0].sum())
    mann_whitney = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return float(mann_whitney / (positive_count * negative_count))


def user_group_auc(
    target: Iterable[float] | npt.ArrayLike,
    score: Iterable[float] | npt.ArrayLike,
    group: Iterable[object] | npt.ArrayLike,
    *,
    weighting: Literal["rows", "uniform"],
) -> GroupAUC:
    """Average per-group AUC while explicitly excluding single-class groups."""

    y_true, y_score = _paired_arrays(target, score)
    groups = np.asarray(group)
    if groups.ndim != 1 or groups.shape != y_true.shape:
        raise ValueError("group must be one-dimensional and aligned with target")
    if weighting not in {"rows", "uniform"}:
        raise ValueError("weighting must be either 'rows' or 'uniform'")

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    aucs: list[float] = []
    weights: list[int] = []
    for group_index in range(unique_groups.size):
        selected = inverse == group_index
        group_target = y_true[selected]
        if np.unique(group_target).size < 2:
            continue
        aucs.append(roc_auc(group_target, y_score[selected]))
        weights.append(int(selected.sum()))

    if not aucs:
        raise ValueError("group AUC is undefined because every group is single-class")
    retained_rows = int(sum(weights))
    metric_weights = np.asarray(weights if weighting == "rows" else [1] * len(aucs))
    value = float(np.average(np.asarray(aucs), weights=metric_weights))
    return GroupAUC(
        value=value,
        retained_groups=len(aucs),
        total_groups=int(unique_groups.size),
        retained_rows=retained_rows,
        total_rows=int(y_true.size),
        group_coverage=len(aucs) / int(unique_groups.size),
        row_coverage=retained_rows / int(y_true.size),
        weighting=weighting,
    )


class _FenwickTree:
    def __init__(self, size: int) -> None:
        self.values = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int, value: int) -> None:
        cursor = index + 1
        while cursor < self.values.size:
            self.values[cursor] += value
            cursor += cursor & -cursor

    def prefix_sum(self, stop: int) -> int:
        total = 0
        cursor = stop
        while cursor > 0:
            total += int(self.values[cursor])
            cursor -= cursor & -cursor
        return total


def global_concordance(
    target: Iterable[float] | npt.ArrayLike,
    prediction: Iterable[float] | npt.ArrayLike,
) -> Concordance:
    """Order-invariant concordance over unequal-target unordered pairs.

    Matching prediction order counts one, reversed order counts zero,
    prediction ties count one half, and target ties are excluded.
    """

    y_true, y_pred = _paired_arrays(target, prediction)
    total_pairs = int(y_true.size * (y_true.size - 1) // 2)
    if total_pairs == 0:
        raise ValueError("concordance needs at least two rows")

    _, prediction_rank = np.unique(y_pred, return_inverse=True)
    order = np.argsort(y_true, kind="mergesort")
    tree = _FenwickTree(int(prediction_rank.max()) + 1)
    prior_count = 0
    concordant = 0
    discordant = 0
    prediction_tied = 0

    start = 0
    while start < order.size:
        stop = start + 1
        target_value = y_true[order[start]]
        while stop < order.size and y_true[order[stop]] == target_value:
            stop += 1

        for row_index in order[start:stop]:
            rank = int(prediction_rank[row_index])
            lower = tree.prefix_sum(rank)
            lower_or_equal = tree.prefix_sum(rank + 1)
            equal = lower_or_equal - lower
            higher = prior_count - lower_or_equal
            concordant += lower
            prediction_tied += equal
            discordant += higher

        for row_index in order[start:stop]:
            tree.add(int(prediction_rank[row_index]), 1)
            prior_count += 1
        start = stop

    comparable_pairs = concordant + discordant + prediction_tied
    if comparable_pairs == 0:
        raise ValueError("concordance is undefined when all targets are tied")
    value = (concordant + 0.5 * prediction_tied) / comparable_pairs
    return Concordance(
        value=float(value),
        comparable_pairs=comparable_pairs,
        total_pairs=total_pairs,
        comparable_pair_coverage=comparable_pairs / total_pairs,
        concordant_pairs=concordant,
        discordant_pairs=discordant,
        prediction_tied_pairs=prediction_tied,
    )


def watch_time_metrics(
    normalized_target: Iterable[float] | npt.ArrayLike,
    normalized_prediction: Iterable[float] | npt.ArrayLike,
    *,
    duration_constant_seconds: float,
) -> WatchTimeMetrics:
    """Frozen VK-LSVD/KuaiRec normalized-MSE and reporting transform."""

    y_true, y_pred = _paired_arrays(normalized_target, normalized_prediction)
    if not np.isfinite(duration_constant_seconds) or duration_constant_seconds <= 0:
        raise ValueError("duration_constant_seconds must be finite and positive")
    if np.any((y_true < 0.0) | (y_true > 1.0)):
        raise ValueError("normalized target must lie in [0, 1]")

    normalized_mse = float(np.mean(np.square(y_pred - y_true)))
    target_seconds = duration_constant_seconds * np.clip(y_true, 0.0, 1.0)
    prediction_seconds = duration_constant_seconds * np.clip(y_pred, 0.0, 1.0)
    residual_seconds = prediction_seconds - target_seconds
    return WatchTimeMetrics(
        normalized_mse=normalized_mse,
        mae_seconds=float(np.mean(np.abs(residual_seconds))),
        mse_seconds2=float(np.mean(np.square(residual_seconds))),
        concordance=global_concordance(target_seconds, prediction_seconds),
    )
