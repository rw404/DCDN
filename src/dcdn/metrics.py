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
class BinaryTargetMetrics:
    log_loss: float
    roc_auc: float
    auprc: float
    user_gauc: GroupAUC
    session_gauc: GroupAUC | None
    session_gauc_availability: str
    observed_rows: int
    total_rows: int
    observed_row_coverage: float


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
class GroupConcordance:
    value: float
    retained_groups: int
    total_groups: int
    retained_rows: int
    total_rows: int
    retained_comparable_pairs: int
    total_within_group_pairs: int
    group_coverage: float
    row_coverage: float
    comparable_pair_coverage: float
    weighting: str


@dataclass(frozen=True)
class WatchTimeMetrics:
    normalized_mse: float
    mae_seconds: float
    mse_seconds2: float
    concordance: Concordance

    @property
    def global_xauc(self) -> Concordance:
        """Explicit reporting name for continuous-label concordance."""

        return self.concordance


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


def average_precision(
    target: Iterable[float] | npt.ArrayLike,
    score: Iterable[float] | npt.ArrayLike,
) -> float:
    """Threshold-grouped average precision, invariant to row order within ties."""

    y_true, y_score = _paired_arrays(target, score)
    if not np.isin(y_true, (0.0, 1.0)).all():
        raise ValueError("binary target must contain only zero and one")
    positive_count = int(y_true.sum())
    if positive_count == 0:
        raise ValueError("average precision is undefined without a positive target")

    order = np.argsort(-y_score, kind="mergesort")
    sorted_score = y_score[order]
    sorted_target = y_true[order]
    true_positives = 0
    predicted_positives = 0
    area = 0.0
    start = 0
    while start < sorted_score.size:
        stop = start + 1
        while stop < sorted_score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        group_positives = int(sorted_target[start:stop].sum())
        true_positives += group_positives
        predicted_positives += stop - start
        if group_positives:
            recall_increment = group_positives / positive_count
            precision = true_positives / predicted_positives
            area += recall_increment * precision
        start = stop
    return float(area)


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

    unique_groups, inverse, counts = np.unique(
        groups, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="mergesort")
    aucs: list[float] = []
    weights: list[int] = []
    start = 0
    for count_value in counts.tolist():
        stop = start + int(count_value)
        selected = order[start:stop]
        group_target = y_true[selected]
        if np.unique(group_target).size >= 2:
            aucs.append(roc_auc(group_target, y_score[selected]))
            weights.append(int(count_value))
        start = stop

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


def masked_binary_target_metrics(
    target: Iterable[float] | npt.ArrayLike,
    probability: Iterable[float] | npt.ArrayLike,
    group: Iterable[object] | npt.ArrayLike,
    mask: Iterable[bool] | npt.ArrayLike,
    *,
    session_group: Iterable[object] | npt.ArrayLike | None = None,
) -> BinaryTargetMetrics:
    """Report one binary head over observed labels with uniform user-GAUC."""

    y_true, y_probability = _paired_arrays(target, probability)
    groups = np.asarray(group)
    observed = np.asarray(mask)
    sessions: np.ndarray | None = None
    if session_group is not None:
        sessions = np.asarray(session_group)
        if sessions.ndim != 1 or sessions.shape != y_true.shape:
            raise ValueError(
                "session_group must be one-dimensional and aligned with target"
            )
    if groups.ndim != 1 or groups.shape != y_true.shape:
        raise ValueError("group must be one-dimensional and aligned with target")
    if observed.ndim != 1 or observed.shape != y_true.shape:
        raise ValueError("mask must be one-dimensional and aligned with target")
    if observed.dtype != np.bool_:
        raise TypeError("mask must contain booleans; observed zeros are valid labels")
    observed_count = int(observed.sum())
    if observed_count == 0:
        raise ValueError("binary target has no observed labels")
    selected_target = y_true[observed]
    selected_probability = y_probability[observed]
    selected_groups = groups[observed]
    selected_sessions = sessions[observed] if sessions is not None else None
    session_gauc: GroupAUC | None = None
    session_availability = "not_available_source_has_no_event_timestamp"
    if selected_sessions is not None:
        try:
            session_gauc = user_group_auc(
                selected_target,
                selected_probability,
                selected_sessions,
                weighting="uniform",
            )
            session_availability = "available"
        except ValueError as error:
            if "single-class" not in str(error):
                raise
            session_availability = "not_available_no_eligible_sessions"
    return BinaryTargetMetrics(
        log_loss=binary_log_loss(selected_target, selected_probability),
        roc_auc=roc_auc(selected_target, selected_probability),
        auprc=average_precision(selected_target, selected_probability),
        user_gauc=user_group_auc(
            selected_target,
            selected_probability,
            selected_groups,
            weighting="uniform",
        ),
        session_gauc=session_gauc,
        session_gauc_availability=session_availability,
        observed_rows=observed_count,
        total_rows=int(y_true.size),
        observed_row_coverage=observed_count / int(y_true.size),
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


def group_concordance(
    target: Iterable[float] | npt.ArrayLike,
    prediction: Iterable[float] | npt.ArrayLike,
    group: Iterable[object] | npt.ArrayLike,
    *,
    weighting: Literal["rows", "uniform", "comparable_pairs"],
) -> GroupConcordance:
    """Average continuous-label X-AUC over eligible groups.

    A group is eligible when it contains at least one pair with unequal target
    values. Target ties are excluded and prediction ties receive half credit,
    exactly as in :func:`global_concordance`.
    """

    y_true, y_pred = _paired_arrays(target, prediction)
    groups = np.asarray(group)
    if groups.ndim != 1 or groups.shape != y_true.shape:
        raise ValueError("group must be one-dimensional and aligned with target")
    if weighting not in {"rows", "uniform", "comparable_pairs"}:
        raise ValueError("weighting must be 'rows', 'uniform', or 'comparable_pairs'")

    unique_groups, inverse, counts = np.unique(
        groups, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="mergesort")
    values: list[float] = []
    row_weights: list[int] = []
    pair_weights: list[int] = []
    total_within_group_pairs = 0
    start = 0
    for count_value in counts.tolist():
        selected_rows = int(count_value)
        stop = start + selected_rows
        selected = order[start:stop]
        total_within_group_pairs += selected_rows * (selected_rows - 1) // 2
        if selected_rows >= 2 and np.unique(y_true[selected]).size >= 2:
            metric = global_concordance(y_true[selected], y_pred[selected])
            values.append(metric.value)
            row_weights.append(selected_rows)
            pair_weights.append(metric.comparable_pairs)
        start = stop

    if not values:
        raise ValueError(
            "group concordance is undefined because every group lacks "
            "unequal-target pairs"
        )
    if weighting == "rows":
        metric_weights = row_weights
    elif weighting == "comparable_pairs":
        metric_weights = pair_weights
    else:
        metric_weights = [1] * len(values)
    retained_rows = int(sum(row_weights))
    retained_pairs = int(sum(pair_weights))
    return GroupConcordance(
        value=float(np.average(np.asarray(values), weights=metric_weights)),
        retained_groups=len(values),
        total_groups=int(unique_groups.size),
        retained_rows=retained_rows,
        total_rows=int(y_true.size),
        retained_comparable_pairs=retained_pairs,
        total_within_group_pairs=total_within_group_pairs,
        group_coverage=len(values) / int(unique_groups.size),
        row_coverage=retained_rows / int(y_true.size),
        comparable_pair_coverage=(
            retained_pairs / total_within_group_pairs
            if total_within_group_pairs
            else 0.0
        ),
        weighting=weighting,
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
