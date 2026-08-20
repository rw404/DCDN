"""Predeclared paired clustered uncertainty and multiplicity utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ClusterBootstrapResult:
    effect_treatment_minus_baseline: float
    confidence_low: float
    confidence_high: float
    confidence_level: float
    clusters: int
    rows: int
    resamples: int
    seed: int
    sampling_unit: str
    distribution_sha256: str


def _finite_vector(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def paired_cluster_bootstrap(
    baseline: npt.ArrayLike,
    treatment: npt.ArrayLike,
    clusters: Sequence[object] | npt.ArrayLike,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    sampling_unit: str,
) -> ClusterBootstrapResult:
    """Percentile CI for a paired row-mean delta, resampling whole clusters.

    Each sampled cluster contributes every one of its rows. The reported sign
    is always treatment minus baseline; metric direction is interpreted by the
    frozen dataset manifest rather than changed inside this function.
    """

    baseline_array = _finite_vector(baseline, "baseline")
    treatment_array = _finite_vector(treatment, "treatment")
    cluster_array = np.asarray(clusters)
    if treatment_array.shape != baseline_array.shape:
        raise ValueError("baseline and treatment must have identical shapes")
    if cluster_array.ndim != 1 or cluster_array.shape != baseline_array.shape:
        raise ValueError("clusters must be one-dimensional and row-aligned")
    if not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(sampling_unit, str) or not sampling_unit.strip():
        raise ValueError("sampling_unit must be a non-empty frozen label")

    grouped_indices: dict[object, list[int]] = {}
    for row_index, cluster in enumerate(cluster_array.tolist()):
        try:
            grouped_indices.setdefault(cluster, []).append(row_index)
        except TypeError as exc:
            raise ValueError("cluster identifiers must be hashable scalars") from exc
    cluster_rows = [
        np.asarray(indices, dtype=np.int64) for indices in grouped_indices.values()
    ]
    if len(cluster_rows) < 2:
        raise ValueError("cluster bootstrap needs at least two clusters")

    paired_delta = treatment_array - baseline_array
    rng = np.random.default_rng(seed)
    distribution = np.empty(n_resamples, dtype=np.float64)
    cluster_count = len(cluster_rows)
    for sample_index in range(n_resamples):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        numerator = 0.0
        denominator = 0
        for cluster_index in sampled:
            rows = cluster_rows[int(cluster_index)]
            numerator += float(paired_delta[rows].sum())
            denominator += int(rows.size)
        distribution[sample_index] = numerator / denominator

    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(distribution, [tail, 1.0 - tail])
    distribution_bytes = np.ascontiguousarray(distribution.astype("<f8")).tobytes()
    return ClusterBootstrapResult(
        effect_treatment_minus_baseline=float(paired_delta.mean()),
        confidence_low=float(low),
        confidence_high=float(high),
        confidence_level=confidence_level,
        clusters=cluster_count,
        rows=int(paired_delta.size),
        resamples=n_resamples,
        seed=seed,
        sampling_unit=sampling_unit,
        distribution_sha256=hashlib.sha256(distribution_bytes).hexdigest(),
    )


@dataclass(frozen=True)
class ClusterSufficientBootstrapResult:
    effect_treatment_minus_baseline: float
    confidence_low: float
    confidence_high: float
    confidence_level: float
    clusters: int
    rows: int
    resamples: int
    seed: int
    sampling_unit: str
    two_sided_bootstrap_p_value: float
    p_value_definition: str
    distribution_sha256: str


def paired_cluster_bootstrap_sufficient(
    baseline: Mapping[str, tuple[int, float]],
    treatment: Mapping[str, tuple[int, float]],
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    sampling_unit: str,
) -> ClusterSufficientBootstrapResult:
    """Paired row-mean delta from aligned per-cluster counts and loss sums.

    The two-sided bootstrap p-value uses a finite-resample plus-one correction:
    twice the smaller of the non-positive and non-negative empirical tail
    probabilities, each computed as ``(1 + count) / (B + 1)`` and clipped at
    one. This definition is frozen before test access for subsequent Holm
    adjustment of the predeclared primary family.
    """

    if not isinstance(baseline, Mapping) or not isinstance(treatment, Mapping):
        raise ValueError("cluster sufficient statistics must be mappings")
    if set(baseline) != set(treatment):
        raise ValueError("baseline and treatment cluster keys must match exactly")
    if len(baseline) < 2:
        raise ValueError("cluster bootstrap needs at least two clusters")
    if type(n_resamples) is not int or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(sampling_unit, str) or not sampling_unit.strip():
        raise ValueError("sampling_unit must be a non-empty frozen label")

    keys = sorted(baseline)
    rows = np.empty(len(keys), dtype=np.int64)
    delta_sums = np.empty(len(keys), dtype=np.float64)
    for index, key in enumerate(keys):
        if not isinstance(key, str) or not key:
            raise ValueError("cluster keys must be non-empty strings")
        baseline_item = baseline[key]
        treatment_item = treatment[key]
        if (
            not isinstance(baseline_item, tuple)
            or len(baseline_item) != 2
            or not isinstance(treatment_item, tuple)
            or len(treatment_item) != 2
        ):
            raise ValueError("each cluster statistic must be a (rows, sum) tuple")
        baseline_rows, baseline_sum = baseline_item
        treatment_rows, treatment_sum = treatment_item
        if (
            type(baseline_rows) is not int
            or baseline_rows <= 0
            or type(treatment_rows) is not int
            or treatment_rows != baseline_rows
        ):
            raise ValueError("paired cluster row counts must match and be positive")
        if (
            isinstance(baseline_sum, bool)
            or not isinstance(baseline_sum, int | float)
            or isinstance(treatment_sum, bool)
            or not isinstance(treatment_sum, int | float)
            or not np.isfinite(float(baseline_sum))
            or not np.isfinite(float(treatment_sum))
            or float(baseline_sum) < 0
            or float(treatment_sum) < 0
        ):
            raise ValueError("cluster loss sums must be finite and non-negative")
        rows[index] = baseline_rows
        delta_sums[index] = float(treatment_sum) - float(baseline_sum)

    total_rows = int(rows.sum())
    effect = float(delta_sums.sum() / total_rows)
    rng = np.random.default_rng(seed)
    distribution = np.empty(n_resamples, dtype=np.float64)
    cluster_count = len(keys)
    for sample_index in range(n_resamples):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        distribution[sample_index] = float(
            delta_sums[sampled].sum() / rows[sampled].sum()
        )
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(distribution, [tail, 1.0 - tail])
    nonpositive = (1 + int(np.count_nonzero(distribution <= 0.0))) / (n_resamples + 1)
    nonnegative = (1 + int(np.count_nonzero(distribution >= 0.0))) / (n_resamples + 1)
    p_value = min(1.0, 2.0 * min(nonpositive, nonnegative))
    distribution_bytes = np.ascontiguousarray(distribution.astype("<f8")).tobytes()
    return ClusterSufficientBootstrapResult(
        effect_treatment_minus_baseline=effect,
        confidence_low=float(low),
        confidence_high=float(high),
        confidence_level=confidence_level,
        clusters=cluster_count,
        rows=total_rows,
        resamples=n_resamples,
        seed=seed,
        sampling_unit=sampling_unit,
        two_sided_bootstrap_p_value=p_value,
        p_value_definition=("two_sided_min_tail_plus_one_over_resamples_plus_one_v1"),
        distribution_sha256=hashlib.sha256(distribution_bytes).hexdigest(),
    )


def holm_adjust(
    p_values: Mapping[str, float] | Iterable[tuple[str, float]],
) -> dict[str, float]:
    """Holm family-wise-error adjustment with monotone adjusted p-values."""

    items = list(p_values.items() if isinstance(p_values, Mapping) else p_values)
    if not items:
        raise ValueError("at least one primary contrast is required")
    labels = [label for label, _ in items]
    if len(labels) != len(set(labels)) or not all(
        isinstance(label, str) and label for label in labels
    ):
        raise ValueError("contrast labels must be unique non-empty strings")
    for _, value in items:
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be finite and lie in [0, 1]")

    ordered = sorted(enumerate(items), key=lambda item: (item[1][1], item[0]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    family_size = len(items)
    for rank, (_, (label, value)) in enumerate(ordered):
        scaled = min(1.0, (family_size - rank) * float(value))
        running_max = max(running_max, scaled)
        adjusted[label] = running_max
    return {label: adjusted[label] for label in labels}
