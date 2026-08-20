from __future__ import annotations

import numpy as np
import pytest

from dcdn.statistics import (
    holm_adjust,
    paired_cluster_bootstrap,
    paired_cluster_bootstrap_sufficient,
)


def test_paired_cluster_bootstrap_is_deterministic_and_reports_sign() -> None:
    baseline = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
    treatment = np.array([0.2, 0.4, 0.4, 0.9, 1.1])
    clusters = np.array(["u1", "u1", "u2", "u3", "u3"])
    kwargs = {
        "n_resamples": 200,
        "confidence_level": 0.95,
        "seed": 2027,
        "sampling_unit": "user_id",
    }
    first = paired_cluster_bootstrap(baseline, treatment, clusters, **kwargs)
    second = paired_cluster_bootstrap(baseline, treatment, clusters, **kwargs)

    assert first == second
    assert first.effect_treatment_minus_baseline == pytest.approx(
        np.mean(treatment - baseline)
    )
    assert first.clusters == 3
    assert first.rows == 5
    assert len(first.distribution_sha256) == 64
    assert first.confidence_low <= first.confidence_high


def test_cluster_bootstrap_rejects_invalid_contract() -> None:
    with pytest.raises(ValueError, match="at least two clusters"):
        paired_cluster_bootstrap(
            [0.0, 1.0],
            [1.0, 0.0],
            ["only", "only"],
            n_resamples=10,
            confidence_level=0.95,
            seed=42,
            sampling_unit="user_id",
        )
    with pytest.raises(ValueError, match="row-aligned"):
        paired_cluster_bootstrap(
            [0.0, 1.0],
            [1.0, 0.0],
            ["a"],
            n_resamples=10,
            confidence_level=0.95,
            seed=42,
            sampling_unit="user_id",
        )


def test_sufficient_cluster_bootstrap_is_deterministic_and_row_weighted() -> None:
    baseline = {"u1": (2, 1.0), "u2": (1, 0.4), "u3": (3, 2.1)}
    treatment = {"u1": (2, 0.8), "u2": (1, 0.5), "u3": (3, 1.8)}
    kwargs = {
        "n_resamples": 2_000,
        "confidence_level": 0.95,
        "seed": 20_260_818,
        "sampling_unit": "user_id",
    }

    first = paired_cluster_bootstrap_sufficient(baseline, treatment, **kwargs)
    second = paired_cluster_bootstrap_sufficient(baseline, treatment, **kwargs)

    assert first == second
    assert first.effect_treatment_minus_baseline == pytest.approx(
        ((0.8 + 0.5 + 1.8) - (1.0 + 0.4 + 2.1)) / 6
    )
    assert first.clusters == 3
    assert first.rows == 6
    assert first.resamples == 2_000
    assert 0.0 <= first.two_sided_bootstrap_p_value <= 1.0
    assert first.p_value_definition == (
        "two_sided_min_tail_plus_one_over_resamples_plus_one_v1"
    )
    assert first.confidence_low <= first.confidence_high
    assert len(first.distribution_sha256) == 64

    rows = np.array([2, 1, 3])
    delta_sums = np.array([-0.2, 0.1, -0.3])
    rng = np.random.default_rng(20_260_818)
    distribution = np.empty(2_000)
    for index in range(2_000):
        sampled = rng.integers(0, 3, size=3)
        distribution[index] = delta_sums[sampled].sum() / rows[sampled].sum()
    nonpositive = (1 + np.count_nonzero(distribution <= 0.0)) / 2_001
    nonnegative = (1 + np.count_nonzero(distribution >= 0.0)) / 2_001
    assert first.two_sided_bootstrap_p_value == pytest.approx(
        min(1.0, 2.0 * min(nonpositive, nonnegative))
    )


@pytest.mark.parametrize(
    ("baseline", "treatment", "message"),
    [
        ({"u1": (1, 0.1), "u2": (1, 0.2)}, {"u1": (1, 0.1)}, "keys"),
        (
            {"u1": (1, 0.1), "u2": (1, 0.2)},
            {"u1": (2, 0.1), "u2": (1, 0.2)},
            "row counts",
        ),
        (
            {"u1": (1, np.nan), "u2": (1, 0.2)},
            {"u1": (1, 0.1), "u2": (1, 0.2)},
            "finite",
        ),
        (
            {"u1": (True, 0.1), "u2": (1, 0.2)},
            {"u1": (True, 0.1), "u2": (1, 0.2)},
            "row counts",
        ),
    ],
)
def test_sufficient_cluster_bootstrap_rejects_invalid_contract(
    baseline: dict[str, tuple[int, float]],
    treatment: dict[str, tuple[int, float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_cluster_bootstrap_sufficient(
            baseline,
            treatment,
            n_resamples=2_000,
            confidence_level=0.95,
            seed=20_260_818,
            sampling_unit="user_id",
        )


def test_holm_adjust_known_example_and_input_order() -> None:
    adjusted = holm_adjust([("large", 0.04), ("small", 0.01), ("middle", 0.03)])
    assert list(adjusted) == ["large", "small", "middle"]
    assert adjusted == pytest.approx({"large": 0.06, "small": 0.03, "middle": 0.06})


def test_holm_adjust_rejects_bad_pvalues_and_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="finite"):
        holm_adjust({"x": np.nan})
    with pytest.raises(ValueError, match="unique"):
        holm_adjust([("x", 0.1), ("x", 0.2)])
