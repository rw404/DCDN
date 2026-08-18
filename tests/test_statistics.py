from __future__ import annotations

import numpy as np
import pytest

from dcdn.statistics import holm_adjust, paired_cluster_bootstrap


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


def test_holm_adjust_known_example_and_input_order() -> None:
    adjusted = holm_adjust([("large", 0.04), ("small", 0.01), ("middle", 0.03)])
    assert list(adjusted) == ["large", "small", "middle"]
    assert adjusted == pytest.approx({"large": 0.06, "small": 0.03, "middle": 0.06})


def test_holm_adjust_rejects_bad_pvalues_and_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="finite"):
        holm_adjust({"x": np.nan})
    with pytest.raises(ValueError, match="unique"):
        holm_adjust([("x", 0.1), ("x", 0.2)])
