from __future__ import annotations

import itertools

import numpy as np
import pytest

from dcdn.metrics import (
    average_precision,
    binary_log_loss,
    global_concordance,
    masked_binary_target_metrics,
    roc_auc,
    user_group_auc,
    watch_time_metrics,
)


def brute_concordance(target: np.ndarray, prediction: np.ndarray) -> tuple:
    concordant = discordant = prediction_tied = comparable = 0
    for left, right in itertools.combinations(range(target.size), 2):
        if target[left] == target[right]:
            continue
        comparable += 1
        target_order = np.sign(target[left] - target[right])
        prediction_order = np.sign(prediction[left] - prediction[right])
        if prediction_order == 0:
            prediction_tied += 1
        elif target_order == prediction_order:
            concordant += 1
        else:
            discordant += 1
    value = (concordant + 0.5 * prediction_tied) / comparable
    return value, comparable, concordant, discordant, prediction_tied


def test_binary_log_loss_known_value_and_contract() -> None:
    target = np.array([0.0, 1.0])
    probability = np.array([0.25, 0.75])
    assert binary_log_loss(target, probability) == pytest.approx(-np.log(0.75))
    with pytest.raises(ValueError):
        binary_log_loss([0, 2], [0.1, 0.9])
    with pytest.raises(ValueError):
        binary_log_loss([0, 1], [-0.1, 1.1])


def test_roc_auc_ties_and_single_class() -> None:
    assert roc_auc([0, 1, 0, 1], [0.0, 1.0, 0.5, 0.5]) == pytest.approx(0.875)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="single-class"):
        roc_auc([1, 1], [0.0, 1.0])


def test_average_precision_is_threshold_grouped_and_tie_invariant() -> None:
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]) == pytest.approx(
        (1.0 + 2 / 3) / 2
    )
    assert average_precision([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="without a positive"):
        average_precision([0, 0], [0.2, 0.1])


def test_user_group_auc_coverage_and_explicit_weighting() -> None:
    target = [0, 1, 0, 1, 1, 1, 0]
    score = [0.0, 1.0, 0.7, 0.6, 0.2, 0.3, 0.4]
    group = ["a", "a", "b", "b", "b", "c", "d"]

    rows = user_group_auc(target, score, group, weighting="rows")
    uniform = user_group_auc(target, score, group, weighting="uniform")
    assert rows.retained_groups == 2
    assert rows.total_groups == 4
    assert rows.retained_rows == 5
    assert rows.row_coverage == pytest.approx(5 / 7)
    assert rows.value != uniform.value


def test_masked_binary_metrics_keep_observed_zeros_and_report_user_coverage() -> None:
    target = np.asarray([0, 1, 0, 1, 1, 1, 0, 0], dtype=float)
    probability = np.asarray([0.1, 0.9, 0.7, 0.6, 0.2, 0.3, 0.4, 0.8])
    group = np.asarray(["a", "a", "b", "b", "b", "c", "d", "e"])
    mask = np.asarray([True, True, True, True, True, True, True, False])
    report = masked_binary_target_metrics(target, probability, group, mask)
    assert report.observed_rows == 7
    assert report.total_rows == 8
    assert report.observed_row_coverage == pytest.approx(7 / 8)
    assert report.user_gauc.weighting == "uniform"
    assert report.user_gauc.retained_groups == 2
    assert report.user_gauc.total_groups == 4
    assert report.user_gauc.retained_rows == 5
    assert report.user_gauc.group_coverage == pytest.approx(0.5)
    assert report.user_gauc.row_coverage == pytest.approx(5 / 7)
    assert 0.0 <= report.roc_auc <= 1.0
    assert 0.0 <= report.auprc <= 1.0


def test_masked_binary_metrics_reject_non_boolean_mask() -> None:
    with pytest.raises(TypeError, match="booleans"):
        masked_binary_target_metrics([0, 1], [0.1, 0.9], ["a", "a"], np.asarray([1, 1]))


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_global_concordance_matches_brute_force_and_is_order_invariant(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 5, size=40).astype(float)
    prediction = rng.integers(0, 7, size=40).astype(float)
    expected = brute_concordance(target, prediction)
    actual = global_concordance(target, prediction)

    assert actual.value == pytest.approx(expected[0])
    assert actual.comparable_pairs == expected[1]
    assert actual.concordant_pairs == expected[2]
    assert actual.discordant_pairs == expected[3]
    assert actual.prediction_tied_pairs == expected[4]

    permutation = rng.permutation(target.size)
    shuffled = global_concordance(target[permutation], prediction[permutation])
    assert shuffled == actual


def test_global_concordance_target_and_prediction_tie_policy() -> None:
    metric = global_concordance(
        [1.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 1.0, 0.5],
    )
    assert metric.total_pairs == 6
    assert metric.comparable_pairs == 5
    assert metric.prediction_tied_pairs == 1
    assert metric.value == pytest.approx((2 + 0.5) / 5)


def test_watch_time_reporting_transform_clips_prediction_only_for_reporting() -> None:
    metric = watch_time_metrics(
        [0.0, 0.5, 1.0],
        [-0.2, 0.4, 1.2],
        duration_constant_seconds=10.0,
    )
    assert metric.normalized_mse == pytest.approx(np.mean([0.04, 0.01, 0.04]))
    assert metric.mae_seconds == pytest.approx(1 / 3)
    assert metric.mse_seconds2 == pytest.approx(1 / 3)
    assert metric.concordance.value == 1.0
