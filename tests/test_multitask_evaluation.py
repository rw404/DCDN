from __future__ import annotations

import numpy as np
import pytest
import torch

from dcdn.multitask import MultiTargetSpec, compute_task_constants
from dcdn.multitask_evaluation import evaluate_multitarget_predictions

SPEC = MultiTargetSpec.from_mapping(
    "tiny",
    {
        "primary_confirmatory_target": "watch_time",
        "targets": [
            {"name": "watch_time", "kind": "regression"},
            {"name": "like", "kind": "binary"},
            {"name": "share", "kind": "binary"},
        ],
    },
)


def _fixture() -> tuple:
    labels = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.2, 1.0, 0.0],
            [0.4, 0.0, 1.0],
            [0.6, 1.0, 0.0],
            [0.8, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    masks = np.ones_like(labels, dtype=np.bool_)
    constants = compute_task_constants(
        SPEC,
        torch.from_numpy(labels),
        torch.from_numpy(masks),
    )
    logits = labels.copy()
    logits[:, 1:] = np.where(labels[:, 1:] == 1.0, 2.0, -2.0)
    users = np.asarray(["a", "a", "b", "b", "c", "c"])
    return labels, masks, constants, logits, users


def test_complete_report_contains_every_head_and_uniform_user_gauc() -> None:
    labels, masks, constants, logits, users = _fixture()
    report = evaluate_multitarget_predictions(
        spec=SPEC,
        constants=constants,
        logits=logits,
        labels=labels,
        masks=masks,
        user_ids=users,
        regression_scales={"watch_time": 157.0},
    ).as_report()
    assert set(report["per_target_normalized"]) == {"watch_time", "like", "share"}
    assert set(report["per_target_binary"]) == {"like", "share"}
    assert set(report["per_target_regression"]) == {"watch_time"}
    assert report["user_gauc_weighting"] == "uniform_over_eligible_users"
    assert report["per_target_binary"]["like"]["user_gauc"]["value"] == 1.0
    assert report["per_target_binary"]["like"]["user_gauc"]["group_coverage"] == 1.0
    watch = report["per_target_regression"]["watch_time"]
    assert watch["observed_row_coverage"] == 1.0
    assert watch["metrics"]["global_xauc"]["value"] == 1.0
    assert watch["user_gauc"]["value"] == 1.0
    assert watch["user_gauc"]["weighting"] == "uniform"
    assert watch["session_gauc"] is None
    assert (
        watch["session_gauc_availability"]
        == "not_available_source_has_no_event_timestamp"
    )


def test_observed_zero_is_not_masked_and_coverage_is_reported() -> None:
    labels, masks, constants, logits, users = _fixture()
    masks[-1, 1] = False
    report = evaluate_multitarget_predictions(
        spec=SPEC,
        constants=constants,
        logits=logits,
        labels=labels,
        masks=masks,
        user_ids=users,
        regression_scales={"watch_time": 157.0},
    ).as_report()
    like = report["per_target_binary"]["like"]
    assert like["observed_rows"] == 5
    assert like["observed_row_coverage"] == pytest.approx(5 / 6)
    assert report["observed_labels"]["like"] == 5


def test_contract_rejects_nonboolean_masks_and_wrong_scale_keys() -> None:
    labels, masks, constants, logits, users = _fixture()
    with pytest.raises(TypeError, match="exact booleans"):
        evaluate_multitarget_predictions(
            spec=SPEC,
            constants=constants,
            logits=logits,
            labels=labels,
            masks=masks.astype(np.int8),
            user_ids=users,
            regression_scales={"watch_time": 157.0},
        )
    with pytest.raises(ValueError, match="match regression targets"):
        evaluate_multitarget_predictions(
            spec=SPEC,
            constants=constants,
            logits=logits,
            labels=labels,
            masks=masks,
            user_ids=users,
            regression_scales={},
        )


def test_user_gauc_fails_closed_when_every_user_is_single_class() -> None:
    labels, masks, constants, logits, _ = _fixture()
    users = np.asarray(["a", "b", "c", "d", "e", "f"])
    with pytest.raises(ValueError, match="single-class|unequal-target"):
        evaluate_multitarget_predictions(
            spec=SPEC,
            constants=constants,
            logits=logits,
            labels=labels,
            masks=masks,
            user_ids=users,
            regression_scales={"watch_time": 157.0},
        )


def test_continuous_session_gauc_is_reported_when_real_session_ids_exist() -> None:
    labels, masks, constants, logits, users = _fixture()
    sessions = np.asarray(["a1", "a1", "b1", "b1", "c1", "c1"])
    report = evaluate_multitarget_predictions(
        spec=SPEC,
        constants=constants,
        logits=logits,
        labels=labels,
        masks=masks,
        user_ids=users,
        regression_scales={"watch_time": 157.0},
        session_ids=sessions,
    ).as_report()
    watch = report["per_target_regression"]["watch_time"]
    assert watch["session_gauc_availability"] == "available"
    assert watch["session_gauc"]["value"] == 1.0
    assert watch["session_gauc"]["retained_groups"] == 3
    like = report["per_target_binary"]["like"]
    assert like["session_gauc_availability"] == "available"
    assert like["session_gauc"]["value"] == 1.0


def test_session_ids_must_be_row_aligned() -> None:
    labels, masks, constants, logits, users = _fixture()
    with pytest.raises(ValueError, match="session_ids"):
        evaluate_multitarget_predictions(
            spec=SPEC,
            constants=constants,
            logits=logits,
            labels=labels,
            masks=masks,
            user_ids=users,
            regression_scales={"watch_time": 157.0},
            session_ids=["one"],
        )
