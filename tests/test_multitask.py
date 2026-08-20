from __future__ import annotations

import math

import pytest
import torch

from dcdn.multitask import (
    MultiTargetContractError,
    MultiTargetPredictionHead,
    MultiTargetSpec,
    TargetSpec,
    TaskConstants,
    compute_task_constants,
    macro_nloss,
)

BINARY = TargetSpec(name="like", kind="binary")
REGRESSION = TargetSpec(name="watch_time", kind="regression")


def make_spec() -> MultiTargetSpec:
    return MultiTargetSpec(
        dataset_id="unit",
        targets=(REGRESSION, BINARY),
        primary_confirmatory_target="watch_time",
    )


def make_train_block(rows: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    watch = torch.rand(rows, 1, generator=generator)
    like = (torch.rand(rows, 1, generator=generator) < 0.3).float()
    labels = torch.cat([watch, like], dim=1)
    masks = torch.ones_like(labels, dtype=torch.bool)
    return labels, masks


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


def test_spec_rejects_duplicate_target_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        MultiTargetSpec(
            dataset_id="unit",
            targets=(BINARY, BINARY),
            primary_confirmatory_target="like",
        )


def test_spec_rejects_unknown_primary_target() -> None:
    with pytest.raises(ValueError, match="primary_confirmatory_target"):
        MultiTargetSpec(
            dataset_id="unit",
            targets=(BINARY,),
            primary_confirmatory_target="share",
        )


def test_spec_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        TargetSpec(name="x", kind="ordinal")  # type: ignore[arg-type]


def test_spec_hash_is_order_sensitive() -> None:
    forward = MultiTargetSpec("unit", (REGRESSION, BINARY), "watch_time")
    reversed_order = MultiTargetSpec("unit", (BINARY, REGRESSION), "watch_time")
    assert forward.contract_sha256 != reversed_order.contract_sha256


def test_spec_hash_is_stable_across_equal_specs() -> None:
    assert make_spec().contract_sha256 == make_spec().contract_sha256


def test_spec_declares_no_adaptive_weighting() -> None:
    state = make_spec().contract_state()
    assert state["adaptive_task_weights"] is False
    assert state["learned_uncertainty_weights"] is False
    assert state["gradnorm"] is False
    assert state["aggregation"] == "unweighted_mean_over_observed_targets"


def test_spec_from_frozen_mapping_shape() -> None:
    spec = MultiTargetSpec.from_mapping(
        "vk",
        {
            "primary_confirmatory_target": "watch_time",
            "targets": [
                {"name": "watch_time", "kind": "regression"},
                {"name": "like", "kind": "binary"},
            ],
        },
    )
    assert spec.names == ("watch_time", "like")
    assert spec.kinds == ("regression", "binary")


# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------


def test_constants_match_closed_form() -> None:
    spec = make_spec()
    labels, masks = make_train_block()
    constants = compute_task_constants(spec, labels, masks)

    expected_variance = float(labels[:, 0].to(torch.float64).var(unbiased=False))
    rate = float(labels[:, 1].mean())
    expected_entropy = -(rate * math.log(rate) + (1 - rate) * math.log1p(-rate))

    assert constants.values[0] == pytest.approx(expected_variance, rel=1e-12)
    assert constants.values[1] == pytest.approx(expected_entropy, rel=1e-12)
    assert constants.observed_counts == (labels.shape[0], labels.shape[0])


def test_constants_ignore_unobserved_rows() -> None:
    spec = MultiTargetSpec("unit", (BINARY,), "like")
    labels = torch.tensor([[1.0], [1.0], [0.0], [7.0]])
    masks = torch.tensor([[True], [True], [True], [False]])
    constants = compute_task_constants(spec, labels, masks)

    rate = 2.0 / 3.0
    expected = -(rate * math.log(rate) + (1 - rate) * math.log1p(-rate))
    assert constants.values[0] == pytest.approx(expected, rel=1e-12)
    assert constants.observed_counts == (3,)


def test_observed_zero_is_a_valid_negative_not_a_missing_value() -> None:
    spec = MultiTargetSpec("unit", (BINARY,), "like")
    labels = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    masks = torch.ones_like(labels, dtype=torch.bool)
    constants = compute_task_constants(spec, labels, masks)
    # A rate of 0.25 proves the three zeros were counted as negatives.
    assert constants.observed_counts == (4,)
    rate = 0.25
    expected = -(rate * math.log(rate) + (1 - rate) * math.log1p(-rate))
    assert constants.values[0] == pytest.approx(expected, rel=1e-12)


def test_constant_binary_target_is_rejected() -> None:
    spec = MultiTargetSpec("unit", (BINARY,), "like")
    labels = torch.ones(16, 1)
    masks = torch.ones_like(labels, dtype=torch.bool)
    with pytest.raises(MultiTargetContractError, match="constant on train"):
        compute_task_constants(spec, labels, masks)


def test_zero_variance_regression_target_is_rejected() -> None:
    spec = MultiTargetSpec("unit", (REGRESSION,), "watch_time")
    labels = torch.full((16, 1), 0.4)
    masks = torch.ones_like(labels, dtype=torch.bool)
    with pytest.raises(MultiTargetContractError, match="zero train variance"):
        compute_task_constants(spec, labels, masks)


def test_non_binary_value_in_binary_target_is_rejected() -> None:
    spec = MultiTargetSpec("unit", (BINARY,), "like")
    labels = torch.tensor([[0.0], [1.0], [0.5]])
    masks = torch.ones_like(labels, dtype=torch.bool)
    with pytest.raises(ValueError, match="only zero and one"):
        compute_task_constants(spec, labels, masks)


def test_constants_may_not_be_estimated_off_train() -> None:
    spec = make_spec()
    labels, masks = make_train_block()
    with pytest.raises(MultiTargetContractError, match="train"):
        compute_task_constants(spec, labels, masks, split="validation")


def test_task_constants_are_immutable() -> None:
    spec = make_spec()
    labels, masks = make_train_block()
    constants = compute_task_constants(spec, labels, masks)
    with pytest.raises(AttributeError):
        constants.values = (1.0, 1.0)  # type: ignore[misc]


def test_non_positive_constant_is_rejected() -> None:
    spec = MultiTargetSpec("unit", (REGRESSION,), "watch_time")
    with pytest.raises(ValueError, match="finite and positive"):
        TaskConstants(spec=spec, values=(0.0,), observed_counts=(4,))


def test_float_mask_is_rejected() -> None:
    spec = MultiTargetSpec("unit", (BINARY,), "like")
    labels = torch.tensor([[1.0], [0.0]])
    with pytest.raises(TypeError, match="boolean"):
        compute_task_constants(spec, labels, torch.ones_like(labels))


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def test_constant_predictor_scores_exactly_one_per_target() -> None:
    """The normalizer is defined so that the frozen constant predictor is 1.0."""

    spec = make_spec()
    labels, masks = make_train_block(4096)
    constants = compute_task_constants(spec, labels, masks)

    rate = float(labels[:, 1].mean())
    mean_watch = float(labels[:, 0].mean())
    logits = torch.empty_like(labels)
    logits[:, 0] = mean_watch
    logits[:, 1] = math.log(rate / (1.0 - rate))

    report = macro_nloss(spec, constants, logits, labels, masks)
    assert float(report.per_target[0]) == pytest.approx(1.0, rel=1e-5)
    assert float(report.per_target[1]) == pytest.approx(1.0, rel=1e-5)
    assert float(report.aggregate) == pytest.approx(1.0, rel=1e-5)


def test_aggregate_is_the_unweighted_mean_of_per_target_values() -> None:
    spec = make_spec()
    labels, masks = make_train_block()
    constants = compute_task_constants(spec, labels, masks)
    logits = torch.randn(labels.shape, generator=torch.Generator().manual_seed(3))
    report = macro_nloss(spec, constants, logits, labels, masks)
    assert float(report.aggregate) == pytest.approx(
        float(report.per_target.mean()), rel=1e-9
    )


def test_masked_rows_do_not_affect_the_loss() -> None:
    spec = make_spec()
    labels, masks = make_train_block(256)
    constants = compute_task_constants(spec, labels, masks)
    logits = torch.randn(labels.shape, generator=torch.Generator().manual_seed(5))

    padded_labels = torch.cat([labels, torch.full((16, 2), 1e6)], dim=0)
    padded_logits = torch.cat([logits, torch.full((16, 2), -1e6)], dim=0)
    padded_masks = torch.cat([masks, torch.zeros(16, 2, dtype=torch.bool)], dim=0)

    base = macro_nloss(spec, constants, logits, labels, masks)
    padded = macro_nloss(spec, constants, padded_logits, padded_labels, padded_masks)
    assert float(base.aggregate) == pytest.approx(float(padded.aggregate), rel=1e-9)


def test_target_with_no_observed_label_in_batch_is_excluded_not_nan() -> None:
    spec = make_spec()
    labels, masks = make_train_block()
    constants = compute_task_constants(spec, labels, masks)

    batch_labels = labels[:32].clone()
    batch_logits = torch.randn(
        batch_labels.shape, generator=torch.Generator().manual_seed(7)
    )
    batch_masks = torch.ones_like(batch_labels, dtype=torch.bool)
    batch_masks[:, 1] = False  # the binary head is entirely unobserved here

    report = macro_nloss(spec, constants, batch_logits, batch_labels, batch_masks)
    assert torch.isfinite(report.aggregate)
    assert int(report.observed_counts[1]) == 0
    assert float(report.per_target[1]) == 0.0
    # With one contributing target the aggregate equals that target alone.
    assert float(report.aggregate) == pytest.approx(
        float(report.per_target[0]), rel=1e-9
    )


def test_fully_unobserved_batch_is_rejected() -> None:
    spec = make_spec()
    labels, masks = make_train_block()
    constants = compute_task_constants(spec, labels, masks)
    empty = torch.zeros_like(masks[:8])
    with pytest.raises(MultiTargetContractError, match="undefined"):
        macro_nloss(spec, constants, torch.zeros(8, 2), labels[:8], empty)


def test_constants_frozen_for_another_spec_are_rejected() -> None:
    spec = make_spec()
    other = MultiTargetSpec("other", (REGRESSION, BINARY), "watch_time")
    labels, masks = make_train_block()
    constants = compute_task_constants(other, labels, masks)
    with pytest.raises(MultiTargetContractError, match="different spec"):
        macro_nloss(spec, constants, torch.zeros_like(labels), labels, masks)


def test_shape_mismatch_between_logits_and_labels_is_rejected() -> None:
    spec = make_spec()
    labels, masks = make_train_block()
    constants = compute_task_constants(spec, labels, masks)
    with pytest.raises(ValueError, match="shape of labels"):
        macro_nloss(spec, constants, torch.zeros(labels.shape[0], 3), labels, masks)


def test_report_publishes_every_target_separately() -> None:
    spec = make_spec()
    labels, masks = make_train_block()
    constants = compute_task_constants(spec, labels, masks)
    report = macro_nloss(spec, constants, torch.zeros_like(labels), labels, masks)
    payload = report.as_report(spec)
    assert set(payload["per_target_normalized"]) == {"watch_time", "like"}
    assert set(payload["per_target_raw"]) == {"watch_time", "like"}
    assert set(payload["observed_labels"]) == {"watch_time", "like"}
    assert "macro_nloss" in payload


def test_gradient_flows_to_every_head() -> None:
    spec = make_spec()
    labels, masks = make_train_block(128)
    constants = compute_task_constants(spec, labels, masks)
    logits = torch.zeros_like(labels, requires_grad=True)
    macro_nloss(spec, constants, logits, labels, masks).aggregate.backward()
    assert logits.grad is not None
    assert float(logits.grad[:, 0].abs().sum()) > 0.0
    assert float(logits.grad[:, 1].abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


def test_head_emits_one_column_per_target() -> None:
    spec = make_spec()
    head = MultiTargetPredictionHead(16, spec)
    out = head(torch.randn(9, 16))
    assert out.shape == (9, len(spec))


def test_head_preserves_batch_of_one() -> None:
    head = MultiTargetPredictionHead(8, make_spec())
    assert head(torch.randn(1, 8)).shape == (1, 2)


def test_head_uses_separate_modules_per_target() -> None:
    spec = make_spec()
    head = MultiTargetPredictionHead(8, spec)
    assert len(head.heads) == len(spec)
    # Zeroing one head must not change the other target's column.
    probe = torch.randn(4, 8)
    before = head(probe).detach().clone()
    with torch.no_grad():
        head.heads[0].weight.zero_()
        head.heads[0].bias.fill_(3.0)
    after = head(probe).detach()
    assert torch.allclose(after[:, 1], before[:, 1])
    assert torch.allclose(after[:, 0], torch.full((4,), 3.0))


def test_head_rejects_wrong_representation_width() -> None:
    head = MultiTargetPredictionHead(8, make_spec())
    with pytest.raises(ValueError, match="representation_dim"):
        head(torch.randn(4, 7))


def test_head_contract_binds_the_spec_hash() -> None:
    spec = make_spec()
    head = MultiTargetPredictionHead(16, spec)
    assert head.contract_state()["spec_sha256"] == spec.contract_sha256
    assert head.contract_state()["heads"] == "one_identical_linear_head_per_target"


def test_head_contract_changes_with_target_count() -> None:
    two = MultiTargetPredictionHead(16, make_spec())
    three = MultiTargetPredictionHead(
        16,
        MultiTargetSpec(
            "unit",
            (REGRESSION, BINARY, TargetSpec("share", "binary")),
            "watch_time",
        ),
    )
    assert two.contract_sha256 != three.contract_sha256


def test_head_reset_is_deterministic_under_a_fixed_seed() -> None:
    spec = make_spec()
    torch.manual_seed(11)
    first = MultiTargetPredictionHead(16, spec)
    torch.manual_seed(11)
    second = MultiTargetPredictionHead(16, spec)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)
