from __future__ import annotations

import pytest
import torch

from dcdn.multitask import compute_task_constants, macro_nloss
from dcdn.synthetic_multiquotient import (
    COLUMNS,
    DENOMINATOR_COLUMNS,
    DENOMINATOR_MASK,
    FROZEN_SLICES,
    MINIMUM_DENOMINATOR,
    SPEC,
    TARGET_NAMES,
    SliceConfig,
    SyntheticContractError,
    analytical_oracle,
    build_benchmark,
    explicit_ratio_features,
    generate_slice,
)


@pytest.fixture(scope="module")
def benchmark():
    return build_benchmark()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_columns_match_the_frozen_input_order() -> None:
    assert COLUMNS == ("n1", "d1", "n2", "d2", "d3", "z1", "z2", "z3")


def test_reciprocal_mask_is_active_only_on_denominators() -> None:
    active = {c for c, m in zip(COLUMNS, DENOMINATOR_MASK, strict=True) if m}
    assert active == set(DENOMINATOR_COLUMNS) == {"d1", "d2", "d3"}


def test_numerators_and_distractors_are_never_denominators() -> None:
    for column in ("n1", "n2", "z1", "z2", "z3"):
        assert DENOMINATOR_MASK[COLUMNS.index(column)] is False


def test_spec_declares_three_regression_targets() -> None:
    assert SPEC.names == TARGET_NAMES
    assert set(SPEC.kinds) == {"regression"}
    assert SPEC.primary_confirmatory_target == "y1"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_all_required_slices_are_materialized(benchmark) -> None:
    assert set(benchmark.slices) == {c.name for c in FROZEN_SLICES}
    assert benchmark.evaluation_slice_names == (
        "id",
        "ood_common_scale",
        "ood_low_ratio",
        "ood_high_ratio",
    )


def test_generation_is_deterministic() -> None:
    config = FROZEN_SLICES[2]
    first, second = generate_slice(config), generate_slice(config)
    assert first.sha256 == second.sha256
    assert torch.equal(first.features, second.features)
    assert torch.equal(first.targets, second.targets)


def test_slices_have_distinct_hashes(benchmark) -> None:
    digests = [s.sha256 for s in benchmark.slices.values()]
    assert len(set(digests)) == len(digests)


def test_manifest_hash_is_stable_and_covers_slice_hashes(benchmark) -> None:
    assert benchmark.manifest_sha256 == build_benchmark().manifest_sha256
    manifest = benchmark.manifest()
    for name, item in benchmark.slices.items():
        assert manifest["slices"][name]["sha256"] == item.sha256


def test_every_input_is_strictly_positive(benchmark) -> None:
    for item in benchmark.slices.values():
        assert bool((item.features > 0).all())


def test_denominators_stay_above_the_reciprocal_floor(benchmark) -> None:
    indices = [COLUMNS.index(c) for c in DENOMINATOR_COLUMNS]
    for item in benchmark.slices.values():
        minimum = float(item.features[:, indices].min())
        assert minimum >= MINIMUM_DENOMINATOR * item.config.scale


def test_masks_are_boolean_and_fully_observed(benchmark) -> None:
    for item in benchmark.slices.values():
        assert item.masks.dtype is torch.bool
        assert bool(item.masks.all())


def test_slice_below_the_minimum_denominator_is_rejected() -> None:
    with pytest.raises(SyntheticContractError, match="minimum"):
        SliceConfig(
            "bad", rows=8, seed=1, scale=1.0, denominator_low=0.01, denominator_span=1.0
        )


def test_non_positive_numerator_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        SliceConfig(
            "bad",
            rows=8,
            seed=1,
            scale=1.0,
            denominator_low=1.0,
            denominator_span=1.0,
            numerator_low=0.0,
        )


# ---------------------------------------------------------------------------
# Target semantics
# ---------------------------------------------------------------------------


def test_targets_equal_the_declared_quotients(benchmark) -> None:
    item = benchmark["id"]
    n1 = item.features[:, COLUMNS.index("n1")]
    d1 = item.features[:, COLUMNS.index("d1")]
    n2 = item.features[:, COLUMNS.index("n2")]
    d2 = item.features[:, COLUMNS.index("d2")]
    d3 = item.features[:, COLUMNS.index("d3")]
    assert torch.allclose(item.targets[:, 0], n1 / d1, atol=1e-6)
    assert torch.allclose(item.targets[:, 1], n2 / d2, atol=1e-6)
    assert torch.allclose(item.targets[:, 2], (n1 + n2) / d3, atol=1e-6)


def test_y3_numerator_is_not_an_input_coordinate(benchmark) -> None:
    """``y3``'s numerator must be assembled, not read off a column."""

    item = benchmark["id"]
    y3 = item.targets[:, 2]
    for index in range(len(COLUMNS)):
        column = item.features[:, index]
        for other in range(len(COLUMNS)):
            candidate = column / item.features[:, other]
            assert not torch.allclose(candidate, y3, atol=1e-4)


def test_common_scale_slice_preserves_the_ratios(benchmark) -> None:
    """The point of the common-scale slice: same targets, different operands."""

    identity = benchmark["id"]
    scaled = benchmark["ood_common_scale"]
    # Operand magnitudes differ by the declared factor of ten.
    assert float(scaled.features.mean()) > 5.0 * float(identity.features.mean())
    # Target distributions coincide because a common factor cancels.
    assert float(scaled.targets.mean()) == pytest.approx(
        float(identity.targets.mean()), rel=0.05
    )
    assert float(scaled.targets.var()) == pytest.approx(
        float(identity.targets.var()), rel=0.05
    )


def test_low_and_high_ratio_slices_move_the_quotient(benchmark) -> None:
    identity_mean = float(benchmark["id"].targets.mean())
    assert float(benchmark["ood_low_ratio"].targets.mean()) < identity_mean
    assert float(benchmark["ood_high_ratio"].targets.mean()) > identity_mean


# ---------------------------------------------------------------------------
# Oracle and positive control
# ---------------------------------------------------------------------------


def test_oracle_reproduces_the_targets_on_every_slice(benchmark) -> None:
    for item in benchmark.slices.values():
        error = float((analytical_oracle(item.features) - item.targets).abs().max())
        # Float32 rounding only; the common-scale slice divides scaled operands.
        assert error < 1e-4


def test_oracle_reaches_essentially_zero_normalized_loss(benchmark) -> None:
    train = benchmark["train"]
    constants = compute_task_constants(SPEC, train.targets, train.masks)
    item = benchmark["id"]
    report = macro_nloss(
        SPEC, constants, analytical_oracle(item.features), item.targets, item.masks
    )
    assert float(report.aggregate) < 1e-8


def test_oracle_rejects_a_wrong_input_width() -> None:
    with pytest.raises(ValueError, match="shape"):
        analytical_oracle(torch.rand(4, 3) + 1.0)


def test_positive_control_view_appends_exactly_the_three_ratios(benchmark) -> None:
    item = benchmark["id"]
    controlled = explicit_ratio_features(item.features)
    assert controlled.shape == (item.rows, len(COLUMNS) + 3)
    assert torch.equal(controlled[:, : len(COLUMNS)], item.features)
    assert torch.allclose(controlled[:, len(COLUMNS) :], item.targets, atol=1e-6)


# ---------------------------------------------------------------------------
# Objective integration
# ---------------------------------------------------------------------------


def test_task_constants_are_estimable_on_the_train_slice(benchmark) -> None:
    train = benchmark["train"]
    constants = compute_task_constants(SPEC, train.targets, train.masks)
    assert len(constants.values) == 3
    assert all(value > 0.0 for value in constants.values)
    assert constants.observed_counts == (train.rows,) * 3


def test_constant_predictor_scores_one_on_the_train_slice(benchmark) -> None:
    train = benchmark["train"]
    constants = compute_task_constants(SPEC, train.targets, train.masks)
    means = train.targets.mean(dim=0, keepdim=True).expand_as(train.targets)
    report = macro_nloss(SPEC, constants, means, train.targets, train.masks)
    assert float(report.aggregate) == pytest.approx(1.0, rel=1e-5)


def test_manifest_binds_the_objective_spec(benchmark) -> None:
    assert benchmark.manifest()["objective_spec_sha256"] == SPEC.contract_sha256
