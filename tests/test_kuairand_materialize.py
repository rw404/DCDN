from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "kuairand_materialize",
        REPO_ROOT / "scripts" / "data" / "kuairand_pure_materialize.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["kuairand_materialize"] = module
    spec.loader.exec_module(module)
    return module


km = _load()


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------


def test_five_heads_match_the_amended_objective() -> None:
    assert km.TARGETS == ("long_view", "like", "forward", "hate", "watch_time")
    assert km.TARGET_KIND["watch_time"] == "regression"
    assert all(
        km.TARGET_KIND[name] == "binary" for name in km.TARGETS[:4]
    )


def test_locked_test_window_is_the_declared_three_days() -> None:
    assert {"20220506", "20220507", "20220508"} == km.TEST_DATES
    # The locked window must not overlap any split that is materialized.
    assert not km.TEST_DATES & km.TRAIN_DATES
    assert not km.TEST_DATES & km.VALIDATION_DATES
    assert not km.TEST_DATES & km.BURN_IN_DATES


def test_splits_are_disjoint_and_ordered() -> None:
    assert not km.BURN_IN_DATES & km.TRAIN_DATES
    assert not km.TRAIN_DATES & km.VALIDATION_DATES
    assert max(km.BURN_IN_DATES) < min(km.TRAIN_DATES)
    assert max(km.TRAIN_DATES) < min(km.VALIDATION_DATES)
    assert max(km.VALIDATION_DATES) < min(km.TEST_DATES)


def test_window_spans_match_the_frozen_contract() -> None:
    assert km.WINDOW_MS["1d"] == 86_400_000
    assert km.WINDOW_MS["7d"] == 604_800_000
    assert km.WINDOW_ORDER == ("1d", "7d", "all")
    assert km.GROUP_ORDER == ("user", "item", "author", "tab")
    assert len(km.STATISTIC_ORDER) == 9


def test_binary_parser_rejects_anything_else() -> None:
    assert km._require_binary("1", "x") == 1
    assert km._require_binary("0.0", "x") == 0
    with pytest.raises(km.MaterializationError, match="non-binary"):
        km._require_binary("2", "x")


def test_negative_numeric_contributions_are_clamped_to_zero() -> None:
    assert km._nonnegative_float("-5") == 0.0
    assert km._nonnegative_float("3.5") == 3.5


# ---------------------------------------------------------------------------
# Rolling windows — the part a leak would hide in
# ---------------------------------------------------------------------------


def _events(times, keys, values, updates=None):
    count = len(times)
    stats = np.zeros((count, 9), dtype=np.float64)
    stats[:, 0] = 1.0
    stats[:, 1] = values
    return (
        {"time_ms": np.asarray(times, dtype=np.int64), "stats": stats},
        np.asarray(keys),
        np.ones(count, dtype=bool) if updates is None else np.asarray(updates),
    )


def test_a_row_never_sees_itself() -> None:
    events, keys, updates = _events([100], ["u"], [7.0])
    windows = km.rolling_windows(events, keys, updates)
    for window in km.WINDOW_ORDER:
        assert windows[window][0, 0] == 0.0
        assert windows[window][0, 1] == 0.0


def test_equal_timestamps_are_invisible_to_each_other() -> None:
    """The contract scores the whole equal-timestamp group before updating it."""

    events, keys, updates = _events([100, 100, 100], ["u"] * 3, [1.0, 1.0, 1.0])
    windows = km.rolling_windows(events, keys, updates)
    assert list(windows["all"][:, 0]) == [0.0, 0.0, 0.0]


def test_strictly_earlier_rows_are_counted() -> None:
    events, keys, updates = _events([100, 200, 300], ["u"] * 3, [1.0, 2.0, 4.0])
    windows = km.rolling_windows(events, keys, updates)
    assert list(windows["all"][:, 0]) == [0.0, 1.0, 2.0]
    assert list(windows["all"][:, 1]) == [0.0, 1.0, 3.0]


def test_keys_do_not_leak_into_each_other() -> None:
    events, keys, updates = _events(
        [100, 200, 300, 400], ["a", "b", "a", "b"], [1.0, 10.0, 1.0, 10.0]
    )
    windows = km.rolling_windows(events, keys, updates)
    # Row 2 (key a) sees only row 0; row 3 (key b) sees only row 1.
    assert windows["all"][2, 1] == 1.0
    assert windows["all"][3, 1] == 10.0


def test_one_day_window_drops_older_rows() -> None:
    day = km.WINDOW_MS["1d"]
    events, keys, updates = _events(
        [0, day // 2, day + 1000], ["u"] * 3, [1.0, 1.0, 1.0]
    )
    windows = km.rolling_windows(events, keys, updates)
    # The last row is more than one day after the first, so only the middle
    # row remains inside its 1d window, while `all` still sees both.
    assert windows["1d"][2, 0] == 1.0
    assert windows["all"][2, 0] == 2.0


def test_seven_day_window_is_wider_than_one_day() -> None:
    day = km.WINDOW_MS["1d"]
    events, keys, updates = _events(
        [0, 2 * day, 3 * day], ["u"] * 3, [1.0, 1.0, 1.0]
    )
    windows = km.rolling_windows(events, keys, updates)
    # Row 2 sits at 3d. Its 1d window is [2d, 3d), which contains row 1 only;
    # its 7d window reaches back past row 0 as well.
    assert windows["1d"][2, 0] == 1.0
    assert windows["7d"][2, 0] == 2.0


def test_window_lower_bound_is_inclusive() -> None:
    """The contract declares `lower_inclusive_ms`, so t - span is inside."""

    day = km.WINDOW_MS["1d"]
    events, keys, updates = _events([0, day, day + 1], ["u"] * 3, [1.0, 1.0, 1.0])
    windows = km.rolling_windows(events, keys, updates)
    # Row 1 is exactly one day after row 0: inclusive lower bound keeps it.
    assert windows["1d"][1, 0] == 1.0
    # Row 2 is one millisecond later, which pushes row 0 out of the window.
    assert windows["1d"][2, 0] == 1.0


def test_non_updating_rows_never_enter_another_rows_history() -> None:
    """Embargoed and label-less rows receive features but contribute none."""

    events, keys, updates = _events(
        [100, 200, 300], ["u"] * 3, [1.0, 5.0, 1.0], updates=[True, False, True]
    )
    windows = km.rolling_windows(events, keys, updates)
    # Row 2 sees row 0 only; row 1 is excluded from the accumulator.
    assert windows["all"][2, 0] == 1.0
    assert windows["all"][2, 1] == 1.0
    # Row 1 still gets its own features from row 0.
    assert windows["all"][1, 0] == 1.0


def test_window_result_shape_and_finiteness() -> None:
    generator = np.random.default_rng(0)
    count = 500
    times = np.sort(generator.integers(0, 10_000_000, count))
    keys = generator.integers(0, 7, count).astype(str)
    events, key_array, updates = _events(times, keys, generator.random(count))
    windows = km.rolling_windows(events, key_array, updates)
    for window in km.WINDOW_ORDER:
        assert windows[window].shape == (count, 9)
        assert np.isfinite(windows[window]).all()
        assert (windows[window] >= 0).all()


def test_all_window_is_never_smaller_than_seven_day() -> None:
    generator = np.random.default_rng(1)
    count = 400
    times = np.sort(generator.integers(0, 50_000_000, count))
    keys = generator.integers(0, 3, count).astype(str)
    events, key_array, updates = _events(times, keys, generator.random(count))
    windows = km.rolling_windows(events, key_array, updates)
    assert (windows["all"] >= windows["7d"] - 1e-9).all()
    assert (windows["7d"] >= windows["1d"] - 1e-9).all()


# ---------------------------------------------------------------------------
# Freeze integrity: the emitted order must be the contract's order
# ---------------------------------------------------------------------------


def _contract() -> dict:
    path = REPO_ROOT / "configs" / "data" / "kuairand_pure" / "features_v1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_r_block_is_emitted_in_the_frozen_contract_order() -> None:
    """The emission loops are group-major; the contract is field-block-major.

    Nothing downstream indexes R positionally, so this changes no metric - but
    the frozen file is what an auditor reads, and an artifact whose column
    order silently disagrees with its own contract is not auditable.
    """

    contract = _contract()
    expected = contract["ordered_fields"]["R"]
    scrambled = list(reversed(expected))
    blocks = [np.full((2,), index, dtype=np.float32) for index in range(len(scrambled))]
    reordered_blocks, reordered_names = km._reorder_to_contract(
        blocks, scrambled, contract
    )
    assert reordered_names == expected
    # Each block must still travel with the name it was emitted under.
    for name, block in zip(reordered_names, reordered_blocks, strict=True):
        assert block[0] == scrambled.index(name)


def test_reorder_refuses_a_set_that_does_not_match_the_contract() -> None:
    contract = _contract()
    names = list(contract["ordered_fields"]["R"])[:-1]
    blocks = [np.zeros((2,), dtype=np.float32) for _ in names]
    with pytest.raises(km.MaterializationError, match="does not match the frozen"):
        km._reorder_to_contract(blocks, names, contract)


def test_contract_declares_the_full_cross_window_ratio_set() -> None:
    """R is 112 fields: 24 per-window ratios plus 4 cross-window, per group."""

    fields = _contract()["ordered_fields"]["R"]
    assert len(fields) == 112
    for group in km.GROUP_ORDER:
        cross = [
            f"r_{group}_exposure_1d_over_7d",
            f"r_{group}_exposure_7d_over_all",
            f"r_{group}_long_view_1d_over_7d",
            f"r_{group}_long_view_7d_over_all",
        ]
        for name in cross:
            assert name in fields, name
