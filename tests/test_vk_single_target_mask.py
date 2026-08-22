"""The VK adapter's reciprocal mask must come from the contract, not the data.

The regression these cover: deriving the mask from `denominator_validity` -
"a coordinate is a denominator when the view gates it on some row" - selected
one coordinate in REDUCED and eighteen in FULL, the latter including every
R_div ratio. `features_multitask_v1.yaml` forbids exactly that.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("run_vk_single_target", REPO_ROOT / "scripts" / "run_vk_single_target.py")
builder = runner.views_builder

REDUCED_FIELDS = tuple(builder.ORDERED_OPERAND_FIELDS)
FULL_FIELDS = REDUCED_FIELDS + tuple(builder.R_DIV_FIELDS)

#: Recorded by the materializer in each view's `cache.json`.
REDUCED_SHA = "186ec50742dcd32b0a823fcfcd5be52bafa9427c7e95a516bec59a0db8bfa5ab"
FULL_SHA = "d86a1590c2ae06f220d8e72dbaf2cd696ebdb24a930f5e1ed623a30aab240d5a"


def test_eligible_set_is_five_historical_denominators() -> None:
    assert runner.ELIGIBLE_DENOMINATORS == (
        "p_duration_seconds",
        "p_user_show_count",
        "p_user_show_count_plus_beta",
        "p_item_show_count",
        "p_item_show_count_plus_beta",
    )


def test_masks_reproduce_the_hashes_the_views_recorded() -> None:
    reduced = runner.contract_denominator_mask(REDUCED_FIELDS, REDUCED_SHA)
    full = runner.contract_denominator_mask(FULL_FIELDS, FULL_SHA)
    assert sum(reduced) == 5
    assert sum(full) == 5
    assert len(reduced) == 55
    assert len(full) == 72


def test_the_two_regimes_gate_the_same_coordinates() -> None:
    reduced = runner.contract_denominator_mask(REDUCED_FIELDS, REDUCED_SHA)
    full = runner.contract_denominator_mask(FULL_FIELDS, FULL_SHA)
    assert full[: len(reduced)] == reduced
    assert not any(full[len(reduced):])


def test_no_engineered_ratio_similarity_or_norm_is_a_denominator() -> None:
    full = runner.contract_denominator_mask(FULL_FIELDS, FULL_SHA)
    forbidden = {"cosine", "dot", "norm"}
    for field, gated in zip(FULL_FIELDS, full, strict=True):
        if gated:
            assert field in runner.ELIGIBLE_DENOMINATORS
            assert not any(token in field for token in forbidden)
    for field in builder.R_DIV_FIELDS:
        assert not full[FULL_FIELDS.index(field)]


def test_a_wrong_mask_is_rejected_rather_than_used() -> None:
    with pytest.raises(SystemExit, match="does not reproduce"):
        runner.contract_denominator_mask(REDUCED_FIELDS, FULL_SHA)
