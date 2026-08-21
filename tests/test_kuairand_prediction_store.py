from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "krm_store", REPO_ROOT / "scripts" / "run_kuairand_multitask.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["krm_store"] = module
    spec.loader.exec_module(module)
    return module


krm = _load()


def test_predictions_are_float32_not_half(tmp_path: Path) -> None:
    """X-AUC counts tied predictions, so half precision would invent ties."""

    store = krm.PredictionStore(tmp_path, "study")
    store.add("dcdn_reduced", 42, np.array([[0.1, 0.2]], dtype=np.float64))
    assert store.arrays["pred__dcdn_reduced__42"].dtype == np.float32


def test_round_trip_preserves_every_arm_and_seed(tmp_path: Path) -> None:
    store = krm.PredictionStore(tmp_path, "study")
    expected = {}
    for arm in ("dcdn_reduced", "dcnv2_full"):
        for seed in (42, 43):
            block = np.random.default_rng(seed).random((5, 3)).astype(np.float32)
            store.add(arm, seed, block)
            expected[f"pred__{arm}__{seed}"] = block
    digest = store.write()
    assert len(digest) == 64

    loaded = np.load(store.path)
    for key, block in expected.items():
        np.testing.assert_array_equal(loaded[key], block)


def test_write_is_atomic_leaving_no_partial_file(tmp_path: Path) -> None:
    """A crashed write must not leave a truncated npz where a valid one was."""

    store = krm.PredictionStore(tmp_path, "study")
    store.add("a", 42, np.zeros((2, 2), dtype=np.float32))
    store.write()
    first = store.path.read_bytes()

    second = krm.PredictionStore(tmp_path, "study")
    second.add("a", 42, np.ones((2, 2), dtype=np.float32))
    second.write()
    assert store.path.read_bytes() != first
    # No temporary files survive either write.
    assert sorted(p.name for p in store.path.parent.iterdir()) == ["study.npz"]


def test_store_writes_outside_the_repository(tmp_path: Path) -> None:
    """Predictions are hundreds of megabytes; they must never enter git."""

    store = krm.PredictionStore(tmp_path, "study")
    assert not store.path.is_relative_to(REPO_ROOT)


def test_digest_is_stable_for_identical_content(tmp_path: Path) -> None:
    blocks = {"a": np.arange(6, dtype=np.float32).reshape(3, 2)}
    digests = []
    for name in ("one", "two"):
        store = krm.PredictionStore(tmp_path, name)
        for arm, block in blocks.items():
            store.add(arm, 42, block)
        digests.append(store.write())
    assert digests[0] == digests[1]


def test_context_carries_what_a_recomputation_needs(tmp_path: Path) -> None:
    """Labels, masks, user ids and target names, or a metric cannot be redone."""

    import torch

    class FakeViews:
        data = {
            "validation": {
                "labels": torch.zeros((4, 2)),
                "masks": torch.ones((4, 2), dtype=torch.bool),
                "categorical": torch.arange(8).reshape(4, 2),
            }
        }

    class FakeSpec:
        names = ("long_view", "watch_time")

    store = krm.PredictionStore(tmp_path, "study")
    store.add_context(FakeViews(), FakeSpec())
    assert set(store.arrays) == {"labels", "masks", "user_ids", "target_names"}
    assert store.arrays["user_ids"].tolist() == [0, 2, 4, 6]


def test_flag_defaults_to_off() -> None:
    """The two runs already in flight were launched without it.

    If the default were on, the code in the repository would no longer be the
    code that produced those bundles.
    """

    parser_source = (
        REPO_ROOT / "scripts" / "run_kuairand_multitask.py"
    ).read_text(encoding="utf-8")
    assert '"--save-predictions",\n        action="store_true"' in parser_source
    assert "store = None" in parser_source
