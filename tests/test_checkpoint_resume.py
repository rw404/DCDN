from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from dcdn.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    CheckpointMetadata,
    load_checkpoint,
    load_model_checkpoint_for_evaluation,
    save_checkpoint,
)
from dcdn.reproducibility import seed_everything
from dcdn.sampler import SamplerStateError, StatefulPermutationSampler

DATASET_SIZE = 32
BATCH_SIZE = 4
SAMPLER_SEED = 919


def _synthetic_train_data() -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(DATASET_SIZE * 4, dtype=torch.float32).reshape(
        DATASET_SIZE, 4
    )
    features = values / 31.0 - 1.5
    targets = (
        0.3 * features[:, 0]
        - 0.2 * features[:, 1]
        + 0.1 * features[:, 2].square()
        + 0.05 * features[:, 3]
    ).unsqueeze(-1)
    return features, targets


FEATURES, TARGETS = _synthetic_train_data()


class DropoutBatchNormToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(4, 8)
        self.batch_norm = nn.BatchNorm1d(8, momentum=0.2)
        self.dropout = nn.Dropout(0.35)
        self.output = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.batch_norm(self.input(inputs)))
        return self.output(self.dropout(hidden))


class ContractDropoutBatchNormToy(DropoutBatchNormToy):
    """Toy whose non-tensor extra state behaves like joint encoder provenance."""

    def __init__(self, contract_id: str) -> None:
        super().__init__()
        self.contract_id = contract_id

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "nested": {"ordered_fields": ["context", "operand"]},
        }

    def set_extra_state(self, state: Any) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("contract mismatch reached live mutation")


class DummyScaler:
    """Small state_dict-compatible stand-in; no AMP kernels are used in tests."""

    def __init__(self) -> None:
        self.scale = 8.0
        self.growth_tracker = 0

    def update(self) -> None:
        self.growth_tracker += 1
        if self.growth_tracker % 2 == 0:
            self.scale *= 2.0

    def state_dict(self) -> dict[str, float | int]:
        return {"scale": self.scale, "growth_tracker": self.growth_tracker}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.scale = float(state["scale"])
        self.growth_tracker = int(state["growth_tracker"])


@dataclass
class TrainingBundle:
    model: DropoutBatchNormToy
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    sampler: StatefulPermutationSampler
    scaler: DummyScaler


def _new_bundle(seed: int) -> TrainingBundle:
    # This test is intentionally CPU-only even when the host exposes GPUs.
    seed_everything(seed, seed_cuda=False)
    model = DropoutBatchNormToy()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    sampler = StatefulPermutationSampler(DATASET_SIZE, seed=SAMPLER_SEED, shuffle=True)
    return TrainingBundle(model, optimizer, scheduler, sampler, DummyScaler())


def _new_contract_bundle(seed: int, contract_id: str) -> TrainingBundle:
    seed_everything(seed, seed_cuda=False)
    model = ContractDropoutBatchNormToy(contract_id)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    sampler = StatefulPermutationSampler(DATASET_SIZE, seed=SAMPLER_SEED, shuffle=True)
    return TrainingBundle(model, optimizer, scheduler, sampler, DummyScaler())


def _metadata(**overrides: Any) -> CheckpointMetadata:
    values: dict[str, Any] = {
        "resolved_config": {
            "model": {"name": "dropout-bn-toy", "depth": 2},
            "optimizer": {"name": "sgd", "lr": 0.05},
            "accumulation_steps": 1,
        },
        "data_hash": "a" * 64,
        "feature_hash": "b" * 64,
        "protocol_hash": "c" * 64,
        "feature_order": ("numerator", "denominator", "context"),
        "denominator_mask": (False, True, False),
        "div_eps": 1e-2,
        "depth": 2,
        "precision": "fp32",
    }
    values.update(overrides)
    return CheckpointMetadata(**values)


def _run_steps(
    bundle: TrainingBundle, steps: int
) -> tuple[list[list[int]], list[float]]:
    bundle.model.train()
    row_trace: list[list[int]] = []
    loss_trace: list[float] = []
    for _ in range(steps):
        row_ids = list(islice(iter(bundle.sampler), BATCH_SIZE))
        if len(row_ids) != BATCH_SIZE:
            raise AssertionError("synthetic test unexpectedly exhausted its epoch")

        # Python/NumPy perturbations verify those RNG streams in addition to
        # Torch RNG exercised by Dropout.
        jitter = random.random() * 1e-3 + float(np.random.random()) * 1e-3
        inputs = FEATURES[row_ids] + jitter
        targets = TARGETS[row_ids]

        bundle.optimizer.zero_grad(set_to_none=True)
        predictions = bundle.model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        bundle.optimizer.step()
        bundle.scheduler.step()
        bundle.scaler.update()

        # Durable progress advances only after the complete successful update.
        bundle.sampler.commit(row_ids)
        row_trace.append(row_ids)
        loss_trace.append(loss.detach().item())
    return row_trace, loss_trace


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list | tuple):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=False):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def _bundle_state(bundle: TrainingBundle) -> dict[str, Any]:
    return copy.deepcopy(
        {
            "model": bundle.model.state_dict(),
            "optimizer": bundle.optimizer.state_dict(),
            "scheduler": bundle.scheduler.state_dict(),
            "sampler": bundle.sampler.state_dict(),
            "scaler": bundle.scaler.state_dict(),
            "model_training": bundle.model.training,
        }
    )


def _assert_bundle_state(bundle: TrainingBundle, expected: dict[str, Any]) -> None:
    _assert_nested_equal(expected["model"], bundle.model.state_dict())
    _assert_nested_equal(expected["optimizer"], bundle.optimizer.state_dict())
    _assert_nested_equal(expected["scheduler"], bundle.scheduler.state_dict())
    _assert_nested_equal(expected["sampler"], bundle.sampler.state_dict())
    _assert_nested_equal(expected["scaler"], bundle.scaler.state_dict())
    assert bundle.model.training is expected["model_training"]


def _rng_state() -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    numpy_state = np.random.get_state()
    copied_numpy = (
        numpy_state[0],
        numpy_state[1].copy(),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    )
    return copy.deepcopy(random.getstate()), copied_numpy, torch.get_rng_state().clone()


def _assert_rng_state(expected: tuple[Any, tuple[Any, ...], torch.Tensor]) -> None:
    python_state, numpy_state, torch_state = expected
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    assert np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def _rewrite_checkpoint(path: Path, mutate: Any) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def _save_training_checkpoint(
    path: Path,
    bundle: TrainingBundle,
    *,
    global_step: int,
    metadata: CheckpointMetadata | None = None,
):
    return save_checkpoint(
        path,
        model=bundle.model,
        optimizer=bundle.optimizer,
        scheduler=bundle.scheduler,
        sampler=bundle.sampler,
        scaler=bundle.scaler,
        global_step=global_step,
        epoch=bundle.sampler.epoch,
        accumulation_step=0,
        accumulation_steps=1,
        metadata=metadata or _metadata(),
        mode="train",
        extra_state={"split": "synthetic-train-only"},
        include_cuda_rng=False,
    )


def _load_training_checkpoint(
    path: Path,
    bundle: TrainingBundle,
    *,
    metadata: CheckpointMetadata | None = None,
):
    return load_checkpoint(
        path,
        model=bundle.model,
        optimizer=bundle.optimizer,
        scheduler=bundle.scheduler,
        sampler=bundle.sampler,
        scaler=bundle.scaler,
        expected_metadata=metadata or _metadata(),
        expected_accumulation_steps=1,
    )


def test_exact_mid_epoch_resume_matches_uninterrupted_cpu_training(
    tmp_path: Path,
) -> None:
    total_steps = 7
    checkpoint_step = 3

    reference = _new_bundle(seed=1234)
    reference_rows, reference_losses = _run_steps(reference, total_steps)

    interrupted = _new_bundle(seed=1234)
    first_rows, first_losses = _run_steps(interrupted, checkpoint_step)
    checkpoint_path = tmp_path / "last.pt"
    record = _save_training_checkpoint(
        checkpoint_path, interrupted, global_step=checkpoint_step
    )
    assert record.path == checkpoint_path.absolute()
    assert record.sidecar_path.read_text(encoding="utf-8").startswith(record.sha256)
    assert record.size_bytes == checkpoint_path.stat().st_size

    resumed = _new_bundle(seed=9999)
    resumed.model.eval()
    resume_state = _load_training_checkpoint(checkpoint_path, resumed)
    assert resume_state.global_step == checkpoint_step
    assert resume_state.epoch == 0
    assert resume_state.accumulation_step == 0
    assert resume_state.accumulation_steps == 1
    assert resume_state.mode == "train"
    assert resume_state.sha256 == record.sha256
    assert resume_state.extra_state == {"split": "synthetic-train-only"}
    assert resumed.model.training
    assert resumed.sampler.cursor == checkpoint_step * BATCH_SIZE

    remaining_rows, remaining_losses = _run_steps(
        resumed, total_steps - checkpoint_step
    )
    assert first_rows + remaining_rows == reference_rows
    assert first_losses + remaining_losses == reference_losses

    _assert_nested_equal(reference.model.state_dict(), resumed.model.state_dict())
    _assert_nested_equal(
        reference.optimizer.state_dict(), resumed.optimizer.state_dict()
    )
    _assert_nested_equal(
        reference.scheduler.state_dict(), resumed.scheduler.state_dict()
    )
    _assert_nested_equal(reference.scaler.state_dict(), resumed.scaler.state_dict())
    _assert_nested_equal(reference.sampler.state_dict(), resumed.sampler.state_dict())


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        (
            "resolved_config",
            {
                "model": {"name": "dropout-bn-toy", "depth": 2},
                "optimizer": {"name": "sgd", "lr": 0.01},
                "accumulation_steps": 1,
            },
        ),
        ("feature_order", ("denominator", "numerator", "context")),
        ("denominator_mask", (True, False, False)),
        ("div_eps", 1e-3),
        ("depth", 3),
        ("precision", "bf16"),
        ("protocol_hash", "d" * 64),
        ("feature_hash", "e" * 64),
        ("data_hash", "f" * 64),
    ],
)
def test_metadata_mismatch_rejected_before_mutation(
    tmp_path: Path, field: str, wrong_value: Any
) -> None:
    source = _new_bundle(seed=11)
    _run_steps(source, 2)
    checkpoint_path = tmp_path / "strict.pt"
    _save_training_checkpoint(checkpoint_path, source, global_step=2)

    target = _new_bundle(seed=22)
    target.model.eval()
    model_before = {
        key: value.clone() for key, value in target.model.state_dict().items()
    }
    optimizer_before = target.optimizer.state_dict()
    sampler_before = target.sampler.state_dict()

    with pytest.raises(CheckpointCompatibilityError, match="metadata mismatch"):
        _load_training_checkpoint(
            checkpoint_path, target, metadata=_metadata(**{field: wrong_value})
        )

    _assert_nested_equal(model_before, target.model.state_dict())
    _assert_nested_equal(optimizer_before, target.optimizer.state_dict())
    _assert_nested_equal(sampler_before, target.sampler.state_dict())
    assert not target.model.training


def test_corrupted_checkpoint_hash_rejected_before_mutation(tmp_path: Path) -> None:
    source = _new_bundle(seed=31)
    _run_steps(source, 1)
    checkpoint_path = tmp_path / "corrupt.pt"
    _save_training_checkpoint(checkpoint_path, source, global_step=1)

    content = bytearray(checkpoint_path.read_bytes())
    content[len(content) // 2] ^= 0x01
    checkpoint_path.write_bytes(content)

    target = _new_bundle(seed=32)
    model_before = {
        key: value.clone() for key, value in target.model.state_dict().items()
    }
    sampler_before = target.sampler.state_dict()
    with pytest.raises(CheckpointIntegrityError, match="SHA-256 mismatch"):
        _load_training_checkpoint(checkpoint_path, target)
    _assert_nested_equal(model_before, target.model.state_dict())
    _assert_nested_equal(sampler_before, target.sampler.state_dict())


def test_checkpoint_rejects_mid_accumulation_save(tmp_path: Path) -> None:
    bundle = _new_bundle(seed=41)
    with pytest.raises(CheckpointCompatibilityError, match="accumulation boundary"):
        save_checkpoint(
            tmp_path / "unsafe.pt",
            model=bundle.model,
            optimizer=bundle.optimizer,
            scheduler=bundle.scheduler,
            sampler=bundle.sampler,
            scaler=bundle.scaler,
            global_step=0,
            epoch=0,
            accumulation_step=1,
            accumulation_steps=2,
            metadata=_metadata(),
            mode="train",
            include_cuda_rng=False,
        )
    assert not (tmp_path / "unsafe.pt").exists()


def test_sampler_state_is_exact_and_invalid_permutation_is_non_mutating() -> None:
    sampler = StatefulPermutationSampler(13, seed=17)
    consumed = list(islice(iter(sampler), 5))
    sampler.commit(consumed)
    state = sampler.state_dict()

    restored = StatefulPermutationSampler(13, seed=17)
    restored.load_state_dict(state)
    assert restored.cursor == 5
    assert list(iter(restored)) == state["permutation"][5:].tolist()

    invalid = dict(state)
    invalid["permutation"] = state["permutation"].clone()
    invalid["permutation"][[0, 1]] = invalid["permutation"][[1, 0]]
    cursor_before = restored.cursor
    with pytest.raises(SamplerStateError, match="deterministic seed/epoch"):
        restored.load_state_dict(invalid)
    assert restored.cursor == cursor_before


def test_module_extra_state_mismatch_rejected_before_any_live_mutation(
    tmp_path: Path,
) -> None:
    source = _new_contract_bundle(seed=51, contract_id="source-contract")
    _run_steps(source, 1)
    checkpoint_path = tmp_path / "joint-contract.pt"
    _save_training_checkpoint(checkpoint_path, source, global_step=1)

    target = _new_contract_bundle(seed=52, contract_id="different-live-contract")
    target.model.eval()
    before = _bundle_state(target)
    rng_before = _rng_state()
    with pytest.raises(CheckpointCompatibilityError, match="model extra state"):
        _load_training_checkpoint(checkpoint_path, target)
    _assert_bundle_state(target, before)
    _assert_rng_state(rng_before)


def test_caller_extra_state_validator_runs_before_any_live_mutation(
    tmp_path: Path,
) -> None:
    source = _new_bundle(seed=61)
    _run_steps(source, 1)
    checkpoint_path = tmp_path / "harness-extra.pt"
    _save_training_checkpoint(checkpoint_path, source, global_step=1)

    target = _new_bundle(seed=62)
    target.model.eval()
    before = _bundle_state(target)
    rng_before = _rng_state()

    def reject_harness_state(extra_state: Any, progress: Any) -> None:
        assert extra_state == {"split": "synthetic-train-only"}
        assert progress.global_step == 1
        raise ValueError("harness counter mismatch")

    with pytest.raises(CheckpointCompatibilityError, match="extra_state was rejected"):
        load_checkpoint(
            checkpoint_path,
            model=target.model,
            optimizer=target.optimizer,
            scheduler=target.scheduler,
            sampler=target.sampler,
            scaler=target.scaler,
            expected_metadata=_metadata(),
            expected_accumulation_steps=1,
            extra_state_validator=reject_harness_state,
        )
    _assert_bundle_state(target, before)
    _assert_rng_state(rng_before)


@pytest.mark.parametrize("component", ["optimizer", "scheduler", "scaler", "sampler"])
def test_component_incompatibility_is_rejected_before_any_live_mutation(
    tmp_path: Path,
    component: str,
) -> None:
    source = _new_bundle(seed=71)
    _run_steps(source, 2)
    checkpoint_path = tmp_path / f"bad-{component}.pt"
    _save_training_checkpoint(checkpoint_path, source, global_step=2)

    def corrupt(payload: Any) -> None:
        components = payload["components"]
        if component == "optimizer":
            components["optimizer"]["state"]["param_groups"][0]["lr"] = "invalid"
        elif component == "scheduler":
            components["scheduler"]["state"]["_step_count"] = "invalid"
        elif component == "scaler":
            components["scaler"]["state"]["scale"] = "invalid"
        else:
            permutation = components["sampler"]["state"]["permutation"]
            permutation[[0, 1]] = permutation[[1, 0]]

    _rewrite_checkpoint(checkpoint_path, corrupt)
    target = _new_bundle(seed=72)
    target.model.eval()
    before = _bundle_state(target)
    rng_before = _rng_state()
    with pytest.raises(CheckpointCompatibilityError):
        _load_training_checkpoint(checkpoint_path, target)
    _assert_bundle_state(target, before)
    _assert_rng_state(rng_before)


def test_model_only_evaluation_restore_leaves_training_state_and_rng_untouched(
    tmp_path: Path,
) -> None:
    source = _new_bundle(seed=81)
    _run_steps(source, 3)
    checkpoint_path = tmp_path / "best.pt"
    record = _save_training_checkpoint(checkpoint_path, source, global_step=3)

    target = _new_bundle(seed=82)
    _run_steps(target, 1)
    target.model.eval()
    optimizer_before = copy.deepcopy(target.optimizer.state_dict())
    scheduler_before = copy.deepcopy(target.scheduler.state_dict())
    sampler_before = copy.deepcopy(target.sampler.state_dict())
    scaler_before = copy.deepcopy(target.scaler.state_dict())
    rng_before = _rng_state()

    state = load_model_checkpoint_for_evaluation(
        checkpoint_path,
        model=target.model,
        expected_metadata=_metadata(),
    )

    _assert_nested_equal(source.model.state_dict(), target.model.state_dict())
    _assert_nested_equal(optimizer_before, target.optimizer.state_dict())
    _assert_nested_equal(scheduler_before, target.scheduler.state_dict())
    _assert_nested_equal(sampler_before, target.sampler.state_dict())
    _assert_nested_equal(scaler_before, target.scaler.state_dict())
    _assert_rng_state(rng_before)
    assert not target.model.training
    assert state.sha256 == record.sha256
    assert state.metadata == _metadata()
    assert state.progress.global_step == 3
    assert state.progress.mode == "train"
    assert state.progress.model_training is True
    assert state.extra_state == {"split": "synthetic-train-only"}


def test_model_only_extra_rejection_is_non_mutating(tmp_path: Path) -> None:
    source = _new_bundle(seed=91)
    _run_steps(source, 1)
    checkpoint_path = tmp_path / "rejected-best.pt"
    _save_training_checkpoint(checkpoint_path, source, global_step=1)

    target = _new_bundle(seed=92)
    target.model.eval()
    before = _bundle_state(target)
    rng_before = _rng_state()

    def reject(_extra_state: Any, _progress: Any) -> None:
        raise ValueError("wrong parent-best harness state")

    with pytest.raises(CheckpointCompatibilityError, match="extra_state was rejected"):
        load_model_checkpoint_for_evaluation(
            checkpoint_path,
            model=target.model,
            expected_metadata=_metadata(),
            extra_state_validator=reject,
        )
    _assert_bundle_state(target, before)
    _assert_rng_state(rng_before)
