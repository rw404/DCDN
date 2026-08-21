from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dcdn.data.vk_lsvd_multitask_runtime import (
    EXPECTED_CONTEXT_FIELDS,
    EXPECTED_TARGET_NAMES,
    MultiTargetRawBatch,
    frozen_target_spec,
)
from dcdn.multitask import TaskConstants, macro_nloss
from dcdn.multitask_training import MultiTargetTrainingHarness
from dcdn.training import TrainingContractError


def _batch(labels: torch.Tensor) -> MultiTargetRawBatch:
    rows = labels.shape[0]
    return MultiTargetRawBatch(
        context_tokens=torch.zeros(
            rows, len(EXPECTED_CONTEXT_FIELDS), dtype=torch.int64
        ),
        operand_features=torch.ones(rows, 3, dtype=torch.float32),
        denominator_validity=torch.zeros(rows, 3, dtype=torch.bool),
        targets=labels,
        target_masks=torch.ones(rows, len(EXPECTED_TARGET_NAMES), dtype=torch.bool),
        user_ids=torch.arange(rows, dtype=torch.int64),
        global_row_indices=torch.arange(rows, dtype=torch.int64),
    )


def test_microbatch_loss_accumulates_to_exact_effective_batch_macro_nloss() -> None:
    spec = frozen_target_spec()
    constants = TaskConstants(
        spec=spec,
        values=(0.2, 0.5, 0.4, 0.3, 0.2, 0.4, 0.5),
        observed_counts=(100,) * len(spec),
    )
    labels = torch.tensor(
        [
            [0.1, 1, 0, 0, 1, 0, 1],
            [0.2, 0, 0, 1, 0, 1, 0],
            [0.3, 1, 1, 0, 0, 0, 1],
            [0.4, 0, 0, 1, 1, 1, 0],
            [0.5, 1, 0, 0, 0, 1, 1],
            [0.6, 0, 1, 1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    logits = torch.linspace(-0.9, 0.9, labels.numel()).reshape_as(labels)
    full = macro_nloss(
        spec,
        constants,
        logits,
        labels,
        torch.ones_like(labels, dtype=torch.bool),
    ).aggregate
    harness = object.__new__(MultiTargetTrainingHarness)
    harness.task_constants = constants
    harness.config = SimpleNamespace(loss="macro_nloss")
    accumulated = (
        harness._loss_sum(logits[:2], _batch(labels[:2]))
        + harness._loss_sum(logits[2:], _batch(labels[2:]))
    ) / labels.shape[0]
    assert float(accumulated) == pytest.approx(float(full), abs=1e-7)


def test_loss_rejects_missing_vk_target_mask() -> None:
    spec = frozen_target_spec()
    constants = TaskConstants(
        spec=spec,
        values=(0.2, 0.5, 0.4, 0.3, 0.2, 0.4, 0.5),
        observed_counts=(100,) * len(spec),
    )
    harness = object.__new__(MultiTargetTrainingHarness)
    harness.task_constants = constants
    harness.config = SimpleNamespace(loss="macro_nloss")
    labels = torch.zeros(2, len(spec))
    batch = _batch(labels)
    batch.target_masks[0, 0] = False
    with pytest.raises(TrainingContractError, match="always-observed"):
        harness._loss_sum(torch.zeros_like(labels), batch)
