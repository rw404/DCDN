"""Seven-head extension of the deterministic public-data training harness."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from dcdn.data.vk_lsvd_multitask_runtime import (
    MultiTargetRawBatch,
    MultiTargetSourceIdentity,
    VKLSVDMultiTargetSource,
)
from dcdn.multitask import TaskConstants, macro_nloss
from dcdn.multitask_evaluation import (
    MultiTargetEvaluation,
    evaluate_multitarget_predictions,
)
from dcdn.multitask_model import MultiTargetTrainableEncodedModel
from dcdn.training import (
    Split,
    SplitAccessError,
    TrainingContractError,
    TrainingHarness,
)


class MultiTargetTrainingHarness(TrainingHarness):
    """Reuse checkpoint/resume/sampler safety with masked seven-head loss."""

    def __init__(
        self,
        *,
        task_constants: TaskConstants,
        regression_scales: Mapping[str, float],
        **kwargs: Any,
    ) -> None:
        model = kwargs.get("model")
        if not isinstance(model, MultiTargetTrainableEncodedModel):
            raise TypeError("multi-target harness requires the joint multi-head model")
        if not isinstance(task_constants, TaskConstants):
            raise TypeError("task_constants must use the frozen immutable type")
        if model.target_spec != task_constants.spec:
            raise TrainingContractError("model and train task constants specs differ")
        expected_regression = {
            target.name
            for target in model.target_spec.targets
            if target.kind == "regression"
        }
        if set(regression_scales) != expected_regression:
            raise TrainingContractError(
                "regression scales must match regression targets exactly"
            )
        self.task_constants = task_constants
        self.regression_scales = dict(regression_scales)
        super().__init__(**kwargs)
        for identity in (
            self.safety.train_source_identity,
            self.safety.validation_source_identity,
        ):
            if not isinstance(identity, MultiTargetSourceIdentity):
                raise TrainingContractError(
                    "multi-target safety requires the complete source identity"
                )
            if identity.task_constants_sha256 != task_constants.contract_sha256:
                raise TrainingContractError(
                    "source identity and train task constants differ"
                )

    def _resolved_checkpoint_config(self) -> dict[str, Any]:
        payload = super()._resolved_checkpoint_config()
        payload["multi_target_objective"] = {
            "target_spec_sha256": self.task_constants.spec.contract_sha256,
            "target_names": list(self.task_constants.spec.names),
            "target_kinds": list(self.task_constants.spec.kinds),
            "task_constants_sha256": self.task_constants.contract_sha256,
            "task_constants": self.task_constants.contract_state(),
            "loss": "macro_nloss",
            "task_weights": "all_exactly_one",
            "mask_policy": "observed_labels_only",
            "regression_scales": self.regression_scales,
        }
        return payload

    def _batch_forward(
        self,
        source: Any,
        indices: Sequence[int],
        *,
        split: Split,
    ) -> tuple[Any, MultiTargetRawBatch]:
        if not isinstance(source, VKLSVDMultiTargetSource):
            raise TrainingContractError("multi-target source type drifted")
        if not isinstance(self.model, MultiTargetTrainableEncodedModel):
            raise TrainingContractError("multi-target model type drifted")
        self._assert_raw_source(source, split)
        batch = source.batch(indices, device=self.device)
        batch.validate(
            source.identity,
            indices,
            runtime_value_checks=source.runtime_value_checks,
        )
        before_hooks = self.monitor.activation_hook_observations
        with self._autocast():
            output = self.model(batch)
        self.monitor.assert_activation_hooks_observed(before_hooks)
        return output, batch

    def _loss_sum(
        self, logits: torch.Tensor, supervision: MultiTargetRawBatch
    ) -> torch.Tensor:
        if not isinstance(supervision, MultiTargetRawBatch):
            raise TrainingContractError("multi-target supervision batch drifted")
        if self.config.loss != "macro_nloss":
            raise TrainingContractError("multi-target harness requires Macro-NLoss")
        if not bool(supervision.target_masks.all()):
            raise TrainingContractError(
                "VK v2 exact accumulation requires always-observed target masks"
            )
        objective = macro_nloss(
            self.task_constants.spec,
            self.task_constants,
            logits,
            supervision.targets,
            supervision.target_masks,
        )
        # Every VK-v2 target is observed on every row.  Multiplying the
        # microbatch mean by its row count makes gradient accumulation exactly
        # equal to one effective-batch Macro-NLoss evaluation.
        return objective.aggregate * supervision.targets.shape[0]

    def evaluate_report(
        self,
        source_factory: Callable[[], VKLSVDMultiTargetSource],
        *,
        batch_size: int | None = None,
    ) -> MultiTargetEvaluation:
        """Run one complete validation pass and compute mandatory user-GAUC."""

        size = batch_size or self.current_microbatch
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("evaluation batch_size must be positive")
        source = self._open_raw_source("validation", source_factory)
        if not isinstance(source, VKLSVDMultiTargetSource):
            raise SplitAccessError("validation source is not multi-target")
        rows = len(source)
        targets = len(self.task_constants.spec)
        logits = np.empty((rows, targets), dtype=np.float32)
        labels = np.empty((rows, targets), dtype=np.float32)
        masks = np.empty((rows, targets), dtype=np.bool_)
        user_ids = np.empty(rows, dtype=np.int64)
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, rows, size):
                    stop = min(start + size, rows)
                    indices = list(range(start, stop))
                    output, batch = self._batch_forward(
                        source, indices, split="validation"
                    )
                    self.monitor.submit(
                        "validation.output", output.logits, kind="output"
                    )
                    self.monitor.submit(
                        "validation.representation",
                        output.representation.values,
                        kind="activation",
                    )
                    logits[start:stop] = output.logits.float().cpu().numpy()
                    labels[start:stop] = batch.targets.float().cpu().numpy()
                    masks[start:stop] = batch.target_masks.cpu().numpy()
                    user_ids[start:stop] = batch.user_ids.numpy()
                    self.monitor.assert_healthy()
            self.monitor.drain()
        finally:
            self.model.train()
            self._close_raw_source(source)
        report = evaluate_multitarget_predictions(
            spec=self.task_constants.spec,
            constants=self.task_constants,
            logits=logits,
            labels=labels,
            masks=masks,
            user_ids=user_ids,
            regression_scales=self.regression_scales,
        )
        if not math.isfinite(report.macro_nloss):
            raise TrainingContractError("multi-target validation report is non-finite")
        self.event_log.emit(
            "multitarget_validation_report",
            split="validation",
            rows=rows,
            report=report.as_report(),
            user_identifiers_published=False,
        )
        return report


__all__ = ["MultiTargetTrainingHarness"]
