from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from dcdn.final_evaluation import (
    FinalEvaluationError,
    recover_published_test_bundle,
    run_one_pass_test_bundle,
)
from dcdn.protocol import (
    BASE_HASH_KEYS,
    ProtocolGate,
    RuntimeIdentity,
    canonical_split_sha256,
    verify_sha256_sidecar,
)
from dcdn.protocol import (
    TestEvaluationClaim as EvaluationClaim,
)
from dcdn.raw import RawAuditBatch, RawFeatureBatch, RawSourceIdentity

ROWS = 6
MODEL_KEY = "dcdn_core"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_hashed(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _hashes() -> dict[str, str]:
    values = {name: _digest(name) for name in BASE_HASH_KEYS if name != "split_sha256"}
    values["split_sha256"] = canonical_split_sha256(
        {"train_weeks": [20, 21, 22, 23, 24], "validation_week": 25, "test": None}
    )
    return values


def _protocol(
    *,
    hashes: dict[str, str],
    unlocked: bool,
    parent: str | None = None,
    selected: str | None = None,
) -> dict[str, Any]:
    bound_hashes = dict(hashes)
    if unlocked:
        assert selected is not None
        bound_hashes["selected_configs_sha256"] = selected
        bound_hashes["test_views_manifest_sha256"] = _digest("test-views")
        bound_hashes["test_full_materialized_scan_sha256"] = _digest("test-scan")
        bound_hashes["test_cache_content_sha256"] = _digest("test-cache")
    return {
        "schema_version": 3,
        "dataset_id": "synthetic-final",
        "feature_regime": "C+P",
        "required_model_keys": [MODEL_KEY],
        "parent_premetric_protocol_sha256": parent,
        "test_unlocked": unlocked,
        "test_path_available": unlocked,
        "hashes": bound_hashes,
        "split_row_counts": {
            "train": 8,
            "validation": 4,
            "test": ROWS if unlocked else None,
        },
        "final_seeds": [42, 43, 44],
        "tuning_seed": 42,
        "tuning_splits": ["train", "validation"],
        "test_evaluations_per_final_seed": 1,
    }


@dataclass(frozen=True)
class Prepared:
    gate: ProtocolGate
    selected_path: Path
    claims_path: Path
    runtime: RuntimeIdentity
    raw_identity: RawSourceIdentity
    candidate_id: str
    resolved_config: str
    resolved_run: str
    checkpoint: str


def _prepare(root: Path) -> Prepared:
    hashes = _hashes()
    active_path = root / "active_protocol.yaml"
    parent_path = root / "protocol_frozen.yaml"
    parent_digest = _write_hashed(
        parent_path,
        _protocol(hashes=hashes, unlocked=False),
    )
    parent_gate = ProtocolGate.load(
        parent_path,
        active_registry_path=active_path,
    )
    parent_gate.activate()

    candidate = "dcdn_core_l2_lr_1e_3"
    resolved_config = _digest("resolved-model")
    resolved_run = _digest("resolved-run-42")
    checkpoint = _digest("checkpoint-42")
    selected_path = root / "selected_configs.lock"
    selected_payload = {
        "schema_version": 2,
        "artifact_role": "final_selected_configs_lock",
        "dataset_id": parent_gate.dataset_id,
        "feature_regime": parent_gate.feature_regime,
        "parent_premetric_protocol_sha256": parent_digest,
        "search_space_sha256": hashes["search_space_sha256"],
        "experiment_matrix_sha256": hashes["experiment_matrix_sha256"],
        "encoder_contract_sha256": hashes["encoder_contract_sha256"],
        "capacity_resolution_sha256": hashes["capacity_resolution_sha256"],
        "required_model_keys": [MODEL_KEY],
        "selected_models": {
            MODEL_KEY: {
                "candidate_id": candidate,
                "resolved_config_sha256": resolved_config,
                "selection_receipt_path": "results/tuning/dcdn-selection.json",
                "selection_receipt_sha256": _digest("selection"),
                "selected_on_validation_only": True,
                "test_data_accessed": False,
                "test_metrics_consulted": False,
                "final_seed_lineage": {
                    str(seed): {
                        "resolved_run_config_path": f"configs/runs/dcdn-s{seed}.yaml",
                        "resolved_run_config_sha256": (
                            resolved_run if seed == 42 else _digest(f"run-{seed}")
                        ),
                        "validation_result_path": f"results/tuning/dcdn-s{seed}.json",
                        "validation_result_sha256": _digest(f"result-{seed}"),
                        "best_checkpoint_path": f"runs/dcdn-s{seed}/best.pt",
                        "best_checkpoint_sha256": (
                            checkpoint if seed == 42 else _digest(f"best-{seed}")
                        ),
                        "selected_validation_metric": 0.02 + seed / 1_000_000,
                    }
                    for seed in (42, 43, 44)
                },
            }
        },
    }
    selected_digest = _write_hashed(selected_path, selected_payload)
    final_path = root / "protocol_final.yaml"
    _write_hashed(
        final_path,
        _protocol(
            hashes=hashes,
            unlocked=True,
            parent=parent_digest,
            selected=selected_digest,
        ),
    )
    claims_path = root / "test_claim_registry.yaml"
    gate = ProtocolGate.load(
        final_path,
        parent_premetric_protocol_path=parent_path,
        active_registry_path=active_path,
    )
    gate.activate(
        selected_configs_path=selected_path,
        test_claim_registry_path=claims_path,
    )
    runtime = RuntimeIdentity(
        dataset_id=gate.dataset_id,
        split="test",
        feature_regime=gate.feature_regime,
        row_count=ROWS,
        data_manifest_sha256=hashes["data_manifest_sha256"],
        views_manifest_sha256=_digest("test-views"),
        features_sha256=hashes["features_sha256"],
        physical_schema_sha256=hashes["physical_schema_sha256"],
        expanded_schema_sha256=hashes["expanded_schema_sha256"],
        operand_schema_sha256=hashes["operand_schema_sha256"],
        denominator_mask_sha256=hashes["denominator_mask_sha256"],
        full_materialized_scan_sha256=_digest("test-scan"),
        cache_content_sha256=_digest("test-cache"),
        encoder_contract_sha256=hashes["encoder_contract_sha256"],
        capacity_resolution_sha256=hashes["capacity_resolution_sha256"],
    )
    raw_identity = RawSourceIdentity(
        dataset_id=runtime.dataset_id,
        split="test",
        feature_regime=runtime.feature_regime,
        row_count=runtime.row_count,
        data_manifest_sha256=runtime.data_manifest_sha256,
        views_manifest_sha256=runtime.views_manifest_sha256,
        feature_contract_sha256=runtime.features_sha256,
        physical_schema_sha256=runtime.physical_schema_sha256,
        expanded_schema_sha256=runtime.expanded_schema_sha256,
        operand_schema_sha256=runtime.operand_schema_sha256,
        denominator_mask_sha256=runtime.denominator_mask_sha256,
        full_materialized_scan_sha256=runtime.full_materialized_scan_sha256,
        cache_content_sha256=runtime.cache_content_sha256,
        ordered_context_fields=("context",),
        ordered_operand_fields=("operand",),
        categorical_cardinalities=(4,),
    )
    return Prepared(
        gate=gate,
        selected_path=selected_path,
        claims_path=claims_path,
        runtime=runtime,
        raw_identity=raw_identity,
        candidate_id=candidate,
        resolved_config=resolved_config,
        resolved_run=resolved_run,
        checkpoint=checkpoint,
    )


class FinalSource:
    def __init__(self, prepared: Prepared) -> None:
        self.identity = prepared.raw_identity
        self.runtime_value_checks = True
        self.close_calls = 0
        self.batch_calls = 0

    def __len__(self) -> int:
        return self.identity.row_count

    def batch(self, indices: Sequence[int], *, device: torch.device) -> RawFeatureBatch:
        raise AssertionError("one-pass runner must use batch_with_audit")

    def batch_with_audit(
        self, indices: Sequence[int], *, device: torch.device
    ) -> tuple[RawFeatureBatch, RawAuditBatch]:
        self.batch_calls += 1
        requested = tuple(indices)
        global_indices = torch.tensor(requested, dtype=torch.int64)
        targets = torch.tensor(
            [index / (ROWS - 1) for index in requested], dtype=torch.float32
        ).reshape(-1, 1)
        raw = RawFeatureBatch(
            context_tokens=torch.full(
                (len(requested), 1), 2, dtype=torch.int64, device=device
            ),
            operand_features=torch.ones(
                (len(requested), 1), dtype=torch.float32, device=device
            ),
            denominator_validity=torch.ones(
                (len(requested), 1), dtype=torch.bool, device=device
            ),
            targets=targets.to(device=device),
            global_row_indices=global_indices,
        )
        audit = RawAuditBatch(
            user_ids=tuple(10 + index // 2 for index in requested),
            stable_row_ids=tuple(f"row-{index}" for index in requested),
            global_row_indices=global_indices,
        )
        return raw, audit

    def close(self) -> None:
        self.close_calls += 1


def _run(
    prepared: Prepared,
    root: Path,
    *,
    source_factory,
    inspect_checkpoint=None,
    restore_checkpoint=None,
    result_path: str = "results/final/dcdn-s42.json",
):
    inspect = inspect_checkpoint or (lambda: prepared.checkpoint)
    restore = restore_checkpoint or (lambda: prepared.checkpoint)
    return run_one_pass_test_bundle(
        gate=prepared.gate,
        selected_configs_path=prepared.selected_path,
        runtime_identity=prepared.runtime,
        model_key=MODEL_KEY,
        seed=42,
        candidate_id=prepared.candidate_id,
        resolved_config_sha256=prepared.resolved_config,
        resolved_run_config_sha256=prepared.resolved_run,
        checkpoint_protocol_sha256=prepared.gate.parent_premetric_digest or "",
        checkpoint_sha256=prepared.checkpoint,
        inspect_checkpoint=inspect,
        restore_checkpoint=restore,
        source_factory=source_factory,
        predict_batch=lambda raw: raw.targets + 0.1,
        repository_root=root,
        result_path=result_path,
        device="cpu",
        batch_size=2,
        duration_constant_seconds=157.0,
    )


def test_one_pass_bundle_opens_before_factory_and_completes(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    sources: list[FinalSource] = []
    factory_calls = 0

    def factory() -> FinalSource:
        nonlocal factory_calls
        factory_calls += 1
        registry = yaml.safe_load(prepared.claims_path.read_text(encoding="utf-8"))
        assert registry["claims"][0]["state"] == "opened"
        source = FinalSource(prepared)
        sources.append(source)
        return source

    record = _run(prepared, tmp_path, source_factory=factory)
    assert record.claim.state == "completed"
    assert record.claim.result_receipt_sha256 == record.sha256
    assert factory_calls == 1
    assert sources[0].batch_calls == 3
    assert sources[0].close_calls == 1
    assert record.payload["evaluation"]["source_passes"] == 1
    assert record.payload["metrics"]["normalized_mse"] == pytest.approx(0.01)
    assert record.payload["pairing"]["unique_stable_row_ids"] == ROWS
    assert record.payload["pairing"]["user_cluster_count"] == 3
    assert verify_sha256_sidecar(record.path) == record.sha256
    assert json.loads(record.path.read_text(encoding="ascii")) == record.payload
    registry = yaml.safe_load(prepared.claims_path.read_text(encoding="utf-8"))
    assert registry["claims"][0]["state"] == "completed"
    assert registry["claims"][0]["result_row_count"] == ROWS


def test_checkpoint_failure_leaves_claim_reserved_and_never_opens_source(
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    factory_calls = 0
    restore_calls = 0

    def factory() -> FinalSource:
        nonlocal factory_calls
        factory_calls += 1
        return FinalSource(prepared)

    def restore() -> str:
        nonlocal restore_calls
        restore_calls += 1
        return prepared.checkpoint

    with pytest.raises(FinalEvaluationError, match="inspection digest"):
        run_one_pass_test_bundle(
            gate=prepared.gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=prepared.runtime,
            model_key=MODEL_KEY,
            seed=42,
            candidate_id=prepared.candidate_id,
            resolved_config_sha256=prepared.resolved_config,
            resolved_run_config_sha256=prepared.resolved_run,
            checkpoint_protocol_sha256=prepared.gate.parent_premetric_digest or "",
            checkpoint_sha256=prepared.checkpoint,
            inspect_checkpoint=lambda: _digest("wrong"),
            restore_checkpoint=restore,
            source_factory=factory,
            predict_batch=lambda raw: raw.targets,
            repository_root=tmp_path,
            result_path="results/final/failure.json",
            device="cpu",
            batch_size=2,
            duration_constant_seconds=157.0,
        )
    assert factory_calls == 0
    assert restore_calls == 0
    registry = yaml.safe_load(prepared.claims_path.read_text(encoding="utf-8"))
    assert registry["claims"][0]["state"] == "reserved"


def test_source_failure_closes_source_and_consumes_open_permission(
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    source = FinalSource(prepared)

    def duplicate_predict(raw: RawFeatureBatch) -> np.ndarray:
        if source.batch_calls == 2:
            raise RuntimeError("synthetic prediction failure")
        return raw.targets.detach().cpu().numpy()

    with pytest.raises(RuntimeError, match="synthetic prediction failure"):
        run_one_pass_test_bundle(
            gate=prepared.gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=prepared.runtime,
            model_key=MODEL_KEY,
            seed=42,
            candidate_id=prepared.candidate_id,
            resolved_config_sha256=prepared.resolved_config,
            resolved_run_config_sha256=prepared.resolved_run,
            checkpoint_protocol_sha256=prepared.gate.parent_premetric_digest or "",
            checkpoint_sha256=prepared.checkpoint,
            inspect_checkpoint=lambda: prepared.checkpoint,
            restore_checkpoint=lambda: prepared.checkpoint,
            source_factory=lambda: source,
            predict_batch=duplicate_predict,
            repository_root=tmp_path,
            result_path="results/final/opened-failure.json",
            device="cpu",
            batch_size=2,
            duration_constant_seconds=157.0,
        )
    assert source.close_calls == 1
    registry = yaml.safe_load(prepared.claims_path.read_text(encoding="utf-8"))
    assert registry["claims"][0]["state"] == "opened"
    assert not (tmp_path / "results/final/opened-failure.json").exists()


def test_output_symlink_fails_before_claim_reservation(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "results/final").symlink_to(outside, target_is_directory=True)
    with pytest.raises(FinalEvaluationError, match="symlink or non-directory"):
        _run(
            prepared,
            tmp_path,
            source_factory=lambda: FinalSource(prepared),
        )
    assert not prepared.claims_path.exists()


def test_published_bundle_recovers_without_reopening_test_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare(tmp_path)
    source = FinalSource(prepared)
    original = EvaluationClaim.mark_completed

    def crash_after_publication(self, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic post-publication crash")

    monkeypatch.setattr(EvaluationClaim, "mark_completed", crash_after_publication)
    with pytest.raises(RuntimeError, match="post-publication crash"):
        _run(prepared, tmp_path, source_factory=lambda: source)
    assert source.close_calls == 1
    registry = yaml.safe_load(prepared.claims_path.read_text(encoding="utf-8"))
    assert registry["claims"][0]["state"] == "opened"
    claim_id = registry["claims"][0]["claim_id"]
    result_path = tmp_path / "results/final/dcdn-s42.json"
    assert result_path.is_file()

    monkeypatch.setattr(EvaluationClaim, "mark_completed", original)
    recovered = recover_published_test_bundle(
        gate=prepared.gate,
        selected_configs_path=prepared.selected_path,
        runtime_identity=prepared.runtime,
        claim_id=claim_id,
        repository_root=tmp_path,
        result_path="results/final/dcdn-s42.json",
    )
    assert recovered.claim.state == "completed"
    assert recovered.claim.result_receipt_sha256 == recovered.sha256
    assert source.batch_calls == 3
    assert source.close_calls == 1
    registry = yaml.safe_load(prepared.claims_path.read_text(encoding="utf-8"))
    assert registry["claims"][0]["state"] == "completed"


def test_recovery_rejects_rehashed_metric_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare(tmp_path)
    source = FinalSource(prepared)
    original = EvaluationClaim.mark_completed

    def crash_after_publication(self, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic post-publication crash")

    monkeypatch.setattr(EvaluationClaim, "mark_completed", crash_after_publication)
    with pytest.raises(RuntimeError, match="post-publication crash"):
        _run(prepared, tmp_path, source_factory=lambda: source)
    monkeypatch.setattr(EvaluationClaim, "mark_completed", original)

    result_path = tmp_path / "results/final/dcdn-s42.json"
    payload = json.loads(result_path.read_text(encoding="ascii"))
    payload["metrics"]["normalized_mse"] += 0.5
    data = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    digest = hashlib.sha256(data).hexdigest()
    result_path.write_bytes(data)
    result_path.with_name(f"{result_path.name}.sha256").write_text(
        f"{digest}  {result_path.name}\n",
        encoding="ascii",
    )
    registry = yaml.safe_load(prepared.claims_path.read_text(encoding="utf-8"))
    with pytest.raises(FinalEvaluationError, match="differs from cluster sums"):
        recover_published_test_bundle(
            gate=prepared.gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=prepared.runtime,
            claim_id=registry["claims"][0]["claim_id"],
            repository_root=tmp_path,
            result_path="results/final/dcdn-s42.json",
        )
    registry = yaml.safe_load(prepared.claims_path.read_text(encoding="utf-8"))
    assert registry["claims"][0]["state"] == "opened"
