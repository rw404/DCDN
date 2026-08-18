from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import pytest
import torch
import yaml

import dcdn.training as training_module
from dcdn.capacity_artifact import (
    capacity_resolution_projection,
    load_verified_capacity_artifact,
    sha256_file,
)
from dcdn.encoder import EncoderContractLock, build_trainable_encoded_model
from dcdn.models import (
    AFNPlusConfig,
    AutoIntPlusConfig,
    DCDNCoreConfig,
    DCNConfig,
    DCNv2CapacityCoreConfig,
    DCNv2FullRankConfig,
    DeepFMConfig,
    HeadConfig,
    MLPConfig,
    WukongSmallConfig,
    build_controlled_model,
)
from dcdn.monitor import NumericalMonitorError
from dcdn.protocol import (
    ProtocolGate,
    canonical_split_sha256,
    verify_sha256_sidecar,
)
from dcdn.raw import (
    RawContractError,
    RawFeatureBatch,
    RawSourceIdentity,
)
from dcdn.sampler import StatefulPermutationSampler
from dcdn.search_space import SearchSpaceLock
from dcdn.training import (
    SafetyAttestation,
    SecondOOMError,
    SplitAccessError,
    TrainingConfig,
    TrainingContractError,
    TrainingHarness,
    count_trainable_non_embedding_parameters,
    deterministic_forward_flops_per_example,
    resolve_metric_free_capacity_preflight,
    runtime_identity_from_raw,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATH = REPOSITORY_ROOT / "configs/search_space_v1.yaml"
ENCODER_PATH = REPOSITORY_ROOT / "configs/encoder_v1.yaml"
CAPACITY_PATH = REPOSITORY_ROOT / "configs/capacity/vk_lsvd_c_plus_p_v1.json"
DATASET_ID = "vk_lsvd_up0.01_ir0.01"
FEATURE_REGIME = "C+P"
TRAIN_ROWS = 8192
VALIDATION_ROWS = 64
SPLIT_MAPPING = {
    "train_weeks": [20, 21, 22, 23, 24],
    "validation_week": 25,
    "test": None,
}


def _write_hashed_yaml(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _model_payload(*, runtime_value_checks: bool) -> dict[str, Any]:
    return {
        "interaction_config": {
            "model_key": "dcdn_core",
            "num_layers": 2,
            "div_eps": 1e-2,
            "product_init_scale": 0.1,
            "div_init_scale": 0.1,
            "runtime_value_checks": runtime_value_checks,
        },
        "head_config": {"hidden_dims": [], "dropout": 0.0, "output_dim": 1},
        "init_seed": 42,
    }


def _selected_model_digest(*, runtime_value_checks: bool) -> str:
    payload = _model_payload(runtime_value_checks=runtime_value_checks)
    return _canonical_digest(
        {
            "interaction_config": payload["interaction_config"],
            "head_config": payload["head_config"],
        }
    )


@dataclass(frozen=True)
class FrozenFixture:
    search: SearchSpaceLock
    encoder: EncoderContractLock
    capacity_digest: str
    train_identity: RawSourceIdentity
    validation_identity: RawSourceIdentity
    safety: SafetyAttestation
    config: TrainingConfig


def _raw_identity(
    encoder: EncoderContractLock,
    hashes: dict[str, str],
    split: Literal["train", "validation"],
    rows: int,
) -> RawSourceIdentity:
    dataset = encoder.payload["datasets"][DATASET_ID]
    regime = dataset["regimes"][FEATURE_REGIME]
    return RawSourceIdentity(
        dataset_id=DATASET_ID,
        split=split,
        feature_regime=FEATURE_REGIME,
        row_count=rows,
        data_manifest_sha256=hashes["data_manifest_sha256"],
        views_manifest_sha256=hashes["views_manifest_sha256"],
        feature_contract_sha256=hashes["features_sha256"],
        physical_schema_sha256=hashes["physical_schema_sha256"],
        expanded_schema_sha256=hashes["expanded_schema_sha256"],
        operand_schema_sha256=hashes["operand_schema_sha256"],
        denominator_mask_sha256=hashes["denominator_mask_sha256"],
        full_materialized_scan_sha256=hashes["full_materialized_scan_sha256"],
        cache_content_sha256=hashes["cache_content_sha256"],
        ordered_context_fields=tuple(dataset["ordered_context_fields"]),
        ordered_operand_fields=tuple(regime["ordered_operand_fields"]),
        categorical_cardinalities=tuple(dataset["categorical_cardinalities"]),
    )


def _fixture(
    tmp_path: Path,
    *,
    runtime_value_checks: bool = False,
) -> FrozenFixture:
    search = SearchSpaceLock.load(SEARCH_PATH)
    encoder = EncoderContractLock.load(ENCODER_PATH)
    capacity_digest = sha256_file(CAPACITY_PATH)
    regime = encoder.payload["datasets"][DATASET_ID]["regimes"][FEATURE_REGIME]
    dataset = encoder.payload["datasets"][DATASET_ID]
    hashes = {
        "data_manifest_sha256": dataset["source_manifest_sha256"],
        "views_manifest_sha256": dataset["views_manifest_sha256"],
        "features_sha256": dataset["feature_contract_sha256"],
        "physical_schema_sha256": regime["physical_schema_sha256"],
        "expanded_schema_sha256": regime["expanded_schema_sha256"],
        "operand_schema_sha256": regime["operand_schema_sha256"],
        "denominator_mask_sha256": regime["denominator_mask_sha256"],
        "full_materialized_scan_sha256": dataset["full_materialized_scan_sha256"],
        "cache_content_sha256": regime["cache_content_sha256"],
        "encoder_contract_sha256": encoder.sha256,
        "capacity_resolution_sha256": capacity_digest,
        "search_space_sha256": search.digest,
        "experiment_matrix_sha256": search.experiment_matrix_digest,
        "split_sha256": canonical_split_sha256(SPLIT_MAPPING),
    }
    train_identity = _raw_identity(encoder, hashes, "train", TRAIN_ROWS)
    validation_identity = _raw_identity(encoder, hashes, "validation", VALIDATION_ROWS)
    protocol_payload = {
        "schema_version": 3,
        "dataset_id": DATASET_ID,
        "feature_regime": FEATURE_REGIME,
        "required_model_keys": ["dcdn_core"],
        "test_unlocked": False,
        "test_path_available": False,
        "hashes": hashes,
        "split_row_counts": {
            "train": TRAIN_ROWS,
            "validation": VALIDATION_ROWS,
            "test": None,
        },
        "final_seeds": [42, 43, 44],
        "tuning_seed": 42,
        "tuning_splits": ["train", "validation"],
        "test_evaluations_per_final_seed": 1,
    }
    protocol_path = tmp_path / "protocol.yaml"
    active_path = tmp_path / "active_protocol.yaml"
    _write_hashed_yaml(protocol_path, protocol_payload)
    gate = ProtocolGate.load(protocol_path, active_registry_path=active_path)
    gate.activate()
    views_payload = {"split": SPLIT_MAPPING}
    safety = SafetyAttestation(
        protocol_gate=gate,
        cost_snapshot={
            "launch_allowed": True,
            "provider_actions_performed": False,
        },
        storage_snapshot={"ok": True},
        leakage_checks={
            "source_manifest_verified": True,
            "split_verified": True,
            "ordered_schema_verified": True,
            "row_alignment_verified": True,
            "temporal_leakage_check_passed": True,
            "reciprocal_mask_verified": True,
            "full_operand_scan_passed": True,
        },
        views_payload=views_payload,
        views_payload_sha256=_canonical_digest(views_payload),
        views_split_sha256=hashes["split_sha256"],
        search_space_sha256=search.digest,
        experiment_matrix_sha256=search.experiment_matrix_digest,
        encoder_contract_sha256=encoder.sha256,
        capacity_resolution_sha256=capacity_digest,
        train_runtime_identity=runtime_identity_from_raw(
            train_identity,
            encoder_contract_sha256=encoder.sha256,
            capacity_resolution_sha256=capacity_digest,
        ),
        validation_runtime_identity=runtime_identity_from_raw(
            validation_identity,
            encoder_contract_sha256=encoder.sha256,
            capacity_resolution_sha256=capacity_digest,
        ),
        train_source_identity=train_identity,
        validation_source_identity=validation_identity,
        registry_on_persistent_storage=True,
    )
    config = TrainingConfig(
        dataset_id=DATASET_ID,
        model_key="dcdn_core",
        feature_regime=FEATURE_REGIME,
        candidate_id="dcdn_core_l2_lr_3e_4",
        seed=42,
        phase="tuning",
        learning_rate=3e-4,
        microbatch_size=4096,
        effective_batch_size=4096,
        maximum_train_examples=40_000_000,
        validate_every_examples=4_000_000,
        patience_validation_checks=3,
        loss="normalized_mean_squared_error",
        selection_metric="normalized_mse",
        selection_direction="minimize",
        search_space_sha256=search.digest,
        experiment_matrix_sha256=search.experiment_matrix_digest,
        encoder_contract_sha256=encoder.sha256,
        capacity_resolution_sha256=capacity_digest,
        resolved_model_config_sha256=_selected_model_digest(
            runtime_value_checks=runtime_value_checks
        ),
        full_schema_sha256=validation_identity.expanded_schema_sha256,
        operand_schema_sha256=validation_identity.operand_schema_sha256,
        denominator_mask_sha256=validation_identity.denominator_mask_sha256,
        autocast_bfloat16=True,
        runtime_value_checks=runtime_value_checks,
        reproduction_command="python scripts/train_model.py --config resolved.yaml",
    )
    return FrozenFixture(
        search=search,
        encoder=encoder,
        capacity_digest=capacity_digest,
        train_identity=train_identity,
        validation_identity=validation_identity,
        safety=safety,
        config=config,
    )


class TinyRawSource:
    def __init__(
        self,
        identity: RawSourceIdentity,
        *,
        runtime_value_checks: bool,
    ) -> None:
        self.identity = identity
        self.runtime_value_checks = runtime_value_checks
        self.close_calls = 0

    def __len__(self) -> int:
        return self.identity.row_count

    def batch(
        self,
        indices: Sequence[int],
        *,
        device: torch.device,
    ) -> RawFeatureBatch:
        requested = tuple(indices)
        global_indices = torch.tensor(requested, dtype=torch.int64)
        context = torch.empty(
            (len(requested), len(self.identity.ordered_context_fields)),
            dtype=torch.int64,
        )
        for column, cardinality in enumerate(self.identity.categorical_cardinalities):
            context[:, column] = global_indices.remainder(cardinality - 2) + 2
        width = len(self.identity.ordered_operand_fields)
        rows = global_indices.to(dtype=torch.float32).unsqueeze(1)
        columns = torch.arange(width, dtype=torch.float32).unsqueeze(0)
        operands = 1.0 + rows / 10_000.0 + columns / 1_000.0
        targets = ((global_indices % 17).to(torch.float32) / 16.0).unsqueeze(1)
        return RawFeatureBatch(
            context_tokens=context.to(device=device),
            operand_features=operands.to(device=device),
            denominator_validity=torch.ones(
                (len(requested), width), dtype=torch.bool, device=device
            ),
            targets=targets.to(device=device),
            global_row_indices=global_indices,
        )

    def close(self) -> None:
        self.close_calls += 1


def _source_factory(
    identity: RawSourceIdentity,
    *,
    runtime_value_checks: bool,
    opened: list[TinyRawSource] | None = None,
):  # type: ignore[no-untyped-def]
    def factory() -> TinyRawSource:
        source = TinyRawSource(
            identity,
            runtime_value_checks=runtime_value_checks,
        )
        if opened is not None:
            opened.append(source)
        return source

    return factory


def _build_harness(
    frozen: FrozenFixture,
    run_directory: Path,
) -> TrainingHarness:
    payload = _model_payload(runtime_value_checks=frozen.config.runtime_value_checks)
    model = build_trainable_encoded_model(
        encoder_contract=frozen.encoder,
        dataset_id=DATASET_ID,
        source_identity=frozen.train_identity,
        interaction_config=DCDNCoreConfig(
            **{
                key: value
                for key, value in payload["interaction_config"].items()
                if key != "model_key"
            }
        ),
        head_config=HeadConfig(hidden_dims=(), dropout=0.0, output_dim=1),
        run_seed=frozen.config.seed,
        runtime_value_checks=frozen.config.runtime_value_checks,
    )
    return TrainingHarness(
        model=model,
        config=frozen.config,
        safety=frozen.safety,
        search_lock=frozen.search,
        capacity_resolution=None,
        operand_feature_order=frozen.train_identity.ordered_operand_fields,
        run_directory=run_directory,
        device="cpu",
        capacity_artifact_path=CAPACITY_PATH,
        repository_root=REPOSITORY_ROOT,
    )


def _clone_state(module: torch.nn.Module) -> dict[str, Any]:
    return copy.deepcopy(module.state_dict())


def _assert_state_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.device == right.device
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_state_equal(left[key], right[key])
        return
    if isinstance(left, list | tuple):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_state_equal(left_item, right_item)
        return
    assert left == right


def test_examples_budget_floor_and_validation_boundary_are_exact(
    tmp_path: Path,
) -> None:
    frozen = _fixture(tmp_path)
    assert frozen.config.realized_maximum_examples == 39_997_440
    assert frozen.config.completed_batch_boundary_at_or_after(4_000_000) == 4_001_792
    assert frozen.config.realized_maximum_examples <= 40_000_000
    seed_43 = replace(frozen.config, phase="final", seed=43)
    assert (
        seed_43.resolved_model_config_sha256
        == frozen.config.resolved_model_config_sha256
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("patience_validation_checks", 4, "validation patience"),
        ("adam_betas", (0.8, 0.999), "AdamW betas"),
        ("adam_epsilon", 2e-8, "AdamW epsilon"),
        ("weight_decay", 2e-6, "AdamW weight decay"),
        ("autocast_bfloat16", False, "BF16 autocast"),
        ("gradient_clip_l2", 2.0, "gradient clipping norm"),
        (
            "allow_one_oom_microbatch_reduction",
            False,
            "one-OOM microbatch policy",
        ),
    ],
)
def test_frozen_common_training_fields_fail_closed(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    frozen = _fixture(tmp_path)
    changed = replace(frozen.config, **{field: value})
    if field == "autocast_bfloat16":
        assert changed.checkpoint_precision == "fp32_parameters_fp32_forward"
    with pytest.raises(TrainingContractError, match=message):
        _build_harness(replace(frozen, config=changed), tmp_path / "run")


def test_microbatch_remains_flexible_and_checkpoint_precision_is_derived(
    tmp_path: Path,
) -> None:
    frozen = _fixture(tmp_path)
    changed = replace(frozen.config, microbatch_size=2048)
    harness = _build_harness(replace(frozen, config=changed), tmp_path / "run")
    try:
        assert harness.config.accumulation_steps == 2
        assert harness.checkpoint_metadata().precision == (
            "fp32_parameters_bf16_autocast"
        )
    finally:
        harness.close()


def test_all_runtime_architectures_bind_exactly_to_frozen_search_and_capacity(
    tmp_path: Path,
) -> None:
    frozen = _fixture(tmp_path)
    artifact = load_verified_capacity_artifact(CAPACITY_PATH)
    cases = [
        (
            "dnn",
            "dnn_lr_3e_4",
            3e-4,
            MLPConfig(hidden_dims=(512, 256, 128), dropout=0.1),
            MLPConfig(hidden_dims=(256, 128), dropout=0.1),
        ),
        (
            "deepfm",
            "deepfm_lr_3e_4",
            3e-4,
            DeepFMConfig(deep_hidden_dims=(512, 256, 128), dropout=0.1),
            DeepFMConfig(deep_hidden_dims=(256, 128), dropout=0.1),
        ),
        (
            "dcn",
            "dcn_lr_3e_4",
            3e-4,
            DCNConfig(num_layers=2, product_init_scale=0.1),
            DCNConfig(num_layers=3, product_init_scale=0.1),
        ),
        (
            "autoint_plus",
            "autoint_plus_lr_3e_4",
            3e-4,
            AutoIntPlusConfig(
                attention_layers=2,
                attention_dim=16,
                num_heads=2,
                dnn_hidden_dims=(256, 128),
                dropout=0.1,
                scale_attention=True,
                layer_norm=True,
            ),
            AutoIntPlusConfig(
                attention_layers=3,
                attention_dim=16,
                num_heads=2,
                dnn_hidden_dims=(256, 128),
                dropout=0.1,
                scale_attention=True,
                layer_norm=True,
            ),
        ),
        (
            "dcnv2_full_rank",
            "dcnv2_full_rank_lr_3e_4",
            3e-4,
            DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1),
            DCNv2FullRankConfig(num_layers=3, product_init_scale=0.1),
        ),
        (
            "afn_plus",
            "afn_plus_lr_3e_4",
            3e-4,
            AFNPlusConfig(
                logarithmic_neurons=128,
                afn_hidden_dims=(512,),
                dnn_hidden_dims=(512,),
                log_floor=1e-4,
                exp_clip=None,
                dropout=0.1,
                batch_norm=True,
            ),
            AFNPlusConfig(
                logarithmic_neurons=127,
                afn_hidden_dims=(512,),
                dnn_hidden_dims=(512,),
                log_floor=1e-4,
                exp_clip=None,
                dropout=0.1,
                batch_norm=True,
            ),
        ),
        (
            "wukong_small",
            "wukong_small_paper_explicit_min",
            3e-4,
            WukongSmallConfig(
                num_layers=2,
                lcb_features=8,
                fmb_features=8,
                fmb_rank=24,
                fmb_hidden_dims=(2048, 2048, 2048),
                dropout=0.0,
                layer_norm=True,
            ),
            WukongSmallConfig(
                num_layers=3,
                lcb_features=8,
                fmb_features=8,
                fmb_rank=24,
                fmb_hidden_dims=(2048, 2048, 2048),
                dropout=0.0,
                layer_norm=True,
            ),
        ),
        (
            "dcdn_core",
            "dcdn_core_l2_lr_3e_4",
            3e-4,
            DCDNCoreConfig(
                num_layers=2,
                div_eps=1e-2,
                product_init_scale=0.1,
                div_init_scale=0.1,
                runtime_value_checks=False,
            ),
            DCDNCoreConfig(
                num_layers=2,
                div_eps=2e-2,
                product_init_scale=0.1,
                div_init_scale=0.1,
                runtime_value_checks=False,
            ),
        ),
        (
            "dcnv2_capacity_matched_core",
            "capacity_rank_lower",
            3e-4,
            DCNv2CapacityCoreConfig(
                num_layers=2,
                low_rank=12,
                num_experts=4,
                product_init_scale=0.1,
            ),
            DCNv2CapacityCoreConfig(
                num_layers=2,
                low_rank=13,
                num_experts=4,
                product_init_scale=0.1,
            ),
        ),
    ]
    for model_key, candidate_id, learning_rate, expected, mutated in cases:
        config = replace(
            frozen.config,
            model_key=model_key,
            candidate_id=candidate_id,
            learning_rate=learning_rate,
        )
        assert (
            training_module._expected_frozen_interaction_config(
                config,
                frozen.search,
                capacity_artifact=artifact,
            )
            == expected
        )
        training_module._validate_frozen_interaction_contract(
            config,
            expected,
            frozen.search,
            capacity_artifact=artifact,
        )
        with pytest.raises(TrainingContractError, match="runtime interaction config"):
            training_module._validate_frozen_interaction_contract(
                config,
                mutated,
                frozen.search,
                capacity_artifact=artifact,
            )

    capacity_upper = replace(
        frozen.config,
        model_key="dcnv2_capacity_matched_core",
        candidate_id="capacity_rank_upper",
    )
    with pytest.raises(TrainingContractError, match="metric-free decision"):
        training_module._validated_capacity_candidate(capacity_upper, artifact)


def test_raw_contract_preserves_duplicate_nonmonotone_order_and_optimized_no_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _fixture(tmp_path, runtime_value_checks=False)
    source = TinyRawSource(
        frozen.train_identity,
        runtime_value_checks=False,
    )
    indices = [TRAIN_ROWS - 1, 0, TRAIN_ROWS - 1, 7]
    batch = source.batch(indices, device=torch.device("cpu"))
    assert batch.global_row_indices.tolist() == indices
    assert batch.context_tokens.dtype == torch.int64
    assert batch.operand_features.dtype == torch.float32
    assert batch.denominator_validity.dtype == torch.bool
    assert batch.targets.dtype == torch.float32
    assert not hasattr(batch, "field_embeddings")

    model = build_trainable_encoded_model(
        encoder_contract=frozen.encoder,
        dataset_id=DATASET_ID,
        source_identity=frozen.train_identity,
        interaction_config=DCDNCoreConfig(
            **{
                key: value
                for key, value in _model_payload(runtime_value_checks=False)[
                    "interaction_config"
                ].items()
                if key != "model_key"
            }
        ),
        head_config=HeadConfig(hidden_dims=(), dropout=0.0, output_dim=1),
        run_seed=42,
        runtime_value_checks=False,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("optimized path synchronized a tensor value to host")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    batch.validate(
        frozen.train_identity,
        indices,
        runtime_value_checks=False,
    )
    output = model(batch)
    assert output.logits.shape == (len(indices), 1)


def test_debug_raw_contract_rejects_bad_token(tmp_path: Path) -> None:
    frozen = _fixture(tmp_path, runtime_value_checks=True)
    source = TinyRawSource(frozen.train_identity, runtime_value_checks=True)
    batch = source.batch([0, 1], device=torch.device("cpu"))
    invalid_context = batch.context_tokens.clone()
    invalid_context[1, 0] = frozen.train_identity.categorical_cardinalities[0]
    invalid = replace(batch, context_tokens=invalid_context)
    with pytest.raises(RawContractError, match="token"):
        invalid.validate(
            frozen.train_identity,
            [0, 1],
            runtime_value_checks=True,
        )


def test_composite_capacity_optimizer_hooks_and_validation_close(
    tmp_path: Path,
) -> None:
    frozen = _fixture(tmp_path)
    harness = _build_harness(frozen, tmp_path / "run")
    opened: list[TinyRawSource] = []
    try:
        artifact = load_verified_capacity_artifact(CAPACITY_PATH)
        assert harness.capacity_resolution == capacity_resolution_projection(artifact)
        assert count_trainable_non_embedding_parameters(harness.model.encoder) == 0
        assert count_trainable_non_embedding_parameters(harness.model) == (
            count_trainable_non_embedding_parameters(harness.controlled_model)
        )
        optimizer_ids = {
            id(parameter)
            for group in harness.optimizer.param_groups
            for parameter in group["params"]
        }
        assert optimizer_ids == {
            id(parameter)
            for parameter in harness.model.parameters()
            if parameter.requires_grad
        }
        assert harness.monitor.activation_hook_count == 3
        value = harness.evaluate(
            "validation",
            _source_factory(
                frozen.validation_identity,
                runtime_value_checks=False,
                opened=opened,
            ),
            batch_size=VALIDATION_ROWS,
        )
        assert value >= 0.0
        assert harness.monitor.activation_hook_observations >= 3
        assert len(opened) == 1 and opened[0].close_calls == 1
    finally:
        harness.close()


def test_test_loss_path_fails_before_factory_or_claim(tmp_path: Path) -> None:
    frozen = _fixture(tmp_path)
    harness = _build_harness(frozen, tmp_path / "run")
    opened = False

    def forbidden_factory() -> TinyRawSource:
        nonlocal opened
        opened = True
        raise AssertionError("test source factory must not run")

    try:
        with pytest.raises(SplitAccessError, match="one-pass"):
            harness.evaluate("test", forbidden_factory)
        assert opened is False
    finally:
        harness.close()


def test_protocol_gate_precedes_factory_and_identity_drift_closes_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _fixture(tmp_path)
    harness = _build_harness(frozen, tmp_path / "run")
    events: list[str] = []
    original = SafetyAttestation.require_split

    def recording_require_split(
        safety: SafetyAttestation,
        split: Literal["train", "validation", "test"],
        config: TrainingConfig,
        *,
        checkpoint_protocol_sha256: str | None = None,
    ):  # type: ignore[no-untyped-def]
        events.append(split)
        return original(
            safety,
            split,
            config,
            checkpoint_protocol_sha256=checkpoint_protocol_sha256,
        )

    monkeypatch.setattr(SafetyAttestation, "require_split", recording_require_split)
    opened: list[TinyRawSource] = []

    def ordered_train_factory() -> TinyRawSource:
        assert events == ["train"]
        source = TinyRawSource(
            frozen.train_identity,
            runtime_value_checks=False,
        )
        opened.append(source)
        return source

    def ordered_factory() -> TinyRawSource:
        assert events == ["train", "validation"]
        source = TinyRawSource(
            frozen.validation_identity,
            runtime_value_checks=False,
        )
        opened.append(source)
        return source

    try:
        train_source = harness._open_raw_source("train", ordered_train_factory)
        harness._close_raw_source(train_source)
        harness.evaluate("validation", ordered_factory, batch_size=VALIDATION_ROWS)
        drifted = replace(
            frozen.validation_identity,
            cache_content_sha256=hashlib.sha256(b"drift-cache").hexdigest(),
        )
        with pytest.raises(SplitAccessError, match="byte-for-byte"):
            harness.evaluate(
                "validation",
                _source_factory(
                    drifted,
                    runtime_value_checks=False,
                    opened=opened,
                ),
                batch_size=VALIDATION_ROWS,
            )
        assert events == ["train", "validation", "validation"]
        assert [source.close_calls for source in opened] == [1, 1, 1]
    finally:
        harness.close()


def test_exact_raw_cpu_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    frozen = _fixture(tmp_path)
    train_factory = _source_factory(
        frozen.train_identity,
        runtime_value_checks=False,
    )
    validation_factory = _source_factory(
        frozen.validation_identity,
        runtime_value_checks=False,
    )

    reference = _build_harness(frozen, tmp_path / "reference")
    try:
        reference_result = reference.fit(
            train_factory,
            validation_factory,
            stop_after_examples=8192,
        )
        reference_model = _clone_state(reference.model)
        reference_optimizer = copy.deepcopy(reference.optimizer.state_dict())
        assert reference.sampler is not None
        reference_sampler = copy.deepcopy(reference.sampler.state_dict())
    finally:
        reference.close()

    interrupted = _build_harness(frozen, tmp_path / "resumed")
    try:
        interrupted.fit(
            train_factory,
            validation_factory,
            stop_after_examples=4096,
        )
    finally:
        interrupted.close()
    resumed = _build_harness(frozen, tmp_path / "resumed")
    try:
        resumed_result = resumed.fit(
            train_factory,
            validation_factory,
            resume=True,
            stop_after_examples=8192,
        )
        assert resumed.sampler is not None
        assert resumed_result.examples_seen == reference_result.examples_seen == 8192
        assert resumed_result.optimizer_steps == reference_result.optimizer_steps == 2
        _assert_state_equal(reference_model, resumed.model.state_dict())
        _assert_state_equal(reference_optimizer, resumed.optimizer.state_dict())
        _assert_state_equal(reference_sampler, resumed.sampler.state_dict())
    finally:
        resumed.close()


def test_fresh_run_rejects_checkpoint_collision_and_resume_requires_pair(
    tmp_path: Path,
) -> None:
    frozen = _fixture(tmp_path)
    train_factory = _source_factory(
        frozen.train_identity,
        runtime_value_checks=False,
    )
    validation_factory = _source_factory(
        frozen.validation_identity,
        runtime_value_checks=False,
    )
    run_directory = tmp_path / "run"
    first = _build_harness(frozen, run_directory)
    try:
        first.fit(
            train_factory,
            validation_factory,
            stop_after_examples=4096,
        )
    finally:
        first.close()

    collision = _build_harness(frozen, run_directory)
    opened = False

    def forbidden_train() -> TinyRawSource:
        nonlocal opened
        opened = True
        return TinyRawSource(frozen.train_identity, runtime_value_checks=False)

    try:
        with pytest.raises(TrainingContractError, match="fresh run refuses"):
            collision.fit(
                forbidden_train,
                validation_factory,
                stop_after_examples=4096,
            )
        assert opened is False
    finally:
        collision.close()

    (run_directory / "last.pt.sha256").unlink()
    missing = _build_harness(frozen, run_directory)
    try:
        with pytest.raises(TrainingContractError, match="intact last"):
            missing.fit(
                forbidden_train,
                validation_factory,
                resume=True,
                stop_after_examples=8192,
            )
        assert opened is False
    finally:
        missing.close()


def _save_selected_best(
    harness: TrainingHarness,
    *,
    metric: float,
) -> tuple[Path, str]:
    if harness.sampler is None:
        harness.sampler = StatefulPermutationSampler(
            TRAIN_ROWS,
            seed=harness.config.seed,
            shuffle=True,
        )
    if harness.global_step == 0:
        indices = []
        iterator = iter(harness.sampler)
        for _ in range(harness.config.effective_batch_size):
            indices.append(next(iterator))
        harness.sampler.commit(indices)
        harness.global_step = 1
        harness.examples_seen = harness.config.effective_batch_size
    harness.best_metric = metric
    harness.validation_checks = max(1, harness.validation_checks)
    path = harness._next_best_checkpoint_path()
    record = harness._save(path, checkpoint_role="best")
    harness.best_checkpoint_path = path
    harness.best_checkpoint_record = record
    harness.selected_best_checkpoint_sha256 = record.sha256
    return path, record.sha256


def test_immutable_best_restore_is_model_only_and_stale_metric_is_nonmutating(
    tmp_path: Path,
) -> None:
    frozen = _fixture(tmp_path)
    run_directory = tmp_path / "run"
    harness = _build_harness(frozen, run_directory)
    try:
        harness.sampler = StatefulPermutationSampler(TRAIN_ROWS, seed=42, shuffle=True)
        selected_path, selected_sha = _save_selected_best(harness, metric=0.4)
        selected_model_state = _clone_state(harness.model)
        harness.last_checkpoint_record = harness._save(
            harness.last_checkpoint_path,
            checkpoint_role="last",
        )
        for parameter in harness.model.parameters():
            parameter.data.add_(1.0)
        optimizer_before = copy.deepcopy(harness.optimizer.state_dict())
        rng_before = torch.get_rng_state().clone()
        restored = harness.restore_best_for_evaluation()
        assert restored.sha256 == selected_sha
        assert harness.best_checkpoint_path == selected_path
        _assert_state_equal(selected_model_state, harness.model.state_dict())
        _assert_state_equal(optimizer_before, harness.optimizer.state_dict())
        assert torch.equal(rng_before, torch.get_rng_state())

        stale_path, _ = _save_selected_best(harness, metric=0.3)
        assert stale_path != selected_path
        harness.best_metric = 0.4
        harness.last_checkpoint_record = harness._save(
            harness.last_checkpoint_path,
            checkpoint_role="last",
        )
    finally:
        harness.close()

    candidate = _build_harness(frozen, run_directory)
    model_before = _clone_state(candidate.model)
    optimizer_before = copy.deepcopy(candidate.optimizer.state_dict())
    rng_before = torch.get_rng_state().clone()
    source = TinyRawSource(frozen.train_identity, runtime_value_checks=False)
    try:
        with pytest.raises(Exception, match="metric differs"):
            candidate.resume(source)
        _assert_state_equal(model_before, candidate.model.state_dict())
        _assert_state_equal(optimizer_before, candidate.optimizer.state_dict())
        assert torch.equal(rng_before, torch.get_rng_state())
    finally:
        source.close()
        candidate.close()


def test_checkpoint_loaders_always_request_cpu_map_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _fixture(tmp_path)
    run_directory = tmp_path / "run"
    writer = _build_harness(frozen, run_directory)
    try:
        writer.sampler = StatefulPermutationSampler(TRAIN_ROWS, seed=42, shuffle=True)
        _save_selected_best(writer, metric=0.25)
        writer.last_checkpoint_record = writer._save(
            writer.last_checkpoint_path,
            checkpoint_role="last",
        )
    finally:
        writer.close()

    observed: list[str | torch.device] = []
    original_train = training_module.load_checkpoint
    original_eval = training_module.load_model_checkpoint_for_evaluation

    def train_wrapper(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        observed.append(kwargs["map_location"])
        return original_train(*args, **kwargs)

    def eval_wrapper(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        observed.append(kwargs["map_location"])
        return original_eval(*args, **kwargs)

    monkeypatch.setattr(training_module, "load_checkpoint", train_wrapper)
    monkeypatch.setattr(
        training_module,
        "load_model_checkpoint_for_evaluation",
        eval_wrapper,
    )
    reader = _build_harness(frozen, run_directory)
    source = TinyRawSource(frozen.train_identity, runtime_value_checks=False)
    try:
        reader.resume(source)
        reader.restore_best_for_evaluation()
        assert observed and set(observed) == {"cpu"}
    finally:
        source.close()
        reader.close()


def test_latency_reinstalls_wrapper_hooks_and_closes_source_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _fixture(tmp_path)
    harness = _build_harness(frozen, tmp_path / "run")
    opened: list[TinyRawSource] = []
    original_forward = harness.model.forward
    calls = 0

    def fail_timed_forward(batch: RawFeatureBatch):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("timed forward failed")
        return original_forward(batch)

    monkeypatch.setattr(harness.model, "forward", fail_timed_forward)
    try:
        with pytest.raises(RuntimeError, match="timed forward"):
            harness.measure_model_latency(
                _source_factory(
                    frozen.validation_identity,
                    runtime_value_checks=False,
                    opened=opened,
                ),
                warmup_repetitions=1,
                measured_repetitions=1,
                batch_size=8,
            )
        assert harness.monitor.activation_hook_count == 3
        assert len(opened) == 1 and opened[0].close_calls == 1
    finally:
        harness.close()


def test_first_oom_fallback_is_durable_and_second_oom_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _fixture(tmp_path)
    run_directory = tmp_path / "run"
    crashing = _build_harness(frozen, run_directory)

    def always_oom(*_args: Any, **_kwargs: Any) -> Any:
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    original_emit = crashing.event_log.emit

    def crash_after_fallback(event: str, **payload: Any) -> None:
        if event == "oom_recovery":
            raise RuntimeError("synthetic crash after durable fallback")
        original_emit(event, **payload)

    monkeypatch.setattr(crashing, "_batch_forward", always_oom)
    monkeypatch.setattr(crashing.event_log, "emit", crash_after_fallback)
    try:
        with pytest.raises(RuntimeError, match="durable fallback"):
            crashing.fit(
                _source_factory(
                    frozen.train_identity,
                    runtime_value_checks=False,
                ),
                _source_factory(
                    frozen.validation_identity,
                    runtime_value_checks=False,
                ),
                stop_after_examples=4096,
            )
    finally:
        crashing.close()

    resumed = _build_harness(frozen, run_directory)
    monkeypatch.setattr(resumed, "_batch_forward", always_oom)
    source = TinyRawSource(frozen.train_identity, runtime_value_checks=False)
    try:
        resumed.resume(source)
        assert resumed.oom_count == 1
        assert resumed.current_microbatch == 2048
        with pytest.raises(SecondOOMError):
            resumed.fit(
                _source_factory(
                    frozen.train_identity,
                    runtime_value_checks=False,
                ),
                _source_factory(
                    frozen.validation_identity,
                    runtime_value_checks=False,
                ),
                resume=True,
                stop_after_examples=4096,
            )
    finally:
        source.close()
        resumed.close()


def test_pending_nonfinite_monitor_rejects_resume_before_live_state_mutation(
    tmp_path: Path,
) -> None:
    frozen = _fixture(tmp_path)
    run_directory = tmp_path / "run"
    writer = _build_harness(frozen, run_directory)
    try:
        writer.sampler = StatefulPermutationSampler(TRAIN_ROWS, seed=42, shuffle=True)
        writer.last_checkpoint_record = writer._save(
            writer.last_checkpoint_path,
            checkpoint_role="last",
        )
    finally:
        writer.close()

    reader = _build_harness(frozen, run_directory)
    model_before = _clone_state(reader.model)
    optimizer_before = copy.deepcopy(reader.optimizer.state_dict())
    rng_before = torch.get_rng_state().clone()
    reader.monitor.submit(
        "pending.nonfinite",
        torch.tensor([float("nan")]),
        kind="activation",
    )
    source = TinyRawSource(frozen.train_identity, runtime_value_checks=False)
    try:
        with pytest.raises(NumericalMonitorError):
            reader.resume(source)
        _assert_state_equal(model_before, reader.model.state_dict())
        _assert_state_equal(optimizer_before, reader.optimizer.state_dict())
        assert torch.equal(rng_before, torch.get_rng_state())
    finally:
        source.close()
        with pytest.raises(NumericalMonitorError):
            reader.close()


def test_metric_free_capacity_preflight_matches_full_artifact_and_common_head(
    tmp_path: Path,
) -> None:
    frozen = _fixture(tmp_path)
    harness = _build_harness(frozen, tmp_path / "run")
    try:
        artifact = load_verified_capacity_artifact(CAPACITY_PATH)
        geometry = artifact["geometry"]
        resolution = resolve_metric_free_capacity_preflight(
            encoder_spec=harness.controlled_model.encoder_spec,
            dcdn_config=DCDNCoreConfig(
                num_layers=geometry["dcdn_num_layers"],
                div_eps=1e-2,
                product_init_scale=0.1,
                div_init_scale=0.1,
                runtime_value_checks=False,
            ),
            ordinary_dcnv2_config=DCNv2FullRankConfig(
                num_layers=geometry["ordinary_dcnv2_num_layers"],
                product_init_scale=0.1,
            ),
            capacity_num_layers=geometry["capacity_num_layers"],
            capacity_num_experts=geometry["capacity_num_experts"],
            capacity_product_init_scale=0.1,
            head_config=HeadConfig(hidden_dims=(), dropout=0.0, output_dim=1),
            init_seed=42,
            search_lock=frozen.search,
        )
        assert resolution == capacity_resolution_projection(artifact)
        assert resolution["target"]["parameters"] == (
            count_trainable_non_embedding_parameters(harness.controlled_model)
        )
        assert resolution["target"]["flops"] == (
            deterministic_forward_flops_per_example(harness.controlled_model)
        )
        ordinary = build_controlled_model(
            encoder_spec=harness.controlled_model.encoder_spec,
            interaction_config=DCNv2FullRankConfig(
                num_layers=geometry["ordinary_dcnv2_num_layers"],
                product_init_scale=0.1,
            ),
            head_config=HeadConfig(hidden_dims=(), dropout=0.0, output_dim=1),
            init_seed=42,
        )
        assert ordinary.head_config == harness.controlled_model.head_config
        assert ordinary.encoder_spec == harness.controlled_model.encoder_spec
        assert (
            count_trainable_non_embedding_parameters(ordinary)
            == (resolution["ordinary"]["parameters"])
        )
        assert (
            deterministic_forward_flops_per_example(ordinary)
            == (resolution["ordinary"]["flops"])
        )
        assert set(resolution["candidates"]) == {
            "capacity_rank_lower",
            "capacity_rank_upper",
        }
    finally:
        harness.close()


def _load_script(name: str) -> Any:
    module_name = f"dcdn_test_{name}"
    path = REPOSITORY_ROOT / "scripts" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    runtime_module_name = "dcdn.data.vk_lsvd_runtime"
    previous_runtime = sys.modules.get(runtime_module_name)
    if name == "train_model":
        runtime_stub = ModuleType(runtime_module_name)
        runtime_stub.C_PLUS_P = "C+P"

        def forbidden_runtime(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("writer unit test must not open the VK runtime")

        runtime_stub.build_vk_lsvd_raw_source = forbidden_runtime
        sys.modules[runtime_module_name] = runtime_stub
    scripts_path = str(path.parent)
    sys.path.insert(0, scripts_path)
    try:
        specification.loader.exec_module(module)
    finally:
        assert sys.path[0] == scripts_path
        sys.path.pop(0)
        if name == "train_model":
            if previous_runtime is None:
                sys.modules.pop(runtime_module_name, None)
            else:
                sys.modules[runtime_module_name] = previous_runtime
    return module


def _write_hashed_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )
    return digest


def test_immutable_result_pair_success_collision_and_second_link_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_model = _load_script("train_model")
    result_path = tmp_path / "result.json"
    payload = {"finite": 1.25, "scope": "validation_only"}
    digest = train_model._write_immutable_json(result_path, payload)
    original_data = result_path.read_bytes()
    original_sidecar = result_path.with_name("result.json.sha256").read_bytes()
    assert verify_sha256_sidecar(result_path) == digest
    with pytest.raises(TrainingContractError, match="exists"):
        train_model._write_immutable_json(result_path, {"different": True})
    assert result_path.read_bytes() == original_data
    assert result_path.with_name("result.json.sha256").read_bytes() == original_sidecar

    real_link = os.link
    calls = 0

    def fail_second_link(source: Any, destination: Any, *args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic sidecar publication failure")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(train_model.os, "link", fail_second_link)
    failed_path = tmp_path / "failed.json"
    with pytest.raises(TrainingContractError, match="publication failed"):
        train_model._write_immutable_json(failed_path, payload)
    assert not failed_path.exists()
    assert not failed_path.with_name("failed.json.sha256").exists()
    assert not list(tmp_path.glob(".failed.json.*.tmp"))
    assert not list(tmp_path.glob(".failed.json.sha256.*.tmp"))


def test_cli_blocks_debug_value_checks_from_scientific_tuning(
    tmp_path: Path,
) -> None:
    train_model = _load_script("train_model")
    frozen = _fixture(tmp_path)
    train_model._require_scientific_runtime_check_mode(frozen.config)
    debug_config = replace(frozen.config, runtime_value_checks=True)
    with pytest.raises(TrainingContractError, match="dedicated one-batch"):
        train_model._require_scientific_runtime_check_mode(debug_config)


def test_cli_binds_exact_views_manifest_before_split_validation(
    tmp_path: Path,
) -> None:
    train_model = _load_script("train_model")
    exact_path = tmp_path / "views.yaml"
    exact_payload = {"split": SPLIT_MAPPING, "identity": "frozen"}
    exact_digest = _write_hashed_yaml(exact_path, exact_payload)

    class Gate:
        def __init__(self) -> None:
            self.payload = {"hashes": {"views_manifest_sha256": exact_digest}}
            self.split_calls = 0

        def require_views_split(self, payload: dict[str, Any]) -> None:
            assert payload == exact_payload
            self.split_calls += 1

    exact_gate = Gate()
    assert (
        train_model._load_exact_views_manifest(exact_gate, exact_path) == exact_payload
    )
    assert exact_gate.split_calls == 1

    altered_path = tmp_path / "altered-views.yaml"
    _write_hashed_yaml(
        altered_path,
        {"split": SPLIT_MAPPING, "identity": "re-signed-alteration"},
    )
    altered_gate = Gate()
    with pytest.raises(TrainingContractError, match="active protocol"):
        train_model._load_exact_views_manifest(altered_gate, altered_path)
    assert altered_gate.split_calls == 0


def _worker_receipt_fixture(
    root: Path,
    *,
    mutation: str | None = None,
) -> tuple[Any, Path, str, dict[str, Any], str]:
    train_model = _load_script("train_model")
    worker = _load_script("run_worker")
    root.mkdir(parents=True, exist_ok=True)
    run_id = "dcdn-test-run"
    protocol_path = root / "protocol.yaml"
    protocol_digest = _write_hashed_yaml(protocol_path, {"kind": "protocol"})
    run_directory = root / "run"
    last_digest = _write_hashed_bytes(run_directory / "last.pt", b"last")
    best_filename = "best-step-000000000001-generation-0000.pt"
    best_digest = _write_hashed_bytes(
        run_directory / best_filename,
        b"best",
    )
    result_path = root / "result.json"
    capacity_digest = sha256_file(CAPACITY_PATH)
    resolved = {
        "run_id": run_id,
        "result_path": str(result_path),
        "protocol_path": str(protocol_path),
        "run_directory": str(run_directory),
        "training": {
            "effective_batch_size": 4096,
            "maximum_train_examples": 40_000_000,
            "reproduction_command": "python scripts/train_model.py --config run.yaml",
            "capacity_resolution_sha256": capacity_digest,
        },
        "runtime": {"device": "cpu"},
    }
    resolved_path = root / "resolved.yaml"
    resolved_digest = _write_hashed_yaml(resolved_path, resolved)
    numerical = {
        "submitted": 6,
        "completed": 6,
        "pending": 0,
        "nonfinite_total": 0,
        "nonfinite_by_kind": {},
        "asynchronous_cuda": False,
        "activation_hook_count": 3,
        "activation_hook_observations": 3,
    }
    receipt = {
        "schema_version": 2,
        "run_id": run_id,
        "resolved_config_sha256": resolved_digest,
        "protocol_sha256": protocol_digest,
        "training": {
            "examples_seen": 4096,
            "optimizer_steps": 1,
            "validation_checks": 1,
            "best_validation_metric": 0.25,
            "stopped_reason": "requested_checkpoint_boundary",
            "elapsed_seconds": 1.0,
            "throughput_examples_per_second": 4096.0,
            "peak_vram_bytes": 0,
            "oom_count": 0,
            "last_checkpoint_sha256": last_digest,
            "best_checkpoint_sha256": best_digest,
            "best_checkpoint_filename": best_filename,
            "numerical_summary": numerical,
            "reproduction_command": resolved["training"]["reproduction_command"],
            "declared_maximum_train_examples": 40_000_000,
            "realized_maximum_train_examples": 39_997_440,
        },
        "selected_validation_metric": 0.25,
        "test_metrics": None,
        "end_to_end_raw_to_logit_validation_latency": None,
        "capacity_resolution_sha256": capacity_digest,
        "joint_model_contract_sha256": hashlib.sha256(b"joint").hexdigest(),
        "final_cost_snapshot": {"launch_allowed": True},
        "final_storage_snapshot": {"ok": True},
        "telemetry": "disabled",
    }
    if mutation == "resolved_digest":
        receipt["resolved_config_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    elif mutation == "protocol_digest":
        receipt["protocol_sha256"] = hashlib.sha256(b"wrong protocol").hexdigest()
    elif mutation == "nonfinite":
        numerical["nonfinite_total"] = 1
        numerical["nonfinite_by_kind"] = {"activation": 1}
    elif mutation == "no_hooks":
        numerical["activation_hook_observations"] = 0
    elif mutation is not None:
        raise AssertionError(f"unknown mutation: {mutation}")
    train_model._write_immutable_json(result_path, receipt)
    return worker, resolved_path, resolved_digest, resolved, run_id


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("resolved_digest", "resolved-config digest"),
        ("protocol_digest", "protocol digest"),
        ("nonfinite", "monitor was not durably drained"),
        ("no_hooks", "hook evidence"),
    ],
)
def test_worker_reopens_exact_result_and_rejects_provenance_or_numerics(
    tmp_path: Path,
    mutation: str | None,
    message: str,
) -> None:
    worker, resolved_path, digest, resolved, run_id = _worker_receipt_fixture(
        tmp_path,
        mutation=mutation,
    )
    with pytest.raises(worker.WorkerError, match=message):
        worker._verify_completed_result(
            project_root=tmp_path,
            resolved_config=resolved_path,
            expected_resolved_digest=digest,
            expected_resolved=resolved,
            run_id=run_id,
        )


def test_worker_accepts_complete_reopened_receipt(tmp_path: Path) -> None:
    worker, resolved_path, digest, resolved, run_id = _worker_receipt_fixture(tmp_path)
    result_digest = worker._verify_completed_result(
        project_root=tmp_path,
        resolved_config=resolved_path,
        expected_resolved_digest=digest,
        expected_resolved=resolved,
        run_id=run_id,
    )
    assert len(result_digest) == 64


def test_worker_code_identity_rejects_diff_drift_and_untracked_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load_script("run_worker")
    commit = "a" * 40
    diff = b"canonical tracked binary diff"
    untracked = b""

    class Completed:
        returncode = 0
        stderr = b""

        def __init__(self, stdout: bytes) -> None:
            self.stdout = stdout

    def fake_run(arguments: list[str], **_kwargs: Any) -> Completed:
        if arguments[1:4] == ["rev-parse", "--verify", "HEAD"]:
            return Completed(f"{commit}\n".encode("ascii"))
        if arguments[1:5] == ["diff", "--binary", "HEAD", "--"]:
            return Completed(diff)
        if arguments[1] == "ls-files":
            return Completed(untracked)
        raise AssertionError(f"unexpected git command: {arguments}")

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    record = {
        "code_commit": commit,
        "code_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }
    worker._require_code_identity(tmp_path, record)
    with pytest.raises(worker.WorkerError, match="code_diff_sha256"):
        worker._require_code_identity(
            tmp_path,
            {**record, "code_diff_sha256": hashlib.sha256(b"drift").hexdigest()},
        )
    untracked = b"src/dcdn/new_module.py\0"
    with pytest.raises(worker.WorkerError, match="untracked"):
        worker._require_code_identity(tmp_path, record)
