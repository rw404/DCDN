from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import signal
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
from dcdn.registry import (
    CHECKPOINT_SIZE_VERIFIER_FAILURE_REASON,
    RunRegistry,
)
from dcdn.sampler import StatefulPermutationSampler
from dcdn.search_space import SearchSpaceLock
from dcdn.selection_artifact import ORDINARY_SELECTION_SHA256
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


class _ReapedProcess:
    pid = 12345

    def poll(self) -> int:
        return 0


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
        binding_fields = (
            {
                "ordinary_dcnv2_selection_sha256": ORDINARY_SELECTION_SHA256,
                "capacity_lr_binding_sha256": "a" * 64,
            }
            if model_key == "dcnv2_capacity_matched_core"
            else {}
        )
        config = replace(
            frozen.config,
            model_key=model_key,
            candidate_id=candidate_id,
            learning_rate=learning_rate,
            **binding_fields,
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

    with pytest.raises(ValueError, match="authorized candidate"):
        replace(
            frozen.config,
            model_key="dcnv2_capacity_matched_core",
            candidate_id="capacity_rank_upper",
            ordinary_dcnv2_selection_sha256=ORDINARY_SELECTION_SHA256,
            capacity_lr_binding_sha256="a" * 64,
        )


def test_c_plus_p_plus_e_does_not_recompute_c_plus_p_capacity_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _fixture(tmp_path)
    artifact = load_verified_capacity_artifact(CAPACITY_PATH)
    projection = capacity_resolution_projection(artifact)
    resolver_calls = 0

    def matching_projection(**_kwargs: Any) -> dict[str, Any]:
        nonlocal resolver_calls
        resolver_calls += 1
        return dict(projection)

    monkeypatch.setattr(
        training_module,
        "resolve_metric_free_capacity_preflight",
        matching_projection,
    )
    cpe_config = replace(
        frozen.config,
        feature_regime="C+P+E",
        model_key="dcnv2_full_rank",
        candidate_id="dcnv2_full_rank_lr_3e_4",
    )
    training_module._validate_capacity_projection_for_feature_regime(
        cpe_config,
        controlled=SimpleNamespace(model_key="dcnv2_full_rank"),
        artifact=artifact,
        projection=projection,
        search_lock=frozen.search,
    )
    assert resolver_calls == 0
    assert (
        training_module._expected_capacity_parameter_count_for_feature_regime(
            cpe_config,
            controlled_model_key="dcnv2_full_rank",
            artifact=artifact,
        )
        is None
    )

    training_module._validate_capacity_projection_for_feature_regime(
        frozen.config,
        controlled=SimpleNamespace(
            model_key="dcdn_core",
            encoder_spec=SimpleNamespace(),
        ),
        artifact=artifact,
        projection=projection,
        search_lock=frozen.search,
    )
    assert resolver_calls == 1
    assert training_module._expected_capacity_parameter_count_for_feature_regime(
        frozen.config,
        controlled_model_key="dcdn_core",
        artifact=artifact,
    ) == int(artifact["target"]["parameters"])

    with pytest.raises(TrainingContractError, match="restricted"):
        training_module._validate_capacity_projection_for_feature_regime(
            replace(
                cpe_config,
                model_key="dcdn_core",
                candidate_id="dcdn_core_l2_lr_3e_4",
            ),
            controlled=SimpleNamespace(model_key="dcdn_core"),
            artifact=artifact,
            projection=projection,
            search_lock=frozen.search,
        )
    with pytest.raises(TrainingContractError, match="restricted"):
        training_module._expected_capacity_parameter_count_for_feature_regime(
            replace(
                cpe_config,
                model_key="dcdn_core",
                candidate_id="dcdn_core_l2_lr_3e_4",
            ),
            controlled_model_key="dcdn_core",
            artifact=artifact,
        )


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
        runtime_stub.C_PLUS_P_PLUS_E = "C+P+E"

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


def test_full_feature_cli_identity_is_isolated_to_dcnv2() -> None:
    train_model = _load_script("train_model")
    encoder = EncoderContractLock.load(ENCODER_PATH)
    dataset = encoder.payload["datasets"][DATASET_ID]
    regime = dataset["regimes"]["C+P+E"]
    gate = SimpleNamespace(
        dataset_id=DATASET_ID,
        feature_regime="C+P+E",
        required_model_keys=("dcnv2_full_rank",),
        payload={
            "hashes": {
                "data_manifest_sha256": dataset["source_manifest_sha256"],
                "views_manifest_sha256": dataset["views_manifest_sha256"],
                "features_sha256": dataset["feature_contract_sha256"],
                "physical_schema_sha256": regime["physical_schema_sha256"],
                "expanded_schema_sha256": regime["expanded_schema_sha256"],
                "operand_schema_sha256": regime["operand_schema_sha256"],
                "denominator_mask_sha256": regime["denominator_mask_sha256"],
                "full_materialized_scan_sha256": dataset[
                    "full_materialized_scan_sha256"
                ],
                "cache_content_sha256": regime["cache_content_sha256"],
            }
        },
        expected_row_count=lambda split: (7_505_820 if split == "train" else 1_516_831),
    )

    identity = train_model._expected_identity(
        gate,
        encoder,
        "train",
        model_key="dcnv2_full_rank",
    )
    assert identity.feature_regime == "C+P+E"
    assert identity.row_count == 7_505_820
    assert identity.ordered_operand_fields == tuple(regime["ordered_operand_fields"])
    assert identity.cache_content_sha256 == regime["cache_content_sha256"]

    with pytest.raises(TrainingContractError, match="restricted"):
        train_model._expected_identity(
            gate,
            encoder,
            "train",
            model_key="dcdn_core",
        )

    gate.required_model_keys = ("dcnv2_full_rank", "dcdn_core")
    with pytest.raises(TrainingContractError, match="isolated"):
        train_model._expected_identity(
            gate,
            encoder,
            "train",
            model_key="dcnv2_full_rank",
        )


def test_worker_full_feature_contract_fails_closed_before_spawn() -> None:
    worker = _load_script("run_worker")
    resolved = {
        "training": {
            "dataset_id": DATASET_ID,
            "feature_regime": "C+P+E",
            "model_key": "dcnv2_full_rank",
        }
    }
    protocol = {
        "dataset_id": DATASET_ID,
        "feature_regime": "C+P+E",
        "required_model_keys": ["dcnv2_full_rank"],
    }
    worker._require_protocol_feature_regime(resolved, protocol)

    wrong_model = copy.deepcopy(resolved)
    wrong_model["training"]["model_key"] = "dcdn_core"
    with pytest.raises(worker.WorkerError, match="restricted"):
        worker._require_protocol_feature_regime(wrong_model, protocol)

    multi_model = copy.deepcopy(protocol)
    multi_model["required_model_keys"] = ["dcnv2_full_rank", "dcdn_core"]
    with pytest.raises(worker.WorkerError, match="isolated"):
        worker._require_protocol_feature_regime(resolved, multi_model)

    wrong_regime = copy.deepcopy(protocol)
    wrong_regime["feature_regime"] = "C+P"
    with pytest.raises(worker.WorkerError, match="differs"):
        worker._require_protocol_feature_regime(resolved, wrong_regime)


def _write_hashed_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )
    return digest


def _write_sparse_hashed_file(path: Path, size_bytes: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.seek(size_bytes - 1)
        handle.write(b"\0")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
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


def test_cli_execution_kind_separates_scientific_and_implementation_scope(
    tmp_path: Path,
) -> None:
    train_model = _load_script("train_model")
    frozen = _fixture(tmp_path)
    scientific_runtime = {
        "resume": True,
        "stop_after_examples": 8192,
        "latency": {
            "warmup_repetitions": 1,
            "measured_repetitions": 1,
            "batch_size": 1,
        },
    }
    assert (
        train_model._validate_execution_contract(
            "scientific_tuning",
            frozen.config,
            scientific_runtime,
        )
        == "scientific_tuning"
    )
    debug_config = replace(frozen.config, runtime_value_checks=True)
    with pytest.raises(TrainingContractError, match="requires runtime_value_checks"):
        train_model._validate_execution_contract(
            "scientific_tuning",
            debug_config,
            scientific_runtime,
        )

    smoke_runtime = {
        "resume": False,
        "stop_after_examples": debug_config.effective_batch_size,
        "latency": None,
    }
    assert (
        train_model._validate_execution_contract(
            "implementation_smoke",
            debug_config,
            smoke_runtime,
        )
        == "implementation_smoke"
    )
    with pytest.raises(TrainingContractError, match="requires runtime_value_checks"):
        train_model._validate_execution_contract(
            "implementation_smoke",
            frozen.config,
            smoke_runtime,
        )
    short_cadence = replace(
        debug_config,
        validate_every_examples=debug_config.effective_batch_size,
    )
    with pytest.raises(TrainingContractError, match="before the first validation"):
        train_model._validate_execution_contract(
            "implementation_smoke",
            short_cadence,
            smoke_runtime,
        )
    wrong_effective_batch = replace(debug_config, effective_batch_size=8192)
    with pytest.raises(TrainingContractError, match="effective_batch_size=4096"):
        train_model._validate_execution_contract(
            "implementation_smoke",
            wrong_effective_batch,
            {**smoke_runtime, "stop_after_examples": 8192},
        )
    for kind in (None, ["implementation_smoke"], "legacy_smoke"):
        with pytest.raises(TrainingContractError, match="execution_kind"):
            train_model._validate_execution_contract(
                kind,
                debug_config,
                smoke_runtime,
            )


def test_final_seed_selection_receipt_is_verified_before_model_creation(
    tmp_path: Path,
) -> None:
    train_model = _load_script("train_model")
    frozen = _fixture(tmp_path)
    model_payload = _model_payload(runtime_value_checks=False)

    selected_config_path = tmp_path / "configs/runs/selected.yaml"
    selected_config_sha256 = _write_hashed_yaml(
        selected_config_path,
        {
            "training": training_module.training_config_payload(frozen.config),
            "model": model_payload,
        },
    )
    result_path = tmp_path / "results/tuning/selected.json"
    result_sha256 = train_model._write_immutable_json(
        result_path,
        {"scope": "validation_only", "test_metrics": None},
    )
    checkpoint_path = tmp_path / "runs/selected/best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"synthetic-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint_path.with_name(f"{checkpoint_path.name}.sha256").write_text(
        f"{checkpoint_sha256}  {checkpoint_path.name}\n",
        encoding="ascii",
    )

    receipt_path = tmp_path / "results/tuning/selection.json"
    receipt_sha256 = train_model._write_immutable_json(
        receipt_path,
        {
            "artifact_role": "per_model_validation_only_selection_evidence",
            "dataset_id": frozen.config.dataset_id,
            "feature_regime": frozen.config.feature_regime,
            "status": "selected_on_validation_only",
            "selected": {
                "candidate_id": frozen.config.candidate_id,
                "learning_rate": frozen.config.learning_rate,
                "resolved_config_path": "configs/runs/selected.yaml",
                "resolved_config_sha256": selected_config_sha256,
                "result_path": "results/tuning/selected.json",
                "result_sha256": result_sha256,
                "best_checkpoint_path": "runs/selected/best.pt",
                "best_checkpoint_sha256": checkpoint_sha256,
                "validation_metric": 0.125,
            },
            "test_firewall": {
                "protocol_test_unlocked": False,
                "protocol_test_path_available": False,
                "selected_on_validation_only": True,
                "test_data_accessed": False,
                "test_metrics_consulted": False,
            },
        },
    )
    final_config = replace(
        frozen.config,
        phase="final",
        seed=43,
        selection_receipt_path="results/tuning/selection.json",
        selection_receipt_sha256=receipt_sha256,
    )
    final_model_payload = copy.deepcopy(model_payload)
    final_model_payload["init_seed"] = 43
    train_model._require_final_seed_selection(
        root=tmp_path,
        training=final_config,
        model_payload=final_model_payload,
        capacity_binding_path=None,
        capacity_binding=None,
    )

    with pytest.raises(TrainingContractError, match="selected candidate"):
        train_model._require_final_seed_selection(
            root=tmp_path,
            training=replace(final_config, candidate_id="dcdn_core_l2_lr_1e_3"),
            model_payload=final_model_payload,
            capacity_binding_path=None,
            capacity_binding=None,
        )
    with pytest.raises(TrainingContractError, match="selection receipt digest"):
        train_model._require_final_seed_selection(
            root=tmp_path,
            training=replace(final_config, selection_receipt_sha256="0" * 64),
            model_payload=final_model_payload,
            capacity_binding_path=None,
            capacity_binding=None,
        )


@pytest.mark.parametrize(
    ("runtime_patch", "message"),
    [
        ({"resume": True}, "fresh run"),
        ({"stop_after_examples": None}, "exactly one effective batch"),
        ({"stop_after_examples": True}, "exactly one effective batch"),
        ({"stop_after_examples": 8192}, "exactly one effective batch"),
        ({"latency": {"batch_size": 1}}, "forbids latency"),
    ],
)
def test_cli_implementation_smoke_contract_fails_closed(
    tmp_path: Path,
    runtime_patch: dict[str, Any],
    message: str,
) -> None:
    train_model = _load_script("train_model")
    debug_config = replace(_fixture(tmp_path).config, runtime_value_checks=True)
    runtime = {
        "resume": False,
        "stop_after_examples": debug_config.effective_batch_size,
        "latency": None,
        **runtime_patch,
    }
    with pytest.raises(TrainingContractError, match=message):
        train_model._validate_execution_contract(
            "implementation_smoke",
            debug_config,
            runtime,
        )


def test_one_batch_implementation_smoke_never_opens_validation(
    tmp_path: Path,
) -> None:
    train_model = _load_script("train_model")
    frozen = _fixture(tmp_path, runtime_value_checks=True)
    harness = _build_harness(frozen, tmp_path / "implementation-smoke")
    opened_train: list[TinyRawSource] = []
    validation_opened = False

    def forbidden_validation() -> TinyRawSource:
        nonlocal validation_opened
        validation_opened = True
        raise AssertionError("implementation smoke opened validation")

    try:
        result = harness.fit(
            _source_factory(
                frozen.train_identity,
                runtime_value_checks=True,
                opened=opened_train,
            ),
            forbidden_validation,
            resume=False,
            stop_after_examples=frozen.config.effective_batch_size,
        )
        assert (
            train_model._validate_result_scope(
                "implementation_smoke",
                frozen.config,
                result,
            )
            is False
        )
        assert result.examples_seen == frozen.config.effective_batch_size
        assert result.optimizer_steps == 1
        assert result.validation_checks == 0
        assert result.best_validation_metric is None
        assert result.best_checkpoint_sha256 is None
        assert result.best_checkpoint_filename is None
        assert validation_opened is False
        assert len(opened_train) == 1 and opened_train[0].close_calls == 1
    finally:
        harness.close()


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
    execution_kind: str = "scientific_tuning",
    mutation: str | None = None,
    oversize_last: bool = False,
) -> tuple[Any, Path, str, dict[str, Any], str]:
    train_model = _load_script("train_model")
    worker = _load_script("run_worker")
    root.mkdir(parents=True, exist_ok=True)
    run_id = "dcdn-test-run"
    smoke = execution_kind == "implementation_smoke"
    if execution_kind not in {"implementation_smoke", "scientific_tuning"}:
        raise AssertionError(f"unsupported fixture execution_kind: {execution_kind}")
    protocol_path = root / "protocol.yaml"
    protocol_digest = _write_hashed_yaml(
        protocol_path,
        {
            "dataset_id": DATASET_ID,
            "feature_regime": FEATURE_REGIME,
            "required_model_keys": ["dcdn_core"],
        },
    )
    run_directory = root / "run"
    last_digest = (
        _write_sparse_hashed_file(run_directory / "last.pt", 64 * 1024 * 1024 + 17)
        if oversize_last
        else _write_hashed_bytes(run_directory / "last.pt", b"last")
    )
    best_filename = "best-step-000000000977-generation-0000.pt"
    best_digest = (
        _write_hashed_bytes(run_directory / best_filename, b"best")
        if not smoke or mutation == "smoke_validation"
        else None
    )
    result_path = root / "result.json"
    capacity_digest = sha256_file(CAPACITY_PATH)
    resolved = {
        "schema_version": 3,
        "execution_kind": execution_kind,
        "run_id": run_id,
        "result_path": str(result_path),
        "protocol_path": str(protocol_path),
        "run_directory": str(run_directory),
        "training": {
            "phase": "tuning",
            "seed": 42,
            "dataset_id": DATASET_ID,
            "feature_regime": FEATURE_REGIME,
            "model_key": "dcdn_core",
            "effective_batch_size": 4096,
            "maximum_train_examples": 40_000_000,
            "validate_every_examples": 4_000_000,
            "runtime_value_checks": smoke,
            "reproduction_command": "python scripts/train_model.py --config run.yaml",
            "capacity_resolution_sha256": capacity_digest,
        },
        "runtime": {
            "device": "cpu",
            "resume": False,
            "stop_after_examples": 4096 if smoke else None,
            "latency": None,
        },
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
        "schema_version": 3,
        "execution_kind": execution_kind,
        "scientific_result_eligible": not smoke,
        "run_id": run_id,
        "resolved_config_sha256": resolved_digest,
        "protocol_sha256": protocol_digest,
        "training": {
            "examples_seen": 4096 if smoke else 4_001_792,
            "optimizer_steps": 1 if smoke else 977,
            "validation_checks": 0 if smoke else 1,
            "best_validation_metric": None if smoke else 0.25,
            "stopped_reason": "requested_checkpoint_boundary",
            "elapsed_seconds": 1.0,
            "throughput_examples_per_second": 4096.0,
            "peak_vram_bytes": 0,
            "oom_count": 0,
            "last_checkpoint_sha256": last_digest,
            "best_checkpoint_sha256": None if smoke else best_digest,
            "best_checkpoint_filename": None if smoke else best_filename,
            "numerical_summary": numerical,
            "reproduction_command": resolved["training"]["reproduction_command"],
            "declared_maximum_train_examples": 40_000_000,
            "realized_maximum_train_examples": 39_997_440,
        },
        "selected_validation_metric": None if smoke else 0.25,
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
    elif mutation == "execution_kind":
        receipt["execution_kind"] = (
            "scientific_tuning" if smoke else "implementation_smoke"
        )
    elif mutation == "eligibility":
        receipt["scientific_result_eligible"] = smoke
    elif mutation == "schema_float":
        receipt["schema_version"] = 3.0
    elif mutation == "examples_float":
        receipt["training"]["examples_seen"] = 4_001_792.0
    elif mutation == "elapsed_int":
        receipt["training"]["elapsed_seconds"] = 1
    elif mutation == "metric_int":
        receipt["training"]["best_validation_metric"] = 0
        receipt["selected_validation_metric"] = 0
    elif mutation == "smoke_validation":
        if not smoke or best_digest is None:
            raise AssertionError("smoke_validation requires implementation_smoke")
        receipt["training"]["validation_checks"] = 1
        receipt["training"]["best_validation_metric"] = 0.25
        receipt["training"]["best_checkpoint_sha256"] = best_digest
        receipt["training"]["best_checkpoint_filename"] = best_filename
        receipt["selected_validation_metric"] = 0.25
    elif mutation == "smoke_examples":
        if not smoke:
            raise AssertionError("smoke_examples requires implementation_smoke")
        receipt["training"]["examples_seen"] = 8192
        receipt["training"]["optimizer_steps"] = 2
    elif mutation == "smoke_latency":
        if not smoke:
            raise AssertionError("smoke_latency requires implementation_smoke")
        receipt["end_to_end_raw_to_logit_validation_latency"] = {
            "warmup_repetitions": 1,
            "measured_repetitions": 2,
            "batch_size": 1,
            "p50_ms": 1.0,
            "p95_ms": 1.1,
            "p99_ms": 1.2,
            "scope": "end_to_end_raw_to_logit_model",
        }
    elif mutation is not None:
        raise AssertionError(f"unknown mutation: {mutation}")
    train_model._write_immutable_json(result_path, receipt)
    return worker, resolved_path, resolved_digest, resolved, run_id


@pytest.mark.parametrize(
    ("execution_kind", "mutation", "message"),
    [
        ("scientific_tuning", "resolved_digest", "resolved-config digest"),
        ("scientific_tuning", "protocol_digest", "protocol digest"),
        ("scientific_tuning", "nonfinite", "monitor was not durably drained"),
        ("scientific_tuning", "no_hooks", "hook evidence"),
        ("scientific_tuning", "execution_kind", "execution_kind mismatch"),
        ("scientific_tuning", "eligibility", "eligibility is inconsistent"),
        ("scientific_tuning", "schema_float", "identity mismatch"),
        ("scientific_tuning", "examples_float", "examples/effective-batch"),
        ("scientific_tuning", "elapsed_int", "elapsed_seconds"),
        ("scientific_tuning", "metric_int", "best generation"),
        ("implementation_smoke", "execution_kind", "execution_kind mismatch"),
        ("implementation_smoke", "eligibility", "explicitly non-scientific"),
        ("implementation_smoke", "smoke_validation", "contains validation"),
        ("implementation_smoke", "smoke_examples", "contains validation"),
        ("implementation_smoke", "smoke_latency", "contains validation"),
    ],
)
def test_worker_reopens_exact_result_and_rejects_provenance_or_numerics(
    tmp_path: Path,
    execution_kind: str,
    mutation: str | None,
    message: str,
) -> None:
    worker, resolved_path, digest, resolved, run_id = _worker_receipt_fixture(
        tmp_path,
        execution_kind=execution_kind,
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


@pytest.mark.parametrize(
    ("execution_kind", "expected_eligibility"),
    [("scientific_tuning", True), ("implementation_smoke", False)],
)
def test_worker_accepts_complete_reopened_receipt(
    tmp_path: Path,
    execution_kind: str,
    expected_eligibility: bool,
) -> None:
    worker, resolved_path, digest, resolved, run_id = _worker_receipt_fixture(
        tmp_path,
        execution_kind=execution_kind,
    )
    result_digest, eligibility = worker._verify_completed_result(
        project_root=tmp_path,
        resolved_config=resolved_path,
        expected_resolved_digest=digest,
        expected_resolved=resolved,
        run_id=run_id,
    )
    assert len(result_digest) == 64
    assert eligibility is expected_eligibility


def test_worker_reconciles_known_oversize_checkpoint_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, resolved_path, digest, resolved, run_id = _worker_receipt_fixture(
        tmp_path,
        execution_kind="scientific_tuning",
        oversize_last=True,
    )
    monkeypatch.setattr(worker, "PROJECT_ROOT", tmp_path)
    registry_root = tmp_path / "runs" / "registry"
    registry = RunRegistry(registry_root)
    execution_commit = "b" * 40
    empty_diff = hashlib.sha256(b"").hexdigest()
    registry.create_pending(
        run_id=run_id,
        execution_kind="scientific_tuning",
        seed=42,
        code_commit=execution_commit,
        code_diff_sha256=empty_diff,
        resolved_config_path=resolved_path.relative_to(tmp_path).as_posix(),
        resolved_config_sha256=digest,
        data_manifest_sha256="1" * 64,
        features_sha256="2" * 64,
        protocol_sha256=verify_sha256_sidecar(Path(resolved["protocol_path"])),
        reproduction_command_path="runs/commands/reconcile.sh",
        history_path="runs/reconcile/history.jsonl",
        log_path="runs/reconcile/worker.log",
    )
    registry.ensure_running(run_id, pid=98765, gpu=0)
    failed = registry.ensure_terminal(
        run_id,
        "failed",
        pid=98765,
        gpu=0,
        reason=CHECKPOINT_SIZE_VERIFIER_FAILURE_REASON,
    )

    def missing_pid(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    verifier_commit = "c" * 40

    def fake_git(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        if command[1:] == ["rev-parse", "--verify", "HEAD"]:
            stdout = f"{verifier_commit}\n".encode("ascii")
        elif command[1:4] == ["diff", "--binary", "HEAD"] or command[1:4] == [
            "ls-files",
            "--others",
            "--exclude-standard",
        ]:
            stdout = b""
        else:
            raise AssertionError(f"unexpected Git command: {command}")
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(worker.os, "kill", missing_pid)
    monkeypatch.setattr(worker.subprocess, "run", fake_git)
    assert (
        worker.reconcile_checkpoint_size_failure(
            project_root=tmp_path,
            registry_root=registry_root,
            run_id=run_id,
            gpu=0,
            resolved_config=resolved_path,
        )
        == 0
    )
    completed = registry.read(run_id)
    assert completed["status"] == "completed"
    assert completed["scientific_result_eligible"] is True
    assert completed["terminal_reconciliation"]["prior_record"] == failed
    assert completed["terminal_reconciliation"][
        "result_sha256"
    ] == verify_sha256_sidecar(Path(resolved["result_path"]))


def test_worker_execution_kind_preflight_rejects_legacy_and_registry_drift(
    tmp_path: Path,
) -> None:
    worker, _path, _digest, resolved, _run_id = _worker_receipt_fixture(
        tmp_path,
        execution_kind="implementation_smoke",
    )
    assert (
        worker._validate_resolved_execution_contract(resolved) == "implementation_smoke"
    )

    legacy = copy.deepcopy(resolved)
    legacy.pop("execution_kind")
    with pytest.raises(worker.WorkerError, match="execution_kind"):
        worker._validate_resolved_execution_contract(legacy)

    wrong_checks = copy.deepcopy(resolved)
    wrong_checks["training"]["runtime_value_checks"] = False
    with pytest.raises(worker.WorkerError, match="runtime_value_checks"):
        worker._validate_resolved_execution_contract(wrong_checks)

    wrong_stop = copy.deepcopy(resolved)
    wrong_stop["runtime"]["stop_after_examples"] = 8192
    with pytest.raises(worker.WorkerError, match="exactly one effective batch"):
        worker._validate_resolved_execution_contract(wrong_stop)

    missing_latency = copy.deepcopy(resolved)
    missing_latency["runtime"].pop("latency")
    with pytest.raises(worker.WorkerError, match="forbids latency"):
        worker._validate_resolved_execution_contract(missing_latency)

    wrong_effective_batch = copy.deepcopy(resolved)
    wrong_effective_batch["training"]["effective_batch_size"] = 8192
    wrong_effective_batch["runtime"]["stop_after_examples"] = 8192
    with pytest.raises(worker.WorkerError, match="training budget"):
        worker._validate_resolved_execution_contract(wrong_effective_batch)

    scientific = copy.deepcopy(resolved)
    scientific["execution_kind"] = "scientific_tuning"
    scientific["training"]["runtime_value_checks"] = False
    scientific["runtime"] = {
        "device": "cpu",
        "resume": True,
        "stop_after_examples": 8192,
        "latency": {"batch_size": 64},
    }
    assert (
        worker._validate_resolved_execution_contract(scientific) == "scientific_tuning"
    )

    receipt_digest = _write_hashed_bytes(tmp_path / "selection.json", b"{}")
    final = copy.deepcopy(scientific)
    final["training"].update(
        {
            "phase": "final",
            "seed": 43,
            "selection_receipt_path": "selection.json",
            "selection_receipt_sha256": receipt_digest,
        }
    )
    assert worker._validate_resolved_execution_contract(final) == "scientific_tuning"
    worker._verify_final_selection_receipt(final, tmp_path, capacity_evidence=None)

    wrong_final_seed = copy.deepcopy(final)
    wrong_final_seed["training"]["seed"] = 42
    with pytest.raises(worker.WorkerError, match="seeds 43 and 44"):
        worker._validate_resolved_execution_contract(wrong_final_seed)

    missing_receipt = copy.deepcopy(final)
    missing_receipt["training"].pop("selection_receipt_sha256")
    with pytest.raises(worker.WorkerError, match="requires a selection receipt"):
        worker._validate_resolved_execution_contract(missing_receipt)

    final_smoke = copy.deepcopy(final)
    final_smoke["execution_kind"] = "implementation_smoke"
    with pytest.raises(worker.WorkerError, match="must be scientific tuning"):
        worker._validate_resolved_execution_contract(final_smoke)

    wrong_receipt = copy.deepcopy(final)
    wrong_receipt["training"]["selection_receipt_sha256"] = "0" * 64
    with pytest.raises(worker.WorkerError, match="digest differs"):
        worker._verify_final_selection_receipt(
            wrong_receipt, tmp_path, capacity_evidence=None
        )

    with pytest.raises(worker.WorkerError, match="registry execution_kind"):
        worker._require_registry_execution_kind(
            {"execution_kind": "scientific_tuning"},
            "implementation_smoke",
        )
    with pytest.raises(worker.WorkerError, match="registry execution_kind"):
        worker._require_registry_execution_kind({}, "implementation_smoke")
    smoke_reason = worker._completed_reason(
        "implementation_smoke",
        "a" * 64,
    )
    assert smoke_reason.startswith("NON-SCIENTIFIC implementation smoke;")
    assert worker._completed_reason("scientific_tuning", "b" * 64).startswith(
        "scientific tuning;"
    )


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


def test_registry_helper_timeout_is_transient_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load_script("run_worker")

    class FakeProcess:
        pid = 9876
        stdin = None
        stdout = __import__("io").BytesIO()
        stderr = __import__("io").BytesIO()

        def __init__(self) -> None:
            self.running = True

        def poll(self) -> int | None:
            return None if self.running else -signal.SIGKILL

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.running = False
            return -signal.SIGKILL

    process = FakeProcess()
    monkeypatch.setattr(
        worker.os,
        "killpg",
        lambda _pid, _signal: setattr(process, "running", False),
    )
    pending = worker._PendingRegistryOperation(
        process=process,
        request={"operation": "heartbeat"},
        started_at=0.0,
    )
    assert worker._poll_registry_operation(pending, 9.999) is None
    result = worker._poll_registry_operation(pending, 10.0)
    assert result is not None
    assert result.ok is False
    assert result.transient_candidate is True
    assert result.stage == "helper_timeout"


def test_heartbeat_controller_retries_then_persists_incident() -> None:
    worker = _load_script("run_worker")
    starts: list[tuple[dict[str, Any], float]] = []
    outcomes = [
        worker._RegistryOperationResult(
            ok=False,
            transient_candidate=True,
            error_type="RegistryIOError",
            message="synthetic EIO",
            stage="record_replace",
            errno=5,
        ),
        worker._RegistryOperationResult(
            ok=True,
            result={
                "status": "running",
                "incident": {"path": "/persistent/incident", "sha256": "c" * 64},
            },
        ),
    ]

    def start(request: Any, now: float, owner: Any) -> Any:
        starts.append((dict(request), now))
        pending = worker._PendingRegistryOperation(
            _ReapedProcess(),
            dict(request),
            now,
        )
        owner.adopt(pending)
        return pending

    def poll(_pending: Any, _now: float) -> Any:
        return outcomes.pop(0)

    local_payloads: list[dict[str, Any]] = []

    def persist_local(_overlay: Path, _registry: Path, payload: Any) -> Any:
        local_payloads.append(dict(payload))
        return {"path": "/overlay/incident", "sha256": "c" * 64}

    controller = worker._HeartbeatController(
        registry_root=Path("/persistent/registry"),
        overlay_path=Path("/local/overlay"),
        run_id="controller-recovery",
        pid=321,
        now=0.0,
        start_fn=start,
        poll_fn=poll,
        emergency_writer=persist_local,
    )
    assert controller.tick(29.999) is None
    assert starts == []
    assert controller.tick(30.0) is None
    assert controller.tick(30.1) is None
    assert len(local_payloads) == 1
    assert controller.local_receipt is not None
    assert controller.tick(31.2) is None
    assert "incident" in starts[-1][0]
    assert controller.tick(31.3) is None
    assert controller.pending_incident is None
    assert controller.local_receipt is None
    assert controller.persistent_receipts[0]["sha256"] == "c" * 64


def test_heartbeat_controller_stops_on_hard_error_and_180_second_deadline() -> None:
    worker = _load_script("run_worker")

    def controller_for(result: Any) -> Any:
        def start(request: Any, now: float, owner: Any) -> Any:
            pending = worker._PendingRegistryOperation(
                _ReapedProcess(),
                dict(request),
                now,
            )
            owner.adopt(pending)
            return pending

        return worker._HeartbeatController(
            registry_root=Path("/persistent/registry"),
            overlay_path=Path("/local/overlay"),
            run_id="controller-failure",
            pid=654,
            now=0.0,
            start_fn=start,
            poll_fn=lambda _pending, _now: result,
            emergency_writer=lambda *_args: {
                "path": "/overlay/incident",
                "sha256": "e" * 64,
            },
        )

    hard = controller_for(
        worker._RegistryOperationResult(
            ok=False,
            transient_candidate=False,
            error_type="RegistryIOError",
            message="no space",
            stage="temporary_create",
            errno=28,
        )
    )
    assert hard.tick(30.0) is None
    hard_reason = hard.tick(30.1)
    assert hard_reason is not None
    assert "hard registry heartbeat failure" in hard_reason

    transient = controller_for(
        worker._RegistryOperationResult(
            ok=False,
            transient_candidate=True,
            error_type="RegistryIOError",
            message="synthetic EIO",
            stage="record_replace",
            errno=5,
        )
    )
    assert transient.tick(30.0) is None
    assert transient.tick(30.1) is None
    deadline_reason = transient.tick(210.0)
    assert deadline_reason is not None
    assert "deadline" in deadline_reason


def test_terminal_registry_retry_reuses_the_exact_intent() -> None:
    worker = _load_script("run_worker")
    clock = [0.0]
    requests: list[dict[str, Any]] = []
    outcomes = [
        worker._RegistryOperationResult(
            ok=False,
            transient_candidate=True,
            error_type="RegistryIOError",
            message="synthetic EIO",
            stage="directory_fsync",
            errno=5,
        ),
        worker._RegistryOperationResult(
            ok=True,
            result={"status": "completed"},
        ),
    ]

    def start(request: Any, now: float, owner: Any) -> Any:
        requests.append(dict(request))
        pending = worker._PendingRegistryOperation(
            _ReapedProcess(),
            dict(request),
            now,
        )
        owner.adopt(pending)
        return pending

    intent = {
        "operation": "ensure_terminal",
        "registry_root": "/persistent/registry",
        "run_id": "exact-terminal-intent",
        "status": "completed",
        "pid": 999,
        "gpu": 1,
        "reason": "verified immutable result",
        "scientific_result_eligible": True,
    }
    result = worker._run_registry_operation_until(
        intent,
        deadline_monotonic=100.0,
        helper_owner=worker._RegistryHelperOwner(),
        now_fn=lambda: clock[0],
        sleep_fn=lambda delay: clock.__setitem__(0, clock[0] + delay),
        start_fn=start,
        poll_fn=lambda _pending, _now: outcomes.pop(0),
    )
    assert result == {"status": "completed"}
    assert requests == [intent, intent]


@pytest.mark.parametrize("deadline", [30.0, 300.0])
def test_synchronous_registry_deadline_retains_unreaped_owner(
    monkeypatch: pytest.MonkeyPatch,
    deadline: float,
) -> None:
    worker = _load_script("run_worker")

    class NeverReapedProcess:
        pid = 8765
        stdin = None
        stdout = __import__("io").BytesIO()
        stderr = __import__("io").BytesIO()

        def __init__(self) -> None:
            self.wait_calls = 0

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.wait_calls += 1
            raise worker.subprocess.TimeoutExpired("registry-helper", 0)

    process = NeverReapedProcess()
    clock = [0.0]
    starts: list[Any] = []
    kills: list[tuple[int, int]] = []
    owner = worker._RegistryHelperOwner()

    def start(request: Any, now: float, helper_owner: Any) -> Any:
        pending = worker._PendingRegistryOperation(process, dict(request), now)
        helper_owner.adopt(pending)
        starts.append(pending)
        return pending

    def poll(pending: Any, now: float) -> Any:
        if now >= 10.0 and not pending.timeout_reported:
            worker._signal_pending_registry_helper(pending)
            pending.timeout_reported = True
            return worker._RegistryOperationResult(
                ok=False,
                transient_candidate=True,
                error_type="RegistryHelperTimeout",
                message="synthetic never-reaped helper",
                stage="helper_timeout",
                retain_pending=True,
            )
        return None

    monkeypatch.setattr(
        worker.os,
        "killpg",
        lambda pid, signum: kills.append((pid, signum)),
    )
    with pytest.raises(worker.WorkerError, match="absolute deadline"):
        worker._run_registry_operation_until(
            {"operation": "ensure_terminal"},
            deadline_monotonic=deadline,
            helper_owner=owner,
            now_fn=lambda: clock[0],
            sleep_fn=lambda _delay: clock.__setitem__(0, clock[0] + 10.0),
            start_fn=start,
            poll_fn=poll,
        )
    assert len(starts) == 1
    assert owner.pending is starts[0]
    assert kills == [(process.pid, signal.SIGKILL)]
    with pytest.raises(worker.WorkerError, match="cannot release"):
        owner.release(starts[0])

    with pytest.raises(worker.WorkerError, match="remains unreaped"):
        worker._run_registry_operation_until(
            {"operation": "ensure_terminal"},
            deadline_monotonic=deadline + 30.0,
            helper_owner=owner,
            now_fn=lambda: clock[0],
            start_fn=start,
            poll_fn=poll,
        )
    assert len(starts) == 1

    with pytest.raises(worker.WorkerError, match="cleanup deadline"):
        owner.close(
            deadline_monotonic=deadline + 0.2,
            now_fn=lambda: clock[0],
            sleep_fn=lambda delay: clock.__setitem__(0, clock[0] + delay),
        )
    assert process.wait_calls >= 1
    assert owner.pending is starts[0]
    assert kills == [(process.pid, signal.SIGKILL)]


def test_unreaped_timeout_keeps_training_until_180_and_retries_after_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load_script("run_worker")

    class DelayedReapProcess:
        pid = 7654
        stdin = None
        stdout = __import__("io").BytesIO()
        stderr = __import__("io").BytesIO()

        def __init__(self, reap_at: float) -> None:
            self.now = 0.0
            self.reap_at = reap_at

        def poll(self) -> int | None:
            return -signal.SIGKILL if self.now >= self.reap_at else None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.now < self.reap_at:
                raise worker.subprocess.TimeoutExpired("helper", 0)
            return -signal.SIGKILL

    kills: list[tuple[int, int]] = []
    monkeypatch.setattr(
        worker.os,
        "killpg",
        lambda pid, signum: kills.append((pid, signum)),
    )
    stuck = DelayedReapProcess(reap_at=999.0)
    starts: list[Any] = []

    def start_stuck(request: Any, now: float, owner: Any) -> Any:
        pending = worker._PendingRegistryOperation(stuck, dict(request), now)
        owner.adopt(pending)
        starts.append(pending)
        return pending

    stuck_controller = worker._HeartbeatController(
        registry_root=Path("/persistent/registry"),
        overlay_path=Path("/local/overlay"),
        run_id="unreaped-deadline",
        pid=765,
        now=-30.0,
        start_fn=start_stuck,
        emergency_writer=lambda *_args: {
            "path": "/overlay/incident",
            "sha256": "f" * 64,
        },
    )
    assert stuck_controller.tick(0.0) is None
    stuck.now = 10.0
    assert stuck_controller.tick(10.0) is None
    assert len(starts) == 1
    assert kills == [(stuck.pid, signal.SIGKILL)]
    stuck.now = 179.0
    assert stuck_controller.tick(179.0) is None
    assert len(starts) == 1
    assert len(kills) == 1
    deadline_reason = stuck_controller.tick(180.0)
    assert deadline_reason is not None
    assert "deadline" in deadline_reason
    assert len(kills) == 1

    recovering = DelayedReapProcess(reap_at=20.0)
    recovery_starts: list[Any] = []

    def start_recovering(request: Any, now: float, owner: Any) -> Any:
        if not recovery_starts:
            pending = worker._PendingRegistryOperation(
                recovering,
                dict(request),
                now,
            )
        else:
            pending = worker._PendingRegistryOperation(
                _ReapedProcess(),
                dict(request),
                now,
            )
        owner.adopt(pending)
        recovery_starts.append(pending)
        return pending

    def poll_recovering(pending: Any, now: float) -> Any:
        if pending.process is recovering:
            recovering.now = now
            return worker._poll_registry_operation(pending, now)
        return worker._RegistryOperationResult(
            ok=True,
            result={
                "status": "running",
                "incident": {
                    "path": "/persistent/incident",
                    "sha256": "a" * 64,
                },
            },
        )

    recovered = worker._HeartbeatController(
        registry_root=Path("/persistent/registry"),
        overlay_path=Path("/local/overlay"),
        run_id="unreaped-recovery",
        pid=766,
        now=-30.0,
        start_fn=start_recovering,
        poll_fn=poll_recovering,
        emergency_writer=lambda *_args: {
            "path": "/overlay/incident",
            "sha256": "a" * 64,
        },
    )
    assert recovered.tick(0.0) is None
    assert recovered.tick(10.0) is None
    assert len(recovery_starts) == 1
    assert recovered.tick(19.0) is None
    assert len(recovery_starts) == 1
    assert recovered.tick(20.0) is None
    assert recovered.tick(20.1) is None
    assert len(recovery_starts) == 2
    assert "incident" in recovery_starts[1].request
    assert recovered.tick(20.2) is None
    assert recovered.pending_incident is None


def test_worker_trainer_command_binds_expected_parent() -> None:
    worker = _load_script("run_worker")
    resolved = Path("/persistent/config.yaml")
    command = worker._trainer_command(resolved)
    separator = command.index("--")
    assert command[:separator] == [
        sys.executable,
        "-m",
        "dcdn.supervised_exec",
        "--expected-parent-pid",
        str(os.getpid()),
    ]
    assert command[separator + 1 :] == [
        sys.executable,
        str(worker._TRAIN_SCRIPT),
        "run",
        "--resolved",
        str(resolved),
    ]


def test_registry_helper_command_binds_expected_parent() -> None:
    worker = _load_script("run_worker")
    command = worker._registry_helper_command()
    separator = command.index("--")
    assert command[:separator] == [
        sys.executable,
        "-m",
        "dcdn.supervised_exec",
        "--expected-parent-pid",
        str(os.getpid()),
    ]
    assert command[separator + 1 :] == [
        sys.executable,
        "-m",
        "dcdn.registry_helper",
    ]


def test_training_config_capacity_bindings_are_exact_and_legacy_payload_is_stable(
    tmp_path: Path,
) -> None:
    frozen = _fixture(tmp_path)
    legacy = training_module.training_config_payload(frozen.config)
    assert "ordinary_dcnv2_selection_sha256" not in legacy
    assert "capacity_lr_binding_sha256" not in legacy
    assert "selection_receipt_path" not in legacy
    assert "selection_receipt_sha256" not in legacy

    capacity = replace(
        frozen.config,
        model_key="dcnv2_capacity_matched_core",
        candidate_id="capacity_rank_lower",
        learning_rate=0.0003,
        ordinary_dcnv2_selection_sha256=ORDINARY_SELECTION_SHA256,
        capacity_lr_binding_sha256="a" * 64,
    )
    serialized = training_module.training_config_payload(capacity)
    assert serialized["ordinary_dcnv2_selection_sha256"] == (ORDINARY_SELECTION_SHA256)
    assert serialized["capacity_lr_binding_sha256"] == "a" * 64
    final_config = replace(
        frozen.config,
        phase="final",
        seed=43,
        selection_receipt_path="results/tuning/selected.json",
        selection_receipt_sha256="b" * 64,
    )
    final_serialized = training_module.training_config_payload(final_config)
    assert final_serialized["selection_receipt_path"] == (
        "results/tuning/selected.json"
    )
    assert final_serialized["selection_receipt_sha256"] == "b" * 64
    with pytest.raises(ValueError, match="final-phase pair"):
        replace(
            frozen.config,
            selection_receipt_path="results/tuning/selected.json",
            selection_receipt_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="final-phase pair"):
        replace(final_config, selection_receipt_sha256=None)

    with pytest.raises(ValueError, match="capacity-only"):
        replace(
            frozen.config,
            ordinary_dcnv2_selection_sha256=ORDINARY_SELECTION_SHA256,
            capacity_lr_binding_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="authorized candidate"):
        replace(
            capacity,
            candidate_id="capacity_rank_upper",
        )
    with pytest.raises(ValueError, match="authorized candidate"):
        replace(capacity, learning_rate=0.001)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 1),
        ("gradient_clip_l2", True),
        ("adam_betas", (0.9, True)),
    ],
)
def test_training_config_rejects_type_equivalent_scalars(
    tmp_path: Path, field: str, value: object
) -> None:
    frozen = _fixture(tmp_path)
    with pytest.raises(TypeError, match="exact float"):
        replace(frozen.config, **{field: value})


def test_head_config_rejects_boolean_dropout() -> None:
    with pytest.raises(TypeError, match="exact float"):
        HeadConfig(hidden_dims=(), dropout=False, output_dim=1)


def test_capacity_binding_failure_precedes_optimizer_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _fixture(tmp_path)
    capacity_config = replace(
        frozen.config,
        model_key="dcnv2_capacity_matched_core",
        candidate_id="capacity_rank_lower",
        learning_rate=0.0003,
        ordinary_dcnv2_selection_sha256=ORDINARY_SELECTION_SHA256,
        capacity_lr_binding_sha256="a" * 64,
    )
    model = build_trainable_encoded_model(
        encoder_contract=frozen.encoder,
        dataset_id=DATASET_ID,
        source_identity=frozen.train_identity,
        interaction_config=DCNv2CapacityCoreConfig(
            num_layers=2,
            low_rank=12,
            num_experts=4,
            product_init_scale=0.1,
        ),
        head_config=HeadConfig(hidden_dims=(), dropout=0.0, output_dim=1),
        run_seed=42,
        runtime_value_checks=True,
    )
    optimizer_called = False

    def forbidden_optimizer(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal optimizer_called
        optimizer_called = True
        raise AssertionError("optimizer must not be created")

    def reject_binding(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("synthetic missing authorization")

    monkeypatch.setattr(training_module, "load_capacity_lr_binding", reject_binding)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden_optimizer)
    with pytest.raises(TrainingContractError, match="before optimizer"):
        TrainingHarness(
            model=model,
            config=capacity_config,
            safety=frozen.safety,
            search_lock=frozen.search,
            capacity_resolution=None,
            operand_feature_order=frozen.train_identity.ordered_operand_fields,
            run_directory=tmp_path / "blocked-capacity",
            device="cpu",
            capacity_artifact_path=CAPACITY_PATH,
            capacity_lr_binding_path=tmp_path / "missing-auth.json",
            repository_root=REPOSITORY_ROOT,
        )
    assert optimizer_called is False


@pytest.mark.parametrize("schema_version", [3.0, True])
def test_trainer_schema_requires_exact_integer(
    tmp_path: Path, schema_version: object
) -> None:
    train_model = _load_script("train_model")
    resolved = tmp_path / "typed-schema.yaml"
    _write_hashed_yaml(
        resolved,
        {
            "schema_version": schema_version,
            "training": {"model_key": "dcnv2_full_rank"},
        },
    )
    with pytest.raises(TrainingContractError, match="exact integer"):
        train_model._run(resolved)


def test_worker_schema_v3_noncapacity_and_v4_capacity_are_disjoint() -> None:
    worker = _load_script("run_worker")
    base = {
        "schema_version": 3,
        "execution_kind": "scientific_tuning",
        "training": {
            "dataset_id": DATASET_ID,
            "feature_regime": FEATURE_REGIME,
            "phase": "tuning",
            "seed": 42,
            "model_key": "dcnv2_full_rank",
            "runtime_value_checks": False,
        },
        "runtime": {},
    }
    assert worker._validate_resolved_execution_contract(base) == "scientific_tuning"

    for invalid_schema in (3.0, True):
        invalid = copy.deepcopy(base)
        invalid["schema_version"] = invalid_schema
        with pytest.raises(worker.WorkerError, match="v3 non-capacity or v4 capacity"):
            worker._validate_resolved_execution_contract(invalid)

    v3_capacity = copy.deepcopy(base)
    v3_capacity["training"]["model_key"] = "dcnv2_capacity_matched_core"
    with pytest.raises(worker.WorkerError, match="v3 non-capacity or v4 capacity"):
        worker._validate_resolved_execution_contract(v3_capacity)

    v4_noncapacity = copy.deepcopy(base)
    v4_noncapacity["schema_version"] = 4
    with pytest.raises(worker.WorkerError, match="v3 non-capacity or v4 capacity"):
        worker._validate_resolved_execution_contract(v4_noncapacity)

    v4_capacity = copy.deepcopy(v3_capacity)
    v4_capacity["schema_version"] = 4
    assert (
        worker._validate_resolved_execution_contract(v4_capacity) == "scientific_tuning"
    )


def test_worker_capacity_checkpoint_inspector_rejects_rehashed_binding_tamper(
    tmp_path: Path,
) -> None:
    from dcdn.checkpoint import (
        CHECKPOINT_FORMAT,
        CHECKPOINT_VERSION,
        CheckpointMetadata,
    )
    from dcdn.selection_artifact import (
        CAPACITY_RESOLUTION_SHA256,
        PREMETRIC_PROTOCOL_SHA256,
        capacity_training_binding_payload,
    )

    worker = _load_script("run_worker")
    joint_digest = "f" * 64
    training = {
        "model_key": "dcnv2_capacity_matched_core",
        "candidate_id": "capacity_rank_lower",
        "learning_rate": 0.0003,
        "capacity_resolution_sha256": CAPACITY_RESOLUTION_SHA256,
        "ordinary_dcnv2_selection_sha256": ORDINARY_SELECTION_SHA256,
        "capacity_lr_binding_sha256": "a" * 64,
        "resolved_model_config_sha256": "b" * 64,
    }
    binding = capacity_training_binding_payload(
        capacity_resolution_sha256=CAPACITY_RESOLUTION_SHA256,
        ordinary_dcnv2_selection_sha256=ORDINARY_SELECTION_SHA256,
        capacity_lr_binding_sha256="a" * 64,
    )

    def write_checkpoint(path: Path, *, include_binding: bool) -> str:
        resolved_checkpoint = {
            "training": dict(training),
            "capacity_resolution_sha256": CAPACITY_RESOLUTION_SHA256,
            "checkpoint_protocol_sha256": PREMETRIC_PROTOCOL_SHA256,
            "joint_model_contract_sha256": joint_digest,
        }
        if include_binding:
            resolved_checkpoint["capacity_training_binding"] = binding
        metadata = CheckpointMetadata(
            resolved_config=resolved_checkpoint,
            data_hash="c" * 64,
            feature_hash="d" * 64,
            protocol_hash=PREMETRIC_PROTOCOL_SHA256,
            feature_order=("operand",),
            denominator_mask=(True,),
            div_eps=0.01,
            depth=2,
            precision="fp32_parameters_bf16_autocast",
        )
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "metadata": metadata.to_dict(),
            "progress": {},
            "components": {},
            "rng_state": {},
            "extra_state": {
                "checkpoint_role": "last",
                "selected_best_checkpoint_sha256": None,
                "selected_best_checkpoint_filename": None,
            },
        }
        torch.save(payload, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name(f"{path.name}.sha256").write_text(
            f"{digest}  {path.name}\n", encoding="ascii"
        )
        return digest

    checkpoint = tmp_path / "last.pt"
    digest = write_checkpoint(checkpoint, include_binding=True)
    worker._verify_capacity_checkpoint_binding(
        checkpoint,
        expected_digest=digest,
        expected_role="last",
        resolved_training=training,
        expected_joint_model_contract_sha256=joint_digest,
        expected_best_sha256=None,
        expected_best_filename=None,
    )

    tampered_digest = write_checkpoint(checkpoint, include_binding=False)
    with pytest.raises(worker.WorkerError, match="training binding is invalid"):
        worker._verify_capacity_checkpoint_binding(
            checkpoint,
            expected_digest=tampered_digest,
            expected_role="last",
            resolved_training=training,
            expected_joint_model_contract_sha256=joint_digest,
            expected_best_sha256=None,
            expected_best_filename=None,
        )
