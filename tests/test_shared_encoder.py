from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from dcdn.encoder import (
    DatasetEncoderSpec,
    EncoderContractError,
    EncoderContractLock,
    SharedFeatureEncoder,
    TrainableEncodedModel,
    build_trainable_encoded_model,
    canonical_encoder_metadata_json,
    resolve_dataset_encoder_spec,
)
from dcdn.models import (
    AFNPlusConfig,
    DCDNCoreConfig,
    DCNv2FullRankConfig,
    HeadConfig,
    MLPConfig,
)
from dcdn.raw import RawFeatureBatch, RawSourceIdentity
from dcdn.training import count_trainable_non_embedding_parameters

ROOT = Path(__file__).resolve().parents[1]
ENCODER_LOCK = ROOT / "configs" / "encoder_v1.yaml"
DATASET_ID = "vk_lsvd_up0.01_ir0.01"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _lock() -> EncoderContractLock:
    return EncoderContractLock.load(ENCODER_LOCK)


def _identity(
    regime: str = "C+P",
    **overrides: object,
) -> RawSourceIdentity:
    lock = _lock()
    dataset = lock.payload["datasets"][DATASET_ID]
    resolved = dataset["regimes"][regime]
    values: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "split": "train",
        "feature_regime": regime,
        "row_count": 8,
        "data_manifest_sha256": dataset["source_manifest_sha256"],
        "views_manifest_sha256": dataset["views_manifest_sha256"],
        "feature_contract_sha256": dataset["feature_contract_sha256"],
        "physical_schema_sha256": resolved["physical_schema_sha256"],
        "expanded_schema_sha256": resolved["expanded_schema_sha256"],
        "operand_schema_sha256": resolved["operand_schema_sha256"],
        "denominator_mask_sha256": resolved["denominator_mask_sha256"],
        "full_materialized_scan_sha256": dataset["full_materialized_scan_sha256"],
        "cache_content_sha256": resolved["cache_content_sha256"],
        "ordered_context_fields": tuple(dataset["ordered_context_fields"]),
        "ordered_operand_fields": tuple(resolved["ordered_operand_fields"]),
        "categorical_cardinalities": tuple(dataset["categorical_cardinalities"]),
    }
    values.update(overrides)
    return RawSourceIdentity(**values)  # type: ignore[arg-type]


def _spec(regime: str = "C+P") -> DatasetEncoderSpec:
    return resolve_dataset_encoder_spec(
        _lock(), dataset_id=DATASET_ID, source_identity=_identity(regime)
    )


def _features(
    spec: DatasetEncoderSpec,
    *,
    batch_size: int = 4,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = torch.empty(
        batch_size, len(spec.ordered_context_fields), dtype=torch.int64
    )
    for column, cardinality in enumerate(spec.categorical_cardinalities):
        context[:, column] = torch.tensor(
            [index % min(cardinality, 3) for index in range(batch_size)]
        )
    operands = torch.arange(
        1,
        batch_size * len(spec.ordered_operand_fields) + 1,
        dtype=torch.float32,
    ).reshape(batch_size, -1)
    operands = (operands / 100.0).to(dtype=dtype)
    validity = torch.ones_like(operands, dtype=torch.bool)
    validity[0, 0] = False
    return context, operands, validity


def _raw_batch(
    spec: DatasetEncoderSpec,
    *,
    targets: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
) -> RawFeatureBatch:
    context, operands, validity = _features(spec)
    return RawFeatureBatch(
        context_tokens=context,
        operand_features=operands,
        denominator_validity=validity,
        targets=(targets if targets is not None else torch.zeros(4, 1)),
        global_row_indices=(
            indices if indices is not None else torch.arange(4, dtype=torch.int64)
        ),
    )


def _head() -> HeadConfig:
    return HeadConfig(hidden_dims=(), dropout=0.0, output_dim=1)


def _mlp() -> MLPConfig:
    return MLPConfig(hidden_dims=(12,), dropout=0.0)


def _dcdn() -> DCDNCoreConfig:
    return DCDNCoreConfig(
        num_layers=2,
        div_eps=0.01,
        product_init_scale=0.1,
        div_init_scale=0.1,
        runtime_value_checks=True,
    )


def _afn() -> AFNPlusConfig:
    return AFNPlusConfig(
        logarithmic_neurons=4,
        afn_hidden_dims=(8,),
        dnn_hidden_dims=(8,),
        log_floor=1e-4,
        exp_clip=8.0,
        dropout=0.0,
        batch_norm=False,
    )


def _tensor_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def _assert_tensor_state_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def test_lock_and_verified_vk_regime_dimensions() -> None:
    lock = _lock()
    assert (
        lock.sha256
        == ENCODER_LOCK.with_name("encoder_v1.yaml.sha256")
        .read_text(encoding="ascii")
        .split()[0]
    )

    base = _spec("C+P")
    assert base.ordered_context_fields == (
        "age",
        "gender",
        "geo",
        "place",
        "platform",
        "agent",
    )
    assert base.categorical_cardinalities == (55, 4, 82, 24, 13, 22)
    assert len(base.ordered_operand_fields) == 58
    assert base.model_spec.bypass_dim == 96
    assert base.model_spec.flat_dim == 154
    assert base.model_spec.num_fields == 64

    full = _spec("C+P+E")
    assert len(full.ordered_operand_fields) == 78
    assert full.model_spec.flat_dim == 174
    assert full.model_spec.num_fields == 84
    assert full.allowed_model_keys == ("dcnv2_full_rank",)


def test_shapes_order_raw_operands_and_validity_are_preserved() -> None:
    spec = _spec()
    encoder = SharedFeatureEncoder(spec, run_seed=42, include_afn_secondary=False)
    context, operands, validity = _features(spec)
    encoded = encoder(context, operands, validity)

    assert encoded.bypass_features.shape == (4, 96)
    assert encoded.field_embeddings.shape == (4, 64, 16)
    assert encoded.operand_features is operands
    assert encoded.denominator_validity is validity
    assert encoded.afn_dnn_embeddings is None
    categorical = encoded.field_embeddings[:, :6]
    assert torch.equal(categorical.flatten(start_dim=1), encoded.bypass_features)
    numeric = encoded.field_embeddings[:, 6:]
    expected = operands.unsqueeze(-1) * encoder.primary.numeric_vectors.weight
    assert torch.equal(numeric, expected)


def test_invalid_tokens_shapes_values_and_dtypes_fail_closed() -> None:
    spec = _spec()
    encoder = SharedFeatureEncoder(spec, run_seed=42, include_afn_secondary=False)
    context, operands, validity = _features(spec)

    bad_tokens = context.clone()
    bad_tokens[0, 1] = spec.categorical_cardinalities[1]
    with pytest.raises(ValueError, match="token column"):
        encoder(bad_tokens, operands, validity)
    with pytest.raises(TypeError, match="int64"):
        encoder(context.to(torch.int32), operands, validity)
    with pytest.raises(TypeError, match="FP16"):
        encoder(context, operands.to(torch.float16), validity)
    with pytest.raises(TypeError, match="FP32 or BF16"):
        encoder(context, operands.to(torch.float64), validity)
    with pytest.raises(TypeError, match="boolean"):
        encoder(context, operands, validity.to(torch.int64))
    with pytest.raises(ValueError, match="numeric width"):
        encoder(context, operands[:, :-1], validity[:, :-1])
    nonfinite = operands.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        encoder(context, nonfinite, validity)


def test_bfloat16_inputs_keep_fp32_parameters_and_bfloat16_views() -> None:
    spec = _spec()
    encoder = SharedFeatureEncoder(spec, run_seed=42, include_afn_secondary=False)
    context, operands, validity = _features(spec, dtype=torch.bfloat16)
    encoded = encoder(context, operands, validity)
    assert {parameter.dtype for parameter in encoder.parameters()} == {torch.float32}
    assert encoded.bypass_features.dtype == torch.bfloat16
    assert encoded.operand_features.dtype == torch.bfloat16
    assert encoded.field_embeddings.dtype == torch.bfloat16


def test_gradients_reach_reserved_tokens_and_numeric_embedding_table() -> None:
    spec = _spec()
    encoder = SharedFeatureEncoder(spec, run_seed=42, include_afn_secondary=False)
    context, operands, validity = _features(spec)
    encoded = encoder(context, operands, validity)
    loss = encoded.bypass_features.sum() + encoded.field_embeddings.sum()
    loss.backward()

    assert isinstance(encoder.primary.numeric_vectors, nn.Embedding)
    assert encoder.primary.numeric_vectors.weight.grad is not None
    assert torch.count_nonzero(encoder.primary.numeric_vectors.weight.grad) > 0
    for table in encoder.primary.categorical_embeddings:
        assert table.weight.grad is not None
        assert torch.count_nonzero(table.weight.grad[0]) > 0
        assert torch.count_nonzero(table.weight.grad[1]) > 0
    composite = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=_identity(),
        interaction_config=_mlp(),
        head_config=_head(),
        run_seed=42,
    )
    assert count_trainable_non_embedding_parameters(composite) == (
        count_trainable_non_embedding_parameters(composite.controlled_model)
    )


def test_primary_initialization_is_identical_across_model_families() -> None:
    identity = _identity()
    mlp = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=_mlp(),
        head_config=_head(),
        run_seed=42,
    )
    dcdn = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=_dcdn(),
        head_config=_head(),
        run_seed=42,
    )
    afn = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=_afn(),
        head_config=_head(),
        run_seed=42,
    )
    expected = _tensor_state(mlp.encoder.primary)
    _assert_tensor_state_equal(expected, _tensor_state(dcdn.encoder.primary))
    _assert_tensor_state_equal(expected, _tensor_state(afn.encoder.primary))

    # Reversing unrelated construction does not change the primary bytes.
    _ = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=_dcdn(),
        head_config=_head(),
        run_seed=99,
    )
    repeated = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=_mlp(),
        head_config=_head(),
        run_seed=42,
    )
    _assert_tensor_state_equal(expected, _tensor_state(repeated.encoder.primary))


def test_afn_has_genuinely_independent_secondary_view() -> None:
    spec = _spec()
    encoder = SharedFeatureEncoder(spec, run_seed=42, include_afn_secondary=True)
    assert encoder.afn_secondary is not None
    assert encoder.primary_seed != encoder.afn_secondary_seed
    primary = encoder.primary.numeric_vectors.weight
    secondary = encoder.afn_secondary.numeric_vectors.weight
    assert primary.data_ptr() != secondary.data_ptr()
    assert not torch.equal(primary, secondary)

    context, operands, validity = _features(spec)
    encoded = encoder(context, operands, validity)
    assert encoded.afn_dnn_embeddings is not None
    assert encoded.afn_dnn_embeddings.shape == encoded.field_embeddings.shape
    assert not torch.equal(encoded.afn_dnn_embeddings, encoded.field_embeddings)


def test_joint_module_checkpoint_optimizer_and_resume_identity() -> None:
    identity = _identity()
    original = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=_mlp(),
        head_config=_head(),
        run_seed=42,
    )
    raw = _raw_batch(original.encoder.spec)
    optimizer = torch.optim.AdamW(original.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = original(raw).logits.square().mean()
    loss.backward()
    optimizer.step()

    state = copy.deepcopy(original.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    assert any(name.startswith("encoder.") for name in state)
    assert any(name.startswith("controlled_model.") for name in state)

    resumed = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=_mlp(),
        head_config=_head(),
        run_seed=42,
    )
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed.load_state_dict(state, strict=True)
    resumed_optimizer.load_state_dict(optimizer_state)
    original.eval()
    resumed.eval()
    assert torch.equal(original(raw).logits, resumed(raw).logits)
    _assert_tensor_state_equal(_tensor_state(original), _tensor_state(resumed))
    assert (
        optimizer.state_dict()["param_groups"]
        == resumed_optimizer.state_dict()["param_groups"]
    )


def test_metadata_binds_schema_regime_orders_cardinalities_and_seeds() -> None:
    identity = _identity()
    model = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=_mlp(),
        head_config=_head(),
        run_seed=42,
    )
    metadata = canonical_encoder_metadata_json(model)
    for value in (
        model.encoder.spec.encoder_contract_sha256,
        model.encoder.spec.full_feature_schema_sha256,
        model.encoder.spec.operand_schema_sha256,
        model.encoder.spec.full_materialized_scan_sha256,
        model.encoder.spec.cache_content_sha256,
        "C+P",
        "primary_seed",
        "runtime_value_checks",
        "categorical_cardinalities",
        "ordered_context_fields",
        "ordered_operand_fields",
    ):
        assert str(value) in metadata

    changed_seed = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=_mlp(),
        head_config=_head(),
        run_seed=43,
    )
    with pytest.raises(RuntimeError, match="metadata"):
        changed_seed.load_state_dict(copy.deepcopy(model.state_dict()))


def test_identity_and_scan_attestation_mismatches_fail_closed() -> None:
    identity = _identity()
    with pytest.raises(EncoderContractError, match="full_materialized_scan_sha256"):
        resolve_dataset_encoder_spec(
            _lock(),
            dataset_id=DATASET_ID,
            source_identity=_identity(
                full_materialized_scan_sha256=_digest("wrong-full-scan")
            ),
        )

    swapped = list(identity.ordered_context_fields)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(EncoderContractError, match="ordered_context_fields"):
        resolve_dataset_encoder_spec(
            _lock(),
            dataset_id=DATASET_ID,
            source_identity=_identity(ordered_context_fields=tuple(swapped)),
        )
    cardinalities = list(identity.categorical_cardinalities)
    cardinalities[0] += 1
    with pytest.raises(EncoderContractError, match="categorical_cardinalities"):
        resolve_dataset_encoder_spec(
            _lock(),
            dataset_id=DATASET_ID,
            source_identity=_identity(categorical_cardinalities=tuple(cardinalities)),
        )
    with pytest.raises(EncoderContractError, match="no verified encoder"):
        resolve_dataset_encoder_spec(
            _lock(),
            dataset_id="kuairand_pure",
            source_identity=identity,
        )


def test_labels_and_audit_indices_cannot_affect_encoder_output() -> None:
    identity = _identity()
    model = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=_mlp(),
        head_config=_head(),
        run_seed=42,
    )
    clean = _raw_batch(model.encoder.spec)
    poisoned = RawFeatureBatch(
        context_tokens=clean.context_tokens,
        operand_features=clean.operand_features,
        denominator_validity=clean.denominator_validity,
        targets=torch.full((4, 1), float("nan")),
        global_row_indices=torch.tensor([-99, -98, -97, -96]),
    )
    first = model.encode_raw_batch(clean)
    second = model.encode_raw_batch(poisoned)
    assert torch.equal(first.bypass_features, second.bypass_features)
    assert torch.equal(first.operand_features, second.operand_features)
    assert torch.equal(first.field_embeddings, second.field_embeddings)


def test_full_feature_regime_is_restricted_to_the_frozen_dcnv2_arm() -> None:
    identity = _identity("C+P+E")
    with pytest.raises(EncoderContractError, match="forbidden"):
        build_trainable_encoded_model(
            encoder_contract=_lock(),
            dataset_id=DATASET_ID,
            source_identity=identity,
            interaction_config=_mlp(),
            head_config=_head(),
            run_seed=42,
        )
    model = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1),
        head_config=_head(),
        run_seed=42,
    )
    assert isinstance(model, TrainableEncodedModel)
    assert model.encoder_spec.operand_dim == 78
    assert model.model_key == "dcnv2_full_rank"


def test_dataset_spec_contract_rejects_hash_drift() -> None:
    spec = _spec()
    with pytest.raises(EncoderContractError, match="operand digest"):
        replace(
            spec,
            ordered_operand_fields=tuple(reversed(spec.ordered_operand_fields)),
        )


def _write_rehashed_lock(tmp_path: Path, payload: dict[str, object]) -> Path:
    lock_path = tmp_path / "encoder_v1.yaml"
    serialized = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    lock_path.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    lock_path.with_name("encoder_v1.yaml.sha256").write_text(
        f"{digest}  {lock_path.name}\n",
        encoding="ascii",
    )
    return lock_path


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("architecture", "categorical_reserved_tokens", "OOV"), 2),
        (("architecture", "numeric_field_view"), "precomputed_numeric_vectors"),
        (("architecture", "accepted_input_dtypes"), ["float32"]),
        (("architecture", "autocast_dtype"), "float16"),
        (("initialization", "primary_seed_domain"), "mutated-seed-domain"),
        (
            ("initialization", "afn_plus", "independent_secondary_view"),
            False,
        ),
        (
            ("runtime_identity", "spec_provenance"),
            "copy_provenance_from_input",
        ),
        (
            (
                "runtime_value_checks",
                "optimized",
                "requires_bound_cache_content_sha256",
            ),
            False,
        ),
        (
            (
                "datasets",
                DATASET_ID,
                "source_manifest_sha256",
            ),
            _digest("mutated-source"),
        ),
        (
            (
                "datasets",
                DATASET_ID,
                "views_manifest_sha256",
            ),
            _digest("mutated-views"),
        ),
        (
            (
                "datasets",
                DATASET_ID,
                "full_materialized_scan_sha256",
            ),
            _digest("mutated-scan"),
        ),
        (
            (
                "datasets",
                DATASET_ID,
                "regimes",
                "C+P",
                "cache_content_sha256",
            ),
            _digest("mutated-cache"),
        ),
        (
            (
                "datasets",
                DATASET_ID,
                "regimes",
                "C+P",
                "operand_schema_sha256",
            ),
            _digest("mutated-operands"),
        ),
        (
            (
                "datasets",
                DATASET_ID,
                "regimes",
                "C+P",
                "denominator_mask_sha256",
            ),
            _digest("mutated-mask"),
        ),
        (
            ("datasets", DATASET_ID, "ordered_context_fields"),
            ["gender", "age", "geo", "place", "platform", "agent"],
        ),
        (
            ("datasets", DATASET_ID, "categorical_cardinalities"),
            [56, 4, 82, 24, 13, 22],
        ),
    ],
)
def test_rehashed_critical_yaml_mutations_fail_closed(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: object,
) -> None:
    payload = copy.deepcopy(dict(_lock().payload))
    cursor: object = payload
    for key in path[:-1]:
        assert isinstance(cursor, dict)
        cursor = cursor[key]
    assert isinstance(cursor, dict)
    cursor[path[-1]] = replacement
    mutated = _write_rehashed_lock(tmp_path, payload)
    with pytest.raises((EncoderContractError, ValueError, TypeError)):
        EncoderContractLock.load(mutated)


def test_raw_source_identity_requires_dataset_and_content_digest() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="dataset_id"):
        replace(identity, dataset_id="")
    with pytest.raises(ValueError, match="cache_content_sha256"):
        replace(identity, cache_content_sha256="A" * 64)
    with pytest.raises(ValueError, match="placeholder"):
        replace(identity, cache_content_sha256="0" * 64)
    with pytest.raises(ValueError, match="distinct content"):
        replace(
            identity,
            cache_content_sha256=identity.full_materialized_scan_sha256,
        )
    with pytest.raises(ValueError, match="distinct content"):
        replace(identity, cache_content_sha256=identity.data_manifest_sha256)


@pytest.mark.parametrize(
    "field",
    [
        "dataset_id",
        "data_manifest_sha256",
        "views_manifest_sha256",
        "feature_contract_sha256",
        "physical_schema_sha256",
        "expanded_schema_sha256",
        "operand_schema_sha256",
        "denominator_mask_sha256",
        "full_materialized_scan_sha256",
        "cache_content_sha256",
        "ordered_context_fields",
        "ordered_operand_fields",
        "categorical_cardinalities",
    ],
)
def test_every_critical_source_identity_field_is_asserted(field: str) -> None:
    identity = _identity()
    if field == "dataset_id":
        changed: object = "wrong_vk_dataset"
    elif field == "ordered_context_fields":
        changed = tuple(reversed(identity.ordered_context_fields))
    elif field == "ordered_operand_fields":
        changed = tuple(reversed(identity.ordered_operand_fields))
    elif field == "categorical_cardinalities":
        values = list(identity.categorical_cardinalities)
        values[0] += 1
        changed = tuple(values)
    else:
        changed = _digest(f"wrong-{field}")
    mismatched = replace(identity, **{field: changed})
    with pytest.raises(EncoderContractError, match=field):
        resolve_dataset_encoder_spec(
            _lock(),
            dataset_id=DATASET_ID,
            source_identity=mismatched,
        )


def test_optimized_path_has_no_finite_any_or_python_scalar_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    model = build_trainable_encoded_model(
        encoder_contract=_lock(),
        dataset_id=DATASET_ID,
        source_identity=identity,
        interaction_config=_mlp(),
        head_config=_head(),
        run_seed=42,
        runtime_value_checks=False,
    )
    raw = _raw_batch(model.encoder.spec)

    def forbidden_sync(*args: object, **kwargs: object) -> None:
        raise AssertionError("optimized path invoked a device-sync value predicate")

    monkeypatch.setattr(torch, "isfinite", forbidden_sync)
    monkeypatch.setattr(torch.Tensor, "any", forbidden_sync)
    monkeypatch.setattr(torch.Tensor, "item", forbidden_sync)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden_sync)

    raw.validate(identity, tuple(range(4)), runtime_value_checks=False)
    encoded = model.encode_raw_batch(raw)
    assert encoded.operand_features is raw.operand_features
    assert encoded.denominator_validity is raw.denominator_validity
    metadata = canonical_encoder_metadata_json(model)
    assert '"mode":"optimized"' in metadata
    assert '"runtime_value_checks":false' in metadata


def test_encoder_lock_symlink_fails_closed(tmp_path: Path) -> None:
    original = ENCODER_LOCK.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    real_path = tmp_path / "real.yaml"
    real_path.write_bytes(original)
    linked_path = tmp_path / "linked.yaml"
    linked_path.symlink_to(real_path.name)
    linked_path.with_name("linked.yaml.sha256").write_text(
        f"{digest}  linked.yaml\n", encoding="ascii"
    )
    with pytest.raises(EncoderContractError, match="securely open encoder YAML"):
        EncoderContractLock.load(linked_path)


def test_encoder_lock_consumes_verified_bytes_after_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_bytes = ENCODER_LOCK.read_bytes()
    original_digest = hashlib.sha256(original_bytes).hexdigest()
    lock_path = tmp_path / "encoder.yaml"
    lock_path.write_bytes(original_bytes)
    sidecar_path = tmp_path / "encoder.yaml.sha256"
    sidecar_path.write_text(f"{original_digest}  encoder.yaml\n", encoding="ascii")
    replacement = tmp_path / "replacement.yaml"
    replacement.write_text("schema_version: 999\n", encoding="utf-8")
    original_safe_load = yaml.safe_load

    def replace_after_verified(data: str) -> object:
        replacement.replace(lock_path)
        return original_safe_load(data)

    monkeypatch.setattr(yaml, "safe_load", replace_after_verified)
    loaded = EncoderContractLock.load(lock_path)
    assert loaded.sha256 == original_digest
    assert loaded.payload["schema_version"] == 1
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() != original_digest
