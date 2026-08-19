from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

import dcdn.protocol as protocol_module
from dcdn.protocol import (
    BASE_HASH_KEYS,
    RUNTIME_HASH_FIELDS,
    ActiveProtocolRegistry,
    ProtocolError,
    ProtocolGate,
    RuntimeIdentity,
    SelectedConfigsLock,
    canonical_split_json_bytes,
    canonical_split_sha256,
    load_verified_yaml,
    load_verified_yaml_artifact,
    read_verified_artifact,
    sha256_file,
    split_sha256_from_views_payload,
    validate_protocol,
    verify_sha256_sidecar,
)
from dcdn.protocol import (
    TestClaimRegistry as ClaimRegistry,
)

SPLIT_MAPPING = {
    "train_weeks": [20, 21],
    "validation": {"week": 25, "filters": [None, True, "public"]},
    "sealed_test_week": 26,
}
TRAIN_ROWS = 101
VALIDATION_ROWS = 23
TEST_ROWS = 31
REQUIRED_MODELS = ["dcnv2_full_rank", "dcdn_core"]


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def base_hashes() -> dict[str, str]:
    result = {name: digest(name) for name in BASE_HASH_KEYS if name != "split_sha256"}
    result["split_sha256"] = canonical_split_sha256(SPLIT_MAPPING)
    return result


def protocol_payload(
    *,
    unlocked: bool = False,
    parent_digest: str | None = None,
    selected_configs_sha256: str | None = None,
) -> dict[str, Any]:
    hashes = base_hashes()
    if unlocked:
        hashes["selected_configs_sha256"] = selected_configs_sha256 or digest(
            "selected-configs"
        )
    return {
        "schema_version": 3,
        "dataset_id": "synthetic-no-test-data",
        "feature_regime": "C+P",
        "required_model_keys": list(REQUIRED_MODELS),
        "parent_premetric_protocol_sha256": (parent_digest if unlocked else None),
        "test_unlocked": unlocked,
        "test_path_available": unlocked,
        "hashes": hashes,
        "split_row_counts": {
            "train": TRAIN_ROWS,
            "validation": VALIDATION_ROWS,
            "test": TEST_ROWS if unlocked else None,
        },
        "final_seeds": [42, 43, 44],
        "tuning_seed": 42,
        "tuning_splits": ["train", "validation"],
        "test_evaluations_per_final_seed": 1,
    }


def write_hashed(path: Path, payload: dict[str, Any] | str) -> str:
    text = (
        payload
        if isinstance(payload, str)
        else yaml.safe_dump(payload, sort_keys=False)
    )
    path.write_text(text, encoding="utf-8")
    artifact_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{artifact_digest}  {path.name}\n",
        encoding="utf-8",
    )
    return artifact_digest


def selected_payload(parent_gate: ProtocolGate) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_role": "final_selected_configs_lock",
        "dataset_id": parent_gate.dataset_id,
        "feature_regime": parent_gate.feature_regime,
        "parent_premetric_protocol_sha256": parent_gate.digest,
        "search_space_sha256": parent_gate.payload["hashes"]["search_space_sha256"],
        "experiment_matrix_sha256": parent_gate.payload["hashes"][
            "experiment_matrix_sha256"
        ],
        "encoder_contract_sha256": parent_gate.payload["hashes"][
            "encoder_contract_sha256"
        ],
        "capacity_resolution_sha256": parent_gate.payload["hashes"][
            "capacity_resolution_sha256"
        ],
        "required_model_keys": list(parent_gate.required_model_keys),
        "selected_models": {
            model_key: {
                "candidate_id": f"{model_key}-candidate-7",
                "resolved_config_sha256": digest(f"{model_key}-resolved-config"),
                "selected_on_validation_only": True,
                "test_data_accessed": False,
                "test_metrics_consulted": False,
            }
            for model_key in parent_gate.required_model_keys
        },
    }


def runtime_identity(
    payload: dict[str, Any] | Any,
    split: str,
) -> RuntimeIdentity:
    hashes = payload["hashes"]
    return RuntimeIdentity(
        dataset_id=payload["dataset_id"],
        split=split,
        feature_regime=payload["feature_regime"],
        row_count=payload["split_row_counts"][split],
        data_manifest_sha256=hashes["data_manifest_sha256"],
        views_manifest_sha256=hashes["views_manifest_sha256"],
        features_sha256=hashes["features_sha256"],
        physical_schema_sha256=hashes["physical_schema_sha256"],
        expanded_schema_sha256=hashes["expanded_schema_sha256"],
        operand_schema_sha256=hashes["operand_schema_sha256"],
        denominator_mask_sha256=hashes["denominator_mask_sha256"],
        full_materialized_scan_sha256=hashes["full_materialized_scan_sha256"],
        cache_content_sha256=hashes["cache_content_sha256"],
        encoder_contract_sha256=hashes["encoder_contract_sha256"],
        capacity_resolution_sha256=hashes["capacity_resolution_sha256"],
    )


@dataclass(frozen=True)
class PreparedProtocol:
    root: Path
    active_path: Path
    claims_path: Path
    parent_path: Path
    parent_gate: ProtocolGate
    selected_path: Path
    selected_payload: dict[str, Any]
    final_path: Path
    final_gate: ProtocolGate


def prepare_final(
    root: Path,
    *,
    activate_parent: bool = True,
    activate_final: bool = False,
) -> PreparedProtocol:
    root.mkdir(parents=True, exist_ok=True)
    active_path = root / "active_protocol.yaml"
    claims_path = root / "test_claim_registry.yaml"
    parent_path = root / "premetric_protocol.yaml"
    write_hashed(parent_path, protocol_payload())
    parent_gate = ProtocolGate.load(
        parent_path,
        active_registry_path=active_path,
    )
    if activate_parent:
        parent_gate.activate()

    selected_path = root / "selected_configs.lock"
    selected = selected_payload(parent_gate)
    selected_digest = write_hashed(selected_path, selected)
    final = protocol_payload(
        unlocked=True,
        parent_digest=parent_gate.digest,
        selected_configs_sha256=selected_digest,
    )
    final_path = root / "final_protocol.yaml"
    write_hashed(final_path, final)
    final_gate = ProtocolGate.load(
        final_path,
        parent_premetric_protocol_path=parent_path,
        active_registry_path=active_path,
    )
    if activate_final:
        final_gate.activate(
            selected_configs_path=selected_path,
            test_claim_registry_path=claims_path,
        )
    return PreparedProtocol(
        root=root,
        active_path=active_path,
        claims_path=claims_path,
        parent_path=parent_path,
        parent_gate=parent_gate,
        selected_path=selected_path,
        selected_payload=selected,
        final_path=final_path,
        final_gate=final_gate,
    )


def model_selection(
    prepared: PreparedProtocol,
    model_key: str = REQUIRED_MODELS[0],
) -> tuple[str, str, str]:
    selected = prepared.selected_payload["selected_models"][model_key]
    return (
        model_key,
        selected["candidate_id"],
        selected["resolved_config_sha256"],
    )


def claim_test(
    prepared: PreparedProtocol,
    *,
    model_key: str = REQUIRED_MODELS[0],
    seed: int = 42,
):
    model_key, candidate_id, resolved_config = model_selection(
        prepared,
        model_key,
    )
    return prepared.final_gate.require_split(
        "test",
        runtime_identity=runtime_identity(prepared.final_gate.payload, "test"),
        selected_configs_path=prepared.selected_path,
        model_key=model_key,
        candidate_id=candidate_id,
        resolved_config_sha256=resolved_config,
        seed=seed,
        checkpoint_protocol_sha256=prepared.parent_gate.digest,
    )


def test_verified_artifact_round_trip_tamper_and_exact_same_parse_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact.yaml"
    artifact_digest = write_hashed(path, {"ok": True})
    artifact = read_verified_artifact(path)
    assert artifact.data == b"ok: true\n"
    assert artifact.digest == artifact_digest
    assert sha256_file(path) == artifact_digest
    assert verify_sha256_sidecar(path) == artifact_digest
    assert load_verified_yaml(path) == {"ok": True}

    original_parser = protocol_module._parse_unique_yaml

    def mutate_after_verified_read(data: bytes, source: Path):
        source.write_text("ok: false\n", encoding="utf-8")
        return original_parser(data, source)

    monkeypatch.setattr(
        protocol_module,
        "_parse_unique_yaml",
        mutate_after_verified_read,
    )
    same_bytes = load_verified_yaml_artifact(path)
    assert same_bytes.digest == artifact_digest
    assert same_bytes.payload == {"ok": True}
    assert path.read_text(encoding="utf-8") == "ok: false\n"

    with pytest.raises(ProtocolError, match="SHA-256 mismatch"):
        verify_sha256_sidecar(path)


@pytest.mark.parametrize(
    "sidecar_builder",
    [
        lambda value, name: f"{value} {name}\n",
        lambda value, name: f"{value}   {name}\n",
        lambda value, name: f"{value}  {name}",
        lambda value, name: f"{value}  ./{name}\n",
        lambda value, name: f"{value}  ../{name}\n",
        lambda value, name: f"{value}  {name}\n\n",
        lambda value, name: f"{value.upper()}  {name}\n",
    ],
)
def test_sidecar_grammar_is_exact(
    tmp_path: Path,
    sidecar_builder,
) -> None:
    path = tmp_path / "artifact.yaml"
    artifact_digest = write_hashed(path, {"ok": True})
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar.write_text(
        sidecar_builder(artifact_digest, path.name),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError):
        verify_sha256_sidecar(path)


def test_source_sidecar_and_parent_symlinks_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.yaml"
    write_hashed(target, {"safe": True})
    linked = tmp_path / "linked.yaml"
    linked.symlink_to(target)
    with pytest.raises(ProtocolError, match="cannot safely open artifact"):
        verify_sha256_sidecar(linked)

    regular = tmp_path / "regular.yaml"
    write_hashed(regular, {"safe": True})
    regular_sidecar = regular.with_name(f"{regular.name}.sha256")
    real_sidecar = tmp_path / "real-sidecar"
    regular_sidecar.replace(real_sidecar)
    regular_sidecar.symlink_to(real_sidecar)
    with pytest.raises(ProtocolError, match="cannot safely open SHA-256 sidecar"):
        verify_sha256_sidecar(regular)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    nested = real_parent / "nested.yaml"
    write_hashed(nested, {"safe": True})
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ProtocolError, match="unsafe symlink"):
        verify_sha256_sidecar(linked_parent / nested.name)


@pytest.mark.parametrize(
    "yaml_text",
    [
        "schema_version: 3\nschema_version: 3\n",
        "hashes:\n  data: first\n  data: second\n",
        "outer:\n  nested:\n    key: first\n    key: second\n",
    ],
)
def test_duplicate_yaml_keys_are_rejected(
    tmp_path: Path,
    yaml_text: str,
) -> None:
    path = tmp_path / "duplicate.yaml"
    write_hashed(path, yaml_text)
    with pytest.raises(ProtocolError, match="duplicate-key YAML"):
        load_verified_yaml(path)


def test_canonical_split_hash_has_exact_recursive_compact_json_derivation() -> None:
    expected = (
        b'{"sealed_test_week":26,"train_weeks":[20,21],'
        b'"validation":{"filters":[null,true,"public"],"week":25}}'
    )
    assert canonical_split_json_bytes(SPLIT_MAPPING) == expected
    assert canonical_split_sha256(SPLIT_MAPPING) == hashlib.sha256(expected).hexdigest()
    reordered = {
        "validation": {"filters": [None, True, "public"], "week": 25},
        "sealed_test_week": 26,
        "train_weeks": [20, 21],
    }
    assert canonical_split_sha256(reordered) == canonical_split_sha256(SPLIT_MAPPING)
    mutated = deepcopy(SPLIT_MAPPING)
    mutated["validation"]["week"] = 24
    assert canonical_split_sha256(mutated) != canonical_split_sha256(SPLIT_MAPPING)
    assert split_sha256_from_views_payload({"split": SPLIT_MAPPING}) == (
        canonical_split_sha256(SPLIT_MAPPING)
    )


@pytest.mark.parametrize(
    "invalid",
    [
        {"nested": {1: "not-a-string-key"}},
        {"nested": (1, 2)},
        {"nested": {1, 2}},
        {"nested": b"bytes"},
        {"nested": float("nan")},
        {"nested": float("inf")},
        {"nested": -float("inf")},
    ],
)
def test_canonical_split_rejects_non_json_recursive_types(invalid: dict) -> None:
    with pytest.raises(ProtocolError):
        canonical_split_sha256(invalid)


def test_views_manifest_requires_the_singular_split_mapping() -> None:
    with pytest.raises(ProtocolError, match="contain a split mapping"):
        split_sha256_from_views_payload({"splits": SPLIT_MAPPING})
    with pytest.raises(ProtocolError, match="root must be a mapping"):
        split_sha256_from_views_payload([])  # type: ignore[arg-type]


def test_exact_v3_schema_and_hash_key_sets() -> None:
    locked = protocol_payload()
    validate_protocol(locked)
    assert set(locked["hashes"]) == BASE_HASH_KEYS

    without_parent_key = deepcopy(locked)
    without_parent_key.pop("parent_premetric_protocol_sha256")
    validate_protocol(without_parent_key)

    without_test_count = deepcopy(locked)
    without_test_count["split_row_counts"].pop("test")
    validate_protocol(without_test_count)

    final = protocol_payload(
        unlocked=True,
        parent_digest=digest("parent"),
        selected_configs_sha256=digest("selected"),
    )
    validate_protocol(final)
    assert set(final["hashes"]) == BASE_HASH_KEYS | {"selected_configs_sha256"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 3.0),
        ("dataset_id", ""),
        ("feature_regime", "UNRESOLVED"),
        ("required_model_keys", []),
        ("required_model_keys", ["dcdn_core", "dcdn_core"]),
        ("test_unlocked", "false"),
        ("test_path_available", True),
        ("final_seeds", [42.0, 43, 44]),
        ("tuning_seed", 42.0),
        ("tuning_splits", ["train", "validation", "test"]),
        ("test_evaluations_per_final_seed", True),
    ],
)
def test_protocol_root_values_are_strict(field: str, value: object) -> None:
    candidate = protocol_payload()
    candidate[field] = value
    with pytest.raises(ProtocolError):
        validate_protocol(candidate)


def test_protocol_rejects_unknown_missing_and_state_wrong_keys() -> None:
    valid = protocol_payload()

    candidate = deepcopy(valid)
    candidate["unknown_root"] = True
    with pytest.raises(ProtocolError, match="protocol root keys must be exact"):
        validate_protocol(candidate)

    candidate = deepcopy(valid)
    candidate.pop("feature_regime")
    with pytest.raises(ProtocolError, match="protocol root keys must be exact"):
        validate_protocol(candidate)

    candidate = deepcopy(valid)
    candidate["hashes"].pop("views_manifest_sha256")
    with pytest.raises(ProtocolError, match="hashes keys must be exact"):
        validate_protocol(candidate)

    candidate = deepcopy(valid)
    candidate["hashes"]["unexpected_sha256"] = digest("unexpected")
    with pytest.raises(ProtocolError, match="hashes keys must be exact"):
        validate_protocol(candidate)

    candidate = deepcopy(valid)
    candidate["hashes"]["selected_configs_sha256"] = digest("selected")
    with pytest.raises(ProtocolError, match="hashes keys must be exact"):
        validate_protocol(candidate)

    final = protocol_payload(
        unlocked=True,
        parent_digest=digest("parent"),
        selected_configs_sha256=digest("selected"),
    )
    final["hashes"].pop("selected_configs_sha256")
    with pytest.raises(ProtocolError, match="hashes keys must be exact"):
        validate_protocol(final)


def test_protocol_rejects_placeholder_hashes_and_wrong_parent_state() -> None:
    candidate = protocol_payload()
    candidate["hashes"]["features_sha256"] = "UNRESOLVED"
    with pytest.raises(ProtocolError, match="features_sha256"):
        validate_protocol(candidate)

    candidate = protocol_payload()
    candidate["hashes"]["features_sha256"] = "a" * 64
    with pytest.raises(ProtocolError, match="repeated-character"):
        validate_protocol(candidate)

    candidate = protocol_payload()
    candidate["parent_premetric_protocol_sha256"] = digest("unexpected-parent")
    with pytest.raises(ProtocolError, match="null or absent"):
        validate_protocol(candidate)

    final = protocol_payload(
        unlocked=True,
        parent_digest=digest("parent"),
        selected_configs_sha256=digest("selected"),
    )
    final["parent_premetric_protocol_sha256"] = None
    with pytest.raises(ProtocolError, match="parent_premetric"):
        validate_protocol(final)


@pytest.mark.parametrize("field", ["train", "validation"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10"])
def test_train_validation_row_counts_must_be_positive_ints(
    field: str,
    value: object,
) -> None:
    candidate = protocol_payload()
    candidate["split_row_counts"][field] = value
    with pytest.raises(ProtocolError, match=field):
        validate_protocol(candidate)


def test_test_row_count_is_sealed_until_final() -> None:
    locked = protocol_payload()
    locked["split_row_counts"]["test"] = TEST_ROWS
    with pytest.raises(ProtocolError, match="null or absent"):
        validate_protocol(locked)

    final = protocol_payload(
        unlocked=True,
        parent_digest=digest("parent"),
        selected_configs_sha256=digest("selected"),
    )
    final["split_row_counts"]["test"] = None
    with pytest.raises(ProtocolError, match="positive integer"):
        validate_protocol(final)

    final = protocol_payload(
        unlocked=True,
        parent_digest=digest("parent"),
        selected_configs_sha256=digest("selected"),
    )
    final["split_row_counts"]["unknown"] = 1
    with pytest.raises(ProtocolError, match="unknown split_row_counts"):
        validate_protocol(final)


def test_final_protocol_requires_verified_exact_parent_lineage(
    tmp_path: Path,
) -> None:
    prepared = prepare_final(tmp_path)

    with pytest.raises(ProtocolError, match="requires verified parent"):
        ProtocolGate.load(
            prepared.final_path,
            active_registry_path=prepared.active_path,
        )

    wrong_parent_path = tmp_path / "wrong_parent.yaml"
    wrong_parent = protocol_payload()
    wrong_parent["dataset_id"] = "different-dataset"
    write_hashed(wrong_parent_path, wrong_parent)
    with pytest.raises(ProtocolError, match="parent digest"):
        ProtocolGate.load(
            prepared.final_path,
            parent_premetric_protocol_path=wrong_parent_path,
            active_registry_path=prepared.active_path,
        )

    with pytest.raises(ProtocolError, match="must not provide a parent"):
        ProtocolGate.load(
            prepared.parent_path,
            parent_premetric_protocol_path=prepared.parent_path,
            active_registry_path=prepared.active_path,
        )


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        ("root", "dataset_id"),
        ("root", "feature_regime"),
        ("root", "required_model_keys"),
        ("hash", "features_sha256"),
        ("hash", "denominator_mask_sha256"),
        ("count", "train"),
        ("count", "validation"),
    ],
)
def test_final_protocol_rejects_immutable_parent_drift(
    tmp_path: Path,
    kind: str,
    name: str,
) -> None:
    prepared = prepare_final(tmp_path)
    final = protocol_payload(
        unlocked=True,
        parent_digest=prepared.parent_gate.digest,
        selected_configs_sha256=prepared.final_gate.payload["hashes"][
            "selected_configs_sha256"
        ],
    )
    if kind == "root":
        final[name] = (
            ["dcdn_core", "dcnv2_full_rank"]
            if name == "required_model_keys"
            else f"changed-{name}"
        )
    elif kind == "hash":
        final["hashes"][name] = digest(f"changed-{name}")
    else:
        final["split_row_counts"][name] += 1
    drift_path = tmp_path / "drift_final_protocol.yaml"
    write_hashed(drift_path, final)
    with pytest.raises(ProtocolError, match="drift"):
        ProtocolGate.load(
            drift_path,
            parent_premetric_protocol_path=prepared.parent_path,
            active_registry_path=prepared.active_path,
        )


def test_complete_runtime_identity_is_required_and_compared(
    tmp_path: Path,
) -> None:
    prepared = prepare_final(tmp_path)
    gate = prepared.parent_gate

    with pytest.raises(ProtocolError, match="RuntimeIdentity is required"):
        gate.require_split("train")

    identity = runtime_identity(gate.payload, "train")
    gate.require_split("train", runtime_identity=identity)

    scalar_mutations = {
        "dataset_id": "wrong-dataset",
        "split": "validation",
        "feature_regime": "wrong-regime",
        "row_count": identity.row_count + 1,
    }
    for field, value in scalar_mutations.items():
        with pytest.raises(ProtocolError, match="runtime identity"):
            gate.require_split(
                "train",
                runtime_identity=replace(identity, **{field: value}),
            )

    for field in RUNTIME_HASH_FIELDS:
        with pytest.raises(ProtocolError, match=field):
            gate.require_split(
                "train",
                runtime_identity=replace(
                    identity,
                    **{field: digest(f"wrong-{field}")},
                ),
            )

    with pytest.raises(ProtocolError, match="unavailable for sealed split"):
        gate.expected_row_count("test")


def test_runtime_views_split_binding_is_independent(tmp_path: Path) -> None:
    prepared = prepare_final(tmp_path)
    prepared.parent_gate.require_views_split({"split": SPLIT_MAPPING})

    changed = deepcopy(SPLIT_MAPPING)
    changed["validation"]["week"] = 24
    with pytest.raises(ProtocolError, match="split mapping digest"):
        prepared.parent_gate.require_views_split({"split": changed})


def test_selected_configs_lock_exact_schema_and_binding(tmp_path: Path) -> None:
    prepared = prepare_final(tmp_path)
    selected = SelectedConfigsLock.load(prepared.selected_path)
    selected.require_for_protocol(prepared.final_gate)

    invalid_mutations: list[dict[str, Any]] = []
    unknown = deepcopy(prepared.selected_payload)
    unknown["unknown"] = True
    invalid_mutations.append(unknown)

    missing_model = deepcopy(prepared.selected_payload)
    missing_model["selected_models"].pop(REQUIRED_MODELS[1])
    invalid_mutations.append(missing_model)

    extra_model = deepcopy(prepared.selected_payload)
    extra_model["selected_models"]["extra"] = deepcopy(
        extra_model["selected_models"][REQUIRED_MODELS[0]]
    )
    invalid_mutations.append(extra_model)

    wrong_flag = deepcopy(prepared.selected_payload)
    wrong_flag["selected_models"][REQUIRED_MODELS[0]]["selected_on_validation_only"] = (
        False
    )
    invalid_mutations.append(wrong_flag)

    test_accessed = deepcopy(prepared.selected_payload)
    test_accessed["selected_models"][REQUIRED_MODELS[0]]["test_data_accessed"] = True
    invalid_mutations.append(test_accessed)

    metrics_consulted = deepcopy(prepared.selected_payload)
    metrics_consulted["selected_models"][REQUIRED_MODELS[0]][
        "test_metrics_consulted"
    ] = True
    invalid_mutations.append(metrics_consulted)

    wrong_schema_type = deepcopy(prepared.selected_payload)
    wrong_schema_type["schema_version"] = 1.0
    invalid_mutations.append(wrong_schema_type)

    for mutation in invalid_mutations:
        write_hashed(prepared.selected_path, mutation)
        with pytest.raises(ProtocolError):
            SelectedConfigsLock.load(prepared.selected_path)

    binding_mutations: list[dict[str, Any]] = []
    wrong_dataset = deepcopy(prepared.selected_payload)
    wrong_dataset["dataset_id"] = "wrong-dataset"
    binding_mutations.append(wrong_dataset)

    wrong_search = deepcopy(prepared.selected_payload)
    wrong_search["search_space_sha256"] = digest("wrong-search")
    binding_mutations.append(wrong_search)

    wrong_order = deepcopy(prepared.selected_payload)
    wrong_order["required_model_keys"].reverse()
    binding_mutations.append(wrong_order)

    for mutation in binding_mutations:
        write_hashed(prepared.selected_path, mutation)
        parsed = SelectedConfigsLock.load(prepared.selected_path)
        with pytest.raises(ProtocolError, match="mismatch"):
            parsed.require_for_protocol(prepared.final_gate)

    write_hashed(prepared.selected_path, prepared.selected_payload)
    selected = SelectedConfigsLock.load(prepared.selected_path)
    model_key, candidate_id, config_digest = model_selection(prepared)
    selected.require_model(
        model_key=model_key,
        candidate_id=candidate_id,
        resolved_config_sha256=config_digest,
    )
    with pytest.raises(ProtocolError, match="candidate_id"):
        selected.require_model(
            model_key=model_key,
            candidate_id="wrong-candidate",
            resolved_config_sha256=config_digest,
        )
    with pytest.raises(ProtocolError, match="resolved_config"):
        selected.require_model(
            model_key=model_key,
            candidate_id=candidate_id,
            resolved_config_sha256=digest("wrong-config"),
        )


def test_selected_configs_lock_is_not_arbitrary_hashed_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selected_configs.lock"
    write_hashed(path, "not: [valid\n")
    with pytest.raises(ProtocolError, match="invalid or duplicate-key YAML"):
        SelectedConfigsLock.load(path)


def test_active_registry_is_required_for_every_split_and_persistent(
    tmp_path: Path,
) -> None:
    prepared = prepare_final(tmp_path, activate_parent=False)
    identity = runtime_identity(prepared.parent_gate.payload, "train")

    with pytest.raises(ProtocolError):
        prepared.parent_gate.require_split(
            "train",
            runtime_identity=identity,
        )

    state = prepared.parent_gate.activate()
    assert state.payload["active_protocol_sha256"] == prepared.parent_gate.digest
    assert verify_sha256_sidecar(prepared.active_path) == state.digest
    assert ActiveProtocolRegistry(prepared.active_path).load().digest == state.digest

    prepared.parent_gate.require_split("train", runtime_identity=identity)
    prepared.parent_gate.require_split(
        "validation",
        runtime_identity=runtime_identity(
            prepared.parent_gate.payload,
            "validation",
        ),
    )
    with pytest.raises(ProtocolError, match="test access is locked"):
        prepared.parent_gate.require_split(
            "test",
            runtime_identity=identity,
        )


def test_final_activation_invalidates_parent_and_forbids_rollback(
    tmp_path: Path,
) -> None:
    prepared = prepare_final(tmp_path)
    final_state = prepared.final_gate.activate(
        selected_configs_path=prepared.selected_path,
        test_claim_registry_path=prepared.claims_path,
    )
    assert final_state.payload["active_protocol_sha256"] == prepared.final_gate.digest
    assert final_state.payload["parent_premetric_protocol_sha256"] == (
        prepared.parent_gate.digest
    )

    with pytest.raises(ProtocolError, match="not active"):
        prepared.parent_gate.require_split(
            "train",
            runtime_identity=runtime_identity(
                prepared.parent_gate.payload,
                "train",
            ),
        )
    with pytest.raises(ProtocolError, match="requires the active premetric"):
        prepared.final_gate.require_split(
            "validation",
            runtime_identity=runtime_identity(
                prepared.final_gate.payload,
                "validation",
            ),
        )
    with pytest.raises(ProtocolError, match="rollback"):
        prepared.parent_gate.activate()

    repeated = prepared.final_gate.activate(
        selected_configs_path=prepared.selected_path,
        test_claim_registry_path=prepared.claims_path,
    )
    assert repeated.digest == final_state.digest


def test_final_cannot_activate_without_its_active_parent(tmp_path: Path) -> None:
    prepared = prepare_final(tmp_path / "source")
    fresh_active = tmp_path / "fresh" / "active_protocol.yaml"
    fresh_active.parent.mkdir()
    final_gate = ProtocolGate.load(
        prepared.final_path,
        parent_premetric_protocol_path=prepared.parent_path,
        active_registry_path=fresh_active,
    )
    with pytest.raises(ProtocolError, match="requires its premetric parent"):
        final_gate.activate(
            selected_configs_path=prepared.selected_path,
            test_claim_registry_path=tmp_path / "fresh" / "test_claim_registry.yaml",
        )


def test_active_registry_tamper_fails_closed(tmp_path: Path) -> None:
    prepared = prepare_final(tmp_path)
    prepared.active_path.write_text("tampered: true\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="SHA-256 mismatch"):
        prepared.parent_gate.require_split(
            "train",
            runtime_identity=runtime_identity(
                prepared.parent_gate.payload,
                "train",
            ),
        )


def test_final_activation_revalidates_selected_lock(tmp_path: Path) -> None:
    prepared = prepare_final(tmp_path)
    mutation = deepcopy(prepared.selected_payload)
    mutation["dataset_id"] = "wrong-dataset"
    write_hashed(prepared.selected_path, mutation)
    with pytest.raises(ProtocolError, match="binding mismatch"):
        prepared.final_gate.activate(
            selected_configs_path=prepared.selected_path,
            test_claim_registry_path=prepared.claims_path,
        )


def test_test_reservation_is_durable_recoverable_and_parent_bound(
    tmp_path: Path,
) -> None:
    prepared = prepare_final(tmp_path, activate_final=True)
    reserved = claim_test(prepared)
    assert reserved is not None
    assert reserved.state == "reserved"
    assert len(reserved.claim_id) == 64
    assert reserved.registry_path == prepared.claims_path
    assert reserved.final_protocol_sha256 == prepared.final_gate.digest
    assert reserved.parent_premetric_protocol_sha256 == prepared.parent_gate.digest
    assert reserved.checkpoint_protocol_sha256 == prepared.parent_gate.digest
    assert (
        reserved.selected_configs_sha256
        == prepared.final_gate.payload["hashes"]["selected_configs_sha256"]
    )
    assert reserved.runtime_identity == runtime_identity(
        prepared.final_gate.payload,
        "test",
    )

    registry = load_verified_yaml(prepared.claims_path)
    assert registry["schema_version"] == 2
    assert len(registry["claims"]) == 1
    assert registry["claims"][0]["state"] == "reserved"
    assert registry["claims"][0]["claim_id"] == reserved.claim_id
    assert registry["claims"][0]["runtime_identity"]["row_count"] == TEST_ROWS
    assert registry["claims"][0]["result_receipt_sha256"] is None
    assert registry["claims"][0]["result_row_count"] is None
    assert verify_sha256_sidecar(prepared.claims_path)

    recovered = claim_test(prepared)
    assert recovered == reserved

    second_seed = claim_test(prepared, seed=43)
    assert second_seed is not None
    assert second_seed.state == "reserved"
    assert second_seed.claim_id != reserved.claim_id

    model_key, candidate_id, config_digest = model_selection(
        prepared,
        REQUIRED_MODELS[1],
    )
    identity = runtime_identity(prepared.final_gate.payload, "test")
    with pytest.raises(ProtocolError, match="candidate_id"):
        prepared.final_gate.require_split(
            "test",
            runtime_identity=identity,
            selected_configs_path=prepared.selected_path,
            model_key=model_key,
            candidate_id="wrong-candidate",
            resolved_config_sha256=config_digest,
            seed=42,
            checkpoint_protocol_sha256=prepared.parent_gate.digest,
        )
    with pytest.raises(ProtocolError, match="resolved_config"):
        prepared.final_gate.require_split(
            "test",
            runtime_identity=identity,
            selected_configs_path=prepared.selected_path,
            model_key=model_key,
            candidate_id=candidate_id,
            resolved_config_sha256=digest("wrong-config"),
            seed=42,
            checkpoint_protocol_sha256=prepared.parent_gate.digest,
        )
    with pytest.raises(ProtocolError, match="parent premetric"):
        prepared.final_gate.require_split(
            "test",
            runtime_identity=identity,
            selected_configs_path=prepared.selected_path,
            model_key=model_key,
            candidate_id=candidate_id,
            resolved_config_sha256=config_digest,
            seed=42,
            checkpoint_protocol_sha256=prepared.final_gate.digest,
        )
    with pytest.raises(ProtocolError, match="final test seed"):
        prepared.final_gate.require_split(
            "test",
            runtime_identity=identity,
            selected_configs_path=prepared.selected_path,
            model_key=model_key,
            candidate_id=candidate_id,
            resolved_config_sha256=config_digest,
            seed=42.0,  # type: ignore[arg-type]
            checkpoint_protocol_sha256=prepared.parent_gate.digest,
        )


def test_claim_lifecycle_open_complete_and_crash_windows(tmp_path: Path) -> None:
    prepared = prepare_final(tmp_path, activate_final=True)
    identity = runtime_identity(prepared.final_gate.payload, "test")
    reserved = claim_test(prepared)
    assert reserved is not None

    with pytest.raises(ProtocolError, match="only an opened claim"):
        reserved.mark_completed(
            gate=prepared.final_gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=identity,
            result_receipt_sha256=digest("premature-receipt"),
            result_row_count=TEST_ROWS,
        )

    pre_open_recovery = claim_test(prepared)
    assert pre_open_recovery.claim_id == reserved.claim_id
    opened = pre_open_recovery.mark_opened(
        gate=prepared.final_gate,
        selected_configs_path=prepared.selected_path,
        runtime_identity=identity,
    )
    assert opened.state == "opened"

    with pytest.raises(ProtocolError, match="automatic re-evaluation"):
        claim_test(prepared)
    with pytest.raises(ProtocolError, match="reopening is forbidden"):
        reserved.mark_opened(
            gate=prepared.final_gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=identity,
        )

    with pytest.raises(ProtocolError, match="result_receipt_sha256"):
        opened.mark_completed(
            gate=prepared.final_gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=identity,
            result_receipt_sha256="a" * 64,
            result_row_count=TEST_ROWS,
        )
    with pytest.raises(ProtocolError, match="attested test rows"):
        opened.mark_completed(
            gate=prepared.final_gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=identity,
            result_receipt_sha256=digest("result-receipt"),
            result_row_count=TEST_ROWS - 1,
        )

    receipt = digest("result-receipt")
    completed = opened.mark_completed(
        gate=prepared.final_gate,
        selected_configs_path=prepared.selected_path,
        runtime_identity=identity,
        result_receipt_sha256=receipt,
        result_row_count=TEST_ROWS,
    )
    assert completed.state == "completed"
    assert completed.result_receipt_sha256 == receipt
    assert completed.result_row_count == TEST_ROWS

    persisted = load_verified_yaml(prepared.claims_path)["claims"][0]
    assert persisted["state"] == "completed"
    assert persisted["result_receipt_sha256"] == receipt
    assert persisted["result_row_count"] == TEST_ROWS

    acknowledged = opened.mark_completed(
        gate=prepared.final_gate,
        selected_configs_path=prepared.selected_path,
        runtime_identity=identity,
        result_receipt_sha256=receipt,
        result_row_count=TEST_ROWS,
    )
    assert acknowledged == completed
    with pytest.raises(ProtocolError, match="different completion"):
        completed.mark_completed(
            gate=prepared.final_gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=identity,
            result_receipt_sha256=digest("different-receipt"),
            result_row_count=TEST_ROWS,
        )
    with pytest.raises(ProtocolError, match="automatic re-evaluation"):
        claim_test(prepared)


def test_claim_transitions_recheck_all_frozen_bindings(tmp_path: Path) -> None:
    prepared = prepare_final(tmp_path, activate_final=True)
    identity = runtime_identity(prepared.final_gate.payload, "test")
    reserved = claim_test(prepared)
    assert reserved is not None

    wrong_runtime = replace(
        identity,
        cache_content_sha256=digest("wrong-runtime-cache"),
    )
    with pytest.raises(ProtocolError, match="cache_content_sha256"):
        reserved.mark_opened(
            gate=prepared.final_gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=wrong_runtime,
        )

    wrong_claim = replace(
        reserved,
        selected_configs_sha256=digest("wrong-selected-binding"),
    )
    with pytest.raises(ProtocolError, match="selected_configs_sha256"):
        wrong_claim.mark_opened(
            gate=prepared.final_gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=identity,
        )

    mutation = deepcopy(prepared.selected_payload)
    mutation["selected_models"][REQUIRED_MODELS[0]]["candidate_id"] = "changed"
    write_hashed(prepared.selected_path, mutation)
    with pytest.raises(ProtocolError):
        reserved.mark_opened(
            gate=prepared.final_gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=identity,
        )


def test_claim_registry_cannot_be_redirected_to_a_run_directory(
    tmp_path: Path,
) -> None:
    prepared = prepare_final(tmp_path, activate_final=True)
    run_dir = tmp_path / "run-42"
    run_dir.mkdir()
    alternate = ClaimRegistry(run_dir / "test_claim_registry.yaml")
    model_key, candidate_id, config_digest = model_selection(prepared)
    with pytest.raises(ProtocolError, match="not the active pinned registry"):
        alternate.claim(
            gate=prepared.final_gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=runtime_identity(prepared.final_gate.payload, "test"),
            model_key=model_key,
            seed=42,
            checkpoint_protocol_sha256=prepared.parent_gate.digest,
            candidate_id=candidate_id,
            resolved_config_sha256=config_digest,
        )


def test_concurrent_reservation_recovers_token_and_only_one_open_wins(
    tmp_path: Path,
) -> None:
    prepared = prepare_final(tmp_path, activate_final=True)
    identity = runtime_identity(prepared.final_gate.payload, "test")

    with ThreadPoolExecutor(max_workers=2) as executor:
        reservations = list(executor.map(lambda _: claim_test(prepared), range(2)))
    assert reservations[0] is not None
    assert reservations[1] is not None
    assert {item.claim_id for item in reservations} == {reservations[0].claim_id}

    def attempt_open(reservation) -> str:
        try:
            reservation.mark_opened(
                gate=prepared.final_gate,
                selected_configs_path=prepared.selected_path,
                runtime_identity=identity,
            )
        except ProtocolError:
            return "rejected"
        return "opened"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt_open, reservations))
    assert sorted(outcomes) == ["opened", "rejected"]


def test_torn_claim_registry_fails_closed_before_open(tmp_path: Path) -> None:
    prepared = prepare_final(tmp_path, activate_final=True)
    reserved = claim_test(prepared)
    assert reserved is not None
    prepared.claims_path.with_name(f"{prepared.claims_path.name}.sha256").unlink()

    with pytest.raises(ProtocolError, match="torn artifact/sidecar pair"):
        claim_test(prepared)
    with pytest.raises(ProtocolError, match="torn artifact/sidecar pair"):
        reserved.mark_opened(
            gate=prepared.final_gate,
            selected_configs_path=prepared.selected_path,
            runtime_identity=runtime_identity(
                prepared.final_gate.payload,
                "test",
            ),
        )


def test_final_gate_authorizes_parent_bound_checkpoint_without_resave(
    tmp_path: Path,
) -> None:
    prepared = prepare_final(tmp_path, activate_final=True)
    model_key, candidate_id, config_digest = model_selection(prepared)
    identity = runtime_identity(prepared.final_gate.payload, "test")

    claim = prepared.final_gate.require_split(
        "test",
        runtime_identity=identity,
        selected_configs_path=prepared.selected_path,
        model_key=model_key,
        candidate_id=candidate_id,
        resolved_config_sha256=config_digest,
        seed=44,
        checkpoint_protocol_sha256=prepared.parent_gate.digest,
    )
    assert claim is not None
    assert claim.checkpoint_protocol_sha256 == prepared.parent_gate.digest

    with pytest.raises(ProtocolError, match="parent premetric"):
        prepared.final_gate.require_split(
            "test",
            runtime_identity=identity,
            selected_configs_path=prepared.selected_path,
            model_key=REQUIRED_MODELS[1],
            candidate_id=prepared.selected_payload["selected_models"][
                REQUIRED_MODELS[1]
            ]["candidate_id"],
            resolved_config_sha256=prepared.selected_payload["selected_models"][
                REQUIRED_MODELS[1]
            ]["resolved_config_sha256"],
            seed=44,
            checkpoint_protocol_sha256=prepared.final_gate.digest,
        )


def test_repository_vk_c_plus_p_plus_e_protocol_is_isolated_and_test_locked() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    protocol_path = (
        repository_root
        / "configs/protocols/vk_lsvd_c_plus_p_plus_e/protocol_frozen.yaml"
    )
    active_path = (
        repository_root
        / "artifacts/protocol_state/vk_lsvd_c_plus_p_plus_e/active_protocol.yaml"
    )

    verify_sha256_sidecar(protocol_path)
    verify_sha256_sidecar(active_path)
    gate = ProtocolGate.load(protocol_path, active_registry_path=active_path)
    gate.require_active()

    assert gate.digest == (
        "6a6e33c908d666c4374ac5052a207c38a9fe012dac147a5e572cb819013a5739"
    )
    assert gate.dataset_id == "vk_lsvd_up0.01_ir0.01"
    assert gate.feature_regime == "C+P+E"
    assert gate.required_model_keys == ("dcnv2_full_rank",)
    assert gate.payload["test_unlocked"] is False
    assert gate.payload["test_path_available"] is False
