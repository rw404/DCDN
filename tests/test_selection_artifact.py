from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

import dcdn.selection_artifact as selection
from dcdn.protocol import load_verified_yaml_artifact

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ordinary_evidence() -> selection.OrdinarySelectionEvidence:
    payload, evidence = selection.load_ordinary_selection_receipt(REPOSITORY_ROOT)
    assert payload["selected"]["candidate_id"] == evidence.candidate_id
    return evidence


def test_production_human_receipt_and_frozen_inputs_reopen_exactly(
    ordinary_evidence: selection.OrdinarySelectionEvidence,
) -> None:
    assert ordinary_evidence == selection.OrdinarySelectionEvidence(
        candidate_id="dcnv2_full_rank_lr_3e_4",
        learning_rate=0.0003,
        validation_metric=0.020908477427111632,
        completion_run_id=("vk-cp-tune-dcnv2-full-rank-l2-s42-lr3e4-v1-resume-v2"),
        completed_result_sha256=selection.SELECTED_RESULT_SHA256,
        completed_resolved_config_sha256=(selection.SELECTED_RESOLVED_CONFIG_SHA256),
        resolved_model_config_seed_invariant_sha256=(
            selection.ORDINARY_MODEL_CONFIG_SHA256
        ),
        selected_best_checkpoint_sha256=(selection.SELECTED_BEST_CHECKPOINT_SHA256),
        completion_code_commit=selection.SELECTED_COMPLETION_CODE_COMMIT,
        completion_code_diff_sha256=selection.EMPTY_DIFF_SHA256,
    )
    capacity, search = selection._verify_frozen_inputs(REPOSITORY_ROOT)
    assert search.digest == selection.SEARCH_SPACE_SHA256
    assert capacity["decision"] == {
        "ordinary_reused": False,
        "mode": "fallback_bracket",
        "selected_candidate": "capacity_rank_lower",
        "selected_low_rank": 12,
        "selection_rule": (
            "minimum_abs_log_parameter_ratio_plus_abs_log_flop_ratio_then_lower_rank"
        ),
        "optimizer_learning_rate": (
            "inherit_selected_same_unit_DCNv2_learning_rate_after_validation_only_tuning"
        ),
    }


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
    ],
)
def test_strict_json_rejects_duplicate_and_nonfinite(raw: bytes) -> None:
    with pytest.raises(selection.SelectionArtifactError):
        selection._strict_json_bytes(raw, "synthetic")


def test_strict_sidecar_and_path_firewall(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    data = b'{"ok":true}\n'
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    sidecar = tmp_path / "evidence.json.sha256"
    sidecar.write_text(f"{digest}  evidence.json\n", encoding="ascii")
    reopened, actual = selection._verified_pair_bytes(
        tmp_path,
        PurePosixPath("evidence.json"),
        expected_sha256=digest,
        field="synthetic",
    )
    assert reopened == data
    assert actual == digest

    sidecar.write_text(f"{digest} evidence.json\n", encoding="ascii")
    with pytest.raises(selection.SelectionArtifactError, match="sidecar"):
        selection._verified_pair_bytes(
            tmp_path,
            PurePosixPath("evidence.json"),
            expected_sha256=digest,
            field="synthetic",
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("x", encoding="ascii")
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(selection.SelectionArtifactError, match="symlink"):
        selection._secure_repository_file(
            tmp_path, PurePosixPath("linked/secret"), "linked"
        )
    with pytest.raises(selection.SelectionArtifactError, match="relative"):
        selection._relative_path("../escape", "escape")


def test_secure_exclusive_pair_never_follows_output_symlink(tmp_path: Path) -> None:
    digest = selection._exclusive_pair(
        tmp_path,
        PurePosixPath("artifacts/selection/auth.json"),
        {"schema_version": 1},
    )
    target = tmp_path / "artifacts/selection/auth.json"
    original = target.read_bytes()
    assert hashlib.sha256(original).hexdigest() == digest
    with pytest.raises(selection.SelectionArtifactError, match="already exists"):
        selection._exclusive_pair(
            tmp_path,
            PurePosixPath("artifacts/selection/auth.json"),
            {"schema_version": 2},
        )
    assert target.read_bytes() == original

    symlink_root = tmp_path / "symlink-root"
    symlink_root.mkdir()
    outside = tmp_path / "outside-output"
    outside.mkdir()
    (symlink_root / "artifacts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(selection.SelectionArtifactError):
        selection._exclusive_pair(
            symlink_root,
            PurePosixPath("artifacts/selection/auth.json"),
            {"schema_version": 1},
        )
    assert list(outside.iterdir()) == []


def test_yaml_payload_is_parsed_from_same_verified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "lock.yaml"
    old = b"value: old\n"
    old_digest = hashlib.sha256(old).hexdigest()
    path.write_bytes(old)
    path.with_name("lock.yaml.sha256").write_text(
        f"{old_digest}  lock.yaml\n", encoding="ascii"
    )

    def replace_after_single_open(source: Path) -> Any:
        artifact = load_verified_yaml_artifact(source)
        new = b"value: new\n"
        new_digest = hashlib.sha256(new).hexdigest()
        source.write_bytes(new)
        source.with_name("lock.yaml.sha256").write_text(
            f"{new_digest}  lock.yaml\n", encoding="ascii"
        )
        return artifact

    monkeypatch.setattr(
        selection, "load_verified_yaml_artifact", replace_after_single_open
    )
    payload = selection._load_verified_yaml_at(
        tmp_path,
        PurePosixPath("lock.yaml"),
        expected_sha256=old_digest,
        field="lock",
    )
    assert payload == {"value": "old"}
    assert path.read_bytes() == b"value: new\n"


class _FakeSearch:
    def require_phase(
        self,
        phase: str,
        *,
        resolutions: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        assert phase == "capacity_control_training"
        assert resolutions == {}
        return {
            "models": {
                "dcnv2_low_rank_mix_capacity": {
                    "base_config": {"optimizer": {"learning_rate": 0.0003}}
                }
            }
        }


def _stub_compact_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    evidence: selection.OrdinarySelectionEvidence,
    materializer_identity: Mapping[str, str],
) -> None:
    monkeypatch.setattr(
        selection,
        "load_ordinary_selection_receipt",
        lambda _root: ({}, evidence),
    )
    monkeypatch.setattr(
        selection,
        "_verify_frozen_inputs",
        lambda _root: ({}, _FakeSearch()),
    )
    monkeypatch.setattr(selection, "capacity_phase_resolutions", lambda *_a, **_k: {})

    def validate_identity(_root: Path, value: Mapping[str, Any]) -> None:
        if dict(value) != dict(materializer_identity):
            raise selection.SelectionArtifactError("materializer identity drifted")

    monkeypatch.setattr(
        selection, "_validate_materializer_code_identity", validate_identity
    )


def _mutate_leaf(payload: dict[str, Any], path: tuple[str, ...]) -> None:
    parent: dict[str, Any] = payload
    for name in path[:-1]:
        parent = parent[name]
    name = path[-1]
    value = parent[name]
    if type(value) is bool:
        parent[name] = not value
    elif type(value) is int:
        parent[name] = value + 1
    elif type(value) is float:
        parent[name] = value + 1.0
    elif isinstance(value, str):
        parent[name] = value + "-tampered"
    else:  # pragma: no cover - the compact schema has only scalar leaves
        raise AssertionError(type(value))


def test_every_compact_authorization_field_is_binding_sensitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ordinary_evidence: selection.OrdinarySelectionEvidence,
) -> None:
    identity = {
        "code_commit": "a" * 40,
        "code_diff_sha256": selection.EMPTY_DIFF_SHA256,
    }
    payload = selection._capacity_binding_payload(ordinary_evidence, identity)
    _stub_compact_dependencies(monkeypatch, ordinary_evidence, identity)
    assert (
        selection.validate_capacity_lr_binding(payload, tmp_path) == ordinary_evidence
    )
    for leaf in selection.iter_leaf_paths(payload):
        tampered = copy.deepcopy(payload)
        _mutate_leaf(tampered, leaf)
        with pytest.raises(selection.SelectionArtifactError):
            selection.validate_capacity_lr_binding(tampered, tmp_path)

    missing = copy.deepcopy(payload)
    del missing["status"]
    with pytest.raises(selection.SelectionArtifactError, match="keys mismatch"):
        selection.validate_capacity_lr_binding(missing, tmp_path)
    unknown = copy.deepcopy(payload)
    unknown["unknown"] = True
    with pytest.raises(selection.SelectionArtifactError, match="keys mismatch"):
        selection.validate_capacity_lr_binding(unknown, tmp_path)

    for path, value in (
        (("schema_version",), True),
        (("identity", "tuning_seed"), 42.0),
        (("capacity_control", "low_rank"), 12.0),
        (("test_firewall", "protocol_test_unlocked"), 0),
    ):
        tampered = copy.deepcopy(payload)
        parent = tampered
        for name in path[:-1]:
            parent = parent[name]
        parent[path[-1]] = value
        with pytest.raises(selection.SelectionArtifactError):
            selection.validate_capacity_lr_binding(tampered, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("learning_rate", 0.0003, "learning-rate"),
        ("test_metrics", {"normalized_mse": 0.0}, "test metrics"),
        ("scientific_result_eligible", False, "scientifically eligible"),
    ],
)
def test_candidate_lr_eligibility_and_test_firewall_fail_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    payload = json.loads(
        (REPOSITORY_ROOT / selection.ORDINARY_SELECTION_PATH).read_text(
            encoding="utf-8"
        )
    )
    candidate = next(item for item in payload["candidates"] if not item["selected"])
    candidate[field] = value
    with pytest.raises(selection.SelectionArtifactError, match=match):
        selection.validate_ordinary_dcnv2_selection(payload, REPOSITORY_ROOT)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("selection_rule", "candidate_budget"), 2.0),
        (("selection_rule", "tuning_seed"), 42.0),
        (("selection_rule", "independent_selection_per_tuning_unit"), 1),
        (("test_firewall", "combined_history_test_event_count"), False),
        (("test_firewall", "protocol_test_unlocked"), 0),
    ],
)
def test_ordinary_selection_scalar_types_fail_closed(
    path: tuple[str, ...], value: object
) -> None:
    payload = json.loads(
        (REPOSITORY_ROOT / selection.ORDINARY_SELECTION_PATH).read_text(
            encoding="utf-8"
        )
    )
    parent = payload
    for name in path[:-1]:
        parent = parent[name]
    parent[path[-1]] = value
    with pytest.raises(selection.SelectionArtifactError, match="type|identity"):
        selection.validate_ordinary_dcnv2_selection(payload, REPOSITORY_ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("best_checkpoint_filename", None),
        ("best_checkpoint_sha256", None),
        ("validation_checks", 0),
        ("validation_checks", 7.0),
        ("nonfinite_total", False),
    ],
)
def test_nonselected_result_scientific_eligibility_is_recomputed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    payload = json.loads(
        (REPOSITORY_ROOT / selection.ORDINARY_SELECTION_PATH).read_text(
            encoding="utf-8"
        )
    )
    candidate = next(item for item in payload["candidates"] if not item["selected"])
    original = selection._verified_json_pair

    def tampered_result(*args: Any, **kwargs: Any) -> Any:
        result, digest = original(*args, **kwargs)
        if kwargs.get("field", "").endswith(".result"):
            result = copy.deepcopy(result)
            if field == "nonfinite_total":
                result["training"]["numerical_summary"][field] = value
            else:
                result["training"][field] = value
        return result, digest

    monkeypatch.setattr(selection, "_verified_json_pair", tampered_result)
    with pytest.raises(
        selection.SelectionArtifactError, match="eligible|drifted|integer"
    ):
        selection._validate_candidate_evidence(candidate, REPOSITORY_ROOT)


@pytest.mark.parametrize("mode", ["tie", "selected_not_min"])
def test_selection_requires_strict_untied_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    payload = json.loads(
        (REPOSITORY_ROOT / selection.ORDINARY_SELECTION_PATH).read_text(
            encoding="utf-8"
        )
    )

    def fake_candidate(candidate: Mapping[str, Any], _root: Path) -> Any:
        candidate_id = candidate["candidate_id"]
        if mode == "tie":
            metric = 1.0
        elif candidate_id == selection.SELECTED_ORDINARY_CANDIDATE_ID:
            metric = 2.0
        else:
            metric = 1.0
        return metric, {}, {}, {}, "a" * 40, selection.EMPTY_DIFF_SHA256

    monkeypatch.setattr(selection, "_validate_candidate_evidence", fake_candidate)
    match = "tie" if mode == "tie" else "strict minimum"
    with pytest.raises(selection.SelectionArtifactError, match=match):
        selection.validate_ordinary_dcnv2_selection(payload, tmp_path)


def test_capacity_selected_model_digest_binds_every_post_validation_input() -> None:
    interaction = {
        "model_key": "dcnv2_capacity_matched_core",
        "num_layers": 2,
        "low_rank": 12,
        "num_experts": 4,
        "product_init_scale": 0.1,
    }
    head = {"hidden_dims": [], "dropout": 0.0, "output_dim": 1}
    binding = selection.capacity_selection_binding_payload(
        capacity_candidate_id=selection.CAPACITY_CANDIDATE_ID,
        learning_rate=selection.SELECTED_LEARNING_RATE,
        capacity_resolution_sha256=selection.CAPACITY_RESOLUTION_SHA256,
        ordinary_dcnv2_selection_sha256=selection.ORDINARY_SELECTION_SHA256,
        capacity_lr_binding_sha256="a" * 64,
        premetric_protocol_sha256=selection.PREMETRIC_PROTOCOL_SHA256,
    )
    baseline = selection.capacity_resolved_model_config_sha256(
        interaction_config=interaction,
        head_config=head,
        capacity_selection_binding=binding,
    )
    mutations = [
        ({**interaction, "low_rank": 13}, binding),
        (interaction, {**binding, "capacity_candidate_id": "capacity_rank_upper"}),
        (interaction, {**binding, "learning_rate": 0.001}),
        (interaction, {**binding, "capacity_resolution_sha256": "b" * 64}),
        (
            interaction,
            {**binding, "ordinary_dcnv2_selection_sha256": "c" * 64},
        ),
        (interaction, {**binding, "capacity_lr_binding_sha256": "d" * 64}),
        (interaction, {**binding, "premetric_protocol_sha256": "e" * 64}),
    ]
    for changed_interaction, changed_binding in mutations:
        assert (
            selection.capacity_resolved_model_config_sha256(
                interaction_config=changed_interaction,
                head_config=head,
                capacity_selection_binding=changed_binding,
            )
            != baseline
        )


def test_untracked_importable_materializer_source_is_rejected(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/untracked.py").write_text("x = 1\n", encoding="ascii")
    with pytest.raises(selection.SelectionArtifactError, match="untracked"):
        selection._reject_untracked_code(tmp_path)
    os.rename(tmp_path / "src/untracked.py", tmp_path / "src/notes.txt")
    selection._reject_untracked_code(tmp_path)
