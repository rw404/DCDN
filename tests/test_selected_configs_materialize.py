from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from dcdn.protocol import SelectedConfigsLock
from dcdn.selected_configs_materialize import (
    CPE_OUTPUT_PATH,
    CPE_SELECTION_RECEIPT_PATH,
    DISPLAY_BY_MODEL,
    OUTPUT_PATH,
    SelectedConfigsMaterializationError,
    _publish_once,
    derive_vk_cpe_selected_configs_payload,
    derive_vk_selected_configs_payload,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _payload() -> dict[str, object]:
    models: dict[str, object] = {}
    for model_key in DISPLAY_BY_MODEL:
        lineage: dict[str, object] = {}
        for seed in (42, 43, 44):
            lineage[str(seed)] = {
                "resolved_run_config_path": (f"configs/runs/{model_key}-s{seed}.yaml"),
                "resolved_run_config_sha256": _sha(f"{model_key}-{seed}-config"),
                "validation_result_path": (
                    f"results/final_validation/{model_key}-s{seed}.json"
                ),
                "validation_result_sha256": _sha(f"{model_key}-{seed}-result"),
                "best_checkpoint_path": (f"runs/{model_key}-s{seed}/best.pt"),
                "best_checkpoint_sha256": _sha(f"{model_key}-{seed}-checkpoint"),
                "selected_validation_metric": 0.01 + seed / 100000,
            }
        models[model_key] = {
            "candidate_id": f"{model_key}_candidate",
            "resolved_config_sha256": _sha(f"{model_key}-model"),
            "selection_receipt_path": (f"results/tuning/{model_key}_selection.json"),
            "selection_receipt_sha256": _sha(f"{model_key}-selection"),
            "final_seed_lineage": lineage,
            "selected_on_validation_only": True,
            "test_data_accessed": False,
            "test_metrics_consulted": False,
        }
    return {
        "schema_version": 2,
        "artifact_role": "final_selected_configs_lock",
        "dataset_id": "vk_lsvd_up0.01_ir0.01",
        "feature_regime": "C+P",
        "parent_premetric_protocol_sha256": _sha("protocol"),
        "search_space_sha256": _sha("search"),
        "experiment_matrix_sha256": _sha("matrix"),
        "encoder_contract_sha256": _sha("encoder"),
        "capacity_resolution_sha256": _sha("capacity"),
        "required_model_keys": list(DISPLAY_BY_MODEL),
        "selected_models": models,
    }


def test_publish_once_creates_strict_immutable_lock(tmp_path) -> None:
    output = tmp_path / OUTPUT_PATH
    output.parent.mkdir(parents=True)
    payload = _payload()

    digest = _publish_once(tmp_path, OUTPUT_PATH, payload)

    loaded = SelectedConfigsLock.load(output)
    assert loaded.digest == digest
    with pytest.raises(FileExistsError):
        _publish_once(tmp_path, OUTPUT_PATH, payload)
    assert SelectedConfigsLock.load(output).digest == digest


def test_derive_rejects_any_unlocked_parent(monkeypatch, tmp_path) -> None:
    artifact = SimpleNamespace(
        payload={
            "test_unlocked": True,
            "test_path_available": True,
        },
        digest=_sha("unsafe-protocol"),
    )
    monkeypatch.setattr(
        "dcdn.selected_configs_materialize.load_verified_yaml_artifact",
        lambda _path: artifact,
    )

    with pytest.raises(
        SelectedConfigsMaterializationError,
        match="premetric protocol must stay locked",
    ):
        derive_vk_selected_configs_payload(tmp_path)


def test_output_name_is_exact() -> None:
    assert (
        PurePosixPath("configs/protocols/vk_lsvd_c_plus_p/selected_configs.lock")
        == OUTPUT_PATH
    )


def test_cpe_output_name_is_exact() -> None:
    assert (
        PurePosixPath("configs/protocols/vk_lsvd_c_plus_p_plus_e/selected_configs.lock")
        == CPE_OUTPUT_PATH
    )


def test_derive_cpe_builds_exact_three_seed_lineage(monkeypatch, tmp_path) -> None:
    protocol = SimpleNamespace(
        payload={
            "dataset_id": "vk_lsvd_up0.01_ir0.01",
            "feature_regime": "C+P+E",
            "required_model_keys": ["dcnv2_full_rank"],
            "test_unlocked": False,
            "test_path_available": False,
            "hashes": {
                "search_space_sha256": _sha("search"),
                "experiment_matrix_sha256": _sha("matrix"),
                "encoder_contract_sha256": _sha("encoder"),
                "capacity_resolution_sha256": _sha("capacity"),
            },
        },
        digest=_sha("cpe-protocol"),
    )
    receipt = {
        "dataset_id": "vk_lsvd_up0.01_ir0.01",
        "feature_regime": "C+P+E",
        "test_firewall": {
            "legacy_test_opened": False,
            "protocol_test_path_available": False,
            "protocol_test_unlocked": False,
            "test_data_accessed": False,
            "test_metrics_consulted": False,
            "selected_on_validation_only": True,
            "legacy_test_metrics": None,
            "test_metrics_in_candidate_results": None,
            "combined_history_test_event_count": 0,
        },
        "selected": {
            "candidate_id": "dcnv2_full_rank_lr_1e_3",
            "resolved_config_path": "configs/runs/cpe-s42.yaml",
            "result_path": "results/tuning/cpe-s42.json",
            "result_sha256": _sha("result-42"),
            "best_checkpoint_path": "runs/cpe-s42/best.pt",
            "best_checkpoint_sha256": _sha("best-42"),
            "validation_metric": 0.02,
        },
    }

    def fake_load_json(_root, relative, *, expected_sha256=None):
        if relative == CPE_SELECTION_RECEIPT_PATH.as_posix():
            return receipt, _sha("receipt")
        seed = 43 if "s43" in relative else 44
        digest = _sha(f"result-{seed}")
        assert expected_sha256 in {None, digest}
        return {
            "run_id": f"cpe-s{seed}",
            "selected_validation_metric": 0.02 + seed / 100000,
        }, digest

    def fake_validate(_root, **kwargs):
        seed = kwargs["seed"]
        assert kwargs["feature_regime"] == "C+P+E"
        return {
            "resolved_run_config_path": kwargs["config_path"],
            "resolved_run_config_sha256": _sha(f"config-{seed}"),
            "validation_result_path": kwargs["result_path"],
            "validation_result_sha256": kwargs["expected_result_sha256"],
            "best_checkpoint_path": (
                kwargs["expected_checkpoint_path"] or f"runs/cpe-s{seed}/best.pt"
            ),
            "best_checkpoint_sha256": (
                kwargs["expected_checkpoint_sha256"] or _sha(f"best-{seed}")
            ),
            "selected_validation_metric": kwargs["expected_metric"],
        }, _sha("cpe-model")

    monkeypatch.setattr(
        "dcdn.selected_configs_materialize.load_verified_yaml_artifact",
        lambda _path: protocol,
    )
    monkeypatch.setattr(
        "dcdn.selected_configs_materialize._load_json",
        fake_load_json,
    )
    monkeypatch.setattr(
        "dcdn.selected_configs_materialize._validate_seed_lineage",
        fake_validate,
    )

    payload = derive_vk_cpe_selected_configs_payload(tmp_path)

    assert payload["feature_regime"] == "C+P+E"
    assert payload["required_model_keys"] == ["dcnv2_full_rank"]
    selected = payload["selected_models"]["dcnv2_full_rank"]
    assert selected["candidate_id"] == "dcnv2_full_rank_lr_1e_3"
    assert selected["selection_receipt_sha256"] == _sha("receipt")
    assert set(selected["final_seed_lineage"]) == {"42", "43", "44"}
