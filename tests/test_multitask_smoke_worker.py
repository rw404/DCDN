from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

DIGEST = hashlib.sha256(b"binding").hexdigest()


def _load_worker():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "run_worker.py"
    spec = importlib.util.spec_from_file_location("multitask_smoke_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_sidecar(path: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    path.write_bytes(data)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )
    return digest


def _resolved(root: Path, *, model_key: str = "dcdn_core") -> dict[str, Any]:
    return {
        "schema_version": 5,
        "trainer_kind": "vk_lsvd_multitask_smoke_v1",
        "execution_kind": "implementation_smoke",
        "run_id": f"vk-multitask-smoke-{model_key}",
        "project_root": str(root),
        "protocol_path": "protocol.yaml",
        "overlay_path": "/tmp",
        "result_path": f"results/{model_key}.json",
        "runtime": {
            "device": "cuda:0",
            "eta_hours": 0.1,
            "resume": False,
            "stop_after_examples": 4096,
            "latency": None,
        },
        "training": {
            "dataset_id": "vk_lsvd_up0.01_ir0.01",
            "feature_regime": "C+P",
            "model_key": model_key,
            "phase": "tuning",
            "seed": 42,
            "effective_batch_size": 4096,
            "validate_every_examples": 4_001_792,
            "runtime_value_checks": True,
            "capacity_resolution_sha256": DIGEST,
            "encoder_contract_sha256": DIGEST,
            "cache_metadata_sha256": DIGEST,
            "cache_content_sha256": DIGEST,
            "full_materialized_scan_sha256": DIGEST,
            "targets_multitask_v2_sha256": DIGEST,
            "task_constants_sha256": DIGEST,
        },
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "float_schema",
        "unknown_trainer",
        "scientific",
        "extra_key",
        "bad_model",
        "bad_budget",
        "checks_disabled",
        "resume",
        "latency",
        "bad_hash",
    ),
)
def test_multitask_smoke_contract_is_fail_closed(tmp_path: Path, mutation: str) -> None:
    worker = _load_worker()
    resolved = _resolved(tmp_path)
    if mutation == "float_schema":
        resolved["schema_version"] = 5.0
    elif mutation == "unknown_trainer":
        resolved["trainer_kind"] = "other"
    elif mutation == "scientific":
        resolved["execution_kind"] = "scientific_tuning"
    elif mutation == "extra_key":
        resolved["unexpected"] = None
    elif mutation == "bad_model":
        resolved["training"]["model_key"] = "afn_plus"
    elif mutation == "bad_budget":
        resolved["runtime"]["stop_after_examples"] = 8192
    elif mutation == "checks_disabled":
        resolved["training"]["runtime_value_checks"] = False
    elif mutation == "resume":
        resolved["runtime"]["resume"] = True
    elif mutation == "latency":
        resolved["runtime"]["latency"] = {}
    elif mutation == "bad_hash":
        resolved["training"]["cache_content_sha256"] = True
    with pytest.raises(worker.WorkerError):
        worker._validate_resolved_execution_contract(resolved)


@pytest.mark.parametrize("model_key", ("dcdn_core", "dcnv2_full_rank"))
def test_multitask_smoke_command_is_supervised_and_bound(
    tmp_path: Path, model_key: str
) -> None:
    worker = _load_worker()
    resolved = _resolved(tmp_path, model_key=model_key)
    assert (
        worker._validate_resolved_execution_contract(resolved) == "implementation_smoke"
    )
    command = worker._trainer_command(
        tmp_path / "resolved.yaml",
        resolved,
        DIGEST,
    )
    assert command[:3] == [
        worker.sys.executable,
        "-m",
        "dcdn.supervised_exec",
    ]
    assert str(worker._MULTITASK_SMOKE_SCRIPT) in command
    assert command[command.index("--model") + 1] == model_key
    assert command[command.index("--run-id") + 1] == resolved["run_id"]
    assert command[command.index("--resolved-config-sha256") + 1] == DIGEST


def _receipt(resolved: dict[str, Any], protocol_sha256: str) -> dict[str, Any]:
    training = resolved["training"]
    return {
        "schema_version": 5,
        "artifact_role": "vk_lsvd_seven_head_implementation_smoke",
        "execution_kind": "implementation_smoke",
        "scientific_result_eligible": False,
        "run_id": resolved["run_id"],
        "resolved_config_sha256": DIGEST,
        "model_key": training["model_key"],
        "feature_regime": "C+P",
        "seed": 42,
        "examples_seen": 4096,
        "optimizer_steps": 1,
        "validation_checks": 0,
        "test_metrics": None,
        "training_loss_non_scientific": 1.0,
        "elapsed_seconds": 1.0,
        "throughput_examples_per_second": 4096.0,
        "peak_vram_bytes": 1024,
        "trainable_parameters": 100,
        "gradient_norm_before_clip": 1.0,
        "capacity_artifact_file_sha256": DIGEST,
        "nonfinite_count": 0,
        "joint_model_contract_sha256": DIGEST,
        "protocol_sha256": protocol_sha256,
        "encoder_contract_sha256": DIGEST,
        "capacity_resolution_sha256": DIGEST,
        "cache_metadata_sha256": DIGEST,
        "cache_content_sha256": DIGEST,
        "full_materialized_scan_sha256": DIGEST,
        "cost_snapshot": {"launch_allowed": True},
        "storage_snapshot": {"ok": True},
        "targets_multitask_v2_sha256": DIGEST,
        "task_constants_sha256": DIGEST,
        "test_unlocked": False,
        "telemetry": "disabled",
    }


def _result_fixture(
    root: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    worker = _load_worker()
    resolved = _resolved(root)
    protocol = {
        "schema_version": 3,
        "dataset_id": "vk_lsvd_up0.01_ir0.01",
        "feature_regime": "C+P",
        "required_model_keys": [
            "dcnv2_full_rank",
            "dcdn_core",
            "afn_plus",
            "dcnv2_capacity_matched_core",
        ],
        "test_unlocked": False,
        "test_path_available": False,
        "hashes": {},
    }
    protocol_data = yaml.safe_dump(protocol, sort_keys=True).encode("utf-8")
    protocol_sha256 = _write_sidecar(root / "protocol.yaml", protocol_data)
    receipt = _receipt(resolved, protocol_sha256)
    return worker, resolved, receipt


def _publish_result(root: Path, resolved: dict[str, Any], receipt: Any) -> None:
    path = root / resolved["result_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=True) + "\n"
    ).encode("ascii")
    _write_sidecar(path, data)


def test_multitask_smoke_result_verifier_accepts_exact_receipt(
    tmp_path: Path,
) -> None:
    worker, resolved, receipt = _result_fixture(tmp_path)
    _publish_result(tmp_path, resolved, receipt)
    digest, eligible = worker._verify_multitask_smoke_result(
        project_root=tmp_path,
        resolved=resolved,
        resolved_digest=DIGEST,
        run_id=resolved["run_id"],
    )
    assert len(digest) == 64
    assert eligible is False


@pytest.mark.parametrize(
    "mutation",
    (
        "scientific",
        "test_metric",
        "hash_drift",
        "missing_key",
        "nonfinite",
        "wrong_model",
        "failed_guard",
    ),
)
def test_multitask_smoke_result_verifier_rejects_tampering(
    tmp_path: Path, mutation: str
) -> None:
    worker, resolved, receipt = _result_fixture(tmp_path)
    altered = copy.deepcopy(receipt)
    if mutation == "scientific":
        altered["scientific_result_eligible"] = True
    elif mutation == "test_metric":
        altered["test_metrics"] = {"gauc": 0.9}
    elif mutation == "hash_drift":
        altered["cache_content_sha256"] = hashlib.sha256(b"drift").hexdigest()
    elif mutation == "missing_key":
        altered.pop("nonfinite_count")
    elif mutation == "nonfinite":
        altered["training_loss_non_scientific"] = float("nan")
    elif mutation == "wrong_model":
        altered["model_key"] = "dcnv2_full_rank"
    elif mutation == "failed_guard":
        altered["storage_snapshot"]["ok"] = False
    _publish_result(tmp_path, resolved, altered)
    with pytest.raises(worker.WorkerError):
        worker._verify_multitask_smoke_result(
            project_root=tmp_path,
            resolved=resolved,
            resolved_digest=DIGEST,
            run_id=resolved["run_id"],
        )
