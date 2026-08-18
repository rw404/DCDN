from __future__ import annotations

import json
from pathlib import Path

import pytest

from dcdn.registry import RegistryError, RunRegistry

DIGEST = "a" * 64
COMMIT = "b" * 40


def pending_fields(run_id: str = "vk-dcdn-s42-trial0") -> dict:
    return {
        "run_id": run_id,
        "execution_kind": "scientific_tuning",
        "seed": 42,
        "code_commit": COMMIT,
        "code_diff_sha256": DIGEST,
        "resolved_config_path": "configs/resolved/example.yaml",
        "resolved_config_sha256": DIGEST,
        "data_manifest_sha256": DIGEST,
        "features_sha256": DIGEST,
        "protocol_sha256": DIGEST,
        "reproduction_command_path": "runs/commands/example.sh",
        "history_path": "runs/history/example.jsonl",
        "log_path": "runs/logs/example.log",
    }


def test_full_lifecycle_and_active_run_query(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry")
    pending = registry.create_pending(**pending_fields())
    assert pending["status"] == "pending"
    assert registry.active_runs() == []

    running = registry.transition(
        pending["run_id"],
        "running",
        pid=1234,
        gpu=1,
    )
    assert running["pid"] == 1234
    assert [row["run_id"] for row in registry.active_runs()] == [pending["run_id"]]
    registry.heartbeat(pending["run_id"], pid=1234)

    completed = registry.transition(
        pending["run_id"],
        "completed",
        reason="all declared outputs verified",
        scientific_result_eligible=True,
    )
    assert completed["pid"] == 1234
    assert completed["gpu"] == 1
    assert completed["execution_kind"] == "scientific_tuning"
    assert completed["scientific_result_eligible"] is True
    assert completed["finished_at_utc"].endswith("Z")
    assert registry.active_runs() == []


def test_run_id_is_immutable_and_duplicate_is_rejected(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry")
    fields = pending_fields()
    registry.create_pending(**fields)
    with pytest.raises(RegistryError, match="already exists"):
        registry.create_pending(**fields)

    path = registry.records / f"{fields['run_id']}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = "different-id"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RegistryError, match="identity"):
        registry.read(fields["run_id"])


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "AA", "has space"])
def test_unsafe_run_id_is_rejected(tmp_path: Path, run_id: str) -> None:
    registry = RunRegistry(tmp_path / "registry")
    with pytest.raises(RegistryError, match="run_id"):
        registry.create_pending(**pending_fields(run_id))


def test_paths_credentials_and_hashes_fail_closed(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry")
    fields = pending_fields()
    fields["log_path"] = "../../escape.log"
    with pytest.raises(RegistryError, match="log_path"):
        registry.create_pending(**fields)

    fields = pending_fields("bad-secret")
    credential_marker = "github_" + "pat_" + ("a" * 40)
    fields["reproduction_command_path"] = f"runs/{credential_marker}.sh"
    with pytest.raises(RegistryError, match="credential"):
        registry.create_pending(**fields)

    fields = pending_fields("bad-digest")
    fields["protocol_sha256"] = "UNRESOLVED"
    with pytest.raises(RegistryError, match="protocol_sha256"):
        registry.create_pending(**fields)


def test_invalid_transitions_and_heartbeat_owner_are_rejected(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry")
    run_id = "transition-test"
    registry.create_pending(**pending_fields(run_id))
    with pytest.raises(RegistryError, match="invalid transition"):
        registry.transition(run_id, "completed", reason="not started")

    registry.transition(run_id, "running", pid=10, gpu=0)
    with pytest.raises(RegistryError, match="does not own"):
        registry.heartbeat(run_id, pid=11)
    with pytest.raises(RegistryError, match="invalid transition"):
        registry.transition(run_id, "pending")
    with pytest.raises(RegistryError, match="reason"):
        registry.transition(run_id, "failed", reason="")


def test_execution_kind_is_required_and_terminal_eligibility_is_fail_closed(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    missing = pending_fields("missing-execution-kind")
    missing.pop("execution_kind")
    with pytest.raises(RegistryError, match="execution_kind"):
        registry.create_pending(**missing)

    invalid = pending_fields("invalid-execution-kind")
    invalid["execution_kind"] = ["implementation_smoke"]
    with pytest.raises(RegistryError, match="execution_kind"):
        registry.create_pending(**invalid)

    smoke_id = "implementation-smoke-lifecycle"
    smoke = pending_fields(smoke_id)
    smoke["execution_kind"] = "implementation_smoke"
    pending = registry.create_pending(**smoke)
    assert pending["scientific_result_eligible"] is None
    running = registry.transition(smoke_id, "running", pid=4321, gpu=0)
    assert running["execution_kind"] == "implementation_smoke"
    assert running["scientific_result_eligible"] is None

    with pytest.raises(RegistryError, match="never be a scientific result"):
        registry.transition(
            smoke_id,
            "completed",
            reason="invalid scientific smoke",
            scientific_result_eligible=True,
        )
    unchanged = registry.read(smoke_id)
    assert unchanged["status"] == "running"
    assert unchanged["scientific_result_eligible"] is None

    completed = registry.transition(
        smoke_id,
        "completed",
        reason="NON-SCIENTIFIC implementation smoke verified",
        scientific_result_eligible=False,
    )
    assert completed["scientific_result_eligible"] is False
    assert completed["execution_kind"] == "implementation_smoke"


def test_completion_requires_eligibility_and_failure_forbids_it(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    completed_id = "completed-eligibility-required"
    registry.create_pending(**pending_fields(completed_id))
    registry.transition(completed_id, "running", pid=20, gpu=0)
    with pytest.raises(RegistryError, match="requires boolean"):
        registry.transition(completed_id, "completed", reason="missing eligibility")
    assert registry.read(completed_id)["status"] == "running"

    failed_id = "failure-forbids-eligibility"
    registry.create_pending(**pending_fields(failed_id))
    registry.transition(failed_id, "running", pid=21, gpu=1)
    with pytest.raises(RegistryError, match="valid only at completion"):
        registry.transition(
            failed_id,
            "failed",
            reason="synthetic failure",
            scientific_result_eligible=False,
        )
    assert registry.read(failed_id)["status"] == "running"
