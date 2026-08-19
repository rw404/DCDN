from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

from dcdn.registry import (
    CHECKPOINT_SIZE_VERIFIER_FAILURE_REASON,
    RegistryError,
    RegistryIOError,
    RunRegistry,
)
from dcdn.registry_helper import execute_request

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


@pytest.mark.parametrize("seed", [42.0, True])
def test_seed_requires_an_exact_integer(tmp_path: Path, seed: object) -> None:
    registry = RunRegistry(tmp_path / "registry")
    fields = pending_fields("bad-seed")
    fields["seed"] = seed
    with pytest.raises(RegistryError, match="seed"):
        registry.create_pending(**fields)


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


def test_heartbeat_uses_unique_temp_and_skips_file_and_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    run_id = "relaxed-heartbeat"
    registry.create_pending(**pending_fields(run_id))
    registry.ensure_running(run_id, pid=1234, gpu=0)
    legacy_temporary = registry.records / f".{run_id}.json.{os.getpid()}.tmp"
    legacy_temporary.write_text("stale", encoding="ascii")
    sync_calls: list[Path] = []
    file_sync_calls: list[int] = []
    monkeypatch.setattr(
        registry,
        "_sync_directory",
        lambda path: sync_calls.append(path),
    )
    monkeypatch.setattr(
        "dcdn.registry.os.fsync",
        lambda descriptor: file_sync_calls.append(descriptor),
    )

    registry.heartbeat(run_id, pid=1234)

    assert sync_calls == []
    assert file_sync_calls == []
    assert legacy_temporary.read_text(encoding="ascii") == "stale"
    registry.ensure_terminal(
        run_id,
        "failed",
        pid=1234,
        gpu=0,
        reason="synthetic terminal",
    )
    assert sync_calls == [registry.records]
    assert len(file_sync_calls) == 1


@pytest.mark.parametrize(
    ("error_number", "transient"),
    [(errno.EIO, True), (errno.ENOSPC, False)],
)
def test_registry_io_error_classifies_errno(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    transient: bool,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    run_id = f"errno-{error_number}"
    registry.create_pending(**pending_fields(run_id))
    registry.ensure_running(run_id, pid=100, gpu=0)

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError(error_number, "synthetic registry failure")

    monkeypatch.setattr("dcdn.registry.os.replace", fail_replace)
    with pytest.raises(RegistryIOError) as caught:
        registry.heartbeat(run_id, pid=100)
    assert caught.value.stage == "record_replace"
    assert caught.value.errno == error_number
    assert caught.value.transient_candidate is transient


def test_terminal_reconcile_survives_ambiguous_post_replace_eio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    run_id = "terminal-reconcile"
    registry.create_pending(**pending_fields(run_id))
    registry.ensure_running(run_id, pid=200, gpu=1)
    calls = 0
    real_sync = registry._sync_directory

    def fail_once(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RegistryIOError(
                "directory_fsync",
                OSError(errno.EIO, "synthetic ambiguous durability"),
            )
        real_sync(path)

    monkeypatch.setattr(registry, "_sync_directory", fail_once)
    with pytest.raises(RegistryIOError, match="directory_fsync"):
        registry.ensure_terminal(
            run_id,
            "completed",
            pid=200,
            gpu=1,
            reason="verified exact output",
            scientific_result_eligible=True,
        )
    assert registry.read(run_id)["status"] == "completed"

    reconciled = registry.ensure_terminal(
        run_id,
        "completed",
        pid=200,
        gpu=1,
        reason="verified exact output",
        scientific_result_eligible=True,
    )
    assert reconciled["status"] == "completed"
    with pytest.raises(RegistryError, match="conflicting terminal"):
        registry.ensure_terminal(
            run_id,
            "failed",
            pid=200,
            gpu=1,
            reason="must not downgrade completed",
        )


def test_incident_receipt_is_hash_bound_idempotent_and_reopened(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    payload = {
        "schema_version": 1,
        "artifact_role": "run_registry_io_incident",
        "incident_id": "registry-io-test-1",
        "run_id": "incident-run",
        "operation": "heartbeat",
        "errno": errno.EIO,
    }
    first = registry.publish_incident(payload)
    second = registry.publish_incident(payload)

    assert first == second
    path = Path(first["path"])
    assert path.name.endswith(f"-{first['sha256']}.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first["sha256"]
    path.write_bytes(b"corrupt\n")
    with pytest.raises(RegistryError, match="collision"):
        registry.publish_incident(payload)


def test_terminal_reconcile_enforces_and_preserves_exact_owner(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    running_id = "terminal-owner-running"
    registry.create_pending(**pending_fields(running_id))
    registry.ensure_running(running_id, pid=300, gpu=0)
    with pytest.raises(RegistryError, match="owner mismatch"):
        registry.ensure_terminal(
            running_id,
            "failed",
            pid=301,
            gpu=0,
            reason="wrong owner",
        )
    assert registry.read(running_id)["status"] == "running"

    pending_id = "terminal-owner-pending"
    registry.create_pending(**pending_fields(pending_id))
    failed = registry.ensure_terminal(
        pending_id,
        "failed",
        pid=302,
        gpu=1,
        reason="spawn failed before claim reconciliation",
    )
    assert failed["pid"] == 302
    assert failed["gpu"] == 1
    assert (
        registry.ensure_terminal(
            pending_id,
            "failed",
            pid=302,
            gpu=1,
            reason="spawn failed before claim reconciliation",
        )["status"]
        == "failed"
    )


def test_incident_post_link_eio_preserves_target_for_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    payload = {
        "schema_version": 1,
        "artifact_role": "run_registry_io_incident",
        "incident_id": "registry-io-post-link",
        "run_id": "incident-post-link",
        "operation": "heartbeat",
    }
    calls = 0
    real_sync = registry._sync_directory

    def fail_once(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RegistryIOError(
                "directory_fsync",
                OSError(errno.EIO, "synthetic post-link ambiguity"),
            )
        real_sync(path)

    monkeypatch.setattr(registry, "_sync_directory", fail_once)
    with pytest.raises(RegistryIOError, match="directory_fsync"):
        registry.publish_incident(payload)
    targets = list(registry.incidents.glob("*.json"))
    assert len(targets) == 1
    preserved = targets[0].read_bytes()

    receipt = registry.publish_incident(payload)
    assert Path(receipt["path"]).read_bytes() == preserved
    assert hashlib.sha256(preserved).hexdigest() == receipt["sha256"]


def test_registry_helper_executes_idempotent_requests(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    run_id = "helper-idempotent"
    registry.create_pending(**pending_fields(run_id))
    common = {
        "registry_root": str(registry.root),
        "run_id": run_id,
        "pid": 444,
    }
    assert execute_request({**common, "operation": "ensure_running", "gpu": 0}) == {
        "status": "running"
    }
    heartbeat = execute_request({**common, "operation": "heartbeat"})
    assert heartbeat["status"] == "running"
    assert execute_request(
        {
            **common,
            "operation": "ensure_terminal",
            "status": "failed",
            "gpu": 0,
            "reason": "synthetic helper terminal",
            "scientific_result_eligible": None,
        }
    ) == {"status": "failed"}


def capacity_fields() -> dict:
    return {
        "model_key": "dcnv2_capacity_matched_core",
        "candidate_id": "capacity_rank_lower",
        "code_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "learning_rate": 0.0003,
        "capacity_low_rank": 12,
        "capacity_resolution_sha256": (
            "d3e65d90fe0184ec48cf15abbe7358f8fd57b4944b5a95f3201b2b4e3fe81565"
        ),
        "ordinary_dcnv2_selection_sha256": (
            "869cbcb7a227b274700cda142f23c59040497bfdfe4da101888e2cc63f39d67a"
        ),
        "capacity_lr_binding_sha256": "c" * 64,
    }


def test_capacity_registry_binding_is_exact_and_legacy_is_unchanged(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    fields = pending_fields("capacity-binding-happy")
    fields.update(capacity_fields())
    created = registry.create_pending(**fields)
    for name, expected in capacity_fields().items():
        assert created[name] == expected

    legacy = registry.create_pending(**pending_fields("legacy-still-valid"))
    assert "model_key" not in legacy


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_key", "dcnv2_full_rank"),
        ("candidate_id", "capacity_rank_upper"),
        ("learning_rate", 0.001),
        ("learning_rate", 3),
        ("capacity_low_rank", 13),
        ("capacity_low_rank", 12.0),
        ("capacity_low_rank", True),
        ("capacity_resolution_sha256", "d" * 64),
        ("ordinary_dcnv2_selection_sha256", "e" * 64),
        ("capacity_lr_binding_sha256", "invalid"),
    ],
)
def test_capacity_registry_binding_rejects_tamper_and_wrong_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    fields = pending_fields(f"capacity-bad-{field.replace('_', '-')}")
    fields.update(capacity_fields())
    fields[field] = value
    with pytest.raises(RegistryError, match="capacity"):
        registry.create_pending(**fields)


def test_partial_capacity_registry_binding_is_rejected(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "registry")
    fields = pending_fields("capacity-binding-partial")
    fields["model_key"] = "dcnv2_capacity_matched_core"
    with pytest.raises(RegistryError, match="incomplete"):
        registry.create_pending(**fields)


def test_checkpoint_size_verifier_reconciliation_preserves_failure_record(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "registry")
    run_id = "checkpoint-size-reconciliation"
    registry.create_pending(**pending_fields(run_id))
    registry.ensure_running(run_id, pid=4567, gpu=1)
    failed = registry.ensure_terminal(
        run_id,
        "failed",
        pid=4567,
        gpu=1,
        reason=CHECKPOINT_SIZE_VERIFIER_FAILURE_REASON,
    )
    record_path = registry.records / f"{run_id}.json"
    prior_digest = hashlib.sha256(record_path.read_bytes()).hexdigest()

    with pytest.raises(RegistryError, match="digest drifted"):
        registry.reconcile_checkpoint_size_verifier_failure(
            run_id,
            expected_failed_record_sha256="0" * 64,
            verifier_code_commit="c" * 40,
            verifier_code_diff_sha256=hashlib.sha256(b"").hexdigest(),
            result_sha256="d" * 64,
        )
    assert registry.read(run_id)["status"] == "failed"

    reconciled = registry.reconcile_checkpoint_size_verifier_failure(
        run_id,
        expected_failed_record_sha256=prior_digest,
        verifier_code_commit="c" * 40,
        verifier_code_diff_sha256=hashlib.sha256(b"").hexdigest(),
        result_sha256="d" * 64,
    )
    evidence = reconciled["terminal_reconciliation"]
    assert reconciled["status"] == "completed"
    assert reconciled["scientific_result_eligible"] is True
    assert evidence["prior_record_sha256"] == prior_digest
    assert evidence["prior_record"] == failed
    assert evidence["result_sha256"] == "d" * 64

    repeated = registry.reconcile_checkpoint_size_verifier_failure(
        run_id,
        expected_failed_record_sha256=prior_digest,
        verifier_code_commit="c" * 40,
        verifier_code_diff_sha256=hashlib.sha256(b"").hexdigest(),
        result_sha256="d" * 64,
    )
    assert repeated == reconciled
