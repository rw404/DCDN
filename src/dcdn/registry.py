"""Persistent per-run registry with immutable identities and atomic updates."""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SECRET_RE = re.compile(
    r"(github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
)
STATUSES = frozenset({"pending", "running", "completed", "failed"})
TRANSITIONS = {
    "pending": frozenset({"running", "failed"}),
    "running": frozenset({"completed", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
}


class RegistryError(RuntimeError):
    """Raised when a run record is incomplete, unsafe, or inconsistent."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise RegistryError(f"{field} must not escape the persistent repository")
    return value


def _validate_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RegistryError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_strings(nested)
    elif isinstance(value, list | tuple):
        for nested in value:
            yield from _walk_strings(nested)


def _reject_credentials(payload: Mapping[str, Any]) -> None:
    if any(SECRET_RE.search(value) for value in _walk_strings(payload)):
        raise RegistryError("run metadata contains credential-like material")


def validate_run_record(record: Mapping[str, Any]) -> None:
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise RegistryError("run_id has an invalid format")
    if record.get("status") not in STATUSES:
        raise RegistryError("status is invalid")
    if not isinstance(record.get("seed"), int):
        raise RegistryError("seed must be an integer")
    if (
        not isinstance(record.get("code_commit"), str)
        or COMMIT_RE.fullmatch(record["code_commit"]) is None
    ):
        raise RegistryError("code_commit must be a full lowercase Git commit")
    for name in (
        "code_diff_sha256",
        "resolved_config_sha256",
        "data_manifest_sha256",
        "features_sha256",
        "protocol_sha256",
    ):
        _validate_digest(record.get(name), name)
    for name in (
        "resolved_config_path",
        "reproduction_command_path",
        "history_path",
        "log_path",
    ):
        _validate_relative_path(record.get(name), name)

    if record["status"] == "running":
        if not isinstance(record.get("pid"), int) or record["pid"] <= 0:
            raise RegistryError("a running record requires a positive PID")
        if not isinstance(record.get("gpu"), int) or record["gpu"] not in (0, 1):
            raise RegistryError("a running record requires GPU 0 or 1")
        if not isinstance(record.get("heartbeat_utc"), str):
            raise RegistryError("a running record requires a heartbeat")
    if record["status"] in {"completed", "failed"}:
        if record.get("pid") is not None and (
            not isinstance(record["pid"], int) or record["pid"] <= 0
        ):
            raise RegistryError("a terminal PID must be positive when present")
        if record.get("gpu") is not None and record["gpu"] not in (0, 1):
            raise RegistryError("a terminal GPU must be zero or one when present")
        if not isinstance(record.get("finished_at_utc"), str):
            raise RegistryError("a terminal record requires finished_at_utc")
        reason = record.get("completion_or_failure_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise RegistryError("a terminal record requires a reason")

    _reject_credentials(record)


class RunRegistry:
    """Directory-backed registry safe for independent single-GPU processes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.records = self.root / "records"
        self.locks = self.root / "locks"
        self.records.mkdir(parents=True, exist_ok=True)
        self.locks.mkdir(parents=True, exist_ok=True)

    def _validate_run_id(self, run_id: str) -> None:
        if RUN_ID_RE.fullmatch(run_id) is None:
            raise RegistryError("run_id has an invalid format")

    def _record_path(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self.records / f"{run_id}.json"

    @contextmanager
    def _lock(self, run_id: str) -> Iterator[None]:
        self._validate_run_id(run_id)
        path = self.locks / f"{run_id}.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o640)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _write(self, record: Mapping[str, Any]) -> None:
        validate_run_record(record)
        target = self._record_path(str(record["run_id"]))
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        serialized = (
            json.dumps(
                record,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
            )
            + "\n"
        )
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(self.records, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def create_pending(self, **fields: Any) -> Mapping[str, Any]:
        run_id = fields.get("run_id")
        if not isinstance(run_id, str):
            raise RegistryError("run_id is required")
        now = _utc_now()
        record = {
            **fields,
            "status": "pending",
            "pid": None,
            "gpu": None,
            "created_at_utc": now,
            "updated_at_utc": now,
            "heartbeat_utc": None,
            "started_at_utc": None,
            "finished_at_utc": None,
            "completion_or_failure_reason": None,
        }
        with self._lock(run_id):
            target = self._record_path(run_id)
            if target.exists():
                raise RegistryError(f"run_id already exists: {run_id}")
            self._write(record)
        return deepcopy(record)

    def read(self, run_id: str) -> Mapping[str, Any]:
        target = self._record_path(run_id)
        try:
            record = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RegistryError(f"unknown run_id: {run_id}") from exc
        if not isinstance(record, Mapping):
            raise RegistryError(f"invalid run record root: {run_id}")
        validate_run_record(record)
        if record["run_id"] != run_id:
            raise RegistryError("record identity does not match its filename")
        return record

    def transition(
        self,
        run_id: str,
        status: str,
        *,
        pid: int | None = None,
        gpu: int | None = None,
        reason: str | None = None,
    ) -> Mapping[str, Any]:
        if status not in STATUSES:
            raise RegistryError(f"invalid target status: {status}")
        with self._lock(run_id):
            current = dict(self.read(run_id))
            if status not in TRANSITIONS[current["status"]]:
                raise RegistryError(
                    f"invalid transition: {current['status']} -> {status}"
                )
            now = _utc_now()
            current["status"] = status
            current["updated_at_utc"] = now
            if status == "running":
                current["pid"] = pid
                current["gpu"] = gpu
                current["started_at_utc"] = now
                current["heartbeat_utc"] = now
            else:
                current["heartbeat_utc"] = None
                current["finished_at_utc"] = now
                current["completion_or_failure_reason"] = reason
            self._write(current)
        return deepcopy(current)

    def heartbeat(self, run_id: str, *, pid: int) -> Mapping[str, Any]:
        with self._lock(run_id):
            current = dict(self.read(run_id))
            if current["status"] != "running":
                raise RegistryError("heartbeat is valid only for a running job")
            if current["pid"] != pid:
                raise RegistryError("heartbeat PID does not own the run")
            now = _utc_now()
            current["heartbeat_utc"] = now
            current["updated_at_utc"] = now
            self._write(current)
        return deepcopy(current)

    def active_runs(self) -> list[Mapping[str, Any]]:
        active: list[Mapping[str, Any]] = []
        for path in sorted(self.records.glob("*.json")):
            record = self.read(path.stem)
            if record["status"] == "running":
                active.append(record)
        return active
