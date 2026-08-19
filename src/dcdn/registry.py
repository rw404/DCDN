"""Persistent per-run registry with immutable identities and atomic updates."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SECRET_RE = re.compile(
    r"(github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
)
STATUSES = frozenset({"pending", "running", "completed", "failed"})
EXECUTION_KINDS = frozenset({"implementation_smoke", "scientific_tuning"})
TRANSITIONS = {
    "pending": frozenset({"running", "failed"}),
    "running": frozenset({"completed", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
}
TRANSIENT_IO_ERRNOS = frozenset(
    value
    for name in (
        "EAGAIN",
        "EBUSY",
        "ECONNABORTED",
        "ECONNRESET",
        "EHOSTDOWN",
        "EHOSTUNREACH",
        "EINTR",
        "EIO",
        "ENETDOWN",
        "ENETRESET",
        "ENETUNREACH",
        "ENOTCONN",
        "ESTALE",
        "ETIMEDOUT",
    )
    if (value := getattr(errno, name, None)) is not None
)
INCIDENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


class RegistryError(RuntimeError):
    """Raised when a run record is incomplete, unsafe, or inconsistent."""


class RegistryIOError(RegistryError):
    """A staged registry filesystem operation failed.

    ``transient_candidate`` permits only bounded supervisor retry. It never
    establishes that an error was harmless or that persistent storage is
    healthy.
    """

    def __init__(self, stage: str, error: OSError) -> None:
        self.stage = stage
        self.errno = error.errno
        self.transient_candidate = error.errno in TRANSIENT_IO_ERRNOS
        detail = error.strerror or str(error)
        super().__init__(
            f"registry I/O failed at {stage}: errno={error.errno}: {detail}"
        )


def _io_error(stage: str, error: OSError) -> RegistryIOError:
    return RegistryIOError(stage, error)


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
    execution_kind = record.get("execution_kind")
    if type(execution_kind) is not str or execution_kind not in EXECUTION_KINDS:
        raise RegistryError("execution_kind is invalid")
    if "scientific_result_eligible" not in record:
        raise RegistryError("scientific_result_eligible is required")
    eligibility = record["scientific_result_eligible"]
    if record["status"] == "completed":
        if type(eligibility) is not bool:
            raise RegistryError(
                "a completed record requires boolean scientific_result_eligible"
            )
        if execution_kind == "implementation_smoke" and eligibility is not False:
            raise RegistryError(
                "an implementation smoke can never be a scientific result"
            )
    elif eligibility is not None:
        raise RegistryError(
            "scientific_result_eligible must remain null before completion"
        )
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
        self.incidents = self.root / "incidents"
        try:
            self._initialize_directories()
        except OSError as exc:
            raise _io_error("registry_initialize", exc) from exc

    def _initialize_directories(self) -> None:
        self.records.mkdir(parents=True, exist_ok=True)
        self.locks.mkdir(parents=True, exist_ok=True)
        self.incidents.mkdir(parents=True, exist_ok=True)

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
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o640)
        except OSError as exc:
            raise _io_error("lock_open", exc) from exc
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except OSError as exc:
                raise _io_error("lock_acquire", exc) from exc
            try:
                yield
            except BaseException:
                with suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                raise
            else:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError as exc:
                    raise _io_error("lock_release", exc) from exc
        finally:
            with suppress(OSError):
                os.close(descriptor)

    @staticmethod
    def _write_temp_bytes(
        parent: Path,
        prefix: str,
        data: bytes,
        *,
        sync_file: bool = True,
    ) -> Path:
        try:
            descriptor, raw_path = tempfile.mkstemp(
                prefix=prefix,
                suffix=".tmp",
                dir=parent,
            )
        except OSError as exc:
            raise _io_error("temporary_create", exc) from exc
        path = Path(raw_path)
        try:
            try:
                os.fchmod(descriptor, 0o640)
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    descriptor = -1
                    handle.write(data)
                    handle.flush()
                    if sync_file:
                        os.fsync(handle.fileno())
            except OSError as exc:
                raise _io_error("temporary_write_fsync", exc) from exc
        except BaseException:
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(OSError):
                path.unlink(missing_ok=True)
            raise
        return path

    @staticmethod
    def _sync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError as exc:
            raise _io_error("directory_open", exc) from exc
        try:
            try:
                os.fsync(descriptor)
            except OSError as exc:
                raise _io_error("directory_fsync", exc) from exc
        finally:
            with suppress(OSError):
                os.close(descriptor)

    @staticmethod
    def _read_bytes(path: Path, *, stage: str) -> bytes:
        try:
            return path.read_bytes()
        except OSError as exc:
            raise _io_error(stage, exc) from exc

    def _write(
        self,
        record: Mapping[str, Any],
        *,
        durability: Literal["relaxed", "full"] = "full",
    ) -> None:
        validate_run_record(record)
        if durability not in {"relaxed", "full"}:
            raise ValueError(f"invalid registry durability: {durability}")
        target = self._record_path(str(record["run_id"]))
        serialized = (
            json.dumps(
                record,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
            )
            + "\n"
        ).encode("ascii")
        temporary = self._write_temp_bytes(
            self.records,
            f".{target.name}.{os.getpid()}-",
            serialized,
            sync_file=durability == "full",
        )
        try:
            try:
                os.replace(temporary, target)
            except OSError as exc:
                raise _io_error("record_replace", exc) from exc
            if durability == "full":
                self._sync_directory(self.records)
            if self._read_bytes(target, stage="record_reopen") != serialized:
                raise RegistryError(
                    "reopened run record bytes differ after publication"
                )
        except BaseException:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            raise _io_error("temporary_cleanup", exc) from exc

    def create_pending(self, **fields: Any) -> Mapping[str, Any]:
        run_id = fields.get("run_id")
        if not isinstance(run_id, str):
            raise RegistryError("run_id is required")
        now = _utc_now()
        record = {
            **fields,
            "status": "pending",
            "scientific_result_eligible": None,
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
        except OSError as exc:
            raise _io_error("record_read", exc) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryError(f"invalid run record JSON: {run_id}") from exc
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
        scientific_result_eligible: bool | None = None,
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
            if status == "completed":
                current["scientific_result_eligible"] = scientific_result_eligible
            elif scientific_result_eligible is not None:
                raise RegistryError(
                    "scientific_result_eligible is valid only at completion"
                )
            self._write(current)
        return deepcopy(current)

    def ensure_running(
        self,
        run_id: str,
        *,
        pid: int,
        gpu: int,
    ) -> Mapping[str, Any]:
        """Publish or reconcile one exact full-durability running owner."""

        with self._lock(run_id):
            current = dict(self.read(run_id))
            if current["status"] == "running":
                if current["pid"] != pid or current["gpu"] != gpu:
                    raise RegistryError("running reconciliation owner mismatch")
                self._sync_directory(self.records)
                return deepcopy(self.read(run_id))
            if current["status"] != "pending":
                raise RegistryError(
                    f"cannot reconcile running state from {current['status']}"
                )
            now = _utc_now()
            current["status"] = "running"
            current["pid"] = pid
            current["gpu"] = gpu
            current["started_at_utc"] = now
            current["heartbeat_utc"] = now
            current["updated_at_utc"] = now
            self._write(current, durability="full")
        return deepcopy(current)

    def ensure_terminal(
        self,
        run_id: str,
        status: Literal["completed", "failed"],
        *,
        pid: int | None,
        gpu: int | None,
        reason: str,
        scientific_result_eligible: bool | None = None,
    ) -> Mapping[str, Any]:
        """Publish or reconcile an exact terminal intent without downgrades."""

        if status not in {"completed", "failed"}:
            raise RegistryError("terminal reconciliation status is invalid")
        with self._lock(run_id):
            current = dict(self.read(run_id))
            if current["status"] == status:
                if current.get("completion_or_failure_reason") != reason:
                    raise RegistryError("terminal reconciliation reason mismatch")
                if (
                    status == "completed"
                    and current.get("scientific_result_eligible")
                    is not scientific_result_eligible
                ):
                    raise RegistryError("terminal reconciliation eligibility mismatch")
                if current.get("pid") != pid:
                    raise RegistryError("terminal reconciliation PID mismatch")
                if current.get("gpu") != gpu:
                    raise RegistryError("terminal reconciliation GPU mismatch")
                self._sync_directory(self.records)
                return deepcopy(self.read(run_id))
            if current["status"] in {"completed", "failed"}:
                raise RegistryError(
                    f"conflicting terminal state: {current['status']} -> {status}"
                )
            if status not in TRANSITIONS[current["status"]]:
                raise RegistryError(
                    f"invalid transition: {current['status']} -> {status}"
                )
            if status == "completed" and type(scientific_result_eligible) is not bool:
                raise RegistryError("completed reconciliation requires eligibility")
            if status == "failed" and scientific_result_eligible is not None:
                raise RegistryError("failed reconciliation forbids eligibility")
            if current["status"] == "running":
                if current.get("pid") != pid or current.get("gpu") != gpu:
                    raise RegistryError("terminal transition owner mismatch")
            else:
                current["pid"] = pid
                current["gpu"] = gpu
            now = _utc_now()
            current["status"] = status
            current["updated_at_utc"] = now
            current["heartbeat_utc"] = None
            current["finished_at_utc"] = now
            current["completion_or_failure_reason"] = reason
            if status == "completed":
                current["scientific_result_eligible"] = scientific_result_eligible
            self._write(current, durability="full")
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
            self._write(current, durability="relaxed")
        return deepcopy(current)

    def publish_incident(self, payload: Mapping[str, Any]) -> Mapping[str, str]:
        """Publish an immutable incident whose filename binds its SHA-256."""

        if payload.get("schema_version") != 1:
            raise RegistryError("incident schema_version must equal one")
        if payload.get("artifact_role") != "run_registry_io_incident":
            raise RegistryError("incident artifact_role is invalid")
        run_id = payload.get("run_id")
        incident_id = payload.get("incident_id")
        if not isinstance(run_id, str):
            raise RegistryError("incident run_id is required")
        self._validate_run_id(run_id)
        if (
            not isinstance(incident_id, str)
            or INCIDENT_ID_RE.fullmatch(incident_id) is None
        ):
            raise RegistryError("incident_id has an invalid format")
        _reject_credentials(payload)
        data = (
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
        ).encode("ascii")
        digest = hashlib.sha256(data).hexdigest()
        target = self.incidents / f"{run_id}-{digest}.json"
        with self._lock(run_id):
            if target.exists():
                if self._read_bytes(target, stage="incident_reopen") != data:
                    raise RegistryError("incident receipt hash-path collision")
                self._sync_directory(self.incidents)
                return {
                    "path": str(target),
                    "sha256": digest,
                }
            temporary = self._write_temp_bytes(
                self.incidents,
                f".{target.name}.{os.getpid()}-",
                data,
            )
            try:
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    if self._read_bytes(target, stage="incident_reopen") != data:
                        raise RegistryError(
                            "incident receipt hash-path collision"
                        ) from None
                except OSError as exc:
                    raise _io_error("incident_link", exc) from exc
                self._sync_directory(self.incidents)
                if self._read_bytes(target, stage="incident_reopen") != data:
                    raise RegistryError("incident receipt reopen verification failed")
            finally:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
        return {
            "path": str(target),
            "sha256": digest,
        }

    def active_runs(self) -> list[Mapping[str, Any]]:
        active: list[Mapping[str, Any]] = []
        for path in sorted(self.records.glob("*.json")):
            record = self.read(path.stem)
            if record["status"] == "running":
                active.append(record)
        return active
