"""One-shot bounded registry I/O helper for the process supervisor."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dcdn.registry import (
    TRANSIENT_IO_ERRNOS,
    RegistryError,
    RegistryIOError,
    RunRegistry,
)

MAX_REQUEST_BYTES = 64 * 1024


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{field} must be a mapping")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegistryError(f"{field} must be a positive integer")
    return value


def execute_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
    root_value = request.get("registry_root")
    if not isinstance(root_value, str) or not root_value:
        raise RegistryError("registry_root must be a non-empty path")
    registry = RunRegistry(Path(root_value).resolve(strict=True))
    operation = request.get("operation")
    run_id = request.get("run_id")
    if not isinstance(run_id, str):
        raise RegistryError("run_id must be a string")

    result: dict[str, Any]
    if operation == "heartbeat":
        record = registry.heartbeat(
            run_id,
            pid=_positive_integer(request.get("pid"), "pid"),
        )
        result = {
            "status": record["status"],
            "heartbeat_utc": record["heartbeat_utc"],
        }
    elif operation == "ensure_running":
        gpu = request.get("gpu")
        if isinstance(gpu, bool) or not isinstance(gpu, int):
            raise RegistryError("gpu must be an integer")
        record = registry.ensure_running(
            run_id,
            pid=_positive_integer(request.get("pid"), "pid"),
            gpu=gpu,
        )
        result = {"status": record["status"]}
    elif operation == "ensure_terminal":
        status = request.get("status")
        if status not in {"completed", "failed"}:
            raise RegistryError("terminal status is invalid")
        reason = request.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise RegistryError("terminal reason must be non-empty")
        eligibility = request.get("scientific_result_eligible")
        if eligibility is not None and type(eligibility) is not bool:
            raise RegistryError("scientific_result_eligible must be boolean or null")
        pid = request.get("pid")
        gpu = request.get("gpu")
        if pid is not None:
            pid = _positive_integer(pid, "pid")
        if gpu is not None and (isinstance(gpu, bool) or not isinstance(gpu, int)):
            raise RegistryError("gpu must be an integer or null")
        record = registry.ensure_terminal(
            run_id,
            status,
            pid=pid,
            gpu=gpu,
            reason=reason,
            scientific_result_eligible=eligibility,
        )
        result = {"status": record["status"]}
    elif operation == "publish_incident":
        incident = _mapping(request.get("incident"), "incident")
        result = {"incident": dict(registry.publish_incident(incident))}
    else:
        raise RegistryError("registry helper operation is invalid")

    incident_value = request.get("incident")
    if operation != "publish_incident" and incident_value is not None:
        incident = _mapping(incident_value, "incident")
        result["incident"] = dict(registry.publish_incident(incident))
    return result


def _response_for_error(error: Exception) -> Mapping[str, Any]:
    if isinstance(error, RegistryIOError):
        return {
            "ok": False,
            "error_type": type(error).__name__,
            "message": str(error)[:500],
            "stage": error.stage,
            "errno": error.errno,
            "transient_candidate": error.transient_candidate,
        }
    if isinstance(error, OSError):
        return {
            "ok": False,
            "error_type": type(error).__name__,
            "message": str(error)[:500],
            "stage": "helper_unwrapped_oserror",
            "errno": error.errno,
            "transient_candidate": error.errno in TRANSIENT_IO_ERRNOS,
        }
    return {
        "ok": False,
        "error_type": type(error).__name__,
        "message": str(error)[:500],
        "stage": None,
        "errno": None,
        "transient_candidate": False,
    }


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise RegistryError("registry helper request is too large")
        request = json.loads(raw.decode("ascii"))
        result = execute_request(_mapping(request, "request"))
    except (RegistryError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        response = _response_for_error(exc)
        return_code = 2
    else:
        response = {"ok": True, "result": result}
        return_code = 0
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
