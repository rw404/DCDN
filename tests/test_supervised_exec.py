from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dcdn.registry import RunRegistry


def _python_environment(repository_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source = str(repository_root / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source + (os.pathsep + inherited if inherited else "")
    return environment


def _process_exited(pid: int) -> bool:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        fields = stat_path.read_text(encoding="ascii").split()
    except (FileNotFoundError, ProcessLookupError):
        return True
    return len(fields) >= 3 and fields[2] == "Z"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux prctl is required")
def test_parent_sigkill_terminates_supervised_child(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    environment = _python_environment(repository_root)
    ready = tmp_path / "child-ready"
    child_code = (
        "from pathlib import Path; import sys, time; "
        "Path(sys.argv[1]).write_text('ready', encoding='ascii'); "
        "time.sleep(60)"
    )
    launcher_code = (
        "import os, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-m', "
        "'dcdn.supervised_exec', '--expected-parent-pid', str(os.getpid()), "
        "'--', sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
        "print(child.pid, flush=True); time.sleep(60)"
    )
    launcher = subprocess.Popen(
        [sys.executable, "-c", launcher_code, child_code, str(ready)],
        cwd="/tmp",
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid: int | None = None
    try:
        assert launcher.stdout is not None
        child_pid = int(launcher.stdout.readline().strip())
        ready_deadline = time.monotonic() + 5.0
        while not ready.exists() and time.monotonic() < ready_deadline:
            time.sleep(0.02)
        assert ready.read_text(encoding="ascii") == "ready"

        os.kill(launcher.pid, signal.SIGKILL)
        launcher.wait(timeout=5.0)
        exit_deadline = time.monotonic() + 5.0
        while not _process_exited(child_pid) and time.monotonic() < exit_deadline:
            time.sleep(0.02)
        assert _process_exited(child_pid)
    finally:
        if launcher.poll() is None:
            os.killpg(launcher.pid, signal.SIGKILL)
            launcher.wait(timeout=5.0)
        if child_pid is not None and not _process_exited(child_pid):
            os.kill(child_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux prctl is required")
def test_worker_sigkill_prevents_blocked_registry_helper_mutation(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    environment = _python_environment(repository_root)
    registry = RunRegistry(tmp_path / "registry")
    run_id = "helper-parent-death-s42"
    digest = "a" * 64
    registry.create_pending(
        run_id=run_id,
        execution_kind="scientific_tuning",
        seed=42,
        code_commit="b" * 40,
        code_diff_sha256=digest,
        resolved_config_path="configs/resolved/example.yaml",
        resolved_config_sha256=digest,
        data_manifest_sha256=digest,
        features_sha256=digest,
        protocol_sha256=digest,
        reproduction_command_path="runs/commands/example.sh",
        history_path="runs/history/example.jsonl",
        log_path="runs/logs/example.log",
    )
    lock_path = registry.locks / f"{run_id}.lock"
    request = json.dumps(
        {
            "operation": "ensure_running",
            "registry_root": str(registry.root),
            "run_id": run_id,
            "pid": 4242,
            "gpu": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    launcher_code = """
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

lock_descriptor = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o640)
fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
child = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "dcdn.supervised_exec",
        "--expected-parent-pid",
        str(os.getpid()),
        "--",
        sys.executable,
        "-m",
        "dcdn.registry_helper",
    ],
    cwd="/tmp",
    env=os.environ.copy(),
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    start_new_session=True,
)
assert child.stdin is not None
child.stdin.write((sys.argv[2] + "\\n").encode("ascii"))
child.stdin.close()
deadline = time.monotonic() + 5.0
while time.monotonic() < deadline:
    try:
        arguments = (Path("/proc") / str(child.pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        break
    fields = arguments.split(b"\\0")
    if fields[1:3] == [b"-m", b"dcdn.registry_helper"]:
        print(child.pid, flush=True)
        time.sleep(60)
        raise AssertionError("unreachable")
    time.sleep(0.01)
raise RuntimeError("registry helper did not reach exec")
"""
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            launcher_code,
            str(lock_path),
            request,
        ],
        cwd="/tmp",
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    helper_pid: int | None = None
    try:
        assert launcher.stdout is not None
        helper_pid = int(launcher.stdout.readline().strip())
        with lock_path.open("a+b") as lock_handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            os.kill(launcher.pid, signal.SIGKILL)
            launcher.wait(timeout=5.0)
            exit_deadline = time.monotonic() + 5.0
            while not _process_exited(helper_pid) and time.monotonic() < exit_deadline:
                time.sleep(0.02)
            assert _process_exited(helper_pid)
            assert registry.read(run_id)["status"] == "pending"
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    finally:
        if launcher.poll() is None:
            os.killpg(launcher.pid, signal.SIGKILL)
            launcher.wait(timeout=5.0)
        if helper_pid is not None and not _process_exited(helper_pid):
            os.kill(helper_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux prctl is required")
def test_parent_race_fails_before_exec(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    marker = tmp_path / "must-not-exist"
    child_code = (
        "from pathlib import Path; import sys; "
        "Path(sys.argv[1]).write_text('bad', encoding='ascii')"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dcdn.supervised_exec",
            "--expected-parent-pid",
            str(os.getpid() + 1_000_000),
            "--",
            sys.executable,
            "-c",
            child_code,
            str(marker),
        ],
        cwd="/tmp",
        env=_python_environment(repository_root),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    assert completed.returncode == 70
    assert not marker.exists()
