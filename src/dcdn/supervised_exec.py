"""Linux parent-death wrapper for one task-owned trainer process."""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import sys
from pathlib import Path
from typing import NoReturn

_PR_SET_PDEATHSIG = 1


class SupervisedExecError(RuntimeError):
    """The trainer could not be bound safely to its worker parent."""


def _arm_parent_death_signal(expected_parent_pid: int) -> None:
    if expected_parent_pid <= 1:
        raise SupervisedExecError("expected parent PID must be greater than one")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(
        _PR_SET_PDEATHSIG,
        signal.SIGKILL,
        0,
        0,
        0,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise SupervisedExecError(
            f"prctl(PR_SET_PDEATHSIG) failed with errno={error_number}"
        )
    if os.getppid() != expected_parent_pid:
        raise SupervisedExecError(
            "worker parent changed before parent-death binding completed"
        )


def supervised_exec(
    expected_parent_pid: int,
    command: list[str],
) -> NoReturn:
    if not command:
        raise SupervisedExecError("trainer command must not be empty")
    executable = Path(command[0])
    if not executable.is_absolute():
        raise SupervisedExecError("trainer executable path must be absolute")
    _arm_parent_death_signal(expected_parent_pid)
    os.execve(str(executable), command, os.environ.copy())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    try:
        supervised_exec(args.expected_parent_pid, command)
    except (OSError, SupervisedExecError) as exc:
        print(f"supervised exec failed: {exc}", file=sys.stderr)
        return 70
    raise AssertionError("os.execve returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
