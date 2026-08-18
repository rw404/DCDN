#!/usr/bin/env python3
"""Read-only local preflight for the frozen DCDN experiment environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from cost_guard import evaluate_cost
from storage_guard import inspect_storage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EPHEMERAL_ROOTS = (Path("/tmp"), Path("/home/codex/.cache"))
BASE_VERSIONS = {
    "numpy": "2.1.2",
    "psutil": "7.1.0",
    "PyYAML": "6.0.3",
    "setuptools": "80.9.0",
    "torch": "2.8.0",
}
DEV_VERSIONS = {
    "hypothesis": "6.161.6",
    "pytest": "9.1.1",
    "ruff": "0.12.12",
}
DATA_VERSIONS = {
    "implicit": "0.7.3",
    "polars": "1.43.0",
    "pyarrow": "25.0.0",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
}
EPHEMERAL_ENV_VARS = (
    "UV_PROJECT_ENVIRONMENT",
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "DCDN_DATA_CACHE",
)
TELEMETRY_ENV = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_DATASETS_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
    "UV_KEYRING_PROVIDER": "disabled",
    "UV_PYTHON_DOWNLOADS": "never",
}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def _is_below(path: Path, roots: tuple[Path, ...]) -> bool:
    resolved = path.expanduser().resolve()
    return any(resolved == root or resolved.is_relative_to(root) for root in roots)


def _check_python() -> Check:
    version = sys.version_info
    ok = version.major == 3 and version.minor == 12
    return Check("python", ok, f"{version.major}.{version.minor}.{version.micro}")


def _check_environment_paths() -> list[Check]:
    checks: list[Check] = []
    for name in EPHEMERAL_ENV_VARS:
        value = os.environ.get(name)
        ok = bool(value) and _is_below(Path(value), ALLOWED_EPHEMERAL_ROOTS)
        detail = value if value else "unset; source scripts/env.sh"
        checks.append(Check(f"ephemeral_path:{name}", ok, detail))

    expected_venv = os.environ.get("UV_PROJECT_ENVIRONMENT")
    active = Path(sys.prefix).resolve()
    active_ok = bool(expected_venv) and active == Path(expected_venv).resolve()
    checks.append(Check("active_isolated_environment", active_ok, str(active)))
    return checks


def _check_telemetry() -> list[Check]:
    return [
        Check(
            f"telemetry_disabled:{name}",
            os.environ.get(name, "").lower() == expected,
            "set" if os.environ.get(name) is not None else "unset",
        )
        for name, expected in TELEMETRY_ENV.items()
    ]


def _version_matches(distribution: str, actual: str, expected: str) -> bool:
    if distribution == "torch":
        return actual == expected or actual.startswith(f"{expected}+")
    return actual == expected


def _check_distributions(require_data: bool) -> list[Check]:
    expected = BASE_VERSIONS | DEV_VERSIONS
    if require_data:
        expected |= DATA_VERSIONS
    checks: list[Check] = []
    for distribution, pinned in expected.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            checks.append(Check(f"distribution:{distribution}", False, "not installed"))
            continue
        checks.append(
            Check(
                f"distribution:{distribution}",
                _version_matches(distribution, actual, pinned),
                f"installed={actual}; pinned={pinned}",
            )
        )
    return checks


def _check_lock() -> list[Check]:
    lock_path = PROJECT_ROOT / "uv.lock"
    checks = [Check("persistent_lock_present", lock_path.is_file(), str(lock_path))]
    if not lock_path.is_file():
        return checks
    environment = os.environ.copy()
    environment["UV_OFFLINE"] = "1"
    result = subprocess.run(
        ["uv", "lock", "--check", "--offline"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    detail = (result.stderr or result.stdout or "lock is current").strip()
    checks.append(Check("uv_lock_current_offline", result.returncode == 0, detail))
    return checks


def _check_gpu() -> list[Check]:
    import torch

    checks = [
        Check(
            "torch_cuda_build", torch.version.cuda == "12.8", str(torch.version.cuda)
        ),
        Check(
            "cuda_available",
            torch.cuda.is_available(),
            str(torch.cuda.is_available()),
        ),
    ]
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    checks.append(Check("two_gpus", count == 2, f"count={count}; names={names}"))
    checks.append(
        Check(
            "a100_gpus",
            count == 2 and all("A100" in name for name in names),
            str(names),
        )
    )
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        checks.append(Check("nvlink_visible", False, type(error).__name__))
    else:
        gpu_rows = [
            line
            for line in result.stdout.splitlines()
            if line.lstrip().startswith("GPU")
        ]
        nvlink = result.returncode == 0 and any("NV" in line for line in gpu_rows[:2])
        checks.append(Check("nvlink_visible", nvlink, f"gpu_rows={gpu_rows[:2]}"))
    return checks


def run_preflight(
    *,
    eta_hours: Decimal,
    phase: str,
    work_class: str,
    require_data: bool,
    skip_gpu: bool,
) -> tuple[list[Check], dict[str, object], dict[str, object]]:
    checks = [_check_python()]
    checks.extend(_check_environment_paths())
    checks.extend(_check_telemetry())
    checks.extend(_check_distributions(require_data))
    checks.extend(_check_lock())

    cost = evaluate_cost(
        datetime.now(UTC),
        eta_hours=eta_hours,
        phase=phase,
        work_class=work_class,
    ).as_dict()
    checks.append(
        Check(
            "cost_and_time_guard",
            bool(cost["launch_allowed"]),
            f"status={cost['status']}; reasons={cost['reasons']}",
        )
    )

    storage = inspect_storage(PROJECT_ROOT, Path("/"))
    checks.append(
        Check(
            "storage_guard",
            bool(storage["ok"]),
            f"violations={storage['violations']}",
        )
    )
    persistent_mount = Path(str(storage["persistent_mount_point"]))
    repository_is_persistent = (
        persistent_mount != Path("/")
        and PROJECT_ROOT.is_relative_to(persistent_mount)
        and storage["persistent_fs_type"] != "overlay"
    )
    checks.append(
        Check(
            "repository_on_persistent_mount",
            repository_is_persistent,
            (f"mount={persistent_mount}; fs_type={storage['persistent_fs_type']}"),
        )
    )
    if not skip_gpu:
        checks.extend(_check_gpu())
    return checks, cost, storage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta-hours", type=Decimal, default=Decimal(0))
    parser.add_argument(
        "--phase", choices=("compute", "aggregation"), default="compute"
    )
    parser.add_argument(
        "--work-class", choices=("mandatory", "optional"), default="mandatory"
    )
    parser.add_argument("--require-data-extra", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checks, cost, storage = run_preflight(
        eta_hours=args.eta_hours,
        phase=args.phase,
        work_class=args.work_class,
        require_data=args.require_data_extra,
        skip_gpu=args.skip_gpu,
    )
    ok = all(check.ok for check in checks)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "checks": [check.as_dict() for check in checks],
                    "cost": cost,
                    "storage": storage,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for check in checks:
            verdict = "PASS" if check.ok else "FAIL"
            print(f"{verdict:4} {check.name}: {check.detail}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
