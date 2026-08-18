"""Deterministic process and RNG-state utilities for resumable experiments.

The checkpoint layer restores this state only after every model/training
component has loaded successfully.  CUDA state capture is enabled by default
for real runs; CPU-only tests can explicitly disable it without initializing a
CUDA context.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

RNG_STATE_VERSION = 1


class RNGStateError(ValueError):
    """Raised when a captured RNG state is malformed or incompatible."""


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
    seed_cuda: bool = True,
) -> None:
    """Seed Python, NumPy, PyTorch, and optionally all visible CUDA devices.

    ``PYTHONHASHSEED`` is recorded for child processes.  Python's hash seed for
    the already-running interpreter is fixed at process start and cannot be
    retroactively changed.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed >= 2**63:
        raise ValueError("seed must be in [0, 2**63)")

    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Must be present before CUDA deterministic matrix multiplication is
        # first used; setdefault preserves an explicit owner configuration.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed % 2**32)
    # Seed the CPU generator directly. ``torch.manual_seed`` also schedules
    # CUDA seeding, which would violate an explicitly CPU-only caller's intent.
    torch.random.default_generator.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            torch.backends.cudnn.benchmark = False


def capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    """Capture RNG and deterministic-backend state using safe primitives.

    NumPy's key array is stored as a Torch tensor so the full checkpoint can be
    deserialized with ``torch.load(..., weights_only=True)``.
    """

    numpy_state = np.random.get_state()
    cuda_states: list[torch.Tensor] = []
    cuda_device_count = 0
    if include_cuda and torch.cuda.is_available():
        cuda_device_count = torch.cuda.device_count()
        cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]

    warn_only_getter = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    deterministic_warn_only = (
        bool(warn_only_getter()) if warn_only_getter is not None else False
    )

    return {
        "version": RNG_STATE_VERSION,
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "cuda_included": bool(include_cuda),
        "cuda_device_count": int(cuda_device_count),
        "torch_cuda": cuda_states,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": deterministic_warn_only,
        "cudnn_deterministic": bool(
            getattr(torch.backends.cudnn, "deterministic", False)
        ),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
    }


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RNGStateError(f"{label} keys mismatch; missing={missing}, extra={extra}")


def validate_rng_state(state: Mapping[str, Any]) -> None:
    """Validate a captured RNG state without mutating global RNGs."""

    if not isinstance(state, Mapping):
        raise RNGStateError("RNG state must be a mapping")
    expected_keys = {
        "version",
        "python",
        "numpy",
        "torch_cpu",
        "cuda_included",
        "cuda_device_count",
        "torch_cuda",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
    }
    _require_exact_keys(state, expected_keys, label="RNG state")
    if type(state["version"]) is not int or state["version"] != RNG_STATE_VERSION:
        raise RNGStateError(f"unsupported RNG state version: {state['version']!r}")

    # Validate Python state against an isolated generator.
    try:
        probe = random.Random()
        probe.setstate(state["python"])
    except (TypeError, ValueError) as exc:
        raise RNGStateError("invalid Python RNG state") from exc

    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise RNGStateError("NumPy RNG state must be a mapping")
    numpy_keys = {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    _require_exact_keys(numpy_state, numpy_keys, label="NumPy RNG state")
    keys = numpy_state["keys"]
    if not isinstance(keys, torch.Tensor):
        raise RNGStateError("NumPy RNG keys must be a tensor")
    if keys.device.type != "cpu" or keys.dtype != torch.uint32 or keys.ndim != 1:
        raise RNGStateError(
            "NumPy RNG keys must be a one-dimensional CPU uint32 tensor"
        )
    try:
        probe_numpy = np.random.RandomState()
        probe_numpy.set_state(
            (
                numpy_state["bit_generator"],
                keys.numpy().copy(),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
    except (TypeError, ValueError) as exc:
        raise RNGStateError("invalid NumPy RNG state") from exc

    torch_cpu = state["torch_cpu"]
    if (
        not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.device.type != "cpu"
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
    ):
        raise RNGStateError(
            "Torch CPU RNG state must be a one-dimensional CPU byte tensor"
        )
    try:
        probe_torch = torch.Generator(device="cpu")
        probe_torch.set_state(torch_cpu)
    except RuntimeError as exc:
        raise RNGStateError("invalid Torch CPU RNG state") from exc

    boolean_fields = (
        "cuda_included",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
    )
    if any(type(state[field]) is not bool for field in boolean_fields):
        raise RNGStateError("RNG boolean settings must contain booleans")

    cuda_device_count = state["cuda_device_count"]
    if isinstance(cuda_device_count, bool) or not isinstance(cuda_device_count, int):
        raise RNGStateError("cuda_device_count must be an integer")
    if cuda_device_count < 0:
        raise RNGStateError("cuda_device_count must be non-negative")
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list) or len(cuda_states) != cuda_device_count:
        raise RNGStateError("CUDA RNG state count does not match cuda_device_count")
    if not state["cuda_included"] and (cuda_device_count != 0 or cuda_states):
        raise RNGStateError("CUDA RNG states are present despite cuda_included=false")
    for cuda_state in cuda_states:
        if (
            not isinstance(cuda_state, torch.Tensor)
            or cuda_state.device.type != "cpu"
            or cuda_state.dtype != torch.uint8
            or cuda_state.ndim != 1
        ):
            raise RNGStateError(
                "each CUDA RNG state must be a one-dimensional CPU byte tensor"
            )

    if state["cuda_included"]:
        current_device_count = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        if current_device_count != cuda_device_count:
            raise RNGStateError(
                "CUDA device count mismatch: "
                f"checkpoint={cuda_device_count}, current={current_device_count}"
            )


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a previously captured state after validating it in full."""

    validate_rng_state(state)
    numpy_state = state["numpy"]

    # CUDA compatibility was checked above.  Restore it before CPU/Python RNGs
    # so an unexpected backend error cannot leave the primary RNGs advanced.
    cuda_states = state["torch_cuda"]
    if cuda_states:
        torch.cuda.set_rng_state_all([value.clone() for value in cuda_states])

    torch.use_deterministic_algorithms(
        state["deterministic_algorithms"],
        warn_only=state["deterministic_warn_only"],
    )
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = state["cudnn_benchmark"]

    random.setstate(state["python"])
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["keys"].numpy().copy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"].clone())
