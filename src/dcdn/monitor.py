"""Non-blocking numerical monitoring for optimized DCDN training.

CUDA observations are reduced on a dedicated stream and copied into pinned
host memory.  ``poll`` only consumes events whose copy has already completed;
it never synchronizes the training stream.  ``drain`` is reserved for durable
checkpoint and shutdown boundaries, where proving that no queued observation
is non-finite is more important than latency.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

ObservationKind = Literal["activation", "loss", "gradient"]


class NumericalMonitorError(FloatingPointError):
    """Raised after the frozen non-finite stop threshold is reached."""


@dataclass(frozen=True)
class NumericalObservation:
    sequence: int
    name: str
    kind: ObservationKind
    nonfinite: bool


@dataclass(frozen=True)
class NumericalSummary:
    submitted: int
    completed: int
    pending: int
    nonfinite_total: int
    nonfinite_by_kind: dict[str, int]
    asynchronous_cuda: bool
    activation_hook_count: int
    activation_hook_observations: int


@dataclass
class _PendingCUDAObservation:
    sequence: int
    name: str
    kind: ObservationKind
    host_flag: torch.Tensor
    event: torch.cuda.Event


def _tensor_outputs(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, tuple | list):
        for child in value:
            yield from _tensor_outputs(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensor_outputs(child)


class AsyncNonFiniteMonitor:
    """Observe activations, losses, and gradients without per-step CUDA sync.

    ``nonfinite_limit`` is deliberately explicit.  The production harness uses
    one (stop on the first completed bad observation); a caller may set two to
    implement the less conservative repeated-observation policy.  No optimizer
    checkpoint is declared durable until :meth:`drain` succeeds.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        nonfinite_limit: int = 1,
    ) -> None:
        self.device = torch.device(device)
        if isinstance(nonfinite_limit, bool) or not isinstance(nonfinite_limit, int):
            raise TypeError("nonfinite_limit must be an integer")
        if nonfinite_limit <= 0:
            raise ValueError("nonfinite_limit must be positive")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("numerical monitor supports only CPU or CUDA")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA monitoring requested but CUDA is unavailable")

        self.nonfinite_limit = nonfinite_limit
        self._stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self._pending: deque[_PendingCUDAObservation] = deque()
        self._completed: deque[NumericalObservation] = deque()
        self._submitted = 0
        self._completed_count = 0
        self._nonfinite_total = 0
        self._nonfinite_by_kind: Counter[str] = Counter()
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._activation_hook_count = 0
        self._activation_hook_observations = 0

    @property
    def asynchronous_cuda(self) -> bool:
        return self._stream is not None

    def _validate_tensor(
        self, name: str, kind: ObservationKind, value: Any
    ) -> torch.Tensor:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("observation name must be non-empty")
        if kind not in {"activation", "loss", "gradient"}:
            raise ValueError(f"unknown observation kind: {kind!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError("observed value must be a tensor")
        if value.device.type != self.device.type:
            raise ValueError("observed tensor is on the wrong device type")
        if self.device.type == "cuda" and value.device != self.device:
            raise ValueError("observed tensor is on the wrong CUDA device")
        if not value.is_floating_point() and not value.is_complex():
            raise TypeError("numerical monitor accepts floating or complex tensors")
        return value

    def submit(
        self,
        name: str,
        value: torch.Tensor,
        *,
        kind: ObservationKind,
    ) -> int:
        """Queue one finite check and return its monotonically increasing ID."""

        tensor = self._validate_tensor(name, kind, value)
        sequence = self._submitted
        self._submitted += 1
        if self._stream is None:
            nonfinite = bool((~torch.isfinite(tensor.detach())).any().item())
            self._record(NumericalObservation(sequence, name.strip(), kind, nonfinite))
            return sequence

        producer = torch.cuda.current_stream(self.device)
        host_flag = torch.empty((), dtype=torch.uint8, device="cpu", pin_memory=True)
        event = torch.cuda.Event(enable_timing=False, blocking=False)
        with torch.cuda.stream(self._stream):
            self._stream.wait_stream(producer)
            device_flag = (~torch.isfinite(tensor.detach())).any().to(torch.uint8)
            host_flag.copy_(device_flag, non_blocking=True)
            event.record(self._stream)
        self._pending.append(
            _PendingCUDAObservation(
                sequence=sequence,
                name=name.strip(),
                kind=kind,
                host_flag=host_flag,
                event=event,
            )
        )
        return sequence

    def _record(self, observation: NumericalObservation) -> None:
        self._completed_count += 1
        if observation.nonfinite:
            self._nonfinite_total += 1
            self._nonfinite_by_kind[observation.kind] += 1
        self._completed.append(observation)

    def poll(self) -> tuple[NumericalObservation, ...]:
        """Consume only ready CUDA events and return new observations."""

        while self._pending and self._pending[0].event.query():
            pending = self._pending.popleft()
            self._record(
                NumericalObservation(
                    pending.sequence,
                    pending.name,
                    pending.kind,
                    bool(pending.host_flag.item()),
                )
            )
        observations = tuple(self._completed)
        self._completed.clear()
        return observations

    def assert_healthy(self) -> tuple[NumericalObservation, ...]:
        observations = self.poll()
        if self._nonfinite_total >= self.nonfinite_limit:
            bad = [item for item in observations if item.nonfinite]
            rendered = ", ".join(
                f"{item.kind}:{item.name}#{item.sequence}" for item in bad[-8:]
            )
            raise NumericalMonitorError(
                "non-finite observation threshold reached"
                + (f": {rendered}" if rendered else "")
            )
        return observations

    def drain(self) -> tuple[NumericalObservation, ...]:
        """Finish queued checks at a checkpoint/shutdown boundary."""

        if self._stream is not None and self._pending:
            self._stream.synchronize()
        return self.assert_healthy()

    def observe_gradients(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.submit(name, parameter.grad, kind="gradient")

    def attach_activation_hooks(
        self,
        model: nn.Module,
        *,
        select: Callable[[str, nn.Module], bool],
        expected_count: int | None = None,
    ) -> int:
        """Attach hooks to an explicit frozen subset of named modules."""

        if self._hook_handles:
            raise RuntimeError("activation hooks are already attached")
        if expected_count is not None and (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count <= 0
        ):
            raise ValueError("expected_count must be a positive integer")
        for module_name, module in model.named_modules():
            if not module_name or not select(module_name, module):
                continue

            def hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                name: str = module_name,
            ) -> None:
                for index, tensor in enumerate(_tensor_outputs(output)):
                    self._activation_hook_observations += 1
                    self.submit(f"{name}[{index}]", tensor, kind="activation")

            self._hook_handles.append(module.register_forward_hook(hook))
        self._activation_hook_count = len(self._hook_handles)
        if self._activation_hook_count == 0:
            raise RuntimeError("activation hook selector matched no modules")
        if expected_count is not None and self._activation_hook_count != expected_count:
            self.close_hooks()
            raise RuntimeError(
                "activation hook count mismatch: "
                f"expected={expected_count}, actual={self._activation_hook_count}"
            )
        return self._activation_hook_count

    @property
    def activation_hook_count(self) -> int:
        return self._activation_hook_count

    @property
    def activation_hook_observations(self) -> int:
        return self._activation_hook_observations

    def assert_activation_hooks_observed(self, previous: int = 0) -> int:
        if isinstance(previous, bool) or not isinstance(previous, int) or previous < 0:
            raise ValueError("previous hook observation count is invalid")
        if self._activation_hook_count <= 0:
            raise NumericalMonitorError("no activation hooks are attached")
        if self._activation_hook_observations <= previous:
            raise NumericalMonitorError(
                "model forward produced no expected activation-hook observations"
            )
        return self._activation_hook_observations

    def close_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def validate_summary(self, value: Mapping[str, Any]) -> None:
        """Validate saved counters without mutating live monitor state."""

        self._validate_summary(value, apply=False)

    def restore_summary(self, value: Mapping[str, Any]) -> None:
        """Restore drained counters captured in an integrity-checked checkpoint."""

        self.drain()
        self._validate_summary(value, apply=True)

    def _validate_summary(self, value: Mapping[str, Any], *, apply: bool) -> None:
        if not isinstance(value, Mapping):
            raise NumericalMonitorError(
                "checkpoint numerical summary must be a mapping"
            )
        expected = {
            "submitted",
            "completed",
            "pending",
            "nonfinite_total",
            "nonfinite_by_kind",
            "asynchronous_cuda",
            "activation_hook_count",
            "activation_hook_observations",
        }
        if set(value) != expected:
            raise NumericalMonitorError("checkpoint numerical summary keys mismatch")

        counters: dict[str, int] = {}
        for name in (
            "submitted",
            "completed",
            "pending",
            "nonfinite_total",
            "activation_hook_count",
            "activation_hook_observations",
        ):
            item = value[name]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise NumericalMonitorError(
                    f"checkpoint numerical counter {name} is invalid"
                )
            counters[name] = item
        if counters["pending"] != 0:
            raise NumericalMonitorError(
                "a durable checkpoint cannot contain pending numerical checks"
            )
        if counters["submitted"] != counters["completed"]:
            raise NumericalMonitorError(
                "checkpoint submitted/completed numerical counters differ"
            )

        by_kind_value = value["nonfinite_by_kind"]
        if not isinstance(by_kind_value, Mapping):
            raise NumericalMonitorError(
                "checkpoint nonfinite_by_kind must be a mapping"
            )
        by_kind: dict[str, int] = {}
        for name, count in by_kind_value.items():
            if name not in {"activation", "loss", "gradient"}:
                raise NumericalMonitorError("checkpoint numerical kind is invalid")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise NumericalMonitorError("checkpoint non-finite count is invalid")
            by_kind[name] = count
        if sum(by_kind.values()) != counters["nonfinite_total"]:
            raise NumericalMonitorError(
                "checkpoint non-finite numerical counters are inconsistent"
            )
        if counters["nonfinite_total"] > counters["submitted"]:
            raise NumericalMonitorError(
                "checkpoint has more non-finite than submitted observations"
            )
        if counters["activation_hook_count"] != self._activation_hook_count:
            raise NumericalMonitorError(
                "checkpoint activation hook count differs from the live model"
            )
        if counters["activation_hook_observations"] > counters["submitted"]:
            raise NumericalMonitorError(
                "checkpoint has more hook observations than submitted checks"
            )
        if counters["nonfinite_total"] >= self.nonfinite_limit:
            raise NumericalMonitorError(
                "checkpoint already reached the non-finite stop threshold"
            )
        asynchronous_cuda = value["asynchronous_cuda"]
        if (
            type(asynchronous_cuda) is not bool
            or asynchronous_cuda != self.asynchronous_cuda
        ):
            raise NumericalMonitorError(
                "checkpoint numerical-monitor device mode mismatch"
            )

        if not apply:
            return
        self._submitted = counters["submitted"]
        self._completed_count = counters["completed"]
        self._nonfinite_total = counters["nonfinite_total"]
        self._nonfinite_by_kind = Counter(by_kind)
        self._activation_hook_observations = counters["activation_hook_observations"]

    def summary(self) -> NumericalSummary:
        return NumericalSummary(
            submitted=self._submitted,
            completed=self._completed_count,
            pending=len(self._pending),
            nonfinite_total=self._nonfinite_total,
            nonfinite_by_kind=dict(sorted(self._nonfinite_by_kind.items())),
            asynchronous_cuda=self.asynchronous_cuda,
            activation_hook_count=self._activation_hook_count,
            activation_hook_observations=self._activation_hook_observations,
        )

    def __enter__(self) -> AsyncNonFiniteMonitor:
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            self.drain()
        finally:
            self.close_hooks()
