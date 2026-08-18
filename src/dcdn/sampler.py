"""Stateful deterministic permutation sampler with explicit consumption ACKs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sized
from numbers import Integral
from typing import Any

import torch
from torch.utils.data import Sampler

SAMPLER_STATE_VERSION = 1
SAMPLER_ALGORITHM = "sha256-seeded-torch-randperm-v1"


class SamplerStateError(ValueError):
    """Raised when sampler state is malformed or incompatible."""


def _dataset_size(data_source_or_size: Sized | int) -> int:
    if isinstance(data_source_or_size, bool):
        raise TypeError("data source size must not be boolean")
    if isinstance(data_source_or_size, int):
        size = data_source_or_size
    else:
        size = len(data_source_or_size)
    if size < 0:
        raise ValueError("data source size must be non-negative")
    return int(size)


def _epoch_seed(seed: int, epoch: int) -> int:
    material = f"dcdn-sampler-v1:{seed}:{epoch}".encode("ascii")
    # torch.Generator.manual_seed accepts a signed-64-bit-safe value here.
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


class StatefulPermutationSampler(Sampler[int]):
    """A deterministic sampler whose cursor tracks *consumed* examples.

    ``__iter__`` deliberately does not advance the persistent cursor.  A
    DataLoader may prefetch indices that the optimizer has not consumed yet;
    advancing on yield would then skip rows after a crash.  The training loop
    must call :meth:`commit` after a batch has completed successfully.  Passing
    the actual indices to ``commit`` additionally detects data-order drift.
    """

    def __init__(
        self,
        data_source_or_size: Sized | int,
        *,
        seed: int,
        shuffle: bool = True,
        epoch: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if seed < 0 or seed >= 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if type(shuffle) is not bool:
            raise TypeError("shuffle must be boolean")

        self.dataset_size = _dataset_size(data_source_or_size)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = epoch
        self._permutation = self._make_permutation(epoch)
        self._cursor = 0

    def _make_permutation(self, epoch: int) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(self.dataset_size, dtype=torch.int64)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_epoch_seed(self.seed, epoch))
        return torch.randperm(self.dataset_size, generator=generator, dtype=torch.int64)

    @property
    def cursor(self) -> int:
        """Number of examples acknowledged as consumed in the current epoch."""

        return self._cursor

    @property
    def remaining(self) -> int:
        return self.dataset_size - self._cursor

    @property
    def exhausted(self) -> bool:
        return self._cursor == self.dataset_size

    @property
    def permutation(self) -> torch.Tensor:
        """Return a defensive copy for audit/debug output."""

        return self._permutation.clone()

    def __iter__(self) -> Iterator[int]:
        # Snapshot the committed offset. DataLoader prefetch may pull ahead,
        # while commit() remains the sole authority for durable progress.
        for index in self._permutation[self._cursor :].tolist():
            yield int(index)

    def __len__(self) -> int:
        return self.remaining

    def commit(self, consumed: int | Iterable[int] | torch.Tensor) -> None:
        """Acknowledge a consumed prefix, optionally verifying exact indices."""

        if isinstance(consumed, bool):
            raise TypeError("consumed must be a count or an iterable of indices")
        if isinstance(consumed, int):
            count = consumed
            actual: torch.Tensor | None = None
        else:
            if isinstance(consumed, torch.Tensor):
                if consumed.ndim != 1:
                    raise SamplerStateError(
                        "consumed index tensor must be one-dimensional"
                    )
                if consumed.dtype not in {
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                }:
                    raise SamplerStateError(
                        "consumed index tensor must use an integer dtype"
                    )
                actual = consumed.detach().to(device="cpu", dtype=torch.int64).clone()
            else:
                values = list(consumed)
                if any(
                    isinstance(value, bool) or not isinstance(value, Integral)
                    for value in values
                ):
                    raise SamplerStateError("consumed indices must be integers")
                actual = torch.tensor(
                    [int(value) for value in values], dtype=torch.int64
                )
            count = int(actual.numel())

        if count < 0 or self._cursor + count > self.dataset_size:
            raise SamplerStateError(
                f"cannot commit {count} rows at cursor {self._cursor} "
                f"for dataset_size {self.dataset_size}"
            )
        expected = self._permutation[self._cursor : self._cursor + count]
        if actual is not None and not torch.equal(actual, expected):
            raise SamplerStateError(
                "consumed indices do not match the next permutation prefix"
            )
        self._cursor += count

    def advance_epoch(self) -> None:
        """Start the next deterministic epoch after all rows were consumed."""

        if not self.exhausted:
            raise SamplerStateError(
                "cannot advance epoch before the current permutation is exhausted"
            )
        self.epoch += 1
        self._permutation = self._make_permutation(self.epoch)
        self._cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": SAMPLER_STATE_VERSION,
            "algorithm": SAMPLER_ALGORITHM,
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "cursor": self._cursor,
            "permutation": self._permutation.clone(),
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate checkpoint state without changing this sampler."""

        if not isinstance(state, Mapping):
            raise SamplerStateError("sampler state must be a mapping")
        expected_keys = {
            "version",
            "algorithm",
            "dataset_size",
            "seed",
            "shuffle",
            "epoch",
            "cursor",
            "permutation",
        }
        actual_keys = set(state)
        if actual_keys != expected_keys:
            raise SamplerStateError(
                "sampler state keys mismatch; "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        if (
            type(state["version"]) is not int
            or state["version"] != SAMPLER_STATE_VERSION
        ):
            raise SamplerStateError(
                f"unsupported sampler state version: {state['version']!r}"
            )
        if (
            not isinstance(state["algorithm"], str)
            or state["algorithm"] != SAMPLER_ALGORITHM
        ):
            raise SamplerStateError("sampler permutation algorithm mismatch")
        if isinstance(state["dataset_size"], bool) or not isinstance(
            state["dataset_size"], int
        ):
            raise SamplerStateError("sampler dataset_size must be an integer")
        if state["dataset_size"] != self.dataset_size:
            raise SamplerStateError(
                "sampler dataset_size mismatch: "
                f"checkpoint={state['dataset_size']!r}, current={self.dataset_size}"
            )
        if isinstance(state["seed"], bool) or not isinstance(state["seed"], int):
            raise SamplerStateError("sampler seed must be an integer")
        if state["seed"] != self.seed:
            raise SamplerStateError(
                "sampler seed mismatch: "
                f"checkpoint={state['seed']!r}, current={self.seed}"
            )
        if type(state["shuffle"]) is not bool or state["shuffle"] is not self.shuffle:
            raise SamplerStateError("sampler shuffle policy mismatch")

        epoch = state["epoch"]
        cursor = state["cursor"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise SamplerStateError("sampler epoch must be a non-negative integer")
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or cursor < 0
            or cursor > self.dataset_size
        ):
            raise SamplerStateError("sampler cursor is outside the dataset")

        permutation = state["permutation"]
        if (
            not isinstance(permutation, torch.Tensor)
            or permutation.device.type != "cpu"
            or permutation.dtype != torch.int64
            or permutation.shape != (self.dataset_size,)
        ):
            raise SamplerStateError(
                "sampler permutation must be a one-dimensional CPU int64 tensor "
                "matching dataset_size"
            )
        expected_permutation = self._make_permutation(epoch)
        if not torch.equal(permutation, expected_permutation):
            raise SamplerStateError(
                "sampler permutation does not match the deterministic seed/epoch"
            )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.validate_state_dict(state)
        # Mutation starts only after the complete state has passed validation.
        self.epoch = int(state["epoch"])
        self._cursor = int(state["cursor"])
        self._permutation = state["permutation"].clone()
