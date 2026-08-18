#!/usr/bin/env python3
"""Deterministic single-GPU DCDN numerical and serialization smoke."""

from __future__ import annotations

import argparse
import hashlib
import json

import torch

from dcdn.core import DCDNCore
from dcdn.reproducibility import seed_everything


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().contiguous().view(torch.uint8).cpu()
    return contiguous.numpy().tobytes()


def _model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _run_once(*, dtype: torch.dtype, seed: int) -> dict[str, object]:
    seed_everything(seed, deterministic=True, warn_only=False, seed_cuda=True)
    device = torch.device("cuda:0")
    mask = [True, False, True, False]
    model = DCDNCore(
        features_count=4,
        num_layers=2,
        denominator_mask=mask,
        div_eps=1e-2,
        runtime_value_checks=dtype == torch.bfloat16,
        feature_schema_sha256="a" * 64,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.tensor(
        [
            [0.005, -2.0, 0.2, 4.0],
            [0.02, 3.0, 2.0, -1.0],
            [1.0, -0.5, 1000.0, 0.0],
            [2.0, 1.5, 0.5, -3.0],
        ],
        device=device,
        dtype=dtype,
    )
    validity = torch.tensor(
        [
            [True, False, True, False],
            [True, False, True, False],
            [True, False, False, False],
            [True, False, True, False],
        ],
        device=device,
        dtype=torch.bool,
    )

    carrier = model.reciprocal_carrier(x, validity)
    if not torch.isfinite(carrier).all():
        raise RuntimeError("reciprocal carrier is non-finite")
    if torch.count_nonzero(carrier[:, 1::2]).item() != 0:
        raise RuntimeError("non-denominator coordinates received a carrier")
    if carrier[2, 2].item() != 0:
        raise RuntimeError("invalid denominator coordinate received a carrier")

    optimizer.zero_grad(set_to_none=True)
    output = model(x, validity)
    loss = output.float().square().mean()
    if not torch.isfinite(output).all() or not torch.isfinite(loss):
        raise RuntimeError("forward pass is non-finite")
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("backward pass is non-finite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("gradient norm is non-finite")
    optimizer.step()

    updated = model(x, validity)
    if not torch.isfinite(updated).all():
        raise RuntimeError("optimizer update produced non-finite output")
    payload = model.state_dict_with_metadata()
    clone = DCDNCore.from_serialization_metadata(payload["metadata"]).to(
        device=device,
        dtype=dtype,
    )
    clone.load_state_dict_with_metadata(payload)
    clone_output = clone(x, validity)
    if not torch.equal(updated, clone_output):
        raise RuntimeError("metadata-aware state round-trip changed the output")

    return {
        "dtype": str(dtype),
        "loss": float(loss.detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "model_sha256": _model_sha256(model),
        "output_sha256": hashlib.sha256(_tensor_bytes(updated)).hexdigest(),
        "metadata_sha256": model.serialization_metadata_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-index", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("launch with exactly one visible GPU via CUDA_VISIBLE_DEVICES")
    if args.physical_index not in (0, 1):
        raise SystemExit("physical-index must be zero or one")

    first = _run_once(dtype=torch.float32, seed=42)
    repeated = _run_once(dtype=torch.float32, seed=42)
    if first != repeated:
        raise RuntimeError("same-seed FP32 GPU smoke is not bitwise deterministic")
    bfloat16 = _run_once(dtype=torch.bfloat16, seed=43)
    payload = {
        "ok": True,
        "physical_index": args.physical_index,
        "visible_device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "fp32": first,
        "bfloat16": bfloat16,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
