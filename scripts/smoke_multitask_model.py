#!/usr/bin/env python3
"""One-effective-batch VK seven-head implementation smoke.

This runner deliberately has no validation or test source path.  It exercises
the exact raw-cache -> shared encoder -> interaction -> seven-head objective ->
optimizer boundary and publishes only non-scientific implementation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import torch
from cost_guard import evaluate_cost
from storage_guard import inspect_storage
from train_model import _disable_telemetry, _write_immutable_json

from dcdn.data.vk_lsvd_multitask_runtime import (
    VKLSVDMultiTargetSource,
    task_constants_from_cache_metadata,
)
from dcdn.models.cross import DCNv2FullRankConfig
from dcdn.models.dcdn import DCDNCoreConfig
from dcdn.multitask import TaskConstants, macro_nloss
from dcdn.multitask_contract import MultiTargetEncoderLock
from dcdn.multitask_model import build_multitarget_encoded_model
from dcdn.protocol import ProtocolGate, read_verified_artifact
from dcdn.training import runtime_identity_from_raw

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/home/codex/.cache/dcdn-public/runtime/vk_lsvd/multitask_v2/c_plus_p")
ENCODER = ROOT / "configs/encoder_multitask_v2.yaml"
CAPACITY = ROOT / "configs/capacity/vk_lsvd_multitask_c_plus_p_v2.json"
PROTOCOL = ROOT / "configs/protocols/vk_lsvd_multitask_c_plus_p/protocol_frozen.yaml"
ACTIVE = (
    ROOT / "artifacts/protocol_state/vk_lsvd_multitask_c_plus_p/active_protocol.yaml"
)
EXPECTED_CACHE_METADATA = (
    "f9cb51ae80d4ce45b532ef2f1f10877c216d85eeab34d5dc1d40db223e6cc293"
)
EXPECTED_CACHE_CONTENT = (
    "72834709c5c1735b8369571cbade96cb7b0f705d2bccaeb0335e444dd49044b8"
)
EXPECTED_SCAN = "e67f72187f73ed9ad37e1a23cdd5c8b679d1fbc3be35a2781968c270a1497bb6"
BATCH = 4096
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke requires exactly one visible CUDA device")
    return torch.device("cuda:0")


def _interaction(model_key: str):
    if model_key == "dcdn_core":
        return DCDNCoreConfig(
            num_layers=2,
            div_eps=0.01,
            product_init_scale=0.1,
            div_init_scale=0.1,
            runtime_value_checks=True,
        )
    if model_key == "dcnv2_full_rank":
        return DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1)
    raise ValueError("smoke model must be dcdn_core or dcnv2_full_rank")


def _load_constants() -> tuple[TaskConstants, str]:
    artifact = read_verified_artifact(CACHE / "cache.json")
    if artifact.digest != EXPECTED_CACHE_METADATA:
        raise RuntimeError("cache metadata digest drifted")
    payload = json.loads(artifact.data)
    constants = task_constants_from_cache_metadata(payload)
    return constants, artifact.digest


def run(model_key: str, output: Path, run_id: str, resolved_config_sha256: str) -> int:
    _disable_telemetry()
    if not output.is_relative_to(ROOT):
        raise RuntimeError("smoke output must be inside the repository")
    if output.exists() or output.with_name(f"{output.name}.sha256").exists():
        raise RuntimeError("immutable smoke output already exists")
    if not run_id or "/" in run_id or ".." in run_id:
        raise RuntimeError("smoke run_id is invalid")
    if _SHA256_RE.fullmatch(resolved_config_sha256) is None:
        raise RuntimeError("resolved config digest is invalid")
    cost = evaluate_cost(
        datetime.now(UTC),
        eta_hours=Decimal("0.1"),
        phase="compute",
        work_class="mandatory",
    ).as_dict()
    storage = inspect_storage(ROOT, Path("/home/codex/.cache/dcdn-public"))
    if cost["launch_allowed"] is not True or storage["ok"] is not True:
        raise RuntimeError("local cost/storage guard blocked smoke")
    gate = ProtocolGate.load(PROTOCOL, active_registry_path=ACTIVE)
    gate.require_active()
    if gate.test_unlocked or gate.payload["test_path_available"]:
        raise RuntimeError("smoke requires a sealed test")
    encoder_lock = MultiTargetEncoderLock.load(ENCODER)
    capacity = read_verified_artifact(CAPACITY)
    if capacity.digest != gate.payload["hashes"]["capacity_resolution_sha256"]:
        raise RuntimeError("capacity artifact differs from Protocol")
    constants, cache_metadata_sha256 = _load_constants()
    source = VKLSVDMultiTargetSource(
        CACHE,
        split="train",
        expected_cache_metadata_sha256=EXPECTED_CACHE_METADATA,
        expected_cache_content_sha256=EXPECTED_CACHE_CONTENT,
        expected_scan_sha256=EXPECTED_SCAN,
        runtime_value_checks=True,
        verify_array_hashes=False,
    )
    try:
        encoder_spec = encoder_lock.resolve(source.identity)
        runtime_identity = runtime_identity_from_raw(
            source.identity,
            encoder_contract_sha256=encoder_lock.sha256,
            capacity_resolution_sha256=capacity.digest,
        )
        gate.require_split("train", runtime_identity=runtime_identity)
        if model_key not in encoder_spec.allowed_model_keys:
            raise RuntimeError("model is not allowed by the encoder lock")

        device = _device()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = build_multitarget_encoded_model(
            encoder_spec=encoder_spec,
            interaction_config=_interaction(model_key),
            target_spec=constants.spec,
            hidden_dims=(),
            dropout=0.0,
            run_seed=42,
            runtime_value_checks=True,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0003,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-6,
        )
        batch = source.batch(range(BATCH), device=device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_values = model(batch)
            objective = macro_nloss(
                constants.spec,
                constants,
                output_values.logits,
                batch.targets,
                batch.target_masks,
            )
        loss = objective.aggregate.float()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("smoke loss is non-finite")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all()
            ):
                raise RuntimeError(f"non-finite gradient: {name}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("gradient norm is non-finite")
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if not bool(torch.isfinite(output_values.logits).all()):
            raise RuntimeError("smoke logits are non-finite")
        receipt = {
            "schema_version": 5,
            "artifact_role": "vk_lsvd_seven_head_implementation_smoke",
            "execution_kind": "implementation_smoke",
            "scientific_result_eligible": False,
            "run_id": run_id,
            "resolved_config_sha256": resolved_config_sha256,
            "model_key": model_key,
            "feature_regime": "C+P",
            "seed": 42,
            "examples_seen": BATCH,
            "optimizer_steps": 1,
            "validation_checks": 0,
            "test_metrics": None,
            "training_loss_non_scientific": float(loss.detach().cpu()),
            "elapsed_seconds": elapsed,
            "throughput_examples_per_second": BATCH / elapsed,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            "capacity_artifact_file_sha256": _sha256(CAPACITY),
            "nonfinite_count": 0,
            "joint_model_contract_sha256": model.contract_sha256,
            "protocol_sha256": gate.digest,
            "encoder_contract_sha256": encoder_lock.sha256,
            "capacity_resolution_sha256": capacity.digest,
            "cache_metadata_sha256": cache_metadata_sha256,
            "cache_content_sha256": source.identity.cache_content_sha256,
            "full_materialized_scan_sha256": (
                source.identity.full_materialized_scan_sha256
            ),
            "cost_snapshot": cost,
            "storage_snapshot": storage,
            "targets_multitask_v2_sha256": source.identity.targets_multitask_v2_sha256,
            "task_constants_sha256": source.identity.task_constants_sha256,
            "test_unlocked": False,
            "telemetry": "disabled",
        }
        digest = _write_immutable_json(output, receipt)
        print(json.dumps({"ok": True, "path": str(output), "sha256": digest}))
        return 0
    finally:
        source.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=("dcdn_core", "dcnv2_full_rank")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resolved-config-sha256", required=True)
    args = parser.parse_args()
    return run(
        args.model,
        args.output.resolve(),
        args.run_id,
        args.resolved_config_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
