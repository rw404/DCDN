#!/usr/bin/env python3
"""Attest VK-LSVD views and build pinned raw runtime caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch  # noqa: E402

from dcdn.data.vk_lsvd_materialize import (  # noqa: E402
    VKLSVDMaterializationError,
    resource_and_cost_guard,
)
from dcdn.data.vk_lsvd_runtime import (  # noqa: E402
    VKLSVDRawBatchSource,
    VKLSVDRuntimeError,
    _cache_expected_bytes,
    build_full_scan_attestation,
    build_vk_lsvd_runtime_cache,
    load_full_scan_attestation,
    runtime_storage_guard,
)

DEFAULT_VIEWS_MANIFEST = Path("data/manifests/vk_lsvd_views_v1.yaml")
DEFAULT_FULL_SCAN_MANIFEST = Path("data/manifests/vk_lsvd_full_scan_v1.yaml")
DEFAULT_FEATURE_CONTRACT = Path("configs/data/vk_lsvd/features_v1.yaml")
DEFAULT_SOURCE_MANIFEST = Path("data/manifests/vk_lsvd_source_v1.yaml")
DEFAULT_LEDGER = Path("docs/cost_ledger.md")
DEFAULT_VIEWS_ROOT = Path("/home/codex/.cache/dcdn-public/processed/vk_lsvd/views_v1")
DEFAULT_CACHE_ROOTS = {
    "C+P": Path("/home/codex/.cache/dcdn-public/runtime/vk_lsvd/c_plus_p_v2"),
    "C+P+E": Path("/home/codex/.cache/dcdn-public/runtime/vk_lsvd/c_plus_p_plus_e_v2"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--views-root", type=Path, default=DEFAULT_VIEWS_ROOT)
    parser.add_argument("--views-manifest", type=Path, default=DEFAULT_VIEWS_MANIFEST)
    parser.add_argument(
        "--full-scan-manifest", type=Path, default=DEFAULT_FULL_SCAN_MANIFEST
    )
    parser.add_argument(
        "--feature-contract", type=Path, default=DEFAULT_FEATURE_CONTRACT
    )
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--eta-hours", type=float, default=2.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("scan", help="independently scan all 12 explicit view files")
    build = subparsers.add_parser(
        "build-cache", help="build one atomic train/validation raw cache"
    )
    build.add_argument("--feature-regime", choices=("C+P", "C+P+E"), required=True)
    build.add_argument("--expected-scan-sha256", required=True)
    verify = subparsers.add_parser("verify-cache", help="verify hashes and batch order")
    verify.add_argument("--split", choices=("train", "validation"), required=True)
    verify.add_argument("--feature-regime", choices=("C+P", "C+P+E"), required=True)
    verify.add_argument("--expected-scan-sha256", required=True)
    verify.add_argument("--expected-cache-content-sha256", required=True)
    return parser


def _resolve(repository_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repository_root / value


def _emit(event: str, **details: Any) -> None:
    print(
        json.dumps({"event": event, **details}, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _cache_root(args: argparse.Namespace) -> Path:
    if args.cache_root is not None:
        return args.cache_root
    regime = getattr(args, "feature_regime", "C+P")
    return DEFAULT_CACHE_ROOTS[regime]


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = args.repository_root.resolve(strict=True)
    output_root = _cache_root(args).parent
    output_root.mkdir(parents=True, exist_ok=True)
    guard = resource_and_cost_guard(
        output_root=output_root,
        ledger_path=_resolve(repository_root, args.ledger),
        eta_hours=args.eta_hours,
    )
    _emit("preflight", **guard)
    return guard


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve(strict=True)
    views_manifest = _resolve(repository_root, args.views_manifest)
    full_scan_manifest = _resolve(repository_root, args.full_scan_manifest)
    cache_root = _cache_root(args)
    try:
        guard = _preflight(args)
        started = time.monotonic()
        if args.command == "scan":
            payload, digest = build_full_scan_attestation(
                views_manifest_path=views_manifest,
                views_root=args.views_root,
                output_manifest_path=full_scan_manifest,
                source_manifest_path=_resolve(repository_root, args.source_manifest),
                feature_contract_path=_resolve(repository_root, args.feature_contract),
            )
            _emit(
                "full_scan_complete",
                full_materialized_scan_sha256=digest,
                rows=payload["row_counts"],
                shard_files=12,
                elapsed_seconds=round(time.monotonic() - started, 6),
                test_unlocked=False,
                week26_path_constructed=False,
                model_metrics_computed=False,
            )
        elif args.command == "build-cache":
            scan, _ = load_full_scan_attestation(
                full_scan_manifest,
                expected_scan_sha256=args.expected_scan_sha256,
                require_expected_digest=True,
            )
            storage = runtime_storage_guard(
                cache_root.parent,
                estimated_output_bytes=_cache_expected_bytes(
                    scan["row_counts"], args.feature_regime
                ),
            )
            _emit("cache_storage_preflight", **storage.__dict__)
            payload, digest = build_vk_lsvd_runtime_cache(
                views_manifest_path=views_manifest,
                views_root=args.views_root,
                full_scan_manifest_path=full_scan_manifest,
                cache_root=cache_root,
                feature_regime=args.feature_regime,
                expected_scan_sha256=args.expected_scan_sha256,
            )
            _emit(
                "runtime_cache_complete",
                cache_manifest_sha256=digest,
                cache_content_sha256=payload["cache_content_sha256"],
                full_materialized_scan_sha256=payload["full_materialized_scan_sha256"],
                feature_regime=payload["feature_regime"],
                rows={name: item["rows"] for name, item in payload["splits"].items()},
                elapsed_seconds=round(time.monotonic() - started, 6),
                test_split_present=False,
                model_metrics_computed=False,
            )
        else:
            source = VKLSVDRawBatchSource(
                cache_root,
                split=args.split,
                full_scan_manifest_path=full_scan_manifest,
                feature_regime=args.feature_regime,
                expected_scan_sha256=args.expected_scan_sha256,
                expected_cache_content_sha256=args.expected_cache_content_sha256,
            )
            requested = [len(source) - 1, 0, len(source) - 1, min(7, len(source) - 1)]
            batch = source.batch(requested, device=torch.device("cpu"))
            batch.validate(source.identity, requested)
            _emit(
                "runtime_cache_verified",
                split=args.split,
                rows=len(source),
                requested_indices=requested,
                preserved_indices=batch.global_row_indices.tolist(),
                full_materialized_scan_sha256=source.identity.full_materialized_scan_sha256,
                cache_content_sha256=source.identity.cache_content_sha256,
                feature_regime=source.identity.feature_regime,
                elapsed_seconds=round(time.monotonic() - started, 6),
                model_metrics_computed=False,
            )
        if guard["warning_threshold_crossed"]:
            _emit("cost_warning", threshold_usd=guard["warning_threshold_usd"])
    except (
        OSError,
        ValueError,
        VKLSVDMaterializationError,
        VKLSVDRuntimeError,
    ) as error:
        print(
            json.dumps(
                {"event": "blocked", "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
