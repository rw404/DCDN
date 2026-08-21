#!/usr/bin/env python3
"""Attest VK-LSVD seven-head views and build sealed runtime caches."""

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
from dcdn.data.vk_lsvd_multitask_runtime import (  # noqa: E402
    VKLSVDMultiTargetRuntimeError,
    VKLSVDMultiTargetSource,
    build_multitask_runtime_cache,
    publish_immutable_json_artifact,
    scan_multitask_views,
)

DEFAULT_VIEWS_MANIFEST = Path("data/manifests/vk_lsvd_multitask_views_v2.yaml")
DEFAULT_SCAN_ARTIFACT = Path("data/manifests/vk_lsvd_multitask_full_scan_v2.json")
DEFAULT_LEDGER = Path("docs/cost_ledger.md")
DEFAULT_VIEWS_ROOT = Path(
    "/home/codex/.cache/dcdn-public/processed/vk_lsvd/multitask_views_v2"
)
DEFAULT_CACHE_ROOTS = {
    "C+P": Path("/home/codex/.cache/dcdn-public/runtime/vk_lsvd/multitask_v2/c_plus_p"),
    "C+P+E": Path(
        "/home/codex/.cache/dcdn-public/runtime/vk_lsvd/multitask_v2/c_plus_p_plus_e"
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--views-root", type=Path, default=DEFAULT_VIEWS_ROOT)
    parser.add_argument("--views-manifest", type=Path, default=DEFAULT_VIEWS_MANIFEST)
    parser.add_argument("--scan-artifact", type=Path, default=DEFAULT_SCAN_ARTIFACT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--eta-hours", type=float, default=1.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("scan", help="hash/reopen all 12 train-validation shards")
    build = subparsers.add_parser("build-cache", help="build one immutable NPY cache")
    build.add_argument("--feature-regime", choices=("C+P", "C+P+E"), required=True)
    build.add_argument("--expected-scan-sha256", required=True)
    verify = subparsers.add_parser("verify-cache", help="reopen cache and verify order")
    verify.add_argument("--split", choices=("train", "validation"), required=True)
    verify.add_argument("--feature-regime", choices=("C+P", "C+P+E"), required=True)
    verify.add_argument("--expected-scan-sha256", required=True)
    verify.add_argument("--expected-cache-metadata-sha256", required=True)
    verify.add_argument("--expected-cache-content-sha256", required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


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
    scan_artifact = _resolve(repository_root, args.scan_artifact)
    cache_root = _cache_root(args)
    try:
        guard = _preflight(args)
        started = time.monotonic()
        if args.command == "scan":
            payload, expected_digest = scan_multitask_views(
                manifest_path=views_manifest,
                views_root=args.views_root,
            )
            digest = publish_immutable_json_artifact(scan_artifact, payload)
            if digest != expected_digest:
                raise VKLSVDMultiTargetRuntimeError("published scan digest drifted")
            _emit(
                "multitask_scan_complete",
                scan_sha256=digest,
                rows=payload["row_counts"],
                shard_files=12,
                elapsed_seconds=round(time.monotonic() - started, 6),
                test_split_present=False,
                week26_path_constructed=False,
                model_metrics_computed=False,
            )
        elif args.command == "build-cache":
            payload, digest = build_multitask_runtime_cache(
                manifest_path=views_manifest,
                views_root=args.views_root,
                scan_path=scan_artifact,
                expected_scan_sha256=args.expected_scan_sha256,
                cache_root=cache_root,
                feature_regime=args.feature_regime,
            )
            _emit(
                "multitask_cache_complete",
                cache_metadata_sha256=digest,
                cache_content_sha256=payload["cache_content_sha256"],
                feature_regime=args.feature_regime,
                rows={
                    name: item["rows"]
                    for name, item in payload["content_identity"]["splits"].items()
                },
                elapsed_seconds=round(time.monotonic() - started, 6),
                test_split_present=False,
                model_metrics_computed=False,
            )
        else:
            source = VKLSVDMultiTargetSource(
                cache_root,
                split=args.split,
                expected_cache_metadata_sha256=args.expected_cache_metadata_sha256,
                expected_cache_content_sha256=args.expected_cache_content_sha256,
                expected_scan_sha256=args.expected_scan_sha256,
                runtime_value_checks=True,
            )
            requested = [len(source) - 1, 0, len(source) - 1, min(7, len(source) - 1)]
            batch = source.batch(requested, device=torch.device("cpu"))
            _emit(
                "multitask_cache_verified",
                split=args.split,
                feature_regime=source.identity.feature_regime,
                rows=len(source),
                requested_indices=requested,
                preserved_indices=batch.global_row_indices.tolist(),
                cache_content_sha256=source.identity.cache_content_sha256,
                scan_sha256=source.identity.full_materialized_scan_sha256,
                target_names=list(source.identity.target_names),
                elapsed_seconds=round(time.monotonic() - started, 6),
                test_split_present=False,
                model_metrics_computed=False,
            )
            source.close()
        if guard["warning_threshold_crossed"]:
            _emit("cost_warning", threshold_usd=guard["warning_threshold_usd"])
    except (
        OSError,
        ValueError,
        VKLSVDMaterializationError,
        VKLSVDMultiTargetRuntimeError,
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
