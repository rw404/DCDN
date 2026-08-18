#!/usr/bin/env python3
"""Run the predeclared, metric-free KuaiRec split/feature audit.

The command reads only columns allowlisted by ``features_v1.yaml``.  Current
``play_duration`` is reduced immediately to the frozen ``> 0`` predicate;
``watch_ratio`` is never selected.  The durable output contains row counts,
split-membership hashes based on source row indices, mapping cardinalities,
schema/invariant evidence, and the train-only video-duration q99 ``c``.  It
cannot materialize a test label shard or evaluate a model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from dcdn.data.kuairec import (
    BIG_MATRIX_COLUMNS,
    BIG_MATRIX_SHA256,
    DAILY_BASE_SOURCES,
    DAILY_COLUMNS,
    DAILY_FEATURES_SHA256,
    DATASET_ID,
    EXPECTED_DAILY_ROWS,
    EXPECTED_PARTIAL_NULL_ROWS,
    MILLISECONDS_PER_SECOND,
    PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES,
    SOURCE_MANIFEST_SHA256,
    KuaiRecContractError,
    load_feature_contract,
    sha256_bytes,
    split_boundaries,
    verify_source_manifest_reference,
)
from scripts.cost_guard import evaluate_cost

DEFAULT_RAW_DATA_DIR = Path(
    "/home/codex/.cache/dcdn-public/raw/kuairec/source-v1-c23f211c/KuaiRec 2.0/data"
)
DEFAULT_CONTRACT = Path("configs/data/kuairec/features_v1.yaml")
DEFAULT_OUTPUT = Path("configs/data/kuairec/blind_scan_v1.yaml")

BIG_MATRIX_BYTES = 1_083_521_226
DAILY_FEATURES_BYTES = 85_855_213
COPY_CHUNK_BYTES = 8 * 1024**2
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_ETA_HOURS = Decimal("0.5")
MIN_OVERLAY_FREE_BYTES = 20 * 1024**3
MIN_PERSISTENT_FREE_BYTES = 10 * 1024**3
MAX_DURABLE_PROJECT_BYTES = 40 * 1024**3
MAX_USAGE_FRACTION = 0.85


class KuaiRecBlindAuditError(RuntimeError):
    """Raised when source, guard, split, or leakage evidence fails closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=COPY_CHUNK_BYTES) as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_raw_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise KuaiRecBlindAuditError(f"raw source must be a regular file: {path}")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise KuaiRecBlindAuditError("raw source must be a private regular file")
    if metadata.st_size != expected_bytes:
        raise KuaiRecBlindAuditError(
            f"raw source size mismatch for {path.name}: {metadata.st_size}"
        )
    digest = _sha256_file(path)
    if digest != expected_sha256:
        raise KuaiRecBlindAuditError(
            f"raw source SHA-256 mismatch for {path.name}: {digest}"
        )


def _filesystem_snapshot(path: Path) -> dict[str, int | float]:
    stats = os.statvfs(path)
    unit = stats.f_frsize or stats.f_bsize
    total = unit * stats.f_blocks
    available = unit * stats.f_bavail
    usage = 1.0 - available / total if total else 1.0
    return {"total_bytes": total, "available_bytes": available, "usage": usage}


def _tree_size(root: Path) -> int:
    total = 0
    for directory, _, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in filenames:
            path = base / name
            if path.is_symlink():
                continue
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def enforce_local_guards(
    *,
    eta_hours: Decimal,
    raw_data_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Enforce the owner-approved time/cost and locally observed disk policy."""

    if eta_hours <= 0 or eta_hours > MAX_ETA_HOURS:
        raise KuaiRecBlindAuditError("eta_hours must lie in (0, 0.5]")
    cost = evaluate_cost(
        datetime.now(UTC),
        eta_hours=eta_hours,
        phase="compute",
        work_class="mandatory",
    )
    if not cost.launch_allowed:
        raise KuaiRecBlindAuditError(
            f"cost/time launch guard blocked audit: {list(cost.reasons)}"
        )

    raw = _filesystem_snapshot(raw_data_dir)
    persistent = _filesystem_snapshot(repository_root)
    durable_bytes = _tree_size(repository_root)
    if int(raw["available_bytes"]) < MIN_OVERLAY_FREE_BYTES:
        raise KuaiRecBlindAuditError("temporary overlay has less than 20 GiB free")
    if float(raw["usage"]) >= MAX_USAGE_FRACTION:
        raise KuaiRecBlindAuditError("temporary overlay usage reached 85%")
    if int(persistent["available_bytes"]) < MIN_PERSISTENT_FREE_BYTES:
        raise KuaiRecBlindAuditError("persistent mount has less than 10 GiB free")
    if durable_bytes > MAX_DURABLE_PROJECT_BYTES:
        raise KuaiRecBlindAuditError("durable project artifacts exceed 40 GiB")
    return {
        "cost": cost.as_dict(),
        "temporary_filesystem": raw,
        "persistent_filesystem": persistent,
        "durable_project_bytes_before_audit": durable_bytes,
    }


def _scan_csv(path: Path, *, columns: tuple[str, ...]) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        has_header=True,
        infer_schema_length=10_000,
        low_memory=True,
        rechunk=False,
        raise_if_empty=True,
    ).select(list(columns))


def _collect(frame: pl.LazyFrame) -> pl.DataFrame:
    return frame.collect(engine="streaming")


def _static_video_types(daily_path: Path) -> tuple[pl.DataFrame, dict[str, int]]:
    daily = _scan_csv(daily_path, columns=("video_id", "video_type")).select(
        pl.col("video_id").cast(pl.Int64, strict=False),
        pl.col("video_type")
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .fill_null("UNK")
        .alias("video_type"),
    )
    frame = _collect(daily)
    invalid_ids = frame.select(pl.col("video_id").is_null().sum()).item()
    if invalid_ids:
        raise KuaiRecBlindAuditError("item_daily_features has invalid video_id")
    grouped = frame.group_by("video_id").agg(
        pl.col("video_type").n_unique().alias("video_type_values"),
        pl.col("video_type").first().alias("video_type"),
    )
    inconsistent = grouped.filter(pl.col("video_type_values") != 1).height
    if inconsistent:
        raise KuaiRecBlindAuditError(
            "video_type is not static by video; time-safe primary filter cannot proceed"
        )
    static = grouped.select("video_id", "video_type")
    evidence = {
        "daily_rows_checked_for_video_type": frame.height,
        "videos_with_static_video_type": static.height,
        "videos_with_inconsistent_video_type": inconsistent,
    }
    return static, evidence


def _audit_daily_numeric(daily_path: Path) -> dict[str, int]:
    selected = ("video_id", "date", *DAILY_BASE_SOURCES)
    frame = _scan_csv(daily_path, columns=selected).select(
        pl.col("video_id").cast(pl.Int64, strict=False),
        pl.col("date").cast(pl.Int64, strict=False),
        *(
            pl.col(source).cast(pl.Float64, strict=False).alias(source)
            for source in DAILY_BASE_SOURCES
        ),
    )
    invalid_expressions = [
        pl.col("video_id").is_null().sum().alias("invalid_video_id"),
        pl.col("date").is_null().sum().alias("invalid_date"),
    ]
    invalid_expressions.extend(
        (pl.col(source).is_null() | ~pl.col(source).is_finite() | (pl.col(source) < 0))
        .sum()
        .alias(source)
        for source in DAILY_BASE_SOURCES
    )
    summary = _collect(frame.select(invalid_expressions)).row(0, named=True)
    invalid_total = sum(int(value) for value in summary.values())
    if invalid_total:
        raise KuaiRecBlindAuditError(
            f"daily base schema has {invalid_total} invalid cells"
        )
    counts = _collect(
        frame.select(
            pl.len().alias("daily_rows"),
            pl.struct("video_id", "date").n_unique().alias("video_date_groups"),
        )
    ).row(0, named=True)
    return {
        "daily_rows": int(counts["daily_rows"]),
        "video_date_groups": int(counts["video_date_groups"]),
        "invalid_numeric_or_key_cells": invalid_total,
    }


def _audit_excluded_partial_null_daily(
    daily_path: Path,
    *,
    expected_rows: int = EXPECTED_DAILY_ROWS,
    expected_null_rows: int = EXPECTED_PARTIAL_NULL_ROWS,
) -> dict[str, Any]:
    """Verify exact missingness of owner-excluded counters without values."""

    if expected_rows <= 0 or not 0 <= expected_null_rows <= expected_rows:
        raise KuaiRecBlindAuditError("invalid expected partial-null counts")
    frame = _scan_csv(daily_path, columns=PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES)
    expressions: list[pl.Expr] = [pl.len().alias("daily_rows")]
    for source in PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES:
        raw = pl.col(source)
        parsed = raw.cast(pl.Float64, strict=False)
        expressions.extend(
            [
                raw.is_null().sum().alias(f"{source}__null"),
                (raw.is_not_null() & parsed.is_null())
                .sum()
                .alias(f"{source}__parse_invalid"),
                (parsed.is_not_null() & ~parsed.is_finite())
                .sum()
                .alias(f"{source}__nonfinite"),
                (parsed.is_not_null() & parsed.is_finite() & (parsed < 0))
                .sum()
                .alias(f"{source}__negative"),
            ]
        )
    summary = _collect(frame.select(expressions)).row(0, named=True)
    rows = int(summary["daily_rows"])
    if rows != expected_rows:
        raise KuaiRecBlindAuditError(f"excluded daily source row count drifted: {rows}")

    columns: dict[str, dict[str, int]] = {}
    for source in PARTIALLY_NULL_EXCLUDED_DAILY_SOURCES:
        counts = {
            "total": rows,
            "null": int(summary[f"{source}__null"]),
            "nonnull": rows - int(summary[f"{source}__null"]),
            "parse_invalid": int(summary[f"{source}__parse_invalid"]),
            "nonfinite": int(summary[f"{source}__nonfinite"]),
            "negative": int(summary[f"{source}__negative"]),
        }
        if counts["null"] != expected_null_rows or any(
            counts[key] != 0 for key in ("parse_invalid", "nonfinite", "negative")
        ):
            raise KuaiRecBlindAuditError(
                f"excluded daily source missingness drifted: {source}"
            )
        columns[source] = counts
    return {
        "daily_rows": rows,
        "expected_null_rows_per_column": expected_null_rows,
        "policy": "owner_approved_full_exclusion_without_zero_imputation",
        "excluded_from_all_feature_regimes": True,
        "columns": columns,
    }


def _load_projected_big_matrix(big_path: Path) -> pl.DataFrame:
    selected = (
        "user_id",
        "video_id",
        "play_duration",
        "video_duration",
        "date",
        "timestamp",
    )
    lazy = (
        pl.scan_csv(
            big_path,
            has_header=True,
            infer_schema_length=10_000,
            low_memory=True,
            rechunk=False,
            raise_if_empty=True,
        )
        .with_row_index("source_row_index0")
        .select("source_row_index0", *selected)
        .select(
            pl.col("source_row_index0").cast(pl.UInt64),
            pl.col("user_id").cast(pl.Int64, strict=False),
            pl.col("video_id").cast(pl.Int64, strict=False),
            pl.col("play_duration").cast(pl.Float64, strict=False),
            pl.col("video_duration").cast(pl.Float64, strict=False),
            pl.col("date").cast(pl.Int64, strict=False),
            pl.col("timestamp").cast(pl.Int64, strict=False),
        )
    )
    frame = _collect(lazy)
    invalid = frame.select(
        pl.col("user_id").is_null().sum().alias("user_id"),
        pl.col("video_id").is_null().sum().alias("video_id"),
        (pl.col("play_duration").is_null() | ~pl.col("play_duration").is_finite())
        .sum()
        .alias("play_duration"),
        (pl.col("video_duration").is_null() | ~pl.col("video_duration").is_finite())
        .sum()
        .alias("video_duration"),
        pl.col("date").is_null().sum().alias("date"),
        pl.col("timestamp").is_null().sum().alias("timestamp"),
    ).row(0, named=True)
    invalid_total = sum(int(value) for value in invalid.values())
    if invalid_total:
        raise KuaiRecBlindAuditError(
            f"big_matrix projected schema has {invalid_total} invalid cells"
        )
    return frame


def _membership_hash(frame: pl.DataFrame, *, split_name: str) -> str:
    ordered = frame.sort("filtered_sorted_rank0")
    indices = ordered.get_column("source_row_index0").to_numpy()
    canonical = np.asarray(indices, dtype=np.uint64).astype(">u8", copy=False)
    digest = hashlib.sha256()
    digest.update(DATASET_ID.encode("ascii"))
    digest.update(b"\0")
    digest.update(BIG_MATRIX_SHA256.encode("ascii"))
    digest.update(b"\0")
    digest.update(split_name.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _cold_drop_evidence(
    split: pl.DataFrame,
    *,
    train_users: pl.DataFrame,
    train_items: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    flags = split.join(
        train_users.with_columns(pl.lit(True).alias("known_user")),
        on="user_id",
        how="left",
    ).join(
        train_items.with_columns(pl.lit(True).alias("known_item")),
        on="video_id",
        how="left",
    )
    flags = flags.with_columns(
        pl.col("known_user").fill_null(False),
        pl.col("known_item").fill_null(False),
    )
    evidence = flags.select(
        pl.len().alias("before_inner_drop"),
        (~pl.col("known_user")).sum().alias("unseen_user_rows"),
        (~pl.col("known_item")).sum().alias("unseen_item_rows"),
        (~(pl.col("known_user") & pl.col("known_item"))).sum().alias("dropped_rows"),
        (pl.col("known_user") & pl.col("known_item")).sum().alias("after_inner_drop"),
    ).row(0, named=True)
    warm = flags.filter(pl.col("known_user") & pl.col("known_item")).drop(
        "known_user", "known_item"
    )
    return warm, {key: int(value) for key, value in evidence.items()}


def run_blind_contract_audit(
    *,
    raw_data_dir: Path,
    contract_path: Path,
    repository_root: Path,
    eta_hours: Decimal,
) -> dict[str, Any]:
    """Execute only the frozen source/filter/split audit and return safe evidence."""

    contract = load_feature_contract(contract_path)
    source_digest = verify_source_manifest_reference(
        contract, repository_root=repository_root
    )
    if source_digest != SOURCE_MANIFEST_SHA256:
        raise KuaiRecBlindAuditError("source-manifest digest drifted")
    guards = enforce_local_guards(
        eta_hours=eta_hours,
        raw_data_dir=raw_data_dir,
        repository_root=repository_root,
    )

    big_path = raw_data_dir / "big_matrix.csv"
    daily_path = raw_data_dir / "item_daily_features.csv"
    _verify_raw_file(
        big_path,
        expected_bytes=BIG_MATRIX_BYTES,
        expected_sha256=BIG_MATRIX_SHA256,
    )
    _verify_raw_file(
        daily_path,
        expected_bytes=DAILY_FEATURES_BYTES,
        expected_sha256=DAILY_FEATURES_SHA256,
    )

    excluded_partial_null_evidence = _audit_excluded_partial_null_daily(daily_path)
    static_types, type_evidence = _static_video_types(daily_path)
    daily_evidence = _audit_daily_numeric(daily_path)
    raw = _load_projected_big_matrix(big_path)
    raw_rows = raw.height

    joined = raw.join(static_types, on="video_id", how="left").with_columns(
        pl.col("video_type").fill_null("UNK")
    )
    non_ad = joined.filter(pl.col("video_type") != "AD")
    positive_play = non_ad.filter(pl.col("play_duration") > 0)
    positive_video = positive_play.filter(pl.col("video_duration") > 0)
    filter_counts = {
        "raw_rows": raw_rows,
        "after_static_non_ad": non_ad.height,
        "after_positive_current_play": positive_play.height,
        "after_positive_current_video_duration": positive_video.height,
    }
    if positive_video.is_empty():
        raise KuaiRecBlindAuditError("all rows were removed by frozen filters")

    # The current label values are discarded before sorting, splitting, c
    # fitting, mappings, hashing, or any persisted evidence is computed.
    filtered = (
        positive_video.drop("play_duration", "video_type")
        .with_columns(
            (pl.col("video_duration") / MILLISECONDS_PER_SECOND).alias(
                "video_duration_seconds"
            )
        )
        .drop("video_duration")
        .sort("timestamp", "user_id", "video_id", "source_row_index0")
    )
    filtered = filtered.with_row_index("filtered_sorted_rank0")
    boundaries = split_boundaries(filtered.height)
    train = filtered.slice(0, boundaries.train_rows)
    validation = filtered.slice(
        boundaries.train_rows, boundaries.validation_rows_before_cold_drop
    )
    sealed_test = filtered.slice(
        boundaries.validation_end_exclusive,
        boundaries.test_rows_before_cold_drop,
    )

    c_seconds = train.select(
        pl.col("video_duration_seconds").quantile(0.99, interpolation="linear")
    ).item()
    if c_seconds is None or not 10.0 <= float(c_seconds) <= 600.0:
        raise KuaiRecBlindAuditError(
            "train-only video-duration q99 violates frozen seconds sanity interval"
        )

    train_users = train.select("user_id").unique().sort("user_id")
    train_items = train.select("video_id").unique().sort("video_id")
    validation_warm, validation_cold = _cold_drop_evidence(
        validation, train_users=train_users, train_items=train_items
    )
    test_warm, test_cold = _cold_drop_evidence(
        sealed_test, train_users=train_users, train_items=train_items
    )

    split_counts = {
        "train": train.height,
        "validation_before_cold_drop": validation.height,
        "validation_after_cold_drop": validation_warm.height,
        "sealed_test_before_cold_drop": sealed_test.height,
        "sealed_test_after_cold_drop": test_warm.height,
    }
    memberships = {
        "train": _membership_hash(train, split_name="train"),
        "validation_after_cold_drop": _membership_hash(
            validation_warm, split_name="validation_after_cold_drop"
        ),
        "sealed_test_after_cold_drop": _membership_hash(
            test_warm, split_name="sealed_test_after_cold_drop"
        ),
    }

    return {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "artifact_role": "label_blind_filter_split_audit",
        "audited_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "feature_contract": {
            "path": contract_path.as_posix(),
            "sha256": _sha256_file(contract_path),
        },
        "source_manifest_sha256": source_digest,
        "test_unlocked": False,
        "test_labels_opened": False,
        "test_label_values_exposed_or_summarized": False,
        "current_play_positive_predicate_applied_blind": True,
        "watch_ratio_read": False,
        "validation_or_test_model_metrics_read": False,
        "test_view_materialized": False,
        "guards": guards,
        "raw_integrity": {
            "big_matrix_sha256": BIG_MATRIX_SHA256,
            "item_daily_features_sha256": DAILY_FEATURES_SHA256,
            "exact_headers_from_source_manifest": {
                "big_matrix": list(BIG_MATRIX_COLUMNS),
                "item_daily_features": list(DAILY_COLUMNS),
            },
        },
        "duration_unit_normalization": {
            "source_units": "milliseconds",
            "canonical_units": "seconds",
            "divisor": MILLISECONDS_PER_SECOND,
            "positive_filters_applied_before_conversion": True,
            "big_matrix_video_duration_converted_before_c_and_P": True,
            "current_play_conversion_deferred_to_sealed_label_materialization": True,
            "lagged_daily_play_duration_conversion_declared_in_feature_contract": True,
        },
        "excluded_partial_null_daily_schema": excluded_partial_null_evidence,
        "static_video_type_invariance": type_evidence,
        "daily_base_schema": daily_evidence,
        "filter_counts": filter_counts,
        "split_counts": split_counts,
        "cold_drop_counts": {
            "validation": validation_cold,
            "sealed_test": test_cold,
        },
        "train_mapping_cardinalities": {
            "user_id": train_users.height,
            "video_id": train_items.height,
            "category_mappings": "fit_train_only_during_feature_materialization",
        },
        "train_only_label_constant": {
            "name": "c_seconds",
            "value": float(c_seconds),
            "source": "filtered_train_current_video_duration_ms_div_1000",
            "quantile": 0.99,
            "interpolation": "linear_R7",
            "validation_or_test_values_used": False,
        },
        "split_membership_sha256": memberships,
        "outcome_free_membership_hash_encoding": (
            "sha256(dataset_id NUL big_matrix_sha256 NUL split_name NUL "
            "ordered_big_endian_uint64_source_row_indices)"
        ),
    }


def _write_fsynced_temp(parent: Path, prefix: str, data: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(dir=parent, prefix=prefix)
    path = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return path


def write_audit(path: Path, payload: dict[str, Any]) -> str:
    """Atomically write a compact YAML audit and exact SHA-256 sidecar."""

    if path.name != "blind_scan_v1.yaml":
        raise KuaiRecBlindAuditError("audit filename must be blind_scan_v1.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    sidecar = parent / f"{path.name}.sha256"
    data = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    ).encode("utf-8")
    if len(data) > MAX_OUTPUT_BYTES:
        raise KuaiRecBlindAuditError("serialized blind audit exceeds size limit")
    digest = sha256_bytes(data)
    sidecar_data = f"{digest}  {target.name}\n".encode("ascii")
    first: Path | None = None
    second: Path | None = None
    try:
        first = _write_fsynced_temp(parent, f".{target.name}.audit-", data)
        second = _write_fsynced_temp(parent, f".{sidecar.name}.sidecar-", sidecar_data)
        os.replace(first, target)
        first = None
        os.replace(second, sidecar)
        second = None
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if first is not None:
            first.unlink(missing_ok=True)
        if second is not None:
            second.unlink(missing_ok=True)
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eta-hours", type=Decimal, default=MAX_ETA_HOURS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    try:
        payload = run_blind_contract_audit(
            raw_data_dir=args.raw_data_dir,
            contract_path=args.contract,
            repository_root=repository_root,
            eta_hours=args.eta_hours,
        )
        digest = write_audit(args.output, payload)
    except (
        KuaiRecBlindAuditError,
        KuaiRecContractError,
        OSError,
        pl.exceptions.PolarsError,
    ) as error:
        raise SystemExit(f"FAIL: {error}") from error
    print(
        "PASS: KuaiRec blind filter/split audit "
        f"sha256={digest}; test_unlocked=false; no model metric"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
