#!/usr/bin/env python3
"""VK-LSVD phase-1 schema and label-constant audit with week 26 sealed.

The command opens Parquet metadata for weeks 0--25 and the two static metadata
tables.  It decodes only metadata keys and ``items_metadata.duration`` needed
to reproduce the label constant.  The week-26 file is never opened, stated,
globbed, schema-inspected, or decoded by this module.
"""

from __future__ import annotations

import argparse
import json
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from dcdn.data.vk_lsvd import (
    INTERACTION_SCHEMA,
    ITEMS_METADATA_SCHEMA,
    LABEL_DURATION_Q99_SECONDS,
    PHASE1_SEMANTIC_WEEKS,
    USERS_METADATA_SCHEMA,
    VKLSVDContractError,
    load_feature_contract,
    semantic_week_relative_path,
    verify_source_manifest_reference,
)

DEFAULT_RAW_ROOT = Path("/home/codex/.cache/dcdn-public/raw/vk_lsvd")
DEFAULT_CONTRACT = Path("configs/data/vk_lsvd/features_v1.yaml")


class Phase1AuditError(RuntimeError):
    """Raised when an allowlisted phase-1 source violates the contract."""


def _require_regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Phase1AuditError(f"cannot inspect {label}: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise Phase1AuditError(f"{label} must be a single-link regular file")


def _parquet_schema_record(path: Path) -> dict[str, Any]:
    _require_regular_file(path, "phase-1 Parquet source")
    try:
        parquet = pq.ParquetFile(path)
    except Exception as error:
        raise Phase1AuditError(f"cannot read Parquet metadata: {path}") from error
    schema = tuple(f"{field.name}:{field.type}" for field in parquet.schema_arrow)
    return {
        "schema": schema,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
    }


def _metadata_value_audit(
    *, users_path: Path, items_path: Path
) -> dict[str, dict[str, Any]]:
    _require_regular_file(users_path, "users metadata")
    _require_regular_file(items_path, "items metadata")
    try:
        users = (
            pl.scan_parquet(users_path)
            .select(
                pl.len().alias("rows"),
                pl.col("user_id").n_unique().alias("unique_ids"),
                pl.col("user_id").null_count().alias("user_id_nulls"),
                pl.col("age").null_count().alias("age_nulls"),
                pl.col("gender").null_count().alias("gender_nulls"),
                pl.col("geo").null_count().alias("geo_nulls"),
            )
            .collect()
            .to_dicts()[0]
        )
        items = (
            pl.scan_parquet(items_path)
            .select(
                pl.len().alias("rows"),
                pl.col("item_id").n_unique().alias("unique_ids"),
                pl.col("item_id").null_count().alias("item_id_nulls"),
                pl.col("duration").null_count().alias("duration_nulls"),
                pl.col("duration").min().alias("duration_min"),
                pl.col("duration").max().alias("duration_max"),
                (pl.col("duration") <= 0).sum().alias("duration_nonpositive"),
                pl.col("duration")
                .quantile(0.99, interpolation="nearest")
                .alias("duration_q99_nearest"),
            )
            .collect()
            .to_dicts()[0]
        )
    except Exception as error:
        raise Phase1AuditError("metadata value audit failed") from error

    if users["rows"] != users["unique_ids"]:
        raise Phase1AuditError("users_metadata.user_id must be unique")
    if any(
        users[key] != 0
        for key in ("user_id_nulls", "age_nulls", "gender_nulls", "geo_nulls")
    ):
        raise Phase1AuditError("required user metadata contains nulls")
    if items["rows"] != items["unique_ids"]:
        raise Phase1AuditError("items_metadata.item_id must be unique")
    if items["item_id_nulls"] != 0 or items["duration_nulls"] != 0:
        raise Phase1AuditError("required item metadata contains nulls")
    if items["duration_nonpositive"] != 0:
        raise Phase1AuditError("items_metadata.duration must be positive")
    if float(items["duration_q99_nearest"]) != LABEL_DURATION_Q99_SECONDS:
        raise Phase1AuditError("metadata duration q99 differs from frozen c")
    return {"users_metadata": users, "items_metadata": items}


def run_phase1_audit(
    *,
    raw_root: Path,
    contract_path: Path,
    repository_root: Path,
    schema_reader: Callable[[Path], dict[str, Any]] = _parquet_schema_record,
) -> dict[str, Any]:
    """Audit only allowlisted pre-test schemas and metadata provenance."""

    try:
        contract = load_feature_contract(contract_path)
        source_digest = verify_source_manifest_reference(
            contract, repository_root=repository_root
        )
    except VKLSVDContractError as error:
        raise Phase1AuditError(str(error)) from error

    if raw_root.is_symlink() or not raw_root.is_dir():
        raise Phase1AuditError("raw_root must be a real directory")

    weeks: list[dict[str, Any]] = []
    opened_paths: set[Path] = set()
    for week in PHASE1_SEMANTIC_WEEKS:
        relative = semantic_week_relative_path(week)
        path = raw_root.joinpath(*relative.parts)
        if path in opened_paths:
            raise Phase1AuditError("duplicate phase-1 source path")
        opened_paths.add(path)
        record = schema_reader(path)
        if tuple(record.get("schema", ())) != INTERACTION_SCHEMA:
            raise Phase1AuditError(f"interaction schema mismatch in week {week}")
        rows = record.get("rows")
        row_groups = record.get("row_groups")
        if type(rows) is not int or rows <= 0:
            raise Phase1AuditError(f"week {week} has no rows")
        if type(row_groups) is not int or row_groups <= 0:
            raise Phase1AuditError(f"week {week} has no row groups")
        weeks.append(
            {
                "week": week,
                "role": (
                    "history" if week <= 19 else "train" if week <= 24 else "validation"
                ),
                "path": relative.as_posix(),
                "rows": rows,
                "row_groups": row_groups,
                "schema_verified": True,
            }
        )

    users_path = raw_root / "metadata/users_metadata.parquet"
    items_path = raw_root / "metadata/items_metadata.parquet"
    users_schema = schema_reader(users_path)
    items_schema = schema_reader(items_path)
    if tuple(users_schema.get("schema", ())) != USERS_METADATA_SCHEMA:
        raise Phase1AuditError("users metadata schema mismatch")
    if tuple(items_schema.get("schema", ())) != ITEMS_METADATA_SCHEMA:
        raise Phase1AuditError("items metadata schema mismatch")
    metadata_values = _metadata_value_audit(
        users_path=users_path, items_path=items_path
    )

    contract_sidecar = contract_path.with_name(f"{contract_path.name}.sha256")
    contract_digest = contract_sidecar.read_text(encoding="ascii").split()[0]
    return {
        "schema_version": 1,
        "dataset_id": "vk_lsvd_up0.01_ir0.01",
        "audit_role": "phase1_schema_and_label_constant_without_test_access",
        "contract_sha256": contract_digest,
        "source_manifest_sha256": source_digest,
        "semantic_weeks_opened": list(PHASE1_SEMANTIC_WEEKS),
        "week26_file_open_attempted": False,
        "week26_schema_inspected": False,
        "week26_rows_or_labels_read": False,
        "test_unlocked": False,
        "validation_or_test_model_metrics_computed": False,
        "decoded_metadata_columns": [
            "users_metadata.user_id",
            "users_metadata.age",
            "users_metadata.gender",
            "users_metadata.geo",
            "items_metadata.item_id",
            "items_metadata.duration",
        ],
        "interaction_values_decoded": False,
        "label_c_seconds": LABEL_DURATION_Q99_SECONDS,
        "weeks": weeks,
        "metadata": {
            "users_schema_verified": True,
            "items_schema_verified": True,
            **metadata_values,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_phase1_audit(
            raw_root=args.raw_root,
            contract_path=args.contract,
            repository_root=args.repository_root,
        )
    except Phase1AuditError as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, "audit": report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
