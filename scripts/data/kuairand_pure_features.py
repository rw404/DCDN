#!/usr/bin/env python3
"""Label-blind KuaiRand-Pure phase-1 schema/date coverage audit.

Only the released ``date``, ``time_ms``, and ``is_rand`` fields plus source
row offsets are decoded or aggregated.  All other CSV fields remain opaque
byte slices and are discarded.  This command cannot materialize a feature
view, open a label, or compute a model metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

from dcdn.data.kuairand_pure import (
    EVENT_COLUMNS,
    EVENT_FILES,
    EXPECTED_EMBARGO_CUTOFFS,
    EXPECTED_EMBARGO_EXCLUDED_COUNTS,
    EXPECTED_POST_EMBARGO_ROLE_COUNTS,
    EXPECTED_PRE_EMBARGO_ROLE_COUNTS,
    EXPECTED_TOTAL_SOURCE_ROWS,
    FILE_POLICY,
    RANDOM_MAIN_FILE,
    STANDARD_HISTORY_FILE,
    STANDARD_MAIN_FILE,
    KuaiRandContractError,
    classify_row_role,
    is_row_embargo_excluded,
    load_feature_contract,
    verify_source_manifest_reference,
)

DEFAULT_RAW_DATA_DIR = Path(
    "/home/codex/.cache/dcdn-public/raw/kuairand_pure/KuaiRand-Pure/data"
)
DEFAULT_CONTRACT = Path("configs/data/kuairand_pure/features_v1.yaml")
EXPECTED_FILE_DATE_RANGES = {
    STANDARD_HISTORY_FILE: (date(2022, 4, 9), date(2022, 4, 21)),
    STANDARD_MAIN_FILE: (date(2022, 4, 22), date(2022, 5, 8)),
    RANDOM_MAIN_FILE: (date(2022, 4, 22), date(2022, 5, 8)),
}
EXPECTED_IS_RAND = {
    STANDARD_HISTORY_FILE: b"0",
    STANDARD_MAIN_FILE: b"0",
    RANDOM_MAIN_FILE: b"1",
}
PINNED_EVENT_FILES = {
    STANDARD_HISTORY_FILE: {
        "bytes": 83_961_282,
        "sha256": "5bb6eb0b3d9f47e5436cb5dc82ee1899b845ebf9750a5560b801e929e18bd41c",
    },
    STANDARD_MAIN_FILE: {
        "bytes": 21_765_075,
        "sha256": "429e3b948828942e572f2c3a5be5a25799ffe75591d22d18cf417b9b534d31fd",
    },
    RANDOM_MAIN_FILE: {
        "bytes": 87_086_116,
        "sha256": "60b80994da969cd53da4d50c37ba3dafd6fb185df804c92c8410df34845a9d2c",
    },
}
MAX_HEADER_BYTES = 16 * 1024
ALLOWLISTED_ROW_PROJECTION_RE = re.compile(
    rb"^(?:[^,\r\n]*,){2}(?P<date>[^,\r\n]*),"
    rb"[^,\r\n]*,(?P<time_ms>[^,\r\n]*),"
    rb"(?:[^,\r\n]*,){12}(?P<is_rand>[^,\r\n]*),[^,\r\n]*$"
)


class CoverageAuditError(RuntimeError):
    """Raised when label-blind source metadata violates the frozen contract."""


def _iso_from_released_date(value: bytes) -> str:
    if len(value) != 8 or not value.isdigit():
        raise CoverageAuditError("date must use exactly YYYYMMDD digits")
    text = value.decode("ascii")
    try:
        parsed = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError as error:
        raise CoverageAuditError(f"invalid released date {text!r}") from error
    return parsed.isoformat()


def _parse_time_ms(value: bytes) -> int:
    if not value or not value.isdigit():
        raise CoverageAuditError("time_ms must be a non-negative decimal integer")
    parsed = int(value)
    if not 1_600_000_000_000 <= parsed <= 1_700_000_000_000:
        raise CoverageAuditError("time_ms falls outside the frozen sanity envelope")
    return parsed


def _strip_record_ending(record: bytes) -> bytes:
    if not record.endswith(b"\n"):
        raise CoverageAuditError("every source row must end with a newline")
    record = record[:-1]
    if record.endswith(b"\r"):
        record = record[:-1]
    if b"\r" in record or b"\n" in record:
        raise CoverageAuditError("embedded newline bytes are forbidden")
    return record


def _project_allowlisted_fields(
    normalized: bytes, *, source_file: str, source_row_index0: int
) -> tuple[bytes, bytes, bytes]:
    if b'"' in normalized:
        raise CoverageAuditError(
            "quoted records are unsupported by the opaque projection audit"
        )
    match = ALLOWLISTED_ROW_PROJECTION_RE.fullmatch(normalized)
    if match is None:
        raise CoverageAuditError(
            f"row width mismatch in {source_file} at source_row_index0="
            f"{source_row_index0}"
        )
    return match.group("date"), match.group("time_ms"), match.group("is_rand")


def _expected_dates(start: date, end: date) -> list[str]:
    values: list[str] = []
    current = start
    while current <= end:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return values


def _scan_one_file(path: Path, *, require_pinned_raw_files: bool) -> dict[str, Any]:
    name = path.name
    if name not in EVENT_FILES:
        raise CoverageAuditError(f"unexpected source file: {name}")
    policy = FILE_POLICY[name]
    if path.is_symlink() or not path.is_file():
        raise CoverageAuditError("source must be a regular non-linked file")
    initial_stat = path.stat()
    if initial_stat.st_nlink != 1:
        raise CoverageAuditError("source must have exactly one hard link")
    try:
        handle: BinaryIO = path.open("rb", buffering=8 * 1024 * 1024)
    except OSError as error:
        raise CoverageAuditError(f"cannot open {path}") from error
    digest = hashlib.sha256()
    with handle:
        header = handle.readline(MAX_HEADER_BYTES + 1)
        digest.update(header)
        if len(header) > MAX_HEADER_BYTES:
            raise CoverageAuditError("CSV header exceeds the safety limit")
        normalized_header = _strip_record_ending(header)
        expected_header = ",".join(EVENT_COLUMNS).encode("ascii")
        if normalized_header != expected_header:
            raise CoverageAuditError(f"unexpected exact CSV header in {name}")

        by_date: Counter[str] = Counter()
        by_role: Counter[str] = Counter()
        released_date_minus_utc_days: Counter[int] = Counter()
        role_min_time_ms: dict[str, int] = {}
        role_max_time_ms: dict[str, int] = {}
        row_count = 0
        min_time_ms: int | None = None
        max_time_ms: int | None = None
        first_data_byte_offset = handle.tell()
        last_data_byte_offset: int | None = None

        while record := handle.readline():
            digest.update(record)
            source_row_index0 = row_count
            last_data_byte_offset = handle.tell() - len(record)
            normalized = _strip_record_ending(record)
            date_value, time_value, is_rand = _project_allowlisted_fields(
                normalized,
                source_file=name,
                source_row_index0=source_row_index0,
            )

            # The regex traverses non-allowlisted fields without capturing or
            # materializing their byte slices. Only these three groups exist.
            iso_date = _iso_from_released_date(date_value)
            time_ms = _parse_time_ms(time_value)
            if is_rand != EXPECTED_IS_RAND[name]:
                raise CoverageAuditError(
                    f"is_rand/file-policy mismatch in {name} at "
                    f"source_row_index0={source_row_index0}"
                )
            released_calendar_date = date.fromisoformat(iso_date)
            utc_date = datetime.fromtimestamp(time_ms / 1000, tz=UTC).date()
            calendar_delta_days = (released_calendar_date - utc_date).days
            if abs(calendar_delta_days) > 1:
                raise CoverageAuditError(
                    f"date/time_ms calendar distance exceeds one day in {name} "
                    f"at source_row_index0={source_row_index0}"
                )
            role = classify_row_role(iso_date=iso_date, policy=policy)
            by_date[iso_date] += 1
            by_role[role] += 1
            released_date_minus_utc_days[calendar_delta_days] += 1
            role_min_time_ms[role] = min(role_min_time_ms.get(role, time_ms), time_ms)
            role_max_time_ms[role] = max(role_max_time_ms.get(role, time_ms), time_ms)
            min_time_ms = time_ms if min_time_ms is None else min(min_time_ms, time_ms)
            max_time_ms = time_ms if max_time_ms is None else max(max_time_ms, time_ms)
            row_count += 1

    final_stat = path.stat()
    identity_before = (
        initial_stat.st_dev,
        initial_stat.st_ino,
        initial_stat.st_size,
        initial_stat.st_mtime_ns,
    )
    identity_after = (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise CoverageAuditError(f"source changed during audit: {name}")
    observed_sha256 = digest.hexdigest()
    pinned_verified = False
    if require_pinned_raw_files:
        expected = PINNED_EVENT_FILES[name]
        if initial_stat.st_size != expected["bytes"]:
            raise CoverageAuditError(f"pinned byte-size mismatch: {name}")
        if observed_sha256 != expected["sha256"]:
            raise CoverageAuditError(f"pinned SHA-256 mismatch: {name}")
        pinned_verified = True

    if row_count == 0 or min_time_ms is None or max_time_ms is None:
        raise CoverageAuditError(f"source file is empty: {name}")
    start_date, end_date = EXPECTED_FILE_DATE_RANGES[name]
    expected_dates = _expected_dates(start_date, end_date)
    observed_dates = sorted(by_date)
    missing_dates = sorted(set(expected_dates) - set(observed_dates))
    unexpected_dates = sorted(set(observed_dates) - set(expected_dates))
    return {
        "source_file": name,
        "policy": policy,
        "required_is_rand": int(EXPECTED_IS_RAND[name]),
        "source_bytes": initial_stat.st_size,
        "source_sha256": observed_sha256,
        "pinned_source_verified": pinned_verified,
        "expected_dates": expected_dates,
        "observed_dates": observed_dates,
        "missing_dates": missing_dates,
        "unexpected_dates": unexpected_dates,
        "date_coverage_exact": not missing_dates and not unexpected_dates,
        "source_rows": row_count,
        "source_row_index0_min": 0,
        "source_row_index0_max": row_count - 1,
        "first_data_byte_offset": first_data_byte_offset,
        "last_data_byte_offset": last_data_byte_offset,
        "min_time_ms": min_time_ms,
        "max_time_ms": max_time_ms,
        "min_utc": datetime.fromtimestamp(min_time_ms / 1000, tz=UTC).isoformat(),
        "max_utc": datetime.fromtimestamp(max_time_ms / 1000, tz=UTC).isoformat(),
        "released_date_minus_utc_days": {
            str(key): value
            for key, value in sorted(released_date_minus_utc_days.items())
        },
        "role_time_ms_bounds": {
            role: {"min": role_min_time_ms[role], "max": role_max_time_ms[role]}
            for role in sorted(role_min_time_ms)
        },
        "rows_by_date": dict(sorted(by_date.items())),
        "rows_by_frozen_role": dict(sorted(by_role.items())),
    }


def _derive_embargo_cutoffs(
    role_time_ms_bounds: dict[str, dict[str, int]],
) -> dict[str, int]:
    return {
        "history_to_train": role_time_ms_bounds["train"]["min"],
        "train_to_validation": min(
            role_time_ms_bounds["validation"]["min"],
            role_time_ms_bounds["validation_state_update_only"]["min"],
        ),
        "validation_to_test": min(
            role_time_ms_bounds["headline_test"]["min"],
            role_time_ms_bounds["in_policy_test"]["min"],
        ),
    }


def _scan_embargo_effects(path: Path, *, cutoffs: dict[str, int]) -> dict[str, Any]:
    name = path.name
    policy = FILE_POLICY[name]
    initial_stat = path.stat()
    excluded: Counter[str] = Counter()
    post_embargo: Counter[str] = Counter()
    role_min_time_ms: dict[str, int] = {}
    role_max_time_ms: dict[str, int] = {}
    row_count = 0
    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        header = handle.readline(MAX_HEADER_BYTES + 1)
        if _strip_record_ending(header) != ",".join(EVENT_COLUMNS).encode("ascii"):
            raise CoverageAuditError(f"unexpected exact CSV header in {name}")
        while record := handle.readline():
            source_row_index0 = row_count
            normalized = _strip_record_ending(record)
            date_value, time_value, is_rand = _project_allowlisted_fields(
                normalized,
                source_file=name,
                source_row_index0=source_row_index0,
            )
            iso_date = _iso_from_released_date(date_value)
            time_ms = _parse_time_ms(time_value)
            if is_rand != EXPECTED_IS_RAND[name]:
                raise CoverageAuditError(
                    f"is_rand/file-policy mismatch in {name} at "
                    f"source_row_index0={source_row_index0}"
                )
            role = classify_row_role(iso_date=iso_date, policy=policy)
            if is_row_embargo_excluded(role=role, time_ms=time_ms, cutoffs=cutoffs):
                excluded[role] += 1
            else:
                post_embargo[role] += 1
                role_min_time_ms[role] = min(
                    role_min_time_ms.get(role, time_ms), time_ms
                )
                role_max_time_ms[role] = max(
                    role_max_time_ms.get(role, time_ms), time_ms
                )
            row_count += 1
    final_stat = path.stat()
    if (
        initial_stat.st_dev,
        initial_stat.st_ino,
        initial_stat.st_size,
        initial_stat.st_mtime_ns,
    ) != (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
    ):
        raise CoverageAuditError(f"source changed during embargo audit: {name}")
    return {
        "source_file": name,
        "source_rows": row_count,
        "excluded_rows_by_role": dict(sorted(excluded.items())),
        "post_embargo_rows_by_role": dict(sorted(post_embargo.items())),
        "post_embargo_role_time_ms_bounds": {
            role: {"min": role_min_time_ms[role], "max": role_max_time_ms[role]}
            for role in sorted(role_min_time_ms)
        },
    }


def _boundary_checks(
    role_time_ms_bounds: dict[str, dict[str, int]],
) -> dict[str, dict[str, int | bool]]:
    validation_max = max(
        role_time_ms_bounds["validation"]["max"],
        role_time_ms_bounds["validation_state_update_only"]["max"],
    )
    test_min = min(
        role_time_ms_bounds["headline_test"]["min"],
        role_time_ms_bounds["in_policy_test"]["min"],
    )
    transitions = (
        (
            "history_before_train",
            role_time_ms_bounds["history_burn_in"]["max"],
            role_time_ms_bounds["train"]["min"],
        ),
        (
            "train_before_validation",
            role_time_ms_bounds["train"]["max"],
            min(
                role_time_ms_bounds["validation"]["min"],
                role_time_ms_bounds["validation_state_update_only"]["min"],
            ),
        ),
        ("validation_before_test", validation_max, test_min),
    )
    return {
        name: {"earlier_max": earlier, "later_min": later, "passed": earlier < later}
        for name, earlier, later in transitions
    }


def run_label_blind_coverage_audit(
    *,
    raw_data_dir: Path,
    contract_path: Path,
    repository_root: Path,
    _allow_unpinned_fixture: bool = False,
) -> dict[str, Any]:
    """Return deterministic metadata coverage without decoding any label."""

    try:
        contract = load_feature_contract(contract_path)
        source_digest = verify_source_manifest_reference(
            contract, repository_root=repository_root
        )
    except KuaiRandContractError as error:
        raise CoverageAuditError(str(error)) from error
    if not raw_data_dir.is_dir() or raw_data_dir.is_symlink():
        raise CoverageAuditError("raw_data_dir must be a real directory")
    require_pinned_raw_files = not _allow_unpinned_fixture
    files = [
        _scan_one_file(
            raw_data_dir / name,
            require_pinned_raw_files=require_pinned_raw_files,
        )
        for name in EVENT_FILES
    ]
    total_rows = sum(int(record["source_rows"]) for record in files)
    pre_embargo_roles: Counter[str] = Counter()
    role_time_ms_bounds: dict[str, dict[str, int]] = {}
    for record in files:
        pre_embargo_roles.update(record["rows_by_frozen_role"])
        for role, bounds in record["role_time_ms_bounds"].items():
            if role not in role_time_ms_bounds:
                role_time_ms_bounds[role] = dict(bounds)
            else:
                role_time_ms_bounds[role]["min"] = min(
                    role_time_ms_bounds[role]["min"], bounds["min"]
                )
                role_time_ms_bounds[role]["max"] = max(
                    role_time_ms_bounds[role]["max"], bounds["max"]
                )
    required_nonempty_roles = {
        "history_burn_in",
        "train",
        "validation",
        "validation_state_update_only",
        "headline_test",
        "in_policy_test",
    }
    missing = sorted(
        role for role in required_nonempty_roles if not pre_embargo_roles[role]
    )
    if missing:
        raise CoverageAuditError(f"frozen roles have zero metadata rows: {missing}")

    derived_cutoffs = _derive_embargo_cutoffs(role_time_ms_bounds)
    embargo_files = [
        _scan_embargo_effects(raw_data_dir / name, cutoffs=derived_cutoffs)
        for name in EVENT_FILES
    ]
    embargo_excluded: Counter[str] = Counter()
    post_embargo_roles: Counter[str] = Counter()
    post_embargo_bounds: dict[str, dict[str, int]] = {}
    for first_pass, second_pass in zip(files, embargo_files, strict=True):
        if first_pass["source_rows"] != second_pass["source_rows"]:
            raise CoverageAuditError(
                f"source row count changed between passes: {first_pass['source_file']}"
            )
        embargo_excluded.update(second_pass["excluded_rows_by_role"])
        post_embargo_roles.update(second_pass["post_embargo_rows_by_role"])
        for role, bounds in second_pass["post_embargo_role_time_ms_bounds"].items():
            if role not in post_embargo_bounds:
                post_embargo_bounds[role] = dict(bounds)
            else:
                post_embargo_bounds[role]["min"] = min(
                    post_embargo_bounds[role]["min"], bounds["min"]
                )
                post_embargo_bounds[role]["max"] = max(
                    post_embargo_bounds[role]["max"], bounds["max"]
                )
    if sum(pre_embargo_roles.values()) != (
        sum(post_embargo_roles.values()) + sum(embargo_excluded.values())
    ):
        raise CoverageAuditError("embargo row-balance invariant failed")

    pre_boundary_checks = _boundary_checks(role_time_ms_bounds)
    post_boundary_checks = _boundary_checks(post_embargo_bounds)
    date_coverage_gate_passed = all(
        bool(record["date_coverage_exact"]) for record in files
    )
    post_boundary_gate_passed = all(
        bool(check["passed"]) for check in post_boundary_checks.values()
    )
    blockers: list[str] = []
    for record in files:
        for missing_date in record["missing_dates"]:
            blockers.append(f"missing_date:{record['source_file']}:{missing_date}")
        for unexpected_date in record["unexpected_dates"]:
            blockers.append(
                f"unexpected_date:{record['source_file']}:{unexpected_date}"
            )
    if not post_boundary_gate_passed:
        blockers.append("post_embargo_time_ms_split_overlap")

    frozen_evidence_matches = True
    if require_pinned_raw_files:
        evidence_checks = {
            "total_source_rows": total_rows == EXPECTED_TOTAL_SOURCE_ROWS,
            "pre_embargo_rows_by_role": dict(sorted(pre_embargo_roles.items()))
            == EXPECTED_PRE_EMBARGO_ROLE_COUNTS,
            "derived_embargo_cutoffs": derived_cutoffs == EXPECTED_EMBARGO_CUTOFFS,
            "embargo_excluded_rows_by_role": dict(sorted(embargo_excluded.items()))
            == EXPECTED_EMBARGO_EXCLUDED_COUNTS,
            "post_embargo_rows_by_role": dict(sorted(post_embargo_roles.items()))
            == EXPECTED_POST_EMBARGO_ROLE_COUNTS,
            "pinned_source_files": all(
                bool(record["pinned_source_verified"]) for record in files
            ),
        }
        for name, passed in evidence_checks.items():
            if not passed:
                blockers.append(f"frozen_evidence_mismatch:{name}")
        frozen_evidence_matches = all(evidence_checks.values())
    else:
        evidence_checks = {"fixture_mode_skips_frozen_real_counts": True}

    return {
        "schema_version": 1,
        "dataset_id": "kuairand_pure",
        "audit_role": "label_blind_schema_date_policy_embargo_coverage",
        "contract_sha256": contract_path.with_name(f"{contract_path.name}.sha256")
        .read_text(encoding="ascii")
        .split()[0],
        "source_manifest_sha256": source_digest,
        "decoded_raw_columns": ["date", "time_ms", "is_rand"],
        "retained_audit_coordinates": ["source_file", "source_row_index0"],
        "labels_decoded_or_aggregated": False,
        "model_features_decoded_or_aggregated": False,
        "validation_or_test_metrics_read": False,
        "test_labels_opened": False,
        "final_test_view_materialized": False,
        "test_unlocked": False,
        "actual_burn_in_dates": ["2022-04-09", "2022-04-21"],
        "april_08_rows_available": False,
        "synthetic_or_imputed_history_rows_created": False,
        "date_coverage_gate_passed": date_coverage_gate_passed,
        "pre_embargo_time_boundary_gate_passed": all(
            bool(check["passed"]) for check in pre_boundary_checks.values()
        ),
        "post_embargo_time_boundary_gate_passed": post_boundary_gate_passed,
        "time_boundary_gate_passed": post_boundary_gate_passed,
        "frozen_real_evidence_checked": require_pinned_raw_files,
        "frozen_real_evidence_matches": frozen_evidence_matches,
        "frozen_evidence_checks": evidence_checks,
        "phase1_gate_passed": not blockers,
        "blockers": blockers,
        "total_source_rows": total_rows,
        "pre_embargo_rows_by_frozen_role": dict(sorted(pre_embargo_roles.items())),
        "embargo_cutoffs_time_ms": derived_cutoffs,
        "embargo_rule": (
            "exclude earlier-role row iff time_ms >= minimum time_ms of "
            "the next-role union; retain every later-role row"
        ),
        "embargo_excluded_rows_by_role": dict(sorted(embargo_excluded.items())),
        "total_embargo_excluded_rows": sum(embargo_excluded.values()),
        "post_embargo_rows_by_frozen_role": dict(sorted(post_embargo_roles.items())),
        "role_time_ms_bounds_pre_embargo": dict(sorted(role_time_ms_bounds.items())),
        "role_time_ms_bounds_post_embargo": dict(sorted(post_embargo_bounds.items())),
        "time_ms_boundary_checks_pre_embargo": pre_boundary_checks,
        "time_ms_boundary_checks_post_embargo": post_boundary_checks,
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_label_blind_coverage_audit(
            raw_data_dir=args.raw_data_dir,
            contract_path=args.contract,
            repository_root=args.repository_root,
        )
    except CoverageAuditError as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 2
    ok = bool(report["phase1_gate_passed"])
    print(json.dumps({"ok": ok, "audit": report}, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
