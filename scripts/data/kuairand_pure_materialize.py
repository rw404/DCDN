#!/usr/bin/env python
"""Materialize KuaiRand-Pure multi-target REDUCED and FULL views.

Implements `configs/data/kuairand_pure/features_v1.yaml` exactly, extended to
the four mandatory heads of the 2026-08-20 owner override
(`long_view`, `like`, `forward`, `hate`).

The override changes nothing structural here: KuaiRand carries no cosine, dot
product, or vector norm, and every one of its twelve engineered formulas is a
division of historical counters or sums. So ``R_div`` is exactly the frozen
``engineered_R`` set and the denominator mask is the frozen one, unchanged.

**The locked test window (2022-05-06 .. 2022-05-08) is never read.** Its rows
are dropped by date before any label or feature is touched, so this script
cannot leak a test label even by accident. Only ``train`` and ``validation``
views are produced.

Rolling aggregates follow the frozen contract literally:

* strict cutoff ``time_ms < t`` — a row never sees itself;
* equal timestamps form one group: the whole group is scored, then the whole
  group updates state, so simultaneous events never see each other;
* windows ``1d`` (-86,400,000 ms), ``7d`` (-604,800,000 ms), and ``all``
  (since burn-in start), each over four key groups and nine statistics;
* embargoed rows never update state;
* pre-validation randomized rows contribute neither labels nor initial state;
* standard rows inside the validation window update state but carry no label.

Normalization uses only the frozen data-independent constants; nothing is
fitted from validation.

Usage::

    python scripts/data/kuairand_pure_materialize.py --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CONTRACT_PATH = REPO_ROOT / "configs" / "data" / "kuairand_pure" / "features_v1.yaml"

LOG_FILES = (
    ("log_standard_4_08_to_4_21_pure.csv", "standard"),
    ("log_standard_4_22_to_5_08_pure.csv", "standard"),
    ("log_random_4_22_to_5_08_pure.csv", "randomized"),
)
POLICY_PRIORITY = {"standard": 0, "randomized": 1}

WINDOW_MS = {"1d": 86_400_000, "7d": 604_800_000}
GROUP_ORDER = ("user", "item", "author", "tab")
WINDOW_ORDER = ("1d", "7d", "all")
STATISTIC_ORDER = (
    "exposure_count",
    "long_view_count",
    "like_count",
    "forward_count",
    "hate_count",
    "follow_count",
    "play_time_ms_sum",
    "duration_ms_sum",
    "dwell_time_ms_sum",
)

#: The five heads of the amended objective: four binary plus normalized
#: clipped watch time, matching the VK-LSVD and KuaiRec construction.
TARGETS = ("long_view", "like", "forward", "hate", "watch_time")
TARGET_KIND = {
    "long_view": "binary",
    "like": "binary",
    "forward": "binary",
    "hate": "binary",
    "watch_time": "regression",
}
TARGET_SOURCE = {
    "long_view": "long_view",
    "like": "is_like",
    "forward": "is_forward",
    "hate": "is_hate",
    "watch_time": "play_time_ms clipped at the train-only duration constant",
}

#: Locked. Rows on these dates are dropped before anything is read from them.
TEST_DATES = frozenset({"20220506", "20220507", "20220508"})

BURN_IN_DATES = frozenset(f"202204{d:02d}" for d in range(9, 22))
TRAIN_DATES = frozenset(
    ["20220422", "20220423", "20220424", "20220425", "20220426", "20220427",
     "20220428", "20220429", "20220430", "20220501", "20220502"]
)
VALIDATION_DATES = frozenset({"20220503", "20220504", "20220505"})


class MaterializationError(RuntimeError):
    """A frozen materialization contract was violated."""


@dataclass(frozen=True)
class Constants:
    binary_success_pseudocount: float
    exposure_pseudocount: float
    time_numerator_floor_ms: float
    duration_denominator_floor_ms: float
    count_scale: float
    time_scale_ms: float
    normalized_denominator_floor: float


def _load_contract() -> dict[str, Any]:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def _data_root() -> Path:
    cache = os.environ.get("DCDN_DATA_CACHE")
    if not cache:
        raise MaterializationError("DCDN_DATA_CACHE is not set; see README Datasets")
    root = Path(cache) / "raw" / "kuairand_pure" / "KuaiRand-Pure" / "data"
    if not root.is_dir():
        raise MaterializationError(f"KuaiRand-Pure raw data not found at {root}")
    return root


def _require_binary(value: str, column: str) -> int:
    if value in ("0", "0.0"):
        return 0
    if value in ("1", "1.0"):
        return 1
    raise MaterializationError(f"non-binary value {value!r} in {column}")


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise MaterializationError("non-finite numeric field")
    return max(number, 0.0)


def load_events(root: Path, cutoffs: dict[str, int]) -> dict[str, np.ndarray]:
    """Read the three logs, drop the locked test window, and order globally."""

    rows: list[tuple] = []
    dropped_test = 0
    for file_index, (name, policy) in enumerate(LOG_FILES):
        with (root / name).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for source_index, record in enumerate(reader):
                date = record["date"]
                if date in TEST_DATES:
                    # Locked window: skipped before any label or field is used.
                    dropped_test += 1
                    continue
                rows.append(
                    (
                        int(record["time_ms"]),
                        POLICY_PRIORITY[policy],
                        file_index,
                        source_index,
                        int(record["user_id"]),
                        int(record["video_id"]),
                        record["tab"],
                        date,
                        _require_binary(record["long_view"], "long_view"),
                        _require_binary(record["is_like"], "is_like"),
                        _require_binary(record["is_forward"], "is_forward"),
                        _require_binary(record["is_hate"], "is_hate"),
                        _require_binary(record["is_follow"], "is_follow"),
                        _nonnegative_float(record["play_time_ms"]),
                        _nonnegative_float(record["duration_ms"]),
                        _nonnegative_float(record["profile_stay_time"])
                        + _nonnegative_float(record["comment_stay_time"]),
                    )
                )
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    print(f"  rows kept={len(rows):,}  locked-test rows never read={dropped_test:,}")

    columns = list(zip(*rows, strict=True))
    events = {
        "time_ms": np.asarray(columns[0], dtype=np.int64),
        "policy_priority": np.asarray(columns[1], dtype=np.int8),
        "source_file": np.asarray(columns[2], dtype=np.int8),
        "source_row_index0": np.asarray(columns[3], dtype=np.int64),
        "user_id": np.asarray(columns[4], dtype=np.int64),
        "video_id": np.asarray(columns[5], dtype=np.int64),
        "tab": np.asarray(columns[6], dtype=object),
        "date": np.asarray(columns[7], dtype=object),
    }
    stats = np.empty((len(rows), len(STATISTIC_ORDER)), dtype=np.float64)
    stats[:, 0] = 1.0
    for offset, key in enumerate(
        ("long_view", "like", "forward", "hate", "follow"), start=1
    ):
        del key
        stats[:, offset] = np.asarray(columns[7 + offset], dtype=np.float64)
    stats[:, 6] = np.asarray(columns[13], dtype=np.float64)
    stats[:, 7] = np.asarray(columns[14], dtype=np.float64)
    stats[:, 8] = np.asarray(columns[15], dtype=np.float64)
    events["stats"] = stats
    events["role"] = _classify_roles(events, cutoffs)
    return events


def _classify_roles(events: dict[str, np.ndarray], cutoffs: dict[str, int]
                    ) -> np.ndarray:
    """Assign each row a role, applying the owner-approved boundary embargo."""

    time_ms = events["time_ms"]
    randomized = events["policy_priority"] == POLICY_PRIORITY["randomized"]
    date = events["date"]
    in_burn_in = np.fromiter((d in BURN_IN_DATES for d in date), bool, len(date))
    in_train = np.fromiter((d in TRAIN_DATES for d in date), bool, len(date))
    in_validation = np.fromiter(
        (d in VALIDATION_DATES for d in date), bool, len(date)
    )

    role = np.full(len(date), "excluded", dtype=object)
    # Burn-in: standard only, and embargoed at the train boundary.
    role[in_burn_in & ~randomized & (time_ms < cutoffs["history_to_train"])] = (
        "history_burn_in"
    )
    # Train: standard only, embargoed at the validation boundary.
    role[in_train & ~randomized & (time_ms < cutoffs["train_to_validation"])] = "train"
    # Validation: the randomized union; standard rows in the window update state.
    role[in_validation & randomized & (time_ms < cutoffs["validation_to_test"])] = (
        "validation"
    )
    role[in_validation & ~randomized & (time_ms < cutoffs["validation_to_test"])] = (
        "state_only"
    )
    # Randomized rows before validation contribute neither label nor state.
    return role


def rolling_windows(
    events: dict[str, np.ndarray], key: np.ndarray, updates_state: np.ndarray
) -> dict[str, np.ndarray]:
    """Strictly lagged 1d / 7d / all sums for one key group, fully vectorized.

    Rows that do not update state are excluded from the accumulators but still
    receive features, so an embargoed or label-less row can never contribute to
    another row's history.
    """

    time_ms = events["time_ms"]
    stats = events["stats"]
    count = len(time_ms)
    contribution = np.where(updates_state[:, None], stats, 0.0)

    order = np.argsort(key, kind="stable")
    sorted_keys = key[order]
    sorted_times = time_ms[order]
    prefix = np.zeros((count + 1, stats.shape[1]), dtype=np.float64)
    np.cumsum(contribution[order], axis=0, out=prefix[1:])

    boundaries = np.flatnonzero(
        np.r_[True, sorted_keys[1:] != sorted_keys[:-1], True]
    )
    result = {
        window: np.empty((count, stats.shape[1]), dtype=np.float64)
        for window in WINDOW_ORDER
    }
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        group_times = sorted_times[start:end]
        # Strict cutoff plus the equal-timestamp rule in one step: a row sees
        # only rows whose time is strictly smaller, so simultaneous events are
        # invisible to each other.
        high = start + np.searchsorted(group_times, group_times, side="left")
        base = prefix[high]
        result["all"][order[start:end]] = base - prefix[start]
        for window, span in WINDOW_MS.items():
            low = start + np.searchsorted(
                group_times, group_times - span, side="left"
            )
            result[window][order[start:end]] = base - prefix[low]
    return result


def build_views(
    events: dict[str, np.ndarray], contract: dict[str, Any]
) -> dict[str, Any]:
    """Assemble ordered C, P, V, and R blocks exactly as the contract orders them."""

    constants = Constants(**contract["smoothing_and_normalization"]["constants"])
    updates_state = np.isin(
        events["role"].astype(str), ("history_burn_in", "train", "validation",
                                     "state_only")
    )

    root = _data_root()
    author_of: dict[int, str] = {}
    meta_fields: dict[str, dict[int, str]] = {
        name: {} for name in ("video_type", "upload_type", "music_id", "music_type")
    }
    with (root / "video_features_basic_pure.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for record in csv.DictReader(handle):
            video = int(record["video_id"])
            author_of[video] = record["author_id"]
            for name in meta_fields:
                meta_fields[name][video] = record[name]

    video = events["video_id"]
    author_key = np.fromiter(
        (author_of.get(int(v), "MISSING") for v in video), object, len(video)
    )
    keys = {
        "user": events["user_id"],
        "item": video,
        "author": author_key,
        "tab": events["tab"],
    }

    raw: dict[str, dict[str, np.ndarray]] = {}
    for group in GROUP_ORDER:
        print(f"  rolling aggregates: {group}")
        raw[group] = rolling_windows(events, keys[group], updates_state)

    p_blocks: list[np.ndarray] = []
    v_blocks: list[np.ndarray] = []
    p_names: list[str] = []
    v_names: list[str] = []
    denominator_flags: list[bool] = []
    eligible = set(contract["denominator_mask"]["eligible_fields"])

    for group in GROUP_ORDER:
        for window in WINDOW_ORDER:
            block = raw[group][window]
            for index, statistic in enumerate(STATISTIC_ORDER):
                values = block[:, index]
                if statistic == "exposure_count":
                    scaled = (values + constants.exposure_pseudocount) / (
                        constants.count_scale
                    )
                elif statistic.endswith("_count"):
                    scaled = (values + constants.binary_success_pseudocount) / (
                        constants.count_scale
                    )
                elif statistic == "duration_ms_sum":
                    scaled = (
                        values + constants.duration_denominator_floor_ms
                    ) / constants.time_scale_ms
                else:
                    scaled = (
                        values + constants.time_numerator_floor_ms
                    ) / constants.time_scale_ms
                name = f"p_{group}_{window}_{statistic}"
                is_denominator = name in eligible
                if is_denominator:
                    scaled = np.maximum(
                        scaled, constants.normalized_denominator_floor
                    )
                p_blocks.append(scaled.astype(np.float32))
                p_names.append(name)
                denominator_flags.append(is_denominator)
            exposure = block[:, 0]
            duration = block[:, 7]
            v_blocks.append((exposure > 0).astype(np.float32))
            v_names.append(f"v_{group}_{window}_exposure_valid")
            v_blocks.append(((exposure > 0) & (duration > 0)).astype(np.float32))
            v_names.append(f"v_{group}_{window}_duration_valid")

    r_blocks: list[np.ndarray] = []
    r_names: list[str] = []
    lineage: dict[str, tuple[str, str]] = {}
    for group in GROUP_ORDER:
        for window in WINDOW_ORDER:
            block = raw[group][window]
            exposure = block[:, 0]
            for index, statistic in enumerate(
                ("long_view", "like", "forward", "hate", "follow"), start=1
            ):
                name = f"r_{group}_{window}_{statistic}_per_exposure"
                r_blocks.append(
                    (
                        (block[:, index] + 0.5) / (exposure + 1.0)
                    ).astype(np.float32)
                )
                r_names.append(name)
                lineage[name] = (
                    f"p_{group}_{window}_{statistic}_count",
                    f"p_{group}_{window}_exposure_count",
                )
            r_blocks.append(
                (((block[:, 6] + 1.0) / (exposure + 1.0)) / 1000.0).astype(np.float32)
            )
            r_names.append(f"r_{group}_{window}_mean_play_time_seconds")
            lineage[r_names[-1]] = (
                f"p_{group}_{window}_play_time_ms_sum",
                f"p_{group}_{window}_exposure_count",
            )
            r_blocks.append(
                ((block[:, 6] + 1.0) / (block[:, 7] + 1000.0)).astype(np.float32)
            )
            r_names.append(f"r_{group}_{window}_play_over_duration")
            lineage[r_names[-1]] = (
                f"p_{group}_{window}_play_time_ms_sum",
                f"p_{group}_{window}_duration_ms_sum",
            )
            r_blocks.append(
                (((block[:, 8] + 1.0) / (exposure + 1.0)) / 1000.0).astype(np.float32)
            )
            r_names.append(f"r_{group}_{window}_mean_dwell_time_seconds")
            lineage[r_names[-1]] = (
                f"p_{group}_{window}_dwell_time_ms_sum",
                f"p_{group}_{window}_exposure_count",
            )

        # The contract's four cross-window ratios, one set per key group. They
        # are divisions of historical counters like every other R field, so
        # they belong to R_div and are removed in the reduced view.
        one_day, seven_day, all_time = (
            raw[group]["1d"], raw[group]["7d"], raw[group]["all"]
        )
        for name, numerator, denominator, pad, source in (
            ("exposure_1d_over_7d", one_day[:, 0], seven_day[:, 0], 1.0,
             ("1d_exposure_count", "7d_exposure_count")),
            ("exposure_7d_over_all", seven_day[:, 0], all_time[:, 0], 1.0,
             ("7d_exposure_count", "all_exposure_count")),
            ("long_view_1d_over_7d", one_day[:, 1], seven_day[:, 1], 0.5,
             ("1d_long_view_count", "7d_long_view_count")),
            ("long_view_7d_over_all", seven_day[:, 1], all_time[:, 1], 0.5,
             ("7d_long_view_count", "all_long_view_count")),
        ):
            r_blocks.append(
                ((numerator + pad) / (denominator + pad)).astype(np.float32)
            )
            r_names.append(f"r_{group}_{name}")
            lineage[r_names[-1]] = (
                f"p_{group}_{source[0]}",
                f"p_{group}_{source[1]}",
            )

    # The emission loops build R group-major (each key group's per-window
    # ratios, then that group's four cross-window ratios), but the frozen
    # contract declares R field-block-major: every group's per-window ratios
    # first, then every group's cross-window ratios. The two are permutations
    # of the same 112 names, and the models are permutation-invariant over
    # concatenated numeric columns, so no metric depends on this. The contract
    # order is nonetheless the normative one - it is what `schema_hashes.R`
    # commits to and what any auditor reading the frozen file will assume - so
    # the block is reordered to match it here rather than left to drift.
    r_blocks, r_names = _reorder_to_contract(r_blocks, r_names, contract)

    categorical, cardinalities, categorical_names = _encode_categorical(
        events, author_key, meta_fields
    )
    # Normalized clipped watch time, built exactly as VK-LSVD and KuaiRec build
    # theirs: clip the current-row watch time at a constant fitted on TRAIN
    # rows only, then divide by it. play_time_ms and duration_ms are current-row
    # outcomes used as label inputs only, never as features of their own row.
    play_seconds = events["stats"][:, 6] / 1000.0
    duration_seconds = events["stats"][:, 7] / 1000.0
    train_rows = events["role"].astype(str) == "train"
    if not train_rows.any():
        raise MaterializationError("no train rows; cannot fit the watch-time constant")
    c_seconds = float(np.quantile(duration_seconds[train_rows], 0.99))
    if not np.isfinite(c_seconds) or c_seconds <= 0.0:
        raise MaterializationError("train duration q0.99 is not positive and finite")
    watch_time = (np.clip(play_seconds, 0.0, c_seconds) / c_seconds).astype(np.float32)
    events["label_watch_time"] = watch_time
    labels = np.stack(
        [events[f"label_{name}"] for name in TARGETS], axis=1
    ).astype(np.float32)

    return {
        "categorical": categorical,
        "categorical_names": categorical_names,
        "cardinalities": cardinalities,
        "P": np.stack(p_blocks, axis=1),
        "P_names": p_names,
        "V": np.stack(v_blocks, axis=1),
        "V_names": v_names,
        "R": np.stack(r_blocks, axis=1),
        "R_names": r_names,
        "R_lineage": lineage,
        "denominator_flags": denominator_flags,
        "labels": labels,
        "role": events["role"].astype(str),
        "constants": constants,
        "watch_time_c_seconds": c_seconds,
        "watch_time_clip_incidence": float(
            (play_seconds > c_seconds).mean()
        ),
    }


def _reorder_to_contract(
    blocks: list[np.ndarray], names: list[str], contract: dict
) -> tuple[list[np.ndarray], list[str]]:
    """Permute the R block into the order the frozen contract declares.

    Raises rather than silently truncating or padding: a name the contract does
    not list, or a contract name nothing emitted, is a contract violation and
    must stop the materialization.
    """

    expected = list(contract["ordered_fields"]["R"])
    if set(expected) != set(names):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise MaterializationError(
            f"R does not match the frozen contract; missing={missing} extra={extra}"
        )
    position = {name: index for index, name in enumerate(names)}
    return [blocks[position[name]] for name in expected], expected


def _encode_categorical(
    events: dict[str, np.ndarray],
    author_key: np.ndarray,
    meta_fields: dict[str, dict[int, str]],
) -> tuple[np.ndarray, list[int], list[str]]:
    """Fit categorical maps on embargo-eligible standard rows only."""

    role = events["role"].astype(str)
    fit_rows = np.isin(role, ("history_burn_in", "train"))
    video = events["video_id"]
    raw_columns = {
        "c_user_id": events["user_id"].astype(str),
        "c_video_id": video.astype(str),
        "c_author_id": author_key.astype(str),
        "c_tab": events["tab"].astype(str),
    }
    for name in ("video_type", "upload_type", "music_id", "music_type"):
        raw_columns[f"c_{name}"] = np.fromiter(
            (meta_fields[name].get(int(v), "\x00MISSING") for v in video),
            object,
            len(video),
        ).astype(str)

    names = [
        "c_user_id", "c_video_id", "c_author_id", "c_tab",
        "c_video_type", "c_upload_type", "c_music_id", "c_music_type",
    ]
    encoded = np.zeros((len(video), len(names)), dtype=np.int64)
    cardinalities: list[int] = []
    for index, name in enumerate(names):
        column = raw_columns[name]
        known = sorted({v for v in column[fit_rows] if v != "\x00MISSING"})
        mapping = {value: position + 2 for position, value in enumerate(known)}
        encoded[:, index] = np.fromiter(
            (1 if v == "\x00MISSING" else mapping.get(v, 0) for v in column),
            np.int64,
            len(column),
        )
        cardinalities.append(len(known) + 2)
    return encoded, cardinalities, names


def _sha256_names(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("DCDN_DATA_CACHE", "/tmp"))
        / "runtime"
        / "kuairand_pure_multitask_v1",
    )
    args = parser.parse_args(argv)

    contract = _load_contract()
    cutoffs = contract["approved_temporal_policy"]["embargo"]["cutoffs_time_ms"]
    print("loading events (locked test window excluded by date)")
    events = load_events(_data_root(), cutoffs)
    # Labels come from the same ordered arrays the stats were built from.
    stats = events["stats"]
    events["label_long_view"] = stats[:, 1]
    events["label_like"] = stats[:, 2]
    events["label_forward"] = stats[:, 3]
    events["label_hate"] = stats[:, 4]

    roles, counts = np.unique(events["role"].astype(str), return_counts=True)
    print(
        "  roles: "
        + "  ".join(f"{r}={c:,}" for r, c in zip(roles, counts, strict=True))
    )

    print("building views")
    views = build_views(events, contract)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    role = views["role"]
    manifest: dict[str, Any] = {
        "artifact_role": "kuairand_pure_multitask_view_manifest",
        "schema_version": 1,
        "dataset_id": "kuairand_pure",
        "targets": list(TARGETS),
        "target_kinds": TARGET_KIND,
        "watch_time_c_seconds": views["watch_time_c_seconds"],
        "watch_time_c_provenance": (
            "train_split_duration_ms_q0.99_seconds; fitted on train rows only, "
            "never on validation or the locked test window"
        ),
        "watch_time_clip_incidence": views["watch_time_clip_incidence"],
        "target_sources": TARGET_SOURCE,
        "locked_test_window_read": False,
        "locked_test_dates": sorted(TEST_DATES),
        "embargo_cutoffs_time_ms": cutoffs,
        "categorical_names": views["categorical_names"],
        "cardinalities": views["cardinalities"],
        "P_names": views["P_names"],
        "V_names": views["V_names"],
        "R_names": views["R_names"],
        "R_lineage": {k: list(v) for k, v in views["R_lineage"].items()},
        "denominator_mask": views["denominator_flags"],
        "denominator_count": int(sum(views["denominator_flags"])),
        "reduced_numeric_width": len(views["P_names"]) + len(views["V_names"]),
        "full_numeric_width": (
            len(views["P_names"]) + len(views["V_names"]) + len(views["R_names"])
        ),
        "schema_hashes": {
            "C": _sha256_names(views["categorical_names"]),
            "P": _sha256_names(views["P_names"]),
            "V": _sha256_names(views["V_names"]),
            "R": _sha256_names(views["R_names"]),
        },
        "splits": {},
    }

    for split in ("train", "validation"):
        selector = role == split
        payload = {
            "categorical": views["categorical"][selector],
            "P": views["P"][selector],
            "V": views["V"][selector],
            "R": views["R"][selector],
            "labels": views["labels"][selector],
        }
        path = args.output_dir / f"{split}.npz"
        np.savez(path, **payload)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["splits"][split] = {
            "rows": int(selector.sum()),
            "path": path.name,
            "sha256": digest,
            "positive_rate": {
                name: float(payload["labels"][:, i].mean())
                for i, name in enumerate(TARGETS)
            },
        }
        print(
            f"  {split}: {int(selector.sum()):,} rows -> {path.name}  "
            + "  ".join(
                f"{n}={payload['labels'][:, i].mean():.4f}"
                for i, n in enumerate(TARGETS)
            )
        )

    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "manifest_v1.json").write_text(text, encoding="utf-8")
    (args.output_dir / "manifest_v1.json.sha256").write_text(
        f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}  manifest_v1.json\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.output_dir}")
    print(f"REDUCED numeric width={manifest['reduced_numeric_width']} "
          f"FULL={manifest['full_numeric_width']} "
          f"denominators={manifest['denominator_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
