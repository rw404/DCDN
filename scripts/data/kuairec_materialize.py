#!/usr/bin/env python
"""Materialize KuaiRec big-matrix runtime views for the single-head study.

Turns the filtered big matrix plus the lagged daily aggregates into the arrays
the runner consumes: ``(categorical, P, V, E, labels)`` per split, with a
manifest recording every frozen hash the contract commits to.

The contract this obeys is `configs/data/kuairec/features_v1.yaml`; the filter
and split decisions were audited label-blind beforehand and frozen in
`configs/data/kuairec/blind_scan_v1.yaml`. Neither file is modified here.

Three things carry the leakage risk and are handled explicitly.

**The daily join must be strictly lagged.** A row dated ``d`` may see the
cumulative daily aggregates through the latest day ``< d`` for its video, never
``d`` itself. The contract states this as
``daily_date_strictly_less_than_event_date`` with ``same_day_allowed: false``,
and it is enforced by a ``searchsorted`` on ``side="left"`` against the per-video
date array — the same construction the KuaiRand materializer uses for its
rolling windows.

**The current row's outcome is a label, never a feature.** ``play_duration`` and
``watch_ratio`` are current-row outcomes. ``watch_ratio`` is never read at all;
``play_duration`` is read only to build the label. The contract records both as
``watch_ratio_as_input: false`` and ``current_play_duration_as_input: false``.

**Everything fitted is fitted on train only.** Unlike KuaiRand, whose contract
fixes data-independent scales, KuaiRec normalizes by ``max(train q99, 1.0)`` per
column and maps categoricals from train-only vocabularies. Both are computed
from the filtered train rows and frozen before any validation feature is built.

The sealed test split is not materialized. Its rows are counted and dropped.

Usage::

    python scripts/data/kuairec_materialize.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dcdn.data.kuairec import (  # noqa: E402
    BIG_MATRIX_SHA256,
    DAILY_BASE_SOURCES,
    DATASET_ID,
    ORDERED_C_FIELDS,
    ORDERED_DENOMINATOR_MASK,
    ORDERED_E_FIELDS,
    ORDERED_NUMERIC_OPERANDS,
    ORDERED_P_FIELDS,
    ORDERED_V_FIELDS,
    RATE_FAMILIES,
    split_boundaries,
)

CONTRACT = REPO_ROOT / "configs" / "data" / "kuairec" / "features_v1.yaml"
BLIND_SCAN = REPO_ROOT / "configs" / "data" / "kuairec" / "blind_scan_v1.yaml"

#: Only these columns of the big matrix are read. `watch_ratio` is absent by
#: design: it is `play_duration / video_duration`, a label-derived outcome, and
#: the contract forbids reading it.
BIG_MATRIX_COLUMNS = (
    "user_id", "video_id", "play_duration", "video_duration", "date", "timestamp",
)

#: Engagement families that get `_per_play` and `_per_show` rates in E.
ENGAGEMENT_SOURCES = tuple(
    s for s in DAILY_BASE_SOURCES
    if s not in {"show_cnt", "show_user_num", "play_cnt", "play_user_num",
                 "play_duration"}
    and not any(s.startswith(f) for f in RATE_FAMILIES)
)


class MaterializationError(RuntimeError):
    """The data does not match what the frozen contract declares."""


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256_names(names: list[str]) -> str:
    payload = json.dumps(list(names), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _publish(path: Path, payload: dict[str, Any]) -> str:
    """Write atomically and hash, matching the repository's artifact rules."""

    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


# ---------------------------------------------------------------------------
# Daily history
# ---------------------------------------------------------------------------


def build_daily_prefix(daily: pl.DataFrame) -> dict[str, Any]:
    """Per-video cumulative daily sums, ready for a strictly-lagged lookup.

    Duplicate video-days are summed before the prefix, as the contract requires,
    and `play_duration` is converted from milliseconds to seconds at this point
    so every downstream consumer sees seconds.
    """

    aggregated = (
        daily.group_by(["video_id", "date"])
        .agg([pl.col(source).sum().alias(source) for source in DAILY_BASE_SOURCES])
        .sort(["video_id", "date"])
    )
    prefix = aggregated.with_columns([
        pl.col(source).cum_sum().over("video_id").alias(source)
        for source in DAILY_BASE_SOURCES
    ])

    video_ids = prefix["video_id"].to_numpy()
    dates = prefix["date"].to_numpy().astype(np.int64)
    values = np.stack(
        [prefix[source].to_numpy().astype(np.float64) for source in DAILY_BASE_SOURCES],
        axis=1,
    )
    # Row offsets so a per-video slice can be found without a Python dict lookup
    # per event row.
    unique_videos, starts = np.unique(video_ids, return_index=True)
    ends = np.append(starts[1:], len(video_ids))
    return {
        "video_ids": unique_videos,
        "starts": starts,
        "ends": ends,
        "dates": dates,
        "values": values,
        "video_days": int(len(video_ids)),
    }


def lookup_history(
    index: dict[str, Any], event_video: np.ndarray, event_date: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative daily values through the last day strictly before each event.

    Returns the ``[rows, len(DAILY_BASE_SOURCES)]`` matrix and a boolean saying
    whether any prior day existed at all. Rows whose video has no daily record,
    or none early enough, receive zeros with the flag cleared — that is what the
    contract's `v_prior_daily_history_present` is for.
    """

    rows = event_video.shape[0]
    out = np.zeros((rows, len(DAILY_BASE_SOURCES)), dtype=np.float64)
    present = np.zeros(rows, dtype=bool)

    position = np.searchsorted(index["video_ids"], event_video)
    known = (position < index["video_ids"].shape[0])
    known[known] &= index["video_ids"][position[known]] == event_video[known]

    for video_slot in np.unique(position[known]):
        selected = np.flatnonzero(known & (position == video_slot))
        start = int(index["starts"][video_slot])
        stop = int(index["ends"][video_slot])
        video_days = index["dates"][start:stop]
        # `side="left"` is the whole leakage guard: it counts days strictly
        # before the event date and excludes the event's own day.
        offset = np.searchsorted(video_days, event_date[selected], side="left")
        has_prior = offset > 0
        chosen = selected[has_prior]
        if chosen.size:
            out[chosen] = index["values"][start + offset[has_prior] - 1]
            present[chosen] = True
    return out, present


# ---------------------------------------------------------------------------
# Feature algebra, vectorized
# ---------------------------------------------------------------------------


def _rate(numerator: np.ndarray, shifted_denominator: np.ndarray) -> np.ndarray:
    """The frozen `_safe_rate`, applied to whole columns.

    Denominators are the `+1`-shifted counters, so they are positive by
    construction and no epsilon is needed.
    """

    if np.any(shifted_denominator <= 0.0):
        raise MaterializationError("shifted denominator is not strictly positive")
    return numerator / shifted_denominator


def build_features(
    duration_seconds: np.ndarray, history: np.ndarray, present: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the frozen P/V/E algebra for every row at once.

    `src/dcdn/data/kuairec.py::build_feature_values` is the per-row oracle this
    reproduces; the test suite checks the two agree row for row. It stays the
    reference because it asserts the frozen field order, which is what makes it
    worth keeping despite being far too slow for 12.26M rows.
    """

    source_index = {name: i for i, name in enumerate(DAILY_BASE_SOURCES)}
    column = {name: history[:, i] for name, i in source_index.items()}

    show = column["show_cnt"] + 1.0
    show_users = column["show_user_num"] + 1.0
    play = column["play_cnt"] + 1.0
    play_users = column["play_user_num"] + 1.0

    p_columns: list[np.ndarray] = [duration_seconds]
    for source in DAILY_BASE_SOURCES:
        p_columns.append(column[source])
    p_columns.extend([show, show_users, play, play_users])

    v_columns = [
        present.astype(np.float64),
        (column["show_cnt"] > 0.0).astype(np.float64),
        (column["show_user_num"] > 0.0).astype(np.float64),
        (column["play_cnt"] > 0.0).astype(np.float64),
        (column["play_user_num"] > 0.0).astype(np.float64),
        np.ones_like(duration_seconds),
    ]

    e_columns: list[np.ndarray] = [
        _rate(column["play_cnt"], show),
        _rate(column["play_user_num"], show_users),
    ]
    for family in RATE_FAMILIES:
        e_columns.append(_rate(column[f"{family}_cnt"], play))
        e_columns.append(_rate(column[f"{family}_cnt"], show))
    for family in RATE_FAMILIES:
        e_columns.append(_rate(column[f"{family}_user_num"], play_users))
        e_columns.append(_rate(column[f"{family}_user_num"], show_users))
    e_columns.append(_rate(column["play_duration"], play))
    e_columns.append(_rate(column["play_duration"], show))
    for source in ENGAGEMENT_SOURCES:
        denominator = show_users if source.endswith("_user_num") else show
        e_columns.append(_rate(column[source], denominator))

    p_block = np.stack(p_columns, axis=1)
    v_block = np.stack(v_columns, axis=1)
    e_block = np.stack(e_columns, axis=1)
    for name, block, expected in (
        ("P", p_block, ORDERED_P_FIELDS),
        ("V", v_block, ORDERED_V_FIELDS),
        ("E", e_block, ORDERED_E_FIELDS),
    ):
        if block.shape[1] != len(expected):
            raise MaterializationError(
                f"{name} width {block.shape[1]} does not match the frozen "
                f"{len(expected)}"
            )
    return p_block, v_block, e_block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(os.environ.get("DCDN_DATA_CACHE", "/tmp"))
    parser.add_argument(
        "--source",
        type=Path,
        default=cache / "raw" / "kuairec" / "source-v1-c23f211c" / "KuaiRec 2.0"
        / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cache / "runtime" / "kuairec_watch_time_v1",
    )
    args = parser.parse_args(argv)

    contract = _load(CONTRACT)
    scan = _load(BLIND_SCAN)
    print(f"contract {CONTRACT.name}, blind scan {BLIND_SCAN.name}")

    print("loading big matrix (projected; watch_ratio is never read)")
    big = pl.read_csv(
        args.source / "big_matrix.csv",
        columns=list(BIG_MATRIX_COLUMNS),
    )
    raw_rows = big.height

    # The audited filter: positive play and positive duration. `video_type` is
    # not in the big matrix; the AD exclusion is applied through the daily
    # table's static video_type, which the blind scan already verified is
    # invariant per video.
    # `source_row_index0` is the zero-based data row of the released file. It is
    # both the final sort tiebreaker and the unit the frozen membership hashes
    # are computed over, so it must be attached before any filtering.
    big = big.with_row_index("source_row_index0")

    print("loading daily aggregates (needed for the AD gate, which filters first)")
    daily = pl.read_csv(
        args.source / "item_daily_features.csv",
        columns=["video_id", "date", "video_type", *DAILY_BASE_SOURCES],
        infer_schema_length=10_000,
    )
    # The contract fixes the predicate ORDER as
    # static_video_type_is_not_AD -> current_play_duration_gt_0 ->
    # current_video_duration_gt_0, and the counts are reported per stage. A
    # missing video_type is UNK and kept; only the literal "AD" is dropped.
    video_type = (
        daily.select(["video_id", "video_type"])
        .with_columns(pl.col("video_type").str.strip_chars())
        .unique()
    )
    ad_videos = video_type.filter(pl.col("video_type") == "AD")["video_id"].to_list()
    big = big.filter(~pl.col("video_id").is_in(ad_videos))
    after_non_ad = big.height
    big = big.filter(pl.col("play_duration") > 0)
    after_positive_play = big.height
    big = big.filter(pl.col("video_duration") > 0)
    filtered_rows = big.height
    big = big.with_columns([
        (pl.col("play_duration") / 1000.0).alias("play_seconds"),
        (pl.col("video_duration") / 1000.0).alias("duration_seconds"),
    ])
    print(
        f"rows raw={raw_rows:,} after_non_ad={after_non_ad:,} "
        f"after_positive_play={after_positive_play:,} filtered={filtered_rows:,}"
    )
    daily = daily.filter(pl.col("video_type") != "AD").drop("video_type")

    # `play_duration` in the daily table is milliseconds; convert once here so
    # every downstream consumer sees seconds.
    daily = daily.with_columns(
        (pl.col("play_duration") / 1000.0).alias("play_duration")
    )

    index = build_daily_prefix(daily)
    print(f"daily prefix over {index['video_days']:,} video-days")

    # The frozen sort. `timestamp` is finer than `date`, and
    # `source_row_index0` is the declared final tiebreaker; using `date` alone
    # produces a differently-sliced split with the same row counts, which is
    # exactly the kind of silent divergence the membership hashes exist to
    # catch.
    big = big.sort(["timestamp", "user_id", "video_id", "source_row_index0"])
    total = big.height
    boundaries = split_boundaries(total)
    rank = np.arange(total, dtype=np.int64)
    role = np.where(
        rank < boundaries.train_end_exclusive, 0,
        np.where(rank < boundaries.validation_end_exclusive, 1, 2),
    )
    print(
        f"split: train={int((role == 0).sum()):,} "
        f"validation={int((role == 1).sum()):,} "
        f"sealed_test={int((role == 2).sum()):,} (never materialized)"
    )

    event_video = big["video_id"].to_numpy()
    event_date = big["date"].to_numpy().astype(np.int64)
    duration_seconds = big["duration_seconds"].to_numpy().astype(np.float64)

    print("joining strictly-lagged daily history")
    history, present = lookup_history(index, event_video, event_date)
    print(f"rows with prior daily history: {present.mean():.4f}")

    p_block, v_block, e_block = build_features(duration_seconds, history, present)

    # ---------------------------------------------------------------------
    # Train-only fits
    # ---------------------------------------------------------------------
    train = role == 0
    if not train.any():
        raise MaterializationError("no train rows")

    numeric = np.concatenate([p_block, v_block], axis=1)
    scale_source = np.concatenate([p_block, e_block], axis=1)
    scales = np.maximum(
        np.quantile(np.abs(scale_source[train]), 0.99, axis=0, method="linear"), 1.0
    )
    p_scale = scales[: p_block.shape[1]]
    e_scale = scales[p_block.shape[1]:]
    p_block = p_block / p_scale
    e_block = e_block / e_scale
    numeric = np.concatenate([p_block, v_block], axis=1)
    full = np.concatenate([p_block, v_block, e_block], axis=1)

    c_seconds = float(
        np.quantile(duration_seconds[train], 0.99, method="linear")
    )
    low, high = contract["label"]["c_expected_sanity_interval_seconds"]
    if not low <= c_seconds <= high:
        raise MaterializationError(
            f"train-only c={c_seconds:.3f}s falls outside the contract's sanity "
            f"interval [{low}, {high}]"
        )
    play_seconds = big["play_seconds"].to_numpy().astype(np.float64)
    watch_time = (np.clip(play_seconds, 0.0, c_seconds) / c_seconds).astype(np.float32)
    clip_incidence = float((play_seconds > c_seconds).mean())
    print(f"label c={c_seconds:.3f}s clip_incidence={clip_incidence:.5f}")

    # Second label: the canonical KuaiRec engagement target, watch_ratio > 2,
    # i.e. the user played the video through more than twice. It is derived
    # from the same current-row outcome the regression label uses, divided by
    # the video duration the model already sees as `p_video_duration_seconds`.
    # No new outcome column is read - in particular the released `watch_ratio`
    # column is still never touched, which is what the contract's
    # `watch_ratio_read: false` commits to; the ratio is recomputed from
    # play_duration and video_duration.
    watch_ratio_gt2 = (play_seconds > 2.0 * duration_seconds).astype(np.float32)
    print(
        f"label watch_ratio_gt2 positive_rate="
        f"{float(watch_ratio_gt2.mean()):.5f}"
    )
    labels = np.stack([watch_time, watch_ratio_gt2], axis=1)

    # ---------------------------------------------------------------------
    # Cold drop, then the split-membership check against the frozen audit
    # ---------------------------------------------------------------------
    # The contract distinguishes two policies that are easy to conflate:
    #   validation_test_cold_user_item_policy: inner_drop_and_report_counts
    #   category_oov_policy:                   retain_row_and_map_to_OOV
    # A validation row whose USER or VIDEO the train split never saw is
    # dropped. A row whose CATEGORY is unseen is kept and mapped to OOV.
    # Mapping unseen users or items to OOV instead would keep 84,406 rows the
    # audit removed, and would quietly change what the metric is computed over.
    train_users = np.unique(big["user_id"].to_numpy()[train])
    train_videos = np.unique(big["video_id"].to_numpy()[train])
    all_users = big["user_id"].to_numpy()
    all_videos = big["video_id"].to_numpy()
    warm = (
        np.isin(all_users, train_users) & np.isin(all_videos, train_videos)
    )
    cold_dropped = int(((role == 1) & ~warm).sum())
    print(
        f"cold drop: {cold_dropped:,} validation rows removed "
        f"-> {int(((role == 1) & warm).sum()):,} kept"
    )

    source_index = big["source_row_index0"].to_numpy().astype(np.uint64)
    membership = {}
    for name, selected in (
        ("train", role == 0),
        ("validation_after_cold_drop", (role == 1) & warm),
        ("sealed_test_after_cold_drop", (role == 2) & warm),
    ):
        payload = (
            f"{DATASET_ID}\0".encode("ascii")
            + bytes.fromhex(BIG_MATRIX_SHA256)
            + b"\0" + name.encode("ascii") + b"\0"
            + source_index[selected].astype(">u8").tobytes()
        )
        membership[name] = hashlib.sha256(payload).hexdigest()
    frozen = scan["split_membership_sha256"]
    matches = {k: membership[k] == frozen.get(k) for k in membership}
    for name, ok in matches.items():
        print(f"  membership {name:<28} {'MATCHES' if ok else 'DIFFERS'} frozen audit")

    print("fitting train-only categorical mappings")
    captions = pl.read_csv(
        args.source / "kuairec_caption_category.csv",
        columns=["video_id", "first_level_category_id", "second_level_category_id",
                 "third_level_category_id"],
        infer_schema_length=10_000,
    )
    big = big.join(captions, on="video_id", how="left")
    categorical_raw = np.stack([
        big["user_id"].to_numpy(),
        big["video_id"].to_numpy(),
        big["first_level_category_id"].fill_null(-1).to_numpy(),
        big["second_level_category_id"].fill_null(-1).to_numpy(),
        big["third_level_category_id"].fill_null(-1).to_numpy(),
    ], axis=1).astype(np.int64)

    categorical = np.zeros_like(categorical_raw)
    cardinalities: list[int] = []
    # Index 0 is the OOV slot. After the cold drop it can only ever be reached
    # by the three category columns, which the contract explicitly allows; a
    # user or video landing there would mean the drop failed.
    for column_index in range(categorical_raw.shape[1]):
        values = categorical_raw[:, column_index]
        vocabulary = np.unique(values[train])
        # Index 0 is the out-of-vocabulary slot for anything the train split
        # never saw, so validation cannot widen the embedding table.
        mapped = np.searchsorted(vocabulary, values)
        mapped = np.where(
            (mapped < vocabulary.shape[0]) & (vocabulary[np.minimum(
                mapped, vocabulary.shape[0] - 1)] == values),
            mapped + 1, 0,
        )
        categorical[:, column_index] = mapped
        cardinalities.append(int(vocabulary.shape[0]) + 1)
    print(f"cardinalities: {cardinalities}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, int] = {}
    for name, code in (("train", 0), ("validation", 1)):
        selected = (role == code) & (warm if code == 1 else True)
        split_rows[name] = int(selected.sum())
        np.savez(
            args.output_dir / f"{name}.npz",
            categorical=categorical[selected],
            P=p_block[selected].astype(np.float32),
            V=v_block[selected].astype(np.float32),
            E=e_block[selected].astype(np.float32),
            labels=labels[selected],
        )
        print(f"{name}: {split_rows[name]:,} rows -> {name}.npz  "
              f"watch_time_mean={watch_time[selected].mean():.4f}  "
              f"watch_ratio_gt2_rate={watch_ratio_gt2[selected].mean():.4f}")

    manifest = {
        "artifact_role": "kuairec_watch_time_runtime_manifest",
        "schema_version": 1,
        "dataset_id": "kuairec_big_matrix",
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "blind_scan_sha256": hashlib.sha256(BLIND_SCAN.read_bytes()).hexdigest(),
        "label_names": ["watch_time", "watch_ratio_gt2"],
        "label_kinds": ["regression", "binary"],
        "watch_ratio_gt2_definition": (
            "play_duration_seconds > 2 * video_duration_seconds"
        ),
        "watch_ratio_gt2_positive_rate": float(watch_ratio_gt2.mean()),
        "released_watch_ratio_column_read": False,
        "C_names": list(ORDERED_C_FIELDS),
        "P_names": list(ORDERED_P_FIELDS),
        "V_names": list(ORDERED_V_FIELDS),
        "E_names": list(ORDERED_E_FIELDS),
        "schema_hashes": {
            "C": _sha256_names(list(ORDERED_C_FIELDS)),
            "P": _sha256_names(list(ORDERED_P_FIELDS)),
            "V": _sha256_names(list(ORDERED_V_FIELDS)),
            "E": _sha256_names(list(ORDERED_E_FIELDS)),
            "numeric_interaction_operands": _sha256_names(
                list(ORDERED_NUMERIC_OPERANDS)
            ),
        },
        "cardinalities": cardinalities,
        "denominator_mask": [bool(x) for x in ORDERED_DENOMINATOR_MASK],
        "reduced_width": int(numeric.shape[1]),
        "full_width": int(full.shape[1]),
        "rows": {
            "raw": raw_rows,
            "after_non_ad": after_non_ad,
            "after_positive_play": after_positive_play,
            "filtered": filtered_rows,
            **split_rows,
            "validation_cold_dropped": cold_dropped,
            "sealed_test_never_materialized": int((role == 2).sum()),
        },
        "split_membership_sha256": membership,
        "split_membership_matches_frozen_audit": matches,
        "prior_daily_history_fraction": float(present.mean()),
        "watch_time_c_seconds": c_seconds,
        "watch_time_clip_incidence": clip_incidence,
        "normalization": {
            "rule": "divide_by_max(train_absolute_q99_linear, 1.0)",
            "fitted_on": "filtered_train_only",
            "applied_to": ["P", "E"],
            "identity_on": ["V"],
            "scale_sha256": hashlib.sha256(
                np.ascontiguousarray(scales, dtype=np.float64).tobytes()
            ).hexdigest(),
        },
        "test_unlocked": False,
        "sealed_test_read": False,
        "watch_ratio_read": False,
    }
    digest = _publish(args.output_dir / "manifest_v1.json", manifest)
    print(f"\nmanifest sha256={digest}")
    print(f"REDUCED numeric width={numeric.shape[1]} FULL={full.shape[1]} "
          f"denominators={sum(ORDERED_DENOMINATOR_MASK)}")
    print(f"output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
