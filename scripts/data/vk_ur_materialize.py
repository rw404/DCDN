#!/usr/bin/env python
"""Materialize the VK-LSVD ur0.01_ir0.01 views from the frozen contract.

Everything here follows `configs/data/vk_lsvd_ur/features_v1.yaml`, frozen at
sha256 92b23a69b097fa00450bff5513b870168d8936881ef2fa6c3c82dcceb1059708 before
this file was written.

Temporal safety is structural rather than checked afterwards: aggregates for a
row in week W are built from rows with week strictly less than W, so a row
cannot contribute to its own features and never sees any part of its own week.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "configs" / "data" / "vk_lsvd_ur" / "features_v1.yaml"
SUBSET = "subsamples/ur0.01_ir0.01"
ACTIONS = ("like", "dislike", "share", "bookmark", "click_on_author", "open_comments")
GROUPS = ("user", "item", "user_place")
WINDOWS = ("w1", "w4", "all")
COLUMNS = ["user_id", "item_id", "place", "platform", "agent", "timespent", *ACTIONS]


def load_weeks(root: Path, weeks: range) -> dict[str, np.ndarray]:
    parts: list[dict] = []
    for week in weeks:
        split = "train" if week <= 24 else "validation"
        path = root / SUBSET / split / f"week_{week:02d}.parquet"
        table = pq.read_table(path, columns=COLUMNS)
        block = {name: table.column(name).to_numpy() for name in COLUMNS}
        block["week"] = np.full(table.num_rows, week, dtype=np.int16)
        parts.append(block)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def factorize(*keys: np.ndarray) -> tuple[np.ndarray, int]:
    """Dense codes for a composite key, so bincount can be used as the group-by."""
    packed = keys[0].astype(np.int64)
    for key in keys[1:]:
        packed = packed * 1000 + key.astype(np.int64)
    unique, codes = np.unique(packed, return_inverse=True)
    return codes, unique.size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(os.environ.get("DCDN_DATA_CACHE", "/tmp"))
    parser.add_argument("--raw", type=Path, default=cache / "raw" / "vk_lsvd")
    parser.add_argument("--output", type=Path,
                        default=cache / "runtime" / "vk_lsvd_ur_v1")
    args = parser.parse_args(argv)

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    digest = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    expected = (REPO_ROOT / "configs/data/vk_lsvd_ur/features_v1.yaml.sha256"
                ).read_text().split()[0]
    if digest != expected:
        raise SystemExit(f"contract changed since freezing: {digest} != {expected}")
    print(f"contract verified {digest[:16]}")

    alpha = contract["smoothing"]["alpha"]
    beta = contract["smoothing"]["beta"]
    # Counts are smoothed by alpha; timespent sums by beta pseudo-impressions
    # at the prior mean, so the no-history value sits on the scale of the
    # quantity being smoothed rather than a thousandth of it.
    alpha_timespent = contract["smoothing"]["alpha_timespent"]
    c_seconds = float(contract["targets"]["duration_constant_c"])

    events = load_weeks(args.raw, range(0, 26))
    week = events["week"]
    print(f"loaded {week.size:,} rows, weeks {week.min()}..{week.max()}")

    codes = {
        "user": factorize(events["user_id"]),
        "item": factorize(events["item_id"]),
        "user_place": factorize(events["user_id"], events["place"]),
    }

    stats = np.empty((week.size, 8), dtype=np.float64)
    stats[:, 0] = 1.0
    stats[:, 1] = events["timespent"].astype(np.float64)
    for offset, action in enumerate(ACTIONS, start=2):
        stats[:, offset] = events[action].astype(np.float64)

    target_weeks = list(range(20, 26))
    target_mask = np.isin(week, target_weeks)
    n_rows = int(target_mask.sum())
    print(f"target rows (weeks 20-25): {n_rows:,}")

    p_blocks: list[np.ndarray] = []
    v_blocks: list[np.ndarray] = []
    r_blocks: list[np.ndarray] = []
    p_names: list[str] = []
    v_names: list[str] = []
    r_names: list[str] = []
    denominator: list[bool] = []

    row_week = week[target_mask]
    for group in GROUPS:
        group_codes, group_size = codes[group]
        target_codes = group_codes[target_mask]
        for window in WINDOWS:
            aggregated = np.zeros((n_rows, 8), dtype=np.float64)
            for target in target_weeks:
                if window == "w1":
                    history = week == target - 1
                elif window == "w4":
                    history = (week >= target - 4) & (week < target)
                else:
                    history = week < target
                if not history.any():
                    continue
                sums = np.stack([
                    np.bincount(group_codes[history], weights=stats[history, s],
                                minlength=group_size)
                    for s in range(8)
                ], axis=1)
                rows_here = row_week == target
                aggregated[rows_here] = sums[target_codes[rows_here]]

            label = f"{group}_{window}"
            shows = aggregated[:, 0]
            timespent = aggregated[:, 1]
            has_history = shows > 0

            for name, values, is_denominator in (
                (f"p_{label}_n_shows", shows, True),
                (f"p_{label}_n_shows_plus_beta", shows + beta, True),
                (f"p_{label}_sum_timespent", timespent, True),
                (f"p_{label}_sum_timespent_plus_alpha", timespent + alpha_timespent, True),
            ):
                p_blocks.append(values.astype(np.float32))
                p_names.append(name)
                denominator.append(is_denominator)
            for offset, action in enumerate(ACTIONS, start=2):
                p_blocks.append((aggregated[:, offset] + alpha).astype(np.float32))
                p_names.append(f"p_{label}_n_{action}_plus_alpha")
                denominator.append(False)

            v_blocks.append(has_history.astype(np.float32))
            v_names.append(f"v_{label}_has_history")

            for offset, action in enumerate(ACTIONS, start=2):
                r_blocks.append(
                    ((aggregated[:, offset] + alpha) / (shows + beta)).astype(np.float32))
                r_names.append(f"e_{label}_{action}_rate")
            r_blocks.append(
                np.where(has_history, timespent / np.maximum(shows, 1.0), 0.0
                         ).astype(np.float32))
            r_names.append(f"e_{label}_avg_timespent")
            for offset, action in enumerate(ACTIONS, start=2):
                r_blocks.append(
                    ((aggregated[:, offset] + alpha) / (timespent + alpha_timespent)).astype(np.float32))
                r_names.append(f"e_{label}_{action}_per_second")
            print(f"  {label:<16} done")

    P = np.stack(p_blocks, axis=1)
    V = np.stack(v_blocks, axis=1)
    R = np.stack(r_blocks, axis=1)
    assert P.shape[1] == 90 and V.shape[1] == 9 and R.shape[1] == 117, (
        P.shape, V.shape, R.shape)
    if p_names != list(contract["ordered_P_fields"]):
        raise SystemExit("P field order diverges from the frozen contract")
    if r_names != list(contract["ordered_R_div_fields"]):
        raise SystemExit("R field order diverges from the frozen contract")

    is_train = row_week <= 24
    # Categorical vocabularies are fitted on train only; slot 0 is OOV.
    categorical = np.zeros((n_rows, 5), dtype=np.int64)
    cardinalities = []
    for index, field in enumerate(("user_id", "item_id", "place", "platform", "agent")):
        values = events[field][target_mask]
        vocabulary = np.unique(values[is_train])
        mapped = np.searchsorted(vocabulary, values)
        mapped = np.clip(mapped, 0, vocabulary.size - 1)
        seen = vocabulary[mapped] == values
        categorical[:, index] = np.where(seen, mapped + 1, 0)
        cardinalities.append(int(vocabulary.size) + 1)
    print(f"cardinalities {cardinalities}")

    # Purely multiplicative, train-only, median of strictly positive values.
    scales = np.ones(P.shape[1], dtype=np.float64)
    for column in range(P.shape[1]):
        positive = P[is_train, column]
        positive = positive[positive > 0]
        if positive.size:
            scales[column] = max(float(np.median(positive)), 1e-12)
    P = (P / scales.astype(np.float32)).astype(np.float32)
    r_scales = np.ones(R.shape[1], dtype=np.float64)
    for column in range(R.shape[1]):
        positive = R[is_train, column]
        positive = positive[positive > 0]
        if positive.size:
            r_scales[column] = max(float(np.median(positive)), 1e-12)
    R = (R / r_scales.astype(np.float32)).astype(np.float32)

    timespent = events["timespent"][target_mask].astype(np.float32)
    labels = np.stack([
        np.clip(timespent, 0, c_seconds) / c_seconds,
        (timespent > 10).astype(np.float32),
        (timespent > 20).astype(np.float32),
        (timespent > 40).astype(np.float32),
    ], axis=1).astype(np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    for split, mask in (("train", is_train), ("validation", ~is_train)):
        path = args.output / f"{split}.npz"
        np.savez(path, categorical=categorical[mask], P=P[mask], V=V[mask],
                 R=R[mask], labels=labels[mask],
                 user_ids=events["user_id"][target_mask][mask].astype(np.int64))
        print(f"{split}: {int(mask.sum()):,} rows -> {path}")

    mask_full = denominator + [False] * (V.shape[1] + R.shape[1])
    manifest = {
        "artifact_role": "vk_lsvd_ur_runtime_views",
        "schema_version": 1,
        "dataset_id": "vk_lsvd_ur0.01_ir0.01",
        "contract_sha256": digest,
        "categorical_cardinalities": cardinalities,
        "P_names": p_names, "V_names": v_names, "R_names": r_names,
        "denominator_mask": [bool(x) for x in denominator] + [False] * V.shape[1],
        "denominator_mask_full": [bool(x) for x in mask_full],
        "denominator_count": int(sum(denominator)),
        "reduced_operands": int(P.shape[1] + V.shape[1]),
        "full_operands": int(P.shape[1] + V.shape[1] + R.shape[1]),
        "target_names": ["timespent", "timespent_gt10", "timespent_gt20", "timespent_gt40"],
        "watch_time_c_seconds": c_seconds,
        "normalization": "train_median_of_strictly_positive_values",
        "test_split_present": False,
        "rows": {"train": int(is_train.sum()), "validation": int((~is_train).sum())},
    }
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    (args.output / "manifest_v1.json").write_text(text, encoding="utf-8")
    print(f"denominators {sum(denominator)} of {P.shape[1]+V.shape[1]} reduced operands "
          f"= {100*sum(denominator)/(P.shape[1]+V.shape[1]):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
