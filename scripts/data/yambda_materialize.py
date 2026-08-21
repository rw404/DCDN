#!/usr/bin/env python
"""Materialize Yambda-50M runtime views for the single-head listening study.

Yambda ships six columns and nothing derived, so every feature here is built
from rolling history. `configs/data/yambda/features_v1.yaml` fixes the design
and was frozen before this ran.

The leakage rule, and how it is enforced
----------------------------------------
A row may see what happened **strictly before its own timestamp**, per key
group. Rows sharing a timestamp are invisible to each other: the whole
equal-timestamp group is scored against the accumulator before any of it is
added. That is implemented by scoring at the `searchsorted(..., side="left")`
position within each group's sorted timestamps, the same construction the
KuaiRand materializer uses.

`played_ratio_pct` is the current row's outcome. It is the sole label source and
never an input to its own row; it does enter *other* rows' history.

Burn-in rows update history and carry no label, so the first trained day already
has a deep history. The sealed test days are counted and dropped.

Usage::

    python scripts/data/yambda_materialize.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "configs" / "data" / "yambda" / "features_v1.yaml"

DAY = 86_400
GROUPS = ("user", "item", "artist", "album")
WINDOWS = (("1d", DAY), ("7d", 7 * DAY), ("all", None))
STATISTICS = (
    "listen_count", "played_ratio_sum", "track_length_sum",
    "listen50_count", "organic_count",
)
#: The four manual divisions the REDUCED view removes.
DIVISIONS = (
    ("mean_played_ratio", "played_ratio_sum"),
    ("listen50_rate", "listen50_count"),
    ("organic_rate", "organic_count"),
    ("mean_track_length", "track_length_sum"),
)


class MaterializationError(RuntimeError):
    """The data does not match the frozen contract."""


def _publish(path: Path, payload: dict[str, Any]) -> str:
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


def rolling(
    times: np.ndarray, keys: np.ndarray, stats: np.ndarray, span: int | None
) -> np.ndarray:
    """Per-key rolling sums over rows strictly earlier than each row.

    Fully vectorized: there is no Python loop over groups. Yambda has 877k
    distinct items, and a per-group loop both crawled and died silently on this
    data, so the group boundary is folded into the sort key instead.

    `key` and `time` are packed into one int64 as ``key << 32 | time``. Because
    a smaller key always sorts before a larger one, a `searchsorted` on the
    packed array cannot cross a group boundary, which is what lets the window
    lookup be a single global call.

    The leakage rule is the `side="left"` position: every row reads the prefix
    at the first index sharing its own (key, time), so it sees everything
    strictly earlier and nothing from its own equal-timestamp group.
    """

    if times.max() >= (1 << 32) or times.min() < 0:
        raise MaterializationError("timestamps do not fit the 32-bit pack")
    packed = (keys.astype(np.int64) << 32) | times.astype(np.int64)
    order = np.argsort(packed, kind="stable")
    sorted_packed = packed[order]
    sorted_keys = sorted_packed >> 32

    prefix = np.zeros((times.shape[0] + 1, stats.shape[1]), dtype=np.float64)
    np.cumsum(stats[order], axis=0, out=prefix[1:])

    # Start index of each key's block, broadcast to every row in it.
    positions = np.arange(times.shape[0])
    key_starts = np.maximum.accumulate(
        np.where(
            np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1])),
            positions, 0,
        )
    )
    # Start index of each (key, time) run: the `side="left"` position.
    run_starts = np.maximum.accumulate(
        np.where(
            np.concatenate(([True], sorted_packed[1:] != sorted_packed[:-1])),
            positions, 0,
        )
    )

    if span is None:
        low = key_starts
    else:
        # A smaller key sorts first, so this search stops at the group boundary
        # on its own and the result is clamped to the block start.
        target = (sorted_keys << 32) | np.maximum(
            (sorted_packed & 0xFFFFFFFF) - span, 0
        )
        low = np.maximum(
            np.searchsorted(sorted_packed, target, side="left"), key_starts
        )

    out = np.empty_like(prefix[:-1])
    out[order] = prefix[run_starts] - prefix[low]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(os.environ.get("DCDN_DATA_CACHE", "/tmp"))
    parser.add_argument("--source", type=Path, default=cache / "raw" / "yambda")
    parser.add_argument(
        "--output-dir", type=Path,
        default=cache / "runtime" / "yambda_listen_v1",
    )
    args = parser.parse_args(argv)

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    temporal = contract["temporal"]
    print(f"contract sha256 "
          f"{hashlib.sha256(CONTRACT.read_bytes()).hexdigest()[:16]}...")

    print("loading listens")
    listens = pl.read_parquet(args.source / "listens.parquet")
    print(f"  {listens.height:,} rows")

    print("joining item-side mappings")
    album = pl.read_parquet(args.source / "album_item_mapping.parquet").unique(
        subset=["item_id"], keep="first"
    )
    artist = pl.read_parquet(args.source / "artist_item_mapping.parquet").unique(
        subset=["item_id"], keep="first"
    )
    listens = listens.join(album, on="item_id", how="left").join(
        artist, on="item_id", how="left"
    ).with_columns([
        pl.col("album_id").fill_null(0), pl.col("artist_id").fill_null(0),
    ])

    listens = listens.sort(["timestamp", "uid", "item_id"])
    day = (listens["timestamp"].to_numpy() // DAY).astype(np.int64)
    burn_lo, burn_hi = temporal["burn_in_days"]
    train_lo, train_hi = temporal["train_days"]
    valid_lo, valid_hi = temporal["validation_days"]
    test_lo, test_hi = temporal["sealed_test_days"]

    role = np.full(day.shape, -1, dtype=np.int8)
    role[(day >= burn_lo) & (day <= burn_hi)] = 0
    role[(day >= train_lo) & (day <= train_hi)] = 1
    role[(day >= valid_lo) & (day <= valid_hi)] = 2
    role[(day >= test_lo) & (day <= test_hi)] = 3
    for code, name in ((0, "burn_in"), (1, "train"), (2, "validation"),
                       (3, "sealed_test")):
        print(f"  {name:<12}{int((role == code).sum()):>12,} rows")
    if (role < 0).any():
        raise MaterializationError("a row fell outside every declared day range")

    times = listens["timestamp"].to_numpy().astype(np.int64)
    played = listens["played_ratio_pct"].to_numpy().astype(np.float64)
    length = listens["track_length_seconds"].to_numpy().astype(np.float64)
    organic = listens["is_organic"].to_numpy().astype(np.float64)

    # The five per-row contributions every window sums. `played_ratio_pct` is
    # the outcome of THIS row; it enters other rows' history only.
    contributions = np.stack([
        np.ones_like(played), played, length,
        (played >= 50.0).astype(np.float64), organic,
    ], axis=1)

    key_arrays = {
        "user": listens["uid"].to_numpy(),
        "item": listens["item_id"].to_numpy(),
        "artist": listens["artist_id"].to_numpy(),
        "album": listens["album_id"].to_numpy(),
    }

    p_columns: list[np.ndarray] = [organic, length]
    p_names: list[str] = ["p_is_organic", "p_track_length_seconds"]
    v_columns: list[np.ndarray] = []
    v_names: list[str] = []
    r_columns: list[np.ndarray] = []
    r_names: list[str] = []
    denominator_columns: list[np.ndarray] = []
    denominator_names: list[str] = []

    for group in GROUPS:
        for window_name, span in WINDOWS:
            print(f"  rolling {group}/{window_name}")
            block = rolling(times, key_arrays[group], contributions, span)
            for index, statistic in enumerate(STATISTICS):
                p_columns.append(block[:, index])
                p_names.append(f"p_{group}_{window_name}_{statistic}")
            count = block[:, 0]
            shifted = count + 1.0
            denominator_columns.append(shifted)
            denominator_names.append(f"p_{group}_{window_name}_listen_count_plus_1")
            v_columns.append((count > 0.0).astype(np.float64))
            v_names.append(f"v_{group}_{window_name}_present")
            for division, source in DIVISIONS:
                numerator = block[:, STATISTICS.index(source)]
                r_columns.append(numerator / shifted)
                r_names.append(f"r_{group}_{window_name}_{division}")

    # Denominators go at the end of P so the frozen mask is a contiguous tail.
    p_columns.extend(denominator_columns)
    p_names.extend(denominator_names)

    p_block = np.stack(p_columns, axis=1)
    v_block = np.stack(v_columns, axis=1)
    r_block = np.stack(r_columns, axis=1)
    expected = contract["ordered_fields"]
    for name, block, want in (
        ("P", p_block, expected["P_count"]),
        ("V", v_block, expected["V_count"]),
        ("R", r_block, expected["R_count"]),
    ):
        if block.shape[1] != want:
            raise MaterializationError(
                f"{name} width {block.shape[1]} does not match the frozen {want}"
            )

    denominator_mask = [False] * (len(p_names) - len(denominator_names))
    denominator_mask += [True] * len(denominator_names)
    denominator_mask += [False] * len(v_names)

    train = role == 1
    scale_source = np.concatenate([p_block, r_block], axis=1)
    scales = np.maximum(
        np.quantile(np.abs(scale_source[train]), 0.99, axis=0, method="linear"), 1.0
    )
    p_block = p_block / scales[: p_block.shape[1]]
    r_block = r_block / scales[p_block.shape[1]:]

    play_fraction = (np.clip(played, 0.0, 100.0) / 100.0).astype(np.float32)
    listen50 = (played >= 50.0).astype(np.float32)
    labels = np.stack([play_fraction, listen50], axis=1)
    print(f"labels: play_fraction mean={play_fraction.mean():.4f} "
          f"listen50 rate={listen50.mean():.4f}")

    print("fitting train-only categorical mappings")
    categorical_raw = np.stack([
        key_arrays["user"], key_arrays["item"],
        key_arrays["artist"], key_arrays["album"],
    ], axis=1).astype(np.int64)
    categorical = np.zeros_like(categorical_raw)
    cardinalities: list[int] = []
    for column in range(categorical_raw.shape[1]):
        values = categorical_raw[:, column]
        vocabulary = np.unique(values[train])
        position = np.searchsorted(vocabulary, values)
        clipped = np.minimum(position, vocabulary.shape[0] - 1)
        # Slot 0 is out-of-vocabulary; the contract keeps cold rows rather than
        # dropping them, because Yambda has 877k items against 9.2k users.
        categorical[:, column] = np.where(
            vocabulary[clipped] == values, position + 1, 0
        )
        cardinalities.append(int(vocabulary.shape[0]) + 1)
    print(f"  cardinalities {cardinalities}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, int] = {}
    for name, code in (("train", 1), ("validation", 2)):
        selected = role == code
        split_rows[name] = int(selected.sum())
        np.savez(
            args.output_dir / f"{name}.npz",
            categorical=categorical[selected],
            P=p_block[selected].astype(np.float32),
            V=v_block[selected].astype(np.float32),
            R=r_block[selected].astype(np.float32),
            labels=labels[selected],
        )
        print(f"{name}: {split_rows[name]:,} rows -> {name}.npz")

    manifest = {
        "artifact_role": "yambda_listen_runtime_manifest",
        "schema_version": 1,
        "dataset_id": "yambda_50m",
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "label_names": ["play_fraction", "listen50"],
        "label_kinds": ["regression", "binary"],
        "P_names": p_names,
        "V_names": v_names,
        "R_names": r_names,
        "cardinalities": cardinalities,
        "denominator_mask": denominator_mask,
        "reduced_width": p_block.shape[1] + v_block.shape[1],
        "full_width": p_block.shape[1] + v_block.shape[1] + r_block.shape[1],
        "rows": {
            "total": int(listens.height),
            "burn_in": int((role == 0).sum()),
            **split_rows,
            "sealed_test_never_materialized": int((role == 3).sum()),
        },
        "play_fraction_mean": float(play_fraction.mean()),
        "listen50_rate": float(listen50.mean()),
        "normalization": {
            "rule": "divide_by_max(train_absolute_q99_linear, 1.0)",
            "fitted_on": "train_split_only",
            "scale_sha256": hashlib.sha256(
                np.ascontiguousarray(scales, dtype=np.float64).tobytes()
            ).hexdigest(),
        },
        "test_unlocked": False,
        "sealed_test_read": False,
    }
    digest = _publish(args.output_dir / "manifest_v1.json", manifest)
    print(f"\nmanifest sha256={digest}")
    print(f"REDUCED numeric width={manifest['reduced_width']} "
          f"FULL={manifest['full_width']} "
          f"denominators={sum(denominator_mask)}")
    print(f"output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
