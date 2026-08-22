#!/usr/bin/env python
"""Session identifiers for KuaiRand-Pure validation rows.

Why this exists
---------------
`configs/metrics_multitask_v1.yaml` freezes a session-grouped metric for every
target and states plainly that KuaiRand supplies what it needs:

    dataset_availability:
      kuairand_pure: available_from_time_ms_after_schema_validation

Yet every KuaiRand run reports `session_gauc: null` with the reason
"kuairand_pure_has_no_session_identifier". That reason is wrong. The dataset
does carry `time_ms` - the whole role split is cut on it - but the
materialized runtime views ship only `categorical`, `P`, `V`, `R` and
`labels`. The timestamp was used to slice the splits and then dropped, so the
runner genuinely cannot construct sessions and correctly refuses to invent
them, since the policy is
`emit_null_with_explicit_reason_and_never_synthesize_sessions`.

This script recovers the key without re-materializing anything. It replays the
materializer's own `load_events` and `_classify_roles`, takes the validation
rows in the same global order, and applies the frozen construction rule:

    consecutive_events_for_same_user_split_after_inactivity_gap
    inactivity_gap_minutes: 30
    stable_sort_keys: [user_id, event_timestamp, stable_row_id]

Alignment is verified rather than assumed. The recovered `user_id` sequence
must induce exactly the same row grouping as the view's own categorical column
0, which is the train-mapped user token. A mismatch aborts: a silently
misaligned session key would produce a plausible number that means nothing,
which is worse than the null it replaces.

Usage::

    python scripts/data/kuairand_sessions.py --output <cache>/runtime/kuairand_pure_multitask_v1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mat = _load(
    "kuairand_pure_materialize",
    REPO_ROOT / "scripts" / "data" / "kuairand_pure_materialize.py",
)

GAP_MINUTES = 30
GAP_MS = GAP_MINUTES * 60 * 1000


def build_sessions(user_id: np.ndarray, time_ms: np.ndarray) -> np.ndarray:
    """Split each user's ordered events wherever the gap exceeds 30 minutes.

    Fully vectorized. `stable_row_id` is the row's position in the global
    materializer order, so `kind="stable"` on the (user, time) sort reproduces
    the contract's three-key ordering exactly without materializing a third key.
    """

    order = np.lexsort((time_ms, user_id))
    sorted_user = user_id[order]
    sorted_time = time_ms[order]

    new_user = np.empty(order.size, dtype=bool)
    new_user[0] = True
    new_user[1:] = sorted_user[1:] != sorted_user[:-1]

    gap = np.zeros(order.size, dtype=bool)
    gap[1:] = (sorted_time[1:] - sorted_time[:-1]) > GAP_MS

    starts = new_user | gap
    sorted_session = np.cumsum(starts) - 1

    session = np.empty(order.size, dtype=np.int64)
    session[order] = sorted_session
    return session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(os.environ.get("DCDN_DATA_CACHE", "/tmp"))
    parser.add_argument(
        "--output", type=Path,
        default=cache / "runtime" / "kuairand_pure_multitask_v1",
    )
    parser.add_argument("--split", default="validation")
    args = parser.parse_args(argv)

    contract = mat._load_contract()
    cutoffs = contract["approved_temporal_policy"]["embargo"]["cutoffs_time_ms"]
    print("replaying the materializer's event load")
    events = mat.load_events(mat._data_root(), cutoffs)

    selector = events["role"].astype(str) == args.split
    user_id = events["user_id"][selector]
    time_ms = events["time_ms"][selector]
    print(f"{args.split}: {user_id.size:,} rows, {np.unique(user_id).size:,} users")

    view = np.load(args.output / f"{args.split}.npz")
    tokens = view["categorical"][:, 0]
    if tokens.size != user_id.size:
        raise SystemExit(f"row count mismatch: view {tokens.size} vs replay {user_id.size}")

    # A row grouping is the same object whichever labels name it, so compare
    # the partitions rather than the ids: identical group boundaries under a
    # stable sort mean the two keys induce the same partition.
    _, raw_codes = np.unique(user_id, return_inverse=True)
    _, token_codes = np.unique(tokens, return_inverse=True)
    if not np.array_equal(
        np.argsort(raw_codes, kind="stable"), np.argsort(token_codes, kind="stable")
    ):
        raise SystemExit(
            "recovered user_id does not induce the view's own user grouping - "
            "row order differs, aborting rather than emitting a plausible number"
        )
    print("alignment verified against the view's categorical user column")

    session = build_sessions(user_id, time_ms)
    sizes = np.bincount(session)
    print(f"sessions: {sizes.size:,}  median rows/session {np.median(sizes):.0f}  "
          f"max {sizes.max():,}  singletons {int((sizes == 1).sum()):,}")

    path = args.output / f"{args.split}_sessions.i64.npy"
    np.save(path, session)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {
        "artifact_role": "kuairand_pure_session_identifiers",
        "schema_version": 1,
        "split": args.split,
        "construction": "consecutive_events_for_same_user_split_after_inactivity_gap",
        "inactivity_gap_minutes": GAP_MINUTES,
        "rows": int(user_id.size),
        "sessions": int(sizes.size),
        "median_rows_per_session": float(np.median(sizes)),
        "singleton_sessions": int((sizes == 1).sum()),
        "alignment_verified_against": "view categorical column 0",
        "sha256": digest,
    }
    (args.output / f"{args.split}_sessions_v1.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {path} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
