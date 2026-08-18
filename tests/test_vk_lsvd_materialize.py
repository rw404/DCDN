from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dcdn.data import vk_lsvd_materialize as vm
from dcdn.data.vk_lsvd import (
    DENOMINATOR_ELIGIBLE_FIELDS,
    ORDERED_C_FIELDS,
    ORDERED_E_FIELDS,
    ORDERED_NUMERIC_OPERANDS,
    ORDERED_P0_FIELDS,
    ORDERED_V_FIELDS,
    stable_row_id,
)


def _interaction_table(
    *,
    user_id: list[int],
    item_id: list[int],
    place: list[int],
    platform: list[int],
    agent: list[int],
    timespent: list[int],
) -> pa.Table:
    rows = len(user_id)
    assert all(
        len(values) == rows for values in (item_id, place, platform, agent, timespent)
    )
    sequence = np.arange(rows)
    return pa.table(
        {
            "user_id": pa.array(user_id, type=pa.uint32()),
            "item_id": pa.array(item_id, type=pa.uint32()),
            "place": pa.array(place, type=pa.uint8()),
            "platform": pa.array(platform, type=pa.uint8()),
            "agent": pa.array(agent, type=pa.uint8()),
            "timespent": pa.array(timespent, type=pa.uint8()),
            "like": pa.array(sequence % 2 == 0, type=pa.bool_()),
            "dislike": pa.array(sequence % 3 == 0, type=pa.bool_()),
            "share": pa.array(sequence % 3 == 1, type=pa.bool_()),
            "bookmark": pa.array(sequence % 4 == 1, type=pa.bool_()),
            "click_on_author": pa.array(sequence % 4 == 2, type=pa.bool_()),
            "open_comments": pa.array(sequence % 2 == 1, type=pa.bool_()),
        }
    )


def _write_interactions(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _synthetic_paths(root: Path) -> vm.ExplicitInputPaths:
    history_table = _interaction_table(
        user_id=[1, 1, 2, 2],
        item_id=[10, 11, 10, 11],
        place=[1, 1, 2, 2],
        platform=[1, 1, 2, 2],
        agent=[1, 2, 1, 2],
        timespent=[20, 40, 30, 50],
    )
    history: list[tuple[int, Path]] = []
    for week in range(20):
        path = root / "source" / "train" / f"week_{week:02d}.parquet"
        _write_interactions(path, history_table)
        history.append((week, path))

    train: list[tuple[int, Path]] = []
    for week in range(20, 25):
        path = root / "source" / "train" / f"week_{week:02d}.parquet"
        _write_interactions(
            path,
            _interaction_table(
                user_id=[1, 1, 2, 3],
                item_id=[10, 10, 11, 12],
                place=[3, 3, 4, 5],
                platform=[1, 1, 2, 3],
                agent=[2, 2, 3, 4],
                timespent=[0, 255, 70, 80],
            ),
        )
        train.append((week, path))

    validation_path = root / "source" / "validation" / "week_25.parquet"
    _write_interactions(
        validation_path,
        _interaction_table(
            user_id=[1, 4, 5],
            item_id=[10, 13, 14],
            place=[3, 9, 9],
            platform=[1, 9, 9],
            agent=[2, 9, 9],
            timespent=[15, 255, 31],
        ),
    )

    metadata = root / "source" / "metadata"
    metadata.mkdir(parents=True)
    users_path = metadata / "users_metadata.parquet"
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array([1, 2, 3, 4], type=pa.uint32()),
                "age": pa.array([2, 3, 4, 9], type=pa.uint8()),
                "gender": pa.array([1, 2, 1, 9], type=pa.uint8()),
                "geo": pa.array([4, 5, 6, 9], type=pa.uint8()),
                "train_interactions_rank": pa.array([1, 2, 3, 4], type=pa.uint32()),
            }
        ),
        users_path,
    )
    items_path = metadata / "items_metadata.parquet"
    pq.write_table(
        pa.table(
            {
                "item_id": pa.array([10, 11, 12, 13], type=pa.uint32()),
                "author_id": pa.array([7, 8, 9, 10], type=pa.uint32()),
                "duration": pa.array([60, 90, 120, 250], type=pa.uint8()),
                "train_interactions_rank": pa.array([1, 2, 3, 4], type=pa.uint32()),
            }
        ),
        items_path,
    )
    return vm.ExplicitInputPaths(
        history=tuple(history),
        train=tuple(train),
        validation=((25, validation_path),),
        users_metadata=users_path,
        items_metadata=items_path,
    )


def _run_synthetic(root: Path, name: str) -> dict[str, object]:
    paths = _synthetic_paths(root / name / "fixture")
    return vm.materialize_explicit_paths(
        paths=paths,
        output_root=root / name / "output",
        fit_config=vm.IALSFitConfig(factors=2, iterations=2),
    )


def test_frozen_paths_exclude_sealed_week() -> None:
    paths = vm.frozen_input_paths(Path("/not-opened"))
    assert [week for week, _ in paths.all_interactions()] == list(range(26))
    assert all("week_26" not in path.name for _, path in paths.all_interactions())


def test_synthetic_materialization_is_deterministic_and_matched(
    tmp_path: Path,
) -> None:
    first = _run_synthetic(tmp_path, "first")
    second = _run_synthetic(tmp_path, "second")

    assert first["history"].fingerprint_sha256 == second["history"].fingerprint_sha256
    assert first["metadata"].fingerprint_sha256 == second["metadata"].fingerprint_sha256
    assert first["categories"] == second["categories"]
    assert first["p0_scales"] == second["p0_scales"]
    assert first["e_scales"] == second["e_scales"]
    assert (first["train_rows"], first["validation_rows"]) == (20, 3)

    first_shards = first["shards"]
    second_shards = second["shards"]
    assert len(first_shards) == len(second_shards) == 6
    for left, right in zip(first_shards, second_shards, strict=True):
        assert left.c_plus_p_sha256 == right.c_plus_p_sha256
        assert left.c_plus_p_plus_e_sha256 == right.c_plus_p_plus_e_sha256
        assert left.shared_values_sha256 == right.shared_values_sha256
        assert left.stable_row_ids_sha256 == right.stable_row_ids_sha256

        cp = pq.read_table(left.c_plus_p_path)
        cpe = pq.read_table(left.c_plus_p_plus_e_path)
        assert cp.num_rows == cpe.num_rows == left.rows
        shared_names = [name for name in cp.column_names if name in cpe.column_names]
        assert cp.select(shared_names).equals(cpe.select(shared_names))

    cp_schema = pq.ParquetFile(first_shards[0].c_plus_p_path).schema_arrow
    cpe_schema = pq.ParquetFile(first_shards[0].c_plus_p_plus_e_path).schema_arrow
    assert len(ORDERED_C_FIELDS) == 6
    assert len(ORDERED_P0_FIELDS) == 31
    assert len(ORDERED_V_FIELDS) == 27
    assert len(ORDERED_NUMERIC_OPERANDS) == 58
    assert len(ORDERED_E_FIELDS) == 20
    assert all(name in cp_schema.names for name in ORDERED_NUMERIC_OPERANDS)
    assert all(name not in cp_schema.names for name in ORDERED_E_FIELDS)
    assert all(name in cpe_schema.names for name in ORDERED_E_FIELDS)


def test_train_only_fit_history_freeze_and_denominator_validity(
    tmp_path: Path,
) -> None:
    result = _run_synthetic(tmp_path, "guarded")
    shards = result["shards"]
    first_train = pq.read_table(shards[0].c_plus_p_path)
    validation = pq.read_table(shards[-1].c_plus_p_path)

    # Same entity pair and context have identical model inputs despite different
    # current-row outcomes and later weeks: target rows never update history.
    for name in (*ORDERED_C_FIELDS, *ORDERED_NUMERIC_OPERANDS):
        assert first_train[name][0].as_py() == first_train[name][1].as_py()
        assert first_train[name][0].as_py() == validation[name][0].as_py()
    assert first_train["label_timespent_seconds"][0].as_py() == 0.0
    assert first_train["label_timespent_seconds"][1].as_py() == 157.0

    # Validation-only categories are OOV; absent metadata uses MISSING.
    for name in ORDERED_C_FIELDS:
        assert validation[name][1].as_py() == 0
    for name in ("age", "gender", "geo"):
        assert validation[name][2].as_py() == 1
    for name in ("place", "platform", "agent"):
        assert validation[name][2].as_py() == 0

    # Validation item duration 250 is outside train q99 and remains unclipped.
    assert validation["p_duration_seconds"][1].as_py() > 1.0
    assert validation["denominator_validity__p_duration_seconds"][1].as_py()
    assert not validation["denominator_validity__p_duration_seconds"][2].as_py()
    assert not validation["denominator_validity__p_user_show_count"][2].as_py()
    assert validation["denominator_validity__p_user_show_count_plus_beta"][2].as_py()
    for name in DENOMINATOR_ELIGIBLE_FIELDS:
        audit = result["denominator_audit"][name]
        assert audit["rows"] == 23
        assert audit["valid_rows"] + audit["invalid_or_zero_rows"] == 23
        assert audit["nonfinite_rows"] == 0

    assert validation["stable_row_id"][0].as_py() == stable_row_id(
        week=25, source_row_index0=0
    )


def test_manifest_round_trip_hash_and_tamper_guard(tmp_path: Path) -> None:
    result = _run_synthetic(tmp_path, "manifest")
    fit = vm.IALSFitConfig(factors=2, iterations=2)
    final_root = tmp_path / "manifest" / "output"
    payload = vm._build_manifest(
        result=result,
        final_root=final_root,
        verified_sources=[],
        resource_guard={"synthetic_fixture": True},
        fit_config=fit,
    )
    path = tmp_path / "views.yaml"
    digest = vm.write_views_manifest(path, payload)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert vm.load_views_manifest(path) == payload
    assert payload["week26_file_open_attempted"] is False
    assert payload["validation_model_metrics_computed"] is False

    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(vm.VKLSVDMaterializationError, match="sidecar mismatch"):
        vm.load_views_manifest(path)
