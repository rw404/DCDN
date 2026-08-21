from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import scripts.data.vk_lsvd_multitask_views as multitask_views
from dcdn.data.vk_lsvd import (
    DENOMINATOR_ELIGIBLE_FIELDS as LEGACY_DENOMINATORS,
)
from dcdn.data.vk_lsvd import (
    ORDERED_C_FIELDS,
    floor_indicator_name,
    stable_row_id,
)
from dcdn.data.vk_lsvd import (
    ORDERED_E_FIELDS as LEGACY_E_FIELDS,
)
from dcdn.data.vk_lsvd import (
    ORDERED_NUMERIC_OPERANDS as LEGACY_OPERANDS,
)
from dcdn.data.vk_lsvd_multitask import (
    DENOMINATOR_ELIGIBLE_FIELDS,
    ORDERED_DENOMINATOR_MASK,
    ORDERED_OPERAND_FIELDS,
    ORDERED_V_FIELDS,
    R_DIV_FIELDS,
    SHARED_SIMILARITY_FIELDS,
    TARGET_COLUMNS,
    TARGET_MASK_COLUMNS,
    VKLSVDMultiTargetError,
    contract_state,
    transform_verified_shard,
    view_schema,
)
from dcdn.data.vk_lsvd_runtime import _expected_schema
from dcdn.protocol import read_verified_artifact
from scripts.data.vk_lsvd_multitask_views import (
    EXPECTED_FEATURE_POLICY_SHA256,
    EXPECTED_MATRIX_SHA256,
    EXPECTED_TARGETS_V2_SHA256,
    MaterializationError,
    _publish_manifest,
    _remove_published_output,
    _verified_frozen_inputs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_yaml_artifact(path: Path, payload: object) -> str:
    data = yaml.safe_dump(
        payload, sort_keys=True, allow_unicode=False, width=100
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_materialization_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    legacy_root = tmp_path / "legacy"
    raw_root = tmp_path / "raw"
    shards: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for week in range(20, 26):
        role = "train" if week < 25 else "validation"
        feature_relative = Path("C_plus_P_plus_E") / role / f"week_{week:02d}.parquet"
        feature_path = legacy_root / feature_relative
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_legacy_full(week=week), feature_path)
        raw_relative = Path(f"subsamples/up0.01_ir0.01/{role}/week_{week:02d}.parquet")
        raw_path = raw_root / raw_relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_interactions(), raw_path)
        feature_digest = _file_sha256(feature_path)
        raw_digest = _file_sha256(raw_path)
        shards.append(
            {
                "week": week,
                "C_plus_P_plus_E": {
                    "path": feature_relative.as_posix(),
                    "sha256": feature_digest,
                },
            }
        )
        sources.append({"path": raw_relative.as_posix(), "sha256": raw_digest})
    manifest_path = tmp_path / "legacy-views.yaml"
    _write_yaml_artifact(
        manifest_path,
        {
            "schema_version": 1,
            "dataset_id": "vk_lsvd_up0.01_ir0.01",
            "artifact_role": "train_validation_feature_views",
            "materialization_state": "complete_and_hash_verified",
            "test_unlocked": False,
            "test_labels_opened": False,
            "test_view_materialized": False,
            "week26_file_open_attempted": False,
            "test_metrics_computed": False,
            "shards": shards,
            "source_objects_verified": sources,
        },
    )
    return manifest_path, legacy_root, raw_root


def _legacy_full(rows: int = 3, *, week: int = 20) -> pa.Table:
    schema = _expected_schema(include_engineered=True)
    arrays: dict[str, object] = {
        "user_id": np.asarray([10, 11, 12], dtype=np.uint32)[:rows],
        "item_id": np.asarray([20, 21, 22], dtype=np.uint32)[:rows],
        "week": np.full(rows, week, dtype=np.uint8),
        "source_row_index0": np.arange(rows, dtype=np.uint64),
        "stable_row_id": [
            stable_row_id(week=week, source_row_index0=index) for index in range(rows)
        ],
        "label_normalized_watch_time": np.asarray([0.0, 0.5, 1.0], dtype=np.float32)[
            :rows
        ],
        "label_timespent_seconds": np.asarray([0.0, 78.5, 157.0], dtype=np.float32)[
            :rows
        ],
    }
    for position, name in enumerate(ORDERED_C_FIELDS):
        arrays[name] = np.full(rows, position + 2, dtype=np.uint32)
    for position, name in enumerate(LEGACY_OPERANDS):
        values = np.full(rows, position + 0.25, dtype=np.float32)
        if name.startswith("v_"):
            values[:] = 1.0
        if name in {floor_indicator_name(field) for field in LEGACY_DENOMINATORS}:
            values[:] = 0.0
        arrays[name] = values
    for position, name in enumerate(LEGACY_E_FIELDS):
        arrays[name] = np.full(rows, position + 0.5, dtype=np.float32)
    for name in LEGACY_OPERANDS:
        arrays[f"denominator_validity__{name}"] = np.ones(rows, dtype=np.bool_)
    return pa.Table.from_arrays(
        [pa.array(arrays[field.name], type=field.type) for field in schema],
        schema=schema,
    )


def _interactions(rows: int = 3) -> pa.Table:
    return pa.table(
        {
            "user_id": pa.array([10, 11, 12][:rows], type=pa.uint32()),
            "item_id": pa.array([20, 21, 22][:rows], type=pa.uint32()),
            "like": pa.array([False, True, False][:rows], type=pa.bool_()),
            "dislike": pa.array([False, False, True][:rows], type=pa.bool_()),
            "share": pa.array([True, False, False][:rows], type=pa.bool_()),
            "bookmark": pa.array([False, True, True][:rows], type=pa.bool_()),
            "click_on_author": pa.array([True, True, False][:rows], type=pa.bool_()),
            "open_comments": pa.array([False, True, False][:rows], type=pa.bool_()),
        }
    )


def test_owner_override_exact_feature_and_target_contract() -> None:
    state = contract_state()
    assert state["target_names"] == [
        "watch_time",
        "like",
        "dislike",
        "share",
        "bookmark",
        "click_on_author",
        "open_comments",
    ]
    assert len(ORDERED_OPERAND_FIELDS) == 55
    assert len(ORDERED_V_FIELDS) == 21
    assert len(R_DIV_FIELDS) == 17
    assert len(SHARED_SIMILARITY_FIELDS) == 3
    assert sum(ORDERED_DENOMINATOR_MASK) == 5
    assert tuple(state["ordered_D"]) == DENOMINATOR_ELIGIBLE_FIELDS
    assert set(SHARED_SIMILARITY_FIELDS).issubset(ORDERED_OPERAND_FIELDS)
    assert set(SHARED_SIMILARITY_FIELDS).isdisjoint(R_DIV_FIELDS)
    assert all("ials" not in name for name in DENOMINATOR_ELIGIBLE_FIELDS)
    assert len(view_schema("FULL")) - len(view_schema("REDUCED")) == 17


def test_targets_v2_is_forward_only_and_binds_all_released_vk_actions() -> None:
    v1 = REPOSITORY_ROOT / "configs/targets_multitask_v1.yaml"
    v2 = REPOSITORY_ROOT / "configs/targets_multitask_v2.yaml"
    assert _file_sha256(v1) == (
        "2a9e3a8e0a5547b500682b9b632800e2bd103dbf67bebec101b281794bc29674"
    )
    verified = read_verified_artifact(v2)
    assert verified.digest == EXPECTED_TARGETS_V2_SHA256
    payload = yaml.safe_load(verified.data)
    assert payload["schema_version"] == 2
    assert payload["supersedes"]["sha256"] == _file_sha256(v1)
    targets = payload["datasets"]["vk_lsvd_up0.01_ir0.01"]["targets"]
    assert [target["name"] for target in targets] == list(
        contract_state()["target_names"]
    )
    assert all(target["always_observed"] is True for target in targets)
    action_names = {target["name"] for target in targets if target["kind"] == "binary"}
    assert action_names == {
        "like",
        "dislike",
        "share",
        "bookmark",
        "click_on_author",
        "open_comments",
    }
    physical_names = set(view_schema("REDUCED").names)
    assert action_names.isdisjoint(physical_names)
    assert {f"label_{name}" for name in action_names}.issubset(physical_names)


def test_transform_is_row_matched_multi_target_and_exact_set_difference() -> None:
    transformed = transform_verified_shard(
        week=20,
        legacy_full=_legacy_full(),
        interactions=_interactions(),
    )
    reduced = transformed.reduced
    full = transformed.full
    assert reduced.num_rows == full.num_rows == 3
    assert set(full.column_names) - set(reduced.column_names) == set(R_DIV_FIELDS)
    assert all(name in reduced.column_names for name in SHARED_SIMILARITY_FIELDS)
    assert all(name not in reduced.column_names for name in R_DIV_FIELDS)
    assert all(name in full.column_names for name in R_DIV_FIELDS)
    shared = [name for name in reduced.column_names if name in full.column_names]
    assert reduced.select(shared).equals(full.select(shared))

    assert reduced["label_watch_time"].to_pylist() == [0.0, 0.5, 1.0]
    assert reduced["label_like"].to_pylist() == [0.0, 1.0, 0.0]
    assert reduced["label_dislike"].to_pylist() == [0.0, 0.0, 1.0]
    assert reduced["label_share"].to_pylist() == [1.0, 0.0, 0.0]
    assert reduced["label_bookmark"].to_pylist() == [0.0, 1.0, 1.0]
    assert reduced["label_click_on_author"].to_pylist() == [1.0, 1.0, 0.0]
    assert reduced["label_open_comments"].to_pylist() == [0.0, 1.0, 0.0]
    for name in TARGET_MASK_COLUMNS:
        assert reduced[name].to_pylist() == [True, True, True]
    for name in ORDERED_OPERAND_FIELDS:
        validity = reduced[f"denominator_validity__{name}"].to_pylist()
        assert validity == [True, True, True]
    assert len(transformed.shared_values_sha256) == 64
    assert len(transformed.labels_and_masks_sha256) == 64
    assert len(transformed.stable_row_ids_sha256) == 64


@pytest.mark.parametrize("week", [0, 19, 26, 27])
def test_transform_rejects_history_and_sealed_test_weeks(week: int) -> None:
    with pytest.raises(VKLSVDMultiTargetError, match="train weeks"):
        transform_verified_shard(
            week=week,
            legacy_full=_legacy_full(week=20),
            interactions=_interactions(),
        )


def test_transform_rejects_row_misalignment_before_returning_views() -> None:
    interactions = _interactions()
    replacement = interactions.set_column(
        interactions.schema.get_field_index("item_id"),
        "item_id",
        pa.array([20, 99, 22], type=pa.uint32()),
    )
    with pytest.raises(VKLSVDMultiTargetError, match="row-aligned"):
        transform_verified_shard(
            week=20,
            legacy_full=_legacy_full(),
            interactions=replacement,
        )


def test_transform_rejects_missing_binary_label_column() -> None:
    interactions = _interactions().drop(["dislike"])
    with pytest.raises(VKLSVDMultiTargetError, match="misses columns"):
        transform_verified_shard(
            week=20,
            legacy_full=_legacy_full(),
            interactions=interactions,
        )


def test_transform_is_deterministic_and_input_immutable() -> None:
    legacy = _legacy_full()
    interactions = _interactions()
    legacy_before = copy.deepcopy(legacy)
    raw_before = copy.deepcopy(interactions)
    first = transform_verified_shard(
        week=20, legacy_full=legacy, interactions=interactions
    )
    second = transform_verified_shard(
        week=20, legacy_full=legacy, interactions=interactions
    )
    assert first.reduced.equals(second.reduced)
    assert first.full.equals(second.full)
    assert first.shared_values_sha256 == second.shared_values_sha256
    assert first.labels_and_masks_sha256 == second.labels_and_masks_sha256
    assert legacy.equals(legacy_before)
    assert interactions.equals(raw_before)


def test_target_and_mask_columns_are_present_in_both_regimes() -> None:
    reduced = view_schema("REDUCED")
    full = view_schema("FULL")
    for name in (*TARGET_COLUMNS, *TARGET_MASK_COLUMNS):
        assert reduced.field(name) == full.field(name)


def test_materializer_binds_exact_frozen_multi_target_inputs() -> None:
    assert _verified_frozen_inputs(REPOSITORY_ROOT) == {
        "targets_multitask_v2_sha256": EXPECTED_TARGETS_V2_SHA256,
        "feature_policy_sha256": EXPECTED_FEATURE_POLICY_SHA256,
        "matrix_multitask_sha256": EXPECTED_MATRIX_SHA256,
    }


def test_manifest_publication_is_exact_immutable_pair(tmp_path: Path) -> None:
    path = tmp_path / "views.yaml"
    payload = {"schema_version": 1, "test_unlocked": False, "rows": 3}
    digest = _publish_manifest(path, payload)
    reopened = read_verified_artifact(path)
    assert reopened.digest == digest
    assert hashlib.sha256(reopened.data).hexdigest() == digest
    with pytest.raises(MaterializationError, match="already exists"):
        _publish_manifest(path, payload)


def test_published_output_cleanup_is_exact_directory_only(tmp_path: Path) -> None:
    output = tmp_path / "fresh-output"
    output.mkdir()
    (output / "part.parquet").write_bytes(b"regenerable")
    sibling = tmp_path / "retain.txt"
    sibling.write_text("retain", encoding="utf-8")
    _remove_published_output(output)
    assert not output.exists()
    assert sibling.read_text(encoding="utf-8") == "retain"


def test_published_output_cleanup_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "output"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(MaterializationError, match="changed"):
        _remove_published_output(link)
    assert target.is_dir()


def test_tiny_materialization_publishes_all_train_validation_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_manifest, legacy_root, raw_root = _tiny_materialization_inputs(tmp_path)
    monkeypatch.setattr(
        multitask_views,
        "EXPECTED_LEGACY_MANIFEST_SHA256",
        _file_sha256(legacy_manifest),
    )
    monkeypatch.setattr(
        multitask_views,
        "_require_clean_code_identity",
        lambda _root: {
            "code_commit": "a" * 40,
            "code_diff_sha256": hashlib.sha256(b"").hexdigest(),
        },
    )
    output_root = tmp_path / "published-views"
    manifest_path = tmp_path / "published-manifest.yaml"
    digest = multitask_views.materialize(
        repository_root=REPOSITORY_ROOT,
        legacy_manifest_path=legacy_manifest,
        legacy_views_root=legacy_root,
        raw_dataset_root=raw_root,
        output_root=output_root,
        manifest_path=manifest_path,
    )
    reopened = read_verified_artifact(manifest_path)
    assert reopened.digest == digest
    payload = yaml.safe_load(reopened.data)
    assert payload["row_counts"] == {"train": 15, "validation": 3, "total": 18}
    assert payload["test_unlocked"] is False
    assert payload["week26_file_open_attempted"] is False
    assert len(payload["shards"]) == 6
    assert len(list(output_root.rglob("*.parquet"))) == 12
    for record in payload["shards"]:
        reduced = pq.read_table(output_root / record["REDUCED"]["path"])
        full = pq.read_table(output_root / record["FULL"]["path"])
        assert reduced.num_rows == full.num_rows == 3
        assert set(full.column_names) - set(reduced.column_names) == set(R_DIV_FIELDS)


def test_materialization_rolls_back_fresh_output_when_manifest_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_manifest, legacy_root, raw_root = _tiny_materialization_inputs(tmp_path)
    monkeypatch.setattr(
        multitask_views,
        "EXPECTED_LEGACY_MANIFEST_SHA256",
        _file_sha256(legacy_manifest),
    )
    monkeypatch.setattr(
        multitask_views,
        "_require_clean_code_identity",
        lambda _root: {
            "code_commit": "a" * 40,
            "code_diff_sha256": hashlib.sha256(b"").hexdigest(),
        },
    )

    def fail_publish(_path: Path, _payload: object) -> str:
        raise OSError("injected manifest publication failure")

    monkeypatch.setattr(multitask_views, "_publish_manifest", fail_publish)
    output_root = tmp_path / "rolled-back-views"
    manifest_path = tmp_path / "missing-manifest.yaml"
    with pytest.raises(OSError, match="injected"):
        multitask_views.materialize(
            repository_root=REPOSITORY_ROOT,
            legacy_manifest_path=legacy_manifest,
            legacy_views_root=legacy_root,
            raw_dataset_root=raw_root,
            output_root=output_root,
            manifest_path=manifest_path,
        )
    assert not output_root.exists()
    assert not manifest_path.exists()
    assert not manifest_path.with_name(f"{manifest_path.name}.sha256").exists()


def test_source_resolution_rejects_symlink_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    source = outside / "source.parquet"
    pq.write_table(_interactions(), source)
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(MaterializationError, match="unsafe ancestor"):
        multitask_views._resolve_under(
            root, Path("redirect/source.parquet"), field="fixture source"
        )


def test_verified_parquet_rejects_wrong_digest(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(_interactions(), source)
    with pytest.raises(MaterializationError, match="hash changed"):
        multitask_views._read_verified_parquet(source, expected_sha256="0" * 64)
