from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dcdn.data.vk_lsvd_multitask_runtime import (
    DATASET_ID,
    EXPECTED_CONTEXT_FIELDS,
    EXPECTED_TARGET_KINDS,
    EXPECTED_TARGET_NAMES,
    FEATURE_POLICY_SHA256,
    MATRIX_MULTITASK_SHA256,
    TARGETS_V2_SHA256,
    VIEWS_MANIFEST_SHA256,
    VKLSVDMultiTargetRuntimeError,
    VKLSVDMultiTargetSource,
    _load_views_manifest,
    frozen_target_spec,
    regime_spec,
)
from dcdn.data.vk_lsvd_runtime import EXPECTED_CARDINALITIES
from dcdn.models.contracts import canonical_sha256
from dcdn.multitask import TaskConstants

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VIEWS_MANIFEST = REPOSITORY_ROOT / "data/manifests/vk_lsvd_multitask_views_v2.yaml"
FAKE_SCAN_SHA256 = hashlib.sha256(b"fixture scan").hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_json(path: Path, payload: dict[str, object]) -> str:
    data = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    digest = hashlib.sha256(data).hexdigest()
    path.write_bytes(data)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _fake_cache(root: Path) -> tuple[str, str, np.ndarray]:
    split = root / "train"
    split.mkdir(parents=True)
    arrays = {
        "context_tokens": np.asarray(
            [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 1, 3, 3, 3, 3]],
            dtype=np.int32,
        ),
        "operand_features": np.asarray(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
            dtype=np.float32,
        ),
        "denominator_validity": np.asarray(
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.uint8
        ),
        "targets": np.asarray(
            [
                [0.2, 1, 0, 0, 1, 0, 1],
                [0.4, 0, 0, 1, 0, 1, 0],
                [0.6, 1, 1, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
        "target_masks": np.ones((3, 7), dtype=np.uint8),
        "user_ids": np.asarray([2**40 + 1, 2**40 + 2, 2**40 + 1], dtype=np.int64),
    }
    filenames = {
        "context_tokens": "context_tokens.i32.npy",
        "operand_features": "operand_features.f32.npy",
        "denominator_validity": "denominator_validity.u8.npy",
        "targets": "targets.f32.npy",
        "target_masks": "target_masks.u8.npy",
        "user_ids": "user_ids.i64.npy",
    }
    records: dict[str, object] = {}
    for name, array in arrays.items():
        path = split / filenames[name]
        np.save(path, array, allow_pickle=False)
        records[name] = {
            "path": f"train/{path.name}",
            "bytes": path.stat().st_size,
            "shape": list(array.shape),
            "sha256": _sha256(path),
        }
    scan_sha256 = FAKE_SCAN_SHA256
    constants = TaskConstants(
        spec=frozen_target_spec(),
        values=(0.02666666666666666, 0.5, 0.4, 0.3, 0.2, 0.4, 0.5),
        observed_counts=(3, 3, 3, 3, 3, 3, 3),
    )
    content = {
        "views_manifest_sha256": VIEWS_MANIFEST_SHA256,
        "full_materialized_scan_sha256": scan_sha256,
        "targets_multitask_v2_sha256": TARGETS_V2_SHA256,
        "feature_policy_sha256": FEATURE_POLICY_SHA256,
        "matrix_multitask_sha256": MATRIX_MULTITASK_SHA256,
        "feature_regime": "C+P",
        "ordered_context_fields": list(EXPECTED_CONTEXT_FIELDS),
        "ordered_operand_fields": ["p_positive", "p_cosine", "p_other"],
        "denominator_mask": [True, False, False],
        "target_names": list(EXPECTED_TARGET_NAMES),
        "target_kinds": list(EXPECTED_TARGET_KINDS),
        "task_constants": constants.contract_state(),
        "task_constants_sha256": constants.contract_sha256,
        "splits": {"train": {"rows": 3, "arrays": records}},
    }
    content_sha256 = canonical_sha256(content)
    metadata = {
        "schema_version": 1,
        "artifact_role": "vk_lsvd_multitask_raw_memmap_cache_v1",
        "cache_state": "complete_and_hash_verified",
        "dataset_id": DATASET_ID,
        "cache_content_sha256": content_sha256,
        "content_identity": content,
        "physical_schema_sha256": "2" * 64,
        "expanded_schema_sha256": "3" * 64,
        "operand_schema_sha256": canonical_sha256(
            tuple(content["ordered_operand_fields"])
        ),
        "denominator_mask_sha256": canonical_sha256(tuple(content["denominator_mask"])),
        "categorical_cardinalities": list(EXPECTED_CARDINALITIES),
        "test_split_present": False,
        "model_metrics_computed": False,
    }
    metadata_sha256 = _publish_json(root / "cache.json", metadata)
    return metadata_sha256, content_sha256, arrays["user_ids"]


def test_frozen_manifest_keeps_cosines_and_masks_them_from_reciprocals() -> None:
    payload, digest = _load_views_manifest(VIEWS_MANIFEST)
    assert digest == VIEWS_MANIFEST_SHA256
    reduced = regime_spec(payload, "C+P")
    full = regime_spec(payload, "C+P+E")
    for name in (
        "e_ials_viewtime_cosine",
        "e_ials_feedback_cosine",
        "e_ials_survival_cosine",
    ):
        index = reduced.ordered_operands.index(name)
        assert reduced.denominator_mask[index] is False
    assert len(full.ordered_operands) == len(reduced.ordered_operands) + 17
    assert all(value is False for value in full.denominator_mask[-17:])


def test_source_preserves_large_user_ids_for_uniform_user_gauc(tmp_path: Path) -> None:
    metadata_sha256, content_sha256, user_ids = _fake_cache(tmp_path)
    source = VKLSVDMultiTargetSource(
        tmp_path,
        split="train",
        expected_cache_metadata_sha256=metadata_sha256,
        expected_cache_content_sha256=content_sha256,
        expected_scan_sha256=FAKE_SCAN_SHA256,
        runtime_value_checks=True,
    )
    requested = [2, 0, 2]
    batch = source.batch(requested, device=torch.device("cpu"))
    assert batch.user_ids.tolist() == user_ids[requested].tolist()
    assert batch.global_row_indices.tolist() == requested
    assert batch.targets.shape == (3, 7)
    assert batch.target_masks.all()


def test_source_rejects_test_split_and_tampered_array(tmp_path: Path) -> None:
    metadata_sha256, content_sha256, _ = _fake_cache(tmp_path)
    with pytest.raises(VKLSVDMultiTargetRuntimeError, match="train/validation"):
        VKLSVDMultiTargetSource(
            tmp_path,
            split="test",  # type: ignore[arg-type]
            expected_cache_metadata_sha256=metadata_sha256,
            expected_cache_content_sha256=content_sha256,
            expected_scan_sha256=FAKE_SCAN_SHA256,
            runtime_value_checks=True,
        )
    path = tmp_path / "train/targets.f32.npy"
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(VKLSVDMultiTargetRuntimeError, match="array SHA"):
        VKLSVDMultiTargetSource(
            tmp_path,
            split="train",
            expected_cache_metadata_sha256=metadata_sha256,
            expected_cache_content_sha256=content_sha256,
            expected_scan_sha256=FAKE_SCAN_SHA256,
            runtime_value_checks=True,
        )
