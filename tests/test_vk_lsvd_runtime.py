from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

import dcdn.data.vk_lsvd_runtime as runtime
from dcdn.data.vk_lsvd import (
    DENOMINATOR_ELIGIBLE_FIELDS,
    ORDERED_C_FIELDS,
    ORDERED_C_PLUS_P_FLAT_SCHEMA,
    ORDERED_C_PLUS_P_PLUS_E_FLAT_SCHEMA,
    ORDERED_DENOMINATOR_MASK,
    ORDERED_E_FIELDS,
    ORDERED_NUMERIC_OPERANDS,
    ORDERED_P0_FIELDS,
    ORDERED_V_FIELDS,
    denominator_mask_sha256,
    floor_indicator_name,
    ordered_schema_sha256,
    stable_row_id,
)
from dcdn.data.vk_lsvd_materialize import write_views_manifest
from dcdn.data.vk_lsvd_runtime import (
    C_PLUS_P,
    C_PLUS_P_PLUS_E,
    C_PLUS_P_PLUS_E_MASK,
    EXPECTED_CARDINALITIES,
    FEATURE_CONTRACT_SHA256,
    SOURCE_MANIFEST_SHA256,
    VKLSVDRawBatchSource,
    VKLSVDRuntimeError,
    _cache_content_payload,
    _canonical_json_sha256,
    _expected_schema,
    _inspect_npy_tensor,
    _physical_schema_sha256,
    _regime_spec,
    _safe_relative_path,
    _sha256_array,
    _sha256_file,
    build_full_scan_attestation,
    build_vk_lsvd_runtime_cache,
    load_full_scan_attestation,
)


def _denominator_scan(shard_count: int) -> dict[str, dict[str, Any]]:
    rows = shard_count * 2
    result = {}
    for name in DENOMINATOR_ELIGIBLE_FIELDS:
        invalid = shard_count if name == "p_item_show_count" else 0
        result[name] = {
            "rows": rows,
            "valid_rows": rows - invalid,
            "invalid_or_zero_rows": invalid,
            "reciprocal_floor_rows": 0,
            "nonfinite_rows": 0,
            "minimum_positive_normalized": 1.0,
            "maximum_normalized": 1.0,
        }
    return result


def _view_arrays(
    week: int,
    *,
    invalid_token: bool = False,
    nonfinite_operand: bool = False,
) -> dict[str, Any]:
    rows = 2
    arrays: dict[str, Any] = {
        "user_id": np.asarray([week * 10 + 1, week * 10 + 2], dtype=np.uint32),
        "item_id": np.asarray([week * 20 + 3, week * 20 + 4], dtype=np.uint32),
        "week": np.full(rows, week, dtype=np.uint8),
        "source_row_index0": np.arange(rows, dtype=np.uint64),
        "stable_row_id": [
            stable_row_id(week=week, source_row_index0=index) for index in range(rows)
        ],
        "label_normalized_watch_time": np.asarray([0.0, 1.0], dtype=np.float32),
        "label_timespent_seconds": np.asarray([0.0, 157.0], dtype=np.float32),
    }
    for position, name in enumerate(ORDERED_C_FIELDS):
        values = np.asarray([2, 3], dtype=np.uint32)
        if invalid_token and position == 0:
            values[1] = EXPECTED_CARDINALITIES[0]
        arrays[name] = values
    for name in ORDERED_NUMERIC_OPERANDS:
        values = np.ones(rows, dtype=np.float32)
        if name.startswith("v_"):
            values[:] = 1.0
        if name in {
            floor_indicator_name(field) for field in DENOMINATOR_ELIGIBLE_FIELDS
        }:
            values[:] = 0.0
        if name == "p_item_show_count":
            values[0] = 0.0
        if nonfinite_operand and name == ORDERED_NUMERIC_OPERANDS[0]:
            values[1] = np.nan
        arrays[name] = values
    for position, name in enumerate(ORDERED_E_FIELDS):
        arrays[name] = np.asarray([position + 0.25, position + 0.5], dtype=np.float32)
    for name in ORDERED_NUMERIC_OPERANDS:
        values = np.ones(rows, dtype=np.bool_)
        if name == "p_item_show_count":
            values[0] = False
        arrays[f"denominator_validity__{name}"] = values
    return arrays


def _table(arrays: dict[str, Any], *, engineered: bool, paired_delta: bool) -> pa.Table:
    schema = _expected_schema(include_engineered=engineered)
    copied = dict(arrays)
    if engineered and paired_delta:
        copied[ORDERED_NUMERIC_OPERANDS[0]] = copied[ORDERED_NUMERIC_OPERANDS[0]].copy()
        copied[ORDERED_NUMERIC_OPERANDS[0]][0] += np.float32(0.5)
    return pa.Table.from_arrays(
        [pa.array(copied[field.name], type=field.type) for field in schema],
        schema=schema,
    )


def _shared_digest(arrays: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in (*ORDERED_C_FIELDS, *ORDERED_NUMERIC_OPERANDS):
        _sha256_array(digest, name, arrays[name])
    return digest.hexdigest()


def _row_id_digest(arrays: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for value in arrays["stable_row_id"]:
        digest.update(value.encode("ascii") + b"\n")
    return digest.hexdigest()


def _make_views(
    root: Path,
    *,
    invalid_token: bool = False,
    nonfinite_operand: bool = False,
    paired_delta: bool = False,
) -> tuple[Path, Path]:
    views_root = root / "views"
    manifest_path = root / "views.yaml"
    shards = []
    for week in range(20, 26):
        role = "train" if week < 25 else "validation"
        arrays = _view_arrays(
            week,
            invalid_token=invalid_token,
            nonfinite_operand=nonfinite_operand,
        )
        cp_relative = Path("C_plus_P") / role / f"week_{week}.parquet"
        cpe_relative = Path("C_plus_P_plus_E") / role / f"week_{week}.parquet"
        cp_path = views_root / cp_relative
        cpe_path = views_root / cpe_relative
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        cpe_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_table(arrays, engineered=False, paired_delta=False), cp_path)
        pq.write_table(
            _table(arrays, engineered=True, paired_delta=paired_delta), cpe_path
        )
        shards.append(
            {
                "week": week,
                "role": role,
                "rows": 2,
                "C_plus_P": {
                    "path": cp_relative.as_posix(),
                    "bytes": cp_path.stat().st_size,
                    "sha256": _sha256_file(cp_path),
                    "row_groups": 1,
                },
                "C_plus_P_plus_E": {
                    "path": cpe_relative.as_posix(),
                    "bytes": cpe_path.stat().st_size,
                    "sha256": _sha256_file(cpe_path),
                    "row_groups": 1,
                },
                "shared_C_plus_P_values_sha256": _shared_digest(arrays),
                "stable_row_ids_sha256": _row_id_digest(arrays),
            }
        )
    mappings = {
        name: {"cardinality_with_reserved_tokens": cardinality}
        for name, cardinality in zip(
            ORDERED_C_FIELDS, EXPECTED_CARDINALITIES, strict=True
        )
    }
    payload = {
        "schema_version": 1,
        "dataset_id": "vk_lsvd_up0.01_ir0.01",
        "artifact_role": "train_validation_feature_views",
        "materialization_state": "complete_and_hash_verified",
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "test_unlocked": False,
        "test_labels_opened": False,
        "test_view_materialized": False,
        "week26_file_open_attempted": False,
        "validation_model_metrics_computed": False,
        "test_metrics_computed": False,
        "split": {
            "history_weeks": list(range(20)),
            "train_weeks": list(range(20, 25)),
            "validation_week": 25,
            "sealed_test_week": 26,
        },
        "train_only_fit": {
            "categorical_mappings": mappings,
            "normalization": {
                "method": "divide_by_train_abs_q99",
                "quantile": 0.99,
                "centering": "none",
                "clipping": "none",
                "P0_scales": {name: 1.0 for name in ORDERED_P0_FIELDS},
                "E_scales": {name: 1.0 for name in ORDERED_E_FIELDS},
            },
        },
        "feature_algebra": {
            "ordered_C": list(ORDERED_C_FIELDS),
            "ordered_P0": list(ORDERED_P0_FIELDS),
            "ordered_V": list(ORDERED_V_FIELDS),
            "ordered_P": list(ORDERED_NUMERIC_OPERANDS),
            "ordered_E": list(ORDERED_E_FIELDS),
            "dimensions": {"C": 6, "P0": 31, "V": 27, "P": 58, "E": 20},
            "ordered_operand_schema_sha256": ordered_schema_sha256(
                ORDERED_NUMERIC_OPERANDS
            ),
            "C_plus_P_expanded_schema_sha256": ordered_schema_sha256(
                ORDERED_C_PLUS_P_FLAT_SCHEMA
            ),
            "C_plus_P_plus_E_expanded_schema_sha256": ordered_schema_sha256(
                ORDERED_C_PLUS_P_PLUS_E_FLAT_SCHEMA
            ),
            "ordered_denominator_mask": list(ORDERED_DENOMINATOR_MASK),
            "denominator_eligible_fields": list(DENOMINATOR_ELIGIBLE_FIELDS),
            "denominator_epsilon": 0.01,
            "C_plus_P_physical_schema_sha256": _physical_schema_sha256(
                _expected_schema(include_engineered=False)
            ),
            "C_plus_P_plus_E_physical_schema_sha256": _physical_schema_sha256(
                _expected_schema(include_engineered=True)
            ),
        },
        "row_counts": {"train": 10, "validation": 2, "total": 12},
        "denominator_scan": _denominator_scan(6),
        "shards": shards,
    }
    write_views_manifest(manifest_path, payload)
    return manifest_path, views_root


def _scan(tmp_path: Path, **mutations: bool) -> tuple[Path, Path, Path]:
    views_manifest, views_root = _make_views(tmp_path, **mutations)
    scan_manifest = tmp_path / "scan.yaml"
    build_full_scan_attestation(
        views_manifest_path=views_manifest,
        views_root=views_root,
        output_manifest_path=scan_manifest,
        enforce_production_locks=False,
    )
    return views_manifest, views_root, scan_manifest


def _resign_yaml(path: Path, payload: dict[str, Any], *, prefix: bytes = b"") -> str:
    data = prefix + yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    ).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    path.write_bytes(data)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _resign_json(path: Path, payload: dict[str, Any]) -> str:
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        + b"\n"
    )
    digest = hashlib.sha256(data).hexdigest()
    path.write_bytes(data)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _build_cache(
    tmp_path: Path,
    *,
    feature_regime: str,
    suffix: str,
) -> tuple[Path, Path, dict[str, Any], str, str]:
    fixture_root = tmp_path / f"fixture_{suffix}"
    views_manifest, views_root, scan_manifest = _scan(fixture_root)
    _, scan_digest = load_full_scan_attestation(
        scan_manifest, enforce_production_locks=False
    )
    cache_root = tmp_path / "runtime" / suffix
    metadata, metadata_digest = build_vk_lsvd_runtime_cache(
        views_manifest_path=views_manifest,
        views_root=views_root,
        full_scan_manifest_path=scan_manifest,
        cache_root=cache_root,
        feature_regime=feature_regime,  # type: ignore[arg-type]
        expected_scan_sha256=scan_digest,
        enforce_production_locks=False,
    )
    return (
        cache_root,
        scan_manifest,
        metadata,
        metadata_digest,
        scan_digest,
    )


def test_full_scan_and_both_raw_regimes_preserve_exact_identity_and_order(
    tmp_path: Path,
) -> None:
    views_manifest, views_root, scan_manifest = _scan(tmp_path / "fixture")
    scan, scan_digest = load_full_scan_attestation(
        scan_manifest,
        expected_scan_sha256=_sha256_file(scan_manifest),
        enforce_production_locks=False,
    )
    assert scan["row_counts"] == {"train": 10, "validation": 2, "total": 12}
    assert scan["schema_identity"]["denominator_mask_sha256"] == {
        "C_plus_P": denominator_mask_sha256(ORDERED_DENOMINATOR_MASK),
        "C_plus_P_plus_E": denominator_mask_sha256(C_PLUS_P_PLUS_E_MASK),
    }
    assert len(scan["shards"]) == 6
    assert (
        sum(
            len([record["C_plus_P"], record["C_plus_P_plus_E"]])
            for record in scan["shards"]
        )
        == 12
    )

    batches = {}
    for regime, width, suffix in (
        (C_PLUS_P, 58, "cp"),
        (C_PLUS_P_PLUS_E, 78, "cpe"),
    ):
        cache_root = tmp_path / "runtime" / suffix
        metadata, metadata_digest = build_vk_lsvd_runtime_cache(
            views_manifest_path=views_manifest,
            views_root=views_root,
            full_scan_manifest_path=scan_manifest,
            cache_root=cache_root,
            feature_regime=regime,
            expected_scan_sha256=scan_digest,
            enforce_production_locks=False,
        )
        assert metadata_digest != metadata["cache_content_sha256"]
        assert metadata["feature_regime"] == regime
        assert (
            metadata["splits"]["train"]["arrays"]["operand_features"]["dtype"] == "<f4"
        )
        assert (
            metadata["splits"]["train"]["arrays"]["denominator_validity"]["dtype"]
            == "|u1"
        )
        sources = {}
        for split in ("train", "validation"):
            sources[split] = VKLSVDRawBatchSource(
                cache_root,
                split=split,
                full_scan_manifest_path=scan_manifest,
                feature_regime=regime,
                expected_scan_sha256=scan_digest,
                expected_cache_content_sha256=metadata["cache_content_sha256"],
                enforce_production_locks=False,
            )
        train_source = sources["train"]
        validation_source = sources["validation"]
        differing = {
            name
            for name, value in train_source.identity.__dict__.items()
            if value != validation_source.identity.__dict__[name]
        }
        assert differing == {"split", "row_count"}
        requested = [9, 0, 9, 2]
        batch = train_source.batch(requested, device=torch.device("cpu"))
        assert batch.global_row_indices.tolist() == requested
        assert batch.context_tokens.dtype == torch.int64
        assert batch.operand_features.dtype == torch.float32
        assert batch.operand_features.shape == (4, width)
        assert batch.denominator_validity.dtype == torch.bool
        assert batch.denominator_validity.shape == (4, width)
        assert train_source.identity.dataset_id == "vk_lsvd_up0.01_ir0.01"
        assert (
            train_source.identity.cache_content_sha256
            == metadata["cache_content_sha256"]
        )
        assert train_source.identity.full_materialized_scan_sha256 == scan_digest
        if regime == C_PLUS_P_PLUS_E:
            assert not bool(batch.denominator_validity[:, 58:].any())
        batches[regime] = batch
    assert torch.equal(
        batches[C_PLUS_P].context_tokens,
        batches[C_PLUS_P_PLUS_E].context_tokens,
    )
    assert torch.equal(
        batches[C_PLUS_P].operand_features,
        batches[C_PLUS_P_PLUS_E].operand_features[:, :58],
    )
    assert torch.equal(
        batches[C_PLUS_P].targets,
        batches[C_PLUS_P_PLUS_E].targets,
    )
    with pytest.raises(VKLSVDRuntimeError, match="only train and validation"):
        VKLSVDRawBatchSource(
            tmp_path / "runtime" / "cp",
            split="test",  # type: ignore[arg-type]
            full_scan_manifest_path=scan_manifest,
            enforce_production_locks=False,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"invalid_token": True}, "context token range"),
        ({"nonfinite_operand": True}, "not finite FP32"),
        ({"paired_delta": True}, "paired C\\+P shared-value mismatch"),
    ],
)
def test_full_scan_rejects_numeric_and_pair_mutations(
    tmp_path: Path, mutation: dict[str, bool], message: str
) -> None:
    with pytest.raises(VKLSVDRuntimeError, match=message):
        _scan(tmp_path, **mutation)


def test_scan_loader_rejects_truncation_structural_mutations_and_source_drift(
    tmp_path: Path,
) -> None:
    _, _, scan_manifest = _scan(tmp_path / "fixture")
    base, _ = load_full_scan_attestation(scan_manifest, enforce_production_locks=False)
    mutations = [
        (
            "truncated",
            lambda payload: payload.pop("numeric_ranges"),
            "keys changed",
        ),
        (
            "five_shards",
            lambda payload: payload["shards"].pop(),
            "exactly six",
        ),
        (
            "algorithm",
            lambda payload: payload.__setitem__("algorithm_source_sha256", "b" * 64),
            "algorithm source",
        ),
        (
            "mask",
            lambda payload: payload["schema_identity"][
                "denominator_mask_sha256"
            ].__setitem__("C_plus_P_plus_E", "b" * 64),
            "denominator_mask_sha256 changed",
        ),
        (
            "token",
            lambda payload: payload["shards"][0]["token_bounds"]["age"].__setitem__(
                "maximum", EXPECTED_CARDINALITIES[0]
            ),
            "token bounds are invalid",
        ),
        (
            "denominator",
            lambda payload: payload["denominator_semantics"].__setitem__(
                "active_values_strictly_positive", False
            ),
            "must be true",
        ),
    ]
    for name, mutation, message in mutations:
        payload = copy.deepcopy(base)
        mutation(payload)
        mutated = tmp_path / f"{name}.yaml"
        digest = _resign_yaml(mutated, payload)
        with pytest.raises(VKLSVDRuntimeError, match=message):
            load_full_scan_attestation(
                mutated,
                expected_scan_sha256=digest,
                enforce_production_locks=False,
            )


def test_scan_manifest_symlink_fails_closed(tmp_path: Path) -> None:
    _, _, scan_manifest = _scan(tmp_path / "fixture")
    real_manifest = tmp_path / "fixture" / "scan.real.yaml"
    scan_manifest.replace(real_manifest)
    scan_manifest.symlink_to(real_manifest.name)
    with pytest.raises(VKLSVDRuntimeError, match="securely open manifest"):
        load_full_scan_attestation(
            scan_manifest,
            enforce_production_locks=False,
        )


def test_scan_manifest_consumes_verified_bytes_after_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, scan_manifest = _scan(tmp_path / "fixture")
    original_bytes = scan_manifest.read_bytes()
    original_digest = hashlib.sha256(original_bytes).hexdigest()
    replacement = tmp_path / "replacement.yaml"
    replacement.write_text("schema_version: 999\n", encoding="utf-8")
    original_yaml_load = runtime.yaml.load

    def replace_after_verified(data: str | bytes, *args: Any, **kwargs: Any) -> Any:
        replacement.replace(scan_manifest)
        return original_yaml_load(data, *args, **kwargs)

    monkeypatch.setattr(runtime.yaml, "load", replace_after_verified)
    loaded, digest = load_full_scan_attestation(
        scan_manifest,
        expected_scan_sha256=original_digest,
        enforce_production_locks=False,
    )
    assert digest == original_digest
    assert loaded["schema_version"] == 1
    assert hashlib.sha256(scan_manifest.read_bytes()).hexdigest() != original_digest


def test_independent_scan_pin_rejects_resigned_equivalent_payload(
    tmp_path: Path,
) -> None:
    _, _, scan_manifest = _scan(tmp_path / "fixture")
    payload, original_digest = load_full_scan_attestation(
        scan_manifest, enforce_production_locks=False
    )
    resigned = tmp_path / "resigned.yaml"
    resigned_digest = _resign_yaml(resigned, payload, prefix=b"# equivalent\n")
    assert resigned_digest != original_digest
    with pytest.raises(VKLSVDRuntimeError, match="independently frozen pin"):
        load_full_scan_attestation(
            resigned,
            expected_scan_sha256=original_digest,
            enforce_production_locks=False,
        )
    loaded, digest = load_full_scan_attestation(
        resigned,
        expected_scan_sha256=resigned_digest,
        enforce_production_locks=False,
    )
    assert loaded == payload
    assert digest == resigned_digest
    with pytest.raises(VKLSVDRuntimeError, match="expected_scan_sha256"):
        load_full_scan_attestation(
            resigned,
            require_expected_digest=True,
            enforce_production_locks=False,
        )


def test_cache_rejects_output_reuse_raw_byte_drift_and_self_signed_content(
    tmp_path: Path,
) -> None:
    (
        cache_root,
        scan_manifest,
        metadata,
        _,
        scan_digest,
    ) = _build_cache(tmp_path, feature_regime=C_PLUS_P, suffix="original")
    with pytest.raises(VKLSVDRuntimeError, match="already exists"):
        build_vk_lsvd_runtime_cache(
            views_manifest_path=tmp_path / "fixture_original" / "views.yaml",
            views_root=tmp_path / "fixture_original" / "views",
            full_scan_manifest_path=scan_manifest,
            cache_root=cache_root,
            feature_regime=C_PLUS_P,
            expected_scan_sha256=scan_digest,
            enforce_production_locks=False,
        )
    operand_path = cache_root / "train" / "operand_features.f32.npy"
    with operand_path.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))
    with pytest.raises(VKLSVDRuntimeError, match="tensor evidence changed"):
        VKLSVDRawBatchSource(
            cache_root,
            split="train",
            full_scan_manifest_path=scan_manifest,
            feature_regime=C_PLUS_P,
            expected_scan_sha256=scan_digest,
            expected_cache_content_sha256=metadata["cache_content_sha256"],
            enforce_production_locks=False,
        )

    (
        resigned_root,
        resigned_scan,
        resigned_metadata,
        _,
        resigned_scan_digest,
    ) = _build_cache(tmp_path, feature_regime=C_PLUS_P, suffix="resigned")
    old_content_digest = resigned_metadata["cache_content_sha256"]
    resigned_operand = resigned_root / "train" / "operand_features.f32.npy"
    with resigned_operand.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))
    metadata_path = resigned_root / "cache_manifest.json"
    payload = json.loads(metadata_path.read_bytes())
    reopened, inspected = _inspect_npy_tensor(
        resigned_operand, expected_dtype="<f4", expected_shape=(10, 58)
    )
    del reopened
    payload["splits"]["train"]["arrays"]["operand_features"] = {
        "path": "train/operand_features.f32.npy",
        **inspected,
    }
    content_identity = _cache_content_payload(payload, _regime_spec(C_PLUS_P))
    payload["content_identity"] = content_identity
    payload["cache_content_sha256"] = _canonical_json_sha256(content_identity)
    assert payload["cache_content_sha256"] != old_content_digest
    _resign_json(metadata_path, payload)
    with pytest.raises(VKLSVDRuntimeError, match="independently frozen pin"):
        VKLSVDRawBatchSource(
            resigned_root,
            split="train",
            full_scan_manifest_path=resigned_scan,
            feature_regime=C_PLUS_P,
            expected_scan_sha256=resigned_scan_digest,
            expected_cache_content_sha256=old_content_digest,
            enforce_production_locks=False,
        )


def test_cache_metadata_is_exact_and_production_pins_cannot_be_bypassed(
    tmp_path: Path,
) -> None:
    (
        cache_root,
        scan_manifest,
        metadata,
        _,
        scan_digest,
    ) = _build_cache(tmp_path, feature_regime=C_PLUS_P, suffix="metadata")
    content_digest = metadata["cache_content_sha256"]
    with pytest.raises(VKLSVDRuntimeError, match="requires expected_scan_sha256"):
        VKLSVDRawBatchSource(
            cache_root,
            split="train",
            full_scan_manifest_path=scan_manifest,
        )
    with pytest.raises(
        VKLSVDRuntimeError, match="requires expected_cache_content_sha256"
    ):
        VKLSVDRawBatchSource(
            cache_root,
            split="train",
            full_scan_manifest_path=scan_manifest,
            expected_scan_sha256=scan_digest,
        )
    with pytest.raises(VKLSVDRuntimeError, match="cannot disable"):
        VKLSVDRawBatchSource(
            cache_root,
            split="train",
            full_scan_manifest_path=scan_manifest,
            expected_scan_sha256=scan_digest,
            expected_cache_content_sha256=content_digest,
            verify_hashes=False,
        )

    metadata_path = cache_root / "cache_manifest.json"
    truncated = json.loads(metadata_path.read_bytes())
    truncated.pop("content_identity")
    _resign_json(metadata_path, truncated)
    with pytest.raises(VKLSVDRuntimeError, match="keys changed"):
        VKLSVDRawBatchSource(
            cache_root,
            split="train",
            full_scan_manifest_path=scan_manifest,
            feature_regime=C_PLUS_P,
            expected_scan_sha256=scan_digest,
            expected_cache_content_sha256=content_digest,
            enforce_production_locks=False,
        )


class _NeverMaterializedSequence(Sequence[int]):
    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> int:
        raise AssertionError(f"oversized Sequence was materialized at {index}")


def test_batch_limit_fails_before_materialization_and_rejects_generators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        cache_root,
        scan_manifest,
        metadata,
        _,
        scan_digest,
    ) = _build_cache(tmp_path, feature_regime=C_PLUS_P, suffix="batch")
    source = VKLSVDRawBatchSource(
        cache_root,
        split="train",
        full_scan_manifest_path=scan_manifest,
        feature_regime=C_PLUS_P,
        expected_scan_sha256=scan_digest,
        expected_cache_content_sha256=metadata["cache_content_sha256"],
        enforce_production_locks=False,
    )
    monkeypatch.setattr(runtime, "MAX_BATCH_ROWS", 4)
    boundary = [3, 1, 3, 0]
    batch = source.batch(boundary, device=torch.device("cpu"))
    assert batch.global_row_indices.tolist() == boundary
    with pytest.raises(VKLSVDRuntimeError, match="MAX_BATCH_ROWS"):
        source.batch(_NeverMaterializedSequence(5), device=torch.device("cpu"))
    with pytest.raises(VKLSVDRuntimeError, match="bounded Sequence"):
        source.batch((index for index in [1, 2]), device=torch.device("cpu"))


@pytest.mark.parametrize(
    "relative",
    (
        "test/file.npy",
        "safe/test_marker/file.npy",
        "safe/week_26/file.npy",
        "safe/week_26_marker/file.npy",
    ),
)
def test_every_relative_path_component_rejects_test_markers(relative: str) -> None:
    with pytest.raises(VKLSVDRuntimeError, match="sealed test"):
        _safe_relative_path(relative, "mutation")
    assert isinstance(PurePosixPath(relative), PurePosixPath)


def test_source_runtime_value_checks_are_frozen_and_predicate_free_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root, scan_manifest, metadata, _, scan_digest = _build_cache(
        tmp_path, feature_regime=C_PLUS_P, suffix="runtime-checks"
    )
    optimized = VKLSVDRawBatchSource(
        cache_root,
        split="train",
        full_scan_manifest_path=scan_manifest,
        feature_regime=C_PLUS_P,
        expected_scan_sha256=scan_digest,
        expected_cache_content_sha256=metadata["cache_content_sha256"],
        enforce_production_locks=False,
        runtime_value_checks=False,
    )
    debug = VKLSVDRawBatchSource(
        cache_root,
        split="train",
        full_scan_manifest_path=scan_manifest,
        feature_regime=C_PLUS_P,
        expected_scan_sha256=scan_digest,
        expected_cache_content_sha256=metadata["cache_content_sha256"],
        enforce_production_locks=False,
    )
    assert optimized.runtime_value_checks is False
    assert debug.runtime_value_checks is True

    def forbidden_predicate(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("synchronous tensor value predicate executed")

    monkeypatch.setattr(torch, "isfinite", forbidden_predicate)
    batch = optimized.batch([0, 1], device=torch.device("cpu"))
    assert batch.global_row_indices.tolist() == [0, 1]
    with pytest.raises(AssertionError, match="synchronous tensor value predicate"):
        debug.batch([0, 1], device=torch.device("cpu"))
    optimized.close()
    debug.close()


def test_cache_array_symlink_fails_closed(
    tmp_path: Path,
) -> None:
    cache_root, scan_manifest, metadata, _, scan_digest = _build_cache(
        tmp_path, feature_regime=C_PLUS_P, suffix="array-symlink"
    )
    operand_path = cache_root / "train" / "operand_features.f32.npy"
    real_path = cache_root / "train" / "operand_features.real.npy"
    operand_path.replace(real_path)
    operand_path.symlink_to(real_path.name)
    with pytest.raises(VKLSVDRuntimeError, match="securely open"):
        VKLSVDRawBatchSource(
            cache_root,
            split="train",
            full_scan_manifest_path=scan_manifest,
            feature_regime=C_PLUS_P,
            expected_scan_sha256=scan_digest,
            expected_cache_content_sha256=metadata["cache_content_sha256"],
            enforce_production_locks=False,
        )


def test_source_consumes_verified_inode_after_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root, scan_manifest, metadata, _, scan_digest = _build_cache(
        tmp_path, feature_regime=C_PLUS_P, suffix="inode-replacement"
    )
    operand_path = cache_root / "train" / "operand_features.f32.npy"
    replacement_path = cache_root / "replacement.npy"
    replacement = np.lib.format.open_memmap(
        replacement_path,
        mode="w+",
        dtype="<f4",
        shape=(10, len(ORDERED_NUMERIC_OPERANDS)),
    )
    replacement[:] = np.float32(7.0)
    replacement.flush()
    del replacement

    original_validate = runtime._validate_cache_metadata

    def replace_after_verified(*args: Any, **kwargs: Any) -> str:
        digest = original_validate(*args, **kwargs)
        if kwargs.get("retained_arrays") is not None:
            replacement_path.replace(operand_path)
        return digest

    monkeypatch.setattr(runtime, "_validate_cache_metadata", replace_after_verified)
    source = VKLSVDRawBatchSource(
        cache_root,
        split="train",
        full_scan_manifest_path=scan_manifest,
        feature_regime=C_PLUS_P,
        expected_scan_sha256=scan_digest,
        expected_cache_content_sha256=metadata["cache_content_sha256"],
        enforce_production_locks=False,
    )
    batch = source.batch([0], device=torch.device("cpu"))
    assert float(batch.operand_features[0, 0]) == 1.0
    assert float(np.load(operand_path, mmap_mode="r")[0, 0]) == 7.0
    source.close()
