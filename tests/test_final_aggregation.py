from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dcdn.final_aggregation import (
    C_PLUS_P,
    C_PLUS_P_PLUS_E,
    CAPACITY,
    DCDN,
    DCNV2,
    FinalAggregationError,
    FinalBundleSpec,
    aggregate_vk_lsvd_rq1,
)
from dcdn.final_evaluation import preflight_final_bundle_path, publish_final_bundle

HASHES = {
    "stable": "1" * 64,
    "users": "2" * 64,
    "targets": "3" * 64,
    "prediction": "4" * 64,
    "errors": "5" * 64,
    "protocol": "6" * 64,
    "parent": "7" * 64,
    "selected": "8" * 64,
    "receipt": "9" * 64,
    "config": "a" * 64,
    "run": "b" * 64,
    "validation": "c" * 64,
    "checkpoint": "d" * 64,
}
CLUSTERS = ("e" * 64, "f" * 64, "0" * 64)
ROWS = (2, 1, 3)
BASE_MSE = {
    (C_PLUS_P, DCDN): 0.15,
    (C_PLUS_P, DCNV2): 0.20,
    (C_PLUS_P_PLUS_E, DCNV2): 0.17,
    (C_PLUS_P, CAPACITY): 0.18,
}


def _payload(
    *,
    regime: str,
    model: str,
    seed: int,
    target_sha256: str = HASHES["targets"],
) -> dict[str, Any]:
    mse = BASE_MSE[(regime, model)] + 0.001 * (seed - 42)
    statistics = [
        {
            "user_cluster_sha256": cluster,
            "row_count": rows,
            "normalized_squared_error_sum": mse * rows,
        }
        for cluster, rows in zip(CLUSTERS, ROWS, strict=True)
    ]
    return {
        "schema_version": 1,
        "artifact_role": "one_pass_test_metric_bundle",
        "claim": {
            "claim_id": f"{regime}-{model}-{seed}",
            "model_key": model,
            "seed": seed,
            "final_protocol_sha256": HASHES["protocol"],
            "parent_premetric_protocol_sha256": HASHES["parent"],
            "selected_configs_sha256": HASHES["selected"],
            "selection_receipt_sha256": HASHES["receipt"],
            "candidate_id": f"{model}-candidate",
            "resolved_config_sha256": HASHES["config"],
            "resolved_run_config_sha256": HASHES["run"],
            "validation_result_sha256": HASHES["validation"],
            "checkpoint_sha256": HASHES["checkpoint"],
            "selected_validation_metric": mse + 0.01,
        },
        "runtime_identity": {
            "dataset_id": "vk_lsvd_up0.01_ir0.01",
            "split": "test",
            "feature_regime": regime,
            "row_count": sum(ROWS),
        },
        "evaluation": {
            "row_count": sum(ROWS),
            "batch_size": 4,
            "source_passes": 1,
            "source_factory_calls": 1,
            "test_metrics_read": True,
            "validation_selection_changed": False,
            "duration_constant_seconds": 157.0,
        },
        "metrics": {
            "normalized_mse": mse,
            "mae_seconds": 1.0,
            "mse_seconds2": mse * 157.0**2,
            "concordance": {},
        },
        "pairing": {
            "stable_row_ids_sha256": HASHES["stable"],
            "user_ids_u64le_sha256": HASHES["users"],
            "normalized_targets_f64le_sha256": target_sha256,
            "normalized_predictions_f64le_sha256": HASHES["prediction"],
            "squared_errors_f64le_sha256": HASHES["errors"],
            "unique_stable_row_ids": sum(ROWS),
            "user_cluster_count": len(CLUSTERS),
            "user_cluster_sufficient_statistics": statistics,
        },
    }


def _materialize_specs(
    root: Path,
    *,
    mismatched_target_cell: tuple[str, str, int] | None = None,
) -> list[FinalBundleSpec]:
    specs: list[FinalBundleSpec] = []
    for regime, model in sorted(BASE_MSE):
        for seed in (42, 43, 44):
            safe_regime = regime.lower().replace("+", "plus")
            relative = Path(f"results/final/input-{safe_regime}-{model}-s{seed}.json")
            preflight_final_bundle_path(root, relative)
            target = (
                "b" * 64
                if mismatched_target_cell == (regime, model, seed)
                else HASHES["targets"]
            )
            _, digest, _ = publish_final_bundle(
                root,
                relative,
                _payload(
                    regime=regime,
                    model=model,
                    seed=seed,
                    target_sha256=target,
                ),
            )
            specs.append(
                FinalBundleSpec(
                    path=relative.as_posix(),
                    expected_sha256=digest,
                    feature_regime=regime,
                    model_key=model,
                    seed=seed,
                )
            )
    return specs


def test_rq1_aggregate_is_exact_paired_and_holm_adjusted(tmp_path: Path) -> None:
    specs = _materialize_specs(tmp_path)
    record = aggregate_vk_lsvd_rq1(
        repository_root=tmp_path,
        bundle_specs=specs,
        result_path="results/final/vk-rq1-primary-v1.json",
    )

    assert record.path.is_file()
    assert record.sidecar_path.is_file()
    assert len(record.sha256) == 64
    assert len(record.payload["input_bundles"]) == 12
    assert len(record.payload["model_metrics"]) == 4
    contrasts = record.payload["primary_contrasts"]
    assert [item["label"] for item in contrasts] == [
        "DCDN(C+P)-DCNv2(C+P)",
        "DCDN(C+P)-DCNv2(C+P+E)",
        "DCDN(C+P)-DCNv2-capacity-matched-core(C+P)",
        "DCNv2(C+P+E)-DCNv2(C+P)",
    ]
    assert [item["mean_effect_treatment_minus_baseline"] for item in contrasts] == (
        pytest.approx([-0.05, -0.02, -0.03, -0.03])
    )
    for item in contrasts:
        bootstrap = item["paired_user_cluster_bootstrap"]
        assert bootstrap["clusters"] == 3
        assert bootstrap["rows"] == 18
        assert bootstrap["resamples"] == 2_000
        assert 0.0 <= item["holm_adjusted_p_value"] <= 1.0
        assert item["reject_at_familywise_alpha_0_05"] is True


def test_rq1_aggregate_rejects_cross_regime_row_target_misalignment(
    tmp_path: Path,
) -> None:
    specs = _materialize_specs(
        tmp_path,
        mismatched_target_cell=(C_PLUS_P_PLUS_E, DCNV2, 44),
    )
    with pytest.raises(FinalAggregationError, match="row/target/user aligned"):
        aggregate_vk_lsvd_rq1(
            repository_root=tmp_path,
            bundle_specs=specs,
            result_path="results/final/vk-rq1-primary-v1.json",
        )
    assert not (tmp_path / "results/final/vk-rq1-primary-v1.json").exists()


def test_rq1_aggregate_rejects_missing_cell_and_plan_drift(tmp_path: Path) -> None:
    specs = _materialize_specs(tmp_path)
    with pytest.raises(FinalAggregationError, match="exactly twelve"):
        aggregate_vk_lsvd_rq1(
            repository_root=tmp_path,
            bundle_specs=specs[:-1],
            result_path="results/final/missing.json",
        )
    with pytest.raises(FinalAggregationError, match="statistical plan"):
        aggregate_vk_lsvd_rq1(
            repository_root=tmp_path,
            bundle_specs=specs,
            result_path="results/final/drift.json",
            n_resamples=1_999,
        )


def test_rq1_aggregate_rejects_changed_expected_digest(tmp_path: Path) -> None:
    specs = _materialize_specs(tmp_path)
    first = specs[0]
    specs[0] = FinalBundleSpec(
        path=first.path,
        expected_sha256="f" * 64,
        feature_regime=first.feature_regime,
        model_key=first.model_key,
        seed=first.seed,
    )
    with pytest.raises(FinalAggregationError, match="digest differs"):
        aggregate_vk_lsvd_rq1(
            repository_root=tmp_path,
            bundle_specs=specs,
            result_path="results/final/digest.json",
        )
