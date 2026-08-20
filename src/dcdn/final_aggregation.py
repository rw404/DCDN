"""Immutable VK-LSVD RQ1 aggregation from one-pass final metric bundles."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dcdn.final_evaluation import (
    load_verified_final_bundle,
    preflight_final_bundle_path,
    publish_final_bundle,
)
from dcdn.statistics import holm_adjust, paired_cluster_bootstrap_sufficient

SHA256_RE = re.compile(r"[0-9a-f]{64}")
FINAL_SEEDS = (42, 43, 44)
DATASET_ID = "vk_lsvd_up0.01_ir0.01"
C_PLUS_P = "C+P"
C_PLUS_P_PLUS_E = "C+P+E"
DCDN = "dcdn_core"
DCNV2 = "dcnv2_full_rank"
CAPACITY = "dcnv2_capacity_matched_core"
EXPECTED_CELLS = frozenset(
    {
        (C_PLUS_P, DCDN),
        (C_PLUS_P, DCNV2),
        (C_PLUS_P_PLUS_E, DCNV2),
        (C_PLUS_P, CAPACITY),
    }
)
PRIMARY_CONTRASTS = (
    ("DCDN(C+P)-DCNv2(C+P)", (C_PLUS_P, DCDN), (C_PLUS_P, DCNV2)),
    (
        "DCDN(C+P)-DCNv2(C+P+E)",
        (C_PLUS_P, DCDN),
        (C_PLUS_P_PLUS_E, DCNV2),
    ),
    (
        "DCDN(C+P)-DCNv2-capacity-matched-core(C+P)",
        (C_PLUS_P, DCDN),
        (C_PLUS_P, CAPACITY),
    ),
    (
        "DCNv2(C+P+E)-DCNv2(C+P)",
        (C_PLUS_P_PLUS_E, DCNV2),
        (C_PLUS_P, DCNV2),
    ),
)


class FinalAggregationError(RuntimeError):
    """Raised when final bundles cannot form the frozen RQ1 family."""


@dataclass(frozen=True)
class FinalBundleSpec:
    path: str
    expected_sha256: str
    feature_regime: str
    model_key: str
    seed: int


@dataclass(frozen=True)
class FinalAggregateRecord:
    path: Path
    sidecar_path: Path
    sha256: str
    size_bytes: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class _LoadedCell:
    spec: FinalBundleSpec
    digest: str
    size_bytes: int
    payload: Mapping[str, Any]
    row_count: int
    normalized_mse: float
    stable_rows_sha256: str
    user_ids_sha256: str
    targets_sha256: str
    cluster_statistics: Mapping[str, tuple[int, float]]


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FinalAggregationError(f"{field} must be numeric")
    converted = float(value)
    if not np.isfinite(converted):
        raise FinalAggregationError(f"{field} must be finite")
    return converted


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FinalAggregationError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_spec(spec: FinalBundleSpec) -> None:
    if (
        not isinstance(spec.path, str)
        or not spec.path
        or Path(spec.path).is_absolute()
        or ".." in Path(spec.path).parts
    ):
        raise FinalAggregationError("final bundle spec path is unsafe")
    _sha256(spec.expected_sha256, "expected final bundle digest")
    if (spec.feature_regime, spec.model_key) not in EXPECTED_CELLS:
        raise FinalAggregationError("final bundle spec is outside the frozen RQ1 cells")
    if type(spec.seed) is not int or spec.seed not in FINAL_SEEDS:
        raise FinalAggregationError(
            "final bundle seed is outside the frozen final seeds"
        )


def _load_cell(repository_root: Path, spec: FinalBundleSpec) -> _LoadedCell:
    _validate_spec(spec)
    _, payload, digest, size_bytes = load_verified_final_bundle(
        repository_root,
        spec.path,
    )
    if digest != spec.expected_sha256:
        raise FinalAggregationError("final bundle digest differs from frozen input")
    if set(payload) != {
        "schema_version",
        "artifact_role",
        "claim",
        "runtime_identity",
        "evaluation",
        "metrics",
        "pairing",
    }:
        raise FinalAggregationError("final bundle root keys mismatch")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise FinalAggregationError("final bundle schema mismatch")
    if payload["artifact_role"] != "one_pass_test_metric_bundle":
        raise FinalAggregationError("final bundle role mismatch")

    claim = payload["claim"]
    if not isinstance(claim, Mapping) or set(claim) != {
        "claim_id",
        "model_key",
        "seed",
        "final_protocol_sha256",
        "parent_premetric_protocol_sha256",
        "selected_configs_sha256",
        "selection_receipt_sha256",
        "candidate_id",
        "resolved_config_sha256",
        "resolved_run_config_sha256",
        "validation_result_sha256",
        "checkpoint_sha256",
        "selected_validation_metric",
    }:
        raise FinalAggregationError("final bundle claim keys mismatch")
    if claim["model_key"] != spec.model_key or claim["seed"] != spec.seed:
        raise FinalAggregationError("final bundle model/seed binding mismatch")
    for name in (
        "final_protocol_sha256",
        "parent_premetric_protocol_sha256",
        "selected_configs_sha256",
        "selection_receipt_sha256",
        "resolved_config_sha256",
        "resolved_run_config_sha256",
        "validation_result_sha256",
        "checkpoint_sha256",
    ):
        _sha256(claim[name], f"claim.{name}")
    _finite_number(claim["selected_validation_metric"], "selected validation metric")

    runtime = payload["runtime_identity"]
    if not isinstance(runtime, Mapping):
        raise FinalAggregationError("final bundle runtime identity must be a mapping")
    if runtime.get("split") != "test":
        raise FinalAggregationError("final bundle must describe the test split")
    if runtime.get("feature_regime") != spec.feature_regime:
        raise FinalAggregationError("final bundle feature-regime binding mismatch")
    dataset_id = runtime.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise FinalAggregationError("final bundle dataset identity is invalid")
    row_count = runtime.get("row_count")
    if type(row_count) is not int or row_count <= 0:
        raise FinalAggregationError("final bundle row count is invalid")

    evaluation = payload["evaluation"]
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("row_count") != row_count
        or evaluation.get("source_passes") != 1
        or type(evaluation.get("source_passes")) is not int
        or evaluation.get("source_factory_calls") != 1
        or type(evaluation.get("source_factory_calls")) is not int
        or evaluation.get("test_metrics_read") is not True
        or evaluation.get("validation_selection_changed") is not False
    ):
        raise FinalAggregationError("final bundle one-pass contract mismatch")

    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise FinalAggregationError("final bundle metrics must be a mapping")
    normalized_mse = _finite_number(metrics.get("normalized_mse"), "normalized_mse")
    if normalized_mse < 0:
        raise FinalAggregationError("normalized_mse must be non-negative")

    pairing = payload["pairing"]
    if not isinstance(pairing, Mapping):
        raise FinalAggregationError("final bundle pairing must be a mapping")
    stable_rows = _sha256(pairing.get("stable_row_ids_sha256"), "stable row digest")
    user_ids = _sha256(pairing.get("user_ids_u64le_sha256"), "user id digest")
    targets = _sha256(
        pairing.get("normalized_targets_f64le_sha256"),
        "normalized target digest",
    )
    if pairing.get("unique_stable_row_ids") != row_count:
        raise FinalAggregationError("final bundle stable row count mismatch")
    statistics = pairing.get("user_cluster_sufficient_statistics")
    if not isinstance(statistics, list) or not statistics:
        raise FinalAggregationError("final bundle user-cluster statistics are missing")
    clusters: dict[str, tuple[int, float]] = {}
    total_rows = 0
    total_loss = 0.0
    for item in statistics:
        if not isinstance(item, Mapping) or set(item) != {
            "user_cluster_sha256",
            "row_count",
            "normalized_squared_error_sum",
        }:
            raise FinalAggregationError("final cluster statistic keys mismatch")
        cluster = _sha256(item["user_cluster_sha256"], "user cluster digest")
        rows = item["row_count"]
        loss_sum = _finite_number(
            item["normalized_squared_error_sum"],
            "normalized squared-error sum",
        )
        if cluster in clusters or type(rows) is not int or rows <= 0 or loss_sum < 0:
            raise FinalAggregationError("final cluster statistic is invalid")
        clusters[cluster] = (rows, loss_sum)
        total_rows += rows
        total_loss += loss_sum
    if total_rows != row_count:
        raise FinalAggregationError("final cluster rows are incomplete")
    if not np.isclose(
        total_loss / total_rows,
        normalized_mse,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise FinalAggregationError("normalized_mse differs from cluster sums")

    return _LoadedCell(
        spec=spec,
        digest=digest,
        size_bytes=size_bytes,
        payload=payload,
        row_count=row_count,
        normalized_mse=normalized_mse,
        stable_rows_sha256=stable_rows,
        user_ids_sha256=user_ids,
        targets_sha256=targets,
        cluster_statistics=clusters,
    )


def _pooled_statistics(
    cells: Sequence[_LoadedCell],
) -> dict[str, tuple[int, float]]:
    if len(cells) != len(FINAL_SEEDS):
        raise FinalAggregationError("pooled statistics require all three final seeds")
    reference = cells[0].cluster_statistics
    pooled = {cluster: [0, 0.0] for cluster in reference}
    for cell in cells:
        if set(cell.cluster_statistics) != set(reference):
            raise FinalAggregationError("user clusters differ across final seeds")
        for cluster, (rows, loss_sum) in cell.cluster_statistics.items():
            if rows != reference[cluster][0]:
                raise FinalAggregationError(
                    "user-cluster row counts differ across seeds"
                )
            pooled[cluster][0] += rows
            pooled[cluster][1] += loss_sum
    return {
        cluster: (int(values[0]), float(values[1]))
        for cluster, values in pooled.items()
    }


def aggregate_vk_lsvd_rq1(
    *,
    repository_root: str | Path,
    bundle_specs: Sequence[FinalBundleSpec],
    result_path: str | Path,
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    random_seed: int = 20_260_818,
    familywise_alpha: float = 0.05,
) -> FinalAggregateRecord:
    """Aggregate the frozen four-contrast VK-LSVD RQ1 family exactly once."""

    if (
        type(n_resamples) is not int
        or n_resamples != 2_000
        or confidence_level != 0.95
        or type(random_seed) is not int
        or random_seed != 20_260_818
        or familywise_alpha != 0.05
    ):
        raise FinalAggregationError("statistical plan differs from the frozen contract")
    expected_keys = {
        (regime, model, seed)
        for regime, model in EXPECTED_CELLS
        for seed in FINAL_SEEDS
    }
    if len(bundle_specs) != len(expected_keys):
        raise FinalAggregationError("RQ1 aggregation requires exactly twelve bundles")
    actual_keys: set[tuple[str, str, int]] = set()
    root = Path(repository_root).resolve()
    relative = preflight_final_bundle_path(root, result_path)
    cells: dict[tuple[str, str, int], _LoadedCell] = {}
    for spec in bundle_specs:
        key = (spec.feature_regime, spec.model_key, spec.seed)
        if key in actual_keys:
            raise FinalAggregationError("duplicate final bundle cell")
        actual_keys.add(key)
        cells[key] = _load_cell(root, spec)
    if actual_keys != expected_keys:
        raise FinalAggregationError("final bundle cells differ from frozen RQ1 matrix")

    dataset_ids = {
        str(cell.payload["runtime_identity"]["dataset_id"]) for cell in cells.values()
    }
    if dataset_ids != {DATASET_ID}:
        raise FinalAggregationError("RQ1 bundles do not describe frozen VK-LSVD")
    alignment = {
        (
            cell.row_count,
            cell.stable_rows_sha256,
            cell.user_ids_sha256,
            cell.targets_sha256,
        )
        for cell in cells.values()
    }
    if len(alignment) != 1:
        raise FinalAggregationError(
            "RQ1 bundles are not exactly row/target/user aligned"
        )
    for regime in (C_PLUS_P, C_PLUS_P_PLUS_E):
        provenance = {
            (
                cell.payload["claim"]["final_protocol_sha256"],
                cell.payload["claim"]["parent_premetric_protocol_sha256"],
                cell.payload["claim"]["selected_configs_sha256"],
            )
            for cell in cells.values()
            if cell.spec.feature_regime == regime
        }
        if len(provenance) != 1:
            raise FinalAggregationError(
                "RQ1 bundles mix protocol or selected-config lineage within a regime"
            )

    model_metrics: list[dict[str, Any]] = []
    for regime, model in sorted(EXPECTED_CELLS):
        values = np.asarray(
            [cells[(regime, model, seed)].normalized_mse for seed in FINAL_SEEDS],
            dtype=np.float64,
        )
        model_metrics.append(
            {
                "feature_regime": regime,
                "model_key": model,
                "seed_values": [
                    {"seed": seed, "normalized_mse": float(value)}
                    for seed, value in zip(FINAL_SEEDS, values, strict=True)
                ],
                "mean_normalized_mse": float(values.mean()),
                "sample_standard_deviation_normalized_mse": float(values.std(ddof=1)),
            }
        )

    contrast_rows: list[dict[str, Any]] = []
    raw_p_values: list[tuple[str, float]] = []
    for label, treatment_key, baseline_key in PRIMARY_CONTRASTS:
        treatment_cells = [cells[(*treatment_key, seed)] for seed in FINAL_SEEDS]
        baseline_cells = [cells[(*baseline_key, seed)] for seed in FINAL_SEEDS]
        per_seed = [
            {
                "seed": seed,
                "effect_treatment_minus_baseline": (
                    treatment.normalized_mse - baseline.normalized_mse
                ),
            }
            for seed, treatment, baseline in zip(
                FINAL_SEEDS,
                treatment_cells,
                baseline_cells,
                strict=True,
            )
        ]
        effects = np.asarray(
            [item["effect_treatment_minus_baseline"] for item in per_seed],
            dtype=np.float64,
        )
        bootstrap = paired_cluster_bootstrap_sufficient(
            _pooled_statistics(baseline_cells),
            _pooled_statistics(treatment_cells),
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=random_seed,
            sampling_unit="user_id",
        )
        if not np.isclose(
            bootstrap.effect_treatment_minus_baseline,
            float(effects.mean()),
            rtol=1e-12,
            atol=1e-15,
        ):
            raise FinalAggregationError("pooled and seed-mean contrast effects differ")
        raw_p_values.append((label, bootstrap.two_sided_bootstrap_p_value))
        contrast_rows.append(
            {
                "label": label,
                "treatment": {
                    "feature_regime": treatment_key[0],
                    "model_key": treatment_key[1],
                },
                "baseline": {
                    "feature_regime": baseline_key[0],
                    "model_key": baseline_key[1],
                },
                "metric": "normalized_mse",
                "direction": "minimize",
                "per_seed": per_seed,
                "mean_effect_treatment_minus_baseline": float(effects.mean()),
                "sample_standard_deviation_effect": float(effects.std(ddof=1)),
                "paired_user_cluster_bootstrap": asdict(bootstrap),
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for row in contrast_rows:
        value = adjusted[row["label"]]
        row["holm_adjusted_p_value"] = value
        row["reject_at_familywise_alpha_0_05"] = value <= familywise_alpha

    bindings = []
    for key in sorted(cells):
        cell = cells[key]
        claim = cell.payload["claim"]
        bindings.append(
            {
                "path": cell.spec.path,
                "sha256": cell.digest,
                "size_bytes": cell.size_bytes,
                "feature_regime": cell.spec.feature_regime,
                "model_key": cell.spec.model_key,
                "seed": cell.spec.seed,
                "final_protocol_sha256": claim["final_protocol_sha256"],
                "parent_premetric_protocol_sha256": (
                    claim["parent_premetric_protocol_sha256"]
                ),
                "selected_configs_sha256": claim["selected_configs_sha256"],
                "checkpoint_sha256": claim["checkpoint_sha256"],
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "vk_lsvd_rq1_final_primary_aggregate",
        "dataset_id": DATASET_ID,
        "statistical_plan": {
            "final_seeds": list(FINAL_SEEDS),
            "metric": "normalized_mse",
            "direction": "minimize",
            "paired_bootstrap_cluster": "user_id",
            "paired_bootstrap_pairing_key": "stable_row_id",
            "confidence_level": confidence_level,
            "replicates": n_resamples,
            "random_seed": random_seed,
            "interval": "percentile",
            "seed_pooling": (
                "all_final_seeds_kept_together_within_each_resampled_user_v1"
            ),
            "primary_family": "four_primary_normalized_mse_contrasts",
            "multiplicity_method": "Holm-Bonferroni",
            "familywise_alpha": familywise_alpha,
        },
        "input_bundles": bindings,
        "model_metrics": model_metrics,
        "primary_contrasts": contrast_rows,
    }
    absolute, digest, size_bytes = publish_final_bundle(root, relative, payload)
    return FinalAggregateRecord(
        path=absolute,
        sidecar_path=absolute.with_name(f"{absolute.name}.sha256"),
        sha256=digest,
        size_bytes=size_bytes,
        payload=payload,
    )
