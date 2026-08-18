"""Fail-closed validation for frozen model provenance and search spaces."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import load_verified_yaml, sha256_file

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
TOKEN_RE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
FROZEN_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

REQUIRED_MODELS = frozenset(
    {
        "dnn",
        "deepfm",
        "dcn",
        "autoint_plus",
        "dcnv2_full_rank",
        "afn_plus",
        "wukong_small",
        "dcdn_core",
        "dcnv2_low_rank_mix_capacity",
    }
)
REQUIRED_PROVENANCE_MODELS = frozenset(
    {
        "dnn",
        "deepfm",
        "dcn",
        "autoint_plus",
        "dcnv2_full_rank",
        "dcnv2_low_rank_mix",
        "afn_plus",
        "wukong_small",
        "dcdn_core",
    }
)
EXPECTED_SOURCE_MODELS = {
    "dnn": "dnn",
    "deepfm": "deepfm",
    "dcn": "dcn",
    "autoint_plus": "autoint_plus",
    "dcnv2_full_rank": "dcnv2_full_rank",
    "afn_plus": "afn_plus",
    "wukong_small": "wukong_small",
    "dcdn_core": "dcdn_core",
    "dcnv2_low_rank_mix_capacity": "dcnv2_low_rank_mix",
}
PHASE_ORDER = {
    "definition": 0,
    "first_model_metric": 1,
    "capacity_control_training": 2,
    "protocol_frozen": 3,
    "test_access": 4,
}


class SearchSpaceError(RuntimeError):
    """Raised when provenance, search, or resolution state is unsafe."""


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SearchSpaceError(f"{field} must be a mapping")
    return value


def _as_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise SearchSpaceError(f"{field} must be a list")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SearchSpaceError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise SearchSpaceError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise SearchSpaceError(f"{field} must be a non-negative integer")
    return value


def _positive_number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise SearchSpaceError(f"{field} must be a finite positive number")
    return float(value)


def _require_bool(value: Any, expected: bool, field: str) -> None:
    if type(value) is not bool or value is not expected:
        raise SearchSpaceError(f"{field} must be {expected}")


def _require_equal(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise SearchSpaceError(f"{field} must equal {expected!r}")


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SearchSpaceError(f"{field} must be a lowercase SHA-256")
    return value


def _flatten_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    leaves: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        leaves.update(_flatten_leaves(child, path))
    return leaves


def _find_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            tokens.update(_find_tokens(child))
    elif isinstance(value, list):
        for child in value:
            tokens.update(_find_tokens(child))
    elif isinstance(value, str):
        match = TOKEN_RE.fullmatch(value)
        if match is not None:
            tokens.add(match.group(1))
    return tokens


def _replace_tokens(value: Any, resolved: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        return {key: _replace_tokens(child, resolved) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_tokens(child, resolved) for child in value]
    if isinstance(value, str):
        match = TOKEN_RE.fullmatch(value)
        if match is not None and match.group(1) in resolved:
            return resolved[match.group(1)]
    return value


def validate_model_sources(payload: Mapping[str, Any]) -> None:
    """Validate pinned code/paper provenance without consulting result values."""

    _require_equal(payload.get("schema_version"), 1, "model_sources.schema_version")
    _require_equal(
        payload.get("status"), "frozen_before_metrics", "model_sources.status"
    )
    frozen_utc = _nonempty_string(payload.get("frozen_utc"), "model_sources.frozen_utc")
    if FROZEN_UTC_RE.fullmatch(frozen_utc) is None:
        raise SearchSpaceError(
            "model_sources.frozen_utc must be a quoted UTC second ending in Z"
        )

    policy = _as_mapping(payload.get("audit_policy"), "model_sources.audit_policy")
    _require_bool(
        policy.get("published_result_values_quarantined"),
        True,
        "audit_policy.published_result_values_quarantined",
    )
    _require_bool(
        policy.get("published_result_values_used_for_selection"),
        False,
        "audit_policy.published_result_values_used_for_selection",
    )
    _require_bool(
        policy.get("readmes_used_for_selection"),
        False,
        "audit_policy.readmes_used_for_selection",
    )
    exposure = _as_mapping(
        policy.get("unexpected_published_result_exposure"),
        "audit_policy.unexpected_published_result_exposure",
    )
    _require_bool(
        exposure.get("values_recorded_or_used_for_selection"),
        False,
        "unexpected_published_result_exposure.values_recorded_or_used_for_selection",
    )

    sources = _as_mapping(payload.get("sources"), "model_sources.sources")
    pinned_sources = {
        "fuxictr": ("0245987fdc1d43c36164ea496a02324885a8557a", "Apache-2.0"),
        "afn_plus_official": (
            "25d7bd995bd0a3c71f562b3d332a0628c3b30d28",
            "MIT",
        ),
        "wukong_standalone": (
            "8a0c14ccee50e02d40201d4dc665ee9fcebeb6b9",
            "MIT",
        ),
    }
    for source_id in ("fuxictr", "afn_plus_official", "wukong_standalone"):
        source = _as_mapping(sources.get(source_id), f"sources.{source_id}")
        _require_equal(source.get("kind"), "external_code", f"sources.{source_id}.kind")
        _nonempty_string(source.get("url"), f"sources.{source_id}.url")
        revision = source.get("revision")
        if not isinstance(revision, str) or COMMIT_RE.fullmatch(revision) is None:
            raise SearchSpaceError(
                f"sources.{source_id}.revision must be a 40-hex commit"
            )
        expected_revision, expected_license = pinned_sources[source_id]
        _require_equal(
            revision, expected_revision, f"sources.{source_id}.pinned_revision"
        )
        _require_equal(
            source.get("license"), expected_license, f"sources.{source_id}.license"
        )
        _require_sha256(
            source.get("license_sha256"), f"sources.{source_id}.license_sha256"
        )
        source_files = _as_mapping(
            source.get("source_files"), f"sources.{source_id}.source_files"
        )
        if not source_files:
            raise SearchSpaceError(f"sources.{source_id}.source_files cannot be empty")
        for file_id, record_value in source_files.items():
            record = _as_mapping(
                record_value, f"sources.{source_id}.source_files.{file_id}"
            )
            _nonempty_string(
                record.get("path"),
                f"sources.{source_id}.source_files.{file_id}.path",
            )
            _require_sha256(
                record.get("sha256"),
                f"sources.{source_id}.source_files.{file_id}.sha256",
            )

    afn = _as_mapping(
        sources["afn_plus_official"].get("audited_structural_facts"),
        "sources.afn_plus_official.audited_structural_facts",
    )
    _require_bool(afn.get("ensemble"), True, "afn official ensemble")
    _require_bool(
        afn.get("separate_embedding_views"), True, "afn separate_embedding_views"
    )
    _require_equal(afn.get("log_input"), "abs_then_clamp_then_log", "afn log_input")
    _require_equal(afn.get("clamp_min"), 0.0001, "afn clamp_min")
    _require_bool(
        afn.get("official_training_has_auxiliary_branch_losses"),
        True,
        "afn official auxiliary losses",
    )
    _require_bool(
        afn.get("official_ensemble_uses_stopped_branch_logits"),
        True,
        "afn official stopped logits",
    )

    paper = _as_mapping(sources.get("wukong_paper"), "sources.wukong_paper")
    _require_equal(paper.get("kind"), "paper", "sources.wukong_paper.kind")
    _require_equal(
        paper.get("url"),
        "https://proceedings.mlr.press/v235/zhang24ao.html",
        "wukong paper canonical URL",
    )
    _require_equal(paper.get("arxiv_identifier"), "arXiv:2403.02545", "wukong arXiv ID")
    _require_equal(
        paper.get("local_artifact_origin_repository"),
        "https://github.com/mlresearch/v235",
        "wukong PDF origin repository",
    )
    _require_equal(
        paper.get("local_pdf_origin_url"),
        "https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhang24ao/zhang24ao.pdf",
        "wukong exact PDF origin",
    )
    _require_equal(paper.get("revision"), "PMLR-235-ICML-2024", "wukong paper revision")
    _require_sha256(paper.get("local_pdf_sha256"), "wukong paper local_pdf_sha256")
    _require_equal(
        paper.get("local_pdf_sha256"),
        "8f4f539a174d3ff12653694d7e3d16f2c3b0c0651770a97445ac4826dcb5fabd",
        "wukong paper pinned PDF hash",
    )
    _require_equal(
        paper.get("forbidden_scope"),
        "metric_result_and_evaluation_columns",
        "wukong paper forbidden_scope",
    )
    afn_arch = _as_mapping(
        paper.get("smallest_documented_afn_plus_architecture"),
        "wukong paper smallest_documented_afn_plus_architecture",
    )
    _require_equal(afn_arch.get("logarithmic_neurons"), 128, "AFN documented nlog")
    _require_equal(afn_arch.get("afn_hidden_dims"), [512], "AFN documented tower")
    _require_equal(afn_arch.get("dnn_hidden_dims"), [512], "AFN+ documented DNN")

    wukong_arch = _as_mapping(
        paper.get("smallest_fully_enumerated_complete_architecture"),
        "wukong paper complete architecture",
    )
    expected_wukong = {
        "num_layers": 2,
        "fmb_features": 8,
        "lcb_features": 8,
        "fmb_rank": 24,
        "fmb_hidden_dims": [2048, 2048, 2048],
    }
    for key, expected in expected_wukong.items():
        _require_equal(wukong_arch.get(key), expected, f"wukong architecture.{key}")
    rejected = _as_mapping(
        paper.get("rejected_derived_cartesian_minimum"),
        "wukong paper rejected derived minimum",
    )
    _require_equal(
        rejected.get("reason"),
        "not_a_fully_enumerated_complete_public_configuration",
        "wukong rejected derived minimum reason",
    )

    provenance = _as_mapping(
        payload.get("model_provenance"), "model_sources.model_provenance"
    )
    if set(provenance) != REQUIRED_PROVENANCE_MODELS:
        raise SearchSpaceError(
            "model_sources.model_provenance must contain exactly the frozen model set"
        )
    for model_id, record_value in provenance.items():
        record = _as_mapping(record_value, f"model_provenance.{model_id}")
        source_ids = _as_list(
            record.get("source_ids"), f"model_provenance.{model_id}.source_ids"
        )
        if not source_ids or any(source_id not in sources for source_id in source_ids):
            raise SearchSpaceError(
                f"model_provenance.{model_id}.source_ids must reference sources"
            )
        _nonempty_string(
            record.get("implementation_basis"),
            f"model_provenance.{model_id}.implementation_basis",
        )

    afn_adaptation = _as_mapping(
        provenance["afn_plus"].get("protocol_adaptation"),
        "model_provenance.afn_plus.protocol_adaptation",
    )
    _require_bool(
        afn_adaptation.get("common_task_loss_only"),
        True,
        "AFN adaptation common_task_loss_only",
    )
    _require_bool(
        afn_adaptation.get("auxiliary_branch_losses"),
        False,
        "AFN adaptation auxiliary_branch_losses",
    )
    _require_bool(
        afn_adaptation.get("stop_gradient_on_branch_logits"),
        False,
        "AFN adaptation stop_gradient",
    )
    _require_bool(
        afn_adaptation.get("exact_official_training_reproduction"),
        False,
        "AFN adaptation exact_official_training_reproduction",
    )


def _model_architecture(model: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    base = _as_mapping(model.get("base_config"), f"models.{model_id}.base_config")
    return _as_mapping(
        base.get("architecture"), f"models.{model_id}.base_config.architecture"
    )


def _validate_candidates(model_id: str, model: Mapping[str, Any]) -> None:
    mode = model.get("search_mode")
    candidates = _as_list(model.get("candidates"), f"models.{model_id}.candidates")
    ids: list[str] = []
    for index, candidate_value in enumerate(candidates):
        candidate = _as_mapping(
            candidate_value, f"models.{model_id}.candidates[{index}]"
        )
        ids.append(
            _nonempty_string(
                candidate.get("id"), f"models.{model_id}.candidates[{index}].id"
            )
        )
    if len(set(ids)) != len(ids):
        raise SearchSpaceError(f"models.{model_id} candidate IDs must be unique")

    if mode == "tuned":
        if len(candidates) != 2:
            raise SearchSpaceError(
                f"tuned model {model_id} must have exactly two candidates"
            )
        vary_only = _nonempty_string(
            model.get("vary_only"), f"models.{model_id}.vary_only"
        )
        _require_equal(
            vary_only,
            "optimizer.learning_rate",
            f"models.{model_id}.vary_only",
        )
        base = _as_mapping(model.get("base_config"), f"models.{model_id}.base_config")
        optimizer = _as_mapping(base.get("optimizer"), f"models.{model_id}.optimizer")
        _require_equal(optimizer.get("learning_rate"), 0.0003, f"{model_id} base LR")
        values: list[Any] = []
        for index, candidate_value in enumerate(candidates):
            candidate = _as_mapping(
                candidate_value, f"models.{model_id}.candidates[{index}]"
            )
            override = _as_mapping(
                candidate.get("override"),
                f"models.{model_id}.candidates[{index}].override",
            )
            leaves = _flatten_leaves(override)
            if set(leaves) != {vary_only}:
                raise SearchSpaceError(
                    f"models.{model_id} candidates may vary only {vary_only}"
                )
            values.append(leaves[vary_only])
        if values[0] == values[1]:
            raise SearchSpaceError(
                f"models.{model_id} candidate values must be distinct"
            )
        _require_equal(
            values, [0.0003, 0.001], f"models.{model_id} frozen learning rates"
        )
    elif mode == "fixed_no_tune":
        if len(candidates) != 1:
            raise SearchSpaceError(
                f"fixed_no_tune model {model_id} must have exactly one candidate"
            )
        _nonempty_string(model.get("fixed_source"), f"models.{model_id}.fixed_source")
        _nonempty_string(model.get("fixed_reason"), f"models.{model_id}.fixed_reason")
        candidate = _as_mapping(candidates[0], f"models.{model_id}.candidates[0]")
        override = _as_mapping(
            candidate.get("override"), f"models.{model_id}.candidates[0].override"
        )
        if override:
            raise SearchSpaceError(
                f"fixed_no_tune model {model_id} override must be empty"
            )
        base = _as_mapping(model.get("base_config"), f"models.{model_id}.base_config")
        optimizer = _as_mapping(base.get("optimizer"), f"models.{model_id}.optimizer")
        _require_equal(optimizer.get("learning_rate"), 0.0003, f"{model_id} fixed LR")
    elif mode == "mechanical_capacity_match":
        if len(candidates) != 2 or ids != [
            "capacity_rank_lower",
            "capacity_rank_upper",
        ]:
            raise SearchSpaceError(
                "capacity control must have the two ordered frozen rank brackets"
            )
        expected_rules = (
            "greatest_positive_integer_with_parameter_count_not_above_target",
            "lower_rank_plus_one",
        )
        for candidate_value, expected_rule in zip(
            candidates, expected_rules, strict=True
        ):
            candidate = _as_mapping(candidate_value, "capacity candidate")
            selector = _as_mapping(
                candidate.get("architecture_selector"), "capacity selector"
            )
            _require_equal(selector.get("field"), "low_rank", "capacity rank field")
            _require_equal(selector.get("rule"), expected_rule, "capacity rank rule")
    else:
        raise SearchSpaceError(f"models.{model_id}.search_mode is unsupported")


def _validate_common_contract(payload: Mapping[str, Any]) -> None:
    firewall = _as_mapping(payload.get("selection_firewall"), "selection_firewall")
    _require_bool(
        firewall.get("published_results_quarantined"),
        True,
        "selection_firewall.published_results_quarantined",
    )
    _require_bool(
        firewall.get("published_results_may_select_candidates"),
        False,
        "selection_firewall.published_results_may_select_candidates",
    )
    _require_bool(
        firewall.get("test_split_may_select_candidates"),
        False,
        "selection_firewall.test_split_may_select_candidates",
    )
    _require_bool(
        firewall.get("test_metrics_before_selected_configs_lock"),
        False,
        "selection_firewall.test_metrics_before_selected_configs_lock",
    )
    _require_equal(firewall.get("tuning_seed"), 42, "selection_firewall.tuning_seed")
    _require_equal(
        firewall.get("tuning_splits"),
        ["train", "validation"],
        "selection_firewall.tuning_splits",
    )
    _require_equal(
        firewall.get("final_seeds"), [42, 43, 44], "selection_firewall.final_seeds"
    )

    budget = _as_mapping(payload.get("trial_budget"), "trial_budget")
    _require_equal(
        budget.get("tuned_candidates_per_tuning_unit"),
        2,
        "trial_budget.tuned_candidates_per_tuning_unit",
    )
    _require_equal(
        budget.get("fixed_no_tune_candidates"),
        1,
        "trial_budget.fixed_no_tune_candidates",
    )
    _require_bool(
        budget.get("unused_trials_may_transfer"),
        False,
        "trial_budget.unused_trials_may_transfer",
    )
    _require_bool(
        budget.get("adaptive_trial_transfer"),
        False,
        "trial_budget.adaptive_trial_transfer",
    )
    _require_bool(
        budget.get("adaptive_candidate_generation"),
        False,
        "trial_budget.adaptive_candidate_generation",
    )

    common = _as_mapping(payload.get("common_contract"), "common_contract")
    optimizer = _as_mapping(common.get("optimizer"), "common_contract.optimizer")
    _require_equal(optimizer.get("family"), "AdamW", "optimizer.family")
    _require_equal(optimizer.get("betas"), [0.9, 0.999], "optimizer.betas")
    _require_equal(optimizer.get("epsilon"), 1.0e-8, "optimizer.epsilon")
    _positive_number(optimizer.get("weight_decay"), "optimizer.weight_decay")

    precision = _as_mapping(common.get("precision"), "common_contract.precision")
    _require_equal(
        precision.get("parameter_dtype"), "float32", "precision.parameter_dtype"
    )
    _require_bool(precision.get("autocast_enabled"), True, "precision.autocast_enabled")
    _require_equal(
        precision.get("autocast_dtype"), "bfloat16", "precision.autocast_dtype"
    )
    _require_equal(
        precision.get("optimizer_state_dtype"),
        "float32",
        "precision.optimizer_state_dtype",
    )
    _require_bool(precision.get("fp16_forbidden"), True, "precision.fp16_forbidden")
    sensitive = _as_list(
        precision.get("fp32_sensitive_operations"),
        "precision.fp32_sensitive_operations",
    )
    for operation in ("dcdn_reciprocal", "afn_abs_clamp_log_and_exp", "loss"):
        if operation not in sensitive:
            raise SearchSpaceError(f"precision must keep {operation} in FP32")

    batching = _as_mapping(common.get("batching"), "common_contract.batching")
    _positive_int(
        batching.get("effective_batch_per_independent_run"),
        "batching.effective_batch_per_independent_run",
    )
    _require_bool(batching.get("ddp"), False, "batching.ddp")
    _require_bool(
        batching.get("gradient_accumulation_preserves_effective_batch"),
        True,
        "batching.gradient_accumulation_preserves_effective_batch",
    )

    stopping = _as_mapping(common.get("stopping"), "common_contract.stopping")
    _require_equal(
        stopping.get("budget_rationale"),
        "project_pre_metric_compute_envelope_choice_not_upstream_default_or_metric_derived",
        "stopping.budget_rationale",
    )
    _require_equal(
        stopping.get("cadence_rationale"),
        "project_pre_metric_examples_based_monitoring_choice_not_upstream_default_or_metric_derived",
        "stopping.cadence_rationale",
    )
    _positive_int(
        stopping.get("patience_validation_checks"),
        "stopping.patience_validation_checks",
    )
    datasets = _as_mapping(stopping.get("datasets"), "stopping.datasets")
    expected_metrics = {
        "kuairand_pure": ("binary_cross_entropy_with_logits", "log_loss"),
        "vk_lsvd_up0.01_ir0.01": (
            "normalized_mean_squared_error",
            "normalized_mse",
        ),
        "kuairec_big_matrix": ("normalized_mean_squared_error", "normalized_mse"),
    }
    if set(datasets) != set(expected_metrics):
        raise SearchSpaceError("stopping.datasets must equal the frozen dataset set")
    for dataset_id, (loss, metric) in expected_metrics.items():
        record = _as_mapping(datasets[dataset_id], f"stopping.datasets.{dataset_id}")
        _positive_int(
            record.get("maximum_train_examples"),
            f"{dataset_id}.maximum_train_examples",
        )
        _positive_int(
            record.get("validate_every_examples"),
            f"{dataset_id}.validate_every_examples",
        )
        _require_equal(record.get("loss"), loss, f"{dataset_id}.loss")
        _require_equal(
            record.get("selection_metric"), metric, f"{dataset_id}.selection_metric"
        )
        _require_equal(
            record.get("selection_direction"),
            "minimize",
            f"{dataset_id}.selection_direction",
        )

    clipping = _as_mapping(
        common.get("gradient_clipping"), "common_contract.gradient_clipping"
    )
    _require_equal(clipping.get("kind"), "global_l2_norm", "gradient_clipping.kind")
    _require_equal(clipping.get("max_norm"), 1.0, "gradient_clipping.max_norm")
    _require_bool(
        clipping.get("apply_after_unscale"),
        True,
        "gradient_clipping.apply_after_unscale",
    )

    initialization = _as_mapping(
        common.get("initialization"), "common_contract.initialization"
    )
    expected_initializers = {
        "component_seeds": "hash_derived_interaction_adapter_head_from_run_seed",
        "embeddings": "normal_std_0.01",
        "relu_hidden_linear": "kaiming_uniform_nonlinearity_relu",
        "representation_task_output_attention_qkv_residual_and_gates": (
            "xavier_uniform"
        ),
        "all_biases": "zeros",
        "cross_product_linear": "kaiming_normal_scale_0.1",
        "dcdn_division_linear": "kaiming_normal_scale_0.1",
        "afn_coefficients_and_branch_outputs": "xavier_uniform",
        "wukong_fmb_pre_relu_output": "kaiming_uniform_nonlinearity_relu",
    }
    for field, expected in expected_initializers.items():
        _require_equal(initialization.get(field), expected, f"initialization.{field}")
    for field in (
        "deterministic_from_run_seed",
        "head_weights_identical_across_variants",
        "adapter_seed_and_distribution_identical_shape_dependent",
    ):
        _require_bool(initialization.get(field), True, f"initialization.{field}")

    _require_bool(
        initialization.get("same_product_initialization_for_dcn_dcnv2_dcdn"),
        True,
        "initialization.same_product_initialization_for_dcn_dcnv2_dcdn",
    )

    encoder = _as_mapping(
        common.get("shared_encoder"), "common_contract.shared_encoder"
    )
    _require_equal(encoder.get("representation_dim"), 64, "encoder.representation_dim")
    _require_equal(
        encoder.get("full_schema_sha256"),
        "${DATASET_FULL_SCHEMA_SHA256}",
        "encoder.full_schema_sha256",
    )
    _require_equal(
        encoder.get("operand_schema_sha256"),
        "${DATASET_OPERAND_SCHEMA_SHA256}",
        "encoder.operand_schema_sha256",
    )

    head = _as_mapping(
        common.get("shared_prediction_head"),
        "common_contract.shared_prediction_head",
    )
    _require_equal(head.get("representation_dim"), 64, "head.representation_dim")
    _require_equal(head.get("hidden_dims"), [], "head.hidden_dims")
    _require_equal(head.get("dropout"), 0.0, "head.dropout")
    _require_equal(
        head.get("per_model_adapter"),
        "linear_to_representation_dim",
        "head.per_model_adapter",
    )
    _require_bool(
        head.get("adapter_parameters_and_flops_counted"),
        True,
        "head.adapter_parameters_and_flops_counted",
    )
    _require_bool(
        head.get("source_model_auxiliary_losses_forbidden"),
        True,
        "head.source_model_auxiliary_losses_forbidden",
    )


def _validate_model_semantics(models: Mapping[str, Any]) -> None:
    dnn = _model_architecture(models["dnn"], "dnn")
    _require_equal(dnn.get("input_width"), "${DATASET_FLAT_WIDTH}", "DNN input")
    _require_equal(
        dnn.get("input_schema_sha256"),
        "${DATASET_FULL_SCHEMA_SHA256}",
        "DNN schema",
    )
    _require_equal(
        dnn.get("feature_scope"),
        "flat_C_plus_numeric_regime",
        "DNN feature scope",
    )
    if "input_field_count" in dnn:
        raise SearchSpaceError("DNN must consume the encoded flat C+P width")

    dcn = _model_architecture(models["dcn"], "dcn")
    _require_equal(dcn.get("input_width"), "${DATASET_FLAT_WIDTH}", "dcn input")
    _require_equal(
        dcn.get("input_schema_sha256"),
        "${DATASET_FULL_SCHEMA_SHA256}",
        "dcn schema",
    )
    _require_equal(
        dcn.get("cross_parameterization"), "original_rank_one", "dcn parameterization"
    )

    autoint = _model_architecture(models["autoint_plus"], "autoint_plus")
    _require_bool(
        autoint.get("dnn_branch_enabled"), True, "AutoInt+ dnn_branch_enabled"
    )
    dnn_dims = autoint.get("dnn_hidden_dims")
    if not isinstance(dnn_dims, list) or not dnn_dims:
        raise SearchSpaceError("AutoInt+ must contain a non-empty DNN branch")
    _require_equal(
        autoint.get("branch_merge"),
        "concatenate_attention_and_dnn_representations_before_visible_adapter",
        "AutoInt+ branch_merge",
    )

    dcnv2 = _model_architecture(models["dcnv2_full_rank"], "dcnv2_full_rank")
    _require_equal(dcnv2.get("input_width"), "${DATASET_FLAT_WIDTH}", "DCNv2 input")
    _require_equal(
        dcnv2.get("cross_parameterization"), "full_rank", "DCNv2 parameterization"
    )
    _require_bool(
        dcnv2.get("use_low_rank_mixture"),
        False,
        "DCNv2 full-rank use_low_rank_mixture",
    )

    afn = _model_architecture(models["afn_plus"], "afn_plus")
    afn_model = _as_mapping(models["afn_plus"], "models.afn_plus")
    _require_equal(
        afn_model.get("implementation_source"),
        "model_sources_v1.model_provenance.afn_plus",
        "AFN+ implementation source",
    )
    _require_equal(
        afn_model.get("architecture_source_role"),
        "hyperparameter_geometry_only_not_implementation",
        "AFN+ architecture source role",
    )

    _require_bool(afn.get("ensemble"), True, "AFN+ ensemble")
    _require_bool(
        afn.get("separate_embedding_views"), True, "AFN+ separate_embedding_views"
    )
    _require_equal(afn.get("logarithmic_neurons"), 128, "AFN+ logarithmic_neurons")
    _require_equal(afn.get("afn_hidden_dims"), [512], "AFN+ AFN tower")
    _require_equal(afn.get("dnn_hidden_dims"), [512], "AFN+ DNN tower")
    _require_bool(afn.get("common_task_loss_only"), True, "AFN+ common_task_loss_only")
    _require_bool(afn.get("auxiliary_branch_losses"), False, "AFN+ auxiliary losses")
    _require_bool(
        afn.get("stop_gradient_on_branch_logits"), False, "AFN+ stop gradient"
    )
    _require_equal(
        afn.get("branch_merge"),
        "concatenate_branch_logits_then_common_adapter_head",
        "AFN+ branch_merge",
    )
    treatment = _as_mapping(
        afn.get("positive_log_treatment"), "AFN+ positive_log_treatment"
    )
    _require_equal(treatment.get("order"), ["abs", "clamp_min", "log"], "AFN order")
    _require_equal(treatment.get("clamp_min"), 0.0001, "AFN clamp_min")
    _require_equal(treatment.get("compute_dtype"), "float32", "AFN log dtype")
    _require_equal(treatment.get("exp_compute_dtype"), "float32", "AFN exp dtype")
    if "exp_clip" not in treatment or treatment["exp_clip"] is not None:
        raise SearchSpaceError("AFN+ exp_clip must be explicit null")
    _require_equal(
        treatment.get("exp_equation"),
        "unbounded_source_equation",
        "AFN exp equation",
    )
    _require_equal(
        treatment.get("nonfinite_action"), "stop_run", "AFN nonfinite action"
    )

    wukong = _model_architecture(models["wukong_small"], "wukong_small")
    expected_wukong = {
        "num_layers": 2,
        "lcb_features": 8,
        "fmb_features": 8,
        "fmb_rank": 24,
        "fmb_hidden_dims": [2048, 2048, 2048],
    }
    for field, expected in expected_wukong.items():
        _require_equal(wukong.get(field), expected, f"Wukong-small {field}")
    _require_bool(
        wukong.get("source_projection_head_replaced_by_common_head"),
        True,
        "Wukong-small common head adaptation",
    )

    dcdn = _model_architecture(models["dcdn_core"], "dcdn_core")
    _require_equal(dcdn.get("label"), "DCDN-Core-L2", "DCDN label")
    _require_equal(
        dcdn.get("features_count"),
        "${DATASET_CORE_OPERAND_WIDTH}",
        "DCDN operand width",
    )
    _require_equal(
        dcdn.get("operand_schema_sha256"),
        "${DATASET_OPERAND_SCHEMA_SHA256}",
        "DCDN operand schema",
    )
    _require_equal(
        dcdn.get("context_bypass"),
        "unchanged_concat_after_core_before_visible_adapter",
        "DCDN context bypass",
    )
    _require_bool(dcdn.get("fixed_anchor"), True, "DCDN fixed_anchor")
    for field in (
        "local_fc_path",
        "local_batch_norm",
        "local_dropout",
        "concatenation_projector",
    ):
        _require_bool(dcdn.get(field), False, f"DCDN {field}")
    reciprocal = _as_mapping(dcdn.get("reciprocal"), "DCDN reciprocal")
    _require_equal(
        reciprocal.get("mask_width"),
        "${DATASET_DENOMINATOR_MASK_WIDTH}",
        "DCDN reciprocal mask width",
    )
    _require_bool(reciprocal.get("mask_required"), True, "DCDN mask_required")
    _require_equal(reciprocal.get("epsilon"), 0.01, "DCDN epsilon")
    _require_equal(reciprocal.get("compute_dtype"), "float32", "DCDN reciprocal dtype")
    _require_equal(reciprocal.get("log_treatment"), "none", "DCDN log treatment")
    _require_bool(
        reciprocal.get("full_materialized_scan_required"),
        True,
        "DCDN full materialized scan",
    )

    capacity_model = models["dcnv2_low_rank_mix_capacity"]
    _require_equal(
        capacity_model.get("display_label"),
        "DCNv2-capacity-matched-core",
        "capacity display_label",
    )
    capacity = _model_architecture(capacity_model, "dcnv2_low_rank_mix_capacity")
    _require_equal(
        capacity.get("model_key"),
        "dcnv2_capacity_matched_core",
        "capacity model_key",
    )
    _require_equal(
        capacity.get("input_width"),
        "${DATASET_CORE_OPERAND_WIDTH}",
        "capacity operand width",
    )
    _require_equal(
        capacity.get("operand_schema_sha256"),
        "${DATASET_OPERAND_SCHEMA_SHA256}",
        "capacity operand schema",
    )
    _require_equal(
        capacity.get("context_bypass"),
        "unchanged_concat_after_core_before_visible_adapter",
        "capacity context bypass",
    )
    _require_equal(
        capacity.get("cross_parameterization"),
        "low_rank_moe",
        "capacity parameterization",
    )
    _require_bool(
        capacity.get("use_low_rank_mixture"),
        True,
        "capacity use_low_rank_mixture",
    )
    _positive_int(capacity.get("num_experts"), "capacity num_experts")
    base = _as_mapping(capacity_model.get("base_config"), "capacity base_config")
    capacity_optimizer = _as_mapping(
        base.get("optimizer"), "capacity base_config.optimizer"
    )
    _require_equal(
        capacity_optimizer.get("learning_rate"),
        "${SELECTED_DCNV2_LEARNING_RATE}",
        "capacity inherited learning rate",
    )
    ordinary = _as_mapping(
        capacity_model.get("ordinary_full_rank_reuse_check"),
        "capacity ordinary_full_rank_reuse_check",
    )
    _require_equal(ordinary.get("tolerance_fraction"), 0.05, "capacity tolerance")
    _require_bool(
        ordinary.get("reuse_exact_run_if_parameters_and_flops_both_within_tolerance"),
        True,
        "capacity ordinary reuse",
    )
    counter = _as_mapping(capacity_model.get("capacity_counter"), "capacity counter")
    for field in (
        "include_common_prediction_head",
        "include_visible_representation_adapter",
        "include_unchanged_context_bypass",
    ):
        _require_bool(counter.get(field), True, f"capacity counter {field}")
    rule = _as_mapping(capacity_model.get("selection_rule"), "capacity selection_rule")
    _require_bool(rule.get("metric_values_forbidden"), True, "capacity metric firewall")
    _require_equal(
        rule.get("distance"),
        "abs_log_parameter_ratio_plus_abs_log_flop_ratio",
        "capacity distance",
    )
    _require_equal(rule.get("tie_break"), "lower_rank", "capacity tie break")


def validate_search_space(
    payload: Mapping[str, Any],
    *,
    model_sources: Mapping[str, Any] | None = None,
) -> None:
    """Validate the immutable pre-metric search-space definition."""

    _require_equal(payload.get("schema_version"), 1, "search_space.schema_version")
    _require_equal(
        payload.get("status"), "frozen_before_metrics", "search_space.status"
    )
    frozen_utc = _nonempty_string(payload.get("frozen_utc"), "search_space.frozen_utc")
    if FROZEN_UTC_RE.fullmatch(frozen_utc) is None:
        raise SearchSpaceError(
            "search_space.frozen_utc must be a quoted UTC second ending in Z"
        )

    source_ref = _as_mapping(payload.get("model_sources"), "model_sources reference")
    _require_equal(
        source_ref.get("path"),
        "configs/model_sources_v1.yaml",
        "model_sources.path",
    )
    _require_sha256(source_ref.get("sha256"), "model_sources.sha256")
    matrix_ref = _as_mapping(
        payload.get("experiment_matrix"), "experiment_matrix reference"
    )
    _require_equal(
        matrix_ref.get("path"),
        "configs/experiment_matrix_v1.yaml",
        "experiment_matrix.path",
    )
    _require_sha256(matrix_ref.get("sha256"), "experiment_matrix.sha256")

    _validate_common_contract(payload)

    aliases = _as_mapping(payload.get("matrix_model_aliases"), "matrix_model_aliases")
    expected_aliases = {
        "dnn": "dnn",
        "deepfm": "deepfm",
        "dcn": "dcn",
        "autoint_plus": "autoint_plus",
        "dcnv2": "dcnv2_full_rank",
        "afn_plus": "afn_plus",
        "wukong_small": "wukong_small",
        "dcdn_core": "dcdn_core",
        "dcnv2_capacity_matched_core": "dcnv2_low_rank_mix_capacity",
    }
    _require_equal(dict(aliases), expected_aliases, "matrix_model_aliases")

    dimensions = _as_mapping(payload.get("symbolic_dimensions"), "symbolic_dimensions")
    hashes = _as_mapping(payload.get("symbolic_hashes"), "symbolic_hashes")
    hyperparameters = _as_mapping(
        payload.get("symbolic_hyperparameters"), "symbolic_hyperparameters"
    )
    declared_tokens = set(dimensions) | set(hashes) | set(hyperparameters)
    referenced_tokens = _find_tokens(payload)
    if referenced_tokens != declared_tokens:
        raise SearchSpaceError(
            "every symbolic token must be declared and every declaration referenced"
        )
    if "UNRESOLVED" in repr(payload):
        raise SearchSpaceError("UNRESOLVED markers are forbidden in search_space_v1")

    models = _as_mapping(payload.get("models"), "models")
    if set(models) != REQUIRED_MODELS:
        raise SearchSpaceError("models must contain exactly the frozen model set")
    for model_id, model_value in models.items():
        model = _as_mapping(model_value, f"models.{model_id}")
        _nonempty_string(model.get("source_model"), f"models.{model_id}.source_model")
        _require_equal(
            model.get("source_model"),
            EXPECTED_SOURCE_MODELS[model_id],
            f"models.{model_id}.source_model",
        )
        _nonempty_string(
            model.get("architecture_source"),
            f"models.{model_id}.architecture_source",
        )
        _nonempty_string(
            model.get("architecture_rationale"),
            f"models.{model_id}.architecture_rationale",
        )
        _model_architecture(model, model_id)
        _validate_candidates(model_id, model)

    _validate_model_semantics(models)

    if model_sources is not None:
        validate_model_sources(model_sources)
        actual_provenance = _as_mapping(
            model_sources.get("model_provenance"), "model_sources.model_provenance"
        )
        for model_id, model_value in models.items():
            model = _as_mapping(model_value, f"models.{model_id}")
            source_model = str(model["source_model"])
            if source_model not in actual_provenance:
                raise SearchSpaceError(
                    f"models.{model_id}.source_model lacks pinned provenance"
                )


def _validate_resolution_values(
    payload: Mapping[str, Any],
    resolutions: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    if phase not in PHASE_ORDER:
        raise SearchSpaceError(f"unknown search-space phase: {phase}")

    resolved: dict[str, Any] = {}
    category_specs = (
        ("dimensions", "symbolic_dimensions"),
        ("hashes", "symbolic_hashes"),
        ("hyperparameters", "symbolic_hyperparameters"),
    )
    allowed_categories = {name for name, _ in category_specs} | {"capacity_resolution"}
    unknown_categories = set(resolutions) - allowed_categories
    if unknown_categories:
        raise SearchSpaceError(
            f"unknown resolution categories: {sorted(unknown_categories)}"
        )

    for resolution_key, declaration_key in category_specs:
        declarations = _as_mapping(payload.get(declaration_key), declaration_key)
        values = _as_mapping(resolutions.get(resolution_key, {}), resolution_key)
        unknown = set(values) - set(declarations)
        if unknown:
            raise SearchSpaceError(
                f"unknown {resolution_key} resolutions: {sorted(unknown)}"
            )
        for token, declaration_value in declarations.items():
            declaration = _as_mapping(declaration_value, f"{declaration_key}.{token}")
            required_phase = _nonempty_string(
                declaration.get("required_before"),
                f"{declaration_key}.{token}.required_before",
            )
            if required_phase not in PHASE_ORDER:
                raise SearchSpaceError(
                    f"{declaration_key}.{token}.required_before is invalid"
                )
            required_now = PHASE_ORDER[phase] >= PHASE_ORDER[required_phase]
            if token not in values:
                if required_now:
                    raise SearchSpaceError(
                        f"{token} must be resolved before phase {phase}"
                    )
                continue
            value = values[token]
            declared_type = declaration.get("type")
            if declared_type == "positive_integer":
                _positive_int(value, token)
            elif declared_type == "non_negative_integer":
                _non_negative_int(value, token)
            elif declared_type == "lowercase_sha256":
                _require_sha256(value, token)
            elif declared_type == "positive_float":
                _positive_number(value, token)
            else:
                raise SearchSpaceError(f"unsupported symbolic type for {token}")
            allowed_values = declaration.get("allowed_values")
            if allowed_values is not None and value not in allowed_values:
                raise SearchSpaceError(f"{token} is outside its frozen allowed_values")
            resolved[token] = value

    dimension_values = _as_mapping(resolutions.get("dimensions", {}), "dimensions")
    required_dimension_names = {
        "DATASET_FLAT_WIDTH",
        "DATASET_CORE_OPERAND_WIDTH",
        "DATASET_CONTEXT_BYPASS_WIDTH",
        "DATASET_DENOMINATOR_MASK_WIDTH",
    }
    if required_dimension_names <= set(dimension_values):
        if (
            dimension_values["DATASET_DENOMINATOR_MASK_WIDTH"]
            != dimension_values["DATASET_CORE_OPERAND_WIDTH"]
        ):
            raise SearchSpaceError("denominator mask width must equal operand width")
        if dimension_values["DATASET_FLAT_WIDTH"] != (
            dimension_values["DATASET_CORE_OPERAND_WIDTH"]
            + dimension_values["DATASET_CONTEXT_BYPASS_WIDTH"]
        ):
            raise SearchSpaceError("flat width must equal operand plus context bypass")

    return resolved


def _counts(record_value: Any, field: str) -> tuple[float, float]:
    record = _as_mapping(record_value, field)
    parameters = _positive_number(record.get("parameters"), f"{field}.parameters")
    flops = _positive_number(record.get("flops"), f"{field}.flops")
    return parameters, flops


def _within_tolerance(
    candidate: tuple[float, float],
    target: tuple[float, float],
    tolerance: float,
) -> bool:
    return all(
        (1.0 - tolerance) <= candidate_value / target_value <= (1.0 + tolerance)
        for candidate_value, target_value in zip(candidate, target, strict=True)
    )


def _capacity_distance(
    candidate: tuple[float, float], target: tuple[float, float]
) -> float:
    return sum(
        abs(math.log(candidate_value / target_value))
        for candidate_value, target_value in zip(candidate, target, strict=True)
    )


def validate_capacity_resolution(
    search_payload: Mapping[str, Any],
    resolution_value: Any,
) -> None:
    """Validate the metric-free capacity-control decision and both brackets."""

    resolution = _as_mapping(resolution_value, "capacity_resolution")
    _require_bool(
        resolution.get("metrics_consulted"),
        False,
        "capacity_resolution.metrics_consulted",
    )
    _require_bool(
        resolution.get("resolved_before_first_model_metric"),
        True,
        "capacity_resolution.resolved_before_first_model_metric",
    )

    forbidden_keys = {
        "auc",
        "log_loss",
        "validation_metric",
        "test_metric",
        "quality_metric",
        "loss",
    }
    flattened_keys = {key.rsplit(".", 1)[-1] for key in _flatten_leaves(resolution)}
    if forbidden_keys & flattened_keys:
        raise SearchSpaceError("capacity resolution contains forbidden metric fields")

    capacity_model = _as_mapping(
        _as_mapping(search_payload.get("models"), "models").get(
            "dcnv2_low_rank_mix_capacity"
        ),
        "capacity model",
    )
    ordinary_rule = _as_mapping(
        capacity_model.get("ordinary_full_rank_reuse_check"),
        "ordinary_full_rank_reuse_check",
    )
    tolerance = _positive_number(
        ordinary_rule.get("tolerance_fraction"), "capacity tolerance"
    )
    target = _counts(resolution.get("target"), "capacity_resolution.target")
    ordinary = _counts(resolution.get("ordinary"), "capacity_resolution.ordinary")
    decision = _as_mapping(resolution.get("decision"), "capacity_resolution.decision")
    mode = decision.get("mode")

    if _within_tolerance(ordinary, target, tolerance):
        _require_equal(mode, "reuse_ordinary", "capacity decision.mode")
        _require_equal(
            decision.get("selected_candidate"),
            "dcnv2_full_rank",
            "capacity selected ordinary",
        )
        return

    candidates = _as_mapping(
        resolution.get("candidates"), "capacity_resolution.candidates"
    )
    if set(candidates) != {"capacity_rank_lower", "capacity_rank_upper"}:
        raise SearchSpaceError("capacity resolution must contain both rank brackets")
    lower_record = _as_mapping(candidates["capacity_rank_lower"], "capacity lower")
    upper_record = _as_mapping(candidates["capacity_rank_upper"], "capacity upper")
    lower_rank = _positive_int(lower_record.get("low_rank"), "capacity lower rank")
    upper_rank = _positive_int(upper_record.get("low_rank"), "capacity upper rank")
    if upper_rank != lower_rank + 1:
        raise SearchSpaceError("capacity ranks must be adjacent lower/upper brackets")
    lower = _counts(lower_record, "capacity lower")
    upper = _counts(upper_record, "capacity upper")
    if lower[0] > target[0] or upper[0] < target[0]:
        raise SearchSpaceError("capacity parameter counts do not bracket target")

    eligible: list[tuple[str, tuple[float, float]]] = []
    for candidate_id, candidate_counts in (
        ("capacity_rank_lower", lower),
        ("capacity_rank_upper", upper),
    ):
        if _within_tolerance(candidate_counts, target, tolerance):
            eligible.append((candidate_id, candidate_counts))

    if not eligible:
        _require_equal(mode, "unavailable", "capacity unavailable decision")
        if decision.get("selected_candidate") is not None:
            raise SearchSpaceError(
                "unavailable capacity decision cannot select a candidate"
            )
        return

    expected = min(
        eligible,
        key=lambda item: (
            _capacity_distance(item[1], target),
            0 if item[0] == "capacity_rank_lower" else 1,
        ),
    )[0]
    _require_equal(mode, "fallback_bracket", "capacity fallback decision")
    _require_equal(
        decision.get("selected_candidate"),
        expected,
        "capacity mechanically selected bracket",
    )


def resolve_search_space(
    payload: Mapping[str, Any],
    *,
    resolutions: Mapping[str, Any] | None = None,
    phase: str = "definition",
    model_sources: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Resolve declared symbols and enforce phase-specific preconditions."""

    validate_search_space(payload, model_sources=model_sources)
    context = resolutions or {}
    resolved_values = _validate_resolution_values(payload, context, phase)
    if PHASE_ORDER[phase] >= PHASE_ORDER["first_model_metric"]:
        if "capacity_resolution" not in context:
            raise SearchSpaceError(
                "capacity resolution is required before the first model metric"
            )
        validate_capacity_resolution(payload, context["capacity_resolution"])

    resolved_payload = _replace_tokens(copy.deepcopy(payload), resolved_values)
    remaining = _find_tokens(resolved_payload)
    if PHASE_ORDER[phase] >= PHASE_ORDER["protocol_frozen"] and remaining:
        raise SearchSpaceError(
            f"all symbols must resolve before protocol freeze: {sorted(remaining)}"
        )
    return resolved_payload


@dataclass(frozen=True)
class SearchSpaceLock:
    """Verified search/provenance/matrix artifacts with phase gates."""

    path: Path
    digest: str
    payload: Mapping[str, Any]
    model_sources_path: Path
    model_sources_digest: str
    model_sources_payload: Mapping[str, Any]
    experiment_matrix_path: Path
    experiment_matrix_digest: str

    @classmethod
    def load(cls, path: str | Path) -> SearchSpaceLock:
        source = Path(path)
        payload = load_verified_yaml(source)
        repository_root = source.parent.parent

        source_ref = _as_mapping(payload.get("model_sources"), "model_sources")
        source_path = _nonempty_string(source_ref.get("path"), "model_sources.path")
        _require_equal(
            source_path,
            "configs/model_sources_v1.yaml",
            "model_sources.path",
        )
        model_sources_path = repository_root / source_path
        model_sources_payload = load_verified_yaml(model_sources_path)
        model_sources_digest = sha256_file(model_sources_path)
        if model_sources_digest != source_ref.get("sha256"):
            raise SearchSpaceError(
                "model_sources digest does not match search-space reference"
            )

        matrix_ref = _as_mapping(payload.get("experiment_matrix"), "experiment_matrix")
        matrix_path = _nonempty_string(matrix_ref.get("path"), "experiment_matrix.path")
        _require_equal(
            matrix_path,
            "configs/experiment_matrix_v1.yaml",
            "experiment_matrix.path",
        )
        experiment_matrix_path = repository_root / matrix_path
        load_verified_yaml(experiment_matrix_path)
        experiment_matrix_digest = sha256_file(experiment_matrix_path)
        if experiment_matrix_digest != matrix_ref.get("sha256"):
            raise SearchSpaceError(
                "experiment_matrix digest does not match search-space reference"
            )

        validate_search_space(payload, model_sources=model_sources_payload)
        return cls(
            path=source,
            digest=sha256_file(source),
            payload=payload,
            model_sources_path=model_sources_path,
            model_sources_digest=model_sources_digest,
            model_sources_payload=model_sources_payload,
            experiment_matrix_path=experiment_matrix_path,
            experiment_matrix_digest=experiment_matrix_digest,
        )

    def require_phase(
        self,
        phase: str,
        *,
        resolutions: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return resolve_search_space(
            self.payload,
            resolutions=resolutions,
            phase=phase,
            model_sources=self.model_sources_payload,
        )
