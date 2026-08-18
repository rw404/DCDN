from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from dcdn.search_space import (
    SearchSpaceError,
    SearchSpaceLock,
    resolve_search_space,
    validate_capacity_resolution,
    validate_model_sources,
    validate_search_space,
)

ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATH = ROOT / "configs" / "search_space_v1.yaml"
SOURCES_PATH = ROOT / "configs" / "model_sources_v1.yaml"
TUNED_MODELS = (
    "dnn",
    "deepfm",
    "dcn",
    "autoint_plus",
    "dcnv2_full_rank",
    "afn_plus",
    "dcdn_core",
)


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def search_payload() -> dict[str, Any]:
    return load_yaml(SEARCH_PATH)


def sources_payload() -> dict[str, Any]:
    return load_yaml(SOURCES_PATH)


def set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node: dict[str, Any] = payload
    for key in path[:-1]:
        child = node[key]
        assert isinstance(child, dict)
        node = child
    node[path[-1]] = value


def capacity_reuse_resolution(*, include_learning_rate: bool) -> dict[str, Any]:
    resolution: dict[str, Any] = {
        "dimensions": {
            "DATASET_MODEL_FIELD_COUNT": 7,
            "DATASET_FLAT_WIDTH": 20,
            "DATASET_CORE_OPERAND_WIDTH": 12,
            "DATASET_CONTEXT_BYPASS_WIDTH": 8,
            "DATASET_DENOMINATOR_MASK_WIDTH": 12,
        },
        "hashes": {
            "DATASET_FULL_SCHEMA_SHA256": "a" * 64,
            "DATASET_OPERAND_SCHEMA_SHA256": "b" * 64,
        },
        "capacity_resolution": {
            "metrics_consulted": False,
            "resolved_before_first_model_metric": True,
            "target": {"parameters": 1000, "flops": 1000},
            "ordinary": {"parameters": 1000, "flops": 1000},
            "decision": {
                "mode": "reuse_ordinary",
                "selected_candidate": "dcnv2_full_rank",
            },
        },
    }
    if include_learning_rate:
        resolution["hyperparameters"] = {"SELECTED_DCNV2_LEARNING_RATE": 0.0003}
    return resolution


def fallback_capacity_resolution() -> dict[str, Any]:
    return {
        "metrics_consulted": False,
        "resolved_before_first_model_metric": True,
        "target": {"parameters": 1000, "flops": 1000},
        "ordinary": {"parameters": 500, "flops": 500},
        "candidates": {
            "capacity_rank_lower": {
                "low_rank": 3,
                "parameters": 960,
                "flops": 960,
            },
            "capacity_rank_upper": {
                "low_rank": 4,
                "parameters": 1040,
                "flops": 1020,
            },
        },
        "decision": {
            "mode": "fallback_bracket",
            "selected_candidate": "capacity_rank_upper",
        },
    }


def write_hashed(path: Path, payload: dict[str, Any]) -> None:
    text = yaml.safe_dump(payload, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )


def test_frozen_manifests_validate_without_data_or_metrics() -> None:
    sources = sources_payload()
    search = search_payload()
    validate_model_sources(sources)
    validate_search_space(search, model_sources=sources)


@pytest.mark.parametrize("model_id", TUNED_MODELS)
def test_every_tuned_model_has_exactly_two_candidates(model_id: str) -> None:
    search = search_payload()
    search["models"][model_id]["candidates"].pop()
    with pytest.raises(SearchSpaceError, match="exactly two"):
        validate_search_space(search)


def test_fixed_and_capacity_candidate_counts_fail_closed() -> None:
    fixed = search_payload()
    candidate = deepcopy(fixed["models"]["wukong_small"]["candidates"][0])
    candidate["id"] = "invented_second_fixed_slot"
    fixed["models"]["wukong_small"]["candidates"].append(candidate)
    with pytest.raises(SearchSpaceError, match="exactly one"):
        validate_search_space(fixed)

    capacity = search_payload()
    capacity["models"]["dcnv2_low_rank_mix_capacity"]["candidates"].pop()
    with pytest.raises(SearchSpaceError, match="two ordered frozen rank brackets"):
        validate_search_space(capacity)


def test_candidates_may_vary_only_the_two_exact_learning_rates() -> None:
    extra_degree = search_payload()
    extra_degree["models"]["dnn"]["candidates"][0]["override"]["architecture"] = {
        "dropout": 0.2
    }
    with pytest.raises(SearchSpaceError, match="may vary only"):
        validate_search_space(extra_degree)

    same_value = search_payload()
    candidates = same_value["models"]["deepfm"]["candidates"]
    candidates[1]["override"] = deepcopy(candidates[0]["override"])
    with pytest.raises(SearchSpaceError, match="distinct"):
        validate_search_space(same_value)

    invented_value = search_payload()
    invented_value["models"]["dcn"]["candidates"][1]["override"]["optimizer"][
        "learning_rate"
    ] = 0.002
    with pytest.raises(SearchSpaceError, match="frozen learning rates"):
        validate_search_space(invented_value)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("trial_budget", "adaptive_trial_transfer"), True),
        (("trial_budget", "adaptive_candidate_generation"), True),
        (("trial_budget", "unused_trials_may_transfer"), True),
        (("selection_firewall", "tuning_seed"), 43),
        (("selection_firewall", "tuning_splits"), ["train", "validation", "test"]),
        (("selection_firewall", "test_split_may_select_candidates"), True),
    ],
)
def test_selection_firewall_and_no_adaptive_transfer(
    path: tuple[str, ...], value: Any
) -> None:
    search = search_payload()
    set_nested(search, path, value)
    with pytest.raises(SearchSpaceError):
        validate_search_space(search)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("models", "dnn", "base_config", "architecture", "input_width"), "bad"),
        (
            (
                "models",
                "dcnv2_full_rank",
                "base_config",
                "architecture",
                "use_low_rank_mixture",
            ),
            True,
        ),
        (
            (
                "models",
                "autoint_plus",
                "base_config",
                "architecture",
                "dnn_branch_enabled",
            ),
            False,
        ),
        (
            ("models", "afn_plus", "base_config", "architecture", "ensemble"),
            False,
        ),
        (
            (
                "models",
                "afn_plus",
                "base_config",
                "architecture",
                "logarithmic_neurons",
            ),
            16,
        ),
        (
            (
                "models",
                "afn_plus",
                "base_config",
                "architecture",
                "positive_log_treatment",
                "exp_clip",
            ),
            20.0,
        ),
        (
            ("models", "wukong_small", "base_config", "architecture", "num_layers"),
            1,
        ),
        (
            (
                "models",
                "dcdn_core",
                "base_config",
                "architecture",
                "local_fc_path",
            ),
            True,
        ),
        (
            (
                "models",
                "dcdn_core",
                "base_config",
                "architecture",
                "reciprocal",
                "epsilon",
            ),
            0.0001,
        ),
        (
            (
                "models",
                "dcdn_core",
                "base_config",
                "architecture",
                "reciprocal",
                "log_treatment",
            ),
            "safe_log",
        ),
        (
            ("models", "dcnv2_low_rank_mix_capacity", "display_label"),
            "DCNv2-capacity-matched",
        ),
        (
            (
                "models",
                "dcnv2_low_rank_mix_capacity",
                "base_config",
                "optimizer",
                "learning_rate",
            ),
            0.0003,
        ),
    ],
)
def test_model_specific_semantics_fail_closed(
    path: tuple[str, ...], value: Any
) -> None:
    search = search_payload()
    set_nested(search, path, value)
    with pytest.raises(SearchSpaceError):
        validate_search_space(search)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("sources", "fuxictr", "revision"), "0" * 40),
        (("sources", "wukong_paper", "url"), "https://example.invalid"),
        (
            (
                "model_provenance",
                "afn_plus",
                "protocol_adaptation",
                "auxiliary_branch_losses",
            ),
            True,
        ),
        (
            (
                "sources",
                "wukong_paper",
                "smallest_fully_enumerated_complete_architecture",
                "num_layers",
            ),
            1,
        ),
        (
            ("audit_policy", "published_result_values_used_for_selection"),
            True,
        ),
    ],
)
def test_model_source_provenance_mutations_fail_closed(
    path: tuple[str, ...], value: Any
) -> None:
    sources = sources_payload()
    set_nested(sources, path, value)
    with pytest.raises(SearchSpaceError):
        validate_model_sources(sources)


def test_definition_allows_only_declared_symbolic_dimensions() -> None:
    search = search_payload()
    resolved = resolve_search_space(search, phase="definition")
    capacity_lr = resolved["models"]["dcnv2_low_rank_mix_capacity"]["base_config"][
        "optimizer"
    ]["learning_rate"]
    assert isinstance(capacity_lr, str)
    assert capacity_lr.startswith("$")


def test_first_metric_requires_dimensions_hashes_and_capacity_decision() -> None:
    search = search_payload()
    with pytest.raises(SearchSpaceError, match="must be resolved"):
        resolve_search_space(search, phase="first_model_metric")

    resolutions = capacity_reuse_resolution(include_learning_rate=False)
    resolved = resolve_search_space(
        search,
        phase="first_model_metric",
        resolutions=resolutions,
    )
    assert resolved["models"]["dnn"]["base_config"]["architecture"]["input_width"] == 20
    assert (
        resolved["models"]["dcdn_core"]["base_config"]["architecture"][
            "operand_schema_sha256"
        ]
        == "b" * 64
    )


def test_dimension_relations_and_schema_hashes_fail_closed() -> None:
    search = search_payload()
    wrong_sum = capacity_reuse_resolution(include_learning_rate=False)
    wrong_sum["dimensions"]["DATASET_FLAT_WIDTH"] = 21
    with pytest.raises(SearchSpaceError, match="flat width"):
        resolve_search_space(
            search,
            phase="first_model_metric",
            resolutions=wrong_sum,
        )

    wrong_mask = capacity_reuse_resolution(include_learning_rate=False)
    wrong_mask["dimensions"]["DATASET_DENOMINATOR_MASK_WIDTH"] = 11
    with pytest.raises(SearchSpaceError, match="mask width"):
        resolve_search_space(
            search,
            phase="first_model_metric",
            resolutions=wrong_mask,
        )

    wrong_hash = capacity_reuse_resolution(include_learning_rate=False)
    wrong_hash["hashes"]["DATASET_OPERAND_SCHEMA_SHA256"] = "not-a-hash"
    with pytest.raises(SearchSpaceError, match="lowercase SHA-256"):
        resolve_search_space(
            search,
            phase="first_model_metric",
            resolutions=wrong_hash,
        )


def test_capacity_matching_is_metric_free_and_mechanical() -> None:
    search = search_payload()
    resolution = fallback_capacity_resolution()
    validate_capacity_resolution(search, resolution)

    wrong_choice = deepcopy(resolution)
    wrong_choice["decision"]["selected_candidate"] = "capacity_rank_lower"
    with pytest.raises(SearchSpaceError, match="mechanically selected"):
        validate_capacity_resolution(search, wrong_choice)

    non_adjacent = deepcopy(resolution)
    non_adjacent["candidates"]["capacity_rank_upper"]["low_rank"] = 5
    with pytest.raises(SearchSpaceError, match="adjacent"):
        validate_capacity_resolution(search, non_adjacent)

    metric_leak = deepcopy(resolution)
    metric_leak["metrics_consulted"] = True
    metric_leak["validation_metric"] = 0.5
    with pytest.raises(SearchSpaceError):
        validate_capacity_resolution(search, metric_leak)


def test_capacity_training_and_protocol_freeze_require_selected_dcnv2_lr() -> None:
    search = search_payload()
    without_lr = capacity_reuse_resolution(include_learning_rate=False)
    with pytest.raises(SearchSpaceError, match="SELECTED_DCNV2_LEARNING_RATE"):
        resolve_search_space(
            search,
            phase="capacity_control_training",
            resolutions=without_lr,
        )

    complete = capacity_reuse_resolution(include_learning_rate=True)
    resolved = resolve_search_space(
        search,
        phase="protocol_frozen",
        resolutions=complete,
    )
    inherited_lr = resolved["models"]["dcnv2_low_rank_mix_capacity"]["base_config"][
        "optimizer"
    ]["learning_rate"]
    assert inherited_lr == 0.0003

    invalid_lr = capacity_reuse_resolution(include_learning_rate=True)
    invalid_lr["hyperparameters"]["SELECTED_DCNV2_LEARNING_RATE"] = 0.0005
    with pytest.raises(SearchSpaceError, match="allowed_values"):
        resolve_search_space(
            search,
            phase="protocol_frozen",
            resolutions=invalid_lr,
        )


def test_search_lock_verifies_all_three_hashed_artifacts() -> None:
    lock = SearchSpaceLock.load(SEARCH_PATH)
    assert (
        lock.digest
        == SEARCH_PATH.with_suffix(SEARCH_PATH.suffix + ".sha256")
        .read_text(encoding="utf-8")
        .split()[0]
    )
    assert lock.model_sources_path == SOURCES_PATH
    assert lock.experiment_matrix_path == ROOT / "configs" / "experiment_matrix_v1.yaml"


def test_search_lock_rejects_path_traversal_before_related_read(
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    payload = search_payload()
    payload["model_sources"]["path"] = "../outside.yaml"
    search_path = configs / "search_space_v1.yaml"
    write_hashed(search_path, payload)

    with pytest.raises(SearchSpaceError, match="model_sources.path"):
        SearchSpaceLock.load(search_path)
