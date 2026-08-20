"""Materialize the VK-LSVD C+P dataset-wide selected-config lineage lock.

The materializer is deliberately validation-only. It reopens the frozen protocol,
the three-seed validation summary, every referenced result/config/selection
receipt, and every selected checkpoint sidecar. It never constructs or opens a
test path and publishes the lock exactly once.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

from dcdn.checkpoint import verify_checkpoint_sidecar
from dcdn.protocol import (
    SelectedConfigsLock,
    load_verified_yaml_artifact,
    read_verified_artifact,
    validate_selected_configs_payload,
)

PROTOCOL_PATH = PurePosixPath("configs/protocols/vk_lsvd_c_plus_p/protocol_frozen.yaml")
SUMMARY_PATH = PurePosixPath(
    "results/final_validation/vk_lsvd_c_plus_p_rq1_validation_summary_v1.json"
)
OUTPUT_PATH = PurePosixPath("configs/protocols/vk_lsvd_c_plus_p/selected_configs.lock")
CAPACITY_MODEL_KEY = "dcnv2_capacity_matched_core"
DISPLAY_BY_MODEL = {
    "dnn": "DNN",
    "deepfm": "DeepFM",
    "dcn": "Original DCN",
    "autoint_plus": "AutoInt+",
    "dcnv2_full_rank": "DCNv2 full rank",
    "afn_plus": "AFN+",
    "wukong_small": "Wukong-small",
    "dcdn_core": "DCDN-Core-L2",
    CAPACITY_MODEL_KEY: "DCNv2-capacity-matched",
}
FINAL_SEEDS = (42, 43, 44)


class SelectedConfigsMaterializationError(RuntimeError):
    """Raised when validation lineage is incomplete, inconsistent, or unsafe."""


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise SelectedConfigsMaterializationError(message)


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SelectedConfigsMaterializationError(
                f"duplicate JSON key in evidence: {key}"
            )
        result[key] = value
    return result


def _repo_root(value: str | Path) -> Path:
    root = Path(value).resolve(strict=True)
    _expect(root.is_dir(), "repository root must be a directory")
    return root


def _relative(value: Any, field: str, *, suffix: str) -> str:
    _expect(isinstance(value, str) and bool(value), f"{field} must be a path")
    path = PurePosixPath(value)
    _expect(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
        and value.endswith(suffix),
        f"{field} is not a canonical repository-relative {suffix} path",
    )
    return value


def _load_json(
    root: Path,
    relative: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[Mapping[str, Any], str]:
    relative = _relative(relative, "JSON evidence path", suffix=".json")
    artifact = read_verified_artifact(root / relative)
    if expected_sha256 is not None:
        _expect(
            artifact.digest == expected_sha256,
            f"JSON digest mismatch: {relative}",
        )
    try:
        payload = json.loads(
            artifact.data.decode("utf-8"),
            object_pairs_hook=_strict_json_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SelectedConfigsMaterializationError(
                    f"non-finite JSON number in {relative}: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectedConfigsMaterializationError(
            f"invalid JSON evidence: {relative}"
        ) from exc
    _expect(isinstance(payload, Mapping), f"JSON root is not a mapping: {relative}")
    return payload, artifact.digest


def _load_config(
    root: Path,
    relative: str,
) -> tuple[Mapping[str, Any], str]:
    relative = _relative(relative, "resolved config path", suffix=".yaml")
    artifact = load_verified_yaml_artifact(root / relative)
    return artifact.payload, artifact.digest


def _finite_metric(value: Any, field: str) -> float:
    _expect(
        type(value) in {int, float} and math.isfinite(float(value)),
        f"{field} must be finite",
    )
    metric = float(value)
    _expect(metric >= 0.0, f"{field} must be non-negative")
    return metric


def _summary_models(summary: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    models = summary.get("models")
    _expect(isinstance(models, list), "validation summary models must be a list")
    by_display: dict[str, Mapping[str, Any]] = {}
    for item in models:
        _expect(isinstance(item, Mapping), "validation summary model is malformed")
        display = item.get("display_label")
        _expect(
            isinstance(display, str) and display not in by_display,
            "validation summary display labels must be unique strings",
        )
        by_display[display] = item
    _expect(
        set(by_display) == set(DISPLAY_BY_MODEL.values()),
        "validation summary model set differs from frozen protocol",
    )
    return by_display


def _seed42_selection(
    root: Path,
    *,
    model_key: str,
    summary_item: Mapping[str, Any],
) -> tuple[str, str, str, str, str, float, str, str]:
    evidence = summary_item.get("evidence")
    _expect(isinstance(evidence, Mapping), f"{model_key} summary evidence malformed")
    seed_entry = evidence.get("42")
    _expect(isinstance(seed_entry, Mapping), f"{model_key} seed-42 evidence missing")
    evidence_path = _relative(
        seed_entry.get("path"),
        f"{model_key} seed-42 evidence path",
        suffix=".json",
    )
    evidence_sha = seed_entry.get("sha256")
    _expect(isinstance(evidence_sha, str), f"{model_key} seed-42 SHA missing")
    payload, evidence_digest = _load_json(
        root,
        evidence_path,
        expected_sha256=evidence_sha,
    )

    if model_key == CAPACITY_MODEL_KEY:
        result_path = evidence_path
        run_id = payload.get("run_id")
        _expect(isinstance(run_id, str) and run_id, "capacity seed-42 run_id missing")
        config_path = f"configs/runs/{run_id}.yaml"
        config, _ = _load_config(root, config_path)
        training = config.get("training")
        _expect(isinstance(training, Mapping), "capacity seed-42 training malformed")
        receipt_path = _relative(
            config.get("capacity_lr_binding_path"),
            "capacity selection receipt path",
            suffix=".json",
        )
        receipt_sha = training.get("capacity_lr_binding_sha256")
        _expect(isinstance(receipt_sha, str), "capacity selection receipt SHA missing")
        _load_json(root, receipt_path, expected_sha256=receipt_sha)
        result_training = payload.get("training")
        _expect(
            isinstance(result_training, Mapping),
            "capacity seed-42 result training malformed",
        )
        best_name = result_training.get("best_checkpoint_filename")
        best_sha = result_training.get("best_checkpoint_sha256")
        metric = _finite_metric(
            result_training.get("best_validation_metric"),
            "capacity seed-42 validation metric",
        )
        run_directory = config.get("run_directory")
        _expect(isinstance(run_directory, str), "capacity run_directory missing")
        best_path = f"{run_directory}/{best_name}"
        return (
            config_path,
            result_path,
            best_path,
            str(best_sha),
            str(training.get("candidate_id")),
            metric,
            receipt_path,
            str(receipt_sha),
        )

    selected = payload.get("selected")
    _expect(isinstance(selected, Mapping), f"{model_key} selected evidence missing")
    config_path = _relative(
        selected.get("resolved_config_path"),
        f"{model_key} selected config path",
        suffix=".yaml",
    )
    result_path = _relative(
        selected.get("result_path"),
        f"{model_key} selected result path",
        suffix=".json",
    )
    best_path = _relative(
        selected.get("best_checkpoint_path"),
        f"{model_key} selected checkpoint path",
        suffix=".pt",
    )
    candidate_id = selected.get("candidate_id")
    best_sha = selected.get("best_checkpoint_sha256")
    _expect(isinstance(candidate_id, str) and candidate_id, "candidate_id missing")
    _expect(isinstance(best_sha, str), "selected checkpoint SHA missing")
    metric = _finite_metric(
        selected.get("validation_metric"),
        f"{model_key} seed-42 validation metric",
    )
    return (
        config_path,
        result_path,
        best_path,
        best_sha,
        candidate_id,
        metric,
        evidence_path,
        evidence_digest,
    )


def _validate_seed_lineage(
    root: Path,
    *,
    protocol_sha256: str,
    model_key: str,
    seed: int,
    config_path: str,
    result_path: str,
    expected_result_sha256: str | None,
    expected_checkpoint_path: str | None,
    expected_checkpoint_sha256: str | None,
    expected_candidate_id: str,
    expected_metric: float,
    selection_receipt_path: str,
    selection_receipt_sha256: str,
) -> tuple[dict[str, Any], str]:
    config, config_digest = _load_config(root, config_path)
    result, result_digest = _load_json(
        root,
        result_path,
        expected_sha256=expected_result_sha256,
    )
    training = config.get("training")
    _expect(isinstance(training, Mapping), f"{model_key}/{seed} training malformed")
    _expect(training.get("dataset_id") == "vk_lsvd_up0.01_ir0.01", "dataset drift")
    _expect(training.get("feature_regime") == "C+P", "feature regime drift")
    _expect(training.get("model_key") == model_key, f"{model_key}/{seed} model drift")
    _expect(
        type(training.get("seed")) is int and training["seed"] == seed, "seed drift"
    )
    _expect(
        training.get("candidate_id") == expected_candidate_id,
        f"{model_key}/{seed} candidate drift",
    )
    phase = training.get("phase")
    _expect(
        phase == ("tuning" if seed == 42 else "final"),
        f"{model_key}/{seed} phase drift",
    )
    _expect(result.get("scientific_result_eligible") is True, "result not eligible")
    _expect(result.get("test_metrics") is None, "test metrics must remain null")
    _expect(result.get("protocol_sha256") == protocol_sha256, "protocol digest drift")
    _expect(
        result.get("resolved_config_sha256") == config_digest, "config digest drift"
    )
    _expect(result.get("run_id") == config.get("run_id"), "run_id drift")
    result_training = result.get("training")
    _expect(isinstance(result_training, Mapping), "result training malformed")
    _expect(result_training.get("oom_count") == 0, "OOM result is not selectable")
    numerical = result_training.get("numerical_summary")
    _expect(isinstance(numerical, Mapping), "numerical summary missing")
    _expect(
        numerical.get("nonfinite_total") == 0, "non-finite result is not selectable"
    )
    _expect(numerical.get("pending") == 0, "numerical monitor is not drained")
    _expect(
        numerical.get("submitted") == numerical.get("completed"),
        "numerical monitor counts differ",
    )
    metric = _finite_metric(
        result.get("selected_validation_metric"),
        f"{model_key}/{seed} selected metric",
    )
    _expect(metric == expected_metric, f"{model_key}/{seed} metric drift")

    best_name = result_training.get("best_checkpoint_filename")
    best_sha = result_training.get("best_checkpoint_sha256")
    run_directory = config.get("run_directory")
    _expect(
        isinstance(best_name, str) and best_name.endswith(".pt"), "best name missing"
    )
    _expect(isinstance(best_sha, str), "best checkpoint SHA missing")
    _expect(isinstance(run_directory, str), "run_directory missing")
    best_path = f"{run_directory}/{best_name}"
    if expected_checkpoint_path is not None:
        _expect(best_path == expected_checkpoint_path, "selected checkpoint path drift")
    if expected_checkpoint_sha256 is not None:
        _expect(best_sha == expected_checkpoint_sha256, "selected checkpoint SHA drift")
    actual_checkpoint_sha = verify_checkpoint_sidecar(root / best_path)
    _expect(actual_checkpoint_sha == best_sha, "checkpoint sidecar mismatch")

    if seed != 42:
        _expect(
            training.get("selection_receipt_path") == selection_receipt_path,
            f"{model_key}/{seed} selection receipt path drift",
        )
        _expect(
            training.get("selection_receipt_sha256") == selection_receipt_sha256,
            f"{model_key}/{seed} selection receipt SHA drift",
        )

    model_digest = training.get("resolved_model_config_sha256")
    _expect(isinstance(model_digest, str), "resolved model digest missing")
    return (
        {
            "resolved_run_config_path": config_path,
            "resolved_run_config_sha256": config_digest,
            "validation_result_path": result_path,
            "validation_result_sha256": result_digest,
            "best_checkpoint_path": best_path,
            "best_checkpoint_sha256": best_sha,
            "selected_validation_metric": metric,
        },
        model_digest,
    )


def derive_vk_selected_configs_payload(repository_root: str | Path) -> dict[str, Any]:
    """Rebuild the lock payload from independently reopened validation evidence."""

    root = _repo_root(repository_root)
    protocol_artifact = load_verified_yaml_artifact(root / PROTOCOL_PATH)
    protocol = protocol_artifact.payload
    _expect(
        protocol.get("test_unlocked") is False, "premetric protocol must stay locked"
    )
    _expect(
        protocol.get("test_path_available") is False,
        "premetric protocol must not expose a test path",
    )
    summary, _ = _load_json(root, SUMMARY_PATH.as_posix())
    _expect(
        summary.get("dataset_id") == protocol.get("dataset_id"), "summary dataset drift"
    )
    _expect(
        summary.get("feature_regime") == protocol.get("feature_regime"),
        "summary feature regime drift",
    )
    summary_models = _summary_models(summary)
    required = protocol.get("required_model_keys")
    _expect(isinstance(required, list), "protocol required_model_keys malformed")
    _expect(
        required == list(DISPLAY_BY_MODEL),
        "protocol model order differs from frozen materializer order",
    )
    hashes = protocol.get("hashes")
    _expect(isinstance(hashes, Mapping), "protocol hashes malformed")

    selected_models: dict[str, Any] = {}
    for model_key in required:
        item = summary_models[DISPLAY_BY_MODEL[model_key]]
        values = item.get("values_by_seed")
        evidence = item.get("evidence")
        _expect(isinstance(values, Mapping), f"{model_key} values_by_seed malformed")
        _expect(isinstance(evidence, Mapping), f"{model_key} evidence malformed")
        seed42 = _seed42_selection(root, model_key=model_key, summary_item=item)
        (
            seed42_config,
            seed42_result,
            seed42_best,
            seed42_best_sha,
            candidate_id,
            seed42_metric,
            receipt_path,
            receipt_sha,
        ) = seed42
        _load_json(root, receipt_path, expected_sha256=receipt_sha)

        lineage: dict[str, Any] = {}
        model_digest: str | None = None
        for seed in FINAL_SEEDS:
            expected_metric = _finite_metric(
                values.get(str(seed)),
                f"{model_key}/{seed} summary metric",
            )
            if seed == 42:
                config_path = seed42_config
                result_path = seed42_result
                result_sha = None
                checkpoint_path = seed42_best
                checkpoint_sha = seed42_best_sha
                _expect(
                    expected_metric == seed42_metric, "seed-42 summary metric drift"
                )
            else:
                entry = evidence.get(str(seed))
                _expect(
                    isinstance(entry, Mapping), f"{model_key}/{seed} evidence missing"
                )
                result_path = _relative(
                    entry.get("path"),
                    f"{model_key}/{seed} result path",
                    suffix=".json",
                )
                result_sha = entry.get("sha256")
                _expect(isinstance(result_sha, str), f"{model_key}/{seed} SHA missing")
                result_payload, _ = _load_json(
                    root,
                    result_path,
                    expected_sha256=result_sha,
                )
                run_id = result_payload.get("run_id")
                _expect(isinstance(run_id, str) and run_id, "final run_id missing")
                config_path = f"configs/runs/{run_id}.yaml"
                checkpoint_path = None
                checkpoint_sha = None
            seed_lineage, current_model_digest = _validate_seed_lineage(
                root,
                protocol_sha256=protocol_artifact.digest,
                model_key=model_key,
                seed=seed,
                config_path=config_path,
                result_path=result_path,
                expected_result_sha256=result_sha,
                expected_checkpoint_path=checkpoint_path,
                expected_checkpoint_sha256=checkpoint_sha,
                expected_candidate_id=candidate_id,
                expected_metric=expected_metric,
                selection_receipt_path=receipt_path,
                selection_receipt_sha256=receipt_sha,
            )
            if model_digest is None:
                model_digest = current_model_digest
            _expect(
                current_model_digest == model_digest,
                f"{model_key} seed-invariant model digest drift",
            )
            lineage[str(seed)] = seed_lineage

        assert model_digest is not None
        selected_models[model_key] = {
            "candidate_id": candidate_id,
            "resolved_config_sha256": model_digest,
            "selection_receipt_path": receipt_path,
            "selection_receipt_sha256": receipt_sha,
            "final_seed_lineage": lineage,
            "selected_on_validation_only": True,
            "test_data_accessed": False,
            "test_metrics_consulted": False,
        }

    payload = {
        "schema_version": 2,
        "artifact_role": "final_selected_configs_lock",
        "dataset_id": protocol["dataset_id"],
        "feature_regime": protocol["feature_regime"],
        "parent_premetric_protocol_sha256": protocol_artifact.digest,
        "search_space_sha256": hashes["search_space_sha256"],
        "experiment_matrix_sha256": hashes["experiment_matrix_sha256"],
        "encoder_contract_sha256": hashes["encoder_contract_sha256"],
        "capacity_resolution_sha256": hashes["capacity_resolution_sha256"],
        "required_model_keys": required,
        "selected_models": selected_models,
    }
    validate_selected_configs_payload(payload)
    return payload


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise SelectedConfigsMaterializationError("short immutable write")
        view = view[written:]


def _publish_once(
    root: Path, relative: PurePosixPath, payload: Mapping[str, Any]
) -> str:
    data = (
        json.dumps(
            payload,
            sort_keys=False,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    digest = hashlib.sha256(data).hexdigest()
    sidecar = f"{digest}  {relative.name}\n".encode("ascii")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[int] = []
    created: list[str] = []
    current = os.open(root, flags)
    descriptors.append(current)
    try:
        for part in relative.parent.parts:
            child = os.open(part, flags, dir_fd=current)
            _expect(stat.S_ISDIR(os.fstat(child).st_mode), "unsafe output ancestor")
            descriptors.append(child)
            current = child
        for name, body in (
            (relative.name, data),
            (f"{relative.name}.sha256", sidecar),
        ):
            descriptor = os.open(name, file_flags, 0o640, dir_fd=current)
            created.append(name)
            try:
                _write_all(descriptor, body)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(current)
    except Exception:
        for name in reversed(created):
            with suppress(OSError):
                os.unlink(name, dir_fd=current)
        raise
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return digest


def materialize_vk_selected_configs_lock(
    repository_root: str | Path,
) -> SelectedConfigsLock:
    """Create the immutable selected lock and reopen it through the strict parser."""

    root = _repo_root(repository_root)
    payload = derive_vk_selected_configs_payload(root)
    digest = _publish_once(root, OUTPUT_PATH, payload)
    lock = SelectedConfigsLock.load(root / OUTPUT_PATH)
    _expect(lock.digest == digest, "published selected lock digest drift")
    return lock
