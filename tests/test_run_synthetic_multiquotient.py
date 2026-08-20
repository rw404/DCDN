from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_synthetic_multiquotient.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "test_run_synthetic_multiquotient_module", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def test_runner_declares_every_frozen_matrix_arm(runner) -> None:
    base = runner._make_arms(len(runner.COLUMNS))
    assert tuple(arm.arm_id for arm in base) == (
        "linear",
        "dcnv2_reduced",
        "dcdn_reduced",
        "afn_reduced",
        "dcnv2_full",
        "dcdn_full",
        "afn_full",
        "oracle",
    )
    full = {arm.arm_id: arm for arm in base}
    assert full["dcnv2_full"].explicit_ratios is True
    assert full["dcdn_full"].explicit_ratios is True
    assert full["afn_full"].explicit_ratios is True
    assert full["afn_full"].with_afn_view is True


def test_full_encoder_keeps_appended_ratios_out_of_reciprocal_mask(runner) -> None:
    spec = runner.encoder_spec_for_full()
    assert spec.operand_dim == len(runner.COLUMNS) + len(runner.TARGET_NAMES)
    assert spec.denominator_mask[-3:] == (False, False, False)
    active = {
        column
        for column, enabled in zip(
            (*runner.COLUMNS, *runner.TARGET_NAMES),
            spec.denominator_mask,
            strict=True,
        )
        if enabled
    }
    assert active == set(runner.DENOMINATOR_COLUMNS)


def test_immutable_publication_hashes_exact_written_bytes(runner, tmp_path) -> None:
    payload = runner._canonical_json_bytes({"b": 2, "a": 1})
    path = tmp_path / "artifact.json"
    digest = runner._publish_bytes(path, payload)

    assert path.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    assert path.with_name(path.name + ".sha256").read_text(encoding="ascii") == (
        f"{digest}  artifact.json\n"
    )
    with pytest.raises(FileExistsError):
        runner._publish_bytes(path, payload)


def test_failed_sidecar_publication_removes_only_new_artifact(runner, tmp_path) -> None:
    path = tmp_path / "artifact.json"
    sidecar = tmp_path / "artifact.json.sha256"
    sidecar.write_text("preexisting\n", encoding="ascii")

    with pytest.raises(FileExistsError):
        runner._publish_bytes(path, b"payload\n")

    assert not path.exists()
    assert sidecar.read_text(encoding="ascii") == "preexisting\n"


def test_postpublication_reopen_failure_preserves_pair(
    runner, tmp_path, monkeypatch
) -> None:
    path = tmp_path / "artifact.json"
    original_read_bytes = Path.read_bytes

    def fail_artifact_reopen(self):
        if self == path:
            raise OSError("synthetic reopen EIO")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fail_artifact_reopen)
    with pytest.raises(OSError, match="reopen EIO"):
        runner._publish_bytes(path, b"payload\n")

    assert path.exists()
    assert path.with_name(path.name + ".sha256").exists()


def test_output_directory_must_be_fresh_and_below_results(
    runner, tmp_path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    results = repo / "results"
    results.mkdir(parents=True)
    monkeypatch.setattr(runner, "REPO_ROOT", repo)

    output = results / "synthetic" / "rq1"
    output.parent.mkdir()
    runner._prepare_output_dir(output)
    assert output.is_dir()

    with pytest.raises(FileExistsError):
        runner._prepare_output_dir(output)
    with pytest.raises(runner.ArtifactPublicationError):
        runner._prepare_output_dir(tmp_path / "outside" / "rq1")


def test_frozen_contract_sidecars_reopen_exactly(runner) -> None:
    hashes = runner._verified_contract_hashes()
    assert set(hashes) == {
        "configs/targets_multitask_v1.yaml",
        "configs/features_manual_hist_divisions_only_v1.yaml",
        "configs/matrix_multitask_v1.yaml",
        "configs/single_target_followup_v1.yaml",
    }
    assert all(len(value) == 64 for value in hashes.values())
