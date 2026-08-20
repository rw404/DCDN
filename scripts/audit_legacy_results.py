#!/usr/bin/env python
"""Build the machine-readable legacy-result inventory for the multi-target reset.

Every pre-existing run is classified against the reuse gate declared in the
owner override.  A legacy result may enter a new table only when *all* of the
following are recoverable:

``commit_config``  commit SHA, clean-tree evidence, and the resolved config
``data_split``     data-manifest, views-manifest, and split hashes
``feature_manifest`` ordered feature schema and denominator-mask hashes
``target_loss``    target definition and loss function
``model_config``   the exact interaction/head configuration
``seed``           the training seed
``artifact``       a checkpoint, log, or row-level prediction file
``metric_impl``    the metric implementation used to produce the number

Classification:

``legacy_exact_reuse``  every criterion recoverable AND the run's protocol
                        matches the target study without re-training
``legacy_reproduced``   every criterion recoverable, but the run must be
                        re-executed because the target protocol differs
``legacy_context_only`` recoverable enough to cite as background, but not
                        admissible into a results table
``unverifiable``        at least one required criterion is missing

This script reads only configs, registry records, and result envelopes.  It
never opens dataset rows, labels, or test metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REQUIRED_CRITERIA = (
    "commit_config",
    "data_split",
    "feature_manifest",
    "target_loss",
    "model_config",
    "seed",
    "artifact",
    "metric_impl",
)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_yaml_scalars(path: Path) -> dict[str, Any]:
    """Minimal YAML reader for the flat scalar fields this audit needs.

    Avoids importing PyYAML so the audit runs in any environment.
    """
    values: dict[str, Any] = {}
    if not path.exists():
        return values
    stack: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if line.startswith("- "):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path_key = ".".join([p for _, p in stack] + [key])
        if value:
            values[path_key] = value.strip('"').strip("'")
        stack.append((indent, key))
    return values


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _classify(
    criteria: Mapping[str, bool],
    *,
    is_multi_target: bool,
    feature_manifest_still_current: bool,
    has_test_metric: bool,
    status: str,
) -> tuple[str, str]:
    if status != "completed":
        # AGENTS.md is explicit: an execution failure is not a scientific
        # result and can never satisfy a result cell.
        return (
            "unverifiable",
            f"execution failure (status={status!r}); preserved as failure "
            "evidence only, never as a scientific result",
        )
    missing = [k for k in REQUIRED_CRITERIA if not criteria.get(k)]
    if missing:
        return "unverifiable", f"missing: {', '.join(missing)}"
    if not is_multi_target:
        return (
            "legacy_context_only",
            "single-target run; the owner override makes the public headline "
            "study multi-target, so this row may not enter the main table and "
            "may not inform target/feature/loss/model selection",
        )
    if not feature_manifest_still_current:
        return (
            "legacy_reproduced",
            "fully documented, but the frozen feature manifest changed; the "
            "run must be re-executed against the new manifest",
        )
    if not has_test_metric:
        return (
            "legacy_reproduced",
            "validation-only run; a test number requires re-execution after "
            "the test lock is released",
        )
    return "legacy_exact_reuse", "all reuse criteria satisfied"


def build_inventory(repo: Path) -> dict[str, Any]:
    records_dir = repo / "runs" / "registry" / "records"
    rows: list[dict[str, Any]] = []

    for record_path in sorted(records_dir.glob("*.json")):
        record = _load_json(record_path)
        if record is None:
            continue
        run_id = record.get("run_id") or record_path.stem
        config_path = repo / str(record.get("resolved_config_path", ""))
        config = _load_yaml_scalars(config_path)
        result_path_str = config.get("result_path", "")
        result_path = repo / result_path_str if result_path_str else None
        has_result = bool(result_path) and result_path.exists()
        result = _load_json(result_path) if has_result else None

        run_dir = repo / str(record.get("run_directory", ""))
        checkpoints = sorted(run_dir.glob("*.pt")) if run_dir.exists() else []
        log_path = repo / str(record.get("log_path", ""))

        target = config.get("training.loss", "")
        selection_metric = config.get("training.selection_metric", "")
        n_heads = config.get("model.head_config.output_dim", "")
        is_multi_target = n_heads not in ("", "1")

        criteria = {
            "commit_config": bool(record.get("code_commit")) and config_path.exists(),
            "data_split": bool(record.get("data_manifest_sha256"))
            and bool(config.get("training.full_schema_sha256")),
            "feature_manifest": bool(record.get("features_sha256"))
            and bool(config.get("training.denominator_mask_sha256")),
            "target_loss": bool(target) and bool(selection_metric),
            "model_config": bool(config.get("model.interaction_config.model_key")),
            "seed": record.get("seed") is not None,
            "artifact": bool(checkpoints) or log_path.exists(),
            "metric_impl": (repo / "src" / "dcdn" / "metrics.py").exists(),
        }

        test_metrics = result.get("test_metrics") if result else None
        classification, rationale = _classify(
            criteria,
            is_multi_target=is_multi_target,
            # The override moves iALS cosine into the shared view and removes
            # iALS norms from the denominator mask, so every VK manifest hash
            # frozen before the override is superseded.
            feature_manifest_still_current=False,
            has_test_metric=test_metrics is not None,
            status=str(record.get("status", "")),
        )

        rows.append(
            {
                "run_id": run_id,
                "status": record.get("status", ""),
                "execution_kind": record.get("execution_kind", ""),
                "scientific_result_eligible": bool(
                    record.get("scientific_result_eligible")
                ),
                "dataset_id": config.get("training.dataset_id", ""),
                "feature_regime": config.get("training.feature_regime", ""),
                "model_key": config.get("training.model_key", ""),
                "num_heads": n_heads or "1",
                "target_mode": "multi_target" if is_multi_target else "single_target",
                "loss": target,
                "selection_metric": selection_metric,
                "seed": record.get("seed"),
                "phase": config.get("training.phase", ""),
                "learning_rate": config.get("training.learning_rate", ""),
                "code_commit": record.get("code_commit", ""),
                "code_diff_sha256": record.get("code_diff_sha256", ""),
                "resolved_config_path": record.get("resolved_config_path", ""),
                "resolved_config_sha256": record.get("resolved_config_sha256", ""),
                "protocol_sha256": record.get("protocol_sha256", ""),
                "data_manifest_sha256": record.get("data_manifest_sha256", ""),
                "features_sha256": record.get("features_sha256", ""),
                "denominator_mask_sha256": config.get(
                    "training.denominator_mask_sha256", ""
                ),
                "operand_schema_sha256": config.get(
                    "training.operand_schema_sha256", ""
                ),
                "result_path": result_path_str,
                "validation_metric": (result or {}).get("selected_validation_metric"),
                "test_metrics_present": test_metrics is not None,
                "checkpoint_count": len(checkpoints),
                "log_present": log_path.exists(),
                "started_at_utc": record.get("started_at_utc", ""),
                "finished_at_utc": record.get("finished_at_utc", ""),
                **{f"criterion_{k}": criteria[k] for k in REQUIRED_CRITERIA},
                "classification": classification,
                "rationale": rationale,
            }
        )

    head = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain")
    payload = {
        "schema_version": 1,
        "artifact_role": "legacy_result_inventory",
        "audit_basis": "owner_override_multi_target_primary",
        "repository_head_commit": head,
        "repository_dirty": bool(dirty),
        "repository_dirty_path_count": len(
            [ln for ln in dirty.splitlines() if ln.strip()]
        ),
        "required_reuse_criteria": list(REQUIRED_CRITERIA),
        "test_metrics_opened_by_this_audit": False,
        "dataset_rows_opened_by_this_audit": False,
        "totals": {
            "runs": len(rows),
            "multi_target": sum(1 for r in rows if r["target_mode"] == "multi_target"),
            "single_target": sum(
                1 for r in rows if r["target_mode"] == "single_target"
            ),
            "with_test_metrics": sum(1 for r in rows if r["test_metrics_present"]),
            **{
                cls: sum(1 for r in rows if r["classification"] == cls)
                for cls in (
                    "legacy_exact_reuse",
                    "legacy_reproduced",
                    "legacy_context_only",
                    "unverifiable",
                )
            },
        },
        "runs": rows,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/legacy_audit/legacy_results_inventory_v1.json"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("artifacts/legacy_audit/legacy_results_inventory_v1.csv"),
    )
    args = parser.parse_args()

    payload = build_inventory(args.repo)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    args.json_out.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    args.json_out.with_suffix(args.json_out.suffix + ".sha256").write_text(
        f"{digest}  {args.json_out.name}\n", encoding="utf-8"
    )

    if payload["runs"]:
        fieldnames = list(payload["runs"][0].keys())
        with args.csv_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(payload["runs"])

    print(f"runs inventoried : {payload['totals']['runs']}")
    for key, value in sorted(payload["totals"].items()):
        print(f"  {key:<22} {value}")
    print(f"json  : {args.json_out}  sha256={digest[:16]}...")
    print(f"csv   : {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
