# Exact reproduction commands — multi-target study

Every command below is run from `deep_division_network/` after sourcing the
environment. `scripts/env.sh` pins the interpreter, caches, and telemetry
kill-switches; nothing here contacts an external service except the explicitly
listed public dataset downloads.

```bash
cd deep_division_network
source scripts/env.sh
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
```

## 0. Environment and static checks

```bash
"$PY" -m ruff check .
"$PY" -m ruff format --check .
PYTHONPATH=src "$PY" -m pytest -q
```

The multi-target additions are covered by three files and can be run alone:

```bash
PYTHONPATH=src "$PY" -m pytest -q \
  tests/test_multitask.py \
  tests/test_synthetic_multiquotient.py \
  tests/test_synthetic_runtime.py
```

## 1. Legacy audit

Regenerates `docs/LEGACY_RESULTS_AUDIT.md`'s machine-readable companion. Reads
only configs, registry records, and result envelopes; opens no dataset row, no
label, and no test metric.

```bash
"$PY" scripts/audit_legacy_results.py --repo .
```

Outputs: `artifacts/legacy_audit/legacy_results_inventory_v1.{json,json.sha256,csv}`.

## 2. Synthetic-MultiQuotient-3 (RQ1)

Status: **pending a clean replacement run**. The eight-arm output committed in
`8382d286` is quarantined diagnostic history, not RQ1 evidence; see
`artifacts/legacy_audit/synthetic_multiquotient_invalid_v0.json`.

The corrected runner executes the frozen M1–M7 matrix plus the linear reference
and analytical oracle (nine arms total). It requires a clean tracked commit,
rejects untracked code under `src/` or `scripts/`, verifies all four frozen
config sidecars, publishes an immutable pretraining manifest before the first
optimizer step, tunes only on seed-42 validation, freezes selected configs, and
then evaluates seeds 42/43/44 on the synthetic evaluation slices.

```bash
"$PY" scripts/run_synthetic_multiquotient.py \
  --device cuda:0 \
  --output-dir results/synthetic_multiquotient/rq1_v1
```

The output directory must not already exist. It contains
`pretraining_manifest_v1.json`, `selected_configs_v1.json`,
`multitask_summary.csv`, `multitask_per_target.csv`, and
`results_v1.json`; every artifact has a strict adjacent SHA-256 sidecar over
the exact written bytes. Record the runtime only after the valid run completes.

## 3. Matched serving-performance benchmark

Full multi-target forward with every head, `eval()` + `inference_mode()`,
device-resident inputs, batches `{1, 64, 1024}`, 200 warmups, 1,000
synchronized forwards, 5 independent rounds. Run it on an **idle** GPU.

```bash
"$PY" scripts/benchmark_serving.py --device cuda:1
```

Outputs: `results/synthetic_multiquotient/serving_performance_v1.json` (+
sidecar) and `efficiency.csv`. The script computes the runtime-matched verdict
itself: a pair may be called runtime matched only when
`p95(B=1) <= 1.10x` *and* `throughput >= 0.90x` at the shared maximum batch.
Otherwise the JSON records that the pair must be reported as a Pareto
trade-off.

## 4. Consolidated publication tables

```bash
"$PY" scripts/aggregate_multitask.py
```

Outputs under `results/multitask/`: `multitask_summary.csv`,
`multitask_per_target.csv`, `efficiency.csv`, `pareto.csv`, and
`multitask_index.json` (+ sidecar). `multitask_index.json` lists every
mandatory dataset that has **not** been executed, with the reason. A missing
cell is reported, never omitted.

## 5. Explanatory notebook

The `.py` percent-format file is the reviewable source of truth; the `.ipynb`
is generated and executed from it.

```bash
"$PY" notebooks/dcdn_explained.py                      # verify it runs
"$PY" scripts/nb_from_percent.py notebooks/dcdn_explained.py \
      --output notebooks/dcdn_explained.ipynb          # deterministic convert
```

To re-execute with outputs embedded (requires `nbformat` and `nbclient`, which
are outside the pinned project environment; the system interpreter carries
them on this host):

```bash
python - <<'PY'
import nbformat
from nbclient import NotebookClient
nb = nbformat.read("notebooks/dcdn_explained.ipynb", as_version=4)
NotebookClient(nb, timeout=3600, kernel_name="python3",
               resources={"metadata": {"path": "."}}, allow_errors=False).execute()
nbformat.write(nb, "notebooks/dcdn_explained.ipynb")
PY
```

MovieLens-1M is downloaded automatically into `$DCDN_DATA_CACHE` on first run
(5.9 MB, `sha256 a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`).
It is didactic only and never a result cell.

Determinism: two consecutive runs of the notebook produce bit-identical values
for every reported metric. Only wall-clock timings and latency measurements
differ, as they must.

## 6. Not yet executable

The remaining mandatory multi-target cells cannot be reproduced yet because
their inputs do not exist in this clone:

| dataset | blocking step |
|---|---|
| VK-LSVD | runtime views must be rebuilt: the override moves the three iALS cosines into `P_shared` and removes the six iALS norms from `D`, which changes `features_sha256` and `denominator_mask_sha256`; the four multi-target labels (`watch_time`, `like`, `share`, `dislike`) must also be materialized |
| KuaiRand-Pure | raw data downloaded and verified, but no runtime views are built |
| KuaiRec big matrix | raw data downloaded and verified, but no runtime views are built |
| Yambda-50M | no acquisition code and no data; the full pipeline is unwritten |

`configs/matrix_multitask_v1.yaml` fixes the execution priority for these
cells. `results/multitask/multitask_index.json` reports each of them as
`executed: false` with the reason, so no reader can mistake absence for
completion.
