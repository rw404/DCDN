# Legacy results audit — multi-target reset

Artifact role: `legacy_result_audit`.
Basis: owner override recorded 2026-08-20 making the public headline study
**multi-target** and single-target training a **separate follow-up**.

Machine-readable companion:
`artifacts/legacy_audit/legacy_results_inventory_v1.json` (+ `.sha256`, `.csv`),
produced by `scripts/audit_legacy_results.py`. That generator reads only
configs, registry records, and result envelopes; it opens no dataset row, no
label, and no test metric.

## 1. Repository state at audit time

| item | value |
|---|---|
| clone | `/workspace/codex-home/dcdn/deep_division_network` |
| remote | `git@github.com:rw404/DCDN.git` (reachable, SSH) |
| branch | `experiments/wsdm27` |
| commit at audit start | `9fbc43d717577cf0be772b5bc52b7fb8e28ae636` |
| commit at inventory generation | `fb9efb0ac0885bf905b6d23ee56cdb70b9940848` |
| working tree | dirty |

The commit advanced during the audit because a concurrent `codex` session was
still recording the VK-LSVD `C+P+E` validation lineage. This audit does not
modify, stage, or revert any path owned by that session. The audit and the
multi-target reset only add new paths.

## 2. Entrypoints, configs, models, pipelines

**Entrypoints.** `scripts/train_model.py`, `scripts/run_worker.py`,
`scripts/preflight.py`, `scripts/materialize_selected_configs.py`,
`scripts/materialize_capacity_lr_binding.py`,
`scripts/validate_capacity_artifact.py`, `scripts/cost_guard.py`,
`scripts/storage_guard.py`, `scripts/gpu_smoke.py`, and the dataset
acquisition/feature scripts under `scripts/data/`.

**Models.** Nine registered interaction blocks in `src/dcdn/models/`:
`dnn`, `deepfm`, `dcn`, `autoint_plus`, `dcnv2_full_rank`, `afn_plus`,
`wukong_small`, `dcdn_core`, `dcnv2_capacity_matched_core`. All share one
context encoder, one linear representation adapter, and one common prediction
head (`src/dcdn/models/common.py`).

**Core.** `src/dcdn/core.py` implements `DCDN-Core-L{L}` with fixed anchor,
recurrence `x_i + x_0*product_block(x_i) + masked_reciprocal(x_0)*division_block(x_i)`,
a canonical semantic denominator mask, an FP32 reciprocal carrier with a
positive floor, and no local FC/BatchNorm/Dropout/concatenation/projector.
`src/dcdn/hybrid.py` preserves the production hybrid separately.

**Data pipelines.** VK-LSVD is fully built (`src/dcdn/data/vk_lsvd*.py`) with
materialized `C+P` and `C+P+E` runtime views. KuaiRand-Pure and KuaiRec have
feature/acquisition code and verified raw downloads but **no materialized
runtime views**. Yambda-50M has **no code and no data**.

**Result artifacts.** `results/tuning/`, `results/final_validation/`,
`results/smoke/`, `results/preflight/`, `results/failures/`, the run registry
under `runs/registry/`, per-run directories under `runs/`, and reproduction
commands under `runs/commands/`.

## 3. Inventory totals

| metric | value |
|---|---|
| registry records inventoried | 61 |
| completed | 40 |
| failed | 21 |
| **multi-target runs** | **0** |
| single-target runs | 61 |
| runs carrying a test metric | 0 |
| `legacy_exact_reuse` | 0 |
| `legacy_reproduced` | 0 |
| `legacy_context_only` | 40 |
| `unverifiable` | 21 |

Coverage by dataset, regime, and phase:

| dataset | regime | phase | records |
|---|---|---|---|
| `vk_lsvd_up0.01_ir0.01` | `C+P` | tuning | 20 |
| `vk_lsvd_up0.01_ir0.01` | `C+P` | final | 30 |
| `vk_lsvd_up0.01_ir0.01` | `C+P+E` | tuning | 7 |
| `vk_lsvd_up0.01_ir0.01` | `C+P+E` | final | 4 |

There is no KuaiRand-Pure, KuaiRec, Yambda-50M, or synthetic run of any kind.

## 4. What was used

| dimension | value |
|---|---|
| datasets | VK-LSVD `up0.01_ir0.01` only |
| splits | history weeks 0–19 (feature fit), train 20–24, validation 25, test 26 **sealed** |
| split row counts | train 7,505,820; validation 1,516,831 |
| targets | **one**: `normalized_watch_time = clip(timespent,0,157)/157` |
| loss | `normalized_mean_squared_error` (all 61 records) |
| selection metric | `normalized_mse`, minimize (all 61 records) |
| feature regimes | `C+P` and `C+P+E` |
| seeds | 42 (tuning + final), 43, 44 (final) |
| tuning budget | 2 learning rates (`1e-3`, `3e-4`), seed 42, validation only |
| batch / cap | effective batch 4096; 39,997,440 example cap; validate every ~4M |

## 5. Test-result exposure

**No test result has been produced or viewed.** Every result envelope carries
`test_metrics: null`; the frozen protocol carries `test_unlocked: false` and
`test_path_available: false`; VK week 26 was never opened by feature code.
Researchers have seen **validation** metrics only.

Two external-prior exposures are recorded and quarantined in
`docs/test_lock_audit.yaml`: a credential-free HEGM provenance lookup that
displayed a published benchmark table from incompatible released processed
views, and a Wukong configuration inspection that displayed adjacent published
results. Neither may inform features, search spaces, configurations, or
protocol decisions, and neither is a test result of this study.

## 6. Reuse gate

A legacy result may enter a new table only when **all** of the following are
recoverable: commit/config; data and split hashes; ordered feature manifest;
target and loss definitions; model configuration; seed; checkpoint/log or
row-level predictions; metric implementation.

The registry satisfies most of this well: every record carries
`code_commit`, `code_diff_sha256`, `resolved_config_sha256`,
`protocol_sha256`, `data_manifest_sha256`, `features_sha256`, `seed`, log and
history paths, and the resolved config additionally pins
`full_schema_sha256`, `operand_schema_sha256`, `denominator_mask_sha256`,
`encoder_contract_sha256`, and `capacity_resolution_sha256`.

Two facts nevertheless block every reuse into the new main table:

1. **All 61 runs are single-target.** The override makes the public headline
   study multi-target. A single-target number is not a row of a multi-target
   table, and per the override it may not be used to choose targets, feature
   families, loss weights, or models.
2. **The frozen feature manifest is superseded.** The override moves the three
   iALS cosines into the shared view and removes the six iALS norms from the
   denominator mask. Both changes alter `features_sha256` and
   `denominator_mask_sha256`, so no pre-override VK run matches the new
   manifest even structurally.

The 21 non-completed records are execution failures. Per AGENTS.md an
execution failure is never a scientific result; they are retained as failure
evidence only.

## 7. Disposition

* **Main multi-target table:** no legacy result is admitted. Every cell is
  produced fresh under the new manifests.
* **Target / feature-family / loss-weight / model selection:** no legacy
  result is consulted.
* **Single-target follow-up:** the 40 completed runs are retained as *material*
  but are **not** admissible results for it either, because the follow-up must
  use the same splits and feature manifests as the multi-target study, and
  those manifests changed. They are re-run under `single_target_followup_v1`.
* **Context:** the 40 completed runs remain citable as engineering evidence
  that the harness, encoder, capacity rule, registry, and nine-model suite
  execute end to end at VK scale, and as the source of the wall-clock cost
  model used for planning (≈4,200–5,200 s per 40M-example single-target A100
  run).

## 8. Per-run classification

| run_id | regime | phase | model | seed | val nMSE | class |
|---|---|---|---|---|---|---|
| `vk-cp-final-afn-plus-s43-v1-resume-v3` | C+P | final | afn_plus | 43 | 0.020785 | legacy_context_only |
| `vk-cp-final-afn-plus-s44-v1-resume-v2` | C+P | final | afn_plus | 44 | 0.020775 | legacy_context_only |
| `vk-cp-final-autoint-plus-s43-v1` | C+P | final | autoint_plus | 43 | 0.020892 | legacy_context_only |
| `vk-cp-final-autoint-plus-s44-v1` | C+P | final | autoint_plus | 44 | 0.020930 | legacy_context_only |
| `vk-cp-final-dcdn-core-l2-s43-v1-rebind-v1` | C+P | final | dcdn_core | 43 | 0.021001 | legacy_context_only |
| `vk-cp-final-dcdn-core-l2-s44-v1-rebind-v1` | C+P | final | dcdn_core | 44 | 0.021013 | legacy_context_only |
| `vk-cp-final-dcn-s43-v1` | C+P | final | dcn | 43 | 0.021029 | legacy_context_only |
| `vk-cp-final-dcn-s44-v1` | C+P | final | dcn | 44 | 0.021051 | legacy_context_only |
| `vk-cp-final-dcnv2-capacity-matched-r12-l2-s43-v1-rebind-v2` | C+P | final | dcnv2_capacity_matched_core | 43 | 0.021007 | legacy_context_only |
| `vk-cp-final-dcnv2-capacity-matched-r12-l2-s44-v1-rebind-v1` | C+P | final | dcnv2_capacity_matched_core | 44 | 0.021065 | legacy_context_only |
| `vk-cp-final-dcnv2-full-rank-l2-s43-v1` | C+P | final | dcnv2_full_rank | 43 | 0.020909 | legacy_context_only |
| `vk-cp-final-dcnv2-full-rank-l2-s44-v1` | C+P | final | dcnv2_full_rank | 44 | 0.020947 | legacy_context_only |
| `vk-cp-final-deepfm-s43-v1` | C+P | final | deepfm | 43 | 0.020838 | legacy_context_only |
| `vk-cp-final-deepfm-s44-v1` | C+P | final | deepfm | 44 | 0.020856 | legacy_context_only |
| `vk-cp-final-dnn-s43-v1` | C+P | final | dnn | 43 | 0.020949 | legacy_context_only |
| `vk-cp-final-dnn-s44-v1` | C+P | final | dnn | 44 | 0.020917 | legacy_context_only |
| `vk-cp-final-wukong-small-s43-v1-rebind-v1` | C+P | final | wukong_small | 43 | 0.020833 | legacy_context_only |
| `vk-cp-final-wukong-small-s44-v1-rebind-v1` | C+P | final | wukong_small | 44 | 0.020838 | legacy_context_only |
| `vk-cp-tune-afn-plus-s42-lr1e3-v1` | C+P | tuning | afn_plus | 42 | 0.020827 | legacy_context_only |
| `vk-cp-tune-afn-plus-s42-lr3e4-v1` | C+P | tuning | afn_plus | 42 | 0.020806 | legacy_context_only |
| `vk-cp-tune-autoint-plus-s42-lr1e3-v1` | C+P | tuning | autoint_plus | 42 | 0.021058 | legacy_context_only |
| `vk-cp-tune-autoint-plus-s42-lr3e4-v1` | C+P | tuning | autoint_plus | 42 | 0.020880 | legacy_context_only |
| `vk-cp-smoke-dcdn-core-l2-s42-1eb-v1` | C+P | tuning | dcdn_core | 42 | None | legacy_context_only |
| `vk-cp-tune-dcdn-core-l2-s42-lr1e3-v1` | C+P | tuning | dcdn_core | 42 | 0.020977 | legacy_context_only |
| `vk-cp-tune-dcdn-core-l2-s42-lr3e4-v1` | C+P | tuning | dcdn_core | 42 | 0.021021 | legacy_context_only |
| `vk-cp-tune-dcn-s42-lr1e3-v1` | C+P | tuning | dcn | 42 | 0.021052 | legacy_context_only |
| `vk-cp-tune-dcn-s42-lr3e4-v1` | C+P | tuning | dcn | 42 | 0.021036 | legacy_context_only |
| `vk-cp-tune-dcnv2-capacity-matched-r12-l2-s42-lr3e4-v1` | C+P | tuning | dcnv2_capacity_matched_core | 42 | 0.021005 | legacy_context_only |
| `vk-cp-smoke-dcnv2-full-rank-l2-s42-1eb-v1` | C+P | tuning | dcnv2_full_rank | 42 | None | legacy_context_only |
| `vk-cp-tune-dcnv2-full-rank-l2-s42-lr1e3-v1` | C+P | tuning | dcnv2_full_rank | 42 | 0.020912 | legacy_context_only |
| `vk-cp-tune-dcnv2-full-rank-l2-s42-lr3e4-v1-resume-v2` | C+P | tuning | dcnv2_full_rank | 42 | 0.020908 | legacy_context_only |
| `vk-cp-tune-deepfm-s42-lr1e3-v1` | C+P | tuning | deepfm | 42 | 0.020877 | legacy_context_only |
| `vk-cp-tune-deepfm-s42-lr3e4-v1` | C+P | tuning | deepfm | 42 | 0.020846 | legacy_context_only |
| `vk-cp-tune-dnn-s42-lr1e3-v1` | C+P | tuning | dnn | 42 | 0.020900 | legacy_context_only |
| `vk-cp-tune-dnn-s42-lr3e4-v1` | C+P | tuning | dnn | 42 | 0.020930 | legacy_context_only |
| `vk-cp-tune-wukong-small-s42-fixed-v1` | C+P | tuning | wukong_small | 42 | 0.020962 | legacy_context_only |
| `vk-cpe-final-dcnv2-full-rank-l2-s43-v1-receipt-v3` | C+P+E | final | dcnv2_full_rank | 43 | 0.020805 | legacy_context_only |
| `vk-cpe-final-dcnv2-full-rank-l2-s44-v1-receipt-v3` | C+P+E | final | dcnv2_full_rank | 44 | 0.020838 | legacy_context_only |
| `vk-cpe-tune-dcnv2-full-rank-l2-s42-lr1e3-v1-retry-v2-resume-v1` | C+P+E | tuning | dcnv2_full_rank | 42 | 0.020827 | legacy_context_only |
| `vk-cpe-tune-dcnv2-full-rank-l2-s42-lr3e4-v1-retry-v2` | C+P+E | tuning | dcnv2_full_rank | 42 | 0.020849 | legacy_context_only |

The 21 `unverifiable` records are omitted from the table above; they are all
`status=failed` execution failures and are listed in full in the JSON/CSV
inventory with their failure receipts under `results/failures/`.

## 9. Protocol conflicts surfaced by this audit

1. `configs/experiment_matrix_v1.yaml` declares `primary_targets_only: true`
   and `do_not_add_secondary_targets_before_mandatory_completion`. Superseded
   by the override; multi-target is now primary.
2. `src/dcdn/training.py` supports exactly one target, one loss, and no label
   masks. Multi-target requires per-target masked losses and Macro-NLoss.
3. `e_ials_viewtime_cosine`, `e_ials_feedback_cosine`, and
   `e_ials_survival_cosine` sit in the removable `E` set. The override requires
   cosine in both FULL and REDUCED; their lineage is `dot/(norm*norm)`, not a
   division of historical counters or sums, so they are not `R_div`.
4. `denominator_contract.eligible_fields` includes all six `p_ials_*_norm`
   fields. The override requires reciprocal mask `false` for cosine, dot, and
   norm, so the norms leave `D`.
5. AGENTS.md freezes Yambda-50M as optional and forbids scope expansion. The
   override makes Yambda-50M mandatory.

Each conflict is resolved in favour of the owner override and recorded in
`AGENTS.md` and `DCDN_HANDOFF.md`.
