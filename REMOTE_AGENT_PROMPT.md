# Remote Codex prompt for the DCDN RunPod

Updated: 2026-08-18 (Europe/Moscow)

Copy this file next to `DCDN_HANDOFF.md`, `AGENTS.md`, `DCDNBlock_provided.py`, `DCDNCore_reference.py`, and the remaining DCDN source files on the RunPod. Open Codex in that directory and paste everything under **Prompt to paste**. The `/goal` workflow is documented at <https://learn.chatgpt.com/use-cases/follow-goals>. If `/goal` is unavailable, run `codex features enable goals`, restart Codex, and retry; otherwise submit the same text as a normal task.

## Prompt to paste

/goal Build a reproducible DCDN research repository on this existing two-A100 RunPod and execute the frozen public offline **multi-target** experiment plan for the WSDM 2027 paper, under the owner override in section 0. Work autonomously through validated core results and the artifact audit, but never inspect test metrics before the protocol and selected configurations are frozen. Do not expand scope or tune after test access. Stop new compute at the earliest hard time or budget limit below.

### 0. Owner override recorded 2026-08-20 — read first, it supersedes conflicts below

This override governs. Where any later section of this prompt disagrees with
it, the override wins and the later text is retained only for provenance.

1. **Multi-target is primary.** The public headline study trains and reports
   multi-target models. Every row of the main table is multi-target.
2. **Single-target is a separate follow-up.** It runs only after the main
   multi-target table is complete, uses the same splits, feature manifests, and
   structural configurations, and changes only the objective. It may **not**
   enter the main table and may **not** inform the choice of targets, feature
   families, loss weights, or models. Its protocol is frozen in advance in
   `configs/single_target_followup_v1.yaml`, before the first test unlock, so
   its headline targets cannot be picked after seeing multi-target results.
3. **Yambda-50M is mandatory**, not optional. The mandatory datasets are
   VK-LSVD, Yambda-50M, KuaiRand-Pure, and KuaiRec big matrix. Sections 7 and
   11 below, which describe Yambda as optional, are superseded.
4. **Only manual historical divisions are removed.** The reduced feature view
   removes exactly the manually materialized divisions of historical counters
   and sums. Cosine similarities, dot products, vector norms, and iALS scores
   are present in **both** FULL and REDUCED, carry reciprocal mask `false`, are
   never members of the denominator set `D`, and are never counted as removed
   ratio features.
5. **Public claim wording is restricted accordingly.** State removal and
   substitution of *manually engineered divisions of historical counters and
   sums*. Never state that all ratio-derived features were removed. The 40.1%
   production reduction stays a separate industrial fact and is never compared
   with the public-dataset counts.
6. **The objective is Macro-NLoss**: per-target losses averaged over observed
   labels only, each divided by a train-estimated frozen constant (`H_t` for
   binary, `V_t` for regression), then averaged unweighted. Per-target label
   masks are mandatory; missing labels are never zeros and an observed `0` is a
   valid negative. Adaptive task weights, learned uncertainty weights,
   GradNorm, and post-result reweighting are forbidden. Publish every target
   separately in addition to the aggregate.
7. **`Synthetic-MultiQuotient-3` replaces the MNIST-like division test** as the
   RQ1 sanity benchmark. The legacy notebooks are follow-up material only.
8. **Infrastructure gaps are not blockers.** Absence of RunPod provider-side
   expiry, a live billing API, or credentialed inspection does not block work.
   Do not use stored RunPod credentials; keep the local ledger per AGENTS.md.

Frozen artifacts: `configs/targets_multitask_v1.yaml`,
`configs/features_manual_hist_divisions_only_v1.yaml`,
`configs/matrix_multitask_v1.yaml`, `configs/single_target_followup_v1.yaml`.
Legacy disposition: `docs/LEGACY_RESULTS_AUDIT.md`. No pre-override result is
admissible into the multi-target table.

The seven-row matrix, run identically on every dataset, is
`M1 DCNv2-REDUCED`, `M2 DCDN-Core-REDUCED`, `M3 AFN+-REDUCED`,
`M4 DCNv2-capacity-matched-REDUCED`, `M5 DCNv2-FULL`, `M6 DCDN-Core-FULL`,
`M7 AFN+-FULL`, with all seven declared contrasts published regardless of
direction. Do not assume in advance that DCDN wins or loses; measure first.

### 1. Read and reconcile before acting

1. Read `DCDN_HANDOFF.md`, `AGENTS.md`, and this `REMOTE_AGENT_PROMPT.md` completely.
2. Inspect every sibling TeX, PDF, notebook, configuration, and source file relevant to DCDN. Treat `DCDNBlock_provided.py` as the user-supplied source for the one-layer production hybrid integration; preserve it unchanged as provenance.
3. Return a compact understanding check covering the contribution, exact architecture found in source, safe theoretical claims, confirmed evidence, frozen experiment scope, external blockers, and first checkpoint.
4. If the handoff, `DCDNBlock_provided.py`, or `DCDNCore_reference.py` is absent, stop and list the missing files exactly. Do not reconstruct the architecture from prose alone. These files remove the earlier “no executable source” blocker; the missing production wrapper/configuration limits only exact end-to-end production reproduction, not the frozen public core.

Use **DCDN — Deep Cross-Division Network** exclusively as the architecture name. “Laurent polynomial” is a mathematical characterization, not a model name. Treat the handoff as the source of truth unless verified source or new user-provided evidence establishes a discrepancy; surface material discrepancies before changing claims or protocol.

This RunPod task covers public offline experiments only. It cannot close missing industrial A/B statistics, reproduce private industrial rows, or prove offline-to-online translation. Industrial results in the handoff are read-only context owned by the user/local manuscript workflow.

For manuscript-facing prose, report the production feature-inventory change only as a **40.1% reduction** and state that DCDN is deployed in the production ranking stack of the major short-video service. Keep absolute industrial feature counts only in private run identifiers or internal reproducibility artifacts; do not place them in the abstract or public narrative.

### 2. Hard resource envelope and machine audit

Create `ledger_epoch_utc` exactly once at the beginning of the first execution and persist it in `docs/environment.md` and `docs/cost_ledger.md`. On every restart, continuation, or repeated prompt, reuse the earliest recorded epoch and cumulative ledger; never reset either limit. If a pre-existing project ledger is found, adopt its earliest verifiable epoch. Separately record any bill already incurred before that epoch if available.

- `MAX_CUMULATIVE_SPEND_USD = 460` before tax, including all metered compute and storage since `ledger_epoch_utc`.
- Warning threshold: USD 370 projected or spent.
- Maximum Pod wall time from `ledger_epoch_utc`: 120 hours.
- Absolute compute stop: the earlier of `ledger_epoch_utc + 120 hours` and **2026-08-24 14:00 Europe/Moscow**.
- Reserve the last 8 hours before that stop for aggregation, integrity checks, and durable backup.
- Do not launch a job if its conservative ETA would cross the reserved window or make projected spend exceed USD 460.

Before installing or downloading anything, record:

- current user, working directory, permissions, and whether the repository path is persistent;
- GPU count/model, NVLink visibility, driver, CUDA, PyTorch, and Python;
- CPU count, RAM, mount points, filesystem types, free space, and measured read/write speed;
- Git state, source hashes, pre-existing dirty/untracked paths, and start time;
- locally visible price/balance information, if available without credentials. Do not block when it is unavailable: the owner-provided deployment quote is USD 3.29 per Pod-hour for this 2×A100 Pod, and the owner has explicitly waived credentialed control-plane inspection.

Expected but not guaranteed: 2×A100 SXM 80GB, 32 vCPU, about 234GB RAM, PyTorch 2.8/CUDA 12.8, a 1TB temporary container disk, and a 50GB persistent network volume at `/workspace`. Trust local OS/filesystem inspection for usable resources; missing control-plane quota evidence is not a blocker.

The repository, run registry, resolved configs, manifests, compact logs, results, and selected final checkpoints must live on verified persistent storage. If the current directory is temporary, place the repository under a writable persistent path such as `/workspace/dcdn`; if relocation is impractical, create a persistent mirror and verify sync after every checkpoint. Use the large temporary disk only for public raw data, caches, active `last` checkpoints, and other regenerable artifacts.

Determine persistent capacity and free space from the mounted filesystem. On the expected 50GB volume, keep at least 10GB free and limit project-owned durable artifacts to 40GB. If the local filesystem is smaller, keep both 20% and at least 5GB free and reduce retention accordingly. Persist `best` checkpoints only for selected final configurations; active `last` checkpoints may remain on temporary storage and must be copied only when needed for recovery. After a run is validated, remove disposable smoke/tuning checkpoints but retain configs, metrics, logs, hashes, and failure evidence. Never delete an irreplaceable artifact.

Maintain a conservative incremental cost ledger and ETA after every checkpoint using USD 3.29 per elapsed Pod-hour plus a fixed conservative USD 25 storage reserve whenever control-plane billing data is unavailable. Label these numbers as estimates, and do not block on unavailable historical spend or account headroom. At the warning threshold, stop optional work and finish the causal core. At the hard limit, stop all training/evaluation processes after preserving evidence. **Owner override recorded 2026-08-18:** no provider-side expiry is configured or required for this existing Pod, and credentialed RunPod inspection is not authorized. The owner is responsible for manual termination. Never stop/terminate the Pod or delete its volume autonomously; finish with zero task-owned processes, verified durable backup, and an explicit warning that billing continues until the owner terminates the Pod.

### 3. Repository and security rules

Use the verified persistent directory as repository root.

- If absent, initialize a local Git repository and create branch `experiments/wsdm27`.
- Inventory pre-existing files and dirty state. Stage only explicit task-owned paths; never use broad staging over unknown files.
- Before each commit, check staged paths for secrets, credentials, SSH material, large data, checkpoints, `.env`, caches, and generated logs.
- Preserve all original sources and unrelated user changes. Make local checkpoint commits after bootstrap, model/tests, each data pipeline, and each validated table.
- Never create or mutate a remote repository, push, open an issue/PR, publish, upload, send a message, trigger a webhook, or write to any external service without explicit approval.
- Disable W&B, MLflow, Hugging Face telemetry, crash reporting, and all other third-party telemetry by default.
- Approved public dataset/code downloads are read-only external actions. Pause before accepting a license on the user's behalf, using credentials, or downloading any unplanned artifact larger than 50GB.
- Use an isolated user-local Python environment and pinned dependencies. Do not use `sudo`, replace system CUDA/drivers, modify the OS/firewall, expose a public port, or create a persistent system service without approval.

Create or normalize without gratuitously moving source files:

- `src/dcdn/`, `configs/`, `scripts/`, `tests/`, `third_party/`;
- `runs/`, `results/`, `docs/`, `paper/`;
- `README.md`, `RUNBOOK.md`, `STATUS.md`, and `AGENTS.md`.

Record a clean-environment reproduction command. Before completion, actually create a fresh isolated environment and run one deterministic end-to-end smoke experiment from the lockfile; record command, versions, result, and artifact hash.

### 4. Controlled model interface and numerical contract

Implement one interface for:

- DNN/MLP;
- DeepFM;
- original DCN;
- AutoInt+;
- DCNv2;
- AFN+;
- Wukong-small;
- DCDN.

Pin source URL, commit, license, and exact configuration for external implementations. AFN+ is the closest algebraic prior and belongs in the causal core. Before validation, freeze AFN+'s positive-input/log treatment. For Wukong-small, freeze one smallest documented public configuration, its interaction stack, width, rank, optimizer, and parameter budget; do not invent a scaled implementation. If no reproducible implementation exists, report this before test unlock.

All models use the same examples, labels, shared context encoder `C`, prediction tower/head, loss, optimizer family, maximum training steps/examples, validation cadence, patience, selection metric, effective global batch, and tuning-trial count. Report realized examples seen. Models within one feature regime use exactly the same schema and order. Causal variants intentionally differ only by predeclared `E/R` fields; assert that all `C+P` columns and values remain identical.

The public-core capacity control is reported as `DCNv2-capacity-matched(C+P)`, never `DCDN-sham`, and must be resolved for every causal table. Retain `dcnv2_capacity_matched_core` as the stable internal key and `DCNv2-capacity-matched-core` only as a historical frozen provenance alias; do not rewrite or re-hash existing search/view locks solely for this naming update. Match trainable non-embedding parameters and FLOPs to `DCDN-Core` within ±5%. If ordinary `DCNv2(C+P)` already meets both tolerances, reuse that exact run and do not create a duplicate cell. Otherwise, predeclare two bracketing ranks and choose mechanically—before any model metric—by the smaller frozen distance `|log(P_dcn/P_dcdn)| + |log(F_dcn/F_dcdn)|`, where `P` is trainable non-embedding parameters and `F` is FLOPs/example. If neither bracket meets tolerance, report both as diagnostics and mark exact capacity matching unavailable; never choose the control by validation quality. This is distinct from the production `Hybrid-second-product-control`, which retains the exact FC/projector wrapper and replaces only the reciprocal carrier with a second product carrier.

Confirm the DCDN recurrence from source rather than copying the legacy symbolic expansion. Preserve the residual and separate product/division biases.

Maintain two explicit architecture labels:

- `DCDN-Core-L{L}` is the canonical public/offline model and the object of the Laurent-support theory. It applies the residual/product/reciprocal recurrence directly to named raw or safely shifted base operands, keeps the original anchor fixed across layers, and has **no local FC path, BatchNorm, Dropout, concatenation, or projector**. Shared context encoding and the common prediction head live outside the core and are identical across models.
- `DCDN-Hybrid-Prod-L1-PLE8` is the deployed integration represented by `DCDNBlock_provided.py`: one fixed-anchor block initialized with `x_i=x_0` over `[B,8,F]` PLE feature embeddings, with its width-32 FC path and final projector. Preserve it for the production section and optional bridge ablation; do not use it in the main public tables.

Never report a `DCDN-Core` public run as an exact reproduction of the deployed hybrid. Conversely, do not apply the raw-space Laurent theorem to the complete production block. With exact reciprocals, the production cross-division subpath is Laurent in the lifted PLE coordinates; PLE is not an algebra homomorphism, and the complete stabilized nonlinear hybrid is generally piecewise rational in raw inputs.

The production block's shape contract is now user-confirmed. Inspect and verify these facts exactly:

- `nn.Linear` acts on the last dimension; `BatchNorm1d(8)` treats the middle dimension as the eight PLE channels; `x_0,x_i` and output have shape `[B,8,F]`;
- production uses exactly one block, initializes `x_i=x_0`, and keeps `x_0` fixed;
- the class applies the reciprocal stabilizer to every PLE coordinate of `x_0` and has no semantic raw-denominator mask;
- its returned state includes the nonlinear FC/BatchNorm/Dropout path and projector, so the global Laurent theorem applies only to the exact cross-division subpath, not the complete training-time hybrid;
- the exact PLE implementation, ordered input schema, surrounding wrapper, and remaining production configuration are still unconfirmed; `x_i=x_0` is confirmed.

Do not silently “fix” the preserved production source. Implement `DCDN-Core` in a separate file with explicit provenance and tests, using the recurrence in the handoff and raw/base operands in the public protocol. The production shape, one-block depth, `x_i=x_0` initialization, fixed anchor, and deployed side path no longer require clarification and must not block public core experiments.

- In `DCDN-Core`, the reciprocal branch may address only a named mask of semantically meaningful positive denominator operands. This raw mask is not a claim about the unmasked production PLE implementation.
- In `DCDN-Core`, categorical, user/item ID, PLE, arbitrary learned, and content embeddings bypass the reciprocal branch or join only product/deep paths.
- Do not z-center denominators. Use positive scale-only normalization, smoothing/floors, missingness/validity indicators, and logged clipping/floor incidence.
- Treat an invalid negative denominator candidate as masked/rejected according to the frozen policy; do not silently reinterpret it.
- Distinguish exact reciprocal theory from stabilized implementation.
- Before optimized training, scan the complete materialized public operand views for finiteness and positivity of every enabled denominator. Use `runtime_value_checks=true` in smoke/debug tests only; disable synchronous Python tensor predicates in the full loop and monitor non-finite losses/activations/gradients asynchronously.
- Freeze small residual-near product/division initialization and apply the same product-branch initialization to DCN/DCNv2 controls. Freeze gradient clipping, BF16-or-FP32 model precision (not FP16 for the reciprocal path), FP32 reciprocal computation under AMP, epsilon, and per-field floor-incidence reporting before validation metrics.

Before full runs, require finite forward/backward/update tests; zero, near-zero, missing, invalid-negative, and large-denominator cases; reciprocal-mask isolation; one-batch overfit; metric tests; checkpoint save/load; and deterministic resume including optimizer, scheduler, sampler, data order, and RNG state.

### 5. Frozen research questions and exact experiment matrix

- **RQ1 — Feature substitution:** can DCDN recover predictive value supplied by explicit engineered ratios from matched base operands?
- **RQ2 — Effectiveness:** on identical `C+P`, does DCDN outperform DCNv2 and representative interaction models?
- **RQ3 — Reciprocal mechanism:** does the division branch help beyond additional parameters/compute in the raw-feature core? A separate compact production-style ablation, when industrial offline results are supplied by the owner, distinguishes division, the FC path, and their interaction. Optional reciprocal input lifting remains lower priority.
- **RQ4 — Public stability/efficiency:** what are numerical stability and model-only efficiency on public tasks, contextualized alongside—without claiming translation to—the separate industrial evidence?

Notation:

- `C`: shared categorical/context/static fields, never divided;
- `P`: predeclared base operands, including matched smoothed numerators/denominators and permitted derived bases such as iALS operands;
- `E/R`: explicit engineered ratios/rates available only to the full-feature DCNv2 control.

#### Tier A — mandatory causal core

For **KuaiRand-Pure** and **VK-LSVD**, run exactly these five rows on the primary target, seeds 42/43/44:

1. `DCNv2(C+P)`;
2. `DCNv2(C+P+E/R)`;
3. `DCDN(C+P)`;
4. `DCNv2-capacity-matched(C+P)`;
5. `AFN+(C+P)`.

For **KuaiRec**, run the analogous five rows on its primary watch-time target, seeds 42/43/44. Valid `DCNv2(C+P)`, `DCDN(C+P)`, and `AFN+(C+P)` results are reused in later tables; never rerun an identical cell just to populate another table.

In every mandatory public row, `DCDN(C+P)` means `DCDN-Core` on raw/base operands with the local production FC path/projector absent. Its parameter/FLOP control must be matched to this core, not to `DCDN-Hybrid-Prod-L1-PLE8`.

#### Tier B — planned `C+P` architecture table

On KuaiRand-Pure and VK-LSVD primary targets only, the full table is DNN/MLP, DeepFM, DCN, AutoInt+, DCNv2, AFN+, Wukong-small, and DCDN, all on `C+P`, seeds 42/43/44. Reuse the three Tier-A rows. Do not multiply this table by `C+P+E/R` or by secondary labels.

After one-epoch throughput measurements, write a projection for all remaining Tier-A and Tier-B cells. If they cannot finish within 96 Pod hours from `ledger_epoch_utc` while reserving artifact time and staying below budget, pause before test unlock and ask the user to approve a reduced Tier-B scope. Never silently drop a baseline.

#### Tuning budget

Before any model metric, freeze one finite validation grid in `configs/search_space_v1.yaml`: use exactly two candidates for every tuned model/variant. A canonical fixed configuration may use one trial only if it is explicitly marked `fixed_no_tune` with its source and reason before metrics. Do not transfer unused trials adaptively. Tune using seed 42 and validation only. The selected seed-42 final run may be reused; do not duplicate it.

Secondary like/share/dislike heads, explicit reciprocal lifting, synthetic tasks, Yambda, and additional datasets are optional only after Tier A, the approved Tier B scope, KuaiRec, uncertainty, and efficiency outputs are complete. If the owner cannot supply the industrial division-by-FC ablation and the paper will attribute production gains specifically to division, run a public bridge on VK-LSVD using the exact PLE-8 hybrid wrapper: the four gated cells `(division, FC) in {(0,0),(0,1),(1,0),(1,1)}` plus `Hybrid-second-product-control`, which retains FC/projector and replaces the reciprocal carrier by a second product carrier. Keep projector/output geometry fixed with zero branch tensors. This bridge is a mandatory fallback only for the production-specific attribution claim; otherwise omit that claim and report the A/B as evidence for the complete package. Never let the bridge replace the raw-core tables.

### 6. Freeze feature algebra and block test access

Before observing any validation model metric, create and hash a dataset-specific `features_v1.yaml` containing:

- every `C`, `P`, and `E/R` field in order;
- grouping keys, windows, exact numerator/denominator formula, smoothing `alpha/beta`, support threshold, missing/OOV fallback, validity indicator, normalization, and denominator mask;
- label transform, split boundaries, state-update policy, tie-breaking/deduplication, and audit-only IDs;
- every primary metric, model-selection rule, paired contrast, bootstrap cluster, and multiplicity rule.

Do not optimize public feature counts to resemble the industrial 40.1% reduction.

Create `protocol_frozen.yaml` with hashes of data/feature manifests, split, primary targets/metrics, search space, seeds, and exact matrix. Training code must suppress all test evaluation while `test_unlocked=false`. Run validation-only tuning, then write and hash `selected_configs.lock`. Only after all planned configurations for that dataset are frozen may `test_unlocked=true`; evaluate each final seed on test exactly once. Never change protocol or selected configuration after any test metric is visible.

Use paired effect estimates and user- or time-clustered 95% bootstrap confidence intervals. Do not call a nonsignificant difference non-inferior. Report continuous deltas and CIs unless a dataset-specific non-inferiority margin was explicitly frozen before test unlock. Apply the frozen multiplicity policy to the small set of primary contrasts.

### 7. Dataset roles

Mandatory:

1. KuaiRand-Pure — fast causal pilot and randomized-exposure-policy test.
2. VK-LSVD `up0.01_ir0.01` — scalable primary public benchmark.
3. KuaiRec big matrix — mechanistic base-operand versus explicit-rate ablation.

Superseded by the override: Yambda-50M is mandatory, not optional-after-mandatory-work.

Do not add KuaiSAR, EB-NeRD, Criteo, Avazu, Tenrec, MicroLens, or another dataset without explicit approval. HEGM supports only VK-LSVD and KuaiRec; it is a preprocessing/split reference, not a ready DCDN feature-substitution dataset. Never upload industrial or private data/code to RunPod.

Retain user/item IDs and the finest released time key as audit-only columns; use a timestamp when released, otherwise `(week, stable in-file row order)` for VK-LSVD. Exclude audit keys from the model unless a field is separately predeclared in `C`. They are required for leakage checks, cold coverage, grouping, and clustered intervals.

### 8. KuaiRand-Pure protocol

Use the official release and validate row counts, timestamps, hashes, and coverage before accepting the split.

- history/burn-in: standard events, 2022-04-08 through 2022-04-21;
- training labels: standard events, 2022-04-22 through 2022-05-02;
- validation: randomized events, 2022-05-03 through 2022-05-05;
- untouched headline test: randomized events, 2022-05-06 through 2022-05-08;
- secondary in-policy test: standard events over the same final dates.

If released coverage differs, stop before metrics and report it; do not move boundaries silently. Freeze model weights after training. Use a rolling/prequential feature-state simulation: ingest the union of standard and randomized rows and order globally by `time_ms`. Score **every row in one timestamp group** against the same state containing only events with time `<t`; only after all rows at `t` are scored may the entire group update state in deterministic source-priority/stable-row order. Never shuffle evaluation. Deduplicate only canonically normalized byte-identical full rows using a full-row fingerprint; never deduplicate merely on `(user_id, video_id, time_ms)`. Every aggregate uses `time_ms < target_time`.

Build candidate-pool histories for predeclared user/item/author/tab keys and 1-day, 7-day, and all-history windows. `P` contains smoothed operands for exposures, long views, likes, forwards, hates, follows, play-time/duration/dwell sums, support, and validity. `R` contains only the exact lagged formulas frozen in `features_v1.yaml`.

Primary target: `long_view`; primary selection/reporting metric: LogLoss. Also report ROC-AUC and user-GAUC. For user-GAUC, exclude single-class users from the average but report retained-user and retained-row coverage. Rare labels and AUPRC are optional finalist analyses.

Keep randomized-test items absent from standard history using an OOV ID, zero-history operands, validity mask, and safe static metadata. Never inner-drop them. Report cold-item coverage and warm-only results.

Never use current-row outcomes, `is_rand`, or monthly `video_features_statistic.csv` as features. Randomized rows estimate performance under the released random-exposure policy within its curated intervention pool/population and reduce item-selection bias; they are not an unbiased full-catalog ranking evaluation, full OPE, or offline A/B test.

### 9. VK-LSVD protocol

Use `up0.01_ir0.01`: history weeks 0–19, train 20–24, validation 25, and untouched test 26.

Compute user/item aggregates and three iALS models on weeks 0–19. Use fixed smoothing `alpha=5`, `beta=100`, and the released `meta.json` duration constant `c`. For mandatory `DCDN-Core` and its baselines, do **not** PLE-encode `P`: fit only positive scale-only normalizers on train weeks 20–24 and freeze them; fit categorical mappings for `C` on train and keep `C` outside the core. Assert that shared raw/scaled `C+P` columns are identical across `C+P` and `C+P+E`. PLE bucket edges are fitted only in a separately manifested optional `DCDN-Hybrid-Prod-L1-PLE8` bridge or exact HEGM-architecture reproduction; old HEGM edges for its 32-column view are not reusable for a rebuilt schema.

Define `C` as the six shared fields `age, gender, geo, place, platform, agent`. Tentative `P` has about 31 numerical base operands before validity indicators; tentative `E` has about 20 engineered numerical divisions. Validate exact fields in the manifest. The numerical operand subspace changes 51→31 (39.2%), while adding six shared context fields gives 57→37 (35.1%) before validity indicators; do not state a final total-input percentage until the manifest is exact.

Use the released `meta.json` constant `c` (approximately 157 seconds, to be verified) defined from item-duration metadata, and set `y=clip(timespent,0,c)/c`. Do not call `c` a watch-time q99. Optimize/select by normalized MSE. Freeze the reporting transform before validation: `t=clip(timespent,0,c)` and `t_hat=c*clip(y_hat,0,1)`; recompute MAE and MSE from `(t_hat,t)` in seconds and seconds². Use one exact global concordance definition: over all unordered pairs with unequal `t`, count matching prediction order as 1, reversed order as 0, prediction ties as 0.5, and exclude target ties; implement it with a tested order-invariant algorithm and report comparable-pair coverage. If reproducing HEGM's batch-dependent implementation, label it `HEGM batch-XAUC` and do not present it as this global concordance.

Retain user ID, item ID, week, and—if released—timestamp; otherwise retain stable in-file row order as the audit time key. For any user-grouped metric, state single-class/group handling and coverage.

### 10. KuaiRec protocol

Use the official big matrix. For the HEGM-compatible primary view, exclude ads and rows with current `play_duration<=0` or `video_duration<=0`, disclose that the estimand is positive-watch interactions, globally sort, then split 80/10/10 by interaction count. Fit ID/category mappings on train and inner-drop validation/test rows with unseen user or item IDs; report row counts before and after. An OOV-retaining alternative is allowed only as a separately frozen modified-protocol sensitivity analysis. Declare in `features_v1.yaml` that train-mapped user/item ID embeddings belong to shared context `C`. Never apply a `.unique(video_id)` join to time-varying daily statistics.

Build as-of features from `item_daily_features.csv`: an event on date `d` may use only daily rows with date `<d`. Name historical play time `hist_play_duration_sum`; current-row `play_duration` and `watch_ratio` remain label-only.

Freeze `C+P` and `E` formulas in the manifest. Use the released `meta.json` constant `c` (approximately 60 seconds, verify) derived from video-duration metadata and label `y=clip(current_play_duration,0,c)/c`. Train and select by normalized MSE. For reporting, use `t=clip(current_play_duration,0,c)` and `t_hat=c*clip(y_hat,0,1)`, then compute MAE/MSE in seconds/seconds² and the same global concordance definition as VK-LSVD. Run the five Tier-A rows and three seeds; do not run the architecture zoo here.

### 11. Yambda protocol (MANDATORY under the 2026-08-20 override; the word "optional" below is superseded)

Do not begin while a mandatory public cell or output is missing. If explicitly reached:

- use Yambda-50M only;
- use the final day as test, the preceding day as validation, a 30-minute gap before validation and another 30-minute gap before test, and all earlier events from the roughly 300-day collection as training;
- restrict to `is_organic=0` recommendation-driven listens;
- define `Listen+ := played_ratio_pct >= 50`;
- call the task completion re-ranking, not CTR or end-to-end retrieval;
- skip audio embeddings unless separately predeclared.

### 12. Launch order and scientific gates

Use one process per GPU and job-level parallelism; do not use DDP.

1. Audit environment, mounts, persistence, budget, deadline, and termination plan.
2. Initialize the persistent repository, lock environment, and pass unit/numerical/resume tests.
3. Build and hash data/feature/protocol manifests without model metrics.
4. Run deterministic tiny and one-epoch validation-only throughput smokes for every required model path.
5. Decide the feasible Tier-B scope from measured throughput; obtain approval for any reduction.
6. Tune every frozen cell on validation only with seed 42; no test access.
7. Hash selected configurations and unlock each dataset's untouched test exactly once.
8. Run Tier A and the approved Tier B scope for seeds 42/43/44, reusing identical seed-42 finals.
9. Complete KuaiRec Tier A, uncertainty, stability, and model-only efficiency tables.
10. Run clean-environment reproduction and durable artifact audit.
11. Only then consider optional lifting, synthetic, secondary-label, or Yambda work.

If `DCNv2(C+P)` is not worse than `DCNv2(C+P+E/R)` by the frozen paired analysis, the substitution premise is not established on that dataset. Report the result; do not hunt for a preferred feature family. If `DCDN(C+P)` does not improve the predeclared paired contrast, report the negative result and only run frozen diagnostics.

A valid negative scientific result counts as a completed cell. Code failure, leakage, OOM, timeout, corrupt data/checkpoint, or missing implementation does not. Required DCNv2/DCDN causal-core runs must succeed technically; an execution failure is a blocker that requires repair or explicit user acceptance of an incomplete table.

### 13. Observability, checkpoints, and stop rules

Every run has an immutable ID and stores resolved config/seed, code commit plus diff hash, data/feature/protocol hashes, reproduction command, histories, runtime, throughput, peak VRAM, parameter/FLOP counts, denominator coverage/clipping/non-finite metrics, and completion/failure reason.

Maintain a machine-readable registry with `pending/running/completed/failed`, PID, GPU, timestamps, heartbeat, and log path. Jobs must survive SSH disconnect via `tmux` or an equivalent local supervisor.

After OOM, allow one microbatch reduction while preserving effective batch through accumulation; a second OOM stops the run. Stop the affected process immediately on leakage, split/schema/order mismatch, mask violation, corrupt checkpoint, inconsistent resume, repeated non-finite values, disk use above 85%, or no progress for 20 minutes. Preserve evidence first. Distinguish autonomously stopping a training process from stopping/terminating the paid Pod, which still requires user approval.

### 14. Required reports and durable backup

Update `STATUS.md` after: audit; bootstrap/tests; each dataset manifest; validation-only tuning; protocol/config freeze; test evaluation; final aggregation; and backup.

Final outputs include per-seed CSV/JSON, mean±SD, paired clustered 95% CIs, primary multiplicity-adjusted comparisons, quality/parameter/FLOP/throughput/VRAM tables, model-only p50/p95/p99 inference latency under a frozen hardware/precision/batch/warmup/repetition protocol, denominator stability slices, failure/negative-result log, and `results/summary.md`.

Create `docs/backup_manifest.json` with persistent paths, sizes, SHA-256 hashes, and artifact class. Reopen and read-test at least the lockfiles, manifests, summary tables, final configs, and one selected checkpoint directly from the persistent mount. Record remaining disk, registry state, active processes, elapsed Pod time, and estimated/projected cost; clearly label control-plane billing data as unavailable rather than blocking or inventing it.

At completion or hard stop, terminate all task-owned training/evaluation processes, verify zero active task PIDs, preserve the durable audit, and tell the user explicitly that the RunPod itself is still billable until they stop/terminate it.

### 15. Verifiable stopping condition

The goal is complete only when:

- the clean-environment smoke reproduction passes;
- all mandatory numerical, leakage, checkpoint, and resume tests pass;
- KuaiRand-Pure and VK-LSVD manifests/protocols pass and their Tier-A core has technically successful three-seed results;
- all five KuaiRec Tier-A rows have technically successful three-seed results, reusing an identical capacity-control row when allowed by the frozen ±5% rule;
- the approved Tier-B scope is complete; any reduction was explicitly accepted by the user before test unlock;
- valid negative findings are reported without changing protocol;
- uncertainty, efficiency, stability, and backup summaries contain no invented values;
- the cost ledger remains within USD 460, the time limit is respected, and no task-owned jobs remain active;
- `STATUS.md` states what is paper-ready, what remains an owner-side industrial blocker or optional experiment, and the exact user action needed to stop billing.

Do not mark the goal complete merely because implementation runs failed. If a mandatory cell remains technically blocked at the hard limit, preserve everything, report the blocker and cost, and request the user's decision; do not fabricate completion.
