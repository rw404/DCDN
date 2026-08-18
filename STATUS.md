# DCDN execution status

Updated: `2026-08-18T21:24:26Z`

## Current checkpoint

- All five mandatory execution files are present and have been read fully.
- `DCDNBlock_provided.py` remains unchanged production provenance for
  `DCDN-Hybrid-Prod-L1-PLE8`.
- `DCDNCore_reference.py` matches the frozen public/core architecture split:
  fixed anchor, residual/product/reciprocal recurrence, named denominator mask,
  and no local FC/BatchNorm/Dropout/concatenation/projector path.
- Persistent workspace and two idle A100-SXM4-80GB GPUs are verified.
- The immutable ledger epoch is recorded as `2026-08-18T09:13:18Z`.
- The owner waived provider-side expiry, credentialed billing inspection, and
  control-plane quota evidence. Frozen accounting is USD 3.29 per elapsed hour
  plus a USD 25 storage reserve.
- Estimated incremental spend at completion of the DCDN tuning pair is USD
  64.54, with USD 305.46 warning-threshold headroom and USD 395.46 hard-cap
  headroom under the owner-approved local accounting basis.
- Test access remains locked. No local test data, labels, DCDN outputs, or
  project-generated metrics were read. A credential-free HEGM provenance
  lookup unexpectedly exposed a published benchmark table from incompatible
  released processed views. The external prior is quarantined in
  `docs/test_lock_audit.yaml` and cannot influence features, search spaces,
  configurations, or protocol decisions.
- Configuration-only inspection of the Wukong paper also displayed adjacent
  published results. Those values are separately quarantined and were not used
  to choose any architecture or candidate.
- The reproducible Python/CUDA environment is locked by `uv.lock`; 101 CPU
  tests, Ruff, environment/cost/storage guards, and independent deterministic
  smokes on both A100s passed. Exact evidence is in
  `docs/preflight_report.md`.
- Bootstrap commits `c82bfbc` and `3152ed0` are durably pushed to
  `rw404/DCDN:experiments/wsdm27`. HTTPS authentication was interactive;
  neither the remote URL nor a credential helper contains the supplied token.
- KuaiRand-Pure, KuaiRec, and the exact VK-LSVD allowlist were downloaded to
  ephemeral storage and byte-verified. Durable manifests contain provenance,
  sizes, hashes, and schema/header boundaries only. VK week 26 remains sealed;
  no source is committed.
- The dataset-agnostic nine-model suite and metric-independent search contract
  are frozen and hashed. Targeted verification passed 35 model tests, 40
  search-lock tests, and 46 acquisition tests; a single combined stable
  `/tmp` snapshot passed all 222 tests plus Ruff/format checks.
- Two acquisition scripts were briefly truncated when concurrent Ruff
  formatters were stopped on FUSE. They were recovered from independent
  `/tmp` snapshots, reviewed, and the full 46-test acquisition suite passed
  against byte-identical copies. Direct concurrent test/format writes on FUSE
  are now prohibited.
- Dataset-specific feature audits completed and passed before any model metric.
  That metric-independent checkpoint itself opened no label-bearing model
  result; subsequent DCDN tuning used only the frozen train/validation splits.
- The VK-LSVD phase-1 contract SHA-256 is
  `380d105c847a73afb4927740e40401124688f895142792309b4fc8267a2a5f2b`.
  Its train/validation views manifest SHA-256 is
  `738d3610fd83a93297d94ccad4c4316dd3ad8b241f8409e84daf5b2322040dbe`.
  It records 12 matched shards (2.44 GB), 7,505,820 train rows, and 1,516,831
  validation rows. Week 26 remained outside the path graph and unopened.
- VK-LSVD runtime/encoder hardening is frozen: encoder-contract SHA-256
  `16fa99db8e92ce4dd55c0bd5e7d7bfd8af8d3d59251cc769b41988e61d414449`
  binds full-scan SHA-256
  `cc7c74b183688c1125cd3e49d32daed2b1e7560d9ee796cfe93de972792ed612`
  and C+P cache-content SHA-256
  `b732b95f80f9ec28f4d9dd5192bf7dba3ed16bb96093bc5add209743edfff590`.
  The combined fresh-`/tmp` runtime/encoder hardening suite passed 64 tests.
- The metric-free VK C+P capacity artifact SHA-256 is
  `d3e65d90fe0184ec48cf15abbe7358f8fd57b4944b5a95f3201b2b4e3fe81565`.
  It mechanically selects two layers, four experts, and rank 12 within the
  frozen ±5% parameter/FLOP rule. The reporting label is
  `DCNv2-capacity-matched`; `dcnv2_capacity_matched_core` is the stable
  runtime key and `DCNv2-capacity-matched-core` is historical provenance only.
- The actual VK C+P premetric Protocol v3 SHA-256 is
  `1919ba14e9a1f881fdf9b45016e16c4d0b92fc71d45944d329822e28816a5a0b`;
  active-registry SHA-256 is
  `2d76236dece1769c08417cc0322f3b3bc51001cfb52ceb945e24fe237ca0fdb4`;
  canonical views-split SHA-256 is
  `f754784d06141daf27aae45191d30d8468720d7b4d9410d7d6470c4632e67e75`.
  Independent views-split, active-state, and exact train/validation
  `RuntimeIdentity` gates passed, as did 70 fresh-`/tmp` Protocol tests.
  There is no selected-config hash, test row count, or test path, and
  `test_unlocked=false`. Capacity resolution and protocol materialization
  opened no data row or label, used no GPU, and computed no model metric.
- The deterministic execution projection keeps frozen
  `effective_batch=4096` but floors the 40,000,000-example hard maximum to
  39,997,440 examples, so execution never exceeds it. Validation runs at the
  first completed batch at or above each declared 4,000,000-example threshold
  (first at 4,001,792), identically for every model. The declared search
  lock/hash remains unchanged; this arithmetic policy triggers no search or
  protocol re-hash.
- The corrected KuaiRand-Pure contract SHA-256 is
  `92a060d34e16d5b3dd2674d9ca75281fac0d20a54adda03c9ecaa206e9e601ef`.
  Its owner-approved embargo excludes exactly 6,639 overlapping earlier-role
  rows and uses the actually available 2022-04-09--2022-04-21 burn-in; the
  phase-1 gate passes while test access remains locked.
- The corrected KuaiRec contract SHA-256 is
  `ab016a7f6d300e7a90de9940891ea974695aa5f61ea85985e239588290522c36`;
  its blind-audit SHA-256 is
  `716e976e0afc642b354c50096ca0d12bdef07e55381b5e91b75cc7c4910f810a`.
  Exactly four partial-null counters and four rates are excluded without
  imputation, approved millisecond-to-second conversion is frozen, and the
  train-only duration constant is 131.684 seconds.
- One isolated `/tmp` data-checkpoint run passed all 46 scoped tests, Ruff,
  formatting, and every SHA-256 sidecar.
- The previous shared-encoder/raw-runtime blocker is resolved and bound to the
  active premetric protocol. The first real validation-only tuning unit has now
  completed for DCDN-Core; every source remained fail-closed on exact
  train/validation runtime identity.
- The raw/composite training harness now binds all nine runtime architectures,
  optimizer/runtime fairness fields, exact capacity candidate/rank, Protocol
  v3 identities, same-FD runtime sources, whole-model checkpoints, numerical
  monitoring, and immutable worker receipts. A fresh independent snapshot
  passed 263 synthetic CPU tests, Ruff, formatting, and compilation. Independent
  review authorizes only two one-effective-batch implementation smoke runs
  after clean Git provenance. Ordinary seed-42 validation tuning may follow a
  successful smoke; capacity-control tuning remains locked until a durable
  selected-DCNv2 learning-rate binding exists, and final/test remains locked.
- The two authorized one-effective-batch implementation smokes completed under
  exact Git commit `38268f305a6c60ae6b8e2fc2f0e62982b03198b1`, empty tracked-diff
  SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
  and Protocol v3 SHA-256
  `1919ba14e9a1f881fdf9b45016e16c4d0b92fc71d45944d329822e28816a5a0b`.
  `vk-cp-smoke-dcdn-core-l2-s42-1eb-v1` ran on physical GPU 0 and
  `vk-cp-smoke-dcnv2-full-rank-l2-s42-1eb-v1` on physical GPU 1. Each
  consumed exactly 4,096 train examples, performed one optimizer step and
  zero validation checks, and stopped at the requested checkpoint boundary.
  Both records are `completed` and explicitly non-scientific; test metrics,
  selected validation metrics, best checkpoints, and latency are all null.
- Both smoke paths reported zero OOMs and zero non-finite observations, with
  all three activation hooks observed. Every config, command, result,
  history, log, registry record, and `last` checkpoint was re-opened and
  hashed; both checkpoints passed integrity-checked CPU-only
  `weights_only=True` loading. The compact audit is
  `results/smoke/implementation_smoke_audit_v1.json`, SHA-256
  `6748d9f643115f0d85b73041b0892a362b1f38efda7c404ca5d329b4ea4650b4`.
  After this verified audit, the two regenerable smoke `last.pt` files and
  their two SHA-256 sidecars were removed at `2026-08-18T19:59:09Z` under the
  retention policy. Their pre-deletion paths, sizes, hashes, safe CPU reopen,
  and sidecar verification remain recorded; retention is now
  `removed_after_verified_audit`.
- The two-candidate DCDN-Core seed-42 validation-only tuning unit is complete.
  Frozen minimization of `normalized_mse` selects `dcdn_core_l2_lr_1e_3`
  (`learning_rate=0.001`) at `0.020977282795389355`, versus
  `0.021020834286119675` for `dcdn_core_l2_lr_3e_4`. The difference is
  `0.000043551490730320` in favor of the selected candidate; no tie occurred
  and no unregistered tie-break was applied. The per-model selection artifact
  SHA-256 is
  `66458ce3d96d6e4f5d5382dab904a5c120134ba52385d8a5bcdbd9d3c2bccb49`.
  `DCNv2-capacity-matched` remains blocked until the selected ordinary-DCNv2
  learning rate is durably bound. Final/test remains blocked because
  `test_unlocked=false`, the dataset-wide selected-config lock is incomplete,
  and the one-pass final/test runner is not yet implemented and audited.
- The current `DCNvsDDN.tex` remains the legacy Russian `DCN vs DDN` draft
  with stale naming and the known two-layer/`FCN` derivation issue. It was not
  changed by the smoke audit and remains a separate manuscript blocker.

## Post-smoke result

**VK IMPLEMENTATION SMOKE PASSED UNDER OWNER OVERRIDE.** At the post-run,
pre-cleanup local guard (`2026-08-18T19:53:13.266664Z` cost;
`2026-08-18T19:53:15.826349Z` storage), the credential-free estimate was
USD 60.09 for 10.665352 elapsed Pod-hours including the fixed USD 25 storage
reserve. The storage guard measured 127,622,144 bytes (0.118857 GiB) of
project-owned durable artifacts, 49.881143 GiB logical headroom under the
frozen 50 GiB envelope, and 62.506607 GiB available on the temporary overlay
at 37.4934% usage. No cost/storage violation or provider action occurred.
At `2026-08-18T19:52:54Z`, both A100s were at 0 MiB and 0% utilization, the
GPU compute-process list was empty, and no task-owned worker/training PID
remained.

The exact four regenerable checkpoint files were absent after cleanup. The
`2026-08-18T20:00:30.583480Z` storage guard passed with 6,381,568 project
bytes (0.005943 GiB), 49.994057 GiB logical headroom, and 62.499523 GiB
temporary-overlay free space. At `2026-08-18T20:01:07.664646Z`, a fresh
read/parse/hash test re-opened all 12 retained per-run config, command, result,
history, log, and registry files and verified eight retained run/protocol
sidecars. All matched their pre-cleanup sizes and SHA-256 values; zero
task-owned training/worker processes remained.

This historical checkpoint validated implementation paths only. The subsequent
DCDN tuning unit described below is now complete. The exact next action is to
continue the remaining eligible seed-42 validation-only tuning units, with test
access still locked. Before the capacity-matched row, durably bind the selected
ordinary-DCNv2 learning rate; before any final/test access, freeze every
selected configuration and exact best-checkpoint lineage, implement/audit the
one-pass final runner, and activate a new protocol revision through the
existing lifecycle gates.

## DCDN validation-only selection

Both frozen DCDN-Core C+P LR candidates completed under Git commit
`b41e1ce9e702b8c9406a4f01042f0d32f91f239c`, Protocol v3 SHA-256
`1919ba14e9a1f881fdf9b45016e16c4d0b92fc71d45944d329822e28816a5a0b`,
seed 42, and the identical 39,997,440-example budget. Each made 9 validation
checks, reported zero OOM and zero non-finite observations, and contains no
test metric. Independent postaudit gave FINAL GO for the strict validation
minimum above.

The selected checkpoint is
`runs/vk-cp-tune-dcdn-core-l2-s42-lr1e3-v1/best-step-000000007813-generation-0000.pt`,
SHA-256
`e12d3b5e769ed3c0a45318e921a2b83a276c9a1ace831cd7575ba93cfebd0f0f`.
All 12 best generations and both `last` checkpoints (14 checkpoint/sidecar
pairs) passed streaming hash verification, strict sidecar validation, safe
CPU-only `weights_only=True` loading, and tensor finiteness checks. The exact
retention inventory is in
`results/tuning/vk_lsvd_c_plus_p_dcdn_core_selection_v1.json`; no checkpoint
has been deleted. It marks only the selected best for retention and identifies
13 redundant pairs as deletion candidates after the parent commits, pushes,
and read-tests the durable evidence.

This is one per-model validation selection, not the final dataset
`selected_configs.lock`. The active protocol remains premetric and
`test_unlocked=false`; no test data, label, or metric was accessed.
