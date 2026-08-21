# DCDN execution status

Updated: `2026-08-19T05:35:14Z`

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
- Estimated spend at the successful ordinary-DCNv2 resume-v2 terminal
  checkpoint is USD 77.59, with USD 292.41 warning-threshold headroom and
  USD 382.41 hard-cap headroom under the owner-approved local accounting
  basis.
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
  passed 263 synthetic CPU tests, Ruff, formatting, and compilation. That
  historical review authorized the two implementation smokes, which later
  completed. After two supervisor/registry `EIO` failures, commit
  `d55d61b3a09c95045872f49893377c01f57c2d45` added isolated bounded registry
  helpers, durable incident receipts, idempotent terminal reconciliation, and
  Linux parent-death supervision; targeted fault injection and the full fresh
  regression passed before resume-v2. Capacity-control tuning now requires a
  validator-owned binding of the selected ordinary-DCNv2 receipt below;
  final/test remains locked.
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
  The ordinary-DCNv2 choice is now frozen below. The compact capacity-LR
  authorization now exists at
  `artifacts/selection/vk_lsvd_c_plus_p_capacity_lr_binding_v1.json`, SHA-256
  `d0ef149274d0f2692acc1f166d72350e44bd8a99331b7ae7f4dbc7d9df1cf1b2`.
  It binds the metric-free rank-12 capacity control to inherited LR 0.0003 and
  preserves the closed test firewall. The schema-v4 capacity-control run is
  now complete and scientifically eligible on validation: rank-12
  `DCNv2-capacity-matched(C+P)` reached normalized MSE
  `0.02100464611993931`. DCDN's selected validation MSE
  `0.020977282795389355` is lower by `0.000027363324549955` (about 0.1303%),
  while ordinary full-rank DCNv2 remains lower than DCDN at
  `0.020908477427111632`. This is validation-only evidence, not a test claim.
  Final/test remains blocked because `test_unlocked=false`, the
  dataset-wide selected-config lock is incomplete, and the one-pass final/test
  runner is not yet implemented and audited.
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
DCDN and ordinary-DCNv2 tuning selections described below are now complete.
The compact authorization now durably binds the selected ordinary-DCNv2
receipt and LR, and the bound schema-v4 capacity-control run has completed.
The exact next action is to continue the remaining eligible seed-42
validation-only tuning units. Before any final/test access, freeze every
selected configuration and exact best-checkpoint lineage, implement/audit the
one-pass final runner, and activate a new protocol revision through the
existing lifecycle gates. Test access remains locked.

## VK-LSVD capacity-control validation result

The mechanically selected rank-12 `DCNv2-capacity-matched(C+P)` control ran
under Git commit `eb5ffb026c62b86378ace59803d4f4a8c5b38c9d`, Protocol v3
SHA-256 `1919ba14e9a1f881fdf9b45016e16c4d0b92fc71d45944d329822e28816a5a0b`,
seed 42, and inherited ordinary-DCNv2 LR 0.0003. It completed 39,997,440
training examples, 9,765 optimizer steps, and nine validation checks with
zero OOM and zero non-finite observations. The immutable result SHA-256 is
`fdcf94786df724aa0a91a525b8e23f296530b8837e2669a2bedfe719bc6d0507`.

The selected validation checkpoint is
`runs/vk-cp-tune-dcnv2-capacity-matched-r12-l2-s42-lr3e4-v1/best-step-000000006836-generation-0000.pt`,
SHA-256
`ee0495ebd57c49fbf4b89d8b15fe0c1ea404ca26578d2b1dd0710c3567dd7984`,
at 28,000,256 examples with normalized MSE `0.02100464611993931`.
The terminal `last.pt` SHA-256 is
`fc4f5d1ef34785e6a908fd928d3d7ea4aa1356bceba0ce7816967d0c9ff5a29e`.
Both checkpoint sidecars pass, both payloads reopen with
`weights_only=True` on CPU, all 181 tensors / 7,590,647 elements are finite,
and both carry the exact capacity authorization SHA-256
`d0ef149274d0f2692acc1f166d72350e44bd8a99331b7ae7f4dbc7d9df1cf1b2`.
The terminal registry is `completed` with
`scientific_result_eligible=true`; test remains locked and no test evaluation
was performed.

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

## Ordinary DCNv2 validation-only status

The frozen LR=0.001 candidate
`vk-cp-tune-dcnv2-full-rank-l2-s42-lr1e3-v1` completed as scientifically
eligible validation-only tuning. It minimized normalized MSE to
`0.02091150474550729` after 28,000,256 examples and seven validation checks;
it stopped by validation patience with zero OOM/non-finite events and
`test_metrics=null`. Resolved-config SHA-256 is
`7e5450a2ca7d0ef49ab3bec08b4c36348e4429d34990d7dc104a0492175b98c9`,
immutable result SHA-256 is
`51a39053240683b8b8c90688844b2a4a544b678db71dd2559e61eefec8fe1e23`,
and its within-candidate best-checkpoint SHA-256 is
`166d3f65b65e9332fddc62a102a21f2358c9fca96606bafad9065f06659317d9`.

The frozen LR=0.0003 candidate originally suffered two supervisor/registry
`EIO` execution failures after durable 12,001,280- and 24,002,560-example
boundaries. Those two attempts remain scientifically ineligible and have no
result JSON. Their immutable failure snapshot SHA-256 values are
`82a7858c1f8eed49c3c4f958a58907166faaf19dee4082903276329cdd1bfab4`
and
`a42d29bf579656dae1fc5eab06f2046bea60e4c5ff8e37e69049740f3f28c5ef`.
After the hardening commit above, exact resume-v2 continued from checkpoint
SHA-256
`9861dc7ee7cf09eafdd36cb7d0e329152634fd6b326db3db176220986e8aff25`
at 24,002,560 examples and completed the same candidate at the frozen
39,997,440-example floor. It made nine validation checks, reported zero OOM
and zero non-finite observations, and contains `test_metrics=null`. The
resume-v2 resolved-config SHA-256 is
`1f1c3e7167394cc1726cdd0e1a9d00a1f5e01e27711684065ecee65a6adae454`
and immutable result SHA-256 is
`0e6381d016a65a48d9f12f48b6c871ae242139b2e9a2341f1ecba3efd03cf58e`.

Strict minimization of frozen validation `normalized_mse` selects
`dcnv2_full_rank_lr_3e_4` (`learning_rate=0.0003`) at
`0.020908477427111632`, versus `0.02091150474550729` for LR=0.001. The
absolute improvement is `0.000003027318395658`; no tie occurred and no
post-metric tie-break was invented. The canonical per-model selection artifact
is `results/tuning/vk_lsvd_c_plus_p_dcnv2_full_rank_selection_v1.json`,
SHA-256
`869cbcb7a227b274700cda142f23c59040497bfdfe4da101888e2cc63f39d67a`.
It freezes candidate ID, LR, the seed-invariant model-config SHA-256
`25a5e2b040fe5ff52981a8499d7d4f8dd453bdd91d24e0d5a2b40b49d7521dcc`,
and the selected best checkpoint
`runs/vk-cp-tune-dcnv2-full-rank-l2-s42-lr3e4-v1/best-step-000000006836-generation-0000.pt`,
SHA-256
`f3a27d6522df474f1cea06abe302d289ff9fc331e9ae7135baa9589a3427618c`.
The selected best passed strict sidecar, CPU-only `weights_only=True`, and
tensor-finiteness checks. All 10 current DCNv2 checkpoint/sidecar pairs were
inventoried; nine pairs are proposed deletion candidates only after durable
commit/push/read-test prerequisites. No checkpoint was deleted.

Two heartbeat-helper timeouts during resume-v2 recovered within the fixed
degraded window. Their local and persistent incident receipts are byte-equal,
with SHA-256 values
`5bc1a25000ba591c322007f56c48705aca52a60e257c22199a765ed1d06891c0`
and
`87d24564d6ebae0eaf5b3ca5977e1f521c6be6265851849b64da3c6c0c9bb1da`.
The terminal receipt's local guard at `2026-08-19T01:12:20.526933Z` estimated
cumulative spend at USD 77.59 (15.984035 hours at USD 3.29/hour plus the
fixed USD 25 storage reserve). It measured 1,461,672,448 project bytes
(about 1.361 GiB) and 62.267612 GiB overlay space available. Independent
terminal audit found zero task-owned training/worker PIDs, zero running
registry records, and both A100s idle.

This is still one per-model validation selection, not the final dataset
`selected_configs.lock`. Its exact receipt and LR are now bound by compact
authorization SHA-256
`d0ef149274d0f2692acc1f166d72350e44bd8a99331b7ae7f4dbc7d9df1cf1b2`.
Capacity-control configuration/prelaunch evidence is the remaining run gate.
Final/test access remains forbidden: the active protocol has
`test_unlocked=false` and no test path.

## AFN+ validation-only selection

Both frozen seed-42 AFN+ C+P learning-rate candidates completed under the
same Protocol v3 and validation-only split. LR=0.0003 stopped by validation
patience after 36,003,840 examples, 8,790 optimizer steps, and nine validation
checks; its minimum normalized MSE is `0.0208063458817398`. LR=0.001 stopped
after 24,002,560 examples, 5,860 steps, and six validation checks; its minimum
is `0.02082650517903153`. Both runs recorded zero OOM and zero non-finite
observations, fully drained asynchronous numerical monitors, and all three
activation hooks.

The original supervisors produced complete immutable results and checkpoints
but then recorded a technical failure because the old generic artifact reader
incorrectly capped checkpoint verification at 64 MiB. Commit
`f450c67efdce86d5dc788c79f4ccbaf173501ca7` added streaming checkpoint
verification without relaxing the bounded JSON/YAML reader. Its exact,
fresh-only reconciliation path re-opened each result and its last/best
checkpoint twice, preserved the full prior failed registry record and SHA-256,
and changed only the terminal registry state. The reconciled registry SHA-256
values are `045ec6196e1132d5b9eb694d0e8b089421fcc48cf2006e7efa7235f78ce94fc8`
for LR=0.0003 and
`bc78b4ed089267f4aab7d2137663867dca93f7ffd45431316da638fd4f3ccd8d`
for LR=0.001; both are now `completed` and
`scientific_result_eligible=true`.

Strict minimization therefore selects `afn_plus_lr_3e_4`, LR=0.0003. It is
lower by `0.00002015929729173166` (about `0.096796%`) than LR=0.001; no tie
occurred and no post-metric tie-break was invented. The canonical per-model
selection receipt is
`results/tuning/vk_lsvd_c_plus_p_afn_plus_selection_v1.json`, SHA-256
`69463d2687d46520ea270ffcdfeaaa8ef47cd11173dc3943780cc4cc486d7e54`.
The retained selected checkpoint is
`runs/vk-cp-tune-afn-plus-s42-lr3e4-v1/best-step-000000005860-generation-0000.pt`,
SHA-256
`42dd6b18e0cfb51769e81718a86163d09057546b5cf00f7b10f8ad5a7d42dc8d`.

All ten AFN checkpoint/sidecar pairs (792,256,766 checkpoint bytes) passed
strict streaming hashes, safe CPU `weights_only=True` reopen, accumulation-
boundary, metadata-lineage, and tensor-finiteness checks. Nine redundant
pairs remain present and are only proposed for deletion after the complete
AFN evidence is committed, pushed, and independently read-tested. This is one
per-model validation selection, not the dataset-wide `selected_configs.lock`.
The Protocol remains `test_unlocked=false` and `test_path_available=false`;
the selection histories contain no test event and no test metric was
consulted.

## VK-LSVD C+P+E premetric checkpoint

A separate Protocol v3 is frozen and active for the already verified
`C+P+E` feature cache. Frozen protocol SHA-256 is
`6a6e33c908d666c4374ac5052a207c38a9fe012dac147a5e572cb819013a5739`
and active-registry SHA-256 is
`7054e2e493afa49dafab7561ed6b7909e8ef56741976570915aff0bad9c1cce3`.
The protocol permits only ordinary full-rank DCNv2, so it cannot be reused
for DCDN or another baseline. Its active stage is `premetric_locked`;
`test_unlocked=false` and `test_path_available=false`.

The runtime CLI and supervisor now bind dataset, feature regime, required
model, protocol digest, cache identities, and result payload before and after
execution. A repository test fixes the exact protocol digest and active
test-lock state. At `2026-08-19T10:12:29.624367Z`, the local mandatory
3-hour guard estimated USD 107.21 cumulative and USD 117.08 projected, well
below the USD 460 cap and before the finalization reserve. Storage measured
9,519,765,504 project bytes (8.866 GiB), 41.134 GiB logical headroom, and
62.000 GiB overlay space available. Both A100s were idle at 0 MiB/0%.
The next action is a clean evidence commit, followed by exact materialization
and independent prelaunch checks for the two frozen seed-42 validation-only
DCNv2 learning-rate candidates. No C+P+E validation or test metric has been
read.

### First C+P+E launch: technical pre-training failure

Both initially materialized candidates failed before the first training batch.
The common harness attempted to compare the frozen C+P capacity-control
projection against a mechanical projection built from the wider C+P+E
encoder. Both histories contain exactly one `local_guard` event; neither run
created a checkpoint or result, and no validation/test event occurred. Their
registry records are terminal `failed` with
`scientific_result_eligible=null`. Immutable paired failure receipt SHA-256 is
`d60f40816564427f41cf0e9748fc4d76f3fd6e613aab73d781630d9c2d6c0bee`.

The narrow correction keeps every artifact, digest, repository, C+P
mechanical-capacity, and model-key gate intact. It only prevents the C+P
capacity projection from being recomputed on C+P+E, which is allowed solely
for ordinary full-rank DCNv2. Focused training-harness regression passed
74/74. At `2026-08-19T10:40:53.693746Z`, estimated cumulative spend was USD
108.76; both A100s were idle at 0 MiB/0%, and no task process remained. The
original run IDs are immutable failures. A fresh full CPU regression
passed 541/541 tests; Ruff check/format and targeted bytecode compilation also
passed. The next action is the clean fix/evidence commit and push, then fresh
candidate IDs with the exact same frozen learning rates and budgets. Test
remains locked.

### First retry: second technical pre-training failure

The retry pair under clean commit
`871b82fb8c817fd2f8739ec2ae38f00b2fbaf92a` passed the corrected projection
gate but stopped before the first batch at a second C+P-only assumption:
`composite non-embedding count differs from capacity artifact`. The wider
C+P+E full-rank model was still being required to equal the C+P-only ordinary
DCNv2 parameter count. Both histories again contain only `local_guard`; no
checkpoint, result, validation, or test event exists. Both records are
terminal `failed` with `scientific_result_eligible=null`. Immutable paired
failure receipt SHA-256 is
`91a5011e5bfb7955e838f3256fac320c13b6a7e2a212ff07063a52760e0ebe17`.

The second narrow correction keeps exact parameter-count binding unchanged
for every C+P model and returns no C+P capacity-count expectation only for the
already isolated C+P+E full-rank DCNv2 path. Focused harness regression passed
74/74; the fresh full CPU suite passed 541/541, with scoped Ruff, format, and
bytecode compilation clean. At
`2026-08-19T11:08:25.444318Z`, estimated cumulative spend was USD 110.27;
storage remained healthy at 9,520,076,800 project bytes, both A100s were idle,
and no task process remained. Before another retry: commit/push the fix and
evidence, then materialize fresh IDs. Test remains locked.

### C+P+E DCNv2 validation-only selection

The second retry pair under clean commit
`c8d82f8f9edfcbbed5d78fea18bc797484cdad8a` completed the frozen seed-42
validation-only tuning unit. LR=0.0003 consumed the full realized budget of
39,997,440 examples, made 9,765 optimizer steps and nine validation checks,
and reached normalized MSE `0.020849287259723712`. Its immutable result
SHA-256 is
`b46d2129618be37c7c318ecd6aa99e0507b1d02c6ebe00c05fd3f388499b1232`.

The LR=0.001 attempt suffered a checkpoint-file `fsync` `EIO` after its
validation split for check four had been computed but before a durable
checkpoint/result could be published. That attempt remains a technical
failure, and the incomplete check-four observation was excluded from
selection. Failure receipt SHA-256 is
`be53192dcff1e57a0b5f5a0808c9aa3a40cd6b109454e3d7349c43340d1e26fc`.
Exact resume from the last accumulation-boundary checkpoint at 12,001,280
examples recomputed check four identically, then completed 39,997,440
examples, 9,765 steps, and nine validation checks. Its immutable result
SHA-256 is
`7ec7c16b5bb5f6c345dc7bd35953e43ce4ea2a072b7acf8544228171600ce3ef`.

Strict minimization selects `dcnv2_full_rank_lr_1e_3` (LR=0.001) at
`0.020826933840839952`. It is lower than LR=0.0003 by
`0.000022353418883760273` (about `0.107214%`); there is no tie. Both terminal
results are `scientific_result_eligible=true`, with zero OOM and non-finite
observations and `test_metrics=null`. Both candidate best/last checkpoints
passed strict sidecar checks and safe CPU `weights_only=True` finite-tensor
reopen. The selected best checkpoint is
`runs/vk-cpe-tune-dcnv2-full-rank-l2-s42-lr1e3-v1-retry-v2/best-step-000000007813-generation-0000.pt`,
SHA-256
`858fec98f915cdec9dea8dac0aebd1156f06d36eabd966c9aa4fc522c015d23f`.

Canonical selection receipt
`results/tuning/vk_lsvd_c_plus_p_plus_e_dcnv2_full_rank_selection_v1.json`
has SHA-256
`85ab7ccbb8d74652f3aa3416a9ada872b36ed2de24c7b8341b56b9b8633f5c41`.
All 14 current checkpoint/sidecar pairs remain intact; no cleanup is allowed
until this evidence is committed, pushed, and independently read-tested.
This freezes only the C+P+E control LR. It does not answer RQ1 and does not
unlock test: `test_unlocked=false`, `test_path_available=false`. The RQ1
README verdict will be written only after the frozen seed-42/43/44 test
comparisons, paired uncertainty, and effect-size analysis are complete.

### VK-LSVD C+P DNN validation-only selection

Both frozen seed-42 DNN learning-rate candidates completed under clean commit
`46add65bea8134d9f1a65373a82d8bcfb5b6579b`. LR=0.0003 stopped by the
predeclared validation-patience rule after 28,000,256 examples, 6,836 optimizer
steps, and seven validation checks; its minimum normalized MSE is
`0.020930008215596697`. LR=0.001 stopped by the same rule after 32,002,048
examples, 7,813 steps, and eight checks; its minimum is
`0.020899830835974933`. Both results are
`scientific_result_eligible=true`, have zero OOM and non-finite observations,
and contain `test_metrics=null`.

Strict validation minimization therefore selects `dnn_lr_1e_3` (LR=0.001),
lower than LR=0.0003 by `0.000030177379621763867` (about `0.144182%`); no tie
or post-metric tie-break occurred. The canonical per-model selection receipt
is `results/tuning/vk_lsvd_c_plus_p_dnn_selection_v1.json`, SHA-256
`7c851bfb7ca37e822fe05faca78a6488f56c447c995286143379a3611a166204`.
The selected best checkpoint is
`runs/vk-cp-tune-dnn-s42-lr1e3-v1/best-step-000000004883-generation-0000.pt`,
SHA-256
`b9a98fe1fd7eff8963b00fe887b35ce73d2c37c4885b97a08e2f8bc674cf82f1`.
All ten DNN checkpoint/sidecar pairs were independently hash-checked and
reopened with `weights_only=True` on CPU; every tensor is finite. No
checkpoint was deleted.

Each run recovered from one bounded registry-heartbeat helper timeout without
interrupting training; the immutable content-addressed incident receipts have
SHA-256 `f24ffdf95cd57cebf8500fe2fe6b04ad382e22930d56a0c607e794ed51e8e7ed`
and `0d31cf173c31a7e29bb17e68da9b2643755135a6d43b372b33dccf7af92fd4ee`.
At the post-selection preflight, estimated spend was USD 121.98, project
storage was 10.2504 GiB with 39.7496 GiB logical headroom, both A100s were
idle, and no task-owned run remained active.

This is a per-model validation choice, not the dataset-wide
`selected_configs.lock`. It also does not establish DCDN superiority: the
selected DNN validation MSE is lower than selected DCDN by
`0.000077451959414422` (about `0.369218%`). That comparison remains
validation-only and descriptive; final claims wait for the untouched test,
paired uncertainty, and the complete frozen RQ1/RQ2 matrix. Test remains locked.

## VK-LSVD C+P DeepFM validation-only selection

Both frozen seed-42 DeepFM candidates completed under clean commit `cd7d54ca13dd57c21821d704f44090bfc2969c47` with zero OOM/non-finite observations and `test_metrics=null`.
LR=0.0003 reached normalized MSE `0.020845575060232564` after 24,002,560 examples and six checks; LR=0.001 reached `0.020876811708431246` after 32,002,048 examples and eight checks.
Strict minimization selects `deepfm_lr_3e_4`, lower by `0.00003123664819868269` (about 0.149624%), with no tie.
Receipt SHA-256 is `a6ed26dcce171cf8c27cb00e2db57c90cfcead0481251ed8e4a3a830ff812aeb`; selected best-checkpoint SHA-256 is `f2ec954cc34ed8d6e2bb650899e0a0405dcf11926300a61a97068f0a82d8ecc3`.
All eight checkpoint pairs passed strict sidecar, CPU `weights_only`, metadata, and finite-tensor checks; two recovered registry-helper timeout receipts are preserved and no checkpoint was deleted.
The next frozen C+P model is DCN. Dataset-wide selected configs and final/test remain locked.

## VK-LSVD C+P Original DCN validation-only selection

Both frozen seed-42 Original DCN candidates completed under clean commit
`db4447aecf0d71f3bb120411683b6e25f4fa04de` with zero OOM/non-finite
observations and `test_metrics=null`. LR=0.0003 consumed 39,997,440 examples
and nine checks, reaching normalized MSE `0.021035584402689534`; LR=0.001
stopped by validation patience after 24,002,560 examples and six checks at
`0.02105195259385249`. Strict minimization selects `dcn_lr_3e_4`, lower by
`0.000016368191162956236` (about 0.077751%), with no tie.

Canonical selection receipt
`results/tuning/vk_lsvd_c_plus_p_dcn_selection_v1.json` has SHA-256
`f352c9d07a9690303d7d0fc5986147310d0f83d9185599b02fcdb58a86e50656`;
selected best-checkpoint SHA-256 is
`38cd618114fbd2a74355932cf3b1a311dfa01354e105b168ba67cd99a44fd615`.
All 11 checkpoint pairs passed strict sidecars, CPU `weights_only`, metadata,
and finite-tensor checks. Three recovered registry-helper incidents are
preserved; no checkpoint was deleted. Both registries are completed and
scientifically eligible, both A100s are idle, and no task process remains.

On matched seed-42 validation, selected DCDN (`0.020977282795389355`) is
lower than selected Original DCN by `0.00005830160730017908`, or about
0.277157% relative to DCN. README records this as support for the narrow
validation hypothesis only; the final multi-seed test RQ1 verdict remains
locked. The next frozen `C+P` model is AutoInt+.

## VK-LSVD C+P AutoInt+ validation-only selection

Both frozen seed-42 AutoInt+ candidates completed under clean commit
`eb4e826d9c1e4d4ddf2a6c0ee9423b0bce5c5b41` with zero OOM/non-finite
observations and `test_metrics=null`. LR=0.0003 consumed 39,997,440 examples
and nine validation checks, reaching normalized MSE
`0.020879945407626305`; LR=0.001 stopped by the predeclared patience rule
after 32,002,048 examples and eight checks at `0.02105809059370641`. Strict
minimization selects `autoint_plus_lr_3e_4`, lower by
`0.0001781451860801038` (about `0.845970%`), with no tie or post-metric
tie-break.

Canonical selection receipt
`results/tuning/vk_lsvd_c_plus_p_autoint_plus_selection_v1.json` has SHA-256
`45ab5878b7a7ad5b57be418e6d17b5583b7cc84130fbef3a856af275c82169bb`;
selected best-checkpoint SHA-256 is
`73f5b18db38d19020f550c99f639694a7ccd3d39a296b6bfdb026c783f54397f`.
All 12 checkpoint pairs passed strict sidecars, CPU `weights_only`, metadata,
and finite-tensor checks. No checkpoint was deleted. Both registries are
completed and scientifically eligible, both A100s are idle, and no task
process remains.

On matched seed-42 validation, selected AutoInt+ is lower than selected DCDN
by `0.00009733738776305026`, or about `0.464013%` relative to DCDN. This does
not invalidate the narrower DCDN-versus-Original-DCN validation result, but it
does reject any current claim of universal DCDN superiority. README records
both facts. The next frozen `C+P` model is Wukong-small; dataset-wide selected
configs and final/test remain locked.

## VK-LSVD C+P Wukong-small fixed validation-only result

The single Wukong-small configuration was frozen as `fixed_no_tune` before
metrics: two interaction layers, LCB/FMB widths 8, FMB rank 24, three
2048-wide FMB hidden layers, and LR=0.0003. Under clean commit
`f008665788983dd34b10d5b98787745a638582ee`, it stopped by the predeclared
validation-patience rule after 24,002,560 examples, 5,860 optimizer steps,
and six validation checks. Its minimum normalized MSE is
`0.02096245653858564` at check three. The result is
`scientific_result_eligible=true`, with zero OOM/non-finite observations and
`test_metrics=null`.

Canonical fixed-candidate receipt
`results/tuning/vk_lsvd_c_plus_p_wukong_small_selection_v1.json` has SHA-256
`4f671539bd70d4054e9573c6d85f1cd991046e1980204fa78352c2cbd660cb16`;
selected best-checkpoint SHA-256 is
`a28af04fb609cd698903562cb6c8b7b4c5b023e40b958d4afa4b776182051f18`.
All four checkpoint pairs passed strict sidecars, CPU `weights_only`,
metadata, and finite-tensor checks. One recovered registry-helper incident is
preserved; no checkpoint was deleted. Registry is completed and scientifically
eligible, both A100s are idle, and no task process remains.

On matched validation, Wukong-small is lower than selected DCDN by
`0.000014826256803716215` (about `0.070678%`) but higher than selected
AutoInt+ by `0.00008251113095933404` (AutoInt+ is about `0.393614%` lower
relative to Wukong). The seed-42 `C+P` per-model selection suite is now
complete. The next frozen action is to build and audit the dataset-wide
`selected_configs.lock` and final runner; final/test remains locked.

The owner-approved multi-task extension is scheduled only after the mandatory frozen matrix: normalized watch time remains primary, while like/dislike/share/bookmark are auxiliary binary labels when raw schema validation permits them.
Per target, report ROC-AUC and user-GAUC over users with both classes, plus eligible-user/event coverage and paired user-bootstrap 95% confidence intervals.
This extension does not alter the current single-target RQ1 selections or expose test data.

## 2026-08-20 VK-LSVD C+P three-seed RQ1 validation checkpoint

All nine frozen `C+P` model rows now have scientifically eligible
validation-only results for seeds 42, 43, and 44. Canonical compact summary
`results/final_validation/vk_lsvd_c_plus_p_rq1_validation_summary_v1.json`
has SHA-256
`b5c64f84e6562add812a1fa5d3db2cbbf1ed712ff6e2237c59217dd6047fc070`;
its strict adjacent sidecar and all 27 referenced source-artifact hashes
reopened successfully.

The narrow RQ1 comparison is positive on validation. `DCDN-Core-L2` is lower
than Original DCN on every seed: the paired improvements are
`0.000058301607300179`, `0.000027781149600347`, and
`0.000038721328376512`, with mean `0.000041601361759013` (about
`0.197739%` relative to the DCN mean). DCDN is also lower than the rank-12
`DCNv2-capacity-matched` control on every seed, with paired mean improvement
`0.000028828551562561` (about `0.137111%`). The three-seed means are
`0.020996908083702307` for DCDN, `0.021025736635264868` for the
capacity-matched control, and `0.021038509445461321` for Original DCN.

Universal DCDN superiority is not supported: full-rank DCNv2 and five other
baselines have lower three-seed means. This is recorded as a valid,
validation-only scientific limitation rather than hidden or reframed. The
failed seed-43 capacity attempt ended in registry/FUSE infrastructure I/O
before any validation metric; incident SHA-256 is
`f26204b8a05cd5d80f0db0359a080b4c6aae7bf0e311c62b549f5ac0f1e53847`.
Its fresh run `vk-cp-final-dcnv2-capacity-matched-r12-l2-s43-v1-rebind-v2`
completed with result SHA-256
`ac97f0a1d7fc090ce714d3dd0c0f9f236815604d8941e1308f9ee8dd29de0c9a`;
seed 44 completed with result SHA-256
`1c98d5c66625a5b6080a184296e53c3042eef88118bbc977a30aaf463eeddcf7`.

At `2026-08-20T12:23:12.516334Z`, estimated cumulative spend was USD 193.33.
Storage passed at 23,639,039,488 project bytes (22.0156 GiB), with 27.9844 GiB
logical headroom and 60.7022 GiB overlay available. Both A100s were idle,
registry `running=[]`, and no task-owned worker, trainer, registry helper, or
supervised process remained. Test stays sealed:
`test_unlocked=false`, `test_path_available=false`, and no test metric was
accessed. Next freeze the dataset-wide selected-config lineage and audit the
one-pass evaluation bundle before any test activation.

## 2026-08-20 VK-LSVD C+P dataset-wide selected-config freeze

The immutable validation-only
`configs/protocols/vk_lsvd_c_plus_p/selected_configs.lock` now binds every
frozen model and all seed-42/43/44 resolved run configs, validation results,
selected metrics, selection receipts, and exact best-checkpoint SHA-256 values.
Its SHA-256 is
`165a82204a039592c140e6be44a3245d1b62ce084946e1a45ac698229d270cfd`;
the strict adjacent sidecar was reopened, and an independent re-derivation from
all 27 validation lineages reproduced the same payload. The full CPU regression
passed 547/547; Ruff, formatting, bytecode compilation, and diff checks pass.

At `2026-08-20T12:58:08Z`, estimated cumulative spend was USD 195.25.
Project storage was 23,633,846,765 bytes (22.0107 GiB), leaving 27.9893 GiB of
logical headroom and 60.6978 GiB on the overlay. Both A100s were idle and no
task-owned process remained. This freeze does not activate final evaluation:
the parent protocol and active state still record `test_unlocked=false` and
`test_path_available=false`. The next gate is the audited one-pass
metric/uncertainty runner and controlled test-view transition; no test file,
label, or metric was opened here.
## 2026-08-20 one-pass final evaluation core checkpoint

The dataset-independent one-pass final evaluator is now implemented without
opening VK-LSVD week 26. It binds the exact selected seed/checkpoint claim,
checks checkpoint bytes before the mutation-safe parent-best restore, moves the
claim to `opened` immediately before the sole source factory call, keeps audit
IDs outside the model signature, consumes every row once in order, drains the
numerical monitor, and publishes an immutable canonical JSON/strict-sidecar
bundle. The bundle contains normalized MSE, seconds-scale MAE/MSE, exact global
concordance coverage, row-alignment digests, and privacy-safe user-cluster
sufficient statistics. An `opened` crash window can only reconcile the already
published verified bundle; it cannot reopen test. Rehashed metric tampering is
rejected against the cluster sums.

The full CPU suite passes 553/553; the focused final/protocol lifecycle passes
77/77, and Ruff, formatting, bytecode compilation, and diff checks pass. At
`2026-08-20T13:38:59.989945Z`, estimated spend was USD 197.49 (USD 200.78 with
a one-hour compute ETA). Storage was 23,639,561,216 bytes (22.0161 GiB), with
27.9839 GiB logical headroom and 60.6967 GiB overlay available. Both A100s were
idle, registry `running=[]`, and no task-owned process remained.

This checkpoint does not activate test. The active protocol still records
`test_unlocked=false` and `test_path_available=false`; no week-26 file, label,
or metric was opened. Remaining gates are the controlled week-26 source/view
transition, the paired 2,000-replicate user-cluster contrast/Holm aggregator,
and final-child protocol materialization/audit.

## 2026-08-20 frozen RQ1 paired aggregation checkpoint

The final RQ1 aggregator is implemented and frozen before test access. It
requires exactly 12 immutable one-pass bundles: DCDN(C+P), full-rank DCNv2(C+P),
full-rank DCNv2(C+P+E), and DCNv2-capacity-matched(C+P), each for seeds 42, 43,
and 44. It rejects mixed dataset/protocol/selected-config lineage, non-test
bundles, digest drift, missing cells, or any mismatch in stable rows,
normalized targets, user ordering, clusters, or per-cluster row counts.

For each of the four predeclared normalized-MSE contrasts, the reported effect
is treatment minus baseline (negative favors treatment because the metric is
minimized). All three seeds remain together within every resampled user
cluster. The report includes per-seed effects, seed mean and sample standard
deviation, a 95% percentile interval from 2,000 user-cluster replicates with
seed 20260818, and the pre-test two-sided plus-one empirical-tail p-value.
Holm--Bonferroni adjustment is applied across the exact four-contrast family at
family-wise alpha 0.05. Output publication is exclusive canonical JSON plus a
strict SHA sidecar and binds every input bundle, protocol, selected-config
lock, and checkpoint digest.

Focused final/statistics tests pass 19/19 and the full CPU suite passes 562/562;
Ruff, formatting, bytecode compilation, and diff checks pass. At
`2026-08-20T14:03:15.587098Z`, estimated spend was USD 198.82 (USD 202.11
with a one-hour mandatory-compute projection). Storage passed at
23,639,831,040 bytes (22.0163 GiB), with 27.9837 GiB logical headroom and
60.7017 GiB overlay available. No worker/trainer/helper process or GPU compute
process was present.

This checkpoint contains no test result. VK week 26 remains unavailable and
the active protocol remains `test_unlocked=false` /
`test_path_available=false`. Remaining gates are the controlled test-view
transition and audited final-child protocol activation.

## 2026-08-20 VK-LSVD C+P+E three-seed validation lineage

The selected seed-42 full-rank DCNv2 control remains LR=0.001 at normalized
MSE `0.020826933840839952`. Under clean commit
`9fbc43d717577cf0be772b5bc52b7fb8e28ae636`, seed 43 completed at
`0.020804612845465798` and seed 44 completed at `0.02083825837621076`.
Both terminal results are scientifically eligible, have zero OOM/non-finite
observations, and contain `test_metrics=null`. Their result SHA-256 values are
`79a0c2add20f4f881bdff68636b5cf1ed94bc16b060f2002d61ac718bda14d43`
and `61b0c5ff6766dd2c1ea2cd650ad3237c6cf4105f083f823213af54d25c7038b1`.
All result/config/command and best/last sidecars passed; four checkpoints
reopened on CPU with finite tensors and exact Protocol lineage.

At `2026-08-20T15:57:09Z`, estimated cumulative spend was USD 204.84.
Storage passed at 22.698 GiB of the 40 GiB cap with 27.302 GiB logical
headroom and 60.378 GiB overlay available. Both A100s were idle, registry
`running=[]`, and no task process remained. Test remains sealed. Next create
and independently audit the immutable C+P+E seed-42/43/44 selected-config
lock before controlled test-view and final-child activation.

## 2026-08-20 VK-LSVD C+P+E selected-config freeze

Immutable `configs/protocols/vk_lsvd_c_plus_p_plus_e/selected_configs.lock`
was published with SHA-256
`b038eee038f925524eb21cc89e495835fb12e347a2bc5afa8c2b5851748cc250`.
It binds exactly one selected model (`dcnv2_full_rank_lr_1e_3`) and complete
seed-42/43/44 validation/config/checkpoint lineage. Strict reopen and
independent derivation match; the full CPU suite passes 564/564. At
`2026-08-20T16:17:14Z`, estimated spend was USD 206.17 and storage was
22.699 GiB with 27.301 GiB logical headroom. Test remains locked; the next
gate is controlled test-view and final-child Protocol activation.

## 2026-08-20 Owner override: multi-target reset and legacy audit

An owner override was recorded on 2026-08-20 and is now the governing
protocol. It supersedes the pre-override single-target framing everywhere it
conflicts.

**What changed.**

1. The public headline study is **multi-target**. Single-target training
   becomes a separate follow-up that may not enter the main table and may not
   inform target, feature-family, loss-weight, or model selection.
2. **VK-LSVD, Yambda-50M, KuaiRand-Pure, and KuaiRec big matrix are all
   mandatory.** Yambda-50M is no longer optional.
3. The reduced feature view removes **only** manually materialized divisions
   of historical counters and sums. Cosine similarities, dot products, vector
   norms, and iALS scores stay in both regimes, carry reciprocal mask `false`,
   and are never counted as removed ratio features.
4. The public removal claim is restricted accordingly. The 40.1% production
   reduction remains a separate industrial fact.
5. Absence of RunPod provider-side expiry, live billing API, or credentialed
   inspection is explicitly not a blocker. No stored RunPod credential is
   used; the local ledger continues per AGENTS.md.

**Protocol conflicts found and resolved in favour of the override.**

| # | Conflict | Resolution |
|---|---|---|
| 1 | `configs/experiment_matrix_v1.yaml` declares `primary_targets_only: true` and forbids secondary targets | superseded by `configs/matrix_multitask_v1.yaml` |
| 2 | `src/dcdn/training.py` supports one target, one loss, no label masks | new `src/dcdn/multitask.py` implements the masked multi-target objective |
| 3 | the three iALS cosines sat in the removable `E` set | moved to `P_shared`; their lineage is `dot/(norm*norm)`, not a counter/sum division |
| 4 | all six `p_ials_*_norm` fields sat in `denominator_contract.eligible_fields` | removed from `D`; cosine/dot/norm carry reciprocal mask `false` |
| 5 | AGENTS.md froze Yambda-50M as optional and forbade scope expansion | AGENTS.md amended with the override, pre-override text retained for provenance |

Conflicts 3 and 4 change `features_sha256` and `denominator_mask_sha256` for
VK-LSVD, so every pre-override VK manifest hash is superseded and the runtime
views must be rebuilt before any multi-target VK cell is executed.

**Legacy audit.** `docs/LEGACY_RESULTS_AUDIT.md` plus the machine-readable
`artifacts/legacy_audit/legacy_results_inventory_v1.json` (+ `.csv`, `.sha256`),
generated by `scripts/audit_legacy_results.py`. All 61 registry records were
classified against the eight-part reuse gate:

| classification | count |
|---|---:|
| `legacy_exact_reuse` | 0 |
| `legacy_reproduced` | 0 |
| `legacy_context_only` | 40 |
| `unverifiable` (all `status=failed`) | 21 |

Every record is single-target; none carries a test metric. **No legacy result
enters the new main table and none informs any selection.** The 40 completed
runs are retained as engineering context and as the source of the wall-clock
cost model (≈4,200–5,200 s per 40M-example single-target A100 run). The audit
opened no dataset row, no label, and no test metric.

**New frozen artifacts.**

- `configs/targets_multitask_v1.yaml` — per-dataset heads, Macro-NLoss,
  mandatory per-target label masks, frozen train-estimated task constants,
  explicit prohibition of adaptive/learned/GradNorm task weights.
- `configs/features_manual_hist_divisions_only_v1.yaml` — the `C / P_shared /
  D / R_div` roles, the `R_div` admission gate, the similarity-feature policy,
  the restricted public claim, and the nine mandatory pre-training checks.
  VK-LSVD is fully resolved: `R_div` is exactly the seventeen manual
  counter/sum divisions.
- `configs/matrix_multitask_v1.yaml` — the seven-row matrix `M1..M7`, the
  seven mandatory contrasts, capacity matching, the serving-performance
  protocol, the test lock, and the execution priority order.
- `configs/single_target_followup_v1.yaml` — frozen **before** any test
  unlock, fixing the headline targets so they cannot be chosen after seeing
  multi-target test results.

**New implementation.**

- `src/dcdn/multitask.py` — `MultiTargetSpec`, `TaskConstants`,
  `compute_task_constants`, `macro_nloss`, `MultiTargetPredictionHead`.
  Structural guarantees: masks are boolean and mandatory; an observed `0` is a
  valid negative; constants may only be estimated on train and are immutable;
  a target unobserved in a batch is excluded rather than producing NaN; a
  constant-on-train binary target or zero-variance regression target is
  rejected instead of silently normalized by zero. 33 tests.
- `src/dcdn/synthetic_multiquotient.py` — `Synthetic-MultiQuotient-3` with
  six pre-hashed slices, an analytical identity oracle, and an explicit-ratio
  positive-control view. The `ood_common_scale` slice shares the `id` seed and
  differs only in scale, so its targets are bitwise identical row for row and
  scale consistency is measurable per row. 24 tests.
- `src/dcdn/synthetic_runtime.py` — the shared per-field encoder and the
  wiring that drives the **audited** nine-model suite on the benchmark. Tests
  assert that encoder and head parameters are identical across arms for one
  seed and that no gradient crosses between target heads. 24 tests.
- `src/dcdn/multitask_capacity.py` — multi-target parameter/FLOP accounting
  that imports the interaction formulas verbatim from
  `dcdn.capacity_artifact`, plus a mechanical ±5% resolver that reads no
  validation metric and returns two bracketing controls when no candidate
  lands inside the tolerance.

The multi-target objective is a **standalone module**, not a rewrite of the
frozen single-target `training.py`, so a concurrent session working in that
file is not disturbed. Wiring it into `training.py` remains an open step.

**Explanatory notebook.** `notebooks/dcdn_explained.ipynb`, generated
deterministically from `notebooks/dcdn_explained.py` by
`scripts/nb_from_percent.py` and executed with outputs embedded. It uses the
real `src/dcdn/core.py`, demonstrates the masked reciprocal carrier, proves
quotient representability with analytically constructed weights
(`max|error| = 0`), and then runs a multi-target MovieLens-1M study under the
full feature policy with all nine pre-training checks passing. Two consecutive
runs are bit-identical on every reported number. MovieLens-1M is didactic only
and is recorded in AGENTS.md as never a result cell.

The notebook publishes its measurements as found, including the ones that cut
against the naive thesis: the reciprocal branch does **not** confer
scale-invariance (an MLP beats DCDN on the common-scale and low-ratio OOD
slices), `DCDN-Core-L2` is markedly less stable out of distribution than `L1`,
the removed manual divisions carry real predictive value, reduced DCDN closes
that gap only partially, and DCDN costs about 1.6x DCNv2 latency at `B=1`,
which fails the `p95 <= 1.10x` runtime-matched threshold and must therefore be
reported as a Pareto trade-off.

**Test access.** Unchanged and still sealed: `test_unlocked=false`, no test
loader, no test metric. Nothing in this reset opened a test row.

## 2026-08-20 Synthetic-MultiQuotient-3 diagnostic invalidated

**Superseding status: RQ1 is not complete.** Commit `8382d286` captured a
useful diagnostic, but it did not execute the frozen M1–M7 matrix: DCDN-FULL
and AFN+-FULL were absent, while an unregistered MLP arm was present. The
committed result, selected-config, serving, and index JSON sidecars also fail
strict exact-byte verification because the written trailing newline was not
included in their digest. These outputs are `diagnostic_only`; their numbers
below are retained solely for forensic context. Exact actual/recorded hashes
and invalidation reasons are in
`artifacts/legacy_audit/synthetic_multiquotient_invalid_v0.json`.

Protocol executed in order: six slices materialized and hashed **before** any
training step; capacity control resolved mechanically in (parameter, FLOP)
space with no validation metric read; validation-only tuning at seed 42 with
two learning rates per arm; selected configurations frozen and hashed; final
evaluation for seeds 42/43/44 over eight arms and four slices.

| identity | SHA-256 |
|---|---|
| benchmark manifest | `659d29f565bd51338e8c311a5c1c94335a499fbb2f62017d26581e785d8e72fd` |
| task constants | `5f40d98442140c776faa35b9558ed94b78540593010047eb704b2bbd3b1987c6` |
| selected configs | `63555e1f3b1405643372e4600366067df48bdabb6fabbc58cf406501936e5653` |

Macro-NLoss, mean over three seeds (lower is better; 1.0 is the frozen
constant predictor):

| arm | interaction params | ID | OOD common-scale | OOD low-ratio | OOD high-ratio |
|---|---:|---:|---:|---:|---:|
| analytical oracle | 0 | 0 | 7.2e-15 | 0 | 0 |
| DCNv2 + explicit ratios (positive control) | 264 | 1.10e-4 | 3.07e+06 | 929 | 69.8 |
| DCNv2-capacity-matched | 148 | 2.67e-4 | 10.65 | 0.860 | 50.8 |
| DCDN-Core-L1 | 144 | 3.12e-4 | 5,975 | 2,990 | **11.76** |
| DCNv2-Core | 144 | 1.14e-3 | 8.15e+07 | 2.81e+06 | 60.4 |
| MLP | 168,832 | 1.26e-3 | 14.61 | 0.797 | 56.9 |
| AFN+ | 560,402 | 4.28e-2 | 29.87 | 0.714 | 117.9 |
| linear probe | 0 | 3.10e-1 | 60.6 | 67.4 | 130.2 |

Zero non-finite predictions and zero reciprocal-floor incidence across every
arm, seed, and slice.

**RQ1 answer: yes, with a bounded scope.** DCDN-Core-L1 beats DCNv2-Core
in-distribution by 3.6x at *identical* interaction parameter count (144 each,
`1·2·8·9 = 2·(8²+8)`) and near-identical FLOPs, and dominates the high-ratio
regime by 4.3x. Because the two architectures differ only by the reciprocal
branch, that is an architecture effect, not a capacity effect.

**What cuts the other way, reported because it is what happened.** The
mechanically capacity-matched low-rank mixture control is *better* than DCDN
in-distribution (2.67e-4 versus 3.12e-4), so the RQ3 answer here is
regime-dependent rather than a clean win for reciprocal structure. DCDN does
not confer scale invariance: its scale-consistency figure is 40.7 against 1.67
for the capacity control, and it is ~560x worse on the common-scale slice,
because the product term grows with operand magnitude even though the pure
quotient is exactly representable. DCDN is also poor in the low-ratio regime.
And handing a model the ratio outright still beats learning it
in-distribution: the positive control is 2.8x better than DCDN on ID.

**Serving performance.** `results/synthetic_multiquotient/serving_performance_v1.json`.
A first pass under peak host contention was discarded as corrupted (`DCNv2-Core`
p99 = 74.8 ms against its own p50 of 0.73 ms). The retained pass has p99 under
1.32 ms everywhere and p50/p95 ratio agreement within 1% for every pair, but is
recorded as `provisional_host_not_idle_p95_tail_is_contaminated` because a
concurrent session held the load average at 6.94. Verdicts:

| pair | p95 ratio | throughput ratio | runtime matched |
|---|---:|---:|---|
| DCDN vs DCNv2-Core | 1.419 | 0.711 | **no** |
| DCDN vs MLP | 1.257 | 0.796 | **no** |
| DCDN vs DCNv2-capacity-matched | 1.043 | 0.971 | yes |
| DCDN vs AFN+ | 0.825 | 1.188 | yes, DCDN cheaper |

Against ordinary DCNv2, DCDN costs ~42% more latency and delivers ~71% of the
throughput, so "at the same serving performance" is **not** an available phrase
for that pair; it is a Pareto trade-off. On the ID slice DCDN is in fact
*dominated* by the capacity-matched control on the quality/latency frontier.
The ordering reverses on `ood_high_ratio`, so a single frontier is not a
summary of this benchmark.

**Consolidated tables.** `results/multitask/`: `multitask_summary.csv` (32
rows), `multitask_per_target.csv` (96), `efficiency.csv` (15), `pareto.csv`
(5), `run_registry.csv` (24), and `multitask_index.json`. The index records
`vk_lsvd_up0.01_ir0.01`, `yambda_50m`, `kuairand_pure`, and
`kuairec_big_matrix` as `executed: false` with reasons. Those four mandatory
cells are **not** complete and are not represented as complete anywhere.

**Reproduction:** `docs/reproduction_commands.md`.

### Data-integrity incident

`src/dcdn/synthetic_multiquotient.py` and `scripts/aggregate_multitask.py` were
each truncated to zero bytes during the session — the FUSE hazard AGENTS.md
already records from the earlier acquisition-script incident, triggered here by
a concurrent repository-wide formatter. Both were rewritten. The restore of
`synthetic_multiquotient.py` was verified by reproducing the recorded benchmark
manifest SHA-256 and all six slice hashes exactly, and the full 81-test suite
for the new modules passes. A copy of every task-owned file is kept outside the
FUSE mount for the remainder of the session.

## 2026-08-20 corrected Synthetic-MultiQuotient-3 RQ1 completed

The corrected replacement `results/synthetic_multiquotient/rq1_v2/` completed
under clean commit `305534223faa6073f1077d2c298150263b1ca788`. It executed
exactly the frozen nine-arm matrix and froze seed-42 validation selections
before evaluating seeds 42, 43, and 44.

| identity | SHA-256 |
|---|---|
| benchmark manifest | `659d29f565bd51338e8c311a5c1c94335a499fbb2f62017d26581e785d8e72fd` |
| pretraining manifest | `3d257ef17b0b8ea402f380e4a6fd4e0de31245f83ab860e88dddc7173bc91549` |
| selected configs | `1f4755903d11d4f8e4066376401179a2091fc0b180429be0f9e52da958cb3c54` |
| result receipt | `292755a33e391729b266372f564e571aa54f734c89ec9723126271cf016f5677` |

All five output pairs pass exact-byte sidecar verification. The result has 27
records, nine unique arms, seeds 42/43/44, zero non-finite predictions, and
zero non-finite training batches.

Macro-NLoss means support RQ1 only in a bounded sense. Reduced DCDN versus
ordinary reduced DCNv2 is `3.64x` better on ID (`3.117e-4` versus `1.136e-3`),
and `5.14x` better on high-ratio (`11.76` versus `60.43`). The mechanically
capacity-matched control is slightly better on ID (`2.667e-4` versus
`3.117e-4`) and much better on common-scale and low-ratio, while reduced DCDN
is `4.32x` better than that control on high-ratio (`11.76` versus `50.76`).
Therefore the reciprocal carrier has demonstrable quotient-regime predictive
value, but universal DCDN superiority and scale invariance are not supported.

The first replacement directory `rq1_v1` is execution-invalid because its
selected-config file SHA-256 is `538e791e...285f0`, while its sidecar records
`9b20efce...bf72`; its otherwise valid outputs remain forensic evidence only
and are excluded from all claims.

No public-dataset test row was opened. The frozen public test firewall and
`test_unlocked=false` remain unchanged. Results are now reported in README.

## 2026-08-20 RQ2 serving performance and efficiency complete

The cost half of RQ2 is **complete**. Full report:
`docs/rq2_serving_performance.md`. Artifacts:
`results/efficiency/rq2_perf_v1/` — `pre_measurement_manifest_v1.json`
(`5de33d29…`), `results_v1.json` (`5d69b484…`), `efficiency.csv` (84 rows,
`766fdfd3…`), `pareto.csv` (32 rows, `57be62ab…`). All four verify against
strict exact-byte sidecars. The manifest was published before the first timed
forward.

This study trains nothing and reads no dataset row, label, or test metric, so
it did not touch any pipeline in flight. Arm ids match
`results/synthetic_multiquotient/rq1_v2` exactly, so cost joins to the
published quality table with no mapping layer.

**Protocol.** Full multi-target forward with every head, `eval()` +
`inference_mode()`, device-resident inputs, batches {1, 64, 1024}, 200 warmups,
1,000 synchronized forwards, 5 independent rounds, on one A100-SXM4-80GB.
14 arms per geometry: the frozen `M1..M7` matrix, a linear floor, a DCDN depth
control, and the Tier-B zoo.

**Two geometries.** `quotient` reproduces the RQ1 widths so cost and quality
share axes. `ranker` is an industrial-scale geometry stand-in (6 categorical
fields, 96 bypass, 61/78 operands, embedding 16, four heads) matching the
counts `features_multitask_v1` derives; **no VK-LSVD row, label, or statistic
is read**, only the shape. The two geometries disagree on a headline verdict,
which is the justification for measuring both.

**Validity.** The load gate was waived (a concurrent session held the load
average at 3.9) and the waiver is recorded with its basis. Every forward was
timed twice — wall clock and CUDA device events. Median wall/device p95 ratio
**1.012**, worst **1.100**, ceiling 2.0, worst p99 anywhere 2.28 ms. Status:
`final_load_gate_waived_by_device_time_agreement`. A load-average threshold is
a proxy; device-time agreement is the measurement.

**Runtime-match verdicts** (`p95(B=1) <= 1.10x` and `throughput >= 0.90x`):

| geometry | pair | p95 ratio | throughput | matched |
|---|---|---:|---:|---|
| quotient | DCDN vs DCNv2 | 1.387 | 0.725 | no |
| quotient | DCDN vs capacity-matched | 1.030 | 0.979 | yes |
| quotient | DCDN vs AFN+ | 0.820 | 1.207 | yes, DCDN cheaper |
| ranker | DCDN vs DCNv2 | 1.343 | 0.764 | no |
| ranker | DCDN vs capacity-matched | 0.735 | 1.364 | yes, DCDN much cheaper |
| ranker | DCDN vs AFN+ | 0.768 | 1.329 | yes, DCDN cheaper |

Full-regime pairs (`M6` vs `M5`, `M6` vs `M7`) behave identically to their
reduced counterparts.

**Findings.**

1. **DCDN is never runtime-matched to DCNv2** — both geometries, both feature
   regimes, p95 ratio 1.33–1.39 and throughput 0.72–0.77. "At the same serving
   performance" is not an available phrase for that pair; it is a Pareto
   trade-off. Stable across a 7.6x change in operand width, so not a toy
   artifact.
2. **DCDN's cost is not FLOP-bound.** At ranker width it carries 6.6x fewer
   interaction parameters (7,564 vs 49,612) and 2.2x fewer FLOPs (68,757 vs
   152,792) than DCNv2 — because it crosses `P` only while DCNv2 crosses the
   full flat width — and is still 1.34x slower. The masked reciprocal carrier
   is elementwise and memory-bound work. Any efficiency claim resting on
   DCDN's FLOP count would be contradicted by measurement.
3. **Parameter/FLOP matching and runtime matching are different checks, now
   measured rather than asserted.** `M4` was resolved mechanically to within
   0.1% of `M2` on parameters *and* FLOPs at both widths, yet the runtime
   verdict inverts: DCDN/M4 p95 is 1.030 at quotient width and 0.735 at ranker
   width, with throughput 0.979 versus 1.364. The low-rank mixture's gating and
   expert loop cost far more wall time than its FLOP count implies.
4. **DCDN dominates AFN+** on latency, throughput, and VRAM (78 MB vs 161 MB at
   ranker width).
5. **Depth is not free**: DCDN-L2 costs +25% p95 at quotient width and +20% at
   ranker width over L1, on top of being markedly less stable out of
   distribution in RQ1.
6. **Removing the manual divisions buys almost nothing at inference.**
   DCNv2-FULL and DCNv2-REDUCED are within 0.2% on latency at ranker width;
   DCDN-FULL and DCDN-REDUCED within 1%. The value of the removal claim is
   feature-pipeline complexity, not serving cost, and the manuscript must not
   imply an inference saving the measurement does not show.

**Pareto.** `pareto.csv` covers the `quotient` geometry only — the geometry
whose models the quality source actually trained. A defect that would have
joined ranker latency to quotient quality was caught before publication, fixed
with an explicit `quality_source_compatible` flag, and covered by a regression
test. On the ID slice DCDN is dominated; on the high-ratio OOD slice it is the
frontier point with the best quality of any arm. Which arm is Pareto-optimal
depends on the operating regime, so both slices are published.

`tests/test_benchmark_efficiency.py` adds 35 tests covering geometry
arithmetic, the denominator mask, arm-id agreement with `rq1_v2`, encoder and
head identity across arms, capacity tolerance, verdict thresholds, the
device-agreement gate, and exact-byte publication including refusal to
overwrite.
## 2026-08-21 VK-LSVD seven-head target freeze

The owner explicitly confirmed the real-dataset VK headline target set:
normalized `watch_time` plus `like`, `dislike`, `share`, `bookmark`,
`click_on_author`, and `open_comments`. The immutable forward-only contract is
`configs/targets_multitask_v2.yaml`, SHA-256
`d03704fb856551cb5d2eee7810992f0e8fa23909860a9c28c626b5362456c6a6`,
with a strict adjacent sidecar. The v1 artifact is unchanged and remains the
lineage of completed Synthetic-MultiQuotient-3 RQ1; no existing result was
rebound.

The VK transformer/materializer now emits all seven ordered label/mask pairs,
reads current-row actions only as labels, keeps FULL and REDUCED shared values
bitwise identical, and binds the v2 target digest. Dataset-independent
reporting now includes masked LogLoss, ROC-AUC, threshold-grouped AUPRC, and
uniform user-GAUC with eligible-user and eligible-row coverage for every
binary head. Observed zero remains a valid negative label.

CPU-only verification: 148 expanded VK/feature/runtime tests passed; 67 final
v2/objective/metric tests passed; an isolated source-only broad regression
passed 687 tests in 82.72 seconds. Scoped Ruff lint/format, bytecode, strict
sidecar, and `git diff --check` pass. No raw-data row, GPU, validation metric,
test path, or test metric was opened. Next gate: exact task-owned commit and
clean-code preflight, then immutable train/validation-only materialization for
VK weeks 20--25; week 26 remains sealed.

## 2026-08-21 VK-LSVD seven-head train/validation views materialized

The forward-only materializer ran under clean commit
`83dfe21de90746000544402e17ac07453020b6fc` and published
`data/manifests/vk_lsvd_multitask_views_v2.yaml`, SHA-256
`e880130ae672e97257343e7c1e57c91c5087dd9e303231dd0b738fe6c3b88bf5`,
with a strict adjacent sidecar. It contains 9,022,651 rows: 7,505,820 train
rows from weeks 20--24 and 1,516,831 validation rows from week 25. Twelve
FULL/REDUCED Parquet shards were independently streamed, rehashed, reopened,
schema-checked, and row-counted. All seven label masks are observed, and every
FULL shard projected onto the ordered REDUCED schema equals the REDUCED shard
bitwise.

The manifest binds target-v2 SHA-256 `d03704fb...6c6a6`, the frozen feature
policy, experiment matrix, and legacy-view lineage. Its firewall records
`test_unlocked=false`, `test_labels_opened=false`,
`test_metrics_computed=false`, `test_view_materialized=false`, and
`week26_file_open_attempted=false`. No model metric was computed during
materialization.

Post-materialization local guards at `2026-08-21T07:21:19.911701Z` pass:
estimated spend is USD 255.74 (USD 259.03 at a conservative one-hour ETA),
project storage is 22.701678 GiB of the 40 GiB cap with 27.298322 GiB logical
headroom, overlay free space is 57.975319 GiB, both A100s are idle at 0 MiB,
and the registry has no running task. The immutable ledger epoch remains
`2026-08-18T09:13:18Z`; compute reserve starts
`2026-08-23T01:13:18Z`. Next: integrate the seven-head runtime/training path
and run deterministic train-only smokes. Week 26 remains sealed.

## 2026-08-21 RQ2 addendum: serving cost under the best available implementation

The reference-implementation verdict is **superseded for cost purposes**. Full
report: `docs/rq2_fused_inference.md`. Artifacts:
`results/efficiency/rq2_fused_v1/` — `results_v1.json` (`f9f1d5a7…`),
`fused_efficiency.csv` (60 rows, `d0a00787…`), pre-measurement manifest
published before the first timed forward. All verify against strict exact-byte
sidecars.

**Why.** `docs/rq2_serving_performance.md` found DCDN 1.33–1.39x slower at p95
than DCNv2 while carrying 6.6x fewer interaction parameters and 2.2x fewer
FLOPs at ranker width. That inversion is the signature of a schedule artifact:
DCDN decomposes into 2 GEMMs + 4 elementwise passes per layer plus a ~17-op
reciprocal carrier, against DCNv2's 1 GEMM + 2 passes. At serving batch sizes
these blocks are launch-bound, so the reference measurement compared
implementations, not architectures.

**What was done.** `src/dcdn/fused_inference.py` adds Triton kernels fusing
DCDN's two per-layer GEMMs into one `[2F, F]` matrix, its epilogue into one
kernel, and its masked reciprocal carrier into one kernel — and equally fusing
the DCNv2 cross epilogue. **Both architectures were optimised**; optimising only
DCDN would replace one unfair comparison with its mirror image. Each arm is
represented by whichever implementation is faster for it.

**Equivalence, verified before any timing.** Worst deviation from the audited
module across every geometry and batch: **`max|Δ| = 1.53e-05`** against a
`1e-03` tolerance — float32 GEMM reassociation only. The fused path is
inference-only, implements the all-valid denominator case the frozen preflight
verifies, and rejects a gating validity tensor rather than ignoring it. No
parameter and no quality number changes.

**Result.** Fusion speeds DCDN by **1.27–1.43x** and *slows* DCNv2 to
**0.91–0.94x** — DCNv2's reference path is already minimal and Triton launch
overhead exceeds the remaining saving. That asymmetry is itself the evidence
that the original gap was schedule, not architecture.

p95 at `B=1`, best implementation per arm:

| geometry | arm | reference | fused | best |
|---|---|---:|---:|---:|
| quotient | DCNv2-REDUCED | 0.772 | 0.845 | 0.772 (ref) |
| quotient | DCDN-REDUCED | 1.163 | **0.811** | 0.811 (fused) |
| ranker | DCNv2-REDUCED | 0.930 | 0.989 | 0.930 (ref) |
| ranker | DCDN-REDUCED | 1.220 | **0.956** | 0.956 (fused) |

**The runtime-matched verdict inverts.** Under the reference implementation
none of the four DCDN-versus-DCNv2 pairs passed (1.33–1.39x latency,
0.72–0.77x throughput). Under the best available implementation **all four
pass**, with margin:

| geometry | pair | p95 ratio | throughput | matched |
|---|---|---:|---:|---|
| quotient | DCDN vs DCNv2 | 1.050 | 0.959 | yes |
| quotient | DCDN-FULL vs DCNv2-FULL | 1.024 | 0.968 | yes |
| ranker | DCDN vs DCNv2 | 1.028 | 0.969 | yes |
| ranker | DCDN-FULL vs DCNv2-FULL | 1.046 | 0.951 | yes |

Worst latency ratio 1.050 against the 1.10 threshold; worst throughput 0.951
against 0.90.

**Consequences for claim wording.** "At comparable latency and throughput"
**becomes available** for the DCDN-versus-DCNv2 pair on both geometries and
both feature regimes, *provided the fused kernel is the deployed
implementation*. The earlier ~40% figure remains correct for unfused PyTorch
and must be cited that way. This does **not** establish that DCDN is cheaper —
only parity within the frozen threshold, since both architectures are
launch-bound rather than arithmetic-bound at `B=1`.

**Unchanged.** RQ2 as a whole still cannot be answered: this supplies the cost
denominator, and no recommendation-dataset quality numerator exists. Validity:
median wall/device p95 agreement 1.014, worst 1.017, ceiling 2.0.
`tests/test_fused_inference.py` adds 43 tests.

## 2026-08-21 Contrast M2 − M5 recorded

`M2` (DCDN on base operands, no ready-made ratios) versus `M5` (DCNv2 on base
operands **plus** the three explicit ratios), from `rq1_v2`:

| slice | M2 DCDN-REDUCED | M5 DCNv2-FULL | better | factor | seed-separated |
|---|---:|---:|---|---:|---|
| ID | 3.12e-4 | 1.10e-4 | M5 | 2.84x | yes |
| OOD common-scale | 5,975 | 3.07e+06 | M2 | 514x | yes |
| OOD low-ratio | 2,990 | 929 | M5 | 3.22x | no |
| OOD high-ratio | 11.8 | 69.8 | M2 | 5.94x | no |

The claim that DCDN on base features beats DCNv2 on the full feature set is
**not supported in general**: two slices each, and only two of four are
separated across seeds. M5 wins in-distribution, which is expected since the
quotient is handed to it. M2 wins decisively when operand scale shifts, where
DCNv2's product path grows with the input and the supplied ratios do not help.
On high-ratio the nominal 5.94x favouring M2 is unstable: DCNv2-FULL's three
seeds span 179.1 / 20.6 / 9.7, an 18-fold range, so the ranges overlap.

## 2026-08-21 RQ2 addendum: batch sweep across the production range

The runtime-parity claim no longer rests on a single operating point. Full
report: `docs/rq2_batch_sweep.md`. Artifacts:
`results/efficiency/rq2_batch_sweep_v1/` — `results_v1.json` (`2fd8aa9f…`),
`batch_sweep.csv` (75 rows, `4cc54f99…`), pre-measurement manifest published
before the first timed forward, all verified against exact-byte sidecars.
Figures: `docs/figures/batch-sweep-{latency,throughput,ratio}[-dark].svg`.

**Why.** The frozen threshold is defined at `p95(B=1)`, but production batches
dynamically over `1..16000`, and `B=1` is the worst point for this comparison:
pure fixed overhead, where DCDN's 2.2x lower interaction FLOPs cannot show.
15 batch sizes from 1 to 16384 were measured under each arm's best available
implementation (DCDN fused, DCNv2 reference), with a declared lighter sweep
protocol of 50 warmups / 300 forwards / 3 rounds.

**Headline: there is no crossover.** Latency is essentially flat across the
whole range — a 16,384-fold increase in work costs 6-11% more time:

| arm | B=1 | B=1024 | B=16384 | growth |
|---|---:|---:|---:|---:|
| DCNv2-REDUCED | 0.920 | 0.972 | 0.974 | +6% |
| DCDN-REDUCED (fused) | 0.952 | 1.025 | 1.059 | +11% |
| DCNv2-capacity-matched | 1.662 | 1.726 | 1.725 | +4% |

On an A100 this model class stays dispatch-bound across the entire production
batch range. Throughput scales near-linearly to ~16.8M examples/s purely by
amortising a fixed ~0.95 ms call.

**DCDN stays runtime-matched at every batch size**, but the margin narrows:
latency ratio 1.035 at `B=1` rising to 1.086 at `B=16384` against the 1.10
threshold; throughput 0.966 falling to 0.920 against the 0.90 threshold. Parity
is comfortable below `B~1024` and thin at the top of the range.

**Why the FLOP advantage never materialises.** Arithmetic is never the
bottleneck: at `B=16384` the adapter is ~165 MFLOP, which an A100 retires in
under 0.01 ms against a ~0.95 ms call. CUDA-event device time confirms this is
real GPU-timeline behaviour, not host overhead — wall-over-device is **1.01 at
every point**, and DCNv2's device latency moves only 0.907 -> 0.961 ms across
the sweep. The ~0.95 ms is dispatch and inter-kernel gaps. **The remaining
lever is dispatch elimination (CUDA Graphs, fusing across the encoder/tower
boundary), not further FLOP reduction.**

**Refutes a plausible expectation.** One might predict DCDN's lower FLOP count
would become a throughput win at large batch. It does not, anywhere in the
production range, on this hardware. Publishing that prediction would have been
wrong.

**Capacity control at scale.** `M4`, matched to DCDN within 0.1% on parameters
and FLOPs, costs 1.72 ms at every batch size and delivers 9.5M/s against DCDN's
15.5M/s at `B=16384` — 39% less throughput. Its gating softmax and expert loop
are many small kernels regardless of FLOP count. This is a second, stronger
demonstration that capacity matching and runtime matching are different checks:
the gap holds across four orders of magnitude in batch size.

Figures are self-contained SVG emitted by `scripts/plot_batch_sweep.py` with no
plotting dependency, so the pinned `uv.lock` environment is untouched; light and
dark surfaces are generated from the same palette steps, series are direct
labelled as well as colour coded, and latency and throughput are separate charts
rather than a dual axis.

## 2026-08-21 VK-LSVD seven-head runtime/cache checkpoint

The seven-head runtime and training integration is locally frozen at commit
`004e753177ce9e981e2b0dbbddbdb1a185b2a2ac`. The full CPU repository suite
passed 766 tests; Ruff, format, bytecode compilation, and `git diff --check`
passed. The legacy scalar final-evaluation runner explicitly rejects the
multi-target model, so week 26 cannot be opened through that path.

The immutable train/validation scan
`data/manifests/vk_lsvd_multitask_full_scan_v2.json` has SHA-256
`e67f72187f73ed9ad37e1a23cdd5c8b679d1fbc3be35a2781968c270a1497bb6`.
It rehashed and reopened all twelve FULL/REDUCED shards and records 7,505,820
train plus 1,516,831 validation rows. It records
`test_split_present=false`, `week26_path_constructed=false`, and
`model_metrics_computed=false`.

Two regenerable overlay caches were built and independently reopened for both
train and validation. `C+P` binds metadata SHA-256
`f9cb51ae80d4ce45b532ef2f1f10877c216d85eeab34d5dc1d40db223e6cc293`
and content SHA-256
`72834709c5c1735b8369571cbade96cb7b0f705d2bccaeb0335e444dd49044b8`;
`C+P+E` binds metadata SHA-256
`2e3d139e5601e50b72bb26d86040b134f442203cf436ee10be4a41d637ebc085`
and content SHA-256
`29208049ffda758f7f302e5080dabf8301631691402f9c0fcef44c72dc39d615`.
Both contain the ordered seven targets and the same row identity; cosine,
dot-product, norm, and iALS fields remain in both regimes with reciprocal mask
false, while only the seventeen frozen historical divisions are added to
`C+P+E`. No GPU or model metric was used.

At `2026-08-21T08:18:59Z`, estimated spend was USD 258.90 and the one-hour
projection USD 262.19. Durable project usage was 22.702611 GiB of 40 GiB;
overlay free space was 51.439293 GiB. The next gate is a new immutable

## 2026-08-21 RQ2 addendum: HiFormer-style layer after the cross

Full report: `docs/rq2_hiformer_stack.md`. Artifacts:
`results/efficiency/rq2_hiformer_v1/` — `results_v1.json` (`f91b3a2c…`),
`hiformer_stack.csv` (24 rows, `2ba66c08…`), pre-measurement manifest
(`7d9259c2…`). All verify against exact-byte sidecars. Figures:
`docs/figures/hiformer-{latency,throughput,ratio}[-dark].svg`.

**Scope.** Cost only. The HiFormer-style layer is outside the frozen nine-model
suite, is not a faithful reproduction of the published architecture, trains
nothing, and produces no quality number.

**Why.** The three cross-only studies found DCDN/DCNv2 runtime parity and that
the reduced feature view buys nothing at inference (0.2-1%). Both follow from a
cross layer costing roughly linearly in width on a dispatch-bound stack. Once a
component with `O(F^2)` cost sits downstream, the reduction stops being
cosmetic: 68 field tokens on REDUCED against 85 on FULL is 295,936 against
462,400 attention FLOPs per example, a factor of **1.562**.

**Attention changes the regime.** Latency is no longer flat — it grows from
~1.95 ms at `B=1` to 24.98 ms (reduced) and 30.89 ms (full) at `B=16384`, and
throughput saturates near 656k/s and 530k/s instead of scaling to 16.8M/s.

**Result — the crossover.** DCDN on reduced against DCNv2 on full:

| B | latency ratio | throughput ratio |
|---:|---:|---:|
| 1 | 1.024 | 0.976 |
| 256 | 1.017 | 0.984 |
| 1024 | 0.926 | 1.080 |
| 4096 | 0.829 | 1.207 |
| 16384 | **0.809** | **1.237** |

The crossover sits between `B=256` and `B=1024`. Below it DCDN is ~2% slower;
above it DCDN-REDUCED reaches 19% lower latency and 24% more throughput than
DCNv2-FULL. Peak VRAM at `B=16384` is 3,286 MB against 4,811 MB — 1.46x lower.

**Two controls keep the reading honest.** At matched token count DCDN and DCNv2
are indistinguishable once attention dominates (24.976 vs 24.983 ms at
`B=16384`, ratio 1.000), so the win is **not** the reciprocal branch being
faster. And DCDN-FULL pays the same 1.24x penalty as DCNv2-FULL. The effect is
entirely token count, identical for both architectures: with attention
downstream, what matters is how many features reach it, and the architecture
matters only insofar as it lets you send fewer.

**Claim boundary.** This is a cost win, not dominance. Quality under this stack
is unmeasured. The nearest quality evidence is contrast M2-M5 on the cross-only
stack, where **DCNv2-FULL was better in-distribution by 2.84x** with clean seed
separation and DCDN-REDUCED won only under operand-scale shift. A cost win plus
an in-distribution quality loss is a trade-off. Establishing the full claim
needs both configurations trained under this stack and compared at matched
cost — a training study on recommendation data, still the outstanding gap.

The crossover position depends on the downstream component: a wider or deeper
attention stack moves it left, and a stack with no superlinear component
removes the effect entirely, which is what the cross-only studies found.

Validity: worst wall-over-device p50 agreement across all 24 points was 1.007.

**Also recorded this session:** a component decomposition of the cross-only
ranker stack at `B=1` — encoder 33-38%, interaction block 21-30%, adapter +
tower + heads 21-24%. The differing part is only 21-30% of the forward, which
is why a 1.29x block-level gap becomes 1.03x end to end. The runtime-parity
verdict is therefore partly a consequence of the shared parts dominating; a
configuration where the block is a larger fraction could read differently.

## VK-LSVD seven-head encoder/capacity/protocol freeze — 2026-08-21

The premetric multi-target execution boundary is now frozen without opening a
validation metric or using a GPU. `configs/encoder_multitask_v2.yaml` binds the
seven-head target/task constants, exact FULL/REDUCED cache identities, field
orders, denominator masks, and model allowlists. The mechanical capacity
artifact selects rank 12 for `DCNv2-capacity-matched`: 22,445 versus 22,503
trainable non-embedding parameters (0.9974x) and 45,862 versus 44,919 FLOPs
(1.0210x), both inside the frozen ±5% rule and without quality metrics.

Separate Protocol v3 locks are active for `C+P` and `C+P+E`; both are
`premetric_locked`, `test_unlocked=false`, and `test_path_available=false`.
An end-to-end CPU reopen verified train/validation row counts and every runtime
hash through cache identity → encoder lock → active Protocol. The focused
multi-target/protocol regression passed 135 tests. No task worker or trainer is
running. The next executable gate is a dedicated seven-head supervised CLI and
one-effective-batch DCDN/DCNv2 implementation smoke; no validation metric may
be used by that smoke.
