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
