# DCDN execution status

Updated: `2026-08-18T18:37:19.390739Z`

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
- Estimated incremental spend is USD 55.93, with USD 314.07 warning-threshold
  headroom and USD 404.07 hard-cap headroom. At the conservative one-hour
  ETA the projection is USD 59.22, leaving USD 400.78 hard-cap headroom.
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
- Dataset-specific feature audits are complete and passed before metrics. No
  training, validation metric, test-label inspection, or model evaluation has
  occurred.
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
  active premetric protocol. Real model training has not started; every source
  remains fail-closed on exact train/validation runtime identity.
- The raw/composite training harness now binds all nine runtime architectures,
  optimizer/runtime fairness fields, exact capacity candidate/rank, Protocol
  v3 identities, same-FD runtime sources, whole-model checkpoints, numerical
  monitoring, and immutable worker receipts. A fresh independent snapshot
  passed 263 synthetic CPU tests, Ruff, formatting, and compilation. Independent
  review authorizes only two one-effective-batch implementation smoke runs
  after clean Git provenance. Ordinary seed-42 validation tuning may follow a
  successful smoke; capacity-control tuning remains locked until a durable
  selected-DCNv2 learning-rate binding exists, and final/test remains locked.

## Preflight result

**VK PRE-SMOKE CHECKPOINT PASSED UNDER OWNER OVERRIDE.** At
`2026-08-18T18:37:19.390739Z`, the credential-free local cost guard allowed a
mandatory one-hour ETA: USD 55.93 estimated current spend and USD 59.22
projected, below both thresholds. The storage guard measured 5,939,200 bytes
(0.005531 GiB) of project-owned durable artifacts, 49.994469 GiB logical
headroom under the frozen 50 GiB envelope, and 62.554615 GiB available on the
temporary overlay at 37.4454% usage. Both A100s were idle at 0 MiB and 0%
utilization. The `/workspace` FUSE `df` capacity is
informational shared-backend metadata; the 40 GiB project cap and 10 GiB
reserve remain enforced. No guard violation or provider action occurred.

The exact next action is an explicit staged-diff audit, clean commit/push, and
two concurrent one-effective-batch train-only implementation smoke runs under
the active protocol (DCDN on GPU0 and full-rank DCNv2 on GPU1).
Only after those pass may seed-42 validation-only tuning begin. Test access
remains locked until selected configurations are frozen and final lineage is
activated.
