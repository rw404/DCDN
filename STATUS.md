# DCDN execution status

Updated: `2026-08-18T15:17:23Z`

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
- Estimated incremental spend is USD 44.96 with USD 415.04 hard-cap headroom at
  this checkpoint.
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
- Real model execution remains fail-closed on the previously unresolved
  trainable shared encoder and bounded raw-shard bridge. Their contracts and a
  dedicated full-materialized-scan hash are being frozen before GPU work.

## Preflight result

**SOURCE/MODEL CHECKPOINT PASSED UNDER OWNER OVERRIDE.** Local guards report
0.00395 GiB of project-owned durable artifacts and 71.95 GiB free on the
temporary overlay. The operational durable-artifact cap remains 40 GiB with
at least 10 GiB kept free.

The exact next actions are to freeze the trainable shared encoder and the
full-scan-attested bounded raw-shard source, integrate them with the
deterministic harness, then run synthetic two-GPU and real train-only
smoke/throughput checks. Test access remains locked.
