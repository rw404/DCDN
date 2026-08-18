# DCDN execution status

Updated: `2026-08-18T14:14:40Z`

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
- Estimated incremental spend is USD 41.53 with USD 418.47 hard-cap headroom at
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
- Dataset-specific feature audits are in progress. No training, validation
  metric, test-label inspection, or model evaluation has occurred.

## Preflight result

**SOURCE/MODEL CHECKPOINT PASSED UNDER OWNER OVERRIDE.** Local guards report
0.0034 GiB of project-owned durable artifacts and 87.62 GiB free on the
temporary overlay. The operational durable-artifact cap remains 40 GiB with
at least 10 GiB kept free.

The exact next actions are to finish dataset-specific leakage/schema audits,
resolve every phase-1 gate before model metrics, then run deterministic
train-only smoke/throughput checks. Test access remains locked.
