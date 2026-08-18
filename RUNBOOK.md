# DCDN RunPod runbook

## Hard envelope

- Immutable epoch: `2026-08-18T09:13:18Z`.
- Estimated accounting: USD 3.29 per elapsed Pod-hour plus USD 25 storage
  reserve.
- Warning threshold: USD 370 estimated incremental spend.
- Hard cap: USD 460 estimated incremental spend.
- Effective stop: `2026-08-23T09:13:18Z`.
- Final aggregation/backup reserve begins: `2026-08-23T01:13:18Z`.
- Durable artifacts: at most 40 GiB and always leave at least 10 GiB free.

The owner waived provider-side expiry and control-plane billing/quota evidence.
Do not use stored RunPod credentials and never stop, terminate, resize, or
otherwise mutate the Pod or its volume. At completion, stop every task-owned
process, verify durable backup, and remind the owner that billing continues
until manual termination.

## Safety defaults

- One independent process per GPU; never DDP for these models.
- W&B, MLflow, Hugging Face telemetry, and crash reporting are disabled.
- Public raw data and regenerable caches belong on temporary storage; code,
  manifests, registries, compact logs/results, and selected checkpoints remain
  under `/workspace`.
- Never access test metrics until the protocol and selected configurations are
  frozen and hashed. Never tune after test access.

## Checkpoint sequence

1. Environment/storage/cost audit.
2. Repository and dependency bootstrap.
3. Deterministic numerical, metric, checkpoint, and resume tests.
4. Dataset-specific feature/data manifests with leakage audits.
5. Validation-only smoke and throughput projection.
6. Seed-42 validation-only tuning and selected-config lock.
7. One-time test evaluation for seeds 42/43/44.
8. Uncertainty, efficiency, stability, clean-environment reproduction, and
   durable backup/read audit.

Exact commands are added only after their corresponding implementation is
validated; `STATUS.md` is updated at every protocol checkpoint.
