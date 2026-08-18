# Deterministic implementation preflight

Checkpoint: `2026-08-18T12:36:18Z`

No public dataset, validation metric, test label, or project test metric was
accessed by this preflight. All numerical cases use synthetic tensors.

## Reproducible environment

- Python 3.12.3.
- PyTorch 2.8.0+cu128; CUDA build 12.8.
- Canonical lock: `uv.lock`, SHA-256
  `af9fcbb29fedabaa0105ff2682c8d858eb3660ef6a83e7d60ccfa5ac96f49df6`.
- `uv lock --check --offline`: passed.
- Isolated environment: `/tmp/dcdn-venv-py312`.
- Package/model/data caches are outside the persistent repository and
  telemetry plus implicit credential reuse are disabled by `scripts/env.sh`.

## CPU and static gates

- Full pytest suite: 101 passed in 5.96 seconds.
- Ruff format check: passed.
- Ruff lint: passed.
- Core recurrence, fixed anchor, carrier/mask isolation, debug and optimized
  invalid-denominator paths, FP32/BF16 policy, gradients, optimizer update,
  serialization metadata, and one-batch overfit are covered.
- The production source hash is pinned and its strict experimental adapter
  covers `[B,8,F]` shape, one-block `x_i=x_0`, BN/dropout behavior, and
  parameter formula without changing the provided file.
- Exact CPU resume matches row IDs, loss sequence, model parameters, BN
  buffers, dropout trajectory, optimizer, scheduler, sampler cursor, and RNG.
- Protocol hash sidecars, fail-closed test access, metrics/tie policies, run
  registry transitions, clustered bootstrap, cost, and storage guards pass.

## Dual-A100 gate

Two independent processes ran concurrently, one per physical A100. Each
performed DCDN-Core forward/backward, finite-gradient clipping, optimizer
update, denominator-mask checks, metadata-aware state round-trip, and repeated
same-seed determinism in FP32 plus a BF16 pass.

| Precision | Model-state SHA-256 | Output SHA-256 |
|---|---|---|
| FP32 | `74bac184c2da3083dff789fa97078d56e98e5e59ec1f45e54aabf8f1c27cbf0c` | `47c43bba955394afdd28e120048ec4e86bf934c10b97b66fed3f06016ef560ce` |
| BF16 | `bef5cc65648f051a7e170090a5e47c0e9f49bda38b735bbf8b206c3332f57f91` | `14bab8811cb0b5ed7052f84a71626399d0130b417ddc040e2832f2d2d37a3adf` |

The hashes matched across GPU 0 and GPU 1. Both processes exited successfully,
and a post-run `nvidia-smi` query reported no compute processes.

## Storage and execution state

- Project-owned durable bytes: 2,878,464 (0.002681 GiB).
- Frozen logical persistent headroom: 49.997319 GiB.
- Temporary overlay available: 92.553829 GiB; usage 7.45%.
- Estimated incremental spend: USD 36.13 under the owner-approved local policy.
- Test access remains locked.

The optimized Core path deliberately avoids synchronous tensor predicates.
An invalid active denominator produces a NaN sentinel and therefore requires
the mandatory asynchronous non-finite monitor in full training. The complete
materialized operand scan remains mandatory before that optimized path.
