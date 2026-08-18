# DCDN RunPod environment audit

Audit checkpoint: `2026-08-18T11:59:30Z`

## Immutable execution clock

- `ledger_epoch_utc: 2026-08-18T09:13:18Z`
- Source: earliest locally verifiable container start (`docker-init`). This is
  conservatively earlier than the first observed Codex process at
  `2026-08-18T09:31:31Z`.
- Spend before this epoch: unknown; reconcile against the RunPod control plane.
- 120-hour stop: `2026-08-23T09:13:18Z`.
- Final eight-hour reserve begins: `2026-08-23T01:13:18Z`.
- The epoch-derived stop is earlier than the absolute deadline
  `2026-08-24T11:00:00Z` (`2026-08-24 14:00 Europe/Moscow`).

## Required execution sources

| File | SHA-256 |
|---|---|
| `AGENTS.md` | `e4bbab7cd5570871f070dbca0e5fa214c5bc64c4caac06fa856e2ea9ed6d3cad` |
| `DCDN_HANDOFF.md` | `867437620d539a05036c9f7c7910c4ea635c232212b989319a20429a8a417fee` |
| `REMOTE_AGENT_PROMPT.md` | `23e29da22d69b6065bfb2b444e705e344470ceb163c90024ded7dc132ef5f2ca` |
| `DCDNBlock_provided.py` | `d58c31c03096d112a47718a643fc91bc5dd9b5a53cd7941948145b34335ec1bb` |
| `DCDNCore_reference.py` | `6b2cdb5db0c248fcd960c42d4ce14873820cc65d382741721214166d5e83d9ac` |
| `DCNvsDDN.tex` | `f75f96d584719099607bbb2a422c9b24a7306775f8b1bb25afa34bcde9cd98c9` |

All five mandatory execution files were present and read completely. The
legacy TeX hash matches the previously completed full source/PDF audit.

## Workspace and repository

- User: `codex` (`uid=1001`, `gid=1001`).
- Working directory: `/workspace/codex-home/dcdn/deep_division_network`.
- `/workspace` is a writable RunPod network-volume FUSE mount:
  `mfs#us-md-1.runpod.net:9421[/podvolumes/njm1qnnqwene/z3kvoe10pebc10]`.
- Project size at audit: 3.1 MiB; all of `/workspace`: 5.8 MiB.
- The mounted-filesystem `df` view reports 443,466,171,875,328 bytes total and
  120,448,792,199,168 bytes available. It is a shared backend view rather than
  control-plane quota metadata. Per the owner override, it is usable local
  capacity evidence; the project still enforces the stricter expected-volume
  limits of 40 GiB durable artifacts and at least 10 GiB free.
- Local Git repository: initialized on the persistent mount with branch
  `experiments/wsdm27` and SSH remote
  `git@github.com:rw404/DCDN.git`. The remote URL contains no credential.
- The pre-bootstrap inventory was persisted before `git init`; the initial
  commit remains gated on explicit staging plus credential and size checks.

## Compute

- GPUs: 2 x NVIDIA A100-SXM4-80GB, 81,920 MiB each, NV12 connectivity.
- GPU memory in use: 0 MiB on both devices; no compute applications.
- Driver: 580.126.16.
- Python: 3.12.3.
- PyTorch: 2.8.0+cu128; CUDA runtime build 12.8; cuDNN 91002.
- `/usr/local/cuda` resolves to CUDA 12.8.
- Host-visible CPU: 128 AMD EPYC 7742 cores; cgroup quota: 27.2 CPUs.
- Host-visible RAM: 2.0 TiB; cgroup limit: 499,999,997,952 bytes
  (approximately 465.7 GiB); no swap.
- Root overlay: 100 GiB, approximately 469 MiB used at audit.
- Measured 256 MiB sequential I/O (single checkpoint, cache effects possible):
  persistent write 520 MB/s, persistent read 3.4 GB/s; temporary-overlay write
  1.9 GB/s, temporary-overlay read 10.9 GB/s. Benchmark files were removed and
  absence was verified.
- No task-owned Python, torchrun, accelerate, deepspeed, raylet, or GPU jobs
  were active.

## Owner-approved local accounting override

- Provider-side auto-stop/expiry is explicitly waived and not configured as an
  agent prerequisite.
- Stored RunPod credentials must not be used.
- Control-plane price, historical spend, balance, and quota remain unavailable;
  no historical billing value is inferred.
- Estimated accounting uses USD 3.29 per elapsed Pod-hour plus a fixed USD 25
  conservative storage reserve.
- At `2026-08-18T11:59:30Z`, elapsed time was 2.770000 hours, estimated compute
  was USD 9.11, total estimated incremental spend was USD 34.11, warning
  headroom was USD 335.89, and hard-cap headroom was USD 425.89.
- The estimated warning threshold is reached at `2026-08-22T18:05:06Z`.

No RunPod credentials were used. The owner accepts responsibility for manually
terminating the Pod. The agent must never mutate Pod or volume lifecycle and
must finish with zero task-owned processes plus a verified durable backup.
