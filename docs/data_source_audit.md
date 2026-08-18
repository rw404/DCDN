# Public data source audit

Audit checkpoint: `2026-08-18T12:03:43Z`

No HEGM, VK-LSVD, KuaiRand-Pure, or KuaiRec data, archives, manifests,
Parquet, Arrow, Feather, or CSV files were present under `/workspace`,
`/home/codex/.cache`, or `/tmp`. No data was downloaded during this audit.

## Frozen source routes

### HEGM

- Source: `https://github.com/rw404/HEGM.git`
- Pinned commit: `d16c097d57f867f438f906949d9cc1efd0f6ddc4`
- Declared license: Apache-2.0.
- Released processed splits and checkpoints are excluded. They do not support
  the strict matched `C+P` versus `C+P+E/R` reconstruction and contain
  test-facing artifacts.

### VK-LSVD

- Source: `https://huggingface.co/datasets/deepvk/VK-LSVD`
- Declared license: Apache-2.0.
- Whole-repository snapshots are prohibited because the repository is about
  452 GB.
- Allowlist: the `up0.01_ir0.01` interactions plus only the required
  metadata.

| Object | Bytes | SHA-256 |
|---|---:|---|
| `item_embeddings.npz` | 2,362,338,243 | `6a4024ad7623f5d8150e59c5faaaf42fb7d7f48e2beec44c6917e5e57debea7c` |
| `items_metadata.parquet` | 155,747,843 | `607ffd09d3fb6627f7eec07ed53583d45c47263a2e6538e4deee66cd8a22e51e` |
| `users_metadata.parquet` | 72,526,148 | `51fa9528f71bb0da947311d7baa4727d325285a3e748b54816477b82492aff20` |
| sealed week-26 file | 8,330,983 | `e540f80917fa7e065a9905a47611955820dddb8fcd7f45385ccb53d6a961fdb7` |

The allowlisted interaction payload is approximately 253 MB; required
metadata totals 2,590,612,234 bytes.

### KuaiRand-Pure

- Source:
  `https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz`
- Size: 47,432,272 bytes.
- MD5: `0820331067a3784d9691136f772b35a7`.

### KuaiRec

- Source: `https://zenodo.org/records/18164998/files/KuaiRec.zip`
- Size: 431,964,858 bytes.
- MD5: `261550d472c48eff4990fb13c0e5bcf7`.

The KuaiRand and KuaiRec GitHub repositories declare CC-BY-SA-4.0 while the
Zenodo records declare CC-BY-4.0. Preserve both provenance records, follow the
more conservative CC-BY-SA-4.0 terms, and never commit or redistribute raw
data.

## Storage routing

- Regenerable public data:
  `/home/codex/.cache/dcdn-public/{downloads,raw,processed,shards,scratch}`.
- Keep at least 20 GiB free on the 100 GiB overlay and stop new materialization
  before 85% usage.
- Persist only manifests, hashes, licenses, row counts, protocol/config locks,
  compact registries/logs/results, and selected best checkpoints.
- Keep project-owned durable artifacts at or below 40 GiB and preserve at
  least 10 GiB headroom.
