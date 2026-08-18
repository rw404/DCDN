# Public data source audit

Initial audit checkpoint: `2026-08-18T12:03:43Z`

No HEGM, VK-LSVD, KuaiRand-Pure, or KuaiRec data, archives, manifests,
Parquet, Arrow, Feather, or CSV files were present under `/workspace`,
`/home/codex/.cache`, or `/tmp` at the initial checkpoint. No data was
downloaded during that audit. The later verified acquisition checkpoint below
supersedes that initial local-presence statement without changing its
historical observation.

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

## Verified acquisition checkpoint

Checkpoint: `2026-08-18T14:14:40Z`.

- KuaiRand-Pure archive: 47,432,272 bytes, SHA-256
  `c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0`;
  manifest SHA-256
  `bc5b441fdd3188d033e0ecbf0bdfd73f8bcf07bbc744d474d260ee220ae4d6e9`.
- KuaiRec archive: 431,964,858 bytes, SHA-256
  `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`;
  manifest SHA-256
  `963019bcb4c8dbd4663f23c24a472fe0332f6494fdcbbaa74ca158a73488c5b4`.
- VK-LSVD revision
  `084a18386e149a454dd6dbe0123478e75c6b09ce`: exactly 30 objects and
  2,843,350,925 bytes; manifest SHA-256
  `5dbbfeb23442a6b1ebdb829dc239c6a3c6c655d71446cde4551b83cc6af456ac`.
  Week 26 was acquired last, hash-verified, set read-only, and never decoded.

Raw bytes remain only under `/home/codex/.cache/dcdn-public`. The durable
repository contains source-only manifests and sidecars; all three retain
`test_unlocked: false`, and no model metric was computed.
