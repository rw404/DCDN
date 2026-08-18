# Frozen public raw acquisition

This checkpoint records source provenance only. It does not construct a
feature view or split, unlock a test set, read test labels, or compute a
validation/test metric. Public raw bytes remain regenerable ephemeral data
under `/home/codex/.cache/dcdn-public`; they are neither copied into the
persistent project area nor tracked by Git.

## Common manifest contract

All three `data/manifests/*_source_v1.yaml` records use schema version 1 and
the following fail-closed top-level state:

- `artifact_role: raw_source_provenance_only`;
- `acquisition_state: raw_download_verified` and
  `raw_download_verified: true`;
- `test_unlocked: false`, `values_not_scanned: true`,
  `test_labels_opened: false`, and
  `validation_or_test_metrics_read: false`;
- unauthenticated public read-only acquisition with no credentials or
  third-party telemetry;
- ephemeral raw storage, no persistent raw copy, and no committed raw data.

The common loader rejects duplicate YAML keys, unsafe filenames, symlinks,
hardlinks, oversized manifests/sidecars, and any noncanonical sidecar. The
writer serializes deterministically, fsyncs random same-directory temporary
files, atomically replaces the manifest and sidecar, fsyncs their directory,
and removes unfinished temporaries. A crash between the two replacements is
detectable as a sidecar mismatch.

## Verified sources

### KuaiRand-Pure

- Official frozen Zenodo object: `47432272` bytes.
- MD5: `0820331067a3784d9691136f772b35a7`.
- SHA-256:
  `c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0`.
- CSV access was limited to each bounded first physical header line; no row
  value was scanned.
- Manifest SHA-256:
  `bc5b441fdd3188d033e0ecbf0bdfd73f8bcf07bbc744d474d260ee220ae4d6e9`.

### KuaiRec big matrix source

- Official frozen Zenodo object: `431964858` bytes.
- MD5: `261550d472c48eff4990fb13c0e5bcf7`.
- SHA-256:
  `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`.
- ZIP validation rejects traversal, absolute/drive/backslash paths, links,
  special or encrypted entries, normalized case collisions, file/parent
  collisions, size overflow, and excessive compression ratio before staged
  extraction. CSV access is limited to the first bounded header line.
- Manifest SHA-256:
  `963019bcb4c8dbd4663f23c24a472fe0332f6494fdcbbaa74ca158a73488c5b4`.

### VK-LSVD `up0.01_ir0.01`

- Frozen revision: `084a18386e149a454dd6dbe0123478e75c6b09ce`.
- Exactly 30 allowlisted objects totaling `2843350925` bytes were locally
  size- and SHA-256-verified without Parquet or NumPy decoding.
- Week 26 was downloaded last, verified only by bytes and hash, made read-only,
  and remains sealed from schema, row, and label access.
- The other 29 objects are marked `not_decoded`; their presence does not
  unlock any split or metric.
- Manifest SHA-256:
  `5dbbfeb23442a6b1ebdb829dc239c6a3c6c655d71446cde4551b83cc6af456ac`.
