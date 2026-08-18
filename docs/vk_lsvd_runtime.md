# VK-LSVD full-scan attestation and raw runtime sources

This pre-metric boundary connects the frozen VK-LSVD feature views to the
common model encoder. It consumes only manifest-listed train weeks 20--24 and
validation week 25. Every relative path component is checked for sealed-test
markers. There is no test source, no week-26 path construction, and no model
forward pass or validation/test metric in this layer.

## Final frozen identities

- Source/data manifest SHA-256:
  `5dbbfeb23442a6b1ebdb829dc239c6a3c6c655d71446cde4551b83cc6af456ac`
- Feature contract SHA-256:
  `380d105c847a73afb4927740e40401124688f895142792309b4fc8267a2a5f2b`
- Materialized views manifest SHA-256:
  `738d3610fd83a93297d94ccad4c4316dd3ad8b241f8409e84daf5b2322040dbe`
- Canonical paired full-scan SHA-256:
  `cc7c74b183688c1125cd3e49d32daed2b1e7560d9ee796cfe93de972792ed612`

The scan digest is the sidecar digest of
`data/manifests/vk_lsvd_full_scan_v1.yaml`. The file name preserves the
feature-view contract version; its algorithm version is
`vk-lsvd-full-materialized-scan-v2`. The digest is deliberately distinct
from the views digest and is used as
`RawSourceIdentity.full_materialized_scan_sha256`.

The exact per-regime schema bindings are:

| Regime | Operand schema SHA-256 | Denominator-mask SHA-256 |
| --- | --- | --- |
| C+P | `bb2f6c47fa12c175a3820830f90ff577acb293c56b107c7d1c40e4c682f2ff4d` | `9e168fae4014f4763008e89f61324858f17d376f56fb5de46f2775911650bcaa` |
| C+P+E | `3b34b22f0779dd059e89016b388952862da074c5bfa2534cc7ba251569ba54e3` | `8bca04940fe8b7e6cf0d2f8e100db0787cc32e066c6c9eadd0ba711652b37715` |

The C+P+E order is exactly P followed by E. Its 78-coordinate mask is the
58-coordinate P mask followed by 20 false E entries.

## Full-scan evidence

The scanner independently byte-hashed and streamed all six matched shard
pairs, 12 Parquet files in total, in exact week/role order. For 7,505,820
train rows and 1,516,831 validation rows it verified:

- exact physical, expanded, operand, and mask schema identities for both
  regimes;
- exact shard bytes, hashes, row counts, row groups, and ordering;
- exact paired stable-row-ID and shared C+P value hashes;
- context-token bounds and cardinalities `(55, 4, 82, 24, 13, 22)`;
- finite FP32 values for all 58 P fields, 20 E fields, and both targets;
- normalized target bounds `[0, 1]` and the exact seconds transform;
- full P validity semantics: non-denominators are always valid, active
  denominators are strictly positive, inactive values are canonical zero, and
  reciprocal-floor indicators are exact;
- disjoint model-input and audit/target columns.

The loader validates the complete attestation structure rather than trusting
its sidecar alone: exact top-level fields, scope, six shards and 12 file
hashes, row totals, schema maps, token/numeric bounds, normalization
provenance, denominator evidence, and the frozen historical scan-generator
source SHA-256
`40edf35afc0cddac1ae93c924337985579df96c13270bc2cb6a973397961f5b1`.
This generator attestation is intentionally distinct from the current loader
source: the same-FD loader hardening changes neither scan/materialization logic
nor any scientific output, so the canonical scan SHA-256
`cc7c74b183688c1125cd3e49d32daed2b1e7560d9ee796cfe93de972792ed612`
remains
valid and unchanged. A production open additionally requires a caller-supplied
independently frozen scan digest, preventing a modified payload and sidecar
from self-authorizing. This is structural and numerical safety evidence, not a
model result.

## Runtime cache v2

Both caches are ephemeral, atomic, non-reusable outputs. The prior
`c_plus_p_v1` cache remains present only as a superseded rollback artifact;
its old scan SHA
`da9523f89c92454eacbff9eebd85b82c0b2a423925b70b52d41b5945e3ca1c02`
and metadata SHA
`72bb7686cfeafa36fb3ff8f83f5979c5fed2a364a606683da00cd9c036c70169`
must not be used by v2 identities.

| Regime | Cache root | Bytes | Cache-content SHA-256 | Metadata SHA-256 |
| --- | --- | ---: | --- | --- |
| C+P | `/home/codex/.cache/dcdn-public/runtime/vk_lsvd/c_plus_p_v2` | 2,869,216,007 | `b732b95f80f9ec28f4d9dd5192bf7dba3ed16bb96093bc5add209743edfff590` | `5c853147abc709e38e594a00358571d4aaba266feaf5ca73368f587c9f63d6a5` |
| C+P+E | `/home/codex/.cache/dcdn-public/runtime/vk_lsvd/c_plus_p_plus_e_v2` | 3,771,482,126 | `34f415bde13633adc6994e22548e9aa44817a586d6d74350c93e859835c9d5d7` | `920458d42418eea203bd7b814ec6eddf5bc3b174ce109724ced7bceb2ffdc4d2` |

Each split stores four read-only C-order NPY tensors: context `<i4`,
operands `<f4`, validity `|u1`, and target `<f4`. The cache manifest
records both complete NPY-file hashes and raw contiguous tensor-byte hashes.
The content digest is SHA-256 over canonical ASCII JSON
(`sort_keys=True`, compact separators) containing immutable upstream/schema
identity and ordered records
`{name,dtype,shape,tensor_nbytes,tensor_bytes_sha256}`. Build timestamps,
elapsed time, storage guards, file headers, and the self-digest are excluded.

A production source open requires independent expected scan and cache-content
digests, recomputes every NPY file and tensor hash, and refuses
`verify_hashes=False`. Manifests and sidecars are opened relative to verified
directory descriptors with `O_NOFOLLOW`; regular-file identity is checked by
`fstat`, and hashes and parsers consume the same captured bytes. Each NPY is
opened and hashed from one verified inode, then exposed through a read-only
FD-backed mapping retained for the source lifetime. A later symlink or path
replacement therefore fails closed during open or leaves the source consuming
the already verified inode; it never silently reopens mutable paths. Train and
validation identities for one regime differ only in `split` and `row_count`;
their scan and content roots are identical.

All four real source opens preserved the requested duplicate/non-monotone
boundary order exactly. Full scan/cache hash validation plus open took about
4.04 seconds for C+P and 5.22 seconds for C+P+E on this host. The mapped files
do not become resident eagerly. Conservative host-plus-returned-tensor staging
at the hard `MAX_BATCH_ROWS=262144` boundary is about 165 MiB for C+P and
215 MiB for C+P+E, plus framework/page-cache overhead.

`batch(indices, device)` requires a bounded `Sequence`, checks its length
before tuple/index materialization, rejects empty, generator, oversized,
non-integer, and out-of-range requests, and preserves duplicates/order. It
returns context int64 `[B,6]`, FP32 operands `[B,58|78]`, bool validity of
the same width, and FP32 target `[B,1]` on the requested device. Global row
indices remain CPU int64. Sources are process-local and reject inherited use
after a fork. Each source owns one frozen `runtime_value_checks` flag and
exposes it to the harness: the safe debug default is `true`, while optimized
execution must select `false` explicitly. The flag is passed unchanged to
`RawFeatureBatch.validate`; the optimized raw-batch path performs structural
checks without tensor value predicates or device synchronization.

## Reproduction

Arguments shared by all commands precede the subcommand. Every material
operation repeats frozen local cost/time/filesystem guards.

```bash
PYTHONPATH=src python scripts/data/vk_lsvd_runtime.py \
  --eta-hours 2 scan

PYTHONPATH=src python scripts/data/vk_lsvd_runtime.py \
  --eta-hours 2 build-cache --feature-regime 'C+P' \
  --expected-scan-sha256 cc7c74b183688c1125cd3e49d32daed2b1e7560d9ee796cfe93de972792ed612

PYTHONPATH=src python scripts/data/vk_lsvd_runtime.py \
  --eta-hours 2 build-cache --feature-regime 'C+P+E' \
  --expected-scan-sha256 cc7c74b183688c1125cd3e49d32daed2b1e7560d9ee796cfe93de972792ed612

PYTHONPATH=src python scripts/data/vk_lsvd_runtime.py \
  --eta-hours 2 verify-cache --split train --feature-regime 'C+P' \
  --expected-scan-sha256 cc7c74b183688c1125cd3e49d32daed2b1e7560d9ee796cfe93de972792ed612 \
  --expected-cache-content-sha256 b732b95f80f9ec28f4d9dd5192bf7dba3ed16bb96093bc5add209743edfff590
```

The final combined runtime-and-encoder gate was 64 tests from a fresh `/tmp`
mirror with targeted Ruff clean. Runtime coverage includes both regimes,
independent pinning, self-signed scan/cache mutations, raw-byte drift, exact
structure, forbidden paths, output reuse, arbitrary order/duplicates, the
pre-materialization batch-size boundary, symlink rejection, post-verification
path replacement, and predicate-free optimized raw validation.

## Integration boundary

`DatasetEncoderSpec.assert_source_identity` must independently pin the exact
source, views, paired-scan, and per-regime cache-content roots above. The
training harness must then instantiate the appropriate source, validate
its identity, read the source-owned `runtime_value_checks` property, pass that
exact mode to the encoder, encode each raw batch, and remove its synthetic-only
execution assertion. The encoder now enforces the same identity and runtime-mode
contract; harness wiring remains the integration boundary.

No test source exists and none is synthesized. Test evaluation remains
blocked until the separate protocol-authorized unlock after validation-only
tuning and frozen configuration hashes.
