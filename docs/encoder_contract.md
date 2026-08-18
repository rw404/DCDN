# Trainable shared encoder contract

Status: frozen before model metrics on 2026-08-18. Test access remains locked.

## Purpose and boundary

The materialized public views contain ordered categorical token IDs `C`,
ordered scale-normalized numeric operands `P` (and `E/R` only in the declared
full-feature control), and a per-row denominator-validity mask. Embeddings are
never materialized or cached. `src/dcdn/encoder.py` is the trainable boundary
between `RawFeatureBatch` and `EncodedInteractionBatch`.

The encoder forward signature contains only `context_tokens`,
`operand_features`, and `denominator_validity`. Labels, stable row IDs,
timestamps, policies, and other audit fields cannot enter it. The joint
`TrainableEncodedModel` owns the encoder and the controlled
interaction/adapter/head, so a single optimizer and `state_dict` cover every
trainable parameter.

## Frozen architecture

- Each ordered categorical field has a separate `nn.Embedding` table.
  Cardinalities include trainable OOV token 0 and MISSING token 1.
- Every embedding dimension is 16. All embedding weights use an independent
  SHA-256-derived CPU seed and `Normal(0, 0.01)` initialization.
- `bypass_features` is the flattened ordered categorical embedding view.
- `operand_features` is the original numeric tensor object, unchanged.
- The field view is ordered categorical embeddings followed by, for numeric
  field `j`, `operand[j] * numeric_embedding[j]`. There is no numeric bias.
  Numeric field vectors live in one `nn.Embedding` table, so the frozen
  capacity counter excludes them as embedding capacity.
- The denominator-validity tensor is passed through unchanged. Only DCDN-Core
  consumes it; other model families receive but ignore it.
- AFN+ alone receives a second fully independent categorical/numeric embedding
  view of the identical fields. The primary view seed excludes the model key,
  and is therefore byte-identical across model families for a fixed run seed
  and schema. The AFN secondary view uses a separate seed domain.
- Parameters remain FP32. FP32 and BF16 inputs are accepted; FP16 inputs are
  rejected. BF16 autocast is permitted by the surrounding model harness.

No encoder width, initializer, field choice, seed, or treatment was selected
from a validation or test model metric. The byte lock is
`configs/encoder_v1.yaml` plus its SHA-256 sidecar. Critical YAML fields are
also checked against independent constants in code, so merely changing and
rehashing the YAML cannot alter the architecture, seed domains, input policy,
runtime identity policy, or frozen VK identity. The YAML and exact sidecar are
opened relative to one verified parent-directory descriptor with
`O_NOFOLLOW`; `fstat` requires regular single-link files, and the SHA-256
check and YAML parser consume the exact same captured bytes. A symlink or
in-open path replacement therefore fails closed and cannot swap different
bytes between verification and use.

## Runtime value-check modes

Shape, dtype, device, order, FP32-parameter, and identity checks are mandatory
in every mode. Debug/smoke mode (`runtime_value_checks=true`, the safe
default) additionally evaluates finite-value and categorical-token predicates.
Those predicates may synchronize a GPU and therefore are forbidden in the
optimized mode. Optimized mode is available only when the independently frozen
full-scan and deterministic cache-content attestations are both present and
match the `RawSourceIdentity`; its path contains no `torch.isfinite`,
`Tensor.any`, `Tensor.item`, or tensor-to-Python-bool call. The raw source
owns one frozen `runtime_value_checks` flag and exposes it through the
`RawBatchSource` protocol. It passes the flag unchanged to
`RawFeatureBatch.validate`. The harness must read that property and construct
the encoder with the exact same mode; optimized mode is an explicit `false`,
never an inferred default. The selected mode and its attestation state are
bound into checkpoint metadata.

## Verified VK-LSVD resolution

The currently resolved dataset is `vk_lsvd_up0.01_ir0.01`:

- ordered C: `age, gender, geo, place, platform, agent`;
- cardinalities: `55, 4, 82, 24, 13, 22`;
- C+P: 58 numeric operands, 64 model fields, 96-dimensional categorical
  bypass, and 154-dimensional flat view;
- C+P+E: 78 numeric operands, 84 model fields, and 174-dimensional flat view;
  this regime is restricted to the full-rank DCNv2 engineered-feature arm.

The schema/order/mask values were adapted from the verified feature and view
manifests before any model metric. The P+E operand hash is the SHA-256 of
compact ASCII JSON over the exact ordered P-then-E name array. Runtime v2 adds
one paired full-materialized-scan attestation and two distinct cache-content
attestations, one for C+P and one for C+P+E. Cache-content hashes cover a
canonical JSON payload containing dataset/regime/upstream identity, exact
schema/order/cardinalities, split row counts, and ordered raw tensor byte
digests; they are not cache-metadata or file-byte hashes.

The independently frozen identity is:

- source manifest: `5dbbfeb23442a6b1ebdb829dc239c6a3c6c655d71446cde4551b83cc6af456ac`;
- views manifest: `738d3610fd83a93297d94ccad4c4316dd3ad8b241f8409e84daf5b2322040dbe`;
- paired full scan: `cc7c74b183688c1125cd3e49d32daed2b1e7560d9ee796cfe93de972792ed612`;
- C+P cache content: `b732b95f80f9ec28f4d9dd5192bf7dba3ed16bb96093bc5add209743edfff590`;
- C+P+E cache content: `34f415bde13633adc6994e22548e9aa44817a586d6d74350c93e859835c9d5d7`.

Resolution is fail-closed against `RawSourceIdentity`: dataset, feature
regime, source/views/feature provenance, physical/expanded/operand/mask
hashes, full scan, per-regime cache content, exact C/P order, and cardinalities
must all match. `DatasetEncoderSpec` is always constructed from lock values;
input identity is asserted against it and is never copied into expected
provenance. Scan/cache content digests must be mutually distinct and distinct
from every metadata/schema digest.

KuaiRand-Pure and KuaiRec remain unresolved in this encoder lock until their
verified materialized views provide exact train-fitted categorical
cardinalities plus canonical full-scan and deterministic cache-content
attestations. The architecture itself can be reused only after a new
independently frozen dataset resolution is added to both code and YAML.

## Initialization and checkpoint binding

Primary, AFN-secondary, and per-table seeds are hash-derived from the run seed,
dataset encoder contract, schema identity, view name, and table identity.
Construction uses isolated CPU generators, does not touch CUDA RNG state, and
does not depend on which interaction model was constructed first.

Both the encoder and the joint module implement strict PyTorch extra-state
metadata. A checkpoint binds:

- encoder YAML SHA-256 and resolved dataset-contract SHA-256;
- feature regime, exact ordered C/P names, cardinalities, and all schema hashes;
- data/views/full-scan and per-regime cache-content provenance hashes;
- debug/smoke versus optimized runtime-value-check mode;
- primary and optional AFN-secondary initialization seeds;
- controlled interaction/adapter/head contract and initialization seeds.

Strict loading fails when any bound metadata differs.

## Verification

Targeted verification is run from a fresh `/tmp` snapshot, never concurrently
with another pytest or Ruff process:

```bash
PYTHONPATH=src /tmp/dcdn-venv-py312/bin/python -m pytest -q \
  tests/test_vk_lsvd_runtime.py tests/test_shared_encoder.py
/tmp/dcdn-venv-py312/bin/python -m ruff check \
  src/dcdn/raw.py src/dcdn/data/vk_lsvd_runtime.py src/dcdn/encoder.py \
  tests/test_vk_lsvd_runtime.py tests/test_shared_encoder.py
/tmp/dcdn-venv-py312/bin/python -m ruff format --check \
  src/dcdn/raw.py src/dcdn/data/vk_lsvd_runtime.py src/dcdn/encoder.py \
  tests/test_vk_lsvd_runtime.py tests/test_shared_encoder.py
```

The tests cover shapes and order, raw-P and mask identity, invalid tokens and
dtypes, FP16 rejection, BF16 views with FP32 parameters, gradients through
OOV/MISSING and numeric embeddings, capacity exclusion, byte-identical primary
initialization across variants, AFN independence, joint optimizer/checkpoint
resume, label/audit non-consumption, full-arm restriction, rehashed-YAML
mutation rejection, exact source/cache identity binding, and a monkeypatched
proof that optimized raw/encoder execution does not call finite/any/item/bool
device-sync predicates. They also cover encoder-lock symlink rejection and
replacement during parse, proving that the returned lock was parsed from the
already hash-verified bytes.

## Integration status

The raw harness integration is complete: the verified `RawBatchSource` feeds
the complete `TrainableEncodedModel`; the source-owned
`runtime_value_checks` mode is passed unchanged; encoder, interaction,
adapter, and head are optimized and checkpointed together; and embedding
capacity remains excluded by the frozen counter. Deterministic smoke/resume
verification passed, followed by the two frozen seed-42 DCDN-Core C+P
validation-only LR candidates. This integration still does not authorize test
access: the active VK protocol remains premetric with `test_unlocked=false`,
and the per-model DCDN selection is not the dataset-wide selected-config lock.
