# Protocol v3 contract

## Purpose and current state

Protocol v3 is the fail-closed boundary between metric-independent
preparation, validation-only selection, and model execution. It binds a runtime
to exact data, views, feature algebra, split, schemas, denominator eligibility,
full-scan evidence, deterministic cache identity, encoder contract, capacity
resolution, search space, and experiment matrix.

The intentionally invalid repository template remains only a negative test
fixture. The actual VK-LSVD C+P premetric protocol is now materialized at
`configs/protocols/vk_lsvd_c_plus_p/protocol_frozen.yaml`, with SHA-256
`1919ba14e9a1f881fdf9b45016e16c4d0b92fc71d45944d329822e28816a5a0b`.
It binds the verified full scan, cache, encoder, capacity, physical/expanded/
operand schemas, and denominator mask, plus 7,505,820 train and 1,516,831
validation rows. Its active registry remains at the premetric stage and
`test_unlocked=false`; the per-model DCDN validation selection is not a final
dataset `selected_configs.lock` and does not authorize test access.

## Verified artifacts and sidecars

Every protocol-controlled YAML artifact is loaded through one verified byte
read:

1. reject path traversal and a symlink in the parent chain;
2. open the parent directory with `O_NOFOLLOW`;
3. open both source and sidecar with `O_NOFOLLOW`;
4. require regular files using `fstat`;
5. read the source bytes once;
6. hash those bytes, verify the already-open sidecar, and parse the same bytes;
7. return both the parsed mapping and verified digest.

The exact adjacent sidecar grammar is:

```text
<64 lowercase hex><two ASCII spaces><artifact basename><newline>
```

There is no path in the target field, no optional newline, and no alternative
spacing. Duplicate YAML keys are rejected at every depth. Unknown schema keys
are rejected. Registry updates use same-directory temporary regular files,
`fsync`, and `os.replace` while holding a durable exclusive lock. Because the
content and sidecar are two files, a crash can expose a torn pair; every reader
detects that state and fails closed.

## Premetric Protocol v3 schema

The root mapping has exactly these keys. A locked root may omit
`parent_premetric_protocol_sha256` or set it to `null`.

| Key | Exact rule |
|---|---|
| `schema_version` | integer `3` |
| `dataset_id` | resolved non-empty string |
| `feature_regime` | resolved non-empty string |
| `required_model_keys` | non-empty ordered list of unique resolved strings |
| `parent_premetric_protocol_sha256` | absent/null while locked |
| `test_unlocked` | Boolean `false` |
| `test_path_available` | Boolean `false` |
| `hashes` | exact base-hash mapping below |
| `split_row_counts` | positive integer `train` and `validation`; `test` absent/null |
| `final_seeds` | exact integer list `[42, 43, 44]` |
| `tuning_seed` | exact integer `42` |
| `tuning_splits` | exact list `[train, validation]` |
| `test_evaluations_per_final_seed` | exact integer `1` |

Each digest is 64 lowercase hexadecimal characters and cannot be a
repeated-character placeholder. The base `hashes` mapping contains exactly:

| Hash key | Independently bound fact |
|---|---|
| `data_manifest_sha256` | verified public-source/data manifest |
| `views_manifest_sha256` | materialized matched-views manifest |
| `features_sha256` | dataset feature contract |
| `split_sha256` | canonical singular views `split` mapping |
| `search_space_sha256` | frozen finite search space |
| `experiment_matrix_sha256` | frozen experiment matrix/model identities |
| `physical_schema_sha256` | exact physical input schema |
| `expanded_schema_sha256` | exact expanded feature schema |
| `operand_schema_sha256` | exact division-operand schema/order |
| `denominator_mask_sha256` | exact semantically valid denominator mask |
| `full_materialized_scan_sha256` | independent complete split scan |
| `cache_content_sha256` | deterministic cache-content identity |
| `encoder_contract_sha256` | shared trainable encoder contract |
| `capacity_resolution_sha256` | metric-independent capacity resolution |

`selected_configs_sha256` is forbidden in the premetric protocol.

## Canonical split derivation

`split_sha256` is derived only from the singular
`views_payload["split"]` mapping:

```python
canonical = json.dumps(
    recursively_validated_split_mapping,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
split_sha256 = hashlib.sha256(canonical).hexdigest()
```

There is no wrapper and no trailing newline. Recursive validation permits only
JSON values: mappings with string keys, lists, strings, exact integers,
finite floats, Booleans, and null. Tuples, sets, bytes, non-string mapping
keys, NaN, and infinities are rejected before serialization. The authoritative
APIs are `canonical_split_json_bytes`, `canonical_split_sha256`,
`split_sha256_from_views_payload`, and
`ProtocolGate.require_views_split`.

## Final child and immutable lineage

A final protocol is a new verified child, never an in-place mutation. It must:

- set `schema_version: 3`;
- set `parent_premetric_protocol_sha256` to the exact verified digest of its
  locked parent;
- set both test Booleans to true;
- add only `hashes.selected_configs_sha256`;
- record a positive full test row count.

Loading it requires
`ProtocolGate.load(final_path, parent_premetric_protocol_path=parent_path, ...)`.
The gate independently verifies both artifacts and requires exact equality of
all immutable root facts, every base hash, and train/validation row counts.
Only the state flags, positive test count, parent digest, and selected-lock
digest may be added or changed.

## Exact selected_configs.lock schema

The artifact must be named exactly `selected_configs.lock`, have its own strict
sidecar, and parse as exactly:

```yaml
schema_version: 2
artifact_role: final_selected_configs_lock
dataset_id: <same dataset>
feature_regime: <same regime>
parent_premetric_protocol_sha256: <parent protocol digest>
search_space_sha256: <same search-space digest>
experiment_matrix_sha256: <same matrix digest>
encoder_contract_sha256: <same encoder digest>
capacity_resolution_sha256: <same capacity digest>
required_model_keys: [<exact ordered protocol model keys>]
selected_models:
  <model key>:
    candidate_id: <resolved non-empty candidate identity>
    resolved_config_sha256: <exact resolved config digest>
    selection_receipt_path: <repository-relative validation selection receipt>
    selection_receipt_sha256: <exact receipt digest>
    final_seed_lineage:
      "<each exact key: 42, 43, and 44>":
        resolved_run_config_path: <configs/runs/*.yaml>
        resolved_run_config_sha256: <exact run config digest>
        validation_result_path: <results/*.json>
        validation_result_sha256: <exact validation result digest>
        best_checkpoint_path: <runs/*.pt>
        best_checkpoint_sha256: <exact selected best digest>
        selected_validation_metric: <finite non-negative value>

    selected_on_validation_only: true
    test_data_accessed: false
    test_metrics_consulted: false
```

`selected_models` must contain exactly one entry for every
`required_model_keys` item and no other model. No root or per-model extension
keys are allowed. `final_seed_lineage` must contain all three exact string keys
`42`, `43`, and `44`, and no others. `SelectedConfigsLock.require_for_protocol` checks every
binding and the exact artifact digest; `require_model` binds a test request to
the current model, candidate, and seed-invariant resolved configuration.
`require_final_seed` additionally binds seed 42/43/44 to its immutable run
config, validation result, selected metric, and exact best-checkpoint SHA before
a central test claim can be reserved.

## Persistent activation state

Every gate used for execution must be loaded with one persistent absolute
registry path named `active_protocol.yaml`. Merely loading a valid protocol
does not authorize a split.

The only transitions are:

```text
(no active state) -> active premetric parent -> active final child
```

The premetric gate is activated first:

```python
parent_gate = ProtocolGate.load(
    parent_path,
    active_registry_path=active_protocol_path,
)
parent_gate.activate()
```

Final activation requires that exact parent digest to still be active and
requires the verified selected lock plus the central persistent claim path:

```python
final_gate = ProtocolGate.load(
    final_path,
    parent_premetric_protocol_path=parent_path,
    active_registry_path=active_protocol_path,
)
final_gate.activate(
    selected_configs_path=selected_path,
    test_claim_registry_path=central_test_claim_registry_path,
)
```

The claim path must be absolute and named exactly
`test_claim_registry.yaml`. It is pinned in the active state. Activation is
idempotent only for the exact same state. Replacement and rollback are
forbidden. Every `require_split` call re-verifies the active-state bytes and
digest. Consequently, activating the final child immediately invalidates the
old parent's train/validation gate, and the final gate itself never authorizes
train or validation.

## Complete runtime identity API

Every allowed split request supplies one independently constructed
`RuntimeIdentity` with exactly these constructor arguments:

```python
RuntimeIdentity(
    dataset_id=...,
    split=...,
    feature_regime=...,
    row_count=...,
    data_manifest_sha256=...,
    views_manifest_sha256=...,
    features_sha256=...,
    physical_schema_sha256=...,
    expanded_schema_sha256=...,
    operand_schema_sha256=...,
    denominator_mask_sha256=...,
    full_materialized_scan_sha256=...,
    cache_content_sha256=...,
    encoder_contract_sha256=...,
    capacity_resolution_sha256=...,
)
```

`dataset_id`, requested `split`, `feature_regime`, and the full split row count
are compared with root facts. Every supplied runtime hash is compared with the
same-named base hash. Missing identity is an error. The values must come from
independent preflight/SafetyAttestation evidence, not be populated from a
source object's self-report.

Search-space and experiment-matrix identities are protocol/selection facts,
not batch-source fields: SafetyAttestation must still compare them before
model execution, and the selected lock repeats their exact binding. The
canonical views split must separately pass `require_views_split`.

## Split authorization and SafetyAttestation ordering

For train or validation, authorization occurs before constructing a source
factory:

```python
parent_gate.require_split(
    requested_split,
    runtime_identity=attested_runtime_identity,
)
# Only now construct/open the train or validation source.
```

After factory construction, independently compare the source-reported identity
with the same attested `RuntimeIdentity` again. The encoder instance/spec and
mechanically resolved capacity control must likewise be compared with
`encoder_contract_sha256` and `capacity_resolution_sha256`. This second check
does not replace the pre-factory gate.

For test, `require_split` atomically reserves the exact cell and returns an
opaque tokenized `TestEvaluationClaim`:

```python
reserved = final_gate.require_split(
    "test",
    runtime_identity=attested_test_identity,
    selected_configs_path=selected_path,
    model_key=current_model_key,
    candidate_id=current_candidate_id,
    resolved_config_sha256=current_resolved_config_sha256,
    resolved_run_config_sha256=current_resolved_run_config_sha256,
    checkpoint_sha256=selected_best_checkpoint_sha256,
    seed=current_seed,
    checkpoint_protocol_sha256=parent_gate.digest,
)
```

Reservation does not authorize opening the test source. It durably binds the
dataset, regime, model, seed, final protocol, parent protocol, selected-lock
digest, selection-receipt digest, seed-invariant model config, seed-specific run
config and validation result, exact selected metric and best-checkpoint SHA,
and the complete test `RuntimeIdentity`. An exact duplicate request can recover
the same token only while it remains `reserved`. Binding drift is rejected.

Immediately before invoking the source factory, consume the at-most-once
opening permission:

```python
opened = reserved.mark_opened(
    gate=final_gate,
    selected_configs_path=selected_path,
    runtime_identity=attested_test_identity,
)
test_source = test_factory()  # the immediately following operation
```

`mark_opened` re-verifies the active final gate, selected lock, runtime
identity, every stored binding, central path, registry sidecar, and token under
the exclusive registry lock. A second open transition is forbidden. Protocol
v3 deliberately has no automatic opened-state resume API.

Only after the full result receipt has been durably written may the harness
persist completion:

```python
completed = opened.mark_completed(
    gate=final_gate,
    selected_configs_path=selected_path,
    runtime_identity=attested_test_identity,
    result_receipt_sha256=durable_result_receipt_sha256,
    result_row_count=attested_test_identity.row_count,
)
```

The receipt digest must be a real lowercase SHA-256 and the result row count
must equal the frozen full test count. Repeating the same completion call is an
idempotent acknowledgement of the already durable receipt; a different
receipt is rejected.

The checkpoint remains bound to the parent premetric protocol throughout all
three states. It is not re-saved, re-tuned, or relabeled under the final child.
The active final gate supplies evaluation-only authorization.

## Central test lifecycle registry

The central `test_claim_registry.yaml` uses lifecycle schema version 3. Its
source and strict sidecar are verified from the same opened file descriptors.
Every reservation or transition runs under the same central exclusive lock,
loads the verified registry bytes, validates the exact schema and all existing
records, and atomically replaces the source/sidecar pair. A torn pair or
binding mismatch fails closed.

Each record has an opaque unique 64-hex `claim_id`, one of the exact states
`reserved`, `opened`, or `completed`, and preserves:

- model key and final seed;
- final and parent premetric protocol digests;
- selected-config lock digest;
- validation-only selection-receipt digest;
- candidate ID and seed-invariant resolved-model digest;
- seed-specific resolved run-config and validation-result digests;
- exact selected validation metric and parent-bound best-checkpoint digest;
- the complete test `RuntimeIdentity`;
- nullable completion evidence while reserved/opened;
- a required receipt digest and full result row count when completed.

The cell key is:

```text
(dataset_id, feature_regime, model_key, seed, final_protocol_sha256)
```

The lifecycle guarantees at-most-once test opening and exactly one durable
completion receipt, not exactly-once execution:

- crash before `mark_opened`: the exact reservation is safely recoverable with
  the same claim token;
- crash after `mark_opened`: the cell remains opened, counts as an execution
  failure, and is not automatically rerun;
- crash after durable completion but before acknowledgement: the exact same
  receipt/count can be acknowledged idempotently;
- an opened or completed cell rejects another reservation/evaluation;
- concurrent exact reservations recover one token, while concurrent opening
  attempts permit only one transition.

The active state pins one absolute central registry path, so a run-local
registry cannot bypass the lifecycle. A completed receipt is evidence of one
full evaluation, not permission to tune, retrain, inspect another model, alter
the checkpoint, or evaluate the cell again.
