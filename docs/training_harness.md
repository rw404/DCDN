# Deterministic training harness

This harness is the fail-closed train/validation boundary for the frozen DCDN
public suite. It now accepts bounded-memory `RawBatchSource` factories and a
model-owned `TrainableEncodedModel`; it never accepts a precomputed public
embedding bundle. The legacy encoded-tensor path remains synthetic-only for
backward determinism tests. No real dataset, GPU, model metric, test claim, or
protocol activation was used while implementing or testing this integration.

## Premetric execution order

Every real source is label-bearing, so authorization happens before its
factory is called:

1. Verify the active immutable Protocol v3 artifact and its sidecar. For the
   capacity control only, also reopen the validation-only ordinary-DCNv2
   selection receipt and the compact capacity-LR authorization before model,
   optimizer, or source creation.
2. Verify exact independently declared train and validation `RuntimeIdentity`
   values, including row counts and all data, views, feature, schema, mask,
   full-scan, cache, encoder, and full-capacity-artifact hashes.
3. Verify that the corresponding `RawSourceIdentity` values project to those
   runtime identities and differ between train and validation only by `split`
   and `row_count`.
4. Verify the exact views payload and canonical split hash, frozen search and
   experiment-matrix hashes, leakage attestations, and local cost/storage
   guards.
5. Call `SafetyAttestation.require_split(split, config)`.
6. Only then call the source factory; compare the returned identity
   byte-for-byte, check its length and runtime-check mode, and bind it to the
   encoder.
7. For every requested index sequence, call `RawFeatureBatch.validate` before
   the model forward. Always close the source in a `finally` block.

The optimized path sets `runtime_value_checks: false` consistently in the
resolved config, source, encoder, and DCDN core. It relies on the bound full
materialized scan and cache-content hashes and performs no per-batch tensor
value synchronization to the host. The debug path sets the flag to true and
performs finite/token-range checks. Vectorized device-side safe-reciprocal
masks remain part of the DCDN arithmetic in both modes. Resolved-config schema
v3 makes that boundary explicit with the required top-level field
`execution_kind`. `scientific_tuning` requires `runtime_value_checks: false`;
resume, a later effective-batch stop boundary, and optional validation latency
retain their existing scientific semantics. `implementation_smoke` requires
`runtime_value_checks: true`, `resume: false`, `stop_after_examples: 4096`
(exactly one frozen effective batch), and `latency: null`. Its validation
factory is a fail-closed sentinel and therefore cannot open validation or emit
validation/test metrics or a selected best checkpoint. The result is
permanently labeled `scientific_result_eligible: false`; it is implementation
evidence only, never a scientific result or a long run.

## Raw VK-LSVD boundary

The production source is
`dcdn.data.vk_lsvd_runtime.VKLSVDRawBatchSource`, opened through its factory
with expected full-scan and cache-content hashes and `verify_hashes=True`.
For the frozen VK-LSVD C+P v2 view, the relevant identities include:

- views manifest:
  `738d3610fd83a93297d94ccad4c4316dd3ad8b241f8409e84daf5b2322040dbe`;
- full materialized scan:
  `cc7c74b183688c1125cd3e49d32daed2b1e7560d9ee796cfe93de972792ed612`;
- C+P cache content:
  `b732b95f80f9ec28f4d9dd5192bf7dba3ed16bb96093bc5add209743edfff590`.

The source streams the materialized cache rather than loading the 12 Parquet
shards into one CPU tensor. It preserves the manifest row order: train is
weeks 20--24 in manifest order and validation is week 25. Arbitrary,
nonmonotone, and duplicate requested indices are preserved exactly.

`RawFeatureBatch` contains no field embeddings:

```text
context_tokens         int64     [B, C]
operand_features       FP32      [B, P]
denominator_validity   bool      [B, P]
targets                FP32      [B, 1]
global_row_indices     CPU int64 [B]
```

The trainable shared encoder owns the 16-dimensional categorical embeddings
and learned numeric-field vector-times-scalar view. Raw P and denominator
validity reach the controlled model without being replaced by fixed features;
targets and audit indices remain outside the encoder forward signature.

## Composite model and capacity firewall

The optimizer owns every unique trainable parameter of the full composite:
encoder, controlled interaction, adapter, and common head. All parameters
remain FP32, and the frozen public-run contract requires BF16 autocast around
the forward while sensitive operations preserve their model-level FP32 rules.
Checkpoint precision is derived from the actual autocast flag, so a standalone
FP32 synthetic config cannot be mislabeled as BF16. DDP is forbidden and each
worker exposes one logical `cuda:0` to one independent process.

Before optimizer construction or any source open, the harness also binds every
runtime interaction-config dataclass to the corresponding frozen search
architecture (including all
DNN/DeepFM/AutoInt+/AFN+/Wukong geometry and DCDN/DCN/DCNv2 depth and
initialization scales). AdamW betas, epsilon, weight decay, validation
patience, global-L2 clipping, effective batch, BF16 autocast, and the
one-reduction/second-OOM-stop policy must equal the frozen common contract.
Microbatch size may vary only as an exact divisor of the unchanged effective
batch.

The full verified capacity artifact is
`configs/capacity/vk_lsvd_c_plus_p_v1.json`. Protocol, resolved config, and
checkpoint bind the SHA-256 of its exact file bytes, not the digest of a
compact projection. Before the first model metric, the harness:

- validates the artifact and every repository source binding;
- projects it to the mechanical `capacity_resolution` schema;
- independently rebuilds DCDN-Core, ordinary full-rank DCNv2, and the adjacent
  low-rank brackets;
- requires the two projections to be identical and accepted by
  `validate_capacity_resolution`;
- requires the runtime capacity candidate and low rank to equal the artifact's
  metric-free selected decision; the adjacent upper rank remains diagnostic.

Parameter matching counts trainable adapter/head/interaction parameters and
excludes every `nn.Embedding`. That exclusion is only the capacity comparison
scope: encoder embeddings remain trainable and are included in the optimizer
and checkpoint. FLOPs are deterministic post-encoder forward FLOPs per
example; embedding lookup and encoder numeric vector multiplication are
outside this frozen capacity-comparison scope.

The rank decision and inherited learning rate are deliberately separate
artifacts. The immutable human receipt is
`results/tuning/vk_lsvd_c_plus_p_dcnv2_full_rank_selection_v1.json` (SHA-256
`869cbcb7a227b274700cda142f23c59040497bfdfe4da101888e2cc63f39d67a`).
A clean committed materializer may later create, exactly once with
`O_EXCL`, the compact authorization
`artifacts/selection/vk_lsvd_c_plus_p_capacity_lr_binding_v1.json` and its
strict sidecar. Publication walks held directory descriptors with
`O_NOFOLLOW`; symlink ancestors, path escape, torn pairs, duplicate JSON keys,
unknown fields, and non-finite numbers fail closed. The materializer rejects a
tracked diff and every untracked executable/importable file below `src/` or
`scripts/`. Its recorded commit is immutable creation provenance; each later
run independently binds its current clean code identity in the registry.

The verifier does not trust receipt booleans. It reopens both candidate
configs, results, registries, and histories, recomputes scientific eligibility
and the strict untied minimum, checks exact best-checkpoint lineage, and
reopens the selected checkpoint. It independently verifies the metric-free
rank-12 capacity decision and invokes
`require_phase("capacity_control_training", ...)` with
`SELECTED_DCNV2_LEARNING_RATE=0.0003`. Capacity rank 13, LR 0.001, or any
mismatch among config, receipt, compact authorization, capacity artifact, and
premetric protocol is rejected before optimizer construction.

## Examples, cadence, and deterministic resume

The declared hard maximum remains 40,000,000 examples and is never rounded
up. With effective batch 4096, the realized maximum is

```text
floor(40,000,000 / 4096) * 4096 = 39,997,440.
```

Validation declarations remain every 4,000,000 examples. A check fires on the
first completed effective-batch boundary at or after its declaration; the
first realized check is therefore 4,001,792. Events, checkpoints, and results
record both the declared threshold and realized examples. The policy is
identical for every model and does not alter the frozen search hash.

`StatefulPermutationSampler` is the only cursor owner. A microbatch is
committed only after its forward/backward succeeds, and a checkpoint is
durable only at accumulation step zero. Resume requires exact equality:

```text
examples_seen = optimizer_steps * effective_batch_size
examples_seen = sampler_epoch * train_row_count + sampler_cursor
```

Checkpoint loading maps the full payload to CPU first so CUDA sampler and RNG
state cannot violate their CPU contracts. All envelope, metadata, tensor
shape/dtype/layout, optimizer, sampler, RNG, module extra-state, and harness
extra-state checks finish before any live component is mutated; RNG restoration
is last. A pending or nonfinite monitor drain fails before checkpoint mutation.

`last.pt` is the mutable resume checkpoint. Validation-selected best weights
use immutable generation names such as
`best-step-000000000123-generation-0000.pt`; an existing generation is never
overwritten. Every last checkpoint stores the selected generation filename,
SHA-256, and metric. Resume pre-inspects that complete best envelope, role,
metric, and digest before loading last, so a stale best cannot be correlated
with a newer last. Old generations are retained during an active run; terminal
garbage collection can be performed only after durable result/backup evidence.

An OOM restores the last verified boundary, halves the microbatch to a divisor
of the unchanged effective batch, records `oom_count: 1`, and immediately
writes the fallback last checkpoint before continuing. A crash therefore
cannot resurrect the first-allowance state. A second OOM stops the run.

Fresh execution rejects any existing last pair or best generation. Resume
requires an intact `last.pt` plus sidecar and never silently starts fresh.

## Best restore and current test firewall

`restore_best_for_evaluation` is a model-only, premetric restore for validation
latency. It verifies role `best`, exact metric, checkpoint metadata, source
identity, composite contract, encoder contract, capacity artifact, and parent
premetric protocol. It leaves optimizer, sampler, scheduler, scaler, RNG, and
their modes untouched before the harness explicitly selects evaluation mode.

`restore_parent_best_for_final` requires caller-supplied exact parent checkpoint
metadata and selected-lock lineage; it does not weaken metadata equality to
bridge tuning and final phases. The selected model digest is seed-invariant and
covers the exact interaction and head configs. Candidate ID and learning rate
are independently frozen for the ordinary tuned/fixed models, while seed
remains exact in the training config, joint model contract, checkpoint, and
eventual claim. For capacity control, the seed-invariant selected-model digest additionally
covers candidate `capacity_rank_lower`, LR 0.0003, the capacity artifact SHA,
the ordinary selection-receipt SHA, the compact authorization SHA, and the
premetric Protocol SHA. The code path is fail-closed until the compact
artifact is materialized and audited; this implementation change does not
itself create that production artifact or authorize a launch.
Final-seed validation training for seeds 43 and 44 still uses the active
premetric protocol and never opens test. Its immutable training config carries
the selected validation receipt path and SHA. Before model or source creation,
the trainer reopens that receipt, the selected seed-42 resolved config/result,
and the selected best checkpoint; candidate, learning rate, architecture,
feature regime, model digest, test-firewall attestations, and all sidecars must
match exactly. Null receipt fields are omitted from legacy tuning checkpoints.

Model-only final restore accepts an expected checkpoint SHA and rejects a
digest mismatch before `load_state_dict`, so wrong-weight lineage is non-mutating.

Loss-only test evaluation and test latency remain forbidden. The
dataset-independent one-pass final runner now reserves the exact central claim,
prevalidates checkpoint SHA and metadata before model mutation, marks the claim
opened immediately before the only source-factory call, consumes model inputs,
targets, stable row IDs, and user-cluster IDs in one ordered pass, drains the
numerical monitor, publishes an immutable strict-sidecar metric bundle, and
marks the claim completed. The bundle contains frozen watch-time metrics,
row-alignment digests, and privacy-safe per-user sufficient statistics for the
predeclared paired cluster bootstrap. A crash after bundle publication but
before claim completion can only reconcile the already verified bundle; it
cannot reopen the source. Synthetic lifecycle, symlink, checkpoint, crash, and
rehashed-metric negatives pass.

Actual test execution is still unavailable in the CLI because the controlled
week-26 source transition and final-child protocol activation are not yet
implemented and audited. The cross-model RQ1 aggregator is now implemented
before test access: it requires the exact twelve final bundles, revalidates
bundle digests and same-row/target/user alignment across C+P and C+P+E, pools
all three seeds within each resampled user, runs the frozen 2,000-replicate
percentile bootstrap, and applies Holm--Bonferroni to the exact four primary
contrasts. Until the remaining source/protocol gates pass, test stays locked.

## Numerical monitoring and latency

Activation hooks select suffixes
`controlled_model.interaction`, `controlled_model.adapter`, and
`controlled_model.head` on the composite wrapper. Construction requires
exactly three hooks, and each normal model forward must increase the
observation count. CPU finite checks are synchronous. CUDA checks use a
dedicated stream and pinned one-byte flags, with a mandatory drain at durable
boundaries. A repeated/nonfinite threshold failure stops the run.

Validation latency is end-to-end raw-to-logit model latency after source batch
materialization; source I/O itself is outside the timed region. A monitored
warmup first proves finite outputs and hook observations. Hooks are then
temporarily detached so monitor kernels do not contaminate the timed forwards,
and are reattached with the exact expected count in `finally`, including on a
timed-forward exception. Test latency is forbidden. The receipt field is
`end_to_end_raw_to_logit_validation_latency`, and its scope value is
`end_to_end_raw_to_logit_model`.

## CLI and worker durability

`scripts/train_model.py` accepts the final VK C+P source/encoder factories but
requires an actual active premetric Protocol v3 file and state. Resolved schema
v3 is preserved byte-for-byte for non-capacity runs and requires exactly one
of `execution_kind: implementation_smoke|scientific_tuning`. Schema v3 rejects
capacity, while schema v4 is capacity-only and requires the top-level
`capacity_lr_binding_path` plus `ordinary_dcnv2_selection_sha256` and
`capacity_lr_binding_sha256` in `training`. No schema has a test source/path
option. A typical single-process command is stored verbatim in the checkpoint
and result, without credentials:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_model.py run \
  --resolved configs/runs/<run-id>.yaml
```

Paid execution is owned by `scripts/run_worker.py`, one process per physical
GPU. The worker verifies the registry record, resolved-config sidecar, active
protocol/data/features digests, and code identity before spawn. Code identity
is exact `git rev-parse --verify HEAD` plus SHA-256 of
`git diff --binary HEAD --`; any untracked executable/importable source under
`src/` or `scripts/` is rejected. The same code identity is rechecked before a successful terminal transition.
For capacity, the worker also reopens the compact binding and a fresh registry
record before and after terminal publication. It safely loads retained
`last.pt` and best checkpoints on CPU with `weights_only=True` and requires
exact training SHA fields plus `capacity_training_binding` in checkpoint
metadata; a checkpoint modified and re-signed with a new sidecar still fails.
This convention requires the harness code to be committed cleanly before a
paid smoke.

SIGINT/SIGTERM handlers are installed before child spawn. The supervisor owns
the child process group in an enclosing `finally`: spawn-transition failures,
signals, guard failures, and unexpected exceptions stop only that task-owned
trainer and terminalize a `running` registry record as failed. The worker does
not mutate the Pod or network volume.

The trainer command is wrapped by the Linux-only supervised-exec module. It
sets PR_SET_PDEATHSIG to SIGKILL, verifies the exact expected worker PID after
prctl, and then execs the trainer without changing PID. A worker SIGKILL
therefore cannot leave a task-owned GPU trainer running; a parent-race exits
before trainer exec.

Registry persistence is isolated from the trainer. Every registry mutation runs
in a CUDA-hidden, telemetry-disabled one-shot helper process. Every helper uses
the same supervised-exec parent-death binding to the exact worker PID as the
trainer. One outer owner retains every helper Popen across operation deadlines
and exceptions, permits at most one helper, and releases a handle only after
poll or wait confirms reap. At a 10-second operation timeout the supervisor
sends SIGKILL exactly once and launches no replacement until the old helper is
reaped.
An unreaped helper is polled without blocking while training remains live only
within the same 180-second degraded window. A successfully claimed run emits
one coalesced heartbeat every 30 seconds. Heartbeats use a same-directory
unique temporary file, buffered flush, atomic replace, and exact reopen, but
deliberately omit both file and directory fsync. Creation, ownership, terminal
transitions, and incident receipts retain full file-plus-directory durability.

Only a typed allowlist of connection/stale-mount errors, including EIO, is a
candidate for bounded retry. The first such heartbeat failure starts one fixed
180-second degraded deadline; training may continue during that window while
one helper at a time retries with bounded backoff. Hard failures such as
ENOSPC, EDQUOT, EROFS, malformed registry bytes, owner mismatch, or helper
protocol corruption stop the trainer immediately. Timeout is evidence of an
unresolved operation, never evidence that a write did not occur, so retries
use idempotent ensure_running/ensure_terminal reconciliation.

At the first degraded heartbeat, the supervisor writes a canonical SHA-256
bound incident receipt to the configured local overlay only if it is on a
filesystem independent of the registry. A successful later heartbeat or
terminal operation copies the same incident payload into the persistent
registry incidents directory, using an immutable filename that embeds the
content digest. Incident schema v1 is separate from the run-record schema,
contains no model/data metrics, and is supervisor I/O evidence only; it does
not itself prove a registry transition. Failure to create independent
emergency evidence is fail-closed.

After the trainer is dead, the worker publishes one fixed terminal intent for
at most 300 seconds. Ambiguous full-durability failures reopen and reconcile
that exact status, reason, PID/GPU owner, and scientific-eligibility value. A
verified completed intent is never downgraded to failed. The enclosing cleanup
always stops the trainer and holds the GPU worker lock through the final
bounded helper wait/poll cleanup attempt. An unreaped helper has already
received SIGKILL and remains parent-death-bound to the worker; it cannot be
forgotten and replaced by a concurrent helper. If persistent storage never
recovers, the task exits as an execution failure with no trainer process; the
registry can remain stale running and must be audited/reconciled before any
later launch.

These helper modules change executable code identity but do not change the
run-record schema. Therefore any resume after this hardening must bind the new
tracked helper/worker/registry bytes in a newly verified code commit/diff hash;
a pending record frozen against earlier bytes cannot be reused unchanged.

The trainer publishes result JSON and its SHA sidecar as a same-directory,
exclusive pair: both temporary files are flushed and fsynced, hard-linked
without overwrite, the directory is fsynced, and the exact bytes are reopened
and verified. A collision leaves the existing pair untouched; a partial
publication/failure removes the newly created target so it cannot be mistaken
for a receipt.

Exit code zero is insufficient for `completed`. The worker reopens the exact
resolved config, result pair, last checkpoint pair, and selected immutable best
generation. It checks run/protocol/resolved-config provenance, examples and
effective-batch equality, declared/realized budgets, finite timing and metric
fields, OOM limit, source-free test fields, capacity/composite hashes, disabled
telemetry, local final guards, and the complete drained numerical summary. Capacity records additionally require
exact model key, candidate, LR, rank, capacity SHA, ordinary-selection SHA, and
compact-authorization SHA. The immutable registry record, resolved config, and
receipt must carry the same `execution_kind`; result schema v3 remains exact
for non-capacity and result schema v4 adds the exact
`capacity_training_binding` only for capacity. A terminal record also carries
the worker-verified boolean
`scientific_result_eligible`; nonterminal records keep it null, and an
implementation smoke can transition to completed only with `false` and an
explicit `NON-SCIENTIFIC implementation smoke` reason.

Checkpoint integrity is verified by a checkpoint-specific streaming SHA-256
reader, so legitimate optimizer checkpoints are not subject to the generic
64 MiB JSON/YAML artifact cap. The generic cap remains unchanged. A historical
run rejected only by that old cap may be repaired without trainer execution by
passing `--reconcile-checkpoint-size-failure` to the same worker command. This
path requires the exact legacy failure reason, a dead recorded trainer PID, an
oversize `last.pt`, a clean committed verifier checkout, and two successful
full result/checkpoint reopen passes. The completed record embeds its entire
prior failed record, prior record SHA-256, verifier commit/empty diff, result
SHA-256, and reconciliation UTC; no metric or checkpoint bytes are rewritten.

## Verification scope

Development checks run only in a fresh `/tmp` repository snapshot. The
targeted CPU suite covers raw dtype/order/duplicate contracts, optimized
no-host-sync validation, debug rejection, exact uninterrupted-versus-resumed
model/optimizer/sampler equality, immutable-best lineage, stale-best
no-mutation failure, CPU checkpoint mapping, OOM fallback durability, source
gate order and deterministic close, wrapper hooks, latency hook restoration,
exact frozen common-training fields, all-nine-model architecture normalization,
capacity candidate/rank selection, metric-free capacity counts/FLOPs, immutable
receipt publication, execution-kind/runtime/result-scope drift, registry
terminal eligibility, worker provenance, and malformed result rejection.
Registry fault injection additionally covers stale same-PID temporary files,
heartbeat directory-fsync coalescing, typed EIO versus ENOSPC classification,
an ambiguous post-replace terminal EIO followed by exact reconciliation,
hash-bound incident reopen/collision handling, a killed 10-second helper,
transient recovery with persistent incident publication, hard-error stop, the
180-second degraded deadline, and byte-for-byte reuse of terminal intent.
Checkpoint tests separately cover pre-mutation component and extra-state
failures.

Example isolated commands (the environment must already contain the locked
dependencies) are:

```bash
env PYTHONPATH=/tmp/<snapshot>/src \
  /tmp/<test-venv>/bin/pytest -q \
  tests/test_training_harness.py tests/test_checkpoint_resume.py

/tmp/<test-venv>/bin/ruff check \
  src/dcdn/training.py src/dcdn/monitor.py src/dcdn/raw.py \
  src/dcdn/checkpoint.py scripts/train_model.py scripts/run_worker.py \
  src/dcdn/registry.py tests/test_training_harness.py \
  tests/test_checkpoint_resume.py tests/test_registry.py
```

These tests use synthetic CPU data only. They do not open the real cache,
compute a public model metric, touch test data, activate a real/project
protocol, launch a GPU job, emit telemetry, or perform Git/network mutations.
The protocol tests activate only an ephemeral synthetic registry under their
own `/tmp` directory.
