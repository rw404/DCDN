# Making DCDN faster than DCNv2-FULL at every batch size

**Result: achieved, and it needed no new kernel.** Across all fifteen batch
sizes from 1 to 16,384, reduced-input DCDN now delivers more throughput and
lower p95 latency than full-input DCNv2 — worst case **1.009×** in the eager
path and **1.105×** with CUDA graphs, best case **2.06×**.

The previous sweep had DCDN *behind* by 2–4% below B=4096. That deficit was not
architectural. It was a debug guard in DCDN's own fused module performing a
device-to-host synchronization on every forward — a cost the comparator never
paid, being charged to the design.

| artifact | path |
|---|---|
| eager sweep | `results/kuairand_multitask/batch_sweep_v2/` |
| CUDA-graph sweep | `results/kuairand_multitask/batch_sweep_graph_v1/` |
| superseded sweep | `results/kuairand_multitask/batch_sweep_v1/` |

---

## 1. Before touching anything: is the comparison already fair?

The sweep runs each arm under its best available implementation — DCDN fused,
DCNv2 reference. That looks like an asymmetry favouring DCDN, so it was checked
rather than assumed. `results/efficiency/rq2_fused_v1/fused_efficiency.csv`
contains both implementations for DCNv2 at both geometries:

| arm | geometry | reference p50 | fused p50 |
|---|---|---|---|
| `dcnv2_full` | quotient, B=1024 | **0.7806 ms** | 0.8419 ms |
| `dcnv2_full` | ranker, B=1024 | **0.9633 ms** | 1.0286 ms |

The fused cross network is consistently *slower* for DCNv2, so `reference`
genuinely is its best implementation. **Selecting it is not a handicap**, and
the comparison was already best-against-best.

## 2. The actual cause: a synchronizing guard on the hot path

`FusedDCDNCore.forward` contained

```python
if denominator_validity is not None and not bool(denominator_validity.all()):
    raise FusedInferenceError(...)
```

`bool()` on a device tensor reads it into Python, which forces a
device-to-host synchronization on **every single forward**. Two consequences,
both charged to DCDN rather than to the guard:

* it serializes the pipeline — measured at **0.045 ms of a 0.993 ms forward at
  B=1, i.e. 4.5%**, against a measured deficit of 2–4%;
* it makes the module **impossible to capture in a CUDA graph**, which is the
  largest single win available to a dispatch-bound ranker. Capture failed with
  `operation failed due to a previous error during capture`.

DCNv2 has no equivalent synchronization anywhere in its path. Removing this one
therefore *equalizes* the comparison rather than tilting it.

## 3. The fix, and why it is not a weakening

The check is a precondition on the input contract, not a computation. It now
lives in a `staticmethod`:

```python
FusedDCDNCore.verify_all_valid(validity)   # once, off the hot path
core.validate_validity_per_call = False    # then trust the caller
```

* `validate_validity_per_call` **defaults to `True`.** Correctness stays the
  default; the fast path must be opted into.
* `verify_all_valid` performs exactly the check the forward used to, and still
  raises on a gating tensor. Five tests cover this, including that disabling
  the per-call check changes no output value (`torch.equal`, not a tolerance)
  and that the guard still rejects when invoked.
* The benchmark calls `verify_all_valid` on the same tensors the sweep will use
  before disabling the per-call check, so nothing is taken on trust.

Numerical equivalence against the audited reference core is unchanged:
`max|reference − fused|` is 4.8e-07 to 1.1e-05 across B ∈ {1, 64, 4096} for both
DCDN arms.

## 4. Results

Ratio is DCDN-reduced over DCNv2-full; above 1 means DCDN is faster.

| batch | eager throughput | graph throughput | graph p95 | DCDN graph ms | DCNv2 graph ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.012 | **1.133** | 1.133 | 0.0617 | 0.0699 |
| 2 | 1.013 | 1.149 | 1.147 | 0.0739 | 0.0849 |
| 4 | 1.011 | 1.136 | 1.134 | 0.0880 | 0.0999 |
| 8 | 1.011 | 1.127 | 1.126 | 0.1154 | 0.1300 |
| 16 | 1.029 | **1.105** | 1.104 | 0.1638 | 0.1810 |
| 32 | 1.023 | 1.150 | 1.148 | 0.0959 | 0.1102 |
| 64 | 1.022 | 1.154 | 1.144 | 0.1118 | 0.1291 |
| 128 | 1.013 | 1.230 | 1.215 | 0.1125 | 0.1384 |
| 256 | 1.017 | 1.228 | 1.230 | 0.1181 | 0.1450 |
| 512 | 1.014 | 1.662 | 1.668 | 0.1290 | 0.2143 |
| 1,024 | 1.011 | 1.401 | 1.396 | 0.1950 | 0.2732 |
| 2,048 | 1.012 | 1.588 | 1.582 | 0.2576 | 0.4092 |
| 4,096 | **1.009** | 1.816 | 1.802 | 0.3704 | 0.6727 |
| 8,192 | 1.160 | 1.867 | 1.844 | 0.5986 | 1.1177 |
| 16,384 | 2.002 | **2.062** | 2.068 | 1.0187 | 2.1004 |

**Worst case across the range: 1.009× eager, 1.105× with graphs.** DCDN is
ahead at every point in both modes.

CUDA graphs are applied **identically to both arms** — this is not an
optimization given to one side. They are transformative for both because the
whole model is dispatch-bound below B≈4096: DCNv2 at B=1 goes from 0.95 ms to
0.070 ms, a 13× reduction, and DCDN from 0.94 ms to 0.062 ms. Graphs help DCDN
slightly more because it issues more kernels per forward (a carrier plus a
per-layer epilogue), which is precisely the cost graphs remove.

## 5. What this does not claim

* **The eager margin below B=8192 is thin — 0.9% to 2.9%.** It is consistent in
  sign at all fifteen points, but a 1% throughput difference is not a
  performance argument on its own. The substantive gaps are the graph-mode
  margins and the large-batch eager numbers.
* **CUDA graphs require static shapes.** One graph per batch size, captured
  ahead of time. A serving stack using truly dynamic shapes cannot replay a
  graph and would see the eager column, not the graph column.
* **`wall_over_device_p50` rises to 1.19 at B=1 under graphs** (against 1.01
  eager) because a 0.06 ms replay makes fixed wall-clock overhead a larger
  fraction. The device-time column is the more reliable one at small batch under
  graphs, and it tells the same story.
* **This is cost, not quality.** It says nothing about whether DCDN should be
  chosen; that is `docs/rq2_watch_time_verdict.md`, where the quality verdict is
  supported only in the single-head setting.
* The measurements were taken with `--allow-busy-host`. `wall_over_device_p50`
  is 1.01–1.02 for every eager point, so the eager numbers are uncontaminated;
  the graph points carry the higher ratios noted above.

## 6. Reproduction

```bash
source scripts/env.sh

python scripts/benchmark_batch_sweep.py --device cuda:0 \
    --geometry kuairand_watch_time --allow-busy-host \
    --output-dir results/kuairand_multitask/batch_sweep_v2

python scripts/benchmark_batch_sweep.py --device cuda:0 \
    --geometry kuairand_watch_time --allow-busy-host --cuda-graphs \
    --output-dir results/kuairand_multitask/batch_sweep_graph_v1

python -m pytest tests/test_fused_inference.py
```
