# RQ2 addendum — serving cost under the best available implementation

Artifact role: `rq2_fused_inference_result`. Status: **complete**.
Artifacts: `results/efficiency/rq2_fused_v1/` (`results_v1.json`
`f9f1d5a7…`, `fused_efficiency.csv` `d0a00787…`, 60 rows,
pre-measurement manifest published before the first timed forward). All verify
against strict exact-byte sidecars.

## 1. Why this study exists

`docs/rq2_serving_performance.md` measured the reference PyTorch forward and
found DCDN 1.33–1.39x slower than DCNv2 at p95, despite DCDN carrying **6.6x
fewer interaction parameters and 2.2x fewer FLOPs** at ranker width. That
inversion is the signature of an implementation artifact, not an architectural
cost:

| | GEMMs/layer | elementwise passes/layer | carrier |
|---|---:|---:|---|
| DCNv2 `x <- x + x0*(Wx+b)` | 1 | 2 | — |
| DCDN `x <- x + x0*(Wp x+bp) + inv(x0)*(Wd x+bd)` | 2 | 4 | ~17 elementwise ops, once |

At serving batch sizes these blocks are tiny, so wall time is dominated by
kernel launches. Reporting that as an architectural verdict would attribute to
the reciprocal branch a cost that belongs to the schedule.

## 2. What was done

`src/dcdn/fused_inference.py` provides Triton kernels that fuse:

* DCDN's two per-layer GEMMs into one, by concatenating the product and
  division weights into a single `[2F, F]` matrix;
* DCDN's epilogue `state + anchor*product + carrier*division` into one kernel;
* DCDN's masked reciprocal carrier into one kernel;
* the DCNv2 cross epilogue `state + anchor*affine` into one kernel.

Per DCDN layer this goes from 2 GEMMs + 4 kernels to 1 GEMM + 1 kernel, and the
carrier from many kernels to one.

**Both architectures were fused.** Optimising only DCDN would replace one unfair
comparison with its mirror image. Each arm is then represented by whichever
implementation is actually faster for it — the *best available implementation*
rule.

## 3. Equivalence, verified before any timing

The fused path changes the schedule, never a parameter and never a result. On
every geometry and batch measured, worst deviation from the audited module:

**`max|Δ| = 1.53e-05`**, against a tolerance of `1e-03`.

That is float32 GEMM reassociation, nothing more. The fused core is
inference-only: it implements the all-valid denominator case that the frozen
preflight verifies, and rejects a gating validity tensor rather than silently
ignoring it.

## 4. Result

p95 latency at `B=1`, milliseconds. "Best" is the faster implementation for
that arm.

### `quotient` geometry

| arm | reference | fused | speedup | best | best p95 | best throughput |
|---|---:|---:|---:|---|---:|---:|
| DCNv2-REDUCED | 0.772 | 0.845 | 0.91x | reference | 0.772 | 1,323,883 |
| **DCDN-REDUCED** | 1.163 | **0.811** | **1.43x** | fused | 0.811 | 1,269,087 |
| DCNv2-FULL | 0.781 | 0.838 | 0.93x | reference | 0.781 | 1,311,827 |
| **DCDN-FULL** | 1.058 | **0.800** | **1.32x** | fused | 0.800 | 1,269,669 |
| DCDN-L2 | 1.319 | 0.938 | 1.41x | fused | 0.938 | 1,093,850 |

### `ranker` geometry

| arm | reference | fused | speedup | best | best p95 | best throughput |
|---|---:|---:|---:|---|---:|---:|
| DCNv2-REDUCED | 0.930 | 0.989 | 0.94x | reference | 0.930 | 1,052,605 |
| **DCDN-REDUCED** | 1.220 | **0.956** | **1.28x** | fused | 0.956 | 1,020,024 |
| DCNv2-FULL | 0.923 | 0.987 | 0.94x | reference | 0.923 | 1,063,004 |
| **DCDN-FULL** | 1.228 | **0.966** | **1.27x** | fused | 0.966 | 1,010,547 |
| DCDN-L2 | 1.487 | 1.092 | 1.36x | fused | 1.092 | 896,850 |

Fusion speeds DCDN by **1.27–1.43x** and *slows* DCNv2 to **0.91–0.94x** —
DCNv2's reference path is already one GEMM plus two elementwise ops, and Triton
launch overhead exceeds what remains to be saved. This asymmetry is itself the
evidence that the original gap was schedule, not architecture.

## 5. The verdict inverts

A pair qualifies as runtime matched only when `p95(B=1) <= 1.10x` **and**
`throughput >= 0.90x` at the shared maximum batch.

| geometry | pair | p95 ratio | throughput ratio | runtime matched |
|---|---|---:|---:|---|
| quotient | DCDN vs DCNv2 | 1.050 | 0.959 | **yes** |
| quotient | DCDN-FULL vs DCNv2-FULL | 1.024 | 0.968 | **yes** |
| ranker | DCDN vs DCNv2 | 1.028 | 0.969 | **yes** |
| ranker | DCDN-FULL vs DCNv2-FULL | 1.046 | 0.951 | **yes** |

Under the reference implementation **none** of these pairs passed
(1.33–1.39x, 0.72–0.77x). Under the best available implementation **all four**
pass, with margin: worst latency ratio 1.050 against a 1.10 threshold, worst
throughput ratio 0.951 against 0.90.

## 6. What this changes, and what it does not

**Changes.** The phrase "at comparable latency and throughput" **becomes
available** for the DCDN-versus-DCNv2 pair, on both geometries and both feature
regimes, provided the fused kernel is the deployed implementation. The earlier
statement — that DCDN costs ~40% more p95 — was true of the reference
implementation and is now superseded for cost purposes; it remains the correct
figure for anyone running unfused PyTorch.

**Confirms the diagnosis.** §5.2 of the earlier report argued DCDN's cost was
kernel- and memory-bound rather than FLOP-bound. Fusing removed most of the gap
without touching a single FLOP, which is exactly what that diagnosis predicted.

**Does not change.** No quality number moves: outputs agree to 1.5e-05 and the
fused path is never used for training. The RQ2 question — better multi-target
*quality* at comparable serving performance — still cannot be answered, because
no recommendation-dataset quality exists. This study supplies the denominator;
the numerator is still missing.

**Does not establish** that DCDN is *cheaper* than DCNv2. It establishes
parity within the frozen threshold. DCDN's lower parameter and FLOP counts do
not translate into a latency win at these widths, because both architectures
are launch-bound rather than arithmetic-bound at `B=1`.

## 7. Validity

Wall clock and CUDA device events were captured for every forward. Median
wall/device p95 agreement **1.014**, worst **1.017**, ceiling 2.0. Status
`final_load_gate_waived_by_device_time_agreement`. The tighter agreement than
the reference run reflects a quieter host.

## 8. Reproduction

```bash
cd deep_division_network
source scripts/env.sh
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
PYTHONPATH=src "$PY" -m pytest -q tests/test_fused_inference.py
"$PY" scripts/benchmark_fused_inference.py --device cuda:0 --allow-busy-host
```

Requires Triton (3.4.0 here) and a CUDA device. The script refuses to write
into an existing directory; a re-run needs `--output-dir` under `results/`.
