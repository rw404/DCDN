# RQ2 — serving performance and efficiency

Artifact role: `rq2_efficiency_result`. Status: **complete**.

This is the *cost* half of RQ2: "does DCDN reach better multi-target quality
**at comparable latency/throughput**". The quality half is
`results/synthetic_multiquotient/rq1_v2`; this study supplies the denominator
of that sentence and decides, per pair, whether the phrase "at the same serving
performance" is available at all.

Nothing here is trained. No dataset row, label, or test metric is read. Inputs
are random tensors of a declared geometry, so the study runs without touching
any data pipeline.

## 1. What was measured

| item | value |
|---|---|
| device | one NVIDIA A100-SXM4-80GB |
| pre-measurement manifest SHA-256 | `5de33d29f3aa86b41b0fdec2860035c8edb7797ef93bc0bbcf66ebf5af0088cc` |
| results SHA-256 | `5d69b48444d5c0c3f90085e277e2e7f22165500dd59062130657c9080764de54` |
| `efficiency.csv` SHA-256 | `766fdfd3ab1adbd9d752d8710a58727114253825721056225623eef29a29e984` (84 rows) |
| `pareto.csv` SHA-256 | `57be62ab98a0fe3a128394ad6a2d6abc3b4572a10db2ae9bf285f8fd279f4418` (32 rows) |
| protocol | full multi-target forward, every head; `eval()` + `inference_mode()`; device-resident inputs; batches {1, 64, 1024}; 200 warmups; 1,000 synchronized forwards; 5 independent rounds |
| arms | 14 per geometry: the frozen `M1..M7` matrix, a linear floor, a DCDN depth control, and the Tier-B zoo |

The manifest was published **before the first timed forward**, so the plan
could not be adjusted after seeing a number. All four artifacts verify against
their strict exact-byte sidecars.

### Two geometries, and why both are needed

`quotient` — the exact widths of the RQ1 benchmark (operand 8 reduced, 11
full, no categorical bypass). Arm ids match `rq1_v2` exactly, so cost and
quality can be placed on the same axes.

`ranker` — an industrial-scale **geometry stand-in**: 6 categorical fields with
real embedding tables, 96-wide bypass, 61 shared operands (78 with `R_div`),
embedding width 16, four heads — the field and width counts that
`configs/data/vk_lsvd/features_multitask_v1.yaml` derives. **No VK-LSVD row,
label, or statistic is read**; only the shape is borrowed and the inputs are
random.

At width 8 every arm is dominated by kernel-launch overhead, so the quotient
geometry alone would say almost nothing about serving a real ranker. §4 below
shows the two geometries disagree on a headline verdict, which is the
justification for measuring both.

## 2. Validity

The protocol asks for an idle A100. A concurrent session held the one-minute
load average at 3.9, so the load gate was **waived** — but not silently, and not
on assertion.

Every forward was timed twice: wall clock around a synchronized region (the
quantity the threshold is defined on) and CUDA events (device execution only,
immune to host scheduling). Agreement between the two is direct evidence the
tail is uncontaminated; disagreement would be direct evidence that it is not.

| statistic | value |
|---|---|
| median wall/device p95 ratio | **1.012** |
| worst wall/device p95 ratio | **1.100** |
| ceiling | 2.0 |
| verdict status | `final_load_gate_waived_by_device_time_agreement` |

For comparison, the discarded first attempt of the earlier serving benchmark
recorded `DCNv2` p99 = 74.8 ms against its own p50 of 0.73 ms. Nothing of that
kind appears here: the worst p99 in this run is 2.28 ms, and round-to-round p50
spread stays in the third decimal. A load-average threshold is a proxy;
device-time agreement is the measurement.

## 3. Cost tables

`B=1` latency in ms, throughput at `B=1024` in examples/s, peak VRAM in MB,
interaction parameters, and FLOPs/example where a frozen closed form exists.
FLOPs are left blank rather than guessed for architectures without one.

### 3.1 `quotient` geometry

| arm | p50 | p95 | p99 | device p95 | throughput | VRAM | interaction | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear (floor) | 0.114 | 0.124 | 0.126 | 0.113 | 8,894,286 | 33.1 | 0 | — |
| M1 DCNv2-REDUCED | 0.752 | 0.762 | 0.777 | 0.749 | 1,334,979 | 34.4 | 144 | 34,656 |
| M2 DCDN-REDUCED | 1.040 | 1.057 | 1.158 | 1.043 | 967,653 | 34.4 | 144 | 34,664 |
| M4 DCNv2-capacity-matched | 1.017 | 1.027 | 1.046 | 1.013 | 988,337 | 34.4 | 148 | 34,673 |
| M3 AFN+-REDUCED | 1.272 | 1.289 | 1.334 | 1.276 | 801,558 | 51.2 | 560,402 | — |
| M5 DCNv2-FULL | 0.750 | 0.761 | 0.785 | 0.747 | 1,332,305 | 34.5 | 264 | 35,280 |
| M6 DCDN-FULL | 1.040 | 1.051 | 1.081 | 1.037 | 965,870 | 34.5 | 264 | 35,291 |
| M7 AFN+-FULL | 1.271 | 1.292 | 1.341 | 1.279 | 799,351 | 51.7 | 573,080 | — |
| DCDN-L2 (depth control) | 1.304 | 1.320 | 1.360 | 1.306 | 775,386 | 34.4 | 288 | 34,952 |
| DNN | 0.841 | 0.855 | 0.894 | 0.842 | 1,190,907 | 37.1 | 168,832 | — |
| DeepFM | 1.046 | 1.058 | 1.093 | 1.044 | 963,953 | 37.1 | 197,569 | — |
| DCN | 0.669 | 0.679 | 0.707 | 0.665 | 1,500,843 | 34.4 | 32 | — |
| AutoInt+ | 2.014 | 2.036 | 2.101 | 2.022 | 486,102 | 41.2 | 201,600 | — |
| Wukong-small | 1.843 | 1.872 | 1.924 | 1.858 | 527,158 | 35.1 | 29,760 | — |

### 3.2 `ranker` geometry

| arm | p50 | p95 | p99 | device p95 | throughput | VRAM | interaction | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear (floor) | 0.115 | 0.125 | 0.128 | 0.114 | 8,849,758 | 33.4 | 0 | — |
| M1 DCNv2-REDUCED | 0.908 | 0.918 | 0.960 | 0.904 | 1,073,033 | 78.4 | 49,612 | 152,792 |
| M2 DCDN-REDUCED | 1.205 | 1.232 | 1.270 | 1.218 | 820,034 | 78.2 | **7,564** | **68,757** |
| M4 DCNv2-capacity-matched | 1.643 | 1.676 | 1.829 | 1.662 | 601,261 | 78.2 | 7,594 | 68,880 |
| M3 AFN+-REDUCED | 1.582 | 1.603 | 1.648 | 1.590 | 617,194 | 161.3 | 1,608,456 | — |
| M5 DCNv2-FULL | 0.907 | 0.916 | 0.930 | 0.902 | 1,066,042 | 80.6 | 60,900 | 177,544 |
| M6 DCDN-FULL | 1.207 | 1.220 | 1.243 | 1.206 | 817,243 | 80.5 | 12,324 | 80,470 |
| M7 AFN+-FULL | 1.581 | 1.610 | 1.761 | 1.595 | 618,301 | 166.2 | 1,749,930 | — |
| DCDN-L2 (depth control) | 1.458 | 1.474 | 1.512 | 1.460 | 676,685 | 78.3 | 15,128 | 83,885 |
| DNN | 1.004 | 1.014 | 1.042 | 1.000 | 967,539 | 79.2 | 245,120 | — |
| DeepFM | 1.212 | 1.257 | 1.294 | 1.243 | 816,269 | 80.9 | 714,673 | — |
| DCN | 0.827 | 0.847 | 0.886 | 0.833 | 1,171,133 | 78.2 | 628 | — |
| AutoInt+ | 2.187 | 2.225 | 2.284 | 2.211 | 445,085 | 207.5 | 718,720 | — |
| Wukong-small | 2.001 | 2.020 | 2.068 | 2.005 | 479,481 | 79.7 | 101,060 | — |

## 4. Runtime-match verdicts

A pair qualifies only when `p95(B=1) <= 1.10x` **and**
`throughput >= 0.90x` at the shared maximum batch.

| geometry | pair | p95 ratio | throughput ratio | runtime matched |
|---|---|---:|---:|---|
| quotient | DCDN vs DCNv2 | 1.387 | 0.725 | **no** |
| quotient | DCDN vs AFN+ | 0.820 | 1.207 | yes, DCDN cheaper |
| quotient | DCDN vs capacity-matched | 1.030 | 0.979 | yes |
| quotient | DCDN-FULL vs DCNv2-FULL | 1.381 | 0.725 | **no** |
| quotient | DCDN-FULL vs AFN+-FULL | 0.813 | 1.208 | yes, DCDN cheaper |
| ranker | DCDN vs DCNv2 | 1.343 | 0.764 | **no** |
| ranker | DCDN vs AFN+ | 0.768 | 1.329 | yes, DCDN cheaper |
| ranker | **DCDN vs capacity-matched** | **0.735** | **1.364** | **yes, DCDN much cheaper** |
| ranker | DCDN-FULL vs DCNv2-FULL | 1.333 | 0.767 | **no** |
| ranker | DCDN-FULL vs AFN+-FULL | 0.758 | 1.322 | yes, DCDN cheaper |

## 5. What the numbers support

**1. DCDN is never runtime-matched to DCNv2.** Both geometries, both feature
regimes: p95 ratio 1.33–1.39, throughput 0.72–0.77. The phrase "at the same
serving performance" is **not available** for the DCDN-versus-DCNv2 pair and
must be replaced by an explicit Pareto trade-off. This is the single most
consequential result here, and it is stable across a 7.6x change in operand
width, so it is not an artifact of the toy benchmark.

**2. DCDN's cost is not FLOP-bound.** At ranker width DCDN carries **6.6x fewer
interaction parameters** (7,564 vs 49,612) and **2.2x fewer FLOPs** (68,757 vs
152,792) than DCNv2, and is still 1.34x slower. DCDN crosses `P` only while
DCNv2 crosses the full flat width, so the arithmetic genuinely is smaller; the
masked reciprocal carrier adds elementwise passes whose kernel and memory
traffic outweigh the arithmetic saved. A FLOP-count argument for DCDN's
efficiency would be wrong, and this points precisely at where a fused
reciprocal kernel would pay off.

**3. Parameter/FLOP matching and runtime matching are genuinely different
checks — measured, not asserted.** `M4` was resolved mechanically to within
**0.1% of `M2` on parameters and FLOPs simultaneously** at both widths. The
runtime verdict nonetheless inverts:

| geometry | M4 param ratio | M4 FLOP ratio | DCDN/M4 p95 | DCDN/M4 throughput |
|---|---:|---:|---:|---:|
| quotient | 1.0002 | 1.0003 | 1.030 | 0.979 |
| ranker | 1.0009 | 1.0018 | **0.735** | **1.364** |

At toy width the two are interchangeable; at ranker width DCDN is 27% faster
and delivers 36% more throughput than a control matched to it within a tenth of
a percent on both capacity axes. The low-rank mixture's gating and expert loop
cost far more wall time than its FLOP count implies. This is direct evidence
for the frozen requirement that both checks be reported separately.

**4. DCDN dominates AFN+ on cost** at both widths: 0.76–0.82x latency,
1.21–1.33x throughput, and roughly half the VRAM (78 MB vs 161 MB at ranker).

**5. Depth is not free.** DCDN-L2 costs +25% p95 over L1 at quotient width and
+20% at ranker width. Since L2 was also markedly less stable out of
distribution in RQ1, the depth choice is a real trade rather than a default.

**6. Removing the manual divisions buys almost nothing at inference.**
DCNv2-FULL and DCNv2-REDUCED are within 0.2% of each other on latency at ranker
width (0.916 vs 0.918 ms); DCDN-FULL and DCDN-REDUCED within 1% (1.220 vs
1.232). Appending 17 `R_div` columns is nearly free in the forward pass. The
value of the removal claim therefore lies in **feature-pipeline complexity**,
not in serving cost, and the paper should say so rather than implying an
inference saving that the measurement does not show.

## 6. Pareto frontier

`pareto.csv` covers the `quotient` geometry only — the geometry whose models
the quality source actually trained. Joining ranker latency to quotient quality
would place two different models on one pair of axes, so those rows are omitted
rather than emitted with a caveat.

On the **ID slice** the non-dominated set is `{DCNv2-FULL, linear}` and
DCDN-REDUCED is dominated. On the **high-ratio OOD slice** the ordering
inverts and DCDN-REDUCED is the frontier point with the best quality of any
arm:

| arm | Macro-NLoss (high-ratio) | p95 `B=1` | on frontier |
|---|---:|---:|---|
| **DCDN-REDUCED** | **11.76** | 1.057 | yes |
| DCDN-FULL | 28.27 | 1.051 | yes |
| AFN+-FULL | 33.05 | 1.292 | no |
| DCNv2-capacity-matched | 50.76 | 1.027 | yes |
| DCNv2-REDUCED | 60.43 | 0.762 | yes |
| DCNv2-FULL | 69.81 | 0.761 | yes |
| linear | 130.20 | 0.124 | yes |

A single frontier over one slice is therefore not a summary of this benchmark.
Which arm is Pareto-optimal depends on the operating regime, and both slices
must be shown.

## 7. Claim wording

Permitted, on this evidence:

* DCDN reaches better quotient accuracy than DCNv2 in the high-ratio regime
  **at a measured cost of ~1.34–1.39x p95 latency and ~0.73–0.77x throughput**;
* DCDN is cheaper than AFN+ on latency, throughput, and VRAM at both widths;
* DCDN is cheaper than its own capacity-matched control at ranker width despite
  matched parameters and FLOPs.

Not permitted:

* "at the same serving performance" for any DCDN-versus-DCNv2 pair;
* any efficiency claim resting on DCDN's lower FLOP count, which the
  measurement contradicts;
* any inference-cost benefit attributed to removing the manual divisions.

## 8. Reproduction

```bash
cd deep_division_network
source scripts/env.sh
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
PYTHONPATH=src "$PY" -m pytest -q tests/test_benchmark_efficiency.py
"$PY" scripts/benchmark_efficiency.py --device cuda:0            # requires an idle host
"$PY" scripts/benchmark_efficiency.py --device cuda:0 --allow-busy-host
```

The second form waives the load-average gate; validity is then decided by the
wall-versus-device agreement recorded in `results_v1.json`. The script refuses
to write into an existing directory, so a re-run needs `--output-dir` pointing
at a fresh path under `results/`.
