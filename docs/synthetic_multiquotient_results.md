# Synthetic-MultiQuotient-3 — quarantined diagnostic

Artifact role: `invalid_diagnostic_report_v0`.
Status: **invalid; RQ1 is not complete**.

> **Superseding correction, 2026-08-20.** The committed run used eight arms
> that do not equal the frozen M1–M7 matrix: it omitted DCDN-FULL and AFN+-FULL
> and included an unregistered MLP arm. Its JSON sidecars also fail strict
> exact-byte verification because they exclude the written trailing newline.
> Every number below is retained only as diagnostic history and must not be
> cited as RQ1 evidence. The machine-readable invalidation receipt is
> `artifacts/legacy_audit/synthetic_multiquotient_invalid_v0.json`.

Scope discipline, stated before the numbers: this benchmark is **evidence
about the implementation's ability to learn quotients**. It is not evidence of
recommendation superiority, and no claim about ranking quality may be derived
from it.

## 1. What was run

| item | value |
|---|---|
| benchmark manifest SHA-256 | `659d29f565bd51338e8c311a5c1c94335a499fbb2f62017d26581e785d8e72fd` |
| objective spec SHA-256 | `5169e5b5e21c967a…` (`MultiTargetSpec`, 3 regression heads) |
| task-constant SHA-256 | `5f40d98442140c776faa35b9558ed94b78540593010047eb704b2bbd3b1987c6` |
| frozen selected configs SHA-256 | `63555e1f3b1405643372e4600366067df48bdabb6fabbc58cf406501936e5653` |
| tuning | seed 42, validation only, 2 learning rates per arm |
| final seeds | 42, 43, 44 |
| device | one A100-SXM4-80GB |

Targets: `y1 = n1/d1`, `y2 = n2/d2`, `y3 = (n1+n2)/d3`. Inputs
`[n1, d1, n2, d2, d3, z1, z2, z3]`, all strictly positive. The reciprocal mask
is active on exactly `d1, d2, d3`. Frozen train constants
`V = (1.7315, 1.7250, 5.5334)`.

Slice hashes were computed **before** any training step:

| slice | rows | SHA-256 prefix |
|---|---:|---|
| `train` | 200,000 | `effa7d0894ab7f25…` |
| `validation` | 40,000 | `bc45137cb3641119…` |
| `id` | 40,000 | `83a8d763b1133f72…` |
| `ood_common_scale` | 40,000 | `dd7f544a25989c8c…` |
| `ood_low_ratio` | 40,000 | `7d1554b1e2c367f1…` |
| `ood_high_ratio` | 40,000 | `de0c0b15570a4e93…` |

`ood_common_scale` shares the `id` seed and differs only in scale, so its
targets are **bitwise identical row for row** and scale consistency is
measurable per row rather than only in distribution.

## 2. Capacity control

Resolved mechanically in (parameter, FLOP) space with no validation metric
read: low-rank mixture with `rank=6, layers=1, experts=1`, giving parameter
ratio `1.00023` and FLOP ratio `1.00026` against DCDN-Core-L1 — inside the ±5%
tolerance on both simultaneously.

A structural coincidence is worth recording: at operand width 8,
`DCDN-Core-L1` and full-rank `DCNv2-L2` have **exactly 144 interaction
parameters each** (`1·2·8·9 = 2·(8²+8) = 144`) and near-identical FLOPs. The
DCDN-versus-DCNv2 comparison on this benchmark is therefore already
capacity-matched by construction, and `dcnv2_capacity` is a third, independent
control.

Caveat, stated rather than buried: the shared tower and heads dominate the
matched parameter count (16,771 of 17,491), so the ±5% rule on *total*
non-embedding parameters is weakly discriminative here. Interaction-only counts
are reported alongside it for that reason.

## 3. Macro-NLoss, mean over seeds 42/43/44

Lower is better; `1.0` is the frozen constant predictor.

| arm | interaction params | ID | OOD common-scale | OOD low-ratio | OOD high-ratio | scale consistency |
|---|---:|---:|---:|---:|---:|---:|
| analytical oracle | 0 | 0 | 7.2e-15 | 0 | 0 | 0 |
| DCNv2 + explicit ratios (positive control) | 264 | **1.10e-4** | 3.07e+06 | 929 | 69.8 | 684 |
| DCNv2-capacity-matched | 148 | 2.67e-4 | **10.65** | 0.860 | 50.8 | **1.67** |
| **DCDN-Core-L1** | 144 | 3.12e-4 | 5,975 | 2,990 | **11.76** | 40.7 |
| DCNv2-Core | 144 | 1.14e-3 | 8.15e+07 | 2.81e+06 | 60.4 | 3,317 |
| MLP | 168,832 | 1.26e-3 | 14.61 | **0.797** | 56.9 | 1.72 |
| AFN+ | 560,402 | 4.28e-2 | 29.87 | 0.714 | 117.9 | 2.74 |
| linear probe | 0 | 3.10e-1 | 60.6 | 67.4 | 130.2 | 4.02 |

Numerical stability: **0 non-finite predictions** across every arm, seed, and
slice; **0.0 reciprocal-floor incidence** on every denominator field, as
designed. Every arm selected learning rate `1e-3`.

Per-target breakdown is in `results/synthetic_multiquotient/multitask_per_target.csv`.
The aggregate is never published without it.

## 4. What the numbers support

**Diagnostic observation only; this is not an RQ1 answer.**

1. `DCDN-Core-L1` beats `DCNv2-Core` in-distribution by **3.6×** at *identical*
   interaction parameter count and near-identical FLOPs. Since the two differ
   only by the reciprocal branch, this is a clean architecture effect, not a
   capacity effect.
2. `DCDN-Core-L1` dominates the **high-ratio** regime — small denominators,
   large quotients — at 11.76 against 50.8 for the next best arm, a **4.3×**
   margin. That is precisely the regime a reciprocal branch exists to serve,
   and it is the strongest single piece of evidence here.
3. The analytical oracle reaches zero on every slice, confirming the task is
   exactly solvable and that the metric implementation is correct.

**What the numbers argue against, reported because it is what happened.**

4. **The in-distribution gain is not uniquely reciprocal.** The mechanically
   capacity-matched low-rank mixture control reaches 2.67e-4 against DCDN's
   3.12e-4 — it is *better* in-distribution at matched parameters and FLOPs.
   The honest RQ3 answer on this benchmark is regime-dependent: architecture
   wins in the high-ratio regime, matched capacity wins in-distribution.
5. **DCDN does not confer scale invariance.** Its scale-consistency figure is
   40.7 against 1.67 for the capacity control and 1.72 for the MLP, and it is
   ~560× worse on the common-scale slice. The product term `x₀ ⊙ W_p x_i` grows
   with operand magnitude, so the learned solution is not the pure quotient
   even though the pure quotient is exactly representable (proved separately by
   analytical weight construction, `max|error| = 0`).
6. **DCDN is poor in the low-ratio regime**: 2,990 against 0.797 for the MLP.
   Large denominators shrink the reciprocal carrier while the product branch
   keeps growing.
7. **Handing the model the ratio still beats learning it in-distribution.** The
   positive control is 2.8× better than DCDN on the ID slice. This bounds what
   learning the quotient can be worth: substituting a materialized ratio is not
   free, it is a trade.

**Summary.** DCDN-Core learns quotients directly from operands — decisively
against its product-only structural twin, and decisively in the regime where
reciprocals matter. It does not learn *scale-invariant* quotients, and it does
not dominate a capacity-matched mixture control in-distribution.

## 5. Serving performance

Full multi-target forward with every head, `eval()` + `inference_mode()`,
device-resident inputs, one precision policy, batches `{1, 64, 1024}`, 200
warmups, 1,000 synchronized forwards, 5 independent rounds, on one A100.

**Host-idleness caveat, stated first.** The protocol requires an *idle* A100.
A concurrent session on this host held the one-minute load average near 7, so
`scripts/benchmark_serving.py` recorded
`verdict_status = provisional_host_not_idle_p95_tail_is_contaminated`.

A first pass taken under peak contention was visibly corrupted — `DCNv2-Core`
returned p99 = 74.8 ms against its own p50 of 0.73 ms — and is not reported.
The pass below was taken as contention eased. Its p99 values are all under
1.32 ms, its round-to-round p50 spread is 0.003 ms, and **the p50 and p95
ratios agree to within 1% for every pair** (`p50_and_p95_disagree` is `false`
throughout). That agreement, not the load average, is the operative evidence
that this measurement is sound; the flag is retained so the reader can apply
their own standard.

| arm | p50 `B=1` (ms) | p95 `B=1` (ms) | p99 `B=1` (ms) | throughput `B=1024` (ex/s) | peak VRAM |
|---|---:|---:|---:|---:|---:|
| DCNv2-Core | 0.719 | 0.732 | 0.760 | 1,398,164 | 34.4 MB |
| MLP | 0.810 | 0.826 | 0.944 | 1,248,288 | 37.1 MB |
| DCNv2-capacity-matched | 0.982 | 0.996 | 1.013 | 1,023,012 | 34.4 MB |
| **DCDN-Core-L1** | 1.012 | 1.039 | 1.066 | 993,602 | 34.4 MB |
| AFN+ | 1.219 | 1.259 | 1.312 | 836,396 | 51.2 MB |

Runtime-match verdicts. A pair qualifies only when `p95(B=1) <= 1.10x` **and**
`throughput >= 0.90x` at the shared maximum batch:

| pair | p95 ratio | p50 ratio | throughput ratio | runtime matched |
|---|---:|---:|---:|---|
| DCDN vs DCNv2-Core | 1.419 | 1.407 | 0.711 | **no** — fails both |
| DCDN vs MLP | 1.257 | 1.249 | 0.796 | **no** — fails both |
| DCDN vs DCNv2-capacity-matched | 1.043 | 1.031 | 0.971 | yes |
| DCDN vs AFN+ | 0.825 | 0.830 | 1.188 | yes — DCDN is cheaper on both |

**Consequence for claim wording.** Against ordinary DCNv2, DCDN costs about
42% more latency per request and delivers about 71% of the throughput. The
phrase "at the same serving performance" is **not** available for that pair; it
must be reported as a Pareto trade-off. Against the mechanically
capacity-matched control DCDN is runtime-equivalent, which is what matching
parameters and FLOPs is supposed to produce and therefore evidence that the
match is real rather than nominal. Against AFN+ DCDN is strictly cheaper on
both axes.

Combined with §3 and §4: DCDN buys its high-ratio accuracy at a real serving
cost relative to plain DCNv2, and buys nothing over the capacity-matched
control in-distribution while costing the same. Any RQ2-style statement must
carry both halves.

### Pareto frontier

`results/multitask/pareto.csv` plots quality against p95 latency at `B=1`.
On the **ID slice** the non-dominated set is `{DCNv2-capacity-matched,
DCNv2-Core}`, and **DCDN-Core-L1 is dominated**: the capacity-matched control
is both more accurate and cheaper. That is stated plainly because it is what
the measurement says.

The frontier is computed on the primary slice only, and the ordering is
regime-dependent. On `ood_high_ratio` DCDN is 4.3x better than the next arm at
1.04x the latency of the capacity control, so it is comfortably non-dominated
there. A single frontier over one slice is therefore not a summary of this
benchmark; §3's full slice table is.

## 6. Reproduction

The tracked `scripts/run_synthetic_multiquotient_v1.py` reproduces only the
quarantined invalid diagnostic and must not be used for a paper result. The
valid replacement is `scripts/run_synthetic_multiquotient.py`, after that file
and its tests are committed under a clean code identity:

```bash
cd deep_division_network
source scripts/env.sh
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run_synthetic_multiquotient.py \
  --device cuda:0 \
  --output-dir results/synthetic_multiquotient/rq1_v1
```

The fresh output directory will contain the pretraining, selected-config,
summary, per-target, and result artifacts with strict exact-byte sidecars.

## 7. Standing relative to the legacy division tests

`divide.ipynb`, `divide copy.ipynb`, `divide_dcnv3_hiformer.ipynb`,
`testDDN.ipynb`, and `TestDivision.py` are two-feature single-target toys. They
are retained as follow-up material and are classified `legacy_context_only` in
`docs/LEGACY_RESULTS_AUDIT.md`. They do not substitute for this benchmark and
none of their numbers enter any table.
