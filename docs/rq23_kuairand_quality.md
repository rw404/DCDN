# RQ2-quality and RQ3 on KuaiRand-Pure

**Status:** executed on the validation split. The test split is locked and was
never read. This is **not** the mandatory main-table cell for `kuairand_pure` —
that cell requires the test unlock and remains unexecuted.

**Artifacts**

| what | where | role |
|---|---|---|
| **run bundle** | `results/kuairand_multitask/rq23_final_v1/` | the result |
| **contrast tables** | `results/kuairand_multitask/contrasts_v1/` | the result |
| pilot, before the R reorder | `results/kuairand_multitask/rq23_pilot_pre_reorder_v1/` | superseded |
| pilot, two-point LR grid | `results/kuairand_multitask/rq23_pilot_two_point_grid_v1/` | evidence the grid was too coarse |
| geometric preconditioner | `results/kuairand_multitask/rq23_geometric_rejected_v1/` | evidence for the rejection, **not a result** |
| single-head `watch_time` | `results/kuairand_multitask/single_target_v1/` | section 9 |
| its contrasts | `results/kuairand_multitask/contrasts_single_target_v1/` | section 9 |
| transfer comparison | `results/kuairand_multitask/transfer_v1/` | section 9 |

The three non-final bundles are kept because each one is the evidence for a
decision taken in section 3; none of their numbers may be quoted as findings.

---

## 1. What was asked

Two questions, on a light dataset, without competing for the compute the VK
work needs:

* **RQ2-quality.** At matched serving performance, what happens to *quality*
  when DCNv2 loses the manually materialized historical divisions — and can
  DCDN on the reduced inputs stand in for DCNv2 on the full inputs?
* **RQ3.** Does DCDN beat a DCNv2 that has been matched to it on capacity
  rather than on feature count?

The serving-performance half of RQ2 is answered elsewhere
(`docs/rq2_serving_performance.md`, `docs/rq2_fused_inference.md`,
`docs/rq2_batch_sweep.md`, `docs/rq2_hiformer_stack.md`). This document is only
about quality.

## 1a. Where the target set is defined

The five heads are specified in
`configs/data/kuairand_pure/targets_rq23_v1.yaml`
(sha256 `2d51b7150948148cf1dd6e897e4d9d143580d34f8882ebe8e1c4005a794f38b6`),
scoped to this study alone.

`configs/targets_multitask_v2.yaml`, which governs the main table, gives
KuaiRand four binary heads. This study adds a fifth, `watch_time`, so that the
single-head follow-up can compare against a corresponding multi-task head — the
consistency the owner asked for across VK-LSVD, KuaiRec, and KuaiRand.

That head was **first added by editing `configs/targets_multitask_v1.yaml`
directly, which was a mistake.** v2 declares v1 "immutable provenance … and
every artifact that already binds it" and pins v1's sha256 as its predecessor,
so the edit broke that supersession chain and failed
`tests/test_vk_lsvd_multitask.py`. The edit was reverted — v1 is back at
`2a9e3a8e0a5547b500682b9b632800e2bd103dbf67bebec101b281794bc29674` — and the
target set moved here. The study-scoped placement is the more accurate one
anyway: a validation-only study has no business redefining the main-table
objective.

`primary_confirmatory_target` stays `long_view`, unchanged from v2. Adding a
head does not change which target is confirmatory, and moving it would change
the scope of the multiplicity correction.

`configs/single_target_followup_v1.yaml` is likewise unmodified; its KuaiRand
headline target remains `long_view`. Running the follow-up on `watch_time`
instead is a change to the follow-up protocol and needs a versioned successor
made before the first test unlock — it is **open and awaiting a decision**, not
silently applied.

## 2. The seven arms

Exactly the frozen matrix M1–M7 from `configs/matrix_multitask_v1.yaml`:

| arm | model | inputs |
|---|---|---|
| `dcnv2_reduced` | DCNv2 full-rank, 2 layers | REDUCED (132) |
| `dcdn_reduced` | DCDN-Core, 2 layers | REDUCED (132) |
| `afn_reduced` | AFN+ | REDUCED (132) |
| `dcnv2_capacity_reduced` | DCNv2, mechanically capacity-matched to DCDN | REDUCED (132) |
| `dcnv2_full` | DCNv2 full-rank, 2 layers | FULL (244) |
| `dcdn_full` | DCDN-Core, 2 layers | FULL (244) |
| `afn_full` | AFN+ | FULL (244) |

REDUCED = P (108 historical counters and sums) + V (24 validity flags).
FULL = REDUCED + R (112 manually materialized historical divisions).

Per the override, **only** manually materialized divisions of historical
counters and sums were removed. No cosine, dot product, vector norm, or iALS
score was touched — KuaiRand-Pure carries none of those, so the reduction here
is exactly and only the R block.

Capacity matching resolved mechanically to rank 97, 1 layer, 1 expert, giving a
parameter ratio of 1.0025 and a FLOPs ratio of 1.0025 against DCDN — both inside
the ±5% band, on trainable non-embedding parameters.

## 3. Three things that had to be fixed before any number meant anything

### 3.1 The inputs overflow float32

The frozen feature contract fixes natural-unit scales and explicitly forbids
fitting them: *"Count and millisecond scales are fixed, not fitted, and
denominators are never z-centered."* Honouring that, `p_tab_all_duration_ms_sum`
arrives at 9.8e7. Two multiplicative interaction layers over operands that large
overflow float32, and every arm diverged on the first attempt.

The fix is a **model-input preconditioner**, not a feature change: each column is
divided by a train-only per-column root-mean-square. It is deliberately **purely
multiplicative — no shift, no log**, and that restriction is load-bearing. Under
`x → x/s` the ratio `n/d` is unchanged up to the constant `s_d/s_n`, which any of
the compared architectures can absorb into its weights, so the transform is
neutral between arms. A log or z-score would instead turn a quotient of two
counters into a *difference* — precisely the structure DCDN's reciprocal branch
is supposed to have to discover — handing it to the additive baselines and making
the FULL-vs-REDUCED contrast meaningless.

### 3.2 The scale was chosen after seeing an arm diverge, so both were run

RMS was the pre-training choice. When DCDN diverged under it at one learning
rate, a geometric-mean scale was tried — and choosing a knob after seeing a
result is post-hoc, not a principled selection. So the whole matrix was run under
both, and the question settled itself: geometric bounds the lower tail but blows
up the sparse count columns, driving `dcnv2_full` to per-seed losses of 1.26e11,
2.96e17 and 1.75e18 with a `long_view` AUC of 0.489 — below chance.

**Geometric is therefore rejected as a failed preconditioner, not published as a
co-equal variant.** Its bundle is retained only as the evidence for the
rejection. This matters because under geometric the arithmetic reports
"C5: DCDN beats DCNv2 by 8.2e11" — a spectacular result that is entirely the
comparator overflowing. `scripts/report_kuairand_contrasts.py` now refuses to
name a winner when either arm produced non-finite predictions, exceeded ten
times the constant-predictor level, or ranked below chance on the primary
target; such rows report `not_comparable`.

### 3.3 Two learning rates were not enough

The pilot grid `{3e-4, 1e-3}` saw arms swing from 12.46 to 2.10 across a single
3× step, which means a two-point grid can publish the penalty of an unlucky
learning rate as though it were an architectural difference. The grid was
widened to `{1e-4, 3e-4, 1e-3, 3e-3}` — identically for every arm, so no arm
receives a tuning budget another lacks. `1e-4` immediately beat both original
points for `dcnv2_reduced` (1.568 against 2.283), confirming the original grid
was mis-centred.

## 4. How seeds are reported

The frozen matrix fixes `final_seeds: [42, 43, 44]` and also records
`seed_42_permitted_for: validation_tuning_only`. Because the test split is
locked, this study evaluates on validation — which is the split seed 42's
learning rate was selected on. Seed 42's number is therefore not an independent
measurement, leaving two clean seeds against an observed per-seed spread that
reached 0.937 / 1.790 / 1.697 in the pilot.

Every table is therefore emitted three times, tagged, never pooled:

* `frozen_protocol` — the three seeds the matrix freezes; **this is the
  protocol table**;
* `frozen_protocol_clean` — those three minus the tuning seed;
* `all_seeds` — the frozen three plus a stability annex (45–49), run to see
  whether a contrast survives more evidence.

The annex is not the frozen cell and never substitutes for it.

## 5. Reading the numbers

Two caveats govern every figure below.

**The aggregate is dominated by `watch_time`.** Per-target normalized losses
span 0.30 to 10.76 on `watch_time` while `long_view` sits near 0.5–1.1 and the
sparse heads at 0.3–2.0. Macro-NLoss is an unweighted mean by frozen decree —
adaptive task weights are explicitly forbidden — so it inherits that domination.
Anyone reading Macro-NLoss on KuaiRand is largely reading the regression head.

**1.0 is not the constant-predictor level here.** Task constants are fitted on
train, which is the *standard* exposure policy; validation is the *randomized*
pool. Positive rates differ sharply (`long_view` 0.3146 train against 0.0880
validation). Cross-arm comparison stays valid because every arm faces the same
shift; the absolute level against 1.0 does not.

For those reasons the per-target tables, and `long_view` ROC-AUC in particular
(the frozen `primary_confirmatory_target`), carry the interpretable signal.

## 6. Results

All figures are validation, `rms` preconditioner, four-point learning-rate grid.
**SEP** marks per-seed ranges that do not overlap.

### 6.1 `long_view` ROC-AUC — the frozen primary confirmatory target

| contrast | question | frozen (3) | clean (2) | annex (8) |
|---|---|---|---|---|
| C1 `dcdn_reduced − dcnv2_reduced` | same inputs, DCDN vs DCNv2 | **0.6171 / 0.5597 SEP** | **0.6282 / 0.5558 SEP** | not comparable |
| C2 `dcdn_reduced − afn_reduced` | same inputs, DCDN vs AFN+ | **0.6171 / 0.5600 SEP** | **0.6282 / 0.5457 SEP** | **0.6135 / 0.5525 SEP** |
| C3 `dcdn_reduced − dcnv2_capacity_reduced` | **RQ3** | 0.6171 / 0.6148 | 0.6282 / 0.6131 | 0.6135 / **0.6198** |
| C4 `dcnv2_full − dcnv2_reduced` | do the removed divisions matter | 0.5875 / 0.5597 | **0.5770 / 0.5558 SEP** | not comparable |
| C5 `dcdn_reduced − dcnv2_full` | **RQ2-quality** | 0.6171 / 0.5875 | **0.6282 / 0.5770 SEP** | 0.6135 / 0.6121 |
| C6 `dcdn_full − dcnv2_full` | full features, DCDN vs DCNv2 | **0.6266 / 0.5875 SEP** | **0.6232 / 0.5770 SEP** | 0.6397 / 0.6121 |
| C7 `afn_full − dcnv2_full` | full features, AFN+ vs DCNv2 | not comparable | not comparable | not comparable |

C7 is withheld at every scope because `afn_full` ranked below chance on
`long_view` on one seed (AUC 0.4545), which the convergence gate treats as a run
that produced no result. C1 and C4 become unquotable once the annex is included
because `dcnv2_reduced` exceeded ten times the constant-predictor level on an
annex seed.

Per-seed values behind the frozen column, for the contrasts that separate:

```
dcdn_reduced   0.594894  0.616066  0.640327
dcnv2_reduced  0.548314  0.563365  0.567471     C1 separates
afn_reduced    0.525423  0.566050  0.588663     C2 separates
dcnv2_capacity 0.606002  0.618066  0.620262     C3 does NOT separate
dcnv2_full     0.566726  0.587274  0.608623     C5 does NOT separate (frozen)
dcdn_full      0.616487  0.629877  0.633439     C6 separates
```

### 6.2 Macro-NLoss aggregate

| arm | frozen mean ± sd | annex mean ± sd | converged |
|---|---|---|---|
| `afn_reduced` | 0.693 ± 0.055 | 0.778 ± 0.111 | yes |
| `afn_full` | 0.739 ± 0.033 | 0.729 ± 0.057 | **no** |
| `dcnv2_capacity_reduced` | 0.939 ± 0.157 | 0.984 ± 0.127 | yes |
| `dcdn_full` | 1.256 ± 0.307 | 1.519 ± 0.498 | yes |
| `dcdn_reduced` | 1.497 ± 0.496 | 1.410 ± 0.330 | yes |
| `dcnv2_full` | 1.689 ± 0.596 | 1.410 ± 0.408 | yes |
| `dcnv2_reduced` | 2.891 ± 1.607 | 4.229 ± 2.292 | **no** |

### 6.3 The aggregate and the primary target disagree, and the aggregate is the misleading one

Per-target normalized losses, frozen scope:

| arm | long_view | like | forward | hate | watch_time |
|---|---|---|---|---|---|
| `afn_reduced` | 0.682 | 0.381 | 0.542 | 1.356 | **0.503** |
| `dcdn_reduced` | 0.909 | 0.555 | 1.049 | 1.935 | **3.038** |
| `dcnv2_reduced` | 1.104 | 0.347 | 0.620 | 1.621 | **10.762** |
| `dcdn_full` | **0.501** | 0.319 | 0.579 | 1.367 | 3.516 |
| `dcnv2_full` | 1.008 | 1.043 | 1.995 | 2.896 | 1.500 |

`watch_time` spans 0.50 to 10.76 while every other head sits between 0.32 and
2.90, so the unweighted mean the frozen objective mandates is largely a
`watch_time` report. This produces a genuine sign conflict: **C2 favours AFN+ on
the aggregate (ratio 2.162, separated) and DCDN on `long_view` AUC (separated).**
Both statements are true of the same runs. The aggregate is not wrong, but on
this dataset it answers "which arm calibrates a regression head best under
policy shift", not "which arm ranks best".

`hate` is above 1.0 for every arm at every scope: no arm learned it, as expected
from roughly 140 train positives.

## 7. What this does and does not establish

**Supported.**

* **DCDN beats both same-input baselines on the primary target.** On identical
  REDUCED inputs, DCDN's `long_view` AUC separates cleanly from DCNv2 (C1) and
  from AFN+ (C2). C2 survives all three seed scopes, which is the single most
  robust result here.
* **DCDN also wins on identical FULL inputs** (C6, separated at both frozen
  scopes) — so the advantage is not an artifact of which feature set it is given.

**Not supported.**

* **RQ3 fails on this dataset.** Against a mechanically capacity-matched DCNv2
  (rank 97, 1 layer, parameter and FLOP ratios 1.0025), DCDN's advantage vanishes:
  0.6171 vs 0.6148 frozen, never separated, and at eight seeds the sign
  **reverses** to 0.6135 vs 0.6198. Whatever DCDN gains over the two-layer DCNv2
  is obtainable by spending the same capacity differently. This is exactly what
  the stability annex was run to detect, and it detected it.
* **RQ2-quality is a tie, not a win.** C5 separates only in the two-seed clean
  scope; at the frozen three it does not separate, and across eight seeds it is
  0.6135 vs 0.6121 — a dead heat. The honest statement is that reduced-input DCDN
  is *statistically indistinguishable* from full-input DCNv2 on the primary
  target, not that it beats it.
* **The removed divisions are not redundant.** C4 shows DCNv2 does better with R
  than without it (separated in the clean scope). Removing them costs DCNv2 real
  quality; the interest of C5 is that DCDN recovers that cost, not that the
  features were worthless.

**Scope limits.**

* Validation only. The test split has never been unlocked, and none of this is
  the mandatory main-table `kuairand_pure` cell.
* Two arms failed the convergence gate on at least one seed (`afn_full`,
  `dcnv2_reduced`), which withholds C7 everywhere and C1/C4 at the annex scope.
  Those are unstable arms, not published wins or losses.
* The learning rate was tuned on seed 42 against validation, so the frozen
  three-seed column contains one tuning-contaminated point. The
  `frozen_protocol_clean` column exists precisely so that dependence is visible;
  note that C4 and C5 separate *only* there, on two seeds, which is the weakest
  evidence in this document and should not be quoted alone.
* Single dataset. Nothing here transfers to VK-LSVD, Yambda, or KuaiRec without
  running them.

## 8. Reproduction

```bash
source scripts/env.sh

# 1. Materialize (locked test window is excluded by date and never read)
python scripts/data/kuairand_pure_materialize.py

# 2. Seven arms x (4-point LR tuning on seed 42 + 8 final seeds), validation only
python scripts/run_kuairand_multitask.py \
    --device cuda:0 --preconditioner rms --annex-seeds \
    --output-dir results/kuairand_multitask/rq23_final_v1

# 3. Contrasts, per-target breakdown, and arm summary for all three seed scopes
python scripts/report_kuairand_contrasts.py \
    --results results/kuairand_multitask/rq23_final_v1/results_v1.json \
    --output-dir results/kuairand_multitask/contrasts_v1

# The rejected preconditioner, for the record
python scripts/run_kuairand_multitask.py \
    --device cuda:0 --preconditioner geometric \
    --output-dir results/kuairand_multitask/rq23_v1
```

The single-head `watch_time` follow-up of section 9, and its paired transfer
comparison against the corresponding multi-task head:

```bash
python scripts/run_kuairand_single_target.py \
    --device cuda:0 --preconditioner rms --annex-seeds \
    --output-dir results/kuairand_multitask/single_target_v1

python scripts/report_kuairand_contrasts.py \
    --results results/kuairand_multitask/single_target_v1/results_v1.json \
    --output-dir results/kuairand_multitask/contrasts_single_target_v1

python scripts/report_kuairand_transfer.py
```

Tests covering the leak-sensitive parts of the materializer, the contract order
of R, the seed scopes, and the convergence gate:

```bash
python -m pytest tests/test_kuairand_materialize.py tests/test_kuairand_contrasts.py
```

## 9. Single-head `watch_time` follow-up

Run because `watch_time` is the one head whose structure is a ratio, and
because section 6.3 showed the multi-target aggregate is largely a `watch_time`
report — which leaves "this arm models watch_time well" and "this arm's shared
trunk was pulled around by four binary heads" indistinguishable. Training it
alone separates them.

**Standing.** This is **not** the frozen single-target study.
`configs/single_target_followup_v1.yaml` requires
`training_may_start_only_after: main_multi_target_table_complete`, and the main
table is not complete. That study is unmodified and still blocked, and its
KuaiRand headline target is still `long_view`. This one is scoped by
`configs/data/kuairand_pure/single_target_rq23_v1.yaml`
(sha256 `07b6e22629650df6ceabcf362077e5bc72e8715ba92e3b3ccce981a3211713f5`)
and may not substitute for it or rescue any main-table conclusion.

Everything is held identical to the multi-target study — splits, materialized
views, preconditioner, encoder, optimizer, batch, steps, patience, grid, seeds.
The single difference is one regression head trained on `watch_time` alone. The
capacity match is recomputed for the narrower head, identically for every arm,
and resolves to the same rank 97.

### 9.1 Task transfer: strongly negative

Paired per arm and per seed against the corresponding multi-task head, 95%
percentile bootstrap over seeds. Positive difference = the single head won.

| arm | single | multi-task head | difference | 95% CI (8 seeds) | verdict |
|---|---|---|---|---|---|
| `dcnv2_reduced` | 4.881 | 17.497 | +12.615 | [+6.51, +20.84] | negative transfer |
| `dcdn_full` | 0.479 | 4.510 | +4.031 | [+2.70, +5.50] | negative transfer |
| `dcdn_reduced` | 0.983 | 2.653 | +1.670 | [+0.64, +2.79] | negative transfer |
| `dcnv2_full` | 0.990 | 1.577 | +0.588 | [−0.22, +1.32] | no detected effect |
| `afn_reduced` | 0.409 | 0.965 | +0.557 | [+0.20, +0.96] | negative transfer |
| `dcnv2_capacity_reduced` | 0.825 | 1.057 | +0.231 | [−0.19, +0.71] | no detected effect |
| `afn_full` | 0.294 | 0.448 | +0.154 | [+0.07, +0.24] | negative transfer |

The sign is the same for every one of the seven arms, and the single head wins
on 7 or 8 of 8 seeds almost everywhere. Multi-tasking costs `watch_time`
quality on this dataset; nothing here shows it buying anything back. The two
"no detected effect" rows are reported as exactly that, not as equality.

The comparison is only meaningful because the normalizer is shared. The report
verifies rather than assumes it: both bundles must record the same
`view_manifest_sha256`, and it refuses to run otherwise. The two
`task_constants_sha256` values are *not* comparable — one covers five targets,
the other one.

### 9.2 The architecture contrasts on the ratio target

Single-head `watch_time`, normalized loss, lower is better.

| arm | frozen mean ± sd | annex mean ± sd | MAE (s) |
|---|---|---|---|
| `afn_full` | **0.310 ± 0.094** | **0.294 ± 0.059** | 12.4 |
| `afn_reduced` | 0.370 ± 0.017 | 0.409 ± 0.100 | 16.8 |
| `dcdn_full` | 0.476 ± 0.018 | 0.479 ± 0.107 | 17.0 |
| `dcnv2_capacity_reduced` | 0.760 ± 0.126 | 0.825 ± 0.207 | 27.8 |
| `dcnv2_full` | 0.880 ± 0.289 | 0.990 ± 0.916 | 20.8 |
| `dcdn_reduced` | 0.992 ± 0.273 | 0.983 ± 0.364 | 23.8 |
| `dcnv2_reduced` | 3.512 ± 1.073 | 4.881 ± 2.901 | 43.9 |

| contrast | frozen (3) | clean (2) | annex (8) |
|---|---|---|---|
| C1 `dcdn_reduced − dcnv2_reduced` | **3.541× SEP** | **3.614× SEP** | not comparable |
| C2 `dcdn_reduced − afn_reduced` | **AFN+ 2.680× SEP** | **AFN+ 3.016× SEP** | AFN+ 2.406× |
| C3 `dcdn_reduced − dcnv2_capacity_reduced` | DCNv2 1.306× | **DCNv2 1.471× SEP** | DCNv2 1.191× |
| C4 `dcnv2_full − dcnv2_reduced` | **3.993× SEP** | **3.949× SEP** | not comparable |
| C5 `dcdn_reduced − dcnv2_full` | DCNv2 1.128× | **DCNv2 1.093× SEP** | DCDN 1.007× |
| C6 `dcdn_full − dcnv2_full` | **DCDN 1.849× SEP** | **DCDN 2.174× SEP** | DCDN 2.067× |
| C7 `afn_full − dcnv2_full` | **AFN+ 2.834× SEP** | **AFN+ 3.037× SEP** | AFN+ 3.370× |

### 9.3 What the ratio target adds, and what it takes away

**It strengthens DCDN against DCNv2.** On the pure ratio target DCDN beats
DCNv2 by 3.54× at identical REDUCED inputs and 1.85× at identical FULL inputs,
both separated — a far larger and cleaner margin than the multi-target
aggregate or `long_view` AUC gave. That is the predicted direction: this is the
target whose structure the reciprocal branch is built for.

**It does not rescue RQ3.** C3 still favours the capacity-matched DCNv2, and in
the clean scope it now separates *against* DCDN. Two independent targets now
agree that the reciprocal structure buys nothing over spending the same
capacity differently.

**It produces the sharpest counter-evidence in this document.** AFN+ beats DCDN
on `watch_time` by 2.68–3.02×, separated in both frozen scopes, and `afn_full`
is the best arm outright. That is coherent rather than surprising: AFN+'s
logarithmic transformation layer turns a ratio into a difference of logs, which
is a *more* direct inductive bias for this target than a reciprocal branch. On
the target that most favours modelling quotients, the log-based competitor wins.

**C4 gets much stronger.** The removed manual divisions are worth 3.99× to
DCNv2 on `watch_time`, against 1.71× on the aggregate. Historical
`play_over_duration` ratios are close relatives of the target, so this is
expected — and it further weakens any reading of C5 as "the divisions were
redundant". C5 remains a tie: DCNv2-full is ahead in the frozen scope and the
two are level at eight seeds.

