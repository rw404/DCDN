# RQ3 — quality against serving cost on three datasets

**The claim this file tests:** at the same input regime, DCDN buys more ranking
quality per millisecond than DCNv2 and AFN+. It does, on all three report
slices, and the margin survives a paired seed-by-seed test in every case. The
sections below also say where the margin comes from, because on one dataset it
is earned on latency rather than on quality, and that distinction matters more
than the headline.

Seven arms, three frozen report slices, quality and cost describing the **same
configurations**: each benchmark runs the exact arm configs the quality run
trained, at the exact operand widths of that dataset's geometry.

| dataset | slice | quality | cost |
|---|---|---|---|
| Yambda-50M | `listen50` | `results/yambda_listen/l50_field_v1` | `batch_sweep_eager_v1` |
| KuaiRec | `watch_ratio_gt2` | `results/kuairec_watch_time/binary_field_v1` | `batch_sweep_field_eager_v1` |
| KuaiRand-Pure | `watch_time` | `results/kuairand_multitask/field_watch_time_rms_v1` | `batch_sweep_field_v1` |

Eight seeds per arm, learning rate tuned per arm. Latency is the eager-mode p50
at **B = 8192** — the work-bound end of the production batch range, where
arithmetic cost dominates rather than launch overhead. CUDA-graph sweeps exist
and are not used here: private memory pools accumulate across captures, which
makes peak VRAM meaningless per arm.

**`RED*` marks `dcnv2_capacity_reduced`**, the only mechanically matched
comparator — within ±5% of `dcdn_reduced` on trainable non-embedding parameters
and FLOPs. AFN+ and plain DCNv2 carry 2x to 44x more parameters, so a
difference against them confounds architecture with capacity.


### Yambda-50M / listen50

| arm | inputs | params | AUC | sd | gAUC | NDCG@5 | p50 @ B=8192 | quality/ms |
|---|---|---|---|---|---|---|---|---|
| dcnv2_full | FULL | 112,469 | 0.81989 | 0.00108 | 0.60847 | 0.7427 | 0.935 ms | 0.3422 |
| dcnv2_reduced | RED | 74,357 | 0.81433 | 0.00095 | 0.60713 | 0.7451 | 0.931 ms | 0.3375 |
| dcnv2_capacity_reduced | RED* | 44,034 | 0.81555 | 0.00063 | 0.60891 | 0.7480 | 1.207 ms | 0.2614 |
| **dcdn_full** | FULL | 69,845 | 0.82065 | 0.00047 | 0.61267 | 0.7480 | 0.924 ms | 0.3472 |
| **dcdn_reduced** | RED | 44,021 | 0.81492 | 0.00065 | 0.60873 | 0.7503 | 0.925 ms | 0.3405 |
| afn_full | FULL | 2,224,727 | 0.82158 | 0.00042 | 0.61402 | 0.7491 | 5.369 ms | 0.0599 |
| afn_reduced | RED | 1,822,199 | 0.81661 | 0.00098 | 0.61124 | 0.7526 | 4.324 ms | 0.0732 |

`quality/ms` = (AUC - 0.5) / latency. A crude ratio, offered as an ordering aid, not a decision metric.

| pair | mean diff | se | seeds | verdict | latency ratio |
|---|---|---|---|---|---|
| dcdn_reduced - dcnv2_capacity_reduced | -0.00064 | 0.00015 | 0/8 | **significant** | 1.31x |
| dcdn_reduced - dcnv2_reduced | +0.00059 | 0.00029 | 6/8 | **significant** | 1.01x |
| dcdn_reduced - afn_reduced | -0.00169 | 0.00048 | 1/8 | **significant** | 4.68x |
| dcdn_reduced - dcnv2_full | -0.00497 | 0.00021 | 0/8 | **significant** | 1.01x |
| dcdn_reduced - afn_full | -0.00667 | 0.00030 | 0/8 | **significant** | 5.80x |

Paired on `quality/ms`, best DCDN arm against each comparator:

| pair | mean diff | se | seeds | verdict |
|---|---|---|---|---|
| dcdn_full - dcnv2_full | +0.0050 | 0.0004 | 8/8 | **significant** |
| dcdn_full - dcnv2_reduced | +0.0097 | 0.0003 | 8/8 | **significant** |
| dcdn_full - dcnv2_capacity_reduced | +0.0858 | 0.0002 | 8/8 | **significant** |
| dcdn_full - dcdn_reduced | +0.0067 | 0.0003 | 8/8 | **significant** |
| dcdn_full - afn_full | +0.2873 | 0.0002 | 8/8 | **significant** |
| dcdn_full - afn_reduced | +0.2740 | 0.0002 | 8/8 | **significant** |

### KuaiRec / watch_ratio_gt2

| arm | inputs | params | AUC | sd | gAUC | NDCG@5 | p50 @ B=8192 | quality/ms |
|---|---|---|---|---|---|---|---|---|
| dcnv2_full | FULL | 93,345 | 0.80482 | 0.00525 | 0.77091 | 0.4166 | 0.952 ms | 0.3201 |
| dcnv2_reduced | RED | 60,685 | 0.80536 | 0.00567 | 0.77124 | 0.4227 | 0.960 ms | 0.3182 |
| dcnv2_capacity_reduced | RED* | 31,794 | 0.80655 | 0.00368 | 0.77323 | 0.4265 | 1.225 ms | 0.2502 |
| **dcdn_full** | FULL | 49,665 | 0.80774 | 0.00349 | 0.77560 | 0.4364 | 0.942 ms | 0.3268 |
| **dcdn_reduced** | RED | 31,725 | 0.80715 | 0.00388 | 0.77295 | 0.4342 | 0.947 ms | 0.3244 |
| afn_full | FULL | 1,914,381 | 0.80533 | 0.00393 | 0.77450 | 0.4264 | 4.607 ms | 0.0663 |
| afn_reduced | RED | 1,528,625 | 0.80114 | 0.00582 | 0.77379 | 0.4296 | 3.756 ms | 0.0802 |

`quality/ms` = (AUC - 0.5) / latency. A crude ratio, offered as an ordering aid, not a decision metric.

| pair | mean diff | se | seeds | verdict | latency ratio |
|---|---|---|---|---|---|
| dcdn_reduced - dcnv2_capacity_reduced | +0.00060 | 0.00154 | 4/8 | not significant | 1.29x |
| dcdn_reduced - dcnv2_reduced | +0.00180 | 0.00152 | 4/8 | not significant | 1.01x |
| dcdn_reduced - afn_reduced | +0.00601 | 0.00155 | 8/8 | **significant** | 3.97x |
| dcdn_reduced - dcnv2_full | +0.00233 | 0.00130 | 6/8 | not significant | 1.01x |
| dcdn_reduced - afn_full | +0.00182 | 0.00205 | 5/8 | not significant | 4.87x |

Paired on `quality/ms`, best DCDN arm against each comparator:

| pair | mean diff | se | seeds | verdict |
|---|---|---|---|---|
| dcdn_full - dcnv2_full | +0.0067 | 0.0017 | 8/8 | **significant** |
| dcdn_full - dcnv2_reduced | +0.0086 | 0.0017 | 8/8 | **significant** |
| dcdn_full - dcnv2_capacity_reduced | +0.0765 | 0.0014 | 8/8 | **significant** |
| dcdn_full - dcdn_reduced | +0.0023 | 0.0006 | 7/8 | **significant** |
| dcdn_full - afn_full | +0.2605 | 0.0014 | 8/8 | **significant** |
| dcdn_full - afn_reduced | +0.2466 | 0.0012 | 8/8 | **significant** |

### KuaiRand-Pure / watch_time

| arm | inputs | params | xAUC | sd | user xAUC | p50 @ B=8192 | quality/ms |
|---|---|---|---|---|---|---|---|
| dcnv2_full | FULL | 325,833 | 0.51854 | 0.01419 | 0.50140 | 1.186 ms | 0.0156 |
| dcnv2_reduced | RED | 173,289 | 0.51470 | 0.00937 | 0.49972 | 0.817 ms | 0.0180 |
| dcnv2_capacity_reduced | RED* | 72,850 | 0.54128 | 0.01128 | 0.50179 | 1.037 ms | 0.0398 |
| **dcdn_full** | FULL | 167,881 | 0.52935 | 0.01600 | 0.50765 | 0.961 ms | 0.0306 |
| **dcdn_reduced** | RED | 72,681 | 0.54075 | 0.00521 | 0.50693 | 0.710 ms | 0.0574 |
| afn_full | FULL | 3,180,475 | 0.55552 | 0.01173 | 0.50999 | 7.955 ms | 0.0070 |
| afn_reduced | RED | 2,241,243 | 0.54291 | 0.01866 | 0.50281 | 5.518 ms | 0.0078 |

`quality/ms` = (xAUC - 0.5) / latency. A crude ratio, offered as an ordering aid, not a decision metric.

| pair | mean diff | se | seeds | verdict | latency ratio |
|---|---|---|---|---|---|
| dcdn_reduced - dcnv2_capacity_reduced | -0.00053 | 0.00388 | 1/8 | not significant | 1.46x |
| dcdn_reduced - dcnv2_reduced | +0.02605 | 0.00305 | 8/8 | **significant** | 1.15x |
| dcdn_reduced - afn_reduced | -0.00215 | 0.00681 | 4/8 | not significant | 7.78x |
| dcdn_reduced - dcnv2_full | +0.02221 | 0.00557 | 8/8 | **significant** | 1.67x |
| dcdn_reduced - afn_full | -0.01477 | 0.00358 | 1/8 | **significant** | 11.21x |

Paired on `quality/ms`, best DCDN arm against each comparator:

| pair | mean diff | se | seeds | verdict |
|---|---|---|---|---|
| dcdn_reduced - dcnv2_full | +0.0418 | 0.0053 | 8/8 | **significant** |
| dcdn_reduced - dcnv2_reduced | +0.0394 | 0.0038 | 8/8 | **significant** |
| dcdn_reduced - dcnv2_capacity_reduced | +0.0176 | 0.0040 | 8/8 | **significant** |
| dcdn_reduced - dcdn_full | +0.0269 | 0.0058 | 8/8 | **significant** |
| dcdn_reduced - afn_full | +0.0504 | 0.0024 | 8/8 | **significant** |
| dcdn_reduced - afn_reduced | +0.0497 | 0.0028 | 8/8 | **significant** |

---

## What holds, and where it comes from

### The headline

On every slice the best DCDN arm has the highest `quality/ms` in the table, and
the paired test confirms it against all six comparators, on 7 or 8 seeds out of
8, every time. That is 18 of 18 pairwise comparisons.

| slice | best DCDN arm | its ratio | best non-DCDN | margin |
|---|---|---|---|---|
| Yambda `listen50` | `dcdn_full` | 0.3472 | `dcnv2_full` 0.3422 | +1.5% |
| KuaiRec `watch_ratio_gt2` | `dcdn_full` | 0.3268 | `dcnv2_full` 0.3201 | +2.1% |
| KuaiRand `watch_time` | `dcdn_reduced` | 0.0574 | `dcnv2_capacity_reduced` 0.0398 | **+44%** |

### Where the margin is earned — decomposed

The ratio has two factors and they do not move together. Splitting
`dcdn_reduced` against its capacity-matched twin:

| slice | AUC difference | verdict | latency ratio | net ratio win |
|---|---|---|---|---|
| Yambda | **-0.00064**, 0/8 seeds | **significantly worse** | 1.31x faster | yes |
| KuaiRec | +0.00060, 4/8 | not significant | 1.29x faster | yes |
| KuaiRand | -0.00053, 1/8 | not significant | 1.46x faster | yes |

**On Yambda the ratio win is entirely a latency win.** `dcdn_reduced` loses on
AUC to `dcnv2_capacity_reduced` on all eight seeds and still comes out ahead per
millisecond, because it runs 1.31x faster at equal parameters and FLOPs. That
is a legitimate serving argument and it is *not* an accuracy argument, and the
two should not be conflated.

On KuaiRec and KuaiRand the quality difference against the matched comparator
is not significant in either direction, so there too the ratio is carried by
latency. **Across all three datasets DCDN does not beat its capacity-matched
comparator on quality on any slice.** What it does, consistently, is deliver
comparable quality 1.3x-1.5x faster.

### Why DCDN is faster at equal capacity

Capacity matching equalizes analytic parameter and FLOP counts; it does not
equalize realized kernels. DCDN's masked reciprocal carrier plus one product
layer maps onto fewer, larger operations than DCNv2's two low-rank layers, and
the fused inference path removes the per-call device-to-host synchronization
that the reference implementation pays. The 1.29x-1.46x gap is stable across
all three geometries, which are 50/96, 86/134 and 132/244 operands wide.

### AFN+ is the clearest result in the file

AFN+ carries 1.5M-3.2M parameters against DCDN's 32k-73k, costs 4x-11x more
latency, and does not buy proportionate quality: on KuaiRec `afn_reduced` is
**significantly worse** than `dcdn_reduced` (+0.00601, 8/8 seeds) while costing
3.97x more. On Yambda and KuaiRand it wins on AUC but at 4.7x and 7.8x the
latency. Its `quality/ms` is 4x-7x below DCDN's on every slice.

## Four caveats that bound the claim

**1. The ratio is not scale-free.** `quality/ms` subtracts 0.5 before dividing,
so it measures margin over chance per millisecond. Where AUC is high the
subtraction barely matters; on KuaiRand `watch_time`, where xAUC sits at
0.515-0.556, the numerator is small and a 0.005 quality difference moves the
ratio by tens of percent. The +44% margin on that slice is therefore far less
robust than the +1.5% on Yambda, despite looking larger. Read the two component
columns, not the ratio alone.

**2. KuaiRand `watch_time` is unstable for part of the field.** Counting seeds
whose macro NLoss exceeds 1.0 — worse than predicting the train mean —
`dcnv2_reduced` fails 8/8, `dcdn_reduced` 5/8 and `dcnv2_full` 4/8. The frozen
convergence gate fires at 10x and lets all of these through. `dcdn_reduced`
beats `dcnv2_full` by 0.0222 on all eight seeds there, but restricted to the
two seeds where both arms beat the constant predictor, `dcnv2_full` wins both.
**That slice's quality column should not carry weight**; its latency column is
unaffected. See `docs/kuairand_watch_time.md` §2.

**3. NDCG is absent for KuaiRand by contract.**
`configs/data/kuairand_pure/features_v1.yaml` records `no_ndcg_reason:
no_released_complete_candidate_lists_per_request`. The Yambda and KuaiRec NDCG
figures use "all of a user's validation rows" as the ranking unit, which is a
construction of this study and not a released candidate list — true there too,
but those contracts make no statement forbidding it.

**4. B = 8192 is a choice.** At B = 1 the ordering differs: both architectures
are launch-bound and fixed overhead dominates, so on KuaiRand `dcnv2_reduced`
(0.596 ms) and `dcnv2_full` (0.601 ms) are ahead of `dcdn_reduced` (0.648 ms).
The DCDN advantage is a work-bound-regime advantage and should be quoted as
one. A ranker doing dynamic batching over 1..16000 spends most of its examples
in the work-bound regime, which is why that end is reported.

## The defensible statement

> Across Yambda `listen50`, KuaiRec `watch_ratio_gt2` and KuaiRand-Pure
> `watch_time`, DCDN delivers the best ranking quality per millisecond of the
> seven arms compared, significantly so against all six comparators on every
> slice. The margin is earned on latency rather than on accuracy: at matched
> parameters and FLOPs DCDN is 1.29x-1.46x faster and statistically
> indistinguishable in quality on two of three slices, and significantly worse
> on the third. Against AFN+, which costs 4x-11x more, DCDN's advantage is
> 4x-7x on the ratio and reaches significance on quality alone on KuaiRec.

What this does **not** say: that DCDN is the most accurate architecture. It is
not, on any of the three slices.

## Reproduction

```bash
source scripts/env.sh
python scripts/report_rq3_matrix.py
```
