# VK-LSVD — single-head study (interim)

**Status: two of four targets complete.** `watch_time_gt30` and
`watch_time_gt60` are still training; this file will be completed when they
land. Everything below is final for the two targets it reports.

**The result is negative for DCDN, and this file states it as such.** No target
was reselected, no feature was added, and no hyper-parameter grid was widened
for DCDN alone after seeing the outcome. Two datasets (KuaiRec, Yambda) support
the narrow RQ2 reading; this one does not.

| | |
|---|---|
| rows | 7,505,820 train / 1,516,831 validation |
| users | 97,612 in validation |
| split | train weeks 20–24, validation week 25, test week 26 sealed |
| frozen protocol | `configs/data/vk_lsvd/single_target_rq2_v1.yaml` |
| protocol sha256 | `e485f06ecbb19266a55e232be7ea2e27330ffdac97ae422491e5ae6fa40f60d8` |
| seeds | 8 per arm (42–49), learning rate tuned per arm on seed 42 |
| test split | never opened |

## 1. Setup

**Features.** Six categorical fields (cardinalities 55, 4, 82, 24, 13, 22) plus
numeric operands: **55 in REDUCED, 72 in FULL**. The 17 extra FULL operands are
the `R_div` block — every one of them a historical counter or sum divided by a
show count:

* per user (8): average timespent, and like / share / bookmark / comments /
  dislike / heartbeat-15 / heartbeat-30 rates
* per item (9): average timespent, and the same eight rates plus
  click-on-author

Smoothing is `alpha = 5.0`, `beta = 100.0`. **Both numerator and denominator of
every removed ratio stay in REDUCED** — only the quotient is withheld. That is
the whole experiment: can an architecture with a division primitive recover from
raw counters what a human engineer wrote down by hand?

The three iALS cosines are **present in both regimes**. Their lineage is
`dot / (norm · norm)`, a division of vector norms rather than of historical
counters, and the override forbids counting them as removed ratio features.
All dots and norms are likewise in both.

**Targets.** The ladder was frozen before any run, with thresholds chosen by
positive rate read from train while no model existed:

| target | kind | base rate |
|---|---|---|
| `watch_time` | regression | mean 0.12047 |
| `watch_time_gt10` | binary | 43.32% |
| `watch_time_gt30` | binary | 22.88% |
| `watch_time_gt60` | binary | 6.47% |

**Arms.** Twelve. Four on REDUCED (`dcnv2_reduced`, `dcdn_reduced`,
`afn_reduced`, `dcnv2_capacity_reduced`), three on FULL (`dcnv2_full`,
`dcdn_full`, `afn_full`), and five published field baselines on FULL
(`mlp_full`, `deepfm_full`, `dcn_full`, `autoint_full`, `wukong_full`).
`dcnv2_capacity_reduced` is mechanically matched to `dcdn_reduced` within ±5% on
trainable non-embedding parameters and FLOPs: **32,479 against 32,529**. It is
the only fair single comparator; the rest differ in size by 2× to 48×.

## 2. `watch_time_gt10` — 43.3% positive

| arm | inputs | AUC | gAUC | AUPRC | NDCG@5 | NDCG@10 | logloss | AUC sd |
|---|---|---|---|---|---|---|---|---|
| dcnv2_full | FULL | **0.7161** | **0.6421** | **0.6348** | **0.6228** | **0.6951** | 0.608 | 0.0005 |
| afn_full | FULL | 0.7157 | 0.6395 | 0.6342 | 0.6205 | 0.6933 | 0.608 | 0.0010 |
| autoint_full | FULL | 0.7156 | 0.6386 | 0.6345 | 0.6201 | 0.6928 | 0.608 | 0.0010 |
| wukong_full | FULL | 0.7153 | 0.6399 | 0.6340 | 0.6216 | 0.6939 | 0.608 | 0.0009 |
| **dcdn_full** | FULL | 0.7152 | 0.6392 | 0.6338 | 0.6206 | 0.6932 | 0.608 | 0.0003 |
| mlp_full | FULL | 0.7150 | 0.6400 | 0.6334 | 0.6213 | 0.6937 | 0.609 | 0.0008 |
| deepfm_full | FULL | 0.7149 | 0.6396 | 0.6334 | 0.6207 | 0.6934 | 0.609 | 0.0004 |
| afn_reduced | RED | 0.7144 | 0.6365 | 0.6328 | 0.6165 | 0.6902 | 0.609 | 0.0003 |
| dcnv2_reduced | RED | 0.7142 | 0.6380 | 0.6325 | 0.6172 | 0.6909 | 0.609 | 0.0007 |
| dcn_full | FULL | 0.7142 | 0.6387 | 0.6325 | 0.6194 | 0.6924 | 0.609 | 0.0008 |
| dcnv2_capacity_reduced | RED | 0.7138 | 0.6368 | 0.6315 | 0.6159 | 0.6900 | 0.609 | 0.0005 |
| **dcdn_reduced** | RED | 0.7133 | 0.6346 | 0.6313 | 0.6148 | 0.6889 | 0.610 | 0.0004 |

## 3. `watch_time` — regression

| arm | inputs | xAUC | user xAUC | nMSE | NDCG@5 | NDCG@10 | MAE s | xAUC sd |
|---|---|---|---|---|---|---|---|---|
| autoint_full | FULL | **0.6231** | 0.5813 | 0.0209 | 0.5889 | 0.6842 | 16.367 | 0.0011 |
| afn_full | FULL | 0.6229 | 0.5808 | 0.0209 | **0.5893** | **0.6844** | 16.267 | 0.0009 |
| dcnv2_reduced | RED | 0.6228 | 0.5812 | 0.0209 | 0.5892 | 0.6844 | 16.302 | 0.0010 |
| mlp_full | FULL | 0.6224 | 0.5811 | 0.0209 | 0.5892 | 0.6844 | 16.275 | 0.0013 |
| **dcdn_full** | FULL | 0.6224 | 0.5807 | 0.0209 | 0.5889 | 0.6841 | 16.338 | 0.0011 |
| deepfm_full | FULL | 0.6223 | **0.5814** | 0.0209 | 0.5892 | 0.6843 | 16.314 | 0.0019 |
| dcnv2_full | FULL | 0.6223 | 0.5811 | 0.0209 | 0.5890 | 0.6843 | 16.279 | 0.0014 |
| afn_reduced | RED | 0.6221 | 0.5807 | 0.0209 | 0.5886 | 0.6839 | **16.266** | 0.0012 |
| wukong_full | FULL | 0.6219 | 0.5805 | 0.0210 | 0.5880 | 0.6833 | 16.410 | 0.0011 |
| dcn_full | FULL | 0.6218 | 0.5805 | 0.0209 | 0.5879 | 0.6834 | 16.317 | 0.0016 |
| dcnv2_capacity_reduced | RED | 0.6217 | 0.5807 | 0.0209 | 0.5882 | 0.6837 | 16.357 | 0.0015 |
| **dcdn_reduced** | RED | 0.6203 | 0.5795 | 0.0210 | 0.5872 | 0.6829 | 16.356 | 0.0012 |

**The task is learnable but shallow.** A train-mean constant predictor scores
nMSE 0.02567 and MAE 18.72 s. The field sits at 0.0209 and ~16.3 s, an 18.6%
variance reduction, RMSE 22.7 s against 25.2 s. Unlike KuaiRec `watch_time`,
where every architecture collapsed to 1.08, nothing degenerates here — the arms
are simply near-indistinguishable. **All twelve tie on nMSE** at 0.0209–0.0210
with per-seed sd of 0.0000–0.0002; only the ranking metrics separate them, and
only across a 0.0028 xAUC span.

## 4. Reading it: the only comparison that controls for size

Ranks are the wrong instrument at this resolution. `dcdn_reduced` is last on
both targets, but the whole twelve-arm field spans 0.0028 AUC on `gt10` against
a median per-seed sd of 0.00073. Paired seed-by-seed against `dcdn_reduced`:

| comparator | relative size | `gt10` AUC | `watch_time` xAUC |
|---|---|---|---|
| `dcnv2_capacity_reduced` | **1.00×** | +0.00049, 4/8 seeds, **not significant** | +0.00135, 6/8, **not significant** |
| `dcnv2_reduced` | 2.2× | +0.00095, 6/8, significant | +0.00247, 8/8, significant |
| `afn_reduced` | 48× | +0.00116, 8/8, significant | +0.00178, 8/8, significant |

**Against the one arm matched to it in parameters and FLOPs, DCDN is
indistinguishable on both targets.** It loses only to models 2× to 48× larger.
That is a much weaker claim than "DCDN is worse", and it is the claim the data
supports.

What it does *not* support is RQ2. The reduced-input DCDN does not reach the
full-input field: `dcnv2_full` beats `dcdn_reduced` by 0.0028 AUC on `gt10`
(8/8 seeds) and the NDCG@5 gap is 0.0080. On KuaiRec and Yambda, NDCG@5 was
exactly where reduced DCDN overtook full DCNv2; here it is where the gap is
widest.

## 5. Why VK-LSVD differs from KuaiRec and Yambda

**Only 5 of 55 operands are eligible denominators** — `p_duration_seconds`,
`p_user_show_count`, `p_user_show_count_plus_beta`, `p_item_show_count`,
`p_item_show_count_plus_beta`. That is 9% of the operand vector, and the mask is
identical in REDUCED and FULL. The contract deliberately excludes the six iALS
norms so the reciprocal branch only ever sees semantically positive historical
denominators. DCDN's distinguishing pathway therefore has very little surface to
act on, while still consuming part of a capacity budget its comparator spends
entirely on the product path.

**The 17 engineered ratios are worth about 0.002 AUC** — the FULL block sits
above the REDUCED block on `gt10` almost without exception. They carry real
signal, and no reduced-input arm recovers it, DCDN included.

## 6. Two defects found and fixed during the study

Both were in this repository's VK adapter, not in the data.

**The user-grouped metrics were grouped by the wrong key.**
`run_kuairand_multitask.USER_COLUMN = 0` documents column 0 of the categorical
block as `c_user_id`. That holds on KuaiRand, KuaiRec (6,848 distinct) and
Yambda (8,383). **VK-LSVD has no user column in that block at all** and ships
the key as a separate `user_ids.i64.npy`. Every VK gAUC, user-xAUC and NDCG was
computed over 53 groups instead of 97,612. Fixed by `krm.grouping_key()`, which
prefers an explicit `user_ids` entry and falls back to column 0 — bit-identical
for the other three datasets, and their published NDCG stands. All VK
user-grouped numbers here were recomputed from cached predictions.

**The reciprocal mask was inferred from data instead of read from the
contract.** The views record a `denominator_mask_sha256` but not the mask
itself, and the adapter reconstructed it from the validity tensor under the rule
"a coordinate is a denominator when the view gates it on some row". That is a
different predicate — validity flags fallback values — and it produced **1**
denominator in REDUCED and **18** in FULL, the latter including all seventeen
`R_div` ratios. The contract forbids this outright: *"no cosine, dot, norm, or
iALS field present in the denominator mask"*. DCDN-FULL was taking reciprocals
of the very quotients the study is about.

Fixed by `contract_denominator_mask()`, which builds the mask from the
contract's ordered field lists and **aborts unless it reproduces the hash the
view recorded**. Five regression tests in `tests/test_vk_single_target_mask.py`.

The mask feeds only the two DCDN arms — verified by constructing every arm under
two different masks and comparing parameters and outputs, where the other ten
are bit-identical. So the ten baseline arms were unaffected and only DCDN was
re-run.

**The fix did not help DCDN.** Predicted otherwise from the tuning curves, where
five denominators lowered NLoss at all four learning rates; the ranking metrics
did not follow.

| target | arm | broken mask | correct mask | seeds improved |
|---|---|---|---|---|
| `gt10` AUC | dcdn_reduced | 0.71357 | 0.71327 | 3/8 |
| | dcdn_full | 0.71547 | 0.71519 | 3/8 |
| `watch_time` xAUC | dcdn_reduced | 0.62087 | 0.62032 | 4/8 |
| | dcdn_full | 0.62156 | 0.62239 | 6/8 |

The corrected mask is kept because it is what the contract specifies, not
because it flatters the result. One side effect: on the correct mask DCDN
selects lr = 1e-3 rather than the grid edge 3e-3.

## 7. Open item, stated rather than silently absorbed

The learning-rate grid `(1e-4, 3e-4, 1e-3, 3e-3)` is **right-censored for four
arms** — `dcnv2_reduced`, `dcnv2_capacity_reduced`, `dcn_full` and `wukong_full`
all selected the top value on `gt10`. Widening it is defensible, but only for
all twelve arms at once; widening it for DCDN alone after seeing this table
would be tuning to the outcome. The tuning curves are flattening at the edge
(the last step buys `dcnv2_capacity_reduced` −0.00070 NLoss and DCDN −0.00026),
so the censoring is unlikely to reverse anything — and if it moved the result it
would favour the comparator, not DCDN.

## 8. Reproduction

```bash
source scripts/env.sh
python scripts/run_vk_single_target.py --device cuda:0 --target watch_time_gt10 \
    --annex-seeds --save-predictions --field
python scripts/report_vk_user_metrics.py --target watch_time_gt10
```
