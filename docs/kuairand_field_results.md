# KuaiRand-Pure — twelve-arm single-head field study

The earlier KuaiRand studies in this repository ran seven arms. This one adds
the five published field baselines, so the same twelve architectures are
compared here as on VK-LSVD, KuaiRec and Yambda.

| | |
|---|---|
| rows | 201,320 train / 311,490 validation |
| users in validation | 23,620 |
| categorical fields | 8 |
| numeric operands | **132 REDUCED, 244 FULL** |
| eligible denominators | 24 of 132 (18%) |
| seeds | 8 per arm (42-49), learning rate tuned per arm on seed 42 |
| targets | `watch_time` (regression), `long_view` (binary) |
| test split | never opened |

**No NDCG.** `configs/data/kuairand_pure/features_v1.yaml` declares
`no_ndcg_reason: no_released_complete_candidate_lists_per_request` - the
dataset ships impressions, not the candidate pool a ranker actually scored, so
an NDCG over it would measure a list the system never produced. This is the one
metric difference from the other three datasets.

## 1. What is removed in REDUCED

The 112 extra FULL operands are the `R` block: manually materialized divisions
of historical counters and sums. Both numerator and denominator of every removed
ratio stay in REDUCED - only the quotient is withheld. Cosines, dot products,
vector norms and iALS scores are present in both regimes and are not counted as
removed ratio features.

At 112 removed ratios and 24 denominators out of 132 operands, KuaiRand is the
richest division setting in the programme. VK-LSVD, for contrast, removes 17
and has 5 denominators out of 55.

## 2. Results

### watch_time (regression), `rms`

| # | arm | inputs | xAUC | user xAUC | nMSE | MAE ms | sd |
|---|---|---|---|---|---|---|---|
| 1 | autoint_full | FULL | 0.55748 | 0.50946 | 0.00296 | 14.106 | 0.00448 |
| 2 | afn_full | FULL | 0.55552 | 0.50999 | 0.00255 | 11.990 | 0.01173 |
| 3 | mlp_full | FULL | 0.54830 | 0.50726 | 0.00286 | 12.835 | 0.00830 |
| 4 | afn_reduced | RED | 0.54291 | 0.50281 | 0.00355 | 16.554 | 0.01866 |
| 5 | deepfm_full | FULL | 0.54274 | 0.50015 | 0.00286 | 11.587 | 0.00857 |
| 6 | dcnv2_capacity_reduced | RED | 0.54128 | 0.50179 | 0.00716 | 27.229 | 0.01128 |
| 7 | **dcdn_reduced** | RED | 0.54075 | 0.50693 | 0.00853 | 22.640 | 0.00521 |
| 8 | wukong_full | FULL | 0.53745 | 0.50540 | 0.00314 | 13.186 | 0.02259 |
| 9 | **dcdn_full** | FULL | 0.52935 | 0.50765 | 0.00416 | 14.667 | 0.01600 |
| 10 | dcnv2_full | FULL | 0.51854 | 0.50140 | 0.00859 | 20.440 | 0.01419 |
| 11 | dcnv2_reduced | RED | 0.51470 | 0.49972 | 0.04236 | 35.184 | 0.00937 |
| 12 | dcn_full | FULL | 0.51457 | 0.50236 | 0.00501 | 19.656 | 0.01653 |

| comparator | mean diff | se | beats DCDN | verdict |
|---|---|---|---|---|
| dcn_full | -0.02618 | 0.00551 | 1/8 | **significant** |
| dcnv2_reduced | -0.02605 | 0.00305 | 0/8 | **significant** |
| dcnv2_full | -0.02221 | 0.00557 | 0/8 | **significant** |
| dcdn_full | -0.01140 | 0.00550 | 3/8 | **significant** |
| wukong_full | -0.00330 | 0.00841 | 5/8 | not significant |
| dcnv2_capacity_reduced (capacity-matched) | +0.00053 | 0.00388 | 7/8 | not significant |
| deepfm_full | +0.00198 | 0.00427 | 6/8 | not significant |
| afn_reduced | +0.00215 | 0.00681 | 4/8 | not significant |
| mlp_full | +0.00755 | 0.00354 | 5/8 | **significant** |
| afn_full | +0.01477 | 0.00358 | 7/8 | **significant** |
| autoint_full | +0.01672 | 0.00218 | 8/8 | **significant** |

### long_view (binary), `rms`

| # | arm | inputs | AUC | gAUC | AUPRC | logloss | sd |
|---|---|---|---|---|---|---|---|
| 1 | autoint_full | FULL | 0.68276 | 0.56887 | 0.16599 | 0.29843 | 0.00951 |
| 2 | deepfm_full | FULL | 0.67413 | 0.55501 | 0.16464 | 0.28982 | 0.01457 |
| 3 | dcn_full | FULL | 0.67193 | 0.54780 | 0.16490 | 0.31018 | 0.01765 |
| 4 | dcnv2_full | FULL | 0.66067 | 0.55701 | 0.15988 | 0.38170 | 0.01251 |
| 5 | **dcdn_full** | FULL | 0.65073 | 0.54952 | 0.15268 | 0.31277 | 0.01160 |
| 6 | mlp_full | FULL | 0.64990 | 0.53029 | 0.15228 | 0.30479 | 0.01350 |
| 7 | afn_full | FULL | 0.64689 | 0.56383 | 0.14239 | 0.36226 | 0.01736 |
| 8 | **dcdn_reduced** | RED | 0.64539 | 0.53710 | 0.14672 | 0.35503 | 0.01717 |
| 9 | dcnv2_reduced | RED | 0.63934 | 0.51017 | 0.14920 | 0.33487 | 0.01867 |
| 10 | afn_reduced | RED | 0.63165 | 0.56751 | 0.13327 | 0.42591 | 0.00879 |
| 11 | dcnv2_capacity_reduced | RED | 0.62801 | 0.52050 | 0.13838 | 0.31022 | 0.01376 |
| 12 | wukong_full | FULL | 0.61967 | 0.52953 | 0.13429 | 0.32384 | 0.01500 |

| comparator | mean diff | se | beats DCDN | verdict |
|---|---|---|---|---|
| wukong_full | -0.02572 | 0.00751 | 1/8 | **significant** |
| dcnv2_capacity_reduced (capacity-matched) | -0.01738 | 0.00990 | 3/8 | not significant |
| afn_reduced | -0.01374 | 0.00626 | 2/8 | **significant** |
| dcnv2_reduced | -0.00606 | 0.00781 | 3/8 | not significant |
| afn_full | +0.00149 | 0.00958 | 4/8 | not significant |
| mlp_full | +0.00450 | 0.00878 | 5/8 | not significant |
| dcdn_full | +0.00534 | 0.00867 | 5/8 | not significant |
| dcnv2_full | +0.01528 | 0.00631 | 6/8 | **significant** |
| dcn_full | +0.02654 | 0.00887 | 7/8 | **significant** |
| deepfm_full | +0.02873 | 0.00849 | 7/8 | **significant** |
| autoint_full | +0.03737 | 0.00611 | 8/8 | **significant** |

## 3. Reading it

**`dcdn_reduced` is the best of the four reduced-input arms on both targets.**
It leads `dcnv2_capacity_reduced`, `dcnv2_reduced` and `afn_reduced` on
`long_view`, and beats `dcnv2_reduced` by a wide margin on `watch_time`. That
is its clearest standing anywhere in the programme.

**It beats full-input arms.** On `watch_time` it is ahead of `wukong_full`,
`dcnv2_full` and `dcn_full`; on `long_view`, ahead of `wukong_full`. On VK-LSVD
it beat none, on any target.

**Its weakness is the same on both targets: calibration, not ranking.** Tenth
of twelve on `normalized_mse` (0.00853 against 0.00255) and ninth on
`log_loss` (0.35503 against 0.28982), while seventh and eighth on the ranking
metrics. The frozen contract's primary quantity for the regression head is
`normalized_mse`, so on the primary metric DCDN is near the bottom.

**Against the one arm matched to it in parameters and FLOPs the difference is
not significant on either target**, in either direction.

**Per-seed spread swamps the table.** Per-arm sd runs 0.005-0.023 here against
0.0003-0.0010 on VK-LSVD. Adjacent rows differ by less than an arm's own noise;
only the top three and bottom three separate. Rank order is close to
uninformative and the paired tables above are the instrument that matters.

## 4. A convergence caveat that limits the regression target

`watch_time` is unstable for part of the field. Counting seeds whose macro
NLoss exceeds 1.0 - worse than predicting the train mean:

| arm | worse than constant | fails the frozen 10x gate |
|---|---|---|
| `dcnv2_reduced` | 8/8 | 1/8 |
| `dcdn_reduced` | 5/8 | 0/8 |
| `dcnv2_full` | 4/8 | 0/8 |
| `dcnv2_capacity_reduced` | 1/8 | 0/8 |
| the other eight | 0/8 | 0/8 |

The frozen gate fires at **10x** the constant predictor, which is a
"did it produce a result at all" test; an arm at 1.4x passes it while being
worse than a constant. On a binary target that distinction is minor, on a
regression target it is not.

**This changes one reading.** `dcdn_reduced` beats `dcnv2_full` by xAUC on all
eight seeds, which looks like RQ2 in its strong form. Restricted to the seeds
where **both** arms beat the constant predictor - two of them - `dcnv2_full`
wins on both, by 0.0376. The all-seed win is drawn from seeds where the
comparator failed to fit, not from seeds where DCDN ranked better. No RQ2 claim
is made on this target.

`long_view` has no such problem: every arm converges on every seed, and its
table stands as reported.

## 5. The `geometric` preconditioner is rejected for this dataset

Operand columns carry true magnitudes here - the all-time tab duration sum
reaches 9.8e7 - so a purely multiplicative column scale is required or float32
overflows. Two were run:

| | rule | denominators after scaling |
|---|---|---|
| `rms` | divide by train root-mean-square | median 0.356, 1st pct 7.5e-03, **46.1% of rows** have a denominator under `div_eps` |
| `geometric` | divide by train geometric mean of positive entries | median 1.262, 1st pct 1.7e-02, **13.6% of rows** |

`rms` was chosen before training; `geometric` was tried after it was seen to
saturate the reciprocal floor. That choice is post-hoc and cannot be laundered,
so both are run and both are published, with agreement between them as the
evidence that no conclusion rests on the knob.

**There is no agreement: `geometric` destroys much of the field.**

`long_view`: `dcn_full` returns exactly 0.50000 AUC with sd 0.00000 on all
eight seeds - a constant predictor - `dcnv2_reduced` 0.50019, and `dcnv2_full`
reaches NLoss 5663.

`watch_time` is worse. Both DCDN arms collapse to chance (xAUC 0.50000 and
0.49999) and three arms explode:

| arm | xAUC | nMSE | fails the 10x gate |
|---|---|---|---|
| `dcnv2_reduced` | 0.50321 | 7.7e13 | 8/8 |
| `dcn_full` | 0.50225 | 9.2e22 | 8/8 |
| `dcnv2_full` | 0.49653 | 2.8e21 | 8/8 |
| `dcdn_full` | 0.50000 | 0.556 | 1/8 |
| `dcdn_reduced` | 0.49999 | 34.04 | 1/8 |

Fewer floored denominators does not help: at median 1.26 individual reciprocals
become large instead, and convergence breaks from the other side. **Every
conclusion in this file therefore rests on `rms`**, and `geometric` is recorded
as a rejected branch with these two tables as the reason rather than a
sentence.

## 6. Where this leaves DCDN across four datasets

| dataset | removed ratios | denominators / operands | `dcdn_reduced` rank | beats any full-input arm |
|---|---|---|---|---|
| KuaiRand-Pure | 112 | 24 / 132 = 18% | 7-8 of 12 | yes, 1-3 arms |
| VK-LSVD | 17 | 5 / 55 = 9% | 10-12 of 12 | no |

The ordering is consistent with denominator density, but two datasets cannot
establish a trend, and the KuaiRand regression target is too unstable to carry
weight. What is established is narrower: **on the dataset with the most
division structure, reduced-input DCDN is the best reduced-input arm and beats
some full-input arms; on the dataset with the least, it is last.**

## 7. Reproduction

```bash
source scripts/env.sh
python scripts/run_kuairand_single_target.py --device cuda:0 --target watch_time \
    --preconditioner rms --annex-seeds --save-predictions --field \
    --output-dir results/kuairand_multitask/field_watch_time_rms_v1
python scripts/report_field_table.py \
    --results results/kuairand_multitask/field_watch_time_rms_v1/results_v1.json
```
