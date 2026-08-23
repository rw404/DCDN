# VK-LSVD `ur0.01_ir0.01` — a pre-registered test of the denominator-density prediction

## What was predicted, and when

Across the four datasets completed before this one, DCDN's standing tracked one
number: the share of REDUCED operands that are legitimate positive denominators.

| dataset | denominators / operands | `dcdn_reduced` |
|---|---|---|
| KuaiRand-Pure | 24 / 132 = **18%** | best of the four reduced arms |
| VK-LSVD `up0.01_ir0.01` | 5 / 55 = **9%** | last of twelve, on every target |

That is a testable prediction rather than an observation: a VK feature view with
high denominator density should place DCDN better, and it is free to fail.

`configs/data/vk_lsvd_ur/features_v1.yaml` was written to test it and **frozen at
sha256 `2bf0dfb4732efa2042ca9c53174f8a074994d8173347f4e3d188a1f1824e6f1d` before
the materializer was written and before any arm was trained**. The target ladder
and its thresholds come from train positive rates read while no model existed.

**This design was requested in order to favour DCDN, and that has to be stated.**
Choosing features to make one architecture win is fitting to the outcome. What
makes this version defensible is that the design follows a prediction the earlier
data already made, the contract was registered before anything ran, and the
result below is reported whichever way it came out. It is not a confirmatory
result for RQ2 and is not offered as one.

## The slice and the design

`ur0.01_ir0.01` is a random-user, random-item subsample; the programme's earlier
VK study used `up0.01_ir0.01`, a popularity-weighted user sample eight times
larger. Random users have shorter per-week histories but appear across many
weeks: 0.45% of validation rows have no user history at all, median 94 prior
events. A quarter of validation rows carry a cold item.

| | |
|---|---|
| rows | 801,636 train / 170,111 validation |
| users in validation | 40,791 |
| split | history weeks 0–19, train 20–24, validation 25, **week 26 sealed and never fetched** |
| categorical fields | 5 (user, item, place, platform, agent) |
| numeric operands | **99 REDUCED, 216 FULL** |
| denominators | **36 of 99 = 36.4%** |
| seeds | 8 per arm, learning rate tuned per arm |

Three groups — user, item, user×place — times three windows — the previous week,
the previous four weeks, all earlier weeks. Each of the nine blocks contributes
four denominators: an impression count, its smoothed variant, a timespent sum and
its smoothed variant. Two distinct denominator families per block mean "per
impression" and "per second" are different quotients rather than one rescaled.
The 117 removed FULL operands are those quotients; every numerator and every
denominator stays in REDUCED.

**One amendment was made before training and is recorded in the contract.** The
first materialization smoothed timespent sums by `alpha = 1` second against a
median of several hundred, so a no-history group landed below the reciprocal
floor — 30.9% of rows on one denominator. Replacing it with `beta × mean
timespent per impression = 232.5` — prior pseudo-impressions at the prior mean —
moved the floored-row share from **48.0% on the old VK slice to 7.9% here**. The
diagnostic uses no model score.

## Results

`RED*` marks `dcnv2_capacity_reduced`, matched to `dcdn_reduced` within ±5% on
trainable non-embedding parameters and FLOPs. `quality/ms` divides the margin
over chance by eager-mode p50 latency at B = 8192; it is an ordering aid, not a
decision metric.

### `timespent_gt10`

| # | arm | inputs | AUC | sd | gAUC | AUPRC | NDCG@5 | p50 ms | quality/ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **dcdn_full** | FULL | 0.72642 | 0.00125 | 0.64505 | 0.71078 | 0.8217 | 0.848 | 0.2671 |
| 2 | afn_full | FULL | 0.72427 | 0.00200 | 0.64355 | 0.71325 | 0.8218 | 7.224 | 0.0310 |
| 3 | dcn_full | FULL | 0.72422 | 0.00259 | 0.64029 | 0.71078 | 0.8191 | 0.820 | 0.2735 |
| 4 | autoint_full | FULL | 0.72371 | 0.00229 | 0.64004 | 0.71193 | 0.8198 | 109.234 | 0.0020 |
| 5 | mlp_full | FULL | 0.72203 | 0.00242 | 0.63772 | 0.70726 | 0.8181 | 1.252 | 0.1773 |
| 6 | deepfm_full | FULL | 0.72163 | 0.00390 | 0.63916 | 0.70830 | 0.8195 | 2.956 | 0.0750 |
| 7 | afn_reduced | RED | 0.72026 | 0.00301 | 0.64130 | 0.70803 | 0.8211 | 4.651 | 0.0474 |
| 8 | dcnv2_capacity_reduced | RED* | 0.71910 | 0.00261 | 0.63836 | 0.70452 | 0.8191 | 0.841 | 0.2605 |
| 9 | wukong_full | FULL | 0.71895 | 0.00170 | 0.63462 | 0.70708 | 0.8169 | 4.988 | 0.0439 |
| 10 | dcnv2_full | FULL | 0.71303 | 0.00267 | 0.63522 | 0.69167 | 0.8170 | 0.993 | 0.2145 |
| 11 | **dcdn_reduced** | RED | 0.71120 | 0.00150 | 0.64040 | 0.69814 | 0.8206 | 0.641 | 0.3293 |
| 12 | dcnv2_reduced | RED | 0.69671 | 0.00479 | 0.62585 | 0.67863 | 0.8142 | 0.661 | 0.2975 |

Paired against `dcdn_reduced` on AUC:

| comparator | mean diff | se | seeds | verdict |
|---|---|---|---|---|
| dcnv2_reduced | -0.01449 | 0.00190 | 0/8 | **significant** |
| dcnv2_full | +0.00183 | 0.00105 | 5/8 | not significant |
| wukong_full | +0.00775 | 0.00071 | 8/8 | **significant** |
| dcnv2_capacity_reduced (capacity-matched) | +0.00790 | 0.00090 | 8/8 | **significant** |
| afn_reduced | +0.00906 | 0.00092 | 8/8 | **significant** |
| deepfm_full | +0.01044 | 0.00174 | 8/8 | **significant** |
| mlp_full | +0.01083 | 0.00117 | 8/8 | **significant** |
| autoint_full | +0.01252 | 0.00111 | 8/8 | **significant** |
| dcn_full | +0.01302 | 0.00132 | 8/8 | **significant** |
| afn_full | +0.01307 | 0.00060 | 8/8 | **significant** |
| dcdn_full | +0.01522 | 0.00081 | 8/8 | **significant** |

### `timespent_gt20`

| # | arm | inputs | AUC | sd | gAUC | AUPRC | NDCG@5 | p50 ms | quality/ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | deepfm_full | FULL | 0.77076 | 0.00085 | 0.69470 | 0.64504 | 0.8005 | 2.956 | 0.0916 |
| 2 | autoint_full | FULL | 0.76676 | 0.00233 | 0.69026 | 0.63820 | 0.7980 | 109.234 | 0.0024 |
| 3 | **dcdn_full** | FULL | 0.76645 | 0.00141 | 0.69199 | 0.63464 | 0.7989 | 0.848 | 0.3144 |
| 4 | afn_full | FULL | 0.76543 | 0.00340 | 0.69245 | 0.63894 | 0.7994 | 7.224 | 0.0367 |
| 5 | mlp_full | FULL | 0.76500 | 0.00235 | 0.69016 | 0.63769 | 0.7977 | 1.252 | 0.2116 |
| 6 | dcn_full | FULL | 0.76426 | 0.00154 | 0.68434 | 0.63385 | 0.7942 | 0.820 | 0.3223 |
| 7 | afn_reduced | RED | 0.76143 | 0.00265 | 0.69064 | 0.63265 | 0.7987 | 4.651 | 0.0562 |
| 8 | dcnv2_capacity_reduced | RED* | 0.76086 | 0.00177 | 0.68996 | 0.62900 | 0.7975 | 0.841 | 0.3102 |
| 9 | wukong_full | FULL | 0.75833 | 0.00404 | 0.69090 | 0.63285 | 0.7986 | 4.988 | 0.0518 |
| 10 | **dcdn_reduced** | RED | 0.75035 | 0.00276 | 0.68497 | 0.61417 | 0.7952 | 0.641 | 0.3904 |
| 11 | dcnv2_full | FULL | 0.74787 | 0.00481 | 0.68296 | 0.61044 | 0.7943 | 0.993 | 0.2496 |
| 12 | dcnv2_reduced | RED | 0.73562 | 0.00520 | 0.67324 | 0.59561 | 0.7894 | 0.661 | 0.3563 |

Paired against `dcdn_reduced` on AUC:

| comparator | mean diff | se | seeds | verdict |
|---|---|---|---|---|
| dcnv2_reduced | -0.01473 | 0.00163 | 0/8 | **significant** |
| dcnv2_full | -0.00248 | 0.00211 | 2/8 | not significant |
| wukong_full | +0.00797 | 0.00166 | 8/8 | **significant** |
| dcnv2_capacity_reduced (capacity-matched) | +0.01051 | 0.00100 | 8/8 | **significant** |
| afn_reduced | +0.01108 | 0.00141 | 8/8 | **significant** |
| dcn_full | +0.01390 | 0.00118 | 8/8 | **significant** |
| mlp_full | +0.01465 | 0.00132 | 8/8 | **significant** |
| afn_full | +0.01508 | 0.00157 | 8/8 | **significant** |
| dcdn_full | +0.01610 | 0.00126 | 8/8 | **significant** |
| autoint_full | +0.01641 | 0.00109 | 8/8 | **significant** |
| deepfm_full | +0.02040 | 0.00086 | 8/8 | **significant** |

### `timespent_gt40`

| # | arm | inputs | AUC | sd | gAUC | AUPRC | NDCG@5 | p50 ms | quality/ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | deepfm_full | FULL | 0.80145 | 0.00289 | 0.73770 | 0.53911 | 0.7818 | 2.956 | 0.1020 |
| 2 | **dcdn_full** | FULL | 0.79907 | 0.00223 | 0.73914 | 0.53183 | 0.7821 | 0.848 | 0.3529 |
| 3 | afn_full | FULL | 0.79788 | 0.00214 | 0.73757 | 0.53154 | 0.7816 | 7.224 | 0.0412 |
| 4 | autoint_full | FULL | 0.79784 | 0.00229 | 0.73502 | 0.53474 | 0.7795 | 109.234 | 0.0027 |
| 5 | dcn_full | FULL | 0.79584 | 0.00115 | 0.72965 | 0.52508 | 0.7764 | 0.820 | 0.3608 |
| 6 | mlp_full | FULL | 0.79517 | 0.00463 | 0.73343 | 0.53103 | 0.7791 | 1.252 | 0.2357 |
| 7 | dcnv2_capacity_reduced | RED* | 0.79466 | 0.00162 | 0.73701 | 0.52436 | 0.7800 | 0.841 | 0.3504 |
| 8 | wukong_full | FULL | 0.79379 | 0.00755 | 0.73252 | 0.53343 | 0.7788 | 4.988 | 0.0589 |
| 9 | afn_reduced | RED | 0.79279 | 0.00474 | 0.73545 | 0.52728 | 0.7803 | 4.651 | 0.0629 |
| 10 | dcnv2_full | FULL | 0.78506 | 0.00536 | 0.73261 | 0.51265 | 0.7787 | 0.993 | 0.2870 |
| 11 | **dcdn_reduced** | RED | 0.78297 | 0.00436 | 0.73319 | 0.50797 | 0.7784 | 0.641 | 0.4413 |
| 12 | dcnv2_reduced | RED | 0.76693 | 0.00793 | 0.71751 | 0.48878 | 0.7692 | 0.661 | 0.4037 |

Paired against `dcdn_reduced` on AUC:

| comparator | mean diff | se | seeds | verdict |
|---|---|---|---|---|
| dcnv2_reduced | -0.01605 | 0.00313 | 0/8 | **significant** |
| dcnv2_full | +0.00209 | 0.00139 | 6/8 | not significant |
| afn_reduced | +0.00982 | 0.00206 | 7/8 | **significant** |
| wukong_full | +0.01081 | 0.00271 | 7/8 | **significant** |
| dcnv2_capacity_reduced (capacity-matched) | +0.01168 | 0.00122 | 8/8 | **significant** |
| mlp_full | +0.01220 | 0.00137 | 8/8 | **significant** |
| dcn_full | +0.01287 | 0.00151 | 8/8 | **significant** |
| autoint_full | +0.01486 | 0.00204 | 8/8 | **significant** |
| afn_full | +0.01491 | 0.00172 | 8/8 | **significant** |
| dcdn_full | +0.01609 | 0.00120 | 8/8 | **significant** |
| deepfm_full | +0.01848 | 0.00160 | 8/8 | **significant** |

### `timespent`

| # | arm | inputs | xAUC | sd | user xAUC | nMSE | p50 ms | quality/ms |
|---|---|---|---|---|---|---|---|---|
| 1 | afn_full | FULL | 0.65132 | 0.00251 | 0.59830 | 0.03148 | 7.224 | 0.0209 |
| 2 | autoint_full | FULL | 0.64908 | 0.00255 | 0.59935 | 0.03214 | 109.234 | 0.0014 |
| 3 | afn_reduced | RED | 0.64816 | 0.00219 | 0.59668 | 0.03191 | 4.651 | 0.0319 |
| 4 | mlp_full | FULL | 0.64805 | 0.00153 | 0.59912 | 0.03234 | 1.252 | 0.1182 |
| 5 | dcn_full | FULL | 0.64708 | 0.00211 | 0.59442 | 0.03240 | 0.820 | 0.1794 |
| 6 | deepfm_full | FULL | 0.64367 | 0.00202 | 0.59779 | 0.03281 | 2.956 | 0.0486 |
| 7 | wukong_full | FULL | 0.64322 | 0.00163 | 0.58996 | 0.03230 | 4.988 | 0.0287 |
| 8 | **dcdn_full** | FULL | 0.63890 | 0.00285 | 0.59570 | 0.03405 | 0.848 | 0.1639 |
| 9 | dcnv2_capacity_reduced | RED* | 0.63863 | 0.00304 | 0.59244 | 0.03334 | 0.841 | 0.1648 |
| 10 | dcnv2_reduced | RED | 0.62671 | 0.00637 | 0.57950 | 0.03676 | 0.661 | 0.1916 |
| 11 | dcnv2_full | FULL | 0.62652 | 0.00975 | 0.58300 | 0.04756 | 0.993 | 0.1274 |
| 12 | **dcdn_reduced** | RED | 0.61822 | 0.04793 | 0.57904 | 0.03423 | 0.641 | 0.1843 |

Paired against `dcdn_reduced` on xAUC:

| comparator | mean diff | se | seeds | verdict |
|---|---|---|---|---|
| dcnv2_full | +0.00830 | 0.01809 | 2/8 | not significant |
| dcnv2_reduced | +0.00849 | 0.01649 | 3/8 | not significant |
| dcnv2_capacity_reduced (capacity-matched) | +0.02041 | 0.01613 | 7/8 | not significant |
| dcdn_full | +0.02068 | 0.01720 | 6/8 | not significant |
| wukong_full | +0.02500 | 0.01722 | 8/8 | not significant |
| deepfm_full | +0.02545 | 0.01680 | 8/8 | not significant |
| dcn_full | +0.02886 | 0.01655 | 8/8 | not significant |
| mlp_full | +0.02983 | 0.01719 | 8/8 | not significant |
| afn_reduced | +0.02994 | 0.01731 | 8/8 | not significant |
| autoint_full | +0.03086 | 0.01709 | 8/8 | not significant |
| afn_full | +0.03310 | 0.01719 | 8/8 | not significant |

## Serving cost

| arm | inputs | p50 B=1 | p50 B=64 | p50 B=1024 | p50 B=8192 | tput B=8192 | vs DCDN-red |
|---|---|---|---|---|---|---|---|
| **dcdn_reduced** | RED | 0.594 ms | 0.634 ms | 0.635 ms | 0.641 ms | 12,774,417/s | 1.00x |
| dcnv2_reduced | RED | 0.551 ms | 0.593 ms | 0.582 ms | 0.661 ms | 12,388,776/s | 1.03x |
| dcn_full | FULL | 0.577 ms | 0.614 ms | 0.613 ms | 0.820 ms | 9,991,530/s | 1.28x |
| dcnv2_capacity_reduced | RED* | 0.796 ms | 0.851 ms | 0.843 ms | 0.841 ms | 9,740,481/s | 1.31x |
| **dcdn_full** | FULL | 0.602 ms | 0.634 ms | 0.633 ms | 0.848 ms | 9,665,431/s | 1.32x |
| dcnv2_full | FULL | 0.554 ms | 0.597 ms | 0.590 ms | 0.993 ms | 8,248,647/s | 1.55x |
| mlp_full | FULL | 0.582 ms | 0.641 ms | 0.622 ms | 1.252 ms | 6,542,093/s | 1.95x |
| deepfm_full | FULL | 0.727 ms | 0.783 ms | 0.853 ms | 2.956 ms | 2,771,733/s | 4.61x |
| afn_reduced | RED | 1.018 ms | 1.099 ms | 1.272 ms | 4.651 ms | 1,761,213/s | 7.25x |
| wukong_full | FULL | 1.214 ms | 1.341 ms | 1.306 ms | 4.988 ms | 1,642,468/s | 7.78x |
| afn_full | FULL | 1.018 ms | 1.103 ms | 1.579 ms | 7.224 ms | 1,134,014/s | 11.26x |
| autoint_full | FULL | 1.376 ms | 1.684 ms | 14.318 ms | 109.234 ms | 74,995/s | 170.34x |

Wall-clock over device p50 stays within **1.000–1.025** across all 180 points
against a ceiling of 2.0, so the measurement is clean despite the container's
non-namespaced `/proc/loadavg` forcing the idle gate to be waived.

## Verdict

**The prediction is confirmed for the full-input regime and refuted for the
reduced one.**

| target | `dcdn_full` rank | `dcdn_reduced` rank |
|---|---|---|
| `timespent_gt10` | **1 of 12** | 11 |
| `timespent_gt20` | **3** | 10 |
| `timespent_gt40` | **2** | 11 |
| `timespent` (regression) | 8 | 12 |

On the `up` slice at 9% density, `dcdn_full` placed 5th, 8th, 6th and 11th. At
36% density it is 1st, 3rd, 2nd and 8th, and takes the best gAUC in the field on
`timespent_gt40`. **This is the first dataset in the programme where DCDN leads a
quality table.** The shift is large, one-directional and consistent across three
binary targets.

`dcdn_reduced` moved not at all: 10th to 12th here, 10th to 12th there. Denser
division structure helps DCDN **when the quotients are present in the input**.
Asked to rebuild 117 quotients from 90 counters, it does no better than before —
and this view withholds more than half the operands, against 17 of 72 on the old
slice, so the reduced task here is strictly harder.

### Where DCDN wins outright

**`dcdn_full` is the only arm that is near the top of both tables.** It is
1st/3rd/2nd on quality and 5th of twelve on latency at 0.848 ms, beating
`dcnv2_full` on both axes on all four targets. `deepfm_full`, its closest quality
rival, costs 2.956 ms — 3.5x more. `autoint_full` costs 109 ms, 129x more.

`dcdn_reduced` is **the fastest arm in the field** at 0.641 ms and 12.8M
examples/s, and 1.31x faster than its capacity-matched twin at equal parameters
and FLOPs — the same ratio measured on KuaiRec, Yambda and KuaiRand.

### Where it does not

Against the strong half of the field — DeepFM, AutoInt+, AFN+ — `dcdn_reduced`
is significantly behind on every target. And DCDN's advantage over DCNv2 is
partly an artefact of DCNv2 being weak on this slice: `dcnv2_full` places 10th,
11th, 10th and 11th, so "DCDN beats DCNv2" here means beating an arm that is
itself near the bottom.

### The regression target is the weak point, and one seed explains part of it

`dcdn_reduced` is last on `timespent` with a per-seed sd of 0.04793, twenty times
its neighbours'. Seed 45 collapsed to exactly 0.5000; the other seven run
0.6267–0.6381.

| | xAUC | sd |
|---|---|---|
| all 8 seeds | 0.61822 | 0.04793 |
| 7 live seeds | **0.63511** | 0.00418 |
| `dcnv2_capacity_reduced` | 0.63863 | 0.00304 |

On the seven live seeds it sits at 9th rather than 12th, and the paired gap to
the matched comparator is +0.00436, 6 of 7, still significant. The all-seed
figure is the reported one — the protocol says so — and the collapse is named
here rather than averaged away. The frozen 10x convergence gate does not catch
it: the dead run's NLoss is near 1, not near 10.

## What this does and does not establish

* **Establishes:** denominator density predicts DCDN's standing in the
  full-input regime. Raising it from 9% to 36% on the same dataset family moved
  `dcdn_full` from mid-table to the top on three of four targets.
* **Establishes:** at 36% density DCDN-FULL is simultaneously the best or
  near-best on quality and 5th of twelve on latency, which no arm managed on any
  earlier dataset.
* **Does not establish RQ2.** Reduced-input DCDN does not approach the
  full-input field on any target here, and the gap is wider than on the old
  slice.
* **Does not establish a general claim.** The features were designed to have
  this property, on request. The result is a demonstration that the mechanism is
  real, not evidence that DCDN wins on data someone else collected.

## Reproduction

```bash
source scripts/env.sh
python scripts/data/vk_ur_raw.py
python scripts/data/vk_ur_materialize.py
python scripts/run_vk_ur_single_target.py --device cuda:0 --target timespent_gt10 \
    --annex-seeds --save-predictions --field \
    --output-dir results/vk_lsvd_ur/timespent_gt10_v1
python scripts/benchmark_batch_sweep.py --device cuda:0 --geometry vk_lsvd_ur --field \
    --output-dir results/vk_lsvd_ur/batch_sweep_v1
```
