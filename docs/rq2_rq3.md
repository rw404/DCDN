# RQ2 / RQ3 — reduced-input quality and serving cost

## Statement

**RQ2.** REDUCED withholds every manually engineered quotient of historical
counters while keeping each numerator and each denominator. Can an architecture
with a division primitive recover from the raw counters what a human engineer
wrote down by hand — that is, does a reduced-input model reach the full-input
field?

**RQ3.** At matched capacity, does DCDN buy more ranking quality per millisecond
of serving cost than the baselines?

| dataset | slice | operands RED / FULL | denominators |
|---|---|---|---|
| Yambda-50M | `listen50` | 86 / 134 | 12 (14%) |
| KuaiRec | `watch_ratio_gt2` | 50 / 96 | 5 (10%) |
| KuaiRand-Pure | `watch_time` | 132 / 244 | 24 (18%) |
| VK-LSVD `ur0.01_ir0.01` | `timespent_gt10` | 99 / 216 | 36 (36%) |

Eight seeds per arm (42–49); the seed controls parameter initialisation and
batch order only. Learning rate tuned per arm on seed 42 over
`(1e-4, 3e-4, 1e-3, 3e-3)`. AdamW, batch 4096, max 3000 steps, validation every
150, early stopping at patience 5, grad-norm clip 1.0. Loss is macro-NLoss:
per-head loss divided by a train-frozen constant-predictor loss, so **1.0 is the
constant predictor**.

## Models

Fourteen architectures. `RED*` marks `dcnv2_capacity_reduced`, the only
mechanically matched comparator — chosen by search over
(rank 1–64) x (layers 1–2) x (experts 1,2,4) to land within ±5% of
`dcdn_reduced` on both trainable non-embedding parameters and FLOPs, with no
metric read. Realised agreement is ±0.2%.

Two arms have division inside their hypothesis class by construction: **AFN+**
through a log-exp branch and **EulerNet** through the polar form, where a
monomial with arbitrary real exponents is linear on log-amplitude and phase.

### Trainable non-embedding parameters

| arm | in | Yambda-50M | KuaiRec | KuaiRand-Pure | VK-LSVD ur | vs dcdn_reduced |
|---|---|---|---|---|---|---|
| dcn_full | FULL | 34,457 | 31,745 | 49,809 | 43,745 | 0.82x |
| **dcdn_reduced** | RED | 44,021 | 31,725 | 72,681 | 51,129 | 1.00x |
| dcnv2_capacity_reduced | RED* | 44,034 | 31,794 | 72,850 | 50,967 | 1.00x |
| dcnv2_reduced | RED | 74,357 | 60,685 | 173,289 | 95,769 | 1.89x |
| **dcdn_full** | FULL | 69,845 | 49,665 | 167,881 | 136,305 | 1.95x |
| dcnv2_full | FULL | 112,469 | 93,345 | 325,833 | 218,385 | 3.61x |
| mlp_full | FULL | 446,993 | 436,977 | 520,113 | 488,817 | 9.86x |
| **eulernet_reduced** | RED | 556,475 | 411,368 | 763,597 | 614,473 | 12.33x |
| wukong_full | FULL | 676,881 | 597,225 | 922,081 | 855,465 | 16.05x |
| **eulernet_full** | FULL | 755,435 | 602,038 | 1,227,837 | 1,099,438 | 18.07x |
| deepfm_full | FULL | 1,253,330 | 1,014,722 | 1,988,274 | 1,788,482 | 30.23x |
| afn_reduced | RED | 1,822,199 | 1,528,625 | 2,241,243 | 1,939,539 | 39.66x |
| autoint_full | FULL | 1,684,897 | 1,295,329 | 2,884,961 | 2,558,689 | 40.26x |
| afn_full | FULL | 2,224,727 | 1,914,381 | 3,180,475 | 2,920,701 | 53.83x |

### Interaction block only

| arm | Yambda-50M | KuaiRec | KuaiRand-Pure | VK-LSVD ur |
|---|---|---|---|---|
| dcn_full | 792 | 704 | 1,488 | 1,184 |
| **dcdn_reduced** | 14,964 | 5,100 | 35,112 | 19,800 |
| dcnv2_capacity_reduced | 14,977 | 5,169 | 35,281 | 19,638 |
| dcnv2_reduced | 45,300 | 34,060 | 135,720 | 64,440 |
| **dcdn_full** | 36,180 | 18,624 | 119,560 | 93,744 |
| dcnv2_full | 78,804 | 62,304 | 277,512 | 175,824 |
| mlp_full | 400,400 | 391,600 | 470,000 | 439,600 |
| **eulernet_reduced** | 504,250 | 360,295 | 709,900 | 561,832 |
| wukong_full | 623,120 | 544,680 | 864,800 | 799,080 |
| **eulernet_full** | 701,674 | 549,493 | 1,170,556 | 1,043,053 |
| deepfm_full | 1,206,609 | 969,217 | 1,938,033 | 1,739,137 |
| afn_reduced | 1,799,862 | 1,508,592 | 2,215,962 | 1,916,370 |
| autoint_full | 1,073,056 | 836,256 | 1,802,656 | 1,604,256 |
| afn_full | 2,199,318 | 1,891,404 | 3,148,026 | 2,890,044 |

### Serving latency, eager-mode p50 at B = 8192 (ms)

| arm | in | Yambda-50M | KuaiRec | KuaiRand-Pure | VK-LSVD ur | vs dcdn_reduced |
|---|---|---|---|---|---|---|
| **dcdn_reduced** | RED | 0.622 | 0.635 | 0.713 | 0.644 | 1.00x |
| dcnv2_reduced | RED | 0.614 | 0.622 | 0.816 | 0.663 | 1.01x |
| dcn_full | FULL | 0.608 | 0.618 | 0.933 | 0.810 | 1.12x |
| **dcdn_full** | FULL | 0.628 | 0.636 | 0.968 | 0.847 | 1.16x |
| dcnv2_capacity_reduced | RED* | 0.718 | 0.731 | 1.045 | 0.838 | 1.23x |
| dcnv2_full | FULL | 0.705 | 0.658 | 1.185 | 0.977 | 1.33x |
| mlp_full | FULL | 1.043 | 1.017 | 1.368 | 1.234 | 1.80x |
| deepfm_full | FULL | 2.105 | 1.816 | 3.253 | 2.931 | 3.97x |
| wukong_full | FULL | 4.127 | 3.618 | 5.100 | 4.944 | 6.89x |
| afn_reduced | RED | 4.331 | 3.752 | 5.512 | 4.648 | 7.09x |
| **eulernet_reduced** | RED | 5.196 | 4.539 | 6.294 | 5.407 | 8.38x |
| afn_full | FULL | 5.376 | 4.603 | 7.932 | 7.199 | 9.89x |
| **eulernet_full** | FULL | 6.168 | 5.436 | 8.704 | 8.005 | 11.06x |
| autoint_full | FULL | 69.461 | 33.448 | 126.219 | 109.295 | 140.70x |

### Yambda-50M / listen50

| # | arm | in | AUC | gAUC | AUPRC | logloss | sd | NDCG@5 | @10 | @50 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **eulernet_full** | FULL | 0.82190 | 0.61405 | 0.89101 | 0.48738 | 0.00034 | 0.7490 | 0.7614 | 0.7897 | 6.168 |
| 2 | wukong_full | FULL | 0.82172 | 0.61348 | 0.89121 | 0.48727 | 0.00026 | 0.7501 | 0.7625 | 0.7903 | 4.127 |
| 3 | afn_full | FULL | 0.82158 | 0.61402 | 0.89074 | 0.48782 | 0.00042 | 0.7491 | 0.7617 | 0.7897 | 5.376 |
| 4 | autoint_full | FULL | 0.82143 | 0.61546 | 0.89093 | 0.48787 | 0.00042 | 0.7499 | 0.7623 | 0.7902 | 69.461 |
| 5 | deepfm_full | FULL | 0.82139 | 0.61289 | 0.89077 | 0.48814 | 0.00052 | 0.7484 | 0.7602 | 0.7887 | 2.105 |
| 6 | mlp_full | FULL | 0.82110 | 0.61330 | 0.89058 | 0.48814 | 0.00038 | 0.7508 | 0.7628 | 0.7900 | 1.043 |
| 7 | **dcdn_full** | FULL | 0.82065 | 0.61267 | 0.89019 | 0.48871 | 0.00047 | 0.7480 | 0.7603 | 0.7884 | 0.628 |
| 8 | dcn_full | FULL | 0.82010 | 0.61115 | 0.88971 | 0.48992 | 0.00049 | 0.7464 | 0.7588 | 0.7874 | 0.608 |
| 9 | dcnv2_full | FULL | 0.81989 | 0.60847 | 0.88958 | 0.48989 | 0.00108 | 0.7427 | 0.7560 | 0.7861 | 0.705 |
| 10 | afn_reduced | RED | 0.81661 | 0.61124 | 0.88676 | 0.49345 | 0.00098 | 0.7526 | 0.7613 | 0.7876 | 4.331 |
| 11 | **eulernet_reduced** | RED | 0.81578 | 0.60979 | 0.88672 | 0.49396 | 0.00056 | 0.7496 | 0.7582 | 0.7851 | 5.196 |
| 12 | dcnv2_capacity_reduced | RED* | 0.81555 | 0.60891 | 0.88712 | 0.49462 | 0.00063 | 0.7480 | 0.7571 | 0.7855 | 0.718 |
| 13 | **dcdn_reduced** | RED | 0.81492 | 0.60873 | 0.88633 | 0.49540 | 0.00065 | 0.7503 | 0.7590 | 0.7860 | 0.622 |
| 14 | dcnv2_reduced | RED | 0.81433 | 0.60713 | 0.88523 | 0.49679 | 0.00095 | 0.7451 | 0.7548 | 0.7842 | 0.614 |

### KuaiRec / watch_ratio_gt2

| # | arm | in | AUC | gAUC | AUPRC | logloss | sd | NDCG@5 | @10 | @50 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **dcdn_full** | FULL | 0.80774 | 0.77560 | 0.28299 | 0.21750 | 0.00349 | 0.4364 | 0.4457 | 0.5490 | 0.636 |
| 2 | **eulernet_full** | FULL | 0.80743 | 0.77565 | 0.28158 | 0.21624 | 0.00259 | 0.4316 | 0.4408 | 0.5452 | 5.436 |
| 3 | **eulernet_reduced** | RED | 0.80730 | 0.77540 | 0.28219 | 0.21623 | 0.00360 | 0.4341 | 0.4449 | 0.5477 | 4.539 |
| 4 | autoint_full | FULL | 0.80725 | 0.77560 | 0.28054 | 0.21727 | 0.00250 | 0.4270 | 0.4393 | 0.5442 | 33.448 |
| 5 | deepfm_full | FULL | 0.80724 | 0.77367 | 0.27996 | 0.21721 | 0.00422 | 0.4265 | 0.4357 | 0.5411 | 1.816 |
| 6 | **dcdn_reduced** | RED | 0.80715 | 0.77295 | 0.28271 | 0.21751 | 0.00388 | 0.4342 | 0.4441 | 0.5469 | 0.635 |
| 7 | dcnv2_capacity_reduced | RED* | 0.80655 | 0.77323 | 0.28054 | 0.21774 | 0.00368 | 0.4265 | 0.4375 | 0.5421 | 0.731 |
| 8 | wukong_full | FULL | 0.80619 | 0.77403 | 0.27951 | 0.21731 | 0.00506 | 0.4245 | 0.4343 | 0.5407 | 3.618 |
| 9 | dcnv2_reduced | RED | 0.80536 | 0.77124 | 0.27797 | 0.21900 | 0.00567 | 0.4227 | 0.4334 | 0.5390 | 0.622 |
| 10 | afn_full | FULL | 0.80533 | 0.77450 | 0.27746 | 0.21935 | 0.00393 | 0.4264 | 0.4377 | 0.5425 | 4.603 |
| 11 | dcnv2_full | FULL | 0.80482 | 0.77091 | 0.27630 | 0.21899 | 0.00525 | 0.4166 | 0.4260 | 0.5339 | 0.658 |
| 12 | afn_reduced | RED | 0.80114 | 0.77379 | 0.27096 | 0.21997 | 0.00582 | 0.4296 | 0.4401 | 0.5438 | 3.752 |
| 13 | mlp_full | FULL | 0.80067 | 0.77022 | 0.26704 | 0.22081 | 0.00259 | 0.4167 | 0.4251 | 0.5325 | 1.017 |
| 14 | dcn_full | FULL | 0.79992 | 0.76609 | 0.26668 | 0.22123 | 0.00269 | 0.4088 | 0.4171 | 0.5259 | 0.618 |

### KuaiRand-Pure / watch_time

| # | arm | in | xAUC | user xAUC | nMSE | MAE | sd | NDCG@5 | @10 | @50 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | autoint_full | FULL | 0.55748 | 0.50946 | 0.00296 | 14.106 | 0.00448 | 0.6083 | 0.6797 | 0.7409 | 126.219 |
| 2 | afn_full | FULL | 0.55552 | 0.50999 | 0.00255 | 11.990 | 0.01173 | 0.6057 | 0.6776 | 0.7389 | 7.932 |
| 3 | mlp_full | FULL | 0.54830 | 0.50726 | 0.00286 | 12.835 | 0.00830 | 0.6061 | 0.6779 | 0.7394 | 1.368 |
| 4 | afn_reduced | RED | 0.54291 | 0.50281 | 0.00355 | 16.554 | 0.01866 | 0.5971 | 0.6696 | 0.7335 | 5.512 |
| 5 | deepfm_full | FULL | 0.54274 | 0.50015 | 0.00286 | 11.587 | 0.00857 | 0.5948 | 0.6679 | 0.7320 | 3.253 |
| 6 | dcnv2_capacity_reduced | RED* | 0.54128 | 0.50179 | 0.00716 | 27.229 | 0.01128 | 0.5994 | 0.6719 | 0.7353 | 1.045 |
| 7 | **dcdn_reduced** | RED | 0.54075 | 0.50693 | 0.00853 | 22.640 | 0.00521 | 0.6046 | 0.6765 | 0.7388 | 0.713 |
| 8 | wukong_full | FULL | 0.53745 | 0.50540 | 0.00314 | 13.186 | 0.02259 | 0.6028 | 0.6748 | 0.7372 | 5.100 |
| 9 | **eulernet_full** | FULL | 0.53552 | 0.51228 | 0.00337 | 14.545 | 0.01858 | 0.6142 | 0.6843 | 0.7450 | 8.704 |
| 10 | **dcdn_full** | FULL | 0.52935 | 0.50765 | 0.00416 | 14.667 | 0.01600 | 0.6058 | 0.6771 | 0.7390 | 0.968 |
| 11 | **eulernet_reduced** | RED | 0.52440 | 0.50786 | 0.00315 | 15.108 | 0.01999 | 0.6097 | 0.6799 | 0.7418 | 6.294 |
| 12 | dcnv2_full | FULL | 0.51854 | 0.50140 | 0.00859 | 20.440 | 0.01419 | 0.6002 | 0.6718 | 0.7353 | 1.185 |
| 13 | dcnv2_reduced | RED | 0.51470 | 0.49972 | 0.04236 | 35.184 | 0.00937 | 0.5927 | 0.6654 | 0.7306 | 0.816 |
| 14 | dcn_full | FULL | 0.51457 | 0.50236 | 0.00501 | 19.656 | 0.01653 | 0.5949 | 0.6673 | 0.7319 | 0.933 |

### VK-LSVD ur / timespent_gt10

| # | arm | in | AUC | gAUC | AUPRC | logloss | sd | NDCG@5 | @10 | @50 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **dcdn_full** | FULL | 0.72642 | 0.64505 | 0.71078 | 0.61739 | 0.00125 | 0.8217 | 0.8545 | 0.8680 | 0.847 |
| 2 | afn_full | FULL | 0.72427 | 0.64355 | 0.71325 | 0.61545 | 0.00200 | 0.8218 | 0.8544 | 0.8680 | 7.199 |
| 3 | dcn_full | FULL | 0.72422 | 0.64029 | 0.71078 | 0.61376 | 0.00259 | 0.8191 | 0.8522 | 0.8658 | 0.810 |
| 4 | autoint_full | FULL | 0.72371 | 0.64004 | 0.71193 | 0.61994 | 0.00229 | 0.8198 | 0.8528 | 0.8664 | 109.295 |
| 5 | **eulernet_full** | FULL | 0.72343 | 0.63804 | 0.71240 | 0.61402 | 0.00459 | 0.8183 | 0.8515 | 0.8651 | 8.005 |
| 6 | mlp_full | FULL | 0.72203 | 0.63772 | 0.70726 | 0.61720 | 0.00242 | 0.8181 | 0.8512 | 0.8651 | 1.234 |
| 7 | deepfm_full | FULL | 0.72163 | 0.63916 | 0.70830 | 0.61682 | 0.00390 | 0.8195 | 0.8526 | 0.8662 | 2.931 |
| 8 | afn_reduced | RED | 0.72026 | 0.64130 | 0.70803 | 0.62197 | 0.00301 | 0.8211 | 0.8537 | 0.8675 | 4.648 |
| 9 | dcnv2_capacity_reduced | RED* | 0.71910 | 0.63836 | 0.70452 | 0.61962 | 0.00261 | 0.8191 | 0.8521 | 0.8661 | 0.838 |
| 10 | wukong_full | FULL | 0.71895 | 0.63462 | 0.70708 | 0.61896 | 0.00170 | 0.8169 | 0.8504 | 0.8641 | 4.944 |
| 11 | **eulernet_reduced** | RED | 0.71601 | 0.63311 | 0.70410 | 0.61984 | 0.00626 | 0.8165 | 0.8498 | 0.8637 | 5.407 |
| 12 | dcnv2_full | FULL | 0.71303 | 0.63522 | 0.69167 | 0.64192 | 0.00267 | 0.8170 | 0.8505 | 0.8644 | 0.977 |
| 13 | **dcdn_reduced** | RED | 0.71120 | 0.64040 | 0.69814 | 0.63296 | 0.00150 | 0.8206 | 0.8535 | 0.8672 | 0.644 |
| 14 | dcnv2_reduced | RED | 0.69671 | 0.62585 | 0.67863 | 0.66775 | 0.00479 | 0.8142 | 0.8481 | 0.8624 | 0.663 |

## Verdict

### RQ2 — fails on three of four datasets

| dataset | best reduced arm | reaches the best full-input arm? |
|---|---|---|
| Yambda `listen50` | `afn_reduced`, 10th of 14 | no — −0.0053, 0/8 seeds |
| KuaiRec `watch_ratio_gt2` | **`eulernet_reduced`, 3rd of 14** | **yes** — −0.00044, not significant |
| KuaiRand `watch_time` | `afn_reduced`, 4th of 14 | no — −0.0146, 2/8 |
| VK-LSVD ur `timespent_gt10` | `afn_reduced`, 8th of 14 | no — −0.0062, 0/8 |

`dcdn_reduced` ranks 13th, 6th, 7th and 13th. It is the best reduced arm on
**none** of the four, and on the one dataset where a reduced arm does reach the
full-input field, the arm that does it is EulerNet.

### RQ3 — holds, and the margin is earned on latency

DCDN takes the best quality-per-millisecond on all four slices; a paired
seed-by-seed test confirms it against all six comparators every time — 24 of 24
pairwise comparisons.

Against the capacity-matched comparator, with 95% percentile-bootstrap intervals
over 20,000 resamples of the per-seed differences:

| dataset | mean difference | 95% CI | seeds | verdict |
|---|---|---|---|---|
| Yambda | −0.00064 | [−0.00092, −0.00035] | 0/8 | **significantly worse** |
| KuaiRec | +0.00060 | [−0.00225, +0.00340] | 4/8 | not significant |
| KuaiRand | −0.00053 | [−0.00565, +0.00763] | 1/8 | not significant |
| VK-LSVD ur | −0.00790 | [−0.00962, −0.00633] | 0/8 | **significantly worse** |

**At equal parameters and FLOPs DCDN never wins on quality** — worse on two,
indistinguishable on two, ahead on none. What it delivers is comparable quality
**1.29x–1.46x faster**, stable across four geometries from 50/96 to 132/244
operands, because capacity matching equalises analytic counts and not realised
kernels.

### Division is not exclusive to DCDN

| arm | parameters vs DCDN-RED | latency vs DCDN-RED | median quality rank |
|---|---|---|---|
| `afn_full` | 53.8x | 9.9x | 2.5 |
| `eulernet_full` | 18.1x | 11.1x | 3.5 |
| **`dcdn_full`** | **1.9x** | **1.2x** | **4.0** |

Both arms with division in their hypothesis class outrank DCDN on quality and
cost an order of magnitude more. **No arm the size of DCDN uses division at
all** — that is the defensible form of the claim, not "the only model that can
divide".

### Evidence quality

Yambda `listen50` and KuaiRec `watch_ratio_gt2` are the two uncontested slices —
eight seeds, no convergence failures, no design chosen after seeing results — and
there the quality-per-millisecond margin is **+1% and +2%**. KuaiRand
`watch_time` has five arms with seeds worse than a constant predictor and should
carry little weight. VK-LSVD ur uses a feature view built to have 36% denominator
density.

## Reproduction

```bash
# one dataset, one target, all fourteen arms, eight seeds
python scripts/run_kuairand_single_target.py --device cuda:0 \
    --target watch_ratio_gt2 --preconditioner none \
    --views $DCDN_DATA_CACHE/runtime/kuairec_watch_time_v1 \
    --annex-seeds --save-predictions --field \
    --output-dir results/kuairec_watch_time/binary_field_v1

# serving cost on the matching geometry, all fourteen arms
python scripts/benchmark_batch_sweep.py --device cuda:0 \
    --geometry kuairec_watch_time --field \
    --output-dir results/kuairec_watch_time/batch_sweep_v1

# join quality, parameters and cost into the tables above
python scripts/report_overall.py
```
