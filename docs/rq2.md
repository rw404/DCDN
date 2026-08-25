# RQ2 · Ranking quality and inference efficiency

> **How does DCDN compare with other feature-interaction models in ranking
> quality and measured inference latency?**

Ranking quality is only half of the question. A model that improves an offline
metric is unattractive in production if it reduces throughput or inflates tail
latency, so RQ2 is answered by two measurements that are deliberately kept
apart: quality under a fixed information budget, and model-only latency under a
fixed serving geometry.

---

## Tasks and feature views

| dataset | target | main evaluation | dim `F_reduced` | dim `F_full` | denominators | loss |
|---|---|---|---|---|---|---|
| Yambda-50M | listened ≥ 50% of track duration | AUC, nDCG@10 | 86 | 134 | 12 (14%) | NE |
| KuaiRec | `watch_ratio > 2` | AUC, nDCG@10 | 50 | 96 | 5 (10%) | NE |
| KuaiRand-Pure | watch time | XAUC, nDCG@10 | 132 | 244 | 24 (18%) | NMSE |
| VK-LSVD `ur0.01_ir0.01` | time spent > 10s | AUC, nDCG@10 | 99 | 216 | 36 (36%) | NE |

`F_reduced = B` contains all base features, including every numerator and
denominator operand, but omits the engineered ratios. `F_full = B ∪ R` adds
those ratios back. The two views differ **only** in whether the normalised
relations are materialised explicitly.

KuaiRand ships no session identifiers, so sessions are reconstructed with the
common 30-minute inactivity heuristic — interactions ordered chronologically per
user, a new session opening whenever the gap exceeds 30 minutes — and nDCG@10 is
computed over those reconstructed sessions.

Loss is macro-NLoss: per-head loss divided by a train-frozen constant-predictor
loss, so **1.0 is the constant predictor**. For binary tasks that is normalised
entropy (NE); for KuaiRand's continuous watch time it is normalised MSE (NMSE).

**Protocol.** Eight evaluation seeds per arm (42–49); the seed controls
parameter initialisation and batch order only, and every reported value is a
mean over those eight runs. Learning rate selected once per arm on a dedicated
tuning seed over `{1e-4, 3e-4, 1e-3, 3e-3}` by validation loss, then fixed.
AdamW, batch 4096, at most 3000 steps, validation every 150 steps, early
stopping at patience 5, grad-norm clip 1.0. Categorical fields use learnable
16-dimensional embeddings. Latency is eager-mode p50 at batch size 8192 on
NVIDIA A100-SXM4-80GB.

---

## Quality under a fixed information budget

Every model in this table receives `F_full`, so no architecture is handed
explicit ratio features another is denied and the comparison isolates the
interaction operator. Higher is better throughout.

| | Yambda | | KuaiRec | | KuaiRand | | VK-LSVD | |
|---|---|---|---|---|---|---|---|---|
| **Model** | **AUC** | **nDCG@10** | **AUC** | **nDCG@10** | **XAUC** | **nDCG@10** | **AUC** | **nDCG@10** |
| MLP | 0.82110 | 0.7628 | 0.80067 | 0.4251 | 0.54830 | 0.6779 | 0.72203 | 0.8512 |
| DeepFM | 0.82139 | 0.7602 | 0.80724 | 0.4357 | 0.54274 | 0.6679 | 0.72163 | 0.8526 |
| DCN | 0.82010 | 0.7588 | 0.79992 | 0.4171 | 0.51457 | 0.6673 | 0.72422 | 0.8522 |
| AutoInt+ | 0.82143 | 0.7623 | 0.80725 | 0.4393 | **0.55748** | 0.6797 | 0.72371 | 0.8528 |
| DCNv2 | 0.81989 | 0.7560 | 0.80482 | 0.4260 | 0.51854 | 0.6718 | 0.71303 | 0.8505 |
| AFN+ | 0.82158 | 0.7617 | 0.80533 | 0.4377 | 0.55552 | 0.6776 | 0.72427 | 0.8544 |
| EulerNet | **0.82190** | 0.7614 | 0.80743 | 0.4408 | 0.53552 | **0.6843** | 0.72343 | 0.8515 |
| Wukong | 0.82172 | **0.7625** | 0.80619 | 0.4343 | 0.53745 | 0.6748 | 0.71895 | 0.8504 |
| **DCDN** | 0.82065 | 0.7603 | **0.80774** | **0.4457** | 0.52935 | 0.6771 | **0.72642** | **0.8545** |

**The comparison that matters most is with DCNv2**, because DCDN changes only
the cross operator while both models see the same `F_full` inputs. DCDN improves
the primary metric on **every** dataset — AUC 0.81989 → 0.82065 on Yambda,
0.80482 → 0.80774 on KuaiRec, 0.71303 → 0.72642 on VK-LSVD, and XAUC 0.51854 →
0.52935 on KuaiRand. The same direction holds for nDCG@10 on all four,
KuaiRand included (0.6718 → 0.6771).

**The broader field is more informative than a win count.** DCDN is the best
model on both reported metrics for KuaiRec and VK-LSVD. On Yambda the leaders
are tightly clustered and EulerNet takes the highest AUC; DCDN stays above
DCNv2 but is not top-ranked. KuaiRand is the clearest counterexample to
universal superiority — AutoInt+ has the highest XAUC and EulerNet the highest
reconstructed-session nDCG@10. Against the two generalised-order baselines
closest in spirit, DCDN wins on KuaiRec and VK-LSVD while AFN+ and EulerNet win
on Yambda and KuaiRand.

---

## Inference efficiency

Measured on `F_reduced`, eager execution, p50 at batch 8192, identical harness
for every architecture. This benchmark isolates model execution and is
deliberately separate from the production measurements in
[`RQ4`](../README.md#rq4--production).

| model | Yambda | KuaiRec | KuaiRand | VK-LSVD | vs DCDN |
|---|---|---|---|---|---|
| **DCDN** | **0.622** | **0.635** | **0.713** | **0.644** | 1.00× |
| DCNv2 | 0.614 | 0.622 | 0.816 | 0.663 | 1.01× |
| DCNv2 capacity-matched | 0.718 | 0.731 | 1.045 | 0.838 | 1.23× |
| AFN+ | 4.331 | 3.752 | 5.512 | 4.648 | 7.09× |
| EulerNet | 5.196 | 4.539 | 6.294 | 5.407 | 8.38× |

DCDN stays in the same sub-millisecond regime as DCNv2 across all four feature
geometries: within **2.1%** of DCNv2 on Yambda and KuaiRec, and *faster* on
KuaiRand and VK-LSVD, where the wider input makes DCNv2's larger cross stack
cost more than the reciprocal branch. The two architectures that can also
express negative or non-integer interaction orders sit an order of magnitude
away — AFN+ at **5.9×–7.7×** the DCDN latency and EulerNet at **7.1×–8.8×**.

That is the trade-off DCDN targets. It does not attempt to match the generality
of AFN+ or EulerNet; it specialises the interaction family to reciprocal
relations while keeping a runtime profile next to DCNv2.

---

## Answer to RQ2

> DCDN provides a favorable quality–efficiency trade-off. It consistently
> improves ranking quality over its direct DCNv2 parent while remaining in
> approximately the same latency regime, and it is substantially cheaper to
> serve than the more general AFN+ and EulerNet alternatives. The result is
> therefore not universal accuracy dominance, but competitive ranking quality at
> a much lighter inference cost.

---

## Full metric set

The paper reports two metrics per dataset. This repository keeps every metric
the runs produced, so the headline numbers can be checked against the ones that
did not make the table. Paper metrics come first; `sd` is the standard deviation
of the primary metric across the eight seeds. `RED` arms see `F_reduced`, `RED*`
is the capacity-matched control of [`RQ3`](rq3.md).

### Yambda-50M · `listen50`

| # | arm | in | AUC | nDCG@10 | gAUC | AUPRC | logloss | sd | nDCG@5 | nDCG@50 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **eulernet_full** | FULL | 0.82190 | 0.7614 | 0.61405 | 0.89101 | 0.48738 | 0.00034 | 0.7490 | 0.7897 | 6.168 |
| 2 | wukong_full | FULL | 0.82172 | 0.7625 | 0.61348 | 0.89121 | 0.48727 | 0.00026 | 0.7501 | 0.7903 | 4.127 |
| 3 | afn_full | FULL | 0.82158 | 0.7617 | 0.61402 | 0.89074 | 0.48782 | 0.00042 | 0.7491 | 0.7897 | 5.376 |
| 4 | autoint_full | FULL | 0.82143 | 0.7623 | 0.61546 | 0.89093 | 0.48787 | 0.00042 | 0.7499 | 0.7902 | 69.461 |
| 5 | deepfm_full | FULL | 0.82139 | 0.7602 | 0.61289 | 0.89077 | 0.48814 | 0.00052 | 0.7484 | 0.7887 | 2.105 |
| 6 | mlp_full | FULL | 0.82110 | 0.7628 | 0.61330 | 0.89058 | 0.48814 | 0.00038 | 0.7508 | 0.7900 | 1.043 |
| 7 | **dcdn_full** | FULL | 0.82065 | 0.7603 | 0.61267 | 0.89019 | 0.48871 | 0.00047 | 0.7480 | 0.7884 | 0.628 |
| 8 | dcn_full | FULL | 0.82010 | 0.7588 | 0.61115 | 0.88971 | 0.48992 | 0.00049 | 0.7464 | 0.7874 | 0.608 |
| 9 | dcnv2_full | FULL | 0.81989 | 0.7560 | 0.60847 | 0.88958 | 0.48989 | 0.00108 | 0.7427 | 0.7861 | 0.705 |
| 10 | afn_reduced | RED | 0.81661 | 0.7613 | 0.61124 | 0.88676 | 0.49345 | 0.00098 | 0.7526 | 0.7876 | 4.331 |
| 11 | **eulernet_reduced** | RED | 0.81578 | 0.7582 | 0.60979 | 0.88672 | 0.49396 | 0.00056 | 0.7496 | 0.7851 | 5.196 |
| 12 | dcnv2_capacity_reduced | RED* | 0.81555 | 0.7571 | 0.60891 | 0.88712 | 0.49462 | 0.00063 | 0.7480 | 0.7855 | 0.718 |
| 13 | **dcdn_reduced** | RED | 0.81492 | 0.7590 | 0.60873 | 0.88633 | 0.49540 | 0.00065 | 0.7503 | 0.7860 | 0.622 |
| 14 | dcnv2_reduced | RED | 0.81433 | 0.7548 | 0.60713 | 0.88523 | 0.49679 | 0.00095 | 0.7451 | 0.7842 | 0.614 |

### KuaiRec · `watch_ratio_gt2`

| # | arm | in | AUC | nDCG@10 | gAUC | AUPRC | logloss | sd | nDCG@5 | nDCG@50 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **dcdn_full** | FULL | 0.80774 | 0.4457 | 0.77560 | 0.28299 | 0.21750 | 0.00349 | 0.4364 | 0.5490 | 0.636 |
| 2 | **eulernet_full** | FULL | 0.80743 | 0.4408 | 0.77565 | 0.28158 | 0.21624 | 0.00259 | 0.4316 | 0.5452 | 5.436 |
| 3 | **eulernet_reduced** | RED | 0.80730 | 0.4449 | 0.77540 | 0.28219 | 0.21623 | 0.00360 | 0.4341 | 0.5477 | 4.539 |
| 4 | autoint_full | FULL | 0.80725 | 0.4393 | 0.77560 | 0.28054 | 0.21727 | 0.00250 | 0.4270 | 0.5442 | 33.448 |
| 5 | deepfm_full | FULL | 0.80724 | 0.4357 | 0.77367 | 0.27996 | 0.21721 | 0.00422 | 0.4265 | 0.5411 | 1.816 |
| 6 | **dcdn_reduced** | RED | 0.80715 | 0.4441 | 0.77295 | 0.28271 | 0.21751 | 0.00388 | 0.4342 | 0.5469 | 0.635 |
| 7 | dcnv2_capacity_reduced | RED* | 0.80655 | 0.4375 | 0.77323 | 0.28054 | 0.21774 | 0.00368 | 0.4265 | 0.5421 | 0.731 |
| 8 | wukong_full | FULL | 0.80619 | 0.4343 | 0.77403 | 0.27951 | 0.21731 | 0.00506 | 0.4245 | 0.5407 | 3.618 |
| 9 | dcnv2_reduced | RED | 0.80536 | 0.4334 | 0.77124 | 0.27797 | 0.21900 | 0.00567 | 0.4227 | 0.5390 | 0.622 |
| 10 | afn_full | FULL | 0.80533 | 0.4377 | 0.77450 | 0.27746 | 0.21935 | 0.00393 | 0.4264 | 0.5425 | 4.603 |
| 11 | dcnv2_full | FULL | 0.80482 | 0.4260 | 0.77091 | 0.27630 | 0.21899 | 0.00525 | 0.4166 | 0.5339 | 0.658 |
| 12 | afn_reduced | RED | 0.80114 | 0.4401 | 0.77379 | 0.27096 | 0.21997 | 0.00582 | 0.4296 | 0.5438 | 3.752 |
| 13 | mlp_full | FULL | 0.80067 | 0.4251 | 0.77022 | 0.26704 | 0.22081 | 0.00259 | 0.4167 | 0.5325 | 1.017 |
| 14 | dcn_full | FULL | 0.79992 | 0.4171 | 0.76609 | 0.26668 | 0.22123 | 0.00269 | 0.4088 | 0.5259 | 0.618 |

### KuaiRand-Pure · `watch_time`

nDCG@10 is computed on sessions reconstructed with the 30-minute inactivity
threshold.

| # | arm | in | XAUC | nDCG@10 | user xAUC | nMSE | MAE | sd | nDCG@5 | nDCG@50 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | autoint_full | FULL | 0.55748 | 0.6797 | 0.50946 | 0.00296 | 14.106 | 0.00448 | 0.6083 | 0.7409 | 126.219 |
| 2 | afn_full | FULL | 0.55552 | 0.6776 | 0.50999 | 0.00255 | 11.990 | 0.01173 | 0.6057 | 0.7389 | 7.932 |
| 3 | mlp_full | FULL | 0.54830 | 0.6779 | 0.50726 | 0.00286 | 12.835 | 0.00830 | 0.6061 | 0.7394 | 1.368 |
| 4 | afn_reduced | RED | 0.54291 | 0.6696 | 0.50281 | 0.00355 | 16.554 | 0.01866 | 0.5971 | 0.7335 | 5.512 |
| 5 | deepfm_full | FULL | 0.54274 | 0.6679 | 0.50015 | 0.00286 | 11.587 | 0.00857 | 0.5948 | 0.7320 | 3.253 |
| 6 | dcnv2_capacity_reduced | RED* | 0.54128 | 0.6719 | 0.50179 | 0.00716 | 27.229 | 0.01128 | 0.5994 | 0.7353 | 1.045 |
| 7 | **dcdn_reduced** | RED | 0.54075 | 0.6765 | 0.50693 | 0.00853 | 22.640 | 0.00521 | 0.6046 | 0.7388 | 0.713 |
| 8 | wukong_full | FULL | 0.53745 | 0.6748 | 0.50540 | 0.00314 | 13.186 | 0.02259 | 0.6028 | 0.7372 | 5.100 |
| 9 | **eulernet_full** | FULL | 0.53552 | 0.6843 | 0.51228 | 0.00337 | 14.545 | 0.01858 | 0.6142 | 0.7450 | 8.704 |
| 10 | **dcdn_full** | FULL | 0.52935 | 0.6771 | 0.50765 | 0.00416 | 14.667 | 0.01600 | 0.6058 | 0.7390 | 0.968 |
| 11 | **eulernet_reduced** | RED | 0.52440 | 0.6799 | 0.50786 | 0.00315 | 15.108 | 0.01999 | 0.6097 | 0.7418 | 6.294 |
| 12 | dcnv2_full | FULL | 0.51854 | 0.6718 | 0.50140 | 0.00859 | 20.440 | 0.01419 | 0.6002 | 0.7353 | 1.185 |
| 13 | dcnv2_reduced | RED | 0.51470 | 0.6654 | 0.49972 | 0.04236 | 35.184 | 0.00937 | 0.5927 | 0.7306 | 0.816 |
| 14 | dcn_full | FULL | 0.51457 | 0.6673 | 0.50236 | 0.00501 | 19.656 | 0.01653 | 0.5949 | 0.7319 | 0.933 |

### VK-LSVD `ur0.01_ir0.01` · `timespent_gt10`

| # | arm | in | AUC | nDCG@10 | gAUC | AUPRC | logloss | sd | nDCG@5 | nDCG@50 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **dcdn_full** | FULL | 0.72642 | 0.8545 | 0.64505 | 0.71078 | 0.61739 | 0.00125 | 0.8217 | 0.8680 | 0.847 |
| 2 | afn_full | FULL | 0.72427 | 0.8544 | 0.64355 | 0.71325 | 0.61545 | 0.00200 | 0.8218 | 0.8680 | 7.199 |
| 3 | dcn_full | FULL | 0.72422 | 0.8522 | 0.64029 | 0.71078 | 0.61376 | 0.00259 | 0.8191 | 0.8658 | 0.810 |
| 4 | autoint_full | FULL | 0.72371 | 0.8528 | 0.64004 | 0.71193 | 0.61994 | 0.00229 | 0.8198 | 0.8664 | 109.295 |
| 5 | **eulernet_full** | FULL | 0.72343 | 0.8515 | 0.63804 | 0.71240 | 0.61402 | 0.00459 | 0.8183 | 0.8651 | 8.005 |
| 6 | mlp_full | FULL | 0.72203 | 0.8512 | 0.63772 | 0.70726 | 0.61720 | 0.00242 | 0.8181 | 0.8651 | 1.234 |
| 7 | deepfm_full | FULL | 0.72163 | 0.8526 | 0.63916 | 0.70830 | 0.61682 | 0.00390 | 0.8195 | 0.8662 | 2.931 |
| 8 | afn_reduced | RED | 0.72026 | 0.8537 | 0.64130 | 0.70803 | 0.62197 | 0.00301 | 0.8211 | 0.8675 | 4.648 |
| 9 | dcnv2_capacity_reduced | RED* | 0.71910 | 0.8521 | 0.63836 | 0.70452 | 0.61962 | 0.00261 | 0.8191 | 0.8661 | 0.838 |
| 10 | wukong_full | FULL | 0.71895 | 0.8504 | 0.63462 | 0.70708 | 0.61896 | 0.00170 | 0.8169 | 0.8641 | 4.944 |
| 11 | **eulernet_reduced** | RED | 0.71601 | 0.8498 | 0.63311 | 0.70410 | 0.61984 | 0.00626 | 0.8165 | 0.8637 | 5.407 |
| 12 | dcnv2_full | FULL | 0.71303 | 0.8505 | 0.63522 | 0.69167 | 0.64192 | 0.00267 | 0.8170 | 0.8644 | 0.977 |
| 13 | **dcdn_reduced** | RED | 0.71120 | 0.8535 | 0.64040 | 0.69814 | 0.63296 | 0.00150 | 0.8206 | 0.8672 | 0.644 |
| 14 | dcnv2_reduced | RED | 0.69671 | 0.8481 | 0.62585 | 0.67863 | 0.66775 | 0.00479 | 0.8142 | 0.8624 | 0.663 |

---

## Beyond the paper: what the reduced view costs on public data

The paper uses `F_reduced` for the latency benchmark and for RQ3, and reports
the production feature removal in RQ4. It does not ask the public-data version
of that question directly, so this repository records it.

**Does any reduced-input arm reach the best full-input arm on its dataset?**

| dataset | best reduced arm | reaches the best full-input arm? |
|---|---|---|
| Yambda `listen50` | `afn_reduced`, 10th of 14 | no — −0.0053, 0/8 seeds |
| KuaiRec `watch_ratio_gt2` | **`eulernet_reduced`, 3rd of 14** | **yes** — −0.00044, not significant |
| KuaiRand `watch_time` | `afn_reduced`, 4th of 14 | no — −0.0146, 2/8 seeds |
| VK-LSVD ur `timespent_gt10` | `afn_reduced`, 8th of 14 | no — −0.0062, 0/8 seeds |

On public data, removing the materialised ratios costs primary-metric quality on
three of four datasets, and on the one dataset where a reduced arm does close
the gap, the arm that closes it is EulerNet rather than DCDN — `dcdn_reduced`
ranks 13th, 6th, 7th and 13th of fourteen on the primary metric.

This does not contradict the production result in RQ4, and the difference is
worth naming precisely. The public slices remove between 10% and 36% of the
input width, and their denominators are a handful of engagement counters. The
production feature set removes **40.1%** of the inputs, and its ratio family is
both larger and more redundant with the operands that remain. RQ4 also measures
XAUC within sessions on a multi-head ranker, not single-head AUC on a flat
split. The honest reading is that ratio removal is cheap when the removed family
is large and reconstructible, and not free in general.

Two further qualifications on the public evidence: KuaiRand `watch_time` has
five arms with seeds worse than a constant predictor and should carry little
weight, and the VK-LSVD view was built to have 36% denominator density, which is
the most favourable geometry in the set rather than a neutral one.

---

## Division is not exclusive to DCDN

| arm | parameters vs `dcdn_reduced` | latency vs `dcdn_reduced` | median quality rank |
|---|---|---|---|
| `afn_full` | 53.8× | 9.9× | 2.5 |
| `eulernet_full` | 18.1× | 11.1× | 3.5 |
| **`dcdn_full`** | **1.9×** | **1.2×** | **4.0** |

Both arms with division in their hypothesis class outrank DCDN on median quality
across the full metric set and cost an order of magnitude more to serve. **No
arm the size of DCDN uses division at all** — that is the defensible form of the
claim, not "the only model that can divide", and it is what makes the RQ2
latency table the load-bearing half of the answer.

---

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

> Materialised views are built by the data pipeline on the full research branch.
> Every contract is frozen and hash-pinned before acquisition; week 26 of
> VK-LSVD is sealed and never fetched.
