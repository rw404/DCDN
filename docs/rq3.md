# RQ3 · Capacity

> **Are the observed differences explained simply by model capacity, or do they
> persist against a capacity-matched DCNv2 control?**

DCDN adds a second projection to every cross block, so the cheapest explanation
of [RQ2](rq2.md) is that it simply buys more interaction capacity. RQ3 removes
that explanation by giving the original architecture the same parameter budget
on the same input.

---

## The control

`dcnv2_capacity_reduced` is the only mechanically matched comparator in the
suite. It is chosen by search over (rank 1–64) × (layers 1–2) × (experts 1, 2, 4)
to land within **±5%** of `dcdn_reduced` on both trainable non-embedding
parameters and FLOPs, **with no metric read** — the selection sees counts only.
Realised agreement is **±0.2%**.

Both models receive exactly the same `F_reduced` inputs. To keep the comparison
uniform across four datasets with three different primary metrics, quality is
reported as **nDCG@10** throughout.

### Trainable non-embedding parameters

| arm | in | Yambda-50M | KuaiRec | KuaiRand-Pure | VK-LSVD ur | vs `dcdn_reduced` |
|---|---|---|---|---|---|---|
| dcn_full | FULL | 34,457 | 31,745 | 49,809 | 43,745 | 0.82× |
| **dcdn_reduced** | RED | 44,021 | 31,725 | 72,681 | 51,129 | **1.00×** |
| **dcnv2_capacity_reduced** | RED* | 44,034 | 31,794 | 72,850 | 50,967 | **1.00×** |
| dcnv2_reduced | RED | 74,357 | 60,685 | 173,289 | 95,769 | 1.89× |
| dcdn_full | FULL | 69,845 | 49,665 | 167,881 | 136,305 | 1.95× |
| dcnv2_full | FULL | 112,469 | 93,345 | 325,833 | 218,385 | 3.61× |
| mlp_full | FULL | 446,993 | 436,977 | 520,113 | 488,817 | 9.86× |
| eulernet_reduced | RED | 556,475 | 411,368 | 763,597 | 614,473 | 12.33× |
| wukong_full | FULL | 676,881 | 597,225 | 922,081 | 855,465 | 16.05× |
| eulernet_full | FULL | 755,435 | 602,038 | 1,227,837 | 1,099,438 | 18.07× |
| deepfm_full | FULL | 1,253,330 | 1,014,722 | 1,988,274 | 1,788,482 | 30.23× |
| afn_reduced | RED | 1,822,199 | 1,528,625 | 2,241,243 | 1,939,539 | 39.66× |
| autoint_full | FULL | 1,684,897 | 1,295,329 | 2,884,961 | 2,558,689 | 40.26× |
| afn_full | FULL | 2,224,727 | 1,914,381 | 3,180,475 | 2,920,701 | 53.83× |

### Interaction block only

| arm | Yambda-50M | KuaiRec | KuaiRand-Pure | VK-LSVD ur |
|---|---|---|---|---|
| dcn_full | 792 | 704 | 1,488 | 1,184 |
| **dcdn_reduced** | 14,964 | 5,100 | 35,112 | 19,800 |
| **dcnv2_capacity_reduced** | 14,977 | 5,169 | 35,281 | 19,638 |
| dcnv2_reduced | 45,300 | 34,060 | 135,720 | 64,440 |
| dcdn_full | 36,180 | 18,624 | 119,560 | 93,744 |
| dcnv2_full | 78,804 | 62,304 | 277,512 | 175,824 |
| mlp_full | 400,400 | 391,600 | 470,000 | 439,600 |
| eulernet_reduced | 504,250 | 360,295 | 709,900 | 561,832 |
| wukong_full | 623,120 | 544,680 | 864,800 | 799,080 |
| eulernet_full | 701,674 | 549,493 | 1,170,556 | 1,043,053 |
| deepfm_full | 1,206,609 | 969,217 | 1,938,033 | 1,739,137 |
| afn_reduced | 1,799,862 | 1,508,592 | 2,215,962 | 1,916,370 |
| autoint_full | 1,073,056 | 836,256 | 1,802,656 | 1,604,256 |
| afn_full | 2,199,318 | 1,891,404 | 3,148,026 | 2,890,044 |

---

## Result

Latency is eager-mode p50 in milliseconds at batch size 8192. Δ is
`dcdn_reduced` minus `dcnv2_capacity_reduced`.

| | nDCG@10 | | | p50 latency (ms) | | |
|---|---|---|---|---|---|---|
| **Dataset** | **DCNv2-cm** | **DCDN-r** | **Δ** | **DCNv2-cm** | **DCDN-r** | **speedup** |
| Yambda | 0.7571 | **0.7590** | **+0.0019** | 0.718 | **0.622** | **1.15×** |
| KuaiRec | 0.4375 | **0.4441** | **+0.0066** | 0.731 | **0.635** | **1.15×** |
| KuaiRand | 0.6719 | **0.6765** | **+0.0046** | 1.045 | **0.713** | **1.47×** |
| VK-LSVD | 0.8521 | **0.8535** | **+0.0014** | 0.838 | **0.644** | **1.30×** |

On the unified nDCG@10 view DCDN is ahead of the capacity-matched control on
**all four datasets**, and cheaper to serve on all four. Deepening the standard
DCNv2 stack to the same parameter count does not recover the same top-ranking
quality at the same realised cost — capacity matching equalises analytic counts,
not realised kernels, and the wider the input geometry the more that gap opens
(1.47× on KuaiRand's 132 operands against 1.15× on Yambda's 86).

---

## Answer to RQ3

> The observed operating point is not explained by parameter count alone. With
> the same reduced input and a comparable parameter budget, DCDN achieves higher
> nDCG@10 and lower p50 latency than the capacity-matched DCNv2 on all four
> datasets. Simply deepening the original DCNv2 cross stack therefore does not
> reproduce the same top-ranking quality–latency trade-off.

---

## Reproduction

```bash
# quality: all fourteen arms including the capacity-matched control
python scripts/run_kuairand_single_target.py --device cuda:0 \
    --target watch_ratio_gt2 --preconditioner none \
    --views $DCDN_DATA_CACHE/runtime/kuairec_watch_time_v1 \
    --annex-seeds --save-predictions --field \
    --output-dir results/kuairec_watch_time/binary_field_v1

# serving cost on the exact configs the quality run trained
python scripts/benchmark_batch_sweep.py --device cuda:0 \
    --geometry kuairec_watch_time --field \
    --output-dir results/kuairec_watch_time/batch_sweep_v1

# join quality, parameters and cost
python scripts/report_overall.py
```

The capacity search itself is an artifact, not a runtime decision: the selected
(rank, layers, experts) triple is frozen and hash-pinned before any training run,
so the control cannot be re-tuned after seeing a result.
