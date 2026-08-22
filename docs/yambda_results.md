# Yambda-50M — single-head listening study

Twelve architectures, two targets, validation only, sealed test never opened.

**Headline, stated plainly: on Yambda DCDN does not win.** Reduced-input DCDN
sits eleventh of twelve on both targets. Every arm on the FULL feature view
beats every arm on REDUCED, so on this dataset the engineered divisions are
worth far more than the choice of architecture. RQ2 fails here and RQ3 fails
again.

This is the opposite of KuaiRec, where the same divisions were worthless and
reduced DCDN led on NDCG. Both results are published; neither is quoted alone.

| artifact | path |
|---|---|
| source manifest | `data/manifests/yambda_source_v1.yaml` |
| feature contract | `configs/data/yambda/features_v1.yaml` |
| `play_fraction` bundle | `results/yambda_listen/pf_field_v1/` |
| `listen50` bundle | `results/yambda_listen/l50_field_v1/` |
| NDCG | `results/yambda_listen/ndcg_v1/` |
| serving cost | `results/yambda_listen/batch_sweep_{eager,graph}_v1/` |

---

## 1. What had to be built

Yambda ships six columns — `uid`, `timestamp`, `item_id`, `is_organic`,
`played_ratio_pct`, `track_length_seconds` — and nothing derived. Unlike
KuaiRand and KuaiRec it had no feature contract in this repository, so one was
authored and **frozen before a single row was materialized**, which is what
stops the feature set being tuned against a result.

Four key groups (user, item, artist, album) × three windows (1d, 7d, all) ×
five statistics, plus two current-row non-outcome inputs and twelve shifted-count
denominators: **P=74, V=12, R=48 — REDUCED 86, FULL 134.** R is the
engineered-division block: mean played ratio, listen50 rate, organic rate and
mean track length per group per window.

Splits are temporal with a deep burn-in — days 0–230 update rolling history and
carry no label, 231–285 train (10.79M rows), 286–295 validate (2.14M), 296–300
sealed and dropped (1.06M).

**Leakage.** A row sees only what happened strictly before its own timestamp,
per key group; rows sharing a timestamp are invisible to each other.
`played_ratio_pct` is the current row's outcome, is the sole label source, and is
never an input to its own row — it does enter other rows' history. Seven
invariants are asserted in `tests`-style checks against the rolling function.

Both targets come from `configs/targets_multitask_v2.yaml` verbatim:
`play_fraction = clip(played_ratio_pct,0,100)/100` — the normalized listening
time, and the declared primary — and `listen50 = played_ratio_pct >= 50`, at
63.4% positive.

## 2. `play_fraction` — the listening-time target

Eight seeds. xAUC is pairwise ordering accuracy; NLoss is relative to the
train-fitted constant predictor at 1.0.

| arm | inputs | xAUC | sd | xgAUC | NLoss |
|---|---|---|---|---|---|
| `mlp_full` **F** | FULL | **0.7481** | 0.0005 | **0.5802** | **0.6606** |
| `deepfm_full` **F** | FULL | 0.7478 | 0.0010 | 0.5799 | 0.6618 |
| `afn_full` | FULL | 0.7478 | 0.0007 | 0.5799 | 0.6629 |
| `autoint_full` **F** | FULL | 0.7473 | 0.0015 | 0.5786 | 0.6635 |
| `wukong_full` **F** | FULL | 0.7473 | 0.0007 | 0.5783 | 0.6644 |
| `dcdn_full` | FULL | 0.7471 | 0.0008 | 0.5789 | 0.6648 |
| `dcnv2_full` | FULL | 0.7467 | 0.0013 | 0.5775 | 0.6675 |
| `dcn_full` **F** | FULL | 0.7467 | 0.0006 | 0.5792 | 0.6650 |
| `dcnv2_capacity_reduced` | REDUCED | 0.7438 | 0.0006 | 0.5761 | 0.6730 |
| `afn_reduced` | REDUCED | 0.7430 | 0.0012 | 0.5776 | 0.6724 |
| **`dcdn_reduced`** | REDUCED | **0.7415** | 0.0026 | 0.5745 | 0.6786 |
| `dcnv2_reduced` | REDUCED | 0.7407 | 0.0013 | 0.5743 | 0.6862 |

**The eight FULL arms occupy the top eight places and the four REDUCED arms the
bottom four, without a single crossing.** The feature view separates the field
completely; architecture only reorders within each block.

## 3. `listen50` — the binary form

| arm | inputs | AUC | sd | gAUC | NDCG@10 | AUPRC |
|---|---|---|---|---|---|---|
| `wukong_full` **F** | FULL | **0.8217** | 0.0003 | 0.6135 | 0.7625 | 0.8912 |
| `afn_full` | FULL | 0.8216 | 0.0004 | 0.6140 | 0.7617 | 0.8907 |
| `autoint_full` **F** | FULL | 0.8214 | 0.0004 | **0.6155** | 0.7623 | 0.8909 |
| `deepfm_full` **F** | FULL | 0.8214 | 0.0005 | 0.6129 | 0.7602 | 0.8908 |
| `mlp_full` **F** | FULL | 0.8211 | 0.0004 | 0.6133 | **0.7628** | 0.8906 |
| `dcdn_full` | FULL | 0.8206 | 0.0005 | 0.6127 | 0.7603 | 0.8902 |
| `dcn_full` **F** | FULL | 0.8201 | 0.0005 | 0.6111 | 0.7588 | 0.8897 |
| `dcnv2_full` | FULL | 0.8199 | 0.0011 | 0.6085 | 0.7560 | 0.8896 |
| `afn_reduced` | REDUCED | 0.8166 | 0.0010 | 0.6112 | 0.7613 | 0.8868 |
| `dcnv2_capacity_reduced` | REDUCED | 0.8156 | 0.0006 | 0.6089 | 0.7571 | 0.8871 |
| **`dcdn_reduced`** | REDUCED | **0.8149** | 0.0006 | 0.6087 | 0.7590 | 0.8863 |
| `dcnv2_reduced` | REDUCED | 0.8143 | 0.0010 | 0.6071 | 0.7548 | 0.8852 |

Same structure: FULL sweeps the top eight. NDCG@10 covers **96.5% of users**
(8,092 of 8,383) with only 1.4% tied adjacent pairs — a far healthier ranking
metric than KuaiRec's, where 24% of pairs were tied by duplicate rows.

## 4. The contrasts, paired over eight seeds

| contrast | `play_fraction` xAUC | | `listen50` AUC | |
|---|---|---|---|---|
| | difference | verdict | difference | verdict |
| C1 `dcdn_reduced − dcnv2_reduced` | +0.00080 5/8 | no difference | **+0.00059** 6/8 | **DCDN** |
| C2 `dcdn_reduced − afn_reduced` | −0.00147 3/8 | no difference | **−0.00169** 1/8 | **AFN+** |
| C3 `dcdn_reduced − capacity` | **−0.00228** 1/8 | **capacity** | **−0.00064** 0/8 | **capacity** |
| C4 `dcnv2_full − dcnv2_reduced` | **+0.00602** 8/8 | **the divisions** | **+0.00556** 8/8 | **the divisions** |
| C5 `dcdn_reduced − dcnv2_full` | **−0.00522** 0/8 | **DCNv2-FULL** | **−0.00497** 0/8 | **DCNv2-FULL** |
| C6 `dcdn_full − dcnv2_full` | +0.00034 4/8 | no difference | **+0.00076** 7/8 | **DCDN** |

**C5 — the RQ2 question — fails decisively on both targets.** Reduced DCDN loses
to full DCNv2 by 0.005, winning **zero of eight seeds** on either target, with
intervals nowhere near zero. On KuaiRec the same contrast favoured DCDN; here it
is unambiguously the other way.

**C4 explains why.** The engineered divisions are worth +0.006 to DCNv2, 8/8 on
both targets. On KuaiRec C4 was *negative* — the divisions were useless there.
Yambda's history is built from listening behaviour where the rates (mean played
ratio, listen50 rate) are exactly the predictive quantity, so removing them
costs real information that no architecture recovers.

**C3 — RQ3 — fails a fourth time**, and here with intervals excluding zero
against DCDN on both targets. The capacity-matched DCNv2 is simply better.

**What does hold for DCDN** is the narrow same-features claim: C1 on `listen50`
(+0.00059, 6/8) and C6 on `listen50` (+0.00076, 7/8). Both are in the fourth
decimal.

## 5. Serving cost

Measured at the study's exact widths — 4 categorical fields, 86 reduced against
134 full operands, 12 denominators — with the field configs imported from the
runner so cost and quality describe the same models. `wall_over_device` is
1.00–1.02 across all 180 eager points.

| arm | inputs | eager /s | ratio | graph /s | ratio | VRAM |
|---|---|---:|---:|---:|---:|---:|
| **`dcdn_reduced`** | RED | 17,697,593 | **1.000** | 22,433,862 | **1.000** | **339 MiB** |
| `dcnv2_capacity_reduced` | RED | 13,532,222 | 0.765 | 22,068,881 | 0.984 | 339 MiB |
| `dcnv2_reduced` | RED | 16,685,955 | 0.943 | 18,363,643 | 0.819 | 340 MiB |
| `dcn_full` | FULL | 15,700,636 | 0.887 | 16,720,099 | 0.745 | 439 MiB |
| `dcdn_full` | FULL | 15,741,660 | 0.889 | 16,048,160 | 0.715 | 439 MiB |
| `dcnv2_full` | FULL | 13,817,734 | 0.781 | 14,356,690 | 0.640 | 439 MiB |
| `mlp_full` | FULL | 8,690,791 | 0.491 | 9,031,147 | 0.403 | 441 MiB |
| `deepfm_full` | FULL | 4,170,032 | 0.236 | 4,239,711 | 0.189 | 444 MiB |
| `wukong_full` | FULL | 2,111,265 | 0.119 | 2,140,521 | 0.095 | 502 MiB |
| `afn_reduced` | RED | 1,957,191 | 0.111 | 1,977,694 | 0.088 | 1,150 MiB |
| `afn_full` | FULL | 1,573,253 | 0.089 | 1,581,437 | 0.070 | 1,347 MiB |
| **`autoint_full`** | FULL | **118,848** | **0.007** | 118,907 | **0.005** | **8,946 MiB** |

`dcdn_reduced` is the fastest arm measured, in both modes. **AutoInt+ is 149×
slower and needs 26× the memory** — sharper than KuaiRec's 70× and 34×, because
Yambda has 138 fields against 101 and attention is quadratic in field count.

VRAM is eager-only; under CUDA graphs each captured graph keeps a private pool
alive and the peak is not per-arm.

## 6. Reading the two results together

| | KuaiRec `watch_ratio_gt2` | Yambda `listen50` |
|---|---|---|
| C4, worth of the divisions | **negative** — they hurt | **+0.0056**, 8/8 |
| C5, reduced DCDN vs full DCNv2 | **DCDN wins** on NDCG, 7/7 | **DCNv2 wins**, 0/8 |
| C3, vs capacity-matched | coin flip | **capacity wins**, 0/8 |
| `dcdn_reduced` rank of 12 | 4th | 11th |

The two datasets give opposite answers to RQ2, and C4 says why: **the value of
the manual divisions decides it.** Where they carry little information
(KuaiRec), dropping them costs nothing and DCDN's cheaper input wins. Where they
carry the predictive signal (Yambda), dropping them costs 0.006 and nothing
recovers it.

That is a more useful finding than either dataset alone would have given, and it
reframes RQ2: the question is not "can DCDN replace the divisions" but "how much
do the divisions carry on this data" — which is measurable before training any
DCDN at all, by running C4.

## 7. Limits

* Validation only; the sealed test days were counted and dropped.
* Margins between neighbouring arms are in the fourth decimal against seed
  standard deviations of 0.0003–0.0026. Only the FULL-versus-REDUCED gap and the
  C4/C5 contrasts are large relative to noise.
* The feature contract is this study's own design. It was frozen before
  materialization, but a different window or statistic set could move the
  FULL/REDUCED balance, and nothing here establishes that this design is optimal.
* Like/dislike streams were excluded by the frozen `event_stream_join_policy`;
  a study that joined them would be measuring something else.

## 8. Reproduction

```bash
source scripts/env.sh
python scripts/data/yambda_raw.py
python scripts/data/yambda_materialize.py

for target in play_fraction listen50; do
  python scripts/run_kuairand_single_target.py --device cuda:0 \
      --target "$target" --preconditioner none --annex-seeds \
      --save-predictions --field \
      --views "$DCDN_DATA_CACHE/runtime/yambda_listen_v1" \
      --output-dir "results/yambda_listen/${target}_field_v1"
done

python scripts/report_ndcg.py \
    --predictions "$DCDN_DATA_CACHE/predictions/l50_field_v1.npz" \
    --target-index 0 --output-dir results/yambda_listen/ndcg_v1

python scripts/benchmark_batch_sweep.py --device cuda:1 \
    --geometry yambda_listen --allow-busy-host --field \
    --output-dir results/yambda_listen/batch_sweep_eager_v1
```
