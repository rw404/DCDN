# KuaiRec — DCDN on 50 inputs against the whole field on 96

Twelve architectures, one binary target, validation only, sealed test never
opened. The question: **can DCDN on the REDUCED feature view stand against every
competitor given the FULL view — all 46 engineered divisions included?**

**Answer: on NDCG@10 it beats all seven, every interval excluding zero. On
global AUC it is level with the four strongest and better than three. On
user-grouped gAUC two competitors beat it.** Three metrics, three different
orderings; all three are reported.

| artifact | path |
|---|---|
| run bundle | `results/kuairec_watch_time/binary_field_v1/` |
| NDCG | `results/kuairec_watch_time/ndcg_v1/` |
| target and protocol | `configs/data/kuairec/single_target_rq2_v1.yaml` |
| serving cost | `docs/kuairec_serving_cost.md` |

Target is `watch_ratio_gt2` — the canonical KuaiRec engagement label, the user
played the track through more than twice, 7.64% positive. Eight seeds, four-point
learning-rate grid, everything else held identical across arms.

**The frozen M1–M7 matrix and the wider field are reported separately.** The
frozen matrix is what the protocol and its multiplicity correction were written
against; quietly growing it from seven arms to twelve would change the
comparison space after the fact. The five additions carry an `F` prefix.

---

## 1. The table

Eight seeds. REDUCED is 50 inputs, FULL is 96.

| arm | inputs | AUC | sd | gAUC | NDCG@10 | AUPRC |
|---|---|---|---|---|---|---|
| `dcdn_full` | FULL | **0.8077** | 0.0035 | **0.7756** | **0.4457** | 0.2830 |
| `autoint_full` **F** | FULL | 0.8072 | 0.0025 | **0.7756** | 0.4393 | 0.2805 |
| `deepfm_full` **F** | FULL | 0.8072 | 0.0042 | 0.7737 | 0.4357 | 0.2800 |
| **`dcdn_reduced`** | **REDUCED** | **0.8072** | 0.0039 | 0.7730 | **0.4441** | 0.2827 |
| `dcnv2_capacity_reduced` | REDUCED | 0.8066 | 0.0037 | 0.7732 | 0.4375 | 0.2805 |
| `wukong_full` **F** | FULL | 0.8062 | 0.0051 | 0.7740 | 0.4343 | 0.2795 |
| `dcnv2_reduced` | REDUCED | 0.8054 | 0.0057 | 0.7712 | 0.4334 | 0.2780 |
| `afn_full` | FULL | 0.8053 | 0.0039 | 0.7745 | 0.4377 | 0.2775 |
| `dcnv2_full` | FULL | 0.8048 | 0.0053 | 0.7709 | 0.4260 | 0.2763 |
| `afn_reduced` | REDUCED | 0.8011 | 0.0058 | 0.7738 | 0.4401 | 0.2710 |
| `mlp_full` **F** | FULL | 0.8007 | 0.0026 | 0.7702 | 0.4251 | 0.2670 |
| `dcn_full` **F** | FULL | 0.7999 | 0.0027 | 0.7661 | 0.4171 | 0.2667 |

**F** = outside the frozen matrix.

## 2. The contrast the study is about

`dcdn_reduced` (50 inputs) against each competitor on FULL (96 inputs). Paired by
seed, percentile bootstrap over eight seeds, 20,000 resamples.

| competitor | NDCG@10 | | AUC | | gAUC | |
|---|---|---|---|---|---|---|
| | difference | verdict | difference | verdict | difference | verdict |
| `dcn_full` | **+0.0271** 8/8 | **DCDN** | **+0.0072** 8/8 | **DCDN** | **+0.0069** 8/8 | **DCDN** |
| `mlp_full` | **+0.0190** 8/8 | **DCDN** | **+0.0065** 7/8 | **DCDN** | **+0.0027** 6/8 | **DCDN** |
| `dcnv2_full` | **+0.0181** 8/8 | **DCDN** | +0.0023 6/8 | DCDN, at zero | +0.0021 6/8 | tie |
| `wukong_full` | **+0.0098** 7/8 | **DCDN** | +0.0010 6/8 | tie | −0.0011 2/8 | tie |
| `deepfm_full` | **+0.0085** 8/8 | **DCDN** | −0.0001 4/8 | tie | −0.0007 3/8 | tie |
| `afn_full` | **+0.0064** 7/8 | **DCDN** | +0.0018 5/8 | tie | −0.0015 1/8 | **AFN+** |
| `autoint_full` | **+0.0048** 6/8 | **DCDN** | −0.0001 3/8 | tie | −0.0026 1/8 | **AutoInt+** |

**On NDCG@10 DCDN wins all seven**, every interval excluding zero, despite
giving up 46 features. On AUC it wins three and ties four. On gAUC it wins two,
ties three, and **loses two** — AutoInt+ and AFN+ each beat it on 7 of 8 seeds.

## 3. Why three metrics disagree, and which to believe

They measure different things and the disagreement is systematic, not noise.

* **AUC** orders the entire validation pool. It is dominated by separating heavy
  users from light ones, which every architecture does about equally well —
  hence the tight 0.80–0.81 band.
* **gAUC** orders within one user's rows, uniformly averaged. AutoInt+ and AFN+
  are better at this than DCDN.
* **NDCG@10** also works within a user but weights the top of the list. DCDN
  wins here by margins two to five times larger than anything visible on AUC.

The combination says something specific: **DCDN is not better at ordering a
user's whole history, but it is better at getting that user's few genuine
replays into the top of the list.** For a ranker that is the operative property;
for a calibration task it is not. That is a claim about which metric matches
which use, not evidence that one metric is right.

## 4. NDCG — why it is computed here and what its list is

The KuaiRand contract declares
`no_ndcg_reason: no_released_complete_candidate_lists_per_request` and NDCG is
not computed there. **KuaiRec's contract says nothing about NDCG and forbids
nothing**, so it is computed for this dataset only.

The underlying objection has not gone away: there is still no released
per-request candidate list. The ranking unit here is **all of a user's
validation rows** — the standard offline convention, and a construction of this
study rather than a property of the data. NDCG@10 answers "did this user's ten
most replayable items reach the top of our ordering across the whole validation
window", not "was one served page ordered well".

**Coverage: 85.1% of users** (5,825 of 6,848). The rest have no positive or a
single row, and NDCG is undefined for them.

**Ties: 24.3% of adjacent within-user pairs carry identical scores** — 272,940
of them, *exactly the same count for every one of the twelve architectures*.
That is structural, not a model property: the contract sets
`deduplication: none_preserve_every_released_big_matrix_row`, so a
(user, item) pair repeats and any model scores the copies identically. It
matters less than it looks: of those 272,940 tied pairs only **1,520 (0.6%)
carry disagreeing labels**, so the arbitrary tie-break can affect just 0.13% of
adjacent pairs. Everything was computed with a fixed tie-break and the incidence
is published beside every value.

No model was retrained to add this metric — it was computed from the per-row
predictions the runner caches under `--save-predictions`, in seconds.

## 5. Two results that cut against the thesis

**The engineered divisions do not help.** `dcnv2_full` is *worse* than
`dcnv2_reduced` on all three metrics, and `afn_full` is worse than `afn_reduced`
on NDCG@10 (0.4377 against 0.4401). The 46 E features were built for exactly
these models and, on this target, do not pay. This inverts KuaiRand, where the
equivalent contrast was worth 1.7× to 4.0×.

**RQ3 still fails.** `dcnv2_capacity_reduced` reaches AUC 0.8066 against DCDN's
0.8072 — a coin flip, as it has been on every dataset and target tried. Whatever
DCDN gains is not gained over the same capacity spent conventionally.

`dcn_full` — the original DCN with all 96 features — is **last on every metric**,
and `mlp_full` second to last. Handing a weak architecture the engineered
divisions does not rescue it.

## 6. Limits

* One dataset, one target, validation only. KuaiRec's other target
  (`watch_time`) cannot separate architectures at all — see
  `docs/kuairec_results.md` §2.
* AUC margins are small relative to seed noise (spread 0.008, standard
  deviations 0.003–0.009). NDCG margins are larger relative to their own
  spread, which is why the NDCG verdicts are cleaner.
* The five field arms were tuned on the same four-point grid as the frozen
  seven. A wider grid could move them; none was given an advantage the others
  lacked, but none was individually optimised either.
* The split does not reproduce the frozen audit's membership hash — an open
  discrepancy recorded in `configs/data/kuairec/single_target_rq2_v1.yaml`.

## 7. Reproduction

```bash
source scripts/env.sh

python scripts/run_kuairand_single_target.py --device cuda:0 \
    --target watch_ratio_gt2 --preconditioner none --annex-seeds \
    --save-predictions --field \
    --views "$DCDN_DATA_CACHE/runtime/kuairec_watch_time_v1" \
    --output-dir results/kuairec_watch_time/binary_field_v1

python scripts/report_ndcg.py \
    --predictions "$DCDN_DATA_CACHE/predictions/binary_field_v1.npz"
```
