# KuaiRand-Pure: four architectures × three task settings

Validation only; the test split has never been unlocked. This is **not** the
mandatory main-table `kuairand_pure` cell.

| artifact | path |
|---|---|
| consolidated tables | `results/kuairand_multitask/matrix_v1/` |
| multi-head bundle | `results/kuairand_multitask/final_mt_v2/` |
| single-head `long_view` | `results/kuairand_multitask/final_lv_v2/` |
| single-head `watch_time` | `results/kuairand_multitask/final_wt_v2/` |
| metric contract | `configs/metrics_multitask_v1.yaml` |
| metric implementation | `src/dcdn/metrics.py` |

All three task settings use **one** metric implementation, so a difference
between tasks is a difference in the task and not in how it was measured.
Everything below is the frozen three seeds (42, 43, 44); the eight-seed annex is
in the CSVs, tagged, never pooled into these numbers.

---

## 1. The result

**Binary tasks** — loss is the normalized task loss, lower better; AUC, gAUC,
AUPRC higher better.

### 1.1 Multi-head (five heads, Macro-NLoss), scored on the `long_view` head

| arm | loss | AUC | gAUC | AUPRC |
|---|---|---|---|---|
| `dcnv2_full` | 1.6885 | 0.5875 | 0.5281 | 0.1182 |
| `dcnv2_reduced` | 2.8907 | 0.5597 | 0.4799 | 0.1095 |
| **`dcdn_reduced`** | 1.4973 | **0.6171** | **0.5357** | 0.1270 |
| **`dcdn_full`** | **1.2560** | **0.6266** | 0.5257 | **0.1417** |

Separated pairs: `dcdn_full > dcnv2_full`, `dcdn_reduced > dcnv2_reduced`,
`dcdn_full > dcnv2_reduced`. **DCDN wins both same-feature comparisons.**

### 1.2 Single-head `long_view`

| arm | loss | AUC | gAUC | AUPRC |
|---|---|---|---|---|
| **`dcnv2_full`** | 0.5036 | **0.6625** | **0.5560** | **0.1590** |
| `dcnv2_reduced` | **0.5034** | 0.6473 | 0.5116 | 0.1546 |
| `dcdn_reduced` | 0.5339 | 0.6422 | 0.5121 | 0.1465 |
| `dcdn_full` | 0.5081 | 0.6521 | 0.5479 | 0.1495 |

**No pair separates.** The ordering flips relative to 1.1 — DCNv2-FULL is now
top on all three ranking metrics — but not one difference among the four arms
survives the disjoint-range test at three seeds.

**Regression task** — xAUC and xgAUC are pairwise ordering accuracy, higher
better; MAE in seconds, lower better.

### 1.3 Single-head `watch_time`

| arm | loss | xAUC | xgAUC | MAE (s) |
|---|---|---|---|---|
| `dcnv2_full` | 0.8796 | 0.5231 | 0.5036 | 21.30 |
| `dcnv2_reduced` | 3.5124 | 0.5148 | 0.5011 | 27.28 |
| `dcdn_reduced` | 0.9919 | **0.5415** | 0.5079 | 20.25 |
| **`dcdn_full`** | **0.4756** | 0.5314 | **0.5102** | **13.00** |

Separated: `dcdn_reduced > dcnv2_full` and `dcdn_reduced > dcnv2_reduced` on
xAUC. **Reduced-input DCDN out-ranks full-input DCNv2 here** — the only place in
this document where that happens with separation.

## 2. Three things the matrix shows that no single task showed

**2.1 DCDN's `long_view` advantage is a multi-task effect, not an architectural
one.** In the multi-head setting DCDN separates from DCNv2 at both feature sets.
Train the *same* head alone and the advantage disappears entirely: DCNv2-FULL
goes to the top and nothing separates. The natural reading is that DCDN is
better at carrying five heads through one trunk, not better at ranking
`long_view` as such. Two supports for that reading and one caveat:

* it is consistent with the transfer result — every arm does better on
  `watch_time` single-head than multi-head, so the multi-head setting is where
  interference lives;
* the effect direction is the same on gAUC and AUPRC, not just AUC;
* **caveat:** the single-head comparison establishes nothing either. It shows
  DCNv2 ahead, but no pair separates, so "the advantage disappears" is
  "the advantage is no longer demonstrable", not "DCNv2 is better".

**2.2 On the ratio target the pattern the hypothesis predicts does appear.**
`watch_time` is `play_time / duration`, a quotient, and it is the one task where
reduced-input DCDN out-ranks full-input DCNv2 with separated ranges. It is also
where DCDN-FULL takes every column: best loss, best xgAUC, and MAE 13.0 s
against 21.3 s. This is the same direction as the synthetic RQ1 result and it is
the strongest real-data evidence in the study for the reciprocal structure.

**2.3 Global AUC and user-grouped gAUC do not agree.** In the multi-head
setting `dcnv2_capacity_reduced` has the best gAUC (0.5374) while `dcdn_full`
has the best AUC. AFN+ is starker: its multi-head gAUC is **0.4462 — below
chance within a user** — while in the single-head setting the same model has the
best gAUC of any arm (0.5688). The two metrics answer different questions
("orders the whole pool" against "orders within one user's impressions") and on
this dataset they rank the arms differently. Reporting only one would have
produced a confident and misleading story either way.

## 3. Reading limits

* **gAUC covers 42.7% of users.** Only that share of validation users have both
  classes; the rest are excluded from the metric by definition. gAUC here is not
  a whole-population number and the coverage must travel with it.
* **xAUC is barely above chance for every arm** (0.515–0.542). The separations
  are real but the absolute ordering skill on `watch_time` is weak for all four
  architectures.
* **`session_gauc` is null everywhere.** KuaiRand-Pure ships no session
  identifier and the frozen `session_policy` forbids synthesizing one.
* **The multi-head loss column is not comparable across tasks.** It is
  Macro-NLoss over five heads in 1.1 and a single-head loss in 1.2 and 1.3.
  Compare within a row block, not down the loss column.
* **Validation is the randomized policy, train is the standard one.** Task
  constants are fitted on train, so 1.0 is not the constant-predictor level on
  validation. Cross-arm comparison is valid; the absolute level against 1.0 is
  not.
* `dcnv2_reduced` fails the convergence gate on the annex scope; its cells there
  read `non-convergent` rather than a number.

## 4. Proposed RQ reformulation

The evidence no longer fits the original three questions. What it fits:

**RQ1 (unchanged, confirmed).** At equal interaction capacity and given only
base operands, does DCDN reconstruct quotient structure better than DCNv2?
Confirmed on synthetic data — `docs/rq1_verdict.md`.

**RQ2 → split in two.** The current RQ2 conflates "reduced DCDN matches full
DCNv2" with "at what serving cost", and the quality half is answered differently
per task.

* **RQ2a (target-conditional quality).** *On a ratio-structured target*, does
  reduced-input DCDN match full-input DCNv2? On `watch_time`: **yes, and it
  out-ranks it** (xAUC separated). On `long_view`: no difference demonstrable.
  The honest claim is conditional on the target having quotient structure, and
  the study should say which targets qualify before measuring, not after.
* **RQ2b (serving cost).** Unchanged and already answered: fused DCDN is
  runtime-matched at B=1, flat to B=16384, and wins with a HiFormer layer
  downstream.

**RQ3 → reformulate from "beats" to "where".** Asking whether DCDN beats a
capacity-matched DCNv2 has now been answered *no* twice on KuaiRand
(`long_view` and `watch_time`) and *yes* once on synthetic high-ratio data. The
productive question is not whether the reciprocal branch is better in general
but **what property of a target makes it better** — the synthetic evidence says
high denominator-to-numerator ratio and composite quotients; KuaiRand's
`watch_time` is a mild version of this and shows a mild effect. Reformulated:
*for which target structures does an explicit reciprocal branch beat spending
the same capacity conventionally?*

**RQ4 (new, forced by 2.1).** Does DCDN's advantage come from the reciprocal
branch or from tolerating multi-task interference? The matrix shows the
`long_view` advantage present in the multi-head setting and absent in the
single-head one. This is currently an observation with a plausible mechanism and
no test. A clean test exists: hold the architecture fixed and vary the number of
heads (1, 2, 3, 5), measuring the same head throughout. That is cheap on this
dataset and would either establish the interference explanation or kill it.

## 5. Reproduction

```bash
source scripts/env.sh

python scripts/run_kuairand_multitask.py --device cuda:0 \
    --preconditioner rms --annex-seeds \
    --output-dir results/kuairand_multitask/final_mt_v2

python scripts/run_kuairand_single_target.py --device cuda:0 \
    --target long_view --annex-seeds \
    --output-dir results/kuairand_multitask/final_lv_v2

python scripts/run_kuairand_single_target.py --device cuda:0 \
    --target watch_time --annex-seeds --save-predictions \
    --output-dir results/kuairand_multitask/final_wt_v2

python scripts/report_kuairand_matrix.py
```
