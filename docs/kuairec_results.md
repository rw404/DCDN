# KuaiRec big matrix — single-head results

Validation only; the sealed test split was counted and dropped, never
materialized. Two task settings on the same materialized views:

* **`watch_time`** — regression, `clip(play_seconds, 0, 131.684) / 131.684`
* **`watch_ratio_gt2`** — binary, the user played the video through more than
  twice; 7.64% positive

**The short version.** The regression target is unlearnable here — all seven
architectures collapse onto the same function within 0.3% and none beats the
validation-optimal constant by more than 2.7%. The binary target *is* learnable
— AUC 0.81, gAUC 0.77 — and there DCDN leads, though by margins that are small
relative to seed noise.

| artifact | path |
|---|---|
| runtime views | `$DCDN_DATA_CACHE/runtime/kuairec_watch_time_v1/` |
| `watch_time` bundle | `results/kuairec_watch_time/single_v1/` |
| `watch_ratio_gt2` bundle | `results/kuairec_watch_time/binary_v1/` |
| study protocol | `configs/data/kuairec/single_target_rq2_v1.yaml` |
| serving cost | `docs/kuairec_serving_cost.md` |

---

## 1. Materialization, and an open discrepancy

Views are built by `scripts/data/kuairec_materialize.py` from the frozen
`configs/data/kuairec/features_v1.yaml`. C=5, P=44, V=6, E=46, so
**REDUCED = P+V = 50** and **FULL = +E = 96**, with 5 denominators. E is
KuaiRec's engineered-division block — the analogue of KuaiRand's R — holding
`e_play_per_show`, `e_complete_play_per_play` and 44 more ratios of historical
counters.

Everything the frozen blind audit reports independently agrees:

| check | result |
|---|---|
| filter stages 12,530,806 → 12,495,027 → 12,255,608 → 12,255,608 | **match** |
| split counts 9,804,486 / 1,225,560 / 1,225,562 | **match** |
| train mapping cardinalities 7,176 users, 9,049 videos | **match** |
| train-only label constant c = 131.684 s | **match** |
| cold drop 84,405 against the audited 84,406 | **off by one** |

The vectorized feature algebra was checked against the contract's per-row
oracle `build_feature_values` and agrees to **0.000e+00 across all 96 fields**.

**The three `split_membership_sha256` values do not reproduce.** Several
encodings of the documented payload were tried and none matched. The cause is
not recoverable from the frozen artifacts, and guessing further would be fitting
to a target rather than reproducing a method. Recorded as an open discrepancy in
`configs/data/kuairec/single_target_rq2_v1.yaml`; every result below is
conditional on a split that conforms to the contract but is not bit-identical to
the audited one.

Three mistakes were made and corrected before this run: sorting by `date`
instead of the contract's `[timestamp, user_id, video_id, source_row_index0]`,
applying the filter predicates out of the declared order, and mapping cold users
and items to OOV instead of dropping them. Each would have produced plausible
but non-conforming data.

**No input preconditioner is applied.** KuaiRand needed one because its contract
forbids fitted scales; KuaiRec's contract *mandates* dividing every P and E
column by `max(train q99, 1.0)` at materialization, so rescaling again would
normalize twice.

## 2. `watch_time` — the dataset cannot separate architectures

Frozen three seeds. Lower MSE is better.

| arm | MSE | sd | xAUC | xgAUC | MAE (s) |
|---|---|---|---|---|---|
| `afn_reduced` | 0.006068 | 0.000002 | 0.5463 | 0.5137 | 5.10 |
| `dcnv2_full` | 0.006075 | 0.000004 | **0.5497** | **0.5142** | 5.11 |
| `dcnv2_reduced` | 0.006079 | 0.000008 | 0.5453 | 0.5124 | 5.12 |
| `dcdn_reduced` | 0.006080 | 0.000007 | 0.5420 | 0.5113 | 5.11 |
| `dcdn_full` | 0.006083 | 0.000001 | 0.5435 | 0.5128 | **5.07** |
| `dcnv2_capacity_reduced` | 0.006084 | 0.000006 | 0.5433 | 0.5110 | 5.08 |
| `afn_full` | 0.006088 | 0.000011 | 0.5427 | 0.5134 | 5.16 |

**Spread between best and worst is 0.33%.** Seed standard deviations are in the
sixth decimal. Seven structurally different architectures converge on
effectively the same function.

**A correction worth stating.** The frozen `macro_nloss` reads 1.08 here, which
looks like "worse than a constant predictor". It is not: that metric divides by
the **train** variance (0.005625) while evaluating on validation, whose variance
is **0.006235**. Against the validation-optimal constant every arm is better, at
0.9737–0.9757 — they explain 2.4–2.6% of the variance. Real, and tiny.

The arm ordering is not stable across seed scopes: `dcdn_full` is fifth on three
seeds and second on eight, `dcnv2_full` moves from second to sixth. At
thousandths of a percent those are noise permutations, not a ranking.

**Why.** KuaiRec's features carry almost nothing about the user. All 44 P and 46
E fields are daily aggregates *per video* — popularity, completion rates, likes
— plus the video's duration. The only personalizing signal is the `user_id`
embedding. Predicting an individual's watch time from a video's popularity may
simply have no solution on these inputs.

## 3. `watch_ratio_gt2` — the task is learnable and DCDN leads

The canonical KuaiRec engagement label, recomputed from `play_duration` and
`video_duration`; the released `watch_ratio` column is still never read, which
is what the contract's `watch_ratio_read: false` commits to.

Frozen three seeds:

| arm | log-loss | AUC | sd | gAUC | AUPRC |
|---|---|---|---|---|---|
| **`dcdn_full`** | **0.2176** | **0.8077** | 0.0043 | **0.7746** | **0.2846** |
| `dcnv2_capacity_reduced` | 0.2176 | 0.8074 | 0.0059 | 0.7734 | 0.2822 |
| `afn_full` | 0.2196 | 0.8072 | 0.0030 | **0.7746** | 0.2828 |
| `dcdn_reduced` | 0.2178 | 0.8070 | 0.0056 | 0.7735 | 0.2817 |
| `dcnv2_full` | 0.2190 | 0.8044 | 0.0085 | 0.7713 | 0.2744 |
| `afn_reduced` | 0.2187 | 0.8034 | 0.0036 | 0.7739 | 0.2755 |
| `dcnv2_reduced` | 0.2205 | 0.8023 | 0.0094 | 0.7706 | 0.2732 |

gAUC coverage is **85.0%** of users (5,822 of 6,848) — far healthier than
KuaiRand's 42.7%.

Paired across eight seeds, bootstrap interval, 20,000 resamples:

| contrast | AUC difference | 95% CI | wins | gAUC difference | 95% CI | wins |
|---|---|---|---|---|---|---|
| C1 `dcdn_reduced − dcnv2_reduced` | +0.00180 | [−0.0009, +0.0047] | 4/8 | +0.00171 | [−0.0009, +0.0048] | 4/8 |
| C2 `dcdn_reduced − afn_reduced` | **+0.00601** | **[+0.0035, +0.0092]** | **8/8** | −0.00083 | [−0.0025, +0.0011] | 4/8 |
| C3 `dcdn_reduced − capacity` | +0.00060 | [−0.0023, +0.0034] | 4/8 | −0.00028 | [−0.0018, +0.0011] | 4/8 |
| C4 `dcnv2_full − dcnv2_reduced` | −0.00054 | [−0.0033, +0.0021] | 3/8 | −0.00033 | [−0.0018, +0.0013] | 4/8 |
| C5 `dcdn_reduced − dcnv2_full` | **+0.00233** | **[+0.0000, +0.0047]** | 6/8 | +0.00205 | [−0.0002, +0.0041] | 6/8 |
| C6 `dcdn_full − dcnv2_full` | **+0.00292** | **[+0.0001, +0.0058]** | 5/8 | **+0.00469** | **[+0.0029, +0.0065]** | **8/8** |
| C7 `afn_full − dcnv2_full` | +0.00051 | [−0.0032, +0.0053] | 2/8 | **+0.00359** | **[+0.0020, +0.0053]** | **8/8** |

### 3.1 What holds

* **C2 — DCDN beats AFN+ on identical reduced inputs**, +0.0060 AUC, interval
  excluding zero, winning every one of eight seeds. The single cleanest result
  on this dataset. Note it does **not** hold on gAUC, where the two are level.
* **C6 — DCDN beats DCNv2 on identical full inputs** on both metrics, and on
  gAUC it wins 8/8 with a comfortably positive interval.
* **C5 — reduced DCDN beats full DCNv2 on AUC**, but the interval's lower bound
  is +0.00001. That is a hair from zero; it should be read as "the direction is
  positive and the evidence is weak", not as an established margin.

### 3.2 What does not

* **RQ3 fails again.** C3 against the capacity-matched DCNv2 is +0.0006 on AUC
  with 4 wins out of 8 — a coin flip. Three datasets and three targets now agree
  that the reciprocal branch buys nothing over the same capacity spent
  differently.
* **The engineered divisions do not help DCNv2.** C4 is *negative*: `dcnv2_full`
  is slightly worse than `dcnv2_reduced` on both metrics. The 46 E features were
  built for exactly this model and, on this target, do not pay. This is the
  opposite of KuaiRand, where C4 was strongly positive.
* **AUC and gAUC disagree on C2 and C7.** DCDN beats AFN+ globally but not
  within users; AFN+-full beats DCNv2-full within users but not globally.
  Reporting either metric alone would have produced a different story.

### 3.3 The margins are small

The best-to-worst spread is 0.0054 AUC while seed standard deviations run
0.003–0.009. Only the paired analysis makes anything visible at all, and only
three of seven contrasts clear zero. Nothing here supports a claim larger than
"DCDN is at or slightly above the field on this target".

## 4. Comparison with KuaiRand

| | KuaiRand `watch_time` | KuaiRec `watch_time` | KuaiRec `watch_ratio_gt2` |
|---|---|---|---|
| task learnable | weakly (xAUC 0.54) | no (all arms tie) | **yes** (AUC 0.81) |
| C2 DCDN vs AFN+ | AFN+ wins 2.7–3.0× | tie | **DCDN wins, 8/8** |
| C3 vs capacity | not supported | tie | not supported |
| C4 divisions worth | **+1.7 to +4.0×** | tie | **negative** |
| C5 reduced DCDN vs full DCNv2 | DCDN wins on xAUC, 8/8 | tie | DCDN wins, interval at zero |

The two datasets disagree on C2 and C4 — the direction of both reverses. That
is a reason to be cautious about any single-dataset conclusion, and it is why
neither is quoted without the other.

## 5. Reproduction

```bash
source scripts/env.sh
python scripts/data/kuairec_materialize.py

python scripts/run_kuairand_single_target.py --device cuda:0 \
    --target watch_time --preconditioner none --annex-seeds --save-predictions \
    --views "$DCDN_DATA_CACHE/runtime/kuairec_watch_time_v1" \
    --output-dir results/kuairec_watch_time/single_v1

python scripts/run_kuairand_single_target.py --device cuda:0 \
    --target watch_ratio_gt2 --preconditioner none --annex-seeds --save-predictions \
    --views "$DCDN_DATA_CACHE/runtime/kuairec_watch_time_v1" \
    --output-dir results/kuairec_watch_time/binary_v1
```
