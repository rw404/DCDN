# The multi-head model, head by head

How every arm behaves on each of the five KuaiRand-Pure heads when all five are
trained together under Macro-NLoss. Validation, frozen three seeds (42, 43, 44),
test never unlocked. Bundle: `results/kuairand_multitask/final_mt_v2/`,
tables: `results/kuairand_multitask/matrix_v1/multi_head_per_head.csv`.

Read §4 before quoting any gAUC number from the sparse heads.

---

## 1. `long_view` — the frozen primary confirmatory target

Train positive rate 0.3146, validation 0.0880.

| arm | log-loss | AUC | gAUC | AUPRC |
|---|---|---|---|---|
| **`dcdn_full`** | **0.3117** | **0.6266** | 0.5257 | **0.1417** |
| `dcdn_reduced` | 0.5664 | 0.6171 | 0.5357 | 0.1270 |
| `dcnv2_capacity_reduced` | 0.5941 | 0.6148 | **0.5374** | 0.1248 |
| `dcnv2_full` | 0.6289 | 0.5875 | 0.5281 | 0.1182 |
| `afn_reduced` | 0.4246 | 0.5600 | 0.4462 | 0.1070 |
| `dcnv2_reduced` | 0.6874 | 0.5597 | 0.4799 | 0.1095 |
| `afn_full` | 0.4665 | 0.5571 | 0.4804 | 0.1080 |

**This is DCDN's best head.** It takes the top two AUC positions, and both
same-feature comparisons separate: `dcdn_full > dcnv2_full` and
`dcdn_reduced > dcnv2_reduced`. Note that gAUC does not agree — the best
within-user ranker is the capacity-matched DCNv2, and `dcdn_full`, top on global
AUC, is only fourth on gAUC.

## 2. `like`

Train positive rate 0.0183, validation 0.0048.

| arm | log-loss | AUC | gAUC | AUPRC |
|---|---|---|---|---|
| `afn_full` | 0.0409 | **0.8073** | 0.4265 | 0.0753 |
| `dcnv2_capacity_reduced` | 0.0408 | 0.8007 | 0.4342 | 0.0888 |
| `dcdn_full` | **0.0291** | 0.7941 | 0.4289 | **0.0984** |
| `dcnv2_reduced` | 0.0317 | 0.7902 | 0.4212 | 0.0538 |
| `dcdn_reduced` | 0.0507 | 0.7793 | **0.4777** | 0.0692 |
| `afn_reduced` | 0.0348 | 0.7729 | 0.4012 | 0.0442 |
| `dcnv2_full` | 0.0965 | 0.7222 | 0.5046 | 0.0532 |

The best-discriminated head by global AUC (0.72–0.81), and the arms are tightly
packed. **`dcnv2_full` is last** — the arm with the most features is the worst
here, which is worth noting against any assumption that more features help
monotonically. Every gAUC except `dcnv2_full`'s sits **below 0.5**: within a
single user's impressions, no arm orders likes better than chance.

## 3. `forward` and `hate` — the sparse heads

`forward`: ~160 train positives, validation rate 0.0003.
`hate`: ~140 train positives, validation rate 0.0011.

| arm | `forward` AUC | `forward` gAUC | `hate` AUC | `hate` gAUC |
|---|---|---|---|---|
| `afn_full` | **0.5481** | 0.4361 | **0.7793** | 0.3456 |
| `dcdn_full` | 0.5474 | 0.4159 | 0.7565 | 0.3445 |
| `afn_reduced` | 0.5466 | 0.4290 | 0.7623 | 0.3734 |
| `dcnv2_full` | 0.5293 | 0.5605 | 0.5466 | 0.4953 |
| `dcdn_reduced` | 0.5151 | 0.5795 | 0.6257 | 0.4402 |
| `dcnv2_capacity_reduced` | 0.5096 | 0.5647 | 0.6545 | 0.4645 |
| `dcnv2_reduced` | 0.5090 | **0.6005** | 0.6868 | 0.4621 |

`forward` is essentially unlearned by everything: global AUC 0.509–0.548 across
all seven arms. `hate` splits the field — AFN+ and `dcdn_full` reach 0.76–0.78
while `dcnv2_full` is at 0.5466, barely above chance.

**Every `hate` gAUC is below 0.5.** Within a user, no arm orders hate better
than random. The global AUC of 0.78 comes from separating users who hate things
from users who do not, not from ordering within a user.

Global AUC and gAUC are close to *anti-correlated* on `forward`: the three arms
with the best gAUC (`dcnv2_reduced` 0.6005, `dcdn_reduced` 0.5795,
`dcnv2_capacity_reduced` 0.5647) are the three with the worst global AUC.

## 4. Why the sparse-head gAUC numbers are nearly meaningless

gAUC is defined only over users whose impressions contain **both** classes.
Coverage by head:

| head | user coverage |
|---|---|
| `long_view` | **42.7%** |
| `like` | 3.6% |
| `hate` | 0.7% |
| `forward` | 0.4% |

A `forward` gAUC is an average over the 0.4% of users who both forwarded and did
not forward something. It is a real number computed correctly and it describes
almost nobody. **Only `long_view`'s gAUC is worth interpreting**, and even that
covers under half the users. Coverage is carried in
`multi_head_per_head.csv` beside every value so this cannot be lost downstream.

## 5. `watch_time` inside the multi-head model

| arm | normalized MSE | X-AUC | user X-GAUC | MAE (s) |
|---|---|---|---|---|
| `dcnv2_capacity_reduced` | 0.0089 | **0.5400** | **0.5061** | 28.07 |
| `afn_reduced` | 0.0044 | 0.5373 | 0.4956 | 17.65 |
| `dcnv2_full` | 0.0130 | 0.5282 | 0.5049 | 17.60 |
| `afn_full` | **0.0047** | 0.5271 | 0.5075 | **13.10** |
| `dcdn_reduced` | 0.0264 | 0.5246 | 0.5018 | 37.79 |
| `dcdn_full` | 0.0305 | 0.5189 | 0.5061 | 39.09 |
| `dcnv2_reduced` | 0.0934 | 0.5102 | 0.4982 | 64.72 |

**Both DCDN arms are near the bottom**, 5th and 6th of seven on X-AUC, with the
worst MAE of any converged arm (37.8 s and 39.1 s against AFN+'s 13.1 s).

Trained *alone* on the same target, `dcdn_reduced` moves to 2nd of seven at
X-AUC 0.5415 — see `docs/rq2_watch_time_verdict.md` §4. The multi-head setting
costs DCDN more on `watch_time` than it costs any other arm, which is the
sharpest single piece of evidence for the interference hypothesis in §6.

## 6. What the head-by-head view shows

**6.1 No arm wins everywhere, and the spread is wide.** `dcdn_full` is first on
`long_view`, third on `like`, second on `forward`, third on `hate`, sixth on
`watch_time`. `afn_full` is first on `like`, `forward` and `hate`, and last but
one on `long_view`. `dcnv2_full` is last on `like` and near-chance on `hate`
while being mid-table on `long_view`. Any single-head headline chosen from these
five would support a different architecture.

**6.2 DCDN's multi-head strength is concentrated in `long_view`.** It is the one
head where DCDN separates from DCNv2 at both feature sets. On `like`, `forward`
and `hate` it is mid-field, and on `watch_time` it is near the bottom. The
aggregate Macro-NLoss ranking therefore reflects `long_view` and `watch_time`
pulling in opposite directions.

**6.3 The interference reading.** `watch_time` is the head DCDN loses most by
being trained jointly, and `long_view` is the head it gains most on. That is
consistent with the trunk being allocated toward `long_view` at `watch_time`'s
expense under the unweighted Macro-NLoss. It is an observation with a plausible
mechanism, not a tested claim; the test proposed in
`docs/kuairand_matrix.md` §4 (RQ4) — fix the architecture, vary the head count,
measure one head throughout — would settle it.

**6.4 Global AUC and gAUC disagree systematically here**, not occasionally. On
`long_view` the AUC leader is fourth on gAUC; on `forward` the ordering is close
to inverted; on `hate` every arm is below chance within users. Reporting one of
the two alone would have produced a confident and different story either way.

## 7. Reproduction

```bash
source scripts/env.sh
python scripts/run_kuairand_multitask.py --device cuda:0 \
    --preconditioner rms --annex-seeds \
    --output-dir results/kuairand_multitask/final_mt_v2
python scripts/report_kuairand_matrix.py
```
