# RQ1 — verdict: **confirmed**

**Claim under test.** Given only base operands that contain none of the target
quotients, does DCDN reconstruct division structure more accurately than DCNv2
at the same interaction capacity?

**Verdict: yes, confirmed.** At an identical 144 interaction parameters DCDN is
**3.64× more accurate in-distribution** and **5.14× more accurate under
high-ratio shift**, with per-seed ranges that do not overlap on either slice or
on any of the three targets individually. The advantage is largest on the
composite quotient `y3 = (n1+n2)/d3`, at **9.34×**.

Section 6 states, with equal prominence, what this does *not* establish. The
verdict is positive about a narrow claim, and the scope discipline is part of
the result rather than a caveat appended to it.

| artifact | value |
|---|---|
| results bundle | `results/synthetic_multiquotient/rq1_v2/results_v1.json` |
| benchmark manifest sha256 | `659d29f565bd51338e8c311a5c1c94335a499fbb2f62017d26581e785d8e72fd` |
| frozen selected configs sha256 | `1f4755903d11d4f8e4066376401179a2…` |
| slices hashed before training | yes, all four |
| seeds | 42, 43, 44 (tuning on 42, validation only) |
| test unlocked | no |
| superseded diagnostic | `docs/synthetic_multiquotient_results.md` (invalid v0, quarantined) |

---

## 1. Scope discipline, stated before the numbers

Synthetic-MultiQuotient-3 is **evidence about an implementation's ability to
learn quotients**. It is not evidence of recommendation quality, and no claim
about ranking may be derived from it. The frozen objective records this as
`role: implementation_capability_benchmark_not_recommendation_evidence`.

Targets are `y1 = n1/d1`, `y2 = n2/d2`, `y3 = (n1+n2)/d3` from strictly positive
inputs `[n1, d1, n2, d2, d3, z1, z2, z3]`. The reciprocal mask is active on
exactly `d1, d2, d3`. Frozen train constants are `V = (1.7315, 1.7250, 5.5334)`,
so a reported value of 1.0 is the constant predictor and lower is better.

## 2. Two kinds of arm, and why the distinction matters

**This is the most important thing to understand before reading any table.**

* **Learning arms** receive the eight base operands only. They must build every
  quotient from operands. These are `dcnv2_reduced`, `dcdn_reduced`,
  `afn_reduced`, `dcnv2_capacity_reduced`, and the `linear` probe.
* **Positive controls** — every arm whose id ends in `_full` — receive the eight
  operands **plus the three target quotients appended verbatim**
  (`explicit_ratio_features` in `src/dcdn/synthetic_multiquotient.py`, an
  11-column input). Its own docstring calls this "the positive-control input …
  what a model achieves when the quotient is handed to it, which bounds what
  learning the quotient can be worth."

A positive control is **not a competitor**. `dcnv2_full` reaching 0.000110
in-distribution does not mean it learned quotients better than anything; it was
given them. The confirmatory comparison is between learning arms only.

*(An earlier draft of this document stated that no arm receives a target ratio
as an input. That was wrong for the three `_full` arms and is corrected here.)*

## 3. The headline result

Macro-NLoss, mean ± sample sd across three seeds. **SEP** marks per-seed ranges
that do not overlap.

**Learning arms — the confirmatory comparison:**

| arm | interaction params | in-distribution | high-ratio OOD |
|---|---:|---|---|
| `linear` probe | — | 0.31030 ± 0.00137 | 130.20 ± 0.42 |
| `dcnv2_reduced` | 144 | 0.001136 ± 0.000488 | 60.43 ± 9.05 |
| **`dcdn_reduced`** | **144** | **0.000312 ± 0.000111** | **11.76 ± 1.91** |
| `dcnv2_capacity_reduced` | 148 | 0.000267 ± 0.000052 | 50.76 ± 3.07 |
| `afn_reduced` | 560,402 | 0.04276 ± 0.00335 | 117.86 ± 52.37 |

**Positive controls — handed the target quotients, shown for reference only:**

| arm | interaction params | in-distribution | high-ratio OOD |
|---|---:|---|---|
| `oracle` (analytic) | — | 0.0000 | 0.0000 |
| `dcnv2_full` | 264 | 0.000110 ± 0.000031 | 69.81 ± 94.85 |
| `dcdn_full` | 264 | 0.000123 ± 0.000072 | 28.27 ± 7.08 |
| `afn_full` | 573,080 | 0.03626 ± 0.00773 | 33.05 ± 8.26 |

The two confirmatory contrasts, plus the capacity control:

| contrast | slice | winner | ratio | separated |
|---|---|---|---|---|
| `dcdn_reduced − dcnv2_reduced` | in-distribution | **DCDN** | **3.644×** | **yes** |
| `dcdn_reduced − dcnv2_reduced` | high-ratio | **DCDN** | **5.138×** | **yes** |
| `dcdn_reduced − dcnv2_capacity_reduced` | in-distribution | DCNv2-capacity | 1.169× | no |
| `dcdn_reduced − dcnv2_capacity_reduced` | high-ratio | **DCDN** | **4.316×** | **yes** |

Per seed, in-distribution: DCDN `0.000244 / 0.000251 / 0.000440` against DCNv2
`0.000711 / 0.001027 / 0.001669`. The intervals are disjoint; DCDN's worst seed
beats DCNv2's best seed by 1.62×.

## 4. The effect is strongest exactly where the hypothesis predicts

Per-target, `dcdn_reduced` against `dcnv2_reduced`, all six comparisons
separated:

| target | definition | in-distribution | high-ratio |
|---|---|---|---|
| `y1` | `n1 / d1` | 3.33× | 4.14× |
| `y2` | `n2 / d2` | 3.87× | 3.68× |
| **`y3`** | **`(n1+n2) / d3`** | **3.90×** | **9.34×** |

`y3` is the composite quotient — a sum in the numerator over a denominator that
appears in no other target. It is the hardest of the three to assemble from
operands, and DCDN's margin there is more than twice its margin on the simple
quotients under high-ratio shift.

One objection worth closing: `y3`'s frozen task constant is 5.5334 against 1.73
for `y1` and `y2`, so its *absolute* normalized loss is on a different scale.
That cannot explain the ordering, because both arms are divided by the **same**
constant — the ratio between them is scale-free.

**The mechanism is not the reciprocal floor.** Floor incidence on `d1, d2, d3`
is **exactly 0.0 on all four slices**. DCDN's accuracy comes from the reciprocal
branch operating in its normal range, not from `div_eps` clamping. This rules
out one specific alternative explanation; it is not by itself proof of the
mechanism.

## 5. Why the capacity control does not undercut the verdict

`dcnv2_capacity_reduced` (148 parameters) is the strongest DCNv2 available at
matched capacity. It **ties DCDN in-distribution** — 0.000267 against 0.000312,
a 1.169× difference whose seed ranges overlap, so no difference is established —
and **loses by 4.32×, separated, under high-ratio shift**.

Read plainly: on the easy slice, extra DCNv2 capacity closes the gap; on the
slice that stresses quotient structure, it does not. The verdict rests on the
second fact. Reporting the first is what makes the second credible.

## 6. What this does **not** establish

**6.1 No scale invariance.** Median relative deviation under a common rescaling
of all inputs, all nine arms:

| arm | median | p95 | kind |
|---|---|---|---|
| `oracle` | 0.000 | 0.000 | analytic |
| `afn_full` | 0.568 | 2.50 | positive control |
| `dcnv2_capacity_reduced` | 1.668 | 6.07 | learning |
| `afn_reduced` | 2.739 | 15.60 | learning |
| `linear` | 4.022 | 106.5 | learning |
| `dcdn_reduced` | 40.74 | 466.0 | learning |
| `dcdn_full` | 54.85 | 402.7 | positive control |
| `dcnv2_full` | 684.0 | 6179 | positive control |
| `dcnv2_reduced` | 3317.0 | 57516 | learning |

Among the five learning arms DCDN is **second worst**, beaten by a plain linear
probe, by AFN+, and by the capacity-matched DCNv2; it is better only than
same-capacity DCNv2. A true reciprocal computation would be invariant to a
common rescale. **DCDN has learned a good local approximation of division, not
division.**

Note that `afn_full`'s leading 0.568 is partly *given* rather than computed: the
appended target ratios are exactly invariant under a common rescale, so a
positive control cannot be used as the standard here.

**6.2 The out-of-distribution slices are bad in absolute terms.** On
`ood_common_scale` and `ood_low_ratio`, `dcdn_reduced` reaches 5975 and 2990
against a constant predictor at 1.0, while `dcnv2_capacity_reduced` reaches
10.65 and 0.860 and `afn_reduced` reaches 29.87 and 0.714. On two of the four
slices DCDN is not merely beaten by other learning arms, it is unusable. The
confirmed result covers the in-distribution and high-ratio slices only, and the
slice name must travel with any quoted number.

**6.3 The task is not trivially linear, but only in-distribution.** The linear
probe sits at 0.31030 in-distribution — 273× behind `dcnv2_reduced`, 995×
behind `dcdn_reduced` and 1163× behind `dcnv2_capacity_reduced`, though only
7.3× behind `afn_reduced`. Under high-ratio shift the linear
probe is **not** far behind: at 130.20 it is 11.1× behind `dcdn_reduced`, 2.2×
behind `dcnv2_reduced`, and only 1.10× behind `afn_reduced` — a 10% gap. On that
slice every arm is doing badly and the ordering among them, not the absolute
level, is what carries the result.

**6.4 It says nothing about recommendation.** Per section 1, and confirmed
empirically: on KuaiRand-Pure the same architecture does **not** beat a
capacity-matched DCNv2 (`docs/rq23_kuairand_quality.md`), and on the
ratio-structured `watch_time` target AFN+ beats DCDN by 2.68–3.02×. A synthetic
capability result does not transfer to ranking quality, and this one demonstrably
did not.

## 7. Summary

**Confirmed:** at equal interaction capacity, and given only base operands,
DCDN reconstructs quotient structure substantially more accurately than DCNv2 —
3.64× in-distribution, 5.14× under high-ratio shift, 9.34× on the composite
quotient — with disjoint seed ranges throughout, a mechanism that does not
depend on the reciprocal floor, and the largest margins where the hypothesis
says they should be.

**Bounded by:** a tie against capacity-matched DCNv2 in-distribution, absent
scale invariance where DCDN is second-worst among learning arms, unusable
behaviour on two of four slices, and no implication whatsoever for
recommendation quality.

## 7a. Addenda

Three follow-up studies in `docs/rq1_addenda.md` explain the OOD behaviour §6.2
reports and change what can be said about it:

* the reciprocal floor is **not** the cause — `div_eps` incidence is exactly 0.0
  on all four slices;
* training each target alone makes DCDN's OOD **worse**, so the three heads were
  regularising rather than interfering;
* the large OOD losses are a property of the **training distribution**. Trained
  over four decades of operand range, DCDN reaches ~1e-06 on every slice and is
  the best of the four learning arms — though every slice is then covered by
  training, which the addendum marks rather than hides.

The verdict below is unchanged; it describes the frozen three-head study. Its
§6.1 reading — a good local approximation of division rather than division — is
confirmed and given a mechanism there.

## 8. Reproduction

```bash
source scripts/env.sh
python scripts/run_synthetic_multiquotient.py \
    --output-dir results/synthetic_multiquotient/rq1_v2
```

Slice hashes are computed and frozen before the first training step; the run
re-verifies them and refuses to proceed on a mismatch.

## 9. How this document was checked

Every numeric claim was recomputed from the bundle by an independent
verification pass (four lenses, each alleged error confirmed by two further
agents). That pass found and this version fixes: a standard deviation inflated
1.7× for `dcnv2_reduced`; an "orders of magnitude" claim about the linear probe
that held only in-distribution and only against some arms; a scale-consistency
table that silently dropped four arms; and, most seriously, the false statement
that no arm receives a target ratio as input — which had allowed three positive
controls to be read as competitors.
