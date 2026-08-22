# RQ1 addenda — why the OOD losses are large, and what fixes them

`docs/rq1_verdict.md` confirms RQ1 but is bounded by one uncomfortable fact: on
`ood_common_scale` and `ood_low_ratio` DCDN's absolute loss is 5,975 and 2,990
against a constant predictor at 1.0. Best-in-class on two slices and unusable on
two others is not the same as having learned division.

These three addenda ask why, and the third one produces losses around **1e-06**.

None of them restates the RQ1 verdict, which is about the frozen three-head
study. All are single-target, on `y1 = n1/d1` unless stated.

| addendum | artifact |
|---|---|
| three independent single-head models | `results/synthetic_multiquotient/single_target_v1/` |
| scale augmentation | `results/synthetic_multiquotient/scale_augment_v1/` |
| training-distribution sweep | `results/synthetic_multiquotient/train_distribution_v1/` |

---

## 1. First, two explanations ruled out by measurement

**The reciprocal floor is not binding.** `div_eps` incidence is **exactly 0.0 on
all four slices** for both DCDN arms, with zero non-finite predictions. Nothing
is being clipped; the floor is not the cause.

**Nor is the model failing to represent the quotient.** A single reciprocal-branch
coordinate computes `inv(x0)_d · W_d(x0)_d = n/d` exactly. The architecture can
express the target; the question is why gradient descent does not find it.

**What the model actually does**, measured directly on a trained DCDN:

| | `id` | `ood_common_scale` |
|---|---|---|
| inputs `n1`, `d1` | 1..10 | **10..100** |
| true `y1` | 0.101..9.760 | **0.101..9.760** (identical) |
| prediction | 0.060..9.718 | **−139.3..417.1** |
| normalized MSE | 0.00023 | **927** |

The targets on `ood_common_scale` are bitwise identical to `id` — same seed, only
the operand scale differs — and a true quotient is invariant to that. The model
returns values from −139 to +417 for a strictly positive target. It has fitted a
surface that agrees with `n/d` on the training box and produces arbitrary values
outside it.

Two mechanisms, both concrete:

* **The head extrapolates.** The model is `head(adapter(core(x)))` with an
  MLP(128, 64) head. Even a perfect reciprocal core feeds an MLP fitted only on
  the training range, and a ReLU network extrapolates linearly with whatever
  slope it happened to learn — including into negatives.
* **The two branches scale differently.** In `x + x₀⊙W_p(x) + inv(x₀)⊙W_d(x)`,
  the product branch is quadratic in operand scale (×10 in → ×100) while the
  reciprocal branch is scale-invariant (`1/(10d)·10n = n/d`). Whatever mixture
  SGD found on the box is destroyed by rescaling.

## 2. Three independent single-head models — the interference hypothesis fails

If the three heads sharing a trunk under an unweighted mean were the problem,
training each target alone should help. Mean over the three per-target models
against the frozen three-head study:

| arm | slice | three-head | single-head | |
|---|---|---|---|---|
| `dcdn_reduced` | id | 0.000312 | 0.000239 | 1.30× better |
| | high_ratio | 11.76 | **14.60** | **0.81× worse** |
| | common_scale | 5,975 | **14,178** | **0.42× worse** |
| | low_ratio | 2,990 | **5,039** | **0.59× worse** |
| `afn_reduced` | id | 0.04276 | 0.000151 | 282× better |
| `dcnv2_capacity_reduced` | id | 0.000267 | 0.0000315 | 8.5× better |

**Single-head training helps almost every arm and hurts DCDN on all three OOD
slices.** The hypothesis is refuted, and the sign is instructive: the three heads
were acting as a *regulariser*, forcing one reciprocal branch to serve three
different quotients and so keeping the representation nearer actual division.
Trained alone, each head overfits the training box harder.

The degradation is even across targets, so it is not one bad head dragging the
mean: `common_scale` is 1.29e4 for `y1`, 2.75e4 for `y2`, 2,214 for `y3`.

**It also breaks the verdict's transfer.** Single-head, `dcnv2_capacity_reduced`
beats `dcdn_reduced` on `id` by 7.6× where the three-head study had them level.
The RQ1 verdict stands for the three-head study and does not carry to this one.

## 3. Scale augmentation — invariance made mandatory

Multiply every training row's operands by a common factor and leave its target
alone: `(s·n)/(s·d) = n/d`. A model fitting a generic surface can no longer do
so. The factor is drawn `LogUniform(1/3, 3)`, **deliberately not covering the
evaluation factor of 10**, so `ood_common_scale` stays a genuine extrapolation
test.

| arm | slice | baseline | augmented | |
|---|---|---|---|---|
| `dcdn_reduced` | common_scale | 13,110 | **43.9** | **299× better** |
| | low_ratio | 3,707 | 206.7 | 17.9× better |
| | high_ratio | 14.4 | **1.65** | 8.7× better |
| | id | 0.00023 | 0.0015 | 6.5× **worse** |
| `afn_reduced` | common_scale | 4.48 | **0.0223** | 201× better |
| | low_ratio | 0.070 | **0.00287** | 24× better |
| | id | 0.0013 | 0.00050 | 2.6× better |
| `dcnv2_capacity_reduced` | common_scale | 4.75 | 0.106 | 45× better |

**Small absolute OOD losses are achievable** — AFN+ reaches 0.0223 and 0.00287,
two to three orders below the constant predictor. DCDN improves 299× but stays at
43.9, still above 1.0.

DCDN takes first place on `high_ratio` (1.65 against AFN+ 7.81 and capacity
18.94) and pays for it on `id`, where it degrades 6.5× while AFN+ and
capacity-matched DCNv2 both *improve*.

**A prediction recorded before the run was wrong.** The script states that
augmentation should not help `high_ratio`, because a common rescale moves operand
magnitudes and not quotient values. It helped 8.7×. The reason: a factor down to
1/3 extends trained denominators from [1,10] to [0.33,10], partially covering
`high_ratio`'s [0.2,0.6]. The caveat was noted for `low_ratio` and underweighted
for `high_ratio`. **Only `ood_common_scale` (×10 against a trained maximum of ×3)
is uncovered**, and that is the honest column.

## 4. Training distribution — where the losses become tiny

Four training distributions, same frozen evaluation slices. `*` marks a slice
lying inside the training operand range, which makes it no longer an
extrapolation test — small numbers under a `*` mean "we trained on it".

| distribution | arm | id | common_scale | low_ratio | high_ratio |
|---|---|---|---|---|---|
| `uniform_1_10` (baseline) | `dcdn_reduced` | 0.000156* | 16,830 | 4,272 | 11.01 |
| | `afn_reduced` | 0.000509* | 2.76 | 0.0484 | 35.61 |
| | `capacity` | 0.000124* | 5.54 | 0.292 | 46.0 |
| `loguniform_1_10` | `dcdn_reduced` | 0.000391* | 1,705 | 600 | 14.16 |
| | `afn_reduced` | 0.000429* | 4.91 | 0.0957 | 24.79 |
| `loguniform_1_100` | `dcdn_reduced` | 0.000106* | 0.000989* | 0.0000369* | **0.445** |
| | `dcnv2_reduced` | 0.000163* | 0.0109* | 0.0000913* | 1.005 |
| | `afn_reduced` | 0.0000436* | 0.000121* | 0.0000118* | 1.243 |
| | `capacity` | 0.0000315* | 0.000619* | 0.000216* | 0.894 |
| **`loguniform_01_1000`** | **`dcdn_reduced`** | **2.39e-06*** | **4.64e-06*** | **2.21e-06*** | **7.81e-06*** |
| | `dcnv2_reduced` | 0.00462* | 0.000468* | 0.000333* | 0.00441* |
| | `afn_reduced` | 0.0000195* | 0.0000105* | 0.0000147* | 0.000172* |
| | `capacity` | 0.000531* | 0.000178* | 0.000142* | 0.0000739* |

**Trained over four decades of operand range, DCDN reaches ~1e-06 on every
slice and is the best of these four arms** — roughly 8× better than AFN+, 60×
better than the capacity-matched DCNv2 and 1,900× better than DCNv2 on `id`.

**That "best" is scoped to the four arms in this table and does not survive the
full field.** `docs/rq1_field_comparison.md` runs the same sweep with nine
architectures: DCDN is then best on `high_ratio` only and third on the other
three, behind DeepFM and AutoInt+. Read that file for the unrestricted
comparison.

Two things this does and does not mean:

* **It answers the practical question.** If the operand range you will serve is
  known and you train across it, DCDN learns the quotient essentially exactly and
  beats the field. Nothing in §1 prevents that; §1 says only that a model fitted
  on a *narrow* box does not extrapolate off it.
* **It is not an extrapolation result.** Every slice carries a `*`. Widening
  training until the test is covered is legitimate engineering and worthless as
  evidence of generalisation, and the table marks it rather than letting the
  small numbers speak.

The one genuinely uncovered cell in the middle rows is `loguniform_1_100`'s
`high_ratio`, where **DCDN leads at 0.445** against DCNv2 1.005, AFN+ 1.243 and
capacity 0.894 — the only arm below the constant predictor there.

## 5. What the three addenda change

* The RQ1 verdict is **unchanged** — it describes the frozen three-head study.
* Its §6.1 reading — *a good local approximation of division, not division* — is
  **confirmed and explained**: the failure is the head extrapolating plus two
  branches that scale differently, not the floor and not interference.
* The verdict's conclusion **does not transfer to single-head training**, where
  the capacity-matched DCNv2 wins in-distribution.
* The large OOD losses are **a property of the training distribution, not of the
  architecture**. Widen it and DCDN is the best of the four arms at ~1e-06.
* Multi-head training was **helping** DCDN's OOD behaviour, which inverts the
  interference hypothesis and is worth carrying into RQ4 on KuaiRand.

## 6. Reproduction

```bash
source scripts/env.sh
python scripts/run_synthetic_single_target.py --device cuda:1
python scripts/run_synthetic_scale_augment.py --device cuda:1
python scripts/run_synthetic_train_distribution.py --device cuda:1
```
