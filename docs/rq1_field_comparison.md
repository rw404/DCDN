# RQ1 against the full field — setup, results, and an honest verdict

This file states the RQ1 setup explicitly, runs **nine architectures** instead of
the frozen four, and gives the `id` / `common_scale` / `low_ratio` / `high_ratio`
table across three training distributions.

**The headline first, because it is not the one that was hoped for. Against the
full field DCDN does not sweep the slices.** It wins `high_ratio` at the frozen
training distribution and again at the widest one, and it wins nothing else
anywhere. On `id`, `common_scale` and `low_ratio` it is third or lower of nine.

An earlier statement of mine — "DCDN is best on all four slices" at the widest
distribution — was true **of the four frozen arms only** and is false against
nine. `docs/rq1_addenda.md` §4 scoped it to four arms; this file is the
unrestricted comparison and supersedes any unqualified reading.

Bundle: `results/synthetic_multiquotient/train_distribution_v2/`

---

## 1. The setup, stated plainly

**Inputs.** Eight strictly positive operands, `[n1, d1, n2, d2, d3, z1, z2, z3]`.
`z1..z3` are distractors. **No target quotient is among the inputs** — every
model must build the ratio from operands. The reciprocal mask that DCDN's
division branch reads is active on exactly `d1, d2, d3`.

**Targets.** Three exact quotients: `y1 = n1/d1`, `y2 = n2/d2`, and the composite
`y3 = (n1+n2)/d3`. This file trains **`y1` alone**, one head per model, so the
comparison is not confounded by multi-task interference.

**Training.** MSE divided by the target's variance on the training set, so **1.0
is the constant predictor** and lower is better. AdamW, 4,000 steps, batch 4,096,
gradient clipping 1.0, learning rate fixed at 1e-3 for every arm, three seeds
(42, 43, 44). No per-arm tuning: the grid is identical, so no arm gets a budget
another lacks.

**Evaluation slices.** Four, differing **only in the denominator range**:

| slice | denominators | what it tests |
|---|---|---|
| `id` | 1..10, the training range | did it learn what it saw |
| `ood_high_ratio` | **0.2..0.6** | quotient *above* the trained range |
| `ood_low_ratio` | **20..60** | quotient *below* the trained range |
| `ood_common_scale` | all operands **×10** | scale invariance — the targets are **bitwise identical to `id`**, so a true quotient is unchanged |

**Architectures.** Nine, all on the REDUCED view: DCDN, DCNv2, DCNv2
capacity-matched, AFN+, DNN/MLP, DeepFM, original DCN, AutoInt+, Wukong-small.
The synthetic `_full` arms are **positive controls** that receive the target
quotients as inputs (`docs/rq1_verdict.md` §2) and are excluded here — putting
the field on FULL would hand it the answer.

**Training distributions.** The frozen benchmark draws operands `U[1,10]`. Two
wider log-uniform alternatives are added, because `docs/rq1_addenda.md` showed
the OOD losses are a property of the training distribution. `*` marks a slice
whose operand range lies **inside** the training range — a covered slice is not
an extrapolation test, and small numbers under a `*` mean "we trained on it".

## 2. The frozen setting — operands `U[1,10]`

Covers `id` only.

| arm | `id` * | `common_scale` | `low_ratio` | `high_ratio` |
|---|---|---|---|---|
| `dcnv2_capacity_reduced` | **0.000124** | 5.54 | 0.292 | 46.0 |
| `autoint_reduced` | 0.000130 | 6.28 | 0.476 | 49.5 |
| **`dcdn_reduced`** | 0.000156 | 16,830 | 4,272 | **11.01** |
| `wukong_reduced` | 0.000242 | **0.0479** | **0.00150** | 53.4 |
| `dcn_reduced` | 0.000355 | 89,000 | 2,073 | 35.9 |
| `afn_reduced` | 0.000509 | 2.76 | 0.0484 | 35.6 |
| `mlp_reduced` | 0.000545 | 9.37 | 0.558 | 51.1 |
| `deepfm_reduced` | 0.000554 | 92.7 | 0.915 | 45.9 |
| `dcnv2_reduced` | 0.001825 | 20,560,000 | 363,300 | 52.0 |

**DCDN wins `high_ratio` outright** — 11.01 against a field where nothing else
gets below 35. That is the slice its reciprocal branch is built for: small
denominators, quotients above the trained range.

**It is third on `id`** and **catastrophic on the other two**, at 16,830 and
4,272 against a constant predictor of 1.0, while Wukong-small sits at 0.048 and
0.0015. The RQ1 verdict's own §6.2 says this; the field makes it starker, because
now four different architectures stay near or below 1.0 on the slices where DCDN
is in the thousands.

**The specific RQ1 claim survives.** Against DCNv2 at matched capacity, DCDN is
11.7× better on `id` (0.000156 against 0.001825) and 4.7× better on `high_ratio`
(11.01 against 52.0). The verdict compared DCDN to three chosen comparators and
that comparison holds; what the field adds is that other architectures beat both
of them on two slices.

## 3. Widening the training distribution

### `loguniform_1_100` — covers `id`, `common_scale`, `low_ratio`

| arm | `id` * | `common_scale` * | `low_ratio` * | `high_ratio` |
|---|---|---|---|---|
| `deepfm_reduced` | **1.28e-05** | 1.13e-04 | 8.38e-06 | 0.471 |
| `wukong_reduced` | 1.86e-05 | **1.24e-05** | **2.37e-06** | 0.133 |
| `autoint_reduced` | 2.32e-05 | 5.14e-05 | 1.50e-05 | **0.0536** |
| `dcnv2_capacity_reduced` | 3.15e-05 | 6.19e-04 | 2.16e-04 | 0.894 |
| `afn_reduced` | 4.36e-05 | 1.21e-04 | 1.18e-05 | 1.243 |
| **`dcdn_reduced`** | 1.06e-04 | 9.89e-04 | 3.69e-05 | 0.445 |
| `dcnv2_reduced` | 1.63e-04 | 1.09e-02 | 9.13e-05 | 1.005 |
| `mlp_reduced` | 2.47e-04 | 5.68e-04 | 9.25e-05 | 0.241 |
| `dcn_reduced` | 4.59e-03 | 2.42e-02 | 1.65e-03 | 0.532 |

**DCDN wins nothing here.** On the only uncovered slice, `high_ratio`, it is
**fourth** at 0.445 behind AutoInt+ 0.054, Wukong 0.133 and MLP 0.241.

### `loguniform_01_1000` — covers everything

| arm | `id` * | `common_scale` * | `low_ratio` * | `high_ratio` * |
|---|---|---|---|---|
| `deepfm_reduced` | **8.66e-07** | 1.46e-06 | 9.29e-07 | 9.38e-06 |
| `autoint_reduced` | 1.99e-06 | **8.01e-07** | **4.21e-07** | 3.13e-05 |
| **`dcdn_reduced`** | 2.39e-06 | 4.64e-06 | 2.21e-06 | **7.81e-06** |
| `mlp_reduced` | 6.18e-06 | 1.77e-05 | 9.36e-06 | 4.51e-03 |
| `wukong_reduced` | 8.66e-06 | 5.72e-06 | 5.06e-06 | 2.32e-05 |
| `afn_reduced` | 1.95e-05 | 1.05e-05 | 1.47e-05 | 1.72e-04 |
| `dcnv2_capacity_reduced` | 5.31e-04 | 1.78e-04 | 1.42e-04 | 7.39e-05 |
| `dcn_reduced` | 1.04e-03 | 2.96e-03 | 7.06e-04 | 1.26e-03 |
| `dcnv2_reduced` | 4.62e-03 | 4.68e-04 | 3.33e-04 | 4.41e-03 |

Every arm reaches the 1e-06 to 1e-03 range: **with enough operand coverage the
quotient is learnable by almost anything.** DCDN wins `high_ratio` at 7.81e-06
and is third on the other three, behind DeepFM and AutoInt+.

Every slice carries a `*`. This is not a generalisation result.

## 4. So does RQ1 hold?

**Yes, as stated — and the statement is narrower than it sounds.**

| claim | verdict |
|---|---|
| At equal interaction capacity DCDN reconstructs quotients better than DCNv2 | **holds**, 11.7× on `id`, 4.7× on `high_ratio` here, and with disjoint seed ranges in the frozen three-head study |
| DCDN is best on the slice where the quotient exceeds the trained range | **holds** at the narrow and the widest distributions |
| DCDN is best overall | **does not hold** — third or lower on three of four slices against the field |
| DCDN generalises out of distribution | **does not hold** — thousands against a constant predictor of 1.0 at the frozen setting |
| Wide training fixes it | **holds, but for everyone** — every arm reaches 1e-06, and DCDN is not the biggest beneficiary |

**The one thing DCDN does that nothing else does** is `high_ratio` at the frozen
distribution: 11.01 where the next best is 35.6 and the field median is 46. Small
denominators are exactly where `1/d` matters most, and that is a real, specific,
reproducible property of the reciprocal branch.

**The one thing it does worst** is `common_scale`, where at the frozen setting it
is 16,830 against Wukong's 0.048 — a factor of 350,000. A common rescale leaves
the quotient unchanged, so this measures scale invariance directly, and DCDN has
almost none.

Both facts follow from the same mechanism (`docs/rq1_addenda.md` §1): the product
branch is quadratic in operand scale while the reciprocal branch is invariant, so
whatever mixture SGD fits on the training box helps where reciprocals dominate
and breaks under rescaling.

## 5. Reading limits

* Single target `y1`, one learning rate for all arms, three seeds. The frozen
  RQ1 study tunes per arm over two learning rates and reports three targets; its
  numbers are not directly comparable to these, and it — not this file — is the
  verdict.
* No per-arm tuning means an arm could be under-served by 1e-3. The grid is
  identical for all nine, so no arm is favoured, but none is optimised either.
* The `*` columns are covered by training and are engineering results, not
  evidence of extrapolation.

## 6. Reproduction

```bash
source scripts/env.sh
python scripts/run_synthetic_train_distribution.py --device cuda:0 \
    --output-dir results/synthetic_multiquotient/train_distribution_v2
```
