# RQ1 — does an explicit reciprocal branch reconstruct division?

## Statement

> Given only base operands that contain none of the target quotients, does an
> explicit reciprocal branch reconstruct division more accurately than
> architectures without one, at comparable capacity?

**Inputs.** Eight strictly positive operands `[n1, d1, n2, d2, d3, z1, z2, z3]`,
where `z1..z3` are distractors. **No target quotient is among the inputs** —
every model must build the ratio from operands. DCDN's reciprocal mask is active
on exactly `d1, d2, d3`.

**Target.** `y1 = n1/d1`, trained alone with one head per model, so the
comparison is not confounded by multi-task interference.

**Loss.** MSE divided by the target variance on train, so **1.0 is exactly the
constant predictor** and lower is better.

**Evaluation slices.** Four, differing only in the denominator range:

| slice in the paper | internal name | denominators | what it tests |
|---|---|---|---|
| Reference range | `id` | 1..10, as trained | did it learn what it saw |
| Common rescaling | `ood_common_scale` | all operands ×10 | scale invariance — targets are **bitwise identical** to the reference slice |
| Large denominator | `ood_low_ratio` | 20..60 | quotient below the trained band |
| Small denominator | `ood_high_ratio` | 0.2..0.6 | quotient above the trained band |

## Models

Nine architectures, all on the REDUCED view: DCDN, DCNv2, DCNv2
capacity-matched, AFN+, DNN/MLP, DeepFM, original DCN, AutoInt+, Wukong-small.
Embedding dim 8, one interaction stack, shared linear head.

AdamW, 4,000 steps, batch 4,096, gradient clipping 1.0, three seeds (42, 43, 44).
**The learning rate is tuned per arm** on a shared four-point grid
`{1e-4, 3e-4, 1e-3, 3e-3}` on seed 42. Every arm gets the same grid and the same
budget — a single fixed rate is not a fair comparison, because each architecture
has its own optimum and one point systematically penalises whichever arms sit
furthest from it. DCDN is one of them: its tuned rate is `3e-3`, and at a fixed
`1e-3` it reported `2.39e-06` on the reference slice against `4.25e-07` when
allowed to pick.

## Result

Operands drawn `LogUniform(0.1, 1000)` — four decades.

| model | tuned lr | Reference range | Common rescaling | Large denominator | Small denominator |
|---|---|---|---|---|---|
| **DCDN** | 3e-3 | **4.25e-07** | 1.21e-06 | **2.81e-07** | **8.74e-07** |
| AutoInt+ | 1e-3 | 1.99e-06 | **8.01e-07** | 4.21e-07 | 3.13e-05 |
| DeepFM | 3e-3 | 2.13e-06 | 2.81e-06 | 2.34e-06 | 3.58e-05 |
| MLP | 1e-3 | 6.18e-06 | 1.77e-05 | 9.36e-06 | 4.51e-03 |
| AFN+ | 1e-4 | 8.55e-06 | 1.82e-05 | 1.06e-05 | 8.08e-04 |
| Wukong-small | 1e-3 | 8.66e-06 | 5.72e-06 | 5.06e-06 | 2.32e-05 |
| DCNv2-capacity-matched | 3e-4 | 2.37e-05 | 1.19e-03 | 2.80e-04 | 2.50e-04 |
| DCN | 1e-4 | 2.91e-05 | 7.03e-04 | 2.83e-05 | 6.21e-04 |
| DCNv2 | 1e-4 | 5.47e-05 | 1.96e-03 | 2.05e-04 | 6.03e-04 |

**DCDN is best of nine on three of four slices** and second on common rescaling,
behind AutoInt+. Its absolute errors run 2.8e-07 to 1.2e-06 — seven orders below
the constant predictor.

| slice | DCDN | next best | factor |
|---|---|---|---|
| Reference range | 4.25e-07 | AutoInt+ 1.99e-06 | **4.7×** |
| Large denominator | 2.81e-07 | AutoInt+ 4.21e-07 | 1.5× |
| Small denominator | 8.74e-07 | Wukong 2.32e-05 | **27×** |

Against DCNv2 at matched capacity: **56×** on the reference slice, 996× on large
denominators, 286× on small ones. The largest margin is on small denominators,
where `1/d` reaches 5 and the quotient leaves the trained range upward — which is
where the hypothesis predicted the reciprocal branch would matter most.

The obvious objection does not hold: `div_eps` incidence is **exactly 0.0 on all
four slices**, with zero non-finite predictions. Nothing is being clipped.

## Two things this does not settle

**It is not an extrapolation result.** All four slices lie inside
`LogUniform(0.1, 1000)`. Widening training until the test is covered is
legitimate engineering and no evidence of generalisation.

**Scale invariance fails at the frozen benchmark.** Trained on `Uniform(1, 10)` —
one decade — DCDN reports **16,830** on common rescaling against a constant
predictor of 1.0, and Wukong-small is 350,000× better there. The targets on that
slice are bitwise identical to the reference slice, so a true quotient would be
unchanged. The branch behaves like a very effective basis function for quotients
inside the range it was fitted on, not like a reciprocal operator.

## Verdict

RQ1 holds in its practical form and fails in its strong one. **If the operand
range you will serve is known and training covers it, DCDN learns the quotient to
within ~1e-06 and is the best of nine architectures on three of four slices.** It
does not learn a scale-invariant quotient, and the advantage is a property of the
training distribution as much as of the architecture.

## Reproduction

```bash
python scripts/run_synthetic_train_distribution.py --device cuda:0 --tune
```
