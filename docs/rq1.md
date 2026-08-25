# RQ1 · Quotient learning

> **Can DCDN learn ratio-like functions directly from their operands?**

Theorem 1 establishes that quotient-like terms are *representable* by a DCDN
stack. RQ1 asks the complementary optimisation question: if the quotient itself
is never supplied as an input, does the reciprocal parameterisation make that
dependency easy to **recover from data**?

---

## Setup

**Inputs.** Eight strictly positive operands `[n1, d1, n2, d2, d3, z1, z2, z3]`,
where `z1..z3` are distractors. **No target quotient is among the inputs** —
every model must build the ratio from operands. DCDN's reciprocal mask covers
exactly `d1, d2, d3`.

**Target.** `y = n1/d1`, trained alone with one head per model, so the
comparison is not confounded by multi-task interference.

**Training distribution.** Operands drawn `LogUniform(0.1, 1000)` — four
decades. The multi-scale draw is deliberate: production counters span orders of
magnitude, and a single-decade task would not exercise the regime the operator
was designed for.

**Loss.** Normalised MSE — MSE divided by the target variance on train, so
**1.0 is exactly the constant predictor** and lower is better.

**Evaluation regimes.** Four, differing only in the denominator range:

| regime in the paper | internal name | denominators | what it tests |
|---|---|---|---|
| Reference range | `id` | 1..10, as trained | did it learn what it saw |
| Common rescaling | `ood_common_scale` | all operands x10 | scale invariance — targets are **bitwise identical** to the reference regime |
| Large denominator | `ood_low_ratio` | 20..60 | quotient below the trained band |
| Small denominator | `ood_high_ratio` | 0.2..0.6 | quotient above the trained band |

---

## Models

Ten architectures, all on the REDUCED view: DCDN, DCNv2, DCNv2
capacity-matched, AFN+, DNN/MLP, DeepFM, original DCN, AutoInt+, Wukong-small,
EulerNet. Embedding dim 8, one interaction stack, shared linear head.

EulerNet is in the field on purpose: it is the only arm besides DCDN with a
division-capable inductive bias. Its polar form makes a monomial with real
exponents linear in the log amplitude, so `n1/d1` is one elementary function for
it too, reached without an explicit reciprocal. It is the control that asks
whether the reciprocal branch is *necessary* or merely *sufficient*.

AdamW, 4,000 steps, batch 4,096, gradient clipping 1.0, three seeds (42, 43, 44).
**The learning rate is tuned per arm** on a shared four-point grid
`{1e-4, 3e-4, 1e-3, 3e-3}` on seed 42. Every arm gets the same grid and the same
budget — a single fixed rate is not a fair comparison, because each architecture
has its own optimum and one point systematically penalises whichever arms sit
furthest from it. DCDN is one of them: its tuned rate is `3e-3`, and at a fixed
`1e-3` it reported `2.39e-06` on the reference regime against `4.25e-07` when
allowed to pick.

---

## Result

Normalised MSE, mean of three seeds, lower is better.

| model | tuned lr | Reference range | Common rescaling | Large denominator | Small denominator |
|---|---|---|---|---|---|
| **DCDN** | 3e-3 | **4.25e-07** | 1.21e-06 | **2.81e-07** | **8.74e-07** |
| EulerNet | 1e-4 | 1.25e-06 | 9.79e-07 | 9.31e-07 | 8.19e-05 |
| AutoInt+ | 1e-3 | 1.99e-06 | **8.01e-07** | 4.21e-07 | 3.13e-05 |
| DeepFM | 3e-3 | 2.13e-06 | 2.81e-06 | 2.34e-06 | 3.58e-05 |
| MLP | 1e-3 | 6.18e-06 | 1.77e-05 | 9.36e-06 | 4.51e-03 |
| AFN+ | 1e-4 | 8.55e-06 | 1.82e-05 | 1.06e-05 | 8.08e-04 |
| Wukong-small | 1e-3 | 8.66e-06 | 5.72e-06 | 5.06e-06 | 2.32e-05 |
| DCNv2-capacity-matched | 3e-4 | 2.37e-05 | 1.19e-03 | 2.80e-04 | 2.50e-04 |
| DCN | 1e-4 | 2.91e-05 | 7.03e-04 | 2.83e-05 | 6.21e-04 |
| DCNv2 | 1e-4 | 5.47e-05 | 1.96e-03 | 2.05e-04 | 6.03e-04 |

DCDN is **best of ten on three of four regimes** — reference, large denominator,
small denominator — and **third on common rescaling**, behind AutoInt+ and
EulerNet. Its absolute error stays between `2.81e-07` and `1.21e-06`, six to
seven orders below the constant predictor.

### Against the direct parent

Relative to DCNv2 on the same task, the error is lower by:

| regime | DCDN | DCNv2 | factor |
|---|---|---|---|
| Reference range | 4.25e-07 | 5.47e-05 | **129x** |
| Common rescaling | 1.21e-06 | 1.96e-03 | **1,620x** |
| Large denominator | 2.81e-07 | 2.05e-04 | **729x** |
| Small denominator | 8.74e-07 | 6.03e-04 | **690x** |

Two to three orders of magnitude in every regime. Deepening DCNv2 to a matched
parameter budget closes part of the gap and none of the order: against
DCNv2-capacity-matched the factors are **56x**, 983x, 996x and 286x.

### Against the rest of the field

| regime | DCDN | next best | factor |
|---|---|---|---|
| Reference range | 4.25e-07 | EulerNet 1.25e-06 | **2.9x** |
| Large denominator | 2.81e-07 | AutoInt+ 4.21e-07 | 1.5x |
| Small denominator | 8.74e-07 | Wukong-small 2.32e-05 | **27x** |

The largest margin is on small denominators, where `1/d` reaches 5 and the
quotient leaves the trained band upward — which is where the hypothesis
predicted the reciprocal branch would matter most.

The obvious objection does not hold: `div_eps` incidence is **exactly 0.0 on all
four regimes**, with zero non-finite predictions. Nothing is being clipped.

---

## Answer to RQ1

> DCDN can learn quotient-like functions directly from their operands with very
> high accuracy over the multi-scale range studied. It is best in three of the
> four evaluation regimes and reduces the error of DCNv2 by two to three orders
> of magnitude throughout, showing that reciprocal crossing provides a strong
> inductive bias for learning ratio-like dependencies rather than merely making
> them representable.

---

## What this does not settle

The four regimes all lie **inside** `LogUniform(0.1, 1000)`. This is not evidence
of unrestricted arithmetic extrapolation — the evaluation ranges are
intentionally covered by the training support, and the clipped reciprocal used
in practice differs from the exact Laurent operator outside its unclipped range.

---

## Reproduction

```bash
python scripts/run_synthetic_train_distribution.py --device cuda:0 --tune \
    --only loguniform_01_1000
```

No download required — the task is generated.

Adding an arm does not perturb the others: every model builds its own encoder
from `init_seed` and draws batches from a per-seed CPU generator, so arm order
consumes no shared randomness. Re-running the sweep with EulerNet appended
reproduced all 27 records of the nine-arm run bit-for-bit, tuned learning rates
included.
