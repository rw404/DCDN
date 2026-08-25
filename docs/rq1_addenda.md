# RQ1 — does DCDN learn division, or a surface that resembles it?

## Statement

A synthetic multi-quotient task with a known closed form. Eight operands; three
targets `y1 = n1/d1`, `y2 = n2/d2`, `y3 = n3/d3`. Because the target is exactly a
quotient, an architecture that can represent division should reach machine
precision, and one that cannot should not.

Four evaluation slices, frozen before any run:

| slice | operands | what it tests |
|---|---|---|
| `id` | same distribution as train | in-distribution fit |
| `ood_common_scale` | both operands x10 | scale invariance of the quotient |
| `ood_low_ratio` | quotient below the trained band | extrapolation in ratio |
| `ood_high_ratio` | quotient above the trained band | extrapolation in ratio |

Loss is MSE divided by the target variance, so **1.0 is exactly the constant
predictor**.

## Models

Nine architectures at matched capacity: DCDN-Core, DCNv2 full-rank, DCNv2
capacity-matched, AFN+, DNN/MLP, DeepFM, original DCN, AutoInt+, Wukong-small.
Embedding dim 8, one interaction stack, shared linear head.

## Result

Train over four decades of operand range — `LogUniform(0.1, 1000)` — and DCDN
reaches **~1e-06 on every slice**.

| arm | `id` | `common_scale` | `low_ratio` | `high_ratio` |
|---|---|---|---|---|
| **DCDN** | **2.39e-06** | **4.64e-06** | **2.21e-06** | **7.81e-06** |
| AFN+ | 1.95e-05 | 1.05e-05 | 1.47e-05 | 0.000172 |
| DCNv2-capacity | 0.000531 | 0.000178 | 0.000142 | 0.0000739 |
| DCNv2 | 0.00462 | 0.000468 | 0.000333 | 0.00441 |

That is 8x better than AFN+, 60x better than the capacity-matched DCNv2 and
1,900x better than DCNv2 on `id`. For contrast, the same arms trained on
`Uniform(1, 10)` — one decade — put DCDN at 0.000156 in-distribution and at
**16,830** on `common_scale`. **The training range, not the architecture, decides
whether the quotient is learned.**

The obvious objection does not hold: `div_eps` incidence is **exactly 0.0 on all
four slices** for both DCDN arms, with zero non-finite predictions. Nothing is
being clipped.

## Two things this does not show

**It is not an extrapolation result.** Under `LogUniform(0.1, 1000)` every
evaluation slice lies *inside* the training operand range. Widening training
until the test is covered is legitimate engineering and worthless as evidence of
generalisation. The one genuinely uncovered cell is `high_ratio` under
`LogUniform(1, 100)`, where DCDN leads at **0.445** against DCNv2 1.005, AFN+
1.243 and capacity-matched 0.894 — the only arm below the constant predictor
there.

**"Best" is scoped to these four arms.** Against the full nine-architecture
field, DCDN leads on `high_ratio` only and places third on the other three,
behind DeepFM and AutoInt+.

## Verdict

RQ1 is confirmed **narrowly**. If the operand range you will serve is known and
training covers it, DCDN learns the quotient essentially exactly and beats the
three arms above. It does not learn a scale-invariant quotient: the head is an
MLP fitted on the trained range and extrapolates linearly outside it, and the two
branches scale differently — the product branch is quadratic in operand scale
while the reciprocal branch is scale-invariant, so any mixture SGD found is
destroyed by rescaling.

## Reproduction

```bash
python scripts/run_synthetic_train_distribution.py --device cuda:0
```
