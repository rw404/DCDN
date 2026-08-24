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

## Results

### The training distribution decides everything

Four training distributions, identical evaluation slices. `*` marks a slice that
lies inside the training operand range — a starred number is a fit, not an
extrapolation.

| training distribution | arm | `id` | `common_scale` | `low_ratio` | `high_ratio` |
|---|---|---|---|---|---|
| `uniform_1_10` | DCDN | 0.000156* | 16,830 | 4,272 | 11.01 |
| | AFN+ | 0.000509* | 2.76 | 0.0484 | 35.61 |
| | DCNv2-capacity | 0.000124* | 5.54 | 0.292 | 46.0 |
| `loguniform_1_100` | **DCDN** | 0.000106* | 0.000989* | 0.0000369* | **0.445** |
| | DCNv2 | 0.000163* | 0.0109* | 0.0000913* | 1.005 |
| | AFN+ | 0.0000436* | 0.000121* | 0.0000118* | 1.243 |
| | DCNv2-capacity | 0.0000315* | 0.000619* | 0.000216* | 0.894 |
| **`loguniform_01_1000`** | **DCDN** | **2.39e-06\*** | **4.64e-06\*** | **2.21e-06\*** | **7.81e-06\*** |
| | DCNv2 | 0.00462* | 0.000468* | 0.000333* | 0.00441* |
| | AFN+ | 0.0000195* | 0.0000105* | 0.0000147* | 0.000172* |
| | DCNv2-capacity | 0.000531* | 0.000178* | 0.000142* | 0.0000739* |

Trained across four decades of operand range, **DCDN reaches ~1e-06 on every
slice** — 8x better than AFN+, 60x better than the capacity-matched DCNv2 and
1,900x better than DCNv2 on `id`.

**That "best" is scoped to these four arms.** Against the full nine-architecture
field, DCDN leads on `high_ratio` only and places third on the other three,
behind DeepFM and AutoInt+.

### The large out-of-distribution losses are a property of the training range

Two explanations were ruled out by measurement:

* **The reciprocal floor is not binding.** `div_eps` incidence is **exactly 0.0
  on all four slices** for both DCDN arms, with zero non-finite predictions.
* **The architecture can represent the target.** One reciprocal coordinate
  computes `inv(x0)_d · W_d(x0)_d = n/d` exactly. Gradient descent not finding it
  is a different failure from not being able to express it.

What actually happens, measured on a trained DCDN:

| | `id` | `ood_common_scale` |
|---|---|---|
| inputs | 1..10 | **10..100** |
| true `y1` | 0.101..9.760 | **identical** |
| prediction | 0.060..9.718 | **-139.3..417.1** |
| normalized MSE | 0.00023 | **927** |

The targets are bitwise identical — only the operand scale moved, and a true
quotient is invariant to that. Two mechanisms: the MLP head extrapolates
linearly outside its trained range, and the two branches scale differently —
the product branch is quadratic in operand scale while the reciprocal branch is
scale-invariant, so any mixture SGD found on the box is destroyed by rescaling.

### Scale augmentation makes the invariance mandatory

Multiply every training row's operands by a common factor and leave its target
alone: `(s·n)/(s·d) = n/d`. The factor is `LogUniform(1/3, 3)`, deliberately not
covering the evaluation factor of 10.

| arm | slice | baseline | augmented | |
|---|---|---|---|---|
| DCDN | `common_scale` | 13,110 | **43.9** | **299x better** |
| | `low_ratio` | 3,707 | 206.7 | 17.9x |
| | `high_ratio` | 14.4 | **1.65** | 8.7x |
| | `id` | 0.00023 | 0.0015 | 6.5x **worse** |
| AFN+ | `common_scale` | 4.48 | **0.0223** | 201x |
| DCNv2-capacity | `common_scale` | 4.75 | 0.106 | 45x |

DCDN takes first place on `high_ratio` and pays for it in-distribution.

### Multi-head training was helping, not hurting

Training each target with its own model, against the frozen three-head study:

| arm | slice | three-head | single-head | |
|---|---|---|---|---|
| DCDN | `id` | 0.000312 | 0.000239 | 1.30x better |
| | `high_ratio` | 11.76 | **14.60** | **0.81x worse** |
| | `common_scale` | 5,975 | **14,178** | **0.42x worse** |
| | `low_ratio` | 2,990 | **5,039** | **0.59x worse** |
| AFN+ | `id` | 0.04276 | 0.000151 | 282x better |
| DCNv2-capacity | `id` | 0.000267 | 0.0000315 | 8.5x better |

Single-head training helps almost every arm and **hurts DCDN on all three
out-of-distribution slices**. The three heads were acting as a regulariser,
forcing one reciprocal branch to serve three quotients and keeping the
representation nearer actual division.

## Verdict

RQ1 is confirmed **narrowly**. DCDN learns the quotient essentially exactly when
the operand range it will serve is covered by training. It does not learn a
scale-invariant quotient: outside the trained range the head extrapolates and
the two branches scale differently. Widening the training distribution is
legitimate engineering and is not evidence of generalisation, and the tables
above mark every such cell.

## Reproduction

```bash
python scripts/run_synthetic_train_distribution.py --device cuda:0
python scripts/run_synthetic_scale_augment.py       --device cuda:0
python scripts/run_synthetic_single_target.py       --device cuda:0
```
