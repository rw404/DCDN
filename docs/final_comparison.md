# Final comparison — what five datasets say about DCDN

Four public datasets with fourteen architectures at eight seeds each, plus one
production feed with four arms at one run each. This file states what is
established, what is not, and which evidence carries which weight.

| dataset | slice | arms | seeds | operands RED / FULL | denominators |
|---|---|---|---|---|---|
| Yambda-50M | `listen50` | 14 | 8 | 86 / 134 | 12 (14%) |
| KuaiRec | `watch_ratio_gt2` | 14 | 8 | 50 / 96 | 5 (10%) |
| KuaiRand-Pure | `watch_time` | 14 | 8 | 132 / 244 | 24 (18%) |
| VK-LSVD `ur0.01_ir0.01` | `timespent_gt10` | 14 | 8 | 99 / 216 | 36 (36%) |
| **Production NR feed** | five heads | 4 | **1** | — | — |

REDUCED withholds the manually engineered quotients of historical counters while
keeping every numerator and every denominator. FULL adds the quotients back.

## 1. RQ2 — can a reduced-input model reach the full-input field?

**On public data, no, with one exception that is not DCDN.**

| dataset | best reduced arm | reaches the best full arm? |
|---|---|---|
| Yambda `listen50` | `afn_reduced`, 10th of 14 | no — −0.0053, 0/8 seeds |
| KuaiRec `watch_ratio_gt2` | **`eulernet_reduced`, 3rd of 14** | **yes** — −0.00044, not significant |
| KuaiRand `watch_time` | `afn_reduced`, 4th of 14 | no — −0.0146, 2/8 |
| VK-LSVD ur `timespent_gt10` | `afn_reduced`, 8th of 14 | no — −0.0062, 0/8 |

`dcdn_reduced` ranks 13th, 6th, 7th and 13th. It is the best reduced arm on
**none** of the four.

**On production traffic, yes, on three heads of five.** DCDN-REDUCED against the
full-feature DCNv2-FULL baseline: fullscreen +0.0008, like −0.0007, share
−0.0011 — all inside the logging spread. It loses on viewtime (−0.0083) and
dislike (−0.0075).

The mechanism is visible in the removal cost. Taking the engineered ratios away
costs DCNv2 −0.0442 viewtime, −0.0163 fullscreen and −0.0152 like; it costs DCDN
−0.0194, −0.0045 and −0.0059. **DCDN absorbs 56–72% of the loss on those three
heads**, and absorbs none of it on dislike (−42%) or share (−9%).

This is the strongest RQ2 evidence in the programme and it is also the weakest
by protocol: **one run per arm, no seed variance**, so differences under ~0.002
are not separable from run-to-run noise. That threshold covers the three
"level with baseline" contrasts entirely. The large numbers — viewtime, and the
whole reduced-vs-reduced column — clear it comfortably.

## 2. RQ3 — quality per unit of serving cost

**DCDN takes the best quality-per-millisecond on all four public slices**, and a
paired seed-by-seed test confirms it against all six comparators every time:
24 of 24 pairwise comparisons.

Where the margin comes from matters more than the margin. Against
`dcnv2_capacity_reduced` — the only comparator matched within 0.2% on parameters
and FLOPs:

| dataset | quality difference | verdict | latency ratio |
|---|---|---|---|
| Yambda | −0.00064, 0/8 | significantly worse | 1.31x faster |
| KuaiRec | +0.00060, 4/8 | not significant | 1.29x faster |
| KuaiRand | −0.00053, 1/8 | not significant | 1.46x faster |
| VK-LSVD ur | −0.00790, 0/8 | significantly worse | 1.31x faster |

**At equal parameters and FLOPs DCDN never wins on quality** — worse on two,
indistinguishable on two, ahead on none. The ratio win is a latency win, stable
across four geometries from 50/96 to 132/244 operands wide, because capacity
matching equalises analytic counts and not realised kernels.

## 3. Division is not exclusive to DCDN

Two arms have division inside their hypothesis class by construction: **AFN+**
through a log-exp branch and **EulerNet** through the polar form, where a
monomial with arbitrary real exponents is linear on log-amplitude and phase.
Both beat DCDN on median quality rank — 2.5 and 3.5 against 4.0 — and both pay
for it:

| arm | parameters vs DCDN-RED | latency vs DCDN-RED |
|---|---|---|
| `eulernet_full` | 18.1x | 11.1x |
| `eulernet_reduced` | 12.3x | 8.4x |
| `afn_full` | 53.8x | 9.9x |
| `afn_reduced` | 39.7x | 7.1x |

**No arm the size of DCDN uses division at all.** That is the defensible form of
the novelty claim: not "the only model that can divide", which
`DCDN_HANDOFF.md` explicitly forbids, but "the only small one".

## 4. What predicts DCDN's standing

Denominator density — the share of REDUCED operands that are legitimate positive
denominators — orders `dcdn_full` across the datasets:

| dataset | density | `dcdn_full` rank of 14 |
|---|---|---|
| KuaiRec | 10% | 1 |
| Yambda | 14% | 7 |
| KuaiRand | 18% | 9 |
| VK-LSVD ur | 36% | 5 |
| VK-LSVD up (older view) | 9% | 5, 8, 6, 11 across four targets |

The prediction was registered before the VK-ur view was built and it half held:
raising density from 9% to 36% on the same dataset family moved `dcdn_full` from
mid-table to first on three of four targets, and moved `dcdn_reduced` not at
all. **Denser division structure helps when the quotients are in the input; it
does not help a model asked to rebuild them.**

## 5. The honest bottom line

**Established.**

* DCDN is **1.29x–1.46x faster** than DCNv2 at matched parameters and FLOPs,
  stable across four geometries, with quality that is indistinguishable on two
  datasets and worse on two.
* `dcdn_full` is the only architecture that is simultaneously top-4 on quality
  and top-4 on cost across four public datasets.
* On production traffic, DCDN absorbs 56–72% of the cost of removing every
  engineered ratio on three heads of five, and lands level with the
  full-feature baseline on those three.
* DCDN-FULL wins all five production heads against DCNv2-FULL.

**Not established.**

* **DCDN is not the most accurate architecture** on any public dataset median,
  and it never beats its capacity-matched comparator on quality.
* **RQ2 fails on three of four public datasets**, and on the fourth the arm that
  succeeds is `eulernet_reduced`.
* **Division is not DCDN's exclusive capability.** Two arms have it and rank
  higher; DCDN's claim is about size and cost.
* **No arm is stable.** Thirteen of fourteen have both a top-three and a
  bottom-three dataset. A single-dataset verdict would have named four
  different winners.

**Evidence quality, ranked.** Yambda `listen50` and KuaiRec `watch_ratio_gt2`
are the two uncontested public slices — eight seeds, no convergence failures, no
design chosen after seeing results — and there the quality-per-millisecond
margin is +1% and +2%. KuaiRand `watch_time` has five arms with seeds worse than
a constant predictor and should carry little weight. VK-LSVD ur uses a feature
view designed on request to have 36% denominator density. The production feed is
the only real-traffic evidence and has one run per arm.
