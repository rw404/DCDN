# RQ2 on `watch_time` — verdict: **supported, single-head only**

**The question, unchanged.** At matched or better serving performance, can DCDN
on the reduced feature set replace DCNv2 on the full feature set?

**Verdict.** On the single-head `watch_time` task, **yes on ranking**: DCDN with
132 inputs out-ranks DCNv2 with 244 inputs on X-AUC by **+0.0222**, a paired
interval of **[+0.0122, +0.0325]** that excludes zero, winning on **8 of 8
seeds** — while running at **1.65× the throughput, 0.61× the p95 latency and
1.67× less VRAM** at production batch size.

**Three limits, stated up front because each one bounds the claim:**

1. **It is a secondary result, not a confirmatory one.** `watch_time` was
   selected as the reporting target *after* the results were seen. See §1.
2. **It holds in the single-head setting only.** In the multi-head setting the
   same contrast reverses and DCNv2-FULL is ahead. See §4.
3. **DCDN is not the best model even where it wins.** AFN+ ranks higher on the
   same task. See §5.

| artifact | path |
|---|---|
| single-head `watch_time` | `results/kuairand_multitask/final_wt_v2/` |
| multi-head | `results/kuairand_multitask/final_mt_v2/` |
| batch sweep | `results/kuairand_multitask/batch_sweep_v1/` |
| consolidated matrix | `results/kuairand_multitask/matrix_v1/` |

---

## 1. Why this is secondary and not confirmatory

The frozen `primary_confirmatory_target` for `kuairand_pure` is `long_view`, in
both `configs/targets_multitask_v2.yaml` and
`configs/data/kuairand_pure/targets_rq23_v1.yaml`. Reporting RQ2 on `watch_time`
instead was decided **after** seeing that `long_view` gives no demonstrable
difference and `watch_time` does. That is post-hoc target selection, and no
amount of subsequent rigour converts it into a confirmatory result.

Two partial mitigations, offered as mitigations and not as a defence:

* the intent to use `watch_time` on KuaiRand for consistency with VK-LSVD and
  KuaiRec was expressed **before** any KuaiRand result existed, and is recorded
  in `configs/data/kuairand_pure/targets_rq23_v1.yaml`;
* `watch_time` is the `primary_confirmatory_target` on two of the three real
  datasets already, so it is not an arbitrary pick from five heads.

Neither changes the classification. **Secondary.** A confirmatory version needs
a dataset whose test split is still sealed, with `watch_time` named first.

## 2. The instrument: paired, not unpaired ranges

Earlier reporting called a difference established when the two arms' per-seed
ranges were disjoint. On this contrast that test is wrong, and wrong in a way
worth recording:

```
seed   dcdn_reduced   dcnv2_full
42        0.5405        0.5138
43        0.5425        0.5163
44        0.5414        0.5391
45        0.5352        0.4997
46        0.5317        0.5278
47        0.5439        0.5316
48        0.5485        0.5000
49        0.5423        0.5200
```

DCDN wins **every seed**. The unpaired ranges nonetheless overlap — DCNv2's best
seed (0.5391) exceeds DCDN's worst (0.5317) — so the range test reports "not
established" for a clean 8/8 sweep. Those are *different* seeds; comparing
ranges discards the pairing the shared seed provides.

Runs are therefore compared seed by seed. A percentile bootstrap over the paired
differences gives the interval (20,000 resamples, seed 20260821, deterministic),
and the win count is reported beside it. `scripts/report_kuairand_contrasts.py`
now emits both instruments so the change is visible rather than silent.

## 3. Quality — single-head `watch_time`

C5 = `dcdn_reduced − dcnv2_full`. Positive favours DCDN on the ranking metrics,
negative favours DCDN on the error metrics.

| metric | scope | paired mean | 95% CI | DCDN wins | excludes 0 |
|---|---|---|---|---|---|
| **X-AUC** | frozen (3) | **+0.01838** | [+0.00224, +0.02664] | 3/3 | **yes** |
| **X-AUC** | all (8) | **+0.02221** | [+0.01218, +0.03247] | **8/8** | **yes** |
| user X-GAUC | frozen (3) | +0.00427 | [−0.00773, +0.01322] | 2/3 | no |
| user X-GAUC | all (8) | +0.00553 | [+0.00084, +0.00956] | 7/8 | **yes** |
| normalized MSE | frozen (3) | +0.00097 | [+0.00036, +0.00132] | 0/3 | yes, **against** DCDN |
| normalized MSE | all (8) | −0.00006 | [−0.00696, +0.00492] | 2/8 | no |
| MAE seconds | all (8) | +2.200 | [−1.280, +6.520] | 3/8 | no |

**Ranking: DCDN wins, and the evidence strengthens with more seeds** — the
interval moves away from zero from three seeds to eight, which is the opposite
of what a small-sample artefact does.

**Calibration: DCDN does not win.** On the frozen three seeds DCNv2-FULL is
better on normalized MSE with an interval excluding zero; across eight seeds it
is a dead tie (−0.00006). MAE in seconds never separates.

The honest summary is a **split verdict**: reduced-input DCDN *orders* watch time
better than full-input DCNv2, and does not *predict its magnitude* better.

## 4. It does not hold in the multi-head setting

The same contrast, same metric, same data, the only change being whether
`watch_time` is trained alone or as one of five heads:

| setting | `dcdn_reduced` | `dcnv2_full` | difference | winner |
|---|---|---|---|---|
| single-head | 0.5415 | 0.5231 | **+0.0184** | DCDN |
| multi-head | 0.5246 | 0.5282 | **−0.0036** | DCNv2 |

DCDN's rank among the seven arms on this metric moves from **5th of 7** in the
multi-head setting to **2nd of 7** single-head. **RQ2 on `watch_time` is
therefore a statement about the single-task setting and must be quoted with that
attached.** The multi-head result is reported in
`docs/kuairand_multihead_per_target.md` and is not favourable.

## 5. DCDN is not the best arm on this task

Single-head `watch_time`, X-AUC, frozen three seeds:

| arm | X-AUC |
|---|---|
| `afn_full` | **0.5567** |
| `dcdn_reduced` | 0.5415 |
| `afn_reduced` | 0.5390 |
| `dcnv2_capacity_reduced` | 0.5359 |
| `dcdn_full` | 0.5314 |
| `dcnv2_full` | 0.5231 |
| `dcnv2_reduced` | 0.5148 |

RQ2 asks specifically about DCDN-reduced against DCNv2-full, and that comparison
DCDN wins. It does not ask which architecture is best, and DCDN is not. A
logarithmic transformation layer turns a ratio into a difference of logs, which
on a ratio target is a more direct inductive bias than a reciprocal branch —
consistent with the same finding in `docs/rq23_kuairand_quality.md` §9.

**RQ3 remains unsupported.** On the frozen three seeds `dcdn_reduced` (0.5415)
is ahead of `dcnv2_capacity_reduced` (0.5359), but the paired interval contains
zero (+0.0055, [−0.0069, +0.0259], DCDN ahead on 1 of 3 seeds). Across eight
seeds the two are level and the sign reverses: −0.0005, [−0.0057, +0.0076],
DCDN ahead on 1 of 8. Adding evidence moves this contrast toward zero, which is
the opposite of what happened to C5.

## 6. Serving cost at this dataset's exact widths

Measured on `kuairand_watch_time` geometry — 8 categorical fields, 132 reduced
and 244 full numeric operands, 24 denominators, one regression head. These are
the widths the quality numbers above were trained at, so cost and quality
describe the same models. Random inputs; no KuaiRand row or label is read. DCDN
runs fused, DCNv2 reference, each under its best available implementation.

| batch | DCDN-reduced tput/s | DCNv2-full tput/s | ratio | p95 ratio |
|---:|---:|---:|---:|---:|
| 1 | 1,016 | 1,048 | 0.970 | 1.030 |
| 64 | 61,859 | 63,137 | 0.980 | 1.021 |
| 1,024 | 975,999 | 1,011,428 | 0.965 | 0.982 |
| 4,096 | 3,902,714 | 4,054,841 | 0.962 | 1.036 |
| **8,192** | **7,631,940** | 6,847,106 | **1.115** | 0.924 |
| **16,384** | **12,328,540** | 7,456,705 | **1.653** | **0.609** |

Peak VRAM at B=16384: **352.7 MiB against 587.4 MiB — 1.67× less.**

Two regimes with a crossover between 4,096 and 8,192. Below it the two are at
parity, DCDN trailing by 2–4% — both are launch-bound and the arithmetic
saving cannot show. Above it DCDN's narrower input pays: at 16,384 it delivers
1.65× the throughput at 61% of the p95 latency.

`wall_over_device_p50` is **1.01 at every one of the 15 points**, so wall clock
and CUDA-event device time agree and the measurement is not contaminated — this
matters because the host was not idle and the run was taken with
`--allow-busy-host`. The load-average gate is a proxy; the wall-versus-device
agreement is the actual check, and it passes.

## 7. Verdict, assembled

**Supported, with conditions.** At production batch sizes DCDN on 132 inputs is
strictly cheaper than DCNv2 on 244 — 1.65× throughput, 0.61× p95, 1.67× less
VRAM — and it *orders* `watch_time` better, by a paired margin that excludes
zero and holds on every one of eight seeds.

**Conditions that must travel with the claim:** the target was chosen after
seeing results, so this is secondary; it holds in the single-head setting and
reverses in the multi-head one; DCDN wins on ranking but not on calibration;
and DCDN is not the best available architecture for this target — AFN+ is.

## 8. Reproduction

```bash
source scripts/env.sh

python scripts/run_kuairand_single_target.py --device cuda:0 \
    --target watch_time --annex-seeds --save-predictions \
    --output-dir results/kuairand_multitask/final_wt_v2

python scripts/benchmark_batch_sweep.py --device cuda:0 \
    --geometry kuairand_watch_time --allow-busy-host \
    --output-dir results/kuairand_multitask/batch_sweep_v1

python scripts/report_kuairand_contrasts.py \
    --results results/kuairand_multitask/final_wt_v2/results_v1.json \
    --output-dir results/kuairand_multitask/contrasts_single_target_v1
```
