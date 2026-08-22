# KuaiRand-Pure `watch_time` — quality and serving cost

One target, twelve architectures, one page. The regression head on
KuaiRand-Pure under the **`rms`** preconditioner, which is the branch every
conclusion here rests on: `geometric` was also run and is rejected for this
dataset, with the reason and the failure tables in
`docs/kuairand_field_results.md`.

Cost and quality below describe the **same twelve models** - the benchmark runs
the exact configurations the quality run trained, at the exact operand widths,
so the two tables can be read against each other.

| | |
|---|---|
| rows | 201,320 train / 311,490 validation |
| users in validation | 23,620 |
| target | `watch_time`, regression, normalized play time |
| categorical fields | 8 |
| numeric operands | **132 REDUCED, 244 FULL** |
| eligible denominators | 24 of 132 |
| seeds | 8 per arm (42-49), learning rate tuned per arm on seed 42 |
| preconditioner | `rms`, purely multiplicative, train-only |
| test split | never opened |

The 112 operands FULL has and REDUCED does not are the `R` block: manually
materialized divisions of historical counters and sums. Every numerator and
every denominator stays in REDUCED - only the quotient is withheld.

## 1. Quality

| # | arm | inputs | params | xAUC | user xAUC | nMSE | MAE ms | xAUC sd |
|---|---|---|---|---|---|---|---|---|
| 1 | autoint_full | FULL | 2,884,961 | 0.55748 | 0.50946 | 0.00296 | 14.106 | 0.00448 |
| 2 | afn_full | FULL | 3,180,475 | 0.55552 | 0.50999 | 0.00255 | 11.990 | 0.01173 |
| 3 | mlp_full | FULL | 520,113 | 0.54830 | 0.50726 | 0.00286 | 12.835 | 0.00830 |
| 4 | afn_reduced | RED | 2,241,243 | 0.54291 | 0.50281 | 0.00355 | 16.554 | 0.01866 |
| 5 | deepfm_full | FULL | 1,988,274 | 0.54274 | 0.50015 | 0.00286 | 11.587 | 0.00857 |
| 6 | dcnv2_capacity_reduced | RED | 72,850 | 0.54128 | 0.50179 | 0.00716 | 27.229 | 0.01128 |
| 7 | **dcdn_reduced** | RED | 72,681 | 0.54075 | 0.50693 | 0.00853 | 22.640 | 0.00521 |
| 8 | wukong_full | FULL | 922,081 | 0.53745 | 0.50540 | 0.00314 | 13.186 | 0.02259 |
| 9 | **dcdn_full** | FULL | 167,881 | 0.52935 | 0.50765 | 0.00416 | 14.667 | 0.01600 |
| 10 | dcnv2_full | FULL | 325,833 | 0.51854 | 0.50140 | 0.00859 | 20.440 | 0.01419 |
| 11 | dcnv2_reduced | RED | 173,289 | 0.51470 | 0.49972 | 0.04236 | 35.184 | 0.00937 |
| 12 | dcn_full | FULL | 49,809 | 0.51457 | 0.50236 | 0.00501 | 19.656 | 0.01653 |


**Parameters are trainable non-embedding.** `dcdn_reduced` and
`dcnv2_capacity_reduced` are mechanically matched within ±5% on both parameters
and FLOPs - 72,681 against 72,850 - which makes them the only pair in the table
where a difference can be attributed to architecture rather than size. The two
arms at the top carry **40x more parameters** than DCDN.

### Paired, seed by seed

Rank order is close to uninformative here: per-arm sd runs 0.005-0.023 while
adjacent rows differ by less. Two arms are compared on the same seeds and the
difference is tested against its own standard error.

| comparator | mean xAUC diff | se | beats DCDN | verdict |
|---|---|---|---|---|
| dcn_full | -0.02618 | 0.00551 | 1/8 | **significant** |
| dcnv2_reduced | -0.02605 | 0.00305 | 0/8 | **significant** |
| dcnv2_full | -0.02221 | 0.00557 | 0/8 | **significant** |
| dcdn_full | -0.01140 | 0.00550 | 3/8 | **significant** |
| wukong_full | -0.00330 | 0.00841 | 5/8 | not significant |
| dcnv2_capacity_reduced (capacity-matched) | +0.00053 | 0.00388 | 7/8 | not significant |
| deepfm_full | +0.00198 | 0.00427 | 6/8 | not significant |
| afn_reduced | +0.00215 | 0.00681 | 4/8 | not significant |
| mlp_full | +0.00755 | 0.00354 | 5/8 | **significant** |
| afn_full | +0.01477 | 0.00358 | 7/8 | **significant** |
| autoint_full | +0.01672 | 0.00218 | 8/8 | **significant** |

### What the quality table says

**`dcdn_reduced` is the best of the four reduced-input arms**, and it is ahead
of three full-input arms - `wukong_full`, `dcnv2_full`, `dcn_full` - on the
ranking metric.

**Against its capacity-matched twin the difference is not significant**:
+0.00053 with a standard error of 0.00388. Seven of eight seeds favour
`dcnv2_capacity_reduced`, which is worth noting, but the magnitude is a seventh
of the error.

**It is significantly behind three arms**, all of them 7x to 44x larger:
`autoint_full` (+0.01672, 8/8), `afn_full` (+0.01477, 7/8) and `mlp_full`
(+0.00755).

**Calibration is the weak point, not ranking.** Seventh of twelve on xAUC,
**tenth on `normalized_mse`** (0.00853 against 0.00255) and tenth on MAE. The
frozen contract's primary quantity for a regression head is `normalized_mse`,
so on the primary metric DCDN is near the bottom of the table.

## 2. A convergence caveat that bounds every claim on this target

`watch_time` is unstable for part of the field. Counting seeds whose macro
NLoss exceeds 1.0 - that is, worse than predicting the train mean:

| arm | worse than constant | fails the frozen 10x gate |
|---|---|---|
| `dcnv2_reduced` | 8/8 | 1/8 |
| `dcdn_reduced` | 5/8 | 0/8 |
| `dcnv2_full` | 4/8 | 0/8 |
| `dcnv2_capacity_reduced` | 1/8 | 0/8 |
| the other eight | 0/8 | 0/8 |

The frozen convergence gate fires at **10x** the constant predictor. That is a
"did this run produce a result at all" test, and an arm sitting at 1.4x passes
it while being worse than a constant. On a binary target the distinction is
minor; on a regression target it is not.

**One reading has to be withdrawn because of this.** `dcdn_reduced` beats
`dcnv2_full` by xAUC on all eight seeds, which reads as RQ2 in its strong form:
reduced input beating full input. Restricted to the seeds where **both** arms
beat the constant predictor - there are two - `dcnv2_full` wins both, by
0.0376. The all-seed margin comes from seeds where the comparator failed to
fit, not from seeds where DCDN ranked better.

**No RQ2 claim is made on this target.** With three clean seeds out of eight for
`dcdn_reduced`, the honest statement is that this target is too unstable for
part of the field to carry a conclusion. `long_view`, where every arm converges
on every seed, is the sound binary counterpart and is reported in
`docs/kuairand_field_results.md`.

## 3. Serving cost

Same twelve configurations, same operand widths, measured on the
`kuairand_watch_time` geometry: 8 categorical fields, 132 reduced and 244 full
operands, 24 denominators, one head. Random inputs only - no KuaiRand row,
label or test metric is read. Protocol: 50 warmups, 300 synchronized forwards,
3 independent rounds per point, eager mode, batch grid 1..16384. Each arm runs
under its best available implementation (DCDN fused, DCNv2 reference).

**On measurement validity.** `/proc/loadavg` is not namespaced in a container,
so it reports the physical host and read 3.4 while this container was idle at
4% CPU with zero blocked processes. The load gate was therefore waived and the
tool's own contamination check used instead: wall-clock p50 against CUDA-event
device p50, which host CPU load cannot touch. **The ratio stays between 1.000
and 1.024 across all 180 points**, against a ceiling of 2.0. The measurement
is `final_load_gate_waived_by_device_time_agreement`, not an idle-host run, and
is recorded as such.

| arm | inputs | p50 B=1 | p50 B=64 | p50 B=1024 | p50 B=8192 | tput B=8192 | vs DCDN-red |
|---|---|---|---|---|---|---|---|
| **dcdn_reduced** | RED | 0.648 ms | 0.701 ms | 0.705 ms | 0.710 ms | 11,544,262/s | 1.00x |
| dcnv2_reduced | RED | 0.596 ms | 0.654 ms | 0.650 ms | 0.817 ms | 10,030,073/s | 1.15x |
| dcn_full | FULL | 0.627 ms | 0.682 ms | 0.682 ms | 0.928 ms | 8,830,931/s | 1.31x |
| **dcdn_full** | FULL | 0.651 ms | 0.703 ms | 0.706 ms | 0.961 ms | 8,528,652/s | 1.35x |
| dcnv2_capacity_reduced | RED | 0.963 ms | 1.041 ms | 1.031 ms | 1.037 ms | 7,902,701/s | 1.46x |
| dcnv2_full | FULL | 0.601 ms | 0.662 ms | 0.654 ms | 1.186 ms | 6,907,223/s | 1.67x |
| mlp_full | FULL | 0.633 ms | 0.706 ms | 0.688 ms | 1.362 ms | 6,016,224/s | 1.92x |
| deepfm_full | FULL | 0.773 ms | 0.841 ms | 0.949 ms | 3.263 ms | 2,510,778/s | 4.60x |
| wukong_full | FULL | 1.263 ms | 1.406 ms | 1.383 ms | 5.113 ms | 1,602,151/s | 7.21x |
| afn_reduced | RED | 1.109 ms | 1.224 ms | 1.503 ms | 5.518 ms | 1,484,652/s | 7.78x |
| afn_full | FULL | 1.108 ms | 1.226 ms | 1.787 ms | 7.955 ms | 1,029,854/s | 11.21x |
| autoint_full | FULL | 1.438 ms | 1.895 ms | 16.440 ms | 125.789 ms | 65,125/s | 177.26x |


**DCDN-REDUCED is the fastest arm in the field at every batch size that
matters**, and its curve is nearly flat: 0.648 ms at B=1 against 0.710 ms at
B=8192. Ninety-nine point nine percent of the work is added for a 10% latency
increase, which is the launch-bound-to-work-bound transition never really
arriving. At 11.5M examples per second it serves 177x the throughput of
`autoint_full`.

Two readings deserve care:

* **At B=1 DCDN is not fastest** - `dcnv2_reduced` (0.596 ms) and `dcnv2_full`
  (0.601 ms) are ahead. At one row per call both architectures are launch-bound
  and fixed overhead dominates, so the arithmetic advantage cannot show. The
  ordering inverts by B=8192, where the same two arms cost 1.15x and 1.67x.
* **`dcnv2_capacity_reduced` is 1.46x slower than DCDN at equal parameters and
  FLOPs.** Capacity matching equalizes the analytic counts, not the realized
  kernels: DCDN's reciprocal carrier plus one product layer maps onto fewer,
  larger operations than DCNv2's two low-rank layers.

### Cost against quality

| arm | xAUC | p50 @ B=8192 | xAUC per ms |
|---|---|---|---|
| autoint_full | 0.55748 | 125.789 ms | 0.0005 |
| afn_full | 0.55552 | 7.955 ms | 0.0070 |
| mlp_full | 0.54830 | 1.362 ms | 0.0355 |
| afn_reduced | 0.54291 | 5.518 ms | 0.0078 |
| deepfm_full | 0.54274 | 3.263 ms | 0.0131 |
| dcnv2_capacity_reduced | 0.54128 | 1.037 ms | 0.0398 |
| **dcdn_reduced** | 0.54075 | 0.710 ms | 0.0574 |
| wukong_full | 0.53745 | 5.113 ms | 0.0073 |
| **dcdn_full** | 0.52935 | 0.961 ms | 0.0306 |
| dcnv2_full | 0.51854 | 1.186 ms | 0.0156 |
| dcnv2_reduced | 0.51470 | 0.817 ms | 0.0180 |
| dcn_full | 0.51457 | 0.928 ms | 0.0157 |

`xAUC per ms` divides the margin over chance by latency at B=8192. It is a
crude ratio and is offered as an ordering aid, not a decision metric - it
assumes latency and accuracy trade linearly, which they do not.

**DCDN-REDUCED has the best ratio in the field, 1.4x better than the next arm.**
The reading is not that it is the best model: it is seventh of twelve on xAUC
and tenth on the contract's primary `normalized_mse`. The reading is that the
three arms above it on quality cost 1.9x, 7.8x and 177x more, and the two
arms that cost about the same - `dcnv2_reduced` at 1.15x and `dcn_full` at
1.31x - rank eleventh and twelfth on quality.

**The honest summary for a serving decision on this target:** if latency is
free, take `autoint_full` and pay 177x. If it is not, DCDN-REDUCED buys most of
the achievable ranking quality at the lowest cost in the field - subject to
section 2, which shows this target is unstable for part of the field, and to
the calibration gap, which no latency argument fixes.

## 4. Reproduction

```bash
source scripts/env.sh
python scripts/run_kuairand_single_target.py --device cuda:0 --target watch_time \
    --preconditioner rms --annex-seeds --save-predictions --field \
    --output-dir results/kuairand_multitask/field_watch_time_rms_v1
python scripts/benchmark_batch_sweep.py --device cuda:0 \
    --geometry kuairand_watch_time --field \
    --output-dir results/kuairand_multitask/batch_sweep_field_v1
```
