# KuaiRec — serving cost across the batch range

Measured at the exact widths the KuaiRec study trained at: **5 categorical
fields, 50 reduced against 96 full numeric operands, 5 denominators, one head**.
The geometry is marked `quality_source_compatible`, so the cost numbers here and
the quality numbers in `docs/kuairec_results.md` describe the same models rather
than being joined across geometries.

Random inputs only. No KuaiRec row, label or statistic is read; nothing is
trained. DCDN runs fused, DCNv2 reference — verified to be DCNv2's best
available implementation, not a handicap (`docs/rq2_fused_always_faster.md` §1).

**Result: DCDN-reduced is faster than DCNv2-full at every batch size, in both
modes** — worst case **1.003×** eager and **1.083×** with CUDA graphs, best case
**1.66×**, at **1.59× less VRAM**.

| artifact | path |
|---|---|
| eager sweep | `results/kuairec_watch_time/batch_sweep_v1/` |
| CUDA-graph sweep | `results/kuairec_watch_time/batch_sweep_graph_v1/` |

---

## 1. The measurement

Ratio is DCDN-reduced over DCNv2-full; above 1 means DCDN is faster. Graph
columns capture one CUDA graph per `(arm, batch)` and time the replay, applied
identically to both arms.

| batch | eager tput | graph tput | graph p95 | DCDN graph ms | DCNv2 graph ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.005 | 1.085 | 1.084 | 0.0570 | 0.0619 |
| 2 | 1.008 | 1.107 | 1.104 | 0.0652 | 0.0722 |
| 4 | 1.005 | 1.112 | 1.114 | 0.0741 | 0.0824 |
| 8 | 1.007 | 1.102 | 1.101 | 0.0913 | 0.1007 |
| 16 | 1.008 | **1.083** | 1.083 | 0.1219 | 0.1319 |
| 32 | 1.008 | 1.186 | 1.184 | 0.0823 | 0.0976 |
| 64 | **1.003** | 1.175 | 1.176 | 0.0922 | 0.1083 |
| 128 | 1.004 | 1.178 | 1.202 | 0.0947 | 0.1116 |
| 256 | 1.006 | 1.201 | 1.196 | 0.0939 | 0.1128 |
| 512 | 1.010 | 1.224 | 1.225 | 0.0994 | 0.1217 |
| 1,024 | 1.004 | 1.466 | 1.478 | 0.1157 | 0.1696 |
| 2,048 | 1.011 | 1.313 | 1.281 | 0.1715 | 0.2251 |
| 4,096 | 1.006 | 1.440 | 1.422 | 0.2389 | 0.3440 |
| 8,192 | 1.005 | 1.547 | 1.550 | 0.3743 | 0.5791 |
| 16,384 | **1.101** | **1.659** | 1.557 | 0.6153 | 1.0207 |

Peak VRAM at B=16,384: **161.3 MiB against 256.2 MiB — 1.59× less.**

## 2. Reading it

**Eager is essentially parity below B=16,384.** The margin is 0.3–1.1%,
consistent in sign at all fifteen points but far too small to be a serving
argument on its own. The whole model is dispatch-bound there: at B=1 a forward
takes about 0.9 ms while the arithmetic difference between 50 and 96 operands is
negligible against per-kernel launch overhead. Only at B=16,384 does eager show
a real gap, 1.101×.

**CUDA graphs are what make the difference visible.** Removing dispatch overhead
collapses the B=1 forward to 0.057 ms and lets the arithmetic show at every
batch size: the ratio rises from 1.085 at B=1 to 1.659 at B=16,384. Graphs are
applied identically to both arms — this is not an optimization handed to one
side; DCDN gains slightly more because it issues more kernels per forward, which
is exactly the cost graphs remove.

**The gap is smaller than on KuaiRand.** There, reduced-versus-full is 132
against 244 operands and the graph ratio reaches 2.06×. Here it is 50 against
96, and the ratio reaches 1.66×. Both are close to a 1.85–1.9× operand ratio at
the top of the range, which is the expected behaviour once the work-bound regime
is reached: the advantage tracks how much less arithmetic the reduced view
implies, not anything specific to either dataset.

## 3. Limits

* **Graphs require static shapes** — one captured graph per batch size. A
  serving stack with truly dynamic shapes cannot replay a graph and sees the
  eager column, where the advantage is under 1.1% below B=16,384.
* **`wall_over_device_p50` reaches 1.22 at B=1 under graphs** (against 1.01–1.02
  eager) because a 0.057 ms replay makes fixed measurement overhead a large
  fraction of the total. The device-time column is the more reliable one at
  small batch under graphs; it tells the same story. Every eager point is at
  1.01–1.02, so the eager numbers are uncontaminated despite the host not being
  idle.
* **This is cost, not quality.** On `watch_time` the KuaiRec dataset cannot
  separate architectures at all, and on `watch_ratio_gt2` the quality margins
  are small relative to seed noise. A cost advantage only matters once a quality
  claim stands, and on this dataset that claim is weak —
  `docs/kuairec_results.md` §3.3.

## 4. Reproduction

```bash
source scripts/env.sh

python scripts/benchmark_batch_sweep.py --device cuda:0 \
    --geometry kuairec_watch_time --allow-busy-host \
    --output-dir results/kuairec_watch_time/batch_sweep_v1

python scripts/benchmark_batch_sweep.py --device cuda:0 \
    --geometry kuairec_watch_time --allow-busy-host --cuda-graphs \
    --output-dir results/kuairec_watch_time/batch_sweep_graph_v1
```
