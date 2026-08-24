# Production NR feed — validation quality of four arms

Four training runs on the production NR feed, five multi-task heads, one run per
arm. `REDUCED` withholds **every manually engineered ratio feature**; `FULL` keeps
them. This is the same contrast the offline studies in this repository run, on
production traffic: the numerators and denominators remain, only the quotients
are removed.

| run name in the logs | arm in this paper |
|---|---|
| `NR feed reduced` | DCNv2-FULL |
| `NR feed reduced \| no days_7` | DCNv2-REDUCED |
| `NR feed reduced \| DCDN` | DCDN-FULL |
| `NR feed reduced \| no days_7 \| DCDN` | DCDN-REDUCED |

The run tag `no days_7` is historical. It marks the removal of **all** manually
engineered ratio features, not of a single history block; the name is kept here
so the rows can be matched against the logs, and the semantics are stated rather
than inferred from it.

120 validation points per curve, logged every 1,545 steps to 185,457. The
summary statistic is the **mean of the last ten points** (steps 171k–185k),
which is where every curve has settled.

**One run per arm, so there is no seed variance.** The bracketed figure below is
the spread of the ten points the mean is taken over — it bounds logging noise,
not initialisation noise, and a difference smaller than it should not be read as
a difference. Every offline study in this repository uses eight seeds; these
production runs do not, and the tables are quoted accordingly.

## Table 1 — validation AUC, tail-10 mean

| arm | input | score_like | score_fullscreen | score_viewtime | score_share | score_dislike |
|---|---|---|---|---|---|---|
| DCNv2-FULL | FULL | 0.7335 | 0.6255 | 0.5592 | 0.5824 | 0.3654 |
| DCNv2-REDUCED | REDUCED | 0.7184 | 0.6092 | 0.5151 | 0.5675 | 0.3713 |
| DCDN-FULL | FULL | **0.7388** | **0.6309** | **0.5703** | **0.5975** | **0.3644** |
| DCDN-REDUCED | REDUCED | 0.7329 | 0.6263 | 0.5509 | 0.5813 | 0.3729 |


Bold marks the best arm per column. `score_dislike` is **lower-is-better**;
all other heads are higher-is-better.

### Spread of the ten averaged points

| arm | score_like | score_fullscreen | score_viewtime | score_share | score_dislike |
|---|---|---|---|---|---|
| DCNv2-FULL | ±0.0010 | ±0.0014 | ±0.0023 | ±0.0040 | ±0.0009 |
| DCNv2-REDUCED | ±0.0020 | ±0.0018 | ±0.0023 | ±0.0023 | ±0.0014 |
| DCDN-FULL | ±0.0006 | ±0.0011 | ±0.0022 | ±0.0035 | ±0.0017 |
| DCDN-REDUCED | ±0.0010 | ±0.0014 | ±0.0032 | ±0.0012 | ±0.0011 |

## Table 2 — contrasts

| contrast | score_like | score_fullscreen | score_viewtime | score_share | score_dislike |
|---|---|---|---|---|---|
| DCDN-FULL − DCNv2-FULL | **+0.0053** | **+0.0054** | **+0.0111** | **+0.0151** | **+0.0010** |
| DCDN-RED − DCNv2-RED | **+0.0145** | **+0.0171** | **+0.0359** | **+0.0138** | -0.0016 |
| DCDN-RED − DCNv2-FULL | -0.0007 | **+0.0008** | -0.0083 | -0.0011 | -0.0075 |
| cost of removing the ratios, DCNv2 | -0.0152 | -0.0163 | -0.0442 | -0.0149 | -0.0059 |
| cost of removing the ratios, DCDN | -0.0059 | -0.0045 | -0.0194 | -0.0162 | -0.0085 |

All contrasts are sign-corrected so **positive means better**, including
`score_dislike`, where the raw metric is lower-is-better.

## Figure 1 — validation quality during training

`figures/production_val_auc.svg` — five panels, one per head, four curves each.
Steps 30k–185k, after the warm-up transient. Faint line is the logged value,
solid line an EMA with α = 0.25; solid strokes are FULL input, dashed are
REDUCED — that is, every engineered ratio withheld. Panels have independent y ranges because the heads differ by 0.2 AUC in
level, and a shared axis would compress every separation to invisibility — the
legend states this.

## What the numbers say

**DCDN-FULL wins all five heads.** Against DCNv2-FULL: +0.0053 like, +0.0054
fullscreen, +0.0111 viewtime, +0.0151 share, +0.0010 dislike. Four of the five
margins are several times the logging spread; dislike (+0.0010 against ±0.0009
and ±0.0017) is not.

**DCDN-REDUCED beats DCNv2-REDUCED on four of five**, and the margins are large:
+0.0359 viewtime, +0.0171 fullscreen, +0.0145 like, +0.0138 share. On dislike it
is behind by 0.0016, which is within the spread of both runs.

**Removing the ratio features costs the two architectures very differently.** For
DCNv2 it costs −0.0442 viewtime, −0.0163 fullscreen and −0.0152 like; for DCDN the same
removal costs −0.0194, −0.0045 and −0.0059. DCDN absorbs 56–72% of the loss on
those three heads. That is the production form of the RQ2 claim and it is the
strongest evidence in this file.

**Consequently DCDN-REDUCED lands level with DCNv2-FULL on three heads** —
fullscreen +0.0008, like −0.0007, share −0.0011, all inside the logging spread.
A model that never sees a single engineered quotient matches the full-feature
baseline on those three heads, which is the deployable form of the argument and
the closest thing in the programme to RQ2 confirmed on real traffic.

### Where the claim does not hold, stated plainly

**On `score_dislike`, DCDN-REDUCED is the worst of the four arms** — 0.3729
against 0.3654 for DCNv2-FULL, 0.3713 for DCNv2-REDUCED and 0.3644 for
DCDN-FULL. Lower is better on this head, so it is last by 0.0075 against the
full-feature baseline and last even among the two reduced arms.

Removing the ratios costs DCDN **more** than it costs DCNv2 on this head:
−0.0085 against −0.0059. Dislike and share are the two heads where DCDN does not absorb
the removal (−42% and −9% recovery, against +56% to +72% on the other three).

The honest summary is therefore: **DCDN-REDUCED replaces the engineered ratios on
like, fullscreen and share, and does not replace them on dislike or viewtime.** Claiming a dislike improvement would be contradicted by the first
table a reader checks.

### Two things to note before quoting any of this

* **Single run per arm.** Differences under ~0.002 are not separable from run-to-
  run variation, which is unmeasured here. That covers the dislike contrast
  entirely and the DCDN-RED vs DCNv2-FULL contrasts on like, fullscreen and
  share. The large ones — viewtime, and the reduced-vs-reduced column — are
  well clear of it.
* **share and viewtime peak early and decline.** Both reach their best value
  before step 30k and fall through training: share from 0.65 to 0.58, viewtime
  from 0.63 to 0.56 on DCNv2-FULL. That is multi-task interference, not
  convergence, and the "best" column of a per-metric table would report a
  checkpoint no one would ship. Tail-10 is the right statistic and the figure
  shows why.
