# VK-LSVD C+P metric-free capacity resolution

This checkpoint resolves the public DCNv2 capacity control before opening any
data row, label, validation metric, test metric, or GPU model execution. The
machine-readable artifact is
`configs/capacity/vk_lsvd_c_plus_p_v1.json`; its adjacent SHA-256 sidecar and
the independent validator make the decision reproducible.

## Naming alias policy

The newest owner instruction uses the user-facing label
`DCNv2-capacity-matched`. Existing frozen repository artifacts use the older
display label `DCNv2-capacity-matched-core` and the stable internal key
`dcnv2_capacity_matched_core`. This artifact keeps the internal key for
compatibility, records the old display label as a historical alias, and uses
`DCNv2-capacity-matched` as the authoritative reporting label. The historical
alias remains immutable provenance inside existing frozen locks; it does not
require shared-lock migration or re-hashing and is not used in reports or the
paper.

## Frozen geometry and accounting

VK C+P has six categorical context fields, each embedded to width 16, so C
bypasses the core as 96 coordinates. The ordered numeric P width is 58, making
the visible adapter input width 154. DCDN-Core-L2, ordinary full-rank DCNv2-L2,
and the low-rank four-expert DCNv2 control all use the same visible 154-to-64
adapter and the same biased 64-to-1 linear head.

The parameter scope is every unique trainable non-embedding parameter in the
joint raw-to-prediction wrapper. The trainable VK encoder owns 4,128 parameters,
but every one is a weight of an `nn.Embedding` module, including the learned
numeric field vectors. It therefore contributes exactly zero under the frozen
non-embedding rule. No hidden non-embedding encoder projection was omitted.
The interaction, visible adapter, and common head are included.

The frozen FLOP convention counts scalar multiplication, addition, reciprocal,
and transcendental operations as one FLOP. Comparisons, masks, concatenation,
memory movement, and embedding lookup cost zero. A biased `I -> O` linear map
costs `2*I*O` FLOPs/example.

| Variant | Parameters | FLOPs/example | Parameter ratio | FLOP ratio | ±5% |
|---|---:|---:|---:|---:|:---:|
| DCDN-Core-L2 target | 23,673 | 47,274 | 1.0000 | 1.0000 | target |
| Ordinary DCNv2-L2 | 57,725 | 115,320 | 2.4384 | 2.4394 | no |
| Capacity rank 12 | 22,853 | 46,750 | 0.9654 | 0.9889 | yes |
| Capacity rank 13 | 23,981 | 49,006 | 1.0130 | 1.0366 | yes |

Ordinary DCNv2 cannot be reused. Both adjacent brackets satisfy the two-sided
±5% rule. Their frozen log-distance values are 0.04639896421317034 for rank 12
and 0.048908970043076774 for rank 13, so the mechanical rule selects rank 12.
The control is therefore two operand-only low-rank-MoE cross layers, rank 12,
four experts, with C concatenated unchanged before the common adapter. Its
learning rate is not selected here; after validation-only DCNv2 tuning it must
inherit the selected learning rate from the same tuning unit.

## Integration projection

`capacity_resolution_projection` exposes the existing compact mechanical
resolver schema for `TrainingHarness`/`validate_capacity_resolution` and is
tested against the frozen search-space validator. This projection is a derived
compatibility view only. Protocols and checkpoints must bind the SHA-256 of the
full JSON artifact bytes, never an ad hoc or separately hashed projection.

Validate from the repository root with:

```bash
source scripts/env.sh
uv run --frozen python scripts/validate_capacity_artifact.py
uv run --frozen pytest -q tests/test_capacity_artifact.py
```
