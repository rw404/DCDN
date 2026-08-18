# Controlled interaction-model source audit

Audit date: 2026-08-18 UTC

Scope: architecture equations, tensor contracts, configuration domains,
licenses, and immutable source identity only. No public-dataset row, local test
label, project model metric, or published comparison result was used to design
the implementation.

## Result

The repository now has a clean-room, dataset-agnostic interaction-block API
for the planned public suite:

- `dnn`
- `deepfm`
- `dcn`
- `autoint_plus`
- `dcnv2_full_rank`
- `dcnv2_capacity_matched_core`
- `afn_plus`
- `wukong_small`
- `dcdn_core`

The eighth-model main table uses the ordinary full-rank DCNv2 row; the
low-rank-MoE path is an implementation mechanism for resolving the separate
`DCNv2-capacity-matched-core` control. It is not an extra main-table model.

No upstream source file was copied or vendored. The project code implements
the audited equations behind a project-owned typed boundary and records every
reference in model contract metadata.

## Encoder and prediction boundary

`EncoderSpec` and `EncodedInteractionBatch` deliberately start after feature
encoding. The still-unresolved dataset encoder must construct all views from
one frozen, ordered raw schema:

- `bypass_features [B,C]`: encoded context/static coordinates that bypass the
  DCDN reciprocal core;
- `operand_features [B,P]`: ordered raw or safely shifted base operands;
- `field_embeddings [B,N,E]`: the common field-wise view used by FM,
  attention, AFN, and Wukong blocks;
- `afn_dnn_embeddings [B,N,E]`: AFN+'s second learned embedding-table view of
  the identical ordered fields, never an added feature regime;
- `denominator_validity [B,P]`: the Boolean DCDN validity gate.

The full `C+P` schema and the operand-only `P` schema have distinct SHA-256
fields. DCDN-Core serializes the operand-only hash because the reciprocal core
sees `P`, not the wider flat representation.

Ordinary DNN/DCN/DCNv2 consume flat `C+P`. The capacity control and DCDN-Core
operate only on `P` and concatenate `C` unchanged. This keeps the capacity
comparison on the same algebraic boundary as DCDN rather than comparing a
flat `C+P` CrossNetMix against a `P`-only DCDN core.

Every interaction block emits a declared raw representation. A visible linear
adapter maps it to the shared representation width, then every model uses the
same `CommonPredictionHead`. The frozen experiment contract uses a width-64
representation and a linear one-output task head (`hidden_dims=[]`). Adapter
parameters and FLOPs must be included in model accounting.

## Deterministic initialization

The factory constructs on CPU inside an isolated RNG context and derives
independent interaction, adapter, and head seeds from the recorded base seed.
Consequently, the final common head weights are byte-identical across model
families for the same base seed; adapter initialization uses the same seed and
distribution but necessarily has a model-specific input shape. CUDA RNG state
is not touched during construction.

Initializers are explicit:

- generic ReLU hidden linears: Kaiming-uniform for ReLU, zero bias;
- representation, scalar, attention, gate, and task-output linears:
  Xavier-uniform, zero bias;
- DCN/DCNv2/DCDN product maps: the separately frozen scaled-Kaiming-normal
  residual-near policy, zero bias;
- DCDN division maps: the same declared scaled-Kaiming-normal policy, zero
  bias;
- the Wukong FMB output immediately preceding ReLU: Kaiming-uniform, zero
  bias.

The complete configuration, display label, source provenance, initializer
policy, component seeds, adapter dimensions, head dimensions, and schema
hashes are canonicalized into the model contract SHA-256.

## Pinned references

### FuxiCTR

- repository: `https://github.com/reczoo/FuxiCTR.git`
- commit: `0245987fdc1d43c36164ea496a02324885a8557a`
- license: Apache-2.0
- license SHA-256:
  `ed24b2447e7415e4baeaf4366d12db81e8a21224f3f1a69c08b8e30807b2ebaa`
- shared cross-layer source SHA-256:
  `bd1627cec81db448e7147b06e64e2457a58ee12e4dbec2a7c0c7310a1f7c07d3`

Audited model-source SHA-256 values:

| Architecture | Pinned path | SHA-256 |
|---|---|---|
| DeepFM | `model_zoo/DeepFM/DeepFM_torch/src/DeepFM.py` | `326fc8158d68e6e8bd922050c81ee44dd165f96e073a6e64ade8b3f703725e32` |
| AutoInt+ | `model_zoo/AutoInt/src/AutoInt.py` | `fdde3cc7f0d0b482bdc5d17967cf48f28f5d8b85babe9d71e6d45960c998cd9e` |
| DCN | `model_zoo/DCN/DCN_torch/src/DCN.py` | `f6d4e6a937c5e4cac6055c7b988cd2ed1c7720facff52d4bc9ecaf9e87037371` |
| DCNv2 | `model_zoo/DCNv2/src/DCNv2.py` | `78fc7d1c4f129dcbb750e410e53e075a646cacf5462ebd186ce0cf687418f4a9` |
| AFN+ adaptation reference | `model_zoo/AFN/src/AFN.py` | `cceba7bba9bd60f11f84a43f00f473fdf615dfdc1de46223d46c11d6252e28d2` |
| Wukong adaptation reference | `model_zoo/WuKong/src/WuKong.py` | `f0be1a2024532acc5ab03129abff5c8b289fb2554d21c58e2db9020845d268d2` |

The Fuxi CrossNetMix reference ends with an unconstrained `squeeze()`, which
can drop the batch axis for `B=1`. The clean-room capacity implementation never
uses that operation and has an explicit `[1,P] -> [1,P]` test.

### Official AFN+

- repository: `https://github.com/WeiyuCheng/AFN-AAAI-20.git`
- commit: `25d7bd995bd0a3c71f562b3d332a0628c3b30d28`
- license: MIT
- license SHA-256:
  `b6a2bc1dc211e2286b0af8106deb43ff704d09c9e1131eb62851e76b10e7d160`
- source: `src/AFN.py`
- source SHA-256:
  `e6728d48a41ae683dd3b16b6a1c8cdfa75e15ee1be9ae1abb7fe97708d77979f`

The official implementation uses distinct AFN and DNN embeddings, positive
absolute/clamped logarithmic inputs, and auxiliary BCE losses for the fused
and two branch outputs. FuxiCTR retains two embeddings and a learned ensemble
but trains through the common model loss.

Model scale is not a constructor default. The frozen search manifest records
dual attribution: AFN+ equations and branch/training provenance come from the
official and FuxiCTR sources above, while the fixed
`128/[512]/[512]` controlled geometry comes only from architecture columns
of the audited Wukong paper appendix comparison row. No adjacent result value
is a selection input.

The controlled AFN+ block makes the following choices explicit:

- independent AFN and DNN embedding views of the same fields are required;
- `abs`, a positive `log_floor`, `log`, and `exp` are explicit;
- `log` and `exp` execute in FP32;
- `exp_clip=None` means the source-equation unbounded exponential, with
  non-finite values delegated to the required stop/monitor policy;
- a finite `exp_clip` is permitted only if the immutable search/protocol
  manifest declares that stabilizing deviation;
- the AFN and DNN branch scalars are concatenated, passed through the visible
  common representation adapter, and then the common prediction head;
- training uses one common task loss and no auxiliary branch loss.

The last choice is a controlled-protocol deviation from the official TF1
training objective and must not be described as an exact optimizer/loss
reproduction of that code.

### Wukong

Pinned implementation references:

- FuxiCTR source and hash listed above (Apache-2.0);
- `https://github.com/clabrugere/wukong-recommendation.git`
- commit: `8a0c14ccee50e02d40201d4dc665ee9fcebeb6b9`
- license: MIT
- license SHA-256:
  `fcb8346bda486d130ed8e1eed4d145d07277e0d40a9c7747a0e69878dc619d7d`
- source: `model/pytorch/wukong.py`
- source SHA-256:
  `56a05170af970f1bd238e264757abd8986c6a89c2a435665fc164231be17ff4e`

Official paper record and inspected artifact:

- canonical PMLR record:
  `https://proceedings.mlr.press/v235/zhang24ao.html`
- inspected PDF origin (the PMLR v235 artifact, not arXiv bytes):
  `https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhang24ao/zhang24ao.pdf`
- secondary identifier: `arXiv:2403.02545`
- audited local copy of that PMLR PDF, SHA-256:
  `8f4f539a174d3ff12653694d7e3d16f2c3b0c0651770a97445ac4826dcb5fabd`

The clean-room layer follows the paper equations:

`LN(concat(FMB(X), LCB(X)) + residual(X))`,

with the optimized interaction matrix `X (X^T Y)`, an FMB MLP, linear LCB,
and a learned residual projection when the field count changes. The common
prediction head replaces the source-specific Wukong top MLP in this controlled
suite and is recorded as a protocol adaptation.

`WukongSmallConfig` has no constructor defaults. The frozen search manifest,
not this module, must bind the smallest fully enumerated complete paper row:
two interaction layers, `nL=8`, `nF=8`, `k=24`, and a three-layer width-2048
FMB MLP. A Cartesian-grid-derived one-layer combination is not an explicitly
enumerated complete model and must not be substituted.

### DCDN

- current audited public core: `src/dcdn/core.py`
- SHA-256 at audit:
  `faeb8b3b8d7cb60eec7416e7b881b34963ef1f3664a9be454e9562a4603468c6`
- reference core: `DCDNCore_reference.py`
- reference SHA-256:
  `6b2cdb5db0c248fcd960c42d4ce14873820cc65d382741721214166d5e83d9ac`
- production-hybrid provenance: `DCDNBlock_provided.py`
- production-hybrid SHA-256:
  `d58c31c03096d112a47718a643fc91bc5dd9b5a53cd7941948145b34335ec1bb`

The suite adapter reuses `src/dcdn/core.py` directly. It preserves the fixed
anchor, one semantic denominator mask, operand validity, FP32 reciprocal,
FP16 rejection, and `DCDN-Core-L{L}` label. It does not add the production
FC/BatchNorm/Dropout/concatenation/projector path.

## Test-lock incident and quarantine

While extracting the official Wukong hyperparameter grid, PDF text extraction
returned adjacent published-result columns on the same page. No value from
those columns is copied into this audit, configuration, source metadata, test,
feature decision, or model decision. Those observed values are quarantined
from all protocol and search-space work. Wukong-small is selected mechanically
from architecture columns only, never by quality, parameter, or compute
results. No local dataset row, project test label, DCDN output, or
project-generated metric was accessed; `test_unlocked` remains false.

## Verification

`tests/test_model_suite.py` independently checks:

- finite forward/backward for every path at `B=1` and `B=3`;
- manual DCN, full-rank DCNv2, low-rank-MoE, and DCDN recurrences;
- DCDN fixed-anchor, mask, operand-hash, and context-bypass semantics;
- DeepFM's pairwise FM identity;
- AutoInt+'s mandatory parallel DNN branch;
- AFN+'s `abs`/floor/log/exp treatment and both ensemble branches;
- Wukong FMB, LCB, residual, layer normalization, and `B=1` shape;
- identical common-head state across model families for one seed;
- deterministic component seeds and preservation of global RNG state;
- explicit zero biases, common output shapes, contract hashing, and pinned
  provenance;
- FP16 rejection, BF16 execution, autocast mismatch handling, and invalid
  input/config failures.
