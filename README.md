# DCDN research repository

Reproducible public-offline experiment workspace for:

**DCDN: Deep Cross-Division Network for Product-and-Ratio Feature Interactions
in Industrial Ranking** (WSDM 2027 target).

## Architecture boundary

- `DCDN-Core-L{L}` is the public experiment and Laurent-theory model. It uses a
  fixed raw/base-operand anchor and only residual, product-cross, and masked
  reciprocal-cross branches. Shared context encoding and the prediction head
  remain outside the core.
- `DCDN-Hybrid-Prod-L1-PLE8` is the deployed one-block integration preserved in
  `DCDNBlock_provided.py`. It operates on `[B, 8, F]` PLE coordinates and
  includes the local FC/BatchNorm/Dropout path and projector. It is provenance,
  not the architecture used in the main public tables.

PLE is not assumed to preserve raw multiplication or division. Exact Laurent
claims apply to exact reciprocal paths on their valid nonzero domains; the
stabilized production hybrid is generally piecewise rational in raw space.

## Frozen public scope

- Core datasets: KuaiRand-Pure and VK-LSVD `up0.01_ir0.01`.
- Mechanistic ablation: KuaiRec big matrix.
- Optional only after mandatory completion: Yambda-50M.
- Test evaluation is locked until dataset features, metrics, search spaces,
  selected configurations, and `protocol_frozen.yaml` are frozen and hashed.

The authoritative execution rules are `AGENTS.md`, `DCDN_HANDOFF.md`, and
`REMOTE_AGENT_PROMPT.md`. See `RUNBOOK.md` and `STATUS.md` for execution state.

## Reproduce the deterministic preflight

```bash
source scripts/env.sh
uv sync --frozen --extra data
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
python scripts/preflight.py --require-data-extra --eta-hours 1
python -m pytest -q
ruff format --check src tests scripts
ruff check src tests scripts
```

The environment and public-data caches are ephemeral; `uv.lock`, source,
manifests, compact results, and the run registry remain on persistent storage.
The exact public experiment matrix is
`configs/experiment_matrix_v1.yaml`. Dataset-specific protocol files are not
valid until their unresolved hashes are replaced, sidecars are written, and
the fail-closed gate accepts them.

## RQ1 validation checkpoint: seeds 42, 43, and 44

The frozen VK-LSVD `C+P` validation-only comparison now covers all three
predeclared seeds. Lower normalized MSE is better:

| Model | Seed 42 | Seed 43 | Seed 44 | Mean |
| --- | ---: | ---: | ---: | ---: |
| AFN+ | `0.020806346` | `0.020784721` | `0.020775092` | `0.020788720` |
| DeepFM | `0.020845575` | `0.020837533` | `0.020856263` | `0.020846457` |
| Wukong-small | `0.020962457` | `0.020833218` | `0.020838377` | `0.020878017` |
| AutoInt+ | `0.020879945` | `0.020891923` | `0.020930326` | `0.020900731` |
| DCNv2 full rank | `0.020908477` | `0.020909154` | `0.020946859` | `0.020921497` |
| DNN | `0.020899831` | `0.020948966` | `0.020917478` | `0.020922092` |
| `DCDN-Core-L2` | `0.020977283` | `0.021000733` | `0.021012709` | `0.020996908` |
| `DCNv2-capacity-matched` | `0.021004646` | `0.021007386` | `0.021065178` | `0.021025737` |
| Original DCN | `0.021035584` | `0.021028514` | `0.021051430` | `0.021038509` |

The narrow RQ1 hypothesis is supported on validation: DCDN is lower than
Original DCN on every seed. Its paired mean improvement is
`0.000041601361759013`, or `0.197739%` relative to the DCN mean. DCDN also
beats the rank-12 capacity-matched control on every seed, with a paired mean
improvement of `0.000028828551562561` (`0.137111%`). These comparisons isolate
the predictive contribution of the reciprocal carrier more directly than a
claim based only on parameter count.

The broader claim that DCDN is the best architecture is not supported by this
validation table. Full-rank DCNv2 is lower than DCDN on all three seeds, and
AFN+, DeepFM, Wukong-small, AutoInt+, and DNN also have lower three-seed means.
The intended paper claim is therefore the narrower matched comparison:
DCDN recovers useful product-and-ratio interactions beyond Original DCN and
its capacity-matched control, not universal superiority over every baseline.

All entries are validation-only model-selection evidence under matched
features, prediction head, loss, example budget, effective batch, and seeds.
One seed-43 capacity-control attempt ended in a documented infrastructure I/O
failure before any validation metric and is excluded; its fresh rerun
completed normally. No test metric has been opened. The untouched test remains
locked until the complete selected-configuration lineage and audited one-pass
evaluation bundle are frozen.

Exact full-precision values, per-seed source SHA-256 bindings, paired
comparisons, and the test firewall are preserved in
`results/final_validation/vk_lsvd_c_plus_p_rq1_validation_summary_v1.json`.

## Frozen source and model checkpoint

The three approved public sources are byte-verified outside the repository.
Their source-only manifests keep test access locked:

- KuaiRand-Pure: `bc5b441fdd3188d033e0ecbf0bdfd73f8bcf07bbc744d474d260ee220ae4d6e9`;
- KuaiRec: `963019bcb4c8dbd4663f23c24a472fe0332f6494fdcbbaa74ca158a73488c5b4`;
- VK-LSVD: `5dbbfeb23442a6b1ebdb829dc239c6a3c6c655d71446cde4551b83cc6af456ac`.

Model provenance is frozen in `configs/model_sources_v1.yaml` and the
metric-independent two-candidate search policy is frozen in
`configs/search_space_v1.yaml`. Their SHA-256 digests are respectively
`0844a838e5a68029299a52aaa31b5f74205fe57fd486b51544061b32c571a3f4`
and `b263b55da768ec33656ebdb1506d58f2809fd5826aa8b4976aedaecee53bb344`.
No validation or test metric was used to define either lock.
