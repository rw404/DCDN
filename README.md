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

## RQ1 validation checkpoint: DCDN versus Original DCN

The frozen VK-LSVD `C+P` seed-42 validation-only tuning comparison currently
supports the narrow RQ1 hypothesis that `DCDN-Core-L2` improves on the
original rank-one `DCN` cross network under the matched public protocol:

| Model | Selected learning rate | Normalized validation MSE |
| --- | ---: | ---: |
| `DCDN-Core-L2` | `1e-3` | `0.020977282795389355` |
| Original `DCN` | `3e-4` | `0.021035584402689534` |

Lower is better. DCDN reduces normalized validation MSE by
`0.00005830160730017908`, or `0.277157%` relative to the selected DCN result.
Both models used the same `C+P` features, prediction head, loss, tuning budget,
effective batch size, seed, and validation-selection rule. This is a
validation result, not a test result, and it does not establish universal
superiority over all baselines or datasets. The untouched test remains locked
until the complete mandatory selection matrix is frozen and hashed. Exact
selection lineage is preserved in
`results/tuning/vk_lsvd_c_plus_p_dcn_selection_v1.json` and the corresponding
DCDN selection receipt.

This narrow result is not a universal baseline win. Under the same frozen
seed-42 validation protocol, selected AutoInt+ reaches `0.020879945407626305`,
which is lower than selected DCDN by `0.00009733738776305026` (about
`0.464013%`). The exact AutoInt+ selection lineage is preserved in
`results/tuning/vk_lsvd_c_plus_p_autoint_plus_selection_v1.json`. Model-wide
and dataset-wide claims remain pending the complete frozen matrix, untouched
test evaluation, paired uncertainty, and effect-size analysis.

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
