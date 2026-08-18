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
