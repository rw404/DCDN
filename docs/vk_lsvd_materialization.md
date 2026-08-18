# VK-LSVD train/validation materialization

## Status and immutable boundary

The frozen `vk_lsvd_up0.01_ir0.01` train/validation views were materialized
successfully on 2026-08-18. The durable record is
`data/manifests/vk_lsvd_views_v1.yaml`, whose sidecar-verified SHA-256 is
`738d3610fd83a93297d94ccad4c4316dd3ad8b241f8409e84daf5b2322040dbe`.
The regenerable Parquet shards remain outside the repository under
`/home/codex/.cache/dcdn-public/processed/vk_lsvd/views_v1`.

The implementation constructs paths only for history weeks 0--19, train weeks
20--24, validation week 25, and the two metadata tables. It did not construct,
open, decode, schema-inspect, or materialize the sealed test week. It computed
no validation or test model metric. The manifest records these gates as
`false` and contains no test shard.

Before feature work, the command byte-hashed the frozen feature contract
(`380d105c847a73afb4927740e40401124688f895142792309b4fc8267a2a5f2b`),
the source/model/search locks, and every allowed source object. The excluded
item-embedding object and sealed test object were not opened by this
materializer.

## Frozen construction

History aggregates and three implicit-ALS states are fit once on all
28,106,627 rows from weeks 0--19 and then frozen. The iALS states use viewtime,
feedback, and 15-second survival signals; each uses 32 factors, 15 iterations,
regularization 0.01, seed 42, ascending integer ID maps, CSR duplicate summing,
and one thread. No train or validation event updates these states. Consequently,
rows at the same or later event time cannot update one another.

Categorical mappings and absolute-q99 scale-only normalizers are fit solely on
the 7,505,820 train rows from weeks 20--24. Unknown validation categories map to
OOV token 0; missing metadata maps to MISSING token 1. Numerical normalization
divides by the train absolute 0.99 quantile using nearest interpolation, a
minimum scale of `1e-12`, no centering, and no clipping.

The exact feature algebra is:

- `C=6`: age, gender, geo, place, platform, and agent categorical tokens;
- `P0=31`: base operands from metadata, strict history aggregates, and iALS;
- `V=27`: provenance/validity and reciprocal-floor indicators;
- `P=P0+V=58`;
- `E=20`: matched engineered ratios, averages, and iALS cosines.

Every shard is emitted twice, as matched `C+P` and `C+P+E` views. Construction
hashes the shared C/P values and stable row IDs for each pair. Denominator
validity is carried explicitly for all P coordinates; the frozen eligibility
mask activates reciprocal use for exactly 11 semantically valid operands.
Invalid denominators are gated out, eligible negative values are rejected, and
valid values below epsilon 0.01 activate a dedicated floor indicator. The full
9,022,651-row denominator scan found zero non-finite values.

Current-row outcomes never enter the input features. Current-row `timespent`
is retained only as the clipped primary label and its normalized form. The raw
IDs, week, source row index, and stable row ID are audit fields, not model
features. Other current-row outcomes are not decoded for train or validation.

## Reproduction command

Use the frozen environment and keep the BLAS and iALS execution single-threaded:

```bash
env PYTHONPATH=src \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 \
  WANDB_MODE=disabled \
  /tmp/dcdn-vk-materialize-venv/bin/python \
  scripts/data/vk_lsvd_materialize.py \
  --repository-root /workspace/codex-home/dcdn/deep_division_network \
  --raw-root /home/codex/.cache/dcdn-public/raw/vk_lsvd \
  --output-root /home/codex/.cache/dcdn-public/processed/vk_lsvd \
  --contract configs/data/vk_lsvd/features_v1.yaml \
  --manifest data/manifests/vk_lsvd_views_v1.yaml \
  --ledger docs/cost_ledger.md \
  --eta-hours 2
```

The command fails closed if a lock or source hash drifts, an output already
exists, ETA exceeds two hours, the local cost/time envelope is crossed, fewer
than 20 GiB remain, overlay use reaches 85%, metadata keys are non-unique,
required values are missing, row order changes, a scale or feature is
non-finite, an active denominator is non-positive, or a written shard cannot be
reopened with its exact schema and row count.

## Verified outputs

| Role | Weeks | Rows | Regimes |
|---|---:|---:|---|
| Train | 20--24 | 7,505,820 | `C+P`, `C+P+E` |
| Validation | 25 | 1,516,831 | `C+P`, `C+P+E` |
| Total | 20--25 | 9,022,651 | matched pairs |

There are 12 Parquet files totaling 2,440,747,697 bytes. Each file's path,
byte count, SHA-256, row count, row-group count, paired shared-value hash, and
stable-row-ID hash is recorded in the manifest. Independent post-run
verification rehashed all 12 files and reopened only their Parquet
schema/metadata; all checks passed. The final overlay snapshot had
77,332,049,920 bytes free (28% used), above the 20-GiB and 85% guards.

The recorded state fingerprints are:

- history state:
  `a0fa09f6cb4e0531b2a1658c09aed3e4215953825ff1d64fa087dfabd4575873`;
- metadata lookup:
  `3950a293a8398904efbf3fee5614ede722f5a76d33bb89f8a811d3f1c7626e88`;
- materialization source:
  `fa2255f64528d6481a77ea5b4e1280e0ed34aa6ffcbf9a8700425b3649966164`.

The physical schema hashes are
`0b334e587d2028e1e39760b735a71e712af5e75ec4fd204cbc76a5efafe12f68`
for `C+P` and
`91061695b06f2a1f4c8743f4ba4c0fa66c14cd7c0dd09d15e6c40d7262b6197b`
for `C+P+E`. The train-only categorical cardinalities, including OOV and
MISSING, are age 55, gender 4, geo 82, place 24, platform 13, and agent 22.

## Consumer rules

Training code must select fields by the ordered feature lists and schema hashes
in the manifest, not by broad dtype selection. It must keep the two regimes on
the same rows and common head/loss protocol. `user_id`, `item_id`, week/index,
stable row ID, and labels are forbidden as inputs. Denominator validity columns
are gating metadata; categorical and other ineligible coordinates bypass the
reciprocal branch. Validation may be used only for the frozen seed-42 selection
protocol. This artifact does not authorize test access.
