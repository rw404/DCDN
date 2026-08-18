# VK-LSVD feature and split audit

Audit date: 2026-08-18 UTC. This record covers only the pre-metric phase-1
contract for `deepvk/VK-LSVD`, subset `up0.01_ir0.01`. No validation model
metric, test label, test prediction, or test metric was read or computed.

## Source and inspection boundary

The raw source is pinned at Hugging Face revision
`084a18386e149a454dd6dbe0123478e75c6b09ce`. The source manifest contains
exactly 30 objects and 2,843,350,925 bytes. Its checksum at feature-contract
freeze was
`5dbbfeb23442a6b1ebdb829dc239c6a3c6c655d71446cde4551b83cc6af456ac`.

The HEGM preprocessing reference was inspected at commit
`d16c097d57f867f438f906949d9cc1efd0f6ddc4`. The allowed source file
`scripts/preprocess_vklsvd.py` has SHA-256
`5e7137820f8d923692afc0055c60069faa27678d98f8fa29054b4777b53be92e`;
the repository license is Apache-2.0 with SHA-256
`eb74dde2903b7a84802e5907b35717441530b25738fc00c0b6639bb8287878c3`.
Only the license, VK preprocessing, and its data/feature support source were
opened. The README, evaluator, metrics implementation, checkpoints, and
published results were not opened or used.

Week 26 remained sealed. Feature code did not open, stat, glob, decode, or
schema-inspect its file. Its pre-existing byte count and cryptographic hash
remain provenance-only facts from the acquisition manifest. The phase-1
audit opened Parquet metadata for weeks 0--25 only. It verified one identical
12-column interaction schema across all 26 allowed weeks and decoded no
interaction values.

The allowed files contain 28,106,627 history rows in weeks 0--19, 7,505,820
training rows in weeks 20--24, and 1,516,831 validation rows in week 25. These
are structural row-count checks, not model metrics. Every released row is
retained in `(week, source_row_index0)` order; no deduplication is applied.

## Metadata and label provenance

The user metadata table contains 10,000,000 unique `user_id` values and no
nulls in `user_id`, `age`, `gender`, or `geo`. The item metadata table contains
19,627,601 unique `item_id` values and no nulls in `item_id` or `duration`.
Duration ranges from 5 to 180 seconds and has no non-positive value.

The primary-label constant is exactly `c = 157.0` seconds. It is the nearest
interpolation 0.99 quantile of the full static `items_metadata.duration`
column, matching the pinned HEGM source. It is not a watch-time quantile. The
label is

`y = clip(current_row.timespent, 0, c) / c`.

Current-row `timespent` is label-only. Current-row like, dislike, share,
bookmark, author-click, and comment-open outcomes are label-only for optional
secondary tasks and never enter the same row's features.

## Frozen split and feature state

- History and all aggregate/iALS fitting: weeks 0--19.
- Training labels: weeks 20--24.
- Validation: week 25.
- Untouched test: week 26.
- Categorical mappings and scale constants: fit on weeks 20--24 only, then
  frozen.
- History aggregates and iALS factors: fit once on weeks 0--19 and frozen;
  train, validation, and eventual test outcomes never update feature state.

The six ordered context fields are `age`, `gender`, `geo`, `place`,
`platform`, and `agent`. They use train-only categorical mappings with
separate OOV and missing tokens, and bypass the reciprocal core. User and item
IDs remain audit/grouping/history-lookup keys, not model inputs.

## Exact feature algebra

The code-generated manifest has 31 ordered base operands (`P0`), 27 support
and numerical-validity indicators (`V`), and 20 explicit engineered features
(`E`). The public `P` regime is `P0 + V`, hence 58 ordered numerical operands.

For both user and item groups, history supplies raw show count, show count
shifted by `beta = 100`, total watch time, and event counts shifted by
`alpha = 5`. User rates cover like, share, bookmark, comments, dislike, and
15/30-second survival. Item rates add author-click. Average watch time uses
the unshifted positive show count with a zero-history validity gate; event
rates use `(event_count + 5) / (show_count + 100)`.

Three iALS models are fit on history targets: watch time, any feedback, and
15-second survival. Each contributes a dot product plus user/item norms to
`P0`; its explicit cosine belongs to `E`. The fixed source geometry is 32
factors, 15 iterations, regularization 0.01, and random state 42. Deterministic
adaptations sort IDs ascending, sum duplicate sparse coordinates, and use one
thread. Unseen factors receive zero operands and explicit seen/positive-norm
indicators.

All P0 and E numerical fields are divided by their train-only absolute 0.99
quantile with nearest interpolation, a minimum positive scale of `1e-12`, no
centering, and no clipping. V remains binary. Eleven semantically valid
denominators are enabled: duration, raw and beta-shifted user/item shows, and
the six iALS norms. Invalid or zero denominators are validity-gated; active
negative denominators are rejected. Per-field indicators record whether the
normalized positive denominator enters the fixed DCDN reciprocal floor at
`0.01`.

The exact contract hashes are:

- numeric operand schema: `bb2f6c47fa12c175a3820830f90ff577acb293c56b107c7d1c40e4c682f2ff4d`;
- denominator mask: `9e168fae4014f4763008e89f61324858f17d376f56fb5de46f2775911650bcaa`;
- encoded C+P flat schema: `2cfd6d12cd85346e413f476ff223e19d57be213d74fe68b75d6f0ce686c0f091`;
- encoded C+P+E flat schema: `381beb02c21c8a56a4111b2dde50eb7c870518e9289a806f6e4ddb8a2c9dda59`.

With 16 coordinates per C embedding, C bypass width is 96. C+P has 64 model
fields and flat width 154; C+P+E has 84 model fields and flat width 174.

The generated counts are reported honestly. Removing E is 39.2% only in the
provisional P0+E numerical subspace (`51 -> 31`) and 35.1% after adding C but
before V (`57 -> 37`). In the actual manifested numerical view including V,
the change is `78 -> 58` (25.6%); over all manifested fields it is `84 -> 64`
(23.8%). None of these values is selected to resemble or compared with the
industrial 40.1% reduction.

## Metrics and remaining gate

Seed-42 validation selection uses normalized MSE. Frozen reports include
normalized MSE, MAE in seconds, MSE in seconds squared, and an order-invariant
global concordance with target-tie exclusion and prediction ties worth 0.5.
Final seeds are 42, 43, and 44. Paired 95% percentile intervals use 2,000
user-cluster bootstrap replicates with seed 20260818; four primary normalized
MSE contrasts use Holm--Bonferroni at family-wise alpha 0.05.

No upstream field or formula needed by this phase remains unresolved, so the
feature-definition gate itself does not block the first validation model
metric. Test access remains blocked until validation-only tuning is complete
and both the protocol manifest and selected-configuration lock are hashed.

The frozen contract is
`configs/data/vk_lsvd/features_v1.yaml`, currently SHA-256
`380d105c847a73afb4927740e40401124688f895142792309b4fc8267a2a5f2b`.
