# KuaiRec feature and split audit

Status: **frozen pre-metric contract; corrected blind audit passed; test remains locked**.

## Inspection boundary

The audit is outcome-blind. `watch_ratio` was never selected. Current-row
`play_duration` was used only for the frozen raw-millisecond `>0` predicate,
and its values were neither exposed nor summarized. No validation or test
model metric, evaluator output, checkpoint, result table, row sample, or
test-label shard was read. The sealed-test feature view was not materialized.
`test_unlocked` remains `false`.

Allowed durable evidence is limited to source and schema hashes, aggregate
filter/split/cold-drop counts, train-only mapping cardinalities, the train-only
duration constant, aggregate missingness of explicitly excluded columns, and
outcome-free split-membership hashes over source row indices.

## Source and reference provenance

- Source manifest: `data/manifests/kuairec_source_v1.yaml`, SHA-256
  `963019bcb4c8dbd4663f23c24a472fe0332f6494fdcbbaa74ca158a73488c5b4`.
- Raw archive SHA-256:
  `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`.
- `big_matrix.csv` SHA-256:
  `4ee1a72ff9ce4e86ef50525da9dfd7aefcefa717b37c9d570010f689c42f4ef9`.
- `item_daily_features.csv` SHA-256:
  `45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b`.
- Pinned HEGM revision:
  `d16c097d57f867f438f906949d9cc1efd0f6ddc4`; only
  `scripts/preprocess_kuairec.py`, `hegm/features.py`, and `LICENSE` were
  inspected. The preprocessing and license hashes are respectively
  `184729b5dd200a3d8f8e13ae83179286242b586fb27162bb7b713e0988ae2807`
  and `eb74dde2903b7a84802e5907b35717441530b25738fc00c0b6639bb8287878c3`.
  HEGM README, results, evaluator, metrics, and checkpoints remain forbidden.

Large raw files remain in the ephemeral public-data cache. The repository
contains only compact contracts, audits, tests, and hashes.

## Corrected exclusion decision and invalidated evidence

The first rejected contract
`c6f539d228dce75644b62cb445bf393e141944c9e66bf84545db77e3b75a606d`
declared four collection counters and four derived rates. An initial diagnostic
incorrectly interpreted `69,683` nulls in each counter as a source-wide
all-null row count. A later aggregate-only diagnostic established the exact
source facts:

- one archive member and one extracted file named `item_daily_features.csv`;
- `343,341` CSV records and `343,341` unique `(video_id,date)` keys, with no
  duplicate-key reduction;
- for each of `collect_cnt`, `collect_user_num`, `cancel_collect_cnt`, and
  `cancel_collect_user_num`: `343,341` total, `69,683` null, `273,658`
  non-null, and zero parse-invalid, non-finite, or negative non-null cells;
- no date contains `69,683` rows.

Thus the old source-wide-all-null premise was false. The owner subsequently
approved full exclusion of exactly those four partially-null counters and
their four rates (`e_collect_per_show`, `e_collect_per_show_user`,
`e_cancel_collect_per_show`, and `e_cancel_collect_per_show_user`) from every
KuaiRec regime. No zero imputation and no row/population filter were added.
The final schemas remain C/P/V/E = `5/44/6/46`.

The intermediate contract
`f0883ced4c8fb3f0945cacb1affa0b4891f748c4294e3a2d1016100191011b8d`
and its generated blind PASS
`3b38391097c9294b7622bc91c7b836283247b7ff41bbf41a0a7a4e84c35357ab`
are **invalid and must not be used**. Their exact bytes are preserved as
`configs/data/kuairec/features_v1.rejected-f0883ced.yaml` and
`configs/data/kuairec/blind_scan_v1.rejected-3b383910.yaml`. The aggregate
diagnostic and invalidation chain are preserved in
`configs/data/kuairec/blind_scan_v1.invalid.yaml`. Every artifact has an exact
SHA-256 sidecar.

## Duration-unit correction and quarantine record

The contract rejected at
`c5aadbeb7da5f05a281681d30a7ef98e8c660019f6c0bc679e430c1014e5a23c`
failed the train-only seconds sanity gate because raw duration fields had been
treated as seconds. The owner approved this deterministic correction:

- run current play/video positivity filters on unchanged raw milliseconds;
- divide big-matrix `video_duration` by `1000` before train-only `c` and P;
- divide current `play_duration` by `1000` only inside future sealed label
  materialization;
- divide lagged daily `play_duration` sums by `1000` before P and E.

While checking the unit, a narrow public search unexpectedly exposed an
official KuaiRec README snippet containing a public example row and a published
dataset-size summary. Those values are quarantined: they are not persisted and
were not used for feature, schema, search-space, or result decisions. The only
retained schema fact is that the relevant duration fields use milliseconds.
No local test label vector, DCDN output, or model metric was exposed.

## Frozen feature algebra and routing

- C contains train-mapped user, video, and three category-level IDs. C bypasses
  division and joins the common context encoder.
- P contains positive video duration in seconds, 39 strict lagged daily base
  operands (with lagged play duration in seconds), and four `+1` shifted
  count denominators.
- V contains six declared history/duration validity indicators.
- E contains 46 rates computed only from the matched P bases and is available
  only to the matched `DCNv2(C+P+E)` arm.
- DCDN-Core receives P+V only; C bypasses the reciprocal path. The ordered
  numeric operand schema contains 50 fields. Exactly five coordinates are
  denominator-eligible: video duration and the four shifted denominators.

Ordered schema hashes are:

- C: `08deb99c9ce479841b47771907a4d65bb022ca12f18f960d1d23bd0f0a2e0404`
- P: `d19573df38f1979066051f1a06408ac12faa227164c116377798ffa94ae658d9`
- V: `024723dc5302f73c4572c6be824944a227ab1c13ecb1c309591ca0e6ea80e9cc`
- E: `dcb3a271d158a98b682bbae7e753f22033f3b15ce964a1c0e27489b7be036df6`
- numeric operands:
  `41a21947778c9c70c7e59989cf51bcc53854c477a523e9d7b5bcd2654943c756`
- denominator mask:
  `28c0d620c8f4a063e01e13a58faf9312c7f1becd48d435409c4023008d0a9a35`.

Every daily aggregate is a duplicate-day sum followed by a per-video prefix
sum queried with `date < d`; same-day values and `.unique(video_id)` shortcuts
are forbidden. Missing prior history uses the declared all-zero base fallback
and explicit V indicator, distinct from the forbidden zero imputation of the
four partially-null excluded source columns.

## Frozen filter, split, mappings, and label constant

Filters are applied in this exact order: static `video_type != AD`, raw current
`play_duration > 0`, then raw current `video_duration > 0`. Rows are globally
sorted by `(timestamp,user_id,video_id,source_row_index0)`. The split is
`floor(0.8N)`, `floor(0.1N)`, and the remainder. User/item/category mappings
are train-only; validation and sealed test inner-drop unseen user/item rows.

The corrected full blind audit produced:

- raw rows: `12,530,806`;
- after non-ad: `12,495,027`;
- after positive current play: `12,255,608`;
- after positive current video duration: `12,255,608`;
- train: `9,804,486`;
- validation: `1,225,560` before and `1,141,154` after cold drop;
- sealed test: `1,225,562` before and `1,027,696` after cold drop;
- validation cold drops: `84,406` unseen-item rows, zero unseen-user rows;
- sealed-test cold drops: `197,866` unseen-item rows, zero unseen-user rows;
- train mapping cardinalities: `7,176` users and `9,049` videos.

The train-only R-7 0.99 quantile of filtered video duration after `/1000` is
`c = 131.684` seconds. Validation and sealed-test durations did not contribute
to `c`. Current play remains label-only and no label shard was materialized.

Outcome-free membership hashes are:

- train: `52106343fb06d25ffc14e667685b93ef8a619dad3c49e535ffa22177e2a3dc4c`;
- validation after cold drop:
  `fa2ea4d67e0d2f19c2e99ab456a41966686e5088d0ce14ccaedb73f2cca56412`;
- sealed test after cold drop:
  `95fa70d0db67301d647e67a3e7998c65fd9b12fc52cc3aaf47dc52ed15830f60`.

## Final artifacts and verification

The corrected contract is `configs/data/kuairec/features_v1.yaml`, SHA-256
`ab016a7f6d300e7a90de9940891ea974695aa5f61ea85985e239588290522c36`.
The corrected blind audit is `configs/data/kuairec/blind_scan_v1.yaml`,
SHA-256
`716e976e0afc642b354c50096ca0d12bdef07e55381b5e91b75cc7c4910f810a`.

The rerun cost guard allowed a 0.5-hour ETA at estimated spend `$43.96` and
projected spend `$45.61`. Scoped verification in a fresh `/tmp` mirror passed
`15` tests; Ruff and Ruff format checks passed for the three KuaiRec Python
files. No global suite or FUSE-resident test run was performed.

Category mappings remain explicitly deferred to train-only feature
materialization. No model training, evaluation, or test metric is authorized
by this audit.
