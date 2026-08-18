# KuaiRand-Pure phase-1 feature audit

Status: **phase-1 temporal gate passed; test remains locked**. The owner
approved the deterministic boundary embargo and the actually released
2022-04-09 through 2022-04-21 burn-in. The feature contract was refrozen and
a fresh pinned-raw, label-blind audit reproduced every approved cutoff and row
count exactly. No feature view, label distribution, validation/test model
metric, or test view was materialized.

## Safety boundary

Both raw passes decoded and aggregated only:

- `date`;
- `time_ms`;
- `is_rand`;
- `source_file` and zero-based `source_row_index0`.

Every other CSV field remained an opaque byte slice and was discarded without
selection, decoding, validation, or aggregation. In particular, neither
`long_view` nor another outcome/static/model-feature value was opened. The
first pass recomputed whole-file byte hashes and coverage metadata; the second
pass applied the approved temporal rule to the same allowlisted projection.
`test_unlocked=false`, `test_labels_opened=false`,
`validation_or_test_metrics_read=false`, and
`final_test_view_materialized=false` remain in force.

Before the scan, the mandatory guards passed at `2026-08-18T14:59:44Z`:

- earliest persisted `ledger_epoch_utc`: `2026-08-18T09:13:18Z`;
- estimated spend: USD 44.00 under the frozen USD 3.29/hour rate plus USD 25
  storage reserve;
- projected spend for the 0.5-hour audit ETA: USD 45.64, below both thresholds;
- project-owned durable files: 4,074,723 bytes;
- locally observed persistent free space: 120,134,823,903,232 bytes;
- ephemeral overlay free space: 77,280,350,208 bytes.

The conservative 40GB project-owned durable cap and 10GB free-space reserve
remain enforced despite the larger capacity reported by the local mount.

No RunPod credential or control-plane mutation was used.

## Refrozen contract

- feature manifest: `configs/data/kuairand_pure/features_v1.yaml`;
- feature manifest SHA-256:
  `92a060d34e16d5b3dd2674d9ca75281fac0d20a54adda03c9ecaa206e9e601ef`;
- referenced source-manifest SHA-256:
  `bc5b441fdd3188d033e0ecbf0bdfd73f8bcf07bbc744d474d260ee220ae4d6e9`;
- raw archive SHA-256:
  `c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0`.

The exact feature algebra is unchanged by the temporal correction: 8
categorical context fields (`C`), 108 strictly lagged positive operands (`P`),
24 validity fields (`V`), and 112 explicit lagged rates (`R`). The public
numeric interaction view is the ordered 132-field `P+V` schema. Exactly 24
exposure-count or duration-sum coordinates are reciprocal-eligible; all `C`,
`R`, other `P`, and all `V` coordinates are masked from the reciprocal carrier.

The pre-metric constants remain binary-success pseudocount 0.5, exposure
pseudocount 1, time-numerator floor 1 ms, duration-denominator floor 1000 ms,
count scale 100, time scale 1000 ms, and normalized positive denominator floor
0.01. No denominator is z-centered. Monthly
`video_features_statistic_pure.csv`, current-row outcomes, and `is_rand` remain
forbidden as model inputs.

## Pinned raw evidence

All three event files were regular single-link files whose sizes and SHA-256
values matched the source manifest during the audit:

| Source | Bytes | SHA-256 |
|---|---:|---|
| `log_standard_4_08_to_4_21_pure.csv` | 83,961,282 | `5bb6eb0b3d9f47e5436cb5dc82ee1899b845ebf9750a5560b801e929e18bd41c` |
| `log_standard_4_22_to_5_08_pure.csv` | 21,765,075 | `429e3b948828942e572f2c3a5be5a25799ffe75591d22d18cf417b9b534d31fd` |
| `log_random_4_22_to_5_08_pure.csv` | 87,086,116 | `60b80994da969cd53da4d50c37ba3dafd6fb185df804c92c8410df34845a9d2c` |

The files contain 2,622,668 rows:

| Source | Policy | Rows | Observed released dates |
|---|---|---:|---|
| `log_standard_4_08_to_4_21_pure.csv` | standard | 1,141,112 | 2022-04-09..2022-04-21 |
| `log_standard_4_22_to_5_08_pure.csv` | standard | 295,497 | 2022-04-22..2022-05-08 |
| `log_random_4_22_to_5_08_pure.csv` | randomized | 1,186,059 | 2022-04-22..2022-05-08 |

There is no released 2022-04-08 row. The approved burn-in is therefore exactly
standard-policy rows from 2022-04-09 through 2022-04-21. No synthetic,
imputed, or claimed 2022-04-08 history was created. Train, validation, and test
calendar dates were not changed.

The released `date` remains the authority for calendar strata, while `time_ms`
remains the authority for global prequential order and strict feature cutoffs.
The earlier audit established that `released_date - UTC_date(time_ms)` is 0 or
+1 day; no timezone coercion is introduced.

## Approved boundary embargo

For each adjacent boundary, every row in the next-role union is retained at
that boundary. An earlier-role row is excluded exactly when

```text
time_ms >= min(time_ms of the next-role union).
```

Only `date`, `time_ms`, policy, source file, and source row index determine the
rule. An embargoed row is outside labels, state updates, and every future
materialized view. The three derived and refrozen cutoffs are:

| Boundary | Cutoff `time_ms` |
|---|---:|
| history -> train | 1650552767170 |
| train -> validation union | 1651503652252 |
| validation union -> test union | 1651762945505 |

The fresh audit reproduced the approved effects exactly:

| Role | Before embargo | Excluded | After embargo |
|---|---:|---:|---:|
| history/burn-in standard | 1,141,112 | 1,204 | 1,139,908 |
| train-label standard | 202,296 | 976 | 201,320 |
| validation randomized | 315,340 | 3,850 | 311,490 |
| validation-period standard update-only | 49,933 | 609 | 49,324 |
| headline randomized test | 322,152 | 0 | 322,152 |
| secondary standard test | 43,268 | 0 | 43,268 |
| pre-validation randomized, protocol-excluded | 548,567 | 0 | 548,567 |

Total embargo exclusions are 6,639. The separately protocol-excluded 548,567
pre-validation randomized rows remain excluded from training labels and the
initial state; their table entry only records that the boundary embargo did
not alter that already-excluded stratum.

Before the embargo, all three adjacent timestamp ranges overlapped. After the
embargo, every inequality is strict:

| Boundary | Pre earlier max | Later min/cutoff | Post earlier max | Passed |
|---|---:|---:|---:|---|
| history -> train | 1650556328855 | 1650552767170 | 1650552767022 | yes |
| train -> validation union | 1651506723458 | 1651503652252 | 1651503651565 | yes |
| validation union -> test union | 1651765930128 | 1651762945505 | 1651762945288 | yes |

The audit reported `blockers=[]`, `frozen_real_evidence_matches=true`,
`post_embargo_time_boundary_gate_passed=true`, and
`phase1_gate_passed=true`.

## Row identity and equal-timestamp semantics

The contract hashes every parsed raw field in released column order using NFC
UTF-8 and unsigned 8-byte big-endian length prefixes. Whitespace and numeric
spelling are preserved. This full-row SHA-256 is the only deduplication key;
deduplication by `(user_id, video_id, time_ms)` remains forbidden. The
canonical duplicate occurrence is the minimum deterministic
`(policy_priority, source_file, source_row_index0)` tuple.

Retained rows are globally grouped by `time_ms`. Every row in an
equal-timestamp group is scored against state containing only smaller
timestamps. Updates occur only after the complete group is scored, in
standard-before-randomized, source-file, source-row order. Embargoed rows never
update state.

## Verification

The final verification ran from an isolated fresh `/tmp` check tree, not from
the persistent FUSE worktree:

```text
14 passed in 2.19s
All checks passed!
4 files already formatted
```

The targeted suite covers strict manifest/sidecar/source validation, exact
field and mask hashes, owner-approved temporal policy tamper rejection,
full-row fingerprint properties, stable row IDs, equal-timestamp ordering,
April 9 split classification, exact embargo cutoff inclusivity, opaque-label
coverage fixtures, and policy mismatch rejection. Ruff check and format-check
were limited to the four scoped Python files. No global or FUSE-resident test
suite was run.

Reproduce the real label-blind audit with:

```bash
source scripts/env.sh
"${UV_PROJECT_ENVIRONMENT}/bin/python" \
  scripts/data/kuairand_pure_features.py \
  --raw-data-dir \
  /home/codex/.cache/dcdn-public/raw/kuairand_pure/KuaiRand-Pure/data \
  --contract configs/data/kuairand_pure/features_v1.yaml \
  --repository-root "$PWD"
```

The expected exit status is zero. The JSON must retain
`test_unlocked=false`, report no blockers, reproduce the exact evidence above,
and record that no label, model feature, metric, or final test view was opened.
