# DCDN cost and time ledger

This ledger is cumulative and must never be reset.

- `ledger_epoch_utc: 2026-08-18T09:13:18Z`
- `max_cumulative_spend_usd_pre_tax: 460.00`
- `warning_threshold_usd_pre_tax: 370.00`
- `maximum_wall_hours: 120`
- `epoch_120h_stop_utc: 2026-08-23T09:13:18Z`
- `absolute_stop_utc: 2026-08-24T11:00:00Z`
- `effective_stop_utc: 2026-08-23T09:13:18Z`
- `finalization_reserve_start_utc: 2026-08-23T01:13:18Z`

The epoch is the earliest locally verifiable container start and is therefore
conservative relative to observed agent activity. Any provider charge before
the epoch is currently unknown and must be recorded separately if discovered.

| Checkpoint UTC | Live price | Actual cumulative spend | Projected spend | Evidence | Decision |
|---|---:|---:|---:|---|---|
| 2026-08-18T11:52:44Z | UNVERIFIED | UNVERIFIED | UNVERIFIED | No credential-free control-plane data; handoff-only historical display was USD 3.29/h | Blocked before owner override |
| 2026-08-18T11:59:30Z | 3.29 assumed | Unavailable and owner-waived | 34.11 | 2.770000 elapsed hours x USD 3.29 + USD 25 fixed storage reserve | Proceed under frozen local-accounting override |
| 2026-08-18T12:36:18Z | 3.29 assumed | Unavailable and owner-waived | 36.13 | 3.383427 elapsed hours x USD 3.29 + USD 25 fixed storage reserve | Bootstrap, CPU suite, and dual-GPU deterministic preflight passed |
| 2026-08-18T13:26:52Z | 3.29 assumed | Unavailable and owner-waived | 38.90 current; 45.48 at 2 h ETA | 4.226242 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; local cost guard allowed the mandatory integration stage | Continue source/model/search integration; no provider action performed |
| 2026-08-18T14:14:40Z | 3.29 assumed | Unavailable and owner-waived | 41.53 current; 44.82 at 1 h ETA | 5.022851 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; acquisition/model/search checkpoint and local guards passed | Continue dataset feature audits; no provider action performed |
| 2026-08-18T14:50:52Z | 3.29 assumed | Unavailable and owner-waived | 43.51 current; 50.09 at 2 h ETA | 5.626101 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; isolated feature-contract tests and local cost/storage guards passed | Continue mandatory VK materialization and harness integration; no provider action performed |
| 2026-08-18T15:17:23Z | 3.29 assumed | Unavailable and owner-waived | 44.96 current; 51.54 at 2 h ETA | 6.068132 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; all three phase-1 data gates and isolated 46-test checkpoint passed | Continue shared encoder/runtime bridge; no provider action performed |
| 2026-08-18T15:38:34Z | 3.29 assumed | Unavailable and owner-waived | 46.13 current; 52.71 at 2 h ETA | 6.421170 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; canonical VK full scan and bounded runtime cache passed local cost/storage guards | Continue mandatory encoder/harness integration and smoke; no provider action performed |
| 2026-08-18T17:05:20.923596Z | 3.29 assumed | Unavailable and owner-waived | 50.88 current; 60.75 at 3 h ETA | 7.867479 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; projected end 2026-08-18T20:05:20.923596Z; storage guard measured 5,564,928 durable bytes and 62.794891 GiB overlay available with no violations | Proceed to mandatory deterministic preflight and train-only smoke under the active Protocol v3; no provider action performed |
| 2026-08-18T18:37:19.390739Z | 3.29 assumed | Unavailable and owner-waived | 55.93 current; 59.22 at 1 h ETA | 9.400386 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; isolated 263-test regression passed; storage guard measured 5,939,200 durable bytes and 62.554615 GiB overlay available; both A100s idle | Proceed to clean commit/push and two concurrent one-batch implementation smoke runs; no provider action performed |
| 2026-08-18T19:53:13.266664Z | 3.29 assumed | Unavailable and owner-waived | 60.09 current; 60.09 at 0 h ETA | 10.665352 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; both one-effective-batch implementation smokes completed with no validation/test metric, OOM, or non-finite event; storage measured 127,622,144 durable bytes and 62.506607 GiB overlay available; zero task/GPU processes; final post-cleanup audit SHA-256 `6748d9f643115f0d85b73041b0892a362b1f38efda7c404ca5d329b4ea4650b4` | Implementation-only smoke checkpoint passed; ordinary seed-42 validation-only tuning may proceed, while capacity-control and final/test gates remain locked; no provider action performed |
| 2026-08-18T21:14:23.806081Z | 3.29 assumed | Unavailable and owner-waived | 64.54 current; 64.54 at 0 h ETA | 12.018279 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; both frozen seed-42 DCDN-Core LR candidates completed with nine validation checks each, zero OOM/non-finite observations, and no test metric; storage measured 852,498,944 durable project bytes and 62.470196 GiB overlay available; independent postaudit confirmed zero task/GPU processes; per-model selection artifact SHA-256 `66458ce3d96d6e4f5d5382dab904a5c120134ba52385d8a5bcdbd9d3c2bccb49` | Validation-only DCDN selection froze LR 0.001; continue remaining tuning, while capacity-control and final/test gates remain locked; no provider action performed |
| 2026-08-18T23:05:25Z | 3.29 assumed | Unavailable and owner-waived | 70.63 current; no new run projected | 13.868611 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; ordinary full-rank DCNv2 LR=0.001 completed as scientifically eligible validation-only tuning, while LR=0.0003 and its exact resume failed on repeated supervisor/registry `EIO` and remain scientifically ineligible; read-only storage snapshot measured 1,398,235,574 project bytes and 66,952,908,800 overlay bytes available; both A100s were at 0 MiB and 0%, GPU compute-apps were empty, and zero task-owned training/worker processes remained | Block a third LR=0.0003 resume until registry/worker hardening, fault-injection tests, and a clean commit/new run ID; capacity control and final/test remain locked; no provider action performed |
| 2026-08-19T01:12:20.526933Z | 3.29 assumed | Unavailable and owner-waived | 77.59 current; no new run projected | 15.984035 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; after committed registry/worker hardening, ordinary full-rank DCNv2 LR=0.0003 exact resume-v2 completed scientifically eligible validation-only tuning at 39,997,440 examples with zero OOM/non-finite observations and `test_metrics=null`; result SHA-256 `0e6381d016a65a48d9f12f48b6c871ae242139b2e9a2341f1ecba3efd03cf58e`; two bounded heartbeat-helper timeouts recovered with byte-identical local/persistent incident receipts; storage guard measured 1,461,672,448 project bytes and 62.267612 GiB overlay available; independent terminal audit found both A100s idle, zero task PIDs, and no running registry record | Strict validation-only minimization selects LR=0.0003; canonical selection receipt SHA-256 `869cbcb7a227b274700cda142f23c59040497bfdfe4da101888e2cc63f39d67a`; bind it before capacity-control tuning; final/test remains locked; no provider action performed |
| 2026-08-19T05:35:14.818387Z | 3.29 assumed | Unavailable and owner-waived | 92.00 current; no new run projected | 20.365783 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; clean commit `bc584157a5845571e72acde7d5075dd5f5d3bcc2` passed 534 local CPU tests; immutable compact capacity authorization SHA-256 `d0ef149274d0f2692acc1f166d72350e44bd8a99331b7ae7f4dbc7d9df1cf1b2` and sidecar independently reopened; storage guard measured 8,303,755,776 project bytes, 42.266525 GiB logical headroom, and 62.153824 GiB overlay available | Compact LR-binding gate closed; next materialize/audit schema-v4 capacity-control run; final/test remains locked; no provider action performed |
| 2026-08-19T06:56:00.678840Z | 3.29 assumed | Unavailable and owner-waived | 96.43 current; no new run projected | 21.711855 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; rank-12 `DCNv2-capacity-matched(C+P)` completed 39,997,440 examples and nine validation checks with zero OOM/non-finite observations; immutable result SHA-256 `fdcf94786df724aa0a91a525b8e23f296530b8837e2669a2bedfe719bc6d0507`; best validation MSE `0.02100464611993931`; storage guard measured 8,726,980,608 project bytes, 41.872366 GiB logical headroom, and 62.145443 GiB overlay available | Capacity-control validation cell complete and scientifically eligible; DCDN is lower by 0.1303% on validation, while full-rank DCNv2 remains lower than DCDN; continue remaining seed-42 validation-only tuning; final/test remains locked; no provider action performed |
| 2026-08-19T09:22:40.681483Z | 3.29 assumed | Unavailable and owner-waived | 104.47 current; 104.47 at 0 h ETA | 24.156300 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; both frozen AFN+ LR candidates completed validation-only tuning with zero OOM/non-finite observations; a committed streaming-checkpoint verifier reconciled the two false 64-MiB terminal failures without trainer rerun; strict validation minimization selected LR=0.0003 at normalized MSE `0.0208063458817398`; immutable per-model selection receipt SHA-256 `69463d2687d46520ea270ffcdfeaaa8ef47cd11173dc3943780cc4cc486d7e54`; all ten checkpoint pairs passed safe CPU/hash/finiteness audit; storage measured 9,519,565,312 project bytes, 41.134214 GiB logical headroom, and 62.117252 GiB overlay available; both A100s idle and no task-owned training process remained | AFN+ validation choice frozen; retain all checkpoints until evidence commit/push/read-test, continue the remaining seed-42 validation-only model matrix, and keep final/test locked; no provider action performed |

Control-plane historical charges and balance are unavailable and are not
invented. The owner explicitly waived provider-side expiry, credentialed
inspection, and quota metadata as execution gates. Estimated warning headroom
at the `2026-08-19T09:22:40.681483Z` checkpoint is USD 265.53 and estimated
hard-cap headroom is USD 355.53. The USD 370 warning estimate is crossed at
`2026-08-22T18:05:06Z`; the USD 460 cost estimate would be crossed at
`2026-08-23T21:26:26Z`, so the earlier 120-hour stop remains binding.

The DCDN, ordinary full-rank DCNv2, and AFN+ candidate pairs are
scientifically eligible validation-only tuning, not final/test results. Their
selected LRs and exact best-checkpoint lineages are durable, but these
per-model artifacts are not the dataset-wide `selected_configs.lock`. The
first two ordinary-DCNv2 LR=0.0003 attempts remain recorded as execution
failures; hardened resume-v2 is the scientifically eligible completion. Strict
validation minimization selects ordinary DCNv2 LR=0.0003 and AFN+ LR=0.0003.
Compact authorization SHA-256
`d0ef149274d0f2692acc1f166d72350e44bd8a99331b7ae7f4dbc7d9df1cf1b2`
binds the ordinary-DCNv2 receipt and inherited capacity-control LR. The bound
rank-12 `DCNv2-capacity-matched` validation run is complete at normalized MSE
`0.02100464611993931`; DCDN is lower by about 0.1303% on validation, while
full-rank DCNv2 remains the lower of the three. AFN+ selected normalized MSE
is `0.0208063458817398`. Final/test remains locked until every required model
selection and checkpoint lineage is frozen, the one-pass final runner is
implemented and audited, and the protocol lifecycle explicitly activates test
access; `test_unlocked=false`.

## 2026-08-19 C+P+E premetric checkpoint

At `2026-08-19T10:12:29.624367Z`, the immutable ledger epoch still yielded
24.986562 elapsed hours. At the owner-approved USD 3.29/hour estimate plus
the fixed USD 25 storage reserve, estimated cumulative spend was USD 107.21
and a conservative three-hour mandatory compute projection was USD 117.08.
The projected end remained before the finalization reserve and the local
guard returned `launch_allowed=true`. Storage was also healthy at
9,519,765,504 project bytes, 41.134027 GiB logical headroom, and 62.000469
GiB overlay space available. No provider action or GPU job was performed.

At `2026-08-19T10:40:53.693746Z`, estimated cumulative spend was USD 108.76.
The first C+P+E pair had stopped within seconds at a shared harness
capacity-projection preflight, before any training batch or validation/test
metric. Both A100s and all task processes were idle after the failure. The
paired technical-failure receipt SHA-256 is
`d60f40816564427f41cf0e9748fc4d76f3fd6e613aab73d781630d9c2d6c0bee`.
The failures do not complete a scientific cell; retry is allowed only after
the scoped fix passes regression, is committed/pushed, and new run IDs are
materialized under the new clean code identity.

At `2026-08-19T11:08:25.444318Z`, estimated cumulative spend was USD 110.27.
The first retry pair had also stopped before a training batch, validation, or
test metric, this time at the remaining C+P-only parameter-count gate. Both
A100s and all task processes were idle afterward. The paired technical-failure
receipt SHA-256 is
`91a5011e5bfb7955e838f3256fac320c13b6a7e2a212ff07063a52760e0ebe17`.
Project-owned durable bytes were 9,520,076,800, logical headroom was
41.133738 GiB, and overlay free space was 61.999126 GiB. These technical
failures do not complete a scientific cell. The fresh full CPU regression
passed 541/541; another retry requires a clean committed/pushed fix and fresh
run IDs.
