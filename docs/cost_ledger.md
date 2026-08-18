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

Control-plane historical charges and balance are unavailable and are not
invented. The owner explicitly waived provider-side expiry, credentialed
inspection, and quota metadata as execution gates. Estimated warning headroom
at the latest checkpoint is USD 314.07 and estimated hard-cap headroom is USD
404.07. For the approved one-hour pre-smoke ETA, projected warning headroom is
USD 310.78 and projected hard-cap headroom is USD 400.78. The USD 370 warning estimate is crossed at
`2026-08-22T18:05:06Z`; the USD 460 cost estimate would be crossed at
`2026-08-23T21:26:26Z`, so the earlier 120-hour stop remains binding.
