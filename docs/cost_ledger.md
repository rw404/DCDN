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
| 2026-08-19T13:10:53.204095Z | 3.29 assumed | Unavailable and owner-waived | 116.99 current; 116.99 at 0 h ETA | 27.959779 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; both frozen VK-LSVD C+P+E full-rank DCNv2 LR candidates completed scientifically eligible validation-only tuning after one exact resume from a checkpoint-file `fsync` EIO; strict minimization selected LR=0.001 at normalized MSE `0.020826933840839952`, lower than LR=0.0003 by `0.000022353418883760273`; immutable selection receipt SHA-256 `85ab7ccbb8d74652f3aa3416a9ada872b36ed2de24c7b8341b56b9b8633f5c41`; all 14 checkpoint pairs passed strict sidecar audit and both terminal best/last pairs passed safe CPU finite-tensor reopen; storage measured 10,374,400,512 project bytes, 40.338087 GiB logical headroom, and 61.975178 GiB overlay available; both A100s idle and no task-owned trainer remained | C+P+E control LR frozen; retain all checkpoints until evidence commit/push/read-test, continue remaining RQ1 validation selections, and keep final/test locked; no provider action performed |
| 2026-08-19T14:42:01.276325Z | 3.29 assumed | Unavailable and owner-waived | 121.98 current; 121.98 at 0 h ETA | 29.478688 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; both frozen VK-LSVD C+P DNN LR candidates completed scientifically eligible validation-only tuning with zero OOM/non-finite observations and `test_metrics=null`; strict minimization selected LR=0.001 at normalized MSE `0.020899830835974933`; immutable selection receipt SHA-256 `7c851bfb7ca37e822fe05faca78a6488f56c447c995286143379a3611a166204`; all ten checkpoint pairs passed strict SHA sidecars and safe CPU finite-tensor reopen; two bounded registry-helper timeouts recovered without trainer interruption; storage measured 11,006,245,888 project bytes, 39.749635 GiB logical headroom, and 61.970650 GiB overlay available; both A100s idle and no task-owned trainer remained | DNN validation choice frozen; retain checkpoints until evidence commit/push/read-test, continue frozen C+P model order with DeepFM, and keep final/test locked; no provider action performed |
| 2026-08-19T16:19:45.321659Z | 3.29 assumed | Unavailable and owner-waived | 127.34 current; 127.34 at 0 h ETA | 31.107589 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; both frozen VK-LSVD C+P DeepFM candidates completed scientifically eligible validation-only tuning with zero OOM/non-finite observations and `test_metrics=null`; strict minimization selected LR=0.0003 at normalized MSE `0.020845575060232564`; immutable selection receipt SHA-256 `a6ed26dcce171cf8c27cb00e2db57c90cfcead0481251ed8e4a3a830ff812aeb`; all eight checkpoint pairs passed strict sidecars and safe CPU finite-tensor reopen; two bounded registry-helper timeouts recovered without trainer interruption; storage measured 11,554,703,872 project bytes, 39.238843 GiB logical headroom, and 61.963726 GiB overlay available; both A100s idle and no task-owned trainer remained | DeepFM validation choice frozen; retain checkpoints until evidence commit/push/read-test, continue frozen C+P model order with DCN, and keep final/test locked; no provider action performed |
| 2026-08-19T17:54:54.415622Z | 3.29 assumed | Unavailable and owner-waived | 132.56 current; 132.56 at 0 h ETA | 32.693449 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; both frozen VK-LSVD C+P Original DCN candidates completed scientifically eligible validation-only tuning with zero OOM/non-finite observations and `test_metrics=null`; strict minimization selected LR=0.0003 at normalized MSE `0.021035584402689534`; immutable selection receipt SHA-256 `f352c9d07a9690303d7d0fc5986147310d0f83d9185599b02fcdb58a86e50656`; all 11 checkpoint pairs passed strict sidecars and safe CPU finite-tensor reopen; three bounded registry-helper incidents recovered without trainer interruption; storage measured 12,217,759,232 project bytes, 38.621325 GiB logical headroom, and 61.952000 GiB overlay available; both A100s idle and no task-owned trainer remained | Original DCN validation choice frozen; DCDN is lower by about 0.277157% on matched seed-42 validation; retain checkpoints until evidence commit/push/read-test, continue with AutoInt+, and keep final/test locked; no provider action performed |
| 2026-08-19T19:22:09.064147Z | 3.29 assumed | Unavailable and owner-waived | 137.35 current; 137.35 at 0 h ETA | 34.147518 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; both frozen VK-LSVD C+P AutoInt+ candidates completed scientifically eligible validation-only tuning with zero OOM/non-finite observations and `test_metrics=null`; strict minimization selected LR=0.0003 at normalized MSE `0.020879945407626305`; immutable selection receipt SHA-256 `45ab5878b7a7ad5b57be418e6d17b5583b7cc84130fbef3a856af275c82169bb`; all 12 checkpoint pairs passed strict sidecars and safe CPU finite-tensor reopen; storage measured 12,993,276,416 project bytes, 37.899068 GiB logical headroom, and 61.948238 GiB overlay available; both A100s idle and no task-owned trainer remained | AutoInt+ validation choice frozen; AutoInt+ is lower than DCDN by about 0.464013% on matched seed-42 validation, so universal DCDN superiority is not supported; retain checkpoints until evidence commit/push/read-test, continue with Wukong-small, and keep final/test locked; no provider action performed |
| 2026-08-19T20:20:52.836314Z | 3.29 assumed | Unavailable and owner-waived | 140.57 current; 140.57 at 0 h ETA | 35.126343 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; the single pre-frozen VK-LSVD C+P Wukong-small fixed-no-tune candidate completed scientifically eligible validation-only training with zero OOM/non-finite observations and `test_metrics=null`; it stopped by patience after 24,002,560 examples and selected normalized MSE `0.02096245653858564`; immutable receipt SHA-256 `4f671539bd70d4054e9573c6d85f1cd991046e1980204fa78352c2cbd660cb16`; all four checkpoint pairs passed strict sidecars and safe CPU finite-tensor reopen; one bounded registry-helper incident recovered without trainer interruption; storage measured 14,255,196,672 project bytes, 36.723814 GiB logical headroom, and 61.942345 GiB overlay available; both A100s idle and no task-owned trainer remained | Wukong-small fixed validation cell frozen; seed-42 C+P per-model selection suite complete; next build/audit dataset selected-config lock and final runner while test remains locked; no provider action performed |
| 2026-08-20T12:23:12.516334Z | 3.29 assumed | Unavailable and owner-waived | 193.33 current; 193.33 at 0 h ETA | 51.165143 elapsed hours x USD 3.29 + USD 25 fixed storage reserve; all nine frozen VK-LSVD C+P validation rows now have scientifically eligible seeds 42/43/44; summary SHA-256 b5c64f84e6562add812a1fa5d3db2cbbf1ed712ff6e2237c59217dd6047fc070 and all 27 source SHA references passed; project storage 23,639,039,488 bytes, 27.984429 GiB logical headroom, 60.702209 GiB overlay available; both A100s idle, registry running empty, no task process | Narrow DCDN-vs-DCN validation hypothesis supported on all three seeds; universal DCDN superiority not supported; freeze selected-config lineage and one-pass bundle before test; no provider action performed |

Control-plane historical charges and balance are unavailable and are not
invented. The owner explicitly waived provider-side expiry, credentialed
inspection, and quota metadata as execution gates. Estimated warning headroom
at the `2026-08-19T20:20:52.836314Z` checkpoint is USD 229.43 and estimated
hard-cap headroom is USD 319.43. The USD 370 warning estimate is crossed at
`2026-08-22T18:05:06Z`; the USD 460 cost estimate would be crossed at
`2026-08-23T21:26:26Z`, so the earlier 120-hour stop remains binding.

The DCDN, ordinary full-rank DCNv2, AFN+, DNN, DeepFM, Original DCN, AutoInt+, and fixed Wukong-small validation cells are
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
is `0.0208063458817398`. DNN selects LR=0.001 at normalized MSE
`0.020899830835974933`, lower than selected DCDN by about 0.3692% on
validation. DeepFM selects LR=0.0003 at normalized MSE `0.020845575060232564`,
lower than selected DCDN by about 0.6279%. Original DCN selects LR=0.0003
at normalized MSE `0.021035584402689534`; selected DCDN is lower by about
0.277157% on matched seed-42 validation. AutoInt+ selects LR=0.0003 at
normalized MSE `0.020879945407626305`, lower than selected DCDN by about
0.464013%. Fixed Wukong-small reaches `0.02096245653858564`, about
0.070678% lower than DCDN but higher than AutoInt+. The seed-42 C+P per-model
selection suite is complete; next build and audit the dataset-wide lock and
final runner.
The separate C+P+E ordinary full-rank DCNv2 control now selects
LR=0.001 at normalized MSE `0.020826933840839952`; this is a
validation-only control selection and not an RQ1 verdict. Final/test remains
locked until every required model
selection and checkpoint lineage is frozen, the one-pass final runner is
implemented and audited, and the protocol lifecycle explicitly activates test
access; `test_unlocked=false`.

## 2026-08-20 three-seed RQ1 validation checkpoint

All nine frozen VK-LSVD `C+P` rows now have scientifically eligible
validation-only evidence for seeds 42, 43, and 44. The exact compact summary
is `results/final_validation/vk_lsvd_c_plus_p_rq1_validation_summary_v1.json`,
SHA-256
`b5c64f84e6562add812a1fa5d3db2cbbf1ed712ff6e2237c59217dd6047fc070`.
DCDN is lower than Original DCN on all three seeds by a paired mean
`0.000041601361759013` (about 0.197739%) and lower than the rank-12
capacity-matched control on all three by `0.000028828551562561` (about
0.137111%). This supports the narrow RQ1 validation hypothesis, while
full-rank DCNv2 and five other baselines have lower three-seed means and rule
out a universal-superiority claim.

These are validation results, not final test results. At
`2026-08-20T12:23:12.516334Z`, estimated warning headroom was USD 176.67 and
estimated hard-cap headroom was USD 266.67. Project storage was 22.0156 GiB
with 27.9844 GiB logical headroom and 60.7022 GiB overlay available. Both
A100s were idle, registry `running=[]`, and no task process remained.
Final/test stays locked until the complete selected-config lineage and audited
one-pass metric/uncertainty bundle are frozen and the protocol explicitly
activates test access; `test_unlocked=false`.

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

At `2026-08-20T12:58:08Z`, the dataset-wide VK-LSVD C+P
`selected_configs.lock` was materialized with SHA-256
`165a82204a039592c140e6be44a3245d1b62ce084946e1a45ac698229d270cfd`.
All 27 validation lineages and selected checkpoint sidecars were reverified;
the full CPU suite passed 547/547. The credential-free estimate was USD 195.25
for 51.747222 elapsed Pod-hours including the fixed USD 25 storage reserve.
Storage was 23,633,846,765 project bytes (22.010735 GiB), with 27.989265 GiB
logical headroom and 60.697773 GiB overlay available. No GPU job or provider
action was performed; final/test remains locked pending the one-pass runner.
At `2026-08-20T13:38:59.989945Z`, the dataset-independent one-pass final
evaluator and published-bundle recovery passed the full 553-test CPU suite
without opening test data. The credential-free estimate was USD 197.49 for
52.428331 elapsed Pod-hours including the fixed USD 25 storage reserve; a
one-hour mandatory compute projection was USD 200.78. Storage was
23,639,561,216 project bytes (22.016057 GiB), with 27.983943 GiB logical
headroom and 60.696690 GiB overlay available. Both A100s were idle, no registry
record was running, and no task-owned process remained. No provider action was
performed. Final/test remains locked pending the controlled week-26 view,
paired clustered contrast/Holm aggregation, and audited final-child activation.

At `2026-08-20T14:03:15.587098Z`, the frozen paired RQ1 aggregator passed the
full 562-test CPU suite without opening test data. It fixes twelve required
one-pass bundles, all-three-seeds-within-user resampling, 2,000 percentile
replicates at seed 20260818, the plus-one two-sided empirical-tail p-value, and
Holm--Bonferroni correction across the four declared normalized-MSE contrasts.
The credential-free estimate was USD 198.82 for 52.832663 elapsed Pod-hours
including the fixed USD 25 storage reserve; a one-hour mandatory compute
projection was USD 202.11. Storage passed at 23,639,831,040 bytes (22.016308
GiB), with 27.983692 GiB logical headroom and 60.701672 GiB overlay available.
No GPU job, test access, or provider action was performed. Final/test remains
locked pending the controlled week-26 view and audited final-child activation.

At `2026-08-20T15:57:09Z`, the VK-LSVD `C+P+E` full-rank DCNv2
validation-only seed-43/44 pair had completed scientifically eligible under
clean commit `9fbc43d717577cf0be772b5bc52b7fb8e28ae636`. Both results have zero
OOM/non-finite observations and `test_metrics=null`; all four selected/last
checkpoints passed strict sidecars and CPU finite-tensor reopen. The
credential-free estimate was USD 204.84 including the fixed USD 25 storage
reserve. Durable project storage passed at 22.698 GiB of the 40 GiB cap, with
27.302 GiB logical headroom and 60.378 GiB overlay available. Both A100s were
idle and no task process remained. No test data, provider action, or external
mutation occurred; final/test remains locked pending C+P+E selected lineage
and controlled activation.

At `2026-08-20T16:17:14Z`, the immutable VK-LSVD `C+P+E`
`selected_configs.lock` was independently re-derived and reopened at SHA-256
`b038eee038f925524eb21cc89e495835fb12e347a2bc5afa8c2b5851748cc250`;
the full CPU regression passed 564/564 without test access. Estimated
cumulative spend was USD 206.17. Durable storage passed at 22.699 GiB, with
27.301 GiB logical headroom and 60.343 GiB overlay available. No GPU job,
provider action, test access, or external mutation occurred in publication.

At `2026-08-20T17:07:24Z`, the owner-override multi-target reset
completed its first mandatory cell. `Synthetic-MultiQuotient-3` executed end to
end on one A100: six slices materialized and hashed before training
(manifest `659d29f565bd51338e8c311a5c1c94335a499fbb2f62017d26581e785d8e72fd`),
the capacity control resolved mechanically at parameter ratio 1.00023 and FLOP
ratio 1.00026, validation-only tuning at seed 42 over two learning rates per
arm, selected configurations frozen and hashed
(`63555e1f3b1405643372e4600366067df48bdabb6fabbc58cf406501936e5653`), and final
evaluation across seeds 42/43/44 for eight arms on four slices. Zero non-finite
predictions and zero reciprocal-floor incidence were observed. The matched
serving-performance benchmark ran on the second A100; a first pass under peak
host contention was discarded as corrupted (p99 74.8 ms against a p50 of
0.73 ms) and the retained pass has p50/p95 ratio agreement within 1% for every
pair, recorded as provisional because the host load average was 6.94.

Compute used by this work was roughly 0.4 GPU-hours across both A100s. The
credential-free estimate is USD 208.92 for 55.901791 elapsed Pod-hours
including the fixed USD 25 storage reserve, leaving USD 161.08 of
warning-threshold headroom and USD 251.08 of hard-cap headroom.
Durable project storage was 16.324 GiB of the 40 GiB cap. No provider
action, external mutation, or test access occurred; `test_unlocked=false`
remains in force and the four remaining mandatory datasets are recorded as
`executed: false` in `results/multitask/multitask_index.json`.

## 2026-08-20 corrected Synthetic-MultiQuotient-3 RQ1

Corrected replacement `results/synthetic_multiquotient/rq1_v2/` was executed
on GPU0 after cost, storage, clean-code, process, and GPU-idle gates passed.
GPU1 remained unused. The run completed 9 arms x 3 seeds and published five
successful immutable artifact/sidecar pairs; result SHA-256 is
`292755a33e391729b266372f564e571aa54f734c89ec9723126271cf016f5677`.

At the post-run aggregation check (`2026-08-20T18:25:41.779682Z`), estimated
spend was USD 213.21 including the fixed USD 25 storage reserve, below the USD
370 warning and USD 460 hard cap. Durable project usage was 22.701 GiB of the
40 GiB cap with 27.299 GiB logical headroom and 60.357 GiB overlay free. Both
A100s were idle at 0 MiB/0%, and no synthetic runner process remained.
No provider lifecycle action, external telemetry, or public-dataset test
access occurred; `test_unlocked=false` remains in force.

At `2026-08-20T19:00:32Z`, the RQ2
serving-performance and efficiency study completed on one A100. It trains
nothing and reads no dataset row, label, or test metric: 14 arms per geometry
across two geometries, batches {1, 64, 1024}, 200 warmups, 1,000 synchronized
forwards, 5 independent rounds, with every forward timed twice by wall clock
and by CUDA device events. Median wall/device p95 agreement was 1.012 and the
worst 1.100 against a ceiling of 2.0, so the waived load-average gate is
recorded as `final_load_gate_waived_by_device_time_agreement`. An immutable
pre-measurement manifest (`5de33d29...`) was published before the first timed
forward, and all four artifacts verify against strict exact-byte sidecars.

A defect that would have joined ranker-geometry latency to quotient-geometry
quality was caught before publication; the run was stopped at five minutes, the
partial output directory removed, the join restricted by an explicit
compatibility flag, and a regression test added.

Compute for this study was roughly 0.35 GPU-hours on one A100; the second A100
stayed idle. The credential-free estimate is USD 215.12 for 57.787351
elapsed Pod-hours including the fixed USD 25 storage reserve, leaving
USD 154.88 of warning-threshold headroom and USD 244.88 of
hard-cap headroom. Durable project storage was 16.325 GiB of the 40 GiB cap.
No provider action, external mutation, or test access occurred;
`test_unlocked=false` remains in force.

## 2026-08-21 VK-LSVD seven-head materialization guard

The immutable train/validation-only materializer completed under commit
`83dfe21de90746000544402e17ac07453020b6fc`. Its manifest SHA-256 is
`e880130ae672e97257343e7c1e57c91c5087dd9e303231dd0b738fe6c3b88bf5`
and covers 9,022,651 rows in twelve FULL/REDUCED shards. Independent streaming
hash/schema/row/alignment checks passed. Week 26 was not opened or
materialized, and no model metric was computed.

At `2026-08-21T07:21:19.911701Z`, the unchanged ledger epoch
`2026-08-18T09:13:18Z` yielded 70.133864 elapsed hours and an owner-approved
credential-free estimate of USD 255.74, including the fixed USD 25 storage
reserve. A conservative one-hour mandatory-compute projection is USD 259.03,
below the USD 370 warning and USD 460 hard cap; launch guard status is `ok`.
The compute reserve begins `2026-08-23T01:13:18Z` and the effective stop is
`2026-08-23T09:13:18Z`.

Persistent project usage is 24,375,740,928 bytes (22.701678 GiB) of the 40 GiB
cap, leaving 27.298322 GiB logical headroom. The temporary overlay has
57.975319 GiB free. Both A100s were idle at 0 MiB/0%, the run registry had no
running record, and no task trainer/worker/helper process remained. No
provider credential, external mutation, test access, or Pod lifecycle action
occurred.

At `2026-08-21T07:31:18Z`, the RQ2 fused-inference
addendum completed on one A100. Triton kernels fuse the DCDN carrier, its two
per-layer GEMMs, and its epilogue, and equally fuse the DCNv2 cross epilogue;
each arm is represented by whichever implementation is faster for it. Numerical
equivalence against the audited modules was verified before any timing at
`max|delta| = 1.53e-05` against a `1e-03` tolerance. Fusion speeds DCDN by
1.27-1.43x and slows DCNv2 to 0.91-0.94x, and all four DCDN-versus-DCNv2 pairs
move from failing to passing the runtime-matched threshold. Median wall/device
p95 agreement was 1.014, worst 1.017.

This study trains nothing and reads no dataset row, label, or test metric.
Compute was roughly 0.3 GPU-hours on one A100; the second stayed idle. The
credential-free estimate is USD 256.29 for 70.300025 elapsed Pod-hours
including the fixed USD 25 storage reserve, leaving USD 113.71 of
warning-threshold headroom and USD 203.71 of hard-cap headroom.
Durable project storage was 16.325 GiB of the 40 GiB cap. No
provider action, external mutation, or test access occurred;
`test_unlocked=false` remains in force.

At `2026-08-21T07:58:14Z`, the RQ2 batch sweep
completed on one A100: 15 batch sizes from 1 to 16384, five arms, each under its
best available implementation. Latency proved essentially flat across the range
(+6% for DCNv2, +11% for DCDN against a 16,384-fold work increase), so the model
class is dispatch-bound throughout the production batch range and DCDN's lower
FLOP count never becomes a throughput win. Wall-over-device agreement was 1.01
at every point. Figures were emitted as dependency-free SVG, leaving `uv.lock`
untouched.

This study trains nothing and reads no dataset row, label, or test metric.
Compute was roughly 0.25 GPU-hours on one A100; the second stayed idle. The
credential-free estimate is USD 257.76 for 70.749102 elapsed Pod-hours
including the fixed USD 25 storage reserve, leaving USD 112.24 of
warning-threshold headroom and USD 202.24 of hard-cap headroom. Durable
project storage was 16.326 GiB of the 40 GiB cap. No provider
action, external mutation, or test access occurred; `test_unlocked=false`
remains in force.

At `2026-08-21T08:18:59Z`, the VK-LSVD seven-head runtime checkpoint completed
without GPU use or model metrics. A full CPU regression passed 766 tests. The
immutable twelve-shard train/validation scan SHA-256 is
`e67f72187f73ed9ad37e1a23cdd5c8b679d1fbc3be35a2781968c270a1497bb6`.
Regenerable `C+P` and `C+P+E` caches occupy 3,085,756,941 and 3,852,682,774
bytes on temporary overlay storage and were independently reopened by exact
content hashes. The credential-free estimate is USD 258.90 for 71.094775
elapsed Pod-hours including the fixed USD 25 storage reserve; a conservative
one-hour projection is USD 262.19. Durable project usage is 22.702611 GiB of
the 40 GiB cap and overlay free space is 51.439293 GiB. The reserve begins
`2026-08-23T01:13:18Z`; no provider action, external mutation, test access, or
Pod lifecycle action occurred.

At `2026-08-21T08:27:28Z`, the HiFormer-style
stack study completed on one A100. Four stacks x six batch sizes, cost only, no
training and no quality number. With one attention layer after the cross the
stack becomes compute-bound and the reduced feature view turns into a serving
win: DCDN-REDUCED reaches 19% lower latency and 24% more throughput than
DCNv2-FULL at B=16384, with a crossover between B=256 and B=1024. Two controls
attribute the effect to token count rather than to the reciprocal branch. Worst
wall-over-device agreement was 1.007.

Compute was roughly 0.6 GPU-hours on one A100; the second stayed idle. The
credential-free estimate is USD 259.37 for 71.236306 elapsed Pod-hours
including the fixed USD 25 storage reserve, leaving USD 110.63 of
warning-threshold headroom and USD 200.63 of hard-cap headroom. Durable
project storage was 16.326 GiB of the 40 GiB cap. No provider
action, external mutation, or test access occurred; `test_unlocked=false`
remains in force.

At `2026-08-21T08:26:45.558692Z`, the local mandatory-compute guard estimated
USD 259.33 cumulative and USD 265.91 at a conservative two-hour ETA, below the
USD 370 warning and USD 460 hard cap and before the finalization reserve. The
multi-target encoder/capacity/Protocol freeze then completed CPU-only with no
model metric. Storage remained healthy at 22.702860 GiB of the 40 GiB durable
cap with 51.435146 GiB overlay free. Both A100s were idle, the test remained
locked, and no provider, external, or Pod-lifecycle action occurred.

At `2026-08-21T11:05:20Z`, the KuaiRand-Pure RQ2-quality and RQ3 study completed
on one A100; the second GPU carried a parallel rejected-variant run for part of
the window. Runtime views were materialized twice (the second time to bring the
R block into the frozen contract's column order), and the seven-arm matrix was
executed four times: a two-point learning-rate pilot, a geometric-preconditioner
run that is retained only as the evidence for rejecting that preconditioner, a
pilot before the R reorder, and the final run of seven arms x (four tuning
trials + eight seeds) on validation.

Findings: DCDN separates from both same-input baselines on the frozen primary
target `long_view` (C1 and C2), and from DCNv2 on identical full inputs (C6).
RQ3 is **not** supported - against a capacity-matched DCNv2 the advantage never
separates and reverses sign at eight seeds - and RQ2-quality is a tie rather
than a win. Two arms failed the convergence gate on at least one seed, which
withholds C7 everywhere.

Compute was roughly 1.6 GPU-hours across both A100s. The credential-free
estimate is USD 268.02 for 73.867499 elapsed Pod-hours including the fixed USD
25 storage reserve, leaving USD 101.98 of warning-threshold headroom and USD
191.98 of hard-cap headroom. Durable project usage is 22.704 GiB of the frozen
50 GiB logical envelope with 27.296 GiB of headroom and 50.831 GiB overlay free.
The finalization reserve begins `2026-08-23T01:13:18Z`. No provider action,
external mutation, test access, or Pod lifecycle action occurred;
`test_unlocked=false` remains in force and the locked KuaiRand test window
(365,420 rows) was never read.

At `2026-08-21T11:45:00Z`, the single-head `watch_time` follow-up on
KuaiRand-Pure completed on one A100; the second stayed idle. Seven arms x (four
tuning trials + eight seeds) on validation, with the test split never opened.
Task transfer against the corresponding multi-task head is negative for all
seven arms - five with a bootstrap interval excluding zero - so multi-tasking
costs watch_time quality on this dataset. On this ratio-structured target DCDN
beats DCNv2 by 3.54x at identical reduced inputs and 1.85x at identical full
inputs, both separated; but AFN+ beats DCDN by 2.68-3.02x and RQ3 fails again,
now separating against DCDN in the clean scope.

Compute was roughly 0.4 GPU-hours. The credential-free estimate is USD 270.20
for 74.528541 elapsed Pod-hours including the fixed USD 25 storage reserve,
leaving USD 99.80 of warning-threshold headroom and USD 189.80 of hard-cap
headroom. The finalization reserve begins `2026-08-23T01:13:18Z`. No provider
action, external mutation, test access, or Pod lifecycle action occurred;
`test_unlocked=false` remains in force.

