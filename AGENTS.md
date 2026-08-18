# DCDN project instructions

## Start here

Before doing any work, read `DCDN_HANDOFF.md` completely, inspect `DCDNBlock_provided.py`, inspect `DCDNCore_reference.py`, and inspect the current TeX. Treat the handoff as the durable source of truth, the preserved block as the user-supplied one-layer production-hybrid source, and the reference core as the starting specification for public experiments. The production wrapper/configuration remains incomplete, but this does not block the separately specified public DCDN core. If the manuscript, original notes, code, or a user statement conflicts with the handoff, surface the discrepancy before changing claims.

For any RunPod or other paid remote-compute task, also read `REMOTE_AGENT_PROMPT.md` completely before acting. Its experiment protocol and the safety rules below are mandatory together; when instructions differ, follow the more conservative cost, data, persistence, or external-action boundary and surface any material conflict.

## Naming and paper target

- Method: **DCDN — Deep Cross-Division Network**.
- Paper title: **DCDN: Deep Cross-Division Network for Product-and-Ratio Feature Interactions in Industrial Ranking**.
- Keep two labels explicit: `DCDN-Core-L{L}` for public raw/base-operand experiments and exact Laurent theory; `DCDN-Hybrid-Prod-L1-PLE8` for the deployed one-block PLE integration initialized with `x_i=x_0` and containing the local FC path/projector.
- Target: WSDM 2027.
- Use “Laurent” for the mathematical characterization, not as a replacement model name.
- Communicate with the user in Russian; write manuscript prose in academic English.

## Claim discipline

- Distinguish fixed-width monomial-support completeness from coefficient completeness of the widened construction.
- Qualify Laurent claims by exact reciprocals, nonzero valid denominators, mask, depth, density, and generic non-cancellation assumptions.
- Never claim that the deployed fixed-width model represents every finite Laurent polynomial.
- Never claim that DCDN is the first neural model capable of division.
- Treat stabilized reciprocal operators separately from exact Laurent theory.
- Say DCDN recovers the **predictive value** of removed engineered features unless direct feature reconstruction has been demonstrated.
- In paper-facing prose, describe the production feature reduction as **40.1%** without disclosing the absolute feature counts. Absolute counts may remain in internal experiment identifiers and private reproducibility records.
- State that DCDN is deployed in the production ranking stack of the major short-video service; do not weaken this verified deployment fact to a merely proposed or offline-only treatment.
- Do not say that PLE preserves multiplication or division. It preserves structured feature information, while the production cross-division path is Laurent in lifted PLE coordinates; its raw-space pullback and the complete stabilized nonlinear hybrid are generally piecewise rational.
- The explicit production FC path is not part of the theorem in `DCNvsDDN.tex`. Its `FCN` notation refers to induced affine terms. Put the actual FC path in Production Integration/Industrial Lessons and evaluate it through a division-by-FC ablation.
- Do not convert missing A/B confidence intervals, offline values, implementation details, or proposed public-dataset features into facts.
- A nonsignificant change in clips shown is not proof of equality.

## Experimental requirements

- Keep the industrial causal core: `DCNv2-187`, `DCNv2-112`, production-hybrid `DCDN-112`, and `Hybrid-second-product-control-112`, which preserves the exact wrapper but replaces the reciprocal carrier by a second product carrier.
- Never use `DCDN-sham`. Call the public raw-core rank/width control `DCNv2-capacity-matched-core`; call the exact production/bridge control `Hybrid-second-product-control`.
- Explicit reciprocal inputs are an offline-only optional control; no extra A/B arm is required.
- Keep the public-dataset scope frozen: VK-LSVD and KuaiRand-Pure are core; KuaiRec big matrix is the mechanistic ablation; Yambda-50M is optional only after all mandatory tables are complete.
- HEGM currently supports exactly VK-LSVD `up0.01_ir0.01` and KuaiRec big matrix. Its released processed views are not sufficient for a strict `C+P` versus `C+P+E/R` comparison; rebuild the matched base-operand and engineered views from allowed raw history.
- On KuaiRand-Pure, use `long_view` as the primary target and the late randomized rows as the headline test. Never use `is_rand` as a feature or claim unbiased evaluation beyond the fixed intervention pool.
- Exclude KuaiRand monthly `video_features_statistic.csv` from the strict causal experiment. On KuaiRec, lag daily aggregates by at least one day. Current-row outcomes are labels, not inputs.
- Run the full eight-model suite only on the shared `C+P` regime of VK-LSVD and KuaiRand-Pure. On KuaiRec prioritize `DCNv2(C+P)`, `DCNv2(C+P+E)`, `DCDN(C+P)`, `DCNv2-capacity-matched-core(C+P)`, and `AFN+(C+P)`.
- Do not add KuaiSAR, EB-NeRD, Criteo/Avazu, Tenrec, MicroLens, or another dataset/model before every mandatory result cell is complete and the user explicitly expands scope.
- Prevent temporal leakage. Every aggregate must precede the target example.
- In public `DCDN-Core`, apply division only to semantically valid raw/base-operand denominator coordinates; categorical, ID, PLE, and content embeddings bypass it. Preserve the confirmed production block separately as an unmasked reciprocal operation over PLE coordinates, and do not silently claim that it uses the public raw-denominator mask.
- Every main public row `DCDN(C+P)` uses `DCDN-Core` with no local FC/BatchNorm/Dropout/concatenation/projector. Shared context encoding and the common prediction head remain outside the core and identical across baselines.
- Match examples, feature regimes, heads, losses, tuning budgets, examples seen, effective batches, and seeds across architecture comparisons.
- Report uncertainty, effect sizes, efficiency, latency, and numerical-stability metrics.

## RunPod execution rules

- Audit actual mounts and free space before downloading. The last shown configuration had a 1TB temporary container disk and a 50GB persistent `/workspace` volume.
- At the start of the ledger, record one immutable UTC timestamp as `ledger_epoch_utc`. The hard envelope is **USD 460 pre-tax**, including compute and storage, with a warning checkpoint at **USD 370**. The Pod may run only until the earlier of **120 wall-clock hours from `ledger_epoch_utc`** and **2026-08-24 14:00 Europe/Moscow**, including idle time; reserve the final 8 hours for aggregation and backup. Do not launch or resume a job when its conservative ETA would cross the time reserve or USD 460 envelope. The owner-provided deployment quote is **USD 3.29 per Pod-hour** for the 2×A100 Pod; use that rate plus a conservative USD 25 storage reserve when control-plane billing data is unavailable, and label the ledger as an estimate.
- Create `ledger_epoch_utc` once and reuse the earliest persisted value across every restart; never reset cumulative spend or time. **Owner override recorded 2026-08-18:** absence of provider-side expiry, live control-plane price/spend/headroom, or credentialed RunPod inspection is not a blocker. Do not use stored RunPod credentials. The owner accepts responsibility for manually terminating the Pod. The agent must still stop every task-owned process by the hard time, preserve and verify the persistent backup, report that the Pod remains billable, and never terminate the Pod or volume autonomously.
- Verify that the repository and run registry reside on persistent storage. If the working repository is on temporary storage, maintain a continuously updated persistent mirror and test that it can be read independently before launching a full run.
- Keep code, manifests, metrics, compact logs, and irreplaceable checkpoints persistent. Large public raw data and regenerable caches may use temporary storage, but never stop the Pod before verifying a durable backup.
- Determine writable capacity and free space from the mounted filesystem itself; missing control-plane volume-quota evidence is not a blocker. The project may consume at most **40GB of durable storage** on the expected 50GB volume, while always leaving at least 10GB free. If the locally observed filesystem is smaller, reduce the project cap so that 20% and at least 5GB remain free. Keep `best` and `last` checkpoints while a run is active. After validation, retain the selected `best` checkpoint for each mandatory result cell; retain `last` only when needed for an active resume or a documented failure investigation. Remove redundant tuning and failed-run checkpoints only after preserving their configs, metrics, logs, and failure evidence. The storage safety cap takes precedence over blanket checkpoint-retention requests.
- Use one independent process per A100; do not use DDP for these tabular models.
- Run deterministic tiny/smoke tests and seed-42 **validation-only** tuning, freeze and hash all selected configurations before any test metric, then evaluate the untouched test for seeds 42, 43, and 44.
- Maintain a run registry, exact reproduction commands, cost ledger, and compact `STATUS.md` checkpoint reports.
- Stop on leakage, schema mismatch, reciprocal-mask violation, repeated non-finite values, inconsistent resume, second OOM, disk pressure, or an unapproved paid/external action.

### RunPod permission and mutation boundaries

- Public reads and downloads explicitly approved by the frozen protocol are allowed. Without explicit user approval, perform **no external write, upload, remote mutation, message, publication, or telemetry** to any service, including GitHub, Hugging Face, S3/GCS, W&B, MLflow, artifact hosts, webhooks, email, or chat. Disable third-party experiment telemetry by default.
- Use an isolated user-local environment and user-writable caches. Do not use `sudo`, replace system Python, alter GPU drivers/CUDA, change firewall or SSH policy, open public network endpoints, or install persistent services without explicit approval.
- Before local Git work, record all pre-existing modified and untracked paths. Stage only task-owned paths explicitly; never use broad staging such as `git add .` or `git add -A`. Before every commit, inspect the staged diff and check it for credentials, tokens, SSH material, `.env` content, private data, generated datasets, checkpoints, and unexpectedly large files. Git pushes and all other remote Git mutations require explicit approval.
- Stopping a **training process/job** is autonomous when required by the failure rules, after its logs and evidence are preserved. Stopping or terminating the **paid Pod** is a separate destructive lifecycle action and remains user-approved. At task completion, stop all training, preprocessing, monitoring, and supervisor jobs; verify zero active task PIDs and no task GPU processes, then report that the Pod is safe for the user to terminate.
- Before declaring durable backup, write a backup manifest containing artifact paths, sizes, and cryptographic hashes. Re-open configs, result tables, the run registry, and every retained mandatory checkpoint from the persistent location and record the successful read test in `STATUS.md`.

### RunPod completion gate

- A valid scientific negative result counts as a completed cell only when the implementation, data manifest, leakage checks, numerical checks, and evaluation all succeeded.
- OOM, timeout, cost exhaustion, missing capacity, corrupted data/checkpoints, code defects, environment failures, or storage failures are execution failures, not scientific results, and do not satisfy a mandatory cell. Preserve the failure report and mark the work blocked or incomplete; only the user may accept such a failure as a terminal substitute for a core result.
- Completion additionally requires: spend below the hard envelope; elapsed time within 120 hours; the final cost ledger reconciled; the durable backup manifest and read test passed; the run registry contains no `running` jobs; zero task processes remain; and `STATUS.md` gives the exact safe user action for Pod termination.

## Manuscript workflow

1. Use the revised full-paper abstract in `DCDN_HANDOFF.md`; it reports only the 40.1% reduction and states that DCDN is deployed in production. Keep the earlier portal wording only in private submission history if needed.
2. Replace the abandoned provisional architecture name with DCDN throughout the working TeX as part of the next intentional manuscript revision.
3. Do not copy the original two-layer derivation without correcting the residual and bias terms described in the handoff.
4. Keep visible placeholders for genuinely missing evidence rather than inventing values.
5. After material decisions or verified results, update `DCDN_HANDOFF.md` and its decision log in the same change.
6. Compile and visually inspect every TeX revision before presenting it as ready.

At the beginning of a new task, first return a short understanding check covering the contribution, safe claims, confirmed evidence, chosen experiments, blockers, and proposed next action. Do not edit until any material discrepancy is resolved.
