# DCDN project handoff

Updated: 2026-08-18 (Europe/Moscow)

This document is the durable source of truth for continuing the DCDN paper in another Codex task or Project. Read it before the manuscript, because the legacy generated draft still contains an abandoned provisional name and placeholder claims.

## 1. Status at a glance

- Target: WSDM 2027 full-paper submission.
- Abstract: submitted.
- Full-paper deadline: 2026-08-24 AoE, i.e. 2026-08-25 14:59 Moscow time. Recheck the official CFP before submission: <https://www.wsdm-conference.org/2027/cffp.html>.
- Final method/title choice:

  **DCDN: Deep Cross-Division Network for Product-and-Ratio Feature Interactions in Industrial Ranking**

- Use **DCDN** exclusively as the architecture name. “Laurent polynomial” describes the induced mathematical function class; it is not a model name.
- Central empirical fact already available on the industrial dataset:

  \[
  Q(\mathrm{DCNv2\!-\!112}) \ll Q(\mathrm{DCNv2\!-\!187})
  < Q(\mathrm{DCDN\!-\!112}).
  \]

  The exact definition and numerical values of \(Q\), uncertainty, split, and training protocol still need to be inserted.
- Central production result: DCDN replaced the DCNv2 block, removed the corresponding engineered features, and reduced the input feature inventory by 40.1%. The resulting DCDN ranker is deployed in the production short-video service.
- Architecture split confirmed by the user on 2026-08-18: the production ranker uses **one** hybrid DCDN block with the auxiliary FC path and final projector shown in `DCDNBlock_provided.py`; public offline experiments use the simplified cross-division core without that FC path or projector.
- Current submission risk is predominantly experimental reporting and causal controls, not lack of prose.

## 2. Source-of-truth priority

When sources disagree, use this order:

1. Explicit user-confirmed facts and decisions in this handoff.
2. The current full-paper abstract below. The industrial arm labels in Section 7 retain the absolute counts only for internal experimental reproducibility.
3. New verified experiment outputs and implementation code.
4. The legacy generated working draft listed below.
5. The original Russian source manuscript.

Do not silently promote proposals, placeholders, or inferred implementation details to facts.

## 3. Files and current artifact state

### Original sources

This handoff is intended to be copied into the directory containing the original DCDN/DCN source files. Once copied, prefer the sibling files in that directory as the canonical working sources.

- PDF supplied by the user: `/Users/m.trapeznikov/Downloads/DCNvsDDN (2).pdf`
- Original TeX: `/Users/m.trapeznikov/Documents/neuralranking_rocketscience/deep_division_network/DCNvsDDN.tex`
- Original notebooks/code: `/Users/m.trapeznikov/Documents/neuralranking_rocketscience/deep_division_network/`
- User-supplied one-layer production hybrid block: `/Users/m.trapeznikov/Documents/Codex/2026-08-17/new-chat/DCDNBlock_provided.py`
- Theorem-aligned public offline reference core: `/Users/m.trapeznikov/Documents/Codex/2026-08-17/new-chat/DCDNCore_reference.py`

### Generated working draft

- TeX: `/Users/m.trapeznikov/Documents/Codex/2026-08-17/new-chat/outputs/laurent_cross_networks.tex`
- bibliography: `/Users/m.trapeznikov/Documents/Codex/2026-08-17/new-chat/outputs/laurent_cross_networks.bib`
- compiled PDF: `/Users/m.trapeznikov/Documents/Codex/2026-08-17/new-chat/outputs/laurent_cross_networks.pdf`
- Word draft: `/Users/m.trapeznikov/Documents/Codex/2026-08-17/new-chat/outputs/laurent_cross_networks_working_draft.docx`
- generated figures/assets: `/Users/m.trapeznikov/Documents/Codex/2026-08-17/new-chat/work/generated_manuscript_assets/`

The generated TeX is useful scaffolding, not a submission-ready paper:

- its title and notation use an abandoned provisional architecture name and must be changed to **DCDN** throughout;
- its abstract is a stale working-draft abstract, not the revised full-paper abstract in Section 4;
- it contains 32 `\tbd{...}` macro occurrences and many individual unfilled result/table fields;
- most missing evidence is in Experimental Design, Results, deployment details, and the exact production implementation;
- its theory appendix is substantially more mature than its empirical sections, but must still be audited against the deployed architecture.

## 4. Current full-paper abstract

This is the paper-facing revision selected after the portal abstract submission. It intentionally reports only the relative 40.1% feature-inventory reduction and explicitly states that DCDN is deployed in the production short-video service. Keep the absolute inventory counts only in internal experiment labels, tables, and the private reproducibility record.

> Learning effective feature interactions is essential for industrial recommender systems. DCN and DCNv2 efficiently automate polynomial feature crossing. However, their cross layers construct interactions using only non-negative integer powers of the input features, with the maximum degree controlled by network depth. Consequently, they cannot directly represent the reciprocal terms required for ratios, rates, and normalized counts unless such features are explicitly provided as inputs.
>
> We introduce DCDN, a Deep Cross-Division Network that augments a DCNv2-style cross layer with a division branch. DCDN jointly learns product, ratio, and mixed product–ratio interactions. We characterize its cross-division core using multivariate Laurent polynomials and show that it can generate Laurent monomials of every interaction order up to a depth-controlled maximum.
>
> We evaluated DCDN against a production-optimized neural-ranking baseline built around DCNv2 and supplied with manually engineered ratio features. Replacing the DCNv2 block with DCDN allowed us to eliminate this entire feature class from the model input, reducing the input feature inventory by 40.1%. DCDN is deployed in the production ranking stack of a major short-video service. In a live A/B test on this service, DCDN increased likes by 2.7%, shares by 16.4%, and total view time by 0.8%, while reducing dislikes by 4.3%; all changes were statistically significant. The number of short videos shown to users remained stable. These results demonstrate that DCDN can improve user engagement while eliminating reliance on manually engineered ratio features.

Notes for the full paper:

- `production-optimized` is acceptable if the incumbent was genuinely tuned and deployed. Never say `optimal`.
- In paper-facing prose, report only the 40.1% reduction; retain the absolute feature counts solely where they are required to define internal experimental arms or confidential reproducibility artifacts.
- The paper should replace broad expressivity prose with the assumption-qualified theorem statements in Section 6 below.
- The abstract's “remained stable” should be supported in the paper by the actual \(-0.3\%\) estimate and uncertainty. A nonsignificant result does not establish exact equality.

## 5. Central paper narrative

The intended DCN/DCNv2-style narrative is:

1. Industrial rankers rely on learned feature interactions, but also consume materialized rates, normalized counts, and other ratios.
2. DCN/DCNv2 cross layers naturally build positive-power polynomial interactions; reciprocal terms must normally be supplied explicitly.
3. DCDN adds a reciprocal/division cross branch to a DCNv2-style recurrence, creating a structured bias for product, ratio, and mixed product-ratio interactions.
4. The resulting exact symbolic interactions are Laurent monomials on nonzero denominator domains.
5. DCDN is deployed in the production ranking stack of the short-video service, where it reduces the input feature inventory by 40.1%, improves the measured engagement outcomes, and slightly increases total view time.
6. Controlled offline experiments must establish that the gain is caused by the reciprocal structure rather than only extra parameters, tuning, or an irrelevant feature removal.

Defensible novelty is the package:

- a drop-in DCNv2-style reciprocal cross operator;
- an exact, assumption-qualified Laurent support analysis;
- controlled feature-substitution evidence;
- a 40.1% production feature-inventory reduction;
- a live short-video A/B test.

Do not claim that DCDN is the first neural model capable of learning division. AFN/AFN+, NALU/NPU, EulerNet-like approaches, and ordinary networks supplied with reciprocal inputs are relevant prior art or controls.

## 6. Architecture and mathematical claim boundaries

### Conceptual recurrence

The working mathematical form is

\[
h^{(0)}=x,
\]

\[
h^{(\ell+1)} = h^{(\ell)}
+ x\odot\left(W_\ell^+h^{(\ell)}+b_\ell^+\right)
+ m\odot\rho(x)\odot\left(W_\ell^-h^{(\ell)}+b_\ell^-\right),
\]

This is the canonical DCDN core. In public raw-feature experiments, \(m\) enables division only for semantically valid denominator coordinates and \(\rho\) is either the exact reciprocal or the declared stabilizer. The production integration instead applies its stabilizer to all PLE coordinates and uses one hybrid block with a parallel FC path and final projection. The exact PLE preprocessing, remaining tower, loss, and head still require confirmation from the complete production configuration.

The reciprocal bias is mathematically important: it makes pure terms such as \(1/x_j\) reachable. If the deployed implementation omits this bias, the theorem and wording must be adjusted.

### User-supplied production hybrid block

On 2026-08-18 the user supplied `DCDNBlock_provided.py` as the production block implementation. With feature width \(F\), its cross-division path is

\[
c_i = x_i + x_0\odot(W_i^+x_i+b_i^+)
      + \rho_\varepsilon(x_0)\odot(W_i^-x_i+b_i^-),
\]

followed by a parallel nonlinear path and projection,

\[
h_{i+1}=P_i\,[\,c_i\,;\,\operatorname{Dropout}(\operatorname{BN}(\operatorname{LeakyReLU}(W_i^f x_i+b_i^f)))\,]+b_i^P.
\]

The supplied stabilizer is the coordinatewise sign-preserving hard floor

\[
\rho_\varepsilon(t)=
\frac{1}{t},\quad |t|>\varepsilon,
\qquad
\rho_\varepsilon(t)=\frac{1}{\operatorname{sign}_+(t)\varepsilon},\quad |t|\le\varepsilon,
\]

where zero is assigned positive sign and the default is \(\varepsilon=10^{-2}\). The division map is an unfactorized dense square parameterization (generically, but not necessarily after training, full rank), includes a bias, is Kaiming-initialized, and is scaled by `div_init_scale=0.1`. The positive cross matrix also includes a bias. The deep branch has width 32, LeakyReLU, `BatchNorm1d(8)`, and dropout 0.1; a final linear projector maps the concatenated \(F+32\) channels back to \(F\).

The user subsequently confirmed that this is the production block, that production uses one such block, initializes `x_i=x_0`, keeps `x_0` fixed, and that the FC path/projector belong only to the production integration. Its input contract is \(x_0,x_i\in\mathbb R^{B\times 8\times F}\): the middle dimension is the eight-dimensional piecewise-linear encoding (PLE) of each feature, while the final dimension indexes features. Each linear map mixes the \(F\) features independently inside one PLE coordinate slice. Treat the file as **user-supplied production-architecture source**. The remaining end-to-end details to record are the exact PLE construction and clipping range, feature width and ordering, input preprocessing, tower/head, loss, and production training configuration.

The supplied production block has no semantic denominator mask: the reciprocal stabilizer is applied to every PLE coordinate of `x_0`. Do not retrofit the raw-operand mask into the preserved production source. The masked reciprocal policy described below belongs to the public raw-feature DCDN core.

Because the production wrapper initializes `x_i=x_0`, the exact Laurent-polynomial theorem applies directly to the **cross-division path in PLE coordinates** when the stabilizer is replaced by exact reciprocals and the reciprocal coordinates are nonzero. The supplied hard-floor operator is only piecewise reciprocal, and the LeakyReLU/BatchNorm/Dropout path means the complete training-time block is not a global Laurent polynomial. At deterministic inference, after BatchNorm statistics are frozen and dropout is disabled, the full block is piecewise Laurent in the lifted PLE coordinates over fixed activation and stabilizer regions.

The supplied block has \(3F^2+67F+48\) trainable parameters. It changes capacity and topology relative to a bare dense DCNv2 cross layer by adding the division map, the width-32 nonlinear path, and the projector. Therefore a mechanism comparison must retain the same input layout, nonlinear path, projector, head, and training protocol while ablating or replacing only the reciprocal mechanism; a bare DCNv2 block is not by itself a capacity-isolating control.

### Architecture split used in the paper

Keep the following variants explicit; do not silently treat them as identical implementations.

| Variant | Definition | Role |
|---|---|---|
| **DCDN core** | residual + product cross + reciprocal cross; no auxiliary FC path and no concatenation projector | canonical method for the theory and all public offline experiments |
| **Production DCDN** | exactly one DCDN cross-division core in parallel with the width-32 FC/LeakyReLU/BatchNorm/Dropout path, followed by the learned projector | deployed industrial model and live A/B evidence only |

The offline core recurrence is

\[
h^{(\ell+1)} = h^{(\ell)}
+ x_0\odot(W_\ell^+h^{(\ell)}+b_\ell^+)
+ m\odot\rho(x_0)\odot(W_\ell^-h^{(\ell)}+b_\ell^-),
\]

with no `fc_block`, concatenation, or `projector` between DCDN layers. Public experiments may vary the number of core layers as a predeclared hyperparameter. The production configuration fixes this depth to one and then mixes the one-layer cross-division representation with the auxiliary nonlinear path.

Consequently, the production model **uses a one-layer Laurent-structured cross-division operator**. Since its wrapper initializes `x_i=x_0`, the cross-division representation is a Laurent polynomial in the PLE coordinates before reciprocal stabilization. The complete hybrid output must not be called a global Laurent polynomial: the hard reciprocal floor and LeakyReLU make it only piecewise Laurent at deterministic inference. The exact Laurent closure/support theorems describe the idealized offline core with exact reciprocals on its valid nonzero domain. The live A/B test evaluates the full one-layer hybrid production package, whereas the public offline experiments isolate the cross-division mechanism. State this evidence boundary explicitly.

#### PLE-space versus raw-feature algebra

Let \(z\in\mathbb R^F\) denote raw numerical features and let \(u=\phi_{\mathrm{PLE}}(z)\in\mathbb R^{8\times F}\) be their production encoding. PLE is a structured piecewise-linear lift and can retain enough information to recover a raw value over its non-clipped range, but it is **not an algebra homomorphism**: in general, \(\phi(xy)\ne\phi(x)\phi(y)\) and \(\phi(x/y)\ne\phi(x)/\phi(y)\).

Under the confirmed production initialization `x_i=x_0=u`, the production cross-division path is Laurent in the lifted coordinates \(u\), separately within each of the eight PLE slices, wherever exact reciprocal coordinates are nonzero. On a fixed PLE cell, each coordinate has the affine form \(u_{rj}=a_{rj}z_j+b_{rj}\), so pulling a Laurent term back to raw space produces terms of the form

\[
\prod_j(a_{rj}z_j+b_{rj})^{\alpha_j},
\]

which are generally rational and piecewise-defined in the raw features rather than Laurent monomials in \(z\). The production stabilizer and nonlinear side path further make the complete inference block a stabilized piecewise-rational map of the raw inputs. The safe interpretation is that PLE preserves structured information about each source feature and lets DCDN learn product-and-ratio interactions **in the lifted feature space**; do not claim that quotients of PLE coordinates literally equal the engineered raw-feature ratios.

For public offline experiments, apply the DCDN core directly to named, positively scaled raw/base operands. With exact reciprocals on nonzero operands—or on an evaluated domain where the declared floor never activates—the raw-space Laurent interpretation is direct. With an active stabilizer it is only piecewise Laurent/rational. The explicit ratio-substitution experiment remains direct in either case. This creates a deliberate two-level evaluation: raw operands isolate the algebraic mechanism; the PLE hybrid documents the production integration.

#### Audit of the original `DCNvsDDN.tex`

The original theorem and induction do **not** include the explicit production FC path. The DCDN definition is the pure residual/product/division recurrence, and its diagram combines those branches by addition. The source mentions parallel FC stacks only as general DCN background. Its later `FCN` label denotes affine terms induced by repeated cross/division operations; it is not the width-32 LeakyReLU/BatchNorm/Dropout branch in `DCDNBlock_provided.py`. Therefore the theorem-aligned public architecture is the DCDN core without `fc_block` or `projector`, while the explicit FC path belongs in the Production Integration and Industrial Lessons sections.

The old induction must not be copied verbatim: its second-layer expansion incorrectly folds the residual into a bias-scaled hidden state. Use the corrected Laurent-support proof in the new appendix.

### Safe claims

- In the idealized cross-division core—exact reciprocals, nonzero enabled denominators, and no intervening nonlinear side path—every hidden coordinate is a multivariate Laurent polynomial. The full supplied hybrid block is generally only piecewise Laurent at deterministic inference.
- Increasing depth expands the set of structurally reachable Laurent-monomial orders; the maximum order is controlled by depth.
- For dense projections and generic parameters, the layer has an exact support recurrence. Accidental coefficient cancellation can occur on exceptional parameter choices.
- With a full denominator mask, the fixed-width network is **monomial-support complete** within the depth-controlled reachable set. This is structural reachability, not independent coefficient control.
- With a selective public/core mask, negative powers are reachable only along denominator-eligible coordinates. State the theorem for the admissible mask or clearly label the full-mask theorem as an idealized result. The supplied production PLE block is unmasked.
- A separate widened basis-channel construction can exactly represent an arbitrary finite Laurent polynomial with enough auxiliary channels and depth. This widened construction is a theorem device unless it is actually deployed.

For the full reciprocal mask, the current draft states the generic output-support union after \(L\) layers as

\[
U_L=(E+\mathcal B_L)\cup(-E+\mathcal B_{L-1}),
\qquad
\mathcal B_r=\{\alpha\in\mathbb Z^d:\|\alpha\|_1\le r\}.
\]

This implies a depth-controlled finite support and contains the Laurent ball \(\mathcal B_L\); it does not imply arbitrary independent coefficients for all those monomials.

### Claims to avoid

- Do not say the deployed fixed-width block can represent **any Laurent polynomial**.
- Do not conflate support completeness with coefficient completeness.
- Do not say the model learns arbitrary rational functions such as \(1/(x_1+x_2)\). Its exact core yields Laurent polynomials with monomial denominators.
- Do not apply a full-mask theorem to a selectively masked implementation without restriction.
- Do not call a stabilized operator globally Laurent. For example, \(1/(x+\varepsilon)\) or \(x/(x^2+\varepsilon^2)\) is only an approximation to \(1/x\) away from zero.
- Do not claim literal reconstruction of every removed feature until the 75-feature taxonomy confirms that all required primitive operands and business logic remain available.

### Known source-manuscript issue

The original two-layer symbolic expansion contains an algebraic error. The second layer must include the residual \(h_j^{(1)}\) and separate bias term \(\widetilde b_j x_j\):

\[
h_j^{(2)}=h_j^{(1)}
+x_j(\widetilde W h^{(1)}+\widetilde b)_j
+x_j^{-1}(\overline W h^{(1)}+\overline b)_j.
\]

Do not replace the residual by a factor such as \((\widetilde b_j+1)h_j^{(1)}\). Re-audit any proof or explicit expansion copied from the original TeX.

## 7. Confirmed industrial evidence

### Offline

- Baseline: production neural ranker built around DCNv2 with 187 inputs, including manually engineered rate/ratio features.
- Reduced baseline: the same intended DCNv2 setup with 112 retained primitive inputs.
- Treatment: DCDN with 112 retained inputs.
- User-reported ordering:

  \[
  Q(\mathrm{DCNv2}_{112}) \ll Q(\mathrm{DCNv2}_{187})
  < Q(\mathrm{DCDN}_{112}).
  \]

Still needed: metric name and direction, exact values, variance/CI, dataset size, split dates, seeds, tuning budget, and confirmation of which model components were held fixed.

### Online A/B test

User-reported relative changes:

| Metric | Relative change | Reported inference |
|---|---:|---|
| Likes | +2.7% | statistically significant |
| Shares | +16.4% | statistically significant |
| Dislikes | −4.3% | statistically significant; lower is beneficial |
| Total view time | +0.8% | statistically significant; small positive lift |
| Number of short videos shown | −0.3% | not statistically significant |

Preferred prose for the last metric:

> We observed no statistically detectable change in the number of short videos shown to users.

Do not say exact invariance or equivalence unless a pre-specified equivalence/non-inferiority test supports it. Keep statistical significance separate from internal practical-significance thresholds: a +0.8% total-view-time estimate can be statistically significant while below an internal +1% materiality threshold.

The A/B test establishes the value of the complete treatment package—DCDN plus the reduced feature set. Offline controls are required to isolate the division branch.

Still required for publication:

- start/end date and duration;
- randomization unit and traffic allocation;
- number of users/sessions/impressions;
- exact metric definitions and denominators;
- confidence intervals or p-values;
- repeated-observation/cluster treatment;
- fixed-horizon versus sequential procedure;
- multiplicity policy and primary metric;
- SRM and balance checks;
- interpretation of the internal materiality thresholds;
- online latency and safety/quality guardrails.

## 8. Research questions

Use these four RQs consistently:

- **RQ1 — Feature substitution.** Can DCDN recover the predictive value of manually engineered ratio features from retained numerator and denominator base operands?
- **RQ2 — Ranking effectiveness.** Given the same base operands, context features, and training/serving budget, does DCDN outperform DCNv2 and representative feature-interaction models?
- **RQ3 — Reciprocal mechanism and production integration.** Are the gains attributable to reciprocal interactions rather than additional parameters, compute, or the parallel FC path? The public core ablation isolates division on raw operands; a compact industrial PLE-space ablation separates division, FC, and their interaction. Explicit reciprocal-input lifting remains an optional reviewer control.
- **RQ4 — Production utility and cross-setting consistency.** Does the deployed DCDN package reduce feature-pipeline complexity while remaining stable and efficient, and are its production outcomes directionally consistent with the separately measured offline mechanism evidence? Do not present this as an identified causal translation from public offline metrics to online lift.

Use “recover the predictive value” rather than “reconstruct every removed feature” unless direct reconstruction probes are actually run.

## 9. Required experimental matrix

### Industrial causal core

| Variant | Feature set | Purpose | Priority |
|---|---:|---|---|
| DCNv2-full | 187 (primitives + engineered) | production-quality baseline | mandatory |
| DCNv2-primitives | 112 | cost of removing the 75 features | mandatory |
| DCDN-primitives | 112 | main treatment | mandatory |
| Hybrid-second-product-control | 112 | controls for the extra active branch in the exact production wrapper | mandatory for production-mechanism attribution |
| DCDN-full | 187 | tests architecture benefit without compression | optional |

Do not use the name `DCDN-sham`. For public raw-core tables, use the user-facing/reporting label **DCNv2-capacity-matched**. Retain `dcnv2_capacity_matched_core` as the stable internal key and **DCNv2-capacity-matched-core** only as a historical frozen provenance alias; do not rewrite or re-hash existing search/view locks solely for this naming update. If ordinary DCNv2 is already within ±5% in trainable non-embedding parameters and FLOPs/example, reuse it; otherwise choose a predeclared bracketing rank mechanically by the frozen parameter/FLOP-distance rule, never by validation quality.

This public control is distinct from the production **Hybrid-second-product-control** below. The latter preserves the exact hybrid topology and replaces only the reciprocal carrier with a second product carrier; it is the direct test of reciprocal structure versus another active matrix/compute path.

### Production integration ablation

The public main tables deliberately omit the local FC path. To support the industrial claim that division is the essential architectural change rather than the parallel DCNv2-style FC integration, run a compact factorial ablation on the industrial offline dataset using the same PLE-8 inputs, one fixed-anchor block, prediction head, projector geometry, optimizer budget, and seeds:

| Division branch | FC branch | Interpretation |
|---:|---:|---|
| off | off | product/residual base inside the common wrapper |
| off | on | DCNv2-style parallel FC integration without division |
| on | off | cross-division core inside the common wrapper |
| on | on | full deployed hybrid topology |

Implement branch removal with explicit gates and zero tensors of the same shapes so that the projector and downstream geometry remain unchanged; retrain every row. Add a fifth **Hybrid-second-product-control** row with the FC path and projector retained: replace \(\rho(x_0)\odot(W^-x_i+b^-)\) by \(x_0\odot(W^-x_i+b^-)\). It has the same active branch count, topology, parameters, and approximately the same FLOPs as the production hybrid, while removing reciprocal structure.

Primary contrasts are:

- `full − FC-only`: division benefit conditional on the production FC path;
- `full − division-only`: incremental value of the FC path;
- `(full − FC-only) − (division-only − base)`: division-by-FC interaction;
- `full − Hybrid-second-product-control`: reciprocal carrier versus an equally budgeted second product mechanism.

The minimum acceptable attribution evidence is `full` versus `FC-only` plus `Hybrid-second-product-control`. Add `division-only` to substantiate the statement that the FC integration is secondary. No additional online arm is required. If the industrial ablation cannot be run and the paper wishes to attribute production gains specifically to division, reproduce the four factorial cells **plus the fifth second-product control** on one public dataset with the production PLE-8 wrapper, keeping them separate from the raw-core main table. If neither attribution experiment is available, restrict the production conclusion to the utility of the complete deployed package and make the division-mechanism claim only from raw-core public evidence.

### Explicit reciprocal input control

`DCNv2(C + P + safe reciprocal lifts)` is an offline-only reviewer control answering “why not compute \(1/x\) inside input preprocessing and use ordinary DCNv2?”. It does **not** require a production A/B test or 75 newly materialized feature-store fields.

Priority:

- lower than the industrial three-way comparison and capacity-matched DCNv2;
- preferably run on VK-LSVD and/or a synthetic benchmark;
- compare quality, input dimension, parameters, FLOPs, latency, and sample efficiency;
- frame DCDN's possible advantage as structured inductive bias and pipeline simplicity, not fundamental representational impossibility for lifted DCNv2.

This lifting control is optional evidence for RQ3. The capacity comparison itself is mandatory; it may reuse ordinary DCNv2 when that row already satisfies the frozen ±5% parameter/FLOP tolerance.

## 10. Public datasets and frozen roles

### What HEGM actually provides

The current HEGM repository (<https://github.com/rw404/HEGM>) supports exactly two public-data pipelines:

- KuaiRec **big matrix**;
- VK-LSVD `up0.01_ir0.01`.

The `k={3,6,9,12}` HEGM configurations vary the number of mixture components; they are not additional datasets. HEGM does not currently contain KuaiRand or Yambda preprocessing. Its released processed parquet files are useful for reproducing HEGM, but they are not sufficient for the DCDN feature-substitution experiment: VK-LSVD already contains engineered rates without every matched base operand, while KuaiRec retains only a small numeric view. Rebuild explicit `C+P` and `C+P+E/R` views from the permitted raw history.

Use the following notation throughout the public experiments:

- **C:** shared context, static, categorical, and other non-ratio fields; C is identical across the feature-substitution arms and never enters DCDN's reciprocal branch;
- **P:** retained positive or safely shifted **base operands** from which ratios can be formed, together with required support/validity indicators;
- **E/R:** explicit engineered ratio, rate, average, normalization, or cosine fields removed in the DCDN substitution arm.

Unless explicitly labeled otherwise, every public-table row written as `DCDN(C+P)` means the **DCDN core without the production FC path/projector**, operating on the declared raw or safely shifted numerical operands in `P`. Shared categorical/context fields in `C` use the common encoder and bypass the reciprocal carrier. The production hybrid is not part of the public offline architecture suite. This keeps the public comparison focused on the cross-division operator and aligns it with the Laurent-support analysis. Public offline results and the live A/B result therefore support complementary—not identical—architectures; do not call a public run a literal reproduction of the deployed ranker.

Do not call every member of P a raw primitive. Shifted counts and iALS dot products/norms are derived base operands even though they are not explicit divisions.

### VK-LSVD — primary scalable public benchmark

Reuse the HEGM preprocessing/split as a starting point, not its embedding-to-DCN architecture unchanged:

- official dataset: <https://huggingface.co/datasets/deepvk/VK-LSVD>;
- selected subset: approximately 38.4M interactions, 100K users, and 171K items;
- history: weeks 0–19;
- train: weeks 20–24;
- validation: week 25;
- test: week 26;
- compute user/item aggregates and fit the three iALS models on history weeks 0–19, then freeze them;
- use the fixed HEGM smoothing constants `alpha = 5` and `beta = 100`; they are protocol constants, not quantities fit on validation or test;
- the released HEGM preprocessor obtains `c` from the 99th percentile of full item-metadata **duration** and uses `clip(timespent, 0, c) / c` as the target (`c` is approximately 157 seconds); despite the README wording, the current code does not estimate `c` from watch time;
- as a reproduction fact, the released HEGM implementation fits PLE numerical quantile edges and categorical cardinalities on processed train weeks 20–24. **Do not feed that PLE representation to the mandatory raw-operand DCDN core.** For the core, fit positive scale-only numerical normalization and categorical `C` mappings on train, then freeze them. Fit new PLE edges only for a separately labeled optional production-hybrid bridge or exact HEGM-architecture reproduction. Record the exact `meta.json` value of `c`; if a stricter history/train-only constant is used, label it as a protocol change rather than exact HEGM reproduction.

Proposed public feature-substitution regimes, to validate and freeze in code before reporting counts:

- **C, 6 shared context fields:** `age`, `gender`, `geo`, `place`, `platform`, and `agent`.
- **P0, 31 candidate base operands before validity indicators:** duration; user/item raw shows, safe `shows + 100` denominators, total watch time, shifted event counts such as `count + 5`; for each of three iALS signals, dot product and user/item norms.
- **V, code-validated support/validity indicators:** missing-history, unseen-factor, and zero/floored-denominator indicators required by the implemented safe-division policy. Define `P = P0 + V` and report the exact count from the manifest.
- **E, 20 candidate engineered features:** average watch time, user/item event rates and survival rates, and three iALS cosine similarities.

Core comparison:

- `DCNv2(C+P)`;
- `DCNv2(C+P+E)`;
- `DCDN(C+P)`;
- resolved `DCNv2-capacity-matched(C+P)`, reusing ordinary DCNv2 when it already satisfies the frozen ±5% tolerance.

The provisional `31` and `20` counts imply a 39.2% reduction **only inside the numerical operand-plus-engineered subspace**: `51 -> 31`, before validity indicators. They do not imply a 39.2% reduction in all public input fields. Including the six shared C fields gives `57 -> 37`, or 35.1%, before validity indicators; with `|V| = v`, the honest total-field comparison is `(57 + v) -> (37 + v)`, a reduction of `20 / (57 + v)`. Report both the subspace and total-field definitions, and do not compare either percentage with the industrial 40.1% until the code-generated manifest fixes C, P, V, and E exactly.

Tasks:

- primary: watch-time regression with the code-defined `clip(timespent, 0, c) / c` target, using the exact released or explicitly refit `c`; report MSE in sec² and MAE in seconds;
- separately preregistered post-mandatory multi-task extension: keep normalized
  watch time as the primary head and add like, dislike, share, and bookmark
  binary heads only when the raw schema validates each label; current-row
  outcomes remain labels and never become inputs. For every binary target,
  report ROC-AUC, LogLoss, AUPRC, and user-GAUC. Compute user-GAUC as the
  unweighted mean of per-user ROC-AUC over users having both positive and
  negative labels for that target; report eligible-user and eligible-row/event
  coverage, and use paired user bootstrap 95% confidence intervals. An
  impression-weighted average may appear only as a labeled sensitivity check.
  This extension must not alter the frozen single-target RQ1 selections or
  unlock test access.

The current HEGM evaluator's “XAUC” is a pairwise concordance computed only within contiguous evaluation batches, so it can depend on row order and batch boundaries. If reproduced, call it **HEGM batch-XAUC**, lock batch size/order, and audit permutation and batch-size sensitivity; do not present it as user-GAUC or global concordance. A standard, explicitly defined global or user-grouped ranking metric may be reported separately.

Retain `user_id`, `item_id`, week/timestamp, source split, and stable row ID as audit-only columns in the rebuilt views. They must not silently become model features, but they are required for leakage checks, user-GAUC, paired user bootstrap, cold-entity coverage, and exact row alignment across arms.

To reproduce HEGM smoothing exactly, expose shifted base operands such as `(count + 5)` and `(shows + 100)`. Raw `count` and `shows` alone do not make `(count + 5)/(shows + 100)` a finite Laurent monomial in those raw variables.

### KuaiRand-Pure — second core benchmark and randomized-exposure test

Official dataset: <https://github.com/chongminggao/KuaiRand>.

Use **KuaiRand-Pure** for the deadline experiment:

- 27,285 users;
- approximately 1.44M standard-policy candidate-pool interactions;
- approximately 1.19M randomized exposures;
- 7,583 videos in the intervention pool;
- exact timestamps and 12 feedback signals;
- approximately 194MB packaged data.

Do not use KuaiRand-1K as the headline randomized test: it retains approximately 11.7M standard interactions but only about 43K randomized rows. KuaiRand-27K is an optional later scaling run: its 322M standard rows are not required for the submission grid. If used, stream them to construct histories and run only predeclared finalist models.

Pre-register the following temporal protocol before inspecting model results, after validating that the released rows cover every stated date:

- history/feature burn-in: standard-policy events from 2022-04-08 through 2022-04-21;
- training labels: standard-policy events from 2022-04-22 through 2022-05-02;
- validation: randomized events from 2022-05-03 through 2022-05-05;
- headline test: randomized events from 2022-05-06 through 2022-05-08;
- secondary in-policy test: standard-policy events from the same final dates.

Freeze the KuaiRand rolling-state policy before validation/test access:

- initialize state from the predeclared standard-policy history and training rows through 2022-05-02;
- keep model weights, feature definitions, smoothing/floor constants, aggregation keys/windows, and state-update code frozen after training;
- for validation and both test strata, score every timestamp group against state containing only rows with `time_ms < t`; after all rows at timestamp `t` are scored, update the common state with the released standard and randomized events at `t`;
- retain source file and original row offset. Deduplicate only canonically byte-identical normalized raw rows using a full-row fingerprint; never collapse distinct events merely because `(user_id, video_id, time_ms)` matches;
- order timestamp groups by `time_ms`; within a post-score update group, use a deterministic `(policy, source_file, source_row)` order. Equal-timestamp rows must never enter one another's features;
- this is a frozen rolling/prequential simulation, so earlier validation/test outcomes may update later aggregate state but may not change weights, feature definitions, hyperparameters, thresholds, or the protocol hash. Report this explicitly and include a frozen-state-at-training-cutoff sensitivity slice if feasible.

Because KuaiRand-Pure contains only candidate-pool rows, call these **candidate-pool histories**, not complete user histories.

Feature regimes:

- **C:** shared safe static/context fields available at scoring time; never include `is_rand`.
- **P:** smoothed positive numerator/denominator base operands for historical exposures, long views, likes, forwards, hates, follows, play-time sums, duration sums, and dwell-time sums; construct supported user/item/author/tab groups and predeclared 1-day, 7-day, and all-history windows.
- **R:** explicit lagged rates such as long-view/exposure, like/exposure, forward/exposure, hate/exposure, play-time/exposure, play-time/duration, short-window/long-window, and supported preference shares.

Headline comparison:

- `DCNv2(C+P)`;
- `DCNv2(C+P+R)`;
- `DCDN(C+P)`;
- `DCNv2-capacity-matched(C+P)`;
- AFN+ as the closest algebraic prior.

Primary target: `long_view`, with LogLoss as the primary metric and ROC-AUC/user-GAUC as secondary metrics. Use AUPRC for rare like/forward/hate targets. Treat `is_click` only as a per-UI/tab secondary target because its released semantics differ between the two-column and single-column interfaces.

Do not use `video_features_statistic.csv` in the strict causal experiment: the documented statistics overlap the collection month and can include current or future outcomes. Use only safe static metadata and prefix aggregates rebuilt from event logs. Never feed `is_rand` as a feature; use it only to define evaluation strata.

The standard-policy candidate-pool log contains 7,551 videos whereas the randomized log contains 7,583. Keep randomized rows whose item is unseen in standard training: map ID inputs to a declared OOV token, use zero-history values plus validity indicators, and report cold-item coverage and warm/cold metric slices. Never inner-drop these rows from the headline randomized population.

Randomized evaluation reduces logging-policy bias only within the curated 7,583-video intervention pool and filtered intervention population. Do not call it a full offline A/B test, full OPE, or an unbiased evaluation over the entire catalog.

### KuaiRec big matrix — mechanistic ratio ablation

Official dataset: <https://github.com/chongminggao/KuaiRec>.

- 7,176 users, 10,728 videos, and approximately 12.53M interactions;
- exact interaction timestamps, play duration, video duration, and watch ratio;
- daily item statistics with shows, plays, complete/valid/long/short plays, total play time, and engagement counts;
- HEGM filters ads and rows with non-positive current `play_duration` or `video_duration`, sorts globally by timestamp, takes an 80/10/10 interaction split, fits ID/category mappings on train, and inner-drops validation/test rows with unseen user or item IDs.

KuaiRec is the cleanest public mechanism test for base counts versus explicit ratios, but the current HEGM processed view is not sufficient. For the HEGM-comparable primary protocol, reproduce the documented filters, split, train-only mappings, and cold-row removal above and report row counts at every step; otherwise label the result as a modified KuaiRec protocol. The positive-watch filter changes the estimand and must not be silently added or removed after results are inspected.

Rebuild a point-in-time dataset from raw `item_daily_features.csv` with an as-of join. For an interaction on date `d`, use daily aggregates only from dates strictly earlier than `d`; same-day totals contain target-period outcomes and are forbidden. Do not reuse HEGM's video-level `.unique(video_id)` shortcut for any time-varying daily statistic. Static fields must come from a genuinely static table or the latest value strictly before `d`.

Feature regimes:

- **C:** shared train-mapped user/video/category context available at scoring time.
- **P:** lagged `show_cnt`, `show_user_num`, `play_cnt`, `play_user_num`, `hist_play_duration_sum`, complete/valid/long/short-play counts, engagement counts, video duration, shifted safe denominators, and validity indicators. `hist_play_duration_sum` is the accumulated lagged daily total and must not be confused with the current interaction outcome.
- **E:** lagged play/show, unique-play/unique-show, complete/play or complete/show, valid/long/short rates, average play duration, play-time/show, and engagement/show rates.

Run the causal matrix `DCNv2(C+P)`, `DCNv2(C+P+E)`, `DCDN(C+P)`, `DCNv2-capacity-matched(C+P)`, and AFN+. The current interaction's `play_duration` is the label source and `watch_ratio` is derived from that outcome; neither may enter the inputs. KuaiRec has no native like label. Use `clip(current_play_duration, 0, c) / c`, where current HEGM code takes `c` from the 99th percentile of video duration (approximately 60 seconds), and report MAE in seconds and MSE in sec². If HEGM batch-XAUC is retained, use the same qualified name and audit rules as for VK-LSVD. Retain original IDs, timestamp/date, and stable row ID as audit-only columns.

### Yambda-50M — optional cross-domain evidence only

Official dataset: <https://huggingface.co/datasets/yandex/yambda>.

Yambda-50M contains approximately 46.47M listens with timestamps, playback ratios, and recommendation/organic flags, but it does not provide non-listen impression rows. It is therefore weaker than KuaiRand for the central response-per-impression and engineered-rate substitution claim. Run it only after the required VK-LSVD, KuaiRand-Pure, and KuaiRec tables are complete.

If used:

- use the official chronological structure: the final one-day interval is test, the preceding one-day interval is validation, with one 30-minute gap between train and validation and a second 30-minute gap between validation and test; training contains all earlier eligible events. The approximately 300 days describe the full collection horizon, not a 300-day training window followed by extra validation/test days;
- restrict the pointwise task to recommendation-driven listens, `is_organic = 0`;
- define `Listen+ := played_ratio_pct >= 50`;
- describe the task as **completion re-ranking**, not CTR or end-to-end retrieval;
- skip the shared audio-embedding file unless a predeclared model requires it.

### Datasets deliberately outside the deadline scope

KuaiSAR is a credible additional short-video log and EB-NeRD is a credible cross-domain impression-ranking benchmark. Both are deferred unless every mandatory table is complete. Do not add either dataset, or Criteo/Avazu/Tenrec/MicroLens, without an explicit scope change. Standard CTR datasets do not by themselves test interpretable base-operand-to-ratio substitution.

### Leakage, denominator, and architecture rules for every dataset

- No outcome from the target row may enter its features.
- All aggregates must use events strictly before the feature cutoff.
- Fit iALS, buckets, normalization constants, and history statistics only on permitted history/train data.
- In mandatory public `DCDN-Core`, apply reciprocal operations only to explicitly designated denominator features. PLE, categorical, ID, and content embeddings bypass the core reciprocal branch. A separately labeled production-hybrid bridge is the only exception and must use the confirmed PLE-space semantics rather than inherit the raw mask.
- Do not z-center denominators; use positive scale-only normalization, smoothing/floors, and validity indicators.
- Report coverage of the denominator mask, missing/zero frequency, clipping/floor incidence, and NaN/Inf rate.
- Scan complete materialized operand views for finiteness and denominator positivity before full training. Use synchronous value assertions only in smoke/debug runs; freeze BF16-or-FP32 model precision, compute reciprocals in FP32 under AMP, and monitor non-finite losses/activations/gradients in optimized runs.
- Freeze a small residual-near branch initialization and gradient-clipping policy before validation metrics, and apply the same product-branch initialization to DCN/DCNv2 controls.
- Keep stable user/item/time/row identifiers as audit-only columns for exact row matching, clustered uncertainty, cold-start slices, and leakage assertions; do not feed them as model features unless the frozen regime explicitly includes an ID field.

## 11. Model suite

### Main table

1. DNN/MLP
2. DeepFM
3. original DCN
4. AutoInt+
5. DCNv2
6. AFN+
7. Wukong-small
8. DCDN

AFN+ is mandatory as the closest algebraic prior: log-exp interaction learning can use negative or fractional powers. Therefore the novelty claim must not be “first model to learn reciprocal interactions.”

Run the full eight-model main suite on **exactly the shared `C+P` regime** for VK-LSVD and KuaiRand-Pure; this is the same-input RQ2 comparison. Run `DCNv2(C+P+E/R)` separately as the engineered-feature arm of the causal RQ1 matrix. Do not run all eight architectures on both feature regimes unless the scope is explicitly expanded. On KuaiRec, prioritize the causal mechanism matrix (`DCNv2(C+P)`, `DCNv2(C+P+E)`, `DCDN(C+P)`, capacity-matched DCNv2, and AFN+) rather than repeating every architecture. Yambda is optional and must not delay any mandatory public-data cell.

### Optional, in order

- EulerNet;
- xDeepFM;
- FinalMLP or MaskNet;
- DLRM.

Do not spend the deadline window reproducing scaled Wukong. Use a small public configuration only if the implementation is working. Do not copy comparison numbers from other papers.

## 12. Fairness, metrics, and uncertainty

Architecture comparisons should share:

- identical examples and temporal splits;
- identical feature regime for a given comparison;
- identical labels, sampling, and candidate sets;
- identical embedding/input encoder where semantically possible;
- identical prediction head, loss, maximum training steps/examples, validation cadence, patience, and selection rule; report the realized examples seen for each run;
- equal hyperparameter-search budget and final seeds;
- equal effective global batch, using gradient accumulation where needed.

For models using the same regime, require an identical ordered schema and identical feature values. The causal RQ1 comparison intentionally adds E/R only in the full arm: assert that every C and P column is bitwise identical and row-aligned between `C+P` and `C+P+E/R`, rather than requiring the two regimes to have equal width.

Before any test label is read or test metric is computed, write an immutable protocol manifest and its SHA-256 hash. It must cover raw-data hashes/version, exclusions, split boundaries, row-ID/dedup/order rules, rolling-state policy, C/P/V/E/R schemas and order, label and metric definitions, smoothing/floors, mappings, model configurations, tuning space, seeds, and stopping rule. Tune the predeclared configurations with seed 42 on train/validation only; suppress test evaluation until the protocol hash and chosen configurations are frozen. Then evaluate the untouched test for seeds 42/43/44 without adapting data, features, thresholds, metrics, or hyperparameters.

Report:

- task quality: LogLoss and AUC/user-GAUC; AUPRC for rare labels;
- list quality only if real candidate lists exist: NDCG@K;
- watch-time metrics: the established production quality metric plus MAE/MSE as appropriate;
- at least three final seeds with mean and standard deviation;
- paired bootstrap confidence intervals over users or temporal blocks for differences;
- dense and total parameters separately;
- FLOPs/example, throughput, peak VRAM, and p50/p95/p99 **model-only inference latency** under a frozen hardware/precision/batch/warmup/repetition protocol; production end-to-end latency remains separate industrial evidence;
- NaN/Inf rate and performance by denominator-magnitude bucket.

No-significant-difference is not proof of non-inferiority. If `DCDN-112` is described as non-inferior to `DCNv2-187`, define the margin before inspecting final results.

## 13. RunPod setup already chosen

The user configured one RunPod Pod with:

- name recommendation: `laurent-forge`;
- template: RunPod PyTorch 2.8.0;
- CUDA: choose 12.8, not 13.0;
- GPUs: 2 × A100 SXM 80GB with NVLink (160GB total VRAM);
- host resources shown: 234GB RAM and 32 vCPU;
- current displayed price: USD 3.29/hour.

Storage shown in the user's RunPod configuration screenshot:

- container disk: 1,000GB, temporary and erased when the Pod is stopped;
- persistent network volume mounted at `/workspace`: 50GB;
- verify the live mounts, free space, and write permissions before any download because the screenshot may no longer match the running Pod;
- keep code, manifests, compact metrics, logs, and irreplaceable best checkpoints under `/workspace`;
- keep raw public datasets and regenerable preprocessing caches on the large temporary container disk when the 50GB persistent volume is insufficient;
- do not stop the Pod until durable outputs have been verified on persistent or otherwise approved backup storage;
- resizing storage is a paid external change and requires user approval.

Execution:

- run independent jobs/seeds on the two GPUs using one process per GPU;
- prefer job-level parallelism over DDP for these tabular models;
- start with a 10%/one-epoch pipeline and throughput check before full sweeps;
- use streaming Arrow/Parquet or prepacked shards so CPU/I/O does not starve the GPUs;
- back up results before terminating the Pod;
- do not upload proprietary industrial logs to Community Cloud without company approval.

Budget and deadline envelope for the remote research agent:

- 24 hours: USD 78.96;
- 3 days: USD 236.88;
- 5 days: USD 394.80;
- hard incremental ceiling from the agent's immutable ledger epoch: **USD 460 pre-tax**, including compute and storage;
- warning threshold: **USD 370**;
- maximum Pod wall time from that epoch: **120 hours**;
- absolute compute stop: the earlier of the 120-hour limit and **2026-08-24 14:00 Europe/Moscow**, with the final 8 hours reserved for aggregation and backup.

The earlier suggestion to fund the account with roughly USD 500 is a balance recommendation, not authorization to spend it all or to exceed the USD 460 experiment envelope. Record any cost incurred before the ledger epoch separately.

Prices and availability are time-sensitive; recheck RunPod before extending the run.

## 14. Highest-priority missing information

Do not invent these fields. Obtain them from the user, experiment logs, or production documentation:

1. Exact industrial offline metric \(Q\), values, direction, uncertainty, split, and sample size.
2. Complete A/B design and inference details listed in Section 7.
3. Taxonomy of the 75 removed features and a mapping to retained primitives; identify ratios, rates, normalizations, windowed combinations, and non-algebraic business rules.
4. Remaining production wrapper details: exact PLE construction/clipping range, feature ordering/width, surrounding tower and head, loss, optimizer, and training configuration. Already confirmed: one fixed-anchor block initialized with `x_i=x_0`; eight PLE channels; no semantic denominator mask; sign-preserving hard floor with `eps=1e-2`; separate biased dense product/division maps; production-only FC path/projector.
5. Which components besides the cross block and input inventory were identical between control and treatment.
6. Parameter counts, FLOPs, training throughput, production p95/p99 latency, and feature-pipeline cost savings.
7. Exact authorship, affiliations, anonymity constraints, and final WSDM formatting requirements.

## 15. Execution order

Run the following two tracks in parallel. The industrial track is owner-side work because public RunPod execution is not authorized to access company data; missing industrial access must not block the frozen public experiments.

**Owner-side industrial/manuscript track**

1. Make the industrial table reproducible: `DCNv2-187`, `DCNv2-112`, production-hybrid `DCDN-112`, then `Hybrid-second-product-control-112`; add the gated division-by-FC factorial cells needed for attribution.
2. Obtain the complete A/B statistical protocol and confidence intervals.
3. Obtain the remaining production wrapper details—especially the exact PLE definition—and align the production notation with the confirmed initialization `x_i=x_0`.

**Public RunPod track**

1. Build the KuaiRand-Pure `C+P` and `C+P+R` rolling views and the VK-LSVD `C+P` and `C+P+E` views. Validate raw hashes, dates, row alignment, deduplication, audit IDs, cold/OOV handling, and leakage before any model result is inspected.
2. Run deterministic train-only smoke tests, denominator/numerical tests, and model-path tests. Do not read test labels or compute test metrics.
3. Run at most the predeclared tuning budget per model with seed 42 on train/validation only. Select configurations without test access.
4. Freeze the exact data views, rolling-state rule, model configurations, seeds, metrics, and stopping rules in the immutable protocol manifest; record its SHA-256 hash. Test access begins only after this checkpoint.
5. Evaluate the causal matrices for seeds 42/43/44. On VK-LSVD and KuaiRand-Pure, run the full eight-model suite only on `C+P`; run the full-feature `DCNv2(C+P+E/R)` arm separately.
6. Rebuild the lagged KuaiRec `C+P` and `C+P+E` daily views using strict as-of joins and the frozen HEGM-comparable filters, then run its causal mechanism matrix without using same-day aggregates.
7. Measure latency, parameters, FLOPs, throughput, peak VRAM, denominator coverage, clipping, non-finite rates, and uncertainty.
8. Add Yambda-50M only if every mandatory **public** VK-LSVD, KuaiRand-Pure, and KuaiRec result cell has a technically successful evaluation. A valid negative scientific result counts; an OOM, timeout, leakage defect, implementation failure, or missing cell does not. Industrial evidence remains an owner-side manuscript dependency, not a RunPod access prerequisite.
9. Run the optional explicit reciprocal-input lifting control for RQ3 and a small synthetic product/ratio/mixed benchmark only if time remains.
10. Replace the abandoned provisional architecture name with DCDN and fill all experimental placeholders in the TeX.
11. Compile, visually inspect, and perform a claim-by-claim evidence audit before submission.

## 16. Decision log

- Final and only architecture name: DCDN, singular “Network”.
- Final title: `DCDN: Deep Cross-Division Network for Product-and-Ratio Feature Interactions in Industrial Ranking`.
- `Laurent` is retained for theory, not for the model name/title.
- Abstract was submitted to WSDM 2027.
- Current full-paper abstract wording reports only the 40.1% relative feature-inventory reduction and explicitly states that DCDN is deployed in the production short-video service; absolute feature counts remain internal experiment identifiers.
- The production model uses one hybrid DCDN block initialized with `x_i=x_0`, keeps `x_0` fixed, and applies it to 8-dimensional piecewise-linear feature encodings; the FC path/projector is production-only. Public offline experiments use DCDN core layers on raw numerical operands without that auxiliary path or projector.
- PLE retains structured information about raw features but does not preserve multiplication or division exactly. Production Laurent claims are made in lifted PLE coordinates; the pullback and complete stabilized hybrid are generally piecewise rational in raw inputs.
- The original `DCNvsDDN.tex` theorem uses the pure core. Its `FCN` label denotes induced affine terms, not the explicit production FC path; that path belongs in Production Integration/Industrial Lessons and in a compact division-by-FC ablation.
- Frozen public dataset roles: VK-LSVD is the scalable primary benchmark; KuaiRand-Pure is the second core benchmark and randomized-exposure test; KuaiRec big matrix is the mechanistic base-operand-versus-rate ablation; Yambda-50M is optional cross-domain evidence only.
- HEGM supports exactly KuaiRec big matrix and VK-LSVD `up0.01_ir0.01`; its released processed views must be rebuilt for strict DCDN `C+P` versus `C+P+E/R` comparisons.
- KuaiRand-1K is not the headline randomized benchmark because it has only about 43K randomized rows. KuaiRand-27K is optional for a finalist scaling run, not a full grid.
- KuaiSAR, EB-NeRD, Criteo/Avazu, Tenrec, and MicroLens are outside the deadline scope unless the user explicitly expands it.
- Main model family follows DCNv2/Wukong comparisons, with AFN+ mandatory as closest algebraic prior. The full eight-model suite is frozen to the shared `C+P` regime; the full-feature DCNv2 arm is a separate causal control.
- `DCDN-sham` terminology was rejected. Use the reporting label `DCNv2-capacity-matched` for public raw-core tables and `Hybrid-second-product-control` for the exact production/bridge wrapper. The stable internal key `dcnv2_capacity_matched_core` and old display alias `DCNv2-capacity-matched-core` remain only as frozen internal provenance; existing search/view locks are not renamed or re-hashed.
- Explicit reciprocal inputs are an offline-only reviewer control, never a required production A/B arm.
- Test evaluation is forbidden until the immutable protocol manifest and SHA-256 hash have been recorded after train/validation-only tuning.
- Industrial reproducibility and A/B documentation are owner-side manuscript work; the public RunPod must neither access industrial data nor wait for it before executing the frozen public protocol.
- Actual RunPod selection is one 2×A100 SXM Pod at USD 3.29/hour, not two separately confirmed Pods.
- Remote execution is capped at USD 460 pre-tax and 120 Pod wall hours from the immutable ledger epoch, with an earlier absolute stop at 2026-08-24 14:00 Europe/Moscow and 8 hours reserved for artifact finalization.
- Owner RunPod override recorded 2026-08-18: control-plane expiry, credentialed billing inspection, account balance/headroom, and quota metadata are waived as execution gates. Use the earliest persisted ledger epoch, USD 3.29 per elapsed Pod-hour, and a fixed USD 25 storage reserve; never use stored RunPod credentials or mutate Pod/volume lifecycle. The owner is responsible for manual termination after verified backup and zero task-owned processes.
- Local source audit recorded 2026-08-18: no public datasets were present. Use the official allowlisted routes and hashes in `docs/data_source_audit.md`; place regenerable data on the temporary overlay and never commit or redistribute raw data.
- Test-lock audit recorded 2026-08-18: a credential-free HEGM provenance lookup unexpectedly exposed a published benchmark table for incompatible released processed views. No local test files, labels, DCDN outputs, or project-generated metrics were accessed. Quarantine the external values from every protocol, feature, search-space, and configuration decision; `test_unlocked` remains false.
- Repository bootstrap recorded 2026-08-18: local commit `c82bfbc` was pushed to `rw404/DCDN` on `experiments/wsdm27`. The supplied GitHub token was used interactively only; it is absent from the remote URL, credential helpers, and tracked files.
- Deterministic preflight recorded 2026-08-18: `uv.lock` SHA-256 is `af9fcbb29fedabaa0105ff2682c8d858eb3660ef6a83e7d60ccfa5ac96f49df6`; 101 CPU tests and Ruff pass. Independent one-process-per-A100 FP32/BF16 numerical smokes produced matching cross-GPU hashes and left no GPU process. No public dataset, validation metric, test label, or project test metric was accessed.
- The exact public matrix is byte-locked at `configs/experiment_matrix_v1.yaml` SHA-256 `867118db089bbbffb4abf02ff7b3c01e2c9f43e804a188a27a3886f66db4e000`. Dataset protocols and selected configurations remain unresolved and `test_unlocked=false`.

- Public-source acquisition recorded 2026-08-18: KuaiRand-Pure, KuaiRec, and
  exactly 30 allowlisted VK-LSVD objects were byte-verified in ephemeral
  storage. Their durable source-manifest SHA-256 values are respectively
  `bc5b441fdd3188d033e0ecbf0bdfd73f8bcf07bbc744d474d260ee220ae4d6e9`,
  `963019bcb4c8dbd4663f23c24a472fe0332f6494fdcbbaa74ca158a73488c5b4`,
  and `5dbbfeb23442a6b1ebdb829dc239c6a3c6c655d71446cde4551b83cc6af456ac`.
  VK week 26 remains sealed and `test_unlocked=false`.
- Model/search freeze recorded 2026-08-18: source provenance SHA-256 is
  `0844a838e5a68029299a52aaa31b5f74205fe57fd486b51544061b32c571a3f4`;
  metric-independent search-space SHA-256 is
  `b263b55da768ec33656ebdb1506d58f2809fd5826aa8b4976aedaecee53bb344`.
  Neither lock used validation or test metrics.
- Public feature phase-1 checkpoint recorded 2026-08-18: the VK-LSVD
  feature-contract SHA-256 is
  `380d105c847a73afb4927740e40401124688f895142792309b4fc8267a2a5f2b`.
  Weeks 0--25 passed structural/schema audit while sealed week 26 remained
  unopened; no validation model metric was computed.
- The corrected KuaiRand-Pure phase-1 contract, SHA-256
  `92a060d34e16d5b3dd2674d9ca75281fac0d20a54adda03c9ecaa206e9e601ef`,
  passed its owner-approved deterministic boundary embargo. It retains
  later-role rows, excludes exactly 6,639 overlapping earlier-role rows, and
  uses the actually available 2022-04-09--2022-04-21 burn-in. Test access and
  model metrics remain locked.
- The corrected KuaiRec phase-1 contract, SHA-256
  `ab016a7f6d300e7a90de9940891ea974695aa5f61ea85985e239588290522c36`,
  and blind-audit artifact, SHA-256
  `716e976e0afc642b354c50096ca0d12bdef07e55381b5e91b75cc7c4910f810a`,
  passed after the owner-approved missingness/unit revision. Four partial-null
  collection counters (69,683 nulls among 343,341 daily rows) and their four
  rates are excluded without imputation; millisecond duration/watch-time
  fields are converted deterministically to seconds. Rejected predecessor
  artifacts remain preserved and explicitly unusable; test access and model
  metrics remain locked.
- VK-LSVD train/validation materialization recorded 2026-08-18: the matched
  12-shard views manifest SHA-256 is
  `738d3610fd83a93297d94ccad4c4316dd3ad8b241f8409e84daf5b2322040dbe`,
  covering 7,505,820 train and 1,516,831 validation rows. Every shard passed
  independent byte-hash, schema, and row-count re-open checks; sealed week 26
  remained outside the path graph and unopened.

- VK-LSVD runtime/encoder hardening recorded 2026-08-18: the combined fresh
  `/tmp` suite passed 64 tests. The encoder contract SHA-256 is
  `16fa99db8e92ce4dd55c0bd5e7d7bfd8af8d3d59251cc769b41988e61d414449`;
  its C+P identity binds full-scan SHA-256
  `cc7c74b183688c1125cd3e49d32daed2b1e7560d9ee796cfe93de972792ed612`
  and deterministic cache-content SHA-256
  `b732b95f80f9ec28f4d9dd5192bf7dba3ed16bb96093bc5add209743edfff590`.
- Metric-free VK-LSVD C+P capacity resolution recorded 2026-08-18: the frozen
  artifact `configs/capacity/vk_lsvd_c_plus_p_v1.json` has SHA-256
  `d3e65d90fe0184ec48cf15abbe7358f8fd57b4944b5a95f3201b2b4e3fe81565`
  and mechanically selects the two-layer, four-expert rank-12 bracket within
  the ±5% parameter/FLOP rule. Its reporting label is
  `DCNv2-capacity-matched`; `dcnv2_capacity_matched_core` remains the
  canonical runtime key.
- Actual VK-LSVD C+P premetric Protocol v3 recorded 2026-08-18:
  `configs/protocols/vk_lsvd_c_plus_p/protocol_frozen.yaml` has SHA-256
  `1919ba14e9a1f881fdf9b45016e16c4d0b92fc71d45944d329822e28816a5a0b`;
  its active registry
  `artifacts/protocol_state/vk_lsvd_c_plus_p/active_protocol.yaml` has
  SHA-256
  `2d76236dece1769c08417cc0322f3b3bc51001cfb52ceb945e24fe237ca0fdb4`.
  The canonical views-split SHA-256 is
  `f754784d06141daf27aae45191d30d8468720d7b4d9410d7d6470c4632e67e75`,
  with 7,505,820 train and 1,516,831 validation rows. Independent
  `require_views_split`, train/validation `RuntimeIdentity`, active-state,
  and 70 fresh-`/tmp` Protocol tests passed. No selected-config hash, test
  row count, or test path is present; `test_unlocked=false`. Capacity
  resolution and Protocol materialization opened no data row or label, used no
  GPU, and computed no model metric.
- Conservative deterministic execution projection recorded 2026-08-18:
  retain frozen `effective_batch=4096`, floor the nominal 40,000,000-example
  maximum to 39,997,440 completed examples so the hard maximum is never
  exceeded, and validate at the first completed batch at or above each
  declared 4,000,000-example threshold (first at 4,001,792), identically for
  every model. The declared search lock and hash remain unchanged; this
  arithmetic execution policy requires no search or Protocol re-hash.
- Pre-GPU local guards recorded at `2026-08-18T17:05:20.923596Z`: the frozen
  local estimate is USD 50.88 current and USD 60.75 at a conservative
  three-hour ETA; storage inspection found 5,564,928 project-owned durable
  bytes and 62.794891 GiB overlay available, with no violation.
- VK-LSVD C+P implementation-smoke checkpoint recorded 2026-08-18: exact Git
  commit `38268f305a6c60ae6b8e2fc2f0e62982b03198b1`, empty tracked-diff SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
  and Protocol v3 SHA-256
  `1919ba14e9a1f881fdf9b45016e16c4d0b92fc71d45944d329822e28816a5a0b`
  bound two independent one-effective-batch runs. DCDN-Core-L2 used physical
  GPU 0 and full-rank DCNv2-L2 used physical GPU 1; each consumed exactly
  4,096 train examples, made one optimizer step and zero validation checks,
  then stopped at the requested checkpoint boundary. Both registry records are
  completed and explicitly non-scientific; validation selection, best
  checkpoint, latency, and test metrics are null. Both paths reported zero
  OOMs and zero non-finite observations with all three activation hooks
  observed. Configs, commands, results, histories, logs, registry records, and
  both `last` checkpoints were re-opened and hashed; the checkpoints passed
  integrity-checked CPU-only loading. The compact audit SHA-256 is
  `6748d9f643115f0d85b73041b0892a362b1f38efda7c404ca5d329b4ea4650b4`.
  At `2026-08-18T19:59:09Z`, after the verified audit, the two regenerable
  smoke `last.pt` files and their two SHA-256 sidecars were removed under the
  retention policy; their pre-deletion paths, sizes, hashes, sidecar checks,
  and safe CPU reopen evidence remain preserved in the audit, with status
  `removed_after_verified_audit`. The post-run cost guard estimated USD 60.09
  at 10.665352 elapsed Pod-hours including storage. The post-cleanup storage
  guard at `2026-08-18T20:00:30.583480Z` passed with 6,381,568 durable project
  bytes, 49.994057 GiB logical headroom, and 62.499523 GiB overlay free. A
  fresh retained-artifact read/parse/hash test re-opened 12 per-run evidence
  files and verified eight run/protocol sidecars; the four deleted paths were
  absent and zero task-owned training/worker processes remained. Ordinary
  seed-42 validation-only tuning may begin. `DCNv2-capacity-matched` tuning
  remains gated on a durable selected ordinary-DCNv2 learning-rate binding,
  while final/test remains locked pending selected configurations, exact best-
  checkpoint lineage, an audited one-pass final runner, and explicit Protocol
  lifecycle activation; `test_unlocked=false`.

- VK-LSVD C+P DCDN-Core validation selection recorded 2026-08-18: both
  predeclared seed-42 LR candidates completed as scientifically eligible
  validation-only tuning under Git commit
  `b41e1ce9e702b8c9406a4f01042f0d32f91f239c`, the same 39,997,440-example
  budget, and Protocol v3 SHA-256
  `1919ba14e9a1f881fdf9b45016e16c4d0b92fc71d45944d329822e28816a5a0b`.
  Strict minimization of frozen `normalized_mse` selected
  `dcdn_core_l2_lr_1e_3` (`learning_rate=0.001`) at
  `0.020977282795389355`, versus `0.021020834286119675` for
  `dcdn_core_l2_lr_3e_4`; the absolute difference is
  `0.000043551490730320`. No tie occurred and no post-metric tie-break was
  invented. The durable per-model selection artifact SHA-256 is
  `66458ce3d96d6e4f5d5382dab904a5c120134ba52385d8a5bcdbd9d3c2bccb49`.
  Its retained selected checkpoint is
  `best-step-000000007813-generation-0000.pt`, SHA-256
  `e12d3b5e769ed3c0a45318e921a2b83a276c9a1ace831cd7575ba93cfebd0f0f`.
  All 12 best generations and two `last` checkpoints passed sidecar/hash,
  safe CPU-only reopen, and tensor-finiteness audit; no checkpoint was deleted.
  This freezes only the DCDN per-model choice, not the dataset-wide
  `selected_configs.lock`. `DCNv2-capacity-matched` remains gated on the
  selected ordinary-DCNv2 LR, and final/test remains locked pending all model
  selections and an audited one-pass runner; `test_unlocked=false`.

- VK-LSVD C+P ordinary full-rank DCNv2 validation selection recorded
  2026-08-19: the seed-42 LR=0.001 candidate completed under Git commit
  `028b9c11a827a5821d20376c39bdc171fa032cf8` with normalized MSE
  `0.02091150474550729`; immutable result SHA-256 is
  `51a39053240683b8b8c90688844b2a4a544b678db71dd2559e61eefec8fe1e23`.
  The paired LR=0.0003 candidate first suffered two supervisor/registry `EIO`
  execution failures after the durable 12,001,280- and 24,002,560-example
  boundaries. Those attempts remain scientifically ineligible; their failure
  snapshot SHA-256 values are
  `82a7858c1f8eed49c3c4f958a58907166faaf19dee4082903276329cdd1bfab4`
  and
  `a42d29bf579656dae1fc5eab06f2046bea60e4c5ff8e37e69049740f3f28c5ef`.
  After registry/worker hardening and fault-injection regression under commit
  `d55d61b3a09c95045872f49893377c01f57c2d45`, exact resume-v2 completed the
  same candidate at the 39,997,440-example floor with nine validation checks,
  zero OOM/non-finite events, and `test_metrics=null`. Its result SHA-256 is
  `0e6381d016a65a48d9f12f48b6c871ae242139b2e9a2341f1ecba3efd03cf58e`;
  two heartbeat-helper timeouts recovered within the bounded degraded window
  and have byte-identical local/persistent incident receipts with SHA-256
  `5bc1a25000ba591c322007f56c48705aca52a60e257c22199a765ed1d06891c0`
  and
  `87d24564d6ebae0eaf5b3ca5977e1f521c6be6265851849b64da3c6c0c9bb1da`.
  Strict minimization of frozen `normalized_mse` selects
  `dcnv2_full_rank_lr_3e_4` (`learning_rate=0.0003`) at
  `0.020908477427111632`, better than LR=0.001 by
  `0.000003027318395658`; no tie or post-metric tie-break occurred. The
  canonical validation-only selection artifact SHA-256 is
  `869cbcb7a227b274700cda142f23c59040497bfdfe4da101888e2cc63f39d67a`.
  The retained selected best checkpoint is
  `best-step-000000006836-generation-0000.pt`, SHA-256
  `f3a27d6522df474f1cea06abe302d289ff9fc331e9ae7135baa9589a3427618c`;
  it passed strict sidecar and safe CPU-only `weights_only=True` tensor-
  finiteness checks. No checkpoint was deleted. This durable per-model choice
  clears only the ordinary-DCNv2 LR dependency for capacity-control binding;
  the dataset-wide `selected_configs.lock` remains incomplete and
  `test_unlocked=false`.

- Capacity-control LR authorization materialized 2026-08-19 from clean commit
  `bc584157a5845571e72acde7d5075dd5f5d3bcc2`. The immutable compact artifact
  `artifacts/selection/vk_lsvd_c_plus_p_capacity_lr_binding_v1.json` has
  SHA-256
  `d0ef149274d0f2692acc1f166d72350e44bd8a99331b7ae7f4dbc7d9df1cf1b2`
  and a strict adjacent sidecar. Its production validator reopens the ordinary
  selection receipt, both candidate evidence chains, the selected checkpoint,
  the frozen search/protocol/capacity inputs, and binds
  `capacity_rank_lower` (rank 12) to inherited LR 0.0003 without using a
  validation metric for rank selection. The full local CPU suite passed
  534 tests; test remains locked. The bound clean schema-v4 capacity-control
  run is now complete as recorded below.

- VK-LSVD C+P capacity-control validation result recorded 2026-08-19: the
  mechanically selected rank-12 `DCNv2-capacity-matched(C+P)` control ran
  under clean commit `eb5ffb026c62b86378ace59803d4f4a8c5b38c9d`, inherited
  the frozen ordinary-DCNv2 LR 0.0003 through compact authorization SHA-256
  `d0ef149274d0f2692acc1f166d72350e44bd8a99331b7ae7f4dbc7d9df1cf1b2`,
  and completed the same 39,997,440-example seed-42 validation-only budget.
  Its immutable result SHA-256 is
  `fdcf94786df724aa0a91a525b8e23f296530b8837e2669a2bedfe719bc6d0507`;
  the selected validation MSE is `0.02100464611993931` at step 6,836, with
  selected best-checkpoint SHA-256
  `ee0495ebd57c49fbf4b89d8b15fe0c1ea404ca26578d2b1dd0710c3567dd7984`.
  The run made nine validation checks, reported zero OOM and zero non-finite
  observations, and completed with `scientific_result_eligible=true`. Safe
  CPU-only reopen verified every tensor in both best and terminal checkpoints
  as finite. On validation, selected DCDN is lower than this capacity-matched
  control by `0.000027363324549955` (about 0.1303%), but ordinary full-rank
  DCNv2 remains lower than DCDN by `0.000068805368277723`. These are
  validation-only comparisons; test remains locked and no test claim is made.
- VK-LSVD C+P AFN+ validation selection recorded 2026-08-19: the two frozen
  seed-42 LR candidates completed the same training/validation-only protocol.
  LR=0.0003 reached normalized MSE `0.0208063458817398` after 36,003,840
  examples and nine validation checks; LR=0.001 reached
  `0.02082650517903153` after 24,002,560 examples and six checks. Both stopped
  by the predeclared validation patience rule with zero OOM/non-finite
  observations. Their trainers had already written valid immutable results
  and checkpoint pairs when the former generic 64-MiB artifact cap caused a
  false supervisor terminal failure. Commit
  `f450c67efdce86d5dc788c79f4ccbaf173501ca7` replaced only checkpoint
  verification with a streaming, sidecar-bound reader and added an exact
  reconciliation gate. It preserved each full prior failed record and SHA,
  reverified result/last/best twice, required the original trainer PID to be
  dead, and published both registries as completed/scientifically eligible.
  Strict validation minimization selects `afn_plus_lr_3e_4` (LR=0.0003),
  lower by `0.00002015929729173166` (about 0.0968%); no tie or post-metric
  tie-break occurred. Canonical selection receipt SHA-256 is
  `69463d2687d46520ea270ffcdfeaaa8ef47cd11173dc3943780cc4cc486d7e54`.
  The retained best checkpoint SHA-256 is
  `42dd6b18e0cfb51769e81718a86163d09057546b5cf00f7b10f8ad5a7d42dc8d`.
  All ten AFN checkpoint/sidecar pairs passed streaming hash, safe CPU
  `weights_only=True`, accumulation-boundary, metadata-lineage, and tensor-
  finiteness checks; no checkpoint was deleted. This is another per-model
  validation-only choice, not `selected_configs.lock`; test remains locked
  and no test event or test metric informed selection.

- VK-LSVD `C+P+E` premetric protocol isolated and activated 2026-08-19.
  The immutable Protocol v3 artifact
  `configs/protocols/vk_lsvd_c_plus_p_plus_e/protocol_frozen.yaml` has
  SHA-256
  `6a6e33c908d666c4374ac5052a207c38a9fe012dac147a5e572cb819013a5739`;
  its separate active-registry SHA-256 is
  `7054e2e493afa49dafab7561ed6b7909e8ef56741976570915aff0bad9c1cce3`.
  It remains `premetric_locked`, permits only `dcnv2_full_rank`, binds the
  already verified `C+P+E` cache and feature identities, and keeps both
  `test_unlocked=false` and `test_path_available=false`. The CLI and worker
  now fail closed on dataset/regime/model mismatches before source or model
  construction and repeat the protocol/regime checks after result reopen.
  The next executable action is the frozen seed-42 validation-only pair for
  learning rates 0.0003 and 0.001, one process per A100; no `C+P+E` metric
  has been observed yet and no test path has been opened.

- The first VK-LSVD `C+P+E` DCNv2 pair stopped before any training batch on
  2026-08-19. The shared harness incorrectly recomputed the C+P-only
  capacity-control projection on the wider C+P+E encoder and rejected both
  candidates with `full capacity artifact projection differs from mechanical
  resolver`. Both registry records remain technical failures with
  `scientific_result_eligible=null`; their histories contain only the local
  guard, with no train, validation, checkpoint, result, or test event. The
  immutable paired failure receipt SHA-256 is
  `d60f40816564427f41cf0e9748fc4d76f3fd6e613aab73d781630d9c2d6c0bee`.
  The corrective code retains exact artifact/SHA/repository validation for
  all raw executions, preserves mechanical recomputation for C+P, and allows
  C+P+E provenance only for `dcnv2_full_rank` without treating the wider
  encoder as the C+P capacity-control object. Focused harness regression is
  74/74 PASS, and the fresh full CPU suite is 541/541 PASS with Ruff and
  targeted bytecode compilation clean. Retry requires a clean
  committed/pushed fix and fresh run IDs; the failed IDs will never be
  overwritten. Test remains locked.

- The first C+P+E retry pair under commit
  `871b82fb8c817fd2f8739ec2ae38f00b2fbaf92a` also stopped before any training
  batch on 2026-08-19. The first projection gate was correctly scoped, but a
  later parameter-count gate still compared the wider C+P+E full-rank model
  against the C+P-only capacity artifact. Both histories again contain only a
  local guard; no checkpoint, result, validation, or test event exists. The
  immutable paired failure receipt SHA-256 is
  `91a5011e5bfb7955e838f3256fac320c13b6a7e2a212ff07063a52760e0ebe17`.
  The second narrow correction preserves exact C+P capacity-count equality
  for DCDN, ordinary DCNv2, and the mechanically selected capacity control,
  while treating the C+P artifact as provenance only for the isolated wider
  C+P+E full-rank DCNv2 model. Focused harness regression is 74/74 PASS, and
  the fresh full CPU suite is 541/541 PASS with scoped Ruff and bytecode
  compilation clean. Another attempt requires a new clean committed/pushed
  code identity and fresh run IDs; neither failed pair will be overwritten.
  Test remains locked.
- VK-LSVD `C+P+E` ordinary full-rank DCNv2 validation selection recorded
  2026-08-19: the second retry pair under clean commit
  `c8d82f8f9edfcbbed5d78fea18bc797484cdad8a` completed the frozen seed-42
  tuning unit. LR=0.0003 completed 39,997,440 examples and nine validation
  checks at minimum normalized MSE `0.020849287259723712`; immutable result
  SHA-256 is
  `b46d2129618be37c7c318ecd6aa99e0507b1d02c6ebe00c05fd3f388499b1232`.
  LR=0.001 first encountered a checkpoint-file `fsync` `EIO`; the incomplete
  check-four observation is not selection evidence and the immutable failure
  receipt SHA-256 is
  `be53192dcff1e57a0b5f5a0808c9aa3a40cd6b109454e3d7349c43340d1e26fc`.
  Exact resume from 12,001,280 examples recomputed check four identically and
  completed the same 39,997,440-example/nine-check budget. Its immutable
  result SHA-256 is
  `7ec7c16b5bb5f6c345dc7bd35953e43ce4ea2a072b7acf8544228171600ce3ef`.
  Strict validation minimization selects `dcnv2_full_rank_lr_1e_3`
  (`learning_rate=0.001`) at `0.020826933840839952`, lower than LR=0.0003 by
  `0.000022353418883760273` (about 0.107214%); no tie occurred. Both terminal
  results are scientifically eligible, with zero OOM/non-finite observations
  and `test_metrics=null`. The canonical per-model selection receipt is
  `results/tuning/vk_lsvd_c_plus_p_plus_e_dcnv2_full_rank_selection_v1.json`,
  SHA-256
  `85ab7ccbb8d74652f3aa3416a9ada872b36ed2de24c7b8341b56b9b8633f5c41`.
  The retained selected checkpoint is
  `runs/vk-cpe-tune-dcnv2-full-rank-l2-s42-lr1e3-v1-retry-v2/best-step-000000007813-generation-0000.pt`,
  SHA-256
  `858fec98f915cdec9dea8dac0aebd1156f06d36eabd966c9aa4fc522c015d23f`;
  best/last CPU finite-tensor reopen and all 14 checkpoint sidecars passed.
  No checkpoint was deleted. This freezes only the C+P+E control LR and does
  not answer RQ1. Final/test remains locked; after RQ1 completes, README must
  report the frozen paired seed-42/43/44 test results, uncertainty, effect
  size, and an explicit supported/not-supported/inconclusive hypothesis
  verdict before any modification is discussed.
- VK-LSVD `C+P` DNN validation selection recorded 2026-08-19: both frozen
  seed-42 LR candidates completed under clean commit
  `46add65bea8134d9f1a65373a82d8bcfb5b6579b`. LR=0.0003 stopped by the
  predeclared validation-patience rule after 28,000,256 examples and seven
  checks at normalized MSE `0.020930008215596697`; LR=0.001 stopped after
  32,002,048 examples and eight checks at `0.020899830835974933`. Both are
  scientifically eligible, with zero OOM/non-finite observations and
  `test_metrics=null`. Strict minimization selects `dnn_lr_1e_3` (LR=0.001),
  lower by `0.000030177379621763867` (about 0.144182%); no tie or post-metric
  tie-break occurred. Canonical per-model selection receipt SHA-256 is
  `7c851bfb7ca37e822fe05faca78a6488f56c447c995286143379a3611a166204`;
  selected best-checkpoint SHA-256 is
  `b9a98fe1fd7eff8963b00fe887b35ce73d2c37c4885b97a08e2f8bc674cf82f1`.
  All ten checkpoint pairs passed strict sidecar, safe CPU `weights_only`, and
  tensor-finiteness audit; no checkpoint was deleted. Each run recovered from
  one bounded registry-heartbeat helper timeout without trainer interruption;
  both content-addressed incident receipts are preserved. This is validation-
  only and does not support a universal DCDN-superiority claim: selected DNN
  is lower than selected DCDN by `0.000077451959414422` (about 0.369218%) on
  validation. The next frozen C+P model is DeepFM; dataset-wide selected
  configs and final/test remain locked.
- VK-LSVD `C+P` DeepFM validation selection recorded 2026-08-19: both frozen seed-42 LR candidates completed under clean commit
  `cd7d54ca13dd57c21821d704f44090bfc2969c47`.
  LR=0.0003 stopped after 24,002,560 examples and six validation checks at normalized MSE `0.020845575060232564`; LR=0.001 stopped after 32,002,048 examples and eight checks at `0.020876811708431246`.
  Both results are scientifically eligible, have zero OOM/non-finite observations, and contain `test_metrics=null`.
  Strict minimization selects `deepfm_lr_3e_4` (LR=0.0003), lower by `0.00003123664819868269` (about 0.149624%); no tie or post-metric tie-break occurred.
  Canonical per-model selection receipt SHA-256 is `a6ed26dcce171cf8c27cb00e2db57c90cfcead0481251ed8e4a3a830ff812aeb`; selected best-checkpoint SHA-256 is `f2ec954cc34ed8d6e2bb650899e0a0405dcf11926300a61a97068f0a82d8ecc3`.
  All eight checkpoint pairs passed strict sidecar, safe CPU `weights_only`, and tensor-finiteness audit; no checkpoint was deleted.
  Two bounded registry-heartbeat helper timeouts recovered without trainer interruption, and both content-addressed incident receipts are preserved.
  This remains a validation-only per-model choice: selected DeepFM is lower than selected DCDN by `0.000131707735156791` (about 0.627859%) on validation, so it does not support a universal DCDN-superiority claim.
  The next frozen C+P model is DCN; dataset-wide selected configs and final/test remain locked.
- VK-LSVD `C+P` Original DCN validation selection recorded 2026-08-19:
  both frozen seed-42 LR candidates completed under clean commit
  `db4447aecf0d71f3bb120411683b6e25f4fa04de`. LR=0.0003 consumed the
  full 39,997,440-example budget and reached normalized MSE
  `0.021035584402689534` after nine validation checks; LR=0.001 stopped
  by the predeclared patience rule after 24,002,560 examples and six checks
  at `0.02105195259385249`. Both results are scientifically eligible, have
  zero OOM/non-finite observations, and contain `test_metrics=null`. Strict
  minimization selects `dcn_lr_3e_4` (LR=0.0003), lower by
  `0.000016368191162956236` (about 0.077751%); no tie or post-metric
  tie-break occurred. Canonical per-model selection receipt SHA-256 is
  `f352c9d07a9690303d7d0fc5986147310d0f83d9185599b02fcdb58a86e50656`;
  selected best-checkpoint SHA-256 is
  `38cd618114fbd2a74355932cf3b1a311dfa01354e105b168ba67cd99a44fd615`.
  All 11 checkpoint pairs passed strict sidecars, safe CPU `weights_only`,
  and tensor-finiteness audit; three bounded registry-helper incidents
  recovered without trainer interruption and are preserved. No checkpoint
  was deleted. On the matched seed-42 validation protocol, selected DCDN is
  lower than selected Original DCN by `0.00005830160730017908` (about
  0.277157%). This supports the narrow DCDN-versus-DCN validation hypothesis
  and is now reported in README, but it is not the final multi-seed test RQ1
  verdict and does not establish universal superiority. The next frozen
  `C+P` model is AutoInt+; dataset-wide selected configs and final/test remain
  locked.
Update this decision log whenever a material choice changes.
