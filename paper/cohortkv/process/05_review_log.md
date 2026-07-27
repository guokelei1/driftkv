# CohortKV revision log

This log records substantive review rounds applied to `manuscript.md`. It is not a prospective
checklist: an item appears under “fixed” only after the manuscript was changed.

## Draft v0

Scope:

- complete Methods/Results-first working draft;
- four paper-native figures;
- manual numbered references plus `references.bib`;
- explicit evidence labels for replicated, adaptive seed-0, interface-validated, and open results.

Known before review:

- no complete-cohort v4 performance result;
- full-affine compiler and two-GPU runtime remain adaptive seed-0;
- no Markdown-to-PDF toolchain is installed in the environment.

## Revision 1: factual, protocol, and numerical audit

Status: **completed**

### Sources checked

- `docs/08_core_insights_and_roadmap.md`
- `docs/eval_protocol.md`
- `results/scaling/kuairand_data_utilization_summary.json`
- `results/exposure/cache_version_matrix_cross_dataset_summary.json`
- `results/motivation_scale/capacity_v2_summary.json`
- `results/motivation_scale/cohort_tiered_migration_v1_summary.json`
- `results/motivation_scale/long_context_4plus12_verified_compiler_seed0.json`
- `results/system/kuairand_long_context_4plus12_two_gpu_migration_system_seed0.json`
- `results/system/kuairand_long_context_4plus12_cohort_jagged_system_seed0.json`
- current compiler, operator, destination, and out-of-core implementation

### Numerical checks passed

- Top-50k/all-chunks BestRank value partition:
  3837.67 / 2952.11 / 885.56 with the manuscript CIs.
- Frozen 3×3 parameter counts, endpoint staleness taxes, seed signs, and seed-0 GPU cost ratios.
- 27-chain cost 0.121080 with CI `[0.111780, 0.130380]`, K/V recovery 0.586683 with
  CI `[0.546620, 0.626746]`, 25/27 fidelity, and 6/9 strict cells.
- Verified compiler certificate and final values for ages 11/7/1.
- Operator medians 3.119/0.838/0.706 ms and relative error 3.66e−4.
- Two-GPU 903.7 records/s, 1.95099× scaling, and 11.21595× over the tuned BF16 exact path.
- Jagged boundary 1.0185× host, 0.9835× HBM, and 2.1587× direct-HBM/host endpoint ratio.

### Problems found and fixed

1. **Structural replay was mislabeled as residual transport in the verified compiler.**
   The replicated cohort-tiered library uses residual-delta transport, but the long-context
   verified action library uses `structural_p4`/`structural_p8`. The manuscript now distinguishes
   the protocols and does not pool the two controls.
2. **“Identical input” wording hid different source representations.**
   Migration reads host-resident normalized capsules, while exact recomputation reads host-resident
   raw histories. The manuscript now claims matched source residency and the same target
   publication boundary, and names both inputs explicitly.
3. **The adaptive status needed a stronger reason.**
   Although the final fit/selection/certificate/test roles are disjoint, earlier design rounds had
   inspected this seed's former test population. This is now stated at first presentation of the
   verified result.
4. **BestRank direction was undefined.**
   The protocol section now defines BestRank as the minimum catalog rank among engaged positives
   and defines positive gain as baseline rank minus evaluated rank.
5. **Ekko bibliography pages were wrong.**
   Official USENIX metadata gives pages 821–839; `references.bib` is corrected.
6. **Endpoint language was inconsistent.**
   “Same/identical host input/output” was replaced with the narrower
   “same host-residency and target-publication boundary.”

### Boundary scan

The manuscript contains no claim of:

- universal task-quality improvement;
- age/drift/task admission;
- exact ranking upper bound;
- production deployment or SLO;
- full-cohort v4 speedup;
- physical SSD/network/remote-GPU performance; or
- source-side full out-of-core execution.

## Revision 2: narrative and system-closure audit

Status: **completed**

### Reference techniques checked

- HCache: the paper retains the two-endpoint opportunity but names same-model restoration as a
  different validity problem.
- Ekko: model freshness motivates the setting, while training, model dissemination, validation,
  and rollback remain outside the system boundary.
- vLLM: the affine program changes the operator and engine interfaces instead of appearing as an
  isolated regression result.
- DistServe: all four motivation observations map to a requirement, a mechanism, and an evaluation
  question.
- Orca: `(migration anchor, served K/V target)` remains the shared compiler, batching, placement,
  and publication unit; it never becomes a safe-reuse prediction.

### Problems found and fixed

1. **The closest-work boundary arrived too late.**
   The Introduction now states that HCache is same-model restoration, DroidSpeak already covers
   cross-variant K/V sharing, and MTServe covers persistent recommender-K/V placement. The claimed
   difference is narrowed to a compiled source-to-target transform for successive streaming HSTU
   versions and a fixed destination-update cohort.
2. **MTServe was described only at the capacity level.**
   After checking the full paper, the comparison now acknowledges per-user K/V persistence,
   Page–Chunk organization, GPU/host movement, and replacement. The remaining boundary is that
   its published workflow does not define a model-version state transform.
3. **“Label-free” could be read as operationally free.**
   The compiler section now states that certificate probes require exact K/V and full-catalog
   score vectors, that v2 excludes this work, and that cohort-size amortization is an open
   measurement.
4. **The runtime architecture implied a completed fallback path.**
   Published plans contain stronger actions, but the destination engine currently consumes affine
   programs only. The limitation now appears in Implementation, Discussion, and the full-cohort
   experiment gate; automatic escalation is not claimed.
5. **The 11.22× result could be mistaken for exact-equivalent output.**
   The evaluation now separates the 64-record label-free systems trace from the 522-user quality
   evidence and states that the speedup compares certified approximation with exact replay.
6. **Manual references did not follow first appearance.**
   The bibliography and citations are reordered to HSTU, Ekko, HCache, DroidSpeak, MTServe,
   KuaiRand, Tenrec, vLLM, CachedAttention, Orca, and DistServe.

### Narrative closure

The contribution order is compiler → operator → destination engine in the Introduction,
Overview, Design, Evaluation, and Conclusion. The destination contribution remains explicitly
interface-validated, and every unresolved performance or orchestration claim is routed to an open
gate rather than filled by extrapolation.

## Revision 3: reviewer attack surface, terminology, and final presentation

Status: **completed**

### Reviewer attacks applied

1. **Is the novelty already covered by HCache, DroidSpeak, or MTServe?**
   The paper claims neither same-model restoration nor cross-model K/V in general. It isolates the
   successive-streaming-version HSTU transform, old-`Norm(x)` compilation surface, semantic
   certificate, and fixed-cohort publication job. A direct DroidSpeak-compatible baseline and a
   physical MTServe-style movement comparison are explicit pre-submission gates.
2. **Is the architecture stronger than its evidence?**
   Compiler evidence is split into replicated simple repair and adaptive seed-0 verification;
   operator/runtime evidence is controlled seed-0; the destination engine remains
   interface-validated. Automatic fallback dispatch and v4 performance are not claimed.
3. **Does 11.22× come from an unequal endpoint?**
   Migration and exact necessarily consume different representations, so the manuscript names
   normalized capsules versus raw histories. Both begin in host memory and publish complete
   pinned-host K/V; all inclusions, exclusions, approximation error, and separate quality
   provenance are stated.
4. **Is one seed being presented as general evidence?**
   `adaptive seed-0` or `controlled seed-0` now appears at the first numerical use in the Abstract,
   Motivation, operator organization, compiler result, and system result.
5. **Is the 50% capsule overhead hidden?**
   The paper calls it additional state, excludes creation from the current runtime, and requires
   physical bytes, creation cost, compression error, and a break-even update frequency.

### Problems found and fixed

1. **The claim matrix retained the phrase “identical host input/output.”**
   It now records different source representations with matched source residency and target
   publication.
2. **The evidence figure called KuaiRand/QB/QK three datasets.**
   It now says three data tables; the manuscript keeps the related-Tenrec limitation.
3. **The architecture figure implied that the engine executes fallback.**
   It now distinguishes a published fallback plan from the still-open automatic dispatch path.
4. **Residual replay was called a monotone tier.**
   Because semantic quality is not guaranteed to be monotone, it is now a predefined structural
   escalation tier.
5. **“Exact” overloaded three meanings.**
   Current-model exact recomputation remains the semantic term; layout/readback correctness now
   uses “element-for-element,” “lossless,” or “byte-exact.”
6. **The future benchmark requested the “same source,” which is impossible for capsules versus
   raw histories.**
   The gate now requires the same source tier/residency and target boundary while reporting each
   representation's bytes separately.
7. **Closest-work comparisons were acknowledged but not operationalized.**
   The open-gaps document now requires a compatible selective-layer recomputation baseline and,
   after physical storage exists, a no-transform and MTServe-style placement/movement baseline.

### Mechanical checks passed

- Abstract: 249 words.
- Structure: 12 numbered sections, one artifact appendix, four figures, and ten main tables.
- Citations: first appearances are `[1]` through `[11]`; all 11 are used and all have BibTeX
  entries.
- Markup: inline/display math delimiters and fenced blocks are balanced.
- Files: all four manuscript figure links and all eight concrete Appendix result paths exist; the
  ninth row correctly records implementation/tests with no performance aggregate.
- SVG: all four sources pass XML validation; evidence labels and figure numbers match the text.
- Whitespace: no trailing whitespace or empty deliverable files.
- Boundary scan: no universal task-quality, production SLO, full-cohort v4, physical SSD/network,
  automatic-fallback, or exact-equivalent 11.22× claim remains.

### Remaining presentation boundary

The environment has no Markdown-to-PDF or LaTeX toolchain, so this delivery is validated source
rather than a venue-typeset PDF. Venue template integration, pagination, and final rasterized
figure inspection remain a formatting pass, not an unreported research result.

## Revision 4 (2026-07-26): editorial and structural pass against the writing guide

Scope: writing quality only. No claim, number, evidence-class label, or boundary statement was
added, strengthened, or removed. Changes to `manuscript_v2_en.md`:

1. **Abstract** rewritten for sentence length and evaluative direction; the 27-chain result is now
   split into two sentences and the residual is described in words instead of the inline
   `fresh − cheap` code span.
2. **Introduction** gains an explicit "Our key insight" sentence and a compact results-summary
   paragraph before the contribution bullets (vLLM-style); the 11.22× number was deduplicated from
   the second contribution bullet.
3. **Tables 1–10** are now numbered with captions and each has an in-text lead-in reference;
   duplicated caption text was removed from adjacent paragraphs.
4. **§9.1** reorganized under run-in headers (Datasets and task protocol / Serving semantics and
   measurement / Evidence levels).
5. **§9.5** reorganized under run-in headers (Operator microbenchmark / Workload / Measurement
   boundary / Results); the former single measurement-boundary paragraph is split.
6. **§10** restructured to systems style: former §10.2 "Achievement, contribution, and impact"
   merged into §10.1 as a closing paragraph; limitations renumbered to §10.2; the full-cohort
   admission-gate sentence converted into a bulleted list; §9.6 cross-reference updated to §10.2.
7. **§5.3** and **§9.4** long paragraphs split by function; **§8** now states implementation size
   (~9.5K core lines, ~26K test/benchmark lines, measured from the repository).

Note: the Abstract now exceeds the previously recorded 249 words; recount before venue submission.

## Revision 5 (2026-07-27): Stage-1 closest-baseline result

Scope: protocol-valid adaptive seed-0 development evidence under
`cohortkv_single_config_stage1_frontier_v1`.

1. Implemented a separate DroidSpeak-adapted HSTU reference that copies old K/V outside one
   contiguous current-model interval; the legacy current-projection helper is not used.
2. Added correctness tests for outside-interval equality, minimal replay, full-depth exactness,
   transition-state validation, and residual hidden-suffix sufficiency.
3. Evaluated 53 intervals plus six anchors/controls for each theta0/theta4/theta10 source pair:
   177 resident points on the frozen 60-user program-selection role, followed by the disjoint
   60-user certificate role. Final-test users were not evaluated.
4. Compiled repair costs 0.0656–0.0664× exact at 0.8755–0.9258 worst-view recovery and strictly
   dominates every selective interval. The strongest selective point is `m12/layers0-11`,
   costs about 0.698×, reaches 0.4495–0.4850 recovery, and fails the contract for all pairs.
5. Revised the manuscript rather than weakening the contract: exact is the publishable fallback;
   `m12/layers0-11` remains only a certificate-failed diagnostic for the later common-destination
   benchmark.
6. Replaced the Figure-6 skeleton with the measured three-pair development frontier and added the
   checked-in summary at `configs/cohortkv_single_config_v1/stage1_frontier_summary.json`.

## Revision 6 (2026-07-27): Stage-2 deployed compiler closure

Scope: adaptive seed-0 compiler implementation evidence under
`cohortkv_single_config_stage2_compiler_v1`.

1. Added strict FP16 runtime-program serialization and loading with source/target version, model
   shape, dtype, contiguity, finiteness, path, provenance, and frozen-input hash checks.
2. Reapplied the unchanged label-free contract to serialized certificate shards on the disjoint
   60-user certificate role for theta0/theta4/theta10; final-test users were not evaluated.
3. All three pairs select compiled full affine at `0.01651–0.01657x` resident exact cost. Cache
   recovery is `0.8810/0.8897/0.9365`; all recovery and coverage lower bounds pass the primary
   contract.
4. Froze ordered plans: theta0/theta10 use `compiled -> p8 -> exact`, while theta4 uses
   `compiled -> exact`. Targets 50%–80% select compiled and 90% selects exact.
5. Rejected the proposed FP16 residual hidden suffix after real old hidden states exceeded the
   finite range by over two orders of magnitude. Only that optional auxiliary state changed to
   BF16; its two-byte accounting and the FP16 primary path are unchanged.
6. Added the checked summary and plan artifacts under
   `configs/cohortkv_single_config_v1/` and the full protocol record in
   `experiments/system/COHORTKV_STAGE2_COMPILER_V1.md`.
7. Kept the evidence boundary narrow: the new `0.0165x` number is a resident deployed-operator
   component, not a full-cohort or destination-transaction speedup, and seed 0 remains adaptive.

## Revision 7 (2026-07-27): Stage-3 common-layout operator closure

Scope: adaptive seed-0 resident operator evidence under
`cohortkv_single_config_stage3_operator_v1`.

1. Added one `execute_into` contract for the FP32-arithmetic transport reference, packed FP16,
   and fused FP16: every path now ends at the same contiguous, unpadded FP16 K/V extent with
   lengths and offsets.
2. Verified all nine batch/bucket layouts on the complete 60-record program-selection
   distribution. Each layout covers 88,085 tokens and 1,443,184,640 valid K/V elements; all
   dense padding is zero and every dense-to-extent comparison is exact.
3. Screened all 18 frozen resident candidates and selected fused FP16, batch four, bucket width
   32. Every fused sample is below every packed-control sample, with a 1.995× median advantage;
   no stable ordering is claimed among the close fused finalists.
4. Replaced Table 7's unequal output-layout microbenchmark with a common-extent comparison. On
   the representative real B4/S2047 shape, reference/packed/fused take
   14.610/5.378/2.729 ms; packed peaks at 402.6 MB of operator temporaries and fused at zero
   beyond the preallocated destination.
5. Retained the prior jagged/page exactness and negative performance boundary without reopening
   layout tuning.
6. Kept the evidence boundary narrow: Stage 3 excludes source I/O, allocation, movement,
   publication, and commit. Stage 4 must independently retune every endpoint and execute the
   complete 682-record transaction.

## Revision 8 (2026-07-27): Stage-2/3 reverse audit

1. Recomputed all 15 Stage-2 threshold certificates from the 180 raw certificate records and
   independently checked every derived action summary. The frozen decisions and values are
   unchanged.
2. Strengthened executable-plan loading with a plan hash, complete certificate/action coverage,
   selected-certificate identity, representation validation, and frozen-input descriptor checks.
3. Hardened the Stage-3 direct-write ABI against malformed offsets and storage aliasing, added
   destination bounds to the Triton stores, and validated metadata before timing.
4. Corrected a latent stability-gate bug so the selected fused candidate, rather than merely the
   fastest fused finalist, is compared with packed. The current selected candidate was already the
   fastest fused finalist, so the decision is unchanged.
5. Re-ran and re-froze Stage 3 after removing redundant packed/reference validation work. The new
   60-record medians are 31.070 ms fused and 61.970 ms packed; every fused sample remains below
   every packed sample.
6. Replaced inaccurate “paired repeat” wording with the actually supported complete sample
   separation and labeled Table 7's fidelity column as a full-distribution measurement.
7. Bound every executable plan's complete action library, contract, certificates, threshold
   sweep, deployed certificate, runtime descriptor, and compiler accounting back to the
   independently rederived Stage-2 raw result.
8. Reconstructed Stage-3 runtime-program coverage directly from the Stage-2 summary and tightened
   the frozen capsule/output contracts, representative shape, timing samples, and environment
   checks.
9. Clarified throughout the active protocol and manuscript that `reference_fp32` widens the same
   serialized FP16 capsule/program for arithmetic. It is a transport/layout oracle, not the
   original FP32 fitted program or current-model exact K/V.

## Revision 9 (2026-07-27): Stage-4 closure and Stage-4.5 admission plan

Scope: full-cohort normal-path evidence under `cohortkv_single_config_stage4_system_v1` and the
downstream plan; no Stage-4.5 performance result is claimed.

1. Completed and froze all 30 independently tuned method/destination/GPU points: compiled,
   selective, exact, residual, and no-transform over HBM/DRAM and 1/2/4 GPUs.
2. Verified every formal point with a complete transport-correctness pass, warmup, three timed
   jobs, capacity preflight for all five executions, and complete duplicate-free manifests.
3. Reversed the expected endpoint claim without changing the resident evidence: compiled is
   2.70–3.49× faster than selective, but exact is 1.20–2.87× faster than compiled at all six
   matched endpoints.
4. Attributed the reversal to the measured supply path: the serialized FP16 capsule is 17.82 GB
   versus 89.1 MB of physical raw-history input for exact, and source processing consumes
   91.35%–96.91% of compiled wall time.
5. Fixed a capacity-preflight underestimation exposed by the formal selective DRAM run. The final
   preflight combines deterministic full-cohort device waves with calibration-role compute slack;
   all earlier formal points were invalidated by implementation hash and rerun.
6. Replaced the target manuscript's expected full-cohort win and Figure 7 placeholder with the
   measured negative result. Automatic fallback, failure recovery, durable SSD, and source-state
   economics remain explicitly open.
7. Inserted Stage 4.5 between normal-path closure and failure semantics. Its objective is a stable
   complete-job advantage over paired exact with an equally favorable source tier and full
   standing-state/lifecycle accounting; compression and HBM residency are candidates, not
   assumptions.
8. Limited iteration to the program-selection role and complete-cohort 1/4-GPU HBM representative
   points. The 30-point matrix is repeated only once after a source plan changes the frontier.
9. Kept Stage 5 blocked until a capacity-accounted source policy passes in its declared operating
   regime. A resident input cannot win by omitting capture/preload or standing occupancy, and an
   out-of-regime cohort must fall through to exact.
10. Preserved the main research route: the Stage-2 semantic recovery and Stage-3 fused resident
    operator remain valid, while the current FP16 source representation is rejected as an
    end-to-end design.

## Revision 10 (2026-07-27): Stage-4.5 direct-old-K/V source plan

Scope: source/state-footprint repair under
`cohortkv_single_config_stage4_5_frozen_v1`; still adaptive seed-0 development evidence.

1. Measured matched HBM/pinned-DRAM ceilings and retained the Stage-4 normalized-capsule result as
   the cold/host negative boundary.
2. Implemented a pinned-DRAM normalized-capsule reclaiming candidate. It beats exact after setup
   but retains about 17.86 GB of host state and preloads in 24.7–39.5 seconds, so it is not the
   primary plan.
3. Established that every source model's stacked K/V projection is full row rank with condition
   numbers 5.97–10.74, then composed its minimum-norm right inverse with the deployed affine.
4. Published three verified FP16 direct-old-K/V programs totaling 100.78 MB and eliminated all
   additional per-record `Norm(x)` state.
5. Reapplied the unchanged certificate. All three source pairs select direct compiled repair;
   minimum worst-view recovery is 0.8810 and maximum cost is 0.0368× exact.
6. Ran the actual fused transform over all 682 records and 17.822B valid elements, including the
   final-test partition only for label-free transport correctness. It has zero tolerance
   mismatches and maximum absolute error 0.01172.
7. Ran one correctness pass, one warmup, and five complete repetitions per method at 1/2/4 GPUs.
   Direct compiled takes 0.930/0.494/0.255 seconds versus paired raw-history-HBM exact at
   18.695/9.729/4.766 seconds; every compiled repetition is below every exact repetition.
8. Added extent-wise old-cache reclamation and capacity preflight covering old K/V, model,
   programs, maximum replacement wave, and a 2-GiB margin. Final old-K/V bytes are zero.
9. Froze the policy to direct old K/V only when capacity, old-state availability, and program
   verification pass; all failed predicates select exact. Transactional fallback execution is
   deliberately left to Stage 5.
10. Updated the target manuscript to claim only the existing-old-K/V hot-HBM regime and to keep
    cold filesystem, SSD, automatic tiering, failure recovery, and replication open.

## Revision 11 (2026-07-27): Stage-4.6 repeated-update lifecycle contract

Scope: planning/protocol correction only; no Stage-4.6 implementation or result is claimed.

1. Identified that Stage 4.5 starts from exact source-version K/V and does not certify a migrated
   approximation as input to another adjacent direct program.
2. Reclassified theta0/theta4/theta10-to-theta11 as a controlled one-target workload rather than
   a continuous lifecycle experiment.
3. Inserted Stage 4.6 before guard/failure work. Every cache at every update must choose exactly
   one of lightweight migration or exact refresh; stale reuse is not a normal third action.
4. Fixed the experiment to one KuaiRand 4+12, seed-0, 16L/H512, one-A40 hot-HBM configuration:
   exact theta0 K/V followed by 11 recursive updates through theta11.
5. Fixed the first router to one calibrated accumulated-risk threshold plus a maximum consecutive
   migration depth. Exact refresh resets both risk and depth.
6. Restricted risk calibration to correction magnitude, one-hop error, and affine propagation on
   disjoint fit/selection records. Recommendation labels, task gain, constructed hotness, and the
   retired drift/JVP/Fisher route remain prohibited.
7. Required every update to report mixed-policy versus all-exact MeanRank, Catalog AUC,
   NDCG@100, Hit@100, state fidelity, exact fraction, migration depth, and measured GPU cost.
8. Set the desired operating region to roughly 0.2–0.3× cumulative all-exact cost at roughly
   0.8–0.9 or higher label-free cache/score/top-100 fidelity with small independent task-metric
   gaps, while requiring honest curve/negative reporting if the measured result misses it.
9. Allowed only a small shared selection-role threshold curve and one matched-budget random
   diagnostic. The complete 682-record chain runs the frozen operating point once; no new
   30-point systems matrix is permitted.
10. Kept Stage 5 blocked until the lifecycle-policy artifact is frozen.

## Revision 12 (2026-07-27): Stage-4.6 balanced lifecycle freeze

Scope: completed single-configuration lifecycle evidence under
`cohortkv_single_config_stage4_6_lifecycle_development_v1`.

1. Compiled and verified all 11 adjacent theta0-to-theta11 direct-old-K/V programs from the
   disjoint 40-record fit role.
2. Built recursive transition DAGs for fit and program-selection roles and confirmed that global
   correction magnitude has negligible one-hop error ranking value.
3. Implemented a fused K/V norm-ratio sketch. Its selected threshold exceeds matched-random p95
   but produces unacceptable exact-refresh waves: 0%–61.7% on selection and 0.15%–65.1% on the
   diagnostic complete chain.
4. Rejected the threshold as the frozen policy because cumulative cost/fidelity alone omits a
   per-update maintenance-peak objective. Preserved both threshold full-chain artifacts as
   explicitly named diagnostics.
5. Replaced it with deterministic age/deadline scheduling. Label-free fit edge severity maps the
   11 updates to 15%–25% exact budgets; depth-four caches are mandatory exact, then older caches
   win remaining slots with a stable SHA256 tie-break.
6. Selected the bounded 20% base/5% severity-amplitude point on 60 program-selection records:
   0.2305× cumulative cost and 0.9542 minimum three-view fidelity.
7. Passed the independent 60-record recursive certificate at 0.2142× cumulative cost, 0.2814×
   maximum step cost, 15%–25% exact refresh, and minimum cache/score/top-100 values
   0.9613/0.999759/0.9898.
8. Ran the complete 682-record chain once at the frozen point. It costs 0.2134× cumulatively,
   stays below 0.2543× per step, refreshes 14.956%–25.073% after rounding, and reaches minimum
   0.9632/0.999950/0.9918.
9. Persisted and mechanically rebuilt all 7,502 record/update lineage rows, proving previous-output
   consumption, exact reset, maximum depth four, record coverage, and adjacent program hashes.
10. Froze the policy and summary artifacts, updated the manuscript to make no selector-optimality
    or organic-traffic claim, and admitted Stage 5 guard/fallback/failure work.
