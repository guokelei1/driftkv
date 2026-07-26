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
