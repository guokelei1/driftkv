# Outline and figure plan

## One-sentence thesis

Streaming updates turn persistent HSTU prefix K/V into model-versioned derived state; CohortKV
compiles each source-to-target version cohort into a verified transformation, executes that
transformation as a one-pass capsule-to-K/V operator, and publishes a complete target-version
state set through an explicit destination job.

## Reader questions and section order

| Section | Reader question | Required answer |
|---|---|---|
| Abstract | What problem, mechanism, evidence, and boundary matter? | Cross-version stale K/V; cohort compiler + operator + engine; replicated algorithm evidence and controlled preliminary system evidence. |
| 1 Introduction | Why is this a systems problem now? | Streaming model versions coexist with persistent user-history state; reuse loses value, exact update is expensive. |
| 2 Background and scope | What exactly is cached and what is outside the paper? | HSTU serving semantics, versioned K/V, capsule, fixed destination update job. |
| 3 Motivation | Which observations force which designs? | Opportunity, non-oracle age/task quality, compilable structure, state-movement boundary. |
| 4 Overview | What is the end-to-end abstraction? | compile → execute → publish; version cohort crosses all three layers. |
| 5 Compiler | How is a valid shared program produced? | affine decomposition, fit, label-free certificate, progressive exact fallback. |
| 6 Operator | How does the program become efficient K/V? | fused projection/bias/mask/split, direct write, variable lengths. |
| 7 Destination engine | How is a whole target version produced? | per-cache migrate-or-exact lifecycle planning, waves, program residency, placement, destination transaction, manifest. |
| 8 Implementation | What exists today? | PyTorch/Triton runtime and four backend interfaces; status boundaries. |
| 9 Evaluation | What is established, and at what evidence level? | opportunity, compiler, operator/runtime, one-hop full-cohort result, and the fixed repeated-update lifecycle gate. |
| 10 Discussion and limitations | When does the approach not apply? | simplified HSTU, adaptive seed-0, repeated migration, storage, and serving-trace boundaries. |
| 11 Related work | What is genuinely different? | same-model restoration, cross-LLM reuse, KV serving/storage, recommender update systems. |
| 12 Conclusion | What should the reader retain? | version cohorts are execution units, not safety predictions; current evidence and next gate. |

## Motivation-to-design-to-evaluation map

This table appears near the end of Motivation so the reader can predict the rest of the paper.

| Observation | Design | Evaluation question |
|---|---|---|
| Stale reuse leaves a maintenance gap across evaluated streams. | Every stale cohort receives compiled repair; exact remains terminal endpoint. | How large and general is the opportunity? |
| Age and task gain are not calibrated safety signals. | Source/target pair is an execution key; label-free semantic certificate replaces admission. | Does fidelity/cost replicate even when task gates do not? |
| HSTU exposes old normalized states and current K/V projections. | Compile shared residual repair into one affine program. | How much K/V gap closes at what measured GPU cost? |
| Kernel gains can disappear in movement and publication. | Fused direct-write operator plus destination-oriented engine. | Does the gain survive the same host boundary, and what remains open at full destination scope? |
| A one-hop certificate does not bound recursively migrated inputs, and cumulative-only routing can create maintenance waves. | Edge-severity exact budgets plus age/deadline priority choose lightweight migration or exact reset. | Across theta0→theta11, what recommendation fidelity and per-step peak remain at what cumulative fraction of all-exact cost? |

## Figures

### Figure 1: Cross-version invalidation and job boundary

Question: why is ordinary cache reuse insufficient after a model update?

Visual:

- theta-v creates a persistent prefix capsule/K/V cohort;
- theta-t arrives;
- stale reuse and exact recomputation are endpoints;
- CohortKV transforms the cohort and publishes one target-version manifest;
- training and online request scheduling sit outside a dashed boundary.

Status: create as `figures/01_problem_and_scope.svg`.

### Figure 2: Three-layer architecture

Question: how does one cohort identity coordinate the system?

Visual:

- compiler keyed by `(source,target)` and its label-free certificate;
- program-resident fused operator;
- destination engine with HBM/DRAM/POSIX/remote adapters;
- manifest visibility only after complete coverage.

Status: create as `figures/02_architecture.svg`.

### Figure 4: Evidence ladder

Question: what is replicated versus preliminary?

Visual:

- 27-run compiled projection: 0.121× cost and 0.587 K/V recovery with CIs;
- seed-0 verified full-affine: about 0.064× and 0.887–0.936 recovery;
- controlled two-GPU host path: 903.7 records/s and 11.22× over matched BF16 exact;
- Stage-4 destination engine: normal path measured on all 30 points; current FP16 capsule loses to
  exact at all six matched HBM/DRAM endpoints.

The visual must label evidence class directly and avoid joining the stages into one statistical
series.

Status: created as `figures/03_evidence_ladder.svg`; it appears fourth in manuscript order.

### Figure 3: Host-staged execution pipeline

Question: where can an operator speedup be lost?

Visual:

- CPU capsule bucket;
- pinned H2D;
- resident program + fused kernel;
- D2H target K/V;
- asynchronous extent publication;
- bounded wave/queue and final manifest.

Status: created as `figures/04_execution_pipeline.svg`; it appears third in manuscript order.

### Planned lifecycle cost–fidelity figure

Question: can selective exact refresh prevent repeated-migration drift without losing the
lightweight path's cost advantage?

Visual:

- one fixed KuaiRand seed-0/one-A40 theta0→theta11 chain;
- all 11 per-version mixed-versus-exact recommendation gaps;
- cumulative cost/all-exact versus minimum cache/score/top-100 fidelity for threshold, periodic,
  and balanced selection candidates;
- the rejected threshold's refresh wave;
- the single frozen full-cohort balanced point, exact fraction, and migration-depth distribution.

Status: Stage 4.6 evidence is frozen; the final paper figure still needs rendering from
`stage4_6_lifecycle_summary.json`.

## Tables

1. Scope and non-goals.
2. Observation → design → evaluation map.
3. Aligned cross-dataset opportunity.
4. 3×3 capacity screen.
5. 27-run compiled migration summary.
6. Verified seed-0 compiler by source age.
7. Operator and two-GPU controlled benchmark.
8. Destination backend contract and evidence status.
9. Repeated-update lifecycle cost/fidelity.
10. Closest-work comparison.

## Figure discipline

- Every figure caption states the take-away, not only what shapes are shown.
- Preliminary and replicated evidence never share an unlabeled axis.
- A speedup always names its baseline and endpoint.
- Exact recomputation is called a K/V semantic reference, not a ranking-quality upper bound.
- HBM-versus-host endpoint differences are never described as operator speedups.
