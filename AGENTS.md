# Repository agent notes

## Research workflow — applies to code and experiments

EvoKV is a paper research repository. Optimize for fast, credible idea
validation and simple, readable implementations. Production readiness,
exhaustive edge-case handling and test coverage are not project goals.

- Start with the smallest implementation and experiment that can answer the
  current research question. Reuse existing code and inspect the signal before
  building a general framework or scaling up.
- Implement the inputs and execution modes the experiment actually uses.
  Add boundary checks only for a concrete risk to numerical correctness,
  evidence, data causality or expensive execution. Do not build speculative
  compatibility layers, generic validators, fallback stacks or exhaustive
  configuration support.
- Defer serialization, concurrent lifecycle handling, fault recovery and
  deployment machinery until the experiment or paper claim needs them.
  A synchronous in-memory prototype is a valid first implementation.
- Keep exploratory configuration and notes lightweight. Reuse an existing
  experiment description; do not create a new contract, audit document or
  approval step for every small code/configuration change. Small development
  probes within an authorized task can proceed directly. Long training and
  formal population runs retain the authorization requirements below.
- Preserve scientific validity: causal data, honest controls and metrics,
  held-out separation, reproducible essential settings and retained evidence.
  These checks serve the research question; they do not require industrial
  hardening or an exhaustive engineering checklist before exploration.

## Current direction

The current paper Design is described in `/home/gkl/work/paper/main.tex`:
personalized read correction, write-time causal influence summaries, shared
translation and state renewal. It has not been implemented or validated as a
complete method. Historical adaptation interfaces do not establish its
effectiveness; retained interfaces and their limits are in `docs/design/plan.md`.

On 2026-10-01 the user reopened Design 1 method selection: achieve at least
80% Full−Reuse AUC-gap recovery through read-time correction with at most 15%
additional FLOPs under the fixed benchmark's Full−Reuse history-cost denominator.
Around 10% cost is acceptable. Q/H motivate feasibility; their architecture and
quality are not constraints or an upper bound. Summaries, inputs, generators,
correction formulas and fitting objectives are replaceable candidates. The
59.45% first prototype is a retained development reference, below this target.
Current choices and the next implementation sequence live in `docs/design/plan.md`.
The user has now authorized implementation and development evaluation of the
two candidates in that plan, including their canaries and fixed 10k-user runs.
Those candidates and a focused objective/producer-scope follow-up are complete.
The current selected Round20 point stores only Item64 hidden states and applies
factored attention plus Response256: 84.98% recovery / 10.46% added FLOPs / 16.67%
extra persistent K/V-relative state on the fixed Medium V4→V5 10k-user panel.
This meets the three criteria after the user's 2026-10-02 storage constraint.
Its fresh calibration uses 64 Item epochs and 100 Response epochs with
differentiable compact Item reads; fitting cost is 8.66% and serving extra
compute is 1.80%, under the unchanged denominator.
Round19 confirmed algebraically equivalent compact inference and retained all
four negative reduced-Item-epoch points. Round20 is complete; no new candidate
or Design2/3 job is queued. Other edges and continuous adaptation remain unvalidated.
Round18's 83.65%/13.46% point needed a complete extra K/V copy and is only a reference.
Its sources remain, including its frozen Round11 dependency. All earlier outcomes
remain. The user asked to continue exploration
while productive, authorized paths remain; completing a batch below target is
not by itself a reason to pause for another instruction. Round 8's genuine
rolling calibration and immutable per-row producer/birth context did not improve
recovery (42.94%/42.04%). Round 9's full-rank affine K/V view also did not improve
it (40.73%/9.40%). Round 10's KV64 with16/64queries also remained below target
(60.11%/59.13%). Round 11 adds only Current item embeddings as read-view inputs,
using real cache-row IDs and separate lookup-byte accounting; its four-GPU
evaluation is complete. Round12 refined this same checkpoint for8epochs at1e-4;
response/joint Full-logit fitting recovered77.87%/78.95%, with full inherited costs.
Round13 joint12 reached76.90%/14.677%, not an improvement. Round14 alternating
pure/append768 refinement reached48.97%/14.776%, not an improvement. Round15 freezes
the successful Round11 mapping and learns six Current-producer residual scales;
it reached60.06%/14.7796%, and a fixed native-layer0-identity control reached69.49%/11.8381%.
Round16 fresh Item64 with16 recent-history queries reached78.03%/11.836%.
Round17's batch-centered teacher logit residuals reached78.38%/13.987%, not an
improvement. Round18's complete evaluation reached AUC0.669820, 83.65% recovery
and13.465% complete added cost, including inherited calibration and teacher access.
Users, requests, Full/Reuse controls and denominator stayed fixed. That round is
complete; the compact-state continuation has also completed. Settings/status are in
the existing plan; completed rounds do not imply a request to rerun them.

The protected main assets are Yambda processing and streaming training,
18 selected Medium/Large/Max V0–V5 models, their 15 adjacent Full/Reuse edges,
the four selective-recompute baselines, current nonlinear Q-v5/H-v4 motivation,
and the latest RecFlow six-model A–F development chain. Model endpoints are
fixed in `docs/unified_training_2026_09/model_versions.md`. Current source,
data, results and figure entry points are mapped in `docs/README.md`.
Current retention, historical dependencies and recovery limits are maintained
in `results/README.md`; retired branches are not active experiment queues.
These status notes do not authorize new long jobs.

The paper studies Transformer recommender state compatibility across model
releases; the concrete experimental models use HSTU. The motivation is that a new model
can improve Full quality while persistent K/V produced by the parent model
prevents that improvement from being fully realized. The repository documents
this through one conceptual design, one concrete experimental design and one
sealed motivation/observation record.

RecFlow is an isolated generative development track. The authorized seed17,
complete-epoch 2L/6L probes and 4096-user A–F expansion are complete; do not
relaunch them. Current settings, decision history and remaining cache questions
live in `docs/recflow/plan.md`; all outcomes, including failed free-catalog
confirmation and reused-day exploration, live in `results/recflow/README.md`.
Keep the frozen initial 1M catalog and uniform1000 NDCG@50 setting; this is
candidate ranking, not stable free generation. Do not retune cutoffs or pools
on later days. The expanded chain is not formal phi admission or evidence of
cache compatibility. Existing authorization covers the recorded six-layer
development/expansion and motivation diagnostics, not ten-layer/final-role
training, new seeds or Yambda theta3. Do not ask again for an already authorized
launch, and do not treat a completed launch as an instruction to rerun it.
All outcomes and random checks remain; preferred gain magnitudes are descriptive,
not thresholds. Numerical/data/lineage failures stop descriptive training chains.
Protect sealed execution sources while any dependent job is active.

Authoritative entry points:

- `docs/README.md`: document map and maintenance rules;
- `docs/paper_design.md`: pointer to the paper design and implementation boundary;
- `docs/experimental_design.md`: concrete architecture, data, version training,
  evaluation definitions and evidence boundaries;
- `docs/design/plan.md` and `docs/design/iterations.md`: reusable six-layer
  adaptation interfaces, implementation boundary and historical exploration;
- `docs/motivation_observations.md`: current observed motivation and results.
- `figures/README.md`: generators and source records for the actual paper figures.
- `scripts/README.md`: retained executable workflows and shared dependencies.

The paper text is maintained in `/home/gkl/work/paper/main.tex`. Its Design
chapter is the narrative source; do not restore the deleted parallel Design 1
draft or use Sketch-to-Sketch as the current method name.

## Evidence and authorization

- Current motivation contracts, hashes, raw seals and adjudications must be
  preserved. Retired-branch documents and historical control code were removed
  during the 2026-08-25 cleanup. The formerly referenced
  `results/checkpoint_cleanup_2026-08-24.md` is absent as of 2026-09-22;
  current retention and missing-archive limits are recorded in `results/README.md`.
  Do not silently recreate retired evidence or missing historical records.
- The first 2026-09-05 cleanup removed obsolete entry points. The user's
  subsequent authorization also removed retired Small weights and obsolete
  raw results; see `results/README.md` for the scope, historical archive metadata,
  retained dependencies and recovery limits (the two 2026-09-05 archives are
  currently missing locally). Preserve the selected six-layer Medium,
  ten-layer Large and sixteen-layer Max assets, current paper raw evidence,
  and their dependencies. The 2026-09-30 exact deletion receipt is
  `results/history/repository_cleanup_2026_09_30.json`.
  Historical contracts and archived conclusions are not an active to-do list.
- The Yambda-50M 8L/H256/context1024 F-only seed17 architecture pilot is
  complete and positive, but it is not a Yambda-500M population-scale result.
- The obsolete 8L M1 N/R/F seed17 run was stopped before its first checkpoint
  for a cost-only scope change; it has no H/S/quality interpretation.
- Retained implementation includes Yambda-500M/5B processing, fixed-UID
  populations, compact item mapping, manifests, HSTU-native training and the
  completed three-scale motivation evaluations. Subsequent Design development
  starts from the frozen six-layer Medium interfaces in `docs/design/plan.md`;
  completed three-scale motivation runs are not authorization for new jobs.
- The user removed the blanket target-KV-fitting restriction on 2026-09-06.
  Summary/KV-derived supervision, reconstruction losses and shared Translator
  calibration are allowed within the adaptation research. Do not gate a design
  on whether it can be called KV fitting. Evaluate candidate read-correction
  mechanisms by actual quality/cost; summary and translation modules are options.
  Routine implementation and small development calibration follow the plan;
  they do not need another permission step merely for this supervision.
  Keep fitting/development/final evaluation separate and report teacher access
  and its cost. Historical sealed contracts still describe their original runs.
- Any Medium/Large long training requires a prospective contract, resource
  estimate, passing canary and explicit user launch.
- Theta3 remains untouched. Its data/release/admission/metric/failure contract
  must be sealed before training or reading any theta3 result.
- The prospective RecFlow three-day chain is `phi0..phi5`; development probes
  are not admitted releases. These names must not be conflated with the
  untouched Yambda theta3. RecFlow uses seed17 only under the latest user scope;
  preserve earlier multi-seed results without adding new seed replications.
  The latest instruction authorizes complete-epoch development runs and
  conditional six-layer progression after stable A/B (and optional C) checks.
  Record prospective settings, resources and a focused canary before long jobs,
  and use tmux; do not ask again for launch already covered by this instruction.
  The daily-window checks and authorized six-layer development-data expansion
  are complete; current settings and remaining questions are in the RecFlow plan.
  Ten-layer and final-population qualification remain outside this scope.
  Do not mistake the daily development branches for admitted releases.
- RecFlow initial and updated models must each clearly outperform uniform
  random video ranking under the same catalog/candidate pool, request panel,
  cutoff and OOV denominator. Parent/current improvement alone is insufficient.
  Report the matched random expectation and random-policy variation for every
  explored metric configuration; do not select favorable cutoffs or pools to
  hide failures. Beating other recommender architectures is not required.

Diagnostic exact-KV splices are interventions, not executable actions. Only the
frozen dependency-closed actions may enter scale frontiers; do not add actions or
predictor complexity on the scale development point.

## Code layout

- `src/hstu_kvcache/models/`: HSTU, persistent K/V and dependency-closed
  transitions.
- `src/hstu_kvcache/data/`: Yambda readers, manifests, release windows,
  population maps and frozen workload/release data primitives.
- `src/hstu_kvcache/adaptation/`: experimental summary, Translator, paired reader
  and in-memory multi-version state implementation.
- `src/hstu_kvcache/design_one/` and `scripts/design_one/`: the retained first
  read-correction candidate, frozen benchmark, native replay and fitting helpers;
  the current candidate's representation is not the next method's specification.
- `src/hstu_kvcache/recflow/` and `scripts/recflow/`: isolated RecFlow preparation,
  structured generation and bounded development evaluation.
- `configs/contracts/`: immutable development evidence and prospective scale contracts.
- `scripts/`: current data, foundation, Full-only and motivation entry points;
  `scripts/design/` retains a small reusable interface closure, not the retired
  AUC/preview/native experiment launch suite.
- `figures/src/`: all Python plotting code; read existing results without running experiments.
- `tests/`: current motivation time causality, cache lineage, manifests and executor.
- `results/`: development evidence; presence does not imply paper qualification.

Keep model modules independent from orchestration. Reuse current primitives;
do not clone the full pipeline for the scale point.

## Development rules

- Use `rg` or `rg --files` for search and `apply_patch` for edits.
- Preserve unrelated dirty-worktree changes.
- Update existing Markdown and directory READMEs in place. Keep current status,
  entry points and historical evidence distinct; do not create cleanup reports
  or duplicate progress documents for repository maintenance.
- Read before deleting. Remove obsolete execution code explicitly and keep the
  current motivation contracts, hashes and results internally consistent.
- Do not revive deleted D1/D2/D3, KuaiRand mainline, neutral-readout repair,
  sampled next-listen candidates, old Q_main/controller/frontier or P4 routes.
- Do not tune workload, release, history, task weights, seeds, metrics, action
  set, predictor or probe rate using qualification/scale outcomes.
- Keep protocol decisions label-free: no future-label scheduling, score mixing,
  selected-edge reporting or artificial K/V perturbation.
- Design 1 first establishes read-time correction at the current 80% recovery /
  15% added-compute objective. Implement the state and read interfaces each
  candidate uses; learned influence propagation and lifecycle policies belong
  to Design 2/3. A runnable pipeline or the first 59.45% result is intermediate.
- Continuous adaptation means the actual cache lifetime across releases:
  mixed producers, appends, evictions, long-lived old state and inherited error.
  Reinitialized single-edge runs do not establish continuous behavior.
- Keep model admission separate from cache compatibility; low H/S is a valid
  No-op condition.
- Do not treat a fixed training endpoint as a release. Seal Parent/Current
  Full-only admission first; only then unlock Reuse evaluation for that edge.
- Build the lineage incrementally. A rejected candidate leaves the serving
  parent and cache lineage unchanged.
- Report all frozen seeds. Training seed, not request count, is the repeat unit.
- For RecFlow, audit request-group boundaries before calling them sessions;
  rebuild long history from chronological realshow events, exclude the complete
  target request from the prefix, and never treat unexposed stage candidates as
  observed user negatives or persistent-history events.

## Verification

Choose the smallest check that addresses the actual change. Neither model
imports nor the full pytest suite are mandatory after every code/dependency edit.

- Text, navigation, plotting and simple file moves: inspect the affected text,
  links, imports or plot as applicable; no model suite.
- Local numerical, data or evaluator changes: run the relevant existing test
  or one tiny reference comparison. Add a focused regression test only when
  the failure could materially change an experiment's conclusion.
- Experiment changes: use a small development sample to check the hypothesis,
  relevant correctness and resource cost. A suitable existing check can serve
  as the canary; do not repeat equivalent checks under different names.
- Run the full suite only when a broad shared change has effects that cannot
  be covered adequately by focused tests, or the user explicitly requests it.
  Once the relevant checks pass, stop; do not broaden testing by habit.

Keep tests for important formulas, causal data/cache behavior, evaluation
aggregation and meaningful regressions. Do not target coverage percentages,
test every input combination, mirror implementation constants, or preserve
formatting and obsolete-route tests without a current research need. Remove
redundant tests when encountered; do not add an exhaustive test-audit phase to
routine work. See `tests/README.md` for scoped entry points.

Do not build the paper unless the user requests compilation.

The user expanded the scale allowlist to GPU 0/1/2/3. A scale model uses at most
one four-rank FSDP job at a time; seeds/releases are queued serially. Parallelize
CPU mapping, joins and aggregation when safe. Every long scale job needs a
focused canary first.
For the authorized RecFlow development, the 2026-09-18 instruction additionally
allows internal multi-GPU execution and independent experiments in parallel on
GPU 0/1/2/3. Choose placement from measured throughput, preserve global batches
and evaluation aggregation, and avoid collisions with active jobs. This does
not expand the Yambda scale or seed scope.
Estimate each experiment's runtime from a small probe or a stated calculation.
Use detached tmux execution for jobs expected to exceed 30 minutes, retain the
log/exit status, and resume analysis when they finish. Monitor shorter jobs
directly; tmux does not replace long-job launch authorization.

## Safety and storage

- Git tracks source, contracts and compact evidence needed to explain results
  and reproduce figures. Datasets, weights, raw payloads, runtime output and
  recovery archives stay local under `.gitignore`. Preserve local evidence
  when removing generated artifacts from the index; do not rewrite Git history
  merely to tidy the current tree.
- Do not launch long experiments without the required authorization.
- Preserve current paper contracts, hashes, raw seals, adjudications and
  invalidations. Retired-branch conclusions and failure records are archived;
  do not claim their removed raw data or weights still exist. Never remove
  unfavorable rows from a retained experiment as part of cleanup.
- Do not retain redundant checkpoints, expanded manifests, logs or temporary
  results by default.
- Before destructive cleanup, resolve exact targets; generated bytecode/cache is
  safe to regenerate, while evidence artifacts are not.
