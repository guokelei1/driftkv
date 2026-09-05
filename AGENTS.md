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

The current EvoKV design is Cross-Version Cache Adaptation: write-time
summarization, release-time translation, read-time correction, and state
maintenance across releases. The existing code provides the model foundation,
Full/Reuse evaluation and paper diagnostics; the new adaptation method remains
prospective, not an implemented or validated system.

The paper studies Transformer recommender state compatibility across model
releases; the concrete experimental models use HSTU. The motivation is that a new model
can improve Full quality while persistent K/V produced by the parent model
prevents that improvement from being fully realized. The repository documents
this through one conceptual design, one concrete experimental design and one
sealed motivation/observation record.

A secondary RecFlow track remains prospective only; it is not part of the
current motivation result and has no long-training authorization.

Authoritative entry points:

- `docs/README.md`: document map and maintenance rules;
- `docs/paper_design.md`: stable conceptual paper design and comparison boundary;
- `docs/experimental_design.md`: concrete architecture, data, version training,
  evaluation and phase plan;
- `docs/motivation_observations.md`: current observed motivation and results.
- `figures/README.md`: generators and source records for the actual paper figures.
- `scripts/README.md`: retained executable workflows and shared dependencies.

The paper text is maintained in `/home/gkl/work/paper/main.tex`. Its Design
chapter is the narrative source; do not restore the deleted parallel Design 1
draft or use Sketch-to-Sketch as the current method name.

## Evidence and authorization

- Current motivation contracts, hashes, raw seals and adjudications must be
  preserved. Retired-branch documents and historical control code were removed
  during the 2026-08-25 cleanup; the remaining result scope is recorded in
  `results/checkpoint_cleanup_2026-08-24.md` and must not be silently recreated.
- The first 2026-09-05 cleanup removed obsolete entry points. The user's
  subsequent authorization also removed retired Small weights and obsolete
  raw results; see `results/README.md` for the scope, archived deletion manifest,
  retained dependencies and recovery limits. Preserve six-layer Medium and
  ten-layer Large assets, current paper raw evidence, and their dependencies.
  Historical contracts and archived conclusions are not an active to-do list.
- The Yambda-50M 8L/H256/context1024 F-only seed17 architecture pilot is
  complete and positive, but it is not a Yambda-500M population-scale result.
- The obsolete 8L M1 N/R/F seed17 run was stopped before its first checkpoint
  for a cost-only scope change; it has no H/S/quality interpretation.
- Current implementation scope is Yambda-500M audit, fixed-UID population,
  compact item mapping, manifests, HSTU-native foundation, Full-only release
  evaluation and motivation correctness canaries.
- Any Medium/Large long training requires a prospective contract, resource
  estimate, passing canary and explicit user launch.
- Theta3 remains untouched. Its data/release/admission/metric/failure contract
  must be sealed before training or reading any theta3 result.
- RecFlow checkpoint names are `phi0..phi3` and must not be conflated with the
  untouched Yambda theta3. RecFlow Medium training remains gated by D0/D1 audit,
  a prospective contract, canary, resources and explicit launch.

Diagnostic exact-KV splices are interventions, not executable actions. Only the
frozen dependency-closed actions may enter scale frontiers; do not add actions or
predictor complexity on the scale development point.

## Code layout

- `src/hstu_kvcache/models/`: HSTU, persistent K/V and dependency-closed
  transitions.
- `src/hstu_kvcache/data/`: Yambda readers, manifests, release windows,
  population maps and frozen workload/release data primitives.
- `configs/contracts/`: immutable development evidence and prospective scale contracts.
- `scripts/`: current data, foundation, Full-only and motivation entry points only.
- `figures/src/`: all Python plotting code; read existing results without running experiments.
- `tests/`: current motivation time causality, cache lineage, manifests and executor.
- `results/`: development evidence; presence does not imply paper qualification.

Keep model modules independent from orchestration. Reuse current primitives;
do not clone the full pipeline for the scale point.

## Development rules

- Use `rg` or `rg --files` for search and `apply_patch` for edits.
- Preserve unrelated dirty-worktree changes.
- Read before deleting. Remove obsolete execution code explicitly and keep the
  current motivation contracts, hashes and results internally consistent.
- Do not revive deleted D1/D2/D3, KuaiRand mainline, neutral-readout repair,
  sampled next-listen candidates, old Q_main/controller/frontier or P4 routes.
- Do not tune workload, release, history, task weights, seeds, metrics, action
  set, predictor or probe rate using qualification/scale outcomes.
- Keep protocol decisions label-free: no future-label scheduling, score mixing,
  selected-edge reporting, artificial K/V perturbation or target-KV fitting.
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
