# Repository agent notes

## Current direction

This repository implements the 37D EvoKV route: release-time convergence of
persistent user K/V state under bounded compute and I/O.

The current paper mainline is HSTU-native recommendation-model state
compatibility across model releases. The current motivation is that a new model
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

## Evidence and authorization

- Current motivation contracts, hashes, raw seals and adjudications must be
  preserved. Retired-branch documents and historical control code were removed
  during the 2026-08-25 cleanup; the remaining result scope is recorded in
  `results/checkpoint_cleanup_2026-08-24.md` and must not be silently recreated.
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

At minimum run:

```bash
PYTHONPATH=src python -c "from hstu_kvcache.models import HSTU, HSTUKVCache; from hstu_kvcache.data import YambdaTrace"
PYTHONPATH=src pytest -q
```

The user expanded the scale allowlist to GPU 0/1/2/3. A scale model uses at most
one four-rank FSDP job at a time; seeds/releases are queued serially. Parallelize
CPU mapping, joins and aggregation when safe. Every long scale job needs a
focused canary first.

## Safety and storage

- Do not launch long experiments without the required authorization.
- Preserve frozen contracts, hashes, seals, negative results and invalidations.
- Do not retain redundant checkpoints, expanded manifests, logs or temporary
  results by default.
- Before destructive cleanup, resolve exact targets; generated bytecode/cache is
  safe to regenerate, while evidence artifacts are not.
