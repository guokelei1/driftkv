# Repository agent notes

## Current direction

This repository implements the 37D EvoKV route: release-time convergence of persistent user K/V state under bounded compute and I/O. The current question is no longer whether staleness exists. P7/P8 established a development `H -> S -> quality` chain for the Yambda-50M Explicit Feedback workload, and P9.2 located diagnostic layer/history-position recovery structure.

The active systems question is whether that structure can become dependency-closed partial migration with a real cost advantage over Exact-All.

Authoritative entry points:

- `docs/project_compact.md`: end-to-end project state and evidence boundary;
- `docs/current_route.md`: concise current authorization and next steps;
- `docs/p9_plan.md`: active tomography/action-space plan;
- `docs/newset.md`: full 37D technical protocol;
- `docs/legacy/README.md`: retired routes and claims that may not be revived.

## Evidence and authorization

- P7: long-state value `H` qualified on F; N is the short-term negative control and R did not qualify.
- P8: R0 is a strict No-op control; R1/R2 establish cross-version `S`; M1-R2 also establishes task-quality harm from Reuse.
- P9.0/P9.1/P9.2: evidence seal, user-level H/S distributions, and all 24 coarse tomography cells are complete.
- Current work: diagnostic-action quality companions, risk concentration, representative 2-D tomography, dependency-closure audit, legal executor, and No-op/Partial/Exact frontier.
- Not authorized: controller training, theta3/blind qualification, tuning the P8 model/release chain, or paper-level claims.

Diagnostic exact-KV splices are interventions, not executable actions. Only dependency-closed actions with measured token-layer work, I/O, history reads, and runtime may enter a system frontier.

## Code layout

- `src/hstu_kvcache/models/`: HSTU, candidate-conditioned residual scoring, and persistent K/V primitives.
- `src/hstu_kvcache/data/`: KuaiRand and Yambda readers, workload/manifests, compact materialization, release snapshots, and P7/P8 data contracts.
- `configs/contracts/`: frozen workload, training, qualification, release-chain, and P9 contracts/results.
- `scripts/`: active P9 entry points plus retained P5-P8 evidence-generation/audit tools.
- `tests/`: focused regression tests for time causality, cache lineage, manifests, frozen base, P7/P8, and P9.
- `results/`: development evidence and sealed artifacts; presence does not imply paper qualification.

Keep core model modules independent from orchestration. Add profiler/executor/scheduler modules only after their contract is defined; a scheduler remains unauthorized until the P9 frontier demonstrates a state-level opportunity.

## Development rules

- Use `rg` or `rg --files` for search and `apply_patch` for edits.
- Preserve unrelated dirty-worktree changes.
- Read before deleting. Retire obsolete evidence in documentation instead of silently rewriting history.
- Do not revive deleted D1/D2/D3 routes, old controller/frontier numbers, neutral-readout repair claims, or sampled next-listen candidate engineering.
- Do not tune workloads, releases, history lengths, task weights, seeds, or metrics using qualification outcomes.
- Keep model scoring raw and protocol decisions label-free: no post-hoc score mixing, selected-edge reporting, future-label scheduling, artificial K/V perturbation, or target-KV fitting.
- Keep model admission separate from cache compatibility. A quality-qualified model with low `H` or `S` is a valid No-op condition.
- Report all frozen seeds. Independent training seeds, not requests, are the formal repeat unit.

## Verification

At minimum run:

```bash
PYTHONPATH=src python -c "from hstu_kvcache.models import HSTU, HSTUKVCache; from hstu_kvcache.data import YambdaTrace, StreamingDataPlan"
PYTHONPATH=src pytest -q
```

Use focused canaries before long runs. For the current P9 work, use only GPU 0 and 1 unless the user changes the allocation; parallelize CPU-only joins and aggregation when safe.

## Safety and storage

- Do not launch long experiments unless requested or already authorized by the current route.
- Preserve frozen contracts, hashes, raw-score seals, negative results, and invalidation records.
- Do not retain redundant large temporary artifacts by default.
- Before destructive cleanup, resolve exact targets and prefer a recoverable staging move.
