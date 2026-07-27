# CohortKV single-configuration Stage 1 frontier

## Status

Stage 1 is complete under `cohortkv_single_config_stage1_frontier_v1`. The implementation and
measurement use the frozen KuaiRand 4+12, theta0/theta4/theta10→theta11, 16-layer, seed-0
development cell. The raw local result is
`results/system/cohortkv_single_config_full_chain_v1/stage1_frontier_seed0.json`; its frozen
checked-in summary is
`configs/cohortkv_single_config_v1/stage1_frontier_summary.json`.

This remains adaptive seed-0 design evidence. It is not a new training-seed replication. The
separate Stage-2 result now validates the compiled deployed-FP16 source path; it does not change
this Stage-1 FP32 measurement boundary.

## Implementation closure

The new `selective_contiguous` reference is separate from `migrate_contiguous_cache`:

- it copies source old K/V exactly outside the chosen interval;
- it starts current-model execution from the old pre-block hidden state, or raw embeddings when
  the interval starts at layer zero;
- it executes full current blocks before the terminal interval layer;
- it executes only current `Norm + Wk/Wv` at the terminal layer;
- the full-depth interval equals exact current-model K/V.

Residual-\(p\) now also has a representation-specific reference that consumes only raw history and
the old hidden suffix `[p..L-1]`. Tests reject an incomplete suffix before execution. These tests
close the Stage-0 concern that one transition state or the default normalized capsule might be
mistaken for sufficient residual state.

## Measurement boundary

Each source pair evaluates the same disjoint roles:

| Role | Records | Use |
|---|---:|---|
| Program selection | 60 | all 53 intervals plus compiled, cheap, p4, p8, reuse, exact |
| Certificate | 60 | the five frozen per-width winners plus reuse and exact |
| Final test | 522 | not evaluated in Stage 1 |

The 177 selection points are resident FP32 algorithm measurements on three A40s. Each batch has
one untimed warmup and three CUDA-event repetitions. Source reads, serialization, destination
allocation, and publication are outside this Stage-1 boundary and remain Stage-4 work.
Recommendation labels are never read.

## Result

For every source pair, the highest-worst-view selective action is \(m=12\), layers 0–11. It starts
at layer zero, so the frozen full-cohort diagnostic needs old FP16 K/V and raw history but no
transition-hidden shard.

| Source | Compiled cost/exact | Compiled worst recovery | Best selective cost/exact | Best selective worst recovery |
|---|---:|---:|---:|---:|
| theta0 | 0.0656 | 0.8787 | 0.6973 | 0.4530 |
| theta4 | 0.0663 | 0.8755 | 0.6976 | 0.4850 |
| theta10 | 0.0664 | 0.9258 | 0.6976 | 0.4495 |

Compiled repair has both lower resident cost and higher worst-view recovery than every one of the
53 selective intervals for all three pairs. No selective per-width winner passes the frozen 70%
cache/score/top-100 certificate. The publishable action therefore falls back to exact, as the
protocol required.

This failed certificate does not erase the external baseline. Stage 4 will execute the frozen
\(m=12\), layers 0–11 action through the common destination transaction as a diagnostic external
baseline, report that its certificate failed, and retain exact as the publishable fallback. It
must not call that system row a certified or publishable synchronized target.

## Commands

```bash
python scripts/evaluate_cohortkv_stage1_frontier.py --validate-only
torchrun --standalone --nproc-per-node=3 \
  scripts/evaluate_cohortkv_stage1_frontier.py
python scripts/freeze_cohortkv_stage1.py
python scripts/freeze_cohortkv_stage1.py --check
```

## Remaining boundary

Stage 2 has reapplied the frozen compiled certificate to serialized FP16
capsules/programs/output and frozen executable plans. Before Stage 4, the selective diagnostic
still needs serialized FP16 correctness and independent runtime tuning per destination and GPU
count. The full-cohort destination result remains open.
