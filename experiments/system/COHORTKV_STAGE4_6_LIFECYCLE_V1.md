# CohortKV Stage 4.6 continuous lifecycle v1

> Naming note: `continuous` is the frozen protocol name for 11 successive model-version edges. It
> does not mean concurrent updates or a high-frequency update stream.

## Status

Frozen on 2026-07-27 under
`cohortkv_single_config_stage4_6_lifecycle_development_v1`.

This is one KuaiRand 4+12, seed-0, 16L/H512, history-2048, one-A40 development
experiment. It begins with exact theta0 K/V and executes the actual 11 adjacent
updates through theta11. Every migrated output is the next update's real input.
It is not an organic request trace, a multi-seed result, or a new endpoint matrix.

Frozen artifacts:

- `configs/cohortkv_single_config_v1/stage4_6_lifecycle_policy.json`;
- `configs/cohortkv_single_config_v1/stage4_6_lifecycle_summary.json`;
- `results/system/cohortkv_single_config_full_chain_v1/stage4_6_certificate_chain_seed0.json`;
- `results/system/cohortkv_single_config_full_chain_v1/stage4_6_full_chain_seed0.json`.

## Policy amendment

The first candidate combined maximum migration depth with a per-cache norm-sketch
risk threshold. It had acceptable cumulative cost/fidelity and exceeded a
matched-random p95 diagnostic, but it omitted a per-step peak objective. Its
diagnostic 682-record chain exactly refreshed

```text
39 / 3 / 35 / 209 / 444 / 1 / 65 / 53 / 139 / 431 / 105
```

records across the 11 updates, or 0.15%–65.10% per step. This synchronized work is
an operational negative result. The threshold policy is retained as a diagnostic,
not frozen as the lifecycle.

The replacement is a deterministic balanced age/deadline heuristic:

1. compute each adjacent edge's label-free severity as the median fit-record
   one-hop cache-error q90;
2. rank severity into a configured exact fraction between 15% and 25%;
3. make every cache already at migration depth four mandatory exact;
4. fill the remaining exact quota by greater age, then a stable SHA256 tie-break;
5. migrate every other cache with the adjacent direct-old-K/V program.

The configured edge fractions are
`25/19/15/17/22/16/20/18/23/24/21%`. Exact refresh resets lineage and depth;
migration increments depth. Recommendation labels never enter edge severity,
policy selection, or routing. The action plan precedes execution, so the frozen
path does not compute and discard speculative migration candidates.

This is an empirically selected bounded heuristic, not an analytical or global
optimality claim.

## Results

| Evidence role | Records | Cumulative GPU cost / all-exact | Maximum step cost / all-exact | Exact-fraction range | Minimum cache fidelity | Minimum score cosine | Minimum top-100 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Program selection | 60 | 0.2305× | attributed-DAG diagnostic | 15.0%–25.0% | 0.9542 | 0.999936 | 0.9908 |
| Independent certificate | 60 | 0.2142× | 0.2814× | 15.0%–25.0% | 0.9613 | 0.999759 | 0.9898 |
| Complete recursive chain | 682 | 0.2134× | 0.2543× | 14.956%–25.073% | 0.9632 | 0.999950 | 0.9918 |

The complete-chain fractions differ from the configured 15%–25% bounds only by
nearest-record rounding. Exact counts are

```text
171 / 130 / 102 / 116 / 163 / 171 / 136 / 123 / 157 / 164 / 143.
```

The freeze checker rebuilds all 7,502 complete-chain lineage rows from the frozen
policy. It verifies previous-output consumption, exact reset, maximum depth four,
record coverage, and all 11 adjacent program hashes. Scheduler planning takes
23.4 ms in total across the 11 complete-cohort updates and is disclosed separately
from GPU cost.

Across the 522 final-test records, the maximum absolute per-step mixed-minus-exact
gaps are:

- MeanRank: 4.171;
- Catalog AUC: 8.35e-5;
- NDCG@100: 3.49e-4;
- Hit@100: 0.00384.

Reuse-to-exact recovery remains diagnostic because several denominators are near
zero. Full recomputation is the K/V reference, not a guaranteed ranking upper
bound.

## Measurement boundary

Old K/V and raw history are treated as hot-HBM sources. Mixed GPU cost includes
direct migration, exact-subset gather and replay, and publication. Offline
all-exact evaluation is excluded from mixed cost. Scheduler CPU time is measured
separately. Stage 4.5 already supplies matched full-cohort source/destination
system evidence; Stage 4.6 isolates recursive lifecycle cost and fidelity on one
GPU.

## Reproduction

```bash
python scripts/compile_cohortkv_stage4_6_edges.py --device cuda:0
python scripts/evaluate_cohortkv_stage4_6_lifecycle.py --phase fit --device cuda:0
python scripts/evaluate_cohortkv_stage4_6_lifecycle.py --phase fit-transitions --device cuda:0
python scripts/evaluate_cohortkv_stage4_6_lifecycle.py --phase selection --device cuda:0
python scripts/evaluate_cohortkv_stage4_6_lifecycle.py --phase policy --device cuda:0
python scripts/run_cohortkv_stage4_6_full_chain.py --role certificate --device cuda:0
python scripts/run_cohortkv_stage4_6_full_chain.py --role all --device cuda:0
python scripts/freeze_cohortkv_stage4_6.py
python scripts/freeze_cohortkv_stage4_6.py --check
```

Stage 5 may now connect this immutable policy to a semantic guard, automatic exact
fallback, transactional rework, and failure visibility. Those mechanisms are not
claimed by Stage 4.6.
