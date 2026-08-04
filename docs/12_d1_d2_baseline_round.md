# D1/D2 Baseline and Selected XP Quality Foundation

Last updated: 2026-08-04

Status: selected development evidence and the baseline-first successor handoff. Checkpoint paths
and hashes are owned by [13_cross_dataset_stream_checkpoint_plan.md](13_cross_dataset_stream_checkpoint_plan.md).

## 1. Purpose

This document records the completed development foundation that now precedes the successor D2
and D3 work. It answers three distinct questions without treating them as one result:

1. does stale K/V reuse leave an observable current-model consistency gap on an embedding-sharded
   paper-scale model under ordinary stream updates;
2. can a D1 compiled repair close a useful part of that gap at low logical Exact replay; historical
   resident measurements are retained only as implementation-feasibility diagnostics;
3. after D1 reduces semantic work, does a naive mixed execution already realize the corresponding
   physical benefit, or is D2 still required?

All results in this document are development evidence. They freeze a useful benchmark and expose
the next mechanism boundary; they are not formal paper repeats.

## 2. Evidence tracks

The repository keeps two complementary D1 tracks.

| Track | Scope | Role |
|---|---|---|
| Cross-dataset semantic evidence | KuaiRand, QB, and QK; three model tiers; four training seeds | Primary D1 generality, logical-work/quality, and negative-boundary evidence |
| XP system bridge | QK, 24L/H1536, E4096 row-sharded embedding, two A40s | Connects reuse/exact quality, the active fitted-residual D1 operator, and the later D2/D3 system workload |

The XP bridge contains two controls. The analytic direct-old-K/V projection preserves the original
D1→D2 causal diagnostic. The successor fits a shared, label-free `fresh - cheap` residual on a
small disjoint role and folds it into the same direct-old-K/V affine. The latter is the active D1
mechanism on this large sharded stack; neither control replaces cross-dataset replication.

## 3. Selected XP stream chain

### 3.1 Model and data

- dataset: Tenrec QK, within-user ordinal exposure order;
- model: 24 HSTU layers, hidden size 1,536, 24 heads, head dimension 64;
- embedding: 2,859,836 physical rows × 4,096 dimensions in global FP32, 43.638 GiB;
- projection: owner-side E4096→H1536;
- execution: two row-sharded ranks on GPU0/GPU1;
- training role: 16,384 users;
- qualification role: 4,096 disjoint users;
- update width: eight exposures;
- optimizer continuity: maintained across all updates in one round;
- epochs: one per update;
- preferred development learning rates: dense/projection `1.5e-5`, embedding `1.5e-4`;
- immutable rollback-anchor learning rates: dense/projection `1e-5`, embedding `1e-4`;
- checkpoint admission: finite state plus a nonzero optimizer update only; ranking quality is never
  an admission oracle.

`theta0→theta1` is an explicit warm-up from the cooccurrence-expanded bootstrap into the next-item
streaming objective. It is trained and retained but never counted as D1 evidence. The three
ordinary evidence edges are:

| Edge | Training window for target | Same-history cache comparison | Future quality window |
|---|---|---|---|
| `theta1→theta2` | `[72,80)` | prefix through 80 | `[80,88)` |
| `theta2→theta3` | `[80,88)` | prefix through 88 | `[88,96)` |
| `theta3→theta4` | `[88,96)` | prefix through 96 | `[96,104)` |

Prediction is item `t+1` from hidden state `t`. Each edge uses the same 4,096 qualification users,
the same frozen positive/999-negative candidate binding, and the same common cache endpoint:
FP32 computation where required, FP16 cache storage, and FP32 cache consumption. Frozen, Reuse,
and Exact are evaluated on the same future window. Exact is the cache-fidelity reference, not a
guaranteed upper bound on every label-ranking metric.

### 3.2 Development selection

Three nested or learning-rate-matched candidates were evaluated. Selection used the minimum
paired Exact-over-Reuse sampled-cross-entropy gap across all three ordinary edges, subject to all
three gaps being positive and at least two stream updates having positive CE utility.

| Candidate | Train users | LR scale | Exact-over-Reuse CE gaps by edge | Disposition |
|---|---:|---:|---|---|
| `lr010_8192` | 8,192 | 0.10 | 0.00139 / 0.02303 / 0.00258 | compact results retained; checkpoint deleted |
| `lr025_8192` | 8,192 | 0.25 | 0.03951 / 0.04626 / 0.00811 | compact results retained; checkpoint deleted |
| `lr010_16384` | 16,384 | 0.10 | 0.01846 / 0.01068 / 0.01340 | compact rollback evidence retained; checkpoints retired after LR0.15 selection |
| `lr015_16384` | 16,384 | 0.15 | 0.03082 / 0.01450 / 0.01095 | preferred development candidate; checkpoints retained |
| `lr020_16384` | 16,384 | 0.20 | 0.03394 / 0.01358 / 0.00765 | rejected after compact archival; checkpoints deleted |

The preferred LR0.15 gaps have record-cluster 95% intervals `[0.02592,0.03578]`,
`[0.01245,0.01666]`, and `[0.00878,0.01303]`. Its mean gap is about 32% larger than the rollback
anchor while all three intervals remain positive. LR0.20 expands the first edge but weakens the
last edge, so it is not retained as a checkpoint candidate. These are development-selected
values, not untouched test evidence; formal quality evidence still requires a predeclared
configuration and training-seed replication.

## 4. D1 bridge results

### 4.1 Analytic causal control

The v1 bridge compares four methods on the exact selected records and candidates:

- `all_reuse`: source-version FP16 prefix K/V consumed by the target model;
- `compiled_direct_oldkv`: one shared analytic affine over source-version FP16 K/V;
- `mixed_fixed20`: the compiled path plus a label-free retained-token budget of approximately 20%
  Exact records;
- `all_exact`: target-version replay with the same FP16 destination endpoint.

The mixed selector uses extent strata and a stable hash. It reads neither recommendation labels nor
per-record realized errors. In this fixed-width quality window, record and retained-token fractions
are both approximately 20%; this equality must not be generalized to the natural-length HET
system workload.

| Edge | Reuse CE | Compiled CE | Mixed CE | Exact CE | Compiled gap closed | Mixed gap closed | Compiled / Exact maintenance | Naive mixed component bound / Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `theta1→theta2` | 5.88698 | 5.87518 | 5.87426 | 5.86852 | 63.9% | 68.9% | 0.162× | 0.764× |
| `theta2→theta3` | 5.89982 | 5.89392 | 5.89317 | 5.88914 | 55.3% | 62.3% | 0.152× | 0.781× |
| `theta3→theta4` | 5.82937 | 5.81999 | 5.81941 | 5.81597 | 70.0% | 74.3% | 0.146× | 0.731× |

The compiled operator therefore gives a stable interior quality point at 14.6%–16.2% of the
measured Exact maintenance component. However, the naive mixed implementation executes Exact and
compiled work in many of the same batches, so its component bound remains 73.1%–78.1% of Exact
despite only about 20% logical Exact work. The mixed number is not an end-to-end runtime: it sums
selected-route batch components and excludes a physical D2 lowering.

This is the desired D1→D2 causal chain:

```text
D1 creates semantic sparsity
    ↓
naive mixed batching fails to preserve most of it physically
    ↓
D2 must separate compatible compiled work, merge exact pools,
owner-compute retained repair, and issue only unavoidable lookup communication
```

It supports a D2 problem statement. It does not yet prove a D2 speedup.

### 4.2 Label-free residual compiler

The LR0.15 successor keeps the same model, records, candidates, endpoint, source K/V representation,
and `24×3072×3072` online direct-old-K/V program shape. It changes only compilation: on each edge,
128 disjoint theta12 records contribute at most 8,192 global tokens per layer to a rank-16 ridge
fit of target exact K/V minus the cheap current projection. The fit reads no recommendation label,
per-record qualification error, or qualification metric. The resulting 1,880,064-parameter
low-rank adapter is folded into a 453,138,469-byte FP16 affine, so neither adapter rank nor fitting
work enters online maintenance.

| Edge | Reuse CE | Analytic CE | Rank-16 CE | Exact CE | Analytic gap closed | Rank-16 gap closed | Rank-16 / Exact maintenance | Rank-16 K/V relative error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `theta1→theta2` | 5.88548 | 5.86320 | 5.85467 | 5.85467 | 72.3% | 99.995% | 0.163× | 0.00413 |
| `theta2→theta3` | 5.88042 | 5.86984 | 5.86593 | 5.86593 | 73.0% | 99.992% | 0.152× | 0.00536 |
| `theta3→theta4` | 5.80336 | 5.79476 | 5.79240 | 5.79241 | 78.6% | 100.024% | 0.148× | 0.00520 |

A simultaneously executed rank-64 control reduces K/V relative error further but does not produce
a material task-quality improvement and uses 7,188,480 adapter parameters before folding. Rank-16
is therefore the preferred development candidate. Recovery slightly above 100% on the final edge
is reported as a paired method-minus-Exact difference, not as a claim that migrated K/V is more
exact than exact recomputation.

This result closes the one-edge compiler search: the same-scale stream chain has a larger, stable
Reuse–Exact opportunity and the active D1 mechanism recovers essentially all of it at the same
14.8%–16.3% online maintenance fraction. It does not close recursive D1. Every row above starts
from exact source-version K/V; it never feeds `theta1→theta2` output into `theta2→theta3` or that
output into `theta3→theta4`. The rank-16 program is therefore the incumbent for the recursive
comparison, not an already validated recursive design. The near-Exact XP result cannot silently
erase cross-dataset exact/fallback cases or be reused as formal D2 evidence.

## 5. Artifacts and retention

| Asset | Path | Retention |
|---|---|---|
| Rollback training chain | `results/baseline_rounds/quality_chain/quality_chain_stream_aligned_train16384_round1/` | retained |
| Rollback checkpoints | `checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/quality_chain_stream_aligned_train16384_round1/` | retired; recoverable only by retraining |
| Three-way candidate comparison | `results/baseline_rounds/quality_chain/quality_chain_candidate_comparison_3way_development.json` | retained |
| Rollback analytic bridge results/programs/plans | `results/baseline_rounds/quality_chain/selected_d1_bridge_round1/` | retained, about 1.3 GiB |
| Rollback analytic bridge summary | `results/baseline_rounds/quality_chain/selected_d1_bridge_round1/summary.json` | retained |
| Accepted development anchor | `results/baseline_rounds/quality_chain/anchors/quality_anchor_20260802_v1.json` | immutable rollback record with summary, program, plan, and checkpoint-manifest hashes |
| LR0.15 training/baseline chain | `results/baseline_rounds/quality_chain/quality_lr_dual_20260802_round1_lr015/` | preferred development quality opportunity |
| LR0.15 checkpoints | `checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/quality_lr_dual_20260802_round1_lr015/` | retained, about 179 GiB |
| Selected QK/QB machine registry | `configs/evokv_foundation/selected_checkpoint_registry_development_v0.json` | authoritative local payload paths, hashes, resume points, and cleanup ledger |
| Selected QB chain | split `u15_e1/theta_0` plus `u30_e3/theta_{1,2,3}` under `checkpoints/evokv_qb_large_mf9_e4096/qb_large_round1/` | retained secondary chain; no theta4 |
| LR0.15 analytic control | `results/baseline_rounds/quality_chain/quality_lr015_analytic_d1_round1/` | retained |
| Rank-16 fitted D1 candidate | `results/baseline_rounds/quality_chain/quality_lr015_d1_residual_20260802_round1_rank16/` | preferred development programs/plans/results |
| Rank-64 fitted D1 control | `results/baseline_rounds/quality_chain/quality_lr015_d1_residual_20260802_round1_rank64/` | retained through the next protocol review |
| Residual comparison | `results/baseline_rounds/quality_chain/explorations/quality_lr015_d1_residual_20260802_round1/comparison.json` | retained |
| Preferred fitted-D1 archive | `results/baseline_rounds/quality_chain/anchors/quality_lr015_d1_residual_20260802_round1.json` | immutable hashes and scope boundary |
| Checkpoint-retirement tombstone | `results/baseline_rounds/quality_chain/anchors/checkpoint_retirement_20260802_v1.json` | records nine deleted checkpoints, hashes, reasons, and retained roots |
| Rejected candidate metrics | their original result roots under `results/baseline_rounds/quality_chain/` | retained |
| Rejected and superseded checkpoints | two 8,192-user candidates, LR0.20, LR0.10 rollback payloads, `baseline_round3`, and historical `seed0/theta_1,2` | deleted, recoverable only by retraining |
| Full K/V payloads | none | never retained |

Primary entry points are:

- `scripts/verify_evokv_selected_checkpoints.py` before consuming either selected chain;
- `scripts/run_evokv_selected_checkpoint_rebuild.sh` for a new bound QK or QB chain;
- `scripts/run_evokv_quality_chain_existing_config.sh` for a fully bound training/baseline chain;
- `scripts/compare_evokv_quality_chain_candidates.py` for development candidate comparison;
- `scripts/run_evokv_selected_d1_bridge_round.sh` for the selected three-edge D1 bridge;
- `scripts/run_evokv_d1_candidate_bridge.sh` for bound analytic or residual-fit D1 candidates;
- `scripts/run_evokv_d1_dual_residual_exploration.sh` for the rank-16/rank-64 two-pair round;
- `scripts/summarize_evokv_selected_d1_bridge.py` for endpoint-parity validation and compact output.

The candidate runner binds exactly two ranks. Current availability permits one job on GPU0/GPU1;
GPU2/GPU3 must not be scheduled until the user explicitly makes them available. It validates that
Reuse and Exact reproduce the selected baseline before computing any recovery ratio.

## 6. Paper mapping

| Evidence | Claim it may support | Claim it may not support |
|---|---|---|
| Selected Reuse/Exact gaps | A large row-sharded stream chain has a measurable stale-cache consistency opportunity | Exact is always better on NDCG/Hit; the selected development chain is a formal replicate |
| Compiled D1 bridge | A cheap shared repair can recover a useful part of the same paired CE gap | The analytic XP bridge replaces cross-dataset fitted-residual D1 evidence |
| Rank-16 residual compiler | A small disjoint, label-free cohort sample can compile a near-Exact shared repair into the same online affine on this XP chain | Rank-16 is universally optimal; development selection is a formal replicate; Exact/fallback routes disappear from every workload |
| Naive mixed component bound | Reducing logical Exact work does not automatically produce proportional physical savings | Communication dominates; complete D2 is already faster |
| Immutable plans/programs | D1 can export bound work for a later D2 runner | The current quality-role plan is the final natural-length HET ActionPlan |

## 7. Completed QK recursive round and next boundary

QK Round A completed on GPU0/GPU1 under
`evokv_qk_recursive_d1_round_a_development_v0`. It executed the incumbent and rollout-aware
programs through all three ordinary edges with true FP16 recursive handoff and no hidden Exact
reset. The compact result root is
`results/baseline_rounds/quality_chain/recursive_d1_round_a/qk_recursive_d1_round_a_20260804_round1/`.

The current system-design point is `ract_kv_exact10`:

| Edge | Exact valid-token fraction | CE-gap recovery | Mean K/V recovery | Cumulative CE recovery |
|---|---:|---:|---:|---:|
| `theta1→theta2` | 10.010% | 99.986% | 97.246% | 99.986% |
| `theta2→theta3` | 10.010% | 99.985% | 94.933% | 99.997% |
| `theta3→theta4` | 10.010% | 100.002% | 94.932% | 100.000% |

The true-recursive one-edge incumbent loses K/V fidelity on the later edges, recovering only
30.95% and 53.49%; deployment-matched RACT-KV with no scheduled Exact recovers 94.33% and 94.26%.
This isolates the benefit of fitting on the method-produced rollout distribution. The 10% renewal
point retains that behavior while defining a finite, label-free Exact schedule.

The frozen v0 summary is `complete_no_admitted_policy` because its old selector treated a
conservative held-out triangle-inequality rollout bound as a per-edge target gate. That quantity
does not prove recommendation accuracy and is now a diagnostic in the system design. The result
file and old selector outcome remain immutable; the design interpretation is not a retroactive
protocol pass.

The next result-dependent boundary is a newly frozen QB confirmation protocol. It must keep the QK
rank, ridge, fit size, sampled-token limit, direct-old-K/V endpoint, and 10% renewal policy fixed,
execute `theta1→theta2→theta3` without QB-specific tuning, and select on recursive K/V/task
quality, logical Exact work, and lineage. The old `stability_certificate` field may remain for
artifact compatibility but is diagnostic rather than a theory or selection gate. Operational
fallback is reported under correctness and cannot be hidden in work accounting.

Only after QB confirmation should a physical D2 round hold the selected recursive
checkpoint/program/plan sequence fixed and compare all-Exact, naive mixed, owner-local compiled
repair, shape-aware/segmented lowering, and complete D2. The older 73%–78% naive component bounds
remain valid diagnostics for their frozen analytic stack but cannot substitute for the new
recursive stack's baselines.
