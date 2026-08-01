# D1/D2 Baseline and Selected XP Quality Foundation

## 1. Purpose

This document records the completed development foundation that now precedes the successor D2
and D3 work. It answers three distinct questions without treating them as one result:

1. does stale K/V reuse leave an observable current-model consistency gap on an embedding-sharded
   paper-scale model under ordinary stream updates;
2. can a D1 compiled repair close a useful part of that gap with much less measured maintenance
   work than exact replay;
3. after D1 reduces semantic work, does a naive mixed execution already realize the corresponding
   physical benefit, or is D2 still required?

All results in this document are development evidence. They freeze a useful benchmark and expose
the next mechanism boundary; they are not formal paper repeats.

## 2. Evidence tracks

The repository keeps two complementary D1 tracks.

| Track | Scope | Role |
|---|---|---|
| Cross-dataset semantic evidence | KuaiRand, QB, and QK; three model tiers; four training seeds | Primary D1 generality, cost/fidelity, and negative-boundary evidence |
| XP system bridge | QK, 24L/H1536, E4096 row-sharded embedding, two A40s | Connects reuse/exact quality, the D1 action domain, and the later D2/D3 system workload |

The XP bridge does not replace the fitted-residual D1 headline. Its direct-old-K/V program is an
analytic projection-only bridge chosen because it can be compiled from a source/target checkpoint
pair and executed on the same large sharded stack used by D2 and D3.

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
- learning rates: dense/projection `1e-5`, embedding `1e-4`;
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
| `lr010_16384` | 16,384 | 0.10 | 0.01846 / 0.01068 / 0.01340 | selected; all four checkpoints retained |

The selected gaps have record-cluster 95% intervals `[0.01520,0.02167]`,
`[0.00917,0.01223]`, and `[0.01073,0.01603]`. This development selection fixes a useful mechanism
benchmark; it cannot be promoted as untouched test evidence. Formal quality evidence requires a
predeclared configuration and training-seed replication.

## 4. D1 bridge result

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

## 5. Artifacts and retention

| Asset | Path | Retention |
|---|---|---|
| Selected training chain | `results/baseline_rounds/quality_chain/quality_chain_stream_aligned_train16384_round1/` | retained |
| Selected checkpoints | `checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/quality_chain_stream_aligned_train16384_round1/` | retained, about 179 GiB |
| Three-way candidate comparison | `results/baseline_rounds/quality_chain/quality_chain_candidate_comparison_3way_development.json` | retained |
| D1 bridge results/programs/plans | `results/baseline_rounds/quality_chain/selected_d1_bridge_round1/` | retained, about 1.3 GiB |
| D1 bridge summary | `results/baseline_rounds/quality_chain/selected_d1_bridge_round1/summary.json` | retained |
| Rejected candidate metrics | their original result roots under `results/baseline_rounds/quality_chain/` | retained |
| Rejected candidate checkpoints | the two 8,192-user quality checkpoint roots | deleted, not recoverable |
| Full K/V payloads | none | never retained |

Primary entry points are:

- `scripts/run_evokv_quality_chain_existing_config.sh` for a fully bound training/baseline chain;
- `scripts/compare_evokv_quality_chain_candidates.py` for development candidate comparison;
- `scripts/run_evokv_selected_d1_bridge_round.sh` for the selected three-edge D1 bridge;
- `scripts/summarize_evokv_selected_d1_bridge.py` for endpoint-parity validation and compact output.

The D1 bridge runner refuses GPU2/GPU3 and four-rank execution under the current availability
constraint. It validates that its Reuse and Exact endpoints reproduce the selected baseline before
computing any recovery ratio.

## 6. Paper mapping

| Evidence | Claim it may support | Claim it may not support |
|---|---|---|
| Selected Reuse/Exact gaps | A large row-sharded stream chain has a measurable stale-cache consistency opportunity | Exact is always better on NDCG/Hit; the selected development chain is a formal replicate |
| Compiled D1 bridge | A cheap shared repair can recover a useful part of the same paired CE gap | The analytic XP bridge replaces cross-dataset fitted-residual D1 evidence |
| Naive mixed component bound | Reducing logical Exact work does not automatically produce proportional physical savings | Communication dominates; complete D2 is already faster |
| Immutable plans/programs | D1 can export bound work for a later D2 runner | The current quality-role plan is the final natural-length HET ActionPlan |

## 7. Next result-dependent boundary

Training-configuration exploration stops here. The next round should hold the selected
checkpoint/program/plan revision fixed and implement the two-card physical D2 comparison:

1. all-Exact at the same endpoint;
2. naive mixed execution;
3. owner-local compiled retained repair plus merged exact/append pools;
4. shape-aware grouping and segmented destination;
5. complete D2 with the same ActionPlan and full output validation.

The first question is whether the 73%–78% naive mixed component bound falls materially once work is
physically separated. If it does not, diagnose embedding lookup, collective count, destination
movement, or synchronization before changing D1. D3 out-of-core scheduling remains a later layer
and must not be mixed into this resident D2 attribution round.
