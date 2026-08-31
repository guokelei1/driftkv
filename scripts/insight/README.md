# Insight analysis scripts

Small analysis-only experiments between the sealed motivation and the future
EvoKV design live here. They reuse existing raw artifacts and do not train
models or mutate sealed evidence.

```bash
PYTHONPATH=src python scripts/insight/analyze_first_pass.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_history_utility.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_kv_mechanism.py
PYTHONPATH=src python scripts/insight/analyze_recommendation_semantics.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_controlled_dilution.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/analyze_embedding_origin.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_embedding_hybrid.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_anchor_replay.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_refinement_algebra.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_recommendation_state_structure.py
PYTHONPATH=src python scripts/insight/adjudicate_recommendation_state_structure.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_candidate_shared_causal.py
PYTHONPATH=src:scripts python scripts/insight/run_candidate_shared_exposed.py
PYTHONPATH=src python scripts/insight/adjudicate_candidate_shared_causal.py
PYTHONPATH=src:scripts python scripts/insight/run_evidence_measure_basis.py --canary-only
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_reader_compatibility_correction.py --scope canary
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_reader_compatibility_correction.py --scope formal-controlled
PYTHONPATH=src:scripts python scripts/insight/run_reader_correction_persistence.py --scope canary
PYTHONPATH=src:scripts python scripts/insight/run_reader_correction_persistence.py --scope formal
PYTHONPATH=src python scripts/insight/adjudicate_reader_compatibility_correction.py
PYTHONPATH=src:scripts python scripts/insight/run_av_broadcast_residual_canary.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_pro_lazy_reader.py \
  --contract configs/contracts/yambda500m_small_hstu_native_pro_lazy_reader_v2.yaml \
  --output results/yambda500m_small_seed17/insight_pro_lazy_reader_v1/correctness_cost_v2
PYTHONPATH=src python scripts/insight/adjudicate_pro_lazy_reader.py \
  --contract configs/contracts/yambda500m_small_hstu_native_pro_lazy_reader_v2.yaml \
  --result results/yambda500m_small_seed17/insight_pro_lazy_reader_v1/correctness_cost_v2
PYTHONPATH=src:scripts python scripts/insight/run_pro_lazy_rolling_quality.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_progressive_pro_decomposition.py --scope canary
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_progressive_pro_decomposition.py --scope formal
PYTHONPATH=src python scripts/insight/adjudicate_progressive_pro_decomposition.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_progressive_pro_frontier.py --scope canary
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/insight/probe_progressive_pro_frontier.py --scope formal
PYTHONPATH=src python scripts/insight/adjudicate_progressive_pro_frontier.py
PYTHONPATH=src:scripts python scripts/insight/run_one_release_auc.py
PYTHONPATH=src:scripts python scripts/insight/summarize_one_release_quality_compute.py
```

The scripts write only compact request features, aggregated tables, and short
reports under `results/yambda500m_small_seed17/insight_*`. The current algebra
probe is `insight_refinement_algebra_v1`. The historical
`insight_state_primitive_discovery_v6` result is retained as evidence but no
longer defines the design.

The recommendation-state structure probe is frozen by
`configs/contracts/yambda500m_small_hstu_native_recommendation_state_structure_v1.yaml`.
Its formal run uses exactly 3,000 fixed users (30% of Small), all five v0..v5
adjacent edges, 512 pre-cutover events and 64 label-free candidate probes per
user-edge. `--max-users` exists only for a focused correctness canary; it is not
a paper-scale substitute. The adjudicator consumes only the compact formal
output and refuses to overwrite an existing adjudication.

The signed causal follow-up is frozen by
`yambda500m_small_hstu_native_candidate_shared_causal_v1.yaml`. It uses signed
per-head contributions without candidate normalization, then validates all five
edges on real same-UID/same-timestamp exposed request banks through raw-first
seals. `adjudicate_candidate_shared_causal.py` combines both scopes and keeps
the oracle/action boundary explicit.

The only mechanism follow-up is frozen by
`yambda500m_small_hstu_native_evidence_measure_basis_v1.yaml`. The runner first
executes a five-edge, label-free, matched-cost canary. Its frozen 4/5 gate failed
0/5, so formal rolling quality is intentionally locked; rerunning it must not be
used to tune the pair rule or budget.

The next observation is frozen by
`yambda500m_small_hstu_native_reader_compatibility_correction_v1.yaml`. It treats
the proved object as a query-dependent reader correction, not a history basis,
and locates its earliest HSTU boundary before testing adjacent real-request
direction and recovery. The four-GPU runner is inference-only and raw-first.
Its mechanism path is locked until the final adjudicator passes both gates.
Both gates passed. The final command runs the only unlocked executable
candidate: a disposable 32-carrier Current probe generates an AV sidecar that
is coverage-scaled and broadcast across candidates. Its label-free score
canary passed 4/5; the contract still prohibits formal quality and admission.

The subsequent PRO audit removes per-position translated-prefix materialization
from the action. The joint version map is pushed into one fixed reader probe;
recent 16/32 Current carriers are replayed against the unmodified Parent prefix.
The v1 absolute-AV gate failed and is retained. The unchanged mechanism passed
the scale-aware v2 audit on the next 32 users: the primary 32-carrier cost is
9.1% of Full and materializes zero translated-prefix positions. The following
runner freezes that same mechanism, performs a five-edge label-free canary, then
seals every full-population E14 raw before label join. The formal result is AUC
5/5 and log-loss 3/5 versus Reuse, with both mean edge deltas favorable. This
supports overall design viability but fails the predeclared strict 4/5+4/5 gate;
it does not admit a serving lineage or runtime/new-seed qualification.

The progressive follow-up first decomposes C32 error into direction, amplitude,
probe disagreement and rolling decay without reading labels. Two probes are
consistent on 5/5 edges, while the absolute direction, amplitude-dominant and
segment-decay gates fail. A held-out C32/C48/C64 frontier then shows C64 improves
C32 relative L2 on 5/5 cutover and 5/5 rolling edges, but fails the absolute
rolling-direction gate and is non-monotonic through C48. The prefrozen rule
therefore retains the original C32 lightweight PRO and does not unlock quality.

The last command is the prospective full-population D14/E14 qualification for
the fixed one-hop `CAST384 + GROUP/PATCH 128->64 + SCALE2` path. It uses GPUs
0/1/2/3, replays all five edges serially, seals raw output before label join,
and writes the summary under
`results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/d14_one_release_refinement_auc_v1/`.
The final command is arithmetic-only: it combines the sealed AUC summary with
the fixed 4L/H128/context512 plan's conservative causal-FLOP count. It does not
benchmark or claim GPU runtime.
