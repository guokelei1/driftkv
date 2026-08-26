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
PYTHONPATH=src:scripts python scripts/insight/run_one_release_auc.py
```

The scripts write only compact request features, aggregated tables, and short
reports under `results/yambda500m_small_seed17/insight_*`. The current algebra
probe is `insight_refinement_algebra_v1`. The historical
`insight_state_primitive_discovery_v6` result is retained as evidence but no
longer defines the design.

The last command is the prospective full-population D14/E14 qualification for
the fixed one-hop `CAST384 + GROUP/PATCH 128->64 + SCALE2` path. It uses GPUs
0/1/2/3, replays all five edges serially, seals raw output before label join,
and writes the summary under
`results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/d14_one_release_refinement_auc_v1/`.
