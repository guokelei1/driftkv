# Medium Insight 1 locality experiment

This directory contains the isolated implementation for the frozen layer,
sparse-token and continuous-window Exact-KV splice diagnostic.  The splices are
observation-only interventions and must not enter an executable migration
frontier.

The pipeline has four stages: prepare the fixed label-free population and
candidate panels, run a four-GPU all-configuration canary, benchmark safe batch
and candidate-chunk settings, then run and adjudicate the formal 3,000-user
observation.  Every GPU rank owns a disjoint UID shard and evaluates all five
edges; checkpoints and edges remain serial within a rank.

```bash
PYTHONPATH=src:scripts python scripts/insight_one_locality/prepare_inputs.py

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:scripts torchrun --standalone --nproc_per_node=4 \
  scripts/insight_one_locality/run_distributed.py --scope canary \
  --batch-size 2 --candidate-chunk 8

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:scripts torchrun --standalone --nproc_per_node=4 \
  scripts/insight_one_locality/run_distributed.py --scope benchmark \
  --output results/yambda500m_medium_seed17/insight1_locality_v1/bench_b8_c16 \
  --max-users 64 --edge-indices 0 --batch-size 8 --candidate-chunk 16

PYTHONPATH=src:scripts python scripts/insight_one_locality/estimate_runtime.py \
  results/yambda500m_medium_seed17/insight1_locality_v1/bench_*

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:scripts torchrun --standalone --nproc_per_node=4 \
  scripts/insight_one_locality/run_distributed.py --scope formal \
  --batch-size RECOMMENDED_BATCH --candidate-chunk RECOMMENDED_CHUNK

# The sealed formal launch is normally run from a detached tmux session:
tmux new-session -d -s evokv_insight1_formal -c /home/gkl/work/evokv
tmux send-keys -t evokv_insight1_formal:0.0 \
  'bash scripts/insight_one_locality/run_formal_tmux.sh' Enter

PYTHONPATH=src:scripts python scripts/insight_one_locality/adjudicate.py
```

All stages refuse to overwrite existing evidence.  A failed run leaves its
`.partial` directory for audit.
