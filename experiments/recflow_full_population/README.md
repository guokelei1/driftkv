# RecFlow full-population version chain

This directory implements the approved 30,000-user, ten-model experiment. Existing
`recflow_request_ranking` checkpoints, populations and results remain historical
artifacts. The new chain starts from seed 17 with a new foundation-only vocabulary.

## Fixed protocol

- Transformer Max: 16 layers, width 320, 10 heads, 1,154,235,993 parameters.
- History is the most recent **up to 1,024** realshow events strictly before the
  entire request timestamp block. Empty and short histories are included.
- Fixed 30,000 training UIDs selected using foundation-period information only:
  20,000 with complete-stage feedback in D24–27, plus 10,000 with earlier feedback.
  Validation 2,000, cache-fit 384 and cache-validation 128 UIDs are disjoint.
- Vocabulary and popularity use only selected training users through D27. The
  3.5-million explicit ID table and 65,536 fallback buckets stay fixed thereafter.
- D1 is January 13, 2024; all daily boundaries use Asia/Shanghai and half-open
  timestamp intervals. There are 37 exposure days and 14 complete-candidate days.

| Model | Training | Next-day evaluation |
|---|---|---|
| M0 | D1–23 exposure BCE, then D24–27 full-candidate alignment | D28, Feb 9 |
| M1 | D28, Feb 9 | D29, Feb 10 |
| M2 | D29, Feb 10 | D30, Feb 11 |
| M3 | D30, Feb 11 | D31, Feb 12 |
| M4 | D31, Feb 12 | D32, Feb 13 |
| M5 | D32, Feb 13 | D33, Feb 14 |
| M6 | D33, Feb 14 | D34, Feb 15 |
| M7 | D34, Feb 15 | D35, Feb 16 |
| M8 | D35, Feb 16 | D36, Feb 17 |
| M9 | D36, Feb 17 | D37, Feb 18 |

M0 has no previous released model, so its evaluation is native only. Each of the
nine subsequent edges compares current Rebuild, current reading its actual
parent's KV (Reuse), and parent native. Every method sees the same complete logged
candidate set, request and causal history. This is a matched-prefix snapshot
experiment; it does not simulate a persistent serving cache or a FIFO policy.

The main evaluation follows the **same fixed training UID list** on future
requests. Users inactive on a day contribute no requests; the list is never
intersected with future activity. Past evaluation days may train a later model,
as in a prequential protocol, and cannot change earlier models or their scores.

## Training and learning checks

All four A40 GPUs use replicated parameters, BF16 training, global batch 64
(16 per GPU), dense AdamW moments and exact averaging of touched item gradients.
Each rank has four mmap CPU workers. The fixed 1,024 padding width is retained;
the ETA does not assume a dynamic-length speedup.

Warmup includes all 4,123,554 requests once. Alignment includes all 647,394
requests per epoch, at most two epochs. The nine one-epoch updates contain
1,440,220 requests in total. AdamW moments continue from the actual parent;
daily learning-rate schedules restart explicitly.

Before training, score an identical seed-17 untrained Max control on a fixed
16,384-request sample from the 2,000 held-out validation UIDs, D24–27. After
4,096 warmup steps, require a positive paired UID-bootstrap lower bound for
the validation BCE improvement. If this gate fails, retain everything and stop.

Complete warmup, then evaluate each complete alignment epoch. Select the epoch
with the highest native validation NDCG@10 (earlier epoch on a tie). Require its
NDCG@10 to beat both expected random ranking and foundation-only popularity,
with positive paired UID-bootstrap lower bounds. This is a development learning
check, not an independent quality confirmation. No successor is launched before
it passes. The BCE gate is early only: the later listwise objective need not
improve BCE monotonically.

The foundation budget is 15 hours including initialization, validation, audits
and checkpoints. The second alignment epoch is admitted only when its measured
projection fits the remaining budget. Incomplete epochs are retained but are
never promoted. Each update must complete its full epoch within two hours.
Negative Rebuild−Reuse gaps and lower quality than the parent do not stop the
chain or change the recipe. All nine edges are evaluated after learning passes.

## Metrics, evidence and numerical limits

Primary: request-macro observed-positive NDCG@10. Secondary: Recall@10; @5 and
@20 are retained. Unexposed candidates remain in the complete candidate set and
are not treated as observed BCE negatives. Zero-positive requests are saved and
counted but excluded from the undefined NDCG/Recall mean. Intervals use paired
UID-cluster bootstrap (2,000 replicates, seed 17), with nominal 95% coverage;
they are not simultaneous multiple-edge confidence statements.

Evaluation uses FP32 parameters, scores and KV, common reader/backends, with
PyTorch TF32 flags disabled. The existing Triton prefix kernel uses TF32x3 for
near-FP32 dot products. Its atomic backward reductions do not guarantee bitwise
identical training trajectories across separate executions. The failed
cross-run bitwise test is preserved under `runs/readiness_attempt_1`; differences
already occurred before interruption. Exact saved parameter/moment/RNG/cursor
restoration is checked separately, without changing the training arithmetic or
loosening the restore equality check.

Every evaluation saves all raw candidate IDs, labels and method scores. A CPU
audit checks exact membership and independently recomputes the ranking metrics.
The run stores source/config snapshots and hashes, stage commands, failures,
learning gates, all checkpoints and all positive/negative results.

## Running and inspecting

The coordinator is `code/chain.py`. It refuses duplicate output directories,
holds a single-instance lock, verifies frozen sources before each stage and
stops on numerical, storage, learning or subprocess failure. There are no hidden
retries and no automatic changes to data, thresholds or hyperparameters.

Read `STATUS.md` for the live tmux session and run path. Within the run:

- `active_process.json`, `progress.json`, `stages/`, `logs/`: actual process/stage.
- `gates/`: early learning, second-epoch budget and foundation selection.
- `model_chain_inventory.json`, `version_metrics.csv`: accumulating results.
- `failure.json`: stopped with artifacts retained; no successor continues.
- `completion.json`: written only after all ten models and ten windows pass audit.

Original estimate: 11–14 hours for foundation, capped at 15; 22–29.5 hours for
the full chain including future-window scoring and audits. These are estimates,
to be refined with actual throughput, not a promise of training convergence.
