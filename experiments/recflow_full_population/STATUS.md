# Full-population training status

Started **2026-10-28 10:04 Asia/Shanghai**. The user explicitly authorized
checking the full pipeline, launching ten models, gating continuation on actual
learning, preserving all positive/negative results, and handing back control
without waiting for the long run.

Handoff check at approximately **10:07**: the untrained control and its
independent audit completed (16,384 requests, 6,471,654 candidates; maximum
metric recomputation error 1.39e-17). Actual foundation training is running,
torchrun PID **1120924**, already at step **200 / 4,096** of the early segment
(12,800 requests), about **0.310 seconds/global batch** after startup. This is
training progress; the held-out learning gate remains pending. There is no
`failure.json` at handoff. The coordinator continues autonomously in tmux.

- tmux: **`recflow-full30k`**, coordinator PID **1119724**.
- Run: `/home/gkl/work/evokv/experiments/recflow_full_population/runs/attempt_1`.
- Coordinator log: `reports/chain_attempt_1.log`.
- Actual live stage/PID: `runs/attempt_1/active_process.json`.
- Four physical A40 GPUs: **0, 1, 2, 3**.
- Data, configuration, numerical source and coordinator are now frozen. Do not
  modify active code, restart this coordinator, start duplicates, or resume old
  chapter-3/RecFlow queues. Source/config snapshots are in the run.

The initial stage scores the fixed untrained control, audits all 16,384 requests,
then starts the 4,096-step early foundation segment. A held-out BCE learning
check gates completion of warmup. The selected complete foundation alignment
epoch must beat random and foundation-only popularity in held-out NDCG@10
before M1–M9 are dispatched. **Learning has not yet been established at launch.**
No update requires a positive Rebuild−Reuse gap or improvement over its parent.

Readiness: 185 input files verified; all 22 request lists checked for causal
history and user roles; 18 CPU tests passed. Actual 1.154B Max on all four GPUs
passed zero/short/full-history reference checks, exact saved-state restoration,
25 continuing optimizer steps, and FP32 same-model Rebuild/Reuse identity.
All 1,154,235,993 parameter elements and 2,308,470,883 optimizer-state elements
were checked per card. Complete-candidate training p95 was approximately
0.662 seconds/global batch 64, peak allocated 21.84 GiB/card. These are short
diagnostics, not convergence evidence.

The initial test expecting bitwise identical *separate training runs* failed;
it and its checkpoints remain in `runs/readiness_attempt_1`. Differences already
occurred before interruption, consistent with the existing atomic backward
kernel. The exact *saved-state* restore test passed without relaxing equality
or changing training arithmetic. See `reports/cross_run_bitwise_diagnostic.json`
and `reports/launch_readiness.json` for the evidence and limits.

Expected foundation: roughly 11–14 hours, hard limit 15 hours including its
validation/checkpoints. Expected complete chain: **22–29.5 hours from launch**,
approximately **2026-10-29 08:05–15:35 Asia/Shanghai**, subject to actual
throughput and learning gates. The early learning check is expected in roughly
30–45 minutes. A failed gate stops the process with `failure.json`; no automatic
hyperparameter change or silent restart follows.

Progress commands:

```bash
tmux attach -t recflow-full30k
cat /home/gkl/work/evokv/experiments/recflow_full_population/runs/attempt_1/active_process.json
tail -n 30 /home/gkl/work/evokv/experiments/recflow_full_population/reports/chain_attempt_1.log
```

Accumulating outputs: `model_chain_inventory.json`, `version_metrics.csv`, all
checkpoint sidecars, per-window raw score arrays and independent candidate/metric
audits. `completion.json` requires all ten models and all ten future windows.
The full method is documented in [README.md](README.md). The paper is unchanged.
