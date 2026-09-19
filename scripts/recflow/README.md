# RecFlow development

This isolated track reuses HSTU primitives and owns its data, generation and
retrieval evaluation. The working stages and authorization boundary are in
[the plan](../../docs/recflow/plan.md). Data and model payloads remain local.

- `expanded_daily_panels.py`: fixed4096 initial-history-eligible development
  users, complete A/day-update pools and D19–24 full/6144-request sampled panels.
  Retains the same1M catalog and verifies causal prefixes and population hashes.
- `run_expanded_daily.py`: run tiny A/B with `--canary`; after it passes, use
  `--canary-report` for the authorized fresh A3 + five daily1epoch updates.
  Run the long chain in tmux; it retains progress, logs, exits and real saved
  optimizer checks. Source hashes freeze executable code during the run.
- `expanded_chain_report.py`: read the completed prefix and refresh
  `chain_summary.json`; include base/random and every completed paired result,
  actual test users, complete epochs and checkpoint/optimizer lineage. Quality
  failures remain visible and never automatically admit a serving release.
- `prepare_data.py`: complete realshow preparation and causal request audit.
- `window_panels.py`: active CPU audit and frozen complete fitting pools plus
  balanced three-day evaluation panels, with no training-request cap.
- `daily_window_panels.py`: same512 development users/catalog, complete D19/D20
  fitting pools and complete D20/D21 future panels, with separate768-request
  sampled diagnostics per day. Initial A training array stays byte-identical.
- `window_chain.py`: active single-seed17, complete-epoch chronological development;
  real parent model/optimizer continuation, fixed training-only NLL and future
  retrieval evaluation through the shared evaluator; single-GPU or torchrun DDP.
  Fit/evaluation days come from the frozen settings; reject parents already
  trained on the new update's days. A full model may initialize a labeled tiny
  canary, but canary weights cannot initialize a full run. PhaseD explicitly
  continues C for the predeclared additional daily confirmation.
- `run_daily_probe.py`: fixed daily1/3-epoch development branches from the
  completed six-layer A, including all four paired/random reports. Run once in
  tmux after its actual-checkpoint daily canary; refuses to overwrite evidence.
- `daily_comparison.py`: summarize both daily branches and all four update
  endpoints, checking complete epochs, dates and AdamW/checkpoint lineage.
  Reports whether each setting works on both days; never admits a release.
- `run_daily_lr_probe.py`: complete one-epoch daily branches at update LR1e-4
  and3e-5, compared with retained1e-3. Reuses exact A/day20 evaluation only after
  checkpoint/decoder/panel/raw-metric checks; all other future evaluations run
  normally. Actual loaded optimizer LRs are recorded by `window_chain.py`.
- `lr_comparison.py`: check complete daily LR branches against the retained1e-3
  reference; report all relative improvement percentages and random checks.
- `metric_grid.py`: all three LRs, NDCG20/50/100 and full/uniform1000/popularity1000
  scopes with their own request/random denominators. Exploratory comparison;
  it does not revise the completed free NDCG@50 outcomes.
- `update_parameter_drift.py`: CPU comparison of the three B1 parameter updates
  from A3, with tied parameters counted once and buffers excluded. Weight drift
  is not evidence of cache compatibility.
- `daily_followup_panels.py`: CPU-only D21 fitting and D22 future panels for an
  additional chronological check after freezing a development setting. D22
  appeared in prior three-day development; this is not untouched final data.
- `daily_confirmation.py`: after the D/CD run finishes, check the selected C→D
  lineage, complete4312-request epoch and full9039-request future panel; report
  all three selected edges. Defaults retain the failed full NDCG@100 branch;
  `--root` and `--settings` also select the separate sampled-primary probe.
  It distinguishes the latter's reused-day exploration from independent
  confirmation. Negative gains and random failures remain; no admission.
- `distributed_probe.py`: fixed-global-batch DDP numerical/resource canary and
  reusable complete-epoch training helper. Measures two or four ranks; no
  repeated sampler padding. `--layers/--hidden/--heads` select the resource
  architecture; `--skip-single` avoids a global128 single-GPU memory trial.
- `parallel_evaluate.py`: independent FP32 beam request shards on selected GPUs;
  restores panel order and recomputes request/user/day aggregates from raw rows.
- `window_comparison.py`: compares declared final A/B/C endpoints with matched
  random, checks parent checkpoint lineage and identical future panels, and
  retains every daily direction and hit-request/user count.
- `evaluation_pair.py`: compares two completed standalone evaluations on the
  identical panel, preserving matched random, daily directions and hit counts;
  a full-window sensitivity result does not overwrite an earlier small panel.
  The default primary stays NDCG@50; `--primary-metric ndcg@100` applies the
  separately frozen daily confirmation choice to random and daily checks too.
  `--primary-scope uniform_1000` uses that candidate task's own768-request
  values, random pool and daily metrics; the default remains full_catalog.
- `development_probe.py`: retained historical bounded/resource probes and
  standalone checkpoint re-evaluation. Its capped-request/time-budget training
  example below is not the active complete-window A/B workflow.
- `check_decode.py`: compare beam outputs with exhaustive structured-path ranking
  on a fixed development panel, using an already trained checkpoint.
- `check_history.py`: fixed-target conditional-loss diagnostics with complete,
  shortened and empty histories; no retraining or model admission.
- `popularity_probe.py`: cheap initial/cumulative/recent count controls on the
  same causal windows as the neural pilot.
- `random_baseline.py`: exact uniform-video random expectations and Monte Carlo
  ranking distributions on saved request panels; reports every candidate pool
  and cutoff, including failures.
- `supervision_coverage.py`: reconstruct the pilot's seeded positive-target
  draws and compare direct supervision coverage with the available training
  pool. This is a CPU data diagnostic, not additional fitting.
- `initial_supervision_coverage.py`: compare the512/2048-user initial positive
  pools' coverage of the same future development labels. Reports full-pool
  upper bounds, not actual three-epoch target draws or model-quality evidence.

The prepared source is the complete local `realshow.tar.gz`; all-stage candidates
are not observed history or observed user negatives. The vocabulary is frozen at
D18; reduced pilot vocabularies use initial development exposure counts only.
Generative outputs are `(category1, category2, video_id)` paths, mapped uniquely
back to raw videos. History remains one exposure token per event. The latest
user instruction first requests one-day update checks before data expansion and
motivation replication. The superseded 50–60-minute proposal
does not require a repeated permission question.

## Daily-window workflow

The latest completed screen is `run_daily_lr_probe.py`, retained under
`results/recflow/development/window_6l_daily_lr_seed17`. Its `lr_comparison.py`
and `metric_grid.py` reports preserve all outcomes. The additional confirmation
uses `configs/recflow/window_6l_daily_confirmation_seed17.json`: D continues the
selected LR1e-4 C, fits all D21 once and evaluates C/D on all D22. Its separately
declared primary is free NDCG@100; that confirmation failed. The subsequent
`window_6l_daily_sampled_seed17.json` continues the original LR1e-3 C and uses
uniform1000 NDCG@50. Its three-edge report at
`window_6l_daily_sampled_seed17/daily_comparison/summary.json` passes the
development checks, with all full-catalog failures retained. See the working
plan for scope and limits; do not duplicate completed probes or overwrite settings.

The preceding daily1/3-epoch probe uses the completed six-layer A_epoch3 and
`configs/recflow/window_6l_daily_seed17.json`, followed by
`window_6l_daily_continuation_seed17.json`. Fit D19, evaluate D20 at the
predeclared1/3-epoch endpoints; continue each branch on D20 and evaluate D21.
`run_daily_probe.py` queues the four-GPU phases and paired/random comparisons;
`daily_comparison.py` summarizes all four endpoints after completion. Both
scripts refuse to replace existing evidence. The initialized run is retained
under `results/recflow/development/window_6l_daily_seed17`; do not launch a duplicate.

## Retained three-day complete-window workflow

Use `configs/recflow/window_development_seed17.json`: 2L/H96, context1024, 1M
catalog, the fixed 512 development users, batch128, learning rate 0.001 and three
complete epochs. A fits D1–18 and is evaluated on D19–21; B continues A on D19–21
and compares A/B/random on D22–24; optional C continues B on D22–24 and compares
B/C/random on D25–27. Eligible fitting requests per epoch are 173395/17046/10854.
Each epoch consumes every eligible request once, including the final partial
batch. The existing loss samples one known positive per request; all-OOV fitting
requests are explicitly counted and excluded from that loss.

The CPU panel helper freezes all three windows before training:

```bash
python scripts/recflow/window_panels.py
```

It writes `results/recflow/development/seed17_complete_epoch_windows/summary.json`
and local `window_panels.npz`, refusing to replace existing evidence. The arrays
are `train_1_18`, `update_19_21`, `update_22_24`, and
`eval_19_21`/`sampled_19_21`, `eval_22_24`/`sampled_22_24`,
`eval_25_27`/`sampled_25_27`. Training arrays contain every positive request;
the trainer applies and verifies the audited known-target filter. Free panels
have 1024 requests per day, 3072 per window. Diagnostic subsets have 256 per
day, 768 per window, selected with independent fixed salts inside the free panel.
Uniform/popularity1000 share that subset and candidate seed17. Their metric and
random-baseline denominators must use the 768-request subset, not the 3072 panel.

The primary is full configured-catalog **FP32 beam300 NDCG@50**, with matched
random. This is actual approximate free generation, not exact Top-K. Keep the
other cutoffs, Recall, daily results and OOV coverage. Sampled diagnostics do
not replace the primary or automatically veto it when they fail. Every epoch
records mean training loss and a fixed training-only 2048-request/target NLL;
retrieval is recorded at epochs 1 and 3. Epoch3 is the predeclared candidate;
future-window labels do not choose the best endpoint, and three epochs do not
prove convergence.

The following are **command examples, not additional launches**. Use fresh
output directories and run expected >30-minute execution in detached tmux with a
retained log and exit status. The historical A run completed under
`development/window_2l_seed17/A`; do not start a duplicate. A/B canaries
verified 263/263 requests, the last 7-request batch,
fixed training-probe identities and real optimizer continuation.

```bash
python scripts/recflow/window_chain.py \
  --config configs/recflow/window_development_seed17.json \
  --phase A --device cuda:0 \
  --output results/recflow/development/window_example_seed17/A
```

B loads the completed, predeclared A epoch3 checkpoint, including its optimizer:

```bash
python scripts/recflow/window_chain.py \
  --config configs/recflow/window_development_seed17.json \
  --phase B --device cuda:0 \
  --checkpoint results/recflow/development/window_example_seed17/A/A_epoch3/checkpoint.pt \
  --output results/recflow/development/window_example_seed17/B
```

Optional C similarly uses `--phase C` and B's `B_epoch3/checkpoint.pt`. The runner
refuses to reuse an existing output directory. `--canary` uses a tiny complete
subset and cannot qualify A/B/C. Main epochs record order/target hashes,
complete request counts, model and optimizer state. Phase summaries mark
completion explicitly; logs alone do not establish it.

The user also authorized all four GPUs, including independent experiments.
The measured two-GPU training path keeps global batch128 (64 per rank, with
correctly weighted uneven tails). The original A continues unchanged; subsequent
B/C may use `window_development_seed17_ddp.json`, which preserves its learning
and evaluation protocol and records updated resource estimates. Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=2 \
  scripts/recflow/window_chain.py \
  --config configs/recflow/window_development_seed17_ddp.json \
  --phase B --eval-devices 0 1 2 3 \
  --checkpoint results/recflow/development/window_example_seed17/A/A_epoch3/checkpoint.pt \
  --output results/recflow/development/window_example_seed17/B
```

Training uses logical GPUs0/1; all four independently evaluate request shards.
Device numbers are interpreted inside `CUDA_VISIBLE_DEVICES`. The tiny
`--canary` retains its 8/4 in-memory evaluation panels on rank0. In a normal run,
evaluation worker memory excludes the resident waiting training model/optimizer;
worker peaks are not total physical-device peaks. The numerical tail/optimizer
canary and serial-versus-parallel ranking/aggregation comparison passed before
main use. Request sharding reduces elapsed time, not necessarily GPU-seconds.
Main-process waits use a CPU Gloo group, while training collectives use NCCL.
An NCCL wait on a training GPU blocked that GPU's independent evaluation worker
in the first B attempt, before any B optimizer update. The unchanged B fitting
was restarted in `B_retry` after a real model/optimizer/DDP-resident eight-request
canary verified the Gloo wait. The interrupted `B` attempt remains on disk.

Random analysis reads each standalone evaluation directory, not the outer phase
summary. For example, the matched B-window comparison is:

```bash
python scripts/recflow/random_baseline.py \
  --runs results/recflow/development/window_example_seed17/B/parent_A_d22_24 \
         results/recflow/development/window_example_seed17/B/B_epoch3/evaluation \
  --output results/recflow/development/window_example_seed17/B_random \
  --draws 5000 --seed 20260918
```

For A versus random use `A/A_epoch3/evaluation`. The active evaluator records
the separate sampled-panel file, which random analysis uses for those
diagnostics. Retain primary failures and every companion result; these
random-policy intervals are not training-seed confidence intervals.

## Preparation and historical bounded/resource probes

Small causal preparation:

```bash
PYTHONPATH=src python scripts/recflow/prepare_data.py
```

This refuses to replace an existing preparation. The compact result is under
`data/processed/recflow_v1/`; its aggregate audit is under
`results/recflow/preparation/summary.json`.

A historical bounded probe example, including one update (not the current
complete-epoch workflow):

```bash
python scripts/recflow/development_probe.py \
  --output results/recflow/development/my_probe \
  --catalog-size 1000000 --cohort-users 512 \
  --context 1024 --layers 6 --hidden 192 --heads 6 \
  --train-requests 150000 --eval-requests 512 \
  --steps 1000 --current-steps 300 --batch-size 32 \
  --beam-width 300 --max-train-seconds 600 --run-current
```

Approximate beam retrieval, sampled-candidate scoring and exact catalogue
ranking are different evaluation modes. Always retain the catalog size, search
mode, cutoff, candidate distribution/count, coverage and seed with a result.
Positive-only request panels retain all-OOV requests; no-positive requests remain
in history and are counted by the preparation audit. The current program selects
development users only and does not perform formal release admission.

`--decoder exact --eval-precision fp32` selects bound-pruned exact generation.
`--evaluate-checkpoint PATH --eval-day 22` re-evaluates saved weights; architecture,
catalogue and `--history-categories` must match their saved configuration. The
optional category-history variant uses only frozen known-item metadata.

Every new evaluation also saves its matched analytic random expectation. For
random-policy variation on a retained panel, run:

```bash
python scripts/recflow/random_baseline.py \
  --runs results/recflow/development/my_probe \
  --output results/recflow/development/my_probe_random --draws 5000
```

The random policy uniformly permutes videos in the same allowed catalog or
sampled pool. It retains OOV positives in the denominator. These draws describe
random-ranking variation on fixed requests, not training-seed uncertainty.
Both the initial and updated model must clearly beat their matched random
baseline before a configuration is called a useful foundation.
