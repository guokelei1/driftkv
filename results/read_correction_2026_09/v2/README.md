# Correction v2 — development exploration, 2026-09-28

Status: complete, 2026-09-28 17:20:30 Asia/Shanghai, exit code 0. All 15 edges,
30 curves and 105 measured points are retained. Its standalone figures and
execution queue are retired; [current Q/H results](../motivation_final/README.md)
use nonlinear Q-v5 and H-v4.
See [the complete result report](reports/20260928_complete/README.md).
The original 09:52 launch completed six edges before a free-memory admission
check stopped the queue at 11:21. The 15:13 tmux recovery verified existing
results and completed the remaining nine edges on GPU 0/1/2/3, waiting for
free memory before every calibration/evaluation stage.
All frozen experiment sources remained unchanged. Original failure records are
retained in `recovery/20260928T071313_905552Z/prior_records/`.
The six-edge snapshot remains in `reports/20260928_partial6/summary.json`.
The paper was unchanged by this run; v1 raw results and sealed evidence remain.

## Question and scope

Explore a steeper low-cost query-only recovery curve, and whether more expressive
explicit user-history processing raises the recovery ceiling at greater cost.
The user's illustrative percentages are development goals, not quality gates.

- Same 15 adjacent release edges and frozen 3,000-user panels per edge.
- Same chronological requests, saved Full/Reuse controls and compute denominator.
- Query-only calibration budgets: 128, 256, 512, 1024 users.
- History-conditioned calibration budgets: 32, 128, 512 users, plus 16 separate
  teacher-validation users per point. All teacher/validation costs are charged.
- Expanded fitting UIDs preserve the original first 512 and exclude every
  evaluation/canary UID at the same scale. No evaluation labels enter fitting.
- The outcome-conditioned evaluation panels are authorized development panels;
  these results do not estimate an unselected population or untouched test set.

## Method changes

**Query-only.** Still a head-local affine function of q, multiplied by retained
history length. Calibration queries combine recent known items from the user's
strictly pre-release history with uniform known-catalog items. A sequential
ridge fit initializes the rule. Joint optimization then follows the frozen
backbone to its final output: normalized within-user centered-logit error and
user-mean-logit error distill the Current teacher. Centered-logit error is
equivalent to an all-pairs logit-difference objective. The last eighth of
calibration UIDs selects the epoch using teacher outputs. Output-unit scaling
is folded into the affine weights before inference. No per-request user history
input is added to this correction.

**Explicit history.** Preserve the corresponding fitted v1 H as a frozen base.
An additional query-conditioned Transformer jointly projects the full K/V width,
adds chronological position, and executes one history attention/FFN block with
width D/2. All historical tokens precede the query. Its masked pooled output
predicts a response residual. The new output is zero initialized, then fitted
with normalized same-query teacher residuals. Sixteen users unseen by either
base or new branch select each layer's epoch. Every request executes the
additional history processing; no persistent summary or translated cache is
introduced. Original v1 fitting is included in each standalone H cost.

Calibration caches frozen lower-layer query states. After fitting one layer,
it applies that layer once and retains its output for the next layer. This is
numerically checked against the complete native reader and avoids repeatedly
executing already fitted lower history encoders.

## Quality and cost

Recovery = (method AUC − Reuse AUC) / (Full AUC − Reuse AUC).
Relative cost = extra calibration/inference FLOPs / (Full FLOPs − Reuse FLOPs).
All signs and every budget remain. A small subset with Full below Reuse is
reported with raw AUCs, not interpreted as positive gap recovery.

The FLOP ledger includes teacher/cache preparation, initialization, validation,
training and inference. Backpropagation/optimizer counts are explicitly labeled
analytical estimates. Costs can exceed 100%; they are not latency percentages.
The teacher cache preserves the paired stream's mapped-item tie order; the
existing saved Full scores retain their original ordering and are not replaced.

## Execution and evidence

`configs/read_correction_2026_09/v2/plan.json` is the prospective configuration.
`preflight.json` binds 231 model/panel/calibration input files. `readiness.json`
binds completed canaries and 58 frozen execution sources; `launch_record.json`
records the launched hashes. Checks passed: 15 focused numerical tests,
15,360 strictly pre-release calibration histories, 227 real rolling requests
against Full/Reuse controls, and an actual shard-resume check. Whole-model
resource probes covered all three scales at 1,024 history tokens; peak reserved
GPU memory was at most 28.11 GiB, below the 31.10 GiB allocation cap.
The completed four-GPU queue waited for available devices, capped GPU allocation
at 70%, and retained stage logs and failures. Quality was not a stopping threshold.

The run contains 105 measured points (15 edges times four Q and three H budgets).
`resource_estimate.json` estimates 3.64 hours on four available GPUs, with a
2.37–6.19 hour planning range. This is a prospective estimate using small probes
and the previous complete-panel workload, not a completion guarantee.

On the fixed 128-user Medium V0→V1 development subset, Q C128 improved recovery
from 11.67% (v1) to 40.85% (v2). The initial H candidate on that same subset
reached 97.05%, below its v1 base's 101.56%; the subsequent batched H fit improved
held-out teacher error. Complete-panel AUCs are now in the final result report.
The Max Q subset had
Full below Reuse and therefore does not establish positive gap recovery. All
preparation outcomes are retained; these probes do not establish the target
curve or replace the 3,000-user evaluation.

Each method writes `{method}/{scale}/{edge}/calibration_c*.{pt,json}` and
`rank0/shard_*.parquet`. Calibration resumes per completed budget; evaluation
resumes per sealed 128-user unit. `runtime/` holds progress and stage logs;
`status.json`, `launch_record.json`, `runtime.log` and `exit_status.txt` describe
the whole queue. The reporting interval is two hours, plus completed stages.

Candidate preparation evidence is under `probes/`; these subpopulations do not
replace the completed 3,000-user panels. Complete metrics are in `summary.json`
and the final report. Historical source hashes remain unchanged, but the retired
v2 queue cannot resume against them in the current source tree.
