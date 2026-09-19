# RecFlow development plan

2026-09-18, updated for the latest user instruction. The user authorizes
single-seed17 complete-window A/B development, parameter iteration, optional C,
and progression to the six-layer model after the small-backbone checks pass.
This supersedes the earlier pending 50–60-minute one-pass proposal: do not ask
again for that same launch permission. Continue within this authorized scope,
using a focused canary, measured resource estimates and detached tmux with a
retained log/exit status for jobs expected to exceed 30 minutes. No admission or
final-role outcomes, formal phi release decisions, or Yambda theta3 enter this
development work.

## Authorized expanded run and background handoff

The latest user asks to launch a larger six-layer run after focused canaries,
with three or five consecutive next-day checks, and return without waiting for
the long training. Use the [sealed4096-user setting](../../configs/recflow/window_6l_expanded_u4096_seed17.json)
and [new data audit](../../results/recflow/development/expanded_daily_panels_u4096_seed17/summary.json).
The4096 users are the fixed hash-order prefix of4569 development users with
at least1024 real exposures by D18; no future-activity filter is applied. The
old512-user cohort is an exact prefix, and the initial1M catalog is unchanged.

| Stage | Complete fitting days | Epochs | Eligible requests/epoch | Future evaluation |
| --- | --- | ---: | ---: | --- |
| A / model0 | D1–18 | 3 | 1,380,132 | D19 base/random |
| B / model1 | D19 | 1 | 50,694 | A/B on D20 |
| C / model2 | D20 | 1 | 40,273 | B/C on D21 |
| D / model3 | D21 | 1 | 33,555 | C/D on D22 |
| E / model4 | D22 | 1 | 31,630 | D/E on D23 |
| F / model5 | D23 | 1 | 29,454 | E/F on D24 |

Train a fresh seed17 A; do not continue the512-user checkpoint. Keep6L/H192,
6 heads,context1024,global batch128,LR1e-3,AdamW and the existing complete-epoch
shuffle/positive-sampling rules. A is evaluated at its declared epoch3 only;
updates at epoch1. Preserve all endpoint weights and training NLL/counts.
All five pairs are evaluated with the frozen uniform1000 NDCG@50 protocol:
6144 hash-selected requests/day, covering2253–2353 actual users/day, same
known-positive-plus1000-distractor pools and OOV denominators for both models.
Full next-day beam generation also remains as companion evidence on59,050–74,320
requests/day. Neither cutoff nor population changes in response to results.

Before launching, run a fresh four-GPU tiny A→B canary on this population,
including actual checkpoint/AdamW0→3→6 and tiny next-day paired/random scoring.
CPU checks cover all six fit/evaluation windows, complete known-index hashes,
nonempty four-rank tails and E/F predecessor mappings. Reuse the existing
numerical and known-positive resident-evaluation canaries for unchanged math.
This canary is execution evidence, never a recommendation-quality pass.

`run_expanded_daily.py` queues A–F and all paired/random reports in one tmux
session. It keeps command logs, exits, progress, source/config hashes and
actual serialized optimizer checks; `expanded_chain_report.py` refreshes a
cumulative report after each stage. All planned quality outcomes, including
regressions, are retained and the descriptive chain continues. Nonfinite
numerics, data or lineage mismatches stop the pipeline. No endpoint is
automatically admitted as a serving phi release or used for Reuse experiments.

Measured four-GPU training~337 requests/s gives3.57hours for4,326,002 optimizer
examples;746,348 full model-request evaluations at~26requests/s give7.97hours.
Allow **13–15hours total** including NLL, sampling, startup and checkpoint I/O.
User authorization covers this launch. After canary success, detach the full
job, verify startup and hand off; the user will return for results later.

2026-09-19 completion: the authorized pipeline exited0 at05:25:50 Asia/Shanghai
after11hours38minutes45seconds. Fresh A and all five daily updates completed
the declared epochs, with actual saved AdamW endpoints verified through33803.
The frozen uniform1000 NDCG@50 gains on D20–24 are respectively
**+32.83%,+45.46%,+15.56%,+21.68%,+14.93%**. A on D19 and all five
parent/current pairs pass their own matched-random checks. See the
[complete result record](../../results/recflow/README.md#completed-expanded4096-user-background-run)
and its raw-report links. This completes the expanded base-model development
chain, not RecFlow Full/Reuse motivation or adaptation validation.

The user confirmed keeping the static initial1M catalog to study cache
compatibility without adding dynamic item registration or historical OOV
recoding. ID mappings and generation paths stay fixed while embeddings,
backbone parameters and real event histories evolve. Preserve the current
primary, OOV misses and all companions; separately report coverage and any
known-item conditional diagnostic. The next research question is the matched
Full/Reuse gap and subsequent adaptation under identical history/candidate
rules. No further training or cache experiment is launched by this status note.

## Current provisional development choice

Use **six layers, context1024, seed17, daily1epoch, LR1e-3, uniform1000
NDCG@50** for the next development stage. On D20/D21/D22, each newly updated
model improves over its own parent on that same future panel by **5.88%,21.10%,
25.91%**. Every parent/current passes both matched-random checks. The
[three-edge report](../../results/recflow/development/window_6l_daily_sampled_seed17/daily_comparison/summary.json)
retains absolute scores, hit requests/users, all source/panel hashes and actual
saved AdamW continuation. The last probe completed in13.11minutes with exit0.

This means candidate ranking using the structured generator's normalized joint
probability: each request has all known positives plus1000 uniform distractors,
not a fixed1000-item total. Each day's768 requests retain all OOV misses. It is
a useful working development configuration, not proof of stable free1M
generation, untouched final-user qualification or a successful cache method.
Selection used the development grid and reused D22; keep this boundary explicit.
The full-catalog low-LR confirmation below failed and remains a failure.
The latest original-LR full-catalog edge improves22.13% at NDCG@50, but that
does not erase its much larger first two gains.

Freeze this candidate protocol for subsequent development; do not keep changing
K or distractor sampling to control each future percentage. Later population
checks and the three requested cache/motivation experiments remain separate.

## Completed learning-rate/metric screen and chronological confirmation

The bounded LR screen completed all four new phases in41.11minutes, with all
eight training/comparison commands exiting0. Together with the retained1e-3
reference, the report includes all six paired outcomes. The [training comparison](../../results/recflow/development/window_6l_daily_lr_seed17/lr_comparison/summary.json)
checks complete epochs, matching first-epoch targets/order, AdamW continuation,
checkpoint lineage and common future panels. The [metric grid](../../results/recflow/development/window_6l_daily_lr_seed17/metric_grid/summary.json)
contains all54 edge cells /27 settings: three learning rates, three scopes,
three NDCG cutoffs and two future days.

| Daily update LR | Full NDCG@50 D20 / D21 improvement | Full NDCG@100 D20 / D21 improvement |
| --- | ---: | ---: |
| 1e-3 | +420.12% / +608.95% | +173.47% / +583.51% |
| 1e-4 | +67.95% / +35.87% | +15.64% / +27.00% |
| 3e-5 | +64.25% / −11.40% | +7.65% / +17.34% |

All parent/current random checks pass in these full-catalog cells; the negative
3e-5 NDCG@50 edge still fails the improvement requirement. Lower LR is not
monotonically better at every cutoff. Sampled uniform1000 NDCG@50 with1e-3
gives+5.88%/+21.10%; with3e-5 it gives+3.00%/+4.90%. These are separate candidate
ranking tasks. Every popularity1000 setting in that original two-edge grid fails at least one parent/current
random check; its moderate gain percentages do not make it a working setting.

Select **1e-4, one complete epoch/day, free-catalog NDCG@100** for one subsequent
chronological development check. It preserves free generation and has moderate
positive gains on both explored days. This is an exploratory selection, not
evidence that NDCG@100 universally suppresses variation. Top100 hit support is
only12 requests/11 users→12/10 on D20 and7/7→8/8 on D21; random superiority is
not uncertainty on the difference between trained models. A separate
[parameter check](../../results/recflow/development/window_6l_daily_lr_seed17/parameter_drift/summary.json)
finds overall relative weight L2 movement7.12%,0.50%,0.10% for1e-3,1e-4,3e-5:
the smaller LR changes the actual update, while changing K changes its readout.
This parameter distance is not a K/V compatibility result.

The [sealed confirmation setting](../../configs/recflow/window_6l_daily_confirmation_seed17.json)
continues the exact selected C checkpoint/AdamW: D fits all4312 eligible D21
requests in34 steps and compares C/D on all9039 D22 requests. Primary NDCG@100,
LR, epoch count, window and cohort were fixed before either new D22 evaluation.
Retain all companion metrics and any failed/negative/large primary result.
D22 has appeared in other historical development branches; this is additional
chronological confirmation, not untouched final-role qualification. Existing
actual lower-LR and resident-evaluation canaries plus focused D date/panel and
NDCG100 checks cover the unchanged numerical paths. The measured runtime
estimate is12–16minutes on GPUs0/1/2/3, directly monitored with log/exit status.

**The additional full-catalog confirmation failed.** Its [three-edge report](../../results/recflow/development/window_6l_daily_confirmation_seed17/daily_confirmation/summary.json)
gives D22 NDCG@100 0.0000161960→0.0000439407 (+171.31%), with only2 requests/2
users→4/4 hitting Top100. Both models fail the random checks. The complete4312
requests/34 steps and actual saved AdamW4166→4200/LR1e-4 were verified; the job
took13.03minutes and exited0. Preserve this as a failed confirmation; the first
two moderate gains do not establish a stable free-generation chain.

The subsequent bounded probe froze **daily1epoch, LR1e-3, uniform1000 NDCG@50**,
whose earlier two edges are+5.88%/+21.10%, all random checks passing with roughly
100 hitting requests per768-request panel. It continued its own retained C through
D21 and compared C/D on D22. All9039 full-generation requests were also evaluated
as companions, including any failures. This explicitly changes the primary task
to ranking a fixed sampled pool with generative item probabilities; it does not
rescue the failed free-generation claim. D22 has now informed development of
another branch, so this is further exploratory development, not independent
confirmation or final-user qualification. New settings are fixed before this
branch's C/D evaluation; retain all outcomes and impose no preferred gain band.
The active scoring path sums normalized log probabilities for category1,
category2 and video; normalization remains over all legal paths at each step,
not the sampled subset. Candidate scoring does not use beam pruning. Same
seed/request identity and catalog give identical parent/current pools, and raw
video ID breaks ties independently of positive injection order. D22's sampled
panel includes451 all-OOV requests out of768, kept at zero; actual candidate
counts are1000–1005. Uniform distractors are evaluation candidates, not confirmed
negative user feedback or persistent history events.

### Original prospective screen

The latest user asks whether smaller, more gradual update gains are possible.
Keep the completed A3, all frozen daily panels, seed17, global128, four ranks,
one full epoch/day and primary free FP32 beam300 NDCG@50. Test only update
learning rate1e-4 and3e-5, using the retained1e-3 one-epoch branch as reference.
Each new setting completes D19→D20 and D20→D21. Restore real AdamW and then set
the configured LR; a tiny actual-A canary verifies saved LR/steps and changed
parameters. Reuse the existing A/day20 evaluation only after checkpoint,
window, ordered panels, decoder and raw-aggregate checks. This saves two
otherwise identical GPU evaluations without changing any denominator.

The prospective settings are
[`window_6l_daily_lr1e4_seed17.json`](../../configs/recflow/window_6l_daily_lr1e4_seed17.json)
and [`window_6l_daily_lr3e5_seed17.json`](../../configs/recflow/window_6l_daily_lr3e5_seed17.json).
Allow40–50minutes for four serial four-GPU phases and six new full evaluations,
with detached tmux and retained logs/exit status. `run_daily_lr_probe.py`
orchestrates this bounded screen. Report **relative improvement percentages**,
absolute NDCG, hit requests/users and matched random for every edge. The user's
preferred20–50% range is descriptive; retain zero, negative and larger outcomes.
Top-K changes need not be monotonic in LR. This experiment addresses update
step size, not initial supervision coverage or final-population qualification.

The subsequent user clarification also permits choosing a different NDCG
configuration during development. Compare the already computed cutoffs20/50/100
across full-catalog generation and the separately defined uniform1000 /
popularity1000 candidate-ranking diagnostics, for all three LRs and both
edges. Each scope uses its own exact request panel, OOV denominator and matched
random; do not equate a sampled-pool result with free1M generation. Preserve the
original frozen free NDCG@50 reports. The exploratory grid records all cells,
absolute scores, gain percentages, hit users and two-edge working flags, with
no automatic winner. Freeze a suitable interpretation/configuration before
checking additional windows; repeated tuning on the same days is not stability
confirmation. The retained1e-3 uniform1000 NDCG@50 improves5.88%/21.10% and passes
the random checks, whereas popularity1000 NDCG@100 improves46.54%/48.92% but
does not pass all parent/current random checks. The latter is not a working
setting merely because its percentages match a preferred range.

## Completed daily-update screen; next expansion remains separate

The latest user request pauses six-layer data expansion to test whether a
one-day update is more consistent. Reuse the completed six-layer A_epoch3;
keep seed17, the512-user cohort,1M catalog,context1024,lr1e-3,global128 and the
free beam300 FP32 NDCG@50 protocol. The metric is NDCG, not AUC: a large ratio
at this small absolute baseline is not comparable to an AUC percentage-point gain.

The [prospective daily configuration](../../configs/recflow/window_6l_daily_seed17.json)
fits D19 completely and evaluates both predeclared epoch1 and epoch3 on all
10,171 D20 positive requests. Then each branch continues from its own B model
and AdamW: fit D20 for the same1 or3 complete epochs, compare its parent and
current on all8,866 D21 requests. The
[continuation configuration](../../configs/recflow/window_6l_daily_continuation_seed17.json)
evaluates only each declared final endpoint. All OOV misses remain; each day's
768-request sampled diagnostics have their own random denominator. Preserve all
four update results, absolute deltas, ratios, hit requests/users, and matched
random expectation/null99. No target percentage or gain upper bound is imposed.

This is a daily-feasibility comparison, not a controlled attribution of the
previous3-day gain: the completed3-day branch has different fit/future dates.
The same-day future arrays are exactly the original D19–21 full panel's D20/D21
slices. The new daily lineage guard rejects a parent already fitted on the
new update's dates. A four-rank263-request canary checks continuation from the
actual trained A. Allow40–60minutes for the complete probe, with serial phases
in tmux and four evaluation workers; complete training itself is only a few
minutes and free-generation evaluation dominates. No larger-user training or
KV-reuse result is admitted by this screen.

The [completed daily report](../../results/recflow/development/window_6l_daily_seed17/daily_comparison/summary.json)
passes all four predeclared edges. One-epoch daily updates give D20
0.0000635683→0.0003306315 and D21 0.0001286080→0.0009117615; three-epoch
updates give D20 0.0000635683→0.0007945708 and D21
0.0002715188→0.0008594383. Every parent/current exceeds twice matched random
and null99 (actual multiples11.29–206.11). Both settings improve on both
future days, but more epochs are not uniformly better. Continue with one
complete epoch per day as the minimum working development setting. This is
one seed,512 users and two updates, with rare hits and low absolute NDCG;
it does not prove long-term stability or formal phi admission.

The entire daily pipeline completed in48.77minutes, all exits0, GPUs released.
All four evaluated endpoints are retained; unused intermediate weights and the
tiny-canary weight are listed in the
[cleanup record](../../results/recflow/development/window_6l_daily_seed17/checkpoint_cleanup.json).
The [supervision coverage analysis](../../results/recflow/development/window_6l_full_seed17/initial_supervision_coverage.json)
shows that only about51–54% of the next three days' known-positive label
occurrences appear in the512-user initial positive pool, versus82–84% in the
fixed2048-user initial pool. This supplies a reason to test broader initial
training data; it is not measured model improvement. Do not force update
gains toward a preferred percentage or compare NDCG ratios to AUC point gains.

After the daily screen, resume the authorized six-layer data expansion only
with a stated setting, and check Full-only quality before cache diagnostics.
The current paper's three targets are: (1) release gain lost when Current reads
Parent KV, (2) limited recovery from small exact-KV replacement interventions,
and (3) compact user/query-dependent response correction plus its change under
real history appends/evictions. The narrative source is
`/home/gkl/work/paper/main.tex`, Motivation and Insight; new competitor scripts
do not replace the paper's34-intervention Motivation2 definition.

For RecFlow, first add an external-history-cache path to the existing generator's
candidate scorer, retaining the same path for decoder companions. Persist real
exposures only; discard `[REC]/c1/c2` generation tokens.
Same-model cached reads must equal Full, and complete same-query response
correction must equal Current Full. Retain native SiLU attention, relative
position bias and1027 normalization: historical oracle readers that omit the
bias cannot be reused unchanged. The paper's quality comparison uses three
rolling paths initialized at cutover; request-local A/B scores cannot be spliced
into a rolling Reuse denominator. Partial exact-KV replacement remains an
oracle intervention, not a compute-saving executable action. Structured video
scores are joint log probabilities, so the old sigmoid/Bernoulli diagnostics
cannot be carried over literally. Oracle anchors/heldout queries should be
distinct legal category paths, avoiding repeated leaf queries for videos in
the same category. No such RecFlow cache result has yet been established.

## Completed six-layer A/B/C development

Six-layer A/B/C completed with exit code 0 and all predeclared full-window
working checks pass in the [final report](../../results/recflow/development/window_6l_full_seed17/ABC_final_comparison/summary.json).
The last update improves the complete D25–27 request mean by 37.84%, but **C
regresses on D27**. This is seed 17 evidence from 512 development users with low
absolute NDCG; it does not establish improvement every day, formal phi
admission, final-user performance, or any benefit from K/V reuse/adaptation.
All four GPUs have been released.

The completed 2L full-window A/B sensitivity check supplied the learning
prerequisite for the authorized six-layer development step. On all 25,694
positive D22–24 requests, A/B free NDCG@50 is 0.0000337935/0.0003984221,
or 9.12x/107.56x matched random. Both exceed null99=0.0000238634 and twice the
random expectation; B improves on each day. Top50 hits cover 10 requests/10
users for A and 63 requests/54 users for B. See the
[complete matched report](../../results/recflow/development/window_2l_seed17_full_windows/AB_comparison/summary.json).
This traffic-weighted sensitivity result preserves the original 3072-request
balanced-panel failures; it is not formal release admission or a qualified
multi-version chain.

The completed six-layer setting is
[`window_6l_full_seed17.json`](../../configs/recflow/window_6l_full_seed17.json):
6L/H192/6 heads, context1024, the same 512 users, 1M catalog, global batch128,
learning rate 0.001 and three complete shuffle epochs. Epoch3 is the retrieval
endpoint; fixed training-only NLL remains recorded each epoch. The
[combined input manifest](../../results/recflow/development/full_window_training_seed17/summary.json)
copies all original training arrays and known-request hashes unchanged and uses
the complete three-day evaluation windows with the original 768 diagnostic
subsets. The [four-GPU resource canary](../../results/recflow/development/ddp_6l_h192_heads6_c1024_k1m_seed17_gpu0123_retry/summary.json)
passed at 337.03 requests/s, versus 199.53 on two GPUs; including proportional CPU
preparation gives 330.66/s. The four-rank pipeline canary consumed 263/263 requests
in three steps including the 7-request tail and passed fixed-NLL and serial 8/4
evaluation checks. Its eight free requests were all OOV, so this establishes
execution only. The [resident-rank parallel-evaluation canary](../../results/recflow/development/gloo_resident_6l_parallel_eval_canary_seed17/summary.json)
also passed with eight known-positive requests and four sampled diagnostics:
serial/parallel results matched and optimizer state was unchanged.
**Six-layer A completed with exit code 0 in 2500.02 seconds (41.67 minutes).** Each
of its three epochs consumed 173,395 requests in 1355 steps, global batch128
on four GPUs. Every epoch's request-order and sampled-target hashes match the
original 2L A. Fixed training-only NLL was
17.40990→13.24079→12.52489→12.11270. On all 29,027 positive D19–21 requests,
free NDCG@50=0.0001196062, or 19.85x random expectation 0.00000602561,
above null99=0.0000256533. Top50 hits cover 27 requests/26 users,
distributed 14/4/9 across the three days. The [completed A report](../../results/recflow/development/window_6l_full_seed17/A_final_comparison/summary.json)
retains the full metric/random comparison and A's passing base check.

**Six-layer B completed with exit code 0 in 1828.98 seconds (30.48 minutes).**
It inherited A_epoch3's model and AdamW state, then completed three epochs of
17,046 requests/134 steps. Epoch request-order and sampled-target hashes match
the original 2L `B_retry`; fixed training-only NLL after each epoch was
12.07335→10.38078→9.66543. The [final A/B report](../../results/recflow/development/window_6l_full_seed17/AB_final_comparison/summary.json)
has `working_flags_met=true`: on all 25,694 D22–24 positive requests, A/B
NDCG@50=0.0000524797/0.0004155888, or 14.17x/112.20x matched random.
Both exceed twice expectation and null99, and B>A on every day. Top50 hits cover
14 requests/14 users for A and 64 requests/54 users for B; B's daily hit counts
are 28/20/16. This is single-seed development evidence, not formal admission.

**Six-layer C completed with exit code 0 in 1761.35 seconds (29.36 minutes).**
It inherited the exact B_epoch3 model and AdamW state and completed three
D22–24 epochs of 10,854 requests/85 steps each. Input-order and sampled-target
hashes match the corresponding 2L C epochs; fixed training-only NLL was
14.57809→12.13766→9.84625→8.90482. On all 25,180 D25–27 positive requests,
B/C NDCG@50=0.0001211737/0.0001670278, or 35.06x/48.33x matched random.
Both exceed 2x expectation and null99; the complete-window C mean improves by
37.84%. Top50 hits cover 14 requests/13 users for B and 32 requests/28 users
for C, with C's daily hit counts 13/12/7.

**The daily result is mixed:** C>B on D25 and D26, but on D27 C=0.0000947085
is below B=0.0001893917. The predeclared gate uses the full three-day
traffic-weighted request mean, so the window check passes; it must not be
described as daily improvement or generally stable updates. The completed
three-model development chain is not formal phi admission and does not test
cross-version K/V reuse. Original small-model failures remain below.

Final six-layer model/optimizer checkpoints are retained at:

- `results/recflow/development/window_6l_full_seed17/A/A_epoch3/checkpoint.pt`
- `results/recflow/development/window_6l_full_seed17/B/B_epoch3/checkpoint.pt`
- `results/recflow/development/window_6l_full_seed17/C/C_epoch3/checkpoint.pt`

The [completed cleanup manifest](../../results/recflow/development/window_6l_full_seed17/checkpoint_cleanup.json)
records removal of six unevaluated epoch1/2 checkpoints and the completed tiny
canary weights, releasing 15.37 GiB. Final A3/B3/C3 model and optimizer states
remain; C's saved AdamW step is 4722. All results, raw metrics/rankings, logs
and failures remain. Replaying the deleted canary weights requires regenerating
its three-step checkpoint from the retained source/configuration.

## Completed 2L setting and retained checks

The first complete-window setting is
[`configs/recflow/window_development_seed17.json`](../../configs/recflow/window_development_seed17.json),
run by `scripts/recflow/window_chain.py`, which reuses the existing evaluator.
It used HSTU 2L/H96, context1024, the same fixed 512 development users and 1M
initial catalog, batch128 and learning rate 0.001. Both A/B canaries passed:
263/263 requests were consumed, including the last 7-request batch; fixed
training-NLL hashes matched, B inherited the actual optimizer state, and the
separate sampled-panel random analysis loaded correctly. Main A completed in
tmux `recflow_window_A_seed17`, with output
`results/recflow/development/window_2l_seed17/A`, adjacent `A.log` and
zero `A.exit_status.txt`. B also completed in `B_retry`. Their final comparison
is `window_2l_seed17/AB_final_comparison/summary.json`; its original balanced
panel **does not pass** the whole working gate because A is too sparse on
D22–24, despite a clear B gain. The separately reported full-window result
above does not overwrite that outcome.

| Phase | Complete fitting window | Known-positive requests per epoch | Fixed future comparison |
| --- | --- | ---: | --- |
| A | D1–18 | 173,395 | A versus random on D19–21 |
| B, continued from A | D19–21 | 17,046 | A, B and random together on D22–24 |
| Optional C, continued from B | D22–24 | 10,854 | B, C and random together on D25–27 |

There is no 150k request cap or timed training endpoint. Each epoch visits every
known-positive request once, including the final partial minibatch; all-OOV
fitting requests are counted explicitly and excluded only from the known-item
loss. The existing objective samples one known positive per request. Each phase
has a predeclared budget of **three complete epochs**, with retrieval evaluation
after epochs 1 and 3 and fixed training-only 2048-request/target NLL before training
and after every epoch. Epoch1 is a learning-curve companion; epoch3 is the
predeclared candidate, not the checkpoint selected by future-label performance.
Three epochs are a budget, not evidence of convergence. Record any remaining
undertraining and state subsequent parameter changes explicitly.

The primary metric is **free-catalog FP32 beam300 NDCG@50** over the configured
1M-video catalog, with matched random comparison. The original 2L protocol froze
1024 positive requests per day, 3072 per three-day window. An independent fixed hash chooses 256 per day
from that panel for the separate 768-request uniform/popularity1000 diagnostics.
Those diagnostics have their own denominators and random baseline; failure of
a sampled diagnostic does not automatically fail the full-catalog primary.
NDCG@20/@100, Recall@20/@50/@100, per-day directions and OOV coverage remain
companions, not replacement primary metrics. A/B and B/C share their respective
future panels; no later-activity user filtering or favorable-day selection.

A must clearly beat random on D19–21. On D22–24, compare A and B against the
same random baseline and require B to improve over A; the active setting also
checks that both models beat random there. Optional C repeats the fixed-window
comparison. Once these checks are stable, proceed to the already authorized 6L
protocol with its own focused canary and resource estimate. This is development
progression, not formal release admission or a multi-seed claim.

The measured 2L batch128 training rate is 353.59 requests/s, giving about 8.17
minutes per A epoch. Including fixed evaluations and allowance for runtime
variation, the preset estimates A 35–45 minutes, B 18–28 and optional C 17–27.
These are resource forecasts, not measured complete-window runs. The frozen
training pools, evaluation arrays and hashes are in
[`seed17_complete_epoch_windows`](../../results/recflow/development/seed17_complete_epoch_windows/summary.json).

The user subsequently allowed all four GPUs for both within-experiment and
between-experiment parallelism. Original A retained its single-GPU process;
its first complete epoch covered 173,395 requests in 406.16 seconds (426.92/s).
The independent same-batch 2L canary on the NVLink-connected GPU2/3 pair measured
437.28/s single-GPU versus 788.07/s DDP (1.80x; 754.31/s including proportional
CPU input preparation). Global batch remains 128, with no repeated tail rows;
two unequal-tail FP32 gradient/AdamW reference updates passed. Evidence:
[`ddp_2l_h96_c1024_k1m_seed17_gpu23`](../../results/recflow/development/ddp_2l_h96_c1024_k1m_seed17_gpu23/summary.json).
This measured option was used for later phases after the runner canary. Parallel
evaluation must restore frozen request order and recompute global user means.
The original A settings file remains unchanged so its recorded hash is valid.

A completed all three fitting epochs, each with 173,395 requests, before B
was allowed to load `A_epoch3/checkpoint.pt`; its final evaluation can overlap
B execution on disjoint GPUs. B uses the unchanged learning/evaluation preset
in `window_development_seed17_ddp.json`, training on physical GPUs2/3 and
evaluating on2/3/1. An initial scheduling attempt (`window_2l_seed17/B`) stopped
before any B update because a waiting NCCL rank blocked its GPU's evaluation
worker. All logs and an explicit interruption record remain. The main waits
now use CPU Gloo, actual training still uses NCCL, and a real-checkpoint
eight-request resident-rank canary passed. `window_2l_seed17/B_retry` is the
fresh main retry; this execution failure has no B learning-quality conclusion.

The completed shuffle baseline gives free NDCG@50 of 0.00035037 for A on
D19–21 (58.68x random, five hit requests). On the identical D22–24 panel,
A=0.00002275 (one hit, below random's99th percentile) and
B=0.00077541 (199.91x random,14 hit requests/13 users). B improves on all three
days; this balanced panel did not establish the old A's random superiority.
Original C subsequently completed a second update without hiding this A failure.

The one new parameter setting is
[`window_chronological_seed17.json`](../../configs/recflow/window_chronological_seed17.json):
within every complete epoch, visit three-day training blocks chronologically
and shuffle requests inside each block. All18 fitting days, each eligible
request, three epochs, seed17, loss, target sampling, global batch128, frozen
future panels and primary metric remain the same. This tests whether uniform
mixing of old and recent training requests weakens the initial future model;
it does not select a favorable evaluation day. A has six blocks; B/C have one,
where order and target draws are exactly the original shuffle behavior. A tiny
CPU reference verified full coverage, monotonic blocks and single-block RNG
equivalence. Chronological A/B completed on GPUs0/1 under
`results/recflow/development/window_chronological_2l_seed17/A`, independently of
original C. Its final comparison is recorded below; this ordering variant does
not proceed to C because its parent random-tail check still fails.

Original C completed: on D25–27, B=0.00001966 (one Top50 hit, random-tail
check fails) and C=0.00023507 (four hits,68.29x random, check passes). D25/26
improve and D27 ties at zero. The original3072-request chain remains unqualified.
To measure whether sparse evaluation obscured the old-model signal, an
independent complete-window sensitivity check kept final epoch3 weights,
seed, catalog, decoder, cutoff and OOV treatment fixed. The full panels contain
every positive request from the same512 users:29027/25694/25180 on
D19–21/D22–24/D25–27, with the original768 sampled-diagnostic subsets. The manifest is
`results/recflow/development/full_window_panels_seed17/summary.json`.
This full-window mean weights requests by traffic rather than1024 per day;
it must not overwrite or be described as passing the original balanced-panel
gate. CPU random diagnostics predict narrower variability, not model success:
null99 on the latter two windows is still6.44/6.37 times the random mean.
Original A and B were both evaluated on the same full D22–24 window; both
endpoints and the complete request set were fixed before comparing outcomes.
The evaluator/merge and resident-rank canaries cover this unchanged scoring path.

The original A full D22–24 evaluation completed: NDCG@50=0.0000337935,
random expectation=0.00000370413, random99th percentile=0.0000238634.
It passes the same2x/null99 working conditions with10 hit requests/10 users,
distributed2/6/2 across D22/23/24. This supports a model signal that the sparse
balanced panel could not establish, without changing its recorded failure.
The paired B evaluation, queued before seeing this A result, also completed:
NDCG@50=0.0003984221,107.56x random,63 Top50 hit requests/54 users
(D22/23/24:31/23/9). Both models pass the full-window random checks and B>A on
each day. All companion cutoffs, sampled diagnostics and original-panel
failures remain in the [matched report](../../results/recflow/development/window_2l_seed17_full_windows/AB_comparison/summary.json).

Chronological A also completed all three epochs. On its original D19–21 panel,
NDCG@50=0.000500554 (83.83x random),11 hit requests/10 users, all days positive.
Its completed D22–24 comparison gives parent A=0.0000368012
(one Top50 hit,9.49x random but below null99) and B=0.0009342337
(12 hits/11 users,240.86x random). B improves each day, but this chronological
variant still fails its parent random-tail check and will not enter C. See its
[final A/B report](../../results/recflow/development/window_chronological_2l_seed17/AB_final_comparison/summary.json).
These parameter experiments do not replace the original frozen endpoints or
their full-window check.

## Stages and advancement criteria

1. Prepare the complete realshow archive in an isolated RecFlow directory. Audit
   request boundaries, strict timestamp prefixes, fixed population/catalog and
   OOV coverage. Advance when a small reference agrees with generated histories.
2. Connect a small generative task end to end: history, structured item path,
   free decoding, raw video lookup, multi-positive request evaluation. Advance
   when teacher-forced and freely scored paths agree and compute is measured.
3. Compare a small development grid of context/data/catalog sizes and retrieval
   cutoffs. All-catalog retrieval is the reference; sampled candidate metrics
   are separately named sensitivity diagnostics. Choose a useful, stable and
   affordable setting, retaining all compared outcomes, then freeze it. Compare
   each setting with its own matched random expectation; do not choose an easier
   candidate pool or cutoff merely because it creates a passing result.
4. Train one small development parent and one current candidate on consecutive
   causal windows. Compare both Full on exactly the same future requests. An
   endpoint is not an admitted release. The initial base and the updated model
   must demonstrate useful learning relative to random under the same declared
   task, as well as a future-window update gain. No selected-edge or final-user tuning.
5. Once the random prerequisite and the preceding development checks pass,
   continue to the authorized six-layer development with a canary and resource
   estimate. The previously considered ten-layer family is not the active A/B
   step. Retain rejected candidates' evidence;
   they do not change the serving parent or its cache lineage.

## Working protocol

- Core modules: `src/hstu_kvcache/recflow/`; orchestration: `scripts/recflow/`;
  configs: `configs/recflow/`; data: `data/processed/recflow_v1/`; compact
  evidence and local runtime output: `results/recflow/`.
- Completed backbones: HSTU-native 2L/H96 and 6L/H192/6 heads, context1024.
  The six-layer A/B/C full-window checks pass, with C's D27 regression retained.
  The retained 10L/H320 probes are historical resource
  evidence. Small development fits are not claimed foundation assets.
- Initial fitting cutoff D18 (2024-01-30). Stable initial UID roles separate
  development, model admission and final evaluation. Only development users
  enter configuration or model-quality probes. A smaller pilot vocabulary is
  selected using initial development data, not future targets.
- Freeze item paths at the initial cutoff: first observed `(category1,
  category2, video_id)`; category pairs need not have a unique parent in the raw
  data. Later metadata changes do not recode existing items. Historical events
  remain one token each. PAD=0, known=1..K, OOV=K+1.
- Every target request uses only events with strictly earlier timestamps,
  excluding the entire target request and other same-timestamp requests. The
  within-request history order is a deterministic encoding convention, not a
  claim about observed watch order.
- Realshow exposure history is retained, including ineffective views. Positive
  targets are distinct effective-view videos in the complete request. No-positive
  requests still update history. Stage candidates never become observed history
  or observed negative feedback.
- Generate category1, conditional category2, then a unique video leaf. All
  category tokens are predicted at inference. Constrained decoding ranks by
  joint log probability and shares the historical prefix. Output tokens are
  transient. No target-conditioned candidate shortlist in the reference metric.
- Active primary: free-catalog FP32 beam300 NDCG@50, with NDCG/Recall at 20/50/100
  retained as specified companions. Use full observed positive
  sets for denominators, retaining OOV misses and all-OOV requests. Report target
  coverage and distinctly named warm-conditional metrics. No-positive requests
  have no positive-retrieval score and their count is reported. Request mean is
  primary, user mean supplementary. New iterations use seed17 only; do not
  infer between-seed stability from request counts. Historical seed23/29
  results stay retained but are not additional new runs.
- The first prototype fixes the initial catalog, including a reduced development
  catalog when explicitly configured. It does not establish cold-start coverage.
  Catalog size and OOV effects are experimental outcomes, not silently excluded.
- Reporting includes absolute quality, Parent/Current Full changes and measured
  training/retrieval cost; loss alone does not establish free-generation quality.
- Historical feedback uses logged effective-view/like values. The archive does
  not provide feedback completion times, so availability before the next request
  remains an offline-data assumption, not an audited arrival-time guarantee.
- Event time deltas are invariant under window eviction: a truncated first
  event retains its gap from the preceding chronological event. Only a user's
  original first event has gap zero.

## Random prerequisite and protocol selection

The user requires convincing superiority to random. The operational development
screen records two separate checks: trained quality exceeds the fixed-panel
random-policy 99th percentile, and is at least twice its analytic expectation.
These numerical checks are our working implementation of the requirement, not
numerical thresholds quoted from the user and not automatic model admission.
Retain all NDCG/Recall@20/50/100 results and every compared candidate pool. A
passing cell does not qualify the other cells or a different task.
Apply the active gate to the predeclared full-catalog primary. Uniform/popularity
sampling is diagnostic; a failing sampler must not be silently promoted into a
veto on a passing primary, or a passing sampler into a substitute for a failing
primary.

For a request with `y` observed distinct positives, `m` inside the configured
catalog and `N` available candidates, random ranks uniformly permute the videos.
The reference uses the complete configured catalog; injected-positive sampled
diagnostics use `N=min(catalog_size,m+distractors)`. Uniform and popularity
distractor pools have the same conditional random-ranking expectation at equal
`y,m,N`, even though the trained model finds them differently difficult. Keep all
OOV targets in Recall and ideal-DCG denominators, all-OOV requests at zero, and
exclude only zero-positive requests from positive-retrieval means.

The retained CPU analysis uses 5000 random-policy trials per fixed panel,
seed 20260918, exact without-replacement positive rank positions, and analytic
expectations. Its intervals describe random-policy variation on those requests;
they are not training-seed or population confidence intervals. Identical panels
reuse identical trial arrays across models and samplers. A deterministic replay
reproduces the saved arrays exactly; a four-item reference also checks duplicate
rank rejection, repeated request groups, all-OOV and zero-label handling.

Candidate construction is part of the task definition. In particular,
Recall@100 with only 100 distractors can nearly saturate under random ranking;
larger absolute scores do not imply more learning. We must choose a protocol for
the intended recommendation scenario and confirm it on untouched development
evidence, rather than switch from popularity to uniform sampling after seeing a
failure. Sampled-only wins do not qualify free generation. A large model/random
ratio on a sparse free-ranking panel is also insufficient if a random-policy
tail can produce it. The new stage fixes seed17 only; historical seed17/23/29
evidence remains reported at its original scope, without a new multi-seed claim.

## Historical prospective three-day release timing

The active A/B/C development windows are defined above. This earlier possible
formal-release schedule is retained for context, not used to launch the current
complete-window tests or to claim admission.

| Candidate | Incremental fitting days | Admission day | Following report days |
| --- | --- | --- | --- |
| phi0 | D1–18 | D19, base qualification | D20–22 |
| phi1 | D19–21 | D22 | D23–25 |
| phi2 | D22–24 | D25 | D26–28 |
| phi3 | D25–27 | D28 | D29–31 |
| phi4 | D28–30 | D31 | D32–34 |
| phi5 | D31–33 | D34 | D35–37 |

Admission uses only its designated UID role and finishes before a serving
switch. Configuration development never reads the final role's outcomes. The
six names supersede the earlier prospective `phi0..phi3` length for RecFlow only.
This is a working development plan, not a sealed launch contract.

## Retained preparation and historical pilot record

The following records precede the active complete-window preset. Preserve their
results and limitations; superseded next-step suggestions are not current
execution instructions.

- Stage 1 complete: all 37 days prepared in 104.99 seconds, 38,245,315 real
  exposures and 7,440,759 requests; 2.00 GB compact arrays. Initial population
  has 5,688 users with at least 1024 exposures; 4,569 are development users.
  Request boundaries and strict-time prefixes pass the data reference check.
- Stage 2 complete as an executable prototype: teacher/free-path probabilities,
  target isolation, gradients, rolling input time semantics and OOV-aware metrics
  pass focused checks. This is not a learned-quality qualification.
- Resource probes fit one A40: 6L/H192/full initial catalog has 950.9M parameters,
  context1024/B16 measures 39.32 requests/s and 21.90 GiB reserved peak. A 1M-item
  catalog gives 194.5M parameters at 6L/H192 and 326.8M at 10L/H320. The 10L
  context1024/B8 probe measures 24.30 requests/s and 11.30 GiB reserved peak.
  These are short probes, not fully trained models or end-to-end runtime promises.
- Stage 3 is open. The first 200k-catalog/context128 development fit is below
  initial popularity on its 1024-request D19 panel. Its original sampled metrics
  used stable sorting after positive injection and must not select the protocol;
  a separate corrected re-evaluation uses raw video ID to break score ties.
  The corrected evaluation also uses the fixed rolling time delta, recorded as
  an input revision. All previous outputs are retained.
- Search audit on 64 requests: beam100/300 recover only 60.16%/77.78% of exact
  Top100 sets. They describe actual approximate generator outputs, not exhaustive
  catalog rankings. A bound-pruned FP32 decoder now agrees with exhaustive
  ranking on all 16 checked requests (maximum score difference 9.54e-7), expanding
  15.63% of paths in 3.11 seconds versus 12.94 seconds. This is a context128
  development check. A subsequent context1024/catalog1M exact FP32 canary with
  both uniform/popularity1000 scoring evaluates 8 D22 requests in 6.531 seconds,
  with 16.406 GiB peak live allocation and 37.053 GiB reserved. It loads a checkpoint
  without optimizer state, so it is a cost/correctness canary, not a quality gate.
- Stage 4 seed17/23/29 development pilots completed with 6L/H192/context1024/catalog1M,
  512 fixed development users and a shared fixed D22 panel; seed17 also has a
  D19 base check. The measured catalogue
  coverage is a limitation: on these users, D22 positive-target coverage is
  12.87%/23.73%/30.32% for 200k/1M/full initial catalogues. It remains in all
  main denominators. No model admission, final-user scoring or long training
  has been launched. All three seeds now have the same fixed-panel random
  comparison; this establishes the limits below, not qualification.
- The matched random analysis does **not** qualify the initial base. For the
  ordinary seed17 pilot, popularity1000 NDCG@50 on D19 is 0.007678 against random
  expectation 0.007985 (0.96x). On D22 the parent is 0.002726 against 0.003981
  (0.68x), while the current model reaches 0.012016 (3.02x) and exceeds the random
  99th percentile 0.007079. Current uniform/popularity1000 NDCG@20/50/100 all pass
  both working checks; popularity Recall@100 does not reach 2x. The history-category
  variant has the same parent-fails/current-improves pattern. These are sampled
  diagnostics, not proof that free generation qualifies.
- The seed23/29 D22 popularity1000 NDCG@50 parent/random ratios are 1.28x/1.07x,
  failing the same combined screen; current/random ratios are 2.50x/2.91x and
  both pass. All three current seeds pass uniform/popularity1000 NDCG@50/100.
  These sampled gains are reproducible across the frozen seeds, but do not
  qualify the parents. The larger free-generation check below adds evidence
  beyond these sampled diagnostics.
- A supervision reconstruction shows that ordinary seed17's parent consumed
  58,784 of 136,928 eligible selected requests (42.93%), choosing 53,179 distinct
  positive videos; the update consumed 16,000 of 17,046 requests (93.86%). Only
  16.01% of known D22 positive occurrences had been selected as a parent training
  target, versus 50.45% present somewhere in the entire selected initial request
  pool. The union of actual parent/current chosen positives covers 24.77%.
  This is consistent with limited direct positive supervision, not evidence that
  other embedding rows had no gradients: history inputs and within-category
  softmax also train them. Original per-draw logs were not retained; request
  counts and selection hashes agree with independent records, while draw identity
  is reconstructed from the saved RNG/seed. More training is a hypothesis to
  test under the random gate, not a promised solution.
- In the original 512-request panels, no retained seed17/23/29 free-catalog beam
  result passes both random checks at any
  reported NDCG/Recall cutoff. For example, ordinary current D22 NDCG@100 is
  0.0001490, about 23x its tiny random expectation, but below the random 99th
  percentile 0.0001906. The corrected 200k probe passes only uniform1000
  NDCG@20/50; its popularity1000, free-catalog and all Recall cells fail the
  combined screen. Original sampled metrics with positive-priority tie breaking
  are excluded entirely from this random comparison.
- The larger free-generation check is complete: FP32 beam300, not exact search,
  on all 9039 D22 positive requests from 437 active users of the same fixed
  512-user cohort. All four checkpoints share identical request, user and
  positive-count arrays. Catalog target coverage is 23.7348%; 5025 all-OOV
  requests remain zero. Each evaluation took 997–1011 seconds, without training.
  Both current models pass both random checks at every NDCG/Recall cutoff;
  both parents fail the random 99th-percentile check at every cutoff. At NDCG@50,
  random expectation is 0.00000392922 and its 99th percentile is 0.0000429740:
  seed23 parent/current score 0.0000256914/0.000668752 (current 170.20x random),
  and seed29 score 0.00000781894/0.000239905 (current 61.06x). Top100 hits occur
  in 3/48 requests for seed23 and 2/22 for seed29; current hits span 45/20 users.
  This establishes neural free-generation learning and update signal on this
  development task, while absolute retrieval remains low and the initial bases
  still fail. It does not qualify a version chain.
- The two complete seed23/29 checkpoint pairs were selected by asset availability
  before inspecting the new panel. The original seed17 run predates parent
  checkpoint retention and remains limited to its retained 128/512-request
  evidence, without an unpaired large-panel run or a fabricated third pair.
  Restricting the new FP32 results to the old 512 requests leaves both current
  models at zero, matching their old BF16 results; seed23 parent's one-hit
  NDCG@50 changes slightly from 0.000160353 to 0.000158603. Thus the current hits
  missed by the small panel are not merely a precision-switch effect. Retain
  both panels and the explicit precision comparison.
- The former 50–60-minute six-layer one-pass proposal,
  `configs/recflow/medium_development_one_pass.json`, was **superseded before
  launch** by the latest user instruction and the active small-backbone preset.
  Its 136,928-request initial pool still came from the old 150k cap; the new full
  D1–18 window contains 173,395 eligible requests. Its single-day 512-request
  evaluation and exact-decoder setting are historical, not the current balanced
  three-day/free-beam protocol. It was never executed and does not impose a
  renewed 50-minute permission question on the authorized work.
- A controlled history diagnostic on the 1024 model with category history
  features gives known-target joint NLL 13.89967 (1024), 13.90070 (128), and
  13.96519 (empty), on the same targets and query gaps. History affects category
  prediction, but this does not yet show useful extra signal beyond 128 events
  or improved leaf prediction. These are conditional losses, not free retrieval
  metrics. The category feature variant reuses existing category embeddings and
  keeps one real event per history token.
- A simple temporal count control on all 9039 D22 requests of the 512-user
  cohort improves NDCG@100 from 0.00013838 (initial) to 0.00067679 (cumulative)
  and 0.00129493 (recent-only). The 512-request free-ranking panel gives a
  different ordering, so it is too sparse to certify small absolute free-ranking
  gains. This validates a temporal data signal, not a neural release or a chain.

Compact evidence lives under [results/recflow](../../results/recflow/); runnable
entry points are documented in [scripts/recflow](../../scripts/recflow/README.md).
