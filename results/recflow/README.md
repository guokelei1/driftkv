# RecFlow development evidence

This is an independent development track, not the current Yambda motivation
result or an admitted RecFlow version chain. The latest user instruction
authorizes the complete-window development and conditional six-layer progression
described below. The evolving interpretation and next-stage criteria are in
[the working plan](../../docs/recflow/plan.md).

## Completed expanded4096-user background run

Launched2026-09-18 17:47 Asia/Shanghai in tmux **`recflow_6l_u4096_seed17`**,
using GPUs0/1/2/3. The [prospective configuration](../../configs/recflow/window_6l_expanded_u4096_seed17.json)
trains a fresh6L/H192/context1024 seed17 base on1,380,132 eligible D1–18
requests per epoch for3 epochs, then five complete daily1epoch updates
D19–23, each evaluated against its parent on the next day D20–24.
The4096 initial-only selected development users retain the same1M catalog.
Uniform1000 NDCG@50 remains primary; each day has6144 sampled requests and
2253–2353 actual sampled users, with complete full-catalog companion evaluation.

The [focused canary](development/window_6l_expanded_u4096_canary_seed17/canary_check.json)
passed actual four-GPU A→B training with263 requests each and serialized AdamW
0→3→6, LR1e-3, plus tiny paired/random scoring. Its tiny scoring requests are
all OOV and provide execution evidence only; the existing known-positive
resident-evaluation and numerical canaries cover unchanged scoring math.
The complete pipeline finished on2026-09-19 at05:25:50 Asia/Shanghai with
exit status0, after **11hours38minutes45seconds**. All six training phases
and five paired comparisons completed; actual serialized optimizer checks
pass through the final33803 AdamW steps.

The frozen candidate-ranking primary gives:

| Update / future day | Parent NDCG@50 | Current NDCG@50 | Relative improvement | Matched random expectation / null99 |
| --- | ---: | ---: | ---: | ---: |
| A→B / D20 | 0.05892434 | 0.07826652 | +32.83% | 0.00550368 / 0.00647636 |
| B→C / D21 | 0.04505077 | 0.06552911 | +45.46% | 0.00455932 / 0.00541419 |
| C→D / D22 | 0.04398815 | 0.05083416 | +15.56% | 0.00396430 / 0.00479926 |
| D→E / D23 | 0.04352218 | 0.05295646 | +21.68% | 0.00372190 / 0.00451184 |
| E→F / D24 | 0.04414526 | 0.05073539 | +14.93% | 0.00361739 / 0.00443337 |

All five updates improve on their own matched future day, and every
parent/current passes both matched-random checks. The initial A on D19 has
NDCG@50=0.09064209, random expectation0.00745335 and null99=0.00854604,
also passing. Every sampled panel contains6144 requests; all OOV misses remain.
These are one-seed development results for candidate ranking with the
generative model. They do not establish free-catalog generation quality,
RecFlow cache/motivation results, formal release admission or general stability.
All full-catalog companions and historical failures remain in their reports.

The prospective runtime estimate was **13–15hours**. The pipeline retains
the base/random report, all five paired reports, raw metrics, actual optimizer
checks and final checkpoints. Quality regressions are retained and the fixed
descriptive training chain continues; numerical/data/lineage errors stop it.
This does not admit formal releases or establish cache compatibility.

Progress and outputs:

- [Job progress](development/window_6l_expanded_u4096_seed17/progress.json)
- [Cumulative report](development/window_6l_expanded_u4096_seed17/chain_summary.json)
- Pipeline log: `development/window_6l_expanded_u4096_seed17.pipeline.log`
- Per-phase logs: `development/window_6l_expanded_u4096_seed17/A.log` through `F.log`
- Final process status: `development/window_6l_expanded_u4096_seed17/pipeline.exit_status.txt`

The background job has exited; it should not be relaunched. On2026-09-19 the
user confirmed retaining the static initial1M catalog for the next cache-study
stage. Keep the frozen primary and OOV denominator; target coverage and any
known-item conditional diagnostics must be reported separately. Dynamic
catalog expansion and OOV-to-known history recoding are outside this setting.

## Current working candidate-ranking setting

The [completed three-update report](development/window_6l_daily_sampled_seed17/daily_comparison/summary.json)
supports a provisional development setting: **6L/H192/context1024, seed17,
LR1e-3, one complete epoch per day, uniform1000 NDCG@50**. The base fits all
eligible D1–18 data for three complete epochs; updates fit D19, D20 and D21
respectively. Each comparison uses the same next-day768-request panel for
parent/current and retains OOV misses. Rank all known positives plus1000
uniform distractors using the generator's normalized joint probability.

| Update / future day | Parent NDCG@50 | Current NDCG@50 | Relative improvement | Matched random expectation / null99 | Top50 hit requests / users |
| --- | ---: | ---: | ---: | ---: | ---: |
| A→B / D20 | 0.02494742 | 0.02641556 | +5.88% | 0.00549621 / 0.00824515 | 126/96 → 132/101 |
| B→C / D21 | 0.02269247 | 0.02748065 | +21.10% | 0.00442705 / 0.00700875 | 102/77 → 107/80 |
| C→D / D22 | 0.01642532 | 0.02068043 | +25.91% | 0.00372118 / 0.00608937 | 72/57 → 89/69 |

Every parent/current passes both matched-random checks, and all three updates
are positive. The last probe took13.11minutes, both commands exited0 and GPUs
were released. Its saved D has actual AdamW4200/LR1e-3 after all4312 requests
and34 steps; source C and evaluated D hashes match the saved optimizer evidence.
All actual endpoints, raw metrics, full-catalog companions and failures remain.

This is **candidate ranking with a generative model**, not a stable free-catalog
generation claim or formal phi admission. Selection used development outcomes;
D22 was reused after another branch's failure, and three edges/one seed do not
establish long-term stability. The same new C→D has full NDCG@50+22.13%, with
both random checks passing, but the earlier full-catalog gains+420.12%/+608.95%
remain large. Popularity1000 NDCG@100 on D22 improves34.95% while its parent
still fails random; do not turn every positive companion into a pass.

## Learning-rate and NDCG development screen

The [complete comparison](development/window_6l_daily_lr_seed17/lr_comparison/summary.json)
and [metric grid](development/window_6l_daily_lr_seed17/metric_grid/summary.json)
retain all three learning rates, both daily edges, three candidate scopes and
NDCG@20/50/100:54 cells /27 settings. Same trained A,512 users,1M catalog,
context1024,seed17,full one-day epochs, inherited AdamW and four-rank global128.
Each pair uses the same complete future day and includes all OOV misses.
The two new LR branches took41.11minutes in total; all eight training/comparison
commands exited0. The original1e-3 branch is reused as a retained reference.

| Update LR | NDCG@50 D20 / D21 gain | NDCG@100 D20 / D21 gain |
| --- | ---: | ---: |
| 1e-3 | +420.12% / +608.95% | +173.47% / +583.51% |
| 1e-4 | +67.95% / +35.87% | +15.64% / +27.00% |
| 3e-5 | +64.25% / −11.40% | +7.65% / +17.34% |

These are full-catalog FP32 beam300 generation results, not exact search.
Parent/current pass both matched-random checks in every listed cell; the
negative3e-5 NDCG@50 edge remains a failed update. The grid also retains sampled
regressions and all nine popularity1000 settings that fail random checks.
Candidate ranking uses768 requests/day, distinct from the full10171/8866 panel;
those scores cannot be presented as free1M generation.

The selected exploratory setting is **LR1e-4 / daily1epoch / NDCG@100**.
D20 gives0.000147987→0.000171136, D21 gives0.000102460→0.000130128.
Matched random expectations are0.00000913823 and0.00000718162, with null99
0.0000522313 and0.0000501565. Top100 hit requests/users are12/11→12/10 and
7/7→8/8; sparse support limits stability claims. Both parent/current pass
random, but that test does not establish statistical confidence in an update.
The [prospective follow-up](../../configs/recflow/window_6l_daily_confirmation_seed17.json)
freezes this choice before comparing C/D on D22 after one complete D21 epoch.
All outcomes remain, and companion metrics cannot rescue a failed primary.
D22 is a further chronological development check, not an untouched final set.

That [D22 check](development/window_6l_daily_confirmation_seed17/daily_confirmation/summary.json)
**failed**: NDCG@100 0.0000161960→0.0000439407, **+171.31%**, Top100 hits2/2→4/4.
Both parent and current fail the matched-random screen. Actual saved AdamW
steps4166→4200 and LR1e-4 match all4312 requests/34 steps; this is a quality
failure, not an incomplete run. Its sampled uniform1000 NDCG@50 improves4.34%
with both random checks passing, but that companion cannot overturn the frozen
full-catalog failure. The selected full-catalog setting is not a stable chain.

The separately declared1e-3 daily probe explicitly makes uniform1000 NDCG@50
its primary; its completed result is reported above. It is a candidate ranking
task; all full-generation companions and the failed confirmation remain.

The [parameter movement report](development/window_6l_daily_lr_seed17/parameter_drift/summary.json)
checks all194,461,521 unique trainable parameters, excluding buffers, with
checkpoint hashes and FP64 accumulation. A→first daily B relative L2 is7.12%,
0.50%,0.10% at the three LRs. This corroborates a smaller actual parameter
update; it does not establish cache compatibility or quality stability.
The [cleanup record](development/window_6l_daily_lr_seed17/checkpoint_cleanup.json)
removes only the completed263-request low-LR canary weight. Its optimizer/loss
checks, config, raw evaluation and log remain; every actual B/C/D endpoint and
all failed outcomes are retained.

## Completed daily-window development

The latest request first tests shorter updates before expanding training.
The [daily input audit](development/daily_window_panels_seed17/summary.json)
keeps the original512 users,1M catalog and exact initial A training array;
full D20/D21 evaluation is exactly the corresponding old full panel's daily
slices. Fit D19 on6929 known-positive requests, then D20 on5805, each with no
cap. Compare predeclared1-epoch and3-epoch daily branches from the same completed
six-layer A, with real optimizer continuation. Every future request and OOV miss
remains; the separate sampled768-per-day diagnostics have their own denominators.
The [daily runner canary](development/window_6l_daily_canary_seed17/summary.json)
passed263 requests/3 steps from A's AdamW step4065. It checks execution/lineage,
not quality; the prior known-positive resident-evaluation canary still applies.
The [complete daily report](development/window_6l_daily_seed17/daily_comparison/summary.json)
passes the declared two-edge screen for **both** settings. All phases exited0;
the serial four-GPU pipeline took48.77minutes and released its four GPUs.

| Daily setting / future day | Parent NDCG@50 | Current NDCG@50 | Absolute gain | Relative improvement | Top50 hit requests / users |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 epoch / D20 | 0.0000635683 | 0.0003306315 | +0.0002670631 | +420.12% | 4/4 → 11/11 |
| 3 epochs / D20 | 0.0000635683 | 0.0007945708 | +0.0007310025 | +1149.95% | 4/4 → 41/41 |
| 1 epoch / D21 | 0.0001286080 | 0.0009117615 | +0.0007831536 | +608.95% | 8/7 → 50/40 |
| 3 epochs / D21 | 0.0002715188 | 0.0008594383 | +0.0005879195 | +216.53% | 13/13 → 48/40 |

Every Parent/Current is11.29–206.11x its matched random expectation and exceeds
null99; these random-policy draws are not model-seed replications. D20 has10,171
full positive requests, D21 has8,866; all OOV misses remain. Every D19 epoch
visits6929 requests in55 steps; D20 visits5805 in46. The C branches have equal
first-epoch order/target and fixed-NLL-probe hashes. AdamW starts4065 in B,
then4120/4230 in the1/3-epoch C branches; inferred final steps are4166/4368.
The rerun A's D20 raw metric arrays exactly match the original three-day
evaluation's D20 slice. Per-phase logs, settings, raw metrics, random draws,
and all four endpoint weights are retained. The
[checkpoint cleanup record](development/window_6l_daily_seed17/checkpoint_cleanup.json)
lists only unused intermediate weights and the tiny canary weight; their
training/evaluation records remain, and replay of the canary regenerates its weight.

The metric is **NDCG, not AUC**. Daily updates work in this development probe,
but relative gains remain large even at one epoch. Three epochs are not
consistently better: the3-epoch C score is slightly below1-epoch C. Use daily
one-epoch updates as the minimum working development setting; this does not
admit formal phi releases, prove longer-term stability, or establish a
same-calendar daily-versus-three-day advantage. Both branches and all outcomes
remain reported.

The [gain analysis](development/window_6l_full_seed17/update_gain_analysis.json)
and [initial supervision coverage](development/window_6l_full_seed17/initial_supervision_coverage.json)
separate evidence from explanations. Only50.87–53.69% of D19–21 known-positive
label occurrences appear anywhere in the512-user initial positive pool; using
the fixed2048-user initial prefix raises that pool coverage to82.41–84.15%.
This is a full-pool upper bound, not actual three-epoch sampled supervision or
proof of improved future quality. Repeating epochs cannot expand that pool.
Large relative gains can reflect a small baseline and recent-item supervision;
the completed probe does not identify their causal contributions. No expanded
training or RecFlow cache/motivation experiment was launched during this screen.

## Completed six-layer A/B/C result

All six-layer A/B/C phases completed with exit code 0. The
[final report](development/window_6l_full_seed17/ABC_final_comparison/summary.json)
has `working_flags_met=true` for the full-window checks. The last update raises
the D25–27 mean by 37.84%, but **C is worse than B on D27**. This remains one
seed and 512 development users with low absolute NDCG, not formal phi admission,
final-user validation, or evidence that K/V reuse/adaptation works. All four
GPUs have been released.

The completed [2L full-window A/B report](development/window_2l_seed17_full_windows/AB_comparison/summary.json)
passes the declared random checks on all 25,694 positive D22–24 requests:

| Fixed epoch3 model | Free NDCG@50 | Multiple of matched random | Top50 hit requests / users |
| --- | ---: | ---: | ---: |
| A | 0.0000337935 | 9.12x | 10 / 10 |
| B | 0.0003984221 | 107.56x | 63 / 54 |

Both exceed random's 99th percentile 0.0000238634 and twice its expectation
0.00000370413; B>A on each of D22/23/24. This is a traffic-weighted full-window
sensitivity result. The original 3072-request balanced-panel failures below
remain unchanged; neither this single-seed edge nor the report admits a formal
release or establishes a complete version chain.

The completed six-layer setting is
[`window_6l_full_seed17.json`](../../configs/recflow/window_6l_full_seed17.json):
6L/H192/6 heads, unchanged 512 users/1M catalog/context1024/global batch128,
three complete shuffle epochs, fixed training-only NLL each epoch and retrieval
at the epoch3 endpoint. Its [combined manifest](development/full_window_training_seed17/summary.json)
copies the original training arrays and known-request hashes byte for byte,
and uses full future windows with the original 768 diagnostic subsets.
The [four-GPU resource canary](development/ddp_6l_h192_heads6_c1024_k1m_seed17_gpu0123_retry/summary.json)
passed at 337.03 requests/s (330.66/s including proportional CPU preparation),
versus 199.53/s on two GPUs. The four-rank pipeline canary completed 263 requests
including its 7-request tail and passed fixed-NLL/serial 8/4 checks; its all-OOV
free panel supplies no quality evidence. The [resident-rank parallel canary](development/gloo_resident_6l_parallel_eval_canary_seed17/summary.json)
also passed on eight known-positive requests/four diagnostics, with equal
serial/parallel outputs and unchanged optimizer state.

**Six-layer A completed with exit code 0 in 2500.02 seconds (41.67 minutes).** Its
three epochs each covered 173,395 requests/1355 steps, global batch128 on four
GPUs, with request-order and sampled-target hashes identical to the corresponding
original 2L A epochs. Fixed training-only NLL decreased
17.40990→13.24079→12.52489→12.11270. On all 29,027 D19–21 positive requests,
NDCG@50=0.0001196062 versus random 0.00000602561 (19.85x), above
null99=0.0000256533. Top50 hits cover 27 requests/26 users, with 14/4/9 hit
requests by day. The [A final report](development/window_6l_full_seed17/A_final_comparison/summary.json)
records this passing base check and all companion/random results.

**Six-layer B completed with exit code 0 in 1828.98 seconds (30.48 minutes).**
After exact A_epoch3 model/AdamW inheritance, its three epochs each consumed
17,046 requests/134 steps, matching the original 2L `B_retry` request-order and
sampled-target hashes. Fixed training-only NLL after the epochs was
12.07335→10.38078→9.66543. The [A/B final report](development/window_6l_full_seed17/AB_final_comparison/summary.json)
records `working_flags_met=true`: full D22–24 NDCG@50 is
0.0000524797 for A and 0.0004155888 for B, or 14.17x/112.20x matched random.
Both exceed twice expectation and null99, with B>A every day. Top50 hits cover
14 requests/14 users for A and 64 requests/54 users for B; B's daily counts
are 28/20/16. This remains single-seed development, not formal release admission.

**Six-layer C completed with exit code 0 in 1761.35 seconds (29.36 minutes).**
After exact B_epoch3 model/AdamW inheritance, its three D22–24 epochs each
consumed 10,854 requests/85 steps; order/target hashes match the original 2L C.
Fixed training-only NLL was 14.57809→12.13766→9.84625→8.90482. On all 25,180
D25–27 positive requests, B/C NDCG@50=0.0001211737/0.0001670278,
or 35.06x/48.33x random. Both exceed 2x expectation and null99; the complete-window
gain is 37.84%. Top50 hits cover 14 requests/13 users for B and 32 requests/28
users for C, with C's daily hit requests 13/12/7.

**C improves D25/26 but regresses on D27:** NDCG@50 is 0.0000947085 for C
versus 0.0001893917 for B. The predeclared complete three-day, traffic-weighted
mean passes; this is not evidence of improvement on every day or generally
stable updates. The report retains that negative day and all companion metrics.

The final model/optimizer checkpoints are:

- `development/window_6l_full_seed17/A/A_epoch3/checkpoint.pt`
- `development/window_6l_full_seed17/B/B_epoch3/checkpoint.pt`
- `development/window_6l_full_seed17/C/C_epoch3/checkpoint.pt`

The [cleanup manifest](development/window_6l_full_seed17/checkpoint_cleanup.json)
records seven deleted payloads: six unevaluated epoch1/2 checkpoints and the
completed tiny-canary weights, releasing 15.37 GiB. Final A3/B3/C3 model and
optimizer states remain, including C's saved AdamW step 4722. Source, results,
raw metrics/rankings, logs and failures are retained; direct canary replay
requires regenerating its three-step checkpoint.

The chronological 2L variant completed A/B but still fails its old-parent
random-tail check, so that variant does not proceed to C.

## Completed seed17 2L development

The completed setting used [`window_development_seed17.json`](../../configs/recflow/window_development_seed17.json)
and `scripts/recflow/window_chain.py`: 2L/H96, context1024, fixed 512 development
users, 1M initial catalog, batch128 and learning rate 0.001. The user explicitly
authorized parameter iteration, optional C and progression to 6L after the small
protocol is stable. Do not repeat the old 50–60-minute launch question. A/B
canaries completed 263/263 requests including the last 7-request batch, retained
matching fixed training-probe hashes and passed real optimizer inheritance and
sampled-panel random-analysis checks. Main A completed in tmux
`recflow_window_A_seed17`, writing `development/window_2l_seed17/A`,
`development/window_2l_seed17/A.log` and
`development/window_2l_seed17/A.exit_status.txt` (zero). B completed in `B_retry`.
Their original balanced-panel gate fails on the old A's D22–24 random comparison.
The full-window check above provides separate evidence for the next development
step. Jobs exceeding30 minutes retain detached tmux logs and exit status.

| Phase | Complete fitting window / known-positive requests per epoch | Fixed future comparison |
| --- | --- | --- |
| A | D1–18 / 173,395 | A and random on D19–21 |
| B, from A | D19–21 / 17,046 | A, B and random on D22–24 |
| Optional C, from B | D22–24 / 10,854 | B, C and random on D25–27 |

The new budget is three complete epochs per phase, without a 150k request cap or
timed endpoint. Retrieval is recorded at epochs 1 and 3; 2048 fixed training-only
requests/targets measure NLL before fitting and after every epoch. Epoch3 is the
predeclared candidate; epoch1 is a learning-curve companion, not an invitation
to choose the best future-label checkpoint. Three epochs do not prove convergence.

The primary is **free-catalog FP32 beam300 NDCG@50 versus matched random**.
Each original frozen three-day panel has3072 positive requests,1024 per day; a separately
hashed 768-request subset, 256 per day, supports uniform/popularity1000 diagnostics.
Sampled failure does not automatically fail the full-catalog primary, and sampled
success cannot replace it. Keep NDCG@20/@100, Recall, per-day results, all OOV
misses and both parent/current scores. This remains development-only, with no
admission/final users or formal release qualification.

At the measured 353.59 requests/s, an A epoch is about 8.17 minutes. Initial total
estimates, including evaluations, are A 35–45 minutes, B 18–28 and optional C 17–27;
the combined-path canary checks these estimates before main execution.

All three A epochs completed173,395 eligible requests each in about406s;
the fixed training-only NLL decreased17.407→13.147→12.435→12.084. A separate matched-input GPU2/3 DDP
canary measured 437.28/s single-GPU and 788.07/s two-GPU (1.80x), or 754.31/s
including proportional CPU preparation, with unchanged global batch128.
Unequal-tail FP32 gradient and AdamW reference comparisons passed. See
[`ddp_2l_h96_c1024_k1m_seed17_gpu23/summary.json`](development/ddp_2l_h96_c1024_k1m_seed17_gpu23/summary.json).
These are resource/correctness results, not additional training seeds.

The first B attempt in `development/window_2l_seed17/B` stopped before any
optimizer update: an NCCL waiting rank blocked a separate evaluation worker on
its GPU. The attempt, nonzero status and `execution_interruption.json` remain.
After replacing main-process waits with CPU Gloo, the real-checkpoint resident
DDP/evaluation canary passed (`development/gloo_resident_parallel_eval_canary_seed17`).
The unchanged B experiment completed in `development/window_2l_seed17/B_retry`.
Tiny DDP A/B canary weights were removed after verified optimizer continuation;
each canary's `checkpoint_cleanup.json` records exact paths, bytes and hashes.
Their summaries, metrics and validation evidence remain; main A weights remain.

Completed original3072-panel comparisons (`development/window_2l_seed17/AB_final_comparison`
and `ABC_final_comparison`):

| Model / future panel | Free NDCG@50 | Random expectation | Hit requests / users | Random working check |
| --- | ---: | ---: | ---: | --- |
| A / D19–21 | 0.00035037 | 0.000005971 | 5 / 5 | Pass |
| A / D22–24 | 0.00002275 | 0.000003879 | 1 / 1 | Fails99th-percentile check |
| B / D22–24 | 0.00077541 | 0.000003879 | 14 / 13 | Pass |
| B / D25–27 | 0.00001966 | 0.000003442 | 1 / 1 | Fails99th-percentile check |
| C / D25–27 | 0.00023507 | 0.000003442 | 4 / 4 | Pass |

B improves over A on each of D22/23/24, but that does not repair the old-parent
random failure. C improves D25/26 and ties D27 at zero; old B has the same
sparse-parent issue. The complete report is `window_2l_seed17/ABC_final_comparison`.
Keep all other cutoffs and sampled diagnostics in the report.
Original C has completed. Independently, a fresh seed17 A/B on GPUs0/1 used
`window_chronological_seed17.json`: chronological three-day training blocks,
shuffled inside each block, with the same complete epochs and evaluation.
It changed training order only; the shuffle baseline remains intact. Both
chronological phases have completed, as reported below.

An additional fixed-checkpoint full-window sensitivity evaluation uses
`development/full_window_panels_seed17`: all29027/25694/25180 positive requests
on the same three windows, with the same users/catalog/OOV treatment and the
unchanged768-request diagnostic subsets. This is a traffic-weighted mean,
whereas the original3072 panel equally weights1024 requests per day. Its CPU
random analysis (`full_window_random_power_seed17`) narrows the null tail but
does not guarantee a model pass. Original A/B epoch3 are evaluated on the same
full D22–24 requests under `window_2l_seed17_full_windows`; the original failure
remains reported and cannot be renamed a balanced-panel pass.

Original A/B's full D22–24 comparison is complete: A=0.0000337935 and
B=0.0003984221,9.12x/107.56x the same random expectation, both above
null99=0.0000238634. A's10 hit requests/10 users are distributed2/6/2 across
D22/23/24; B's63 hit requests/54 users are distributed31/23/9. B improves
each day's NDCG@50. The [matched report](development/window_2l_seed17_full_windows/AB_comparison/summary.json)
retains both endpoints, all cutoffs, both sampled diagnostics and random
comparisons. The original3072 panel continues to fail its own check.

Chronological A/B also completed. A on D19–21 gives NDCG@50=0.000500554,
11 hits/10 users. On D22–24, A=0.0000368012 (one hit,9.49x random,
below null99) and B=0.0009342337 (12 hits/11 users,240.86x random).
B improves all days, but the old-parent random-tail failure remains, so no
chronological C is planned. Its [final report](development/window_chronological_2l_seed17/AB_final_comparison/summary.json)
is retained alongside the shuffle/full-window result. No six-layer quality
result follows from either2L experiment.

## Retained evidence

The earlier six-layer pilot results remain historical evidence: both updated
models clearly beat random on the 9039-request panel, while their initial parents
failed its working screen. Earlier popularity-sampled failures describe that
diagnostic, not an automatic veto on the newly frozen free-catalog primary.
The old 50–60-minute one-pass JSON is marked `superseded_before_launch`; it was
never run. No earlier result qualifies the new complete-window chain in advance.

- `development/seed17_complete_epoch_windows/summary.json` and local
  `window_panels.npz`: complete fitting pools and frozen 3072/768 three-day panels,
  counts, cohort/catalog/panel hashes and causal checks. The training arrays keep
  every positive request; the known-item objective explicitly filters all-OOV
  requests and reports that coverage.
- `development/window_2l_seed17/`: completed seed17 2L A/B/C execution;
  phase summaries, completed-epoch training/NLL records and separate future
  evaluation directories retain outcomes, including the balanced-panel
  parent failures. Each update inherited the saved predecessor and optimizer.
- `development/window_2l_seed17_full_windows/AB_comparison/summary.json`:
  completed matched epoch3 A/B check on all25,694 D22–24 positive requests;
  both pass random and B improves every day, without replacing original failures.
- `development/window_chronological_2l_seed17/AB_final_comparison/summary.json`:
  completed chronological-block ordering comparison; old A still fails the
  D22–24 random-tail check and this variant stops before C.
- `development/full_window_training_seed17/summary.json`: combined input with
  all nine arrays byte-identical to their original training/full-evaluation
  sources and independently recomputed, unchanged known-training hashes.

- `preparation/summary.json`: complete realshow request/causality and coverage
  audit. Exposure and relevance statistics are data audits, not model quality.
- `development/catalog_coverage/summary.json`: 200k/1M/full initial catalogues
  on the same fixed development cohort and time windows.
- `development/resource_*/`: short architecture, memory and throughput probes;
  near-zero retrieval quality here is not a trained-model conclusion.
- `development/grid_6l_h192_c128_k200k_seed17/`: first bounded learning grid.
  Retain its original outcomes. **Its sampled metrics are superseded** because
  tied scores preferred injected positives; free beam outputs are unaffected
  by that ranking bug but are approximate search.
- `development/sampling_fixed_c128_k200k_seed17/`: corrected sampling comparison
  on 512 stable requests, also using the corrected rolling time-delta input.
  This is not a like-for-like quality delta against the original 1024 panel.
- `development/decode_*`: label-independent search accuracy and compute checks.
  The FP32 bound-pruned result agrees with exhaustive ranking on its test panel;
  BF16 branch batch shapes can change close scores.
- `development/tree_cardinality/`: unbalanced taxonomy initialization diagnostic;
  no new prior variant or learned-quality claim follows from this calculation.
- `development/history_*`: complete/short/empty history conditional-loss checks;
  these retain fixed targets and query gaps but do not measure free generation.
- `development/popularity_temporal_k1m/`: causal initial/cumulative/recent count
  controls. Both the noisy 512-panel result and the complete 9039-request result
  are retained. This is evidence of temporal data signal, not neural quality.
- `development/pilot_*`: bounded neural Parent/Current probes on identical D22
  development requests. These do not perform model admission or final-user
  evaluation; a single training seed does not establish a stable version chain.
- `development/population_eval_canary_seed17/`: eight-request exact FP32 plus
  uniform/popularity1000 evaluation resource canary on a 1M/context1024 checkpoint.
  This verifies the execution combination, not random superiority or admission.
- `development/random_baseline_seed17/summary.json`: analytic matched random
  expectations and 5000 fixed-panel random-policy trials, seed 20260918. Reports
  every NDCG/Recall@20/50/100 cell, including failures, null means/quantiles,
  model/random ratios and the separate `>null_p99` and `>=2x expectation` flags.
  Local `null_trials_*.npz` files retain complete trial metric arrays. The old
  positive-priority sampled results are rejected by the analysis entry point.
- `development/random_baseline_three_seeds/summary.json`: the same calculation
  for ordinary seeds17/23/29, retaining the history-category seed17 and corrected
  200k diagnostics too. Identical panels reuse identical 5000-trial null arrays;
  each learned seed is reported separately, not replaced by a favorable seed.
- `development/free_fullpanel_{parent,current}_seed{23,29}/`: completed FP32
  beam300 evaluation, not exact search, of the two complete stored checkpoint
  pairs on the identical 9039-request D22 panel. No new training was performed.
- `development/random_baseline_fullpanel/summary.json`: matched analytic random
  expectations and 5000-trial null distributions for those larger-panel results.
- `development/free_fullpanel_comparison/summary.json`: compact paired results,
  shared panel hashes, hit request/user counts and a same 512-request BF16/FP32
  comparison. No absent seed17 parent is reconstructed or counted as a third pair.
- `development/supervision_coverage/summary.json`: CPU reconstruction of actual
  seed17 request/positive sampling, checked against stored counts and independent
  selection hashes. Positive identities rely on recorded RNG/seed reconstruction;
  there is no original draw-level log for independent comparison.
- `development/population_budget/summary.json`: fixed 4569-development-user
  training-request counts and one-pass cost calculations. These are resource
  forecasts, not measured population training or launch authorization; they no
  longer describe the next proposed experiment.

## Historical 512-request random comparison

On D19, the ordinary seed17 base's popularity1000 NDCG@50 is 0.007678 against
random expectation 0.007985 (0.96x); NDCG@100 is 0.012577 against 0.012963 (0.97x).
That proposed popularity diagnostic did not pass; it is not the active
free-catalog primary gate. Its uniform1000 NDCG@50/100 instead
gives 2.73x/2.22x and passes the working screen. The criterion is not merely
that an update improves on a weak parent or one easier pool passes.

The table reports all ordinary seeds on the identical 512-request D22 panel.
Each cell is **trained NDCG (times matched random expectation)**. Random
expectations are 0.00398133 at 50 and 0.00646344 at 100 for both injected-positive
candidate samplers; their random-policy 99th percentiles are 0.00707920/0.00972934.

| Seed/model | Uniform1000 @50 | Uniform1000 @100 | Popularity1000 @50 | Popularity1000 @100 |
| --- | ---: | ---: | ---: | ---: |
| 17 parent | 0.008916 (2.24x) | 0.012482 (1.93x) | 0.002726 (0.68x) | 0.005154 (0.80x) |
| 17 current | 0.021628 (5.43x) | 0.026359 (4.08x) | 0.012016 (3.02x) | 0.014742 (2.28x) |
| 23 parent | 0.011910 (2.99x) | 0.013103 (2.03x) | 0.005088 (1.28x) | 0.008337 (1.29x) |
| 23 current | 0.020194 (5.07x) | 0.024189 (3.74x) | 0.009962 (2.50x) | 0.013044 (2.02x) |
| 29 parent | 0.009693 (2.43x) | 0.012759 (1.97x) | 0.004257 (1.07x) | 0.006143 (0.95x) |
| 29 current | 0.018671 (4.69x) | 0.023175 (3.59x) | 0.011569 (2.91x) | 0.014141 (2.19x) |

All current @50/@100 cells in this table pass both working checks; every
popularity parent cell fails. This supports a sampled-ranking update signal
across the frozen seeds, not qualification of the base. The separate larger-panel
free-generation result is reported below.

In the seed17 grid, the ordinary D19 parent passes uniform1000 NDCG@20/50/100
but fails every popularity1000 cell. Ordinary and history-category D22 parents
pass uniform1000 NDCG@20/50, fail NDCG@100's 2x check, and fail every
popularity1000 cell. Both current models pass uniform/popularity1000
NDCG@20/50/100; their popularity Recall@100 still fails the 2x check. None of
the original 512-request ordinary seeds17/23/29 or history-category seed17
free-catalog beam results passes both checks. The corrected 200k result
passes uniform1000 NDCG@20/50 only; other pools and all its Recall cells fail
the combined check. The JSON retains exact values for every cell.

The supervision audit limits what can be inferred from the failed parent:
58,784 requests were processed out of 136,928 eligible selected requests, yielding
53,179 distinct directly sampled positive videos. On the fixed D22 panel,
16.01% of known positive occurrences were seen as selected parent targets;
50.45% existed in the entire initial request pool. Parent/current selected
positives together cover 24.77%. These are direct positive-supervision counts,
not claims that remaining embedding rows received no history or softmax gradient.
They motivate investigating training sufficiency but do not show that a longer
run will pass random, improve free retrieval, or produce a valid version chain.

The two numerical flags are a working development screen for the user's request
to clearly outperform random, not numerical thresholds stated by the user or
automatic admission rules. Failure of a 2x screen does not necessarily mean a
value is below the random mean. Conversely, current D22 free-beam NDCG@100 is
about 23x random expectation but below the random 99th percentile; sparse hits
make the ratio alone misleading.

The random policy permutes available video IDs uniformly, with OOV positives
retained in denominators. For sampled pools, every known positive is injected
and the actual pool size is `min(catalog_size,m+distractors)`. Therefore uniform
and popularity pools have the same random expectation conditional on known
positive count and pool size, while their difficulty for the learned model
differs. Selecting uniform sampling or a favorable cutoff solely because it
passes would change the claim after observing results. Free generation and
sampled ranking remain distinct tasks; all outcomes remain visible.

Random-policy intervals on a fixed request panel are **not** uncertainty across
training seeds, user populations or future windows. The script records input
and source hashes, deterministic seeds and shared trial-array identities;
replaying the same panel reproduces the arrays exactly. Use
`scripts/recflow/random_baseline.py --runs ... --output ... --draws 5000 --seed 20260918`.

## Historical completed 9039-request free-generation check

All four checkpoints use the identical D22 panel: 9039 positive requests,
437 active users from the fixed 512-user cohort, 23.7348% target coverage and
5025 all-OOV requests retained at zero. Evaluation uses **FP32 free beam300,
not exact ranking** and took 997–1011 seconds per checkpoint. There was no new
model training. At NDCG@50, the matched random expectation is 0.00000392922 and
its fixed-panel random-policy 99th percentile is 0.0000429740.

| Seed/model | NDCG@50 | Times random expectation | Top100 hit requests / users | Both random checks |
| --- | ---: | ---: | ---: | --- |
| 23 parent | 0.0000256914 | 6.54x | 3 / 3 | fail |
| 23 current | 0.000668752 | 170.20x | 48 / 45 | pass |
| 29 parent | 0.00000781894 | 1.99x | 2 / 2 | fail |
| 29 current | 0.000239905 | 61.06x | 22 / 20 | pass |

Both current models pass both checks at every reported NDCG/Recall@20/50/100;
both parents fail the random 99th-percentile check at every cutoff. The result
shows neural free-generation learning and a future-window update signal in two
stored model pairs. Absolute quality remains low, initial base qualification
still fails, and a complete version chain is not established.

The two seed23/29 pairs were selected by checkpoint completeness before their
new outcomes were inspected. Seed17 predates parent-checkpoint saving, so its
retained 128/512-request evidence remains separate; no missing asset is silently
recreated and no three-seed paired claim is made. On the original 512 requests,
both current FP32 scores are still all zero, as in the old BF16 evaluation;
seed23 parent's one-hit NDCG@50 changes slightly from 0.000160353 to 0.000158603.
This subset check confirms that the newly observed current hits are not merely
caused by changing precision. The paired summary retains that comparison rather
than attributing every difference between the old and new evaluations to size.

The earlier `configs/recflow/medium_development_one_pass.json` is superseded
before launch. Its 136,928 eligible initial requests came from the old 150k cap,
whereas the active complete D1–18 window has 173,395. Its single-day evaluation,
exact decoder and 50–60-minute permission question are no longer the next-step
protocol. The latest user authorization applies to the active small-backbone
window experiments and, after they work, the six-layer progression described at
the top. Historical failures remain visible; no longer-run success is assumed.

Only compact summaries/configurations are tracked. Prepared data, weights,
per-request arrays/rankings and runtime logs remain local and ignored by Git.
