# CohortKV Stage 4.7 organic lifecycle v1

## Status

Implementation and the one predeclared single-configuration run are complete under
`cohortkv_single_config_organic_lifecycle_v1`. The execution/cost/task-output contracts pass, but
the minimum cache-fidelity gate does not. This is therefore a completed mixed result, not a fully
passing lifecycle certificate and not permission to retune the policy on the observed final
labels.

This protocol corrects the fixed-history boundary of Stage 4.6. Stage 4.6 remains a controlled
same-input accumulation diagnostic; none of its numeric results are inherited by this experiment.

## Frozen scope

- KuaiRand-1K 4+12;
- training seed 0;
- 16 layers, hidden/K/V width 512, maximum history 2,048;
- one A40;
- 682 users selected deterministically from the 945-user base-only prepared cohort without future
  activity;
- one `theta0 -> theta1 -> ... -> theta11` run;
- no dataset, model-size, seed, GPU-count, storage-backend, or policy matrix.

The base-only cohort is split before reading an online window into 40 fit, 60 program-selection,
60 certificate, and 522 final-test records. The program-selection role is retained for role
compatibility but does not tune the policy in this experiment.

## Causal timeline

The twelve task endpoints are:

| Model | Available history | Unseen target window |
|---|---|---|
| theta0 | through 2022-04-11 | 2022-04-12 |
| theta1 | through 2022-04-12 | 2022-04-13 |
| theta2 | through 2022-04-13 | 2022-04-14 |
| theta3 | through 2022-04-14 | 2022-04-15 |
| theta4 | through 2022-04-15 | 2022-04-16 |
| theta5 | through 2022-04-16 | 2022-04-17 |
| theta6 | through 2022-04-17 | 2022-04-18 |
| theta7 | through 2022-04-18 | 2022-04-19 |
| theta8 | through 2022-04-19 | 2022-04-20 |
| theta9 | through 2022-04-20 | 2022-04-21 |
| theta10 | through 2022-04-21 | 2022-04-22 |
| theta11 | through 2022-04-22 | 2022-04-23 |

At endpoint `v`, both mixed and all-exact use the same canonical history, latest token, current
checkpoint, candidate catalog, and next-window engaged positives. The target window is evaluated
before it is ingested. After it occurs, its complete logged exposure sequence may be appended to
the cache and used to train the next checkpoint.

The causality gate follows the dataset's canonical date partitions rather than treating raw
`time_ms` as a globally strict day boundary. For each user, `known` starts as the resident
window-zero history. Before a target partition is admitted, its current resident history-event
identities must be an exact suffix of `known`. Only after that check passes are the current target
partition's event identities appended to `known`. Thus rolling-window expiration and token-cap
tail truncation are allowed, while inserting any identity from the current or a future partition
into history fails the run. Target-partition timestamps at or after that partition's raw-time
minimum remain an internal consistency check. The result field
`request_events_not_before_request_start` is a legacy key for this narrower check; it is not a
claim that `as_of_timestamp_ms` is a real request-arrival timestamp.

Raw history timestamps at or after the target partition's raw-time minimum are a non-gating
diagnostic because KuaiRand date partitions and `time_ms` have a small measured boundary overlap.
The diagnostic reports
resident record-windows, history tokens, active/inactive overlap counts, maximum lead, fractions,
and the complete per-version breakdown. On the frozen 682-user reconstruction, 147 of 8,167
resident record-windows and 3,521 of 11,797,055 history tokens overlap this raw boundary
(0.029846% of tokens), with a maximum lead of 6,128.618 seconds. These observations are recorded;
they are not represented as zero and do not replace the date-partition identity gate.

## Recursive cache transition

For each `theta_v -> theta_(v+1)` edge:

1. consume the previous endpoint's actual mixed full cache;
2. align it to the next canonical history by stable event identity;
3. for a positive overlap, remove the expired left prefix and append only the newly observed
   `H_(v+1)[:-1]` events with `theta_v`;
4. execute one lightweight migration candidate for every reusable continued prefix;
5. route that candidate to either publication or exact `theta_(v+1)` prefix replay;
6. send cold, re-entered, and zero-overlap prefixes directly to exact `theta_(v+1)` replay without
   rebuilding or probing a source-model cache;
7. treat a length-one history as an empty prefix followed only by the common latest-token path;
8. execute the latest token under `theta_(v+1)`, score the next unseen window, and retain the
   returned full mixed cache for the next edge.

An overlap of zero is a no-reuse transition and therefore receives natural current-model exact
prefix replay. A partial left crop does not by itself trigger full recomputation. Deep retained
K/V may still encode evicted context; this is a measured approximation against exact replay, not
an equivalence claim.

The exact reference is always `F(theta_(v+1), H_(v+1)[:-1])`. Regenerating a source-model cache
from raw history when a real prior mixed overlap exists is prohibited. Creating one for a
cold, re-entered, or zero-overlap record is also prohibited.

## Compiler and policy

Each adjacent direct program is fitted on the 40 fit users using only the history available after
the just-observed update window. It uses no next-window recommendation label. Early programs
cannot use the final-window history distribution.

The policy is predeclared without a search:

- exact fraction 20% of the reusable continued prefixes on every edge, subject to nearest-record
  rounding;
- maximum consecutive migration depth four;
- execute the lightweight candidate for every reusable continued prefix and compute its label-free
  per-cache q90 absolute log K/V norm shift;
- make depth-four records mandatory exact, then fill the remaining exact budget by greater
  migration age and greater current-edge norm shift;
- use stable SHA256 only for an exact numeric tie;
- no future-edge severity rank and no recommendation label in routing.

Natural exact work for cold, re-entered, and zero-overlap prefixes is outside the 20% selector
budget and is reported separately. Consequently the total exact fraction among resident records
may exceed 20%. Length-one histories use only the common latest-token path and do not enter either
prefix denominator or selector budget.

The 15%-25% severity-ranked Stage 4.6 schedule is not used because its early fractions were chosen
after observing all eleven fixed-history edges. The earlier free-running norm-sketch threshold is
also not restored: ranking operates inside a fixed budget, so the signal cannot recreate its
0%-65% refresh waves. Candidate work discarded by an exact decision remains charged.

## Measurements

Every task endpoint reports mixed and all-exact MeanRank, catalog AUC, NDCG@100, Hit@100, and
paired differences on all final-test records with next-window engaged positives. Reuse-only task
quality and recovery are separate diagnostics on the reusable continued subset. Natural exact
records do not fabricate reuse-equals-exact observations. Exact-relative ratios are diagnostic;
full recomputation is not assumed to be a ranking-quality upper bound.

Every update reports cache q90 fidelity, hidden and score cosine, top-100 overlap, exact fraction,
migration depth, resident/expired/re-entered records, retained/evicted/appended tokens, and complete
per-record lineage.

GPU time is separated into:

- foreground append and eviction/repack;
- candidate migration and router probe;
- selector-scheduled exact replay;
- natural no-reuse target exact replay;
- publication;
- common latest-token execution;
- an independent all-exact prefix reference.

The primary ratio is mixed update divided by the all-exact prefix reference. Mixed update includes
candidate migration, router probe, selector-scheduled exact replay, natural no-reuse target exact
replay, and mixed-prefix publication. A symmetric lifecycle ratio adds the identical foreground
work to both numerator and denominator. A common-inclusive diagnostic adds latest-token execution
and its publication to both sides; the primary and symmetric ratios exclude this common work.

The denominator performs exactly one target-model `H[:-1]` exact batch, of at most four records,
for every nonempty fixed group. It includes every resident record in that group with history length
at least two, regardless of whether the mixed path can reuse its prior cache. Natural exact
numerator work executes separately and is not physically shared with this reference. Offline
reference row selection and quality scoring are excluded from both sides. Compiler fit/compile,
recommendation scoring, scheduler CPU time, and host-to-device sequence construction are
excluded and are not independently timed in this v1 artifact. Consequently even the
common-inclusive ratio is a measured GPU lifecycle boundary, not a complete application
end-to-end claim.

## Development gates

- causality, role isolation, history identity, recursive input, and lineage coverage: 100%;
- selector exact fraction: nearest-record 20% of reusable continued prefixes on every edge;
- maximum migration depth: four;
- minimum per-step cache fidelity: 0.90;
- minimum per-step score cosine: 0.995;
- minimum per-step top-100 overlap: 0.95;
- cumulative update-only GPU ratio: at most 0.30;
- maximum per-step update-only GPU ratio: at most 0.35.

These gates classify the one run; they do not authorize changing its actions after task labels are
observed.

## Measured result

The frozen output is
`results/system/cohortkv_single_config_full_chain_v1/stage4_7_organic_full_chain_seed0.json`.
The versionable result and implementation-hash index is
`configs/cohortkv_single_config_v1/stage4_7_organic_summary.json`; it is required because the
runtime JSON's recorded base commit predates the uncommitted Stage-4.7 source files.
It contains 12 task endpoints, 11 recursive updates, 8,184 reconstructed user-windows, and 7,502
update-lineage rows. Every causality, compiler-provenance, history-hash, adjacent-version,
previous-actual-consumption,
label-isolation, candidate-coverage, depth, and per-step execution check is true.

### Routing and action balance

Across the 11 updates, 6,711 reusable continued prefixes entered the selector. The policy chose
1,344 exact refreshes (`20.0268%`) and 5,367 migrations. Another 771 cold, re-entered, or
zero-overlap prefixes required natural exact replay, and three length-one histories used only the
common latest-token path. Thus the approximately 20% contract applies only to reusable prefixes;
the complete resident action stream is approximately 28.3% exact.

The exact set was not sampled randomly. Of the 1,344 selector refreshes, 476 were mandatory
depth-four resets and 868 filled the remaining age-then-current-norm-shift quota. All eleven edges
had distinct norm-shift values at the cutoff and SHA256 was never used. The posthoc diagnostic is
nevertheless weak rather than a selector-optimality result: the mean per-edge Spearman correlation
between current norm shift and realized candidate error is `0.0341`
(`-0.1358` to `0.1782`), selected candidates have higher mean error than migrated candidates on
8/11 edges, and mean overlap with the exact-error top-20% oracle is `23.46%` versus a 20% random
expectation. The defensible mechanism is a bounded age/deadline scheduler with a weak label-free
secondary ranking signal, not a learned or calibrated per-cache failure predictor.

### Cost, fidelity, and task output

| View | Measured | Gate | Result |
|---|---:|---:|---|
| cumulative update-only GPU / all-exact prefix | 0.2703 | at most 0.30 | pass |
| maximum per-step update-only GPU / all-exact prefix | 0.2892 | at most 0.35 | pass |
| symmetric lifecycle ratio | 0.5069 | diagnostic | — |
| common-inclusive lifecycle ratio | 0.5372 | diagnostic | — |
| minimum q90 cache fidelity | 0.8744 | at least 0.90 | **fail** |
| minimum score cosine | 0.999876 | at least 0.995 | pass |
| minimum top-100 overlap | 0.97357 | at least 0.95 | pass |

The 11 evaluated target windows contain 4,368 final-role positive records. Record-weighted
mixed/all-exact ratios are `0.999987` for catalog AUC, `0.994590` for NDCG@100, and `0.997180`
for Hit@100. The worst per-window ratios are `0.999786`, `0.953324`, and `0.977778`,
respectively. Maximum absolute mixed-minus-exact gaps are 14.473 MeanRank, 0.0002894 catalog AUC,
0.0005353 NDCG@100, and 0.0028249 Hit@100. These are paired single-seed development outcomes;
full recomputation is not assumed to be the upper bound of a ranking metric.

The largest CUDA allocated-memory peak recorded by the runner is 35.824 GiB, including models and
batch temporaries, while the largest logical published K/V state is 31.892 GiB. A concurrent
`nvidia-smi` sample reached 44.755 GB of device-visible allocation/reservation. The latter was not
captured in the result schema and is a diagnostic warning: one A40 runs the chain, but the
allocator margin is not yet a comfortable deployment bound.

### Classification and limitations

The run establishes a real growing-history, mixed exact/migrate chain under canonical KuaiRand
date-partition order. It does not establish strict raw-event/request-time causality. The 147
raw-boundary overlaps remain explicit, and local timestamp inversions mean the frozen
`searchsorted` crop differs from a stable boolean suffix filter in 4 of 8,184 selected
user-windows. Repairing that prepared-data boundary would change training histories, checkpoint
hashes, and compiler inputs, so it requires a new protocol and retraining rather than a silent
patch to this result.

The execution, cost, maximum-depth, score, top-100, and recommendation-output targets pass. The
predeclared 0.90 cache-fidelity target fails on the late recursive edges, whose worst value is
0.8744. Stage 4.7 is therefore complete as evidence but not fully passed. Any next attempt to
improve the recursive operator, reset budget, or ranking signal must use a new protocol and must
not erase this result.
