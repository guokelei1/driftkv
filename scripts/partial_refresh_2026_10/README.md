# Chapter 4.4 partial refresh

Isolated research prototype. Existing datasets, models, calibrations and results
are read-only. All new outputs live in `results/partial_refresh_2026_10/`.

The user reopened exploration on2026-10-14:4.4 refresh/maintenance FLOPs must
save30%–50% and the quality-retention interval lower bound must exceed90%.
Length grouping is an example, not a required restriction. New alternatives are
in `round2/`; their completed evidence is in the
[second-round index](../../results/partial_refresh_2026_10/round2/README.md).

Second-round selected computation paths meet that target: same encoder function
with batch-local item reuse saves35.10% maintenance FLOPs, or a fixed4k-item
table saves49.94% (4 independently prepared deployment replicas). Both retain
about100% full-code gain on3360users/69,361requests. Main savings come from
factoring encoder arithmetic, while these two paths still refresh all required
rows. The third tested point combines static16k with the original strict partial
policy (61.13% savings, retention96.04% with lower95%bound93.35%).
These are fixed-model function/efficiency cases; pure-new-user method regression
and low item-repeat cost boundaries remain in
[the report](../../results/partial_refresh_2026_10/round2/report.md).

`round2/factored_encoder.py` folds fixed center/scale and shares only the
Current-item affine term. `factored_session.py` charges constructor preparation,
every unique miss and rowKV operation; user KV/codes stay unchanged.
`adaptive_session.py` supplies explored first-read length and compact-code
priority alternatives; their adverse outcomes are retained.
`prepare_item_tables.py` freezes fit-only prerelease IDs; `factored_cost.py`
replays actual fixed30k encode calls and `verify_factored_cost.py` verifies
execution counters. `round2/aggregate.py` checks fixed settings, actual ratios,
coverage and all original controls; `round2/make_report.py` reads completed data.

The static-cache offline prototype retains3release encoder objects, including
all3tables. Shared-table retirement is not implemented; one-current-release
table size is a logical serving working set, not the observed complete peak.

- `session.py`: actual retained codes, independent int32 encoder versions,
  tail refresh union missing/older rows, native FIFO and causal retirement.
- `run.py validate`: fit-only error scale and publication-time validation of
  every candidate rho, then causal replay of the selected policy.
- `run.py evaluate`: continuous quality replay against retained matched
  Reuse/full-code/once-Rebuild scores; never selects rho from evaluation labels.
- `analyze.py`: equal-release gains, partial/full preservation, paired UID
  intervals and separately defined user cohorts.
- `cost.py`: exact analytical metadata replay and fixed30k complete cost.
- `aggregate.py`: merge completed disjoint runs; check complete coverage and
  input-identical prefixes before reusing their scores.
- `prepare_panel.py`: pure-new-user panel with all earlier roles excluded.
- `prepare_benchmark_panel.py`: hash-ordered alternative benchmark population;
  final3360 includes every eligible user, without quality/activity filtering.
- `make_report.py`: assemble retained positive/negative results and cost bounds.

Initial configuration: frozen Medium V2--V5, 6 layers/6 heads/32 head width,
rank8 full-item local32/joint16 support-masked parameters. Candidates
`[0, .125, .25, .5, .75, 1]`; E(rho) is complete-query score MSE against full
current codes. Main epsilon is 1% of full-code-minus-Reuse score energy on
fit users. Ratios are evaluated independently, without assuming monotonicity.
The original calibration roles stay separate from all evaluation roles.

Pending native appends are marked missing and encoded at the first read;
active appends are encoded immediately, matching the existing lazy policy.
Each row's version becomes current only after all its layers/heads are encoded.
V3 has no initial old codes and necessarily performs full initialization,
regardless of the nominal chosen rho. V4/V5 expose actual code reuse.

Quality4096 is an existing development panel, not independent confirmation.
Predeclared cohorts: initial history length <=256 / 257--768 / >768 and
activity in the 72h preceding the first release (zero / positive). Cohorts
describe boundaries; every predetermined release remains in the main result.

Cost includes encoder FLOPs (166464 per row), support masking (288 per row),
native-query-relative added reads, appended codes, forced refreshes and new
shared publication validation. Version tags add 4 bytes per native history row.
Offline full-reference encodings are validation cost, never credited as free
serving preparation. No wall-time saving or loaded-serving claim is made.

Runs are launched only after inspecting shared GPU occupancy; only owned
processes are managed. CPU numerical checks precede GPU canaries. New settings
and conclusions are retained even when an attempted policy fails.

## Raw code storage with causal read gating

`--storage-mode raw` keeps the encoder output as the actual saved compact code.
Each read multiplies one temporary layer of codes by the existing layer/row
inherited-support flags before the same Key/Value reader consumes it. The flags
and native writes retain their original meaning. Full refresh produces the same
scores as the original method, which multiplies those flags during encoding.

Moving the multiply preserves code coordinates for rows whose correction is
currently disabled. Those coordinates are available when the existing flags
reset at the next publication. In the original `masked` storage branch, a zero
written during encoding has discarded that information, which can limit reuse
under the next model's projections. The raw branch retains the same single code
tensor and widths; it neither regenerates old codes offline nor retains a second
persistent copy. The temporary mask covers one layer at a time.

Section 4.4's selection rule applies to the saved raw codes: tail rows union
missing or older-than-previous encoder rows, excluding current-version rows.
Every selected row's encoder version changes only after all its layers/heads
have been generated. Unselected rows keep their exact saved values and versions.
Append/FIFO maintenance and all-layer causal retirement remain unchanged.

Raw storage charges `288*N` extra FLOPs and `6*N` logical flag-transfer bytes
per complete batched read, shared by up to 16 candidates; these replace the
encode-time mask charge. Publication validation occurs immediately after flags
reset to true, so its raw candidate/reference reads need no multiply. Costs
retain full-reference encoding, each candidate query, and preparation of the
actual old-code trajectories. `cost.py` reports both raw full-refresh and the
original masked full-refresh baselines. Version tags remain 4 bytes per row.

## Final evidence and scope

The strict raw policy uses tolerance.005 and candidate ratios
`[0,.125,.25,.5,.75,.875,1]`, selecting V3/V4/V5 rho0/.875/.75.
Shared adapter weights remain frozen; raw storage, tolerance and grid were
development choices. [The strategy](../../results/partial_refresh_2026_10/frozen_raw_candidates.json)
was frozen before benchmark scoring. All3360 eligible benchmark users retain
96.0448% [93.3510,97.4663]% of full-code gain; overall95% noninferiority is not
established. Long-history users fare better, while pure-new users regress even
under the full method. See [the report](../../results/partial_refresh_2026_10/report.md).

This prototype prepares selected layers synchronously. It uses valid native
cache inputs of one fixed architecture; automatic unsupported-cache fallback
and fifthchapter pipelining are outside the implemented experiment.

## Reproduction

Run from the repository root. Output paths must be fresh directories; retained
completed results are indexed in [the result README](../../results/partial_refresh_2026_10/README.md).
Every launch writes its essential settings and source/input/weight hashes.
Inspect shared GPU occupancy before choosing the `--gpu` index.

Publication validation, reusing identical fit-only correction-energy scales
(the logical fit preparation remains charged):

```bash
python scripts/partial_refresh_2026_10/run.py validate \
  --storage-mode raw --tolerance .005 \
  --ratios 0 .125 .25 .5 .75 .875 1 \
  --scale-reference results/partial_refresh_2026_10/validation_original/summary.json \
  --gpu 0 --output results/partial_refresh_2026_10/reproduction_validation
```

One disjoint shard of the complete frozen alternative population (use indices
0–3 with distinct output directories for all four shards):

```bash
python scripts/partial_refresh_2026_10/run.py evaluate \
  --users 3360 --shards 4 --shard-index 0 \
  --quality-panel results/partial_refresh_2026_10/benchmark3360/panel \
  --compute-controls --storage-mode raw --gpu 0 --verify \
  --policy results/partial_refresh_2026_10/validation_raw_strict/summary.json \
  --output results/partial_refresh_2026_10/reproduction_shard0
```

The retained final merge reused input-identical1024 prefixes, avoiding repeated
GPU controls/users. Source run paths and their configurations are indexed in
`benchmark3360/quality/summary.json` under `sources`; process logs are in the
result root's `logs/`. For any new completed
set, use `aggregate.py --runs ... --panel ... --users 3360 --cohorts ...
--bootstrap 1000 --output <fresh-dir>`.

Cost verification for the retained final quality run uses CPU metadata only:

```bash
python scripts/partial_refresh_2026_10/cost.py \
  --compare-gpu-summary results/partial_refresh_2026_10/benchmark3360/quality/summary.json \
  --quality-panel results/partial_refresh_2026_10/benchmark3360/panel \
  --output results/partial_refresh_2026_10/reproduction_cost_verification.json
```

`cost.py --audit-validation-summary <summary.json> --output <cost_audit.json>`
includes the preceding stored-code preparation. Fixed30k accounting uses
`--storage-mode raw --validation-summaries <name-to-summary JSON>`; each named
summary must have its audited sibling `cost_audit.json`. Native-source
reconstruction sensitivities remain separate from the primary available-input
boundary. `make_report.py` only reads completed results; it does not launch runs.

## Second-round reproduction

One NoCache shard (indices0–3 for the complete frozen3360user panel):

```bash
python scripts/partial_refresh_2026_10/run.py evaluate \
  --users 3360 --shards 4 --shard-index 0 --storage-mode masked --gpu 0 --verify \
  --strategy results/partial_refresh_2026_10/round2/configs/factored_nocache_full.json \
  --quality-panel results/partial_refresh_2026_10/benchmark3360/panel \
  --reference results/partial_refresh_2026_10/benchmark3360/quality/scores.parquet \
  --output results/partial_refresh_2026_10/round2/reproduction_nocache_shard0
```

For Static4k, substitute `factored_static4096_full.json` as strategy and use
fresh outputs. Both use the original immutable full/Reuse/Rebuild controls,
without rerunning those controls. The completed Static4k/RawStrict experiments
reuse their512user prefix and execute only the remaining2848users.
Source-run `policy` is the nominal launch fallback; strategy overrides it at
the first read. Per-request `rho` and final `actual_rho_by_release` retain the
actual execution setting. Current round2 strategies plug into `evaluate`;
`validate` remains the original publication-validation pipeline.

CPU cost verification of a completed run:

```bash
python scripts/partial_refresh_2026_10/round2/verify_factored_cost.py \
  results/partial_refresh_2026_10/round2/nocache3360/shard0/summary.json \
  --output results/partial_refresh_2026_10/round2/reproduction_cost_check.json
```

The helper also verifies aggregate sources and their separately charged
preparation. For fresh disjoint runs, `round2/aggregate.py --runs ... --panel ...
--users 3360 --frozen .../frozen_candidates.json --candidate nocache_full
--cohorts .../cohorts.json --bootstrap 1000 --output <fresh-dir>` binds all
settings and original controls. Fixed30k accounting uses `round2/factored_cost.py
--output <fresh-dir>` and its fixed `round2/item_tables/item_ids.npz` input;
`--add-nocache-to-summary <fresh-dir>/summary.json` derives NoCache from the same
full-call unique-item ledger without rescanning data. Then
`--add-replica-sensitivity <fresh-dir>/summary.json --replicas 4` charges
independent deployment preparation for the report's conservative primary point.
