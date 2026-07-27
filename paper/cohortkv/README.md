# CohortKV paper project

This directory contains the paper-only deliverables for the repository. No source code, test,
experiment, checkpoint, or result file is modified by this project.

## Working title

**CohortKV: Compiled Cross-Version K/V Migration for Streaming Generative Recommendation**

The repository implementation and protocol strings retain the existing `streamkv_*` names. The
paper uses **CohortKV** because “StreamKV” is already the title of an unrelated 2026 paper on
streaming video question answering. Renaming implementation artifacts would break traceability and
is outside this writing task.

## Deliverables

- `manuscript_v3_target_en.md`: the current English target manuscript.
- `references.bib`: bibliography source.
- `figures/`: paper-native SVG figures.
- `process/01_project_plan.md`: boundary-controlled, multi-round writing plan.
- `process/02_reference_reverse_engineering.md`: what is learned from the five target papers and
  where that technique is used.
- `process/03_claim_evidence_matrix.md`: the auditable claim–evidence–design map.
- `process/04_outline_and_figure_plan.md`: section and figure logic.
- `process/05_review_log.md`: actual issues found and changes made in successive revision rounds.
- `process/06_open_experiment_gaps.md`: experiments still required before a strong systems
  submission.

## Evidence labels used in the draft

- **Replicated**: training-seed replication under one frozen protocol.
- **Controlled seed-0**: real-checkpoint evidence from a fixed, disjoint trace, but still adaptive
  system development rather than confirmatory replication.
- **Interface-validated**: executable correctness and publication semantics only.
- **Negative**: an attempted route that is retained to delimit the design.
- **Open**: a required result that is not yet available and is never filled with a projected
  number.

## Source precedence

1. `docs/08_core_insights_and_roadmap.md`
2. `docs/eval_protocol.md`
3. current experiment records and current result families
4. implementation, used to verify mechanics rather than recover abandoned claims

When an older experiment note conflicts with the roadmap, the draft follows the roadmap. In
particular, task quality is **not** an admission oracle, version cohorts are **not** predictors of
safe reuse, and fixed-prefix/suffix/interval actions are baselines rather than the active method.

## Current manuscript status

The manuscript is a complete working draft after three substantive revision rounds: factual and
protocol audit, narrative and system-closure audit, and reviewer-attack/final-presentation audit.
It can support discussion with an advisor and guide the remaining experiments. The seed-0
compiler artifact and deployed-representation certificate path are complete, but frozen
new-seed replication remains open. The capsule/operator path now has one common unpadded extent
API and a full development-length-distribution resident result; it remains controlled seed-0
evidence. The Stage-4 destination engine now has a complete 30-point normal-path
compiled/selective/exact/control evaluation at HBM/DRAM over 1/2/4 GPUs. That result contradicts
the expected endpoint speedup: the current FP16 capsule source path loses to exact in all six
matched conditions because source processing dominates. The manuscript records this negative
result. Stage 4.5 then freezes a direct-old-K/V hot-HBM source policy: it eliminates additional
per-record `Norm(x)`, preserves the deployed certificate and complete real transport, reclaims old
extents, and beats paired raw-history-resident exact on full-cohort 1/2/4-GPU points. Stage 4.6
then freezes one KuaiRand seed-0/one-A40 theta0-to-theta11 lifecycle: balanced edge-severity and
age/deadline scheduling costs 0.2134× cumulative all-exact GPU time, keeps refresh near 15%–25%,
and bounds migration depth at four. Its per-cache threshold predecessor remains a refresh-wave
negative result. This is still controlled fixed-history evidence, not cold storage, automatic
tiering, organic traffic, or replicated lifecycle evidence. Automatic fallback, failure
injection, the remaining comparison economics, physical durable-storage measurements, new-seed
replication, and venue-specific typesetting remain open.
