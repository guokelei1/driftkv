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

- `manuscript.md`: the complete English systems-paper working draft.
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
5. `docs/paper_draft_intro_motivation.md`, used only for advisor-facing exposition

When an older experiment note conflicts with the roadmap, the draft follows the roadmap. In
particular, task quality is **not** an admission oracle, version cohorts are **not** predictors of
safe reuse, and fixed-prefix/suffix/interval actions are baselines rather than the active method.

## Current manuscript status

The manuscript is a complete working draft after three substantive revision rounds: factual and
protocol audit, narrative and system-closure audit, and reviewer-attack/final-presentation audit.
It can support discussion with an advisor and guide the remaining experiments. It is not
represented as submission-ready because the verified compiler still needs frozen replication and
the destination-oriented engine still lacks automatic fallback execution and a complete-cohort,
identical-boundary compiled-versus-exact evaluation. Physical storage measurements and
venue-specific typesetting also remain open.
