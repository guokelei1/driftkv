# Structured-log test contamination notice

The historical `logs/pipeline.jsonl` contained log-only `admission_sealed` events for synthetic
D7 edge 1/2 reports emitted by the pipeline unit test before its temporary
output paths were fully isolated. These events did not read or modify any formal
raw score, adjudication, admission seal, checkpoint, summary, or lineage state.

The affected synthetic pairs are the edge 1/2 events at:

- 2026-08-27 18:50:43 UTC and 18:50:47 UTC;
- 2026-08-28 06:44:10, 06:48:13, 06:51:21, 06:51:33 and 06:53:45 UTC.

The formal D7 admission events at 2026-08-28 03:04:24 through 06:29:41 UTC
were unaffected. At the time, `D7/admission/*.seal.json` was authoritative;
the D7 branch has since been removed, and historical outcomes remain in
[the experiment summary](medium_scale_experiment_summary.md).
The test was fixed to redirect both log paths into its temporary output root;
the subsequent regression check left the formal log line count unchanged.
This notice records the original contamination boundary, not a current log or pending test.
