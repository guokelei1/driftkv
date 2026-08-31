# Structured-log test contamination notice

`logs/pipeline.jsonl` contains log-only `admission_sealed` events for synthetic
D7 edge 1/2 reports emitted by the pipeline unit test before its temporary
output paths were fully isolated. These events did not read or modify any formal
raw score, adjudication, admission seal, checkpoint, summary, or lineage state.

The affected synthetic pairs are the edge 1/2 events at:

- 2026-08-27 18:50:43 UTC and 18:50:47 UTC;
- 2026-08-28 06:44:10, 06:48:13, 06:51:21, 06:51:33 and 06:53:45 UTC.

The formal D7 admission events at 2026-08-28 03:04:24 through 06:29:41 UTC
remain valid. On-disk `D7/admission/*.seal.json` files are authoritative. The
test now redirects `logs` and `pipeline.jsonl` together with its temporary
output root; a regression check confirmed that rerunning it does not change the
formal structured-log line count.
