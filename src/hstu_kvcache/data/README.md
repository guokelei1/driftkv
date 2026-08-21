# Data and workload primitives

This package contains reusable, protocol-aware data components rather than a monolithic experiment corpus builder.

- `kuairand.py`, `streaming_plan.py`: retained KuaiRand and chronological streaming primitives.
- `yambda.py`: Yambda loading and timestamp-correct incremental deltas.
- `cc.py`, `identifiability.py`: candidate-conditioned proposals and the frozen next-listen identifiability audits.
- `stateful_workloads.py`: N/R/F workload semantics and prospective feedback strata v2.
- `compact_manifest.py`: compact request materialization and qualification access guard.

Formal experiments must use frozen manifests and release cutoffs. Do not fit catalog maps, feature transforms, candidates, or cohorts on future/qualification data. Fidelity views must remain free of target, label, rankability, and future shortcut fields.
