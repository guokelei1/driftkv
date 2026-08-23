# Retired pre-P7 Yambda tools

The repository cleanup on 2026-08-22 removed obsolete top-level scripts tied to
the superseded Q_main/neutral-readout/controller route. Their relevant scientific
lessons remain in `docs/evidence_summary_through_p11.md` and `docs/newset.md`, and
their generated development artifacts remain under ignored `results/` where
available.

Removed categories:

- v1 data audit and old dilution/panel/proxy/ranker reports;
- old Q_main panel/profiler/release-snapshot builders;
- pre-P7 No-op/Exact oracle and metadata/GBDT controller scripts;
- old compatibility, multi-panel, large-candidate and repair/bypass evaluators;
- P4 adjudication superseded by the retained P5/P6 No-Go chain.

These tools depended on one or more invalidated assumptions: old release cutoff,
old temporal-delta execution, neutral readout, sampled next-listen candidates,
request-local rather than materialized lineage, or a controller/frontier that was
superseded by P9–P11.

The following are deliberately retained:

- `cc_p5_seenmix_requalification.py` and `cc_p6_identifiability_adjudication.py`;
- their CC qualification dependencies;
- `train_yambda_two_edges.py` and its model/data helpers needed by regression tests;
- all P7–P11 evidence-generation, seal and adjudication scripts;
- HSTU correctness canaries and P7 manifest verification.

Do not recreate removed tools unless a new prospective contract requires a
specific reusable primitive. In that case, implement the primitive against the
current F workload, frozen lineage and target-free protocol rather than restoring
the old route.
