# D=14/E=14 direct long-age KV Reuse: current v5

Every row uses v5 exact rolling cache as Recompute. The named producer materializes the entire pre-cutover prefix at day 287; v5 then reads that producer KV and appends all events in days [287, 301). This is direct long-age Reuse, not recursive lineage.

| Current | KV producer | Version gap | Current - Direct Reuse ROC-AUC (pp) | Current - Direct Reuse PR-AUC (pp) | Direct Reuse - Current event log-loss | User-equal log-loss | JS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v5 | v0 | 5 | +0.848679 | +0.162810 | +0.001312 | +0.000907 | 1.07e-04 |
| v5 | v1 | 4 | +0.341018 | +0.025958 | +0.000454 | +0.000495 | 3.99e-05 |
| v5 | v2 | 3 | +0.258975 | +0.013682 | +0.000280 | +0.000403 | 2.55e-05 |
| v5 | v3 | 2 | +0.205613 | -0.000479 | +0.000311 | +0.000385 | 3.08e-05 |
| v5 | v4 | 1 | +0.063736 | +0.054249 | +0.000042 | +0.000005 | 2.31e-06 |
