# D=14/E=14 direct long-age KV Reuse

Every row uses the current model's exact rolling cache as Recompute. For Direct Reuse, the named producer recomputes the **entire** pre-cutover prefix; the current model then reads that producer KV and appends every post-cutover event. This is direct long-age Reuse, not recursive lineage.

| Current | KV producer | Version gap | Current − Direct Reuse ROC-AUC (pp) | Current − Direct Reuse PR-AUC (pp) | Direct Reuse − Current event log-loss | User-equal log-loss | JS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | v0 | 3 | +0.452777 | +0.180662 | +0.000596 | +0.000338 | 3.23e-05 |
| v3 | v1 | 2 | +0.224708 | +0.116440 | +0.000268 | +0.000023 | 4.26e-06 |
| v3 | v2 | 1 | +0.148693 | +0.052190 | +0.000195 | -0.000041 | 2.31e-06 |
| v4 | v0 | 4 | +0.694185 | +0.101075 | +0.000370 | +0.000397 | 8.85e-05 |
| v4 | v1 | 3 | +0.360988 | +0.072525 | +0.000124 | +0.000143 | 2.91e-05 |
| v4 | v2 | 2 | +0.253164 | +0.064441 | +0.000063 | -0.000012 | 1.76e-05 |
| v4 | v3 | 1 | +0.193984 | -0.001293 | -0.000080 | -0.000095 | 2.27e-05 |
