# D=14/E=14 direct long-age KV Reuse: current v2

Both rows use v2 exact rolling cache as Recompute over days [245, 259). The v0 row is direct long-age Reuse; the v1 row reuses the already sealed adjacent one-hop result.

| Current | KV producer | Version gap | Current - Reuse ROC-AUC (pp) | Current - Reuse PR-AUC (pp) | Reuse - Current event log-loss | User-equal log-loss | JS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 | v0 | 2 | +0.530852 | +0.324660 | +0.000781 | +0.000471 | 3.49e-05 |
| v2 | v1 | 1 | +0.199662 | +0.181358 | +0.000274 | +0.000110 | 3.18e-06 |
