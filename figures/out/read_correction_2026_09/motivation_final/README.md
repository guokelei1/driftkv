# Two-probe motivation overview

Status: **complete**. Q: 45/45 points; H: 45/45 points.

[One large figure](overview.png) · [Vector PDF](overview.pdf) · [Raw point table](points.csv) · [Equal-edge means](equal_edge_means.csv) · [Evidence manifest](manifest.json)

The 19 panels contain the overall mean, three scale means, and all 15 individual edges.
Q uses the retained v5 nonlinear-feature correction (`cross_phi_joint8`) at C128/256/512. H uses the nonlinear v4 all-token map at C64/128/256; C128 is reused from the completed v4 population run. Q's 45 points come from `results/read_correction_2026_09/v5/population_run`; all 45 H points are unchanged from the earlier complete H curve.
Every calibration budget is shown, without picking the best budget per edge. Lines connect actual measurements within each probe. Missing budgets are not filled or extrapolated.

| Probe | Calibration users | Overall compute | Overall gap recovered |
|---|---:|---:|---:|
| query_only | 128 | 1.49% | 11.96% |
| query_only | 256 | 2.79% | 33.35% |
| query_only | 512 | 5.43% | 37.32% |
| history_conditioned | 64 | 72.11% | 69.37% |
| history_conditioned | 128 | 74.22% | 70.95% |
| history_conditioned | 256 | 78.44% | 71.69% |

Each mean weights edges equally. Recovery is averaged after normalizing each edge by its own Full−Reuse AUC gap; it is not the recovery calculated from pooled predictions or the ratio of averaged AUC differences. The AUC columns in the mean CSV are descriptive edge means.
Incomplete budget groups are omitted from means until all 15 (overall) or five (one scale) edges are present.

Compute includes the complete calibration cost and additional inference FLOPs, divided by Full-history recompute minus rolling Reuse-append FLOPs. Reading an existing fitted artifact does not make its calibration cost zero.
Reuse (0,0) and Full (100,100) are normalization references and are not extra correction measurements.
Negative and above-100% recovery are retained; the top row shares a Y-range and each scale row shares its own Y-range.
All points use the frozen 3,000-user per-edge development panels selected in the earlier exploration. These are outcome-conditioned development results, not an independent population qualification.

Reproduce from saved results:

```bash
python figures/src/read_correction_motivation_2026_09.py --input-root results/read_correction_2026_09/motivation_final --output-dir figures/out/read_correction_2026_09/motivation_final --require-complete
```

The generator reads saved results only, verifies paired Full/Reuse controls and compute denominators, checks every retained Q and H score hash, and records source-summary hashes in the CSV and manifest.
