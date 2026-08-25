# Checkpoint cleanup record — 2026-08-24

This record documents a storage-only cleanup before migrating the repository to
`/home/gkl/work/o1`. No contracts, hashes, raw-score artifacts, seals,
adjudications, negative results, invalidations or F-only checkpoints were
deleted.

## Deleted weight payloads

The following selected checkpoint payloads were removed after their frozen P7/P8
raw evaluations and adjudications had completed:

```text
results/p7/theta0_training/runs/m0_n_seed{17,37,71}/theta0_selected.pt
results/p7/theta0_training/runs/m0_r_seed{17,37,71}/theta0_selected.pt
results/p7/theta0_training/runs/m1_seed{17,37,71}/theta0_selected.pt
results/p8/release_training/{r0,r1_edge1,r1_edge2,r2}/m1_seed{17,37,71}/selected.pt
```

- P7 removed bytes: `43,284,701,196`.
- P8 removed bytes: `57,712,931,652`.
- Total removed bytes: `100,997,632,848` (`94.061 GiB`).

These models are outside the active F-only S/M/L route. Their exact historical
checkpoint paths and SHA-256 values remain frozen in the P7 qualification
contract and P8 training/result metadata. Reproducing their forward passes now
requires retraining from the frozen contracts.

## Retained checkpoint payloads

- all P7 `m0_f_seed{17,37,71}` theta0 checkpoints;
- all P8 `m0_f_seed{17,37,71}` R0/R1/R2 release checkpoints;
- all current and future S/M/L F-only artifacts.

## Architecture-pilot payload cleanup

After clarifying that the completed 8L run used Yambda-50M and was not the
prospective Yambda-500M Large scale, its five inference checkpoints were also
removed:

```text
results/scale_8l_v1/theta0/m0_f_seed17/theta0_selected.pt
results/scale_8l_v1/releases/{r0,r1_edge1,r1_edge2,r2}/m0_f_seed17/selected.pt
```

- Additional removed bytes: `48,134,093,985` (`44.828 GiB`).
- Cumulative removed bytes: `149,131,726,833` (`138.890 GiB`).

The pilot remains valid only as sealed architecture-development evidence.
Compact summaries, seals, invalidation records and adjudications remain, while
the bulky raw pilot tables were removed in the later artifact cleanup recorded
below. Re-running pilot forwards now requires retraining from the frozen
historical contract. The formal Yambda-500M Large lineage has never been
trained.

## Retired root checkpoint cleanup

The original root `checkpoints/` payloads predated the frozen F-only evidence
chain. They covered obsolete medium-lineage prototypes and sampled next-listen
CC models, and were referenced only by archived development scripts. Ten `.pt`
files totaling `3,468,048,468` bytes (`3.230 GiB`) were removed:

```text
cc_theta0_v1.pt
cc_theta0_v2_dense.pt
yambda50m_v2_theta{0,1,2}_medium.pt
yambda50m_v2_theta{1,2}_medium_lineage_v2.pt
yambda50m_v2_theta{0,1,2}_medium_batchfix_v3.pt
```

Their historical conclusions remain represented by manifests, raw evaluations
and invalidation/No-Go records. The root `checkpoints/` directory is reserved
for future semantically named S/M/L and RecFlow model payloads.

## Retired branch and intermediate-artifact cleanup

The active route is now F-only. The following artifacts had been superseded by
compact adjudications and are not inputs to scale or external-validation work:

- P7 N/R and M1 run metadata and raw qualification tables;
- P8 M1 release metadata and N/R/F raw staleness tables;
- P9 diagnostic tomography, executor, profiler, rolling-quality and debug raw
  directories;
- P10/P11 intermediate profiler, runtime, recursive-population, scheduler and
  quality raw directories;
- bulky raw tables and canaries from the retired Yambda-50M 8L architecture
  pilot and the aborted N/R/F M1 pilot.

The exact resolved targets totalled `1,895,804,494` bytes (`1.765 GiB`). They
were deleted only after preserving the corresponding compact reports, seals,
contracts, adjudications and invalidation records. P7 M0-F qualification raw
tables, P8 M0-F staleness raw tables and all 15 core F-only checkpoints remain.
Consequently, obsolete branches can still be audited from their compact frozen
records, but request-level reanalysis now requires regeneration from contract.
