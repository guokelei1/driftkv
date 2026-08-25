# Data and workload primitives

This package contains reusable, protocol-aware data components rather than a monolithic experiment corpus builder.

- `yambda.py`: Yambda loading and timestamp-correct incremental deltas.
- `foundation_manifests.py`, `compact_manifest.py`: causal request/snapshot manifests and access guards.
- `release_windows.py`, `scale_population.py`, `yambda_scale_dataset.py`: current Yambda-500M
  population, window and shared-store processing.
- `oov.py`: stable catalog/OOV mapping for the HSTU-native lineage.

Formal experiments must use frozen manifests and release cutoffs. Do not fit catalog maps, feature transforms, candidates, or cohorts on future/qualification data. Fidelity views must remain free of target, label, rankability, and future shortcut fields.
