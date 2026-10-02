# Data and workload primitives

Current Yambda experiments share causal history loading, frozen populations and
catalog mappings across training, Full/Recompute, Reuse and Motivation.

| Component | Role |
| --- | --- |
| `yambda.py`, `yambda_history.py` | Event loading, deterministic history order and timestamp deltas |
| `scale_population.py` | Initial-history eligibility and stable user selection |
| `release_windows.py` | Half-open training and evaluation windows |
| `yambda_scale_dataset.py` | Shared physical store and logical scale views |
| `oov.py` | Stable catalog and OOV mapping |
| `foundation_manifests.py` | Shared causal request/snapshot definitions |

Medium and Large use `data/processed/yambda500m_unified_v1`; Max uses
`data/processed/yambda5b_max_200k_v1`. Their fixed users, item mappings and
request manifests support the [18 selected models](../../../docs/unified_training_2026_09/model_versions.md)
and [15 adjacent Full/Reuse edges](../../../results/unified_reuse_2026_09/README.md).
Full and Recompute denote the same request-local history reconstruction.

Preparation entrypoints are listed in [scripts/README.md](../../../scripts/README.md):
the 500M population/shared-store builders, `build_yambda500m_hstu_native_matrix_manifest.py`,
and Max `unified_training/prepare_max_data.py` plus `audit_max_data.py`.
The retired Small manifest builder is not the current request pipeline.

The [four recomputation baselines](../../../results/selective_recompute_2026_09/README.md)
and [Q-v5/H-v4 correction](../../../results/read_correction_2026_09/motivation_final/README.md)
reuse fixed diagnostic evaluation panels, whose outcome-based selection is recorded
separately from the initial training population. Keep catalog fitting and initial
population selection before the relevant release cutoff.

[RecFlow](../../../docs/recflow/plan.md) has an independent data pipeline in
`hstu_kvcache.recflow`; its current six-layer A–F chain retains the frozen initial
catalog and complete realshow history. Its data and user roles are separate from Yambda.
