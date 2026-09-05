# Test inventory

The focused suite protects the current evidence chain:

- `test_yambda_*`: batch alignment, incremental time continuity, and release snapshot lineage.
- `test_foundation_*`: HSTU-native foundation manifests, metrics, cache lineage and launch contracts.
- `test_yambda500m_*`: scale population, streaming windows, unified preprocessing and stable OOV.
- `test_state_transition.py`: causal rolling state transitions, tail retention and work accounting.
- `test_download_scale_datasets.py`: download-plan and verification behavior.
- `test_insight_one_locality.py`: frozen Medium locality action enumeration, dependency closure and
  metric aggregation.
- `test_insight_two_functional_boundary.py`: stage corrections, rank diagnostics, temporal
  persistence/coordinates and estimator cost helpers.
- `test_insight_two_signed_response_memory.py`: signed chronological K/V coreset construction,
  native-query read, source-position semantics and all-position Exact reconstruction.

Add tests before implementing a new dependency-closed action. In particular, an executor must prove boundary-state availability, Full/Reuse baseline identity, chunk/order invariance, work accounting, and that future labels do not enter decisions.
