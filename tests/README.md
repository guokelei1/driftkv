# Test inventory

The focused suite protects the current evidence chain:

- `test_yambda_*`: batch alignment, incremental time continuity, and release snapshot lineage.
- `test_foundation_*`: HSTU-native foundation manifests, metrics, cache lineage and launch contracts.
- `test_yambda500m_*`: scale population, streaming windows, unified preprocessing and stable OOV.
- `test_state_transition.py`: causal rolling state transitions, tail retention and work accounting.
- `test_download_scale_datasets.py`: download-plan and verification behavior.

Add tests before implementing a new dependency-closed action. In particular, an executor must prove boundary-state availability, Full/Reuse baseline identity, chunk/order invariance, work accounting, and that future labels do not enter decisions.
