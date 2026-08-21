# Test inventory

The focused suite protects the current evidence chain:

- `test_yambda_*`: batch alignment, incremental time continuity, and release snapshot lineage.
- `test_cc_*`: candidate-conditioned scoring, candidate causality, seen-mix, and P5/P6 protocol boundaries.
- `test_p7_*`: stateful workload contracts, compact qualification guard, Frozen Base, theta0 training, and H qualification.
- `test_p8_*`: release definitions, execution handoff, admission, and lineage.
- `test_p9_contract.py`: P8 evidence sealing, GPU allowlist, label-free tomography, splice invariants, and diagnostic/executable separation.

Add tests before implementing a new dependency-closed action. In particular, an executor must prove boundary-state availability, Full/Reuse baseline identity, chunk/order invariance, work accounting, and that future labels do not enter decisions.
