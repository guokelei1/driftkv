# Archived: P11.2–P11.3 frozen scheduler transfer to recursive lineage

The frozen P10 scheduler algorithm was replayed without retuning on the true
theta0/theta1 recursive state at theta2. Assignments were sealed before any
same-cost comparison. The primary configuration remains a deterministic 1%
probe, StandardScaler + Ridge(alpha=1), the six frozen actions, and 5%/10%/25%
exact-equivalent token-layer budgets. The 2% probe is retained as a companion.

Both models pass the same-cost state-level scheduler gate at every primary
budget. Every comparison is positive in all three training seeds, and every
paired-user bootstrap confidence interval is positive.

| Model | Budget | Ridge recovery (three seeds) | Advantage over strongest deterministic baseline, seed mean |
|---|---:|---:|---:|
| M0-F | 5% | 29.8% / 35.8% / 37.2% | +18.2 pp |
| M0-F | 10% | 41.0% / 56.4% / 57.7% | +11.2 pp |
| M0-F | 25% | 69.4% / 81.7% / 88.6% | +20.2 pp |
| M1 | 5% | 46.3% / 48.5% / 32.9% | +28.1 pp |
| M1 | 10% | 71.2% / 66.8% / 60.3% | +12.0 pp |
| M1 | 25% | 95.3% / 93.3% / 95.4% | +17.3 pp |

Random Exact recovers approximately its budget fraction (about 5%, 10% and
25%), so the gain is not a trivial consequence of migrating any states. The
offline near-optimal upper bound remains higher, especially at low budgets,
leaving room for future profiling improvements; those improvements are not
being searched on this development edge.

P11.4 is the already-frozen next check: join these sealed actions to rolling
development feedback requests and determine whether target-free fidelity
recovery preserves aggregate and rare-class quality. Theta3 remains untouched.
