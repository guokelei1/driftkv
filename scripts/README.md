# Script inventory

Scripts are grouped by evidence role. The filename alone does not authorize an experiment; consult `docs/current_route.md` and the corresponding frozen contract.

## Active P9

- `seal_p9_evidence.py`: seal the complete P8 substrate.
- `analyze_p9_hs_distribution.py`: user-level H/S distributions and cohorts.
- `run_p9_tomography.py`, `run_p9_tomography_queue.py`: GPU-0/1 diagnostic tomography execution.
- `eval_p9_tomography_raw.py`: produce label-free diagnostic raw scores.
- `seal_p9_tomography.py`: seal per-cell raw outputs.
- `adjudicate_p9_tomography.py`, `adjudicate_p9_tomography_matrix.py`: per-cell and 24-cell summaries.

P9.2 is complete. Before P9.3, add a small versioned CPU analysis for frozen-logit quality companions and risk concentration. Do not change existing raw outputs or select regions using labels.

## Frozen evidence chain

- P7: `build_p7_*`, `fit_p7_frozen_base.py`, `train_p7_theta0.py`, `eval_p7_h_raw.py`, `seal_p7_*`, `adjudicate_p7_h.py`.
- P8: `build_p8_release_manifests.py`, `train_p8_release.py`, `eval_p8_release_raw.py`, `seal_p8_*`, `adjudicate_p8_*`, `run_p8_pipeline.py`.
- P5/P6: `cc_p5_seenmix_requalification.py` and `cc_p6_identifiability_adjudication.py` preserve failed next-listen qualification evidence.

These scripts may reproduce or audit frozen evidence; they must not be used to tune the established model/release chain.

## Historical development tools

The remaining `audit_yambda_*`, `eval_yambda_*`, medium-model, neutral-readout, oracle/frontier, risk-ranker, and large-candidate scripts record earlier protocol development. Their old outputs were invalidated or superseded where documented. In particular, they are not current controller/frontier entry points and must not be used to revive pre-P7 claims.
