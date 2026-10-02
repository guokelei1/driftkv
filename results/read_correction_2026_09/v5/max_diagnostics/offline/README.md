# Max request-level diagnosis

This is a CPU-only analysis of retained v5 Q and v4 H scores. It does not run
models or change fitting, evaluation, or plotting sources. All 15 edges, all
three Q budgets, and both retained H scopes remain in the output.

Reproduce with `python results/read_correction_2026_09/v5/max_diagnostics/offline/analyze_scores.py`.
`manifest.json` binds the exact input parquet files and output CSVs.

## Findings

1. **The first Max edge loses actual AUC, not merely normalized recovery.**
   Reuse is 0.62008, Q128 is 0.57301, Q512 is 0.61212, and H-all is 0.63639.
   Relative to Full, Q128 raises total logit MSE from 0.10981 to 0.68204 and
   within-UID error MSE from 0.00706 to 0.48910. Its correction has correlation
   -0.028 with the required Full-minus-Reuse logit change. Q512's correlation
   is only 0.024; its mean change is -0.0615, versus a required -0.2933.

2. **Full-length mixed caches carry most of the observed loss.** On this edge,
   20,642 requests from 2,499 users have length 1,024 and a mixture of producers.
   Their Reuse/Q128/Q512/H-all AUCs are 0.6247/0.5722/0.6142/0.6446. The loss
   persists in every time bin, and is not explained by a small short-history
   subgroup. Q128 loses AUC in all four activity bins.

3. **The fourth Max edge exposes a mean-error versus ranking mismatch.**
   Q512 lowers total Full-logit MSE from 0.19457 to 0.16539, but AUC falls from
   0.56480 to 0.54562. The global error mean improves from +0.35353 to +0.15806;
   after removing that constant mean, error MSE rises from 0.06959 to 0.14040.
   Within-UID error MSE rises from 0.01812 to 0.06051. A constant shift cannot
   repair AUC. The full-length mixed-state subgroup contains 30,807 requests /
   2,685 users, and also loses AUC: 0.5745 to 0.5513; H-all reaches 0.5852.
   High-activity users (17+ evaluated requests) contribute 22,860 requests from
   420 users, with AUC 0.5849 to 0.5583. The 1–4-request group improves instead.

4. **Even the positive Max edge is not a faithful Full-logit repair.** On
   V1→V2, Q512 improves pooled AUC from 0.52035 to 0.53558, while Full-logit MSE
   increases from 0.04472 to 0.84571. Equal-UID AUC among users with both labels
   changes from 0.56946 to 0.56740. H-all improves pooled AUC and reduces logit
   MSE to 0.02261. A positive pooled recovery alone does not establish accurate
   restoration of each user's response.

5. **The state shift also appears within the same users.** For the 93 users
   with both pure inherited and mixed full-length states on Max V0→V1, Q128
   MSE is 0.0775 before append versus 0.6776 in the mixed state; Q512 is
   0.0556 versus 0.1307, while Reuse is 0.0997 versus 0.1291. This supports
   investigating calibration-to-rolling-state mismatch. It is not a causal
   isolation: requests and times change, and pure-state AUC can already be bad.

6. **The three scales differ under the same descriptive state condition.**
   For V0→V1 full-length mixed caches, Q512's correction-to-required-change
   correlation is 0.619 / 0.107 / 0.005 for Medium / Large / Max. Q512 raises
   AUC by 0.0971 / 0.0378 / -0.0105 respectively. The model-specific panels
   are different: only 120 Medium/Max and 141 Large/Max users overlap here.
   These are descriptive comparisons, not an isolated effect of model depth.

## Reading the outputs

- `overall_metrics.csv`: 75 complete-panel method rows.
- `bucket_metrics.csv`: 2,475 rows, including count/UID/positive counts, raw
  AUC, Full-logit errors, correction bias/alignment, and within-user AUC.
- `diagnostic_summary.json`: compact priority-edge records and limitations.
- `manifest.json`: verified source hashes, metric definitions, and user-panel
  overlap counts. The script verifies all Q/H request fields exactly agree.
- `auc_cluster_bootstrap.{json,csv}`: paired UID-cluster bootstrap of the two
  priority Max edges, C128/C512 versus Reuse, on all 3,000 users (400 draws,
  seed17). Max V0→V1 C128/C512 and V3→V4 C512 have negative 95% intervals;
  V3→V4 C128's interval crosses zero.
- `intervention_cluster_bootstrap.{json,csv}`: the completed 256-user
  interventions and random-user controls. Disabling early-half corrections
  improves Max V0→V1 at both budgets, and V3→V4 at C512, with positive paired
  intervals; every drop-constant interval crosses zero. Random-user V3→V4
  retains negative intervals for both budgets. Random-user V0→V1 intervals
  are wide and cross zero, so this smaller sample does not resolve that edge.
- `random2048_cluster_bootstrap.{json,csv}`: the larger, nested, independently
  hashed Max V0→V1 panel (2,048 users; 21,313 requests). Q128 minus Reuse is
  −0.051874 [−0.088551, −0.008368]; Q512 is −0.011703 [−0.029906, +0.007816].
  The second interval crosses zero. Full AUC is 0.687354 and Reuse is
  0.688985, so use absolute AUC differences here; a ratio with this negative
  Full−Reuse denominator must not be interpreted as positive recovery.
- `fitting_v0_to_v1_*` and `fitting_v3_to_v4_*`: paired 256-user evaluations
  and uncertainty for N²-weighted fitting and prerelease mixed-state
  calibration. Both bring partial changes, with no uniform AUC repair.
  The diagnostic root README contains the four-budget comparison table.

Bootstrap uncertainty is conditional on the fixed user panel and correction
weights. It is not a replacement for independent training seeds. The retained
draw CSVs allow verification without repeating any inference. Run
`bootstrap_auc.py` and `bootstrap_interventions.py` in this directory to
reproduce the initial CPU-only calculations. The follow-ups use
`bootstrap_fitting.py` and `bootstrap_random2048.py`; their manifests retain
input score hashes and the comparison definitions.

Bucket AUCs are not additive contributions to global AUC. User counts can
overlap between time/state bins. The activity partition is descriptive and
uses the fixed evaluation panel's request counts. All observations concern
these outcome-conditioned development panels; there is no causal architecture
claim or evidence that increasing the calibration budget is a universal fix.
