# CohortKV Stage 4.10 renewal-calibrated program

## Status

The implementation and two-edge real-cohort smoke are complete as of 2026-07-28. The artifacts
are explicitly `scientific_result=false`. They validate the new lifecycle and its accounting; they
do not measure recommendation quality and do not replace the frozen Stage-4.9 eleven-edge result.

## Design change

The frozen Stage-4.9 path loads one adjacent direct-old-K/V program that was fitted offline on 40
separate fit users. Its `U/E` excludes that program construction. Stage 4.10 removes this separate
fit cohort from the candidate lifecycle:

1. H12 fixes the edge action partition before any candidate output is observed.
2. Every scheduled-exact record exposes its retained previous-actual K/V and receives one
   current-model retained-prefix exact replay.
3. The aligned actual-old/fresh-target pairs fit one shared adjacent program.
4. The same fresh target K/V refreshes those scheduled-exact records; it is not recomputed for
   program construction.
5. The shared program migrates the remaining H12 migrant records.
6. Natural exact records retain the Stage-4.9 semantics.

Thus scheduled exact has two uses but one exact computation. There are zero additional
fit-recompute records, no fixed 40-user program load, no recommendation-label routing, and no
semantic admission gate.

## Runtime representation

The lightweight runtime already consumes actual old K/V directly. It never reconstructs
`Norm(x)` in the record migration path. Both Stage-4.10 fit variants publish the same
`DirectOldKVProgram` ABI and use the unchanged fused direct-old-K/V operator.

Let \(Z=[K_{\mathrm{old}},V_{\mathrm{old}}]\) and
\(Y=[K_{\mathrm{fresh}},V_{\mathrm{fresh}}]\) for the scheduled-exact retained tokens.

### Direct K/V residual ridge

This variant fits an affine residual around identity:

\[
Y-Z \approx (Z-\mu_Z)\Delta A + \mu_R,
\qquad
A=I+\Delta A.
\]

The identity prior makes a poorly identified direction preserve actual old K/V instead of
shrinking toward an unconditional mean.

### Inverse-Norm ridge

For source projection \(P_s=[W_{K,s},W_{V,s}]\) and bias \(b_s\), this variant first estimates

\[
\widehat N=(Z-b_s)P_s^\top(P_sP_s^\top)^{-1}.
\]

It fits the residual from the target cheap projection
\(\widehat N P_t+b_t\) to \(Y\), then analytically composes the result back into one direct
\(Z\mapsto Y\) affine. Approximate-Norm recovery is therefore a program-build option, not a
runtime source-state requirement.

Both variants use centered ridge with `ridge=0.001`, at most 8,192 deterministically selected
paired tokens, FP32 fitting, and one FP16 direct program per adjacent edge.

## Cost boundary

For each edge, \(U\) includes:

- scheduled-old retained crop;
- the scheduled-exact retained replay that also supplies calibration targets;
- ridge solve, direct-program construction, FP16 cast, and device preparation;
- migrant retained crop and direct transform;
- missing-cache exact rebuild when present;
- retained output materialization.

\(E\) independently recomputes the same timed retained-prefix population under the target model.
Target delta/latest append remains common foreground work outside both timers. Calibration-old
H2D, migrant H2D, and next-state D2H are measured in separate movement ledgers, consistent with
the Stage-4.9 device-resident primary boundary. No program-build component is silently omitted
from \(U\).

## Smoke

The smoke runs the real seed-0 KuaiRand 4+12, 16L/H512 H12 cohort over
\(\theta_0\rightarrow\theta_1\rightarrow\theta_2\), with zero warmup and one timing repetition.
The second edge consumes the first edge's actual post-append mixed cache and continuous H12 state.

| Fit variant | Edge | Migrate | Scheduled exact | Natural exact | Program build GPU ms | \(U/E\) |
|---|---:|---:|---:|---:|---:|---:|
| Direct K/V residual ridge | 0→1 | 553 | 50 | 79 | 123.698 | 0.146283 |
| Direct K/V residual ridge | 1→2 | 548 | 46 | 88 | 80.004 | 0.112352 |
| Inverse-Norm ridge | 0→1 | 553 | 50 | 79 | 111.332 | 0.144788 |
| Inverse-Norm ridge | 1→2 | 548 | 46 | 88 | 74.805 | 0.111780 |

The two-edge aggregate is `0.128764×` for direct K/V residual ridge and `0.127694×` for
inverse-Norm ridge. These are smoke diagnostics from different GPUs, not a performance ranking.
Program construction is approximately 75–124 ms per edge rather than the old fixed compiler's
multi-pair offline setup, but no amortization or end-to-end claim follows from two edges.

Both artifacts pass:

- calibration IDs exactly equal scheduled-exact IDs;
- calibration and migrant IDs are disjoint;
- no extra exact fit records;
- no serialized direct-program load;
- no semantic gate or action reroute;
- target-version, finite, shape, length, and CPU recursive-store checks;
- continuous scheduler state and previous-next store agreement;
- edge-2 calibration includes records migrated on edge 1.

## Artifacts

- `scripts/run_cohortkv_stage4_10_renewal_calibrated_smoke.py`
- `src/hstu_kvcache/migration/renewal_calibration.py`
- `tests/test_renewal_calibration.py`
- `results/system/cohortkv_single_config_full_chain_v1/stage4_10_renewal_calibrated_h12_direct_kv_residual_ridge_2edges_seed0.json`
- `results/system/cohortkv_single_config_full_chain_v1/stage4_10_renewal_calibrated_h12_inverse_norm_ridge_2edges_seed0.json`

## Open gate

No AUC, NDCG@100, Hit@100, cache-fidelity, or score-quality result is produced by this smoke.
Neither fit variant is selected. Before this route can replace the frozen Stage-4.9 program path,
the corrected evaluator must measure paired task quality and complete cost over the full recursive
chain under a new formal protocol. The old Stage-5 semantic canary is not part of this route; basic
hash, version, shape, finite, capacity, and transaction-integrity checks remain required.
