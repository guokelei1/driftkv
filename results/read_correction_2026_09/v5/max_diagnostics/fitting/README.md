# Max fitting diagnostics

CPU-only inspection of all 15 v5 edges, C128/C256/C512: 45 calibration artifacts and 480 layers. No model execution or frozen-source changes. `analyze.py` produces `analysis.json`, `overview.csv`, and `layers.csv`; input hashes are in the JSON.

## Observed differences

- Max and Large both use D320, 10 heads, head dimension 32, legacy inclusive ELU+1 attention, and qk_scale=1. Max has 16 layers versus Large's 10 and a different catalog/training lineage. This is not a controlled comparison of depth alone.
- First-layer residual-rate RMS units span 3,062–11,141 in Max, 268–912 in Large, and 107–482 in Medium. Every inspected parameter is finite. After accounting for target output units, the largest local Jacobian gain per artifact is 2.19–4.34 in Max versus 2.36–3.51 in Large; raw large parameter norms alone do not prove a numerical explosion.
- Max has 50 layer/budget instances where validation physical read MSE increases after ridge, Large has 20, Medium has none. Max2 C128 is nevertheless a positive control with ten such validation layers and 70.87% population recovery. Layerwise validation loss is not an AUC substitute.
- Max1's Full–Reuse AUC gap is only 0.008904. It amplifies recovery percentages, but the failures are real AUC changes: C128 loses 0.047076 versus Reuse; C512 loses 0.007960. Max4 C512 loses 0.019182 with a larger 0.059537 gap.

## Strongest fitting mismatch to test

Ridge minimizes the equally weighted rate target `(teacher_read - reused_read)/N`. Physical read error instead weights the rate error by N². Some failing Max layers get worse **on their own fitting users**, despite lower ridge loss:

| Edge / budget | Layer, zero based | Fitted / zero rate MSE, train | Fitted / zero read MSE, train | Same ratio, validation read |
|---|---:|---:|---:|---:|
| Max1 C128 | 3 | 0.037 | 3.219 | 930.237 |
| Max1 C128 | 6 | 0.018 | 1.425 | 27.709 |
| Max1 C512 | 3 | 0.573 | 4.133 | 2.825 |
| Max1 C512 | 6 | 0.115 | 1.627 | 1.430 |
| Max4 C512 | 9 | 0.331 | 1.852 | 1.591 |
| Max4 C512 | 10 | 0.307 | 1.154 | 1.396 |

Training histories include lengths down to 1. Approximately 86% of Max1 calibration histories and 91% of Max4 C512 histories are full length. The aggregate losses establish a length-weighting mismatch; they do not identify particular users as the cause.

Useful counterfactuals on the same frozen users are (a) temporarily disabling the listed layers and (b) refitting the same query basis with sample weights N² / mean(N²), leaving all users and teacher queries fixed. The latter directly tests the loss weighting, without discarding short-history users.

## Near-constant ELU features

Use the prospectively proposed threshold `input_scale <= 1e-5` (implementation floor 1e-6). Counts at C128 / C512 are Medium1 0/0, Large1 31/32, Max1 10/9, Max4 16/18, Max2 18/12. All are in the ELU half. Except for four layer-11 features in Max2 C128, these are entirely in layer 0.

Therefore joint optimization changing lower-layer queries cannot explain the main layer-0 near-constant features: layer 0 has no corrected predecessor. Their number also does not distinguish success from failure. Joint fitting does amplify their weights in Max1 C512 (norm 2,494 → 11,993; effective W/scale norm 1.11e9 → 7.32e9) and Max4 C512 (3,612 → 5,732). A weight-row masking ablation can test their contribution; a large effective slope alone is insufficient because ELU's local derivative is tiny in the same saturated region.

The nonlinear features are computed in FP64 for ridge and FP32 when serving; saved means/scales are FP32. This is an additional numerical hypothesis for saturated features, not an established cause. The raw fit's FP64 and explicit FP32 training MSE are recorded separately in `analysis.json` for inspection.

## Calibration-to-serving transfer and controls

- Max1 C128/C512 select epochs 3/4; Max4 C512 selects epoch 5. Their independent pure-snapshot validation objective improves, yet rolling population AUC is negative.
- Max2 selects epoch 0 at all three budgets and remains positive (70.87%, 47.50%, 23.57%). Raw-versus-joint rolling evaluation distinguishes a preexisting fit problem from a refinement-induced change.
- This problem predates the nonlinear query basis. Old v2 raw-query Q has negative Max1 recovery at all C128/256/512/1024 budgets (−69.18%, −84.25%, −51.79%, −54.08%), and negative Max4 recovery throughout (−10.51%, −0.58%, −7.82%, −5.27%). Its C512 validation set has 64 users, so validation size 16 is not a sufficient explanation by itself.
- Old v2 Max1 C512 improves its held-out teacher objective 0.930 → 0.0655 and logit MSE 0.2047 → 0.01986, but still loses population AUC. Max4 C512 likewise improves held-out objective 0.1785 → 0.1064 but remains negative.
- Explicit-history v4 remains positive on the same population: Max1 all/old-prefix recovery 183.16%/69.48%; Max4 25.05%/16.61%. Missing user-history information in Q is therefore a plausible structural limitation, but this comparison alone does not prove an irreducible bound on all query-only methods.

## Priority for small causal probes

1. Same rolling caches, raw versus joint; fixed correction multipliers 0.25/0.5/1 and layer suppression. This tests amplitude and accumulated layer interaction without score mixing.
2. Mask the fixed near-constant feature rows, retaining Max2 and Large1 as controls. If ineffective, do not continue treating saturation as the primary explanation.
3. Refit N²-weighted ridge on identical prerelease train/validation users. Compare rate MSE, physical read MSE, and final outputs separately.
4. For a subsequent fitting probe, use aligned prerelease mixed-cache states and a mixed-cache validation objective. Keep true Full teacher terminal history and request causality fixed. This distinguishes pure-snapshot transfer failure from the query-only information limit.

These diagnostics guide counterfactuals; none authorizes discarding unfavorable edges or choosing a policy from evaluation labels.
