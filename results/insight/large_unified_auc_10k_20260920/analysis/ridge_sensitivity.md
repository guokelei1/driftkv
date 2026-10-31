# Large pilot ridge sensitivity

The fixed probe fits 256 teachers with 16 queries each and evaluates the same 96 pilot users across all five releases. All 30 ridge/arm/release results are retained in `ridge_sensitivity.csv`. These are pilot observations; the main 10,000-user experiment retains ridge 0.01.

AUC uses pooled ranks of raw logits. Unclipped recovery is `100*(AUC_method-AUC_Reuse)/(AUC_Exact-AUC_Reuse)` under the frozen positive-gap floor of 0.0001; clipped recovery is its [0,100] display value. All five pilot gaps exceed the floor.

| Arm | Alternative ridge | Higher / tied / lower AUC versus 0.01 | Mean AUC difference (percentage points) | Range |
| --- | ---: | ---: | ---: | ---: |
| shared | 0.001 | 2 / 1 / 2 | -0.00997 | -0.04713 to +0.01643 |
| shared | 0.1 | 4 / 0 / 1 | +0.06628 | -0.06409 to +0.21995 |
| personal | 0.001 | 2 / 0 / 3 | -0.03111 | -0.21360 to +0.15020 |
| personal | 0.1 | 1 / 0 / 4 | -0.12934 | -0.38832 to +0.04985 |

Neither alternative improves both arms consistently across releases. Shared correction at 0.1 improves four of five pilot AUCs, with a mean increase of 0.06628 percentage points; personalized correction at 0.1 declines on four of five releases. The 0.001 setting has mixed outcomes for both arms.

Adding the user's native response improves AUC on four of five releases at the fixed 0.01 ridge. The mean increase over shared-only correction is 1.10895 AUC percentage points; the per-edge changes are +0.03107, +1.44540, +1.92757, +2.32144 and -0.18074 points. At ridge 0.001, conditioning improves four of five releases (+1.08781 points on average); at 0.1, it improves three of five (+0.91332 points on average).

The 2,142 pilot feedback requests contain 124 negatives in total (17–31 per release). Small rank changes therefore produce visible recovery changes. Negative recoveries and recoveries above 100% remain in the unclipped CSV column.

Verification checked the frozen settings/source hashes, teacher and pilot UID separation, ten sequentially fitted layers per rule, request and label identity, raw/quality/calibration seals, and independently recomputed all 45 AUC values from 19,278 raw score values. Original calibration and raw files were unchanged.

For V0→V1, both 0.01 probe rules also match `expanded_canary/rules.pt` at the 256-teacher budget exactly: every parameter tensor in all ten layers has zero difference. This checks the actual Large-model inputs in addition to the algebraic CPU reference.

Source summary SHA256: `a110d52f782276d8f5325143affa41a3ab5ce757b46b452bb52457b917b731fa`.
CSV SHA256: `e9bd94ec334f1a27bb43f0378c065deda63781d9a11c63bc172340f9b7f6f979`.
