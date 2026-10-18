# Round 2 CPU diagnostic notes

Existing raw/main/strict/masked trajectories are matched request-for-request. No user was filtered by future labels or activity. These are exploratory development diagnostics, not a new confirmation.

## Exact encoder factorization

The encoder uses fixed coordinate centers/scales, not request-dependent normalization. Folding those constants lets the Current-item affine term be reused without changing the code function. Per-row KV + combine/SiLU costs 37,440 FLOPs; each unique item affine costs 110,880 including bias; parameter folding costs 221,184 per release. User code storage stays rank8. Shared item tables, replication, lookup and transient memory need their own accounting.

Bounded LRU quality is numerically causal because the cached affine is the same known function, but user-major cache hits are not a valid chronological serving-cost estimate. Prefer fixed top-K items selected from calibration users before publication; compute other unique items per call without admission. The cost agent owns fixed30k accounting.

## Quality boundaries

Raw main has smaller relative logit MSE than masked main yet worse AUC, especially V5. Current full-code gains must be positive before interpreting retention. Full/strict protection for short histories is plausible; long histories dominate row volume. The JSON includes all first-read length, missing/older, newly appended and delay bins, including adverse outcomes.

Whole-user full-short / partial-long mixtures in the JSON are exact compositions of retained trajectories and use prerelease length. They can screen a conditional strategy without new GPU scores; switching rho independently at each release is not equivalent to this screen. Whole-window append/refresh ratios are descriptive only and must not be used as an online selector.

## Candidate paths

1. Full refresh with factored item-affine reuse; test no global cache and fixed top-K 4096/16384/65536. This targets maintenance savings without sacrificing code quality, subject to floating-point checks.
2. Combine exact factorization with sparse encoding of rows/layers whose inherited support is true; disabled rows need no current correction. Charge later forced regeneration when discarded outputs become necessary.
3. Protect prerelease history <=768 with full refresh; use .5/.25 or .5/.5 on longer histories, keeping all users. Lower-rho quality is unmeasured; fit/validation and held-out scores must decide.
4. Refresh by query-independent impact from old compact codes under new versus previous projections, with a small reserved tail. This only scans rank8 state; do not encode all new rows to pick the survivors.
5. Warm-start or constrain encoder coordinates across releases, then test aggressive tail refresh. Code-coordinate stability must be fitted on calibration roles, not selected by held-out labels.

All candidates retain missing/older mandatory refresh and report the effective union, not nominal rho alone. Negative previous outcomes stay indexed. Neither a better teacher MSE nor a favorable cohort point proves the requested paired UID lower bound.

## Lifecycle scope metadata, 2026-10-14

[lifecycle_scope_metadata.json](lifecycle_scope_metadata.json) reads only the existing3360-user panel's event and request timestamps, using the current inherited-support recurrence. Of69,361 requests,3545(5.11094%;301 distinct users) have no direct pre-release producer rows but retain possible upper-layer inherited dependence. No request has all layers clear. This establishes occurrence of mixed causal scope; it does not measure numerical error, AUC benefit, or natural-retirement savings. All users/releases remain included; no model inference, training, outcome-based filtering or parameter selection was performed. The JSON retains the exact formula, per-release counts and input/source hashes.

## Broader design discussion: coordinate compatibility

[coordinate_rotation_counterexample.json](coordinate_rotation_counterexample.json) retains a small CPU algebraic example: applying the same orthogonal rotation to Q and K preserves the attention function, yet naive reuse of unrotated K changes scores. Moving the query back to the cache coordinate system restores the original scores. The induced Key delta is rank32 and cannot be represented exactly for all queries by rank8 codes and a linear Key decoder. This is a structural counterexample, not a real-checkpoint/AUC result or proof that coordinate drift dominates current releases. It motivates investigating explicit coordinate compatibility versus genuine context change; no new training or GPU experiment was launched.
