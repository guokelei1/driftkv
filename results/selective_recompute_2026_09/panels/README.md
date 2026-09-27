# Fixed users for selective-recomputation diagnostics

These panels answer: **among users selected for existing Full-versus-Reuse
ranking loss, how much of that loss can partial recomputation recover?** They
are explicitly outcome-conditioned diagnostics, not random population samples
or new estimates of population cache harm. The complete 15-edge population
Full/Reuse experiment remains in `results/unified_reuse_2026_09/`.

Each of the 15 scale/edge directories contains 3,000 evaluation users, 32
calibration users, and 16 canary users. The three user sets are disjoint within
each edge. All four baselines and every budget must use the same frozen
evaluation requests, including every retained scored request for each selected
user. Edges can use different user sets.

## One-pass selection rule

1. Sort source-population UIDs by
   `SHA256("selective_recompute_2026_09/user_reservation_v1/" + str(uid))`.
   Reserve the first 32 for calibration and the next 16 for canaries, without
   consulting labels or model scores.
2. On the remaining source requests, compute each request's fraction of
   correctly ranked opposite-label requests separately under Full and Reuse;
   tied scores receive half credit. Use sigmoid probabilities, matching the
   existing metric implementation.
3. Multiply each request's Full-minus-Reuse concordance difference by
   `N / (2 * N_class)`. The mean of these values over the remaining population
   exactly equals its Full-minus-Reuse AUC difference.
4. Average the values within each user. Select the 3,000 highest-scoring users,
   breaking ties by the same UID hash and then UID. This is a user ranking
   against the population's opposite-label requests, not a personal AUC;
   users with only one label remain eligible.
5. Freeze the selected UIDs and all their request IDs. Recompute subset Full
   and Reuse metrics once from retained logits. Do not redraw panels, adjust
   the recipe after viewing their gaps, or use new baseline outcomes to choose
   users or requests.

Ranking users by observed loss can increase the subset gap substantially; its
size must not be interpreted as a population estimate. This selection does not
mathematically guarantee positive AUC inside the selected subset, because its
comparison population changes. Every one-pass result is retained regardless of
the observed direction.

## Files and checks

- `users.json`: frozen `evaluation`, `calibration`, and `canary` UID lists.
- `{split}_requests.parquet`: request identity, item index, label, existing
  Full/Reuse logits, historical/cache lengths, append count, and evictions.
- `{split}_users.parquet`: user request/label counts, timestamps, history
  extrema, append/eviction maxima, and selection score where defined.
- `binding.json` and `summary.json`: selection recipe, source/output SHA256,
  source-population and subset metrics, coverage, and preparation source hashes.

Preparation verifies retained Full raw seals, every referenced Reuse shard,
request/UID/timestamp agreement, label agreement, and reproduction of both
population AUCs. Checkpoint hashes are inherited from the completed Reuse
binding; GPU execution preflight verifies the actual retained weights once.
Original evidence is read only. Runtime estimates must use these panels'
actual request and history counts, not just their 3,000-user size.

```bash
python scripts/selective_recompute_2026_09/prepare.py --all
python -m pytest -q tests/selective_recompute/cohort
```

Preparation is CPU-only and does not launch baseline evaluation. Repeating it
validates and reuses existing frozen panels. A different recipe cannot overwrite
an existing binding.

## Frozen panels, 2026-09-26

The single pass produced all 15 panels, each with exactly 3,000 evaluation users.
Together they contain 401,187 evaluation requests. Every panel has Full AUC
above Reuse AUC. These are selected-subset differences; they are not comparable
to population gaps as unbiased effect-size estimates. Max V0→V1 has the smallest
selected gap, 0.890 AUC percentage points; no reselection was performed.

| Scale | Edge | Requests | Full AUC | Reuse AUC | Gap (AUC pp) |
|---|---|---:|---:|---:|---:|
| medium | v0_to_v1 | 26,014 | 0.614560 | 0.495560 | 11.900 |
| medium | v1_to_v2 | 26,001 | 0.508618 | 0.425697 | 8.292 |
| medium | v2_to_v3 | 27,313 | 0.533160 | 0.450283 | 8.288 |
| medium | v3_to_v4 | 24,155 | 0.581778 | 0.388368 | 19.341 |
| medium | v4_to_v5 | 29,860 | 0.617133 | 0.514210 | 10.292 |
| large | v0_to_v1 | 23,450 | 0.606178 | 0.524610 | 8.157 |
| large | v1_to_v2 | 25,975 | 0.567562 | 0.501922 | 6.564 |
| large | v2_to_v3 | 27,553 | 0.578361 | 0.464884 | 11.348 |
| large | v3_to_v4 | 29,546 | 0.652913 | 0.481188 | 17.172 |
| large | v4_to_v5 | 23,880 | 0.592276 | 0.423122 | 16.915 |
| max | v0_to_v1 | 24,626 | 0.628986 | 0.620082 | 0.890 |
| max | v1_to_v2 | 26,163 | 0.584946 | 0.520352 | 6.459 |
| max | v2_to_v3 | 25,140 | 0.678007 | 0.508057 | 16.995 |
| max | v3_to_v4 | 33,512 | 0.624337 | 0.564800 | 5.954 |
| max | v4_to_v5 | 27,999 | 0.648943 | 0.504078 | 14.486 |
