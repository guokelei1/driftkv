# Archived: P11.1 — recursive population/version-debt execution

P11.0 established on 32 users that a true rolling lineage differs from both a
one-hop state and the non-deployable shortcut of recomputing the edge-2 prefix
under theta0. P11.1 freezes and scales that comparison to all 8,229 users shared
by the edge-1 and edge-2 materialized populations.

For every M0-F/M1 model and seed 17/37/71, the executor:

1. materializes theta0 K/V once at edge 1;
2. appends every intervening event under theta1, evicting before each append at
   the 512-token cap;
3. evaluates that recursive state under theta2;
4. replays the already frozen legal actions under theta2;
5. writes only per-user target-free fidelity metrics.

The `direct_age2` condition remains a diagnostic comparison and may not be
described as deployable lineage. Qualification labels, theta3 and all scheduler
or action retuning remain prohibited.

The 32-user batched canary matched the earlier single-user implementation with
maximum per-user MSE difference `2.61e-8` and maximum JS difference `7.27e-10`.
On that canary, recovery from recursive No-op was 98.6% for Layer0-Full, 86.1%
for Layer0-Middle, and about 47%–48% for the two legal Tail-128 actions. These
are implementation/development observations; the six-cell population result is
not adjudicated until every raw artifact is sealed.
