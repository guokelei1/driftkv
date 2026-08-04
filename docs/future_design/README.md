# Active design documents

This directory contains the active recursive D1 design, one design/status pair for the implemented
D2 mechanisms, plus a flexible D3 direction and a foundation document that separates the
historical GPU0/GPU1 mechanism chain from the successor paper benchmark.

| Document | Role |
|---|---|
| [DESIGN1_RECURSIVE_KV_MIGRATION.md](DESIGN1_RECURSIVE_KV_MIGRATION.md) | single-serving-model premise, seven-version capacity ledger, rollout-aware RACT-KV, 10% Exact renewal, completed QK development result, diagnostic/fallback boundary, and locked QB next step |
| [DESIGN2_FINAL_PLAN.md](DESIGN2_FINAL_PLAN.md) | D2 mechanism, required `ActionPlan`→D3-facing constraint interface, baselines, timer, and formal gate |
| [DESIGN2_DEVELOPMENT_STATUS.md](DESIGN2_DEVELOPMENT_STATUS.md) | frozen inputs, implemented state, non-scientific evidence, W4/formal gaps, and D3 handoff |
| [DESIGN3_FUTURE_DIRECTION.md](DESIGN3_FUTURE_DIRECTION.md) | D3 core DRAM↔HBM problem, isolation/co-design tracks, candidate mechanisms, and later paper conditions |
| [DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md](DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md) | historical H12/QK two-rank ledger plus flexible HET/XP/rolling foundation, strongest-baseline construction, and backtracking |

The completed QK v0 semantic contract is recorded at
`configs/evokv_d1/development/recursive_large_chain_design_v0.json`; the executable binding is
`configs/evokv_d1/development/qk_recursive_round_a_two_gpu_v0.json`. Neither contains a physical
GPU timing gate.

The QK round completed at
`results/baseline_rounds/quality_chain/recursive_d1_round_a/qk_recursive_d1_round_a_20260804_round1/`.
Its 10% RACT-KV path achieves near-Exact CE and more than 94.9% mean K/V recovery on all three true
recursive edges. It is development evidence, not a formal result. The old v0 selector's
`stability_certificate` is now interpreted as a held-out rollout diagnostic, not an accuracy
theorem or current headline gate; the original `complete_no_admitted_policy` summary remains
unchanged. The next boundary is a newly frozen, locked QB confirmation protocol.

The current architecture is:

```text
D1 semantic ActionPlan
  → D2 distributed WavePlan constraints
  → D3 capacity-bounded ResidencyPlan
```

The architecture begins from one non-negotiable premise: exactly one recommendation-model version
serves at a time. Versions form a sequential update chain; `source_version`/`target_version` are
K/V lineage, not simultaneous model service. Rolling migration may temporarily retain old and new
K/V groups, but every request uses the one current model.

D1 freezes semantic actions and logical Exact valid-token work only. D2 proves whether those fixed
actions reduce real multi-GPU work and time; D3 handles capacity placement and rolling movement.

The interface above is the current paper decomposition, not a permanent exploration boundary.
Isolation-track experiments keep one D1/D2 `WorkManifest` fixed. Co-design experiments may
globally regenerate actions, owners, pools, or layout before execution, record a new
`stack_revision`, and rerun their baselines.

The next mechanism round starts from the verified local checkpoint registry, not from another
training-policy search. True recursive D1 on QK LR0.15 theta1--theta4 is complete and freezes the
10% rollout-aware system-design point; the next run confirms that method on QB `u30_e3`
theta1--theta3 without QB-specific tuning. D2/D3 performance work follows this result-dependent
boundary. Paths, hashes, rebuild commands, and cleanup are in
[../13_cross_dataset_stream_checkpoint_plan.md](../13_cross_dataset_stream_checkpoint_plan.md).

The successor paper boundary uses natural-length `X-QK-HET` as the headline workload and
same-record masked-512 `X-QK-HOM` only as a matched physical-shape control. XP fixes
2,859,835 base-period semantic rows plus one padding row in a 43.638-GiB physical FP32 table,
adds owner-side E4096→H1536 projection, and forces
distributed placement after optimizer-updated active bytes alone exceed the single-card budget;
the all-comparator request union across both formal edges must be active. The integrated action
domain is `compiled|exact`; progressive residual replay remains a D1-only supporting extension.
D3 keeps one live cache plus bounded shadow/staging and performs per-group validation, commit, and
old-group reclaim on a 1/2/4-rank-capable runner. The planned matrix is in
[../10_paper_experiment_blueprint.md](../10_paper_experiment_blueprint.md); promotion checks are
registered in [../11_benchmark_qualification.md](../11_benchmark_qualification.md) and do not
block foundation design or implementation.

Current D2 code has a single-rank wave adapter and capacity-specific W3 resident extents. Historical
D3 development has both the minimal H12/W2 adapter and a real-capacity two-rank QK M1 chain. On the fixed 288-GiB
old/private-target boundary, fair S0 is 48.238 seconds, strong S1 is 32.703 seconds, and the
historical v1 fixed-order bidirectionally segmented I/O precursor is 28.885 seconds. Under the
current exact stack/hash, route-major `(8,8,8)` takes 28.514442098 seconds and the hashed
ResidencyPlan order takes 28.147194647 seconds (1.013047x; 1.2879% lower wall time) with complete
exactly-once byte parity. The plan uses same-source joint profiles, a one-lookahead/one-drain flow
model, and a synchronized per-rank capacity preflight. It is still a development mechanism, not a general
exporter or formal result. Grouped development E0 now measures 44.639 seconds sequentially and
33.549 seconds with the strong action-oblivious two-slot baseline; owner-local naive-staged
D1-only is 57.597 seconds and the current-binary sequential D1+D2 rerun is 49.753 seconds.
Formal evaluation promotes a same-stack fixed-FIFO bidirectionally segmented path to the strongest
generic S2 baseline and also includes a profile-aware generic scheduler; route-aware D3 must beat
their winner, not merely whole-group S1. Independently tuned E0/S2/generic controls, HET/HOM,
XP, rolling lifecycle, segmented-consumer closure, formal repeats, and qualification remain open.

The H12 workload is the first semantic canary, not physical oversubscription evidence. The
fixed-512 QK old-plus-private-target point is also historical development evidence; successor
capacity is defined by one live HET cache version plus bounded group shadow, not two complete
versions. Its dedicated model checkpoint copies were retired in the 2026-08-03 storage cleanup;
compact results and reusable mechanism code remain. A software HBM cap remains development-only
capacity emulation.

The former D2 four-stage controller and Stage-A/Stage-B handoff documents were consolidated into
the D2 design/status pair. The pre-rewrite D2/D3 idea comparison was removed. Git history is not a
source of current tasks.

Complex organic mixed-version graphs and cross-wave controllers remain later directions. Per-group
rolling commit/reclaim is part of the current formal endpoint, not that future controller.
Cross-layer changes directly motivated by out-of-core profiles are allowed during D3 mechanism
discovery, but a changed stack must rerun its baselines.

When these files conflict with
[../08_core_insights_and_roadmap.md](../08_core_insights_and_roadmap.md) or
[../eval_protocol.md](../eval_protocol.md), the roadmap and protocol win.
