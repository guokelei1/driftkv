# Active design documents

This directory contains one design/status pair for the implemented D2 mechanisms, plus a flexible
D3 direction and GPU0/GPU1 benchmark-first execution plan.

| Document | Role |
|---|---|
| [DESIGN2_FINAL_PLAN.md](DESIGN2_FINAL_PLAN.md) | D2 mechanism, required `ActionPlan`→D3-facing constraint interface, baselines, timer, and formal gate |
| [DESIGN2_DEVELOPMENT_STATUS.md](DESIGN2_DEVELOPMENT_STATUS.md) | frozen inputs, implemented state, non-scientific evidence, W4/formal gaps, and D3 handoff |
| [DESIGN3_FUTURE_DIRECTION.md](DESIGN3_FUTURE_DIRECTION.md) | D3 core DRAM↔HBM problem, isolation/co-design tracks, candidate mechanisms, and later paper conditions |
| [DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md](DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md) | flexible H12/QK route, minimal two-rank adapter, GPU0/GPU1 S0/S1 benchmark, real-capacity construction, and mechanism exploration |

The current architecture is:

```text
D1 semantic ActionPlan
  → D2 distributed WavePlan constraints
  → D3 capacity-bounded ResidencyPlan
```

The interface above is the current paper decomposition, not a permanent exploration boundary.
Isolation-track experiments keep one D1/D2 `WorkManifest` fixed. Co-design experiments may
globally regenerate actions, owners, pools, or layout before execution, record a new
`stack_revision`, and rerun their baselines.

Current D2 code has a single-rank wave adapter and capacity-specific W3 resident extents. D3 now
has both the minimal H12/W2 adapter and a real-capacity two-rank QK M1 chain. On the fixed 288-GiB
old/private-target boundary, fair S0 is 48.238 seconds, strong S1 is 32.703 seconds, and the
historical v1 fixed-order bidirectionally segmented I/O precursor is 28.885 seconds. Under the
current exact stack/hash, route-major `(8,8,8)` takes 28.514442098 seconds and the hashed
ResidencyPlan order takes 28.147194647 seconds (1.013047x; 1.2879% lower wall time) with complete
exactly-once byte parity. The plan uses same-source joint profiles, a one-lookahead/one-drain flow
model, and a synchronized per-rank capacity preflight. It is still a development mechanism, not a general
exporter or formal result. Grouped development E0 now measures 44.639 seconds sequentially and
33.549 seconds with the strong action-oblivious two-slot baseline; owner-local naive-staged
D1-only is 57.597 seconds and the current-binary sequential D1+D2 rerun is 49.753 seconds.
Independently tuned formal E0, held-out qualification, formal repeats, action/capacity mixes, and
transaction closure still precede a frozen protocol.

The H12 workload is the first semantic canary, not physical oversubscription evidence. The preferred
D3 mechanism-selection route audits a larger real-history QK workload whose owner-local source
plus private target physically exceeds two A40s. A software HBM cap remains development-only
capacity emulation.

The former D2 four-stage controller and Stage-A/Stage-B handoff documents were consolidated into
the D2 design/status pair. The pre-rewrite D2/D3 idea comparison was removed. Git history is not a
source of current tasks.

Complex organic mixed-version graphs and lifecycle controllers remain later directions. Small
cross-layer changes directly motivated by the two-card out-of-core profile are allowed during D3
mechanism discovery.

When these files conflict with
[../08_core_insights_and_roadmap.md](../08_core_insights_and_roadmap.md) or
[../eval_protocol.md](../eval_protocol.md), the roadmap and protocol win.
