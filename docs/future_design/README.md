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

Current D2 code has a single-rank wave adapter and capacity-specific W3 resident extents. The first
D3 task is a minimal two-rank H12/W2 adapter and ordinary-DRAM benchmark, not a general exporter.
Normalization, stable hashes, and formal transaction closure follow after the mechanism is clear.

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
