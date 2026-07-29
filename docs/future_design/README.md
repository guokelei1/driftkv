# Active design documents

This directory now contains one contract and one status ledger for the implemented D2 mechanisms,
plus the D3 problem contract and its executable foundation/exploration plan.

| Document | Role |
|---|---|
| [DESIGN2_FINAL_PLAN.md](DESIGN2_FINAL_PLAN.md) | D2 mechanism, required `ActionPlan`→D3-facing constraint interface, baselines, timer, and formal gate |
| [DESIGN2_DEVELOPMENT_STATUS.md](DESIGN2_DEVELOPMENT_STATUS.md) | frozen inputs, implemented state, non-scientific evidence, W4/formal gaps, and D3 handoff |
| [DESIGN3_FUTURE_DIRECTION.md](DESIGN3_FUTURE_DIRECTION.md) | D3 `WavePlan` constraints→`ResidencyPlan` problem, source/capacity contract, candidate mechanisms, baselines, and go/no-go |
| [DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md](DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md) | two-layer H12/QK foundation, two-A40 physical out-of-core benchmark, staged implementation, strong baselines, candidate ladder, and backtracking gates |

The current architecture is:

```text
D1 semantic ActionPlan
  → D2 distributed WavePlan constraints
  → D3 capacity-bounded ResidencyPlan
```

D2 and D3 never reselect D1 actions. D3 may cut a compatible D2 bin/pool into smaller
capacity-safe slices, but it must preserve owner, operator, membership, collective dependencies,
segmented layout, coverage, and lineage.

The interface above is the target architecture, not a claim that its handoff artifact already
exists. Current D2 code has a single-rank wave adapter and capacity-specific W3 resident extents.
The first D3-readiness task is to export their capacity-independent constraints, validate runtime
parity, serialize them, and assign a stable content hash. Scheduler comparisons start only after
that closure.

The H12 workload is the first semantic canary, not physical oversubscription evidence. The preferred
D3 mechanism-selection route audits and freezes a larger real-history QK workload whose
owner-local source plus private target physically exceeds two A40s. It becomes F1 only after that
capacity gate passes. A software HBM cap remains development-only capacity emulation.

The former D2 four-stage controller and Stage-A/Stage-B handoff documents were consolidated into
the D2 design/status pair. The pre-rewrite D2/D3 idea comparison was removed. Git history is not a
source of current tasks.

Organic mixed-version graphs, program composition, communication-aware semantic selection, and
cross-update renewal control are later feedback directions. They are not current D3.

When these files conflict with
[../08_core_insights_and_roadmap.md](../08_core_insights_and_roadmap.md) or
[../eval_protocol.md](../eval_protocol.md), the roadmap and protocol win.
