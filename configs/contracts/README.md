# Contract registry boundary

Contracts in this directory are immutable evidence or prospective records; a
filename does not authorize execution.

- Historical numbered P7--P11, CC and 8L contracts were removed together with
  their old execution paths. They are not supported evidence inputs for the
  current motivation.
- `yambda500m_scale_population_v1.yaml` freezes the active label-free Medium/
  Large UID selection and foundation-only compact mapping boundary; it does not
  authorize training or theta3 access.
- `yambda500m_streaming_windows_v1.yaml` defines composable daily slices and
  calendar-only release capacity. Locked theta3/later slots are not executable
  qualification contracts.
- `yambda500m_unified_scales_v1.yaml` defines the current one-pass shared
  preprocessing route for nested Yambda-500M S/M/L logical datasets.
- `yambda500m_small_foundation_chain_v1.yaml` freezes Small foundation
  data/model/cache/metric semantics. It authorizes implementation and correctness
  canaries, not real Base fitting or HSTU training.
- `yambda500m_small_seed17_launch_v1.yaml` is the user-authorized execution
  record for the four-rank Small seed17 `v0 -> v1` chain, with an optional
  same-recipe `v2` extension. It does not unlock v3/theta3 or M/L.
- New release-chain contracts must separate candidate training, Full-only
  admission and accepted-release sealing. Admission may not read Reuse or any
  cache-compatibility artifact; rejected candidates do not enter lineage.
- New active contracts must use the EvoKV-HSTU-S/M/L naming and follow
  `docs/experimental_design.md`.

Do not edit old hashes to make them appear current. Retirement and replacement
are recorded in route documentation and new contracts.
