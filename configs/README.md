# Experiment contracts

`contracts/` is the machine-readable evidence boundary. A result contract records what happened; it does not authorize later stages unless the current route says so.

## Active chain

- `p7_*`: N/R/F workload, compact manifests, Frozen Base, theta0 training, and one-time H qualification.
- `f_release_chain_contract_v1.yaml`: frozen F R0/R1/R2 release definitions and admission rules.
- P8 result/seal contracts: frozen development H/S evidence; the underlying models and releases must not be tuned further.
- `p9_tomography_contract_v1.yaml`: frozen P9 scope, GPU allowlist, diagnostic interventions, mandatory quality companions, and authorization gates.

P9.0-P9.2 are complete. The immediate work is P9.2 quality-companion closure and risk concentration, followed by the preselected P9.3 semantic cells. Do not mutate a frozen contract after observing scores; create a versioned prospective addendum when a new executable action is ready.

## Retired boundaries

P5/P6 next-listen identifiability contracts and their failed gates remain auditable but may not be requalified. Older Yambda contracts preserve implementation/invalidation history only. Deleted D1/D2/D3, foundation, root-cause, and CohortKV configurations must not be reconstructed.
