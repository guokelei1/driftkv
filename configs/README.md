# Experiment contracts

`configs/contracts/` is the machine-readable evidence boundary. Frozen contracts
and result contracts are immutable even after a phase is complete.

## Completed chain

- `p7_*`: N/R/F, compact manifests, Frozen Base, theta0 and H qualification.
- `f_release_chain_contract_v1.yaml` plus P8 artifacts: R0/R1/R2 and H/S.
- `p9_*`: tomography, dependency closure, legal executor, full-population cost,
  rolling quality and frontier.
- `p10_*`: sparse profiler, policy seal, same-cost gate, executor optimization
  and development full-stack freeze.
- `p11_*`: recursive version debt, all-population legal actions, frozen scheduler
  replay and recursive rolling quality.

Presence records evidence; it does not authorize rerunning or tuning the phase.

## Next contract family

The next prospective family will use `scale_*` and cover, in order:

1. P11 full-stack input seal;
2. 8L/H256/context1024 resource audit;
3. Full/Append/rolling correctness canary;
4. theta0 H scale gate;
5. R0/R1/R2 scale gate;
6. frozen action/scheduler replay;
7. fixed-count/capped-rate probe sensitivity.

Do not create one monolithic contract. Each long stage requires the preceding
seal and stopping gate.

## Retired boundaries

P5/P6 failure contracts and old Yambda audit contracts remain immutable for
audit. They may not be requalified or used to revive sampled next-listen,
neutral-readout repair, artificial K/V perturbation, or old controller claims.
