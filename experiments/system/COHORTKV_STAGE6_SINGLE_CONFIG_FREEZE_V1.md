# CohortKV Stage 6 single-configuration freeze v1

## Scope

Status: complete on 2026-07-28.

Stage 6 is a CPU-only assembly and validation pass over the frozen
KuaiRand-1K seed-0 single-configuration evidence. It does not rerun the
Stage-1 frontier, Stage-2 compiler, Stage-3 operator, Stage-4 system
matrix, Stage-4.5 hot-HBM jobs, Stage-4.6 fixed-history lifecycle, or
Stage-4.9 corrected growing-history confirmation.

The deployed lifecycle candidate is
`staggered_renewal_h12`. It was retained before formal confirmation as
the bounded-renewal candidate. `token_debt_total10` remains the measured
cost endpoint and is not promoted to the deployed policy. Recommendation
labels do not select either action.

## Inputs

The assembler verifies the path, protocol, status, byte size, and
SHA-256 of every Stage-1 through Stage-5 input. It additionally verifies:

- all 177 Stage-1 frontier points and the three frozen profiled
  selective actions;
- the three Stage-2 deployed certificates;
- Stage-3 transport correctness;
- all 30 Stage-4 method/destination/GPU-count points;
- the Stage-4.5 direct-old-K/V and Stage-4.6 lifecycle frozen summaries;
- both Stage-4.9 raw candidates and their same-device aggregate;
- the selected Stage-4.9 candidate against the Stage-5 formal binding;
- the Stage-5 copy-on-write closure with the existing cross-field
  validator;
- the artifact-derived source-state accounting table.

Missing or changed input artifacts stop assembly. The final aggregate is
validated by JSON Schema and an additional whole-aggregate semantic
validator.

## Stage-4.9 boundary

The corrected growing-history result is a device-resident retained-prefix
`U/E` result. The evaluator uses groupwise CPU staging for single-GPU
memory containment and reports H2D/D2H movement separately. Stage 6
therefore freezes neither a full-cohort HBM-resident 11-edge claim nor an
end-to-end state-movement claim.

The fixed-history depth-four guarantee remains a Stage-4.6 result.
Stage-4.9 H12 has a renewal horizon of 12, while the measured trajectory
contains 11 updates. The paper must disclose that a complete 12-edge
renewal cycle has not yet been observed.

## Outputs

The assembler publishes the final aggregate last, after atomically
publishing:

- a correctness report;
- a timing and memory report;
- paper table data;
- paper figure data;
- an artifact-to-claim map;
- a negative-results log;
- a current-manuscript disposition;
- a code-snapshot manifest.

The final aggregate is
`results/system/cohortkv_single_config_full_chain_v1/final_summary_seed0.json`.
All outputs remain single-seed adaptive development evidence.

The formal build verifies 18 source artifacts, all eight sidecar descriptors,
the selected H12 binding, the Stage-5 closure, the amended JSON Schema, and
the whole-aggregate semantic contract. All checks pass. After the evidence-bound
manuscript rewrite, the manuscript disposition contains zero `TBD` markers.
Open Stage-7 replication and optional post-v1 work remain explicit prose
limitations rather than placeholders or seed-0 substitutions. This refresh is
CPU-only and does not rerun any GPU matrix.

The final aggregate SHA-256 is
`fc94100fbcda56b6e8f2b663d28e02924670710003a760948eef6f7c766e403e`.
The bound formal Stage-5 artifact SHA-256 is
`b700edac86cb8452dd3e844ab2664243846814e68abfcf2924dbf7c749ffb981`.

## Commands

Generate the frozen outputs after the formal Stage-5 artifact exists:

```bash
python scripts/freeze_cohortkv_stage6.py
```

Verify the complete deterministic output set:

```bash
python scripts/freeze_cohortkv_stage6.py --check
```
