# CohortKV Stage 2 deployed compiler certificate

## Status

Complete under `cohortkv_single_config_stage2_compiler_v1` and frozen by
`cohortkv_single_config_stage2_frozen_v1`.

This is adaptive seed-0, single-configuration development evidence. It closes the executable
compiler artifact and deployed-representation certificate path. It does not evaluate the 522
final-test users, execute the 682-record mixed-version job, or establish an end-to-end speedup.

## Frozen configuration

- KuaiRand 4+12, theta0/theta4/theta10 to theta11;
- 16 layers, hidden/K/V width 512, maximum sequence length 2,048;
- training seed 0;
- disjoint 40 fit / 60 program-selection / 60 certificate / 522 final-test roles;
- attention mix 1.0, ridge 0.001, and 8,192 sampled fit tokens per layer;
- actions: cheap projection, compiled full affine, residual p4, residual p8, and exact;
- primary contract: 70% recovery, 80% coverage, 90% one-sided confidence, and cost at most
  0.30× exact;
- recovery-target sensitivity: 50%, 60%, 70%, 80%, and 90%;
- three timing repeats, batch size four, and 1,000 bootstrap samples.

The certificate uses no recommendation labels. Each source pair uses the same 60 certificate
records and 95,660 prefix tokens. The 522 final-test records are reconstructed only to audit the
role split and are never forwarded through a model.

## Artifact path

For every source/target pair, the compiler:

1. validates the prepared data, four checkpoints, selected FP32 full-affine source program,
   workload manifest, model signature, and hashes;
2. writes and reloads one contiguous FP16 runtime program;
3. materializes temporary certificate shards on the declared `/data` ext4 device;
4. reloads and hash-validates every shard;
5. evaluates cache, full-catalog score, and top-100 views against FP32 current-model exact K/V;
6. writes a checked executable JSON plan with an ordered exact-terminated fallback chain; and
7. deletes the temporary certificate shards.

The three runtime programs are local checkpoint artifacts of 16,813,213, 16,813,213, and
16,813,221 bytes. Their combined size is 50,439,647 bytes. The checked-in plans retain paths,
hashes, shapes, dtypes, source/target versions, provenance, certificates, threshold sensitivity,
and source-state requirements. Loading a plan revalidates every frozen input hash and the runtime
program before returning the action chain.

## Numeric representation correction

The initial contract proposed FP16 for the optional residual hidden suffix. Real certificate
materialization rejected that representation: unnormalized old hidden states reach absolute
maxima of 7.38–19.58 million, well above the FP16 finite maximum of 65,504. Across the three
counterfactual source-pair certificate shards, 176.8 million, 339.2 million, and 320.8 million
values would overflow.

The residual auxiliary state therefore uses BF16. BF16 preserves the two-byte storage accounting
while providing sufficient exponent range. The primary normalized capsule, compiled program, and
published K/V remain FP16. This changes neither the compiler's logical goal nor its selected
compiled path; it is a measured representation correction required to keep the residual fallback
executable.

## Primary certificate result

| Source | Selected action | Ordered fallback | Resident cost / exact | Cache recovery | Score recovery | Top-100 recovery | Worst recovery lower bound | Worst coverage lower bound |
|---|---|---|---:|---:|---:|---:|---:|---:|
| theta0 | compiled full affine | p8 → exact | 0.01657 | 0.8810 | 0.9845 | 0.9479 | 0.8514 | 0.9224 |
| theta4 | compiled full affine | exact | 0.01652 | 0.8897 | 0.9201 | 0.9046 | 0.8391 | 0.9005 |
| theta10 | compiled full affine | p8 → exact | 0.01651 | 0.9365 | 0.9717 | 0.9470 | 0.9231 | 0.9459 |

All three deployed certificates pass. The cost ratio is the resident GPU migration component on
reloaded serialized inputs. It excludes source reads and destination publication and must not be
reported as Stage-4 end-to-end cost.

The lower resident ratio than the Stage-1 FP32 frontier is expected: Stage 2 measures the packed
FP16 compiled runtime program against FP32 current-model exact compute, with both paths casting
their output to FP16 inside the timer. The semantic values are comparable because both stages use
the same frozen certificate views, but the cost boundaries and numeric paths must remain named.

## Threshold sensitivity

Compiled full affine remains selected for all three source pairs at recovery targets 50%, 60%,
70%, and 80%. At 90%, every pair selects exact recomputation. Thus the primary 70% setting is
inside the observed stable region rather than the only threshold at which the compiled action
passes.

Fallback membership changes with the threshold because it records every stronger certified
action, including actions above the primary 0.30× cost budget. Under the primary contract,
theta0 and theta10 retain p8 before exact; theta4 proceeds directly to exact.

## Compiler and certificate cost

| Component | Three-pair summed work |
|---|---:|
| Historical FP32 fit | 31.243 s |
| FP16 runtime-program preparation | 4.316 s |
| Deployed certificate | 273.343 s |
| Full-catalog scoring within the certificate | 0.756 s |
| Summed one-time work amortized over 682 records | 0.453 s/record |

The per-pair resident break-even floors are 2,936, 2,865, and 2,872 records for
theta0/theta4/theta10. These values include historical fit, runtime-program preparation, and
certificate work, divided by resident compiled-versus-exact time saved. They are not end-to-end
break-even points. Stage 4 must recompute amortization after including source materialization,
source reads, destination writes, and manifest commit.

The aggregate `0.453 s/record` sums the three version-pair work values; it is not the elapsed time
of the three-worker Stage-2 run. An idealized fully parallel critical path from the same per-pair
components is `104.306 s`, or `0.153 s/record` at 682 records. Neither number substitutes for the
Stage-4 measured compiler wall time and end-to-end crossover.

Each temporary 60-record certificate source is 6,092,751,520 physical bytes and includes all
representations needed to verify the complete action library. The temporary files are deleted
after verification and are not the Stage-4 full-cohort source layout.

## Commands

```bash
python scripts/compile_cohortkv_stage2.py --validate-only
torchrun --standalone --nproc-per-node=3 scripts/compile_cohortkv_stage2.py
python scripts/freeze_cohortkv_stage2.py
python scripts/freeze_cohortkv_stage2.py --check
python scripts/freeze_cohortkv_single_config_v1.py --check
pytest -q tests/test_migration_artifacts.py tests/test_single_config_stage2.py
```

## Tracked evidence

- `configs/cohortkv_single_config_v1/stage2_compiler_summary.json`
- `configs/cohortkv_single_config_v1/stage2_plans/theta0_to_theta11_executable.json`
- `configs/cohortkv_single_config_v1/stage2_plans/theta4_to_theta11_executable.json`
- `configs/cohortkv_single_config_v1/stage2_plans/theta10_to_theta11_executable.json`

The raw per-user result and runtime `.pt` programs remain local and ignored. The frozen summary
records their hashes and preserves the parent-blueprint hash used by the measurement.

The freeze check independently rebuilds every action summary and all 15 threshold certificates
from the 180 raw per-pair certificate records. Plan loading verifies the plan's own hash, complete
certificate/action coverage, selected-certificate identity, runtime-program hash, and every frozen
input hash. This audit leaves all frozen decisions unchanged.

## Downstream boundary

Stage 2 is complete because all three source pairs have hash-checked runtime programs,
deployed-representation certificates, threshold sweeps, measured one-time costs, and directly
loadable plans. Stage 3 now owns the common reference/packed/fused operator contract. Stage 4 owns
automatic fallback dispatch, full-cohort source streaming, destination publication, and
end-to-end amortization.
