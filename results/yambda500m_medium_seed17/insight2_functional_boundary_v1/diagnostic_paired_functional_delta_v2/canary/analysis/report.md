# Medium paired functional-delta canary

The primary Design contrast is `native_causal_closure_R64` versus `native_parent_conditioned_R64`. They share the model, users, carrier positions, masses, Parent control and native serving reader; only causal propagation of earlier functional deltas changes.

| method | evidence type | edge-equal recovery | minimum edge | positive edges | >=80% edges | incremental KV ratio | maximum total compute |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| carrier_oracle_native_R64 | carrier_state_oracle | -0.4532 | -1.9112 | 2/5 | 0/5 | 6.2609% | n/a |
| carrier_oracle_native_R128 | carrier_state_oracle | 0.3719 | -0.1914 | 3/5 | 2/5 | 12.5217% | n/a |
| closure_affine_compiler_R64_P8 | legal_affine_compiler_ablation | -0.1853 | -1.0733 | 2/5 | 0/5 | 1.6113% | n/a |
| closure_affine_compiler_R128_P8 | legal_affine_compiler_ablation | 0.3689 | -0.1107 | 3/5 | 2/5 | 1.6113% | n/a |
| native_parent_conditioned_R64 | legal_independent_ablation | -0.3431 | -1.1895 | 2/5 | 0/5 | 6.2609% | 17.1901% |
| native_parent_conditioned_R128 | legal_independent_ablation | 0.2454 | -0.2186 | 4/5 | 0/5 | 12.5217% | 27.4077% |
| native_exact_layer0_closure_R64 | legal_layer0_consistency_ablation | -0.3765 | -1.6894 | 2/5 | 0/5 | 6.2609% | n/a |
| native_causal_closure_R64 | legal_recursive_candidate | -0.1962 | -1.0764 | 2/5 | 0/5 | 6.2609% | 17.5871% |
| native_causal_closure_R128 | legal_recursive_candidate | 0.3703 | -0.1074 | 3/5 | 2/5 | 12.5217% | 29.0053% |
| representation_full_affine_bulk_P8 | representation_oracle | 0.9951 | 0.9873 | 5/5 | 5/5 | 1.6113% | n/a |
| representation_full_affine_bulk_P32 | representation_oracle | 0.9949 | 0.9858 | 5/5 | 5/5 | 1.6113% | n/a |
| Current_Reuse | serving_baseline | 0.0000 | 0.0000 | 0/5 | 0/5 | 0.0000% | n/a |

## Adjudication

- Functional representation: PASS.
- Exact-carrier oracle: FAIL.
- R64 closure quality: FAIL; recovery -0.1962.
- Causal-closure mechanism: FAIL; gain over matched independent carriers 0.1468, wins 4/5 edges.
- Paired user-edge bootstrap: 95% CI [-0.0060, 0.3834] from 10000 fixed-seed resamples.
- Honest R64 compute: PASS; neural 14.3676%, selection 3.2196%, total 17.5871%.
- Design-candidate gate: FAIL.
- Interpretation: `functional_representation_supported_constructor_not_admitted`.
- R128 is diagnostic and cannot be selected even if its quality is higher.
- The affine compiler may reduce persistent storage, but moments, sampling and landmark selection are not treated as novelty.
- Passing this canary does not freeze Design 1 or authorize discovery; a resource estimate and a new prospective contract remain required.
