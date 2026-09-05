# Medium user attention-cone response moments: canary

The full row is an Exact-state representation oracle. Compact rows keep the complete Parent moment but estimate the Current moment from Exact upper-layer samples; they are constructor oracles, not executable migration actions.

| method | Current samples | edge-equal recovery | minimum edge | positive edges | >=80% edges | persistent KV ratio | temporary Current-sample ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_cone_moment | 1024 | 0.9957 | 0.9892 | 5/5 | 5/5 | 1.6113% | 100.0000% |
| chronological_R8 | 8 | -19.3219 | -48.5466 | 0/5 | 0/5 | 1.6113% | 0.7812% |
| chronological_R16 | 16 | -9.4892 | -15.2580 | 0/5 | 0/5 | 1.6113% | 1.5625% |
| chronological_R32 | 32 | -10.5540 | -29.8114 | 0/5 | 0/5 | 1.6113% | 3.1250% |
| chronological_R64 | 64 | -4.6511 | -9.9453 | 0/5 | 0/5 | 1.6113% | 6.2500% |
| chronological_R128 | 128 | -2.9455 | -5.4618 | 0/5 | 0/5 | 1.6113% | 12.5000% |
| address_R8 | 8 | -8.9061 | -14.9726 | 0/5 | 0/5 | 1.6113% | 0.7812% |
| address_R16 | 16 | -4.7007 | -8.7100 | 0/5 | 0/5 | 1.6113% | 1.5625% |
| address_R32 | 32 | -2.0731 | -6.5970 | 0/5 | 0/5 | 1.6113% | 3.1250% |
| address_R64 | 64 | -0.7329 | -3.0125 | 2/5 | 0/5 | 1.6113% | 6.2500% |
| address_R128 | 128 | 0.2733 | -0.5077 | 4/5 | 0/5 | 1.6113% | 12.5000% |

## Adjudication

- Full attention-cone representation gate: PASS; recovery 0.9957, minimum edge 0.9892.
- Address-R128 canary gate: FAIL; recovery 0.2733, advantage over chronological 3.2188.
- Canary-to-discovery launch: FAIL.
- Sampled-moment 80% gate: FAIL; smallest passing address R: None.
- Interpretation: `cone_moment_representation_supported_sampled_constructor_rejected`.
- A passing full moment establishes only a functional representation. Exact sampled Current upper-layer state prevents every compact row from admitting Design 1.
- No labels, output fitting, ridge/MLP, confirmation users or executable-compute claims are used.

## Cone diagnostics (row-equal over user/layer/head)

- `current_negative_activation_fraction`: 0.038654
- `current_negative_response_fraction`: 0.037583
- `current_parent_sign_crossing_fraction`: 0.082924
- `heldout_current_majority_agreement`: 0.994453
- `heldout_parent_majority_agreement`: 0.994460
- `parent_negative_activation_fraction`: 0.044827
- `parent_negative_response_fraction`: 0.043663
- `qk_abs_p50`: 157.777427
- `qk_abs_p95`: 298.012003
