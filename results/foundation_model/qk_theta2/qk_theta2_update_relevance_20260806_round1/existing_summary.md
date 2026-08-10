# QK theta2 update-relevance round

Status: `no_existing_admission_continue_fallback`

| Candidate | All NDCG gap | All MRR gap | Best predeclared relation | Relation NDCG gap | Admitted |
|---|---:|---:|---|---:|---|
| theta2_route_a_e3_lr100 | -1.0272% | -0.7732% | boundary_multi_positive/context_h16_support_ge1 | -0.2820% | False |
| theta2_route_a_e4_lr100 | -1.3699% | -0.3050% | boundary_multi_positive/context_h16_support_ge1 | 1.6565% | False |
| theta2_route_a_e3_lr150 | -1.7192% | -0.8744% | boundary_multi_positive/context_h16_support_ge1 | 0.7316% | False |

All cohorts were frozen from window2 identities, labels, and ordinals before reading any window3 quality value.
Candidate-set protocols are supporting diagnostics; only the full-catalog relation gate can stop fallback training.
