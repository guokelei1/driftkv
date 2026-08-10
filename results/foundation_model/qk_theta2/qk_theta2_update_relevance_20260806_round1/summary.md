# QK theta2 update-relevance round

Status: `complete_no_preferred_candidate`

| Candidate | All NDCG gap | All MRR gap | Best predeclared relation | Relation NDCG gap | Admitted |
|---|---:|---:|---|---:|---|
| theta2_route_a_e3_lr100 | -1.0272% | -0.7732% | boundary_multi_positive/context_h16_support_ge1 | -0.2820% | False |
| theta2_route_a_e4_lr100 | -1.3699% | -0.3050% | boundary_multi_positive/context_h16_support_ge1 | 1.6565% | False |
| theta2_route_a_e3_lr150 | -1.7192% | -0.8744% | boundary_multi_positive/context_h16_support_ge1 | 0.7316% | False |
| theta2_relevance_e1_lr100_n8 | -1.2114% | -1.9617% | rolling_next_item/context_h16_support_ge1 | -0.8684% | False |
| theta2_relevance_e2_lr075_n8 | 0.5216% | -0.0739% | rolling_next_item/context_h16_support_ge1 | 1.3128% | False |
| theta2_relevance_e2_lr075_n32 | 1.4348% | 0.3291% | rolling_next_item/context_h32_support_ge1_offset_lt_16 | 1.5961% | False |

All cohorts were frozen from window2 identities, labels, and ordinals before reading any window3 quality value.
Candidate-set protocols are supporting diagnostics; only the full-catalog relation gate can stop fallback training.
