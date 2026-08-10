# KuaiRand true next-item architecture canary

Status: completed development canary; `scientific_result=false` and `formal_result=false`.
Checkpoint payloads were retired after recording the compact result.

The predecessor H512 architecture screen was stopped after one epoch. It combined an appended
all-zero query, ReLU pointwise attention, and a SiLU gate. The query could not consume history:
`q=0`, `ReLU(qk)=0`, and `SiLU(gate(0))=0`. Both 8L and 16L therefore produced the exact constant
training loss `log(65)=4.174387`. It was an invalid next-item foundation.

The replacement predicts item `i_(t+1)` from the hidden state of the real latest history item
`i_t`. Reuse caches only `i_1...i_(t-1)` under the old model and computes `i_t` under the current
model. The two canaries used the same 32 users, natural 2022-04-22 update, unseen 2022-04-23
evaluation, H512, T512, one base epoch, one update epoch, and 49 frozen negatives.

| architecture | Fresh MRR | Fresh NDCG@5 | Reuse NDCG@5 | NDCG@5 gap | MRR gap | history NDCG@5 value |
|---|---:|---:|---:|---:|---:|---:|
| 8L/8H | 0.124685 | 0.114965 | 0.068887 | +66.890% | +37.796% | +153.986% |
| 16L/8H | 0.085535 | 0.047431 | 0.058964 | -19.560% | -13.794% | 0.000% |

The 8L mean per-layer K relative errors were
`[0.195585, 0.387385, 0.646575, 0.673304, 0.692437, 0.613704, 0.725757, 0.776539]`.
The 16L model's Fresh endpoint was effectively identical to its no-prefix endpoint, so extra depth
was not admitted merely because it created more cache tensors. Both implementations passed the
same-model incremental/full-forward check with maximum hidden error below `5.8e-6`.

The 32-user ranking intervals are too wide for a quality claim. This round selects H512/8L/8H for
a 512-user stability run; it does not establish the final Reuse loss magnitude.
