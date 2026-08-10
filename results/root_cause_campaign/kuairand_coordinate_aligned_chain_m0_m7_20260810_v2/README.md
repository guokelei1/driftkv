# Rejected synthetic coordinate control

Status: rejected on 2026-08-10 as a natural streaming baseline. The compact measurements are
retained only as a synthetic positive control; `scientific_result=false` and
`formal_result=false` remain unchanged.

M1–M7 used real KuaiRand next-day updates and an independent holdout, but each publication also
applied a deliberate function-preserving K/V coordinate scale. The current model's paired Q/K and
V/output transforms cancel under Fresh recomputation, while an older cache remains in the old
coordinate system. This intervention therefore manufactures a stable Reuse penalty.

The holdout triangle exposes the intervention's saturation fingerprint. In the M7 row, NDCG@5
Recompute-over-Reuse is `6.263%` at age 1, `9.270%` at age 2, and approximately `9.744%` for all
older sources. With the steady-state `key_log_step=0.5` and `value_log_step=1.8`, old-history
contributions decay exponentially and become almost indistinguishable after two or three ages.
This plateau is unsuitable as evidence of natural optimizer-induced cache drift.

Permitted future use is limited to evaluator validation, a synthetic stress test, or a fallback
simulation that is explicitly labeled as artificial. It must not be cited as the paper's natural
Reuse–Recompute opportunity, baseline chain, or production-representative age curve. The 2L/H64
checkpoint payload was retired; the JSON results, table, configuration, logs, and code remain for
audit and reproduction of the synthetic control.
