# Affine query-correction primitive (v1/v2)

This module is the retained affine primitive. Current Motivation uses the
nonlinear query-feature implementation in `read_correction_v5/query_only/`;
see [the selected methods](../../../../docs/design/read_correction_probe_2026_09.md).
Its historical results must not be described as the current Q-v5 curve.

Each layer and attention head fits an affine response rate from its native query.
Calibration-only means/scales keep the FP64 ridge solve stable. The intercept is
unpenalized; the ridge term applies to slopes in normalized-query coordinates.
At serving time the rate is multiplied by the current valid history length and
added to the native historical read. It does not modify persistent K/V.

`fit_query(q, target_rate, counts, ridge=.01)` returns a `QueryCorrection` and
fit statistics. Both tensors are `[users, heads, queries, head_dim]`; target_rate
has already been divided by history length. It must use the teacher response at
the same q, including corrections from already fitted lower layers.

Keep module `get_config()` and `state_dict()` together for reproducible loading.
