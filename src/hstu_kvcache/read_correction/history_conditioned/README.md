# Head-local history-correction primitive (v1)

This is the retained lightweight v1 implementation. Current Motivation uses
nonlinear temporary K/V mapping from `read_correction_v4/`, described in
[the selected methods](../../../../docs/design/read_correction_probe_2026_09.md).
The two implementations have different inputs, operations and costs.

This exploratory method adds a query-conditioned full-history branch to the
same affine query correction:

`delta = N * (b + A*q_norm + B*mean_i SiLU(Wq*q_norm + Wk*K_norm_i + Wv*V_norm_i + c))`.

The shared head-local encoder starts at width 32. This is an implementation
setting, not a dimensionality experiment. Query and retained history interact
before the nonlinearity. Every query explicitly accesses all retained positions;
no persistent summary, translator or incremental summary maintenance is used.

Initialize from a fitted Q rule with `initialize_query(q_module)`, set K/V
normalization from calibration data using `set_history_normalization`, and train
`rate(q,k,v,counts)` against same-query teacher residual divided by N. All
parameters remain differentiable. The history output projection starts at zero:
the first step trains that projection, subsequent steps reach the encoder.

Query/token chunks bound inference intermediates. Training retains backward
activations across chunks and therefore needs bounded user minibatches. Counts
mark valid prefix positions when a calibration batch contains padding. Serving
uses the native reader's original history-length conventions. The rate is only
applied to the current request; source K/V are never mutated.
