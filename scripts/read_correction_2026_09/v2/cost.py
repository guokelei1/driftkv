"""v2 correction arithmetic; the sealed Full/Reuse denominator is unchanged.

Multiply-add=2, elementary activation=1, affine LayerNorm=7*r+2 per row.
The history encoder charges all dense N*N attention pairs, including padded
keys/queries, and five softmax/scale operations per pair (minus one reduction
addition per softmax row). SDPA is priced by its dense mathematical operations,
not GPU kernel instruction counts. Copies, masks and comparisons are separate
memory/logical work. All Q-dependent encoder operations repeat for every query.

Calibration records state their backward/optimizer estimates separately.
Ordinary native query reads cancel in the same convention as the v1 ledger.
"""
from read_correction_2026_09.cost import (
    CostModel, correction_forward as v1_correction_forward,
    eager_read, normalize_metrics, query_ridge_fit, teacher_history_read,
)


def base_config(config):
    """The retained v1 H is a real, fully executed residual branch."""
    return {"heads": config["heads"], "head_dim": config["head_dim"],
            "width": config["base_width"], "token_chunk": config["base_token_chunk"],
            "query_chunk": config["base_query_chunk"]}


def history_encoder_forward(config, history_length, *, queries=1, batch=1):
    """One layer's residual_rate, excluding the frozen base and its addition.

    K/V/position projections are computed once within a call. Every query
    subsequently executes its own nonlinear full-history Transformer. The
    mean includes only valid rows; dense padded rows still execute the encoder.
    """
    n, q, b = int(history_length), int(queries), int(batch)
    if not n or not q:
        return 0
    d = int(config["heads"]) * int(config["head_dim"])
    r, a = int(config["encoder_width"]), int(config["attention_heads"])
    # Normalize both K/V, three biased input projections, their two additions,
    # and position/N plus position squared. Position comparisons are not FLOPs.
    shared = 4*n*d + 4*n*d*r + 9*n*r + 2*n
    # Query normalization/projection; pooled output projection/bias/unit scale.
    per_query = 4*d + 4*d*r + r
    # SiLU conditioning; two affine LayerNorms; QKV/out and 2r FFN, all biased;
    # two residual additions; GELU; masked sum and division by valid count.
    per_query += n * (16*r*r + 28*r + 4)
    # Full bidirectional attention: QK/AV plus scale and stable row softmax.
    per_query += 4*n*n*r + a*n*(5*n - 1)
    return int(b * (shared + q * per_query))


def correction_forward(config, history_length, *, queries=1, batch=1, include_scale_add=True):
    """One complete correction layer, optionally including N*rate/native+delta."""
    if "encoder_width" not in config:
        return v1_correction_forward(config, history_length, queries=queries,
                                     batch=batch, include_scale_add=include_scale_add)
    base = v1_correction_forward(base_config(config), history_length, queries=queries,
                                batch=batch, include_scale_add=include_scale_add)
    heavy = history_encoder_forward(config, history_length, queries=queries, batch=batch)
    # rate() explicitly adds base.rate and residual_rate, including an empty prefix.
    addition = batch * queries * config["heads"] * config["head_dim"]
    return int(base + heavy + addition)
