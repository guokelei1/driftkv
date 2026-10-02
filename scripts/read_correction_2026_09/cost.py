"""Analytical correction FLOPs; multiply-add=2 and elementary activation=1.

Ordinary query reads use one canonical common reader and cancel. The paired
eager reader implements the same prefix/self arithmetic as score_one_query;
no credit is taken for a backend change versus the saved native controls.
Wall time and GPU memory are measured separately. The denominator retains
the sealed Full-history minus actual rolling-append convention, including
short-user Torch execution and batched appends.
"""
from math import ceil

from selective_recompute_2026_09.cost import CostModel, normalized_point


def correction_forward(config, history_length, *, queries=1, batch=1, include_scale_add=True):
    """One layer, including response length scaling and addition to native read.

    History K/V projections are reused across queries inside one call, but
    recalculated on every request. Dense padded tokens are charged. Mask
    comparisons and copies are not FLOPs; multiplying by the mask is charged.
    Chunk reductions plus accumulations execute exactly N adds per output.
    """
    h, d = int(config["heads"]), int(config["head_dim"])
    n, q, width = int(history_length), int(queries), config.get("width")
    total = q * (2 * h * d * d + 3 * h * d)  # normalized affine rate
    if include_scale_add:
        total += 2 * q * h * d  # N*rate, native+delta
    if width is not None and n and q:
        w = int(width)
        total += n * (4 * h * d + 4 * h * d * w + h * w)
        total += q * (3 * h * d + 4 * h * d * w + 2 * h * w)
        total += 4 * q * n * h * w
    return int(batch * total)


def eager_read(cost, n, *, queries=1, batch=1):
    """Frozen legacy paired reader; independent query rows and self attention."""
    rows = batch * queries
    d, h = cost.hidden_size, cost.num_heads
    # Dense prefix reads have scale, ELU and +1, but no causal-mask multiply.
    # Self: q*k and reduction, scale/ELU/+1, then weight*v + prefix.
    per_layer = (cost.block(rows, 0, residual_scales=False)
                 + teacher_history_read(cost, n, queries=queries, batch=batch)
                 + rows * (4 * d + 2 * h))
    return (cost.embedding(rows, query=True)
            + cost.num_layers * per_layer
            + cost.rms_norm(rows) + rows * (2 * cost.hidden_size + 1))


def teacher_history_read(cost, n, *, queries=1, batch=1):
    """Same-query historical QK, scale/ELU+1 and AV; no target query projection."""
    return int(batch * queries * n * (4 * cost.hidden_size + 3 * cost.num_heads))


def query_ridge_fit(heads, head_dim, observations):
    """Ridge normal equations, solve and diagnostics; LU leading-term estimate.

    The solve's exact backend instructions are implementation-dependent. This
    declared analytical estimate includes FP64 normal equations, statistics,
    regularization, fitted residuals and zero-target MSE; FP64 has no arbitrary
    arithmetic multiplier. It is never described as measured instructions.
    """
    h, d, m = int(heads), int(head_dim), int(observations)
    p = d + 1
    return int(ceil(h * (2*m*p*p + 4*m*p*d + (2/3)*p**3 + 2*p*p*d
                         + 10*m*d + 3*p*p + p*d + 4*d)))


def normalize_metrics(*, full_auc, reuse_auc, baseline_auc, extra_flops, full_minus_reuse_flops):
    if any(value is None for value in (full_auc, reuse_auc, baseline_auc)):
        return {"auc_gap": None, "recovery_percent": None,
                "relative_flops_percent": 100 * extra_flops / full_minus_reuse_flops
                if full_minus_reuse_flops else None}
    return normalized_point(full_auc=full_auc, reuse_auc=reuse_auc,
                            baseline_auc=baseline_auc, extra_flops=extra_flops,
                            full_minus_reuse_flops=full_minus_reuse_flops)
