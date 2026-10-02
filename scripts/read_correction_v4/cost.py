"""Executed-shape arithmetic for the transient cross-head token K/V map.

Multiply-add=2; elementary activation=1, as in the existing experiment ledger.
Input normalization remains an executed subtraction and division. The affine
map executes for every retained token, including tokens outside old_counts;
gating does not earn a dense-compute discount. Masks/comparisons/concatenation
are logical or memory operations; multiplying the delta and attention weights
by masks is charged. Fixed native query computation cancels against Reuse.

Temporary mapped K/V is discarded after each request. Several queries in one
call may share the transform within that call; distinct requests never do.
"""

from math import ceil

from read_correction_2026_09.cost import (
    CostModel,
    correction_forward as query_affine_forward,
    eager_read,
    normalize_metrics,
    query_ridge_fit,
    teacher_history_read,
)


def correction_parts(config, history_length, *, queries=1, batch=1, trace=False):
    """One layer's actual new-history read, beyond the common native reader.

    The history_override callback already receives native_history, so there is
    no second old-history read. The core returns the replacement directly;
    only trace=True computes replacement-minus-native as a diagnostic.
    """
    heads, head_dim = int(config["heads"]), int(config["head_dim"])
    width = heads * head_dim
    n, q, b = int(history_length), int(queries), int(batch)
    parts = {
        "token_input_normalization_flops": b * n * 4 * width,
        "token_affine_flops": b * n * (8 * width * width + 2 * width),
        "old_prefix_gate_multiplication_flops": b * n * 2 * width,
        "temporary_kv_residual_addition_flops": b * n * 2 * width,
        # Existing history_read counts QK+AV, scale/ELU/+1, plus valid mask.
        "mapped_history_read_flops": b * q * n * (4 * width + 4 * heads),
        "query_affine_residual_flops": 0,
        "trace_difference_flops": b * q * width if trace else 0,
    }
    if config.get("query_affine", False):
        # q normalization, head-local affine, N*rate, then new_history+delta.
        parts["query_affine_residual_flops"] = query_affine_forward(
            {"heads": heads, "head_dim": head_dim}, n, queries=q, batch=b,
            include_scale_add=True)
    return parts


def correction_forward(config, history_length, *, queries=1, batch=1, trace=False):
    """Total per-layer additional FLOPs, including optional Q correction."""
    return sum(correction_parts(config, history_length, queries=queries,
                                batch=batch, trace=trace).values())


def token_ridge_fit(observations, input_width):
    """Centered FP64 positive-ridge fit; ``input_width`` is already 2D.

    Match fit_affine_tokens: d*d normal equations, with the intercept obtained
    from target centering instead of an extra augmented design coordinate.
    Matrix work is 6*m*d^2 (Gram, RHS, residual diagnostics) plus the declared
    LU/triangular-solve leading term 8*d^3/3. Statistics, normalization,
    centering and diagnostic arithmetic add 15*m*d+d; Gram/RHS scaling,
    diagonal-penalty construction and matrix addition add 4*d^2.

    This is an analytical arithmetic estimate, not hardware instructions.
    Count actual selected token pairs, never token pairs times query count.
    Teacher-minus-old target construction, cache capture and held-out
    evaluation are separate caller-ledger entries. The probe uses ridge>0.
    """
    m, d = int(observations), int(input_width)
    return int(ceil(6*m*d*d + (8/3)*d**3 + 15*m*d + 4*d*d + d))
