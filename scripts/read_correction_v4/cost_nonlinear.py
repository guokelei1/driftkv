"""Additive nonlinear token-map arithmetic, reusing the unchanged affine ledger."""

from read_correction_v4.cost import (
    CostModel, correction_parts as affine_parts, eager_read, normalize_metrics,
    query_ridge_fit, teacher_history_read, token_ridge_fit,
)


def nonlinear_forward_parts(config, tokens, *, batch=1, include_base_add=True):
    """The added branch shares normalized inputs and producer gating with base.

    2D->W->2D dense matrices, both biases, SiLU and physical output units.
    The addition to the affine residual occurs only in deployed token_delta;
    training may cache base-mapped K/V and assemble the temporary K/V later.
    """
    d = int(config["heads"]) * int(config["head_dim"])
    w = int(config["nonlinear_width"])
    rows = int(batch) * int(tokens)
    return {
        "nonlinear_dense_flops": rows * 8*d*w,
        "nonlinear_bias_and_silu_flops": rows * (2*w + 2*d),
        "nonlinear_output_units_flops": rows * 2*d,
        "nonlinear_add_to_affine_residual_flops": rows * 2*d if include_base_add else 0,
    }


def nonlinear_forward_flops(config, tokens, *, batch=1, include_base_add=True):
    return sum(nonlinear_forward_parts(config, tokens, batch=batch,
                                      include_base_add=include_base_add).values())


def correction_parts(config, history_length, *, queries=1, batch=1, trace=False):
    """Complete deployment cost: affine base once plus one nonlinear branch."""
    result = affine_parts(config, history_length, queries=queries, batch=batch, trace=trace)
    result.update(nonlinear_forward_parts(config, history_length, batch=batch))
    return result


def correction_forward(config, history_length, *, queries=1, batch=1, trace=False):
    return sum(correction_parts(config, history_length, queries=queries,
                                batch=batch, trace=trace).values())


def training_forward_parts(config, tokens, *, queries=16, batch=1,
                           include_read=True, query_residual=True):
    """Actual normalized_losses forward from cached Xnorm/baseKV/q/teacher.

    No repeated normalization, ridge map or teacher access is charged here;
    their one-time construction belongs to the preparation ledger. The
    current joint fitter always passes a cached query-residual tensor, even
    when it is all zeros, and therefore executes its addition.

    include_read=False is valid only for a real KV-only fitting call that
    skips temporary assembly, history_read and read loss. It must not be
    used to undercharge the existing joint normalized_losses implementation.
    """
    d = int(config["heads"]) * int(config["head_dim"])
    h = int(config["heads"])
    n, q, b = int(tokens), int(queries), int(batch)
    result = nonlinear_forward_parts(config, n, batch=b, include_base_add=False)
    result.update({
        "temporary_residual_gate_and_base_add_flops": 4*b*n*d if include_read else 0,
        "mapped_read_flops": b*q*n*(4*d + 4*h) if include_read else 0,
        "cached_query_residual_add_flops": b*q*d if include_read and query_residual else 0,
        # KV subtract/divide/square/mask/reduction; read subtract/divide/square/mean.
        "normalized_kv_loss_forward_flops": 10*b*n*d,
        "normalized_read_loss_forward_flops": 4*b*q*d if include_read else 0,
        "joint_loss_add_flops": 1 if include_read else 0,
    })
    return result


def training_step_parts(config, tokens, *, queries=16, batch=1,
                        trainable_parameters, include_read=True, query_residual=True):
    """Declared conservative forward+backward and optimizer arithmetic estimate.

    Use 3x for learned MLP, mapped read/assembly and scalar losses; with fixed
    q, mapped-read matrix arithmetic alone is 2.5x its forward, so 3x is a
    stated upper estimate. Optimizer/gradient handling is estimated20 FLOPs
    per trainable parameter per executed step. These are not GPU instructions.
    """
    forward = training_forward_parts(config, tokens, queries=queries, batch=batch,
                                      include_read=include_read, query_residual=query_residual)
    result = {key.removesuffix("_flops") + "_forward_backward_estimate_flops": 3*value
              for key, value in forward.items()}
    result["optimizer_estimate_flops"] = 20*int(trainable_parameters)
    return result
