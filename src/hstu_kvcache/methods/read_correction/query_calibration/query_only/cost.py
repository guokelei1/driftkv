"""Analytical operation counts for the explicit feature/ridge implementation."""


def correction_forward(config, n, *, queries=1, batch=1, include_scale_add=True):
    heads, dim = int(config["heads"]), int(config["head_dim"])
    width = heads * dim
    if config["feature_mode"] == "head_phi":
        per_query = 2 * heads * dim * dim + 5 * width
    elif config["feature_mode"] == "cross_phi":
        per_query = 4 * width * width + 6 * width
    else:
        raise ValueError("unknown query feature mode")
    if include_scale_add:
        per_query += 2 * width  # rate*N and addition to the native read
    return int(batch * queries * per_query)


def feature_ridge_fit(config, samples):
    """Centered FP64 solve, prediction, and fit-statistics arithmetic estimate.

    Dense matmuls/solve dominate. Scalar reductions, normalization, objective
    diagnostics and ELU are estimated explicitly; this is not a timing model.
    """
    heads, dim = int(config["heads"]), int(config["head_dim"])
    width = heads * dim
    if config["feature_mode"] == "head_phi":
        groups, p, outputs, activation = heads, dim, dim, 2 * samples * width
    elif config["feature_mode"] == "cross_phi":
        groups, p, outputs, activation = 1, 2 * width, width, samples * width
    else:
        raise ValueError("unknown query feature mode")
    m = samples
    matrix = 2*m*p*p + 4*m*p*outputs + (2/3)*p**3 + 2*p*p*outputs
    # mean/std/standardize, target centering, matrix division/penalty,
    # fitted bias/prediction error and fitted/zero objective reductions.
    scalar = 6*m*p + 2*p + 9*m*outputs + outputs + 3*p*p + p*outputs
    return int(activation + groups * (matrix + scalar))
