"""Shared constant versus native-response-conditioned read correction.

The predictor receives no query, item features, summary, or per-user fitted
parameter. A shared affine map may use the actual native full-head response.
Teachers and normalization statistics enter only calibration.
"""

from __future__ import annotations

import torch


def fit_layer(wanted, observed, counts, use_response):
    """Fit b or b+Tr to per-event response deltas using CPU FP64 ridge.

    wanted is [users, heads, queries, head_width]; observed is the native
    response divided by history count, in [users, queries, full_width] order.
    User weights are count**2 / mean(count**2), matching aggregate-response
    fitting; each user's queries are averaged. Ridge is .01*users and includes
    the intercept. All returned parameter values are CPU double tensors.
    """
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        target = wanted.cpu().double().transpose(1, 2).flatten(2)
        users, queries, _ = target.shape
        counts = counts.cpu().double()
        weight = counts.square() / counts.square().mean()
        x = torch.ones(users, queries, 1, dtype=torch.double)
        parameters = {}
        if use_response:
            native = observed.cpu().double()
            center = native.mean((0, 1))
            scale = native.std((0, 1)).clamp_min(1e-4)
            x = torch.cat((x, (native-center)/scale), dim=-1)
            parameters.update(read_center=center, read_scale=scale)
        design, y = x.flatten(0, 1), target.flatten(0, 1)
        row_weight = weight[:, None].expand(users, queries).reshape(-1, 1)
        gram = design.T @ (design * row_weight) / queries
        rhs = design.T @ (y * row_weight) / queries
        system = gram + .01 * users * torch.eye(gram.shape[0], dtype=torch.double)
        solution = torch.linalg.solve(system, rhs)
        torch.testing.assert_close(system @ solution, rhs, atol=1e-7, rtol=1e-6)
        parameters["bias"] = solution[0]
        if use_response:
            parameters["read_weights"] = solution[1:]
        residual = (design @ solution - y).reshape_as(target)
        stats = dict(
            fitting_rate_mse=float(residual.square().mean()),
            fitting_aggregate_objective=float((residual.square().mean((1, 2))*weight).mean()),
            normal_relative_residual=float((system @ solution-rhs).abs().max()
                                           / rhs.abs().max().clamp_min(1e-30)),
            count_square_mean=float(counts.square().mean()),
            shared_parameters=solution.numel(), read_dimension=observed.shape[-1] if use_response else 0,
            fitting_scenes=users, fitting_queries=queries, ridge=.01,
            use_native_response=bool(use_response), cpu_solver_threads=1,
        )
    finally:
        torch.set_num_threads(previous_threads)
    return parameters, stats


def make_history_override(parameters, counts):
    """Install fitted layers in FP32; q is intentionally unused by both arms."""
    counts = counts.float()
    views = []
    for layer in parameters:
        p = {key: value.to(device=counts.device, dtype=torch.float32) for key, value in layer.items()}
        view = {"bias": p["bias"]}
        if "read_weights" in p:
            weights = p["read_weights"] / p["read_scale"][:, None]
            view.update(read_weights=weights, bias=p["bias"]-p["read_center"] @ weights)
        views.append(view)

    def transform(layer, query, native):
        del query
        if layer >= len(views):
            return native
        view = views[layer]
        batch, heads, queries, width = native.shape
        rate = view["bias"].expand(batch, queries, -1)
        if "read_weights" in view:
            observed = native.transpose(1, 2).flatten(2) / counts[:, None, None]
            rate = rate + observed @ view["read_weights"]
        delta = rate.reshape(batch, queries, heads, width).transpose(1, 2)
        return native + delta * counts[:, None, None, None]

    return transform


def self_check():
    """Reference weighted ridge and runtime folding for both input arms."""
    generator = torch.Generator().manual_seed(172220)
    observed = torch.randn(7, 5, 6, generator=generator, dtype=torch.double)
    wanted = torch.randn(7, 2, 5, 3, generator=generator, dtype=torch.double)
    counts = torch.tensor([32., 64., 128., 256., 512., 768., 1024.], dtype=torch.double)
    target = wanted.transpose(1, 2).flatten(2)
    weight = counts.square()/counts.square().mean()
    max_normal = max_callback = 0.0
    original_threads = torch.get_num_threads()
    for use_response in (False, True):
        p, stats = fit_layer(wanted, observed, counts, use_response)
        x = torch.ones(7, 5, 1, dtype=torch.double)
        if use_response:
            x = torch.cat((x, (observed-p["read_center"])/p["read_scale"]), dim=-1)
        gram = torch.einsum("bqi,bqj,b->ij", x, x, weight)/5
        rhs = torch.einsum("bqi,bqd,b->id", x, target, weight)/5
        reference = torch.linalg.solve(gram+.07*torch.eye(x.shape[-1], dtype=torch.double), rhs)
        actual = torch.cat((p["bias"][None], p["read_weights"]), dim=0) if use_response else p["bias"][None]
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
        native = (observed*counts[:, None, None]).reshape(7, 5, 2, 3).transpose(1, 2)
        prediction = x @ reference
        expected = native + prediction.reshape(7, 5, 2, 3).transpose(1, 2)*counts[:, None, None, None]
        callback = make_history_override([p], counts.float())
        result = callback(0, None, native.float())
        torch.testing.assert_close(result, expected.float(), atol=5e-4, rtol=2e-5)
        assert torch.equal(callback(1, None, native.float()), native.float())
        if not use_response:
            # This control cannot obtain individual information through fitting inputs.
            altered, _ = fit_layer(wanted, 100*observed+3, counts, False)
            torch.testing.assert_close(altered["bias"], p["bias"], atol=0, rtol=0)
        max_normal = max(max_normal, stats["normal_relative_residual"])
        max_callback = max(max_callback, float((result-expected.float()).abs().max()))
    assert torch.get_num_threads() == original_threads
    return dict(status="passed", normal_equation_atol=1e-10,
                max_normal_relative_residual=max_normal,
                max_fp32_callback_absolute_error=max_callback,
                constant_fit_ignores_native_input=True, restored_cpu_threads=original_threads)


if __name__ == "__main__":
    import json
    torch.set_num_threads(4)
    print(json.dumps(self_check(), indent=2))
