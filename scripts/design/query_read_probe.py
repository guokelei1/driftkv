"""Matched shared b+Aq and b+Aq+Tr probes, without state summaries."""

from __future__ import annotations

import torch

from design.diagnose_native_input import fit_joint
from design.shared_read_probe import apply_prepared_layer, prepare_layer
from design.shared_read_probe import fit_layer as fit_query_response


def fit_layer(query, wanted, observed, counts, use_response):
    """Use the same response ridge for both arms, removing only native columns.

    Query/wanted: [users, heads, queries, head_width]; observed is the native
    response per retained event, [users, queries, width]. All inputs are CPU
    calibration tensors. q is head-local in both arms; r includes all heads.
    """
    latent = torch.ones(len(query), 1, dtype=torch.double)
    if use_response:
        parameters, stats = fit_query_response(latent, query, wanted, observed, counts)
    else:
        # Empty read columns leave precisely the same regularized q block.
        # Supply empty statistics explicitly rather than calling std on no data.
        q = query.double()
        empty_read = observed[..., :0].double()
        coordinates = dict(query_center=q.mean((0, 2), keepdim=True),
                           query_scale=q.std((0, 2), keepdim=True).clamp_min(1e-4),
                           read_center=torch.empty(0, dtype=torch.double),
                           read_scale=torch.empty(0, dtype=torch.double))
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            parameters, stats = fit_joint(latent, q, wanted, empty_read, counts,
                                          coordinates=coordinates)
        finally:
            torch.set_num_threads(previous_threads)
        stats["cpu_solver_threads"] = 1
    stats["use_native_response"] = bool(use_response)
    return parameters, stats


def make_history_override(parameters, counts):
    """Install fitted prefix layers; no teacher or summary enters prediction."""
    latent = torch.ones(len(counts), 1, device=counts.device)
    views = [prepare_layer(latent, p) for p in parameters]

    def transform(layer, query, native):
        if layer >= len(views):
            return native
        view = views[layer]
        if view["read_weights"].shape[-2]:
            return apply_prepared_layer(query, native, counts, view)
        delta = view["intercept"][:, :, None] + query @ view["slope"]
        return native + delta * counts[:, None, None, None]

    return transform


def self_check():
    """CPU checks against explicit normal equations and installed callbacks."""
    generator = torch.Generator().manual_seed(172121)
    def random(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.double)
    query, wanted = random(7, 2, 5, 3), random(7, 2, 5, 3)
    observed = random(7, 5, 6)
    counts = torch.tensor([32., 64., 128., 256., 512., 768., 1024.], dtype=torch.double)
    relative_residual, callback_error = [], []
    for use_response in (False, True):
        p, stats = fit_layer(query, wanted, observed, counts, use_response)
        q = (query - p["query_center"]) / p["query_scale"]
        x = torch.cat((torch.ones_like(q[..., :1]), q), dim=-1)
        if use_response:
            r = (observed - p["read_center"]) / p["read_scale"]
            x = torch.cat((x, r[:, None].expand(-1, 2, -1, -1)), dim=-1)
        weight = counts.square() / counts.square().mean()
        gram = torch.einsum("bhqi,bhqj,b->hij", x, x, weight) / 5
        rhs = torch.einsum("bhqi,bhqd,b->hid", x, wanted, weight) / 5
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            reference = torch.linalg.solve(gram + .07 * torch.eye(x.shape[-1], dtype=torch.double), rhs)
        finally:
            torch.set_num_threads(previous)
        actual = torch.cat((p["weights"][0], p["read_weights"]), dim=1)
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
        native = (observed * counts[:, None, None]).reshape(7, 5, 2, 3).transpose(1, 2)
        expected = native + torch.einsum("bhqi,hid->bhqd", x, actual) * counts[:, None, None, None]
        # Serving deliberately uses the FP32 path used by frozen HSTU models.
        callback = make_history_override([p], counts.float())
        result = callback(0, query.float(), native.float())
        torch.testing.assert_close(result, expected.float(), atol=5e-4, rtol=2e-5)
        assert torch.equal(callback(1, query.float(), native.float()), native.float())
        callback_error.append(float((result - expected.float()).abs().max()))
        relative_residual.append(stats["normal_relative_residual"])
    return dict(status="passed", normal_equation_atol=1e-10,
                max_normal_relative_residual=max(relative_residual),
                max_fp32_callback_absolute_error=max(callback_error),
                query_only_read_columns=0, query_response_read_columns=6)


if __name__ == "__main__":
    import json
    torch.set_num_threads(4)
    print(json.dumps(self_check(), indent=2))
