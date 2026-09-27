"""Shared read-correction probes with and without an actual user response.

Both probes see only the candidate item's Current embedding. The conditioned
probe additionally sees the already computed native history response. Neither
probe receives the reader query, time features, summaries, or evaluation-user
teacher information. These are fixed-state diagnostics, not serving methods.
"""

from __future__ import annotations

import torch


def fit_layer(candidate_features, native_rate, target_rate, use_user_response):
    """Fit one shared layer rule on calibration users only, using CPU FP64.

    Candidate/native inputs have shape [users, candidates, width]. Targets
    have shape [users, heads, candidates, head_width] and are teacher-minus-
    native responses divided by history length. The objective averages each
    user's candidates equally, with the fixed ridge coefficient .01. Returned
    parameter values are all tensors; optional native_* keys identify the
    branch that uses the user response.
    """
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        candidate = candidate_features.cpu().double()
        target = target_rate.cpu().double().transpose(1, 2).flatten(2)
        users, queries, _ = candidate.shape
        center = candidate.mean((0, 1))
        scale = candidate.std((0, 1)).clamp_min(1e-4)
        normalized = (candidate - center) / scale
        inputs = [torch.ones_like(normalized[..., :1]), normalized]
        parameters = {"candidate_center": center, "candidate_scale": scale}
        if use_user_response:
            native = native_rate.cpu().double()
            native_center = native.mean((0, 1))
            native_scale = native.std((0, 1)).clamp_min(1e-4)
            inputs.append((native - native_center) / native_scale)
            parameters.update(native_center=native_center, native_scale=native_scale)
        design = torch.cat(inputs, dim=-1).flatten(0, 1)
        wanted = target.flatten(0, 1)
        gram = design.T @ design / queries
        rhs = design.T @ wanted / queries
        system = gram + .01 * users * torch.eye(gram.shape[0], dtype=torch.double)
        weights = torch.linalg.solve(system, rhs)
        torch.testing.assert_close(system @ weights, rhs, atol=1e-7, rtol=1e-6)
        parameters["weights"] = weights
        prediction = design @ weights
        stats = {
            "fitting_rate_mse": float((prediction - wanted).square().mean()),
            "target_rate_energy": float(wanted.square().mean()),
            "normal_relative_residual": float((system @ weights - rhs).abs().max()
                                              / rhs.abs().max().clamp_min(1e-30)),
            "shared_parameters": weights.numel(),
            "candidate_dimension": candidate.shape[-1],
            "native_dimension": native_rate.shape[-1] if use_user_response else 0,
            "fitting_users": users,
            "fitting_queries_per_user": queries,
            "use_user_response": bool(use_user_response),
            "ridge": .01,
            "cpu_solver_threads": 1,
        }
    finally:
        torch.set_num_threads(previous_threads)
    return parameters, stats


def prepare_layer(parameters, candidate_features):
    """Prepare a candidate-only term; no user-history information enters it."""
    p = {name: value.to(device=candidate_features.device, dtype=candidate_features.dtype)
         for name, value in parameters.items()}
    width = candidate_features.shape[-1]
    candidate = (candidate_features - p["candidate_center"]) / p["candidate_scale"]
    rate = p["weights"][0] + candidate @ p["weights"][1:width + 1]
    view = {"candidate_rate": rate}
    if "native_center" in p:
        native_weights = p["weights"][width + 1:] / p["native_scale"][:, None]
        view["candidate_rate"] = rate - p["native_center"] @ native_weights
        view["native_weights"] = native_weights
    return view


def apply_prepared_layer(native, counts, view):
    """Return corrected native history heads without a query or teacher input."""
    rate = view["candidate_rate"]
    if "native_weights" in view:
        observed = native.transpose(1, 2).flatten(2) / counts[:, None, None]
        rate = rate + observed @ view["native_weights"]
    batch, heads, queries, head_width = native.shape
    delta = rate.reshape(batch, queries, heads, head_width).transpose(1, 2)
    return native + delta * counts[:, None, None, None]


def make_history_override(parameters, candidate_features, counts):
    """Build a callback for reader.score, ignoring the reader's query tensor."""
    views = [prepare_layer(layer, candidate_features) for layer in parameters]

    def transform(layer, query, native):
        del query
        if layer >= len(views):
            return native
        return apply_prepared_layer(native, counts, views[layer])

    return transform


def self_check():
    """Check a direct ridge reference and equal-item correction invariance."""
    generator = torch.Generator().manual_seed(172020)
    def random(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.double)
    candidate, native = random(7, 5, 6), random(7, 5, 6)
    target = random(7, 2, 5, 3)
    original_threads = torch.get_num_threads()
    records = {}
    for use_user in (False, True):
        p, stats = fit_layer(candidate, native, target, use_user)
        x = [(candidate - p["candidate_center"]) / p["candidate_scale"]]
        if use_user:
            x.append((native - p["native_center"]) / p["native_scale"])
        x = torch.cat((torch.ones(7, 5, 1, dtype=torch.double), *x), dim=-1)
        y = target.transpose(1, 2).flatten(2)
        gram = torch.einsum("bqi,bqj->ij", x, x) / 35
        rhs = torch.einsum("bqi,bqd->id", x, y) / 35
        reference = torch.linalg.solve(gram + .01 * torch.eye(gram.shape[0], dtype=torch.double), rhs)
        torch.testing.assert_close(p["weights"], reference, atol=1e-10, rtol=1e-10)
        counts = torch.full((7,), 1024., dtype=torch.double)
        native_heads = (native * counts[:, None, None]).reshape(7, 5, 2, 3).transpose(1, 2)
        callback = make_history_override([p], candidate, counts)
        actual = callback(0, random(7, 2, 5, 3), native_heads)
        expected = native_heads + (x @ p["weights"]).reshape(7, 5, 2, 3).transpose(1, 2) * counts[:, None, None, None]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        assert torch.equal(callback(1, None, native_heads), native_heads)
        records["candidate_and_response" if use_user else "candidate_only"] = stats
        if not use_user:
            same_items = candidate[:1].expand(2, -1, -1)
            same_item_view = prepare_layer(p, same_items)
            assert torch.equal(same_item_view["candidate_rate"][0], same_item_view["candidate_rate"][1])
            # Changing the ignored native fit input cannot change this rule.
            altered, _ = fit_layer(candidate, native * 100 + 3, target, False)
            torch.testing.assert_close(altered["weights"], p["weights"], atol=0, rtol=0)
    assert torch.get_num_threads() == original_threads
    return {
        "status": "passed",
        "candidate_only_same_item_correction": "identical across users at fixed history length",
        "candidate_only_native_input_independence": "bitwise fitted weights unchanged",
        "normal_reference_and_callback": "both arms agree within 1e-10",
        "max_normal_relative_residual": max(row["normal_relative_residual"] for row in records.values()),
        "restored_cpu_threads": original_threads,
    }


if __name__ == "__main__":
    import json

    torch.set_num_threads(4)
    print(json.dumps(self_check(), indent=2))
