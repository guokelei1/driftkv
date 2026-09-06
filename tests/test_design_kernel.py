"""Kernel release-view algebra against a direct free-intercept solve."""

import torch

from hstu_kvcache.adaptation.translator import RidgeTranslator


def test_source_kernel_matches_direct_solve_and_time_factorization():
    torch.manual_seed(17)
    mapper = RidgeTranslator(layers=2, width=2, target=1, rank=4,
        history_scope="all", temporal=True, source_kernel=True).double()
    source = torch.randn(12, mapper.base_inputs, dtype=torch.float64).repeat_interleave(4, 0)
    source[:, 8] = .4
    source[:, 17] = .6
    phi = torch.randn(48, 32, dtype=torch.float64)
    features = mapper.with_time(source, phi)
    target = torch.randn(48, 2, 2, dtype=torch.float64)
    record = mapper.fit(features, target)
    normalized = (features-mapper.center)/mapper.input_scale
    x, t = normalized[:, :mapper.base_inputs], normalized[:, mapper.base_inputs:]
    squared = (x[:, None]-x[None]).square().sum(-1)
    gram = (-squared/mapper.kernel_bandwidth_squared).exp()+t @ t.T/t.shape[-1]
    system = torch.zeros(49, 49, dtype=torch.float64)
    system[:48, :48] = gram+record["source_kernel"]["regularizer"]*torch.eye(48, dtype=torch.float64)
    system[:48, -1] = system[-1, :48] = 1
    coefficients = torch.linalg.solve(system, torch.cat((target.flatten(1), torch.zeros(1, 4, dtype=torch.float64))))
    query_source = source[::4]+torch.randn(12, mapper.base_inputs, dtype=torch.float64)*.1
    query_phi = torch.randn(12, 32, dtype=torch.float64)
    query_features = mapper.with_time(query_source, query_phi)
    query = (query_features-mapper.center)/mapper.input_scale
    qx, qt = query[:, :mapper.base_inputs], query[:, mapper.base_inputs:]
    cross = (-(qx[:, None]-x[None]).square().sum(-1)/mapper.kernel_bandwidth_squared).exp()+qt @ t.T/t.shape[-1]
    expected = (cross @ coefficients[:-1]+coefficients[-1]).reshape(12, 2, 2)
    actual = mapper.rates(query_features)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    factored = mapper.rates(query_source)+torch.einsum("nt,nltw->nlw", query_phi, mapper.time_coefficients(query_source))
    torch.testing.assert_close(factored, actual, atol=1e-8, rtol=1e-8)
