"""The factorized direct-response solve matches its explicit observation design."""

import torch
from design.diagnose_summary_objective import fit_shared


def test_response_normal_equations_match_explicit_query_source_design():
    torch.manual_seed(17)
    h = torch.randn(7, 3, dtype=torch.double)
    h[:, 0] = 1
    query = torch.randn(7, 2, 5, 2, dtype=torch.double)
    wanted = torch.randn_like(query)
    counts = torch.tensor([7, 32, 64, 128, 256, 512, 1024], dtype=torch.double)
    weight = counts.square()/counts.square().mean()
    actual, _ = fit_shared(h, query, wanted, "response", counts)
    x = torch.cat((torch.ones_like(query[..., :1]), query), -1)
    expected = []
    for head in range(2):
        design = torch.einsum("sr,sqj->sqrj", h, x[:, head]).reshape(35, 9)
        weights = weight.repeat_interleave(5)
        gram = design.T @ (design*weights[:, None])/5 + .01*7*torch.eye(9, dtype=torch.double)
        rhs = design.T @ (wanted[:, head].reshape(35, 2)*weights[:, None])/5
        expected.append(torch.linalg.solve(gram, rhs).reshape(3, 3, 2))
    torch.testing.assert_close(actual, torch.stack(expected, 1), atol=1e-10, rtol=1e-10)
