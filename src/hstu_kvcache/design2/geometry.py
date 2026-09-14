"""Conditional ridge geometry in frozen Design 1 coordinates."""

import torch


def coordinates(query, observed_rate, parameters):
    q = (query.double()-parameters["query_center"])/parameters["query_scale"]
    q = torch.cat((torch.ones_like(q[..., :1]), q), -1)
    o = (observed_rate.double()-parameters["read_center"])/parameters["read_scale"]
    return q, o


def features(latent, query, observed_rate, parameters):
    q, o = coordinates(query, observed_rate, parameters)
    joint = torch.einsum("sr,shqj->shqrj", latent.double(), q).flatten(-2)
    return torch.cat((joint, o[:, None].expand(-1, q.shape[1], -1, -1)), -1)


def gram_sum(latent, query, observed_rate, counts, parameters, mu):
    """Return sum_s N_s²/mu * mean_query f f^T, without a teacher."""
    h = latent.double()
    q, o = coordinates(query, observed_rate, parameters)
    _, heads, queries, augmented = q.shape
    rank = h.shape[-1]
    w = counts.double().square()/mu
    outer = (h[:, :, None]*h[:, None, :]*w[:, None, None]).flatten(1)
    qgram = q.transpose(-2, -1)@q/queries
    gram = (outer.T@qgram.flatten(1)).reshape(rank, rank, heads, augmented, augmented)
    gram = gram.permute(2, 0, 3, 1, 4).reshape(heads, rank*augmented, rank*augmented)
    cross = torch.einsum("sr,shja->hrja", h*w[:, None], q.transpose(-2, -1)@o[:, None]/queries)
    cross = cross.reshape(heads, rank*augmented, o.shape[-1])
    native = torch.einsum("sqa,sqb,s->ab", o, o, w)/queries
    return torch.cat((torch.cat((gram, cross), -1),
                      torch.cat((cross.transpose(-2, -1), native[None].expand(heads, -1, -1)), -1)), -2)


def factorize(gram, scene_count, ridge=.01):
    h = gram/scene_count
    h = (h+h.transpose(-2, -1))/2
    h = h+ridge*torch.eye(h.shape[-1], device=h.device, dtype=h.dtype)
    chol = torch.linalg.cholesky(h)
    residual = (chol@chol.transpose(-2, -1)-h).abs().max()/h.abs().max()
    assert float(residual) < 1e-10
    return chol, float(residual)


def score_layer(latent, query, observed_rate, counts, active, parameters, chol):
    f = features(latent, query, observed_rate, parameters)
    # H=L L^T, f^T H^-1 f = ||L^-1 f||². No inverse or approximation.
    b, heads, queries, p = f.shape
    rhs = f.permute(1, 3, 0, 2).reshape(heads, p, b*queries)
    solved = torch.linalg.solve_triangular(chol, rhs, upper=False)
    rate = solved.square().sum(1).sqrt().reshape(heads, b, queries).permute(1, 0, 2)
    aggregate = rate*(counts.double()*active)[:, None, None]
    return aggregate.amax(1), rate.amax(1)
