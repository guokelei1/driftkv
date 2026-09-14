"""State-conditioned, batched observed geometry and reusable read execution.

No new detector: all 225 selected coordinates and all layer/head factors remain.
The CUDA graph is a bounded shape slot; callers pay input copies explicitly.
"""
from types import SimpleNamespace
import torch
from hstu_kvcache.adaptation.reader import score


class Evidence:
    def __init__(self, packs, parameters):
        self.chol = torch.stack([p['observed'] for p in packs])
        self.xx = self.chol[..., :33, :33].contiguous()
        self.ox = self.chol[..., 33:, :33].contiguous()
        self.oo = self.chol[..., 33:, 33:].contiguous()
        self.center = torch.stack([p['read_center'] for p in parameters])
        self.scale = torch.stack([p['read_scale'] for p in parameters])

    def state(self, x):
        # Research runner has one UID per slot. All 36 independent factors retained.
        rhs = x[0, :, None].expand(6, 6, -1, -1)
        t = torch.linalg.solve_triangular(self.xx, rhs, upper=False)
        return self.ox @ t, t.square().sum(-2)

    def evaluate(self, observations, counts, active, prepared):
        h, d = prepared
        o = (torch.stack(observations)[:, 0].double()-self.center[:, None])/self.scale[:, None]
        rhs = o.transpose(-1, -2)[:, None]-h
        v = torch.linalg.solve_triangular(self.oo, rhs, upper=False)
        u = (v.square().sum(-2)+d).sqrt()
        return (u*counts[0].double()*active[0, :, None, None]).amax((0, 1))[None]

    def batched(self, x, observations, counts, active):
        o = (torch.stack(observations)[:, 0].double()-self.center[:, None])/self.scale[:, None]
        q = o.shape[1]
        rhs = torch.cat((x[0, :, None].expand(6, 6, -1, q),
                         o.transpose(-1, -2)[:, None].expand(-1, 6, -1, -1)), -2)
        v = torch.linalg.solve_triangular(self.chol, rhs, upper=False)
        u = v.square().sum(-2).sqrt()
        return (u*counts[0].double()*active[0, :, None, None]).amax((0, 1))[None]


def read(model, cache, panel, p, parameters):
    """Same native correction, retain only evidence inputs rather than full traces."""
    observations = []
    counts, active = p['counts'], p['active']
    def transform(layer, query, native):
        observed = native.transpose(1, 2).flatten(2)/counts[:, None, None]
        param = parameters[layer]
        normalized = (observed-param['read_center'].float())/param['read_scale'].float()
        conditional = torch.einsum('bqa,had->bhqd', normalized, param['read_weights'].float())
        delta = p['b'][:, layer, :, None]*counts[:, None, None, None]+query@(p['a'][:, layer]*counts[:, None, None, None])
        delta = delta+conditional*(counts*active[:, layer])[:, None, None, None]
        observations.append(observed)
        return native+delta
    z = score(model, cache, *panel, history_override=transform)[0]
    return z, observations


class Slot:
    """One shape at one target; input update cost is part of caller measurement."""
    def __init__(self, fn, cache, panel, p):
        self.cache = SimpleNamespace(k=cache.k.clone(), v=cache.v.clone())
        self.panel = tuple(v.clone() for v in panel)
        self.p = {k:p[k].clone() for k in ('latent', 'counts', 'active', 'b', 'a')}
        self.fn = fn
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):fn(self.cache, self.panel, self.p)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = fn(self.cache, self.panel, self.p)

    def load(self, cache, panel, p, dirty=True):
        if dirty:
            self.cache.k.copy_(cache.k); self.cache.v.copy_(cache.v)
            for k in self.p:self.p[k].copy_(p[k])
        for a,b in zip(self.panel, panel):a.copy_(b)

    def run(self, cache, panel, p, dirty=True):
        self.load(cache, panel, p, dirty)
        self.graph.replay()
        return self.output
