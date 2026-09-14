"""Merge model reads across paths; retain state-local corrections and evidence."""
import torch
from hstu_kvcache.adaptation.reader import score


def execute(model, cache, panel, p, parameters, evidence, corrected, screened):
    observations = []
    def transform(layer, query, native):
        if corrected.numel():
            count = p['counts'][corrected]
            observed = native[corrected].transpose(1, 2).flatten(2)/count[:, None, None]
            param = parameters[layer]
            normalized = (observed-param['read_center'])/param['read_scale']
            conditional = torch.einsum('bqa,had->bhqd', normalized, param['read_weights'])
            delta = p['b'][corrected, layer, :, None]*count[:, None, None, None]
            delta = delta + query[corrected]@(p['a'][corrected, layer]*count[:, None, None, None])
            delta = delta + conditional*(count*p['active'][corrected, layer])[:, None, None, None]
        if screened.numel():
            observations.append(native[screened].transpose(1, 2).flatten(2)/p['counts'][screened, None, None])
        if corrected.numel():
            native = native.index_add(0, corrected, delta)
        return native
    z = score(model, cache, *panel, history_override=transform)[0]
    if not screened.numel():
        return z, z.new_empty((0, z.shape[1]), dtype=torch.float64)
    # Same original selected-coordinate triangular solve, now across task rows.
    o = torch.stack(observations, 1).double()
    o = (o-evidence.center[None, :, None])/evidence.scale[None, :, None]
    x = p['latent'][screened]
    q = o.shape[2]
    rhs = torch.cat((x[:, None, None, :, None].expand(-1, 6, 6, -1, q),
                     o.transpose(-1, -2)[:, :, None].expand(-1, -1, 6, -1, -1)), -2)
    v = torch.linalg.solve_triangular(evidence.chol[None], rhs, upper=False)
    u = v.square().sum(-2).sqrt()
    u = u*p['counts'][screened, None, None, None].double()*p['active'][screened, :, None, None]
    return z, u.amax((1, 2))
