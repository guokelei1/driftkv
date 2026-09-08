#!/usr/bin/env python3
"""One matched prototype: actual native response versus producer-mean response input."""

import argparse
import json
import time
from collections import defaultdict
from types import SimpleNamespace

import pandas as pd
import torch
from design.diagnose_decoder_closure import SMALL, main
from design.diagnose_query_holdout import batch_cache, panels
from design.diagnose_summary_objective import install_layer
from design.run import training_state, write_json

from hstu_kvcache.adaptation.reader import history_read, score


def fit_joint(latent, query, wanted, observed, counts, *, coordinates=None, count_square_mean=None):
    """Same aggregate-response ridge for both arms, joint source/query + read input."""
    h, q, y, observed = latent.double(), query.double(), wanted.double(), observed.double()
    center, scale = (q.mean((0, 2), keepdim=True), q.std((0, 2), keepdim=True).clamp_min(1e-4)) if coordinates is None else (coordinates["query_center"], coordinates["query_scale"])
    q = (q-center)/scale
    x = torch.cat((torch.ones_like(q[..., :1]), q), -1)
    read_center, read_scale = (observed.mean((0, 1)), observed.std((0, 1)).clamp_min(1e-4)) if coordinates is None else (coordinates["read_center"], coordinates["read_scale"])
    observed = (observed-read_center)/read_scale
    scenes, heads, queries, dimension = q.shape
    rank, augmented, read_dim = h.shape[-1], dimension+1, observed.shape[-1]
    count_square_mean = counts.double().square().mean() if count_square_mean is None else count_square_mean
    weight = counts.double().square()/count_square_mean
    wh = h*weight[:, None]
    qgram = x.transpose(-2, -1)@x/queries
    outer = (wh[:, :, None]*h[:, None]).flatten(1)
    gram = (outer.T@qgram.flatten(1)).reshape(rank, rank, heads, augmented, augmented)
    gram = gram.permute(2, 0, 3, 1, 4).reshape(heads, rank*augmented, rank*augmented)
    qty = x.transpose(-2, -1)@y/queries
    rhs = torch.einsum("sr,shjd->hrjd", wh, qty).reshape(heads, rank*augmented, dimension)
    qta = x.transpose(-2, -1)@observed[:, None]/queries
    cross = torch.einsum("sr,shja->hrja", wh, qta).reshape(heads, rank*augmented, read_dim)
    read_gram = torch.einsum("sqa,sqb,s->ab", observed, observed, weight)/queries
    read_rhs = torch.einsum("sqa,shqd,s->had", observed, y, weight)/queries
    system = torch.cat((torch.cat((gram, cross), -1),
                        torch.cat((cross.transpose(-2, -1), read_gram[None].expand(heads, -1, -1)), -1)), -2)
    system = system+.01*scenes*torch.eye(system.shape[-1], dtype=system.dtype, device=system.device)
    rhs = torch.cat((rhs, read_rhs), 1)
    solution = torch.linalg.solve(system, rhs)
    torch.testing.assert_close(system@solution, rhs, atol=1e-7, rtol=1e-6)
    weights = solution[:, :rank*augmented].reshape(heads, rank, augmented, dimension).transpose(0, 1)
    native_weights = solution[:, rank*augmented:]
    prediction = torch.einsum("shqj,sr,rhjd->shqd", x, h, weights)+torch.einsum("sqa,had->shqd", observed, native_weights)
    params = dict(weights=weights, query_center=center, query_scale=scale,
                  read_weights=native_weights, read_center=read_center, read_scale=read_scale)
    stats = dict(fitting_rate_mse=float((prediction-y).square().mean()),
                 fitting_aggregate_objective=float(((prediction-y).square().mean((1, 2, 3))*weight).mean()),
                 normal_relative_residual=float((system@solution-rhs).abs().max()/rhs.abs().max().clamp_min(1e-30)),
                 shared_parameters=int(solution.numel()), rank=rank-1, read_dimension=read_dim, ridge=.01,
                 count_square_mean=float(count_square_mean), fitting_scenes=scenes, fitting_queries=queries,
                 ridge_objective=float(.01*solution.square().sum()/(heads*dimension)))
    return params, stats


def solver_canary(device):
    generator = torch.Generator(device=device).manual_seed(717)
    def random(*shape):
        return torch.randn(*shape, device=device, dtype=torch.float64, generator=generator)
    h, q, y, observed = random(5, 3), random(5, 2, 7, 3), random(5, 2, 7, 3), random(5, 7, 4)
    h[:, 0] = 1
    counts = torch.tensor([16, 17, 128, 512, 1024], device=device)
    p, _ = fit_joint(h, q, y, observed, counts)
    q = (q-p["query_center"])/p["query_scale"]
    x = torch.cat((torch.ones_like(q[..., :1]), q), -1)
    observed = (observed-p["read_center"])/p["read_scale"]
    design = torch.cat((torch.einsum("sr,shqj->shqrj", h, x).flatten(-2), observed[:, None].expand(-1, 2, -1, -1)), -1)
    w = counts.double().square()/counts.double().square().mean()
    gram = torch.einsum("shqa,shqb,s->hab", design, design, w)/7
    rhs = torch.einsum("shqa,shqd,s->had", design, y, w)/7
    reference = torch.linalg.solve(gram+.05*torch.eye(gram.shape[-1], device=device, dtype=torch.float64), rhs)
    actual = torch.cat((p["weights"].transpose(0, 1).flatten(1, 2), p["read_weights"]), 1)
    torch.testing.assert_close(actual, reference, atol=1e-8, rtol=1e-7)


def producer_mean_cache(sources, target):
    payloads, masses = [], []
    for source in sources:
        values, counts = [], []
        for producer in range(target+1):
            mass = source.count*(source.producer[:, None] == producer)
            count = mass.sum()
            values.append((source.payload*mass[..., None, None, None]).sum((0, 1))/count.clamp_min(1))
            counts.append(count)
        payloads.append(torch.stack(values))
        masses.append(torch.stack(counts))
    payload = torch.stack(payloads)  # B,P,L,KV,W
    return SimpleNamespace(k=payload[:, :, :, 0].permute(2, 0, 1, 3),
                           v=payload[:, :, :, 1].permute(2, 0, 1, 3)), torch.stack(masses)


def read_input(model, cache, panel, b, a, parameters, counts, active, mean_cache=None, mean_counts=None, capture=True):
    observations, corrections = [], []
    def transform(layer, query, native):
        observed = native if mean_cache is None else history_read(model.blocks[layer].attn, query,
            mean_cache.k[layer], mean_cache.v[layer], count=mean_counts)
        observed = observed.transpose(1, 2).flatten(2)/counts[:, None, None]
        delta = torch.zeros_like(native)
        if layer < len(parameters):
            p = parameters[layer]
            normalized = (observed-p["read_center"].float())/p["read_scale"].float()
            conditional = torch.einsum("bqa,had->bhqd", normalized, p["read_weights"].float())
            delta = b[:, layer, :, None]*counts[:, None, None, None]+query@(a[:, layer]*counts[:, None, None, None])
            delta = delta+conditional*(counts*active[:, layer])[:, None, None, None]
        if capture:
            observations.append(observed)
            corrections.append(delta)
        return native+delta
    z, trace = score(model, cache, *panel, trace=capture, history_override=transform)
    return z, trace, observations, corrections


def experiment(current, scenes, mapper, history, cutover, fit_uids, uid_count, batch_size, out, target, canary):
    """Joint fixed-PCA32 affine +192 read-input regression, native vs source-mean control.

    Exactly one input contrast. Both arms have the same parameters, ridge .01,
    aggregate-response objective and fit64 query budget; no diagnostic teacher
    enters fitting or the serving predictor. No probe or latent-width change.
    """
    if canary:
        solver_canary(next(current.parameters()).device)
    states = [training_state(scene, current, mapper) for scene in scenes]
    sources = [state.pack_source(target) for state in states]
    base = torch.stack([mapper.features(source) for source in sources])
    counts = torch.stack([source.count.sum() for source in sources])
    active = mapper.active_layers(base)
    projection = torch.load(SMALL / f"source_projection_mean_m{target}.pt", map_location=base.device, weights_only=True)
    latent = (base.double()-projection["center"])/projection["scale"]@projection["projection"]
    latent = torch.cat((torch.ones_like(latent[:, :1]), latent), -1)
    fit_mask = torch.tensor([s.uid in fit_uids for s in scenes], device=base.device)
    fitting = fit_mask.nonzero().flatten().tolist()
    groups, panel_list, metadata = defaultdict(list), [], []
    stride = 2 if target == 1 else 3
    for i, state in enumerate(states):
        groups[state.cache.seq_len].append(i)
        panel_list.append(panels(history, scenes[i].uid, cutover, float(scenes[i].query_delta.max()), base.device)[0])
        metadata.append(dict(target=target, uid=scenes[i].uid, scene=i,
            group="fitting_uid" if scenes[i].uid in fit_uids else "diagnostic_uid",
            state_group="continuous_cutover" if i < uid_count*stride and i % stride == 0 else "auxiliary"))
    batches = [indices[j:j+batch_size] for indices in groups.values() for j in range(0, len(indices), batch_size)]
    def panel(indices, name):
        return tuple(torch.cat([panel_list[i][name][j] for i in indices]) for j in range(2))
    means, masses = producer_mean_cache(sources, target)
    torch.testing.assert_close(masses.sum(-1), counts)
    rows, local_rows, fit_rows, cost = [], [], [], defaultdict(float)
    shape = (len(scenes), 6, 6, 32)
    for method in ("native_input", "summary_input"):
        b, a, parameters = base.new_zeros(shape), base.new_zeros(*shape, 32), []
        def read(indices, name, method=method, b=b, a=a, parameters=parameters):
            source = batch_cache([states[i].cache for i in indices])
            mean_cache = SimpleNamespace(k=means.k[:, indices], v=means.v[:, indices]) if method == "summary_input" else None
            return read_input(current, source, panel(indices, name), b[indices], a[indices], parameters,
                counts[indices], active[indices], mean_cache, masses[indices])
        for l in range(6):
            start = time.perf_counter()
            query, wanted = base.new_zeros(len(scenes), 6, 64, 32), base.new_zeros(len(scenes), 6, 64, 32)
            observed = base.new_zeros(len(scenes), 64, 192)
            for batch in batches:
                indices = [i for i in batch if bool(fit_mask[i])]
                if not indices:
                    continue
                _, trace, obs, _ = read(indices, "fit64")
                teacher = batch_cache([scenes[i].teacher for i in indices])
                query[indices], observed[indices] = trace.queries[l], obs[l]
                wanted[indices] = (history_read(current.blocks[l].attn, trace.queries[l], teacher.k[l], teacher.v[l])-
                                   trace.history_heads[l])/counts[indices, None, None, None]
            p, record = fit_joint(latent[fitting], query[fitting], wanted[fitting], observed[fitting], counts[fitting])
            parameters.append(p)
            b[:, l], a[:, l] = install_layer(latent, p["weights"], p["query_center"], p["query_scale"], active[:, l])
            torch.cuda.synchronize()
            cost[method+"_calibration_seconds"] += time.perf_counter()-start
            fit_rows.append(dict(target=target, method=method, layer=l, **record))
        torch.save([{k: v.cpu() for k, v in p.items()} for p in parameters], out / f"translator_{method}_m{target}.pt")
        for batch_id, indices in enumerate(batches):
            teacher = batch_cache([scenes[i].teacher for i in indices])
            source = batch_cache([states[i].cache for i in indices])
            raw = dict(scene_indices=indices)
            for name in ("fit64", "held64"):
                exact = score(current, teacher, *panel(indices, name))[0]
                torch.cuda.synchronize()
                start = time.perf_counter()
                z, trace, _, corrections = read(indices, name)
                torch.cuda.synchronize()
                cost[method+"_diagnostic_read_seconds"] += time.perf_counter()-start
                raw[name] = dict(exact=exact.cpu(), prediction=z.cpu())
                error = (z.double()-exact.double()).square().mean(-1)
                for j, i in enumerate(indices):
                    rows.append(dict(**metadata[i], panel=name, method=method, path="actual", logit_mse=float(error[j])))
                for l, q in enumerate(trace.queries):
                    eps = corrections[l]-(history_read(current.blocks[l].attn, q, teacher.k[l], teacher.v[l])-trace.history_heads[l])
                    local_error = eps.double().square().mean((1, 2, 3))
                    for j, i in enumerate(indices):
                        local_rows.append(dict(**metadata[i], panel=name, method=method, layer=l, response_mse=float(local_error[j])))
                if method == "native_input":
                    reuse = score(current, source, *panel(indices, name))[0]
                    shared_b, shared_a = mapper.query_view(base[indices])
                    shared = score(current, source, *panel(indices, name), response_delta=shared_b*counts[indices, None, None],
                        response_query_delta=shared_a*counts[indices, None, None, None, None])[0]
                    for reference, prediction in (("reuse", reuse), ("shared15", shared)):
                        raw[reference+"_"+name] = prediction.cpu()
                        err = (prediction.double()-exact.double()).square().mean(-1)
                        for j, i in enumerate(indices):
                            rows.append(dict(**metadata[i], panel=name, method=reference, path="actual", logit_mse=float(err[j])))
            torch.save(raw, out / f"raw_{method}_m{target}_batch{batch_id:03d}.pt")
        print(json.dumps(dict(target=target, method=method, status="complete")), flush=True)
    pd.DataFrame(rows).to_parquet(out / f"outputs_m{target}.parquet", index=False)
    pd.DataFrame(local_rows).to_parquet(out / f"responses_m{target}.parquet", index=False)
    write_json(out / f"fits_m{target}.json", fit_rows)
    write_json(out / f"cost_m{target}.json", dict(cost, extra_persistent_summary_bytes=0,
        extra_shared_read_weights=6*6*192*32, extra_read_matrix_multiply_macs_per_query=6*192*192,
        native_extra_kv_scans=0, read_cost_scope="diagnostic batching/capture included, not a production service or Exact-All recomputation comparison"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fit-users", type=int, default=64)
    parser.add_argument("--diagnostic-users", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--estimate-seconds", type=int, required=True)
    main(parser.parse_args(), experiment)
