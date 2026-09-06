#!/usr/bin/env python3
"""Calibrate shared affine-query views progressively from lower to upper layers."""

import argparse
import json
from collections import defaultdict
from types import SimpleNamespace

import torch
from design import run
from design.data import ROOT

from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.models import HSTUKVCache


@torch.no_grad()
def fit_query_layers(current, scenes, mapper, args):
    states = [run.training_state(scene, current, mapper) for scene in scenes]
    sources = [state.pack_source(mapper.target) for state in states]
    features = torch.stack([mapper.features(source) for source in sources])
    counts = torch.stack([source.count.sum() for source in sources])
    groups = defaultdict(list)
    for index, state in enumerate(states):
        groups[state.cache.seq_len].append(index)
    records = []
    for layer, block in enumerate(current.blocks):
        query = features.new_zeros(len(scenes), mapper.heads, 64, mapper.head_dim)
        wanted_rates = torch.zeros_like(query)
        intercept, slopes = mapper.query_view(features)
        for length, indices in groups.items():
            for offset in range(0, len(indices), args.calibration_batch):
                batch = indices[offset:offset+args.calibration_batch]
                source = HSTUKVCache(torch.cat([states[i].cache.k for i in batch], 1),
                                    torch.cat([states[i].cache.v for i in batch], 1), length)
                teacher = HSTUKVCache(torch.cat([scenes[i].teacher.k for i in batch], 1),
                                     torch.cat([scenes[i].teacher.v for i in batch], 1), length)
                _, trace = score(current, source, torch.cat([scenes[i].candidates for i in batch]),
                    torch.cat([scenes[i].query_delta for i in batch]), trace=True,
                    response_delta=intercept[batch]*counts[batch, None, None],
                    response_query_delta=slopes[batch]*counts[batch, None, None, None, None])
                q = trace.queries[layer]
                wanted = history_read(block.attn, q, teacher.k[layer], teacher.v[layer])
                query[batch] = q
                wanted_rates[batch] = (wanted-trace.history_heads[layer])/counts[batch, None, None, None]
        mapper.query_center[layer] = query.mean((0, 2))
        mapper.query_scale[layer] = query.std((0, 2)).clamp_min(1e-4)
        x = ((query-mapper.query_center[layer, :, None])/mapper.query_scale[layer, :, None]).double()
        y = wanted_rates.double()
        centered = x-x.mean(2, keepdim=True)
        gram = centered.transpose(2, 3) @ centered
        regularizer = (gram.diagonal(dim1=-2, dim2=-1).sum(-1)/mapper.head_dim*1e-4).clamp_min(1e-8)
        system = gram+regularizer[..., None, None]*torch.eye(mapper.head_dim, device=gram.device, dtype=gram.dtype)
        rhs = centered.transpose(2, 3) @ (y-y.mean(2, keepdim=True))
        solution = torch.linalg.solve(system, rhs)
        torch.testing.assert_close(system @ solution, rhs, atol=1e-7, rtol=1e-6)
        intercept = y.mean(2)-(x.mean(2, keepdim=True) @ solution).squeeze(2)
        coefficients = torch.cat((intercept[:, None], solution.transpose(1, 2)), 1).float().flatten(2)
        record = mapper.fit_layer(layer, features, coefficients)
        record.update(target=mapper.target, calibration_scenes=len(scenes), fitting_queries_per_scene=64,
            query_scale_min=float(mapper.query_scale[layer].min()), query_scale_max=float(mapper.query_scale[layer].max()),
            supervision="affine coefficients fitted only on calibration teacher responses at actual queries; then shared source-to-coefficient fit")
        records.append(record)
        print(json.dumps(dict(phase="query_view_calibration", **record)), flush=True)
    return mapper.eval(), records, []


def main(cli):
    config = json.loads((ROOT / "results/design/v9_stratified_stable64_01/configuration.json").read_text())
    args = SimpleNamespace(**config)
    args.run_id, args.fit_users, args.lifetime_users = cli.run_id, cli.users, cli.users//4
    args.targets = cli.targets
    args.steps = 1
    args.query_affine = True
    args.temporal = args.source_confidence = args.second_moments = args.source_kernel = False
    args.trajectory_users = 4 if cli.users == 16 else 0
    args.prospective_resources = dict(expected_seconds=[45, 240] if cli.users == 16 else [90, 400],
        expected_gpu_mib=18000, estimate_basis="existing64 fit44s; six shared layer maps and per-scene32-dimensional coefficient solves; same64 query budget",
        scope="eager full four-component prototype; actual native writes, no per-user teacher at development")
    run.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--users", type=int, choices=(16, 64), default=64)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    main(parser.parse_args())
