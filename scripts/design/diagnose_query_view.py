#!/usr/bin/env python3
"""Teacher diagnostic for an affine response view at each actual head query.

Fit on the same 64 legal times and 16 items as the broad-time diagnostic, then
evaluate the original four groups. Per-scene teacher access is not a method.
"""

import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from design import diagnose_query_dispersion as driver
from design.data import ROOT
from design.run import training_state, write_json

from hstu_kvcache.adaptation import reader
from hstu_kvcache.models import HSTUKVCache


def affine_read(model, cache, items, delta, slopes, intercepts, counts, *, trace=False):
    original = reader.history_read
    layer_ids = {id(block.attn):layer for layer, block in enumerate(model.blocks)}
    native_heads = []

    def history_read(attention, query, key, value, *, count=None):
        native = original(attention, query, key, value, count=count)
        layer = layer_ids[id(attention)]
        correction = (query @ slopes[:, layer]+intercepts[:, layer, :, None])*counts[:, None, None, None]
        if trace:
            native_heads.append(native)
        return native+correction

    reader.history_read = history_read
    try:
        logits, result = reader.score(model, cache, items, delta, trace=trace)
    finally:
        reader.history_read = original
    return logits, result, native_heads


@torch.no_grad()
def diagnose(current, scenes, mapper, batch_size):
    states = [training_state(scene, current, mapper) for scene in scenes]
    sources = [state.pack_source(mapper.target) for state in states]
    features = torch.stack([mapper.features(source) for source in sources])
    counts = torch.stack([source.count.sum() for source in sources])
    active = mapper.active_layers(features)
    heads = current.cfg.num_heads
    dimension = mapper.width//heads
    groups = defaultdict(list)
    for index, state in enumerate(states):
        groups[state.cache.seq_len].append(index)
    rows, normal_errors, zero_errors = [], [], []
    for length, indices in groups.items():
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset:offset+batch_size]
            source = HSTUKVCache(torch.cat([states[i].cache.k for i in batch], 1),
                                torch.cat([states[i].cache.v for i in batch], 1), length)
            teacher = HSTUKVCache(torch.cat([scenes[i].teacher.k for i in batch], 1),
                                 torch.cat([scenes[i].teacher.v for i in batch], 1), length)
            items = torch.cat([scenes[i].candidates for i in batch])
            delta = torch.cat([scenes[i].query_delta for i in batch])
            assert items.shape[1] == 64
            n = counts[batch]
            exact = reader.score(current, teacher, items, delta)[0]
            values = dict(reuse=reader.score(current, source, items, delta)[0],
                stable=reader.score(current, source, items, delta,
                    response_delta=mapper.rates(features[batch])*n[:, None, None],
                    response_time_delta=mapper.time_coefficients(features[batch])*n[:, None, None, None])[0])
            slopes = features.new_zeros(len(batch), mapper.layers, heads, dimension, dimension)
            intercepts = features.new_zeros(len(batch), mapper.layers, heads, dimension)
            zero = affine_read(current, source, items, delta, slopes, intercepts, n)[0]
            torch.testing.assert_close(zero, values["reuse"], atol=0, rtol=0)
            zero_errors.append(float((zero-values["reuse"]).abs().max()))
            fraction = torch.linspace(0, 1, 64, device=delta.device)
            query_times = (delta.max(1).values.log()[:, None]*fraction).exp().round().clamp_min(1)
            query_times = torch.minimum(query_times, delta.max(1).values[:, None])
            assert (query_times >= 1).all() and (query_times <= delta.max(1).values[:, None]).all()
            candidates = items[:, :16].repeat(1, 64)
            query_delta = query_times.repeat_interleave(16, 1)
            for _ in range(6):
                _, trace, native_heads = affine_read(current, source, candidates, query_delta, slopes, intercepts, n, trace=True)
                for layer, (block, query, native) in enumerate(zip(current.blocks, trace.queries, native_heads, strict=True)):
                    wanted = reader.history_read(block.attn, query, teacher.k[layer], teacher.v[layer])
                    q = query.double()
                    y = ((wanted-native)/n[:, None, None, None]).double()
                    center, scale = q.mean(2, keepdim=True), q.std(2, keepdim=True).clamp_min(1e-7)
                    x = (q-center)/scale
                    gram = x.transpose(2, 3) @ x
                    regularizer = (gram.diagonal(dim1=-2, dim2=-1).sum(-1)/dimension*1e-4).clamp_min(1e-8)
                    system = gram+regularizer[..., None, None]*torch.eye(dimension, device=gram.device, dtype=gram.dtype)
                    rhs = x.transpose(2, 3) @ (y-y.mean(2, keepdim=True))
                    solution = torch.linalg.solve(system, rhs)
                    error = system @ solution-rhs
                    normal_errors.append(float(error.abs().max()/rhs.abs().max().clamp_min(1e-20)))
                    torch.testing.assert_close(system @ solution, rhs, atol=1e-7, rtol=1e-6)
                    slope = solution/scale.transpose(2, 3)
                    intercept = y.mean(2)-(center @ slope).squeeze(2)
                    slopes[:, layer] = slope.float()*active[batch, layer, None, None, None]
                    intercepts[:, layer] = intercept.float()*active[batch, layer, None, None]
            values["query_affine_oracle"] = affine_read(current, source, items, delta, slopes, intercepts, n)[0]
            assert all(torch.isfinite(value).all() for value in values.values())
            errors = {name:(value-exact).square().mean(1).cpu().tolist() for name, value in values.items()}
            for j, index in enumerate(batch):
                rows.append(dict(uid=scenes[index].uid, scene=index, retained_events=length,
                                 **{name:error[j] for name, error in errors.items()}))
    frame = pd.DataFrame(rows)
    per_user = frame.groupby("uid")[["reuse", "stable", "query_affine_oracle"]].mean()
    return dict(users=len(per_user), scenes=len(scenes), scene_diagnostics=rows,
        uid_equal_logit_mse=per_user.mean().to_dict(),
        uid_fraction_improved_vs_stable=float((per_user.query_affine_oracle < per_user.stable).mean()),
        maximum_normal_equation_relative_residual=max(normal_errors), zero_view_maximum_logit_difference=max(zero_errors),
        scope="per-scene teacher intervention at actual corrected queries, six passes; broad-time training shares probe items/endpoints; not deployable quality or a strict upper bound")


def main(cli):
    out = ROOT / "results/design" / cli.run_id
    files = [Path(__file__), ROOT / "scripts/design/diagnose_query_dispersion.py",
             ROOT / "scripts/design/run.py", ROOT / "scripts/design/data.py",
             *sorted((ROOT / "src/hstu_kvcache/adaptation").glob("*.py"))]
    args = SimpleNamespace(run_id=cli.run_id, reference="v9_stratified_stable64_01", users=cli.users,
        independent=False, targets=cli.targets,
        override_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prospective_resources=dict(expected_seconds=[15, 120] if cli.users == 8 else [100, 400], expected_gpu_mib=16000,
            estimate_basis="six-pass broad-time64 diagnostic121s plus per-head32-dimensional solves",
            scope="teacher diagnostic, no method or population cost claim"))
    driver.dispersion = diagnose
    try:
        driver.main(args)
    finally:
        if out.exists():
            with tarfile.open(out / "query_view_source.tar.gz", "w:gz") as archive:
                for path in files:
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
    summary = json.loads((out / "summary.json").read_text())
    summary["passing_checks"] = "zero affine view bitwise equal to native reader, legal times, finite outputs and double-precision normal equations"
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--users", type=int, choices=(8, 64), required=True)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    main(parser.parse_args())
