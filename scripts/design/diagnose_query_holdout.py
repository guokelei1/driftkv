#!/usr/bin/env python3
"""Matched-budget affine teacher with strict item holdout and actual-query geometry.

No M2 teacher or adaptation. Native M2 writes remain in the ordinary lineage.
These unlabeled, pre-release teacher interventions are not deployable quality.
"""

import argparse
import hashlib
import json
import tarfile
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from design.data import DAY, KNOWN_ITEMS, ROOT, diagnostic_admissions, frozen_model, histories
from design.run import (
    initialize_calibration,
    make_scenes,
    new_translator,
    replay_calibration,
    timed,
    training_state,
    write_json,
)
from insight_two.common import CUTOVER_DAYS

from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.models import HSTUKVCache


def panels(history, uid, cutover, gap, device):
    """Three disjoint item banks, balanced over causal recent and random items."""
    recent = list(dict.fromkeys(int(i) for i in history.prefix(uid, cutover, 1024)[0][::-1]
                               if 0 < i < KNOWN_ITEMS))[:24]
    rng = np.random.default_rng(170907 + uid)
    bank = recent.copy()
    while len(bank) < 48:
        item = int(rng.integers(1, KNOWN_ITEMS))
        if item not in bank:
            bank.append(item)
    # Every third item distributes the scarce real-history items across banks.
    banks = [bank[i::3] for i in range(3)]
    assert all(not set(a) & set(b) for i, a in enumerate(banks) for b in banks[i+1:])
    result = {}
    for name, items, shift in (("fit64", banks[0], 0), ("held64", banks[1], .5),
                               ("refit64", banks[2], .25)):
        fraction = (torch.arange(64, device=device) + shift) / 64
        delta = (fraction * np.log(gap)).exp().round().clamp(1, gap)
        result[name] = (torch.tensor([items * 4], device=device), delta[None])
    dense_times = result["fit64"][1].repeat_interleave(16, 1)
    result["dense1024"] = (torch.tensor([banks[0] * 64], device=device), dense_times)
    overlap = len(set(result["fit64"][1].flatten().tolist()) & set(result["held64"][1].flatten().tolist()))
    return result, dict(gap_seconds=gap, fit_items=banks[0], held_items=banks[1], refit_items=banks[2],
                        shared_integer_times=overlap,
                        unique_fit_times=len(set(result["fit64"][1].flatten().tolist())),
                        unique_held_times=len(set(result["held64"][1].flatten().tolist())))


def solve_affine(query, wanted, center=None, scale=None):
    """Ridge in fixed query coordinates; return both normalized and raw views."""
    q, y = query.double(), wanted.double()
    if center is None:
        center = q.mean(2, keepdim=True)
        scale = q.std(2, keepdim=True).clamp_min(1e-4)
    x = (q - center) / scale
    xc = x - x.mean(2, keepdim=True)
    gram = xc.transpose(-2, -1) @ xc
    dim = q.shape[-1]
    ridge = (gram.diagonal(dim1=-2, dim2=-1).sum(-1) / dim * 1e-4).clamp_min(1e-8)
    system = gram + ridge[..., None, None] * torch.eye(dim, device=q.device, dtype=q.dtype)
    rhs = xc.transpose(-2, -1) @ (y - y.mean(2, keepdim=True))
    slope = torch.linalg.solve(system, rhs)
    torch.testing.assert_close(system @ slope, rhs, atol=1e-7, rtol=1e-6)
    intercept = y.mean(2) - (x.mean(2, keepdim=True) @ slope).squeeze(2)
    coefficients = torch.cat((intercept[:, :, None], slope), 2)
    raw_slope = slope / scale.transpose(-2, -1)
    raw_intercept = intercept - (center @ raw_slope).squeeze(2)
    return raw_intercept.float(), raw_slope.float(), coefficients, center, scale


def geometry(query, center, scale):
    x = (query.double() - center) / scale
    x = torch.cat((torch.ones_like(x[..., :1]), x), -1)
    singular = torch.linalg.svdvals(x)
    energy = singular.square()
    probs = energy / energy.sum(-1, keepdim=True).clamp_min(1e-30)
    effective_rank = (-(probs * probs.clamp_min(1e-30).log()).sum(-1)).exp()
    rank = (singular > singular[..., :1] * 1e-6).sum(-1)
    condition = singular[..., 0] / singular[..., -1].clamp_min(1e-30)
    return singular, effective_rank, rank, condition


def batch_cache(caches):
    length = caches[0].seq_len
    assert all(c.seq_len == length for c in caches)
    return HSTUKVCache(torch.cat([c.k for c in caches], 1), torch.cat([c.v for c in caches], 1), length)


def view_score(model, cache, panel, intercept, slopes, counts, trace=False):
    return score(model, cache, *panel, trace=trace,
                 response_delta=intercept.flatten(2) * counts[:, None, None],
                 response_query_delta=slopes * counts[:, None, None, None, None])


def response_change(model, source, teacher, trace, layer, counts):
    q = trace.queries[layer]
    target = history_read(model.blocks[layer].attn, q, teacher.k[layer], teacher.v[layer])
    return (target - trace.history_heads[layer]) / counts[:, None, None, None]


def update_decomposition(attn, query, source_k, source_v, teacher_k, teacher_v):
    # This is an algebra audit. Large nearly cancelling FP32 terms otherwise
    # obscure the identity; reader predictions themselves remain native FP32.
    query = query.double()
    b, h, _, d = query.shape
    k_s, v_s, k_c, v_c = [v.double().reshape(b, -1, h, d).transpose(1, 2)
                           for v in (source_k, source_v, teacher_k, teacher_v)]
    w_s = attn._activate(query @ k_s.transpose(-2, -1) * attn.scale)
    w_c = attn._activate(query @ k_c.transpose(-2, -1) * attn.scale)
    value = w_s @ (v_c - v_s)
    key = (w_c - w_s) @ v_c
    total = w_c @ v_c - w_s @ v_s
    # Summation-order roundoff is measured, not interpreted as mechanism error.
    residual = (total - value - key).square().mean((1, 2, 3))
    signal = total.square().mean((1, 2, 3))
    torch.testing.assert_close(total, value + key, atol=1e-8, rtol=1e-8)
    return dict(value_energy=value.square().mean((1, 2, 3)),
                key_energy=key.square().mean((1, 2, 3)),
                cross_energy=2 * (value * key).mean((1, 2, 3)),
                total_energy=signal, decomposition_residual=residual)


@torch.no_grad()
def diagnose(current, scenes, mapper, history, cutover, fit_uids, batch_size):
    states = [training_state(s, current, mapper) for s in scenes]
    sources = [s.pack_source(mapper.target) for s in states]
    features = torch.stack([mapper.features(s) for s in sources])
    counts = torch.stack([s.count.sum() for s in sources])
    groups = defaultdict(list)
    for index, state in enumerate(states):
        groups[state.cache.seq_len].append(index)
    rows, geometry_rows, response_rows, panel_rows = [], [], [], []
    heads, dimension, layers = mapper.heads, mapper.head_dim, mapper.layers
    for length, indices in groups.items():
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset:offset+batch_size]
            source = batch_cache([states[i].cache for i in batch])
            teacher = batch_cache([scenes[i].teacher for i in batch])
            n = counts[batch]
            raw_panels = []
            for i in batch:
                panel, meta = panels(history, scenes[i].uid, cutover,
                                     float(scenes[i].query_delta.max()), n.device)
                raw_panels.append(panel)
                panel_rows.append(dict(scene=i, uid=scenes[i].uid, **meta))
            panel = {name: tuple(torch.cat([p[name][j] for p in raw_panels]) for j in range(2))
                     for name in raw_panels[0]}
            shape = (len(batch), layers, heads, dimension)
            zero_b = features.new_zeros(shape)
            zero_a = features.new_zeros(*shape, dimension)
            exact = {name: score(current, teacher, *panel[name])[0] for name in ("fit64", "held64")}
            reuse = {name: score(current, source, *panel[name])[0] for name in exact}
            zero = view_score(current, source, panel["held64"], zero_b, zero_a, n)[0]
            torch.testing.assert_close(zero, reuse["held64"], atol=0, rtol=0)
            shared_b, shared_a = mapper.query_view(features[batch])
            shared_b = shared_b.reshape(shape)
            predictions = {"reuse": reuse,
                           "shared15": {name: view_score(current, source, panel[name], shared_b, shared_a, n)[0]
                                        for name in exact}}
            for budget in ("fit64", "dense1024"):
                intercept, slopes = zero_b.clone(), zero_a.clone()
                for layer in range(layers):
                    _, trace = view_score(current, source, panel[budget], intercept, slopes, n, True)
                    q = trace.queries[layer]
                    wanted = response_change(current, source, teacher, trace, layer, n)
                    b, a, coefficients, center, scale = solve_affine(q, wanted)
                    if budget == "fit64":
                        # Same lower-layer view and same coordinates isolate coefficient instability.
                        _, other_trace = view_score(current, source, panel["refit64"], intercept, slopes, n, True)
                        other_wanted = response_change(current, source, teacher, other_trace, layer, n)
                        _, _, other_coef, _, _ = solve_affine(other_trace.queries[layer], other_wanted, center, scale)
                        _, held_trace = view_score(current, source, panel["held64"], intercept, slopes, n, True)
                        held_q = (held_trace.queries[layer].double() - center) / scale
                        held_x = torch.cat((torch.ones_like(held_q[..., :1]), held_q), -1)
                        coef_delta = other_coef - coefficients
                        coefficient_error = coef_delta.square().mean((-2, -1))
                        coefficient_energy = coefficients.square().mean((-2, -1))
                        function_error = (held_x @ coef_delta).square().mean((-2, -1))
                        function_energy = (held_x @ coefficients).square().mean((-2, -1))
                        singular, erank, rank, cond = geometry(q, center, scale)
                        for j, i in enumerate(batch):
                            for head in range(heads):
                                geometry_rows.append(dict(uid=scenes[i].uid, scene=i, layer=layer, head=head,
                                    singular_values=singular[j, head].cpu().tolist(),
                                    effective_rank=float(erank[j, head]), numeric_rank=int(rank[j, head]),
                                    condition_number=float(cond[j, head]),
                                    coefficient_refit_mse=float(coefficient_error[j, head]),
                                    coefficient_energy=float(coefficient_energy[j, head]),
                                    held_function_refit_mse=float(function_error[j, head]),
                                    held_function_energy=float(function_energy[j, head])))
                    intercept[:, layer], slopes[:, layer] = b, a
                method = "oracle64" if budget == "fit64" else "oracle1024"
                predictions[method] = {}
                for panel_name in exact:
                    logits, trace = view_score(current, source, panel[panel_name], intercept, slopes, n, True)
                    predictions[method][panel_name] = logits
                    for layer, block in enumerate(current.blocks):
                        wanted = response_change(current, source, teacher, trace, layer, n)
                        estimated = trace.corrections[layer] / n[:, None, None, None]
                        error = (wanted - estimated).square().mean((1, 2, 3))
                        energy = wanted.square().mean((1, 2, 3))
                        decomposition = (update_decomposition(block.attn, trace.queries[layer],
                            source.k[layer], source.v[layer], teacher.k[layer], teacher.v[layer])
                            if method == "oracle64" and panel_name == "held64" else {})
                        for j, i in enumerate(batch):
                            response_rows.append(dict(uid=scenes[i].uid, scene=i, layer=layer,
                                method=method, panel=panel_name, error=float(error[j]), energy=float(energy[j]),
                                **{name: float(value[j] / n[j].square()) for name, value in decomposition.items()}))
            for j, i in enumerate(batch):
                for method, values in predictions.items():
                    for panel_name, logits in values.items():
                        assert torch.isfinite(logits).all()
                        rows.append(dict(uid=scenes[i].uid, scene=i, retained_events=length,
                            release_writes=states[i].writes_since_release,
                            producer_counts={str(p): float(sources[i].count[sources[i].producer == p].sum())
                                             for p in sources[i].producer.unique().tolist()},
                            group="fitting_uid" if scenes[i].uid in fit_uids else "diagnostic_uid",
                            method=method, panel=panel_name,
                            logit_mse=float((logits[j] - exact[panel_name][j]).square().mean()),
                            probability_mse=float((logits[j].sigmoid() - exact[panel_name][j].sigmoid()).square().mean())))
    return rows, geometry_rows, response_rows, panel_rows


def aggregate(frame):
    user = frame.groupby(["target", "group", "method", "panel", "uid"])[["logit_mse", "probability_mse"]].mean().reset_index()
    result = []
    for keys, group in user.groupby(["target", "group", "method", "panel"]):
        value = group.logit_mse
        result.append(dict(zip(("target", "group", "method", "panel"), keys, strict=True),
                           users=len(group), logit_mse=float(value.mean()), probability_mse=float(group.probability_mse.mean()),
                           logit_mse_median=float(value.median()), logit_mse_p90=float(value.quantile(.9)),
                           max_uid_error_fraction=float(value.max() / max(value.sum(), 1e-30))))
    return result


@torch.no_grad()
def main(cli):
    out = ROOT / "results/design" / cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    reference = ROOT / "results/design/v15_query_view64_01"
    config = json.loads((reference / "configuration.json").read_text())
    profile = json.loads((ROOT / "results/design/pre_release_auc128_01/configuration.json").read_text())
    fit = config["calibration_uids"][:cli.fit_users]
    diagnostic = profile["calibration_uids"][:cli.diagnostic_users]
    split = json.loads((ROOT / config["split_path"]).read_text())
    assert not set(fit) & set(diagnostic)
    assert not (set(fit) | set(diagnostic)) & (set(split["development"]) | set(split["confirmation"]))
    lifetime_fit, lifetime_diagnostic = fit[:len(fit)//4], diagnostic[:len(diagnostic)//4]
    uids = lifetime_fit + lifetime_diagnostic + fit[len(lifetime_fit):] + diagnostic[len(lifetime_diagnostic):]
    args = SimpleNamespace(**config)
    args.query_affine = False  # make_scenes leaves one causal gap; panels are explicit here.
    args.lifetime_users = len(lifetime_fit) + len(lifetime_diagnostic)
    files = [Path(__file__), ROOT / "scripts/design/run.py", ROOT / "scripts/design/data.py",
             *sorted((ROOT / "src/hstu_kvcache/adaptation").glob("*.py"))]
    write_json(out / "configuration.json", dict(vars(cli), fitting_uids=fit, diagnostic_uids=diagnostic,
        adaptation_targets=[1, 3, 4, 5], no_op_targets=[2], confirmation_read=False,
        frozen_backbone_seed=17, actual_operator="legacy/ELU+1", m5_scope="E14_partial",
        prospective_seconds=cli.estimate_seconds, expected_gpu_mib=18000,
        admissions=diagnostic_admissions(5), checkpoint_inputs=config["checkpoint_inputs"],
        split_sha256=hashlib.sha256((ROOT / config["split_path"]).read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        query_protocol="64 matched pairs; disjoint16 items held out;1024 teacher extra;64 refit third item bank; no labels",
        scope="per-scene teacher diagnostic, not shared fitting or service AUC;128 previously inspected diagnostic users",
        lifetime_users=args.lifetime_users))
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    start, ledger = time.perf_counter(), defaultdict(float)
    device = torch.device("cuda:0")
    all_rows = []
    try:
        history = timed(lambda: histories(uids, CUTOVER_DAYS[-1] + 1), ledger, "history_io")
        previous = timed(lambda: frozen_model(0, device), ledger, "model_io")
        states, early = initialize_calibration(previous, history, uids, CUTOVER_DAYS[0]*DAY, args, ledger)
        for target in range(1, 6):
            current = timed(partial(frozen_model, target, device), ledger, "model_io")
            cutover = CUTOVER_DAYS[target-1]*DAY
            if target != 2:
                mapper_args = SimpleNamespace(**vars(args))
                mapper_args.query_affine = True
                mapper = new_translator(mapper_args, target, device)
                saved = torch.load(reference / f"translator_v{target}.pt", map_location=device, weights_only=False)
                mapper.load_state_dict(saved["state_dict"])
                mapper.supported_producers = set(saved["supported_producers"])
                scenes = make_scenes(current, history, uids, states, early, cutover, ledger, previous, target, args)
                records = timed(partial(diagnose, current, scenes, mapper, history, cutover, set(fit), cli.batch_size),
                                ledger, "query_holdout_diagnostic")
                for name, rows in zip(("outputs", "geometry", "responses", "panels"), records, strict=True):
                    frame = pd.DataFrame(rows).assign(target=target)
                    frame.to_parquet(out / f"{name}_m{target}.parquet", index=False)
                    if name == "outputs":
                        all_rows.extend(frame.to_dict("records"))
                del scenes
                print(json.dumps(dict(target=target, status="complete", elapsed_seconds=time.perf_counter()-start)), flush=True)
            # All native writes; no method scoring here mutates ordinary caches.
            for state in states.values():
                state.release(target, None)
            if target < 5:
                early = replay_calibration(current, history, states, uids, cutover, CUTOVER_DAYS[target]*DAY,
                                           ledger, 128, args.calibration_batch)
            previous = current
        write_json(out / "summary.json", dict(status="diagnostic_complete", rows=aggregate(pd.DataFrame(all_rows)),
            ledger_seconds=dict(ledger), elapsed_seconds=time.perf_counter()-start,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/(1 << 20), confirmation_read=False,
            adaptation_targets=[1, 3, 4, 5], no_op_targets=[2],
            checks="disjoint fit/held/refit items; causal integer gaps; zero affine equals native; solve residual; exact value+key decomposition; finite outputs"))
    except Exception as exc:
        write_json(out / "summary.json", dict(status="failed", error=repr(exc), ledger_seconds=dict(ledger)))
        raise
    print(json.dumps(dict(status="diagnostic_complete", elapsed_seconds=time.perf_counter()-start)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fit-users", type=int, default=64)
    parser.add_argument("--diagnostic-users", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--estimate-seconds", type=int, required=True)
    main(parser.parse_args())
