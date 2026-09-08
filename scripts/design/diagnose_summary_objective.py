#!/usr/bin/env python3
"""A matched 2x2: mean/functional source x coefficient/response supervision.

Both objectives use the same source-only PCA32 affine family and ridge .01.
This is a mechanism experiment, not a bitwise reimplementation of scheme15's
response-dependent rank truncation. No hyperparameter selection or M2 target.
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

import pandas as pd
import torch
from design.data import DAY, ROOT, diagnostic_admissions, frozen_model, histories
from design.diagnose_query_holdout import (
    aggregate,
    batch_cache,
    panels,
    response_change,
    solve_affine,
    view_score,
)
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

from hstu_kvcache.adaptation.functional_summary import FunctionalSummary, source_producers
from hstu_kvcache.adaptation.reader import score


def source_projection(features, fit_mask, rank=32):
    fitting = features[fit_mask].double()
    center = fitting.mean(0)
    scale = fitting.std(0).clamp_min(1e-3)
    x = (fitting-center)/scale
    eigen, vectors = torch.linalg.eigh(x @ x.T)
    rank = min(rank, int((eigen > eigen[-1]*1e-8).sum()))
    retained = eigen[-rank:]
    projection = x.T @ vectors[:, -rank:] * ((len(x)-1)**.5 / retained)[None]
    latent = (features.double()-center)/scale @ projection
    latent = torch.cat((torch.ones_like(latent[:, :1]), latent), -1)
    return latent, dict(center=center.cpu(), scale=scale.cpu(), projection=projection.cpu()), dict(
        input_dimension=features.shape[1], rank=rank, fitting_scenes=len(x),
        source_variance_retained=float(retained.sum()/eigen.clamp_min(0).sum()))


def fit_shared(latent, query, wanted, objective, counts=None):
    """Same coefficients W[r,head,1+qdim,vdim], different observation geometry."""
    h, q, y = latent.double(), query.double(), wanted.double()
    x = torch.cat((torch.ones_like(q[..., :1]), q), -1)
    scenes, heads, queries, dimension = q.shape
    r, j = h.shape[-1], dimension+1
    ridge = .01 * scenes
    # Installed aggregate B_s = count_s * rate_B_s. The same count metric is
    # required on BOTH coefficient and response objectives for a clean factor.
    sample_weight = (torch.ones(scenes, device=h.device, dtype=h.dtype) if counts is None
                     else counts.double().square()/counts.double().square().mean())
    weighted_h = h * sample_weight[:, None]
    if objective == "coefficient":
        _, _, coefficients, _, _ = solve_affine(q, y, torch.zeros_like(q[:, :, :1]), torch.ones_like(q[:, :, :1]))
        gram = h.T @ weighted_h
        system = gram + ridge*torch.eye(r, device=h.device, dtype=h.dtype)
        rhs = weighted_h.T @ coefficients.flatten(1)
        solution = torch.linalg.solve(system, rhs)
        weights = solution.reshape(r, heads, j, dimension)
    else:
        # Sum over observations without materializing [scenes*queries, r*j].
        qgram = x.transpose(-2, -1) @ x / queries
        houter = (weighted_h[:, :, None]*h[:, None]).flatten(1)
        gram = (houter.T @ qgram.flatten(1)).reshape(r, r, heads, j, j)
        gram = gram.permute(2, 0, 3, 1, 4).reshape(heads, r*j, r*j)
        system = gram + ridge*torch.eye(r*j, device=h.device, dtype=h.dtype)
        qty = x.transpose(-2, -1) @ y / queries
        rhs = torch.einsum("sr,shjd->hrjd", weighted_h, qty).reshape(heads, r*j, dimension)
        solution = torch.linalg.solve(system, rhs)
        weights = solution.reshape(heads, r, j, dimension).transpose(0, 1)
    residual = system @ solution - rhs
    torch.testing.assert_close(system @ solution, rhs, atol=1e-7, rtol=1e-6)
    prediction = torch.einsum("sr,rhjd->shjd", h, weights)
    response_error = (x @ prediction-y).square().mean()
    return weights, dict(objective=objective, ridge=.01,
        count_metric="aggregate_response" if counts is not None else "per_event_rate",
        weighted_response_mse=float(((x @ prediction-y).square().mean((1, 2, 3))*sample_weight).mean()),
        fitting_response_mse=float(response_error), response_energy=float(y.square().mean()),
        normal_relative_residual=float(residual.abs().max()/rhs.abs().max().clamp_min(1e-20)))


def install_layer(latent, weights, center, scale, active):
    coefficients = torch.einsum("sr,rhjd->shjd", latent, weights)
    slopes = coefficients[:, :, 1:] / scale.transpose(-2, -1)
    intercept = coefficients[:, :, 0] - (center @ slopes).squeeze(-2)
    return (intercept*active[:, None, None]).float(), (slopes*active[:, None, None, None]).float()


@torch.no_grad()
def fixed_probes(model, states, history, fit_uids, rich=False):
    """Nested 4/32 M0 probe banks, fixed before M1 teacher access."""
    queries, traces = [], []
    for uid, position in zip(fit_uids[:3], (0, 21, 42), strict=True):
        state = states[uid]
        cutover = CUTOVER_DAYS[0]*DAY
        panel, _ = panels(history, uid, cutover, cutover-int(state.writer.events[-1][3]), state.cache.k.device)
        _, trace = score(model, state.cache, *panel["fit64"], trace=True)
        traces.append(trace)
        queries.append(torch.stack([q[0, :, position] for q in trace.queries]))
    if rich:
        original = {(0, 0), (1, 21), (2, 42)}
        for position in range(64):
            for user, trace in enumerate(traces):
                if (user, position) not in original and len(queries) < 31:
                    queries.append(torch.stack([q[0, :, position] for q in trace.queries]))
    return torch.stack((torch.zeros_like(queries[0]), *queries), 2)


@torch.no_grad()
def experiment(current, scenes, mapper, probes, history, cutover, fit_uids, batch_size, out, target, ledger, rich=False, aggregate_response=False):
    states = [training_state(scene, current, mapper) for scene in scenes]
    sources = [state.pack_source(target) for state in states]
    base = torch.stack([mapper.features(source) for source in sources])
    counts = torch.stack([s.count.sum() for s in sources])
    fit_mask = torch.tensor([s.uid in fit_uids for s in scenes], device=base.device)
    fit_indices = fit_mask.nonzero().flatten().tolist()
    extras, storage = [], []
    for state in states:
        sketch = FunctionalSummary(probes, current.blocks[0].attn.scale)
        timed(partial(sketch.update, state.cache.k[:, 0], state.cache.v[:, 0], source_producers(state)),
              ledger, "diagnostic_functional_backfill")
        # Zero probe must equal sum V; do not duplicate it in predictor inputs.
        torch.testing.assert_close(sketch.responses[:, :, :, 0].sum(0).flatten(1),
                                   state.cache.v[:, 0].float().sum(1), atol=.01, rtol=1e-5)
        extras.append(sketch.responses[:target+1, :, :, 1:].flatten()/state.cache.seq_len)
        storage.append(sketch.state_bytes)
    features = dict(mean=base, functional=torch.cat((base, torch.stack(extras)), 1))
    groups = defaultdict(list)
    raw_panels = []
    for i, state in enumerate(states):
        groups[state.cache.seq_len].append(i)
        raw_panels.append(panels(history, scenes[i].uid, cutover, float(scenes[i].query_delta.max()), base.device)[0])
    batches = [indices[offset:offset+batch_size] for indices in groups.values()
               for offset in range(0, len(indices), batch_size)]

    def query_batch(indices, name):
        return tuple(torch.cat([raw_panels[i][name][j] for i in indices]) for j in range(2))

    projections = {}
    for source_name, f in features.items():
        latent, parameters, record = source_projection(f, fit_mask, rank=128 if rich else 32)
        projections[source_name] = latent
        torch.save(parameters, out / f"source_projection_{source_name}_m{target}.pt")
        write_json(out / f"source_projection_{source_name}_m{target}.json", record)
    shape = (len(scenes), mapper.layers, mapper.heads, mapper.head_dim)
    rows, fit_records, response_rows = [], [], []
    exact_scores, reuse_scores = {}, {}
    for indices in batches:
        source = batch_cache([states[i].cache for i in indices])
        teacher = batch_cache([scenes[i].teacher for i in indices])
        for panel_name in ("fit64", "held64"):
            panel = query_batch(indices, panel_name)
            exact = score(current, teacher, *panel)[0]
            reuse = score(current, source, *panel)[0]
            for j, i in enumerate(indices):
                exact_scores[i, panel_name], reuse_scores[i, panel_name] = exact[j], reuse[j]
    for source_name, latent in projections.items():
        for objective in (("response",) if rich else ("coefficient", "response")):
            method = source_name + "_" + objective
            intercept, slopes = base.new_zeros(shape), base.new_zeros(*shape, mapper.head_dim)
            saved_layers = []
            for layer in range(mapper.layers):
                query = base.new_zeros(len(scenes), mapper.heads, 64, mapper.head_dim)
                wanted = torch.zeros_like(query)
                for indices in batches:
                    indices = [i for i in indices if bool(fit_mask[i])]
                    if not indices:
                        continue
                    source = batch_cache([states[i].cache for i in indices])
                    teacher = batch_cache([scenes[i].teacher for i in indices])
                    _, trace = view_score(current, source, query_batch(indices, "fit64"),
                                          intercept[indices], slopes[indices], counts[indices], True)
                    query[indices] = trace.queries[layer]
                    wanted[indices] = response_change(current, source, teacher, trace, layer, counts[indices])
                q, y = query[fit_indices].double(), wanted[fit_indices].double()
                center, scale = q.mean((0, 2), keepdim=True), q.std((0, 2), keepdim=True).clamp_min(1e-4)
                normalized = (q-center)/scale
                weights, record = timed(partial(fit_shared, latent[fit_mask], normalized, y, objective,
                                               counts[fit_indices] if aggregate_response else None),
                                        ledger, "shared_" + objective + "_fit")
                active = mapper.active_layers(base)[:, layer]
                intercept[:, layer], slopes[:, layer] = install_layer(latent, weights, center, scale, active)
                saved_layers.append(dict(weights=weights.cpu(), query_center=center.cpu(), query_scale=scale.cpu()))
                fit_records.append(dict(target=target, method=method, layer=layer, **record))
            torch.save(saved_layers, out / f"translator_{method}_m{target}.pt")
            for indices in batches:
                source = batch_cache([states[i].cache for i in indices])
                teacher = batch_cache([scenes[i].teacher for i in indices])
                for panel_name in ("fit64", "held64"):
                    prediction, trace = view_score(current, source, query_batch(indices, panel_name),
                        intercept[indices], slopes[indices], counts[indices], True)
                    assert torch.isfinite(prediction).all()
                    for layer in range(mapper.layers):
                        wanted = response_change(current, source, teacher, trace, layer, counts[indices])
                        predicted = trace.corrections[layer]/counts[indices, None, None, None]
                        error = (wanted-predicted).square().mean((1, 2, 3))
                        energy = wanted.square().mean((1, 2, 3))
                        for j, i in enumerate(indices):
                            response_rows.append(dict(target=target, method=method, panel=panel_name,
                                uid=scenes[i].uid, scene=i, layer=layer,
                                group="fitting_uid" if bool(fit_mask[i]) else "diagnostic_uid",
                                error=float(error[j]), energy=float(energy[j])))
                    for j, i in enumerate(indices):
                        exact = exact_scores[i, panel_name]
                        rows.append(dict(target=target, method=method, panel=panel_name, uid=scenes[i].uid, scene=i,
                            retained_events=states[i].cache.seq_len, release_writes=states[i].writes_since_release,
                            group="fitting_uid" if bool(fit_mask[i]) else "diagnostic_uid",
                            logit_mse=float((prediction[j]-exact).square().mean()),
                            probability_mse=float((prediction[j].sigmoid()-exact.sigmoid()).square().mean())))
            print(json.dumps(dict(target=target, method=method, status="complete")), flush=True)
    for (i, panel_name), reuse in reuse_scores.items():
        exact = exact_scores[i, panel_name]
        rows.append(dict(target=target, method="reuse", panel=panel_name, uid=scenes[i].uid, scene=i,
            group="fitting_uid" if bool(fit_mask[i]) else "diagnostic_uid",
            logit_mse=float((reuse-exact).square().mean()),
            probability_mse=float((reuse.sigmoid()-exact.sigmoid()).square().mean())))
    pd.DataFrame(rows).to_parquet(out / f"outputs_m{target}.parquet", index=False)
    pd.DataFrame(response_rows).to_parquet(out / f"responses_m{target}.parquet", index=False)
    write_json(out / f"fits_m{target}.json", fit_records)
    write_json(out / f"cost_m{target}.json", dict(functional_auxiliary_state_bytes=max(storage),
        probe_shared_bytes=probes.numel()*probes.element_size(),
        query_view_bytes_per_user=mapper.layers*mapper.heads*(mapper.head_dim+1)*mapper.head_dim*4,
        scope="diagnostic scans charged; incremental primitive checked separately; not a population/online ledger"))
    return rows


@torch.no_grad()
def main(cli):
    out = ROOT / "results/design" / cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "results/design/v15_query_view64_01/configuration.json").read_text())
    diagnostic_config = json.loads((ROOT / "results/design/pre_release_auc128_01/configuration.json").read_text())
    fit, diagnostic = config["calibration_uids"][:cli.fit_users], diagnostic_config["calibration_uids"][:cli.diagnostic_users]
    life_fit, life_diag = fit[:len(fit)//4], diagnostic[:len(diagnostic)//4]
    uids = life_fit + life_diag + fit[len(life_fit):] + diagnostic[len(life_diag):]
    split = json.loads((ROOT / config["split_path"]).read_text())
    assert not set(fit) & set(diagnostic)
    assert not set(uids) & (set(split["development"]) | set(split["confirmation"]))
    args = SimpleNamespace(**config)
    args.query_affine = False
    args.lifetime_users = len(life_fit) + len(life_diag)
    files = [Path(__file__), ROOT / "scripts/design/diagnose_query_holdout.py", ROOT / "scripts/design/run.py",
             ROOT / "scripts/design/data.py", *sorted((ROOT / "src/hstu_kvcache/adaptation").glob("*.py"))]
    write_json(out / "configuration.json", dict(vars(cli), fitting_uids=fit, diagnostic_uids=diagnostic,
        adaptation_targets=[1, 3, 4, 5], no_op_targets=[2], frozen_backbone_seed=17, confirmation_read=False,
        m5_scope="E14_partial", actual_operator="legacy/ELU+1", admissions=diagnostic_admissions(5),
        checkpoint_inputs=config["checkpoint_inputs"],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        design=("single richer-source diagnostic:32 nested fixed probes and common source PCA128; mean/response capacity control; ridge .01;64 pairs"
                if cli.rich_diagnostic else "producer means+fractions+age vs same plus3 nonzero functional probes; common source-only PCA32; ridge .01;64 pairs"),
        objectives="per-scene coefficient distance vs actual-query response distance; identical affine W family and .01 ||W||^2 penalty",
        count_metric=("both objectives use count^2 / mean_fit_count^2: actual aggregate coefficient/response, normalized by one fitting-set scalar"
                      if cli.aggregate_response else "both objectives use per-event rate, inherited historical count normalization"),
        source_scope="all producers including Current descendants; native ordinary trajectory unchanged",
        probes="4 or nested32 actual M0 fitting-user query bank including zero; fixed before M1 teacher access, never changed at later release",
        cost_scope="diagnostic source scans/teacher/fitting charged; writer incremental primitive only, not full service qualification",
        expected_seconds=cli.estimate_seconds, expected_gpu_mib=30000 if cli.rich_diagnostic else 22000))
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    start, ledger, rows = time.perf_counter(), defaultdict(float), []
    try:
        history = timed(partial(histories, uids, CUTOVER_DAYS[-1]+1), ledger, "history_io")
        previous = timed(partial(frozen_model, 0, device), ledger, "model_io")
        states, early = initialize_calibration(previous, history, uids, CUTOVER_DAYS[0]*DAY, args, ledger)
        probes = timed(partial(fixed_probes, previous, states, history, fit, cli.rich_diagnostic), ledger, "writer_probe_initialization")
        torch.save(probes.cpu(), out / "fixed_m0_probes.pt")
        for target in range(1, 6):
            current = timed(partial(frozen_model, target, device), ledger, "model_io")
            cutover = CUTOVER_DAYS[target-1]*DAY
            if target != 2:
                mapper_args = SimpleNamespace(**vars(args))
                mapper_args.query_affine = True
                mapper = new_translator(mapper_args, target, device)
                scenes = make_scenes(current, history, uids, states, early, cutover, ledger, previous, target, args)
                record = timed(partial(experiment, current, scenes, mapper, probes, history, cutover,
                                       set(fit), cli.batch_size, out, target, ledger, cli.rich_diagnostic,
                                       cli.aggregate_response), ledger, "factorial_total")
                rows.extend(record)
                del scenes
            for state in states.values():
                state.release(target, None)
            if target < 5:
                early = replay_calibration(current, history, states, uids, cutover, CUTOVER_DAYS[target]*DAY,
                                           ledger, 128, args.calibration_batch)
            previous = current
        write_json(out / "summary.json", dict(status="diagnostic_complete", rows=aggregate(pd.DataFrame(rows)),
            elapsed_seconds=time.perf_counter()-start, ledger_seconds=dict(ledger),
            ledger_note="factorial_total includes diagnostic_functional_backfill/shared_*_fit; do not double count nested entries",
            peak_allocated_mib=torch.cuda.max_memory_allocated()/(1 << 20), confirmation_read=False,
            adaptation_targets=[1, 3, 4, 5], no_op_targets=[2],
            checks="disjoint UID/items, pre-release source, frozen M0 probes, zero-probe sumV, exact normal equations, finite output"))
    except Exception as exc:
        write_json(out / "summary.json", dict(status="failed", error=repr(exc), ledger_seconds=dict(ledger)))
        raise
    print(json.dumps(dict(status="diagnostic_complete", elapsed_seconds=time.perf_counter()-start)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fit-users", type=int, default=64)
    parser.add_argument("--diagnostic-users", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--estimate-seconds", type=int, required=True)
    parser.add_argument("--rich-diagnostic", action="store_true",
                        help="Single predeclared fallback:32 probes/PCA128, two response arms; not a deployable candidate")
    parser.add_argument("--aggregate-response", action="store_true",
                        help="Use actual aggregate B and responses in both objectives; expert-equation match")
    main(parser.parse_args())
