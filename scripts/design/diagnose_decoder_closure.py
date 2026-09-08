#!/usr/bin/env python3
"""Frozen decoder capacity and ordered six-layer interventions; no population evaluation."""

import argparse
import hashlib
import json
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from design.data import DAY, ROOT, frozen_model, histories
from design.diagnose_query_holdout import batch_cache, panels
from design.diagnose_summary_objective import install_layer
from design.run import (
    initialize_calibration,
    make_scenes,
    new_translator,
    replay_calibration,
    training_state,
    write_json,
)
from insight_two.common import CUTOVER_DAYS

from hstu_kvcache.adaptation.functional_summary import FunctionalSummary, source_producers
from hstu_kvcache.adaptation.reader import history_read, score

SMALL = ROOT / "results/design/mechanism_aggregate_factorial192_01"
WIDE = ROOT / "results/design/mechanism_rich_source192_01"
REFERENCE = ROOT / "results/design/v15_query_view64_01"
METHODS = {"mean_coefficient32": (SMALL, "mean_coefficient"),
           "functional_response32": (SMALL, "functional_response"),
           "functional_response128": (WIDE, "functional_response")}


def free_code(query, wanted, layer):
    """One shared-across-heads latent per scene/layer; fixed intercept and decoder.

    Tall QR followed by a small SVD avoids squaring the condition number.
    No learned regularizer or target-dependent choice of rank threshold.
    """
    q = (query.double()-layer["query_center"])/layer["query_scale"]
    q = torch.cat((torch.ones_like(q[..., :1]), q), -1)
    weights = layer["weights"]
    design = torch.einsum("bhqj,rhjd->bhqdr", q, weights[1:]).flatten(1, 3)
    rhs = (wanted.double()-torch.einsum("bhqj,hjd->bhqd", q, weights[0])).flatten(1)
    orthogonal, triangular = torch.linalg.qr(design, mode="reduced")
    u, singular, vh = torch.linalg.svd(triangular, full_matrices=False)
    tolerance = max(design.shape[-2:])*torch.finfo(torch.float64).eps*singular[:, :1]
    keep = singular > tolerance
    projected = orthogonal.transpose(-2, -1) @ rhs[..., None]
    coefficients = (u.transpose(-2, -1) @ projected).squeeze(-1)
    coefficients = torch.where(keep, coefficients/singular.clamp_min(1e-300), 0)
    latent = (vh.transpose(-2, -1) @ coefficients[..., None]).squeeze(-1)
    residual = (design @ latent[..., None]).squeeze(-1)-rhs
    normal = design.transpose(-2, -1) @ residual[..., None]
    stats = dict(rank=keep.sum(-1), effective_rank=(singular > singular[:, :1]*1e-6).sum(-1),
                 condition=singular[:, 0]/singular[:, -1].clamp_min(1e-300),
                 solve_mse=residual.square().mean(-1),
                 relative_normal_residual=normal.squeeze(-1).norm(dim=-1)/
                 (design.norm(dim=(-2, -1))*rhs.norm(dim=-1)).clamp_min(1e-30),
                 latent_norm=latent.norm(dim=-1))
    return torch.cat((torch.ones_like(latent[:, :1]), latent), -1), stats


def spectrum(x):
    gram = x.double() @ x.double().T
    eigen = torch.linalg.eigvalsh(gram).clamp_min(0)
    probability = eigen/eigen.sum().clamp_min(1e-30)
    return dict(energy=float(eigen.sum()), entropy_rank=float((-(probability*probability.clamp_min(1e-30).log()).sum()).exp()),
                eigenvalues=eigen.cpu().tolist())


def load_views(states, scenes, mapper, target, fit_uids, out):
    sources = [state.pack_source(target) for state in states]
    base = torch.stack([mapper.features(source) for source in sources])
    counts = torch.stack([source.count.sum() for source in sources])
    active = mapper.active_layers(base)
    features, audit, views, layers_by_method = {}, {}, {}, {}
    fit = torch.tensor([s.uid in fit_uids for s in scenes], device=base.device)
    for folder in (SMALL, WIDE):
        probes = torch.load(folder / "fixed_m0_probes.pt", map_location=base.device, weights_only=True)
        extras = []
        for state in states:
            sketch = FunctionalSummary(probes, mapper.diagnostic_attention_scale)
            sketch.update(state.cache.k[:, 0], state.cache.v[:, 0], source_producers(state))
            extras.append(sketch.responses[:target+1, :, :, 1:].flatten()/state.cache.seq_len)
        features[folder] = dict(mean=base, functional=torch.cat((base, torch.stack(extras)), 1))
        bank = probes.permute(2, 0, 1, 3).flatten(1)
        audit[folder.name] = dict(probes_uncentered=spectrum(bank), probes_centered=spectrum(bank-bank.mean(0)))
    for method, (folder, old_name) in METHODS.items():
        source_name = old_name.split("_")[0]
        f = features[folder][source_name].double()
        projection = torch.load(folder / f"source_projection_{source_name}_m{target}.pt", map_location=base.device, weights_only=True)
        standardized = (f-projection["center"])/projection["scale"]
        latent = standardized @ projection["projection"]
        basis = torch.nn.functional.normalize(projection["projection"], dim=0)
        fitting = standardized[fit]
        fitting_latent = latent[fit]
        uid_values = [s.uid for s in scenes if s.uid in fit_uids]
        means = torch.stack([fitting_latent[torch.tensor([u == uid for u in uid_values], device=base.device)].mean(0)
                             for uid in sorted(set(uid_values))])
        uid_mean = {uid: means[j] for j, uid in enumerate(sorted(set(uid_values)))}
        within = torch.stack([value-uid_mean[uid] for value, uid in zip(fitting_latent, uid_values, strict=True)])
        between = torch.stack([uid_mean[uid] for uid in uid_values])-fitting_latent.mean(0)
        audit[method] = dict(rank=latent.shape[1], fitting_source_retained=float((fitting@basis).square().sum()/fitting.square().sum()),
            between_uid_energy_by_dimension=between.square().mean(0).cpu().tolist(),
            within_uid_energy_by_dimension=within.square().mean(0).cpu().tolist(),
            fitting_latent_uncentered=spectrum(fitting_latent),
            fitting_latent_centered=spectrum(fitting_latent-fitting_latent.mean(0)))
        latent = torch.cat((torch.ones_like(latent[:, :1]), latent), -1)
        layers = torch.load(folder / f"translator_{old_name}_m{target}.pt", map_location=base.device, weights_only=True)
        bs, slopes = [], []
        for l, layer in enumerate(layers):
            b, a = install_layer(latent, layer["weights"], layer["query_center"], layer["query_scale"], active[:, l])
            bs.append(b)
            slopes.append(a)
        views[method] = (torch.stack(bs, 1), torch.stack(slopes, 1))
        layers_by_method[method] = layers
    b, a = mapper.query_view(base)
    views["shared15"] = (b.reshape(len(states), 6, 6, 32), a)
    write_json(out / f"source_audit_m{target}.json", audit)
    return base, counts, active, views, layers_by_method


def run_target(current, scenes, mapper, history, cutover, fit_uids, uid_count, batch_size, out, target, canary):
    states = [training_state(scene, current, mapper) for scene in scenes]
    mapper.diagnostic_attention_scale = current.blocks[0].attn.scale
    base, counts, active, views, layers = load_views(states, scenes, mapper, target, fit_uids, out)
    groups, panel_list = defaultdict(list), []
    metadata = []
    stride = 2 if target == 1 else 3
    for i, state in enumerate(states):
        groups[state.cache.seq_len].append(i)
        panel_list.append(panels(history, scenes[i].uid, cutover, float(scenes[i].query_delta.max()), base.device)[0])
        metadata.append(dict(target=target, scene=i, uid=scenes[i].uid,
            group="fitting_uid" if scenes[i].uid in fit_uids else "diagnostic_uid",
            state_group="continuous_cutover" if i < uid_count*stride and i % stride == 0 else "auxiliary",
            count=float(counts[i]), release_age=float(states[i].pack_source(target).release_age)))
    pd.DataFrame(metadata).to_parquet(out / f"scenes_m{target}.parquet", index=False)
    rows, local_rows, solver_rows, cross_rows = [], [], [], []
    batches = [indices[j:j+batch_size] for indices in groups.values() for j in range(0, len(indices), batch_size)]
    for batch_id, indices in enumerate(batches):
        source, teacher = batch_cache([states[i].cache for i in indices]), batch_cache([scenes[i].teacher for i in indices])
        n = counts[indices]
        panel = {name: tuple(torch.cat([panel_list[i][name][j] for i in indices]) for j in range(2)) for name in ("fit64", "held64")}
        tensors = dict(scene_indices=indices)
        exact = {name: score(current, teacher, *p, trace=True) for name, p in panel.items()}
        tensors["exact"] = {name: z.cpu() for name, (z, _) in exact.items()}

        def record(method, name, prediction, path="actual", exact=exact, indices=indices):
            error = (prediction.double()-exact[name][0].double()).square().mean(-1)
            for j, i in enumerate(indices):
                rows.append(dict(**metadata[i], method=method, panel=name, path=path, logit_mse=float(error[j])))

        def read(b, a, name, mask=None, source=source, teacher=teacher, panel=panel, n=n):
            def replace(layer_index, actual_query, heads):
                if mask[layer_index]:
                    return heads
                return history_read(current.blocks[layer_index].attn, actual_query, teacher.k[layer_index], teacher.v[layer_index])
            return score(current, source, *panel[name], trace=True,
                         response_delta=b.flatten(2)*n[:, None, None],
                         response_query_delta=a*n[:, None, None, None, None],
                         history_override=None if mask is None else replace)

        for name, p in panel.items():
            reuse = score(current, source, *p)[0]
            record("reuse", name, reuse)
            tensors["reuse_"+name] = reuse.cpu()
        common_traces = {}
        for method, (all_b, all_a) in views.items():
            b, a = all_b[indices], all_a[indices]
            method_raw = {}
            for name in ("fit64", "held64"):
                shared, shared_trace = read(b, a, name)
                record(method, name, shared)
                epsilon = torch.stack([shared_trace.corrections[l]-(history_read(current.blocks[l].attn,
                    q, teacher.k[l], teacher.v[l])-shared_trace.history_heads[l])
                    for l, q in enumerate(shared_trace.queries)], 1)
                method_raw[name] = dict(shared=shared.cpu(), epsilon=epsilon.cpu())
                if name == "held64":
                    common_traces[method] = shared_trace
                prefix = [exact[name][0]]
                prefix_changes, single_changes, single_epsilon, single_logits = [], [], [], []
                for k in range(7):
                    prediction, trace = read(b, a, name, [l < k for l in range(6)])
                    if k == 0:
                        torch.testing.assert_close(prediction, exact[name][0], atol=2e-5, rtol=1e-5)
                    else:
                        prefix.append(prediction)
                    if k == 6:
                        torch.testing.assert_close(prediction, shared, atol=0, rtol=0)
                    record(method, name, prediction, f"prefix{k}")
                    # Per query, layer: query/hidden changes relative to exact-context branch.
                    prefix_changes.append(torch.stack([torch.stack(((q-exact[name][1].queries[l]).square().mean((1, 3)),
                        (trace.layer_outputs[l]-exact[name][1].layer_outputs[l]).square().mean(-1)), -1)
                        for l, q in enumerate(trace.queries)], 1))
                for k in range(6):
                    prediction, trace = read(b, a, name, [l == k for l in range(6)])
                    record(method, name, prediction, f"single{k+1}")
                    q = trace.queries[k]
                    eps = trace.corrections[k]-(history_read(current.blocks[k].attn, q, teacher.k[k], teacher.v[k])-trace.history_heads[k])
                    single_epsilon.append(eps)
                    single_logits.append(prediction)
                    single_changes.append(torch.stack([torch.stack(((query-exact[name][1].queries[l]).square().mean((1, 3)),
                        (trace.layer_outputs[l]-exact[name][1].layer_outputs[l]).square().mean(-1)), -1)
                        for l, query in enumerate(trace.queries)], 1))
                d = torch.stack(prefix, 1).double().diff(dim=1)
                torch.testing.assert_close(d.sum(1), shared.double()-exact[name][0].double(), atol=1e-12, rtol=1e-12)
                method_raw[name].update(d=d.cpu(), prefix=torch.stack(prefix, 1).cpu(),
                    single=torch.stack(single_logits, 1).cpu(), single_epsilon=torch.stack(single_epsilon, 1).cpu(),
                    prefix_changes=torch.stack(prefix_changes, 1).cpu(), single_changes=torch.stack(single_changes, 1).cpu())
                for context, eps in (("shared", epsilon), ("single", torch.stack(single_epsilon, 1))):
                    error = eps.double().square().mean((2, 3, 4))
                    for j, i in enumerate(indices):
                        for l in range(6):
                            local_rows.append(dict(**metadata[i], method=method, panel=name, context=context,
                                                   layer=l, response_mse=float(error[j, l])))
            tensors[method] = method_raw
        # Same-q cross evaluation of the two principal functions on three contexts.
        for context, trace in {"exact": exact["held64"][1], **{m: common_traces[m] for m in
                ("mean_coefficient32", "functional_response32")}}.items():
            for method in ("mean_coefficient32", "functional_response32"):
                b, a = (v[indices] for v in views[method])
                epsilons = []
                for l, q in enumerate(trace.queries):
                    wanted = history_read(current.blocks[l].attn, q, teacher.k[l], teacher.v[l])-history_read(current.blocks[l].attn, q, source.k[l], source.v[l])
                    eps = (b[:, l, :, None]+q@a[:, l])*n[:, None, None, None]-wanted
                    epsilons.append(eps.cpu())
                    error = eps.double().square().mean((1, 2, 3))
                    for j, i in enumerate(indices):
                        cross_rows.append(dict(**metadata[i], method=method, context=context, layer=l, response_mse=float(error[j])))
                tensors[f"cross_{context}_{method}"] = torch.stack(epsilons, 1)
        for method in ("functional_response32", "functional_response128"):
            b, a = (torch.zeros_like(v[indices]) for v in views[method])
            latent_values = []
            for l, layer in enumerate(layers[method]):
                _, trace = read(b, a, "fit64")
                query = trace.queries[l]
                wanted = (history_read(current.blocks[l].attn, query, teacher.k[l], teacher.v[l])-trace.history_heads[l])/n[:, None, None, None]
                latent, stats = free_code(query, wanted, layer)
                b[:, l], a[:, l] = install_layer(latent, layer["weights"], layer["query_center"], layer["query_scale"], active[indices, l])
                latent_values.append(latent.cpu())
                if canary:
                    # Independent CPU SVD least squares for the first active scene.
                    q = (query[:1].double()-layer["query_center"])/layer["query_scale"]
                    q = torch.cat((torch.ones_like(q[..., :1]), q), -1)
                    design = torch.einsum("bhqj,rhjd->bhqdr", q, layer["weights"][1:]).flatten(1, 3)[0].cpu()
                    rhs = (wanted[:1].double()-torch.einsum("bhqj,hjd->bhqd", q, layer["weights"][0])).flatten(1)[0].cpu()
                    reference = torch.linalg.lstsq(design, rhs, driver="gelsd").solution
                    torch.testing.assert_close(design@latent[0, 1:].cpu(), design@reference, atol=1e-7, rtol=1e-6)
                for j, i in enumerate(indices):
                    solver_rows.append(dict(**metadata[i], method=method, layer=l, active=bool(active[i, l]),
                                             **{key: float(value[j]) for key, value in stats.items()}))
            free_raw = dict(latent=torch.stack(latent_values, 1))
            for name in ("fit64", "held64"):
                prediction, trace = read(b, a, name)
                record(method, name, prediction, "free64")
                eps = torch.stack([trace.corrections[l]-(history_read(current.blocks[l].attn, q, teacher.k[l], teacher.v[l])-trace.history_heads[l])
                                   for l, q in enumerate(trace.queries)], 1)
                free_raw[name] = dict(logits=prediction.cpu(), epsilon=eps.cpu())
                error = eps.double().square().mean((2, 3, 4))
                for j, i in enumerate(indices):
                    for l in range(6):
                        local_rows.append(dict(**metadata[i], method=method, panel=name, context="free64", layer=l,
                                               response_mse=float(error[j, l])))
            tensors[method+"_free64"] = free_raw
        torch.save(tensors, out / f"raw_m{target}_batch{batch_id:03d}.pt")
    pd.DataFrame(rows).to_parquet(out / f"outputs_m{target}.parquet", index=False)
    pd.DataFrame(local_rows).to_parquet(out / f"responses_m{target}.parquet", index=False)
    pd.DataFrame(cross_rows).to_parquet(out / f"common_query_m{target}.parquet", index=False)
    pd.DataFrame(solver_rows).to_parquet(out / f"solvers_m{target}.parquet", index=False)
    print(json.dumps(dict(target=target, status="complete", scenes=len(scenes))), flush=True)


@torch.no_grad()
def main(cli, experiment=run_target):
    out = ROOT / "results/design" / cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((REFERENCE / "configuration.json").read_text())
    source_config = json.loads((SMALL / "configuration.json").read_text())
    fit, diagnostic = source_config["fitting_uids"][:cli.fit_users], source_config["diagnostic_uids"][:cli.diagnostic_users]
    cohort = None
    if getattr(cli, "cohort_path", None):
        cohort = json.loads(Path(cli.cohort_path).read_text())
        fit = cohort["original_fitting_uids"]+cohort["additional_fitting_uids"]
        diagnostic = cohort["diagnostic_uids"]
        life_fit = cohort["lifetime_fitting_uids"]
    else:
        life_fit = fit[:len(fit)//4]
    life_diag = diagnostic[:len(diagnostic)//4]
    uids = life_fit+life_diag+[u for u in fit if u not in life_fit]+diagnostic[len(life_diag):]
    args = SimpleNamespace(**config)
    args.query_affine, args.lifetime_users = False, len(life_fit)+len(life_diag)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    files = list(dict.fromkeys([Path(__file__), Path(experiment.__code__.co_filename).resolve(),
        ROOT / "src/hstu_kvcache/adaptation/reader.py", ROOT / "docs/design/expert_route_2026-09-07.md"]))
    if cohort is not None:
        files += [ROOT / "scripts/design/diagnose_native_input.py", Path(cli.cohort_path).resolve()]
        files += [ROOT / "scripts/design" / name for name in
                  ("run.py", "data.py", "diagnose_query_holdout.py", "diagnose_summary_objective.py")]
    weights = sorted(set(p for folder, _ in METHODS.values() for p in folder.glob("*.pt")))
    if cohort is not None:
        weights += sorted((ROOT / "results/design/mechanism_native_input192_01").glob("translator_native_input_m*.pt"))
        weights += sorted((ROOT / "results/design/native_coverage384_01").glob("translator_C_m*.pt"))
    write_json(out / "configuration.json", dict(vars(cli), fitting_uids=fit, diagnostic_uids=diagnostic,
        frozen_reference_methods={key: [str(p.relative_to(ROOT)), name] for key, (p, name) in METHODS.items()},
        actual_experiment=experiment.__name__, experiment_description=experiment.__doc__, cohort=cohort,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files+weights},
        confirmation_read=False, adaptation_targets=[1, 3, 4, 5], no_op_targets=[2], m5_scope="E14_partial",
        frozen_decoder_definition=("frozen native coordinates/source PCA32; experiment.json specifies fitting versus frozen read audit" if cohort is not None else
                                   "functional_response32/128, per scene and layer, intercept/count/mask fixed"),
        evidence="per-scene outputs and experiment-specific retained raw; diagnostic teacher access is distinct from executable predictors",
        expected_seconds=cli.estimate_seconds))
    write_json(out / "experiment.json", dict(entry_point=experiment.__name__, description=experiment.__doc__))
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for p in files:
            archive.add(p, arcname=str(p.relative_to(ROOT)))
    start, ledger = time.perf_counter(), defaultdict(float)
    try:
        history = histories(uids, CUTOVER_DAYS[-1]+1)
        previous = frozen_model(0, device)
        states, early = initialize_calibration(previous, history, uids, CUTOVER_DAYS[0]*DAY, args, ledger)
        for target in range(1, 6):
            current = frozen_model(target, device)
            cutover = CUTOVER_DAYS[target-1]*DAY
            if target != 2:
                mapper_args = SimpleNamespace(**vars(args))
                mapper_args.query_affine = True
                mapper = new_translator(mapper_args, target, device)
                saved = torch.load(REFERENCE / f"translator_v{target}.pt", map_location=device, weights_only=False)
                mapper.load_state_dict(saved["state_dict"])
                mapper.supported_producers = set(saved["supported_producers"])
                if cohort is None:
                    scenes = make_scenes(current, history, uids, states, early, cutover, ledger, previous, target, args)
                else:
                    scenes, preparation = [], {}
                    for label, members in (("original_fitting", cohort["original_fitting_uids"]),
                                           ("additional_fitting", cohort["additional_fitting_uids"]),
                                           ("diagnostic", diagnostic)):
                        ordered = [u for u in uids if u in members]
                        group_args = SimpleNamespace(**vars(args))
                        group_args.lifetime_users = sum(u in life_fit+life_diag for u in ordered)
                        group_ledger = defaultdict(float)
                        scenes.extend(make_scenes(current, history, ordered, states, early, cutover,
                                                  group_ledger, previous, target, group_args))
                        preparation[label] = dict(group_ledger)
                        for key, value in group_ledger.items():
                            ledger[key] += value
                    write_json(out / f"preparation_m{target}.json", preparation)
                experiment(current, scenes, mapper, history, cutover, set(fit), len(uids), cli.batch_size, out, target, cli.canary)
                del scenes, mapper
            for state in states.values():
                state.release(target, None)
            if target < 5:
                early = replay_calibration(current, history, states, uids, cutover, CUTOVER_DAYS[target]*DAY, ledger, 128, args.calibration_batch)
            previous = current
        write_json(out / "summary.json", dict(status="complete", elapsed_seconds=time.perf_counter()-start,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/(1 << 20), confirmation_read=False,
            state_construction_ledger_seconds=dict(ledger)))
    except Exception as exc:
        write_json(out / "summary.json", dict(status="failed", error=repr(exc), elapsed_seconds=time.perf_counter()-start))
        raise
    print((out / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fit-users", type=int, default=64)
    parser.add_argument("--diagnostic-users", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--estimate-seconds", type=int, required=True)
    main(parser.parse_args())
