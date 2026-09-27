#!/usr/bin/env python3
"""Small fixed-state, user-disjoint Insight diagnostic on the current 6L chain."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from design.competitor_data import history_arrays, make_candidate_panel
from design.competitor_models import load_model_pair, sha256_file
from design.shared_read_probe import SummaryProjection, fit_layer, make_history_override, self_check
from design import user_information_probe
from evaluate_yambda500m_foundation_raw import load_histories, load_model
from insight_two.low_rank_correction import low_rank_layered_correction
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.models import HSTUKVCache


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def cache_slice(cache, start, stop):
    return HSTUKVCache(cache.k[:, start:stop], cache.v[:, start:stop], cache.seq_len)


def collect_cache(model, arrays, users, batch_size, device):
    pieces = []
    for start in range(0, users, batch_size):
        events = tuple(torch.as_tensor(a[start:min(start + batch_size, users)], device=device)
                       for a in arrays[1:4])
        pieces.append(model.compute_kv(*events))
    return HSTUKVCache(torch.cat([p.k for p in pieces], 1),
                       torch.cat([p.v for p in pieces], 1), pieces[0].seq_len)


def device_parameters(parameters, device):
    return [{name: value.to(device=device, dtype=torch.float32) for name, value in p.items()}
            for p in parameters]


def fit_rule(model, source, teacher, panel, deltas, latent, batch_size, device,
             *, information_probe=False, use_user_response=False):
    users, length = source.k.shape[1], source.seq_len
    parameters, records = [], []
    acquisition_seconds = solve_seconds = 0.0
    latent = latent.to(device=device, dtype=torch.float32)
    counts = torch.full((users,), float(length), device=device)
    for layer, block in enumerate(model.blocks):
        query, wanted, observed = [], [], []
        started = clock(device)
        installed = device_parameters(parameters, device)
        for start in range(0, users, batch_size):
            stop = min(start + batch_size, users)
            src, dst = cache_slice(source, start, stop), cache_slice(teacher, start, stop)
            factory = user_information_probe.make_history_override if information_probe else make_history_override
            override = factory(installed, latent[start:stop], counts[start:stop])
            _, trace = score(model, src, panel[start:stop], deltas[start:stop],
                             trace=True, history_override=override)
            q, native = trace.queries[layer], trace.history_heads[layer]
            target = history_read(block.attn, q, dst.k[layer], dst.v[layer])
            query.append(q.cpu())
            wanted.append(((target - native) / length).cpu())
            observed.append((native.transpose(1, 2).flatten(2) / length).cpu())
        acquisition_seconds += clock(device) - started
        started = time.perf_counter()
        if information_probe:
            p, record = user_information_probe.fit_layer(
                latent.cpu().double(), torch.cat(observed), torch.cat(wanted), use_user_response=use_user_response)
        else:
            p, record = fit_layer(latent.cpu().double(), torch.cat(query), torch.cat(wanted),
                                  torch.cat(observed), counts.cpu())
        solve_seconds += time.perf_counter() - started
        parameters.append(p)
        records.append(dict(layer=layer, **record))
        print(f"  fitted layer {layer + 1}/{len(model.blocks)}", flush=True)
    return parameters, dict(layers=records, acquisition_seconds=acquisition_seconds,
                            solve_seconds=solve_seconds)


def reader_canary(model, source, teacher, panel, deltas):
    src, dst = cache_slice(source, 0, 2), cache_slice(teacher, 0, 2)
    candidates, delta = panel[:2, :4], deltas[:2]
    native, _ = score(model, src, candidates, delta)
    reference = model.score_cc_reuse(src, candidates, delta)
    torch.testing.assert_close(native, reference, atol=2e-5, rtol=2e-5)
    exact = model.score_cc_reuse(dst, candidates, delta)
    def replace(layer, q, response):
        return history_read(model.blocks[layer].attn, q, dst.k[layer], dst.v[layer])
    reconstructed, _ = score(model, src, candidates, delta, history_override=replace)
    torch.testing.assert_close(reconstructed, exact, atol=2e-5, rtol=2e-5)
    return dict(native_max_abs=float((native-reference).abs().max()),
                exact_replacement_max_abs=float((reconstructed-exact).abs().max()), passed=True)


def evaluate(model, source, teacher, panel, deltas, latent, rules, batch_size, config, device):
    users, length = source.k.shape[1], source.seq_len
    counts = torch.full((users,), float(length), device=device)
    latent = latent.to(device=device, dtype=torch.float32)
    rules = {name: device_parameters(p, device) for name, p in rules.items()}
    pieces = {name: [] for name in ("reuse", "exact", "per_user_rank1", *rules)}
    timings = {name: 0.0 for name in pieces}
    local = {name: [] for name in rules}
    for start in range(0, users, batch_size):
        stop = min(start + batch_size, users)
        src, dst = cache_slice(source, start, stop), cache_slice(teacher, start, stop)
        full_panel, delta = panel[start:stop], deltas[start:stop]
        anchors = full_panel[:, config["oracle_anchor_indices"]]
        held = full_panel[:, config["evaluation_candidate_indices"]]
        for name, cache in (("reuse", src), ("exact", dst)):
            started = clock(device)
            z, _ = score(model, cache, held, delta)
            timings[name] += clock(device) - started
            pieces[name].append(z.cpu().numpy())
        started = clock(device)
        oracle = low_rank_layered_correction(model, dst, src, anchors, held, delta,
                                             stage="av_aggregation", rank=1)
        timings["per_user_rank1"] += clock(device) - started
        pieces["per_user_rank1"].append(oracle.scores.cpu().numpy())
        for name, parameters in rules.items():
            information_probe = config.get("probe_kind") == "user_information"
            if information_probe:
                h = latent[start:stop][:, config["evaluation_candidate_indices"]]
            else:
                h = latent[start:stop] if name == "shared_summary" else torch.ones_like(latent[start:stop, :1])
            started = clock(device)
            factory = user_information_probe.make_history_override if information_probe else make_history_override
            override = factory(parameters, h, counts[start:stop])
            timings[name + "_view"] = timings.get(name + "_view", 0.0) + clock(device) - started
            started = clock(device)
            z, trace = score(model, src, held, delta, trace=True, history_override=override)
            timings[name] += clock(device) - started
            pieces[name].append(z.cpu().numpy())
            # Evaluation-only same-query teacher; never enters the installed predictor.
            residual, signal = [], []
            for layer, q in enumerate(trace.queries):
                native = trace.history_heads[layer]
                target = history_read(model.blocks[layer].attn, q, dst.k[layer], dst.v[layer])
                corrected = override(layer, q, native)
                residual.append((corrected-target).double().square().mean((1, 2, 3)))
                signal.append((native-target).double().square().mean((1, 2, 3)))
            local[name].append(torch.stack((torch.stack(residual).sum(0), torch.stack(signal).sum(0)), 1).cpu().numpy())
        print(f"  evaluated {stop}/{users} users", flush=True)
    return ({name: np.concatenate(values) for name, values in pieces.items()},
            {name: np.concatenate(values) for name, values in local.items()}, timings)


def metrics(scores, uids, edge):
    probabilities = {name: torch.from_numpy(z).double().sigmoid().numpy() for name, z in scores.items()}
    error = {name: np.abs(p-probabilities["exact"]).mean(1) for name, p in probabilities.items()}
    denominator = error["reuse"]
    if not np.isfinite(denominator).all() or np.any(denominator <= 1e-12):
        raise RuntimeError("near-zero/nonfinite per-user Reuse gap; do not silently change the primary metric")
    rows, per_user = [], []
    for name in scores:
        recovery = 1 - error[name] / denominator
        if not np.isfinite(recovery).all():
            raise RuntimeError("nonfinite recovery")
        rows.append(dict(edge=edge, method=name, users=len(uids),
                         mean_user_recovery=float(recovery.mean()), median_user_recovery=float(np.median(recovery)),
                         p10_user_recovery=float(np.quantile(recovery, .1)),
                         improved_user_fraction=float((recovery > 0).mean()),
                         probability_mae=float(error[name].mean()),
                         pooled_recovery=float(1-error[name].mean()/denominator.mean()),
                         minimum_reuse_user_mae=float(denominator.min())))
        per_user.extend(dict(edge=edge, uid=uid, method=name, probability_mae=float(err),
                             reuse_probability_mae=float(base), recovery=float(rec))
                        for uid, err, base, rec in zip(uids, error[name], denominator, recovery, strict=True))
    return rows, per_user


@torch.inference_mode()
def run(args):
    config = json.loads(args.config.read_text())
    information_probe = config.get("probe_kind") == "user_information"
    if args.output.exists():
        raise FileExistsError(args.output)
    assert config["scale"] == "medium" and config["ridge"] == .01
    assert not set(config["calibration_uids"]) & set(config["evaluation_uids"])
    source_split = ROOT / config["source_split"]["path"]
    assert sha256_file(source_split) == config["source_split"]["sha256"]
    assert sha256_file(ROOT / config["model_chain"]) == config["model_chain_sha256"]
    torch.set_num_threads(args.threads)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    args.output.mkdir(parents=True)
    files = [Path(__file__), ROOT / "scripts/design/shared_read_probe.py",
             ROOT / "scripts/design/user_information_probe.py",
             ROOT / "scripts/design/diagnose_native_input.py", ROOT / "scripts/design/competitor_data.py",
             ROOT / "scripts/design/diagnose_summary_objective.py",
             ROOT / "scripts/insight/candidate_shared_causal.py",
             ROOT / "scripts/insight/reader_compatibility_correction.py",
             ROOT / "scripts/design/competitor_models.py", ROOT / "scripts/insight_two/low_rank_correction.py",
             ROOT / "src/hstu_kvcache/adaptation/reader.py", ROOT / "src/hstu_kvcache/models/attention.py"]
    settings = dict(config=config, config_sha256=sha256_file(args.config), mode="canary" if args.canary else "diagnostic",
                    device=str(device), threads=args.threads, batch_size=args.batch_size,
                    torch_version=torch.__version__, backend=os.environ.get("EVOKV_ATTENTION_BACKEND"),
                    source_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in files})
    write_json(args.output / "configuration.json", settings)
    numerical_check = (user_information_probe.self_check() if information_probe else self_check()) if args.canary else None
    edges = config["edges"][:1] if args.canary else config["edges"]
    nc = config["canary"]["calibration_users"] if args.canary else len(config["calibration_uids"])
    ne = config["canary"]["evaluation_users"] if args.canary else len(config["evaluation_uids"])
    fit_uids, eval_uids = config["calibration_uids"][:nc], config["evaluation_uids"][:ne]
    pairs = [load_model_pair("medium", edge, verify_hashes=True) for edge in edges]
    assert all(p["admission"]["reuse_eligible"] for p in pairs)
    data = pairs[0]["dataset"]
    started = time.perf_counter()
    # Full declared groups define banks even in the small correctness canary.
    all_uids = config["calibration_uids"] + config["evaluation_uids"]
    history = load_histories(all_uids, dataset_path=Path(data["manifest"]), known_vocab_size=data["known_items"],
                             oov_buckets=data["oov_buckets"], start_timestamp=min(p["cutover"] for p in pairs)-1,
                             end_timestamp=max(p["cutover"] for p in pairs)+1, max_history=config["history_length"],
                             threads=args.threads)
    history_seconds = time.perf_counter() - started
    print(f"Loaded causal histories for {len(all_uids)} declared users in {history_seconds:.2f}s", flush=True)
    summaries, all_rows, all_users = [], [], []
    for pair in pairs:
        edge_started = clock(device)
        print(f"Starting {pair['edge']}, fit={nc}, evaluation={ne}", flush=True)
        arrays, panels = {}, {}
        for name, uids in (("fit", config["calibration_uids"]), ("eval", config["evaluation_uids"])):
            arrays[name] = history_arrays(history, np.asarray(uids), pair["cutover"], config["history_length"])
            panels[name], _ = make_candidate_panel(arrays[name][1], data["known_items"])
        models = []
        for role in ("parent", "current"):
            model, payload = load_model(Path(pair[role]["checkpoint"]), device)
            assert payload["config"] == pair["config"]
            model.requires_grad_(False)
            models.append(model)
            del payload
        parent, current = models
        caches, costs = {}, {}
        for group, users in (("fit", nc), ("eval", ne)):
            for role, model in (("source", parent), ("teacher", current)):
                started = clock(device)
                caches[group, role] = collect_cache(model, arrays[group], users, args.batch_size, device)
                costs[f"{group}_{role}_prefill_seconds"] = clock(device) - started
        fit_panel = torch.as_tensor(panels["fit"][:nc, config["calibration_candidate_indices"]], device=device)
        eval_panel = torch.as_tensor(panels["eval"][:ne], device=device)
        fit_delta = torch.as_tensor(arrays["fit"][4][:nc], device=device)
        eval_delta = torch.as_tensor(arrays["eval"][4][:ne], device=device)
        check = reader_canary(current, caches["eval", "source"], caches["eval", "teacher"], eval_panel, eval_delta)
        started = clock(device)
        if information_probe:
            fit_latent = current.lookup_item_embeddings(fit_panel).cpu()
            eval_latent = current.lookup_item_embeddings(eval_panel).cpu()
            costs["candidate_feature_lookup_seconds"] = clock(device) - started
            input_metadata = dict(candidate_features="frozen Current item embedding; same item has same features across users",
                                  user_features="actual native response of this user's cache at the branch's query",
                                  no_summary=True)
            arms = (("shared_candidate", fit_latent), ("shared_user_response", fit_latent))
        else:
            projection = SummaryProjection.fit(caches["fit", "source"], rank=config["projection_rank"])
            fit_latent = projection.encode(caches["fit", "source"])
            eval_latent = projection.encode(caches["eval", "source"])
            costs["summary_projection_and_encoding_seconds"] = clock(device) - started
            input_metadata = projection.metadata
            arms = (("shared_no_summary", torch.ones_like(fit_latent[:, :1])), ("shared_summary", fit_latent))
        rules, fitting = {}, {}
        for name, h in arms:
            print(f"Fitting {name}", flush=True)
            rules[name], fitting[name] = fit_rule(current, caches["fit", "source"], caches["fit", "teacher"],
                                                fit_panel, fit_delta, h, args.batch_size, device,
                                                information_probe=information_probe,
                                                use_user_response=name == "shared_user_response")
        scores, local, scoring_time = evaluate(current, caches["eval", "source"], caches["eval", "teacher"],
                                               eval_panel, eval_delta, eval_latent, rules, args.batch_size, config, device)
        np.savez_compressed(args.output / f"{pair['edge']}.npz", uids=np.asarray(eval_uids),
                            fit_uids=np.asarray(fit_uids), fit_candidates=fit_panel.cpu().numpy(),
                            evaluation_panel=eval_panel.cpu().numpy(), evaluation_deltas=eval_delta.cpu().numpy(),
                            **scores, **{f"response_{k}": v for k, v in local.items()})
        rows, per_user = metrics(scores, eval_uids, pair["edge"])
        all_rows.extend(rows)
        all_users.extend(per_user)
        summary = dict(edge=pair["edge"], model_pair=pair, calibration_users=nc, evaluation_users=ne,
                       reader_check=check, inputs=input_metadata, fitting=fitting,
                       costs=costs, score_path_seconds=scoring_time, metrics=rows,
                       elapsed_seconds=clock(device)-edge_started)
        summaries.append(summary)
        print(json.dumps(dict(edge=pair["edge"], metrics=rows, elapsed_seconds=summary["elapsed_seconds"])), flush=True)
        write_json(args.output / "summary.json", dict(status="in_progress", edges=summaries))
        del parent, current, models, caches, rules
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = dict(status="completed", mode=settings["mode"], history_seconds=history_seconds,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  edges=summaries, interpretation="Fixed-state development diagnostic, one backbone seed; no AUC or lifecycle claim.")
    write_json(args.output / "summary.json", result)
    for name, rows in (("metrics.csv", all_rows), ("per_user.csv", all_users)):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.canary:
        write_json(args.output / "canary.pass.json", dict(passed=True, reader_checks=[s["reader_check"] for s in summaries],
                   numerical_check=numerical_check,
                   estimated_full_seconds=history_seconds + 2 * summaries[0]["elapsed_seconds"] * max(256/nc, 128/ne),
                   estimate="Conservative linear scaling of complete small run by calibration/evaluation population size."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/user_information_6l_20260920.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
