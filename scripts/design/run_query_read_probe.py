#!/usr/bin/env python3
"""UID-disjoint q versus q+r diagnostic with bounded evaluation-cache memory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from design.competitor_data import history_arrays, make_candidate_panel
from design.competitor_models import load_model_pair, sha256_file
from design.query_read_probe import fit_layer, make_history_override, self_check
from design.run_shared_read_probe import (
    cache_slice, clock, collect_cache, device_parameters, metrics, reader_canary, write_json,
)
from evaluate_yambda500m_foundation_raw import load_histories, load_model
from hstu_kvcache.adaptation.reader import history_read, score

METHODS = ("shared_query", "shared_query_response")


def fit_rule(model, source, teacher, panel, deltas, use_response, batch_size, device):
    """Each layer sees this arm's corrected lower layers and a matched teacher q."""
    users, length = source.k.shape[1], source.seq_len
    counts = torch.full((users,), float(length), device=device)
    parameters, records = [], []
    acquisition_seconds = solve_seconds = 0.0
    for layer, block in enumerate(model.blocks):
        queries, targets, observations = [], [], []
        installed = device_parameters(parameters, device)
        started = clock(device)
        for start in range(0, users, batch_size):
            stop = min(start + batch_size, users)
            src, dst = cache_slice(source, start, stop), cache_slice(teacher, start, stop)
            override = make_history_override(installed, counts[start:stop])
            _, trace = score(model, src, panel[start:stop], deltas[start:stop],
                             trace=True, history_override=override)
            q, native = trace.queries[layer], trace.history_heads[layer]
            target = history_read(block.attn, q, dst.k[layer], dst.v[layer])
            queries.append(q.cpu())
            targets.append(((target - native) / length).cpu())
            observations.append((native.transpose(1, 2).flatten(2) / length).cpu())
        acquisition_seconds += clock(device) - started
        started = time.perf_counter()
        p, record = fit_layer(torch.cat(queries), torch.cat(targets), torch.cat(observations),
                              counts.cpu(), use_response)
        solve_seconds += time.perf_counter() - started
        parameters.append(p)
        records.append(dict(layer=layer, **record))
        print(f"  fitted {'q+r' if use_response else 'q'} layer {layer+1}/{len(model.blocks)}", flush=True)
    return parameters, dict(layers=records, acquisition_seconds=acquisition_seconds,
                            solve_seconds=solve_seconds)


def evaluate_streaming(parent, current, arrays, panel, rules, batch_size, device):
    """Build and discard only one evaluation batch's source/teacher cache pair."""
    users, length = len(panel), arrays[1].shape[1]
    rules = {name: device_parameters(parameters, device) for name, parameters in rules.items()}
    pieces = {name: [] for name in ("reuse", "exact", *METHODS)}
    costs = {"eval_source_prefill_seconds": 0.0, "eval_teacher_prefill_seconds": 0.0}
    timings = {name: 0.0 for name in pieces}
    checks = None
    for start in range(0, users, batch_size):
        stop = min(start + batch_size, users)
        events = tuple(torch.as_tensor(a[start:stop], device=device) for a in arrays[1:4])
        started = clock(device)
        source = parent.compute_kv(*events)
        costs["eval_source_prefill_seconds"] += clock(device) - started
        started = clock(device)
        teacher = current.compute_kv(*events)
        costs["eval_teacher_prefill_seconds"] += clock(device) - started
        candidates = torch.as_tensor(panel[start:stop], device=device)
        deltas = torch.as_tensor(arrays[4][start:stop], device=device)
        counts = torch.full((stop-start,), float(length), device=device)
        if checks is None:
            checks = reader_canary(current, source, teacher, candidates, deltas)
        for name, cache in (("reuse", source), ("exact", teacher)):
            started = clock(device)
            prediction, _ = score(current, cache, candidates, deltas)
            timings[name] += clock(device) - started
            pieces[name].append(prediction.cpu().numpy())
        for name, parameters in rules.items():
            started = clock(device)
            override = make_history_override(parameters, counts)
            timings[name+"_view"] = timings.get(name+"_view", 0.0) + clock(device) - started
            started = clock(device)
            prediction, _ = score(current, source, candidates, deltas, history_override=override)
            timings[name] += clock(device) - started
            pieces[name].append(prediction.cpu().numpy())
        # No evaluation cache or teacher response is kept for fitting or later batches.
        del source, teacher, events, candidates, deltas, counts, override, prediction, cache, _
        if stop % 64 == 0 or stop == users:
            print(f"  evaluated {stop}/{users} users", flush=True)
    return {name: np.concatenate(values) for name, values in pieces.items()}, checks, costs, timings


def model_binding(pair):
    return {role: pair[role]["checkpoint_sha256"] for role in ("parent", "current")}


def execution_sources():
    paths = [
        "scripts/design/run_query_read_probe.py", "scripts/design/query_read_probe.py",
        "scripts/design/run_shared_read_probe.py", "scripts/design/shared_read_probe.py",
        "scripts/design/diagnose_native_input.py", "scripts/design/diagnose_summary_objective.py",
        "scripts/design/competitor_data.py", "scripts/design/competitor_models.py",
        "scripts/evaluate_yambda500m_foundation_raw.py",
        "src/hstu_kvcache/adaptation/reader.py", "src/hstu_kvcache/models/hstu.py",
        "src/hstu_kvcache/models/attention.py", "src/hstu_kvcache/models/kv_cache.py",
        "src/hstu_kvcache/data/yambda_history.py", "src/hstu_kvcache/training/foundation.py",
    ]
    return {path: sha256_file(ROOT/path) for path in paths}


def load_calibration(directory, pair, settings, fit_uids):
    """Require exactly the pilot's frozen parameters, implementation and protocol."""
    prior_settings = json.loads((directory/"configuration.json").read_text())
    prior_summary = json.loads((directory/"summary.json").read_text())
    assert prior_summary["status"] == "completed" and prior_summary["mode"] == "pilot"
    assert prior_settings["config_sha256"] == settings["config_sha256"]
    assert prior_settings["source_sha256"] == settings["source_sha256"]
    record = next(edge for edge in prior_summary["edges"] if edge["edge"] == pair["edge"])
    path = directory/f"{pair['edge']}_rules.pt"
    digest = sha256_file(path)
    assert digest == record["calibration_rules_sha256"]
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    assert artifact["calibration_uids"] == fit_uids
    assert artifact["model_binding"] == model_binding(pair)
    assert artifact["config_sha256"] == settings["config_sha256"]
    assert set(artifact["rules"]) == set(METHODS)
    return artifact, path, digest


@torch.inference_mode()
def run(args):
    config = json.loads(args.config.read_text())
    assert config["scale"] == "medium" and config["ridge"] == .01
    groups = [config[name] for name in ("calibration_uids", "pilot_uids", "evaluation_uids")]
    assert all(len(group) == len(set(group)) for group in groups)
    assert all(not set(a) & set(b) for i, a in enumerate(groups) for b in groups[i+1:])
    assert sha256_file(ROOT/config["source_split"]["path"]) == config["source_split"]["sha256"]
    assert sha256_file(ROOT/config["model_chain"]) == config["model_chain_sha256"]
    if args.output.exists():
        raise FileExistsError(args.output)
    mode = "canary" if args.canary else "pilot" if args.pilot else "diagnostic"
    if args.calibration_from is not None and mode != "diagnostic":
        raise ValueError("--calibration-from is for the expanded diagnostic after a completed pilot")
    torch.set_num_threads(args.threads)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    args.output.mkdir(parents=True)
    edges = config["edges"][:1] if args.canary else config["edges"]
    fit_uids = config["calibration_uids"][:config["canary"]["calibration_users"]] if args.canary else config["calibration_uids"]
    bank_uids = config["pilot_uids"] if mode in ("canary", "pilot") else config["evaluation_uids"]
    eval_uids = bank_uids[:config["canary"]["evaluation_users"]] if args.canary else bank_uids
    settings = dict(config=config, config_sha256=sha256_file(args.config), mode=mode,
                    device=str(device), threads=args.threads, batch_size=args.batch_size,
                    torch_version=torch.__version__, backend=os.environ.get("EVOKV_ATTENTION_BACKEND"),
                    source_sha256=execution_sources(),
                    evaluation_bank_uids=bank_uids, calibration_uids=fit_uids, evaluation_uids=eval_uids,
                    calibration_from=str(args.calibration_from.resolve()) if args.calibration_from else None)
    write_json(args.output/"configuration.json", settings)
    numerical_check = self_check() if args.canary else None
    pairs = [load_model_pair("medium", edge, verify_hashes=True) for edge in edges]
    assert all(pair["admission"]["reuse_eligible"] for pair in pairs)
    data = pairs[0]["dataset"]
    needed_uids = bank_uids if args.calibration_from else config["calibration_uids"] + bank_uids
    started = time.perf_counter()
    history = load_histories(needed_uids, dataset_path=Path(data["manifest"]),
                             known_vocab_size=data["known_items"], oov_buckets=data["oov_buckets"],
                             start_timestamp=min(p["cutover"] for p in pairs)-1,
                             end_timestamp=max(p["cutover"] for p in pairs)+1,
                             max_history=config["history_length"], threads=args.threads)
    history_seconds = time.perf_counter()-started
    print(f"Loaded {len(needed_uids)} users in {history_seconds:.2f}s; mode={mode}", flush=True)
    summaries, all_rows, all_users = [], [], []
    for pair in pairs:
        edge_started = clock(device)
        print(f"Starting {pair['edge']}: fit={len(fit_uids)}, evaluation={len(eval_uids)}", flush=True)
        all_arrays = history_arrays(history, np.asarray(bank_uids), pair["cutover"], config["history_length"])
        bank_panel, _ = make_candidate_panel(all_arrays[1], data["known_items"])
        arrays = tuple(value[:len(eval_uids)] for value in all_arrays)
        evaluation_panel = bank_panel[:len(eval_uids)]
        models = []
        for role in ("parent", "current"):
            model, payload = load_model(Path(pair[role]["checkpoint"]), device)
            assert payload["config"] == pair["config"]
            model.requires_grad_(False)
            models.append(model)
            del payload
        parent, current = models
        calibration_started = clock(device)
        rule_path = args.output/f"{pair['edge']}_rules.pt"
        reused_from = None
        if args.calibration_from:
            artifact, original_path, rules_digest = load_calibration(args.calibration_from, pair, settings, fit_uids)
            shutil.copyfile(original_path, rule_path)
            assert sha256_file(rule_path) == rules_digest
            reused_from = str(original_path.resolve())
            rules, fitting = artifact["rules"], artifact["fitting"]
            fit_candidates, fit_deltas = artifact["calibration_candidates"], artifact["calibration_deltas"]
            calibration_costs = artifact["calibration_costs"]
        else:
            fit_arrays = history_arrays(history, np.asarray(config["calibration_uids"]), pair["cutover"], config["history_length"])
            full_fit_panel, _ = make_candidate_panel(fit_arrays[1], data["known_items"])
            fit_candidates = torch.as_tensor(full_fit_panel[:len(fit_uids), config["calibration_candidate_indices"]])
            fit_deltas = torch.as_tensor(fit_arrays[4][:len(fit_uids)])
            calibration_costs = {}
            started = clock(device)
            fit_source = collect_cache(parent, fit_arrays, len(fit_uids), args.batch_size, device)
            calibration_costs["fit_source_prefill_seconds"] = clock(device)-started
            started = clock(device)
            fit_teacher = collect_cache(current, fit_arrays, len(fit_uids), args.batch_size, device)
            calibration_costs["fit_teacher_prefill_seconds"] = clock(device)-started
            rules, fitting = {}, {}
            for name in METHODS:
                rules[name], fitting[name] = fit_rule(current, fit_source, fit_teacher,
                    fit_candidates.to(device), fit_deltas.to(device), name == "shared_query_response",
                    args.batch_size, device)
            artifact = dict(rules=rules, fitting=fitting, calibration_costs=calibration_costs,
                            calibration_uids=fit_uids, calibration_candidates=fit_candidates,
                            calibration_deltas=fit_deltas, model_binding=model_binding(pair),
                            config_sha256=settings["config_sha256"])
            torch.save(artifact, rule_path)
            rules_digest = sha256_file(rule_path)
            del fit_source, fit_teacher, fit_arrays, full_fit_panel
        calibration_seconds = clock(device)-calibration_started
        evaluation_started = clock(device)
        scores, checks, eval_costs, score_time = evaluate_streaming(parent, current, arrays,
            evaluation_panel[:, config["evaluation_candidate_indices"]], rules, args.batch_size, device)
        evaluation_seconds = clock(device)-evaluation_started
        # Preserve raw evidence before denominators or summary aggregation can fail.
        np.savez_compressed(args.output/f"{pair['edge']}.npz", uids=np.asarray(eval_uids),
            fit_uids=np.asarray(fit_uids), panel_population_uids=np.asarray(bank_uids),
            fit_candidates=fit_candidates.numpy(), fit_deltas=fit_deltas.numpy(),
            evaluation_panel=evaluation_panel, evaluation_deltas=arrays[4], **scores)
        rows, per_user = metrics(scores, eval_uids, pair["edge"])
        all_rows.extend(rows)
        all_users.extend(per_user)
        summary = dict(edge=pair["edge"], model_pair=pair, calibration_users=len(fit_uids),
                       evaluation_users=len(eval_uids), reader_check=checks, fitting=fitting,
                       calibration_rules_file=rule_path.name, calibration_rules_sha256=rules_digest,
                       calibration_reused_from=reused_from,
                       original_calibration_costs=calibration_costs,
                       costs={**eval_costs, **({} if reused_from else calibration_costs)},
                       calibration_execution_seconds=calibration_seconds,
                       calibration_teacher_users_this_run=0 if reused_from else len(fit_uids),
                       evaluation_seconds=evaluation_seconds, score_path_seconds=score_time,
                       metrics=rows, elapsed_seconds=clock(device)-edge_started)
        summaries.append(summary)
        write_json(args.output/"summary.json", dict(status="in_progress", mode=mode, edges=summaries))
        print(json.dumps(dict(edge=pair["edge"], metrics=rows, elapsed_seconds=summary["elapsed_seconds"])), flush=True)
        del parent, current, models, model, rules, artifact, all_arrays, arrays, bank_panel, scores
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = dict(status="completed", mode=mode, history_seconds=history_seconds,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  edges=summaries, interpretation="Two frozen shared probes on development users; no oracle, summary, AUC or lifecycle claim.")
    write_json(args.output/"summary.json", result)
    for filename, rows in (("metrics.csv", all_rows), ("per_user.csv", all_users)):
        with (args.output/filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.canary:
        measured = summaries[0]
        estimate = history_seconds + len(config["edges"]) * (
            measured["calibration_execution_seconds"] * len(config["calibration_uids"])/len(fit_uids)
            + measured["evaluation_seconds"] * len(config["evaluation_uids"])/len(eval_uids))
        write_json(args.output/"canary.pass.json", dict(passed=True, numerical_check=numerical_check,
                   reader_checks=[edge["reader_check"] for edge in summaries], estimated_full_seconds=estimate,
                   estimate="Scale calibration and streaming evaluation separately; retained resource probe, not service latency."))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--canary", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    parser.add_argument("--calibration-from", type=Path, help="Completed pilot directory; reuse frozen per-edge rules exactly.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
