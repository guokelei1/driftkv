#!/usr/bin/env python3
"""Retain a fixed ridge sensitivity screen on the 96-user Large pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design import bias_read_probe
from design.competitor_models import load_model_pair, sha256_file
from design.run_auc_read_probe import fit_rule
from design.run_large_auc import join_quality, requests
from design.run_shared_read_probe import cache_slice, collect_cache, device_parameters, reader_canary, write_json
from design.run_unified_auc import take_events
from design.run_unified_auc_parallel import load_pair, prepared_arrays, setup_gpu
from hstu_kvcache.adaptation.reader import score

DEFAULT_ROOT = ROOT / "results/insight/large_unified_auc_10k_20260920"
RIDGES = (.001, .01, .1)


def fit_layer_ridge(wanted, observed, counts, use_response, ridge):
    """The original FP64 global solve with its penalty exposed as an argument."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        target = wanted.cpu().double().transpose(1, 2).flatten(2)
        users, queries, _ = target.shape
        counts = counts.cpu().double()
        weight = counts.square()/counts.square().mean()
        x = torch.ones(users, queries, 1, dtype=torch.double)
        parameters = {}
        if use_response:
            native = observed.cpu().double()
            center = native.mean((0, 1))
            scale = native.std((0, 1)).clamp_min(1e-4)
            x = torch.cat((x, (native-center)/scale), dim=-1)
            parameters.update(read_center=center, read_scale=scale)
        design, y = x.flatten(0, 1), target.flatten(0, 1)
        row_weight = weight[:, None].expand(users, queries).reshape(-1, 1)
        gram = design.T @ (design*row_weight)/queries
        rhs = design.T @ (y*row_weight)/queries
        system = gram + ridge*users*torch.eye(gram.shape[0], dtype=torch.double)
        solution = torch.linalg.solve(system, rhs)
        torch.testing.assert_close(system @ solution, rhs, atol=1e-7, rtol=1e-6)
        parameters["bias"] = solution[0]
        if use_response:
            parameters["read_weights"] = solution[1:]
        residual = (design @ solution-y).reshape_as(target)
        stats = dict(fitting_rate_mse=float(residual.square().mean()),
            fitting_aggregate_objective=float((residual.square().mean((1, 2))*weight).mean()),
            normal_relative_residual=float((system @ solution-rhs).abs().max()/rhs.abs().max().clamp_min(1e-30)),
            count_square_mean=float(counts.square().mean()), shared_parameters=solution.numel(),
            read_dimension=observed.shape[-1] if use_response else 0,
            fitting_scenes=users, fitting_queries=queries, ridge=ridge,
            use_native_response=bool(use_response), cpu_solver_threads=1)
    finally:
        torch.set_num_threads(previous_threads)
    return parameters, stats


def self_check():
    generator = torch.Generator().manual_seed(172220)
    observed = torch.randn(7, 5, 6, generator=generator, dtype=torch.double)
    wanted = torch.randn(7, 2, 5, 3, generator=generator, dtype=torch.double)
    counts = torch.tensor([32., 64., 128., 256., 512., 768., 1024.], dtype=torch.double)
    maximum = 0.0
    for conditioned in (False, True):
        reference, old_stats = bias_read_probe.fit_layer(wanted, observed, counts, conditioned)
        actual, new_stats = fit_layer_ridge(wanted, observed, counts, conditioned, .01)
        assert old_stats == new_stats
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], atol=1e-12, rtol=1e-12)
            maximum = max(maximum, float((actual[name]-reference[name]).abs().max()))
    return dict(status="passed", reference_ridge=.01, maximum_parameter_difference=maximum)


@torch.inference_mode()
def evaluate_pilot(parent, current, arrays, uids, rows, rules, device):
    """Read the same fixed snapshots and real requests under all six rules."""
    values = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in ["parent", "reuse", "exact", *rules]}
    by_uid = {int(uid): frame.index.to_numpy() for uid, frame in rows.groupby("uid", sort=False)}
    installed = {name: device_parameters(parameters, device) for name, parameters in rules.items()}
    counts = np.zeros(len(uids), dtype=np.int64)
    began = time.perf_counter()
    for begin in range(0, len(uids), 16):
        end = min(begin+16, len(uids))
        events = take_events(arrays, begin, end, device)
        source, teacher = parent.compute_kv(*events), current.compute_kv(*events)
        groups = [by_uid.get(int(uid), np.empty(0, dtype=np.int64)) for uid in uids[begin:end]]
        counts[begin:end] = [len(group) for group in groups]
        lengths = torch.full((end-begin,), 1024., device=device)
        overrides = {name: bias_read_probe.make_history_override(parameters, lengths)
                     for name, parameters in installed.items()}
        maximum = max(map(len, groups), default=0)
        for offset in range(0, maximum, 32):
            width = min(32, maximum-offset)
            item = np.ones((end-begin, width), dtype=np.int64)
            delta = np.ones((end-begin, width), dtype=np.float32)
            indices = np.full((end-begin, width), -1, dtype=np.int64)
            for local, group in enumerate(groups):
                take = group[offset:offset+width]
                indices[local, :len(take)] = take
                if len(take):
                    item[local, :len(take)] = rows.loc[take, "item_idx"].to_numpy(dtype=np.int64)
                    delta[local, :len(take)] = rows.loc[take, "query_timestamp"].to_numpy()-arrays[0][begin+local, -1]
            panel, query_delta = torch.as_tensor(item, device=device), torch.as_tensor(delta, device=device)
            mask = indices >= 0
            for name in values:
                model = parent if name == "parent" else current
                cache = teacher if name == "exact" else source
                prediction, _ = score(model, cache, panel, query_delta, history_override=overrides.get(name))
                values[name][indices[mask]] = prediction.cpu().numpy()[mask]
        del source, teacher, events, overrides
    assert sum(counts) == len(rows) and all(np.isfinite(value).all() for value in values.values())
    last = {int(uid): int(stamps[-1]) for uid, stamps in zip(uids, arrays[0], strict=True)}
    raw = rows.assign(prefix_last_timestamp=rows.uid.map(last), prefix_length=1024, **values)
    assert (raw.prefix_last_timestamp < raw.query_timestamp).all()
    return raw, dict(snapshot_users=len(uids), feedback_users=len(by_uid),
        no_feedback_users=len(uids)-len(by_uid), requests=len(rows),
        per_user_requests=counts.tolist(), elapsed_seconds=time.perf_counter()-began)


@torch.inference_mode()
def worker(args):
    settings = json.loads((args.output / "configuration.json").read_text())
    config, pair = settings["config"], settings["model_pairs"][args.edge]
    edge = pair["edge"]
    directory = args.output / edge
    started = time.perf_counter()
    try:
        device = setup_gpu(args.gpu, args.threads)
        parent, current = load_pair(pair, device)
        arrays = prepared_arrays(args.prepared, edge, "fit", config["calibration_uids"])
        panel_path = args.prepared / edge / "fit" / "panel.npy"
        panel = torch.as_tensor(np.load(panel_path), device=device)
        delta = torch.as_tensor(arrays[4], device=device)
        source = collect_cache(parent, arrays, 256, 64, device)
        teacher = collect_cache(current, arrays, 256, 64, device)
        numerical = reader_canary(current, source, teacher, panel, delta)
        rules, fits, methods = {}, {}, {}
        original_fit = bias_read_probe.fit_layer
        try:
            for ridge in RIDGES:
                def with_ridge(wanted, observed, counts, conditioned, penalty=ridge):
                    return fit_layer_ridge(wanted, observed, counts, conditioned, penalty)
                bias_read_probe.fit_layer = with_ridge
                for conditioned in (False, True):
                    name = f"{'personal' if conditioned else 'shared'}_ridge_{str(ridge).replace('.', 'p')}"
                    print(f"{edge}: fit {name}", flush=True)
                    rules[name], fits[name] = fit_rule(current, source, teacher, panel, delta,
                                                      conditioned, 64, device)
                    methods[name] = dict(ridge=ridge, native_response_conditioning=conditioned, teacher_users=256)
        finally:
            bias_read_probe.fit_layer = original_fit
        artifact = dict(rules=rules, fitting=fits, methods=methods, calibration_uids=config["calibration_uids"],
            model_binding={role: pair[role]["checkpoint_sha256"] for role in ("parent", "current")},
            config_sha256=settings["config_sha256"], panel_sha256=sha256_file(panel_path))
        rule_path = directory / "calibration.pt"
        torch.save(artifact, rule_path)
        fitting_seconds = time.perf_counter()-started
        del source, teacher, arrays, panel, delta
        torch.cuda.empty_cache()
        uids = config["canary"]["mature_pilot_uids"]
        arrays = prepared_arrays(args.prepared, edge, "pilot", uids)
        rows = requests(config, uids, pair["cutover"])
        raw, execution = evaluate_pilot(parent, current, arrays, uids, rows, rules, device)
        record = dict(status="completed", edge=edge, model_pair=pair, methods=methods,
            calibration_uids=config["calibration_uids"], evaluation_uids=uids, rules_sha256=sha256_file(rule_path),
            numerical=numerical, fitting_seconds=fitting_seconds, **execution,
            **join_quality(config, raw, directory, edge))
        record.update(elapsed_seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30), device=str(device))
        write_json(directory / "summary.json", record)
    except Exception as exc:
        write_json(directory / "summary.json", dict(status="failed", error=repr(exc),
            elapsed_seconds=time.perf_counter()-started))
        raise


def run(args):
    config = json.loads(args.config.read_text())
    assert config["scale"] == "large" and config["edges"] == [0, 1, 2, 3, 4]
    assert len(config["calibration_uids"]) == 256 and len(config["canary"]["mature_pilot_uids"]) == 96
    assert not set(config["canary"]["mature_pilot_uids"]) & set(config["calibration_uids"])
    assert not set(config["canary"]["mature_pilot_uids"]) & set(config["evaluation_uids"])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    began = time.perf_counter()
    children = []
    try:
        check = self_check()
        prepared = json.loads((args.prepared / "metadata.json").read_text())
        assert prepared["status"] == "completed" and prepared["config_sha256"] == sha256_file(args.config)
        pairs = [load_model_pair("large", index, verify_hashes=True) for index in config["edges"]]
        for pair in pairs:
            assert pair["admission"]["reuse_eligible"] or pair["edge"] in config["allowed_unadmitted_edges"]
            for group in ("fit", "pilot"):
                for name, detail in prepared["outputs"][pair["edge"]][group]["files"].items():
                    assert sha256_file(args.prepared / pair["edge"] / group / f"{name}.npy") == detail["sha256"]
        for name in ("requests_fidelity", "requests_quality"):
            record = config["request_manifest"][name]
            assert sha256_file(ROOT / record["path"]) == record["sha256"]
        sources = [Path(__file__), ROOT / "scripts/design/run_auc_read_probe.py",
            ROOT / "scripts/design/bias_read_probe.py", ROOT / "scripts/design/run_large_auc.py",
            ROOT / "src/hstu_kvcache/adaptation/reader.py"]
        settings = dict(mode="development_ridge_sensitivity", config=config, config_sha256=sha256_file(args.config),
            model_pairs=pairs, evaluation_uids=config["canary"]["mature_pilot_uids"], ridges=list(RIDGES),
            calibration_users=256, candidates_per_user=16, primary_metric="pooled ROC_AUC",
            main_diagnostic_ridge=.01, parameter_selection="none; retain every ridge and edge",
            source_sha256={str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
            prepared_metadata_sha256=sha256_file(args.prepared / "metadata.json"),
            execution=dict(gpus=[0, 1, 2, 3], concurrent_edges=4, fifth_edge_queued=True,
                calibration_batch_size=64, evaluation_batch_size=16, candidate_chunk=32, worker_threads=args.threads),
            self_check=check)
        write_json(args.output / "configuration.json", settings)
        for wave in (range(4), range(4, 5)):
            children = []
            for index in wave:
                edge = pairs[index]["edge"]
                directory = args.output / edge
                directory.mkdir()
                log = (directory / "runtime.log").open("w")
                command = [sys.executable, str(Path(__file__)), "--worker", "--edge", str(index),
                    "--gpu", str(index % 4), "--config", str(args.config), "--prepared", str(args.prepared),
                    "--output", str(args.output), "--threads", str(args.threads)]
                env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
                children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log))
            for process, log in children:
                if process.wait() != 0:
                    raise RuntimeError("A ridge worker failed; see its retained runtime.log")
                log.close()
        records = [json.loads((args.output / pair["edge"] / "summary.json").read_text()) for pair in pairs]
        assert all(record["status"] == "completed" for record in records)
        write_json(args.output / "summary.json", dict(status="completed", mode=settings["mode"],
            elapsed_seconds=time.perf_counter()-began, self_check=check, ridges=list(RIDGES), edges=records))
    except Exception as exc:
        write_json(args.output / "summary.json", dict(status="failed", error=repr(exc),
            elapsed_seconds=time.perf_counter()-began))
        raise
    finally:
        for process, log in children:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/large_unified_auc_10k_20260920.json")
    parser.add_argument("--prepared", type=Path, default=DEFAULT_ROOT / "prepared")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "ridge_probe")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--edge", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    worker(args) if args.worker else run(args)
