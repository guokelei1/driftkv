#!/usr/bin/env python3
"""Two-edge 16-layer AUC diagnostic using the frozen dynamic evaluators.

Only initial cache placement and orchestration differ from the Large run.
Translation sees all 256 paired users on CPU; read fitting uses the original
GPU fitter through 128 users and the existing four-GPU fitter from 256 onward.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design.competitor_data import make_candidate_panel
from design.competitor_models import sha256_file
from design.expanded_read_calibration import calibrate_expanded, compare_reference
from design.large_auc_primitives import compare_translators, evaluate, fit_translators_streamed
from design.max_auc_models import load_model_pair
from design.run_auc_read_probe import fit_rule
from design.run_large_auc import (
    BUDGETS, META_COLUMNS, concatenate, join_quality, launch, requests, worker,
)
from design.run_shared_read_probe import cache_slice, reader_canary, write_json
from design.run_unified_auc import take_events
from design.run_unified_auc_parallel import (
    batch_spans, load_pair, prepared_arrays, setup_gpu, source_hashes,
)
from design.unified_auc_baselines import calibrate_layer_intervals, fit_translators
from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.baselines.kv_translate import KVTranslator
from hstu_kvcache.models import HSTUKVCache

DEFAULT_ROOT = ROOT / "results/insight/max_unified_auc_10k_20260921"


def gpu_cache(cache, count, device):
    return HSTUKVCache(cache.k[:, :count].to(device), cache.v[:, :count].to(device), cache.seq_len)


@torch.inference_mode()
def initial_calibration(args, config, pair):
    """Identical fitting inputs, with complete paired caches allocated on CPU."""
    device = setup_gpu(0, args.threads)
    arrays = prepared_arrays(args.prepared, pair["edge"], "fit", config["calibration_uids"])
    bank, _ = make_candidate_panel(arrays[1], pair["dataset"]["oov_bucket_start"])
    selected = bank[:, config["calibration_candidate_indices"]]
    for group in ("fit", "expanded_fit"):
        assert np.array_equal(selected, np.load(args.prepared / pair["edge"] / group / "panel.npy")[:256])
    users, budgets = (32, [8, 16, 32]) if args.canary else (256, BUDGETS[:3])
    profile_users = 8 if args.canary else 32
    began = time.perf_counter()
    parent, current = load_pair(pair, device)
    shape = (16, users, 1024, 320)
    source_cpu = HSTUKVCache(torch.empty(shape), torch.empty(shape), 1024)
    teacher_cpu = HSTUKVCache(torch.empty(shape), torch.empty(shape), 1024)
    for begin in range(0, users, args.batch_size):
        end = min(begin + args.batch_size, users)
        events = take_events(arrays, begin, end, device)
        for model, destination in ((parent, source_cpu), (current, teacher_cpu)):
            cache = model.compute_kv(*events)
            destination.k[:, begin:end].copy_(cache.k)
            destination.v[:, begin:end].copy_(cache.v)
            del cache
        del events
    del model, destination
    source, teacher = gpu_cache(source_cpu, max(budgets), device), gpu_cache(teacher_cpu, max(budgets), device)
    panel = torch.as_tensor(selected[:max(budgets)], device=device)
    delta = torch.as_tensor(arrays[4][:max(budgets)], device=device)
    numerical = reader_canary(current, cache_slice(source, 0, 32), cache_slice(teacher, 0, 32), panel[:32], delta[:32])
    rules, fits = {}, {}
    for count in budgets:
        for conditioned in (False, True):
            name = f"{'personal' if conditioned else 'shared'}_{count}"
            rules[name], fits[name] = fit_rule(current, cache_slice(source, 0, count),
                cache_slice(teacher, 0, count), panel[:count], delta[:count], conditioned, args.batch_size, device)
    del source, teacher
    gc.collect()
    torch.cuda.empty_cache()
    events = take_events(arrays, 0, profile_users, device)
    state = lr.capture_state(parent, *events)
    del parent
    exact = gpu_cache(teacher_cpu, profile_users, device)
    print("Profile all 136 contiguous intervals of the 16-layer model", flush=True)
    intervals, profiles, statistics = calibrate_layer_intervals(current, state, events,
        panel[:profile_users], delta[:profile_users], max_profile_users=profile_users,
        exact_cache=exact, candidate_chunk=2)
    assert len(profiles) == 136 and set(intervals) == set(range(1, 17))
    statistics["candidate_chunk"] = 2
    profile_reference = None
    if args.canary:
        reference_intervals, reference_profiles, _ = calibrate_layer_intervals(current, state, events,
            panel[:profile_users], delta[:profile_users], max_profile_users=profile_users,
            exact_cache=exact, candidate_chunk=8)
        assert intervals == reference_intervals
        actual_errors = np.array([row["mean_squared_logit_error"] for row in profiles])
        reference_errors = np.array([row["mean_squared_logit_error"] for row in reference_profiles])
        assert [row["interval"] for row in profiles] == [row["interval"] for row in reference_profiles]
        np.testing.assert_allclose(actual_errors, reference_errors, atol=2e-5, rtol=2e-5)
        profile_reference = dict(status="passed", users=profile_users, candidates_per_user=16,
            intervals_compared=136, candidate_chunks=[2, 8], selected_intervals_identical=True,
            maximum_absolute_mse_difference=float(np.max(np.abs(actual_errors-reference_errors))))
    del current, state, exact, events, panel, delta
    gc.collect()
    torch.cuda.empty_cache()
    reference = None
    if args.canary:
        source, teacher = gpu_cache(source_cpu, users, device), gpu_cache(teacher_cpu, users, device)
        reference = fit_translators(source, teacher, pair["config"]["num_heads"])
        del source, teacher
        gc.collect()
        torch.cuda.empty_cache()
    translators = fit_translators_streamed(source_cpu, teacher_cpu, pair["config"]["num_heads"], device)
    comparison = compare_translators(translators, reference) if reference is not None else None
    artifact = dict(rules=rules, fitting=fits, calibration_uids=config["calibration_uids"][:users],
        teacher_budgets=budgets, profile_users=profile_users, translate_users=users,
        intervals=intervals, interval_profiles=profiles, interval_profile_stats=statistics,
        translators={str(k): vars(mapper) for k, mapper in translators.items()}, numerical=numerical,
        translation_reference=comparison, profile_chunk_reference=profile_reference,
        translation_fitting="unchanged layer-sized GPU fitting from CPU paired KV")
    resource = dict(seconds=time.perf_counter()-began,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30),
        paired_cpu_cache_gib=4*source_cpu.k.numel()*source_cpu.k.element_size()/(1 << 30))
    return artifact, resource


def initial_worker(args):
    settings = json.loads((args.output / "configuration.json").read_text())
    pair = settings["model_pairs"][args.edge]
    directory = args.output / "initial" / pair["edge"]
    artifact, resource = initial_calibration(args, settings["config"], pair)
    path = directory / "calibration.pt"
    torch.save(artifact, path)
    write_json(directory / "summary.json", dict(status="completed", resource=resource,
        artifact_sha256=sha256_file(path), physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]),
        model_pair=pair, config_sha256=settings["config_sha256"], batch_size=args.batch_size))


def launch_initial(args, pairs):
    """Independent small baseline calibrations; never overlap expanded caches."""
    children = []
    try:
        for index, pair in enumerate(pairs):
            directory = args.output / "initial" / pair["edge"]
            directory.mkdir(parents=True)
            log = (directory / "runtime.log").open("w")
            command = [sys.executable, str(Path(__file__)), "--initial-worker", "--edge", str(index),
                "--config", str(args.config), "--prepared", str(args.prepared), "--output", str(args.output),
                "--batch-size", str(args.batch_size), "--threads", str(args.threads)]
            if args.canary:
                command.append("--canary")
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index),
                OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
            children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment), log))
        for process, _ in children:
            if process.wait() != 0:
                raise RuntimeError(f"Max initial calibration failed; see {args.output / 'initial'}")
    finally:
        for process, log in children:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()


@torch.inference_mode()
def resource_probe(args, config, pair, artifact):
    """Run one real global batch64 through every canary path before full launch."""
    device = setup_gpu(0, args.threads)
    all_uids = config["canary"]["mature_pilot_uids"]
    arrays = tuple(a[:64] for a in prepared_arrays(args.prepared, pair["edge"], "pilot", all_uids))
    uids = all_uids[:64]
    parent, current = load_pair(pair, device)
    translators = {int(k): KVTranslator(**value) for k, value in artifact["translators"].items()}
    raw, execution = evaluate(parent, current, arrays, uids, requests(config, uids, pair["cutover"]),
        artifact, translators, 64, 32, device)
    raw.to_parquet(args.output / "batch64_resource_scores.parquet", index=False)
    return dict(status="passed", batch_size=64, labels_read=False, **execution,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30))


def run(args):
    config = json.loads(args.config.read_text())
    assert config["scale"] == "max" and config["edges"] == [0, 1]
    assert config["teacher_budgets"] == BUDGETS and config["history_length"] == 1024
    assert len(config["calibration_uids"]) == 256 and len(config["expanded_calibration_uids"]) == 7144
    assert config["expanded_calibration_uids"][:256] == config["calibration_uids"]
    assert len(config["evaluation_uids"]) == 10000
    assert not set(config["expanded_calibration_uids"]) & set(config["evaluation_uids"])
    assert not set(config["pilot_uids"]) & (set(config["evaluation_uids"]) | set(config["expanded_calibration_uids"]))
    assert args.batch_size == (16 if args.canary else 64) and args.candidate_chunk == 32
    assert not (args.canary and (args.initial_only or args.initial_record))
    assert not (args.initial_only and args.initial_record)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    try:
        metadata_path = args.prepared / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        assert metadata["status"] == "completed" and metadata["config_sha256"] == sha256_file(args.config)
        for name in ("requests_fidelity", "requests_quality"):
            record = config["request_manifest"][name]
            assert sha256_file(ROOT / record["path"]) == record["sha256"]
        assert sha256_file(ROOT / config["model_chain"]) == config["model_chain_sha256"]
        files = source_hashes()
        for path in (Path(__file__), ROOT / "scripts/design/max_auc_models.py", ROOT / "scripts/design/run_large_auc.py",
                     ROOT / "scripts/design/large_auc_primitives.py", ROOT / "scripts/design/expanded_read_calibration.py",
                     ROOT / "scripts/insight_one_locality/common.py"):
            files[str(path.relative_to(ROOT))] = sha256_file(path)
        if not args.canary:
            canary = json.loads(args.canary_record.read_text())
            checked = json.loads((args.canary_record.parent / "configuration.json").read_text())
            assert canary["status"] == "completed" and canary["mode"] == "canary"
            assert checked["config_sha256"] == sha256_file(args.config) and checked["source_sha256"] == files
            assert canary["expanded_reference"]["status"] == canary["batch64_resource"]["status"] == "passed"
            assert canary["edges"][0]["translation_reference"]["status"] == "passed"
            assert canary["edges"][0]["profile_chunk_reference"]["status"] == "passed"
        pairs = [load_model_pair("max", index, verify_hashes=True)
                 for index in (config["edges"][:1] if args.canary else config["edges"])]
        uids = config["canary"]["mature_pilot_uids"] if args.canary else config["evaluation_uids"]
        assert len(uids) == (96 if args.canary else 10000)
        for pair in pairs:
            assert (pair["config"]["num_layers"], pair["config"]["hidden_size"], pair["config"]["num_heads"]) == (16, 320, 10)
            assert pair["admission"]["reuse_eligible"] or pair["edge"] in config["allowed_unadmitted_edges"]
            for group, expected in {"fit": config["calibration_uids"], "expanded_fit": config["expanded_calibration_uids"],
                                    "pilot" if args.canary else "evaluation": uids}.items():
                directory = args.prepared / pair["edge"] / group
                assert np.array_equal(np.load(directory / "uids.npy"), expected)
                for name, detail in metadata["outputs"][pair["edge"]][group]["files"].items():
                    assert sha256_file(directory / f"{name}.npy") == detail["sha256"]
        mode = "canary" if args.canary else "initial_resource_preflight" if args.initial_only else "diagnostic"
        settings = dict(mode=mode, config=config,
            config_sha256=sha256_file(args.config), evaluation_uids=uids, model_pairs=pairs, source_sha256=files,
            prepared_metadata_sha256=sha256_file(metadata_path), batch_size=args.batch_size,
            candidate_chunk=args.candidate_chunk, global_batch_spans=batch_spans(len(uids), args.batch_size),
            execution_overlay=dict(gpu_ids=[0, 1, 2, 3], worker_torch_threads=args.threads,
                initial_calibration="independent CPU-cache calibrations on GPUs0/1", serial_expanded_edges=True,
                data_threads=metadata.get("threads"), translator="unchanged layer-sized GPU fitting",
                profiler_candidate_chunk=2,
                expanded_calibration_cache="CPU memory; at most one edge", expanded_budgets=BUDGETS[3:]),
            aggregation="concatenate raw scores, then pooled AUC per edge using the Max request-ID label source")
        write_json(args.output / "configuration.json", settings)
        if args.initial_record:
            checked = json.loads((args.initial_record / "configuration.json").read_text())
            complete = json.loads((args.initial_record / "summary.json").read_text())
            assert complete["status"] == "completed" and complete["mode"] == "initial_resource_preflight"
            for key in ("config_sha256", "source_sha256", "model_pairs", "prepared_metadata_sha256", "batch_size"):
                assert checked[key] == settings[key], key
            initial_root = args.initial_record / "initial"
        else:
            launch_initial(args, pairs)
            initial_root = args.output / "initial"
        initial_records = {pair["edge"]: json.loads((initial_root / pair["edge"] / "summary.json").read_text())
                           for pair in pairs}
        if args.initial_record:
            assert initial_records == checked["initial_calibration_execution"]["artifacts"] == complete["edges"]
        for pair in pairs:
            edge, item = pair["edge"], initial_records[pair["edge"]]
            assert item["status"] == "completed"
            assert item["model_pair"] == pair and item["config_sha256"] == settings["config_sha256"]
            assert item["batch_size"] == args.batch_size
            assert sha256_file(initial_root / edge / "calibration.pt") == item["artifact_sha256"]
        settings["initial_calibration_execution"] = dict(
            directory=str(initial_root), source_sha256=files["scripts/design/run_max_auc.py"],
            model_edges=[pair["edge"] for pair in pairs],
            artifacts=initial_records, preflight_record=str(args.initial_record) if args.initial_record else None)
        write_json(args.output / "configuration.json", settings)
        if args.initial_only:
            write_json(args.output / "summary.json", dict(status="completed", mode=mode,
                elapsed_seconds=time.perf_counter()-began, config_sha256=settings["config_sha256"],
                source_sha256=files, edges=initial_records))
            return
        summaries, additional = [], {}
        for index, pair in enumerate(pairs):
            started, edge = time.perf_counter(), pair["edge"]
            initial = initial_root / edge
            execution_record = json.loads((initial / "summary.json").read_text())
            assert execution_record["status"] == "completed"
            assert execution_record["artifact_sha256"] == sha256_file(initial / "calibration.pt")
            artifact = torch.load(initial / "calibration.pt", map_location="cpu", weights_only=False)
            resource = execution_record["resource"]
            directory = args.prepared / edge / "expanded_fit"
            if not args.canary:
                print(f"Expanded Max {edge}: one paired 7144-user CPU cache, four GPUs", flush=True)
                extension = calibrate_expanded(pair, directory, directory / "panel.npy", BUDGETS[3:],
                    batch_size=64, gpus=(0, 1, 2, 3), threads=args.threads)
                artifact["rules"].update(extension["rules"])
                artifact["fitting"].update(extension["fitting"])
                artifact.update(calibration_uids=config["expanded_calibration_uids"], teacher_budgets=BUDGETS,
                    extension_execution=extension["execution"], panel_sha256=extension["panel_sha256"])
                del extension
            artifact.update(config_sha256=settings["config_sha256"],
                model_binding={role: pair[role]["checkpoint_sha256"] for role in ("parent", "current")})
            rule_path = args.output / f"{edge}_calibration.pt"
            torch.save(artifact, rule_path)
            if args.canary:
                launch(args, index, edge, reference=True)
            launch(args, index, edge)
            raw, execution = concatenate(args, config, pair, uids)
            paths = [name for name in raw if name not in META_COLUMNS]
            assert len(paths) == (35 if args.canary else 47)
            record = dict(edge=edge, model_pair=pair, rules_sha256=sha256_file(rule_path),
                teacher_budgets=artifact["teacher_budgets"], profile_users=artifact["profile_users"],
                translate_users=artifact["translate_users"], intervals=artifact["intervals"],
                interval_profiles=artifact["interval_profiles"], interval_profile_stats=artifact["interval_profile_stats"],
                numerical=artifact["numerical"], translation_reference=artifact["translation_reference"],
                profile_chunk_reference=artifact["profile_chunk_reference"],
                calibration_seconds=resource["seconds"], calibration_peak_allocated_gib=resource["peak_allocated_gib"],
                calibration_execution=artifact.get("extension_execution"), **execution)
            if args.canary:
                reference = pd.read_parquet(args.output / edge / "reference" / "scores.parquet")
                assert raw[META_COLUMNS].equals(reference[META_COLUMNS])
                differences = {}
                for name in paths:
                    np.testing.assert_allclose(raw[name], reference[name], atol=2e-5, rtol=2e-5)
                    differences[name] = float(np.max(np.abs(raw[name].to_numpy()-reference[name].to_numpy())))
                record.update(maximum_absolute_logit_difference=max(differences.values()),
                    per_path_logit_difference=differences, labels_read=False)
                probe = calibrate_expanded(pair, directory, directory / "panel.npy", [128],
                    batch_size=32, gpus=(0, 1, 2, 3), threads=args.threads)
                torch.save(probe, args.output / "expanded128_calibration.pt")
                additional["expanded_reference"] = compare_reference(pair, directory, directory / "panel.npy",
                    probe, budget=128, batch_size=32, threads=args.threads)
                additional["expanded_execution"] = probe["execution"]
                del probe
                gc.collect()
                torch.cuda.empty_cache()
                additional["batch64_resource"] = resource_probe(args, config, pair, artifact)
            else:
                record.update(**join_quality(config, raw, args.output, edge))
            record["elapsed_seconds"] = time.perf_counter()-started
            summaries.append(record)
            write_json(args.output / f"{edge}_summary.json", record)
            write_json(args.output / "summary.json", dict(status="running", mode=settings["mode"], edges=summaries))
            print(f"Completed Max {edge} in {record['elapsed_seconds']:.1f}s after initial calibration", flush=True)
        write_json(args.output / "summary.json", dict(status="completed", mode=settings["mode"],
            elapsed_seconds=time.perf_counter()-began, edges=summaries, **additional))
    except Exception as exc:
        write_json(args.output / "summary.json", dict(status="failed", error=repr(exc),
            elapsed_seconds=time.perf_counter()-began))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/max_unified_auc_10k_20260921.json")
    parser.add_argument("--prepared", type=Path, default=DEFAULT_ROOT / "prepared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--canary-record", type=Path, default=DEFAULT_ROOT / "canary_v2/summary.json")
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--initial-record", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-chunk", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--initial-worker", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--edge", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    if args.initial_worker:
        initial_worker(args)
    elif args.worker:
        worker(args)
    else:
        run(args)
