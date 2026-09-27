#!/usr/bin/env python3
"""Five-edge, ten-layer AUC diagnostic with bounded-memory calibration."""

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
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design.competitor_data import make_candidate_panel
from design.competitor_models import load_model_pair, sha256_file
from design.expanded_read_calibration import calibrate_expanded
from design.large_auc_primitives import compare_translators, evaluate, fit_translators_streamed
from design.run_auc_read_probe import fit_rule
from design.run_shared_read_probe import cache_slice, collect_cache, reader_canary, write_json
from design.run_unified_auc import take_events
from design.run_unified_auc_parallel import (
    REQUEST_COLUMNS, batch_spans, load_pair, prepared_arrays, setup_gpu, source_hashes,
)
from design.unified_auc_baselines import calibrate_layer_intervals, fit_translators
from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.baselines.kv_translate import KVTranslator
from hstu_kvcache.evaluation.binary_metrics import binary_metrics, _roc_auc
from hstu_kvcache.models import HSTUKVCache

DEFAULT_ROOT = ROOT / "results/insight/large_unified_auc_10k_20260920"
BUDGETS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 7144]
META_COLUMNS = [*REQUEST_COLUMNS, "prefix_last_timestamp", "prefix_length"]


def requests(config, uids, cutover):
    """The original filter, using the sealed Large-ID fidelity panel."""
    return pq.read_table(ROOT / config["request_manifest"]["requests_fidelity"]["path"], filters=[
        ("uid", "in", uids), ("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
        ("query_timestamp", ">=", cutover), ("query_timestamp", "<", cutover+14*86400)],
        columns=list(REQUEST_COLUMNS)).to_pandas().sort_values(
        ["uid", "query_timestamp", "item_idx", "request_id"]).reset_index(drop=True)


def join_quality(config, raw, output, edge):
    """Seal Large logits before joining their Large request-ID label source."""
    path = output / f"{edge}_scores.parquet"
    raw.to_parquet(path, index=False)
    digest = sha256_file(path)
    write_json(output / f"{edge}_scores.seal.json", dict(sha256=digest, rows=len(raw), labels_joined=False))
    labels = pq.read_table(ROOT / config["request_manifest"]["requests_quality"]["path"],
        filters=[("request_id", "in", raw.request_id.tolist())],
        columns=[*REQUEST_COLUMNS, "label"]).to_pandas()
    full = raw.merge(labels, on=list(REQUEST_COLUMNS), how="left", validate="one_to_one")
    assert len(full) == len(raw) and full.label.isin([0, 1]).all()
    full.to_parquet(output / f"{edge}_quality.parquet", index=False)
    paths = [name for name in raw if name not in META_COLUMNS]
    metrics = {name: binary_metrics(full.label.to_numpy(), full[name].to_numpy()) for name in paths}
    for name in paths:
        metrics[name]["ROC_AUC"] = _roc_auc(full.label.to_numpy(), full[name].to_numpy(dtype=np.float64))
    auc = {name: value["ROC_AUC"] for name, value in metrics.items()}
    gap, gain = auc["exact"]-auc["reuse"], auc["exact"]-auc["parent"]
    quality = {name: dict(auc=auc[name], delta_auc_pp=100*(auc[name]-auc["reuse"]),
        auc_gap_recovery=(auc[name]-auc["reuse"])/gap if gap > 1e-4 else None,
        metrics=metrics[name]) for name in paths}
    return dict(quality=quality, positives=int(full.label.sum()), negatives=int((full.label == 0).sum()),
        exact_minus_reuse_auc=gap, current_minus_parent_auc=gain,
        update_benefit_lost=gap/gain if gain > 1e-4 else None,
        scores_sha256=digest, quality_sha256=sha256_file(output / f"{edge}_quality.parquet"))


@torch.inference_mode()
def initial_calibration(args, config, pair):
    device = setup_gpu(0, args.threads)
    arrays = prepared_arrays(args.prepared, pair["edge"], "fit", config["calibration_uids"])
    bank, _ = make_candidate_panel(arrays[1], pair["dataset"]["known_items"])
    selected_panel = bank[:, config["calibration_candidate_indices"]]
    assert np.array_equal(selected_panel, np.load(args.prepared / pair["edge"] / "fit" / "panel.npy"))
    assert np.array_equal(selected_panel,
        np.load(args.prepared / pair["edge"] / "expanded_fit" / "panel.npy", mmap_mode="r")[:256])
    users = 32 if args.canary else 256
    budgets = [8, 16, 32] if args.canary else BUDGETS[:4]
    profile_users = 8 if args.canary else 32
    parent, current = load_pair(pair, device)
    began = time.perf_counter()
    source = collect_cache(parent, arrays, users, args.batch_size, device)
    teacher = collect_cache(current, arrays, users, args.batch_size, device)
    panel = torch.as_tensor(selected_panel[:users], device=device)
    delta = torch.as_tensor(arrays[4][:users], device=device)
    numerical = reader_canary(current, source, teacher, panel, delta)
    rules, fits = {}, {}
    for count in budgets:
        for conditioned in (False, True):
            name = f"{'personal' if conditioned else 'shared'}_{count}"
            rules[name], fits[name] = fit_rule(current, cache_slice(source, 0, count),
                cache_slice(teacher, 0, count), panel[:count], delta[:count],
                conditioned, args.batch_size, device)
    print("Profile all55 contiguous intervals of the ten-layer model", flush=True)
    events = take_events(arrays, 0, profile_users, device)
    state = lr.capture_state(parent, *events)
    intervals, profiles, profile_stats = calibrate_layer_intervals(current, state, events,
        panel[:profile_users], delta[:profile_users], max_profile_users=profile_users,
        exact_cache=cache_slice(teacher, 0, profile_users))
    assert len(profiles) == 55 and set(intervals) == set(range(1, 11))
    del state, events, parent, current, panel, delta
    source_cpu = HSTUKVCache(source.k.cpu(), source.v.cpu(), source.seq_len)
    teacher_cpu = HSTUKVCache(teacher.k.cpu(), teacher.v.cpu(), teacher.seq_len)
    reference = None
    if args.canary:
        print("Canary: compare streamed translation with the original fitter", flush=True)
        reference = fit_translators(source, teacher, pair["config"]["num_heads"])
    del source, teacher
    gc.collect()
    torch.cuda.empty_cache()
    translators = fit_translators_streamed(source_cpu, teacher_cpu,
        pair["config"]["num_heads"], device)
    comparison = compare_translators(translators, reference) if reference is not None else None
    artifact = dict(rules=rules, fitting=fits, calibration_uids=config["calibration_uids"][:users],
        teacher_budgets=budgets, profile_users=profile_users, translate_users=users,
        intervals=intervals, interval_profiles=profiles, interval_profile_stats=profile_stats,
        translators={str(k): vars(mapper) for k, mapper in translators.items()}, numerical=numerical,
        translation_reference=comparison, translation_fitting="layer-sized GPU fitting from CPU paired KV")
    resource = dict(seconds=time.perf_counter()-began,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30))
    del source_cpu, teacher_cpu, reference, translators, arrays
    gc.collect()
    torch.cuda.empty_cache()
    return artifact, resource


@torch.inference_mode()
def worker(args):
    settings = json.loads((args.output / "configuration.json").read_text())
    pair = settings["model_pairs"][args.edge]
    edge, uids = pair["edge"], settings["evaluation_uids"]
    begin, end = (0, len(uids)) if args.reference else batch_spans(len(uids), args.batch_size)[args.rank]
    arrays = tuple(array[begin:end] for array in prepared_arrays(args.prepared, edge,
        "pilot" if settings["mode"] == "canary" else "evaluation", uids))
    uids = uids[begin:end]
    rows = requests(settings["config"], uids, pair["cutover"])
    rule_path = args.output / f"{edge}_calibration.pt"
    artifact = torch.load(rule_path, map_location="cpu", weights_only=False)
    assert artifact["config_sha256"] == settings["config_sha256"]
    assert artifact["model_binding"] == {role: pair[role]["checkpoint_sha256"] for role in ("parent", "current")}
    translators = {int(k): KVTranslator(**value) for k, value in artifact["translators"].items()}
    directory = args.output / edge / ("reference" if args.reference else f"rank{args.rank}")
    device = setup_gpu(args.rank, args.threads)
    parent, current = load_pair(pair, device)
    raw, execution = evaluate(parent, current, arrays, uids, rows, artifact, translators,
        args.batch_size, args.candidate_chunk, device)
    path = directory / "scores.parquet"
    raw.to_parquet(path, index=False)
    write_json(directory / "execution.json", dict(status="completed", **execution,
        evaluation_uids=uids, scores_sha256=sha256_file(path), rules_sha256=sha256_file(rule_path),
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30), device=str(device)))


def launch(args, edge_index, edge, reference=False):
    children = []
    try:
        for rank in ([0] if reference else range(4)):
            directory = args.output / edge / ("reference" if reference else f"rank{rank}")
            directory.mkdir(parents=True)
            log = (directory / "runtime.log").open("w")
            command = [sys.executable, str(Path(__file__)), "--worker", "--edge", str(edge_index),
                "--rank", str(rank), "--config", str(args.config), "--prepared", str(args.prepared),
                "--output", str(args.output), "--batch-size", str(args.batch_size),
                "--candidate-chunk", str(args.candidate_chunk), "--threads", str(args.threads)]
            if reference:
                command.append("--reference")
            env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
            children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        for child, _ in children:
            if child.wait() != 0:
                raise RuntimeError(f"Large evaluation worker failed; see {args.output / edge}")
    finally:
        for child, log in children:
            if child.poll() is None:
                child.terminate()
            log.close()
        for child, _ in children:
            child.wait()


def concatenate(args, config, pair, uids):
    edge = pair["edge"]
    directories = [args.output / edge / f"rank{rank}" for rank in range(4)]
    workers = [json.loads((directory / "execution.json").read_text()) for directory in directories]
    assert all(record["status"] == "completed" for record in workers)
    assert [uid for record in workers for uid in record["evaluation_uids"]] == uids
    for directory, record in zip(directories, workers, strict=True):
        assert record["scores_sha256"] == sha256_file(directory / "scores.parquet")
        assert record["rules_sha256"] == sha256_file(args.output / f"{edge}_calibration.pt")
    raw = pd.concat([pd.read_parquet(directory / "scores.parquet") for directory in directories], ignore_index=True)
    raw = raw.sort_values(["uid", "query_timestamp", "item_idx", "request_id"]).reset_index(drop=True)
    assert raw[list(REQUEST_COLUMNS)].equals(requests(config, uids, pair["cutover"]))
    assert not raw.request_id.duplicated().any()
    counts = raw.groupby("uid").size().reindex(uids, fill_value=0).to_numpy(dtype=np.int64)
    assert np.array_equal(counts, [n for record in workers for n in record["per_user_requests"]])
    return raw, dict(snapshot_users=len(uids), feedback_users=int((counts > 0).sum()),
        no_feedback_users=int((counts == 0).sum()), requests=len(raw), per_user_requests=counts.tolist(),
        checks=sorted({check for record in workers for check in record["checks"]}), workers=workers)


def run(args):
    config = json.loads(args.config.read_text())
    assert config["scale"] == "large" and config["edges"] == [0, 1, 2, 3, 4]
    assert config["teacher_budgets"] == BUDGETS and config["history_length"] == 1024
    assert len(config["calibration_uids"]) == 256 and len(config["expanded_calibration_uids"]) == 7144
    assert config["expanded_calibration_uids"][:256] == config["calibration_uids"]
    assert len(config["evaluation_uids"]) == 10000
    assert not set(config["expanded_calibration_uids"]) & set(config["evaluation_uids"])
    assert not set(config["pilot_uids"]) & (set(config["evaluation_uids"]) | set(config["expanded_calibration_uids"]))
    assert args.batch_size == (16 if args.canary else 64) and args.candidate_chunk == 32
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    began = time.perf_counter()
    try:
        metadata_path = args.prepared / "metadata.json"
        prepared = json.loads(metadata_path.read_text())
        assert prepared["status"] == "completed" and prepared["config_sha256"] == sha256_file(args.config)
        for name in ("requests_fidelity", "requests_quality"):
            record = config["request_manifest"][name]
            assert sha256_file(ROOT / record["path"]) == record["sha256"]
        assert sha256_file(ROOT / config["model_chain"]) == config["model_chain_sha256"]
        files = source_hashes()
        for path in (Path(__file__), ROOT / "scripts/design/large_auc_primitives.py",
                     ROOT / "scripts/design/expanded_read_calibration.py",
                     ROOT / "scripts/insight_one_locality/common.py"):
            files[str(path.relative_to(ROOT))] = sha256_file(path)
        if not args.canary:
            canary = json.loads(args.canary_record.read_text())
            checked = json.loads((args.canary_record.parent / "configuration.json").read_text())
            assert canary["status"] == "completed" and canary["mode"] == "canary"
            assert checked["config_sha256"] == sha256_file(args.config) and checked["source_sha256"] == files
            assert canary["edges"][0]["translation_reference"]["status"] == "passed"
        pairs = [load_model_pair("large", index, verify_hashes=True)
                 for index in (config["edges"][:1] if args.canary else config["edges"])]
        for pair in pairs:
            assert (pair["config"]["num_layers"], pair["config"]["hidden_size"], pair["config"]["num_heads"]) == (10, 320, 10)
            assert pair["admission"]["reuse_eligible"] or pair["edge"] in config["allowed_unadmitted_edges"]
            groups = {"fit": config["calibration_uids"], "expanded_fit": config["expanded_calibration_uids"],
                      "pilot" if args.canary else "evaluation": config["canary"]["mature_pilot_uids"] if args.canary else config["evaluation_uids"]}
            for group, uids in groups.items():
                directory = args.prepared / pair["edge"] / group
                assert np.array_equal(np.load(directory / "uids.npy"), uids)
                for name, detail in prepared["outputs"][pair["edge"]][group]["files"].items():
                    assert sha256_file(directory / f"{name}.npy") == detail["sha256"]
        uids = config["canary"]["mature_pilot_uids"] if args.canary else config["evaluation_uids"]
        assert len(uids) == (96 if args.canary else 10000)
        settings = dict(mode="canary" if args.canary else "diagnostic", config=config,
            config_sha256=sha256_file(args.config), evaluation_uids=uids, model_pairs=pairs, source_sha256=files,
            prepared_metadata_sha256=sha256_file(metadata_path), batch_size=args.batch_size,
            candidate_chunk=args.candidate_chunk, global_batch_spans=batch_spans(len(uids), args.batch_size),
            execution_overlay=dict(gpu_ids=[0, 1, 2, 3], worker_torch_threads=args.threads,
                serial_edges=True, data_threads=prepared.get("threads"), translator="layer-sized GPU fitting",
                expanded_calibration_cache="CPU memory"),
            aggregation="concatenate all raw scores, then pooled AUC per edge using the Large-ID label source")
        write_json(args.output / "configuration.json", settings)
        summaries = []
        for index, pair in enumerate(pairs):
            started, edge = time.perf_counter(), pair["edge"]
            print(f"Calibrate Large {edge}: original teacher budgets and baselines", flush=True)
            artifact, resource = initial_calibration(args, config, pair)
            if not args.canary:
                directory = args.prepared / edge / "expanded_fit"
                extension = calibrate_expanded(pair, directory, directory / "panel.npy", BUDGETS[4:],
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
            assert len(paths) == (29 if args.canary else 41)
            record = dict(edge=edge, model_pair=pair, rules_sha256=sha256_file(rule_path),
                teacher_budgets=artifact["teacher_budgets"], profile_users=artifact["profile_users"],
                translate_users=artifact["translate_users"], intervals=artifact["intervals"],
                interval_profiles=artifact["interval_profiles"], interval_profile_stats=artifact["interval_profile_stats"],
                numerical=artifact["numerical"], translation_reference=artifact["translation_reference"],
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
            else:
                record.update(**join_quality(config, raw, args.output, edge))
            record["elapsed_seconds"] = time.perf_counter()-started
            summaries.append(record)
            write_json(args.output / f"{edge}_summary.json", record)
            write_json(args.output / "summary.json", dict(status="running", mode=settings["mode"], edges=summaries))
            print(f"Completed Large {edge} in {record['elapsed_seconds']:.1f}s", flush=True)
            del artifact, raw
        write_json(args.output / "summary.json", dict(status="completed", mode=settings["mode"],
            elapsed_seconds=time.perf_counter()-began, edges=summaries))
    except Exception as exc:
        write_json(args.output / "summary.json", dict(status="failed", error=repr(exc),
            elapsed_seconds=time.perf_counter()-began))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/large_unified_auc_10k_20260920.json")
    parser.add_argument("--prepared", type=Path, default=DEFAULT_ROOT / "prepared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--canary-record", type=Path, default=DEFAULT_ROOT / "canary/summary.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-chunk", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--edge", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    worker(args) if args.worker else run(args)
