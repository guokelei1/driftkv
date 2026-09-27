#!/usr/bin/env python3
"""Add 64-teacher KV translation to the existing four-edge AUC preview."""

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

from design.competitor_models import load_model_pair, sha256_file
from design.run_shared_read_probe import collect_cache, write_json
from design.run_unified_auc import evaluate as original_evaluate
from design.run_unified_auc import join_quality, requests, take_events
from design.run_unified_auc_parallel import (
    REQUEST_COLUMNS, batch_spans, load_pair, prepared_arrays, setup_gpu, source_hashes,
)
from design.unified_auc_baselines import fit_translators
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.baselines.kv_translate import KVTranslator

DEFAULT_ROOT = ROOT / "results/insight/translation64_auc_preview_20260920"
META_COLUMNS = [*REQUEST_COLUMNS, "prefix_last_timestamp", "prefix_length"]
MAP_NAMES = [f"translate64_{k}" for k in range(1, 5)]


@torch.inference_mode()
def fit_maps(args, config, pair):
    device = setup_gpu(0, args.threads)
    prepared = ROOT / config["edge_sources"][pair["edge"]]["prepared_root"]
    arrays = prepared_arrays(prepared, pair["edge"], "fit", config["calibration_uids"])
    parent, current = load_pair(pair, device)
    started = time.perf_counter()
    source = collect_cache(parent, arrays, 64, 64, device)
    teacher = collect_cache(current, arrays, 64, 64, device)
    maps = fit_translators(source, teacher, current.cfg.num_heads, ks=(1, 2, 3, 4), ridge=.01)
    artifact = dict(calibration_uids=config["calibration_uids"], translate_users=64,
        source_layer_counts=[1, 2, 3, 4], ridge=.01,
        candidate_bank_used=False, fitting="unchanged select_source_layers and full-width affine ridge",
        config_sha256=sha256_file(args.config),
        model_binding={role: pair[role]["checkpoint_sha256"] for role in ("parent", "current")},
        translators={str(k): {name: value.cpu() if torch.is_tensor(value) else value
            for name, value in vars(mapper).items()} for k, mapper in maps.items()})
    resource = dict(seconds=time.perf_counter()-started, cache_batch_size=64,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30))
    del parent, current, source, teacher, maps, arrays
    gc.collect()
    torch.cuda.empty_cache()
    return artifact, resource


@torch.inference_mode()
def evaluate_translation(parent, current, arrays, uids, rows, translators, batch_size, chunk, device):
    """Original evaluate's batch/chunk/observe loop, restricted to five paths."""
    values = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in ["reuse", *MAP_NAMES]}
    by_uid = {int(uid): frame.index.to_numpy() for uid, frame in rows.groupby("uid", sort=False)}
    counts = np.zeros(len(uids), dtype=np.int64)
    started = time.perf_counter()
    for begin in range(0, len(uids), batch_size):
        end = min(begin+batch_size, len(uids))
        source = parent.compute_kv(*take_events(arrays, begin, end, device))
        groups = [by_uid.get(int(uid), np.empty(0, dtype=np.int64)) for uid in uids[begin:end]]
        counts[begin:end] = [len(group) for group in groups]
        chunks = []
        for offset in range(0, max(map(len, groups), default=0), chunk):
            width = min(chunk, max(map(len, groups))-offset)
            item = np.ones((end-begin, width), dtype=np.int64)
            delta = np.ones((end-begin, width), dtype=np.float32)
            indices = np.full((end-begin, width), -1, dtype=np.int64)
            for local, group in enumerate(groups):
                take = group[offset:offset+width]
                indices[local, :len(take)] = take
                if len(take):
                    item[local, :len(take)] = rows.loc[take, "item_idx"].to_numpy(dtype=np.int64)
                    delta[local, :len(take)] = rows.loc[take, "query_timestamp"].to_numpy()-arrays[0][begin+local, -1]
            chunks.append((torch.as_tensor(item, device=device), torch.as_tensor(delta, device=device), indices))

        def observe(name, cache):
            for panel, delta, indices in chunks:
                prediction, _ = score(current, cache, panel, delta)
                mask = indices >= 0
                values[name][indices[mask]] = prediction.cpu().numpy()[mask]

        observe("reuse", source)
        for k, mapper in translators.items():
            migrated = mapper.apply(source)
            observe(f"translate64_{k}", migrated)
            del migrated
        del source, chunks
        if end % 512 < batch_size or end == len(uids):
            print(f"  {end}/{len(uids)} snapshots; {time.perf_counter()-started:.1f}s", flush=True)
    assert sum(counts) == len(rows) and all(np.isfinite(value).all() for value in values.values())
    last = {int(uid): int(stamps[-1]) for uid, stamps in zip(uids, arrays[0], strict=True)}
    raw = rows.assign(prefix_last_timestamp=rows.uid.map(last), prefix_length=1024, **values)
    assert (raw.prefix_last_timestamp < raw.query_timestamp).all()
    return raw, dict(per_user_requests=counts.tolist(), requests=len(rows),
                     elapsed_seconds=time.perf_counter()-started)


@torch.inference_mode()
def worker(args):
    settings = json.loads((args.output / "configuration.json").read_text())
    pair = settings["model_pairs"][args.edge]
    edge, uids = pair["edge"], settings["evaluation_uids"]
    begin, end = (0, len(uids)) if args.reference else batch_spans(len(uids), args.batch_size)[args.rank]
    source = settings["config"]["edge_sources"][edge]
    arrays = tuple(array[begin:end] for array in prepared_arrays(ROOT / source["prepared_root"], edge,
        "pilot" if settings["mode"] == "canary" else "evaluation", uids))
    uids = uids[begin:end]
    rows = requests(uids, pair["cutover"])
    rule_path = args.output / f"{edge}_calibration.pt"
    artifact = torch.load(rule_path, map_location="cpu", weights_only=False)
    assert artifact["config_sha256"] == settings["config_sha256"]
    assert artifact["model_binding"] == {role: pair[role]["checkpoint_sha256"] for role in ("parent", "current")}
    translators = {int(k): KVTranslator(**value) for k, value in artifact["translators"].items()}
    directory = args.output / edge / ("reference" if args.reference else f"rank{args.rank}")
    device = setup_gpu(args.rank, args.threads)
    parent, current = load_pair(pair, device)
    if args.reference:
        # Retain all original baseline intervals solely to run the old evaluator.
        base_artifact = torch.load(ROOT / source["base_result_root"] / f"{edge}_calibration.pt",
                                   map_location="cpu", weights_only=False)
        base_artifact["rules"] = {}
        raw, execution = original_evaluate(parent, current, arrays, uids, rows, base_artifact,
            translators, args.batch_size, args.candidate_chunk, device)
        raw = raw[[*META_COLUMNS, "reuse", *[f"translate_{k}" for k in range(1, 5)]]].rename(
            columns={f"translate_{k}": f"translate64_{k}" for k in range(1, 5)})
    else:
        raw, execution = evaluate_translation(parent, current, arrays, uids, rows, translators,
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
                "--rank", str(rank), "--output", str(args.output), "--config", str(args.config),
                "--batch-size", str(args.batch_size), "--candidate-chunk", str(args.candidate_chunk),
                "--threads", str(args.threads)]
            if reference:
                command.append("--reference")
            env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
            children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        for child, _ in children:
            if child.wait() != 0:
                raise RuntimeError(f"Translation worker failed; see {args.output / edge}")
    finally:
        for child, log in children:
            if child.poll() is None:
                child.terminate()
            log.close()
        for child, _ in children:
            child.wait()


def concatenate(args, edge, pair, uids):
    directories = [args.output / edge / f"rank{rank}" for rank in range(4)]
    workers = [json.loads((directory / "execution.json").read_text()) for directory in directories]
    assert all(record["status"] == "completed" for record in workers)
    assert [uid for record in workers for uid in record["evaluation_uids"]] == uids
    for directory, record in zip(directories, workers, strict=True):
        assert record["scores_sha256"] == sha256_file(directory / "scores.parquet")
        assert record["rules_sha256"] == sha256_file(args.output / f"{edge}_calibration.pt")
    raw = pd.concat([pd.read_parquet(directory / "scores.parquet") for directory in directories], ignore_index=True)
    raw = raw.sort_values(["uid", "query_timestamp", "item_idx", "request_id"]).reset_index(drop=True)
    assert raw[list(REQUEST_COLUMNS)].equals(requests(uids, pair["cutover"]))
    assert not raw.request_id.duplicated().any()
    counts = raw.groupby("uid").size().reindex(uids, fill_value=0).to_numpy(dtype=np.int64)
    assert np.array_equal(counts, [n for record in workers for n in record["per_user_requests"]])
    return raw, dict(snapshot_users=len(uids), feedback_users=int((counts > 0).sum()),
        no_feedback_users=int((counts == 0).sum()), requests=len(raw),
        per_user_requests=counts.tolist(), workers=workers)


def run(args):
    config = json.loads(args.config.read_text())
    assert config["edges"] == [0, 2, 3, 4]
    assert len(config["calibration_uids"]) == 64 and len(config["evaluation_uids"]) == 10000
    assert not set(config["calibration_uids"]) & set(config["evaluation_uids"])
    assert args.batch_size == (16 if args.canary else 64) and args.candidate_chunk == 32
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    began = time.perf_counter()
    try:
        files = source_hashes()
        for path in (Path(__file__), ROOT / "scripts/insight_one_locality/common.py"):
            files[str(path.relative_to(ROOT))] = sha256_file(path)
        if not args.canary:
            canary = json.loads(args.canary_record.read_text())
            checked = json.loads((args.canary_record.parent / "configuration.json").read_text())
            assert canary["status"] == "completed" and canary["mode"] == "canary"
            assert checked["config_sha256"] == sha256_file(args.config) and checked["source_sha256"] == files
        pairs = [load_model_pair("medium", index, verify_hashes=True)
                 for index in (config["edges"][:1] if args.canary else config["edges"])]
        assert all(pair["admission"]["reuse_eligible"] for pair in pairs)
        base_records = {}
        for pair in pairs:
            edge = pair["edge"]
            source = config["edge_sources"][edge]
            prepared_root = ROOT / source["prepared_root"]
            base_root = ROOT / source["base_result_root"]
            assert sha256_file(prepared_root / "metadata.json") == source["prepared_metadata_sha256"]
            assert sha256_file(base_root / "summary.json") == source["base_summary_sha256"]
            prepared = json.loads((prepared_root / "metadata.json").read_text())
            summary = json.loads((base_root / "summary.json").read_text())
            assert prepared["status"] == summary["status"] == "completed"
            base_record = next(record for record in summary["edges"] if record["edge"] == edge)
            base_records[edge] = base_record
            assert sha256_file(base_root / f"{edge}_scores.parquet") == base_record["scores_sha256"]
            assert sha256_file(base_root / f"{edge}_calibration.pt") == base_record["rules_sha256"]
            for group in ("fit", "pilot" if args.canary else "evaluation"):
                for name, details in prepared["outputs"][edge][group]["files"].items():
                    assert sha256_file(prepared_root / edge / group / f"{name}.npy") == details["sha256"]
        for name in ("requests_fidelity", "requests_quality"):
            record = config["request_manifest"][name]
            assert sha256_file(ROOT / record["path"]) == record["sha256"]
        assert sha256_file(ROOT / config["model_chain"]) == config["model_chain_sha256"]
        uids = config["canary"]["mature_pilot_uids"] if args.canary else config["evaluation_uids"]
        assert len(uids) == (96 if args.canary else 10000)
        settings = dict(mode="canary" if args.canary else "diagnostic", config=config,
            config_sha256=sha256_file(args.config), evaluation_uids=uids, model_pairs=pairs, source_sha256=files,
            batch_size=args.batch_size, candidate_chunk=args.candidate_chunk,
            execution_overlay=dict(gpu_ids=[0, 1, 2, 3], worker_torch_threads=args.threads,
                serial_edges=True, cache_calibration_batch=64, candidate_bank_used=False),
            aggregation="new translation logits plus unchanged old anchors; pooled AUC once per edge")
        write_json(args.output / "configuration.json", settings)
        summaries = []
        for index, pair in enumerate(pairs):
            started, edge = time.perf_counter(), pair["edge"]
            print(f"Fit {edge}: 64 teacher users, source layers1/2/3/4", flush=True)
            artifact, resource = fit_maps(args, config, pair)
            rule_path = args.output / f"{edge}_calibration.pt"
            torch.save(artifact, rule_path)
            if args.canary:
                launch(args, index, edge, reference=True)
            launch(args, index, edge)
            raw, execution = concatenate(args, edge, pair, uids)
            fresh_path = args.output / f"{edge}_new_scores.parquet"
            raw.to_parquet(fresh_path, index=False)
            fresh_hash = sha256_file(fresh_path)
            write_json(args.output / f"{edge}_new_scores.seal.json", dict(sha256=fresh_hash, rows=len(raw), labels_joined=False))
            source = config["edge_sources"][edge]
            record = dict(edge=edge, model_pair=pair, translate_users=64, calibration_uids=config["calibration_uids"],
                source_layer_counts=[1, 2, 3, 4], candidate_bank_used=False,
                rules_sha256=sha256_file(rule_path), new_scores_sha256=fresh_hash,
                base_result=source["base_result_root"], base_scores_sha256=base_records[edge]["scores_sha256"],
                calibration_seconds=resource["seconds"], calibration_peak_allocated_gib=resource["peak_allocated_gib"],
                **execution)
            if args.canary:
                reference = pd.read_parquet(args.output / edge / "reference" / "scores.parquet")
                assert raw[META_COLUMNS].equals(reference[META_COLUMNS])
                differences = {}
                for name in ["reuse", *MAP_NAMES]:
                    np.testing.assert_allclose(raw[name], reference[name], atol=2e-5, rtol=2e-5)
                    differences[name] = float(np.max(np.abs(raw[name].to_numpy()-reference[name].to_numpy())))
                record.update(maximum_absolute_logit_difference=max(differences.values()),
                    per_path_logit_difference=differences, labels_read=False)
            else:
                old = pd.read_parquet(ROOT / source["base_result_root"] / f"{edge}_scores.parquet")
                assert old[META_COLUMNS].equals(raw[META_COLUMNS])
                assert base_records[edge]["per_user_requests"] == execution["per_user_requests"]
                np.testing.assert_allclose(raw.reuse, old.reuse, atol=2e-5, rtol=2e-5)
                record["maximum_reuse_logit_difference"] = float(np.max(np.abs(raw.reuse.to_numpy()-old.reuse.to_numpy())))
                merged = old[[*META_COLUMNS, "parent", "reuse", "exact"]].copy()
                for name in MAP_NAMES:
                    merged[name] = raw[name].to_numpy()
                record.update(**join_quality(merged, args.output, edge))
                for anchor in ("parent", "reuse", "exact"):
                    assert record["quality"][anchor]["auc"] == base_records[edge]["quality"][anchor]["auc"]
            record["elapsed_seconds"] = time.perf_counter()-started
            summaries.append(record)
            write_json(args.output / f"{edge}_summary.json", record)
            write_json(args.output / "summary.json", dict(status="running", mode=settings["mode"], edges=summaries))
            print(f"Completed {edge} in {record['elapsed_seconds']:.1f}s", flush=True)
            del artifact, raw
        write_json(args.output / "summary.json", dict(status="completed", mode=settings["mode"],
            elapsed_seconds=time.perf_counter()-began, edges=summaries))
    except Exception as exc:
        write_json(args.output / "summary.json", dict(status="failed", error=repr(exc),
            elapsed_seconds=time.perf_counter()-began))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/translation64_auc_preview_20260920.json")
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
