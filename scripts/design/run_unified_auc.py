#!/usr/bin/env python3
"""One release-snapshot AUC protocol for motivation and shared-read curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"scripts"), str(ROOT/"src")]

from design import bias_read_probe
from design.competitor_data import history_arrays, make_candidate_panel
from design.competitor_models import load_model_pair, sha256_file
from design.run_auc_read_probe import fit_rule, requests, REQUEST_ROOT
from design.run_shared_read_probe import cache_slice, clock, collect_cache, device_parameters, reader_canary, write_json
from design.unified_auc_baselines import calibrate_layer_intervals, fit_translators
from evaluate_yambda500m_foundation_raw import load_histories, load_model
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.baselines.tail_recompute import recompute_tail
from hstu_kvcache.evaluation.binary_metrics import binary_metrics, _roc_auc

TAILS = (32, 64, 128, 256, 512, 1024)
MAP_KS = (1, 2, 3, 4)
TEACHERS = (32, 64, 128, 256)


def take_events(arrays, begin, end, device):
    return tuple(torch.as_tensor(a[begin:end], device=device) for a in arrays[1:4])


def calibration(parent, current, history, pair, config, batch_size, device, canary):
    all_fit = config["calibration_uids"]
    arrays = history_arrays(history, np.asarray(all_fit), pair["cutover"], config["history_length"])
    bank, _ = make_candidate_panel(arrays[1], pair["dataset"]["known_items"])
    users = 32 if canary else 256
    budgets = (8, 16, 32) if canary else TEACHERS
    profile_users = 8 if canary else 32
    source = collect_cache(parent, arrays, users, batch_size, device)
    teacher = collect_cache(current, arrays, users, batch_size, device)
    panel = torch.as_tensor(bank[:users, config["calibration_candidate_indices"]], device=device)
    delta = torch.as_tensor(arrays[4][:users], device=device)
    numerical = reader_canary(current, source, teacher, panel, delta)
    rules, fits = {}, {}
    for count in budgets:
        for conditioned in (False, True):
            name = f"{'personal' if conditioned else 'shared'}_{count}"
            print(f"Calibrate {name}", flush=True)
            rules[name], fits[name] = fit_rule(current, cache_slice(source, 0, count),
                cache_slice(teacher, 0, count), panel[:count], delta[:count],
                conditioned, batch_size, device)
    print("Profile DroidSpeak intervals", flush=True)
    events = take_events(arrays, 0, profile_users, device)
    parent_state = lr.capture_state(parent, *events)
    intervals, profiles, profile_stats = calibrate_layer_intervals(current, parent_state,
        events, panel[:profile_users], delta[:profile_users], max_profile_users=profile_users,
        exact_cache=cache_slice(teacher, 0, profile_users))
    del events, parent_state
    print(f"Fit Translate on {users} users", flush=True)
    translators = fit_translators(source, teacher, current.cfg.num_heads, ks=MAP_KS, ridge=.01)
    translators = {k: type(mapper)(**{name: value.cpu() if torch.is_tensor(value) else value
        for name, value in vars(mapper).items()}) for k, mapper in translators.items()}
    artifact = dict(rules=rules, fitting=fits, calibration_uids=all_fit[:users],
        teacher_budgets=list(budgets), profile_users=profile_users, translate_users=users,
        intervals=intervals, interval_profiles=profiles, interval_profile_stats=profile_stats,
        translators={str(k): vars(mapper) for k, mapper in translators.items()}, numerical=numerical)
    return artifact, translators


def evaluate(parent, current, arrays, uids, rows, artifact, translators, batch_size, chunk, device):
    paths = ["parent", "reuse", "exact", *[f"droid_{k}" for k in range(1, 7)],
             *[f"tail_{n}" for n in TAILS], *[f"translate_{k}" for k in MAP_KS], *artifact["rules"]]
    values = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in paths}
    by_uid = {int(uid): frame.index.to_numpy() for uid, frame in rows.groupby("uid", sort=False)}
    rules = {name: device_parameters(parameters, device) for name, parameters in artifact["rules"].items()}
    checks = []
    total_requests = np.zeros(len(uids), dtype=np.int64)
    started = clock(device)
    for begin in range(0, len(uids), batch_size):
        end = min(begin+batch_size, len(uids))
        events = take_events(arrays, begin, end, device)
        state = lr.capture_state(parent, *events)
        exact = current.compute_kv(*events)
        groups = [by_uid.get(int(uid), np.empty(0, dtype=np.int64)) for uid in uids[begin:end]]
        counts = torch.full((end-begin,), float(state.cache.seq_len), device=device)
        total_requests[begin:end] = [len(indices) for indices in groups]
        maximum = max(map(len, groups), default=0)
        chunks = []
        for offset in range(0, maximum, chunk):
            width = min(chunk, maximum-offset)
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

        def observe(name, model, cache, parameters=None):
            override = bias_read_probe.make_history_override(parameters, counts) if parameters is not None else None
            for panel, delta, indices in chunks:
                prediction, _ = score(model, cache, panel, delta, history_override=override)
                mask = indices >= 0
                values[name][indices[mask]] = prediction.cpu().numpy()[mask]

        observe("parent", parent, state.cache)
        observe("reuse", current, state.cache)
        observe("exact", current, exact)
        for layers, interval in artifact["intervals"].items():
            migrated = lr.recompute_interval(current, state, *events, tuple(interval))
            if begin == 0 and int(layers) == 6:
                torch.testing.assert_close(migrated.cache.k, exact.k, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(migrated.cache.v, exact.v, atol=2e-5, rtol=2e-5)
                checks.append("full_interval_matches_exact_KV")
            observe(f"droid_{layers}", current, migrated.cache)
            del migrated
        for n in TAILS:
            migrated = recompute_tail(current, state.cache, *events, n)
            if begin == 0 and n == 1024:
                torch.testing.assert_close(migrated.k, exact.k, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(migrated.v, exact.v, atol=2e-5, rtol=2e-5)
                checks.append("full_tail_matches_exact_KV")
            observe(f"tail_{n}", current, migrated)
            del migrated
        for k, translator in translators.items():
            migrated = translator.apply(state.cache)
            observe(f"translate_{k}", current, migrated)
            del migrated
        for name, parameters in rules.items():
            observe(name, current, state.cache, parameters)
        del state, exact, events, counts, chunks
        if end % 128 < batch_size or end == len(uids):
            print(f"  {end}/{len(uids)} cache snapshots; {clock(device)-started:.1f}s", flush=True)
    assert sum(total_requests) == len(rows)
    assert all(np.isfinite(x).all() for x in values.values())
    prefix_last = {int(uid): int(stamps[-1]) for uid, stamps in zip(uids, arrays[0], strict=True)}
    raw = rows.assign(prefix_last_timestamp=rows.uid.map(prefix_last), prefix_length=arrays[1].shape[1], **values)
    assert (raw.prefix_last_timestamp < raw.query_timestamp).all()
    return raw, dict(checks=checks, snapshot_users=len(uids), feedback_users=len(by_uid),
        no_feedback_users=len(uids)-len(by_uid), requests=len(rows),
        elapsed_seconds=clock(device)-started, per_user_requests=total_requests.tolist())


def join_quality(raw, output, edge):
    path = output/f"{edge}_scores.parquet"
    raw.to_parquet(path, index=False)
    digest = sha256_file(path)
    write_json(output/f"{edge}_scores.seal.json", dict(sha256=digest, rows=len(raw), labels_joined=False))
    labels = pq.read_table(REQUEST_ROOT/"requests_quality.parquet", filters=[
        ("request_id", "in", raw.request_id.tolist())], columns=["request_id", "uid", "query_timestamp", "item_idx", "label"]).to_pandas()
    full = raw.merge(labels, on=["request_id", "uid", "query_timestamp", "item_idx"], how="left", validate="one_to_one")
    assert len(full) == len(raw) and full.label.isin([0, 1]).all()
    full.to_parquet(output/f"{edge}_quality.parquet", index=False)
    paths = [column for column in raw if column not in ("request_id", "uid", "query_timestamp", "item_idx", "prefix_last_timestamp", "prefix_length")]
    metrics = {name: binary_metrics(full.label.to_numpy(), full[name].to_numpy()) for name in paths}
    # AUC depends on ranks. Rank logits directly to avoid artificial sigmoid ties.
    for name in paths:
        metrics[name]["ROC_AUC"] = _roc_auc(full.label.to_numpy(), full[name].to_numpy(dtype=np.float64))
    auc = {name: m["ROC_AUC"] for name, m in metrics.items()}
    gap = None if auc["exact"] is None else auc["exact"]-auc["reuse"]
    improvement = None if auc["exact"] is None else auc["exact"]-auc["parent"]
    quality = {name: dict(auc=auc[name],
        delta_auc_pp=None if auc[name] is None else 100*(auc[name]-auc["reuse"]),
        auc_gap_recovery=(auc[name]-auc["reuse"])/gap if gap is not None and gap>1e-4 else None,
        metrics=metrics[name]) for name in paths}
    return dict(quality=quality, positives=int(full.label.sum()), negatives=int((full.label==0).sum()),
        exact_minus_reuse_auc=gap, current_minus_parent_auc=improvement,
        update_benefit_lost=gap/improvement if improvement is not None and improvement>1e-4 else None,
        scores_sha256=digest, quality_sha256=sha256_file(output/f"{edge}_quality.parquet"))


@torch.inference_mode()
def run(args):
    config = json.loads(args.config.read_text())
    assert config["edges"] == [0, 1, 2, 3] and config["history_length"] == 1024
    fit = config["calibration_uids"]
    evaluation = config["canary"]["mature_pilot_uids"][:16] if args.canary else config["evaluation_uids"]
    assert len(fit) == 256 and len(config["evaluation_uids"]) == 10000
    assert not set(fit) & set(evaluation)
    assert not set(config["pilot_uids"]) & set(config["evaluation_uids"])
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    args.output.mkdir(parents=True)
    source_files = [Path(__file__), ROOT/"scripts/design/unified_auc_baselines.py",
        ROOT/"scripts/design/run_auc_read_probe.py", ROOT/"scripts/design/bias_read_probe.py",
        ROOT/"scripts/design/run_shared_read_probe.py", ROOT/"scripts/design/competitor_data.py",
        ROOT/"scripts/design/competitor_models.py", ROOT/"scripts/evaluate_yambda500m_foundation_raw.py",
        ROOT/"src/hstu_kvcache/data/yambda_history.py", ROOT/"src/hstu_kvcache/training/foundation.py",
        ROOT/"src/hstu_kvcache/evaluation/binary_metrics.py", ROOT/"src/hstu_kvcache/adaptation/reader.py",
        *sorted((ROOT/"src/hstu_kvcache/models").glob("*.py")),
        *sorted((ROOT/"src/hstu_kvcache/baselines").glob("*/core.py"))]
    for name in ("requests_fidelity", "requests_quality"):
        record = config["request_manifest"][name]
        assert sha256_file(ROOT/record["path"]) == record["sha256"]
    assert sha256_file(ROOT/config["model_chain"]) == config["model_chain_sha256"]
    settings = dict(config=config, config_sha256=sha256_file(args.config),
        mode="canary" if args.canary else "diagnostic", evaluation_uids=evaluation,
        batch_size=args.batch_size, candidate_chunk=args.candidate_chunk,
        torch_version=torch.__version__, device=str(device),
        source_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in source_files})
    write_json(args.output/"configuration.json", settings)
    started = time.perf_counter()
    try:
        check = bias_read_probe.self_check()
        edges = config["edges"][:1] if args.canary else config["edges"]
        pairs = [load_model_pair("medium", i, verify_hashes=True) for i in edges]
        assert all(p["admission"]["reuse_eligible"] for p in pairs)
        data = pairs[0]["dataset"]
        history = load_histories(fit+evaluation, dataset_path=Path(data["manifest"]),
            known_vocab_size=data["known_items"], oov_buckets=data["oov_buckets"],
            start_timestamp=min(p["cutover"] for p in pairs)-1,
            end_timestamp=max(p["cutover"] for p in pairs)+1, max_history=1024, threads=args.threads)
        history_seconds = time.perf_counter()-started
        print(f"Histories loaded in {history_seconds:.1f}s", flush=True)
        summaries = []
        for pair in pairs:
            edge = pair["edge"]
            began = clock(device)
            print(f"Begin {edge}", flush=True)
            parent, payload = load_model(Path(pair["parent"]["checkpoint"]), device)
            assert payload["config"] == pair["config"]
            del payload
            current, payload = load_model(Path(pair["current"]["checkpoint"]), device)
            assert payload["config"] == pair["config"]
            del payload
            artifact, translators = calibration(parent, current, history, pair, config, args.batch_size, device, args.canary)
            artifact.update(model_binding={r:pair[r]["checkpoint_sha256"] for r in ("parent", "current")},
                            config_sha256=settings["config_sha256"])
            rule_path = args.output/f"{edge}_calibration.pt"
            torch.save(artifact, rule_path)
            arrays = history_arrays(history, np.asarray(evaluation), pair["cutover"], 1024)
            frame = requests(evaluation, pair["cutover"])
            raw, execution = evaluate(parent, current, arrays, evaluation, frame, artifact,
                translators, args.batch_size, args.candidate_chunk, device)
            record = dict(edge=edge, model_pair=pair, rules_sha256=sha256_file(rule_path),
                teacher_budgets=artifact["teacher_budgets"], profile_users=artifact["profile_users"],
                translate_users=artifact["translate_users"], intervals=artifact["intervals"],
                interval_profiles=artifact["interval_profiles"], numerical=artifact["numerical"],
                **execution, **join_quality(raw, args.output, edge))
            record["elapsed_seconds"] = clock(device)-began
            summaries.append(record)
            write_json(args.output/f"{edge}_summary.json", record)
            write_json(args.output/"summary.json", dict(status="running", mode=settings["mode"], edges=summaries))
            print(json.dumps(dict(edge=edge, elapsed_seconds=record["elapsed_seconds"], quality=record["quality"])), flush=True)
            del parent, current, artifact, translators, arrays, raw, frame
        write_json(args.output/"summary.json", dict(status="completed", mode=settings["mode"],
            self_check=check, history_seconds=history_seconds, elapsed_seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1<<30), edges=summaries))
    except Exception as exc:
        write_json(args.output/"summary.json", dict(status="failed", error=repr(exc),
            elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/insight/unified_auc_10k_20260920.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-chunk", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
