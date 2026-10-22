#!/usr/bin/env python3
"""Two independent GPU workers for the Max adjacent-edge baseline study.

prepare/fit/probe are bounded readiness work. evaluate is the explicit full
population command. Neither training backbones nor carrying state across edges
is supported. CUDA_VISIBLE_DEVICES selects the two physical GPUs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow.parquet as pq
import torch

from max_baselines_gpu01_2026_10 import baselines as b
from max_baselines_gpu01_2026_10.fast_baselines import NativeGraphs, evaluate_batched, evaluate_checkpointed
from design_one.prepare_history import load_histories_semijoin
from evaluate_yambda500m_foundation_raw import load_model
from read_correction_2026_09.cost import CostModel


def sources():
    paths = list(Path(__file__).parent.glob("*.py")) + list((ROOT / "src").rglob("*.py"))
    # Include the retained numerical fitting, data and accounting dependencies.
    for folder in ("read_correction_2026_09", "read_correction_v4", "read_correction_v5",
                   "selective_recompute_2026_09", "unified_reuse_2026_09", "design_one",
                   "implementation_2026_10"):
        paths.extend((ROOT / "scripts" / folder).rglob("*.py"))
    paths.append(ROOT / "scripts/evaluate_yambda500m_foundation_raw.py")
    return {str(p.relative_to(ROOT)): b.sha256(p) for p in sorted(set(paths))}


def binding_at(out):
    binding = json.loads((out / "configuration.json").read_text())
    assert binding["execution_source_sha256"] == sources(), "frozen execution source changed"
    assert binding["requests_sha256"] == b.sha256(out / "requests.parquet"), "requests changed"
    assert binding["dataset_sha256"] == b.sha256(b.dataset_path("max")), "dataset changed"
    assert binding["launch_config_sha256"] == b.sha256(Path(binding["launch_config"])), "config changed"
    return binding


def prepare(args):
    cfg = json.loads(args.config.read_text())
    assert cfg["scale"] == "max" and not cfg.get("completed_fit_sources")
    if args.resume:
        binding_at(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    binding = b.prepare_inputs(cfg, args.output)
    paths = [ROOT / f"results/unified_training_2026_09/max/seed17/checkpoints/v{v}/checkpoint_100.pt"
             for v in (cfg["edge"] - 1, cfg["edge"])]
    configs = [torch.load(p, map_location="cpu", weights_only=False, mmap=True)["config"] for p in paths]
    assert configs[0] == configs[1]
    actual = configs[0]
    assert (actual["num_layers"], actual["hidden_size"], actual["num_heads"], actual["max_seq_len"]) == (16, 320, 10, 1024)
    binding.update(launch_config=str(args.config.resolve()), launch_config_sha256=b.sha256(args.config),
                   execution_source_sha256=sources(), model_config=actual,
                   checkpoint_sha256={str(p): b.sha256(p) for p in paths},
                   physical_gpus=cfg["physical_gpus"], preparation_seconds=time.perf_counter()-started)
    b.write_json(args.output / "configuration.json", binding)
    print(json.dumps(dict(status="prepared", quality_users=cfg["quality_users"], requests=binding["requests"],
                          seconds=binding["preparation_seconds"])), flush=True)


def runtime(out, rank, selected):
    binding = binding_at(out)
    cfg = binding["settings"]
    if cfg.get("cuda_allocator_config"):
        os.environ["PYTORCH_ALLOC_CONF"]=cfg["cuda_allocator_config"]
    os.environ["EVOKV_ATTENTION_BACKEND"] = cfg["attention_backend"]
    torch.set_num_threads(cfg["torch_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    began = time.perf_counter()
    ds_path = b.dataset_path("max")
    ds = json.loads(ds_path.read_text())
    histories = load_histories_semijoin(selected, dataset_path=ds_path, known_vocab_size=ds["foundation_items"],
        oov_buckets=256, start_timestamp=cfg["release_day"]*b.DAY, end_timestamp=cfg["end_day"]*b.DAY,
        max_history=1024, threads=cfg["history_threads"])
    by_user = defaultdict(lambda: defaultdict(list))
    for row in pq.read_table(out / "requests.parquet", filters=[("uid", "in", selected)]).to_pylist():
        by_user[row["uid"]][row["query_timestamp"]].append(row)
    input_seconds = time.perf_counter() - began
    began = time.perf_counter()
    paths = list(binding["checkpoint_sha256"])
    for p in paths:
        assert b.sha256(Path(p)) == binding["checkpoint_sha256"][p], "checkpoint changed"
    parent, payload = load_model(Path(paths[0]), device)
    del payload
    current, payload = load_model(Path(paths[1]), device)
    del payload
    parent.eval().requires_grad_(False)
    current.eval().requires_grad_(False)
    cm = CostModel(320, 16, 10, cfg["attention_backend"])
    return binding, cfg, histories, by_user, parent, current, cm, device, dict(
        input_seconds=input_seconds, model_load_and_hash_seconds=time.perf_counter()-began)


def fit(args):
    started = time.perf_counter()
    binding = binding_at(args.output)
    cfg, roles = binding["settings"], binding["roles"]
    # Q512 is the largest joint task. Balance it against the two larger H fits.
    assignment = [("query_only", 512), ("history_conditioned", 64)] if args.rank == 0 else [
        ("query_only", 128), ("query_only", 256), ("history_conditioned", 128), ("history_conditioned", 256)]
    complete = all((args.output / f"{method}_fit{n}/summary.json").exists() for method,n in assignment)
    if complete and (args.output / "partial_budgets.json").exists():
        for method,n in assignment:
            folder=args.output/f"{method}_fit{n}"
            saved=json.loads((folder/"summary.json").read_text())
            assert saved["fit_users"]==roles["fit"][:n] and saved["validation_users"]==roles["validation"]
            assert saved["weights_sha256"]==b.sha256(folder/"calibration.pt")
        print(json.dumps(dict(status="completed_fits_validated",rank=args.rank)),flush=True)
        return
    train_count = max(n for _, n in assignment)
    selected = roles["fit"][:train_count] + roles["validation"]
    _, cfg, histories, _, parent, current, cm, device, timings = runtime(args.output, args.rank, selected)
    began = time.perf_counter()
    rows, counts, transfer = b.capture(parent, current, histories, selected, cfg, device)
    timings["capture_seconds"] = time.perf_counter()-began
    profile_path = args.output / "partial_budgets.json"
    if args.rank == 0 and not profile_path.exists():
        for uid in roles["fit"][:cfg["profile_users"]]:
            events = histories.rows[uid]
            stop = int(np.searchsorted(events[0], cfg["release_day"]*b.DAY, side="left"))
            rows[uid]["raw"] = b.raw_tensors(tuple(x[max(0, stop-1024):stop] for x in events), device)
        began = time.perf_counter()
        gains, flops = b.profile_layers(current, rows, roles["fit"][:cfg["profile_users"]], cm, cfg, device)
        b.write_json(profile_path, dict(layer_profile_gains=gains, profile_flops=flops))
        timings["layer_profile_seconds"] = time.perf_counter()-began
    timings["fits"] = []
    for method, budget in assignment:
        subcfg = dict(cfg, query_fit_users=[budget] if method == "query_only" else [],
                      history_fit_users=[budget] if method == "history_conditioned" else [])
        torch.manual_seed(cfg["seed"])
        began = time.perf_counter()
        b.fit_learned(current, rows, roles, cm, subcfg, device, args.output)
        torch.cuda.synchronize(device)
        timings["fits"].append(dict(method=method, users=budget, seconds=time.perf_counter()-began))
        print(json.dumps(dict(status="fit_point_complete", rank=args.rank, **timings["fits"][-1])), flush=True)
    b.write_json(args.output / f"fit_worker_{args.rank}.json", dict(status="complete", rank=args.rank,
        timings=timings, seconds=time.perf_counter()-started, capture_transfer_bytes=transfer,
        capture_history_counts=dict(counts), peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))


def policies_at(out, current, cm, cfg, roles, device, shapes):
    saved = json.loads((out / "partial_budgets.json").read_text())
    policies, denominator = b.choose_repairs(shapes, cm, saved["layer_profile_gains"], saved["profile_flops"], cfg)
    policies += b.reuse_learned(out, roles, dict(cfg,write_fit_source_records=False), device, out)
    return policies, denominator


def difference(actual, wanted):
    key = lambda r: (r["method"], r["uid"], r["query_timestamp"], r["raw_item_id"])
    a, w = {key(r): r["score"] for r in actual}, {key(r): r["score"] for r in wanted}
    assert len(a) == len(actual) and a.keys() == w.keys()
    maximum = max((abs(a[k]-w[k]) for k in a), default=0.)
    np.testing.assert_allclose(list(a.values()), [w[k] for k in a], atol=3e-5, rtol=3e-5)
    return dict(scores=len(a), max_absolute_logit_error=maximum, atol=3e-5, rtol=3e-5)


def sampled(roles, users):
    return sorted(roles["quality"], key=lambda uid: hashlib.sha256(
        f"max_speed_20261022/17/{uid}".encode()).digest())[:users]


def probe(args):
    binding = binding_at(args.output)
    full = sampled(binding["roles"], args.users)
    selected = full if args.single_gpu else full[args.rank::2]
    _, cfg, histories, by_user, parent, current, cm, device, timings = runtime(args.output, args.rank, selected)
    shapes = b.event_shapes(histories, by_user, selected, cfg)
    policies, _ = policies_at(args.output, current, cm, cfg, binding["roles"], device, shapes)
    target = args.probe_output
    target.mkdir(parents=True, exist_ok=True)
    # Serial comparison includes behavior-before-query and all candidate policies.
    verify = [uid for uid in selected if by_user.get(uid)][:args.verify_users]
    began = time.perf_counter()
    reference, _ = b.evaluate(parent, current, histories, by_user, verify, policies, cfg, device, target)
    timings["serial_reference_seconds"] = time.perf_counter()-began
    engine = NativeGraphs()
    actual, _ = evaluate_batched(parent, current, histories, by_user, verify, policies, cfg, device,
                               graphs=engine, stop_after_last_query=True)
    parity = difference(actual, reference)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    variants = [(128,16), (64,16), (256,16), (128,32)] if args.sweep else [
        (args.append_band_size or cfg["append_band_size"], args.query_batch or cfg["query_batch"])]
    records = []
    sweep_reference = None
    for band, tile in variants:
        engine = NativeGraphs()
        tuned = dict(cfg, append_band_size=band, query_batch=tile, repair_query_batch=16)
        torch.cuda.reset_peak_memory_stats(device)
        actual, warm = evaluate_batched(parent, current, histories, by_user, selected, policies, tuned, device,
                                       graphs=engine, stop_after_last_query=True)
        if sweep_reference is None:
            sweep_reference = actual
        check = difference(actual, sweep_reference)
        repeats = []
        for repeat in range(args.repeats):
            actual, stats = evaluate_batched(parent, current, histories, by_user, selected, policies, tuned, device,
                                            graphs=engine, stop_after_last_query=True)
            difference(actual, sweep_reference)
            repeats.append(stats["seconds"])
        record = dict(append_band_size=band, query_batch=tile, repair_query_batch=16, users=len(selected),
            feedback_events=stats["stats"]["feedback_events"], cold_seconds=warm["seconds"], warm_seconds=repeats,
            graphs=engine.describe(), equivalence=check, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        records.append(record)
        b.write_json(target / f"worker_{args.rank}.json", dict(status="running", rank=args.rank, records=records))
        print(json.dumps(dict(status="probe_variant_complete", rank=args.rank, **record)), flush=True)
        del engine, actual
        gc.collect()
        torch.cuda.empty_cache()
    b.write_json(target / f"worker_{args.rank}.json", dict(status="complete", rank=args.rank, users=selected,
        records=records, serial_equivalence=parity, timings=timings, total_population=len(binding["roles"]["quality"]),
        population_run_started=False, selector_candidates=16, gpu=torch.cuda.get_device_name(device),
        scope="Simultaneous two-GPU bounded sample, execution estimate only; not formal paper timing"))


def quality(args):
    started = time.perf_counter()
    binding = binding_at(args.output)
    cfg, roles = binding["settings"], binding["roles"]
    size = cfg["evaluation_checkpoint_users"]
    offsets = list(range(0, len(roles["quality"]), size))[args.rank::2]
    selected = [uid for offset in offsets for uid in roles["quality"][offset:offset+size]]
    _, cfg, histories, by_user, parent, current, cm, device, timings = runtime(args.output, args.rank, selected)
    shapes = b.event_shapes(histories, by_user, selected, cfg)
    b.write_json(args.output / f"event_shapes_worker_{args.rank}.json", {
        name: [dict(shape=list(key) if isinstance(key, tuple) else [key], count=value) for key,value in counts.items()]
        for name,counts in shapes.items()})
    policies, _ = policies_at(args.output, current, cm, cfg, roles, device, shapes)
    # Population-local cost percentages are derived later from the merged ledger.
    canonical = [{k:v for k,v in p.items() if k not in ("modules", "cost_percent")} for p in policies]
    identity = dict(binding_sha256=b.sha256(args.output / "configuration.json"), policies=canonical)
    def progress(value):
        print(json.dumps(dict(status="quality_block", rank=args.rank, **value)), flush=True)
        b.write_json(args.output / f"progress_worker_{args.rank}.json", value)
    _, stats = evaluate_checkpointed(parent, current, histories, by_user, roles["quality"], policies, cfg, device,
        args.output, identity=identity, progress=progress, assigned_offsets=set(offsets), retain_scores=False)
    b.write_json(args.output / f"quality_worker_{args.rank}.json", dict(status="complete", timings=timings,
        seconds=time.perf_counter()-started, execution=stats, offsets=offsets,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))


def aggregate(args):
    binding = binding_at(args.output)
    cfg, roles = binding["settings"], binding["roles"]
    shapes = {name: Counter() for name in ("first", "initial", "appends", "queries")}
    workers = [json.loads((args.output / f"quality_worker_{rank}.json").read_text()) for rank in range(2)]
    assert workers[0]["execution"]["fingerprint"]==workers[1]["execution"]["fingerprint"]
    for rank in range(2):
        assert workers[rank]["status"] == "complete"
        raw = json.loads((args.output / f"event_shapes_worker_{rank}.json").read_text())
        for name, rows in raw.items():
            shapes[name].update({row["shape"][0] if name == "initial" else tuple(row["shape"]): row["count"] for row in rows})
    assert sum(shapes["initial"].values()) == len(roles["quality"])
    cm = CostModel(320,16,10,cfg["attention_backend"])
    policies, denominator = policies_at(args.output, None, cm, cfg, roles, torch.device("cpu"), shapes)
    scores = []
    expected_offsets = list(range(0,len(roles["quality"]),cfg["evaluation_checkpoint_users"]))
    assert sorted(x for w in workers for x in w["offsets"]) == expected_offsets
    for offset in expected_offsets:
        path = args.output / "quality_shards" / f"part_{offset:06d}.parquet"
        marker = json.loads(path.with_suffix(".json").read_text())
        assert marker["fingerprint"]==workers[0]["execution"]["fingerprint"]
        assert marker["scores_sha256"] == b.sha256(path)
        block = roles["quality"][offset:offset+cfg["evaluation_checkpoint_users"]]
        assert marker["uids_sha256"] == hashlib.sha256(json.dumps(block).encode()).hexdigest()
        scores.extend(pq.read_table(path).to_pylist())
    observed = {p["tag"]: Counter(candidate_scores=len(scores)//len(policies)) for p in policies}
    summary = b.finish(scores, observed, policies, shapes, cm, denominator, cfg, args.output)
    summary.update(physical_gpus=cfg["physical_gpus"], offline_evaluation_execution=workers,
                   configuration_sha256=b.sha256(args.output / "configuration.json"))
    # Learned weights are fixed shared parameters; mapped K/V replace the native
    # cache. No second persistent full K/V cache is required by these controls.
    summary["auxiliary_memory"] = dict(learned_parameter_bytes={p["tag"]:sum(
        x.numel()*x.element_size() for m in p.get("modules",[]) for x in list(m.parameters())+list(m.buffers()))
        for p in policies if "modules" in p}, layer_boundary=summary["layer_boundary_state"],
        execution_peak_bytes={str(i):dict(allocated=w["peak_allocated_bytes"],reserved=w["peak_reserved_bytes"])
                              for i,w in enumerate(workers)})
    rows=sum(n*count for n,count in shapes["initial"].items())
    summary["auxiliary_memory"].update(common_native_kv_initial_bytes=2*16*320*4*rows,
        common_raw_history_initial_bytes=24*rows,
        raw_history_convention="int64 timestamp/item/behavior, bounded original history; common input for reconstructing controls",
        cache_convention="One persistent native-sized KV per point; persistent history maps replace it and query-only retains native writes. Layer boundary bytes are additional. Peak CUDA memory is this packed research runner across all 24 points, not per-method deployment memory.")
    b.write_json(args.output / "summary.json", summary)


def launch_pair(args, phase):
    logs = []
    processes = []
    started = time.perf_counter()
    try:
        for rank in range(2):
            path = (args.probe_output if phase == "probe" else args.output) / f"{phase}_{rank}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a")
            logs.append(handle)
            cmd = [sys.executable,"-u",str(Path(__file__).resolve()),"--phase",phase,"--output",str(args.output),"--rank",str(rank)]
            if phase == "probe":
                cmd += ["--probe-output",str(args.probe_output),"--users",str(args.users),
                        "--verify-users",str(args.verify_users),"--repeats",str(args.repeats)]
                if args.sweep: cmd.append("--sweep")
                if args.append_band_size:cmd += ["--append-band-size",str(args.append_band_size)]
                if args.query_batch:cmd += ["--query-batch",str(args.query_batch)]
            processes.append(subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            failed = [p.returncode for p in processes if p.poll() is not None and p.returncode]
            if failed: raise RuntimeError(f"{phase} worker failed; inspect {phase}_0.log and {phase}_1.log")
            time.sleep(1)
        assert all(p.returncode == 0 for p in processes), f"{phase} worker failed"
    finally:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes: p.wait()
        for handle in logs: handle.close()
    target = args.probe_output if phase == "probe" else args.output
    b.write_json(target / f"{phase}_wall.json",dict(status="complete",seconds=time.perf_counter()-started,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), command=sys.argv))


def main():
    def stop(signum,frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=["prepare","fit","probe","quality","aggregate","evaluate"],required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--rank",type=int,choices=[0,1])
    p.add_argument("--resume",action="store_true")
    p.add_argument("--probe-output",type=Path)
    p.add_argument("--users",type=int,default=512)
    p.add_argument("--verify-users",type=int,default=4)
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--sweep",action="store_true")
    p.add_argument("--single-gpu",action="store_true",help="probe the whole sample on one --rank")
    p.add_argument("--append-band-size",type=int,choices=[32,64,128,256])
    p.add_argument("--query-batch",type=int,choices=[8,16,32,64])
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.phase in ("fit","probe","quality","evaluate"):
        assert os.environ.get("CUDA_VISIBLE_DEVICES")=="0,1", "This entry reserves g202 physical GPU0/1; set CUDA_VISIBLE_DEVICES=0,1"
    if args.single_gpu:assert args.phase=="probe" and args.rank is not None
    lock = None
    if args.rank is None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        lock = (args.output.parent / (args.output.name+".lock")).open("w")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.phase == "prepare": prepare(args)
    elif args.phase == "aggregate": aggregate(args)
    elif args.phase == "evaluate":
        if not args.resume: prepare(args)
        else: binding_at(args.output)
        launch_pair(args,"fit")
        launch_pair(args,"quality")
        aggregate(args)
    elif args.rank is None:
        launch_pair(args,args.phase)
    else:
        {"fit":fit,"probe":probe,"quality":quality}[args.phase](args)


if __name__ == "__main__": main()
