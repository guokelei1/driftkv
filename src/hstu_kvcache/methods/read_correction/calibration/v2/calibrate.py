"""Shared causal mixed-candidate snapshots; the completed v2 fitting CLI retired."""
import gc
import json
import os
import time
import numpy as np
import pyarrow as pa
import torch
from hstu_kvcache.evaluation.yambda import load_histories, load_model
from hstu_kvcache.methods.read_correction.calibration.calibrate import capture
from hstu_kvcache.methods.read_correction.calibration.v2.common import ROOT, edge_name, sha256
from hstu_kvcache.methods.baselines.protocol.calibrate import snapshot


def mixed_candidates(uid, history_items, known, count=16, seed=17, recent_budget=None):
    """Recent unique known IDs, then uniform fill; default recent quota is half."""
    recent, seen = [], set()
    for value in reversed(history_items):
        item = int(value)
        if 0 < item < known and item not in seen:
            recent.append(item)
            seen.add(item)
            if len(recent) == (count//2 if recent_budget is None else recent_budget):
                break
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(uid)]))
    result = list(recent)
    while len(result) < count:
        item = int(rng.integers(1, known))
        if item not in seen:
            result.append(item)
            seen.add(item)
    return np.asarray(result, dtype=np.int64)


def load_data(args, config):
    """Shared bounded snapshot preparation; usable by a separate H v2 fitter."""
    binding_path = args.panel_root/args.scale/edge_name(args.edge)/"binding.json"
    binding = json.loads(binding_path.read_text())
    reservation_path = args.reservations/args.scale/edge_name(args.edge)/"calibration_users.json"
    reservation = json.loads(reservation_path.read_text())
    if reservation["source_binding"]["sha256"] != sha256(binding_path):
        raise RuntimeError("calibration reservation panel binding changed")
    uids = reservation["uids"][:max(args.budgets)]
    if len(uids)<max(args.budgets):
        raise RuntimeError("requested budget exceeds the expanded independent reservation")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    free,total = torch.cuda.mem_get_info(device)
    if free/total<config["initial_free_fraction"]:
        raise RuntimeError("GPU lacks initial free-memory reserve")
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"],device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(config["torch_threads"])
    pa.set_cpu_count(config["history_threads"])
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ["EVOKV_ATTENTION_BACKEND"]=config["attention_backend"]
    started=time.perf_counter()
    parent,pp=load_model(ROOT/binding["sources"]["parent"]["path"],device)
    current,cp=load_model(ROOT/binding["sources"]["current"]["path"],device)
    if pp["config"]!=cp["config"]:
        raise RuntimeError("parent/current architecture differs")
    current.requires_grad_(False);parent.requires_grad_(False)
    dataset_path=ROOT/binding["sources"]["dataset"]["path"]
    dataset=json.loads(dataset_path.read_text())
    known=int(cp.get("known_vocab_size",dataset["foundation_items"]))
    cutover=int(binding["cutover"])
    history=load_histories(uids,dataset_path=dataset_path,known_vocab_size=known,
        oov_buckets=int(cp["config"]["num_items"])-known,start_timestamp=cutover,end_timestamp=cutover+1,
        max_history=config["history_length"],threads=config["history_threads"])
    histories={uid:snapshot(history,uid,cutover,config["history_length"]) for uid in uids}
    del history,pp
    width=current.blocks[0].attn.num_heads*current.blocks[0].attn.head_dim
    cache_bytes=16*len(current.blocks)*width*sum(len(histories[uid][0]) for uid in uids)
    if cache_bytes>128*(1<<30):
        raise RuntimeError("CPU calibration caches exceed 128 GiB per process")
    batch=config["capture_batches"][args.scale]
    rows=capture(parent,current,histories,uids,cutover=cutover,known=known,
        queries=config["calibration_queries_per_user"],device=device,batch_size=batch,
        history_length=config["history_length"],attention_backend=config["attention_backend"])
    for uid in uids:
        rows[uid]["candidates"]=torch.from_numpy(mixed_candidates(uid,histories[uid][1],known,
            config["calibration_queries_per_user"],config["seed"]))
    del parent,histories
    gc.collect()
    metadata={"panel_binding_sha256":sha256(binding_path),"calibration_users_sha256":sha256(reservation_path),
        "cutover":cutover,"model_config":cp["config"],"cache_bytes":cache_bytes,
        "shared_snapshot_users":len(uids),
        "cache_prepare_seconds":time.perf_counter()-started,
        "checkpoint_hashes":{key:binding["sources"][key]["sha256"] for key in ("parent","current")},
        "candidate_rule":"half most-recent unique known items from strictly pre-release snapshot, half uniform known catalog; missing recent slots filled uniformly; SeedSequence([17,uid])",
        "teacher":"Current Full on aligned strictly-pre-release history; no feedback labels; mapped-item tie order shared with inherited cache"}
    del cp
    return current,rows,uids,device,metadata
