"""Frozen populations and canonical inputs shared by the new Chapter 3 studies.

Only pre-release counts determine sampling. Old role membership is retained,
but every model score and quality anchor is computed again.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from hstu_kvcache.evaluation.history import load_histories_semijoin, restore_pack
from hstu_kvcache.evaluation.yambda import load_model
from hstu_kvcache.methods.baselines.layer_recompute.core import _capture_boundaries
from hstu_kvcache.evaluation.trajectory import _short_prefix_backend
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.methods.read_correction.calibration.v2.calibrate import mixed_candidates
from hstu_kvcache.common.artifacts import sha256

DAY = 86400
DATASET = ROOT / "data/processed/yambda500m_unified_v1/scales/large/dataset.json"
ROLE_SOURCE = ROOT / "experiments/01_quality_cost/runs/yambda/large/baselines_full_v0_v1/seed_17/attempt_1/configuration.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def quality_role(users, *, diagnostic=False):
    if len(set(users)) != len(users):
        raise ValueError("duplicate evaluation users")
    if len(users) < 512 and not diagnostic:
        raise ValueError("fewer than 512 users may only be a numerical/performance diagnostic")
    return "numerical_performance_diagnostic" if diagnostic else "measured_partial_population"


def quoted(path):
    return str(path).replace("'", "''")


def stratified_order(rows, seed):
    """Proportional interleaving; counts must come entirely from pre-release data."""
    groups = defaultdict(list)
    for uid, count, recent in rows:
        length_bin = int(np.searchsorted([32, 128, 256, 512, 1024], count, side="right"))
        active_bin = int(np.searchsorted([1, 8, 32, 128], recent, side="right"))
        groups[length_bin, active_bin].append(int(uid))
    weighted = []
    for stratum, values in groups.items():
        values.sort(key=lambda uid: hashlib.sha256(f"ch3/{seed}/{uid}".encode()).digest())
        weighted.extend(((i + .5) / len(values), stratum, uid) for i, uid in enumerate(values))
    weighted.sort()
    return [uid for _, _, uid in weighted]


def prepare_population(output, *, threads=12, seed=20261022):
    """Freeze the role/size protocol once, before any new model prediction."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "population.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["role_source_sha256"] != sha256(ROLE_SOURCE):
            raise ValueError("frozen role source changed")
        return saved
    original = json.loads(ROLE_SOURCE.read_text())["roles"]
    eligible = original["quality"]
    ds = json.loads(DATASET.read_text())
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)}")
    con.register("selected", pa.table({"uid": pa.array(eligible, type=pa.uint64())}))
    listens = quoted((DATASET.parent / ds["shared_listens_glob"]).resolve())
    rows = con.execute(f"""
        SELECT uid, count(*) AS n, count(*) FILTER (WHERE timestamp >= ?) AS recent
        FROM read_parquet('{listens}', hive_partitioning=true)
        SEMI JOIN selected USING(uid) WHERE timestamp < ? GROUP BY uid
    """, [217 * DAY, 231 * DAY]).fetchall()
    con.close()
    ordered = stratified_order(rows, seed)
    if set(ordered) != set(eligible):
        raise ValueError("eligible users must all have pre-release histories")
    # Disjoint roles; E3/E5 main is nested inside the E4 population.
    development, main, confirmation = ordered[:512], ordered[512:5512], ordered[5512:7512]
    used = set(development + main + confirmation)
    sensitivity = {}
    for sample_seed in (20261023, 20261024):
        sensitivity[str(sample_seed)] = [uid for uid in stratified_order(rows,sample_seed)
            if uid not in used][:2000]
    value = dict(seed=seed, role_source=str(ROLE_SOURCE.relative_to(ROOT)),
        role_source_sha256=sha256(ROLE_SOURCE), dataset_sha256=sha256(DATASET),
        selection="Proportional length/activity strata using only events before D231; SHA256 order within strata; no AUC/gap/outcome selection",
        strata_cutoffs=dict(length=[32,128,256,512,1024], recent_14d=[1,8,32,128]),
        fit=original["fit"], validation=original["validation"], development=development,
        main=main[:2000], mismatch=main, confirmation=confirmation,
        population_sensitivity=sensitivity,
        reserve=ordered[7512:], counts=dict(eligible=len(eligible), fit=len(original["fit"]),
            validation=len(original["validation"]), development=512, main=2000, mismatch=5000, confirmation=2000),
        minimum_quality_users=512, original_scores_reused=False)
    write_json(destination, value)
    return value


def save_histories(path, histories, uids):
    offsets = np.r_[0, np.cumsum([len(histories.rows[u][0]) for u in uids])]
    arrays = dict(uids=np.asarray(uids, dtype=np.uint64), offsets=offsets)
    for column, name in enumerate(("timestamps", "item_ids", "behaviors")):
        arrays[name] = np.concatenate([histories.rows[u][column] for u in uids])
    temporary = Path(str(path) + ".writing")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def prepare_edge(output, population, edge, *, threads=12):
    """Prepare one bounded window once; CPU packs shared by both GPU workers."""
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / "inputs.json"
    if binding_path.exists():
        saved = json.loads(binding_path.read_text())
        if saved["population_sha256"] != sha256(output.parent / "population.json"):
            raise ValueError("population changed")
        for name in ("history", "requests"):
            if sha256(output / saved[name]["path"]) != saved[name]["sha256"]:
                raise ValueError(f"frozen {name} pack changed")
        return saved
    release = 231 + (edge - 1) * 14
    cfg = dict(scale="large", edge=edge, release_day=release, end_day=release+14,
        history_length=1024, append_band_size=128, query_batch=16, repair_query_batch=16,
        attention_backend="triton", seed=17, history_threads=threads, torch_threads=4,
        calibration_queries_per_user=16, profile_users=64, confidence_replicates=1000,
        evaluation_checkpoint_users=256, stop_after_last_query=True,
        partial_points=dict(layer=[1,2,4,6,8], tail=[.05,.1,.2,.4,.8],
            deviation=[.05,.1,.2,.4,.6,.8], query=[.05,.1,.2,.4,.6,.8]),
        candidate_reads_per_user=230, behavior_appends_reference=20)
    if edge == 5:
        cfg["evaluation_window"] = dict(name="E14_partial", complete_source_horizon=False,
            interpretation="13 complete source days plus partial source day 300")
    uids = sorted(set(population["fit"] + population["validation"] + population["development"]
        + population["mismatch"] + population["confirmation"]
        + [uid for group in population["population_sensitivity"].values() for uid in group]))
    ds = json.loads(DATASET.read_text())
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)}")
    # Feedback presence qualifies E6 users, never ranks them by model outcomes.
    pool = sorted(set(uids + population["reserve"]))
    con.register("selected", pa.table({"uid": pa.array(pool, type=pa.uint64())}))
    feedback = quoted((DATASET.parent / ds["shared_feedback_glob"]).resolve())
    mapping = quoted((DATASET.parent / ds["item_mapping_path"]).resolve())
    requests = con.execute(f"""
        WITH grouped AS (
          SELECT uid,timestamp,raw_item_id,min(label) AS label
          FROM read_parquet('{feedback}',hive_partitioning=true)
          SEMI JOIN selected USING(uid) WHERE timestamp>=? AND timestamp<?
          GROUP BY uid,timestamp,raw_item_id HAVING min(label)=max(label)
        ) SELECT g.uid,g.timestamp AS query_timestamp,g.raw_item_id,m.item_idx,g.label
        FROM grouped g JOIN read_parquet('{mapping}') m USING(raw_item_id)
        ORDER BY uid,query_timestamp,raw_item_id
    """, [release*DAY, (release+14)*DAY]).fetch_arrow_table()
    con.close()
    active = set(int(x) for x in requests["uid"].to_numpy())
    e6_candidates = population["mismatch"] + population["reserve"]
    e6 = [uid for uid in e6_candidates if uid in active][:2000]
    if len(e6) != 2000:
        raise ValueError("not enough held-out feedback users for E6")
    uids = sorted(set(uids + e6))
    import pyarrow.compute as pc
    requests = requests.filter(pc.is_in(requests["uid"], value_set=pa.array(uids, type=pa.uint64())))
    pq.write_table(requests, output / "requests.parquet")
    histories = load_histories_semijoin(uids, dataset_path=DATASET,
        known_vocab_size=ds["foundation_items"], oov_buckets=256,
        start_timestamp=release*DAY, end_timestamp=(release+14)*DAY,
        max_history=1024, threads=threads)
    save_histories(output / "histories.npz", histories, uids)
    value = dict(settings=cfg, population_sha256=sha256(output.parent / "population.json"),
        e6_users=e6, packed_users=len(uids), feedback_events=requests.num_rows,
        history=dict(path="histories.npz", sha256=sha256(output / "histories.npz")),
        requests=dict(path="requests.parquet", sha256=sha256(output / "requests.parquet")),
        role="CPU input preparation only; no model prediction",
        seconds=time.perf_counter()-started)
    write_json(binding_path, value)
    return value


def load_edge(directory):
    directory = Path(directory)
    binding = json.loads((directory / "inputs.json").read_text())
    histories = restore_pack(directory / binding["history"]["path"])
    by_user = defaultdict(lambda: defaultdict(list))
    for row in pq.read_table(directory / binding["requests"]["path"]).to_pylist():
        by_user[int(row["uid"])][int(row["query_timestamp"])].append(row)
    return binding, histories, by_user


def load_models(edge, device, *, backend="triton"):
    import os
    os.environ["EVOKV_ATTENTION_BACKEND"] = backend
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    base = ROOT / "results/unified_training_2026_09/large/seed17/checkpoints"
    models, identities = [], {}
    for version in (edge-1, edge):
        path = base / f"v{version}/checkpoint_100.pt"
        model, info = load_model(path, device)
        model.eval().requires_grad_(False)
        models.append(model)
        identities[str(path.relative_to(ROOT))] = sha256(path)
        if model.cfg.hidden_size != 320 or len(model.blocks) != 10:
            raise ValueError("expected the actual 10L/H320 Large checkpoint")
    return *models, identities


def raw_prefix(histories, uid, cfg, device):
    from hstu_kvcache.evaluation.baselines import raw_tensors
    events = histories.rows[uid]
    end = int(np.searchsorted(events[0], cfg["release_day"]*DAY, side="left"))
    start = max(0, end-cfg["history_length"])
    return raw_tensors(tuple(column[start:end] for column in events), device), events, end


@torch.no_grad()
def capture_samples(parent, current, histories, uids, cfg, device, *, boundaries=False):
    samples = {}
    for uid in uids:
        raw, events, end = raw_prefix(histories, uid, cfg, device)
        n = raw[0].shape[1]
        with _short_prefix_backend(parent,n,1024), _capture_boundaries(parent) as captured:
            old = parent.compute_kv(*raw)
        with _short_prefix_backend(current,n,1024):
            teacher = current.compute_kv(*raw)
        candidates = mixed_candidates(uid, raw[0][0].cpu().numpy(), current.cfg.num_items-256,
            count=cfg["calibration_queries_per_user"], seed=cfg["seed"])
        delta = float(cfg["release_day"]*DAY-int(events[0][end-1]))
        # CPU snapshots bound resident memory; callers move one sample/batch at a time.
        old_cpu, teacher_cpu = old.to("cpu"), teacher.to("cpu")
        samples[uid] = dict(parent=old_cpu, teacher=teacher_cpu,
            parent_cache=old_cpu, teacher_cache=teacher_cpu,
            raw=tuple(x.cpu() for x in raw), item_ids=raw[0].cpu(),
            candidates=torch.as_tensor(candidates,dtype=torch.long)[None],
            query_delta=torch.tensor([delta],dtype=torch.float32), history_count=n)
        if boundaries:
            samples[uid]["boundaries"] = {k:v.cpu() for k,v in captured.items()}
    return samples


def sample_to(sample, device):
    return {key: (value.to(device) if isinstance(value,(torch.Tensor,HSTUKVCache)) else
        tuple(x.to(device) for x in value) if key=="raw" else
        {k:v.to(device) for k,v in value.items()} if key=="boundaries" else value)
        for key,value in sample.items()}
