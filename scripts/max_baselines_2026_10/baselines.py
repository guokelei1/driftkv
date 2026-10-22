#!/usr/bin/env python3
"""Small independent-edge baseline study, reusing retained numerical methods.

No CT weights are fitted. Population and feedback inputs are frozen before
prediction. Partial repair persists for the edge; every next edge starts again
from native old-model state. Quality and deployment-cost populations are
explicitly bound. Wall seconds are progress, never formal timing evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from design_one.prepare_history import load_histories_semijoin
from evaluate_yambda500m_foundation_raw import load_model
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.baselines.layer_recompute.core import _capture_boundaries
from hstu_kvcache.baselines.tail_recompute import recompute_tail
from hstu_kvcache.baselines.deviation_recompute import layer0_deviation_scores
from hstu_kvcache.baselines.query_recompute import query_attention_scores
from hstu_kvcache.baselines.sparse_recompute import recompute_positions, select_top_positions
from hstu_kvcache.design_two.replay import NativeTrajectory, _short_prefix_backend
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.read_correction import score_corrected
from read_correction_2026_09.cost import CostModel, eager_read
from read_correction_2026_09.v2.calibrate import mixed_candidates
from read_correction_v4.probe import fit_modules
from read_correction_v4.history_conditioned.fit_nonlinear import fit_nonlinear
from read_correction_v4.cost_nonlinear import correction_parts as history_parts
from read_correction_v5.query_only.fit import fit_query_features
from read_correction_v5.query_only.refine import refine_query
from read_correction_v5.query_only.cost import correction_forward as query_cost
from unified_reuse_2026_09.common import sha256, write_json as write_plain_json

DAY = 86400


def write_json(path,value):
    """Commit manifests/progress atomically so interruption cannot truncate them."""
    staged=path.with_suffix(path.suffix+".writing")
    write_plain_json(staged,value)
    staged.replace(path)


def dataset_path(scale):
    name = ("data/processed/yambda500m_unified_v1/scales/large/dataset.json" if scale == "large"
            else "data/processed/yambda5b_max_200k_v1/scales/max/dataset.json")
    return ROOT / name


def historical_users():
    """Preserve fit/validation/development reservations; include prior test users.

    This is a full development comparison, not a new blind confirmation. Prior
    evaluation membership is not used to filter users by observed quality.
    """
    paths = set((ROOT / "results/read_correction_2026_09").rglob("calibration_users.json"))
    paths.update((ROOT / "results/selective_recompute_2026_09/panels").rglob("users.json"))
    paths.update((ROOT / "results/design_two_2026_10").rglob("users.json"))
    excluded, records = set(), []
    for path in sorted(paths):
        value = json.loads(path.read_text())
        for key, items in value.items():
            if key in ("uids", "fit", "validation", "pilot", "calibration", "canary"):
                if isinstance(items, list):
                    excluded.update(int(x) for x in items)
        records.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)))
    return excluded, records


def prepare_inputs(cfg, output):
    path = dataset_path(cfg["scale"])
    ds = json.loads(path.read_text())
    excluded, reservations = historical_users()
    users = pq.read_table(path.parent / ds["users_path"], columns=["uid", "first_timestamp"])
    start, end = cfg["release_day"]*DAY, cfg["end_day"]*DAY
    eligible = [int(row["uid"]) for row in users.to_pylist()
                if int(row["first_timestamp"]) < start and int(row["uid"]) not in excluded]
    def key(uid):
        return hashlib.sha256(f"baseline_feedback_20261021/{cfg['seed']}/{uid}".encode()).digest()
    eligible.sort(key=key)
    max_fit = max(*cfg["query_fit_users"], *cfg["history_fit_users"])
    if cfg.get("fitting_role_source"):
        prior=json.loads((ROOT/cfg["fitting_role_source"]).read_text())["roles"]
        train=prior["fit"][:max_fit]
        validation=prior["validation"][:cfg["validation_users"]]
    else:
        train = eligible[:max_fit]
        validation = eligible[max_fit:max_fit+cfg["validation_users"]]
    reserved=set(train+validation)
    remaining=[uid for uid in eligible if uid not in reserved]
    if cfg["quality_users"]=="all_eligible":
        quality=remaining
        cfg["quality_users"]=len(quality)
    else:
        quality=remaining[:cfg["quality_users"]]
    if cfg["cost_users"]=="same_as_quality":
        deployment=quality
        cfg["cost_users"]=len(deployment)
    else:
        deployment=remaining[:cfg["cost_users"]]
    assert len(deployment) == cfg["cost_users"] and len(quality) == cfg["quality_users"]
    assert not (set(train) & set(validation) or set(train+validation) & set(deployment))
    selected = train+validation+deployment
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={cfg['history_threads']}")
    con.register("selected",pa.table({"uid":pa.array(selected,type=pa.uint64())}))
    feedback = str((path.parent/ds["shared_feedback_glob"]).resolve()).replace("'","''")
    mapping = str((path.parent/ds["item_mapping_path"]).resolve()).replace("'","''")
    requests = con.execute(f"""
        WITH grouped AS (
          SELECT uid,timestamp,raw_item_id,min(label) AS label
          FROM read_parquet('{feedback}',hive_partitioning=true)
          SEMI JOIN selected USING(uid)
          WHERE timestamp>=? AND timestamp<?
          GROUP BY uid,timestamp,raw_item_id HAVING min(label)=max(label)
        ) SELECT g.uid,g.timestamp AS query_timestamp,g.raw_item_id,m.item_idx,g.label
        FROM grouped g JOIN read_parquet('{mapping}') m USING(raw_item_id)
        ORDER BY uid,query_timestamp,raw_item_id
    """,[start,end]).fetch_arrow_table()
    con.close()
    # Fitting uses only prerelease history/teacher outputs, never these labels.
    pq.write_table(requests,output/"requests.parquet")
    binding = dict(settings=cfg,dataset=str(path.relative_to(ROOT)),dataset_sha256=sha256(path),
        roles=dict(fit=train,validation=validation,quality=quality,cost=deployment),
        historical_reservations=reservations,excluded_historical_users=len(excluded),
        selection="SHA256(seed,uid), only prerelease first behavior and retained role exclusions; no feedback or model gains used",
        request_rule="known targets; group uid/timestamp/raw item, discard conflicting labels; all feedback in the fixed window",
        requests_sha256=sha256(output/"requests.parquet"),requests=requests.num_rows,
        populations=(f"same {len(quality)} eligible users for quality and complete deployment costs"
            if quality==deployment else
            f"{len(quality)} quality users and a separate {len(deployment)}-user cost panel"))
    write_json(output/"configuration.json",binding)
    return binding


def raw_tensors(events,device):
    times,items,behaviors = events
    delta = np.zeros(len(times),dtype=np.float32)
    delta[1:] = np.diff(times)
    return (torch.as_tensor(items,device=device,dtype=torch.long)[None],
            torch.as_tensor(behaviors,device=device,dtype=torch.long)[None],
            torch.as_tensor(delta,device=device)[None])


@torch.no_grad()
def capture(parent,current,histories,uids,cfg,device):
    rows,counts,transfer = {},Counter(),0
    at = cfg["release_day"]*DAY
    for i,uid in enumerate(uids):
        events = histories.rows[uid]
        n = min(int(np.searchsorted(events[0],at,side="left")),cfg["history_length"])
        if not n:
            raise ValueError(f"no prerelease history for fitting user {uid}")
        tensors = raw_tensors(tuple(x[:n] for x in events),device)
        with _short_prefix_backend(parent,n,cfg["history_length"]), _capture_boundaries(parent) as bounds:
            old = parent.compute_kv(*tensors)
        with _short_prefix_backend(current,n,cfg["history_length"]):
            teacher = current.compute_kv(*tensors)
        ids = mixed_candidates(uid,tensors[0][0].cpu().numpy(),
            current.cfg.num_items-256,count=cfg["calibration_queries_per_user"],seed=cfg["seed"])
        rows[uid] = dict(parent=HSTUKVCache(old.k.cpu(),old.v.cpu(),n),
            teacher=HSTUKVCache(teacher.k.cpu(),teacher.v.cpu(),n),
            candidates=torch.as_tensor(ids,dtype=torch.long),
            query_delta=float(at-int(events[0][n-1])))
        if i < cfg["profile_users"]:
            rows[uid]["boundaries"] = {k:v.cpu() for k,v in bounds.items()}
        transfer += sum(x.numel()*x.element_size() for x in (old.k,old.v,teacher.k,teacher.v))
        counts[n] += 1
        if (i+1)%32 == 0:
            print(json.dumps(dict(status="capturing_prerelease_teachers",users=i+1,total=len(uids))),flush=True)
    return rows,counts,transfer


def cache_cost(rows,uids,cm,cfg):
    return 2*sum((cm if rows[u]["parent"].seq_len==cfg["history_length"] else
                  CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,"torch")).full_cache(rows[u]["parent"].seq_len)
                 for u in uids)


@torch.no_grad()
def profile_layers(current,rows,uids,cm,cfg,device):
    """Profile individual layers on fit users; select contiguous budget windows."""
    gains = np.zeros(cm.num_layers)
    flops = cache_cost(rows,uids,cm,cfg)
    for uid in uids:
        row = rows[uid]
        cache,teacher = row["parent"].to(device),row["teacher"].to(device)
        n,q = cache.seq_len,len(row["candidates"])
        cost = cm if n==cfg["history_length"] else CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,"torch")
        ids = row["candidates"].to(device)[None]
        delta = torch.tensor([row["query_delta"]],device=device)
        wanted = score(current,teacher,ids,delta)[0]
        native = score(current,cache,ids,delta)[0]
        baseline = (native-wanted).square().mean()
        state = lr.LayerRecomputeState(cache,{k:v.to(device) for k,v in row["boundaries"].items()})
        # Layer zero consumes raw events; filled by the caller from the same source window.
        tensors = row["raw"]
        flops += 2*eager_read(cm,n,queries=q)+3*q
        with _short_prefix_backend(current,n,cfg["history_length"]):
            for layer in range(cm.num_layers):
                repaired = lr.recompute_interval(current,state,*tensors,(layer,layer))
                actual = score(current,repaired.cache,ids,delta)[0]
                gains[layer] += float(baseline-(actual-wanted).square().mean())/len(uids)
                flops += cost.layer_rebuild(n,(layer,layer))+eager_read(cm,n,queries=q)+4*q
    return gains.tolist(),int(flops)


def save_modules(output,tag,modules,metadata):
    folder = output/tag
    folder.mkdir(exist_ok=True)
    staged=folder/"calibration.pt.tmp"
    torch.save([dict(config=m.get_config(),state_dict={k:v.detach().cpu() for k,v in m.state_dict().items()})
                for m in modules],staged)
    staged.replace(folder/"calibration.pt")
    write_json(folder/"summary.json.tmp",dict(status="fitted",**metadata,
        weights_sha256=sha256(folder/"calibration.pt")))
    (folder/"summary.json.tmp").replace(folder/"summary.json")


def fit_learned(current,rows,roles,cm,cfg,device,output,fit_source=None):
    policies=[]
    sources=[output]+([fit_source] if fit_source is not None else [])+[ROOT/p for p in cfg.get("completed_fit_sources",[])]
    for method,budgets in (("query_only",cfg["query_fit_users"]),("history_conditioned",cfg["history_fit_users"])):
        for budget in budgets:
            completed=next((p for p in sources if (p/f"{method}_fit{budget}/summary.json").exists()),None)
            if completed is not None:
                subset=dict(cfg,query_fit_users=[budget] if method=="query_only" else [],
                    history_fit_users=[budget] if method=="history_conditioned" else [])
                policies.extend(reuse_learned(completed,roles,subset,device,output))
                continue
            train,val = roles["fit"][:budget],roles["validation"]
            captured = cache_cost(rows,train+val,cm,cfg)
            if method == "query_only":
                modules,first = fit_query_features(current,rows,train,val,cfg,device,cfg["scale"],batch_size=8)
                units = [max(1e-12,math.sqrt(r["diagnostics"]["train"]["baseline_rate_mse"])) for r in first["layers"]]
                modules,second = refine_query(current,rows,train,val,modules,{**cfg,"output_units":units},device,cfg["scale"],batch_size=4)
                calibration = captured+first["cost"]["calibration_flops"]+second["cost"]["calibration_flops"]
                details = dict(ridge=first,joint=second)
            else:
                base,base_records,base_cost = fit_modules(current,rows,train,val,cfg,device,cfg["scale"])
                modules,second = fit_nonlinear(current,rows,train,val,base,config=cfg,device=device,scale=cfg["scale"],batch_size=8)
                calibration = base_cost["calibration_flops"]+second["cost"]["calibration_flops"]
                details = dict(affine=base_records,affine_cost=base_cost,nonlinear=second)
            tag = f"{method}_fit{budget}"
            save_modules(output,tag,modules,dict(method=method,fit_users=train,validation_users=val,
                calibration_flops=int(calibration),details=details))
            policies.append(dict(tag=tag,method=method,modules=modules,calibration_flops=int(calibration),fit_users=budget,
                weights=str(output/tag/"calibration.pt"),weights_sha256=sha256(output/tag/"calibration.pt")))
    return policies


def reuse_learned(source,roles,cfg,device,output):
    """Reuse completed fitting after an evaluation failure, without refitting."""
    from hstu_kvcache.read_correction_v4.nonlinear import NonlinearTokenReadCorrection
    from hstu_kvcache.read_correction_v5.query_only.core import QueryFeatureCorrection
    policies=[]
    for method,budgets in (("query_only",cfg["query_fit_users"]),("history_conditioned",cfg["history_fit_users"])):
        for budget in budgets:
            tag=f"{method}_fit{budget}"
            folder=source/tag
            metadata=json.loads((folder/"summary.json").read_text())
            assert metadata["fit_users"]==roles["fit"][:budget]
            assert metadata["validation_users"]==roles["validation"]
            assert metadata["weights_sha256"]==sha256(folder/"calibration.pt")
            cls=QueryFeatureCorrection if method=="query_only" else NonlinearTokenReadCorrection
            artifact=torch.load(folder/"calibration.pt",map_location="cpu",weights_only=True)
            modules=[]
            for layer in artifact:
                module=cls(**layer["config"]).to(device)
                module.load_state_dict(layer["state_dict"])
                modules.append(module.eval().requires_grad_(False))
            if cfg.get("write_fit_source_records",True):
                write_json(output/(tag+"_source.json"),dict(path=str(folder),weights_sha256=metadata["weights_sha256"]))
            policies.append(dict(tag=tag,method=method,modules=modules,
                calibration_flops=metadata["calibration_flops"],fit_users=budget,
                weights=str(folder/"calibration.pt"),weights_sha256=metadata["weights_sha256"]))
    return policies


def event_shapes(histories,by_user,uids,cfg):
    """Exact band/score shapes for this runner; no model outcomes are needed."""
    first,initial,appends,queries = Counter(),Counter(),Counter(),Counter()
    at,end = cfg["release_day"]*DAY,cfg["end_day"]*DAY
    cap,band = cfg["history_length"],cfg["append_band_size"]
    for uid in uids:
        times = histories.rows[uid][0]
        cursor = int(np.searchsorted(times,at,side="left"))
        n = min(cursor,cap)
        initial[n] += 1
        rounds = by_user.get(uid,{})
        first_query = min(rounds,default=end)
        first_behavior = int(times[cursor]) if cursor<len(times) else end
        if min(first_query,first_behavior)<end:
            first[n, "behavior" if first_behavior<first_query else "query",
                  1 if first_behavior<first_query else min(len(rounds[first_query]),cfg.get("repair_query_batch",16))] += 1
        for until in sorted(set(rounds)|{end}):
            stop = int(np.searchsorted(times,until,side="left"))
            while cursor<stop:
                width = 1 << (min(band,stop-cursor).bit_length()-1)
                appends[n,width,n==cap] += 1
                cursor += width
                n = min(cap,n+width)
            if until in rounds:
                for pos in range(0,len(rounds[until]),cfg["query_batch"]):
                    queries[n,min(cfg["query_batch"],len(rounds[until])-pos)] += 1
    return dict(first=first,initial=initial,appends=appends,queries=queries)


def repair_cost(policy,shapes,cm):
    total,sorts = 0,0.
    for (n,kind,q),count in shapes["first"].items():
        cost = cm if n==1024 else CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,"torch")
        method = policy["method"]
        if method == "rebuild":
            work = dict(total_flops=cost.full_cache(n),selection_sort_comparisons_estimate=0)
        else:
            work = cost.operation_cost(method,n,selected_tokens=math.ceil(n*policy.get("fraction",0)),
                interval=policy.get("interval"),queries=q)
            if method=="query" and kind=="behavior" and 0<math.ceil(n*policy["fraction"])<n:
                work["total_flops"] += cost.embedding(q)-cost.embedding(q,query=True)
        total += count*work["total_flops"]
        sorts += count*work["selection_sort_comparisons_estimate"]
    return int(total+policy.get("calibration_flops",0)),sorts


def choose_repairs(shapes,cm,gains,profile_flops,cfg):
    denominator = repair_cost(dict(method="rebuild"),shapes,cm)[0]
    policies=[dict(tag="reuse",method="reuse",calibration_flops=0),
              dict(tag="rebuild",method="rebuild",calibration_flops=0)]
    for method in ("layer","tail","deviation","query"):
        for value in cfg["partial_points"][method]:
            if method=="layer":
                width=int(value)
                start=max(range(cm.num_layers-width+1),key=lambda s:sum(gains[s:s+width]))
                policy=dict(method=method,interval=(start,start+width-1),layers=width,
                    calibration_flops=profile_flops,tag=f"layer_{width}layers")
            else:
                policy=dict(method=method,fraction=value,calibration_flops=0,tag=f"{method}_{100*value:g}pct")
            policy["cost_percent"]=100*repair_cost(policy,shapes,cm)[0]/denominator
            policies.append(policy)
    return policies,denominator


@torch.no_grad()
def query_behavior_scores(current,cache,item,behavior,delta):
    block=current.blocks[0]
    x=block.norm(current.embed_inputs(item,behavior,delta))
    q=block.attn.q_proj(x).reshape(1,1,current.cfg.num_heads,32).transpose(1,2)
    k=cache.k[0].reshape(1,cache.seq_len,current.cfg.num_heads,32).transpose(1,2)
    return block.attn._activate(q@k.transpose(-2,-1)*block.attn.scale).abs().mean((1,2))


@torch.no_grad()
def prepare_policy(trajectory,boundaries,policy,current,consumer,cfg,device,*,selection_scores=None):
    method=policy["method"]
    if method in ("reuse","query_only"):
        return
    cache=trajectory.cache
    n=cache.seq_len
    if not n:
        return
    raw=raw_tensors(trajectory.retained_events,device)
    with _short_prefix_backend(current,n,cfg["history_length"]):
        if method=="rebuild":
            trajectory.cache=trajectory.fresh_cache()
        elif method=="layer":
            trajectory.cache=lr.recompute_interval(current,lr.LayerRecomputeState(cache,boundaries),*raw,policy["interval"]).cache
        elif method=="tail":
            trajectory.cache=recompute_tail(current,cache,*raw,math.ceil(n*policy["fraction"]))
        elif method in ("deviation","query"):
            count=min(n,math.ceil(n*policy["fraction"]))
            if count==n:
                trajectory.cache=trajectory.fresh_cache()
            else:
                if selection_scores is not None:
                    importance=selection_scores
                elif method=="deviation":
                    importance=layer0_deviation_scores(current,cache,*raw)
                elif consumer["kind"]=="query":
                    importance=query_attention_scores(current,cache,consumer["items"],consumer["delta"])
                else:
                    importance=query_behavior_scores(current,cache,consumer["items"],consumer["behavior"],consumer["delta"])
                trajectory.cache=recompute_positions(current,cache,*raw,select_top_positions(importance,count),query_chunk_size=64)
        elif method=="history_conditioned":
            counts=torch.tensor([n],device=device)
            mapped=[m.map_tokens(cache.k[l],cache.v[l],counts) for l,m in enumerate(policy["modules"])]
            trajectory.cache=HSTUKVCache(torch.stack([x[0] for x in mapped]),torch.stack([x[1] for x in mapped]),n)
        else:
            raise ValueError(method)


@torch.no_grad()
def evaluate(parent,current,histories,by_user,uids,policies,cfg,device,output):
    scores=[]
    observed={p["tag"]:Counter() for p in policies}
    at,end=cfg["release_day"]*DAY,cfg["end_day"]*DAY
    for index,uid in enumerate(uids):
        with _capture_boundaries(parent) as boundaries:
            base=NativeTrajectory(parent,cfg["edge"]-1,histories.rows[uid],at,
                max_length=cfg["history_length"],append_band_size=cfg["append_band_size"])
        for policy in policies:
            trajectory=copy.copy(base)
            trajectory.ledger=Counter(base.ledger)
            trajectory.producers=base.producers.copy()
            trajectory.switch(current,cfg["edge"],at)
            prepared=False
            def prepare_first(consumer):
                nonlocal prepared
                if not prepared:
                    prepare_policy(trajectory,boundaries,policy,current,consumer,cfg,device)
                    observed[policy["tag"]]["prepare_rows"]+=trajectory.cache.seq_len
                    prepared=True
            for timestamp,requests in sorted(by_user.get(uid,{}).items()):
                stop=int(np.searchsorted(trajectory.events[0],timestamp,side="left"))
                if not prepared and trajectory.cursor<stop:
                    pos=trajectory.cursor
                    prepare_first(dict(kind="behavior",items=torch.tensor([[int(trajectory.events[1][pos])]],device=device),
                        behavior=torch.tensor([[int(trajectory.events[2][pos])]],device=device),
                        delta=torch.tensor([[min(7*DAY,int(trajectory.events[0][pos])-trajectory.last_timestamp)]],device=device,dtype=torch.float32)))
                trajectory.advance(timestamp)
                # Freeze selection inputs independently of inference tiling.
                selection=requests[:cfg.get("repair_query_batch",16)]
                prepare_first(dict(kind="query",items=torch.tensor([[r["item_idx"] for r in selection]],device=device),
                    delta=torch.tensor([timestamp-trajectory.last_timestamp],device=device,dtype=torch.float32)))
                for pos in range(0,len(requests),cfg["query_batch"]):
                    batch=requests[pos:pos+cfg["query_batch"]]
                    ids=torch.tensor([[r["item_idx"] for r in batch]],device=device)
                    delta=torch.tensor([timestamp-trajectory.last_timestamp],device=device,dtype=torch.float32)
                    if policy["method"]=="query_only":
                        values=score_corrected(current,trajectory.cache,ids,delta,policy["modules"],
                            torch.tensor([trajectory.cache.seq_len],device=device))[0]
                    else:
                        values=score(current,trajectory.cache,ids,delta)[0]
                    for row,value in zip(batch,values[0].cpu().tolist(),strict=True):
                        scores.append(dict(row,method=policy["tag"],score=value))
                    observed[policy["tag"]]["candidate_scores"]+=len(batch)
            if not prepared and trajectory.cursor<len(trajectory.events[0]):
                pos=trajectory.cursor
                prepare_first(dict(kind="behavior",items=torch.tensor([[int(trajectory.events[1][pos])]],device=device),
                    behavior=torch.tensor([[int(trajectory.events[2][pos])]],device=device),
                    delta=torch.tensor([[min(7*DAY,int(trajectory.events[0][pos])-trajectory.last_timestamp)]],device=device,dtype=torch.float32)))
            trajectory.advance(end)
            observed[policy["tag"]]["appended_rows"]+=trajectory.appended_tokens
        if (index+1)%8==0:
            print(json.dumps(dict(status="baseline_quality",users=index+1,total=len(uids),scores=len(scores))),flush=True)
            write_json(output/"progress.json",dict(status="baseline_quality",users=index+1,total=len(uids)))
    pq.write_table(pa.Table.from_pylist(scores),output/"scores.parquet")
    return scores,observed


def layer_boundary_accounting(policies,shapes,cm):
    rows=sum(n*count for n,count in shapes["initial"].items())
    points=[]
    for policy in policies:
        if policy["method"]!="layer":continue
        start,end=policy["interval"]
        points.append(dict(method=policy["tag"],interval=[start,end],
            required_old_boundary_layers=[start] if start else [],
            required_persistent_old_boundary_bytes=int(rows*cm.hidden_size*4) if start else 0,
            required_persistent_old_boundary_relative_to_native_kv=1/(2*cm.num_layers) if start else 0.))
    return dict(selected_points=points,
        policy="A prefix starting at layer zero recomputes from raw events and needs no old hidden boundary. A nonzero start needs its actual old input, not teacher hidden.",
        prototype="Shared research capture keeps all old block inputs transiently for profiling/evaluation; this is not a persistent-state requirement of a selected prefix point.",
        scope="Point-specific additional persistent old boundary state only; raw history/native KV and temporary reconstruction workspace are separate.",
        nonprefix_source="Non-prefix boundaries must come from the old producer or separately charged old-state reconstruction.")


def finish(scores,observed,policies,shapes,cm,denominator,cfg,output):
    from max_baselines_2026_10.statistics import paired_user_bootstrap,write_tables
    from sklearn.metrics import roc_auc_score
    groups=defaultdict(list)
    for row in scores:
        groups[row["method"]].append(row)
    metrics={tag:float(roc_auc_score([r["label"] for r in rows],[r["score"] for r in rows])) for tag,rows in groups.items()}
    gap=metrics["rebuild"]-metrics["reuse"]
    native_append=sum(count*(cm if full else CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,"torch")).band_append(n,w,window_size=cfg["history_length"])
                      for (n,w,full),count in shapes["appends"].items())
    native_query=sum(count*eager_read(cm,n,queries=q) for (n,q),count in shapes["queries"].items())
    initial=sum(count*(cm if n==cfg["history_length"] else CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,"torch")).full_cache(n)
                for n,count in shapes["initial"].items())
    results=[]
    for policy in policies:
        method=policy["method"]
        sorts=0
        if method=="reuse":extra=0
        elif method=="query_only":
            extra=policy["calibration_flops"]+sum(count*sum(query_cost(m.get_config(),n,queries=q) for m in policy["modules"])
                for (n,q),count in shapes["queries"].items())
        elif method=="history_conditioned":
            extra=policy["calibration_flops"]
            for (n,_,_),count in shapes["first"].items():
                for module in policy["modules"]:
                    parts=history_parts(module.get_config(),n,queries=0)
                    extra+=count*sum(parts.values())
        else:extra,sorts=repair_cost(policy,shapes,cm)
        row=dict(method=policy["tag"],auc=metrics[policy["tag"]],auc_gain_over_reuse=metrics[policy["tag"]]-metrics["reuse"],
            recovery_percent=100*(metrics[policy["tag"]]-metrics["reuse"])/gap if gap!=0 else None,
            complete_added_flops=int(extra),percent_of_once_rebuild=100*extra/denominator,
            complete_native_plus_method_flops=int(native_append+native_query+extra),
            calibration_flops=policy.get("calibration_flops",0),
            setting={k:v for k,v in policy.items() if k not in ("modules","tag","method","calibration_flops")},
            quality_operations=dict(observed[policy["tag"]]),selection_sort_comparisons_estimate=sorts)
        results.append(row)
    summary=dict(status="complete",machine=socket.gethostname(),physical_gpu=cfg["physical_gpu"],
        model_scale=cfg["scale"],edge=f"v{cfg['edge']-1}_to_v{cfg['edge']}",evaluation_role=cfg["evaluation_role"],
        quality_users=cfg["quality_users"],cost_users=cfg["cost_users"],
        quality_requests=len(groups["reuse"]),positive_feedback=sum(r["label"] for r in groups["reuse"]),
        rebuild_minus_reuse_auc=gap,once_rebuild_flops=int(denominator),native_append_flops=int(native_append),
        native_query_flops=int(native_query),common_old_cache_initialization_flops=int(initial),results=results,
        cost_population_scope=cfg["cost_population_scope"],timing_scope=cfg["timing_scope"],
        preparation_rule=cfg["repair_schedule"],query_only_schedule=cfg["query_only_schedule"],
        arithmetic="executed-shape analytical FLOPs, FMA=2; retained fitting conventions estimate solves/backward/optimizer arithmetic; sorting and transfers are separate",
        created_utc=datetime.now(timezone.utc).isoformat())
    summary["evaluation_window"]=dict(start_day=cfg["release_day"],end_day_exclusive=cfg["end_day"],
        **cfg.get("evaluation_window",{}))
    summary["layer_boundary_state"]=layer_boundary_accounting(policies,shapes,cm)
    quality_uids=json.loads((output/"configuration.json").read_text())["roles"]["quality"]
    confidence,metadata=paired_user_bootstrap(groups,quality_uids,
        replicates=cfg.get("confidence_replicates",1000),seed=cfg.get("confidence_seed",17021),
        progress=lambda value:print(json.dumps(value),flush=True))
    for row in results:row.update(confidence[row["method"]])
    summary["confidence"]=metadata
    write_tables(summary,output)
    write_json(output/"summary.json",summary)
    return summary


def run(args):
    started=time.perf_counter()
    cfg=json.loads(args.config.read_text())
    output=args.output.resolve()
    config_sha=sha256(args.config)
    execution_sources={str(path.relative_to(ROOT)):sha256(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))}
    if args.resume:
        binding=json.loads((output/"configuration.json").read_text())
        assert binding["launch_config_sha256"]==config_sha,"launch config changed; use a new attempt"
        assert binding["execution_source_sha256"]==execution_sources,"execution code changed; use a new attempt"
        assert binding["requests_sha256"]==sha256(output/"requests.parquet"),"request input changed"
        cfg=binding["settings"]
        assert binding["dataset_sha256"]==sha256(dataset_path(cfg["scale"])),"dataset manifest changed"
    else:
        output.mkdir(parents=True,exist_ok=False)
    if args.canary:
        cfg.update(quality_users=8,cost_users=1024,query_fit_users=[4,8,16],history_fit_users=[4,8,16],
            validation_users=4,profile_users=2,query_joint_epochs=1,nonlinear_epochs=1)
    if not args.resume:
        binding=prepare_inputs(cfg,output)
        binding.update(launch_config_sha256=config_sha,execution_source_sha256=execution_sources)
        write_json(output/"configuration.json",binding)
    if args.reuse_fits:
        binding["completed_fits_source"]=str(args.reuse_fits)
        write_json(output/"configuration.json",binding)
    print(json.dumps(dict(status="inputs_frozen",quality_users=cfg["quality_users"],cost_users=cfg["cost_users"],requests=binding["requests"])),flush=True)
    roles=binding["roles"]
    requests=pq.read_table(output/"requests.parquet").to_pylist()
    by_user=defaultdict(lambda:defaultdict(list))
    for row in requests:
        by_user[int(row["uid"])][int(row["query_timestamp"])].append(row)
    ds_path=dataset_path(cfg["scale"])
    ds=json.loads(ds_path.read_text())
    uids=roles["fit"]+roles["validation"]+roles["cost"]
    histories=load_histories_semijoin(uids,dataset_path=ds_path,known_vocab_size=ds["foundation_items"],oov_buckets=256,
        start_timestamp=cfg["release_day"]*DAY,end_timestamp=cfg["end_day"]*DAY,max_history=cfg["history_length"],threads=cfg["history_threads"])
    shapes=event_shapes(histories,by_user,roles["cost"],cfg)
    write_json(output/"event_shapes.json",{name:[dict(shape=list(key) if isinstance(key,tuple) else [key],count=value)
        for key,value in counts.items()] for name,counts in shapes.items()})
    os.environ["EVOKV_ATTENTION_BACKEND"]=cfg["attention_backend"]
    torch.set_num_threads(cfg["torch_threads"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(cfg["seed"])
    device=torch.device("cuda:0")
    torch.cuda.set_device(device)
    base=ROOT/f"results/unified_training_2026_09/{cfg['scale']}/seed17/checkpoints"
    parent,old=load_model(base/f"v{cfg['edge']-1}/checkpoint_100.pt",device)
    current,new=load_model(base/f"v{cfg['edge']}/checkpoint_100.pt",device)
    assert old["config"]==new["config"]
    cfg["actual_model_config"]=new["config"]
    binding["model_config"]=new["config"]
    binding["checkpoint_paths"]=[str(base/f"v{version}/checkpoint_100.pt")
        for version in (cfg["edge"]-1,cfg["edge"])]
    checkpoint_sha={path:sha256(Path(path)) for path in binding["checkpoint_paths"]}
    if args.resume and "checkpoint_sha256" in binding:
        assert binding["checkpoint_sha256"]==checkpoint_sha,"backbone checkpoints changed"
    binding["checkpoint_sha256"]=checkpoint_sha
    binding["command"]=sys.argv
    binding["cuda_visible_devices"]=os.environ.get("CUDA_VISIBLE_DEVICES")
    write_json(output/"configuration.json",binding)
    del old,new
    parent.eval().requires_grad_(False)
    current.eval().requires_grad_(False)
    cm=CostModel(current.cfg.hidden_size,len(current.blocks),current.cfg.num_heads,cfg["attention_backend"])
    sources=[output]+([args.reuse_fits] if args.reuse_fits else [])+[ROOT/p for p in cfg.get("completed_fit_sources",[])]
    missing=any(not any((source/f"{method}_fit{budget}/summary.json").exists() for source in sources)
        for method,budgets in (("query_only",cfg["query_fit_users"]),("history_conditioned",cfg["history_fit_users"])) for budget in budgets)
    if missing:
        rows,counts,transfer=capture(parent,current,histories,roles["fit"]+roles["validation"],cfg,device)
        write_json(output/"capture_summary.json",dict(transfer_bytes=transfer,history_counts=dict(counts)))
    else:
        rows={}
        capture_summary=output/"capture_summary.json"
        transfer=json.loads(capture_summary.read_text())["transfer_bytes"] if capture_summary.exists() else None
    profile_source=(output/"partial_budgets.json" if args.resume and (output/"partial_budgets.json").exists() else
        args.reuse_fits/"partial_budgets.json" if args.reuse_fits else
        ROOT/cfg["layer_profile_source"] if cfg.get("layer_profile_source") else None)
    if profile_source is not None:
        saved=json.loads(profile_source.read_text())
        gains,profile_flops=saved["layer_profile_gains"],saved["profile_flops"]
    else:
        for uid in roles["fit"][:cfg["profile_users"]]:
            events=histories.rows[uid]
            stop=int(np.searchsorted(events[0],cfg["release_day"]*DAY,side="left"))
            rows[uid]["raw"]=raw_tensors(tuple(x[max(0,stop-cfg["history_length"]):stop] for x in events),device)
        gains,profile_flops=profile_layers(current,rows,roles["fit"][:cfg["profile_users"]],cm,cfg,device)
    repairs,denominator=choose_repairs(shapes,cm,gains,profile_flops,cfg)
    write_json(output/"partial_budgets.json",dict(layer_profile_gains=gains,profile_flops=profile_flops,
        once_rebuild_flops=denominator,policies=repairs))
    learned=fit_learned(current,rows,roles,cm,cfg,device,output,fit_source=args.reuse_fits)
    del rows
    gc.collect()
    policies=repairs+learned
    fast_stats=None
    if cfg.get("evaluation_backend")=="policy_batch_graph":
        from max_baselines_2026_10.fast_baselines import evaluate_checkpointed
        def progress(value):
            print(json.dumps(dict(status="baseline_quality",**value)),flush=True)
            write_json(output/"progress.json",dict(status="baseline_quality",**value))
        identity=dict(config_sha=config_sha,execution_source_sha256=execution_sources,
            checkpoint_sha256=checkpoint_sha,requests_sha256=binding["requests_sha256"],
            population=roles["quality"],settings=cfg,
            policies=[{k:v for k,v in p.items() if k!="modules"} for p in policies])
        scores,fast_stats=evaluate_checkpointed(parent,current,histories,by_user,roles["quality"],policies,cfg,device,
            output,identity=identity,progress=progress)
        observed={p["tag"]:Counter(candidate_scores=fast_stats["stats"]["feedback_events"]) for p in policies}
    else:
        scores,observed=evaluate(parent,current,histories,by_user,roles["quality"],policies,cfg,device,output)
    summary=finish(scores,observed,policies,shapes,cm,denominator,cfg,output)
    summary.update(calibration_snapshot_transfer_bytes=transfer,torch_version=torch.__version__,
        cuda_device=torch.cuda.get_device_name(device),canary=args.canary,source_sha256=sha256(Path(__file__)),
        configuration_sha256=sha256(output/"configuration.json"),execution_progress_wall_seconds=time.perf_counter()-started,
        offline_evaluation_execution=fast_stats,completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/"summary.json",summary)
    print(json.dumps(dict(status="complete",output=str(output),results=summary["results"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--canary",action="store_true")
    parser.add_argument("--reuse-fits",type=Path)
    parser.add_argument("--resume",action="store_true",help="verify frozen inputs/code and reuse committed fits/user blocks")
    run(parser.parse_args())
