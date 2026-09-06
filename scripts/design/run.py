#!/usr/bin/env python3
"""Small complete Design loop: calibration, paired reads and a real release chain.

Example: PYTHONPATH=src:scripts python scripts/design/run.py --run-id v0_canary \
  --fit-users 2 --dev-users 2 --trajectory-users 1 --steps 6 --targets 2 --tail-days 0
"""

# Timing callbacks run synchronously, within the defining loop iteration.
# ruff: noqa: E402, B023

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from design.data import (
    DAY,
    REQUEST_ROOT,
    SPLIT_PATH,
    V5_REQUEST_ROOT,
    calibration_candidates,
    diagnostic_admissions,
    fixed_split,
    frozen_model,
    histories,
    prefix,
    quality_requests,
    stratified_calibration,
)
from insight_two.common import (
    CUTOVER_DAYS,
    load_frozen_inputs,
    metrics_row,
    score_metrics,
)

from hstu_kvcache.adaptation import (
    AdaptationState,
    FunctionalTranslator,
    QueryTranslator,
    RidgeTranslator,
    Translator,
    inference,
)
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.adaptation.state import release_many
from hstu_kvcache.adaptation.summary import SummaryWriter
from hstu_kvcache.evaluation.binary_metrics import binary_metrics
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band, append_with_rolling_cap


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def timed(fn, ledger, key):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    ledger[key] += time.perf_counter() - start
    return result


def tensor(value, device, *, floating=False):
    return torch.as_tensor(value, dtype=torch.float32 if floating else torch.long, device=device)


def new_translator(args, target, device):
    if args.representation == "ridge":
        if getattr(args,"query_affine",False):
            return QueryTranslator(target=target,rank=args.rank,
                                   layer_clearance=args.layer_clearance).to(device)
        options = getattr(args, "target_options", {}).get(str(target), {})
        return RidgeTranslator(target=target,rank=args.rank,history_scope=getattr(args,"history_scope","old"),
                               temporal=getattr(args,"temporal",False),
                               layer_clearance=getattr(args,"layer_clearance",False),
                               temporal_source_rank=options.get("temporal_source_rank",getattr(args,"temporal_source_rank",0)),
                               response_strength=options.get("response_strength"),
                               source_confidence=getattr(args,"source_confidence",False),
                               second_moments=getattr(args,"second_moments",False),
                               source_kernel=getattr(args,"source_kernel",False)).to(device)
    if args.representation == "functional":
        return FunctionalTranslator(hidden=args.hidden, target=target).to(device)
    return Translator(hidden=args.hidden, target=target, context=args.context).to(device)


def new_state(cache, times, producer, args):
    options = dict(segment_size=1024, slots=1) if args.representation != "kv" else {}
    return AdaptationState(cache, times, producer,
        write_correction=getattr(args,"write_mode","shared") == "shared",
        history_scope=getattr(args,"history_scope","old"),
        second_moments=getattr(args,"second_moments",False), **options)


def event_range(history, uid, start, stop):
    ts, items, actions = history.rows[uid]
    left, right = np.searchsorted(ts, [start, stop], side="left")
    return list(zip(ts[left:right].tolist(), items[left:right].tolist(), actions[left:right].tolist(), strict=True))


def cache_at(model, history, uid, timestamp):
    items, actions, delta, times = prefix(history, uid, timestamp, next(model.parameters()).device)
    return model.compute_kv(items, actions, delta), times


@torch.no_grad()
def cache_at_many(model, history, requests, batch_size):
    """Existing Exact computation grouped by real prefix length, without padding."""
    device=next(model.parameters()).device
    groups=defaultdict(list)
    result=[None]*len(requests)
    for index,(uid,timestamp) in enumerate(requests):
        inputs=prefix(history,uid,timestamp,device)
        groups[inputs[0].shape[1]].append((index,inputs))
    for length,rows in groups.items():
        for offset in range(0,len(rows),batch_size):
            part=rows[offset:offset+batch_size]
            cache=inference.compute_prefix(model,*[torch.cat([value[i] for _,value in part]) for i in range(3)])
            for row,(index,inputs) in enumerate(part):
                # Independent calibration snapshots may be deep-copied later;
                # they must not each clone an entire backing batch allocation.
                result[index]=(HSTUKVCache(cache.k[:,row:row+1].contiguous(),
                                          cache.v[:,row:row+1].contiguous(),length),inputs[3])
    return result


def initialize_calibration(model, history, uids, cutover, args, ledger):
    current=timed(lambda:cache_at_many(model,history,[(uid,cutover) for uid in uids],args.calibration_batch),
                  ledger,"calibration_source_backfill")
    # A short prefix may have fewer than four events, or all events at the
    # same timestamp. Reuse its cutover scene if no nonempty earlier prefix
    # exists; never invent an empty source or a timestamp between tied events.
    snapshots=[(uid,int(times[-4]) if len(times)>4 and times[-4]>times[0] else cutover)
               for uid,(_,times) in zip(uids,current,strict=True)]
    before=timed(lambda:cache_at_many(model,history,snapshots,args.calibration_batch),ledger,"calibration_source_backfill")
    states,early={},{}
    for uid,(cache,times),(old,old_times),(_,snapshot) in zip(uids,current,before,snapshots,strict=True):
        states[uid]=timed(lambda:new_state(cache,times,0,args),ledger,"calibration_source_backfill")
        early[uid]=(timed(lambda:new_state(old,old_times,0,args),ledger,"calibration_source_backfill"),
                    event_range(history,uid,snapshot,cutover),snapshot)
    return states,early


def append_event(model, state, event, last_time):
    timestamp, item, action = event
    device = next(model.parameters()).device
    items, actions = tensor([[item]], device), tensor([[action]], device)
    delta = tensor([[min(7 * DAY, timestamp - last_time)]], device, floating=True)
    if isinstance(state, AdaptationState):
        state.append(model, items, actions, delta, timestamp)
        return state
    return append_with_rolling_cap(model, state, items, actions, delta, 1024)


def append_events(model, state, events, last_time, chunk_size):
    """Batch only intervals with no intervening observed candidate request."""
    device = next(model.parameters()).device
    stamps = [event[0] for event in events]
    items = tensor([[event[1] for event in events]], device)
    actions = tensor([[event[2] for event in events]], device)
    deltas = tensor([[min(7*DAY,b-a) for a,b in
        zip(([last_time]+stamps)[:-1],stamps,strict=True)]],device,floating=True)
    for start in range(0,len(events),chunk_size):
        stop = min(len(events),start+chunk_size)
        if isinstance(state,AdaptationState):
            state.append_native_chunk(model,items[:,start:stop],actions[:,start:stop],
                                      deltas[:,start:stop],stamps[start:stop])
        else:
            state = append_with_rolling_band(model,state,items[:,start:stop],actions[:,start:stop],
                                             deltas[:,start:stop],1024)
    return state


@torch.no_grad()
def append_events_many(model, states, event_lists, last_times, chunk_size, batch_size):
    """Batch independent historical calibration replays of identical real lengths.

    No CC requests occur inside these pre-release calibration tails. The native
    rolling kernel and each source writer retain their usual causal membership.
    """
    device = next(model.parameters()).device
    inputs = []
    for events,last in zip(event_lists,last_times,strict=True):
        stamps = [event[0] for event in events]
        inputs.append((tensor([[event[1] for event in events]],device),
            tensor([[event[2] for event in events]],device),
            tensor([[min(7*DAY,b-a) for a,b in zip(([last]+stamps)[:-1],stamps,strict=True)]],device,floating=True),
            stamps))
    positions = [0]*len(states)
    while True:
        groups = defaultdict(list)
        for i,(state,values,position) in enumerate(zip(states,inputs,positions,strict=True)):
            length = min(chunk_size,len(values[3])-position)
            if length:
                cache = state.cache if isinstance(state,AdaptationState) else state
                groups[cache.seq_len,length].append(i)
        if not groups:
            break
        for (cache_length,length),indices in groups.items():
            for offset in range(0,len(indices),batch_size):
                batch = indices[offset:offset+batch_size]
                caches = [states[i].cache if isinstance(states[i],AdaptationState) else states[i] for i in batch]
                combined = HSTUKVCache(torch.cat([c.k for c in caches],1),torch.cat([c.v for c in caches],1),cache_length)
                result = append_with_rolling_band(model,combined,
                    *[torch.cat([inputs[i][field][:,positions[i]:positions[i]+length] for i in batch]) for field in range(3)],1024)
                for row,i in enumerate(batch):
                    # Own each retained result: a finished short tail must not
                    # keep another user's full intermediate batch allocation alive.
                    part = HSTUKVCache(result.k[:,row:row+1].contiguous(),result.v[:,row:row+1].contiguous(),result.seq_len)
                    if isinstance(states[i],AdaptationState):
                        states[i].install_native_chunk(part,inputs[i][3][positions[i]:positions[i]+length])
                    else:
                        states[i] = part
                    positions[i] += length
    return states


@dataclass
class Scene:
    uid: int
    state: AdaptationState
    teacher: object
    candidates: torch.Tensor
    query_delta: torch.Tensor
    replay: list
    last_time: int
    teacher_scores: torch.Tensor | None


def make_scenes(current, history, uids, states, early_states, cutover, ledger, previous, target, args):
    scenes = []
    pending_lifetime = []
    device = next(current.parameters()).device
    need_logits = args.representation != "ridge" or args.score_steps > 0
    lifetime_points={}
    for uid in uids[:args.lifetime_users]:
        stamps=history.rows[uid][0]
        end=int(np.searchsorted(stamps,cutover,side="left"))
        lifetime_points[uid]=list(dict.fromkeys(int(stamps[index]) for tail in (64,256,1024,2048,4096,6144)
            if (index:=max(1024,end-tail))<end))
    # Reuse immutable teacher prefixes within this calibration call. A replay
    # returns a new cache, so the Current-Exact zero control can share its base.
    teachers,adjacent_sources,lifetime_sources=None,None,None
    if args.calibration_batch>1:
        lifetime_requests=[(uid,timestamp) for uid,points in lifetime_points.items() for timestamp in points]
        teacher_requests=list(dict.fromkeys([(uid,timestamp) for uid in uids
            for timestamp in (cutover,early_states[uid][2])]+lifetime_requests))
        values=timed(lambda:cache_at_many(current,history,teacher_requests,args.calibration_batch),ledger,"teacher_build")
        teachers=dict(zip(teacher_requests,values,strict=True))
        values=timed(lambda:cache_at_many(previous,history,lifetime_requests,args.calibration_batch),ledger,"calibration_lifetime_source")
        lifetime_sources=dict(zip(lifetime_requests,values,strict=True))
        if args.adjacent_calibration and target>1:
            requests=[(uid,cutover) for uid in uids]
            values=timed(lambda:cache_at_many(previous,history,requests,args.calibration_batch),ledger,"calibration_adjacent_source")
            adjacent_sources=dict(zip(requests,values,strict=True))
    for uid in uids:
        for state, replay, snapshot in ((states[uid], [], cutover), early_states[uid]):
            teacher,times=(teachers[uid,snapshot] if teachers is not None else
                           timed(lambda:cache_at(current,history,uid,snapshot),ledger,"teacher_build"))
            last = int(times[-1])
            if replay and args.replay_chunk > 1:
                teacher = timed(lambda:append_events(current,teacher,replay,last,args.replay_chunk),ledger,"teacher_replay")
                last = replay[-1][0]
            else:
                for event in replay:
                    teacher = timed(lambda: append_event(current, teacher, event, last), ledger, "teacher_replay")
                    last = event[0]
            candidates = tensor([calibration_candidates(history, uid, cutover)], device)
            query_delta = tensor([cutover - last], device, floating=True)
            teacher_scores = (timed(lambda: score(current, teacher, candidates, query_delta)[0].detach(),
                                   ledger, "teacher_query") if need_logits else None)
            scenes.append(Scene(uid, state, teacher, candidates, query_delta, replay,
                int(state.writer.events[-1][3]), teacher_scores))
        if args.adjacent_calibration and target > 1:
            # A separate legitimate homogeneous Parent cache, not a splice into
            # the continuous branch. The Current teacher uses the same prefix.
            base = scenes[-2]
            parent_cache,times=(adjacent_sources[uid,cutover] if adjacent_sources is not None else
                                timed(lambda:cache_at(previous,history,uid,cutover),ledger,"calibration_adjacent_source"))
            adjacent = new_state(parent_cache,times,target-1,args)
            scenes.append(Scene(uid,adjacent,base.teacher,base.candidates,base.query_delta,
                                [],int(times[-1]),base.teacher_scores))
    for uid,points in lifetime_points.items():
        for snapshot in points:
            cache,times=(lifetime_sources[uid,snapshot] if lifetime_sources is not None else
                         timed(lambda:cache_at(previous,history,uid,snapshot),ledger,"calibration_lifetime_source"))
            state = timed(lambda:new_state(cache,times,target-1,args),ledger,"calibration_lifetime_source")
            timed(lambda:state.release(target,None),ledger,"calibration_lifetime_source")
            teacher,_=(teachers[uid,snapshot] if teachers is not None else
                       timed(lambda:cache_at(current,history,uid,snapshot),ledger,"teacher_build"))
            events = event_range(history,uid,snapshot,cutover)
            last = int(times[-1])
            candidates = tensor([calibration_candidates(history,uid,cutover)],device)
            delta = tensor([cutover-events[-1][0]],device,floating=True)
            if args.calibration_batch > 1:
                scene = Scene(uid,state,teacher,candidates,delta,[],events[-1][0],None)
                scenes.append(scene)
                pending_lifetime.append((scene,events,last))
            else:
                state = timed(lambda:append_events(current,state,events,last,128),ledger,"calibration_lifetime_replay")
                teacher = timed(lambda:append_events(current,teacher,events,last,128),ledger,"teacher_replay")
                scores = timed(lambda:score(current,teacher,candidates,delta)[0].detach(),ledger,"teacher_query") if need_logits else None
                scenes.append(Scene(uid,state,teacher,candidates,delta,[],events[-1][0],scores))
        # Legitimate Current-Exact initialization supplies a zero-change control.
        teacher,times=(teachers[uid,cutover] if teachers is not None else
                       timed(lambda:cache_at(current,history,uid,cutover),ledger,"teacher_build"))
        state = timed(lambda:new_state(teacher,times,target,args),ledger,"calibration_lifetime_source")
        candidates = tensor([calibration_candidates(history,uid,cutover)],device)
        delta = tensor([cutover-int(times[-1])],device,floating=True)
        scores = timed(lambda:score(current,teacher,candidates,delta)[0].detach(),ledger,"teacher_query") if need_logits else None
        scenes.append(Scene(uid,state,teacher,candidates,delta,[],int(times[-1]),scores))
    if pending_lifetime:
        entries,event_lists,last_times = map(list,zip(*pending_lifetime,strict=True))
        replayed = timed(lambda:append_events_many(current,[s.state for s in entries],event_lists,last_times,128,args.calibration_batch),
                        ledger,"calibration_lifetime_replay")
        teachers = timed(lambda:append_events_many(current,[s.teacher for s in entries],event_lists,last_times,128,args.calibration_batch),
                         ledger,"teacher_replay")
        for scene,state,teacher in zip(entries,replayed,teachers,strict=True):
            scene.state,scene.teacher = state,teacher
            if need_logits:
                scene.teacher_scores = timed(lambda:score(current,teacher,scene.candidates,scene.query_delta)[0].detach(),ledger,"teacher_query")
    if args.temporal:
        for scene in scenes:
            gap=float(scene.query_delta.item())
            assert gap>=1
            candidates=scene.candidates.shape[1]
            values=[gap,1.,min(60.,gap),min(3600.,gap)]
            scene.candidates=scene.candidates.repeat(1,len(values))
            scene.query_delta=tensor([values],device,floating=True).repeat_interleave(candidates,dim=1)
    if getattr(args,"query_affine",False):
        for scene in scenes:
            gap=float(scene.query_delta.item())
            assert gap>=1 and scene.candidates.shape[1]==16
            scene.candidates=scene.candidates.repeat(1,4)
            values=(torch.linspace(0,1,64,device=device)*np.log(gap)).exp().round().clamp(1,gap)
            scene.query_delta=values[None]
    return scenes


def training_state(scene, model, translator):
    if not scene.replay:
        return scene.state
    # Replay real pre-release events with this optimizer step's Translator.
    state = copy.deepcopy(scene.state)
    state.release(translator.target, translator)
    last = scene.last_time
    if not state.write_correction:
        return append_events(model,state,scene.replay,last,128)
    for event in scene.replay:
        append_event(model, state, event, last)
        last = event[0]
    return state


def response_targets(model, result, cache, teacher, old):
    targets = []
    with torch.no_grad():
        for layer, (block, q) in enumerate(zip(model.blocks, result.queries, strict=True)):
            target = history_read(block.attn, q.detach(), teacher.k[layer, 0, old], teacher.v[layer, 0, old])
            source = history_read(block.attn, q.detach(), cache.k[layer, 0, old], cache.v[layer, 0, old])
            targets.append(target - source)
    return targets


def fit_translator(current, scenes, target, args, ledger):
    device = next(current.parameters()).device
    translator = new_translator(args, target, device)
    if isinstance(translator, RidgeTranslator):
        translator, records, scale = fit_ridge(current,scenes,translator,args)
        if args.score_steps:
            records.extend(refine_ridge(current,scenes,translator,args))
        return translator, records, scale
    sources = [scene.state.writer.pack(target) for scene in scenes[::2]]
    with torch.no_grad():
        values = torch.cat([source.payload.flatten(0, 1) for source in sources])
        translator.scale.copy_(values.square().mean((0, 3), keepdim=False).sqrt()[..., None].clamp_min(1e-3))
        scale_samples, score_scale_samples = [], []
        for scene in scenes[::2]:
            reuse_scores, result = score(current, scene.state.cache, scene.candidates, scene.query_delta, trace=True)
            score_scale_samples.append((reuse_scores - scene.teacher_scores).square().mean())
            targets = response_targets(current, result, scene.state.cache, scene.teacher, scene.state.writer.old_mask(target))
            scale_samples.append(torch.stack([value.square().mean() for value in targets]))
        response_scale = torch.stack(scale_samples).mean(0).sqrt().clamp_min(1e-3)
        score_scale = torch.stack(score_scale_samples).mean().sqrt().clamp_min(1e-3)
        if isinstance(translator, FunctionalTranslator):
            translator.response_scale.copy_(response_scale[:, None])
    optimizer = torch.optim.Adam(translator.parameters(), lr=args.learning_rate)
    records = []
    rng = np.random.default_rng(17 + target)
    order = rng.permutation(len(scenes))
    seen = 0
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        batch_rows = []
        for _ in range(args.batch_users):
            if args.batch_users > 1 and seen and seen % len(order) == 0:
                order = rng.permutation(len(scenes))
            scene = scenes[int(order[seen % len(order)])]
            seen += 1
            state = training_state(scene, current, translator)
            source = state.writer.pack(target)
            predicted = translator(source)
            observed_scores, result = score(current, state.cache, scene.candidates, scene.query_delta, source, predicted, trace=True)
            targets = response_targets(current, result, state.cache, scene.teacher, state.writer.old_mask(target))
            response_loss = torch.stack([((observed - wanted) / response_scale[layer]).square().mean()
                                        for layer, (observed, wanted) in enumerate(zip(result.corrections, targets, strict=True))]).mean()
            reconstruction = observed_scores.new_zeros(())
            if args.summary_weight:
                teacher_summary = state.writer.reference_summary(scene.teacher, target)
                reconstruction = (((predicted.payload - teacher_summary.payload) / translator.scale).square()
                                  * source.count[..., None, None, None]).sum() / (source.count.sum() * 6 * 2 * 192)
            score_loss = ((observed_scores - scene.teacher_scores) / score_scale).square().mean()
            loss = response_loss + args.summary_weight * reconstruction + args.score_weight * score_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite calibration loss at target {target}, step {step}")
            (loss / args.batch_users).backward()
            batch_rows.append(dict(uid=scene.uid, replay_events=len(scene.replay),
                loss=float(loss.detach()), response_loss=float(response_loss.detach()),
                score_loss=float(score_loss.detach()), reconstruction=float(reconstruction.detach())))
        grad = torch.nn.utils.clip_grad_norm_(translator.parameters(), 10.)
        if not torch.isfinite(grad):
            raise FloatingPointError("nonfinite Translator gradient")
        optimizer.step()
        if step % 20 == 0 or step == args.steps - 1:
            row = dict(target=target, step=step, uids=[row["uid"] for row in batch_rows],
                       gradient_norm=float(grad), **{key:float(np.mean([row[key] for row in batch_rows]))
                       for key in ("loss", "response_loss", "score_loss", "reconstruction", "replay_events")})
            records.append(row)
            print(json.dumps({"phase": "calibration", **row}), flush=True)
    return translator.eval(), records, response_scale.detach().cpu().tolist()


@torch.no_grad()
def fit_ridge(current, scenes, translator, args):
    if args.calibration_batch > 1:
        return fit_ridge_batched(current,scenes,translator,args)
    records = []
    for iteration in range(args.steps):
        features, rates = [], []
        for scene in scenes:
            state = training_state(scene,current,translator)
            source = state.pack_source(translator.target)
            translated = translator(source)
            _, trace = score(current,state.cache,scene.candidates,scene.query_delta,
                             source,translated,trace=True)
            changes = response_targets(current,trace,state.cache,scene.teacher,
                                       state.response_mask(translator.target))
            features.append(translator.features(source))
            rates.append(torch.stack([value.mean(2).reshape(-1) for value in changes])/source.count.sum())
        record = translator.fit(torch.stack(features),torch.stack(rates))
        record.update(target=translator.target,pass_index=iteration,calibration_scenes=len(scenes),
                fitting="shared ridge at current corrected queries; rank fixed before development",
                history_scope=args.history_scope)
        records.append(record)
        print(json.dumps(dict(phase="ridge_calibration",**record)),flush=True)
    return translator.eval(),records,[]


@torch.no_grad()
def fit_ridge_batched(current, scenes, translator, args):
    """Equivalent native-write calibration grouped by actual cache length.

    Real writes do not consume the mapper, so their states/features are constant
    across the shared refits. Every pass still recomputes corrected queries.
    """
    if getattr(translator,"query_affine",False):
        from design.fit_query_view import fit_query_layers
        return fit_query_layers(current,scenes,translator,args)
    states=[training_state(scene,current,translator) for scene in scenes]
    sources=[state.pack_source(translator.target) for state in states]
    features=torch.stack([translator.features(source) for source in sources])
    time_source=translator.fit_time_source(features) if translator.temporal_source_rank else None
    counts=torch.stack([source.count.sum() for source in sources])
    masks=[state.response_mask(translator.target) for state in states]
    time_groups=4 if translator.temporal else 1
    candidates_per_time=scenes[0].candidates.shape[1]//time_groups
    if translator.temporal:
        phi=torch.stack([current.temporal_enc.features(scene.query_delta[:,::candidates_per_time])[0] for scene in scenes])
        fit_features=translator.with_time(features[:,None,:].expand(-1,time_groups,-1),phi).flatten(0,1)
    else:
        fit_features=features
    groups=defaultdict(list)
    for index,state in enumerate(states):
        groups[state.cache.seq_len].append(index)
    records=[]
    for iteration in range(args.steps):
        rates=torch.zeros(len(scenes),time_groups,translator.layers,translator.width,device=features.device)
        for indices in groups.values():
            for start in range(0,len(indices),args.calibration_batch):
                batch=indices[start:start+args.calibration_batch]
                student=HSTUKVCache(torch.cat([states[i].cache.k for i in batch],1),
                    torch.cat([states[i].cache.v for i in batch],1),states[batch[0]].cache.seq_len)
                teacher=HSTUKVCache(torch.cat([scenes[i].teacher.k for i in batch],1),
                    torch.cat([scenes[i].teacher.v for i in batch],1),student.seq_len)
                mask=torch.stack([masks[i] for i in batch])
                delta=translator.rates(features[batch])*counts[batch,None,None]
                temporal=translator.time_coefficients(features[batch])
                if temporal is not None:
                    temporal=temporal*counts[batch,None,None,None]
                if translator.history_scope == "all":
                    rates[batch]=inference.calibration_rates(current,student,teacher,
                        torch.cat([scenes[i].candidates for i in batch]),torch.cat([scenes[i].query_delta for i in batch]),
                        counts[batch],delta,temporal,time_groups)
                    continue
                _,trace=score(current,student,torch.cat([scenes[i].candidates for i in batch]),
                    torch.cat([scenes[i].query_delta for i in batch]),trace=True,response_delta=delta,response_time_delta=temporal)
                for layer,(block,q) in enumerate(zip(current.blocks,trace.queries,strict=True)):
                    wanted=history_read(block.attn,q,teacher.k[layer],teacher.v[layer],count=mask)
                    actual=history_read(block.attn,q,student.k[layer],student.v[layer],count=mask)
                    values=(wanted-actual).reshape(len(batch),block.attn.num_heads,time_groups,candidates_per_time,block.attn.head_dim)
                    rates[batch,:,layer]=values.mean(3).transpose(1,2).flatten(2)/counts[batch,None,None]
        record=translator.fit(fit_features,rates.flatten(0,1))
        record.update(target=translator.target,pass_index=iteration,calibration_scenes=len(scenes),
                      history_scope=args.history_scope,calibration_batch=args.calibration_batch,
                      query_time_groups=time_groups,
                      time_source_conditioning=time_source,
                      fitting="shared ridge at recomputed corrected queries; native-write states fixed across passes")
        records.append(record)
        print(json.dumps(dict(phase="ridge_calibration",**record)),flush=True)
    return translator.eval(),records,[]


def refine_ridge(current, scenes, translator, args):
    """Calibrate a shared decoder against closed-loop scores, then fold it in.

The source projection and rank remain fixed. A normalized residual makes one
learning rate meaningful across layers; inference retains the same two maps.
All examples and teacher states belong to calibration users before release.
"""
    device = translator.decode.device
    with torch.no_grad():
        latents, response_moments, score_moments = [], [], []
        for scene in scenes:
            state = training_state(scene,current,translator)
            source = state.pack_source(translator.target)
            x = (translator.features(source)-translator.center)/translator.input_scale
            latents.append(x @ translator.encode)
            reuse, trace = score(current,state.cache,scene.candidates,scene.query_delta,trace=True)
            changes = response_targets(current,trace,state.cache,scene.teacher,
                                       state.response_mask(translator.target))
            response_moments.append(torch.stack([v.square().mean() for v in changes]))
            score_moments.append((reuse-scene.teacher_scores).square().mean())
        latent_scale = torch.stack(latents).std(0).clamp_min(1e-3)
        response_scale = torch.stack(response_moments).mean(0).sqrt().clamp_min(1e-3)
        score_scale = torch.stack(score_moments).mean().sqrt().clamp_min(1e-3)
        # The original fitted decoder supplies a stable per-layer rate scale.
        rate_scale = translator.offset.reshape(translator.layers,-1).square().mean(1).sqrt().clamp_min(1e-7)
        rate_scale = rate_scale.repeat_interleave(translator.width)
        base_decode, base_offset = translator.decode.clone(), translator.offset.clone()
    residual = torch.nn.Parameter(torch.zeros(translator.rank+1,translator.layers*translator.width,device=device))
    optimizer = torch.optim.Adam([residual],lr=args.learning_rate)
    rng = np.random.default_rng(1017+translator.target)
    order, seen, records = rng.permutation(len(scenes)), 0, []

    def install():
        translator.decode = base_decode + residual[:-1] / latent_scale[:,None] * rate_scale
        translator.offset = base_offset + residual[-1] * rate_scale

    for step in range(args.score_steps):
        optimizer.zero_grad(set_to_none=True)
        batch_rows = []
        for _ in range(args.batch_users):
            if seen and seen % len(order) == 0:
                order = rng.permutation(len(scenes))
            scene = scenes[int(order[seen % len(order)])]
            seen += 1
            # Build a fresh graph for each accumulated example. Persisted event
            # states are detached; their forward replay uses this same mapper.
            install()
            state = training_state(scene,current,translator)
            source = state.pack_source(translator.target)
            observed, trace = score(current,state.cache,scene.candidates,scene.query_delta,
                                    source,translator(source),trace=True)
            changes = response_targets(current,trace,state.cache,scene.teacher,
                                       state.response_mask(translator.target))
            response_loss = torch.stack([((actual-wanted)/response_scale[layer]).square().mean()
                for layer,(actual,wanted) in enumerate(zip(trace.corrections,changes,strict=True))]).mean()
            score_loss = ((observed-scene.teacher_scores)/score_scale).square().mean()
            loss = score_loss + 0.1*response_loss + 1e-4*residual.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite shared decoder refinement loss")
            (loss/args.batch_users).backward()
            batch_rows.append(dict(score_loss=float(score_loss.detach()),response_loss=float(response_loss.detach())))
        grad = torch.nn.utils.clip_grad_norm_([residual],10.)
        if not torch.isfinite(grad):
            raise FloatingPointError("nonfinite shared decoder refinement gradient")
        optimizer.step()
        if step % 20 == 0 or step == args.score_steps-1:
            row = dict(target=translator.target,step=step,gradient_norm=float(grad),
                **{key:float(np.mean([r[key] for r in batch_rows])) for key in ("score_loss","response_loss")})
            records.append(row)
            print(json.dumps(dict(phase="ridge_score_refinement",**row)),flush=True)
    with torch.no_grad():
        install()
        translator.decode = translator.decode.detach()
        translator.offset = translator.offset.detach()
    return records


@torch.no_grad()
def evaluate_cutover(current, previous, translator, history, uid, cutover, panel, target, ledger, args):
    old, times = timed(lambda: cache_at(previous, history, uid, cutover), ledger, "diagnostic_parent_build")
    exact, _ = timed(lambda: cache_at(current, history, uid, cutover), ledger, "diagnostic_exact_build")
    state = timed(lambda: new_state(old, times, target - 1, args), ledger, "summary_backfill")
    timed(lambda: state.release(target, translator), ledger, "release_translation")
    device = next(current.parameters()).device
    candidates = tensor(panel[None], device)
    query_delta = tensor([cutover - int(times[-1])], device, floating=True)
    reuse = timed(lambda: score(current, old, candidates, query_delta)[0], ledger, "reuse_read")
    reference = timed(lambda: score(current, exact, candidates, query_delta)[0], ledger, "reference_read")
    learned = timed(lambda: state.score(current, candidates, query_delta), ledger, "paired_read")
    oracle_summary = state.writer.reference_summary(exact, target)
    oracle = score(current, old, candidates, query_delta, state.source, oracle_summary)[0]
    rows = []
    for name, logits in (("reuse", reuse), ("exact_summary_diagnostic", oracle), ("learned", learned)):
        rows.append(dict(uid=uid, target=target, method=name, **metrics_row(score_metrics(reference, reuse, logits))))
    raw = dict(uid=uid, target=target, candidates=panel.tolist(), exact=reference.cpu().tolist(),
               reuse=reuse.cpu().tolist(), learned=learned.cpu().tolist(), oracle=oracle.cpu().tolist())
    source_view_bytes = sum(value.numel() * value.element_size()
                            for value in (state.source.payload, state.source.count,
                                          state.source.ordinal, state.source.timestamp, state.source.producer))
    if state.source.second_moment is not None:
        source_view_bytes += state.source.second_moment.numel()*state.source.second_moment.element_size()
    translated_bytes=state.translated.payload.numel()*4
    if state.translated.temporal_coefficients is not None:
        translated_bytes+=state.translated.temporal_coefficients.numel()*4
    if state.translated.query_coefficients is not None:
        translated_bytes+=state.translated.query_coefficients.numel()*4
    return rows, raw, state.writer.storage_bytes(), int(translated_bytes), source_view_bytes


def correctness_canary(current, previous, history, uid, cutover, panel, args):
    old, times = cache_at(previous, history, uid, cutover)
    exact, _ = cache_at(current, history, uid, cutover)
    device = next(current.parameters()).device
    candidates = tensor(panel[None, :8], device)
    delta = tensor([cutover - int(times[-1])], device, floating=True)
    state = new_state(old, times, 0, args)
    identity = new_translator(args, 1, device)
    state.release(1, identity)
    native = current.score_cc_reuse(old, candidates, delta)
    ours = state.score(current, candidates, delta)
    torch.testing.assert_close(ours, native, atol=2e-4, rtol=2e-5)
    # With one slot per event, the paired difference must reconstruct Exact.
    full_writer = SummaryWriter.from_cache(old, times, 0, slots=64)
    with torch.no_grad():
        diagnostic = score(current, old, candidates, delta,
            full_writer.pack(1), full_writer.reference_summary(exact, 1))[0]
        exact_scores = current.score_cc_reuse(exact, candidates, delta)
    torch.testing.assert_close(diagnostic, exact_scores, atol=2e-4, rtol=2e-5)
    source = state.writer.pack(1)
    translated = identity(source)
    # Closed-form Translators have buffers rather than optimizer parameters;
    # this check is specifically the differentiable reader's payload-to-Q path.
    if not translated.payload.requires_grad:
        translated.payload.requires_grad_()
    translated.payload.retain_grad()
    _, result = score(current, old, candidates, delta, source, translated, trace=True)
    # Only upper-layer Q supervises this check: nonzero lower payload gradient
    # proves that the actual six-layer query feedback path wasn't detached.
    result.queries[-1].square().mean().backward()
    lower_gradient = float(translated.payload.grad[:, :, 0].norm())
    if not lower_gradient > 0:
        raise AssertionError("lower-layer correction cannot influence final-layer query")
    event = event_range(history, uid, cutover, (CUTOVER_DAYS[1]) * DAY)
    batch_difference = None
    if event:
        last = int(times[-1])
        reference = append_event(current, old, event[0], last)
        before_revision = state.counts["refreshes"]
        append_event(current, state, event[0], last)
        torch.testing.assert_close(state.cache.k, reference.k, atol=2e-4, rtol=2e-5)
        torch.testing.assert_close(state.cache.v, reference.v, atol=2e-4, rtol=2e-5)
        torch.testing.assert_close(state.writer.pack(1).payload,
                                   state.writer.reference_summary(state.cache, 1).payload,
                                   atol=2e-5, rtol=2e-5)
        if not state.write_correction:
            assert state.view_dirty
            state.score(current,candidates,delta)
            assert not state.view_dirty
        assert state.counts["refreshes"] == before_revision + 1
    if len(event)>1 and args.replay_chunk==128 and args.calibration_batch>1:
        # Two genuine prefixes of the same canary history exercise unequal
        # replay tails without inventing events or changing their order.
        tails = [[],event[:64],event[:129]]
        initial = new_state(old,times,0,args)
        initial.release(1,identity)
        scalar = [append_events(current,copy.deepcopy(initial),tail,int(times[-1]),128) for tail in tails]
        batched = append_events_many(current,[copy.deepcopy(initial) for _ in tails],tails,
                                     [int(times[-1])]*len(tails),128,args.calibration_batch)
        batch_difference = 0.
        for a,b in zip(scalar,batched,strict=True):
            for field in ("k","v"):
                left,right = getattr(a.cache,field),getattr(b.cache,field)
                torch.testing.assert_close(left,right,atol=2e-5,rtol=2e-5)
                batch_difference = max(batch_difference,float((left-right).abs().max()))
            assert list(a.writer.events)==list(b.writer.events) and a.counts==b.counts
            torch.testing.assert_close(a.pack_source().payload,b.pack_source().payload,atol=2e-5,rtol=2e-5)
    return dict(passed=True, identity_max_abs=float((ours-native).abs().max()),
                exact_delta_max_abs=float((diagnostic-exact_scores).abs().max()),
                lower_payload_to_final_query_gradient=lower_gradient,
                real_append_checked=bool(event), native_batch_max_abs=batch_difference,
                checkpoint_operator=current.cfg.activation)


def replay_calibration(current, history, states, uids, start, stop, ledger, chunk_size=1, batch_size=1):
    early = {}
    if chunk_size>1 and batch_size>1:
        before,after,snapshots,lasts=[],[],[],[]
        for uid in uids:
            events=event_range(history,uid,start,stop)
            snapshot=events[max(0,len(events)-4)][0] if events else stop
            snapshots.append(snapshot)
            before.append([event for event in events if event[0]<snapshot])
            after.append([event for event in events if event[0]>=snapshot])
            lasts.append(int(states[uid].writer.events[-1][3]))
        values=timed(lambda:append_events_many(current,[states[uid] for uid in uids],before,lasts,chunk_size,batch_size),
                     ledger,"calibration_lineage_replay")
        for uid,state,events,snapshot in zip(uids,values,after,snapshots,strict=True):
            states[uid]=state
            early[uid]=(copy.deepcopy(state),events,snapshot)
        values=timed(lambda:append_events_many(current,[states[uid] for uid in uids],after,
            [int(states[uid].writer.events[-1][3]) for uid in uids],chunk_size,batch_size),ledger,"calibration_lineage_replay")
        states.update(zip(uids,values,strict=True))
        return early
    for uid in uids:
        events = event_range(history, uid, start, stop)
        snapshot = events[max(0, len(events) - 4)][0] if events else stop
        last = int(states[uid].writer.events[-1][3])
        if chunk_size > 1:
            before = [event for event in events if event[0] < snapshot]
            after = [event for event in events if event[0] >= snapshot]
            if before:
                states[uid] = timed(lambda:append_events(current,states[uid],before,last,chunk_size),
                                    ledger,"calibration_lineage_replay")
                last = before[-1][0]
            early[uid] = (copy.deepcopy(states[uid]),after,snapshot)
            if after:
                states[uid] = timed(lambda:append_events(current,states[uid],after,last,chunk_size),
                                    ledger,"calibration_lineage_replay")
            continue
        saved = False
        for event in events:
            if not saved and event[0] >= snapshot:
                early[uid] = (copy.deepcopy(states[uid]), [e for e in events if e[0] >= snapshot], snapshot)
                saved = True
            timed(lambda: append_event(current, states[uid], event, last), ledger, "calibration_lineage_replay")
            last = event[0]
        if not saved:
            early[uid] = (copy.deepcopy(states[uid]), [], stop)
    return early


def replay_development(current, history, chains, requests, start, stop, target, ledger, chunk_size=1):
    raw = []
    device = next(current.parameters()).device
    for uid, branches in chains.items():
        events = event_range(history, uid, start, stop)
        grouped_events, grouped_requests = defaultdict(list), defaultdict(list)
        for event in events:
            grouped_events[event[0]].append(event)
        for request in requests.get(uid, []):
            grouped_requests[int(request["query_timestamp"])].append(request)
        last = int(branches["learned"].writer.events[-1][3])
        pending_events = []

        def flush_writes():
            nonlocal last
            if not pending_events:
                return
            for name in ("reuse","exact","learned"):
                branches[name] = timed(lambda:append_events(current,branches[name],pending_events,last,chunk_size),
                                       ledger,"service_"+name+"_append")
            last = pending_events[-1][0]
            pending_events.clear()

        for timestamp in sorted(set(grouped_events) | set(grouped_requests)):
            rows = grouped_requests[timestamp]
            if rows:
                flush_writes()
                assert last < timestamp, "same-timestamp writes leaked into query prefix"
                candidates = tensor([[int(row["item_idx"]) for row in rows]], device)
                delta = tensor([timestamp - last], device, floating=True)
                with torch.no_grad():
                    scores = {}
                    for name in ("reuse", "exact", "learned"):
                        fn = (lambda: branches[name].score(current, candidates, delta)) if name == "learned" else (
                             lambda: inference.score_ready(current, branches[name], candidates, delta))
                        scores[name] = timed(fn, ledger, "service_" + name + "_read")[0].cpu().tolist()
                old = branches["learned"].writer.old_mask(target)
                for index, row in enumerate(rows):
                    raw.append(dict(uid=uid, target=target, timestamp=timestamp,
                        request_id=row["request_id"], label=int(row["label"]),
                        old_events=int(old.sum()), covered=branches["learned"].covered,
                        correction_active=branches["learned"].correction_active,
                        writes_since_release=branches["learned"].writes_since_release,
                        cleared_layers=(min(len(current.blocks),branches["learned"].writes_since_release//current.cfg.max_seq_len)
                                        if not branches["learned"].write_correction else None),
                        **{name: values[index] for name, values in scores.items()}))
            if chunk_size > 1:
                pending_events.extend(grouped_events[timestamp])
            else:
                for event in grouped_events[timestamp]:
                    for name in ("reuse", "exact", "learned"):
                        branches[name] = timed(lambda: append_event(current, branches[name], event, last),
                                                ledger, "service_" + name + "_append")
                    last = event[0]
        flush_writes()
        print(json.dumps(dict(phase="trajectory", uid=uid, target=target, events=len(events),
                               requests=sum(len(rows) for rows in grouped_requests.values()),
                               counts=branches["learned"].counts)), flush=True)
    return raw


@torch.no_grad()
def warm_compiled_reads(model, branch, candidates, delta):
    """Compile the actual read regimes and check that saved views own their data."""
    state = copy.copy(branch["learned"])
    state.counts = dict(state.counts)
    records = []
    retained = []
    timings = defaultdict(float)
    for size in (1, 2, 4, 8, 16, 3, 17):
        ids = candidates[:, :size]
        for name in ("reuse", "exact"):
            cache = branch[name]
            expected = score(model, cache, ids, delta)[0]
            actual = timed(lambda:inference.score_ready(model, cache, ids, delta),timings,"native_read")
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            records.append(dict(path=name, candidates=size, maximum=float((actual-expected).abs().max())))
        expected = state.score(model, ids, delta, use_compiled=False)
        for dirty in (False, True, False):
            state.view_dirty = dirty
            actual = timed(lambda:state.score(model, ids, delta),timings,"dirty_read" if dirty else "paired_read")
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            records.append(dict(path="dirty" if dirty else "ready", candidates=size,
                                maximum=float((actual-expected).abs().max())))
            retained.append((actual, actual.clone(), state.response_delta, state.response_delta.clone(),
                             state.response_time_delta, state.response_time_delta.clone()))
    for values in retained:
        for actual, expected in zip(values[::2], values[1::2], strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    return dict(passed=True, comparisons=records, published_views_do_not_alias_graph_outputs=True,
                measured_call_seconds=dict(timings),
                timing_scope="actual compile/graph/warmup calls; numerical reference/check overhead remains in the enclosing warmup ledger")


def main(args):
    args.no_op_targets = getattr(args,"no_op_targets",[]) or []
    out = ROOT / "results/design" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    ledger = defaultdict(float)
    calibration_reference = None
    if args.evaluation_from:
        calibration_path = ROOT/"results/design"/args.evaluation_from
        fitted = json.loads((calibration_path/"configuration.json").read_text())
        args.no_op_targets = sorted(set(args.no_op_targets) | set(fitted.get("no_op_targets",[])))
        assert json.loads((calibration_path/"summary.json").read_text())["status"] == "development_complete"
        assert args.targets <= fitted["targets"]
        for field in ("representation","context","rank","hidden"):
            setattr(args,field,fitted[field])
        args.write_mode = fitted.get("write_mode","shared")
        args.history_scope = fitted.get("history_scope","old")
        args.temporal = fitted.get("temporal",False)
        args.temporal_source_rank = fitted.get("temporal_source_rank",0)
        args.source_confidence = fitted.get("source_confidence",False)
        args.second_moments = fitted.get("second_moments",False)
        args.source_kernel = fitted.get("source_kernel",False)
        args.query_affine = fitted.get("query_affine",False)
        args.target_options = fitted.get("target_options", {})
        args.layer_clearance = args.layer_clearance or fitted.get("layer_clearance",False)
        args.fit_users = args.steps = args.score_steps = 0
        args.lifetime_users = 0
        calibration_reference = dict(run=args.evaluation_from,
            configuration_sha256=hashlib.sha256((calibration_path/"configuration.json").read_bytes()).hexdigest(),
            translator_sha256={str(target):hashlib.sha256((calibration_path/f"translator_v{target}.pt").read_bytes()).hexdigest()
                for target in range(1,args.targets+1) if target not in args.no_op_targets},
            cost_scope="teacher, fitting and calibration lineage remain in this referenced run; not zero method preparation")
    assert args.replay_chunk == 1 or args.write_mode == "reuse", "batched replay requires native writes"
    assert not args.no_op_targets or (args.write_mode=="reuse" and 1 not in args.no_op_targets
        and not args.compile_reads and set(args.no_op_targets)<=set(range(1,args.targets+1)))
    assert args.history_scope == "old" or (args.write_mode == "reuse" and args.representation == "ridge")
    assert not args.layer_clearance or (args.history_scope == "all" and args.write_mode == "reuse")
    assert not args.temporal_source_rank or args.temporal
    assert not args.source_confidence or (args.representation == "ridge" and (args.evaluation_from or args.calibration_batch > 1))
    assert not args.second_moments or args.representation == "ridge"
    assert not getattr(args,"source_kernel",False) or (args.representation == "ridge" and args.temporal
        and not args.source_confidence and not args.second_moments and not args.temporal_source_rank
        and (args.evaluation_from or args.fit_users <= 64))
    assert not getattr(args,"query_affine",False) or (args.representation=="ridge" and args.history_scope=="all"
        and args.write_mode=="reuse" and not args.temporal and not args.source_confidence
        and not args.second_moments and not getattr(args,"source_kernel",False) and not args.score_steps
        and (args.evaluation_from or args.calibration_batch>1))
    assert not args.lifetime_users or (args.history_scope == "all" and args.replay_chunk == 128)
    assert args.calibration_batch == 1 or (args.write_mode == "reuse" and args.representation == "ridge")
    assert args.release_batch == 1 or (args.write_mode == "reuse" and args.representation == "ridge")
    assert not args.temporal or (args.write_mode == "reuse" and args.representation == "ridge" and not args.score_steps
                                and (args.calibration_batch>1 or args.evaluation_from))
    assert not args.compile_reads or (args.evaluation_from and args.temporal and args.history_scope=="all"
                                     and args.release_batch==16 and args.trajectory_users>0)
    split_path = Path(args.split_path) if args.split_path else SPLIT_PATH
    split = json.loads(split_path.read_text()) if args.split_path else fixed_split()
    assert not args.stratified_calibration or args.split_path
    fit_uids = (stratified_calibration(split,args.fit_users) if args.stratified_calibration
                else split["calibration"][:args.fit_users])
    dev_uids = split["development"][args.dev_offset:args.dev_offset+args.dev_users]
    chain_uids = dev_uids[:args.trajectory_users]
    assert len(dev_uids) == args.dev_users and len(chain_uids) == args.trajectory_users
    assert not set(fit_uids) & (set(dev_uids) | set(split["confirmation"]))
    assert not set(dev_uids) & set(split["confirmation"])
    assert not args.quality_only or (args.evaluation_from and chain_uids == dev_uids)
    if args.evaluation_from:
        assert not set(fitted["calibration_uids"]) & set(dev_uids)
    final_day = CUTOVER_DAYS[args.targets - 1] + args.tail_days
    inputs = yaml.safe_load((ROOT / "configs/contracts/yambda500m_medium_hstu_native_insight1_locality_v1.yaml").read_text())["frozen_inputs"]
    admissions = diagnostic_admissions(args.targets)
    files = (sorted((ROOT / "src/hstu_kvcache/adaptation").glob("*.py"))
             + sorted((ROOT / "src/hstu_kvcache/models").glob("*.py"))
             + sorted((ROOT / "scripts/design").glob("*.py")))
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for file in files:
            archive.add(file, arcname=str(file.relative_to(ROOT)))
    patch = subprocess.check_output(["git", "diff"], cwd=ROOT)
    (out / "worktree.patch").write_bytes(patch)
    configuration = dict(vars(args), command=sys.argv, frozen_backbone_seed=17, optimizer_seed=17,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={str(file.relative_to(ROOT)):hashlib.sha256(file.read_bytes()).hexdigest() for file in files},
        split_path=str(split_path.resolve().relative_to(ROOT)), split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        calibration_uids=fit_uids, development_uids=dev_uids, trajectory_uids=chain_uids,
        adaptation_targets=[target for target in range(1,args.targets+1) if target not in args.no_op_targets],
        no_op_scope="declared research scope; native model release, writes and source maintenance continue; no Translator fitting or correction at these targets",
        reserved_confirmation_count=len(split["confirmation"]), confirmation_read=False,
        checkpoint_inputs=inputs["checkpoints"], checkpoint_seals=inputs["checkpoint_seals"],
        admission_inputs=admissions, dataset_input=inputs["dataset"], item_mapping_input=inputs["item_mapping"],
        quality_request_manifests={str((path/"manifest.json").relative_to(ROOT)):
            hashlib.sha256((path/"manifest.json").read_bytes()).hexdigest()
            for path in ([REQUEST_ROOT,V5_REQUEST_ROOT] if args.targets==5 else [REQUEST_ROOT])},
        v5_partial_tail=args.targets==5 and final_day>300,
        actual_operator="legacy ELU+1, no sequence denominator, SiLU gate, inclusive self",
        calibration_reference=calibration_reference,
        summary_segment_size=1024 if args.representation != "kv" else 64,
        summary_slots=1 if args.representation != "kv" else 2,
        result_scope="development synchronous-ready mechanism; one backbone seed; not confirmation",
        supervision="pre-release calibration-only summary and same-query old-event response; teacher detached",
        initial_estimate="120 s fixed + 0.06 s/calibration example + 0.007 s/calibration event + 0.016 s/development event",
        estimate_basis="v0_short_01 and v0_score_01: 0.035-0.041 s/example, about 0.005 s/calibration event and 0.010 s/three-branch event; rounded up",
        cost_target_population=30000, cost_estimates_are_extrapolated=True)
    if args.replay_chunk > 1:
        configuration["initial_estimate"] = "120s fixed + 0.06s/calibration example + 0.0005s/calibration event + 0.001s/development event"
        configuration["estimate_basis"] = "v3_rolling_band_canary_01: 2353 real events, maximum CC score difference 7.2e-7; conservative allowance for request boundaries"
    if args.history_scope == "all":
        configuration["supervision"] = "pre-release calibration-only full-history response, including native Current descendants; actual query, detached teacher"
        configuration["lifetime_calibration"] = dict(users=min(args.lifetime_users,args.fit_users),
            tails=[64,256,1024,2048,4096,6144], source="independent Parent-Exact at historical prefix, then native Current replay",
            zero_control="independent Current-Exact initialization; no fabricated lifetime or event",
            refresh_dependency="all retained source plus writes-since-release; invalidate on every write, refresh before CC")
    if args.temporal:
        configuration["temporal_view"] = dict(basis="unchanged native 16-frequency sin/cos features",
            coefficients="shared target-specific, conditioned on actual producer masses",
            calibration_offsets=["cutover_gap",1,"min(60,gap)","min(3600,gap)"],
            causality="all query timestamps after last visible event and no later than cutover; same prefix, no future events",
            read_dependency="actual query time weights installed coefficients; time alone does not invalidate source view")
    if args.layer_clearance:
        configuration["layer_clearance_policy"] = dict(
            rule="zero layer j intercept and temporal response after (j+1)*1024 actual native writes since release",
            full_clearance="after6144 writes skip correction and refresh; continue writer maintenance; reset on next release",
            evidence="results/design/v6_clearance_probe_01/summary.json",
            scope="six frozen layers, cap1024, native writes; no labels, teacher or learned scheduling")
    if args.temporal_source_rank:
        configuration["temporal_source_conditioning"] = dict(rank=args.temporal_source_rank,
            basis="PCA of standardized calibration source inputs; whiten retained directions; no response or label access",
            features="producer masses plus source coordinates, each multiplied by the same native Fourier basis",
            cost="source projection and larger coefficient map counted in fitting, publication and every dirty read")
    if args.source_confidence:
        configuration["source_confidence_rule"] = dict(
            rule="min(1,sqrt(calibration_source_squared_norm_q95/current_source_squared_norm))",
            coordinates="source inputs standardized by this release's fitting users; exclude query-time features",
            application="same source-dependent strength on intercept and temporal coefficients; recompute after source changes",
            selection="fixed rule and 0.95 quantile, no model-ID branch or evaluation-label tuning",
            cost="norm and scaling included in fitting, publication and every dirty read")
    if args.second_moments:
        configuration["summary_second_moments"] = dict(
            writer="FP32 actual per-segment sums of K,V,K²,V²; add/subtract the retained real contributions",
            translator="producer-weighted means and central diagonal variances; no new teacher input at evaluation",
            variance="max(0, weighted_second_moment-weighted_mean²/producer_fraction)",
            cost="additional writer updates, retained second sums, expanded fitting and dirty-read features all counted")
    if getattr(args,"source_kernel",False):
        configuration["source_kernel_rule"] = dict(
            kernel="Gaussian source-state similarity plus producer-conditioned linear Fourier time block",
            bandwidth="median squared distance between distinct standardized fitting states, recomputed per release",
            intercept="free intercept with centered source kernel; time view remains factored at publication",
            support="at most384 fitting states for current64/16 budget; shared across users, no evaluation teacher input",
            cost="shared support tensors, fitting, publication and every dirty read include kernel evaluation")
    if getattr(args,"query_affine",False):
        configuration["query_affine_view"] = dict(
            correction="per-layer/per-head b + actual Q @ A, count-weighted; self unchanged",
            fitting="one progressive pass over six layers; lower shared maps frozen before collecting the next layer's queries",
            query_normalization="shared release/layer/head coordinate means and scales from fitting queries; folded into published b/A",
            probes="64 paired item/time queries, same16 existing items repeated4, log-spaced legal integer gaps",
            source="unchanged producer means, masses and actual release age; per-layer rank32 maps",
            cost="larger shared maps and query matrices counted; eager prototype, no compiled-service claim")
    configuration["initial_estimate"] += "; lifetime scenes add 7 examples/pass and at most 13632 replay events/user/target at 0.0004s/event"
    if args.temporal:
        configuration["initial_estimate"] += "; calibration example count is multiplied by four query-time groups"
    if args.temporal_source_rank and not args.evaluation_from:
        configuration["initial_estimate"] += "; source PCA and time conditioning add a30s fitting allowance (v7_source_time_canary_01:29.12s whole16/4 run)"
    if args.compile_reads:
        configuration["compiled_reads"] = dict(mode="reduce-overhead", dynamic=True, fullgraph=True,
            torch_version=torch.__version__, cxx=os.environ.get("CXX","g++"),
            scope="same native and learned CC math; graph outputs cloned before publishing or retaining",
            candidate_buckets=[1,2,4,8,16],
            candidate_padding="repeat first transient candidate, discard padded outputs; groups over16 split; all three branches identical",
            startup="all measured compilation/warmup separately retained; existing Inductor disk cache may be reused")
        configuration["initial_estimate"] += "; compiled inference adds a conservative 600s startup allowance"
    write_json(out / "configuration.json", configuration)
    try:
        device = torch.device("cuda:0")
        torch.set_num_threads(4)
        torch.manual_seed(17)
        np.random.seed(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        history = timed(lambda: histories(fit_uids + dev_uids, max(final_day, 246)), ledger, "history_io")
        if args.quality_only:
            # Transient, label-free canary candidates only. Real quality below
            # always consumes the original explicit-feedback request stream.
            all_panels = np.asarray([[calibration_candidates(history,uid,day*DAY,count=64)
                for uid in dev_uids[:1]] for day in CUTOVER_DAYS[:args.targets]])
        else:
            assert dev_uids == fixed_split()["development"][:args.dev_users]
            _, all_panels, _ = load_frozen_inputs()
        counts = {uid: len(event_range(history, uid, CUTOVER_DAYS[0]*DAY, final_day*DAY))
                  for uid in fit_uids + chain_uids}
        calibration_examples = args.steps*args.targets*(2*args.fit_users if args.representation=="ridge" else args.batch_users)
        if args.adjacent_calibration and args.representation == "ridge":
            calibration_examples += args.steps*(args.targets-1)*args.fit_users
        calibration_examples += args.score_steps*args.batch_users*args.targets
        calibration_examples += args.steps*args.targets*min(args.lifetime_users,args.fit_users)*7
        if args.temporal:
            calibration_examples *= 4
        estimate = (120 + calibration_examples*0.06
                    + args.targets*min(args.lifetime_users,args.fit_users)*13632*0.0004
                    + sum(counts[uid] for uid in fit_uids)*(0.0005 if args.replay_chunk>1 else 0.007)
                    + sum(counts[uid] for uid in chain_uids)*(0.001 if args.replay_chunk>1 else 0.016))
        if args.temporal_source_rank and not args.evaluation_from:
            estimate += 30
        if args.compile_reads:
            estimate += 600
        print(json.dumps(dict(phase="resource_probe", event_counts=counts, estimated_seconds=estimate)), flush=True)
        write_json(out / "resource_estimate.json", dict(estimated_seconds=estimate, replay_events=counts,
            calculation=configuration["initial_estimate"], basis=configuration["estimate_basis"],
            execution="direct" if estimate <= 1800 else "tmux required before model execution"))
        if estimate > 1800 and not args.detached:
            raise RuntimeError("probe projects >30 min; retain this probe and launch a new detached run")
        previous = timed(lambda: frozen_model(0, device), ledger, "model_io")
        chains = {}
        cutover = CUTOVER_DAYS[0]*DAY
        states,early=initialize_calibration(previous,history,fit_uids,cutover,args,ledger)
        for uid in chain_uids:
            cache, times = cache_at(previous, history, uid, cutover)
            state = timed(lambda: new_state(cache,times,0,args),ledger,"chain_initial_summary_backfill")
            chains[uid] = dict(learned=state, reuse=cache, exact=cache)
        cutover_rows, cutover_raw, losses, quality, releases = [], [], [], [], []
        summary_bytes = view_bytes = source_view_bytes = None
        for target in range(1, args.targets + 1):
            current = timed(lambda: frozen_model(target, device), ledger, "model_io")
            cutover = CUTOVER_DAYS[target - 1]*DAY
            stop = (CUTOVER_DAYS[target] if target < args.targets else final_day)*DAY
            if target == 1:
                checks = correctness_canary(current, previous, history, (fit_uids or dev_uids)[0], cutover, all_panels[0, 0], args)
                write_json(out / "canary.pass.json", checks)
                print(json.dumps({"phase":"correctness", **checks}), flush=True)
            if target in args.no_op_targets:
                translator = None
            elif args.evaluation_from:
                translator = new_translator(args,target,device)
                saved = torch.load(calibration_path/f"translator_v{target}.pt",map_location=device,weights_only=False)
                translator.load_state_dict(saved["state_dict"])
                translator.supported_producers = set(saved["supported_producers"])
                translator.eval()
            else:
                scenes = make_scenes(current, history, fit_uids, states, early, cutover, ledger, previous, target, args)
                translator, fit_rows, response_scale = timed(
                    lambda scenes=scenes: fit_translator(current, scenes, target, args, ledger), ledger, "translator_fit")
                losses.extend(fit_rows)
                torch.save(dict(state_dict=translator.state_dict(), hidden=args.hidden, target=target, context=args.context,
                                representation=args.representation,
                                rank=args.rank,supported_producers=sorted(translator.supported_producers),
                                configuration_sha256=hashlib.sha256((out/"configuration.json").read_bytes()).hexdigest()),
                           out / f"translator_v{target}.pt")
                del scenes
            for index, uid in enumerate([] if args.quality_only or target in args.no_op_targets else dev_uids):
                rows, raw, summary_bytes, view_bytes, source_view_bytes = evaluate_cutover(current, previous, translator, history, uid,
                    cutover, all_panels[target-1, index], target, ledger, args)
                cutover_rows.extend(rows)
                cutover_raw.append(raw)
            for state in states.values():
                state.release(target, translator)
            if args.release_batch > 1:
                for offset in range(0,len(chain_uids),args.release_batch):
                    users=chain_uids[offset:offset+args.release_batch]
                    timed(lambda:release_many([chains[uid]["learned"] for uid in users],target,translator),
                          ledger,"chain_release_translation")
                    rebuilt=timed(lambda:cache_at_many(current,history,[(uid,cutover) for uid in users],args.release_batch),
                                  ledger,"chain_exact_release")
                    for uid,(cache,_) in zip(users,rebuilt,strict=True):
                        chains[uid]["exact"]=cache
            for uid, branch in chains.items():
                if args.release_batch == 1:
                    timed(lambda: branch["learned"].release(target, translator), ledger, "chain_release_translation")
                    branch["exact"], _ = timed(lambda: cache_at(current, history, uid, cutover), ledger, "chain_exact_release")
                if args.quality_only:
                    continue
                # Observe the inherited states at every cutover, before any real append.
                candidates = tensor(all_panels[target-1, dev_uids.index(uid)][None], device)
                query_delta = tensor([cutover - branch["learned"].writer.events[-1][3]], device, floating=True)
                with torch.no_grad():
                    ref = score(current, branch["exact"], candidates, query_delta)[0]
                    reuse = score(current, branch["reuse"], candidates, query_delta)[0]
                    observed = branch["learned"].score(current, candidates, query_delta, use_compiled=False)
                releases.append(dict(uid=uid, target=target, producer_counts={str(p):sum(
                    branch["learned"].writer.segments[sid]["producer"] == p for sid, _, _, _ in branch["learned"].writer.events)
                    for p in range(target+1)}, **metrics_row(score_metrics(ref,reuse,observed))))
            if stop > cutover:
                if args.compile_reads:
                    if not inference.enabled():
                        inference.enable()
                    first = chains[chain_uids[0]]
                    ids = tensor(all_panels[target-1,0][None],device)
                    delta = tensor([cutover-first["learned"].writer.events[-1][3]],device,floating=True)
                    checked = timed(lambda:warm_compiled_reads(current,first,ids,delta),ledger,"inference_compilation_and_warmup")
                    compiled_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
                    write_json(out/f"compiled_v{target}.pass.json",checked)
                    print(json.dumps(dict(phase="compiled_inference_warmup",target=target,**checked)),flush=True)
                if target < args.targets:
                    early = replay_calibration(current, history, states, fit_uids, cutover, stop, ledger,
                                               args.replay_chunk,args.calibration_batch)
                request_rows = quality_requests(chain_uids, cutover, stop)
                quality.extend(replay_development(current, history, chains, request_rows, cutover, stop, target, ledger,args.replay_chunk))
                if args.compile_reads:
                    compiled_after = torch._dynamo.utils.counters["stats"]["unique_graphs"]
                    write_json(out/f"compiled_service_v{target}.json",dict(
                        graphs_before=compiled_before,graphs_after=compiled_after,
                        additional_graphs_during_service=compiled_after-compiled_before,
                        scope="Dynamo graph count; CUDA graph recording may still occur without a new Dynamo graph"))
            print(json.dumps(dict(phase="edge_complete", target=target,
                learned_probability_recovery=(None if args.quality_only or target in args.no_op_targets else float(np.mean([r["probability_gap_recovery"] for r in cutover_rows if r["target"]==target and r["method"]=="learned"]))),
                ledger=dict(ledger))), flush=True)
            # Preserve each completed edge even if a later edge fails.
            pd.DataFrame(cutover_rows).to_csv(out/"cutover_metrics.csv", index=False)
            write_json(out/"cutover_raw.json", cutover_raw)
            write_json(out/"losses.json", losses)
            write_json(out/"chain_releases.json", releases)
            pd.DataFrame(quality).to_parquet(out/"quality_raw.parquet", index=False)
            previous = current
        pd.DataFrame(cutover_rows).to_csv(out/"cutover_metrics.csv", index=False)
        write_json(out/"cutover_raw.json", cutover_raw)
        write_json(out/"losses.json", losses)
        write_json(out/"chain_releases.json", releases)
        pd.DataFrame(quality).to_parquet(out/"quality_raw.parquet", index=False)
        mechanisms = [] if args.quality_only else pd.DataFrame(cutover_rows).groupby(["target","method"])[[
            "probability_gap_recovery","observed_probability_gap","reuse_probability_gap"]].mean().reset_index().to_dict("records")
        quality_summary = []
        for target in range(1, args.targets+1):
            rows = [row for row in quality if row["target"]==target]
            if rows:
                quality_summary.append(dict(target=target, requests=len(rows), users=len({r['uid'] for r in rows}),
                    metrics={name:binary_metrics(np.array([r["label"] for r in rows]), np.array([r[name] for r in rows]))
                             for name in ("exact","reuse","learned")}))
        summary = dict(status="development_complete", configuration_sha256=hashlib.sha256((out/"configuration.json").read_bytes()).hexdigest(),
            canary=checks, mechanisms=mechanisms, quality=quality_summary, chain_releases=releases,
            ledger_seconds=dict(ledger), elapsed_seconds=time.perf_counter()-start,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
            peak_reserved_mib=torch.cuda.max_memory_reserved()/(1<<20),
            last_user_source_storage_bytes=summary_bytes, last_user_translated_storage_bytes=view_bytes,
            last_user_source_view_bytes=source_view_bytes,
            storage_scope="tensor and logical metadata bytes; excludes Python allocator/container overhead",
            state_counts={str(uid):branch['learned'].counts for uid,branch in chains.items()},
            model_seed_repeats=1, confirmation_read=False,
            limitations=["small development cohort", "synchronous ready views; no waiting-period coverage claim",
                         "no population cost qualification", "legacy ELU+1 frozen checkpoint semantics"])
        write_json(out/"summary.json", summary)
        print(json.dumps({key:summary[key] for key in ("status","quality","elapsed_seconds","peak_allocated_mib")}), flush=True)
    except Exception as exc:
        write_json(out/"summary.json", dict(status="failed", error=repr(exc), ledger_seconds=dict(ledger),
                                          elapsed_seconds=time.perf_counter()-start))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fit-users", type=int, default=32)
    parser.add_argument("--dev-users", type=int, default=8)
    parser.add_argument("--split-path", help="Frozen derived development/calibration split; never confirmation selection")
    parser.add_argument("--stratified-calibration", action="store_true", help="Sample the calibration pool by pre-release population history length")
    parser.add_argument("--dev-offset", type=int, default=0)
    parser.add_argument("--quality-only", action="store_true", help="Frozen trajectory evaluation without repeated static diagnostics")
    parser.add_argument("--trajectory-users", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-users", type=int, default=1)
    parser.add_argument("--targets", type=int, default=2, choices=(1,2,3,4,5))
    parser.add_argument("--tail-days", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--context", choices=("cumulative", "local"), default="cumulative")
    parser.add_argument("--representation", choices=("kv", "functional", "ridge"), default="kv")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--adjacent-calibration", action="store_true")
    parser.add_argument("--score-steps", type=int, default=0)
    parser.add_argument("--write-mode", choices=("shared","reuse"), default="shared")
    parser.add_argument("--evaluation-from", help="Completed Design run whose fixed Translators are evaluated without refitting")
    parser.add_argument("--replay-chunk",type=int,choices=(1,128),default=1)
    parser.add_argument("--history-scope",choices=("old","all"),default="old")
    parser.add_argument("--lifetime-users",type=int,default=0)
    parser.add_argument("--calibration-batch",type=int,default=1)
    parser.add_argument("--release-batch",type=int,choices=(1,16),default=1)
    parser.add_argument("--temporal",action="store_true")
    parser.add_argument("--layer-clearance",action="store_true")
    parser.add_argument("--temporal-source-rank",type=int,choices=(0,8),default=0)
    parser.add_argument("--source-confidence",action="store_true")
    parser.add_argument("--second-moments",action="store_true")
    parser.add_argument("--source-kernel",action="store_true")
    parser.add_argument("--query-affine",action="store_true")
    parser.add_argument("--no-op-targets",type=int,nargs="+",default=[],choices=(2,3,4,5),
                        help="Declared native No-op releases; keep their model and real cache lineage, skip Translator fitting/reads")
    parser.add_argument("--compile-reads",action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--summary-weight", type=float, default=0.1)
    parser.add_argument("--score-weight", type=float, default=0.)
    parser.add_argument("--detached", action="store_true")
    main(parser.parse_args())
