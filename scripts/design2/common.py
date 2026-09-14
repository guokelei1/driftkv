"""Frozen-C reads and source-only reconstruction of its original scene schedule."""

import copy
import json
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from design.data import ROOT
from design.diagnose_native_input import read_input
from design.diagnose_query_holdout import panels
from design.diagnose_summary_objective import install_layer
from design.run import append_events, append_events_many, cache_at_many, event_range, new_state, timed
from hstu_kvcache.adaptation.translator import QueryTranslator

CROOT = ROOT/"results/design/native_coverage384_01"
PROJECTION = ROOT/"results/design/mechanism_aggregate_factorial192_01"
COHORT = json.loads((ROOT/"results/design/native_coverage_cohort_01/cohort.json").read_text())
TARGETS = (1, 3, 4, 5)


def original_args():
    args = json.loads((ROOT/"results/design/v15_query_view64_01/configuration.json").read_text())
    args.update(query_affine=False, calibration_batch=16)
    return SimpleNamespace(**args)


def source_scenes(current, previous, history, uids, states, early, cutover, target, lifetime, ledger):
    """Same source schedule as make_scenes; no target-response/teacher construction.

    Current-Exact zero-control KV is an original fitting INPUT and is retained
    as source work. No teacher output or response is computed here.
    """
    args = original_args()
    scenes = []
    def add(uid, state, kind):
        scenes.append(SimpleNamespace(uid=uid, state=state, kind=kind))
    for uid in uids:
        add(uid, states[uid], "continuous")
        state, events, _ = early[uid]
        if events:
            state = copy.deepcopy(state)
            state.release(target, None)
            last = int(state.writer.events[-1][3])
            state = timed(lambda:append_events(current, state, events, last, 128), ledger, "early_source_replay")
        add(uid, state, "early_current_writes")
        if target > 1:
            # Inserted below after one shared grouped prefix computation.
            add(uid, None, "adjacent")
    if target > 1:
        values = timed(lambda:cache_at_many(previous, history, [(u, cutover) for u in uids], 16), ledger, "adjacent_source")
        adjacent = {u:new_state(kv, times, target-1, args) for u,(kv,times) in zip(uids, values, strict=True)}
        for s in scenes:
            if s.kind == "adjacent":
                s.state = adjacent[s.uid]
    requests = []
    for uid in uids:
        if uid not in lifetime:
            continue
        stamps = history.rows[uid][0]
        end = int(np.searchsorted(stamps, cutover, side="left"))
        points = list(dict.fromkeys(int(stamps[index]) for tail in (64,256,1024,2048,4096,6144)
                                   if (index := max(1024, end-tail)) < end))
        requests.extend((uid, t) for t in points)
    if requests:
        values = timed(lambda:cache_at_many(previous, history, requests, 16), ledger, "lifetime_source")
        pending, events, lasts = [], [], []
        for (uid, stamp),(kv,times) in zip(requests, values, strict=True):
            state = new_state(kv, times, target-1, args)
            state.release(target, None)
            pending.append(state)
            events.append(event_range(history, uid, stamp, cutover))
            lasts.append(int(times[-1]))
        replayed = timed(lambda:append_events_many(current, pending, events, lasts, 128, 16), ledger, "lifetime_source_replay")
        for (uid,_),state in zip(requests,replayed,strict=True):
            add(uid,state,"lifetime_current_writes")
    zero_uids = [u for u in uids if u in lifetime]
    if zero_uids:
        values = timed(lambda:cache_at_many(current,history,[(u,cutover) for u in zero_uids],16), ledger, "exact_control_source")
        for uid,(kv,times) in zip(zero_uids,values,strict=True):
            add(uid,new_state(kv,times,target,args),"exact_source_control")
    # Match per-UID ordinal order, including lifetime and identity at the end.
    ordinal = Counter()
    for s in scenes:
        s.ordinal = ordinal[s.uid]
        ordinal[s.uid] += 1
    return scenes


class FrozenC:
    def __init__(self, target, device):
        self.target = target
        self.mapper = QueryTranslator(target=target).to(device)
        self.projection = torch.load(PROJECTION/f"source_projection_mean_m{target}.pt", map_location=device, weights_only=True)
        self.parameters = torch.load(CROOT/f"translator_C_m{target}.pt", map_location=device, weights_only=True)

    def prepare(self, scenes):
        sources = [s.state.pack_source(self.target) for s in scenes]
        base = torch.stack([self.mapper.features(s) for s in sources])
        p = self.projection
        standardized = (base.double()-p["center"])/p["scale"]
        h = standardized@p["projection"]
        h = torch.cat((torch.ones_like(h[:, :1]), h), -1)
        active = self.mapper.active_layers(base)
        counts = torch.stack([s.count.sum() for s in sources])
        views = [install_layer(h, p["weights"], p["query_center"], p["query_scale"], active[:, l])
                 for l,p in enumerate(self.parameters)]
        b, a = torch.stack([v[0] for v in views], 1), torch.stack([v[1] for v in views], 1)
        return dict(latent=h, counts=counts, active=active, b=b, a=a,
                    source_norm=standardized.square().sum(-1).sqrt(), sources=sources)

    def read(self, model, cache, panel, prepared, indices):
        p = prepared
        return read_input(model, cache, panel, p["b"][indices], p["a"][indices], self.parameters,
                          p["counts"][indices], p["active"][indices])


def make_panel(history, scenes, indices, cutover, device, fitting=False):
    values = []
    for i in indices:
        s = scenes[i]
        last = int(s.state.writer.events[-1][3])
        assert last < cutover
        p = panels(history, s.uid, cutover, cutover-last, device)[0]
        if fitting:
            values.append(tuple(v[:, ::4] for v in p["fit64"]))
        else:
            # Sixteen distinct held-bank items, all at the actual cutover.
            values.append((p["held64"][0][:, :16], torch.full((1,16),float(cutover-last),device=device)))
    return tuple(torch.cat([v[j] for v in values]) for j in range(2))


def batches(scenes, size=4):
    groups = defaultdict(list)
    for i,s in enumerate(scenes):
        groups[s.state.cache.seq_len].append(i)
    return [v[j:j+size] for v in groups.values() for j in range(0,len(v),size)]
