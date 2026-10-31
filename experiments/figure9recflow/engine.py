"""One release, independent policies, persistent causal user timelines."""
from __future__ import annotations

import time
import numpy as np
import torch

from hstu_kvcache.models.kv_cache import HSTUKVCache
from experiments.recflow_full_population.code.model import RequestRanker
from .alignment import apply_map
from .cache_ops import (encode_cache, recompute_interval, recompute_tail,
    deviation_scores, select_positions, recompute_positions, append_behaviors, score_candidates)
from .common import ranking_metrics
from .cost import RecFlowCostModel, deployment_flops


def load_model(checkpoint, data, device, *, backend="torch"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("quality_use_permitted") is False:
        raise ValueError("Diagnostic recommendation weights cannot be used")
    model = RequestRanker(data / "vocabulary_max", payload["scale"], payload["proposal"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    model.checkpoint_metadata=dict(days=[int(x) for x in payload['days']],scale=payload['scale'])
    for block in model.backbone.blocks:
        block.attn.backend = backend
    return model


def tensors(features, behaviors, deltas, device):
    return (torch.tensor(np.array(features, copy=True), device=device).unsqueeze(0),
            torch.tensor(np.array(behaviors, copy=True), dtype=torch.long, device=device).unsqueeze(0),
            torch.tensor(np.array(deltas, copy=True), dtype=torch.float32, device=device).unsqueeze(0))


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class ReleaseEngine:
    def __init__(self, old, current, policies, maps, *, candidate_chunk=128,
                 backend="torch", append_backend=None, cuda_graphs=False):
        self.old, self.current, self.policies, self.maps = old, current, policies, maps
        self.device = next(current.parameters()).device
        self.chunk = candidate_chunk
        self.backend, self.append_backend = backend, append_backend or backend
        self.cost = RecFlowCostModel.from_model(current, attention_backend=backend)
        self.append_cost = RecFlowCostModel.from_model(current, attention_backend=self.append_backend)
        self.graph = None
        if cuda_graphs:
            from .graph_ops import GraphExecutor
            self.graph = GraphExecutor(current,method_count=len(policies),history_limit=1024,
                                       chunk_size=candidate_chunk,borrow_state=True)

    def set_backend(self, name):
        for block in self.current.backbone.blocks:
            block.attn.backend = name

    @torch.no_grad()
    def initialize(self, payload):
        f, b, d = tensors(payload.initial_history['features'],
                          payload.initial_history['behavior'].astype(np.int64)+1,
                          payload.initial_deltas, self.device)
        source = encode_cache(self.old, f, b, d, capture=True)
        rebuilt = encode_cache(self.current, f, b, d)
        old, n = source.cache, source.cache.seq_len
        selector = (deviation_scores(self.current, old, f, b, d) if any(
            p['method']=='deviation' and 0<p['tokens']<n for p in self.policies) else None)
        states, costs = [], []
        for policy in self.policies:
            kind = policy['method']
            if kind == 'reuse':
                cache, cost = old, 0
            elif kind == 'rebuild':
                cache, cost = rebuilt, self.cost.full_cache(n)
            elif kind == 'layer':
                start, end = policy['interval']
                if start == 0:
                    # Exact native current prefix is common to every prefix interval.
                    k, v = old.k.clone(), old.v.clone()
                    k[:end+1], v[:end+1] = rebuilt.k[:end+1], rebuilt.v[:end+1]
                    cache = HSTUKVCache(k, v, n)
                else:
                    cache = recompute_interval(self.current, old, f, b, d, (start, end),
                                               layer_inputs=source.layer_inputs)
                cost = self.cost.layer(n, start, end)
            elif kind == 'tail':
                cache = recompute_tail(self.current, old, f, b, d, policy['tokens'])
                cost = self.cost.tail(n, policy['tokens'])
            elif kind == 'deviation':
                if policy['tokens']>=n:
                    cache, cost = rebuilt, self.cost.full_cache(n)
                elif policy['tokens']==0:
                    cache, cost = old, 0
                else:
                    pos = select_positions(selector, policy['tokens'])
                    cache = recompute_positions(self.current, old, f, b, d, pos)
                    cost = self.cost.deviation_selector(n) + self.cost.selected(n, policy['tokens'])
            elif kind == 'kv_map':
                cache = apply_map(old, self.maps[policy['tag']])
                cost = deployment_flops(n, policy['width'], 2*self.cost.num_layers*self.cost.num_heads,
                                        self.cost.head_dim)
            else:
                raise ValueError(kind)
            states.append(cache)
            costs.append(cost)
        cache = HSTUKVCache(torch.cat([x.k for x in states], 1),
                            torch.cat([x.v for x in states], 1), n)
        starts = sorted({p['interval'][0] for p in self.policies if p['method']=='layer' and p['interval'][0]})
        boundary_bytes = sum(source.layer_inputs[i].numel()*source.layer_inputs[i].element_size() for i in starts)
        return cache, dict(repair_flops=costs, source_cache_flops=self.cost.full_cache(n),
                          once_rebuild_flops=self.cost.full_cache(n),
                          old_boundary_union_bytes=boundary_bytes,
                          initial_cache_bytes=old.k.numel()*old.k.element_size()*2)

    @torch.no_grad()
    def run_user(self, payload, *, save_top=True, save_scores=False):
        self.set_backend(self.backend)
        synchronize(self.device)
        begin = time.perf_counter()
        if not payload.requests and not len(payload.events):
            n=len(payload.initial_history)
            return dict(uid=payload.uid,records=[],top=[],full_scores=[],ledger=dict(
                repair_flops=[0]*len(self.policies),source_cache_flops=self.cost.full_cache(n),
                once_rebuild_flops=0,old_boundary_union_bytes=0,initial_cache_bytes=0,
                ordinary_append_flops_per_policy=0,ordinary_read_flops_per_policy=0,
                source_cache_is_common_sunk_cost=True,requests=0,candidates=0,append_events=0,
                initial_rows=n,final_rows=n,initialization_seconds=0,maintenance_and_read_seconds=0,
                total_seconds=time.perf_counter()-begin,has_history_consumer=False,
                post_last_query_append_events=0,no_query_append_events=0))
        state, ledger = self.initialize(payload)
        synchronize(self.device)
        initialized = time.perf_counter()
        records, top_data, full_scores = [], [], []
        events_timing={'append':[], 'read':[]}
        def mark_start(kind):
            if self.device.type=='cuda':
                pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                pair[0].record()
                events_timing[kind].append(pair)
                return pair[1]
            return None
        append_flops = read_flops = graph_padding_rows = 0
        request_count = candidate_count = append_count = 0
        # Transfer each user's complete behavior stream once, not per scalar row.
        ef, eb, ed = tensors(payload.events['features'], payload.events['behavior'].astype(np.int64)+1,
                             payload.event_deltas, self.device)
        event_cursor = 0
        for block in payload.timeline():
            if block.kind == 'append':
                end_event=mark_start('append')
                m = len(block.events)
                append_flops += self.append_cost.rolling_append(state.seq_len, m)
                if self.graph is not None and self.device.type=='cuda' and state.seq_len==1024:
                    # Graph performs embedding/frequency setup separately per event.
                    append_flops += (m-1)*3*self.append_cost.temporal_frequencies
                self.set_backend(self.append_backend)
                append_fn=self.graph.append if self.graph is not None else lambda *a: append_behaviors(self.current,*a)
                state = append_fn(state, ef[:,event_cursor:event_cursor+m],
                    eb[:,event_cursor:event_cursor+m], ed[:,event_cursor:event_cursor+m])
                if end_event is not None: end_event.record()
                event_cursor += m
                append_count += m
            else:
                for request in block.requests:
                    end_event=mark_start('read')
                    if state.seq_len != request.history_length:
                        raise RuntimeError('Persistent history length differs from causal input ledger')
                    features = torch.tensor(np.array(request.features, copy=True), device=self.device).unsqueeze(0)
                    delta = torch.tensor([request.query_delta], device=self.device)
                    if self.graph is not None:
                        gpu_scores=self.graph.score(state,features,delta)
                    else:
                        gpu_scores=score_candidates(self.current,state,features,delta,chunk_size=self.chunk)
                    if end_event is not None: end_event.record()
                    scores=gpu_scores.cpu().numpy()
                    metrics, top = ranking_metrics(request.candidate_ids, request.labels, scores)
                    positives = int((request.labels==1).sum())
                    records.append(dict(uid=payload.uid, rid=request.rid, ts=request.ts,
                        candidates=len(request.labels), positives=positives, metrics=metrics))
                    if save_top:
                        top_data.append(dict(rid=request.rid, ids=request.candidate_ids[top].copy(),
                            labels=request.labels[top].copy(), scores=np.take_along_axis(scores, top, axis=1)))
                    if save_scores:
                        full_scores.append(dict(rid=request.rid, ids=request.candidate_ids.copy(),
                                                labels=request.labels.copy(), scores=scores))
                    cost_queries=len(request.labels)
                    if self.graph is not None and self.device.type=='cuda' and state.seq_len==1024:
                        cost_queries=((cost_queries+self.chunk-1)//self.chunk)*self.chunk
                        graph_padding_rows+=cost_queries-len(request.labels)
                    read_flops += self.cost.read(state.seq_len, cost_queries, chunk_size=self.chunk)
                    request_count += 1
                    candidate_count += len(request.labels)
        if event_cursor != len(payload.events):
            raise RuntimeError('Full-day maintenance incomplete')
        synchronize(self.device)
        finish = time.perf_counter()
        ledger.update(ordinary_append_flops_per_policy=append_flops,
                      ordinary_read_flops_per_policy=read_flops,
                      source_cache_is_common_sunk_cost=True,
                      requests=request_count, candidates=candidate_count, append_events=append_count,
                      initial_rows=len(payload.initial_history), final_rows=state.seq_len,
                      has_history_consumer=True,
                      post_last_query_append_events=int((payload.events['ts']>=payload.requests[-1].ts).sum()) if payload.requests else 0,
                      no_query_append_events=len(payload.events) if not payload.requests else 0,
                      graph_padding_candidate_rows=graph_padding_rows,
                      initialization_seconds=initialized-begin, maintenance_and_read_seconds=finish-initialized,
                      total_seconds=finish-begin)
        ledger.update({kind+'_device_seconds':sum(a.elapsed_time(b) for a,b in pairs)/1000
                       for kind,pairs in events_timing.items()})
        if self.graph is not None:
            ledger['graph_cumulative_stats']=dict(self.graph.stats)
        return dict(uid=payload.uid, records=records, top=top_data, full_scores=full_scores, ledger=ledger)
