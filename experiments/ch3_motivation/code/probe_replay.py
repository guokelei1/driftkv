"""Independent-edge causal replay of a frozen Chapter 3 correction probe.

Candidate reads are transient. f is prepared once before the first consumer;
g/h are used by both scoring and behavior encoding. Native rolling band
append is used for f and after inherited rows have all been evicted.
"""
from __future__ import annotations

from collections import Counter
import copy

import numpy as np
import torch

from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.evaluation.trajectory import MAX_DELTA_SECONDS, NativeTrajectory
from hstu_kvcache.models import HSTUKVCache


class ProbeTrajectory:
    """Small single-user reference for the frozen probe's own persistent state."""

    def __init__(self, parent, current, probe, events, cfg, *, initial=None):
        at = int(cfg["release_day"]) * 86400
        self.cfg, self.model, self.probe = cfg, current, probe
        self.device = next(current.parameters()).device
        self.native = (NativeTrajectory(parent, int(cfg["edge"])-1, events, at,
                       max_length=int(cfg.get("history_length", 1024)),
                       append_band_size=int(cfg.get("append_band_size", 128)))
                       if initial is None else copy.copy(initial))
        if initial is not None:
            self.native.ledger = Counter(initial.ledger)
            self.native.producers = initial.producers.copy()
        if self.native.time != at:
            raise ValueError("shared initial trajectory must be exactly at cutover")
        self.native.switch(current, int(cfg["edge"]), at)
        self.source = self.native.cache
        self.source_ids = torch.as_tensor(self.native.retained_events[1], device=self.device,
                                           dtype=torch.long)[None]
        self.features = None
        self.prepared = False
        self.counts = Counter()
        self.read_shapes, self.append_shapes = Counter(), Counter()
        self.counts["initial_native_rows"] = self.native.cache.seq_len
        self.initial_length = self.native.cache.seq_len

    @property
    def cache(self):
        return self.native.cache

    def _prepare(self):
        if self.prepared:
            return
        # Reusable features are immutable publishing-time inputs. They are
        # sliced on FIFO eviction and never recomputed on a candidate read.
        self.features = self.probe.prepare_read_features(self.model, self.source, self.source_ids)
        self.native.cache = self.probe.prepare_cache(self.model, self.source, self.source_ids,
                                                     read_features=self.features)
        self.counts["prepare_calls"] += 1
        self.counts["prepare_rows"] += self.source.seq_len
        self.prepared = True

    def _slice_source(self, dropped):
        dropped = min(int(dropped), self.source.seq_len)
        self.source = HSTUKVCache(self.source.k[:, :, dropped:], self.source.v[:, :, dropped:],
                                  self.source.seq_len-dropped)
        self.source_ids = self.source_ids[:, dropped:]
        if self.features is not None:
            self.features = tuple(rows[:, :, dropped:] for rows in self.features)

    @torch.no_grad()
    def advance(self, until):
        until = int(until)
        if until < self.native.time:
            raise ValueError("probe trajectory cannot move backward")
        stop = int(np.searchsorted(self.native.events[0], until, side="left"))
        if self.native.cursor < stop:
            self._prepare()
        while self.native.cursor < stop:
            if self.probe.stage == "f" or self.source.seq_len == 0:
                # No dynamic inherited-row correction remains: the native
                # rolling-band path is the correct, much faster execution.
                before_ledger = Counter(self.native.ledger)
                before_evicted = self.native.evicted_tokens
                self.native.advance(until)
                for (action, version, n, width), calls in (self.native.ledger-before_ledger).items():
                    if action == "append":
                        self.append_shapes[n, width, 0] += calls
                        self.counts["native_band_append_calls"] += calls
                        self.counts["appended_rows"] += width*calls
                self._slice_source(self.native.evicted_tokens-before_evicted)
                return
            position = self.native.cursor
            timestamp = int(self.native.events[0][position])
            item = torch.as_tensor([[int(self.native.events[1][position])]], device=self.device)
            behavior = torch.as_tensor([[int(self.native.events[2][position])]], device=self.device)
            gap = (0 if self.native.last_timestamp is None else timestamp-self.native.last_timestamp)
            delta = torch.as_tensor([[min(MAX_DELTA_SECONDS, max(0, gap))]],
                                    device=self.device, dtype=torch.float32)
            before, inherited = self.cache.seq_len, self.source.seq_len
            cache, source, ids, _ = self.probe.append(self.model, self.cache, self.source, self.source_ids,
                item, behavior, delta, max_length=self.native.max_length, read_features=self.features)
            dropped = inherited-source.seq_len
            if self.features is not None:
                self.features = tuple(rows[:, :, dropped:] for rows in self.features)
            self.source, self.source_ids = source, ids
            self.native.cache = cache
            self.native.cursor += 1
            self.native.producers = np.r_[self.native.producers, self.native.version][-self.native.max_length:]
            self.native.ledger["append", self.native.version, before, 1] += 1
            self.native.appended_tokens += 1
            self.native.evicted_tokens += max(0, before+1-self.native.max_length)
            # append() evicts before reading; its scored history is cap-1.
            read_length = min(before, self.native.max_length-1)
            self.read_shapes["behavior", read_length, source.seq_len, 1] += 1
            self.append_shapes[before, 1, source.seq_len] += 1
            self.counts["corrected_append_calls"] += 1
            self.counts["appended_rows"] += 1
        self.native.time = until

    @torch.no_grad()
    def score(self, candidates, query_delta):
        self._prepare()
        queries = candidates.shape[1]
        if self.probe.stage == "f" or self.source.seq_len == 0:
            values, _ = native_score(self.model, self.cache, candidates, query_delta)
        else:
            values, _ = self.probe.score(self.model, self.source, self.source_ids, candidates, query_delta,
                                          prepared_cache=self.cache, read_features=self.features)
        self.read_shapes["query", self.cache.seq_len, self.source.seq_len, queries] += 1
        self.counts["candidate_reads"] += candidates.numel()
        self.counts["candidate_batches"] += 1
        return values

    def report(self):
        return dict(counts=dict(self.counts),
            initial_shapes=[dict(history_length=self.initial_length, calls=1)],
            prepare_shapes=[dict(history_length=self.initial_length, calls=1)] if self.prepared else [],
            read_shapes=[dict(kind=kind, history_length=n, inherited_rows=inherited, queries=q, calls=count)
                         for (kind,n,inherited,q), count in sorted(self.read_shapes.items())],
            append_shapes=[dict(before_length=n, width=width, inherited_rows=inherited, calls=count)
                           for (n,width,inherited),count in sorted(self.append_shapes.items())],
            source_rows_remaining=self.source.seq_len,
            scope="single independent edge; candidate reads transient; corrected behavior writes; full event tail replay")


@torch.no_grad()
def replay_probe_user(parent, current, probe, events, requests_by_timestamp, uid, cfg, *, tag=None):
    trajectory = ProbeTrajectory(parent, current, probe, events, cfg)
    result = []
    at, end = int(cfg["release_day"])*86400, int(cfg["end_day"])*86400
    for timestamp, requests in sorted(requests_by_timestamp.items()):
        if not at <= int(timestamp) < end:
            raise ValueError("request outside the frozen release evaluation window")
        trajectory.advance(timestamp)
        for start in range(0, len(requests), int(cfg.get("query_batch", 16))):
            rows = requests[start:start+int(cfg.get("query_batch", 16))]
            candidates = torch.as_tensor([[int(row["item_idx"]) for row in rows]], device=trajectory.device)
            gap = 0 if trajectory.native.last_timestamp is None else timestamp-trajectory.native.last_timestamp
            delta = torch.as_tensor([gap], device=trajectory.device, dtype=torch.float32)
            values = trajectory.score(candidates, delta)[0].cpu().tolist()
            result.extend(dict(row, uid=int(uid), method=tag or probe.stage, score=float(value))
                          for row,value in zip(rows,values,strict=True))
    trajectory.advance(end)
    return result, trajectory.report()


@torch.no_grad()
def evaluate_probe(parent, current, probe, histories, by_user, uids, cfg, device=None, *, tag=None,
                   on_user=None):
    """Evaluate a bounded driver shard; the driver commits returned scores."""
    scores, counts, reads, appends = [], Counter(), Counter(), Counter()
    initial, preparation = Counter(), Counter()
    for index, uid in enumerate(uids):
        values, report = replay_probe_user(parent,current,probe,histories.rows[uid],by_user.get(uid,{}),uid,cfg,tag=tag)
        scores.extend(values)
        counts.update(report["counts"])
        for row in report["initial_shapes"]:
            initial[row["history_length"]] += row["calls"]
        for row in report["prepare_shapes"]:
            preparation[row["history_length"]] += row["calls"]
        for row in report["read_shapes"]:
            reads[row["kind"],row["history_length"],row["inherited_rows"],row["queries"]] += row["calls"]
        for row in report["append_shapes"]:
            appends[row["before_length"],row["width"],row["inherited_rows"]] += row["calls"]
        if on_user is not None:
            on_user(index+1, len(uids), uid, report)
    return scores, dict(counts=dict(counts),
        initial_shapes=[dict(history_length=n,calls=count) for n,count in sorted(initial.items())],
        prepare_shapes=[dict(history_length=n,calls=count) for n,count in sorted(preparation.items())],
        read_shapes=[dict(kind=kind,history_length=n,inherited_rows=c,queries=q,calls=count)
                     for (kind,n,c,q),count in sorted(reads.items())],
        append_shapes=[dict(before_length=n,width=w,inherited_rows=c,calls=count)
                       for (n,w,c),count in sorted(appends.items())])
