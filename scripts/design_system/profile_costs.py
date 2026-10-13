#!/usr/bin/env python3
"""Shared native event-shape arithmetic used by the current lazy cost ledger.

The old task-coordination profiling CLI has retired. These retained counters
keep the original reference arithmetic unchanged. No labels or scores enter.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np

from design_two.common import END, RELEASES
from hstu_kvcache.design_two.coordination import ReadCoordinator
from read_correction_2026_09.cost import CostModel, eager_read

START = RELEASES[0][1]


class ShapeCosts:
    def __init__(self, config, adapters, query_batch=32):
        self.config, self.adapters, self.query_batch = config, adapters, query_batch
        self.cap, self.layers = config["max_seq_len"], config["num_layers"]
        self.width, self.heads = config["hidden_size"], config["num_heads"]
        self.models = {backend: CostModel(self.width, self.layers, self.heads, backend)
                       for backend in ("torch", "triton")}
        self.coordinator = ReadCoordinator(self.layers, self.width, mode="static")

    def model(self, length):
        return self.models["triton" if length >= self.cap else "torch"]

    @lru_cache(None)
    def prefill(self, length):
        return self.model(length).full_cache(length)

    @lru_cache(None)
    def append(self, length, width):
        return self.model(length).band_append(length, width, window_size=self.cap)

    @lru_cache(None)
    def correction(self, version, length, candidates=1):
        return self.adapters[version].estimate_flops(tokens=length, candidates=candidates)

    @lru_cache(None)
    def query(self, length, candidates):
        return eager_read(self.model(length), length, queries=candidates)

    @lru_cache(None)
    def coordination(self, length, candidates):
        return self.coordinator.estimate_added_flops(tokens=length, heads=self.heads,
                                                    candidates=candidates)["total"]


def profile_user(uid, timestamps, requests, costs):
    """Shapes depend on chronological event/request times, never on K/V values."""
    cursor = int(np.searchsorted(timestamps, START, side="left"))
    length = min(cursor, costs.cap)
    initial = costs.prefill(length)
    by_version = defaultdict(Counter)
    for row in requests:
        by_version[row["version"]][int(row["query_timestamp"])] += 1
    releases = {}
    for index, (version, at) in enumerate(RELEASES):
        until = RELEASES[index+1][1] if index+1 < len(RELEASES) else END
        row = Counter(uid=int(uid), requests=sum(by_version[version].values()))
        row["exact_rebuild_flops"] = costs.prefill(length)
        row["initial_encode_flops"] = costs.correction(version, length)["token_transform"]
        active = length > 0
        remaining_support_appends = costs.layers*(costs.cap-1)+1 if active else 0
        row["lifecycle_initial_encode_flops"] = row["initial_encode_flops"] if active else 0
        row["initial_history_length"] = length

        def advance(timestamp):
            nonlocal cursor, length, active, remaining_support_appends
            stop = int(np.searchsorted(timestamps, timestamp, side="left"))
            while cursor < stop:
                width = 1 << (min(32, stop-cursor).bit_length()-1)
                row["native_append_flops"] += costs.append(length, width)
                row["appends"] += width
                row["support_token_layers"] += width*costs.layers
                encoded = costs.correction(version, width)["token_transform"]
                row["append_encode_flops"] += encoded
                cursor += width
                length = min(length+width, costs.cap)
                if active:
                    remaining_support_appends -= width
                    if remaining_support_appends <= 0:
                        active = False
                        row["causal_retirement_events"] += 1
                    else:
                        row["lifecycle_append_encode_flops"] += encoded

        for request_index, (timestamp, count) in enumerate(sorted(by_version[version].items())):
            if not at <= timestamp < until:
                raise RuntimeError("request timestamp does not belong to its release")
            advance(timestamp)
            row["fresh_per_request_history_flops"] += count*costs.prefill(length)
            if request_index == 0:
                row["first_request_rebuild_flops"] = costs.prefill(length)
            for offset in range(0, count, costs.query_batch):
                candidates = min(costs.query_batch, count-offset)
                row["native_query_flops"] += costs.query(length, candidates)
                work = costs.correction(version, length, candidates)
                row["extra_history_read_flops"] += work["extra_history_read"]
                row["response_flops"] += work["candidate_reads"]
                if active:
                    row["lifecycle_extra_history_read_flops"] += work["extra_history_read"]
                    row["lifecycle_response_flops"] += work["candidate_reads"]
                    row["coordination_flops"] += costs.coordination(length, candidates)
                    row["corrected_requests"] += candidates
                else:
                    row["natural_native_requests"] += candidates
        advance(until)  # Includes inactive users and the tail after their last query.
        row["final_history_length"] = length
        releases[version] = dict(row)
    return dict(uid=int(uid), initial_v2_flops=initial, releases=releases)


