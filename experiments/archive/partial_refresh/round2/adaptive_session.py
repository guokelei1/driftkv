"""Small, causal alternatives for the second refresh exploration.

Length is measured at the first read. Priority selectors spend a total-row
budget after mandatory refreshes; they never inspect the current query or
request labels. This budget rule is an experimental alternative to paper4.4's
tail-union rule, not silently the same policy.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from experiments.archive.partial_refresh.session import PartialRefreshSession, RefreshPlan


def priority_plan(versions, version, rho, scores):
    versions = np.asarray(versions)
    missing = versions < 0
    older = (versions < version - 1) & ~missing
    current = versions == version
    forced = missing | older
    budget = math.ceil(rho * len(versions))
    selected = forced & ~current
    eligible = np.flatnonzero(~selected & ~current)
    count = min(len(eligible), max(0, budget - int(selected.sum())))
    if count:
        # Stable ties select recent rows; only rank8-derived scores are scanned.
        chosen = eligible[np.lexsort((-eligible, -np.asarray(scores)[eligible]))[:count]]
        selected[chosen] = True
    return RefreshPlan(np.flatnonzero(selected), budget, int(missing.sum()),
        int(older.sum()), int(forced.sum()), int(current.sum()),
        int(((versions == version - 1) & ~selected).sum()))


def session_factory(strategy):
    encoder = strategy.get("encoder", "dense")
    if encoder == "sparse":
        from experiments.archive.partial_refresh.round2.sparse_session import SparseMaskedSession
        base = SparseMaskedSession
    elif encoder == "factored":
        from experiments.archive.partial_refresh.round2.factored_session import FactoredRefreshSession
        base = FactoredRefreshSession
    elif encoder == "dense":
        base = PartialRefreshSession
    else:
        raise ValueError(f"unknown explored encoder: {encoder}")
    gram_cache = {}
    shared = dict(projection_gram_flops=0)

    class AdaptiveSession(base):
        shared_selection_costs = shared

        def __init__(self, *args, **kwargs):
            if encoder == "factored":
                kwargs["cache_capacity"] = strategy.get("cache_capacity")
                kwargs["static_item_file"] = strategy.get("static_item_file")
            super().__init__(*args, **kwargs)

        def publish(self, *args, **kwargs):
            self.previous_adapter = self.adapter
            super().publish(*args, **kwargs)

        def prepare_read(self, at):
            self.advance(at)
            if self.pending_encode:
                version = f"v{self.trajectory.version}"
                n = self.cache.seq_len
                if strategy.get("tail_rows") is not None:
                    rows = strategy["tail_rows"].get(version, n)
                    self.rho = min(1., rows / max(n, 1))
                elif n >= strategy.get("min_partial_length", 0):
                    self.rho = float(strategy.get("long_rho", {}).get(version, 1.))
                else:
                    self.rho = float(strategy.get("short_rho", 1.))
            super().prepare_read(at)

        def _refresh_plan(self, rho=None):
            rho = self.rho if rho is None else rho
            if strategy.get("selection", "tail") == "tail":
                return super()._refresh_plan(rho)
            versions, version = self.code_versions, self.trajectory.version
            forced = (versions < version - 1)
            budget = math.ceil(rho * len(versions))
            if self.state is None or budget <= int(forced.sum()) or rho == 1.:
                return priority_plan(versions, version, rho, np.zeros(len(versions)))
            z = self.state.codes[:, 0]  # [layer, row, head, rank]
            l, n, h, r = z.shape
            selector = strategy["selection"]
            if selector == "code_energy":
                scores = z.square().sum((0, 2, 3))
                self.costs["selection_code_scan_flops"] += 2 * l * n * h * r
            elif selector == "projection_drift":
                key = (id(self.previous_adapter), id(self.adapter))
                if key not in gram_cache:
                    grams = []
                    for previous, current in zip(self.previous_adapter.layers, self.adapter.layers, strict=True):
                        dk = current.key_decoder - previous.key_decoder
                        dv = current.decoder - previous.decoder
                        grams.append(dk @ dk.transpose(-1, -2) + dv @ dv.transpose(-1, -2))
                    gram_cache[key] = torch.stack(grams)
                    d = self.adapter.head_dim
                    shared["projection_gram_flops"] += l*h*(4*r*r*d + 2*r*d + r*r)
                scores = torch.einsum("lnhr,lhrs,lnhs->n", z, gram_cache[key], z)
                self.costs["selection_code_scan_flops"] += n*(l*h*(2*r*r + 2*r) + l*h-1)
            else:
                raise ValueError(f"unknown explored selection: {selector}")
            return priority_plan(versions, version, rho, scores.cpu().numpy())

    return AdaptiveSession
