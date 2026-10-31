#!/usr/bin/env python3
"""Lazy compact codes with learned Key and Value read corrections.

The shared codes change attention weights and values in one history scan.
Native writes and the established lazy code lifecycle remain unchanged.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import torch

from experiments.archive.compact_read import (
    evaluate_lightweight as base, evaluate_lightweight_fullitem_lazy as lazy,
)
from experiments.archive.compact_read.lightweight_lazy import LazyLightweightSession
from experiments.archive.compact_read.profile_lightweight_lazy import lazy_candidate_release, OUTPUT
from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.methods.cachetide.compact.lightweight_kv_kernel import make_history_read
from hstu_kvcache.common.artifacts import sha256


class LazyKVSession(LazyLightweightSession):
    def read_callback(self, candidates, *, force_torch=False):
        if self.state is None:
            return None
        if (not force_torch and self.kernel == "vector" and self.cache.k.is_cuda
                and self.cache.k.dtype == torch.float32 and self.adapter.heads == 6
                and self.adapter.head_dim == 32 and self.adapter.rank == 8
                and self.adapter.response_hidden_width == 0
                and 0 < self.cache.seq_len <= 1024 and 0 < candidates.shape[1] <= 16):
            # Explicitly use both decoders. The inherited callback only
            # implements Value correction and must not handle this candidate.
            return make_history_read(self.adapter, self.state)
        return self.adapter.make_history_read(self.state)


def candidate_release(counts, *, layers, heads, head_dim, rank, response_width):
    if response_width != 0:
        raise ValueError("this Key/Value candidate has no Response")
    row = lazy_candidate_release(counts, layers=layers, heads=heads,
        head_dim=head_dim, rank=rank, response_width=response_width)
    queries = counts["lazy_corrected_requests"]
    pairs = counts["lazy_corrected_token_query_pairs"]
    # q Dk^T is per query; its product with z and addition to q K^T
    # are per query-history-row. Both branches share the existing codes.
    added = dict(query_key_projection=layers * queries * 2 * heads * head_dim * rank,
                 query_key_low_rank_logits=layers * pairs * 2 * heads * rank,
                 query_key_logit_add=layers * pairs * heads)
    row["components_flops"].update(added)
    extra = sum(added.values())
    row["variable_flops"] += extra
    row["extra_query_flops"] += extra
    row["variable_percent_of_once_rebuild"] = 100 * row["variable_flops"] / row["first_request_rebuild_flops"]
    row["extra_percent_of_native_whole_query"] = 100 * row["extra_query_flops"] / row["native_whole_query_flops"]
    row["extra_percent_of_native_history_attention"] = 100 * row["extra_query_flops"] / row["native_history_attention_flops"]
    row["shared_work_budget_at_20_percent_flops"] -= extra
    row["key_decoder_shared_parameters"] = layers * heads * rank * head_dim
    return row


def execution_sources():
    names = ("experiments/archive/compact_read/evaluate_lightweight_fullitem_kv.py",
        "src/hstu_kvcache/methods/cachetide/compact/lightweight_fullitem_kv.py",
        "src/hstu_kvcache/methods/cachetide/compact/lightweight_kv_kernel.py")
    return {**lazy.execution_sources(), **{name:sha256(ROOT / name) for name in names}}


def bind_candidate():
    lazy.bind_candidate()
    base.LightweightValueAdapter = LightweightFullItemKVAdapter
    base.LightweightSession = LazyKVSession
    base.candidate_release = candidate_release
    base.execution_sources = execution_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=base.PANEL)
    parser.add_argument("--reference", type=Path, default=base.REFERENCE)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--population-cost", type=Path, default=OUTPUT / "summary.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--query-batch", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--kernel", choices=("vector", "torch"), default="vector")
    parser.add_argument("--verify-controls", action="store_true")
    parser.add_argument("--verify-state", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    bind_candidate()
    (base.aggregate if args.aggregate else base.run)(args)


if __name__ == "__main__":
    main()
