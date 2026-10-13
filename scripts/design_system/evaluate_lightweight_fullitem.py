#!/usr/bin/env python3
"""Full-item Value encoder, sharing the unchanged continuous replay and kernel.

Only the code producer changes. Native K/V writes, narrow code layout and the
single native-attention inference path are the existing Value implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"scripts/design_system"),str(ROOT/"src")]

from design_system import evaluate_lightweight as base
from hstu_kvcache.design_one.lightweight_fullitem import LightweightFullItemAdapter
from unified_reuse_2026_09.common import sha256

_base_sources=base.execution_sources
_base_candidate_release=base.candidate_release


def candidate_release(counts,*,layers,heads,head_dim,rank,response_width):
    row=_base_candidate_release(counts,layers=layers,heads=heads,head_dim=head_dim,
        rank=rank,response_width=response_width)
    inputs=(2+heads)*head_dim
    per_token=layers*heads*(2*inputs+2*inputs*rank+2*rank)
    row["components_flops"]["resident_release_initialization"]=counts["initial_history_length"]*per_token
    row["components_flops"]["observed_append_encoding"]=counts["appends"]*per_token
    row["variable_flops"]=sum(row["components_flops"].values())
    row["variable_percent_of_once_rebuild"]=100*row["variable_flops"]/row["first_request_rebuild_flops"]
    row["shared_work_budget_at_20_percent_flops"]=(
        .2*row["first_request_rebuild_flops"]-row["variable_flops"])
    row["encoder"]=dict(inputs_per_head=inputs,flops_per_token=per_token,
        scope="Each head's old K/V plus the complete Current item embedding. Query and code-storage costs stay unchanged; per-row encoding is charged at the expanded input width.")
    return row


def execution_sources():
    names=("scripts/design_system/evaluate_lightweight_fullitem.py",
           "src/hstu_kvcache/design_one/lightweight_fullitem.py")
    return {**_base_sources(),**{name:sha256(ROOT/name) for name in names}}


def bind_candidate():
    base.LightweightValueAdapter=LightweightFullItemAdapter
    base.candidate_release=candidate_release
    base.execution_sources=execution_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",type=Path,default=base.PANEL)
    parser.add_argument("--reference",type=Path,default=base.REFERENCE)
    parser.add_argument("--calibration-dir",type=Path,required=True)
    parser.add_argument("--population-cost",type=Path,default=base.PHASE/"population_cost/summary_complete.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-users",type=int,default=0)
    parser.add_argument("--query-batch",type=int,default=16)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--num-shards",type=int,default=1)
    parser.add_argument("--shard-index",type=int,default=0)
    parser.add_argument("--kernel",choices=("vector","torch"),default="vector")
    parser.add_argument("--verify-controls",action="store_true")
    parser.add_argument("--verify-state",action="store_true")
    parser.add_argument("--aggregate",action="store_true")
    args=parser.parse_args()
    bind_candidate()
    (base.aggregate if args.aggregate else base.run)(args)


if __name__=="__main__":main()
