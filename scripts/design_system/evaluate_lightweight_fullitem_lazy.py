#!/usr/bin/env python3
"""Isolated lazy-code policy over the unchanged full-item read evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design_system import evaluate_lightweight as base
from design_system import evaluate_lightweight_fullitem as fullitem
from design_system.lightweight_lazy import LazyLightweightSession
from design_system.profile_lightweight_lazy import lazy_candidate_release, OUTPUT
from unified_reuse_2026_09.common import sha256

_base_cost = base.complete_cost


def execution_sources():
    names = ("scripts/design_system/evaluate_lightweight_fullitem_lazy.py",
        "scripts/design_system/lightweight_lazy.py", "scripts/design_system/profile_lightweight_lazy.py")
    return {**fullitem.execution_sources(), **{name:sha256(ROOT/name) for name in names}}


def complete_cost(common):
    result = _base_cost(common)
    result["role"] = "Exact analytic real30k lazy-code event-shape ledger, verified against the executed eight-user eager/lazy probe; not all-population GPU quality or latency."
    result["state_accounting"] = (
        "Bind release parameters without materializing codes; native writes and support always advance. "
        "Encode the current retained rows at the first actual request, then arriving rows until the next "
        "release or causal retirement. No request prediction, label use, or denominator change.")
    result["initialization_latency_scope"] = (
        "First-request code preparation is additional critical-path work; the separate eight-user probe "
        "measured it. These FLOPs do not establish loaded-service latency.")
    return result


def bind_candidate():
    fullitem.bind_candidate()
    base.LightweightSession = LazyLightweightSession
    base.candidate_release = lazy_candidate_release
    base.complete_cost = complete_cost
    base.execution_sources = execution_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",type=Path,default=base.PANEL)
    parser.add_argument("--reference",type=Path,default=base.REFERENCE)
    parser.add_argument("--calibration-dir",type=Path,required=True)
    parser.add_argument("--population-cost",type=Path,default=OUTPUT / "summary.json")
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
    args = parser.parse_args()
    bind_candidate()
    (base.aggregate if args.aggregate else base.run)(args)


if __name__ == "__main__":
    main()
