#!/usr/bin/env python3
"""Factored KV evaluation with one full-context fast path.

N=1024 uses the retained specialized kernel; other lengths use the validated
runtime-length kernel. This process-local binding preserves prior sources.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

from experiments.archive.compact_read import evaluate_lightweight_fullitem_kv as kv
from hstu_kvcache.methods.cachetide.compact.lightweight_kv_dispatch import make_history_read
from hstu_kvcache.common.artifacts import sha256

_bind_candidate = kv.bind_candidate


def execution_sources():
    names = ("experiments/archive/compact_read/evaluate_lightweight_fullitem_kv_dispatch.py",
        "src/hstu_kvcache/methods/cachetide/compact/lightweight_kv_dispatch.py",
        "src/hstu_kvcache/methods/cachetide/compact/lightweight_kv_runtime_kernel.py")
    return {**kv.execution_sources(), **{name:sha256(ROOT / name) for name in names}}


def bind_candidate():
    kv.make_history_read = make_history_read
    _bind_candidate()
    kv.base.execution_sources = execution_sources


if __name__ == "__main__":
    kv.bind_candidate = bind_candidate
    kv.main()
