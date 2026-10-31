"""Shared experiment I/O for the isolated three-release Design 2 probe."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import torch

from hstu_kvcache.evaluation.yambda import load_model
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256, write_json

DAY = 86400
RELEASES = (("v3", 259 * DAY), ("v4", 273 * DAY), ("v5", 287 * DAY))
END = 301 * DAY


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sources(*entrypoints):
    result = {}
    paths = [ROOT / "src/hstu_kvcache/evaluation/yambda.py",
             ROOT / "src/hstu_kvcache/adaptation/reader.py",
             ROOT / "src/hstu_kvcache/models/hstu.py",
             ROOT / "src/hstu_kvcache/models/attention.py",
             ROOT / "src/hstu_kvcache/models/state_transition.py",
             ROOT / "src/hstu_kvcache/evaluation/history.py",
             Path(__file__), ROOT / "src/hstu_kvcache/evaluation/panels.py",
             ROOT / "src/hstu_kvcache/evaluation/trajectory.py",
             *(ROOT / path for path in entrypoints)]
    result.update({str(path.relative_to(ROOT)): sha256(path) for path in paths})
    return result


def setup_device(gpu, threads=4):
    torch.set_num_threads(threads)
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.70, device)
    torch.cuda.reset_peak_memory_stats(device)
    return device


def load_models(panel, device, last_version=5, backend="triton"):
    models = {}
    for version in range(2, last_version + 1):
        name = f"v{version}"
        record = panel["models"][name]
        checkpoint = ROOT / record["checkpoint"]
        if sha256(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint differs from the fixed endpoint: {name}")
        model, payload = load_model(checkpoint, device)
        model.eval().requires_grad_(False)
        for block in model.blocks:
            block.attn.backend = backend
        models[name] = model
        if name != "v2" and model.cfg != models["v2"].cfg:
            raise RuntimeError("continuous releases must share the frozen architecture")
        del payload
    return models


def ledger_flops(ledger, model, backend):
    costs = {name: CostModel(model.cfg.hidden_size, len(model.blocks), model.cfg.num_heads, name)
             for name in ("torch", backend)}
    result = Counter()
    for (action, version, before, width), count in ledger.items():
        length = before if action == "append" else width
        cost = costs[backend if length >= model.cfg.max_seq_len else "torch"]
        value = (cost.band_append(before, width, window_size=model.cfg.max_seq_len)
                 if action == "append" else cost.full_cache(width))
        result[f"v{version}/{action}"] += int(count * value)
    return dict(result)
