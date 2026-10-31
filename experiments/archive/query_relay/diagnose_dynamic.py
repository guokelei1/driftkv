#!/usr/bin/env python3
"""Offline held-out teacher diagnostics for dynamic versus mean gates.

No fitting, future requests or labels. Teacher closeness and ranking utility
are distinct questions; this probe reports common and relative score error.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import torch

from hstu_kvcache.evaluation.capture import capture_rows
from hstu_kvcache.evaluation.release_context import RELEASES, load_models, setup_device
from hstu_kvcache.evaluation.panels import load_history, load_panel
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.methods.cachetide.adapter import UnifiedRelayAdapter, operation_gate_query
from hstu_kvcache.common.artifacts import sha256, write_json

PANEL = ROOT / "results/design_two_2026_10/benchmark_panel"


@torch.no_grad()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(args.checkpoint.with_suffix(".json").read_text())
    uids = metadata["validation_uids"]
    target = metadata["target"]
    binding, groups, _ = load_panel(PANEL)
    if not set(uids) <= set(groups["validation"]):
        raise ValueError("diagnostics must use reserved validation roles")
    device = setup_device(args.gpu)
    models = load_models(binding, device, int(target[1:]))
    model = models[target]
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    adapter = UnifiedRelayAdapter(**payload["config"]).to(device).eval().requires_grad_(False)
    adapter.load_state_dict(payload["state_dict"])
    means = torch.as_tensor(payload["gate_constants"], device=device)
    constants = tuple((row[0], row[1]) for row in means)
    histories = load_history(PANEL, requested_uids=uids)
    if args.source_pack:
        pack = torch.load(args.source_pack, map_location="cpu", weights_only=False)
        captured = pack["rows"]
    else:
        if target != "v3":
            raise ValueError("later-release teacher diagnostics need the candidate's source pack")
        captured, _, _ = capture_rows(models, histories, uids, target, binding["known_vocab_size"])
    units = []
    cut = dict(RELEASES)[target]
    for uid in uids:
        raw = captured[uid]
        cache, teacher = raw["parent"].to(device), raw["teacher"].to(device)
        items = histories.rows[uid][1][:np.searchsorted(histories.rows[uid][0], cut, side="left")][-cache.seq_len:]
        items = torch.as_tensor(items, device=device, dtype=torch.long)[None]
        previous = raw.get("previous_codes")
        if previous is None:
            previous = cache.k.new_zeros(adapter.num_layers, 1, cache.seq_len, adapter.heads, adapter.rank)
        state = adapter.advance_codes(cache, previous.to(device), raw["item_features"].to(device), item_ids=items)
        candidates = raw["candidates"].to(device)[None]
        delta = torch.tensor([raw["query_delta"]], device=device)
        wanted = score(model, teacher, candidates, delta)[0]
        local = []
        gate_query = operation_gate_query(adapter, model, candidates)
        base = adapter.make_history_read(state, gate_query=gate_query)

        def read(index, attention, q, k, v):
            actual = base(index, attention, q, k, v)
            constant = adapter.layers[index].read(attention, q, k, v, state.codes[index],
                                                  gate_constants=constants[index])
            target_read = history_read(attention, q, teacher.k[index], teacher.v[index])
            n = max(1, cache.seq_len)
            local.append(dict(layer=index,
                dynamic_read_rate_mse=float(((actual - target_read) / n).square().mean()),
                constant_read_rate_mse=float(((constant - target_read) / n).square().mean())))
            return actual

        actual = score(model, cache, candidates, delta, history_read_fn=read)[0]
        constant = score(model, cache, candidates, delta,
            history_read_fn=adapter.make_history_read(state, gate_constants=constants,
                                                      gate_query=gate_query))[0]
        errors = {}
        for label, value in (("dynamic", actual), ("mean_gate", constant)):
            residual = value - wanted
            errors[label] = dict(logit_mse=float(residual.square().mean()),
                common_offset_mse=float(residual.mean(-1).square().mean()),
                relative_logit_mse=float((residual - residual.mean(-1, keepdim=True)).square().mean()))
        units.append(dict(uid=uid, errors=errors, local=local))
    overall = {label: {metric: float(np.mean([row["errors"][label][metric] for row in units]))
                      for metric in ("logit_mse", "common_offset_mse", "relative_logit_mse")}
               for label in ("dynamic", "mean_gate")}
    write_json(args.output / "summary.json", dict(status="complete", target=target,
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint), validation_uids=uids,
        overall=overall, units=units, no_labels_or_future_queries=True,
        scope="Held-out first-read teacher diagnostic; no quality or complete-cost acceptance claim",
        source_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(status="complete", target=target, overall=overall)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-pack", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    run(p.parse_args())
