#!/usr/bin/env python3
"""Diagnostic: can a constant correction spuriously activate zero-response heads?

This process-local reader comparison reuses frozen weights and the independent
calibration pipeline. It is not an installed population method or cost claim.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from design import probe_source_confidence as probe
from design.data import ROOT

from hstu_kvcache.adaptation import reader
from hstu_kvcache.models import HSTUKVCache

ORIGINAL_READ = reader.read_embedded
GATE_ACTIVE = True
CHECKS = []
COUNTS = {}


def inactive_read(model, cache, x, source=None, translated=None, *, trace=False, response_delta=None,
                  response_time_delta=None, time_features=None):
    # Uncorrected Exact/Reuse paths stay on the original reader.
    if response_delta is None:
        return ORIGINAL_READ(model, cache, x, source, translated, trace=trace,
            response_delta=response_delta, response_time_delta=response_time_delta, time_features=time_features)
    assert source is None and translated is None and model.cfg.block_variant == "legacy"
    changes = response_delta[:, None]
    if response_time_delta is not None:
        phi = time_features[:, None] if time_features.ndim == 2 else time_features
        changes = changes+torch.einsum("bqt,bltw->bqlw", phi, response_time_delta)
    kvs, queries, corrections, history_heads = [], [], [], []
    for layer, block in enumerate(model.blocks):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        heads = reader.history_read(block.attn, q, cache.k[layer], cache.v[layer])
        original_heads = heads
        delta = changes[:, :, layer].expand(-1, x.shape[1], -1)
        delta = delta.reshape(x.shape[0], x.shape[1], block.attn.num_heads, block.attn.head_dim).transpose(1, 2)
        if GATE_ACTIVE:
            active = heads.ne(0).any(-1, keepdim=True)
            COUNTS[layer][0] += int((~active).sum())
            COUNTS[layer][1] += active.numel()
            delta = delta*active
        heads = heads+delta
        if block.attn.causal_diagonal == "inclusive":
            weight = block.attn._activate((q*k_new).sum(-1, keepdim=True)*block.attn.scale)
            heads = heads+weight*v_new
        x = residual+block.attn._finish(heads)*F.silu(block.gate_proj(x_norm))
        kvs.append((k_new.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1),
                    v_new.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)))
        if trace:
            queries.append(q)
            corrections.append(delta)
            history_heads.append(original_heads)
    return reader.ReadResult(model.final_norm(x), HSTUKVCache.from_layer_list(kvs, x.shape[1]),
                             tuple(queries), tuple(corrections), tuple(history_heads))


@torch.no_grad()
def compare(current, scenes, stable, candidate, batch_size):
    global COUNTS
    COUNTS = {layer:[0, 0] for layer in range(len(current.blocks))}
    original_score = probe.score
    checked = False

    def score_variant(*args, **kwargs):
        nonlocal checked
        global GATE_ACTIVE
        kwargs.pop("source_features")
        expected = original_score(*args, **kwargs)[0] if not checked else None
        reader.read_embedded = inactive_read
        try:
            if not checked:
                GATE_ACTIVE = False
                actual = original_score(*args, **kwargs)[0]
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                CHECKS.append(dict(target=stable.target, maximum_logit_difference=float((actual-expected).abs().max())))
                checked = True
            GATE_ACTIVE = True
            return original_score(*args, **kwargs)
        finally:
            reader.read_embedded = ORIGINAL_READ

    result = ORIGINAL_COMPARE(current, scenes, stable, candidate, batch_size, candidate_score=score_variant)
    frame, record = result
    record["diagnostic"] = "candidate correction zeroed only when that head's actual history response is exactly zero"
    record["zero_response_head_queries"] = {str(layer):dict(zero=values[0], total=values[1]) for layer, values in COUNTS.items()}
    return frame, record


ORIGINAL_COMPARE = probe.compare


def main(cli):
    probe.compare = compare
    args = SimpleNamespace(run_id=cli.run_id, users=cli.users, targets=cli.targets,
        candidate_reference="v9_stratified_stable64_01", inactive_head_diagnostic=True,
        override_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = ROOT / "results/design" / cli.run_id
    try:
        probe.main(args)
    finally:
        if out.exists():
            (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    summary = json.loads((out / "summary.json").read_text())
    summary["reader_canary"] = CHECKS
    probe.write_json(out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--users", type=int, default=128)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    main(parser.parse_args())
