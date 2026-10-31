#!/usr/bin/env python3
"""Small CPU checks of exact containment and strictly expanded code access."""
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
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[3]

import torch
from torch import nn
from torch.nn import functional as F

from hstu_kvcache.methods.cachetide.structure import StructuredKVAdapter, StructuredRefinement
from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.models import HSTUKVCache

LAYOUTS = ("group2", "shared6_private7", "shared12_private6")
LOCAL = ROOT/"results/design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_lazy/calibration"


def attention():
    return SimpleNamespace(position_bias=None, training=False, block_variant="legacy",
                           cfg=SimpleNamespace(activation="elu_plus1"), scale=32**-.5,
                           attn_dropout=nn.Identity(), _activate=lambda scores: F.elu(scores)+1)


def baseline_coordinates(adapter, codes):
    flat = codes.flatten(-2)
    inverse = [adapter.layers[0].owners.index((h, r)) for h in range(6) for r in range(8)]
    return flat[..., inverse].reshape(*codes.shape[:-2], 6, 8)


def split_encode(adapter, cache, item):
    split = adapter.effective_weights()
    rows = []
    for layer in range(adapter.num_layers):
        k, v = cache.k[layer].reshape(1, cache.seq_len, 6, 32), cache.v[layer].reshape(1, cache.seq_len, 6, 32)
        terms = []
        for branch, weights in zip(split["branches"], split["kv_weights"], strict=True):
            heads = branch["heads"]
            x = torch.cat((k[..., heads, :].flatten(-2), v[..., heads, :].flatten(-2)), dim=-1)
            terms.append(x @ weights[layer])
        z = F.silu(torch.cat(terms, dim=-1)+item @ split["item_weight"][layer]+split["bias"][layer])
        rows.append(z.reshape(1, cache.seq_len, 6, 8))
    return torch.stack(rows)


@torch.no_grad()
def verify_centered_coordinates(original, adapter, cache, item):
    refiner = StructuredRefinement(adapter)
    expected = []
    for index, (old, new) in enumerate(zip(original.layers, adapter.layers, strict=True)):
        delta = torch.empty_like(old.input_weight).uniform_(-.01, .01)
        bias_delta = torch.empty_like(old.input_bias).uniform_(-.01, .01)
        weight_scale = old.input_weight.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        bias_rms = old.input_bias.square().mean(-1, keepdim=True).sqrt()
        bias_scale = torch.where(bias_rms > 0, bias_rms, weight_scale.squeeze(-1))
        shape = (*cache.k[index].shape[:-1], 6, 32)
        features = torch.cat((cache.k[index].reshape(shape), cache.v[index].reshape(shape),
                              item[:, :, None, :].expand(*shape[:-1], 192)), dim=-1)
        normalized = (features-old.input_center)/old.input_scale
        z = F.silu(normalized.transpose(1, 2) @ (old.input_weight+weight_scale*delta)[None]
                   +(old.input_bias+bias_scale*bias_delta)[None, :, None])
        expected.append(z.transpose(1, 2))
        blocks = new.weight_blocks(refiner.weight_delta[index])
        for branch, block in zip(new.branches, blocks, strict=True):
            for column in range(branch.code_width):
                owner, coordinate = new.owners[branch.code_start+column]
                slot = branch.heads.index(owner)
                value_start = len(branch.heads)*32+slot*32
                block[slot*32:(slot+1)*32, column].copy_(delta[owner, :32, coordinate])
                block[value_start:value_start+32, column].copy_(delta[owner, 32:64, coordinate])
                block[2*len(branch.heads)*32:, column].copy_(delta[owner, 64:, coordinate])
                refiner.bias_delta[index, branch.code_start+column].copy_(bias_delta[owner, coordinate])
    actual = baseline_coordinates(adapter, refiner.encode_cache(adapter, cache, item).codes)
    return float((actual-torch.stack(expected)).abs().max())


def verify_source(original, name):
    original = original.double().eval().requires_grad_(False)
    layers = original.num_layers
    n = 9
    cache = HSTUKVCache(torch.randn(layers, 1, n, 192, dtype=torch.float64),
                       torch.randn(layers, 1, n, 192, dtype=torch.float64), n)
    unchanged = (cache.k.clone(), cache.v.clone())
    item = torch.randn(1, n, 192, dtype=torch.float64)
    q = torch.randn(1, 6, 3, 32, dtype=torch.float64)
    old_codes = original.encode_cache(cache, item).codes
    results = []
    for layout in LAYOUTS:
        adapter = StructuredKVAdapter.from_adapter(original, layout).eval().requires_grad_(False)
        state = adapter.encode_cache(cache, item)
        code_error = float((baseline_coordinates(adapter, state.codes)-old_codes).abs().max())
        split_error = float((split_encode(adapter, cache, item)-state.codes).abs().max())
        read_error = max(float((original.layers[i].read(attention(), q, cache.k[i], cache.v[i], old_codes[i])
                               -adapter.layers[i].read(attention(), q, cache.k[i], cache.v[i], state.codes[i])).abs().max())
                         for i in range(layers))
        refiner = StructuredRefinement(adapter)
        refined = refiner.encode_cache(adapter, cache, item)
        identity_error = float((refined.codes-state.codes).abs().max().detach())
        centered_error = verify_centered_coordinates(original, adapter, cache, item)
        with torch.no_grad():
            for parameter in refiner.parameters():
                parameter.uniform_(-.01, .01)
        before = refiner.encode_cache(adapter, cache, item)
        before_read = refiner.make_history_read(adapter, before)(0, attention(), q, cache.k[0], cache.v[0])
        refiner.fold_into(adapter)
        after = adapter.encode_cache(cache, item)
        fold_error = max(float((before.codes-after.codes).abs().max().detach()),
                         float((before_read-adapter.make_history_read(after)(0, attention(), q, cache.k[0], cache.v[0])).abs().max().detach()))
        # Cross-head channels and all existing coordinates must retain gradients.
        refiner = StructuredRefinement(adapter)
        train_codes = refiner.encode_cache(adapter, cache, item)
        train_read = refiner.make_history_read(adapter, train_codes)(0, attention(), q, cache.k[0], cache.v[0])
        train_read.square().mean().backward()
        gradients = {key: float(parameter.grad.norm()) for key, parameter in refiner.named_parameters()}
        for error in (code_error, split_error, read_error, identity_error, centered_error, fold_error):
            if error > 1e-9:
                raise AssertionError(f"{name}/{layout}: exact containment/fold error {error}")
        if not all(value > 0 for value in gradients.values()):
            raise AssertionError(f"{name}/{layout}: disconnected trainable coordinates")
        if state.codes.numel() != old_codes.numel() or state.storage_bytes() != old_codes.numel()*8:
            raise AssertionError("a layout changed persistent code size")
        if not torch.equal(cache.k, unchanged[0]) or not torch.equal(cache.v, unchanged[1]):
            raise AssertionError("code reads/fits changed native K/V")
        results.append(dict(source=name, layout=layout, accessible_rank=adapter.accessible_rank,
                            stored_floats_per_layer_row=48, code_error=code_error,
                            split_encode_error=split_error, read_error=read_error,
                            zero_coordinate_error=identity_error, fold_error=fold_error,
                            original_centered_coordinate_error=centered_error,
                            coordinate_gradient_norms=gradients,
                            added_flops=adapter.estimate_flops(tokens=1024, candidates=16)))
    return results


def verify_cross_head():
    old = LightweightFullItemKVAdapter(1, 6, 32).double().eval().requires_grad_(False)
    item = torch.zeros(1, 1, 192, dtype=torch.float64)
    q = torch.zeros(1, 6, 1, 32, dtype=torch.float64)
    caches = []
    for key in (-1., 1.):
        k = torch.zeros(1, 1, 1, 192, dtype=torch.float64)
        k[0, 0, 0, 32] = key  # only native head1 changes
        caches.append(HSTUKVCache(k, torch.zeros_like(k), 1))
    old_reads = [old.layers[0].read(attention(), q, c.k[0], c.v[0], old.encode_cache(c, item).codes[0])
                 for c in caches]
    results = []
    for layout in LAYOUTS:
        new = StructuredKVAdapter.from_adapter(old, layout).eval().requires_grad_(False)
        layer = new.layers[0]
        with torch.no_grad():
            # Open one permitted head1->head0 correction path. All native head0
            # inputs and the complete Current embedding are identical.
            branch = layer.branches[0]
            weight = layer.weight_blocks()[0]
            weight[branch.heads.index(1)*32, 0] = 1.
            slot = layer.access_indices[0].tolist().index(branch.code_start)
            layer.decoder[0, slot, 0] = 1.
        outputs = [layer.read(attention(), q, c.k[0], c.v[0], new.encode_cache(c, item).codes[0])[0, 0, 0, 0].item()
                   for c in caches]
        target = [F.silu(torch.tensor(-1., dtype=torch.float64)).item(), F.silu(torch.tensor(1., dtype=torch.float64)).item()]
        error = max(abs(x-y) for x, y in zip(outputs, target, strict=True))
        if error > 1e-12 or outputs[0] == outputs[1]:
            raise AssertionError("new layout did not represent the missing cross-head dependence")
        torch.manual_seed(17)
        allowed_decoder = torch.randn(new.accessible_rank, 64, dtype=torch.float64)
        output_rank = int(torch.linalg.matrix_rank(allowed_decoder))
        results.append(dict(layout=layout, old_head0_outputs=[float(x[0, 0, 0, 0]) for x in old_reads],
                            new_head0_outputs=outputs, target=target, target_max_error=error,
                            allowed_combined_key_value_decoder_rank=output_rank,
                            baseline_combined_decoder_rank_upper_bound=8,
                            interpretation="Expression-space witness only; no real-model quality claim."))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    results = []
    for version in ("v3", "v4", "v5"):
        saved = torch.load(LOCAL/version/"read_only/calibration.pt", map_location="cpu", weights_only=False)
        adapter = LightweightFullItemKVAdapter(**saved["config"])
        adapter.load_state_dict(saved["state_dict"], strict=True)
        results.extend(verify_source(adapter, version))
    report = dict(status="passed", device="cpu", dtype="float64", seed=17,
                  exact_containment=results, expression_witness=verify_cross_head())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
