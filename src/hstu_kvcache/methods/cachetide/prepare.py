"""Shared item preparation for the frozen affine-A correction scheme.

All-layer projection batching changes scheduling, not the feature function.
The fixed bank and each release table have different lifetimes.  Native-row
features remain transient, and SiLU stays before the release's Q transform.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from hstu_kvcache.methods.cachetide.adapter import ReleasePrefix
from hstu_kvcache.methods.cachetide.features import FrozenFeatureBank


@torch.no_grad()
def project_fixed_item_terms(source, origin_embeddings, *, chunk_size=16384):
    """Build phi_star with one embedding read/projection per catalogue chunk.

    `effective_weights` already includes the frozen affine preprocessing. Bias
    is kept separate to preserve (native + item) + bias before SiLU.
    """
    if chunk_size <= 0:
        raise ValueError("item chunk must be positive")
    parts = [layer.effective_weights() for layer in source.layers]
    weights = torch.cat([part[1] for part in parts], -1)
    bias = torch.stack([part[2] for part in parts])
    rows, width = origin_embeddings.shape
    layers, code = len(parts), bias.shape[-1]
    table = torch.empty(layers, rows, code, device=weights.device, dtype=weights.dtype)
    chunks = 0
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        embeddings = origin_embeddings[start:stop].to(weights)
        projected = embeddings @ weights
        table[:, start:stop].copy_(projected.reshape(stop-start, layers, code).transpose(0, 1))
        chunks += 1
    bank = FrozenFeatureBank(table, bias,
        tuple(tuple(weight for weight in part[0]) for part in parts),
        dict(kind="implementation_batched_fixed_projection", catalogue_items=rows,
             num_layers=layers, code_width=code, item_width=width))
    work = dict(item_projection_flops=2*rows*width*layers*code,
        item_projection_calls=chunks, reference_item_projection_calls=layers*chunks,
        embedding_read_payload_bytes=rows*width*origin_embeddings.element_size(),
        reference_embedding_read_payload_bytes=layers*rows*width*origin_embeddings.element_size(),
        table_bytes=table.numel()*table.element_size(),
        scope="One immutable bank across releases; no row-state saving or SiLU folding")
    return bank, work


@torch.no_grad()
def fixed_features_from_bank(layer, k, v, item_ids, bank, *, item_terms=None):
    """Exact frozen feature placement; all row-specific native work is retained."""
    if layer.feature_mode != "origin_encoder" or layer.mode != "A":
        raise ValueError("this preparation path implements the selected origin-encoder A branch")
    native = torch.cat([
        layer.kv_features(k, v, branch).to(weight.dtype) @ weight
        for branch, weight in zip(layer.branches, bank.native_weights[layer.layer_index], strict=True)
    ], -1)
    if item_terms is None:
        item_terms = bank.table[layer.layer_index][item_ids]
    return F.silu((native + item_terms) + bank.bias[layer.layer_index])


@torch.no_grad()
def native_feature_chunks(adapter, cache, item_ids, bank, *, chunk_size=256):
    """Yield transient [layer,batch,rows,48] eta chunks, never a second row state."""
    if chunk_size <= 0:
        raise ValueError("native-row chunk must be positive")
    for start in range(0, cache.seq_len, chunk_size):
        stop = min(start + chunk_size, cache.seq_len)
        ids = item_ids[:, start:stop]
        shared_item_terms = bank.table[:, ids]
        eta = torch.stack([fixed_features_from_bank(layer, cache.k[index, :, start:stop],
            cache.v[index, :, start:stop], ids, bank, item_terms=shared_item_terms[index])
            for index, layer in enumerate(adapter.layers)])
        yield start, stop, eta


@torch.no_grad()
def compile_item_tables(adapter, model, previous=None, *, chunk_size=16384):
    """Compile full-catalogue D using one Current-embedding lookup per chunk.

    Matmul concatenates all layers' H projections. Previous-D transforms remain
    block sparse, batched over layers, rather than using a wasteful dense 48x48
    matrix for every item. Arithmetic counts equal the original compiler.
    """
    if adapter.mode != "A" or any(layer.xi_width != 1 for layer in adapter.layers):
        raise ValueError("this compiler implements the selected affine A branch")
    if chunk_size <= 0:
        raise ValueError("item chunk must be positive")
    reference = next(adapter.parameters())
    layers, code = adapter.num_layers, adapter.code_width
    rows, item_width = model.item_emb.weight.shape
    if previous is not None:
        if previous.D.shape != (layers, rows, code, 1):
            raise ValueError("previous and current catalogue geometry differ")
        if previous.layers is None or len(previous.layers) != layers:
            raise ValueError("previous release lacks layer feature metadata")
        for index, (layer, prior) in enumerate(zip(adapter.layers, previous.layers, strict=True)):
            mode = getattr(prior, "feature_mode", None)
            xi_width = getattr(prior, "xi_width", None)
            if mode is None or xi_width is None:
                raise ValueError(f"previous release lacks fixed-feature metadata at layer {index}")
            if mode != layer.feature_mode or xi_width != layer.xi_width:
                raise ValueError("previous release changed fixed feature mode or summary width")
            if mode == "origin_encoder":
                old_sha = getattr(prior, "feature_bank_sha256", None)
                if not old_sha or not layer.feature_bank_sha256:
                    raise ValueError("release lacks immutable feature-bank SHA256 metadata")
                if old_sha != layer.feature_bank_sha256:
                    raise ValueError("previous release changed the immutable feature bank")
            else:
                for j in range(len(layer.branches)):
                    old_basis = getattr(prior, f"native_basis_{j}", None)
                    if old_basis is None:
                        raise ValueError("previous release lacks fixed native basis metadata")
                    if not torch.equal(old_basis, getattr(layer, f"native_basis_{j}")):
                        raise ValueError("previous release changed the fixed native basis")
    item_weight = torch.cat([layer.item_update_weight[..., 0] for layer in adapter.layers], -1)
    item_bias = torch.stack([layer.item_update_bias[..., 0] for layer in adapter.layers])
    rotations = [layer.rotation_blocks() for layer in adapter.layers]
    widths = [branch.code_width for branch in adapter.layers[0].branches]
    d = torch.empty(layers, rows, code, 1, device=reference.device, dtype=reference.dtype)
    p, q = [], []
    arithmetic = dict(item_projection_flops=0, history_composition_flops=0,
        matrix_composition_flops=0, cayley_preparation_flops_estimate=0)
    for index, layer in enumerate(adapter.layers):
        arithmetic["cayley_preparation_flops_estimate"] += sum(
            (8*width**3+2)//3+6*width**2 for width in widths)
        pb, qb = [], []
        for branch, a, b in zip(layer.branches, rotations[index], layer.native_update, strict=True):
            section = slice(branch.code_start, branch.code_start+branch.code_width)
            if previous is None:
                pb.append(a.clone()); qb.append(b.clone())
            else:
                pb.append(a @ previous.P[index, section, section])
                qb.append(a @ previous.Q[index, section, section]+b)
                arithmetic["matrix_composition_flops"] += 4*branch.code_width**3+branch.code_width**2
        p.append(torch.block_diag(*pb)); q.append(torch.block_diag(*qb))
    batched_rotations = [torch.stack([rotation[j] for rotation in rotations])
                         for j in range(len(widths))]
    chunks = 0
    for start in range(0, rows, chunk_size):
        stop = min(start+chunk_size, rows)
        ids = torch.arange(start, stop, device=reference.device)[None]
        features = model.lookup_item_embeddings(ids)[0]
        current = (features @ item_weight).reshape(stop-start, layers, code).transpose(0, 1)
        current = current + item_bias[:, None]
        if previous is None:
            d[:, start:stop, :, 0].copy_(current)
        else:
            for branch, a in zip(adapter.layers[0].branches, batched_rotations, strict=True):
                section = slice(branch.code_start, branch.code_start+branch.code_width)
                d[:, start:stop, section, 0] = torch.einsum("lcd,lmd->lmc", a,
                    previous.D[:, start:stop, section, 0])
            d[:, start:stop, :, 0] += current
            arithmetic["history_composition_flops"] += (stop-start)*layers*(
                2*sum(width**2 for width in widths)+code)
        arithmetic["item_projection_flops"] += (stop-start)*layers*(2*item_width*code+code)
        chunks += 1
    prefix = ReleasePrefix(torch.stack(p), torch.stack(q), d, tuple(adapter.layers),
                           1 if previous is None else previous.release+1)
    work = dict(arithmetic=arithmetic, total_flops=sum(arithmetic.values()),
        item_lookup_calls=chunks, reference_item_lookup_calls=layers*chunks,
        item_projection_calls=chunks, reference_item_projection_calls=layers*chunks,
        history_block_calls=0 if previous is None else len(widths)*chunks,
        reference_history_block_calls=0 if previous is None else layers*len(widths)*chunks,
        embedding_read_payload_bytes=rows*item_width*model.item_emb.weight.element_size(),
        reference_embedding_read_payload_bytes=layers*rows*item_width*model.item_emb.weight.element_size(),
        item_projection_parameter_bytes=item_weight.numel()*item_weight.element_size(),
        table_bytes=d.numel()*d.element_size(),
        scope="Equal floating-point work; fewer lookup/projection operations, not loaded serving throughput")
    return prefix, work
