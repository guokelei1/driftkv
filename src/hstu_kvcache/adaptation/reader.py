"""Differentiable paired reads using the frozen checkpoint's actual operator.

One user, many independent transient queries. No candidate-candidate attention.
The retained Medium models are legacy ELU+1; never relabel them SiLU-native.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hstu_kvcache.models import HSTUKVCache

from .summary import Summary


def history_read(attention, q, k, v, *, count=None):
    """Independent users: q=[batch,heads,queries,dim], k/v=[batch?,positions,width]."""
    if attention.position_bias is not None:
        raise ValueError("the frozen Medium experiment requires no position bias")
    heads, dim = attention.num_heads, attention.head_dim
    batch = k.shape[0] if k.ndim == 3 else 1
    k = k.to(q.dtype).reshape(batch, -1, heads, dim).transpose(1, 2)
    v = v.to(q.dtype).reshape(batch, -1, heads, dim).transpose(1, 2)
    weights = attention._activate(q @ k.transpose(-2, -1) * attention.scale)
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    if count is not None:
        weights = weights * count.reshape(batch, 1, 1, -1).to(weights.dtype)
    return attention.attn_dropout(weights) @ v


def summary_read(attention, q, summary: Summary, layer):
    k = summary.payload[:, :, layer, 0].flatten(0, 1)
    v = summary.payload[:, :, layer, 1].flatten(0, 1)
    return history_read(attention, q, k, v, count=summary.count.flatten())


@dataclass
class ReadResult:
    hidden: torch.Tensor
    new_kv: HSTUKVCache
    queries: tuple[torch.Tensor, ...]
    corrections: tuple[torch.Tensor, ...]
    history_heads: tuple[torch.Tensor, ...]
    layer_outputs: tuple[torch.Tensor, ...] = ()


def read_embedded(model, cache, x, source=None, translated=None, *, trace=False, response_delta=None,
                  response_time_delta=None, time_features=None, response_query_delta=None, history_override=None):
    """The six-layer query path stays differentiable with a frozen backbone."""
    kvs, queries, corrections, history_heads = [], [], [], []
    layer_outputs = []
    constant_delta = None
    if translated is not None and translated.read_mode == "constant_elu":
        value_delta = ((translated.payload[..., 1, :] - source.payload[..., 1, :])
                       * source.count[..., None, None]).sum((0, 1))
        constant_delta = value_delta[None, None]
    elif response_delta is not None:
        constant_delta = response_delta[:, None]
    coefficients = (translated.temporal_coefficients[None]*source.count.sum()
                    if translated is not None and translated.temporal_coefficients is not None
                    else response_time_delta)
    if coefficients is not None:
        phi = time_features[:, None, :] if time_features.ndim == 2 else time_features
        time_delta = torch.einsum("bqt,bltw->bqlw", phi, coefficients)
        constant_delta = time_delta if constant_delta is None else constant_delta + time_delta
    query_coefficients = (translated.query_coefficients[None]*source.count.sum()
                          if translated is not None and translated.query_coefficients is not None
                          else response_query_delta)
    if constant_delta is not None and (model.cfg.activation != "elu_plus1" or model.cfg.block_variant != "legacy"):
        raise ValueError("constant functional correction requires the frozen ELU+1 operator")
    for layer, block in enumerate(model.blocks):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        heads = history_read(block.attn, q, cache.k[layer], cache.v[layer])
        if trace:
            history_heads.append(heads)
        delta = torch.zeros_like(heads)
        if constant_delta is not None:
            value = constant_delta[:, :, layer].expand(-1, x.shape[1], -1)
            delta = value.reshape(x.shape[0], x.shape[1], block.attn.num_heads, block.attn.head_dim).transpose(1, 2)
            heads = heads + delta
        elif source is not None and translated is not None:
            delta = summary_read(block.attn, q, translated, layer) - summary_read(block.attn, q, source, layer)
            heads = heads + delta
        if query_coefficients is not None:
            query_correction = q @ query_coefficients[:, layer]
            delta = delta+query_correction
            heads = heads+query_correction
        if history_override is not None:
            # Read-response prototypes and diagnostic replacement use THIS
            # branch's query. Self and the remaining native block stay below.
            heads = history_override(layer, q, heads)
        if block.attn.causal_diagonal == "inclusive":
            weight = block.attn._activate((q * k_new).sum(-1, keepdim=True) * block.attn.scale)
            if block.attn.block_variant == "hstu_reference":
                weight = weight / block.attn.cfg.max_seq_len
            heads = heads + weight * v_new
        attention_out = block.attn._finish(heads)
        if block.block_variant == "hstu_reference":
            update = block.attn.out_proj(block.attn_output_norm(attention_out) * F.silu(block.gate_proj(x_norm)))
        else:
            # All six frozen Medium models use precisely this gate.
            if block.gating != "silu_gate":
                raise ValueError("unexpected frozen Medium gating")
            update = attention_out * F.silu(block.gate_proj(x_norm))
        x = residual + update
        kvs.append((k_new.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1),
                    v_new.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)))
        if trace:
            queries.append(q)
            corrections.append(delta)
            layer_outputs.append(x)
    return ReadResult(model.final_norm(x), HSTUKVCache.from_layer_list(kvs, x.shape[1]),
                      tuple(queries), tuple(corrections), tuple(history_heads), tuple(layer_outputs))


def score(model, cache, candidates, query_delta, source=None, translated=None, *, trace=False, response_delta=None,
          response_time_delta=None, response_query_delta=None, history_override=None):
    x = model.embed_query_tokens(candidates, query_delta)
    temporal = response_time_delta is not None or (translated is not None and translated.temporal_coefficients is not None)
    phi = model.temporal_enc.features(query_delta) if temporal else None
    result = read_embedded(model, cache, x, source, translated, trace=trace,response_delta=response_delta,
                           response_time_delta=response_time_delta,time_features=phi,response_query_delta=response_query_delta,
                           history_override=history_override)
    return model.cc_score_head(result.hidden).squeeze(-1), result


def full_response_rates(model, cache, teacher, candidates, query_delta, counts,
                        response_delta, response_time_delta, time_groups):
    """Calibration-only full-history targets at the actual corrected queries."""
    _, trace = score(model, cache, candidates, query_delta, trace=True,
                     response_delta=response_delta, response_time_delta=response_time_delta)
    batch = candidates.shape[0]
    per_time = candidates.shape[1] // time_groups
    rates = []
    for layer, (block, q, actual) in enumerate(zip(model.blocks, trace.queries, trace.history_heads, strict=True)):
        wanted = history_read(block.attn, q, teacher.k[layer], teacher.v[layer])
        values = (wanted - actual).reshape(batch, block.attn.num_heads, time_groups, per_time, block.attn.head_dim)
        rates.append(values.mean(3).transpose(1, 2).flatten(2) / counts[:, None, None])
    return torch.stack(rates, dim=2)
