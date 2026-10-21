"""Inference-only fused reader for the frozen shared12/private6 relay.

Torch prepares the operation-sized query projections/gates. Triton streams the
native K/V and directly addresses shared/private code coordinates, applying
score correction before native ELU+1 and accumulating V/Z in the same tile.
No expanded per-head history codes, scores, weights or corrected K/V are saved.
The explicit ``torch`` backend is the differentiable research reference; no
implicit fallback occurs if a Triton request is unsupported or fails.
"""
from __future__ import annotations

import math
import torch


def _validate(layer, attention, q, k, v, codes, code_count):
    if torch.is_grad_enabled():
        raise ValueError("optimized reader is inference-only; use torch.no_grad()")
    if (attention.position_bias is not None or attention.block_variant != "legacy"
            or attention.cfg.activation != "elu_plus1"):
        raise ValueError("fused relay reader requires legacy ELU+1 without position bias")
    if attention.training and getattr(attention.attn_dropout, "p", 0):
        raise ValueError("fused relay reader does not implement attention dropout")
    b, h, m, d = q.shape
    n = k.shape[1]
    if (h != layer.heads or h not in (6, 10)
            or (d, layer.read_shared, layer.read_private) != (32, 3, 3)
            or layer.layout != "shared12_private6"):
        raise ValueError("fused reader implements H6/H10 D32, stored12+6/read3+3 geometry")
    if k.shape != (b, n, h*d) or v.shape != k.shape:
        raise ValueError("native K/V must be [batch,history,heads*32]")
    if codes.ndim == 4 and codes.shape[-2:] == layer.code_shape:
        flat = codes.flatten(-2)
    elif codes.ndim == 3 and codes.shape[-1] == layer.code_width:
        flat = codes
    else:
        raise ValueError("codes must match the layer's explicit packed or flat row layout")
    count = flat.shape[1] if code_count is None else int(code_count)
    if flat.shape[0] != b or not 0 <= count <= min(n, flat.shape[1]):
        raise ValueError("code prefix exceeds available native/code rows")
    if q.stride(-1) != 1 or any(x.stride(-1) != 1 for x in (k, v, flat)):
        raise ValueError("native dimensions and packed code coordinates must have unit final stride")
    if any(x.dtype != torch.float32 or x.device != q.device for x in (q, k, v, flat)):
        raise ValueError("fused reader currently requires matching FP32 tensors")
    return flat, count


def _mask_geometry(mask, b, h, m, n, device):
    if mask is None:
        return None, (0, 0, 0, 0)
    if mask.device != device:
        raise ValueError("mask must be on the query device")
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4 or any(x not in (1, y) for x, y in zip(mask.shape, (b, h, m, n))):
        raise ValueError("mask must broadcast to [batch,heads,queries,history]")
    return mask, tuple(0 if size == 1 else stride for size, stride in zip(mask.shape, mask.stride()))


def reader_work_model(*, batch, queries, history, code_count, has_mask=False,
                      block_queries=16, block_history=64, heads=6):
    """Logical arithmetic, global tensor and tile traffic models (not timings).

    Traffic is requested bytes without assumptions about L1/L2 reuse. Tile read
    bytes include repeated per-head shared loads and repeated query tiles.
    FLOPs use logical width6, not padded Tensor Core instruction work.
    """
    b, q, n, c, h, d, r = batch, queries, history, code_count, heads, 32, 6
    vector = q <= 4
    block_queries = 1 if vector else block_queries
    pairs, cpairs = b*h*q*n, b*h*q*c
    pre = b*h*q*(2*d*r*3 + 2*r) if c else 0
    native = 4*pairs*d
    correction = 4*cpairs*r + b*h*q*2*r*d if c else 0
    tiles = math.ceil(q/block_queries) if q else 0
    native_read = b*h*tiles*n*2*d*4
    code_read = b*h*tiles*c*r*4
    return dict(
        logical_matmul_flops=dict(native_qk_av=native, correction=correction,
                                 query_projection_and_two_gate_projections=pre,
                                 total=native+correction+pre),
        additional_pointwise=dict(score_scale_elements=pairs,
                                  native_elu_plus1_elements=pairs,
                                  rms_input_elements=b*h*q*d if c else 0,
                                  tanh_gate_elements=2*b*h*q*r if c else 0),
        tensor_bytes=dict(output=b*h*q*d*4,
                          operation_projected_query_and_value_gate=2*b*h*q*r*4 if c else 0,
                          eager_score_or_weight_each=b*h*q*n*4,
                          eager_head_gather=b*h*c*r*4,
                          fused_history_sized_temporary=0),
        requested_history_read_bytes=dict(native_kv=native_read, code=code_read,
                                           mask=b*h*q*n if has_mask else 0),
        tiling=dict(kernel="vector_fp32" if vector else "matrix_tf32x3",
                    query_tile=block_queries, history_tile=block_history,
                    rank_padded_to=8 if vector else 16,
                    multiplication="FP32 products/reductions" if vector else "TF32x3 products, FP32 accumulation",
                    caveat="Padded/precision-lowered instruction work is not logical FLOPs; requested traffic is not measured DRAM traffic"))


def fused_relay_read(layer, attention, q, k, v, codes, *, valid_mask=None,
                     code_count=None, gate_constants=None, gate_q=None,
                     prepared_gates=None, prepared_decoders=None,
                     backend="triton", return_stats=False):
    """RelayLayer.read-compatible bounded inference entry, with explicit backend.

    ``q`` is [B,H,Q,32], K/V are native [B,N,H*32]. Codes may be shorter than N
    because entering/current-version suffixes are native-only. Arbitrary masks
    broadcastable from [Q,N], [B,Q,N], [B,H,Q,N] retain their numeric meaning.
    """
    if backend not in ("triton", "torch"):
        raise ValueError("backend must explicitly be 'triton' or 'torch'")
    flat, count = _validate(layer, attention, q, k, v, codes, code_count)
    b, h, m, d = q.shape
    n = k.shape[1]
    mask, mask_strides = _mask_geometry(valid_mask, b, h, m, n, q.device)
    stats = dict(backend=backend, fallback=False, batch=b, queries=m, history=n,
                 kernel="vector_fp32" if m<=4 else "matrix_tf32x3",
                 coded_rows=count, model=reader_work_model(batch=b, queries=m,
                   history=n, code_count=count, has_mask=mask is not None, heads=h))
    if backend == "torch":
        if prepared_gates is None:
            out = layer.read(attention, q, k, v, codes, valid_mask=valid_mask,
                             code_count=count, gate_constants=gate_constants, gate_q=gate_q)
        else:
            keys=k.reshape(b,n,h,d).transpose(1,2)
            values=v.reshape(b,n,h,d).transpose(1,2)
            scores=q@keys.transpose(-2,-1)
            if count:
                z=layer.accessible_codes(codes[:,:count])
                gk,gv=prepared_gates
                projected=(q@layer.key_decoder[:,layer.read_slots].transpose(-2,-1)[None])*gk
                scores=torch.cat((scores[...,:count]+projected@z.transpose(-2,-1),scores[...,count:]),-1)
            weights=attention._activate(scores*attention.scale)
            if mask is not None:
                weights=weights*mask.to(q)
            out=weights@values
            if count:
                out=out+((weights[...,:count]@z)*gv)@layer.decoder[:,layer.read_slots][None]
        return (out, stats) if return_stats else out
    if q.device.type != "cuda":
        raise ValueError("Triton backend requires CUDA; request backend='torch' explicitly")
    if n == 0 or m == 0:
        out = torch.zeros_like(q, memory_format=torch.contiguous_format)
        stats["launches_history_kernel"] = 0
        return (out, stats) if return_stats else out
    from implementation_2026_10.reader_kernel import launch_read
    r = layer.accessible_rank
    if count:
        if prepared_gates is not None:
            if gate_constants is not None:
                raise ValueError("prepared_gates and legacy constant gates are mutually exclusive")
            gk,gv=prepared_gates
            if any(g.shape != (b,h,m,r) or g.device != q.device or g.dtype != q.dtype for g in (gk,gv)):
                raise ValueError("prepared_gates must both be FP32 [B,H,Q,6] on query device")
        elif gate_constants is None:
            if gate_q is None and layer.gate_source == "item":
                raise ValueError("item-source reader requires Current operation embedding gate_q")
            gates = q if gate_q is None else gate_q.to(q)
            if gates.shape != q.shape:
                raise ValueError("gate_q must match [B,H,Q,D]")
            gk, gv = layer.query_gates(gates)
        else:
            gk, gv = gate_constants
            if gk.shape[-1] == 18:
                gk, gv = gk[..., layer.read_slots], gv[..., layer.read_slots]
            gk, gv = (g.to(q).reshape(1,h,1,r) for g in (gk,gv))
        if prepared_decoders is None:
            dk = layer.key_decoder[:, layer.read_slots].to(q).contiguous()
            dv = layer.decoder[:, layer.read_slots].to(q).contiguous()
        else:
            dk,dv=prepared_decoders
            if any(x.shape != (h,r,d) or x.dtype != q.dtype or x.device != q.device or not x.is_contiguous() for x in (dk,dv)):
                raise ValueError("prepared_decoders must both be contiguous FP32 [H,6,D]")
        projected = ((q @ dk.transpose(-2,-1)[None]) * gk).contiguous()
        gv = gv.expand(b,h,m,r).contiguous()
    else:
        # Not dereferenced by the no-code kernel path.
        projected = gv = dv = q
    out = launch_read(q,k,v,flat,projected,gv,dv,mask,mask_strides,
                      code_count=count,scale=float(attention.scale))
    stats["launches_history_kernel"] = 1
    return (out, stats) if return_stats else out


def make_history_read(adapter, state, *, code_count=None, gate_query=None,
                      valid_mask=None, backend="triton", counters=None):
    """Build the same per-layer callback as UnifiedRelayAdapter.make_history_read."""
    def read(index, attention, q, k, v):
        gate = gate_query
        if adapter.gate_source == "first":
            raise ValueError("optimized reader is validated for frozen item-source gates")
        out, stats = fused_relay_read(adapter.layers[index], attention, q,k,v,
            state.codes[index],valid_mask=valid_mask,code_count=code_count,
            gate_q=gate,backend=backend,return_stats=True)
        if counters is not None:
            counters["triton_history_calls"] = counters.get("triton_history_calls",0)+(backend=="triton")
            counters["torch_history_calls"] = counters.get("torch_history_calls",0)+(backend=="torch")
        return out
    return read
