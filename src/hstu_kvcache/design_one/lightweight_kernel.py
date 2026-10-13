"""Inference-only tiled reader for the lightweight Medium development probe.

One QK/ELU+1 tile feeds native Value and narrow-code reductions. Neither a
concatenated history tensor nor a full attention matrix is materialized. A
matching rank-zero native path makes operator timing comparisons symmetric.
FP32 tile sums and TF32x3 dots are numerically close, not bit-identical, to
the Torch reference. Narrow dots pad rank to 16 hardware lanes; logical FLOPs
alone therefore do not predict runtime. Training stays on lightweight.py.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _read(Q, K, V, Z, Decoder, Out,
          Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
          K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
          V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
          Z0: tl.constexpr, Z1: tl.constexpr, Z2: tl.constexpr,
          H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, P: tl.constexpr,
          SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dim = tl.arange(0, 32)
    q = tl.load(Q + batch * Q0 + head * Q1 + rows[:, None] * Q2 + dim[None, :],
                rows[:, None] < M, 0)
    acc = tl.zeros((BM, 32), tl.float32)
    if P > 0:
        rank = tl.arange(0, 16)
        latent = tl.zeros((BM, 16), tl.float32)
    for start in range(0, N, BN):
        cols = start + tl.arange(0, BN)
        k = tl.load(K + batch * K0 + head * K1 + cols[:, None] * K2 + dim[None, :],
                    cols[:, None] < N, 0)
        v = tl.load(V + batch * V0 + head * V1 + cols[:, None] * V2 + dim[None, :],
                    cols[:, None] < N, 0)
        score = tl.dot(q, tl.trans(k), input_precision="tf32x3")
        score = score * SCALE
        # Retain the reference's ELU then +1 boundary, including tiny-negative
        # cancellation, rather than replacing the negative branch with exp.
        elu = tl.where(score > 0, score, tl.exp(score) - 1.)
        weight = tl.where((rows[:, None] < M) & (cols[None, :] < N), elu + 1., 0.)
        acc = tl.dot(weight, v, acc, input_precision="tf32x3")
        if P > 0:
            z = tl.load(Z + batch * Z0 + head * Z1 + cols[:, None] * Z2 + rank[None, :],
                        (cols[:, None] < N) & (rank[None, :] < P), 0)
            latent = tl.dot(weight, z, latent, input_precision="tf32x3")
    if P > 0:
        decoder = tl.load(Decoder + head * P * 32 + rank[:, None] * 32 + dim[None, :],
                          rank[:, None] < P, 0)
        delta = tl.dot(latent, decoder, input_precision="tf32x3")
        acc = acc + delta
    tl.store(Out + (bh * M + rows[:, None]) * 32 + dim[None, :], acc, rows[:, None] < M)


@triton.jit
def _vector_read(Q, K, V, Z, Decoder, Out,
                 Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
                 K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                 V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                 Z0: tl.constexpr, Z1: tl.constexpr, Z2: tl.constexpr,
                 H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, P: tl.constexpr,
                 SCALE: tl.constexpr, BN: tl.constexpr):
    # One head/query reduction avoids padded tensor-core code products. The
    # history is small and bounded; no temporary reduction buffer is stored.
    query, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    rows, dim = tl.arange(0, BN), tl.arange(0, 32)
    q = tl.load(Q + batch * Q0 + head * Q1 + query * Q2 + dim)
    k = tl.load(K + batch * K0 + head * K1 + rows[:, None] * K2 + dim[None, :],
                rows[:, None] < N, 0)
    v = tl.load(V + batch * V0 + head * V1 + rows[:, None] * V2 + dim[None, :],
                rows[:, None] < N, 0)
    scores = tl.sum(k * q[None, :], axis=1) * SCALE
    elu = tl.where(scores > 0, scores, tl.exp(scores) - 1.)
    weights = tl.where(rows < N, elu + 1., 0.)
    native = tl.sum(weights[:, None] * v, axis=0)
    if P > 0:
        rank = tl.arange(0, P)
        z = tl.load(Z + batch * Z0 + head * Z1 + rows[:, None] * Z2 + rank[None, :],
                    rows[:, None] < N, 0)
        latent = tl.sum(weights[:, None] * z, axis=0)
        decoder = tl.load(Decoder + head * P * 32 + rank[:, None] * 32 + dim[None, :])
        native += tl.sum(latent[:, None] * decoder, axis=0)
    tl.store(Out + (bh * M + query) * 32 + dim, native)


def fused_history_read(q, k, v, codes=None, decoder=None, *, scale=1., backend="tiled"):
    """Read [B,H,Q,32] over native [B,N,H*32], optionally with [B,N,H,p].

    Frozen FP32 CUDA inputs only, Q<=16, N<=1024 and p in 2/4/8. All queries
    are independent and may read the entire historical prefix; the new-token
    self term belongs to the ordinary query reader. No autograd fallback.
    """
    if torch.is_grad_enabled():
        raise ValueError("the lightweight kernel is inference-only; use torch.no_grad()")
    batch, heads, queries, dim = q.shape
    length = k.shape[1]
    if (q.device.type != "cuda" or q.dtype != torch.float32 or k.dtype != q.dtype
            or v.dtype != q.dtype or k.device != q.device or v.device != q.device
            or dim != 32 or not 1 <= queries <= 16 or not 1 <= length <= 1024
            or k.shape != v.shape or k.shape != (batch, length, heads * dim)
            or q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1):
        raise ValueError("kernel requires FP32 CUDA Medium queries and aligned native history")
    rank = 0 if codes is None else codes.shape[-1]
    if codes is not None and (
            rank not in (2, 4, 8) or codes.shape != (batch, length, heads, rank)
            or codes.dtype != q.dtype or codes.device != q.device or codes.stride(-1) != 1
            or decoder is None or decoder.shape != (heads, rank, dim)
            or decoder.dtype != q.dtype or decoder.device != q.device or not decoder.is_contiguous()):
        raise ValueError("codes and decoder must match the native head layout")
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    # K/V stay in native row-major layout. No reshape/copy of the cache occurs.
    code_strides = (codes.stride(0), codes.stride(2), codes.stride(1)) if codes is not None else (0, 0, 0)
    arguments = (q, k, v, codes, decoder, out, *q.stride()[:3],
                 k.stride(0), dim, k.stride(1), v.stride(0), dim, v.stride(1), *code_strides)
    meta = dict(H=heads, M=queries, N=length, P=rank, SCALE=float(scale))
    if backend == "tiled":
        _read[(triton.cdiv(queries, 16), batch * heads)](
            *arguments, **meta, BM=16, BN=64, num_warps=4, num_stages=1, enable_fp_fusion=False)
    elif backend == "vector":
        _vector_read[(queries, batch * heads)](
            *arguments, **meta, BN=triton.next_power_of_2(length),
            num_warps=8, num_stages=1, enable_fp_fusion=False)
    else:
        raise ValueError("unknown bounded reader probe backend")
    return out


def make_history_read(adapter=None, state=None, counts=None, *, backend="tiled"):
    """Construct the current reader callback; adapter=None is matching native."""
    if (adapter is None) != (state is None):
        raise ValueError("adapter and codes must be supplied together")

    def read(layer, attention, q, k, v):
        if (attention.training or attention.position_bias is not None
                or attention.block_variant != "legacy" or attention.cfg.activation != "elu_plus1"):
            raise ValueError("kernel is limited to frozen legacy ELU+1 history reads")
        if adapter is None:
            return fused_history_read(q, k, v, scale=attention.scale, backend=backend)
        if state.seq_len != k.shape[1]:
            raise ValueError("code rows must align with retained native history")
        result = fused_history_read(q, k, v, state.codes[layer], adapter.layers[layer].decoder,
                                    scale=attention.scale, backend=backend)
        if adapter.response is not None:
            lengths = (torch.full((q.shape[0],), state.seq_len, device=q.device)
                       if counts is None else counts)
            result = adapter.response[layer](q, result, lengths)
        return result

    return read
