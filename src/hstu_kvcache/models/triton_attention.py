"""Tiled pointwise HSTU attention, preserving the eager parameter/cache layout.

Unlike softmax attention, HSTU applies an elementwise activation and a fixed
divisor. No [batch, heads, length, length] tensor is saved or materialized.
The tile decomposition follows the Triton fused-attention tutorial, while the
activation, rounding stages and relative-bias gradient are specific to EvoKV:
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

This module is imported lazily by the optional CUDA backend. Dropout and custom
masks remain in the PyTorch reference implementation.

FP32 dots use three TF32 products for near-FP32 accuracy on Ampere. BF16 eager
rounding boundaries are retained. Bias gradients accumulate tile diagonals in
FP32 with atomics; this changes reduction order and avoids the eager BF16 batch
reduction rounding, so neither backward determinism nor bitwise parity is claimed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scores(q, k, Bias, rows, cols, head, H: tl.constexpr,
            BIAS_OFFSET: tl.constexpr, HAS_BIAS: tl.constexpr,
            SCALE: tl.constexpr, QSTART: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    dtype: tl.constexpr = q.dtype
    # Eager matmul, multiplication and addition each round to the input dtype.
    x = tl.dot(q, tl.trans(k), input_precision="tf32x3").to(dtype)
    x = (x.to(tl.float32) * SCALE).to(dtype)
    if HAS_BIAS:
        index = cols[None, :] - (rows[:, None] + QSTART) + BIAS_OFFSET
        bias = tl.load(Bias + index * H + head,
                       (rows[:, None] < M) & (cols[None, :] < N)
                       & (index >= 0) & (index <= 2 * BIAS_OFFSET), 0).to(dtype)
        x = (x.to(tl.float32) + bias.to(tl.float32)).to(dtype)
    return x


@triton.jit
def _weights(x, rows, cols, M: tl.constexpr, N: tl.constexpr,
             QSTART: tl.constexpr, EXCLUSIVE: tl.constexpr, WINDOW: tl.constexpr,
             ACTIVATION: tl.constexpr, DIVISOR: tl.constexpr):
    dtype: tl.constexpr = x.dtype
    xf = x.to(tl.float32)
    if ACTIVATION == "silu":
        a = (xf * tl.sigmoid(xf)).to(dtype)
    elif ACTIVATION == "relu":
        a = tl.maximum(xf, 0).to(dtype)
    else:
        elu = tl.where(xf > 0, xf, tl.exp(xf) - 1).to(dtype)
        a = (elu.to(tl.float32) + 1).to(dtype)
    a = (a.to(tl.float32) / DIVISOR).to(dtype)
    keep = (rows[:, None] < M) & (cols[None, :] < N)
    keep &= cols[None, :] <= rows[:, None] + QSTART - EXCLUSIVE
    if WINDOW > 0:
        keep &= cols[None, :] > rows[:, None] + QSTART - WINDOW
    return tl.where(keep, a, 0).to(dtype), keep


@triton.jit
def _score_grad(do, v, x, keep, ACTIVATION: tl.constexpr,
                SCALE: tl.constexpr, DIVISOR: tl.constexpr):
    dtype: tl.constexpr = x.dtype
    da = tl.dot(do, tl.trans(v), input_precision="tf32x3").to(dtype)
    da = tl.where(keep, da, 0).to(dtype)
    da = (da.to(tl.float32) / DIVISOR).to(dtype)
    xf = x.to(tl.float32)
    if ACTIVATION == "silu":
        sig = tl.sigmoid(xf)
        deriv = sig * (1 + xf * (1 - sig))
    elif ACTIVATION == "relu":
        deriv = tl.where(xf > 0, 1., 0.)
    else:
        deriv = tl.where(xf > 0, 1., tl.exp(xf))
    db = (da.to(tl.float32) * deriv).to(dtype)
    ds = (db.to(tl.float32) * SCALE).to(dtype)
    return ds, db


@triton.jit
def _forward(Q, K, V, Bias, Out,
             Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
             K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
             V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
             H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
             BIAS_OFFSET: tl.constexpr, HAS_BIAS: tl.constexpr,
             SCALE: tl.constexpr, DIVISOR: tl.constexpr, ACTIVATION: tl.constexpr,
             QSTART: tl.constexpr, EXCLUSIVE: tl.constexpr, WINDOW: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    rows = block * BM + tl.arange(0, BM)
    dim = tl.arange(0, BD)
    q = tl.load(Q + batch * Q0 + head * Q1 + rows[:, None] * Q2 + dim[None, :],
                (rows[:, None] < M) & (dim[None, :] < D), 0)
    acc = tl.zeros((BM, BD), tl.float32)
    end = tl.minimum(N, QSTART + (block + 1) * BM - EXCLUSIVE)
    start = 0
    if WINDOW > 0:
        start = tl.maximum(0, (QSTART + block * BM - WINDOW + 1) // BN * BN)
    for col_start in range(start, end, BN):
        cols = col_start + tl.arange(0, BN)
        k = tl.load(K + batch * K0 + head * K1 + cols[:, None] * K2 + dim[None, :],
                    (cols[:, None] < N) & (dim[None, :] < D), 0)
        v = tl.load(V + batch * V0 + head * V1 + cols[:, None] * V2 + dim[None, :],
                    (cols[:, None] < N) & (dim[None, :] < D), 0)
        x = _scores(q, k, Bias, rows, cols, head, H, BIAS_OFFSET, HAS_BIAS,
                    SCALE, QSTART, M, N)
        a, _ = _weights(x, rows, cols, M, N, QSTART, EXCLUSIVE, WINDOW,
                        ACTIVATION, DIVISOR)
        acc = tl.dot(a, v, acc, input_precision="tf32x3")
    tl.store(Out + (bh * M + rows[:, None]) * D + dim[None, :], acc,
             (rows[:, None] < M) & (dim[None, :] < D))


@triton.jit
def _backward_q(Q, K, V, Bias, DO, DQ, DB,
                Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
                K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr,
                H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                BIAS_OFFSET: tl.constexpr, HAS_BIAS: tl.constexpr, BIAS_GRAD: tl.constexpr,
                SCALE: tl.constexpr, DIVISOR: tl.constexpr, ACTIVATION: tl.constexpr,
                QSTART: tl.constexpr, EXCLUSIVE: tl.constexpr, WINDOW: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    rows = block * BM + tl.arange(0, BM)
    dim = tl.arange(0, BD)
    valid = (rows[:, None] < M) & (dim[None, :] < D)
    q = tl.load(Q + batch * Q0 + head * Q1 + rows[:, None] * Q2 + dim[None, :], valid, 0)
    do = tl.load(DO + batch * O0 + head * O1 + rows[:, None] * O2 + dim[None, :], valid, 0)
    dq = tl.zeros((BM, BD), tl.float32)
    end = tl.minimum(N, QSTART + (block + 1) * BM - EXCLUSIVE)
    start = 0
    if WINDOW > 0:
        start = tl.maximum(0, (QSTART + block * BM - WINDOW + 1) // BN * BN)
    for col_start in range(start, end, BN):
        cols = col_start + tl.arange(0, BN)
        valid_k = (cols[:, None] < N) & (dim[None, :] < D)
        k = tl.load(K + batch * K0 + head * K1 + cols[:, None] * K2 + dim[None, :], valid_k, 0)
        v = tl.load(V + batch * V0 + head * V1 + cols[:, None] * V2 + dim[None, :], valid_k, 0)
        x = _scores(q, k, Bias, rows, cols, head, H, BIAS_OFFSET, HAS_BIAS,
                    SCALE, QSTART, M, N)
        _, keep = _weights(x, rows, cols, M, N, QSTART, EXCLUSIVE, WINDOW,
                           ACTIVATION, DIVISOR)
        ds, db = _score_grad(do, v, x, keep, ACTIVATION, SCALE, DIVISOR)
        dq = tl.dot(ds, k, dq, input_precision="tf32x3")
        if BIAS_GRAD:
            # Reduce each tile's relative-position diagonals before atomics.
            # This needs O(BM * BN) scratch, never an L-by-L gradient buffer.
            offsets = tl.arange(0, triton.next_power_of_2(BM + BN)) - (BM - 1)
            gather_cols = tl.arange(0, BM)[:, None] + offsets[None, :]
            values = tl.gather(db.to(tl.float32), tl.minimum(tl.maximum(gather_cols, 0), BN - 1), 1)
            values = tl.where((gather_cols >= 0) & (gather_cols < BN), values, 0.)
            sums = tl.sum(values, 0)
            index = col_start - block * BM - QSTART + offsets + BIAS_OFFSET
            tl.atomic_add(DB + index * H + head, sums,
                          (index >= 0) & (index <= 2 * BIAS_OFFSET), sem="relaxed")
    tl.store(DQ + (bh * M + rows[:, None]) * D + dim[None, :], dq, valid)


@triton.jit
def _backward_kv(Q, K, V, Bias, DO, DK, DV,
                 Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
                 K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                 V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                 O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr,
                 H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                 BIAS_OFFSET: tl.constexpr, HAS_BIAS: tl.constexpr,
                 SCALE: tl.constexpr, DIVISOR: tl.constexpr, ACTIVATION: tl.constexpr,
                 QSTART: tl.constexpr, EXCLUSIVE: tl.constexpr, WINDOW: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    cols = block * BN + tl.arange(0, BN)
    dim = tl.arange(0, BD)
    valid_k = (cols[:, None] < N) & (dim[None, :] < D)
    k = tl.load(K + batch * K0 + head * K1 + cols[:, None] * K2 + dim[None, :], valid_k, 0)
    v = tl.load(V + batch * V0 + head * V1 + cols[:, None] * V2 + dim[None, :], valid_k, 0)
    dk = tl.zeros((BN, BD), tl.float32)
    dv = tl.zeros((BN, BD), tl.float32)
    start = tl.maximum(0, (block * BN - QSTART + EXCLUSIVE) // BM * BM)
    end = M
    if WINDOW > 0:
        end = tl.minimum(M, (block + 1) * BN - QSTART + WINDOW)
    for row_start in range(start, end, BM):
        rows = row_start + tl.arange(0, BM)
        valid = (rows[:, None] < M) & (dim[None, :] < D)
        q = tl.load(Q + batch * Q0 + head * Q1 + rows[:, None] * Q2 + dim[None, :], valid, 0)
        do = tl.load(DO + batch * O0 + head * O1 + rows[:, None] * O2 + dim[None, :], valid, 0)
        x = _scores(q, k, Bias, rows, cols, head, H, BIAS_OFFSET, HAS_BIAS,
                    SCALE, QSTART, M, N)
        a, keep = _weights(x, rows, cols, M, N, QSTART, EXCLUSIVE, WINDOW,
                           ACTIVATION, DIVISOR)
        ds, _ = _score_grad(do, v, x, keep, ACTIVATION, SCALE, DIVISOR)
        dk = tl.dot(tl.trans(ds), q, dk, input_precision="tf32x3")
        dv = tl.dot(tl.trans(a), do, dv, input_precision="tf32x3")
    tl.store(DK + (bh * N + cols[:, None]) * D + dim[None, :], dk, valid_k)
    tl.store(DV + (bh * N + cols[:, None]) * D + dim[None, :], dv, valid_k)


class _PointwiseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, scale, divisor, activation, query_start,
                exclusive, window):
        batch, heads, length, dim = q.shape
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        bias_offset = (bias.shape[0] - 1) // 2 if bias is not None else 0
        meta = dict(H=heads, M=length, N=k.shape[2], D=dim,
                    BIAS_OFFSET=bias_offset, HAS_BIAS=bias is not None,
                    SCALE=scale, DIVISOR=divisor, ACTIVATION=activation,
                    QSTART=query_start, EXCLUSIVE=exclusive, WINDOW=window,
                    BM=16 if dim == 128 else 32,
                    BN=32 if dim == 128 else 64,
                    BD=triton.next_power_of_2(dim))
        strides = (*q.stride()[:3], *k.stride()[:3], *v.stride()[:3])
        _forward[(triton.cdiv(length, meta['BM']), batch * heads)](
            q, k, v, bias, out, *strides, **meta, num_warps=4, num_stages=1,
            enable_fp_fusion=False)
        ctx.save_for_backward(q, k, v, bias)
        ctx.meta = meta
        ctx.strides = strides
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, bias = ctx.saved_tensors
        meta = ctx.meta
        batch, heads = q.shape[:2]
        # A final contiguous dimension is the only stride restriction in the kernels.
        if do.stride(-1) != 1:
            do = do.contiguous()
        dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        need_bias = bias is not None and ctx.needs_input_grad[3]
        db = torch.zeros(bias.shape, dtype=torch.float32, device=bias.device) if need_bias else None
        _backward_q[(triton.cdiv(q.shape[2], meta['BM']), batch * heads)](
            q, k, v, bias, do, dq, db, *ctx.strides, *do.stride()[:3],
            **meta, BIAS_GRAD=need_bias, num_warps=4, num_stages=1,
            enable_fp_fusion=False)
        _backward_kv[(triton.cdiv(k.shape[2], meta['BN']), batch * heads)](
            q, k, v, bias, do, dk, dv, *ctx.strides, *do.stride()[:3],
            **meta, num_warps=4, num_stages=1, enable_fp_fusion=False)
        if need_bias:
            db = db.to(bias.dtype)
        return dq, dk, dv, db, None, None, None, None, None, None


def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     position_bias: torch.Tensor | None = None, *,
                     scale: float = 1.0, divisor: float = 1.0,
                     activation: str = "silu", query_start: int = 0,
                     causal_diagonal: str = "inclusive",
                     window_size: int | None = None) -> torch.Tensor:
    """Aggregate [B,H,M,D] queries over [B,H,N,D] keys/values.

    Keys have positions 0..N-1 and queries query_start..query_start+M-1.
    The caller selects this backend only for CUDA FP32/BF16, last stride 1,
    no dropout, known causal masks and in-range relative-position indices.
    """
    return _PointwiseAttention.apply(
        q, k, v, position_bias, scale, divisor, activation, query_start,
        int(causal_diagonal == "exclusive"), window_size or 0)
