"""Inference-only affine-A code conversion, fused over layers and rows.

One program owns a complete shared/private branch of each output row. Native
K/V and the frozen item table are read-only; current/anchor codes overwrite the
same compact allocation. All native and compact dot products use TF32x3, while
offset arithmetic remains FP32. This backend requires validated GPU item IDs;
the session's CPU item mapping checks bounds before dispatch, without a GPU
scalar round trip for each layer. The old stable prefix is the numerical oracle.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "B", "KL", "KB", "KN", "VL", "VB", "VN",
                               "CL", "CB", "CN", "IB", "IN"])
def _convert(K, V, Codes, Ids, Bank, Bias, Weight, Q, Transform, D,
             N, B, M: tl.constexpr,
             KL, KB, KN, VL, VB, VN, CL, CB, CN, IB, IN,
             PRIVATE: tl.constexpr, INVERSE: tl.constexpr,
             INPUT: tl.constexpr, WIDTH: tl.constexpr, ROWS: tl.constexpr):
    block = tl.program_id(0)
    layer = tl.program_id(1)
    branch = tl.program_id(2)
    r = block * ROWS + tl.arange(0, ROWS)
    b, n = r // N, r % N
    valid = r < B * N
    d = tl.arange(0, 64)
    c = tl.arange(0, 16)
    if PRIVATE:
        code_start = 12 + branch * 6
        native_start = branch * 32
        native_width = 32
        weight_base = (layer * 6 + branch) * INPUT * 16
    else:
        code_start = 0
        native_start = 0
        native_width = 192
        weight_base = layer * INPUT * 16
    native = tl.full((ROWS, 16), 0, tl.float32)
    # Bound live operands rather than retaining a full 384-coordinate shared
    # dot. The first prototype's large live operands were much slower on A40.
    for start in range(0, native_width * 2, 64):
        feature_index = start + d
        native_index = native_start + feature_index % native_width
        key = tl.load(K + layer * KL + b[:, None] * KB + n[:, None] * KN + native_index[None, :],
                      valid[:, None] & (feature_index[None, :] < native_width), 0)
        value = tl.load(V + layer * VL + b[:, None] * VB + n[:, None] * VN + native_index[None, :],
                        valid[:, None] & (feature_index[None, :] >= native_width), 0)
        weight = tl.load(Weight + weight_base + feature_index[:, None] * 16 + c[None, :])
        native += tl.dot(key + value, weight, input_precision="tf32x3")
    ids = tl.load(Ids + b * IB + n * IN, valid, 0)
    item = tl.load(Bank + layer * M * 48 + ids[:, None] * 48 + code_start + c[None, :],
                   valid[:, None] & (c[None, :] < WIDTH), 0)
    bias = tl.load(Bias + layer * 48 + code_start + c, c < WIDTH, 0)
    feature = (native + item) + bias[None, :]
    eta = feature / (1.0 + tl.exp(-feature))
    # Both compact matrices are stored [out,in], hence transpose in the dot.
    q = tl.load(Q + layer * 48 * 48 + (code_start + c[None, :]) * 48 + code_start + c[:, None],
                (c[:, None] < WIDTH) & (c[None, :] < WIDTH), 0)
    offset = tl.dot(eta, q, input_precision="tf32x3")
    offset += tl.load(D + layer * M * 48 + ids[:, None] * 48 + code_start + c[None, :],
                      valid[:, None] & (c[None, :] < WIDTH), 0)
    ptr = Codes + layer * CL + b[:, None] * CB + n[:, None] * CN + code_start + c[None, :]
    source = tl.load(ptr, valid[:, None] & (c[None, :] < WIDTH), 0)
    transform = tl.load(Transform + layer * 48 * 48 + (code_start + c[None, :]) * 48 + code_start + c[:, None],
                        (c[:, None] < WIDTH) & (c[None, :] < WIDTH), 0)
    if INVERSE:
        output = tl.dot(source - offset, transform, input_precision="tf32x3")
    else:
        output = tl.dot(source, transform, input_precision="tf32x3") + offset
    tl.store(ptr, output, valid[:, None] & (c[None, :] < WIDTH))


class FusedReleasePrefix:
    """Two launches per conversion, preserving the selected 6-head geometry."""

    def __init__(self, prefix, bank):
        self.reference, self.bank = prefix, bank
        self.P, self.Q, self.D = prefix.P, prefix.Q, prefix.D
        self.layers, self.release = prefix.layers, prefix.release
        if self.P.dtype != torch.float32 or not self.P.is_cuda:
            raise ValueError("this backend uses CUDA FP32")
        if any(layer.mode != "A" or layer.heads != 6 or layer.head_dim != 32
               or layer.code_width != 48 or layer.feature_mode != "origin_encoder"
               for layer in self.layers):
            raise ValueError("expected the selected Medium affine-A encoder")
        if not all(t.is_contiguous() for t in (self.P, self.Q, self.D, bank.table, bank.bias)):
            raise ValueError("shared publication tensors must be contiguous")
        shared = torch.zeros(len(self.layers), 512, 16, device=self.P.device)
        private = torch.zeros(len(self.layers), 6, 64, 16, device=self.P.device)
        for l, weights in enumerate(bank.native_weights):
            shared[l, :384, :12].copy_(weights[0])
            for h in range(6):
                private[l, h, :, :6].copy_(weights[h + 1])
        self.shared_weights, self.private_weights = shared, private
        self.inverse = torch.stack([torch.block_diag(*blocks) for blocks in prefix.inverse_blocks])

    @classmethod
    def wrap(cls, prefix, *, bank):
        return cls(prefix, bank)

    def _run(self, cache, state, ids, *, inverse):
        if torch.is_grad_enabled():
            raise RuntimeError("conversion requires torch.no_grad()")
        if not isinstance(ids, torch.Tensor) or ids.device != self.P.device or ids.dtype != torch.long:
            raise ValueError("session must provide validated int64 IDs on the serving GPU")
        layers, batch, rows, width = cache.k.shape
        code = state.codes
        if cache.v.shape != cache.k.shape or width != 192 or layers != len(self.layers) \
                or ids.shape != (batch, rows) or code.shape != (layers, batch, rows, 6, 8):
            raise ValueError("cache / code / item geometry disagrees")
        if code.stride(-1) != 1 or code.stride(-2) != 8 or code.dtype != torch.float32:
            raise ValueError("packed FP32 code rows required")
        if rows == 0:
            return state
        for private in (False, True):
            _convert[(triton.cdiv(batch * rows, 16), layers, 6 if private else 1)](
                cache.k, cache.v, code, ids, self.bank.table, self.bank.bias,
                self.private_weights if private else self.shared_weights,
                self.Q, self.inverse if inverse else self.P, self.D,
                rows, batch, self.D.shape[1], *cache.k.stride()[:3], *cache.v.stride()[:3],
                *code.stride()[:3], *ids.stride(), private, inverse,
                64 if private else 512, 6 if private else 12, 16,
                num_warps=4, enable_fp_fusion=False)
        return state

    def anchorize(self, cache, state, item_ids, *, table_item_ids=None):
        if table_item_ids is not None:
            raise ValueError("full catalogue uses native model IDs")
        return self._run(cache, state, item_ids, inverse=True)

    def materialize(self, cache, state, item_ids, *, table_item_ids=None):
        if table_item_ids is not None:
            raise ValueError("full catalogue uses native model IDs")
        return self._run(cache, state, item_ids, inverse=False)

    def storage_bytes(self):
        return self.reference.storage_bytes() + sum(t.numel() * t.element_size()
            for t in (self.shared_weights, self.private_weights, self.inverse))

    def geometry(self, *, batch=1, rows=1024):
        useful = 2 * (384 * 12 + 6 * 64 * 6) + 4 * (12**2 + 6 * 6**2)
        padded = 2 * (384 * 16 + 6 * 64 * 16) + 4 * 7 * 16**2
        return dict(layers=len(self.layers), batch=batch, rows=rows,
            useful_dot_flops=len(self.layers)*batch*rows*useful,
            tile_padded_dot_flops=len(self.layers)*triton.cdiv(batch*rows, 16)*16*padded,
            extra_shared_bytes=self.storage_bytes()-self.reference.storage_bytes(),
            extra_persistent_row_bytes=0, conversion_launches=2,
            note="Dot products only; vector/activation work is separate. Padded logical matrix work is not TF32x3 instruction count. Useful FLOPs alone do not describe this kernel's executed work.")

