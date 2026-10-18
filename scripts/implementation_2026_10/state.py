"""Chunked in-place current-code/anchor conversion for the frozen A updater.

The old reference remains the numerical oracle.  These methods mutate only the
existing compact buffer; native K/V, item IDs, and shared publication tables are
read-only.  A temporary feature/code tile is released before moving to the next
layer/tile.  Training continues to use the differentiable reference instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from hstu_kvcache.design_one.lightweight import LightweightValueState
from query_relay_2026_10.adapter import _block_apply
from query_relay_2026_10.fixed_features import load_bank
from query_relay_2026_10.precision import StableReleasePrefix


def _nonoverlapping(tensor):
    """The retained contiguous buffer and FIFO slices pass this simple check."""
    span = 1
    for size, stride in reversed(list(zip(tensor.shape, tensor.stride()))):
        if size > 1:
            if stride < span:
                return False
            span += (size - 1) * stride
    return True


@dataclass
class InplaceReleasePrefix(StableReleasePrefix):
    """Same publication tensors and method signatures, one mutable code buffer.

    ``feature_fn(layer, k, v, raw_ids)`` may supply an equivalent preparation
    kernel.  It returns eta (or the reference ``(eta, xi)`` pair).  No features
    are retained in this object.  ``row_chunk`` bounds rows per batch member;
    the prototype supports the current affine A / xi=1 geometry only.
    """

    inverse_blocks: tuple = field(default_factory=tuple, repr=False)
    row_chunk: int = 128
    feature_fn: Callable | None = field(default=None, repr=False)

    def __post_init__(self):
        if not isinstance(self.row_chunk, int) or self.row_chunk <= 0:
            raise ValueError("row_chunk must be a positive integer")
        if any(layer.mode != "A" or layer.xi_width != 1 for layer in self.layers):
            raise ValueError("in-place conversion supports the current A / xi=1 updater")
        if any(layer.code_width != 48 or layer.heads != 6 or layer.rank != 8
               for layer in self.layers):
            raise ValueError("in-place conversion uses shared12/private6 packed as [6,8]")
        if not self.inverse_blocks:
            super().__post_init__()

    @classmethod
    def wrap(cls, prefix, *, row_chunk=128, feature_fn=None):
        if isinstance(prefix, cls) and prefix.row_chunk == row_chunk \
                and prefix.feature_fn is feature_fn:
            return prefix
        # If actual inverses already exist, reuse them without recomputation.
        return cls(prefix.P, prefix.Q, prefix.D, prefix.layers, prefix.release,
                   inverse_blocks=getattr(prefix, "inverse_blocks", ()),
                   row_chunk=row_chunk, feature_fn=feature_fn)

    def _inputs(self, cache, state, item_ids, table_item_ids):
        if torch.is_grad_enabled():
            raise RuntimeError("in-place code conversion is inference-only; enter torch.no_grad()")
        if not isinstance(state, LightweightValueState):
            raise TypeError("in-place conversion requires LightweightValueState")
        codes = state.codes
        layers, batch, rows, width = cache.k.shape
        if cache.v.shape != cache.k.shape or layers != len(self.layers) \
                or cache.seq_len != rows or state.seq_len != rows \
                or codes.shape != (layers, batch, rows, 6, 8):
            raise ValueError("K/V, state, and row geometry disagree")
        if width != self.layers[0].heads * self.layers[0].head_dim:
            raise ValueError("native K/V width disagrees with publication")
        if codes.requires_grad or cache.k.requires_grad or cache.v.requires_grad:
            raise ValueError("in-place conversion requires detached inference state")
        if any(t.device != self.P.device or t.dtype != self.P.dtype
               for t in (codes, cache.k, cache.v, self.Q, self.D)):
            raise ValueError("K/V, codes and publication must share dtype and device")
        if codes.stride(-1) != 1 or codes.stride(-2) != 8 or not _nonoverlapping(codes):
            raise ValueError("compact buffer must have non-overlapping packed row storage")
        if codes.numel():
            storage = codes.untyped_storage().data_ptr()
            if any(t.numel() and t.untyped_storage().data_ptr() == storage
                   for t in (cache.k, cache.v, self.P, self.Q, self.D)):
                raise ValueError("compact buffer must not alias native K/V or publication tables")

        def ids(value, name):
            result = torch.as_tensor(value, device=self.P.device)
            if result.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64,
                                    torch.uint8):
                raise ValueError(f"{name} must contain integer model IDs")
            if result.shape == (rows,):
                result = result[None].expand(batch, rows)
            if result.shape != (batch, rows):
                raise ValueError(f"{name} must have [rows] or [batch,rows] shape")
            result = result.long()
            if result.numel() and int(result.min()) < 0:
                raise ValueError(f"{name} contains a negative model ID")
            return result

        raw = ids(item_ids, "raw item_ids")
        table = raw if table_item_ids is None else ids(table_item_ids, "table_item_ids")
        if table.numel() and int(table.max()) >= self.D.shape[1]:
            raise ValueError("table_item_ids exceeds the publication table")
        # Check the entire raw-ID range before writing even the first chunk.
        # Loading the immutable bank is amortized by its existing device cache.
        for layer in self.layers:
            if layer.feature_mode == "origin_encoder":
                bank = load_bank(layer.feature_bank_path, layer.feature_bank_sha256,
                                 device=self.P.device, dtype=self.P.dtype)
                if raw.numel() and int(raw.max()) >= bank.table.shape[1]:
                    raise ValueError("raw item_ids exceeds the immutable feature bank")
        return codes.view(layers, batch, rows, 48), raw, table

    def _convert(self, cache, state, item_ids, table_item_ids, *, inverse):
        flat, raw, table = self._inputs(cache, state, item_ids, table_item_ids)
        for index, layer in enumerate(self.layers):
            for start in range(0, cache.seq_len, self.row_chunk):
                stop = min(start + self.row_chunk, cache.seq_len)
                item_tile = raw[:, start:stop]
                k = cache.k[index, :, start:stop]
                v = cache.v[index, :, start:stop]
                features = (layer.fixed_features(k, v, item_tile) if self.feature_fn is None
                            else self.feature_fn(layer, k, v, item_tile))
                eta = features[0] if isinstance(features, tuple) else features
                # Preserve reference addition order: block(Q,eta)+D, followed
                # by either current-offset or block(P,anchor)+offset.
                offset = _block_apply(eta, self.Q[index], layer.branches)
                offset = offset + self.D[index][table[:, start:stop]][..., 0]
                source = flat[index, :, start:stop]
                if inverse:
                    rhs = source - offset
                    pieces = []
                    for branch, block in zip(layer.branches, self.inverse_blocks[index], strict=True):
                        section = slice(branch.code_start, branch.code_start + branch.code_width)
                        pieces.append(rhs[..., section] @ block.T)
                    output = torch.cat(pieces, -1)
                else:
                    output = _block_apply(source, self.P[index], layer.branches) + offset
                # Every output coordinate is complete before source is changed.
                source.copy_(output)
        return state

    def materialize(self, cache, anchors, item_ids, *, table_item_ids=None):
        return self._convert(cache, anchors, item_ids, table_item_ids, inverse=False)

    def anchorize(self, cache, current, item_ids, *, table_item_ids=None):
        return self._convert(cache, current, item_ids, table_item_ids, inverse=True)

    materialize_ = materialize
    anchorize_ = anchorize

    def geometry(self, *, batch=1):
        """Explicit compact work; fixed-encoder FLOPs are unchanged by tiling."""
        squares = sum(branch.code_width ** 2 for branch in self.layers[0].branches)
        return dict(row_chunk=self.row_chunk, maximum_tile_rows=batch * self.row_chunk,
                    persistent_extra_row_bytes=0,
                    compact_conversion_flops_per_layer_row=4 * squares + 2 * 48,
                    note="Both block transforms plus two vector adds; fixed eta work and shared table reads also occur. Chunking changes scratch space, not arithmetic complexity.")
