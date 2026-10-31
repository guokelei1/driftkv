"""Development candidate: one native attention with a narrow Value residual.

Each head stores only a few nonlinear code coordinates per historical row.
The native QK activation is evaluated once; one AV product aggregates native
values and these codes together. There is no corrected-Key path or second
attention. The PyTorch reference concatenates V and codes into a temporary
buffer at every read. It is not a fused kernel or a demonstrated speedup.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from hstu_kvcache.adaptation.reader import history_read
from .nonlinear import NonlinearResponseLayer


@dataclass
class LightweightValueState:
    codes: torch.Tensor  # [layers, owners, positions, heads, rank] or explicit flat [...,code_width]
    seq_len: int

    def select(self, indices):
        return LightweightValueState(self.codes[:, indices], self.seq_len)

    def storage_bytes(self):
        return self.codes.numel() * self.codes.element_size()


class LightweightValueLayer(nn.Module):
    def __init__(self, heads, head_dim, rank):
        super().__init__()
        self.heads, self.head_dim, self.rank = heads, head_dim, rank
        self.register_buffer("input_center", torch.zeros(heads, 3 * head_dim))
        self.register_buffer("input_scale", torch.ones(heads, 3 * head_dim))
        self.input_weight = nn.Parameter(torch.empty(heads, 3 * head_dim, rank))
        self.input_bias = nn.Parameter(torch.zeros(heads, rank))
        # No output bias: the narrow weighted sum suffices, without a separate
        # attention-weight-mass reduction. Zero decoder gives native identity.
        self.decoder = nn.Parameter(torch.zeros(heads, rank, head_dim))
        nn.init.uniform_(self.input_weight, -1 / math.sqrt(3 * head_dim),
                         1 / math.sqrt(3 * head_dim))

    def encode(self, k, v, item_features):
        shape = (*k.shape[:-1], self.heads, self.head_dim)
        features = torch.cat((k.reshape(shape), v.reshape(shape),
                              item_features.reshape(shape)), dim=-1)
        features = (features.to(self.input_weight.dtype) - self.input_center) / self.input_scale
        encoded = features.transpose(1, 2) @ self.input_weight[None]
        return F.silu(encoded + self.input_bias[None, :, None]).transpose(1, 2).to(k.dtype)

    def read(self, attention, q, k, v, codes, decoder=None):
        if (attention.position_bias is not None or attention.training
                or attention.block_variant != "legacy" or attention.cfg.activation != "elu_plus1"):
            raise ValueError("lightweight reads use the frozen legacy ELU+1 evaluation operator")
        batch, length, _ = k.shape
        shape = (batch, length, self.heads, self.head_dim)
        keys = k.to(q.dtype).reshape(shape).transpose(1, 2)
        values = v.to(q.dtype).reshape(shape).transpose(1, 2)
        weights = attention.attn_dropout(attention._activate(q @ keys.transpose(-2, -1) * attention.scale))
        packed = torch.cat((values, codes.to(q.dtype).transpose(1, 2)), dim=-1)
        aggregated = weights @ packed
        native, code_read = aggregated.split((self.head_dim, self.rank), dim=-1)
        decoder = self.decoder if decoder is None else decoder
        return native + code_read @ decoder.to(q.dtype)[None]


class LightweightValueAdapter(nn.Module):
    """Small head-local encoders plus an optional small cross-head Response.

Native cache writes remain the caller's responsibility. Encode arriving rows
only, then concatenate/evict codes with precisely the same row indices as KV.
Changing release parameters requires re-encoding the retained codes. This
module does not implement a lazy publication policy or change native writes.
"""
    kind = "design_one_lightweight_value_v1"

    def __init__(self, num_layers, heads, head_dim, rank=4,
                 response_hidden_width=16, max_length=1024):
        super().__init__()
        self.num_layers, self.heads, self.head_dim = num_layers, heads, head_dim
        self.rank, self.response_hidden_width, self.max_length = rank, response_hidden_width, max_length
        self.layers = nn.ModuleList([
            LightweightValueLayer(heads, head_dim, rank) for _ in range(num_layers)])
        self.response = (nn.ModuleList([
            NonlinearResponseLayer(heads, head_dim, hidden_width=response_hidden_width)
            for _ in range(num_layers)]) if response_hidden_width else None)

    def get_config(self):
        return dict(num_layers=self.num_layers, heads=self.heads, head_dim=self.head_dim,
                    rank=self.rank, response_hidden_width=self.response_hidden_width,
                    max_length=self.max_length)

    def encode_cache(self, cache, item_features):
        if item_features.shape != (cache.k.shape[1], cache.seq_len, self.heads * self.head_dim):
            raise ValueError("current item features must align with retained native KV rows")
        codes = [layer.encode(cache.k[index], cache.v[index], item_features)
                 for index, layer in enumerate(self.layers)]
        return LightweightValueState(torch.stack(codes), cache.seq_len)

    def make_history_read(self, state, counts=None, fitted_layers=None, strengths=None):
        prefix = self.num_layers if fitted_layers is None else fitted_layers
        # During offline task calibration gradients can reach the shared
        # strengths. Serving folds the selected strengths into decoder weights
        # once at release; it needs no extra per-query scaling operation.
        decoders = (None if strengths is None else
                    [layer.decoder * strengths[index] for index, layer in enumerate(self.layers)])

        def read(layer, attention, q, k, v):
            if layer >= prefix:
                return history_read(attention, q, k, v)
            if k.shape[1] != state.seq_len or k.shape[:2] != state.codes.shape[1:3]:
                raise ValueError("lightweight codes must align with native KV owners and rows")
            mapped = self.layers[layer].read(attention, q, k, v, state.codes[layer],
                decoder=None if decoders is None else decoders[layer])
            if self.response is not None:
                lengths = (torch.full((q.shape[0],), state.seq_len, device=q.device)
                           if counts is None else counts)
                mapped = self.response[layer](q, mapped, lengths)
            return mapped

        return read

    def estimate_flops(self, *, batch=1, tokens=1, candidates=1):
        """Added arithmetic only, multiply-add=2 and SiLU=1, as old ledgers.

        Source/teacher collection, fitting and validation are separate costs.
        Native QK/activation/AV is performed once and charged to the native
        reader. Copying the temporary concatenation has bytes, not FLOPs.
        """
        heads, dim, rank, layers = self.heads, self.head_dim, self.rank, self.num_layers
        encode = heads * (6 * dim + 6 * dim * rank + 2 * rank)
        read = 2 * heads * tokens * rank + 2 * heads * rank * dim + heads * dim
        response = (sum(layer.forward_flops(batch=batch, candidates=candidates)
                        for layer in self.response) if self.response is not None else 0)
        return dict(token_transform=int(layers * batch * tokens * encode),
                    extra_history_read=int(layers * batch * candidates * read),
                    candidate_reads=int(response))

    def packing_bytes(self, *, batch=1, tokens=1, element_size=4):
        """Temporary concatenation bytes written for one all-layer read call.

        Inputs of the same aggregate size are read by cat, and the AV consumes
        its output. Concurrent peak scratch is one layer. Q queries in a batch
        share a pack; separate query calls do not. No dense KV copy persists.
        """
        per_layer = batch * tokens * self.heads * (self.head_dim + self.rank) * element_size
        return dict(total_written=self.num_layers * per_layer, peak_scratch=per_layer)
