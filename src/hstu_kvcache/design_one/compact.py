"""Execute the frozen Item+Response correction with compact per-row state.

The Item MLP's nonlinear hidden vector is the only persistent user state.
Its linear output is contracted with each query and with the weighted hidden
sum, so no expanded corrected K/V is materialized. This is an algebraic
execution change of the existing checkpoint, not a newly fitted correction.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CompactItemState:
    hidden: torch.Tensor  # [layers, owners, historical positions, hidden width]
    seq_len: int

    def select(self, indices):
        return CompactItemState(self.hidden[:, indices], self.seq_len)

    def storage_bytes(self):
        return self.hidden.numel() * self.hidden.element_size()


class CompactItemResponseAdapter(nn.Module):
    """An execution wrapper around one frozen Item+Response checkpoint.

    Native K/V are supplied by the caller at read time and remain unchanged.
    Folded output weights are shared across all users. Per-user state contains
    the SiLU hidden activation only, in the checkpoint's original precision.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.num_layers = base.num_layers
        self.heads, self.head_dim = base.heads, base.head_dim
        self.max_length = base.max_length
        self.hidden_width = base.item.hidden_width
        self.kind = base.kind
        weights = torch.stack([
            layer.output.weight.detach() * layer.output_scale[:, None]
            for layer in self.layers
        ]).reshape(self.num_layers, 2, self.heads, self.head_dim, self.hidden_width)
        biases = torch.stack([
            layer.output.bias.detach() * layer.output_scale
            for layer in self.layers
        ]).reshape(self.num_layers, 2, self.heads, self.head_dim)
        self.register_buffer("folded_k_weight", weights[:, 0].contiguous())
        self.register_buffer("folded_v_weight", weights[:, 1].contiguous())
        self.register_buffer("folded_k_bias", biases[:, 0].contiguous())
        self.register_buffer("folded_v_bias", biases[:, 1].contiguous())

    @property
    def item(self):
        return self.base.item

    @property
    def response(self):
        return self.base.response

    @property
    def layers(self):
        return self.item.layers

    def get_config(self):
        return self.base.get_config()

    def encode_cache(self, cache, item_features):
        width = self.heads * self.head_dim
        if item_features.shape != (cache.k.shape[1], cache.seq_len, width):
            raise ValueError("item features must align with the native cache rows")
        hidden = []
        for index, layer in enumerate(self.layers):
            features = torch.cat((cache.k[index], cache.v[index],
                                  item_features.to(cache.k)), dim=-1)
            normalized = ((features.to(layer.input_center.dtype) - layer.input_center)
                          / layer.input_scale)
            hidden.append(F.silu(layer.input(normalized)))
        return CompactItemState(torch.stack(hidden), cache.seq_len)

    def history_read(self, attention, layer, query, native_cache, state):
        """Read K+Az+b and V+Cz+d without expanding either correction.

        ELU+1 is applied AFTER combining the native and correction logits.
        Attention weights therefore remain exactly query dependent; the
        nonlinear activation is never moved through a historical reduction.
        """
        if (attention.position_bias is not None or attention.block_variant != "legacy"
                or attention.cfg.activation != "elu_plus1" or attention.training):
            raise ValueError("compact reads require the frozen legacy ELU+1 evaluation path")
        if native_cache.seq_len != state.seq_len:
            raise ValueError("compact hidden rows and native cache must have the same length")
        batch, heads, _, dim = query.shape
        k = native_cache.k[layer].reshape(batch, state.seq_len, heads, dim).transpose(1, 2)
        v = native_cache.v[layer].reshape(batch, state.seq_len, heads, dim).transpose(1, 2)
        hidden = state.hidden[layer]
        projected_query = query @ self.folded_k_weight[layer]
        key_bias = (query * self.folded_k_bias[layer][None, :, None]).sum(-1, keepdim=True)
        logits = query @ k.transpose(-2, -1)
        logits = logits + projected_query @ hidden[:, None].transpose(-2, -1) + key_bias
        weights = attention._activate(logits * attention.scale)
        native_value_read = weights @ v
        hidden_read = weights @ hidden[:, None]
        value_delta = hidden_read @ self.folded_v_weight[layer].transpose(-2, -1)
        bias_delta = weights.sum(-1, keepdim=True) * self.folded_v_bias[layer][None, :, None]
        return native_value_read + value_delta + bias_delta

    def make_history_override(self, model, native_cache, state, counts=None):
        if counts is None:
            counts = torch.full((native_cache.k.shape[1],), native_cache.seq_len,
                                device=native_cache.k.device)

        def override(layer, query, native):
            mapped = self.history_read(model.blocks[layer].attn, layer, query,
                                       native_cache, state)
            return self.response.layers[layer](query, mapped, counts)

        return override

    def projection_storage_bytes(self):
        """Shared folded decoder buffers; never multiplied by user count."""
        return sum(value.numel() * value.element_size() for value in (
            self.folded_k_weight, self.folded_v_weight,
            self.folded_k_bias, self.folded_v_bias))

    def setup_flops(self):
        """One output-scale multiplication per decoder weight and bias."""
        return sum(value.numel() for value in (
            self.folded_k_weight, self.folded_v_weight,
            self.folded_k_bias, self.folded_v_bias))

    def estimate_flops(self, *, batch=1, tokens=1, candidates=1):
        """Use the retained multiply-add=2, activation=1 arithmetic ledger.

        Encoding retains normalization, the first affine map and SiLU only.
        The read includes its complete native QK/AV work: the caller's earlier
        native read is still executed, and no cancellation is claimed. Added
        contractions are qA, (qA)z, weights*z and its output projection, plus
        the two bias terms and additions. Response is charged exactly once.
        """
        width, hidden = self.heads * self.head_dim, self.hidden_width
        encode = sum(2 * layer.input.in_features
                     + 2 * layer.input.in_features * hidden + 2 * hidden
                     for layer in self.layers)
        native_read = 4 * width + 3 * self.heads
        compact_per_token = 4 * self.heads * hidden + 3 * self.heads
        compact_per_query = 4 * width * hidden + 5 * width
        return dict(
            token_transform=int(batch * tokens * encode),
            extra_history_read=int(self.num_layers * batch * candidates * (
                tokens * (native_read + compact_per_token) + compact_per_query)),
            **self.response.estimate_flops(batch=batch, candidates=candidates),
        )
