"""Small shared nonlinear read corrections for the next Design 1 probes.

Response maps use the actual branch query and its native history response.
Token maps create a separate, reusable read view; they never write native K/V.
The context variant additionally uses two immutable causal fields per row.
"""

import torch
from torch import nn
from torch.nn import functional as F

from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.models import HSTUKVCache


class _NormalizedMLP(nn.Module):
    def __init__(self, heads, head_dim, hidden_width, output_width, extra_inputs=0):
        super().__init__()
        self.heads, self.head_dim = int(heads), int(head_dim)
        self.hidden_width = int(hidden_width)
        self.width = self.heads * self.head_dim
        input_width = 2 * self.width + extra_inputs
        self.register_buffer("input_center", torch.zeros(input_width))
        self.register_buffer("input_scale", torch.ones(input_width))
        self.register_buffer("output_scale", torch.ones(output_width))
        self.input = nn.Linear(input_width, self.hidden_width)
        self.output = nn.Linear(self.hidden_width, output_width)
        # Exact identity at publication before fitting; the first update can
        # train the output map, then gradients reach the hidden map as well.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @torch.no_grad()
    def set_normalization(self, center, scale, output_scale):
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("input normalization scales must be finite and positive")
        if not torch.isfinite(center).all() or not torch.isfinite(output_scale).all():
            raise ValueError("normalization must be finite")
        self.input_center.copy_(center)
        self.input_scale.copy_(scale)
        self.output_scale.copy_(output_scale)

    def get_config(self):
        return dict(heads=self.heads, head_dim=self.head_dim, hidden_width=self.hidden_width)

    def _delta(self, features):
        normalized = (features.to(self.input_center.dtype) - self.input_center) / self.input_scale
        return self.output(F.silu(self.input(normalized))) * self.output_scale

    def _network_flops(self, rows):
        """Analytic operations with multiply-add=2 and activation/derivative=1.

        Forward includes normalization, both affine maps, SiLU and output
        units. Backward assumes frozen inputs, as in layerwise calibration:
        both weight/bias gradients, hidden gradient and output-unit scaling.
        Reduction counts use rows rather than rows-1, matching the existing
        ledgers. Loss, optimizer, teacher and attention work are separate.
        These arithmetic estimates are not measured kernel instructions.
        """
        inputs, hidden, outputs = self.input.in_features, self.hidden_width, self.output.out_features
        forward = rows * (2 * inputs + 2 * hidden * (inputs + outputs)
                          + 2 * hidden + 2 * outputs)
        backward = rows * (2 * inputs * hidden + 4 * hidden * outputs
                           + 2 * hidden + 2 * outputs)
        return dict(forward=int(forward), backward=int(backward))


class NonlinearResponseLayer(_NormalizedMLP):
    """Cross-head delta(q, r/N), returned in per-event response units."""

    def __init__(self, heads, head_dim, hidden_width=256):
        super().__init__(heads, head_dim, hidden_width, heads * head_dim)

    def features(self, query, native, counts):
        # q/r are [B,H,Q,D]; each independent query gives one cross-head row.
        query_flat = query.transpose(1, 2).flatten(2)
        native_flat = native.transpose(1, 2).flatten(2)
        count = counts.to(device=native.device, dtype=native.dtype).clamp_min(1)
        return torch.cat((query_flat, native_flat / count[:, None, None]), dim=-1)

    def delta_rate(self, query, native, counts):
        rate = self._delta(self.features(query, native, counts))
        return rate.reshape(query.shape[0], query.shape[2], self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query, native, counts):
        rate = self.delta_rate(query, native, counts).to(native.dtype)
        count = counts.to(device=native.device, dtype=native.dtype)
        return native + count[:, None, None, None] * rate

    def fit_flops(self, *, batch=1, candidates=1):
        """delta_rate forward/backward only; frozen q/r, no loss or optimizer."""
        rows = int(batch) * int(candidates)
        result = self._network_flops(rows)
        result["forward"] += rows * self.width  # r/N
        return result

    def forward_flops(self, *, batch=1, candidates=1):
        return (self.fit_flops(batch=batch, candidates=candidates)["forward"]
                + 2 * int(batch) * int(candidates) * self.width)  # N*rate + r


class KVViewLayer(_NormalizedMLP):
    """Independent cross-head [K,V] residuals, shared across all producers."""

    def __init__(self, heads, head_dim, hidden_width=32):
        super().__init__(heads, head_dim, hidden_width, 2 * heads * head_dim)

    def map_tokens(self, k, v):
        source = torch.cat((k, v), dim=-1)
        mapped = source + self._delta(source).to(source.dtype)
        return mapped.split(self.width, dim=-1)

    def fit_flops(self, *, batch=1, length=1):
        """Token-map forward/backward; mapped read/loss/optimizer are separate."""
        rows = int(batch) * int(length)
        result = self._network_flops(rows)
        result["forward"] += rows * 2 * self.width  # source + residual
        return result

    def token_forward_flops(self, *, batch=1, length=1):
        return self.fit_flops(batch=batch, length=length)["forward"]


class ContextKVViewLayer(KVViewLayer):
    """Joint [K,V] residual conditioned on producer and old fraction at write."""

    def __init__(self, heads, head_dim, hidden_width=32):
        _NormalizedMLP.__init__(self, heads, head_dim, hidden_width,
                                2 * heads * head_dim, extra_inputs=2)

    def map_tokens(self, k, v, context):
        source = torch.cat((k, v), dim=-1)
        features = torch.cat((source, context.to(source)), dim=-1)
        mapped = source + self._delta(features).to(source.dtype)
        return mapped.split(self.width, dim=-1)


class _Adapter(nn.Module):
    layer_class = None

    def __init__(self, num_layers, heads, head_dim, hidden_width, max_length=1024):
        super().__init__()
        self.num_layers, self.heads, self.head_dim = int(num_layers), int(heads), int(head_dim)
        self.hidden_width, self.max_length = int(hidden_width), int(max_length)
        self.layers = nn.ModuleList([
            self.layer_class(heads, head_dim, hidden_width) for _ in range(num_layers)
        ])

    def get_config(self):
        return dict(num_layers=self.num_layers, heads=self.heads, head_dim=self.head_dim,
                    hidden_width=self.hidden_width, max_length=self.max_length)

    def export_state(self):
        return dict(config=self.get_config(),
                    state_dict={key: value.detach().cpu().clone()
                                for key, value in self.state_dict().items()})

    @classmethod
    def from_state_dict(cls, state):
        module = cls(**state["config"])
        # Saved tensor precision is retained; serving may explicitly choose FP32.
        first = next(iter(state["state_dict"].values()))
        module.to(dtype=first.dtype)
        module.load_state_dict(state["state_dict"])
        return module


class NonlinearResponseAdapter(_Adapter):
    kind = "design_one_nonlinear_response_v1"
    layer_class = NonlinearResponseLayer

    def __init__(self, num_layers, heads, head_dim, hidden_width=256, max_length=1024):
        super().__init__(num_layers, heads, head_dim, hidden_width, max_length)

    def make_history_override(self, counts, fitted_layers=None):
        prefix = len(self.layers) if fitted_layers is None else fitted_layers

        def override(layer, query, native):
            return self.layers[layer](query, native, counts) if layer < prefix else native

        return override

    def estimate_flops(self, *, batch=1, candidates=1, tokens=1):
        return dict(candidate_reads=sum(layer.forward_flops(batch=batch, candidates=candidates)
                                        for layer in self.layers))


class KVReadViewAdapter(_Adapter):
    kind = "design_one_kv_view_v1"
    layer_class = KVViewLayer

    def __init__(self, num_layers, heads, head_dim, hidden_width=32, max_length=1024):
        super().__init__(num_layers, heads, head_dim, hidden_width, max_length)

    def map_cache(self, cache, fitted_layers=None):
        """Map all supplied rows; callers retain and update this separate view.

        A fitted prefix is useful when capturing the next calibration layer's
        actual query. An empty prefix has zero layers and performs no mapping.
        """
        prefix = len(self.layers) if fitted_layers is None else fitted_layers
        if prefix == 0:
            return HSTUKVCache(cache.k[:0], cache.v[:0], cache.seq_len)
        return HSTUKVCache.from_layer_list([
            self.layers[index].map_tokens(cache.k[index], cache.v[index])
            for index in range(prefix)
        ], cache.seq_len)

    def make_history_override(self, model, mapped_cache):
        def override(layer, query, native):
            if layer >= mapped_cache.k.shape[0]:
                return native
            return history_read(model.blocks[layer].attn, query,
                                mapped_cache.k[layer], mapped_cache.v[layer])

        return override

    def estimate_flops(self, *, batch=1, tokens=1, candidates=1):
        """Charge stored token transforms and the actual second history read.

        The shared reader already executes its native read before this view's
        override. No cancellation is claimed. The frozen legacy ELU+1 read is
        QK + scale/ELU/+1 + AV, without a validity-mask multiplication.
        """
        width = self.heads * self.head_dim
        return dict(
            token_transform=sum(layer.token_forward_flops(batch=batch, length=tokens)
                                for layer in self.layers),
            extra_history_read=int(len(self.layers) * batch * candidates * tokens
                                   * (4 * width + 3 * self.heads)),
        )


class ContextKVReadViewAdapter(KVReadViewAdapter):
    """Read view with per-row [is_parent, old_fraction_at_write] inputs.

    Context remains fixed after publication, so retained mapped rows survive
    later appends unchanged. This model maps all producers; context is an
    input to the learned map, not a hard producer gate.
    """

    kind = "design_one_context_kv_view_v1"
    layer_class = ContextKVViewLayer

    def map_cache(self, cache, context=None, fitted_layers=None):
        prefix = len(self.layers) if fitted_layers is None else fitted_layers
        if prefix == 0:
            return HSTUKVCache(cache.k[:0], cache.v[:0], cache.seq_len)
        if context is None:
            # Only the initial pure-Parent state has this implicit context.
            context = cache.k.new_ones((cache.k.shape[1], cache.seq_len, 2))
        if context.shape != (cache.k.shape[1], cache.seq_len, 2):
            raise ValueError("write context must align with the supplied K/V rows")
        return HSTUKVCache.from_layer_list([
            self.layers[index].map_tokens(cache.k[index], cache.v[index], context)
            for index in range(prefix)
        ], cache.seq_len)


def append_context(context, old_length, width, max_length):
    """Publish immutable context for native Current rows and evict in order.

    Every entering row records [0, parent_count / capped_window_length] for
    its own causal window INCLUDING that row, exactly as rolling-band append.
    Retained rows keep their original birth fraction even after all Parent
    rows have departed. Batch owners can have different producer histories.
    """
    if context.shape[1:] != (old_length, 2) or not 0 <= old_length <= max_length:
        raise ValueError("context must match the actual retained native cache")
    if not 0 <= width <= max_length:
        raise ValueError("append width must fit the retained context window")
    if width == 0:
        return context
    batch = context.shape[0]
    removed = max(0, old_length + width - max_length)
    parent_count = context[:, :, 0].sum(dim=1)
    # Only rows evicted by this append need a cumulative reduction.
    departed = torch.cat((context.new_zeros((batch, 1)),
                          context[:, :removed, 0].cumsum(dim=1)), dim=1)
    positions = torch.arange(1, width + 1, device=context.device) + old_length
    evictions = (positions - max_length).clamp_min(0)
    lengths = positions.clamp_max(max_length).to(context.dtype)
    entering = context.new_zeros((batch, width, 2))
    entering[:, :, 1] = (parent_count[:, None] - departed[:, evictions]) / lengths[None]
    return torch.cat((context[:, removed:], entering), dim=1)


def append_context_flops(*, batch=1, old_length, width, max_length):
    """Context arithmetic only; initialization/copies and integer indices cost0.

    Count Parent flags once, cumulatively count only the evicted prefix, then
    subtract and divide for each entering row. As in the model ledger, each
    reduction charges its input length rather than length-1.
    """
    if width == 0:
        return 0
    removed = max(0, old_length + width - max_length)
    return int(batch * (old_length + removed + 2 * width))
