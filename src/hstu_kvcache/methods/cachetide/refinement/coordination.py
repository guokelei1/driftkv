"""Query-time coordination inside a frozen compact Item+Response read.

Only the coordinator is fitted. Its three gains scale key corrections before
attention activation, value corrections after the weighted compact read, and
the Response increment. Native cache writes and per-row Item codes stay fixed.
There is one mapped history read, with no expanded or persistent corrected K/V.
"""

import math

import torch
from torch import nn


class ReadCoordinator(nn.Module):
    """Three gains per layer and candidate, shared across attention heads.

    ``static`` has three learned logits per layer. ``dynamic`` gives each
    layer a small MLP over the normalized actual query, normalized native
    history read per token, and normalized log history length. No teacher,
    label or correction outcome is an online input. Both modes start at the
    complete frozen D1 read: zero output logits give 1+tanh(0)=1.
    """

    def __init__(self, num_layers, width, mode="dynamic", hidden_width=16,
                 max_length=1024):
        super().__init__()
        if mode not in ("static", "dynamic"):
            raise ValueError("coordination mode must be static or dynamic")
        self.num_layers, self.width = int(num_layers), int(width)
        self.mode, self.hidden_width = mode, int(hidden_width)
        self.max_length = int(max_length)
        if mode == "static":
            self.logits = nn.Parameter(torch.zeros(self.num_layers, 3))
            self.networks = nn.ModuleList()
        else:
            self.register_parameter("logits", None)
            self.networks = nn.ModuleList([
                nn.Sequential(nn.Linear(2*self.width+1, self.hidden_width),
                              nn.SiLU(), nn.Linear(self.hidden_width, 3))
                for _ in range(self.num_layers)])
            for network in self.networks:
                nn.init.zeros_(network[-1].weight)
                nn.init.zeros_(network[-1].bias)

    def get_config(self):
        return dict(num_layers=self.num_layers, width=self.width, mode=self.mode,
                    hidden_width=self.hidden_width, max_length=self.max_length)

    @classmethod
    def from_state(cls, config, state_dict):
        """Restore the small coordinator; D1 checkpoints are separate inputs."""
        result = cls(**config)
        result.load_state_dict(state_dict)
        return result

    def gains(self, layer, query, native, tokens):
        """Return [owners, candidates, (K,V,Response)] without detaching inputs."""
        batch, _, candidates, _ = query.shape
        if self.mode == "static":
            return (1+torch.tanh(self.logits[layer]))[None, None].expand(batch, candidates, 3)
        q = query.transpose(1, 2).flatten(-2)
        per_token = native.transpose(1, 2).flatten(-2)/max(int(tokens), 1)
        # The scalar stabilization protects empty/tiny native reads. Keeping
        # the input dtype also permits useful CPU double-precision references.
        q = q*torch.rsqrt(q.square().mean(-1, keepdim=True)+1e-6)
        per_token = per_token*torch.rsqrt(per_token.square().mean(-1, keepdim=True)+1e-6)
        length = q.new_full((batch, candidates, 1),
                            math.log1p(int(tokens))/math.log1p(self.max_length))
        logits = self.networks[layer](torch.cat((q, per_token, length), dim=-1))
        return 1+torch.tanh(logits)

    def forward(self, layer, query, native, tokens):
        return self.gains(layer, query, native, tokens)

    def persistent_state_bytes(self):
        """Additional per-user state: no gains or query intermediates persist."""
        return 0

    def shared_storage_bytes(self):
        return sum(value.numel()*value.element_size() for value in self.parameters())

    def estimate_added_flops(self, *, tokens, heads, candidates=1, batch=1):
        """Extra serving arithmetic OVER the complete existing D1 read.

        D1 fitting, folding, code maintenance and its complete Item+Response
        read remain separately charged, including at zero/unit gains. MAdd=2;
        tanh/rsqrt=1, SiLU=2. Reductions use the repository's length convention.
        Endpoint comparisons/selections are logical and memory operations.
        No training backward or optional capture/aggregation is included.
        """
        rows, layers, width = int(batch*candidates), self.num_layers, self.width
        if self.mode == "static":
            # The three logits are transformed once then broadcast to B*Q.
            gates = layers*6
        else:
            inputs, hidden = 2*width+1, self.hidden_width
            # Two RMS normalizations, native/N, small MLP, tanh and +1.
            # The length log/divide is computed once per layer invocation.
            gates = layers*(rows*(7*width+6 + 2*inputs*hidden+2*hidden
                                  + 2*hidden*3+6)+2)
        # K scales the position product and affine bias separately to retain
        # the original D1 association at gK=1. V scales its two increments.
        # Response contributes subtract, multiply and add per read coordinate.
        scaling = layers*rows*(int(heads)*(int(tokens)+1)+5*width)
        return dict(gates=int(gates), scaling=int(scaling), total=int(gates+scaling))


def _response_gain(mapped, corrected, gain):
    blended = mapped+gain*(corrected-mapped)
    # Exact endpoint values matter for the frozen control. Plain where at
    # unit gain would suppress d(output)/d(gain), preventing learning from
    # the D1 initialization. The zero-valued term keeps the smooth formula's
    # derivative while changing only floating-point endpoint rounding.
    endpoint = torch.where(gain == 0, mapped,
                           torch.where(gain == 1, corrected, blended))
    if torch.is_grad_enabled() and blended.requires_grad:
        return endpoint.detach()+(blended-blended.detach())
    return endpoint


def make_override(adapter, model, cache, state, coordinator, *, capture_gains=None):
    """Build a differentiable actual-history-read callback.

    ``capture_gains`` may be a caller-owned list: one (layer, detached [B,Q,3])
    tensor is appended per callback. It is disabled by default. A caller that
    captures statistics owns their reduction/lifetime; no autograd graph is
    captured. Native read, self attention and subsequent layers are retained.
    """
    if cache.seq_len != state.seq_len:
        raise ValueError("Item codes must align with the native cache")
    if (coordinator.num_layers != len(model.blocks)
            or coordinator.width != model.cfg.hidden_size):
        raise ValueError("coordinator must match the frozen model")
    if model.training or adapter.training:
        raise ValueError("coordination requires frozen eval-mode D1 and model")
    if any(block.attn.position_bias is not None or block.attn.block_variant != "legacy"
           or block.attn.cfg.activation != "elu_plus1" for block in model.blocks):
        raise ValueError("coordination uses the frozen legacy ELU+1 compact read")
    counts = torch.full((cache.k.shape[1],), cache.seq_len, device=cache.k.device)

    def override(layer, query, native):
        gains = coordinator.gains(layer, query, native, cache.seq_len)
        if capture_gains is not None:
            capture_gains.append((layer, gains.detach()))
        key_gain, value_gain, response_gain = (
            gains[..., index][:, None, :, None] for index in range(3))
        batch, heads, _, dim = query.shape
        k = cache.k[layer].reshape(batch, cache.seq_len, heads, dim).transpose(1, 2)
        v = cache.v[layer].reshape(batch, cache.seq_len, heads, dim).transpose(1, 2)
        hidden = state.hidden[layer]
        projected_query = query @ adapter.folded_k_weight[layer]
        key_bias = (query*adapter.folded_k_bias[layer][None, :, None]).sum(-1, keepdim=True)
        logits = query @ k.transpose(-2, -1)
        logits = (logits + key_gain*(projected_query @ hidden[:, None].transpose(-2, -1))
                  + key_gain*key_bias)
        attention = model.blocks[layer].attn
        weights = attention._activate(logits*attention.scale)
        native_value_read = weights @ v
        hidden_read = weights @ hidden[:, None]
        value_delta = hidden_read @ adapter.folded_v_weight[layer].transpose(-2, -1)
        bias_delta = weights.sum(-1, keepdim=True)*adapter.folded_v_bias[layer][None, :, None]
        mapped = native_value_read+value_gain*value_delta+value_gain*bias_delta
        corrected = adapter.response.layers[layer](query, mapped, counts)
        return _response_gain(mapped, corrected, response_gain)

    return override
