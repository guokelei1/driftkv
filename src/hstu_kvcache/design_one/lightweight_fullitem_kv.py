"""One narrow code supplies factored Key and Value read corrections.

The native cache is never rewritten. A Key decoder changes attention scores
inside the same history scan; the existing Value decoder changes its reduction.
Construction consumes no random numbers: calibration first fits the unchanged
Value adapter, then copies it and adds an exactly zero Key decoder.
"""
import torch
from torch import nn

from .lightweight_fullitem import FullItemValueLayer, LightweightFullItemAdapter


class FullItemKVLayer(FullItemValueLayer):
    def __init__(self, heads, head_dim, rank):
        nn.Module.__init__(self)
        self.heads, self.head_dim, self.rank = heads, head_dim, rank
        self.input_width = (heads + 2) * head_dim
        self.register_buffer("input_center", torch.zeros(heads, self.input_width))
        self.register_buffer("input_scale", torch.ones(heads, self.input_width))
        self.input_weight = nn.Parameter(torch.zeros(heads, self.input_width, rank))
        self.input_bias = nn.Parameter(torch.zeros(heads, rank))
        self.decoder = nn.Parameter(torch.zeros(heads, rank, head_dim))
        self.key_decoder = nn.Parameter(torch.zeros(heads, rank, head_dim))

    def read(self, attention, q, k, v, codes, decoder=None, key_decoder=None):
        if (attention.position_bias is not None or attention.training
                or attention.block_variant != "legacy" or attention.cfg.activation != "elu_plus1"):
            raise ValueError("factored KV reads require frozen legacy ELU+1 attention")
        b, n, _ = k.shape
        keys = k.to(q.dtype).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        values = v.to(q.dtype).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        z = codes.to(q.dtype).transpose(1, 2)
        dk = self.key_decoder if key_decoder is None else key_decoder
        dv = self.decoder if decoder is None else decoder
        projected = q @ dk.to(q.dtype).transpose(-2, -1)[None]
        scores = q @ keys.transpose(-2, -1) + projected @ z.transpose(-2, -1)
        weights = attention.attn_dropout(attention._activate(scores * attention.scale))
        aggregate = weights @ torch.cat((values, z), dim=-1)
        native, latent = aggregate.split((self.head_dim, self.rank), dim=-1)
        return native + latent @ dv.to(q.dtype)[None]


class LightweightFullItemKVAdapter(LightweightFullItemAdapter):
    kind = "design_one_lightweight_full_item_kv_v1"

    def __init__(self, num_layers, heads, head_dim, rank=8, response_hidden_width=0, max_length=1024):
        nn.Module.__init__(self)
        if response_hidden_width:
            raise ValueError("this factored KV candidate has no Response")
        self.num_layers, self.heads, self.head_dim = num_layers, heads, head_dim
        self.rank, self.response_hidden_width, self.max_length = rank, 0, max_length
        self.layers = nn.ModuleList([FullItemKVLayer(heads, head_dim, rank) for _ in range(num_layers)])
        self.response = None

    @classmethod
    def from_value(cls, value):
        """Copy every fitted Value tensor; add zeros without touching RNG state."""
        reference = next(value.parameters())
        result = cls(**value.get_config()).to(device=reference.device, dtype=reference.dtype)
        missing, unexpected = result.load_state_dict(value.state_dict(), strict=False)
        expected = [f"layers.{i}.key_decoder" for i in range(value.num_layers)]
        if missing != expected or unexpected:
            raise RuntimeError("Value-to-KV conversion did not preserve the known tensor layout")
        return result

    def estimate_flops(self, *, batch=1, tokens=1, candidates=1):
        result = super().estimate_flops(batch=batch, tokens=tokens, candidates=candidates)
        h, d, r = self.heads, self.head_dim, self.rank
        key = self.num_layers * batch * candidates * (2*h*d*r + (2*h*r+h)*tokens)
        result["extra_history_read"] += int(key)
        return result
