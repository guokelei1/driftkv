"""Final-teacher-logit coordinates for the existing codes and both decoders."""
import math

import torch
from torch import nn

from .joint_value_refinement import JointValueRefinement


@torch.no_grad()
def fit_key_scales(adapter, rows, train_uids):
    """Fit-only RMS coordinates using the already prepared frozen codes.

    A unit Key coordinate has roughly native-Key magnitude when independent
    code coordinates are combined. Zero code energy uses unit code RMS; zero
    Key energy is floored at 1e-6. No validation row or extra encoding is used.
    """
    ref = next(adapter.parameters())
    key_sum = torch.zeros(adapter.num_layers, adapter.heads, device=ref.device, dtype=ref.dtype)
    code_sum = torch.zeros_like(key_sum)
    tokens = 0
    flops = 0
    for uid in train_uids:
        row = rows[uid]
        k = row["cache"].k.reshape(adapter.num_layers, 1, -1, adapter.heads, adapter.head_dim)
        z = row["state"].codes
        key_sum += k.square().sum((1, 2, 4))
        code_sum += z.square().sum((1, 2, 4))
        tokens += row["cache"].seq_len
        flops += 2*(k.numel()+z.numel()) + 2*key_sum.numel()
    key_rms = (key_sum/(tokens*adapter.head_dim)).sqrt().clamp_min(1e-6)
    code_rms = (code_sum/(tokens*adapter.rank)).sqrt()
    denominator = torch.where(code_rms > 1e-6, code_rms, torch.ones_like(code_rms))
    scale = key_rms/(math.sqrt(adapter.rank)*denominator)
    flops += 8*key_sum.numel()
    return scale[..., None, None], dict(fit_users=len(train_uids), fit_tokens=tokens,
        key_rms=key_rms.cpu().tolist(), code_rms=code_rms.cpu().tolist(),
        key_scale=scale.cpu().tolist(), statistics_flops=flops,
        zero_code_fallback="unit code RMS when code RMS <=1e-6", key_rms_floor=1e-6)


class JointKVRefinement(JointValueRefinement):
    def __init__(self, adapter, fallback_scales, key_scales):
        super().__init__(adapter, fallback_scales)
        origin = torch.stack([layer.key_decoder.detach().clone() for layer in adapter.layers])
        if torch.count_nonzero(origin):
            raise ValueError("this candidate starts from the unchanged Value fit and zero Key decoders")
        if key_scales.shape != (*origin.shape[:2], 1, 1) or not torch.isfinite(key_scales).all():
            raise ValueError("Key coordinates need finite fit-only per-layer/head scales")
        self.register_buffer("key_origin", origin)
        self.register_buffer("key_scale", key_scales.detach().clone())
        self.key_delta = nn.Parameter(torch.zeros_like(origin))

    def key_decoders(self):
        return self.key_origin + self.key_scale*self.key_delta

    def make_history_read(self, adapter, state):
        values, keys = self.decoder.decoders(), self.key_decoders()

        def read(index, attention, q, k, v):
            if state.seq_len != k.shape[1]:
                raise ValueError("factored KV codes must align with native rows")
            return adapter.layers[index].read(attention, q, k, v, state.codes[index],
                decoder=values[index], key_decoder=keys[index])

        return read

    @torch.no_grad()
    def fold_into(self, adapter):
        super().fold_into(adapter)
        for layer, key in zip(adapter.layers, self.key_decoders(), strict=True):
            layer.key_decoder.copy_(key)
