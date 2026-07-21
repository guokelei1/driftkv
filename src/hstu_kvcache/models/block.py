from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import PointwiseAttention, PointwiseAttentionConfig
from .rmsnorm import RMSNorm


@dataclass
class HSTUBlockConfig:
    attn: PointwiseAttentionConfig
    # FFN-replacement gating strategy. HSTU merges the FFN into the attention
    # block via a pointwise gate applied to the attention output.
    #   "none"     : plain residual (out = attn_out)
    #   "silu_gate": out = attn_out * silu(W_g(x_norm))   <-- HSTU default
    #   "glu"      : out = attn_out * sigmoid(W_g(x_norm))
    #   "ffn"      : standard SwiGLU MLP on top of attention (non-HSTU baseline)
    gating: Literal["none", "silu_gate", "glu", "ffn"] = "silu_gate"
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6


class HSTUBlock(nn.Module):
    """A single HSTU layer: RMSNorm -> PointwiseAttention -> (gating) -> residual.

    Designed as a standalone class so future design tweaks (roadmap U1/U2 -
    layer-wise drift sensitivity, pointwise-attention structure) are localised
    edits that do not touch the surrounding model.
    """

    def __init__(self, cfg: HSTUBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.attn.hidden_size
        self.norm = RMSNorm(h, eps=cfg.norm_eps)
        self.attn = PointwiseAttention(cfg.attn)
        self.gating = cfg.gating
        if cfg.gating in ("silu_gate", "glu"):
            self.gate_proj = nn.Linear(h, h, bias=False)
        elif cfg.gating == "ffn":
            inner = int(h * cfg.mlp_ratio)
            self.fc1 = nn.Linear(h, inner, bias=False)
            self.fc2 = nn.Linear(inner, h, bias=False)
            self.fc3 = nn.Linear(h, inner, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_kv: bool = False,
    ):
        residual = x
        x_norm = self.norm(x)
        if return_kv:
            attn_out, (k, v) = self.attn(x_norm, attn_mask=attn_mask, return_kv=True)
        else:
            attn_out = self.attn(x_norm, attn_mask=attn_mask, return_kv=False)
            k = v = None

        if self.gating == "silu_gate":
            out = attn_out * F.silu(self.gate_proj(x_norm))
        elif self.gating == "glu":
            out = attn_out * torch.sigmoid(self.gate_proj(x_norm))
        elif self.gating == "ffn":
            out = self.fc2(F.silu(self.fc1(x_norm)) * self.fc3(x_norm))
        else:
            out = attn_out

        x = residual + out
        if return_kv:
            return x, (k, v)
        return x

    def forward_with_cache(
        self,
        x_new: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
    ):
        """Incremental forward: prefix KV from old model + new positions with current model.

        Args:
            x_new: [B, m, H] hidden states for new positions.
            cached_k/v: [B, n, inner] prefix K,V (may be from theta_old).

        Returns: x_new_out [B, m, H], (k_all, v_all) [B, n+m, inner].
        """
        residual = x_new
        x_norm = self.norm(x_new)
        attn_out, (k_all, v_all) = self.attn.forward_with_cache(x_norm, cached_k, cached_v)

        if self.gating == "silu_gate":
            out = attn_out * F.silu(self.gate_proj(x_norm))
        elif self.gating == "glu":
            out = attn_out * torch.sigmoid(self.gate_proj(x_norm))
        elif self.gating == "ffn":
            out = self.fc2(F.silu(self.fc1(x_norm)) * self.fc3(x_norm))
        else:
            out = attn_out

        x_new_out = residual + out
        return x_new_out, (k_all, v_all)

    def forward_stale_kv(
        self,
        x: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
    ) -> torch.Tensor:
        """Full-sequence forward with entirely stale K,V. Only Q/gating from current model."""
        residual = x
        x_norm = self.norm(x)
        attn_out = self.attn.forward_stale_kv(x_norm, cached_k, cached_v)
        if self.gating == "silu_gate":
            out = attn_out * F.silu(self.gate_proj(x_norm))
        elif self.gating == "glu":
            out = attn_out * torch.sigmoid(self.gate_proj(x_norm))
        elif self.gating == "ffn":
            out = self.fc2(F.silu(self.fc1(x_norm)) * self.fc3(x_norm))
        else:
            out = attn_out
        return residual + out
