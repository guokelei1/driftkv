from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HSTUKVCache:
    """Per-user derived KV cache: ``F(theta, x_u)``.

    Holds, for every layer, the K and V tensors produced by the pointwise
    attention for that user's behaviour sequence. When the model parameters
    theta change, this representation drifts - estimating that drift cheaply
    without recomputing F is the core problem of this project.

    Shape convention: ``[num_layers, num_heads, L_u, head_dim]`` (no batch dim;
    one cache per user). ``seq_len`` records the sequence length used.
    """

    k: torch.Tensor  # [num_layers, num_heads, L, head_dim]
    v: torch.Tensor  # [num_layers, num_heads, L, head_dim]
    seq_len: int
    user_id: int | None = None

    @classmethod
    def from_layer_list(cls, kvs: list[tuple[torch.Tensor, torch.Tensor]], seq_len: int, user_id: int | None = None) -> HSTUKVCache:
        ks = torch.stack([kv[0] for kv in kvs], dim=0)  # [L_layers, B?, num_heads, L, head_dim]
        vs = torch.stack([kv[1] for kv in kvs], dim=0)
        return cls(k=ks, v=vs, seq_len=seq_len, user_id=user_id)

    def to(self, device) -> HSTUKVCache:
        return HSTUKVCache(k=self.k.to(device), v=self.v.to(device), seq_len=self.seq_len, user_id=self.user_id)

    def detach(self) -> HSTUKVCache:
        return HSTUKVCache(k=self.k.detach(), v=self.v.detach(), seq_len=self.seq_len, user_id=self.user_id)

    def drift_norm(self, other: HSTUKVCache) -> dict[str, float]:
        """Ground-truth drift ||F(theta+dtheta) - F(theta)|| metrics vs another cache."""
        assert self.k.shape == other.k.shape, f"shape mismatch {self.k.shape} vs {other.k.shape}"
        dk = (self.k - other.k).float()
        dv = (self.v - other.v).float()
        return {
            "k_l2": dk.norm().item(),
            "v_l2": dv.norm().item(),
            "k_rms": (dk.pow(2).mean().sqrt()).item(),
            "v_rms": (dv.pow(2).mean().sqrt()).item(),
            "k_fro_rel": (dk.norm() / (self.k.float().norm() + 1e-12)).item(),
            "v_fro_rel": (dv.norm() / (self.v.float().norm() + 1e-12)).item(),
            "numel": self.k.numel() + self.v.numel(),
        }
