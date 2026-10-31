from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HSTUKVCache:
    """Batched derived prefix K/V cache: ``F(theta, x)``.

    Holds, for every layer, the K and V tensors produced by the pointwise
    attention for a batch of behavior sequences. A cache captured under an old
    model version can be reused, migrated, or fully recomputed under the
    current version.

    Shape convention: ``[num_layers, batch, sequence, kv_width]``. ``seq_len``
    records the padded sequence width used by the batch.
    """

    k: torch.Tensor
    v: torch.Tensor
    seq_len: int

    @classmethod
    def from_layer_list(cls, kvs: list[tuple[torch.Tensor, torch.Tensor]], seq_len: int) -> HSTUKVCache:
        ks = torch.stack([kv[0] for kv in kvs], dim=0)
        vs = torch.stack([kv[1] for kv in kvs], dim=0)
        return cls(k=ks, v=vs, seq_len=seq_len)

    def to(self, device) -> HSTUKVCache:
        return HSTUKVCache(k=self.k.to(device), v=self.v.to(device), seq_len=self.seq_len)

    def detach(self) -> HSTUKVCache:
        return HSTUKVCache(k=self.k.detach(), v=self.v.detach(), seq_len=self.seq_len)

    def difference_metrics(self, other: HSTUKVCache) -> dict[str, float]:
        """Absolute and relative tensor differences from another cache."""
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
