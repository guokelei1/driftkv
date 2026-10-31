"""Legacy row-state type used by the shared lifecycle append hook."""
from dataclasses import dataclass
import torch


@dataclass
class CompactItemState:
    hidden: torch.Tensor  # [layers, owners, historical positions, hidden width]
    seq_len: int

    def select(self, indices):
        return CompactItemState(self.hidden[:, indices], self.seq_len)

    def storage_bytes(self):
        return self.hidden.numel() * self.hidden.element_size()
