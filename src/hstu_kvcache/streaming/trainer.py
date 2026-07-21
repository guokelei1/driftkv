from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import HSTU


def _device_of(m: nn.Module) -> torch.device:
    return next(m.parameters()).device


def model_params_vec(model: nn.Module) -> torch.Tensor:
    """Flatten all trainable parameters into a single 1-D tensor (detach)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_model_from_vec(model: nn.Module, vec: torch.Tensor) -> None:
    """Load a flattened parameter vector back into the model (in-place)."""
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[idx : idx + n].view_as(p))
            idx += n


@dataclass
class Checkpoint:
    step: int
    state_vec: torch.Tensor  # flattened theta
    model_state: dict[str, torch.Tensor]  # state_dict (cpu)


def checkpoint_diff(theta0: torch.Tensor, theta1: torch.Tensor) -> torch.Tensor:
    """dtheta = theta1 - theta0 (flattened). The input variable for drift."""
    return (theta1 - theta0).detach()


@torch.no_grad()
def oracle_recompute_kv(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> object:
    """Ground-truth KV: recompute F(theta, x_u) under current model params."""
    model.eval()
    item_ids = batch["item_ids"].to(device)
    behaviors = batch["behaviors"].to(device)
    time_deltas = batch["time_deltas"].to(device)
    kv = model.compute_kv(item_ids, behaviors, time_deltas)
    return kv.detach().to(torch.device("cpu"))


def train_step(
    model: HSTU,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One streaming SGD/Adam step on a next-item prediction objective.

    Objective: predict the next item's behaviour/label from the hidden state at
    the previous position. We use a simple cross-entropy over a sampled
    negative set derived from the in-batch items (cheap, sufficient for
    producing realistic dtheta sequences - the exact ranking loss is not the
    research focus).
    """
    model.train()
    item_ids = batch["item_ids"].to(device)
    behaviors = batch["behaviors"].to(device)
    time_deltas = batch["time_deltas"].to(device)

    hidden, _ = model(item_ids, behaviors, time_deltas, return_kv=False)
    # score against the *next* item (shift): use item_emb table directly.
    # logits[i, t] = hidden[i, t] . item_emb[item_ids[i, t+1]]
    B, L, H = hidden.shape
    target_items = item_ids  # predict item at same position from hidden (autoregressive)
    # in-batch negative sampling
    all_items = item_ids.clamp(min=1).reshape(-1)
    neg = all_items[torch.randint(0, all_items.numel(), (B, L, 8), device=device)]
    pos = target_items.unsqueeze(-1)  # [B, L, 1]
    cands = torch.cat([pos, neg], dim=-1)  # [B, L, 9]
    logits = model.item_emb.score(hidden, cands)  # [B, L, 9]
    # label: 1 for pos. mask padding & first position.
    target = torch.zeros(B, L, 9, device=device)
    target[..., 0] = 1.0
    mask = (item_ids > 0).float().unsqueeze(-1)  # [B, L, 1]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    ) * mask
    loss = loss.sum() / (mask.sum() + 1e-6)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


class StreamingTrainer:
    """Produces a sequence of checkpoints theta_0, theta_1, ..., theta_t.

    Each ``stream_chunk`` call does N gradient steps on a chunk of the streaming
    data and records a new checkpoint. The resulting dtheta sequence is the
    raw material for drift experiments (Phase 0 V3/V4 and Phase 3).
    """

    def __init__(self, model: HSTU, lr: float = 3e-4, device: str | torch.device = "cuda") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.checkpoints: list[Checkpoint] = []
        self._record_checkpoint(step=0)

    def _record_checkpoint(self, step: int) -> None:
        vec = model_params_vec(self.model).detach().cpu()
        state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.checkpoints.append(Checkpoint(step=step, state_vec=vec, model_state=state))

    def stream_chunk(self, dataloader: DataLoader, steps: int, step_log: int = 50) -> list[float]:
        losses: list[float] = []
        it = iter(dataloader)
        for s in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)
            losses.append(train_step(self.model, batch, self.optimizer, self.device))
            if (s + 1) % step_log == 0:
                pass
        self._record_checkpoint(step=len(self.checkpoints))
        return losses

    def dtheta_sequence(self) -> list[torch.Tensor]:
        return [
            checkpoint_diff(self.checkpoints[i].state_vec, self.checkpoints[i + 1].state_vec)
            for i in range(len(self.checkpoints) - 1)
        ]

    def load_checkpoint(self, idx: int) -> None:
        self.model.load_state_dict(self.checkpoints[idx].model_state)
        self.model.to(self.device)
