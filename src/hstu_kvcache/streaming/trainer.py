from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
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
    """Return the flattened parameter difference between two model versions."""
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
    lengths = batch.get("lengths")
    kv = model.compute_kv(
        item_ids,
        behaviors,
        time_deltas,
        lengths=None if lengths is None else lengths.to(device),
    )
    return kv.detach().to(torch.device("cpu"))


def build_next_item_targets(
    item_ids: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor | None = None,
    train_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = item_ids[:, 1:]
    positions = torch.arange(targets.shape[1], device=item_ids.device)
    valid = positions.unsqueeze(0) < (lengths - 1).clamp_min(0).unsqueeze(1)
    valid = valid & (item_ids[:, :-1] > 0) & (targets > 0)
    if labels is not None:
        valid = valid & (labels[:, 1:] > 0)
    if train_mask is not None:
        valid = valid & train_mask[:, 1:].bool()
    return targets, valid


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_scale: float = 1.0,
    zero_grad: bool = True,
    optimizer_step: bool = True,
) -> float:
    """One streaming SGD/Adam step on a next-item prediction objective.

    Objective: predict the next item's behaviour/label from the hidden state at
    the previous position. We use a simple cross-entropy over a sampled
    negative set sampled from the fitted item catalog (cheap, sufficient for
    producing realistic dtheta sequences - the exact ranking loss is not the
    research focus).
    """
    model.train()
    core_model = model.module if hasattr(model, "module") else model
    if not isinstance(core_model, HSTU):
        raise TypeError("train_step requires HSTU or a distributed HSTU wrapper")
    item_ids = batch["item_ids"].to(device)
    behaviors = batch["behaviors"].to(device)
    time_deltas = batch["time_deltas"].to(device)

    lengths = batch.get("lengths")
    if lengths is None:
        lengths = (item_ids > 0).sum(dim=1)
    else:
        lengths = lengths.to(device)
    labels = batch.get("labels")
    train_mask = batch.get("train_mask")
    labels = None if labels is None else labels.to(device)
    train_mask = None if train_mask is None else train_mask.to(device)

    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    hidden, _ = model(
        item_ids,
        behaviors,
        time_deltas,
        return_kv=False,
        lengths=lengths,
    )
    target_items, valid = build_next_item_targets(item_ids, lengths, labels, train_mask)
    if not torch.any(valid):
        if (dist.is_available() and dist.is_initialized()) or not zero_grad:
            loss = hidden.sum() * 0.0
            loss.backward()
            if optimizer_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        return 0.0
    source_hidden = hidden[:, :-1]
    neg = torch.randint(
        1,
        core_model.cfg.num_prediction_items + 1,
        (*target_items.shape, 8),
        device=device,
    )
    pos = target_items.unsqueeze(-1)
    neg = torch.where(
        neg == pos,
        neg.remainder(core_model.cfg.num_prediction_items) + 1,
        neg,
    )
    cands = torch.cat([pos, neg], dim=-1)
    logits = core_model.item_emb.score(source_hidden, cands)
    target = torch.zeros_like(target_items)
    per_target = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target.flatten(),
        reduction="none",
    ).view_as(target_items)
    loss = (per_target * valid).sum() / valid.sum()
    (loss * loss_scale).backward()
    if optimizer_step:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return float(loss.item())


class StreamingTrainer:
    """Produces a sequence of checkpoints theta_0, theta_1, ..., theta_t.

    Each ``stream_chunk`` call does N gradient steps on a chunk of the streaming
    data and records a new checkpoint for versioned cache evaluation.
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
