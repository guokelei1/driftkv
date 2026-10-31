"""Exact replicated AdamW with compact communication of touched item rows.

The dense item gradient and full AdamW update are retained: momentum/decay of
previously touched rows are never silently skipped. Only communication and
gradient-norm reads omit provably zero CURRENT gradients outside the union.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def union_rows(local_rows, world):
    rows = torch.unique(torch.cat(local_rows), sorted=True)
    if world == 1:
        return rows
    size = torch.tensor([rows.numel()], dtype=torch.int64, device=rows.device)
    sizes = [torch.empty_like(size) for _ in range(world)]
    dist.all_gather(sizes, size)
    maximum = max(int(value.item()) for value in sizes)
    padded = torch.full((maximum,), -1, dtype=torch.int64, device=rows.device)
    padded[:len(rows)] = rows
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)
    complete = torch.unique(torch.cat(gathered), sorted=True)
    return complete[complete >= 0]


def synchronize_and_clip(model, local_rows, world, *, dense=False):
    parameter = model.features.item.weight
    others = [p.grad for p in model.parameters() if p.requires_grad and p is not parameter]
    if any(grad is None for grad in others) or parameter.grad is None:
        raise ValueError("Missing gradients in replicated data parallel training")
    rows = None if dense else union_rows(local_rows, world)
    item_gradient = parameter.grad if dense else parameter.grad.index_select(0, rows)
    if world > 1:
        work = dist.all_reduce(item_gradient, async_op=True)
        flat = torch.cat([grad.reshape(-1) for grad in others])
        dist.all_reduce(flat)
        flat.div_(world)
        start = 0
        for grad in others:
            grad.copy_(flat[start:start+grad.numel()].view_as(grad))
            start += grad.numel()
        work.wait()
        item_gradient.div_(world)
    norm = torch.linalg.vector_norm(torch.stack([item_gradient.norm(), *torch._foreach_norm(others)]))
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite global gradient norm")
    factor = (1/(norm+1e-6)).clamp_max(1)
    item_gradient.mul_(factor)
    torch._foreach_mul_(others, factor)
    if rows is not None:
        parameter.grad.index_copy_(0, rows, item_gradient)
    return norm, parameter.shape[0] if dense else len(rows)
