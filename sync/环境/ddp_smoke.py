#!/usr/bin/env python3
"""Two-GPU HSTU update equivalence on synthetic inputs; no foundation training."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import platform
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from hstu_kvcache.models import HSTU, HSTUConfig


class Scorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = HSTU(HSTUConfig(
            num_items=128, num_behaviors=4, hidden_size=64, num_layers=2,
            num_heads=2, max_seq_len=64, input_dropout=0.0, attn_dropout=0.0,
        ))

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.model.score_cc_full(*inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", timeout=dt.timedelta(seconds=90), device_id=device)
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 2:
            raise ValueError("Run this smoke with exactly two ranks")
        if args.output.exists():
            raise FileExistsError(f"Keep the previous result; choose a new output: {args.output}")
        communication = torch.tensor([float(rank + 1)], device=device)
        dist.all_reduce(communication)
        torch.testing.assert_close(communication, torch.tensor([3.0], device=device))

        torch.manual_seed(17)
        model = Scorer().to(device)
        reference = copy.deepcopy(model)
        distributed = DistributedDataParallel(model, device_ids=[local_rank])
        optimizer = torch.optim.SGD(distributed.parameters(), lr=0.01, momentum=0.9)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)

        # All ranks see the same fixed global batch; each trains on its own slice.
        torch.manual_seed(23)
        global_inputs = (
            torch.randint(1, 128, (8, 32)).to(device),
            torch.randint(0, 4, (8, 32)).to(device),
            torch.rand(8, 32).to(device),
            torch.randint(1, 128, (8, 2)).to(device),
            torch.ones(8, device=device),
        )
        targets = torch.randint(0, 2, (8, 2)).float().to(device)
        section = slice(rank * 4, (rank + 1) * 4)
        local_inputs = tuple(value[section] for value in global_inputs)
        steps = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                distributed(*local_inputs), targets[section])
            loss.backward()
            reference_loss = nn.functional.binary_cross_entropy_with_logits(
                reference(*global_inputs), targets)
            reference_loss.backward()
            mean_loss = loss.detach().clone()
            dist.all_reduce(mean_loss)
            mean_loss /= world
            torch.testing.assert_close(mean_loss, reference_loss.detach(), atol=2e-6, rtol=2e-5)
            gradient_error = 0.0
            for parameter, expected in zip(model.parameters(), reference.parameters()):
                if parameter.grad is None or expected.grad is None:
                    if parameter.grad is not expected.grad:
                        raise RuntimeError("DDP/reference gradient presence differs")
                    continue
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Nonfinite gradient")
                torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-5, rtol=2e-4)
                gradient_error = max(gradient_error, float((parameter.grad - expected.grad).abs().max()))
            optimizer.step()
            reference_optimizer.step()
            parameter_error = 0.0
            replica_error = 0.0
            for parameter, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, expected, atol=2e-6, rtol=2e-5)
                parameter_error = max(parameter_error, float((parameter - expected).detach().abs().max()))
                rank_zero = parameter.detach().clone()
                dist.broadcast(rank_zero, src=0)
                torch.testing.assert_close(parameter.detach(), rank_zero, atol=0, rtol=0)
                replica_error = max(replica_error, float((parameter.detach() - rank_zero).abs().max()))
            errors = torch.tensor([gradient_error, parameter_error, replica_error], device=device)
            dist.all_reduce(errors, op=dist.ReduceOp.MAX)
            steps.append({
                "step": step + 1, "global_mean_loss": float(mean_loss),
                "reference_loss": float(reference_loss.detach()),
                "max_gradient_error": float(errors[0]),
                "max_parameter_error": float(errors[1]),
                "max_replica_parameter_error": float(errors[2]),
            })
        if rank == 0:
            result = {
                "status": "passed_two_gpu_ddp_smoke", "host": platform.node(),
                "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                "backend": dist.get_backend(), "world_size": world,
                "devices": [torch.cuda.get_device_name(i) for i in range(world)],
                "model": "Randomly initialized legacy HSTU 2L/H64, 2 heads, FP32, dropout=0, TF32 disabled",
                "model_seed": 17, "input_seed": 23,
                "global_batch": 8, "per_rank_batch": 4, "history_rows": 32,
                "candidates": 2, "optimizer": "SGD lr=0.01 momentum=0.9",
                "nccl_all_reduce_passed": True, "steps": steps,
                "synthetic_inputs": True, "foundation_weights_used": False,
                "quality_claim": False, "throughput_claim": False,
                "command": "CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 sync/环境/ddp_smoke.py --output results/environment_201_2026_10/ddp/summary.json",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
