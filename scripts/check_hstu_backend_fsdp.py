#!/usr/bin/env python3
"""Compare six-layer PyTorch/Triton execution through two-rank BF16 FSDP.

CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc-per-node=2 \
    scripts/check_hstu_backend_fsdp.py --output results/backend_acceleration/fsdp.json

This synthetic one-update check exercises distributed execution; it does not
measure training quality or establish identical optimization trajectories.
"""

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.models.backend_info import attention_backend_info


class Embedded(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.model = HSTU(HSTUConfig(
            num_items=32, num_behaviors=4, hidden_size=192, num_layers=6,
            num_heads=6, max_seq_len=1027, input_dropout=0.0,
            block_variant="hstu_reference", activation="silu",
            relative_position_bias=True,
        ))
        for block in self.model.blocks:
            block.attn.backend = backend

    def forward(self, x):
        return self.model.forward_embedded(x)[0]


@torch.no_grad()
def reduced_pair(left, right, device):
    sums = torch.zeros(3, device=device, dtype=torch.float64)
    maximum = torch.zeros((), device=device)
    for a, b in zip(left, right, strict=True):
        assert a.shape == b.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        difference = b.float() - a.float()
        sums += torch.stack([a.double().square().sum(), difference.double().square().sum(),
                             torch.tensor(a.numel(), device=device, dtype=torch.float64)])
        if a.numel():
            maximum = torch.maximum(maximum, difference.abs().max())
    dist.all_reduce(sums)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return dict(relative_l2=float((sums[1] / sums[0].clamp_min(1e-30)).sqrt()),
                max_abs=float(maximum), elements=int(sums[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    assert dist.get_world_size() == 2, "This canary fixes two FSDP ranks."
    torch.set_num_threads(2)
    execution = attention_backend_info()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    models, optimizers = [], []
    for backend in ("torch", "triton"):
        torch.manual_seed(17)
        model = FSDP(Embedded(backend).to(device), device_id=device, use_orig_params=True,
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                    reduce_dtype=torch.float32,
                                                    buffer_dtype=torch.float32),
                     sync_module_states=True)
        models.append(model)
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=1e-3))

    torch.manual_seed(1017 + rank)
    x = torch.randn(2, 128, 192, device=device)
    target = torch.randn_like(x)
    outputs, losses = [], []
    for model in models:
        output = model(x)
        loss = (output.float() - target).square().mean()
        loss.backward()
        outputs.append(output.detach())
        losses.append(loss.detach())

    output_parity = reduced_pair(outputs[:1], outputs[1:], device)
    loss_parity = reduced_pair(losses[:1], losses[1:], device)
    gradients, biases = [[], []], [[], []]
    for (name_a, a), (name_b, b) in zip(models[0].named_parameters(), models[1].named_parameters(), strict=True):
        assert name_a == name_b
        assert (a.grad is None) == (b.grad is None), name_a
        if a.grad is not None:
            gradients[0].append(a.grad)
            gradients[1].append(b.grad)
            if "position_bias" in name_a:
                biases[0].append(a.grad)
                biases[1].append(b.grad)
    gradient_parity = reduced_pair(*gradients, device)
    bias_parity = reduced_pair(*biases, device)
    for optimizer in optimizers:
        optimizer.step()
    parameter_parity = reduced_pair(list(models[0].parameters()), list(models[1].parameters()), device)
    assert output_parity["relative_l2"] < 0.01, output_parity
    assert loss_parity["relative_l2"] < 0.002, loss_parity
    assert gradient_parity["relative_l2"] < 0.02, gradient_parity
    assert parameter_parity["relative_l2"] < 0.001, parameter_parity
    assert attention_backend_info() == execution, "Attention execution sources changed during canary."
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == script_hash
    if rank == 0:
        report = dict(
            passed=True,
            scope="Synthetic six-layer two-rank FSDP BF16 forward/backward and one AdamW update; no data or quality claim",
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            device_name=torch.cuda.get_device_name(device), world_size=dist.get_world_size(),
            local_batch_size=2, length=128, hidden_size=192, heads=6, layers=6,
            precision="FSDP BF16 parameters including relative bias; FP32 reductions",
            output=output_parity, loss=loss_parity, gradients=gradient_parity,
            position_bias_gradients=bias_parity, parameters_after_update=parameter_parity,
            thresholds=dict(output_relative_l2=0.01, loss_relative_l2=0.002,
                            gradient_relative_l2=0.02, parameter_relative_l2=0.001),
            compared_module_backends=["torch", "triton"], initialization_seed=17,
            synthetic_input_seed="1017 + local_rank",
            optimizer="AdamW lr=0.001, default remaining options",
            attention_execution=execution, script_sha256=script_hash,
            worker_command=shlex.join([sys.executable, *sys.argv]),
            launcher_command=shlex.join(Path(f"/proc/{os.getppid()}/cmdline").read_bytes().decode().strip("\0").split("\0")),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
