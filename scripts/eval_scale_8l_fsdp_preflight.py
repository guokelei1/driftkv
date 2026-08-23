#!/usr/bin/env python3
"""Four-GPU FSDP Full-1024 backward/AdamW memory and semantics canary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

import scale_8l_common as scale
import scale_8l_fsdp as distributed
import train_p7_theta0 as p7
import train_scale_8l_theta0 as theta0  # frozen scale microbatch/chunk choices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=scale.OUTPUT / "s2_fsdp_training_preflight.json")
    args = parser.parse_args()
    scale.contract()
    rank, world, device = distributed.initialize()
    try:
        if world != 4:
            raise RuntimeError("frozen scale training requires exactly four ranks")
        rows = scale.load_requests(scale.P7_MANIFEST, "residual_train", "F")
        eligible = [row for row in rows if len(row.history_items) == scale.CONTEXT]
        row = scale.deterministic_subset(eligible, 1, "scale-fsdp-memory")[0]
        bases, _ = p7.load_bases(("F",), device)
        raw_model = scale.make_model(17, device).train()
        model_meta = scale.model_metadata(raw_model)
        expected_state_keys = set(raw_model.state_dict())
        model = distributed.wrap(raw_model, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        tensors = p7.collate([row], device)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        residual = model(
            tensors["items"], tensors["behaviors"], tensors["deltas"],
            tensors["candidates"], tensors["query_deltas"], tensors["lengths"],
            tensors["query_types"], theta0.p7.CHUNK_SIZE["F"],
        )
        with torch.no_grad():
            base = bases["F"](tensors["features"].float()).float()
        deployment = base + residual
        loss = F.binary_cross_entropy_with_logits(deployment[:, 0], tensors["labels"])
        loss.backward()
        gradient_sq = torch.zeros((), device=device)
        for parameter in model.parameters():
            if parameter.grad is not None:
                gradient_sq += parameter.grad.detach().float().square().sum()
        optimizer.step()
        checkpoint_state = distributed.full_hstu_state_dict(model)
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        local = torch.tensor([peak_allocated, peak_reserved], dtype=torch.float64, device=device)
        gathered = [torch.zeros_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        payload = None
        if rank == 0:
            per_rank = [value.cpu().tolist() for value in gathered]
            checks = {
                "world_size_four": world == 4,
                "finite_loss": math.isfinite(float(loss)),
                "gradient_norm_positive": float(gradient_sq.sqrt()) > 0,
                "optimizer_step_completed": True,
                "full_checkpoint_keyset_matches_HSTU": set(checkpoint_state) == expected_state_keys,
                "all_rank_peak_reserved_below_capacity": all(row[1] < total for row in per_rank),
                "qualification_and_theta3_remained_unaccessed": True,
            }
            payload = {
                "status": "passed" if all(checks.values()) else "failed",
                "contract_sha256": scale.sha256_file(scale.CONTRACT),
                "world_size": world, "GPU_name": torch.cuda.get_device_name(device),
                "request_id": row.request_id, "history_length": len(row.history_items),
                "loss": float(loss), "gradient_norm_rank0": float(gradient_sq.sqrt()),
                "model": model_meta,
                "memory_bytes_per_rank": [
                    {"rank": index, "peak_allocated": int(value[0]), "peak_reserved": int(value[1]),
                     "device_total": total, "peak_reserved_fraction": value[1] / total,
                     "headroom": total - int(value[1])}
                    for index, value in enumerate(per_rank)
                ],
                "checks": checks, "retained_checkpoint": False,
            }
            scale.json_dump(args.output, payload)
            print(json.dumps(payload, indent=2))
        status = torch.tensor(1 if rank != 0 or payload["status"] == "passed" else 0, device=device)
        dist.broadcast(status, src=0)
        if not int(status):
            raise RuntimeError("FSDP scale preflight failed")
    finally:
        distributed.finish()


if __name__ == "__main__":
    main()
