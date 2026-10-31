"""Real complete-candidate inputs: cache oracle, gradients, and row communication."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.distributed as dist

from .batches import RequestBatches
from .communication import synchronize_and_clip
from .data_common import json_save
from .model import RequestRanker
from .objective import ranking_loss
from .runtime import distributed_init, source_binding, to_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    args = p.parse_args()
    rank, world, device = distributed_init(17)
    dataset = RequestBatches(args.data, [24], role=0, pilot=True, limit=64)
    indices = list(range(rank*2, rank*2+2))
    batch = to_device(dataset[indices], device)
    model = RequestRanker(args.data / "vocabulary").to(device).eval()
    with torch.no_grad():
        cache = model.encode_history(batch)
        reference = model.encode_history(batch, reference=True)
        torch.testing.assert_close(cache.k, reference.k, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(cache.v, reference.v, rtol=2e-6, atol=2e-6)
        query = model.read(batch, cache)
        reference_query = model.read(batch, reference)
        torch.testing.assert_close(query, reference_query, rtol=2e-5, atol=2e-6)
        cache_error = max(float((cache.k-reference.k).abs().max()), float((cache.v-reference.v).abs().max()))
    dense = copy.deepcopy(model)
    optimizers = [torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4, fused=True)
                  for net in [model, dense]]
    losses = []
    for net, optimizer, mode in zip([model, dense], optimizers, [False, True], strict=True):
        net.train()
        logits = net(batch)
        loss, _ = ranking_loss(logits, batch["labels"], batch["candidate_mask"],
                               global_active_requests=world*2, ddp_world_size=world)
        loss.backward()
        rows = [net.features.map_ids(batch["history_features"][..., 0]).reshape(-1),
                net.features.map_ids(batch["candidate_features"][..., 0]).reshape(-1)]
        norm, count = synchronize_and_clip(net, rows, world, dense=mode)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss))
    parameter_error, moment_error = 0., 0.
    for (name, actual), (ref_name, expected) in zip(model.named_parameters(), dense.named_parameters(), strict=True):
        assert name == ref_name
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6, msg=name)
        parameter_error = max(parameter_error, float((actual-expected).abs().max()))
    for actual, expected in zip(optimizers[0].state.values(), optimizers[1].state.values(), strict=True):
        for name in ["exp_avg", "exp_avg_sq", "step"]:
            torch.testing.assert_close(actual[name], expected[name], rtol=2e-5, atol=2e-6, msg=name)
            moment_error = max(moment_error, float((actual[name]-expected[name]).abs().max()))
    report = {"status": "passed", "world_size": world, "cache_max_abs": cache_error,
              "parameter_max_abs": parameter_error, "adam_moment_max_abs": moment_error,
              "losses": losses, "history_lengths": batch["lengths"].tolist(),
              "candidate_counts": batch["counts"].tolist(), "source_binding": source_binding(args.data),
              "scope": "real-data numerical correctness only; one diagnostic optimizer step, no recommendation-quality claim"}
    gathered = [None]*world
    if world > 1:
        dist.all_gather_object(gathered, report)
    else:
        gathered = [report]
    if rank == 0:
        json_save(args.output, {"status": "passed", "ranks": gathered})
        print({"status": "passed", "parameter_max_abs": parameter_error, "adam_moment_max_abs": moment_error}, flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
