"""Two-GPU NCCL gradient check with unequal/empty ranks; no optimizer steps."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .objective import ranking_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a new output path")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise SystemExit("This bounded check requires exactly two ranks")
    torch.set_num_threads(2)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(17)
        module = torch.nn.Linear(2, 1, bias=False).to(device)
        model = DistributedDataParallel(module, device_ids=[local_rank])
        features = torch.tensor([
            [[1., .5], [2., -1.], [.1, .3]], [[3., 2.], [4., 1.], [-1., 1.]],
            [[1., -1.], [2., 3.], [4., .5]], [[0., 0.], [1., 1.], [2., 1.]],
        ], device=device)
        rows = []
        for name, labels in [
            ("unequal_active_requests", [[1, 0, -1], [0, 1, -1], [1, 0, -1], [-1, -1, -1]]),
            ("empty_rank", [[1, 0, -1], [0, 1, -1], [-1, -1, -1], [-1, -1, -1]]),
        ]:
            labels = torch.tensor(labels, device=device)
            masks = torch.ones_like(labels, dtype=torch.bool)
            reference = copy.deepcopy(module)
            reference.zero_grad(set_to_none=True)
            full_scores = reference(features).squeeze(-1)
            full_loss, _ = ranking_loss(full_scores, labels, masks)
            full_loss.backward()
            expected = reference.weight.grad.detach().clone()
            sl = slice(2 * rank, 2 * rank + 2)
            local_count = (labels[sl] >= 0).any(dim=1).sum()
            count = local_count.clone()
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            model.zero_grad(set_to_none=True)
            local_scores = model(features[sl]).squeeze(-1)
            local_loss, _ = ranking_loss(local_scores, labels[sl], masks[sl],
                                         global_active_requests=count, ddp_world_size=world)
            local_loss.backward()
            error = (module.weight.grad - expected).abs().max()
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
            rows.append({"case": name, "global_active_requests": int(count),
                         "gradient_max_abs": float(error), "limit": 1e-6,
                         "passed": bool(torch.isfinite(error) and error <= 1e-6)})
        if rank == 0:
            sources = [Path(__file__), Path(__file__).with_name("objective.py")]
            report = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                      "scope": "synthetic_gradient_equivalence_no_optimizer_or_model_training",
                      "backend": "nccl", "world_size": world, "rows": rows,
                      "passed": all(row["passed"] for row in rows),
                      "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        if not all(row["passed"] for row in rows):
            raise RuntimeError("Distributed gradient mismatch")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
