"""Actual Max catalog/depth: zero/short/full prefix, independent reference, gradients."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .batches import RequestBatches
from .data_common import json_save, sha256
from .model import RequestRanker
from .objective import ranking_loss
from .runtime import distributed_init, source_binding, to_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    rank, world, device = distributed_init(17)
    dataset = RequestBatches(args.data, list(range(1,24)))
    desired = [[0, 1024], [1, 1023], [127, 511], [0, 1024]][rank]
    indices = [int(np.flatnonzero(dataset.records["history_length"] == length)[0]) for length in desired]
    batch = to_device(dataset[indices], device)
    model = RequestRanker(args.data / "vocabulary_max", "max").to(device).eval()
    with torch.no_grad():
        fast, ref = model.encode_history(batch), model.encode_history(batch, reference=True)
        torch.testing.assert_close(fast.k, ref.k, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(fast.v, ref.v, rtol=2e-6, atol=2e-6)
        scores, oracle = model.read(batch, fast), model.read(batch, ref)
        torch.testing.assert_close(scores, oracle, rtol=2e-5, atol=2e-6)
        cache_error = max(float((fast.k-ref.k).abs().max()), float((fast.v-ref.v).abs().max()))
        score_error = float((scores-oracle).abs().max())
        # Independently trim each true prefix (one fully masked slot for empty).
        trimmed_error = 0.
        for row, length in enumerate(desired):
            single = {k: v[row:row+1] for k,v in batch.items()}
            for key in ["history_features", "behaviors", "deltas"]:
                single[key] = single[key][:,:max(1,length)]
            trimmed = model.read(single, model.encode_history(single, reference=True))
            torch.testing.assert_close(scores[row:row+1], trimmed, rtol=2e-5, atol=2e-6)
            trimmed_error = max(trimmed_error, float((scores[row:row+1]-trimmed).abs().max()))
    del fast, ref, oracle, scores, trimmed
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        scores = model(batch)
        loss, _ = ranking_loss(scores, batch["labels"], batch["candidate_mask"], listwise_weight=0)
    loss.backward()
    gradients = {name: p.grad for name,p in model.named_parameters() if p.requires_grad}
    if any(value is None or not torch.isfinite(value).all() for value in gradients.values()):
        raise ValueError("Actual Max empty/short-history backward is disconnected or nonfinite")
    result = dict(rank=rank, history_lengths=desired, candidate_counts=batch["counts"].tolist(),
                  parameters=sum(p.numel() for p in model.parameters()), cache_max_abs=cache_error,
                  score_max_abs=score_error, trimmed_score_max_abs=trimmed_error,
                  finite_gradients=len(gradients), bce=float(loss), peak_bytes=torch.cuda.max_memory_allocated())
    reports = [None]*world
    if world>1:
        dist.all_gather_object(reports,result)
    else:
        reports[0]=result
    if rank==0:
        json_save(args.output, dict(status="passed", quality_use_permitted=False, ranks=reports,
            source_binding=source_binding(args.data,"vocabulary_max"), checker_sha256=sha256(__file__),
            wall_seconds=time.monotonic()-started))
        print("Actual Max mixed-history reference and backward passed", flush=True)
    if world>1:
        dist.destroy_process_group()


if __name__=="__main__":
    main()
