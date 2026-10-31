"""Distributed initialization, source binding, and resumable atomic checkpoints."""
from __future__ import annotations

import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .data_common import json_save, sha256


def distributed_init(seed=17):
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                          [("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)])
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=15), device_id=torch.device("cuda", local))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # FP32 quality uses ordinary FP32 rather than different TF32 policies.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return rank, world, torch.device("cuda", local)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def source_binding(root, vocabulary_name="vocabulary"):
    directory = Path(__file__).parent
    paths = [directory / name for name in ["model.py", "shared_reader.py", "objective.py", "batches.py",
                                          "runtime.py", "configuration.py", "protocol.py", "communication.py", "sampling.py", "train.py", "evaluate.py"]]
    from hstu_kvcache.models import attention, block, embeddings, hstu, kv_cache, triton_attention
    paths += [Path(module.__file__) for module in [attention, block, embeddings, hstu, kv_cache, triton_attention]]
    result = {str(path): sha256(path) for path in paths}
    for name in ["history/manifest.json", f"{vocabulary_name}/manifest.json", "stage_index_manifest.json", "population_manifest.json", "selection_manifest.json"]:
        result[str(Path(root) / name)] = sha256(Path(root) / name)
    return result


def save_checkpoint(path, model, optimizer, scheduler, metadata, *, started, prior_elapsed, rank, world):
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state()}
    rngs = [None]*world
    if world > 1:
        dist.all_gather_object(rngs, rng)
    else:
        rngs[0] = rng
    if rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        payload = dict(metadata, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(), rngs=rngs)
        torch.save(payload, temporary)
        os.replace(temporary, path)
        json_save(path.with_suffix(".json"), {"status": "committed", "checkpoint_sha256": sha256(path),
                  "elapsed_budget_seconds": prior_elapsed+time.monotonic()-started,
                  "metadata": metadata, "bytes": path.stat().st_size})
    if world > 1:
        dist.barrier()


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if sidecar["status"] != "committed":
        raise ValueError("Checkpoint is incomplete")
    return torch.load(path, map_location=device, weights_only=False), sidecar


def restore_rng(payload, rank, world):
    if len(payload["rngs"]) != world:
        raise ValueError("Exact resume requires the original world size")
    rng = payload["rngs"][rank]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    torch.cuda.set_rng_state(rng["cuda"].cpu())
