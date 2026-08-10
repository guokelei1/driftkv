from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from hstu_kvcache.streaming.kuairand_projected_persistent import (
    _build_workloads,
    _training_document,
    load_persistent_config,
)
from hstu_kvcache.streaming.kuairand_projected_scale import (
    _capture_old,
    _distributed,
    _evaluation_batches,
    _initialize_model,
    _seed,
    _train_epochs,
)
from hstu_kvcache.streaming.kuairand_query_transition import (
    _atomic_json,
    file_sha256,
    load_config,
)


def _maximum_difference(left, right) -> dict[str, float]:
    output = {"cache_k": 0.0, "cache_v": 0.0, "metrics": 0.0}
    for left_batch, right_batch in zip(left, right, strict=True):
        output["cache_k"] = max(
            output["cache_k"],
            float(torch.max(torch.abs(left_batch["cache"].k - right_batch["cache"].k)).item()),
        )
        output["cache_v"] = max(
            output["cache_v"],
            float(torch.max(torch.abs(left_batch["cache"].v - right_batch["cache"].v)).item()),
        )
        for metric in left_batch["previous_metrics"]:
            output["metrics"] = max(
                output["metrics"],
                float(
                    torch.max(
                        torch.abs(
                            left_batch["previous_metrics"][metric]
                            - right_batch["previous_metrics"][metric]
                        )
                    ).item()
                ),
            )
    return output


def _reduce_max(values: dict[str, float], device: torch.device) -> dict[str, float]:
    output = {}
    for key, value in values.items():
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        output[key] = float(tensor.item())
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.config)
    document = load_persistent_config(path)
    rank, world_size, device = _distributed(document)
    _seed(int(document["training"]["seed"]))
    base_config = load_config(document["parent"]["base_config"]["path"])
    _, workloads = _build_workloads(document, base_config, rank, 1)
    workload = workloads[0]
    embedding_rows = int(workload["metadata"]["embedding_rows"])
    large_dense, large_embedding, large_tracker, large_geometry = _initialize_model(
        document,
        base_config,
        embedding_rows,
        rank,
        world_size,
        device,
    )
    reference_document = json.loads(json.dumps(document))
    reference_document["model"]["embedding_replicas"] = 1
    reference_document["model"]["require_single_card_overflow"] = False
    reference_dense, reference_embedding, reference_tracker, reference_geometry = (
        _initialize_model(
            reference_document,
            base_config,
            embedding_rows,
            rank,
            world_size,
            device,
        )
    )
    batches = _evaluation_batches(workload, 2, rank, world_size)[:2]
    reference_initial = _capture_old(
        reference_dense,
        reference_embedding,
        batches,
        workload,
        base_config,
        device,
    )
    large_initial = _capture_old(
        large_dense,
        large_embedding,
        batches,
        workload,
        base_config,
        device,
    )
    initial_difference = _reduce_max(
        _maximum_difference(reference_initial, large_initial),
        device,
    )
    candidate = {
        "dense_lr": 0.0001,
        "embedding_lr": 0.00001,
        "kv_lr": 0.00025,
        "maximum_update_examples": 8,
        "name": "single_step_equivalence",
        "projection_lr": 0.00000001,
        "update_epochs": 1,
    }
    candidate_document = _training_document(document, candidate)
    training_seed = int(document["training"]["seed"]) + 2003
    _seed(training_seed)
    reference_training = _train_epochs(
        reference_dense,
        reference_embedding,
        reference_tracker,
        workload["update_examples"],
        workload,
        base_config,
        candidate_document,
        rank,
        world_size,
        device,
        "canary_reference",
        training_seed,
    )
    _seed(training_seed)
    large_training = _train_epochs(
        large_dense,
        large_embedding,
        large_tracker,
        workload["update_examples"],
        workload,
        base_config,
        candidate_document,
        rank,
        world_size,
        device,
        "canary_large",
        training_seed,
    )
    reference_after = _capture_old(
        reference_dense,
        reference_embedding,
        batches,
        workload,
        base_config,
        device,
    )
    large_after = _capture_old(
        large_dense,
        large_embedding,
        batches,
        workload,
        base_config,
        device,
    )
    update_difference = _reduce_max(
        _maximum_difference(reference_after, large_after),
        device,
    )
    active = torch.tensor(
        [reference_tracker.local_active_count, large_tracker.local_active_count],
        dtype=torch.int64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(active)
    if rank == 0:
        tolerances = {
            "initial_cache_absolute": 0.00002,
            "single_step_cache_absolute": 0.0002,
            "ranking_metric_absolute": 0.000001,
        }
        result = {
            "protocol": "evokv_kuairand_replicated_capacity_canary_v0",
            "status": "complete",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(path), "sha256": file_sha256(path)},
            "workload": workload["metadata"],
            "reference_geometry": reference_geometry,
            "large_geometry": large_geometry,
            "initial_maximum_absolute_difference": initial_difference,
            "single_step_maximum_absolute_difference": update_difference,
            "reference_training": reference_training,
            "large_training": large_training,
            "optimizer_active_rows": {
                "reference": int(active[0].item()),
                "large": int(active[1].item()),
            },
            "tolerances": tolerances,
            "passed": bool(
                max(initial_difference["cache_k"], initial_difference["cache_v"])
                <= tolerances["initial_cache_absolute"]
                and initial_difference["metrics"]
                <= tolerances["ranking_metric_absolute"]
                and max(update_difference["cache_k"], update_difference["cache_v"])
                <= tolerances["single_step_cache_absolute"]
                and update_difference["metrics"]
                <= tolerances["ranking_metric_absolute"]
                and int(active[1].item()) >= 7 * int(active[0].item())
                and bool(large_geometry["single_gpu_parameter_overflow"])
            ),
        }
        _atomic_json(Path(args.output), result)
        print(json.dumps(result, indent=2, sort_keys=True))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
