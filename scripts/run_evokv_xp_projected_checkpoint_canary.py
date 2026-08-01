from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from hstu_kvcache.streaming.sharded_edge import ExternalEmbeddingHSTU
from hstu_kvcache.streaming.xp_projected_edge import (
    OptimizerActiveRowTracker,
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
    load_xp_projected_checkpoint,
    save_xp_projected_checkpoint,
    sparse_embedding_sgd,
    tracked_sparse_optimizer_step,
)

PROTOCOL = "evokv_xp_projected_checkpoint_canary_development_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backend")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/tmp/evokv_xp_projected_canary"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/evokv_xp_projected_canary.json"),
    )
    return parser.parse_args()


def _device(kind: str) -> torch.device:
    if kind == "cpu":
        return torch.device("cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _batches(device: torch.device) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    ...,
]:
    return (
        (
            torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 16]],
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor([4, 3], dtype=torch.int64, device=device),
        ),
        (
            torch.tensor(
                [[7, 8, 9, 10], [11, 12, 13, 14]],
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor([4, 4], dtype=torch.int64, device=device),
        ),
    )


def _masked_reference(
    weight: torch.Tensor,
    projection: torch.Tensor,
    item_ids: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    output = F.linear(F.embedding(item_ids, weight), projection)
    valid = (
        torch.arange(
            item_ids.shape[1],
            device=item_ids.device,
        ).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    return output * valid.unsqueeze(-1)


def _dense_equal(
    first: ExternalEmbeddingHSTU,
    second: ExternalEmbeddingHSTU,
) -> bool:
    first_state = first.state_dict()
    second_state = second.state_dict()
    return first_state.keys() == second_state.keys() and all(
        torch.equal(first_state[name], second_state[name])
        for name in first_state
    )


def run(args: argparse.Namespace) -> dict[str, object] | None:
    backend = args.backend or (
        "nccl" if args.device == "cuda" else "gloo"
    )
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError("XP projected canary requires two ranks")
    device = _device(args.device)
    spec = XPProjectedModelSpec(
        num_embeddings=17,
        embedding_width=7,
        hidden_size=4,
        num_prediction_items=8,
        num_behaviors=3,
        num_layers=1,
        num_heads=2,
        head_dim=2,
        max_seq_len=8,
    )
    global_weight = (
        torch.arange(
            spec.num_embeddings * spec.embedding_width,
            dtype=torch.float32,
            device=device,
        ).reshape(spec.num_embeddings, spec.embedding_width)
        / 97.0
    )
    global_weight[0].zero_()
    projection = (
        torch.arange(
            spec.hidden_size * spec.embedding_width,
            dtype=torch.float32,
            device=device,
        ).reshape(spec.hidden_size, spec.embedding_width)
        / 53.0
        - 0.2
    )
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=global_weight[rank::world_size].clone(),
        projection_weight=projection.clone(),
        num_embeddings=spec.num_embeddings,
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(8123)
    dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
    tracker = OptimizerActiveRowTracker(
        num_embeddings=spec.num_embeddings,
        rank=rank,
        world_size=world_size,
    )
    batches = _batches(device)
    item_ids, lengths = batches[rank]
    output = embedding(item_ids, lengths)
    expected = _masked_reference(
        global_weight,
        projection,
        item_ids,
        lengths,
    )
    forward_error = float(
        torch.max(torch.abs(output - expected)).item()
    )
    scale = float(rank + 1)
    loss = (output * scale).square().sum()
    loss.backward()
    reference_weight = global_weight.detach().clone().requires_grad_()
    reference_projection = (
        projection.detach().clone().requires_grad_()
    )
    reference_loss = torch.zeros((), device=device)
    for other_rank, (other_ids, other_lengths) in enumerate(batches):
        reference_output = _masked_reference(
            reference_weight,
            reference_projection,
            other_ids,
            other_lengths,
        )
        reference_loss = reference_loss + (
            reference_output * float(other_rank + 1)
        ).square().sum()
    reference_loss.backward()
    local_reference_gradient = reference_weight.grad[
        rank::world_size
    ]
    embedding_gradient = embedding.local_weight.grad
    if embedding_gradient is None or not embedding_gradient.is_sparse:
        raise RuntimeError("XP projected canary gradient is not sparse")
    embedding_gradient_error = float(
        torch.max(
            torch.abs(
                embedding_gradient.to_dense()
                - local_reference_gradient
            )
        ).item()
    )
    projection_gradient_error = float(
        torch.max(
            torch.abs(
                embedding.projection_weight.grad
                - reference_projection.grad
            )
        ).item()
    )
    sparse_optimizer = sparse_embedding_sgd(
        embedding,
        learning_rate=0.1,
    )
    projection_optimizer = torch.optim.SGD(
        [embedding.projection_weight],
        lr=0.05,
    )
    projection_optimizer.step()
    tracked_sparse_optimizer_step(
        embedding,
        sparse_optimizer,
        tracker,
    )
    expected_active = set(
        int(value)
        for value in torch.nonzero(
            torch.any(reference_weight.grad != 0, dim=1),
            as_tuple=False,
        ).flatten().tolist()
        if int(value) != 0
    )
    local_active = tracker.local_global_row_ids()
    gathered_active: list[object] = [None] * world_size
    dist.all_gather_object(gathered_active, local_active)
    observed_active = {
        int(value)
        for rank_values in gathered_active
        for value in rank_values
    }
    active_rows_match = observed_active == expected_active
    saved_embedding = embedding.local_weight.detach().clone()
    saved_projection = embedding.projection_weight.detach().clone()
    manifest = save_xp_projected_checkpoint(
        args.checkpoint_dir,
        0,
        spec,
        dense,
        embedding,
        tracker,
    )
    reloaded_embedding = TrainableProjectedModuloEmbedding(
        local_weight=torch.zeros_like(embedding.local_weight),
        projection_weight=torch.zeros_like(
            embedding.projection_weight
        ),
        num_embeddings=spec.num_embeddings,
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(1907)
    reloaded_dense = ExternalEmbeddingHSTU(
        spec.hstu_config()
    ).to(device)
    reloaded_tracker = OptimizerActiveRowTracker(
        num_embeddings=spec.num_embeddings,
        rank=rank,
        world_size=world_size,
    )
    loaded = load_xp_projected_checkpoint(
        args.checkpoint_dir,
        0,
        spec,
        reloaded_dense,
        reloaded_embedding,
        reloaded_tracker,
    )
    local_passed = bool(
        forward_error < 1e-5
        and embedding_gradient_error < 1e-5
        and projection_gradient_error < 1e-5
        and active_rows_match
        and torch.equal(
            saved_embedding,
            reloaded_embedding.local_weight,
        )
        and torch.equal(
            saved_projection,
            reloaded_embedding.projection_weight,
        )
        and tracker.local_global_row_ids()
        == reloaded_tracker.local_global_row_ids()
        and _dense_equal(dense, reloaded_dense)
        and loaded == manifest
    )
    report = {
        "rank": rank,
        "forward_max_abs_error": forward_error,
        "embedding_gradient_max_abs_error": (
            embedding_gradient_error
        ),
        "projection_gradient_max_abs_error": (
            projection_gradient_error
        ),
        "local_active_row_ids": list(local_active),
        "active_rows_match_reference": active_rows_match,
        "checkpoint_reload_passed": local_passed,
    }
    gathered_reports: list[object] = [None] * world_size
    dist.all_gather_object(gathered_reports, report)
    dist.barrier()
    result = None
    if rank == 0:
        checkpoint_bytes = sum(
            path.stat().st_size
            for path in (
                args.checkpoint_dir / "theta_0"
            ).iterdir()
            if path.is_file()
        )
        result = {
            "protocol": PROTOCOL,
            "status": (
                "complete"
                if all(
                    bool(value["checkpoint_reload_passed"])
                    for value in gathered_reports
                )
                else "failed"
            ),
            "scientific_result": False,
            "formal_design2": False,
            "formal_design3": False,
            "artifact_role": (
                "two_rank_projected_autograd_checkpoint_canary"
            ),
            "device": args.device,
            "backend": backend,
            "world_size": world_size,
            "spec": {
                key: value
                for key, value in spec.__dict__.items()
            },
            "owner_side_projection": True,
            "projection_bias": False,
            "optimizer_active_rows": manifest[
                "optimizer_active_rows"
            ],
            "expected_active_row_ids": sorted(expected_active),
            "checkpoint": {
                "root": str(args.checkpoint_dir),
                "version": 0,
                "bytes": checkpoint_bytes,
                "schema": manifest["schema"],
            },
            "ranks": gathered_reports,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    dist.barrier()
    dist.destroy_process_group()
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "complete":
            raise RuntimeError("XP projected checkpoint canary failed")


if __name__ == "__main__":
    main()
