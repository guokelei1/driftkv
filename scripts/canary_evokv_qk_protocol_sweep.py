from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from hstu_kvcache.streaming.qk_protocol_sweep_runner import (
    candidate_score_sums,
    nested_popular_candidate_ids,
    nested_uniform_candidate_ids,
)
from hstu_kvcache.streaming.qk_stream_version import (
    distributed_projected_candidate_scores,
)
from hstu_kvcache.streaming.sharded_edge import modulo_local_rows
from hstu_kvcache.streaming.xp_projected_edge import (
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError("QK protocol sweep canary requires two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    try:
        spec = XPProjectedModelSpec(
            num_embeddings=256,
            embedding_width=32,
            hidden_size=16,
            num_prediction_items=128,
            num_behaviors=5,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=32,
        )
        local_weight = torch.randn(
            modulo_local_rows(spec.num_embeddings, rank, world_size),
            spec.embedding_width,
            generator=torch.Generator(device=device).manual_seed(101 + rank),
            device=device,
        )
        if rank == 0:
            local_weight[0].zero_()
        projection = torch.randn(
            spec.hidden_size,
            spec.embedding_width,
            generator=torch.Generator(device=device).manual_seed(211),
            device=device,
        )
        embedding = TrainableProjectedModuloEmbedding(
            local_weight=local_weight,
            projection_weight=projection,
            num_embeddings=spec.num_embeddings,
            rank=rank,
            world_size=world_size,
        )
        positives = torch.tensor([7 + rank, 29 + rank])
        uniform = [
            nested_uniform_candidate_ids(
                positives,
                num_prediction_items=spec.num_prediction_items,
                maximum_negative_count=99,
                seed=seed + rank,
            )
            for seed in (31, 43, 59)
        ]
        popular = nested_popular_candidate_ids(
            positives,
            torch.arange(1, spec.num_prediction_items + 1),
            maximum_negative_count=99,
        )
        positive_popular = nested_popular_candidate_ids(
            positives,
            torch.arange(spec.num_prediction_items, 0, -1),
            maximum_negative_count=99,
        )
        candidates = torch.cat(
            (*uniform, popular, positive_popular), dim=1
        ).to(device)
        hidden = torch.randn(
            2,
            2,
            spec.hidden_size,
            generator=torch.Generator(device=device).manual_seed(307 + rank),
            device=device,
        )
        scores = distributed_projected_candidate_scores(
            embedding, hidden, candidates, 2
        )
        candidate_vectors = embedding(
            candidates,
            torch.full((2,), candidates.shape[1], dtype=torch.int64, device=device),
        )
        expected = torch.einsum("mnh,nch->mnc", hidden, candidate_vectors)
        score_pass = torch.allclose(scores, expected, atol=1e-4, rtol=1e-4)
        metric_pass = True
        for segment in range(5):
            left = segment * 100
            for count in (49, 99):
                values = candidate_score_sums(scores[:, :, left : left + count + 1])
                metric_pass = metric_pass and values.shape == (2, 7)
                metric_pass = metric_pass and bool(torch.from_numpy(values).isfinite().all())
        passed = torch.tensor(
            int(score_pass and metric_pass), dtype=torch.int64, device=device
        )
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        reports: list[object] = [None] * world_size
        dist.all_gather_object(
            reports,
            {
                "rank": rank,
                "five_pool_score_equivalence": bool(score_pass),
                "nested_metric_slices": bool(metric_pass),
                "candidate_width": candidates.shape[1],
            },
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "pass" if passed.item() else "fail",
                        "world_size": world_size,
                        "ranks": reports,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        if not passed.item():
            raise RuntimeError("QK protocol sweep canary failed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
