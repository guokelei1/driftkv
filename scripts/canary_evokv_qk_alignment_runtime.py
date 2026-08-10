from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from hstu_kvcache.streaming.qk_stream_version import (
    distributed_full_catalog_metrics,
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
        raise RuntimeError("QK alignment canary requires two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    try:
        spec = XPProjectedModelSpec(
            num_embeddings=128,
            embedding_width=32,
            hidden_size=16,
            num_prediction_items=64,
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
        hidden = torch.randn(
            4,
            3,
            spec.hidden_size,
            generator=torch.Generator(device=device).manual_seed(307 + rank),
            device=device,
        )
        positive_ids = torch.tensor([1 + rank, 5 + rank, 9 + rank], device=device)
        rolling_nll, rolling_ranks = distributed_full_catalog_metrics(
            embedding,
            hidden[:2],
            positive_ids,
            3,
            num_prediction_items=spec.num_prediction_items,
            item_chunk=11,
        )
        boundary_nll, boundary_ranks = distributed_full_catalog_metrics(
            embedding,
            hidden[2:],
            positive_ids,
            3,
            num_prediction_items=spec.num_prediction_items,
            item_chunk=11,
        )
        nll = torch.cat((rolling_nll, boundary_nll), dim=0)
        ranks = torch.cat((rolling_ranks, boundary_ranks), dim=0)
        catalog_ids = torch.arange(
            1, spec.num_prediction_items + 1, device=device
        ).unsqueeze(0)
        catalog_vectors = embedding(
            catalog_ids,
            torch.tensor(
                [spec.num_prediction_items], dtype=torch.int64, device=device
            ),
        )
        scores = torch.einsum("mnh,ceh->mne", hidden, catalog_vectors)
        positive_scores = scores.gather(
            2,
            (positive_ids - 1).view(1, 3, 1).expand(4, -1, -1),
        ).squeeze(-1)
        expected_nll = torch.logsumexp(scores, dim=-1) - positive_scores
        expected_ranks = (scores >= positive_scores.unsqueeze(-1)).sum(dim=-1)
        passed = torch.tensor(
            int(
                torch.allclose(nll, expected_nll, atol=1e-4, rtol=1e-4)
                and torch.equal(ranks, expected_ranks)
            ),
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        reports: list[object] = [None] * world_size
        dist.all_gather_object(
            reports,
            {
                "rank": rank,
                "four_method_full_catalog_equivalence": bool(passed.item()),
                "shape": list(nll.shape),
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
            raise RuntimeError("QK alignment canary failed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
