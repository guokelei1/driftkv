from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from hstu_kvcache.streaming.qk_stream_version import (
    distributed_full_catalog_metrics,
    distributed_full_catalog_topk,
    distributed_projected_candidate_scores,
    fp16_storage_fp32_consumption,
)
from hstu_kvcache.streaming.sharded_edge import (
    ExternalEmbeddingHSTU,
    modulo_local_rows,
)
from hstu_kvcache.streaming.xp_projected_edge import (
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError("QK stream runtime canary requires two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", device_id=device
    )
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
        generator = torch.Generator(device=device).manual_seed(101 + rank)
        local_weight = torch.randn(
            modulo_local_rows(spec.num_embeddings, rank, world_size),
            spec.embedding_width,
            generator=generator,
            device=device,
        )
        if rank == 0:
            local_weight[0].zero_()
        projection_generator = torch.Generator(device=device).manual_seed(211)
        projection = torch.randn(
            spec.hidden_size,
            spec.embedding_width,
            generator=projection_generator,
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
            2,
            3,
            spec.hidden_size,
            generator=torch.Generator(device=device).manual_seed(307 + rank),
            device=device,
        )
        candidates = (
            torch.tensor(
                [[1, 2, 3, 4, 5], [7, 9, 11, 13, 15], [16, 18, 20, 22, 24]],
                device=device,
            )
            + rank
        )
        direct = distributed_projected_candidate_scores(
            embedding, hidden, candidates, 3
        )
        candidate_vectors = embedding(
            candidates,
            torch.full((3,), 5, dtype=torch.int64, device=device),
        )
        expected = torch.einsum("mnh,nch->mnc", hidden, candidate_vectors)
        score_pass = torch.allclose(direct, expected, atol=1e-4, rtol=1e-4)
        positive_ids = torch.tensor(
            [1 + rank, 5 + rank, 9 + rank], device=device
        )
        full_nll, full_ranks = distributed_full_catalog_metrics(
            embedding,
            hidden,
            positive_ids,
            3,
            num_prediction_items=spec.num_prediction_items,
            item_chunk=11,
        )
        catalog_ids = torch.arange(
            1, spec.num_prediction_items + 1, device=device
        ).unsqueeze(0)
        catalog_vectors = embedding(
            catalog_ids,
            torch.tensor(
                [spec.num_prediction_items], dtype=torch.int64, device=device
            ),
        )
        full_scores = torch.einsum(
            "mnh,ceh->mne", hidden, catalog_vectors
        )
        positive_scores = full_scores.gather(
            2,
            (positive_ids - 1).view(1, 3, 1).expand(2, -1, -1),
        ).squeeze(-1)
        expected_full_nll = (
            torch.logsumexp(full_scores, dim=-1) - positive_scores
        )
        expected_full_ranks = (
            full_scores >= positive_scores.unsqueeze(-1)
        ).sum(dim=-1)
        full_catalog_pass = bool(
            torch.allclose(
                full_nll, expected_full_nll, atol=1e-4, rtol=1e-4
            )
            and torch.equal(full_ranks, expected_full_ranks)
        )
        topk_scores, topk_ids = distributed_full_catalog_topk(
            embedding,
            hidden[:, 0],
            num_prediction_items=spec.num_prediction_items,
            maximum_k=10,
            item_chunk=11,
        )
        expected_topk_scores, expected_topk_positions = torch.topk(
            full_scores[:, 0],
            10,
            dim=1,
        )
        topk_pass = bool(
            torch.allclose(
                topk_scores,
                expected_topk_scores,
                atol=1e-4,
                rtol=1e-4,
            )
            and torch.equal(topk_ids, expected_topk_positions + 1)
        )
        torch.manual_seed(401)
        source_dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
        current_dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
        current_dense.load_state_dict(source_dense.state_dict())
        with torch.no_grad():
            next(current_dense.parameters()).add_(0.01)
        prefix_length = 7 + rank * 4
        suffix_length = 3 + rank * 2
        prefix_items = (
            torch.arange(1, prefix_length + 1, device=device).unsqueeze(0)
            + rank
        )
        prefix_lengths = torch.tensor(
            [prefix_length], dtype=torch.int64, device=device
        )
        prefix_vectors = embedding(prefix_items, prefix_lengths)
        behaviors = torch.ones_like(prefix_items)
        deltas = torch.ones_like(prefix_items, dtype=torch.float32)
        deltas[:, 0] = 0.0
        source_cache = source_dense.core.compute_kv_from_item_embeddings(
            prefix_vectors,
            behaviors,
            deltas,
            prefix_lengths,
        )
        exact_cache = current_dense.core.compute_kv_from_item_embeddings(
            prefix_vectors,
            behaviors,
            deltas,
            prefix_lengths,
        )
        source_cache = fp16_storage_fp32_consumption(source_cache)
        exact_cache = fp16_storage_fp32_consumption(exact_cache)
        suffix_items = (
            torch.arange(33, 33 + suffix_length, device=device).unsqueeze(0)
            + rank
        )
        suffix_lengths = torch.tensor(
            [suffix_length], dtype=torch.int64, device=device
        )
        suffix_vectors = embedding(suffix_items, suffix_lengths)
        suffix_behaviors = torch.ones_like(suffix_items)
        suffix_deltas = torch.ones_like(suffix_items, dtype=torch.float32)
        reuse_hidden, _ = current_dense.core.forward_with_cache_from_item_embeddings(
            source_cache,
            suffix_vectors,
            suffix_behaviors,
            suffix_deltas,
        )
        exact_hidden, _ = current_dense.core.forward_with_cache_from_item_embeddings(
            exact_cache,
            suffix_vectors,
            suffix_behaviors,
            suffix_deltas,
        )
        cache_pass = bool(
            torch.all(torch.isfinite(reuse_hidden))
            and torch.all(torch.isfinite(exact_hidden))
            and reuse_hidden.shape == exact_hidden.shape
            and not torch.equal(reuse_hidden, exact_hidden)
        )
        passed = torch.tensor(
            int(
                score_pass
                and full_catalog_pass
                and topk_pass
                and cache_pass
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
                "score_equivalence": bool(score_pass),
                "full_catalog_equivalence": full_catalog_pass,
                "full_catalog_topk_equivalence": topk_pass,
                "natural_prefix_length": prefix_length,
                "natural_append_length": suffix_length,
                "cache_paths_differ": cache_pass,
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
            raise RuntimeError("QK stream runtime canary failed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
