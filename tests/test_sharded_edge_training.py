from __future__ import annotations

import importlib.util
import json
import multiprocessing
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from hstu_kvcache.streaming.sharded_edge import (
    SHARDED_EDGE_CHECKPOINT_SCHEMA,
    ExternalEmbeddingHSTU,
    ShardedEdgeModelSpec,
    TrainableModuloRowShardedEmbedding,
    evaluate_fixed_heldout,
    load_sharded_edge_checkpoint,
    make_fixed_heldout_batch,
    model_memory_estimate,
    save_sharded_edge_checkpoint,
    sharded_edge_train_step,
    sparse_sgd,
)

ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts" / "train_evokv_design3_m1_qk_sharded_edge.py"
TRAINER_SPEC = importlib.util.spec_from_file_location(
    "train_evokv_design3_m1_qk_sharded_edge",
    TRAINER_PATH,
)
assert TRAINER_SPEC is not None and TRAINER_SPEC.loader is not None
TRAINER = importlib.util.module_from_spec(TRAINER_SPEC)
sys.modules[TRAINER_SPEC.name] = TRAINER
TRAINER_SPEC.loader.exec_module(TRAINER)
CALIBRATION_PATH = ROOT / "scripts" / "calibrate_evokv_design3_m1_qk_sharded_edge.py"
CALIBRATION_SPEC = importlib.util.spec_from_file_location(
    "calibrate_evokv_design3_m1_qk_sharded_edge",
    CALIBRATION_PATH,
)
assert CALIBRATION_SPEC is not None and CALIBRATION_SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(CALIBRATION_SPEC)
sys.modules[CALIBRATION_SPEC.name] = CALIBRATION
CALIBRATION_SPEC.loader.exec_module(CALIBRATION)


def tiny_spec() -> ShardedEdgeModelSpec:
    return ShardedEdgeModelSpec(
        num_embeddings=33,
        num_prediction_items=24,
        num_behaviors=3,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        head_dim=4,
        max_seq_len=8,
    )


def tiny_batch(
    item_ids: list[int] | None = None,
    labels: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    items = item_ids or [1, 2, 3, 4, 5, 6]
    target_labels = labels or [1] * len(items)
    return {
        "item_ids": torch.tensor([items], dtype=torch.int64),
        "behaviors": torch.tensor(
            [[1 + index % 2 for index in range(len(items))]],
            dtype=torch.int64,
        ),
        "time_deltas": torch.arange(
            len(items),
            dtype=torch.float32,
        ).unsqueeze(0),
        "labels": torch.tensor([target_labels], dtype=torch.int64),
        "lengths": torch.tensor([len(items)], dtype=torch.int64),
        "train_mask": torch.ones(
            (1, len(items)),
            dtype=torch.bool,
        ),
    }


def test_large_m1_geometry() -> None:
    spec = ShardedEdgeModelSpec(
        num_embeddings=2_859_836,
        num_prediction_items=250_000,
        num_behaviors=5,
        hidden_size=1536,
        num_layers=24,
        num_heads=24,
        head_dim=64,
        max_seq_len=512,
    )
    memory = model_memory_estimate(spec, world_size=2, kv_records=2048)
    assert memory["global_embedding_bytes_fp32"] == (2_859_836 * 1536 * 4)
    assert memory["maximum_local_embedding_bytes_fp32"] == (1_429_918 * 1536 * 4)
    assert memory["dense_parameters"] == 285_571_584
    assert memory["kv_bytes_fp16_per_record"] == 72 * 2**20
    assert memory["old_plus_target_kv_bytes_fp16"] == 288 * 2**30
    assert memory["old_plus_target_kv_gib"] == 288.0
    trainer_spec = TRAINER.model_spec(250_000, 5)
    assert trainer_spec == spec
    assert TRAINER.DEFAULT_PREPARED_DATA.endswith("evokv_d3_m1_qk_entity_2560.npz")
    assert TRAINER.KV_RECORDS == 2048
    assert CALIBRATION.model_spec(250_000, 5) == spec
    assert CALIBRATION.KV_RECORDS == 2048


def test_calibration_selects_one_positive_real_batch_per_rank() -> None:
    class FakePlan:
        def iter_base_train_batches(self, *args, **kwargs):
            yield tiny_batch(labels=[0, 0, 0, 0, 0, 0])
            yield tiny_batch([1, 2, 3], [1, 1, 1])
            yield tiny_batch([4, 5, 6], [1, 1, 1])
            yield tiny_batch([7, 8, 9], [1, 1, 1])

    selected = CALIBRATION.select_real_base_batches(
        FakePlan(),
        batch_size=1,
        world_size=2,
    )
    assert len(selected) == 2
    assert selected[0]["item_ids"].tolist() == [[1, 2, 3]]
    assert selected[1]["item_ids"].tolist() == [[4, 5, 6]]


def test_single_rank_sparse_training_heldout_and_checkpoint(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    spec = tiny_spec()
    dense = ExternalEmbeddingHSTU(spec.hstu_config())
    assert model_memory_estimate(
        spec,
        world_size=1,
        kv_records=1,
    )["dense_parameters"] == sum(value.numel() for value in dense.parameters())
    embedding = TrainableModuloRowShardedEmbedding.initialize(
        spec.num_embeddings,
        spec.hidden_size,
        rank=0,
        world_size=1,
        device="cpu",
        seed=11,
    )
    dense_optimizer = torch.optim.AdamW(
        dense.parameters(),
        lr=1e-3,
        foreach=False,
    )
    embedding_optimizer = sparse_sgd(embedding, 1e-2)
    batch = tiny_batch()
    heldout = [
        make_fixed_heldout_batch(
            batch,
            spec.num_prediction_items,
            negative_count=15,
            seed=101,
        )
    ]
    before = evaluate_fixed_heldout(
        dense,
        embedding,
        heldout,
        torch.device("cpu"),
    )
    loss, local_targets, global_targets = sharded_edge_train_step(
        dense,
        embedding,
        batch,
        dense_optimizer,
        embedding_optimizer,
        torch.device("cpu"),
        spec.num_prediction_items,
        negative_count=4,
        negative_seed=303,
    )
    after = evaluate_fixed_heldout(
        dense,
        embedding,
        heldout,
        torch.device("cpu"),
    )
    assert torch.isfinite(torch.tensor(loss))
    assert local_targets == global_targets == 5
    assert embedding.local_weight.grad is not None
    assert embedding.local_weight.grad.is_sparse
    assert before["positive_targets"] == after["positive_targets"] == 5
    assert 0.0 <= float(after["hit_rate_at_10"]) <= 1.0
    dense_reference = {name: value.detach().clone() for name, value in dense.state_dict().items()}
    embedding_reference = embedding.local_weight.detach().clone()
    manifest = save_sharded_edge_checkpoint(
        tmp_path,
        0,
        spec,
        dense,
        embedding,
    )
    assert manifest["schema"] == SHARDED_EDGE_CHECKPOINT_SCHEMA
    with torch.no_grad():
        for parameter in dense.parameters():
            parameter.zero_()
        embedding.local_weight.zero_()
    loaded = load_sharded_edge_checkpoint(
        tmp_path,
        0,
        spec,
        dense,
        embedding,
    )
    assert loaded == manifest
    assert torch.equal(
        embedding.local_weight,
        embedding_reference,
    )
    for name, value in dense.state_dict().items():
        assert torch.equal(value, dense_reference[name])
    raw_manifest = json.loads((tmp_path / "theta_0" / "manifest.json").read_text())
    assert raw_manifest["embedding_shards"][0]["local_rows"] == 33


def _two_rank_worker(
    rank: int,
    init_path: str,
    output_directory: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        embedding_group = dist.new_group(ranks=[0, 1])
        hidden_size = 3
        global_weight = torch.arange(
            5 * hidden_size,
            dtype=torch.float32,
        ).reshape(5, hidden_size)
        local_weight = global_weight[rank::2].clone().requires_grad_(False)
        embedding = TrainableModuloRowShardedEmbedding(
            local_weight,
            num_embeddings=5,
            rank=rank,
            world_size=2,
            process_group=embedding_group,
        )
        requests = (
            torch.tensor([[1, 2, 3]], dtype=torch.int64)
            if rank == 0
            else torch.tensor([[2, 3, 4]], dtype=torch.int64)
        )
        vectors = embedding(
            requests,
            torch.tensor([3], dtype=torch.int64),
        )
        assert torch.equal(
            vectors,
            global_weight.index_select(
                0,
                requests.flatten(),
            ).reshape(1, 3, hidden_size),
        )
        (vectors.sum() * (rank + 1)).backward()
        gradient = embedding.local_weight.grad.coalesce().to_dense()
        expected = (
            torch.tensor([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [2.0, 2.0, 2.0]])
            if rank == 0
            else torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        )
        assert torch.equal(gradient, expected)
        spec = tiny_spec()
        torch.manual_seed(17)
        dense = ExternalEmbeddingHSTU(spec.hstu_config())
        dense_ddp = DistributedDataParallel(dense)
        train_embedding = TrainableModuloRowShardedEmbedding.initialize(
            spec.num_embeddings,
            spec.hidden_size,
            rank,
            2,
            "cpu",
            seed=19,
            process_group=embedding_group,
        )
        dense_optimizer = torch.optim.SGD(
            dense_ddp.parameters(),
            lr=1e-3,
        )
        embedding_optimizer = sparse_sgd(
            train_embedding,
            1e-2,
        )
        batch = (
            tiny_batch([1, 2, 3, 4], [1, 1, 1, 1])
            if rank == 0
            else tiny_batch([5, 6, 7, 8], [1, 0, 0, 1])
        )
        loss, local_targets, global_targets = sharded_edge_train_step(
            dense_ddp,
            train_embedding,
            batch,
            dense_optimizer,
            embedding_optimizer,
            torch.device("cpu"),
            spec.num_prediction_items,
            negative_count=3,
            negative_seed=401 + rank,
        )
        assert torch.isfinite(torch.tensor(loss))
        assert local_targets == (3 if rank == 0 else 1)
        assert global_targets == 4
        checksum = torch.stack(
            [parameter.detach().double().sum() for parameter in dense_ddp.module.parameters()]
        ).sum()
        checksums = [torch.zeros_like(checksum) for _ in range(2)]
        dist.all_gather(checksums, checksum)
        assert torch.equal(checksums[0], checksums[1])
        manifest = save_sharded_edge_checkpoint(
            Path(output_directory) / "checkpoint",
            0,
            spec,
            dense_ddp,
            train_embedding,
        )
        assert manifest["world_size"] == 2
        assert len(manifest["embedding_shards"]) == 2
        Path(output_directory, f"rank_{rank}.json").write_text(
            json.dumps(
                {
                    "local_targets": local_targets,
                    "global_targets": global_targets,
                    "embedding_gradient_sparse": bool(train_embedding.local_weight.grad.is_sparse),
                    "checkpoint_schema": manifest["schema"],
                }
            )
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_owner_gradient_and_target_normalization(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "gloo_init"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_two_rank_worker,
            args=(rank, str(init_path), str(tmp_path)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    records = [json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(2)]
    assert [value["local_targets"] for value in records] == [3, 1]
    assert [value["global_targets"] for value in records] == [4, 4]
    assert all(value["embedding_gradient_sparse"] for value in records)
    assert all(value["checkpoint_schema"] == SHARDED_EDGE_CHECKPOINT_SCHEMA for value in records)
