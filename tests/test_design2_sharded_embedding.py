import torch
import torch.distributed as dist

from hstu_kvcache.migration.design2_embedding import (
    ForbiddenFullItemEmbedding,
    ModuloRowShardedEmbedding,
    build_modulo_sharded_hstu_from_cpu,
    modulo_embedding_local_id,
    modulo_embedding_local_rows,
    modulo_embedding_owner,
    sharded_append_padded_cache,
    sharded_exact_jagged_hidden_and_kv,
)
from hstu_kvcache.migration.recompute import (
    RawHistoryBatch,
    exact_hidden_and_kv,
)
from hstu_kvcache.migration.stage46_chain import pack_padded_cache
from hstu_kvcache.models import HSTU, HSTUConfig


def test_modulo_embedding_routing_formula_and_uneven_rows() -> None:
    item_ids = torch.tensor([0, 1, 2, 3, 4, 8, 9, 10])
    assert modulo_embedding_owner(item_ids, 4).tolist() == [
        0,
        1,
        2,
        3,
        0,
        0,
        1,
        2,
    ]
    assert modulo_embedding_local_id(item_ids, 4).tolist() == [
        0,
        0,
        0,
        0,
        1,
        2,
        2,
        2,
    ]
    assert [
        modulo_embedding_local_rows(11, rank, 4)
        for rank in range(4)
    ] == [3, 3, 3, 2]


def test_world_one_gloo_lookup_exact_and_empty_rank(tmp_path) -> None:
    rendezvous = tmp_path / "gloo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        weight = torch.arange(30, dtype=torch.float32).reshape(6, 5)
        embedding = ModuloRowShardedEmbedding(
            local_weight=weight,
            num_embeddings=6,
            rank=0,
            world_size=1,
        )
        item_ids = torch.tensor(
            [[0, 2, 99], [3, 3, -7]],
            dtype=torch.long,
        )
        lengths = torch.tensor([2, 2])
        lookup = embedding.lookup(item_ids, lengths)
        assert torch.equal(lookup.item_vectors[0, 0], weight[0])
        assert torch.equal(lookup.item_vectors[0, 1], weight[2])
        assert torch.equal(lookup.item_vectors[1, 0], weight[3])
        assert torch.equal(lookup.item_vectors[1, 1], weight[3])
        assert torch.count_nonzero(lookup.item_vectors[:, 2]) == 0
        assert lookup.metrics.requested_tokens == 4
        assert lookup.metrics.unique_tokens == 3
        assert lookup.metrics.local_requested_tokens == 4
        assert lookup.metrics.remote_requested_tokens == 0
        assert lookup.metrics.collective_calls == 0
        assert lookup.metrics.actual_collective_tensor_payload_bytes == 0

        torch.manual_seed(4)
        full_model = HSTU(
            HSTUConfig(
                num_items=7,
                num_behaviors=3,
                hidden_size=8,
                num_layers=2,
                num_heads=2,
                max_seq_len=4,
                input_dropout=0.0,
            )
        ).eval()
        sharded = build_modulo_sharded_hstu_from_cpu(
            full_model,
            rank=0,
            world_size=1,
            device="cpu",
        )
        assert isinstance(
            sharded.dense_model.item_emb,
            ForbiddenFullItemEmbedding,
        )
        assert not any(
            key.startswith("item_emb.")
            for key in sharded.dense_model.state_dict()
        )
        try:
            sharded.dense_model.lookup_item_embeddings(
                torch.tensor([1])
            )
        except RuntimeError as error:
            assert "forbidden" in str(error)
        else:
            raise AssertionError("full embedding sentinel did not fail")
        batch = RawHistoryBatch(
            record_ids=(10, 11),
            migration_anchor_version="theta1",
            item_ids=torch.tensor([[0, 2, 3], [4, 0, 0]]),
            behaviors=torch.tensor([[1, 2, 3], [2, 0, 0]]),
            time_deltas=torch.tensor(
                [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]
            ),
            lengths=torch.tensor([3, 1]),
        )
        expected_hidden, expected_cache = exact_hidden_and_kv(
            full_model,
            batch.item_ids,
            batch.behaviors,
            batch.time_deltas,
            lengths=batch.lengths,
        )
        expected_fragment = pack_padded_cache(
            expected_cache,
            batch.lengths,
            batch.record_ids,
            "theta2",
            "theta2",
        )
        result = sharded_exact_jagged_hidden_and_kv(
            sharded,
            batch,
            target_version="theta2",
        )
        assert result.fragment is not None
        assert torch.equal(result.fragment.k, expected_fragment.k)
        assert torch.equal(result.fragment.v, expected_fragment.v)
        assert torch.equal(
            result.last_hidden,
            full_model.last_hidden(expected_hidden, batch.lengths),
        )
        empty = RawHistoryBatch(
            record_ids=(),
            migration_anchor_version="theta1",
            item_ids=torch.empty((0, 3), dtype=torch.long),
            behaviors=torch.empty((0, 3), dtype=torch.long),
            time_deltas=torch.empty((0, 3), dtype=torch.float32),
            lengths=torch.empty(0, dtype=torch.long),
        )
        empty_result = sharded_exact_jagged_hidden_and_kv(
            sharded,
            empty,
            target_version="theta2",
        )
        assert empty_result.fragment is None
        assert empty_result.last_hidden.shape == (0, 8)
        assert empty_result.lookup_metrics.requested_tokens == 0
    finally:
        dist.destroy_process_group()


def test_world_one_sharded_delta_and_latest_append_bitwise(
    tmp_path,
) -> None:
    rendezvous = tmp_path / "append-gloo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        torch.manual_seed(7)
        full_model = HSTU(
            HSTUConfig(
                num_items=11,
                num_behaviors=3,
                hidden_size=8,
                num_layers=2,
                num_heads=2,
                max_seq_len=8,
                input_dropout=0.0,
            )
        ).eval()
        sharded = build_modulo_sharded_hstu_from_cpu(
            full_model,
            rank=0,
            world_size=1,
            device="cpu",
        )
        prefix_ids = torch.tensor([[1, 2, 3]])
        prefix_behaviors = torch.tensor([[1, 2, 1]])
        prefix_times = torch.tensor([[0.0, 1.0, 2.0]])
        _, retained_cache = full_model(
            prefix_ids,
            prefix_behaviors,
            prefix_times,
            return_kv=True,
        )
        assert retained_cache is not None
        delta_ids = torch.tensor([[4, 5, 0]])
        delta_behaviors = torch.tensor([[2, 1, 0]])
        delta_times = torch.tensor([[3.0, 4.0, 0.0]])
        expected_delta_hidden, expected_delta_cache = (
            full_model.forward_with_cache(
                retained_cache,
                delta_ids[:, :2],
                delta_behaviors[:, :2],
                delta_times[:, :2],
            )
        )
        delta = sharded_append_padded_cache(
            sharded,
            retained_cache,
            delta_ids,
            delta_behaviors,
            delta_times,
            suffix_lengths=torch.tensor([2]),
        )
        assert torch.equal(delta.updated_cache.k, expected_delta_cache.k)
        assert torch.equal(delta.updated_cache.v, expected_delta_cache.v)
        assert torch.equal(
            delta.last_hidden,
            expected_delta_hidden[:, -1],
        )
        assert delta.lengths.tolist() == [5]
        assert delta.lookup_metrics.requested_tokens == 2
        latest_ids = torch.tensor([[6]])
        latest_behaviors = torch.tensor([[3]])
        latest_times = torch.tensor([[5.0]])
        expected_latest_hidden, expected_latest_cache = (
            full_model.forward_with_cache(
                expected_delta_cache,
                latest_ids,
                latest_behaviors,
                latest_times,
            )
        )
        latest = sharded_append_padded_cache(
            sharded,
            delta.updated_cache,
            latest_ids,
            latest_behaviors,
            latest_times,
            suffix_lengths=torch.tensor([1]),
            retained_lengths=delta.lengths,
        )
        assert torch.equal(
            latest.updated_cache.k,
            expected_latest_cache.k,
        )
        assert torch.equal(
            latest.updated_cache.v,
            expected_latest_cache.v,
        )
        assert torch.equal(
            latest.last_hidden,
            expected_latest_hidden[:, -1],
        )
        assert latest.lengths.tolist() == [6]
        assert latest.lookup_metrics.requested_tokens == 1
    finally:
        dist.destroy_process_group()
