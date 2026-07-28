import pytest
import torch

from hstu_kvcache.migration import (
    JaggedMigratedKVBatch,
    append_jagged_suffix,
    drop_last_jagged_token,
    pack_padded_cache,
    plan_history_overlap,
    slice_jagged_token_ranges,
    tail_slice_jagged_cache,
)
from hstu_kvcache.models import HSTU, HSTUConfig


def tiny_model() -> HSTU:
    torch.manual_seed(11)
    model = HSTU(
        HSTUConfig(
            num_items=40,
            num_behaviors=4,
            hidden_size=16,
            num_layers=3,
            num_heads=2,
            head_dim=8,
            max_seq_len=16,
            input_dropout=0.0,
        )
    )
    model.eval()
    return model


def deterministic_cache(
    lengths: tuple[int, ...] = (2, 3, 1),
) -> JaggedMigratedKVBatch:
    tokens = sum(lengths)
    k = torch.arange(
        2 * tokens * 2,
        dtype=torch.float16,
    ).reshape(2, tokens, 2)
    length_tensor = torch.tensor(lengths)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), length_tensor.cumsum(0))
    )
    return JaggedMigratedKVBatch(
        record_ids=tuple(range(len(lengths))),
        migration_anchor_version="theta2",
        served_kv_target="theta2",
        k=k,
        v=(k + 100).contiguous(),
        lengths=length_tensor,
        offsets=offsets,
    )


def test_token_range_and_tail_slices_route_empty_rows() -> None:
    source = deterministic_cache()
    selected = slice_jagged_token_ranges(
        source,
        (1, 1, 1),
        (2, 3, 1),
    )
    assert selected.lengths == (1, 2, 0)
    assert selected.retained_rows == (0, 1)
    assert selected.empty_rows == (2,)
    assert selected.cache is not None
    assert selected.cache.record_ids == (0, 1)
    assert selected.cache.lengths.tolist() == [1, 2]
    assert torch.equal(
        selected.cache.record_kv(0)[0],
        source.record_kv(0)[0][:, 1:2],
    )
    assert torch.equal(
        selected.cache.record_kv(1)[1],
        source.record_kv(1)[1][:, 1:3],
    )

    tail = tail_slice_jagged_cache(source, (1, 2, 0))
    assert tail.starts == selected.starts
    assert tail.stops == selected.stops
    assert tail.empty_rows == (2,)
    assert tail.cache is not None
    assert torch.equal(tail.cache.k, selected.cache.k)
    assert torch.equal(tail.cache.v, selected.cache.v)

    empty = tail_slice_jagged_cache(source, (0, 0, 0))
    assert empty.cache is None
    assert empty.empty_rows == (0, 1, 2)


def test_token_slice_rejects_invalid_ranges() -> None:
    source = deterministic_cache()
    with pytest.raises(ValueError, match="count"):
        slice_jagged_token_ranges(source, (0,), (1,))
    with pytest.raises(ValueError, match="outside"):
        slice_jagged_token_ranges(source, (0, 0, 0), (3, 3, 1))
    with pytest.raises(ValueError, match="outside"):
        tail_slice_jagged_cache(source, (2, 4, 1))


def test_drop_last_produces_nonempty_prefix() -> None:
    source = deterministic_cache((2, 3))
    prefix = drop_last_jagged_token(source)
    assert prefix.lengths.tolist() == [1, 2]
    assert torch.equal(
        prefix.record_kv(0)[0],
        source.record_kv(0)[0][:, :-1],
    )
    assert torch.equal(
        prefix.record_kv(1)[1],
        source.record_kv(1)[1][:, :-1],
    )
    with pytest.raises(ValueError, match="empty prefix"):
        drop_last_jagged_token(deterministic_cache())


def test_variable_jagged_append_matches_per_record_full_forward() -> None:
    model = tiny_model()
    old_items = torch.tensor(
        [
            [1, 2, 0],
            [3, 4, 5],
            [6, 7, 8],
            [9, 0, 0],
        ]
    )
    old_behaviors = torch.tensor(
        [
            [1, 2, 0],
            [1, 2, 3],
            [2, 3, 1],
            [4, 0, 0],
        ]
    )
    old_deltas = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [0.0, 0.0, 0.0],
        ]
    )
    old_lengths = torch.tensor([2, 3, 3, 1])
    old_cache = model.compute_kv(
        old_items,
        old_behaviors,
        old_deltas,
        lengths=old_lengths,
    )
    packed = pack_padded_cache(
        old_cache,
        old_lengths,
        (10, 11, 12, 13),
        "theta0",
        "theta0",
    )
    sliced = tail_slice_jagged_cache(packed, (2, 3, 3, 0))
    suffix_items = torch.tensor(
        [
            [10, 11],
            [12, 0],
            [0, 0],
            [13, 14],
        ]
    )
    suffix_behaviors = torch.tensor(
        [
            [2, 3],
            [4, 0],
            [0, 0],
            [1, 2],
        ]
    )
    suffix_deltas = torch.tensor(
        [
            [6.0, 7.0],
            [8.0, 0.0],
            [0.0, 0.0],
            [9.0, 10.0],
        ]
    )
    suffix_lengths = torch.tensor([2, 1, 0, 2])
    result = append_jagged_suffix(
        model,
        sliced,
        suffix_items,
        suffix_behaviors,
        suffix_deltas,
        suffix_lengths,
    )
    assert result.cache.record_ids == (10, 11, 12, 13)
    assert result.cache.lengths.tolist() == [4, 4, 3, 2]
    assert result.cache.k.dtype == torch.float16
    assert result.appended_mask.tolist() == [True, True, False, True]
    assert result.last_appended_hidden is not None
    assert torch.count_nonzero(result.last_appended_hidden[2]) == 0

    expected_parts = (
        (
            torch.cat((old_items[0, :2], suffix_items[0, :2])),
            torch.cat((old_behaviors[0, :2], suffix_behaviors[0, :2])),
            torch.cat((old_deltas[0, :2], suffix_deltas[0, :2])),
        ),
        (
            torch.cat((old_items[1, :3], suffix_items[1, :1])),
            torch.cat((old_behaviors[1, :3], suffix_behaviors[1, :1])),
            torch.cat((old_deltas[1, :3], suffix_deltas[1, :1])),
        ),
        (
            old_items[2, :3],
            old_behaviors[2, :3],
            old_deltas[2, :3],
        ),
        (
            suffix_items[3, :2],
            suffix_behaviors[3, :2],
            suffix_deltas[3, :2],
        ),
    )
    for row, (items, behaviors, deltas) in enumerate(expected_parts):
        hidden, exact = model(
            items.unsqueeze(0),
            behaviors.unsqueeze(0),
            deltas.unsqueeze(0),
            return_kv=True,
            lengths=torch.tensor([len(items)]),
        )
        assert exact is not None
        actual_k, actual_v = result.cache.record_kv(10 + row)
        assert torch.allclose(
            actual_k,
            exact.k[:, 0].half(),
            atol=2e-3,
            rtol=2e-3,
        )
        assert torch.allclose(
            actual_v,
            exact.v[:, 0].half(),
            atol=2e-3,
            rtol=2e-3,
        )
        if suffix_lengths[row] > 0:
            assert torch.allclose(
                result.last_appended_hidden[row],
                hidden[0, -1],
                atol=3e-4,
                rtol=3e-4,
            )


def test_append_without_new_tokens_returns_no_hidden() -> None:
    model = tiny_model()
    items = torch.tensor([[1, 2], [3, 4]])
    behaviors = torch.ones_like(items)
    deltas = torch.zeros_like(items, dtype=torch.float32)
    lengths = torch.tensor([2, 2])
    cache = model.compute_kv(items, behaviors, deltas, lengths=lengths)
    packed = pack_padded_cache(
        cache,
        lengths,
        (0, 1),
        "theta0",
        "theta0",
    )
    sliced = tail_slice_jagged_cache(packed, (2, 2))
    result = append_jagged_suffix(
        model,
        sliced,
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((2, 0), dtype=torch.float32),
        torch.zeros(2, dtype=torch.long),
    )
    assert result.last_appended_hidden is None
    assert result.appended_mask.tolist() == [False, False]
    assert torch.equal(result.cache.k, packed.k)
    assert torch.equal(result.cache.v, packed.v)


def test_append_rejects_empty_result_and_bad_suffix() -> None:
    model = tiny_model()
    base_items = torch.tensor([[1], [2], [3]])
    base_behaviors = torch.ones_like(base_items)
    base_deltas = torch.zeros_like(base_items, dtype=torch.float32)
    base_lengths = torch.ones(3, dtype=torch.long)
    base_cache = model.compute_kv(
        base_items,
        base_behaviors,
        base_deltas,
        lengths=base_lengths,
    )
    source = pack_padded_cache(
        base_cache,
        base_lengths,
        (0, 1, 2),
        "theta0",
        "theta0",
    )
    sliced = tail_slice_jagged_cache(source, (1, 1, 0))
    items = torch.zeros((3, 1), dtype=torch.long)
    behaviors = torch.zeros_like(items)
    deltas = torch.zeros_like(items, dtype=torch.float32)
    with pytest.raises(ValueError, match="empty record"):
        append_jagged_suffix(
            model,
            sliced,
            items,
            behaviors,
            deltas,
            torch.zeros(3, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="outside"):
        append_jagged_suffix(
            model,
            sliced,
            items,
            behaviors,
            deltas,
            torch.tensor([0, 0, 2]),
        )


def test_strict_history_overlap_plan() -> None:
    plan = plan_history_overlap(
        [1, 2, 3, 4],
        [10, 20, 30, 40],
        [1, 2, 3, 4],
        [3, 4, 5],
        [30, 40, 50],
        [3, 4, 1],
    )
    assert plan.overlap_length == 2
    assert plan.evicted_tokens == 2
    assert plan.appended_tokens == 1
    assert plan.retained_old_start == 2
    assert plan.appended_new_start == 2

    duplicate = plan_history_overlap(
        [1, 2, 3, 4],
        [7, 8, 7, 8],
        [1, 1, 1, 1],
        [3, 4, 5],
        [7, 8, 9],
        [1, 1, 1],
    )
    assert duplicate.overlap_length == 2

    none = plan_history_overlap(
        torch.tensor([1, 2]),
        torch.tensor([4, 5]),
        torch.tensor([1, 1]),
        torch.tensor([3]),
        torch.tensor([5]),
        torch.tensor([2]),
    )
    assert none.overlap_length == 0
    assert none.evicted_tokens == 2
    assert none.appended_tokens == 1

    empty = plan_history_overlap(
        [1],
        [4],
        [1],
        [],
        [],
        [],
    )
    assert empty.new_prefix_length == 0
    assert empty.overlap_length == 0


def test_history_overlap_rejects_invalid_events() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        plan_history_overlap(
            [1, 2],
            [3],
            [1, 1],
            [3],
            [4],
            [1],
        )
    with pytest.raises(ValueError, match="nondecreasing"):
        plan_history_overlap(
            [2, 1],
            [3, 4],
            [1, 1],
            [3],
            [4],
            [1],
        )
    with pytest.raises(ValueError, match="integers"):
        plan_history_overlap(
            [1.5],
            [3],
            [1],
            [2],
            [4],
            [1],
        )
