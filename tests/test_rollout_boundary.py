import hashlib

import torch

from hstu_kvcache.migration import (
    JaggedMigratedKVBatch,
    append_jagged_suffix,
    pack_padded_cache,
    plan_retained_prefix,
    retained_population_sha256,
    tail_slice_jagged_cache,
)
from hstu_kvcache.models import HSTU, HSTUConfig


def tiny_model() -> HSTU:
    torch.manual_seed(17)
    model = HSTU(
        HSTUConfig(
            num_items=48,
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


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _perturb(
    value: JaggedMigratedKVBatch,
    target_version: int,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=value.record_ids,
        migration_anchor_version=f"theta{target_version}",
        served_kv_target=f"theta{target_version}",
        k=(value.k + 0.01).contiguous(),
        v=(value.v - 0.02).contiguous(),
        lengths=value.lengths,
        offsets=value.offsets,
    )


def _append(
    model: HSTU,
    cache: JaggedMigratedKVBatch,
    items: list[int],
    behaviors: list[int],
) -> tuple[JaggedMigratedKVBatch, torch.Tensor]:
    result = append_jagged_suffix(
        model,
        tail_slice_jagged_cache(
            cache,
            tuple(int(value) for value in cache.lengths.tolist()),
        ),
        torch.tensor([items], dtype=torch.long),
        torch.tensor([behaviors], dtype=torch.long),
        torch.zeros((1, len(items)), dtype=torch.float32),
        torch.tensor([len(items)], dtype=torch.long),
        dtype=torch.float32,
    )
    assert result.last_appended_hidden is not None
    return result.cache, result.last_appended_hidden


def test_retained_prefix_plan_separates_overlap_delta_and_latest() -> None:
    reusable = plan_retained_prefix(
        7,
        11,
        ("A", "B", "C", "D"),
        ("C", "D", "E", "F"),
        "old",
        "target",
        True,
        True,
    )
    cold = plan_retained_prefix(
        8,
        12,
        ("A", "B", "C", "D"),
        ("C", "D", "E", "F"),
        "old",
        "target",
        False,
        False,
    )
    zero = plan_retained_prefix(
        9,
        13,
        ("A", "B"),
        ("C", "D"),
        "old",
        "target",
        True,
        True,
    )
    missing = plan_retained_prefix(
        10,
        14,
        ("A", "B", "C", "D"),
        ("C", "D", "E", "F"),
        "old",
        "target",
        True,
        False,
    )

    assert reusable.status == "reusable"
    assert reusable.retained_start == 2
    assert reusable.retained_tokens == 2
    assert reusable.delta_start == 2
    assert reusable.delta_tokens == 1
    assert reusable.latest_tokens == 1
    assert reusable.final_tokens == 4
    assert cold.status == "cold"
    assert cold.potential_overlap_tokens == 2
    assert cold.retained_tokens == 0
    assert cold.delta_tokens == 3
    assert not cold.migration_eligible
    assert zero.status == "zero_overlap"
    assert zero.retained_tokens == 0
    assert missing.status == "missing_cache"
    assert missing.retained_tokens == 2
    assert missing.delta_tokens == 1
    assert missing.missing_expected_cache
    assert missing.timed_retained_rebuild
    assert not missing.migration_eligible
    assert retained_population_sha256((reusable, cold, zero)) != (
        retained_population_sha256((reusable, cold, zero, missing))
    )
    assert retained_population_sha256((reusable, cold, zero, missing)) == (
        retained_population_sha256((zero, missing, reusable, cold))
    )


def test_exact_retained_then_target_append_matches_one_shot() -> None:
    model = tiny_model()
    retained_items = torch.tensor([[1, 2, 3], [4, 5, 0]])
    retained_behaviors = torch.tensor([[1, 2, 3], [2, 1, 0]])
    retained_deltas = torch.zeros((2, 3), dtype=torch.float32)
    retained_lengths = torch.tensor([3, 2])
    retained_cache = model.compute_kv(
        retained_items,
        retained_behaviors,
        retained_deltas,
        lengths=retained_lengths,
    )
    packed = pack_padded_cache(
        retained_cache,
        retained_lengths,
        (20, 21),
        "theta1",
        "theta1",
        dtype=torch.float32,
    )
    appended_items = torch.tensor([[6, 7], [8, 0]])
    appended_behaviors = torch.tensor([[2, 1], [3, 0]])
    appended_deltas = torch.zeros((2, 2), dtype=torch.float32)
    appended_lengths = torch.tensor([2, 1])
    two_stage = append_jagged_suffix(
        model,
        tail_slice_jagged_cache(packed, (3, 2)),
        appended_items,
        appended_behaviors,
        appended_deltas,
        appended_lengths,
        dtype=torch.float32,
    )
    full_items = torch.tensor([[1, 2, 3, 6, 7], [4, 5, 8, 0, 0]])
    full_behaviors = torch.tensor([[1, 2, 3, 2, 1], [2, 1, 3, 0, 0]])
    full_deltas = torch.zeros((2, 5), dtype=torch.float32)
    full_lengths = torch.tensor([5, 3])
    hidden, full_cache = model(
        full_items,
        full_behaviors,
        full_deltas,
        return_kv=True,
        lengths=full_lengths,
    )
    assert full_cache is not None
    full_packed = pack_padded_cache(
        full_cache,
        full_lengths,
        (20, 21),
        "theta1",
        "theta1",
        dtype=torch.float32,
    )
    full_hidden = model.last_hidden(hidden, full_lengths)
    assert two_stage.last_appended_hidden is not None

    assert torch.allclose(two_stage.cache.k, full_packed.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(two_stage.cache.v, full_packed.v, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        two_stage.last_appended_hidden,
        full_hidden,
        atol=1e-5,
        rtol=1e-5,
    )
    all_items = torch.arange(1, model.cfg.num_items)
    with torch.inference_mode():
        two_stage_scores = model.item_emb.score(
            two_stage.last_appended_hidden,
            all_items,
        )
        full_scores = model.item_emb.score(full_hidden, all_items)
    assert torch.allclose(two_stage_scores, full_scores, atol=1e-5, rtol=1e-5)
    assert torch.equal(
        torch.topk(two_stage_scores, k=10, dim=1).indices,
        torch.topk(full_scores, k=10, dim=1).indices,
    )


def test_crop_is_physical_tail_not_fresh_context() -> None:
    model = tiny_model()
    items = torch.tensor([[1, 2, 3, 4]])
    behaviors = torch.tensor([[1, 2, 3, 1]])
    deltas = torch.zeros((1, 4), dtype=torch.float32)
    lengths = torch.tensor([4])
    old_cache = model.compute_kv(items, behaviors, deltas, lengths=lengths)
    packed = pack_padded_cache(
        old_cache,
        lengths,
        (3,),
        "theta0",
        "theta0",
        dtype=torch.float32,
    )
    sliced = tail_slice_jagged_cache(packed, (2,))
    assert sliced.cache is not None
    old_k, old_v = packed.record_kv(3)
    assert torch.equal(sliced.cache.k, old_k[:, -2:])
    assert torch.equal(sliced.cache.v, old_v[:, -2:])
    fresh_cache = model.compute_kv(
        items[:, -2:],
        behaviors[:, -2:],
        deltas[:, -2:],
        lengths=torch.tensor([2]),
    )
    fresh = pack_padded_cache(
        fresh_cache,
        torch.tensor([2]),
        (3,),
        "theta0",
        "theta0",
        dtype=torch.float32,
    )
    assert not torch.allclose(sliced.cache.k[1:], fresh.k[1:])
    assert not torch.allclose(sliced.cache.v[1:], fresh.v[1:])


def test_two_edges_consume_previous_actual_mixed_cache() -> None:
    model = tiny_model()
    initial_items = torch.tensor([[1, 2, 3]])
    initial_behaviors = torch.tensor([[1, 2, 3]])
    initial_deltas = torch.zeros((1, 3), dtype=torch.float32)
    initial_cache = model.compute_kv(
        initial_items,
        initial_behaviors,
        initial_deltas,
        lengths=torch.tensor([3]),
    )
    cache = pack_padded_cache(
        initial_cache,
        torch.tensor([3]),
        (5,),
        "theta0",
        "theta0",
        dtype=torch.float32,
    )
    first_retained = tail_slice_jagged_cache(cache, (2,))
    assert first_retained.cache is not None
    first = _perturb(first_retained.cache, 1)
    first, _ = _append(model, first, [4], [1])
    first, _ = _append(model, first, [5], [2])
    first_hash = _tensor_hash(first.k)

    second_retained = tail_slice_jagged_cache(first, (2,))
    assert second_retained.cache is not None
    assert _tensor_hash(first.k) == first_hash
    assert torch.equal(
        second_retained.cache.k,
        first.record_kv(5)[0][:, -2:],
    )
    second = _perturb(second_retained.cache, 2)
    second, _ = _append(model, second, [6], [3])
    second, second_hidden = _append(model, second, [7], [1])
    exact_hidden, exact_cache = model(
        torch.tensor([[4, 5, 6, 7]]),
        torch.tensor([[1, 2, 3, 1]]),
        torch.zeros((1, 4), dtype=torch.float32),
        return_kv=True,
        lengths=torch.tensor([4]),
    )
    assert exact_cache is not None
    exact = pack_padded_cache(
        exact_cache,
        torch.tensor([4]),
        (5,),
        "theta2",
        "theta2",
        dtype=torch.float32,
    )

    assert second.lengths.tolist() == [4]
    assert second.migration_anchor_version == "theta2"
    assert torch.isfinite(second.k).all()
    assert torch.isfinite(second.v).all()
    assert torch.isfinite(second_hidden).all()
    assert not torch.allclose(second.k, exact.k)
    assert not torch.allclose(
        second_hidden,
        model.last_hidden(exact_hidden, torch.tensor([4])),
    )
