from pathlib import Path

import torch
import torch.distributed as dist

from hstu_kvcache.migration.design2_embedding import (
    build_modulo_sharded_hstu_from_cpu,
    sharded_append_padded_cache,
)
from hstu_kvcache.migration.design2_integrated import (
    build_integrated_exact_pool_schedule,
    build_integrated_schedule,
    integrated_exact_reason_counts,
    integrated_lookup_token_ledger,
    integrated_route,
    integrated_sharded_append,
    integrated_sharded_append_only,
    integrated_sharded_exact,
    materialize_integrated_append_only,
    select_integrated_records,
    slice_integrated_jagged_ranges,
)
from hstu_kvcache.migration.design2_plan import (
    D2ActionPlan,
    build_d2_record_owner_map,
)
from hstu_kvcache.migration.organic import slice_jagged_token_ranges
from hstu_kvcache.migration.recompute import RawHistoryBatch
from hstu_kvcache.migration.stage46_chain import (
    pack_padded_cache,
    unpack_jagged_cache,
)
from hstu_kvcache.models import HSTU, HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
ACTION_PLAN = (
    ROOT
    / "configs/cohortkv_d2"
    / "action_plan_theta1_theta2_staggered_renewal_h12.json"
)


def test_w3_pilot_selection_and_global_route_schedule() -> None:
    plan = D2ActionPlan.load(ACTION_PLAN)
    owner_map = build_d2_record_owner_map(
        plan,
        3,
        "strict_cow_lpt",
    )
    selected = select_integrated_records(
        plan,
        owner_map,
        3,
        "pilot192",
    )
    natural = tuple(
        value
        for value in plan.records
        if value.requested_reason == "natural_exact"
    )
    zero_delta = tuple(
        value
        for value in selected
        if value.delta_tokens == 0
    )
    assert all(value.retained_tokens == 0 for value in natural)
    assert all(
        value.previous_cache_present and value.old_tokens > 0
        for value in natural
    )
    assert sum(value.old_tokens for value in natural) == 82700
    assert [
        (
            value.record_id,
            integrated_route(value),
            owner_map[value.record_id],
        )
        for value in zero_delta
    ] == [(575, "compiled", 2)]
    assert integrated_lookup_token_ledger(
        plan.records,
        "staged",
    )["total"] == 347062
    assert integrated_lookup_token_ledger(
        plan.records,
        "fused_finalization",
    )["total"] == 347062
    assert integrated_lookup_token_ledger(
        selected,
        "staged",
    )["total"] == integrated_lookup_token_ledger(
        selected,
        "fused_finalization",
    )["total"]
    assert len(selected) == 192
    expected = {
        0: {"compiled": 51, "scheduled_exact": 5, "natural_exact": 8},
        1: {"compiled": 51, "scheduled_exact": 4, "natural_exact": 9},
        2: {"compiled": 51, "scheduled_exact": 4, "natural_exact": 9},
    }
    for rank in range(3):
        assert {
            route: sum(
                owner_map[value.record_id] == rank
                and integrated_route(value) == route
                for value in selected
            )
            for route in (
                "compiled",
                "scheduled_exact",
                "natural_exact",
            )
        } == expected[rank]
    route_schedule = build_integrated_schedule(
        selected,
        owner_map,
        3,
        16,
        route_major=True,
    )
    assert [
        (
            value.route,
            [len(record_ids) for record_ids in value.record_ids_by_rank],
        )
        for value in route_schedule
    ] == [
        ("compiled", [16, 16, 16]),
        ("compiled", [16, 16, 16]),
        ("compiled", [16, 16, 16]),
        ("compiled", [3, 3, 3]),
        ("scheduled_exact", [5, 4, 4]),
        ("natural_exact", [8, 9, 9]),
    ]
    all_schedule = build_integrated_schedule(
        selected,
        owner_map,
        3,
        16,
        route_major=False,
    )
    assert len(all_schedule) == 4
    assert all(
        [len(record_ids) for record_ids in value.record_ids_by_rank]
        == [16, 16, 16]
        for value in all_schedule
    )
    full_route_schedule = build_integrated_schedule(
        plan.records,
        owner_map,
        3,
        16,
        route_major=True,
    )
    scheduled_tail = next(
        value
        for value in full_route_schedule
        if value.route == "scheduled_exact" and value.ordinal == 1
    )
    assert [
        len(record_ids)
        for record_ids in scheduled_tail.record_ids_by_rank
    ] == [4, 0, 0]
    shape_schedule = build_integrated_schedule(
        selected,
        owner_map,
        3,
        16,
        route_major=True,
        compiled_order="suffix_retained",
    )
    selected_by_id = {value.record_id: value for value in selected}
    for extent in shape_schedule:
        if extent.route != "compiled":
            continue
        for record_ids in extent.record_ids_by_rank:
            actions = [selected_by_id[record_id] for record_id in record_ids]
            assert [
                (value.delta_tokens + value.latest_tokens, value.retained_tokens)
                for value in actions
            ] == sorted(
                (
                    value.delta_tokens + value.latest_tokens,
                    value.retained_tokens,
                )
                for value in actions
            )


def test_w3_exact_pool_preserves_owner_reasons_and_lookup_ledger() -> None:
    plan = D2ActionPlan.load(ACTION_PLAN)
    owner_map = build_d2_record_owner_map(
        plan,
        3,
        "strict_cow_lpt",
    )
    selected = select_integrated_records(
        plan,
        owner_map,
        3,
        "pilot192",
    )
    actions_by_id = {value.record_id: value for value in selected}
    exact_schedule = build_integrated_exact_pool_schedule(
        selected,
        owner_map,
        3,
        16,
    )
    assert len(exact_schedule) == 1
    assert [
        len(record_ids)
        for record_ids in exact_schedule[0].record_ids_by_rank
    ] == [13, 13, 13]
    expected_reason_counts = (
        {"scheduled_exact": 5, "natural_exact": 8},
        {"scheduled_exact": 4, "natural_exact": 9},
        {"scheduled_exact": 4, "natural_exact": 9},
    )
    assert tuple(
        integrated_exact_reason_counts(
            exact_schedule[0],
            rank,
            actions_by_id,
        )
        for rank in range(3)
    ) == expected_reason_counts
    observed_exact_ids = {
        record_id
        for extent in exact_schedule
        for record_ids in extent.record_ids_by_rank
        for record_id in record_ids
    }
    expected_exact_ids = {
        value.record_id
        for value in selected
        if integrated_route(value) != "compiled"
    }
    assert observed_exact_ids == expected_exact_ids
    assert all(
        owner_map[record_id] == rank
        for extent in exact_schedule
        for rank, record_ids in enumerate(extent.record_ids_by_rank)
        for record_id in record_ids
    )
    merged_tokens = sum(
        value.delta_tokens + value.latest_tokens
        for value in selected
        if integrated_route(value) == "compiled"
    ) + sum(
        value.final_tokens
        for value in selected
        if integrated_route(value) != "compiled"
    )
    assert merged_tokens == integrated_lookup_token_ledger(
        selected,
        "fused_finalization",
    )["total"]
    exact_schedule_shape_order = build_integrated_exact_pool_schedule(
        selected,
        owner_map,
        3,
        16,
    )
    for compiled_order in ("final_length", "suffix_retained"):
        route_schedule = build_integrated_schedule(
            selected,
            owner_map,
            3,
            16,
            route_major=True,
            compiled_order=compiled_order,
        )
        assert {
            record_id
            for extent in route_schedule
            if extent.route == "compiled"
            for record_ids in extent.record_ids_by_rank
            for record_id in record_ids
        } == {
            value.record_id
            for value in selected
            if integrated_route(value) == "compiled"
        }
        assert len(exact_schedule) < sum(
            extent.route != "compiled"
            for extent in route_schedule
        )
        assert (
            build_integrated_exact_pool_schedule(
                selected,
                owner_map,
                3,
                16,
            )
            == exact_schedule_shape_order
        )


def test_merged_exact_batch_matches_reason_separated_execution(
    tmp_path: Path,
) -> None:
    rendezvous = tmp_path / "integrated-merged-exact-gloo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        torch.manual_seed(17)
        full_model = HSTU(
            HSTUConfig(
                num_items=17,
                num_behaviors=4,
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
        scheduled = RawHistoryBatch(
            record_ids=(10,),
            migration_anchor_version="theta2",
            item_ids=torch.tensor([[1, 2, 3]]),
            behaviors=torch.tensor([[1, 2, 3]]),
            time_deltas=torch.tensor([[0.0, 1.0, 2.0]]),
            lengths=torch.tensor([3]),
        )
        natural = RawHistoryBatch(
            record_ids=(20,),
            migration_anchor_version="theta2",
            item_ids=torch.tensor([[4, 5, 0]]),
            behaviors=torch.tensor([[2, 1, 0]]),
            time_deltas=torch.tensor([[0.0, 1.0, 0.0]]),
            lengths=torch.tensor([2]),
        )
        merged = RawHistoryBatch(
            record_ids=(10, 20),
            migration_anchor_version="theta2",
            item_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
            behaviors=torch.tensor([[1, 2, 3], [2, 1, 0]]),
            time_deltas=torch.tensor(
                [[0.0, 1.0, 2.0], [0.0, 1.0, 0.0]]
            ),
            lengths=torch.tensor([3, 2]),
        )
        scheduled_exact = integrated_sharded_exact(
            sharded,
            scheduled,
            "theta2",
        )
        natural_exact = integrated_sharded_exact(
            sharded,
            natural,
            "theta2",
        )
        merged_exact = integrated_sharded_exact(
            sharded,
            merged,
            "theta2",
        )
        assert scheduled_exact.fragment is not None
        assert natural_exact.fragment is not None
        assert merged_exact.fragment is not None
        assert merged_exact.fragment.record_ids == (10, 20)
        assert merged_exact.fragment.lengths.tolist() == [3, 2]
        assert torch.allclose(
            merged_exact.fragment.k[:, :3],
            scheduled_exact.fragment.k,
        )
        assert torch.allclose(
            merged_exact.fragment.v[:, :3],
            scheduled_exact.fragment.v,
        )
        assert torch.allclose(
            merged_exact.fragment.k[:, 3:],
            natural_exact.fragment.k,
        )
        assert torch.allclose(
            merged_exact.fragment.v[:, 3:],
            natural_exact.fragment.v,
        )
        separated_ids = torch.cat(
            (
                scheduled.item_ids[0, : scheduled.lengths[0]],
                natural.item_ids[0, : natural.lengths[0]],
            )
        )
        merged_ids = torch.cat(
            (
                merged.item_ids[0, : merged.lengths[0]],
                merged.item_ids[1, : merged.lengths[1]],
            )
        )
        assert torch.equal(
            torch.sort(separated_ids).values,
            torch.sort(merged_ids).values,
        )
        assert (
            merged_exact.lookup_metrics.requested_tokens
            == scheduled_exact.lookup_metrics.requested_tokens
            + natural_exact.lookup_metrics.requested_tokens
        )
    finally:
        dist.destroy_process_group()


def test_integrated_vectorized_append_matches_row_reference(
    tmp_path: Path,
) -> None:
    rendezvous = tmp_path / "integrated-gloo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        torch.manual_seed(12)
        full_model = HSTU(
            HSTUConfig(
                num_items=17,
                num_behaviors=4,
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
        prefix = RawHistoryBatch(
            record_ids=(10, 20),
            migration_anchor_version="theta1",
            item_ids=torch.tensor([[1, 2, 3], [4, 0, 0]]),
            behaviors=torch.tensor([[1, 2, 3], [2, 0, 0]]),
            time_deltas=torch.tensor(
                [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]
            ),
            lengths=torch.tensor([3, 1]),
        )
        exact = integrated_sharded_exact(
            sharded,
            prefix,
            "theta1",
        )
        assert exact.fragment is not None
        suffix = RawHistoryBatch(
            record_ids=(10, 20),
            migration_anchor_version="theta2",
            item_ids=torch.tensor([[5, 6, 0], [7, 8, 9]]),
            behaviors=torch.tensor([[1, 2, 0], [3, 1, 2]]),
            time_deltas=torch.tensor(
                [[3.0, 4.0, 0.0], [1.0, 2.0, 3.0]]
            ),
            lengths=torch.tensor([2, 3]),
        )
        integrated = integrated_sharded_append(
            sharded,
            exact.fragment,
            suffix,
            "theta2",
        )
        assert integrated.fragment is not None
        reference = sharded_append_padded_cache(
            sharded,
            unpack_jagged_cache(exact.fragment, dtype=torch.float32),
            suffix.item_ids,
            suffix.behaviors,
            suffix.time_deltas,
            suffix.lengths,
            retained_lengths=exact.fragment.lengths,
        )
        reference_fragment = pack_padded_cache(
            reference.updated_cache,
            reference.lengths,
            suffix.record_ids,
            "theta2",
            "theta2",
        )
        assert integrated.fragment.record_ids == reference_fragment.record_ids
        assert integrated.fragment.lengths.tolist() == [5, 4]
        assert torch.allclose(
            integrated.fragment.k,
            reference_fragment.k,
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.allclose(
            integrated.fragment.v,
            reference_fragment.v,
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.allclose(
            integrated.last_hidden,
            reference.last_hidden,
            atol=1e-5,
            rtol=1e-5,
        )
        assert integrated.lookup_metrics.requested_tokens == 5
        append_only = integrated_sharded_append_only(
            sharded,
            exact.fragment,
            suffix,
            "theta2",
        )
        assert append_only.fragment is not None
        assert (
            append_only.fragment.retained.k.untyped_storage().data_ptr()
            == exact.fragment.k.untyped_storage().data_ptr()
        )
        assert (
            append_only.fragment.retained.v.untyped_storage().data_ptr()
            == exact.fragment.v.untyped_storage().data_ptr()
        )
        assert append_only.fragment.suffix.lengths.tolist() == [2, 3]
        assert append_only.fragment.lengths.tolist() == [5, 4]
        materialized_append_only = materialize_integrated_append_only(
            append_only.fragment
        )
        assert torch.allclose(
            materialized_append_only.k,
            integrated.fragment.k,
            atol=2e-2,
            rtol=2e-2,
        )
        assert torch.allclose(
            materialized_append_only.v,
            integrated.fragment.v,
            atol=2e-2,
            rtol=2e-2,
        )
        assert torch.allclose(
            append_only.last_hidden,
            integrated.last_hidden,
            atol=2e-5,
            rtol=2e-5,
        )
        delta_suffix = RawHistoryBatch(
            record_ids=(10, 20),
            migration_anchor_version="theta2",
            item_ids=torch.tensor([[5, 0], [7, 8]]),
            behaviors=torch.tensor([[1, 0], [3, 1]]),
            time_deltas=torch.tensor(
                [[3.0, 0.0], [1.0, 2.0]]
            ),
            lengths=torch.tensor([1, 2]),
        )
        latest_suffix = RawHistoryBatch(
            record_ids=(10, 20),
            migration_anchor_version="theta2",
            item_ids=torch.tensor([[6], [9]]),
            behaviors=torch.tensor([[2], [2]]),
            time_deltas=torch.tensor([[4.0], [3.0]]),
            lengths=torch.tensor([1, 1]),
        )
        staged_delta = integrated_sharded_append(
            sharded,
            exact.fragment,
            delta_suffix,
            "theta2",
        )
        staged_final = integrated_sharded_append(
            sharded,
            staged_delta.fragment,
            latest_suffix,
            "theta2",
        )
        assert staged_final.fragment is not None
        assert torch.allclose(
            staged_final.fragment.k,
            integrated.fragment.k,
            atol=2e-2,
            rtol=2e-2,
        )
        assert torch.allclose(
            staged_final.fragment.v,
            integrated.fragment.v,
            atol=2e-2,
            rtol=2e-2,
        )
        full = RawHistoryBatch(
            record_ids=(10, 20),
            migration_anchor_version="theta2",
            item_ids=torch.tensor(
                [[1, 2, 3, 5, 6], [4, 7, 8, 9, 0]]
            ),
            behaviors=torch.tensor(
                [[1, 2, 3, 1, 2], [2, 3, 1, 2, 0]]
            ),
            time_deltas=torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 0.0],
                ]
            ),
            lengths=torch.tensor([5, 4]),
        )
        one_shot = integrated_sharded_exact(
            sharded,
            full,
            "theta2",
        )
        assert one_shot.fragment is not None
        assert torch.allclose(
            staged_final.fragment.k,
            one_shot.fragment.k,
            atol=2e-2,
            rtol=2e-2,
        )
        assert torch.allclose(
            staged_final.fragment.v,
            one_shot.fragment.v,
            atol=2e-2,
            rtol=2e-2,
        )
        zero_suffix = RawHistoryBatch(
            record_ids=(10, 20),
            migration_anchor_version="theta2",
            item_ids=torch.tensor([[0, 0], [7, 8]]),
            behaviors=torch.tensor([[0, 0], [3, 1]]),
            time_deltas=torch.tensor(
                [[0.0, 0.0], [1.0, 2.0]]
            ),
            lengths=torch.tensor([0, 2]),
        )
        zero_appended = integrated_sharded_append(
            sharded,
            exact.fragment,
            zero_suffix,
            "theta2",
        )
        assert zero_appended.fragment is not None
        assert zero_appended.fragment.lengths.tolist() == [3, 3]
        assert zero_appended.lookup_metrics.requested_tokens == 2
        assert torch.count_nonzero(zero_appended.last_hidden[0]) == 0
        assert torch.equal(
            zero_appended.fragment.k[:, :3],
            exact.fragment.k[:, :3],
        )
        assert torch.equal(
            zero_appended.fragment.v[:, :3],
            exact.fragment.v[:, :3],
        )
        fast_slice = slice_integrated_jagged_ranges(
            exact.fragment,
            (1, 0),
            (3, 1),
        )
        reference_slice = slice_jagged_token_ranges(
            exact.fragment,
            (1, 0),
            (3, 1),
        ).cache
        assert reference_slice is not None
        assert torch.equal(fast_slice.k, reference_slice.k)
        assert torch.equal(fast_slice.v, reference_slice.v)
    finally:
        dist.destroy_process_group()
