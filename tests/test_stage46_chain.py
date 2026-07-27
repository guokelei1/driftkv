import pytest
import torch

from hstu_kvcache.migration.cohort_jagged import JaggedMigratedKVBatch
from hstu_kvcache.migration.lifecycle import (
    CacheLifecycleState,
    LifecyclePolicy,
    MonotoneRiskCalibration,
)
from hstu_kvcache.migration.stage46_chain import (
    Stage46CostLedger,
    Stage46KVStore,
    absolute_log_norm_ratio_values,
    aggregate_layer_values,
    assemble_jagged_rows,
    pack_padded_cache,
    relative_cache_values,
    route_extent,
    select_jagged_rows,
    transition_sketch_values,
    unpack_jagged_cache,
)
from hstu_kvcache.models import HSTUKVCache


def batch() -> JaggedMigratedKVBatch:
    k = torch.arange(20, dtype=torch.float16).reshape(2, 5, 2)
    return JaggedMigratedKVBatch(
        record_ids=(0, 3),
        migration_anchor_version="theta0",
        served_kv_target="theta0",
        k=k,
        v=k + 1,
        lengths=torch.tensor([2, 3]),
        offsets=torch.tensor([0, 2, 5]),
    )


def policy(depth: int = 3, threshold: float = 1.0) -> LifecyclePolicy:
    return LifecyclePolicy(
        max_migration_depth=depth,
        risk_threshold=threshold,
        calibration=MonotoneRiskCalibration(
            correction_upper_bounds=(0.1, 1.0),
            one_hop_risks=(0.01, 0.2),
            propagation_gain=1.0,
            quantile=0.9,
        ),
    )


def test_pack_unpack_respects_lengths() -> None:
    k = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)
    cache = HSTUKVCache(k=k, v=k + 10, seq_len=3)
    packed = pack_padded_cache(
        cache,
        torch.tensor([2, 3]),
        (0, 3),
        "theta0",
        "theta0",
    )
    restored = unpack_jagged_cache(packed)
    assert restored.seq_len == 3
    assert torch.equal(restored.k[:, 0, :2], k[:, 0, :2])
    assert torch.count_nonzero(restored.k[:, 0, 2]) == 0
    assert torch.equal(restored.k[:, 1], k[:, 1])


def test_relative_values_are_per_record_and_layer() -> None:
    reference = batch()
    actual = JaggedMigratedKVBatch(
        record_ids=reference.record_ids,
        migration_anchor_version="theta0",
        served_kv_target="theta1",
        k=reference.k * 2,
        v=reference.v * 2,
        lengths=reference.lengths,
        offsets=reference.offsets,
    )
    values = relative_cache_values(actual, reference)
    assert values.shape == (2, 2)
    assert torch.allclose(values, torch.ones_like(values))
    assert aggregate_layer_values(values, 0.9).tolist() == pytest.approx(
        [1.0, 1.0]
    )
    sketch = transition_sketch_values(reference, actual)
    assert torch.allclose(
        sketch["relative_correction"],
        torch.ones_like(values),
    )
    assert torch.allclose(
        sketch["absolute_log_norm_ratio"],
        torch.full_like(values, torch.log(torch.tensor(2.0))),
        atol=1e-4,
    )
    assert torch.allclose(
        absolute_log_norm_ratio_values(reference, actual),
        sketch["absolute_log_norm_ratio"],
    )
    assert torch.allclose(
        sketch["cosine_distance"],
        torch.zeros_like(values),
        atol=2e-7,
    )


def test_store_consumes_written_previous_output() -> None:
    source = batch()
    store = Stage46KVStore.from_batch(source, 0)
    replacement = JaggedMigratedKVBatch(
        record_ids=source.record_ids,
        migration_anchor_version="theta0",
        served_kv_target="theta1",
        k=source.k + 7,
        v=source.v + 9,
        lengths=source.lengths,
        offsets=source.offsets,
    )
    store.write_records(0, replacement, (0, 1))
    store.advance_version(1)
    next_source = store.read_extent(0, 2)
    assert next_source.migration_anchor_version == "theta1"
    assert torch.equal(next_source.k, replacement.k)
    assert torch.equal(next_source.v, replacement.v)


def test_select_and_assemble_jagged_rows() -> None:
    source = batch()
    first = select_jagged_rows(source, (0,))
    second = select_jagged_rows(source, (1,))
    first = JaggedMigratedKVBatch(
        record_ids=first.record_ids,
        migration_anchor_version="theta0",
        served_kv_target="theta1",
        k=first.k + 4,
        v=first.v + 4,
        lengths=first.lengths,
        offsets=first.offsets,
    )
    second = JaggedMigratedKVBatch(
        record_ids=second.record_ids,
        migration_anchor_version="theta0",
        served_kv_target="theta1",
        k=second.k + 8,
        v=second.v + 8,
        lengths=second.lengths,
        offsets=second.offsets,
    )
    assembled = assemble_jagged_rows(source, (second, first), 1)
    assert assembled.record_ids == source.record_ids
    assert torch.equal(
        assembled.record_kv(0)[0],
        source.record_kv(0)[0] + 4,
    )
    assert torch.equal(
        assembled.record_kv(3)[1],
        source.record_kv(3)[1] + 8,
    )


def test_router_has_no_exact_reference_argument() -> None:
    source = batch()
    candidate = JaggedMigratedKVBatch(
        record_ids=source.record_ids,
        migration_anchor_version="theta0",
        served_kv_target="theta1",
        k=source.k.clone(),
        v=source.v.clone(),
        lengths=source.lengths,
        offsets=source.offsets,
    )
    routed = route_extent(
        policy(),
        (
            CacheLifecycleState.exact(0, 0),
            CacheLifecycleState.exact(3, 0),
        ),
        1,
        source,
        candidate,
        0.9,
    )
    assert [value.action for value in routed.decisions] == [
        "migrate",
        "migrate",
    ]


def test_exact_reference_is_excluded_from_mixed_cost() -> None:
    ledger = Stage46CostLedger(
        migration_ms=2,
        exact_refresh_ms=3,
        router_ms=1,
        publication_ms=4,
        all_exact_reference_ms=100,
        discarded_migration_records=2,
    )
    assert ledger.mixed_policy_ms == 10
    assert ledger.to_dict()["all_exact_reference_ms"] == 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fused_norm_ratio_matches_reference() -> None:
    device = torch.device("cuda:0")
    source = batch().to(device)
    candidate = JaggedMigratedKVBatch(
        record_ids=source.record_ids,
        migration_anchor_version="theta0",
        served_kv_target="theta1",
        k=(source.k * 1.25).contiguous(),
        v=(source.v * 0.75).contiguous(),
        lengths=source.lengths,
        offsets=source.offsets,
    )
    expected = transition_sketch_values(
        source,
        candidate,
    )["absolute_log_norm_ratio"]
    actual = absolute_log_norm_ratio_values(
        source,
        candidate,
        block=256,
    )
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
