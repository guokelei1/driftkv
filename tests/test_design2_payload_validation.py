import torch

from hstu_kvcache.migration.cohort_jagged import JaggedMigratedKVBatch
from hstu_kvcache.migration.design2_integrated import (
    IntegratedAppendOnlyKVBatch,
)
from hstu_kvcache.migration.design2_payload_validation import (
    D2PayloadComparisonAccumulator,
    compare_jagged_payloads,
    compare_jagged_to_append_only,
)


def _jagged(
    record_ids: tuple[int, ...],
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=record_ids,
        migration_anchor_version="theta2",
        served_kv_target="theta2",
        k=k,
        v=v,
        lengths=lengths,
        offsets=torch.cat(
            (torch.zeros(1, dtype=torch.long), lengths.cumsum(0))
        ),
    )


def test_full_payload_comparison_materializes_segmented_records() -> None:
    retained = _jagged(
        (10, 20),
        torch.arange(20, dtype=torch.float16).reshape(1, 5, 4),
        torch.arange(20, 40, dtype=torch.float16).reshape(1, 5, 4),
        torch.tensor([2, 3]),
    )
    suffix = _jagged(
        (10, 20),
        torch.arange(40, 52, dtype=torch.float16).reshape(1, 3, 4),
        torch.arange(52, 64, dtype=torch.float16).reshape(1, 3, 4),
        torch.tensor([1, 2]),
    )
    segmented = IntegratedAppendOnlyKVBatch(
        retained=retained,
        suffix=suffix,
        lengths=torch.tensor([3, 5]),
        offsets=torch.tensor([0, 3, 8]),
        target_version="theta2",
    )
    contiguous = _jagged(
        (10, 20),
        torch.cat(
            (
                torch.cat((retained.k[:, :2], suffix.k[:, :1]), dim=1),
                torch.cat((retained.k[:, 2:], suffix.k[:, 1:]), dim=1),
            ),
            dim=1,
        ),
        torch.cat(
            (
                torch.cat((retained.v[:, :2], suffix.v[:, :1]), dim=1),
                torch.cat((retained.v[:, 2:], suffix.v[:, 1:]), dim=1),
            ),
            dim=1,
        ),
        torch.tensor([3, 5]),
    )
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    accumulator = D2PayloadComparisonAccumulator()
    records = compare_jagged_to_append_only(
        accumulator,
        route="compiled",
        left=contiguous,
        left_last_hidden=hidden,
        right=segmented,
        right_last_hidden=hidden.clone(),
    )
    report = accumulator.report()
    assert len(records) == 2
    assert report["records"] == 2
    assert report["tokens"] == 8
    assert report["elements"] == 72
    assert report["allclose"]
    assert report["bitwise_equal"]
    assert report["hashes_match"]
    assert report["record_counts_by_route"] == {"compiled": 2}
    assert report["token_counts_by_route"] == {"compiled": 8}


def test_full_payload_comparison_reports_all_elements_and_error() -> None:
    left = _jagged(
        (7,),
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
        torch.tensor([2]),
    )
    right = _jagged(
        (7,),
        left.k.clone(),
        left.v.clone(),
        torch.tensor([2]),
    )
    right.k[0, 1, 1] += 0.5
    left_hidden = torch.tensor([[9.0, 10.0]])
    right_hidden = left_hidden.clone()
    accumulator = D2PayloadComparisonAccumulator(
        kv_atol=1e-4,
        kv_rtol=1e-4,
    )
    compare_jagged_payloads(
        accumulator,
        route="natural_exact",
        left=left,
        left_last_hidden=left_hidden,
        right=right,
        right_last_hidden=right_hidden,
    )
    report = accumulator.report()
    assert report["records"] == 1
    assert report["tokens"] == 2
    assert report["elements"] == 10
    assert not report["allclose"]
    assert not report["bitwise_equal"]
    assert not report["hashes_match"]
    assert report["failed_record_ids"] == [7]
    assert report["components"]["k"]["max_absolute_error"] == 0.5
    assert report["components"]["k"]["mean_absolute_error"] == 0.125
    assert report["mean_absolute_error"] == 0.05
