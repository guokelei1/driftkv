"""CPU checks for padding, frozen-backbone backward, and source accounting."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
from types import SimpleNamespace

from hstu_kvcache.methods.cachetide.calibration_cost import (
    OperationCounts, corrected_query_flops, warmstart_origin_cost, update_forward_flops,
)
from hstu_kvcache.methods.cachetide.cost import origin_feature_bank_cost, row_update_flops


def geometry():
    branches = [SimpleNamespace(code_width=12, heads=tuple(range(6)))]
    branches += [SimpleNamespace(code_width=6, heads=(h,)) for h in range(6)]
    layer = SimpleNamespace(accessible_rank=6, code_width=48, item_width=192,
                            xi_width=3, branches=branches)
    return SimpleNamespace(mode="B", num_layers=6, heads=6, head_dim=32,
                            max_length=1024, layers=[layer] * 6)


def main():
    adapter = geometry()
    small, padded = OperationCounts(), OperationCounts()
    small.record_local(adapter, batch=2, padded_length=3, queries=4)
    padded.record_local(adapter, batch=2, padded_length=5, queries=4)
    assert sum(padded.components.values()) > sum(small.components.values())
    native, correction = corrected_query_flops(adapter, length=5, queries=4)
    joint = OperationCounts()
    joint.record_query(adapter, length=5, queries=4, backward=True)
    assert joint.components["query_frozen_backbone_backward_flops_estimate"] == 2 * native
    assert joint.components["query_correction_backward_flops_estimate"] == 2 * correction
    assert corrected_query_flops(adapter, length=5, queries=4, coded_rows=0)[1] == 0
    adapter.layers[0].gate_normalization = "rms"
    normalized = corrected_query_flops(adapter, length=5, queries=4)[1]
    assert normalized - correction == 6 * 4 * 6 * (4 * 32 + 2)
    assert corrected_query_flops(adapter, length=5, queries=4, coded_rows=0)[1] == 0
    source = dict(teacher_cache_capture_flops=10, source_trajectory_flops=20,
                  source_correction_flops=30)
    available = OperationCounts()
    available.record_native_capture(source, source_available=True)
    report = available.report()
    assert report["calibration_flops"] == 10
    assert report["source_reconstruction_sensitivity"]["additional_flops"] == 50
    required = OperationCounts()
    required.record_native_capture(source, source_available=False)
    report = required.report()
    assert report["calibration_flops"] == 60
    assert report["source_reconstruction_sensitivity"]["additional_flops"] == 0
    origin = warmstart_origin_cost(
        "results/code_structure_2026_10/shared12_private6/calibration/v3/calibration.pt")
    assert origin["calibration_flops"] == 9_694_771_272_324
    raw = geometry()
    raw.layers[0].xi_width = 1
    native_cost = update_forward_flops(raw, rows=10)
    raw.layers[0].feature_mode = "origin_encoder"
    assert update_forward_flops(raw, rows=10) < native_cost
    assert row_update_flops("A", 1, block_sparse=True, feature_mode="origin_encoder") \
        < row_update_flops("A", 1, block_sparse=True)
    bank = origin_feature_bank_cost(100, replicas=4)
    assert bank["preparation_flops"] == 4 * 6 * 100 * 2 * 192 * 48
    broadcast = origin_feature_bank_cost(100, replicas=4, preparation_mode="compile_once_broadcast")
    assert broadcast["preparation_flops"] * 4 == bank["preparation_flops"]
    assert broadcast["publication_transfer_bytes"] == 3 * bank["payload_bytes_per_replica"]
    counts = OperationCounts()
    counts.record_feature_bank_preparation(catalogue_items=100)
    report = counts.report()
    assert report["calibration_flops"] == 0
    assert report["immutable_feature_bank_preparation"]["actual_construction_flops"] > 0
    print("calibration padding/backward/source/origin checks passed")


if __name__ == "__main__":
    main()
