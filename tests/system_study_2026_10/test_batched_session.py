from types import SimpleNamespace
import pytest
import torch

from system_study_2026_10.batched_session import BatchedSessions, grouped_uids, zero_tail_execution, append_code_buffers
from system_study_2026_10.batched_session_probe import frontiers
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.models import HSTUKVCache


def test_grouping_does_not_mix_lengths_or_duplicate_users():
    geometry = {1: (3, 1024, 900), 2: (3, 50, 50), 3: (3, 1024, 900),
                4: (3, 1024, 700), 5: (4, 1024, 900)}
    groups = grouped_uids(geometry, geometry.__getitem__, 2)
    assert groups == [[1, 3], [2], [4], [5]]
    assert sorted(u for group in groups for u in group) == sorted(geometry)


def test_frontier_keeps_each_users_order_and_release_boundary():
    def event(uid, timestamp, version="v3"):
        return SimpleNamespace(kind="recommendation", uid=uid, logical_time=timestamp, version=version)
    trace = [event(1, 10), event(2, 11), event(1, 12), event(2, 13), event(3, 20, "v4")]
    waves = list(frontiers(trace, "v3"))
    assert [[(e.uid, e.logical_time) for e in wave] for wave in waves] == [[(1, 10), (2, 11)], [(1, 12), (2, 13)]]


def test_zero_tail_charges_extra_pairs_and_reactivated_zero_code_queries():
    result = zero_tail_execution([100, 20, 0], 16)
    assert result["coded_rows"] == 100
    assert result["additional_zero_code_pairs"] == (300-120)*16
    assert result["additional_active_code_queries"] == 16
    assert result["additional_coalescing_flops"] > result["additional_code_pair_flops"] > 0
    assert zero_tail_execution([100, 100], 16)["additional_coalescing_flops"] == 0
    assert zero_tail_execution([0, 0], 16)["additional_coalescing_flops"] == 0


def test_single_user_graph_output_is_copied_before_next_replay():
    graph_output = HSTUKVCache(torch.ones(2, 1, 3, 4), torch.ones(2, 1, 3, 4)*2, 3)
    graph_output.transient_graph_output = True
    retained = BatchedSessions._slice_cache(graph_output, 0, 1)
    graph_output.k.zero_()
    graph_output.v.zero_()
    assert torch.all(retained.k == 1) and torch.all(retained.v == 2)
    assert retained.seq_len == 3
    assert BatchedSessions._slice_cache(retained, 0, 1) is retained


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA code-copy kernel")
def test_code_append_exact_shift_zero_suffix_and_independent_ownership():
    for batch, rows, width, cap in [(1, 5, 2, 1024), (1, 1024, 1, 1024),
                                   (3, 13, 4, 16), (8, 1024, 32, 1024)]:
        counts = [rows if i % 2 == 0 else 0 for i in range(batch)]
        states = []
        for count in counts:
            codes = torch.randn(6, 1, rows, 6, 8, device="cuda")
            codes[:, :, count:] = 0
            states.append(LightweightValueState(codes, rows))
        original = [s.codes.clone() for s in states]
        packed = LightweightValueState(torch.cat([s.codes for s in states], dim=1), rows)
        n = min(cap, rows+width)
        actual = append_code_buffers(packed, states, counts, width, n)
        pointers = []
        for i, result in enumerate(actual):
            zeros = original[i].new_zeros((6, 1, width, 6, 8))
            expected = torch.cat((original[i], zeros), 2)[:, :, -n:].contiguous()
            torch.testing.assert_close(result.codes, expected, atol=0, rtol=0)
            torch.testing.assert_close(states[i].codes, original[i], atol=0, rtol=0)
            pointers.append(result.codes.untyped_storage().data_ptr())
            assert result.codes.untyped_storage().nbytes() == result.codes.numel()*4
            if counts[i] == 0 and n == rows:
                assert result is states[i]
        assert len(set(pointers)) == batch
