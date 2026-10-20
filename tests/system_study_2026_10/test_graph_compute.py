import torch

from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.design_one.lightweight import LightweightValueState
from system_study_2026_10.graph_compute import GraphCompute


def test_shared_history_inputs_are_staged_without_modifying_user_state():
    graphs = GraphCompute(share_history_inputs=True)
    source = HSTUKVCache(torch.ones(2, 1, 7, 4), torch.full((2, 1, 7, 4), 2.), 7)
    state = LightweightValueState(torch.full((2, 1, 7, 6, 8), 3.), 7)
    original = [t.clone() for t in (source.k, source.v, state.codes)]
    def entry():
        return dict(operation='append', cache=HSTUKVCache(graphs._history_tensor('k', source.k),
            graphs._history_tensor('v', source.v), 7), state=LightweightValueState(
            graphs._history_tensor('code', state.codes), 7), ids=torch.empty(1, 2, dtype=torch.long),
            delta=torch.empty(1, 2), behavior=torch.empty(1, 2, dtype=torch.long), count=torch.tensor(-1))
    first, second = entry(), entry()
    assert first['cache'].k is second['cache'].k
    assert first['state'].codes is second['state'].codes
    assert first['cache'].k.data_ptr() != first['cache'].v.data_ptr()
    assert first['cache'].k.data_ptr() != source.k.data_ptr()
    ids, delta = torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2)
    graphs._stage(first, source, state, ids, delta, ids, 7)
    newer = HSTUKVCache(source.k+10, source.v+10, 7)
    zeros = LightweightValueState(torch.zeros_like(state.codes), 7)
    graphs._stage(second, newer, zeros, ids*2, delta*2, ids*2, 0)
    torch.testing.assert_close(first['cache'].k, newer.k, atol=0, rtol=0)
    assert not torch.count_nonzero(first['state'].codes)
    assert first['count'].item() == 7 and second['count'].item() == 0
    for user_tensor, saved in zip((source.k, source.v, state.codes), original):
        torch.testing.assert_close(user_tensor, saved, atol=0, rtol=0)
    assert len(graphs.history_inputs) == 3


def test_staging_pool_respects_clone_layout_and_dtype_default_does_not_share():
    source = torch.arange(32.).reshape(2, 1, 4, 4)
    graphs = GraphCompute(share_history_inputs=True)
    normal = graphs._history_tensor('k', source)
    transposed = graphs._history_tensor('k', source.transpose(-1, -2))
    assert normal.stride() == source.clone().stride()
    assert transposed.stride() == source.transpose(-1, -2).clone().stride()
    assert normal is not transposed
    assert graphs._history_tensor('k', source.double()).dtype == torch.float64
    sliced = source[:, :, :3]
    assert graphs._history_tensor('k', sliced).stride() == sliced.clone().stride()
    default = GraphCompute()
    first, second = default._history_tensor('k', source), default._history_tensor('k', source)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, source, atol=0, rtol=0)
