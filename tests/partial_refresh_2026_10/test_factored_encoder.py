"""Factorization preserves code values while charging real misses/evictions."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))

from hstu_kvcache.design_one.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.models import HSTUKVCache
from partial_refresh_2026_10.round2.factored_encoder import FactoredEncoder


class ItemModel:
    def __init__(self, width):
        self.table=torch.randn(40,width)
        self.looked_up=[]

    def lookup_item_embeddings(self, ids):
        self.looked_up.append(ids.clone())
        return self.table[ids]


def setup(layers=3,heads=2,dim=4,rank=2):
    torch.manual_seed(20261014)
    adapter=LightweightFullItemKVAdapter(layers,heads,dim,rank=rank).eval().requires_grad_(False)
    with torch.no_grad():
        for layer in adapter.layers:
            layer.input_weight.normal_(std=.15)
            layer.input_bias.normal_(std=.2)
            layer.input_center.normal_(std=.3)
            layer.input_scale.uniform_(.6,1.6)
    model=ItemModel(heads*dim)
    def cache(length):
        return HSTUKVCache(torch.randn(layers,1,length,heads*dim),
                           torch.randn(layers,1,length,heads*dim),length)
    return adapter,model,cache


@torch.inference_mode()
def test_nontrivial_folded_encoder_matches_all_heads_and_reuses_items():
    adapter,model,make=setup(layers=6,heads=6,dim=32,rank=8)
    encoder=FactoredEncoder(adapter,model,capacity=16)
    ids=np.array([5,2,5,7,2,9])
    cache=make(len(ids))
    before_k,before_v=cache.k.clone(),cache.v.clone()
    actual=encoder.encode(cache,ids)
    first=encoder.last_delta
    assert model.looked_up[0].numel()==4
    expected=adapter.encode_cache(cache,model.table[torch.tensor(ids)[None]])
    torch.testing.assert_close(actual.codes,expected.codes,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(cache.k,before_k,atol=0,rtol=0)
    torch.testing.assert_close(cache.v,before_v,atol=0,rtol=0)
    assert actual.codes.shape==(6,1,6,6,8)
    assert first['parameter_preparation_flops']==221184
    assert first['row_kv_flops']==len(ids)*37440
    assert first['item_projection_flops']==4*110880
    second_cache=make(4)
    second_ids=np.array([9,2,10,10])
    result=encoder.encode(second_cache,second_ids)
    second=encoder.last_delta
    torch.testing.assert_close(result.codes,adapter.encode_cache(
        second_cache,model.table[torch.tensor(second_ids)[None]]).codes,atol=2e-6,rtol=2e-5)
    assert model.looked_up[1].tolist()==[[10]]
    assert 'parameter_preparation_flops' not in second
    assert second['unique_item_misses']==1 and second['unique_item_hits']==2
    assert encoder.memory['shared_item_table_peak_bytes']==16*1152


@pytest.mark.parametrize('capacity',[0,2,None])
@torch.inference_mode()
def test_batched_cache_eviction_and_item_release_binding(capacity):
    adapter,model,make=setup()
    encoder=FactoredEncoder(adapter,model,capacity=capacity)
    for ids in ([1,2,3,4,1],[1,4,6,2],[6,2,8,9,2]):
        cache=make(len(ids))
        actual=encoder.encode(cache,ids)
        expected=adapter.encode_cache(cache,model.table[torch.tensor(ids)[None]])
        torch.testing.assert_close(actual.codes,expected.codes,atol=1e-6,rtol=1e-5)
        if capacity is not None:
            assert len(encoder.slots)<=capacity and encoder.table.shape[0]<=capacity
    if capacity==2:
        assert list(encoder.slots)==[9,2]
        assert encoder.counters['item_cache_evictions']>0
    # A new release has a separate table and uses its new Current embedding.
    current=ItemModel(adapter.heads*adapter.head_dim)
    current.table.add_(.5)
    other=FactoredEncoder(adapter,current,capacity=2)
    cache=make(2);actual=other.encode(cache,[2,9]);cost=other.last_delta
    assert cost['unique_item_misses']==2
    torch.testing.assert_close(actual.codes,adapter.encode_cache(
        cache,current.table[torch.tensor([[2,9]])]).codes,atol=1e-6,rtol=1e-5)


@torch.inference_mode()
def test_static_item_table_is_charged_bounded_and_order_independent():
    adapter,model,make=setup()
    encoder=FactoredEncoder(adapter,model,static_ids=[5,2])
    width=adapter.num_layers*adapter.heads*adapter.rank
    item_cost=(2*adapter.heads*adapter.head_dim+1)*width
    assert encoder.counters['static_precomputed_items']==2
    assert encoder.counters['item_projection_flops']==2*item_cost
    assert model.looked_up[0].tolist()==[[2,5]]
    requests=[[2,5,8,8,9],[8,2]]
    caches=[make(len(ids)) for ids in requests]
    for cache,ids in zip(caches,requests):
        actual=encoder.encode(cache,ids)
        torch.testing.assert_close(actual.codes,adapter.encode_cache(
            cache,model.table[torch.tensor(ids)[None]]).codes,atol=1e-6,rtol=1e-5)
    assert model.looked_up[1].tolist()==[[8,9]]
    assert model.looked_up[2].tolist()==[[8]]  # Uncached 8 was not admitted.
    assert set(encoder.slots)=={2,5} and encoder.table.shape[0]==2
    assert encoder.counters['item_projection_flops']==5*item_cost
    reverse=FactoredEncoder(adapter,model,static_ids=[5,2])
    for cache,ids in reversed(list(zip(caches,requests))):
        reverse.encode(cache,ids)
    assert reverse.counters['item_projection_flops']==encoder.counters['item_projection_flops']
    assert reverse.counters['row_kv_flops']==encoder.counters['row_kv_flops']
