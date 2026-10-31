"""Saved-product inversion and unchanged single-buffer serving semantics."""
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[4]

from hstu_kvcache.methods.cachetide.structure import StructuredKVAdapter
from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueState
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.methods.cachetide.adapter import UnifiedRelayAdapter
from hstu_kvcache.methods.cachetide.precision import StableReleasePrefix, stable_anchorize


def example():
    torch.manual_seed(20261015)
    source=StructuredKVAdapter(2,6,4,layout='shared12_private6').eval()
    with torch.no_grad():
        for layer in source.layers:
            layer.input_weight.normal_(std=.05)
            layer.input_bias.normal_(std=.02)
    actor=UnifiedRelayAdapter.from_structured(source,mode='A').eval()
    with torch.no_grad():
        for layer in actor.layers:
            for rotation in layer.rotation:
                rotation.normal_(std=.08)
    cache=HSTUKVCache(torch.randn(2,1,7,24),torch.randn(2,1,7,24),7)
    catalogue=torch.randn(13,24)
    items=torch.tensor([[1,2,3,4,5,6,7]])
    return actor,cache,catalogue,items


@torch.no_grad()
def test_wrap_shares_publication_tensors_and_adds_only_global_inverse():
    actor,cache,catalogue,items=example()
    prefix=actor.compile_release(None,catalogue)
    stable=StableReleasePrefix.wrap(prefix)
    for key in ('P','Q','D'):
        assert getattr(stable,key).data_ptr()==getattr(prefix,key).data_ptr()
    assert StableReleasePrefix.wrap(stable) is stable
    cost=stable.inverse_cost()
    assert cost['global_inverse_bytes']==2*(12**2+6*6**2)*4
    assert cost['preparation_flops_estimate']==2*((8*12**3)//3+6*(8*6**3)//3)
    assert cost['extra_per_row_flops']==cost['extra_per_row_persistent_bytes']==0
    assert stable.storage_bytes()==prefix.storage_bytes()+cost['global_inverse_bytes']
    current=actor.encode_initial(cache,catalogue[items])
    anchor=stable_anchorize(stable,cache,current,items)
    assert anchor.storage_bytes()==current.storage_bytes()
    torch.testing.assert_close(stable.materialize(cache,anchor,items).codes,current.codes,
                               atol=2e-6,rtol=2e-6)


@torch.no_grad()
def test_long_composition_actual_inverse_reduces_zero_birth_error():
    actor,cache,catalogue,items=example()
    prefix=None
    for _ in range(128):
        prefix=actor.compile_release(prefix,catalogue)
    stable=StableReleasePrefix.wrap(prefix)
    original_zero=prefix.materialize(cache,prefix.zero_anchors(cache,items),items)
    fixed_zero=stable.materialize(cache,stable.zero_anchors(cache,items),items)
    old=float(original_zero.codes.square().mean().sqrt())
    new=float(fixed_zero.codes.square().mean().sqrt())
    assert new<old*.15
    assert new>0  # Actual inversion does not eliminate FP32 offset cancellation.
    # A current-born zero row is updated exactly once in the next release.
    next_prefix=StableReleasePrefix.wrap(actor.compile_release(prefix,catalogue))
    next_direct=next_prefix.materialize(cache,stable.zero_anchors(cache,items),items)
    wanted=actor.advance_codes(cache,torch.zeros_like(next_direct.codes),catalogue[items])
    torch.testing.assert_close(next_direct.codes,wanted.codes,atol=3e-5,rtol=3e-5)


@torch.no_grad()
def test_subset_raw_ids_fifo_and_float64_roundtrip():
    actor,cache,catalogue,items=example()
    actor=actor.double()
    cache=HSTUKVCache(cache.k.double(),cache.v.double(),cache.seq_len)
    ids=items-1
    prefix=StableReleasePrefix.wrap(actor.compile_release(None,catalogue[1:8].double()))
    current=actor.encode_initial(cache,catalogue.double()[items])
    anchor=prefix.anchorize(cache,current,items,table_item_ids=ids)
    restored=prefix.materialize(cache,anchor,items,table_item_ids=ids)
    torch.testing.assert_close(restored.codes,current.codes,atol=1e-12,rtol=1e-12)
    retained=HSTUKVCache(cache.k[:,:,2:],cache.v[:,:,2:],5)
    held=LightweightValueState(anchor.codes[:,:,2:],5)
    result=prefix.materialize(retained,held,items[:,2:],table_item_ids=ids[:,2:])
    torch.testing.assert_close(result.codes,current.codes[:,:,2:],atol=1e-12,rtol=1e-12)
