"""Query coordinate folding and count-weighted publication references."""

import copy

import torch

from hstu_kvcache.adaptation import AdaptationState, QueryTranslator
from hstu_kvcache.adaptation.state import release_many
from hstu_kvcache.models import HSTUKVCache


def test_query_normalization_folds_into_raw_view_and_count_weighted_publication():
    torch.manual_seed(17)
    values = torch.randn(2, 1, 7, 4)
    state = AdaptationState(HSTUKVCache(values, values.sin(), 7), range(7), 0,
                           segment_size=8, slots=1, write_correction=False, history_scope="all")
    mapper = QueryTranslator(layers=2, width=4, heads=2, rank=3, target=1)
    mapper.encode.normal_(std=.1)
    mapper.decode.normal_(std=.1)
    mapper.offset.normal_(std=.1)
    mapper.center.normal_(std=.2)
    mapper.input_scale.uniform_(.3, 1.3)
    mapper.query_center.normal_()
    mapper.query_scale.uniform_(.4, 1.6)
    source = state.pack_source(1)
    features = mapper.features(source)
    x = (features-mapper.center)/mapper.input_scale
    latent = torch.einsum("i,lir->lr", x, mapper.encode)
    coefficients = (torch.einsum("lr,lro->lo", latent, mapper.decode)+mapper.offset).reshape(2, 3, 2, 2)
    query = torch.randn(2, 2, 5, 2)
    normalized_query = (query-mapper.query_center[:, :, None])/mapper.query_scale[:, :, None]
    expected = coefficients[:, 0, :, None]+normalized_query @ coefficients[:, 1:].transpose(1, 2)
    intercept, slopes = mapper.query_view(features)
    actual = intercept.reshape(2, 2, 1, 2)+query @ slopes
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    scalar, batched = copy.deepcopy(state), copy.deepcopy(state)
    scalar.release(1, mapper)
    release_many([batched], 1, mapper)
    paired = ((scalar.translated.payload[..., 1, :]-scalar.source.payload[..., 1, :])
              *scalar.source.count[..., None, None]).sum((0, 1))
    torch.testing.assert_close(batched.response_delta[0], paired, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(batched.response_query_delta[0], scalar.translated.query_coefficients*7,
                               atol=2e-6, rtol=2e-5)
    assert scalar.counts == batched.counts
    torch.testing.assert_close(batched.cache.k, state.cache.k, atol=0, rtol=0)
    features[-1] = 1.5
    cleared_intercept, cleared_slopes = mapper.query_view(features)
    assert torch.count_nonzero(cleared_intercept[0]) == torch.count_nonzero(cleared_slopes[0]) == 0
    assert torch.count_nonzero(cleared_slopes[1]) > 0


def test_native_noop_clears_old_view_but_preserves_next_release_lineage():
    torch.manual_seed(17)
    k,v=torch.randn(2,1,7,4),torch.randn(2,1,7,4)
    state=AdaptationState(HSTUKVCache(k,v,7),range(7),0,segment_size=8,slots=1,
                          write_correction=False,history_scope="all")
    first=QueryTranslator(layers=2,width=4,heads=2,rank=3,target=1)
    release_many([state],1,first)
    assert state.response_query_delta is not None
    release_many([state],2,None)
    assert state.translated is state.response_delta is state.response_query_delta is None
    assert not state.correction_active and state.covered
    torch.testing.assert_close(state.cache.k,k,atol=0,rtol=0)
    # Three actual new payloads under M2, after evicting the first two events.
    new_k,new_v=torch.randn(2,1,3,4),torch.randn(2,1,3,4)
    retained=HSTUKVCache(torch.cat((k[:,:,2:],new_k),2),torch.cat((v[:,:,2:],new_v),2),8)
    state.install_native_chunk(retained,[7,8,9])
    third=QueryTranslator(layers=2,width=4,heads=2,rank=3,target=3)
    release_many([state],3,third)
    source=state.pack_source()
    assert source.producer.tolist()==[0,2]
    assert source.count.sum(1).tolist()==[5,3]
    assert state.target==3 and state.writes_since_release==0
    torch.testing.assert_close(state.cache.k,retained.k,atol=0,rtol=0)
