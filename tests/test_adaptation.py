"""Small tensor references for event membership and paired-summary algebra."""

import copy
from types import SimpleNamespace

import torch

from hstu_kvcache.adaptation import (
    AdaptationState,
    FunctionalTranslator,
    RidgeTranslator,
    Translator,
)
from hstu_kvcache.adaptation.reader import history_read, summary_read
from hstu_kvcache.adaptation.state import release_many
from hstu_kvcache.adaptation.summary import SummaryWriter
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import retain_latest_cache


def test_writer_matches_retained_events_across_partial_eviction_and_release():
    values = torch.arange(6 * 97 * 4, dtype=torch.float32).reshape(6, 1, 97, 4) / 97
    cache = HSTUKVCache(values, values.sin(), 97)
    writer = SummaryWriter.from_cache(cache, range(97), 0, second_moments=True)
    writer.close_segment()
    new = HSTUKVCache(values[:, :, :7] + 1, values[:, :, :7].cos(), 7)
    writer.add(new, range(97, 104), 1)
    cache = HSTUKVCache(torch.cat((cache.k, new.k), 2), torch.cat((cache.v, new.v), 2), 104)
    for _ in range(45):
        writer.evict_first(cache)
        cache = retain_latest_cache(cache, cache.seq_len - 1)
    for target in (1, 2):
        source = writer.pack(target)
        reference = writer.reference_summary(cache, target)
        torch.testing.assert_close(source.payload, reference.payload, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(source.second_moment, reference.second_moment, atol=2e-4, rtol=2e-5)
    assert writer.pack(2).producer.tolist() == [0, 0, 1]
    assert writer.pack(1).count.sum().item() == 52
    assert list(writer.events)[0][2:] == (45, 45)


def test_count_weighted_read_equals_repeated_slots_and_identity_cancels():
    torch.manual_seed(17)
    attention = SimpleNamespace(position_bias=None, num_heads=2, head_dim=3, scale=1.,
        _activate=lambda x: torch.nn.functional.elu(x) + 1,
        block_variant="legacy", attn_dropout=lambda x: x)
    k, v = torch.randn(1, 1, 3, 6).repeat_interleave(32, dim=2), torch.randn(1, 1, 3, 6).repeat_interleave(32, dim=2)
    cache = HSTUKVCache(k, v, 96)
    source = SummaryWriter.from_cache(cache, range(96), 0).pack(1)
    q = torch.randn(1, 2, 5, 3, requires_grad=True)
    full = history_read(attention, q, k[0, 0], v[0, 0])
    compressed = summary_read(attention, q, source, 0)
    torch.testing.assert_close(full, compressed, atol=5e-5, rtol=2e-5)
    assert torch.equal(compressed - summary_read(attention, q, source, 0), torch.zeros_like(compressed))
    compressed.square().mean().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


def test_local_refresh_equals_full_translation_after_partial_and_whole_segment_eviction():
    torch.manual_seed(17)
    values = torch.randn(6, 1, 97, 4)
    state = AdaptationState(HSTUKVCache(values, values.cos(), 97), range(97), 0)
    translator = Translator(layers=6, width=4, hidden=8, context="local", target=1)
    torch.nn.init.normal_(translator.decoder.weight, std=0.02)
    state.release(1, translator)
    for _ in range(65):
        sid = state.writer.events[0][0]
        state.writer.evict_first(state.cache)
        state.cache = retain_latest_cache(state.cache, state.cache.seq_len - 1)
        state.refresh(sid)
        with torch.no_grad():
            reference = translator(state.writer.pack(1))
        torch.testing.assert_close(state.translated.payload, reference.payload, rtol=1e-5, atol=1e-6)
    # One initial 2-segment conversion, then one dirty segment except at deletion.
    assert state.counts["translated_segments"] == 66


def test_functional_translation_is_count_weighted_zero_key_paired_read():
    torch.manual_seed(17)
    values = torch.randn(6, 1, 97, 4)
    source = SummaryWriter.from_cache(HSTUKVCache(values, values.cos(), 97), range(97), 0).pack(1)
    translator = FunctionalTranslator(layers=6, width=4, hidden=8)
    torch.nn.init.constant_(translator.decoder.bias, 0.25)
    translated = translator(source)
    attention = SimpleNamespace(position_bias=None, num_heads=2, head_dim=2, scale=1.,
        _activate=lambda x: torch.nn.functional.elu(x) + 1,
        block_variant="legacy", attn_dropout=lambda x: x)
    q = torch.randn(1, 2, 5, 2)
    k = torch.zeros(4, 4)
    old_read = history_read(attention, q, k, source.payload[:, :, 0, 1].flatten(0, 1), count=source.count.flatten())
    new_read = history_read(attention, q, k, translated.payload[:, :, 0, 1].flatten(0, 1), count=source.count.flatten())
    torch.testing.assert_close(new_read-old_read, torch.full_like(old_read, 0.25), atol=1e-5, rtol=1e-5)


def test_shared_ridge_matches_analytic_one_direction_shrinkage():
    model = RidgeTranslator(layers=6,width=4,rank=4)
    n = 20
    z = torch.linspace(-1,1,n)
    x = torch.zeros(n,49)
    x[:,0],x[:,-1] = z,1
    slopes = torch.linspace(-0.3,0.3,24).reshape(6,4)
    intercept = torch.linspace(0.1,0.4,24).reshape(6,4)
    y = z[:,None,None]*slopes+intercept
    model.fit(x,y)
    predicted = ((x-model.center)/model.input_scale) @ model.encode @ model.decode+model.offset
    expected = z[:,None,None]*slopes*(n/(n+0.01))+intercept
    torch.testing.assert_close(predicted.reshape_as(y),expected,rtol=2e-5,atol=2e-6)


def test_bulk_replay_preserves_open_segment_membership_after_prefix_expires():
    # The initial partial segment is filled by new writes before its old rows
    # expire. Bulk removal must not reset that segment's write coordinate.
    values = torch.arange(2*13*3,dtype=torch.float32).reshape(2,1,13,3)
    initial = HSTUKVCache(values[:,:,:5],values[:,:,:5].cos(),5)
    sequential = SummaryWriter.from_cache(initial,range(5),0,segment_size=8,slots=2,second_moments=True)
    bulk = SummaryWriter.from_cache(initial,range(5),0,segment_size=8,slots=2,second_moments=True)
    cache = initial
    for position in range(5,13):
        if cache.seq_len==8:
            sequential.evict_first(cache)
            cache = retain_latest_cache(cache,7)
        new = HSTUKVCache(values[:,:,position:position+1],values[:,:,position:position+1].cos(),1)
        sequential.add(new,[position],0)
        cache = HSTUKVCache(torch.cat((cache.k,new.k),2),torch.cat((cache.v,new.v),2),cache.seq_len+1)
    bulk.evict_prefix(initial,5)
    bulk.add(HSTUKVCache(values[:,:,5:],values[:,:,5:].cos(),8),range(5,13),0)
    assert list(bulk.events)==list(sequential.events)
    a,b=bulk.pack(1),sequential.pack(1)
    assert a.segment_ids==b.segment_ids
    torch.testing.assert_close(a.count,b.count)
    torch.testing.assert_close(a.payload,b.payload,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(a.second_moment,b.second_moment,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(a.ordinal,b.ordinal)
    torch.testing.assert_close(a.timestamp,b.timestamp)


def test_producer_variance_features_match_actual_retained_tokens_and_batched_writer():
    values = torch.linspace(-2,3,6*13*4).reshape(6,1,13,4)
    cache = HSTUKVCache(values,values.square(),13)
    writer = SummaryWriter.from_cache(cache,range(13),0,segment_size=8,slots=1,second_moments=True)
    writer.close_segment()
    tail = HSTUKVCache(values[:,:,:5]+3,values[:,:,:5].cos(),5)
    writer.add(tail,range(13,18),1)
    writer.evict_prefix(cache,7)
    source = writer.pack(2,include_current=True)
    mapper = RidgeTranslator(layers=6,width=4,target=2,history_scope="all",second_moments=True)
    actual = mapper.features(source)
    expected = []
    for k, v in ((values[:,0,7:],values[:,0,7:].square()), (tail.k[:,0],tail.v[:,0])):
        tokens = torch.stack((k,v),dim=2)
        mass = tokens.shape[1]/11
        expected.extend(((tokens.mean(1)*mass).flatten(),(tokens.var(1,correction=0)*mass).flatten(),torch.tensor([mass])))
    expected.extend((torch.zeros(96),torch.zeros(1),torch.zeros(1)))
    torch.testing.assert_close(actual,torch.cat(expected),atol=2e-5,rtol=2e-5)
    packed,count = mapper.writer_features([writer],[0])
    torch.testing.assert_close(packed[0],actual,atol=2e-5,rtol=2e-5)
    assert count.item() == 11


def test_full_history_source_retains_current_descendants_after_old_producer_expires():
    values = torch.arange(6*4*4,dtype=torch.float32).reshape(6,1,4,4)/100
    state = AdaptationState(HSTUKVCache(values,values.cos(),4),range(4),0,
                            segment_size=8,slots=1,write_correction=False,history_scope="all")
    translator = RidgeTranslator(layers=6,width=4,target=1,history_scope="all")
    state.release(1,translator)
    state.writer.evict_prefix(state.cache,4)
    state.writer.add(HSTUKVCache(values+1,values.sin(),4),range(4,8),1)
    state.cache = HSTUKVCache(values+1,values.sin(),4)
    state.refresh()
    assert state.writer.pack(1) is None
    assert state.source.producer.tolist() == [1]
    assert state.source.release_age == 4
    torch.testing.assert_close(state.source.payload[0,0,:,0],(values+1)[:,0].mean(1))
    torch.testing.assert_close(state.source.payload[0,0,:,1],values[:,0].sin().mean(1))
    assert state.response_mask(1).all() and state.covered
    other = AdaptationState(HSTUKVCache(values[:,:,:3],values[:,:,:3].cos(),3),range(3),0,
                            segment_size=8,slots=1,write_correction=False,history_scope="all")
    translator = RidgeTranslator(layers=6,width=4,target=2,rank=4,history_scope="all")
    translator.offset.copy_(torch.linspace(-.1,.1,24))
    translator.encode.copy_(torch.linspace(-.01,.01,translator.encode.numel()).reshape_as(translator.encode))
    translator.decode.fill_(.03)
    scalar = [copy.deepcopy(state),copy.deepcopy(other)]
    batch = [copy.deepcopy(state),copy.deepcopy(other)]
    for item in scalar:
        item.release(2,translator)
    release_many(batch,2,translator)
    for a,b in zip(scalar,batch,strict=True):
        expected=((a.translated.payload[...,1,:]-a.source.payload[...,1,:])*a.source.count[...,None,None]).sum((0,1))
        torch.testing.assert_close(b.response_delta[0],expected,atol=1e-6,rtol=1e-6)
        assert b.correction_active and b.counts == a.counts
        assert b.release_ordinal == a.release_ordinal
        torch.testing.assert_close(a.cache.k,b.cache.k)


def test_temporal_rate_factorization_keeps_centering_and_producer_conditioning():
    torch.manual_seed(17)
    translator=RidgeTranslator(layers=6,width=4,target=1,rank=4,history_scope="all",temporal=True,
                               layer_clearance=True,temporal_source_rank=8)
    source=torch.randn(20,translator.base_inputs)
    source[:,48]=torch.linspace(.1,.9,20)
    source[:,97]=1-source[:,48]
    source[:,-1]=torch.linspace(0,6,20)
    translator.fit_time_source(source)
    latent=translator.time_condition(source)[:,translator.producer_count:]
    torch.testing.assert_close(latent.T @ latent/19,torch.eye(8),atol=2e-5,rtol=2e-5)
    phi=torch.randn(20,32)
    features=translator.with_time(source,phi)
    translator.fit(features,torch.randn(20,6,4)*.1)
    direct=translator.rates(features)
    factored=translator.rates(source)+torch.einsum("nt,nltw->nlw",phi,translator.time_coefficients(source))
    torch.testing.assert_close(factored,direct,atol=2e-6,rtol=2e-5)
    mask=translator.active_layers(source)
    assert torch.count_nonzero(direct.masked_select(~mask[...,None])) == 0
    assert mask[0].all() and not mask[-1].any()
    strength=torch.tensor([0., .2, .4, .6, .8, 1.])
    translator.response_strength.copy_(strength)
    scaled=translator.rates(source)+torch.einsum("nt,nltw->nlw",phi,translator.time_coefficients(source))
    torch.testing.assert_close(scaled,direct*strength[None,:,None],atol=2e-6,rtol=2e-5)

    confidence=RidgeTranslator(layers=6,width=4,target=1,rank=4,history_scope="all",temporal=True,
                               layer_clearance=True,temporal_source_rank=8,source_confidence=True)
    confidence.fit_time_source(source)
    confidence.fit(features,torch.randn(20,6,4)*.1)
    # A larger source must reduce both terms together, independently of time.
    outside=confidence.center[:confidence.base_inputs]+8*confidence.input_scale[:confidence.base_inputs]
    outside[-1]=0
    factor=confidence.confidence_strength(outside)
    assert 0 < factor < 1
    joint=confidence.with_time(outside,phi[0])
    torch.testing.assert_close(confidence.confidence_strength(joint),factor)
    factored=confidence.rates(outside)+torch.einsum("t,ltw->lw",phi[0],confidence.time_coefficients(outside))
    torch.testing.assert_close(factored,confidence.rates(joint),atol=2e-5,rtol=2e-5)
    confidence.source_confidence=False
    torch.testing.assert_close(factored,confidence.rates(joint)*factor,atol=2e-5,rtol=2e-5)


def test_native_clearance_keeps_writer_and_resets_at_next_release(monkeypatch):
    values=torch.linspace(-1,1,6*1024*4).reshape(6,1,1024,4)
    cache=HSTUKVCache(values,values.cos(),1024)
    state=AdaptationState(cache,range(1024),0,segment_size=1024,slots=1,
                          write_correction=False,history_scope="all")
    mapper=RidgeTranslator(layers=6,width=4,history_scope="all",layer_clearance=True)
    state.release(1,mapper)
    for step in range(6):
        state.install_native_chunk(cache,range((step+1)*1024,(step+2)*1024))
    assert state.clearance_complete and state.writes_since_release == 6144
    refreshes=state.counts["refreshes"]
    native_calls=[]

    def native_read(model, retained, candidates, delta):
        native_calls.append(retained)
        return torch.ones_like(candidates,dtype=torch.float32)

    monkeypatch.setattr("hstu_kvcache.adaptation.inference.score_ready",native_read)
    state.score(None,torch.ones(1,1,dtype=torch.long),torch.ones(1))
    assert native_calls == [cache] and state.counts["refreshes"] == refreshes
    assert not state.correction_active and state.counts["cleared_reads"] == 1
    assert state.writer.next_ordinal == 7168 and len(state.writer.events) == 1024
    next_mapper=RidgeTranslator(layers=6,width=4,target=2,history_scope="all",layer_clearance=True)
    state.release(2,next_mapper)
    assert state.writes_since_release == 0 and not state.clearance_complete
    assert state.correction_active and state.writer.next_ordinal == 7168
