"""Configured reads and in-place transitions share the causal service path."""
from pathlib import Path
import copy
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]

from code_structure_2026_10.structure import StructuredKVAdapter
from hstu_kvcache.design_two.replay import NativeTrajectory
from hstu_kvcache.models import HSTU,HSTUConfig
from implementation_2026_10.backend import ImplementationConfig,InferenceAdapter,configure_prefix
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from query_relay_2026_10.precision import StableReleasePrefix
from query_relay_2026_10.session import RelaySession


@torch.no_grad()
@pytest.mark.parametrize('heads',[6,10])
def test_configured_torch_service_matches_reference_across_release_write_fifo(monkeypatch,heads):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch')
    torch.manual_seed(20261015)
    cfg=HSTUConfig(num_items=48,num_behaviors=2,hidden_size=heads*32,num_layers=2,
        num_heads=heads,max_seq_len=4,input_dropout=0.,attn_dropout=0.)
    models=[HSTU(cfg).eval().requires_grad_(False)]
    for _ in range(3):
        current=copy.deepcopy(models[-1])
        current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight)*.01)
        models.append(current)
    source=StructuredKVAdapter(2,heads,32,max_length=4,layout='shared12_private6').eval()
    for layer in source.layers:
        layer.input_weight.normal_(std=.01)
        layer.input_bias.normal_(std=.01)
        layer.key_decoder.normal_(std=.01)
        layer.decoder.normal_(std=.01)
    adapters=[]
    for _ in range(3):
        a=UnifiedRelayAdapter.from_structured(source,mode='A',read_shared=3,read_private=3,
            gate_source='item',gate_normalization='rms').eval().requires_grad_(False)
        for layer in a.layers:
            layer.key_gate_weight.normal_(std=.02)
            layer.value_gate_weight.normal_(std=.02)
            layer.item_update_bias.normal_(std=.01)
        adapters.append(a)
    events=(np.arange(20)*10,np.arange(1,21),np.zeros(20,dtype=np.int64))
    make=lambda:RelaySession(NativeTrajectory(models[0],2,events,40,max_length=4,append_band_size=4))
    reference,optimized=make(),make()
    config=ImplementationConfig(reader_backend='torch',row_chunk=2)
    prefix=None
    for index,at in enumerate((40,70,100)):
        model,adapter=models[index+1],adapters[index]
        prefix=adapter.compile_release(prefix,model.lookup_item_embeddings(torch.arange(49)[None])[0])
        stable=StableReleasePrefix.wrap(prefix)
        wrapper=InferenceAdapter(adapter,config)
        reference.publish(model,index+3,at,adapter,prefix=stable)
        optimized.publish(model,index+3,at,wrapper,prefix=configure_prefix(stable,config))
        for timestamp in (at,at+1,at+11):
            item=torch.tensor([[30,31,32]])
            original=reference.score(timestamp,item)
            actual=optimized.score(timestamp,item)
            torch.testing.assert_close(actual,original,atol=5e-6,rtol=3e-5)
            torch.testing.assert_close(optimized.cache.k,reference.cache.k,atol=5e-6,rtol=3e-5)
            torch.testing.assert_close(optimized.state.codes,reference.state.codes,atol=5e-6,rtol=3e-5)
            assert optimized.coded_rows==reference.coded_rows
        assert reference.costs==optimized.costs
        assert reference.trajectory.ledger==optimized.trajectory.ledger
        assert all('forward_with_cache' not in block.attn.__dict__ for block in model.blocks)


def test_registration_has_all_strata_and_unchanged_acceptance_scope():
    import json
    registration=ROOT/'results/implementation_2026_10/registration.json'
    if not registration.exists():
        pytest.skip('real-data registration is a separate artifact')
    value=json.loads(registration.read_text())
    assert len(set(value['selected_uids']))==32
    assert set(value['selected_initial_quotas'].values())=={8}
    assert value['protocols']==['behavior_stride16_candidates16','original_feedback']
    assert 'No training or model/role/workload tuning' in value['exclusions']
