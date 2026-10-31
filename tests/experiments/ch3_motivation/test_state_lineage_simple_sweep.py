"""Removing unused fits keeps the actual chain algorithm and checkpoints."""
from pathlib import Path
import sys

import pytest
import torch

from experiments.ch3_motivation.code import state_lineage_simple_sweep as sweep
from experiments.ch3_motivation.code.state_lineage_probe import branch_inputs, fit_map, load_bank
from experiments.ch3_motivation.code.score_state_lineage_cached_gap import project_final
from hstu_kvcache.models import HSTUKVCache


@pytest.mark.parametrize('target',[3,4])
def test_simple_fits_are_exact_legacy_actual_chain_sequence(target,monkeypatch,tmp_path):
    torch.set_num_threads(1);torch.manual_seed(116)
    rows={version:dict(fit=torch.randn(2,24,4),validation=torch.randn(2,8,4)) for version in range(1,target+1)}
    original,original_val=rows[1]['fit'],rows[1]['validation']
    kwargs=dict(hidden=8,steps=8,batch_rows=16,lr=.01,seed=17)
    previous,previous_val=original,original_val;legacy={}
    for version in range(2,target+1):
        legacy[version]={}
        for kind in ('chain','direct','anchored'):
            if version==2 and kind!='chain':legacy[version][kind]=legacy[version]['chain'];continue
            first,second=branch_inputs(original,previous,kind)
            vf,vs=branch_inputs(original_val,previous_val,kind)
            legacy[version][kind],_=fit_map(first,second,rows[version]['fit'],vf,vs,rows[version]['validation'],
                **kwargs,validate_every=32)
        with torch.no_grad():
            previous=legacy[version]['chain'](previous,previous)
            previous_val=legacy[version]['chain'](previous_val,previous_val)
    recorded=[]
    def track(first,second,*args,**options):
        recorded.append((first.clone(),second.clone()))
        return fit_map(first,second,*args,**options)
    monkeypatch.setattr(sweep,'fit_map',track)
    bank,reports=sweep.fit_rows(rows,target=target,**kwargs)
    assert len(recorded)==target  # target−1 chain fits plus exactly one final direct.
    assert set(bank[target])=={'direct','chain'}
    assert all(set(stage)=={'chain'} for v,stage in bank.items() if v<target)
    with torch.no_grad():actual_previous=bank[2]['chain'](original,original)
    torch.testing.assert_close(recorded[1][0],actual_previous,atol=0,rtol=0)
    assert not torch.equal(recorded[1][0],rows[2]['fit'])  # No teacher forcing.
    torch.testing.assert_close(recorded[-1][0],original,atol=0,rtol=0)
    for version,stage in bank.items():
        for kind,model in stage.items():
            for name,value in model.state_dict().items():
                torch.testing.assert_close(value,legacy[version][kind].state_dict()[name],atol=0,rtol=0)
    path=tmp_path/'maps.pt'
    torch.save(dict(format='state_lineage_maps_v1',pairs=2,width=4,hidden=8,
        maps={v:{k:m.state_dict() for k,m in stage.items()} for v,stage in bank.items()}),path)
    loaded,_=load_bank(path,'cpu')
    cache=HSTUKVCache(torch.randn(1,1,6,4),torch.randn(1,1,6,4),6)
    outputs=project_final(cache,loaded,target,heads=2)
    assert set(outputs)=={'direct','chain'}
