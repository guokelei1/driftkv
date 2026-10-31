"""Native-pair fit inputs differ from predicted-state deployment inputs."""
from pathlib import Path
import copy
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import state_lineage_native_pair as probe
from experiments.ch3_motivation.code.state_lineage_probe import RowResidualMap, load_bank
from experiments.ch3_motivation.code.score_state_lineage_cached_gap import project_final
from hstu_kvcache.models import HSTUKVCache


@pytest.mark.parametrize('target',[3,4])
def test_native_training_then_predicted_inference_exact_reused_maps(target,monkeypatch,tmp_path):
    torch.set_num_threads(1);torch.manual_seed(322)
    rows={v:dict(fit=torch.randn(2,24,4)+v,validation=torch.randn(2,9,4)+v) for v in range(1,target+1)}
    first=RowResidualMap(2,4,8).eval().requires_grad_(False)
    direct=RowResidualMap(2,4,8).eval().requires_grad_(False)
    with torch.no_grad():first.b2.fill_(.27);direct.b2.fill_(.83)
    reference={2:{'chain':first},target:{'direct':direct}}
    reports={'2':{'chain':dict(selected_step=0,validation_kv_mse=1.)},str(target):{'direct':dict(selected_step=0,validation_kv_mse=2.)}}
    recorded=[]
    def fake_fit(a,b,teacher,va,vb,vt,**kwargs):
        version=len(recorded)+3
        assert torch.equal(a,rows[version-1]['fit']) and torch.equal(a,b)
        assert torch.equal(va,rows[version-1]['validation']) and torch.equal(va,vb)
        assert torch.equal(teacher,rows[version]['fit']) and torch.equal(vt,rows[version]['validation'])
        model=RowResidualMap(2,4,8).eval().requires_grad_(False)
        with torch.no_grad():model.b2.fill_(version/10)
        recorded.append((a.clone(),va.clone()))
        return model,dict(selected_step=7,validation_kv_mse=probe.mse(model(va,vb),vt))
    monkeypatch.setattr(probe,'fit_map',fake_fit)
    untouched={v:{s:x.clone() for s,x in values.items()} for v,values in rows.items()}
    bank,actual_reports,diagnostics=probe.fit_native_pairs(rows,reference,reports,target=target,hidden=8,steps=8)
    assert len(recorded)==target-2 and set(bank[target])=={'chain','direct'}
    for field,value in first.state_dict().items():assert torch.equal(bank[2]['chain'].state_dict()[field],value)
    for field,value in direct.state_dict().items():assert torch.equal(bank[target]['direct'].state_dict()[field],value)
    previous=rows[1]['validation']
    for version in range(2,target+1):
        if version>=3:
            diag=diagnostics[version]
            expected=probe.mse(bank[version]['chain'](previous,previous),rows[version]['validation'])
            assert diag['actual_chain_input_validation_mse']==expected
            assert not diag['actual_chain_diagnostics_used_for_selection']
            assert diag['validation_input_native_minus_actual_chain_mse']>0
            assert actual_reports[version]['chain']['selected_step']==7
        previous=bank[version]['chain'](previous,previous)
    for v,values in rows.items():
        for split,x in values.items():assert torch.equal(x,untouched[v][split])
    path=tmp_path/'maps.pt';torch.save(dict(format='state_lineage_maps_v1',pairs=2,width=4,hidden=8,
        maps={v:{kind:model.state_dict() for kind,model in stage.items()} for v,stage in bank.items()}),path)
    loaded,_=load_bank(path,'cpu');original=HSTUKVCache(torch.randn(1,1,6,4),torch.randn(1,1,6,4),6)
    actual=project_final(original,loaded,target,heads=2)
    assert set(actual)=={'direct','chain'}
