"""Reject mislabeled or drifting supplement weights before quality scoring."""
from pathlib import Path
import sys
import json
import torch
import pytest
ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    frozen_predecessor_quality as audit, run_probe_frozen_predecessors as driver,
)
from hstu_kvcache.common.artifacts import sha256
from tests.experiments.ch3_motivation.test_run_probe_frozen_predecessors import setup, backend
from experiments.ch3_motivation.code.plan_probe_formal import _input_binding


def prepared(tmp_path,monkeypatch):
    fx,args,calls=setup(tmp_path,monkeypatch);result=driver.run(args)
    source=json.loads((args.source_f.parent/'configuration.json').read_text())
    job=dict(output=str(args.output),source_job=dict(output=str(args.source_f.parent),expected_identity=source,seed=17,fit_users=64,steps=512,edge=1))
    binding=_input_binding(args.input_root,1)
    binding.update(sampling_manifest=source['sampling_manifest'],sampling_manifest_sha256=source['sampling_manifest_sha256'])
    return args,job,binding


def test_actual_ladder_protocol_source_and_frozen_states(tmp_path,monkeypatch):
    args,job,binding=prepared(tmp_path,monkeypatch)
    result=audit.audit_ladder(job,binding)
    assert result['identity']['training_protocol']['format']=='e5_new_term_only_v1'
    assert [row['method_label'] for row in result['weights']]==['f','fg','fgh']
    assert all(row['predecessors_exact'] for row in result['weights'])


@pytest.mark.parametrize('mutation',['f_weight','identity','cost','source_hash'])
def test_reject_rewritten_frozen_state_identity_or_cost(tmp_path,monkeypatch,mutation):
    args,job,binding=prepared(tmp_path,monkeypatch)
    path=args.output/'fgh.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
    if mutation=='f_weight':payload['state_dict']['f.0.weights.0'].add_(.1)
    elif mutation=='identity':payload['metadata']['identity']['resident_inputs']=False
    elif mutation=='cost':payload['metadata']['cumulative_calibration_flops_estimate']+=1
    else:payload['metadata']['identity']['source_f_sha256']='0'*64
    torch.save(payload,path)
    path.with_suffix('.json').write_text(json.dumps(dict(payload['metadata'],checkpoint=str(path),checkpoint_sha256=sha256(path))))
    with pytest.raises(ValueError):audit.audit_ladder(job,binding)
