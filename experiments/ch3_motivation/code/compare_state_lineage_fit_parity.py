#!/usr/bin/env python3
"""CPU-only retained-map parity after row caching and unused-fit removal."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
import torch
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.common.artifacts import sha256


def compare(target,new_folder,old_folder):
    torch.set_num_threads(1)
    nsummary,osummary=[json.loads((p/'summary.json').read_text()) for p in (new_folder,old_folder)]
    if any(s['status']!='complete' for s in (nsummary,osummary)):raise ValueError('Two complete fits required')
    new_path,old_path=[p/'maps.pt' for p in (new_folder,old_folder)]
    for path,summary in ((new_path,nsummary),(old_path,osummary)):
        if sha256(path)!=summary['checkpoint_sha256']:raise ValueError('Checkpoint SHA changed')
    new,old=[torch.load(p,map_location='cpu',weights_only=True,mmap=True) for p in (new_path,old_path)]
    keys=('final_target','source_version','cutoff','history_cap','fit_uids','validation_users','population_sha256',
        'models','row_budget','hidden','steps','batch_rows','lr','seed','history_sha256','input_binding_sha256')
    for key in keys:
        if new['identity'][key]!=old['identity'][key]:raise ValueError(f'Numerical input/configuration differs: {key}')
    if new['hidden']!=64 or new['identity']['steps']!=256:raise ValueError('Only original W64/256 parity is intended')
    if set(new['maps'])!=set(range(2,target+1)):raise ValueError('Unexpected migration stages')
    reports={};passed=bitwise=True
    for version,stage in new['maps'].items():
        if set(stage)!=({'chain','direct'} if version==target else {'chain'}):raise ValueError('Unused map retained or required map absent')
        for kind,state in stage.items():
            reference=old['maps'][version][kind]
            if set(state)!=set(reference):raise ValueError('State tensor names differ')
            name=f'v{version}_{kind}';checks={}
            for field,a in state.items():
                b=reference[field]
                if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('State tensor geometry differs')
                same=torch.equal(a,b);ok=bool(torch.allclose(a,b,atol=1e-6,rtol=1e-5))
                passed &= ok;bitwise &= same
                checks[field]=dict(bitwise=same,passed=ok,max_abs=float((a-b).abs().max()),elements=a.numel())
            selected_new=nsummary['stages'][str(version)][kind]['selected_step']
            selected_old=osummary['stages'][str(version)][kind]['selected_step']
            passed &= selected_new==selected_old
            reports[name]=dict(tensors=checks,selected_step_new=selected_new,selected_step_old=selected_old)
    files=[new_path,old_path,new_folder/'summary.json',old_folder/'summary.json',Path(__file__)]
    return dict(format='state_lineage_fit_parity_v1',passed=passed,all_weights_bitwise=bitwise,target=target,
        atol=1e-6,rtol=1e-5,maps=reports,quality_use_permitted=False,
        scope='Same W64/256 algorithm and captured raw snapshots; numerical readiness only, no quality or independent replication claim',
        sources=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in files])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=int,choices=(3,4),required=True);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--old',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    old=args.old or ROOT/f'experiments/ch3_motivation/runs/lineage_candidate/attempt_1/state/final_{args.target}'
    try:result=compare(args.target,args.new,old)
    except BaseException as error:result=dict(passed=False,target=args.target,error=str(error))
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='maps' and k!='sources'}))
    raise SystemExit(0 if result['passed'] else 1)
