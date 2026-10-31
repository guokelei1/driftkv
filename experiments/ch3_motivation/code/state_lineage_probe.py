#!/usr/bin/env python3
"""Small fixed-snapshot lineage maps; ordinary MLPs, no Design implementation.

Every stage trains on its own actual chain input. Anchored is a rescue from
the same chain_previous, never an independently recursive anchored state.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
import numpy as np
import torch
from torch import nn
from experiments.ch3_motivation.code.common import load_model, restore_pack, write_json
from experiments.ch3_motivation.code.simple_kv_maps import head_rows, rows_to_cache
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.common.artifacts import sha256

RUN=ROOT/'experiments/ch3_motivation/runs/lineage_candidate/attempt_1/state'
INPUT=ROOT/'experiments/ch3_motivation/inputs_feedback_pool/fit256'
KINDS=('chain','direct','anchored')


def row_indices(length,budget):
    if length<1 or budget<1:raise ValueError('Positive history and row budget required')
    return np.linspace(0,length-1,min(length,budget),dtype=np.int64)


def branch_inputs(original,previous,kind):
    if kind=='chain':return previous,previous
    if kind=='direct':return original,original
    if kind=='anchored':return original,previous
    raise ValueError(kind)


class RowResidualMap(nn.Module):
    """Independent layer/head MLPs: [firstKV,secondKV] -> secondKV + residual."""
    def __init__(self,pairs,width,hidden):
        super().__init__();self.pairs=pairs;self.width=width;self.hidden=hidden
        self.w1=nn.Parameter(torch.empty(pairs,2*width,hidden))
        self.b1=nn.Parameter(torch.zeros(pairs,hidden))
        self.w2=nn.Parameter(torch.zeros(pairs,hidden,width))
        self.b2=nn.Parameter(torch.zeros(pairs,width))
        nn.init.uniform_(self.w1,-1/math.sqrt(2*width),1/math.sqrt(2*width))
        self.register_buffer('mean',torch.zeros(pairs,1,2*width))
        self.register_buffer('scale',torch.ones(pairs,1,2*width))
        self.register_buffer('output_scale',torch.ones(pairs,1,1))

    @torch.no_grad()
    def normalize_from_fit(self,first,second,teacher):
        x=torch.cat((first,second),-1)
        self.mean.copy_(x.mean(1,keepdim=True))
        self.scale.copy_(x.std(1,keepdim=True,unbiased=False).clamp_min(1e-4))
        self.output_scale.copy_(teacher.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-4))

    def forward(self,first,second):
        x=(torch.cat((first,second),-1)-self.mean)/self.scale
        hidden=torch.nn.functional.silu(torch.bmm(x,self.w1)+self.b1[:,None])
        return second+(torch.bmm(hidden,self.w2)+self.b2[:,None])*self.output_scale

    def forward_flops(self,rows):
        return int(self.pairs*rows*(2*(2*self.width)*self.hidden+2*self.hidden*self.width
            +3*(2*self.width)+7*self.hidden+4*self.width))


def fit_map(first,second,teacher,val_first,val_second,val_teacher,*,hidden=64,steps=256,
            batch_rows=512,lr=1e-3,seed=17,validate_every=32):
    """Uniform sampled-row MSE; validation is teacher KV only, includes step0."""
    torch.manual_seed(seed)
    model=RowResidualMap(first.shape[0],first.shape[-1],hidden).to(first.device)
    model.normalize_from_fit(first,second,teacher)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    generator=torch.Generator(device=first.device).manual_seed(seed+1)
    def validation():
        with torch.no_grad():return float((model(val_first,val_second)-val_teacher).square().mean())
    best=validation();selected=0;state=copy.deepcopy(model.state_dict());trace=[dict(step=0,validation_kv_mse=best)]
    for step in range(1,steps+1):
        index=torch.randint(first.shape[1],(batch_rows,),generator=generator,device=first.device)
        loss=(model(first[:,index],second[:,index])-teacher[:,index]).square().mean()
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training MSE')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%validate_every==0 or step==steps:
            value=validation();trace.append(dict(step=step,training_kv_mse=float(loss.detach()),validation_kv_mse=value))
            if value<best:best=value;selected=step;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval().requires_grad_(False)
    return model,dict(selected_step=selected,validation_kv_mse=best,trace=trace,
        parameter_count=sum(p.numel() for p in model.parameters()),actual_optimizer_steps=steps,
        train_rows=first.shape[1],validation_rows=val_first.shape[1],sampled_rows_per_step=batch_rows)


def load_native(version,device):
    path=ROOT/f'results/unified_training_2026_09/large/seed17/checkpoints/v{version}/checkpoint_100.pt'
    model,info=load_model(path,device);del info
    model.eval().requires_grad_(False)
    if model.cfg.hidden_size!=320 or len(model.blocks)!=10 or model.cfg.num_heads!=10:
        raise ValueError('Expected native Large10L/H320/10heads')
    return model,path


@torch.no_grad()
def native_snapshot(model,version,events,cutoff):
    return NativeTrajectory(model,version,events,cutoff,max_length=1024,append_band_size=128).cache


@torch.no_grad()
def replay_rows(original,bank,target):
    previous=original;result={};stage2=None
    for version in range(2,target+1):
        stage=bank[version];old_previous=previous
        result={kind:stage[kind](*branch_inputs(original,old_previous,kind)) for kind in KINDS}
        previous=result['chain']
        if version==2:stage2=previous
    if target==4 and 'short' in bank:
        for kind,model in bank['short'].items():
            result['short_'+kind]=model(*branch_inputs(original,stage2,kind))
    return result


@torch.no_grad()
def replay_caches(original,bank,target,heads=10):
    outputs=replay_rows(head_rows(original,heads),bank,target)
    return {kind:rows_to_cache(value,original,heads) for kind,value in outputs.items()}


def load_bank(path,device):
    payload=torch.load(path,map_location=device,weights_only=True)
    if payload['format']!='state_lineage_maps_v1':raise ValueError('Unknown lineage checkpoint')
    bank={}
    for version,stage in payload['maps'].items():
        bank[int(version)]={}
        for kind,state in stage.items():
            model=RowResidualMap(payload['pairs'],payload['width'],payload['hidden']).to(device)
            model.load_state_dict(state);bank[int(version)][kind]=model.eval().requires_grad_(False)
    if payload.get('short_maps'):
        bank['short']={}
        for kind,state in payload['short_maps'].items():
            model=RowResidualMap(payload['pairs'],payload['width'],payload['hidden']).to(device)
            model.load_state_dict(state);bank['short'][kind]=model.eval().requires_grad_(False)
    return bank,payload


def run_fit(args):
    if args.gpu!=1:raise ValueError('Physical GPU1 only')
    population=json.loads(args.population.read_text());evaluation=population['users']
    roles=json.loads((INPUT/'population.json').read_text());fit=roles['fit'][:100];val=roles['validation']
    if (len(fit)!=100 or len(val)!=16 or len(set(fit+val))!=116 or set(evaluation)&set(fit+val)
            or population['fit_uids']!=fit or population['validation_uids']!=val):
        raise ValueError('Frozen100fit/16val disjoint roles required')
    directory=INPUT/f'edge_{args.final_target}';binding_path=directory/'inputs.json';binding=json.loads(binding_path.read_text())
    hp=directory/binding['history']['path']
    if sha256(hp)!=binding['history']['sha256'] or binding['population_sha256']!=sha256(INPUT/'population.json'):
        raise ValueError('Frozen fit histories/roles changed')
    cutoff=int(binding['settings']['release_day'])*86400
    sources={str(p.resolve()):sha256(p) for p in (Path(__file__),Path(__file__).with_name('simple_kv_maps.py'),
        Path(__file__).with_name('common.py'),ROOT/'src/hstu_kvcache/evaluation/trajectory.py',
        ROOT/'src/hstu_kvcache/models/state_transition.py',ROOT/'src/hstu_kvcache/models/hstu.py',
        ROOT/'src/hstu_kvcache/models/attention.py')}
    versions=list(range(1,args.final_target+1));model_paths={v:ROOT/f'results/unified_training_2026_09/large/seed17/checkpoints/v{v}/checkpoint_100.pt' for v in versions}
    identity=dict(format='state_lineage_fit_v1',source_version=1,final_target=args.final_target,versions=versions,
        cutoff=cutoff,history_cap=1024,fit_uids=fit,validation_users=val,population_sha256=sha256(args.population),
        fit_population_sha256=sha256(INPUT/'population.json'),input_binding_sha256=sha256(binding_path),history_sha256=sha256(hp),
        models={str(p.relative_to(ROOT)):sha256(p) for p in model_paths.values()},sources=sources,
        row_budget=args.rows,hidden=args.hidden,steps=args.steps,batch_rows=args.batch_rows,lr=args.lr,seed=args.seed,
        objective='Raw native teacher KV MSE; no labels or candidates used in fitting/selection',
        input_rules={'chain':'chain_previous,chain_previous','direct':'original,original','anchored':'original,chain_previous'},
        normalization='Fit-row input mean/std; teacher per-head RMS sets residual units; validation only chooses teacher-MSE checkpoint',
        snapshot_scope='One final-target prerelease raw prefix, held fixed for all versions and all migration stages; not a full serving replay',
        first_stage='All three inputs coincide; one fit shared as exactly matched weights',physical_gpu=1)
    identity.update(torch_version=str(torch.__version__),short_path=[1,2,4] if args.final_target==4 else None,
        short_control='Same final4 teacher/history as long1→2→3→4; short chain and rescue reuse exact stage2 previous')
    args.output.mkdir(parents=True,exist_ok=True);cp=args.output/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=identity:raise ValueError('Frozen fit configuration changed')
    write_json(cp,identity)
    if (args.output/'maps.pt').exists():raise ValueError('Completed checkpoint already exists; do not refit')
    os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['EVOKV_ATTENTION_BACKEND']='triton'
    torch.cuda.set_device(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');started=time.perf_counter();histories=restore_pack(hp);samples={};lengths={}
    for version in versions:
        model,_=load_native(version,device);sample=[]
        for index,uid in enumerate(fit+val):
            cache=native_snapshot(model,version,histories.rows[uid],cutoff)
            indices=row_indices(cache.seq_len,args.rows);lengths[uid]=cache.seq_len
            sample.append(head_rows(cache,10)[:,indices].cpu());del cache
            if (index+1)%32==0:print(json.dumps(dict(stage='capture',version=version,users=index+1)),flush=True)
        split=sum(min(lengths[u],args.rows) for u in fit)
        joined=torch.cat(sample,1).to(device);samples[version]=joined[:,:split],joined[:,split:]
        del sample,joined,model;torch.cuda.empty_cache()
    original,original_val=samples[1];previous,previous_val=original,original_val;bank={};reports={};stage2=None
    for version in versions[1:]:
        teacher,teacher_val=samples[version];stage={};report={}
        for kind in KINDS:
            if version==2 and kind!='chain':
                stage[kind]=stage['chain'];report[kind]=dict(report['chain'],exact_first_stage_shared_fit=True);continue
            first,second=branch_inputs(original,previous,kind);vf,vs=branch_inputs(original_val,previous_val,kind)
            stage[kind],report[kind]=fit_map(first,second,teacher,vf,vs,teacher_val,hidden=args.hidden,steps=args.steps,
                batch_rows=args.batch_rows,lr=args.lr,seed=args.seed,validate_every=32)
            print(json.dumps(dict(stage='fit',version=version,kind=kind,selected_step=report[kind]['selected_step'],validation_kv_mse=report[kind]['validation_kv_mse'])),flush=True)
        with torch.no_grad():previous=stage['chain'](previous,previous);previous_val=stage['chain'](previous_val,previous_val)
        if version==2:stage2=(previous,previous_val)
        bank[version]=stage;reports[version]=report
    short={};short_reports={}
    if args.final_target==4:
        for kind in ('chain','anchored'):
            first,second=branch_inputs(original,stage2[0],kind);vf,vs=branch_inputs(original_val,stage2[1],kind)
            short[kind],short_reports[kind]=fit_map(first,second,samples[4][0],vf,vs,samples[4][1],
                hidden=args.hidden,steps=args.steps,batch_rows=args.batch_rows,lr=args.lr,seed=args.seed,validate_every=32)
            print(json.dumps(dict(stage='short_fit',kind=kind,selected_step=short_reports[kind]['selected_step'],
                validation_kv_mse=short_reports[kind]['validation_kv_mse'])),flush=True)
    if any(sha256(Path(p))!=h for p,h in sources.items()):raise ValueError('Numerical source changed while fitting')
    payload=dict(format='state_lineage_maps_v1',pairs=100,width=64,hidden=args.hidden,identity=identity,
        maps={v:{k:{name:value.cpu() for name,value in m.state_dict().items()} for k,m in stage.items()} for v,stage in bank.items()},
        short_maps={kind:{name:value.cpu() for name,value in model.state_dict().items()} for kind,model in short.items()})
    temp=args.output/'maps.pt.writing';torch.save(payload,temp);temp.replace(args.output/'maps.pt')
    summary=dict(status='complete',identity=identity,checkpoint_sha256=sha256(args.output/'maps.pt'),stages=reports,
        short_stages=short_reports,
        sampled_fit_rows=original.shape[1],sampled_validation_rows=original_val.shape[1],history_lengths=lengths,
        seconds=time.perf_counter()-started,physical_gpu=1,quality_auc_computed=False)
    write_json(args.output/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final-target',type=int,choices=(3,4),required=True)
    p.add_argument('--population',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--rows',type=int,default=128)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--steps',type=int,default=256)
    p.add_argument('--batch-rows',type=int,default=512);p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--seed',type=int,default=17);args=p.parse_args()
    try:run_fit(args);write_json(args.output/'exit.json',dict(exit_code=0))
    except BaseException as error:write_json(args.output/'exit.json',dict(exit_code=1,error=str(error)));raise
