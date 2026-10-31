#!/usr/bin/env python3
"""Three cheap head-local ridge recipes from one256fit+16val native capture."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
import os
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import (
    ROOT, capture_samples, load_models, restore_pack, write_json,
)
from experiments.ch3_motivation.code.fit_simple_kv_maps import LAMBDAS, RowMoments, fit_from_moments
from experiments.ch3_motivation.code.simple_kv_maps import head_rows, maps_from_tensors
from hstu_kvcache.methods.baselines.protocol.cost import CostModel
from hstu_kvcache.common.artifacts import sha256
import torch

RECIPES={'linear100_users':(100,'users'),'linear256_rows':(256,'rows'),'linear256_users':(256,'users')}


@torch.no_grad()
def add_scaled_moments(total,user,weight):
    """Scale ALL sufficient statistics, including means and the row measure."""
    for name in ['sx','sy','yy','xx','xy']:
        getattr(total,name).add_(getattr(user,name),alpha=float(weight))
    total.rows+=user.rows*float(weight)


def load_robust_maps(path,device='cpu'):
    payload=torch.load(Path(path),map_location=device,weights_only=True)
    if payload['format']!='candidate_robust_maps_v1':raise ValueError('not robust KV maps')
    return {name:maps_from_tensors(values,payload['num_heads'],payload['metadata'])['kv_linear']
        for name,values in payload['maps'].items()}


def run(args):
    if args.gpu!=1:raise ValueError('physical GPU1 only')
    os.environ['CUDA_VISIBLE_DEVICES']='1';torch.cuda.set_device(0);device=torch.device('cuda:0')
    start=time.perf_counter();input_root=ROOT/'experiments/ch3_motivation/inputs_feedback_pool/fit256'
    pop_path=input_root/'population.json';pop=json.loads(pop_path.read_text())
    fit,val=pop['fit'][:256],pop['validation']
    if len(fit)!=256 or len(val)!=16 or len(set(fit+val))!=272:raise ValueError('fixed256fit/16val required')
    evaluation=json.loads(args.population.read_text())
    if len(set(evaluation['users']))!=7000 or set(evaluation['users'])&set(fit+val):raise ValueError('7000/fit256/val16 role separation failed')
    binding_path=input_root/f'edge_{args.edge}/inputs.json';binding=json.loads(binding_path.read_text())
    history_path=binding_path.parent/binding['history']['path']
    if binding['population_sha256']!=sha256(pop_path) or sha256(history_path)!=binding['history']['sha256']:raise ValueError('fitting input changed')
    old=torch.load(args.old_maps,map_location=device,weights_only=True);old_meta=old['metadata'];old_identity=old_meta['identity']
    if (old['format']!='simple_kv_maps_v1' or old_identity['edge']!=args.edge or old_identity['fit_uids']!=fit[:100]
        or old_identity['validation_users']!=val or old_identity['input_binding_sha256']!=sha256(binding_path)):
        raise ValueError('old100row map does not share source/prefix/roles')
    parent,current,models=load_models(args.edge,device)
    if old_identity['models']!=models:raise ValueError('old and new native models differ')
    cfg=binding['settings'];heads=current.cfg.num_heads;pairs=len(current.blocks)*heads;width=2*current.blocks[0].attn.head_dim
    paths=[Path(__file__),Path(__file__).with_name('fit_simple_kv_maps.py'),Path(__file__).with_name('simple_kv_maps.py'),
        Path(__file__).with_name('common.py'),ROOT/'src/hstu_kvcache/methods/baselines/protocol/cost.py',
        ROOT/'src/hstu_kvcache/models/hstu.py',ROOT/'src/hstu_kvcache/models/attention.py']
    sources={str(p.resolve()):sha256(p) for p in paths}
    identity=dict(format='candidate_robust_map_fit_v1',edge=args.edge,models=models,input_root=str(input_root),
        input_binding_sha256=sha256(binding_path),population_sha256=sha256(pop_path),history_sha256=sha256(history_path),
        evaluation_population_sha256=sha256(args.population),fit_uids=fit,validation_users=val,
        old_maps=str(args.old_maps.resolve()),old_maps_sha256=sha256(args.old_maps),numerical_sources=sources,
        recipes={name:dict(fit_uids=fit[:n],weighting=weight) for name,(n,weight) in RECIPES.items()},
        objective='raw teacher KV MSE; rows=all rows equally weighted; users=each user total weight1',
        regularizations=list(LAMBDAS),labels_used=False,candidate_queries_used=False,physical_gpu=1)
    args.output.mkdir(parents=True,exist_ok=True);configuration=args.output/'configuration.json'
    if configuration.exists() and json.loads(configuration.read_text())!=identity:raise ValueError('fitting identity changed')
    write_json(configuration,identity);checkpoint=args.output/'robust_maps.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['metadata']['identity']!=identity:raise ValueError('checkpoint identity changed')
        return saved['metadata']
    histories=restore_pack(history_path);moments={name:RowMoments(pairs,width,device) for name in RECIPES}
    lengths=[];fit_rows=0
    for index,uid in enumerate(fit):
        sample=capture_samples(parent,current,histories,[uid],cfg,device)[uid]
        source,target=sample['parent_cache'],sample['teacher_cache']
        x,y=head_rows(source,heads).to(device),head_rows(target,heads).to(device)
        user=RowMoments(pairs,width,device);user.add(x,y)
        for name,(n,weight) in RECIPES.items():
            if index<n:add_scaled_moments(moments[name],user,1/user.rows if weight=='users' else 1.)
        lengths.append(source.seq_len);fit_rows+=source.seq_len
        del sample,source,target,x,y,user
        if (index+1)%32==0:print(json.dumps(dict(stage='native_fit_statistics',users=index+1)),flush=True)
    candidates={name:fit_from_moments(moment) for name,moment in moments.items()}
    transforms={name:{reg:maps_from_tensors(tensors,heads)['kv_linear'] for reg,tensors in values.items()} for name,values in candidates.items()}
    old_map=maps_from_tensors(old['tensors'],heads)['kv_linear']
    errors={name:{reg:torch.zeros(2,device=device,dtype=torch.float64) for reg in LAMBDAS} for name in RECIPES}
    denominators={name:0. for name in RECIPES};old_error=torch.zeros(2,device=device,dtype=torch.float64);val_rows=0
    for uid in val:
        sample=capture_samples(parent,current,histories,[uid],cfg,device)[uid]
        source,target=sample['parent_cache'].to(device),sample['teacher_cache'].to(device);n=source.seq_len
        for name,(fit_count,weight) in RECIPES.items():
            scale=1/n if weight=='users' else 1.
            for reg,transform in transforms[name].items():
                prediction=transform(source)
                error=torch.stack(((prediction.k.double()-target.k.double()).square().sum(),(prediction.v.double()-target.v.double()).square().sum()))
                errors[name][reg].add_(error,alpha=scale)
            denominators[name]+=n*scale*pairs*(width//2)
        prediction=old_map(source)
        old_error+=torch.stack(((prediction.k.double()-target.k.double()).square().sum(),(prediction.v.double()-target.v.double()).square().sum()))
        lengths.append(n);val_rows+=n
        del sample,source,target,prediction,error
    maps={'linear100_rows':old['tensors']};recipe_metadata={}
    def mse(value,denominator):
        return dict(k_mse=float(value[0]/denominator),v_mse=float(value[1]/denominator),pooled_mse=float(value.sum()/(2*denominator)))
    recipe_metadata['linear100_rows']=dict(fit_uids=fit[:100],fit_users=100,weighting='rows',exact_old_checkpoint_reuse=True,
        selected_regularization=old_meta['selected_regularization'],validation_mse_original=old_meta['validation_mse'],
        validation_mse_rechecked=mse(old_error,val_rows*pairs*(width//2)),calibration_flops=old_meta['total_calibration_flops_estimate'])
    for name,(n,weight) in RECIPES.items():
        table={str(reg):mse(value,denominators[name]) for reg,value in errors[name].items()}
        selected=min(LAMBDAS,key=lambda reg:table[str(reg)]['pooled_mse']);maps[name]=candidates[name][selected]
        recipe_metadata[name]=dict(fit_uids=fit[:n],fit_users=n,weighting=weight,weighted_training_rows=moments[name].rows,
            selected_regularization=selected,validation_mse=table,
            validation_weighting='same rows/users weighting as fitting recipe')
    cm=CostModel(current.cfg.hidden_size,len(current.blocks),heads,cfg['attention_backend'])
    prefill=sum(2*(cm if n>=1024 else CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,'torch')).full_cache(n) for n in lengths)
    statistics=4*fit_rows*pairs*width**2+4*fit_rows*pairs*width+(2*100+256+2*256)*(2*pairs*width**2+3*pairs*width)
    solves=int(3*len(LAMBDAS)*(8/3)*pairs*width**3);normalization=3*(16*pairs*width**2+16*pairs*width)
    validation=val_rows*pairs*(3*len(LAMBDAS)+1)*(2*width**2+6*width)
    total=int(prefill+statistics+solves+normalization+validation)
    per_map={'linear100_rows':int(old_meta['total_calibration_flops_estimate']),**{name:total for name in RECIPES}}
    for name in RECIPES:recipe_metadata[name]['calibration_flops']=total
    metadata=dict(identity=identity,recipes=recipe_metadata,fit_users=256,validation_users=16,fit_rows=fit_rows,validation_rows=val_rows,
        num_heads=heads,pairs=pairs,feature_width=width,coefficient_parameters_per_map=pairs*(width*width+width),
        calibration_cost=dict(source_teacher_prefill_flops=int(prefill),streaming_statistics_flops=int(statistics),
            regularized_solve_flops_estimate=solves,normalization_flops_estimate=normalization,validation_flops=int(validation)),
        total_calibration_flops_estimate=total,per_map_calibration_flops_estimate=per_map,
        cost_scope='Actual shared256fit+16val native prefills, one raw Gram per UID, all weighted accumulators/solves/validation. Each new map conservatively charges full shared new fitter; old100row keeps its original calibration charge.',
        quality_auc_computed=False,physical_gpu=1,seconds=time.perf_counter()-start)
    if any(sha256(Path(p))!=h for p,h in sources.items()):raise ValueError('fitter sources changed during execution')
    payload=dict(format='candidate_robust_maps_v1',num_heads=heads,maps={name:{key:value.cpu() for key,value in tensors.items()} for name,tensors in maps.items()},metadata=metadata)
    temp=checkpoint.with_suffix('.pt.writing');torch.save(payload,temp);temp.replace(checkpoint)
    write_json(args.output/'summary.json',dict(metadata,status='complete',checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint)))
    print(json.dumps(dict(status='complete',edge=args.edge,selected={name:v['selected_regularization'] for name,v in recipe_metadata.items()})),flush=True)
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edge',type=int,choices=[2,3],required=True)
    parser.add_argument('--old-maps',type=Path,required=True)
    parser.add_argument('--population',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=1)
    args=parser.parse_args()
    try:run(args);write_json(args.output/'exit.json',dict(exit_code=0))
    except BaseException as error:write_json(args.output/'exit.json',dict(exit_code=1,error=str(error)));raise
