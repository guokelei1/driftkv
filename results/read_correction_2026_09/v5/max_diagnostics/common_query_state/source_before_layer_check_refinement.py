#!/usr/bin/env python3
"""Fixed-query layer-zero state diagnostic; no fitting, feedback labels or AUC."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]

import numpy as np
import torch
import torch.nn.functional as F

from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.read_correction_v5.query_only import QueryFeatureCorrection
from read_correction_2026_09.calibrate import candidates
from read_correction_2026_09.v2 import calibrate as pure_loader
from read_correction_v5.history_conditioned import capture as mixed_loader
from read_correction_v5.common import configuration, PANEL_ROOT, RESERVATIONS, edge_name, sources, sha256

OUTPUT = ROOT/'results/read_correction_2026_09/v5/max_diagnostics/common_query_state'


def facts(raw, known):
    times,items,behaviors = [np.asarray(value,dtype=np.int64) for value in raw]
    ties = times[1:]==times[:-1]
    inverse = ties & (items[1:]<items[:-1])
    known_pair = (items[1:]<known)&(items[:-1]<known)
    digest = hashlib.sha256()
    for array in (times,items,behaviors):
        digest.update(array.tobytes())
    return {'length':len(times),'last_timestamp':int(times[-1]),'terminal_sha256':digest.hexdigest(),
        'tied_adjacent_pairs':int(ties.sum()),'mapped_item_inversions_within_ties':int(inverse.sum()),
        'known_item_inversions_within_ties':int((inverse&known_pair).sum())}


def distribution(values):
    values=np.asarray(values,dtype=float)
    return {'count':len(values),'quantiles':{str(q):float(np.quantile(values,q)) for q in (0,.1,.25,.5,.75,.9,.99,1)},
            'mean':float(values.mean()),'zero_count':int((values==0).sum())}


def energy(x):
    return float(x.double().square().mean())


def state_statistics(target, prediction, full_update=None):
    mean=target.double().mean(0,keepdim=True)
    zero=energy(target)
    variance=energy(target.double()-mean)
    actual=energy(target-prediction)
    result={'users':len(target),'zero_correction_mse':zero,'oracle_common_query_mse':variance,
        'oracle_variance_over_zero_error':variance/zero if zero else None,
        'oracle_common_query_mean_rms':energy(mean)**.5,'actual_Q512_mse':actual,
        'actual_Q512_over_zero_error':actual/zero if zero else None,
        'actual_Q512_mean_bias_mse':energy(prediction.double()-mean)}
    if full_update is not None:
        denominator=energy(full_update)
        result.update(full_update_rms=denominator**.5,
            target_rms_over_full_update_rms=(zero/denominator)**.5 if denominator else None,
            actual_error_rms_over_full_update_rms=(actual/denominator)**.5 if denominator else None)
    return result


def paired_statistics(pure, mixed):
    delta=mixed.double()-pure.double()
    total=energy(delta)
    meanshift=energy(delta.mean(0,keepdim=True))
    return {'paired_target_change_rms':total**.5,'state_mean_shift_rms':meanshift**.5,
        'paired_change_variance':max(0.,total-meanshift),
        'paired_change_rms_over_pure_target_rms':(total/energy(pure))**.5 if energy(pure) else None}


@torch.no_grad()
def execute(scale, edge, gpu, folder):
    cfg={**configuration(),'calibration_users':64,'validation_users':0,
         'calibration_queries_per_user':32,'candidate_mode':'uniform_known'}
    args=argparse.Namespace(scale=scale,edge=edge,gpu=gpu,panel_root=PANEL_ROOT,reservations=RESERVATIONS,budgets=[64])
    panel_path=PANEL_ROOT/scale/edge_name(edge)/'binding.json'
    panel=json.loads(panel_path.read_text())
    checkpoint=torch.load(ROOT/panel['sources']['current']['path'],map_location='cpu',weights_only=False,mmap=True)
    dataset=json.loads((ROOT/panel['sources']['dataset']['path']).read_text())
    known=int(checkpoint.get('known_vocab_size',dataset['foundation_items']))
    del checkpoint
    common_candidates=torch.from_numpy(candidates(0,known,32))
    pure_facts,mixed_facts={},{}
    original_pure,original_mixed=pure_loader.capture,mixed_loader.capture_mixed

    def observe_pure(parent,current,histories,uids,**kwargs):
        pure_facts.update({str(uid):facts(histories[uid],known) for uid in uids})
        return original_pure(parent,current,histories,uids,**kwargs)

    def observe_mixed(parent,current,timelines,uids,**kwargs):
        mixed_facts.update({str(uid):facts(timelines[uid]['terminal'],known) for uid in uids})
        return original_mixed(parent,current,timelines,uids,**kwargs)

    pure_loader.capture,mixed_loader.capture_mixed=observe_pure,observe_mixed
    try:
        print(json.dumps({'status':'capture_pure','scale':scale,'edge':edge}),flush=True)
        current,pure,uids,device,pure_meta=pure_loader.load_data(args,cfg)
        peak_alloc=torch.cuda.max_memory_allocated(gpu);peak_reserved=torch.cuda.max_memory_reserved(gpu)
        del current
        gc.collect();torch.cuda.empty_cache()
        print(json.dumps({'status':'capture_mixed','scale':scale,'edge':edge}),flush=True)
        current,mixed,mixed_uids,device,mixed_meta=mixed_loader.load_mixed_data(args,cfg)
    finally:
        pure_loader.capture,mixed_loader.capture_mixed=original_pure,original_mixed
    assert uids==mixed_uids
    assert pure_meta['checkpoint_hashes']==mixed_meta['checkpoint_hashes']
    retained=[uid for uid in uids if pure[uid]['parent'].seq_len==1024]
    if len(retained)<2:
        raise RuntimeError('too few full-context users for descriptive variance')
    teacher_checks={'exact_users':0,'max_abs_k_difference':0.,'max_abs_v_difference':0.,
                    'rtol':2e-5,'atol':2e-5}
    for uid in uids:
        assert pure_facts[str(uid)]==mixed_facts[str(uid)], f'terminal events differ for {uid}'
        assert pure[uid]['query_delta']==mixed[uid]['query_delta']
        assert pure[uid]['teacher'].seq_len==mixed[uid]['teacher'].seq_len
        exact=True
        for field in ('k','v'):
            a,b=getattr(pure[uid]['teacher'],field),getattr(mixed[uid]['teacher'],field)
            exact &= torch.equal(a,b)
            difference=float((a-b).abs().max())
            teacher_checks[f'max_abs_{field}_difference']=max(teacher_checks[f'max_abs_{field}_difference'],difference)
            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-5)
        teacher_checks['exact_users']+=int(exact)
    block=current.blocks[0]
    if block.block_variant!='legacy' or block.gating!='silu_gate':
        raise ValueError('this diagnostic implements the actual frozen legacy gate only')
    ids=common_candidates[None].to(device)
    embedding=current.embed_query_tokens(ids,torch.zeros(1,device=device))
    normalized=block.norm(embedding)
    query,new_k,new_v=block.attn._project(normalized)
    gate=F.silu(block.gate_proj(normalized))
    self_heads=block.attn._activate((query*new_k).sum(-1,keepdim=True)*block.attn.scale)*new_v
    calibration_root=ROOT/'results/read_correction_2026_09/v5/population_run/calibration/query_only/pure/cross_phi_joint8/c512'
    calibration=calibration_root/scale/edge_name(edge)
    artifact=torch.load(calibration/'calibration.pt',map_location='cpu',weights_only=False)
    layer=artifact['modules'][0]
    correction=QueryFeatureCorrection(**layer['config']).to(device)
    correction.load_state_dict(layer['state_dict'])
    prediction=correction(query,None,None,torch.tensor([1024],device=device))

    def project_delta(read):
        flat=read.transpose(1,2).flatten(2)
        # Difference of two out-projections cancels any bias exactly.
        return F.linear(flat,block.attn.out_proj.weight,None)*gate

    def full_update(read):
        flat=(read+self_heads).transpose(1,2).flatten(2)
        return block.attn.out_proj(flat)*gate

    targets={key:[] for key in ('pure','mixed')}
    native_hidden={key:[] for key in ('pure','mixed')}
    full_hidden=[]
    for start in range(0,len(retained),4):
        selected=retained[start:start+4]
        q=query.expand(len(selected),-1,-1,-1)
        tk=torch.cat([pure[u]['teacher'].k[0] for u in selected]).to(device)
        tv=torch.cat([pure[u]['teacher'].v[0] for u in selected]).to(device)
        teacher=history_read(block.attn,q,tk,tv)
        full_hidden.append(full_update(teacher).cpu())
        for name,rows in (('pure',pure),('mixed',mixed)):
            keys=torch.cat([rows[u]['parent'].k[0] for u in selected]).to(device)
            values=torch.cat([rows[u]['parent'].v[0] for u in selected]).to(device)
            native=history_read(block.attn,q,keys,values)
            targets[name].append((teacher-native).cpu())
            native_hidden[name].append(full_update(native).cpu())
    targets={key:torch.cat(parts) for key,parts in targets.items()}
    native_hidden={key:torch.cat(parts) for key,parts in native_hidden.items()}
    hidden_targets={key:project_delta(value.to(device)).cpu() for key,value in targets.items()}
    full_hidden=torch.cat(full_hidden)
    prediction=prediction.cpu();hidden_prediction=project_delta(prediction.to(device)).cpu()
    statistics={}
    state_info=mixed_meta['mixed_states']
    buckets={'all':list(range(len(retained)))}
    for band in cfg['mixed_append_targets']:
        buckets[f'append_target_{band}']=[i for i,u in enumerate(retained) if state_info[str(u)]['append_target']==band]
    for bucket,indices in buckets.items():
        if not indices:
            continue
        summary={'users':len(indices),'actual_append_distribution':distribution([state_info[str(retained[i])]['actual_append'] for i in indices]),
            'inherited_count_distribution':distribution([state_info[str(retained[i])]['inherited_count'] for i in indices])}
        for space,values,pred in (('read',targets,prediction),('hidden_update',hidden_targets,hidden_prediction)):
            summary[space]={name:state_statistics(value[indices],pred,full_hidden[indices] if space=='hidden_update' else None)
                for name,value in values.items()}
            summary[space]['paired_change']=paired_statistics(values['pure'][indices],values['mixed'][indices])
            combined=torch.cat((values['pure'][indices],values['mixed'][indices]))
            summary[space]['pooled_states']=state_statistics(combined,pred,
                torch.cat((full_hidden[indices],full_hidden[indices])) if space=='hidden_update' else None)
        summary['hidden_native_update_rms']={key:energy(value[indices])**.5 for key,value in native_hidden.items()}
        statistics[bucket]=summary
    per_user=[]
    for i,uid in enumerate(retained):
        item={'uid':uid,**state_info[str(uid)],'original_pure_query_delta':pure[uid]['query_delta']}
        for space,values,pred in (('read',targets,prediction),('hidden',hidden_targets,hidden_prediction)):
            for name,value in values.items():
                item[f'{space}_{name}_target_rms']=energy(value[i])**.5
                item[f'{space}_{name}_actual_error_rms']=energy(value[i:i+1]-pred)**.5
            item[f'{space}_paired_target_change_rms']=energy(values['mixed'][i]-values['pure'][i])**.5
        per_user.append(item)
    source_hashes=sources()
    source_hashes[str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__))
    peak_alloc=max(peak_alloc,torch.cuda.max_memory_allocated(gpu));peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved(gpu))
    return {'status':'complete','scale':scale,'edge':edge_name(edge),'gpu':gpu,'reserved_users':uids,
        'retained_full_context_users':retained,'excluded_short_history_users':[u for u in uids if u not in retained],
        'settings':{'reserved_first_users':64,'common_uniform_candidate_seed':[17,0],'candidates':common_candidates.tolist(),
                    'query_delta':0.,'history_length':1024,'layer':0,'append_targets':cfg['mixed_append_targets'],
                    'calibration_Q':'original cross_phi_joint8 C512, layer0 only','capture_settings':cfg},
        'terminal_event_facts':pure_facts,'terminal_teacher_checks':teacher_checks,
        'original_pure_query_delta_all64':distribution([pure[u]['query_delta'] for u in uids]),
        'original_pure_query_delta_retained':distribution([pure[u]['query_delta'] for u in retained]),
        'common_query_rms':energy(query)**.5,'fixed_Q512_read_prediction_rms':energy(prediction)**.5,
        'fixed_Q512_hidden_prediction_rms':energy(hidden_prediction)**.5,
        'fixed_prediction_identity':'same q, same N, same module; query-only ignores K/V, prediction tensor reused exactly',
        'statistics':statistics,'per_user_summary':per_user,'pure_capture_metadata':pure_meta,
        'mixed_capture_metadata':mixed_meta,'calibration_source':str(calibration),
        'calibration_weights_sha256':sha256(calibration/'calibration.pt'),'execution_sources':source_hashes,
        'peak_allocated_gib':peak_alloc/1024**3,'peak_reserved_gib':peak_reserved/1024**3,
        'limitations':['No feedback labels and no AUC; fixed artificial common candidates/time delta for mechanism isolation.',
            'Empirical common-q oracle uses this finite cohort mean, not an estimated population lower bound.',
            'Identification concerns layer0 under identical q,N only; later-layer queries can encode user/cache information.',
            'Cache append-target buckets use a prospectively fixed balanced UID assignment; full-context filtering uses length only.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',choices=('max','large'),required=True)
    parser.add_argument('--edge',type=int,choices=(1,4),required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    folder=OUTPUT/args.scale/edge_name(args.edge);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():
        print(json.dumps({'status':'already_complete','path':str(folder/'summary.json')}));return
    started=time.perf_counter();code=0
    (folder/'settings.json').write_text(json.dumps({'scale':args.scale,'edge':args.edge,'gpu':args.gpu,
        'users':64,'queries':32,'source_sha256':sha256(Path(__file__))},indent=2)+'\n')
    with (folder/'run.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            result=execute(args.scale,args.edge,args.gpu,folder)
            result['elapsed_seconds']=time.perf_counter()-started
            result['cpu_peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
            (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        except BaseException:
            code=1;traceback.print_exc()
    final={'exit_code':code,'elapsed_seconds':time.perf_counter()-started,'output':str(folder)}
    (folder/'exit.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)
    raise SystemExit(code)


if __name__=='__main__':
    main()
