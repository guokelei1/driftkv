#!/usr/bin/env python3
"""Paired, diagnostic-only interventions on retained query correction weights."""
import argparse
import copy
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from evaluate_yambda500m_foundation_raw import load_histories,load_model
from hstu_kvcache.evaluation.binary_metrics import binary_metrics
from read_correction_v5.common import PANEL_ROOT,OUTPUT,configuration,edge_name,sha256,sources,write_json
from read_correction_v5.scoring import load_policies,score_unit
from selective_recompute_2026_09.scheduling import ordered_uids


def calibration(scale,edge,budget,raw=False):
    feature='cross_phi' if raw else 'cross_phi_joint8'
    suffix=Path('query_only/pure')/feature/f'c{budget}'/scale/edge_name(edge)
    for root in [OUTPUT/'population_run/calibration',OUTPUT/'resource_canary',OUTPUT]:
        if (root/suffix/'calibration.json').exists():
            return root/suffix
    raise FileNotFoundError(str(suffix))


def policies_for(args,device):
    entries=[]
    for budget in args.budgets:
        for raw in [False,True]:
            entries.append(dict(tag=f'c{budget}_'+('raw' if raw else 'original'),method='query_only',
                variant='diagnostic',calibration_dir=str(calibration(args.scale,args.edge,budget,raw))))
    for entry in entries:
        folder=Path(entry['calibration_dir']);record=json.loads((folder/'calibration.json').read_text())
        if record['status']!='complete' or sha256(folder/'calibration.pt')!=record['weights_sha256']:
            raise RuntimeError('retained correction weights changed')
    originals=load_policies(entries,device)
    policies=[];recipes=[]
    for original in originals:
        tag=original['tag']
        if tag.endswith('_raw'):
            if args.original_only:continue
            policies.append(original);recipes.append(dict(tag=tag,operation='retained pre-joint ridge',source=original['calibration_dir']))
            continue
        budget=int(tag.split('_')[0][1:])
        recipes.append(dict(tag=tag,operation='unchanged retained joint8',source=original['calibration_dir']))
        policies.append(original)
        if args.original_only:continue
        for operation in ['alpha025','alpha050','early_only','late_only','drop_constant']:
            modules=copy.deepcopy(original['modules'])
            changed=[]
            with torch.no_grad():
                for layer,module in enumerate(modules):
                    if operation=='drop_constant':
                        # A training-derived input standard deviation, not a label threshold.
                        mask=module.input_scale<=1e-5
                        if module.feature_mode!='cross_phi':raise ValueError('cross-head diagnosis expected')
                        module.weight[mask]=0
                        changed.append(int(mask.sum()))
                    else:
                        factor={'alpha025':.25,'alpha050':.5,'early_only':float(layer<len(modules)//2),
                            'late_only':float(layer>=len(modules)//2)}[operation]
                        module.weight.mul_(factor);module.bias.mul_(factor)
                        changed.append(factor)
            variant=f'c{budget}_{operation}'
            policies.append({**original,'tag':variant,'modules':modules})
            recipes.append(dict(tag=variant,operation=operation,per_layer=changed,source=original['calibration_dir']))
    for recipe in recipes:
        folder=Path(recipe['source'])
        recipe['calibration_sha256']=sha256(folder/'calibration.json')
        recipe['weights_sha256']=sha256(folder/'calibration.pt')
    return policies,recipes


def run(args):
    started=time.perf_counter();cfg=configuration()
    output=args.output.resolve()
    if (output/'summary.json').exists():raise RuntimeError('diagnostic result already exists; do not overwrite')
    panel_path=PANEL_ROOT/args.scale/edge_name(args.edge)/'binding.json'
    panel=json.loads(panel_path.read_text())
    requests_path=args.requests.resolve() if args.requests else panel_path.parent/'evaluation_requests.parquet'
    by_user=defaultdict(list)
    for row in pq.read_table(requests_path).to_pylist():by_user[int(row['uid'])].append(row)
    selected=sorted(by_user,key=lambda uid:hashlib.sha256(f'read-correction-v2-probe:17:{uid}'.encode()).digest())
    if args.limit_users:selected=selected[:args.limit_users]
    uids=ordered_uids(by_user,max_length=cfg['history_length'],uids=selected)
    for budget in args.budgets:
        record=json.loads((calibration(args.scale,args.edge,budget)/'calibration.json').read_text())
        if set(uids).intersection(record['uids']+record['validation_uids']):raise RuntimeError('fitting/evaluation users overlap')
        if record['panel_binding_sha256']!=sha256(panel_path):raise RuntimeError('retained fit source panel changed')
    hashes=sources();hashes[str(Path(__file__).relative_to(ROOT))]=sha256(__file__)
    write_json(output/'inputs.json',dict(role='diagnostic-only interventions; not a selected method or formal frontier',
        panel=dict(path=str(panel_path),sha256=sha256(panel_path)),requests=dict(path=str(requests_path),sha256=sha256(requests_path)),
        uids=uids,settings=cfg,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},execution_sources=hashes))
    os.environ['EVOKV_ATTENTION_BACKEND']=cfg['attention_backend']
    torch.set_num_threads(cfg['torch_threads']);pa.set_cpu_count(cfg['history_threads'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_device(args.gpu)
    device=torch.device(f'cuda:{args.gpu}')
    free,total=torch.cuda.mem_get_info(device)
    if free/total<cfg['initial_free_fraction']:raise RuntimeError('insufficient initial GPU reserve')
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'],device);torch.cuda.reset_peak_memory_stats(device)
    parent,pp=load_model(ROOT/panel['sources']['parent']['path'],device)
    current,cp=load_model(ROOT/panel['sources']['current']['path'],device)
    if pp['config']!=cp['config']:raise RuntimeError('model architectures differ')
    parent.requires_grad_(False);current.requires_grad_(False)
    dataset_path=ROOT/panel['sources']['dataset']['path'];dataset=json.loads(dataset_path.read_text())
    known=int(cp.get('known_vocab_size',dataset['foundation_items']))
    del pp,cp
    policies,recipes=policies_for(args,device)
    write_json(output/'recipes.json',recipes)
    history=load_histories(uids,dataset_path=dataset_path,known_vocab_size=known,
        oov_buckets=current.cfg.num_items-known,start_timestamp=int(panel['cutover']),
        end_timestamp=int(panel['days'][1])*86400,max_history=cfg['history_length'],threads=cfg['history_threads'])
    print(json.dumps(dict(status='loaded',scale=args.scale,edge=args.edge,users=len(uids),policies=len(policies),seconds=time.perf_counter()-started)),flush=True)
    records,hist,controls=score_unit(current,parent,history,by_user,uids,int(panel['cutover']),policies,cfg,args.scale,device,verify=True)
    summaries=[]
    for policy in policies:
        rows=sorted(records[policy['tag']],key=lambda row:row['request_id'])
        path=output/(policy['tag']+'.parquet');pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd')
        labels=np.asarray([r['label'] for r in rows]);full=np.asarray([r['full_logit'] for r in rows]);reuse=np.asarray([r['reuse_logit'] for r in rows]);actual=np.asarray([r['hstu_logit'] for r in rows])
        f,r,a=[binary_metrics(labels,s)['ROC_AUC'] for s in [full,reuse,actual]]
        summaries.append(dict(tag=policy['tag'],requests=len(rows),full_auc=f,reuse_auc=r,auc=a,
            recovery_percent=100*(a-r)/(f-r) if f!=r else None,delta_auc_vs_reuse=a-r,
            logit_mse_to_full=float(np.mean((actual-full)**2)),mean_correction=float(np.mean(actual-reuse)),
            scores=dict(path=str(path),sha256=sha256(path))))
    result=dict(status='complete',role='paired mechanism diagnosis; no cost frontier or per-edge deployment selection',
        scale=args.scale,edge=edge_name(args.edge),users=len(uids),requests=summaries[0]['requests'],points=summaries,
        controls=controls,histograms=hist,elapsed_seconds=time.perf_counter()-started,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
    write_json(output/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',choices=['medium','large','max'],required=True)
    parser.add_argument('--edge',type=int,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--budgets',type=int,nargs='+',default=[128,512])
    parser.add_argument('--limit-users',type=int,default=256)
    parser.add_argument('--requests',type=Path)
    parser.add_argument('--original-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
