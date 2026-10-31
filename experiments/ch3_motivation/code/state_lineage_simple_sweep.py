#!/usr/bin/env python3
"""Direct/chain-only small fits sharing one frozen native-row capture.

Three planned settings vary width and budget only. Captures contain original
V1 and native teacher rows at the same final-cutoff prefix, never test labels.
Each chain stage consumes the actual selected predecessor map outputs.
"""
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

ROOT=Path(__file__).resolve().parents[3]
import torch
from experiments.ch3_motivation.code.common import restore_pack, write_json
from experiments.ch3_motivation.code.state_lineage_probe import (
    INPUT, fit_map, load_native, native_snapshot, row_indices,
)
from experiments.ch3_motivation.code.simple_kv_maps import head_rows
from hstu_kvcache.common.artifacts import sha256

RUN=ROOT/'experiments/ch3_motivation/runs/lineage_candidate/attempt_1'


def setup_gpu(gpu):
    if gpu!=1:raise ValueError('Physical GPU1 only')
    os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['EVOKV_ATTENTION_BACKEND']='triton'
    torch.cuda.set_device(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    return torch.device('cuda:0')


def sources():
    paths=[Path(__file__),Path(__file__).with_name('state_lineage_probe.py'),
        Path(__file__).with_name('simple_kv_maps.py'),Path(__file__).with_name('common.py'),
        ROOT/'src/hstu_kvcache/evaluation/yambda.py',ROOT/'src/hstu_kvcache/evaluation/trajectory.py',
        ROOT/'src/hstu_kvcache/models/state_transition.py',ROOT/'src/hstu_kvcache/models/hstu.py',
        ROOT/'src/hstu_kvcache/models/attention.py']
    return {str(p.resolve()):sha256(p) for p in paths}


def expected_capture(args):
    population=json.loads(args.population.read_text());roles=json.loads((INPUT/'population.json').read_text())
    fit,val=roles['fit'][:100],roles['validation']
    if (len(fit)!=100 or len(val)!=16 or len(set(fit+val))!=116 or set(population['users'])&set(fit+val)
            or population['fit_uids']!=fit or population['validation_uids']!=val):raise ValueError('Frozen roles differ')
    folder=INPUT/f'edge_{args.final_target}';bp=folder/'inputs.json';binding=json.loads(bp.read_text())
    history=folder/binding['history']['path']
    if sha256(history)!=binding['history']['sha256'] or binding['population_sha256']!=sha256(INPUT/'population.json'):
        raise ValueError('Frozen histories/role bindings differ')
    legacy=RUN/f'state/final_{args.final_target}/configuration.json';reference=json.loads(legacy.read_text())
    cutoff=int(binding['settings']['release_day'])*86400
    if (reference['cutoff']!=cutoff or reference['fit_uids']!=fit or reference['validation_users']!=val
            or reference['history_sha256']!=sha256(history) or reference['population_sha256']!=sha256(args.population)):
        raise ValueError('New capture must match the original fixed snapshot exactly')
    identity=dict(format='state_lineage_row_capture_v1',final_target=args.final_target,source_version=1,
        versions=list(range(1,args.final_target+1)),cutoff=cutoff,history_cap=1024,row_budget=128,
        fit_uids=fit,validation_users=val,population_sha256=sha256(args.population),
        fit_population_sha256=sha256(INPUT/'population.json'),input_binding_sha256=sha256(bp),
        history_path=str(history.resolve()),history_sha256=sha256(history),models=reference['models'],
        legacy_snapshot_reference_sha256=sha256(legacy),sources=sources(),torch_version=str(torch.__version__),
        row_sampling='Same128 evenly spaced rows per UID including first/last, all rows if shorter; same indices at every native version',
        labels_used=False,candidates_used=False,physical_gpu=1)
    return identity


def capture_or_load(args,device):
    identity=expected_capture(args);folder=args.cache_dir;folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'rows.pt';summary_path=folder/'summary.json';configuration=folder/'configuration.json'
    if configuration.exists() and json.loads(configuration.read_text())!=identity:raise ValueError('Frozen row capture changed')
    if checkpoint.exists():
        summary=json.loads(summary_path.read_text())
        if summary['identity']!=identity or sha256(checkpoint)!=summary['checkpoint_sha256']:raise ValueError('Cached row hash/identity differs')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        if payload['identity']!=identity:raise ValueError('Cached row payload identity differs')
        return payload,summary
    write_json(configuration,identity);started=time.perf_counter()
    histories=restore_pack(Path(identity['history_path']));all_users=identity['fit_uids']+identity['validation_users']
    rows={};lengths={}
    for version in identity['versions']:
        model,path=load_native(version,device)
        if sha256(path)!=identity['models'][str(path.relative_to(ROOT))]:raise ValueError('Native model changed')
        values=[]
        for index,uid in enumerate(all_users):
            with torch.no_grad():
                cache=native_snapshot(model,version,histories.rows[uid],identity['cutoff'])
                indices=row_indices(cache.seq_len,128);lengths[uid]=cache.seq_len
                values.append(head_rows(cache,10)[:,indices].cpu());del cache
            if (index+1)%32==0:print(json.dumps(dict(stage='capture',version=version,users=index+1)),flush=True)
        split=sum(min(128,lengths[u]) for u in identity['fit_uids']);joined=torch.cat(values,1)
        rows[version]=dict(fit=joined[:,:split].contiguous(),validation=joined[:,split:].contiguous())
        del model,values,joined;torch.cuda.empty_cache()
    if any(sha256(Path(p))!=h for p,h in identity['sources'].items()):raise ValueError('Source changed while capturing')
    payload=dict(format='state_lineage_native_rows_v1',identity=identity,rows=rows,history_lengths=lengths)
    temp=folder/'rows.pt.writing';torch.save(payload,temp);temp.replace(checkpoint)
    summary=dict(status='complete',identity=identity,checkpoint_sha256=sha256(checkpoint),
        sampled_fit_rows=rows[1]['fit'].shape[1],sampled_validation_rows=rows[1]['validation'].shape[1],
        seconds=time.perf_counter()-started,history_lengths=lengths,backbone_versions_captured=len(rows))
    write_json(summary_path,summary);print(json.dumps(dict(stage='capture_complete',seconds=summary['seconds'])),flush=True)
    return payload,summary


def fit_rows(rows,*,target,hidden,steps,batch_rows,lr,seed):
    """Only actual chain states propagate; Direct sees the original V1 rows."""
    original,original_val=rows[1]['fit'],rows[1]['validation'];previous,previous_val=original,original_val
    bank={};reports={}
    for version in range(2,target+1):
        teacher,teacher_val=rows[version]['fit'],rows[version]['validation']
        chain,report=fit_map(previous,previous,teacher,previous_val,previous_val,teacher_val,
            hidden=hidden,steps=steps,batch_rows=batch_rows,lr=lr,seed=seed,validate_every=32)
        bank[version]={'chain':chain};reports[version]={'chain':report}
        print(json.dumps(dict(stage='fit',version=version,kind='chain',selected_step=report['selected_step'],
            validation_kv_mse=report['validation_kv_mse'])),flush=True)
        if version==target:
            direct,report=fit_map(original,original,teacher,original_val,original_val,teacher_val,
                hidden=hidden,steps=steps,batch_rows=batch_rows,lr=lr,seed=seed,validate_every=32)
            bank[version]['direct']=direct;reports[version]['direct']=report
            print(json.dumps(dict(stage='fit',version=version,kind='direct',selected_step=report['selected_step'],
                validation_kv_mse=report['validation_kv_mse'])),flush=True)
        with torch.no_grad():previous=chain(previous,previous);previous_val=chain(previous_val,previous_val)
    return bank,reports


def run(args):
    device=setup_gpu(args.gpu);payload,capture=capture_or_load(args,device)
    if args.capture_only:return capture
    if args.output is None:raise ValueError('--output required for fitting')
    cache_identity=payload['identity'];source_hashes=sources()
    identity=dict(format='state_lineage_simple_sweep_fit_v1',source_version=1,final_target=args.final_target,
        versions=cache_identity['versions'],cutoff=cache_identity['cutoff'],history_cap=1024,
        fit_uids=cache_identity['fit_uids'],validation_users=cache_identity['validation_users'],
        population_sha256=cache_identity['population_sha256'],fit_population_sha256=cache_identity['fit_population_sha256'],
        input_binding_sha256=cache_identity['input_binding_sha256'],history_sha256=cache_identity['history_sha256'],
        models=cache_identity['models'],sources=source_hashes,row_budget=128,hidden=args.hidden,steps=args.steps,
        batch_rows=args.batch_rows,lr=args.lr,seed=args.seed,torch_version=str(torch.__version__),
        native_row_cache=str((args.cache_dir/'rows.pt').resolve()),native_row_cache_sha256=capture['checkpoint_sha256'],
        row_cache_identity=cache_identity,physical_gpu=1,
        objective='Raw native teacher KV MSE; no labels/candidates used in fitting or checkpoint selection',
        input_rules={'direct':'original V1 duplicated','chain':'actual selected predecessor chain output duplicated'},
        method_scope='Only chain stages and one final direct map; no anchor or additional module',
        snapshot_scope='Same final-cutoff prefix at every model version; held fixed for quality, cached-prefix-gap evaluation')
    args.output.mkdir(parents=True,exist_ok=True);cp=args.output/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=identity:raise ValueError('Frozen fit configuration changed')
    write_json(cp,identity)
    if (args.output/'maps.pt').exists():raise ValueError('Completed checkpoint exists; do not refit')
    started=time.perf_counter();rows={v:{kind:value.to(device) for kind,value in values.items()} for v,values in payload['rows'].items()}
    bank,reports=fit_rows(rows,target=args.final_target,hidden=args.hidden,steps=args.steps,batch_rows=args.batch_rows,lr=args.lr,seed=args.seed)
    if any(sha256(Path(p))!=h for p,h in source_hashes.items()):raise ValueError('Numerical source changed during fit')
    checkpoint=dict(format='state_lineage_maps_v1',pairs=100,width=64,hidden=args.hidden,identity=identity,
        maps={v:{kind:{name:value.cpu() for name,value in model.state_dict().items()} for kind,model in stage.items()} for v,stage in bank.items()})
    temp=args.output/'maps.pt.writing';torch.save(checkpoint,temp);temp.replace(args.output/'maps.pt')
    summary=dict(status='complete',identity=identity,checkpoint_sha256=sha256(args.output/'maps.pt'),stages=reports,short_stages={},
        sampled_fit_rows=capture['sampled_fit_rows'],sampled_validation_rows=capture['sampled_validation_rows'],
        history_lengths=capture['history_lengths'],seconds=time.perf_counter()-started,physical_gpu=1,
        backbone_capture_reused=True,backbone_capture_seconds=capture['seconds'],quality_auc_computed=False)
    write_json(args.output/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final-target',type=int,choices=(3,4),required=True)
    p.add_argument('--population',type=Path,default=RUN/'population.json');p.add_argument('--cache-dir',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--capture-only',action='store_true')
    p.add_argument('--hidden',type=int,choices=(32,64,128),default=64);p.add_argument('--steps',type=int,default=256)
    p.add_argument('--batch-rows',type=int,default=512);p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--gpu',type=int,default=1)
    args=p.parse_args()
    if args.cache_dir is None:args.cache_dir=RUN/f'state/simple_sweep/row_cache/final_{args.final_target}'
    try:
        run(args)
        write_json((args.cache_dir if args.capture_only else args.output)/'exit.json',dict(exit_code=0))
    except BaseException as error:
        if args.output is not None:write_json(args.output/'exit.json',dict(exit_code=1,error=str(error)))
        raise
