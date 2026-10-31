"""Fixed7000-user real-feedback replay for cheap static-map robustness."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import ROOT, load_edge, load_models, write_json
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.simple_kv_maps import load_maps
from experiments.ch3_motivation.code.probe_cost import cached_preparation_flops
from hstu_kvcache.common.artifacts import sha256
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.models import HSTUKVCache
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

SCHEMA=pa.schema([('uid',pa.uint64()),('query_timestamp',pa.int64()),('item_idx',pa.int64()),
    ('label',pa.int8()),('method',pa.string()),('score',pa.float64()),
    ('appended_events',pa.int64()),('history_rows',pa.int64()),('query_gap',pa.int64())])


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def run(args):
    from experiments.ch3_motivation.code.candidate_robust_paths import replay_user
    from experiments.ch3_motivation.code.fit_candidate_robust_maps import load_robust_maps
    if args.gpu!=1:raise ValueError('physical GPU1 only')
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    torch.cuda.set_device(0);device=torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    population=json.loads(args.population.read_text());fixed=population['users']
    if len(fixed)!=7000 or len(set(fixed))!=7000:raise ValueError('fixed7000 development cohort required')
    source=ROOT/'experiments/ch3_motivation/inputs'
    binding,histories,requests=load_edge(source/f'edge_{args.edge}')
    for name in ['history','requests']:
        if sha256(source/f'edge_{args.edge}'/binding[name]['path'])!=binding[name]['sha256']:
            raise ValueError('native history/observations changed')
    if sha256(source/'population.json')!=binding['population_sha256']:
        raise ValueError('original roles changed')
    if population['source_population']['sha256']!=binding['population_sha256']:
        raise ValueError('new cohort belongs to different original roles')
    roles=json.loads((source/'population.json').read_text())
    if fixed!=roles['mismatch']+roles['population_sensitivity']['20261023']:
        raise ValueError('population differs from the complete original mismatch+sensA cohort')
    reserved=set(roles['fit']+roles['validation']+roles['development']+roles['confirmation'])
    if set(fixed)&reserved:raise ValueError('evaluation intersects reserved roles')
    users=([u for u in fixed if requests.get(u)][:args.users] if args.diagnostic else fixed)
    eligible=[u for u in users if requests.get(u)]
    if args.diagnostic:
        if len(users)!=args.users:raise ValueError('not enough numerical users')
    elif len(eligible)<512:raise ValueError('at least512 actual feedback users required')
    f_path=ROOT/f'experiments/ch3_motivation/runs/insight_mechanisms/attempt_1/fitting/edge_{args.edge}/f.pt'
    f,f_meta=load_variant(f_path,device)
    parent,current,models=load_models(args.edge,device)
    identity=f_meta['identity'];f.eval().requires_grad_(False)
    if f.stage!='f' or identity['edge']!=args.edge or identity['models']!=models:
        raise ValueError('fixed F or backbone mismatch')
    fit_binding=ROOT/f'experiments/ch3_motivation/inputs_feedback_pool/fit256/edge_{args.edge}/inputs.json'
    if sha256(fit_binding)!=identity['input_binding_sha256']:raise ValueError('F calibration input changed')
    old_maps_path=ROOT/f'experiments/ch3_motivation/runs/real_quality_angles/attempt_1/simple_maps/edge_{args.edge}/simple_kv_maps.pt'
    old_maps=load_maps(old_maps_path,device)
    old_meta=json.loads((old_maps_path.parent/'summary.json').read_text())
    if old_meta['checkpoint_sha256']!=sha256(old_maps_path) or old_meta['identity']['models']!=models:
        raise ValueError('old exact linear checkpoint changed')
    new_maps=load_robust_maps(args.maps,device)
    expected={'linear100_rows','linear100_users','linear256_rows','linear256_users'}
    if set(new_maps)!=expected:raise ValueError('unexpected robustness map set')
    exact_reference=new_maps.pop('linear100_rows')
    if any(not torch.equal(getattr(exact_reference,name),getattr(old_maps['kv_linear'],name))
           for name in ['xmean','xscale','weight','ymean']):
        raise ValueError('original linear100_rows must retain its exact frozen coefficients')
    new_meta=json.loads((args.maps.parent/'summary.json').read_text())
    new_identity=new_meta['identity']
    if (new_identity['edge']!=args.edge or new_identity['models']!=models
            or new_identity['fit_uids']!=roles['fit'][:256]
            or new_identity['validation_users']!=roles['validation']
            or new_identity['evaluation_population_sha256']!=sha256(args.population)
            or new_meta['checkpoint_sha256']!=sha256(args.maps)):
        raise ValueError('new calibration does not share native models/reserved roles')
    memo={}
    def f1(cache,ids):
        value=f.prepare_cache(current,cache,ids);memo['f1']=value;return value
    def f2(cache,ids):
        value=memo.pop('f1')
        return HSTUKVCache(cache.k+2*(value.k-cache.k),cache.v+2*(value.v-cache.v),cache.seq_len)
    transforms={'f1':f1,'f2':f2,'linear100_rows':lambda cache,ids:old_maps['kv_linear'](cache)}
    transforms.update({name:(lambda cache,ids,t=transform:t(cache)) for name,transform in new_maps.items()})
    if len(transforms)!=6:raise ValueError('six requested static maps only')
    cfg=dict(binding['settings'],history_length=1024)
    numerical_files=[Path(__file__),*[Path(__file__).with_name(name) for name in
        ['candidate_robust_paths.py','real_quality_paths.py','kv_evolution.py','fit_candidate_robust_maps.py','simple_kv_maps.py','fit_simple_kv_maps.py',
         'common.py','probe_variants.py','probes.py','probe_cost.py']],
        ROOT/'src/hstu_kvcache/evaluation/trajectory.py',ROOT/'src/hstu_kvcache/models/state_transition.py',
        ROOT/'src/hstu_kvcache/models/hstu.py',ROOT/'src/hstu_kvcache/models/attention.py',
        ROOT/'src/hstu_kvcache/adaptation/reader.py',ROOT/'src/hstu_kvcache/methods/baselines/protocol/cost.py']
    sources={str(p.resolve()):sha256(p) for p in numerical_files}
    config=dict(format='candidate_robustness_quality_v1',edge=args.edge,settings=cfg,
        users=users,fixed_resident_users=fixed,feedback_users=eligible,
        population=str(args.population.resolve()),population_sha256=sha256(args.population),
        input_binding_sha256=sha256(source/f'edge_{args.edge}/inputs.json'),sources=sources,models=models,
        checkpoint=str(f_path),checkpoint_sha256=sha256(f_path),maps=str(args.maps.resolve()),
        maps_sha256=sha256(args.maps),old_linear_checkpoint_sha256=sha256(old_maps_path),
        simple_map_metadata=new_meta,method_order=['reuse',*transforms,'rebuild'],method_names=['reuse',*transforms,'rebuild'],
        diagnostic=args.diagnostic,check_independent=args.check_independent,block_users=args.block_users,
        physical_gpu=1,hardware=torch.cuda.get_device_name(0),
        fitting_scope='Reserved100/256 fit users,16 validation; no feedback labels or test outcomes fit maps',
        teacher='Once current rebuild at cutover then independent native causal appends',
        quality_scope='All original known-item observed real feedback; no synthetic negatives; same keys across methods',
        shortcut='No-feedback residents and unscored tails skipped only for quality; cost reference uses all residents separately')
    args.output.mkdir(parents=True,exist_ok=True);cpath=args.output/'configuration.json'
    if cpath.exists() and json.loads(cpath.read_text())!=config:raise ValueError('frozen run changed; new attempt required')
    write_json(cpath,config);fingerprint=digest(config)
    artifacts=[];stats=[];started=time.perf_counter();scored=score_rows=0
    for offset in range(0,len(users),args.block_users):
        selected=users[offset:offset+args.block_users]
        block=args.output/f'blocks/part_{offset:06d}.parquet';manifest=block.with_suffix('.json')
        if manifest.exists():
            record=json.loads(manifest.read_text())
            if record['configuration_sha256']!=fingerprint or record['uids']!=selected or sha256(block)!=record['score_sha256']:
                raise ValueError('committed block changed')
        else:
            rows=[];details=[];ledgers=[]
            for uid in selected:
                if not requests.get(uid):continue
                values,detail,ledger=replay_user(parent,current,transforms,histories.rows[uid],requests[uid],uid,cfg,
                    check_independent=args.check_independent)
                rows.extend(values);details.append(detail);ledgers.append(dict(uid=uid,operations=ledger))
            if any(sha256(Path(p))!=h for p,h in sources.items()):raise ValueError('numerical source changed before commit')
            block.parent.mkdir(exist_ok=True);temporary=block.with_suffix('.parquet.writing')
            pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),temporary);temporary.replace(block)
            record=dict(configuration_sha256=fingerprint,offset=offset,uids=selected,scored_users=len(details),
                score_rows=len(rows),score_sha256=sha256(block),stats=details,ledgers=ledgers)
            write_json(manifest,record)
        stats.extend(record['stats']);scored+=record['scored_users'];score_rows+=record['score_rows']
        artifacts.append(dict(path=str(block.relative_to(args.output)),sha256=sha256(block),
            manifest=str(manifest.relative_to(args.output)),manifest_sha256=sha256(manifest)))
        progress=dict(status='running',edge=args.edge,completed_residents=offset+len(selected),residents=len(users),
            completed_feedback_users=scored,feedback_users=len(eligible),seconds=time.perf_counter()-started)
        write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    cm=CostModel(320,10,10,'triton');dense=CostModel(320,10,10,'torch')
    rebuild_all=rebuild_feedback=0;material_all={m:0 for m in transforms};material_feedback=dict(material_all)
    for uid in users:
        n=min(1024,int(np.searchsorted(histories.rows[uid][0],cfg['release_day']*86400,side='left')))
        value=(cm if n>=1024 else dense).full_cache(n);rebuild_all+=value
        if requests.get(uid):rebuild_feedback+=value
        for name in transforms:
            if name in ['f1','f2']:
                work=cached_preparation_flops(f,n)+(3*n*10*2*320 if name=='f2' else 0)
            else:
                transform=old_maps['kv_linear'] if name=='linear100_rows' else new_maps[name]
                work=transform.materialization_flops(n)
            material_all[name]+=work
            if requests.get(uid):material_feedback[name]+=work
    costs={}
    for name in transforms:
        calibration=(f_meta['cumulative_calibration_flops_estimate'] if name in ['f1','f2']
            else old_meta['total_calibration_flops_estimate'] if name=='linear100_rows'
            else new_meta['total_calibration_flops_estimate'])
        costs[name]=dict(shared_calibration_flops=int(calibration),all_resident_materialization_flops=material_all[name],
            feedback_materialization_flops=material_feedback[name],
            all_resident_preparation_percent_once_rebuild=100*(calibration+material_all[name])/rebuild_all,
            feedback_preparation_percent_once_rebuild=100*(calibration+material_feedback[name])/rebuild_feedback)
    summary=dict(status='complete',edge=args.edge,users=len(users),feedback_users=scored,score_rows=score_rows,
        method_order=config['method_order'],artifacts=artifacts,diagnostic=args.diagnostic,
        quality_use_permitted=not args.diagnostic and scored>=512,physical_gpu=1,
        seconds=time.perf_counter()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cost_reference=dict(all_resident_once_rebuild_flops=rebuild_all,feedback_resident_once_rebuild_flops=rebuild_feedback,
            persistent_method_preparation=costs,scope='Shared calibration + once materialization. Each new map conservatively charged entire combined fitter. Common native append/read shapes cancel; no throughput claim.'),
        independent_check_statistics=stats if args.diagnostic else None)
    write_json(args.output/'summary.json',summary);write_json(args.output/'progress.json',dict(progress,status='complete'))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edge',type=int,choices=[2,3],required=True)
    p.add_argument('--population',type=Path,required=True);p.add_argument('--maps',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--block-users',type=int,default=64);p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--users',type=int,default=8);p.add_argument('--check-independent',action='store_true')
    a=p.parse_args()
    try:run(a);write_json(a.output/'exit.json',dict(exit_code=0))
    except BaseException as e:
        write_json(a.output/'exit.json',dict(exit_code=1,error=str(e)));raise
