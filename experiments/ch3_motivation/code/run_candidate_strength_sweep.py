#!/usr/bin/env python3
"""Fixed global strengths of the frozen linear256_rows migration recipe.

Each alpha initializes its own cache once, then appends natively. This is a
comparison of global migration recipes, not a read-time gate intervention.
No fitting or evaluation-label selection of strength is performed here.
"""
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

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.common import load_edge, load_models, write_json
from experiments.ch3_motivation.code import candidate_robust_paths
from experiments.ch3_motivation.code.candidate_robust_paths import replay_user
from experiments.ch3_motivation.code.fit_candidate_robust_maps import load_robust_maps
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.methods.baselines.protocol.cost import CostModel
from hstu_kvcache.common.artifacts import sha256
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROBUST=ROOT/'experiments/ch3_motivation/runs/candidate_robustness/attempt_1'
RUN=ROOT/'experiments/ch3_motivation/runs/candidate_strength/attempt_1'
BASE='linear256_rows'
STRENGTHS=(('alpha025',.25),('alpha05',.5),('alpha1',1.),('alpha2',2.),('alpha4',4.),('alpha8',8.))
SCHEMA=pa.schema([('uid',pa.uint64()),('query_timestamp',pa.int64()),('item_idx',pa.int64()),
    ('label',pa.int8()),('method',pa.string()),('score',pa.float64()),
    ('appended_events',pa.int64()),('history_rows',pa.int64()),('query_gap',pa.int64())])


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def strength_maps(linear):
    """One base linear execution and one shared delta per user; exact alpha1."""
    memo={};result={}
    for name,alpha in STRENGTHS:
        def transform(cache,ids,a=alpha):
            if a==.25:
                if memo:raise ValueError('Previous user strength callbacks did not complete')
                mapped=linear(cache)
                memo.update(source=cache,mapped=mapped,dk=mapped.k-cache.k,dv=mapped.v-cache.v)
            if memo.get('source') is not cache:raise ValueError('Strength callbacks must run once in fixed order')
            value=(memo['mapped'] if a==1 else HSTUKVCache(cache.k+a*memo['dk'],cache.v+a*memo['dv'],cache.seq_len))
            if a==8:memo.clear()
            return value
        result[name]=transform
    return result


def run(args):
    if args.gpu!=1:raise ValueError('Physical GPU1 only')
    # The high-alpha real-input probe found FP32 reduction-shape drift of
    # <=2.10e-5 in cache elements, while all scores passed the original gate
    # and same-input FP64 batch/single differences were <=1.78e-14.
    # Only this new run's cache comparison changes; scoring math, the shared
    # core, and its 1e-6 score tolerance remain unchanged.
    evidence_path=RUN/f'numeric_diagnostic/edge_{args.edge}/numeric_comparison.json'
    evidence=json.loads(evidence_path.read_text())
    evidence_config_path=Path(evidence['configuration'])
    evidence_config=json.loads(evidence_config_path.read_text())
    if not (evidence['status']=='complete' and len(evidence['users'])==8
            and evidence['score_original_tolerance_passed'] and evidence['same_input_fp64_passed']
            and sha256(evidence_config_path)==evidence['configuration_sha256']
            and evidence_config['edge']==args.edge and evidence_config['base_map_name']==BASE
            and evidence_config['method_strengths']==dict(STRENGTHS)):
        raise ValueError('Real-input score and FP64 numerical proof required')
    candidate_robust_paths.CACHE_NUMERIC_ATOL=3e-5
    plan=json.loads(args.plan.read_text())
    if (plan['physical_gpu']!=1 or plan['edges']!=[2,3] or plan['alphas']!=[v for _,v in STRENGTHS]
            or BASE not in plan['baseline']):raise ValueError('Frozen plan must specify the six linear256_rows strengths')
    population_path=Path(plan['population']);population=json.loads(population_path.read_text());fixed=population['users']
    if sha256(population_path)!=plan['population_sha256'] or len(fixed)!=7000 or len(set(fixed))!=7000:
        raise ValueError('Frozen same7000 population changed')
    source=ROOT/'experiments/ch3_motivation/inputs';role_path=source/'population.json'
    roles=json.loads(role_path.read_text())
    if (fixed!=roles['mismatch']+roles['population_sensitivity']['20261023']
            or population['source_population']['sha256']!=sha256(role_path)
            or set(fixed)&set(roles['fit']+roles['validation']+roles['development']+roles['confirmation'])):
        raise ValueError('Original source roles or reserved-role separation changed')
    binding,histories,requests=load_edge(source/f'edge_{args.edge}')
    if binding['population_sha256']!=sha256(role_path):raise ValueError('Input role binding changed')
    for key in ('history','requests'):
        if sha256(source/f'edge_{args.edge}'/binding[key]['path'])!=binding[key]['sha256']:
            raise ValueError('Frozen native input changed')
    attributes=ROBUST/f'edge_{args.edge}/manifest.json';attribute_manifest=json.loads(attributes.read_text())
    if attribute_manifest['population_sha256']!=sha256(population_path):raise ValueError('Candidate population changed')
    for key in ('definition','event_mapping','global_frequency'):
        reference=attribute_manifest[key]
        if sha256(Path(reference['path']))!=reference['sha256']:raise ValueError('Previous candidate rules/data changed')
    users=[u for u in fixed if requests.get(u)][:args.users] if args.diagnostic else fixed
    eligible=[u for u in users if requests.get(u)]
    if args.diagnostic and len(users)!=args.users:raise ValueError('Insufficient diagnostic users')
    if not args.diagnostic and len(eligible)<512:raise ValueError('At least512 observed-feedback UIDs required')
    os.environ['CUDA_VISIBLE_DEVICES']='1';torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    device=torch.device('cuda:0');parent,current,models=load_models(args.edge,device)
    checkpoint=ROBUST/f'fitting/edge_{args.edge}/robust_maps.pt'
    metadata=json.loads((checkpoint.parent/'summary.json').read_text());identity=metadata['identity']
    if (metadata['checkpoint_sha256']!=sha256(checkpoint) or identity['edge']!=args.edge
            or identity['models']!=models or identity['fit_uids']!=roles['fit'][:256]
            or identity['validation_users']!=roles['validation']
            or identity['evaluation_population_sha256']!=sha256(population_path)):
        raise ValueError('Frozen linear256_rows checkpoint/models/roles changed')
    maps=load_robust_maps(checkpoint,device);linear=maps[BASE];transforms=strength_maps(linear)
    cfg=dict(binding['settings'],history_length=1024)
    numerical_files=[Path(__file__),*[Path(__file__).with_name(name) for name in
        ('candidate_robust_paths.py','real_quality_paths.py','kv_evolution.py','fit_candidate_robust_maps.py',
         'fit_simple_kv_maps.py','simple_kv_maps.py','common.py')],
        ROOT/'src/hstu_kvcache/evaluation/trajectory.py',ROOT/'src/hstu_kvcache/models/state_transition.py',
        ROOT/'src/hstu_kvcache/models/hstu.py',ROOT/'src/hstu_kvcache/models/attention.py',
        ROOT/'src/hstu_kvcache/adaptation/reader.py',ROOT/'src/hstu_kvcache/methods/baselines/protocol/cost.py']
    sources={str(p.resolve()):sha256(p) for p in numerical_files}
    config=dict(format='candidate_strength_quality_v1',edge=args.edge,settings=cfg,users=users,
        fixed_resident_users=fixed,feedback_users=eligible,population=str(population_path.resolve()),
        population_sha256=sha256(population_path),plan=str(args.plan.resolve()),plan_sha256=sha256(args.plan),
        input_binding_sha256=sha256(source/f'edge_{args.edge}/inputs.json'),models=models,sources=sources,
        maps=str(checkpoint.resolve()),maps_sha256=sha256(checkpoint),base_map_name=BASE,
        method_strengths=dict(STRENGTHS),method_names=['reuse',*transforms,'rebuild'],
        method_order=['reuse',*transforms,'rebuild'],candidate_attribute_manifest=str(attributes.resolve()),
        candidate_attribute_manifest_sha256=sha256(attributes),diagnostic=args.diagnostic,
        check_independent=args.check_independent,block_users=args.block_users,physical_gpu=1,
        numeric_tolerances=dict(cache_atol=3e-5,score_atol=1e-6,rtol=1e-5),
        precision_evidence=str(evidence_path.resolve()),precision_evidence_sha256=sha256(evidence_path),
        fitting_scope='No new fit. Reuse frozen reserved256-user row-weighted linear coefficients; labels never select alpha.',
        execution_scope='Global cutover migration recipes, each followed by its own native causal appends; not read-time gating.',
        interpolation='old+alpha*(linear(old)-old); alpha1 uses exact original mapped cache; one shared linear/delta per UID',
        teacher='Once current cutover Rebuild then native causal appends',
        shortcut='No-feedback residents/unscored tails skipped for quality only; preparation cost retains all residents')
    args.output.mkdir(parents=True,exist_ok=True);cp=args.output/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Frozen run changed; use a new attempt')
    write_json(cp,config);frozen=digest(config);artifacts=[];stats=[];scored=score_rows=0;started=time.perf_counter()
    for offset in range(0,len(users),args.block_users):
        selected=users[offset:offset+args.block_users]
        block=args.output/f'blocks/part_{offset:06d}.parquet';manifest=block.with_suffix('.json')
        if manifest.exists():
            record=json.loads(manifest.read_text())
            if record['configuration_sha256']!=frozen or record['uids']!=selected or sha256(block)!=record['score_sha256']:
                raise ValueError('Committed strength block changed')
        else:
            rows=[];details=[];ledgers=[]
            for uid in selected:
                if not requests.get(uid):continue
                values,detail,ledger=replay_user(parent,current,transforms,histories.rows[uid],requests[uid],uid,cfg,
                    check_independent=args.check_independent)
                ledger['shared_base_linear_materializations']=1;ledger['shared_delta_subtractions']=1
                ledger['interpolation_maps_excluding_exact_alpha1']=5
                rows.extend(values);details.append(detail);ledgers.append(dict(uid=uid,operations=ledger))
            if any(sha256(Path(p))!=h for p,h in sources.items()):raise ValueError('Frozen numerical source changed')
            block.parent.mkdir(exist_ok=True);temporary=block.with_suffix('.parquet.writing')
            pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),temporary);temporary.replace(block)
            record=dict(configuration_sha256=frozen,offset=offset,uids=selected,scored_users=len(details),
                score_rows=len(rows),score_sha256=sha256(block),stats=details,ledgers=ledgers)
            write_json(manifest,record)
        stats.extend(record['stats']);scored+=record['scored_users'];score_rows+=record['score_rows']
        artifacts.append(dict(path=str(block.relative_to(args.output)),sha256=sha256(block),
            manifest=str(manifest.relative_to(args.output)),manifest_sha256=sha256(manifest)))
        progress=dict(status='running',edge=args.edge,completed_residents=offset+len(selected),residents=len(users),
            completed_feedback_users=scored,feedback_users=len(eligible),seconds=time.perf_counter()-started)
        write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    dense=CostModel(320,10,10,'torch');fused=CostModel(320,10,10,'triton')
    rebuild_all=rebuild_feedback=shared_material=0;material_all={n:0 for n,_ in STRENGTHS};material_feedback=dict(material_all)
    for uid in users:
        n=min(1024,int(np.searchsorted(histories.rows[uid][0],cfg['release_day']*86400,side='left')))
        denominator=(fused if n>=1024 else dense).full_cache(n);rebuild_all+=denominator
        if requests.get(uid):rebuild_feedback+=denominator
        base=linear.materialization_flops(n);components=2*10*n*320
        shared_material+=base+components*(1+2*5)
        for name,alpha in STRENGTHS:
            work=base+(3*components if alpha!=1 else 0);material_all[name]+=work
            if requests.get(uid):material_feedback[name]+=work
    calibration=int(metadata['per_map_calibration_flops_estimate'][BASE])
    costs=dict(new_fit_flops=0,reused_shared_calibration_flops=calibration,
        calibration_reuse_count=1,all_resident_once_rebuild_flops=int(rebuild_all),
        feedback_resident_once_rebuild_flops=int(rebuild_feedback),
        standalone_materialization_flops=material_all,feedback_standalone_materialization_flops=material_feedback,
        shared_eight_path_research_materialization_flops=int(shared_material),
        standalone_preparation_percent_once_rebuild={name:100*(calibration+work)/rebuild_all for name,work in material_all.items()},
        scope='Prior shared calibration counted once, no new training. Per-alpha standalone: one linear plus3 arithmetic ops/KV element except exact alpha1. Shared research execution: one linear, one delta, five mul+adds. Common native appends/reads are excluded, so this is preparation not full-release cost.')
    if scored!=len(eligible):raise ValueError('Incomplete feedback user coverage')
    summary=dict(status='complete',edge=args.edge,users=len(users),feedback_users=scored,score_rows=score_rows,
        artifacts=artifacts,method_names=config['method_names'],method_order=config['method_names'],
        diagnostic=args.diagnostic,quality_use_permitted=not args.diagnostic and scored>=512,
        physical_gpu=1,seconds=time.perf_counter()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cost_reference=costs,independent_check_statistics=stats if args.diagnostic else None)
    write_json(args.output/'summary.json',summary);write_json(args.output/'progress.json',dict(progress,status='complete'))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edge',type=int,choices=(2,3),required=True)
    p.add_argument('--plan',type=Path,default=RUN/'plan.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--block-users',type=int,choices=range(1,129),default=64)
    p.add_argument('--users',type=int,default=8);p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--check-independent',action='store_true')
    args=p.parse_args()
    try:run(args);write_json(args.output/'exit.json',dict(exit_code=0))
    except BaseException as error:
        write_json(args.output/'exit.json',dict(exit_code=1,error=str(error)));raise
