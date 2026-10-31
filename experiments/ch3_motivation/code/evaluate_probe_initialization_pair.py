#!/usr/bin/env python3
"""Matched standard/balanced h test from actual PhaseB checkpoints.

Reuse a completed point only when architecture, every parameter/buffer and its
standalone calibration ledger match exactly. Otherwise evaluate it. Native
anchors were freshly measured in this goal and are reused only after the
existing shard/context/hash audit. Each newly scored point has its own causal
state trajectory and full deployment ledger.
"""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
from pathlib import Path
import argparse,csv,json,os,sys
ROOT=Path(__file__).resolve().parents[3]
import numpy as np
import torch
from experiments.ch3_motivation.code import (
    plan_probe_quality as quality, plan_selected_development as planned,
)
from experiments.ch3_motivation.code.analyze_probe_quality import (
    ShardLoader, _context, _same_events, _interval, analyze,
)
from experiments.ch3_motivation.code.common import load_models, write_json
from experiments.ch3_motivation.code.evaluate_probes import run as evaluate
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256


def prepare(args):
    plan=json.loads(args.followup_plan.read_text())
    jobs={name:next(j for j in plan['jobs'] if j['edge']==args.edge and j['name']==name) for name in ('reference','balanced_h_init')}
    audits={};payloads={};loader=ShardLoader()
    for mode,job in jobs.items():
        audit_job=dict(job['expected_identity']);audit_job.update(job)
        for stage in ('f','fg','fgh'):
            path=Path(job['output'])/f'{stage}.pt'
            checked=quality.audit_weight(path,audit_job,plan['input_bindings'][str(args.edge)],stage)
            x=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            if x['metadata']['identity']!=job['expected_identity']:raise ValueError('initialization source identity differs')
            audits[mode,stage]=dict(checkpoint=str(path),**checked);payloads[mode,stage]=x
    a,b=[jobs[name]['expected_identity'] for name in ('reference','balanced_h_init')]
    if ({k:v for k,v in a.items() if k not in ('h_initialization','h_initialization_spec')}!=
            {k:v for k,v in b.items() if k not in ('h_initialization','h_initialization_spec')}):
        raise ValueError('paired fitting recipes differ beyond h initialization')
    def same(left,right):
        if (left['format']!=right['format'] or left['variant']!=right['variant'] or left['config']!=right['config'] or
                left['state_dict'].keys()!=right['state_dict'].keys() or
                any(not torch.equal(value,right['state_dict'][key]) for key,value in left['state_dict'].items())):return False
        fields=('best_step','completed_steps','operation_counts','arithmetic','cumulative_calibration_flops_estimate')
        return all(left['metadata'][name]==right['metadata'][name] for name in fields)
    if any(not same(payloads['reference',stage],payloads['balanced_h_init',stage]) for stage in ('f','fg')):
        raise ValueError('standard/balanced predecessors or calibration ledgers differ')
    population=json.loads((planned.INPUTS/'population.json').read_text());users=population['development']
    if len(users)!=512:raise ValueError('use the fixed development512')
    old={};reused={};todo=[]
    for stage in ('f','fg','fgh'):
        path=args.reference_quality/f'edge_{args.edge}_{stage}/summary.json';point=loader.load(path);old[stage]=point
        identity=point['summary']['identity']
        if (identity['users']!=users or identity['edge']!=args.edge or identity['models']!=a['models'] or
                identity['input_binding_sha256']!=a['input_binding_sha256'] or identity['sources']!=planned.runtime_sources() or
                identity['backend']!='projected' or identity['scenario'] is not None):raise ValueError('reference quality context differs')
        source=torch.load(identity['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        if same(source,payloads['reference',stage]):reused['reference_'+stage]=point['path']
        else:todo.append(dict(name='reference_'+stage,checkpoint=audits['reference',stage]['checkpoint']))
    bal=payloads['balanced_h_init','fgh'];std=payloads['reference','fgh']
    if same(bal,std) and 'reference_fgh' in reused:reused['balanced_fgh']=reused['reference_fgh']
    else:todo.append(dict(name='balanced_fgh',checkpoint=audits['balanced_h_init','fgh']['checkpoint']))
    frozen={str(args.followup_plan.resolve()):sha256(args.followup_plan)}
    for key,row in audits.items():frozen[row['checkpoint']]=row['checkpoint_sha256']
    frozen.update(loader.hashes)
    for name,digest in planned.sources().items():
        path=Path(name);path=(path if path.is_absolute() else ROOT/path).resolve()
        if str(path) in frozen and frozen[str(path)]!=digest:raise ValueError('initialization source binding conflict')
        frozen[str(path)]=digest
    for path in (Path(__file__),ROOT/'experiments/ch3_motivation/code/evaluate_probes.py',ROOT/'experiments/ch3_motivation/code/analyze_probe_quality.py',ROOT/'src/hstu_kvcache/evaluation/statistics.py'):
        frozen[str(path)]=sha256(path)
    result=dict(status='prepared_actual_initialization_pair',edge=args.edge,quality_users=512,reused_summaries=reused,
        actual_weight_audit=list(audits.values()),jobs=todo,frozen_sha256=frozen,reference_quality=str(args.reference_quality),
        predecessors_exact=True,scope='one shared matched F/FG prefix, standard and balanced H; all fresh-goal native anchor shards verified; no independent or three-seed confirmation')
    return result,old,loader


def run(args):
    prepared,old,loader=prepare(args);args.output.mkdir(parents=True,exist_ok=True)
    existing=args.output/'plan.json'
    if existing.exists() and json.loads(existing.read_text())!=prepared:raise ValueError('initialization quality plan changed')
    write_json(existing,prepared)
    if args.check:
        print(json.dumps(dict(status=prepared['status'],edge=args.edge,reused=len(prepared['reused_summaries']),new_points=len(prepared['jobs']))));return
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);torch.cuda.set_device(0);device=torch.device('cuda:0')
    if torch.cuda.mem_get_info()[0]<32*1024**3:raise ValueError('need measured32GiB headroom for concurrent pair evaluation')
    parent,current,models=load_models(args.edge,device)
    anchor=old['f'];context=_context(anchor['summary']['identity'])
    if models!=context['models']:raise ValueError('backbone files differ from reference native anchors')
    source=dict(anchor['summary']['anchor_source'],scope='verified completed native anchors from identical context in this same goal',
        source_summary=anchor['path'],source_summary_sha256=sha256(Path(anchor['path'])),verified_cross_process_reuse=True)
    # anchor_execution is unused by this frozen evaluator's correction ledger;
    # that ledger is regenerated from the new point's actual events/trajectory.
    scores=anchor['anchors']['reuse']+anchor['anchors']['rebuild']
    shared=dict(model_key=(args.edge,str(device),args.gpu),models=(parent,current,models),
        anchors={source['key']:(scores,None,source)})
    summaries=dict(prepared['reused_summaries'])
    for job in prepared['jobs']:
        values=argparse.Namespace(input_root=planned.INPUTS,edge=args.edge,gpu=args.gpu,checkpoint=Path(job['checkpoint']),
            output=args.output/job['name'],panel='development',population_seed=20261023,population_file=None,backend='projected')
        result=evaluate(values,shared_context=shared)
        if result['anchor_source']['path']!=source['path']:raise ValueError('verified native anchors were not reused')
        summaries[job['name']]=str(values.output/'summary.json')
    reports={}
    for mode,h_name in (('standard','reference_fgh'),('balanced','balanced_fgh')):
        paths=[Path(summaries['reference_f']),Path(summaries['reference_fg']),Path(summaries[h_name])]
        reports[mode]=analyze(paths,replicates=1000,seed=17051)
        write_json(args.output/f'analysis_{mode}.json',reports[mode])
    left=loader.load(Path(summaries['reference_fgh']));right=loader.load(Path(summaries['balanced_fgh']))
    if (_context(left['summary']['identity'])!=_context(right['summary']['identity']) or
            not _same_events(left['rows'],right['rows']) or left['anchors']!=right['anchors']):raise ValueError('H pair scores/context differ')
    users=np.asarray(sorted(context['users']),dtype=np.int64);base=left['rows']
    labels=np.asarray([row['label'] for row in base],dtype=np.float64)
    indices=np.searchsorted(users,np.asarray([row['uid'] for row in base],dtype=np.int64));rng=np.random.default_rng(17051)
    weights=np.r_[np.ones((1,512),dtype=np.uint16),np.asarray([np.bincount(rng.integers(512,size=512),minlength=512) for _ in range(1000)],dtype=np.uint16)]
    values=[weighted_auc(np.asarray([row['score'] for row in point['rows']]),labels,indices,weights) for point in (left,right)]
    delta=values[1]-values[0];ci=_interval(delta[1:])
    contrast=dict(contrast='balanced_minus_standard_h_auc_and_h_increment',auc_difference=float(delta[0]),auc_ci95_low=ci[0],auc_ci95_high=ci[1],valid_replicates=int(np.isfinite(delta[1:]).sum()))
    with (args.output/'contrast.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(contrast));writer.writeheader();writer.writerow(contrast)
    for path,digest in prepared['frozen_sha256'].items():
        if sha256(Path(path))!=digest:raise ValueError('paired initialization source/weight changed')
    write_json(args.output/'summary.json',dict(status='complete_initialization_pair_development',edge=args.edge,quality_users=512,
        evaluation_window=old['f']['summary']['evaluation_window'],
        actual_executed_points=len(prepared['jobs']),reused_points=len(prepared['reused_summaries']),summaries=summaries,contrast=contrast,
        confidence='1000 identical paired UID draws; common FG cancels in the difference of h increments; conditional on fixed seed17 weights',
        independent_confirmation=False,predecessors_exact=True,native_anchor_inference_reused=True))
    print(json.dumps(dict(status='complete_initialization_pair_development',edge=args.edge,contrast=contrast)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--followup-plan',type=Path,required=True)
    p.add_argument('--reference-quality',type=Path,required=True);p.add_argument('--edge',type=int,choices=range(1,6),required=True)
    p.add_argument('--gpu',type=int,choices=(0,1),default=0);p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true')
    run(p.parse_args())
