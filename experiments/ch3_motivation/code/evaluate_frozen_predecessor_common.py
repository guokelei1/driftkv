#!/usr/bin/env python3
"""Same fixed pre-release three-context validation for joint and frozen ladders."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse,csv,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
import torch
from experiments.ch3_motivation.code import (
    validate_probe_temporal_common as temporal, frozen_predecessor_quality as audit,
)
from experiments.ch3_motivation.code.validate_probe_common import _check_weight_binding, evaluate_common_probe
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.common.artifacts import sha256


def run(args):
    plan=json.loads(args.fit_plan.read_text());original_path=Path(plan['source_plan'])
    if sha256(original_path)!=plan['source_plan_sha256']:raise ValueError('original plan changed')
    original=json.loads(original_path.read_text());job=next(j for j in plan['jobs'] if j['edge']==args.edge)
    checked=audit.audit_ladder(job,original['input_bindings'][str(args.edge)])
    common_path=Path(original['common_plan_path'])
    if sha256(common_path)!=original['common_plan_sha256']:raise ValueError('common validation plan changed')
    common=json.loads(common_path.read_text());binding=common['edge_bindings'][str(args.edge)]
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0]<16*1024**3:raise ValueError('insufficient common-validation headroom')
    contexts=temporal.prepare_contexts(binding['input_root'],args.edge,torch.device('cuda:0'),binding['gap_manifest'],backend='torch')
    joint_summary=json.loads((Path(job['source_job']['output'])/'summary.json').read_text())
    joint_hashes={row['method_label']:row['checkpoint_sha256'] for row in joint_summary['stages']}
    fingerprints={rule:context.fingerprint for rule,context in contexts.items()}
    rows=[];records=[];f_metrics={}
    for protocol,folder in (('joint',Path(job['source_job']['output'])),('new_term_only',Path(job['output']))):
        for stage in ('f','fg','fgh'):
            path=folder/f'{stage}.pt';digest=sha256(path)
            if protocol=='joint' and joint_hashes[stage]!=digest:raise ValueError('joint checkpoint committed hash changed')
            probe,metadata=load_variant(path,'cuda:0')
            if protocol=='new_term_only' and metadata['identity']!=checked['identity']:raise ValueError('supplement identity changed')
            if protocol=='joint' and metadata['identity']!=job['source_job']['expected_identity']:raise ValueError('joint reference identity changed')
            _check_weight_binding(contexts[temporal.RULES[0]],probe,metadata)
            metrics={}
            for rule,context in contexts.items():
                with torch.no_grad():value=evaluate_common_probe(context,probe)
                metrics[rule]=value
                rows.append(dict(edge=args.edge,protocol=protocol,stage=stage,context=rule,
                    loss=value['loss'],read_mse=value['read_mse'],score_mse=value['score_mse'],
                    checkpoint=str(path),checkpoint_sha256=digest))
            if stage=='f':
                if protocol=='joint':f_metrics=metrics
                elif metrics!=f_metrics:raise ValueError('same source f produced different common validation')
            if sha256(path)!=digest:raise ValueError('common checkpoint changed')
            records.append(dict(protocol=protocol,stage=stage,checkpoint=str(path),checkpoint_sha256=digest,
                contexts=metrics,mean_context_loss=sum(m['loss'] for m in metrics.values())/3,best_step=metadata['best_step']))
            del probe
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'points.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(args.output/'summary.json',dict(status='complete_paired_common_validation',edge=args.edge,
        points=records,contexts=fingerprints,fit_plan_sha256=sha256(args.fit_plan),common_plan_sha256=sha256(common_path),
        sources={str(p):sha256(p) for p in (Path(__file__),Path(audit.__file__),Path(temporal.__file__))},
        quality_auc_computed=False,validation_users=16,scope='pre-release equal-context validation only, no population quality; both complete ladders retained, no selection performed'))
    print(json.dumps(dict(status='complete_paired_common_validation',edge=args.edge,weights=6,cells=18)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit-plan',type=Path,required=True)
    p.add_argument('--edge',type=int,choices=range(1,6),required=True);p.add_argument('--gpu',type=int,choices=(0,1),required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
