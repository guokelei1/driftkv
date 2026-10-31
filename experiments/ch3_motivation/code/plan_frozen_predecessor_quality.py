#!/usr/bin/env python3
"""Bind one committed supplementary ladder to the unchanged development512."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    frozen_predecessor_quality as audit, plan_selected_development as existing,
)
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.common.artifacts import sha256


def generate(fit_plan,edge):
    fit_plan=Path(fit_plan).resolve();plan=json.loads(fit_plan.read_text())
    if plan['expected_ladders']!=5:raise ValueError('require the complete five-edge supplement plan')
    job=next(j for j in plan['jobs'] if j['edge']==edge)
    original_path=Path(plan['source_plan']);original=json.loads(original_path.read_text())
    if sha256(original_path)!=plan['source_plan_sha256']:raise ValueError('original fitting plan changed')
    sources=existing.sources()
    for path in (Path(__file__),Path(audit.__file__)):
        sources[str(path.relative_to(ROOT))]=sha256(path)
    result=dict(status='waiting_actual_supplementary_weights',edge=edge,jobs=[],source_sha256=sources,
        expected_points=3,users_per_point=512,seed=17,scope='supplementary development, no independent confirmation or three-seed claim')
    if not (Path(job['output'])/'summary.json').exists():return result
    checked=audit.audit_ladder(job,original['input_bindings'][str(edge)])
    fit_identity=checked['identity']
    population_path=existing.INPUTS/'population.json';population=json.loads(population_path.read_text())
    users=population['development']
    if (len(users)!=len(set(users)) or len(users)!=512 or
            set(users)&set(fit_identity['fit_uids']+fit_identity['validation_users']) or
            set(users)&set(population['main']+population['confirmation'])):
        raise ValueError('development512 or held-out roles changed')
    input_path=existing.INPUTS/f'edge_{edge}/inputs.json';binding=json.loads(input_path.read_text())
    if sha256(input_path)!=fit_identity['input_binding_sha256'] or sha256(population_path)!=fit_identity['population_sha256']:
        raise ValueError('quality input context differs from fitted weights')
    frozen=dict(plan['frozen_sha256'])
    def add(path,digest=None):
        path=Path(path);path=(path if path.is_absolute() else ROOT/path).resolve();digest=digest or sha256(path)
        if str(path) in frozen and frozen[str(path)]!=digest:raise ValueError('quality binding conflict')
        frozen[str(path)]=digest
    for path,digest in sources.items():add(path,digest)
    for path in (fit_plan,original_path,population_path,input_path,Path(job['output'])/'summary.json'):add(path)
    for kind in ('history','requests'):add(input_path.parent/binding[kind]['path'],binding[kind]['sha256'])
    for path,digest in fit_identity['models'].items():add(path,digest)
    for row in checked['weights']:
        add(row['checkpoint'],row['checkpoint_sha256']);add(Path(row['checkpoint']).with_suffix('.json'))
    result.update(status='prepared_for_root_review',actual_weight_audit=checked['weights'],
        training_identity=fit_identity,frozen_sha256=frozen,development_uids_sha256=existing.digest(users),
        jobs=[dict(name=f'edge_{edge}_{row["method_label"]}',edge=edge,checkpoint=row['checkpoint'],
            input_root=str(existing.INPUTS),panel='development',population_file=None,backend='projected') for row in checked['weights']])
    for path,digest in frozen.items():
        if sha256(Path(path) if Path(path).is_absolute() else ROOT/path)!=digest:raise ValueError('quality dependency changed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit-plan',type=Path,required=True)
    p.add_argument('--edge',type=int,choices=range(1,6),required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();value=generate(args.fit_plan,args.edge);write_json(args.output/'plan.json',value)
    print(json.dumps(dict(status=value['status'],jobs=len(value['jobs']),edge=args.edge)))
