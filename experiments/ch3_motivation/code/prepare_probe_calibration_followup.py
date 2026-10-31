#!/usr/bin/env python3
"""Bind seven matched PhaseB recipes to the six-input PhaseA recommendation.

Plans only. An incomplete local PhaseA selection produces zero jobs. The
combined370 winner is retained as a benchmark and cannot change this input law.
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
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import plan_probe_calibration_formal as formal
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.probe_initialization import initialization_spec
from experiments.ch3_motivation.code.sampling_identity import validate_sampling_identity
from hstu_kvcache.common.artifacts import sha256

DRIVER='run_probe_calibration_init.py'
EDGES=[1,2,3,4,5]
ORDER=('reference','balanced_h_init','lr1e5','width256','cap1024','fit64','fit128')
DEFAULT_CONFIG=ROOT/'experiments/ch3_motivation/configs/probe_calibration_followup.json'
DEFAULT_SELECTION=ROOT/'experiments/ch3_motivation/runs/e5/calibration_sampling_common_selection/attempt_1/selection.json'
DEFAULT_COMMON_PLAN=ROOT/'experiments/ch3_motivation/configs/probe_common_temporal_validation.json'
DEFAULT_RUN_ROOT=ROOT/'experiments/ch3_motivation/runs/e5/calibration_followup/attempt_1'


def _recipes(config):
    base=dict(fit_users=256,width=128,learning_rate=3e-5,steps=2048,patience=1024,h_initialization='standard')
    changes=dict(reference={},fit64=dict(fit_users=64),fit128=dict(fit_users=128),
        lr1e5=dict(learning_rate=1e-5),width256=dict(width=256),cap1024=dict(steps=1024),
        balanced_h_init=dict(h_initialization='balanced'))
    expected={name:dict(base,**change) for name,change in changes.items()}
    supplied={row['name']:{key:value for key,value in row.items() if key!='name'} for row in config['recipes']}
    if (len(config['recipes'])!=7 or supplied!=expected or config['edges']!=EDGES or config['seed']!=17 or
            config['queries']!=16 or config['validation_every']!=8 or config['controls'] is not False or
            config['candidate_catalog_and_time_law_source_users']!=256):
        raise ValueError('followup must retain exactly the seven fixed matched recipes and catalog/law256')
    return supplied


def _local_choice(selection):
    phase=selection.get('sampling_phase_a',{})
    if (phase.get('status')!='recommendation_for_review' or not phase.get('all_five_edges_complete') or
            phase.get('recommendation') is None):
        return None
    if phase.get('expected_input_recipes')!=6 or phase.get('selection_seed')!=17:
        raise ValueError('followup requires the explicit six-recipe seed17 PhaseA selection')
    groups=phase.get('recipes',[])
    if len(groups)!=6 or any(not row.get('complete_five_edges') or not row.get('selection_eligible') for row in groups):
        return None
    combinations=set()
    for group in groups:
        settings=group['settings']
        if (group['method_label']!='fgh' or group['seed']!=17 or group['variant']!='full_item' or
                group['fit_users']!=256 or group['width']!=128 or group['depth']!=1 or
                group['validated_edges']!=EDGES or set(group['common_loss_by_edge'])!=set(map(str,EDGES)) or
                settings['fitting_driver']!='run_probe_calibration.py' or
                settings['candidate_catalog_source_users']!=256 or settings['time_law_source_users']!=256):
            raise ValueError('PhaseA input recipe changes its real five-edge law256 source or architecture')
        losses=list(group['common_loss_by_edge'].values())
        if (not all(math.isfinite(value) and value>=0 for value in losses) or
                not math.isclose(math.fsum(losses)/5,group['mean_common_loss'],rel_tol=1e-7,abs_tol=1e-12)):
            raise ValueError('PhaseA common loss is not the equal-five-edge mean')
        combinations.add((settings['recent_fraction'],settings['query_delta_rule']))
    if combinations!={(fraction,rule) for fraction in (.5,.75) for rule in ('cutover','gap_mix50','gap_empirical100')}:
        raise ValueError('PhaseA selection does not cover the six fixed candidate/time inputs')
    ordered=sorted(groups,key=lambda row:(row['mean_common_loss'],row['complete_calibration_flops_estimate_sum'],row['recipe']))
    chosen=phase['recommendation']
    if (chosen['recipe']!=ordered[0]['recipe'] or chosen['settings']!=ordered[0]['settings'] or
            chosen['edges']!=EDGES or chosen['fitting_driver']!='run_probe_calibration.py' or
            chosen['query_delta_rule']!=chosen['settings']['query_delta_rule']):
        raise ValueError('PhaseA recommendation differs from its raw common-loss input winner')
    return chosen


def _numerical_sources():
    names=(DRIVER,'calibration_sampling.py','probe_initialization.py','calibration_gaps.py','common.py',
        'probes.py','probe_variants.py','prepare_candidates.py')
    paths=[Path(__file__).with_name(name) for name in names]
    paths += [ROOT/'src/hstu_kvcache/adaptation/reader.py',ROOT/'src/hstu_kvcache/models/hstu.py',
        ROOT/'src/hstu_kvcache/models/embeddings.py',ROOT/'src/hstu_kvcache/evaluation/cost.py',
        ROOT/'src/hstu_kvcache/methods/baselines/protocol/cost.py']
    return {str(path.resolve().relative_to(ROOT)):sha256(path) for path in paths}


def validation_schema():
    return dict(format='ch3_calibration_followup_common_v1',expected_weights=105,expected_context_cells=315,
        recipes=list(ORDER),edges=EDGES,seed=17,stages=['f','fg','fgh'],
        plan_binding=['followup_plan_path','followup_plan_sha256','common_plan_path','common_plan_sha256'],
        per_weight=['recipe','edge','seed','method_label','checkpoint','checkpoint_sha256','status',
            'training_identity','config','best_step','completed_steps','step_zero','sampling_view_key',
            'cumulative_calibration_flops_estimate','contexts','mean_context_loss','selection_eligible'],
        statuses=['validated','pending','failed'],
        actual_identity='exact job.expected_identity; actual typed full_item configuration; driver/init/source/input/UID/count aliases all verified',
        contexts=['cutover','gap_mix50','gap_empirical100'],
        common_context_policy='same fit64-ID validation16 candidates, targets, three-context fingerprints and fixed common RMS as old280 plus sampling90; own-N fitting RMS is not substituted for common validation RMS',
        context_cell=['status','common_validation_sha256','metrics'],
        metrics=['loss','read_mse','score_mse','raw_score_mse','per_user'],
        paired_initialization_check='per edge, reference and balanced f/fg typed state_dict tensors and normalization buffers exactly equal; identical input/cap/seed; each FG/fullH pair retains its own mode identity',
        grouping='all five edges for every recipe/stage; equal three-context loss then equal five-edge loss; retain failures, adverse edges and step0',
        quality_auc_computed=False,global370_role='benchmark only; it cannot change the fixed PhaseA input view')


def generate(config_path,selection_path,*,run_root=DEFAULT_RUN_ROOT,common_plan_path=DEFAULT_COMMON_PLAN):
    config_path,selection_path,run_root=Path(config_path),Path(selection_path),Path(run_root).resolve()
    config=json.loads(config_path.read_text());recipes=_recipes(config)
    common_plan_path=Path(common_plan_path).resolve()
    common_sha=sha256(common_plan_path)
    if common_sha!=sha256(DEFAULT_COMMON_PLAN):
        raise ValueError('followup must preserve the original shared three-context common plan')
    code=formal.source_manifest(DRIVER)
    code[str(Path(__file__).resolve())]=sha256(Path(__file__))
    sweep=Path(__file__).with_name('sweep_probe_calibration_followup.py')
    if sweep.exists():code[str(sweep.resolve())]=sha256(sweep)
    result=dict(format='probe_calibration_followup_v1',status='pending_sampling_phase_a',gpu_launched=False,recipe_selection_performed=False,
        source_config=str(config_path.resolve()),source_config_sha256=sha256(config_path),
        source_selection=str(selection_path.resolve()),source_selection_sha256=None,
        common_plan_path=str(common_plan_path),common_plan_sha256=common_sha,
        planner_sources_sha256=code,jobs=[],gpu_jobs={'0':[],'1':[]},expected_ladders=35,expected_weights=105,
        run_root=str(run_root),validation_required_schema=validation_schema(),
        input_selection='selection.sampling_phase_a.recommendation only; six complete law256 inputs, raw common-loss winner; global370 comparison retained separately')
    if not selection_path.exists():return result
    result['source_selection_sha256']=sha256(selection_path)
    selection=json.loads(selection_path.read_text());chosen=_local_choice(selection)
    global_choice=selection.get('recommendation')
    result['global370_comparison']={key:global_choice.get(key) for key in
        ('recipe','mean_common_loss','fitting_driver','query_delta_rule','fit_users')} if global_choice else None
    if chosen is None:return result
    if set(chosen['checkpoints_by_edge'])!=set(map(str,EDGES)):
        raise ValueError('PhaseA recommendation must preserve all five source checkpoints')
    missing=[row['checkpoint'] for row in chosen['checkpoints_by_edge'].values() if not Path(row['checkpoint']).exists()]
    if missing:return dict(result,status='pending_phase_a_weights',missing_selected_weights=missing)
    settings=chosen['settings'];formal._supported(settings,chosen)
    population=json.loads((Path(settings['input_root'])/'population.json').read_text())
    numerical=_numerical_sources();cache={};bindings={};selected={};jobs=[]
    for edge in EDGES:
        original,config_actual,sampling,binding=formal._read_selected(chosen,settings,edge,cache)
        bindings[str(edge)]=binding
        selected[str(edge)]=chosen['checkpoints_by_edge'][str(edge)]
        gpu=1 if edge%2 else 0
        for name in ORDER:
            recipe=recipes[name];n=recipe['fit_users'];mode=recipe['h_initialization']
            spec=initialization_spec(config_actual['head_dim'],2*config_actual['head_dim']+config_actual['item_dim'],mode)
            identity=copy.deepcopy(original)
            identity.update(recipe,fit_uids=population['fit'][:n],gradient_fitting_users=n,teacher_fitting_users=n,
                fitting_driver=DRIVER,fitting_driver_sha256=numerical[f'experiments/ch3_motivation/code/{DRIVER}'],
                numerical_sources=numerical,h_initialization_spec=spec,initialization_helper_sha256=spec['helper_sha256'])
            prospective_config=dict(config_actual,hidden_width=recipe['width'],depth=1)
            provenance=validate_sampling_identity(identity,sampling_cache=cache,config=prospective_config)
            output=run_root/name/f'edge_{edge}'/'seed_17'
            job=dict(name=name,edge=edge,gpu=gpu,input_root=settings['input_root'],sampling_manifest=identity['sampling_manifest'],
                recent_fraction=identity['recent_fraction'],gap_rule=identity['query_delta_rule'],seed=17,
                validation_every=8,stages='f,fg,fgh',controls=False,device='cuda',**recipe,
                output=str(output),fitting_driver=DRIVER,fitting_driver_sha256=identity['fitting_driver_sha256'],
                expected_identity=identity,sampling_provenance=provenance,
                expected_checkpoints={stage:str(output/f'{stage}.pt') for stage in ('f','fg','fgh')},
                source_phase_a_checkpoint=selected[str(edge)],reuse_existing_weights=False)
            job.update({key:identity[key] for key in (*formal.SAMPLING_FIELDS,*formal.INITIALIZATION_FIELDS)})
            job.update(variant='full_item',depth=1,queries=16)
            argv=['.venv/bin/python','experiments/ch3_motivation/code/'+DRIVER]
            for key in ('input_root','output','sampling_manifest','edge','gap_rule','recent_fraction','fit_users','width',
                    'learning_rate','steps','patience','validation_every','stages','h_initialization','seed','gpu','device'):
                argv += ['--'+key.replace('_','-'),str(job[key])]
            argv.append('--no-controls');job['argv']=argv;jobs.append(job)
    result.update(status='prepared_for_root_review',jobs=jobs,
        gpu_jobs={str(gpu):[row for row in jobs if row['gpu']==gpu] for gpu in (0,1)},
        selected_input=dict(recipe=chosen['recipe'],recent_fraction=settings['recent_fraction'],query_delta_rule=chosen['query_delta_rule'],
            candidate_catalog_source_users=256,time_law_source_users=256,checkpoints_by_edge=selected),
        input_bindings=bindings,phase_a_failure_review=selection['sampling_phase_a'].get('needs_failure_review',False),
        schedule='whole edges: odd1/3/5 onGPU1, even2/4 onGPU0; per edge N256 reference/balanced/LR/width/cap contiguous, thenN64/N128; pass one preparation_cache to repeated driver.run calls',
        initialization_matching='reference/balanced share N,width,LR,cap,patience,seed and inputs; only h first-layer group gains differ; actual f/fg equality must be verified',
        weight_reuse_policy='all35 ladders are fresh under the init driver; no relabelled PhaseA metadata or implicit source mixing',
        search_cost='each point records standalone teacherN+16, fitting and validation costs; raw preparation reuse is separate search overhead')
    frozen={**code,str(config_path.resolve()):result['source_config_sha256'],str(selection_path.resolve()):result['source_selection_sha256'],
        str(common_plan_path):common_sha}
    for edge,binding in bindings.items():
        root=Path(binding['input_root'])
        frozen.update({str(root/'population.json'):binding['population_sha256'],str(root/f'edge_{edge}/inputs.json'):binding['input_binding_sha256'],
            binding['candidate_file']:binding['candidate_file_sha256'],binding['sampling_manifest']:binding['sampling_manifest_sha256'],
            selected[edge]['checkpoint']:selected[edge]['checkpoint_sha256']})
    for path,digest in frozen.items():
        if sha256(Path(path))!=digest:raise ValueError(f'followup plan dependency changed: {path}')
    result['frozen_sha256']=frozen
    return result


def main(args):
    result=generate(args.config,args.selection,run_root=args.run_root,common_plan_path=args.common_plan)
    write_json(args.output/'plan.json',result)
    write_json(args.output/'validation_schema.json',result['validation_required_schema'])
    if result['status']=='prepared_for_root_review':
        for gpu,jobs in result['gpu_jobs'].items():
            write_json(args.output/f'gpu_{gpu}_fitting.json',dict(status=result['status'],gpu=int(gpu),jobs=jobs,
                frozen_sha256=result['frozen_sha256'],scope='fitting plan only; no launch performed'))
    print(json.dumps(dict(status=result['status'],jobs=len(result['jobs']),gpu_launched=False)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--selection',type=Path,default=DEFAULT_SELECTION)
    parser.add_argument('--common-plan',type=Path,default=DEFAULT_COMMON_PLAN)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,default=DEFAULT_RUN_ROOT)
    main(parser.parse_args())
