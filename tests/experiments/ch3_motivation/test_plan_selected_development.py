"""Actual typed metadata/manifests route only the selected development512."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import (
    plan_selected_development as planner, sweep_evaluate_probes as sweep,
)
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.common.artifacts import sha256
import tests.experiments.ch3_motivation.test_plan_probe_calibration_formal as fixture


def world(tmp_path,tmp_path_factory,driver='run_probe_calibration.py'):
    assets=fixture.assets.__wrapped__(tmp_path_factory)
    assets.population['development']=list(range(30000,30512))
    write_json(assets.root/'population.json',assets.population)
    for edge in planner.EDGES:
        directory=assets.root/f'edge_{edge}'
        binding=json.loads((directory/'inputs.json').read_text())
        candidate=json.loads((directory/'calibration_feedback_candidates.json').read_text())
        binding['population_sha256']=candidate['population_sha256']=sha256(assets.root/'population.json')
        write_json(directory/'inputs.json',binding);write_json(directory/'calibration_feedback_candidates.json',candidate)
        manifest=json.loads(assets.manifests[edge].read_text())
        manifest['sourcebinding'].update(population_sha256=binding['population_sha256'],
            sha256=sha256(directory/'inputs.json'),candidate_sha256=sha256(directory/'calibration_feedback_candidates.json'))
        write_json(assets.manifests[edge],manifest)
    source_driver=driver if driver in planner.SAMPLING_DRIVERS else 'run_probe_calibration.py'
    chosen=fixture.selection(tmp_path/'selected',assets,driver=source_driver,
        rule='cutover' if driver=='run_probe_variants.py' else 'gap_empirical100')
    models={}
    for version in range(6):
        path=tmp_path/f'model{version}.pt';path.write_bytes(f'geometry binding fixture{version}'.encode())
        models[version]=(str(path),sha256(path))
    record=chosen.value['recommendation']
    record['input_bindings_by_edge']={}
    for edge in planner.EDGES:
        reference=record['checkpoints_by_edge'][str(edge)];path=Path(reference['checkpoint'])
        payload=torch.load(path,map_location='cpu',weights_only=False)
        identity=payload['metadata']['identity'];identity['models']=dict(models[v] for v in (edge-1,edge))
        if driver not in planner.SAMPLING_DRIVERS:
            for key in ('sampling_manifest','sampling_manifest_sha256','candidate_catalog_source_users',
                    'time_law_source_users','gradient_fitting_users','teacher_fitting_users'):
                identity.pop(key)
            identity['fitting_driver']=driver
            identity['fitting_driver_sha256']=sha256(ROOT/'experiments/ch3_motivation/code'/driver)
            identity['numerical_sources']={f'experiments/ch3_motivation/code/{driver}':identity['fitting_driver_sha256']}
            if driver=='run_probe_variants.py':
                for key in ('query_delta_rule','gap_manifest','gap_manifest_sha256','time_rule','source_cutoff',
                            'candidate_order','query_delta_layout','fitting_driver','fitting_driver_sha256'):
                    identity.pop(key)
        for method in planner.METHODS:
            saved=copy.deepcopy(payload);saved['stage']=saved['metadata']['method_label']=method
            torch.save(saved,path.parent/f'{method}.pt')
        reference['checkpoint_sha256']=sha256(path)
        record['input_bindings_by_edge'][str(edge)]=planner._input_binding(assets.root,edge)
        if edge==1:
            record['settings']=dict(identity,input_root=str(assets.root),fitting_driver=driver,
                fitting_driver_sha256=sha256(ROOT/'experiments/ch3_motivation/code'/driver),
                query_delta_rule=identity.get('query_delta_rule','cutover'))
    record.update(fitting_driver=driver,fitting_driver_sha256=sha256(ROOT/'experiments/ch3_motivation/code'/driver),
        query_delta_rule=record['settings']['query_delta_rule'])
    if driver not in planner.SAMPLING_DRIVERS:record.pop('sampling_provenance_by_edge',None)
    chosen.value['phase_b_completed']=True
    write_json(chosen.path,chosen.value)
    return SimpleNamespace(assets=assets,selection=chosen,existing=tmp_path/'existing')


@pytest.mark.parametrize('driver',planner.DRIVERS)
def test_all_four_real_driver_schemas_keep_original_controls_and_bind_sweep(tmp_path,tmp_path_factory,driver):
    data=world(tmp_path,tmp_path_factory,driver)
    result=planner.generate(data.selection.path,input_root=data.assets.root,existing_root=None)
    assert result['status']=='prepared_for_root_review' and len(result['jobs'])==15
    assert result['gpu_launch_performed'] is result['formal_training_launched'] is False
    assert result['fitting_driver']==driver and len(result['actual_weight_audit'])==15
    assert result['development_uids']==data.assets.population['development']
    assert {job['panel'] for job in result['jobs']}=={'development'}
    assert all(job['population_file'] is None and job['backend']=='projected' for job in result['jobs'])
    for checked in result['audited_edges'].values():
        assert checked['actual_training_identity']['controls'] is False
        assert checked['audit_job']['controls'] is False  # No future formal controls substitution.
        if driver in planner.SAMPLING_DRIVERS:
            assert checked['calibration_binding']['sampling_manifest_sha256']
        if driver.endswith('_init.py'):assert checked['audit_job']['h_initialization']=='balanced'
    plan=tmp_path/'quality_plan.json';write_json(plan,result)
    frozen=sweep._bindings(result,result['jobs'],plan,sha256(plan),{})
    sweep._verify_bindings(frozen)  # Actual accepted sweep binding format; no GPU.


@pytest.mark.parametrize('fault',['stage_control','wrong_rule','wrong_seed','wrong_manifest','selected_sha','overlap','provided_manifest'])
def test_inconsistent_actual_weight_role_or_selected_binding_refused(tmp_path,tmp_path_factory,fault):
    data=world(tmp_path,tmp_path_factory,'run_probe_calibration_init.py')
    row=data.selection.value['recommendation']['checkpoints_by_edge']['2'];path=Path(row['checkpoint'])
    if fault=='selected_sha':row['checkpoint_sha256']='0'*64;write_json(data.selection.path,data.selection.value)
    elif fault=='provided_manifest':
        data.selection.value['recommendation']['sampling_provenance_by_edge']['2']['sampling_manifest_sha256']='0'*64
        write_json(data.selection.path,data.selection.value)
    elif fault=='overlap':
        data.assets.population['development'][0]=data.assets.population['main'][0]
        write_json(data.assets.root/'population.json',data.assets.population)
    else:
        target=path.parent/'fg.pt';payload=torch.load(target,map_location='cpu',weights_only=False)
        identity=payload['metadata']['identity']
        if fault=='stage_control':identity['controls']=True
        elif fault=='wrong_rule':identity['query_delta_rule']='cutover'
        elif fault=='wrong_seed':identity['seed']=23
        else:identity['sampling_manifest_sha256']='0'*64
        torch.save(payload,target)
    with pytest.raises(ValueError):planner.generate(data.selection.path,input_root=data.assets.root,existing_root=None)


def test_final_selection_and_missing_siblings_produce_zero_jobs(tmp_path,tmp_path_factory):
    result=planner.generate(tmp_path/'missing_selection.json',input_root=tmp_path/'not_loaded')
    assert result['status']=='waiting_final_selection' and result['jobs']==[]
    data=world(tmp_path,tmp_path_factory)
    data.selection.value['phase_b_completed']=False;write_json(data.selection.path,data.selection.value)
    assert planner.generate(data.selection.path,input_root=data.assets.root)['jobs']==[]
    data.selection.value['phase_b_completed']=True;write_json(data.selection.path,data.selection.value)
    Path(data.selection.value['recommendation']['checkpoints_by_edge']['5']['checkpoint']).with_name('fg.pt').unlink()
    result=planner.generate(data.selection.path,input_root=data.assets.root)
    assert result['status']=='waiting_selected_weights' and result['jobs']==[]


def committed(data,result):
    for job,audit in zip(result['jobs'],result['actual_weight_audit'],strict=True):
        edge=job['edge'];method=Path(job['checkpoint']).stem
        folder=data.existing/job['name']
        training=result['audited_edges'][str(edge)]['actual_training_identity']
        identity=dict(edge=edge,panel='development',users=result['development_uids'],models=training['models'],
            input_binding_sha256=training['input_binding_sha256'],backend='projected',scenario=None,
            sources=planner.runtime_sources(),checkpoint=job['checkpoint'],checkpoint_sha256=audit['checkpoint_sha256'])
        write_json(folder/'configuration.json',identity)
        scores=folder/'probe_blocks/part_000000.parquet';scores.parent.mkdir(parents=True)
        scores.write_bytes(b'opaque committed block; planner hashes but never reads scores or AUC')
        write_json(scores.with_suffix('.json'),dict(users=result['development_uids'],fingerprint=planner.digest(identity),sha256=sha256(scores)))
        anchors=folder/'anchors';values=anchors/'quality_shards/part_000000.parquet';values.parent.mkdir(parents=True)
        values.write_bytes(b'opaque committed matched anchor block')
        write_json(values.with_suffix('.json'),dict(offset=0,users=512,fingerprint=planner.digest(identity),
            uids_sha256=hashlib.sha256(json.dumps(result['development_uids']).encode()).hexdigest(),scores_sha256=sha256(values)))
        key=planner.digest({name:identity[name] for name in ('edge','models','users','input_binding_sha256','sources')})
        write_json(folder/'summary.json',dict(status='complete',experiment='E5',seed=17,method_label=method,
            quality_users=512,identity=identity,anchor_source=dict(path=str(anchors),key=key)))


def test_exact_sha_context_and_committed_hashes_reuse_without_gpu_jobs(tmp_path,tmp_path_factory):
    data=world(tmp_path,tmp_path_factory)
    result=planner.generate(data.selection.path,input_root=data.assets.root,existing_root=None)
    committed(data,result)
    ready=planner.generate(data.selection.path,input_root=data.assets.root,existing_root=data.existing)
    assert ready['status']=='reuse_existing_complete' and ready['jobs']==[]
    assert len(ready['planned_jobs'])==15 and ready['reuse']['complete']
    assert all(row['status']=='exact_sha_reuse' and not row['score_values_read'] for row in ready['reuse']['points'])
    changed=data.existing/'edge_2_f/probe_blocks/part_000000.parquet'
    changed.write_bytes(b'changed committed scores')
    with pytest.raises(ValueError,match='dependency changed'):
        planner.generate(data.selection.path,input_root=data.assets.root,existing_root=data.existing)


def test_same_recipe_name_with_different_sha_is_not_reused(tmp_path,tmp_path_factory):
    data=world(tmp_path,tmp_path_factory)
    result=planner.generate(data.selection.path,input_root=data.assets.root,existing_root=None)
    committed(data,result)
    path=data.existing/'edge_1_f/summary.json';value=json.loads(path.read_text())
    value['identity']['checkpoint_sha256']='0'*64;write_json(path,value)
    ready=planner.generate(data.selection.path,input_root=data.assets.root,existing_root=data.existing)
    assert ready['status']=='prepared_for_root_review' and len(ready['jobs'])==1
    assert ready['jobs'][0]['name']=='edge_1_f' and len(ready['planned_jobs'])==15
    assert ready['reuse']['reused_points']==14
    assert not ready['reuse']['complete']


def test_pending_cli_has_no_gpu_or_sweep_side_effects(tmp_path):
    args=SimpleNamespace(selection=tmp_path/'absent.json',input_root=tmp_path/'absent_inputs',
                         existing_root=None,output=tmp_path/'plan')
    result=planner.main(args)
    assert result['status']=='waiting_final_selection' and result['jobs']==[]
    assert json.loads((args.output/'plan.json').read_text())==result
