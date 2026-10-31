"""Matched PhaseB plans wait for the six new law256 input comparisons."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import prepare_probe_calibration_followup as planner
from experiments.ch3_motivation.code.select_probe_calibration_common import choose_sampling_inputs
from experiments.ch3_motivation.code.run_probe_calibration_init import parser as driver_parser
from tests.experiments.ch3_motivation.test_plan_probe_calibration_formal import (
    assets, selection as typed_selection,
)
from hstu_kvcache.common.artifacts import sha256


def selection(tmp_path,assets):
    """Synthetic common losses with a genuinely bound five-edge typed winner."""
    typed=typed_selection(tmp_path,assets,n=256,rule='gap_mix50')
    chosen=typed.value['recommendation'];groups=[]
    for fraction in (.5,.75):
        for rule in ('cutover','gap_mix50','gap_empirical100'):
            group=copy.deepcopy(chosen);loss=.1 if (fraction,rule)==(.75,'gap_mix50') else .2+len(groups)*.01
            group.update(recipe=f'input_{fraction}_{rule}',complete_five_edges=True,selection_eligible=True,
                validated_edges=planner.EDGES,common_loss_by_edge={str(e):loss for e in planner.EDGES},
                mean_common_loss=loss,complete_calibration_flops_estimate_sum=100)
            group['settings'].update(recent_fraction=fraction,query_delta_rule=rule)
            groups.append(group)
    value=dict(status='recommendation_for_review',recommendation=dict(recipe='old_law64_benchmark',
        fitting_driver='run_probe_temporal.py',query_delta_rule='gap_empirical100',fit_users=64),
        sampling_phase_a=choose_sampling_inputs(groups,True,[]))
    path=tmp_path/'combined_selection.json';path.write_text(json.dumps(value))
    return path,value


def test_pending_local_phase_never_dispatches_global_winner(tmp_path):
    missing=tmp_path/'missing.json'
    result=planner.generate(planner.DEFAULT_CONFIG,missing)
    assert result['status']=='pending_sampling_phase_a' and not result['jobs']
    assert result['common_plan_sha256']==sha256(planner.DEFAULT_COMMON_PLAN)
    missing.write_text(json.dumps(dict(status='recommendation_for_review',recommendation=dict(recipe='old_best'))))
    result=planner.generate(planner.DEFAULT_CONFIG,missing)
    assert not result['jobs'] and result['global370_comparison']['recipe']=='old_best'


def test_full_five_edge_choice_produces35_real_init_ladders_with_fixed_input(tmp_path,assets):
    path,_=selection(tmp_path,assets)
    result=planner.generate(planner.DEFAULT_CONFIG,path,run_root=tmp_path/'future_weights')
    assert result['status']=='prepared_for_root_review' and result['gpu_launched'] is False
    assert result['recipe_selection_performed'] is False and len(result['jobs'])==35
    assert result['expected_weights']==105 and len(result['gpu_jobs']['0'])==14 and len(result['gpu_jobs']['1'])==21
    assert result['global370_comparison']['recipe']=='old_law64_benchmark'
    assert result['selected_input']['query_delta_rule']=='gap_mix50'
    for edge in planner.EDGES:
        jobs=[job for job in result['jobs'] if job['edge']==edge]
        assert [job['name'] for job in jobs]==list(planner.ORDER)
        assert [job['fit_users'] for job in jobs]==[256]*5+[64,128]
        by_name={job['name']:job for job in jobs};reference=by_name['reference'];balanced=by_name['balanced_h_init']
        assert all(reference[k]==balanced[k] for k in ('fit_users','width','learning_rate','steps','patience','seed','sampling_manifest','recent_fraction','gap_rule'))
        assert reference['h_initialization']=='standard' and balanced['h_initialization']=='balanced'
        for job in jobs:
            identity=job['expected_identity'];n=job['fit_users']
            assert identity['fitting_driver']==planner.DRIVER and job['reuse_existing_weights'] is False
            assert identity['candidate_catalog_source_users']==identity['time_law_source_users']==256
            assert job['candidate_catalog_source_users']==job['time_law_source_users']==256
            assert job['gradient_fitting_users']==job['teacher_fitting_users']==n
            assert identity['fit_users']==identity['gradient_fitting_users']==identity['teacher_fitting_users']==n
            assert len(identity['fit_uids'])==n and len(identity['validation_users'])==16
            assert identity['h_initialization']==job['h_initialization']
            assert job['recent_fraction']==.75 and job['gap_rule']=='gap_mix50'
            parsed=driver_parser().parse_args(job['argv'][2:])
            assert parsed.fit_users==n and parsed.h_initialization==job['h_initialization'] and parsed.controls is False
    assert result['frozen_sha256'][str(planner.DEFAULT_COMMON_PLAN)]==sha256(planner.DEFAULT_COMMON_PLAN)
    assert result['validation_required_schema']['expected_context_cells']==315


@pytest.mark.parametrize('fault',['missing_recipe','missing_edge','missing_weight','wrong_mode_driver','wrong_source_sha'])
def test_incomplete_or_mismatched_phase_a_cannot_generate_partial_jobs(tmp_path,assets,fault):
    path,value=selection(tmp_path,assets);phase=value['sampling_phase_a']
    if fault=='missing_recipe':phase['recipes'].pop()
    elif fault=='missing_edge':phase['recipes'][0]['complete_five_edges']=False
    elif fault=='missing_weight':Path(phase['recommendation']['checkpoints_by_edge']['5']['checkpoint']).unlink()
    else:
        ref=phase['recommendation']['checkpoints_by_edge']['3'];p=Path(ref['checkpoint'])
        payload=torch.load(p,map_location='cpu',weights_only=False)
        if fault=='wrong_mode_driver':payload['metadata']['identity']['fitting_driver']='run_probe_calibration_init.py'
        else:payload['metadata']['identity']['gap_manifest_sha256']='0'*64
        torch.save(payload,p);ref['checkpoint_sha256']=sha256(p)
    path.write_text(json.dumps(value))
    if fault.startswith('wrong'):
        with pytest.raises(ValueError):planner.generate(planner.DEFAULT_CONFIG,path)
    else:
        result=planner.generate(planner.DEFAULT_CONFIG,path)
        assert result['status'].startswith('pending') and not result['jobs']


def test_reference_and_balanced_must_remain_matched(tmp_path):
    config=json.loads(planner.DEFAULT_CONFIG.read_text())
    next(row for row in config['recipes'] if row['name']=='balanced_h_init')['learning_rate']=1e-5
    path=tmp_path/'changed_config.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='seven fixed matched'):
        planner.generate(path,tmp_path/'missing.json')


def test_local_winner_identity_cannot_be_replaced_by_global_or_another_input(tmp_path,assets):
    path,value=selection(tmp_path,assets)
    value['sampling_phase_a']['recommendation']['recipe']='old_law64_benchmark'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='raw common-loss input winner'):
        planner.generate(planner.DEFAULT_CONFIG,path)
