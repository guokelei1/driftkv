"""Label-free fit-only gaps, fixed candidate alignment and stale-target purge."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.calibration_gaps import (
    fit_gap_law, make_views, load_gap_manifest, apply_gap_view, RULES, STALE_FIELDS, TIME_RULE,
)
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from hstu_kvcache.common.artifacts import sha256


def fixture():
    fit=list(range(1,65));validation=list(range(101,117));cutoff=1000000
    behavior={uid:[cutoff-1000,cutoff-500,cutoff-500,cutoff-100] for uid in fit}
    feedback={uid:[cutoff-500,cutoff-200,cutoff-100,cutoff,cutoff+100] for uid in fit}
    law=fit_gap_law(feedback,behavior,fit,cutoff)
    candidate=dict(known_items=10000,fitting_pool_uids=fit,validation_uids=validation,
        recent_items={str(uid):list(range(1000+uid*20,1016+uid*20)) for uid in fit+validation})
    release={uid:100+uid for uid in fit+validation}
    return fit,validation,cutoff,behavior,feedback,law,candidate,release


def test_strict_past_query_gap_handles_ties_without_query_or_future_behavior_leakage():
    fit,val,cutoff,behavior,feedback,law,_,_=fixture()
    observations=law['observations']['1']
    assert [row['gap_seconds'] for row in observations]==[500,300,400]
    assert all(row['source_feedback_timestamp']<cutoff for row in observations)
    assert all(row['source_previous_behavior_timestamp']<row['source_feedback_timestamp'] for row in observations)
    extra_behavior=copy.deepcopy(behavior);extra_feedback=copy.deepcopy(feedback)
    extra_behavior[1].extend([cutoff+1,cutoff+100]);extra_feedback[1].append(cutoff+1)
    extra_behavior[val[0]]=[1,2,3];extra_feedback[val[0]]=[4,5,6]
    assert fit_gap_law(extra_feedback,extra_behavior,fit,cutoff)==law
    assert set(law['users'])==set(fit) and not set(law['users'])&set(val)


def test_no_previous_behavior_is_native_zero_and_fallback_is_explicit_only_for_empty_feedback():
    law=fit_gap_law({1:[100]}, {1:[100,110]}, [1],200)
    assert law['observations']['1'][0]['gap_seconds']==0
    assert law['coverage']['no_prior_behavior_instances']==1
    with pytest.raises(ValueError,match='coverage='):
        fit_gap_law({}, {1:[100,110]}, [1],200)
    fallback=fit_gap_law({}, {1:[100,110,200,210]}, [1],200,allow_behavior_fallback=True)
    assert fallback['fallback_used'] and fallback['source'].endswith('behavior_gap_fallback')
    assert [row['gap_seconds'] for row in fallback['observations']['1']]==[10]
    assert fallback['coverage']['observed_feedback_instances']==0


def test_candidate_ids_unchanged_and_mix50_balanced_with_shared_empirical_bank():
    fit,val,cutoff,_,_,law,candidate,release=fixture()
    views=make_views(fit,val,candidate,release,law,edge=1,cutoff=cutoff)
    assert views==make_views(fit,val,candidate,release,law,edge=1,cutoff=cutoff)
    for uid in fit+val:
        user=views[str(uid)]
        assert user['candidate_ids']==candidates_for(uid,candidate['recent_items'][str(uid)],10000,count=16,recent_fraction=.5,seed=17)
        rules=user['rules'];mix=rules['gap_mix50'];empirical=rules['gap_empirical100']
        for first in (0,8):
            assert sum(row['source']=='cutover_release_gap' for row in mix[first:first+8])==4
        for index,row in enumerate(mix):
            if row['source']!='cutover_release_gap':
                assert row==empirical[index]
            else:
                assert row['gap_seconds']==release[uid]
                assert row['source_view_timestamp']-row['source_previous_behavior_timestamp']==release[uid]
        assert all(row['source_uid'] in fit and row['source']=='fit_pre_release_feedback' for row in empirical)
        assert all(row['source']=='cutover_release_gap' for row in rules['cutover'])


@pytest.mark.parametrize('fault',['val_in_law','short_feedback_quota','wrong_fit_role'])
def test_mismatched_causal_roles_or_candidate_composition_are_rejected(fault):
    fit,val,cutoff,_,_,law,candidate,release=fixture()
    if fault=='val_in_law':law['users'].append(val[0])
    elif fault=='short_feedback_quota':candidate['recent_items'][str(fit[0])]=[1,2]
    else:candidate['fitting_pool_uids']=list(reversed(fit))
    with pytest.raises(ValueError):make_views(fit,val,candidate,release,law,edge=1,cutoff=cutoff)


def test_gap_application_is_before_target_preparation_and_preserves_read_only_caches():
    fit,val,cutoff,_,_,law,candidate,release=fixture()
    views=make_views(fit,val,candidate,release,law,edge=1,cutoff=cutoff)
    tensors={int(uid):torch.tensor([[row['gap_seconds'] for row in data['rules']['gap_mix50']]]) for uid,data in views.items()}
    provenance=dict(rule='gap_mix50',manifest_sha256='fixed',candidate_ids={uid:data['candidate_ids'] for uid,data in views.items()})
    cache=object()
    samples={uid:dict(parent_cache=cache,candidates=torch.tensor([views[str(uid)]['candidate_ids']]),query_delta=torch.tensor([release[uid]]),
        **{key:'stale' for key in STALE_FIELDS}) for uid in fit+val}
    updated=apply_gap_view(samples,tensors,provenance)
    for uid,sample in updated.items():
        assert sample['parent_cache'] is cache
        assert sample['query_delta'].shape==(1,16)
        assert not set(sample)&set(STALE_FIELDS)
        assert sample['query_delta_rule']=='gap_mix50'
        assert sample['gap_manifest_sha256']=='fixed'
        assert samples[uid]['teacher_scores']=='stale'
        sample['query_delta'][0,0]=-1
        assert tensors[uid][0,0]>=0
    samples[fit[0]]['candidates'][0,0]+=1
    with pytest.raises(ValueError,match='candidate IDs/order'):apply_gap_view(samples,tensors,provenance)


def write_manifest(tmp_path):
    fit,val,cutoff,_,_,law,candidate,release=fixture()
    root=tmp_path/'inputs';folder=root/'edge_1';folder.mkdir(parents=True)
    files=dict(population_path=root/'population.json',path=folder/'inputs.json',candidate_path=folder/'calibration_feedback_candidates.json',history_path=folder/'histories.npz')
    files['population_path'].write_text(json.dumps(dict(fit=fit,validation=val)))
    # Loader cutoff follows the real protocol's day-to-second conversion.
    cutoff=12*86400
    files['path'].write_text(json.dumps(dict(settings=dict(release_day=12))))
    files['candidate_path'].write_text(json.dumps(candidate));files['history_path'].write_bytes(b'frozen prefix bytes')
    binding=dict(input_root=str(root),**{key:str(path) for key,path in files.items()},
        population_sha256=sha256(files['population_path']),sha256=sha256(files['path']),candidate_sha256=sha256(files['candidate_path']),history_sha256=sha256(files['history_path']))
    manifest=dict(schema_version=1,status='prepared',edge=1,source_cutoff=cutoff,release_day=12,time_rule=TIME_RULE,
        fit_uids=fit,validation_uids=val,candidate_order='fixed8feedback8uniform',
        users=make_views(fit,val,candidate,release,law,edge=1,cutoff=cutoff),input_binding=binding,
        sources=dict(source_sha256={}),gap_law=law,cpu_seconds=dict(fit_law=.01,assignments=.01),scope='synthetic input fixture')
    path=tmp_path/'gap_manifest.json';path.write_text(json.dumps(manifest))
    return root,path,files


def test_loader_binds_input_candidate_order_and_exposes_exact_driver_api(tmp_path):
    root,path,_=write_manifest(tmp_path)
    for rule in RULES:
        tensors,provenance=load_gap_manifest(path,root,1,rule)
        assert len(tensors)==80 and all(t.shape==(1,16) for t in tensors.values())
        assert provenance['rule']==rule and provenance['manifest_sha256']==sha256(path)
        assert provenance['source_cutoff']==12*86400
        assert provenance['candidate_order']=='fixed8feedback8uniform'
        json.dumps(provenance)


@pytest.mark.parametrize('fault',['raw_candidate_hash','candidate_index','cutoff'])
def test_loader_rejects_changed_bindings_or_assignment_order(tmp_path,fault):
    root,path,files=write_manifest(tmp_path)
    data=json.loads(path.read_text())
    if fault=='raw_candidate_hash':files['candidate_path'].write_text('{}')
    elif fault=='candidate_index':data['users']['1']['rules']['gap_mix50'][0]['candidate_index']=9
    else:data['source_cutoff']-=1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):load_gap_manifest(path,root,1,'gap_mix50')
