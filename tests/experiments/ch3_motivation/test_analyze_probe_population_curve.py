"""Paired N64/128/256 f curves require full planned fitting and quality grids."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.analyze_probe_population_curve import (
    aggregate_loaded, analyze, audit_curve_plan, curve_training_record, _temporal_binding, COUNTS, SEEDS,
    EDGES, DEFAULT_SCOPE,
)
from experiments.ch3_motivation.code.analyze_probe_quality import ShardLoader
from experiments.ch3_motivation.code.analyze_probe_quality_multiedge import JOB_SETTINGS
from tests.experiments.ch3_motivation.test_analyze_probe_quality_multiedge import (
    loaded_fixture, checkpoint_payload,
)
from hstu_kvcache.common.artifacts import sha256


def scope():
    return json.loads(DEFAULT_SCOPE.read_text())


def payload_for(summary):
    payload=checkpoint_payload(summary)
    metadata=payload["metadata"];identity=metadata["identity"];n=summary["fit_users"]
    identity.update(fit_uids=list(range(n)),validation_users=list(range(1000,1016)),
        stages="f",controls=False,steps=512,patience=0)
    metadata.update(training_users=n,validation_users=16,teacher_preparation_users=n+16,
        completed_steps=512,best_step=0 if n==256 else 256)
    metadata["fit_pool_loss_scales"].update(users=n,validation_uses_fitting_scales=True,
        read_scales=[[n/64]],score_scale=n/64)
    return payload


def fixture(*,correlation=False,gap="positive"):
    inputs=[item for item in loaded_fixture(correlation=correlation,gap=gap)
        if item["summary"]["panel"]=="main" and item["summary"]["method_label"]=="f"]
    result=[]
    for item in inputs:
        for n in COUNTS:
            value=copy.deepcopy(item);s=value["summary"];s["fit_users"]=n
            labels=np.asarray([r["label"] for r in value["rows"]])
            scores=np.asarray([r["score"] for r in value["rows"]])+(n/64 if correlation else np.log2(n/64)*.12*labels)
            for r,score in zip(value["rows"],scores,strict=True):r["score"]=float(score)
            s["auc"]=float(roc_auc_score(labels,scores))
            s["identity"]["checkpoint_sha256"]=hashlib.sha256(f'{s["edge"]}/{s["seed"]}/{n}'.encode()).hexdigest()
            value["training"]=curve_training_record(s,payload_for(s));value["path"]=f'/curve/{s["edge"]}/{s["seed"]}/{n}/summary.json'
            result.append(value)
    return result


def test_shared_uid_draws_preserve_inverse_cross_edge_correlation_and_zero_size_contrasts():
    r=aggregate_loaded(fixture(correlation=True),scope(),replicates=100)
    assert r["status"]=="complete" and r["confidence"]["shared_uid_draw_across_edges_sizes_seeds"]
    assert r["quality_users"]==2000 and r["confidence"]["all_eligible_uids_including_no_feedback"]
    for method in r["across_seed"]["methods"].values():
        assert method["mean_auc"]==.5
        assert method["conditional_uid_auc_ci95"]==pytest.approx([.5,.5],abs=1e-14)
    for value in r["across_seed"]["increments"].values():
        assert value["mean_auc_difference"]==0
        assert value["conditional_uid_auc_difference_ci95"]==[0.,0.]
    assert r["across_seed"]["seed_status"]["three_seed_complete"]


def test_equal_edge_differences_precede_ratio_and_n_contrasts_are_directly_paired():
    inputs=fixture();r=aggregate_loaded(inputs,scope(),replicates=50)
    summaries=[x["summary"] for x in inputs if x["summary"]["seed"]==17 and x["summary"]["fit_users"]==64]
    gains=np.array([s["auc"]-s["reuse_auc"] for s in summaries]);gaps=np.array([s["rebuild_auc"]-s["reuse_auc"] for s in summaries])
    point=r["per_seed"][0]["methods"]["64"]
    assert point["recovery_percent"]==pytest.approx(100*gains.mean()/gaps.mean())
    assert abs(point["recovery_percent"]-np.mean(100*gains/gaps))>1
    direct=r["across_seed"]["increments"]["f256_minus_f64"]
    assert direct["mean_auc_difference"]>0 and direct["conditional_uid_auc_difference_ci95"][0]>0
    assert direct["mean_auc_difference"]==pytest.approx(r["across_seed"]["methods"]["256"]["mean_auc"]-r["across_seed"]["methods"]["64"]["mean_auc"])
    seed_values=[s["increments"]["f256_minus_f64"]["mean_auc_difference"] for s in r["per_seed"]]
    assert direct["training_seed_point_range"]["auc_difference"]==[min(seed_values),max(seed_values)]
    record=r["edge_records"][0]["points"][-1]
    assert record["gradient_users_seen"]==256 and record["best_checkpoint_gradient_users_seen"]==0
    assert record["normalization_users"]==256 and record["teacher_preparation_users"]==272
    assert r["edge_records"][-1]["evaluation_window"]["name"]=="E14_partial"


@pytest.mark.parametrize("missing",["seed","edge","n","cell"])
def test_partial_grid_has_no_primary_aggregate_or_three_seed_claim(missing):
    items=fixture()
    def keep(item):
        s=item["summary"]
        return not ((missing=="seed" and s["seed"]==29) or (missing=="edge" and s["edge"]==5)
            or (missing=="n" and s["fit_users"]==256) or (missing=="cell" and (s["edge"],s["seed"],s["fit_users"])==(5,29,256)))
    r=aggregate_loaded([x for x in items if keep(x)],scope(),replicates=3)
    assert r["status"]=="not_ready" and r["missing"] and not r["primary_aggregate_computed"]
    assert not r["observed_seed_status"]["three_seed_complete"] and "across_seed" not in r
    if missing=="seed":assert r["observed_seed_status"]["missing_expected_seeds"]==[29]


def test_full_quality_grid_without_formal_binding_is_not_ready():
    r=aggregate_loaded(fixture(),scope(),formal_plan_ready=False,replicates=3)
    assert r["complete_grid"] and r["status"]=="not_ready" and "per_seed" not in r


@pytest.mark.parametrize("gap",["negative","zero","crossing"])
def test_nonpositive_or_zero_crossing_gap_preserves_absolute_ci_only(gap):
    r=aggregate_loaded(fixture(gap=gap),scope(),replicates=50)
    point=r["across_seed"]["increments"]["f256_minus_f64"]
    assert point["mean_auc_difference"]>0 and point["conditional_uid_auc_difference_ci95"][0]>0
    assert point["conditional_uid_recovery_difference_ci95"] is None
    assert r["anchors"]["nonpositive_or_invalid_gap_replicates"]>0


@pytest.mark.parametrize("fault",["main_weights","completed","rms","val_rms","teacher","validation"])
def test_checkpoint_metadata_requires_separate_full_cap_own_n_calibration(fault):
    summary=fixture()[1]["summary"];payload=payload_for(summary);metadata=payload["metadata"]
    if fault=="main_weights":metadata["identity"].update(stages="f,fg,fgh",controls=True)
    elif fault=="completed":metadata["completed_steps"]=128
    elif fault=="rms":metadata["fit_pool_loss_scales"]["users"]=64
    elif fault=="val_rms":metadata["fit_pool_loss_scales"]["validation_uses_fitting_scales"]=False
    elif fault=="teacher":metadata["teacher_preparation_users"]=80
    else:metadata["validation_users"]=8
    with pytest.raises(ValueError):curve_training_record(summary,payload)


@pytest.mark.parametrize("fault",["duplicate","uids","architecture","recipe","labels","anchor","auc"])
def test_invalid_pairing_or_recipe_is_rejected(fault):
    values=fixture()
    if fault=="duplicate":values.append(values[0])
    elif fault=="uids":values[0]["summary"]["identity"]["users"].reverse()
    elif fault=="architecture":values[0]["summary"]["width"]=256
    elif fault=="recipe":values[0]["training"]["identity"]["learning_rate"]=.00003
    elif fault=="labels":values[0]["rows"][0]["label"]^=1
    elif fault=="anchor":values[0]["anchors"]["reuse"][0]["score"]+=.1
    else:values[0]["summary"]["auc"]+=.1
    with pytest.raises(ValueError):aggregate_loaded(values,scope(),replicates=3)


def prepared_plan(tmp_path,items):
    root=tmp_path/'inputs';root.mkdir();population=dict(fit=list(range(256)),validation=list(range(1000,1016)),main=list(range(10000,12000)))
    (root/'population.json').write_text(json.dumps(population));bindings={};jobs=[]
    for edge in EDGES:
        directory=root/f'edge_{edge}';directory.mkdir();(directory/'inputs.json').write_text(json.dumps(dict(edge=edge)))
        candidate=directory/'candidates.json';candidate.write_text(json.dumps(dict(edge=edge,population_sha256=sha256(root/'population.json'),fitting_pool_uids=population['fit'],validation_uids=population['validation'])))
        bindings[str(edge)]=dict(input_root=str(root),population_sha256=sha256(root/'population.json'),input_binding_sha256=sha256(directory/'inputs.json'),candidate_file=str(candidate),candidate_file_sha256=sha256(candidate),candidate_catalog_source_users=256)
    for item in items:
        s=item['summary'];identity=item['training']['identity'];binding=bindings[str(s['edge'])]
        identity.update(population_sha256=binding['population_sha256'],input_binding_sha256=binding['input_binding_sha256'],calibration_candidate_sha256=binding['candidate_file_sha256'])
        s['identity']['input_binding_sha256']=binding['input_binding_sha256']
        jobs.append(dict(**{k:s[k] for k in ('edge','seed','variant','fit_users','width','depth')},**{k:identity[k] for k in JOB_SETTINGS},input_root=str(root)))
    return dict(status='prepared_for_root_review',population_curve_policy=dict(candidate_catalog_source_users=256),source_scope=str(DEFAULT_SCOPE),source_scope_sha256=sha256(DEFAULT_SCOPE),curve_jobs=jobs,curve_input_bindings=bindings,main_input_bindings=bindings),population


@pytest.mark.parametrize('fault',[None,'missing_job','candidate_catalog','nonnested','cap','main_uids','candidate_sha'])
def test_formal_curve_jobs_catalog_and_roles_are_strictly_bound(tmp_path,fault):
    items=fixture();plan,population=prepared_plan(tmp_path,items)
    if fault=='missing_job':plan['curve_jobs'].pop()
    elif fault=='candidate_catalog':plan['curve_input_bindings']['1']['candidate_catalog_source_users']=64
    elif fault=='nonnested':items[0]['training']['identity']['fit_uids'].reverse()
    elif fault=='cap':plan['curve_jobs'][0]['steps']=256
    elif fault=='main_uids':items[0]['summary']['identity']['users'].reverse()
    elif fault=='candidate_sha':plan['curve_input_bindings']['1']['candidate_file_sha256']='changed'
    if fault is None or fault=='missing_job':
        r=audit_curve_plan(items,plan,ShardLoader(),DEFAULT_SCOPE)
        assert r['ready']==(fault is None)
        if fault=='missing_job':assert len(r['missing_jobs'])==1
    else:
        with pytest.raises(ValueError):audit_curve_plan(items,plan,ShardLoader(),DEFAULT_SCOPE)


def test_temporal_curve_remains_pending_without_dedicated_fit256_target_manifest(tmp_path):
    items=fixture();plan,population=prepared_plan(tmp_path,items)
    identity=items[0]['training']['identity'];job=plan['curve_jobs'][0];binding=plan['curve_input_bindings']['1']
    identity.update(query_delta_rule='gap_mix50',gap_manifest_sha256='a'*64)
    job.update(query_delta_rule='gap_mix50',gap_manifest_sha256='a'*64)
    assert not _temporal_binding(identity,job,binding,population,ShardLoader(),dict(time_law_source_users=64))
    plan['population_curve_policy']['time_law_source_users']=64
    r=audit_curve_plan(items,plan,ShardLoader(),DEFAULT_SCOPE)
    assert not r['ready'] and not r['temporal_bindings_ready']
    identity['query_delta_rule']='gap_empirical100'
    with pytest.raises(ValueError,match='query delta rule'):_temporal_binding(identity,job,binding,population,ShardLoader(),dict(time_law_source_users=64))


def test_missing_and_partial_quality_inputs_remain_not_ready(tmp_path):
    partial=tmp_path/'summary.json';partial.write_text(json.dumps(dict(status='running')))
    missing=tmp_path/'pending.json'
    r=analyze([partial,missing],replicates=3)
    assert r['status']=='not_ready' and len(r['pending_summary_paths'])==2
    assert r['supplied_points']==0 and len(r['missing'])==45


def test_fixed256_sampling_cutover_binds_real_loader_api_and_reuses_check_across_seeds(tmp_path,monkeypatch):
    import types
    from experiments.ch3_motivation.code.analyze_probe_population_curve import _sampling_binding
    population=dict(fit=list(range(256)),validation=list(range(1000,1016)))
    manifest=tmp_path/'sampling.json';manifest.write_text(json.dumps(dict(sourcebinding=dict(input_root=str(tmp_path)))))
    binding=dict(input_root=str(tmp_path),sampling_manifest=str(manifest),sampling_manifest_sha256=sha256(manifest))
    job=dict(edge=1,fit_users=128,recent_fraction=.75,query_delta_rule='cutover',
        sampling_manifest=str(manifest),sampling_manifest_sha256=sha256(manifest),gap_manifest=str(manifest),gap_manifest_sha256=sha256(manifest),
        fitting_driver='run_probe_calibration.py',fitting_driver_sha256='a'*64)
    provenance=dict(fit_uids=population['fit'][:128],validation_uids=population['validation'],
        sourcebinding=dict(sha256='input-sha',population_sha256='population-sha',candidate_sha256='candidate-sha'),
        view_key='feedback75_cutover',view_users_sha256='view-sha',
        catalog_source_users=256,time_law_source_users=256,time_rule={'strict':'<cutoff'},source_cutoff=86400,candidate_order='feedback12_uniform4',manifest_sha=sha256(manifest),gap_rule='cutover',feedback_fraction=.75,candidate_count=16,candidate_seed=17)
    identity=dict(job,seed=17,candidate_catalog_source_users=256,queries=16,candidate_seed=17,
        candidate_source='feedback',query_delta_layout='[B,16]',gradient_fitting_users=128,teacher_fitting_users=128,
        fit_uids=population['fit'][:128],validation_users=population['validation'],fitting_uses_future_feedback=False,
        input_binding_sha256='input-sha',population_sha256='population-sha',calibration_candidate_sha256='candidate-sha',
        numerical_sources={'experiments/ch3_motivation/code/run_probe_calibration.py':'a'*64},
        time_law_source_users=256,**{k:provenance[k] for k in ['time_rule','source_cutoff','candidate_order']})
    calls=[]
    def load(path,root,edge,fraction,rule,*,fit_users):
        calls.append((edge,fraction,rule,fit_users));return {},provenance
    module=types.ModuleType('experiments.ch3_motivation.code.calibration_sampling');module.load_sampling_manifest=load
    monkeypatch.setitem(sys.modules,'experiments.ch3_motivation.code.calibration_sampling',module)
    policy=dict(candidate_catalog_source_users=256,time_law_source_users=256)
    cache={};loader=ShardLoader()
    for seed in SEEDS:
        identity['seed']=seed
        assert _sampling_binding(identity,job,binding,population,loader,policy,cache)
    assert calls==[(1,.75,'cutover',128)]
    assert not _sampling_binding(identity,job,binding,population,loader,dict(policy,time_law_source_users=64),cache)
    identity['gap_manifest_sha256']='b'*64
    with pytest.raises(ValueError,match='manifest binding'):_sampling_binding(identity,job,binding,population,loader,policy,cache)


def test_only_formal_bound_edge_varying_sampling_fields_are_normalized_for_recipe():
    from experiments.ch3_motivation.code.analyze_probe_population_curve import _curve_recipe
    items=fixture();left=copy.deepcopy(items[0]['training']);right=copy.deepcopy(next(
        x['training'] for x in items if x['summary']['edge']==2 and x['summary']['fit_users']==128 and x['summary']['seed']==17))
    for edge,t in [(1,left),(2,right)]:
        t['identity'].update(fitting_driver='run_probe_calibration.py',fitting_driver_sha256='same-driver',
            query_delta_rule='gap_mix50',sampling_manifest=f'/edge{edge}/sampling.json',sampling_manifest_sha256=str(edge)*64,
            gap_manifest=f'/edge{edge}/sampling.json',gap_manifest_sha256=str(edge)*64,source_cutoff=edge*86400,
            candidate_catalog_source_users=256,time_law_source_users=256,
            gradient_fitting_users=len(t['identity']['fit_uids']),teacher_fitting_users=len(t['identity']['fit_uids']))
    assert _curve_recipe(left,across_edges=True)!=_curve_recipe(right,across_edges=True)
    left['formal_curve_calibration_bound']=right['formal_curve_calibration_bound']=True
    assert _curve_recipe(left,across_edges=True)==_curve_recipe(right,across_edges=True)
    right['identity']['time_law_source_users']=64
    assert _curve_recipe(left,across_edges=True)!=_curve_recipe(right,across_edges=True)


@pytest.mark.parametrize('rule',['cutover','gap_mix50','gap_empirical100'])
def test_real_fixed256_sampling_helper_manifest_binds_all_timing_rules(tmp_path,rule):
    from tests.experiments.ch3_motivation.test_calibration_sampling import fixture as sampling_fixture
    from experiments.ch3_motivation.code.analyze_probe_population_curve import _sampling_binding
    manifest,candidates=sampling_fixture()
    root=tmp_path/'fit256';folder=root/'edge_1';folder.mkdir(parents=True)
    population=dict(fit=manifest['target_fit_uids'],validation=manifest['validation_uids'])
    (root/'population.json').write_text(json.dumps(population))
    (folder/'inputs.json').write_text(json.dumps(dict(settings=dict(release_day=20))))
    (folder/'calibration_feedback_candidates.json').write_text(json.dumps(candidates))
    (folder/'histories.npz').write_bytes(b'fixed immutable CPU fixture history')
    manifest['sourcebinding']=dict(input_root=str(root),population_sha256=sha256(root/'population.json'),
        sha256=sha256(folder/'inputs.json'),candidate_sha256=sha256(folder/'calibration_feedback_candidates.json'),
        history_path=str(folder/'histories.npz'),history_sha256=sha256(folder/'histories.npz'))
    manifest['source_sha256']={}
    path=tmp_path/'sampling.json';path.write_text(json.dumps(manifest))
    binding=dict(input_root=str(root),sampling_manifest=str(path),sampling_manifest_sha256=sha256(path))
    job=dict(edge=1,fit_users=256,recent_fraction=.75,query_delta_rule=rule,sampling_manifest=str(path),
        sampling_manifest_sha256=sha256(path),gap_manifest=str(path),gap_manifest_sha256=sha256(path),
        fitting_driver='run_probe_calibration.py',fitting_driver_sha256='a'*64)
    identity=dict(job,queries=16,candidate_seed=17,candidate_catalog_source_users=256,time_law_source_users=256,
        candidate_source='feedback',query_delta_layout='[B,16]',gradient_fitting_users=256,teacher_fitting_users=256,
        fit_uids=population['fit'],validation_users=population['validation'],fitting_uses_future_feedback=False,
        input_binding_sha256=manifest['sourcebinding']['sha256'],population_sha256=manifest['sourcebinding']['population_sha256'],
        calibration_candidate_sha256=manifest['sourcebinding']['candidate_sha256'],
        numerical_sources={'experiments/ch3_motivation/code/run_probe_calibration.py':'a'*64},
        time_rule=manifest['time_rule'],source_cutoff=manifest['source_cutoff'],candidate_order=manifest['candidate_order'])
    assert _sampling_binding(identity,job,binding,population,ShardLoader(),
        dict(candidate_catalog_source_users=256,time_law_source_users=256),{})


@pytest.mark.parametrize('fault',['missing_gradient','missing_teacher','wrong_gradient','capture','validation_capture'])
def test_sampling_curve_actual_gradient_and_teacher_n_are_explicit(fault):
    summary=fixture()[1]['summary'];payload=payload_for(summary);metadata=payload['metadata'];identity=metadata['identity']
    n=summary['fit_users']
    identity.update(fitting_driver='run_probe_calibration.py',fit_users=n,
        gradient_fitting_users=n,teacher_fitting_users=n)
    metadata.update(actual_teacher_users_captured=n,actual_validation_users_captured=16)
    assert curve_training_record(summary,payload)['teacher_preparation_users']==n+16
    if fault=='missing_gradient':identity.pop('gradient_fitting_users')
    elif fault=='missing_teacher':identity.pop('teacher_fitting_users')
    elif fault=='wrong_gradient':identity['gradient_fitting_users']=64
    elif fault=='capture':metadata['actual_teacher_users_captured']=256
    else:metadata.pop('actual_validation_users_captured')
    with pytest.raises(ValueError,match='actual N|membership count|user coverage'):
        curve_training_record(summary,payload)


def test_sampling_grid_cannot_use_legacy_law64_source_policy():
    items=fixture()
    for item in items:
        item['training']['identity']['fitting_driver']='run_probe_calibration.py'
    result=aggregate_loaded(items,scope(),replicates=3,
        calibration_source_policy=dict(candidate_catalog_source_users=256,time_law_source_users=64))
    assert result['status']=='not_ready' and not result['primary_aggregate_computed']
