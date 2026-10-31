"""The two fixed256 drivers bind every timing view before quality dispatch."""
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.plan_probe_quality import audit_weight
from experiments.ch3_motivation.code.probe_variants import save_variant
from tests.experiments.ch3_motivation.test_fg_trajectory_control import sampling_pair
from hstu_kvcache.common.artifacts import sha256


def fixture(tmp_path,rule='cutover',mode=None,method='fg'):
    _,_,fg,full,fg_meta,full_meta,_=sampling_pair(tmp_path,rule,mode=mode)
    probe,metadata=(full,full_meta) if method=='fgh' else (fg,fg_meta)
    identity=metadata['identity']
    identity.update(width=probe.cfg.hidden_width,depth=probe.cfg.depth,learning_rate=3e-5,
        steps=512,read_weight=1.,score_weight=1.,patience=0,validation_every=8,
        stages='f,fg,fgh',controls=False)
    metadata.update(seed=17,completed_steps=512,best_step=256,
        fit_pool_loss_scales=dict(version='fit_pool_equal_user_length_normalized_v1',source='fitting_users_only',
            users=128,validation_uses_fitting_scales=True))
    job=dict(identity,variant='full_item')
    job['input_root']=str(tmp_path/'fit256')
    candidate=Path(job['input_root'])/'edge_1/calibration_feedback_candidates.json'
    binding=dict(input_root=job['input_root'],population_sha256=identity['population_sha256'],
        input_binding_sha256=identity['input_binding_sha256'],candidate_file=str(candidate),
        candidate_file_sha256=sha256(candidate),sampling_manifest=identity['sampling_manifest'],
        sampling_manifest_sha256=identity['sampling_manifest_sha256'])
    path=tmp_path/f'{method}.pt'
    save_variant(path,probe,metadata)
    return path,probe,metadata,job,binding


@pytest.mark.parametrize('rule',['cutover','gap_mix50','gap_empirical100'])
@pytest.mark.parametrize('mode',[None,'standard','balanced'])
def test_actual_typed_weights_all_three_rules_bind_sources_and_actual_n(tmp_path,rule,mode):
    path,_,_,job,binding=fixture(tmp_path,rule,mode)
    result=audit_weight(path,job,binding,'fg')
    identity=result['sampling_identity']
    assert identity['query_delta_rule']==rule and identity['sampling_manifest_sha256'] is not None
    assert identity['candidate_catalog_source_users']==identity['time_law_source_users']==256
    assert identity['gradient_fitting_users']==identity['teacher_fitting_users']==128
    if mode is not None:assert identity['h_initialization']==mode


@pytest.mark.parametrize('fault',['null_sha','job_sha','binding_sha','job_driver','job_driver_sha',
    'job_teacher','teacher_capture','rms','completed','manifest_changed'])
def test_cutover_has_the_same_strict_sampling_gate_as_temporal_views(tmp_path,fault):
    path,probe,metadata,job,binding=fixture(tmp_path)
    if fault=='null_sha':metadata['identity']['sampling_manifest_sha256']=None
    elif fault=='job_sha':job['sampling_manifest_sha256']='b'*64
    elif fault=='binding_sha':binding['sampling_manifest_sha256']=None
    elif fault=='job_driver':job.pop('fitting_driver')
    elif fault=='job_driver_sha':job.pop('fitting_driver_sha256')
    elif fault=='job_teacher':job['teacher_fitting_users']=256
    elif fault=='teacher_capture':metadata['actual_teacher_users_captured']=256
    elif fault=='rms':metadata['fit_pool_loss_scales']['users']=256
    elif fault=='completed':metadata['completed_steps']=256
    else:Path(job['sampling_manifest']).write_text('{}')
    save_variant(path,probe,metadata)
    with pytest.raises(ValueError):audit_weight(path,job,binding,'fg')


def test_actual_manifest_uid_order_is_checked_before_dispatch(tmp_path):
    path,probe,metadata,job,binding=fixture(tmp_path)
    metadata['identity']['fit_uids'].reverse()
    save_variant(path,probe,metadata)
    with pytest.raises(ValueError,match='fitting roles'):
        audit_weight(path,job,binding,'fg')


def test_initialization_driver_cannot_relabel_phase_a_or_change_its_actual_dimensions(tmp_path):
    path,probe,metadata,job,binding=fixture(tmp_path,mode='balanced')
    job['h_initialization']='standard'
    with pytest.raises(ValueError,match='initialization differs'):
        audit_weight(path,job,binding,'fg')
    job['h_initialization']='balanced'
    metadata['identity']['h_initialization_spec']['query_dim']+=1
    save_variant(path,probe,metadata)
    with pytest.raises(ValueError,match='actual dimensions'):
        audit_weight(path,job,binding,'fg')


def test_new_sampler_does_not_relax_legacy_cutover_or_temporal_identity(tmp_path):
    path,probe,metadata,job,binding=fixture(tmp_path)
    for value in (metadata['identity'],job):
        for key in list(value):
            if key.startswith('sampling_') or key in ('fitting_driver','fitting_driver_sha256','gap_manifest','gap_manifest_sha256'):
                value.pop(key)
    save_variant(path,probe,metadata)
    assert 'sampling_identity' not in audit_weight(path,job,binding,'fg')
    metadata['identity']['sampling_manifest_sha256']='a'*64
    save_variant(path,probe,metadata)
    with pytest.raises(ValueError,match='recognized explicit fitting driver'):
        audit_weight(path,job,binding,'fg')
