"""Freeze the concrete sampling identity branches before E6 or quality audits."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import plan_probe_quality, run_lowrank_release
from experiments.ch3_motivation.code.run_lowrank import source_paths
from experiments.ch3_motivation.code.sampling_identity import sampling_source_paths
from hstu_kvcache.common.artifacts import sha256


def test_all_three_entry_source_lists_cover_real_sampling_dependencies():
    paths=sampling_source_paths()
    assert {p.name for p in paths}=={'sampling_identity.py','calibration_sampling.py',
        'calibration_gaps.py','prepare_candidates.py','probe_initialization.py',
        'run_probe_calibration.py','run_probe_calibration_init.py'}
    required={str(p.relative_to(ROOT)) for p in paths}
    for sources in (run_lowrank_release.sources(),plan_probe_quality.sources()):
        assert required<=set(sources)
        assert all(sha256(ROOT/path)==digest for path,digest in sources.items())
    alternate=source_paths()
    assert len(set(alternate))==len(alternate) and set(paths)<=set(alternate)
    assert ROOT/'experiments/ch3_motivation/code/energy_certificate.py' in alternate


def test_changing_identity_helper_changes_e6_and_quality_freeze(monkeypatch):
    target='experiments/ch3_motivation/code/sampling_identity.py'
    before_e6,before_quality=run_lowrank_release.sources(),plan_probe_quality.sources()
    def changed(path):
        return 'f'*64 if Path(path).resolve()==(ROOT/target).resolve() else sha256(path)
    monkeypatch.setattr(run_lowrank_release,'sha256',changed)
    monkeypatch.setattr(plan_probe_quality,'sha256',changed)
    after_e6,after_quality=run_lowrank_release.sources(),plan_probe_quality.sources()
    assert {k for k in before_e6 if before_e6[k]!=after_e6[k]}=={target}
    assert {k for k in before_quality if before_quality[k]!=after_quality[k]}=={target}


def test_quality_waiting_plan_records_complete_source_manifest_without_dispatch(tmp_path):
    import json
    formal=tmp_path/'formal.json'
    formal.write_text(json.dumps(dict(status='waiting_selection')))
    result=plan_probe_quality.generate(formal,tmp_path/'main',tmp_path/'curve',tmp_path/'output')
    assert result['status']=='waiting_selection' and result['gpu_launch_performed'] is False
    assert result['source_sha256']==plan_probe_quality.sources()
