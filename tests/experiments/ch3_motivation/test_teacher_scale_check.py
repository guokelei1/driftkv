import importlib.util
from pathlib import Path
import sys
import numpy as np

scripts=Path(__file__).resolve().parents[3]/'experiments/ch3_motivation/code'
spec=importlib.util.spec_from_file_location('teacher_scale',scripts/'check_teacher_candidate_scale.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_common_amplitude_increase_does_not_become_relative_error_worsening():
    teacher=np.arange(16,dtype=float)
    prediction=teacher+np.sin(teacher)
    snapshots=[{'checkpoint':'cutover','fixed_panel_scores':{'f':prediction,'rebuild':teacher}},
               {'checkpoint':'append_8','fixed_panel_scores':{'f':3*prediction,'rebuild':3*teacher}}]
    result=m.compare([{'snapshots':snapshots} for _ in range(8)],'append_8',replicates=50)
    assert result['paired_change']['centered_f_error_mse']['mean']>0
    assert result['paired_change']['teacher_candidate_variance']['mean']>0
    np.testing.assert_allclose(result['paired_change']['normalized_ratio']['mean'],0,atol=1e-16)
    np.testing.assert_allclose(result['paired_change']['normalized_ratio']['ci95'],[0,0],atol=1e-16)
    assert not result['quality_use_permitted']
