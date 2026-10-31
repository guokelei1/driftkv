import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('residual_angles', Path(__file__).resolve().parents[3]/'experiments/ch3_motivation/code/analyze_residual_angles.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_common_logit_shift_does_not_change_relative_candidate_error():
    teacher = np.arange(16, dtype=float)
    snapshot = {'fixed_panel_scores': {'rebuild': teacher, 'f': teacher+3.}}
    value = m.score_metrics(snapshot, 'f')
    assert value['total_mse'] == 9.
    assert value['common_offset_mse'] == 9.
    assert value['candidate_centered_mse'] == 0.
    assert value['teacher_pair_disagreement'] == 0.


def test_centered_logit_error_has_pairwise_difference_interpretation():
    rng = np.random.default_rng(17)
    teacher = rng.normal(size=16)
    prediction = rng.normal(size=16)
    snapshot = {'fixed_panel_scores': {'rebuild': teacher, 'f': prediction}}
    result = m.score_metrics(snapshot, 'f')
    i, j = np.triu_indices(16, 1)
    pairwise_error = ((prediction[i]-prediction[j]) - (teacher[i]-teacher[j]))**2
    np.testing.assert_allclose(pairwise_error.mean(), (32/15)*result['candidate_centered_mse'])
    assert result['total_mse'] >= result['candidate_centered_mse']
