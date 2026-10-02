from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from read_correction_2026_09.cost import (CostModel, correction_forward, eager_read,
    normalize_metrics, query_ridge_fit, teacher_history_read)


def test_query_cost_independent_of_history_and_history_cost_scales():
    q = {"heads": 6, "head_dim": 32}
    h = {**q, "width": 32, "query_chunk": 4, "token_chunk": 256}
    assert correction_forward(q, 10) == correction_forward(q, 1024)
    assert correction_forward(h, 1024) > correction_forward(q, 1024)
    slope = correction_forward(h, 1024) - correction_forward(h, 512)
    assert slope == 2 * (correction_forward(h, 512) - correction_forward(h, 256))
    assert correction_forward(h, 100, batch=3) == 3 * correction_forward(h, 100)
    # Q>1 shares K/V projections within a call, but each query still scans all rows.
    assert correction_forward(h, 100, queries=2) < 2 * correction_forward(h, 100)
    assert correction_forward(h, 100, queries=2) > correction_forward(h, 100)
    assert correction_forward(h, 100) == correction_forward({**h, "token_chunk": 7}, 100)


def test_calibration_cost_and_signed_recovery():
    assert query_ridge_fit(6, 32, 512) > query_ridge_fit(6, 32, 128) > 0
    result = normalize_metrics(full_auc=.8, reuse_auc=.7, baseline_auc=.65,
                               extra_flops=150, full_minus_reuse_flops=100)
    assert abs(result["recovery_percent"] + 50) < 1e-10
    assert result["relative_flops_percent"] == 150
    single_class = normalize_metrics(full_auc=None, reuse_auc=None, baseline_auc=None,
                                     extra_flops=50, full_minus_reuse_flops=100)
    assert single_class["recovery_percent"] is None
    assert single_class["relative_flops_percent"] == 50


def test_eager_reader_has_no_mask_multiply_and_keeps_self_read():
    cost = CostModel.for_scale("medium")
    rows, n = 3 * 16, 1024
    historical = teacher_history_read(cost, n, queries=16, batch=3)
    qk_av = rows * n * 4 * cost.hidden_size
    scale_elu_plus_one = rows * n * 3 * cost.num_heads
    assert historical == qk_av + scale_elu_plus_one
    # Adding prefix rows changes only the per-layer history read, not self.
    assert eager_read(cost, n, queries=16, batch=3) - eager_read(cost, 0, queries=16, batch=3) == cost.num_layers * historical
    assert eager_read(cost, 0) > cost.embedding(1, query=True)
    old_upper_bound = (cost.embedding(rows, query=True)
        + cost.num_layers * cost.block(rows, rows * (n + 1), residual_scales=False)
        + cost.rms_norm(rows) + rows * (2 * cost.hidden_size + 1))
    assert old_upper_bound - eager_read(cost, n, queries=16, batch=3) == cost.num_layers * rows * cost.num_heads * (n + 2)
