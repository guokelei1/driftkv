import numpy as np
import pytest

from experiments.recflow_request_ranking.code.configuration import (
    backbone_config,
    load_proposal,
    validate_proposal,
)
from experiments.recflow_request_ranking.code.preflight import stage_inventory
from experiments.recflow_request_ranking.code.protocol import (
    day_bounds_ms,
    day_date,
    download_files,
    observed_labels,
    paired_uid_interval,
    request_metrics,
    version_schedule,
)


def test_complete_candidate_period_supports_ten_causal_versions():
    rows = version_schedule()
    assert len(rows) == 10
    assert rows[0]["training_days"] == [12, 26]
    assert rows[0]["training_days"][1] - rows[0]["training_days"][0] + 1 == 15
    for index, row in enumerate(rows):
        assert row["training_days"][1] < row["evaluation_days"][0] < 37
        assert row["candidate_alignment_days"][0] >= 24
        assert row["available_from_day"] == row["evaluation_days"][0]
        assert row["training_interval_ms_half_open"][1] == row["evaluation_interval_ms_half_open"][0]
        if index:
            assert row["parent"] == rows[index - 1]["version"]
            assert row["training_days"][0] == rows[index - 1]["training_days"][1] + 1
    assert day_date(24) == "2024-02-05"
    assert day_date(37) == "2024-02-18"
    assert download_files() == [f"2024-02-{day:02d}.feather" for day in range(5, 19)]


def test_daily_windows_use_beijing_midnight_and_milliseconds():
    from datetime import datetime, timezone
    lo, hi = day_bounds_ms(24)
    assert hi - lo == 86_400_000
    assert datetime.fromtimestamp(lo / 1000, timezone.utc).isoformat() == "2024-02-04T16:00:00+00:00"
    assert lo <= 1707062400054 < hi
    assert day_bounds_ms(26)[1] == day_bounds_ms(27)[0] == 1707321600000
    assert day_bounds_ms(37)[1] == 1708272000000


def test_unknown_is_distinct_and_missing_exposure_is_rejected():
    np.testing.assert_array_equal(observed_labels([10, 20, 30], [10, 20], [20]), [0, 1, -1])
    with pytest.raises(ValueError, match="Exposure missing"):
        observed_labels([10, 30], [10, 20], [20])
    with pytest.raises(ValueError, match="observed exposure"):
        observed_labels([10, 20], [10], [20])
    with pytest.raises(ValueError, match="deduplicated"):
        observed_labels([10, 10], [10], [10])


def test_ndcg_is_per_request_dcg_over_idcg_not_an_auc_or_hit_rate():
    result = request_metrics([10, 20, 30, 40], [4, 3, 2, 1], [0, 1, -1, 1], ks=(1, 3, 10))
    expected = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    assert result["ndcg@3"] == pytest.approx(expected)
    assert result["recall@3"] == 0.5
    assert result["ndcg@1"] == 0
    assert result["recall@10"] == 1
    assert result["ndcg@10"] < 1  # all positives retrieved, but poorly ordered


def test_zero_positives_and_label_independent_score_ties():
    empty = request_metrics([10, 20], [0, 0], [-1, 0])
    assert empty["zero_positive"] and empty["ndcg@10"] is None
    a = request_metrics([20, 10], [1, 1], [1, 0], ks=(1, 2))
    b = request_metrics([10, 20], [1, 1], [0, 1], ks=(1, 2))
    assert a == b
    assert a["ndcg@1"] == 0


def test_cluster_interval_preserves_request_macro_estimand():
    # Three requests from one user, one from another: request macro is .1,
    # whereas the mean of user means would be zero.
    result = paired_uid_interval([7, 7, 7, 8], [.6, .6, .6, .2], [.4, .4, .4, .4], samples=100)
    assert result["difference"] == pytest.approx(.1)
    assert result["users"] == 2 and result["requests"] == 4
    identical = paired_uid_interval([7, 7, 8], [.1, .2, .3], [.1, .2, .3], samples=30)
    assert identical["difference"] == 0 and identical["ci95"] == [0, 0]


def test_missing_all_stage_never_reports_training_ready(tmp_path):
    result = stage_inventory(tmp_path)
    assert result["status"] == "missing_data"
    assert len(result["missing"]) == 14
    assert result["training_ready"] is False


def test_schema_only_cannot_admit_training(tmp_path):
    import pyarrow as pa
    import pyarrow.feather as feather
    for name in download_files():
        feather.write_feather(pa.table({"video_id": [1]}), tmp_path / name)
    result = stage_inventory(tmp_path)
    assert not result["missing"]
    assert all("rank_index" in row["missing_columns"] for row in result["present"])
    assert result["training_ready"] is False
    assert result["status"] == "invalid_headers"


def test_user_history_constraint_is_consumed_by_every_backbone_scale():
    proposal = load_proposal()
    for scale in ["pilot", "max"]:
        config = backbone_config(proposal, scale, num_items=10)
        assert config.max_seq_len == 1025 and config.input_dropout == 0
        assert config.block_variant == "hstu_reference" and config.relative_position_bias
    proposal["history"]["max_events"] = 2048
    with pytest.raises(ValueError, match="User-fixed history"):
        validate_proposal(proposal)


def test_configuration_rejects_overlap_and_temporal_drift():
    proposal = load_proposal()
    proposal["roles"]["confirmation"] = [89, 99]
    with pytest.raises(ValueError, match="without overlap"):
        validate_proposal(proposal)
    proposal = load_proposal()
    proposal["timeline"]["foundation_end_day"] = 15
    with pytest.raises(ValueError, match="timeline"):
        validate_proposal(proposal)
