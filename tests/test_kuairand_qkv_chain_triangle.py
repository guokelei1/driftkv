from hstu_kvcache.streaming.kuairand_qkv_chain_triangle import (
    _decision,
    _render_matrix,
    load_qkv_chain_config,
)

CONFIG = (
    "configs/evokv_root_cause/"
    "kuairand_qkv_chain_theta5_theta12_20260810_v0.json"
)


def _cell(target: int, source: int, value: float) -> dict:
    comparison = {
        metric: {"relative_percent": value}
        for metric in ("ndcg_at_5", "mrr", "hit_rate_at_5")
    }
    return {
        "target_version": target,
        "source_version": source,
        "holdout": {"comparisons": {"recompute_over_reuse": comparison}},
        "all_users": {"sanity": {"passed": True}},
    }


def test_qkv_chain_config_freezes_eight_models_and_qkv_scope():
    document = load_qkv_chain_config(CONFIG)
    assert document["lineage_selection"]["minimum_source_version"] == 5
    assert document["lineage_selection"]["versions"] == list(range(6, 13))
    assert all(
        candidate["dense_update_scope"] == "frozen"
        for candidate in document["training"]["candidate_ladder"]
    )
    assert len(document["training"]["candidate_ladder"]) == 1
    assert document["coordinate_drift"]["selection_basis"] == (
        "label_free_top10_change_target_extrapolation"
    )


def test_qkv_chain_matrix_and_decision_cover_twenty_eight_cells():
    cells = [
        _cell(target, source, float(target - source))
        for target in range(6, 13)
        for source in range(5, target)
    ]
    decision = _decision(cells)
    assert decision["matrix_versions"] == 8
    assert decision["off_diagonal_cells"] == 28
    assert decision["positive_cells"]["ndcg_at_5"] == 28
    table = _render_matrix(cells, list(range(5, 13)), "ndcg_at_5")
    assert any("| M7 |" in line for line in table)
