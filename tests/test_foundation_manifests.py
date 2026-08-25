from __future__ import annotations

import math
from pathlib import Path

import yaml

from hstu_kvcache.data import BASE_FEATURE_NAMES, CausalFeatureState, foundation_request_id, time_block


ROOT = Path(__file__).resolve().parents[1]


def test_small_foundation_contract_locks_expert_corrections() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/contracts/yambda500m_small_foundation_chain_v1.yaml").read_text()
    )
    assert "parent_exact_rolling" in contract["paths"]["names"]
    assert contract["causality"]["timestamp_group_atomicity"][
        "score_all_queries_from_common_pre_timestamp_state"
    ] is True
    assert contract["base"]["features"] == "request_time_as_of_strictly_prior"
    assert contract["metrics"]["traffic_persistence"]["observational_not_causal"] is True
    assert contract["metrics"]["fixed_query_dilution"]["causal_diagnostic"] is True
    assert contract["authorization"]["any_real_HSTU_training"] is False


def test_request_time_features_do_not_see_same_timestamp_listens() -> None:
    state = CausalFeatureState(max_history=3)
    state.append_listen(uid=1, timestamp=10, raw_item_id=7, item_idx=2, artist_id=4)
    before = state.request_features(uid=1, timestamp=20, raw_item_id=7, artist_id=4)
    # Every request at t=20 must be scored before this simultaneous event is appended.
    assert before[0] == math.log1p(1)
    assert before[2] == math.log1p(10)
    state.append_listen(uid=1, timestamp=20, raw_item_id=7, item_idx=2, artist_id=4)
    after = state.request_features(uid=1, timestamp=21, raw_item_id=7, artist_id=4)
    assert after[0] == math.log1p(2)
    assert after[2] == math.log1p(1)


def test_bounded_history_tracks_oov_after_eviction() -> None:
    state = CausalFeatureState(max_history=2)
    state.append_listen(uid=1, timestamp=1, raw_item_id=10, item_idx=0, artist_id=-1)
    state.append_listen(uid=1, timestamp=2, raw_item_id=11, item_idx=3, artist_id=5)
    assert state.history_summary(1)["history_oov_fraction"] == 0.5
    state.append_listen(uid=1, timestamp=3, raw_item_id=12, item_idx=4, artist_id=5)
    assert state.history_summary(1)["history_oov_fraction"] == 0.0


def test_foundation_ids_and_blocks_are_deterministic() -> None:
    assert len(BASE_FEATURE_NAMES) == 7
    assert foundation_request_id(1, 2, 3) == foundation_request_id(1, 2, 3)
    assert foundation_request_id(1, 2, 3) != foundation_request_id(1, 2, 4)
    assert time_block(217 * 86_400) == "update1"
    assert time_block(238 * 86_400) is None
