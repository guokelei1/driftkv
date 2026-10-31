from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.prepare_scenario_strata import causal_counts, subgroup_memberships


def test_activity_windows_exclude_release_and_deduplicate_feedback(tmp_path):
    listens=tmp_path/'listens.parquet'
    feedback=tmp_path/'feedback.parquet'
    pq.write_table(pa.Table.from_pylist([dict(uid=7,timestamp=t) for t in (1,19,20,21)]+[dict(uid=8,timestamp=19)]),listens)
    pq.write_table(pa.Table.from_pylist([dict(uid=7,timestamp=t,raw_item_id=3) for t in (19,19,20,21)]),feedback)
    rows=causal_counts(listens,feedback,[7],{1:20},threads=1)
    assert rows==[dict(uid=7,edge=1,release_timestamp=20,pre_release_history_events=2,
        retained_history_rows=2,pre14_behavior_events=2,pre14_feedback_events=1)]


def test_scenario_thresholds_use_only_declared_pre_release_features():
    common=dict(edge=1,retained_history_rows=1024,pre14_behavior_events=128,pre14_feedback_events=16)
    rows=[dict(common,uid=7),dict(common,uid=8,pre14_feedback_events=0),
          dict(common,uid=9,retained_history_rows=511)]
    groups=subgroup_memberships(rows)['1']
    assert set(groups['long_active']['uids'])=={7,8}
    assert groups['long_feedback_dense']['uids']==[7]
    assert groups['full_window_feedback_active']['uids']==[7]
