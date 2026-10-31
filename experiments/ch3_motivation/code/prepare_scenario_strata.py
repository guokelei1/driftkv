#!/usr/bin/env python3
"""Publishing-time activity cohorts; no model score or future label is read."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.ch3_motivation.code.common import DATASET, DAY, quoted, write_json
from hstu_kvcache.common.artifacts import sha256


def causal_counts(listens,feedback,uids,cuts,*,threads=8):
    """One scan per input; all windows exclude the release timestamp itself."""
    connection=duckdb.connect()
    connection.execute(f"PRAGMA threads={int(threads)}")
    connection.register("selected",pa.table({"uid":pa.array(uids,type=pa.uint64())}))
    behavior_fields=[]
    feedback_fields=[]
    for edge,cut in cuts.items():
        behavior_fields.extend([
            f"count(*) FILTER (WHERE timestamp < {cut}) AS history_{edge}",
            f"count(*) FILTER (WHERE timestamp >= {cut-14*DAY} AND timestamp < {cut}) AS behavior14_{edge}"])
        feedback_fields.append(f"count(*) FILTER (WHERE timestamp >= {cut-14*DAY} AND timestamp < {cut}) AS feedback14_{edge}")
    behavior_rows=connection.execute(f"""
        SELECT uid,{','.join(behavior_fields)} FROM read_parquet('{quoted(listens)}',hive_partitioning=true)
        SEMI JOIN selected USING(uid) WHERE timestamp < {max(cuts.values())} GROUP BY uid
    """).fetch_arrow_table().to_pylist()
    feedback_rows=connection.execute(f"""
        WITH events AS (
          SELECT DISTINCT uid,timestamp,raw_item_id
          FROM read_parquet('{quoted(feedback)}',hive_partitioning=true) SEMI JOIN selected USING(uid)
          WHERE timestamp >= {min(cuts.values())-14*DAY} AND timestamp < {max(cuts.values())}
        ) SELECT uid,{','.join(feedback_fields)} FROM events GROUP BY uid
    """).fetch_arrow_table().to_pylist()
    connection.close()
    behavior={int(row['uid']):row for row in behavior_rows}
    feedback={int(row['uid']):row for row in feedback_rows}
    rows=[]
    for uid in uids:
        for edge,cut in cuts.items():
            row=behavior.get(uid,{})
            rows.append(dict(uid=int(uid),edge=int(edge),release_timestamp=int(cut),
                pre_release_history_events=int(row.get(f'history_{edge}',0)),
                retained_history_rows=min(1024,int(row.get(f'history_{edge}',0))),
                pre14_behavior_events=int(row.get(f'behavior14_{edge}',0)),
                pre14_feedback_events=int(feedback.get(uid,{}).get(f'feedback14_{edge}',0))))
    return rows


DEFINITIONS=dict(
    long_active=dict(retained_history_rows_at_least=512,pre14_behavior_events_at_least=32,pre14_feedback_events_at_least=0),
    long_feedback_active=dict(retained_history_rows_at_least=512,pre14_behavior_events_at_least=32,pre14_feedback_events_at_least=4),
    long_feedback_dense=dict(retained_history_rows_at_least=512,pre14_behavior_events_at_least=128,pre14_feedback_events_at_least=16),
    full_window_feedback_active=dict(retained_history_rows_at_least=1024,pre14_behavior_events_at_least=32,pre14_feedback_events_at_least=4))


def subgroup_memberships(rows,*,seed=20261026):
    groups={}
    for edge in sorted({row['edge'] for row in rows}):
        groups[str(edge)]={}
        for name,definition in DEFINITIONS.items():
            users=[row['uid'] for row in rows if row['edge']==edge
                and row['retained_history_rows']>=definition['retained_history_rows_at_least']
                and row['pre14_behavior_events']>=definition['pre14_behavior_events_at_least']
                and row['pre14_feedback_events']>=definition['pre14_feedback_events_at_least']]
            users.sort(key=lambda uid:hashlib.sha256(f'ch3-precut-cohort/{seed}/{edge}/{name}/{uid}'.encode()).digest())
            groups[str(edge)][name]=dict(users=len(users),uids=users,first_2000_uid_hashes=users[:2000],
                has_at_least_2000_users=len(users)>=2000)
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population',type=Path,default=ROOT/'experiments/ch3_motivation/inputs/population.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    started=time.perf_counter()
    population=json.loads(args.population.read_text())
    users=sorted(set(population['development']+population['mismatch']+population['confirmation']+population['reserve']))
    if set(users)&set(population['fit']+population['validation']):
        raise ValueError('scenario candidates overlap fitting/validation roles')
    dataset=json.loads(DATASET.read_text())
    cuts={edge:(231+(edge-1)*14)*DAY for edge in range(1,6)}
    rows=causal_counts((DATASET.parent/dataset['shared_listens_glob']).resolve(),
        (DATASET.parent/dataset['shared_feedback_glob']).resolve(),users,cuts,threads=args.threads)
    args.output.mkdir(parents=True,exist_ok=True)
    features=args.output/'pre_release_features.parquet'
    pq.write_table(pa.Table.from_pylist(rows),features)
    value=dict(role='publishing-time scenario candidates only; no model quality evaluated',
        source_population=dict(path=str(args.population.resolve()),sha256=sha256(args.population)),
        dataset=dict(path=str(DATASET),sha256=sha256(DATASET)),source_sha256=sha256(Path(__file__)),
        definitions=DEFINITIONS,subgroups_by_edge=subgroup_memberships(rows),
        features=dict(path=str(features.resolve()),sha256=sha256(features)),
        population_users=len(users),fit_validation_excluded=True,
        timestamp_rule='all counts strictly before release; recent window [release-14days,release); feedback unique(uid,timestamp,item)',
        seed=20261026,model_scores_read=False,future_feedback_read=False,
        scope='candidate scenario definitions; choosing a final example after baseline qualification remains outcome-conditioned diagnostic',
        seconds=time.perf_counter()-started)
    write_json(args.output/'scenario_strata_manifest.json',value)
    print(json.dumps(dict(users=len(users),seconds=value['seconds'],
        cohort_counts={edge:{name:entry['users'] for name,entry in groups.items()} for edge,groups in value['subgroups_by_edge'].items()})),flush=True)


if __name__=='__main__':
    main()
