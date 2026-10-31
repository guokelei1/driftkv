#!/usr/bin/env python3
"""Freeze two causal candidate attributes, then count real-request coverage.

No model predictions are opened. Popularity uses all Large-scale users'
listening events in the14days strictly before each release, never fit-only
support or future feedback labels.
"""
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
import math
from pathlib import Path
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
DAY=86400


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def count(rows):
    users={r['uid'] for r in rows};pos={r['uid'] for r in rows if r['label']==1};neg={r['uid'] for r in rows if r['label']==0}
    return dict(feedback_users=len(users),events=len(rows),positive_events=sum(r['label']==1 for r in rows),
        negative_events=sum(r['label']==0 for r in rows),positive_users=len(pos),negative_users=len(neg),
        both_label_users=len(pos&neg),mechanism_support_permitted=len(users)>=512 and bool(pos) and bool(neg))


def run(population_path,output,threads=4):
    population_path,output=Path(population_path),Path(output)
    population=json.loads(population_path.read_text());users=population['users'];selected=set(users)
    if len(users)!=4000 or len(selected)!=4000:raise ValueError('fixed4000 population required')
    dataset_path=ROOT/'data/processed/yambda500m_unified_v1/scales/large/dataset.json'
    dataset=json.loads(dataset_path.read_text());limit=int(dataset['rank_limit'])
    shared=(dataset_path.parent/dataset['shared_listens_glob']).resolve().parent.parent
    # The source builder partitions with week=timestamp//604800.
    source_builder=ROOT/'src/hstu_kvcache/data/yambda/build_scales.py'
    definitions=dict(format='causal_candidate_group_definitions_v1',population_sha256=sha(population_path),
        users=users,history_cap=1024,edges={},
        groups=dict(history_seen='candidate item_idx appears in this UID retained last1024 events with timestamp<release',
            history_unseen='candidate item_idx does not appear in that frozen cutover prefix',
            popular='top20% of items observed in preceding14days among ALL Large-scale users; count descending then raw_item_id ascending; ceil count',
            long_tail='all other candidate items, including items with zero preceding14day events'),
        popularity_rank_fraction=.2,popularity_rank_tie_break='raw_item_id ascending',
        popularity_time_window='[release-14days,release), strict exclusion of release timestamp',
        popularity_population=f'all source users with selector_rank<={limit}; independent of fixed4000 test membership',
        popularity_event_type='real behavior/listening events, counting repeated listens; no feedback labels',
        dataset=dict(path=str(dataset_path),sha256=sha(dataset_path)),
        partition_builder=dict(path=str(source_builder),sha256=sha(source_builder)),
        script_sha256=sha(__file__),score_values_read=False,
        group_rules_frozen_before_future_requests_parsed=True,original_roles_modified=False,
        threshold_search_performed=False,minimum_feedback_users_for_mechanism=512)
    for edge in [2,3]:
        entry=population['edges'][str(edge)];cfg=entry['settings'];cut=int(cfg['release_day'])*DAY
        weeks=range((cut-14*DAY)//(7*DAY),(cut-1)//(7*DAY)+1)
        files=[p for week in weeks for p in sorted(shared.glob(f'week={week}/*.parquet'))]
        if len(files)!=len(list(weeks)):raise ValueError('expected one existing behavior shard per required week')
        for name,ref in entry['inputs'].items():
            if sha(ref['path'])!=ref['sha256']:raise ValueError('frozen input binding changed')
        definitions['edges'][str(edge)]=dict(release_timestamp=cut,start_timestamp=cut-14*DAY,
            rank_limit=limit,behavior_shards=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files],
            native_inputs=entry['inputs'])
    output.mkdir(parents=True,exist_ok=True);definition_path=output/'definitions.json'
    if definition_path.exists():
        if json.loads(definition_path.read_text())!=definitions:raise ValueError('definitions changed; use another attempt')
    else:write(definition_path,definitions)
    for edge in [2,3]:
        edge_definition=dict((key,value) for key,value in definitions.items() if key not in ['edges'])
        edge_definition.update(edge=edge,source=definitions['edges'][str(edge)],global_definition_sha256=sha(definition_path))
        write(output/f'edge_{edge}/definition.json',edge_definition)
    # Definitions and all causal sources are now frozen, before future requests.
    con=duckdb.connect();con.execute(f'PRAGMA threads={int(threads)}')
    con.execute("PRAGMA memory_limit='2GB'")
    coverage=dict(format='causal_candidate_group_coverage_v1',definition_sha256=sha(definition_path),
        population_sha256=sha(population_path),score_values_read=False,edges={})
    for edge in [2,3]:
        started=time.perf_counter();entry=definitions['edges'][str(edge)];cfg=population['edges'][str(edge)]
        directory=output/f'edge_{edge}';directory.mkdir(exist_ok=True)
        files=[v['path'] for v in entry['behavior_shards']]
        con.execute('DROP TABLE IF EXISTS frequencies')
        con.execute('''CREATE TEMP TABLE frequencies AS
            SELECT raw_item_id,count(*)::BIGINT AS global_pre14_item_events
            FROM read_parquet(?,hive_partitioning=true)
            WHERE selector_rank<=? AND timestamp>=? AND timestamp<? GROUP BY raw_item_id''',
            [files,limit,entry['start_timestamp'],entry['release_timestamp']])
        observed_items,total_events=con.execute('SELECT count(*),sum(global_pre14_item_events) FROM frequencies').fetchone()
        head_count=math.ceil(int(observed_items)*.2)
        con.execute('''CREATE OR REPLACE TEMP TABLE ranked_frequencies AS
            SELECT *,row_number() OVER(ORDER BY global_pre14_item_events DESC,raw_item_id ASC)::BIGINT AS global_pre14_item_rank
            FROM frequencies''')
        frequency_path=directory/'global_pre14_item_frequency.parquet'
        pq.write_table(con.execute('SELECT *,global_pre14_item_rank<=? AS popular FROM ranked_frequencies ORDER BY global_pre14_item_rank',
            [head_count]).fetch_arrow_table(),frequency_path,compression='zstd')
        requests=[r for r in pq.read_table(cfg['inputs']['requests']['path']).to_pylist() if int(r['uid']) in selected]
        keys=[(r['uid'],r['query_timestamp'],r['item_idx']) for r in requests]
        if len(keys)!=len(set(keys)):raise ValueError('duplicate/conflicting real observation keys')
        raw_ids=sorted({int(r['raw_item_id']) for r in requests})
        con.register('request_items',pa.table({'raw_item_id':pa.array(raw_ids,type=pa.uint64())}))
        lookup={int(item):(int(n),int(rank),int(rank)<=head_count) for item,n,rank in con.execute(
            'SELECT raw_item_id,global_pre14_item_events,global_pre14_item_rank FROM ranked_frequencies SEMI JOIN request_items USING(raw_item_id)').fetchall()}
        with np.load(cfg['inputs']['history']['path'],allow_pickle=False) as history:
            offsets,timestamps,items=history['offsets'],history['timestamps'],history['item_ids'];history_seen={}
            for index,uid in enumerate(history['uids']):
                uid=int(uid)
                if uid not in selected:continue
                a,b=map(int,offsets[index:index+2]);times=timestamps[a:b]
                cut=int(np.searchsorted(times,entry['release_timestamp'],side='left'));n=min(cut,1024)
                if cut!=population['edges'][str(edge)]['user_metadata'][str(uid)]['history_count']:
                    raise ValueError('cutover prefix differs from frozen UID metadata')
                history_seen[uid]=set(map(int,items[a+cut-n:a+cut]))
        rows=[];groups={name:[] for name in definitions['groups']}
        for request in requests:
            uid=int(request['uid']);seen=int(request['item_idx']) in history_seen[uid]
            n,rank,popular=lookup.get(int(request['raw_item_id']),(0,0,False))
            rows.append(dict(uid=uid,query_timestamp=int(request['query_timestamp']),item_idx=int(request['item_idx']),
                raw_item_id=int(request['raw_item_id']),history_seen=seen,popular=popular,
                global_pre14_item_events=n,global_pre14_item_rank=rank))
            groups['history_seen' if seen else 'history_unseen'].append(request)
            groups['popular' if popular else 'long_tail'].append(request)
        schema=pa.schema([('uid',pa.uint64()),('query_timestamp',pa.int64()),('item_idx',pa.int64()),
            ('raw_item_id',pa.uint64()),('history_seen',pa.bool_()),('popular',pa.bool_()),
            ('global_pre14_item_events',pa.int64()),('global_pre14_item_rank',pa.int64())])
        mapping_path=directory/'eventgroups.parquet';pq.write_table(pa.Table.from_pylist(rows,schema=schema),mapping_path)
        cutoff=con.execute('SELECT raw_item_id,global_pre14_item_events FROM ranked_frequencies WHERE global_pre14_item_rank=?',[head_count]).fetchone()
        paired_classes={}
        for attribute,a,b in [('history_familiarity','history_seen','history_unseen'),('popularity','popular','long_tail')]:
            common={r['uid'] for r in groups[a]}&{r['uid'] for r in groups[b]}
            paired_classes[attribute]=dict(common_feedback_users=len(common),uids=sorted(common),
                mechanism_support_permitted=len(common)>=512,
                groups={name:count([r for r in groups[name] if r['uid'] in common]) for name in [a,b]},
                scope='Conditional same UID intersection with feedback in both candidate classes; fixed4000 population unchanged')
        result=dict(all=count(requests),groups={name:count(v) for name,v in groups.items()},paired_classes=paired_classes,
            resident_users=4000,observed_global_pre14_items=int(observed_items),global_pre14_behavior_events=int(total_events),
            popular_item_count=head_count,popular_last_item_raw_id=int(cutoff[0]),popular_min_count=int(cutoff[1]),
            frequency=dict(path=str(frequency_path.resolve()),sha256=sha(frequency_path)),
            event_mapping=dict(path=str(mapping_path.resolve()),sha256=sha(mapping_path),rows=len(rows)),
            seconds=time.perf_counter()-started)
        coverage['edges'][str(edge)]=result
        write(directory/'manifest.json',dict(definition=dict(path=str((directory/'definition.json').resolve()),sha256=sha(directory/'definition.json')),
            event_mapping=result['event_mapping'],frequency=result['frequency'],population_sha256=sha(population_path),
            global_definition_sha256=sha(definition_path),score_values_read=False,coverage=result))
        print(json.dumps(dict(edge=edge,global_items=observed_items,global_events=total_events,groups=result['groups'],seconds=result['seconds'])),flush=True)
    con.close();write(output/'coverage.json',coverage)
    lines=['# Causal candidate groups: coverage only','',
        'Fixed4000 residents. Rules were frozen before reading future requests; no model scores were opened. History seen/unseen uses retained cutover1024 events. Popularity uses all Large users’ listening events in preceding14days, not a fit-only catalog.','',
        '| Edge | Group | Feedback UIDs | Events | Positive/negative | At least512 UIDs |','|---|---|---:|---:|---:|---|']
    for edge,result in coverage['edges'].items():
        for name,values in result['groups'].items():
            lines.append(f"| {edge} | {name} | {values['feedback_users']} | {values['events']} | {values['positive_events']}/{values['negative_events']} | {'yes' if values['mechanism_support_permitted'] else 'diagnostic only'} |")
        lines.append(f"\nEdge{edge}: {result['global_pre14_behavior_events']} global prerelease events over {result['observed_global_pre14_items']} observed items; fixed top20%={result['popular_item_count']} items; last included count={result['popular_min_count']}.\n")
        for name,values in result['paired_classes'].items():
            lines.append(f"Same UID both-class support {name}: {values['common_feedback_users']} users; eligible={values['mechanism_support_permitted']}.\n")
    lines+=['','Join each score method to eventgroups.parquet on(uid,query_timestamp,item_idx). `history_seen=True/False` and `popular=True/False` each partition the same original observations. A UID can contribute to both groups: bootstrap all4000 UID clusters with shared multiplicities for group comparisons.','',
        'Zero-pre14-frequency items are long_tail with rank0. Count ties at the20% boundary use raw_item_id ascending. Thresholds were not searched or changed for coverage or quality. Groups with fewer than512 actual feedback UIDs are diagnostic only. No differences in method performance are presumed.']
    (output/'coverage.md').write_text('\n'.join(lines)+'\n')
    return coverage


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();run(args.population,args.output,args.threads)
