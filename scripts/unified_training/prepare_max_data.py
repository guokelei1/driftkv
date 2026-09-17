#!/usr/bin/env python3
"""Prepare the authorized 200k Max population using the existing Yambda rules."""
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hstu_kvcache.data.scale_population import UID_SELECTOR_NAMESPACE, uid_selector_digest

CONTRACT = ROOT / 'configs/unified_training_2026_09/max_data_preparation.yaml'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    c = yaml.safe_load(CONTRACT.read_text())
    out = ROOT / c['output']
    out.mkdir(parents=True, exist_ok=False)
    (out / 'shared').mkdir()
    started = time.time()
    def stage(name):
        print(name, flush=True)
        write(out / 'progress.json', {'stage': name, 'elapsed_seconds': time.time()-started})
    stage('verify_source_hashes')
    download = json.loads((ROOT / c['source_download_manifest']).read_text())
    sources = {}
    for row in download['files']:
        path = ROOT / row['output']
        assert path.stat().st_size == row['size'] and sha(path) == row['sha256']
        sources[path.stem] = {'path': str(path), 'sha256': row['sha256'],
                              'rows': pq.read_metadata(path).num_rows}
    con = duckdb.connect()
    con.execute(f"SET threads={c['execution']['preprocessing_threads']}")
    con.execute(f"SET memory_limit='{c['execution']['preprocessing_memory_limit']}'")
    con.execute(f"SET temp_directory='{out / '.spill'}'")
    con.execute('SET preserve_insertion_order=false')
    raw = sources['listens']['path']
    cutoff = c['population']['cutoff_seconds']
    assert c['population']['selector_namespace'] == UID_SELECTOR_NAMESPACE
    stage('select_eligible_population')
    eligible = con.execute(f"SELECT uid::UBIGINT,min(timestamp)::UBIGINT FROM read_parquet('{raw}') WHERE timestamp < {cutoff} GROUP BY uid").fetchall()
    ranked = sorted(eligible, key=lambda row: (uid_selector_digest(row[0]), row[0]))
    selected = ranked[:c['population']['users']]
    assert len(selected) == 200000
    users = pa.table({'uid': pa.array([r[0] for r in selected], type=pa.uint64()),
                      'selector_rank': pa.array(range(1,len(selected)+1), type=pa.uint32()),
                      'first_timestamp': pa.array([r[1] for r in selected], type=pa.uint64())})
    con.register('selected_users', users)
    scale = out / 'scales/max'
    scale.mkdir(parents=True)
    pq.write_table(users.sort_by('uid'), scale/'users.parquet', compression='zstd')
    del eligible, ranked, selected
    stage('write_selected_listens')
    con.execute(f"""COPY (
        SELECT (e.timestamp // 604800)::INTEGER AS week,e.uid::UBIGINT AS uid,u.selector_rank,
               e.timestamp::UBIGINT AS timestamp,e.item_id::UBIGINT AS raw_item_id,
               (1+(1-e.is_organic))::UTINYINT AS behavior,e.is_organic::UTINYINT AS is_organic,
               e.played_ratio_pct::USMALLINT AS played_ratio_pct,e.track_length_seconds::UINTEGER AS track_length_seconds
        FROM read_parquet('{raw}') e JOIN selected_users u USING(uid)
    ) TO '{out / 'shared/listens'}' (FORMAT PARQUET,PARTITION_BY(week),COMPRESSION ZSTD,ROW_GROUP_SIZE 262144)""")
    stage('write_selected_feedback')
    con.execute(f"""COPY (
        SELECT (e.timestamp // 604800)::INTEGER AS week,e.uid::UBIGINT AS uid,u.selector_rank,
               e.timestamp::UBIGINT AS timestamp,e.item_id::UBIGINT AS raw_item_id,
               e.label::UTINYINT AS label,e.is_organic::UTINYINT AS is_organic
        FROM (SELECT *,1 AS label FROM read_parquet('{sources['likes']['path']}')
              UNION ALL SELECT *,0 AS label FROM read_parquet('{sources['dislikes']['path']}')) e
        JOIN selected_users u USING(uid)
    ) TO '{out / 'shared/feedback'}' (FORMAT PARQUET,PARTITION_BY(week),COMPRESSION ZSTD,ROW_GROUP_SIZE 262144)""")
    stage('freeze_foundation_item_mapping')
    listen_glob = str(out / 'shared/listens/**/*.parquet')
    con.execute(f"""COPY (
        SELECT row_number() OVER(ORDER BY raw_item_id)::UBIGINT AS item_idx,raw_item_id
        FROM (SELECT DISTINCT raw_item_id FROM read_parquet('{listen_glob}',hive_partitioning=true)
              WHERE timestamp < {cutoff}) ORDER BY raw_item_id
    ) TO '{scale / 'item_mapping.parquet'}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    known = pq.read_metadata(scale/'item_mapping.parquet').num_rows
    dataset = {'dataset': 'yambda5b_max_200k_v1','contract': str(CONTRACT.relative_to(ROOT)),
        'scale':'max','rank_limit':200000,'users':200000,'foundation_items':known,
        'model':'16L_H320','context':1024,'shared_listens_glob':'../../shared/listens/**/*.parquet',
        'shared_feedback_glob':'../../shared/feedback/**/*.parquet','users_path':'users.parquet',
        'item_mapping_path':'item_mapping.parquet','timestamp_windows':'half_open',
        'oov_item_idx':0,'oov_bucket_start':known+1,'oov_buckets':256,
        'history_tie_order':'timestamp_raw_item_behavior',
        'training_authorized':False,'feedback_access':{'physical_sharding_does_not_authorize_label_use':True,
        'authorized_preparation_days_half_open':[0,301],'formal_training_authorized':False}}
    write(scale/'dataset.json',dataset)
    stage('audit_physical_store')
    stats = {}
    for kind in ['listens','feedback']:
        glob = str(out / f'shared/{kind}/**/*.parquet')
        rows = con.execute(f"SELECT (timestamp//86400)::INTEGER AS day,count(*) FROM read_parquet('{glob}',hive_partitioning=true) GROUP BY day ORDER BY day").fetchall()
        stats[kind] = {'rows':sum(r[1] for r in rows),'per_day':dict(rows)}
    # Check membership, rank consistency and source field domains across the materialized store.
    bad = con.execute(f"""SELECT count(*) FROM read_parquet('{listen_glob}',hive_partitioning=true) l
        LEFT JOIN selected_users u USING(uid) WHERE u.uid IS NULL OR l.selector_rank != u.selector_rank
        OR l.behavior NOT IN (1,2) OR l.is_organic NOT IN (0,1) OR l.timestamp>26000000""").fetchone()[0]
    assert bad == 0
    file_records = []
    stage('seal_data_files')
    for path in sorted(out.rglob('*.parquet')):
        file_records.append({'path':str(path.relative_to(out)),'bytes':path.stat().st_size,
                             'rows':pq.read_metadata(path).num_rows,'sha256':sha(path)})
    manifest = {'status':'max_200k_data_prepared','contract':str(CONTRACT.relative_to(ROOT)),
        'contract_sha256':sha(CONTRACT),'builder_sha256':sha(__file__),'source_files':sources,
        'users':200000,'selector_namespace':UID_SELECTOR_NAMESPACE,'foundation_items':known,
        'stats':stats,'files':file_records,'training_authorized':False,'elapsed_seconds':time.time()-started}
    write(out/'manifest.json',manifest)
    con.close()
    stage('complete')
    print(json.dumps({'users':200000,'known_items':known,'listens':stats['listens']['rows'],
                      'feedback':stats['feedback']['rows'],'minutes':(time.time()-started)/60}),flush=True)


if __name__ == '__main__':
    main()
