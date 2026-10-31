#!/usr/bin/env python3
"""Merge disjoint frozen-user shards; retain paired UID uncertainty."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import pyarrow as pa
import pyarrow.parquet as pq
from experiments.archive.partial_refresh.analyze import summarize
from hstu_kvcache.common.artifacts import write_json


def run(args):
    users, tables, layouts = [], [], set()
    for folder in args.inputs:
        result = json.loads((folder / 'summary.json').read_text())
        if result['status'] != 'complete':
            raise RuntimeError('incomplete shard')
        if result.get('cross_connection_ablation', False):
            raise RuntimeError('do not merge connection diagnostics into candidate quality')
        users.extend(result['selected_uids'])
        layouts.add(result['layout'])
        tables.append(pq.read_table(folder / 'scores.parquet'))
    if len(users) != len(set(users)) or len(layouts) != 1:
        raise RuntimeError('overlapping users or different layouts')
    scores = pa.concat_tables(tables).sort_by([('request_id', 'ascending')])
    quality = summarize(scores, sorted(users), repetitions=args.bootstrap, releases=['v3', 'v4', 'v5'])
    args.output.mkdir(parents=True, exist_ok=False)
    pq.write_table(scores, args.output / 'scores.parquet')
    write_json(args.output / 'summary.json', dict(status='complete', layout=next(iter(layouts)),
        selected_uids=sorted(users), quality=quality, input_directories=[str(p) for p in args.inputs]))
    print(json.dumps(quality['overall']['paired'], indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=1000)
    run(p.parse_args())
