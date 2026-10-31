#!/usr/bin/env python3
"""CPU-only score parity on overlapping UIDs; never a quality evaluation.

Reads committed blocks only. New diagnostic UIDs absent from the old4000 are
reported and excluded; every event of each overlapping UID must match exactly.
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
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'experiments/ch3_motivation/runs/real_quality_angles/attempt_1/quality_cap1024'
METHODS={'reuse':'reuse','f1':'f1','f2':'f2','linear100_rows':'kv_linear'}
ATOL,RTOL=1e-6,1e-5


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def committed(folder,config):
    frozen=digest(config);sources=[];records=[]
    summary_path=folder/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        if summary['status']!='complete':raise ValueError('Summary exists but run is not complete')
        manifests=[(folder/a['manifest'],a) for a in summary['artifacts']]
        sources.append(dict(path=str(summary_path.resolve()),sha256=sha(summary_path)))
    else:manifests=[(p,None) for p in sorted((folder/'blocks').glob('part_*.json'))]
    for manifest,artifact in manifests:
        if artifact and sha(manifest)!=artifact['manifest_sha256']:raise ValueError('Manifest SHA differs from complete summary')
        record=json.loads(manifest.read_text());block=manifest.with_suffix('.parquet')
        if (record['configuration_sha256']!=frozen or sha(block)!=record['score_sha256']
                or (artifact and sha(block)!=artifact['sha256'])):
            raise ValueError('Committed block/configuration hash differs')
        sources.append(dict(path=str(manifest.resolve()),sha256=sha(manifest)))
        sources.append(dict(path=str(block.resolve()),sha256=record['score_sha256']))
        records.append((block,record))
    if not records:raise ValueError('No committed blocks yet')
    return records,sources


def read_selected(records,uids,methods):
    result={m:{} for m in methods}
    for path,record in records:
        selected=sorted(set(record['uids'])&set(uids))
        if not selected:continue
        rows=pq.read_table(path,filters=[('uid','in',selected),('method','in',list(methods))]).to_pylist()
        for row in rows:
            key=tuple(int(row[k]) for k in ('uid','query_timestamp','item_idx'))
            target=result[row['method']]
            if key in target:raise ValueError('Duplicate observation/method in committed scores')
            target[key]=row
    return result


def compare(new_folder,old_folder,output):
    new_path,old_path=new_folder/'configuration.json',old_folder/'configuration.json'
    new,old=[json.loads(p.read_text()) for p in (new_path,old_path)]
    for key in ('edge','models','input_binding_sha256','checkpoint_sha256'):
        if new[key]!=old[key]:raise ValueError(f'Old/new frozen {key} differs')
    if new['settings']['history_length']!=old['settings']['history_length']:
        raise ValueError('History caps differ')
    if new['old_linear_checkpoint_sha256']!=old['simple_maps_sha256']:
        raise ValueError('The compared raw linear checkpoint differs')
    new_records,new_sources=committed(new_folder,new)
    old_records,old_sources=committed(old_folder,old)
    actual=read_selected(new_records,new['users'],METHODS)
    scored=sorted({key[0] for rows in actual.values() for key in rows})
    overlap=sorted(set(scored)&set(old['users']))
    previous=read_selected(old_records,overlap,METHODS.values()) if overlap else {}
    reports={};passed=bool(overlap)
    for method,prior_name in METHODS.items():
        a={k:r for k,r in actual[method].items() if k[0] in overlap}
        b=previous.get(prior_name,{})
        if set(a)!=set(b):raise ValueError(f'{method}: old/new events differ within overlapping UIDs')
        keys=sorted(a)
        for key in keys:
            for field in ('label','appended_events','history_rows','query_gap'):
                if a[key][field]!=b[key][field]:raise ValueError(f'{method}: old/new {field} differs at{key}')
        if not keys:
            reports[method]=dict(old_method=prior_name,events=0,passed=False,reason='No overlapping scored UID')
            continue
        x=np.array([a[k]['score'] for k in keys]);y=np.array([b[k]['score'] for k in keys])
        if not (np.isfinite(x).all() and np.isfinite(y).all()):raise ValueError('Nonfinite old/new logit')
        error=np.abs(x-y);tol=ATOL+RTOL*np.abs(y);worst=int(np.argmax(error/tol))
        ok=bool(np.all(error<=tol));passed &= ok
        reports[method]=dict(old_method=prior_name,events=len(keys),feedback_users=len({k[0] for k in keys}),
            max_abs=float(error.max()),rms_abs=float(np.sqrt(np.mean(error**2))),
            relative_rms=float(np.sqrt(np.mean(error**2))/max(np.sqrt(np.mean(y**2)),1e-30)),
            failed_elements=int((error>tol).sum()),passed=ok,
            worst_event=dict(zip(('uid','query_timestamp','item_idx'),keys[worst],strict=True)),
            new_at_worst=float(x[worst]),old_at_worst=float(y[worst]),
            max_tolerance_ratio=float((error/tol).max()))
    result=dict(format='candidate_robust_readiness_comparison_v1',status='passed' if passed else 'failed_or_no_overlap',
        edge=new['edge'],quality_use_permitted=False,
        scope='CPU numeric parity for identical observed events of overlapping UIDs only; not population quality or an independent replication.',
        new_diagnostic=bool(new.get('diagnostic')),new_independent_check=bool(new.get('check_independent')),
        new_committed_blocks=len(new_records),new_scored_users=len(scored),matched_users=overlap,
        excluded_new_scored_users=sorted(set(scored)-set(overlap)),method_comparison=reports,
        tolerances=dict(atol=ATOL,rtol=RTOL,reference='old frozen logits'),
        sources=[dict(path=str(p.resolve()),sha256=sha(p)) for p in (new_path,old_path)]+new_sources+old_sources,
        comparison_source_sha256=sha(__file__))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new',type=Path,required=True)
    parser.add_argument('--old',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    edge=json.loads((args.new/'configuration.json').read_text())['edge']
    result=compare(args.new,args.old or OLD/f'edge_{edge}',args.output)
    print(json.dumps({k:result[k] for k in ('status','edge','new_scored_users','matched_users','excluded_new_scored_users','method_comparison')}))
    raise SystemExit(0 if result['status']=='passed' else 1)
