#!/usr/bin/env python3
"""CPU parity of alpha1 and native anchors against frozen robustness scores."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.ch3_motivation.code.compare_candidate_robust_readiness import (
    ATOL, RTOL, committed, read_selected, sha,
)

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'experiments/ch3_motivation/runs/candidate_robustness/attempt_1/quality'
METHODS={'reuse':'reuse','alpha1':'linear256_rows','rebuild':'rebuild'}


def compare(new_folder,old_folder,output):
    new_path,old_path=new_folder/'configuration.json',old_folder/'configuration.json'
    new,old=[json.loads(p.read_text()) for p in (new_path,old_path)]
    for key in ('edge','models','input_binding_sha256','population_sha256','maps_sha256','settings'):
        if new[key]!=old[key]:raise ValueError(f'Frozen old/new {key} differs')
    if new['base_map_name']!='linear256_rows' or new['method_strengths']['alpha1']!=1:
        raise ValueError('The exact alpha1 baseline is not linear256_rows')
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
        if set(a)!=set(b):raise ValueError(f'{method}: old/new observed events differ')
        keys=sorted(a)
        for key in keys:
            for field in ('label','appended_events','history_rows','query_gap'):
                if a[key][field]!=b[key][field]:raise ValueError(f'{method}: {field} differs at {key}')
        if not keys:
            reports[method]=dict(old_method=prior_name,events=0,passed=False,reason='No overlapping scored UID')
            continue
        x=np.array([a[k]['score'] for k in keys]);y=np.array([b[k]['score'] for k in keys])
        if not(np.isfinite(x).all() and np.isfinite(y).all()):raise ValueError('Nonfinite raw logit')
        error=np.abs(x-y);tolerance=ATOL+RTOL*np.abs(y);worst=int(np.argmax(error/tolerance))
        ok=bool(np.all(error<=tolerance));passed &= ok
        reports[method]=dict(old_method=prior_name,events=len(keys),feedback_users=len({k[0] for k in keys}),
            max_abs=float(error.max()),rms_abs=float(np.sqrt(np.mean(error**2))),
            failed_elements=int((error>tolerance).sum()),passed=ok,
            worst_event=dict(zip(('uid','query_timestamp','item_idx'),keys[worst],strict=True)),
            new_at_worst=float(x[worst]),old_at_worst=float(y[worst]),
            max_tolerance_ratio=float((error/tolerance).max()))
    result=dict(format='candidate_strength_readiness_comparison_v1',status='passed' if passed else 'failed_or_no_overlap',
        edge=new['edge'],quality_use_permitted=False,
        scope='CPU numerical parity of identical observed events only; no quality claim or independent replication.',
        base_map='linear256_rows',new_diagnostic=bool(new.get('diagnostic')),
        new_independent_check=bool(new.get('check_independent')),new_committed_blocks=len(new_records),
        new_scored_users=len(scored),matched_users=overlap,excluded_new_scored_users=sorted(set(scored)-set(overlap)),
        method_comparison=reports,tolerances=dict(atol=ATOL,rtol=RTOL,reference='frozen robustness raw logits'),
        sources=[dict(path=str(p.resolve()),sha256=sha(p)) for p in (new_path,old_path)]+new_sources+old_sources,
        comparison_source_sha256=sha(__file__),
        committed_reader_source_sha256=sha(Path(__file__).with_name('compare_candidate_robust_readiness.py')))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new',type=Path,required=True)
    parser.add_argument('--old',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();edge=json.loads((args.new/'configuration.json').read_text())['edge']
    result=compare(args.new,args.old or OLD/f'edge_{edge}',args.output)
    print(json.dumps({key:result[key] for key in ('status','edge','new_scored_users','matched_users','method_comparison')}))
    raise SystemExit(0 if result['status']=='passed' else 1)
