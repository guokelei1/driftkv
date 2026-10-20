#!/usr/bin/env python3
"""Extend frozen V4 calibration sources only inside the original fit256 role.

Reuse original80 rows unchanged. Capture the next64 and last128 separately so
fit128/256 packs contain exactly their fit+16validation rows and exact summed
operation counts. New source replay and serialization timers remain separate
from the historical80-row expense.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]

import torch
from design_two.common import load_models,setup_device
from design_two.prepare import load_history,load_panel
from query_relay_2026_10.capture import capture_candidate_rows,load_prior_candidates
from unified_reuse_2026_09.common import sha256,write_json


def merge_geometry(left,right):
    out={}
    for name in ('native','code_reads','explicit_updates'):
        counts=Counter()
        for record in left[name]+right[name]:
            counts[tuple((k,v) for k,v in record.items() if k!='count')]+=record['count']
        out[name]=[dict(key,count=value) for key,value in sorted(counts.items())]
    key='gate_query_normalization_flops_included_in_reader_total'
    out[key]=left[key]+right[key]
    out['scope']='Exact sum of disjoint original and newly captured UID operation geometry'
    return out


@torch.no_grad()
def run(args):
    began=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=False)
    branch=ROOT/'results/query_relay_2026_10/probes/A_itemgate6_rms_roll4'
    panel_path=ROOT/'results/design_two_2026_10/benchmark_panel'
    panel,roles,requests=load_panel(panel_path)
    original_path=branch/'source_v4/source.pt'
    original=torch.load(original_path,map_location='cpu',weights_only=False,mmap=True)
    if len(roles['fit'])!=256 or original['settings']['fit_uids']!=roles['fit'][:64] \
            or original['settings']['validation_uids']!=roles['validation'][:16]:
        raise ValueError('original fit/validation membership differs')
    selected=roles['fit']+roles['validation'][:16]
    if set(selected)&set(roles['evaluation']):raise ValueError('capture intersects evaluation')
    old_sha=sha256(original_path)
    if old_sha!=json.loads((branch/'calibration/v4/calibration.json').read_text())['source_pack_sha256']:
        raise ValueError('original source identity mismatch')
    device=setup_device(0,2)
    models=load_models(panel,device,last_version=4)
    adapters,checkpoints=load_prior_candidates(branch/'calibration',['v3'],device)
    if checkpoints['v3']['sha256']!=original['prior_checkpoints']['v3']['sha256']:
        raise ValueError('source predecessor changed')
    histories=load_history(panel_path,requested_uids=roles['fit'][64:])
    report=dict(status='running',target='v4',source_original=str(original_path),source_original_sha256=old_sha,
        original_rows_reused=80,original_capture_seconds=json.loads((branch/'source_v4/summary.json').read_text())['elapsed_seconds'],
        panel_sha256=sha256(panel_path/'binding.json'),entry_sha256=sha256(Path(__file__)),
        startup_and_original_load_seconds=time.perf_counter()-began,phases=[],outputs={},
        scope='Original fit256 and validation16 only. V3 predecessor frozen. No evaluation labels or future events. Existing80 are not recaptured.')
    merged=dict(original['rows']);costs=Counter(original['costs']);records=list(original['records']);geometry=original['geometry']
    for count,subset in ((128,roles['fit'][64:128]),(256,roles['fit'][128:256])):
        start=time.perf_counter()
        rows,new_cost,new_records,new_geometry=capture_candidate_rows(models,adapters,histories,requests,subset,'v4',
            panel['known_vocab_size'],workload='behavior',stride=16)
        torch.cuda.synchronize()
        report['phases'].append(dict(name='additional_source_teacher_capture',new_fit_rows=len(subset),
            resulting_fit_users=count,wall_seconds=time.perf_counter()-start,costs=new_cost))
        if set(merged)&set(rows):raise ValueError('overlapping source extension')
        merged.update(rows);costs.update(new_cost);records+=new_records;geometry=merge_geometry(geometry,new_geometry)
        settings=dict(original['settings'],fit_uids=roles['fit'][:count],validation_uids=roles['validation'][:16],
            source_extension=dict(original_path=str(original_path),original_sha256=old_sha,reused_rows=80,new_rows=count-64))
        wanted=set(settings['fit_uids']+settings['validation_uids'])
        assert set(merged)==wanted
        payload=dict(rows=dict(merged),costs=dict(costs),records=list(records),geometry=geometry,settings=settings,prior_checkpoints=checkpoints)
        path=args.output/f'source_fit{count}.pt'
        start=time.perf_counter();torch.save(payload,path);digest=sha256(path)
        report['phases'].append(dict(name='combined_source_save_and_hash',resulting_fit_users=count,
            wall_seconds=time.perf_counter()-start,file_bytes=path.stat().st_size))
        report['outputs'][str(count)]=dict(path=str(path),sha256=digest,users=len(merged),
            fit_users=count,validation_users=16,costs=dict(costs),geometry=geometry)
        write_json(args.output/'summary.json',report)
        print(json.dumps(dict(status='source_extended',fit_users=count,rows=len(merged),path=str(path))),flush=True)
    report.update(status='complete',new_execution_wall_seconds=time.perf_counter()-began,
        timer_scope='This actual command includes common startup and both outputs. Historical80-row capture listed once separately, never included in new_execution_wall_seconds. Separate capture/save phases permit standalone128/256 accounting without double-charging80.')
    write_json(args.output/'summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
