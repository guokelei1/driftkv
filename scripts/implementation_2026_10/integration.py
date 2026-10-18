#!/usr/bin/env python3
"""Frozen three-publication engineering equivalence on real causal histories.

Selection uses only pre-D259 history geometry. Both the retained feedback and
registered next-item16/16 paths remain unchanged. No fitting, new acceptance
population or general-quality claim is made by this numerical integration run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from design_two.common import END, RELEASES, load_models, setup_device
from design_two.prepare import load_history, load_panel
from hstu_kvcache.design_two.replay import NativeTrajectory
from implementation_2026_10.backend import (ImplementationConfig, InferenceAdapter,
                                           configure_prefix)
from implementation_2026_10.prepare import (compile_item_tables, fixed_features_from_bank,
                                           project_fixed_item_terms)
from query_relay_2026_10.evaluate import load_candidates
from query_relay_2026_10.precision import StableReleasePrefix
from query_relay_2026_10.publish import load_prefix
from query_relay_2026_10.session import RelaySession
from query_relay_2026_10.workload import candidate_rows
from unified_reuse_2026_09.common import sha256, write_json

PANEL = ROOT/'results/design_system_2026_10/phase2/representative4096/panel'
BASE = ROOT/'results/query_relay_2026_10/probes/A_itemgate6_rms_roll4'
RESULTS = ROOT/'results/implementation_2026_10'
ATOL, RTOL = 3e-5, 3e-5


def utc():
    return datetime.now(timezone.utc).isoformat()


def register(output):
    if output.exists():
        raise FileExistsError(output)
    panel, groups, _ = load_panel(PANEL)
    histories = load_history(PANEL, requested_uids=groups['evaluation'])
    buckets = defaultdict(list)
    profiles = {}
    at = RELEASES[0][1]
    for uid in sorted(groups['evaluation']):
        times = histories.rows[uid][0]
        before = int(np.searchsorted(times, at, side='left'))
        length = min(1024, before)
        recent = before > 0 and at-int(times[before-1]) <= 72*3600
        key = ('short_0_256' if length <= 256 else 'medium_257_767' if length < 768
               else 'long_active72h' if recent else 'long_inactive72h')
        buckets[key].append(uid)
        profiles[uid] = dict(uid=uid, initial_length=length, recent72h=bool(recent), stratum=key)
    chosen = []
    quotas = {}
    for key in ('short_0_256', 'medium_257_767', 'long_active72h', 'long_inactive72h'):
        take = buckets[key][:8]
        quotas[key] = len(take)
        chosen.extend(take)
    if len(chosen) < 32:
        chosen += [u for u in sorted(groups['evaluation']) if u not in chosen][:32-len(chosen)]
    assert len(chosen) == len(set(chosen)) == 32
    value = dict(status='registered_before_integration', created_utc=utc(),
        objective='Equivalent implementation of frozen D1/D2/D3, not a new quality search',
        panel=str(PANEL), panel_sha256=sha256(PANEL/'binding.json'),
        frozen_candidate=str(BASE), freeze_sha256=sha256(BASE/'freeze.json'),
        selected_uids=chosen, profiles=[profiles[u] for u in chosen],
        population_stratum_counts={k:len(v) for k,v in buckets.items()}, requested_stratum_quota=8,
        selected_initial_quotas=quotas,
        selection='First8 UID per pre-D259 length/activity bucket; deterministic fill to32. No post-cut behavior, requests, predictions or labels used to select.',
        protocols=['behavior_stride16_candidates16', 'original_feedback'],
        versions=[v for v,_ in RELEASES], configuration=dict(reader_backend='triton',
            pack_item_gates=True, row_chunk=128, item_chunk=16384, query_batch=16),
        comparisons=['reference', 'reader_only', 'all_optimized'],
        numerical_contract=dict(final_logit_atol=ATOL,final_logit_rtol=RTOL,
            native_and_compact_elementwise_atol=ATOL,native_and_compact_elementwise_rtol=RTOL,
            note='FP32 implementation equivalence, not bitwise equality. Near-zero compact outputs require an absolute tolerance; raw read magnitudes are reported separately.'),
        invariants=['same requests, labels, items and strict-past timestamps',
            'same event cursor, FIFO lengths, coded prefix and state format',
            'same native operation ledger and method service counters',
            'current-born rows remain zero-code within the birth version',
            'no candidate KV writes; corrected behavior writes in every branch'],
        exclusions=['No training or model/role/workload tuning',
            'No new80/20 quality claim from32users',
            'No loaded production timing/throughput claim',
            'Original-feedback quality/cost failure remains an independent boundary'])
    write_json(output, value)
    print(json.dumps(dict(status='registered',path=str(output),users=chosen,quotas=quotas)),flush=True)


def difference(actual, expected, *, check=True):
    if actual.shape != expected.shape:
        raise RuntimeError('Implementation changed state geometry')
    if not actual.numel():
        return dict(max_abs=0., rms=0., relative_l2=0., reference_abs_max=0., passed=True)
    if check:
        torch.testing.assert_close(actual,expected,atol=ATOL,rtol=RTOL)
    delta=actual.double()-expected.double()
    return dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                relative_l2=float(delta.norm()/expected.double().norm().clamp_min(1e-30)),
                reference_abs_max=float(expected.abs().max()),
                passed=bool(torch.isfinite(actual).all()))


def state_check(reference, actual):
    for key in ('state_format','coded_rows'):
        if getattr(reference,key) != getattr(actual,key):
            raise RuntimeError(f'Implementation changed {key}')
    if reference.trajectory.cursor != actual.trajectory.cursor:
        raise RuntimeError('Implementation changed the causal event cursor')
    return dict(format=reference.state_format, coded_rows=reference.coded_rows,
        history_rows=reference.cache.seq_len,
        native_k=difference(actual.cache.k,reference.cache.k),
        native_v=difference(actual.cache.v,reference.cache.v),
        compact=None if reference.state is None else difference(actual.state.codes,reference.state.codes))


@torch.no_grad()
def run(args):
    start=time.perf_counter()
    registered=json.loads(args.registration.read_text())
    if registered['panel_sha256'] != sha256(PANEL/'binding.json') \
            or registered['freeze_sha256'] != sha256(BASE/'freeze.json'):
        raise ValueError('Frozen inputs differ from registration')
    args.output.mkdir(parents=True,exist_ok=False)
    panel,groups,feedback=load_panel(PANEL)
    chosen=registered['selected_uids']
    if not set(chosen).issubset(groups['evaluation']):
        raise ValueError('Registered users no longer have the frozen evaluation role')
    device=setup_device(args.gpu,threads=args.threads)
    models=load_models(panel,device)
    versions=[v for v,_ in RELEASES]
    adapters,weights,_=load_candidates(BASE/'calibration',device,versions,groups['evaluation'])
    config=ImplementationConfig(**{k:v for k,v in registered['configuration'].items()
                                  if k in ('reader_backend','pack_item_gates','row_chunk')})
    wrapped={v:InferenceAdapter(adapters[v],config) for v in versions}
    # Initial coordinate encoder/embeddings are frozen in V3. Effective input
    # weights are frozen in the A branch, so this builds the same phi function.
    bank,bank_work=project_fixed_item_terms(adapters['v3'],models['v3'].item_emb.weight,
        chunk_size=registered['configuration']['item_chunk'])
    feature_fn=partial(fixed_features_from_bank,bank=bank)
    reference_prefixes,optimized_prefixes,prefix_bindings={},{},{}
    previous=None
    for version in versions:
        path=BASE/'publication'/f'{version}.pt'
        old,metadata=load_prefix(path,adapters[version],device=device)
        if metadata['bindings']['calibration_sha256'] != weights[version]['sha256'] \
                or metadata['bindings']['current_model_sha256'] != panel['models'][version]['checkpoint_sha256']:
            raise ValueError('Publication/model/calibration binding mismatch')
        stable=StableReleasePrefix.wrap(old)
        current,work=compile_item_tables(adapters[version],models[version],previous,
            chunk_size=registered['configuration']['item_chunk'])
        reference_prefixes[version]=stable
        optimized_prefixes[version]=configure_prefix(current,config,feature_fn=feature_fn)
        previous=current
        prefix_bindings[version]=dict(old_artifact_sha256=sha256(path),
            optimized_arithmetic=work['arithmetic'], original_arithmetic=metadata['actual_compilation_components_flops'])
        assert work['arithmetic']==metadata['actual_compilation_components_flops']
    sources=[Path(__file__), *sorted(Path(__file__).parent.glob('*.py')),
        ROOT/'scripts/query_relay_2026_10/adapter.py',ROOT/'scripts/query_relay_2026_10/session.py',
        ROOT/'scripts/query_relay_2026_10/precision.py',ROOT/'scripts/query_relay_2026_10/workload.py']
    settings=dict(started_utc=utc(), registration_path=str(args.registration),
        registration_sha256=sha256(args.registration), selected_uids=chosen,
        calibration_weights=weights, publications=prefix_bindings, bank_work=bank_work,
        implementation={v:wrapped[v].describe() for v in versions}, gpu=args.gpu,
        torch=str(torch.__version__), device=torch.cuda.get_device_name(device),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in dict.fromkeys(sources)},
        reference='unchanged original reader,original fixed bank,retained full P/Q/D,actual block inverse',
        reader_only='new reader/operation gates; original fixed bank and retained P/Q/D',
        all_optimized='new reader/gates; new all-catalogue phi/D compilation; chunk128 in-place conversion',
        no_training=True, no_quality_selection=True, no_new_quality_claim=True)
    write_json(args.output/'configuration.json',settings)
    histories=load_history(PANEL,requested_uids=chosen)
    rows,units=[],[]
    maxima=defaultdict(float)
    for protocol in registered['protocols']:
        requests=({u:candidate_rows(u,histories.rows[u],stride=16,candidates=16,
            known_vocab_size=panel['known_vocab_size']) for u in chosen}
            if protocol.startswith('behavior') else feedback)
        for number,uid in enumerate(chosen):
            make=lambda:RelaySession(NativeTrajectory(models['v2'],2,histories.rows[uid],RELEASES[0][1]))
            sessions={name:make() for name in registered['comparisons']}
            stages=[]
            for i,(version,at) in enumerate(RELEASES):
                for name,session in sessions.items():
                    adapter=adapters[version] if name=='reference' else wrapped[version]
                    prefix=optimized_prefixes[version] if name=='all_optimized' else reference_prefixes[version]
                    session.publish(models[version],int(version[1:]),at,adapter,prefix=prefix)
                at_time=defaultdict(list)
                for request in requests[uid]:
                    if request['version']==version:
                        at_time[int(request['query_timestamp'])].append(request)
                for timestamp,pending in sorted(at_time.items()):
                    pending=sorted(pending,key=lambda row:row['request_id'])
                    batch_size=registered['configuration']['query_batch']
                    for offset in range(0,len(pending),batch_size):
                        batch=pending[offset:offset+batch_size]
                        candidates=torch.tensor([[r['item_idx'] for r in batch]],device=device)
                        scores={name:session.score(timestamp,candidates).flatten()
                                for name,session in sessions.items()}
                        for name in ('reader_only','all_optimized'):
                            error=difference(scores[name],scores['reference'])
                            maxima[name+'_logit']=max(maxima[name+'_logit'],error['max_abs'])
                        for j,request in enumerate(batch):
                            rows.append(dict(protocol=protocol,request_id=request['request_id'],uid=uid,
                                version=version,label=int(request['label']),item_idx=int(request['item_idx']),
                                query_timestamp=timestamp,**{name:float(value[j]) for name,value in scores.items()}))
                until=RELEASES[i+1][1] if i+1<len(RELEASES) else END
                for session in sessions.values():
                    session.advance(until)
                ref=sessions['reference']
                comparisons={}
                for name in ('reader_only','all_optimized'):
                    session=sessions[name]
                    if ref.costs != session.costs or ref.trajectory.ledger != session.trajectory.ledger \
                            or ref.coded_read_ledger != session.coded_read_ledger:
                        raise RuntimeError('Implementation changed literal service operation counts')
                    value=state_check(ref,session)
                    comparisons[name]=value
                    for field in ('native_k','native_v','compact'):
                        if value[field] is not None:
                            key=name+'_'+field
                            maxima[key]=max(maxima[key],value[field]['max_abs'])
                stages.append(dict(version=version,state=comparisons,
                    counters=dict(ref.costs),events=ref.events))
            units.append(dict(protocol=protocol,uid=uid,stages=stages,
                scored_rows=sum(r['uid']==uid and r['protocol']==protocol for r in rows)))
            write_json(args.output/'progress.json',dict(status='running',completed_units=len(units),
                total_units=2*len(chosen),compared_candidate_scores=len(rows),maxima=dict(maxima)))
            print(json.dumps(dict(status='user_complete',protocol=protocol,uid=uid,
                users=number+1,requests=len(rows),maxima=dict(maxima))),flush=True)
    pq.write_table(pa.Table.from_pylist(rows),args.output/'scores.parquet',compression='zstd')
    report=dict(status='complete',settings=settings,completed_utc=utc(),units=units,
        compared_candidate_scores=len(rows),maxima=dict(maxima),
        code_and_service_count_checks=True,all_registered_users_retained=True,
        numerical_contract=registered['numerical_contract'],
        adapter_counters={v:dict(wrapped[v].counters) for v in versions},
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        elapsed_seconds=time.perf_counter()-start,
        scope='Real frozen32user,3release numerical engineering check; not independent80/20 quality evidence')
    write_json(args.output/'summary.json',report)
    print(json.dumps(dict(status='complete',compared_candidate_scores=len(rows),maxima=dict(maxima))),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--register',action='store_true')
    p.add_argument('--registration',type=Path,default=RESULTS/'registration.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--threads',type=int,default=2)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.register:
        register(args.registration)
    else:
        if args.output is None:
            raise SystemExit('--output is required for an integration run')
        try:
            run(args)
        except Exception as error:
            if args.output.exists():
                write_json(args.output/'failure.json',dict(at_utc=utc(),type=type(error).__name__,
                    message=str(error),traceback=traceback.format_exc()))
            raise
