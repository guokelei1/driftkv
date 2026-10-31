#!/usr/bin/env python3
"""Eight-user numerical comparison; no training or new quality claim.

Both sessions replay identical real requests/events with the same frozen model,
decoder, gate, update parameters and full publications. Only anchor inversion
differs. No native-control predictions, AUC or recovery are computed here.
"""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]

import torch

from hstu_kvcache.evaluation.release_context import END, RELEASES, load_models, setup_device
from hstu_kvcache.evaluation.panels import load_history, load_panel
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from experiments.archive.query_relay.evaluate import load_candidates
from hstu_kvcache.methods.cachetide.precision import StableReleasePrefix
from hstu_kvcache.methods.cachetide.publication import load_prefix
from hstu_kvcache.methods.cachetide.session import RelaySession
from hstu_kvcache.common.artifacts import sha256, write_json

PANEL=ROOT/'results/design_system_2026_10/phase2/representative4096/panel'
CANDIDATE=ROOT/'results/query_relay_2026_10/probes/A_origin6_rms_roll4'


def errors(a,b):
    if a.shape!=b.shape:
        raise RuntimeError('Numerical branches no longer have matching state geometry')
    if not a.numel():
        return dict(max_abs=0.,relative_l2=0.)
    delta=a.double()-b.double()
    return dict(max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm()/b.double().norm().clamp_min(1e-30)))


def state_errors(original,stable):
    if original.state_format!=stable.state_format or original.coded_rows!=stable.coded_rows:
        raise RuntimeError('Changing anchor numerics changed service scheduling')
    return dict(state_format=original.state_format,coded_rows=original.coded_rows,
        native_k=errors(stable.cache.k,original.cache.k),
        native_v=errors(stable.cache.v,original.cache.v),
        compact=errors(stable.state.codes,original.state.codes))


@torch.no_grad()
def run(args):
    started=time.perf_counter()
    panel,groups,requests=load_panel(args.panel)
    chosen=list(groups['evaluation'])[:args.users]
    if args.users!=8 or len(chosen)!=8:
        raise ValueError('This numerical probe is fixed to the first eight development UIDs')
    if args.panel.resolve()!=PANEL.resolve():
        raise ValueError('Use the existing development panel, keeping confirmation users untouched')
    args.output.mkdir(parents=True,exist_ok=False)
    device=setup_device(args.gpu,args.threads)
    models=load_models(panel,device)
    versions=[v for v,_ in RELEASES]
    adapters,weights,_=load_candidates(args.calibrations,device,versions,
                                       groups['evaluation'],None,None)
    prefixes,stable_prefixes,records={},{},{}
    for version in versions:
        path=args.published_prefixes/f'{version}.pt'
        prefix,metadata=load_prefix(path,adapters[version],device=device)
        if metadata['target_version']!=int(version[1:]) or \
                metadata['bindings']['calibration_sha256']!=weights[version]['sha256'] or \
                metadata['bindings']['current_model_sha256']!=panel['models'][version]['checkpoint_sha256']:
            raise ValueError('The complete publication differs from the frozen model/candidate')
        prefixes[version]=prefix
        stable_prefixes[version]=StableReleasePrefix.wrap(prefix)
        records[version]=dict(path=str(path),sha256=sha256(path),
                              inverse_cost=stable_prefixes[version].inverse_cost())
    settings=dict(role='numerical_extension_probe',users=chosen,panel=str(args.panel),
        panel_sha256=sha256(args.panel/'binding.json'),calibrations=weights,publications=records,
        alteration='Actual stored P block inverse computed in FP64 then retained in FP32; all other paths unchanged',
        no_training=True,no_quality_recovery_claim=True,request_labels_used=False,
        extra_global_inverse_bytes=sum(r['inverse_cost']['global_inverse_bytes'] for r in records.values()),
        extra_publication_flops_estimate=sum(r['inverse_cost']['preparation_flops_estimate'] for r in records.values()),
        extra_row_persistent_bytes=0,extra_row_flops=0,
        source_sha256={name:sha256(Path(__file__).parent/name) for name in
                       ('precision_compare.py','precision.py','adapter.py','session.py','publish.py')})
    write_json(args.output/'configuration.json',settings)
    histories=load_history(args.panel,requested_uids=chosen)
    samples,units=[],[]
    for uid in chosen:
        make=lambda: RelaySession(NativeTrajectory(models['v2'],2,histories.rows[uid],RELEASES[0][1]))
        original,stable=make(),make()
        stages=[]
        for index,(version,at) in enumerate(RELEASES):
            original.publish(models[version],int(version[1:]),at,adapters[version],prefix=prefixes[version])
            stable.publish(models[version],int(version[1:]),at,adapters[version],prefix=stable_prefixes[version])
            by_time=defaultdict(list)
            for request in requests[uid]:
                if request['version']==version:
                    by_time[int(request['query_timestamp'])].append(request)
            for timestamp,pending in sorted(by_time.items()):
                for offset in range(0,len(pending),16):
                    batch=pending[offset:offset+16]
                    candidates=torch.tensor([[r['item_idx'] for r in batch]],device=device)
                    a=original.score(timestamp,candidates)
                    b=stable.score(timestamp,candidates)
                    samples.append(dict(uid=uid,version=version,timestamp=timestamp,
                        requests=[r['request_id'] for r in batch],logit=errors(b,a),
                        state=state_errors(original,stable)))
            until=RELEASES[index+1][1] if index+1<len(RELEASES) else END
            original.advance(until)
            stable.advance(until)
            if original.costs!=stable.costs or original.trajectory.ledger!=stable.trajectory.ledger:
                raise RuntimeError('Anchor numerics changed service operation counts')
            stages.append(dict(version=version,final_state=state_errors(original,stable),
                               service_counters=dict(stable.costs),memory=dict(stable.memory)))
        units.append(dict(uid=uid,stages=stages))
        print(json.dumps(dict(status='precision_user_complete',uid=uid,users=len(units))),flush=True)
    report=dict(settings=settings,units=units,query_batches=samples,
        compared_requests=sum(len(s['requests']) for s in samples),
        maximum_logit_delta=max((s['logit']['max_abs'] for s in samples),default=0.),
        maximum_native_k_delta=max((s['state']['native_k']['max_abs'] for s in samples),default=0.),
        maximum_native_v_delta=max((s['state']['native_v']['max_abs'] for s in samples),default=0.),
        maximum_current_code_delta=max((s['state']['compact']['max_abs'] for s in samples),default=0.),
        seconds=time.perf_counter()-started)
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(status='precision_compare_complete',requests=report['compared_requests'],
                         maximum_logit_delta=report['maximum_logit_delta'])),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,default=PANEL)
    p.add_argument('--calibrations',type=Path,default=CANDIDATE/'calibration')
    p.add_argument('--published-prefixes',type=Path,default=CANDIDATE/'publication')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--users',type=int,default=8)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--threads',type=int,default=4)
    return p


if __name__=='__main__':
    run(parser().parse_args())
