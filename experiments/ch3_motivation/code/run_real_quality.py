#!/usr/bin/env python3
"""Paired native replay of simple maps, scored on all eligible real feedback."""
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
import os
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import ROOT, load_edge, load_models, write_json
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.probe_cost import cached_preparation_flops
from experiments.ch3_motivation.code.real_quality_paths import replay_user
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

SCHEMA = pa.schema([
    ('uid', pa.uint64()), ('query_timestamp', pa.int64()), ('item_idx', pa.int64()),
    ('label', pa.int8()), ('method', pa.string()), ('score', pa.float64()),
    ('kv_mse', pa.float64()), ('k_mse', pa.float64()), ('v_mse', pa.float64()),
    ('initial_kv_mse', pa.float64()), ('appended_events', pa.int64()),
    ('history_rows', pa.int64()), ('query_gap', pa.int64()),
])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def run(args):
    if args.gpu != 1:
        raise ValueError('this study uses physical GPU1 only')
    if not 1 <= args.block_users <= 128:
        raise ValueError('bounded resident block required')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    population = json.loads(args.population.read_text())
    fixed_users = population['users']
    if len(fixed_users) != 4000 or len(set(fixed_users)) != 4000:
        raise ValueError('requires frozen main+sensA4000')
    input_root = ROOT/'experiments/ch3_motivation/inputs'
    binding, histories, requests = load_edge(input_root/f'edge_{args.edge}')
    for key in ('history','requests'):
        if sha256(input_root/f'edge_{args.edge}'/binding[key]['path']) != binding[key]['sha256']:
            raise ValueError('frozen history or real requests changed')
    if binding['population_sha256'] != sha256(input_root/'population.json'):
        raise ValueError('original role binding changed')
    users = ([u for u in fixed_users if requests.get(u)][:args.users]
             if args.diagnostic else fixed_users)
    eligible = [u for u in users if requests.get(u)]
    if args.diagnostic:
        if len(users) != args.users:
            raise ValueError('requested numerical users unavailable')
    elif len(eligible) < 512 or args.users != 4000:
        raise ValueError('formal real-feedback comparison needs full4000 and >=512 feedback UIDs')
    probe, metadata = load_variant(args.checkpoint, device)
    parent, current, models = load_models(args.edge, device)
    identity = metadata['identity']
    if (probe.stage != 'f' or identity['edge'] != args.edge or identity['models'] != models
            or set(fixed_users) & set(identity['fit_uids']+identity['validation_users'])):
        raise ValueError('model, F or fitting/evaluation role mismatch')
    # Fit and evaluation packs intentionally differ; bind both rather than
    # relabeling one pack as the other.
    fit_root = Path(identity['input_root']) if 'input_root' in identity else ROOT/'experiments/ch3_motivation/inputs_feedback_pool/fit256'
    fit_binding = fit_root/f'edge_{args.edge}/inputs.json'
    if sha256(fit_binding) != identity['input_binding_sha256']:
        raise ValueError('the F calibration source changed')
    probe.eval().requires_grad_(False)
    extra_maps, map_metadata = None, None
    if args.maps:
        from experiments.ch3_motivation.code.simple_kv_maps import load_maps
        extra_maps = load_maps(args.maps, device)
        map_metadata = json.loads((args.maps.parent/'summary.json').read_text())
        map_identity = map_metadata['identity']
        if (map_metadata['checkpoint_sha256'] != sha256(args.maps)
                or map_identity['edge'] != args.edge or map_identity['models'] != models
                or map_identity['fit_uids'] != identity['fit_uids']
                or map_identity['validation_users'] != identity['validation_users']
                or map_identity['input_binding_sha256'] != identity['input_binding_sha256']
                or map_identity['reference_f_sha256'] != sha256(args.checkpoint)):
            raise ValueError('simple maps differ from the paired F calibration source/roles/models')
    cfg = dict(binding['settings'], history_length=args.history_cap)
    source_paths = [Path(__file__), *[Path(__file__).with_name(n) for n in
        ('real_quality_paths.py','common.py','probe_variants.py','probes.py','probe_cost.py')],
        ROOT/'src/hstu_kvcache/evaluation/trajectory.py', ROOT/'src/hstu_kvcache/models/state_transition.py',
        ROOT/'src/hstu_kvcache/models/hstu.py', ROOT/'src/hstu_kvcache/models/attention.py',
        ROOT/'src/hstu_kvcache/adaptation/reader.py']
    if args.maps:
        source_paths.append(Path(__file__).with_name('simple_kv_maps.py'))
    sources = {str(p.resolve()):sha256(p) for p in source_paths}
    configuration = dict(format='real_quality_angles_v1', edge=args.edge, release_day=cfg['release_day'],
        settings=cfg, users=users, fixed_resident_users=fixed_users, feedback_users=eligible,
        population=str(args.population.resolve()), population_sha256=sha256(args.population),
        input_binding_sha256=sha256(input_root/f'edge_{args.edge}/inputs.json'),
        fitting_input_binding_sha256=sha256(fit_binding), original_input_binding=binding,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        simple_maps=str(args.maps.resolve()) if args.maps else None,
        simple_maps_sha256=sha256(args.maps) if args.maps else None,
        simple_map_metadata=map_metadata, models=models, sources=sources, physical_gpu=1,
        hardware=torch.cuda.get_device_name(0), diagnostic=args.diagnostic, check_batch=args.check_batch,
        block_users=args.block_users, history_cap=args.history_cap,
        structural_controls=args.structural_controls,
        fitting_users=len(identity['fit_uids']), validation_users=len(identity['validation_users']),
        fit_to_resident_ratio=len(identity['fit_uids'])/len(fixed_users),
        fit_to_feedback_user_ratio=len(identity['fit_uids'])/len(eligible),
        feedback_scope='Frozen known-item observed feedback; conflicting duplicate labels removed; no synthetic negatives',
        teacher='cutover once-Rebuild then native causal appends, not fresh per request',
        oracle_scope='request-local F-to-Rebuild KV blends only for scoring; teacher needed; never used for writes',
        quality_shortcut='Skip no-feedback residents and stop after last scored request; complete preparation cost references separately retained')
    args.output.mkdir(parents=True,exist_ok=True)
    path = args.output/'configuration.json'
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError('configuration/input/weights/sources changed; use a different attempt')
    write_json(path,configuration)
    frozen = digest(configuration)
    artifacts, all_stats = [], []
    completed_feedback = event_rows = 0
    started = time.perf_counter()
    for offset in range(0,len(users),args.block_users):
        selected = users[offset:offset+args.block_users]
        score_path = args.output/f'blocks/part_{offset:06d}.parquet'
        record_path = score_path.with_suffix('.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if (record['configuration_sha256'] != frozen or record['uids'] != selected
                    or record['score_sha256'] != sha256(score_path)):
                raise ValueError('committed real-feedback block changed')
        else:
            rows, stats, ledgers = [], [], []
            for uid in selected:
                if not requests.get(uid):
                    continue
                values, details, ledger = replay_user(parent,current,probe,histories.rows[uid],
                    requests[uid],uid,cfg,history_cap=args.history_cap,
                    structural_controls=args.structural_controls,check_batch=args.check_batch,
                    extra_maps=extra_maps)
                rows.extend(values);stats.append(details);ledgers.append(dict(uid=uid,operations=ledger))
            if any(sha256(Path(p)) != h for p,h in sources.items()):
                raise ValueError('numerical source changed before score commit')
            score_path.parent.mkdir(exist_ok=True)
            temp = score_path.with_suffix('.parquet.writing')
            pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),temp)
            temp.replace(score_path)
            record = dict(configuration_sha256=frozen,offset=offset,uids=selected,
                scored_users=len(stats),score_rows=len(rows),stats=stats,ledgers=ledgers,
                score_sha256=sha256(score_path))
            write_json(record_path,record)
        all_stats.extend(record['stats']);completed_feedback+=record['scored_users'];event_rows+=record['score_rows']
        artifacts.append(dict(path=str(score_path.relative_to(args.output)),sha256=sha256(score_path),
            manifest=str(record_path.relative_to(args.output)),manifest_sha256=sha256(record_path)))
        progress = dict(status='running',edge=args.edge,completed_residents=offset+len(selected),
            residents=len(users),completed_feedback_users=completed_feedback,feedback_users=len(eligible),
            seconds=time.perf_counter()-started)
        write_json(args.output/'progress.json',progress)
        print(json.dumps(progress),flush=True)
    # Resident and feedback-only preparation denominators remain distinct.
    cm = CostModel(current.cfg.hidden_size,len(current.blocks),current.cfg.num_heads,cfg['attention_backend'])
    rebuild_all = rebuild_active = f_all = f_active = 0
    blend_all = blend_active = 0
    map_all = {name:0 for name in extra_maps or {}}
    map_active = dict(map_all)
    for uid in users:
        n = min(args.history_cap,int(np.searchsorted(histories.rows[uid][0],cfg['release_day']*86400,side='left')))
        cost = cm if n == 1024 else CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,'torch')
        rebuild = cost.full_cache(n)
        preparation = cached_preparation_flops(probe,n)
        # F + alpha*(F-source): subtraction, multiplication and addition on
        # both K and V. Head-only controls use views without arithmetic.
        blend = 3*n*cm.num_layers*2*cm.hidden_size
        rebuild_all += rebuild;f_all += preparation
        blend_all += blend
        for name, transform in (extra_maps or {}).items():
            value = transform.materialization_flops(n)
            map_all[name] += value
            if requests.get(uid):map_active[name] += value
        if requests.get(uid):
            rebuild_active+=rebuild;f_active+=preparation;blend_active+=blend
    calibration = metadata['cumulative_calibration_flops_estimate']
    preparation_points = {}
    for name in all_stats[0]['method_names']:
        if name.startswith('kv_oracle'):
            continue
        if name == 'reuse':
            shared, active, residents = 0,0,0
        elif name == 'rebuild':
            shared, active, residents = 0,rebuild_active,rebuild_all
        elif name in map_all:
            shared = map_metadata['total_calibration_flops_estimate']
            active, residents = map_active[name],map_all[name]
        else:
            shared = calibration
            active, residents = f_active,f_all
            if name in {'f025','f05','f15','f2'}:
                active+=blend_active;residents+=blend_all
        preparation_points[name] = dict(shared_calibration_flops=int(shared),
            feedback_materialization_flops=int(active),all_resident_materialization_flops=int(residents),
            feedback_preparation_percent_once_rebuild=100*(shared+active)/rebuild_active,
            all_resident_preparation_percent_once_rebuild=100*(shared+residents)/rebuild_all)
    costs = dict(f_calibration_flops=calibration,all_resident_once_rebuild_flops=int(rebuild_all),
        feedback_resident_once_rebuild_flops=int(rebuild_active),all_resident_f_materialization_flops=int(f_all),
        feedback_resident_f_materialization_flops=int(f_active),
        f_feedback_preparation_percent_rebuild=100*(calibration+f_active)/rebuild_active,
        f_all_resident_preparation_percent_rebuild=100*(calibration+f_all)/rebuild_all,
        persistent_method_preparation=preparation_points,
        oracle_scope='Teacher trajectory plus request-local cache mixtures and extra native reads are research-only. Actual calls/shapes are retained in block ledgers; never compared as a cheap deployable method.',
        scope='Shared fitting + once-only materialization including alpha/shift/linear arithmetic. Static maps execute the same native append/read shapes so common serving arithmetic cancels in added cost. Simple maps each conservatively charged full combined fitter, including both100fit+16val prefills. These are preparation/added-compute references, not total research runtime or final Design cost.')
    summary = dict(status='complete',edge=args.edge,users=len(users),fixed_residents=4000,
        feedback_users=completed_feedback,score_rows=event_rows,artifacts=artifacts,physical_gpu=1,
        diagnostic=args.diagnostic,history_cap=args.history_cap,structural_controls=args.structural_controls,
        quality_use_permitted=not args.diagnostic and completed_feedback>=512,cost_reference=costs,
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        max_batch_abs_error=(max(s['max_batch_abs_error'] for s in all_stats)
                             if args.check_batch else None),
        statistics='all observed requests; evaluate each method on identical labels/UIDs/time; score arrays retained')
    write_json(args.output/'summary.json',summary)
    write_json(args.output/'progress.json',dict(progress,status='complete'))
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edge',type=int,choices=(2,3),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--population',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--maps',type=Path)
    p.add_argument('--history-cap',type=int,choices=(128,512,1024),default=1024)
    p.add_argument('--structural-controls',action='store_true')
    p.add_argument('--users',type=int,default=4000)
    p.add_argument('--block-users',type=int,default=64)
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--check-batch',action='store_true')
    args=p.parse_args()
    try:
        run(args);write_json(args.output/'exit.json',dict(exit_code=0))
    except BaseException as error:
        write_json(args.output/'exit.json',dict(exit_code=1,error=str(error)));raise
