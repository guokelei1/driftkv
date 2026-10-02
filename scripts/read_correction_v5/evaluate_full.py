#!/usr/bin/env python3
"""Score saved v5 Q/H candidates together on the frozen rolling request panel."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/home/gkl/work/evokv')
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from evaluate_yambda500m_foundation_raw import load_histories, load_model
from hstu_kvcache.evaluation.binary_metrics import binary_metrics
from read_correction_2026_09.cost import normalize_metrics
from read_correction_v3.evaluate import cost_record
from read_correction_v4.evaluate_full import HISTOGRAMS, signature, validate_rows, verify_saved_unit
from read_correction_v5.common import PANEL_ROOT, configuration, edge_name, sha256, sources, write_json
from read_correction_v5.scoring import load_policies, score_unit
from selective_recompute_2026_09.scheduling import ordered_uids


def calibration_inputs(path, panel_path, evaluation_uids):
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError('calibration list must be a nonempty JSON list')
    records, inputs = {}, {}
    for entry in entries:
        tag = entry['tag']
        if not isinstance(tag, str) or not tag or Path(tag).name != tag or tag in records:
            raise ValueError('candidate tags must be unique file-name components')
        if entry['method'] not in ('query_only', 'history_conditioned'):
            raise ValueError('unknown candidate method')
        if entry['method'] == 'history_conditioned' and entry['variant'] not in ('map_all', 'map_old_prefix'):
            raise ValueError('history candidate requires an explicit token-map variant')
        folder = Path(entry['calibration_dir']).resolve()
        entry['calibration_dir'] = str(folder)
        record_path, weights = folder / 'calibration.json', folder / 'calibration.pt'
        record = json.loads(record_path.read_text())
        if record['status'] != 'complete' or sha256(weights) != record['weights_sha256']:
            raise RuntimeError(f'unfinished or changed calibration: {tag}')
        if record['panel_binding_sha256'] != sha256(panel_path):
            raise RuntimeError(f'calibration and evaluation panel differ: {tag}')
        fit_uids = record['uids'] + record['validation_uids']
        if len(fit_uids) != len(set(fit_uids)) or set(fit_uids).intersection(evaluation_uids):
            raise RuntimeError(f'calibration/validation/evaluation users overlap: {tag}')
        if not isinstance(record['budget'], (int, float)):
            raise ValueError('calibration budget must be numeric')
        records[tag] = record
        inputs[tag] = {**entry, 'budget': record['budget'],
            'calibration': {'path': str(record_path), 'sha256': sha256(record_path), 'weights_sha256': sha256(weights)},
            'calibration_sources': record['execution_sources'], 'calibration_settings': record['settings']}
    return entries, records, inputs


def projected_cost(policy, scale, edge):
    """A cost-only projection; probe quality remains on its actual subset."""
    path = ROOT / 'results/read_correction_2026_09/runtime/development/v1' / scale / edge_name(edge) / 'summary.json'
    old = json.loads(path.read_text())
    correction = sum(int(count) * sum(policy['cost_fn'](module.get_config(), int(n))
        for module in policy['modules']) for n, count in old['histograms']['full_history_hist'].items())
    return {'scope': 'cost only on the unchanged complete 3000-user panel; no projected quality',
        'users': old['users'], 'requests': old['requests'],
        'source': {'path': str(path), 'sha256': sha256(path)},
        **cost_record(scale, old['histograms'], correction, policy['calibration']['cost']['calibration_flops'])}


def run(args):
    started = time.perf_counter()
    output, cfg = args.output.resolve(), configuration()
    panel_path = PANEL_ROOT / args.scale / edge_name(args.edge) / 'binding.json'
    panel = json.loads(panel_path.read_text())
    requests_path = panel_path.parent / 'evaluation_requests.parquet'
    if sha256(requests_path) != panel['files']['evaluation_requests']['sha256']:
        raise RuntimeError('frozen request panel changed')
    by_user = defaultdict(list)
    for row in pq.read_table(requests_path).to_pylist():
        by_user[int(row['uid'])].append(row)
    if len(by_user) != 3000:
        raise RuntimeError('expected the fixed 3000-user evaluation population')
    entries, calibrations, calibration_bindings = calibration_inputs(args.calibration_list, panel_path, set(by_user))
    tags = tuple(entry['tag'] for entry in entries)
    if args.limit_users < 0 or args.limit_users > len(by_user) or args.unit_users < 1:
        raise ValueError('invalid user limit or continuation unit size')
    selected = sorted(by_user, key=lambda uid: hashlib.sha256(f'read-correction-v2-probe:17:{uid}'.encode()).digest())
    if args.limit_users:
        selected = selected[:args.limit_users]
    uids = ordered_uids(by_user, max_length=cfg['history_length'], uids=selected)
    probe = len(uids) != len(by_user)
    inputs = {'execution_sources': sources(), 'settings': json.loads(json.dumps(cfg)),
        'calibration_list': {'path': str(args.calibration_list.resolve()), 'sha256': sha256(args.calibration_list)},
        'policies': calibration_bindings, 'panel_sha256': sha256(panel_path),
        'requests_sha256': sha256(requests_path), 'uids': uids, 'unit_users': args.unit_users,
        'verification': 'every probe unit; first complete-population unit'}
    binding = signature(inputs)
    units = []
    for index, start in enumerate(range(0, len(uids), args.unit_users)):
        path = output / 'units' / f'unit_{index:05d}.json'
        record = json.loads(path.read_text()) if path.exists() else None
        if record is not None:
            verify_saved_unit(record, binding, uids[start:start + args.unit_users], tags)
            cfg['cohort_sizes'][args.scale] = min(cfg['cohort_sizes'][args.scale], record['cohort_size'])
            cfg['query_batches'][args.scale] = min(cfg['query_batches'][args.scale], record['query_batch'])
        units.append(record)
    summary_path = output / 'summary.json'
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous['input_signature'] != binding or any(unit is None for unit in units):
            raise RuntimeError('completed evaluation has different inputs or missing units')
        for record in previous['scores'].values():
            if sha256(Path(record['path'])) != record['sha256']:
                raise RuntimeError('completed scores changed')
        print(json.dumps({'status': 'already_complete', 'output': str(summary_path)}), flush=True)
        return previous
    write_json(output / 'inputs.json', dict(inputs, input_signature=binding))
    model_seconds, history_seconds, policies = 0., 0., None
    if any(unit is None for unit in units):
        os.environ['EVOKV_ATTENTION_BACKEND'] = cfg['attention_backend']
        torch.set_num_threads(cfg['torch_threads'])
        pa.set_cpu_count(cfg['history_threads'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        free, total = torch.cuda.mem_get_info(device)
        while free / total < cfg['initial_free_fraction']:
            print(json.dumps({'status': 'waiting_for_memory', 'gpu': args.gpu, 'free_fraction': free / total}), flush=True)
            time.sleep(30)
            free, total = torch.cuda.mem_get_info(device)
        torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'], device)
        torch.cuda.reset_peak_memory_stats(device)
        beginning = time.perf_counter()
        parent, parent_payload = load_model(ROOT / panel['sources']['parent']['path'], device)
        current, current_payload = load_model(ROOT / panel['sources']['current']['path'], device)
        if parent_payload['config'] != current_payload['config']:
            raise RuntimeError('parent and current model architectures differ')
        dataset_path = ROOT / panel['sources']['dataset']['path']
        dataset = json.loads(dataset_path.read_text())
        known = int(current_payload.get('known_vocab_size', dataset['foundation_items']))
        parent.requires_grad_(False); current.requires_grad_(False)
        policies = load_policies(entries, device)
        if any(len(policy['modules']) != len(current.blocks) for policy in policies):
            raise RuntimeError('correction layer count differs from model')
        del parent_payload, current_payload
        model_seconds = time.perf_counter() - beginning
        remaining = [uid for index, start in enumerate(range(0, len(uids), args.unit_users)) if units[index] is None
                     for uid in uids[start:start + args.unit_users]]
        beginning = time.perf_counter()
        history = load_histories(remaining, dataset_path=dataset_path, known_vocab_size=known,
            oov_buckets=current.cfg.num_items - known, start_timestamp=int(panel['cutover']),
            end_timestamp=int(panel['days'][1]) * 86400, max_history=cfg['history_length'], threads=cfg['history_threads'])
        history_seconds = time.perf_counter() - beginning
        for index, start in enumerate(range(0, len(uids), args.unit_users)):
            if units[index] is not None:
                continue
            selected = uids[start:start + args.unit_users]
            reference = sorted([row for uid in selected for row in by_user[uid]], key=lambda row: row['request_id'])
            beginning, reductions = time.perf_counter(), []
            while True:
                try:
                    scores, histograms, controls = score_unit(current, parent, history, by_user, selected,
                        int(panel['cutover']), policies, cfg, args.scale, device, verify=probe or index == 0)
                    break
                except torch.cuda.OutOfMemoryError:
                    cohort, queries = cfg['cohort_sizes'][args.scale], cfg['query_batches'][args.scale]
                    if cohort == queries == 1:
                        raise
                    reductions.append({'unit': index, 'cohort_size': cohort, 'query_batch': queries})
                    cfg['cohort_sizes'][args.scale], cfg['query_batches'][args.scale] = max(1, cohort // 2), max(1, queries // 2)
                    gc.collect(); torch.cuda.empty_cache()
            paths = {}
            for tag in tags:
                rows = validate_rows(scores[tag], reference)
                path = output / 'units' / f'unit_{index:05d}.{tag}.parquet'
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.parquet.partial')
                pq.write_table(pa.Table.from_pylist(rows), temporary, compression='zstd')
                temporary.replace(path)
                paths[tag] = {'path': str(path), 'sha256': sha256(path), 'rows': len(rows)}
            record = {'input_signature': binding, 'uids': selected, 'requests': len(reference), 'outputs': paths,
                'histograms': histograms, 'controls': controls, 'verified': probe or index == 0,
                'seconds': time.perf_counter() - beginning, 'batch_reductions': reductions,
                'cohort_size': cfg['cohort_sizes'][args.scale], 'query_batch': cfg['query_batches'][args.scale],
                'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
                'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30}
            write_json(output / 'units' / f'unit_{index:05d}.json', record)
            units[index] = record
            print(json.dumps({'status': 'scoring', 'scale': args.scale, 'edge': edge_name(args.edge),
                'users': start + len(selected), 'total_users': len(uids), 'unit_seconds': record['seconds']}), flush=True)
    histograms = {key: Counter() for key in HISTOGRAMS}
    controls = {key: 0 for key in units[0]['controls']}
    for unit in units:
        for key in HISTOGRAMS:
            histograms[key].update(unit['histograms'][key])
        for key, value in unit['controls'].items():
            controls[key] = controls[key] + value if key == 'requests' else max(controls[key], value)
    histograms = {key: dict(value) for key, value in histograms.items()}
    reference = sorted([row for uid in uids for row in by_user[uid]], key=lambda row: row['request_id'])
    labels = np.asarray([row['label'] for row in reference])
    full = binary_metrics(labels, np.asarray([row['full_logit'] for row in reference]))
    reuse = binary_metrics(labels, np.asarray([row['reuse_logit'] for row in reference]))
    if probe and policies is None:
        policies = load_policies(entries, torch.device('cpu'))
    policy_by_tag = {policy['tag']: policy for policy in policies} if policies is not None else {}
    points, combined = [], {}
    for entry in entries:
        tag = entry['tag']
        table = pa.concat_tables([pq.read_table(unit['outputs'][tag]['path']) for unit in units]).sort_by([('request_id', 'ascending')])
        rows = validate_rows(table.to_pylist(), reference)
        measured = binary_metrics(labels, table['hstu_logit'].to_numpy())
        ledger = cost_record(args.scale, histograms, sum(row['correction_flops'] for row in rows),
                             calibrations[tag]['cost']['calibration_flops'])
        normalized = normalize_metrics(full_auc=full['ROC_AUC'], reuse_auc=reuse['ROC_AUC'],
            baseline_auc=measured['ROC_AUC'], extra_flops=ledger['extra_flops'], full_minus_reuse_flops=ledger['full_minus_reuse_flops'])
        path = output / f'{tag}.parquet'
        temporary = path.with_suffix('.parquet.partial')
        pq.write_table(table, temporary, compression='zstd'); temporary.replace(path)
        combined[tag] = {'path': str(path), 'sha256': sha256(path), 'rows': len(rows)}
        point = {'tag': tag, 'method': entry['method'], 'baseline': entry['method'], 'variant': entry['variant'],
            'scale': args.scale, 'edge': edge_name(args.edge), 'kind': 'measurement', 'budget': calibrations[tag]['budget'],
            'calibration_users': len(calibrations[tag]['uids']), 'users': len(uids), 'requests': len(rows),
            'full_auc': full['ROC_AUC'], 'reuse_auc': reuse['ROC_AUC'], 'baseline_auc': measured['ROC_AUC'],
            'metrics': measured, 'partition': 'evaluation', 'evaluation_role': 'development_exploration',
            **ledger, **normalized}
        if probe:
            point['projected_full_panel_cost'] = projected_cost(policy_by_tag[tag], args.scale, args.edge)
        points.append(point)
    summary = {'status': 'complete', 'revision': 'v5', 'evaluation_role': 'development_exploration',
        'probe_only': probe, 'scale': args.scale, 'edge': edge_name(args.edge), 'users': len(uids),
        'requests': len(reference), 'partition': 'evaluation', 'tags': list(tags), 'points': points,
        'current_full': full, 'current_reuse': reuse, 'inputs': inputs, 'input_signature': binding,
        'scores': combined, 'histograms': histograms, 'controls': controls,
        'cost_scope': 'complete calibration charged once per policy to actual evaluated requests; probe projection is separate',
        'model_load_seconds': model_seconds, 'history_load_seconds': history_seconds,
        'scoring_seconds': sum(unit['seconds'] for unit in units), 'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_gib': max(unit['peak_allocated_gib'] for unit in units),
        'peak_reserved_gib': max(unit['peak_reserved_gib'] for unit in units),
        'batch_reductions': [item for unit in units for item in unit['batch_reductions']]}
    write_json(summary_path, summary)
    print(json.dumps({'status': 'complete', 'output': str(summary_path), 'users': len(uids),
        'points': [{key: point[key] for key in ('tag', 'baseline_auc', 'recovery_percent', 'relative_flops_percent')}
                   for point in points]}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('medium', 'large', 'max'), required=True)
    parser.add_argument('--edge', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--calibration-list', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-users', type=int, default=0)
    parser.add_argument('--unit-users', type=int, default=256)
    run(parser.parse_args())

