#!/usr/bin/env python3
"""Matched continuous development replay of one frozen structured-code candidate."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from code_structure_2026_10.structure import StructuredKVAdapter
from code_structure_2026_10.session import StructuredSession
from design_two.common import END, RELEASES, load_models, setup_device
from design_two.prepare import load_history, load_panel
from hstu_kvcache.design_two.replay import NativeTrajectory
from partial_refresh_2026_10.analyze import summarize
from unified_reuse_2026_09.common import sha256, write_json

PANEL = ROOT / 'results/design_system_2026_10/phase2/representative4096/panel'
REFERENCE = ROOT / 'results/design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_support_lazy/development4096/scores.parquet'


@torch.no_grad()
def run(args):
    began = time.perf_counter()
    panel, groups, requests = load_panel(PANEL)
    population = list(groups['evaluation'])[:args.users] if args.users else list(groups['evaluation'])
    selected = population[args.start_user:][args.shard_index::args.shards]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'configuration.json', dict(selected_uids=selected, population=population,
        calibration_root=str(args.calibrations), panel_sha256=sha256(PANEL / 'binding.json'),
        reference_sha256=sha256(REFERENCE), gpu=args.gpu, shards=args.shards, shard_index=args.shard_index,
        start_user=args.start_user,
        cross_connection_ablation=args.ablate_cross_connections,
        native_writes='unchanged', cost_scope='fixed30k separately; no time claim',
        role='development_exploration', storage_floats_per_layer_row=48))
    device = setup_device(args.gpu, args.threads)
    models = load_models(panel, device)
    adapters = {}
    for version, _ in RELEASES:
        path = args.calibrations / version / 'calibration.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        meta = json.loads(path.with_suffix('.json').read_text())
        if (set(meta['fit_uids']) | set(meta['validation_uids'])) & set(population):
            raise RuntimeError('quality overlaps fitting roles')
        adapter = StructuredKVAdapter(**payload['config']).to(device).eval().requires_grad_(False)
        adapter.load_state_dict(payload['state_dict'])
        if args.ablate_cross_connections:
            from code_structure_2026_10.ablate import remove_connections
            adapter = remove_connections(adapter, encoder=True, decoder=True)
        adapters[version] = adapter
    references = {row['request_id']: row for row in pq.read_table(REFERENCE,
        filters=[('uid', 'in', selected)]).to_pylist()}
    histories = load_history(PANEL, requested_uids=selected)
    rows = []
    counters = Counter()
    units = []
    max_code_error = 0.
    for index, uid in enumerate(selected):
        session = StructuredSession(NativeTrajectory(models['v2'], 2, histories.rows[uid], RELEASES[0][1]))
        for release_index, (version, at) in enumerate(RELEASES):
            session.publish(models[version], int(version[1:]), at, adapters[version])
            by_time = defaultdict(list)
            for request in requests[uid]:
                if request['version'] == version:
                    by_time[int(request['query_timestamp'])].append(request)
            for timestamp, pending in sorted(by_time.items()):
                session.prepare_read(timestamp)
                if index < 2:
                    session.verify()
                for offset in range(0, len(pending), 16):
                    batch = pending[offset:offset + 16]
                    ids = torch.tensor([[r['item_idx'] for r in batch]], device=device)
                    actual = session.score(timestamp, ids)[0]
                    if not torch.isfinite(actual).all():
                        raise RuntimeError('nonfinite structured scores')
                    for request, value in zip(batch, actual.cpu().tolist(), strict=True):
                        old = references[request['request_id']]
                        if any(old[k] != request[k] for k in ('uid', 'version', 'label', 'item_idx', 'query_timestamp')):
                            raise RuntimeError('reference request identity differs')
                        rows.append(dict(**{k: old[k] for k in ('request_id', 'uid', 'version', 'label',
                            'reuse_logit', 'first_request_rebuild_logit')}, full_logit=old['lightweight_logit'],
                            partial_logit=value, history_length=session.cache.seq_len))
            session.advance(RELEASES[release_index + 1][1] if release_index + 1 < 3 else END)
        max_code_error = max(max_code_error, session.max_code_error)
        counters.update(session.costs)
        units.append(dict(uid=uid, counters=dict(session.costs)))
        if (index + 1) % 16 == 0:
            print(json.dumps(dict(status='quality', users=index + 1, requests=len(rows))), flush=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.output / 'scores.parquet')
    quality = summarize(table, selected, repetitions=args.bootstrap, releases=['v3', 'v4', 'v5'])
    write_json(args.output / 'summary.json', dict(status='complete', quality=quality,
        selected_uids=selected, layout=adapters['v3'].layout, elapsed_seconds=time.perf_counter() - began,
        cross_connection_ablation=args.ablate_cross_connections,
        actual_service_counters=dict(counters), units=units,
        checks=dict(incremental_code_max_error=max_code_error)))
    print(json.dumps(dict(status='complete', users=len(selected), overall=quality['overall']['paired'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibrations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--users', type=int, default=512, help='0 means full fixed4096')
    p.add_argument('--start-user', type=int, default=0, help='continue the frozen population without repeating earlier users')
    p.add_argument('--ablate-cross-connections', action='store_true', help='diagnostic: remove centered cross-head encoder and other-owner decoder connections without refitting')
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--bootstrap', type=int, default=1000)
    run(p.parse_args())
