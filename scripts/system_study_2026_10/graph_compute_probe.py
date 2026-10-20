"""Separate framework dispatch from GPU work on retained real model inputs.

Both native and corrected computation receive the same graph optimization.
This is an operation diagnostic, not a session or population speedup result.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import numpy as np
import torch

from design_two.common import load_model, setup_device, sha256, write_json
from design_two.prepare import load_history
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.adaptation.reader import score
from query_relay_2026_10.adapter import UnifiedRelayAdapter, operation_gate_query
from system_study_2026_10.fused_query_projection import QueryProjectionAdapter
from system_study_2026_10.batched_session import BatchedSessions
from system_study_2026_10.budgeted_rebuild import native_history_read


def tensors(value):
    return (value.k, value.v) if isinstance(value, HSTUKVCache) else (value,)


@torch.no_grad()
def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    device = setup_device(args.gpu, threads=2)
    branch = ROOT/config['branch']
    panel = ROOT/config['panel']
    binding = json.loads((panel/'binding.json').read_text())
    model_record = binding['models']['v4']
    model_path = ROOT/model_record['checkpoint']
    if sha256(model_path) != model_record['checkpoint_sha256']:
        raise ValueError('frozen model changed')
    model, _ = load_model(model_path, device)
    model.eval().requires_grad_(False)
    for block in model.blocks:
        block.attn.backend = 'triton'
    payload = torch.load(branch/'calibration/v4/calibration.pt', map_location='cpu', weights_only=False)
    adapter = UnifiedRelayAdapter(**payload['config']).to(device).eval().requires_grad_(False)
    adapter.load_state_dict(payload['state_dict'])
    adapter = QueryProjectionAdapter(adapter)
    source_path = branch/'source_v5/source.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False, mmap=True)
    fit = json.loads((branch/'calibration/v5/calibration.json').read_text())['fit_uids']
    users = sorted(u for u, r in source['rows'].items() if u in fit and r['parent'].seq_len == 1024)[:8]
    # Source rows are calibration users, deliberately outside the quality panel.
    histories = load_history(ROOT/config['source_panel'], requested_uids=users).rows
    rows = [source['rows'][u] for u in users]
    cache0 = HSTUKVCache(torch.cat([r['parent'].k for r in rows], 1).to(device),
                        torch.cat([r['parent'].v for r in rows], 1).to(device), 1024)
    codes0 = torch.cat([r['previous_codes'] for r in rows], 1).to(device)
    query_ids = torch.stack([r['candidates'] for r in rows]).to(device)
    query_delta = torch.tensor([r['query_delta'] for r in rows], device=device, dtype=torch.float32)
    retained = []
    for uid, row in zip(users, rows):
        events = histories[uid]
        stop = np.searchsorted(events[0], 287*86400, side='left')
        values = tuple(x[stop-1024:stop] for x in events)
        if not np.array_equal(values[1], row['item_ids'][0].numpy()):
            raise ValueError('source item history mismatch')
        retained.append(values)
    history_ids = torch.as_tensor(np.stack([r[1] for r in retained]), device=device, dtype=torch.long)
    history_behavior = torch.as_tensor(np.stack([r[2] for r in retained]), device=device, dtype=torch.long)
    history_delta = torch.as_tensor(np.stack([np.concatenate(([0], np.diff(r[0]))) for r in retained]),
                                    device=device, dtype=torch.float32)
    engine = BatchedSessions({})
    results = []
    if config.get('dynamic_only'):
        from system_study_2026_10.graph_compute import GraphCompute
        session_graphs = GraphCompute()
    write_json(args.output/'configuration.json', dict(config=config, users=users, command=sys.argv,
        model=model_record, source_sha256=sha256(source_path), code_sha256=sha256(Path(__file__)),
        scope='Real retained V4 model/cache operands; repeated operation diagnostic. Appends reuse retained tail operands to measure shapes, not a new causal quality trace. Both methods graphed; no population timing inference.'))
    for batch in config['batch_sizes']:
        for operation, width in [('score',16), ('append',1), ('append',16), ('rebuild',1024)]:
            for method in (['native'] if operation == 'rebuild' else ['native', 'relay']):
                cache = HSTUKVCache(cache0.k[:,:batch].clone(), cache0.v[:,:batch].clone(), 1024)
                state = LightweightValueState(codes0[:,:batch].clone(), 1024)
                ids = (query_ids[:batch].clone() if operation == 'score' else history_ids[:batch,-width:].clone())
                behaviors = history_behavior[:batch,-width:].clone()
                delta = (query_delta[:batch].clone() if operation == 'score' else history_delta[:batch,-width:].clone())
                count = 1024
                def compute():
                    if operation == 'score':
                        reader = (adapter.make_history_read(state, code_count=count,
                            gate_query=operation_gate_query(adapter, model, ids)) if method == 'relay' else native_history_read)
                        return score(model, cache, ids, delta, history_read_fn=reader)[0]
                    if operation == 'rebuild':
                        return model.compute_kv(ids, behaviors, delta)
                    with engine._corrected_append(model, adapter if method == 'relay' else None,
                            cache, state if method == 'relay' else None, ids, width,
                            count if method == 'relay' else 0, 1024):
                        return append_with_rolling_band(model, cache, ids, behaviors, delta, 1024)
                if config.get('dynamic_only'):
                    checks = []
                    for count in ([1024,511,7,0] if method == 'relay' else [0]):
                        ids.copy_(ids.flip(0 if batch > 1 else 1))
                        state.codes.copy_(codes0[:,:batch]); state.codes[:,:,count:].zero_()
                        expected = compute()
                        actual = session_graphs.run(operation,model,adapter if method == 'relay' else None,
                            cache,state if method == 'relay' else None,ids,delta,
                            behavior=None if operation == 'score' else behaviors,coded_rows=count)
                        for a,e in zip(tensors(actual),tensors(expected)):
                            torch.testing.assert_close(a,e,atol=3e-5,rtol=3e-5)
                        checks.append(dict(count=count,max_abs=max(float((a-e).abs().max())
                            for a,e in zip(tensors(actual),tensors(expected)))))
                    record = dict(batch=batch,operation=operation,width=width,method=method,checks=checks)
                    results.append(record); print(json.dumps(record),flush=True)
                    write_json(args.output/'summary.json',dict(status='running',records=results,graphs=session_graphs.describe()))
                    continue
                # Warm and capture on a side stream, then test changed item inputs.
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        compute()
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                before = torch.cuda.memory_allocated(device)
                begin = time.perf_counter()
                with torch.cuda.graph(graph, stream=stream):
                    output = compute()
                torch.cuda.synchronize(device)
                capture_seconds = time.perf_counter()-begin
                graph_bytes = torch.cuda.memory_allocated(device)-before
                original_ids = ids.clone()
                ids.copy_(original_ids.flip(0 if batch > 1 else 1))
                reference = compute()
                graph.replay()
                torch.cuda.synchronize(device)
                errors = []
                for actual, expected in zip(tensors(output), tensors(reference)):
                    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
                    errors.append(float((actual-expected).abs().max()))
                ids.copy_(original_ids)
                times = {'eager':[], 'graph':[], 'graph_with_staging':[]}
                def staged():
                    # Staging is real device-to-device copying, not omitted for graphs.
                    if operation != 'rebuild':
                        cache.k.copy_(cache0.k[:,:batch]); cache.v.copy_(cache0.v[:,:batch])
                    if method == 'relay':
                        state.codes.copy_(codes0[:,:batch])
                    ids.copy_(original_ids)
                    graph.replay()
                actions = {'eager':compute, 'graph':graph.replay, 'graph_with_staging':staged}
                for repeat in range(config['repeats']):
                    order = list(actions) if repeat%2 == 0 else list(reversed(actions))
                    for name in order:
                        torch.cuda.synchronize(device)
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin = time.perf_counter(); start.record()
                        for _ in range(config['iterations']):
                            actions[name]()
                        end.record(); end.synchronize()
                        times[name].append(dict(wall_ms=1000*(time.perf_counter()-begin)/config['iterations'],
                            cuda_ms=start.elapsed_time(end)/config['iterations']))
                record = dict(batch=batch, operation=operation, width=width, method=method,
                    max_abs=max(errors), capture_seconds=capture_seconds, graph_persistent_bytes=graph_bytes,
                    samples=times, median_wall_ms={k:statistics.median(r['wall_ms'] for r in v) for k,v in times.items()})
                results.append(record)
                print(json.dumps(record), flush=True)
                write_json(args.output/'summary.json', dict(status='running', records=results))
                del graph, output, reference
    write_json(args.output/'summary.json', dict(status='complete', records=results,
        graphs=session_graphs.describe() if config.get('dynamic_only') else None,
        limits='Resident fixed shapes only. Staging includes native KV/code/IDs, excludes host paging, user scatter, session ledger, compilation/setup amortization. Same graph optimization also applies to native rebuild. No new quality experiment.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    run(parser.parse_args())
