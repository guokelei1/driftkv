"""Four-GPU extraction with the unchanged global shared-read ridge fit.

Paired caches live in CPU memory. Each GPU owns round-robin complete batches,
so every nested teacher budget uses all four devices without changing batch
shapes. Only the requested cache batch and teacher layer enter GPU memory.
"""

from __future__ import annotations

import gc
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design import bias_read_probe
from design.competitor_models import sha256_file
from design.run_auc_read_probe import fit_rule
from design.run_shared_read_probe import collect_cache, device_parameters
from evaluate_yambda500m_foundation_raw import load_model
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.models import HSTUKVCache

ARRAY_NAMES = ("timestamps", "items", "behaviors", "deltas", "query_deltas")


def _arrays(directory, users):
    return tuple(np.load(Path(directory) / f"{name}.npy", mmap_mode="c")[:users]
                 for name in ARRAY_NAMES)


def _device(gpu, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    return device


def _model(pair, role, device):
    model, payload = load_model(Path(pair[role]["checkpoint"]), device)
    assert payload["config"] == pair["config"]
    del payload
    return model


@torch.inference_mode()
def _worker(rank, gpus, pair, directory, panel_path, users, batch_size, threads,
            wanted, observed, connection):
    try:
        began = time.perf_counter()
        device = _device(gpus[rank], threads)
        arrays = _arrays(directory, users)
        panel = np.load(panel_path, mmap_mode="c")[:users]
        length, layers, width = arrays[1].shape[1], pair["config"]["num_layers"], pair["config"]["hidden_size"]
        batches = [(start, min(start + batch_size, users))
                   for start in range(rank * batch_size, users, len(gpus) * batch_size)]
        local_users = sum(stop-start for start, stop in batches)
        shape = (layers, local_users, length, width)
        source = HSTUKVCache(torch.empty(shape), torch.empty(shape), length)
        teacher = HSTUKVCache(torch.empty(shape), torch.empty(shape), length)
        parent, current = _model(pair, "parent", device), _model(pair, "current", device)
        offset = 0
        locations = []
        for start, stop in batches:
            events = tuple(torch.as_tensor(array[start:stop], device=device) for array in arrays[1:4])
            for model, destination in ((parent, source), (current, teacher)):
                cache = model.compute_kv(*events)
                destination.k[:, offset:offset+stop-start].copy_(cache.k)
                destination.v[:, offset:offset+stop-start].copy_(cache.v)
                del cache
            locations.append((start, stop, offset))
            offset += stop-start
            del events
        del parent
        gc.collect()
        torch.cuda.empty_cache()
        connection.send(dict(status="ready", rank=rank, users=local_users,
            paired_cache_bytes=4 * source.k.numel() * source.k.element_size(),
            prefill_seconds=time.perf_counter()-began,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30)))
        while True:
            command = connection.recv()
            if command is None:
                break
            count, layer, parameters = command
            started = time.perf_counter()
            installed = device_parameters(parameters, device)
            processed = 0
            for start, stop, offset in locations:
                if start >= count:
                    break
                stop = min(stop, count)
                size = stop-start
                src = HSTUKVCache(source.k[:, offset:offset+size].to(device),
                                  source.v[:, offset:offset+size].to(device), length)
                counts = torch.full((size,), float(length), device=device)
                override = bias_read_probe.make_history_override(installed, counts)
                _, trace = score(current, src, torch.as_tensor(panel[start:stop], device=device),
                    torch.as_tensor(arrays[4][start:stop], device=device),
                    trace=True, history_override=override)
                query, native = trace.queries[layer], trace.history_heads[layer]
                target = history_read(current.blocks[layer].attn, query,
                    teacher.k[layer, offset:offset+size].to(device),
                    teacher.v[layer, offset:offset+size].to(device))
                wanted[start:stop].copy_(((target-native)/length).cpu())
                observed[start:stop].copy_((native.transpose(1, 2).flatten(2)/length).cpu())
                processed += size
                del src, counts, override, trace, query, native, target, _
            connection.send(dict(status="extracted", rank=rank, users=processed,
                seconds=time.perf_counter()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30)))
        connection.close()
    except Exception:
        connection.send(dict(status="failed", rank=rank, traceback=traceback.format_exc()))
        connection.close()
        raise


def _receive(connections, expected):
    records = [connection.recv() for connection in connections]
    failed = [record for record in records if record["status"] != expected]
    if failed:
        raise RuntimeError(str(failed))
    return records


def calibrate_expanded(pair, array_directory, panel_path, budgets, *, batch_size=64,
                       gpus=(0, 1, 2, 3), threads=4):
    """Return CPU rules/fits; caller owns sealed UID and candidate selection.

    ``panel_path`` contains [users,16] int64 candidate IDs in the same ordered
    UID axis as the five prepared arrays. A budget uses exactly the first
    ``budget`` users. This routine reads no evaluation users or labels.
    """
    budgets = list(budgets)
    assert budgets == sorted(set(budgets)) and budgets[0] >= len(gpus)*batch_size
    users = max(budgets)
    arrays = _arrays(array_directory, users)
    uids = np.load(Path(array_directory) / "uids.npy")[:users]
    panel = np.load(panel_path, mmap_mode="r")
    assert len(uids) == users and len(set(uids.tolist())) == users
    assert panel.shape[0] >= users and panel.shape[1] == 16
    assert arrays[1].shape == (users, 1024) and arrays[4].shape == (users,)
    assert all(count % batch_size == 0 or count == users for count in budgets)
    length, layers = arrays[1].shape[1], pair["config"]["num_layers"]
    heads, width = pair["config"]["num_heads"], pair["config"]["hidden_size"]
    wanted = torch.empty(users, heads, 16, width//heads).share_memory_()
    observed = torch.empty(users, 16, width).share_memory_()
    context = mp.get_context("spawn")
    children, connections = [], []
    started = time.perf_counter()
    try:
        for rank in range(len(gpus)):
            parent_connection, child_connection = context.Pipe()
            process = context.Process(target=_worker, args=(rank, gpus, pair, str(array_directory),
                str(panel_path), users, batch_size, threads, wanted, observed, child_connection))
            process.start()
            child_connection.close()
            children.append(process)
            connections.append(parent_connection)
        resources = _receive(connections, "ready")
        print(f"Prepared paired CPU caches for {users} calibration users on {len(gpus)} GPUs", flush=True)
        rules, fitting, extraction = {}, {}, {}
        for count in budgets:
            for conditioned in (False, True):
                name = f"{'personal' if conditioned else 'shared'}_{count}"
                parameters, fits, times = [], [], []
                for layer in range(layers):
                    for connection in connections:
                        connection.send((count, layer, parameters))
                    records = _receive(connections, "extracted")
                    assert sum(record["users"] for record in records) == count
                    solve_start = time.perf_counter()
                    p, fit = bias_read_probe.fit_layer(wanted[:count], observed[:count],
                        torch.full((count,), float(length)), conditioned)
                    parameters.append(p)
                    fits.append(fit)
                    times.append(dict(workers=records, fit_seconds=time.perf_counter()-solve_start))
                    print(f"  {name} layer {layer+1}/{layers}", flush=True)
                rules[name], fitting[name], extraction[name] = parameters, fits, times
        return dict(rules=rules, fitting=fitting, teacher_budgets=budgets,
            calibration_uids=uids.tolist(), model_binding={r: pair[r]["checkpoint_sha256"] for r in ("parent", "current")},
            panel_sha256=sha256_file(Path(panel_path)), source_sha256=sha256_file(Path(__file__)),
            execution=dict(gpus=list(gpus), batch_size=batch_size, threads_per_worker=threads,
                assignment="round-robin complete global batches", paired_cache_location="CPU memory",
                fitting="unchanged bias_read_probe.fit_layer on globally ordered CPU FP64 inputs",
                workers=resources, extraction=extraction, elapsed_seconds=time.perf_counter()-started))
    finally:
        for connection, process in zip(connections, children):
            if process.is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError):
                    pass
            connection.close()
        for process in children:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()


@torch.inference_mode()
def compare_reference(pair, array_directory, panel_path, artifact, *, budget=256,
                      batch_size=64, gpu=0, threads=4):
    """Compare both fitted arms with the original in-GPU-cache implementation."""
    device = _device(gpu, threads)
    arrays = _arrays(array_directory, budget)
    panel = torch.as_tensor(np.load(panel_path)[:budget], device=device)
    delta = torch.as_tensor(arrays[4], device=device)
    parent, current = _model(pair, "parent", device), _model(pair, "current", device)
    source = collect_cache(parent, arrays, budget, batch_size, device)
    teacher = collect_cache(current, arrays, budget, batch_size, device)
    checks = {}
    for conditioned in (False, True):
        name = f"{'personal' if conditioned else 'shared'}_{budget}"
        reference, _ = fit_rule(current, source, teacher, panel, delta, conditioned, batch_size, device)
        actual = artifact["rules"][name]
        differences = []
        for old, new in zip(reference, actual, strict=True):
            assert old.keys() == new.keys()
            for key in old:
                torch.testing.assert_close(new[key], old[key], atol=2e-5, rtol=2e-5)
                differences.append(float((new[key]-old[key]).abs().max()))
        maximum_logit_difference = 0.0
        for start in range(0, budget, batch_size):
            stop = min(start+batch_size, budget)
            src = HSTUKVCache(source.k[:, start:stop], source.v[:, start:stop], source.seq_len)
            counts = torch.full((stop-start,), float(source.seq_len), device=device)
            predictions = [score(current, src, panel[start:stop], delta[start:stop],
                history_override=bias_read_probe.make_history_override(device_parameters(p, device), counts))[0]
                for p in (reference, actual)]
            torch.testing.assert_close(predictions[0], predictions[1], atol=2e-5, rtol=2e-5)
            maximum_logit_difference = max(maximum_logit_difference,
                float((predictions[0]-predictions[1]).abs().max()))
        checks[name] = dict(maximum_parameter_difference=max(differences),
            maximum_logit_difference=maximum_logit_difference)
    return dict(status="passed", budget=budget, batch_size=batch_size, arms=checks,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30))
