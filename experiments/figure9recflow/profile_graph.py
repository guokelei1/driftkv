"""Coordinator-run Max graph parity/profile using the completed edge-4 maps.

No training, quality scoring, configuration changes, or successor dispatch.
All 29 states, full 1024 history, and complete real request candidates remain.
Chunk choice uses latency only. CUDA is touched only after CLI parsing.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.figure9recflow.cache_ops import append_behaviors, score_candidates
from experiments.figure9recflow.common import DATA, CHAIN, HERE, write_json
from experiments.figure9recflow.engine import ReleaseEngine, load_model, tensors
from experiments.figure9recflow.graph_ops import GraphExecutor
from experiments.figure9recflow.inputs import RecFlowInputs
from hstu_kvcache.models.kv_cache import HSTUKVCache


DEFAULT_MAPS = HERE / "runs/full_recipe_e4_benchmark/V3_to_V4/maps.pt"


def clone(cache):
    return HSTUKVCache(cache.k.clone(), cache.v.clone(), cache.seq_len)


def check(a, b, name):
    if a.shape != b.shape or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise AssertionError(f"{name}: shape/nonfinite mismatch")
    error = float((a - b).abs().max()) if a.numel() else 0.
    if error > 2e-5:
        raise AssertionError(f"{name}: max abs {error} > 2e-5")
    return error


def check_cache(a, b, name):
    if a.seq_len != b.seq_len:
        raise AssertionError(f"{name}: history length mismatch")
    return max(check(a.k, b.k, name + ".k"), check(a.v, b.v, name + ".v"))


def request_inputs(request, device, *, diagnostic=False):
    features = torch.tensor(np.array(request.features, copy=True), device=device).unsqueeze(0)
    delta = torch.tensor([0. if diagnostic else request.query_delta], device=device, dtype=torch.float32)
    return features, delta


def latency_summary(seconds):
    values = np.asarray(seconds, dtype=np.float64)
    return dict(iterations=len(values), seconds=values.tolist(), mean_seconds=float(values.mean()),
                median_seconds=float(np.median(values)), min_seconds=float(values.min()),
                max_seconds=float(values.max()))


def timed_iterations(function, count, device):
    values = []
    for _ in range(count):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        values.append(time.perf_counter() - started)
    return latency_summary(values)


def profile_chunk(model, initial, user, chunk, *, steps, iterations, progress):
    device = initial.k.device
    executor = GraphExecutor(model, method_count=29, history_limit=1024,
                             chunk_size=chunk, borrow_state=True)
    torch.cuda.reset_peak_memory_stats(device)
    result = dict(chunk_size=chunk, checks=[], validation_scope=(
        "Real chronological requests and complete behavior timestamp blocks until >= requested steps; "
        "extra read-only numerical probes every 4 behaviors use the last request's complete candidates, delta=0. "
        "No labels or ranking-quality metrics are consumed."))
    try:
        progress(f"chunk_{chunk}_parity")
        untouched = clone(initial)
        eager_state, graph_state = clone(initial), initial
        ef, eb, ed = tensors(user.events['features'], user.events['behavior'].astype(np.int64) + 1,
                             user.event_deltas, device)
        cursor, real_requests, last_request = 0, 0, user.requests[0]
        for block in user.timeline():
            if block.kind == "request":
                for request in block.requests:
                    features, delta = request_inputs(request, device)
                    if eager_state.seq_len != request.history_length:
                        raise AssertionError("Actual causal request/history length differs")
                    actual = executor.score(graph_state, features, delta)
                    expected = score_candidates(model, eager_state, features, delta, chunk_size=chunk)
                    result["checks"].append(dict(kind="real_request", rid=int(request.rid),
                        candidates=len(request.features), after_behaviors=cursor,
                        score_max_abs=check(actual, expected, "real request")))
                    # A score can load the initial cache into the resident.
                    graph_state = executor.resident
                    last_request, real_requests = request, real_requests + 1
            else:
                for _ in range(len(block.events)):
                    raw = ef[:, cursor:cursor + 1], eb[:, cursor:cursor + 1], ed[:, cursor:cursor + 1]
                    graph_state = executor.append(graph_state, *raw)
                    eager_state = append_behaviors(model, eager_state, *raw)
                    cursor += 1
                    # Includes the first write: capture/warmup must not consume
                    # an extra event before the first actual replay.
                    if cursor == 1 or cursor % 4 == 0:
                        item = dict(kind="scalar_fifo", after_behaviors=cursor,
                                    cache_max_abs=check_cache(graph_state, eager_state, "scalar FIFO"))
                        if cursor % 4 == 0:
                            features, delta = request_inputs(last_request, device, diagnostic=True)
                            item.update(candidates=len(last_request.features),
                                score_max_abs=check(executor.score(graph_state, features, delta),
                                    score_candidates(model, eager_state, features, delta, chunk_size=chunk),
                                    "periodic diagnostic read"))
                        result["checks"].append(item)
                if cursor >= steps:
                    break
        if cursor < steps:
            raise ValueError("The fixed user has fewer than the requested real append events")
        result.update(real_append_events=cursor, actual_requests_checked=real_requests,
                      final_cache_max_abs=check_cache(graph_state, eager_state, "final real state"),
                      initial_input_max_abs=check_cache(initial, untouched, "untouched full input"))
        del eager_state, untouched

        # Warm synchronized microbenchmarks. Reads use all candidates, including
        # the final partial graph chunk. The graph resident avoids artificial
        # repeated 29-cache input transfers that serving would not perform.
        progress(f"chunk_{chunk}_warm_microbench")
        features, delta = request_inputs(user.requests[0], device)
        executor.score(initial, features, delta)
        read_state = executor.resident
        for _ in range(3):
            executor.score(read_state, features, delta)
            score_candidates(model, initial, features, delta, chunk_size=chunk)
        result["eager_read"] = timed_iterations(
            lambda: score_candidates(model, initial, features, delta, chunk_size=chunk), iterations, device)
        result["graph_read"] = timed_iterations(
            lambda: executor.score(read_state, features, delta), iterations, device)

        # Repeating the same real input row is solely a microbenchmark. Both
        # paths begin at the same initial cache and perform the same 3 warmups
        # and measured writes; their complete resulting states are compared.
        raw = ef[:, :1], eb[:, :1], ed[:, :1]
        eager_holder = [initial]
        executor.score(initial, features, delta)  # restore, without consuming events
        graph_holder = [executor.resident]
        def eager_append():
            eager_holder[0] = append_behaviors(model, eager_holder[0], *raw)
        def graph_append():
            graph_holder[0] = executor.append(graph_holder[0], *raw)
        for _ in range(3):
            eager_append()
            graph_append()
        result["eager_scalar_append"] = timed_iterations(eager_append, iterations, device)
        result["graph_scalar_append"] = timed_iterations(graph_append, iterations, device)
        result["repeated_write_final_max_abs"] = check_cache(graph_holder[0], eager_holder[0], "microbench FIFO")
        result["microbench"] = dict(warmup_iterations=3, synchronized_each_iteration=True,
            candidates=len(user.requests[0].features), methods=29, history_rows=1024,
            append_scope="same real first behavior repeated; both paths consume identical 3+iterations rows; timing only")
        result["read_speedup"] = result["eager_read"]["median_seconds"] / result["graph_read"]["median_seconds"]
        result["append_speedup"] = result["eager_scalar_append"]["median_seconds"] / result["graph_scalar_append"]["median_seconds"]
        result["graph_stats"] = dict(executor.stats)
        result["graph_count"] = int(executor.read_graph is not None) + int(executor.append_graph is not None)
        if result["graph_count"] != 2 or executor.stats["read_captures"] != 1 or executor.stats["append_captures"] != 1:
            raise AssertionError("Expected exactly two bounded graphs, each captured once")
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        result["status"] = "passed"
        return result
    finally:
        torch.cuda.synchronize(device)
        executor.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--chain", type=Path, default=CHAIN)
    parser.add_argument("--chunks", type=int, nargs="+", choices=(64, 128, 256), default=[64, 128, 256])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 16 <= args.steps <= 64 or not 10 <= args.iterations <= 100 or len(set(args.chunks)) != len(args.chunks):
        parser.error("Use 16..64 real steps, 10..100 timed iterations, and distinct chunks")
    output = args.output or HERE / "runs/graph_profile" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    if output.exists():
        raise FileExistsError(f"Retain previous output: {output}")
    report = dict(status="started", uid=246, edge="V3_to_V4", tolerance_max_abs=2e-5,
                  maps=str(args.maps.resolve()), chunk_results=[], scope="numerical validation and latency only; no quality-based choices")
    started = time.perf_counter()
    def progress(phase):
        report["phase"] = phase
        write_json(output, report)
        print(json.dumps(dict(phase=phase, elapsed_seconds=time.perf_counter() - started)), flush=True)
    try:
        progress("read_fixed_inputs_and_completed_recipe")
        inputs = RecFlowInputs(args.data, args.chain)
        if inputs.eval_uids[:2].tolist() != [32422, 246]:
            raise ValueError("The recorded fixed-prefix numerical user changed")
        edge = next(e for e in inputs.edges if e.id == "V3_to_V4")
        user = inputs.day(edge.evaluation_day, uids=[246])[0]
        if len(user.initial_history) != 1024 or not user.requests or len(user.events) < args.steps:
            raise ValueError("Fixed numerical user lacks the required full input")
        plan_path = args.maps.parent.parent / "plan.json"
        plan = json.loads(plan_path.read_text())
        config, rows = plan["config"], plan["policies"]
        fitted = torch.load(args.maps, map_location="cpu", weights_only=False)
        if fitted.get("diagnostic_only") is not False or fitted.get("edge_id") != edge.id:
            raise ValueError("Expected the completed full eight-epoch edge-4 recipe")
        map_rows = [p for p in rows if p["method"] == "kv_map"]
        if len(rows) != 29 or len(map_rows) != 6 or fitted["actual_policies"] != map_rows:
            raise ValueError("Saved full-map policies differ from the 29-point plan")
        for policy in map_rows:
            profile = fitted["profiles"][policy["tag"]]
            if profile["flop_breakdown"]["epochs"] != 8 or profile["fit_users"] != policy["fit_users"]:
                raise ValueError("A map did not execute its declared full fit recipe")
        torch.set_num_threads(4)
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda", args.gpu)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        progress("load_actual_max_and_six_trained_maps")
        old = load_model(edge.source_checkpoint, inputs.data, device, backend=config["backend"])
        current = load_model(edge.target_checkpoint, inputs.data, device, backend=config["backend"])
        maps = {name: weight.to(device) for name, weight in fitted["maps"].items()}
        engine = ReleaseEngine(old, current, rows, maps, candidate_chunk=128,
                               backend=config["backend"], append_backend=config["append_backend"])
        report.update(source_checkpoint=str(edge.source_checkpoint), target_checkpoint=str(edge.target_checkpoint),
            backend=config["backend"], append_backend=config["append_backend"], tf32=False,
            real_day_events=len(user.events), real_day_requests=len(user.requests),
            first_request_candidates=len(user.requests[0].features), full_history_rows=1024,
            initialization_scope="existing ReleaseEngine.initialize, six real trained nonidentity maps")
        if config["backend"] != config["append_backend"]:
            raise ValueError("This fixed-model graph profile requires one unchanged attention backend")
        with torch.no_grad():
            progress("initialize_all_29_real_states")
            initial, ledger = engine.initialize(user)
            report["initialization_ledger"] = ledger
            map_effects = {}
            for index, policy in enumerate(rows):
                if policy["method"] == "kv_map":
                    difference = max(float((initial.k[:, index] - initial.k[:, 0]).abs().max()),
                                     float((initial.v[:, index] - initial.v[:, 0]).abs().max()))
                    if difference == 0:
                        raise AssertionError(f"Trained map unexpectedly acts as identity: {policy['tag']}")
                    map_effects[policy["tag"]] = difference
            report["trained_map_max_abs_change_from_reuse"] = map_effects
            for chunk in args.chunks:
                report["chunk_results"].append(profile_chunk(current, initial, user, chunk,
                    steps=args.steps, iterations=args.iterations, progress=progress))
                gc.collect()
                torch.cuda.empty_cache()
                write_json(output, report)
        fastest = min(report["chunk_results"], key=lambda result: result["graph_read"]["median_seconds"])
        report.update(status="passed", recommended_chunk_for_this_input=fastest["chunk_size"],
            recommendation_scope="lowest warm median graph-read latency on fixed UID246 first request only; not a full-population optimum; no configuration changed")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output, report)
        print(f"{report['status']}: {output}", flush=True)


if __name__ == "__main__":
    main()
