"""Bounded numerical GPU check; this entry point never launches successors.

Example (run only through the experiment coordinator):
    PYTHONPATH=src:. python -m experiments.figure9recflow.gpu_check \
        --gpu 0 --edge V0_to_V1 --users 1 --output /tmp/recflow_gpu_check.json

Uses at most two real users with a 1024-row cutover history and one complete
candidate request. This deterministic shape selection is ONLY a numerical
diagnostic, not a quality population. The six maps are explicit identity maps;
the check makes no claim about fitted-map quality or complete serving metrics.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.figure9recflow.alignment import MapWeights, apply_map
from experiments.figure9recflow.cache_ops import (
    append_behaviors, deviation_scores, encode_cache, recompute_interval,
    recompute_positions, recompute_tail, score_candidates, select_positions,
)
from experiments.figure9recflow.common import CHAIN, DATA, HERE, policies, write_json
from experiments.figure9recflow.inputs import RecFlowInputs
from experiments.recflow_continuous_chain.code.evaluate import load_model
from experiments.recflow_release_design.code.interval_protocol import ancestry
from hstu_kvcache.models.kv_cache import HSTUKVCache


def _tensor(values, device, *, dtype=None):
    return torch.as_tensor(np.array(values, copy=True), device=device, dtype=dtype).unsqueeze(0)


def _events(events, deltas, device):
    return (_tensor(events["features"], device, dtype=torch.long),
            _tensor(events["behavior"].astype(np.int64) + 1, device),
            _tensor(deltas, device, dtype=torch.float32))


def _identity_map(cache, width, heads):
    layers, _, _, hidden = cache.k.shape
    dim, channels = hidden // heads, 2 * layers * heads
    def zeros(*shape):
        return cache.k.new_zeros(shape)
    mx, sx = zeros(channels, 1, dim), cache.k.new_ones((channels, 1, dim))
    tensors = (mx, sx, zeros(channels, dim, width), zeros(channels, 1, width),
               zeros(channels, width, dim), zeros(channels, 1, dim),
               zeros(channels, 1, dim), cache.k.new_ones((channels, 1, dim)))
    return MapWeights(layers, heads, dim, tensors)


def _stack(caches):
    return HSTUKVCache(torch.cat([c.k for c in caches], dim=1),
                      torch.cat([c.v for c in caches], dim=1), caches[0].seq_len)


def _error(actual, expected, name, tolerance=2e-5):
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != {expected.shape}")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError(f"{name}: nonfinite result")
    error = float((actual - expected).abs().max()) if actual.numel() else 0.
    if error > tolerance:
        raise AssertionError(f"{name}: max absolute error {error} exceeds {tolerance}")
    return error


def _cache_error(actual, expected, name):
    if actual.seq_len != expected.seq_len:
        raise AssertionError(f"{name}: cache lengths differ")
    return max(_error(actual.k, expected.k, name + ".k"),
               _error(actual.v, expected.v, name + ".v"))


@contextmanager
def _timed(record, name, device):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        record[name] = time.perf_counter() - started


def _choose_users(inputs, edge, count):
    """First qualifying UIDs in the fixed permutation; never inspect scores."""
    day = inputs.day(edge.evaluation_day)
    selected = []
    for index in range(len(day)):
        # Avoid materializing request candidates for obviously short histories.
        snapshot = inputs.history(int(day.uids[index]), day.start_ms)
        if len(snapshot.events) != 1024:
            continue
        user = day[index]
        if user.requests:
            selected.append(user)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError("Insufficient real full-history users for this numerical diagnostic")
    return selected


def _check_user(old, current, user, rows, device, append_count):
    report = dict(uid=int(user.uid), history_rows=len(user.initial_history),
                  methods=len(rows), timings_seconds={}, checks={})
    timing, checks = report["timings_seconds"], report["checks"]
    raw = _events(user.initial_history, user.initial_deltas, device)
    request = user.requests[0]
    features = _tensor(request.features, device, dtype=torch.long)
    # This is a fixed synthetic read of the cutover cache, not a metric at the
    # later request's timestamp. Candidates remain the complete logged set.
    query_delta = torch.tensor([0.], device=device)
    report.update(candidate_count=len(request.features), candidate_source_rid=int(request.rid),
                  read_scope="complete real candidate features; synthetic cutover query_delta=0; no quality metric")
    with _timed(timing, "encode_old_current", device):
        captured = encode_cache(old, *raw, capture=True)
        rebuilt = encode_cache(current, *raw)
    batch = dict(history_features=raw[0], behaviors=raw[1], deltas=raw[2],
                 lengths=torch.tensor([len(user.initial_history)], device=device))
    with _timed(timing, "production_encode_parity", device):
        checks["current_vs_production_fast"] = _cache_error(rebuilt, current.encode_history(batch), "native fast")
        checks["current_vs_production_dense_output"] = _cache_error(
            rebuilt, current.encode_history(batch, reference=True), "native full-output")
    with _timed(timing, "prepare_29_states", device):
        scores = deviation_scores(current, captured.cache, *raw)
        states = []
        for policy in rows:
            kind = policy["method"]
            if kind == "reuse":
                state = captured.cache
            elif kind == "rebuild":
                state = rebuilt
            elif kind == "layer":
                state = recompute_interval(current, captured.cache, *raw, policy["interval"],
                                           layer_inputs=captured.layer_inputs)
            elif kind == "tail":
                state = recompute_tail(current, captured.cache, *raw, policy["tokens"])
            elif kind == "deviation":
                state = recompute_positions(current, captured.cache, *raw,
                                            select_positions(scores, policy["tokens"]))
            elif kind == "kv_map":
                state = apply_map(captured.cache, _identity_map(captured.cache, policy["width"], 10))
                checks[policy["tag"] + "_identity"] = _cache_error(state, captured.cache, policy["tag"])
            else:
                raise ValueError(kind)
            states.append(state)
        resident = _stack(states)
    candidate_vectors = current.features(features)
    with _timed(timing, "score_29_batched", device):
        batched_scores = score_candidates(current, resident, features, query_delta,
                                          candidate_vectors=candidate_vectors)
    with _timed(timing, "score_29_individual", device):
        single_scores = torch.cat([score_candidates(current, state, features, query_delta,
                                                    candidate_vectors=candidate_vectors)
                                   for state in states])
    checks["score_method_batch_max_abs"] = _error(batched_scores, single_scores, "29 method scores")
    checks["score_per_method"] = {p["tag"]: _error(batched_scores[i], single_scores[i], p["tag"])
                                   for i, p in enumerate(rows)}
    # Actual next behaviors where available; otherwise explicitly synthetic
    # writes reuse raw event features, retaining valid token fields and dt.
    if len(user.events) >= append_count:
        appended_raw = _events(user.events[:append_count], user.event_deltas[:append_count], device)
        report["append_input"] = "first real post-cutover behavior events"
    else:
        appended_raw = tuple(value[:, -append_count:].clone() for value in raw)
        report["append_input"] = "synthetic repeat of last observed event features/deltas; numerical only"
    report["append_events"] = appended_raw[0].shape[1]
    with _timed(timing, "append_29_batched", device):
        appended = append_behaviors(current, resident, *appended_raw, window_size=1024)
    with _timed(timing, "append_29_individual", device):
        independent_appends = [append_behaviors(current, state, *appended_raw, window_size=1024)
                               for state in states]
    errors = {}
    for index, (policy, expected) in enumerate(zip(rows, independent_appends, strict=True)):
        actual = HSTUKVCache(appended.k[:, index:index + 1], appended.v[:, index:index + 1], appended.seq_len)
        errors[policy["tag"]] = _cache_error(actual, expected, policy["tag"] + " append")
    checks["append_per_method"] = errors
    checks["append_method_batch_max_abs"] = max(errors.values())
    del independent_appends
    # Reader equality after writes also checks that batched state layout is not
    # accidentally mixing the different methods' histories.
    with _timed(timing, "post_append_read_batch_vs_single", device):
        combined = score_candidates(current, appended, features, query_delta, candidate_vectors=candidate_vectors)
        outputs = []
        for index in range(len(rows)):
            state = HSTUKVCache(appended.k[:, index:index + 1], appended.v[:, index:index + 1], appended.seq_len)
            outputs.append(score_candidates(current, state, features, query_delta, candidate_vectors=candidate_vectors))
        checks["post_append_score_method_batch_max_abs"] = _error(combined, torch.cat(outputs), "post-append scores")
    report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="Visible CUDA device index; no multi-GPU dispatch")
    parser.add_argument("--edge", required=True, help="Published edge ID, e.g. V0_to_V1, or 1-based accepted edge number")
    parser.add_argument("--users", type=int, default=1, choices=(1, 2))
    parser.add_argument("--append-events", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--chain", type=Path, default=CHAIN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or HERE / "runs" / "gpu_check" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    if output.exists():
        raise FileExistsError(f"Retain existing diagnostic; use a new output path: {output}")
    report = dict(status="started", scope="bounded numerical check; no model training, quality claim, or successor",
                  gpu=args.gpu, requested_users=args.users, tolerance_max_abs=2e-5, tf32=False,
                  map_scope="six identity maps only; fitted map learning not tested",
                  timing_scope="synchronized single-pass wall time including allocations; not a warmed throughput estimate; post-append-read phase includes comparison",
                  users=[])
    write_json(output, report)
    started = time.perf_counter()
    try:
        torch.set_num_threads(4)
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda", args.gpu)
        torch.manual_seed(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
        inputs = RecFlowInputs(args.data, args.chain)
        edge = (inputs.edges[int(args.edge) - 1] if args.edge.isdecimal() and 1 <= int(args.edge) <= len(inputs.edges)
                else next((e for e in inputs.edges if e.id == args.edge), None))
        if edge is None:
            raise ValueError(f"Unknown accepted published edge: {args.edge}")
        lineage = ancestry(edge.target_checkpoint, edge.source_checkpoint)
        users = _choose_users(inputs, edge, args.users)
        report.update(edge=edge.id, evaluation_day=edge.evaluation_day,
                      source_checkpoint=str(edge.source_checkpoint), target_checkpoint=str(edge.target_checkpoint),
                      actual_lineage=lineage,
                      uid_selection="first fixed-permutation UIDs with full cutover history and any request; numerical shapes only",
                      timings_seconds={}, device_name=torch.cuda.get_device_name(device))
        with _timed(report["timings_seconds"], "load_models", device):
            old, old_meta = load_model(edge.source_checkpoint, args.data, device)
            current, current_meta = load_model(edge.target_checkpoint, args.data, device)
        if old_meta["proposal"] != current_meta["proposal"]:
            raise ValueError("Source/target model proposals differ")
        cfg = current.backbone.cfg
        if (cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.max_seq_len) != (16, 320, 10, 32, 1025):
            raise ValueError("Expected actual RecFlow Max 16x320/10x32 with 1024 history + one query")
        report["geometry"] = dict(layers=cfg.num_layers, width=cfg.hidden_size, heads=cfg.num_heads,
                                  head_dim=cfg.head_dim, max_seq_len=cfg.max_seq_len,
                                  block_variant=cfg.block_variant, relative_position_bias=cfg.relative_position_bias)
        report["attention_backend"] = [block.attn.backend for block in current.backbone.blocks]
        rows = policies()
        if len(rows) != 29:
            raise ValueError("Expected all 29 declared method configurations")
        with torch.inference_mode():
            for user in users:
                report["users"].append(_check_user(old, current, user, rows, device, args.append_events))
                write_json(output, report)
                gc.collect()
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        if torch.cuda.is_initialized():
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(args.gpu)
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(args.gpu)
        write_json(output, report)
        print(f"{report['status']}: {output}", flush=True)


if __name__ == "__main__":
    main()
