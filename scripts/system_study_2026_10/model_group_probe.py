#!/usr/bin/env python3
"""Measure native operation shapes with existing real checkpoints and histories.

Batch 8 repeats one selected history; it is a synthetic operator shape, not an
eight-user service workload. No labels, correction fitting or quality metric
are used. Inputs are already resident during each measured API call.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import torch

from design_one.prepare_history import load_histories_semijoin
from evaluate_yambda500m_foundation_raw import load_model
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models.state_transition import append_with_rolling_band
from hstu_kvcache.design_two.replay import _short_prefix_backend, MAX_DELTA_SECONDS
from read_correction_2026_09.cost import CostModel, eager_read
from unified_reuse_2026_09.common import sha256, write_json

DAY = 86400


def select_extremes(histories, at, context):
    """Use strictly pre-release length only; never select on later appends."""
    profiles = []
    for uid, (times, _, _) in histories.items():
        before = int(np.searchsorted(times, at, side="left"))
        if before:
            profiles.append(dict(uid=int(uid), length=min(before, context)))
    profiles.sort(key=lambda row: (row["length"], row["uid"]))
    if len(profiles) < 2:
        raise ValueError("two nonempty pre-release histories required")
    shortest = profiles[0]
    # Resolve both length ties by smallest uid, selecting a distinct second user.
    longest = min((r for r in profiles if r["uid"] != shortest["uid"]),
                  key=lambda r: (-r["length"], r["uid"]))
    return [dict(bucket="shortest_in_fixed_pool", **shortest),
            dict(bucket="longest_in_fixed_pool", **longest)], profiles


def history_digest(events):
    digest = hashlib.sha256()
    for column in events:
        digest.update(np.asarray(column, dtype="<i8").tobytes())
    return digest.hexdigest()


def repeated_input(values, batch, device, dtype):
    return torch.as_tensor(values, device=device, dtype=dtype)[None].repeat(batch, 1)


def native_fused_read(index, attention, q, k, v):
    """Existing zero-code kernel, independently checked here for H6 and H10."""
    del index
    if (q.shape[-1], attention.activation, attention.block_variant) != (32, "elu_plus1", "legacy"):
        raise ValueError("probe requires the retained legacy head32 models")
    if attention.position_bias is not None:
        raise ValueError("position-bias kernel is outside this probe")
    from implementation_2026_10.reader_kernel import launch_read
    # HAS_CODE=False eliminates all code pointers and coordinate arithmetic.
    return launch_read(q, k, v, k[:, :0, :1], q, q, q, None, (0, 0, 0, 0),
                       code_count=0, scale=float(attention.scale))


def timed_calls(function, *, device, warmup, repeats):
    started = time.perf_counter()
    for _ in range(warmup):
        result = function()
        del result
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - started
    rows, result = [], None
    for repeat in range(repeats):
        del result
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        result = function()
        end.record()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        rows.append(dict(repeat=repeat, wall_seconds=elapsed,
            cuda_event_seconds=begin.elapsed_time(end) / 1000,
            allocated_before_bytes=before,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_increment_bytes=torch.cuda.max_memory_allocated(device)-before,
            allocated_after_bytes=torch.cuda.memory_allocated(device)))
    return result, dict(warmup_seconds=warmup_seconds, measured_calls=rows,
        median_seconds=float(np.median([r["wall_seconds"] for r in rows])),
        mean_seconds=float(np.mean([r["wall_seconds"] for r in rows])),
        min_seconds=min(r["wall_seconds"] for r in rows),
        max_seconds=max(r["wall_seconds"] for r in rows),
        peak_increment_bytes=max(r["peak_increment_bytes"] for r in rows))


def check_close(actual, reference, plan):
    torch.testing.assert_close(actual, reference.expand_as(actual),
                               atol=plan["atol"], rtol=plan["rtol"])
    return float((actual-reference).abs().max().item())


def note(output, **values):
    values = dict(pid=os.getpid(), **values)
    write_json(output / "progress.json", values)
    print(json.dumps(values), flush=True)


@torch.inference_mode()
def run(args):
    plan = json.loads(args.config.read_text())
    device = torch.device(f"cuda:{args.gpu}")
    torch.set_num_threads(plan["threads"])
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    settings = dict(created_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        command=sys.argv, config=plan, config_sha256=sha256(args.config),
        source_sha256=sha256(Path(__file__)), device=torch.cuda.get_device_name(device),
        torch_version=torch.__version__, precision="FP32; native Triton attention and fused query use their existing policies",
        scope="Native operator coefficients only; no Large/Max correction adapter or method quality evidence",
        timing_scope="resident GPU inputs; synchronous API wall time incl launches; IO/H2D/load/warmup separate",
        batch8_scope="eight copies of one selected actual history, not eight independent users",
        query_scope="fixed valid catalog IDs 1..16, synthetic operator inputs, no recommendation labels",
        selection_scope="Medium reuses the D259 engineering panel, so V2 is retrospective operator sampling; all actual V2 inputs are strictly before D245. Large/Max pools use first-behavior-before-D231 plus uid only",
        flops_scope="analytical executed-shape convention, FMA=2; not hardware instructions or peak-throughput estimates")
    write_json(args.output / "configuration.json", settings)
    units, selections, history_loads, model_loads = [], [], [], []
    for scale in plan["scales"]:
        dataset_path = ROOT / plan["datasets"][scale]
        dataset = json.loads(dataset_path.read_text())
        known = int(dataset["foundation_items"])
        uids = (json.loads((ROOT / plan["medium_selection"]).read_text())["selected_uids"]
                if scale == "medium" else plan["candidate_uids"][scale])
        note(args.output, status="loading_histories", scale=scale, users=len(uids))
        started = time.perf_counter()
        histories = load_histories_semijoin(uids, dataset_path=dataset_path,
            known_vocab_size=known, oov_buckets=256, start_timestamp=min(plan["endpoints"].values())*DAY,
            end_timestamp=plan["history_end_day"]*DAY, max_history=plan["context"], threads=plan["threads"]).rows
        history_loads.append(dict(scale=scale, seconds=time.perf_counter()-started,
            loaded_rows=sum(len(value[0]) for value in histories.values()),
            dataset=str(dataset_path), dataset_sha256=sha256(dataset_path),
            mapping=str((dataset_path.parent/dataset["item_mapping_path"]).resolve()),
            known_vocab_size=known, candidate_uids=uids))
        # Freeze every endpoint's selection before inspecting post-release availability.
        selected_by_version = {}
        for version, day in plan["endpoints"].items():
            selected, profiles = select_extremes(histories, day*DAY, plan["context"])
            selected_by_version[version] = selected
            selections.append(dict(scale=scale, version=version, at=day*DAY,
                profiles=profiles, selected=selected,
                rule="minimum/maximum retained pre-release rows within fixed pool; uid ties"))
        write_json(args.output / "selection.json", selections)
        write_json(args.output / "history_loads.json", history_loads)
        for version, day in plan["endpoints"].items():
            at = day*DAY
            path = ROOT / f"results/unified_training_2026_09/{scale}/seed17/checkpoints/{version}/checkpoint_100.pt"
            note(args.output, status="loading_model", scale=scale, version=version)
            started = time.perf_counter()
            model, payload = load_model(path, device)
            model.requires_grad_(False)
            cfg = dict(payload["config"])
            del payload
            if model.cfg.num_items != known+256 or model.cfg.max_seq_len != plan["context"]:
                raise ValueError("checkpoint item space or context disagrees with this scale's dataset")
            if model.cfg.hidden_size != model.cfg.num_heads*32:
                raise ValueError("unsupported head width")
            for block in model.blocks:
                block.attn.backend = "triton"
            torch.cuda.synchronize(device)
            load_seconds = time.perf_counter()-started
            model_loads.append(dict(scale=scale, version=version, path=str(path),
                config=cfg, load_seconds=load_seconds, checkpoint_bytes=path.stat().st_size,
                model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                resident_allocated_bytes=torch.cuda.memory_allocated(device)))
            write_json(args.output / "model_loads.json", model_loads)
            for selected in selected_by_version[version]:
                uid = selected["uid"]
                times, items, behaviors = histories[uid]
                stop = int(np.searchsorted(times, at, side="left"))
                start = max(0, stop-plan["context"])
                prefix = (times[start:stop], items[start:stop], behaviors[start:stop])
                n = len(prefix[0])
                backend = "triton" if n == plan["context"] else "torch"
                cost = CostModel(model.cfg.hidden_size, model.cfg.num_layers,
                                 model.cfg.num_heads, backend)
                post_stop = min(stop+plan["append_width"], int(np.searchsorted(times, (day+14)*DAY, side="left")))
                post = tuple(column[stop:post_stop] for column in (times, items, behaviors))
                can_append = len(post[0]) == plan["append_width"]
                common = dict(scale=scale, version=version, uid=uid, bucket=selected["bucket"],
                    history_length=n, history_sha256=history_digest(prefix),
                    prefix_last_timestamp=int(prefix[0][-1]), release_timestamp=at,
                    native_backend=backend, checkpoint_config=cfg,
                    append_available=can_append, post_rows_available=len(post[0]))
                reference = {}
                for batch in plan["batch_sizes"]:
                    note(args.output, status="measuring", scale=scale, version=version,
                         uid=uid, history_length=n, batch=batch)
                    deltas = np.zeros(n, dtype=np.float32)
                    deltas[1:] = np.diff(prefix[0])
                    started = time.perf_counter()
                    item = repeated_input(prefix[1], batch, device, torch.long)
                    behavior = repeated_input(prefix[2], batch, device, torch.long)
                    delta = repeated_input(deltas, batch, device, torch.float32)
                    candidates = torch.arange(1, plan["candidates"]+1, device=device)[None].repeat(batch, 1)
                    query_delta = torch.full((batch,), float(at-prefix[0][-1]), device=device)
                    torch.cuda.synchronize(device)
                    transfer_seconds = time.perf_counter()-started
                    def rebuild():
                        with _short_prefix_backend(model, n, plan["context"]):
                            return model.compute_kv(item, behavior, delta)
                    def measure(name, function, flops, **extra):
                        value, timing = timed_calls(function, device=device,
                            warmup=plan["warmup_calls"], repeats=plan["measured_calls"])
                        row = dict(**common, batch=batch, operation=name, effective_flops=int(flops),
                            input_prepare_h2d_seconds=transfer_seconds, **timing, **extra)
                        units.append(row)
                        return value, row
                    cache, row = measure("rebuild", rebuild, cost.full_cache(n, batch=batch))
                    row["persistent_output_bytes"] = 2*cache.k.numel()*cache.k.element_size()
                    if batch == 1:
                        reference["k"], reference["v"] = cache.k.clone(), cache.v.clone()
                        row["batch_identity_max_abs"] = 0.
                    else:
                        row["batch_identity_max_abs"] = max(check_close(cache.k, reference["k"], plan),
                                                            check_close(cache.v, reference["v"], plan))
                    raw_scores, row = measure("query16_torch", lambda: score(model, cache, candidates, query_delta)[0],
                                             eager_read(cost, n, queries=plan["candidates"], batch=batch))
                    if batch == 1:
                        reference["scores"] = raw_scores.clone()
                    row["batch_identity_max_abs"] = check_close(raw_scores, reference["scores"], plan)
                    fused_scores, row = measure("query16_fused", lambda: score(model, cache, candidates, query_delta,
                        history_read_fn=native_fused_read)[0],
                        eager_read(cost, n, queries=plan["candidates"], batch=batch),
                        flops_note="logical query FLOPs; Triton Q16/K64 padding and TF32x3 instruction expansion excluded")
                    row["torch_reference_max_abs"] = check_close(fused_scores, raw_scores, plan)
                    if can_append:
                        appended_deltas = np.diff(np.concatenate((prefix[0][-1:], post[0]))).astype(np.float32)
                        np.clip(appended_deltas, 0, MAX_DELTA_SECONDS, out=appended_deltas)
                        new_items = repeated_input(post[1], batch, device, torch.long)
                        new_behaviors = repeated_input(post[2], batch, device, torch.long)
                        new_deltas = repeated_input(appended_deltas, batch, device, torch.float32)
                        def append():
                            with _short_prefix_backend(model, n, plan["context"]):
                                return append_with_rolling_band(model, cache, new_items, new_behaviors,
                                                                new_deltas, plan["context"])
                        appended, row = measure("append16", append, cost.band_append(n, plan["append_width"],
                                                batch=batch, window_size=plan["context"]),
                                                appended_events_sha256=history_digest(post),
                                                appended_last_timestamp=int(post[0][-1]))
                        if batch == 1:
                            reference["append_k"], reference["append_v"] = appended.k.clone(), appended.v.clone()
                        row["batch_identity_max_abs"] = max(check_close(appended.k, reference["append_k"], plan),
                                                            check_close(appended.v, reference["append_v"], plan))
                        del appended, new_items, new_behaviors, new_deltas
                    write_json(args.output / "completed_units.json", units)
                    del cache, raw_scores, fused_scores, item, behavior, delta, candidates, query_delta
                del reference
            del model
            gc.collect()
            torch.cuda.empty_cache()
            note(args.output, status="model_complete", scale=scale, version=version, units=len(units))
    summary = dict(status="complete_native_model_group_probe", units=units,
        selections=selections, history_loads=history_loads, model_loads=model_loads,
        scope=settings["scope"], timing_scope=settings["timing_scope"],
        unsupported_claims=["cross-model correction quality", "production throughput/p99",
            "whole deployment time without workload-shape counts", "synthetic B8 as eight independent users"])
    write_json(args.output / "summary.json", summary)
    note(args.output, status=summary["status"], units=len(units))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.stdout = sys.stderr = (args.output / "run.log").open("a", buffering=1)
    try:
        run(args)
    except BaseException as error:
        write_json(args.output / "summary.json", dict(status="failed", pid=os.getpid(),
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        raise
