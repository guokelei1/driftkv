"""Causal mixed-producer snapshots; terminal teachers always recompute Full.

Each user contributes one state. A synthetic release within their pre-release
history splits complete timestamp groups: Parent initializes a bounded prefix,
then Current genuinely appends the suffix with rolling eviction. The historical
``parent`` row key holds this mixed cache for compatibility with frozen fitters.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
import time

import numpy as np
import pyarrow as pa
import torch

from evaluate_yambda500m_foundation_raw import load_histories, load_model
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band
from read_correction_2026_09.calibrate import candidates
from read_correction_2026_09.v2.calibrate import mixed_candidates
from read_correction_v4.common import ROOT, edge_name, sha256
from selective_recompute_2026_09.cost import CostModel
from selective_recompute_2026_09.evaluate import native_backend, prefix_events


def assign_append_targets(train_uids, validation_uids, targets=(0, 256, 512, 896), seed=17):
    """Balance each disjoint partition without inspecting history or labels."""
    if set(train_uids).intersection(validation_uids):
        raise ValueError("training and validation users overlap")
    result = {}
    for partition, uids in (("train", train_uids), ("validation", validation_uids)):
        ordered = sorted(uids, key=lambda uid: hashlib.sha256(
            f"mixed-cache-v5:{seed}:{partition}:{int(uid)}".encode()).digest())
        for index, uid in enumerate(ordered):
            result[int(uid)] = int(targets[index % len(targets)])
    return result


def prepare_timeline(raw, *, cutover, history_length, append_target):
    """Select a complete-group release split and aligned bounded terminal events.

    If the desired release bisects a timestamp group, move it after that group;
    actual appends never exceed the requested target. Window eviction itself may
    split a group, exactly as the event-count-capped serving history does.
    """
    if history_length < 1 or append_target < 0:
        raise ValueError("invalid history length or append target")
    events = prefix_events(raw, cutover, history_length + append_target)
    if not events:
        raise ValueError("calibration user has no strictly pre-release events")
    times = np.asarray([int(event[0]) for event in events], dtype=np.int64)
    split = max(1, len(events) - append_target)
    if split < len(events) and times[split - 1] == times[split]:
        split = int(np.searchsorted(times, times[split], side="right"))
    prefix = events[max(0, split - history_length):split]
    suffix = events[split:]
    terminal = (prefix + suffix)[-history_length:]
    arrays = lambda values: tuple(np.asarray([int(e[i]) for e in values], dtype=np.int64) for i in range(3))
    return {"prefix": arrays(prefix), "suffix": arrays(suffix), "terminal": arrays(terminal),
        "append_target": int(append_target), "actual_append": len(suffix),
        "inherited_count": max(0, len(terminal) - len(suffix)),
        "pseudo_release_timestamp": int(suffix[0][0]) if suffix else int(cutover),
        "parent_last_timestamp": int(prefix[-1][0]), "terminal_last_timestamp": int(terminal[-1][0])}


def _inputs(timelines, selected, key, device):
    values = [timelines[uid][key] for uid in selected]
    times = torch.as_tensor(np.stack([v[0] for v in values]), device=device)
    items = torch.as_tensor(np.stack([v[1] for v in values]), dtype=torch.long, device=device)
    behaviors = torch.as_tensor(np.stack([v[2] for v in values]), dtype=torch.long, device=device)
    return times, items, behaviors


@torch.no_grad()
def capture_mixed(parent, current, timelines, uids, *, cutover, known, queries,
                  device, batch_size, history_length, attention_backend,
                  append_band_size=32, candidate_mode="uniform_known", seed=17):
    """Return fitter rows, executed-shape capture ledger and per-user state facts."""
    if append_band_size < 1:
        raise ValueError("append band size must be positive")
    if candidate_mode not in ("uniform_known", "mixed_recent_uniform"):
        raise ValueError(f"unknown calibration candidate mode: {candidate_mode}")
    groups = defaultdict(list)
    for uid in uids:
        t = timelines[uid]
        groups[(len(t["prefix"][0]), t["actual_append"], len(t["terminal"][0]))].append(uid)
    rows, state_records = {}, {}
    ledger = {"parent_full_flops": 0, "append_flops": 0, "teacher_full_flops": 0}
    hist = {name: Counter() for name in ("parent_calls", "teacher_calls", "append_calls")}
    costs = {backend: CostModel(current.cfg.hidden_size, len(current.blocks),
                 current.blocks[0].attn.num_heads, backend) for backend in ("torch", "triton")}
    for (prefix_n, suffix_n, terminal_n), group in sorted(groups.items()):
        rolling_backend = attention_backend if prefix_n == history_length else "torch"
        teacher_backend = attention_backend if terminal_n == history_length else "torch"
        for start in range(0, len(group), batch_size):
            selected = group[start:start + batch_size]
            batch = len(selected)
            times, items, behaviors = _inputs(timelines, selected, "prefix", device)
            deltas = torch.zeros_like(times, dtype=torch.float32)
            deltas[:, 1:] = times[:, 1:] - times[:, :-1]
            with native_backend([parent, current], rolling_backend):
                mixed = parent.compute_kv(items, behaviors, deltas)
                ledger["parent_full_flops"] += costs[rolling_backend].full_cache(prefix_n, batch=batch)
                hist["parent_calls"][f"{rolling_backend}:{prefix_n}:{batch}"] += 1
                if suffix_n:
                    suffix_times, suffix_items, suffix_behaviors = _inputs(timelines, selected, "suffix", device)
                    previous = torch.cat((times[:, -1:], suffix_times[:, :-1]), dim=1)
                    suffix_deltas = (suffix_times - previous).clamp(0, 7 * 86400).float()
                    offset = 0
                    while offset < suffix_n:
                        available = min(append_band_size, suffix_n - offset)
                        width = 1 << (available.bit_length() - 1)
                        before = mixed.seq_len
                        stop = offset + width
                        mixed = append_with_rolling_band(current, mixed,
                            suffix_items[:, offset:stop], suffix_behaviors[:, offset:stop],
                            suffix_deltas[:, offset:stop], history_length)
                        ledger["append_flops"] += costs[rolling_backend].band_append(
                            before, width, batch=batch, window_size=history_length)
                        hist["append_calls"][f"{rolling_backend}:{before}:{width}:{batch}"] += 1
                        offset = stop
            times, items, behaviors = _inputs(timelines, selected, "terminal", device)
            deltas = torch.zeros_like(times, dtype=torch.float32)
            deltas[:, 1:] = times[:, 1:] - times[:, :-1]
            with native_backend([current], teacher_backend):
                teacher = current.compute_kv(items, behaviors, deltas)
            ledger["teacher_full_flops"] += costs[teacher_backend].full_cache(terminal_n, batch=batch)
            hist["teacher_calls"][f"{teacher_backend}:{terminal_n}:{batch}"] += 1
            if mixed.seq_len != teacher.seq_len or mixed.seq_len != terminal_n:
                raise RuntimeError("mixed and terminal Full cache positions do not align")
            for index, uid in enumerate(selected):
                timeline = timelines[uid]
                ids = (candidates(uid, known, queries) if candidate_mode == "uniform_known" else
                       mixed_candidates(uid, timeline["terminal"][1], known, queries, seed))
                rows[uid] = {
                    "parent": HSTUKVCache(mixed.k[:, index:index+1].cpu(), mixed.v[:, index:index+1].cpu(), terminal_n),
                    "teacher": HSTUKVCache(teacher.k[:, index:index+1].cpu(), teacher.v[:, index:index+1].cpu(), terminal_n),
                    "candidates": torch.as_tensor(ids, dtype=torch.long),
                    "query_delta": float(cutover - timeline["terminal_last_timestamp"]),
                    "inherited_count": timeline["inherited_count"],
                    "actual_append": timeline["actual_append"]}
                state_records[str(uid)] = {key: timeline[key] for key in ("append_target", "actual_append",
                    "inherited_count", "pseudo_release_timestamp", "parent_last_timestamp", "terminal_last_timestamp")}
                state_records[str(uid)].update(parent_length=prefix_n, terminal_length=terminal_n,
                    rolling_backend=rolling_backend, teacher_backend=teacher_backend)
            del mixed, teacher
            print(json.dumps({"status": "mixed_cache_capture", "users": len(rows), "total": len(uids)}), flush=True)
    ledger["total_flops"] = sum(ledger.values())
    ledger["histograms"] = {name: dict(counts) for name, counts in hist.items()}
    ledger["convention"] = "analytical native FLOPs at actual executed batch and band shapes; each parent capture, rolling append and exact terminal teacher charged once"
    return rows, ledger, state_records


def load_mixed_data(args, config):
    """Frozen-loader-compatible bounded preparation for one mixed state per UID."""
    binding_path = args.panel_root / args.scale / edge_name(args.edge) / "binding.json"
    binding = json.loads(binding_path.read_text())
    reservation_path = args.reservations / args.scale / edge_name(args.edge) / "calibration_users.json"
    reservation = json.loads(reservation_path.read_text())
    if reservation["source_binding"]["sha256"] != sha256(binding_path):
        raise RuntimeError("calibration reservation panel binding changed")
    uids = reservation["uids"][:max(args.budgets)]
    if len(uids) < max(args.budgets):
        raise RuntimeError("requested budget exceeds independent reserved users")
    train_count = min(int(config["calibration_users"]), len(uids))
    assignments = assign_append_targets(uids[:train_count], uids[train_count:],
        targets=config["mixed_append_targets"], seed=config["seed"])
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free / total < config["initial_free_fraction"]:
        raise RuntimeError("GPU lacks initial free-memory reserve")
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"], device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(config["torch_threads"])
    pa.set_cpu_count(config["history_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ["EVOKV_ATTENTION_BACKEND"] = config["attention_backend"]
    started = time.perf_counter()
    parent, pp = load_model(ROOT / binding["sources"]["parent"]["path"], device)
    current, cp = load_model(ROOT / binding["sources"]["current"]["path"], device)
    if pp["config"] != cp["config"]:
        raise RuntimeError("parent/current architecture differs")
    parent.requires_grad_(False)
    current.requires_grad_(False)
    dataset_path = ROOT / binding["sources"]["dataset"]["path"]
    dataset = json.loads(dataset_path.read_text())
    known = int(cp.get("known_vocab_size", dataset["foundation_items"]))
    cutover = int(binding["cutover"])
    raw_limit = config["history_length"] + max(config["mixed_append_targets"])
    history = load_histories(uids, dataset_path=dataset_path, known_vocab_size=known,
        oov_buckets=int(cp["config"]["num_items"]) - known, start_timestamp=cutover,
        end_timestamp=cutover + 1, max_history=raw_limit, threads=config["history_threads"])
    timelines = {uid: prepare_timeline(history.rows[uid], cutover=cutover,
        history_length=config["history_length"], append_target=assignments[uid]) for uid in uids}
    del history, pp
    width = current.blocks[0].attn.num_heads * current.blocks[0].attn.head_dim
    cache_bytes = 16 * len(current.blocks) * width * sum(len(timelines[u]["terminal"][0]) for u in uids)
    if cache_bytes > 128 * (1 << 30):
        raise RuntimeError("CPU calibration caches exceed 128 GiB per process")
    rows, ledger, states = capture_mixed(parent, current, timelines, uids,
        cutover=cutover, known=known, queries=config["calibration_queries_per_user"],
        device=device, batch_size=config["capture_batches"][args.scale],
        history_length=config["history_length"], attention_backend=config["attention_backend"],
        append_band_size=config["append_band_size"], candidate_mode=config["candidate_mode"], seed=config["seed"])
    del parent, timelines
    gc.collect()
    metadata = {"panel_binding_sha256": sha256(binding_path),
        "calibration_users_sha256": sha256(reservation_path), "cutover": cutover,
        "model_config": cp["config"], "cache_bytes": cache_bytes, "shared_snapshot_users": len(uids),
        "cache_prepare_seconds": time.perf_counter() - started, "capture_flops": ledger["total_flops"],
        "capture_cost": ledger, "mixed_states": states, "raw_pre_history_limit": raw_limit,
        "checkpoint_hashes": {key: binding["sources"][key]["sha256"] for key in ("parent", "current")},
        "candidate_rule": ("uniform known catalog without replacement; SeedSequence([17,uid]); no feedback labels"
            if config["candidate_mode"] == "uniform_known" else
            "half most-recent unique known terminal-history items, half uniform known catalog; no feedback labels"),
        "cache_state": "one genuine Parent-prefix plus Current-rolling-append state per user, strictly before release; parent row key denotes mixed cache",
        "mix_assignment": "train and validation separately hash-sorted by seed and UID then balanced over append targets; complete timestamp groups; actual append rounded down",
        "teacher": "exact Current Full recomputation of aligned terminal window, first delta reset to zero; never rolling Current as teacher"}
    del cp
    return current, rows, uids, device, metadata
