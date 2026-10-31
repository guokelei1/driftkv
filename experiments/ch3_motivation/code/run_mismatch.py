#!/usr/bin/env python3
"""Fresh E4 mismatch measurement at the pre-release history snapshot.

Only events strictly before cutover enter either cache. The two models encode
the same ordered raw prefix, including its original retained-window deltas.
Equal-length users are batched without padding. No feedback labels, teacher
scores, old measurement outputs, or outcome-selected users enter this study.

Small invocations are numerical/performance diagnostics, never paper quality
evidence. Main E4 uses all 5,000 previously frozen mismatch users per edge.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from experiments.ch3_motivation.code.analysis import MismatchAccumulator
from experiments.ch3_motivation.code.common import (
    load_edge, load_models, quality_role, raw_prefix, write_json,
)
from hstu_kvcache.evaluation.trajectory import _short_prefix_backend
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.common.artifacts import sha256


def raw_groups(histories, uids, cfg, batch_size):
    """Stage raw inputs on CPU, then make only three transfers per GPU batch."""
    grouped = defaultdict(list)
    for uid in uids:
        raw, _, _ = raw_prefix(histories, uid, cfg, torch.device("cpu"))
        if not raw[0].shape[1]:
            raise ValueError(f"user {uid} has no pre-release history")
        grouped[raw[0].shape[1]].append((uid, raw))
    for length in sorted(grouped):
        values = grouped[length]
        for offset in range(0, len(values), batch_size):
            group = values[offset:offset + batch_size]
            yield [row[0] for row in group], tuple(torch.cat([row[1][column] for row in group], dim=0)
                                                  for column in range(3))


@torch.no_grad()
def compute_pair(parent, current, raw, cfg, device):
    raw = tuple(value.to(device) for value in raw)
    length = raw[0].shape[1]
    with _short_prefix_backend(parent, length, cfg["history_length"]):
        old = parent.compute_kv(*raw)
    with _short_prefix_backend(current, length, cfg["history_length"]):
        rebuilt = current.compute_kv(*raw)
    return old, rebuilt


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def numerical_performance_check(parent, current, histories, uids, cfg, device, *, batch_size=16, repeats=2):
    """Compare scalar and batched K/V, then time warmed identical work.

    This is a <=16-user diagnostic, with no AUC or concentration conclusion.
    Both timings include host-to-device raw inputs and model execution, and
    exclude checkpoint loading and E4 summary computation.
    """
    if not 1 <= len(uids) <= 16:
        raise ValueError("the numerical/performance check uses one to sixteen users")
    groups = list(raw_groups(histories, uids, cfg, batch_size))
    scalar_groups = list(raw_groups(histories, uids, cfg, 1))
    # Warm both shapes before recording performance; keep scalar tensors only
    # for this bounded comparison, not for the 5,000-user measurement.
    references = {}
    for ids, raw in scalar_groups:
        references[ids[0]] = compute_pair(parent, current, raw, cfg, device)
    max_error = dict(old_k=0., old_v=0., rebuilt_k=0., rebuilt_v=0.)
    for ids, raw in groups:
        old, rebuilt = compute_pair(parent, current, raw, cfg, device)
        for index, uid in enumerate(ids):
            wanted_old, wanted_new = references[uid]
            for name, actual, wanted in (("old_k", old.k, wanted_old.k), ("old_v", old.v, wanted_old.v),
                                        ("rebuilt_k", rebuilt.k, wanted_new.k), ("rebuilt_v", rebuilt.v, wanted_new.v)):
                error = float((actual[:, index:index + 1] - wanted).abs().max())
                max_error[name] = max(max_error[name], error)
    del references
    timings = {}
    for name, work in (("scalar", scalar_groups), ("batched", groups)):
        measured = []
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter()
            for _, raw in work:
                pair = compute_pair(parent, current, raw, cfg, device)
                del pair
            _sync(device)
            measured.append(time.perf_counter() - started)
        timings[name] = measured
    maximum = max(max_error.values())
    return dict(role="numerical_performance_diagnostic", users=len(uids), batch_size=batch_size,
                maximum_kv_absolute_error=maximum, kv_absolute_error_by_tensor=max_error,
                required_maximum_absolute_error=1e-4, passed=maximum <= 1e-4,
                repeated_seconds=timings,
                scalar_median_seconds=float(np.median(timings["scalar"])),
                batched_median_seconds=float(np.median(timings["batched"])),
                batched_speedup=float(np.median(timings["scalar"]) / np.median(timings["batched"])),
                scope="warmed direct reconstruction plus raw-input transfers; not controlled serving timing or quality evidence")


def _commit_block(directory, block_index, accumulator, identity):
    stem = f"part_{block_index:06d}"
    records = directory / f"{stem}.parquet"
    temporary = directory / f"{stem}.parquet.writing"
    pq.write_table(pa.Table.from_pylist(accumulator.rows), temporary)
    temporary.replace(records)
    row_path = directory / f"{stem}.npz"
    row_temporary = directory / f"{stem}.npz.writing"
    offsets = np.r_[0, np.cumsum([r["history_rows"] for r in accumulator.rows])]
    with row_temporary.open("wb") as stream:
        np.savez(stream, offsets=offsets,
                 uids=np.asarray([r["uid"] for r in accumulator.rows], dtype=np.uint64),
                 absolute=np.concatenate([v["absolute"].numpy() for v in accumulator._row_values]),
                 relative=np.concatenate([v["relative"].numpy() for v in accumulator._row_values]))
    row_temporary.replace(row_path)
    marker = dict(identity=identity, uids=[r["uid"] for r in accumulator.rows],
                  parquet=dict(path=records.name, sha256=sha256(records)),
                  row_values=dict(path=row_path.name, sha256=sha256(row_path)))
    write_json(directory / f"{stem}.json", marker)
    return marker


def _restore_block(directory, block_index, uids, identity):
    marker_path = directory / f"part_{block_index:06d}.json"
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text())
    if marker["identity"] != identity or set(marker["uids"]) != set(uids):
        raise ValueError("committed E4 block identity/users changed")
    for kind in ("parquet", "row_values"):
        if sha256(directory / marker[kind]["path"]) != marker[kind]["sha256"]:
            raise ValueError("committed E4 block hash changed")
    records = pq.read_table(directory / marker["parquet"]["path"]).to_pylist()
    accumulator = MismatchAccumulator(rows=records)
    with np.load(directory / marker["row_values"]["path"], allow_pickle=False) as values:
        if values["uids"].tolist() != marker["uids"]:
            raise ValueError("E4 row-value UID order changed")
        for start, stop in zip(values["offsets"][:-1], values["offsets"][1:], strict=True):
            accumulator._row_values.append({name: torch.from_numpy(values[name][int(start):int(stop)].copy())
                                            for name in ("absolute", "relative")})
    return accumulator


def _export_summary(summary, output, edge):
    fields = ["edge", "group", "metric", "statistic", "layer", "value", "users", "history_rows"]
    exported = []
    for group in ("all", "history_at_least_256"):
        data = summary[group]
        if not data["users"]:
            continue
        for name in ("absolute", "relative"):
            for metric, value in data[name].items():
                if isinstance(value, dict):
                    for statistic, number in value.items():
                        exported.append([edge, group, f"{name}_{metric}", statistic, None, number, data["users"], data["history_rows"]])
                elif isinstance(value, list):
                    for layer, number in enumerate(value):
                        if isinstance(number, dict):
                            for statistic, part in number.items():
                                exported.append([edge, group, f"{name}_{metric}", statistic, layer + 1, part, data["users"], data["history_rows"]])
                        else:
                            exported.append([edge, group, f"{name}_{metric}", "pooled", layer + 1, number, data["users"], data["history_rows"]])
                else:
                    exported.append([edge, group, f"{name}_{metric}", "pooled", None, value, data["users"], data["history_rows"]])
    with (output / "points.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(exported)


def run(args):
    if args.edge not in range(1, 6) or args.batch_size < 1:
        raise ValueError("expected edge 1..5 and a positive batch size")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    directory = args.input_root / f"edge_{args.edge}"
    input_path = directory / "inputs.json"
    population_path = args.input_root / "population.json"
    population = json.loads(population_path.read_text())
    diagnostic = args.diagnostic_users is not None
    uids = population["mismatch"]
    if diagnostic:
        if not 1 <= args.diagnostic_users <= len(uids):
            raise ValueError("diagnostic user count outside the frozen E4 population")
        uids = uids[:args.diagnostic_users]
    elif len(uids) < 5000:
        raise ValueError("main E4 requires at least 5,000 frozen users per edge")
    role = quality_role(uids, diagnostic=diagnostic)
    input_binding = json.loads(input_path.read_text())
    if input_binding["population_sha256"] != sha256(population_path):
        raise ValueError("input population changed")
    for name in ("history", "requests"):
        if sha256(directory / input_binding[name]["path"]) != input_binding[name]["sha256"]:
            raise ValueError(f"frozen {name} input changed")
    binding, histories, _ = load_edge(directory)
    cfg = dict(binding["settings"], attention_backend=args.backend)
    parent, current, checkpoints = load_models(args.edge, device, backend=args.backend)
    sources = [Path(__file__), Path(__file__).with_name("analysis.py"), Path(__file__).with_name("common.py")]
    sources += list((ROOT / "src/hstu_kvcache/models").glob("*.py"))
    identity = dict(edge=args.edge, users=uids, role=role, batch_size=args.batch_size,
                    attention_backend=args.backend,
                    input_binding_sha256=sha256(input_path), population_sha256=sha256(population_path),
                    checkpoints=checkpoints, source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted(sources)},
                    snapshot="immediately before release; identical retained raw inputs encoded independently by old/new model",
                    relative_floor=1e-12, top_fraction=.1)
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configuration = output / "configuration.json"
    if configuration.exists():
        if json.loads(configuration.read_text())["identity"] != identity:
            raise ValueError("existing attempt settings or numerical sources changed")
    else:
        write_json(configuration, dict(identity=identity, machine=socket.gethostname(), physical_gpu=args.gpu,
                                        command=sys.argv, torch_version=torch.__version__, settings=cfg))
    numerical_path = output / "numerical_performance.json"
    if numerical_path.exists():
        numerical = json.loads(numerical_path.read_text())
        if numerical.get("identity_hash") != identity_hash:
            raise ValueError("numerical check identity changed")
    else:
        numerical = numerical_performance_check(parent, current, histories, uids[:16], cfg, device,
                                                  batch_size=args.batch_size)
        numerical["identity_hash"] = identity_hash
        write_json(numerical_path, numerical)
    print(json.dumps(dict(status="numerical_performance_checked", **numerical)), flush=True)
    if not numerical["passed"]:
        raise RuntimeError("batched/scalar K/V differs by more than 1e-4; stop before measurement")
    blocks = output / "blocks"
    blocks.mkdir(exist_ok=True)
    accumulator = MismatchAccumulator()
    started = time.perf_counter()
    measured_users = 0
    for offset in range(0, len(uids), 256):
        block_uids = uids[offset:offset + 256]
        part = _restore_block(blocks, offset, block_uids, identity_hash)
        resumed = part is not None
        if part is None:
            part = MismatchAccumulator()
            for ids, raw in raw_groups(histories, block_uids, cfg, args.batch_size):
                old, rebuilt = compute_pair(parent, current, raw, cfg, device)
                part.update(old, rebuilt, uids=ids)
                del old, rebuilt
            _commit_block(blocks, offset, part, identity_hash)
            measured_users += len(block_uids)
        accumulator.rows.extend(part.rows)
        accumulator._row_values.extend(part._row_values)
        write_json(output / "progress.json", dict(status="e4_mismatch", role=role, edge=args.edge,
                    users=len(accumulator.rows), total=len(uids), last_block_resumed=resumed,
                    execution_seconds=time.perf_counter() - started))
        print(json.dumps(dict(status="e4_block_committed", edge=args.edge, users=len(accumulator.rows),
                              total=len(uids), resumed=resumed, seconds=time.perf_counter() - started)), flush=True)
    _sync(device)
    result = dict(status="complete", edge=args.edge, role=role, users=len(uids),
                  configuration_sha256=sha256(configuration), identity_hash=identity_hash,
                  source_window=cfg.get("evaluation_window", dict(name="E14", complete_source_horizon=True)),
                  performance=numerical, execution_seconds=time.perf_counter() - started,
                  newly_measured_users=measured_users,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  interpretation="snapshot K/V mismatch distribution; not AUC importance or serving throughput")
    if diagnostic:
        result["paper_result_eligible"] = False
        result["diagnostic_scope"] = "numerical/performance only; no paper quality or distribution conclusion"
        write_json(output / "diagnostic.json", result)
    else:
        result["results"] = accumulator.summary()
        result["paper_result_eligible"] = len(uids) >= 5000
        _export_summary(result["results"], output, args.edge)
        write_json(output / "summary.json", result)
    print(json.dumps(dict(status="complete", edge=args.edge, role=role, users=len(uids),
                          output=str(output), seconds=result["execution_seconds"])), flush=True)
    del parent, current
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=int, required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT / "experiments/ch3_motivation/inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--diagnostic-users", type=int)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--backend", choices=("torch", "triton"), default="triton")
    run(parser.parse_args())
