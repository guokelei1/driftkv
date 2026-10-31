#!/usr/bin/env python3
"""Additional E4 population/window/normalization sensitivity, never the primary.

The fixed 512 development users differ from E4's primary 5,000 users. All
windows use the original 1,024-window backbone and its unchanged attention
divisor; only the retained raw prefix changes. Publishing-time normalization
comes from 64 disjoint fitting users' rebuilt 1,024-window caches, then stays
fixed for the 256/512/1024 comparisons. No feedback labels are loaded.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from experiments.ch3_motivation.code.analysis import mismatch_rows
from experiments.ch3_motivation.code.common import load_models, quality_role, raw_prefix, write_json
from hstu_kvcache.evaluation.history import restore_pack
from hstu_kvcache.evaluation.trajectory import _short_prefix_backend
from hstu_kvcache.common.artifacts import sha256


@torch.no_grad()
def compute_cache(model, histories, uid, cfg, device, *, window):
    raw, events, end = raw_prefix(histories, uid, dict(cfg, history_length=window), device)
    if not raw[0].shape[1] or int(events[0][end - 1]) >= int(cfg["release_day"]) * 86400:
        raise ValueError("cache input must be nonempty and strictly pre-release")
    # Crucially the model's original max window stays 1024 even for a
    # sensitivity window of 256 or 512. Only backend compilation is avoided.
    with _short_prefix_backend(model, raw[0].shape[1], 1024):
        return model.compute_kv(*raw)


def head_values(tensor, heads):
    layers, batch, rows, width = tensor.shape
    if batch != 1 or width % heads:
        raise ValueError("this scalar reference expects one aligned user and whole heads")
    return tensor.detach().double().reshape(layers, rows, heads, width // heads).permute(0, 2, 1, 3)


@torch.no_grad()
def fit_rms_statistics(current, histories, uids, cfg, device):
    heads = current.cfg.num_heads
    sums = {name: torch.zeros(len(current.blocks), heads, dtype=torch.float64, device=device)
            for name in ("key", "value")}
    elements = 0
    for uid in uids:
        cache = compute_cache(current, histories, uid, cfg, device, window=1024)
        for name, tensor in (("key", cache.k), ("value", cache.v)):
            values = head_values(tensor, heads)
            sums[name] += values.square().sum((2, 3))
        elements += cache.seq_len * (cache.k.shape[-1] // heads)
        del cache
    if not elements:
        raise ValueError("normalization fitting requires nonempty pre-release rows")
    return dict(users=list(uids), fit_users=len(uids), teacher_window=1024,
        elements_per_layer_head=elements,
        key_rms=(sums["key"] / elements).sqrt().clamp_min(1e-6).cpu().tolist(),
        value_rms=(sums["value"] / elements).sqrt().clamp_min(1e-6).cpu().tolist(),
        squared_norm_floor=1e-12,
        definition="per-component rebuilt K or V RMS, separately per layer/head; fit64 only, fixed across all evaluation users/windows")


def _mass_metrics(key, value):
    """CPU arrays are [layers,heads,rows], already in the desired units."""
    total_rows = (key + value).sum((0, 1))
    layers, heads, rows = key.shape
    count = int(math.ceil(.1 * rows))
    selected = np.argpartition(total_rows, rows - count)[rows - count:]
    total = float(total_rows.sum())
    combined = (key + value).sum(2)
    layer_mass = combined.sum(1)
    key_mass, value_mass = key.sum(2), value.sum(2)
    def share(numerator):
        return float(numerator / total) if total > 0 else None
    evenness = layer_mass * layers / total if total > 0 else np.full(layers, np.nan)
    within_layer = np.divide(combined * heads, layer_mass[:, None],
        out=np.zeros_like(combined), where=layer_mass[:, None] > 0)
    return dict(total_deviation=total, top10_row_share=share(total_rows[selected].sum()),
        top10_rows_deviation=float(total_rows[selected].sum()),
        top10_position_mean=float(np.mean((selected + .5) / rows)),
        tail10_mass_share=share(total_rows[-count:].sum()),
        head10_mass_share=share(total_rows[:count].sum()),
        key_share=share(key.sum()), value_share=share(value.sum()),
        key_layer_head_mass=key_mass.tolist(), value_layer_head_mass=value_mass.tolist(),
        layer_mass=layer_mass.tolist(), layer_evenness=evenness.tolist(),
        max_layer_evenness=float(np.max(evenness)) if total > 0 else None,
        layer_head_mass=combined.tolist(), within_layer_head_evenness=within_layer.tolist())


@torch.no_grad()
def cache_diagnostics(old, new, *, uid, window, heads, fit_statistics):
    if old.k.shape != new.k.shape or old.v.shape != new.v.shape or old.seq_len != new.seq_len:
        raise ValueError("old/rebuilt caches are not aligned")
    key_old, key_new = head_values(old.k, heads), head_values(new.k, heads)
    value_old, value_new = head_values(old.v, heads), head_values(new.v, heads)
    arrays = dict(key_error=(key_new-key_old).square().sum(-1).cpu().numpy(),
        value_error=(value_new-value_old).square().sum(-1).cpu().numpy(),
        old_key_norm=key_old.square().sum(-1).cpu().numpy(),
        new_key_norm=key_new.square().sum(-1).cpu().numpy(),
        old_value_norm=value_old.square().sum(-1).cpu().numpy(),
        new_value_norm=value_new.square().sum(-1).cpu().numpy())
    joint_row_norm = (arrays["new_key_norm"] + arrays["new_value_norm"]).sum(1)
    key_scale = np.asarray(fit_statistics["key_rms"], dtype=np.float64)
    value_scale = np.asarray(fit_statistics["value_rms"], dtype=np.float64)
    variants = dict(absolute=(arrays["key_error"], arrays["value_error"]),
        per_row_relative=(arrays["key_error"] / np.maximum(joint_row_norm[:, None], 1e-12),
                          arrays["value_error"] / np.maximum(joint_row_norm[:, None], 1e-12)),
        fit_rms=(arrays["key_error"] / np.maximum(key_scale[..., None] ** 2, 1e-12),
                 arrays["value_error"] / np.maximum(value_scale[..., None] ** 2, 1e-12)))
    record = dict(uid=int(uid), window=int(window), history_rows=old.seq_len)
    for variant, (key, value) in variants.items():
        record.update({f"{variant}_{name}": number for name, number in _mass_metrics(key, value).items()})
    head_dim = old.k.shape[-1] // heads
    for name in ("old_key_norm", "new_key_norm", "old_value_norm", "new_value_norm"):
        record[f"{name}_rms"] = np.sqrt(arrays[name].mean(2) / head_dim).tolist()
    return record


def aggregate_records(records):
    """Quantiles across users; sub-512 groups expose counts only."""
    result = dict(users=len(records), eligible=len(records) >= 512)
    if not result["eligible"]:
        return result
    statistics = {}
    for name in records[0]:
        if name in ("uid", "window", "history_rows"):
            continue
        values = np.asarray([row[name] for row in records], dtype=np.float64)
        valid = np.isfinite(values).all(axis=tuple(range(1, values.ndim))) if values.ndim > 1 else np.isfinite(values)
        if valid.sum() < 512:
            statistics[name] = dict(users=int(valid.sum()), eligible=False)
            continue
        q25, median, q75 = np.quantile(values[valid], [.25, .5, .75], axis=0)
        statistics[name] = dict(users=int(valid.sum()), eligible=True, shape=list(values.shape[1:]),
            q25=q25.tolist(), median=median.tolist(), q75=q75.tolist())
    result["metrics"] = statistics
    return result


def export_points(results, path, edge):
    fields = ["edge", "window", "group", "metric", "layer", "head", "users", "q25", "median", "q75"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for window, groups in results.items():
            for group, summary in groups.items():
                for metric, value in summary.get("metrics", {}).items():
                    if not value["eligible"]:
                        continue
                    shape = value["shape"]
                    for index in np.ndindex(tuple(shape)) if shape else [()]:
                        def number(statistic):
                            return float(np.asarray(value[statistic])[index])
                        writer.writerow(dict(edge=edge, window=window, group=group, metric=metric,
                            layer=index[0] + 1 if index else None, head=index[1] + 1 if len(index) > 1 else None,
                            users=value["users"], q25=number("q25"), median=number("median"), q75=number("q75")))


def run(args):
    if args.edge not in range(1, 6) or any(window not in (256, 512, 1024) for window in args.windows):
        raise ValueError("expected one Large edge and natural windows 256/512/1024")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    directory = args.input_root / f"edge_{args.edge}"
    binding = json.loads((directory / "inputs.json").read_text())
    population_path = args.input_root / "population.json"
    population = json.loads(population_path.read_text())
    if sha256(population_path) != binding["population_sha256"]:
        raise ValueError("frozen population changed")
    history_path = directory / binding["history"]["path"]
    if sha256(history_path) != binding["history"]["sha256"]:
        raise ValueError("frozen raw histories changed")
    histories = restore_pack(history_path)
    uids = population["development"]
    diagnostic = args.diagnostic_users is not None
    if diagnostic:
        if not 1 <= args.diagnostic_users <= 16:
            raise ValueError("the performance/numerical diagnostic is bounded to 1..16 users")
        uids = uids[:args.diagnostic_users]
    elif len(uids) != 512:
        raise ValueError("additional sensitivity uses the fixed 512-user development population")
    role = quality_role(uids, diagnostic=diagnostic)
    fit_uids = population["fit"][:64]
    if set(fit_uids) & set(uids) or len(fit_uids) != 64:
        raise ValueError("64 fitting users must be disjoint from development users")
    if set(uids) & set(population["mismatch"]):
        raise ValueError("additional population must differ from primary E4 users")
    parent, current, models = load_models(args.edge, device)
    cfg = binding["settings"]
    files = [Path(__file__), Path(__file__).with_name("analysis.py"), Path(__file__).with_name("common.py"),
             ROOT / "src/hstu_kvcache/evaluation/baselines.py"]
    files.extend((ROOT / "src/hstu_kvcache/models").glob("*.py"))
    identity = dict(edge=args.edge, users=uids, fit_users=fit_uids, role=role, windows=args.windows,
        models=models, input_binding_sha256=sha256(directory / "inputs.json"),
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)},
        snapshot="strictly pre-release prefix; same user population for all windows",
        model_window=1024, attention_divisor_unchanged=True, statistics_window=1024,
        comparison_role="additional population/config sensitivity; does not replace primary 5000-user E4",
        normalizations=["absolute", "per_row_relative_combined_KV_heads", "fit64_teacher_layer_head_separate_KV_RMS"])
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError("attempt code/data/settings changed; use a fresh attempt")
    write_json(config_path, identity)
    scales_path = output / "fit_scales.json"
    if scales_path.exists():
        scales = json.loads(scales_path.read_text())
        if scales["identity"] != token:
            raise ValueError("fitting normalization identity changed")
    else:
        scales = fit_rms_statistics(current, histories, fit_uids, cfg, device)
        scales["identity"] = token
        write_json(scales_path, scales)
    results, numerical = {}, []
    started = time.perf_counter()
    for window in args.windows:
        records = []
        blocks = output / f"window_{window}" / "blocks"
        blocks.mkdir(parents=True, exist_ok=True)
        window_started = time.perf_counter()
        for offset in range(0, len(uids), 256):
            block_uids = uids[offset:offset + 256]
            marker_path = blocks / f"part_{offset:06d}.json"
            if marker_path.exists():
                marker = json.loads(marker_path.read_text())
                if marker["identity"] != token or marker["uids"] != block_uids:
                    raise ValueError("sensitivity block identity/users changed")
                if sha256(blocks / marker["parquet"]) != marker["sha256"]:
                    raise ValueError("sensitivity block changed")
                records.extend(pq.read_table(blocks / marker["parquet"]).to_pylist())
                continue
            part = []
            for uid in block_uids:
                old = compute_cache(parent, histories, uid, cfg, device, window=window)
                new = compute_cache(current, histories, uid, cfg, device, window=window)
                row = cache_diagnostics(old, new, uid=uid, window=window, heads=current.cfg.num_heads,
                                        fit_statistics=scales)
                if diagnostic:
                    reference = mismatch_rows(old, new, uids=[uid])[0]
                    error = max(abs(row["absolute_top10_row_share"] - reference["absolute_top10_row_share"]),
                        abs(row["per_row_relative_top10_row_share"] - reference["relative_top10_row_share"]))
                    if error > 1e-10:
                        raise RuntimeError("split-KV/head statistics differ from the audited primary E4 definition")
                    numerical.append(dict(uid=uid, window=window, maximum_primary_share_error=error))
                part.append(row)
                del old, new
            temporary = blocks / f"part_{offset:06d}.parquet.writing"
            final = blocks / f"part_{offset:06d}.parquet"
            pq.write_table(pa.Table.from_pylist(part), temporary)
            temporary.replace(final)
            write_json(marker_path, dict(identity=token, uids=block_uids, parquet=final.name, sha256=sha256(final)))
            records.extend(part)
            print(json.dumps(dict(status="e4_variants_block_committed", edge=args.edge, window=window,
                                  users=len(records), total=len(uids), seconds=time.perf_counter() - window_started)), flush=True)
        if diagnostic:
            continue
        results[str(window)] = dict(all=aggregate_records(records),
            history_under_256=aggregate_records([r for r in records if r["history_rows"] < 256]),
            history_256_to_1023=aggregate_records([r for r in records if 256 <= r["history_rows"] < 1024]),
            history_1024=aggregate_records([r for r in records if r["history_rows"] == 1024]))
    torch.cuda.synchronize(device)
    result = dict(status="complete", edge=args.edge, users=len(uids), role=role, identity=token,
        seconds=time.perf_counter() - started, peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        source_window=cfg.get("evaluation_window", dict(name="E14", complete_source_horizon=True)),
        paper_result_eligible=not diagnostic and len(uids) >= 512,
        interpretation="additional fixed-population/window/normalization sensitivity; numerical mismatch, not AUC importance")
    if diagnostic:
        result.update(numerical=numerical, maximum_primary_share_error=max(
            (row["maximum_primary_share_error"] for row in numerical), default=0.),
            scope="<=16-user correctness/performance diagnostic only; no distribution conclusion")
        write_json(output / "diagnostic.json", result)
    else:
        result["results"] = results
        export_points(results, output / "points.csv", args.edge)
        write_json(output / "summary.json", result)
    print(json.dumps(dict(status="complete", edge=args.edge, users=len(uids), role=role,
                          seconds=result["seconds"], output=str(output))), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT / "experiments/ch3_motivation/inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--diagnostic-users", type=int)
    parser.add_argument("--gpu", type=int, default=1)
    run(parser.parse_args())
