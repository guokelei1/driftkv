#!/usr/bin/env python3
"""CPU explanation of E4 using the committed 5,000-user row masses.

No cache is recomputed and no feedback label is loaded. Position/length/item
statistics are descriptive diagnostics of the fixed population, not a search
for an outcome-favorable evaluation sample. Every reported subgroup metric
requires at least 512 users. K/V norms, separate K/V errors, and head errors
were not retained by E4 and are explicitly left for a new numerical probe.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from experiments.ch3_motivation.code.common import DATASET, write_json
from hstu_kvcache.evaluation.history import restore_pack
from hstu_kvcache.common.artifacts import sha256


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def metric_summary(values, *, minimum_users=512):
    values = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=np.float64)
    if len(values) < minimum_users:
        return dict(users=len(values), eligible=False, q25=None, median=None, q75=None)
    q25, median, q75 = np.quantile(values, [.25, .5, .75]).tolist()
    return dict(users=len(values), eligible=True, q25=q25, median=median, q75=q75)


def row_diagnostics(values, item_ids, *, extension_start):
    """One user's position/item diagnostics; caller enforces sample support."""
    values, item_ids = np.asarray(values), np.asarray(item_ids)
    if values.ndim != 1 or values.shape != item_ids.shape or not len(values):
        raise ValueError("row masses must align with a nonempty retained raw prefix")
    n = len(values)
    k = int(math.ceil(n * .1))
    top = np.argpartition(values, n - k)[n - k:]
    total = values.sum()
    deciles = np.minimum(np.arange(n) * 10 // n, 9)
    selected_deciles = np.bincount(deciles[top], minlength=10) / k
    mass_deciles = np.bincount(deciles, weights=values, minlength=10)
    top_items, top_counts = np.unique(item_ids[top], return_counts=True)
    all_items, all_counts = np.unique(item_ids, return_counts=True)
    extension = item_ids >= extension_start
    result = dict(top10_row_share=ratio(values[top].sum(), total),
        top10_position_mean=float(np.mean((top + .5) / n)),
        head10_mass_share=ratio(values[:k].sum(), total),
        tail10_mass_share=ratio(values[-k:].sum(), total),
        top10_in_head10_fraction=float(np.mean(top < k)),
        top10_in_tail10_fraction=float(np.mean(top >= n - k)),
        mass_weighted_position=ratio(np.dot(values, (np.arange(n) + .5) / n), total),
        top10_unique_item_fraction=float(len(top_items) / k),
        top10_dominant_item_row_fraction=float(top_counts.max() / k),
        history_unique_item_fraction=float(len(all_items) / n),
        history_dominant_item_row_fraction=float(all_counts.max() / n),
        extension_id_row_fraction=float(extension.mean()),
        extension_id_top10_row_fraction=float(extension[top].mean()),
        extension_id_mass_share=ratio(values[extension].sum(), total))
    for decile in range(10):
        result[f"position_decile_{decile + 1}_mass_share"] = ratio(mass_deciles[decile], total)
        result[f"position_decile_{decile + 1}_top10_fraction"] = float(selected_deciles[decile])
    return result, top


def load_blocks(directory):
    config = json.loads((directory / "configuration.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if summary["users"] < 5000 or not summary["paper_result_eligible"]:
        raise ValueError("sensitivity study requires a completed main E4 population")
    records, values, hashes = {}, {}, {}
    for path in sorted((directory / "blocks").glob("part_*.json")):
        marker = json.loads(path.read_text())
        if marker["identity"] != summary["identity_hash"]:
            raise ValueError("E4 block identity changed")
        for kind in ("parquet", "row_values"):
            file = path.parent / marker[kind]["path"]
            if sha256(file) != marker[kind]["sha256"]:
                raise ValueError("E4 committed block changed")
            hashes[str(file.relative_to(ROOT))] = marker[kind]["sha256"]
        current = pq.read_table(path.parent / marker["parquet"]["path"]).to_pylist()
        records.update({row["uid"]: row for row in current})
        with np.load(path.parent / marker["row_values"]["path"], allow_pickle=False) as arrays:
            # Load each archive member once rather than re-reading an entire
            # block for every user inside the archive.
            uids, offsets = arrays["uids"], arrays["offsets"]
            absolute, relative = arrays["absolute"], arrays["relative"]
            for index, uid in enumerate(uids):
                start, stop = offsets[index:index + 2]
                values[int(uid)] = dict(absolute=absolute[start:stop], relative=relative[start:stop])
    if set(records) != set(config["identity"]["users"]) or set(records) != set(values):
        raise ValueError("E4 blocks do not cover exactly the frozen population")
    return config, records, values, hashes


def aggregate_group(records):
    names = [name for name in records[0] if name not in ("uid", "history_rows")] if records else []
    return dict(users=len(records), eligible=len(records) >= 512,
        metrics={name: metric_summary([row[name] for row in records]) for name in names})


def item_concentration(item_ids, users, masses):
    """Vectorized global item totals, with individual-item support labeled."""
    order = np.argsort(item_ids, kind="stable")
    unique, starts = np.unique(item_ids[order], return_index=True)
    occurrences = np.diff(np.r_[starts, len(order)])
    aggregates = {name: np.add.reduceat(value[order], starts) for name, value in masses.items()}
    result, exported = {}, []
    for name, mass in aggregates.items():
        count = min(50, len(unique))
        top = np.argpartition(mass, len(mass) - count)[-count:]
        top = top[np.argsort(mass[top])[::-1]]
        total = mass.sum()
        result[name] = dict(unique_item_ids=len(unique),
            top1_item_mass_share=ratio(mass[top[:1]].sum(), total),
            top10_items_mass_share=ratio(mass[top[:10]].sum(), total),
            top50_items_mass_share=ratio(mass[top].sum(), total))
        for index in top:
            mask = item_ids == unique[index]
            support = len(np.unique(users[mask]))
            exported.append(dict(kind=name, item_idx=int(unique[index]), users=support,
                occurrences=int(occurrences[index]), deviation=float(mass[index]),
                pooled_mass_share=ratio(mass[index], total),
                role="measured_item_subgroup" if support >= 512 else "diagnostic_item_count_only"))
    return result, exported


def analyze_edge(run_directory, input_directory, edge):
    config, original, values, block_hashes = load_blocks(run_directory)
    inputs = json.loads((input_directory / "inputs.json").read_text())
    history_path = input_directory / inputs["history"]["path"]
    if sha256(history_path) != inputs["history"]["sha256"]:
        raise ValueError("frozen E4 history input changed")
    histories = restore_pack(history_path)
    extension_start = int(json.loads(DATASET.read_text())["foundation_items"])
    cutover = int(config["settings"]["release_day"]) * 86400
    cap = int(config["settings"]["history_length"])
    output_rows, item_arrays, user_arrays = [], [], []
    masses = dict(absolute=[], relative=[])
    max_layers = dict(absolute=Counter(), relative=Counter())
    for uid in config["identity"]["users"]:
        old = original[uid]
        events = histories.rows[uid]
        end = int(np.searchsorted(events[0], cutover, side="left"))
        ids = events[1][max(0, end - cap):end]
        if len(ids) != old["history_rows"]:
            raise ValueError("retained item IDs do not align with E4 row masses")
        row = dict(uid=uid, history_rows=len(ids))
        for name in ("absolute", "relative"):
            diagnostics, _ = row_diagnostics(values[uid][name], ids, extension_start=extension_start)
            row.update({f"{name}_{key}": value for key, value in diagnostics.items()})
            row[f"{name}_max_layer_evenness"] = old[f"{name}_max_layer_evenness"]
            for layer, value in enumerate(old[f"{name}_layer_evenness"]):
                row[f"{name}_layer_{layer + 1}_evenness"] = value
            max_layers[name][int(np.argmax(old[f"{name}_layer_deviation"])) + 1] += 1
            masses[name].append(values[uid][name])
        item_arrays.append(ids)
        user_arrays.append(np.full(len(ids), uid, dtype=np.int64))
        output_rows.append(row)
    groups = dict(all=output_rows,
        history_under_256=[r for r in output_rows if r["history_rows"] < 256],
        history_256_to_1023=[r for r in output_rows if 256 <= r["history_rows"] < 1024],
        history_1024=[r for r in output_rows if r["history_rows"] == 1024])
    grouped = {name: aggregate_group(rows) for name, rows in groups.items()}
    correlations = {}
    for kind in ("absolute", "relative"):
        correlations[kind] = float(spearmanr([r["history_rows"] for r in output_rows],
            [r[f"{kind}_top10_row_share"] for r in output_rows]).statistic)
    item_summary, item_rows = item_concentration(np.concatenate(item_arrays), np.concatenate(user_arrays),
        {name: np.concatenate(arrays) for name, arrays in masses.items()})
    result = dict(edge=edge, users=len(output_rows), history_rows=sum(r["history_rows"] for r in output_rows),
        groups=grouped, history_length_top10_concentration_spearman=correlations,
        max_deviation_layer_counts={name: dict(counts) for name, counts in max_layers.items()},
        item_concentration=item_summary,
        input_hashes=dict(configuration=sha256(run_directory / "configuration.json"),
                         e4_summary=sha256(run_directory / "summary.json"), histories=sha256(history_path),
                         blocks=block_hashes),
        limitations=["All row masses sum K/V and heads; separate K/V errors, head errors, and old/new norms require new model execution.",
            "Comparing user history lengths is observational; a 256/512/1024 window intervention must use the same users and re-encode each window.",
            "Extension-ID classification uses the frozen legacy ID range; it is not a raw-item novelty label.",
            "Any subgroup metric with fewer than 512 users is suppressed; small individual-item rows are diagnostic counts, not paper subgroup results."])
    return result, output_rows, item_rows


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries, all_rows, item_rows = [], [], []
    for edge in args.edges:
        result, rows, items = analyze_edge(args.run_root / f"edge_{edge}" / "attempt_1",
            args.input_root / f"edge_{edge}", edge)
        summaries.append(result)
        all_rows.extend(dict(edge=edge, **row) for row in rows)
        item_rows.extend(dict(edge=edge, **row) for row in items)
        print(json.dumps(dict(status="e4_sensitivity_edge_complete", edge=edge, users=result["users"],
            absolute_top10=result["groups"]["all"]["metrics"]["absolute_top10_row_share"]["median"],
            absolute_top10_in_tail=result["groups"]["all"]["metrics"]["absolute_top10_in_tail10_fraction"]["median"])), flush=True)
    with (output / "per_user.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (output / "item_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(item_rows[0]))
        writer.writeheader()
        writer.writerows(item_rows)
    fields = ["edge", "group", "metric", "users", "eligible", "q25", "median", "q75"]
    with (output / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in summaries:
            for group, data in result["groups"].items():
                for metric, statistics in data["metrics"].items():
                    writer.writerow(dict(edge=result["edge"], group=group, metric=metric, **statistics))
    summary = dict(status="complete", role="descriptive sensitivity of existing fixed 5000-user E4 measurements",
        source_sha256=sha256(Path(__file__)), edges=summaries,
        next_numerical_comparisons=[
            "Use the same fixed >=512 development users for history caps 256/512/1024; keep the model attention divisor and time-feature rules unchanged.",
            "For each window retain separate K/V mismatch, head/layer masses, and old/new squared norms before collapsing dimensions.",
            "Compare each row's norm-relative mismatch with layer-RMS normalization fitted only on pre-release fitting users; preserve the absolute primary analysis.",
            "Replicate on the pre-frozen independent 2000-user sensitivity population; do not choose users by concentration or recovery."])
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "experiments/ch3_motivation/runs/e4")
    parser.add_argument("--input-root", type=Path, default=ROOT / "experiments/ch3_motivation/inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edges", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    run(parser.parse_args())
