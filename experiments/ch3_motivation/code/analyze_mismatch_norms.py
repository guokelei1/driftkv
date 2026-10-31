#!/usr/bin/env python3
"""Compare squared-error mass and row Frobenius-norm mass on the same E4 users.

The primary d_i is the sum of squared K/V differences over layers and heads.
sqrt(d_i) is a different diagnostic: the row's Frobenius difference norm.
It preserves the ranking of rows but changes their additive contributions.
Neither definition replaces the other or changes the frozen 5,000-user panel.
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
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np

from experiments.ch3_motivation.code.analyze_mismatch_sensitivity import load_blocks, metric_summary
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.common.artifacts import sha256


def norm_shares(difference):
    """Use one top-10% set for both d and sqrt(d), without changing the rows."""
    difference = np.asarray(difference, dtype=np.float64)
    if difference.ndim != 1 or not len(difference) or np.any(difference < 0):
        raise ValueError("nonempty nonnegative squared row masses required")
    count = int(math.ceil(.1 * len(difference)))
    selected = np.argpartition(difference, len(difference) - count)[len(difference) - count:]
    norm = np.sqrt(difference)
    squared_total, norm_total = difference.sum(), norm.sum()
    return dict(rows=len(difference), selected_rows=count,
        squared_total=float(squared_total), squared_top=float(difference[selected].sum()),
        norm_total=float(norm_total), norm_top=float(norm[selected].sum()),
        squared_top10_share=float(difference[selected].sum() / squared_total) if squared_total > 0 else None,
        norm_top10_share=float(norm[selected].sum() / norm_total) if norm_total > 0 else None)


def summarize_records(records):
    summary = dict(users=len(records), eligible=len(records) >= 512)
    if not summary["eligible"]:
        return summary
    summary["metrics"] = {}
    for kind in ("absolute", "relative"):
        for key in ("squared_top10_share", "norm_top10_share", "squared_minus_norm_share"):
            summary["metrics"][f"{kind}_{key}"] = metric_summary([row[f"{kind}_{key}"] for row in records])
        for mass in ("squared", "norm"):
            denominator = sum(row[f"{kind}_{mass}_total"] for row in records)
            numerator = sum(row[f"{kind}_{mass}_top"] for row in records)
            summary[f"{kind}_{mass}_pooled_within_user_top10_share"] = numerator / denominator if denominator else None
    return summary


def analyze_edge(directory, edge):
    config, original, values, block_hashes = load_blocks(directory)
    records = []
    for uid in config["identity"]["users"]:
        row = dict(edge=edge, uid=uid, history_rows=original[uid]["history_rows"])
        for kind in ("absolute", "relative"):
            metrics = norm_shares(values[uid][kind])
            row.update({f"{kind}_{key}": value for key, value in metrics.items() if key not in ("rows", "selected_rows")})
            a, b = metrics["squared_top10_share"], metrics["norm_top10_share"]
            row[f"{kind}_squared_minus_norm_share"] = a - b if a is not None and b is not None else None
        records.append(row)
    grouped = dict(all=records,
        history_under_256=[row for row in records if row["history_rows"] < 256],
        history_256_to_1023=[row for row in records if 256 <= row["history_rows"] < 1024],
        history_1024=[row for row in records if row["history_rows"] == 1024])
    summaries = {name: summarize_records(rows) for name, rows in grouped.items()}
    global_shares = {}
    for kind in ("absolute", "relative"):
        merged = np.concatenate([values[uid][kind] for uid in config["identity"]["users"]])
        global_shares[kind] = norm_shares(merged)
    return dict(edge=edge, users=len(records), groups=summaries,
        globally_merged_row_shares=global_shares,
        source_hashes=dict(configuration=sha256(directory / "configuration.json"),
                           primary_summary=sha256(directory / "summary.json"), blocks=block_hashes)), records


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows, edges = [], []
    for edge in args.edges:
        summary, rows = analyze_edge(args.run_root / f"edge_{edge}" / "attempt_1", edge)
        edges.append(summary)
        all_rows.extend(rows)
        metrics = summary["groups"]["all"]["metrics"]
        print(dict(edge=edge, users=summary["users"],
            squared_top10=metrics["absolute_squared_top10_share"]["median"],
            frobenius_top10=metrics["absolute_norm_top10_share"]["median"]), flush=True)
    with (args.output / "per_user.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["edge", "group", "metric", "users", "eligible", "q25", "median", "q75"])
        writer.writeheader()
        for edge in edges:
            for name, group in edge["groups"].items():
                for metric, values in group.get("metrics", {}).items():
                    writer.writerow(dict(edge=edge["edge"], group=name, metric=metric, **values))
    result = dict(status="complete", source_sha256=sha256(Path(__file__)), edges=edges,
        definitions=dict(primary_squared="d_i = sum of squared K/V differences over all layers/heads; additive squared-error contribution",
            diagnostic_frobenius="sqrt(d_i); row Frobenius difference norm, summed over rows for its own concentration ratio",
            relative_norm="sqrt of the previously defined per-row norm-relative mass, not the raw K/V Frobenius norm",
            selection="identical ceil(10%*actual_rows) highest-d rows for both definitions; square root is monotone",
            scope="same fixed primary 5000 users/edge, no GPU/model execution; alternative statistic does not replace the primary squared metric",
            minimum_users="subgroups with fewer than 512 users have counts only"))
    write_json(args.output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "experiments/ch3_motivation/runs/e4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edges", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    run(parser.parse_args())
