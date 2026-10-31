#!/usr/bin/env python3
"""Explain E4 concentration using prerelease timing, without selecting users.

This descriptive study keeps the existing 5000 users and squared row masses.
One hour defines a session gap; five minutes defines a short median gap.
Neither threshold is selected using mismatch or quality outcomes. No request
labels are read, no new cache inference is performed and no quality is inferred.
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
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
from scipy.stats import spearmanr

from experiments.ch3_motivation.code.analyze_mismatch_sensitivity import load_blocks, metric_summary
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.evaluation.history import restore_pack
from hstu_kvcache.common.artifacts import sha256


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def describe_user(times, items, behaviors, absolute, relative, cutover):
    n = len(times)
    if not n or any(len(v) != n for v in (items, behaviors, absolute, relative)):
        raise ValueError("prerelease raw rows and committed mismatch rows must align")
    if np.any(np.diff(times) < 0) or np.any(times >= cutover):
        raise ValueError("timing analysis must use ordered strictly prerelease rows")
    # Match raw_prefix/raw_tensors: the first retained row has delta zero.
    # Its predecessor may be outside the retained pack, so it is not a
    # measured short gap in the association statistics below.
    gaps = np.r_[0., np.diff(times).astype(float)]
    observed = gaps[1:]
    long = gaps >= 3600
    proportions = np.unique(behaviors, return_counts=True)[1] / n
    result = dict(history_rows=n, history_span_seconds=int(times[-1] - times[0]),
        cutover_gap_seconds=int(cutover - times[-1]),
        median_observed_gap_seconds=float(np.median(observed)) if len(observed) else None,
        long_gap_row_fraction=float(long.mean()),
        observed_zero_gap_fraction=float(np.mean(observed == 0)) if len(observed) else None,
        prerelease14_retained_behavior_count=int(np.sum(times >= cutover - 14 * 86400)),
        unique_item_fraction=float(len(np.unique(items)) / n),
        behavior_entropy=float(-np.sum(proportions * np.log(proportions))))
    for kind, values in (("absolute", absolute), ("relative", relative)):
        values = np.asarray(values, dtype=float)
        if np.any(values < 0) or not np.isfinite(values).all():
            raise ValueError("committed squared mismatch must be nonnegative and finite")
        total = float(values.sum())
        count = int(np.ceil(.1 * n))
        top = np.argpartition(values, n-count)[n-count:]
        result.update({f"{kind}_top10_row_share": float(values[top].sum() / total) if total else None,
            f"{kind}_long_gap_mass_share": float(values[long].sum() / total) if total else None,
            f"{kind}_long_gap_enrichment": float(values[long].sum() / total / long.mean())
                if total and long.any() else None,
            f"{kind}_top10_long_gap_fraction": float(np.mean(long[top])),
            f"{kind}_first_row_mass_share": float(values[0] / total) if total else None,
            f"{kind}_within_user_gap_spearman": correlation(observed, values[1:])})
    return result


def analyze_edge(run_root, input_root, edge):
    directory = run_root / f"edge_{edge}" / "attempt_1"
    config, original, values, block_hashes = load_blocks(directory)
    input_path = input_root / f"edge_{edge}" / "inputs.json"
    binding = json.loads(input_path.read_text())
    history_path = input_path.parent / binding["history"]["path"]
    if sha256(history_path) != binding["history"]["sha256"] or config["settings"] != binding["settings"]:
        raise ValueError("frozen E4 settings/history changed")
    histories = restore_pack(history_path)
    cutover = config["settings"]["release_day"] * 86400
    cap = config["settings"]["history_length"]
    rows = []
    for uid in config["identity"]["users"]:
        events = histories.rows[uid]
        end = int(np.searchsorted(events[0], cutover, side="left"))
        times, items, behaviors = (v[max(0, end-cap):end] for v in events)
        if len(times) != original[uid]["history_rows"]:
            raise ValueError("history snapshot length differs from committed E4")
        result = describe_user(times, items, behaviors, **values[uid], cutover=cutover)
        for kind in ("absolute", "relative"):
            expected = original[uid][f"{kind}_top10_row_share"]
            got = result[f"{kind}_top10_row_share"]
            if (got is None) != (expected is None) or got is not None and abs(got-expected) > 1e-12:
                raise ValueError("temporal analysis changed the primary squared concentration")
        rows.append(dict(edge=edge, uid=uid, **result))
    groups = {"all": rows,
        "history_ge256_median_gap_le300": [r for r in rows if r["history_rows"] >= 256
            and r["median_observed_gap_seconds"] <= 300],
        "history_ge256_median_gap_gt300": [r for r in rows if r["history_rows"] >= 256
            and r["median_observed_gap_seconds"] > 300]}
    metrics = [k for k in rows[0] if k not in ("edge", "uid")]
    feature_names = [k for k in metrics if not k.startswith(("absolute_", "relative_"))]
    summary = dict(edge=edge, users=len(rows), groups={name: dict(users=len(group),
        role="existing5000_main" if name == "all" else "prerelease_defined_descriptive_subgroup",
        statistics={key: metric_summary([r[key] for r in group]) for key in metrics})
        for name, group in groups.items()},
        user_level_spearman={kind: {key: correlation([r[key] for r in rows],
            [r[f"{kind}_top10_row_share"] for r in rows]) for key in feature_names}
            for kind in ("absolute", "relative")},
        input_sha256={str(input_path.resolve()): sha256(input_path),
            str(history_path.resolve()): binding["history"]["sha256"],
            str((directory/"configuration.json").resolve()): sha256(directory/"configuration.json"),
            str((directory/"summary.json").resolve()): sha256(directory/"summary.json"), **block_hashes})
    return summary, rows


def run(args):
    if (args.output / "summary.json").exists():
        raise ValueError("retain completed temporal analysis; use a fresh output for changed analysis")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name("analyze_mismatch_sensitivity.py"),
        ROOT / "src/hstu_kvcache/evaluation/history.py"]
    hashes = {str(p.resolve()): sha256(p) for p in sources}
    summaries, rows = [], []
    for edge in range(1, 6):
        summary, current = analyze_edge(args.run_root, args.input_root, edge)
        summaries.append(summary); rows.extend(current)
        print(json.dumps(dict(edge=edge, users=summary["users"], status="temporal_diagnosis_complete")), flush=True)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("analysis source changed during execution")
    with (args.output / "per_user.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = dict(status="complete", edges=summaries, source_sha256=hashes,
        thresholds=dict(session_gap_seconds=3600, short_median_gap_seconds=300, long_history_rows=256),
        limitations=["Descriptive input associations, not a causal intervention or evidence of AUC importance.",
            "No user is removed from the existing5000 main result; subgroups are supplemental and values below512 users are suppressed.",
            "The first retained delta is zero by encoding convention and excluded from observed-gap association statistics.",
            "Behavior count covers only retained prerelease rows, capped1024; it is not the full14-day activity count.",
            "No new model evaluation, population selection or future-feedback read; absolute and relative squared metrics stay separate."])
    write_json(args.output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "experiments/ch3_motivation/runs/e4")
    parser.add_argument("--input-root", type=Path, default=ROOT / "experiments/ch3_motivation/inputs")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
