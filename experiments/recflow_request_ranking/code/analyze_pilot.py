"""Retain negative pilot outcomes and quantify request-level cache-score changes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data_common import json_save, sha256
from .protocol import paired_uid_interval


def metrics(directory):
    rows = {}
    for path in sorted(directory.glob("rank_*.csv")):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                key = tuple(int(row[k]) for k in ["uid", "day", "rid"])
                if key in rows:
                    raise ValueError("Duplicate request in metrics")
                rows[key] = row
    return rows


def score_rows(directory):
    for path in sorted(directory.glob("rank_*_scores/batch_*.npz")):
        with np.load(path) as batch:
            # NpzFile does not cache decoded members: read each array once,
            # rather than decoding a complete batch again for every request.
            arrays = {name: batch[name] for name in batch.files}
        for index, count in enumerate(arrays["counts"]):
            key = tuple(int(arrays[k][index]) for k in ["uid", "day", "rid"])
            yield key, {name: arrays[name][index, :count].copy() for name in
                        ["raw_video_ids", "labels", "current", "parent", "reuse"] if name in arrays}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    results = {"scope": "small-model development; all signs retained; confirmation unopened", "edges": []}
    for train_day in [27, 28]:
        full = root/f"full_day_d{train_day+1}_history1024"
        first = root/f"edge_d{train_day}_eval_d{train_day+1}"
        first_scores = dict(score_rows(first))
        total_energy, centered_energy, requests, max_abs, numeric_error = 0., 0., 0, 0., 0.
        overlap, means, centered_rms = [], [], []
        matched = set()
        for key, values in score_rows(full):
            delta = values["current"].astype(np.float64)-values["reuse"].astype(np.float64)
            mean = delta.mean()
            total_energy += float(delta@delta)
            centered_energy += float((delta-mean)@(delta-mean))
            max_abs = max(max_abs, float(np.abs(delta).max()))
            means.append(abs(mean))
            centered_rms.append(float(np.sqrt(np.mean((delta-mean)**2))))
            top = [np.lexsort((values["raw_video_ids"], -values[name]))[:10] for name in ["current", "reuse"]]
            overlap.append(len(set(top[0]) & set(top[1]))/min(10, len(delta)))
            requests += 1
            if key in first_scores:
                reference = first_scores[key]
                for name in ["labels", "raw_video_ids"]:
                    if not np.array_equal(values[name], reference[name]):
                        raise ValueError("First/full candidate or label set changed")
                for name in ["current", "parent", "reuse"]:
                    numeric_error = max(numeric_error, float(np.abs(values[name]-reference[name]).max()))
                matched.add(key)
        if matched != set(first_scores):
            raise ValueError("Full window omitted an original first request")
        edge = {"train_day": train_day, "requests": requests, "centered_score_delta_energy_fraction": centered_energy/max(total_energy, 1e-30),
                "mean_abs_request_shift": float(np.mean(means)), "mean_centered_score_delta_rms": float(np.mean(centered_rms)),
                "mean_top10_overlap": float(np.mean(overlap)), "requests_with_changed_top10_fraction": float(np.mean(np.asarray(overlap) < 1)),
                "maximum_score_delta": max_abs, "matched_first_requests": len(matched), "first_vs_full_score_max_abs": numeric_error,
                "first_summary_sha256": sha256(first/"summary.json"), "full_summary_sha256": sha256(full/"summary.json")}
        results["edges"].append(edge)
    for label, long, short in [
        ("edge1_full_day", root/"full_day_d28_history1024", root/"full_day_d28_history128"),
        ("edge1_first", root/"edge_d27_eval_d28", root/"edge_d27_eval_d28_history128"),
        ("edge2_first", root/"edge_d28_eval_d29", root/"edge_d28_eval_d29_history128")]:
        left, right = metrics(long), metrics(short)
        if set(left) != set(right):
            raise ValueError("History diagnostic changed the request set")
        keys = [key for key, row in left.items() if int(row["positives"]) > 0]
        results[label+"_long_minus_short_ndcg10"] = paired_uid_interval(
            [key[0] for key in keys], [float(left[key]["current_ndcg@10"]) for key in keys],
            [float(right[key]["current_ndcg@10"]) for key in keys])
    results["quality_gate_passed"] = False
    results["conclusion"] = "Native full-day ranking exceeds baselines. Neither1024-history update has a positive paired reuse-gap interval;128-history positive finding is not a replacement. Do not claim the cache-loss premise established."
    json_save(root/"retained_findings.json", results)
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
