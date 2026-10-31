"""Independently reconstruct metrics and full candidate membership from scores.

CPU-only audit of completed evaluations. It never runs a model, changes a score,
or opens a new population. Sorting uses two stable permutations, independently
of the evaluator's per-request lexsort implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .batches import RequestBatches
from .data_common import json_save, sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def independent_metrics(ids, labels, counts, scores):
    columns = np.arange(ids.shape[1])[None, :]
    keep = columns < counts[:, None]
    if not np.isfinite(scores[keep]).all():
        raise ValueError("Nonfinite saved model scores")
    id_order = np.argsort(np.where(keep, ids, np.iinfo(np.int64).max), axis=1, kind="stable")
    values = np.take_along_axis(np.where(keep, scores, -np.inf), id_order, axis=1)
    score_order = np.argsort(-values, axis=1, kind="stable")
    order = np.take_along_axis(id_order, score_order, axis=1)
    gains = np.take_along_axis((labels == 1) & keep, order, axis=1)
    positives = ((labels == 1) & keep).sum(axis=1)
    discounts = 1 / np.log2(np.arange(2, ids.shape[1] + 2, dtype=np.float64))
    ideal_prefix = np.r_[0., np.cumsum(discounts)]
    result = {}
    for k in [5, 10, 20]:
        dcg = (gains[:, :k] * discounts[None, :k]).sum(axis=1)
        idcg = ideal_prefix[np.minimum(positives, k)]
        result[f"ndcg@{k}"] = np.divide(dcg, idcg, out=np.full(len(counts), np.nan), where=idcg > 0)
        result[f"recall@{k}"] = np.divide(gains[:, :k].sum(axis=1), positives,
                                           out=np.full(len(counts), np.nan), where=positives > 0)
    return result


def audit(directory, data):
    summary = read_json(directory / "summary.json")
    args = summary["arguments"]
    if summary["status"] != "passed" or Path(args["data"]).resolve() != data.resolve():
        raise ValueError("Expected a completed evaluation on the specified data")
    if summary["role"] == 3 and not (data / "formal_recipe_freeze.json").exists():
        raise ValueError("Confirmation scores lack their required recipe freeze")
    dataset = RequestBatches(data, summary["days"], role=summary["role"], first_only=summary["first_only"],
                             limit=args["limit"], allow_confirmation=summary["role"] == 3,
                             history_limit=summary["history_limit"])
    expected = {(int(row["uid"]), int(row["day"]), int(row["rid"])): row for row in dataset.records}
    if len(expected) != len(dataset):
        raise ValueError("Evaluation population has duplicate request keys")
    rows, hashes = {}, {str(directory / "summary.json"): sha256(directory / "summary.json")}
    integer = {"uid", "rid", "day", "candidates", "exposed", "positives", "history_length"}
    csv_paths = sorted(directory.glob("rank_*.csv"))
    if len(csv_paths) != summary["world_size"]:
        raise ValueError("Missing distributed metric shard")
    for path in csv_paths:
        hashes[str(path)] = sha256(path)
        with path.open() as handle:
            for raw in csv.DictReader(handle):
                row = {key: (int(value) if key in integer else float(value)) if value else None for key, value in raw.items()}
                key = tuple(row[name] for name in ["uid", "day", "rid"])
                if key in rows:
                    raise ValueError("Duplicated metric request")
                rows[key] = row
    if rows.keys() != expected.keys() or len(rows) != summary["requests"]:
        raise ValueError("Metric shards do not cover the exact planned population")
    daily = {day: (np.load(data / dataset.stage_dir / f"d{day:02d}/features.npy", mmap_mode="r"),
                   np.load(data / dataset.stage_dir / f"d{day:02d}/labels.npy", mmap_mode="r")) for day in summary["days"]}
    methods = [name for name in summary["metrics"] if name not in ["random", "popularity"]]
    seen, sums = set(), {method: {f"{metric}@{k}": 0. for metric in ["ndcg", "recall"] for k in [5, 10, 20]}
                         for method in methods}
    max_error, candidate_total, positive_total, positive_requests = 0., 0, 0, 0
    paths = sorted(directory.glob("rank_*_scores/batch_*.npz"))
    for path in paths:
        hashes[str(path)] = sha256(path)
        with np.load(path) as stored:
            arrays = {key: stored[key] for key in stored.files}
        ids, labels, counts = arrays["raw_video_ids"], arrays["labels"], arrays["counts"]
        actual = {method: independent_metrics(ids, labels, counts, arrays[method]) for method in methods}
        for j in range(len(counts)):
            key = (int(arrays["uid"][j]), int(arrays["day"][j]), int(arrays["rid"][j]))
            if key in seen or key not in expected:
                raise ValueError("Saved scores duplicate or add a request outside the population")
            seen.add(key)
            req, row, count = expected[key], rows[key], int(counts[j])
            features, truth = daily[key[1]]
            start, stop = int(req["start"]), int(req["stop"])
            if (count != stop - start or row["candidates"] != count
                    or not np.array_equal(ids[j, :count], features[start:stop, 0])
                    or not np.array_equal(labels[j, :count], truth[start:stop])):
                raise ValueError("Saved candidates or labels differ from the entire actual logged request")
            if len(np.unique(ids[j, :count])) != count:
                raise ValueError("Saved request contains duplicate candidate items")
            positives, exposed = int((labels[j, :count] == 1).sum()), int((labels[j, :count] >= 0).sum())
            if (row["positives"] != positives or row["exposed"] != exposed
                    or row["history_length"] != min(int(req["history_length"]), summary["history_limit"])):
                raise ValueError("Feedback counts or actual history lengths differ")
            candidate_total += count
            positive_total += positives
            positive_requests += positives > 0
            for method in methods:
                for metric, values in actual[method].items():
                    expected_value = row[f"{method}_{metric}"]
                    if positives:
                        if expected_value is None:
                            raise ValueError("Positive request has missing saved metric")
                        error = abs(float(values[j]) - expected_value)
                        max_error = max(max_error, error)
                        if error > 1e-12:
                            raise ValueError("Saved request metric differs from independent recomputation")
                        sums[method][metric] += float(values[j])
                    elif expected_value is not None:
                        raise ValueError("Zero-positive request was treated as a defined NDCG/Recall")
    if seen != expected.keys() or positive_requests != summary["positive_requests"]:
        raise ValueError("Raw score shards omit requests or change the positive-request denominator")
    for method in methods:
        for metric, value in sums[method].items():
            actual = value / positive_requests if positive_requests else None
            expected_value = summary["metrics"][method][metric]
            if actual is None:
                if expected_value is not None:
                    raise ValueError("Incorrect zero-positive aggregate")
            elif abs(actual - expected_value) > 1e-12:
                raise ValueError("Summary is not the complete positive-request macro mean")
    return {"status": "passed", "evaluation": str(directory), "days": summary["days"], "role": summary["role"],
            "checkpoint_sha256": summary["checkpoint_sha256"], "parent_sha256": summary["parent_sha256"],
            "requests": len(seen), "positive_requests": positive_requests, "candidate_instances": candidate_total,
            "observed_positive_instances": positive_total, "saved_score_shards": len(paths), "methods": methods,
            "maximum_per_request_metric_error": max_error, "history_limit": summary["history_limit"],
            "candidate_membership": "exact full raw request item IDs/labels matched, not only counts",
            "zero_positive_requests_retained": True, "source_artifact_sha256": hashes, "audit_sha256": sha256(__file__)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("experiments/recflow_full_population/data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the existing audit; use a fresh output for a new one")
    result = audit(args.evaluation, args.data)
    json_save(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "source_artifact_sha256"}))


if __name__ == "__main__":
    main()
