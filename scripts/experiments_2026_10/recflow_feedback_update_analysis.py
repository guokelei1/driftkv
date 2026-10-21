#!/usr/bin/env python3
"""CPU-only paired UID intervals for already frozen RecFlow feedback scores."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_rows(folder):
    pieces = []
    inputs = {}
    for path in sorted(folder.glob("rank_*.npz")):
        checksum = digest(path)
        assert checksum == json.loads(path.with_suffix(".json").read_text())["sha256"]
        inputs[str(path)] = checksum
        with np.load(path) as archive:
            pieces.append(dict(archive))
    rows = {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}
    order = np.argsort(rows["event_index"])
    return {key: value[order] for key, value in rows.items()}, inputs


def user_aucs(rows, uid_index, population_size):
    result = np.full(population_size, np.nan)
    order = np.argsort(uid_index, kind="stable")
    users, starts, counts = np.unique(uid_index[order], return_index=True, return_counts=True)
    for uid, start, count in zip(users, starts, counts, strict=True):
        selected = order[start:start + count]
        labels, scores = rows["label"][selected], rows["logit"][selected]
        positive_count = labels.sum()
        negative_count = count - positive_count
        if not positive_count or not negative_count:
            continue
        sorted_index = np.argsort(scores, kind="stable")
        scores, labels = scores[sorted_index], labels[sorted_index]
        ties = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
        positive = np.add.reduceat(labels.astype(np.float64), ties)
        negative = np.diff(np.r_[ties, count]) - positive
        below = np.cumsum(negative) - negative * 0.5
        result[uid] = (positive * below).sum() / (positive_count * negative_count)
    return result


def request_ndcg(rows, inverse, cutoff):
    order = np.lexsort((rows["event_index"], -rows["logit"], inverse))
    groups = inverse[order]
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    positives = np.bincount(inverse, weights=rows["label"]).astype(int)
    discounts = 1 / np.log2(np.arange(cutoff) + 2)
    ideal = np.r_[0, discounts.cumsum()][np.minimum(positives, cutoff)]
    gain = rows["label"][order] / np.log2(rank + 2) * (rank < cutoff)
    dcg = np.bincount(groups, weights=gain, minlength=len(positives))
    return np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, default=Path("experiments/foundation_recflow/runs/chain_2gpu/attempt_1/large"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.training_root / "update_analysis"
    population_archive = np.load("data/processed/recflow_v1/population.npz")
    population = np.sort(population_archive["uids"][population_archive["role"] == 2])
    rng = np.random.default_rng(20261021)
    draws = np.array([np.bincount(rng.integers(len(population), size=len(population)),
        minlength=len(population)) for _ in range(1000)], dtype=np.float64)
    edges, bootstrap_by_metric, sources = [], {}, {}
    for summary_path in sorted((args.training_root / "evaluation").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        parent, current = summary["parent"], summary["current"]
        before, inputs = load_rows(summary_path.parent / "scores" / parent)
        sources.update(inputs)
        after, inputs = load_rows(summary_path.parent / "scores" / current)
        sources.update(inputs)
        for key in ["event_index", "uid", "request_index", "label", "known"]:
            assert np.array_equal(before[key], after[key]), key
        uid_index = np.searchsorted(population, before["uid"])
        assert np.array_equal(population[uid_index], before["uid"])
        _, first, inverse = np.unique(before["request_index"], return_index=True, return_inverse=True)
        request_uid = uid_index[first]
        request_counts = np.bincount(request_uid, minlength=len(population))
        intervals = {}
        for cutoff in [1, 3, 5]:
            key = f"within_exposure_request_NDCG@{cutoff}"
            values = [request_ndcg(rows, inverse, cutoff) for rows in [before, after]]
            for version, scores in zip([parent, current], values, strict=True):
                assert abs(scores.mean() - summary["metrics"][version]["all"][key]) < 1e-12
            delta = values[1] - values[0]
            per_user = np.bincount(request_uid, weights=delta, minlength=len(population))
            sampled = draws @ per_user / (draws @ request_counts)
            intervals[key] = dict(difference=float(delta.mean()), uid_ci95=np.percentile(sampled, [2.5, 97.5]).tolist())
            bootstrap_by_metric.setdefault(key, []).append(sampled)
        auces = [user_aucs(rows, uid_index, len(population)) for rows in [before, after]]
        valid = np.isfinite(auces[0])
        assert np.array_equal(valid, np.isfinite(auces[1]))
        for version, scores in zip([parent, current], auces, strict=True):
            assert abs(np.nanmean(scores) - summary["metrics"][version]["all"]["user_macro_AUC"]) < 1e-12
        delta = np.nan_to_num(auces[1] - auces[0])
        sampled = draws @ delta / (draws @ valid.astype(float))
        intervals["user_macro_AUC"] = dict(difference=float(delta[valid].mean()),
            eligible_users=int(valid.sum()), uid_ci95=np.percentile(sampled, [2.5, 97.5]).tolist())
        bootstrap_by_metric.setdefault("user_macro_AUC", []).append(sampled)
        edges.append(dict(parent=parent, current=current, day=summary["day"],
            targets=len(uid_index), active_users=len(np.unique(uid_index)), metrics=intervals,
            original_differences=summary[f"{current}_minus_{parent}"],
            original_primary_auc_gain_ci95=summary["paired_uid_auc_gain_ci95"]))
    means = {key:dict(difference=float(np.mean([row["metrics"][key]["difference"] for row in edges])),
        uid_ci95=np.percentile(np.mean(values, axis=0), [2.5, 97.5]).tolist())
        for key, values in bootstrap_by_metric.items()}
    result = dict(status="complete_cpu_analysis_existing_scores", eligible_population=len(population),
        bootstrap_repetitions=1000, seed=20261021, edges=edges, equal_release_mean=means,
        score_sha256=sources, analysis_source_sha256=digest(Path(__file__)),
        scope="Post-hoc companion confidence intervals for already predeclared metrics, paired complete-population UID resampling shared across all edges. All-negative request NDCG=0. Macro-AUC includes only users with both labels. Conditional on fixed checkpoints/windows, not training-seed uncertainty. Original primary ROC-AUC and every adverse result retained. No new inference, training or GPU use.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(v4_v5=edges[-1], equal_release_mean=means), indent=2))


if __name__ == "__main__":
    main()
