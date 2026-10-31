#!/usr/bin/env python3
"""Exploratory ROC operating points and request AUC from frozen scores, CPU only."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from experiments.archive.baseline_studies.recflow_feedback_update_analysis import digest, load_rows, user_aucs


def roc_tprs(rows, uid_index, draws, fprs):
    """Resample whole users; group equal logits before interpolating the ROC."""
    order = np.argsort(-rows["logit"], kind="stable")
    score = rows["logit"][order]
    labels = rows["label"][order].astype(float)
    users = uid_index[order]
    ends = np.flatnonzero(np.r_[score[1:] != score[:-1], True])
    result = np.empty((len(draws), len(fprs)))
    for start in range(0, len(draws), 16):
        weights = draws[start:start + 16, users]
        tp = np.cumsum(weights * labels, axis=1)[:, ends]
        fp = np.cumsum(weights * (1 - labels), axis=1)[:, ends]
        tp = np.column_stack((np.zeros(len(weights)), tp))
        fp = np.column_stack((np.zeros(len(weights)), fp))
        for offset in range(len(weights)):
            assert tp[offset, -1] > 0 and fp[offset, -1] > 0
            result[start + offset] = np.interp(
                np.asarray(fprs) * fp[offset, -1], fp[offset], tp[offset]
            ) / tp[offset, -1]
    return result


def interval(before, after, sampled_before, sampled_after, random):
    relative = 100 * (sampled_after / sampled_before - 1)
    return dict(parent=float(before), current=float(after),
        difference=float(after - before),
        relative_improvement_percent=float(100 * (after / before - 1)),
        random=float(random), current_over_random_multiple=float(after / random),
        paired_uid_difference_ci95=np.percentile(sampled_after - sampled_before, [2.5, 97.5]).tolist(),
        paired_uid_relative_improvement_percent_ci95=np.percentile(relative, [2.5, 97.5]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "experiments/foundation_recflow/runs/chain_2gpu/attempt_1/large"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load("data/processed/recflow_v1/population.npz") as archive:
        population = np.sort(archive["uids"][archive["role"] == 2])
    seed, repetitions = 20261021, 1000
    rng = np.random.default_rng(seed)
    draws = np.asarray([np.bincount(rng.integers(len(population), size=len(population)),
        minlength=len(population)) for _ in range(repetitions)], dtype=np.float64)
    fprs = [.01, .05, .10]
    edges, inputs, samples = [], {}, {}
    for path in sorted((args.root / "evaluation").glob("*/summary.json")):
        summary = json.loads(path.read_text())
        parent, current = summary["parent"], summary["current"]
        before, hashes = load_rows(path.parent / "scores" / parent)
        inputs.update(hashes)
        after, hashes = load_rows(path.parent / "scores" / current)
        inputs.update(hashes)
        for key in ["event_index", "uid", "request_index", "label", "known"]:
            assert np.array_equal(before[key], after[key]), key
        uid = np.searchsorted(population, before["uid"])
        assert np.array_equal(population[uid], before["uid"])
        _, first, inverse = np.unique(before["request_index"], return_index=True, return_inverse=True)
        request_uid = uid[first]
        curves, sampled_curves = [], []
        for rows in [before, after]:
            x, y, _ = roc_curve(rows["label"], rows["logit"])
            point = np.interp(fprs, x, y)
            check = roc_tprs(rows, uid, np.ones((1, len(population))), fprs)[0]
            assert np.max(np.abs(point - check)) < 1e-12
            curves.append(point)
            sampled_curves.append(roc_tprs(rows, uid, draws, fprs))
        metrics = {}
        for column, fpr in enumerate(fprs):
            key = f"TPR_at_FPR_{fpr:g}"
            old, new = curves[0][column], curves[1][column]
            a, b = sampled_curves[0][:, column], sampled_curves[1][:, column]
            metrics[key] = interval(old, new, a, b, fpr)
            samples.setdefault(key, []).append((a, b))
        request_auc = [user_aucs(rows, inverse, len(first)) for rows in [before, after]]
        valid = np.isfinite(request_auc[0])
        assert np.array_equal(valid, np.isfinite(request_auc[1]))
        counts = np.bincount(request_uid[valid], minlength=len(population))
        totals = [np.bincount(request_uid[valid], weights=v[valid], minlength=len(population))
            for v in request_auc]
        for user_mean in [False, True]:
            key = "user_mean_request_AUC" if user_mean else "request_AUC"
            weights = (counts > 0).astype(float) if user_mean else counts
            values = [np.divide(v, counts, out=np.zeros_like(v), where=counts > 0)
                if user_mean else v for v in totals]
            points = [v.sum() / weights.sum() for v in values]
            sampled = [(draws @ v) / (draws @ weights) for v in values]
            metrics[key] = interval(*points, *sampled, .5)
            samples.setdefault(key, []).append(tuple(sampled))
        edge = dict(parent=parent, current=current, day=summary["day"],
            targets=len(uid), requests=len(first), mixed_label_requests=int(valid.sum()),
            mixed_label_active_users=int((counts > 0).sum()), metrics=metrics)
        edges.append(edge)
        print(json.dumps(edge), flush=True)
    means = {}
    for key, values in samples.items():
        differences = np.mean([b - a for a, b in values], axis=0)
        relative = np.mean([100 * (b / a - 1) for a, b in values], axis=0)
        means[key] = dict(
            difference=float(np.mean([e["metrics"][key]["difference"] for e in edges])),
            mean_relative_improvement_percent=float(np.mean([
                e["metrics"][key]["relative_improvement_percent"] for e in edges])),
            paired_uid_difference_ci95=np.percentile(differences, [2.5, 97.5]).tolist(),
            paired_uid_mean_relative_improvement_percent_ci95=np.percentile(relative, [2.5, 97.5]).tolist())
    result = dict(status="complete_exploratory_existing_score_analysis", edges=edges,
        equal_release_mean=means, eligible_population=len(population), seed=seed,
        bootstrap_repetitions=repetitions, fprs=fprs, score_sha256=inputs,
        source_sha256={str(Path(__file__)): digest(Path(__file__)),
            "experiments/archive/baseline_studies/recflow_feedback_update_analysis.py":
            digest(Path(__file__).with_name("recflow_feedback_update_analysis.py"))},
        scope="Post-hoc metric exploration on frozen checkpoints and all eligible test exposures. "
        "The 1/5/10% FPR grid is retained in full, not a fine threshold search. "
        "ROC points use test-curve interpolation and do not specify a prefit serving threshold. "
        "Request AUC is defined only for requests with both observed labels; ties get half credit. "
        "User mean averages each active mixed-request user's request AUC first, then users. "
        "Random TPR=FPR; random request AUC=0.5. "
        "Shared paired UID draws include all 3210 eligible users, including inactive ones. "
        "Pointwise percentile intervals are unadjusted for metric selection/multiple comparisons "
        "and condition on this single trained chain. Do not replace the predeclared global AUC "
        "or claim independent confirmation. No new inference, training or GPU use.")
    output = args.output or args.root / "operating_point_analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(equal_release_mean=means, output=str(output)), indent=2), flush=True)


if __name__ == "__main__":
    main()
