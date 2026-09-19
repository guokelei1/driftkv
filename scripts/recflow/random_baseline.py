#!/usr/bin/env python3
"""Matched random-policy expectations and fixed-panel Monte Carlo diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hstu_kvcache.recflow.metrics import random_expected_metrics  # noqa: E402

KS = (20, 50, 100)
METRICS = tuple(f"{metric}@{k}" for metric in ("ndcg", "recall") for k in KS)


def draw_random_panel(y, m, n, draws, seed):
    """Each trial independently permutes each request's available video pool.

    Only the m distinct positive ranks are needed. Rejection of duplicate-rank
    rows makes their distribution exactly uniform without replacement; every
    remaining permutation of the irrelevant items gives the same metrics.
    """
    rng = np.random.default_rng(seed)
    values = {key: np.zeros(draws, dtype=np.float64) for key in METRICS}
    analytic = dict.fromkeys(METRICS, 0.0)
    valid_count = int((y > 0).sum())
    if not valid_count:
        return {key: np.full(draws, np.nan) for key in METRICS}, dict.fromkeys(METRICS)
    groups, counts = np.unique(np.column_stack([y, m, n]), axis=0, return_counts=True)
    for (positives, known, pool), requests in zip(groups, counts, strict=True):
        positives, known, pool, requests = map(int, (positives, known, pool, requests))
        if not positives:
            continue
        if not (0 <= known <= min(positives, pool)):
            raise ValueError("Known positives must fit both the target set and candidate pool")
        expected = random_expected_metrics(positives, known, pool, ks=KS)
        for key in METRICS:
            analytic[key] += requests * expected[key] / valid_count
        if not known:
            continue
        ranks = rng.integers(1, pool + 1, size=(draws * requests, known))
        if known > 1:
            duplicate = np.any(np.diff(np.sort(ranks, axis=1), axis=1) == 0, axis=1)
            while duplicate.any():
                ranks[duplicate] = rng.integers(1, pool + 1, size=(int(duplicate.sum()), known))
                duplicate = np.any(np.diff(np.sort(ranks, axis=1), axis=1) == 0, axis=1)
        ranks = ranks.reshape(draws, requests, known)
        discounts = 1.0 / np.log2(ranks + 1)
        for k in KS:
            hits = ranks <= k
            values[f"recall@{k}"] += hits.sum(axis=(1, 2)) / positives / valid_count
            ideal = (1.0 / np.log2(np.arange(2, min(k, positives) + 2))).sum()
            values[f"ndcg@{k}"] += (discounts * hits).sum(axis=(1, 2)) / ideal / valid_count
    return values, analytic


def compare_metric(trials, expectation, trained):
    if expectation is None or trained is None:
        return {"trained": trained, "analytic_expectation": expectation, "defined": False}
    p025, p975, p99 = np.quantile(trials, [0.025, 0.975, 0.99])
    mean = float(trials.mean())
    mean_se = float(trials.std(ddof=1) / np.sqrt(len(trials)))
    ratio = trained / expectation if expectation > 0 else None
    above99 = bool(trained > p99)
    twice = bool(expectation > 0 and trained >= 2 * expectation)
    return {
        "trained": trained, "analytic_expectation": expectation,
        "null_mean": mean, "null_p2.5": float(p025), "null_p97.5": float(p975), "null_p99": float(p99),
        "monte_carlo_mean_standard_error": mean_se,
        "monte_carlo_mean_minus_analytic": mean - expectation,
        "trained_over_expected": ratio, "trained_exceeds_null_p99": above99,
        "trained_at_least_2x_expected": twice, "both_conditions": above99 and twice,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        raise SystemExit("Existing random-baseline evidence found; use a new output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cache, results = {}, []
    for directory in args.runs:
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        configuration = summary["configuration"]
        catalog_size = int(configuration["catalog_items"])
        for phase, evaluation in summary.items():
            if not isinstance(evaluation, dict) or "full_catalog" not in evaluation:
                continue
            panel_path = directory / f"{phase}_request_metrics.npz"
            panel = np.load(panel_path)
            y = panel["positives"].astype(np.int64)
            m = panel["known_positives"].astype(np.int64)
            if len(y) != evaluation["full_catalog"]["requests"]:
                raise ValueError("Saved panel and evaluation summary request counts differ")
            comparisons = {"full_catalog": evaluation["full_catalog"]}
            sampled = evaluation.get("sampled_candidate_diagnostics", {})
            if sampled and evaluation.get("sampled_tie_break") != "raw_item_id ascending":
                raise ValueError(f"Unverified sampled-score tie handling in {directory}/{phase}; exclude old positive-priority results")
            comparisons.update(sampled)
            sampled_path = directory / f"{phase}_sampled_request_metrics.npz"
            if sampled and sampled_path.exists():
                sampled_panel = np.load(sampled_path)
            else:
                # Historical evaluators used the entire free-retrieval panel
                # for sampled diagnostics and did not write a separate NPZ.
                sampled_panel = panel
                if sampled and evaluation.get("sampled_request_panel", {}).get("requests", len(y)) != len(y):
                    raise ValueError("The declared sampled subpanel is missing its request-metrics evidence")
            outcome = {
                "run": str(directory), "phase": phase, "training_seed": configuration["seed"],
                "requests": len(y), "positive_requests": int((y > 0).sum()),
                "zero_positive_requests": int((y == 0).sum()), "all_oov_positive_requests": int(((y > 0) & (m == 0)).sum()),
                "positive_targets": int(y.sum()), "known_positive_targets": int(m.sum()),
                "catalog_items": catalog_size, "retrieval_scope": evaluation.get("retrieval_scope"),
                "evaluation_precision": evaluation.get("evaluation_precision"),
                "input_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "input_panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                "comparisons": {},
            }
            for name, trained in comparisons.items():
                metric_panel = panel if name == "full_catalog" else sampled_panel
                metric_path = panel_path if name == "full_catalog" or sampled_panel is panel else sampled_path
                metric_y = metric_panel["positives"].astype(np.int64)
                metric_m = metric_panel["known_positives"].astype(np.int64)
                if len(metric_y) != trained["requests"]:
                    raise ValueError(f"Saved {name} panel and evaluation summary request counts differ")
                pool_key = f"{name}__candidate_count"
                if name == "full_catalog":
                    n = np.full(len(metric_y), catalog_size, dtype=np.int64)
                elif pool_key in metric_panel:
                    n = metric_panel[pool_key].astype(np.int64)
                else:
                    n = np.minimum(catalog_size, metric_m + int(name.rsplit("_", 1)[1]))
                signature = hashlib.sha256(np.column_stack([metric_y, metric_m, n]).tobytes() + metric_panel["indices"].tobytes()).hexdigest()
                if signature not in cache:
                    derived_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{signature}".encode()).digest()[:8], "little")
                    trials, expected = draw_random_panel(metric_y, metric_m, n, args.draws, derived_seed)
                    trial_file = f"null_trials_{signature[:16]}.npz"
                    np.savez_compressed(args.output / trial_file, **trials)
                    cache[signature] = (trials, expected, trial_file)
                trials, expected, trial_file = cache[signature]
                outcome["comparisons"][name] = {
                    "requests": len(metric_y), "positive_requests": int((metric_y > 0).sum()),
                    "zero_positive_requests": int((metric_y == 0).sum()),
                    "all_oov_positive_requests": int(((metric_y > 0) & (metric_m == 0)).sum()),
                    "positive_targets": int(metric_y.sum()), "known_positive_targets": int(metric_m.sum()),
                    "input_panel_sha256": hashlib.sha256(metric_path.read_bytes()).hexdigest(),
                    "request_indices_sha256": hashlib.sha256(metric_panel["indices"].tobytes()).hexdigest(),
                    "candidate_count_min": int(n.min()), "candidate_count_max": int(n.max()),
                    "null_trial_file": trial_file,
                    "metrics": {key: compare_metric(trials[key], expected[key], trained[key]) for key in METRICS},
                }
            results.append(outcome)
            print(json.dumps({"run": directory.name, "phase": phase, "seconds": time.monotonic()-started}), flush=True)
    report = {
        "scope": "CPU random-policy diagnostic on retained development panels; no training or final-user evaluation.",
        "draws": args.draws, "seed": args.seed, "ks": KS,
        "null_definition": "Uniform video permutations: full catalog N, or actual injected-positive sampled pool N=min(catalog_items,known_positives+distractors). Uniform/popularity distractor pools share the same random-ranking null conditional on known-positive count and size, though trained model difficulty differs.",
        "denominators": "Total observed distinct positives y, including OOV, define Recall and ideal DCG. All-OOV positive requests score zero; zero-label requests are excluded from macro metric means exactly as in the retained evaluator.",
        "interpretation": "Monte Carlo intervals describe independent random-policy variation on this fixed request panel; they are NOT training-seed confidence intervals or population/generalization uncertainty. Random ranking samples unique positive rank positions without replacement. The neural full-catalog column retains its actual beam/exact retrieval behavior, and is compared with a uniform-video ranking baseline.",
        "decision_flags": "Report every K/pool. trained_exceeds_null_p99 and trained_at_least_2x_expected are separate predeclared checks, with both_conditions their conjunction. No automatic release admission; all failures are retained. A zero expected baseline yields no ratio or 2x evidence.",
        "monte_carlo_reuse": "Identical fixed panels and (y,m,N) share the same Monte Carlo arrays across parent/current and uniform/popularity pools; analytic expectations are exact.",
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__).resolve(), ROOT / "src/hstu_kvcache/recflow/metrics.py"]},
        "elapsed_seconds": time.monotonic() - started, "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
