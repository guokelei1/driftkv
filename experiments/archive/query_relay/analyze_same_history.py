#!/usr/bin/env python3
"""Offset-invariant ranking and centered-score analysis of saved shadows.

These are diagnostic endpoints, not candidate-selection or acceptance metrics.
The primary matched AUC/recovery remains in diagnose_same_history summary.json.
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
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
import numpy as np
import pyarrow.parquet as pq
from hstu_kvcache.common.artifacts import sha256, write_json


def run(directory, repetitions=1000, seed=17):
    summary = json.loads((directory / "summary.json").read_text())
    configuration = summary["configuration"]
    selected, versions = summary["selected_uids"], configuration["versions"]
    user_index, release_index = {u: i for i, u in enumerate(selected)}, {v: i for i, v in enumerate(versions)}
    sets = defaultdict(list)
    for row in pq.read_table(directory / "scores.parquet").to_pylist():
        sets[row["uid"], row["version"], row["round_id"]].append(row)
    # Four summed metrics and one count per user/release. Idle UIDs keep zero.
    totals = np.zeros((len(selected), len(versions), 5), dtype=np.float64)
    for (uid, version, _), rows in sets.items():
        positive = np.asarray([row["label"] for row in rows], dtype=bool)
        if len(rows) != 16 or positive.sum() != 1:
            raise ValueError("expected fixed one-positive 16-item sets")
        wanted = np.asarray([row["first_request_rebuild_logit"] for row in rows])
        metrics = []
        for field in ("partial_logit", "same_set_mean_logit"):
            actual = np.asarray([row[field] for row in rows])
            residual = actual - wanted
            centered = np.mean((residual - residual.mean()) ** 2)
            margins = actual[positive] - actual[~positive]
            pair_auc = np.mean((margins > 0) + .5 * (margins == 0))
            metrics.extend((centered, pair_auc))
        totals[user_index[uid], release_index[version]] += [*metrics, 1]

    def estimate(values):
        combined = values.sum(0)
        rates = combined[:, :4] / combined[:, 4:5]
        overall = rates.mean(0)
        return np.asarray([overall[1] - overall[3], overall[0] - overall[2],
                           100 * (1 - overall[0] / overall[2])]), rates

    observed, rates = estimate(totals)
    rng = np.random.default_rng(seed)
    draws = np.stack([estimate(totals[rng.integers(len(selected), size=len(selected))])[0]
                      for _ in range(repetitions)])
    metrics = {}
    for index, name in enumerate(("dynamic_minus_same_set_mean_pair_auc",
                                  "dynamic_minus_same_set_mean_centered_mse",
                                  "dynamic_centered_mse_reduction_percent")):
        values = draws[:, index]
        finite = values[np.isfinite(values)]
        metrics[name] = dict(estimate=float(observed[index]),
            percentile_95_interval=np.quantile(finite, [.025, .975]).tolist(),
            finite_draws=len(finite), undefined_draws=repetitions-len(finite))
    by_release = {v: dict(dynamic_centered_mse=float(r[0]), dynamic_pair_auc=float(r[1]),
                         same_set_mean_centered_mse=float(r[2]), same_set_mean_pair_auc=float(r[3]))
                  for v, r in zip(versions, rates, strict=True)}
    result = dict(status="complete_cpu_analysis", by_release=by_release, paired=metrics,
        bootstrap=dict(seed=seed, repetitions=repetitions, resampling="Shared UID multiset including idle users; mean over sets in each release, then equal releases"),
        ranking="Per-set positive-versus-15-negative pair AUC, invariant to common score offsets; diagnostic only",
        reference="Retained once-per-release Rebuild, not per-request Fresh teacher",
        scores_sha256=sha256(directory / "scores.parquet"), source_sha256=sha256(Path(__file__)))
    write_json(directory / "within_set_analysis.json", result)
    print(json.dumps(dict(directory=str(directory), paired=metrics)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", type=Path, nargs="+")
    args = p.parse_args()
    for directory in args.directories:
        run(directory)
