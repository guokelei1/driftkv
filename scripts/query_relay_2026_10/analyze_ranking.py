#!/usr/bin/env python3
"""Additional within-round diagnostics for the registered next-item task.

Primary acceptance remains equal-release pooled AUC recovery. These metrics
expose whether changing gates affects ranking inside one recommendation round.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def evaluate_groups(table, column):
    labels = np.asarray(table["label"])
    scores = np.asarray(table[column])
    keys = np.asarray([value.rsplit(":", 1)[0] for value in table["request_id"].to_pylist()])
    versions = np.asarray(table["version"].to_pylist())
    order = np.argsort(keys, kind="stable")
    boundaries = np.r_[0, np.where(keys[order][1:] != keys[order][:-1])[0] + 1, len(order)]
    rows = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        positions = order[start:stop]
        if labels[positions].sum() != 1:
            raise ValueError("next-item round must have one positive")
        positive = scores[positions][labels[positions] == 1][0]
        negative = scores[positions][labels[positions] == 0]
        greater, tied = int((negative > positive).sum()), int((negative == positive).sum())
        ranks = np.arange(greater + 1, greater + tied + 2)
        metrics = dict(pair_auc=float(((positive > negative) + .5 * (positive == negative)).mean()),
            hit1=float((ranks <= 1).mean()), hit5=float((ranks <= 5).mean()),
            ndcg5=float(np.where(ranks <= 5, 1 / np.log2(ranks + 1), 0).mean()))
        rows.append(dict(version=versions[positions[0]], **metrics))
    by_release = {version: {metric: float(np.mean([row[metric] for row in rows if row["version"] == version]))
                           for metric in ("pair_auc", "hit1", "hit5", "ndcg5")}
                  for version in sorted(set(versions))}
    return dict(rounds=len(rows), by_release=by_release,
        equal_release_mean={metric: float(np.mean([values[metric] for values in by_release.values()]))
                            for metric in ("pair_auc", "hit1", "hit5", "ndcg5")})


def run(args):
    settings = json.loads((args.scores.parent / "configuration.json").read_text())
    if settings["workload"]["kind"] != "behavior":
        raise ValueError("round ranking applies only to the registered next-item task")
    table = pq.read_table(args.scores)
    columns = ("reuse_logit", "full_logit", "partial_logit", "mean_gate_logit")
    reports = {column: evaluate_groups(table, column) for column in columns if column in table.column_names}
    output = dict(status="complete", scores=str(args.scores), candidates=settings["workload"]["candidates"],
        diagnostics=reports, tie_rule="Uniform random tie breaking; report expected hits/NDCG",
        scope="Secondary within-round diagnostics; not primary pooled-AUC or independent confirmation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value["equal_release_mean"] for key, value in reports.items()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
