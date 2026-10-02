#!/usr/bin/env python3
"""Fixed-panel Reuse comparisons and label-free request concentration audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow.parquet as pq

from design_one.compare import PANEL, FixedOrderAUC, load_scores, paired_bootstrap
from unified_reuse_2026_09.common import sha256, write_json


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"probe output exists: {args.output}")
    tag = "pure_mean8_rank8"
    readers, sources, panel = load_scores([(tag, str(args.primary))], args.panel)
    path = ROOT / panel["requests"]["path"]
    table = pq.read_table(path).sort_by([("request_id", "ascending")])
    labels = table["label"].to_numpy()
    uids, index, counts = np.unique(table["uid"].to_numpy(), return_inverse=True, return_counts=True)
    # One shared stream produces both requested contrasts. Reuse is the helper's
    # base, and signs/interval endpoints are reversed below for gain reporting.
    readers = {"Reuse": FixedOrderAUC(labels, table["reuse_logit"].to_numpy(), index),
        tag: readers[tag], "Full": FixedOrderAUC(labels, table["full_logit"].to_numpy(), index)}
    result = paired_bootstrap(readers, "Reuse", users=len(uids),
                              repetitions=args.repetitions, seed=args.seed)
    for contrast in result["comparisons"]:
        primary_auc, comparator_auc = contrast["comparator_auc"], contrast["primary_auc"]
        lower, upper = contrast["percentile_95_interval"]
        contrast.update(primary=contrast["comparator"], comparator="Reuse",
            primary_auc=primary_auc, comparator_auc=comparator_auc,
            auc_difference=-contrast["auc_difference"],
            bootstrap_mean_auc_difference=-contrast["bootstrap_mean_auc_difference"],
            percentile_95_interval=[-upper, -lower])
    positives = np.bincount(index, weights=labels, minlength=len(uids)).astype(np.int64)
    order = np.lexsort((uids, -counts))
    concentration = {"total_users": len(uids), "total_requests": len(table),
        "positive_requests": int(positives.sum()), "negative_requests": int(len(table)-positives.sum()),
        "ranking": "descending request count, then ascending UID; every user retained",
        "max_requests_per_uid": int(counts[order[0]]), "top_user_groups": [], "largest_10_users": []}
    for size in (1, 10, 100):
        top = order[:size]
        requests, positive = int(counts[top].sum()), int(positives[top].sum())
        concentration["top_user_groups"].append({"users": size, "requests": requests,
            "request_share": requests / len(table), "positive_requests": positive,
            "negative_requests": requests-positive})
    for i in order[:10]:
        concentration["largest_10_users"].append({"uid": int(uids[i]), "requests": int(counts[i]),
            "request_share": float(counts[i]/len(table)), "positive_requests": int(positives[i]),
            "negative_requests": int(counts[i]-positives[i])})
    result.update(status="complete", evaluation_role="development_descriptive",
        panel=panel, scores=sources,
        control_scores={"path": str(path), "sha256": sha256(path),
            "columns": {"Full": "full_logit", "Reuse": "reuse_logit"}},
        request_concentration=concentration,
        scope="Paired user-cluster resampling of the entire fixed selected development panel. No user exclusions, training-seed replication, qualification, or adjustment for prior development selection.",
        estimand="Pooled request ROC-AUC(method) minus pooled request ROC-AUC(Reuse), using the same sigmoid tie definition as the retained evaluations.",
        resampling="Sample the same 10000-user multiset with replacement for all three methods on each draw; every request receives its user's multiplicity. Single-class draws, if any, are counted and excluded without replacement draws.",
        confidence_interval="Two-sided 95% percentile interval; linear quantiles, no clipping. Helper Reuse-minus-method differences are sign-reversed including interval endpoints.",
        configuration={"primary": str(args.primary.resolve()), "seed": args.seed,
            "repetitions": args.repetitions, "comparisons": [[tag, "Reuse"], ["Full", "Reuse"]]},
        execution_sources={str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "scripts/design_one/compare.py",
            ROOT / "src/hstu_kvcache/evaluation/binary_metrics.py")},
        numpy_version=np.__version__, elapsed_seconds=time.perf_counter()-started)
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "comparisons": result["comparisons"],
        "request_concentration": concentration, "valid_draws": result["valid_draws"],
        "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
