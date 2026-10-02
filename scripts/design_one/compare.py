#!/usr/bin/env python3
"""Paired user-cluster bootstrap of explicit Design 1 score comparisons.

Each draw samples the same 10000-user multiset for every method and retains all
requests of every sampled user, including repeated users. The estimand remains
pooled request ROC-AUC. Intervals describe this selected development user panel;
they are not training-seed replication, qualification, or selection adjustment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow.parquet as pq

from hstu_kvcache.evaluation.binary_metrics import sigmoid
from unified_reuse_2026_09.common import sha256, write_json

PANEL = ROOT / "results/design_one_2026_10/benchmark_panel"
ALIGNMENT = ("request_id", "uid", "label", "query_timestamp", "full_logit", "reuse_logit")


class FixedOrderAUC:
    """Weighted positive/negative pair concordance with half credit for ties."""
    def __init__(self, labels, logits, user_index):
        probabilities = sigmoid(logits)  # Same tie definition as binary_metrics.
        order = np.argsort(probabilities, kind="mergesort")
        sorted_scores = probabilities[order]
        self.groups = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        self.users = np.asarray(user_index)[order]
        self.positive = np.asarray(labels)[order] == 1
        self.negative = ~self.positive

    def __call__(self, user_multiplicity):
        weights = user_multiplicity[self.users]
        positive = np.add.reduceat(weights * self.positive, self.groups).astype(np.float64)
        negative = np.add.reduceat(weights * self.negative, self.groups).astype(np.float64)
        denominator = positive.sum() * negative.sum()
        if denominator == 0:
            return float("nan")
        below_and_half_tied = np.cumsum(negative) - .5 * negative
        return float(np.sum(positive * below_and_half_tied) / denominator)


def load_scores(entries, panel):
    binding_path = panel / "binding.json"
    binding = json.loads(binding_path.read_text())
    request_source = binding["files"]["evaluation_requests"]
    request_path = ROOT / request_source["path"]
    if sha256(request_path) != request_source["sha256"]:
        raise RuntimeError("frozen request panel changed")
    reference = pq.read_table(request_path, columns=list(ALIGNMENT)).sort_by([("request_id", "ascending")])
    ids = reference["request_id"].to_pylist()
    uids, user_index = np.unique(reference["uid"].to_numpy(), return_inverse=True)
    if len(reference) != 90296 or len(uids) != 10000 or len(set(ids)) != len(ids):
        raise RuntimeError("expected the complete frozen 10000-user/90296-request panel")
    labels = reference["label"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError("labels are not binary")
    readers, sources = {}, {}
    for tag, path_text in entries:
        if not tag or tag in readers:
            raise ValueError("comparison tags must be nonempty and unique")
        path = Path(path_text).resolve()
        table = pq.read_table(path, columns=[*ALIGNMENT, "hstu_logit"]).sort_by([("request_id", "ascending")])
        for key in ALIGNMENT:
            if not np.array_equal(table[key].to_numpy(), reference[key].to_numpy()):
                raise RuntimeError(f"{tag}: {key} differs from the fixed request panel")
        logits = table["hstu_logit"].to_numpy()
        if not np.isfinite(logits).all():
            raise RuntimeError(f"{tag}: nonfinite logits")
        readers[tag] = FixedOrderAUC(labels, logits, user_index)
        sources[tag] = {"path": str(path), "sha256": sha256(path), "requests": len(table)}
    metadata = {"binding": {"path": str(binding_path.resolve()), "sha256": sha256(binding_path)},
        "requests": request_source, "users": len(uids), "request_count": len(reference),
        "sorted_uid_sha256": hashlib.sha256(uids.astype("<i8").tobytes()).hexdigest()}
    return readers, sources, metadata


def paired_bootstrap(readers, primary, *, users, repetitions=1000, seed=17):
    tags = list(readers)
    observed = {tag: reader(np.ones(users, dtype=np.int64)) for tag, reader in readers.items()}
    if not all(np.isfinite(value) for value in observed.values()):
        raise RuntimeError("observed panel must contain both labels")
    draws = np.empty((repetitions, len(tags)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for iteration in range(repetitions):
        sampled = rng.integers(0, users, size=users)
        multiplicity = np.bincount(sampled, minlength=users)
        draws[iteration] = [readers[tag](multiplicity) for tag in tags]
    valid = np.isfinite(draws).all(axis=1)
    if not valid.any():
        raise RuntimeError("no bootstrap draw contains both labels")
    primary_index = tags.index(primary)
    comparisons = []
    for index, tag in enumerate(tags):
        if tag == primary:
            continue
        differences = draws[valid, primary_index] - draws[valid, index]
        interval = np.quantile(differences, [.025, .975], method="linear")
        comparisons.append({"primary": primary, "comparator": tag,
            "primary_auc": observed[primary], "comparator_auc": observed[tag],
            "auc_difference": observed[primary] - observed[tag],
            "bootstrap_mean_auc_difference": float(differences.mean()),
            "percentile_95_interval": interval.tolist()})
    return {"comparisons": comparisons, "repetitions": repetitions, "seed": seed,
        "valid_draws": int(valid.sum()), "single_class_draws": int((~valid).sum()),
        "draws_sha256": hashlib.sha256(draws.astype("<f8").tobytes()).hexdigest(),
        "draw_column_order": tags}


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"comparison output exists: {args.output}")
    if args.repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    entries = [args.primary, *args.compare]
    readers, sources, panel = load_scores(entries, args.panel)
    result = paired_bootstrap(readers, args.primary[0], users=panel["users"],
                              repetitions=args.repetitions, seed=args.seed)
    result.update(status="complete", evaluation_role="development_descriptive",
        panel=panel, scores=sources,
        scope="Paired user-cluster resampling of this fixed selected development panel. Training seed is not replicated; no qualification or adjustment for choosing the primary after development comparisons.",
        estimand="Pooled request ROC-AUC(primary) minus pooled request ROC-AUC(comparator), using canonical sigmoid score ties.",
        resampling="Sample 10000 UIDs with replacement per draw; every request gets its user's multiplicity. The identical UID multiset is shared by all policies. Single-class draws, if any, are counted and excluded without resampling.",
        confidence_interval="Two-sided 95% percentile interval; NumPy linear quantiles; no clipping.",
        configuration={"primary": args.primary, "compare": args.compare,
            "seed": args.seed, "repetitions": args.repetitions},
        execution_sources={str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "src/hstu_kvcache/evaluation/binary_metrics.py")},
        numpy_version=np.__version__, elapsed_seconds=time.perf_counter() - started)
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "comparisons": result["comparisons"],
        "valid_draws": result["valid_draws"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", nargs=2, metavar=("TAG", "PARQUET"), required=True)
    parser.add_argument("--compare", nargs=2, metavar=("TAG", "PARQUET"), action="append", required=True)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
