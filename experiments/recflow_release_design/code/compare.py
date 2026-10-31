"""Pair each new branch with all original requests; retain either effect sign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.recflow_full_population.code.data_common import json_save, sha256
from experiments.recflow_full_population.code.protocol import paired_uid_interval
from .common import read


def parity(directory, spec_path):
    spec = read(spec_path)
    original = Path(spec["original_evaluation_summary"]).parent
    summary = read(directory / "summary.json")
    if not summary["diagnostic_only"] or not summary["same_version_repeated_read_bitwise_equal"]:
        raise ValueError("Expected the numerical diagnostic")
    differences, bindings = {}, {}
    for rank in range(4):
        name = f"rank_{rank}_scores/batch_000000.npz"
        with np.load(original / name) as a, np.load(directory / name) as b:
            if set(a.files) != set(b.files):
                raise ValueError("Score fields changed")
            for key in a.files:
                if key in ("current", "parent", "reuse"):
                    error = float(np.max(np.abs(a[key] - b[key])))
                    differences[key] = max(differences.get(key, 0.), error)
                    if not np.allclose(a[key], b[key], rtol=1e-6, atol=1e-6):
                        raise ValueError(f"Original score parity failed: {key} {error}")
                elif not np.array_equal(a[key], b[key]):
                    raise ValueError(f"Original requests/candidates differ: {key}")
        bindings[str(original / name)] = sha256(original / name)
        bindings[str(directory / name)] = sha256(directory / name)
    return {"status": "passed", "diagnostic_only": True, "requests": summary["requests"],
            "maximum_score_errors": differences, "score_rtol": 1e-6, "score_atol": 1e-6,
            "same_version_read_bitwise_equal": True, "source_sha256": bindings}


def comparison(directory, spec_path):
    spec = read(spec_path)
    original = Path(spec["original_evaluation_summary"]).parent
    new_summary, old_summary = read(directory / "summary.json"), read(original / "summary.json")
    if new_summary.get("diagnostic_only"):
        raise ValueError("Not a full comparison")
    for summary in (new_summary, old_summary):
        parent = summary.get("parent_checkpoint")
        if parent and Path(parent).resolve() != Path(spec["previous_published_checkpoint"]).resolve():
            raise ValueError("Different published parent in comparison")
    keys = ["uid", "day", "rid"]
    columns = keys + ["candidates", "exposed", "positives", "history_length", "current_ndcg@10", "reuse_ndcg@10", "parent_ndcg@10"]
    def frame(path):
        return pd.concat([pd.read_csv(path / f"rank_{r}.csv", usecols=columns) for r in range(4)]).set_index(keys).sort_index()
    old, new = frame(original), frame(directory)
    if not old.index.equals(new.index) or not old.index.is_unique:
        raise ValueError("Original and branch request sets differ")
    for key in ("candidates", "exposed", "positives", "history_length", "parent_ndcg@10"):
        if not np.array_equal(old[key], new[key], equal_nan=True):
            raise ValueError(f"Original and new context/reference differs: {key}")
    old, new = old[old.positives > 0], new[new.positives > 0]
    uid = new.index.get_level_values("uid").to_numpy()
    difference = lambda a, b: paired_uid_interval(uid, np.asarray(a), np.asarray(b))
    current = new_summary["metrics"]["current"]["ndcg@10"]
    reuse = new_summary["metrics"]["reuse"]["ndcg@10"]
    return {"status": "passed", "branch": spec["branch"], "development_only": True,
            "original_evaluation": str(original), "new_evaluation": str(directory),
            "previous_published_checkpoint_sha256": new_summary["parent_sha256"],
            "actual_optimizer_parent_sha256": new_summary["actual_optimizer_parent_sha256"],
            "old_metrics": old_summary["metrics"], "new_metrics": new_summary["metrics"],
            "new_relative_loss_pct": 100*(current-reuse)/current,
            "new_gap_ci95": new_summary["paired_current_minus"]["reuse"]["ci95"],
            "new_minus_old_rebuild": difference(new["current_ndcg@10"], old["current_ndcg@10"]),
            "new_minus_old_reuse": difference(new["reuse_ndcg@10"], old["reuse_ndcg@10"]),
            "change_in_cache_gap": difference(new["current_ndcg@10"]-new["reuse_ndcg@10"], old["current_ndcg@10"]-old["reuse_ndcg@10"]),
            "paired_requests": len(new), "paired_users": len(np.unique(uid)),
            "all_original_request_keys_candidates_history_and_published_parent_metrics_match": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--branch-spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parity", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Keep the original comparison result")
    result = (parity if args.parity else comparison)(args.evaluation, args.branch_spec)
    json_save(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "source_sha256"}), flush=True)


if __name__ == "__main__":
    main()
