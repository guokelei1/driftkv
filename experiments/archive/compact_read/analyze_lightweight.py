#!/usr/bin/env python3
"""Paired UID uncertainty for completed lightweight continuous development.

All three releases share each bootstrap UID multiset, including idle users.
Optional comparison requires exactly the same population, requests and controls;
the script describes fixed candidates and never chooses a winning endpoint.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow.parquet as pq

from hstu_kvcache.evaluation.paired import FixedOrderAUC
from hstu_kvcache.common.artifacts import sha256, write_json

VERSIONS = ("v3", "v4", "v5")
IDENTITY = ("request_id", "uid", "query_timestamp", "item_idx", "label", "edge", "version")
CONTROLS = ("reuse", "first_request_rebuild")


def source(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def interval(values, observed):
    values = np.asarray(values)
    valid = np.isfinite(values)
    return dict(estimate=float(observed) if np.isfinite(observed) else None,
        percentile_95_interval=np.quantile(values[valid], [.025, .975]).tolist() if valid.any() else None,
        finite_draws=int(valid.sum()), undefined_draws=int((~valid).sum()))


def cost_summary(summary):
    cost = summary["cost"]
    additions = sum(row["additional_flops"]
                    for row in cost["source_reconstruction_sensitivity"].values())
    return dict(fixed_population_users=cost["population_users"], requests=cost["requests"],
        denominator=cost["denominator"], denominator_flops=cost["first_request_rebuild_flops"],
        calibration_flops=cost["calibration_flops"], population_dependent_flops=cost["population_dependent_flops"],
        complete_added_flops=cost["complete_added_flops"], complete_added_percent=cost["complete_added_percent"],
        extra_query_percent_of_native_whole_query=cost["extra_query_percent_of_native_whole_query"],
        code_storage_percent_of_native_kv=cost["code_storage_percent_of_native_kv"],
        causal_flags_percent_of_native_kv=cost["causal_flags_percent_of_native_kv"],
        by_release={version: dict(calibration=row["calibration"],
            service_components_flops=row["components_flops"],
            variable_flops=row["variable_flops"]) for version,row in cost["by_release"].items()},
        development_source_trajectory_resources=cost["source_trajectory_resources"],
        source_reconstruction_sensitivity=dict(by_release=cost["source_reconstruction_sensitivity"],
            additional_flops=additions,
            complete_added_flops_if_source_required=cost["complete_added_flops"]+additions,
            complete_added_percent_if_source_required=
                100*(cost["complete_added_flops"]+additions)/cost["first_request_rebuild_flops"],
            boundary="Common pre-existing native KV is the baseline input. If deployment must reconstruct source trajectories exclusively for calibration, add this separately reported work."),
        population_projections=cost["population_projections"], projection_scope=cost["projection_scope"],
        quality_sample_does_not_change_denominator=True)


def load(entries):
    summaries, sources, logits = {}, {}, {}
    reference, population = None, None
    for tag, path in entries:
        if tag in summaries or tag in CONTROLS:
            raise ValueError("candidate tags must be distinct and cannot name a control")
        path = Path(path).resolve()
        summary = json.loads(path.read_text())
        if summary["status"] != "complete":
            raise RuntimeError("analysis requires a completed score summary")
        score_path = Path(summary["scores"]["path"])
        if sha256(score_path) != summary["scores"]["sha256"]:
            raise RuntimeError("completed scores changed")
        table = pq.read_table(score_path).sort_by([("request_id", "ascending")])
        users = np.asarray(sorted(summary["inputs"]["population"]), dtype=np.int64)
        if len(set(users)) != len(users) or len(users) != summary["users"]:
            raise RuntimeError("summary must retain the complete unique UID population, including idle users")
        if len(set(table["request_id"].to_pylist())) != table.num_rows:
            raise RuntimeError("duplicate request score")
        if reference is None:
            reference, population = table, users
            for control in CONTROLS:
                logits[control] = table[control + "_logit"].to_numpy()
        else:
            if not np.array_equal(population, users):
                raise RuntimeError("paired candidates must use the same complete UID population")
            for key in (*IDENTITY, *(control + "_logit" for control in CONTROLS)):
                if not np.array_equal(reference[key].to_numpy(), table[key].to_numpy()):
                    raise RuntimeError(f"paired candidate requests or frozen controls differ: {key}")
        logits[tag] = table["lightweight_logit"].to_numpy()
        if not np.isfinite(logits[tag]).all():
            raise RuntimeError("nonfinite candidate logits")
        summaries[tag] = summary
        sources[tag] = dict(summary=source(path), scores=source(score_path))
    return reference, population, logits, summaries, sources


def paired_uncertainty(table, population, logits, *, repetitions, seed):
    names = list(logits)
    candidate_names = [name for name in names if name not in CONTROLS]
    labels, uids = table["label"].to_numpy(), table["uid"].to_numpy()
    indices = np.searchsorted(population, uids)
    if not np.array_equal(population[indices], uids):
        raise RuntimeError("request UID is absent from the declared population")
    versions = table["version"].to_numpy()
    counts, readers = {}, {}
    for version in VERSIONS:
        mask = versions == version
        selected_labels, selected_uids = labels[mask], uids[mask]
        positive, negative = selected_labels == 1, selected_labels == 0
        counts[version] = dict(requests=int(mask.sum()),
            positive_requests=int(positive.sum()), negative_requests=int(negative.sum()),
            active_uids=len(np.unique(selected_uids)),
            idle_uids=len(population)-len(np.unique(selected_uids)),
            positive_uids=len(np.unique(selected_uids[positive])),
            negative_uids=len(np.unique(selected_uids[negative])),
            both_label_uids=len(set(selected_uids[positive]) & set(selected_uids[negative])))
        readers[version] = {name: FixedOrderAUC(selected_labels, values[mask], indices[mask])
                           for name, values in logits.items()} if mask.any() else {}
    ones = np.ones(len(population), dtype=np.int64)
    observed = np.array([[readers[v][name](ones) if readers[v] else np.nan for name in names]
                         for v in VERSIONS])
    draws = np.full((repetitions, len(VERSIONS), len(names)), np.nan)
    rng = np.random.default_rng(seed)
    for draw in range(repetitions):
        multiplicity = np.bincount(rng.integers(0, len(population), size=len(population)), minlength=len(population))
        for index,version in enumerate(VERSIONS):
            for column,name in enumerate(names):
                if readers[version]:
                    draws[draw,index,column] = readers[version][name](multiplicity)
    reuse, once = names.index("reuse"), names.index("first_request_rebuild")
    gap = observed[:,once]-observed[:,reuse]
    gap_draws = draws[:,:,once]-draws[:,:,reuse]
    mean_gap, mean_gap_draws = gap.mean(), gap_draws.mean(axis=1)
    comparisons = [(name, control) for name in candidate_names for control in CONTROLS]
    if len(candidate_names) == 2:
        comparisons.append(tuple(candidate_names))
    by_release = {}
    for index,version in enumerate(VERSIONS):
        by_release[version] = dict(counts=counts[version],
            auc={name: float(observed[index,column]) if np.isfinite(observed[index,column]) else None
                 for column,name in enumerate(names)},
            once_minus_reuse_auc=interval(gap_draws[:,index], gap[index]),
            paired_auc_differences={f"{left}_minus_{right}": interval(
                draws[:,index,names.index(left)]-draws[:,index,names.index(right)],
                observed[index,names.index(left)]-observed[index,names.index(right)]) for left,right in comparisons})
    overall = dict(once_minus_reuse_mean_auc=interval(mean_gap_draws,mean_gap), candidates={},
        paired_mean_auc_differences={f"{left}_minus_{right}": interval(
            (draws[:,:,names.index(left)]-draws[:,:,names.index(right)]).mean(axis=1),
            (observed[:,names.index(left)]-observed[:,names.index(right)]).mean()) for left,right in comparisons})
    for name in candidate_names:
        index = names.index(name)
        gain = (observed[:,index]-observed[:,reuse]).mean()
        gains = (draws[:,:,index]-draws[:,:,reuse]).mean(axis=1)
        valid = np.isfinite(gains) & np.isfinite(mean_gap_draws) & (mean_gap_draws > 1e-12)
        recoveries = np.full(repetitions,np.nan)
        recoveries[valid] = 100*gains[valid]/mean_gap_draws[valid]
        observed_recovery = 100*gain/mean_gap if np.isfinite(gain) and mean_gap > 1e-12 else np.nan
        overall["candidates"][name] = dict(recovery_percent=interval(recoveries, observed_recovery),
            mean_auc_gain=interval(gains,gain),
            margin_above_80_percent_recovery_auc=interval(gains-.8*mean_gap_draws,gain-.8*mean_gap),
            finite_draws_with_nonpositive_full_gap=int((np.isfinite(mean_gap_draws)&(mean_gap_draws <= 1e-12)).sum()),
            missing_class_draws=int((~np.isfinite(mean_gap_draws)).sum()))
    return dict(by_release=by_release, overall=overall,
        draws_sha256=hashlib.sha256(draws.astype("<f8").tobytes()).hexdigest(), draw_method_order=names)


def main(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    entries = [args.primary] + ([args.compare] if args.compare else [])
    table,population,logits,summaries,sources = load(entries)
    result = paired_uncertainty(table,population,logits,repetitions=args.repetitions,seed=args.seed)
    result.update(status="complete", role="fixed_panel_development_uncertainty", users=len(population),
        requests=table.num_rows, repetitions=args.repetitions, seed=args.seed,
        resampling="Sample all declared UIDs with replacement, including idle users. The same UID multiplicities apply to every method and all three releases in each draw; never resample edges independently.",
        estimand="Equal-release mean candidate-minus-Reuse AUC divided by equal-release mean once-rebuild-minus-Reuse AUC. This is not a mean of per-edge recovery percentages.",
        interval="Two-sided percentile95 using linear quantiles, without clipping or redrawing. Recovery is undefined on missing-class or nonpositive mean Full-gap draws; all excluded-draw counts are reported. AUC differences remain reported regardless of gap sign.",
        scope="One training seed, one fixed development panel. UID resampling describes panel uncertainty, not seed replication or unseen-dataset generalization. No automatic candidate or endpoint selection.",
        costs={name:cost_summary(summary) for name,summary in summaries.items()}, sources=sources,
        execution_sources={str(path.relative_to(ROOT)):sha256(path) for path in (
            Path(__file__).resolve(), ROOT/"src/hstu_kvcache/evaluation/paired.py",
            ROOT/"src/hstu_kvcache/evaluation/binary_metrics.py")},
        elapsed_seconds=time.perf_counter()-started)
    write_json(args.output,result)
    print(json.dumps(dict(output=str(args.output),users=len(population),requests=table.num_rows,
        overall=result["overall"],elapsed_seconds=result["elapsed_seconds"])),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", nargs=2, metavar=("NAME","SUMMARY"), required=True)
    parser.add_argument("--compare", nargs=2, metavar=("NAME","SUMMARY"))
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
