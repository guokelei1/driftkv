#!/usr/bin/env python3
"""Analyze one continuous partial-refresh policy on an explicit user population.

`full_logit` means all current compact codes, not native K/V rebuilding. The
optional `first_request_rebuild_logit` is the separate native rebuilding control.
All releases share UID-cluster bootstrap multiplicities, including idle users.
This analysis does not choose a refresh ratio or a favorable user cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from design_one.compare import FixedOrderAUC


def interval(values, observed):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    return dict(estimate=float(observed) if np.isfinite(observed) else None,
        percentile_95_interval=np.quantile(values[finite], [.025, .975]).tolist()
            if finite.any() else None,
        finite_draws=int(finite.sum()), undefined_draws=int((~finite).sum()))


def ratio_interval(numerator, denominator, observed_numerator, observed_denominator):
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 1e-12)
    values = np.full(numerator.shape, np.nan)
    positive_observed_reference = bool(np.isfinite(observed_denominator) and observed_denominator > 1e-12)
    if positive_observed_reference:
        values[valid] = 100 * numerator[valid] / denominator[valid]
    observed = (100 * observed_numerator / observed_denominator
        if observed_denominator > 1e-12 else np.nan)
    result = interval(values, observed)
    result["nonpositive_reference_gain_draws"] = int(
        (np.isfinite(denominator) & (denominator <= 1e-12)).sum())
    result["positive_reference_gain_draws"] = int(valid.sum())
    result["ratio_defined_at_observed_reference"] = positive_observed_reference
    if not positive_observed_reference:
        result["undefined_reason"] = "Observed reference gain is nonpositive or missing; neither a preservation/recovery point nor its ratio interval is reported. Direct paired AUC differences remain defined where classes exist."
    return result


def score_deviation(partial, full, reuse):
    error = np.asarray(partial) - np.asarray(full)
    correction = np.asarray(full) - np.asarray(reuse)
    mse, energy = float(np.mean(error ** 2)), float(np.mean(correction ** 2))
    return dict(partial_minus_full_logit_mse=mse,
        partial_minus_full_logit_rmse=float(np.sqrt(mse)),
        partial_minus_full_mean_logit_error=float(np.mean(error)),
        partial_minus_full_max_absolute_logit_error=float(np.max(np.abs(error))),
        full_minus_reuse_logit_correction_energy=energy,
        correction_relative_mse=mse / energy if energy > 1e-20 else None,
        correction_relative_rmse=float(np.sqrt(mse / energy)) if energy > 1e-20 else None)


def summarize(scores, selected_uids, *, repetitions=500, seed=17,
              releases=None, preservation_threshold=.95):
    """Return paired quality evidence for one policy (rho may vary by release).

    Accept Arrow tables, pandas DataFrames, or lists of request dictionaries.
    `selected_uids` must include every selected user, including users without
    requests. Supplying `releases` makes absent releases explicit instead of
    silently dropping them from the equal-release estimand.
    """
    if isinstance(scores, pa.Table):
        table = scores
    elif hasattr(scores, "columns"):
        table = pa.Table.from_pandas(scores, preserve_index=False)
    else:
        table = pa.Table.from_pylist(list(scores))
    required = ("request_id", "uid", "version", "label", "reuse_logit",
                "full_logit", "partial_logit")
    missing = set(required) - set(table.column_names)
    if missing:
        raise ValueError(f"missing score columns: {sorted(missing)}")
    if table.num_rows == 0:
        raise ValueError("cannot analyze an empty score panel")
    users = np.sort(np.asarray(selected_uids, dtype=np.int64))
    if not len(users) or len(np.unique(users)) != len(users):
        raise ValueError("selected_uids must be a nonempty unique population")
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    table = table.sort_by([("request_id", "ascending")])
    ids = table["request_id"].to_pylist()
    if len(set(ids)) != len(ids):
        raise ValueError("pass one policy at a time; request_id must be unique")
    labels = table["label"].to_numpy()
    uids = table["uid"].to_numpy()
    versions = table["version"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    indices = np.searchsorted(users, uids)
    if (indices >= len(users)).any() or not np.array_equal(users[indices], uids):
        raise ValueError("a request UID is absent from selected_uids")
    releases = list(releases) if releases is not None else sorted(set(versions.tolist()))
    if set(versions.tolist()) - set(releases):
        raise ValueError("releases excludes observed versions")
    methods = ["reuse", "full", "partial"]
    if "first_request_rebuild_logit" in table.column_names:
        methods.append("first_request_rebuild")
    logits = {name: table[name + "_logit"].to_numpy().astype(np.float64)
              for name in methods}
    if not all(np.isfinite(values).all() for values in logits.values()):
        raise ValueError("all matched logits must be finite")
    readers, counts, masks = {}, {}, {}
    for version in releases:
        mask = versions == version
        masks[version] = mask
        active = len(np.unique(uids[mask]))
        counts[version] = dict(requests=int(mask.sum()), active_uids=active,
            idle_uids=len(users)-active, positive_requests=int((labels[mask] == 1).sum()),
            negative_requests=int((labels[mask] == 0).sum()))
        readers[version] = {name: FixedOrderAUC(labels[mask], values[mask], indices[mask])
            for name, values in logits.items()} if mask.any() else {}
    ones = np.ones(len(users), dtype=np.int64)
    observed = np.array([[readers[v][name](ones) if readers[v] else np.nan
                         for name in methods] for v in releases])
    draws = np.full((repetitions, len(releases), len(methods)), np.nan)
    rng = np.random.default_rng(seed)
    for draw in range(repetitions):
        multiplicity = np.bincount(rng.integers(0, len(users), size=len(users)),
                                  minlength=len(users))
        for i, version in enumerate(releases):
            for j, name in enumerate(methods):
                if readers[version]:
                    draws[draw, i, j] = readers[version][name](multiplicity)

    def comparisons(values, samples):
        reuse, full, partial = (methods.index(name) for name in ("reuse", "full", "partial"))
        full_gain = values[full] - values[reuse]
        partial_gain = values[partial] - values[reuse]
        full_draws = samples[:, full] - samples[:, reuse]
        partial_draws = samples[:, partial] - samples[:, reuse]
        margin = interval(partial_draws - preservation_threshold * full_draws,
                          partial_gain - preservation_threshold * full_gain)
        ci = margin["percentile_95_interval"]
        result = dict(full_minus_reuse_auc=interval(full_draws, full_gain),
            partial_minus_reuse_auc=interval(partial_draws, partial_gain),
            partial_minus_full_auc=interval(samples[:, partial]-samples[:, full],
                                           values[partial]-values[full]),
            full_gain_retained_percent=ratio_interval(partial_draws, full_draws,
                                                      partial_gain, full_gain),
            preservation_threshold=preservation_threshold,
            preservation_margin_auc=margin,
            positive_full_gain=bool(np.isfinite(full_gain) and full_gain > 1e-12),
            point_preserves_threshold=bool(full_gain > 1e-12 and
                                          partial_gain >= preservation_threshold * full_gain),
            lower_95_margin_above_zero=bool(ci is not None and ci[0] > 0),
            noninferiority_supported_at_threshold=bool(full_gain > 1e-12 and
                                                       ci is not None and ci[0] > 0))
        if "first_request_rebuild" in methods:
            rebuild = methods.index("first_request_rebuild")
            rebuild_gap = values[rebuild] - values[reuse]
            rebuild_draws = samples[:, rebuild] - samples[:, reuse]
            result["first_request_rebuild_minus_reuse_auc"] = interval(rebuild_draws, rebuild_gap)
            result["full_rebuild_recovery_percent"] = ratio_interval(
                full_draws, rebuild_draws, full_gain, rebuild_gap)
            result["partial_rebuild_recovery_percent"] = ratio_interval(
                partial_draws, rebuild_draws, partial_gain, rebuild_gap)
        return result

    by_release = {}
    for i, version in enumerate(releases):
        mask = masks[version]
        row = dict(counts=counts[version], auc={name: float(observed[i,j])
            if np.isfinite(observed[i,j]) else None for j,name in enumerate(methods)},
            paired=comparisons(observed[i], draws[:,i]))
        if mask.any():
            row["score_deviation"] = score_deviation(
                logits["partial"][mask], logits["full"][mask], logits["reuse"][mask])
            if "rho" in table.column_names:
                ratios, frequencies = np.unique(table["rho"].to_numpy()[mask], return_counts=True)
                row["rho_request_counts"] = {str(float(rho)):int(n)
                                              for rho,n in zip(ratios,frequencies)}
        by_release[version] = row
    mean_observed, mean_draws = observed.mean(axis=0), draws.mean(axis=1)
    deviations = [row["score_deviation"] for row in by_release.values()
                  if "score_deviation" in row]
    deviation = dict(pooled_requests=score_deviation(
        logits["partial"], logits["full"], logits["reuse"]))
    if len(deviations) == len(releases):
        mean_mse = np.mean([row["partial_minus_full_logit_mse"] for row in deviations])
        mean_energy = np.mean([row["full_minus_reuse_logit_correction_energy"] for row in deviations])
        deviation["equal_release_mean"] = dict(partial_minus_full_logit_mse=float(mean_mse),
            full_minus_reuse_logit_correction_energy=float(mean_energy),
            correction_relative_mse=float(mean_mse / mean_energy) if mean_energy > 1e-20 else None)
    return dict(users=len(users), requests=table.num_rows,
        active_uids=len(np.unique(uids)), idle_uids=len(users)-len(np.unique(uids)),
        selected_uids=users.tolist(), releases=releases, by_release=by_release,
        overall=dict(equal_release_mean_auc={name:float(mean_observed[j])
            if np.isfinite(mean_observed[j]) else None for j,name in enumerate(methods)},
            paired=comparisons(mean_observed, mean_draws), score_deviation=deviation),
        bootstrap=dict(repetitions=repetitions, seed=seed,
            draws_sha256=hashlib.sha256(draws.astype("<f8").tobytes()).hexdigest(),
            method_order=methods,
            resampling="One shared UID multiset per draw for every release and method, including idle selected users.",
            interval="Two-sided percentile95, linear quantiles; no clipping or redrawing. Ratio draws with nonpositive reference gain are undefined and counted."),
        estimand="Equal-release mean AUC gains. Preservation divides partial-minus-Reuse by full-current-code-minus-Reuse; Rebuild recovery uses a separate native first-request rebuilding denominator.",
        scope="Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.")


def report(result):
    rows = ["# Partial-refresh quality evidence", "",
        f"Selected users: {result['users']}; active: {result['active_uids']}; requests: {result['requests']}.", "",
        "Full denotes all current compact codes. Native first-request rebuilding is a separate reference.", "",
        "| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for version,row in result["by_release"].items():
        auc = row["auc"]
        retained = row["paired"]["full_gain_retained_percent"]["estimate"]
        error = row.get("score_deviation", {}).get("correction_relative_mse")
        values = [auc["reuse"],auc["full"],auc["partial"],retained,error]
        rows.append("| " + version + " | " + " | ".join(
            "undefined" if value is None else f"{value:.6g}" for value in values) + " |")
    overall = result["overall"]["paired"]
    retained = overall["full_gain_retained_percent"]
    rows.extend(["", f"Equal-release gain retention: {retained['estimate']}%; UID95: {retained['percentile_95_interval']}.",
        f"Preservation margin at {100*overall['preservation_threshold']:g}%: {overall['preservation_margin_auc']}.",
        f"Noninferiority supported at this threshold: {overall['noninferiority_supported_at_threshold']}.",
        "", result["scope"], ""])
    return "\n".join(rows)


def summarize_cohorts(scores, selected_uids, cohorts, **kwargs):
    """Describe externally fixed, label-free UID cohorts; do not select one.

    `cohorts` maps descriptive names to UID lists fixed before inspecting score
    outcomes. Membership is intersected with the selected population, retaining
    inactive selected members. Every cohort keeps the same release estimand.
    """
    table = scores if isinstance(scores, pa.Table) else (
        pa.Table.from_pandas(scores, preserve_index=False) if hasattr(scores, "columns")
        else pa.Table.from_pylist(list(scores)))
    population = set(selected_uids)
    all_releases = kwargs.pop("releases", None)
    if all_releases is None:
        all_releases = sorted(set(table["version"].to_pylist()))
    results = {}
    for name, members in cohorts.items():
        chosen = sorted(population & set(members))
        if chosen:
            subset = table.filter(pa.compute.is_in(table["uid"], value_set=pa.array(chosen)))
            results[name] = (summarize(subset, chosen, releases=all_releases, **kwargs)
                if subset.num_rows else dict(users=len(chosen), requests=0,
                                             status="no_requests", selected_uids=chosen))
        else:
            results[name] = dict(users=0, requests=0, status="no_selected_members")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--users", type=Path, required=True,
                        help="JSON UID list or object with selected_uids/population")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--releases", nargs="+")
    parser.add_argument("--cohorts", type=Path,
                        help="JSON name->UID lists defined from label-free pre-release data")
    args = parser.parse_args()
    users = json.loads(args.users.read_text())
    if isinstance(users, dict):
        users = users["selected_uids"] if "selected_uids" in users else users["population"]
    table = pq.read_table(args.scores)
    policies = set(table["policy"].to_pylist()) if "policy" in table.column_names else {"default"}
    cohorts = json.loads(args.cohorts.read_text()) if args.cohorts else None
    results = {}
    for policy in sorted(policies):
        subset = table.filter(pa.compute.equal(table["policy"], policy)) if "policy" in table.column_names else table
        result = summarize(subset, users, repetitions=args.repetitions,
                           seed=args.seed, releases=args.releases)
        if cohorts is not None:
            result["cohorts"] = summarize_cohorts(subset, users, cohorts,
                repetitions=args.repetitions, seed=args.seed, releases=args.releases)
        results[policy] = result
    result = next(iter(results.values())) if len(results) == 1 else dict(policies=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text("\n\n".join(
        (f"Policy: {name}\n\n" if len(results) > 1 else "") + report(value)
        for name, value in results.items()))
    print(json.dumps(dict(output=str(args.output), policies={name:value["overall"]["paired"]
                                                           for name,value in results.items()})))


if __name__ == "__main__":
    main()
