#!/usr/bin/env python3
"""Merge disjoint frozen-user replays and recompute matched quality evidence.

Shards are execution units, never statistical units. One shared UID bootstrap
covers every release and method, including selected users without requests.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hstu_kvcache.evaluation.paired import FixedOrderAUC
from experiments.archive.partial_refresh.analyze import interval, ratio_interval, score_deviation

PANEL = ROOT / "results/design_system_2026_10/phase2/representative4096/panel"
OBSERVER_CHANGE = ROOT / "results/query_relay_2026_10/observer_change.json"


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def paired(values, samples, *, candidate, threshold=.8):
    """The retained summarize() estimand, including undefined ratio draws."""
    reuse, full, current = values[0], values[1], values[candidate]
    reference_gain, gain = full - reuse, current - reuse
    reference_draws, gains = samples[:, 1] - samples[:, 0], samples[:, candidate] - samples[:, 0]
    margin = interval(gains - threshold * reference_draws, gain - threshold * reference_gain)
    ci = margin["percentile_95_interval"]
    supported = bool(reference_gain > 1e-12 and ci is not None and ci[0] > 0)
    full_ratio = ratio_interval(reference_draws, reference_draws, reference_gain, reference_gain)
    candidate_ratio = ratio_interval(gains, reference_draws, gain, reference_gain)
    return dict(full_minus_reuse_auc=interval(reference_draws, reference_gain),
        partial_minus_reuse_auc=interval(gains, gain),
        partial_minus_full_auc=interval(samples[:, candidate] - samples[:, 1], current - full),
        full_gain_retained_percent=candidate_ratio, preservation_threshold=threshold,
        preservation_margin_auc=margin, positive_full_gain=bool(reference_gain > 1e-12),
        point_preserves_threshold=bool(reference_gain > 1e-12 and gain >= threshold * reference_gain),
        lower_95_margin_above_zero=bool(ci is not None and ci[0] > 0),
        noninferiority_supported_at_threshold=supported,
        first_request_rebuild_minus_reuse_auc=interval(reference_draws, reference_gain),
        full_rebuild_recovery_percent=full_ratio, partial_rebuild_recovery_percent=candidate_ratio)


def joint_quality(table, selected_uids, *, releases, repetitions=1000, seed=17):
    """Four methods, one bootstrap pass; uses the existing weighted AUC reader."""
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    users = np.sort(np.asarray(selected_uids, dtype=np.int64))
    if not len(users) or len(np.unique(users)) != len(users):
        raise ValueError("selected user population must be nonempty and unique")
    table = table.sort_by([("request_id", "ascending")])
    if pc.count_distinct(table["request_id"]).as_py() != table.num_rows:
        raise ValueError("candidate request IDs are duplicated")
    labels, uids = table["label"].to_numpy(), table["uid"].to_numpy()
    versions = table["version"].to_numpy()
    if not np.isin(labels, (0, 1)).all() or set(versions) - set(releases):
        raise ValueError("unexpected labels or release")
    indices = np.searchsorted(users, uids)
    if (indices >= len(users)).any() or not np.array_equal(users[indices], uids):
        raise ValueError("scores contain an unselected UID")
    names = ["reuse", "full", "partial"]
    if "mean_gate_logit" in table.column_names:
        names.append("mean_gate")
    logits = {name: table[name + "_logit"].to_numpy().astype(np.float64) for name in names}
    if not all(np.isfinite(values).all() for values in logits.values()):
        raise ValueError("matched logits must be finite")
    if not np.array_equal(logits["full"], table["first_request_rebuild_logit"].to_numpy()):
        raise ValueError("this new-task audit requires Full to mean native first-request Rebuild")
    readers, counts, masks = {}, {}, {}
    for version in releases:
        mask = versions == version
        masks[version] = mask
        active = len(np.unique(uids[mask]))
        counts[version] = dict(requests=int(mask.sum()), active_uids=active,
            idle_uids=len(users) - active, positive_requests=int((labels[mask] == 1).sum()),
            negative_requests=int((labels[mask] == 0).sum()))
        readers[version] = [FixedOrderAUC(labels[mask], logits[name][mask], indices[mask])
                            for name in names] if mask.any() else []
    ones = np.ones(len(users), dtype=np.int64)
    observed = np.asarray([[reader(ones) for reader in readers[version]] if readers[version]
                           else [np.nan] * len(names) for version in releases])
    draws = np.full((repetitions, len(releases), len(names)), np.nan)
    rng = np.random.default_rng(seed)
    for draw in range(repetitions):
        weights = np.bincount(rng.integers(0, len(users), size=len(users)), minlength=len(users))
        for j, version in enumerate(releases):
            if readers[version]:
                draws[draw, j] = [reader(weights) for reader in readers[version]]
        if (draw + 1) % 100 == 0:
            print(json.dumps(dict(status="uid_bootstrap", completed=draw + 1,
                                  repetitions=repetitions)), flush=True)
    mean_observed, mean_draws = observed.mean(axis=0), draws.mean(axis=1)
    bootstrap = dict(repetitions=repetitions, seed=seed,
        draws_sha256=hashlib.sha256(draws.astype("<f8").tobytes()).hexdigest(), method_order=names,
        resampling="One shared UID multiset per draw for every release and method, including idle selected users.",
        interval="Two-sided percentile95, linear quantiles; nonpositive reference gain ratio draws are undefined and counted.")

    def format_quality(candidate):
        index = names.index(candidate)
        method_indices = (0, 1, index, 1)
        method_names = ("reuse", "full", "partial", "first_request_rebuild")
        by_release = {}
        for j, version in enumerate(releases):
            record = dict(counts=counts[version],
                auc={name: float(observed[j, k]) if np.isfinite(observed[j, k]) else None
                     for name, k in zip(method_names, method_indices)},
                paired=paired(observed[j], draws[:, j], candidate=index))
            if masks[version].any():
                record["score_deviation"] = score_deviation(logits[candidate][masks[version]],
                    logits["full"][masks[version]], logits["reuse"][masks[version]])
            by_release[version] = record
        deviations = [row["score_deviation"] for row in by_release.values() if "score_deviation" in row]
        deviation = dict(pooled_requests=score_deviation(logits[candidate], logits["full"], logits["reuse"]))
        if len(deviations) == len(releases):
            mse = float(np.mean([row["partial_minus_full_logit_mse"] for row in deviations]))
            energy = float(np.mean([row["full_minus_reuse_logit_correction_energy"] for row in deviations]))
            deviation["equal_release_mean"] = dict(partial_minus_full_logit_mse=mse,
                full_minus_reuse_logit_correction_energy=energy,
                correction_relative_mse=mse / energy if energy > 1e-20 else None)
        return dict(users=len(users), requests=table.num_rows, active_uids=len(np.unique(uids)),
            idle_uids=len(users) - len(np.unique(uids)), selected_uids=users.tolist(), releases=list(releases),
            by_release=by_release, overall=dict(
                equal_release_mean_auc={name: float(mean_observed[k]) if np.isfinite(mean_observed[k]) else None
                                       for name, k in zip(method_names, method_indices)},
                paired=paired(mean_observed, mean_draws, candidate=index), score_deviation=deviation),
            bootstrap=bootstrap,
            estimand="Equal-release mean AUC gain recovery: (Relay-Reuse)/(once-per-release native Rebuild-Reuse).",
            scope="Complete selected development panel; no adjustment for exploratory selection or independent-confirmation claim.")
    quality = format_quality("partial")
    constant_quality = dynamic_gain = None
    if "mean_gate" in names:
        mean_index = names.index("mean_gate")
        constant_quality = format_quality("mean_gate")
        dynamic_gain = dict(overall=interval(mean_draws[:, 2] - mean_draws[:, mean_index],
            mean_observed[2] - mean_observed[mean_index]),
            by_release={version: interval(draws[:, j, 2] - draws[:, j, mean_index],
                observed[j, 2] - observed[j, mean_index]) for j, version in enumerate(releases)})
    return quality, constant_quality, dynamic_gain


def within_round(table, candidates):
    """Matrix form of analyze_ranking.evaluate_groups with identical tie rule."""
    keys = pc.replace_substring_regex(table["request_id"], pattern=r":[^:]+$", replacement="")
    ordered = table.append_column("_round", keys).sort_by([("_round", "ascending")])
    encoded = pc.dictionary_encode(ordered["_round"].combine_chunks())
    indices = encoded.indices.to_numpy()
    starts = np.r_[0, np.flatnonzero(indices[1:] != indices[:-1]) + 1, len(indices)]
    if not np.all(np.diff(starts) == candidates):
        raise ValueError("recommendation round does not contain exactly the configured candidates")
    labels = ordered["label"].to_numpy().reshape(-1, candidates)
    if not np.all(labels.sum(axis=1) == 1):
        raise ValueError("recommendation round must contain exactly one positive")
    uids = ordered["uid"].to_numpy().reshape(-1, candidates)
    versions = ordered["version"].to_numpy().reshape(-1, candidates)
    if not np.all(uids == uids[:, :1]) or not np.all(versions == versions[:, :1]):
        raise ValueError("recommendation round crosses users or releases")
    rounds, reports = len(labels), {}
    selected_versions = versions[:, 0]
    for column in ("reuse_logit", "full_logit", "partial_logit", "mean_gate_logit"):
        if column not in ordered.column_names:
            continue
        scores = ordered[column].to_numpy().reshape(-1, candidates)
        positive = (scores * labels).sum(axis=1)
        negatives = labels == 0
        greater = ((scores > positive[:, None]) & negatives).sum(axis=1)
        ties = ((scores == positive[:, None]) & negatives).sum(axis=1)
        pair = (((scores < positive[:, None]) & negatives).sum(axis=1) + .5 * ties) / (candidates - 1)
        def hits(k):
            return np.maximum(0, np.minimum(ties + 1, k - greater)) / (ties + 1)
        # At most k=5 possible discounted hit positions, including tied ranks.
        ndcg = sum(((greater < rank) & (greater + ties + 1 >= rank)) /
                   ((ties + 1) * np.log2(rank + 1)) for rank in range(1, 6))
        values = dict(pair_auc=pair, hit1=hits(1), hit5=hits(5), ndcg5=ndcg)
        by_release = {version: {metric: float(array[selected_versions == version].mean())
                               for metric, array in values.items()} for version in sorted(set(selected_versions))}
        reports[column] = dict(rounds=rounds, by_release=by_release,
            equal_release_mean={metric: float(np.mean([row[metric] for row in by_release.values()]))
                                for metric in values})
    return dict(diagnostics=reports, candidates=candidates,
        tie_rule="Uniform random tie breaking; report expected hits/NDCG",
        scope="Secondary within-round diagnostics; not primary pooled-AUC or independent confirmation")


def merge_gate_statistics(summaries):
    grouped = {}
    for summary in summaries:
        for row in summary["gate_statistics"]["rows"]:
            key = row["version"], row["layer"], row["side"]
            mean, std, count = np.asarray(row["mean"]), np.asarray(row["std"]), row["queries"]
            target = grouped.setdefault(key, dict(count=0, active=0,
                sum=np.zeros_like(mean), squared=np.zeros_like(mean)))
            target["count"] += count
            target["active"] += row["queries_with_coded_prefix"]
            target["sum"] += mean * count
            target["squared"] += (std**2 + mean**2) * count
    rows = []
    for (version, layer, side), value in sorted(grouped.items()):
        mean = value["sum"] / value["count"]
        std = np.sqrt(np.maximum(0, value["squared"] / value["count"] - mean**2))
        rows.append(dict(version=version, layer=layer, side=side, queries=value["count"],
            queries_with_coded_prefix=value["active"], mean=mean.tolist(), std=std.tolist(),
            mean_coordinate_std=float(std.mean()), nonconstant_coordinate_fraction=float((std > 1e-5).mean())))
    return dict(rows=rows, scope="Query-count-weighted moment merge; variation alone is not evidence of quality contribution.")


def merge_ledger(summaries, name, fields):
    counts = Counter()
    for summary in summaries:
        for row in summary[name]:
            counts[tuple(row[field] for field in fields)] += row["count"]
    return [dict(**dict(zip(fields, key)), count=int(count)) for key, count in sorted(counts.items())]


def aggregate(inputs, *, panel=PANEL, expected_uids=None, repetitions=1000, seed=17):
    panel = Path(panel)
    panel_data = read(panel / "binding.json")
    population_source = Path(panel_data["users_file"]["path"])
    if not population_source.is_absolute():
        population_source = ROOT / population_source
    if sha256(population_source) != panel_data["users_file"]["sha256"]:
        raise ValueError("frozen user role membership changed")
    roles = read(population_source)
    population = roles["evaluation"]
    expected = list(population if expected_uids is None else expected_uids)
    if len(expected) != len(set(expected)) or not set(expected) <= set(population):
        raise ValueError("expected population must be a unique subset of frozen evaluation roles")
    panel_sha = sha256(panel / "binding.json")
    observer_change = read(OBSERVER_CHANGE) if OBSERVER_CHANGE.exists() else None
    permitted_observer_shas = set() if observer_change is None else {
        observer_change["old_evaluate_sha256"], observer_change["new_evaluate_sha256"]}
    common_fields = ("panel_sha256", "versions", "code_preparation", "native_writes",
        "storage_floats_per_layer_row", "read_shared", "read_private", "mean_gate_control",
        "role", "no_online_labels_or_teacher", "workload", "anchor_inverse")
    configurations, summaries, tables, seen, bindings = [], [], [], set(), []
    for directory in map(Path, inputs):
        configuration, summary = read(directory / "configuration.json"), read(directory / "summary.json")
        configuration["anchor_inverse"] = configuration.get("anchor_inverse", "transpose")
        if summary["status"] != "complete" or configuration["panel_sha256"] != panel_sha:
            raise ValueError(f"incomplete replay or wrong panel: {directory}")
        if not all(summary["checks"].get(key, False) for key in
                   ("frozen_user_roles", "matched_request_identities", "corrected_writes", "new_current_rows_zero_code")):
            raise ValueError("input replay did not pass its causal serving checks")
        if configurations and any(configuration[key] != configurations[0][key] for key in common_fields):
            raise ValueError(f"replay protocol or frozen serving source differs: {directory}")
        if configurations:
            old_sources, new_sources = configurations[0]["source_sha256"], configuration["source_sha256"]
            if old_sources != new_sources:
                if {k: v for k, v in old_sources.items() if k != "evaluate.py"} != \
                        {k: v for k, v in new_sources.items() if k != "evaluate.py"} \
                        or {old_sources["evaluate.py"], new_sources["evaluate.py"]} != permitted_observer_shas:
                    raise ValueError("serving source differs outside the exact registered observer change")
        selected = configuration["selected_uids"]
        if len(selected) != len(set(selected)) or set(selected) & seen or not set(selected) <= set(expected):
            raise ValueError(f"duplicate, overlapping, or unexpected selected UIDs: {directory}")
        if summary["selected_uids"] != selected or {unit["uid"] for unit in summary["units"]} != set(selected) \
                or len(summary["units"]) != len(selected):
            raise ValueError(f"summary UID units do not match selected population: {directory}")
        if summaries:
            if summary["weights"] != summaries[0]["weights"] or summary["compiled_subset"] != summaries[0]["compiled_subset"]:
                raise ValueError(f"calibration or published prefix SHA/config differs: {directory}")
        if configuration["code_preparation"] != "compiled" or not configuration["published_prefixes"]:
            raise ValueError("population aggregation requires the frozen full-catalogue published prefixes")
        if configuration["anchor_inverse"] == "actual" and not configuration["source_sha256"].get("precision.py"):
            raise ValueError("actual anchor inverse requires its recorded precision source")
        for version, published in summary["compiled_subset"].items():
            if published["kind"] != "query_relay_full_catalogue_release_v1" \
                    or published["bindings"]["calibration_sha256"] != summary["weights"][version]["sha256"]:
                raise ValueError("publication is not bound to the evaluated calibration")
            if configuration["anchor_inverse"] == "actual" and not published.get("anchor_numerics_extension"):
                raise ValueError("actual anchor inverse requires its recorded publication extension")
        table = pq.read_table(directory / "scores.parquet")
        if set(table["uid"].to_pylist()) - set(selected):
            raise ValueError("scores contain users absent from the shard")
        expected_counts = {(unit["uid"], version): record["requests"]
            for unit in summary["units"] for version, record in unit["by_release"].items()}
        actual_counts = table.group_by(["uid", "version"]).aggregate([("request_id", "count")]).to_pylist()
        observed_counts = {(row["uid"], row["version"]): row["request_id_count"] for row in actual_counts}
        if any(observed_counts.get(key, 0) != count for key, count in expected_counts.items()) \
                or set(observed_counts) - set(expected_counts):
            raise ValueError("score rows do not match per-user/release request counters")
        if bool(configuration["mean_gate_control"]) != ("mean_gate_logit" in table.column_names):
            raise ValueError("mean-gate control columns differ from replay configuration")
        seen.update(selected)
        configurations.append(configuration); summaries.append(summary); tables.append(table)
        bindings.append(dict(directory=str(directory.resolve()), users=len(selected), requests=table.num_rows,
            execution_source_sha256=configuration["source_sha256"],
            sources={name: sha256(directory / name) for name in ("configuration.json", "summary.json", "scores.parquet")}))
    if seen != set(expected):
        raise ValueError(f"aggregate user coverage incomplete: expected {len(expected)}, observed {len(seen)}")
    table = pa.concat_tables(tables).combine_chunks()
    if pc.count_distinct(table["request_id"]).as_py() != table.num_rows:
        raise ValueError("candidate request identity is duplicated across inputs")
    settings = {key: configurations[0][key] for key in common_fields}
    settings.update(selected_uids=expected, population=expected, panel=str(panel.resolve()),
        source_sha256=configurations[0]["source_sha256"],
        calibration_root=configurations[0]["calibration_root"],
        published_prefixes=configurations[0]["published_prefixes"], inputs=bindings,
        expected_uid_scope="Complete frozen evaluation population" if set(expected) == set(population) else "Explicit frozen evaluation subset",
        aggregation="All matched scores recomputed; no averaging of shard AUC or recovery percentages",
        bootstrap=dict(repetitions=repetitions, seed=seed), source_sha256_aggregate=sha256(__file__))
    calibration_directory = Path(settings["calibration_root"])
    if not calibration_directory.is_absolute():
        calibration_directory = ROOT / calibration_directory
    frozen_path = calibration_directory.parent / "freeze.json"
    if frozen_path.exists():
        frozen = read(frozen_path)
        if frozen.get("anchor_inverse", "transpose") != settings["anchor_inverse"]:
            raise ValueError("evaluated anchor inverse differs from the frozen selection")
        for version in settings["versions"]:
            sealed = frozen["bindings"][version]
            if sealed["weights_sha256"] != summaries[0]["weights"][version]["sha256"] \
                    or sealed["publication_sha256"] != summaries[0]["compiled_subset"][version]["artifact_sha256"]:
                raise ValueError("evaluated weights or publication differ from the frozen selection")
        settings["frozen_candidate"] = dict(path=str(frozen_path), sha256=sha256(frozen_path),
                                            status=frozen["status"], decision=frozen["decision"])
        for field in ("frozen_at_utc", "numerical_path_frozen_at_utc", "metadata_normalization"):
            if field in frozen:
                settings["frozen_candidate"][field] = frozen[field]
    if len({configuration["source_sha256"]["evaluate.py"] for configuration in configurations}) > 1:
        settings["registered_observer_change"] = dict(path=str(OBSERVER_CHANGE),
            sha256=sha256(OBSERVER_CHANGE), record=observer_change,
            scope="Only the exact registered old/new evaluate hashes are permitted; adapter/session and other source hashes remain identical.")
    print(json.dumps(dict(status="inputs_validated", users=len(expected), requests=table.num_rows,
                          shards=len(inputs))), flush=True)
    quality, mean_quality, dynamic_gain = joint_quality(table, expected,
        releases=settings["versions"], repetitions=repetitions, seed=seed)
    ranking = within_round(table, settings["workload"]["candidates"]) if settings["workload"]["kind"] == "behavior" else None
    counters = Counter()
    for summary in summaries:
        counters.update(summary["actual_service_counters"])
    merged = dict(status="complete", selected_uids=expected, weights=summaries[0]["weights"], quality=quality,
        shared_feature_banks=summaries[0]["shared_feature_banks"], mean_gate_quality=mean_quality,
        dynamic_minus_mean_gate_auc=dynamic_gain,
        mean_gate_control_scope="Global fit-only mean gates, read-only replacement on the dynamic candidate trajectory; not a separate writer policy",
        within_round_ranking=ranking,
        units=[unit for summary in summaries for unit in summary["units"]],
        actual_service_counters=dict(counters), gate_statistics=merge_gate_statistics(summaries),
        native_ledger=merge_ledger(summaries, "native_ledger", ("action", "version", "before", "width")),
        code_read_ledger=merge_ledger(summaries, "code_read_ledger",
            ("kind", "version", "old_length", "queries", "coded_rows", "native_rows")),
        compiled_subset=summaries[0]["compiled_subset"], inputs=bindings,
        checks=dict(frozen_user_roles=True, disjoint_selected_uids=True, expected_user_coverage=True,
            unique_candidate_request_ids=True, matched_request_counters=True,
            identical_panel_workload_serving_sources=True, identical_calibration_and_publication_bindings=True,
            corrected_writes=True, new_current_rows_zero_code=True),
        cost_scope="Quality panel operation shapes only; fixed30k complete cost reported separately",
        source_sha256=settings["source_sha256"], source_sha256_aggregate=settings["source_sha256_aggregate"])
    if "registered_observer_change" in settings:
        merged["registered_observer_change"] = settings["registered_observer_change"]
    if "frozen_candidate" in settings:
        merged["frozen_candidate"] = settings["frozen_candidate"]
    return table, settings, merged


def run(args):
    began = time.perf_counter()
    if args.output.exists():
        raise ValueError("aggregate output must be a new directory")
    expected = None
    if args.expected_uids is not None:
        expected = read(args.expected_uids)
        if isinstance(expected, dict):
            expected = expected.get("selected_uids", expected.get("evaluation", expected.get("population")))
    table, settings, summary = aggregate(args.inputs, panel=args.panel,
        expected_uids=expected, repetitions=args.bootstrap, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    settings["elapsed_aggregation_seconds"] = time.perf_counter() - began
    (args.output / "configuration.json").write_text(json.dumps(settings, indent=2) + "\n")
    pq.write_table(table, args.output / "scores.parquet")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    paired_result = summary["quality"]["overall"]["paired"]
    print(json.dumps(dict(status="complete", output=str(args.output), users=len(summary["selected_uids"]),
        requests=table.num_rows, recovery=paired_result["full_gain_retained_percent"],
        dynamic_minus_mean_gate_auc=summary["dynamic_minus_mean_gate_auc"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--expected-uids", type=Path,
                        help="Optional frozen subset UID JSON; default is the panel's complete evaluation membership")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    run(parser.parse_args())
