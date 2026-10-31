#!/usr/bin/env python3
"""Whole-panel read-only same-candidate-set gate diagnostic.

Native controls and the corrected dynamic writer trajectory are retained.
Only transient reads replace candidate-varying gates by their set mean. This
does not remove the query projections or ordinary query-dependent attention.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from experiments.archive.partial_refresh.analyze import interval
from experiments.archive.query_relay.aggregate import PANEL, joint_quality, read, sha256, within_round


def merge_gates(summaries):
    """Recover pooled moments from the saved equal-set gate summaries."""
    grouped = {}
    scalar_names = ("mean_within_set_coordinate_std", "nonconstant_coordinate_set_fraction",
                    "saturated_coordinate_fraction")
    for summary in summaries:
        for row in summary["gate_statistics"]["rows"]:
            key = row["version"], row["layer"], row["side"]
            target = grouped.setdefault(key, dict(active=0, inactive=0,
                scalars={name: 0. for name in scalar_names}, second_mean_scalar=0.))
            target["inactive"] += row["inactive_sets"]
            count = row["active_sets"]
            target["active"] += count
            if not count:
                continue
            mean = np.asarray(row["mean_gate"], dtype=np.float64)
            variance = np.asarray(row["within_set_coordinate_rms_std"], dtype=np.float64)**2
            if "mean_sum" not in target:
                target["mean_sum"], target["variance_sum"] = np.zeros_like(mean), np.zeros_like(mean)
            target["mean_sum"] += count * mean
            target["variance_sum"] += count * variance
            target["second_mean_scalar"] += count * (
                row["rms_between_set_mean_std"]**2 + float(np.mean(mean**2)))
            for name in scalar_names:
                target["scalars"][name] += count * row[name]
    rows = []
    for (version, layer, side), entry in sorted(grouped.items()):
        count = entry["active"]
        row = dict(version=version, layer=layer, side=side,
                   active_sets=count, inactive_sets=entry["inactive"])
        if count:
            mean, variance = entry["mean_sum"] / count, entry["variance_sum"] / count
            row.update(mean_gate=mean.tolist(), within_set_coordinate_rms_std=np.sqrt(variance).tolist(),
                rms_within_set_coordinate_std=float(np.sqrt(variance.mean())),
                rms_between_set_mean_std=float(np.sqrt(max(0.,
                    entry["second_mean_scalar"] / count - float(np.mean(mean**2))))),
                **{name: value / count for name, value in entry["scalars"].items()})
        rows.append(row)
    return dict(rows=rows, scope="Active-set-weighted moment merge. Zero-code inactive sets excluded from variation metrics. Set-specific means do not constitute a universal static cache mapping.")


def centered_quality(table, selected_uids, releases, *, repetitions=1000, seed=17):
    """Equal sets per release, equal releases; the same sorted UID draws as AUC."""
    users = np.sort(np.asarray(selected_uids, dtype=np.int64))
    ordered = table.sort_by([("round_id", "ascending")])
    candidates = 16
    labels = ordered["label"].to_numpy().reshape(-1, candidates)
    if not np.all(labels.sum(1) == 1):
        raise ValueError("same-history rounds require one positive and15 negatives")
    uids = ordered["uid"].to_numpy().reshape(-1, candidates)[:, 0]
    versions = ordered["version"].to_numpy().reshape(-1, candidates)[:, 0]
    uid_indices = np.searchsorted(users, uids)
    release_indices = np.asarray([list(releases).index(v) for v in versions])
    wanted = ordered["first_request_rebuild_logit"].to_numpy().reshape(-1, candidates)
    metrics = []
    for column in ("partial_logit", "same_set_mean_logit"):
        actual = ordered[column].to_numpy().reshape(-1, candidates)
        residual = actual - wanted
        centered = np.mean((residual - residual.mean(1, keepdims=True))**2, axis=1)
        positive = actual[labels == 1][:, None]
        negative = labels == 0
        pair = (((actual < positive) & negative).sum(1)
                + .5 * ((actual == positive) & negative).sum(1)) / (candidates - 1)
        metrics.extend((centered, pair))
    # Every selected UID, including idle members, has an explicit zero unit.
    totals = np.zeros((len(users), len(releases), 5), dtype=np.float64)
    np.add.at(totals, (uid_indices, release_indices), np.stack(
        [*metrics, np.ones(len(uids))], axis=1))

    def estimate(units):
        combined = units.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = combined[:, :4] / combined[:, 4:5]
            means = rates.mean(0)
            point = np.asarray([means[1] - means[3], means[0] - means[2],
                                100 * (1 - means[0] / means[2])])
        return point, rates

    observed, rates = estimate(totals)
    rng = np.random.default_rng(seed)
    draws = np.stack([estimate(totals[rng.integers(len(users), size=len(users))])[0]
                      for _ in range(repetitions)])
    names = ("dynamic_minus_same_set_mean_pair_auc", "dynamic_minus_same_set_mean_centered_mse",
             "dynamic_centered_mse_reduction_percent")
    return dict(by_release={version: dict(dynamic_centered_mse=float(row[0]),
        dynamic_pair_auc=float(row[1]), same_set_mean_centered_mse=float(row[2]),
        same_set_mean_pair_auc=float(row[3])) for version, row in zip(releases, rates, strict=True)},
        paired={name: interval(draws[:, i], observed[i]) for i, name in enumerate(names)},
        bootstrap=dict(seed=seed, repetitions=repetitions,
            selected_uid_order="sorted, identical to the joint pooled-AUC bootstrap",
            resampling="Shared UID multiset across every release and method, including idle users; equal sets within release, then equal releases"),
        signs="Positive pair-AUC difference and MSE reduction favor dynamic; negative centered-MSE difference favors dynamic.",
        reference="Retained once-per-release Rebuild; not per-request Fresh teacher",
        scope="Offset-invariant secondary diagnostics, not candidate selection or primary acceptance.")


def aggregate(inputs, *, panel=PANEL, expected_uids=None, repetitions=1000, seed=17):
    panel = Path(panel)
    panel_record = read(panel / "binding.json")
    roles_path = Path(panel_record["users_file"]["path"])
    if not roles_path.is_absolute():
        roles_path = ROOT / roles_path
    if sha256(roles_path) != panel_record["users_file"]["sha256"]:
        raise ValueError("frozen user membership changed")
    roles = read(roles_path)
    expected = list(roles["evaluation"] if expected_uids is None else expected_uids)
    if len(expected) != len(set(expected)) or not set(expected) <= set(roles["evaluation"]):
        raise ValueError("expected UIDs must preserve frozen evaluation roles")
    common_fields = ("panel_sha256", "versions", "workload", "code_preparation", "ablation", "writer", "fitting", "labels")
    tables, configurations, summaries, bindings, seen = [], [], [], [], set()
    publication_indexes, publication_index_records = {}, {}
    maximum_replay_error = 0.
    for directory in map(Path, inputs):
        cfg, summary = read(directory / "configuration.json"), read(directory / "summary.json")
        if summary["status"] != "complete" or cfg["panel_sha256"] != sha256(panel / "binding.json"):
            raise ValueError("incomplete diagnostic or mismatched frozen panel")
        if configurations and any(cfg[key] != configurations[0][key] for key in common_fields):
            raise ValueError("same-history counterfactual protocol differs")
        if cfg["fitting"] or cfg["workload"]["candidates"] != 16 or cfg["code_preparation"] != "compiled":
            raise ValueError("use the frozen16-candidate no-fit compiled diagnostic")
        chosen = cfg["selected_uids"]
        if len(chosen) != len(set(chosen)) or seen & set(chosen) or not set(chosen) <= set(expected):
            raise ValueError("duplicate, overlapping, or unexpected selected UIDs")
        if summary["selected_uids"] != chosen or not all(summary["checks"].get(k, False) for k in
                ("native_controls_reused", "no_shadow_writes", "matched_request_identities")):
            raise ValueError("selected population or read-only trajectory checks differ")
        if summaries and (summary["weights"] != summaries[0]["weights"]
                           or summary["published_prefixes"] != summaries[0]["published_prefixes"]):
            raise ValueError("frozen calibrated weights or publication lineage differs")
        for version, publication in summary["published_prefixes"].items():
            if publication["kind"] != "query_relay_full_catalogue_release_v1" \
                    or publication["bindings"]["calibration_sha256"] != summary["weights"][version]["sha256"]:
                raise ValueError("publication is not bound to the diagnostic weights")
            # diagnose saves the artifact's INTERNAL metadata; the artifact
            # checksum belongs to the external publication index, not itself.
            if version not in publication_indexes:
                publication_dir = Path(cfg["published_prefixes"])
                if not publication_dir.is_absolute():
                    publication_dir = ROOT / publication_dir
                index_path = publication_dir / (version + ".json")
                index = read(index_path)
                artifact_path = publication_dir / (version + ".pt")
                if sha256(artifact_path) != index["artifact_sha256"]:
                    raise ValueError("published artifact differs from its sealed index")
                publication_indexes[version] = index
                publication_index_records[version] = dict(index_path=str(index_path),
                    index_sha256=sha256(index_path), artifact_path=str(artifact_path),
                    artifact_sha256=index["artifact_sha256"])
            if any(publication_indexes[version].get(key) != value for key, value in publication.items()):
                raise ValueError("diagnostic publication metadata differs from its sealed index")
        table = pq.read_table(directory / "scores.parquet").sort_by([("request_id", "ascending")])
        if pc.count_distinct(table["request_id"]).as_py() != len(table) or set(table["uid"].to_pylist()) - set(chosen):
            raise ValueError("score identities duplicated or outside selected users")
        reference_path = Path(cfg["reference"])
        if not reference_path.is_absolute():
            reference_path = ROOT / reference_path
        if sha256(reference_path) != cfg["reference_sha256"]:
            raise ValueError("retained native/dynamic reference changed")
        reference = pq.read_table(reference_path, filters=[("uid", "in", chosen)]).sort_by([("request_id", "ascending")])
        if len(reference) != len(table) or any(not np.array_equal(table[field].to_numpy(), reference[field].to_numpy())
                for field in ("request_id", "uid", "version", "label", "item_idx", "query_timestamp")):
            raise ValueError("diagnostic does not cover exactly its retained candidate identities")
        for field in ("reuse_logit", "first_request_rebuild_logit"):
            if not np.array_equal(table[field].to_numpy(), reference[field].to_numpy()):
                raise ValueError("native controls were changed")
        if not np.array_equal(table["retained_dynamic_logit"].to_numpy(), reference["partial_logit"].to_numpy()):
            raise ValueError("retained dynamic scores differ from the reference")
        error = float(np.max(np.abs(table["partial_logit"].to_numpy() - reference["partial_logit"].to_numpy()))) if len(table) else 0.
        if error > 2e-4:
            raise ValueError("diagnostic changed the dynamic writer trajectory")
        maximum_replay_error = max(maximum_replay_error, error)
        seen.update(chosen); tables.append(table); configurations.append(cfg); summaries.append(summary)
        bindings.append(dict(directory=str(directory.resolve()), users=len(chosen), requests=len(table),
            execution_sources=cfg["source_sha256"],
            sources={name: sha256(directory / name) for name in ("configuration.json", "summary.json", "scores.parquet")}))
    if seen != set(expected):
        raise ValueError(f"incomplete user coverage: expected {len(expected)}, observed {len(seen)}")
    table = pa.concat_tables(tables).combine_chunks()
    if pc.count_distinct(table["request_id"]).as_py() != len(table):
        raise ValueError("request identities overlap across diagnostic inputs")
    configuration = {key: configurations[0][key] for key in common_fields}
    configuration.update(selected_uids=expected, panel=str(panel.resolve()), inputs=bindings,
        publication_indexes=publication_index_records,
        bootstrap=dict(repetitions=repetitions, seed=seed),
        statistical_mapping={"mean_gate_logit": "same_set_mean_logit"},
        scope="Read-only same-set gate mean on the unchanged dynamic corrected-writer trajectory. Native controls reused. No fitting and no claim that all query dependence is removed.")
    frozen_path = Path(configurations[0]["calibrations"])
    if not frozen_path.is_absolute():
        frozen_path = ROOT / frozen_path
    frozen_path = frozen_path.parent / "freeze.json"
    if frozen_path.exists():
        frozen = read(frozen_path)
        for version in configuration["versions"]:
            if frozen["bindings"][version]["weights_sha256"] != summaries[0]["weights"][version]["sha256"] \
                    or frozen["bindings"][version]["publication_sha256"] != publication_index_records[version]["artifact_sha256"]:
                raise ValueError("diagnostic weights/publication do not match the frozen selection")
        configuration["frozen_candidate"] = dict(path=str(frozen_path), sha256=sha256(frozen_path))
    print(json.dumps(dict(status="same_history_inputs_validated", users=len(expected), requests=len(table))), flush=True)
    mapped = table.append_column("mean_gate_logit", table["same_set_mean_logit"])
    dynamic, constant, difference = joint_quality(mapped, expected, releases=configuration["versions"],
        repetitions=repetitions, seed=seed)
    ranking = within_round(mapped, candidates=16)
    result = dict(status="complete", configuration=configuration, selected_uids=expected,
        weights=summaries[0]["weights"], published_prefixes=summaries[0]["published_prefixes"],
        dynamic_quality=dynamic, same_set_mean_quality=constant,
        dynamic_minus_same_set_mean_auc=difference, gate_statistics=merge_gates(summaries),
        within_round_ranking=ranking,
        within_set_analysis=centered_quality(table, expected, configuration["versions"],
                                             repetitions=repetitions, seed=seed),
        checks=dict(complete_disjoint_user_coverage=True, unique_matched_request_ids=True,
            calibrated_publication_lineage_matched=True, native_controls_reused=True,
            no_shadow_writes=True, dynamic_replay_max_absolute_logit_error=maximum_replay_error),
        source_sha256=sha256(__file__), scope=configuration["scope"])
    return table, configuration, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--expected-uids", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh aggregate output directory")
    expected = read(args.expected_uids) if args.expected_uids else None
    if isinstance(expected, dict):
        expected = expected.get("selected_uids", expected.get("evaluation", expected.get("population")))
    table, cfg, result = aggregate(args.inputs, panel=args.panel, expected_uids=expected,
        repetitions=args.bootstrap, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    pq.write_table(table, args.output / "scores.parquet")
    (args.output / "configuration.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="complete", users=len(result["selected_uids"]),
        pooled_auc_difference=result["dynamic_minus_same_set_mean_auc"],
        within_set=result["within_set_analysis"]["paired"])), flush=True)


if __name__ == "__main__":
    main()
