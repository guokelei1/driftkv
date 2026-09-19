#!/usr/bin/env python3
"""Compare two completed standalone evaluations as development sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from random_baseline import ROOT
from window_comparison import PRIMARY, check_shared_panel, digest, read_json
from hstu_kvcache.recflow.metrics import aggregate_metrics


def load_evaluation(directory):
    """Read only configuration + one completed evaluation phase."""
    source = read_json(directory / "summary.json")
    phases = [key for key in source if key != "configuration"]
    if len(phases) != 1 or "full_catalog" not in source[phases[0]]:
        raise ValueError("Expected one standalone evaluation phase plus configuration")
    phase = phases[0]
    config, evaluation = source["configuration"], source[phase]
    for key, saved_key in (("eval_precision", "evaluation_precision"),
                           ("candidate_seed", "candidate_seed"), ("beam_width", "beam_width")):
        if config[key] != evaluation[saved_key]:
            raise ValueError(f"Configuration and evaluated {key} differ")
    requests = np.load(Path(config["data"]) / "requests.npy", mmap_mode="r")
    hit_counts, evidence = {}, {}
    paths = {"full_catalog": directory / f"{phase}_request_metrics.npz",
             "sampled": directory / f"{phase}_sampled_request_metrics.npz"}
    panels = {name: dict(np.load(path)) for name, path in paths.items()}
    for name, panel in panels.items():
        recorded = evaluation["full_request_panel" if name == "full_catalog" else "sampled_request_panel"]
        if (len(panel["indices"]) != recorded["requests"]
                or hashlib.sha256(panel["indices"].tobytes()).hexdigest() != recorded["indices_sha256"]):
            raise ValueError(f"{name} panel counts/hash differ from the summary")
        if not np.array_equal(requests["uid"][panel["indices"]], panel["uids"]):
            raise ValueError(f"{name} panel user IDs differ from prepared requests")
        evidence[name] = dict(path=str(paths[name]), sha256=digest(paths[name]),
                              requests=len(panel["indices"]), indices_sha256=recorded["indices_sha256"])
    if not np.isin(panels["sampled"]["indices"], panels["full_catalog"]["indices"]).all():
        raise ValueError("Sampled requests are not a subset of the full evaluation")
    scopes = {"full_catalog": evaluation["full_catalog"], **evaluation["sampled_candidate_diagnostics"]}
    for name, saved in scopes.items():
        panel = panels["full_catalog" if name == "full_catalog" else "sampled"]
        prefix = "" if name == "full_catalog" else name + "__"
        metrics = [key for key in saved if "@" in key]
        rows = [dict(positives=float(panel["positives"][i]),
                     known_positives=float(panel["known_positives"][i]),
                     **{key: float(panel[prefix + key][i]) for key in metrics})
                for i in range(len(panel["indices"]))]
        computed = aggregate_metrics(rows, panel["uids"])
        # These arrays preserve the evaluator's float64 values and row order;
        # recomputing its existing aggregation must reproduce all saved fields.
        if any(saved[key] != value for key, value in computed.items()):
            raise ValueError(f"{name} metrics do not match their saved request rows")
        days = requests["day"][panel["indices"]]
        hit_counts[name] = {}
        for k in (20, 50, 100):
            hit = panel[f"{prefix}recall@{k}"] > 0
            hit_counts[name][str(k)] = dict(requests=int(hit.sum()),
                users=int(len(np.unique(panel["uids"][hit]))),
                per_day={str(day): dict(requests=int((hit & (days == day)).sum()),
                    users=int(len(np.unique(panel["uids"][hit & (days == day)]))))
                    for day in np.unique(days)})
        # The same rows also substantiate the saved day-level directions.
        for day in np.unique(days):
            selected = np.flatnonzero(days == day)
            expected = aggregate_metrics([rows[i] for i in selected], panel["uids"][selected])
            day_saved = evaluation["per_day"][str(day)]
            day_saved = day_saved["full_catalog"] if name == "full_catalog" else day_saved["sampled_candidate_diagnostics"][name]
            if any(day_saved[key] != value for key, value in expected.items()):
                raise ValueError(f"{name} day{day} metrics differ from their request rows")
    return phase, config, dict(run=str(directory), phase=phase, configuration=config,
        evaluation=evaluation, hit_counts=hit_counts, panel_evidence=evidence,
        input_summary_sha256=digest(directory / "summary.json"))


def metric_deltas(parent, current):
    return {key: dict(parent=value, current=current[key],
                     current_minus_parent=current[key] - value
                     if value is not None and current[key] is not None else None)
            for key, value in parent.items() if "@" in key}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-metric", choices=("ndcg@50", "ndcg@100"), default=PRIMARY)
    parser.add_argument("--primary-scope", choices=("full_catalog", "uniform_1000"), default="full_catalog")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory to retain previous comparisons")
    parent_dir, current_dir = args.parent.resolve(), args.current.resolve()
    parent_phase, parent_config, parent = load_evaluation(parent_dir)
    current_phase, current_config, current = load_evaluation(current_dir)
    for key in ("model", "history_categories", "catalog_items", "catalog_sha256", "cohort_sha256",
                "candidate_seed", "decoder", "beam_width", "eval_precision", "sampled_distractors", "seed"):
        if parent_config[key] != current_config[key]:
            raise ValueError(f"Parent/current configurations differ in {key}")
    check_shared_panel(parent_dir, parent_phase, current_dir, current_phase)
    if parent["evaluation"]["request_selection"] != current["evaluation"]["request_selection"]:
        raise ValueError("Parent/current request selection descriptions differ")
    args.output.mkdir(parents=True)
    random_output = args.output / "random"
    random_script = Path(__file__).with_name("random_baseline.py")
    subprocess.run([sys.executable, str(random_script), "--runs", str(parent_dir), str(current_dir),
                    "--output", str(random_output), "--draws", "5000", "--seed", "20260918"], check=True)
    random_report = read_json(random_output / "summary.json")
    # Run paths identify the two models even if their phase labels coincide.
    matched = {row["run"]: row["comparisons"] for row in random_report["results"]}
    old = matched[str(parent_dir)][args.primary_scope]["metrics"][args.primary_metric]
    new = matched[str(current_dir)][args.primary_scope]["metrics"][args.primary_metric]
    delta = new["trained"] - old["trained"]
    old_eval, new_eval = parent["evaluation"], current["evaluation"]
    daily = {day: metric_deltas(value["full_catalog"], new_eval["per_day"][day]["full_catalog"])
             for day, value in old_eval["per_day"].items()}
    primary_daily = daily if args.primary_scope == "full_catalog" else {
        day: metric_deltas(value["sampled_candidate_diagnostics"][args.primary_scope],
                          new_eval["per_day"][day]["sampled_candidate_diagnostics"][args.primary_scope])
        for day, value in old_eval["per_day"].items()}
    report = dict(
        scope="Read-only development window-sensitivity comparison; no training or automatic admission.",
        training_seed=parent_config["seed"], repeat_units=1,
        interpretation="Metrics average requests in each recorded panel. The original1024-per-day panel gives equal day weights; complete-window positive-request traffic gives traffic-weighted day averages. A complete-window result does not establish a pass on the old3072-request balanced gate. Random-policy trials are not training-seed replications or uncertainty estimates for the model update.",
        primary_metric=f"{args.primary_scope} {args.primary_metric}",
        primary=dict(parent=old, current=new, current_minus_parent=delta, current_gt_parent=delta > 0,
            sensitivity_working_flags_met=old.get("both_conditions", False) and new.get("both_conditions", False) and delta > 0,
            per_day={day: values[args.primary_metric] for day, values in primary_daily.items()}),
        parent=parent, current=current,
        companion_deltas=dict(full_catalog=metric_deltas(old_eval["full_catalog"],new_eval["full_catalog"]),
            sampled_candidate_diagnostics={name: metric_deltas(value,new_eval["sampled_candidate_diagnostics"][name])
                for name,value in old_eval["sampled_candidate_diagnostics"].items()}),
        per_day_all_full_metrics=daily,
        random_comparison_file=str(random_output / "summary.json"),
        companion_rule="All original metrics, scopes, panel-selection metadata, per-day rows and full/sampled matched random comparisons are retained. Full and sampled diagnostics use their own request denominators.",
        admissible=False,
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in
            (Path(__file__).resolve(), random_script.resolve(), Path(__file__).with_name("window_comparison.py").resolve())})
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), primary=report["primary"]), indent=2))


if __name__ == "__main__":
    main()
