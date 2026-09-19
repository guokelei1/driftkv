#!/usr/bin/env python3
"""Read completed seed17 A/B/C phases at their declared final epochs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WINDOWS = {"A": "19_21", "B": "22_24", "C": "25_27"}
PRIMARY = "ndcg@50"


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def phase_record(directory, phase):
    record = read_json(directory / "summary.json")
    config = record["configuration"]
    final_epoch = config["requested_complete_epochs"]
    if config["phase"] != phase or config["seed"] != 17:
        raise ValueError(f"{directory} is not the requested seed17 phase {phase}")
    observed = [entry["training"]["epoch"] for entry in record["epochs"]]
    expected = list(range(config["start_epoch"] + 1, final_epoch + 1))
    if not record["complete"] or observed != expected:
        raise ValueError(f"{phase} is incomplete; this report requires its declared final epoch")
    previous_nll = record["pre_training_nll"]["mean_nll"]
    probe = record["pre_training_nll"]
    rows = []
    for entry in record["epochs"]:
        training, current = entry["training"], entry["fixed_training_nll"]
        if not training["complete"] or training["processed_requests"] != record["eligible_requests"]:
            raise ValueError(f"{phase} contains an incomplete training epoch")
        for key in ("indices_sha256", "sampled_targets_sha256"):
            if current[key] != probe[key]:
                raise ValueError(f"{phase} training-only NLL probe changed between epochs")
        rows.append(dict(training, fixed_training_nll=current,
                         probe_nll_minus_previous=current["mean_nll"] - previous_nll,
                         probe_nll_minus_initial=current["mean_nll"] - probe["mean_nll"]))
        previous_nll = current["mean_nll"]
    result = dict(run=str(directory), complete=True, final_epoch=final_epoch,
                  start_epoch=config["start_epoch"], canary=config["canary"],
                  training_window=record["training_window"],
                  all_positive_requests=record["all_positive_requests"],
                  eligible_requests=record["eligible_requests"],
                  audited_eligible_requests=record["audited_eligible_requests"],
                  full_training_window_used=record["eligible_requests"] == record["audited_eligible_requests"],
                  pre_training_nll=probe, epochs=rows,
                  source_checkpoint=config["source_checkpoint"],
                  source_checkpoint_sha256=config["source_checkpoint_sha256"],
                  input_summary_sha256=digest(directory / "summary.json"))
    return config, result


def check_shared_panel(parent_dir, parent_label, current_dir, current_label):
    for suffix in ("request_metrics", "sampled_request_metrics"):
        with np.load(parent_dir / f"{parent_label}_{suffix}.npz") as parent:
            with np.load(current_dir / f"{current_label}_{suffix}.npz") as current:
                keys = ["indices", "uids", "positives", "known_positives"]
                keys += [key for key in parent.files if key.endswith("__candidate_count")]
                if any(not np.array_equal(parent[key], current[key]) for key in keys):
                    raise ValueError("Parent/current evaluation panels or denominators differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-run", type=Path, required=True)
    parser.add_argument("--b-run", type=Path)
    parser.add_argument("--c-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.c_run and not args.b_run:
        parser.error("C comparison also requires the B run")
    if args.output.exists():
        parser.error("Use a fresh output directory to preserve comparison evidence")
    directories = {phase: path.resolve() for phase, path in
                   (("A", args.a_run), ("B", args.b_run), ("C", args.c_run)) if path}
    configs, phases, evaluations, evaluation_dirs = {}, {}, {}, {}
    for phase, directory in directories.items():
        config, phases[phase] = phase_record(directory, phase)
        configs[phase] = config
        if phase != "A":
            for key in ("model", "catalog_sha256", "cohort_sha256", "panels_sha256",
                        "candidate_seed", "decoder", "beam_width", "eval_precision", "canary"):
                if config[key] != configs["A"][key]:
                    raise ValueError(f"A/{phase} differ in {key}")
            predecessor = chr(ord(phase) - 1)
            parent_checkpoint = directories[predecessor] / f"{predecessor}_epoch{phases[predecessor]['final_epoch']}" / "checkpoint.pt"
            if (Path(config["source_checkpoint"]).resolve() != parent_checkpoint
                    or config["source_checkpoint_sha256"] != digest(parent_checkpoint)):
                raise ValueError(f"{phase} did not start from the supplied predecessor's declared final checkpoint")
            parent_label = f"parent_{predecessor}_d{WINDOWS[phase]}"
            evaluation_dirs[parent_label] = directory / parent_label
        label = f"{phase}_epoch{phases[phase]['final_epoch']}_d{WINDOWS[phase]}"
        evaluation_dirs[label] = directory / f"{phase}_epoch{phases[phase]['final_epoch']}" / "evaluation"
        if phase != "A":
            check_shared_panel(evaluation_dirs[parent_label], parent_label, evaluation_dirs[label], label)
    for label, directory in evaluation_dirs.items():
        source = read_json(directory / "summary.json")
        value = source[label]
        evaluations[label] = {key: value[key] for key in (
            "full_catalog", "sampled_candidate_diagnostics", "random_expected", "per_day",
            "full_request_panel", "sampled_request_panel", "retrieval_scope", "evaluation_precision",
            "request_selection")}
        evaluations[label].update(run=str(directory), input_summary_sha256=digest(directory / "summary.json"))
        # A large ratio to a tiny random mean is not enough to describe how
        # much observed recommendation evidence supports a single-seed result.
        with np.load(directory / f"{label}_request_metrics.npz") as panel:
            requests = np.load(Path(source["configuration"]["data"]) / "requests.npy", mmap_mode="r")
            days = requests["day"][panel["indices"]]
            counts = {}
            for k in (20, 50, 100):
                hit = panel[f"recall@{k}"] > 0
                counts[str(k)] = dict(requests=int(hit.sum()), users=int(len(np.unique(panel["uids"][hit]))),
                    per_day={str(day): dict(requests=int((hit & (days == day)).sum()),
                        users=int(len(np.unique(panel["uids"][hit & (days == day)]))))
                        for day in np.unique(days)})
            evaluations[label]["hit_counts"] = counts
    args.output.mkdir(parents=True)
    random_output = args.output / "random"
    script = Path(__file__).with_name("random_baseline.py")
    # One invocation shares null draws across identical parent/current panels
    # and uses the evaluator's separate sampled NPZ for every diagnostic pool.
    subprocess.run([sys.executable, str(script), "--runs", *map(str, evaluation_dirs.values()),
                    "--output", str(random_output), "--draws", "5000", "--seed", "20260918"], check=True)
    random_report = read_json(random_output / "summary.json")
    matched_random = {row["phase"]: row["comparisons"] for row in random_report["results"]}
    a_label = f"A_epoch{phases['A']['final_epoch']}_d19_21"
    a_random = matched_random[a_label]["full_catalog"]["metrics"][PRIMARY]
    primary = {"A_19_21": dict(a_random, working_flags_met=a_random.get("both_conditions", False))}
    for phase in ("B", "C"):
        if phase not in directories:
            continue
        predecessor = chr(ord(phase) - 1)
        parent_label = f"parent_{predecessor}_d{WINDOWS[phase]}"
        current_label = f"{phase}_epoch{phases[phase]['final_epoch']}_d{WINDOWS[phase]}"
        old = matched_random[parent_label]["full_catalog"]["metrics"][PRIMARY]
        new = matched_random[current_label]["full_catalog"]["metrics"][PRIMARY]
        delta = new["trained"] - old["trained"]
        daily = {}
        for day, value in evaluations[parent_label]["per_day"].items():
            old_day = value["full_catalog"][PRIMARY]
            new_day = evaluations[current_label]["per_day"][day]["full_catalog"][PRIMARY]
            daily[day] = dict(parent=old_day, current=new_day, current_minus_parent=new_day - old_day)
        primary[f"{predecessor}_{phase}_{WINDOWS[phase]}"] = {
            "parent": old, "current": new, "current_minus_parent": delta,
            "current_gt_parent": delta > 0, "per_day": daily,
            "working_flags_met": old.get("both_conditions", False) and new.get("both_conditions", False) and delta > 0}
    diagnostic_only = any(value["canary"] or not value["full_training_window_used"] for value in phases.values())
    summary = dict(
        scope="Read-only single-seed development report; no training, automatic admission or launch.",
        training_seed=17, repeat_units=1, primary_metric="full_catalog ndcg@50",
        endpoint_rule="Use configuration.requested_complete_epochs only; never select a best observed epoch.",
        interpretation="Random-policy Monte Carlo variation on fixed requests is not seed replication or a confidence interval for B-A. Three completed epochs do not establish convergence. Metrics average requests in the recorded panel: the original1024-per-day panel gives equal day weights, while a complete-window panel weights days by their request traffic. Preserve each evaluation's request_selection. Sampled diagnostics have their own smaller request panels. Inspect hit-request/user counts and all daily update directions before describing a working-flag pass as stable.",
        diagnostic_only=diagnostic_only,
        phase_status="completed tiny canaries; not qualification evidence" if diagnostic_only else "declared phase endpoints complete",
        working_flags_met=all(value["working_flags_met"] for value in primary.values()),
        admissible=False, admission_rule="This script only reports working checks; it never admits a model or launches a next phase.",
        phases=phases, primary=primary, evaluations=evaluations,
        random_comparison_file=str(random_output / "summary.json"),
        companion_rule="All free and sampled NDCG/Recall@20/50/100 values and matched random comparisons are retained; sampled pools never replace the primary.",
        source_sha256={str(path): digest(path) for path in (Path(__file__).resolve(), script.resolve())})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), diagnostic_only=diagnostic_only, primary=primary), indent=2))


if __name__ == "__main__":
    main()
