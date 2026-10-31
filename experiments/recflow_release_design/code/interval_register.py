"""Register one audited release pair; replication alone supplies primary metrics.

This command does not launch work or search experiments. The caller explicitly
chooses a completed pilot and a completed, disjoint-user replication. Adverse
replications are retained and are not counted as usable positive model groups.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from experiments.recflow_full_population.code.protocol import day_bounds_ms, day_date
from .interval_protocol import ancestry


TIMEZONE = ZoneInfo("Asia/Shanghai")
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "release_interval/model_groups.json"


def read(path):
    return json.loads(Path(path).read_text())


def canonical(path):
    return str(Path(path).resolve())


def same_number(left, right):
    return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
        float(left), float(right), rel_tol=0., abs_tol=1e-12)


def interval(days):
    days = sorted(set(map(int, days)))
    start, end = day_bounds_ms(days[0])[0], day_bounds_ms(days[-1])[1]
    return {"days": days, "dates": [day_date(day) for day in days],
            "start_inclusive": datetime.fromtimestamp(start / 1000, TIMEZONE).isoformat(),
            "end_exclusive": datetime.fromtimestamp(end / 1000, TIMEZONE).isoformat(),
            "timezone": "Asia/Shanghai"}


def load_completed(run):
    """Check terminal completion before opening any potentially partial results."""
    run = Path(run).resolve()
    completion_path = run / "completion.json"
    if not completion_path.is_file() or read(completion_path).get("status") != "completed":
        raise ValueError(f"Experiment is not completed: {run}")
    completion = read(completion_path)
    plan, summary = read(run / "plan.json"), read(run / "summary.json")
    audit, analysis = read(run / "audit.json"), read(run / "analysis.json")
    if (summary.get("status") != "completed" or analysis.get("status") != "completed"
            or audit.get("status") != "passed"):
        raise ValueError(f"Completed scoring, analysis and candidate audit are required: {run}")
    if (not audit.get("all_chosen_users_complete_day_requests")
            or audit.get("sample_native_metric_comparison", {}).get("status") != "passed"):
        raise ValueError("Complete user-day coverage and native metric comparison must pass")
    chosen = np.load(run / "chosen_uids.npy", allow_pickle=False)
    if chosen.ndim != 1 or not len(chosen) or len(np.unique(chosen)) != len(chosen):
        raise ValueError("Chosen UIDs must be a nonempty unique vector")
    if plan.get("history_limit") != 1024 or summary.get("history_limit") != 1024:
        raise ValueError("Registry requires the unchanged 1024-history task")
    lineage = ancestry(plan["checkpoint"], plan["cache_source"])
    day = int(plan["eval_day"])
    if day != lineage[-1]["train_end_day"] + 1 or analysis["evaluation_day"] != day:
        raise ValueError("Evaluation must be the next complete day after actual target training")
    if (analysis["evaluation_interval_ms_half_open"] != list(day_bounds_ms(day))
            or canonical(analysis["checkpoint"]) != canonical(plan["checkpoint"])
            or canonical(analysis["cache_source"]) != canonical(plan["cache_source"])):
        raise ValueError("Recorded analysis model pair or full-day interval differs")
    if audit["actual_training_path"] != lineage or analysis["actual_training_path"] != lineage:
        raise ValueError("Audit and analysis do not describe the actual model ancestry")
    overall = analysis["overall"]
    for field in ("requests", "positive_requests", "positive_users"):
        if len({summary[field], audit[field], overall[field], completion[field]}) != 1:
            raise ValueError(f"Completed artifacts disagree on {field}")
    if (summary["users"] != audit["active_users"] or audit["active_users"] != overall["active_users"]
            or overall["active_users"] != completion["active_users"]):
        raise ValueError("Completed artifacts disagree on active-user count")
    if any(value != len(chosen) for value in (
            audit["chosen_users"], overall["chosen_users"], completion["chosen_users"],
            plan["selection"]["chosen_users"])):
        raise ValueError("Saved chosen UIDs disagree with reported population size")
    if overall["positive_requests"] < 1:
        raise ValueError("No positive requests exist for defined ranking metrics")
    for metric in ("ndcg@10", "recall@10"):
        values = overall["metrics"][metric]
        gap = overall["comparisons"][metric]["ancestor_cache_loss"]
        if (not same_number(values["rebuild"], summary["metrics"]["current"][metric])
                or not same_number(values["ancestor_reuse"], summary["metrics"]["reuse"][metric])
                or not same_number(values["rebuild"] - values["ancestor_reuse"], gap["difference"])):
            raise ValueError("Primary metric means or paired differences disagree")
        if gap["requests"] != overall["positive_requests"] or gap["users"] != overall["positive_users"]:
            raise ValueError("Paired interval uses a different positive-request population")
        bounds = gap["ci95"]
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
            raise ValueError("Invalid paired interval")
    return {"run": str(run), "plan": plan, "summary": summary, "audit": audit,
            "analysis": analysis, "completion": completion, "chosen": chosen, "lineage": lineage}


def validate_pair(pilot, replication):
    if pilot["run"] == replication["run"]:
        raise ValueError("Pilot and replication must be different completed runs")
    for field in ("checkpoint", "cache_source", "data"):
        if canonical(pilot["plan"][field]) != canonical(replication["plan"][field]):
            raise ValueError(f"Pilot and replication differ in {field}")
    if pilot["plan"]["eval_day"] != replication["plan"]["eval_day"]:
        raise ValueError("Pilot and replication must evaluate the same complete day")
    if np.intersect1d(pilot["chosen"], replication["chosen"]).size:
        raise ValueError("Pilot and replication chosen UID populations overlap")


def window_record(run, role):
    analysis, audit, summary = run["analysis"], run["audit"], run["summary"]
    overall, day = analysis["overall"], int(run["plan"]["eval_day"])
    ndcg, recall = overall["metrics"]["ndcg@10"], overall["metrics"]["recall@10"]
    gap = overall["comparisons"]["ndcg@10"]["ancestor_cache_loss"]
    recall_gap = overall["comparisons"]["recall@10"]["ancestor_cache_loss"]
    return {"window_id": f"D{day}_full_day_{Path(run['run']).name}", "evidence_role": role,
            "interval": interval([day]), "evaluation_directory": run["run"],
            "summary_path": str(Path(run["run"]) / "summary.json"),
            "analysis_path": str(Path(run["run"]) / "analysis.json"),
            "independent_candidate_and_metric_audit_path": str(Path(run["run"]) / "audit.json"),
            "chosen_user_count": len(run["chosen"]), "chosen_uids_path": str(Path(run["run"]) / "chosen_uids.npy"),
            "request_count": overall["requests"], "active_user_count": overall["active_users"],
            "positive_request_count": overall["positive_requests"], "positive_user_count": overall["positive_users"],
            "zero_positive_request_count": overall["zero_positive_requests"],
            "candidate_instances": audit["candidate_instances"],
            "rebuild_ndcg10": ndcg["rebuild"], "reuse_ndcg10": ndcg["ancestor_reuse"],
            "ndcg10_rebuild_minus_reuse": gap["difference"], "ndcg10_relative_loss_pct": gap["relative_loss_pct"],
            "ndcg10_paired_user_ci95": gap["ci95"], "bootstrap_replicates": gap["samples"],
            "bootstrap_seed": gap["seed"], "rebuild_recall10": recall["rebuild"],
            "reuse_recall10": recall["ancestor_reuse"], "recall10_rebuild_minus_reuse": recall_gap["difference"],
            "recall10_paired_user_ci95": recall_gap["ci95"], "audit_status": audit["status"],
            "maximum_per_request_metric_recomputation_error": audit["maximum_independent_per_request_metric_error"],
            "evaluation_wall_seconds": summary["wall_seconds"], "elapsed": analysis["elapsed"],
            "predetermined_slices": analysis["predetermined_slices"]}


def update_registry(registry, pilot, replication):
    """Return a new registry, with one ordered checkpoint pair updated once."""
    validate_pair(pilot, replication)
    registry = copy.deepcopy(registry)
    plan, lineage = replication["plan"], replication["lineage"]
    source, target = canonical(plan["cache_source"]), canonical(plan["checkpoint"])
    catalog = registry["model_catalog"]
    ids = {canonical(value["checkpoint"]): name for name, value in catalog.items()}
    if source not in ids or target not in ids:
        raise ValueError("Source and target must already have actual model records in the catalog")
    source_id, target_id = ids[source], ids[target]
    existing = [g for g in registry["groups"] if
                (canonical(g["cache_source_checkpoint"]), canonical(g["target_checkpoint"])) == (source, target)]
    if len(existing) > 1:
        raise ValueError("Registry already duplicates this ordered model pair")
    primary, development = window_record(replication, "primary_disjoint_user_replication"), window_record(pilot, "development_pilot")
    positive = primary["ndcg10_rebuild_minus_reuse"] > 0 and development["ndcg10_rebuild_minus_reuse"] > 0
    ci = primary["ndcg10_paired_user_ci95"]
    direction = "Both pilot and disjoint-user replication have Rebuild > Reuse." if positive else "Positive direction did not reproduce in both pilot and replication; retained as adverse or inconclusive evidence."
    uncertainty = ("Replication nominal 95% interval is entirely above zero." if ci[0] > 0
                   else "Replication nominal 95% interval is entirely below zero." if ci[1] < 0
                   else "Replication nominal 95% interval includes zero; direction remains uncertain.")
    limits = [
        "Primary reported numbers use replication users only; pilot metrics are separate development evidence, never pooled to strengthen the result.",
        "Pilot and replication chosen UID lists are disjoint, including inactive users; the same target models and calendar day are shared, so this is not an independent training seed or time-window replication.",
        "Pair was selected during exploration; nominal user-cluster bootstrap intervals are not multiple-comparison-adjusted.",
        "Both samples come from the original fixed training-user cohort with strictly future requests; this is not unseen-user generalization.",
        "Complete logged candidates and matched causal history prefixes capped at 1024; this is not persistent-cache/FIFO serving.",
        "Only the recorded complete natural-day windows are established; no additional-window stability is assumed."]
    group = copy.deepcopy(existing[0]) if existing else {}
    windows = [w for w in group.get("windows", []) if w.get("evaluation_directory") != primary["evaluation_directory"]]
    pilots = [w for w in group.get("development_evidence", []) if w.get("evaluation_directory") != development["evaluation_directory"]]
    update_days = sorted({day for row in lineage[1:] for day in row["train_days"]})
    group.update(group_id=f"{source_id}_to_{target_id}",
                 label="replicated_release_interval_positive" if positive else "release_interval_not_replicated_positive",
                 counts_as_distinct_model_group=True, counts_as_usable_positive_group=positive,
                 cache_source_model_id=source_id, target_model_id=target_id,
                 cache_source_checkpoint=source, target_checkpoint=target,
                 actual_immediate_training_parent_checkpoint=lineage[-1]["actual_parent"],
                 training_lineage_model_ids=catalog[target_id]["training_lineage_model_ids"],
                 actual_training_path_between_releases=lineage, new_training_between_releases=interval(update_days),
                 publication_interval_days=lineage[-1]["train_end_day"]-lineage[0]["train_end_day"],
                 publication_relation="Recorded daily optimizer path retained; intermediate checkpoints are treated as unpublished, and the target is released after its final training cutoff.",
                 cache_relation="At every request both models encode the identical causal prefix capped at 1024. Target reads its own KV for Rebuild and the designated ancestor KV for Reuse.",
                 primary_window_id=primary["window_id"], windows=windows+[primary],
                 development_evidence=pilots+[development],
                 pilot_replication_disjoint_chosen_users=True, reliability=direction+" "+uncertainty,
                 reliability_limits=limits)
    if existing:
        registry["groups"][registry["groups"].index(existing[0])] = group
    else:
        registry["groups"].append(group)
    pair_ids = {(canonical(g["cache_source_checkpoint"]), canonical(g["target_checkpoint"])) for g in registry["groups"]}
    if len(pair_ids) != len(registry["groups"]):
        raise ValueError("Registry contains duplicate ordered model pairs")
    registry["distinct_model_group_count"] = len(pair_ids)
    registry["usable_positive_model_group_count"] = sum(g.get("counts_as_usable_positive_group", g.get("label") == "retained_original_positive_baseline") for g in registry["groups"])
    registry["status"] = "original_pairs_and_completed_release_interval_replications"
    registry["updated_at"] = datetime.now(TIMEZONE).isoformat()
    return registry


def csv_rows(registry):
    rows = []
    for group in registry["groups"]:
        windows = group["windows"]
        selected = [w for w in windows if w.get("window_id") == group.get("primary_window_id")]
        window = selected[0] if selected else windows[0]
        catalog = registry["model_catalog"]
        source, target = group["cache_source_model_id"], group["target_model_id"]
        ci = window["ndcg10_paired_user_ci95"]
        rows.append({
            "group_id": group["group_id"], "label": group["label"], "cache_source_model": source,
            "target_model": target, "cache_source_checkpoint": group["cache_source_checkpoint"],
            "target_checkpoint": group["target_checkpoint"],
            "actual_training_parent_checkpoint": group["actual_immediate_training_parent_checkpoint"],
            "training_lineage_models": ">".join(group["training_lineage_model_ids"]),
            "training_lineage_checkpoints": json.dumps([catalog[m]["checkpoint"] for m in group["training_lineage_model_ids"]]),
            "source_training_cutoff_date": catalog[source]["cumulative_training_cutoff_date"],
            "target_training_cutoff_date": catalog[target]["cumulative_training_cutoff_date"],
            "publication_relation": group["publication_relation"], "cache_relation": group["cache_relation"],
            "publication_interval_days": group["publication_interval_days"],
            "eval_dates": ";".join(window["interval"]["dates"]),
            "eval_start_inclusive": window["interval"]["start_inclusive"],
            "eval_end_exclusive": window["interval"]["end_exclusive"], "timezone": "Asia/Shanghai",
            "resident_users": registry["population"]["fixed_resident_training_users"],
            "chosen_users": window.get("chosen_user_count", registry["population"]["fixed_resident_training_users"]),
            "active_users": window["active_user_count"], "requests": window["request_count"],
            "positive_users": window["positive_user_count"], "positive_requests": window["positive_request_count"],
            "candidate_instances": window["candidate_instances"], "rebuild_ndcg10": window["rebuild_ndcg10"],
            "reuse_ndcg10": window["reuse_ndcg10"], "rebuild_minus_reuse": window["ndcg10_rebuild_minus_reuse"],
            "relative_loss_pct": window["ndcg10_relative_loss_pct"], "ci95_low": ci[0], "ci95_high": ci[1],
            "rebuild_recall10": window["rebuild_recall10"], "reuse_recall10": window["reuse_recall10"],
            "reliability": group["reliability"], "reliability_limits": " | ".join(group["reliability_limits"]),
            "evaluation_directory": window["evaluation_directory"],
            "independent_audit_path": window["independent_candidate_and_metric_audit_path"],
            "all_original_nine_edge_metrics": registry["all_original_evidence"]["all_nine_adjacent_pairs_metrics_csv"],
            "all_failed_extraepoch_metrics": registry["failed_extraepoch_evidence"]["metrics_csv"],
            "primary_evidence_role": window.get("evidence_role", "original_full_population_evaluation"),
            "development_pilot_directories": ";".join(w["evaluation_directory"] for w in group.get("development_evidence", [])),
            "usable_positive_group": group.get("counts_as_usable_positive_group", group["label"] == "retained_original_positive_baseline"),
            "evaluated_window_records": len(windows)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    pilot, replication = load_completed(args.pilot), load_completed(args.replication)
    result = update_registry(read(args.registry), pilot, replication)
    rows = csv_rows(result)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    # The JSON is authoritative; CSV is its one-row-per-pair rendering. Neither
    # operation alters model files, experiment results, or the exploration log.
    for path, content in ((args.registry, json.dumps(result, ensure_ascii=False, indent=2)+"\n"),
                          (args.registry.with_suffix(".csv"), output.getvalue())):
        temporary = path.with_name(path.name+".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    print(json.dumps({"registry": str(args.registry.resolve()),
                      "distinct_model_groups": result["distinct_model_group_count"],
                      "usable_positive_model_groups": result["usable_positive_model_group_count"],
                      "primary_evidence": replication["run"], "development_evidence": pilot["run"]}))


if __name__ == "__main__":
    main()
