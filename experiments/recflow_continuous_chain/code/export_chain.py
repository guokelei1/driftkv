"""Export the actual accepted chain separately from preferred ten-node slots."""
from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from experiments.recflow_continuous_chain.code.validate_chain import BASE, REPO, inspect_manifest, read, resolve
from experiments.recflow_full_population.code.protocol import day_bounds_ms, day_date


def optional(path):
    return read(path) if path and Path(path).is_file() else None


def existing(path):
    return str(Path(path).resolve()) if path and Path(path).is_file() else None


def evidence_paths(directory, original=False):
    if not directory:
        return {}
    directory = resolve(directory)
    names = {"summary": "summary.json", "analysis": "analysis.json", "plan": "plan.json",
             "audit": "independent_audit.json" if original else "audit.json",
             "completion": "completion.json", "decision": "decision.json",
             "chosen_uids": "chosen_uids.npy", "selection": "selection.npy"}
    return {key: existing(directory / filename) for key, filename in names.items()}


def full_population_supplement(directory, source, target, day, data):
    """Verify a descriptive full-cohort attachment without changing acceptance."""
    directory, data = resolve(directory), resolve(data)
    paths = {name: directory / f"{name}.json" for name in ("completion", "analysis", "audit")}
    completion, analysis, audit = (read(paths[name]) for name in paths)

    def require(condition, message):
        if not condition:
            raise ValueError(f"Full-population supplement: {message}")

    require(completion.get("status") == analysis.get("status") == "completed"
            and audit.get("status") == "passed", "completion/analysis/audit has not passed")
    require(resolve(analysis["checkpoint"]) == resolve(target)
            and resolve(analysis["cache_source"]) == resolve(source)
            and resolve(analysis["data"]) == data
            and analysis.get("evaluation_day") == day and analysis.get("history_limit") == 1024
            and analysis.get("evaluation_interval_ms_half_open") == list(day_bounds_ms(day)),
            "model pair, data, history cap or full-day window differs from this edge")
    require(all(resolve(completion[name]) == paths[name] for name in ("analysis", "audit")),
            "completion points at different evidence")
    require(analysis.get("aggregation_type") == completion.get("aggregation_type")
            == "cpu_union_of_three_audited_scoring_runs"
            and analysis.get("independent_confirmation") is False
            and analysis.get("single_full_population_gpu_run") is False,
            "supplement does not preserve its three-run development aggregation semantics")
    with np.load(data / "population.npz") as population:
        train = np.array(population["train"], copy=True)
    require(len(train) == len(np.unique(train)) == 30000, "original population is not 30000 distinct UIDs")
    manifest = read(data / "selection_manifest.json")
    ledger_path = data / manifest["datasets"][f"2:{day}"]["file"]
    ledger = np.load(ledger_path, mmap_mode="r")
    lo, hi = day_bounds_ms(day)
    require(np.isin(ledger["uid"], train).all() and np.all(ledger["day"] == day)
            and np.all((ledger["ts"] >= lo) & (ledger["ts"] < hi)), "full-day ledger has invalid users or times")
    positive = ledger["positive"] > 0
    counts = {"chosen_users": 30000, "requests": len(ledger), "positive_requests": int(positive.sum()),
              "zero_positive_requests": int((~positive).sum()), "active_users": len(np.unique(ledger["uid"])),
              "positive_users": len(np.unique(ledger["uid"][positive]))}
    overall = analysis["overall"]
    require(all(record.get(k) == v for record in (completion, audit, overall) for k, v in counts.items()),
            "full-population counts disagree with the original full-day ledger")
    require(audit.get("candidate_instances") == int((ledger["stop"]-ledger["start"]).sum())
            and audit.get("observed_positive_instances") == int(ledger["positive"].sum())
            and audit.get("exposed_instances") == int(ledger["observed"].sum())
            and resolve(audit["full_selection_path"]) == ledger_path.resolve(), "candidate/label ledger counts differ")
    sources = analysis["sources"]
    source_dirs = [resolve(item["run"]) for item in sources]
    require(len(source_dirs) == len(set(source_dirs)) == 3
            and [resolve(p) for p in completion["source_runs"]] == source_dirs,
            "three distinct actual source runs are not consistently recorded")
    permutation = np.random.default_rng(20261029).permutation(np.sort(train))
    chosen_parts = []
    for item, run, (offset, size) in zip(sources, source_dirs, ((0, 4096), (4096, 8192), (12288, 17712))):
        done, checked, planned, scored = (read(run / f"{n}.json") for n in ("completion", "audit", "plan", "summary"))
        require(done.get("status") == scored.get("status") == "completed" and checked.get("status") == "passed"
                and checked.get("all_chosen_users_complete_day_requests") is True, "a source has not completed its full-day audit")
        require(resolve(planned["checkpoint"]) == resolve(target)
                and resolve(planned["cache_source"]) == resolve(source) and resolve(planned["data"]) == data
                and planned.get("eval_day") == day and planned.get("history_limit") == scored.get("history_limit") == 1024,
                "a source belongs to a different model pair or evaluation protocol")
        require(resolve(item["evidence_paths"]["audit"]) == run / "audit.json"
                and resolve(done["audit"]) == run / "audit.json", "a source references another audit")
        chosen = np.load(run / "chosen_uids.npy")
        require(np.array_equal(chosen, permutation[offset:offset+size]), "source fixed UID interval differs")
        selected = ledger[np.isin(ledger["uid"], chosen)]
        require(np.array_equal(np.load(run / "selection.npy", mmap_mode="r"), selected),
                "source selection does not preserve each chosen user's complete day")
        expected_counts = {"chosen_users": size, "requests": len(selected),
                           "positive_requests": int((selected["positive"] > 0).sum())}
        require(all(record.get(k) == v for record in (done, checked, item["original_overall"])
                    for k, v in expected_counts.items()), "source request counts disagree with its actual population")
        require(checked.get("candidate_instances") == int((selected["stop"]-selected["start"]).sum()),
                "source candidate count differs")
        chosen_parts.append(chosen)
    require(np.array_equal(np.concatenate(chosen_parts), permutation), "source UID union differs from the full population")
    for metric in ("ndcg@10", "recall@10"):
        values, gap = overall["metrics"][metric], overall["comparisons"][metric]["ancestor_cache_loss"]
        rebuild, reuse = values["rebuild"], values["ancestor_reuse"]
        require(np.isfinite([rebuild, reuse, gap["difference"], gap["relative_loss_pct"], *gap["ci95"]]).all()
                and len(gap["ci95"]) == 2 and gap["ci95"][0] <= gap["ci95"][1]
                and abs(rebuild-reuse-gap["difference"]) < 1e-12
                and abs(100*(rebuild-reuse)/rebuild-gap["relative_loss_pct"]) < 1e-10
                and gap["requests"] == counts["positive_requests"] and gap["users"] == counts["positive_users"],
                "supplement metric fields or denominators are inconsistent")
    limitations = list(analysis["limitations"])
    for name, item in analysis.get("predetermined_slices", {}).items():
        for metric in ("ndcg@10", "recall@10"):
            interval = item.get("comparisons", {}).get(metric, {}).get("ancestor_cache_loss", {}).get("ci95")
            if interval and interval[0] <= 0 <= interval[1]:
                limitations.append(f"Full-population slice {name}: nominal {metric} gap CI includes zero ({interval}); retain this uncertainty.")
    return {"directory": str(directory), "verification_status": "verified",
            "use": "Full-population descriptive sensitivity only; never replaces fixed replication primary or prior combined result",
            "evidence_paths": {k: str(v) for k, v in paths.items()},
            "result": analysis, "audit": audit, "limitations": limitations}


def export(manifest_path, output):
    manifest_path, output = manifest_path.resolve(), output.resolve()
    manifest = read(manifest_path)
    audit = inspect_manifest(manifest)  # Existing checks; no new acceptance rule.
    with np.load(resolve(manifest["data"]) / "population.npz") as population:
        cohort_users = len(population["train"])
    declared_nodes = {row.get("id"): row for row in manifest.get("nodes", [])}
    checked_nodes = {row.get("id"): row for row in audit["nodes"]}
    declared_edges = {row.get("id"): row for row in manifest.get("edges", [])}
    checked_edges = {row.get("id"): row for row in audit["edges"]}
    models, edges, supplements, recall_supplements, full_supplements = [], [], {}, {}, {}
    for index in range(10):
        node_id = f"V{index}"
        node, checked = declared_nodes.get(node_id, {}), checked_nodes.get(node_id, {})
        checkpoint = resolve(node["checkpoint"]) if node.get("checkpoint") else None
        sidecar_path = checkpoint.with_suffix(".json") if checkpoint else None
        sidecar = optional(sidecar_path) or {}
        meta = sidecar.get("metadata", {})
        arguments = meta.get("arguments", {})
        completion_path = checkpoint.parent / "completion.json" if checkpoint else None
        completion = optional(completion_path) or {}
        cutoff = checked.get("train_end_day")
        notes = [checked["reason"]] if checked.get("reason") else []
        if completion and meta.get("next_epoch") != completion.get("next_epoch"):
            notes.append("Saved checkpoint epoch differs from final training-stage epoch; phase completion does not replace this selected checkpoint.")
        if index == 0 and arguments.get("parent"):
            parent_meta = (optional(resolve(arguments["parent"]).with_suffix(".json")) or {}).get("metadata", {})
            if parent_meta.get("days"):
                prior_days = parent_meta["days"]
                notes.append(f"training_days_in_saved_phase describes only this {meta.get('phase')} checkpoint. "
                             f"Its actual parent records earlier {parent_meta.get('phase')} D{min(prior_days)}..D{max(prior_days)} "
                             f"({day_date(min(prior_days))}..{day_date(max(prior_days))}); this earlier training is part of the foundation history.")
        models.append({"id": node_id, "label": node.get("label"),
                       "row_role": "actual_accepted_prefix" if index < audit["actual_accepted_model_count"] else "actual_checkpoint_unproven" if checkpoint else "pending_preferred_slot",
                       "verification_status": checked.get("status", "unproven"),
                       "checkpoint": str(checkpoint) if checkpoint else None,
                       "checkpoint_exists": checkpoint.is_file() if checkpoint else None,
                       "checkpoint_sidecar": existing(sidecar_path),
                       "checkpoint_status": sidecar.get("status"),
                       "actual_optimizer_parent": str(resolve(arguments["parent"])) if arguments.get("parent") else None,
                       "training_phase": meta.get("phase"), "training_days_in_saved_phase": meta.get("days"),
                       "training_cutoff_day": cutoff, "training_cutoff_date": day_date(cutoff) if cutoff else None,
                       "listwise_weight": arguments.get("listwise_weight"),
                       "checkpoint_next_epoch": meta.get("next_epoch"), "checkpoint_next_step": meta.get("next_step"),
                       "training_completion_status": completion.get("status"),
                       "training_completion_path": existing(completion_path), "limitations": notes})
    by_node = {row["id"]: row for row in models}
    for index in range(9):
        edge_id = f"V{index}_to_V{index + 1}"
        declared, checked = declared_edges.get(edge_id, {}), checked_edges.get(edge_id, {})
        evidence = checked.get("evidence", {})
        original = evidence.get("kind") == "original_full_population"
        primary = evidence if original else evidence.get("replication", {})
        metrics = checked.get("primary_metrics", {})
        counts = checked.get("primary_population") or primary.get("counts", {})
        paths = evidence_paths(primary.get("directory"), original)
        analysis = optional(paths.get("analysis")) or {}
        recall = analysis.get("overall", {}).get("comparisons", {}).get("recall@10", {}).get("ancestor_cache_loss") or {}
        decision = optional(paths.get("decision")) or {}
        limitations = checked.get("weaknesses", []) + decision.get("weaknesses", [])
        if original and not recall.get("ci95"):
            extra_path = manifest_path.parent / "original_edge1_recall_uncertainty.json"
            extra = optional(extra_path) or {}
            if (extra.get("status") == "completed" and extra.get("metric") == "recall@10"
                    and extra.get("source_evaluation_directory") == primary.get("directory")
                    and extra.get("source_checkpoint") == by_node[f"V{index}"]["checkpoint"]
                    and extra.get("target_checkpoint") == by_node[f"V{index + 1}"]["checkpoint"]
                    and extra.get("evaluation_day") == declared.get("evaluation_day")
                    and all(extra.get(key) == counts.get(key) for key in ("requests", "users", "positive_requests", "positive_users"))
                    and extra.get("rebuild") == metrics.get("rebuild_recall10")
                    and extra.get("reuse") == metrics.get("reuse_recall10")):
                recall = extra.get("paired_bootstrap") or {}
                paths["recall_uncertainty"] = str(extra_path)
                recall_supplements[edge_id] = {"path": str(extra_path), "result": extra}
                limitations.append("Recall CI added by the existing UID bootstrap on this same original full-cohort request set; descriptive uncertainty, not a new confirmation.")
        if original and not recall.get("ci95"):
            limitations.append("Original full-cohort evidence has no saved Recall@10 CI; paired_current_minus contains NDCG@10 only. Recall CIs from sampled cohorts or other model pairs cannot replace it.")
        for note in (checked.get("reason"), declared.get("interpretation_note")):
            if note:
                limitations.append(note)
        combined_path = declared.get("combined_descriptive_sensitivity")
        combined = optional(combined_path)
        if combined_path:
            supplements[edge_id] = {"path": combined_path, "result": combined,
                                    "use": "Descriptive sensitivity only; never replaces replication primary metrics"}
            if combined is None:
                limitations.append("Declared combined descriptive evidence file is missing.")
        pending_slot = checked.get("status") == "pending_preferred_slot" or not by_node[f"V{index + 1}"]["checkpoint"]
        day = None if pending_slot else declared.get("evaluation_day")
        bounds = day_bounds_ms(day) if type(day) is int and 1 <= day <= 37 else (None, None)
        interval = [datetime.fromtimestamp(value / 1000, ZoneInfo("Asia/Shanghai")).isoformat()
                    if value is not None else None for value in bounds]
        ci, recall_ci = metrics.get("ci95") or [None, None], recall.get("ci95") or [None, None]
        rebuild_recall, reuse_recall = metrics.get("rebuild_recall10"), metrics.get("reuse_recall10")
        recall_difference = rebuild_recall - reuse_recall if rebuild_recall is not None and reuse_recall is not None else None
        source, target = f"V{index}", f"V{index + 1}"
        full_directory = declared.get("full_population_sensitivity_directory")
        full = {}
        if full_directory:
            try:
                full = full_population_supplement(full_directory, by_node[source]["checkpoint"],
                                                 by_node[target]["checkpoint"], day, manifest["data"])
            except (ValueError, KeyError, TypeError, OSError) as error:
                full = {"directory": str(resolve(full_directory)), "verification_status": "unverified",
                        "reason": str(error), "result": None, "limitations": [str(error)]}
            full_supplements[edge_id] = full
            limitations.extend(full["limitations"])
        full_overall = (full.get("result") or {}).get("overall", {})
        full_ndcg = full_overall.get("metrics", {}).get("ndcg@10", {})
        full_recall = full_overall.get("metrics", {}).get("recall@10", {})
        full_ndcg_gap = full_overall.get("comparisons", {}).get("ndcg@10", {}).get("ancestor_cache_loss", {})
        full_recall_gap = full_overall.get("comparisons", {}).get("recall@10", {}).get("ancestor_cache_loss", {})
        full_ndcg_ci, full_recall_ci = full_ndcg_gap.get("ci95", [None, None]), full_recall_gap.get("ci95", [None, None])
        edges.append({"id": edge_id, "source": source, "target": target,
                      "row_role": "actual_accepted_prefix" if index < audit["actual_accepted_edge_count"] else "pending_preferred_slot" if pending_slot else "actual_edge_unproven",
                      "source_checkpoint": by_node[source]["checkpoint"], "target_checkpoint": by_node[target]["checkpoint"],
                      "actual_training_path": checked.get("actual_training_path"),
                      "verification_status": checked.get("status", "unproven"), "evaluation_day": day,
                      "evaluation_date": day_date(day) if bounds[0] is not None else None,
                      "window_start_inclusive": interval[0], "window_end_exclusive": interval[1],
                      "window_status": "evaluated" if metrics else "preferred_placeholder_not_scheduled" if pending_slot else "unproven",
                      "preferred_placeholder_day": declared.get("evaluation_day") if pending_slot else None,
                      "primary_population_rule": "original complete training cohort" if original else "replication alone" if primary else None,
                      "rebuild_ndcg10": metrics.get("rebuild_ndcg10"), "reuse_ndcg10": metrics.get("reuse_ndcg10"),
                      "ndcg_rebuild_minus_reuse": metrics.get("rebuild_minus_reuse"),
                      "ndcg_relative_loss_pct": metrics.get("relative_loss_pct"),
                      "ndcg_gap_ci95_low": ci[0], "ndcg_gap_ci95_high": ci[1],
                      "rebuild_recall10": rebuild_recall, "reuse_recall10": reuse_recall,
                      "recall_rebuild_minus_reuse": recall_difference,
                      "recall_relative_loss_pct": 100 * recall_difference / rebuild_recall if rebuild_recall else None,
                      "recall_gap_ci95_low": recall_ci[0], "recall_gap_ci95_high": recall_ci[1],
                      "chosen_users": cohort_users if original else primary.get("chosen_users"),
                      **{key: counts.get(key) for key in ("requests", "positive_requests", "users", "positive_users", "candidate_instances")},
                      "primary_directory": primary.get("directory"), "primary_evidence_paths": paths,
                      "pilot_evidence_paths": evidence_paths(evidence.get("pilot", {}).get("directory")),
                      "combined_descriptive_path": combined_path,
                      "combined_descriptive_ndcg_loss_pct": (combined or {}).get("metrics", {}).get("ndcg@10", {}).get("relative_loss_pct"),
                      "full_population_evidence_directory": full.get("directory"),
                      "full_population_verification_status": full.get("verification_status"),
                      "full_population_evidence_paths": full.get("evidence_paths"),
                      "full_population_rebuild_ndcg10": full_ndcg.get("rebuild"),
                      "full_population_reuse_ndcg10": full_ndcg.get("ancestor_reuse"),
                      "full_population_ndcg_rebuild_minus_reuse": full_ndcg_gap.get("difference"),
                      "full_population_ndcg_relative_loss_pct": full_ndcg_gap.get("relative_loss_pct"),
                      "full_population_ndcg_gap_ci95_low": full_ndcg_ci[0],
                      "full_population_ndcg_gap_ci95_high": full_ndcg_ci[1],
                      "full_population_rebuild_recall10": full_recall.get("rebuild"),
                      "full_population_reuse_recall10": full_recall.get("ancestor_reuse"),
                      "full_population_recall_rebuild_minus_reuse": full_recall_gap.get("difference"),
                      "full_population_recall_relative_loss_pct": full_recall_gap.get("relative_loss_pct"),
                      "full_population_recall_gap_ci95_low": full_recall_ci[0],
                      "full_population_recall_gap_ci95_high": full_recall_ci[1],
                      **{f"full_population_{key}": full_overall.get(key) for key in
                         ("chosen_users", "requests", "positive_requests", "zero_positive_requests", "active_users", "positive_users")},
                      "full_population_candidate_instances": full.get("audit", {}).get("candidate_instances"),
                      "full_population_limitations": full.get("limitations"),
                      "limitations": list(dict.fromkeys(limitations))})
    log_path = manifest_path.parent / "experiment_log.json"
    log = optional(log_path) or {}
    journal = []
    for entry in log.get("completed_experiments", []):
        paths = evidence_paths(entry.get("run"))
        completion, run_audit = optional(paths.get("completion")) or {}, optional(paths.get("audit")) or {}
        decision = optional(paths.get("decision")) or entry.get("decision", {})
        gap = decision.get("result", {}).get("ancestor_cache_loss") or {}
        outcome = None
        if gap.get("difference") is not None:
            outcome = "nonpositive_point" if gap["difference"] <= 0 else "positive_point"
            if gap.get("ci95") and gap["ci95"][0] <= 0 <= gap["ci95"][1]:
                outcome += "_interval_inconclusive"
        journal.append({"recorded_entry": entry, "evidence_paths": paths,
                        "actual_completion_status": completion.get("status"), "actual_audit_status": run_audit.get("status"),
                        "completed_primary_result": completion.get("primary_result"),
                        "descriptive_outcome": outcome, "saved_decision": decision})
    registry_path = manifest.get("previous_six_distinct_pairs")
    registry = optional(registry_path) or {}
    epoch_root = REPO / "experiments/recflow_release_design/runs/epoch2_attempt_1"
    report = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "status": audit["status"],
              "goal_requirements_proven": audit["goal_requirements_proven"], "manifest": str(manifest_path),
              **{key: audit[key] for key in ("structural_and_point_requirements_proven", "proof_scope",
                                           "actual_prefix_structural_and_point_requirements_proven",
                                           "actual_accepted_model_count", "actual_accepted_edge_count",
                                           "preferred_target_achieved", "objective_scope", "data_exhaustion_review", "terminal_basis",
                                           "stability_review_status", "stability_review_required", "stability_review_note", "cli_exit_semantics")},
              "verified_real_model_nodes": audit["verified_real_model_nodes"],
              "verified_positive_edges": audit["verified_positive_edges"],
              "verified_continuous_prefix_edges": audit["verified_continuous_prefix_edges"],
              "models": models, "edges": edges, "current_evidence_inspection": audit,
              "actual_chain_models": models[:audit["actual_accepted_model_count"]],
              "actual_chain_edges": edges[:audit["actual_accepted_edge_count"]],
              "preferred_pending_model_slots": [row for row in models if row["row_role"] == "pending_preferred_slot"],
              "preferred_pending_edge_slots": [row for row in edges if row["row_role"] == "pending_preferred_slot"],
              "planned_release_scenario": manifest.get("planned_release_scenario"),
              "combined_descriptive_sensitivity": supplements,
              "original_recall_uncertainty_supplements": recall_supplements,
              "full_population_descriptive_sensitivity": full_supplements,
              "experiment_log_path": existing(log_path), "completed_explorations": journal,
              "retained_nonpositive_or_inconclusive_results": [
                  {"run": item["recorded_entry"].get("run"), "evidence_paths": item["evidence_paths"],
                   "descriptive_outcome": item["descriptive_outcome"]} for item in journal
                  if item["descriptive_outcome"] and ("nonpositive" in item["descriptive_outcome"]
                                                      or "inconclusive" in item["descriptive_outcome"])],
              "experiment_log_snapshot": log,
              "historical_references_only": {
                  "counting_note": "Historical aggregates are not additional nodes/edges in this chain. Only evidence explicitly linked by this manifest counts.",
                  "previous_six_distinct_pairs": registry_path,
                  "previous_pair_experiment_log": existing(Path(registry_path).parent / "experiment_log.json") if registry_path else None,
                  "all_original_nine_edges": registry.get("all_original_evidence"),
                  "four_extraepoch_counterexamples": registry.get("failed_extraepoch_evidence"),
                  "extraepoch_checkpoint_cleanup": existing(epoch_root / "extra_epoch_checkpoint_cleanup.json")},
              "definitions": {"relative_loss_pct": "100*(Rebuild-Reuse)/Rebuild; positive means Reuse is lower",
                              "ci": "Nominal paired UID-cluster interval for the absolute metric difference, not percent loss",
                              "descriptive_outcome": "Sign and interval description only; does not add an acceptance threshold",
                              "tables": "models/edges and chain_models.csv/chain_edges.csv retain all preferred slots for compatibility; row_role identifies actual accepted prefix vs pending/unproven. actual_chain_models/actual_chain_edges and their named CSV files contain only the verified continuous accepted prefix. Edge actual_training_path preserves unpublished intermediate checkpoints.",
                              "pending_windows": "preferred_placeholder_day is an old count-target placeholder, not an actual release or scheduled evaluation. Actual evaluation_day is null for pending slots; see planned_release_scenario separately.",
                              "missing": "JSON null and empty CSV cell mean missing/unproven, never zero",
                              "training_days": "Days represented in this checkpoint's saved phase; earlier phases follow actual optimizer parents"},
              "limitations": audit["limitations"], "goal_status_changed": False,
              "gpu_work_launched": False, "automatic_successors": False}
    output.mkdir(parents=True, exist_ok=True)
    for name, rows, columns in (
            ("chain_models.csv", models, models), ("chain_edges.csv", edges, edges),
            ("actual_chain_models.csv", report["actual_chain_models"], models),
            ("actual_chain_edges.csv", report["actual_chain_edges"], edges)):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(columns[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)
        (output / name).write_text(buffer.getvalue())
    (output / "chain_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {key: report[key] for key in ("status", "goal_requirements_proven", "structural_and_point_requirements_proven",
                                       "actual_prefix_structural_and_point_requirements_proven", "actual_accepted_model_count",
                                       "actual_accepted_edge_count", "preferred_target_achieved", "terminal_basis",
                                       "stability_review_status", "verified_real_model_nodes", "verified_positive_edges", "verified_continuous_prefix_edges")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=BASE / "chain_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=BASE)
    args = parser.parse_args()
    print(json.dumps(export(args.manifest, args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
