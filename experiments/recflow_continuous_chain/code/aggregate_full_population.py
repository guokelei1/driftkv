"""CPU union of three completed, audited, disjoint full-day scoring samples.

This preserves the sequential development history. It is neither a new GPU run
nor an independent confirmation, and never accepts a model or dispatches work.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from experiments.recflow_full_population.code.batches import RequestBatches
from experiments.recflow_full_population.code.data_common import json_save
from experiments.recflow_full_population.code.protocol import day_bounds_ms, day_date
from experiments.recflow_release_design.code.interval_protocol import ancestry
from .analyze import METRICS, describe_slice, key, metric_rows, read


DATA = Path(__file__).resolve().parents[3] / "experiments/recflow_full_population/data"
SEED = 20261029
COUNT_FIELDS = ("chosen_users", "requests", "positive_requests", "zero_positive_requests",
                "active_users", "positive_users")
PLAN_FIELDS = ("checkpoint", "cache_source", "data", "eval_day", "history_limit", "batch_size",
               "protocol", "primary_metric", "secondary_metric", "evaluation_interval_ms_half_open")


def same(left, right, context):
    """Compare semantic fields and numerical statistics, never fingerprints."""
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError(f"Different fields: {context}")
        for name in left:
            same(left[name], right[name], f"{context}.{name}")
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"Different lengths: {context}")
        for i, (a, b) in enumerate(zip(left, right)):
            same(a, b, f"{context}[{i}]")
    elif isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        if left is None or right is None or not np.isfinite([left, right]).all() or abs(left-right) > 1e-12:
            raise ValueError(f"Different numerical value: {context}: {left!r} != {right!r}")
    elif left != right:
        raise ValueError(f"Different value: {context}: {left!r} != {right!r}")


def completed_documents(run):
    result = {}
    for name, status in (("completion", "completed"), ("audit", "passed"),
                         ("summary", "completed"), ("analysis", "completed")):
        filename = run / f"{name}.json"
        if not filename.is_file():
            raise ValueError(f"Not ready: {filename} does not exist; no aggregate is written")
        result[name] = read(filename)
        if result[name].get("status") != status:
            raise ValueError(f"Not ready: {filename} is not {status}; no aggregate is written")
    return result


def load_context(plan):
    data = Path(plan["data"]).resolve()
    if data != DATA.resolve() or plan.get("history_limit") != 1024:
        raise ValueError("Expected original full-population data and history 1024")
    day = int(plan["eval_day"])
    lineage = ancestry(plan["checkpoint"], plan["cache_source"])
    if not 28 <= day <= 37 or day <= lineage[-1]["train_end_day"]:
        raise ValueError("Expected an actual ancestor and a future available full day")
    # Publication may skip unpublished optimizer checkpoints. ancestry() follows
    # every real parent link; retain and validate those intermediate models too.
    for entry in lineage:
        metadata = read(Path(entry["checkpoint"]).with_suffix(".json"))["metadata"]
        architecture = metadata["proposal"]["architecture"]
        expected = {"layers": 16, "hidden_size": 320, "heads": 10, "head_dim": 32}
        if (metadata["scale"] != "max" or any(architecture["max"][k] != v for k, v in expected.items())
                or architecture["max_seq_len_including_query"] != 1025
                or metadata["proposal"]["history"]["max_events"] != 1024
                or Path(metadata["arguments"]["data"]).resolve() != data):
            raise ValueError("Checkpoint is not the agreed full-population Max/1024 model")
    with np.load(data / "population.npz") as population:
        train = np.array(population["train"], copy=True)
    if train.ndim != 1 or len(train) != 30000 or len(np.unique(train)) != 30000:
        raise ValueError("Expected the original 30000 distinct training UIDs")
    permutation = np.random.default_rng(SEED).permutation(np.sort(train))
    dataset = RequestBatches(data, [day], role=2, history_limit=1024)
    full = dataset.records
    lo, hi = day_bounds_ms(day)
    if (not np.isin(full["uid"], train).all() or not np.all(full["day"] == day)
            or np.any((full["ts"] < lo) | (full["ts"] >= hi))):
        raise ValueError("Original full-day selection has an invalid UID or date")
    identities = [key(row) for row in full]
    if len(set(identities)) != len(full):
        raise ValueError("Original full-day selection has duplicate requests")
    return {"data": data, "day": day, "lineage": lineage, "permutation": permutation,
            "full": full, "identities": identities, "selection_path": dataset._selection_path,
            "plan": plan, "bounds": (lo, hi)}


def slices_for(records, rows, chosen, bounds):
    identities = [key(row) for row in records]
    overall = describe_slice(identities, rows, None, len(chosen))
    slices = {}
    for name, users in (("chosen_uid_positions_even", chosen[::2]),
                        ("chosen_uid_positions_odd", chosen[1::2])):
        users = set(map(int, users))
        local = [identity for identity in identities if identity[0] in users]
        slices[name] = describe_slice(local, rows, None, len(users))
    lo, hi = bounds
    middle = lo + (hi-lo)//2
    for name, start, stop in (("first_12_hours", lo, middle), ("second_12_hours", middle, hi)):
        local = [identity for identity, row in zip(identities, records) if start <= int(row["ts"]) < stop]
        slices[name] = describe_slice(local, rows, None, len(chosen))
        slices[name]["timestamp_interval_ms_half_open"] = [start, stop]
    return overall, slices


def check_statistics(actual, saved, context):
    for name in COUNT_FIELDS:
        same(actual[name], saved[name], f"{context}.{name}")
    for metric in METRICS:
        for method in ("rebuild", "ancestor_reuse"):
            same(actual["metrics"][metric][method], saved["metrics"][metric][method],
                 f"{context}.{metric}.{method}")
    for metric in ("ndcg@10", "recall@10"):
        same(actual["comparisons"][metric]["ancestor_cache_loss"],
             saved["comparisons"][metric]["ancestor_cache_loss"], f"{context}.{metric}.gap")


def validate_source(run, context, documents=None):
    """Read one completed source; also usable for a read-only partial check."""
    run = Path(run).resolve()
    docs = documents or completed_documents(run)
    plan = read(run / "plan.json")
    completion, audit, summary, analysis = (docs[n] for n in ("completion", "audit", "summary", "analysis"))
    for field in PLAN_FIELDS:
        same(plan[field], context["plan"][field], f"{run.name}.plan.{field}")
        same(plan[field], summary["plan"][field], f"{run.name}.summary.plan.{field}")
    for evidence in (plan, summary, analysis, audit):
        same(evidence["actual_training_path"], context["lineage"], f"{run.name}.lineage")
    for field in ("checkpoint", "cache_source"):
        same(analysis[field], plan[field], f"{run.name}.analysis.{field}")
    same(analysis["evaluation_day"], context["day"], "analysis day")
    same(analysis["evaluation_interval_ms_half_open"], list(context["bounds"]), "analysis window")
    if (summary.get("history_limit") != 1024 or summary.get("precision") != "float32_tf32_disabled"
            or not audit.get("all_chosen_users_complete_day_requests")):
        raise ValueError("Source lacks the required audited evaluation protocol")
    for field in ("audit", "analysis"):
        if Path(completion[field]).resolve() != run / f"{field}.json":
            raise ValueError("Completion points to a different source evidence file")
    selection = plan["selection"]
    same(selection, summary["plan"]["selection"], "recorded selection")
    for field, filename in (("selection_path", "selection.npy"), ("chosen_uids_path", "chosen_uids.npy"),
                            ("uid_permutation_path", "uid_permutation.npy")):
        if Path(selection[field]).resolve() != run / filename:
            raise ValueError("Source selection path does not bind to this run")
    if (selection["seed"] != SEED or selection["history_limit"] != 1024
            or selection["population_users"] != 30000 or selection["activity_or_feedback_selection"] is not False
            or selection["day"] != context["day"] or Path(selection["data"]).resolve() != context["data"]):
        raise ValueError("Source population protocol differs")
    chosen = np.load(run / "chosen_uids.npy")
    offset, count = selection["user_offset"], selection["chosen_users"]
    if (offset < 0 or count < 1 or offset+count > 30000
            or not np.array_equal(np.load(run / "uid_permutation.npy"), context["permutation"])
            or not np.array_equal(chosen, context["permutation"][offset:offset+count])):
        raise ValueError("Source does not use its declared fixed permutation interval")
    records = np.load(run / "selection.npy", mmap_mode="r")
    expected = context["full"][np.isin(context["full"]["uid"], chosen)]
    if records.dtype != expected.dtype or not np.array_equal(records, expected):
        raise ValueError("Source selection is not every chosen user's original full-day request")
    rows, shards = metric_rows(run)
    if (rows.keys() != {key(row) for row in records} or len(shards) != summary["world_size"]
            or len(shards) != audit["metric_files"]
            or sum(1 for _ in run.glob("rank_*_scores/batch_*.npz")) != audit["raw_score_files"]):
        raise ValueError("Audited raw/metric files or complete request identities are missing")
    required = {"uid", "rid", "day", "candidates", "exposed", "positives", "history_length"}
    required |= {f"{method}_{metric}" for method in ("current", "reuse") for metric in METRICS}
    for record in records:
        row = rows[key(record)]
        if row.keys() != required:
            raise ValueError("Unexpected per-request CSV fields")
        expected_fields = {"candidates": int(record["stop"]-record["start"]),
                           "exposed": int(record["observed"]), "positives": int(record["positive"]),
                           "history_length": min(1024, int(record["history_length"]))}
        for name, value in expected_fields.items():
            same(row[name], value, f"{run.name}.{key(record)}.{name}")
        for method in ("current", "reuse"):
            for metric in METRICS:
                value = row[f"{method}_{metric}"]
                if (row["positives"] > 0 and (value is None or not np.isfinite(value) or not 0 <= value <= 1)
                        or row["positives"] == 0 and value is not None):
                    raise ValueError("Invalid metric value or positive-request denominator")
    overall, slices = slices_for(records, rows, chosen, context["bounds"])
    check_statistics(overall, analysis["overall"], f"{run.name}.overall")
    for name in slices:
        check_statistics(slices[name], analysis["predetermined_slices"][name], f"{run.name}.{name}")
    for name in COUNT_FIELDS:
        same(overall[name], audit[name], f"{run.name}.audit.{name}")
        if name in completion:
            same(overall[name], completion[name], f"{run.name}.completion.{name}")
    for name in ("requests", "positive_requests", "zero_positive_requests", "positive_users"):
        same(overall[name], summary[name], f"{run.name}.summary.{name}")
    same(overall["active_users"], summary["users"], "summary active users")
    same(overall["requests"], selection["requests"], "selected requests")
    same(selection["full_window_requests"], len(context["full"]), "full window requests")
    for metric in METRICS:
        for name, method in (("rebuild", "current"), ("ancestor_reuse", "reuse")):
            same(overall["metrics"][metric][name], summary["metrics"][method][metric], "summary mean")
    same(completion["primary_result"], analysis["overall"]["comparisons"]["ndcg@10"], "completion primary")
    same(audit["candidate_instances"], int((records["stop"]-records["start"]).sum()), "candidate count")
    same(audit["observed_positive_instances"], int(records["positive"].sum()), "positive instance count")
    evidence = {"run": str(run), "permutation_offset": offset, "chosen_users": count,
                "source_completed_at": completion["completed_at"],
                "evidence_paths": {name: str(run / f"{name}.json") for name in docs},
                "metric_files": [str(p) for p in shards], "raw_score_files": audit["raw_score_files"],
                "candidate_instances": audit["candidate_instances"],
                "original_overall": analysis["overall"], "original_predetermined_slices": analysis["predetermined_slices"],
                "original_uncertainty": analysis["uncertainty"], "elapsed": analysis["elapsed"],
                "recomputed_source_statistics_match": True,
                "decision_path": str(run / "decision.json") if (run / "decision.json").is_file() else None}
    signature = {name: summary[name] for name in ("history_limit", "precision", "protocol", "world_size")}
    return rows, chosen, evidence, signature


def aggregate(runs):
    started = time.monotonic()
    runs = [Path(run).resolve() for run in runs]
    if len(runs) != 3 or len(set(runs)) != 3:
        raise ValueError("Exactly three distinct source runs are required")
    # Check every completion before opening any source's per-request scores.
    documents = [completed_documents(run) for run in runs]
    context = load_context(read(runs[0] / "plan.json"))
    rows, chosen_parts, sources, signatures = {}, [], [], []
    for run, docs, expected_interval in zip(runs, documents, ((0, 4096), (4096, 8192), (12288, 17712))):
        local, chosen, source, signature = validate_source(run, context, docs)
        if (source["permutation_offset"], len(chosen)) != expected_interval:
            raise ValueError("Expected pilot 4096, disjoint replication 8192, and remaining 17712")
        if rows.keys() & local.keys() or any(np.intersect1d(chosen, previous).size for previous in chosen_parts):
            raise ValueError("Source populations or requests overlap")
        rows.update(local)
        chosen_parts.append(chosen)
        sources.append(source)
        signatures.append(signature)
    if not np.array_equal(np.concatenate(chosen_parts), context["permutation"]):
        raise ValueError("Chosen users do not exactly cover the original fixed 30000 permutation")
    if rows.keys() != set(context["identities"]):
        raise ValueError("Union does not cover the original full-day request ledger exactly once")
    for signature in signatures[1:]:
        same(signature, signatures[0], "source evaluation configurations")
    overall, slices = slices_for(context["full"], rows, context["permutation"], context["bounds"])
    full = context["full"]
    audit = {"status": "passed", **{k: overall[k] for k in COUNT_FIELDS},
             "candidate_instances": int((full["stop"]-full["start"]).sum()),
             "observed_positive_instances": int(full["positive"].sum()),
             "exposed_instances": int(full["observed"].sum()),
             "all_chosen_users_complete_day_requests": True,
             "source_uids_pairwise_disjoint": True, "union_is_original_fixed_30000": True,
             "union_matches_complete_day_selection_ledger": True,
             "csv_identity_candidate_feedback_history_fields_match_original_ledger": True,
             "all_source_overall_and_slice_statistics_reproduced": True,
             "all_source_candidate_audits_passed": True,
             "candidate_audit_scope": "Reuse completed independent raw-candidate/label/metric audits; aggregation rereads CSV and original request ledger but does not rerank raw candidate scores.",
             "source_audits": [s["evidence_paths"]["audit"] for s in sources],
             "full_selection_path": str(context["selection_path"]),
             "population_path": str(context["data"] / "population.npz"),
             "uid_permutation_seed": SEED, "actual_training_path": context["lineage"],
             "evaluation_configuration": signatures[0]}
    limitations = [
        "Three completed GPU scoring runs are combined on their disjoint users; no single 30000-user GPU run was performed.",
        "Expansion to the remaining users was chosen after inspecting the pilot and replication uncertainty. This is sequential development evidence, not an untouched independent confirmation or a new training seed.",
        "The final result uses every original fixed training UID and every corresponding day request; no population, time slice, metric or sign is selected for reporting.",
        "The 95% paired UID-cluster bootstrap intervals use seed 17 and 2000 draws; they are nominal and do not adjust for sequential looks, recipe search or multiple slices.",
        "UID halves and 12-hour slices are fixed descriptive sensitivity checks. Users, weights and dates are shared with earlier experiments; this is not independent retraining or temporal confirmation.",
        "Only positive requests enter NDCG/Recall macro means. Inactive chosen users and zero-positive requests remain in coverage counts, not the metric denominator.",
        "Observed-positive ranking over all logged candidates and a matched-prefix cache snapshot do not represent unbiased relevance labels or persistent serving.",
        "All source results and prior intervals are retained. This aggregation does not accept a chain edge or modify its primary evidence automatically."
    ]
    plan = context["plan"]
    prior_combined = [{"path": str(run / "combined_cohort_diagnostic.json"),
                       "original_record": read(run / "combined_cohort_diagnostic.json")}
                      for run in runs if (run / "combined_cohort_diagnostic.json").is_file()]
    analysis = {"status": "completed", "aggregation_type": "cpu_union_of_three_audited_scoring_runs",
                "checkpoint": plan["checkpoint"], "cache_source": plan["cache_source"],
                "immediate_optimizer_parent": context["lineage"][-1]["actual_parent"],
                "actual_training_path": context["lineage"], "data": str(context["data"]),
                "history_limit": 1024, "evaluation_day": context["day"],
                "evaluation_date": day_date(context["day"]),
                "evaluation_interval_ms_half_open": list(context["bounds"]),
                "primary_metric": "positive-request macro observed-positive NDCG@10",
                "secondary_metric": "positive-request macro Recall@10",
                "positive_gap_definition": "Rebuild minus Reuse; positive means reuse loses ranking quality",
                "overall": overall, "predetermined_slices": slices, "sources": sources,
                "prior_combined_diagnostics": prior_combined,
                "slice_rule": "Positions even/odd in the original fixed 30000 UID permutation, and local-day [00:00,12:00)/[12:00,24:00); retain all slices",
                "uncertainty": "Nominal paired UID-cluster bootstrap, seed 17, 2000 draws; sequential development aggregation",
                "limitations": limitations, "independent_confirmation": False,
                "single_full_population_gpu_run": False, "all_signs_retained": True,
                "cpu_aggregation_wall_seconds": time.monotonic()-started,
                "source_scoring_wall_seconds_sum": sum(s["elapsed"]["gpu_evaluator_wall_seconds"] for s in sources),
                "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    return analysis, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--remaining", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Preserve existing output; choose a fresh aggregate directory")
    analysis, audit = aggregate([args.pilot, args.replication, args.remaining])
    output.mkdir(parents=True, exist_ok=False)
    json_save(output / "audit.json", audit)
    json_save(output / "analysis.json", analysis)
    completion = {"status": "completed", "completed_at": analysis["completed_at"],
                  "aggregation_type": analysis["aggregation_type"],
                  "audit": str(output / "audit.json"), "analysis": str(output / "analysis.json"),
                  **{k: analysis["overall"][k] for k in COUNT_FIELDS},
                  "primary_result": analysis["overall"]["comparisons"]["ndcg@10"],
                  "source_runs": [s["run"] for s in analysis["sources"]],
                  "source_scoring_run_count": 3, "new_gpu_run": False,
                  "independent_confirmation": False, "model_accepted": False,
                  "next_experiment_dispatched": False}
    json_save(output / "completion.json", completion)
    print(json.dumps({"output": str(output), "completion": completion,
                      "metrics": analysis["overall"]["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
