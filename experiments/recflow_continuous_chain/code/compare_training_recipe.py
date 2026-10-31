"""Compare completed different-target recipes with one real optimizer parent.

This is a CPU paired quality comparison, not a same-checkpoint parity check.
It reads already audited scores and never dispatches training or evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from experiments.recflow_full_population.code.data_common import json_save
from experiments.recflow_full_population.code.protocol import day_bounds_ms, day_date, paired_uid_interval
from .analyze import INTEGER_FIELDS, METRICS, key, metric_rows, read


def path(value):
    return Path(value).resolve()


def argument_differences(original, new):
    return {name: {"baseline": original.get(name), "new": new.get(name)}
            for name in sorted(original.keys() | new.keys())
            if original.get(name) != new.get(name)}


def evaluation_protocol(directory, summary):
    """Read semantic score bindings from both historical summary formats."""
    directory = path(directory)
    args = summary.get("arguments") or {}
    embedded = summary.get("plan") or {}
    actual = read(directory / "plan.json") if (directory / "plan.json").is_file() else {}
    candidates = {
        "checkpoint": [summary.get("checkpoint"), args.get("checkpoint"),
                       embedded.get("checkpoint"), actual.get("checkpoint")],
        "cache_source": [summary.get("parent_checkpoint"), summary.get("cache_source"),
                         args.get("parent"), embedded.get("cache_source"), actual.get("cache_source")],
        "data": [summary.get("data"), args.get("data"), embedded.get("data"), actual.get("data")],
        "days": [summary.get("days"), args.get("days"),
                 [embedded["eval_day"]] if "eval_day" in embedded else None,
                 [actual["eval_day"]] if "eval_day" in actual else None],
    }
    result = {}
    for name, values in candidates.items():
        values = [value for value in values if value is not None]
        if not values:
            raise ValueError(f"Evaluation lacks its actual {name} binding")
        if name != "days":
            values = [path(value) for value in values]
        elif any(not isinstance(value, list) or not value
                 or any(type(day) is not int for day in value) for value in values):
            raise ValueError("Evaluation has invalid recorded days")
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Evaluation summary and recorded plan disagree on {name}")
        result[name] = values[0]
    result["layout"] = "continuous_sample" if embedded or actual else "full_population"
    if result["layout"] == "continuous_sample":
        completed = read(directory / "completion.json")
        if (completed.get("status") != "completed"
                or any(completed.get(name) != summary.get(name)
                       for name in ("requests", "positive_requests"))):
            raise ValueError("Sampled evaluation completion does not match its scored summary")
    return result


def checkpoint_metadata(checkpoint):
    checkpoint = path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError("A compared target checkpoint is missing")
    sidecar = read(checkpoint.with_suffix(".json"))
    metadata = sidecar.get("metadata", {})
    if sidecar.get("status") != "committed" or metadata.get("quality_use_permitted") is False:
        raise ValueError("A compared endpoint is incomplete or diagnostic-only")
    if not metadata.get("arguments", {}).get("parent") or not metadata.get("days"):
        raise ValueError("Endpoint lacks its actual parent or training-day record")
    training_plan = read(checkpoint.parent / "plan.json")
    if training_plan.get("arguments") != metadata["arguments"]:
        raise ValueError("Saved target arguments differ from the actual training plan")
    training_completion = read(checkpoint.parent / "completion.json")
    if training_completion.get("status") != "completed":
        raise ValueError("A compared training run is not complete")
    return metadata, training_plan


def training_selection_comparison(baseline, new):
    left, right = baseline.parent / "request_selection.npz", new.parent / "request_selection.npz"
    with np.load(left) as original, np.load(right) as current:
        if set(original.files) != set(current.files):
            raise ValueError("Training selection fields differ")
        fields = {name: bool(np.array_equal(original[name], current[name])) for name in original.files}
        requests = len(current["uid"])
    return {"baseline_selection": str(left), "new_selection": str(right),
            "requests": requests, "field_element_equality": fields,
            "all_recorded_training_selection_fields_equal": all(fields.values())}


def compare_metric(uids, new_rebuild, new_reuse, old_rebuild, old_reuse):
    new_rebuild, new_reuse, old_rebuild, old_reuse = map(
        lambda x: np.asarray(x, dtype=np.float64), (new_rebuild, new_reuse, old_rebuild, old_reuse))
    new_gap, old_gap = new_rebuild - new_reuse, old_rebuild - old_reuse
    zeros = np.zeros_like(new_gap)
    new_loss = paired_uid_interval(uids, new_gap, zeros)
    old_loss = paired_uid_interval(uids, old_gap, zeros)
    new_loss["relative_loss_pct"] = float(100 * new_gap.mean() / new_rebuild.mean()) if new_rebuild.mean() else None
    old_loss["relative_loss_pct"] = float(100 * old_gap.mean() / old_rebuild.mean()) if old_rebuild.mean() else None
    return {"new_rebuild": float(new_rebuild.mean()), "baseline_rebuild": float(old_rebuild.mean()),
            "new_reuse": float(new_reuse.mean()), "baseline_reuse": float(old_reuse.mean()),
            "rebuild_new_minus_baseline": paired_uid_interval(uids, new_rebuild, old_rebuild),
            "reuse_new_minus_baseline": paired_uid_interval(uids, new_reuse, old_reuse),
            "new_same_parent_cache_loss": new_loss, "baseline_same_parent_cache_loss": old_loss,
            "change_in_cache_loss": paired_uid_interval(uids, new_gap, old_gap)}


def compare(run, baseline):
    started = time.monotonic()
    run, baseline = path(run), path(baseline)
    completed = read(run / "completion.json")
    new_audit, new_summary, plan = read(run / "audit.json"), read(run / "summary.json"), read(run / "plan.json")
    old_summary = read(baseline / "summary.json")
    old_audit_path = baseline / "independent_audit.json"
    if not old_audit_path.exists():
        old_audit_path = baseline / "audit.json"
    old_audit = read(old_audit_path)
    if (completed.get("status") != "completed" or new_summary.get("status") != "completed"
            or old_summary.get("status") not in ("passed", "completed")
            or new_audit.get("status") != "passed" or old_audit.get("status") != "passed"):
        raise ValueError("Both scored runs and their independent candidate/metric audits must be complete")
    for summary, audit in ((new_summary, new_audit), (old_summary, old_audit)):
        if any(summary.get(name) != audit.get(name) for name in ("requests", "positive_requests")):
            raise ValueError("Completed audit and summary disagree on request coverage")
    if old_audit.get("evaluation") and path(old_audit["evaluation"]) != baseline:
        raise ValueError("Baseline audit refers to another evaluation")
    new_protocol = evaluation_protocol(run, new_summary)
    old_protocol = evaluation_protocol(baseline, old_summary)
    new_target, old_target = new_protocol["checkpoint"], old_protocol["checkpoint"]
    if new_target == old_target:
        raise ValueError("Recipe comparison requires different target checkpoints")
    new_meta, new_training = checkpoint_metadata(new_target)
    old_meta, old_training = checkpoint_metadata(old_target)
    new_parent, old_parent = path(new_meta["arguments"]["parent"]), path(old_meta["arguments"]["parent"])
    if (new_parent != old_parent or new_parent != new_protocol["cache_source"]
            or old_parent != old_protocol["cache_source"]):
        raise ValueError("Both recipes and Reuse evaluations must share their actual immediate optimizer parent")
    if new_meta["days"] != old_meta["days"]:
        raise ValueError("Training windows differ")
    day = int(plan["eval_day"])
    if (old_protocol["days"] != [day] or new_protocol["days"] != [day]
            or day <= max(new_meta["days"])):
        raise ValueError("Evaluation windows differ or do not follow training")
    new_data, old_data = path(new_meta["arguments"]["data"]), path(old_meta["arguments"]["data"])
    if (new_data != old_data or new_data != new_protocol["data"]
            or old_protocol["data"] != old_data
            or new_summary.get("history_limit") != 1024 or old_summary.get("history_limit") != 1024):
        raise ValueError("Data root or history protocol differs")
    rows, shards = metric_rows(run)
    selected = np.load(run / "selection.npy", mmap_mode="r")
    chosen = np.load(run / "chosen_uids.npy")
    expected = {key(row) for row in selected}
    if (len(expected) != len(selected) or expected != rows.keys()
            or len(rows) != new_audit["requests"] or len(shards) != new_summary["world_size"]):
        raise ValueError("New metric files do not cover exactly the selected audited requests")
    originals, _ = metric_rows(baseline, expected)
    if originals.keys() != rows.keys():
        raise ValueError("Baseline lacks a selected request")
    for identity, current in rows.items():
        original = originals[identity]
        if any(current[name] != original[name] for name in INTEGER_FIELDS):
            raise ValueError("Selected request, candidate count, feedback count or history length differs")
        for method in ("current", "reuse"):
            for metric in METRICS:
                for record in (current, original):
                    value = record[f"{method}_{metric}"]
                    if (record["positives"] > 0 and (value is None or not np.isfinite(value))) or (record["positives"] == 0 and value is not None):
                        raise ValueError("Undefined or nonfinite metric disagrees with actual feedback")
    valid = [identity for identity in rows if rows[identity]["positives"] > 0]
    if not valid or len(valid) != new_audit["positive_requests"]:
        raise ValueError("Compared valid-request denominator differs")
    uids = [rows[identity]["uid"] for identity in valid]
    metrics = {}
    for metric in ("ndcg@10", "recall@10"):
        metrics[metric] = compare_metric(
            uids, [rows[k][f"current_{metric}"] for k in valid], [rows[k][f"reuse_{metric}"] for k in valid],
            [originals[k][f"current_{metric}"] for k in valid], [originals[k][f"reuse_{metric}"] for k in valid])
        for field, method in (("new_rebuild", "current"), ("new_reuse", "reuse")):
            if abs(metrics[metric][field] - new_summary["metrics"][method][metric]) > 1e-12:
                raise ValueError("New matched means differ from completed new summary")
    differences = argument_differences(old_meta["arguments"], new_meta["arguments"])
    administrative = {"output", "log_every", "checkpoint_every", "wall_hours", "budget_spent_seconds"}
    selections = training_selection_comparison(old_target, new_target)
    if not selections["all_recorded_training_selection_fields_equal"]:
        raise ValueError("Training samples/order differ; this is not the declared matched recipe comparison")
    return {"status": "completed", "comparison_type": "different_target_matched_training_recipe",
            "run": str(run), "baseline": str(baseline), "new_checkpoint": str(new_target),
            "baseline_checkpoint": str(old_target), "actual_shared_parent": str(new_parent),
            "training_days": new_meta["days"], "evaluation_day": day, "evaluation_date": day_date(day),
            "evaluation_interval_ms_half_open": list(day_bounds_ms(day)),
            "counts": {"chosen_users": len(chosen), "active_users": len({row["uid"] for row in rows.values()}),
                       "requests": len(rows), "positive_requests": len(valid), "positive_users": len(set(uids)),
                       "zero_positive_requests": len(rows) - len(valid),
                       "candidate_instances": sum(row["candidates"] for row in rows.values()),
                       "baseline_evaluation_requests": old_summary["requests"],
                       "baseline_full_window_requests": old_summary["requests"] if old_protocol["layout"] == "full_population" else None},
            "baseline_evaluation_layout": old_protocol["layout"],
            "metrics": metrics,
            "training": {"baseline_arguments": old_meta["arguments"], "new_arguments": new_meta["arguments"],
                         "all_argument_differences": differences,
                         "optimization_or_data_argument_differences": {k: v for k, v in differences.items() if k not in administrative},
                         "baseline_requests": old_training["requests"], "new_requests": new_training["requests"],
                         "baseline_steps": old_training["planned_steps"], "new_steps": new_training["planned_steps"],
                         "selection_comparison": selections},
            "completed_audits": {"new": str(run / "audit.json"), "baseline": str(old_audit_path)},
            "interpretation": "A changed-target quality comparison with the same parent and matched future requests; not numerical native parity or an independent retraining-seed control",
            "uncertainty": "Nominal paired user-cluster intervals; seeds and samples are matched but atomic training kernels do not guarantee bitwise reproducibility",
            "difference_conventions": {"quality_change": "new minus baseline", "cache_loss": "Rebuild minus Reuse", "change_in_cache_loss": "new cache loss minus baseline cache loss"},
            "all_signs_retained": True, "wall_seconds": time.monotonic() - started,
            "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else args.run.resolve() / "recipe_comparison.json"
    if output.exists():
        raise FileExistsError("Preserve the existing comparison; use another explicit output path")
    result = compare(args.run, args.baseline)
    json_save(output, result)
    print(json.dumps({"status": result["status"], "output": str(output), "counts": result["counts"],
                      "metrics": result["metrics"], "wall_seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
