"""Check real chain structure, ancestry and positive point-estimate evidence.

This checker does not determine per-edge stability or prove the complete user
goal. CLI exit 2 means ongoing/incomplete/contradicted evidence; exit 3 means
the preferred count, or an externally reviewed data-exhaustion endpoint, has
verified structure and positive points. Separate stability/effect-size review
is still needed. A verified shorter prefix alone is not a terminal result.

CPU semantic inspection only: this never launches work, changes goal status or
checks file fingerprints. A manifest is a claim, not evidence of completion.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from experiments.recflow_full_population.code.protocol import day_bounds_ms


BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
DATA = REPO / "experiments/recflow_full_population/data"
PROTOCOL = "matched_causal_prefix_snapshot_complete_candidates"


class Unproven(ValueError):
    """Required work or evidence is still missing, without claiming a result."""


def read(path):
    path = Path(path)
    if not path.is_file():
        raise Unproven(f"Missing evidence file: {path}")
    return json.loads(path.read_text())


def resolve(path):
    if not path:
        raise Unproven("No actual checkpoint/evidence path has been supplied")
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def equal_number(a, b):
    return (math.isfinite(float(a)) and math.isfinite(float(b))
            and math.isclose(float(a), float(b), rel_tol=0., abs_tol=1e-12))


def native_quality_diagnostic(summary):
    current = summary["metrics"]["current"]["ndcg@10"]
    previous = summary["metrics"].get("parent", {}).get("ndcg@10")
    return {"target_native_ndcg10": current, "previous_native_ndcg10": previous,
            "target_minus_previous_native": current-previous if previous is not None else None,
            "previous_native_available": previous is not None,
            "acceptance_condition": False,
            "note": "Native quality is diagnostic only; improvement over the previous model is not required."}


class Inspector:
    def __init__(self):
        self.models = {}
        self.selections = {}
        with np.load(DATA / "population.npz") as pop:
            self.train = np.array(pop["train"], copy=True)
        self.permutation = np.random.default_rng(20261029).permutation(np.sort(self.train))

    def model(self, path):
        path = resolve(path)
        if str(path) in self.models:
            return self.models[str(path)]
        if not path.is_file():
            raise Unproven(f"Actual model checkpoint not present: {path}")
        sidecar = read(path.with_suffix(".json"))
        if sidecar.get("status") != "committed":
            raise Unproven(f"Checkpoint save not committed: {path}")
        meta = sidecar["metadata"]
        if meta.get("quality_use_permitted") is False or meta.get("scale") != "max":
            raise ValueError(f"Not a quality-permitted Max model: {path}")
        args = meta.get("arguments", {})
        if resolve(args.get("data")) != DATA.resolve():
            raise ValueError("Checkpoint uses a different data/vocabulary/population lineage")
        proposal = meta["proposal"]
        architecture = proposal["architecture"]
        expected = {"layers": 16, "hidden_size": 320, "heads": 10, "head_dim": 32}
        if (any(architecture["max"].get(key) != value for key, value in expected.items())
                or architecture.get("max_seq_len_including_query") != 1025
                or proposal.get("history", {}).get("max_events") != 1024):
            raise ValueError("Checkpoint does not implement the agreed Max/1024-history task")
        days = meta.get("days")
        if not days or any(type(day) is not int or not 1 <= day <= 36 for day in days):
            raise ValueError("Checkpoint lacks valid actual training days")
        parent = str(resolve(args["parent"])) if args.get("parent") else None
        record = {"checkpoint": str(path), "sidecar": str(path.with_suffix(".json")),
                  "train_days": days, "train_end_day": max(days), "actual_parent": parent,
                  "scale": meta["scale"], "history_limit": 1024, "architecture": architecture}
        self.models[str(path)] = record
        return record

    def ancestry(self, target, source):
        """Follow actual optimizer-parent paths; no sibling/branch substitution."""
        source, current = str(resolve(source)), str(resolve(target))
        reverse, seen = [], set()
        if source == current:
            raise ValueError("Source and target are the same checkpoint")
        while current:
            if current in seen:
                raise ValueError("Cycle in actual optimizer ancestry")
            seen.add(current)
            item = self.model(current)
            if reverse:
                if item["train_end_day"] > reverse[-1]["train_end_day"]:
                    raise ValueError("Ancestor training cutoff exceeds its descendant")
                if item["architecture"] != reverse[-1]["architecture"]:
                    raise ValueError("Architecture changes inside the purported chain")
            reverse.append(item)
            if current == source:
                return list(reversed(reverse))
            current = item["actual_parent"]
        raise ValueError("Cache source is not an actual optimizer ancestor; this would splice branches")

    def full_selection(self, day):
        if day not in self.selections:
            records = np.load(DATA / f"selections/role2_d{day}_d{day}.npy", mmap_mode="r")
            lo, hi = day_bounds_ms(day)
            if (not np.isin(records["uid"], self.train).all()
                    or not np.all(records["day"] == day)
                    or np.any((records["ts"] < lo) | (records["ts"] >= hi))):
                raise ValueError("Original complete-day selection is inconsistent with population/time")
            self.selections[day] = records
        return self.selections[day]

    def counts(self, records):
        valid = records[records["positive"] > 0]
        return {"requests": len(records), "positive_requests": len(valid),
                "users": len(np.unique(records["uid"])),
                "positive_users": len(np.unique(valid["uid"])),
                "candidate_instances": int((records["stop"] - records["start"]).sum())}

    def metrics(self, summary, paired=None):
        metrics = summary["metrics"]
        rebuild, reuse = metrics["current"]["ndcg@10"], metrics["reuse"]["ndcg@10"]
        if not all(math.isfinite(float(x)) for x in (rebuild, reuse)) or rebuild <= 0:
            raise ValueError("NDCG values are nonfinite or denominator is zero")
        paired = paired or summary["paired_current_minus"]["reuse"]
        if not equal_number(rebuild-reuse, paired["difference"]):
            raise ValueError("Recorded paired difference disagrees with metric means")
        ci = paired["ci95"]
        if len(ci) != 2 or not all(math.isfinite(float(x)) for x in ci) or ci[0] > ci[1]:
            raise ValueError("Invalid recorded user-cluster interval")
        if (paired["requests"] != summary["positive_requests"]
                or paired["users"] != summary["positive_users"]):
            raise ValueError("Paired metric population differs from the summary")
        recall = [metrics[name]["recall@10"] for name in ("current", "reuse")]
        if not all(math.isfinite(float(x)) for x in recall):
            raise ValueError("Recall values are nonfinite")
        return {"rebuild_ndcg10": rebuild, "reuse_ndcg10": reuse,
                "rebuild_minus_reuse": paired["difference"],
                "relative_loss_pct": 100*(rebuild-reuse)/rebuild, "ci95": ci,
                "rebuild_recall10": recall[0], "reuse_recall10": recall[1],
                "point_positive": rebuild > reuse, "ci_excludes_zero_positive": ci[0] > 0}

    def original(self, directory, source, target, day):
        directory = resolve(directory)
        summary = read(directory / "summary.json")
        audit = read(directory / "independent_audit.json")
        if summary.get("status") != "passed" or audit.get("status") != "passed":
            raise Unproven("Original full-population evaluation/audit has not passed")
        args = summary.get("arguments", {})
        if (resolve(summary.get("checkpoint") or args.get("checkpoint")) != resolve(target)
                or resolve(summary.get("parent_checkpoint") or args.get("parent")) != resolve(source)):
            raise ValueError("Original evidence scores a different source/target pair")
        if (resolve(args.get("data")) != DATA.resolve() or summary.get("days") != [day]
                or audit.get("days") != [day] or summary.get("history_limit") != 1024
                or summary.get("role") != 2 or summary.get("first_only")):
            raise ValueError("Original evidence differs in data, complete window or task")
        counts = self.counts(self.full_selection(day))
        for key in ("requests", "positive_requests", "users", "positive_users"):
            if counts[key] != summary[key]:
                raise ValueError(f"Original evidence does not cover the full selected day: {key}")
        if (audit["requests"] != counts["requests"]
                or audit["positive_requests"] != counts["positive_requests"]
                or audit["candidate_instances"] != counts["candidate_instances"]
                or not audit.get("candidate_membership")
                or audit["maximum_per_request_metric_error"] > 1e-12):
            raise ValueError("Original candidate/metric audit is inconsistent")
        for rank in range(summary["world_size"]):
            if not (directory / f"rank_{rank}.csv").is_file():
                raise Unproven("Original per-request metric evidence is missing")
        return {"kind": "original_full_population", "directory": str(directory),
                "counts": counts, "metrics": self.metrics(summary),
                "native_quality_diagnostic": native_quality_diagnostic(summary),
                "reliability": "Previously completed full30000-cohort complete-day evaluation and independent candidate/metric audit; nominal CI informative."}

    def sampled(self, directory, source, target, day, expected_users):
        directory = resolve(directory)
        completion = read(directory / "completion.json")
        if completion.get("status") != "completed":
            raise Unproven("Sampled evaluation has not completed")
        plan, summary = read(directory / "plan.json"), read(directory / "summary.json")
        audit, analysis = read(directory / "audit.json"), read(directory / "analysis.json")
        if (summary.get("status") != "completed" or analysis.get("status") != "completed"
                or audit.get("status") != "passed"):
            raise Unproven("Sampled scoring, analysis or independent audit is incomplete")
        if (resolve(plan["checkpoint"]) != resolve(target) or resolve(plan["cache_source"]) != resolve(source)
                or resolve(analysis["checkpoint"]) != resolve(target)
                or resolve(analysis["cache_source"]) != resolve(source)):
            raise ValueError("Sampled evidence describes a different source/target pair")
        if (resolve(plan["data"]) != DATA.resolve() or plan["eval_day"] != day
                or analysis["evaluation_day"] != day
                or analysis["evaluation_interval_ms_half_open"] != list(day_bounds_ms(day))
                or plan.get("history_limit") != 1024 or summary.get("history_limit") != 1024):
            raise ValueError("Sampled evidence differs in population, history or future day")
        if not audit.get("all_chosen_users_complete_day_requests"):
            raise ValueError("No evidence for complete selected-user day requests")
        chosen = np.load(directory / "chosen_uids.npy", allow_pickle=False)
        offset = int(plan["selection"]["user_offset"])
        if (chosen.ndim != 1 or len(chosen) != expected_users or len(np.unique(chosen)) != len(chosen)
                or plan["selection"].get("seed") != 20261029
                or offset < 0 or offset + len(chosen) > len(self.train)
                or not np.array_equal(chosen, self.permutation[offset:offset+len(chosen)])):
            raise ValueError("Sample does not match its declared fixed pre-outcome UID permutation")
        full = self.full_selection(day)
        records = np.load(directory / "selection.npy", mmap_mode="r")
        if not np.array_equal(records, full[np.isin(full["uid"], chosen)]):
            raise ValueError("Actual sample omits/changes complete chosen-user day requests")
        counts = self.counts(records)
        for key in ("requests", "positive_requests", "users", "positive_users"):
            if summary[key] != counts[key]:
                raise ValueError(f"Sample summary population differs from actual records: {key}")
        for key in ("requests", "positive_requests", "positive_users"):
            if audit[key] != counts[key] or completion[key] != counts[key] or analysis["overall"][key] != counts[key]:
                raise ValueError("Audit, completion and analysis use different request populations")
        if (audit["chosen_users"] != len(chosen) or audit["active_users"] != counts["users"]
                or audit["candidate_instances"] != counts["candidate_instances"]
                or not audit.get("candidate_membership")
                or audit["maximum_independent_per_request_metric_error"] > 1e-12
                or audit["maximum_independent_summary_error"] > 1e-12):
            raise ValueError("Independent candidate/metric audit does not support this sample")
        expected_path = [{key: entry[key] for key in ("checkpoint", "train_days", "train_end_day", "actual_parent")}
                         for entry in self.ancestry(target, source)]
        if audit["actual_training_path"] != expected_path or analysis["actual_training_path"] != expected_path:
            raise ValueError("Recorded audited ancestry differs from actual checkpoint parents")
        for rank in range(summary["world_size"]):
            if not (directory / f"rank_{rank}.csv").is_file():
                raise Unproven("Sampled per-request evidence is missing")
        paired = analysis["overall"]["comparisons"]["ndcg@10"]["ancestor_cache_loss"]
        metrics = self.metrics(summary, paired)
        weaknesses = []
        if not metrics["ci_excludes_zero_positive"]:
            weaknesses.append("Nominal NDCG user-cluster interval includes zero or is nonpositive; no universal interval threshold is imposed.")
        slice_results = {}
        for name, value in analysis.get("predetermined_slices", {}).items():
            item = value.get("comparisons", {}).get("ndcg@10", {}).get("ancestor_cache_loss")
            if item:
                slice_results[name] = item
                if item["difference"] <= 0:
                    weaknesses.append(f"Predetermined slice {name} has a nonpositive point difference.")
        return {"directory": str(directory), "chosen_users": len(chosen), "chosen_uid_offset": offset,
                "counts": counts, "metrics": metrics, "predetermined_slices": slice_results,
                "native_quality_diagnostic": native_quality_diagnostic(summary),
                "weaknesses": weaknesses, "audit": str(directory / "audit.json")}, chosen


def objective_scope(manifest):
    """Read the explicit count revision; a planned shorter chain proves nothing."""
    revision_path = manifest.get("objective_revision")
    revision = read(resolve(revision_path)) if revision_path else {}
    preferred = manifest.get("preferred_model_count", 10)
    if preferred != 10:
        raise ValueError("The preferred target remains ten real models")
    allow_shorter = (manifest.get("allow_fewer_models_at_verified_data_exhaustion") is True
                     and bool(revision.get("authorized_termination_alternative")))
    return {"preferred_model_count": preferred, "preferred_edge_count": preferred - 1,
            "allow_fewer_at_reviewed_data_exhaustion": allow_shorter,
            "objective_revision_path": str(resolve(revision_path)) if revision_path else None,
            "objective_revision": revision}


def terminal_review(manifest, scope, nodes, edges):
    """Check a separately authored terminal review's binding, not its scientific judgment.

    Optional manifest.data_exhaustion_review points to JSON with status=completed,
    reason=data_exhaustion, actual_model_checkpoints, actual_edge_ids,
    evaluation_days, last_available_data_day, latest_training_cutoff_with_future_evaluation,
    remaining_window_review (nonempty explanation), evidence_paths (existing files),
    and no_pending_work=true. This is an external review, never inferred here.
    """
    path = manifest.get("data_exhaustion_review")
    result = {"status": "missing_external_review", "path": path,
              "references_verified": False, "independently_proven_by_this_tool": False}
    if not path:
        return result
    try:
        review = read(resolve(path))
        result.update(path=str(resolve(path)), record=review)
        boundary = scope["objective_revision"].get("data_boundary", {})
        if not scope["allow_fewer_at_reviewed_data_exhaustion"]:
            raise ValueError("Shorter-chain data-exhaustion termination is not authorized by the recorded revision")
        if review.get("status") != "completed" or review.get("reason") != "data_exhaustion":
            raise Unproven("External data-exhaustion review is not completed")
        if ([str(resolve(p)) for p in review.get("actual_model_checkpoints", [])]
                != [node["checkpoint"] for node in nodes]
                or review.get("actual_edge_ids") != [edge["id"] for edge in edges]
                or review.get("evaluation_days") != [edge["evaluation_day"] for edge in edges]):
            raise ValueError("External review is not bound to this exact actual chain")
        for key in ("last_available_data_day", "latest_training_cutoff_with_future_evaluation"):
            if boundary.get(key) is None or review.get(key) != boundary[key]:
                raise ValueError(f"External review does not match the recorded data boundary: {key}")
        if not review.get("remaining_window_review") or review.get("no_pending_work") is not True:
            raise Unproven("External review must explain remaining windows and explicitly record no pending work")
        references = review.get("evidence_paths")
        if not isinstance(references, list) or not references:
            raise Unproven("External review has no actual supporting evidence references")
        for reference in references:
            if not resolve(reference).is_file():
                raise Unproven(f"External review evidence is missing: {reference}")
        run = manifest.get("planned_release_scenario", {}).get("current_training_run")
        if run:
            completion = resolve(run) / "completion.json"
            if not completion.is_file() or read(completion).get("status") != "completed":
                raise Unproven("The declared current training/evaluation run is still pending")
        result.update(status="external_review_references_verified", references_verified=True)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result.update(status="unverified_external_review", reason=str(exc))
    return result


def inspect_manifest(manifest):
    issues, node_results, edge_results = [], [], []

    def issue(kind, location, detail):
        issues.append({"kind": kind, "location": location, "detail": str(detail)})

    nodes, edges = manifest.get("nodes", []), manifest.get("edges", [])
    try:
        scope = objective_scope(manifest)
    except (ValueError, KeyError, OSError) as exc:
        issue("unproven" if isinstance(exc, Unproven) else "contradicted", "objective_revision", exc)
        scope = objective_scope({})
    valid_count = 1 <= len(nodes) <= 10 if scope["allow_fewer_at_reviewed_data_exhaustion"] else len(nodes) == 10
    if not valid_count or [node.get("id") for node in nodes] != [f"V{i}" for i in range(len(nodes))]:
        issue("contradicted", "nodes", "Need ordered adjacent publication nodes V0 onward, at most ten; distinct model pairs cannot substitute for a chain")
    if len(edges) != len(nodes) - 1:
        issue("contradicted", "edges", "Declared model slots must have exactly their adjacent edge slots")
    supplied_nodes = sum(bool(node.get("checkpoint")) for node in nodes)
    if any(node.get("checkpoint") for node in nodes[supplied_nodes:]):
        issue("contradicted", "nodes", "Actual released model nodes must form a contiguous prefix before preferred placeholders")
    if (manifest.get("protocol") != PROTOCOL or manifest.get("history_limit") != 1024
            or manifest.get("model_scale") != "max" or manifest.get("primary_metric") != "request_macro_observed_positive_ndcg@10"):
        issue("contradicted", "protocol", "Manifest must declare Max,1024-history,matched-prefix complete-candidate NDCG task")
    if manifest.get("data") and resolve(manifest["data"]) != DATA.resolve():
        issue("contradicted", "data", "Manifest uses another model/data lineage")
    inspector = Inspector()
    node_paths, known_paths = {}, set()
    for index, node in enumerate(nodes):
        entry = {"id": node.get("id"), "status": "unproven"}
        if not node.get("checkpoint"):
            entry["status"] = "pending_preferred_slot"
            node_results.append(entry)
            continue
        try:
            model = inspector.model(node.get("checkpoint"))
            if model["checkpoint"] in known_paths:
                raise ValueError("A released version reuses the exact same checkpoint as another node")
            known_paths.add(model["checkpoint"])
            node_paths[node.get("id")] = model["checkpoint"]
            entry.update(status="verified", checkpoint=model["checkpoint"], train_end_day=model["train_end_day"],
                         actual_optimizer_parent=model["actual_parent"])
        except (ValueError, KeyError, OSError) as exc:
            kind = "unproven" if isinstance(exc, (Unproven, FileNotFoundError)) else "contradicted"
            entry.update(status=kind, reason=str(exc));issue(kind, f"nodes[{index}]", exc)
        node_results.append(entry)
    previous_day = None
    for index, edge in enumerate(edges):
        entry = {"id": edge.get("id"), "source": edge.get("source"), "target": edge.get("target"),
                 "evaluation_day": edge.get("evaluation_day"), "status": "unproven"}
        try:
            if index >= 9 or edge.get("source") != f"V{index}" or edge.get("target") != f"V{index+1}":
                raise ValueError("Edge does not connect exactly previous published node to current node")
            if edge.get("id") != f"V{index}_to_V{index+1}":
                raise ValueError("Edge identity does not match its adjacent publication nodes")
            if index >= supplied_nodes - 1:
                evidence = edge.get("evidence") or {}
                if any(evidence.get(key) for key in ("directory", "pilot", "replication")):
                    raise ValueError("Evidence is attached to a preferred slot without both actual published models")
                entry.update(status="pending_preferred_slot", evaluation_day=None,
                             preferred_placeholder_day=edge.get("evaluation_day"))
                edge_results.append(entry)
                continue
            day = edge.get("evaluation_day")
            if type(day) is not int or not 28 <= day <= 37:
                raise ValueError("Evaluation day must be D28..37 after actual catalog/cohort availability")
            if previous_day is not None and day <= previous_day:
                raise ValueError("Evaluation days are not strictly increasing")
            previous_day = day
            source, target = node_paths.get(edge["source"]), node_paths.get(edge["target"])
            if not source or not target:
                raise Unproven("One or both actual model nodes remain missing or unverified")
            path = inspector.ancestry(target, source)
            if day <= path[-1]["train_end_day"]:
                raise ValueError("Evaluation occurs at or before target training cutoff")
            entry["actual_training_path"] = [{key: item[key] for key in ("checkpoint", "train_days", "train_end_day", "actual_parent")} for item in path]
            evidence = edge.get("evidence") or {}
            if evidence.get("kind") == "original_full_population":
                result = inspector.original(evidence.get("directory"), source, target, day)
                if not result["metrics"]["point_positive"]:
                    raise ValueError("Original edge has Rebuild <= Reuse")
                entry.update(evidence=result, primary_metrics=result["metrics"])
            elif evidence.get("kind") == "pilot_replication":
                pilot, pu = inspector.sampled(evidence.get("pilot"), source, target, day, 4096)
                replication, ru = inspector.sampled(evidence.get("replication"), source, target, day, 8192)
                if np.intersect1d(pu, ru).size:
                    raise ValueError("Pilot and replication chosen users overlap")
                entry.update(evidence={"kind": "pilot_replication", "pilot": pilot, "replication": replication,
                                       "chosen_uids_disjoint": True,
                                       "reliability": "Positive4096-user pilot plus disjoint8192-user replication; same models/date, not independent retraining/time."},
                             primary_metrics=replication["metrics"], primary_population=replication["counts"],
                             weaknesses=pilot["weaknesses"]+replication["weaknesses"])
                if not pilot["metrics"]["point_positive"] or not replication["metrics"]["point_positive"]:
                    raise ValueError("Both declared pilot and replication must have Rebuild > Reuse")
            else:
                raise Unproven("No declared completed full-population or pilot-plus-replication evidence")
            entry["status"] = "verified_positive"
        except (ValueError, KeyError, OSError) as exc:
            kind = "unproven" if isinstance(exc, (Unproven, FileNotFoundError)) else "contradicted"
            entry.update(status=kind, reason=str(exc));issue(kind, f"edges[{index}]", exc)
        edge_results.append(entry)
    prefix = 0
    for item in edge_results:
        if item["status"] != "verified_positive":
            break
        prefix += 1
    actual_nodes = node_results[:supplied_nodes]
    actual_edges = edge_results[:max(0, supplied_nodes - 1)]
    actual_complete = (not issues and supplied_nodes > 0
                       and all(node["status"] == "verified" for node in actual_nodes)
                       and prefix == supplied_nodes - 1)
    verified_prefix_nodes = prefix + 1 if node_results and node_results[0]["status"] == "verified" else 0
    preferred_complete = actual_complete and supplied_nodes == 10 and prefix == 9
    termination = terminal_review(manifest, scope, actual_nodes, actual_edges)
    structural_and_point_complete = actual_complete and (preferred_complete or termination["references_verified"])
    status = ("contradicted" if any(x["kind"] == "contradicted" for x in issues) else
              "structure_and_points_verified_stability_review_required" if structural_and_point_complete else "ongoing")
    return {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "status": status,
            "structural_and_point_requirements_proven": structural_and_point_complete,
            "actual_prefix_structural_and_point_requirements_proven": actual_complete,
            "declared_actual_model_count": supplied_nodes,
            "actual_accepted_model_count": verified_prefix_nodes,
            "actual_accepted_edge_count": prefix,
            "preferred_target_achieved": preferred_complete,
            "objective_scope": scope, "data_exhaustion_review": termination,
            "terminal_basis": "preferred_count" if preferred_complete else "externally_reviewed_data_exhaustion" if structural_and_point_complete else None,
            "proof_scope": "Actual models, ancestry, adjacent edges, increasing future windows, audited population/metrics and positive NDCG point estimates only; positive-edge counts do not establish stability.",
            "goal_requirements_proven": False,
            "stability_review_status": "not_assessed_by_this_tool",
            "stability_review_required": True,
            "stability_review_note": "A separate actual per-edge review must assess uncertainty, cohort/time consistency and any supplemental evidence, retaining weak/adverse results. This checker imposes no new significance or slice threshold and does not infer stability from two positive points or a declared review path.",
            "cli_exit_semantics": {"2": "Ongoing, incomplete or contradicted; a valid actual prefix alone is not terminal",
                                   "3": "Preferred count or externally reviewed data-exhaustion endpoint has verified structure/positive points; separate stability/effect-size review required; not goal completion"},
            "preferred_real_model_nodes": 10, "preferred_positive_adjacent_edges": 9,
            "verified_real_model_nodes": sum(x["status"] == "verified" for x in node_results),
            "verified_positive_edges": sum(x["status"] == "verified_positive" for x in edge_results),
            "verified_continuous_prefix_edges": prefix, "nodes": node_results, "edges": edge_results,
            "issues": issues, "ci_is_universal_pass_threshold": False,
            "limitations": ["Structure and positive points alone cannot establish stable adjacent edges or the preferred effect-size distribution.",
                            "Preferred pending slots are not actual models, releases or scheduled evaluation dates. A shorter terminal chain needs a separate evidence-backed data-exhaustion review.",
                            "External terminal-review bindings and file existence are checked, but the substantive remaining-window judgment and final stability/effect-size review remain external.",
                            "Selected all-positive development chain; not an unbiased prevalence estimate.",
                            "Repeated users, models and dates create dependence; disjoint pilot/replication users are not independent training seeds.",
                            "Matched causal-prefix snapshots, not persistent-cache/FIFO serving.",
                            "Metadata intent, old six distinct pairs and unfinished training do not establish the preferred ten-node target or data exhaustion."],
            "goal_status_changed": False, "gpu_work_launched": False}


def cli_exit_code(result):
    # Neither outcome is a claim that the full stability goal is satisfied.
    return 3 if result["structural_and_point_requirements_proven"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=BASE / "chain_manifest.json")
    parser.add_argument("--output", type=Path, default=BASE / "completion_audit.json")
    args = parser.parse_args()
    result = inspect_manifest(read(args.manifest))
    result["manifest"] = str(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("status", "structural_and_point_requirements_proven",
          "actual_prefix_structural_and_point_requirements_proven", "actual_accepted_model_count",
          "actual_accepted_edge_count", "preferred_target_achieved", "terminal_basis",
          "goal_requirements_proven", "stability_review_status", "stability_review_required",
          "verified_real_model_nodes", "verified_positive_edges", "verified_continuous_prefix_edges", "issues")}, ensure_ascii=False))
    raise SystemExit(cli_exit_code(result))


if __name__ == "__main__":
    main()
