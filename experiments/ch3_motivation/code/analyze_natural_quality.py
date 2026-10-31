#!/usr/bin/env python3
"""CPU, per-edge reports for the fixed natural active2000 supplement.

Edges may have different UID cohorts.  This entry never pools releases, fits
weights, or treats the predeclared seed17 comparison as three-seed evidence.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow.parquet as pq

from experiments.ch3_motivation.code import (
    analyze_probe_quality as paired, analyze_selective_quality as selective_analysis,
    calibration_cost_correction as correction, plan_probe_natural_quality as natural_plan,
    qualify_scenario, run_selective_population,
)
from experiments.ch3_motivation.code.audit_scenario_qualification import audit_blocks
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.scenario_inputs import load_scenario_population
from hstu_kvcache.evaluation.statistics import paired_user_bootstrap
from hstu_kvcache.common.artifacts import sha256

BASE = ROOT/"experiments/ch3_motivation"
DEFAULT_CONFIG = BASE/"configs/natural_active_comparison.json"
DEFAULT_ANCHORS = BASE/"runs/scenario_qualification/natural_active/attempt_1"
DEFAULT_E3 = BASE/"runs/e3/natural_active/attempt_1"
DEFAULT_E5_PLAN = BASE/"runs/e5/natural_quality_plan/attempt_1/plan.json"
DEFAULT_E5 = BASE/"runs/e5/natural_quality/attempt_1"
EDGES, METHODS = tuple(range(1, 6)), ("f", "fg", "fgh")


def sources():
    names = ("analyze_probe_quality.py", "analyze_selective_quality.py",
        "audit_scenario_qualification.py", "calibration_cost_correction.py",
        "plan_probe_natural_quality.py", "scenario_inputs.py")
    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in names],
        ROOT/"src/hstu_kvcache/evaluation/statistics.py"]
    return {str(path): sha256(path) for path in paths}


def _json(path, hashes):
    path = Path(path).resolve()
    hashes[str(path)] = sha256(path)
    return json.loads(path.read_text())


def _missing(paths):
    return [str(path) for path in paths if not path.is_file()]


def _pending(paths):
    return dict(status="pending", missing=[str(path) for path in paths], points=0)


def _window(context):
    return context["binding"]["settings"].get("evaluation_window", {"name": "E14"})


def _anchors(folder, edge, hashes):
    """Natural anchors use current/, not the old union_current_anchors layout."""
    manifest_path = folder/"scenario_manifest.json"
    binding, users, population, provenance = load_scenario_population(
        manifest_path, edge, minimum_users=2000, require_positive=False)
    selection = provenance["selection"]
    if (len(users) != 2000 or selection["panel_name"] != "long_feedback_active" or
            selection["definition"] != natural_plan.DEFINITION or
            binding["candidate_panels"]["long_feedback_active"] != users):
        raise ValueError("natural report requires the exact fixed active2000 cohort")
    strata_path = Path(selection["strata_manifest_path"])
    strata = _json(strata_path, hashes)
    if (hashes[str(strata_path.resolve())] != selection["strata_manifest_sha256"] or
            strata["subgroups_by_edge"][str(edge)]["long_feedback_active"]["first_2000_uid_hashes"] != users):
        raise ValueError("natural report UID membership differs from the frozen pre-release stratum")
    summary = _json(folder/"summary.json", hashes)
    identity = _json(folder/"identity.json", hashes)
    if (summary["status"] != "complete" or summary["identity"] != identity or
            identity["experiment"] != "fixed_natural_three_anchors" or identity["edge"] != edge or
            identity["input_binding_sha256"] != provenance["input_binding_sha256"] or
            identity["models"] != provenance["anchor_models"] or identity["settings"] != binding["settings"]):
        raise ValueError("natural anchor summary/input/model identity differs")
    expected_sources = dict(qualify_scenario.sources())
    for name in ("scenario_inputs.py", "qualify_natural_scenario.py"):
        path = Path(__file__).with_name(name)
        expected_sources[str(path.relative_to(ROOT))] = sha256(path)
    if identity["sources"] != expected_sources:
        raise ValueError("natural anchor numerical sources differ")
    selected = set(users)
    events = [row for row in pq.read_table(provenance["requests_path"]).to_pylist() if row["uid"] in selected]
    old, old_audit = audit_blocks(folder/"old_full/quality_shards", users,
        dict(identity, branch="old_full"), ["old_full"], events,
        block_size=binding["settings"]["evaluation_checkpoint_users"])
    current, current_audit = audit_blocks(folder/"current/quality_shards", users,
        dict(identity, branch="current"), ["reuse", "rebuild"], events,
        block_size=binding["settings"]["evaluation_checkpoint_users"])
    measured = qualify_scenario.bootstrap_anchors(old+current, users, replicates=1000, seed=28121+edge)
    if any(summary[key] != value for key, value in measured.items() if key != "scope"):
        raise ValueError("natural anchor summary differs from verified scores/paired intervals")
    for path, digest in ((manifest_path, provenance["manifest_sha256"]),
            (provenance["history_path"], provenance["history_sha256"]),
            (provenance["requests_path"], provenance["requests_sha256"])):
        hashes[str(Path(path).resolve())] = digest
    return dict(binding=binding, users=users, population=population, provenance=provenance,
        current=current, events=events, summary=summary,
        audit=dict(old_full=old_audit, current=current_audit))


def _e3(folder, manifest_path, edge, context, hashes):
    # Supplying the already audited natural current rows bypasses only the
    # old analyzer's union-anchor lookup; its numerical/policy/cost checks run.
    users, rows, costs, audit = selective_analysis._audit_edge(folder, manifest_path,
        edge, run_selective_population.sources(), {edge: context["current"]})
    if users != context["users"]:
        raise ValueError("natural E3 and anchors measured different users")
    summary = _json(folder/"summary.json", hashes)
    intervals, confidence = paired_user_bootstrap(selective_analysis._groups(rows), users,
        replicates=1000, seed=17021+edge)
    if confidence != summary["confidence"]:
        raise ValueError("natural E3 paired UID confidence differs")
    for point in summary["results"]:
        if any(point[key] != value for key, value in intervals[point["method"]].items()):
            raise ValueError("natural E3 point intervals differ from committed scores")
    return dict(status="complete", points=24, summary=str((folder/"summary.json").resolve()),
        summary_sha256=hashes[str((folder/"summary.json").resolve())], quality_users=2000,
        feedback_events=summary["feedback_events"], reference_gap=summary["reference_gap"],
        results=summary["results"], confidence=confidence, costs=costs, audit=audit)


def _check_costs(costs):
    for account, fields in ((costs, ("calibration_flops", "release_preparation_flops", "extra_read_flops",
            "extra_native_writer_flops", "shared_graph_setup_flops")),
            (costs["candidate_reads_230"], ("calibration_flops", "release_preparation_flops",
            "extra_candidate_read_flops", "actual_behavior_read_extra_flops",
            "extra_native_writer_flops", "shared_graph_setup_flops"))):
        total, denominator = account["complete_added_flops"], account["once_rebuild_flops"]
        if (sum(account[key] for key in fields) != total or denominator <= 0 or
                not np.isfinite(account["percent_of_once_rebuild"]) or
                abs(account["percent_of_once_rebuild"]-100*total/denominator) > 1e-10):
            raise ValueError("natural E5 complete cost arithmetic or denominator differs")


def _e5(jobs, plan, root, edge, context, hashes, e3):
    paths = [root/job["name"]/"summary.json" for job in jobs]
    expected = {str(Path(record["checkpoint"]).resolve()): record
        for record in plan["edges"][str(edge)]["weights"]}
    summaries, costs = {}, {}
    for job, path in zip(jobs, paths, strict=True):
        saved = _json(path, hashes)
        checkpoint = str(Path(job["checkpoint"]).resolve())
        record = expected[checkpoint]
        method = record["method_label"]
        identity = saved["identity"]
        recorded_sources = identity.get("sources", {})
        entry = "experiments/ch3_motivation/code/evaluate_probes.py"
        planned_entry = plan.get("source_sha256", {}).get(entry)
        if not recorded_sources or not planned_entry or recorded_sources.get(entry) != planned_entry:
            raise ValueError("natural E5 execution sources lack the planned evaluate entry SHA")
        for name, digest in recorded_sources.items():
            source_path = (ROOT/name).resolve()
            actual = sha256(source_path)
            if actual != digest:
                raise ValueError("natural E5 recorded execution source SHA differs from its actual file")
            hashes[str(source_path)] = actual
        if (method not in METHODS or method in summaries or saved["method_label"] != method or saved["seed"] != 17 or
                saved["quality_users"] != 2000 or saved["edge"] != edge or identity["panel"] != "natural_active" or
                identity["users"] != context["users"] or identity["models"] != context["provenance"]["anchor_models"] or
                identity["scenario"] != context["provenance"] or identity["backend"] != "projected" or
                str(Path(identity["checkpoint"]).resolve()) != checkpoint or
                identity["checkpoint_sha256"] != record["checkpoint_sha256"]):
            raise ValueError("natural E5 stage differs from the ready seed17 plan/cohort")
        corrected = correction.checkpoint_correction(checkpoint, expected_sha256=record["checkpoint_sha256"])
        if corrected["edge"] != edge or corrected["seed"] != 17 or corrected["method_label"] != method:
            raise ValueError("natural E5 actual typed checkpoint identity differs")
        _check_costs(saved["costs"])
        accounts = correction.apply_to_costs(saved["costs"], corrected)
        _check_costs(accounts["corrected"])
        if not all(np.isfinite(saved[key]) for key in ("auc", "reuse_auc", "rebuild_auc")):
            raise ValueError("natural E5 summary contains nonfinite AUC")
        if e3["status"] == "complete":
            for name in ("actual_event_ledger", "candidate_reads_230"):
                account = saved["costs"] if name == "actual_event_ledger" else saved["costs"][name]
                if account["once_rebuild_flops"] != e3["costs"][name]["once_rebuild_flops"]:
                    raise ValueError("natural E3/E5 Rebuild cost denominators differ")
        summaries[method] = saved
        costs[method] = accounts
    if set(summaries) != set(METHODS):
        raise ValueError("natural E5 needs exactly f/fg/fgh at seed17")
    result = paired.analyze(paths, replicates=1000, seed=17021+edge)
    if len(result["groups"]) != 1 or len(result["groups"][0]["per_seed"]) != 1:
        raise ValueError("natural E5 must have one actual recipe and the single prespecified seed")
    group = result["groups"][0]
    single = group["per_seed"][0]
    if single["seed"] != 17 or set(single["methods"]) != set(METHODS) or set(single["increments"]) != {"g", "h"}:
        raise ValueError("natural E5 direct contrast coverage differs")
    for method, metrics in single["methods"].items():
        saved = summaries[method]
        intervals = saved["intervals"][method]
        if (saved["confidence"]["replicates"] != 1000 or saved["confidence"]["seed"] != 17021+edge or
                metrics["auc_ci95"] != intervals["auc_ci95"] or
                metrics["recovery_percent_ci95"] != intervals["recovery_percent_ci95"]):
            raise ValueError("natural E5 runner and joint paired method intervals differ")
        metrics.update(auc_gain_over_reuse=saved["auc"]-saved["reuse_auc"],
            auc_gain_over_reuse_ci95=intervals["auc_gain_over_reuse_ci95"])
    first = summaries["f"]
    anchor_folder = Path(first["anchor_source"]["path"])
    anchor_identity = _json(anchor_folder.parent/"configuration.json", hashes)
    anchor_rows, anchor_audit = audit_blocks(anchor_folder/"quality_shards", context["users"],
        anchor_identity, ["reuse", "rebuild"], context["events"])
    alignment = selective_analysis.compare_fresh_anchors(anchor_rows, context["current"], context["users"])
    hashes.update(result["verification"]["artifact_sha256"])
    return dict(status="complete", points=3, quality_users=2000, feedback_events=result["feedback_events"],
        expected_seeds=[17], seed=17, three_seed_claim=False, independent_confirmation=False,
        recipe={key: group[key] for key in ("variant", "fit_users", "width", "depth")},
        methods=single["methods"], increments=single["increments"], costs=costs,
        anchors=result["anchors"], confidence=result["confidence"],
        summary_bindings={method: dict(path=str((root/job["name"]/"summary.json").resolve()),
            sha256=hashes[str((root/job["name"]/"summary.json").resolve())])
            for job in jobs for method in (expected[str(Path(job["checkpoint"]).resolve())]["method_label"],)},
        anchor_audit=anchor_audit, fresh_anchor_comparison=alignment,
        scope="prespecified seed17 f/fg/fgh supplement; per-edge paired g/h intervals, no three-seed completion requirement; no f_extended/fg_extended controls, so increments alone do not exclude extra training-budget effects")


def _write_csv(path, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tables(edges):
    rows = {name: [] for name in ("e3_points", "e3_costs", "e5_points", "e5_increments", "e5_costs")}
    for edge, report in edges.items():
        common = dict(edge=int(edge), evaluation_window=report.get("evaluation_window", {}).get("name"), quality_users=2000)
        e3, e5 = report["e3"], report["e5"]
        if e3["status"] == "complete":
            for point in e3["results"]:
                value = {key: val for key, val in point.items() if key != "setting"}
                value.update(point["setting"])
                rows["e3_points"].append(dict(common, **value))
            for name in ("actual_event_ledger", "candidate_reads_230"):
                account = e3["costs"][name]
                for point in account["results"]:
                    rows["e3_costs"].append(dict(common, account=name,
                        once_rebuild_flops=account["once_rebuild_flops"], **point))
        if e5["status"] == "complete":
            for method, metrics in e5["methods"].items():
                rows["e5_points"].append(dict(common, seed=17, method=method, **metrics))
            for contrast, metrics in e5["increments"].items():
                rows["e5_increments"].append(dict(common, seed=17, contrast=contrast, **metrics))
            for method, accounts in e5["costs"].items():
                for version in ("raw", "corrected"):
                    for name in ("actual_event_ledger", "candidate_reads_230"):
                        account = accounts[version] if name == "actual_event_ledger" else accounts[version][name]
                        scalar = {key: value for key, value in account.items() if isinstance(value, (int, float, bool))}
                        rows["e5_costs"].append(dict(common, seed=17, method=method, version=version, account=name, **scalar))
    return rows


def analyze(output, *, comparison_config=DEFAULT_CONFIG, anchors_root=DEFAULT_ANCHORS,
            e3_root=DEFAULT_E3, e5_plan=DEFAULT_E5_PLAN, e5_root=DEFAULT_E5):
    output, comparison_config, anchors_root, e3_root, e5_plan, e5_root = map(Path,
        (output, comparison_config, anchors_root, e3_root, e5_plan, e5_root))
    hashes, code = {}, sources()
    config = _json(comparison_config, hashes)
    if (config["cohort"] != "long_feedback_active" or config["users_per_edge"] != 2000 or
            config["definition"] != natural_plan.DEFINITION or not config["all_edges_retained"] or config["require_positive"]):
        raise ValueError("natural comparison definition/complete five-edge scope differs")
    e5 = _json(e5_plan, hashes) if e5_plan.is_file() else None
    ready_e5 = e5 is not None and e5["status"] == "prepared_for_root_review"
    jobs = {edge: [] for edge in EDGES}
    if ready_e5:
        if (len(e5["jobs"]) != 15 or len({job["name"] for job in e5["jobs"]}) != 15 or
                e5["seeds"] != [17] or e5["methods"] != list(METHODS) or
                e5["definition"] != natural_plan.DEFINITION):
            raise ValueError("natural E5 ready plan must contain exactly15 prespecified seed17 points")
        for path, digest in e5["frozen_sha256"].items():
            if sha256(Path(path)) != digest:
                raise ValueError("natural E5 frozen plan dependency differs")
            hashes[str(Path(path).resolve())] = digest
        for job in e5["jobs"]:
            if job["edge"] not in jobs or job["panel"] != "natural_active" or job["backend"] != "projected":
                raise ValueError("natural E5 prepared runtime scope differs")
            jobs[job["edge"]].append(job)
        if any(len(value) != 3 for value in jobs.values()):
            raise ValueError("natural E5 must retain all three stages on every edge")
    reports = {}
    for edge in EDGES:
        anchor_folder = anchors_root/f"edge_{edge}"
        manifest = anchor_folder/"scenario_manifest.json"
        missing = _missing([manifest, anchor_folder/"identity.json", anchor_folder/"summary.json"])
        record = dict(anchors=_pending(missing), e3=_pending(missing), e5=_pending(missing))
        reports[str(edge)] = record
        if missing:
            continue
        context = _anchors(anchor_folder, edge, hashes)
        record.update(evaluation_window=_window(context), quality_users=2000,
            uids_sha256=context["summary"]["quality_uids_sha256"],
            anchors=dict(status="complete", summary=context["summary"], audit=context["audit"]))
        folder = e3_root/f"edge_{edge}"
        missing = _missing([folder/name for name in ("summary.json", "identity.json", "status.json",
            "policies.json", "layer_profile.json", "costs.json")])
        record["e3"] = _pending(missing) if missing else _e3(folder, manifest, edge, context, hashes)
        if not ready_e5:
            record["e5"] = dict(_pending([e5_plan]), upstream_status=e5["status"] if e5 else "missing_plan")
            continue
        if any(Path(job["population_file"]).resolve() != manifest.resolve() for job in jobs[edge]):
            raise ValueError("natural E5 job changed the measured scenario manifest")
        missing = _missing([e5_root/job["name"]/"summary.json" for job in jobs[edge]])
        record["e5"] = _pending(missing) if missing else _e5(jobs[edge], e5, e5_root, edge, context, hashes, record["e3"])
    families = {name: dict(status="complete" if all(row[name]["status"] == "complete" for row in reports.values()) else "pending",
        completed_edges=[int(edge) for edge, row in reports.items() if row[name]["status"] == "complete"])
        for name in ("anchors", "e3", "e5")}
    result = dict(format="natural_quality_per_edge_v1", status="complete" if all(f["status"] == "complete" for f in families.values()) else "pending",
        gpu_used=False, cohort="long_feedback_active", users_per_edge=2000, edges=reports, families=families,
        expected_e3_points=120, expected_e5_points=15, expected_e5_seeds=[17],
        e5_plan_status=e5["status"] if e5 else "missing_plan",
        cross_edge_aggregate_computed=False, independent_confirmation=False,
        scope="fixed pre-release natural cohorts, possibly different UIDs per edge; all five edges/signs retained; E5 uses unchanged main seed17 weights",
        input_sha256=hashes, source_sha256=code)
    if sources() != code:
        raise ValueError("natural analysis sources changed during execution")
    write_json(output/"summary.json", result)
    tables = _tables(reports)
    for name, rows in tables.items():
        family = "e3" if name.startswith("e3") else "e5"
        _write_csv(output/(name+".csv"), rows if families[family]["status"] == "complete" else [])
    (output/"README.md").write_text("Natural active2000 per-edge supplement\n\nStatus: "+result["status"]+". No GPU used.\n\n"
        "Each family exports CSV only after all five predeclared edges are complete. Pending or partial inventory remains in summary.json. "
        "There is no five-edge UID aggregate, independent-population confirmation, or three-seed claim. "
        "Paired g/h intervals condition on the actual fixed seed17 weights. Without f_extended/fg_extended controls they alone cannot exclude extra training-budget effects. E5 raw and corrected complete costs retain separate actual/230 denominators. "
        "Edge5 remains E14_partial; adverse signs and undefined ratio intervals remain visible.\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (("comparison-config", DEFAULT_CONFIG), ("anchors-root", DEFAULT_ANCHORS),
            ("e3-root", DEFAULT_E3), ("e5-plan", DEFAULT_E5_PLAN), ("e5-root", DEFAULT_E5)):
        parser.add_argument("--"+name, type=Path, default=default)
    args = parser.parse_args()
    result = analyze(**vars(args))
    print(json.dumps(dict(status=result["status"], families=result["families"], gpu_used=False)))
