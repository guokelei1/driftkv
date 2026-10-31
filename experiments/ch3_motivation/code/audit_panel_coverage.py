#!/usr/bin/env python3
"""Read-only coverage audit of fixed Chapter 3 panels and saved anchor results."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow.parquet as pq

from experiments.ch3_motivation.code.common import stratified_order, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(users=len(values), mean=float(values.mean()),
        quantiles=dict(zip(("min", "p25", "median", "p75", "p90", "p99", "max"),
                          np.quantile(values, [0, .25, .5, .75, .9, .99, 1]).tolist())),
        zero_fraction=float(np.mean(values == 0))) if len(values) else dict(users=0)


def cdf_distance(sample, reference):
    sample, reference = np.sort(sample), np.sort(reference)
    at = np.unique(np.r_[sample, reference])
    return float(np.max(np.abs(np.searchsorted(sample, at, side="right") / len(sample)
        - np.searchsorted(reference, at, side="right") / len(reference))))


def main():
    base = ROOT / "experiments/ch3_motivation"
    population_path = base / "inputs/population.json"
    population = json.loads(population_path.read_text())
    source = base / "runs/scenario_qualification/strata_pre_release/attempt_1"
    manifest_path = source / "scenario_strata_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    features_path = source / "pre_release_features.parquet"
    if (sha(population_path) != manifest["source_population"]["sha256"]
            or sha(features_path) != manifest["features"]["sha256"]
            or sha(ROOT / "experiments/ch3_motivation/code/prepare_scenario_strata.py") != manifest["source_sha256"]
            or manifest["model_scores_read"] or manifest["future_feedback_read"]):
        raise ValueError("saved publishing-time feature identity differs")
    table = pq.read_table(features_path)
    columns = {name: table[name].to_numpy() for name in table.column_names}
    original_path = ROOT / population["role_source"]
    if sha(original_path) != population["role_source_sha256"]:
        raise ValueError("original frozen role source changed")
    eligible = json.loads(original_path.read_text())["roles"]["quality"]
    panels = dict(eligible=eligible, main=population["main"], confirmation=population["confirmation"],
                  development=population["development"])
    if len(eligible) != 75807 or any(len(set(users)) != len(users) for users in panels.values()):
        raise ValueError("fixed eligible/panel membership differs")
    fit_validation = set(population["fit"] + population["validation"])
    if any(set(users) & fit_validation for users in panels.values()):
        raise ValueError("evaluation panel overlaps fitting/validation users")
    if any(set(panels[left]) & set(panels[right]) for left, right in
           (("main", "confirmation"), ("main", "development"), ("confirmation", "development"))):
        raise ValueError("fixed evaluation roles are not disjoint")
    first = columns["edge"] == 1
    order = stratified_order(zip(columns["uid"][first], columns["pre_release_history_events"][first],
                                 columns["pre14_behavior_events"][first]), population["seed"])
    if (order[:512] != population["development"] or order[512:5512] != population["mismatch"]
            or order[5512:7512] != population["confirmation"] or order[7512:] != population["reserve"]
            or population["main"] != population["mismatch"][:2000]):
        raise ValueError("fixed roles cannot be reconstructed from publishing-time strata")
    fields = ("pre_release_history_events", "retained_history_rows", "pre14_behavior_events", "pre14_feedback_events")
    cutoffs = dict(history=[32, 128, 256, 512, 1024], behavior=[1, 8, 32, 128], feedback=[1, 4, 16, 64])
    edges, cells = [], []
    for edge in range(1, 6):
        mask = columns["edge"] == edge
        if set(columns["uid"][mask]) != set(eligible) or mask.sum() != len(eligible):
            raise ValueError("publishing-time features do not cover every eligible UID exactly once per edge")
        values = {name: columns[name][mask] for name in columns}
        if (not np.all(values["release_timestamp"] == (231 + 14 * (edge - 1)) * 86400)
                or not np.array_equal(values["retained_history_rows"], np.minimum(values["pre_release_history_events"], 1024))):
            raise ValueError("feature cutover or retained history cap differs")
        length = np.searchsorted(cutoffs["history"], values["pre_release_history_events"], side="right")
        activity = np.searchsorted(cutoffs["behavior"], values["pre14_behavior_events"], side="right")
        feedback = np.searchsorted(cutoffs["feedback"], values["pre14_feedback_events"], side="right")
        joint = length * 5 + activity
        reference = np.bincount(joint, minlength=30) / len(eligible)
        record = dict(edge=edge, release_timestamp=int(values["release_timestamp"][0]), panels={})
        for panel, users in panels.items():
            selected = np.isin(values["uid"], users)
            counts = np.bincount(joint[selected], minlength=30)
            proportions = counts / len(users)
            active = selected & (values["pre14_behavior_events"] > 0)
            density = values["pre14_feedback_events"][active] / values["pre14_behavior_events"][active]
            reference_active = values["pre14_behavior_events"] > 0
            reference_density = values["pre14_feedback_events"][reference_active] / values["pre14_behavior_events"][reference_active]
            natural = {}
            for name, definition in manifest["definitions"].items():
                cohort = ((values["retained_history_rows"] >= definition["retained_history_rows_at_least"])
                    & (values["pre14_behavior_events"] >= definition["pre14_behavior_events_at_least"])
                    & (values["pre14_feedback_events"] >= definition["pre14_feedback_events_at_least"]))
                natural[name] = dict(users=int(np.sum(selected & cohort)), fraction=float(np.mean(cohort[selected])),
                    eligible_fraction=float(np.mean(cohort)))
            absent = np.flatnonzero((reference > 0) & (counts == 0))
            record["panels"][panel] = dict(users=len(users),
                features={name: describe(values[name][selected]) for name in fields},
                pre14_feedback_events_per_behavior=describe(density),
                feedback_density_maximum_absolute_cdf_difference=cdf_distance(density, reference_density),
                pre_release_natural_cohort_proportions=natural,
                feedback_density_undefined_users=int(np.sum(selected & ~active)),
                marginal_bin_counts=dict(history=np.bincount(length[selected], minlength=6).tolist(),
                    behavior=np.bincount(activity[selected], minlength=5).tolist(),
                    feedback=np.bincount(feedback[selected], minlength=5).tolist()),
                joint_length_activity=dict(eligible_occupied_cells=int(np.sum(reference > 0)),
                    panel_occupied_cells=int(np.sum(counts > 0)),
                    eligible_mass_in_covered_cells=float(1 - reference[absent].sum()),
                    total_variation_distance=float(np.abs(proportions - reference).sum() / 2),
                    maximum_absolute_proportion_difference=float(np.max(np.abs(proportions - reference))),
                    absent_cells=[dict(length_bin=int(cell // 5), activity_bin=int(cell % 5),
                        eligible_users=int(round(reference[cell] * len(eligible))), eligible_fraction=float(reference[cell])) for cell in absent]),
                maximum_absolute_feature_cdf_difference={name:cdf_distance(values[name][selected], values[name]) for name in fields})
            cells.extend(dict(edge=edge, panel=panel, length_bin=cell // 5, activity_bin=cell % 5,
                users=int(counts[cell]), fraction=float(proportions[cell]), eligible_fraction=float(reference[cell])) for cell in range(30))
        edges.append(record)
    audit_path = base / "runs/e3/main_confirmation_bridge_queue/attempt_1/qualification_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["state"] != "complete" or not audit["all_sources_and_blocks_passed"]:
        raise ValueError("saved three-anchor qualification was not audited")
    anchors, anchor_files = [], []
    for edge in audit["edges"]:
        entries = {}
        for panel in ("main", "confirmation", "sensitivity_20261023", "sensitivity_20261024"):
            path = base / f"runs/scenario_qualification/attempt_1/edge_{edge['edge']}/{panel}/summary.json"
            if sha(path) != edge["panels"][panel]["summary_sha256"]:
                raise ValueError("saved anchor summary changed after its raw-block audit")
            saved = json.loads(path.read_text())
            expected = population[panel] if panel in population else population["population_sensitivity"][panel.rsplit("_",1)[1]]
            if saved["quality_uids"] != expected:
                raise ValueError("qualification population differs from predeclared panel")
            entries[panel] = {key:saved[key] for key in ("quality_users", "feedback_users", "feedback_events", "auc",
                "new_rebuild_minus_old_full", "new_rebuild_minus_old_full_ci95", "new_rebuild_minus_reuse",
                "new_rebuild_minus_reuse_ci95", "point_qualifies", "uid95_qualifies", "evaluation_window")}
            anchor_files.append(path)
        anchors.append(dict(edge=edge["edge"], panels=entries))
    development = []
    for edge in range(1, 6):
        path = base / f"runs/e3/edge_{edge}/development/attempt_1/summary.json"
        saved = json.loads(path.read_text())
        # Only the two baseline anchors are inspected. No compared-method score informs this audit.
        baseline = {row["method"]:row["auc"] for row in saved["results"] if row["method"] in ("reuse", "rebuild")}
        development.append(dict(edge=edge, quality_users=saved["quality_users"], feedback_users=saved["users_with_feedback"],
            feedback_events=saved["feedback_events"], auc=baseline, rebuild_minus_reuse=saved["reference_gap"],
            rebuild_minus_reuse_ci95=saved["confidence"]["reference_gap_ci95"], old_full="not newly measured for this 512-user panel"))
        anchor_files.append(path)
    output = base / "runs/scenario_qualification/panel_coverage/attempt_1"
    output.mkdir(parents=True, exist_ok=True)
    result = dict(state="complete",role="CPU read-only fixed-panel representativeness audit",panels={name:len(users) for name,users in panels.items()},
        source_features=dict(path=str(features_path),sha256=sha(features_path),rows=table.num_rows,
            causal_definition=manifest["timestamp_rule"],all_eligible_users_and_five_edges=True),
        original_score_blind_stratified_roles_reconstructed_exactly=True,fit_validation_excluded=True,
        no_population_or_role_changes=True,no_new_predictions_or_gpu_work=True,bin_cutoffs=cutoffs,
        edges=edges,baseline_qualification=dict(aggregate=audit["aggregate"],edges=anchors,
            criterion=audit["criterion"],population_selection=audit["population_selection"]),development_anchors=development,
        limitations=["Coverage is relative to the fixed 75807 eligible users, not all dataset users or another model scale.",
            "Coarse feature balance does not guarantee balance in items, labels, temporal dependencies or model update response.",
            "Main/confirmation are fixed once at the first release; later edges are longitudinal checks, not resampled panels.",
            "Pre-release feedback frequency was not an original sampling dimension; event-weighted AUC can remain sensitive to heavy users.",
            "Full eligible-population fresh OldFull/Rebuild/Reuse inference was not performed in this study.",
            "Four predeclared baseline panels were inspected for setting qualification; this conditioning must remain visible, and all inspected panels are retained."],
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),population_path,original_path,manifest_path,audit_path]+anchor_files})
    write_json(output / "summary.json",result)
    with (output / "joint_strata.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    print(json.dumps(dict(output=str(output),feature_rows=table.num_rows,exact_role_reconstruction=True)))


if __name__ == "__main__":
    main()
