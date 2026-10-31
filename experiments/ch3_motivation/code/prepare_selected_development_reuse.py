#!/usr/bin/env python3
"""Index exact checkpoint identities before the existing selected-plan audit.

This only avoids repeated inference. The original planner remains responsible
for auditing the actual winner, inputs, code, score blocks and native anchors.
No score values participate in choosing an artifact or a recipe.
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

from experiments.ch3_motivation.code import plan_selected_development as planner
from experiments.ch3_motivation.code.common import ROOT, write_json
from hstu_kvcache.common.artifacts import sha256

RUNS = ROOT / "experiments/ch3_motivation/runs/e5"


def committed_candidates():
    roots = (
        RUNS / "development_quality/attempt_1",
        RUNS / "temporal_comparison512_quality/attempt_1",
        RUNS / "candidate_fraction_comparison512_quality/attempt_1",
        RUNS / "initialization_pair_development/attempt_1",
    )
    rows = []
    for root in roots:
        for path in sorted(root.rglob("summary.json")):
            value = json.loads(path.read_text())
            identity = value.get("identity", {})
            if (value.get("status") == "complete" and value.get("experiment") == "E5"
                    and value.get("quality_users") == 512 and value.get("seed") == 17
                    and value.get("method_label") in planner.METHODS
                    and identity.get("checkpoint_sha256")):
                rows.append((path, value))
    return rows


def prepare(selection_path, output):
    selection_path = planner.resolved(selection_path)
    selection = json.loads(selection_path.read_text())
    if (selection.get("status") != "recommendation_for_review"
            or selection.get("all_five_edges_complete") is not True
            or selection.get("phase_b_completed") is not True):
        raise ValueError("require the actual completed final common-validation selection")
    chosen = selection["recommendation"]
    candidates = committed_candidates()
    inventory = output / "exact_checkpoint_inventory"
    inventory.mkdir(parents=True, exist_ok=False)
    records = []
    for edge in planner.EDGES:
        reference = chosen["checkpoints_by_edge"][str(edge)]
        for method in planner.METHODS:
            checkpoint = planner.resolved(reference["checkpoint"]).parent / f"{method}.pt"
            checkpoint_sha = sha256(checkpoint)
            matching = [(path, value) for path, value in candidates
                        if value["identity"]["checkpoint_sha256"] == checkpoint_sha
                        and value["identity"].get("edge") == edge
                        and value["method_label"] == method]
            if matching:
                path, _ = matching[0]
                (inventory / f"edge_{edge}_{method}").symlink_to(path.parent.resolve(), target_is_directory=True)
                records.append(dict(edge=edge, method=method, checkpoint_sha256=checkpoint_sha,
                                    candidate_summary=str(path.resolve()), score_values_read=False))
    # This performs the original full checks and binds every actual artifact.
    result = planner.generate(selection_path, existing_root=inventory)
    if len(result.get("planned_jobs", [])) != 15:
        raise ValueError("actual final recipe did not produce the complete fifteen-point plan")
    write_json(output / "inventory.json", dict(selection=str(selection_path),
        selection_sha256=sha256(selection_path), candidates=records,
        source_sha256=sha256(Path(__file__)), score_values_read=False,
        accepted_points=result["reuse"]["reused_points"], execution_points=len(result["jobs"])))
    write_json(output / "plan.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=planner.SELECTION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.selection, args.output.resolve())
    print(json.dumps(dict(status=result["status"], reused=result["reuse"]["reused_points"],
                          new_jobs=len(result["jobs"]), gpu_launched=False)))
