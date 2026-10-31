#!/usr/bin/env python3
"""Wait for the frozen common-validation queue, then run CPU-only selection."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.ch3_motivation.code.select_probe_common import ROOT, sha256, write_json


def audit_completed(queue):
    """A completion marker alone cannot establish complete or successful data."""
    queue = Path(queue)
    if not (queue / "completed.txt").exists():
        raise ValueError("common validation has no completion marker")
    result = json.loads((queue / "summary.json").read_text())
    inventory = json.loads((queue / "checkpoint_inventory.json").read_text())
    if (result["status"] not in ("complete", "complete_with_failures") or
            result["points"] != 240 or len(inventory) != 240 or result["quality_auc_computed"] is not False):
        raise ValueError("common validation queue is not a completed 240-weight comparison")
    planned = {str(Path(row["checkpoint"]).resolve()): row for row in inventory}
    if len(planned) != 240 or sorted(row["edge"] for row in result["edges"]) != [1, 2, 3, 4, 5]:
        raise ValueError("common validation inventory/edge coverage differs")
    observed, failed, summaries = set(), [], []
    for edge in result["edges"]:
        path = Path(edge["summary"])
        if sha256(path) != edge["summary_sha256"]:
            raise ValueError("completed common validation edge summary changed")
        summary = json.loads(path.read_text())
        if summary["common_validation"]["edge"] != edge["edge"] or summary["status"] != edge["status"]:
            raise ValueError("common validation edge/status differs")
        edge_paths = set()
        edge_failed = []
        for point in summary["points"]:
            checkpoint = str(Path(point["checkpoint"]).resolve())
            if checkpoint in observed or checkpoint not in planned or planned[checkpoint]["edge"] != edge["edge"]:
                raise ValueError("common validation checkpoint coverage differs")
            if point.get("checkpoint_sha256") != planned[checkpoint]["checkpoint_sha256"]:
                raise ValueError("common validation checkpoint hash differs")
            observed.add(checkpoint)
            edge_paths.add(checkpoint)
            if point["status"] == "failed":
                edge_failed.append(point)
            elif point["status"] != "validated":
                raise ValueError("common validation point is unfinished")
        if (edge_paths != {path for path, row in planned.items() if row["edge"] == edge["edge"]} or
                len(edge_paths) != edge["points"] or edge["failed"] != len(edge_failed) or
                summary["failed"] != len(edge_failed) or summary["validated"] != len(edge_paths) - len(edge_failed)):
            raise ValueError("common validation completed counts differ")
        failed.extend(dict(edge=edge["edge"], **point) for point in edge_failed)
        summaries.append(dict(edge=edge["edge"], path=str(path.resolve()), sha256=edge["summary_sha256"]))
    if len(observed) != 240 or (result["status"] == "complete") != (not failed):
        raise ValueError("common validation final success/failure status differs")
    return dict(points=240, failed=len(failed), failed_points=failed, summaries=summaries,
                inventory_sha256=sha256(queue / "checkpoint_inventory.json"))


def run(args):
    args.state.mkdir(parents=True, exist_ok=True)
    selector = Path(__file__).with_name("select_probe_common.py").resolve()
    try:
        if sha256(selector) != args.selector_sha256:
            raise ValueError("frozen common selector source changed")
        marker = args.queue / "completed.txt"
        write_json(args.state / "progress.json", dict(status="waiting_common_validation", pid=os.getpid(),
            wait_for=str(marker), predecessor_pid=args.wait_pid))
        while not marker.exists():
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError as error:
                raise RuntimeError("common validation predecessor exited without a completion marker") from error
            time.sleep(10)
        audit = audit_completed(args.queue)
        write_json(args.state / "input_audit.json", audit)
        if sha256(selector) != args.selector_sha256:
            raise ValueError("frozen common selector source changed while waiting")
        command = [sys.executable, str(selector), "--inventory", str(args.queue / "checkpoint_inventory.json"),
            "--common-root", str(args.common_root), "--expected-points", "240", "--output", str(args.output)]
        write_json(args.state / "progress.json", dict(status="running_cpu_selection", command=command,
            failed_validation_weights=audit["failed"]))
        subprocess.run(command, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        selection = json.loads((args.output / "selection.json").read_text())
        if selection["status"] != "recommendation_for_review" or not selection["all_five_edges_complete"]:
            raise ValueError(f"common selector did not produce a complete recommendation: {selection['status']}")
        if sha256(selector) != args.selector_sha256:
            raise ValueError("frozen common selector source changed during analysis")
        status = "complete_with_validation_failures" if audit["failed"] else "complete_for_root_review"
        write_json(args.state / "summary.json", dict(status=status, points=240,
            failed_validation_weights=audit["failed"], selector_sha256=args.selector_sha256,
            selection=str((args.output / "selection.json").resolve()),
            selection_sha256=sha256(args.output / "selection.json"), gpu_training_started=False,
            recommendation=selection["recommendation"]))
        name = "review_required.txt" if audit["failed"] else "completed.txt"
        (args.state / name).write_text(datetime.datetime.now().astimezone().isoformat() + "\n")
        print(json.dumps(dict(status=status, failed_validation_weights=audit["failed"])), flush=True)
        return 2 if audit["failed"] else 0
    except Exception as error:
        write_json(args.state / "failure.json", dict(status="failed", error_type=type(error).__name__,
            error=str(error), gpu_training_started=False,
            time=datetime.datetime.now().astimezone().isoformat()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--common-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--selector-sha256", required=True)
    sys.exit(run(parser.parse_args()))
