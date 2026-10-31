#!/usr/bin/env python3
"""Run one GPU's declared calibration jobs, sharing immutable edge preparation."""
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
import time

from experiments.ch3_motivation.code.common import ROOT, write_json
from experiments.ch3_motivation.code.run_probe_calibration import run
from hstu_kvcache.common.artifacts import sha256


def main(args):
    plan = json.loads(args.plan.read_text())
    identity = dict(plan=str(args.plan.resolve()), plan_sha256=sha256(args.plan), physical_gpu=args.gpu,
                    source_sha256=plan["source_sha256"], input_sha256=plan["input_sha256"])
    output = args.output
    config = output / "configuration.json"
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError("calibration sweep identity changed")
    write_json(config, identity)

    def verify():
        for path, digest in {**identity["source_sha256"], **identity["input_sha256"],
                             str(args.plan.resolve()): identity["plan_sha256"]}.items():
            if sha256(Path(path) if Path(path).is_absolute() else ROOT / path) != digest:
                raise ValueError(f"calibration sweep frozen source/input changed: {path}")

    jobs = [job for job in plan["jobs"] if job["gpu"] == args.gpu]
    if not jobs:
        raise ValueError("no declared jobs for this physical GPU")
    points, completed, cache = [], [], {}
    started, last_edge = time.perf_counter(), None
    for index, job in enumerate(jobs):
        verify()
        if job["edge"] != last_edge:
            cache.clear()
        last_edge = job["edge"]
        values = dict(plan["fixed_settings"], **job)
        name = values.pop("name")
        for field in ("input_root", "output", "sampling_manifest"):
            values[field] = Path(values[field])
        write_json(output / "progress.json", dict(status="running", index=index, total=len(jobs),
            edge=job["edge"], recipe=name, completed=completed, seconds=time.perf_counter()-started))
        result = run(argparse.Namespace(**values), preparation_cache=cache)
        labels = [stage["method_label"] for stage in result["stages"]]
        if result["status"] != "fitted" or labels != ["f", "fg", "fgh"]:
            raise ValueError("sampling sweep requires the declared three-stage ladder")
        for stage in result["stages"]:
            if sha256(Path(stage["checkpoint"])) != stage["checkpoint_sha256"] or stage["quality_auc_computed"]:
                raise ValueError("sampling stage checkpoint or scope changed")
            chosen = next(row for row in stage["history"] if row["step"] == stage["best_step"])
            points.append(dict(name=name, edge=job["edge"], seed=values["seed"], method_label=stage["method_label"],
                fit_users=values["fit_users"], width=values["width"], recent_fraction=job["recent_fraction"],
                query_delta_rule=job["gap_rule"], best_step=stage["best_step"], completed_steps=stage["completed_steps"],
                training_view_validation_loss=chosen["validation_loss"],
                calibration_flops_estimate=stage["cumulative_calibration_flops_estimate"],
                checkpoint=stage["checkpoint"], checkpoint_sha256=stage["checkpoint_sha256"],
                actual_search_preparation_reused=stage["actual_search_preparation_reused"], seconds=stage["seconds"]))
        summary = values["output"] / "summary.json"
        completed.append(dict(name=name, edge=job["edge"], summary=str(summary), summary_sha256=sha256(summary)))
        temporary = output / "points.csv.writing"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
        temporary.replace(output / "points.csv")
    verify()
    write_json(output / "summary.json", dict(status="complete", identity=identity, jobs=len(jobs), points=points,
        completed=completed, seconds=time.perf_counter()-started, quality_auc_computed=False,
        scope=plan["scope"], cost_scope="each standalone method charges complete preparation; actual shared search preparation is recorded separately"))
    write_json(output / "progress.json", dict(status="complete", completed=completed, total=len(jobs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    main(parser.parse_args())
