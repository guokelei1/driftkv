#!/usr/bin/env python3
"""Aggregate all correction budgets on identical requests; no quality gate."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hstu_kvcache.evaluation.binary_metrics import binary_metrics
from read_correction_2026_09.common import (METHODS, OUTPUT, PANEL_ROOT, edge_name,
    method_directory, plan, sha256, signature, write_json)
from read_correction_2026_09.cost import CostModel, normalize_metrics
from read_correction_2026_09.worker import runtime_directory, verify_unit


def ordered(table):
    table = table.sort_by([("request_id", "ascending")]).combine_chunks()
    ids = table["request_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate request in one correction budget")
    return table


def validate_scores(rows, reference):
    rows = ordered(rows)
    for column in ("request_id", "uid", "query_timestamp"):
        if not np.array_equal(rows[column].to_numpy(), reference[column].to_numpy()):
            raise RuntimeError(f"correction and reference differ in {column}")
    if not np.isfinite(rows["hstu_logit"].to_numpy()).all():
        raise RuntimeError("nonfinite correction logits")
    if not np.array_equal(rows["total_flops"].to_numpy(), rows["correction_flops"].to_numpy()):
        raise RuntimeError("correction FLOPs do not sum to total")
    return rows


def adjudicate(scale, edge, output_root=OUTPUT, panel_root=PANEL_ROOT,
               world_size=4, calibration_root=None):
    output_root, panel_root = Path(output_root), Path(panel_root)
    name = edge_name(edge)
    panel_path = panel_root / scale / name / "binding.json"
    panel = json.loads(panel_path.read_text())
    histogram_names = ("full_history_hist", "append_prefix_hist", "initial_history_hist",
        "torch_full_history_hist", "torch_append_prefix_hist", "band_append_hist", "torch_band_append_hist")
    histograms = {key: Counter() for key in histogram_names}
    tables, completed_users, rank_records = {m: [] for m in METHODS}, [], []
    inputs_common = None
    controls = {}
    for rank in range(world_size):
        rank_dir = runtime_directory(output_root, scale, edge) / f"rank{rank}"
        complete_path = rank_dir / "complete.json"
        complete = json.loads(complete_path.read_text())
        inputs = complete["inputs"]
        if complete["status"] != "complete" or complete["rank"] != rank:
            raise RuntimeError("rank is not complete")
        if (complete["scale"], complete["edge"]) != (scale, name):
            raise RuntimeError("rank scale/edge differs")
        if complete["source_signature"] != signature(inputs):
            raise RuntimeError("rank signature differs from inputs")
        common = {k: v for k, v in inputs.items() if k != "uids"}
        if inputs_common is None:
            inputs_common = common
        elif inputs_common != common:
            raise RuntimeError("ranks used different inputs/calibration/sources")
        if inputs["panel_binding"] != sha256(panel_path) or inputs["world_size"] != world_size:
            raise RuntimeError("rank panel/world size differs")
        uids = inputs["uids"]
        if len(uids) != len(set(uids)) or len(uids) != complete["users"]:
            raise RuntimeError("rank users inconsistent")
        unit_paths = sorted(rank_dir.glob("unit_*.json"))
        if len(unit_paths) != complete["units"]:
            raise RuntimeError("completed rank is missing user units")
        unit_users, unit_requests = [], 0
        for path in unit_paths:
            unit = json.loads(path.read_text())
            selected = uids[len(unit_users):len(unit_users) + len(unit["uids"])]
            verify_unit(unit, complete["source_signature"], selected)
            unit_users.extend(unit["uids"])
            unit_requests += unit["requests"]
            for key, histogram in histograms.items():
                histogram.update({(str(k) if "band_append" in key else int(k)): int(v)
                                  for k, v in unit["stats"].get(key, {}).items()})
            for key, value in unit["controls"].items():
                if key.endswith("error"):
                    controls[key] = max(controls.get(key, 0.0), value)
            for method in METHODS:
                record = unit["outputs"][method]
                table = pq.read_table(record["path"])
                if len(table) != record["rows"] or len(table) != unit["requests"] * len(inputs["budgets"]):
                    raise RuntimeError("shard row count differs from sealed budgets/requests")
                if set(table["uid"].to_pylist()) != set(unit["uids"]):
                    raise RuntimeError("shard users differ from seal")
                tables[method].append(table)
        if unit_users != uids or unit_requests != complete["requests"]:
            raise RuntimeError("rank totals differ from its units")
        completed_users.extend(uids)
        rank_records.append({"path": str(complete_path), "sha256": sha256(complete_path),
            "users": len(uids), "requests": unit_requests})
    if len(completed_users) != len(set(completed_users)):
        raise RuntimeError("a user appears on multiple ranks")
    partition = inputs_common["partition"]
    requests_path = panel_path.parent / f"{partition}_requests.parquet"
    if sha256(requests_path) != inputs_common["requests"] or sha256(requests_path) != panel["files"][f"{partition}_requests"]["sha256"]:
        raise RuntimeError("frozen request panel changed")
    reference = pq.read_table(requests_path)
    available_users = set(reference["uid"].to_pylist())
    if not set(completed_users) <= available_users:
        raise RuntimeError("scored users are outside the panel")
    reference = ordered(reference.filter(pc.is_in(reference["uid"],
        value_set=pa.array(completed_users, type=pa.int64()))))
    if len(reference) != sum(r["requests"] for r in rank_records):
        raise RuntimeError("same-user reference has a different request count")
    if sum(histograms["full_history_hist"].values()) != len(reference):
        raise RuntimeError("Full histogram does not cover requests once")
    if sum(histograms["initial_history_hist"].values()) != len(completed_users):
        raise RuntimeError("initial history histogram does not cover users once")
    cost = CostModel.for_scale(scale, plan()["attention_backend"])
    denominator = cost.workload_denominator(histograms["full_history_hist"], histograms["append_prefix_hist"],
        append_new_kv_only=False, torch_full_history_hist=histograms["torch_full_history_hist"],
        torch_append_prefix_hist=histograms["torch_append_prefix_hist"],
        band_append_hist=histograms["band_append_hist"], torch_band_append_hist=histograms["torch_band_append_hist"])
    labels = reference["label"].to_numpy()
    full = binary_metrics(labels, reference["full_logit"].to_numpy())
    reuse = binary_metrics(labels, reference["reuse_logit"].to_numpy())
    points = []
    for method in METHODS:
        table = pa.concat_tables(tables[method])
        if set(table["budget"].to_pylist()) != set(inputs_common["budgets"]):
            raise RuntimeError("missing/unexpected correction budget")
        method_points = []
        for budget in inputs_common["budgets"]:
            rows = validate_scores(table.filter(pc.equal(table["budget"], budget)), reference)
            cal_files = inputs_common["calibrations"][method][str(budget)]
            for item in cal_files.values():
                if sha256(item["path"]) != item["sha256"]:
                    raise RuntimeError("calibration artifact changed")
            calibration = json.loads(Path(cal_files["json"]["path"]).read_text())
            calibration_flops = int(calibration["cost"]["calibration_flops"])
            forward = int(pc.sum(rows["correction_flops"]).as_py())
            extra = forward + calibration_flops
            metric = binary_metrics(labels, rows["hstu_logit"].to_numpy())
            point = {"baseline": method, "method": method, "scale": scale, "edge": name,
                "budget": budget, "kind": "measurement", "evaluation_role": "development_exploration",
                "partition": partition, "users": len(completed_users), "requests": len(reference),
                "full_auc": full["ROC_AUC"], "reuse_auc": reuse["ROC_AUC"], "baseline_auc": metric["ROC_AUC"],
                "metrics": metric, "correction_flops": forward, "calibration_flops": calibration_flops,
                "extra_flops": extra, **denominator,
                **normalize_metrics(full_auc=full["ROC_AUC"], reuse_auc=reuse["ROC_AUC"],
                    baseline_auc=metric["ROC_AUC"], extra_flops=extra,
                    full_minus_reuse_flops=denominator["full_minus_reuse_flops"])}
            method_points.append(point)
        for endpoint, metric, extra in (("reuse", reuse, 0), ("full", full, denominator["full_minus_reuse_flops"])):
            method_points.append({"baseline": method, "method": method, "scale": scale,
                "edge": name, "budget": endpoint, "kind": "reference",
                "partition": partition, "evaluation_role": "development_exploration",
                "users": len(completed_users), "requests": len(reference),
                "full_auc": full["ROC_AUC"], "reuse_auc": reuse["ROC_AUC"],
                "baseline_auc": metric["ROC_AUC"], "metrics": metric,
                "correction_flops": 0, "calibration_flops": 0, "extra_flops": extra, **denominator,
                **normalize_metrics(full_auc=full["ROC_AUC"], reuse_auc=reuse["ROC_AUC"],
                    baseline_auc=metric["ROC_AUC"], extra_flops=extra,
                    full_minus_reuse_flops=denominator["full_minus_reuse_flops"])})
        write_json(method_directory(method, scale, edge, output_root=output_root) / "summary.json", {
            "status": "complete", "evaluation_role": "development_exploration", "method": method,
            "scale": scale, "edge": name, "current_full": full, "current_reuse": reuse,
            "points": method_points, "rank_records": rank_records})
        points.extend(method_points)
    report = {"status": "complete", "evaluation_role": "development_exploration",
        "scale": scale, "edge": name, "partition": partition, "users": len(completed_users),
        "available_partition_users": len(available_users), "requests": len(reference),
        "whole_partition": set(completed_users) == available_users,
        "current_full": full, "current_reuse": reuse, "cost": denominator,
        "cost_convention": "common eager query reads cancel; correction forwards and complete per-budget calibration charged; Full-history minus actual rolling append denominator; no backend speedup credit",
        "histograms": {k: dict(v) for k, v in histograms.items()}, "controls": controls,
        "execution_source_hashes": inputs_common["execution_sources"],
        "rank_records": rank_records, "points": points,
        "scope": "outcome-conditioned development users; all signed outcomes retained; no population or held-out claim"}
    write_json(runtime_directory(output_root, scale, edge) / "summary.json", report)
    collect_summary(output_root, allow_partial=True)
    return report


def collect_summary(output_root=OUTPUT, *, allow_partial=False):
    directory = Path(output_root) / "runtime" / "development" / plan()["revision"]
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*/v*_to_v*/summary.json"))]
    expected = {(scale, edge_name(edge)) for scale in plan()["scales"] for edge in plan()["edges"]}
    observed = {(r["scale"], r["edge"]) for r in records}
    if len(observed) != len(records) or not observed <= expected:
        raise RuntimeError("unexpected or duplicate edge summaries")
    if not allow_partial and observed != expected:
        raise RuntimeError(f"need all 15 edges; have {len(observed)}")
    if len({r["partition"] for r in records}) > 1:
        raise RuntimeError("cannot mix canary and evaluation partitions")
    if len({signature(r["execution_source_hashes"]) for r in records}) > 1:
        raise RuntimeError("edge summaries contain different scoring source versions")
    report = {"status": "complete" if observed == expected else "partial",
        "evaluation_role": "development_exploration", "completed_edges": len(records),
        "completed_curves": len(records) * len(METHODS), "expected_edges": len(expected),
        "points": [p for r in records for p in r["points"]]}
    write_json(Path(output_root) / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"))
    parser.add_argument("--edge", type=int, choices=range(1, 6))
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.all:
        for scale in plan()["scales"]:
            for edge in plan()["edges"]:
                directory = runtime_directory(args.output_root, scale, edge)
                if args.allow_partial and not all((directory / f"rank{rank}/complete.json").exists() for rank in range(args.world_size)):
                    continue
                adjudicate(scale, edge, args.output_root, args.panel_root, args.world_size, args.calibration_root)
        report = collect_summary(args.output_root, allow_partial=args.allow_partial)
    elif args.scale and args.edge:
        report = adjudicate(args.scale, args.edge, args.output_root, args.panel_root, args.world_size, args.calibration_root)
    else:
        parser.error("provide --all or --scale and --edge")
    print(json.dumps({k: v for k, v in report.items() if k not in ("points", "histograms", "execution_source_hashes")}))


if __name__ == "__main__":
    main()
