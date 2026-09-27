#!/usr/bin/env python3
"""Aggregate sealed baseline shards on exactly the completed users' requests."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from hstu_kvcache.evaluation.binary_metrics import binary_metrics
from selective_recompute_2026_09.common import METHODS, OUTPUT, budgets, edge_name, plan, sha256, signature, write_json
from selective_recompute_2026_09.cost import CostModel, normalized_point


def resolved(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def verify_file(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise RuntimeError(f"sealed file changed: {path}")


def ordered(table: pa.Table) -> pa.Table:
    table = table.sort_by([("request_id", "ascending")]).combine_chunks()
    ids = table["request_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate requests within one baseline budget")
    return table


def normalize_metrics(*, full_auc, reuse_auc, baseline_auc, extra_flops, full_minus_reuse_flops):
    """Small canaries can have a single label; their FLOPs remain measurable."""
    if full_auc is None or reuse_auc is None or baseline_auc is None:
        return {"auc_gap": None, "recovery_percent": None,
                "relative_flops_percent": (100 * extra_flops / full_minus_reuse_flops
                                           if full_minus_reuse_flops != 0 else None)}
    return normalized_point(full_auc=full_auc, reuse_auc=reuse_auc, baseline_auc=baseline_auc,
                            extra_flops=extra_flops, full_minus_reuse_flops=full_minus_reuse_flops)


def collect_summary(output_root: Path, *, allow_partial: bool = False) -> dict:
    """Write the compact plotting input; formal completeness means all 60 curves."""
    edges = []
    for path in sorted((output_root / "runtime").glob("*/v*_to_v*/summary.json")):
        record = json.loads(path.read_text())
        if record["status"] != "complete":
            raise RuntimeError(f"incomplete edge summary: {path}")
        edges.append(record)
    expected = {(scale, edge_name(edge)) for scale in plan()["scales"] for edge in plan()["edges"]}
    observed = {(row["scale"], row["edge"]) for row in edges}
    if len(observed) != len(edges) or not observed <= expected:
        raise RuntimeError("duplicate or out-of-scope edge summaries")
    if not allow_partial and observed != expected:
        raise RuntimeError(f"formal plot requires all 15 edges; have {len(observed)}")
    partitions = {row["partition"] for row in edges}
    if len(partitions) > 1:
        raise RuntimeError("cannot mix development and formal panels in one result summary")
    versions = {signature(row["execution_source_hashes"]) for row in edges}
    if len(versions) > 1:
        raise RuntimeError("edge summaries contain different scoring source versions")
    points = [point for row in edges for point in row["points"]]
    result = {
        "status": "complete" if observed == expected else "partial",
        "partition": next(iter(partitions), None), "formal_evaluation": partitions == {"evaluation"},
        "completed_edges": len(edges), "completed_curves": len(edges) * len(METHODS),
        "expected_edges": len(expected), "expected_curves": len(expected) * len(METHODS),
        "missing_edges": [{"scale": scale, "edge": edge} for scale, edge in sorted(expected - observed)],
        "execution_source_hashes": edges[0]["execution_source_hashes"] if edges else {},
        "points": points,
        "scope": "frozen outcome-conditioned diagnostic users; no population effect-size claim",
    }
    write_json(output_root / "summary.json", result)
    return result


def adjudicate(
    scale: str, edge: int, output_root: Path,
    panel_root: Path = OUTPUT / "panels", world_size: int = 4,
    calibration_root: Path | None = None,
) -> dict:
    output_root, panel_root = Path(output_root), Path(panel_root)
    calibration_root = output_root if calibration_root is None else Path(calibration_root)
    name = edge_name(edge)
    panel_dir = panel_root / scale / name
    panel_path = panel_dir / "binding.json"
    panel = json.loads(panel_path.read_text())
    if panel["scale"] != scale or panel["edge"] != name:
        raise RuntimeError("panel version differs from requested edge")
    cal_path = calibration_root / "layer" / scale / name / "calibration.json"
    calibration = json.loads(cal_path.read_text())
    if calibration["status"] != "complete" or calibration["scale"] != scale or calibration["edge"] != name:
        raise RuntimeError("layer calibration is incomplete or belongs to another edge")
    if calibration["panel_binding_sha256"] != sha256(panel_path):
        raise RuntimeError("layer calibration belongs to another user panel")
    cost = CostModel.for_scale(scale, plan()["attention_backend"])
    expected_budgets = {method: [item["name"] for item in configs]
                        for method, configs in budgets(cost.num_layers).items()}
    tables = {method: [] for method in METHODS}
    histograms = {key: Counter() for key in (
        "full_history_hist", "append_prefix_hist", "initial_history_hist",
        "torch_full_history_hist", "torch_append_prefix_hist",
        "band_append_hist", "torch_band_append_hist",
    )}
    completed_users, sources, rank_records = [], None, []
    inputs_common = None
    partition = None
    for rank in range(world_size):
        rank_dir = output_root / "runtime" / scale / name / f"rank{rank}"
        complete_path = rank_dir / "complete.json"
        complete = json.loads(complete_path.read_text())
        inputs = complete["inputs"]
        if complete["status"] != "complete" or complete["rank"] != rank:
            raise RuntimeError(f"rank {rank} is not a complete matching rank")
        if complete["scale"] != scale or complete["edge"] != name:
            raise RuntimeError("rank output belongs to another scale/edge")
        if complete["source_signature"] != signature(inputs):
            raise RuntimeError("rank source signature disagrees with its inputs")
        common = {key: value for key, value in inputs.items() if key != "uids"}
        if inputs_common is None:
            inputs_common, sources, partition = common, inputs["execution_sources"], inputs["partition"]
        elif common != inputs_common:
            raise RuntimeError("ranks used different scoring sources, calibration or request panels")
        if partition not in {"evaluation", "canary"} or complete["partition"] != partition:
            raise RuntimeError("unrecognized or mixed rank partition")
        if inputs["panel_binding"] != sha256(panel_path) or inputs["calibration"] != sha256(cal_path):
            raise RuntimeError("rank input panel/calibration changed")
        if inputs["scale"] != scale or inputs["edge"] != name:
            raise RuntimeError("rank bound inputs name another edge")
        if calibration["execution_source_hashes"] != sources:
            raise RuntimeError("calibration and scored ranks used different source versions")
        uids = [int(uid) for uid in inputs["uids"]]
        if len(uids) != complete["users"] or len(uids) != len(set(uids)):
            raise RuntimeError("rank user count or identity is inconsistent")
        unit_paths = sorted(rank_dir.glob("unit_*.json"))
        if len(unit_paths) != complete["units"]:
            raise RuntimeError("completed rank is missing sealed user units")
        unit_users, unit_requests = [], 0
        for unit_path in unit_paths:
            unit = json.loads(unit_path.read_text())
            if unit["source_signature"] != complete["source_signature"]:
                raise RuntimeError("unit source signature differs from completed rank")
            if set(unit["outputs"]) != set(METHODS):
                raise RuntimeError("sealed unit does not contain all four baselines")
            unit_users.extend(int(uid) for uid in unit["uids"])
            unit_requests += int(unit["requests"])
            for key, histogram in histograms.items():
                optional = key.startswith("torch_") or key == "band_append_hist"
                values = unit["stats"].get(key, {}) if optional else unit["stats"][key]
                histogram.update({(str(length) if "band_append" in key else int(length)): int(count)
                                  for length, count in values.items()})
            for method in METHODS:
                record = unit["outputs"][method]
                path = resolved(record["path"])
                verify_file(path, record["sha256"])
                table = pq.read_table(path)
                if len(table) != record["rows"] or set(table["uid"].to_pylist()) != set(unit["uids"]):
                    raise RuntimeError("baseline shard rows/users disagree with its unit seal")
                if len(table) != unit["requests"] * len(expected_budgets[method]):
                    raise RuntimeError("baseline shard request/budget count differs from its unit")
                tables[method].append(table)
        if unit_users != uids or unit_requests != complete["requests"]:
            raise RuntimeError("completed rank and its user units disagree")
        completed_users.extend(uids)
        rank_records.append({"path": str(complete_path), "sha256": sha256(complete_path),
                             "source_signature": complete["source_signature"], "users": len(uids),
                             "requests": complete["requests"]})
    if len(completed_users) != len(set(completed_users)):
        raise RuntimeError("a user was evaluated on more than one rank")
    requests_path = panel_dir / f"{partition}_requests.parquet"
    verify_file(requests_path, inputs_common["requests"])
    verify_file(requests_path, panel["files"][f"{partition}_requests"]["sha256"])
    reference = pq.read_table(requests_path)
    panel_users = set(reference["uid"].to_pylist())
    if not set(completed_users) <= panel_users:
        raise RuntimeError("scored users are outside the frozen panel")
    if partition == "evaluation" and (set(completed_users) != panel_users or calibration["probe_only"]):
        raise RuntimeError("formal evaluation requires the whole panel and full calibration")
    reference = reference.filter(pc.is_in(reference["uid"], value_set=pa.array(completed_users, type=pa.int64())))
    reference = ordered(reference)
    count = len(reference)
    if sum(row["requests"] for row in rank_records) != count:
        raise RuntimeError("completed rank request totals differ from the same-user reference")
    if sum(histograms["full_history_hist"].values()) != count:
        raise RuntimeError("Full-cost histogram does not cover each scored request once")
    if sum(histograms["initial_history_hist"].values()) != len(completed_users):
        raise RuntimeError("initial history histogram does not cover each scored user once")
    for key in ("full_history_hist", "append_prefix_hist", "band_append_hist"):
        if any(count < 0 or count > histograms[key][length]
               for length, count in histograms[f"torch_{key}"].items()):
            raise RuntimeError("Torch execution histogram is not a subset of total workload")
    labels = reference["label"].to_numpy()
    full = binary_metrics(labels, reference["full_logit"].to_numpy())
    reuse = binary_metrics(labels, reference["reuse_logit"].to_numpy())
    auc_defined = full["ROC_AUC"] is not None
    if not auc_defined and partition == "evaluation":
        raise RuntimeError("formal diagnostic requests need both labels for an AUC curve")
    denominator = cost.workload_denominator(histograms["full_history_hist"], histograms["append_prefix_hist"],
                                             append_new_kv_only=False,
                                             torch_full_history_hist=histograms["torch_full_history_hist"],
                                             torch_append_prefix_hist=histograms["torch_append_prefix_hist"],
                                             band_append_hist=histograms["band_append_hist"],
                                             torch_band_append_hist=histograms["torch_band_append_hist"])
    calibration_flops = int(calibration["cost"]["calibration_flops"])
    all_points = []
    for method in METHODS:
        method_rows = pa.concat_tables(tables[method])
        if set(method_rows["budget"].to_pylist()) != set(expected_budgets[method]):
            raise RuntimeError("baseline output has missing or unexpected budget levels")
        method_points = []
        for budget in expected_budgets[method]:
            rows = ordered(method_rows.filter(pc.equal(method_rows["budget"], budget)))
            for column in ("request_id", "uid", "query_timestamp"):
                if not np.array_equal(rows[column].to_numpy(), reference[column].to_numpy()):
                    raise RuntimeError(f"{method}/{budget} does not match the reference {column}")
            if not np.isfinite(rows["hstu_logit"].to_numpy()).all():
                raise RuntimeError("nonfinite baseline logits")
            if not np.array_equal(rows["total_flops"].to_numpy(),
                                  rows["recompute_flops"].to_numpy() + rows["selection_flops"].to_numpy()):
                raise RuntimeError("repair/selection FLOPs do not sum to the per-request total")
            measured = binary_metrics(labels, rows["hstu_logit"].to_numpy())
            components = {key: int(pc.sum(rows[key]).as_py()) for key in ("recompute_flops", "selection_flops")}
            components["calibration_flops"] = calibration_flops if method == "layer" else 0
            extra = sum(components.values())
            point = {"baseline": method, "scale": scale, "edge": name, "budget": budget,
                     "kind": "measurement", "partition": partition, "auc_defined": auc_defined, "requests": count,
                     "users": len(completed_users), "full_auc": full["ROC_AUC"],
                     "reuse_auc": reuse["ROC_AUC"], "baseline_auc": measured["ROC_AUC"],
                     "metrics": measured, "extra_flops": extra, **components, **denominator,
                     "selection_sort_comparisons_estimate": float(pc.sum(rows["selection_sort_comparisons_estimate"]).as_py()),
                     **normalize_metrics(full_auc=full["ROC_AUC"], reuse_auc=reuse["ROC_AUC"],
                                        baseline_auc=measured["ROC_AUC"], extra_flops=extra,
                                        full_minus_reuse_flops=denominator["full_minus_reuse_flops"])}
            method_points.append(point)
        for endpoint, metric, extra in (("reuse", reuse, 0), ("full", full, denominator["full_minus_reuse_flops"])):
            method_points.append({
                "baseline": method, "scale": scale, "edge": name, "budget": endpoint,
                "kind": "reference", "partition": partition, "auc_defined": auc_defined,
                "requests": count, "users": len(completed_users),
                "full_auc": full["ROC_AUC"], "reuse_auc": reuse["ROC_AUC"], "baseline_auc": metric["ROC_AUC"],
                "metrics": metric, "extra_flops": extra, **denominator,
                "calibration_flops": 0, "selection_flops": 0,
                "recompute_flops": denominator["full_history_flops"] if endpoint == "full" else 0,
                **normalize_metrics(full_auc=full["ROC_AUC"], reuse_auc=reuse["ROC_AUC"],
                                   baseline_auc=metric["ROC_AUC"], extra_flops=extra,
                                   full_minus_reuse_flops=denominator["full_minus_reuse_flops"]),
            })
        write_json(output_root / method / scale / name / "summary.json", {
            "status": "complete", "baseline": method, "scale": scale, "edge": name,
            "partition": partition, "auc_defined": auc_defined,
            "requests": count, "users": len(completed_users),
            "current_full": full, "current_reuse": reuse, "points": method_points,
            "execution_source_hashes": sources, "panel_binding_sha256": sha256(panel_path),
            "calibration_sha256": sha256(cal_path), "rank_records": rank_records,
        })
        all_points.extend(method_points)
    report = {
        "status": "complete", "scale": scale, "edge": name, "partition": partition,
        "auc_defined": auc_defined,
        "auc_note": ("both labels present" if auc_defined else
                     "single-label canary: AUC and recovery undefined; scoring and cost checks retained; no user reselection"),
        "requests": count, "users": len(completed_users), "available_partition_users": len(panel_users),
        "current_full": full, "current_reuse": reuse, "cost": denominator,
        "histograms": {key: dict(value) for key, value in histograms.items()},
        "execution_source_hashes": sources, "panel_binding_sha256": sha256(panel_path),
        "calibration_sha256": sha256(cal_path), "rank_records": rank_records, "points": all_points,
        "checks": ["completed rank and unit source signatures agree", "all baseline shard hashes verified",
                   "ranks have disjoint users and identical code/calibration inputs",
                   "each budget uses exactly the scored users' reference request IDs, UIDs and timestamps",
                   "Full/Reuse endpoints recomputed on those same requests",
                   "append cost subtracted once and profiling cost charged once per layer budget",
                   "negative and greater-than-100-percent normalized values preserved"],
    }
    write_json(output_root / "runtime" / scale / name / "summary.json", report)
    collect_summary(output_root, allow_partial=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"))
    parser.add_argument("--edge", type=int, choices=range(1, 6))
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--panel-root", type=Path, default=OUTPUT / "panels")
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.all:
        for scale in plan()["scales"]:
            for edge in plan()["edges"]:
                directory = args.output_root / "runtime" / scale / edge_name(edge)
                if args.allow_partial and not all((directory / f"rank{rank}/complete.json").exists() for rank in range(args.world_size)):
                    continue
                adjudicate(scale, edge, args.output_root, args.panel_root, args.world_size, args.calibration_root)
        report = collect_summary(args.output_root, allow_partial=args.allow_partial)
        print(json.dumps({key: report[key] for key in ("status", "completed_edges", "completed_curves")}))
    elif args.scale and args.edge:
        report = adjudicate(args.scale, args.edge, args.output_root, args.panel_root,
                            args.world_size, args.calibration_root)
        print(json.dumps({key: report[key] for key in ("status", "scale", "edge", "requests", "users")}))
    else:
        parser.error("provide --all, or --scale and --edge")


if __name__ == "__main__":
    main()
