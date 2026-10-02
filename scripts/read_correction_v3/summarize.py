#!/usr/bin/env python3
"""Combine frozen Q-v2 results and complete-panel H-v3 compute endpoints.

Reads saved evidence only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path("/home/gkl/work/evokv")
OUTPUT = ROOT / "results/read_correction_2026_09/v3/budget_run"
Q_SUMMARY = ROOT / "results/read_correction_2026_09/v2/summary.json"
SCALES = ("medium", "large", "max")
EDGES = tuple(f"v{i}_to_v{i + 1}" for i in range(5))
EXPECTED = {(scale, edge) for scale in SCALES for edge in EDGES}
TARGETS = {60, 75, 90}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_point(point):
    if point["partition"] != "evaluation" or point["users"] != 3000:
        raise RuntimeError("formal summary requires the complete 3000-user evaluation panel")
    for name in ("full_auc", "reuse_auc", "baseline_auc", "relative_flops_percent", "recovery_percent"):
        if point[name] is None or not math.isfinite(point[name]):
            raise RuntimeError(f"undefined or nonfinite {name} must be reported, not silently omitted")
    if point["full_minus_reuse_flops"] <= 0 or point["full_auc"] <= point["reuse_auc"]:
        raise RuntimeError("frozen full-panel gap/compute denominator must remain positive")
    if point["kind"] == "measurement":
        if point["extra_flops"] != point["calibration_flops"] + point["correction_flops"]:
            raise RuntimeError("correction and calibration costs do not sum to extra compute")
    expected_x = 100 * point["extra_flops"] / point["full_minus_reuse_flops"]
    expected_y = 100 * (point["baseline_auc"] - point["reuse_auc"]) / (point["full_auc"] - point["reuse_auc"])
    if not math.isclose(point["relative_flops_percent"], expected_x, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("reported cost uses a different denominator")
    if not math.isclose(point["recovery_percent"], expected_y, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("reported AUC recovery does not match retained controls")


def references(report, first):
    common = {name: first[name] for name in (
        "method", "baseline", "scale", "edge", "partition", "users", "requests",
        "full_auc", "reuse_auc", "full_minus_reuse_flops", "full_history_flops", "reuse_append_flops")}
    for name, metric, percent in (("reuse", report["current_reuse"], 0.),
                                   ("full", report["current_full"], 100.)):
        yield dict(common, kind="reference", budget=name, budget_kind="reference",
                   evaluation_role="development_exploration", source_revision="v3", source_reused=False,
                   baseline_auc=metric["ROC_AUC"], metrics=metric,
                   calibration_flops=0, correction_flops=0,
                   extra_flops=0 if name == "reuse" else first["full_minus_reuse_flops"],
                   auc_gap=first["full_auc"] - first["reuse_auc"],
                   relative_flops_percent=percent, recovery_percent=percent)


def collect_summary(output_root=OUTPUT, *, q_summary=Q_SUMMARY, evaluation_root=None, allow_partial=False):
    output_root, q_summary = Path(output_root), Path(q_summary)
    evaluation_root = Path(evaluation_root) if evaluation_root else output_root / "evaluation"
    frozen = json.loads(q_summary.read_text())
    if frozen["status"] != "complete":
        raise RuntimeError("Q source must be the completed frozen v2 summary")
    q_points = [copy.deepcopy(point) for point in frozen["points"] if point["method"] == "query_only"]
    q_by_edge = {}
    for point in q_points:
        validate_point(point)
        point.update(source_revision="v2", source_reused=True,
                     budget_kind="calibration_users" if point["kind"] == "measurement" else "reference")
        if point["kind"] == "measurement":
            point["calibration_users"] = int(point["budget"])
            q_by_edge.setdefault((point["scale"], point["edge"]), []).append(point)
    if set(q_by_edge) != EXPECTED:
        raise RuntimeError("frozen Q source must cover all 15 adjacent edges")
    for points in q_by_edge.values():
        if len(points) != 4 or {point["budget"] for point in points} != {128, 256, 512, 1024}:
            raise RuntimeError("frozen Q curve must retain every original calibration budget")
    points, observed, source_records, source_signatures = list(q_points), set(), [], set()
    for path in sorted(evaluation_root.glob("*/v*_to_v*/summary.json")):
        report = json.loads(path.read_text())
        key = report["scale"], report["edge"]
        if key not in EXPECTED or key in observed:
            raise RuntimeError("unexpected or duplicate H edge summary")
        if report["status"] != "complete" or report["probe_only"] or report["partition"] != "evaluation" or report["users"] != 3000:
            raise RuntimeError("small probes or unfinished evaluations cannot enter the complete-panel result")
        if report["method"] != "history_conditioned" or set(report["targets"]) != TARGETS:
            raise RuntimeError("H summary method or compute targets differ")
        if len(report["points"]) != 3 or {point["target_cost_percent"] for point in report["points"]} != TARGETS:
            raise RuntimeError("every H edge must retain all three compute endpoints")
        score_path = Path(report["scores"]["path"])
        if not score_path.is_absolute():
            score_path = ROOT / score_path
        if sha256(score_path) != report["scores"]["sha256"]:
            raise RuntimeError(f"retained H scores changed: {score_path}")
        source_signatures.add(json.dumps(report["inputs"]["execution_sources"], sort_keys=True))
        q_reference = q_by_edge[key][0]
        for original in report["points"]:
            point = copy.deepcopy(original)
            validate_point(point)
            if (point["scale"], point["edge"]) != key or point["method"] != "history_conditioned" or point["calibration_users"] != 512:
                raise RuntimeError("H point must be the same edge and fixed C512 history correction")
            if point["budget"] != point["target_cost_percent"]:
                raise RuntimeError("H budget field denotes compute target, not calibration-user count")
            for name in ("users", "requests", "full_minus_reuse_flops", "full_history_flops", "reuse_append_flops"):
                if point[name] != q_reference[name]:
                    raise RuntimeError(f"Q and H panels/cost conventions differ in {name}")
            for name in ("full_auc", "reuse_auc"):
                if not math.isclose(point[name], q_reference[name], rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(f"Q and H controls differ in {name}")
            point.update(source_revision="v3", source_reused=False, budget_kind="compute_target_percent")
            points.append(point)
        points.extend(references(report, report["points"][0]))
        observed.add(key)
        source_records.append({"scale": key[0], "edge": key[1], "path": str(path.resolve()), "sha256": sha256(path)})
    if len(source_signatures) > 1:
        raise RuntimeError("H summaries contain multiple execution-source versions")
    if not allow_partial and observed != EXPECTED:
        raise RuntimeError(f"need all 15 H edges; have {len(observed)}")
    actual = [point for point in points if point["kind"] == "measurement"]
    aggregates = []
    for method, budgets in (("query_only", (128, 256, 512, 1024)), ("history_conditioned", (60, 75, 90))):
        for scale in SCALES:
            for budget in budgets:
                subset = [p for p in actual if (p["method"], p["scale"], p["budget"]) == (method, scale, budget)]
                if subset:
                    aggregates.append({"method": method, "scale": scale, "budget": budget, "edges": len(subset),
                        **{name: sum(p[name] for p in subset) / len(subset)
                           for name in ("baseline_auc", "recovery_percent", "relative_flops_percent")}})
    h_points = [p for p in actual if p["method"] == "history_conditioned"]
    result = {"status": "complete" if observed == EXPECTED else "partial", "revision": "v3",
        "evaluation_role": "development_exploration", "completed_edges": len(observed), "expected_edges": 15,
        "completed_curves": 15 + len(observed), "measurement_points": len(actual), "expected_measurement_points": 105,
        "missing_history_edges": [dict(scale=s, edge=e) for s, e in sorted(EXPECTED - observed)],
        "query_source": {"path": str(q_summary.resolve()), "sha256": sha256(q_summary), "revision": "v2", "reused": True},
        "history_sources": source_records, "points": points, "equal_edge_means": aggregates,
        "history_cost_band": {"target_percent": [60, 90],
            "observed_min": min((p["relative_flops_percent"] for p in h_points), default=None),
            "observed_max": max((p["relative_flops_percent"] for p in h_points), default=None),
            "all_within_band": all(60 <= p["relative_flops_percent"] <= 90 for p in h_points) if h_points else None},
        "budget_definitions": {"query_only": "calibration users; frozen v2 values128/256/512/1024",
            "history_conditioned": "target extra-compute percent60/75/90; fixed512 fitting users and16 independent teacher-validation users"},
        "cost_convention": "(complete calibration + additional correction inference) / (Full-history compute - actual rolling Reuse append compute); common query read cancels; Reuse0 and Recompute100",
        "scope": "outcome-conditioned development panels; Medium/Large guide development; Max reported without method retuning; all signed outcomes retained"}
    write_json(output_root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--q-summary", type=Path, default=Q_SUMMARY)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = collect_summary(args.output_root, q_summary=args.q_summary,
                             evaluation_root=args.evaluation_root, allow_partial=args.allow_partial)
    print(json.dumps({key: result[key] for key in ("status", "completed_edges", "completed_curves", "measurement_points", "history_cost_band")}))


if __name__ == "__main__":
    main()
