#!/usr/bin/env python3
"""Describe saved common-validation contexts without selecting a new recipe.

Reads the 280/370/475 common selector's existing points/groups, never weights.
Incomplete methods retain their edge cells but receive no five-edge mean or
rank. All differences are after-minus-before absolute losses (negative helps).
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


EDGES = (1, 2, 3, 4, 5)
CONTEXTS = ("cutover", "gap_mix50", "gap_empirical100")
METRICS = ("loss", "read_mse", "score_mse", "raw_score_mse")
INPUT_FORMATS = {
    "ch3_temporal_common_selection_v1",
    "ch3_combined_sampling_common_selection_v1",
    "ch3_final_calibration_common_selection_v1",
}
FORMAT = "ch3_common_context_analysis_v1"


def _metrics(cell):
    metrics = cell["metrics"]
    users = metrics["per_user"]
    uids = [int(row["uid"]) for row in users]
    if metrics["users"] != 16 or len(uids) != 16 or len(set(uids)) != 16:
        raise ValueError("common context must retain the same sixteen reserved validation UIDs")
    result = {}
    for name in METRICS:
        value = float(metrics[name])
        per_user = [float(row[name]) for row in users]
        if not all(math.isfinite(x) and x >= 0 for x in [value, *per_user]):
            raise ValueError("nonfinite or negative common validation loss")
        if not math.isclose(value, math.fsum(per_user)/16, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("stored common loss is not the equal-user mean")
        result[name] = value
    if not math.isclose(result["loss"], result["read_mse"]+result["score_mse"], rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("common objective differs from read+score")
    return result, uids


def aggregate_loaded(selection):
    """Aggregate an existing selector JSON; no checkpoint or model is loaded."""
    if selection.get("format") not in INPUT_FORMATS:
        raise ValueError("expected a three-context common selection summary, not population quality or old cutover-only losses")
    points = {}
    groups = {}
    for group in selection.get("groups", []):
        key = (group["recipe"], group["method_label"], int(group["seed"]))
        if key in groups:
            raise ValueError("duplicate common method group")
        groups[key] = group
    for row in selection["points"]:
        key = (row["recipe"], row["method_label"], int(row["seed"]))
        edge = int(row["edge"])
        if edge not in EDGES or (key, edge) in points:
            raise ValueError("duplicate or unexpected common edge point")
        point = row["validation"]
        if point["status"] not in ("validated", "pending", "missing", "failed"):
            raise ValueError("common point is not an actual validation or explicit missing result")
        points[key, edge] = row
        groups.setdefault(key, {})

    point_rows, cells, fingerprints, validation_uids = [], {}, {}, {}
    for key in sorted(groups):
        recipe, method, seed = key
        for edge in EDGES:
            row = points.get((key, edge), {})
            point = row.get("validation", {"status": "missing", "contexts": {}})
            for context in CONTEXTS:
                cell = point.get("contexts", {}).get(context, {})
                values = dict.fromkeys(METRICS)
                if cell.get("status") == "validated":
                    values, uids = _metrics(cell)
                    fingerprint = cell.get("common_validation_sha256")
                    if not isinstance(fingerprint, str) or not fingerprint:
                        raise ValueError("validated common cell lacks its context fingerprint")
                    if fingerprints.setdefault((edge, context), fingerprint) != fingerprint:
                        raise ValueError("weights use different common context fingerprints")
                    if validation_uids.setdefault(edge, uids) != uids:
                        raise ValueError("common contexts or weights change validation UID ordering")
                record = dict(recipe=recipe, method_label=method, seed=seed, edge=edge, context=context,
                    point_status=point["status"], context_status=cell.get("status", "missing"),
                    checkpoint=row.get("checkpoint"), checkpoint_sha256=point.get("checkpoint_sha256"),
                    best_step=point.get("best_step"), completed_steps=point.get("completed_steps"),
                    step_zero=point.get("step_zero"), selection_eligible=point.get("selection_eligible", False),
                    error=point.get("error"), **values)
                point_rows.append(record)
                cells[key, edge, context] = record

    context_rows = []
    complete = {}
    for key, source_group in sorted(groups.items()):
        recipe, method, seed = key
        valid_edges = [edge for edge in EDGES if all(
            cells[key, edge, context]["point_status"] == "validated" and
            cells[key, edge, context]["context_status"] == "validated" for context in CONTEXTS)]
        complete[key] = valid_edges == list(EDGES)
        step_zero_edges = [edge for edge in EDGES if cells[key, edge, CONTEXTS[0]]["step_zero"] is True]
        for context in CONTEXTS:
            means = {name: math.fsum(cells[key, edge, context][name] for edge in EDGES)/5
                     if complete[key] else None for name in METRICS}
            stored = source_group.get("mean_loss_by_context")
            if complete[key] and stored is not None and not math.isclose(
                    means["loss"], float(stored[context]), rel_tol=1e-7, abs_tol=1e-12):
                raise ValueError("selector context mean disagrees with its five edge points")
            context_rows.append(dict(recipe=recipe, method_label=method, seed=seed, context=context,
                variant=source_group.get("variant"), width=source_group.get("width"),
                depth=source_group.get("depth"), fit_users=source_group.get("fit_users"),
                complete_five_edges=complete[key], validated_edges=valid_edges,
                missing_or_failed_edges=[edge for edge in EDGES if edge not in valid_edges],
                step_zero_edges=step_zero_edges, selection_eligible=source_group.get("selection_eligible", False),
                fgh_rank=None, **means))
    for row in context_rows:
        if row["method_label"] == "fgh" and row["complete_five_edges"]:
            row["fgh_rank"] = 1+sum(other["loss"] < row["loss"] for other in context_rows
                if other["method_label"] == "fgh" and other["complete_five_edges"] and
                other["seed"] == row["seed"] and other["context"] == row["context"])

    contrasts = []
    for recipe, seed in sorted({(key[0], key[2]) for key in groups}):
        for before, after in (("f", "fg"), ("fg", "fgh")):
            a, b = (recipe, before, seed), (recipe, after, seed)
            for context in CONTEXTS:
                edge_rows = []
                for edge in EDGES:
                    left, right = points.get((a, edge)), points.get((b, edge))
                    status = "stage_not_planned" if a not in groups or b not in groups else "missing_or_failed"
                    difference = None
                    if left is not None and right is not None:
                        ac, bc = cells[a, edge, context], cells[b, edge, context]
                        if all(cell["point_status"] == "validated" and cell["context_status"] == "validated" for cell in (ac, bc)):
                            av, bv = left["validation"], right["validation"]
                            identity = av.get("training_identity")
                            matched = (isinstance(identity, dict) and bool(identity) and identity == bv.get("training_identity")
                                and av.get("config") is not None and av["config"] == bv.get("config")
                                and av.get("variant") == bv.get("variant")
                                and av.get("fitting_users") == bv.get("fitting_users"))
                            status = "paired" if matched else "configuration_or_identity_mismatch"
                            if matched:
                                difference = bc["loss"]-ac["loss"]
                    edge_rows.append(dict(edge=edge, status=status, loss_after_minus_before=difference,
                        before_step_zero=left["validation"].get("step_zero") if left else None,
                        after_step_zero=right["validation"].get("step_zero") if right else None))
                all_paired = complete.get(a, False) and complete.get(b, False) and all(row["status"] == "paired" for row in edge_rows)
                contrasts.append(dict(recipe=recipe, seed=seed, before=before, after=after, context=context,
                    complete_five_edges=all_paired,
                    mean_loss_after_minus_before=math.fsum(row["loss_after_minus_before"] for row in edge_rows)/5 if all_paired else None,
                    edges=edge_rows))

    statuses = Counter(row["validation"]["status"] for row in points.values())
    expected = int(selection["expected_weights"])
    inventory_complete = len(points) == expected
    observed_cells = sum(row["context_status"] == "validated" for row in point_rows)
    terminal = inventory_complete and all(
        row["validation"]["status"] == "failed" or
        (row["validation"]["status"] == "validated" and all(
            row["validation"].get("contexts", {}).get(context, {}).get("status") == "validated"
            for context in CONTEXTS)) for row in points.values())
    status = ("complete_with_failures" if statuses["failed"] else "complete") if terminal else (
        "partial_common_validation" if observed_cells else "waiting_common_validation")
    return dict(format=FORMAT, status=status, input_format=selection["format"], expected_weights=expected,
        supplied_weights=len(points), point_status_counts=dict(statuses), inventory_complete=inventory_complete,
        validated_context_cells=observed_cells, complete_method_groups=sum(complete.values()),
        source_selection_status=selection.get("selection", {}).get("status"),
        source_recommendation=(selection.get("selection", {}).get("recommendation") or {}).get("recipe"),
        context_metrics=context_rows, points=point_rows, stage_differences=contrasts,
        context_definition="fixed cutover caches with synthetic gap features; exact 24/48 cutover-source and24/48 empirical-source assignments across three contexts, not chronological streaming replay",
        aggregation="equal five-edge mean within each context; incomplete methods retain edge cells but have no mean/rank",
        rank_scope="descriptive loss rank among all complete FGH recipes with the same seed/context; variants retained; no new selection rule",
        difference_convention="after-minus-before absolute loss; negative is improvement; zero and positive differences retained",
        quality_metrics_computed=False, selection_changed=False,
        scope="pre-release reserved validation diagnostics only; not formal population quality or independent confirmation")


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def analyze(selection_path, output=None):
    path = Path(selection_path).resolve()
    raw = path.read_bytes()
    result = aggregate_loaded(json.loads(raw))
    result["source"] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    source = Path(__file__).resolve()
    result["analyzer_source"] = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if output is not None:
        output = Path(output)
        if (output/"summary.json").exists():
            raise ValueError("retain the prior analysis snapshot; use a new output attempt")
        output.mkdir(parents=True, exist_ok=True)
        (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        _write_csv(output/"context_metrics.csv", result["context_metrics"])
        _write_csv(output/"point_metrics.csv", result["points"])
        _write_csv(output/"stage_differences.csv", [dict(
            **{key: value for key, value in row.items() if key != "edges"},
            **{f"edge_{edge['edge']}": edge["loss_after_minus_before"] for edge in row["edges"]},
            edge_statuses={str(edge["edge"]): edge["status"] for edge in row["edges"]}) for row in result["stage_differences"]])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="common selector summary.json, which contains points/groups")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.selection, args.output)
    print(json.dumps({key: report[key] for key in ("status", "supplied_weights", "validated_context_cells", "complete_method_groups")}))
