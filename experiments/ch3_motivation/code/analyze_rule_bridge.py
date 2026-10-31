#!/usr/bin/env python3
"""CPU comparison of the two repair lifetimes on the same frozen 512 users."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from experiments.ch3_motivation.code import common, run_selective
from experiments.ch3_motivation.code.audit_scenario_qualification import audit_blocks
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256


def _hash(value, *, sort=False):
    return hashlib.sha256(json.dumps(value, sort_keys=sort).encode()).hexdigest()


def paired_lifetimes(persistent, request_local, users, *, replicates=1000, seed=20261030):
    """AUC differences share UID multiplicities; endpoint ratios stay separate."""
    common.quality_role(users)
    groups = []
    for rows in (persistent, request_local):
        group = defaultdict(list)
        for row in rows:
            group[row["method"]].append(row)
        for values in group.values():
            values.sort(key=lambda row: tuple(row[key] for key in
                ("uid", "query_timestamp", "raw_item_id", "item_idx", "label")))
        groups.append(group)
    if not set(groups[1]) <= set(groups[0]):
        raise ValueError("request-local method has no persistent comparison")
    population = np.asarray(sorted(users), dtype=np.int64)
    rng = np.random.default_rng(seed)
    weights = np.asarray([np.bincount(rng.integers(len(users), size=len(users)), minlength=len(users))
                          for _ in range(replicates)], dtype=np.uint16)
    results = []
    for tag in sorted(set(groups[0]) & set(groups[1])):
        left, right = groups[0][tag], groups[1][tag]
        keys = lambda rows: [tuple(row[key] for key in
            ("uid", "query_timestamp", "raw_item_id", "item_idx", "label")) for row in rows]
        if keys(left) != keys(right) or not left or len(set(keys(left))) != len(left):
            raise ValueError("repair lifetimes do not share exactly the same feedback events")
        index = np.searchsorted(population, [row["uid"] for row in left])
        if np.any(index >= len(population)) or not np.array_equal(population[index], [row["uid"] for row in left]):
            raise ValueError("score UID outside the frozen comparison population")
        labels = np.asarray([row["label"] for row in left], dtype=np.float64)
        values = [np.asarray([row["score"] for row in rows]) for rows in (left, right)]
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError("nonfinite bridge score")
        points = [float(roc_auc_score(labels, value)) for value in values]
        delta = weighted_auc(values[1], labels, index, weights) - weighted_auc(values[0], labels, index, weights)
        finite = delta[np.isfinite(delta)]
        results.append(dict(method=tag, persistent_auc=points[0], request_local_auc=points[1],
            request_local_minus_persistent_auc=points[1] - points[0],
            request_local_minus_persistent_auc_ci95=np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None))
        if tag == "reuse":
            error = float(np.max(np.abs(values[1] - values[0])))
            if error > 1e-4:
                raise ValueError("unchanged Reuse trajectory differs across repair lifetimes")
    if not {"reuse", "rebuild"} <= {row["method"] for row in results}:
        raise ValueError("bridge comparison requires both independently recomputed endpoints")
    return dict(quality_users=len(users), results=results, reuse_maximum_absolute_score_difference=error,
        confidence=dict(replicates=replicates, seed=seed, shared_uid_multiplicity=True,
            scope="paired within the same development users; fitting weights held fixed"))


def analyze(edge, output, *, run_root=ROOT / "experiments/ch3_motivation/runs"):
    run_root = Path(run_root)
    original = run_root / f"e3/edge_{edge}/development/attempt_1"
    bridge = run_root / f"rule_sensitivity/edge_{edge}/development/attempt_1"
    p = json.loads((original / "summary.json").read_text())
    q = json.loads((bridge / "summary.json").read_text())
    for folder in (original, bridge):
        if json.loads((folder / "status.json").read_text())["state"] != "complete":
            raise ValueError("both repair-lifetime runs must complete before analysis")
    population_path = ROOT / "experiments/ch3_motivation/inputs/population.json"
    population = json.loads(population_path.read_text())
    users = population["development"]
    left, right = p["identity"], q["identity"]
    if (len(users) != 512 or right["users"] != users or left["evaluation_users_sha256"] != _hash(users)
            or left["population_sha256"] != sha256(population_path) or right["population_sha256"] != sha256(population_path)
            or left["input_binding_sha256"] != right["input_binding_sha256"]
            or left["checkpoint_sha256"] != right["models"] or left["settings"] != right["settings"]
            or right["semantics"] != "request_local" or right["profile_users"] != 64
            or any(right["sources"].get(path) != value for path, value in left["source_sha256"].items())):
        raise ValueError("bridge population, models, inputs, settings or shared sources differ")
    binding_path = ROOT / f"experiments/ch3_motivation/inputs/edge_{edge}/inputs.json"
    if sha256(binding_path) != left["input_binding_sha256"]:
        raise ValueError("original frozen input binding changed")
    binding = json.loads(binding_path.read_text())
    request_path = binding_path.parent / binding["requests"]["path"]
    for name in ("history", "requests"):
        if sha256(binding_path.parent / binding[name]["path"]) != left["input_hashes"][name]:
            raise ValueError("original frozen history or requests changed")
    expected_methods = {"reuse", "rebuild", "tail_20pct", "deviation_20pct"} | {
        f"layer_{width}layers" for width in (1, 2, 4, 6, 8)}
    if len(q["results"]) != len(expected_methods) or {row["method"] for row in q["results"]} != expected_methods:
        raise ValueError("request-local bridge does not contain all nine prescribed methods")
    selected = set(users)
    events = [row for row in pq.read_table(request_path).to_pylist() if row["uid"] in selected]
    raw, audit = audit_blocks(original / "quality_shards", users,
        dict(left, layer_profile_sha256=sha256(original / "layer_profile.json")),
        [row["method"] for row in p["results"]], events)
    selector = bridge / "selector_sensitivity.json"
    fingerprint = _hash(dict(right, selector_sensitivity_sha256=sha256(selector)), sort=True)
    local = []
    markers = sorted((bridge / "score_blocks").glob("part_*.json"))
    if [path.stem for path in markers] != ["part_000000", "part_000256"]:
        raise ValueError("request-local 512-user block coverage differs")
    for offset, marker in zip((0, 256), markers, strict=True):
        saved = json.loads(marker.read_text())
        parquet = marker.with_suffix(".parquet")
        if (saved["fingerprint"] != fingerprint or saved["users"] != users[offset:offset + 256]
                or saved["scores_sha256"] != sha256(parquet)):
            raise ValueError("request-local score block identity/hash differs")
        local.extend(pq.read_table(parquet).to_pylist())
    expected = {(row["uid"], row["query_timestamp"], row["raw_item_id"], row["item_idx"], row["label"], method)
                for row in events for method in [row["method"] for row in q["results"]]}
    actual = [(row["uid"], row["query_timestamp"], row["raw_item_id"], row["item_idx"], row["label"], row["method"]) for row in local]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("request-local feedback/method coverage differs")
    result = paired_lifetimes(raw, local, users, seed=20261030 + edge)
    for row in result["results"]:
        persistent = next(value for value in p["results"] if value["method"] == row["method"])
        ephemeral = next(value for value in q["results"] if value["method"] == row["method"])
        if abs(row["persistent_auc"] - persistent["auc"]) > 1e-14 or abs(row["request_local_auc"] - ephemeral["auc"]) > 1e-14:
            raise ValueError("lifetime summary AUC differs from actual scores")
        row.update(persistent_recovery_percent=persistent["recovery_percent"],
            request_local_recovery_percent=ephemeral["recovery_percent"],
            persistent_added_flops=persistent["added_flops"],
            request_local_added_flops=ephemeral["shared_reference_added_flops"],
            persistent_cost_percent_of_once_rebuild=persistent["percent_of_once_rebuild"],
            request_local_cost_percent_of_per_request_full=ephemeral["percent_of_per_request_full_repair"])
    result.update(state="complete", edge=edge, evaluation_window=p["evaluation_window"],
        persistent_reference_gap=p["reference_gap"], request_local_reference_gap=q["reference_gap"],
        persistent_reference_gap_ci95=p["confidence"]["reference_gap_ci95"],
        request_local_reference_gap_ci95=q["confidence"]["reference_gap_ci95"],
        both_reference_gap_points_positive=bool(p["reference_gap"] > 0 and q["reference_gap"] > 0),
        cost_denominators=dict(persistent_once_rebuild_flops=p["costs"]["actual_event_ledger"]["once_rebuild_flops"],
            request_local_full_repair_flops=q["costs"]["per_request_full_repair_flops"],
            request_local_standalone_full_minus_reuse_flops=q["costs"]["standalone_full_minus_reuse_flops"]),
        selector_comparison=json.loads(selector.read_text())["comparison"],
        audit=dict(persistent=audit, request_local_blocks=len(markers), request_local_scores=len(local),
            shared_population_inputs_models_and_sources=True),
        inputs={str(path): sha256(path) for path in (original / "summary.json", bridge / "summary.json", selector)},
        interpretation="same development population; distinct Full trajectories and cost denominators; descriptive rule sensitivity, not independent confirmation; a nonpositive reference gap makes algebraic recovery percentages unsuitable as recovered benefit")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    common.write_json(output / "summary.json", result)
    with (output / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["results"][0]))
        writer.writeheader()
        writer.writerows(result["results"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=int, choices=(1, 5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(dict(edge=args.edge, output=str(args.output), state=analyze(args.edge, args.output)["state"])))
