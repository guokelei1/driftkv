#!/usr/bin/env python3
"""Reserve label-free fitting users, disjoint from every scale evaluation panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import pyarrow.parquet as pq

from selective_recompute_2026_09.common import edge_name, sha256, write_json

PANEL_ROOT = ROOT / "results/selective_recompute_2026_09/panels"
OUTPUT = ROOT / "results/read_correction_2026_09/panels"
NAMESPACE = "read_correction_2026_09/calibration_uid_v0"


def choose_users(population, excluded, maximum=512):
    """Nested deterministic membership; no labels or model scores are inputs."""
    candidates = set(map(int, population)) - set(map(int, excluded))
    ordered = sorted(candidates, key=lambda uid: (
        hashlib.sha256(f"{NAMESPACE}/{uid}".encode()).digest(), uid))
    if len(ordered) < maximum:
        raise ValueError(f"only {len(ordered)} independent fitting users, requested {maximum}")
    return ordered[:maximum]


def prepare(scale, edge, *, maximum=512, panel_root=PANEL_ROOT, output_root=OUTPUT):
    excluded, panel_records = set(), []
    for index in range(1, 6):
        directory = panel_root / scale / edge_name(index)
        binding_path = directory / "binding.json"
        binding = json.loads(binding_path.read_text())
        users_path = ROOT / binding["users_file"]["path"]
        if sha256(users_path) != binding["users_file"]["sha256"]:
            raise RuntimeError(f"frozen panel users changed: {users_path}")
        users = json.loads(users_path.read_text())
        excluded.update(users["evaluation"])
        excluded.update(users["canary"])
        panel_records.append({"path": str(binding_path.relative_to(ROOT)),
                              "sha256": sha256(binding_path)})
    binding_path = panel_root / scale / edge_name(edge) / "binding.json"
    binding = json.loads(binding_path.read_text())
    source = binding["sources"]["reuse_binding"]
    source_path = ROOT / source["path"]
    if sha256(source_path) != source["sha256"]:
        raise RuntimeError("the retained Reuse population binding changed")
    population_binding = json.loads(source_path.read_text())
    population, request_sources = set(), []
    for rank in population_binding["ranks"]:
        path = source_path.parent / f"requests_rank{rank['rank']}.parquet"
        if sha256(path) != rank["sha256"]:
            raise RuntimeError(f"retained population requests changed: {path}")
        population.update(pq.read_table(path, columns=["uid"])["uid"].to_pylist())
        request_sources.append({"path": str(path.relative_to(ROOT)), "sha256": rank["sha256"]})
    selected = choose_users(population, excluded, maximum)
    record = {
        "status": "reserved", "scale": scale, "edge": edge_name(edge), "edge_index": edge,
        "cutover": binding["cutover"], "uids": selected, "maximum": maximum,
        "rule": "SHA256(namespace/uid) then UID; budgets are nested prefixes",
        "uid_hash_namespace": NAMESPACE,
        "population_users": len(population), "excluded_users": len(excluded),
        "excluded": "union of all five scale evaluation and canary UID sets",
        "selection_reads": ["UID only; no labels, logits or quality outcomes"],
        "history_rule": "strictly before release cutover; checked when histories are loaded",
        "evaluation_scope": "existing outcome-conditioned 3k panels are also development exploration panels",
        "panel_bindings": panel_records, "population_requests": request_sources,
        "source_binding": {"path": str(binding_path.relative_to(ROOT)), "sha256": sha256(binding_path)},
    }
    output = output_root / scale / edge_name(edge) / "calibration_users.json"
    if output.exists() and json.loads(output.read_text()) != record:
        raise RuntimeError(f"refusing to change existing fitting reservation: {output}")
    write_json(output, record)
    print(json.dumps({"status": "prepared", "scale": scale, "edge": edge_name(edge),
                      "users": len(selected), "output": str(output)}), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), nargs="+", default=["medium", "large", "max"])
    parser.add_argument("--edge", type=int, choices=range(1, 6), nargs="+", default=list(range(1, 6)))
    parser.add_argument("--maximum", type=int, default=512)
    parser.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for scale in args.scale:
        for edge in args.edge:
            prepare(scale, edge, maximum=args.maximum, panel_root=args.panel_root, output_root=args.output_root)
