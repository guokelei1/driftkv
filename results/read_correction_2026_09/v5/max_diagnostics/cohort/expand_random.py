#!/usr/bin/env python3
"""One fixed expansion of Max edge1; retains the original hash order/exclusions."""
from pathlib import Path
import hashlib
import json

from prepare_random import ROOT, OUTPUT, NAMESPACE, paired_population, metrics, sha256, write_json
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def main():
    previous_path = OUTPUT / "random256/max/v0_to_v1/manifest.json"
    previous = json.loads(previous_path.read_text())
    excluded = set()
    def visit(obj):
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            if key in ("uids", "train_uids", "validation_uids", "calibration_uids") and isinstance(value, list):
                excluded.update(int(uid) for uid in value)
            elif isinstance(value, dict):
                visit(value)
    for source in previous["exclusion_sources"]:
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"]:
            raise RuntimeError(f"original calibration exclusion source changed: {path}")
        visit(json.loads(path.read_text()))
    if len(excluded) != previous["excluded_total"]:
        raise RuntimeError("original excluded UID set did not reproduce")
    binding_path = ROOT / previous["sources"]["reuse_binding"]["path"]
    if sha256(binding_path) != previous["sources"]["reuse_binding"]["sha256"]:
        raise RuntimeError("original population binding changed")
    binding = json.loads(binding_path.read_text())
    all_uids = set()
    for rank in binding["ranks"]:
        path = binding_path.parent / f"requests_rank{rank['rank']}.parquet"
        all_uids.update(int(uid) for uid in pq.read_table(path, columns=["uid"])["uid"].to_numpy())
    selected = sorted(all_uids-excluded, key=lambda uid: (hashlib.sha256(f"{NAMESPACE}/{uid}".encode()).digest(), uid))[:2048]
    if len(selected) != 2048 or selected[:256] != previous["uids"]:
        raise RuntimeError("expansion must preserve the original 256-user hash prefix")
    population, _, sources = paired_population("max", 1)
    panel = population.filter(pc.is_in(population["uid"], value_set=pa.array(selected, type=pa.int64())))
    panel = panel.sort_by([("uid", "ascending"), ("query_timestamp", "ascending"), ("request_id", "ascending")])
    old_requests = pq.read_table(ROOT / previous["requests"]["path"])
    old_in_new = panel.filter(pc.is_in(panel["uid"], value_set=pa.array(previous["uids"], type=pa.int64())))
    if not old_in_new.equals(old_requests):
        raise RuntimeError("original 256-user requests or scores changed within expansion")
    folder = OUTPUT / "random2048/max/v0_to_v1"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "requests.parquet"
    if (folder / "manifest.json").exists():
        old = json.loads((folder / "manifest.json").read_text())
        if old["uids"] != selected or old["requests"]["sha256"] != sha256(path):
            raise RuntimeError("existing expansion changed")
        print(json.dumps({"status": "already_complete", "folder": str(folder)}))
        return
    pq.write_table(panel, path, compression="zstd")
    original = ROOT / previous["original_panel"]["path"]
    original_users = set(json.loads((original.parent / "users.json").read_text())["evaluation"])
    record = {**previous, "uids": selected,
        "selection": "same original SHA256(namespace/uid) ordering and exclusions, first2048; retains original256 prefix; all requests retained",
        "role": "one fixed random diagnostic expansion for precision; not a formal panel",
        "reason": "original256 cluster uncertainty wide; fixed2048 expansion chosen once, without score-based user choice",
        "previous_panel": {"path": str(previous_path.relative_to(ROOT)), "sha256": sha256(previous_path),
            "preserved_users": 256, "preserved_requests": len(old_requests), "requests_exactly_equal": True},
        "sources": sources, "overlap_original_selected_users": len(set(selected) & original_users),
        "requests": {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": len(panel)},
        "metrics_after_selection": metrics(panel)}
    write_json(folder / "manifest.json", record)
    print(json.dumps({"status": "complete", "folder": str(folder), "users": len(selected),
        "requests": len(panel), "overlap_original": record["overlap_original_selected_users"],
        "metrics_after_selection": record["metrics_after_selection"]}), flush=True)


if __name__ == "__main__":
    main()
