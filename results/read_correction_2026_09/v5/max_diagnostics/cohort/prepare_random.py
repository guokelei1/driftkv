#!/usr/bin/env python3
"""Frozen-seed diagnostic users from source UIDs, never selected by outcomes."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scripts.selective_recompute_2026_09.prepare import paired_population, metrics
from read_correction_v4.common import PANEL_ROOT, RESERVATIONS, sha256, write_json

OUTPUT = Path(__file__).resolve().parent
NAMESPACE = "read_correction_max_cohort_diagnostic_2026_09/seed17"


def calibration_exclusions():
    """All preserved correction calibration/reservation users, across scales."""
    excluded, records = set(), []
    for path in sorted((ROOT / "results/read_correction_2026_09").rglob("calibration*.json")):
        obj = json.loads(path.read_text())
        found = set()
        def visit(value):
            if not isinstance(value, dict):
                return
            for key, child in value.items():
                if key in ("uids", "train_uids", "validation_uids", "calibration_uids") and isinstance(child, list):
                    found.update(int(uid) for uid in child)
                elif isinstance(child, dict):
                    visit(child)
        visit(obj)
        if found:
            excluded.update(found)
            records.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "users": len(found)})
    return excluded, records


def main():
    excluded, exclusion_records = calibration_exclusions()
    jobs = (("medium", 1), ("large", 1), ("max", 1), ("max", 4))
    for scale, edge in jobs:
        name = f"v{edge-1}_to_v{edge}"
        source_panel = PANEL_ROOT / scale / name / "binding.json"
        old = json.loads(source_panel.read_text())
        binding_path = ROOT / old["sources"]["reuse_binding"]["path"]
        source_binding = json.loads(binding_path.read_text())
        # Selection precedes any read of labels or new correction outcomes.
        all_uids = set()
        for rank in source_binding["ranks"]:
            path = binding_path.parent / f"requests_rank{rank['rank']}.parquet"
            all_uids.update(int(uid) for uid in pq.read_table(path, columns=["uid"])["uid"].to_numpy())
        eligible = all_uids - excluded
        selected = sorted(eligible, key=lambda uid: (
            hashlib.sha256(f"{NAMESPACE}/{uid}".encode()).digest(), uid))[:256]
        if len(selected) != 256:
            raise RuntimeError("fewer than 256 independent eligible users")
        population, _, sources = paired_population(scale, edge)
        panel = population.filter(pc.is_in(population["uid"], value_set=pa.array(selected, type=pa.int64())))
        panel = panel.sort_by([("uid", "ascending"), ("query_timestamp", "ascending"), ("request_id", "ascending")])
        selected_original = set(json.loads((source_panel.parent / "users.json").read_text())["evaluation"])
        folder = OUTPUT / "random256" / scale / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "requests.parquet"
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["uids"] != selected or previous["requests"]["sha256"] != sha256(path):
                raise RuntimeError("existing random cohort changed; do not overwrite")
            print(json.dumps({"status": "already_complete", "scale": scale, "edge": name}), flush=True)
            continue
        pq.write_table(panel, path, compression="zstd")
        manifest = {"status": "complete", "role": "fixed random diagnostic subset; not a new formal panel",
            "scale": scale, "edge": name, "seed": 17, "uids": selected,
            "selection": "SHA256(namespace/uid), then uid; first256 from source UID pool after calibration exclusions; all requests retained",
            "namespace": NAMESPACE, "source_users": len(all_uids), "eligible_users": len(eligible),
            "excluded_in_source": len(all_uids & excluded), "excluded_total": len(excluded),
            "overlap_original_selected_users": len(set(selected) & selected_original),
            "selection_reads": ["source UID only", "prior calibration/reservation UID only"],
            "original_panel": {"path": str(source_panel.relative_to(ROOT)), "sha256": sha256(source_panel)},
            "days": old["days"], "cutover": old["cutover"], "sources": sources,
            "requests": {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": len(panel)},
            "exclusion_sources": exclusion_records, "metrics_after_selection": metrics(panel),
            "history_loading": "existing load_histories with these UIDs, original dataset and compact mapping, cutover and history cap1024; ordinary rolling replay",
            "pairing": "retained Full, Reuse and labels aligned by request ID, UID and timestamp by paired_population"}
        write_json(manifest_path, manifest)
        print(json.dumps({"status": "complete", "scale": scale, "edge": name, "folder": str(folder),
            "users": len(selected), "requests": len(panel), "excluded_in_source": len(all_uids & excluded),
            "overlap_original": manifest["overlap_original_selected_users"], "metrics": manifest["metrics_after_selection"]}), flush=True)


if __name__ == "__main__":
    main()
