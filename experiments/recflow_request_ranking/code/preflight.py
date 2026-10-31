"""CPU-only inventory. Missing stage data never falls back to sampled candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .configuration import load_proposal
from .protocol import day_date, download_files, version_schedule

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/recflow_request_ranking"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command_output(command: list[str]) -> dict:
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {"exit_code": run.returncode, "stdout": run.stdout.strip(), "stderr": run.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def history_inventory(prepared: Path) -> dict:
    requests = np.load(prepared / "requests.npy", mmap_mode="r")
    # This is a feasibility audit of previous raw-exposure-derived arrays, not
    # acceptance of the previous ID vocabulary, roles, or evaluation protocol.
    unique_uids, inverse = np.unique(requests["uid"], return_inverse=True)
    cutoff_rows = []
    lengths = requests["history_stop"] - requests["history_start"]
    for day in [11, 15, 23, 24, 26]:
        before = requests["day"] <= day
        history = np.zeros(len(unique_uids), dtype=np.int64)
        np.maximum.at(history, inverse[before],
                      requests["event_stop"][before] - requests["history_start"][before])
        cutoff_rows.append({"day": day, "date": day_date(day),
                            "users_seen": int((history > 0).sum()),
                            **{f"users_ge_{n}": int((history >= n).sum()) for n in [512, 1024, 2048]}})
    # The proposed resident population is selected using only history by D26.
    population = unique_uids[history >= 1024]
    resident = np.isin(requests["uid"], population)
    daily = []
    for day in range(12, 38):
        on_day = requests["day"] == day
        selected = on_day & resident
        daily.append({
            "day": day, "date": day_date(day), "exposure_requests": int(on_day.sum()),
            "requests_with_history_ge_1024": int((on_day & (lengths >= 1024)).sum()),
            "requests_with_history_ge_2048": int((on_day & (lengths >= 2048)).sum()),
            "d26_resident_requests": int(selected.sum()),
            "d26_resident_active_users": int(len(np.unique(requests["uid"][selected]))),
        })
    return {"status": "feasibility_only_reaudit_raw_history_before_training",
            "source": str(prepared), "requests": len(requests),
            "source_sha256": {name: sha256(prepared / name) for name in ["manifest.json", "requests.npy"]},
            "cutoffs": cutoff_rows, "daily": daily,
            "notes": ["Counts are exposure rows, including previously retained duplicate rows.",
                      "Existing role and item columns are NOT adopted by the new experiment.",
                      "Stage-only requests without exposures are absent from this source."]}


def stage_inventory(stage_root: Path) -> dict:
    missing, present = [], []
    required = {"user_id", "request_id", "request_timestamp", "video_id", "realshow",
                "effective_view", "rank_index", "rerank_index", "rank_pos", "rank_neg",
                "coarse_neg", "prerank_neg", "author_id", "category_level_one", "category_level_two", "duration"}
    for name in download_files():
        path = stage_root / name
        if not path.is_file():
            missing.append(str(path))
            continue
        import pyarrow as pa
        import pyarrow.ipc as ipc
        row = {"path": str(path), "bytes": path.stat().st_size}
        try:
            with pa.memory_map(str(path), "r") as stream:
                reader = ipc.open_file(stream)
                normalized = [column.strip() for column in reader.schema.names]
                row.update(schema=str(reader.schema), record_batches=reader.num_record_batches,
                           missing_columns=sorted(required - set(normalized)),
                           duplicate_normalized_columns=len(normalized) != len(set(normalized)))
        except (OSError, pa.ArrowException) as error:
            row["header_error"] = str(error)
        present.append(row)
    archive_root = stage_root.parent / "downloads"
    archives = sorted(str(p) for p in archive_root.glob("all_stage*")
                      if p.is_file() and p.name.endswith((".tar.gz", ".tgz", ".tar", ".zip", ".tar.xz")))
    invalid = any(row.get("header_error") or row.get("missing_columns") or
                  row.get("duplicate_normalized_columns") for row in present)
    status = "invalid_headers" if invalid else "headers_only_not_candidate_audited"
    if missing:
        status = "archive_available_requires_extraction" if archives else "missing_data"
    return {"root": str(stage_root), "required_files": 14, "missing": missing, "present": present,
            "archives": archives, "status": status,
            "training_ready": False,
            "remaining_checks": ["Verify rank_index sentinel and exact rank-stage membership from actual rows",
                                 "Audit full candidate counts, duplicates and complete exposure/positive joins",
                                 "Verify timestamp units, causal history, raw metadata availability and ID maps",
                                 "Freeze checksums, roles, cohorts and training/validation request manifests"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/processed/recflow_v1")
    parser.add_argument("--stage-root", type=Path, default=ROOT / "data/raw/recflow/all_stage")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "reports/preflight.json")
    args = parser.parse_args()
    proposal = load_proposal()
    disk = shutil.disk_usage(ROOT)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "preparation_only",
        "host": socket.gethostname(), "cpu_count": os.cpu_count(),
        "free_disk_bytes": disk.free,
        "gpus": command_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"]),
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "history": history_inventory(args.prepared), "stages": stage_inventory(args.stage_root),
        "versions": version_schedule(), "confirmation_day": 37,
        "proposal_check": "passed", "model_history_events": proposal["history"]["max_events"],
        "training_started": False, "quality_results": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"report": str(args.output), "status": report["stages"]["status"],
                      "missing_files": len(report["stages"]["missing"]),
                      "cutoffs": report["history"]["cutoffs"],
                      "training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
