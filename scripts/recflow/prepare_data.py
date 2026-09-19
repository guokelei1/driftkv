#!/usr/bin/env python3
"""Read realshow Feather members in memory, retain compact causal arrays only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from hstu_kvcache.recflow.data import RAW_DTYPE, prepare_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/recflow/downloads/realshow.tar.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/recflow_v1"))
    parser.add_argument("--result", type=Path, default=Path("results/recflow/preparation/summary.json"))
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        raise SystemExit(f"Prepared data already exists: {args.output}; inspect before replacing.")
    started = time.monotonic()
    columns = ["user_id", "request_timestamp", "request_id", "video_id", "category_level_one", " category_level_two", "effective_view", "like", "realshow"]
    mapping = {"uid": "user_id", "ts": "request_timestamp", "request_id": "request_id", "raw_item_id": "video_id", "c1": "category_level_one", "c2": " category_level_two"}
    chunks, source_days = [], []
    with tarfile.open(args.source, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".feather"):
                continue
            day = (date.fromisoformat(Path(member.name).stem) - date(2024, 1, 13)).days + 1
            table = feather.read_table(io.BytesIO(archive.extractfile(member).read()), columns=columns)
            raw = np.empty(len(table), dtype=RAW_DTYPE)
            for target, source in mapping.items():
                values = table[source].to_numpy()
                info = np.iinfo(RAW_DTYPE.fields[target][0])
                if len(values) and (values.min() < info.min or values.max() > info.max):
                    raise ValueError(f"{source} exceeds chosen compact dtype")
                raw[target] = values
            if not np.all(table["realshow"].to_numpy() == 1):
                raise ValueError("Non-exposure rows found in realshow archive")
            effective, like = table["effective_view"].to_numpy(), table["like"].to_numpy()
            if not (np.isin(effective, [0, 1]).all() and np.isin(like, [0, 1]).all()):
                raise ValueError("Unexpected non-binary behavioral feature")
            raw["behavior"] = effective + 2 * like
            raw["day"] = day
            chunks.append(raw)
            source_days.append(day)
            print(json.dumps({"loaded_day": day, "rows": len(raw), "elapsed_seconds": round(time.monotonic()-started, 2)}), flush=True)
    if sorted(source_days) != list(range(1, 38)):
        raise ValueError(f"Expected exactly D1..D37; got {source_days}")
    all_events = np.concatenate(chunks)
    del chunks
    print(json.dumps({"sorting_events": len(all_events)}), flush=True)
    audit = prepare_arrays(all_events, args.output)
    with args.source.open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    audit["source"] = str(args.source)
    audit["source_sha256"] = source_hash
    audit["elapsed_seconds"] = round(time.monotonic() - started, 3)
    audit["prepared_bytes"] = sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "daily"}), flush=True)


if __name__ == "__main__":
    main()
