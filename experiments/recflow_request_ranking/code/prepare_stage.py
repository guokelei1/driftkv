"""Stage conversion after full raw audit: quarantine inconsistent requests.

All raw exposures are checked against realshow *before* membership filtering.
The candidate array is never supplemented with missing exposures. Ambiguous
requests remain in the all-request ledger and are marked invalid for learning
and metrics; a later request must not replace the day's actual first request.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa

from .data_common import (
    FEATURES,
    RAW_DTYPE,
    check_exposure_join,
    deduplicate_records,
    json_save,
    rank_membership,
    sha256,
)
from .prepare import projected_reader
from .protocol import day_bounds_ms, day_date


def convert(day, source_root, root, output_folder, threads):
    pa.set_cpu_count(threads)
    source = source_root / f"{day_date(day)}.feather"
    dest = root / output_folder / f"d{day:02d}"
    if (dest / "manifest.json").exists():
        manifest = json.loads((dest / "manifest.json").read_text())
        if sha256(source) != manifest["source_sha256"]:
            raise ValueError("Raw source changed")
        for artifact in manifest["outputs"]:
            if sha256(dest / artifact["name"]) != artifact["sha256"]:
                raise ValueError("Completed output changed")
        return {"day": day, "status": "verified_existing"}
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    columns = ["user_id", "request_id", "request_timestamp", *FEATURES, "realshow", "effective_view", "like",
               "rank_index", "coarse_neg", "prerank_neg", "rank_pos", "rank_neg", "rerank_pos", "final_neg"]
    reader = projected_reader(source, columns)
    bad, exposure_parts, raw_request_parts = [], [], []
    raw_rows, kept_rows = 0, 0
    low, high = day_bounds_ms(day)
    temp = dest / "raw.partial.bin"
    try:
        with temp.open("wb") as handle:
            for bi in range(reader.num_record_batches):
                batch = reader.get_batch(bi)
                if any(col.null_count for col in batch.columns):
                    raise ValueError("Raw nulls need explicit handling")
                a = {name.strip(): batch.column(i).to_numpy() for i, name in enumerate(batch.schema.names)}
                member = rank_membership(a["rank_index"])
                exposed = a["realshow"] == 1
                if a["request_timestamp"].min() < low or a["request_timestamp"].max() >= high:
                    raise ValueError("Unexpected raw day/timezone")
                for flag in ["realshow", "effective_view", "like", "coarse_neg", "prerank_neg",
                             "rank_pos", "rank_neg", "rerank_pos", "final_neg"]:
                    if not np.isin(a[flag], [0, 1]).all():
                        raise ValueError(f"Nonbinary {flag}")
                for flag, sentinel in [("coarse_neg", 3000), ("prerank_neg", 8000)]:
                    if np.any((a[flag] == 1) & (a["rank_index"] != sentinel)):
                        raise ValueError("Early-stage marker disagrees with sentinel")
                for flag in ["rank_pos", "rank_neg", "rerank_pos", "final_neg"]:
                    if np.any((a[flag] == 1) & ~member):
                        raise ValueError("Later-stage flag contradicts sentinel")
                raw = np.empty(len(member), RAW_DTYPE)
                for field, source_field in [("uid", "user_id"), ("rid", "request_id"), ("ts", "request_timestamp")]:
                    value = a[source_field]
                    if value.min() < 0 or value.max() > np.iinfo(RAW_DTYPE.fields[field][0]).max:
                        raise ValueError(f"Out-of-range {source_field}")
                    raw[field] = value
                for i, field in enumerate(FEATURES):
                    if a[field].min() < 0 or a[field].max() > np.iinfo(np.int32).max:
                        raise ValueError(f"Out-of-range feature {field}")
                    raw["features"][:, i] = a[field]
                raw["behavior"] = a["effective_view"] + 2*a["like"]
                raw["label"] = np.where(exposed, a["effective_view"], -1)
                exposure_parts.append(raw[exposed])
                bad.append(a["request_id"][exposed & ~member])
                raw_request_parts.append(np.unique(a["request_id"]))
                raw[member].tofile(handle)
                raw_rows += len(member)
                kept_rows += int(member.sum())
        invalid = np.unique(np.concatenate(bad))
        exposure_raw, exposure_keys, _, duplicate_exposures = deduplicate_records(np.concatenate(exposure_parts))
        expected = np.load(root / "exposures" / f"d{day:02d}" / "events.npy", mmap_mode="r")
        exposure_join = check_exposure_join(exposure_raw, exposure_keys, expected)
        del exposure_parts, exposure_raw, exposure_keys
        raw, keys, requests, duplicates = deduplicate_records(np.memmap(temp, mode="r", dtype=RAW_DTYPE))
        raw_request_ids = np.unique(np.concatenate(raw_request_parts))
        if not np.array_equal(raw_request_ids, requests["rid"]):
            raise ValueError("A raw request has no retained ranking candidates; preserve first-request ledger")
        valid_expected = expected[~np.isin(expected["rid"], invalid)]
        locations = np.searchsorted(keys, (valid_expected["rid"].astype(np.uint64) << np.uint64(32)) |
                                    valid_expected["features"][:, 0].astype(np.uint64))
        # Full all-exposure audit already passed; this independent join proves
        # that valid requests have complete targets in the retained stage pool.
        if np.any(locations >= len(raw)) or not np.array_equal(raw[locations], valid_expected):
            raise ValueError("A valid-request exposure is missing from candidate pool")
        np.save(dest / "features.npy", raw["features"])
        np.save(dest / "labels.npy", raw["label"])
        np.save(dest / "requests.npy", requests)
        np.save(dest / "invalid_request_ids.npy", invalid)
        report = {"status": "passed", "day": day, "raw_rows": raw_rows, "rows": len(raw),
                  "requests": len(requests), "invalid_requests": len(invalid),
                  "valid_requests": len(requests)-len(invalid),
                  "invalid_reason": "any real exposure outside inferred ranking stage; independent of effective_view",
                  "invalid_handling": "retain full ledger and original candidate rows; exclude whole request from train/metrics; do not substitute first requests",
                  "candidate_rule": "0 <= rank_index < 3000; sentinel 3000/8000 excluded",
                  "all_raw_exposure_join": exposure_join,
                  "valid_request_exposures_matched": len(valid_expected),
                  "candidate_count_quantiles": np.quantile(requests["stop"]-requests["start"], [0,.5,.95,.99,1]).tolist(),
                  "duplicate_exposure_rows": duplicate_exposures, "duplicate_candidate_rows": duplicates,
                  "source_sha256": sha256(source),
                  "exposure_manifest_sha256": sha256(root / "exposures" / f"d{day:02d}" / "manifest.json"),
                  "outputs": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(dest.glob("*.npy"))],
                  "preparer_sha256": sha256(__file__), "wall_seconds": time.monotonic()-started,
                  "arrow_threads": threads}
        json_save(dest / "manifest.json", report)
        temp.unlink()
        return {k: report[k] for k in ["day", "rows", "requests", "invalid_requests", "wall_seconds", "status"]}
    except BaseException as error:
        json_save(dest / "failure.json", {"error": repr(error), "wall_seconds": time.monotonic()-started})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/raw/recflow/all_stage"))
    p.add_argument("--root", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    p.add_argument("--output-folder", default="stage_v2")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    reports = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(convert, day, args.source, args.root, args.output_folder, 4) for day in range(24, 38)]
        for future in as_completed(futures):
            result = future.result()
            reports.append(result)
            print(json.dumps(result), flush=True)
    json_save(args.root / f"{args.output_folder}_conversion.json", {"status": "passed", "days": sorted(reports, key=lambda x:x["day"])})


if __name__ == "__main__":
    main()
