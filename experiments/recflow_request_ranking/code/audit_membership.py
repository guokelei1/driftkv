"""Full input-only diagnostic of inconsistent stage/exposure markers."""
from __future__ import annotations

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa

from .data_common import json_save, rank_membership
from .prepare import projected_reader
from .protocol import day_date


def audit(day, root, output):
    pa.set_cpu_count(4)
    fields = ["user_id", "request_id", "request_timestamp", "video_id", "realshow", "effective_view",
              "rank_index", "rank_pos", "rank_neg", "rerank_pos", "final_neg", "coarse_neg", "prerank_neg"]
    reader = projected_reader(root / f"{day_date(day)}.feather", fields)
    totals = dict(rows=0, exposed=0, positives=0, early_exposed=0, early_positive=0,
                  early_later_flags=0, stage_early_flags=0)
    bad_keys, examples = [], []
    started = time.monotonic()
    for bi in range(reader.num_record_batches):
        batch = reader.get_batch(bi)
        a = {name.strip(): batch.column(i).to_numpy() for i, name in enumerate(batch.schema.names)}
        member = rank_membership(a["rank_index"])
        exposed, positive = a["realshow"] == 1, (a["realshow"] == 1) & (a["effective_view"] == 1)
        inconsistent = exposed & ~member
        totals["rows"] += len(member)
        for key, mask in [("exposed", exposed), ("positives", positive), ("early_exposed", inconsistent),
                          ("early_positive", positive & ~member),
                          ("early_later_flags", ~member & ((a["rank_pos"] + a["rank_neg"] + a["rerank_pos"] + a["final_neg"]) > 0)),
                          ("stage_early_flags", member & ((a["coarse_neg"] + a["prerank_neg"]) > 0))]:
            totals[key] += int(mask.sum())
        if inconsistent.any():
            ids = np.flatnonzero(inconsistent)
            bad_keys.append(np.stack([a["user_id"][ids], a["request_id"][ids], a["video_id"][ids],
                                      a["rank_index"][ids], a["effective_view"][ids]], axis=1))
            for j in ids[:max(0, 12-len(examples))]:
                examples.append({key: int(a[key][j]) for key in fields})
    keys = np.concatenate(bad_keys) if bad_keys else np.empty((0, 5), np.int64)
    np.save(output / f"d{day:02d}_inconsistent_exposures.npy", keys)
    totals.update(day=day, inconsistent_requests=len(np.unique(keys[:, 1])),
                  inconsistent_users=len(np.unique(keys[:, 0])), examples=examples,
                  wall_seconds=time.monotonic()-started)
    json_save(output / f"d{day:02d}.json", totals)
    return {k: v for k, v in totals.items() if k != "examples"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/raw/recflow/all_stage"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(audit, day, args.root, args.output) for day in range(24, 38)]
        reports = []
        for f in as_completed(futures):
            value = f.result()
            reports.append(value)
            print(value, flush=True)
    json_save(args.output / "summary.json", {"days": sorted(reports, key=lambda v: v["day"])})


if __name__ == "__main__":
    main()
