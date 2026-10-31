"""Input-only item coverage of the actual proposed training requests/history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .batches import RequestBatches
from .data_common import json_save, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-limit", type=int)
    parser.add_argument("--alignment-limit", type=int)
    args = parser.parse_args()
    data = args.data
    vocabulary = json.loads((data/"vocabulary_max/manifest.json").read_text())
    mapping = np.load(data/"vocabulary_max/item_map.npy", mmap_mode="r")
    seen_raw = np.zeros(len(mapping), np.bool_)
    seen_table = np.zeros(vocabulary["explicit_ids"]+vocabulary["hash_buckets"]+1, np.bool_)
    events = np.load(data/"history/events.npy", mmap_mode="r")
    phases, history_ranges = {}, []

    def record(ids):
        if not len(ids):
            return {"occurrences": 0, "explicit_occurrences": 0}
        if ids.max() >= len(mapping) or ids.min() < 0:
            raise ValueError("Foundation raw IDs exceed the frozen input mapping")
        mapped = mapping[ids]
        seen_raw[ids] = True
        seen_table[mapped] = True
        return {"occurrences": len(ids), "explicit_occurrences": int((mapped <= vocabulary["explicit_ids"]).sum())}

    for name, days in [("warmup", range(12, 24)), ("alignment", range(24, 27))]:
        dataset = RequestBatches(data, days, role=0,
                                 limit=args.warmup_limit if name == "warmup" else args.alignment_limit)
        history_ranges.append(np.stack([dataset.records["history_stop"]-dataset.records["history_length"],
                                        dataset.records["history_stop"]]))
        totals = {"requests": len(dataset), "users": len(np.unique(dataset.records["uid"])),
                  "candidate_occurrences": 0, "explicit_candidate_occurrences": 0}
        for day in days:
            directory = data/("exposures" if day < 24 else "stage_v2")/f"d{day:02d}"
            ids = np.load(directory/"events.npy", mmap_mode="r")["features"][:, 0] if day < 24 else np.load(
                directory/"features.npy", mmap_mode="r")[:, 0]
            include = np.zeros(len(ids), np.bool_)
            for request in dataset.records[dataset.records["day"] == day]:
                include[request["start"]:request["stop"]] = True
            for start in range(0, len(ids), 2_000_000):
                result = record(ids[start:start+2_000_000][include[start:start+2_000_000]])
                totals["candidate_occurrences"] += result["occurrences"]
                totals["explicit_candidate_occurrences"] += result["explicit_occurrences"]
            del include
        phases[name] = totals
    ranges = np.concatenate(history_ranges, axis=1)
    difference = np.zeros(len(events)+1, np.int32)
    np.add.at(difference, ranges[0], 1)
    np.add.at(difference, ranges[1], -1)
    covered = np.cumsum(difference[:-1], dtype=np.int32) > 0
    history = {"distinct_causal_event_positions": int(covered.sum()), "explicit_occurrences": 0}
    for start in range(0, len(events), 2_000_000):
        result = record(events["features"][start:start+2_000_000, 0][covered[start:start+2_000_000]])
        history["explicit_occurrences"] += result["explicit_occurrences"]
    json_save(args.output, {"status": "passed", "quality_labels_used_for_selection": False,
              "vocabulary_sha256": sha256(data/"vocabulary_max/manifest.json"), "phases": phases,
              "history_union": history, "distinct_raw_training_items": int(seen_raw.sum()),
              "visited_explicit_embedding_rows": int(seen_table[1:vocabulary["explicit_ids"]+1].sum()),
              "total_explicit_embedding_rows": vocabulary["explicit_ids"],
              "visited_hash_embedding_rows": int(seen_table[vocabulary["explicit_ids"]+1:].sum()),
              "interpretation": "Exposure targets, complete candidate targets and exact union of causal history ranges; input coverage is not independent label count or trained quality"})


if __name__ == "__main__":
    main()
