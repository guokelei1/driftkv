"""Causal 1024-event indices, fixed UID roles, and foundation-only vocabulary."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data_common import REQUEST_DTYPE, json_save, roles, sha256, stable_hash
from .protocol import day_bounds_ms

INDEX_DTYPE = np.dtype(REQUEST_DTYPE.descr + [
    ("history_stop", "<i8"), ("history_length", "<i2"), ("day", "u1"),
    ("role", "i1"), ("resident", "?"), ("first_request", "?"), ("valid_request", "?"),
])


def time_keys(uid, ts):
    if len(uid) and (uid.min() < 0 or uid.max() >= 2**22 or ts.min() < 0 or ts.max() >= 2**42):
        raise ValueError("User/timestamp exceeds checked time-key encoding")
    return (np.asarray(uid, np.uint64) << np.uint64(42)) | np.asarray(ts, np.uint64)


def history_index(events, requests, offsets, keys, day, resident_ids):
    result = np.empty(len(requests), INDEX_DTYPE)
    for name in REQUEST_DTYPE.names:
        result[name] = requests[name]
    stop = np.searchsorted(keys, time_keys(requests["uid"], requests["ts"]), side="left")
    begin = offsets[np.minimum(requests["uid"], len(offsets)-1)]
    result["history_stop"] = stop
    result["history_length"] = np.minimum(1024, stop - begin)
    if np.any(stop < begin):
        raise ValueError("Invalid user range in history join")
    nonempty = result["history_length"] > 0
    if np.any(events["ts"][stop[nonempty] - 1] >= requests["ts"][nonempty]):
        raise ValueError("History includes current or future timestamp")
    if np.any(events["uid"][stop[nonempty] - 1] != requests["uid"][nonempty]):
        raise ValueError("History crossed users")
    result["day"], result["role"] = day, roles(requests["uid"])
    result["resident"] = np.isin(requests["uid"], resident_ids)
    result["first_request"] = False
    order = np.lexsort((requests["rid"], requests["ts"], requests["uid"]))
    first = np.r_[True, requests["uid"][order][1:] != requests["uid"][order][:-1]]
    result["first_request"][order[first]] = True
    result["valid_request"] = True
    return result


def build_history(root):
    output = root / "history"
    if (output / "manifest.json").exists():
        return json.loads((output / "manifest.json").read_text())
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    daily = [np.load(root / "exposures" / f"d{day:02d}" / "events.npy", mmap_mode="r") for day in range(1, 38)]
    events = np.concatenate(daily)
    order = np.lexsort((events["features"][:, 0], events["rid"], events["ts"], events["uid"]))
    events = events[order]
    del order, daily
    keys = time_keys(events["uid"], events["ts"])
    user_counts = np.bincount(events["uid"])
    offsets = np.r_[0, np.cumsum(user_counts)]
    initial_counts = np.bincount(events["uid"][events["ts"] < day_bounds_ms(27)[0]], minlength=len(user_counts))
    resident = np.flatnonzero(initial_counts >= 1024).astype(np.int32)
    resident_roles = roles(resident)
    train = resident[resident_roles == 0]
    train = train[np.argsort(stable_hash(train, 271027), kind="stable")]
    np.save(output / "events.npy", events)
    np.save(output / "time_keys.npy", keys)
    np.save(output / "user_offsets.npy", offsets)
    np.savez(output / "population.npz", uids=resident, role=resident_roles,
             initial_history_count=initial_counts[resident], pilot_train_uids=train[:1024],
             cache_fit_uids=train[1024:1408], cache_validation_uids=train[1408:1536])
    for day in range(1, 38):
        source = root / "exposures" / f"d{day:02d}"
        req = np.load(source / "requests.npy", mmap_mode="r")
        result = history_index(events, req, offsets, keys, day, resident)
        np.save(source / "index.npy", result)
    manifest = {"status": "passed", "events": len(events), "resident_history_minimum": 1024,
                "resident_count": len(resident),
                "roles": {name: int((resident_roles == i).sum()) for i, name in enumerate(
                    ["train", "validation", "development", "confirmation"])},
                "causal_cutoff": "strictly_before_entire_user_timestamp_block",
                "features": "original_event_metadata_as_recorded_at_that_request; no_future_backfill",
                "sources": {str(day): sha256(root / "exposures" / f"d{day:02d}" / "manifest.json")
                            for day in range(1, 38)},
                "outputs": {p.name: sha256(p) for p in output.glob("*.np*")},
                "wall_seconds": time.monotonic()-started}
    json_save(output / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    return manifest


def build_stage_indices(root, stage_dir="stage"):
    events = np.load(root / "history/events.npy", mmap_mode="r")
    keys = np.load(root / "history/time_keys.npy", mmap_mode="r")
    offsets = np.load(root / "history/user_offsets.npy", mmap_mode="r")
    population = np.load(root / "history/population.npz")
    report = []
    for day in range(24, 38):
        source = root / stage_dir / f"d{day:02d}"
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest["status"] != "passed":
            raise ValueError("Incomplete daily candidate audit")
        requests = np.load(source / "requests.npy", mmap_mode="r")
        index = history_index(events, requests, offsets, keys, day, population["uids"])
        if (source / "invalid_request_ids.npy").exists():
            index["valid_request"] = ~np.isin(index["rid"], np.load(source / "invalid_request_ids.npy"))
        # Mark FIRST before validity filtering: never replace a rejected first
        # request by a later request with more convenient labels or candidates.
        np.save(source / "index.npy", index)
        report.append({"day": day, "sha256": sha256(source / "index.npy"),
                       "requests": len(index), "valid_requests": int(index["valid_request"].sum()),
                       "resident_first_requests": int((index["resident"] & index["first_request"]).sum()),
                       "valid_resident_first_requests": int((index["resident"] & index["first_request"] & index["valid_request"]).sum())})
    json_save(root / "stage_index_manifest.json", {"status": "passed", "stage_dir": stage_dir, "days": report})


def hash_item(ids, buckets):
    ids = np.asarray(ids, np.int64)
    return ((ids ^ (ids >> 16)) * 2654435761 + 17) % buckets


def build_vocabulary(root, stage_dir="stage", explicit_max=1048576, buckets=262144, output_name="vocabulary"):
    output = root / output_name
    if (output / "manifest.json").exists():
        return
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    events = np.load(root / "history/events.npy", mmap_mode="r")
    selected = (events["ts"] < day_bounds_ms(27)[0]) & (roles(events["uid"]) == 0)
    raw = events["features"][selected, 0]
    counts = np.bincount(raw).astype(np.int64)
    effective = selected & ((events["behavior"] & 1) != 0)
    popularity = np.bincount(events["features"][effective, 0], minlength=len(counts)).astype(np.int32)
    del selected, raw, effective
    sources = [sha256(root / "history/manifest.json")]
    for day in range(24, 27):
        source = root / stage_dir / f"d{day:02d}"
        index = np.load(source / "index.npy", mmap_mode="r")
        features = np.load(source / "features.npy", mmap_mode="r")
        selected = np.repeat((index["role"] == 0) & index["valid_request"], index["stop"]-index["start"])
        addition = np.bincount(features[selected, 0])
        if len(addition) > len(counts):
            counts = np.pad(counts, (0, len(addition)-len(counts)))
        counts[:len(addition)] += addition
        sources.append(sha256(source / "manifest.json"))
        del selected, addition
    observed = np.flatnonzero(counts)
    count = min(explicit_max, len(observed))
    # Exact frequency cutoff, deterministically break ties by raw item ID.
    threshold = np.partition(counts[observed], len(observed)-count)[len(observed)-count]
    above = observed[counts[observed] > threshold]
    tied = observed[counts[observed] == threshold]
    explicit = np.sort(np.r_[above, tied[:count-len(above)]]).astype(np.int32)
    item_map = (count + 1 + hash_item(np.arange(len(counts), dtype=np.int64), buckets)).astype(np.int32)
    item_map[explicit] = np.arange(1, count+1, dtype=np.int32)
    np.save(output / "item_map.npy", item_map)
    np.save(output / "explicit_raw_ids.npy", explicit)
    np.save(output / "foundation_effective_popularity.npy", popularity)
    np.save(output / "foundation_item_counts.npy", counts)
    hashed = observed[item_map[observed] > count]
    occupied = len(np.unique(item_map[hashed]))
    report = {"status": "passed", "explicit_ids": count, "hash_buckets": buckets,
              "item_embedding_rows": count+buckets+1, "raw_id_lookup_length": len(counts),
              "hash_formula": "((raw_id ^ (raw_id >> 16)) * 2654435761 + 17) % buckets",
              "frequency_rule": "all train-role exposure events D1..26 + full candidate occurrences D24..26 on valid train-role requests",
              "foundation_unique_items": len(observed), "foundation_hashed_unique_items": len(hashed),
              "foundation_occupied_hash_buckets": occupied,
              "foundation_excess_hashed_id_collisions": len(hashed)-occupied,
              "foundation_frequency_explicit_coverage": float(counts[explicit].sum()/counts.sum()),
              "metadata_rule": "use each request/event raw metadata available then; no future metadata backfill",
              "future_ids": "same hash fallback; never discard or expand vocabulary between versions",
              "source_manifest_sha256": sources,
              "outputs": {p.name: sha256(p) for p in output.glob("*.npy")},
              "wall_seconds": time.monotonic()-started}
    json_save(output / "manifest.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    parser.add_argument("--stage-dir", default="stage")
    parser.add_argument("--explicit-ids", type=int, default=1048576)
    parser.add_argument("--hash-buckets", type=int, default=262144)
    parser.add_argument("--vocabulary-name", default="vocabulary")
    parser.add_argument("--phase", choices=["history", "stage", "vocabulary", "all"], default="all")
    args = parser.parse_args()
    if args.phase in ["history", "all"]:
        build_history(args.root)
    if args.phase in ["stage", "all"]:
        build_stage_indices(args.root, args.stage_dir)
    if args.phase in ["vocabulary", "all"]:
        build_vocabulary(args.root, args.stage_dir, args.explicit_ids, args.hash_buckets, args.vocabulary_name)


if __name__ == "__main__":
    main()
