"""Foundation-only UID selection, audited mmap request lists, and new vocabulary."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .batches import selection_key
from .data_common import json_save, sha256, stable_hash
from .protocol import day_bounds_ms


def read(path):
    return json.loads(Path(path).read_text())


def ordered(uids):
    return uids[np.argsort(stable_hash(uids, 20261028), kind="stable")]


def select_population(supervised):
    pool = np.unique(np.concatenate([supervised[d] for d in range(1, 28)]))
    ranking = np.unique(np.concatenate([supervised[d] for d in range(24, 28)]))
    early_only = np.setdiff1d(pool, ranking)
    train = np.sort(np.r_[ordered(ranking)[:20000], ordered(early_only)[:10000]]).astype(np.int32)
    remainder = ordered(np.setdiff1d(ranking, train))
    groups = dict(train=train, validation=np.sort(remainder[:2000]),
                  cache_fit=np.sort(remainder[2000:2384]), cache_validation=np.sort(remainder[2384:2512]))
    expected = dict(train=30000, validation=2000, cache_fit=384, cache_validation=128)
    for key, value in groups.items():
        if len(value) != expected[key] or len(np.unique(value)) != len(value):
            raise ValueError(f"Insufficient/duplicate foundation-only users: {key}")
    joined = np.concatenate(list(groups.values()))
    if len(np.unique(joined)) != len(joined):
        raise ValueError("User roles overlap")
    return groups, dict(supervised_users=len(pool), ranking_supervised_users=len(ranking), early_only_users=len(early_only))


def causal_audit(records, events, offsets):
    for start in range(0, len(records), 100000):
        req = records[start:start+100000]
        stop = req["history_stop"]
        begin = offsets[req["uid"]]
        length = np.minimum(stop-begin, 1024)
        if np.any(stop < begin) or not np.array_equal(length, req["history_length"]):
            raise ValueError("History length/bounds inconsistent")
        nonempty = length > 0
        last = events[stop[nonempty]-1]
        if np.any(last["uid"] != req["uid"][nonempty]) or np.any(last["ts"] >= req["ts"][nonempty]):
            raise ValueError("History crosses users or includes current/future timestamp")
        first = events[stop[nonempty]-length[nonempty]]
        if np.any(first["uid"] != req["uid"][nonempty]):
            raise ValueError("History prefix crosses user boundary")
        interior = stop < len(events)
        nxt = events[stop[interior]]
        if np.any((nxt["uid"] == req["uid"][interior]) & (nxt["ts"] < req["ts"][interior])):
            raise ValueError("History omits an available past timestamp")


def vocabulary(source, output, train_uids):
    started = time.monotonic()
    output.mkdir()
    events = np.load(source / "history/events.npy", mmap_mode="r")
    selected = np.isin(events["uid"], train_uids) & (events["ts"] < day_bounds_ms(28)[0])
    counts = np.bincount(events["features"][selected, 0]).astype(np.int64)
    positive = selected & ((events["behavior"] & 1) != 0)
    popularity = np.bincount(events["features"][positive, 0], minlength=len(counts)).astype(np.int32)
    del selected, positive
    for day in range(24, 28):
        directory = source / f"stage_v2/d{day:02d}"
        index = np.load(directory / "index.npy", mmap_mode="r")
        features = np.load(directory / "features.npy", mmap_mode="r")
        chosen = np.isin(index["uid"], train_uids) & index["valid_request"]
        mask = np.repeat(chosen, index["stop"]-index["start"])
        addition = np.bincount(features[mask, 0])
        if len(addition) > len(counts):
            counts = np.pad(counts, (0, len(addition)-len(counts)))
        counts[:len(addition)] += addition
        del addition, mask, features
    observed = np.flatnonzero(counts)
    size, buckets = 3500000, 65536
    if len(observed) < size:
        raise ValueError("Insufficient IDs for unchanged Max embedding size")
    threshold = np.partition(counts[observed], len(observed)-size)[len(observed)-size]
    above = observed[counts[observed] > threshold]
    tied = observed[counts[observed] == threshold]
    explicit = np.sort(np.r_[above, tied[:size-len(above)]]).astype(np.int32)
    raw = np.arange(len(counts), dtype=np.int64)
    item_map = (size+1+((raw ^ (raw >> 16))*2654435761+17) % buckets).astype(np.int32)
    item_map[explicit] = np.arange(1, size+1, dtype=np.int32)
    for name, arr in dict(item_map=item_map, explicit_raw_ids=explicit,
                          foundation_effective_popularity=popularity, foundation_item_counts=counts).items():
        np.save(output / f"{name}.npy", arr)
    json_save(output / "manifest.json", dict(status="passed", explicit_ids=size, hash_buckets=buckets,
        item_embedding_rows=size+buckets+1, raw_id_lookup_length=len(counts),
        frequency_rule="selected 30000 train UIDs; exposure D1-27 plus all valid complete candidates D24-27",
        cutoff_exclusive_ms=day_bounds_ms(28)[0], foundation_unique_items=len(observed),
        foundation_frequency_explicit_coverage=float(counts[explicit].sum()/counts.sum()),
        future_ids="fixed hash fallback; no vocabulary growth", outputs={p.name:sha256(p) for p in output.glob("*.npy")},
        wall_seconds=time.monotonic()-started))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    p.add_argument("--output", type=Path, default=Path("experiments/recflow_full_population/data"))
    args = p.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("Preserve existing data preparation; inspect before rerunning")
    output.mkdir()
    started = time.monotonic()
    for name in ["exposures", "stage_v2", "stage_index_manifest.json"]:
        (output / name).symlink_to(source / name, target_is_directory=(source/name).is_dir())
    (output / "history").mkdir()
    for name in ["events.npy", "time_keys.npy", "user_offsets.npy", "manifest.json"]:
        (output / "history" / name).symlink_to(source / "history" / name)
    # Verify complete input files against the previously completed ingestion audit.
    inputs = {}
    for directory in [source/f"exposures/d{d:02d}" for d in range(1,38)] + [source/f"stage_v2/d{d}" for d in range(24,38)]:
        manifest = read(directory / "manifest.json")
        if manifest["status"] != "passed":
            raise ValueError("Input conversion not complete")
        inputs[str(directory / "manifest.json")] = sha256(directory / "manifest.json")
        for entry in manifest["outputs"]:
            inputs[str(directory / entry["name"])] = entry["sha256"]
    original_history = read(source / "history/manifest.json")
    for name, digest in original_history["outputs"].items():
        inputs[str(source / "history" / name)] = digest
    print(json.dumps(dict(phase="input_manifests_loaded", files=len(inputs))), flush=True)
    arrays = {d:np.load(source/(f"exposures/d{d:02d}/index.npy" if d<24 else f"stage_v2/d{d:02d}/index.npy"),mmap_mode="r") for d in range(1,38)}
    supervised = {d:np.unique(a["uid"][a["valid_request"] & (a["observed"]>0)]) for d,a in arrays.items() if d<=27}
    groups, counts = select_population(supervised)
    digest = hashlib.sha256(groups["train"].astype("<i4").tobytes()).hexdigest()
    np.savez(output / "population.npz", **groups)
    json_save(output / "population_manifest.json", dict(status="passed", seed=20261028,
        source_cutoff_day=27, counts=counts, role_counts={k:len(v) for k,v in groups.items()},
        population_sha256=sha256(output/"population.npz"), train_uid_sha256=digest,
        minimum_history=0, future_activity_or_outcome_selection=False))
    events = np.load(output/"history/events.npy",mmap_mode="r")
    offsets = np.load(output/"history/user_offsets.npy",mmap_mode="r")
    (output/"selections").mkdir()
    schedules = [(0,list(range(1,24))), (0,list(range(24,28))), (1,list(range(24,28)))]
    schedules += [(0,[d]) for d in range(28,37)] + [(2,[d]) for d in range(28,38)]
    datasets = {}
    for role,days in schedules:
        chosen_uids=groups["validation" if role==1 else "train"]
        parts=[]
        for day in days:
            a=arrays[day]
            selection=np.isin(a["uid"],chosen_uids)&a["valid_request"]
            if role==0:
                selection &= a["observed"]>0
            parts.append(a[selection])
        records=np.concatenate(parts)
        if role==1:
            subset=np.argsort(stable_hash(records["rid"],9427),kind="stable")[:16384]
            records=records[np.sort(subset)]
        causal_audit(records,events,offsets)
        for day in days:
            day_records=records[records["day"]==day]
            lo,hi=day_bounds_ms(day)
            if np.any((day_records["ts"]<lo)|(day_records["ts"]>=hi)):
                raise ValueError("Daily target timestamps disagree")
        name=f"selections/role{role}_d{days[0]}_d{days[-1]}.npy"
        np.save(output/name,records)
        datasets[selection_key(role,days)] = dict(file=name,sha256=sha256(output/name),requests=len(records),
            users=len(np.unique(records["uid"])),positive_requests=int((records["positive"]>0).sum()),
            empty_history_requests=int((records["history_length"]==0).sum()),
            full_history_requests=int((records["history_length"]==1024).sum()),history_mean=float(records["history_length"].mean()))
    if datasets[selection_key(0,range(1,24))]["requests"]!=4123554 or datasets[selection_key(0,range(24,28))]["requests"]!=647394:
        raise ValueError("Full-data training counts differ from approved plan")
    json_save(output/"selection_manifest.json",dict(status="passed",datasets=datasets,
        population_manifest_sha256=sha256(output/"population_manifest.json"),
        original_indices={str(source/(f"exposures/d{d:02d}/index.npy" if d<24 else f"stage_v2/d{d:02d}/index.npy")):sha256(source/(f"exposures/d{d:02d}/index.npy" if d<24 else f"stage_v2/d{d:02d}/index.npy")) for d in range(1,38)},
        history_audit="all selected request boundaries, timestamps, actual lengths and UID membership verified"))
    print(json.dumps(dict(phase="population_and_requests_verified",datasets=len(datasets))),flush=True)
    vocabulary(source,output/"vocabulary_max",groups["train"])
    json_save(output/"preparation_audit.json",dict(status="passed",inputs_sha256=inputs,
        preparation_sha256=sha256(__file__),wall_seconds=time.monotonic()-started,
        training_requests_complete=True,selection_count=len(datasets),minimum_history=0,
        original_data_mutated=False))
    print(json.dumps(dict(status="passed",wall_seconds=time.monotonic()-started)),flush=True)


if __name__=="__main__":
    main()
