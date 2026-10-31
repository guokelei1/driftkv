"""Compact, version-independent input contracts for the new RecFlow protocol."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

FEATURES = ("video_id", "author_id", "category_level_one", "category_level_two", "duration")
RAW_DTYPE = np.dtype([
    ("uid", "<i4"), ("ts", "<i8"), ("rid", "<i8"),
    ("features", "<i4", (5,)), ("behavior", "u1"), ("label", "i1"),
])
REQUEST_DTYPE = np.dtype([
    ("uid", "<i4"), ("ts", "<i8"), ("rid", "<i8"),
    ("start", "<i8"), ("stop", "<i8"), ("observed", "<i4"), ("positive", "<i4"),
])


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def stable_hash(values, salt=0):
    x = np.asarray(values, dtype=np.uint64) + np.uint64(salt) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def roles(uids):
    bucket = stable_hash(uids, 20261027) % 100
    return np.select([bucket < 70, bucket < 80, bucket < 90], [0, 1, 2], default=3).astype(np.int8)


def packed_keys(rids, video_ids):
    if len(rids) and (np.min(rids) < 0 or np.max(rids) >= 2**32 or
                      np.min(video_ids) < 0 or np.max(video_ids) >= 2**32):
        raise ValueError("Request/video IDs exceed audited 32-bit key encoding")
    return (np.asarray(rids, np.uint64) << np.uint64(32)) | np.asarray(video_ids, np.uint64)


def rank_membership(rank_index):
    """D24..37: early-stage sampled rows have sentinel 3000 or 8000.

    Zero is a legitimate logged ranking-stage member, including exposed items.
    This rule is inferred from the actual raw encoding and checked against stage
    flags and every exposure; it is not a model input or an oracle score.
    """
    rank_index = np.asarray(rank_index)
    known = ((rank_index >= 0) & (rank_index < 3000)) | np.isin(rank_index, [3000, 8000])
    if not known.all():
        raise ValueError(f"Unrecognized rank_index values: {np.unique(rank_index[~known])[:20]}")
    return rank_index < 3000


def deduplicate_records(raw):
    """Canonical request/video order; duplicated keys must agree on all fields."""
    keys = packed_keys(raw["rid"], raw["features"][:, 0])
    order = np.argsort(keys, kind="stable")
    keys, raw = keys[order], raw[order]
    duplicate = np.r_[False, keys[1:] == keys[:-1]]
    if duplicate.any():
        right = np.flatnonzero(duplicate)
        for name in RAW_DTYPE.names:
            if not np.array_equal(raw[name][right], raw[name][right - 1]):
                raise ValueError(f"Inconsistent duplicated request/video rows: {name}")
        raw, keys = raw[~duplicate], keys[~duplicate]
    start = np.flatnonzero(np.r_[True, raw["rid"][1:] != raw["rid"][:-1]])
    stop = np.r_[start[1:], len(raw)]
    # A request cannot change user/time even within a single contiguous block.
    same = raw["rid"][1:] == raw["rid"][:-1]
    if (same & ((raw["uid"][1:] != raw["uid"][:-1]) | (raw["ts"][1:] != raw["ts"][:-1]))).any():
        raise ValueError("One request ID spans multiple users/timestamps")
    req = np.empty(len(start), REQUEST_DTYPE)
    for name in ["uid", "ts", "rid"]:
        req[name] = raw[name][start]
    req["start"], req["stop"] = start, stop
    req["observed"] = np.add.reduceat((raw["label"] >= 0).astype(np.int32), start)
    req["positive"] = np.add.reduceat((raw["label"] == 1).astype(np.int32), start)
    return raw, keys, req, int(duplicate.sum())


def check_exposure_join(raw, keys, exposed):
    exposure_keys = packed_keys(exposed["rid"], exposed["features"][:, 0])
    locations = np.searchsorted(keys, exposure_keys)
    valid = locations < len(keys)
    valid[valid] &= keys[locations[valid]] == exposure_keys[valid]
    if not valid.all():
        raise ValueError(f"{int((~valid).sum())} realshow keys missing from complete ranking pool")
    actual = raw[locations]
    for name in RAW_DTYPE.names:
        if not np.array_equal(actual[name], exposed[name]):
            unequal = actual[name] != exposed[name]
            raise ValueError(f"all_stage/realshow disagree in {name}: {int(unequal.sum())} entries")
    if int((raw["label"] >= 0).sum()) != len(exposed):
        raise ValueError("all_stage contains exposures absent from realshow")
    return {"matched_exposures": len(exposed), "missing": 0, "extra": 0, "field_mismatches": 0}
