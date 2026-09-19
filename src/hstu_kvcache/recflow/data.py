"""Small, causal RecFlow development data interface.

The prepared arrays retain every realshow event. A request's prefix ends before
its entire user/timestamp block, and its targets retain unknown raw video IDs.
Admission/final users are stored for a later sealed protocol; this interface's
sample selectors expose development users only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROLE_NAMES = {0: "development", 1: "admission", 2: "final", -1: "not_initial_population"}
RAW_DTYPE = np.dtype([
    ("uid", "i4"), ("ts", "i8"), ("request_id", "i8"),
    ("raw_item_id", "i4"), ("c1", "i4"), ("c2", "i4"),
    ("behavior", "u1"), ("day", "u1"),
])
EVENT_DTYPE = np.dtype([
    ("uid", "i4"), ("ts", "i8"), ("request_id", "i8"),
    ("raw_item_id", "i4"), ("item", "i4"), ("behavior", "u1"), ("day", "u1"),
])
REQUEST_DTYPE = np.dtype([
    ("uid", "i4"), ("ts", "i8"), ("request_id", "i8"), ("day", "u1"),
    ("event_start", "i8"), ("event_stop", "i8"),
    ("history_start", "i8"), ("history_stop", "i8"),
    ("positive_start", "i8"), ("positive_stop", "i8"), ("role", "i1"),
])
POSITIVE_DTYPE = np.dtype([("raw_item_id", "i4"), ("item", "i4")])


def stable_hash(values: np.ndarray, salt: int = 0) -> np.ndarray:
    """SplitMix64; stable across processes and independent of labels."""
    x = np.asarray(values, dtype=np.uint64) + np.uint64(salt) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def user_roles(uids: np.ndarray) -> np.ndarray:
    buckets = stable_hash(uids, salt=20260918) % 100
    return np.where(buckets < 80, 0, np.where(buckets < 90, 1, 2)).astype(np.int8)


def map_items(raw_ids: np.ndarray, catalog_ids: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(catalog_ids, raw_ids)
    valid = positions < len(catalog_ids)
    valid[valid] &= catalog_ids[positions[valid]] == raw_ids[valid]
    return np.where(valid, positions + 1, len(catalog_ids) + 1).astype(np.int32)


def prepare_arrays(raw: np.ndarray, output: Path, initial_day: int = 18) -> dict:
    """Sort compact input events and write the single experiment's mmap arrays."""
    output.mkdir(parents=True, exist_ok=True)
    initial = raw[raw["day"] <= initial_day]
    initial_uids, initial_counts = np.unique(initial["uid"], return_counts=True)
    roles = user_roles(initial_uids)
    # First observed time fixes metadata, irrespective of archive row ordering.
    first_order = np.lexsort((initial["c2"], initial["c1"], initial["ts"], initial["raw_item_id"]))
    ordered = initial[first_order]
    first = np.r_[True, ordered["raw_item_id"][1:] != ordered["raw_item_id"][:-1]]
    catalog_ids = ordered["raw_item_id"][first]
    catalog_c1, catalog_c2 = ordered["c1"][first], ordered["c2"][first]
    initial_item = map_items(initial["raw_item_id"], catalog_ids) - 1
    dev_initial = user_roles(initial["uid"]) == 0
    catalog = {
        "raw_item_ids": catalog_ids, "c1": catalog_c1, "c2": catalog_c2,
        "exposure_count": np.bincount(initial_item, minlength=len(catalog_ids)),
        "effective_count": np.bincount(initial_item[(initial["behavior"] & 1) != 0], minlength=len(catalog_ids)),
        "development_exposure_count": np.bincount(initial_item[dev_initial], minlength=len(catalog_ids)),
        "development_effective_count": np.bincount(initial_item[dev_initial & ((initial["behavior"] & 1) != 0)], minlength=len(catalog_ids)),
    }
    metadata_changes = (initial["c1"] != catalog_c1[initial_item]) | (initial["c2"] != catalog_c2[initial_item])
    catalog_change_rows = int(metadata_changes.sum())
    np.savez(output / "catalog.npz", **catalog)
    np.savez(output / "population.npz", uids=initial_uids, initial_history_count=initial_counts, role=roles)
    del initial, first_order, ordered, first, initial_item, dev_initial, metadata_changes

    order = np.lexsort((raw["raw_item_id"], raw["request_id"], raw["ts"], raw["uid"]))
    raw = raw[order]
    events = np.empty(len(raw), dtype=EVENT_DTYPE)
    for name in ("uid", "ts", "request_id", "raw_item_id", "behavior", "day"):
        events[name] = raw[name]
    events["item"] = map_items(raw["raw_item_id"], catalog_ids)
    del raw, order

    new_uid = np.r_[True, events["uid"][1:] != events["uid"][:-1]]
    new_time = new_uid | np.r_[True, events["ts"][1:] != events["ts"][:-1]]
    new_request = new_time | np.r_[True, events["request_id"][1:] != events["request_id"][:-1]]
    request_starts = np.flatnonzero(new_request)
    request_stops = np.r_[request_starts[1:], len(events)]
    requests = np.empty(len(request_starts), dtype=REQUEST_DTYPE)
    for name in ("uid", "ts", "request_id", "day"):
        requests[name] = events[name][request_starts]
    requests["event_start"], requests["event_stop"] = request_starts, request_stops
    uid_starts, time_starts = np.flatnonzero(new_uid), np.flatnonzero(new_time)
    requests["history_start"] = uid_starts[np.searchsorted(uid_starts, request_starts, side="right") - 1]
    requests["history_stop"] = time_starts[np.searchsorted(time_starts, request_starts, side="right") - 1]
    requests["role"] = user_roles(requests["uid"])
    requests["role"][~np.isin(requests["uid"], initial_uids)] = -1

    # A request ID assigned to several timestamps for one user would make an
    # arbitrary request split causal evidence. Stop and inspect if encountered.
    req_order = np.lexsort((requests["ts"], requests["uid"], requests["request_id"]))
    req_sorted = requests[req_order]
    repeated_group = ((req_sorted["request_id"][1:] == req_sorted["request_id"][:-1])
                      & (req_sorted["uid"][1:] == req_sorted["uid"][:-1]))
    if repeated_group.any():
        raise ValueError("A (user, request_id) spans multiple timestamps; inspect grouping before fitting.")
    reused_request_ids = int(np.sum(req_sorted["request_id"][1:] == req_sorted["request_id"][:-1]))
    del req_order, req_sorted, repeated_group

    positive_events = np.flatnonzero((events["behavior"] & 1) != 0)
    positive_requests = np.searchsorted(request_starts, positive_events, side="right") - 1
    positive_raw = events["raw_item_id"][positive_events]
    unique_positive = np.r_[True, (positive_requests[1:] != positive_requests[:-1]) | (positive_raw[1:] != positive_raw[:-1])]
    positive_events, positive_requests = positive_events[unique_positive], positive_requests[unique_positive]
    positives = np.empty(len(positive_events), dtype=POSITIVE_DTYPE)
    positives["raw_item_id"], positives["item"] = events["raw_item_id"][positive_events], events["item"][positive_events]
    pos_counts = np.bincount(positive_requests, minlength=len(requests))
    requests["positive_stop"] = np.cumsum(pos_counts)
    requests["positive_start"] = requests["positive_stop"] - pos_counts
    duplicate_events = int(np.sum(~new_request[1:] & (events["raw_item_id"][1:] == events["raw_item_id"][:-1])))
    same_time_requests = ~np.r_[True, (requests["uid"][1:] != requests["uid"][:-1]) | (requests["ts"][1:] != requests["ts"][:-1])]
    event_known = events["item"] <= len(catalog_ids)
    oov_id = len(catalog_ids) + 1
    daily = []
    for day in np.unique(events["day"]):
        ev = events["day"] == day
        req = requests["day"] == day
        pos = requests["day"][positive_requests] == day
        effective = ev & ((events["behavior"] & 1) != 0)
        daily.append({
            "day": int(day), "events": int(ev.sum()), "requests": int(req.sum()),
            "effective_events": int(effective.sum()), "unique_request_positives": int(pos.sum()),
            "positive_requests": int(np.sum(req & (pos_counts > 0))),
            "catalog_exposure_coverage": float(event_known[ev].mean()),
            "catalog_effective_coverage": float(event_known[effective].mean()) if effective.any() else None,
            "catalog_unique_positive_coverage": float((positives["item"][pos] != oov_id).mean()) if pos.any() else None,
        })
    audit = {
        "schema_version": 1, "scope": "Label-free preparation plus coverage audit; no model or final-user quality evaluation.",
        "initial_day": initial_day, "first_date": "2024-01-13", "initial_users": len(initial_uids),
        "all_users": int(new_uid.sum()), "events": len(events), "requests": len(requests),
        "catalog_items": len(catalog_ids), "positive_items_per_request_total": len(positives),
        "initial_users_ge_1024": int(np.sum(initial_counts >= 1024)),
        "initial_users_ge_2048": int(np.sum(initial_counts >= 2048)),
        "roles": {name: {"users": int(np.sum(roles == role)), "users_ge_1024": int(np.sum((roles == role) & (initial_counts >= 1024)))} for role, name in ROLE_NAMES.items() if role >= 0},
        "request_id_reused_across_users": reused_request_ids,
        "same_user_timestamp_extra_requests": int(same_time_requests.sum()),
        "duplicate_request_video_rows_retained_in_history": duplicate_events,
        "duplicate_positive_rows_removed_from_targets": int(len(unique_positive) - unique_positive.sum()),
        "initial_rows_different_from_frozen_item_categories": catalog_change_rows,
        "request_exposure_count_quantiles": dict(zip(["min", "p50", "p95", "max"], [float(x) for x in np.quantile(request_stops-request_starts, [0,.5,.95,1])], strict=True)),
        "history_rule": "All real exposures strictly earlier than request timestamp; entire same-timestamp block excluded, including other requests.",
        "positive_rule": "All distinct effective_view=1 video IDs per complete request; unknown targets retained and scored as misses.",
        "role_rule": "SplitMix64(uid+20260918) mod 100: <80 development, <90 admission, else final; population fixed by D18.",
        "category_rule": "Earliest initial-window timestamp per video; ties resolved lexicographically by (c1,c2); fixed across releases.",
        "item_ids": {"padding": 0, "known": f"1..{len(catalog_ids)}", "oov": oov_id},
        "daily": daily,
    }
    np.save(output / "events.npy", events)
    np.save(output / "requests.npy", requests)
    np.save(output / "positives.npy", positives)
    (output / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


class PreparedRecFlow:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.catalog = dict(np.load(self.root / "catalog.npz"))
        self.population = dict(np.load(self.root / "population.npz"))
        self.events = np.load(self.root / "events.npy", mmap_mode="r")
        self.requests = np.load(self.root / "requests.npy", mmap_mode="r")
        self.positives = np.load(self.root / "positives.npy", mmap_mode="r")
        self.catalog_size = len(self.catalog["raw_item_ids"])
        self.oov_id = self.catalog_size + 1

    def cohort(self, min_history: int = 1024, limit: int | None = 512) -> np.ndarray:
        eligible = (self.population["role"] == 0) & (self.population["initial_history_count"] >= min_history)
        uids = self.population["uids"][eligible]
        order = np.argsort(stable_hash(uids, salt=1818), kind="stable")
        return uids[order[:limit]] if limit is not None else uids[order]

    def request_indices(self, day_start: int, day_end: int, uids: np.ndarray | None = None,
                        limit: int | None = None, positive_only: bool = True) -> np.ndarray:
        req = self.requests
        mask = (req["role"] == 0) & (req["day"] >= day_start) & (req["day"] <= day_end)
        if uids is not None:
            mask &= np.isin(req["uid"], uids)
        if positive_only:
            mask &= req["positive_stop"] > req["positive_start"]
        indices = np.flatnonzero(mask)
        if limit is not None and len(indices) > limit:
            hashes = stable_hash(req["request_id"][indices], salt=3718) ^ stable_hash(req["uid"][indices])
            indices = indices[np.argsort(hashes, kind="stable")[:limit]]
        return indices

    def history(self, request_index: int, max_length: int = 1024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        request = self.requests[request_index]
        stop = int(request["history_stop"])
        user_start = int(request["history_start"])
        start = max(user_start, stop - max_length)
        events = self.events[start:stop]
        deltas = np.zeros(len(events), dtype=np.float32)
        # Preserve each event's original delta when the rolling window evicts
        # its predecessor. Re-zeroing the head would change an existing token's
        # embedding and make its previously written K/V inconsistent.
        if len(events) and start > user_start:
            deltas[0] = (int(events["ts"][0]) - int(self.events["ts"][start - 1])) / 1000
        if len(events) > 1:
            deltas[1:] = np.diff(events["ts"]).astype(np.float32) / 1000
        return events["item"], events["behavior"], deltas

    def targets(self, request_index: int) -> tuple[np.ndarray, np.ndarray]:
        request = self.requests[request_index]
        positives = self.positives[int(request["positive_start"]):int(request["positive_stop"])]
        return positives["raw_item_id"], positives["item"]

    def item_paths(self) -> np.ndarray:
        paths = np.full((self.catalog_size + 2, 2), -1, dtype=np.int64)
        paths[1:-1, 0], paths[1:-1, 1] = self.catalog["c1"], self.catalog["c2"]
        return paths
