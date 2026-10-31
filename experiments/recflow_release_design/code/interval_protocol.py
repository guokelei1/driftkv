"""Actual release ancestry and complete-user samples for interval experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.recflow_full_population.code.batches import RequestBatches


def ancestry(checkpoint, cache_source):
    """Return the recorded path from a genuine ancestor to the target release.

    Parent links describe optimizer lineage, including unpublished intermediate
    checkpoints. Model loading and numerical verification belong to the scorer.
    """
    target, source = Path(checkpoint).resolve(), Path(cache_source).resolve()
    if target == source:
        raise ValueError("Cache source must be an earlier ancestor, not the target")
    reverse, seen, architecture = [], set(), None
    current = target
    while True:
        if current in seen:
            raise ValueError("Cycle in recorded checkpoint ancestry")
        seen.add(current)
        if not current.is_file():
            raise ValueError(f"Recorded checkpoint does not exist: {current}")
        sidecar = json.loads(current.with_suffix(".json").read_text())
        metadata = sidecar.get("metadata", {})
        if sidecar.get("status") != "committed":
            raise ValueError(f"Checkpoint is not committed: {current}")
        if metadata.get("quality_use_permitted") is False:
            raise ValueError(f"Diagnostic checkpoint cannot provide quality: {current}")
        days = metadata.get("days")
        if (not isinstance(days, list) or not days
                or any(type(day) is not int or not 1 <= day <= 37 for day in days)):
            raise ValueError(f"Checkpoint has invalid actual training days: {current}")
        signature = (metadata.get("scale"), metadata.get("proposal"))
        if architecture is None:
            architecture = signature
        elif signature != architecture:
            raise ValueError("Architecture differs within the release ancestry")
        raw_parent = metadata.get("arguments", {}).get("parent")
        parent = Path(raw_parent).resolve() if raw_parent else None
        record = {"checkpoint": str(current), "train_days": list(days),
                  "train_end_day": max(days),
                  "actual_parent": str(parent) if parent else None}
        if reverse and record["train_end_day"] > reverse[-1]["train_end_day"]:
            raise ValueError("Parent training cutoff is later than its descendant")
        reverse.append(record)
        if current == source:
            break
        if parent is None:
            raise ValueError("Requested cache source is not an actual training ancestor")
        current = parent
    result = list(reversed(reverse))
    if result[0]["train_end_day"] >= result[-1]["train_end_day"]:
        raise ValueError("Release interval must advance the actual training cutoff")
    return result


class SampledRequestBatches(RequestBatches):
    """Use original batch assembly for a saved complete-user, whole-day sample."""

    def __init__(self, root, day, selection_path):
        super().__init__(root, [int(day)], role=2, history_limit=1024)
        selection_path = Path(selection_path).resolve()
        selected = np.load(selection_path, mmap_mode="r")
        if selected.ndim != 1 or selected.dtype != self.records.dtype:
            raise ValueError("Sample records must use the original request schema")
        if not len(selected):
            raise ValueError("Selected users have no requests in this window")
        users = np.unique(selected["uid"])
        expected = self.records[np.isin(self.records["uid"], users)]
        if not np.array_equal(selected, expected):
            raise ValueError("Sample must preserve every chosen user's complete day requests")
        self.records = selected
        # The base class removes records during pickling and reloads this path
        # in spawned workers. Point it at the sample, never the full population.
        self._selection_path = selection_path


def prepare_selection(data, day, output, user_count=4096, seed=20261029, user_offset=0):
    """Sample original training UIDs before looking at activity or feedback.

    All requests of chosen users are retained. The complete UID permutation
    supports reproducible nested expansion or a disjoint offset sample later.
    """
    data, output = Path(data).resolve(), Path(output).resolve()
    if type(user_count) is not int or user_count < 1 or type(user_offset) is not int or user_offset < 0:
        raise ValueError("Expected a positive user count and nonnegative user offset")
    with np.load(data / "population.npz") as population:
        train = np.array(population["train"], copy=True)
    if train.ndim != 1 or len(np.unique(train)) != len(train):
        raise ValueError("Original training population must contain distinct UIDs")
    if user_offset + user_count > len(train):
        raise ValueError("Requested user range exceeds the original training population")
    permutation = np.random.default_rng(seed).permutation(np.sort(train))
    chosen = permutation[user_offset:user_offset + user_count]
    dataset = RequestBatches(data, [int(day)], role=2, history_limit=1024)
    if (not np.isin(dataset.records["uid"], train).all()
            or not np.all(dataset.records["day"] == day)):
        raise ValueError("Original evaluation selection differs from the training cohort or day")
    selected = dataset.records[np.isin(dataset.records["uid"], chosen)]
    output.mkdir(parents=True, exist_ok=True)
    paths = {"uid_permutation_path": output / "uid_permutation.npy",
             "chosen_uids_path": output / "chosen_uids.npy",
             "selection_path": output / "selection.npy"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Preserve an existing selection; choose a fresh output directory")
    np.save(paths["uid_permutation_path"], permutation)
    np.save(paths["chosen_uids_path"], chosen)
    np.save(paths["selection_path"], selected)
    return {"data": str(data), "day": int(day), "seed": int(seed),
            "user_offset": user_offset, "chosen_users": len(chosen),
            "population_users": len(train), "active_users": len(np.unique(selected["uid"])),
            "requests": len(selected), "full_window_requests": len(dataset),
            "history_limit": 1024,
            "selection_rule": "fixed permutation of original training UIDs; all chosen users' day requests",
            "activity_or_feedback_selection": False,
            **{key: str(path) for key, path in paths.items()}}
