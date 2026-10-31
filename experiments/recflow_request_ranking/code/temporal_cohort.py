"""Fixed existing-user population; strictly future requests, including no-feedback ones.

This is a separately declared temporal holdout after the original unseen-user
study. It never changes that study or selects users using future activity/loss.
The original batch collator and all model/metric implementations stay unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .batches import RequestBatches
from .data_common import sha256, stable_hash

PROTOCOL = "existing_train_users_future_requests_v1"
SEED = 20261028


def select_users(population, count=2048, seed=SEED):
    reserved = np.union1d(population["cache_fit_uids"], population["cache_validation_uids"])
    eligible = np.sort(population["uids"][(population["role"] == 0)
                       & (population["initial_history_count"] >= 1024)])
    eligible = eligible[~np.isin(eligible, reserved)]
    if len(np.unique(eligible)) != len(eligible) or not 0 < count <= len(eligible):
        raise ValueError("Invalid fixed population size or duplicate user IDs")
    order = np.lexsort((eligible, stable_hash(eligible, seed)))
    return np.sort(eligible[order[:count]]).astype(np.int32)


def evaluation_mask(records, uids):
    # Deliberately no observed/positive/first-request/future-activity filter.
    return ((records["role"] == 0) & records["resident"] & records["valid_request"]
            & np.isin(records["uid"], uids))


class TemporalCohortBatches(RequestBatches):
    def __init__(self, root, days, population_path):
        self.root, self.history_limit = Path(root), 1024
        self.stage_dir = json.loads((self.root / "stage_index_manifest.json").read_text())["stage_dir"]
        with np.load(population_path) as population:
            uids = population["uids"]
        if len(np.unique(uids)) != len(uids) or len(days) != len(set(days)):
            raise ValueError("Duplicate population or evaluation days")
        parts = []
        for day in days:
            if day not in range(28, 37):
                raise ValueError("Only the frozen nine next-day windows are allowed")
            records = np.load(self.root / self.stage_dir / f"d{day:02d}/index.npy", mmap_mode="r")
            selected = records[evaluation_mask(records, uids)]
            if len(selected) and np.any(selected["history_length"] < 1024):
                raise ValueError("This population should have real 1024-event causal prefixes")
            parts.append(selected)
        self.records = np.concatenate(parts)
        self._events, self._daily = None, {}


def validate_checkpoint_pair(current, parent, current_sha, parent_sha, case):
    if (current_sha != case["checkpoint_sha256"] or parent_sha != case["parent_sha256"]
            or current.get("parent_checkpoint_sha256") != parent_sha
            or current["scale"] != "max" or parent["scale"] != "max"
            or current.get("quality_use_permitted") is not True
            or parent.get("quality_use_permitted") is not True):
        raise ValueError("Different checkpoint pair or diagnostic state")
    if (current["days"] != [case["day"] - 1] or max(parent["days"]) >= min(current["days"])
            or max(current["days"]) >= case["day"]):
        raise ValueError("Evaluation is not strictly after the actual training cutoff")


def read_plan(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    if (plan["protocol"] != PROTOCOL or plan["status"] != "frozen_before_cohort_scores"
            or [row["day"] for row in plan["schedule"]] != list(range(28, 37))):
        raise ValueError("Invalid frozen temporal protocol")
    for name, digest in {**plan["driver_sources"], **plan["input_sha256"]}.items():
        if sha256(name) != digest:
            raise ValueError(f"Frozen temporal input/source changed: {name}")
    with np.load(plan["population"]) as selected, np.load(Path(plan["data"]) / "history/population.npz") as original:
        expected = select_users(original, plan["population_size"], plan["population_seed"])
        if not np.array_equal(selected["uids"], expected):
            raise ValueError("Cohort differs from the outcome-independent UID selection")
    return plan
