"""Nested request budgets stratified by user and day, independent of labels."""
from __future__ import annotations

import numpy as np

from .data_common import stable_hash


def stratified_indices(records, limit, salt=9427):
    if limit >= len(records):
        return np.arange(len(records))
    if limit < 1:
        raise ValueError("Request budget must be positive")
    hashed = stable_hash(records["rid"], salt)
    order = np.lexsort((hashed, records["uid"], records["day"]))
    day, uid = records["day"][order], records["uid"][order]
    starts = np.flatnonzero(np.r_[True, (day[1:] != day[:-1]) | (uid[1:] != uid[:-1])])
    counts = np.diff(np.r_[starts, len(records)])
    within = np.arange(len(records))-np.repeat(starts, counts)
    capacity = np.repeat(counts, counts)
    # All nonempty user-days enter once before additional requests. Subsequent
    # quota increases are proportional to remaining cell size. One fixed total
    # order guarantees that every smaller requested dataset nests in a larger.
    priority = (within-.5)/np.maximum(1, capacity-1)
    priority[within == 0] = -2 + hashed[order][within == 0].astype(np.float64)/2**64
    chosen = order[np.lexsort((hashed[order], priority))[:limit]]
    return np.sort(chosen)
