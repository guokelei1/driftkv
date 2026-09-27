"""Group frozen users by stream length without changing their rank or membership."""

from __future__ import annotations


def ordered_uids(by_user, max_length: int = 1024, *, uids=None) -> list[int]:
    """Full histories first, then decreasing append workload, requests, and UID.

    ``uids`` permits a caller to retain an already selected canary subset. Apply
    any user limit using the old request-count order before calling this helper.
    Source fields describe the retained workload; no labels or scores are read.
    """
    chosen = list(by_user) if uids is None else list(uids)
    workload = {}
    for uid in chosen:
        rows = by_user[uid]
        prefixes = {int(row["cache_length"]) - int(row["append_count_since_cutover"])
                    + int(row["rolling_evictions"]) for row in rows}
        if len(prefixes) != 1:
            raise RuntimeError(f"user {uid} has inconsistent saved initial cache lengths")
        prefix = prefixes.pop()
        if not 0 < prefix <= max_length:
            raise RuntimeError(f"user {uid} has invalid saved initial cache length {prefix}")
        workload[uid] = (prefix < max_length,
                         -max(int(row["append_count_since_cutover"]) for row in rows),
                         -len(rows), int(uid))
    return sorted(chosen, key=workload.__getitem__)
