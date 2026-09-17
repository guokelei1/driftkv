"""Scale-independent, causal inputs for executable competitor diagnostics.

Dataset loading and checkpoint vocabulary metadata remain explicit caller
inputs. Candidate panels depend on all selected evaluation histories, never
on model outputs, future labels or the evaluator's batch boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_uid_split(path: Path) -> dict[str, list[int]]:
    """Read ordered evaluation/fitting/selection users without a frozen population."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    split = {name: raw.get(name, []) for name in ("evaluation", "fit", "selection")}
    seen: set[int] = set()
    for name, uids in split.items():
        if not isinstance(uids, list) or any(type(uid) is not int for uid in uids):
            raise ValueError(f"{name} must be a list of integer UIDs")
        if len(set(uids)) != len(uids):
            raise ValueError(f"duplicate {name} UIDs")
        if seen.intersection(uids):
            raise ValueError("evaluation, fit and selection UID groups must be disjoint")
        seen.update(uids)
    if not split["evaluation"]:
        raise ValueError("evaluation must contain at least one UID")
    return split


def history_arrays(
    history, uids: np.ndarray, cutover: int, history_length: int
) -> tuple[np.ndarray, ...]:
    """Return timestamps/items/behaviors/deltas/query_deltas for static windows.

    Each window contains exactly history_length events strictly before cutover.
    Its first delta is zero, matching static Full initialization; subsequent
    deltas are real adjacent timestamp differences. Tail replay must retain
    these values rather than reset the first replayed token's delta again.
    """
    if history_length < 1 or len(uids) == 0:
        raise ValueError("history_length and the number of users must be positive")
    timestamps, items, behaviors = [], [], []
    for uid in uids:
        row_items, row_behaviors, row_timestamps = history.prefix(int(uid), cutover, history_length)
        row_timestamps = np.asarray(row_timestamps, dtype=np.int64)
        if len(row_items) != history_length:
            raise ValueError(f"selected uid {uid} lacks a full {history_length}-event prefix")
        if np.any(row_timestamps >= cutover) or np.any(np.diff(row_timestamps) < 0):
            raise ValueError("history must be chronological and strictly before cutover")
        timestamps.append(row_timestamps)
        items.append(row_items)
        behaviors.append(row_behaviors)
    timestamp_array = np.stack(timestamps)
    item_array = np.asarray(items, dtype=np.int64)
    behavior_array = np.asarray(behaviors, dtype=np.int64)
    deltas = np.zeros_like(timestamp_array, dtype=np.float32)
    deltas[:, 1:] = np.diff(timestamp_array, axis=1)
    query_deltas = (cutover - timestamp_array[:, -1]).astype(np.float32)
    return timestamp_array, item_array, behavior_array, deltas, query_deltas


def _recent_unique(values: np.ndarray, excluded: set[int], known_items: int) -> list[int]:
    selected = []
    for value in values[::-1]:
        item = int(value)
        if 0 < item < known_items and item not in excluded:
            selected.append(item)
            excluded.add(item)
            if len(selected) == 16:
                break
    return selected


def make_candidate_panel(
    items: np.ndarray, known_items: int, count: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Insight 1's fixed 16 recent / 16 old / novel-fill policy.

    known_items is the exclusive upper bound for known compact item IDs; zero
    and OOV buckets are ineligible candidates. The caller supplies all selected
    evaluation users at once to form a common pre-cutover popularity bank.
    Modes are 0=recent, 1=old-only and 2=unseen-in-this-user's-history. Mode 2
    describes history novelty, not an observed negative label.
    """
    items = np.asarray(items)
    if items.ndim != 2 or not items.shape[0] or not items.shape[1]:
        raise ValueError("items must be a nonempty [users,history] array")
    if count != 64 or known_items <= 1:
        raise ValueError("this panel uses 64 candidates and a positive known-item range")
    known = items[(items > 0) & (items < known_items)]
    observed, counts = np.unique(known, return_counts=True)
    bank = observed[np.lexsort((observed, -counts))[:16_384]].tolist()
    panels, modes = [], []
    for row in items:
        recent_set = {int(value) for value in row[-256:] if 0 < int(value) < known_items}
        full_set = {int(value) for value in row if 0 < int(value) < known_items}
        recent = _recent_unique(row[-256:], set(), known_items)
        old = _recent_unique(row[:-256], recent_set.copy(), known_items)
        novel_count = count - len(recent) - len(old)
        novel = [int(value) for value in bank if int(value) not in full_set][:novel_count]
        if len(novel) != novel_count:
            raise ValueError(
                f"candidate panel cannot be filled: recent={len(recent)} old={len(old)} "
                f"novel={len(novel)}/{novel_count}; supply a sufficient evaluation history bank"
            )
        panels.append(recent + old + novel)
        modes.append([0] * len(recent) + [1] * len(old) + [2] * len(novel))
    return np.asarray(panels, dtype=np.int64), np.asarray(modes, dtype=np.uint8)
