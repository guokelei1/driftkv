"""Small state helpers for the KV-map evolution mechanism study.

These functions do not choose users, query panels, fitting settings or teacher
semantics. Counters are measured from the trajectory's cutover initialization.
"""
from __future__ import annotations

from collections import Counter
import copy

from hstu_kvcache.models import HSTUKVCache


def clone_native_trajectory(native):
    """Fork a native trajectory while borrowing immutable events and K/V.

    NativeTrajectory.advance replaces its cache rather than modifying it.
    Callers must likewise treat the initially shared cache/events as immutable.
    The two fields that advance mutates in place or replaces independently are
    copied explicitly; callbacks are borrowed unchanged.
    """
    result = copy.copy(native)
    result.ledger = Counter(native.ledger)
    result.producers = native.producers.copy()
    return result


def teacher_forced_cache(initial_mapped_cache, teacher_cache, evicted_tokens):
    """Keep surviving initial mapped rows and use teacher K/V for new rows.

    This is a diagnostic counterfactual, not a deployable correction method.
    The teacher must have the same causal event prefix and FIFO cap, starting
    from a current-model rebuild of the same cutover prefix. Its new rows were
    produced through its own teacher history, never through the student's.

    For native immutable old rows, replacing every newly written row with its
    teacher counterpart is equivalent to this snapshot composition. No teacher
    or initial mapped tensor is modified. ``evicted_tokens`` counts all FIFO
    removals since cutover, including removals after old rows have disappeared.
    """
    if int(evicted_tokens) != evicted_tokens or evicted_tokens < 0:
        raise ValueError("evicted tokens must be a nonnegative integer")
    for cache in (initial_mapped_cache, teacher_cache):
        if (cache.k.ndim != 4 or cache.k.shape != cache.v.shape
                or cache.seq_len != cache.k.shape[2]):
            raise ValueError("teacher forcing requires unpadded aligned K/V caches")
    old, teacher = initial_mapped_cache, teacher_cache
    if (old.k.shape[:2] != teacher.k.shape[:2]
            or old.k.shape[3] != teacher.k.shape[3]
            or old.k.dtype != teacher.k.dtype or old.v.dtype != teacher.v.dtype
            or old.k.device != teacher.k.device or old.v.device != teacher.v.device):
        raise ValueError("teacher and mapped cache geometry, dtype or device differ")
    dropped = min(int(evicted_tokens), old.seq_len)
    inherited = old.seq_len - dropped
    if inherited > teacher.seq_len:
        raise ValueError("teacher cache cannot contain the surviving inherited prefix")
    keys, values = teacher.k.detach().clone(), teacher.v.detach().clone()
    if inherited:
        keys[:, :, :inherited].copy_(old.k[:, :, dropped:].detach())
        values[:, :, :inherited].copy_(old.v[:, :, dropped:].detach())
    return HSTUKVCache(keys, values, teacher.seq_len)


def checkpoint_metadata(native, initial_length):
    """Describe growth and inherited-row survival since cutover, not accuracy.

    ``native`` must retain its cutover-relative appended_tokens/evicted_tokens
    counters. First real request and zero-appended cutover need not coincide.
    """
    if int(initial_length) != initial_length or initial_length < 0:
        raise ValueError("initial length must be a nonnegative integer")
    appended, evicted = int(native.appended_tokens), int(native.evicted_tokens)
    inherited = max(0, int(initial_length) - evicted)
    length = int(native.cache.seq_len)
    if appended < 0 or evicted < 0 or inherited > length:
        raise ValueError("trajectory counters do not describe this cutover prefix")
    return dict(time=int(native.time), event_cursor=int(native.cursor),
                appended_events=appended, evicted_events=evicted,
                history_rows=length, surviving_inherited_rows=inherited,
                inherited_fraction=inherited / length if length else 0.)
