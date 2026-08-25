"""Stable fallback identities for items absent from a frozen mapping.

The original Small development mapping used a single OOV id.  That makes all
new items indistinguishable and, as the catalog drifts, can dominate a long
history.  These helpers preserve the frozen known ids and deterministically
route only unknown raw ids into a fixed extension of the embedding table.
"""

from __future__ import annotations

import hashlib

import numpy as np


def stable_oov_bucket(raw_item_id: object, *, known_vocab_size: int, buckets: int) -> int:
    """Return the embedding id for an unknown item in a stable hash bucket."""
    if buckets < 1:
        return 0
    digest = hashlib.blake2b(str(raw_item_id).encode("utf-8"), digest_size=8, person=b"evokv-oov-v1").digest()
    bucket = int.from_bytes(digest, "little") % buckets
    return known_vocab_size + bucket


def apply_stable_oov_buckets(
    raw_item_ids: np.ndarray,
    mapped_item_ids: np.ndarray,
    *,
    known_vocab_size: int,
    buckets: int,
) -> np.ndarray:
    """Keep mapped ids and replace mapping misses (encoded as zero) stably."""
    values = np.asarray(mapped_item_ids, dtype=np.int64).copy()
    if buckets < 1:
        return values
    raw = np.asarray(raw_item_ids)
    if raw.shape != values.shape:
        raise ValueError("raw and mapped item arrays must have the same shape")
    missing = values == 0
    values[missing] = [
        stable_oov_bucket(value, known_vocab_size=known_vocab_size, buckets=buckets)
        for value in raw[missing]
    ]
    return values
