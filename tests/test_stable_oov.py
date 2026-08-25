from __future__ import annotations

import numpy as np

from hstu_kvcache.data import apply_stable_oov_buckets, stable_oov_bucket


def test_stable_oov_buckets_preserve_known_ids_and_are_repeatable() -> None:
    raw = np.asarray(["known", "new-a", "new-b", "new-a"])
    mapped = np.asarray([12, 0, 0, 0])
    values = apply_stable_oov_buckets(raw, mapped, known_vocab_size=100, buckets=32)
    assert values[0] == 12
    assert values[1] == values[3] == stable_oov_bucket("new-a", known_vocab_size=100, buckets=32)
    assert (100 <= values[1] < 132) and (100 <= values[2] < 132)


def test_zero_oov_buckets_retains_legacy_single_oov() -> None:
    raw = np.asarray(["new-a"])
    assert apply_stable_oov_buckets(raw, np.asarray([0]), known_vocab_size=100, buckets=0).tolist() == [0]
