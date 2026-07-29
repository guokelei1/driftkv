from __future__ import annotations

import mmap

import pytest
import torch

from hstu_kvcache.migration.design3_store import (
    PageableDramExtentStore,
)


def _values(
    layers: int,
    tokens: int,
    width: int,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(
        layers * tokens * width,
        dtype=torch.float16,
    ).view(layers, tokens, width)
    return base + offset, base + offset + 100


def test_pageable_store_ranges_persist_and_track_coverage(tmp_path) -> None:
    path = tmp_path / "rank0.oldkv.bin"
    store = PageableDramExtentStore.create(
        path,
        (10, 20, 30),
        (3, 2, 4),
        num_layers=2,
        width=3,
    )
    assert store.is_pageable
    assert store.extent(10) == (0, 3)
    assert store.extent(30) == (5, 9)
    assert store.nbytes == 2 * 2 * 9 * 3 * 2
    assert path.stat().st_size == store.mapped_nbytes

    k10, v10 = _values(2, 3, 3, 0)
    store.write_record(10, k10, v10)
    k20a, v20a = _values(2, 1, 3, 20)
    store.write_record(20, k20a, v20a)
    first = store.ledger()
    assert first.complete_records == 1
    assert first.partial_records == 1
    assert first.missing_records == 1
    assert first.covered_tokens == 4

    missing_k = torch.empty((2, 1, 3), dtype=torch.float16)
    missing_v = torch.empty_like(missing_k)
    with pytest.raises(RuntimeError):
        store.read_record_into(
            20,
            missing_k,
            missing_v,
            start=1,
        )

    k20b, v20b = _values(2, 1, 3, 30)
    store.write_record(20, k20b, v20b, start=1)
    k30, v30 = _values(2, 4, 3, 40)
    store.write_record(30, k30, v30)
    read_k = torch.empty((2, 5, 3), dtype=torch.float16)
    read_v = torch.empty_like(read_k)
    expected_bytes = store.read_ranges_into(
        (30, 10),
        (1, 0),
        (3, 3),
        read_k,
        read_v,
    )
    assert expected_bytes == 2 * 5 * 2 * 3 * 2
    assert torch.equal(read_k[:, :2], k30[:, 1:3])
    assert torch.equal(read_v[:, :2], v30[:, 1:3])
    assert torch.equal(read_k[:, 2:], k10)
    assert torch.equal(read_v[:, 2:], v10)
    before_prefault = store.ledger()
    assert store.prefault() == (
        store.mapped_nbytes + mmap.PAGESIZE - 1
    ) // mmap.PAGESIZE
    assert store.prefault(write=True) > 0
    after_prefault = store.ledger()
    assert after_prefault.covered_tokens == before_prefault.covered_tokens
    assert after_prefault.prefault_calls == 2
    store.close(flush=True)
    assert store.closed
    with pytest.raises(RuntimeError):
        store.ledger()

    reopened = PageableDramExtentStore.open(
        path,
        (10, 20, 30),
        (3, 2, 4),
        num_layers=2,
        width=3,
    )
    assert reopened.is_pageable
    assert reopened.ledger().complete_records == 3
    reopened_k = torch.empty_like(k20b)
    reopened_v = torch.empty_like(v20b)
    reopened.read_record_into(
        20,
        reopened_k,
        reopened_v,
        start=1,
    )
    assert torch.equal(reopened_k, k20b)
    assert torch.equal(reopened_v, v20b)
    reopened.close()


def test_pageable_store_rejects_layout_and_transfer_mismatch(
    tmp_path,
) -> None:
    path = tmp_path / "rank1.target.bin"
    with PageableDramExtentStore.create(
        path,
        (1, 2),
        (2, 3),
        num_layers=1,
        width=2,
        dtype=torch.float32,
    ) as store:
        k, v = _values(1, 2, 2, 0)
        with pytest.raises(ValueError):
            store.write_record(1, k, v)
        valid_k = k.float()
        valid_v = v.float()
        with pytest.raises(ValueError):
            store.write_ranges(
                (1,),
                (0,),
                (3,),
                valid_k,
                valid_v,
            )
        with pytest.raises(ValueError):
            store.write_ranges(
                (1, 1),
                (0, 0),
                (1, 1),
                valid_k,
                valid_v,
            )
        store.write_record(1, valid_k, valid_v)
    with pytest.raises(ValueError):
        PageableDramExtentStore.open(
            path,
            (1, 2),
            (2, 4),
            num_layers=1,
            width=2,
            dtype=torch.float32,
        )
    with pytest.raises(ValueError):
        PageableDramExtentStore.open(
            path,
            (3, 4),
            (2, 3),
            num_layers=1,
            width=2,
            dtype=torch.float32,
        )
