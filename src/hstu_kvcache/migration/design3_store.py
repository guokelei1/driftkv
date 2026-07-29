from __future__ import annotations

import gc
import hashlib
import json
import mmap
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import torch

_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
_HEADER_BYTES = 4096
_MAGIC = "evokv_pageable_dram_extent_v1"


@dataclass(frozen=True)
class DramExtentStoreLedger:
    path: str
    layout_sha256: str
    records: int
    total_tokens: int
    complete_records: int
    partial_records: int
    missing_records: int
    covered_tokens: int
    payload_nbytes: int
    mapped_nbytes: int
    read_bytes: int
    written_bytes: int
    prefault_calls: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


class PageableDramExtentStore:
    def __init__(
        self,
        path: str | Path,
        record_ids: Sequence[int],
        lengths: Sequence[int],
        *,
        num_layers: int,
        width: int,
        dtype: torch.dtype = torch.float16,
        create: bool,
    ) -> None:
        resolved_ids = tuple(int(value) for value in record_ids)
        resolved_lengths = tuple(int(value) for value in lengths)
        if (
            not resolved_ids
            or resolved_ids != tuple(sorted(resolved_ids))
            or len(set(resolved_ids)) != len(resolved_ids)
            or len(resolved_lengths) != len(resolved_ids)
            or any(value < 0 for value in resolved_ids)
            or any(value < 1 for value in resolved_lengths)
            or num_layers < 1
            or width < 1
            or dtype not in _DTYPES
        ):
            raise ValueError("pageable DRAM extent layout is invalid")
        self.path = Path(path)
        self.record_ids = resolved_ids
        self.lengths = resolved_lengths
        self.num_layers = int(num_layers)
        self.width = int(width)
        self.dtype = dtype
        self._record_rows = {
            record_id: row
            for row, record_id in enumerate(self.record_ids)
        }
        offsets = [0]
        for length in self.lengths:
            offsets.append(offsets[-1] + length)
        self.offsets = tuple(offsets)
        self.total_tokens = offsets[-1]
        self.element_size = torch.empty((), dtype=dtype).element_size()
        self.payload_nbytes = (
            2
            * self.num_layers
            * self.total_tokens
            * self.width
            * self.element_size
        )
        self.layout_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "record_ids": self.record_ids,
                    "lengths": self.lengths,
                    "num_layers": self.num_layers,
                    "width": self.width,
                    "dtype": str(self.dtype),
                    "layout": "kv_layer_major_stable_record_extent_v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.mapped_nbytes = (
            _HEADER_BYTES + self.payload_nbytes + self.total_tokens
        )
        self._read_bytes = 0
        self._written_bytes = 0
        self._prefault_calls = 0
        self._closed = False
        self._mapping: mmap.mmap | None = None
        self._raw: torch.Tensor | None = None
        self._values: torch.Tensor | None = None
        self._coverage: torch.Tensor | None = None
        self._open(create)

    @classmethod
    def create(
        cls,
        path: str | Path,
        record_ids: Sequence[int],
        lengths: Sequence[int],
        *,
        num_layers: int,
        width: int,
        dtype: torch.dtype = torch.float16,
    ) -> Self:
        return cls(
            path,
            record_ids,
            lengths,
            num_layers=num_layers,
            width=width,
            dtype=dtype,
            create=True,
        )

    @classmethod
    def open(
        cls,
        path: str | Path,
        record_ids: Sequence[int],
        lengths: Sequence[int],
        *,
        num_layers: int,
        width: int,
        dtype: torch.dtype = torch.float16,
    ) -> Self:
        return cls(
            path,
            record_ids,
            lengths,
            num_layers=num_layers,
            width=width,
            dtype=dtype,
            create=False,
        )

    @property
    def nbytes(self) -> int:
        return self.payload_nbytes

    @property
    def is_pageable(self) -> bool:
        self._ensure_open()
        assert self._values is not None
        return (
            self._values.device.type == "cpu"
            and not self._values.is_pinned()
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def _open(self, create: bool) -> None:
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        else:
            flags = os.O_RDWR
        descriptor = os.open(self.path, flags, 0o600)
        try:
            if create:
                os.ftruncate(descriptor, self.mapped_nbytes)
                os.pwrite(descriptor, self._header(), 0)
            elif os.fstat(descriptor).st_size != self.mapped_nbytes:
                raise ValueError(
                    "pageable DRAM extent file size differs from its layout"
                )
            elif os.pread(descriptor, _HEADER_BYTES, 0) != self._header():
                raise ValueError(
                    "pageable DRAM extent file layout differs"
                )
            mapping = mmap.mmap(
                descriptor,
                self.mapped_nbytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(descriptor)
        raw = torch.frombuffer(
            mapping,
            dtype=torch.uint8,
            count=self.mapped_nbytes,
        )
        values = raw[
            _HEADER_BYTES : _HEADER_BYTES + self.payload_nbytes
        ].view(self.dtype).view(
            2,
            self.num_layers,
            self.total_tokens,
            self.width,
        )
        coverage = raw[_HEADER_BYTES + self.payload_nbytes :]
        if (
            values.device.type != "cpu"
            or values.is_pinned()
            or coverage.device.type != "cpu"
            or coverage.is_pinned()
        ):
            mapping.close()
            raise RuntimeError("extent mapping is not ordinary pageable DRAM")
        self._mapping = mapping
        self._raw = raw
        self._values = values
        self._coverage = coverage

    def _header(self) -> bytes:
        payload = json.dumps(
            {
                "magic": _MAGIC,
                "layout_sha256": self.layout_sha256,
                "payload_nbytes": self.payload_nbytes,
                "coverage_bytes": self.total_tokens,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if len(payload) + 8 > _HEADER_BYTES:
            raise RuntimeError("pageable DRAM extent header is too large")
        return (
            len(payload).to_bytes(8, "little")
            + payload
            + bytes(_HEADER_BYTES - len(payload) - 8)
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("pageable DRAM extent store is closed")

    def extent(self, record_id: int) -> tuple[int, int]:
        self._ensure_open()
        row = self._record_rows.get(int(record_id))
        if row is None:
            raise KeyError(f"unknown extent record {record_id}")
        return self.offsets[row], self.offsets[row + 1]

    def _resolve_ranges(
        self,
        record_ids: Sequence[int],
        starts: Sequence[int],
        stops: Sequence[int],
    ) -> tuple[tuple[int, int, int], ...]:
        ids = tuple(int(value) for value in record_ids)
        resolved_starts = tuple(int(value) for value in starts)
        resolved_stops = tuple(int(value) for value in stops)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or len(resolved_starts) != len(ids)
            or len(resolved_stops) != len(ids)
        ):
            raise ValueError("extent ranges are invalid")
        output = []
        for record_id, start, stop in zip(
            ids,
            resolved_starts,
            resolved_stops,
            strict=True,
        ):
            row = self._record_rows.get(record_id)
            if (
                row is None
                or start < 0
                or stop <= start
                or stop > self.lengths[row]
            ):
                raise ValueError("extent range exceeds its record")
            output.append(
                (
                    self.offsets[row] + start,
                    self.offsets[row] + stop,
                    stop - start,
                )
            )
        return tuple(output)

    def _validate_buffer(
        self,
        value: torch.Tensor,
        tokens: int,
    ) -> None:
        if (
            value.shape
            != (self.num_layers, tokens, self.width)
            or value.dtype != self.dtype
            or value.device.type != "cpu"
            or not value.is_contiguous()
        ):
            raise ValueError("extent transfer buffer differs")

    def write_ranges(
        self,
        record_ids: Sequence[int],
        starts: Sequence[int],
        stops: Sequence[int],
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> int:
        self._ensure_open()
        ranges = self._resolve_ranges(record_ids, starts, stops)
        tokens = sum(length for _, _, length in ranges)
        self._validate_buffer(k, tokens)
        self._validate_buffer(v, tokens)
        assert self._values is not None
        assert self._coverage is not None
        source_offset = 0
        for target_start, target_stop, length in ranges:
            source_stop = source_offset + length
            self._values[0, :, target_start:target_stop].copy_(
                k[:, source_offset:source_stop]
            )
            self._values[1, :, target_start:target_stop].copy_(
                v[:, source_offset:source_stop]
            )
            self._coverage[target_start:target_stop].fill_(1)
            source_offset = source_stop
        written = (
            2
            * tokens
            * self.num_layers
            * self.width
            * self.element_size
        )
        self._written_bytes += written
        return written

    def write_record(
        self,
        record_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        start: int = 0,
    ) -> int:
        stop = start + k.shape[1] if k.ndim == 3 else start
        return self.write_ranges(
            (record_id,),
            (start,),
            (stop,),
            k,
            v,
        )

    def read_ranges_into(
        self,
        record_ids: Sequence[int],
        starts: Sequence[int],
        stops: Sequence[int],
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> int:
        self._ensure_open()
        ranges = self._resolve_ranges(record_ids, starts, stops)
        tokens = sum(length for _, _, length in ranges)
        self._validate_buffer(k, tokens)
        self._validate_buffer(v, tokens)
        assert self._values is not None
        assert self._coverage is not None
        target_offset = 0
        for source_start, source_stop, length in ranges:
            if not bool(
                torch.all(
                    self._coverage[source_start:source_stop] == 1
                )
            ):
                raise RuntimeError("extent range has not been materialized")
            target_stop = target_offset + length
            k[:, target_offset:target_stop].copy_(
                self._values[0, :, source_start:source_stop]
            )
            v[:, target_offset:target_stop].copy_(
                self._values[1, :, source_start:source_stop]
            )
            target_offset = target_stop
        read = (
            2
            * tokens
            * self.num_layers
            * self.width
            * self.element_size
        )
        self._read_bytes += read
        return read

    def read_record_into(
        self,
        record_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        start: int = 0,
    ) -> int:
        stop = start + k.shape[1] if k.ndim == 3 else start
        return self.read_ranges_into(
            (record_id,),
            (start,),
            (stop,),
            k,
            v,
        )

    def prefault(self, *, write: bool = False) -> int:
        self._ensure_open()
        assert self._raw is not None
        pages = self._raw[:: mmap.PAGESIZE]
        if write:
            pages.bitwise_or_(0)
        else:
            int(pages.sum(dtype=torch.int64))
        self._prefault_calls += 1
        return pages.numel()

    def ledger(self) -> DramExtentStoreLedger:
        self._ensure_open()
        assert self._coverage is not None
        covered_tokens = int(
            torch.count_nonzero(self._coverage == 1)
        )
        complete = 0
        partial = 0
        missing = 0
        for start, stop in zip(
            self.offsets[:-1],
            self.offsets[1:],
            strict=True,
        ):
            covered = int(
                torch.count_nonzero(
                    self._coverage[start:stop] == 1
                )
            )
            if covered == stop - start:
                complete += 1
            elif covered == 0:
                missing += 1
            else:
                partial += 1
        return DramExtentStoreLedger(
            path=str(self.path),
            layout_sha256=self.layout_sha256,
            records=len(self.record_ids),
            total_tokens=self.total_tokens,
            complete_records=complete,
            partial_records=partial,
            missing_records=missing,
            covered_tokens=covered_tokens,
            payload_nbytes=self.payload_nbytes,
            mapped_nbytes=self.mapped_nbytes,
            read_bytes=self._read_bytes,
            written_bytes=self._written_bytes,
            prefault_calls=self._prefault_calls,
        )

    def close(self, *, flush: bool = False) -> None:
        if self._closed:
            return
        mapping = self._mapping
        if flush and mapping is not None:
            mapping.flush()
        self._coverage = None
        self._values = None
        self._raw = None
        gc.collect()
        if mapping is not None:
            mapping.close()
        self._mapping = None
        self._closed = True

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type
        del exc_value
        del traceback
        self.close()
