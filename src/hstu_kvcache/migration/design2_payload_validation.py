from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from dataclasses import dataclass

import torch

from .cohort_jagged import JaggedMigratedKVBatch
from .design2_integrated import IntegratedAppendOnlyKVBatch

PAYLOAD_HASH_PROTOCOL = "cohortkv-d2-full-payload-v1"


def _canonical_header(
    record_id: int,
    route: str,
    component: str,
    tensor: torch.Tensor,
) -> bytes:
    return json.dumps(
        {
            "component": component,
            "dtype": str(tensor.dtype),
            "record_id": record_id,
            "route": route,
            "shape": list(tensor.shape),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes()


def _update_digest(
    digest,
    header: bytes,
    payload: bytes,
) -> None:
    digest.update(struct.pack("<Q", len(header)))
    digest.update(header)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


@dataclass
class _ComponentState:
    atol: float
    rtol: float
    elements: int = 0
    absolute_error_sum: float = 0.0
    max_absolute_error: float = 0.0
    allclose: bool = True
    bitwise_equal: bool = True
    finite: bool = True


class D2PayloadComparisonAccumulator:
    def __init__(
        self,
        *,
        kv_atol: float = 2e-2,
        kv_rtol: float = 2e-2,
        hidden_atol: float = 2e-5,
        hidden_rtol: float = 2e-5,
    ) -> None:
        if min(kv_atol, kv_rtol, hidden_atol, hidden_rtol) < 0:
            raise ValueError("payload comparison tolerance is negative")
        self._components = {
            "k": _ComponentState(kv_atol, kv_rtol),
            "v": _ComponentState(kv_atol, kv_rtol),
            "last_hidden": _ComponentState(hidden_atol, hidden_rtol),
        }
        self._left_digest = hashlib.sha256()
        self._right_digest = hashlib.sha256()
        self._left_digest.update(PAYLOAD_HASH_PROTOCOL.encode())
        self._right_digest.update(PAYLOAD_HASH_PROTOCOL.encode())
        self._record_ids: set[int] = set()
        self._record_count_by_route: Counter[str] = Counter()
        self._token_count_by_route: Counter[str] = Counter()
        self._failed_record_ids: set[int] = set()
        self._bitwise_failed_record_ids: set[int] = set()
        self._nonfinite_record_ids: set[int] = set()
        self._record_reports: list[dict[str, object]] = []

    def _compare_tensor(
        self,
        record_id: int,
        route: str,
        component: str,
        left: torch.Tensor,
        right: torch.Tensor,
        record_left_digest,
        record_right_digest,
    ) -> dict[str, object]:
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.device != right.device
        ):
            raise ValueError("payload tensor metadata differs")
        state = self._components[component]
        left_finite = bool(torch.isfinite(left).all())
        right_finite = bool(torch.isfinite(right).all())
        finite = left_finite and right_finite
        bitwise = torch.equal(left, right)
        allclose = finite and torch.allclose(
            left,
            right,
            atol=state.atol,
            rtol=state.rtol,
        )
        delta = (left.float() - right.float()).abs()
        elements = left.numel()
        absolute_error_sum = (
            float(delta.sum(dtype=torch.float64).item()) if elements else 0.0
        )
        max_absolute_error = (
            float(delta.max().item()) if elements else 0.0
        )
        header = _canonical_header(record_id, route, component, left)
        left_payload = _tensor_bytes(left)
        right_payload = _tensor_bytes(right)
        _update_digest(self._left_digest, header, left_payload)
        _update_digest(self._right_digest, header, right_payload)
        _update_digest(record_left_digest, header, left_payload)
        _update_digest(record_right_digest, header, right_payload)
        state.elements += elements
        state.absolute_error_sum += absolute_error_sum
        state.max_absolute_error = max(
            state.max_absolute_error,
            max_absolute_error,
        )
        state.allclose = state.allclose and allclose
        state.bitwise_equal = state.bitwise_equal and bitwise
        state.finite = state.finite and finite
        return {
            "allclose": allclose,
            "bitwise_equal": bitwise,
            "elements": elements,
            "finite": finite,
            "max_absolute_error": max_absolute_error,
            "mean_absolute_error": (
                absolute_error_sum / elements if elements else 0.0
            ),
        }

    def add_record(
        self,
        *,
        record_id: int,
        route: str,
        left_k: torch.Tensor,
        left_v: torch.Tensor,
        left_last_hidden: torch.Tensor,
        right_k: torch.Tensor,
        right_v: torch.Tensor,
        right_last_hidden: torch.Tensor,
    ) -> dict[str, object]:
        if record_id in self._record_ids:
            raise ValueError("payload record is duplicated")
        if (
            left_k.ndim != 3
            or left_v.shape != left_k.shape
            or right_k.ndim != 3
            or right_v.shape != right_k.shape
            or left_last_hidden.ndim != 1
            or right_last_hidden.ndim != 1
        ):
            raise ValueError("payload record shape is invalid")
        self._record_ids.add(record_id)
        token_count = left_k.shape[1]
        self._record_count_by_route[route] += 1
        self._token_count_by_route[route] += token_count
        left_digest = hashlib.sha256()
        right_digest = hashlib.sha256()
        left_digest.update(PAYLOAD_HASH_PROTOCOL.encode())
        right_digest.update(PAYLOAD_HASH_PROTOCOL.encode())
        components = {
            "k": self._compare_tensor(
                record_id,
                route,
                "k",
                left_k,
                right_k,
                left_digest,
                right_digest,
            ),
            "v": self._compare_tensor(
                record_id,
                route,
                "v",
                left_v,
                right_v,
                left_digest,
                right_digest,
            ),
            "last_hidden": self._compare_tensor(
                record_id,
                route,
                "last_hidden",
                left_last_hidden,
                right_last_hidden,
                left_digest,
                right_digest,
            ),
        }
        allclose = all(
            value["allclose"] for value in components.values()
        )
        bitwise = all(
            value["bitwise_equal"] for value in components.values()
        )
        finite = all(value["finite"] for value in components.values())
        if not allclose:
            self._failed_record_ids.add(record_id)
        if not bitwise:
            self._bitwise_failed_record_ids.add(record_id)
        if not finite:
            self._nonfinite_record_ids.add(record_id)
        elements = sum(
            int(value["elements"]) for value in components.values()
        )
        absolute_error_sum = sum(
            float(value["mean_absolute_error"]) * int(value["elements"])
            for value in components.values()
        )
        report = {
            "record_id": record_id,
            "route": route,
            "tokens": token_count,
            "elements": elements,
            "allclose": allclose,
            "bitwise_equal": bitwise,
            "finite": finite,
            "max_absolute_error": max(
                float(value["max_absolute_error"])
                for value in components.values()
            ),
            "mean_absolute_error": (
                absolute_error_sum / elements if elements else 0.0
            ),
            "left_sha256": left_digest.hexdigest(),
            "right_sha256": right_digest.hexdigest(),
            "components": components,
        }
        self._record_reports.append(report)
        return report

    def report(self) -> dict[str, object]:
        components = {
            name: {
                "atol": state.atol,
                "rtol": state.rtol,
                "elements": state.elements,
                "allclose": state.allclose,
                "bitwise_equal": state.bitwise_equal,
                "finite": state.finite,
                "max_absolute_error": state.max_absolute_error,
                "mean_absolute_error": (
                    state.absolute_error_sum / state.elements
                    if state.elements
                    else 0.0
                ),
            }
            for name, state in self._components.items()
        }
        elements = sum(
            state.elements for state in self._components.values()
        )
        absolute_error_sum = sum(
            state.absolute_error_sum
            for state in self._components.values()
        )
        return {
            "hash_protocol": PAYLOAD_HASH_PROTOCOL,
            "records": len(self._record_ids),
            "tokens": sum(self._token_count_by_route.values()),
            "elements": elements,
            "allclose": all(
                state.allclose for state in self._components.values()
            ),
            "bitwise_equal": all(
                state.bitwise_equal
                for state in self._components.values()
            ),
            "finite": all(
                state.finite for state in self._components.values()
            ),
            "max_absolute_error": max(
                state.max_absolute_error
                for state in self._components.values()
            ),
            "mean_absolute_error": (
                absolute_error_sum / elements if elements else 0.0
            ),
            "left_sha256": self._left_digest.hexdigest(),
            "right_sha256": self._right_digest.hexdigest(),
            "hashes_match": (
                self._left_digest.hexdigest()
                == self._right_digest.hexdigest()
            ),
            "record_counts_by_route": dict(
                sorted(self._record_count_by_route.items())
            ),
            "token_counts_by_route": dict(
                sorted(self._token_count_by_route.items())
            ),
            "failed_record_ids": sorted(self._failed_record_ids),
            "bitwise_failed_record_ids": sorted(
                self._bitwise_failed_record_ids
            ),
            "nonfinite_record_ids": sorted(
                self._nonfinite_record_ids
            ),
            "record_ids": sorted(self._record_ids),
            "components": components,
            "record_reports": sorted(
                self._record_reports,
                key=lambda value: int(value["record_id"]),
            ),
        }


def _offsets(lengths: torch.Tensor) -> tuple[int, ...]:
    values = tuple(int(value) for value in lengths.tolist())
    output = [0]
    for value in values:
        output.append(output[-1] + value)
    return tuple(output)


def compare_jagged_payloads(
    accumulator: D2PayloadComparisonAccumulator,
    *,
    route: str,
    left: JaggedMigratedKVBatch,
    left_last_hidden: torch.Tensor,
    right: JaggedMigratedKVBatch,
    right_last_hidden: torch.Tensor,
) -> tuple[dict[str, object], ...]:
    if (
        left.record_ids != right.record_ids
        or not torch.equal(left.lengths, right.lengths)
        or left_last_hidden.shape
        != (left.batch_size, left.k.shape[2])
        or right_last_hidden.shape != left_last_hidden.shape
    ):
        raise ValueError("jagged payload comparison metadata differs")
    left_offsets = _offsets(left.lengths)
    right_offsets = _offsets(right.lengths)
    output = []
    for row, record_id in enumerate(left.record_ids):
        output.append(
            accumulator.add_record(
                record_id=record_id,
                route=route,
                left_k=left.k[
                    :, left_offsets[row] : left_offsets[row + 1]
                ],
                left_v=left.v[
                    :, left_offsets[row] : left_offsets[row + 1]
                ],
                left_last_hidden=left_last_hidden[row],
                right_k=right.k[
                    :, right_offsets[row] : right_offsets[row + 1]
                ],
                right_v=right.v[
                    :, right_offsets[row] : right_offsets[row + 1]
                ],
                right_last_hidden=right_last_hidden[row],
            )
        )
    return tuple(output)


def compare_jagged_to_append_only(
    accumulator: D2PayloadComparisonAccumulator,
    *,
    route: str,
    left: JaggedMigratedKVBatch,
    left_last_hidden: torch.Tensor,
    right: IntegratedAppendOnlyKVBatch,
    right_last_hidden: torch.Tensor,
) -> tuple[dict[str, object], ...]:
    if (
        left.record_ids != right.record_ids
        or not torch.equal(left.lengths, right.lengths)
        or left_last_hidden.shape
        != (left.batch_size, left.k.shape[2])
        or right_last_hidden.shape != left_last_hidden.shape
    ):
        raise ValueError("append-only payload comparison metadata differs")
    left_offsets = _offsets(left.lengths)
    retained_offsets = _offsets(right.retained.lengths)
    suffix_offsets = _offsets(right.suffix.lengths)
    output = []
    for row, record_id in enumerate(left.record_ids):
        right_k = torch.cat(
            (
                right.retained.k[
                    :,
                    retained_offsets[row] : retained_offsets[row + 1],
                ],
                right.suffix.k[
                    :,
                    suffix_offsets[row] : suffix_offsets[row + 1],
                ],
            ),
            dim=1,
        )
        right_v = torch.cat(
            (
                right.retained.v[
                    :,
                    retained_offsets[row] : retained_offsets[row + 1],
                ],
                right.suffix.v[
                    :,
                    suffix_offsets[row] : suffix_offsets[row + 1],
                ],
            ),
            dim=1,
        )
        output.append(
            accumulator.add_record(
                record_id=record_id,
                route=route,
                left_k=left.k[
                    :, left_offsets[row] : left_offsets[row + 1]
                ],
                left_v=left.v[
                    :, left_offsets[row] : left_offsets[row + 1]
                ],
                left_last_hidden=left_last_hidden[row],
                right_k=right_k,
                right_v=right_v,
                right_last_hidden=right_last_hidden[row],
            )
        )
        del right_k
        del right_v
    return tuple(output)
