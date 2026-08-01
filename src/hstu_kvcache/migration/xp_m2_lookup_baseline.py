from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..streaming.sharded_edge import modulo_local_rows
from ..streaming.xp_projected_edge import (
    XP_PROJECTED_CHECKPOINT_SCHEMA,
    TrainableProjectedModuloEmbedding,
)
from .xp_exact_baseline import (
    XPBaselineInputs,
    XPBaselineRecord,
    canonical_sha256,
    file_sha256,
)

PROTOCOL = "evokv_xp_m2_append_aware_lookup_development_v0"
APPEND_TOKENS = 32
RETAINED_STRATA = (
    ("64_127", 64, 128),
    ("128_255", 128, 256),
    ("256_383", 256, 384),
    ("384_480", 384, 481),
)


@dataclass(frozen=True)
class LookupRequest:
    record_id: int
    owner_rank: int
    item_ids: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.record_id < 0
            or self.owner_rank < 0
            or self.item_ids.ndim != 1
            or self.item_ids.dtype.kind not in {"i", "u"}
            or len(self.item_ids) < 1
        ):
            raise ValueError("XP M2 lookup request differs")


@dataclass(frozen=True)
class RetainedBudgetSelection:
    fraction_requested: float
    universe_records: int
    universe_retained_tokens: int
    selected_record_ids: tuple[int, ...]
    selected_retained_tokens: int
    strata: tuple[dict[str, object], ...]

    @property
    def fraction_realized(self) -> float:
        if self.universe_retained_tokens == 0:
            return 0.0
        return self.selected_retained_tokens / self.universe_retained_tokens

    @property
    def record_ids_sha256(self) -> str:
        return canonical_sha256(
            {"record_ids": list(self.selected_record_ids)}
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "fraction_requested": self.fraction_requested,
            "fraction_realized": self.fraction_realized,
            "universe_records": self.universe_records,
            "universe_retained_tokens": self.universe_retained_tokens,
            "selected_records": len(self.selected_record_ids),
            "selected_retained_tokens": self.selected_retained_tokens,
            "selected_record_ids_sha256": self.record_ids_sha256,
            "strata": list(self.strata),
        }


def record_extents(
    record: XPBaselineRecord,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(record.items("target"), dtype=np.int64)
    retained_length = len(target) - APPEND_TOKENS
    if retained_length < 1:
        raise ValueError("XP M2 retained extent is empty")
    retained = target[:retained_length]
    append = target[retained_length:]
    old = np.asarray(record.items("old"), dtype=np.int64)
    evicted_length = len(old) - retained_length
    if (
        len(append) != APPEND_TOKENS
        or evicted_length < 0
        or not np.array_equal(old[evicted_length:], retained)
    ):
        raise ValueError("XP M2 natural HET extents differ")
    return retained, append


def retained_shape_stratum(length: int) -> str:
    for name, lower, upper in RETAINED_STRATA:
        if lower <= length < upper:
            return name
    raise ValueError("XP M2 retained length is outside frozen strata")


def _selection_order(
    benchmark_sha256: str,
    stratum: str,
    record_id: int,
) -> bytes:
    return hashlib.sha256(
        (
            f"{PROTOCOL}:{benchmark_sha256}:{stratum}:"
            f"{record_id}"
        ).encode()
    ).digest()


def _nearest_prefix_count(
    lengths: Sequence[int],
    target_tokens: float,
) -> int:
    cumulative = 0
    candidates = [(abs(target_tokens), 0, 0)]
    for count, length in enumerate(lengths, start=1):
        cumulative += int(length)
        candidates.append(
            (abs(cumulative - target_tokens), cumulative, count)
        )
    return min(candidates)[2]


def select_retained_budget(
    records: Sequence[XPBaselineRecord],
    fraction: float,
    benchmark_sha256: str,
) -> RetainedBudgetSelection:
    if not records or not 0.0 <= fraction <= 1.0:
        raise ValueError("XP M2 retained budget differs")
    grouped: dict[str, list[tuple[XPBaselineRecord, int]]] = {
        name: [] for name, _, _ in RETAINED_STRATA
    }
    total_tokens = 0
    for record in records:
        retained, _ = record_extents(record)
        length = len(retained)
        grouped[retained_shape_stratum(length)].append((record, length))
        total_tokens += length
    selected: list[int] = []
    selected_tokens = 0
    strata = []
    for name, _, _ in RETAINED_STRATA:
        ordered = sorted(
            grouped[name],
            key=lambda value: (
                _selection_order(
                    benchmark_sha256,
                    name,
                    value[0].record_id,
                ),
                value[0].record_id,
            ),
        )
        stratum_tokens = sum(value[1] for value in ordered)
        if fraction == 0.0:
            count = 0
        elif fraction == 1.0:
            count = len(ordered)
        else:
            count = _nearest_prefix_count(
                [value[1] for value in ordered],
                fraction * stratum_tokens,
            )
        chosen = ordered[:count]
        chosen_tokens = sum(value[1] for value in chosen)
        selected.extend(value[0].record_id for value in chosen)
        selected_tokens += chosen_tokens
        strata.append(
            {
                "name": name,
                "universe_records": len(ordered),
                "universe_retained_tokens": stratum_tokens,
                "selected_records": len(chosen),
                "selected_retained_tokens": chosen_tokens,
                "fraction_realized": (
                    chosen_tokens / stratum_tokens
                    if stratum_tokens
                    else 0.0
                ),
            }
        )
    return RetainedBudgetSelection(
        fraction_requested=fraction,
        universe_records=len(records),
        universe_retained_tokens=total_tokens,
        selected_record_ids=tuple(sorted(selected)),
        selected_retained_tokens=selected_tokens,
        strata=tuple(strata),
    )


def build_lookup_requests(
    records: Sequence[XPBaselineRecord],
    route: str,
    selected_record_ids: Sequence[int] | None = None,
) -> tuple[LookupRequest, ...]:
    if route not in {"retained", "append"}:
        raise ValueError("XP M2 lookup route differs")
    selected = (
        None
        if selected_record_ids is None
        else {int(value) for value in selected_record_ids}
    )
    requests = []
    for record in records:
        if route == "retained" and (
            selected is not None and record.record_id not in selected
        ):
            continue
        retained, append = record_extents(record)
        values = retained if route == "retained" else append
        requests.append(
            LookupRequest(
                record_id=record.record_id,
                owner_rank=record.owner_rank,
                item_ids=values.copy(),
            )
        )
    if route == "retained" and selected is not None and {
        value.record_id for value in requests
    } != selected:
        raise ValueError("XP M2 selected records are outside universe")
    return tuple(requests)


def account_lookup_requests(
    requests: Sequence[LookupRequest],
    *,
    world_size: int,
    hidden_size: int,
) -> dict[str, object]:
    if world_size < 1 or hidden_size < 1:
        raise ValueError("XP M2 accounting geometry differs")
    by_rank = []
    for rank in range(world_size):
        local = [value for value in requests if value.owner_rank == rank]
        requested = sum(len(value.item_ids) for value in local)
        remote = sum(
            int(
                np.count_nonzero(
                    value.item_ids.astype(np.int64, copy=False)
                    % world_size
                    != rank
                )
            )
            for value in local
        )
        by_rank.append(
            {
                "rank": rank,
                "records": len(local),
                "requested_tokens": requested,
                "remote_tokens": remote,
                "id_request_bytes": remote * torch.int64.itemsize,
                "h1536_fp32_response_bytes": (
                    remote * hidden_size * torch.float32.itemsize
                ),
            }
        )
    requested_values = [
        int(value["requested_tokens"]) for value in by_rank
    ]
    remote_values = [int(value["remote_tokens"]) for value in by_rank]

    def imbalance(values: Sequence[int]) -> dict[str, float | int]:
        mean = sum(values) / len(values)
        return {
            "minimum": min(values),
            "maximum": max(values),
            "mean": mean,
            "max_over_mean": max(values) / mean if mean else 1.0,
        }

    return {
        "records": len(requests),
        "requested_tokens": sum(requested_values),
        "remote_tokens": sum(remote_values),
        "id_request_bytes": sum(
            int(value["id_request_bytes"]) for value in by_rank
        ),
        "h1536_fp32_response_bytes": sum(
            int(value["h1536_fp32_response_bytes"])
            for value in by_rank
        ),
        "per_rank": by_rank,
        "rank_imbalance": {
            "requested_tokens": imbalance(requested_values),
            "remote_tokens": imbalance(remote_values),
        },
    }


def _checkpoint_artifact(
    directory: Path,
    descriptor: Mapping[str, object],
    *,
    verify_hash: bool,
) -> Path:
    path = directory / str(descriptor["path"])
    if not path.is_file() or path.stat().st_size != int(descriptor["bytes"]):
        raise ValueError("XP M2 checkpoint artifact differs")
    if verify_hash and file_sha256(path) != str(descriptor["sha256"]):
        raise ValueError("XP M2 checkpoint artifact hash differs")
    return path


def load_lookup_checkpoint(
    inputs: XPBaselineInputs,
    checkpoint_root: str | Path,
    checkpoint_version: int,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    verify_hash: bool,
    copy_chunk_rows: int = 8_192,
) -> tuple[TrainableProjectedModuloEmbedding, dict[str, object]]:
    directory = Path(checkpoint_root) / f"theta_{checkpoint_version}"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != XP_PROJECTED_CHECKPOINT_SCHEMA
        or int(manifest.get("version", -1)) != checkpoint_version
        or int(manifest.get("world_size", -1)) != world_size
        or manifest.get("spec") != asdict(inputs.spec)
    ):
        raise ValueError("XP M2 checkpoint manifest differs")
    shards = manifest.get("embedding_shards")
    if (
        not isinstance(shards, list)
        or len(shards) != world_size
        or int(shards[rank]["rank"]) != rank
    ):
        raise ValueError("XP M2 checkpoint shard binding differs")
    projection_path = _checkpoint_artifact(
        directory,
        manifest["projection"],
        verify_hash=verify_hash,
    )
    shard_path = _checkpoint_artifact(
        directory,
        shards[rank],
        verify_hash=verify_hash,
    )
    projection_payload = torch.load(
        projection_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    shard_payload = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    source = shard_payload["local_weight"]
    expected_rows = modulo_local_rows(
        inputs.spec.num_embeddings,
        rank,
        world_size,
    )
    if source.shape != (expected_rows, inputs.spec.embedding_width):
        raise ValueError("XP M2 checkpoint local embedding shape differs")
    local_weight = torch.empty(
        source.shape,
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, expected_rows, copy_chunk_rows):
        stop = min(start + copy_chunk_rows, expected_rows)
        local_weight[start:stop].copy_(source[start:stop])
    projection = projection_payload["projection_weight"].to(device)
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=local_weight,
        projection_weight=projection,
        num_embeddings=inputs.spec.num_embeddings,
        rank=rank,
        world_size=world_size,
    )
    embedding.requires_grad_(False)
    embedding.eval()
    del source, shard_payload, projection_payload
    gc.collect()
    return embedding, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "version": checkpoint_version,
        "world_size": world_size,
        "projection": dict(manifest["projection"]),
        "embedding_shards": [dict(value) for value in shards],
        "local_embedding_shard": dict(shards[rank]),
        "artifact_hashes_verified": verify_hash,
        "provenance": manifest.get("provenance"),
    }


def _device_batches(
    requests: Sequence[LookupRequest],
    *,
    rank: int,
    world_size: int,
    micro_batch_records: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if micro_batch_records < 1:
        raise ValueError("XP M2 micro batch differs")
    by_rank = [
        [value for value in requests if value.owner_rank == candidate]
        for candidate in range(world_size)
    ]
    steps = max(
        math.ceil(len(values) / micro_batch_records)
        for values in by_rank
    )
    local = by_rank[rank]
    batches = []
    for step in range(steps):
        begin = step * micro_batch_records
        selected = local[begin : begin + micro_batch_records]
        width = max((len(value.item_ids) for value in selected), default=1)
        item_ids = torch.zeros(
            (len(selected), width),
            dtype=torch.int64,
            device=device,
        )
        lengths = torch.empty(
            len(selected),
            dtype=torch.int64,
            device=device,
        )
        for row, request in enumerate(selected):
            length = len(request.item_ids)
            item_ids[row, :length] = torch.from_numpy(
                request.item_ids.astype(np.int64, copy=False)
            ).to(device)
            lengths[row] = length
        batches.append((item_ids, lengths))
    return tuple(batches)


@torch.inference_mode()
def _execute_batches(
    embedding: TrainableProjectedModuloEmbedding,
    batch_groups: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:
    checksum = torch.zeros(
        (),
        dtype=torch.float32,
        device=embedding.local_weight.device,
    )
    for batches in batch_groups:
        for item_ids, lengths in batches:
            output = embedding(item_ids, lengths)
            checksum = checksum + output[..., 0].sum()
    return checksum


def _measure(
    embedding: TrainableProjectedModuloEmbedding,
    batch_groups: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]],
    *,
    rank: int,
    world_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, object] | None:
    device = embedding.local_weight.device
    for _ in range(warmup):
        if world_size > 1:
            dist.barrier()
        _execute_batches(embedding, batch_groups)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    gathered_repeats = []
    for repeat in range(repeats):
        if world_size > 1:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        checksum = _execute_batches(embedding, batch_groups)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        local = {
            "rank": rank,
            "seconds": elapsed,
            "checksum": float(checksum.item()),
        }
        reports: list[object] = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(reports, local)
        else:
            reports[0] = local
        if rank == 0:
            resolved = [
                value for value in reports if isinstance(value, dict)
            ]
            if len(resolved) != world_size:
                raise RuntimeError("XP M2 timing rank reports differ")
            gathered_repeats.append(
                {
                    "repeat": repeat,
                    "max_rank_seconds": max(
                        float(value["seconds"]) for value in resolved
                    ),
                    "min_rank_seconds": min(
                        float(value["seconds"]) for value in resolved
                    ),
                    "max_over_min_rank_seconds": (
                        max(float(value["seconds"]) for value in resolved)
                        / min(float(value["seconds"]) for value in resolved)
                    ),
                    "per_rank": resolved,
                }
            )
    if rank != 0:
        return None
    maxima = [
        float(value["max_rank_seconds"])
        for value in gathered_repeats
    ]
    time_imbalance = [
        float(value["max_over_min_rank_seconds"])
        for value in gathered_repeats
    ]
    return {
        "warmup_repeats": warmup,
        "measured_repeats": repeats,
        "primary_statistic": "median max-rank wall seconds",
        "median_max_rank_seconds": statistics.median(maxima),
        "minimum_max_rank_seconds": min(maxima),
        "maximum_max_rank_seconds": max(maxima),
        "median_max_over_min_rank_seconds": statistics.median(
            time_imbalance
        ),
        "raw_repeats": gathered_repeats,
    }


def _phase_result(
    name: str,
    requests: Sequence[LookupRequest],
    batch_groups: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]],
    timing: dict[str, object] | None,
    *,
    world_size: int,
    hidden_size: int,
) -> dict[str, object] | None:
    if timing is None:
        return None
    calls = sum(len(value) for value in batch_groups)
    accounting = account_lookup_requests(
        requests,
        world_size=world_size,
        hidden_size=hidden_size,
    )
    accounting.update(
        {
            "name": name,
            "lookup_calls_per_rank": calls,
            "all_to_all_collective_invocations_per_rank": 3 * calls,
            "all_to_all_collective_invocations_all_ranks": (
                3 * calls * world_size
            ),
            "timing": timing,
        }
    )
    return accounting


def run_lookup_communication_baseline(
    inputs: XPBaselineInputs,
    embedding: TrainableProjectedModuloEmbedding,
    *,
    fractions: Sequence[float],
    rank: int,
    world_size: int,
    micro_batch_records: int,
    warmup: int,
    repeats: int,
    checkpoint_binding: Mapping[str, object],
) -> dict[str, object] | None:
    if (
        world_size != 2
        or embedding.world_size != world_size
        or embedding.rank != rank
        or warmup < 0
        or repeats < 1
        or not fractions
    ):
        raise ValueError("XP M2 run configuration differs")
    benchmark_hash = str(
        inputs.bindings["benchmark_config"]["sha256"]
    )
    checkpoint_reports: list[object] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(checkpoint_reports, dict(checkpoint_binding))
    else:
        checkpoint_reports[0] = dict(checkpoint_binding)
    if rank == 0:
        resolved_checkpoints = [
            value
            for value in checkpoint_reports
            if isinstance(value, Mapping)
        ]
        if (
            len(resolved_checkpoints) != world_size
            or len(
                {
                    str(value["manifest_sha256"])
                    for value in resolved_checkpoints
                }
            )
            != 1
        ):
            raise RuntimeError("XP M2 checkpoint rank bindings differ")
    append_requests = build_lookup_requests(inputs.records, "append")
    append_batches = _device_batches(
        append_requests,
        rank=rank,
        world_size=world_size,
        micro_batch_records=micro_batch_records,
        device=embedding.local_weight.device,
    )
    append_timing = _measure(
        embedding,
        (append_batches,),
        rank=rank,
        world_size=world_size,
        warmup=warmup,
        repeats=repeats,
    )
    append_result = _phase_result(
        "append_only_common_to_every_record",
        append_requests,
        (append_batches,),
        append_timing,
        world_size=world_size,
        hidden_size=inputs.spec.hidden_size,
    )
    cells = []
    for fraction in fractions:
        selection = select_retained_budget(
            inputs.records,
            float(fraction),
            benchmark_hash,
        )
        retained_requests = build_lookup_requests(
            inputs.records,
            "retained",
            selection.selected_record_ids,
        )
        retained_batches = _device_batches(
            retained_requests,
            rank=rank,
            world_size=world_size,
            micro_batch_records=micro_batch_records,
            device=embedding.local_weight.device,
        )
        retained_timing = _measure(
            embedding,
            (retained_batches,),
            rank=rank,
            world_size=world_size,
            warmup=warmup,
            repeats=repeats,
        )
        complete_timing = _measure(
            embedding,
            (retained_batches, append_batches),
            rank=rank,
            world_size=world_size,
            warmup=warmup,
            repeats=repeats,
        )
        complete_requests = tuple(retained_requests) + tuple(
            append_requests
        )
        if rank == 0:
            retained_result = _phase_result(
                "retained_only_exact_selected",
                retained_requests,
                (retained_batches,),
                retained_timing,
                world_size=world_size,
                hidden_size=inputs.spec.hidden_size,
            )
            complete_result = _phase_result(
                "complete_wave_retained_exact_plus_common_append",
                complete_requests,
                (retained_batches, append_batches),
                complete_timing,
                world_size=world_size,
                hidden_size=inputs.spec.hidden_size,
            )
            cells.append(
                {
                    "retained_budget": selection.to_dict(),
                    "retained_only": retained_result,
                    "append_only": append_result,
                    "complete_wave": complete_result,
                }
            )
    if rank != 0:
        return None
    all_exact_tokens = sum(
        record.target_length for record in inputs.records
    )
    return {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "method": "append_aware_row_sharded_embedding_lookup_characterization",
        "claim_boundary": (
            "development foundation for embedding lookup communication only; "
            "it executes the real projected modulo all-to-all but excludes "
            "HSTU dense recomputation, K/V materialization, D1 compiled repair, "
            "D2 lowering, D3 storage, and recommendation quality"
        ),
        "benchmark_id": inputs.benchmark["benchmark_id"],
        "capacity_name": inputs.capacity_name,
        "records": len(inputs.records),
        "world_size": world_size,
        "micro_batch_records": micro_batch_records,
        "fractions": [float(value) for value in fractions],
        "append_tokens_per_record": APPEND_TOKENS,
        "all_exact_target_lookup_tokens": all_exact_tokens,
        "selection_policy": {
            "label_free": True,
            "item_identity_free": True,
            "nested_stable_hash_order": True,
            "budget_unit": "retained valid tokens",
            "shape_strata": [
                {
                    "name": name,
                    "minimum_inclusive": lower,
                    "maximum_exclusive": upper,
                }
                for name, lower, upper in RETAINED_STRATA
            ],
            "per_stratum_rounding": (
                "stable-hash prefix whose retained-token sum is nearest "
                "the requested stratum budget; ties prefer fewer tokens"
            ),
        },
        "measurement_boundary": {
            "inside_timer": (
                "three real all-to-all collectives per projected lookup "
                "call, owner-side E4096-to-H1536 projection, local lookup, "
                "and output completion"
            ),
            "outside_timer": (
                "checkpoint loading and verification, deterministic action "
                "selection, HET extent extraction, ID packing, HBM staging, "
                "barriers, and rank-report collection"
            ),
            "response_payload": "owner-projected H1536 FP32 vectors",
            "request_payload": "remote int64 row IDs",
        },
        "bindings": inputs.bindings,
        "checkpoint": {
            "manifest_path": checkpoint_binding["manifest_path"],
            "manifest_sha256": checkpoint_binding["manifest_sha256"],
            "version": checkpoint_binding["version"],
            "world_size": checkpoint_binding["world_size"],
            "projection": checkpoint_binding["projection"],
            "embedding_shards": checkpoint_binding["embedding_shards"],
            "artifact_hashes_verified_on_owning_rank": all(
                bool(value["artifact_hashes_verified"])
                for value in resolved_checkpoints
            ),
            "rank_reports": resolved_checkpoints,
            "provenance": checkpoint_binding["provenance"],
        },
        "append_only_common": append_result,
        "cells": cells,
    }
