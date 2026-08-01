from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..models import HSTUKVCache
from ..streaming.sharded_edge import (
    ExternalEmbeddingHSTU,
    modulo_local_rows,
)
from ..streaming.xp_projected_edge import (
    XP_PROJECTED_CHECKPOINT_SCHEMA,
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
)
from .design3_store import PageableDramExtentStore

PROTOCOL = "evokv_xp_fixed_baselines_development_v0"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class XPBaselineRecord:
    record_id: int
    user_id: int
    owner_rank: int
    item_ids: np.ndarray
    old_start: int
    old_length: int
    target_start: int
    target_length: int
    old_valid_bytes: int
    target_valid_bytes: int

    def __post_init__(self) -> None:
        if (
            self.record_id < 0
            or self.user_id < 0
            or self.owner_rank < 0
            or self.item_ids.ndim != 1
            or self.item_ids.dtype.kind not in {"i", "u"}
            or self.old_start < 0
            or self.old_length < 1
            or self.target_start < 0
            or self.target_length < 1
            or self.old_start + self.old_length > len(self.item_ids)
            or self.target_start + self.target_length > len(self.item_ids)
            or self.old_valid_bytes < 1
            or self.target_valid_bytes < 1
        ):
            raise ValueError("XP baseline record differs")

    def items(self, endpoint: str) -> np.ndarray:
        if endpoint == "old":
            start = self.old_start
            length = self.old_length
        elif endpoint == "target":
            start = self.target_start
            length = self.target_length
        else:
            raise ValueError("XP baseline endpoint differs")
        return self.item_ids[start : start + length]

    def length(self, endpoint: str) -> int:
        if endpoint == "old":
            return self.old_length
        if endpoint == "target":
            return self.target_length
        raise ValueError("XP baseline endpoint differs")


@dataclass(frozen=True)
class XPBaselineGroup:
    ordinal: int
    record_ids_by_rank: tuple[tuple[int, ...], ...]
    target_valid_bytes: int

    def __post_init__(self) -> None:
        flattened = tuple(
            record_id
            for values in self.record_ids_by_rank
            for record_id in values
        )
        if (
            self.ordinal < 0
            or not self.record_ids_by_rank
            or not flattened
            or len(flattened) != len(set(flattened))
            or self.target_valid_bytes < 1
        ):
            raise ValueError("XP baseline group differs")

    @property
    def record_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                record_id
                for values in self.record_ids_by_rank
                for record_id in values
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "record_ids_by_rank": [
                list(values) for values in self.record_ids_by_rank
            ],
            "target_valid_bytes": self.target_valid_bytes,
        }


@dataclass(frozen=True)
class XPBaselineInputs:
    benchmark: dict[str, object]
    bindings: dict[str, object]
    spec: XPProjectedModelSpec
    records: tuple[XPBaselineRecord, ...]
    capacity_name: str
    capacity_prefix_records: int


@dataclass
class XPHostSlot:
    item_ids: torch.Tensor
    behaviors: torch.Tensor
    time_deltas: torch.Tensor
    lengths: torch.Tensor
    output_k: torch.Tensor
    output_v: torch.Tensor
    group: XPBaselineGroup | None = None
    local_records: tuple[XPBaselineRecord, ...] = ()
    endpoint: str = ""
    valid_output_tokens: int = 0
    d2h_event: torch.cuda.Event | None = None
    device_references: tuple[torch.Tensor, ...] = ()

    @property
    def nbytes(self) -> int:
        storages = {
            value.untyped_storage().data_ptr(): (
                value.untyped_storage().nbytes()
            )
            for value in (
                self.item_ids,
                self.behaviors,
                self.time_deltas,
                self.lengths,
                self.output_k,
                self.output_v,
            )
        }
        return sum(storages.values())


class XPRollingJournal:
    def __init__(
        self,
        path: Path,
        *,
        mode: str,
        binding: Mapping[str, object],
        source_manifest: Mapping[str, object] | None,
    ) -> None:
        self.path = path
        self.mode = mode
        self.binding = json.loads(
            json.dumps(binding, sort_keys=True)
        )
        if mode == "create":
            if path.exists():
                raise FileExistsError(
                    f"XP rolling journal already exists: {path}"
                )
            self.state = {
                "schema": "evokv_xp_rolling_store_v0",
                "phase": "source_materializing",
                "binding": self.binding,
                "source": None,
                "target_commits": [],
            }
            self._write()
        elif mode == "open":
            state = _load_json(path)
            if (
                state.get("schema") != "evokv_xp_rolling_store_v0"
                or state.get("phase") != "source_complete"
                or state.get("binding") != self.binding
                or source_manifest is None
                or state.get("source_manifest_sha256")
                != canonical_sha256(source_manifest)
            ):
                raise ValueError("XP rolling source journal differs")
            self.state = state
            self.state["phase"] = "target_running"
            self.state["target_commits"] = []
            self._write()
        else:
            raise ValueError("XP rolling journal mode differs")

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, self.path)

    def commit_group(
        self,
        receipt: Mapping[str, object],
    ) -> float:
        started = time.perf_counter()
        commits = self.state["target_commits"]
        if not isinstance(commits, list):
            raise RuntimeError("XP rolling journal commits differ")
        ordinal = int(receipt["group_ordinal"])
        if any(
            int(value["group_ordinal"]) == ordinal
            for value in commits
        ):
            raise ValueError("XP rolling group committed twice")
        commits.append(json.loads(json.dumps(receipt, sort_keys=True)))
        self._write()
        return time.perf_counter() - started

    def finalize_source(
        self,
        *,
        local_report: Mapping[str, object],
        source_manifest: Mapping[str, object],
    ) -> None:
        if self.mode != "create":
            raise ValueError("XP rolling source finalization differs")
        self.state["phase"] = "source_complete"
        self.state["source"] = {
            "rank": int(local_report["rank"]),
            "records": int(local_report["records"]),
            "record_ids_sha256": local_report[
                "record_ids_sha256"
            ],
            "record_hashes_sha256": canonical_sha256(
                local_report["record_hashes"]
            ),
            "store": local_report["store"],
        }
        self.state["source_manifest_sha256"] = canonical_sha256(
            source_manifest
        )
        self.state["target_commits"] = []
        self._write()

    def finalize_target(
        self,
        *,
        expected_groups: int,
        expected_record_ids: Sequence[int],
    ) -> dict[str, object]:
        if self.mode != "open":
            raise ValueError("XP rolling target finalization differs")
        commits = self.state["target_commits"]
        if not isinstance(commits, list):
            raise RuntimeError("XP rolling target commits differ")
        ordered = sorted(
            commits,
            key=lambda value: int(value["group_ordinal"]),
        )
        observed_ids = sorted(
            int(record_id)
            for value in ordered
            for record_id in value["record_ids"]
        )
        if (
            [int(value["group_ordinal"]) for value in ordered]
            != list(range(expected_groups))
            or observed_ids
            != sorted(int(value) for value in expected_record_ids)
        ):
            raise RuntimeError(
                "XP rolling commit coverage differs"
            )
        self.state["phase"] = "target_complete"
        self.state["target_record_ids_sha256"] = (
            _rank_record_ids_sha256(observed_ids)
        )
        self._write()
        return {
            "path": str(self.path),
            "sha256": file_sha256(self.path),
            "groups_committed": len(ordered),
            "records_committed": len(observed_ids),
            "record_ids_sha256": _rank_record_ids_sha256(
                observed_ids
            ),
        }


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolve_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _validate_declared_hash(
    root: Path,
    descriptor: Mapping[str, object],
    *,
    path_key: str = "path",
    hash_key: str = "sha256",
) -> tuple[Path, str]:
    path = _resolve_path(root, descriptor[path_key])
    observed = file_sha256(path)
    declared = descriptor.get(hash_key)
    if declared is not None and str(declared) != observed:
        raise ValueError(f"artifact hash differs: {path}")
    return path, observed


def _spec_from_config(value: Mapping[str, object]) -> XPProjectedModelSpec:
    model = value["model"]
    if not isinstance(model, Mapping):
        raise ValueError("XP benchmark model is missing")
    data = value["data"]
    if not isinstance(data, Mapping):
        raise ValueError("XP benchmark data is missing")
    catalog = data["catalog"]
    if not isinstance(catalog, Mapping):
        raise ValueError("XP benchmark catalog is missing")
    num_embeddings = int(catalog["physical_rows"])
    return XPProjectedModelSpec(
        num_embeddings=num_embeddings,
        embedding_width=int(model["embedding_width"]),
        hidden_size=int(model["hidden_size"]),
        num_prediction_items=int(
            catalog.get(
                "prediction_rows",
                min(250_000, num_embeddings - 1),
            )
        ),
        num_behaviors=int(model["num_behaviors"]),
        num_layers=int(model["layers"]),
        num_heads=int(model["heads"]),
        head_dim=int(model["head_dim"]),
        max_seq_len=int(model["maximum_context"]),
    )


def _capacity_descriptor(
    benchmark: Mapping[str, object],
    capacity_name: str,
) -> tuple[int, int]:
    points = benchmark["capacity_points"]
    if not isinstance(points, Mapping):
        raise ValueError("XP benchmark capacity points are missing")
    if capacity_name == "resident_m2":
        point = points["resident_m2"]
        if not isinstance(point, Mapping):
            raise ValueError("XP resident capacity differs")
        return int(point["prefix_records"]), int(
            point["target_valid_bytes"]
        )
    candidates = points["out_of_core_primary"]
    if not isinstance(candidates, list):
        raise ValueError("XP out-of-core capacities differ")
    for point in candidates:
        if (
            isinstance(point, Mapping)
            and str(point["single_version_target_gib_nominal"])
            == capacity_name
        ):
            return int(point["prefix_records"]), (
                int(float(capacity_name) * (1 << 30))
            )
    raise ValueError(f"unknown XP capacity point: {capacity_name}")


def load_fixed_inputs(
    config_path: str | Path,
    capacity_name: str,
    *,
    world_size: int,
    record_limit: int | None = None,
) -> XPBaselineInputs:
    if world_size not in {1, 2, 4}:
        raise ValueError("XP baseline world size must be 1, 2, or 4")
    path = Path(config_path).resolve()
    root = path.parents[2]
    benchmark = _load_json(path)
    hardware = benchmark.get("hardware")
    if not isinstance(hardware, Mapping) or world_size not in tuple(
        int(value) for value in hardware["world_sizes_supported_by_code"]
    ):
        raise ValueError("XP baseline world size is not admitted")
    data = benchmark["data"]
    if not isinstance(data, Mapping):
        raise ValueError("XP baseline data descriptor differs")
    workload_descriptor = data["het_workload"]
    roles_descriptor = data["roles"]
    edge_descriptor = data.get("fixed_edge_inputs")
    if (
        not isinstance(workload_descriptor, Mapping)
        or not isinstance(roles_descriptor, Mapping)
    ):
        raise ValueError("XP baseline workload binding differs")
    workload_path, workload_hash = _validate_declared_hash(
        root,
        workload_descriptor,
    )
    roles_path, roles_hash = _validate_declared_hash(
        root,
        roles_descriptor,
    )
    edge_path = (
        _resolve_path(
            root,
            edge_descriptor["path"],
        )
        if isinstance(edge_descriptor, Mapping)
        else root
        / "data/processed/evokv_foundation/qk_xp_fixed_edge_inputs.npz"
    )
    edge_hash = file_sha256(edge_path)
    if (
        isinstance(edge_descriptor, Mapping)
        and edge_descriptor.get("sha256") is not None
        and str(edge_descriptor["sha256"]) != edge_hash
    ):
        raise ValueError("XP fixed edge input hash differs")
    prefix_records, nominal_bytes = _capacity_descriptor(
        benchmark,
        capacity_name,
    )
    if record_limit is not None:
        if record_limit < 1:
            raise ValueError("XP record limit must be positive")
        prefix_records = min(prefix_records, record_limit)
    with np.load(workload_path, allow_pickle=False) as source:
        required = {
            "record_user_ids",
            "history_offsets",
            "history_item_idx",
            "old_start",
            "old_length",
            "target_start",
            "target_length",
            "het_old_valid_kv_bytes",
            "het_target_valid_kv_bytes",
            f"owner_rank_{world_size}",
        }
        missing = required.difference(source.files)
        if missing:
            raise ValueError(
                f"XP HET workload lacks arrays: {sorted(missing)}"
            )
        users = source["record_user_ids"][:prefix_records].copy()
        offsets = source["history_offsets"][: prefix_records + 1].copy()
        items = source["history_item_idx"][: int(offsets[-1])].copy()
        old_start = source["old_start"][:prefix_records].copy()
        old_length = source["old_length"][:prefix_records].copy()
        target_start = source["target_start"][:prefix_records].copy()
        target_length = source["target_length"][:prefix_records].copy()
        old_bytes = source[
            "het_old_valid_kv_bytes"
        ][:prefix_records].copy()
        target_bytes = source[
            "het_target_valid_kv_bytes"
        ][:prefix_records].copy()
        owners = source[
            f"owner_rank_{world_size}"
        ][:prefix_records].copy()
        metadata = json.loads(str(source["metadata_json"].item()))
    spec = _spec_from_config(benchmark)
    bytes_per_token = (
        2
        * spec.num_layers
        * spec.hidden_size
        * torch.float16.itemsize
    )
    records = []
    for record_id in range(prefix_records):
        start = int(offsets[record_id])
        stop = int(offsets[record_id + 1])
        record = XPBaselineRecord(
            record_id=record_id,
            user_id=int(users[record_id]),
            owner_rank=int(owners[record_id]),
            item_ids=np.asarray(items[start:stop], dtype=np.int64),
            old_start=int(old_start[record_id]),
            old_length=int(old_length[record_id]),
            target_start=int(target_start[record_id]),
            target_length=int(target_length[record_id]),
            old_valid_bytes=int(old_bytes[record_id]),
            target_valid_bytes=int(target_bytes[record_id]),
        )
        if (
            record.owner_rank >= world_size
            or record.old_valid_bytes
            != record.old_length * bytes_per_token
            or record.target_valid_bytes
            != record.target_length * bytes_per_token
        ):
            raise ValueError("XP HET extent byte ledger differs")
        records.append(record)
    actual_target_bytes = sum(value.target_valid_bytes for value in records)
    if record_limit is None and capacity_name == "resident_m2":
        declared = benchmark["capacity_points"]["resident_m2"]
        if int(declared["target_valid_bytes"]) != actual_target_bytes:
            raise ValueError("XP resident capacity boundary differs")
    if (
        record_limit is None
        and capacity_name != "resident_m2"
        and actual_target_bytes < nominal_bytes
    ):
        raise ValueError("XP out-of-core capacity boundary was not reached")
    bindings = {
        "benchmark_config": {
            "path": str(path),
            "sha256": file_sha256(path),
        },
        "het_workload": {
            "path": str(workload_path),
            "sha256": workload_hash,
            "content_sha256": metadata.get("content_sha256"),
        },
        "fixed_edge_inputs": {
            "path": str(edge_path),
            "sha256": edge_hash,
        },
        "roles": {
            "path": str(roles_path),
            "sha256": roles_hash,
        },
    }
    return XPBaselineInputs(
        benchmark=benchmark,
        bindings=bindings,
        spec=spec,
        records=tuple(records),
        capacity_name=capacity_name,
        capacity_prefix_records=prefix_records,
    )


def select_partial_exact(
    records: Sequence[XPBaselineRecord],
    fraction: float,
    benchmark_sha256: str,
) -> tuple[XPBaselineRecord, ...]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("XP exact fraction must lie in [0, 1]")
    count = round(len(records) * fraction)
    ordered = sorted(
        records,
        key=lambda record: (
            hashlib.sha256(
                f"{benchmark_sha256}:{record.record_id}".encode()
            ).digest(),
            record.record_id,
        ),
    )
    return tuple(
        sorted(ordered[:count], key=lambda record: record.record_id)
    )


def ordinal_inter_event_time_deltas(
    record: XPBaselineRecord,
    endpoint: str,
) -> torch.Tensor:
    if endpoint == "old":
        start = record.old_start
        length = record.old_length
    elif endpoint == "target":
        start = record.target_start
        length = record.target_length
    else:
        raise ValueError("XP baseline endpoint differs")
    values = torch.ones(length, dtype=torch.float32)
    if start == 0:
        values[0] = 0.0
    return values


def build_groups(
    records: Sequence[XPBaselineRecord],
    *,
    world_size: int,
    group_target_bytes: int,
) -> tuple[XPBaselineGroup, ...]:
    if (
        not records
        or world_size < 1
        or group_target_bytes < 1
        or any(value.owner_rank >= world_size for value in records)
    ):
        raise ValueError("XP baseline grouping inputs differ")
    groups = []
    current: list[XPBaselineRecord] = []
    current_bytes = 0
    for record in records:
        if (
            current
            and current_bytes + record.target_valid_bytes
            > group_target_bytes
        ):
            by_rank = tuple(
                tuple(
                    value.record_id
                    for value in current
                    if value.owner_rank == rank
                )
                for rank in range(world_size)
            )
            groups.append(
                XPBaselineGroup(
                    ordinal=len(groups),
                    record_ids_by_rank=by_rank,
                    target_valid_bytes=current_bytes,
                )
            )
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += record.target_valid_bytes
    if current:
        groups.append(
            XPBaselineGroup(
                ordinal=len(groups),
                record_ids_by_rank=tuple(
                    tuple(
                        value.record_id
                        for value in current
                        if value.owner_rank == rank
                    )
                    for rank in range(world_size)
                ),
                target_valid_bytes=current_bytes,
            )
        )
    return tuple(groups)


def group_plan_sha256(groups: Sequence[XPBaselineGroup]) -> str:
    return canonical_sha256(
        {"groups": [value.to_dict() for value in groups]}
    )


def _checkpoint_artifact(
    directory: Path,
    descriptor: Mapping[str, object],
) -> Path:
    path = directory / str(descriptor["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(descriptor["bytes"])
        or file_sha256(path) != str(descriptor["sha256"])
    ):
        raise ValueError("XP checkpoint artifact differs")
    return path


def load_inference_checkpoint(
    root: str | Path,
    version: int,
    spec: XPProjectedModelSpec,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    process_group: dist.ProcessGroup | None = None,
    copy_chunk_rows: int = 8_192,
) -> tuple[
    ExternalEmbeddingHSTU,
    TrainableProjectedModuloEmbedding,
    dict[str, object],
]:
    if copy_chunk_rows < 1:
        raise ValueError("XP checkpoint copy chunk must be positive")
    directory = Path(root) / f"theta_{version}"
    manifest_path = directory / "manifest.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != XP_PROJECTED_CHECKPOINT_SCHEMA
        or int(manifest.get("version", -1)) != version
        or int(manifest.get("world_size", -1)) != world_size
        or manifest.get("spec") != asdict(spec)
    ):
        raise ValueError("XP inference checkpoint binding differs")
    dense_path = _checkpoint_artifact(
        directory,
        manifest["dense"],
    )
    projection_path = _checkpoint_artifact(
        directory,
        manifest["projection"],
    )
    shard_records = manifest["embedding_shards"]
    if (
        not isinstance(shard_records, list)
        or len(shard_records) != world_size
        or int(shard_records[rank]["rank"]) != rank
    ):
        raise ValueError("XP inference shard descriptor differs")
    shard_path = _checkpoint_artifact(
        directory,
        shard_records[rank],
    )
    dense_payload = torch.load(
        dense_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
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
    dense = ExternalEmbeddingHSTU(spec.hstu_config())
    dense.load_state_dict(dense_payload["state_dict"], strict=True)
    dense.to(device)
    dense.eval()
    local_rows = modulo_local_rows(
        spec.num_embeddings,
        rank,
        world_size,
    )
    source_weight = shard_payload["local_weight"]
    if source_weight.shape != (
        local_rows,
        spec.embedding_width,
    ):
        raise ValueError("XP inference embedding shard shape differs")
    local_weight = torch.empty(
        source_weight.shape,
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, local_rows, copy_chunk_rows):
        stop = min(start + copy_chunk_rows, local_rows)
        local_weight[start:stop].copy_(
            source_weight[start:stop],
            non_blocking=False,
        )
    projection_weight = projection_payload[
        "projection_weight"
    ].to(device)
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=local_weight,
        projection_weight=projection_weight,
        num_embeddings=spec.num_embeddings,
        rank=rank,
        world_size=world_size,
        process_group=process_group,
    )
    embedding.eval()
    return dense, embedding, {
        "path": str(manifest_path),
        "sha256": file_sha256(manifest_path),
        "version": version,
        "provenance": manifest.get("provenance"),
        "optimizer_active_rows": manifest.get(
            "optimizer_active_rows"
        ),
    }


def release_inference_checkpoint(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
) -> None:
    del dense, embedding
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _allocate_host_slot(
    records: Sequence[XPBaselineRecord],
    groups: Sequence[XPBaselineGroup],
    *,
    rank: int,
    spec: XPProjectedModelSpec,
    pin_memory: bool,
) -> XPHostSlot:
    by_id = {value.record_id: value for value in records}
    max_rows = 1
    max_width = 1
    max_tokens = 1
    for group in groups:
        local = tuple(
            by_id[record_id]
            for record_id in group.record_ids_by_rank[rank]
        )
        max_rows = max(max_rows, len(local))
        max_width = max(
            max_width,
            max((value.target_length for value in local), default=1),
            max((value.old_length for value in local), default=1),
        )
        max_tokens = max(
            max_tokens,
            sum(value.target_length for value in local),
            sum(value.old_length for value in local),
        )
    return XPHostSlot(
        item_ids=torch.empty(
            (max_rows, max_width),
            dtype=torch.int64,
            pin_memory=pin_memory,
        ),
        behaviors=torch.empty(
            (max_rows, max_width),
            dtype=torch.int64,
            pin_memory=pin_memory,
        ),
        time_deltas=torch.empty(
            (max_rows, max_width),
            dtype=torch.float32,
            pin_memory=pin_memory,
        ),
        lengths=torch.empty(
            max_rows,
            dtype=torch.int64,
            pin_memory=pin_memory,
        ),
        output_k=torch.empty(
            (spec.num_layers, max_tokens, spec.hidden_size),
            dtype=torch.float16,
            pin_memory=pin_memory,
        ),
        output_v=torch.empty(
            (spec.num_layers, max_tokens, spec.hidden_size),
            dtype=torch.float16,
            pin_memory=pin_memory,
        ),
    )


def _fill_slot(
    slot: XPHostSlot,
    group: XPBaselineGroup,
    records_by_id: Mapping[int, XPBaselineRecord],
    *,
    rank: int,
    endpoint: str,
) -> float:
    started = time.perf_counter()
    local = tuple(
        records_by_id[record_id]
        for record_id in group.record_ids_by_rank[rank]
    )
    rows = len(local)
    width = max((value.length(endpoint) for value in local), default=1)
    if rows:
        slot.item_ids[:rows, :width].zero_()
        slot.behaviors[:rows, :width].zero_()
        slot.time_deltas[:rows, :width].zero_()
    for row, record in enumerate(local):
        values = record.items(endpoint)
        length = len(values)
        slot.item_ids[row, :length].copy_(
            torch.from_numpy(values.astype(np.int64, copy=False))
        )
        slot.behaviors[row, :length].fill_(1)
        slot.time_deltas[row, :length].copy_(
            ordinal_inter_event_time_deltas(record, endpoint)
        )
        slot.lengths[row] = length
    slot.group = group
    slot.local_records = local
    slot.endpoint = endpoint
    slot.valid_output_tokens = 0
    if slot.d2h_event is not None or slot.device_references:
        raise RuntimeError("XP host slot was reused before drain")
    return time.perf_counter() - started


def _pack_cache(
    slot: XPHostSlot,
    cache: HSTUKVCache,
    lengths: torch.Tensor,
    token_offset: int,
    d2h_stream: torch.cuda.Stream | None,
) -> int:
    offset = token_offset
    references = list(slot.device_references)
    if cache.k.device.type == "cuda":
        if d2h_stream is None:
            raise ValueError("XP CUDA D2H stream is absent")
        d2h_stream.wait_stream(
            torch.cuda.current_stream(cache.k.device)
        )
    for row, length_value in enumerate(lengths.tolist()):
        length = int(length_value)
        stop = offset + length
        if cache.k.device.type == "cuda":
            assert d2h_stream is not None
            with torch.cuda.stream(d2h_stream):
                k_value = cache.k[:, row, :length].to(
                    dtype=torch.float16
                )
                v_value = cache.v[:, row, :length].to(
                    dtype=torch.float16
                )
                slot.output_k[:, offset:stop].copy_(
                    k_value,
                    non_blocking=True,
                )
                slot.output_v[:, offset:stop].copy_(
                    v_value,
                    non_blocking=True,
                )
                k_value.record_stream(d2h_stream)
                v_value.record_stream(d2h_stream)
                cache.k.record_stream(d2h_stream)
                cache.v.record_stream(d2h_stream)
                references.extend(
                    (k_value, v_value, cache.k, cache.v)
                )
        else:
            slot.output_k[:, offset:stop].copy_(
                cache.k[:, row, :length].to(
                    device="cpu",
                    dtype=torch.float16,
                )
            )
            slot.output_v[:, offset:stop].copy_(
                cache.v[:, row, :length].to(
                    device="cpu",
                    dtype=torch.float16,
                )
            )
        offset = stop
    if cache.k.device.type == "cuda":
        assert d2h_stream is not None
        event = torch.cuda.Event()
        event.record(d2h_stream)
        slot.d2h_event = event
        slot.device_references = tuple(references)
    return offset


def _lookup_accounting(
    item_ids: torch.Tensor,
    lengths: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    hidden_size: int,
) -> dict[str, int]:
    requested = 0
    remote = 0
    for row, length_value in enumerate(lengths.tolist()):
        length = int(length_value)
        values = item_ids[row, :length]
        requested += length
        remote += int(
            torch.count_nonzero(
                values.remainder(world_size) != rank
            ).item()
        )
    return {
        "requested_tokens": requested,
        "remote_tokens": remote,
        "request_bytes": remote * torch.int64.itemsize,
        "response_bytes": (
            remote * hidden_size * torch.float32.itemsize
        ),
    }


def _merge_counts(
    target: dict[str, int],
    value: Mapping[str, int],
) -> None:
    for name, count in value.items():
        target[name] = target.get(name, 0) + int(count)


@torch.inference_mode()
def _compute_slot(
    slot: XPHostSlot,
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    micro_batch_records: int,
    d2h_stream: torch.cuda.Stream | None,
) -> dict[str, object]:
    if slot.group is None or not slot.endpoint:
        raise ValueError("XP exact slot is not staged")
    local_rows = len(slot.local_records)
    local_steps = math.ceil(local_rows / micro_batch_records)
    step_count = torch.tensor(
        local_steps,
        dtype=torch.int64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(step_count, op=dist.ReduceOp.MAX)
    output_offset = 0
    lookup: dict[str, int] = {}
    h2d_bytes = 0
    compute_started = time.perf_counter()
    for step in range(int(step_count.item())):
        start = step * micro_batch_records
        stop = min(start + micro_batch_records, local_rows)
        rows = stop - start
        width = max(
            (
                value.length(slot.endpoint)
                for value in slot.local_records[start:stop]
            ),
            default=1,
        )
        item_cpu = slot.item_ids[start:stop, :width]
        behavior_cpu = slot.behaviors[start:stop, :width]
        delta_cpu = slot.time_deltas[start:stop, :width]
        length_cpu = slot.lengths[start:stop]
        _merge_counts(
            lookup,
            _lookup_accounting(
                item_cpu,
                length_cpu,
                rank=rank,
                world_size=world_size,
                hidden_size=embedding.hidden_size,
            ),
        )
        h2d_bytes += (
            item_cpu.numel() * item_cpu.element_size()
            + behavior_cpu.numel() * behavior_cpu.element_size()
            + delta_cpu.numel() * delta_cpu.element_size()
            + length_cpu.numel() * length_cpu.element_size()
        )
        item_ids = item_cpu.to(device, non_blocking=True)
        behaviors = behavior_cpu.to(device, non_blocking=True)
        time_deltas = delta_cpu.to(device, non_blocking=True)
        lengths = length_cpu.to(device, non_blocking=True)
        item_vectors = embedding(item_ids, lengths)
        if rows:
            cache = dense.core.compute_kv_from_item_embeddings(
                item_vectors,
                behaviors,
                time_deltas,
                lengths=lengths,
            )
            output_offset = _pack_cache(
                slot,
                cache,
                length_cpu,
                output_offset,
                d2h_stream,
            )
    slot.valid_output_tokens = output_offset
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    return {
        "compute_seconds": time.perf_counter() - compute_started,
        "lookup": lookup,
        "h2d_bytes": h2d_bytes,
        "d2h_bytes": (
            output_offset
            * 2
            * dense.cfg.num_layers
            * dense.cfg.hidden_size
            * torch.float16.itemsize
        ),
    }


def _record_digest(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    record_id: int,
    hash_mode: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(record_id.to_bytes(8, "little"))
    digest.update(k.shape[1].to_bytes(8, "little"))
    for value in (k, v):
        if hash_mode == "full":
            for layer in range(value.shape[0]):
                digest.update(
                    value[layer].contiguous().numpy().tobytes()
                )
        elif hash_mode == "sampled":
            flat = value.reshape(-1)
            if flat.numel():
                count = min(flat.numel(), 1024)
                indices = torch.linspace(
                    0,
                    flat.numel() - 1,
                    count,
                    dtype=torch.float64,
                ).round().to(torch.int64)
                digest.update(
                    flat.index_select(0, indices)
                    .contiguous()
                    .numpy()
                    .tobytes()
                )
        else:
            raise ValueError("XP output hash mode differs")
    return digest.hexdigest()


def _validate_slot(
    slot: XPHostSlot,
    hash_mode: str,
) -> tuple[list[dict[str, object]], float]:
    started = time.perf_counter()
    if slot.d2h_event is not None:
        slot.d2h_event.synchronize()
        slot.d2h_event = None
        slot.device_references = ()
    records = []
    offset = 0
    for record in slot.local_records:
        length = record.length(slot.endpoint)
        stop = offset + length
        k = slot.output_k[:, offset:stop]
        v = slot.output_v[:, offset:stop]
        sample = torch.cat(
            [
                k.reshape(-1)[:128],
                k.reshape(-1)[-128:],
                v.reshape(-1)[:128],
                v.reshape(-1)[-128:],
            ]
        )
        finite = bool(torch.isfinite(sample).all())
        if not finite:
            raise ValueError("XP exact output contains nonfinite values")
        records.append(
            {
                "record_id": record.record_id,
                "tokens": length,
                "bytes": (
                    length
                    * 2
                    * k.shape[0]
                    * k.shape[2]
                    * k.element_size()
                ),
                "sha256": _record_digest(
                    k,
                    v,
                    record_id=record.record_id,
                    hash_mode=hash_mode,
                ),
            }
        )
        offset = stop
    if offset != slot.valid_output_tokens:
        raise ValueError("XP exact output coverage differs")
    return records, time.perf_counter() - started


def _publish_slot(
    store: PageableDramExtentStore,
    slot: XPHostSlot,
) -> tuple[int, float]:
    started = time.perf_counter()
    written = 0
    offset = 0
    for record in slot.local_records:
        length = record.length(slot.endpoint)
        stop = offset + length
        written += store.write_record(
            record.record_id,
            slot.output_k[:, offset:stop].contiguous(),
            slot.output_v[:, offset:stop].contiguous(),
        )
        offset = stop
    if offset != slot.valid_output_tokens:
        raise RuntimeError("XP publication coverage differs")
    return written, time.perf_counter() - started


def _drain_slot(
    store: PageableDramExtentStore | None,
    slot: XPHostSlot,
    hash_mode: str,
    journal: XPRollingJournal | None,
) -> dict[str, object]:
    records, validation_seconds = _validate_slot(slot, hash_mode)
    written = 0
    publication_seconds = 0.0
    if store is not None:
        written, publication_seconds = _publish_slot(store, slot)
    receipt = {
        "group_ordinal": slot.group.ordinal if slot.group else -1,
        "record_ids": [
            int(value["record_id"]) for value in records
        ],
        "record_ids_sha256": _rank_record_ids_sha256(
            [int(value["record_id"]) for value in records]
        ),
        "records": len(records),
        "validated_before_publication": True,
        "published_bytes": written,
    }
    commit_seconds = (
        journal.commit_group(receipt)
        if journal is not None
        else 0.0
    )
    return {
        "group_ordinal": slot.group.ordinal if slot.group else -1,
        "records": records,
        "validation_seconds": validation_seconds,
        "publication_seconds": publication_seconds,
        "metadata_commit_seconds": commit_seconds,
        "written_bytes": written,
        "commit_receipt": receipt,
    }


def _rank_record_ids_sha256(record_ids: Sequence[int]) -> str:
    return canonical_sha256(
        {"record_ids": sorted(int(value) for value in record_ids)}
    )


def _create_store(
    path: Path,
    records: Sequence[XPBaselineRecord],
    rank: int,
    spec: XPProjectedModelSpec,
    *,
    create: bool,
) -> PageableDramExtentStore:
    local = tuple(
        value for value in records if value.owner_rank == rank
    )
    factory = (
        PageableDramExtentStore.create
        if create
        else PageableDramExtentStore.open
    )
    return factory(
        path,
        tuple(value.record_id for value in local),
        tuple(value.target_length for value in local),
        num_layers=spec.num_layers,
        width=spec.hidden_size,
        dtype=torch.float16,
    )


def _run_groups(
    groups: Sequence[XPBaselineGroup],
    records: Sequence[XPBaselineRecord],
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    endpoint: str,
    method: str,
    micro_batch_records: int,
    hash_mode: str,
    store: PageableDramExtentStore | None,
    journal: XPRollingJournal | None,
) -> dict[str, object]:
    if method not in {"s0", "s1"}:
        raise ValueError("XP exact baseline method differs")
    records_by_id = {
        value.record_id: value for value in records
    }
    slot_count = 1 if method == "s0" else 2
    slots = tuple(
        _allocate_host_slot(
            records,
            groups,
            rank=rank,
            spec=XPProjectedModelSpec(
                **asdict(embedding_spec(dense, embedding))
            ),
            pin_memory=device.type == "cuda",
        )
        for _ in range(slot_count)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baseline_hbm = (
        torch.cuda.memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    phase_seconds = {
        "host_stage": 0.0,
        "compute_and_transfer": 0.0,
        "validation": 0.0,
        "publication": 0.0,
        "metadata_commit": 0.0,
    }
    lookup: dict[str, int] = {}
    h2d_bytes = 0
    d2h_bytes = 0
    written_bytes = 0
    record_hashes: list[dict[str, object]] = []
    commit_receipts: list[dict[str, object]] = []
    wall_started = time.perf_counter()
    d2h_stream = (
        torch.cuda.Stream(device=device)
        if device.type == "cuda"
        else None
    )
    if method == "s0":
        slot = slots[0]
        for group in groups:
            phase_seconds["host_stage"] += _fill_slot(
                slot,
                group,
                records_by_id,
                rank=rank,
                endpoint=endpoint,
            )
            computed = _compute_slot(
                slot,
                dense,
                embedding,
                rank=rank,
                world_size=world_size,
                device=device,
                micro_batch_records=micro_batch_records,
                d2h_stream=d2h_stream,
            )
            phase_seconds["compute_and_transfer"] += float(
                computed["compute_seconds"]
            )
            _merge_counts(lookup, computed["lookup"])
            h2d_bytes += int(computed["h2d_bytes"])
            d2h_bytes += int(computed["d2h_bytes"])
            drained = _drain_slot(
                store,
                slot,
                hash_mode,
                journal,
            )
            phase_seconds["validation"] += float(
                drained["validation_seconds"]
            )
            phase_seconds["publication"] += float(
                drained["publication_seconds"]
            )
            phase_seconds["metadata_commit"] += float(
                drained["metadata_commit_seconds"]
            )
            written_bytes += int(drained["written_bytes"])
            record_hashes.extend(drained["records"])
            commit_receipts.append(drained["commit_receipt"])
    else:
        pending: list[Future[dict[str, object]] | None] = [
            None,
            None,
        ]
        prefetched: Future[float] | None = None
        with (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="xp-baseline-prefetch",
            ) as prefetch,
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="xp-baseline-drain",
            ) as drain,
        ):
            def consume(
                future: Future[dict[str, object]],
            ) -> None:
                nonlocal written_bytes
                drained = future.result()
                phase_seconds["validation"] += float(
                    drained["validation_seconds"]
                )
                phase_seconds["publication"] += float(
                    drained["publication_seconds"]
                )
                phase_seconds["metadata_commit"] += float(
                    drained["metadata_commit_seconds"]
                )
                written_bytes += int(drained["written_bytes"])
                record_hashes.extend(drained["records"])
                commit_receipts.append(
                    drained["commit_receipt"]
                )

            phase_seconds["host_stage"] += _fill_slot(
                slots[0],
                groups[0],
                records_by_id,
                rank=rank,
                endpoint=endpoint,
            )
            for index, group in enumerate(groups):
                slot_index = index % 2
                slot = slots[slot_index]
                prior = pending[slot_index]
                if prior is not None:
                    consume(prior)
                    pending[slot_index] = None
                if prefetched is not None:
                    phase_seconds["host_stage"] += prefetched.result()
                    prefetched = None
                if slot.group != group:
                    raise RuntimeError("XP S1 prefetched group differs")
                next_slot_index = (
                    (index + 1) % 2
                    if index + 1 < len(groups)
                    else None
                )
                if index + 1 < len(groups):
                    assert next_slot_index is not None
                    if pending[next_slot_index] is None:
                        next_slot = slots[next_slot_index]
                        prefetched = prefetch.submit(
                            _fill_slot,
                            next_slot,
                            groups[index + 1],
                            records_by_id,
                            rank=rank,
                            endpoint=endpoint,
                        )
                computed = _compute_slot(
                    slot,
                    dense,
                    embedding,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    micro_batch_records=micro_batch_records,
                    d2h_stream=d2h_stream,
                )
                phase_seconds["compute_and_transfer"] += float(
                    computed["compute_seconds"]
                )
                _merge_counts(lookup, computed["lookup"])
                h2d_bytes += int(computed["h2d_bytes"])
                d2h_bytes += int(computed["d2h_bytes"])
                if (
                    next_slot_index is not None
                    and prefetched is None
                ):
                    blocked = pending[next_slot_index]
                    if blocked is None:
                        raise RuntimeError(
                            "XP S1 next slot state differs"
                        )
                    consume(blocked)
                    pending[next_slot_index] = None
                    prefetched = prefetch.submit(
                        _fill_slot,
                        slots[next_slot_index],
                        groups[index + 1],
                        records_by_id,
                        rank=rank,
                        endpoint=endpoint,
                    )
                pending[slot_index] = drain.submit(
                    _drain_slot,
                    store,
                    slot,
                    hash_mode,
                    journal,
                )
            for future in pending:
                if future is None:
                    continue
                consume(future)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_started
    observed_ids = [
        int(value["record_id"]) for value in record_hashes
    ]
    expected_ids = [
        value.record_id
        for value in records
        if value.owner_rank == rank
    ]
    if sorted(observed_ids) != sorted(expected_ids):
        raise RuntimeError("XP exact rank coverage differs")
    ordered_receipts = sorted(
        commit_receipts,
        key=lambda value: int(value["group_ordinal"]),
    )
    if [
        int(value["group_ordinal"])
        for value in ordered_receipts
    ] != list(range(len(groups))):
        raise RuntimeError("XP exact group commit coverage differs")
    return {
        "rank": rank,
        "wall_seconds": wall_seconds,
        "phase_seconds": phase_seconds,
        "lookup": lookup,
        "h2d_bytes": h2d_bytes,
        "d2h_bytes": d2h_bytes,
        "written_bytes": written_bytes,
        "records": len(record_hashes),
        "record_ids_sha256": _rank_record_ids_sha256(observed_ids),
        "record_hashes": sorted(
            record_hashes,
            key=lambda value: int(value["record_id"]),
        ),
        "commit_receipts": ordered_receipts,
        "host_slot_count": slot_count,
        "host_slot_bytes": sum(value.nbytes for value in slots),
        "pipeline": {
            "method": method,
            "whole_group_host_slots": slot_count,
            "input_lookahead_groups": 0 if method == "s0" else 1,
            "output_drain_credit": 0 if method == "s0" else 1,
            "d2h_to_pinned_host": device.type == "cuda",
            "validation_and_pageable_publication_overlap": (
                method == "s1"
            ),
        },
        "baseline_hbm_allocated_bytes": baseline_hbm,
        "peak_hbm_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0
        ),
        "peak_hbm_reserved_bytes": (
            torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else 0
        ),
        "store": store.ledger().to_dict() if store else None,
    }


def embedding_spec(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
) -> XPProjectedModelSpec:
    cfg = dense.cfg
    return XPProjectedModelSpec(
        num_embeddings=embedding.num_embeddings,
        embedding_width=embedding.embedding_width,
        hidden_size=embedding.hidden_size,
        num_prediction_items=int(cfg.num_prediction_items),
        num_behaviors=cfg.num_behaviors,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        head_dim=int(cfg.head_dim),
        max_seq_len=cfg.max_seq_len,
    )


def _aggregate_reports(
    local: dict[str, object],
    *,
    rank: int,
    world_size: int,
) -> dict[str, object] | None:
    reports: list[object] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(reports, local)
    else:
        reports[0] = local
    if rank != 0:
        return None
    resolved = [
        value for value in reports if isinstance(value, dict)
    ]
    if len(resolved) != world_size:
        raise RuntimeError("XP rank reports differ")
    record_hashes = sorted(
        (
            value
            for report in resolved
            for value in report["record_hashes"]
        ),
        key=lambda value: int(value["record_id"]),
    )
    ids = [int(value["record_id"]) for value in record_hashes]
    if len(ids) != len(set(ids)):
        raise RuntimeError("XP global record coverage overlaps")
    lookup_names = (
        "requested_tokens",
        "remote_tokens",
        "request_bytes",
        "response_bytes",
    )
    rank_capacity = []
    for value in resolved:
        store = value.get("store")
        live_bytes = (
            int(store["payload_nbytes"])
            if isinstance(store, Mapping)
            else 0
        )
        slot_bytes = int(value["host_slot_bytes"])
        rank_capacity.append(
            {
                "rank": int(value["rank"]),
                "live_store_payload_bytes": live_bytes,
                "bounded_host_slot_bytes": slot_bytes,
                "peak_live_plus_slots_bytes": live_bytes + slot_bytes,
                "peak_hbm_allocated_bytes": int(
                    value["peak_hbm_allocated_bytes"]
                ),
                "peak_hbm_reserved_bytes": int(
                    value["peak_hbm_reserved_bytes"]
                ),
            }
        )
    return {
        "rank_reports": resolved,
        "max_rank_wall_seconds": max(
            float(value["wall_seconds"]) for value in resolved
        ),
        "records": len(ids),
        "record_ids_sha256": _rank_record_ids_sha256(ids),
        "output_hash": {
            "mode": "per_record",
            "sha256": canonical_sha256(
                {"record_hashes": record_hashes}
            ),
        },
        "lookup": {
            name: sum(
                int(value["lookup"].get(name, 0))
                for value in resolved
            )
            for name in lookup_names
        },
        "h2d_bytes": sum(
            int(value["h2d_bytes"]) for value in resolved
        ),
        "d2h_bytes": sum(
            int(value["d2h_bytes"]) for value in resolved
        ),
        "written_bytes": sum(
            int(value["written_bytes"]) for value in resolved
        ),
        "peak_hbm_allocated_bytes_max_rank": max(
            int(value["peak_hbm_allocated_bytes"])
            for value in resolved
        ),
        "peak_hbm_reserved_bytes_max_rank": max(
            int(value["peak_hbm_reserved_bytes"])
            for value in resolved
        ),
        "host_slot_bytes_sum": sum(
            int(value["host_slot_bytes"]) for value in resolved
        ),
        "capacity": {
            "rank_ledgers": rank_capacity,
            "global_live_store_payload_bytes": sum(
                int(value["live_store_payload_bytes"])
                for value in rank_capacity
            ),
            "global_bounded_host_slot_bytes": sum(
                int(value["bounded_host_slot_bytes"])
                for value in rank_capacity
            ),
            "global_peak_live_plus_slots_bytes": sum(
                int(value["peak_live_plus_slots_bytes"])
                for value in rank_capacity
            ),
            "max_rank_peak_live_plus_slots_bytes": max(
                int(value["peak_live_plus_slots_bytes"])
                for value in rank_capacity
            ),
        },
    }


def run_exact_baseline(
    inputs: XPBaselineInputs,
    *,
    checkpoint_root: str | Path,
    checkpoint_version: int,
    rank: int,
    world_size: int,
    device: torch.device,
    method: str,
    endpoint: str,
    group_target_bytes: int,
    micro_batch_records: int,
    hash_mode: str,
    store_path: str | Path | None = None,
    store_mode: str = "none",
    source_manifest: Mapping[str, object] | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, object] | None:
    if (
        micro_batch_records < 1
        or store_mode not in {"none", "create", "open"}
        or (store_path is None) != (store_mode == "none")
        or (store_mode == "create" and endpoint != "old")
        or (store_mode == "open" and endpoint != "target")
        or (
            store_mode == "open"
            and (
                source_manifest is None
                or source_manifest.get("endpoint") != "old"
                or source_manifest.get("benchmark_id")
                != inputs.benchmark["benchmark_id"]
                or source_manifest.get("capacity_name")
                != inputs.capacity_name
                or int(source_manifest.get("records", -1))
                != len(inputs.records)
                or source_manifest.get("bindings") != inputs.bindings
            )
        )
    ):
        raise ValueError("XP microbatch size must be positive")
    groups = build_groups(
        inputs.records,
        world_size=world_size,
        group_target_bytes=group_target_bytes,
    )
    dense, embedding, checkpoint = load_inference_checkpoint(
        checkpoint_root,
        checkpoint_version,
        inputs.spec,
        rank=rank,
        world_size=world_size,
        device=device,
        process_group=process_group,
    )
    store = None
    journal = None
    if store_path is not None:
        rank_path = Path(
            f"{store_path}.rank{rank:02d}.dram"
        )
        store = _create_store(
            rank_path,
            inputs.records,
            rank,
            inputs.spec,
            create=store_mode == "create",
        )
        journal = XPRollingJournal(
            rank_path.with_suffix(
                rank_path.suffix + ".ledger.json"
            ),
            mode=store_mode,
            binding={
                "benchmark_id": inputs.benchmark["benchmark_id"],
                "capacity_name": inputs.capacity_name,
                "rank": rank,
                "world_size": world_size,
                "store_path": str(rank_path),
                "store_layout_sha256": store.layout_sha256,
                "bindings": inputs.bindings,
            },
            source_manifest=source_manifest,
        )
        if store_mode == "open":
            source_ledger = store.ledger()
            expected_source_tokens = sum(
                value.old_length
                for value in inputs.records
                if value.owner_rank == rank
            )
            recorded_source = journal.state.get("source")
            if (
                source_ledger.covered_tokens
                != expected_source_tokens
                or not isinstance(recorded_source, Mapping)
                or not isinstance(
                    recorded_source.get("store"),
                    Mapping,
                )
                or int(
                    recorded_source["store"].get(
                        "covered_tokens",
                        -1,
                    )
                )
                != expected_source_tokens
            ):
                raise RuntimeError(
                    "XP rolling source coverage differs"
                )
    if world_size > 1:
        dist.barrier(group=process_group)
    local = _run_groups(
        groups,
        inputs.records,
        dense,
        embedding,
        rank=rank,
        world_size=world_size,
        device=device,
        endpoint=endpoint,
        method=method,
        micro_batch_records=micro_batch_records,
        hash_mode=hash_mode,
        store=store,
        journal=journal,
    )
    if store is not None:
        store_ledger = local.get("store")
        expected_tokens = sum(
            value.length(endpoint)
            for value in inputs.records
            if value.owner_rank == rank
        )
        if (
            not isinstance(store_ledger, Mapping)
            or int(store_ledger.get("covered_tokens", -1))
            != expected_tokens
            or (
                endpoint == "target"
                and int(store_ledger.get(
                    "complete_records",
                    -1,
                ))
                != sum(
                    value.owner_rank == rank
                    for value in inputs.records
                )
            )
        ):
            raise RuntimeError(
                "XP rolling publication coverage differs"
            )
    if journal is not None and store_mode == "open":
        local["rolling_journal"] = journal.finalize_target(
            expected_groups=len(groups),
            expected_record_ids=[
                value.record_id
                for value in inputs.records
                if value.owner_rank == rank
            ],
        )
        local["reclaimed_old_valid_bytes"] = sum(
            value.old_valid_bytes
            for value in inputs.records
            if value.owner_rank == rank
        )
    else:
        local["rolling_journal"] = None
        local["reclaimed_old_valid_bytes"] = 0
    if world_size > 1:
        dist.barrier(group=process_group)
    aggregated = _aggregate_reports(
        local,
        rank=rank,
        world_size=world_size,
    )
    if aggregated is not None:
        aggregated.update(
            {
                "protocol": PROTOCOL,
                "scientific_result": False,
                "formal_result": False,
                "benchmark_id": inputs.benchmark["benchmark_id"],
                "capacity_name": inputs.capacity_name,
                "method": method,
                "endpoint": endpoint,
                "world_size": world_size,
                "checkpoint": checkpoint,
                "bindings": inputs.bindings,
                "records_expected": len(inputs.records),
                "groups": len(groups),
                "group_target_bytes": group_target_bytes,
                "group_plan_sha256": group_plan_sha256(groups),
                "micro_batch_records": micro_batch_records,
                "feature_policy": {
                    "items": "trace-grounded mapped QK HET items",
                    "behaviors": "constant exposure behavior id 1",
                    "time_deltas": (
                        "within-user ordinal-derived inter-event delta "
                        "seconds; a crop beginning after ordinal zero keeps "
                        "a one-second predecessor delta"
                    ),
                },
                "hash_mode": hash_mode,
                "lookup_definition": {
                    "requested_tokens": (
                        "all valid exact-route history tokens"
                    ),
                    "remote_tokens": (
                        "requested item rows whose modulo owner differs "
                        "from the requesting record-owner rank"
                    ),
                    "request_bytes": (
                        "logical remote row-id payload at int64 width"
                    ),
                    "response_bytes": (
                        "logical owner-projected FP32 H-width response "
                        "payload"
                    ),
                    "excluded_bytes": (
                        "collective count exchange and transport framing"
                    ),
                },
                "transaction": {
                    "validate_group_before_publication": True,
                    "group_metadata_commit_after_publication": True,
                    "old_extent_reclaimed_by_same-arena overwrite": (
                        store_mode == "open"
                        and endpoint == "target"
                    ),
                    "complete_private_target_store": False,
                    "store_mode": store_mode,
                    "source_manifest_sha256": (
                        canonical_sha256(source_manifest)
                        if source_manifest is not None
                        else None
                    ),
                    "commit_scope": (
                        "rank-local durable metadata after rank-local "
                        "publication; identical global group order"
                    ),
                    "reclaimed_old_valid_bytes": sum(
                        int(value.get(
                            "reclaimed_old_valid_bytes",
                            0,
                        ))
                        for value in aggregated["rank_reports"]
                    ),
                },
            }
        )
    if journal is not None and store_mode == "create":
        envelope: list[object] = [aggregated]
        if world_size > 1:
            dist.broadcast_object_list(
                envelope,
                src=0,
                group=process_group,
            )
        source_result = envelope[0]
        if not isinstance(source_result, Mapping):
            raise RuntimeError(
                "XP source manifest broadcast differs"
            )
        journal.finalize_source(
            local_report=local,
            source_manifest=source_result,
        )
    if store is not None:
        store.close()
    release_inference_checkpoint(dense, embedding)
    return aggregated


def checkpoint_manifest_binding(
    root: str | Path,
    version: int,
    spec: XPProjectedModelSpec,
    world_size: int,
) -> dict[str, object]:
    path = Path(root) / f"theta_{version}" / "manifest.json"
    manifest = _load_json(path)
    if (
        manifest.get("schema") != XP_PROJECTED_CHECKPOINT_SCHEMA
        or int(manifest.get("version", -1)) != version
        or int(manifest.get("world_size", -1)) != world_size
        or manifest.get("spec") != asdict(spec)
    ):
        raise ValueError("XP checkpoint manifest binding differs")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "version": version,
        "provenance": manifest.get("provenance"),
        "optimizer_active_rows": manifest.get("optimizer_active_rows"),
    }


def run_partial_exact_baseline(
    inputs: XPBaselineInputs,
    *,
    checkpoint_root: str | Path,
    checkpoint_version: int,
    fraction: float,
    rank: int,
    world_size: int,
    device: torch.device,
    group_target_bytes: int,
    micro_batch_records: int,
    hash_mode: str,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, object] | None:
    benchmark_hash = str(
        inputs.bindings["benchmark_config"]["sha256"]
    )
    selected = select_partial_exact(
        inputs.records,
        fraction,
        benchmark_hash,
    )
    selection_hash = _rank_record_ids_sha256(
        [value.record_id for value in selected]
    )
    if selected:
        selected_inputs = replace(inputs, records=selected)
        result = run_exact_baseline(
            selected_inputs,
            checkpoint_root=checkpoint_root,
            checkpoint_version=checkpoint_version,
            rank=rank,
            world_size=world_size,
            device=device,
            method="s0",
            endpoint="target",
            group_target_bytes=group_target_bytes,
            micro_batch_records=micro_batch_records,
            hash_mode=hash_mode,
            process_group=process_group,
        )
    else:
        checkpoint = checkpoint_manifest_binding(
            checkpoint_root,
            checkpoint_version,
            inputs.spec,
            world_size,
        )
        if world_size > 1:
            dist.barrier(group=process_group)
        started = time.perf_counter()
        if world_size > 1:
            dist.barrier(group=process_group)
        local = {
            "rank": rank,
            "wall_seconds": time.perf_counter() - started,
            "phase_seconds": {},
            "lookup": {},
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "written_bytes": 0,
            "records": 0,
            "record_ids_sha256": _rank_record_ids_sha256(()),
            "record_hashes": [],
            "host_slot_count": 0,
            "host_slot_bytes": 0,
            "baseline_hbm_allocated_bytes": 0,
            "peak_hbm_allocated_bytes": 0,
            "peak_hbm_reserved_bytes": 0,
            "store": None,
        }
        result = _aggregate_reports(
            local,
            rank=rank,
            world_size=world_size,
        )
        if result is not None:
            result.update(
                {
                    "protocol": PROTOCOL,
                    "scientific_result": False,
                    "formal_result": False,
                    "benchmark_id": inputs.benchmark["benchmark_id"],
                    "capacity_name": inputs.capacity_name,
                    "method": "resident_partial_exact",
                    "endpoint": "target",
                    "world_size": world_size,
                    "checkpoint": checkpoint,
                    "bindings": inputs.bindings,
                    "hash_mode": hash_mode,
                }
            )
    if result is not None:
        result["method"] = "resident_partial_exact"
        result.setdefault(
            "lookup_definition",
            {
                "requested_tokens": (
                    "all valid exact-route history tokens"
                ),
                "remote_tokens": (
                    "requested item rows whose modulo owner differs "
                    "from the requesting record-owner rank"
                ),
                "request_bytes": (
                    "logical remote row-id payload at int64 width"
                ),
                "response_bytes": (
                    "logical owner-projected FP32 H-width response "
                    "payload"
                ),
                "excluded_bytes": (
                    "collective count exchange and transport framing"
                ),
            },
        )
        result["partial_exact"] = {
            "fraction_requested": fraction,
            "universe_records": len(inputs.records),
            "exact_records": len(selected),
            "stale_reuse_records": len(inputs.records) - len(selected),
            "selection": (
                "stable SHA256 order over benchmark hash and record id"
            ),
            "exact_record_ids_sha256": selection_hash,
            "nonexact_route": (
                "stale reuse with no embedding lookup or exact output "
                "materialization in this communication characterization"
            ),
        }
        result["records_expected"] = len(inputs.records)
    return result
