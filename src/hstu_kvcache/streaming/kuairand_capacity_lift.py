from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .kuairand_projected_persistent import (
    CHECKPOINT_SCHEMA,
    PROTOCOL,
    _accepted_path,
    _artifact_path,
    _broadcast,
    _manifest_path,
    _read_manifest,
    _save_checkpoint,
    load_persistent_config,
)
from .kuairand_projected_scale import (
    _capacity_physical_ids,
    _copy_semantic_weight_to_embedding,
    _distributed,
    _initialize_model,
    _seed,
)
from .kuairand_query_transition import (
    _atomic_json,
    file_sha256,
    load_config,
)

CAPACITY_LIFT_SCHEMA = "evokv_kuairand_capacity_lift_v0"


def _load_documents(
    config_path: str | Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    target_path = Path(config_path)
    target = load_persistent_config(target_path)
    lift = target.get("capacity_lift")
    if not isinstance(lift, dict):
        raise ValueError("KuaiRand capacity lift binding is absent")
    source_record = lift.get("source_config")
    source_path = (
        Path(source_record.get("path", ""))
        if isinstance(source_record, dict)
        else Path()
    )
    if (
        lift.get("schema") != CAPACITY_LIFT_SCHEMA
        or lift.get("mapping") != "strided_hash_v0"
        or not isinstance(source_record, dict)
        or not source_path.is_file()
        or file_sha256(source_path) != source_record.get("sha256")
    ):
        raise ValueError("KuaiRand capacity lift binding differs")
    source = load_persistent_config(source_path)
    versions = int(target["checkpoint"]["versions"])
    source_versions = lift.get("source_versions")
    geometry_fields = ("hidden_size", "embedding_width", "num_layers", "num_heads")
    if (
        target.get("protocol") != PROTOCOL
        or source.get("protocol") != PROTOCOL
        or source_versions != list(range(1, versions + 1))
        or int(source["checkpoint"]["versions"]) < versions
        or target["transitions"] != source["transitions"][:versions]
        or target["parent"] != source["parent"]
        or any(target["model"][field] != source["model"][field] for field in geometry_fields)
        or target["model"].get("query_mode") != source["model"].get("query_mode")
        or int(source["model"].get("embedding_replicas", 1)) != 1
        or int(source["model"].get("embedding_capacity_multiplier", 1)) != 1
        or int(target["model"].get("embedding_replicas", 1)) != 1
        or int(target["model"].get("embedding_capacity_multiplier", 1)) < 2
        or int(source["execution"]["world_size"]) != 1
        or int(target["execution"]["world_size"]) != 2
        or target["checkpoint"].get("embedding_storage") != "full"
        or int(target["checkpoint"].get("imported_prefix_versions", -1)) != versions
        or Path(lift.get("source_checkpoint_root", ""))
        != Path(source["outputs"]["checkpoint_root"])
        or Path(lift.get("source_result_root", "")) != Path(source["outputs"]["root"])
        or Path(target["outputs"]["checkpoint_root"])
        == Path(source["outputs"]["checkpoint_root"])
        or Path(target["outputs"]["root"]) == Path(source["outputs"]["root"])
    ):
        raise ValueError("KuaiRand capacity lift contract differs")
    return target_path, target, source_path, source


def _load_source_version(
    source_root: Path,
    source_result_root: Path,
    source: dict[str, Any],
    source_sha256: str,
    version: int,
    current_weight: torch.Tensor | None,
    verify_hash: bool,
) -> tuple[
    torch.Tensor,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    manifest = _read_manifest(
        source_root,
        version,
        source,
        source_sha256,
        verify_hash,
    )
    directory = source_root / f"theta_{version}"
    dense_payload = torch.load(
        _artifact_path(directory, manifest["dense"], False),
        map_location="cpu",
        weights_only=True,
    )
    projection_payload = torch.load(
        _artifact_path(directory, manifest["projection"], False),
        map_location="cpu",
        weights_only=True,
    )
    embedding_payload = torch.load(
        _artifact_path(directory, manifest["embedding_shards"][0], False),
        map_location="cpu",
        weights_only=True,
    )
    tracker_payload = torch.load(
        _artifact_path(directory, manifest["tracker_shards"][0], False),
        map_location="cpu",
        weights_only=True,
    )
    accepted_path = _accepted_path(source_result_root, version)
    if not accepted_path.is_file():
        raise ValueError("KuaiRand capacity lift source acceptance is absent")
    accepted = json.loads(accepted_path.read_text())
    if (
        dense_payload.get("schema") != CHECKPOINT_SCHEMA
        or dense_payload.get("version") != version
        or projection_payload.get("schema") != CHECKPOINT_SCHEMA
        or projection_payload.get("version") != version
        or embedding_payload.get("schema") != CHECKPOINT_SCHEMA
        or embedding_payload.get("version") != version
        or embedding_payload.get("rank") != 0
        or embedding_payload.get("world_size") != 1
        or tracker_payload.get("schema") != CHECKPOINT_SCHEMA
        or tracker_payload.get("version") != version
        or tracker_payload.get("rank") != 0
        or accepted.get("protocol") != PROTOCOL
        or accepted.get("version") != version
        or accepted.get("status") != "accepted"
    ):
        raise ValueError("KuaiRand capacity lift source payload differs")
    storage = embedding_payload.get("storage", "full")
    if storage == "full":
        source_weight = embedding_payload.get("local_weight")
        if not isinstance(source_weight, torch.Tensor):
            raise ValueError("KuaiRand capacity lift source embedding differs")
    elif storage == "sparse_delta":
        indices = embedding_payload.get("local_indices")
        values = embedding_payload.get("local_values")
        if (
            current_weight is None
            or embedding_payload.get("parent_version") != version - 1
            or not isinstance(indices, torch.Tensor)
            or not isinstance(values, torch.Tensor)
            or indices.dtype != torch.int64
            or values.shape != (indices.numel(), current_weight.shape[1])
        ):
            raise ValueError("KuaiRand capacity lift source delta differs")
        current_weight.index_copy_(0, indices, values)
        source_weight = current_weight
    else:
        raise ValueError("KuaiRand capacity lift source storage differs")
    if (
        source_weight.ndim != 2
        or source_weight.shape[0] != int(embedding_payload["num_embeddings"])
        or source_weight.shape[1] != int(embedding_payload["embedding_width"])
    ):
        raise ValueError("KuaiRand capacity lift source embedding shape differs")
    del embedding_payload
    gc.collect()
    return (
        source_weight,
        dense_payload,
        projection_payload,
        tracker_payload,
        manifest,
        accepted,
    )


def _map_tracker(
    tracker,
    tracker_payload: dict[str, Any],
    multiplier: int,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    bitmap = tracker_payload.get("local_bitmap")
    counts = tracker_payload.get("local_update_counts")
    if (
        not isinstance(bitmap, torch.Tensor)
        or not isinstance(counts, torch.Tensor)
        or bitmap.shape != counts.shape
        or bitmap.ndim != 1
        or bool(torch.any((bitmap != 0) & (bitmap != 1)))
        or bool(torch.any(counts < 0))
        or not torch.equal(bitmap, (counts > 0).to(torch.uint8))
        or bool(bitmap[0])
        or bool(counts[0])
    ):
        raise ValueError("KuaiRand capacity lift source tracker differs")
    tracker.local_bitmap.zero_()
    tracker.local_update_counts.zero_()
    semantic_ids = torch.nonzero(bitmap, as_tuple=False).flatten()
    physical_ids = _capacity_physical_ids(semantic_ids, multiplier)
    owned = torch.remainder(physical_ids, world_size) == rank
    local_ids = torch.div(physical_ids[owned], world_size, rounding_mode="floor")
    tracker.local_bitmap.index_copy_(0, local_ids, bitmap.index_select(0, semantic_ids[owned]))
    tracker.local_update_counts.index_copy_(
        0,
        local_ids,
        counts.index_select(0, semantic_ids[owned]),
    )
    return int(semantic_ids.numel()), int(owned.sum().item())


@torch.no_grad()
def _verify_active_embedding(
    embedding,
    source_weight: torch.Tensor,
    multiplier: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> float:
    maximum = torch.zeros((), dtype=torch.float32, device=device)
    semantic_rows = source_weight.shape[0] - 1
    for start in range(1, semantic_rows + 1, 65_536):
        end = min(semantic_rows + 1, start + 65_536)
        semantic_ids = torch.arange(start, end, dtype=torch.int64)
        physical_ids = _capacity_physical_ids(semantic_ids, multiplier)
        owned = torch.remainder(physical_ids, world_size) == rank
        if bool(torch.any(owned)):
            local_ids = torch.div(
                physical_ids[owned],
                world_size,
                rounding_mode="floor",
            ).to(device)
            expected = source_weight.index_select(0, semantic_ids[owned]).to(device)
            observed = embedding.local_weight.index_select(0, local_ids)
            maximum = torch.maximum(maximum, torch.max(torch.abs(observed - expected)))
    if dist.is_initialized():
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    result = float(maximum.item())
    if result != 0.0:
        raise RuntimeError("KuaiRand capacity lift changed active embedding values")
    return result


def _target_accepted(
    source_accepted: dict[str, Any],
    source_manifest_path: Path,
    source_accepted_path: Path,
    target_manifest: dict[str, Any],
    target_manifest_path: Path,
    target: dict[str, Any],
    maximum_embedding_error: float,
) -> dict[str, Any]:
    version = int(source_accepted["version"])
    return {
        "protocol": PROTOCOL,
        "version": version,
        "source_version": version - 1,
        "status": "accepted",
        "candidate": source_accepted["candidate"],
        "checkpoint": {
            "path": str(target_manifest_path),
            "sha256": file_sha256(target_manifest_path),
            "bytes": int(target_manifest["checkpoint_bytes"]),
        },
        "selection": target["selection"],
        "capacity_lift": {
            "schema": CAPACITY_LIFT_SCHEMA,
            "mapping": "strided_hash_v0",
            "function_preserving": True,
            "maximum_active_embedding_absolute_error": maximum_embedding_error,
            "source_manifest": {
                "path": str(source_manifest_path),
                "sha256": file_sha256(source_manifest_path),
            },
            "source_acceptance": {
                "path": str(source_accepted_path),
                "sha256": file_sha256(source_accepted_path),
            },
        },
        "scientific_result": False,
        "formal_result": False,
    }


def preflight_capacity_lift(
    config_path: str | Path,
    stop_after_version: int | None = None,
    verify_source_hashes: bool = False,
) -> dict[str, Any]:
    target_path, target, source_path, source = _load_documents(config_path)
    target_sha256 = file_sha256(target_path)
    source_sha256 = file_sha256(source_path)
    configured_versions = int(target["checkpoint"]["versions"])
    final_version = configured_versions if stop_after_version is None else stop_after_version
    if not 1 <= final_version <= configured_versions:
        raise ValueError("KuaiRand capacity lift final version differs")
    source_root = Path(source["outputs"]["checkpoint_root"])
    source_result_root = Path(source["outputs"]["root"])
    target_root = Path(target["outputs"]["checkpoint_root"])
    target_result_root = Path(target["outputs"]["root"])
    completed = []
    missing = []
    for version in range(1, final_version + 1):
        _read_manifest(
            source_root,
            version,
            source,
            source_sha256,
            verify_source_hashes,
        )
        if not _accepted_path(source_result_root, version).is_file():
            raise ValueError("KuaiRand capacity lift source acceptance is absent")
        target_manifest_exists = _manifest_path(target_root, version).is_file()
        target_accepted_exists = _accepted_path(target_result_root, version).is_file()
        if target_accepted_exists and not target_manifest_exists:
            raise RuntimeError("KuaiRand capacity lift target boundary differs")
        if target_manifest_exists:
            _read_manifest(
                target_root,
                version,
                target,
                target_sha256,
                False,
            )
            completed.append(version)
        else:
            missing.append(version)
    usage = shutil.disk_usage(target_root.parent)
    required = (
        len(missing)
        * int(target["checkpoint"]["expected_checkpoint_bytes_per_version"])
        + int(target["checkpoint"]["write_reserve_bytes"])
    )
    if usage.free < required:
        raise RuntimeError("KuaiRand capacity lift disk preflight failed")
    return {
        "status": "ready",
        "schema": CAPACITY_LIFT_SCHEMA,
        "target_config": {"path": str(target_path), "sha256": target_sha256},
        "source_config": {"path": str(source_path), "sha256": source_sha256},
        "configured_versions": configured_versions,
        "final_version": final_version,
        "completed_versions": completed,
        "missing_versions": missing,
        "free_bytes": usage.free,
        "required_free_bytes": required,
        "world_size": int(target["execution"]["world_size"]),
        "scientific_result": False,
        "formal_result": False,
    }


def run_capacity_lift(
    config_path: str | Path,
    stop_after_version: int | None = None,
    verify_source_hashes: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    preflight = preflight_capacity_lift(
        config_path,
        stop_after_version,
        verify_source_hashes=False,
    )
    target_path, target, source_path, source = _load_documents(config_path)
    target_sha256 = file_sha256(target_path)
    source_sha256 = file_sha256(source_path)
    final_version = int(preflight["final_version"])
    rank, world_size, device = _distributed(target)
    _seed(int(target["training"]["seed"]))
    source_root = Path(source["outputs"]["checkpoint_root"])
    source_result_root = Path(source["outputs"]["root"])
    target_root = Path(target["outputs"]["checkpoint_root"])
    target_result_root = Path(target["outputs"]["root"])
    target_root.mkdir(parents=True, exist_ok=True)
    target_result_root.mkdir(parents=True, exist_ok=True)
    source_first_manifest = _read_manifest(
        source_root,
        1,
        source,
        source_sha256,
        verify_source_hashes and rank == 0,
    )
    semantic_rows = int(source_first_manifest["geometry"]["num_embeddings"]) - 1
    base_config = load_config(target["parent"]["base_config"]["path"])
    dense, embedding, tracker, geometry = _initialize_model(
        target,
        base_config,
        semantic_rows,
        rank,
        world_size,
        device,
    )
    if int(geometry["global_model_parameter_bytes"]) != int(
        target["checkpoint"]["expected_global_parameter_bytes"]
    ) or not bool(geometry["single_gpu_parameter_overflow"]):
        raise RuntimeError("KuaiRand capacity lift target geometry differs")
    multiplier = int(target["model"]["embedding_capacity_multiplier"])
    current_weight = None
    version_records = []
    for version in range(1, final_version + 1):
        (
            current_weight,
            dense_payload,
            projection_payload,
            tracker_payload,
            source_manifest,
            source_accepted,
        ) = _load_source_version(
            source_root,
            source_result_root,
            source,
            source_sha256,
            version,
            current_weight,
            verify_source_hashes and rank == 0,
        )
        target_manifest_path = _manifest_path(target_root, version)
        target_accepted_path = _accepted_path(target_result_root, version)
        if target_manifest_path.is_file():
            target_manifest = _read_manifest(
                target_root,
                version,
                target,
                target_sha256,
                False,
            )
            maximum_error = 0.0
            status = "already_complete"
        else:
            dense.load_state_dict(dense_payload["state_dict"], strict=True)
            with torch.no_grad():
                embedding.projection_weight.copy_(
                    projection_payload["projection_weight"].to(device)
                )
            _copy_semantic_weight_to_embedding(
                embedding,
                current_weight,
                rank,
                world_size,
                device,
            )
            source_active_rows, local_active_rows = _map_tracker(
                tracker,
                tracker_payload,
                multiplier,
                rank,
                world_size,
            )
            maximum_error = _verify_active_embedding(
                embedding,
                current_weight,
                multiplier,
                rank,
                world_size,
                device,
            )
            local_count = torch.tensor(local_active_rows, dtype=torch.int64, device=device)
            if dist.is_initialized():
                dist.all_reduce(local_count)
            if int(local_count.item()) != source_active_rows:
                raise RuntimeError("KuaiRand capacity lift tracker cardinality differs")
            target_manifest = _save_checkpoint(
                target_root,
                version,
                dense,
                embedding,
                tracker,
                geometry,
                target,
                target_sha256,
                {
                    "round_id": target["round_id"],
                    "config": {"path": str(target_path), "sha256": target_sha256},
                    "capacity_lift": {
                        "schema": CAPACITY_LIFT_SCHEMA,
                        "mapping": "strided_hash_v0",
                        "function_preserving": True,
                        "source_config": {
                            "path": str(source_path),
                            "sha256": source_sha256,
                        },
                        "source_manifest": {
                            "path": str(_manifest_path(source_root, version)),
                            "sha256": file_sha256(_manifest_path(source_root, version)),
                        },
                        "maximum_active_embedding_absolute_error": maximum_error,
                    },
                    "source_version": version,
                },
                rank,
                world_size,
            )
            status = "lifted"
        if rank == 0 and not target_accepted_path.is_file():
            accepted = _target_accepted(
                source_accepted,
                _manifest_path(source_root, version),
                _accepted_path(source_result_root, version),
                target_manifest,
                target_manifest_path,
                target,
                maximum_error,
            )
            _atomic_json(target_accepted_path, accepted)
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            version_records.append(
                {
                    "version": version,
                    "status": status,
                    "source_manifest_sha256": file_sha256(
                        _manifest_path(source_root, version)
                    ),
                    "target_manifest_sha256": file_sha256(target_manifest_path),
                    "checkpoint_bytes": int(target_manifest["checkpoint_bytes"]),
                    "maximum_active_embedding_absolute_error": maximum_error,
                }
            )
        del dense_payload, projection_payload, tracker_payload, source_manifest
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        result = {
            "schema": CAPACITY_LIFT_SCHEMA,
            "status": "complete",
            "scientific_result": False,
            "formal_result": False,
            "source_config": {"path": str(source_path), "sha256": source_sha256},
            "target_config": {"path": str(target_path), "sha256": target_sha256},
            "mapping": "strided_hash_v0",
            "function_preserving": True,
            "final_version": final_version,
            "geometry": geometry,
            "versions": version_records,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(target_result_root / f"capacity_lift_theta1_theta{final_version}.json", result)
    else:
        result = None
    result = _broadcast(result, rank)
    if dist.is_initialized():
        dist.destroy_process_group()
    return result
