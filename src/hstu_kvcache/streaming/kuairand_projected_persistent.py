from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .kuairand_projected_scale import (
    _active_rows,
    _capture_old,
    _copy_semantic_weight_to_embedding,
    _distributed,
    _evaluate_captured,
    _evaluation_batches,
    _initialize_model,
    _seed,
    _train_epochs,
)
from .kuairand_query_multiversion import _edge_config
from .kuairand_query_transition import (
    _atomic_json,
    _atomic_torch,
    _summary,
    build_workload,
    file_sha256,
    load_config,
)

PROTOCOL = "evokv_kuairand_projected_persistent_chain_v0"
CHECKPOINT_SCHEMA = "evokv_kuairand_projected_checkpoint_v0"
CANDIDATE_PROBE_SCHEMA = "evokv_kuairand_persistent_candidate_probe_config_v0"
CANDIDATE_REQUIRED_FIELDS = {
    "name",
    "update_epochs",
    "maximum_update_examples",
    "embedding_lr",
    "projection_lr",
    "dense_lr",
}
CANDIDATE_OPTIONAL_FIELDS = {
    "dense_update_scope",
    "evaluation_temperatures",
    "final_epoch_examples",
    "kv_lr",
    "latest_update_dates",
    "sampling",
    "examples_per_user",
    "temporal_training",
    "training_temperature",
}


def _candidate_valid(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    sampling = candidate.get("sampling", "global_random")
    examples_per_user = int(candidate.get("examples_per_user", 0))
    temporal_training = candidate.get("temporal_training", "pooled")
    evaluation_temperatures = candidate.get("evaluation_temperatures", [])
    final_epoch_examples = int(candidate.get("final_epoch_examples", 0))
    latest_update_dates = int(candidate.get("latest_update_dates", 0))
    training_temperature = float(candidate.get("training_temperature", 0.0))
    dense_update_scope = str(candidate.get("dense_update_scope", "full"))
    return bool(
        CANDIDATE_REQUIRED_FIELDS.issubset(candidate)
        and set(candidate).issubset(CANDIDATE_REQUIRED_FIELDS | CANDIDATE_OPTIONAL_FIELDS)
        and isinstance(candidate.get("name"), str)
        and int(candidate.get("update_epochs", 0)) >= 1
        and int(candidate.get("maximum_update_examples", 0)) >= 1
        and min(
            float(candidate.get("embedding_lr", 0)),
            float(candidate.get("projection_lr", 0)),
            float(candidate.get("dense_lr", 0)),
        )
        > 0
        and float(candidate.get("kv_lr", candidate.get("dense_lr", 0))) > 0
        and isinstance(evaluation_temperatures, list)
        and not any(float(value) <= 0 for value in evaluation_temperatures)
        and evaluation_temperatures
        == sorted(set(float(value) for value in evaluation_temperatures))
        and final_epoch_examples >= 0
        and final_epoch_examples <= int(candidate.get("maximum_update_examples", 0))
        and not (final_epoch_examples > 0 and int(candidate.get("update_epochs", 0)) < 2)
        and not ("training_temperature" in candidate and training_temperature <= 0)
        and sampling in ("global_random", "recent_per_user")
        and temporal_training in ("pooled", "sequential_dates")
        and dense_update_scope in ("full", "qkv_only", "frozen")
        and not (
            "latest_update_dates" in candidate
            and (temporal_training != "sequential_dates" or latest_update_dates < 1)
        )
        and not (sampling == "global_random" and examples_per_user != 0)
        and not (sampling == "recent_per_user" and examples_per_user < 1)
    )


def load_candidate_probe_config(
    path: str | Path, source_config_path: str | Path
) -> dict[str, Any]:
    probe_path = Path(path)
    source_path = Path(source_config_path)
    document = json.loads(probe_path.read_text())
    source = document.get("source_config")
    candidates = document.get("candidates")
    if (
        document.get("schema") != CANDIDATE_PROBE_SCHEMA
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(source, dict)
        or Path(source.get("path", "")).resolve() != source_path.resolve()
        or source.get("sha256") != file_sha256(source_path)
        or not isinstance(candidates, list)
        or not candidates
        or not all(_candidate_valid(candidate) for candidate in candidates)
        or len({candidate["name"] for candidate in candidates}) != len(candidates)
    ):
        raise ValueError("KuaiRand candidate probe config differs")
    return document


def _selection_groups_valid(groups: Any, versions: int) -> bool:
    if versions == 1:
        return groups in (None, [])
    if not isinstance(groups, list) or not groups:
        return False
    expected_start = 2
    for group in groups:
        if (
            not isinstance(group, list)
            or len(group) != 2
            or not all(isinstance(value, int) for value in group)
            or group[0] != expected_start
            or not group[0] <= group[1] <= versions
        ):
            return False
        expected_start = group[1] + 1
    return expected_start == versions + 1


def _lineage_partition_summary(
    evaluation: dict[str, Any],
    document: dict[str, Any],
    split_seed: int,
    tuning_fraction: float,
    bootstrap_samples: int,
    partition: str,
) -> dict[str, Any]:
    if partition not in ("tuning", "holdout"):
        raise ValueError("KuaiRand persistent lineage partition differs")
    records = []
    for record in evaluation["records"]:
        user_id = int(record["user_id"])
        digest = hashlib.sha256(f"{split_seed}:{user_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "little") / float(1 << 64)
        selected = value < tuning_fraction
        if selected == (partition == "tuning"):
            records.append(record)
    if not records:
        raise RuntimeError("KuaiRand persistent lineage partition is empty")
    summary_document = json.loads(json.dumps(document))
    summary_document["evaluation"]["bootstrap_samples"] = bootstrap_samples
    result = _summary(
        {"records": records, "sanity": evaluation["sanity"]},
        summary_document,
    )
    result["partition"] = {
        "name": partition,
        "split_seed": split_seed,
        "tuning_fraction": tuning_fraction,
        "records": len(records),
        "users": len({int(record["user_id"]) for record in records}),
        "bootstrap_samples": bootstrap_samples,
    }
    return result


def _lineage_holdout_summary(
    evaluation: dict[str, Any],
    document: dict[str, Any],
    split_seed: int,
    tuning_fraction: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    return _lineage_partition_summary(
        evaluation,
        document,
        split_seed,
        tuning_fraction,
        bootstrap_samples,
        "holdout",
    )


def _temperature_edge_document(
    document: dict[str, Any], temperature: float | None
) -> dict[str, Any]:
    if temperature is None:
        return document
    output = json.loads(json.dumps(document))
    output["training"]["temperature"] = float(temperature)
    return output


def load_persistent_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    parent = document.get("parent")
    model = document.get("model")
    training = document.get("training")
    evaluation = document.get("evaluation")
    execution = document.get("execution")
    selection = document.get("selection")
    checkpoint = document.get("checkpoint")
    outputs = document.get("outputs")
    transitions = document.get("transitions")
    candidates = training.get("candidate_ladder") if isinstance(training, dict) else None
    initial = training.get("initial_candidate") if isinstance(training, dict) else None
    candidate_schedule = (
        training.get("candidate_schedule", {}) if isinstance(training, dict) else None
    )
    checkpoint_policy = (
        training.get("checkpoint_policy", "quality_gate") if isinstance(training, dict) else None
    )
    embedding_storage = (
        checkpoint.get("embedding_storage", "full")
        if isinstance(checkpoint, dict)
        else None
    )
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not all(
            isinstance(value, dict)
            for value in (
                parent,
                model,
                training,
                evaluation,
                execution,
                selection,
                checkpoint,
                outputs,
            )
        )
        or not isinstance(transitions, list)
        or not 1 <= len(transitions) <= 13
        or not isinstance(candidates, list)
        or not candidates
        or not isinstance(initial, dict)
        or not isinstance(candidate_schedule, dict)
        or checkpoint_policy not in ("quality_gate", "fixed_schedule")
        or training.get("calibration_dense_update_scope", "full")
        not in ("full", "qkv_only", "frozen")
        or int(model.get("hidden_size", 0)) < 64
        or int(model.get("embedding_width", 0)) < int(model.get("hidden_size", 0))
        or not 1 <= int(model.get("embedding_replicas", 1)) <= 16
        or not 1 <= int(model.get("embedding_capacity_multiplier", 1)) <= 16
        or (
            int(model.get("embedding_replicas", 1)) > 1
            and int(model.get("embedding_capacity_multiplier", 1)) > 1
        )
        or (
            (
                int(model.get("embedding_replicas", 1)) > 1
                or int(model.get("embedding_capacity_multiplier", 1)) > 1
            )
            and int(model.get("embedding_width", 0)) != int(model.get("hidden_size", 0))
        )
        or not 2 <= int(model.get("num_layers", 0)) <= 24
        or int(model.get("num_heads", 0)) < 1
        or int(model.get("hidden_size", 0)) % int(model.get("num_heads", 0)) != 0
        or model.get("query_mode", "history_only_zero")
        not in ("history_only_zero", "last_history_item", "latest_item_query")
        or int(execution.get("world_size", 0)) not in (1, 2)
        or not 1 <= int(execution.get("lineage_source_chunk_size", 1)) <= 4
        or int(evaluation.get("candidate_count", 0)) != 100
        or int(evaluation.get("targets_per_user", 0)) not in (1, 4, 8)
        or int(evaluation.get("local_batch_size", 0)) < 1
        or selection.get("metrics") != ["ndcg_at_5"]
        or selection.get("positive_metrics") != ["mrr", "ndcg_at_5", "hit_rate_at_5"]
        or selection.get("report_metrics") != ["mrr", "ndcg_at_5", "hit_rate_at_5"]
        or float(selection.get("minimum_relative_percent", 0)) != 1.0
        or selection.get("group_metric") != "ndcg_at_5"
        or float(selection.get("minimum_group_mean_relative_percent", 0)) != 3.0
        or not _selection_groups_valid(selection.get("groups"), len(transitions))
        or int(checkpoint.get("versions", 0)) != len(transitions)
        or not isinstance(checkpoint.get("retain_bootstrap", False), bool)
        or not 0 <= int(checkpoint.get("imported_prefix_versions", 0)) <= len(transitions)
        or int(checkpoint.get("expected_global_parameter_bytes", 0)) < 1
        or int(
            checkpoint.get(
                "expected_checkpoint_bytes_per_version",
                checkpoint.get("expected_global_parameter_bytes", 0),
            )
        )
        < 1
        or embedding_storage
        not in (
            "full",
            "sparse_delta_after_imported_prefix",
            "sparse_warmup_full_suffix",
        )
        or (
            checkpoint.get("retain_bootstrap", False)
            and embedding_storage != "full"
        )
        or (
            embedding_storage == "sparse_delta_after_imported_prefix"
            and int(checkpoint.get("imported_prefix_versions", 0)) < 1
        )
        or (
            embedding_storage == "sparse_warmup_full_suffix"
            and (
                int(checkpoint.get("imported_prefix_versions", 0)) != 0
                or not 2
                <= int(checkpoint.get("full_checkpoint_from_version", 0))
                <= len(transitions)
                or int(checkpoint.get("expected_sparse_checkpoint_bytes_per_version", 0))
                < 1
            )
        )
        or int(checkpoint.get("minimum_free_bytes", 0)) < 1
        or not isinstance(outputs.get("root"), str)
        or not isinstance(outputs.get("checkpoint_root"), str)
    ):
        raise ValueError("KuaiRand persistent-chain config differs")
    names = []
    for candidate in [initial, *candidates]:
        if not _candidate_valid(candidate):
            raise ValueError("KuaiRand persistent candidate differs")
        names.append(candidate["name"])
    if len(names) != len(set(names)):
        raise ValueError("KuaiRand persistent candidate names differ")
    scheduled_versions = {str(version) for version in range(2, len(transitions) + 1)}
    candidate_names = {candidate["name"] for candidate in candidates}
    if checkpoint_policy == "fixed_schedule":
        if candidate_schedule:
            if (
                set(candidate_schedule) != scheduled_versions
                or any(name not in candidate_names for name in candidate_schedule.values())
            ):
                raise ValueError("KuaiRand persistent candidate schedule differs")
        elif len(candidates) != 1:
            raise ValueError("KuaiRand persistent fixed schedule differs")
    elif candidate_schedule:
        raise ValueError("KuaiRand persistent quality gate cannot freeze a schedule")
    for field in ("base_config", "theta0"):
        artifact = parent.get(field)
        artifact_path = Path(artifact.get("path", "")) if isinstance(artifact, dict) else Path()
        if (
            not isinstance(artifact, dict)
            or not artifact_path.is_file()
            or file_sha256(artifact_path) != artifact.get("sha256")
        ):
            raise ValueError("KuaiRand persistent parent differs")
    base_document = json.loads(Path(parent["base_config"]["path"]).read_text())
    base_model = base_document.get("model", {})
    if (
        int(base_model.get("hidden_size", 0)) != int(model["hidden_size"])
        or int(base_model.get("num_layers", 0)) != int(model["num_layers"])
        or int(base_model.get("num_heads", 0)) != int(model["num_heads"])
        or base_model.get("query_mode", "history_only_zero")
        != model.get("query_mode", "history_only_zero")
    ):
        raise ValueError("KuaiRand persistent parent geometry differs")
    for index, transition in enumerate(transitions):
        update_date = str(transition.get("update_date", ""))
        update_dates = transition.get("update_dates", [update_date])
        if (
            int(transition.get("source_version", -1)) != index
            or int(transition.get("target_version", -1)) != index + 1
            or len(update_date) != 8
            or len(str(transition.get("evaluation_date", ""))) != 8
            or not isinstance(update_dates, list)
            or not update_dates
            or update_dates != sorted(set(update_dates))
            or any(len(str(value)) != 8 for value in update_dates)
            or str(update_dates[-1]) != update_date
            or update_date >= str(transition["evaluation_date"])
        ):
            raise ValueError("KuaiRand persistent transition differs")
    return document


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _replica_fingerprint(dense, embedding) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(dense.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(_tensor_sha256(embedding.projection_weight).encode())
    return digest.hexdigest()


def _artifact(path: Path, directory: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(directory)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _artifact_path(directory: Path, record: dict[str, Any], verify_hash: bool) -> Path:
    path = directory / str(record.get("path", ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or (verify_hash and file_sha256(path) != record.get("sha256"))
    ):
        raise ValueError("KuaiRand persistent checkpoint artifact differs")
    return path


def _manifest_path(root: Path, version: int) -> Path:
    return root / f"theta_{version}" / "manifest.json"


def _embedding_storage_for_version(document: dict[str, Any], version: int) -> str:
    checkpoint = document["checkpoint"]
    configured = checkpoint.get("embedding_storage", "full")
    if configured == "full":
        return "full"
    if configured == "sparse_delta_after_imported_prefix":
        imported = int(checkpoint.get("imported_prefix_versions", 0))
        return "sparse_delta" if version > imported else "full"
    if configured == "sparse_warmup_full_suffix":
        full_from = int(checkpoint["full_checkpoint_from_version"])
        return "full" if version == 1 or version >= full_from else "sparse_delta"
    raise ValueError("KuaiRand persistent embedding storage differs")


def _expected_checkpoint_bytes_for_version(
    document: dict[str, Any], version: int
) -> int:
    checkpoint = document["checkpoint"]
    if _embedding_storage_for_version(document, version) == "sparse_delta":
        return int(
            checkpoint.get(
                "expected_sparse_checkpoint_bytes_per_version",
                checkpoint.get(
                    "expected_checkpoint_bytes_per_version",
                    checkpoint["expected_global_parameter_bytes"],
                ),
            )
        )
    return int(
        checkpoint.get(
            "expected_checkpoint_bytes_per_version",
            checkpoint["expected_global_parameter_bytes"],
        )
    )


def _read_manifest(
    root: Path,
    version: int,
    document: dict[str, Any],
    config_sha256: str,
    verify_hash: bool,
) -> dict[str, Any]:
    directory = root / f"theta_{version}"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("KuaiRand persistent checkpoint manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    compatible_hashes = {
        config_sha256,
        *document["checkpoint"].get("compatible_config_sha256", []),
    }
    expected_world_size = int(document.get("execution", {}).get("world_size", 2))
    expected_storage = _embedding_storage_for_version(document, version)
    manifest_storage = manifest.get("embedding_storage", "full")
    if (
        manifest.get("schema") != CHECKPOINT_SCHEMA
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("version") != version
        or manifest.get("world_size") != expected_world_size
        or manifest.get("config_sha256") not in compatible_hashes
        or manifest.get("scientific_result") is not False
        or manifest.get("formal_result") is not False
        or manifest.get("geometry", {}).get("global_model_parameter_bytes")
        != int(document["checkpoint"]["expected_global_parameter_bytes"])
        or len(manifest.get("embedding_shards", [])) != expected_world_size
        or len(manifest.get("tracker_shards", [])) != expected_world_size
        or manifest_storage != expected_storage
        or (
            manifest_storage == "sparse_delta"
            and (
                manifest.get("parent_version") != version - 1
                or len(manifest.get("embedding_delta_rows_by_rank", []))
                != expected_world_size
                or any(
                    int(value) < 0
                    for value in manifest.get("embedding_delta_rows_by_rank", [])
                )
                or not isinstance(manifest.get("parent_manifest_sha256"), str)
            )
        )
    ):
        raise ValueError("KuaiRand persistent checkpoint manifest differs")
    _artifact_path(directory, manifest["dense"], verify_hash)
    _artifact_path(directory, manifest["projection"], verify_hash)
    for record in manifest["embedding_shards"]:
        _artifact_path(directory, record, verify_hash)
    for record in manifest["tracker_shards"]:
        _artifact_path(directory, record, verify_hash)
    return manifest


def _save_checkpoint(
    root: Path,
    version: int,
    dense,
    embedding,
    tracker,
    geometry: dict[str, Any],
    document: dict[str, Any],
    config_sha256: str,
    provenance: dict[str, Any],
    rank: int,
    world_size: int,
    embedding_delta_rows: torch.Tensor | None = None,
) -> dict[str, Any]:
    manifest_path = _manifest_path(root, version)
    if manifest_path.is_file():
        return _read_manifest(root, version, document, config_sha256, True)
    directory = manifest_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = _replica_fingerprint(dense, embedding)
    if dist.is_initialized():
        fingerprints: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(fingerprints, fingerprint)
    else:
        fingerprints = [fingerprint]
    if len(set(fingerprints)) != 1:
        raise RuntimeError("KuaiRand persistent replicas differ")
    embedding_storage = _embedding_storage_for_version(document, version)
    if (embedding_storage == "sparse_delta") != (embedding_delta_rows is not None):
        raise ValueError("KuaiRand persistent embedding delta rows differ")
    embedding_path = directory / f"embedding_rank_{rank:05d}.pt"
    tracker_path = directory / f"tracker_rank_{rank:05d}.pt"
    if embedding_storage == "full":
        embedding_payload = {
            "schema": CHECKPOINT_SCHEMA,
            "version": version,
            "rank": rank,
            "world_size": world_size,
            "num_embeddings": embedding.num_embeddings,
            "embedding_width": embedding.embedding_width,
            "storage": embedding_storage,
            "local_weight": embedding.local_weight.detach().cpu(),
        }
        local_delta_rows = embedding.local_rows
    else:
        assert embedding_delta_rows is not None
        rows = torch.unique(
            embedding_delta_rows.detach().to(device="cpu", dtype=torch.int64),
            sorted=True,
        )
        if rows.ndim != 1 or (
            rows.numel()
            and (
                bool(torch.any(rows < 0))
                or bool(torch.any(rows >= embedding.local_rows))
            )
        ):
            raise ValueError("KuaiRand persistent embedding delta row range differs")
        values = embedding.local_weight.detach().index_select(
            0, rows.to(device=embedding.local_weight.device)
        ).cpu()
        embedding_payload = {
            "schema": CHECKPOINT_SCHEMA,
            "version": version,
            "rank": rank,
            "world_size": world_size,
            "num_embeddings": embedding.num_embeddings,
            "embedding_width": embedding.embedding_width,
            "storage": embedding_storage,
            "parent_version": version - 1,
            "local_indices": rows,
            "local_values": values,
        }
        local_delta_rows = int(rows.numel())
    _atomic_torch(
        embedding_path,
        embedding_payload,
    )
    del embedding_payload
    _atomic_torch(
        tracker_path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "version": version,
            "rank": rank,
            "world_size": world_size,
            "num_embeddings": tracker.num_embeddings,
            "local_bitmap": tracker.local_bitmap,
            "local_update_counts": tracker.local_update_counts,
        },
    )
    if dist.is_initialized():
        delta_rows_by_rank: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(delta_rows_by_rank, local_delta_rows)
        dist.barrier()
    else:
        delta_rows_by_rank = [local_delta_rows]
    if rank == 0:
        dense_path = directory / "dense.pt"
        projection_path = directory / "projection.pt"
        _atomic_torch(
            dense_path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "version": version,
                "config": asdict(dense.cfg),
                "state_dict": dense.state_dict(),
            },
        )
        _atomic_torch(
            projection_path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "version": version,
                "shape": list(embedding.projection_weight.shape),
                "projection_weight": embedding.projection_weight.detach().cpu(),
            },
        )
        embedding_shards = [
            {
                "rank": shard_rank,
                **_artifact(directory / f"embedding_rank_{shard_rank:05d}.pt", directory),
            }
            for shard_rank in range(world_size)
        ]
        tracker_shards = [
            {
                "rank": shard_rank,
                **_artifact(directory / f"tracker_rank_{shard_rank:05d}.pt", directory),
            }
            for shard_rank in range(world_size)
        ]
        dense_record = _artifact(dense_path, directory)
        projection_record = _artifact(projection_path, directory)
        checkpoint_bytes = sum(
            int(value["bytes"])
            for value in [
                dense_record,
                projection_record,
                *embedding_shards,
                *tracker_shards,
            ]
        )
        manifest = {
            "schema": CHECKPOINT_SCHEMA,
            "protocol": PROTOCOL,
            "version": version,
            "world_size": world_size,
            "status": "complete",
            "scientific_result": False,
            "formal_result": False,
            "config_sha256": config_sha256,
            "geometry": geometry,
            "dense_config": asdict(dense.cfg),
            "replica_fingerprint": fingerprint,
            "dense": dense_record,
            "projection": projection_record,
            "embedding_shards": embedding_shards,
            "embedding_storage": embedding_storage,
            "embedding_delta_rows_by_rank": [int(value) for value in delta_rows_by_rank],
            "parent_version": version - 1 if embedding_storage == "sparse_delta" else None,
            "parent_manifest_sha256": file_sha256(_manifest_path(root, version - 1))
            if embedding_storage == "sparse_delta"
            else None,
            "tracker_shards": tracker_shards,
            "checkpoint_bytes": checkpoint_bytes,
            "optimizer_state": "fresh_per_update_no_persistent_state",
            "provenance": provenance,
        }
        _atomic_json(manifest_path, manifest)
    if dist.is_initialized():
        dist.barrier()
    return json.loads(manifest_path.read_text())


def _load_checkpoint(
    root: Path,
    version: int,
    dense,
    embedding,
    tracker,
    document: dict[str, Any],
    config_sha256: str,
    rank: int,
    verify_hash: bool = False,
) -> dict[str, Any]:
    manifest = _read_manifest(root, version, document, config_sha256, verify_hash)
    embedding_storage = manifest.get("embedding_storage", "full")
    if embedding_storage == "sparse_delta":
        parent_version = int(manifest.get("parent_version", -1))
        parent_path = _manifest_path(root, parent_version)
        if (
            parent_version != version - 1
            or not parent_path.is_file()
            or file_sha256(parent_path) != manifest.get("parent_manifest_sha256")
        ):
            raise ValueError("KuaiRand persistent embedding delta parent differs")
        _load_checkpoint(
            root,
            parent_version,
            dense,
            embedding,
            tracker,
            document,
            config_sha256,
            rank,
            verify_hash,
        )
    elif embedding_storage != "full":
        raise ValueError("KuaiRand persistent embedding storage differs")
    directory = root / f"theta_{version}"
    dense_path = _artifact_path(directory, manifest["dense"], verify_hash)
    projection_path = _artifact_path(directory, manifest["projection"], verify_hash)
    embedding_record = manifest["embedding_shards"][rank]
    tracker_record = manifest["tracker_shards"][rank]
    if embedding_record.get("rank") != rank or tracker_record.get("rank") != rank:
        raise ValueError("KuaiRand persistent shard rank differs")
    embedding_path = _artifact_path(directory, embedding_record, verify_hash)
    tracker_path = _artifact_path(directory, tracker_record, verify_hash)
    dense_payload = torch.load(dense_path, map_location="cpu", weights_only=True)
    projection_payload = torch.load(projection_path, map_location="cpu", weights_only=True)
    embedding_payload = torch.load(embedding_path, map_location="cpu", weights_only=True)
    tracker_payload = torch.load(tracker_path, map_location="cpu", weights_only=True)
    if (
        dense_payload.get("schema") != CHECKPOINT_SCHEMA
        or dense_payload.get("version") != version
        or dense_payload.get("config") != asdict(dense.cfg)
        or projection_payload.get("schema") != CHECKPOINT_SCHEMA
        or projection_payload.get("version") != version
        or embedding_payload.get("schema") != CHECKPOINT_SCHEMA
        or embedding_payload.get("version") != version
        or embedding_payload.get("rank") != rank
        or tracker_payload.get("schema") != CHECKPOINT_SCHEMA
        or tracker_payload.get("version") != version
        or tracker_payload.get("rank") != rank
    ):
        raise ValueError("KuaiRand persistent checkpoint payload differs")
    if embedding_payload.get("storage", "full") != embedding_storage:
        raise ValueError("KuaiRand persistent embedding payload storage differs")
    if embedding_storage == "full":
        local_weight = embedding_payload.get("local_weight")
        if not isinstance(local_weight, torch.Tensor) or local_weight.shape != embedding.local_weight.shape:
            raise ValueError("KuaiRand persistent full embedding payload differs")
    else:
        local_indices = embedding_payload.get("local_indices")
        local_values = embedding_payload.get("local_values")
        if (
            embedding_payload.get("parent_version") != version - 1
            or not isinstance(local_indices, torch.Tensor)
            or local_indices.dtype != torch.int64
            or local_indices.ndim != 1
            or not isinstance(local_values, torch.Tensor)
            or local_values.shape != (local_indices.numel(), embedding.embedding_width)
            or (
                local_indices.numel()
                and (
                    bool(torch.any(local_indices < 0))
                    or bool(torch.any(local_indices >= embedding.local_rows))
                    or not torch.equal(local_indices, torch.unique(local_indices, sorted=True))
                )
            )
        ):
            raise ValueError("KuaiRand persistent sparse embedding payload differs")
    dense.load_state_dict(dense_payload["state_dict"])
    with torch.no_grad():
        if embedding_storage == "full":
            embedding.local_weight.copy_(embedding_payload["local_weight"])
        elif local_indices.numel():
            embedding.local_weight.index_copy_(
                0,
                local_indices.to(device=embedding.local_weight.device),
                local_values.to(device=embedding.local_weight.device),
            )
        embedding.projection_weight.copy_(projection_payload["projection_weight"])
    tracker.load_activity(tracker_payload["local_bitmap"], tracker_payload["local_update_counts"])
    del dense_payload, projection_payload, embedding_payload, tracker_payload
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def _reset_bootstrap(
    dense,
    embedding,
    tracker,
    document: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    payload = torch.load(
        document["parent"]["theta0"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    source_state = payload["state_dict"]
    source_weight = source_state["item_emb.weight"]
    hidden = int(document["model"]["hidden_size"])
    generator = torch.Generator(device=device).manual_seed(
        int(document["model"]["embedding_seed"]) + rank * 1_000_003
    )
    with torch.no_grad():
        embedding.local_weight.normal_(
            mean=0.0,
            std=float(document["model"]["extra_embedding_std"]),
            generator=generator,
        )
        _copy_semantic_weight_to_embedding(
            embedding,
            source_weight,
            rank,
            world_size,
            device,
        )
        embedding.projection_weight.zero_()
        embedding.projection_weight[:, :hidden].copy_(torch.eye(hidden, device=device))
        if embedding.embedding_width > hidden:
            projection_generator = torch.Generator(device=device).manual_seed(
                int(document["model"]["projection_seed"])
            )
            embedding.projection_weight[:, hidden:].normal_(
                mean=0.0,
                std=float(document["model"]["extra_projection_std"]),
                generator=projection_generator,
            )
    dense_state = {name: value for name, value in source_state.items() if name != "item_emb.weight"}
    dense.core.load_state_dict(dense_state, strict=True)
    dense.core.query_mode = str(
        document["model"].get("query_mode", "history_only_zero")
    )
    tracker.local_bitmap.zero_()
    tracker.local_update_counts.zero_()
    del payload, source_state, source_weight, dense_state
    gc.collect()


def _training_document(document: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "training": {
            "calibration_examples": int(document["training"]["calibration_examples"]),
            "local_batch_size": int(document["training"]["local_batch_size"]),
            "weight_decay": float(document["training"]["weight_decay"]),
            **{
                key: value
                for key, value in candidate.items()
                if key not in ("evaluation_temperatures", "name")
            },
        }
    }


def _calibrate(
    dense,
    embedding,
    tracker,
    workload: dict[str, Any],
    base_config: dict[str, Any],
    document: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, Any]:
    training = {
        "name": "calibration",
        "update_epochs": 1,
        "maximum_update_examples": int(document["training"]["calibration_examples"]),
        "embedding_lr": float(document["training"]["calibration_embedding_lr"]),
        "projection_lr": float(document["training"]["calibration_projection_lr"]),
        "dense_lr": float(document["training"]["calibration_dense_lr"]),
        "dense_update_scope": str(
            document["training"].get("calibration_dense_update_scope", "full")
        ),
        "kv_lr": float(
            document["training"].get(
                "calibration_kv_lr",
                document["training"]["calibration_dense_lr"],
            )
        ),
    }
    calibration_document = _training_document(document, training)
    return _train_epochs(
        dense,
        embedding,
        tracker,
        workload["base_examples"],
        workload,
        base_config,
        calibration_document,
        rank,
        world_size,
        device,
        "calibration",
        int(document["training"]["seed"]) + 1009,
    )


def _build_workloads(
    document: dict[str, Any],
    base_config: dict[str, Any],
    rank: int,
    stop_after_version: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edge_documents = []
    workloads = []
    transitions = document["transitions"][:stop_after_version]
    for index, transition in enumerate(transitions):
        edge_document = _edge_config(base_config, transition, 1.0)
        if "update_dates" in transition:
            edge_document["data"]["update_dates"] = transition["update_dates"]
        edge_document["data"]["evaluation_targets_per_user"] = int(
            document["evaluation"]["targets_per_user"]
        )
        edge_document["data"]["user_limit"] = document["data"].get("user_limit")
        edge_document["evaluation"]["candidate_count"] = int(
            document["evaluation"]["candidate_count"]
        )
        workload = build_workload(edge_document)
        edge_documents.append(edge_document)
        workloads.append(workload)
        if rank == 0:
            metadata = workload["metadata"]
            print(
                f"phase=kuairand_persistent_workload version={index + 1} "
                f"users={metadata['selected_users']} "
                f"updates={metadata['update_examples']} "
                f"evaluations={metadata['evaluation_records']}",
                flush=True,
            )
    return edge_documents, workloads


def _passing(summary: dict[str, Any], document: dict[str, Any]) -> list[str]:
    stale = summary["comparisons"]["recompute_over_reuse"]
    threshold = float(document["selection"]["minimum_relative_percent"])
    return [
        metric
        for metric in document["selection"]["metrics"]
        if stale[metric]["relative_percent"] >= threshold
    ]


def _primary_value(summary: dict[str, Any], document: dict[str, Any]) -> float:
    metric = document["selection"]["metrics"][0]
    return float(summary["comparisons"]["recompute_over_reuse"][metric]["relative_percent"])


def _edge_admitted(summary: dict[str, Any], document: dict[str, Any]) -> bool:
    stale = summary["comparisons"]["recompute_over_reuse"]
    return bool(
        summary["sanity"]["passed"]
        and len(_passing(summary, document)) == len(document["selection"]["metrics"])
        and all(
            stale[metric]["relative_percent"] > 0
            for metric in document["selection"]["positive_metrics"]
        )
    )


def _admitted(
    summary: dict[str, Any],
    document: dict[str, Any],
    previous_primary_values: list[float],
) -> bool:
    if not _edge_admitted(summary, document):
        return False
    version = len(previous_primary_values) + 1
    for start, end in document["selection"]["groups"]:
        if version == end:
            group_values = previous_primary_values[start - 1 :] + [
                _primary_value(summary, document)
            ]
            return bool(
                np.mean(group_values)
                >= float(document["selection"]["minimum_group_mean_relative_percent"])
            )
    return True


def _accepted_primary_values(
    accepted_records: list[dict[str, Any]], document: dict[str, Any]
) -> list[float]:
    return [_primary_value(record["candidate"]["summary"], document) for record in accepted_records]


def _global_new_rows(tracker, before: torch.Tensor, device: torch.device) -> tuple[int, int]:
    local = int(torch.count_nonzero((tracker.local_bitmap != 0) & (before == 0)).item())
    value = torch.tensor(local, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(value)
    return int(value.item()), _active_rows(tracker, device)


def _broadcast(value: Any, rank: int) -> Any:
    if not dist.is_initialized():
        return value
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _accepted_path(output_root: Path, version: int) -> Path:
    return output_root / "edges" / f"theta_{version}" / "accepted.json"


def _repair_manifest_only_boundary(
    checkpoint_root: Path,
    output_root: Path,
    document: dict[str, Any],
    config_sha256: str,
    rank: int,
) -> None:
    if rank == 0:
        for version in range(1, int(document["checkpoint"]["versions"]) + 1):
            manifest_path = _manifest_path(checkpoint_root, version)
            accepted_path = _accepted_path(output_root, version)
            if accepted_path.is_file() or not manifest_path.is_file():
                continue
            manifest = _read_manifest(checkpoint_root, version, document, config_sha256, True)
            provenance = manifest["provenance"]
            candidate_record = provenance["accepted_candidate"]
            candidate_path = Path(candidate_record["path"])
            if (
                not candidate_path.is_file()
                or file_sha256(candidate_path) != candidate_record["sha256"]
            ):
                raise RuntimeError("KuaiRand persistent orphan checkpoint candidate differs")
            cell = json.loads(candidate_path.read_text())
            if (
                cell.get("protocol") != PROTOCOL
                or cell.get("version") != version
                or not cell.get("admitted")
            ):
                raise RuntimeError("KuaiRand persistent orphan checkpoint admission differs")
            accepted = {
                "protocol": PROTOCOL,
                "version": version,
                "source_version": version - 1,
                "status": "accepted",
                "candidate": {
                    "path": str(candidate_path),
                    "sha256": candidate_record["sha256"],
                    "candidate": cell["candidate"],
                    "summary": cell["summary"],
                    "passing_metrics": cell["passing_metrics"],
                    "admitted": True,
                    "new_optimizer_active_rows": cell["new_optimizer_active_rows"],
                    "cumulative_optimizer_active_rows": cell["cumulative_optimizer_active_rows"],
                },
                "checkpoint": {
                    "path": str(manifest_path),
                    "sha256": file_sha256(manifest_path),
                    "bytes": int(manifest["checkpoint_bytes"]),
                },
                "selection": document["selection"],
                "scientific_result": False,
                "formal_result": False,
                "recovered_after_manifest_commit": True,
            }
            _atomic_json(accepted_path, accepted)
    if dist.is_initialized():
        dist.barrier()


def _completed_prefix(
    checkpoint_root: Path,
    output_root: Path,
    document: dict[str, Any],
    config_sha256: str,
) -> int:
    minimum_retained_version = int(
        document.get("lineage_selection", {}).get("minimum_source_version", 1)
    )
    completed = minimum_retained_version - 1
    gap = False
    for version in range(1, int(document["checkpoint"]["versions"]) + 1):
        manifest_exists = _manifest_path(checkpoint_root, version).is_file()
        accepted_exists = _accepted_path(output_root, version).is_file()
        if version < minimum_retained_version:
            if manifest_exists:
                _read_manifest(checkpoint_root, version, document, config_sha256, False)
            continue
        if manifest_exists != accepted_exists:
            raise RuntimeError("KuaiRand persistent checkpoint/result boundary differs")
        if manifest_exists:
            if gap:
                raise RuntimeError("KuaiRand persistent checkpoint versions are noncontiguous")
            _read_manifest(checkpoint_root, version, document, config_sha256, False)
            completed = version
        else:
            gap = True
    return completed


def _disk_preflight(
    document: dict[str, Any], checkpoint_root: Path, completed: int
) -> dict[str, int]:
    usage = shutil.disk_usage(checkpoint_root.parent)
    final_version = int(document["checkpoint"]["versions"])
    required_checkpoint_bytes = sum(
        _expected_checkpoint_bytes_for_version(document, version)
        for version in range(completed + 1, final_version + 1)
    )
    bootstrap_remaining = bool(
        document["checkpoint"].get("retain_bootstrap", False)
        and not _manifest_path(checkpoint_root, 0).is_file()
    )
    if bootstrap_remaining:
        required_checkpoint_bytes += _expected_checkpoint_bytes_for_version(document, 0)
    remaining = final_version - completed
    required = required_checkpoint_bytes + int(document["checkpoint"]["write_reserve_bytes"])
    if completed == 0:
        required = max(required, int(document["checkpoint"]["minimum_free_bytes"]))
    if usage.free < required:
        raise RuntimeError("KuaiRand persistent checkpoint disk preflight failed")
    return {
        "free_bytes": usage.free,
        "required_remaining_bytes": required,
        "remaining_versions": remaining + int(bootstrap_remaining),
    }


def _candidate_sequence(document: dict[str, Any], version: int) -> list[dict[str, Any]]:
    if version == 1:
        return [document["training"]["initial_candidate"]]
    candidates = document["training"]["candidate_ladder"]
    scheduled = document["training"].get("candidate_schedule", {}).get(str(version))
    if scheduled is not None:
        return [candidate for candidate in candidates if candidate["name"] == scheduled]
    transition = document["transitions"][version - 1]
    if len(transition.get("update_dates", [transition["update_date"]])) > 1:
        return sorted(
            candidates,
            key=lambda value: (
                value.get("temporal_training", "pooled") != "sequential_dates",
                candidates.index(value),
            ),
        )
    return candidates


def _train_candidate(
    dense,
    embedding,
    tracker,
    workload: dict[str, Any],
    base_config: dict[str, Any],
    candidate_document: dict[str, Any],
    candidate: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    training_base_config = _temperature_edge_document(
        base_config,
        float(candidate["training_temperature"]) if "training_temperature" in candidate else None,
    )
    if candidate.get("temporal_training", "pooled") == "pooled":
        return _train_epochs(
            dense,
            embedding,
            tracker,
            workload["update_examples"],
            workload,
            training_base_config,
            candidate_document,
            rank,
            world_size,
            device,
            "update",
            seed,
        )
    available_dates = workload["metadata"]["update_dates"]
    latest_update_dates = int(candidate.get("latest_update_dates", len(available_dates)))
    if latest_update_dates > len(available_dates):
        raise RuntimeError("KuaiRand latest update-date scope exceeds the transition")
    selected_dates = available_dates[-latest_update_dates:]
    stages = []
    for date in selected_dates:
        index = available_dates.index(date)
        examples = workload["update_examples_by_date"][date]
        if not examples:
            continue
        stages.append(
            {
                "date": date,
                "result": _train_epochs(
                    dense,
                    embedding,
                    tracker,
                    examples,
                    workload,
                    training_base_config,
                    candidate_document,
                    rank,
                    world_size,
                    device,
                    f"update_{date}",
                    seed + index * 100003,
                ),
            }
        )
    if len(stages) != len(selected_dates):
        raise RuntimeError("KuaiRand sequential update stage differs")
    return {
        "phase": "update",
        "temporal_training": "sequential_dates",
        "available_update_dates": available_dates,
        "selected_update_dates": selected_dates,
        "stages": stages,
        "unique_examples": sum(stage["result"]["unique_examples"] for stage in stages),
        "processed_examples": sum(stage["result"]["processed_examples"] for stage in stages),
        "minimum_active_embedding_dimensions": min(
            stage["result"]["minimum_active_embedding_dimensions"] for stage in stages
        ),
    }


def _candidate_path(
    output_root: Path,
    version: int,
    candidate_index: int,
    candidate: dict[str, Any],
) -> Path:
    return (
        output_root
        / "edges"
        / f"theta_{version}"
        / "candidates"
        / f"{candidate_index:02d}_{candidate['name']}.json"
    )


def _cached_failed_candidate(
    output_root: Path,
    version: int,
    candidate_index: int,
    candidate: dict[str, Any],
    transition: dict[str, Any],
    document: dict[str, Any],
    previous_primary_values: list[float],
) -> bool:
    path = _candidate_path(output_root, version, candidate_index, candidate)
    if not path.is_file():
        return False
    cell = json.loads(path.read_text())
    if (
        cell.get("protocol") != PROTOCOL
        or cell.get("version") != version
        or cell.get("source_version") != version - 1
        or cell.get("transition") != transition
        or cell.get("candidate_index") != candidate_index
        or cell.get("candidate") != candidate
        or cell.get("scientific_result") is not False
        or cell.get("formal_result") is not False
    ):
        raise RuntimeError("KuaiRand cached failed candidate differs")
    if document["training"].get("checkpoint_policy") == "fixed_schedule":
        return not bool(cell["summary"]["sanity"]["passed"])
    return not _admitted(cell["summary"], document, previous_primary_values)


def _train_missing_versions(
    dense,
    embedding,
    tracker,
    geometry: dict[str, Any],
    workloads: list[dict[str, Any]],
    edge_documents: list[dict[str, Any]],
    base_config: dict[str, Any],
    document: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    checkpoint_root: Path,
    output_root: Path,
    completed: int,
    stop_after_version: int | None,
    candidate_priority: str | None,
    rank: int,
    world_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    accepted_records = []
    for version in range(1, completed + 1):
        accepted_records.append(json.loads(_accepted_path(output_root, version).read_text()))
    if completed:
        _load_checkpoint(
            checkpoint_root,
            completed,
            dense,
            embedding,
            tracker,
            document,
            config_sha256,
            rank,
        )
    configured_final_version = int(document["checkpoint"]["versions"])
    final_version = configured_final_version if stop_after_version is None else stop_after_version
    for version in range(completed + 1, final_version + 1):
        edge_index = version - 1
        workload = workloads[edge_index]
        edge_document = edge_documents[edge_index]
        batches = _evaluation_batches(
            workload,
            int(document["evaluation"]["local_batch_size"]),
            rank,
            world_size,
        )
        captured = _capture_old(dense, embedding, batches, workload, base_config, device)
        del batches
        accepted = None
        previous_primary_values = _accepted_primary_values(accepted_records, document)
        candidate_sequence = _candidate_sequence(document, version)
        if candidate_priority is not None:
            candidate_sequence = [
                candidate
                for candidate in candidate_sequence
                if candidate["name"] == candidate_priority
            ]
            if len(candidate_sequence) != 1:
                raise ValueError("KuaiRand candidate priority differs")
        for candidate_index, candidate in enumerate(candidate_sequence):
            if rank == 0:
                cached_failed = _cached_failed_candidate(
                    output_root,
                    version,
                    candidate_index,
                    candidate,
                    document["transitions"][edge_index],
                    document,
                    previous_primary_values,
                )
            else:
                cached_failed = None
            cached_failed = bool(_broadcast(cached_failed, rank))
            if cached_failed:
                if rank == 0:
                    print(
                        f"phase=kuairand_persistent_candidate_cached "
                        f"version={version} candidate={candidate['name']}",
                        flush=True,
                    )
                continue
            if candidate_index:
                if version == 1:
                    _reset_bootstrap(
                        dense,
                        embedding,
                        tracker,
                        document,
                        rank,
                        world_size,
                        device,
                    )
                    _calibrate(
                        dense,
                        embedding,
                        tracker,
                        workloads[0],
                        base_config,
                        document,
                        rank,
                        world_size,
                        device,
                    )
                else:
                    _load_checkpoint(
                        checkpoint_root,
                        version - 1,
                        dense,
                        embedding,
                        tracker,
                        document,
                        config_sha256,
                        rank,
                    )
            before = tracker.local_bitmap.clone()
            sparse_checkpoint = (
                _embedding_storage_for_version(document, version) == "sparse_delta"
            )
            counts_before = tracker.local_update_counts.clone() if sparse_checkpoint else None
            candidate_document = _training_document(document, candidate)
            training = _train_candidate(
                dense,
                embedding,
                tracker,
                workload,
                base_config,
                candidate_document,
                candidate,
                rank,
                world_size,
                device,
                int(document["training"]["seed"]) + 2003 + edge_index * 100003,
            )
            compact, evaluation = _evaluate_captured(
                dense,
                embedding,
                captured,
                workload,
                edge_document,
                rank,
                world_size,
                device,
            )
            new_rows, cumulative_rows = _global_new_rows(tracker, before, device)
            del before
            embedding_delta_rows = (
                torch.nonzero(
                    tracker.local_update_counts != counts_before,
                    as_tuple=False,
                ).flatten()
                if counts_before is not None
                else None
            )
            del counts_before
            if rank == 0:
                assert compact is not None and evaluation is not None
                quality_admitted = _admitted(compact, document, previous_primary_values)
                admitted = quality_admitted
                if document["training"].get("checkpoint_policy") == "fixed_schedule":
                    admitted = bool(compact["sanity"]["passed"])
                stale = compact["comparisons"]["recompute_over_reuse"]
                cell = {
                    "protocol": PROTOCOL,
                    "version": version,
                    "source_version": version - 1,
                    "transition": document["transitions"][edge_index],
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "training": training,
                    "workload": workload["metadata"],
                    "summary": compact,
                    "passing_metrics": _passing(compact, document),
                    "admitted": admitted,
                    "quality_admitted": quality_admitted,
                    "new_optimizer_active_rows": new_rows,
                    "cumulative_optimizer_active_rows": cumulative_rows,
                    "scientific_result": False,
                    "formal_result": False,
                }
                if "lineage_selection" in document:
                    selection = document["lineage_selection"]
                    cell["partition_summaries"] = {
                        partition: _lineage_partition_summary(
                            evaluation,
                            edge_document,
                            int(selection["split_seed"]),
                            float(selection["tuning_fraction"]),
                            int(selection["tuning_bootstrap_samples"]),
                            partition,
                        )
                        for partition in ("tuning", "holdout")
                    }
                candidate_path = _candidate_path(output_root, version, candidate_index, candidate)
                _atomic_json(candidate_path, cell)
                candidate_record = {
                    "path": str(candidate_path),
                    "sha256": file_sha256(candidate_path),
                    "candidate": candidate,
                    "summary": compact,
                    "passing_metrics": cell["passing_metrics"],
                    "admitted": admitted,
                    "quality_admitted": quality_admitted,
                    "new_optimizer_active_rows": new_rows,
                    "cumulative_optimizer_active_rows": cumulative_rows,
                }
                if "partition_summaries" in cell:
                    candidate_record["partition_summaries"] = cell["partition_summaries"]
                print(
                    f"phase=kuairand_persistent_candidate version={version} "
                    f"candidate={candidate['name']} "
                    f"mrr={stale['mrr']['relative_percent']:.3f}% "
                    f"ndcg5={stale['ndcg_at_5']['relative_percent']:.3f}% "
                    f"hr5={stale['hit_rate_at_5']['relative_percent']:.3f}% "
                    f"quality_admitted={quality_admitted} "
                    f"checkpoint_admitted={admitted}",
                    flush=True,
                )
            else:
                candidate_record = None
                admitted = None
            candidate_record = _broadcast(candidate_record, rank)
            admitted = bool(candidate_record["admitted"])
            if admitted:
                provenance = {
                    "round_id": document["round_id"],
                    "config": {
                        "path": str(config_path),
                        "sha256": config_sha256,
                    },
                    "source_version": version - 1,
                    "transition": document["transitions"][edge_index],
                    "accepted_candidate": {
                        "path": candidate_record["path"],
                        "sha256": candidate_record["sha256"],
                        "name": candidate["name"],
                    },
                    "checkpoint_policy": document["training"].get(
                        "checkpoint_policy", "quality_gate"
                    ),
                }
                manifest = _save_checkpoint(
                    checkpoint_root,
                    version,
                    dense,
                    embedding,
                    tracker,
                    geometry,
                    document,
                    config_sha256,
                    provenance,
                    rank,
                    world_size,
                    embedding_delta_rows,
                )
                if rank == 0:
                    accepted = {
                        "protocol": PROTOCOL,
                        "version": version,
                        "source_version": version - 1,
                        "status": "accepted",
                        "candidate": candidate_record,
                        "checkpoint": {
                            "path": str(_manifest_path(checkpoint_root, version)),
                            "sha256": file_sha256(_manifest_path(checkpoint_root, version)),
                            "bytes": int(manifest["checkpoint_bytes"]),
                        },
                        "selection": document["selection"],
                        "checkpoint_policy": document["training"].get(
                            "checkpoint_policy", "quality_gate"
                        ),
                        "scientific_result": False,
                        "formal_result": False,
                    }
                    _atomic_json(_accepted_path(output_root, version), accepted)
                accepted = _broadcast(accepted, rank)
                accepted_records.append(accepted)
                break
        del captured
        if "embedding_delta_rows" in locals():
            del embedding_delta_rows
        gc.collect()
        torch.cuda.empty_cache()
        if accepted is None:
            raise RuntimeError(f"KuaiRand theta{version} has no candidate above the 1% gate")
    return accepted_records


def _reset_and_calibrate(
    dense,
    embedding,
    tracker,
    workloads: list[dict[str, Any]],
    base_config: dict[str, Any],
    document: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    _reset_bootstrap(dense, embedding, tracker, document, rank, world_size, device)
    _calibrate(
        dense,
        embedding,
        tracker,
        workloads[0],
        base_config,
        document,
        rank,
        world_size,
        device,
    )


def _lineage_markdown(result: dict[str, Any]) -> str:
    use_holdout = any(
        "holdout" in lineage for target in result["targets"] for lineage in target["lineage"]
    )
    split_label = "held-out users" if use_holdout else "all users"

    def selected_summary(lineage: dict[str, Any]) -> dict[str, Any]:
        return lineage.get("holdout", lineage["summary"])

    final_version = len(result["targets"])
    retained_bootstrap = "bootstrap_checkpoint" in result
    displayed_versions = final_version + int(retained_bootstrap)
    lines = [
        (
            f"# KuaiRand theta0–theta{final_version} Reuse loss"
            if retained_bootstrap
            else f"# KuaiRand theta1–theta{final_version} Reuse loss"
        ),
        "",
        f"Primary table split: {split_label}.",
        "",
        "## Adjacent-version measurements",
        "",
        "| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for target in result["targets"]:
        adjacent = target["lineage"][-1]
        summary = selected_summary(adjacent)
        stale = summary["comparisons"]["recompute_over_reuse"]
        accepted = result["accepted_versions"][target["target_version"] - 1]
        config_transition = target["transition"]
        update_label = "+".join(
            config_transition.get("update_dates", [config_transition["update_date"]])
        )
        mrr_ci = stale["mrr"]["user_bootstrap_95"]
        ndcg_ci = stale["ndcg_at_5"]["user_bootstrap_95"]
        lines.append(
            "| theta{target} | theta{source} | {update}→{evaluation} | {candidate} | {mrr:+.3f}% | {ndcg:+.3f}% | {hr:+.3f}% | [{mrr_low:+.5f}, {mrr_high:+.5f}] | [{ndcg_low:+.5f}, {ndcg_high:+.5f}] |".format(
                target=target["target_version"],
                source=adjacent["source_version"],
                update=update_label,
                evaluation=config_transition["evaluation_date"],
                candidate=accepted["candidate"]["candidate"]["name"],
                mrr=stale["mrr"]["relative_percent"],
                ndcg=stale["ndcg_at_5"]["relative_percent"],
                hr=stale["hit_rate_at_5"]["relative_percent"],
                mrr_low=mrr_ci["lower"],
                mrr_high=mrr_ci["upper"],
                ndcg_low=ndcg_ci["lower"],
                ndcg_high=ndcg_ci["upper"],
            )
        )
    metric_titles = (
        ("ndcg_at_5", "NDCG@5 relative Recompute-over-Reuse"),
        ("mrr", "MRR relative Recompute-over-Reuse"),
        ("hit_rate_at_5", "HR@5 relative Recompute-over-Reuse"),
    )
    cache_versions = list(
        range(final_version + 1 if retained_bootstrap else final_version)
    )
    for metric, title in metric_titles:
        lines.extend(
            [
                "",
                f"## {displayed_versions}x{displayed_versions} triangular matrix: {title}",
                "",
                "Positive means Recompute is better; negative means Reuse has the higher point estimate.",
                "",
                "| current \\ cache | "
                + " | ".join(f"theta{version}" for version in cache_versions)
                + " |",
                "|---|" + "---:|" * len(cache_versions),
            ]
        )
        if retained_bootstrap:
            lines.append(
                "| theta0 | "
                + " | ".join(
                    "+0.000%" if version == 0 else "—"
                    for version in cache_versions
                )
                + " |"
            )
        for target in result["targets"]:
            target_version = int(target["target_version"])
            by_source = {int(lineage["source_version"]): lineage for lineage in target["lineage"]}
            cells = []
            for source_version in cache_versions:
                lineage = by_source.get(source_version)
                if retained_bootstrap and source_version == target_version:
                    cells.append("+0.000%")
                elif lineage is None:
                    cells.append("—")
                else:
                    value = selected_summary(lineage)["comparisons"]["recompute_over_reuse"][
                        metric
                    ]["relative_percent"]
                    cells.append(f"{value:+.3f}%")
            lines.append(f"| theta{target_version} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Direct cache-age matrix",
            "",
            "| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target in result["targets"]:
        for lineage in target["lineage"]:
            summary = selected_summary(lineage)
            stale = summary["comparisons"]["recompute_over_reuse"]
            endpoints = summary["endpoints"]
            lines.append(
                "| theta{target} | theta{source} | {age} | {reuse_mrr:.6f} | {fresh_mrr:.6f} | {mrr:+.3f}% | {reuse_ndcg:.6f} | {fresh_ndcg:.6f} | {ndcg:+.3f}% | {hr:+.3f}% |".format(
                    target=target["target_version"],
                    source=lineage["source_version"],
                    age=lineage["cache_age"],
                    reuse_mrr=endpoints["reuse"]["mrr"],
                    fresh_mrr=endpoints["recompute"]["mrr"],
                    mrr=stale["mrr"]["relative_percent"],
                    reuse_ndcg=endpoints["reuse"]["ndcg_at_5"],
                    fresh_ndcg=endpoints["recompute"]["ndcg_at_5"],
                    ndcg=stale["ndcg_at_5"]["relative_percent"],
                    hr=stale["hit_rate_at_5"]["relative_percent"],
                )
            )
    lines.extend(
        [
            "",
            "All values are development measurements. Direct age-k is not recursive mixed-version append lineage.",
            "",
        ]
    )
    return "\n".join(lines)


def render_persistent_reuse_loss_table(
    result_path: str | Path,
) -> dict[str, Any]:
    path = Path(result_path)
    result = json.loads(path.read_text())
    if (
        result.get("status") != "complete"
        or not result.get("targets")
        or len(result["targets"]) != int(result.get("checkpoint_count", -1))
    ):
        raise ValueError("KuaiRand persistent result differs")
    table_path = Path(
        result.get("reuse_loss_table", {}).get("path", path.parent / "reuse_loss_table.md")
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = table_path.with_suffix(table_path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(_lineage_markdown(result))
    os.replace(temporary, table_path)
    result["reuse_loss_table"] = {
        "path": str(table_path),
        "sha256": file_sha256(table_path),
    }
    _atomic_json(path, result)
    return result["reuse_loss_table"]


def _direct_lineage(
    dense,
    embedding,
    tracker,
    workloads: list[dict[str, Any]],
    edge_documents: list[dict[str, Any]],
    base_config: dict[str, Any],
    document: dict[str, Any],
    config_sha256: str,
    checkpoint_root: Path,
    output_root: Path,
    accepted_records: list[dict[str, Any]],
    geometry: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, Any]:
    result_path = output_root / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    final_version = int(document["checkpoint"]["versions"])
    targets = []
    source_chunk_size = int(
        document["execution"].get("lineage_source_chunk_size", 1)
    )
    for target_version in range(1, final_version + 1):
        target_index = target_version - 1
        target_path = output_root / "lineage" / f"theta_{target_version}.json"
        if rank == 0 and target_path.is_file():
            cached_target = json.loads(target_path.read_text())
            cached_lineage = cached_target.get("lineage")
            if (
                cached_target.get("target_version") != target_version
                or cached_target.get("transition") != document["transitions"][target_index]
                or not isinstance(cached_lineage, list)
                or len(cached_lineage) != target_version
                or [row.get("source_version") for row in cached_lineage]
                != list(range(target_version))
                or any(
                    row.get("target_version") != target_version
                    or row.get("cache_age") != target_version - row.get("source_version", -1)
                    or not isinstance(row.get("summary"), dict)
                    for row in cached_lineage
                )
                or "chain_admitted" not in cached_lineage[-1]
            ):
                raise RuntimeError("KuaiRand persistent cached lineage target differs")
            accepted_summary = accepted_records[target_index]["candidate"]["summary"]
            if target_version > int(document["checkpoint"].get("imported_prefix_versions", 0)):
                for metric in ("mrr", "ndcg_at_5", "hit_rate_at_5"):
                    observed = cached_lineage[-1]["summary"]["comparisons"][
                        "recompute_over_reuse"
                    ][metric]["relative_percent"]
                    expected = accepted_summary["comparisons"]["recompute_over_reuse"][metric][
                        "relative_percent"
                    ]
                    if not np.isclose(observed, expected, rtol=0, atol=1e-6):
                        raise RuntimeError(
                            "KuaiRand persistent cached lineage checkpoint replay differs"
                        )
            cached_target = cached_target | {
                "path": str(target_path),
                "sha256": file_sha256(target_path),
            }
        else:
            cached_target = None
        cached_target = _broadcast(cached_target, rank)
        if cached_target is not None:
            if rank == 0:
                targets.append(cached_target)
                print(
                    f"phase=kuairand_persistent_lineage_cached target={target_version} "
                    f"sources={target_version}",
                    flush=True,
                )
            continue
        evaluation_temperature = accepted_records[target_index]["candidate"].get(
            "evaluation_temperature"
        )
        lineage_rows = []
        for source_start in range(0, target_version, source_chunk_size):
            source_end = min(target_version, source_start + source_chunk_size)
            captured_chunk = {}
            for source_version in range(source_start, source_end):
                if source_version == 0:
                    _reset_and_calibrate(
                        dense,
                        embedding,
                        tracker,
                        workloads,
                        base_config,
                        document,
                        rank,
                        world_size,
                        device,
                    )
                else:
                    _load_checkpoint(
                        checkpoint_root,
                        source_version,
                        dense,
                        embedding,
                        tracker,
                        document,
                        config_sha256,
                        rank,
                    )
                batches = _evaluation_batches(
                    workloads[target_index],
                    int(document["evaluation"]["local_batch_size"]),
                    rank,
                    world_size,
                )
                captured_chunk[source_version] = _capture_old(
                    dense,
                    embedding,
                    batches,
                    workloads[target_index],
                    base_config,
                    device,
                )
                del batches
            _load_checkpoint(
                checkpoint_root,
                target_version,
                dense,
                embedding,
                tracker,
                document,
                config_sha256,
                rank,
            )
            for source_version in sorted(list(captured_chunk)):
                captured = captured_chunk.pop(source_version)
                compact, evaluation = _evaluate_captured(
                    dense,
                    embedding,
                    captured,
                    workloads[target_index],
                    _temperature_edge_document(
                        edge_documents[target_index], evaluation_temperature
                    ),
                    rank,
                    world_size,
                    device,
                )
                if rank == 0:
                    assert compact is not None and evaluation is not None
                    lineage_row = {
                        "source_version": source_version,
                        "target_version": target_version,
                        "cache_age": target_version - source_version,
                        "summary": compact,
                        "above_one_percent": _edge_admitted(compact, document),
                    }
                    if "lineage_selection" in document:
                        selection = document["lineage_selection"]
                        lineage_row["holdout"] = _lineage_holdout_summary(
                            evaluation,
                            _temperature_edge_document(
                                edge_documents[target_index], evaluation_temperature
                            ),
                            int(selection["split_seed"]),
                            float(selection["tuning_fraction"]),
                            int(selection["tuning_bootstrap_samples"]),
                        )
                    lineage_rows.append(lineage_row)
                del captured, compact, evaluation
            if rank == 0:
                print(
                    f"phase=kuairand_persistent_lineage_chunk "
                    f"target={target_version} sources={source_start}-{source_end - 1}",
                    flush=True,
                )
            del captured_chunk
            gc.collect()
            torch.cuda.empty_cache()
        if rank == 0:
            adjacent = lineage_rows[-1]["summary"]
            accepted_summary = accepted_records[target_index]["candidate"]["summary"]
            if target_version > int(document["checkpoint"].get("imported_prefix_versions", 0)):
                for metric in ("mrr", "ndcg_at_5", "hit_rate_at_5"):
                    observed = adjacent["comparisons"]["recompute_over_reuse"][metric][
                        "relative_percent"
                    ]
                    expected = accepted_summary["comparisons"]["recompute_over_reuse"][metric][
                        "relative_percent"
                    ]
                    if not np.isclose(observed, expected, rtol=0, atol=1e-6):
                        raise RuntimeError("KuaiRand persistent adjacent checkpoint replay differs")
            else:
                lineage_rows[-1]["imported_anchor_replay"] = True
            lineage_rows[-1]["chain_admitted"] = _admitted(
                adjacent,
                document,
                _accepted_primary_values(accepted_records[:target_index], document),
            )
            target = {
                "target_version": target_version,
                "transition": document["transitions"][target_index],
                "lineage": lineage_rows,
            }
            _atomic_json(target_path, target)
            targets.append(target | {"path": str(target_path), "sha256": file_sha256(target_path)})
            print(
                f"phase=kuairand_persistent_lineage target={target_version} "
                f"sources={len(lineage_rows)}",
                flush=True,
            )
        del lineage_rows
        gc.collect()
    if rank == 0:
        checkpoint_records = []
        checkpoint_bytes = 0
        bootstrap_checkpoint = None
        if document["checkpoint"].get("retain_bootstrap", False):
            bootstrap_manifest_path = _manifest_path(checkpoint_root, 0)
            bootstrap_manifest = _read_manifest(
                checkpoint_root, 0, document, config_sha256, False
            )
            checkpoint_bytes += int(bootstrap_manifest["checkpoint_bytes"])
            bootstrap_checkpoint = {
                "version": 0,
                "path": str(bootstrap_manifest_path),
                "sha256": file_sha256(bootstrap_manifest_path),
                "bytes": int(bootstrap_manifest["checkpoint_bytes"]),
            }
        for version in range(1, final_version + 1):
            manifest_path = _manifest_path(checkpoint_root, version)
            manifest = json.loads(manifest_path.read_text())
            checkpoint_bytes += int(manifest["checkpoint_bytes"])
            checkpoint_records.append(
                {
                    "version": version,
                    "path": str(manifest_path),
                    "sha256": file_sha256(manifest_path),
                    "bytes": int(manifest["checkpoint_bytes"]),
                }
            )
        adjacent_pass = all(target["lineage"][-1]["chain_admitted"] for target in targets)
        oldest_stronger = sum(
            target["lineage"][0]["summary"]["comparisons"]["recompute_over_reuse"]["mrr"][
                "relative_percent"
            ]
            > target["lineage"][-1]["summary"]["comparisons"]["recompute_over_reuse"]["mrr"][
                "relative_percent"
            ]
            for target in targets[1:]
        )
        ordinary_holdout = [
            lineage
            for target in targets[1:]
            for lineage in target["lineage"]
            if lineage["source_version"] >= 1 and "holdout" in lineage
        ]
        holdout_metrics = ("mrr", "ndcg_at_5", "hit_rate_at_5")
        holdout_positive = {
            metric: sum(
                lineage["holdout"]["comparisons"]["recompute_over_reuse"][metric][
                    "relative_percent"
                ]
                > 0
                for lineage in ordinary_holdout
            )
            for metric in holdout_metrics
        }
        holdout_gate = "lineage_selection" not in document or bool(
            ordinary_holdout
            and all(count == len(ordinary_holdout) for count in holdout_positive.values())
        )
        result = {
            "protocol": PROTOCOL,
            "round_id": document["round_id"],
            "status": "complete",
            "scientific_result": False,
            "formal_result": False,
            "config": {
                "path": str(document["config_path"]),
                "sha256": config_sha256,
            },
            "geometry": geometry,
            "accepted_versions": accepted_records,
            "checkpoints": checkpoint_records,
            "checkpoint_count": len(checkpoint_records),
            "retained_checkpoint_count": len(checkpoint_records)
            + int(bootstrap_checkpoint is not None),
            "checkpoint_bytes": checkpoint_bytes,
            "imported_prefix_versions": int(
                document["checkpoint"].get("imported_prefix_versions", 0)
            ),
            "targets": targets,
            "decision": {
                "all_adjacent_edges_above_one_percent": adjacent_pass,
                "oldest_cache_mrr_stronger_than_adjacent_edges": oldest_stronger,
                "oldest_cache_comparisons": len(targets) - 1,
                "direct_age_accumulation_observed": oldest_stronger >= max(1, len(targets) - 2),
                "ordinary_holdout_cells": len(ordinary_holdout),
                "positive_ordinary_holdout_cells": holdout_positive,
                "all_ordinary_holdout_ranking_positive": holdout_gate,
                "next": f"freeze_theta1_theta{final_version}_chain"
                if adjacent_pass and holdout_gate
                else "revise_failed_holdout_or_adjacent_edge",
            },
        }
        if bootstrap_checkpoint is not None:
            result["bootstrap_checkpoint"] = bootstrap_checkpoint
        table_path = output_root / "reuse_loss_table.md"
        _atomic_json(result_path, result)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = table_path.with_suffix(table_path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(_lineage_markdown(result))
        os.replace(temporary, table_path)
        result["reuse_loss_table"] = {
            "path": str(table_path),
            "sha256": file_sha256(table_path),
        }
        _atomic_json(result_path, result)
    else:
        result = None
    return _broadcast(result, rank)


def preflight_persistent_chain(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    document = load_persistent_config(path)
    config_sha256 = file_sha256(path)
    checkpoint_root = Path(document["outputs"]["checkpoint_root"])
    output_root = Path(document["outputs"]["root"])
    checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(checkpoint_root.parent)
    completed = _completed_prefix(checkpoint_root, output_root, document, config_sha256)
    remaining = int(document["checkpoint"]["versions"]) - completed
    bootstrap_remaining = bool(
        document["checkpoint"].get("retain_bootstrap", False)
        and not _manifest_path(checkpoint_root, 0).is_file()
    )
    required = sum(
        _expected_checkpoint_bytes_for_version(document, version)
        for version in range(completed + 1, int(document["checkpoint"]["versions"]) + 1)
    ) + (
        _expected_checkpoint_bytes_for_version(document, 0)
        if bootstrap_remaining
        else 0
    ) + int(
        document["checkpoint"]["write_reserve_bytes"]
    )
    if completed == 0:
        required = max(required, int(document["checkpoint"]["minimum_free_bytes"]))
    if usage.free < required:
        raise RuntimeError("KuaiRand persistent preflight free disk differs")
    return {
        "status": "ready",
        "config_sha256": config_sha256,
        "completed_versions": completed,
        "remaining_versions": remaining + int(bootstrap_remaining),
        "free_bytes": usage.free,
        "required_free_bytes": required,
        "world_size": int(document["execution"]["world_size"]),
        "checkpoint_versions": int(document["checkpoint"]["versions"]),
    }


def run_candidate_probe(
    config_path: str | Path,
    version: int,
    candidate_name: str,
    output_path: str | Path,
    candidate_config_path: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    path = Path(config_path)
    document = load_persistent_config(path)
    document["config_path"] = str(path)
    config_sha256 = file_sha256(path)
    rank, world_size, device = _distributed(document)
    _seed(int(document["training"]["seed"]))
    output = Path(output_path)
    if output.is_file():
        result = json.loads(output.read_text()) if rank == 0 else None
        result = _broadcast(result, rank)
        if dist.is_initialized():
            dist.destroy_process_group()
        return result
    if not 2 <= version <= int(document["checkpoint"]["versions"]):
        raise ValueError("KuaiRand candidate probe version differs")
    if candidate_config_path is None:
        candidates = document["training"]["candidate_ladder"]
        candidate_config = None
    else:
        candidate_config_document = load_candidate_probe_config(candidate_config_path, path)
        candidates = candidate_config_document["candidates"]
        candidate_config = {
            "path": str(candidate_config_path),
            "sha256": file_sha256(candidate_config_path),
        }
    matches = [value for value in candidates if value["name"] == candidate_name]
    if len(matches) != 1:
        raise ValueError("KuaiRand candidate probe name differs")
    candidate = matches[0]
    transition = document["transitions"][version - 1]
    base_config = load_config(document["parent"]["base_config"]["path"])
    edge_document = _edge_config(base_config, transition, 1.0)
    if "update_dates" in transition:
        edge_document["data"]["update_dates"] = transition["update_dates"]
    edge_document["data"]["evaluation_targets_per_user"] = int(
        document["evaluation"]["targets_per_user"]
    )
    edge_document["data"]["user_limit"] = document["data"].get("user_limit")
    edge_document["evaluation"]["candidate_count"] = int(document["evaluation"]["candidate_count"])
    workload = build_workload(edge_document)
    embedding_rows = int(workload["metadata"]["embedding_rows"])
    dense, embedding, tracker, geometry = _initialize_model(
        document, base_config, embedding_rows, rank, world_size, device
    )
    checkpoint_root = Path(document["outputs"]["checkpoint_root"])
    source_manifest = _load_checkpoint(
        checkpoint_root,
        version - 1,
        dense,
        embedding,
        tracker,
        document,
        config_sha256,
        rank,
    )
    batches = _evaluation_batches(
        workload,
        int(document["evaluation"]["local_batch_size"]),
        rank,
        world_size,
    )
    captured = _capture_old(dense, embedding, batches, workload, base_config, device)
    del batches
    before = tracker.local_bitmap.clone()
    training = _train_candidate(
        dense,
        embedding,
        tracker,
        workload,
        base_config,
        _training_document(document, candidate),
        candidate,
        rank,
        world_size,
        device,
        int(document["training"]["seed"]) + 2003 + (version - 1) * 100003,
    )
    compact, evaluation = _evaluate_captured(
        dense,
        embedding,
        captured,
        workload,
        edge_document,
        rank,
        world_size,
        device,
    )
    new_rows, cumulative_rows = _global_new_rows(tracker, before, device)
    if rank == 0:
        assert compact is not None and evaluation is not None
        output_root = Path(document["outputs"]["root"])
        accepted_records = [
            json.loads(_accepted_path(output_root, value).read_text())
            for value in range(1, version)
        ]
        result = {
            "protocol": f"{PROTOCOL}_candidate_probe_v0",
            "status": "complete",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(path), "sha256": config_sha256},
            "candidate_config": candidate_config,
            "version": version,
            "source_version": version - 1,
            "source_checkpoint": {
                "path": str(_manifest_path(checkpoint_root, version - 1)),
                "sha256": file_sha256(_manifest_path(checkpoint_root, version - 1)),
                "checkpoint_bytes": int(source_manifest["checkpoint_bytes"]),
            },
            "transition": transition,
            "candidate": candidate,
            "training": training,
            "workload": workload["metadata"],
            "summary": compact,
            "passing_metrics": _passing(compact, document),
            "would_admit": _admitted(
                compact,
                document,
                _accepted_primary_values(accepted_records, document),
            ),
            "new_optimizer_active_rows": new_rows,
            "cumulative_optimizer_active_rows": cumulative_rows,
            "geometry": geometry,
            "elapsed_seconds": time.monotonic() - started,
        }
        if "lineage_selection" in document:
            selection = document["lineage_selection"]
            result["partition_summaries"] = {
                partition: _lineage_partition_summary(
                    evaluation,
                    edge_document,
                    int(selection["split_seed"]),
                    float(selection["tuning_fraction"]),
                    int(selection["tuning_bootstrap_samples"]),
                    partition,
                )
                for partition in ("tuning", "holdout")
            }
        _atomic_json(output, result)
    else:
        result = None
    result = _broadcast(result, rank)
    if dist.is_initialized():
        dist.destroy_process_group()
    return result


def run_candidate_lineage_probe(
    config_path: str | Path,
    version: int,
    candidate_name: str,
    output_path: str | Path,
    candidate_config_path: str | Path | None = None,
    minimum_source_version: int = 1,
) -> dict[str, Any]:
    started = time.monotonic()
    path = Path(config_path)
    document = load_persistent_config(path)
    document["config_path"] = str(path)
    config_sha256 = file_sha256(path)
    rank, world_size, device = _distributed(document)
    _seed(int(document["training"]["seed"]))
    output = Path(output_path)
    if output.is_file():
        result = json.loads(output.read_text()) if rank == 0 else None
        result = _broadcast(result, rank)
        if dist.is_initialized():
            dist.destroy_process_group()
        return result
    if not 2 <= version <= int(document["checkpoint"]["versions"]):
        raise ValueError("KuaiRand candidate lineage-probe version differs")
    if not 1 <= minimum_source_version < version:
        raise ValueError("KuaiRand candidate lineage-probe source boundary differs")
    if candidate_config_path is None:
        candidates = document["training"]["candidate_ladder"]
        candidate_config = None
    else:
        candidate_config_document = load_candidate_probe_config(candidate_config_path, path)
        candidates = candidate_config_document["candidates"]
        candidate_config = {
            "path": str(candidate_config_path),
            "sha256": file_sha256(candidate_config_path),
        }
    matches = [value for value in candidates if value["name"] == candidate_name]
    if len(matches) != 1:
        raise ValueError("KuaiRand candidate lineage-probe name differs")
    candidate = matches[0]
    transition = document["transitions"][version - 1]
    base_config = load_config(document["parent"]["base_config"]["path"])
    edge_document = _edge_config(base_config, transition, 1.0)
    if "update_dates" in transition:
        edge_document["data"]["update_dates"] = transition["update_dates"]
    edge_document["data"]["evaluation_targets_per_user"] = int(
        document["evaluation"]["targets_per_user"]
    )
    edge_document["data"]["user_limit"] = document["data"].get("user_limit")
    edge_document["evaluation"]["candidate_count"] = int(
        document["evaluation"]["candidate_count"]
    )
    workload = build_workload(edge_document)
    embedding_rows = int(workload["metadata"]["embedding_rows"])
    dense, embedding, tracker, geometry = _initialize_model(
        document,
        base_config,
        embedding_rows,
        rank,
        world_size,
        device,
    )
    checkpoint_root = Path(document["outputs"]["checkpoint_root"])
    captured_by_source = {}
    source_manifest = None
    for source_version in range(minimum_source_version, version):
        source_manifest = _load_checkpoint(
            checkpoint_root,
            source_version,
            dense,
            embedding,
            tracker,
            document,
            config_sha256,
            rank,
        )
        batches = _evaluation_batches(
            workload,
            int(document["evaluation"]["local_batch_size"]),
            rank,
            world_size,
        )
        captured_by_source[source_version] = _capture_old(
            dense,
            embedding,
            batches,
            workload,
            base_config,
            device,
        )
        del batches
    assert source_manifest is not None
    before = tracker.local_bitmap.clone()
    training = _train_candidate(
        dense,
        embedding,
        tracker,
        workload,
        base_config,
        _training_document(document, candidate),
        candidate,
        rank,
        world_size,
        device,
        int(document["training"]["seed"]) + 2003 + (version - 1) * 100003,
    )
    lineage = []
    for source_version in range(minimum_source_version, version):
        compact, evaluation = _evaluate_captured(
            dense,
            embedding,
            captured_by_source.pop(source_version),
            workload,
            edge_document,
            rank,
            world_size,
            device,
        )
        if rank == 0:
            assert compact is not None
            lineage.append(
                {
                    "source_version": source_version,
                    "target_version": version,
                    "cache_age": version - source_version,
                    "summary": compact,
                }
            )
        del compact, evaluation
    new_rows, cumulative_rows = _global_new_rows(tracker, before, device)
    if rank == 0:
        ndcg_values = [
            row["summary"]["comparisons"]["recompute_over_reuse"]["ndcg_at_5"][
                "relative_percent"
            ]
            for row in lineage
        ]
        result = {
            "protocol": f"{PROTOCOL}_candidate_lineage_probe_v0",
            "status": "complete",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(path), "sha256": config_sha256},
            "candidate_config": candidate_config,
            "version": version,
            "training_source_version": version - 1,
            "minimum_source_version": minimum_source_version,
            "source_checkpoint": {
                "path": str(_manifest_path(checkpoint_root, version - 1)),
                "sha256": file_sha256(_manifest_path(checkpoint_root, version - 1)),
                "checkpoint_bytes": int(source_manifest["checkpoint_bytes"]),
            },
            "transition": transition,
            "candidate": candidate,
            "training": training,
            "workload": workload["metadata"],
            "lineage": lineage,
            "row_gate": {
                "all_ndcg_at_5_positive": all(value > 0.0 for value in ndcg_values),
                "all_ndcg_at_5_above_one_percent": all(
                    value >= float(document["selection"]["minimum_relative_percent"])
                    for value in ndcg_values
                ),
                "adjacent_ndcg_at_5_relative_percent": ndcg_values[-1],
                "minimum_ndcg_at_5_relative_percent": min(ndcg_values),
            },
            "new_optimizer_active_rows": new_rows,
            "cumulative_optimizer_active_rows": cumulative_rows,
            "geometry": geometry,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(output, result)
    else:
        result = None
    result = _broadcast(result, rank)
    if dist.is_initialized():
        dist.destroy_process_group()
    return result


def run_persistent_lineage_prefix(
    config_path: str | Path,
    final_version: int,
    output_path: str | Path,
) -> dict[str, Any]:
    started = time.monotonic()
    path = Path(config_path)
    document = load_persistent_config(path)
    configured_final_version = int(document["checkpoint"]["versions"])
    if not 2 <= final_version <= configured_final_version:
        raise ValueError("KuaiRand persistent lineage-prefix version differs")
    document["config_path"] = str(path)
    config_sha256 = file_sha256(path)
    rank, world_size, device = _distributed(document)
    _seed(int(document["training"]["seed"]))
    base_config = load_config(document["parent"]["base_config"]["path"])
    edge_documents, workloads = _build_workloads(
        document,
        base_config,
        rank,
        final_version,
    )
    embedding_rows = int(workloads[0]["metadata"]["embedding_rows"])
    dense, embedding, tracker, geometry = _initialize_model(
        document,
        base_config,
        embedding_rows,
        rank,
        world_size,
        device,
    )
    if int(geometry["global_model_parameter_bytes"]) != int(
        document["checkpoint"]["expected_global_parameter_bytes"]
    ):
        raise RuntimeError("KuaiRand persistent lineage-prefix geometry differs")
    checkpoint_root = Path(document["outputs"]["checkpoint_root"])
    accepted_root = Path(document["outputs"]["root"])
    for version in range(1, final_version + 1):
        _read_manifest(
            checkpoint_root,
            version,
            document,
            config_sha256,
            False,
        )
    if rank == 0:
        accepted_records = [
            json.loads(_accepted_path(accepted_root, version).read_text())
            for version in range(1, final_version + 1)
        ]
    else:
        accepted_records = None
    accepted_records = _broadcast(accepted_records, rank)
    prefix_document = json.loads(json.dumps(document))
    prefix_document["checkpoint"]["versions"] = final_version
    prefix_document["transitions"] = prefix_document["transitions"][:final_version]
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    result = _direct_lineage(
        dense,
        embedding,
        tracker,
        workloads,
        edge_documents,
        base_config,
        prefix_document,
        config_sha256,
        checkpoint_root,
        output_root,
        accepted_records,
        geometry,
        rank,
        world_size,
        device,
    )
    if rank == 0:
        result["prefix_lineage"] = {
            "configured_final_version": configured_final_version,
            "evaluated_final_version": final_version,
        }
        result["elapsed_seconds"] = time.monotonic() - started
        _atomic_json(output_root / "result.json", result)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return _broadcast(result, rank) if dist.is_initialized() else result


def run_persistent_chain(
    config_path: str | Path,
    stop_after_version: int | None = None,
    candidate_priority: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    path = Path(config_path)
    document = load_persistent_config(path)
    document["config_path"] = str(path)
    config_sha256 = file_sha256(path)
    rank, world_size, device = _distributed(document)
    _seed(int(document["training"]["seed"]))
    output_root = Path(document["outputs"]["root"])
    checkpoint_root = Path(document["outputs"]["checkpoint_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    base_config = load_config(document["parent"]["base_config"]["path"])
    edge_documents, workloads = _build_workloads(document, base_config, rank, stop_after_version)
    embedding_rows = int(workloads[0]["metadata"]["embedding_rows"])
    dense, embedding, tracker, geometry = _initialize_model(
        document, base_config, embedding_rows, rank, world_size, device
    )
    if int(geometry["global_model_parameter_bytes"]) != int(
        document["checkpoint"]["expected_global_parameter_bytes"]
    ) or bool(geometry["single_gpu_parameter_overflow"]) != bool(
        document["model"]["require_single_card_overflow"]
    ):
        raise RuntimeError("KuaiRand persistent model geometry differs")
    _repair_manifest_only_boundary(
        checkpoint_root,
        output_root,
        document,
        config_sha256,
        rank,
    )
    completed = _completed_prefix(checkpoint_root, output_root, document, config_sha256)
    calibration = (
        _calibrate(
            dense,
            embedding,
            tracker,
            workloads[0],
            base_config,
            document,
            rank,
            world_size,
            device,
        )
        if completed == 0
        else {
            "status": "skipped_on_checkpoint_resume",
            "completed_versions": completed,
        }
    )
    configured_final_version = int(document["checkpoint"]["versions"])
    if stop_after_version is not None and not (
        completed < stop_after_version <= configured_final_version
    ):
        raise ValueError("KuaiRand stop-after version differs")
    if candidate_priority is not None and (
        stop_after_version is None or stop_after_version != completed + 1
    ):
        raise ValueError("KuaiRand candidate priority boundary differs")
    disk = _disk_preflight(document, checkpoint_root, completed)
    if document["checkpoint"].get("retain_bootstrap", False):
        if completed == 0:
            _save_checkpoint(
                checkpoint_root,
                0,
                dense,
                embedding,
                tracker,
                geometry,
                document,
                config_sha256,
                {
                    "round_id": document["round_id"],
                    "config": {"path": str(path), "sha256": config_sha256},
                    "role": "calibrated_stream_bootstrap",
                    "parent": document["parent"],
                    "calibration": calibration,
                },
                rank,
                world_size,
            )
        else:
            _read_manifest(
                checkpoint_root, 0, document, config_sha256, False
            )
    if rank == 0:
        preflight_path = output_root / "preflight.json"
        _atomic_json(
            preflight_path,
            {
                "protocol": PROTOCOL,
                "round_id": document["round_id"],
                "config_sha256": config_sha256,
                "completed_versions": completed,
                "disk": disk,
                "geometry": geometry,
                "calibration": calibration,
                "scientific_result": False,
                "formal_result": False,
            },
        )
    accepted_records = _train_missing_versions(
        dense,
        embedding,
        tracker,
        geometry,
        workloads,
        edge_documents,
        base_config,
        document,
        path,
        config_sha256,
        checkpoint_root,
        output_root,
        completed,
        stop_after_version,
        candidate_priority,
        rank,
        world_size,
        device,
    )
    if len(accepted_records) < configured_final_version:
        if rank == 0:
            result = {
                "protocol": PROTOCOL,
                "round_id": document["round_id"],
                "status": "partial_checkpoint_chain",
                "completed_versions": len(accepted_records),
                "latest_accepted": accepted_records[-1],
                "config": {"path": str(path), "sha256": config_sha256},
                "geometry": geometry,
                "elapsed_seconds": time.monotonic() - started,
                "scientific_result": False,
                "formal_result": False,
            }
            _atomic_json(output_root / "progress.json", result)
        else:
            result = None
        if dist.is_initialized():
            dist.barrier()
        result = _broadcast(result, rank)
        if dist.is_initialized():
            dist.destroy_process_group()
        return result
    result = _direct_lineage(
        dense,
        embedding,
        tracker,
        workloads,
        edge_documents,
        base_config,
        document,
        config_sha256,
        checkpoint_root,
        output_root,
        accepted_records,
        geometry,
        rank,
        world_size,
        device,
    )
    if rank == 0:
        result["elapsed_seconds"] = time.monotonic() - started
        _atomic_json(output_root / "result.json", result)
    if dist.is_initialized():
        dist.barrier()
    result = _broadcast(result if rank == 0 else None, rank)
    if dist.is_initialized():
        dist.destroy_process_group()
    return result
