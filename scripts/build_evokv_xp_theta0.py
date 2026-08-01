from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.migration.foundation_workload import (
    array_sha256 as foundation_array_sha256,
)
from hstu_kvcache.streaming.sharded_edge import (
    SHARDED_EDGE_CHECKPOINT_SCHEMA,
    ExternalEmbeddingHSTU,
    ShardedEdgeModelSpec,
    modulo_local_rows,
)
from hstu_kvcache.streaming.xp_projected_edge import (
    OptimizerActiveRowTracker,
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
    save_xp_projected_checkpoint,
    sparse_embedding_sgd,
    tracked_sparse_optimizer_step,
)
from hstu_kvcache.streaming.xp_theta0 import (
    XP_THETA0_PROTOCOL,
    StructuredSemiOrthogonalExpansion,
    file_sha256,
    load_xp_base_pair_corpus,
    projected_pairwise_contrastive_loss,
)

DEFAULT_SOURCE_ROOT = Path(
    "checkpoints/evokv_design3_m1_qk_entity_h1536/seed0"
)
DEFAULT_PAIRS = Path(
    "data/processed/evokv_foundation/"
    "qk_xp_base_row_cooccurrence.npz"
)
DEFAULT_PAIR_SUMMARY = Path(
    "configs/evokv_foundation/"
    "qk_xp_base_row_cooccurrence_summary.json"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "checkpoints/evokv_xp_qk_e4096_h1536/seed0"
)
DEFAULT_FOUNDATION_WORKLOAD = Path(
    "data/processed/evokv_foundation/x_qk_het_foundation.npz"
)
DEFAULT_OUTPUT = Path(
    "results/system/evokv_design3_foundation/"
    "xp_theta0_cooccurrence_training_development.json"
)
CANONICAL_NUM_EMBEDDINGS = 2_859_836
CANONICAL_SOURCE_WIDTH = 1536
CANONICAL_TARGET_WIDTH = 4096
CANONICAL_LAYERS = 24
CANONICAL_HEADS = 24
CANONICAL_HEAD_DIM = 64
CANONICAL_NEIGHBOR_ROWS = 2_859_736
CANONICAL_ISOLATED_ROWS = 99
CANONICAL_MINIMUM_ACTIVE_ROWS = 2_840_105


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkpoint-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )
    parser.add_argument("--source-version", type=int, default=0)
    parser.add_argument(
        "--cooccurrence",
        type=Path,
        default=DEFAULT_PAIRS,
    )
    parser.add_argument(
        "--cooccurrence-summary",
        type=Path,
        default=DEFAULT_PAIR_SUMMARY,
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument(
        "--foundation-workload",
        type=Path,
        default=DEFAULT_FOUNDATION_WORKLOAD,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--backend")
    parser.add_argument(
        "--target-embedding-width",
        type=int,
        default=CANONICAL_TARGET_WIDTH,
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--expansion-row-chunk",
        type=int,
        default=8192,
    )
    parser.add_argument("--oracle-samples", type=int, default=1024)
    parser.add_argument(
        "--nullspace-norm-ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--embedding-learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--projection-learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--negative-stride",
        type=int,
        default=1_000_003,
    )
    parser.add_argument(
        "--expected-neighbor-rows",
        type=int,
        default=CANONICAL_NEIGHBOR_ROWS,
    )
    parser.add_argument(
        "--expected-isolated-rows",
        type=int,
        default=CANONICAL_ISOLATED_ROWS,
    )
    parser.add_argument(
        "--minimum-active-rows",
        type=int,
        default=CANONICAL_MINIMUM_ACTIVE_ROWS,
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--development-canary", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch_size": args.batch_size,
        "expansion_row_chunk": args.expansion_row_chunk,
        "oracle_samples": args.oracle_samples,
        "target_embedding_width": args.target_embedding_width,
        "expected_neighbor_rows": args.expected_neighbor_rows,
        "minimum_active_rows": args.minimum_active_rows,
        "negative_stride": args.negative_stride,
        "progress_every": args.progress_every,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(
            f"positive values required for: {', '.join(invalid)}"
        )
    if (
        args.expected_isolated_rows < 0
        or args.source_version < 0
        or args.embedding_learning_rate <= 0
        or args.projection_learning_rate <= 0
        or args.temperature <= 0
        or args.nullspace_norm_ratio <= 0
        or args.source_checkpoint_root.resolve()
        == args.checkpoint_root.resolve()
    ):
        raise ValueError("XP theta0 builder arguments are invalid")
    targets = (
        args.output,
        args.checkpoint_root / "theta_0" / "manifest.json",
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "XP theta0 output exists; pass --force to replace: "
            f"{existing}"
        )


def _init_process_group(
    args: argparse.Namespace,
) -> tuple[int, int, torch.device, str]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2:
        raise RuntimeError("XP theta0 builder requires exactly two ranks")
    backend = args.backend or (
        "nccl" if args.device == "cuda" else "gloo"
    )
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("XP theta0 CUDA execution is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if not args.development_canary:
            raise RuntimeError(
                "CPU execution is restricted to development canaries"
            )
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("XP theta0 process group differs")
    return rank, world_size, device, backend


def _verify_artifact(path: Path, descriptor: dict[str, object]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(descriptor.get("bytes", -1))
        or file_sha256(path) != str(descriptor.get("sha256", ""))
    ):
        raise ValueError("source checkpoint artifact binding differs")


def _load_source_checkpoint(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[
    ShardedEdgeModelSpec,
    ExternalEmbeddingHSTU,
    torch.Tensor,
    dict[str, object],
    dict[str, object],
]:
    directory = (
        args.source_checkpoint_root
        / f"theta_{args.source_version}"
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != SHARDED_EDGE_CHECKPOINT_SCHEMA
        or manifest.get("version") != args.source_version
        or manifest.get("world_size") != world_size
        or not isinstance(manifest.get("spec"), dict)
        or not isinstance(manifest.get("dense"), dict)
        or not isinstance(manifest.get("embedding_shards"), list)
        or len(manifest["embedding_shards"]) != world_size
    ):
        raise ValueError("source checkpoint manifest differs")
    spec = ShardedEdgeModelSpec(**manifest["spec"])
    if not args.development_canary and (
        spec.num_embeddings != CANONICAL_NUM_EMBEDDINGS
        or spec.hidden_size != CANONICAL_SOURCE_WIDTH
        or spec.num_layers != CANONICAL_LAYERS
        or spec.num_heads != CANONICAL_HEADS
        or spec.head_dim != CANONICAL_HEAD_DIM
        or args.target_embedding_width != CANONICAL_TARGET_WIDTH
    ):
        raise ValueError("canonical XP theta0 source geometry differs")
    dense_path = directory / str(manifest["dense"]["path"])
    verification: list[object] = [None]
    if rank == 0:
        try:
            _verify_artifact(dense_path, manifest["dense"])
            verification[0] = {"passed": True}
        except Exception as error:
            verification[0] = {
                "passed": False,
                "error": str(error),
            }
    dist.broadcast_object_list(verification, src=0)
    if not bool(verification[0]["passed"]):
        raise ValueError(
            "source dense artifact verification failed: "
            f"{verification[0].get('error', '')}"
        )
    shard = manifest["embedding_shards"][rank]
    if not isinstance(shard, dict) or shard.get("rank") != rank:
        raise ValueError("source checkpoint rank descriptor differs")
    shard_path = directory / str(shard["path"])
    _verify_artifact(shard_path, shard)
    dense_payload = torch.load(
        dense_path,
        map_location=device,
        weights_only=True,
    )
    shard_payload = torch.load(
        shard_path,
        map_location=device,
        weights_only=True,
    )
    local_rows = modulo_local_rows(
        spec.num_embeddings,
        rank,
        world_size,
    )
    local_weight = shard_payload.get("local_weight")
    if (
        dense_payload.get("schema")
        != SHARDED_EDGE_CHECKPOINT_SCHEMA
        or dense_payload.get("version") != args.source_version
        or dense_payload.get("config") != asdict(spec.hstu_config())
        or shard_payload.get("schema")
        != SHARDED_EDGE_CHECKPOINT_SCHEMA
        or shard_payload.get("version") != args.source_version
        or shard_payload.get("rank") != rank
        or shard_payload.get("world_size") != world_size
        or shard_payload.get("num_embeddings")
        != spec.num_embeddings
        or shard_payload.get("hidden_size") != spec.hidden_size
        or not isinstance(local_weight, torch.Tensor)
        or local_weight.shape != (local_rows, spec.hidden_size)
        or local_weight.dtype != torch.float32
    ):
        raise ValueError("source checkpoint payload differs")
    dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
    dense.load_state_dict(dense_payload["state_dict"])
    source_binding = {
        "root": str(args.source_checkpoint_root),
        "version": args.source_version,
        "manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        },
        "schema": manifest["schema"],
        "spec": manifest["spec"],
        "dense": manifest["dense"],
        "embedding_shards": manifest["embedding_shards"],
    }
    del dense_payload, shard_payload
    return spec, dense, local_weight, manifest, source_binding


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _phase_time(
    device: torch.device,
    started: float,
) -> float:
    _synchronize(device)
    return time.perf_counter() - started


def _canonical_source_code() -> dict[str, dict[str, object]]:
    paths = (
        Path(__file__),
        Path(
            "src/hstu_kvcache/streaming/xp_projected_edge.py"
        ),
        Path("src/hstu_kvcache/streaming/xp_theta0.py"),
    )
    return {
        path.name: {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for path in paths
    }


def _load_request_union(
    path: Path,
    num_embeddings: int,
) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as source:
        required = {
            "semantic_request_union_item_idx",
            "semantic_request_union_eligible_for_update",
            "metadata_json",
        }
        if not required.issubset(source.files):
            raise ValueError(
                "foundation request-union artifact is incomplete"
            )
        rows = np.asarray(
            source["semantic_request_union_item_idx"],
            dtype=np.int32,
        )
        eligible = np.asarray(
            source[
                "semantic_request_union_eligible_for_update"
            ],
            dtype=np.uint8,
        )
        metadata = json.loads(str(source["metadata_json"].item()))
    observed_hash = foundation_array_sha256(rows)
    expected = metadata.get("semantic_request_union", {})
    if (
        rows.ndim != 1
        or len(rows) < 1
        or eligible.shape != rows.shape
        or not np.all(eligible == 1)
        or np.any(rows < 1)
        or np.any(rows >= num_embeddings)
        or not np.array_equal(rows, np.unique(rows))
        or expected.get("unique_rows") != len(rows)
        or expected.get("unique_rows_sha256") != observed_hash
    ):
        raise ValueError("foundation request union differs")
    return rows, {
        "workload_path": str(path),
        "workload_file_sha256": file_sha256(path),
        "unique_rows": len(rows),
        "unique_rows_sha256": observed_hash,
        "all_rows_base_update_eligible": True,
    }


def _request_union_coverage(
    rows: np.ndarray,
    tracker: OptimizerActiveRowTracker,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    owned = rows[rows % world_size == rank].astype(
        np.int64,
        copy=False,
    )
    local_indices = torch.from_numpy(
        (owned // world_size).copy()
    )
    active = tracker.local_bitmap.index_select(
        0,
        local_indices,
    ).numpy()
    local_missing = [
        int(value)
        for value in owned[active == 0].tolist()
    ]
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_missing)
    missing = sorted(
        int(value)
        for rank_values in gathered
        for value in rank_values
    )
    return {
        "requested_rows": len(rows),
        "active_requested_rows": len(rows) - len(missing),
        "missing_active_rows": len(missing),
        "missing_row_ids": missing,
        "passed": len(missing) == 0,
        "blocks_formal_qualification_if_unresolved": (
            len(missing) != 0
        ),
    }


def _global_oracle(
    local: dict[str, int | float | bool],
    device: torch.device,
) -> dict[str, int | float | bool]:
    values = torch.tensor(
        [
            float(local["max_abs_error"]),
            float(local["mean_abs_error"])
            * int(local["sample_rows"]),
            float(local["sample_rows"]),
            float(
                local["maximum_projection_row_norm_error"]
            ),
            float(bool(local["all_target_coordinates_used"])),
        ],
        dtype=torch.float64,
        device=device,
    )
    maximum = values[[0, 3]].clone()
    totals = values[[1, 2]].clone()
    minimum_used = values[4].clone()
    energy = torch.tensor(
        [
            float(local["source_energy"]),
            float(local["nullspace_energy"]),
        ],
        dtype=torch.float64,
        device=device,
    )
    null_response = torch.tensor(
        float(local["sampled_projected_nullspace_max_abs"]),
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    dist.all_reduce(minimum_used, op=dist.ReduceOp.MIN)
    dist.all_reduce(energy, op=dist.ReduceOp.SUM)
    dist.all_reduce(null_response, op=dist.ReduceOp.MAX)
    selected_per_rank: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(
        selected_per_rank,
        local["local_selected_nullspace_basis_ids"],
    )
    selected = sorted(
        {
            int(value)
            for rank_values in selected_per_rank
            for value in rank_values
        }
    )
    return {
        "sample_rows": int(totals[1].item()),
        "max_abs_error": float(maximum[0].item()),
        "mean_abs_error": float(
            (totals[0] / totals[1]).item()
        ),
        "maximum_projection_row_norm_error": float(
            maximum[1].item()
        ),
        "nullspace_dimension": int(local["nullspace_dimension"]),
        "nullspace_norm_ratio": float(
            local["nullspace_norm_ratio"]
        ),
        "source_energy": float(energy[0].item()),
        "nullspace_energy": float(energy[1].item()),
        "nullspace_to_source_energy_ratio": float(
            energy[1].item() / energy[0].item()
            if energy[0].item() > 0
            else 0.0
        ),
        "sampled_projected_nullspace_max_abs": float(
            null_response.item()
        ),
        "selected_nullspace_basis_directions": len(selected),
        "all_nullspace_basis_directions_selected": (
            len(selected) == int(local["nullspace_dimension"])
        ),
        "all_target_coordinates_used": bool(
            minimum_used.item()
        ),
    }


def _hbm_report(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "device_total_bytes": 0,
        }
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device)
        ),
        "device_total_bytes": torch.cuda.get_device_properties(
            device
        ).total_memory,
    }


def _training_pass(
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    corpus,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    anchors = corpus.anchor_rows[rank::world_size]
    positives = corpus.positive_rows[rank::world_size]
    negatives = corpus.negative_rows[rank::world_size]
    counts: list[object] = [None] * world_size
    dist.all_gather_object(counts, len(anchors))
    if len(set(int(value) for value in counts)) != 1:
        raise ValueError(
            "XP pair shards require equal per-rank steps"
        )
    embedding_optimizer = sparse_embedding_sgd(
        embedding,
        args.embedding_learning_rate,
    )
    projection_optimizer = torch.optim.SGD(
        [embedding.projection_weight],
        lr=args.projection_learning_rate,
        momentum=0.0,
        weight_decay=0.0,
        foreach=False,
    )
    local_loss_sum = 0.0
    local_pairs = 0
    steps = (
        len(anchors) + args.batch_size - 1
    ) // args.batch_size
    started = time.perf_counter()
    for step, start in enumerate(
        range(0, len(anchors), args.batch_size),
        start=1,
    ):
        stop = min(start + args.batch_size, len(anchors))
        ids = np.stack(
            (
                anchors[start:stop],
                positives[start:stop],
                negatives[start:stop],
            ),
            axis=1,
        )
        item_ids = torch.from_numpy(ids).to(
            device=device,
            dtype=torch.int64,
        )
        lengths = torch.full(
            (len(ids),),
            3,
            dtype=torch.int64,
            device=device,
        )
        embedding_optimizer.zero_grad(set_to_none=True)
        projection_optimizer.zero_grad(set_to_none=True)
        vectors = embedding(item_ids, lengths)
        loss = projected_pairwise_contrastive_loss(
            vectors,
            temperature=args.temperature,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("XP contrastive loss is not finite")
        (loss / world_size).backward()
        projection_gradient = embedding.projection_weight.grad
        if (
            projection_gradient is None
            or not bool(torch.all(torch.isfinite(projection_gradient)))
        ):
            raise RuntimeError(
                "XP projection gradient is not finite"
            )
        projection_optimizer.step()
        tracked_sparse_optimizer_step(
            embedding,
            embedding_optimizer,
            tracker,
        )
        local_loss_sum += float(loss.detach().item()) * len(ids)
        local_pairs += len(ids)
        if rank == 0 and (
            step == 1
            or step == steps
            or step % args.progress_every == 0
        ):
            print(
                f"phase=xp_theta0_training step={step}/{steps} "
                f"loss={float(loss.detach().item()):.6f} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    totals = torch.tensor(
        [local_loss_sum, local_pairs],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {
        "objective": (
            "pairwise cosine contrastive ranking over one true "
            "same-user adjacent positive and one cross-user negative "
            "drawn from another true adjacent pair"
        ),
        "epochs": 1,
        "global_pairs": int(totals[1].item()),
        "pairs_per_rank": [int(value) for value in counts],
        "steps_per_rank": steps,
        "batch_size_per_rank": args.batch_size,
        "mean_loss": float((totals[0] / totals[1]).item()),
        "temperature": args.temperature,
        "embedding_learning_rate": args.embedding_learning_rate,
        "projection_learning_rate": args.projection_learning_rate,
        "dense_core_updated": False,
        "isolated_rows_used": False,
        "cold_or_post_base_rows_used": False,
        "artificial_touch_or_zero_gradient_gate_used": False,
    }


def run(args: argparse.Namespace) -> dict[str, object] | None:
    validate_args(args)
    rank, world_size, device, backend = _init_process_group(args)
    total_started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        started = time.perf_counter()
        (
            source_spec,
            dense,
            source_weight,
            source_manifest,
            source_binding,
        ) = _load_source_checkpoint(
            args,
            rank,
            world_size,
            device,
        )
        phase_seconds["source_checkpoint_load_and_verify"] = (
            _phase_time(device, started)
        )
        target_spec = XPProjectedModelSpec(
            num_embeddings=source_spec.num_embeddings,
            embedding_width=args.target_embedding_width,
            hidden_size=source_spec.hidden_size,
            num_prediction_items=(
                source_spec.num_prediction_items
            ),
            num_behaviors=source_spec.num_behaviors,
            num_layers=source_spec.num_layers,
            num_heads=source_spec.num_heads,
            head_dim=source_spec.head_dim,
            max_seq_len=source_spec.max_seq_len,
        )
        expansion = StructuredSemiOrthogonalExpansion(
            source_width=source_spec.hidden_size,
            target_width=args.target_embedding_width,
        )
        started = time.perf_counter()
        projection = expansion.projection_weight(
            device=device,
            dtype=source_weight.dtype,
        )
        expanded = expansion.expand_rows(
            source_weight,
            row_chunk=args.expansion_row_chunk,
            global_row_start=rank,
            global_row_stride=world_size,
            nullspace_norm_ratio=args.nullspace_norm_ratio,
        )
        local_oracle = expansion.numeric_oracle(
            source_weight,
            expanded,
            projection,
            maximum_samples=args.oracle_samples,
            global_row_start=rank,
            global_row_stride=world_size,
            nullspace_norm_ratio=args.nullspace_norm_ratio,
        )
        oracle = _global_oracle(local_oracle, device)
        phase_seconds["structured_expansion_and_oracle"] = (
            _phase_time(device, started)
        )
        embedding = TrainableProjectedModuloEmbedding(
            local_weight=expanded,
            projection_weight=projection,
            num_embeddings=target_spec.num_embeddings,
            rank=rank,
            world_size=world_size,
        )
        tracker = OptimizerActiveRowTracker(
            num_embeddings=target_spec.num_embeddings,
            rank=rank,
            world_size=world_size,
        )
        del source_weight, expanded, projection
        if device.type == "cuda":
            torch.cuda.empty_cache()
        started = time.perf_counter()
        corpus = load_xp_base_pair_corpus(
            args.cooccurrence,
            args.cooccurrence_summary,
            num_embeddings=target_spec.num_embeddings,
            expected_neighbor_rows=args.expected_neighbor_rows,
            expected_isolated_rows=args.expected_isolated_rows,
            negative_stride=args.negative_stride,
        )
        request_union_rows, request_union_binding = (
            _load_request_union(
                args.foundation_workload,
                target_spec.num_embeddings,
            )
        )
        phase_seconds["cooccurrence_load_and_validation"] = (
            _phase_time(device, started)
        )
        started = time.perf_counter()
        training = _training_pass(
            embedding,
            tracker,
            corpus,
            args,
            rank,
            world_size,
            device,
        )
        phase_seconds["projected_contrastive_optimizer_pass"] = (
            _phase_time(device, started)
        )
        trained_nullspace_response = (
            expansion.projection_nullspace_response(
                embedding.projection_weight,
            )
        )
        request_union_coverage = _request_union_coverage(
            request_union_rows,
            tracker,
            rank,
            world_size,
        )
        local_active = torch.tensor(
            tracker.local_active_count,
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(local_active, op=dist.ReduceOp.SUM)
        byte_gate_before_checkpoint = (
            int(local_active.item())
            >= args.minimum_active_rows
        )
        code = _canonical_source_code()
        provenance = {
            "protocol": XP_THETA0_PROTOCOL,
            "scientific_result": False,
            "formal_design2": False,
            "formal_design3": False,
            "source_checkpoint": source_binding,
            "cooccurrence": {
                "path": str(args.cooccurrence),
                "file_sha256": corpus.file_sha256,
                "content_sha256": corpus.content_sha256,
                "semantic_rows": corpus.semantic_rows,
                "neighbor_pairs": corpus.pair_count,
                "isolated_rows_skipped": corpus.isolated_rows,
                "pair_arrays_sha256": (
                    corpus.pair_arrays_sha256
                ),
                "negative_policy": {
                    "initial_stride": args.negative_stride,
                    "cross_user_required": True,
                    "negative_differs_from_anchor_and_positive": True,
                    "negative_source": (
                        "positive row of another true adjacent pair"
                    ),
                },
            },
            "semantic_request_union": {
                **request_union_binding,
                "optimizer_active_coverage": (
                    request_union_coverage
                ),
            },
            "structured_expansion": {
                "source_width": expansion.source_width,
                "target_width": expansion.target_width,
                "full_repeats": expansion.full_repeats,
                "remainder": expansion.remainder,
                "projection_bias": False,
                "oracle_before_training": oracle,
                "projection_response_to_initial_nullspace_after_training": (
                    trained_nullspace_response
                ),
            },
            "training": training,
            "forced_sharding_byte_gate_before_checkpoint": {
                "minimum_active_rows": args.minimum_active_rows,
                "observed_active_rows": int(
                    local_active.item()
                ),
                "passed": byte_gate_before_checkpoint,
            },
            "source_code": code,
        }
        started = time.perf_counter()
        checkpoint = save_xp_projected_checkpoint(
            args.checkpoint_root,
            0,
            target_spec,
            dense,
            embedding,
            tracker,
            provenance=provenance,
        )
        phase_seconds["checkpoint_save_and_hash"] = _phase_time(
            device,
            started,
        )
        active = checkpoint["optimizer_active_rows"]
        gate_passed = (
            int(active["global_active_rows"])
            >= args.minimum_active_rows
        )
        local_report = {
            "rank": rank,
            "source_local_rows": modulo_local_rows(
                source_spec.num_embeddings,
                rank,
                world_size,
            ),
            "target_local_embedding_bytes": (
                embedding.local_weight.numel()
                * embedding.local_weight.element_size()
            ),
            "local_active_rows": tracker.local_active_count,
            "phase_seconds": phase_seconds,
            "memory": _hbm_report(device),
        }
        ranks: list[object] = [None] * world_size
        dist.all_gather_object(ranks, local_report)
        _synchronize(device)
        total_seconds = time.perf_counter() - total_started
        result = None
        if rank == 0:
            dense_parameter_bytes = sum(
                value.numel() * value.element_size()
                for value in dense.parameters()
            )
            active_embedding_bytes = (
                int(active["global_active_rows"])
                * target_spec.embedding_width
                * 4
            )
            result = {
                "protocol": XP_THETA0_PROTOCOL,
                "status": (
                    "complete"
                    if gate_passed
                    else "active_row_gate_failed"
                ),
                "scientific_result": False,
                "formal_design2": False,
                "formal_design3": False,
                "artifact_role": (
                    "successor_xp_base_only_theta0_checkpoint"
                ),
                "execution": {
                    "device": args.device,
                    "backend": backend,
                    "world_size": world_size,
                    "visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES",
                        "",
                    ),
                    "development_canary": (
                        args.development_canary
                    ),
                    "total_wall_seconds": total_seconds,
                },
                "source_checkpoint": source_binding,
                "source_manifest_reloaded": (
                    source_manifest == json.loads(
                        (
                            args.source_checkpoint_root
                            / f"theta_{args.source_version}"
                            / "manifest.json"
                        ).read_text()
                    )
                ),
                "target_spec": asdict(target_spec),
                "structured_expansion": (
                    provenance["structured_expansion"]
                ),
                "cooccurrence": provenance["cooccurrence"],
                "semantic_request_union": (
                    provenance["semantic_request_union"]
                ),
                "training": training,
                "optimizer_active_gate": {
                    "minimum_active_rows": (
                        args.minimum_active_rows
                    ),
                    "observed_active_rows": int(
                        active["global_active_rows"]
                    ),
                    "passed": gate_passed,
                    "active_embedding_bytes_fp32": (
                        active_embedding_bytes
                    ),
                    "dense_parameter_bytes_fp32": (
                        dense_parameter_bytes
                    ),
                    "projection_parameter_bytes_fp32": (
                        target_spec.projection_bytes_fp32
                    ),
                    "active_plus_dense_plus_projection_bytes": (
                        active_embedding_bytes
                        + dense_parameter_bytes
                        + target_spec.projection_bytes_fp32
                    ),
                    "padding_row_excluded": True,
                    "isolated_rows_excluded": True,
                    "ledger": active,
                },
                "checkpoint": {
                    "root": str(args.checkpoint_root),
                    "version": 0,
                    "manifest": checkpoint,
                },
                "source_code": code,
                "ranks": ranks,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(
                args.output.suffix + ".tmp"
            )
            temporary.write_text(
                json.dumps(result, indent=2, sort_keys=True)
                + "\n"
            )
            temporary.replace(args.output)
        dist.barrier()
        return result
    finally:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    result = run(args)
    if result is not None:
        print(args.output)
        if result["status"] != "complete":
            raise RuntimeError("XP theta0 active-row gate failed")


if __name__ == "__main__":
    main()
