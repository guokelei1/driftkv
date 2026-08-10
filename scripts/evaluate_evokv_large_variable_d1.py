from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.migration.program import compile_migration_program
from hstu_kvcache.migration.recursive_d1 import (
    fit_rollout_aware_direct_program,
    select_depth_balanced_tokens,
    stale_token_score_renewal,
    storage_cache,
)
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVProgram,
    compile_direct_oldkv_program,
    load_direct_oldkv_program,
    write_direct_oldkv_program,
)
from hstu_kvcache.migration.variable_inference import (
    VariableInferenceCorpus,
    array_sha256,
    file_sha256,
    load_corpus,
)
from hstu_kvcache.migration.xp_d1_quality import (
    apply_direct_oldkv,
    cache_relative_error,
    direct_program_sha256,
    recommendation_sums,
    summarize_recommendation_sums,
)
from hstu_kvcache.migration.xp_exact_baseline import load_inference_checkpoint
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.streaming.multifield_projected import (
    lookup_multifield_projected,
)
from hstu_kvcache.streaming.sharded_edge import fixed_candidate_ids
from hstu_kvcache.streaming.xp_projected_edge import XPProjectedModelSpec


@dataclass(frozen=True)
class RecordData:
    record_id: int
    user_id: int
    features: torch.Tensor
    target_items: torch.Tensor
    behaviors: torch.Tensor
    time_deltas: torch.Tensor
    labels: torch.Tensor
    schedule: tuple[int, ...]


@dataclass
class CacheState:
    cache: HSTUKVCache
    depth: int
    last_exact_version: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("qk", "qb"), required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qualification-record-limit", type=int)
    parser.add_argument("--edge-limit", type=int)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def initialize(args: argparse.Namespace) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    if not dist.is_initialized():
        options = {}
        if args.backend == "nccl" and device.type == "cuda":
            options["device_id"] = device
        dist.init_process_group(
            backend=args.backend,
            rank=rank,
            world_size=world_size,
            **options,
        )
    return rank, world_size, device


def gather(value: object, world_size: int) -> list[object]:
    values: list[object] = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def broadcast(value: object, rank: int) -> object:
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def load_config(
    path: Path,
    dataset: str,
    world_size: int,
) -> tuple[dict[str, object], dict[str, object]]:
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "evokv_large_variable_d1_score_sweep_two_gpu_v0"
        or value.get("status") != "ready_for_user_execution"
        or value.get("scientific_result") is not False
        or value.get("formal_result") is not False
        or int(value.get("world_size", -1)) != world_size
        or world_size != 2
        or value.get("serving_model_invariant", {}).get(
            "concurrent_recommendation_models"
        )
        != 1
        or dataset not in value.get("datasets", {})
    ):
        raise ValueError("large variable D1 config differs")
    registry = value["bindings"]["checkpoint_registry"]
    registry_path = Path(registry["path"])
    if (
        not registry_path.is_file()
        or file_sha256(registry_path) != registry["sha256"]
    ):
        raise ValueError("large variable D1 registry binding differs")
    selected = value["datasets"][dataset]
    thresholds = [int(item) for item in value["score_thresholds"]]
    if (
        len(thresholds) < 2
        or thresholds != sorted(set(thresholds))
        or min(thresholds) < 1
        or int(selected["edge_count"]) != len(selected["versions"]) - 1
    ):
        raise ValueError("large variable D1 threshold or edge config differs")
    return value, selected


def checkpoint_spec(root: Path, version: int) -> XPProjectedModelSpec:
    manifest = json.loads((root / f"theta_{version}" / "manifest.json").read_text())
    return XPProjectedModelSpec(**manifest["spec"])


def checkpoint_binding(root: Path, version: int) -> dict[str, object]:
    path = root / f"theta_{version}" / "manifest.json"
    return {
        "root": str(root),
        "version": version,
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
    }


def local_role_records(
    corpus: VariableInferenceCorpus,
    role: str,
    rank: int,
    world_size: int,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    global_records = corpus.role_records(role)
    if limit is not None:
        if limit < world_size or limit > len(global_records):
            raise ValueError("large variable D1 role limit differs")
        global_records = global_records[:limit]
    if len(global_records) % world_size:
        raise ValueError("large variable D1 role is not rank balanced")
    return global_records, global_records[rank::world_size]


def materialize_records(
    corpus: VariableInferenceCorpus,
    records: np.ndarray,
) -> list[RecordData]:
    result = []
    for raw_record in records:
        record = int(raw_record)
        extent = corpus.record_slice(record)
        result.append(
            RecordData(
                record_id=record,
                user_id=int(corpus.arrays["record_user_ids"][record]),
                features=torch.from_numpy(
                    corpus.arrays["feature_ids"][extent].astype(
                        np.int64, copy=True
                    )
                ),
                target_items=torch.from_numpy(
                    corpus.arrays["target_item_ids"][extent].astype(
                        np.int64, copy=True
                    )
                ),
                behaviors=torch.from_numpy(
                    corpus.arrays["behaviors"][extent].astype(
                        np.int64, copy=True
                    )
                ),
                time_deltas=torch.from_numpy(
                    corpus.arrays["time_deltas"][extent].astype(
                        np.float32, copy=True
                    )
                ),
                labels=torch.from_numpy(
                    corpus.arrays["labels"][extent].astype(
                        np.bool_, copy=True
                    )
                ),
                schedule=tuple(
                    int(item)
                    for item in corpus.arrays["edge_prefix_lengths"][record]
                ),
            )
        )
    return result


def lookup_features(
    dataset: str,
    embedding,
    features: torch.Tensor,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    values = features[:length].unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.int64, device=device)
    if dataset == "qk":
        return embedding(values[:, :, 0], lengths)
    return lookup_multifield_projected(embedding, values, lengths)


def lookup_candidates(
    embedding,
    candidates: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    candidates = candidates.to(device)
    lengths = torch.full(
        (len(candidates),),
        candidates.shape[1],
        dtype=torch.int64,
        device=device,
    )
    return embedding(candidates, lengths)


def float_cache(cache: HSTUKVCache, device: torch.device) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.to(device=device, dtype=torch.float32),
        v=cache.v.to(device=device, dtype=torch.float32),
        seq_len=cache.seq_len,
    )


@torch.no_grad()
def exact_cache(
    dataset: str,
    dense,
    embedding,
    record: RecordData,
    length: int,
    device: torch.device,
) -> HSTUKVCache:
    vectors = lookup_features(dataset, embedding, record.features, length, device)
    cache = dense.core.compute_kv_from_item_embeddings(
        vectors,
        record.behaviors[:length].unsqueeze(0).to(device),
        record.time_deltas[:length].unsqueeze(0).to(device),
        torch.tensor([length], dtype=torch.int64, device=device),
    )
    return storage_cache(cache)


@torch.no_grad()
def append_cache(
    dataset: str,
    dense,
    embedding,
    record: RecordData,
    cache: HSTUKVCache,
    left: int,
    right: int,
    device: torch.device,
) -> tuple[torch.Tensor, HSTUKVCache]:
    vectors = lookup_features(
        dataset,
        embedding,
        record.features[left:right],
        right - left,
        device,
    )
    hidden, updated = dense.core.forward_with_cache_from_item_embeddings(
        float_cache(cache, device),
        vectors,
        record.behaviors[left:right].unsqueeze(0).to(device),
        record.time_deltas[left:right].unsqueeze(0).to(device),
    )
    return hidden, storage_cache(updated)


def cache_error(cache: HSTUKVCache, exact: HSTUKVCache) -> float:
    return float(
        cache_relative_error(
            cache,
            exact,
            torch.tensor([exact.seq_len], dtype=torch.int64),
        )[0]
    )


def scores_from_hidden(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    candidate_vectors: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("nh,nch->nc", hidden[0][mask], candidate_vectors)


def exact_caches(
    dataset: str,
    dense,
    embedding,
    records: list[RecordData],
    edge: int,
    device: torch.device,
    phase: str,
) -> dict[int, HSTUKVCache]:
    result = {}
    for index, record in enumerate(records):
        result[record.record_id] = exact_cache(
            dataset,
            dense,
            embedding,
            record,
            record.schedule[edge],
            device,
        )
        if index == 0 or (index + 1) % 64 == 0 or index + 1 == len(records):
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "rank": dist.get_rank(),
                        "record": index + 1,
                        "records": len(records),
                    }
                ),
                flush=True,
            )
    return result


def load_model(
    root: Path,
    version: int,
    spec: XPProjectedModelSpec,
    rank: int,
    world_size: int,
    device: torch.device,
):
    dense, embedding, manifest = load_inference_checkpoint(
        root,
        version,
        spec,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    dense.eval()
    embedding.eval()
    return dense, embedding, manifest


def quantiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("large variable D1 metric values are empty")
    ordered = sorted(float(value) for value in values)
    return {
        "records": len(ordered),
        "mean": float(statistics.fmean(ordered)),
        "p50": float(ordered[min(len(ordered) - 1, math.ceil(0.5 * len(ordered)) - 1)]),
        "p90": float(ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]),
        "p95": float(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]),
        "maximum": float(ordered[-1]),
    }


def merge_edge_metrics(
    local_sums: dict[str, dict[str, float]],
    local_errors: dict[str, list[float]],
    local_candidates: str,
    methods: list[str],
    rank: int,
    world_size: int,
) -> dict[str, object]:
    gathered_sums = gather(local_sums, world_size)
    gathered_errors = gather(local_errors, world_size)
    candidate_hashes = gather(local_candidates, world_size)
    if rank == 0:
        merged_sums = {method: defaultdict(float) for method in methods}
        merged_errors = {method: [] for method in methods}
        for rank_sums in gathered_sums:
            for method, values in rank_sums.items():
                for name, value in values.items():
                    merged_sums[method][name] += float(value)
        for rank_errors in gathered_errors:
            for method, values in rank_errors.items():
                merged_errors[method].extend(float(value) for value in values)
        quality = {
            method: summarize_recommendation_sums(merged_sums[method])
            for method in methods
        }
        fidelity = {
            method: quantiles(merged_errors[method]) for method in methods
        }
        reuse_ce = float(quality["reuse"]["sampled_cross_entropy"])
        exact_ce = float(quality["exact"]["sampled_cross_entropy"])
        denominator = reuse_ce - exact_ce
        reuse_error = float(fidelity["reuse"]["mean"])
        for method in methods:
            quality[method]["ce_gap_recovery"] = (
                None
                if denominator <= 0
                else (
                    reuse_ce
                    - float(quality[method]["sampled_cross_entropy"])
                )
                / denominator
            )
            fidelity[method]["mean_error_recovery"] = (
                None
                if reuse_error <= 0
                else 1.0 - float(fidelity[method]["mean"]) / reuse_error
            )
        result = {
            "recommendation": quality,
            "cache_fidelity": fidelity,
            "candidate_sha256_per_rank": candidate_hashes,
        }
    else:
        result = None
    return broadcast(result, rank)


def fit_program(
    source_dense,
    target_dense,
    exact_source: dict[int, HSTUKVCache],
    deployed_source: dict[int, CacheState],
    exact_target: dict[int, HSTUKVCache],
    records: list[RecordData],
    edge: int,
    source_version: int,
    target_version: int,
    settings: dict[str, object],
    numeric_precision: dict[str, object],
    output_root: Path,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[DirectOldKVProgram, dict[str, object], dict[str, object]]:
    source_dense.to(device)
    analytic = compile_migration_program(
        target_dense.core,
        f"theta{source_version}",
        f"theta{target_version}",
    )
    base, compile_metrics = compile_direct_oldkv_program(
        source_dense.core, analytic
    )
    source_dense.to("cpu")
    del analytic
    record_ids = [torch.tensor([record.record_id], dtype=torch.int64) for record in records]
    lengths = [torch.tensor([record.schedule[edge]], dtype=torch.int64) for record in records]
    depths = [
        torch.tensor([deployed_source[record.record_id].depth], dtype=torch.int64)
        for record in records
    ]
    selected, sampling = select_depth_balanced_tokens(
        record_ids,
        lengths,
        depths,
        maximum_global_tokens=int(settings["maximum_global_tokens_per_layer"]),
        seed=int(settings["sampling_seed"]) + edge * 1_000_003,
    )
    evaluation_precision = torch.get_float32_matmul_precision()
    ridge_precision = str(numeric_precision["ridge_float32_matmul_precision"])
    torch.set_float32_matmul_precision(ridge_precision)
    try:
        float_program, program, metrics = fit_rollout_aware_direct_program(
            base,
            [exact_source[record.record_id] for record in records],
            [deployed_source[record.record_id].cache for record in records],
            [exact_target[record.record_id] for record in records],
            lengths,
            selected,
            mode="ract_kv",
            rank=int(settings["rank"]),
            ridge=float(settings["ridge"]),
            seed=int(settings["solver_seed"]) + edge * 1_000_003,
            device=device,
            maximum_jitter_multiplier=float(settings["maximum_jitter_multiplier"]),
        )
    finally:
        torch.set_float32_matmul_precision(evaluation_precision)
    identity = direct_program_sha256(program)
    identities = gather(identity, world_size)
    if len(set(str(value) for value in identities)) != 1:
        raise RuntimeError("large variable D1 program differs across ranks")
    metrics = {
        **metrics,
        "sampling": sampling,
        "ridge_float32_matmul_precision": ridge_precision,
        "restored_evaluation_float32_matmul_precision": evaluation_precision,
    }
    path = output_root / "programs" / f"theta{source_version}_to_theta{target_version}_direct_oldkv_fp16.pt"
    if rank == 0:
        descriptor = write_direct_oldkv_program(
            program,
            path,
            provenance={
                "protocol": "evokv_large_variable_d1_score_sweep_v0",
                "dataset": "qb",
                "source_version": source_version,
                "target_version": target_version,
                "program_identity_sha256": identity,
                "fit_metrics": metrics,
            },
            compile_metrics=compile_metrics,
        )
        descriptor["identity_sha256"] = identity
        descriptor["source"] = "variable_inference_disjoint_fit_role"
    else:
        descriptor = None
    descriptor = broadcast(descriptor, rank)
    del float_program
    return program, descriptor, metrics


def load_qk_program(
    descriptor: dict[str, object],
    source_version: int,
    target_version: int,
) -> tuple[DirectOldKVProgram, dict[str, object]]:
    path = Path(descriptor["path"])
    program, payload = load_direct_oldkv_program(
        path,
        expected_sha256=descriptor["sha256"],
        expected_source_version=f"theta{source_version}",
        expected_target_version=f"theta{target_version}",
    )
    return program, {
        "path": str(path),
        "sha256": descriptor["sha256"],
        "bytes": path.stat().st_size,
        "identity_sha256": direct_program_sha256(program),
        "source": "retained_QK_rollout_aware_ract_kv_exact0_program",
        "payload_provenance": payload.get("provenance"),
    }


@torch.no_grad()
def advance_support_states(
    dataset: str,
    dense,
    embedding,
    records: list[RecordData],
    states: dict[int, CacheState],
    program: DirectOldKVProgram,
    edge: int,
    target_version: int,
    device: torch.device,
) -> dict[int, CacheState]:
    deployed_program = program.to(device)
    output = {}
    for record in records:
        state = states[record.record_id]
        compiled = apply_direct_oldkv(deployed_program, state.cache.to(device))
        _, updated = append_cache(
            dataset,
            dense,
            embedding,
            record,
            storage_cache(compiled),
            record.schedule[edge],
            record.schedule[edge + 1],
            device,
        )
        output[record.record_id] = CacheState(
            cache=updated,
            depth=state.depth + 1,
            last_exact_version=state.last_exact_version,
        )
    return output


@torch.no_grad()
def probe_program(
    program: DirectOldKVProgram,
    states: dict[int, CacheState],
    exact_target: dict[int, HSTUKVCache],
    records: list[RecordData],
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    deployed = program.to(device)
    errors = []
    for record in records:
        compiled = apply_direct_oldkv(
            deployed, states[record.record_id].cache.to(device)
        )
        errors.append(cache_error(storage_cache(compiled), exact_target[record.record_id]))
    gathered = gather(errors, world_size)
    if rank == 0:
        result = quantiles([item for values in gathered for item in values])
    else:
        result = None
    return broadcast(result, rank)


def selection_for_edge(
    qualification_records: np.ndarray,
    corpus: VariableInferenceCorpus,
    edge: int,
    thresholds: list[int],
    score_states: dict[int, dict[int, int]],
) -> tuple[dict[str, set[int]], dict[str, dict[str, object]]]:
    exact_ids = {}
    ledgers = {}
    for threshold in thresholds:
        started = time.perf_counter()
        selected, next_scores, ledger = stale_token_score_renewal(
            [
                (
                    int(record),
                    int(corpus.arrays["edge_prefix_lengths"][record, edge]),
                    int(score_states[threshold][int(record)]),
                )
                for record in qualification_records
            ],
            threshold=threshold,
        )
        score_states[threshold] = next_scores
        method = f"score_{threshold}"
        exact_ids[method] = selected
        ledgers[method] = {
            **ledger,
            "selection_cpu_milliseconds": 1000.0 * (time.perf_counter() - started),
        }
    return exact_ids, ledgers


@torch.no_grad()
def evaluate_edge(
    dataset: str,
    dense,
    embedding,
    records: list[RecordData],
    states: dict[str, dict[int, CacheState]],
    program: DirectOldKVProgram,
    exact_ids: dict[str, set[int]],
    edge: int,
    target_version: int,
    negative_count: int,
    candidate_seed: int,
    spec: XPProjectedModelSpec,
    methods: list[str],
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[dict[str, dict[int, CacheState]], dict[str, object]]:
    deployed_program = program.to(device)
    local_sums = {method: defaultdict(float) for method in methods}
    local_errors = {method: [] for method in methods}
    candidate_digest = hashlib.sha256()
    for index, record in enumerate(records):
        left = record.schedule[edge]
        right = record.schedule[edge + 1]
        exact_cpu = exact_cache(dataset, dense, embedding, record, left, device)
        exact_device = exact_cpu.to(device)
        suffix_vectors = lookup_features(
            dataset,
            embedding,
            record.features[left:right],
            right - left,
            device,
        )
        suffix_behaviors = record.behaviors[left:right].unsqueeze(0).to(device)
        suffix_deltas = record.time_deltas[left:right].unsqueeze(0).to(device)
        hidden_exact, _ = dense.core.forward_with_cache_from_item_embeddings(
            float_cache(exact_cpu, device),
            suffix_vectors,
            suffix_behaviors,
            suffix_deltas,
        )
        mask = record.labels[left + 1 : right + 1].to(device)
        positives = record.target_items[left + 1 : right + 1].to(device)[mask]
        candidates = fixed_candidate_ids(
            positives,
            spec.num_prediction_items,
            negative_count,
            candidate_seed + edge * 1_000_003 + record.record_id,
        )
        candidate_digest.update(np.asarray([record.record_id, edge], dtype="<i8").tobytes())
        candidate_digest.update(
            np.asarray(candidates.detach().cpu(), dtype="<i8").tobytes()
        )
        candidate_vectors = lookup_candidates(embedding, candidates, device)
        exact_scores = scores_from_hidden(hidden_exact, mask, candidate_vectors)
        exact_sums = recommendation_sums(exact_scores, exact_scores)
        for name, value in exact_sums.items():
            local_sums["exact"][name] += float(value)
        local_errors["exact"].append(0.0)
        for method in (value for value in methods if value != "exact"):
            state = states[method][record.record_id]
            deployed = state.cache.to(device)
            if method == "reuse":
                post = deployed
                action_exact = False
            else:
                compiled = apply_direct_oldkv(deployed_program, deployed)
                action_exact = record.record_id in exact_ids.get(method, set())
                post = exact_device if action_exact else compiled
            post_cpu = storage_cache(post)
            local_errors[method].append(cache_error(post_cpu, exact_cpu))
            hidden, updated = dense.core.forward_with_cache_from_item_embeddings(
                float_cache(post_cpu, device),
                suffix_vectors,
                suffix_behaviors,
                suffix_deltas,
            )
            scores = scores_from_hidden(hidden, mask, candidate_vectors)
            sums = recommendation_sums(scores, exact_scores)
            for name, value in sums.items():
                local_sums[method][name] += float(value)
            states[method][record.record_id] = CacheState(
                cache=storage_cache(updated),
                depth=0 if action_exact else state.depth + 1,
                last_exact_version=(
                    target_version if action_exact else state.last_exact_version
                ),
            )
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == len(records):
            print(
                json.dumps(
                    {
                        "phase": "variable_qualification",
                        "dataset": dataset,
                        "edge": edge,
                        "rank": rank,
                        "record": index + 1,
                        "records": len(records),
                    }
                ),
                flush=True,
            )
    merged = merge_edge_metrics(
        {method: dict(values) for method, values in local_sums.items()},
        local_errors,
        candidate_digest.hexdigest(),
        methods,
        rank,
        world_size,
    )
    return states, merged


def aggregate_cumulative(
    edges: list[dict[str, object]],
    methods: list[str],
) -> dict[str, object]:
    result = {}
    for method in methods:
        count = sum(
            int(edge["metrics"]["recommendation"][method]["positive_targets"])
            for edge in edges
        )
        weighted_ce = sum(
            int(edge["metrics"]["recommendation"][method]["positive_targets"])
            * float(edge["metrics"]["recommendation"][method]["sampled_cross_entropy"])
            for edge in edges
        )
        result[method] = {
            "positive_targets": count,
            "sampled_cross_entropy": weighted_ce / count,
        }
    reuse_ce = result["reuse"]["sampled_cross_entropy"]
    exact_ce = result["exact"]["sampled_cross_entropy"]
    denominator = reuse_ce - exact_ce
    for method in methods:
        result[method]["ce_gap_recovery"] = (
            None
            if denominator <= 0
            else (reuse_ce - result[method]["sampled_cross_entropy"])
            / denominator
        )
    return result


def fairness_summary(
    qualification_records: np.ndarray,
    thresholds: list[int],
    exact_counts: dict[int, dict[int, int]],
    score_states: dict[int, dict[int, int]],
    edge_count: int,
    minimum_prefix_tokens: int,
) -> dict[str, object]:
    output = {}
    for threshold in thresholds:
        counts = np.asarray(
            [exact_counts[threshold][int(record)] for record in qualification_records],
            dtype=np.int64,
        )
        ending = np.asarray(
            [score_states[threshold][int(record)] for record in qualification_records],
            dtype=np.int64,
        )
        output[f"score_{threshold}"] = {
            "records": len(counts),
            "never_exact_records": int(np.count_nonzero(counts == 0)),
            "never_exact_fraction": float(np.mean(counts == 0)),
            "exact_count_histogram": {
                str(value): int(np.count_nonzero(counts == value))
                for value in range(edge_count + 1)
            },
            "maximum_exact_count": int(counts.max()),
            "ending_score": {
                "minimum": int(ending.min()),
                "median": float(np.median(ending)),
                "p95": float(np.quantile(ending, 0.95)),
                "maximum": int(ending.max()),
            },
            "debt_strictly_below_threshold": bool(ending.max() < threshold),
            "forced_exact_within_updates_at_minimum_prefix": math.ceil(
                threshold / minimum_prefix_tokens
            ),
            "minimum_prefix_tokens": minimum_prefix_tokens,
        }
    return output


def main() -> None:
    args = parse_args()
    rank, world_size, device = initialize(args)
    config, selected = load_config(args.config, args.dataset, world_size)
    numeric_precision = config.get("numeric_precision", {})
    if (
        numeric_precision.get("evaluation_float32_matmul_precision") != "high"
        or numeric_precision.get("ridge_float32_matmul_precision") != "highest"
        or numeric_precision.get("nvidia_tf32_override") != "unset"
        or numeric_precision.get("ridge_gram_accumulation") != "ieee_fp32"
        or os.environ.get("NVIDIA_TF32_OVERRIDE") not in {None, ""}
    ):
        raise ValueError("large variable D1 numeric precision differs")
    torch.set_float32_matmul_precision(
        str(numeric_precision["evaluation_float32_matmul_precision"])
    )
    corpus = load_corpus(args.corpus)
    expected_roles = {
        "fit": int(selected.get("fit_records", 0)),
        "probe": int(selected.get("probe_records", 0)),
        "qualification": int(selected["qualification_records"]),
    }
    role_names = (
        ("qualification",)
        if args.dataset == "qk"
        else ("fit", "probe", "qualification")
    )
    record_bindings = {
        f"{role}_{kind}_ids_sha256": array_sha256(
            corpus.arrays[name][corpus.role_records(role)]
        )
        for role in role_names
        for kind, name in (
            ("source", "record_source_ids"),
            ("user", "record_user_ids"),
        )
    }
    if (
        args.corpus != Path(selected["corpus"])
        or corpus.dataset != args.dataset
        or corpus.edge_count != int(selected["edge_count"])
        or corpus.file_sha256 != file_sha256(args.corpus)
        or corpus.metadata.get("roles") != expected_roles
        or corpus.metadata.get("minimum_initial_tokens")
        != int(config["minimum_initial_tokens"])
        or corpus.metadata.get("selection_salt")
        != config["record_selection_salt"]
        or record_bindings != selected["record_bindings"]
        or corpus.metadata.get("record_bindings") != record_bindings
        or corpus.metadata.get("quality_action_independence") is not True
        or corpus.metadata.get("positive_audit", {}).get(
            "all_edges_have_positive_targets"
        )
        is not True
        or len(set(corpus.arrays["record_valid_lengths"].tolist())) < 2
    ):
        raise ValueError("large variable D1 corpus differs")
    thresholds = [int(value) for value in config["score_thresholds"]]
    versions = [int(value) for value in selected["versions"]]
    edge_count = len(versions) - 1
    if args.edge_limit is not None:
        if not 1 <= args.edge_limit <= edge_count:
            raise ValueError("large variable D1 edge limit differs")
        edge_count = args.edge_limit
        versions = versions[: edge_count + 1]
    qualification_global, qualification_local = local_role_records(
        corpus,
        "qualification",
        rank,
        world_size,
        args.qualification_record_limit,
    )
    qualification = materialize_records(corpus, qualification_local)
    if args.dataset == "qb":
        fit_global, fit_local = local_role_records(corpus, "fit", rank, world_size)
        probe_global, probe_local = local_role_records(corpus, "probe", rank, world_size)
        fit_records = materialize_records(corpus, fit_local)
        probe_records = materialize_records(corpus, probe_local)
    else:
        fit_global = np.empty(0, dtype=np.int64)
        probe_global = np.empty(0, dtype=np.int64)
        fit_records = []
        probe_records = []
    if rank == 0:
        if args.output_root.exists():
            raise FileExistsError(args.output_root)
        args.output_root.mkdir(parents=True)
    dist.barrier()
    started = time.perf_counter()
    root = Path(selected["checkpoint_root"])
    spec = checkpoint_spec(root, versions[0])
    if (
        spec.max_seq_len != 512
        or spec.hidden_size != 1536
        or spec.num_layers != 24
        or spec.embedding_width != 4096
        or spec.num_heads != 24
        or spec.head_dim != 64
        or corpus.feature_fields != int(selected["feature_fields"])
    ):
        raise ValueError("large variable D1 large-model geometry differs")
    methods = ["reuse", "compiled", *[f"score_{value}" for value in thresholds], "exact"]
    dense, embedding, _ = load_model(
        root, versions[0], spec, rank, world_size, device
    )
    initial_qualification = exact_caches(
        args.dataset,
        dense,
        embedding,
        qualification,
        0,
        device,
        "initial_qualification_source_cache",
    )
    states = {
        method: {
            record.record_id: CacheState(
                cache=initial_qualification[record.record_id],
                depth=0,
                last_exact_version=versions[0],
            )
            for record in qualification
        }
        for method in methods
        if method != "exact"
    }
    del initial_qualification
    if args.dataset == "qb":
        initial_fit = exact_caches(
            args.dataset,
            dense,
            embedding,
            fit_records,
            0,
            device,
            "initial_fit_source_cache",
        )
        initial_probe = exact_caches(
            args.dataset,
            dense,
            embedding,
            probe_records,
            0,
            device,
            "initial_probe_source_cache",
        )
        fit_states = {
            record.record_id: CacheState(initial_fit[record.record_id], 0, versions[0])
            for record in fit_records
        }
        probe_states = {
            record.record_id: CacheState(initial_probe[record.record_id], 0, versions[0])
            for record in probe_records
        }
        del initial_fit, initial_probe
    else:
        fit_states = {}
        probe_states = {}
    score_states = {
        threshold: {int(record): 0 for record in qualification_global}
        for threshold in thresholds
    }
    exact_counts = {
        threshold: {int(record): 0 for record in qualification_global}
        for threshold in thresholds
    }
    checkpoint_bindings = [checkpoint_binding(root, versions[0])]
    edges = []
    for edge in range(edge_count):
        source_version = versions[edge]
        target_version = versions[edge + 1]
        edge_started = time.perf_counter()
        if args.dataset == "qb":
            if edge == 0:
                exact_source_fit = {
                    record.record_id: fit_states[record.record_id].cache
                    for record in fit_records
                }
            else:
                exact_source_fit = exact_caches(
                    args.dataset,
                    dense,
                    embedding,
                    fit_records,
                    edge,
                    device,
                    "fit_exact_source_cache",
                )
            dense.to("cpu")
            source_dense = dense
        else:
            source_dense = None
        del embedding
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dist.barrier()
        if args.dataset == "qk":
            del dense
            gc.collect()
        target_dense, target_embedding, _ = load_model(
            root, target_version, spec, rank, world_size, device
        )
        checkpoint_bindings.append(checkpoint_binding(root, target_version))
        if args.dataset == "qb":
            exact_target_fit = exact_caches(
                args.dataset,
                target_dense,
                target_embedding,
                fit_records,
                edge,
                device,
                "fit_exact_target_cache",
            )
            exact_target_probe = exact_caches(
                args.dataset,
                target_dense,
                target_embedding,
                probe_records,
                edge,
                device,
                "probe_exact_target_cache",
            )
            program, program_descriptor, fit_metrics = fit_program(
                source_dense,
                target_dense,
                exact_source_fit,
                fit_states,
                exact_target_fit,
                fit_records,
                edge,
                source_version,
                target_version,
                config["fit"],
                numeric_precision,
                args.output_root,
                rank,
                world_size,
                device,
            )
            probe_metrics = probe_program(
                program,
                probe_states,
                exact_target_probe,
                probe_records,
                device,
                rank,
                world_size,
            )
            del exact_source_fit, exact_target_fit, exact_target_probe
        else:
            program, program_descriptor = load_qk_program(
                selected["programs"][f"theta{source_version}_to_theta{target_version}"],
                source_version,
                target_version,
            )
            fit_metrics = None
            probe_metrics = None
        exact_ids, selection_ledgers = selection_for_edge(
            qualification_global,
            corpus,
            edge,
            thresholds,
            score_states,
        )
        for threshold in thresholds:
            for record in exact_ids[f"score_{threshold}"]:
                exact_counts[threshold][record] += 1
        next_states, metrics = evaluate_edge(
            args.dataset,
            target_dense,
            target_embedding,
            qualification,
            states,
            program,
            exact_ids,
            edge,
            target_version,
            int(config["negative_candidates"]),
            int(config["candidate_seed"]),
            spec,
            methods,
            rank,
            world_size,
            device,
        )
        if args.dataset == "qb":
            fit_states = advance_support_states(
                args.dataset,
                target_dense,
                target_embedding,
                fit_records,
                fit_states,
                program,
                edge,
                target_version,
                device,
            )
            probe_states = advance_support_states(
                args.dataset,
                target_dense,
                target_embedding,
                probe_records,
                probe_states,
                program,
                edge,
                target_version,
                device,
            )
        total_tokens = int(
            corpus.arrays["edge_prefix_lengths"][qualification_global, edge].sum()
        )
        edge_result = {
            "edge_ordinal": edge,
            "source_version": source_version,
            "target_version": target_version,
            "records": len(qualification_global),
            "valid_prefix_tokens": total_tokens,
            "prefix_length": {
                "minimum": int(
                    corpus.arrays["edge_prefix_lengths"][qualification_global, edge].min()
                ),
                "median": float(
                    np.median(
                        corpus.arrays["edge_prefix_lengths"][qualification_global, edge]
                    )
                ),
                "p95": float(
                    np.quantile(
                        corpus.arrays["edge_prefix_lengths"][qualification_global, edge],
                        0.95,
                    )
                ),
                "maximum": int(
                    corpus.arrays["edge_prefix_lengths"][qualification_global, edge].max()
                ),
            },
            "append_length": {
                "minimum": int(
                    np.diff(
                        corpus.arrays["edge_prefix_lengths"][qualification_global, edge : edge + 2],
                        axis=1,
                    ).min()
                ),
                "median": float(
                    np.median(
                        np.diff(
                            corpus.arrays["edge_prefix_lengths"][qualification_global, edge : edge + 2],
                            axis=1,
                        )
                    )
                ),
                "p95": float(
                    np.quantile(
                        np.diff(
                            corpus.arrays["edge_prefix_lengths"][qualification_global, edge : edge + 2],
                            axis=1,
                        ),
                        0.95,
                    )
                ),
                "maximum": int(
                    np.diff(
                        corpus.arrays["edge_prefix_lengths"][qualification_global, edge : edge + 2],
                        axis=1,
                    ).max()
                ),
            },
            "selection": {
                "reuse": {"exact_records": 0, "exact_valid_tokens": 0, "exact_valid_token_fraction": 0.0},
                "compiled": {"exact_records": 0, "exact_valid_tokens": 0, "exact_valid_token_fraction": 0.0},
                **{
                    method: {
                        **ledger,
                        "exact_valid_token_fraction": float(
                            ledger["scheduled_exact_valid_tokens"] / total_tokens
                        ),
                    }
                    for method, ledger in selection_ledgers.items()
                },
                "exact": {"exact_records": len(qualification_global), "exact_valid_tokens": total_tokens, "exact_valid_token_fraction": 1.0},
            },
            "program": program_descriptor,
            "fit": fit_metrics,
            "probe_cache_fidelity": probe_metrics,
            "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - edge_started,
        }
        edges.append(edge_result)
        states = next_states
        if source_dense is not None:
            del source_dense
        dense = target_dense
        embedding = target_embedding
        del target_dense, target_embedding
        del program
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if rank == 0:
        cumulative_selection = {}
        all_exact_tokens = sum(int(edge["valid_prefix_tokens"]) for edge in edges)
        for method in methods:
            if method in {"reuse", "compiled"}:
                tokens = 0
            elif method == "exact":
                tokens = all_exact_tokens
            else:
                tokens = sum(
                    int(edge["selection"][method]["scheduled_exact_valid_tokens"])
                    for edge in edges
                )
            cumulative_selection[method] = {
                "exact_valid_tokens": tokens,
                "all_exact_valid_tokens": all_exact_tokens,
                "exact_valid_token_fraction": tokens / all_exact_tokens,
                "exact_component_over_all_exact": tokens / all_exact_tokens,
                "edge_exact_valid_token_fractions": [
                    float(edge["selection"][method]["exact_valid_token_fraction"])
                    for edge in edges
                ],
            }
        result = {
            "schema": "evokv_large_variable_d1_score_sweep_result_v0",
            "status": "pass",
            "scientific_result": False,
            "formal_result": False,
            "dataset": args.dataset,
            "full_kv_payloads_persisted": 0,
            "serving_model_invariant": config["serving_model_invariant"],
            "large_model_only": True,
            "spec": asdict(spec),
            "versions": versions,
            "corpus": {
                "path": str(args.corpus),
                "sha256": corpus.file_sha256,
                "content_sha256": corpus.content_sha256,
                "metadata": corpus.metadata,
            },
            "roles": {
                "fit_records": len(fit_global),
                "probe_records": len(probe_global),
                "qualification_records": len(qualification_global),
                "pairwise_disjoint": True,
            },
            "checkpoint_bindings": checkpoint_bindings,
            "score_thresholds": thresholds,
            "score_definition": "S_i(next)=S_i(current)+current_valid_prefix_tokens; Exact resets S_i to zero; newly appended target-version tokens enter with zero debt",
            "selection_reads_quality": False,
            "edges": edges,
            "cumulative": {
                "selection": cumulative_selection,
                "recommendation": aggregate_cumulative(edges, methods),
                "fairness": fairness_summary(
                    qualification_global,
                    thresholds,
                    exact_counts,
                    score_states,
                    edge_count,
                    int(
                        corpus.arrays["edge_prefix_lengths"][
                            qualification_global, :edge_count
                        ].min()
                    ),
                ),
            },
            "execution": {
                "world_size": world_size,
                "device": str(device),
                "qualification_record_limit": args.qualification_record_limit,
                "edge_limit": args.edge_limit,
                "negative_candidates": int(config["negative_candidates"]),
                "numeric_precision": numeric_precision,
                "observed_float32_matmul_precision": torch.get_float32_matmul_precision(),
                "observed_nvidia_tf32_override": (
                    os.environ.get("NVIDIA_TF32_OVERRIDE") or "unset"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            },
            "bindings": {
                "config": {"path": str(args.config), "sha256": file_sha256(args.config)},
                "checkpoint_registry": config["bindings"]["checkpoint_registry"],
                "source_code": {"path": str(Path(__file__)), "sha256": file_sha256(Path(__file__))},
            },
        }
        atomic_json(args.output_root / "result.json", result)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "dataset": args.dataset,
                    "output": str(args.output_root / "result.json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    dist.barrier()
    del dense, embedding
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
