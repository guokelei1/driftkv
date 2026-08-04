from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.migration.program import compile_migration_program
from hstu_kvcache.migration.recursive_d1 import (
    RECURSIVE_D1_PROTOCOL,
    RECURSIVE_METHODS,
    RecursiveBatchState,
    action_plan_document,
    fit_rollout_aware_direct_program,
    mix_exact_cache,
    rollout_stability_certificate,
    select_depth_balanced_tokens,
    storage_cache,
    token_balanced_renewal,
    update_lineage,
)
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVProgram,
    compile_direct_oldkv_program,
    load_direct_oldkv_program,
    write_direct_oldkv_program,
)
from hstu_kvcache.migration.xp_d1_quality import (
    apply_direct_oldkv,
    cache_relative_error,
    direct_program_sha256,
    recommendation_sums,
    summarize_recommendation_sums,
)
from hstu_kvcache.migration.xp_exact_baseline import (
    canonical_sha256,
    file_sha256,
    load_fixed_inputs,
    load_inference_checkpoint,
)
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.streaming.sharded_edge import fixed_candidate_ids
from hstu_kvcache.streaming.trainer import build_next_item_targets
from hstu_kvcache.streaming.xp_multiversion import (
    XPUpdateWindow,
    _materialize_window_record,
)
from hstu_kvcache.streaming.xp_version_training import (
    XPFixedEdgeCorpus,
    load_xp_fixed_edge_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=RECURSIVE_METHODS, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--qualification-record-limit", type=int)
    parser.add_argument("--edge-limit", type=int, choices=(1, 2, 3))
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"recursive D1 artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


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
        group_args = {}
        if args.backend == "nccl" and device.type == "cuda":
            group_args["device_id"] = device
        dist.init_process_group(
            backend=args.backend,
            rank=rank,
            world_size=world_size,
            **group_args,
        )
    return rank, world_size, device


def gather_objects(value: object, world_size: int) -> list[object]:
    values: list[object] = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def broadcast_object(value: object, rank: int) -> object:
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def resolve(repository_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repository_root / path


def load_round_config(
    path: Path,
    *,
    method: str,
    world_size: int,
) -> tuple[dict[str, object], Path]:
    value = json.loads(path.read_text())
    repository_root = Path(__file__).resolve().parents[1]
    if (
        value.get("schema")
        != "evokv_qk_recursive_d1_round_a_two_gpu_development_v0"
        or value.get("status") != "ready_for_user_execution"
        or value.get("scientific_result") is not False
        or value.get("formal_result") is not False
        or int(value.get("world_size", -1)) != world_size
        or world_size != 2
        or method not in value.get("methods", [])
        or value.get("serving_model_invariant", {}).get(
            "concurrent_recommendation_models"
        )
        != 1
    ):
        raise ValueError("recursive D1 round config differs")
    for binding in value["bindings"].values():
        if not isinstance(binding, Mapping) or "path" not in binding:
            continue
        binding_path = resolve(repository_root, binding["path"])
        if (
            not binding_path.is_file()
            or "sha256" in binding
            and file_sha256(binding_path) != str(binding["sha256"])
        ):
            raise ValueError(
                f"recursive D1 binding differs: {binding_path}"
            )
    return value, repository_root


def selected_role_records(
    corpus: XPFixedEdgeCorpus,
    config: Mapping[str, object],
    *,
    qualification_record_limit: int | None,
) -> dict[str, list[int]]:
    roles = config["roles"]
    source_role = str(roles["fit_and_probe_source_role"])
    salt = str(roles["selection_salt"])
    source = [int(value) for value in corpus.role_records(source_role)]
    ordered = sorted(
        source,
        key=lambda record_id: (
            hashlib.sha256(f"{salt}:{record_id}".encode()).digest(),
            record_id,
        ),
    )
    fit_count = int(roles["fit_records_global"])
    probe_count = int(roles["stability_probe_records_global"])
    qualification = [
        int(value)
        for value in corpus.role_records(str(roles["qualification_role"]))
    ]
    if qualification_record_limit is not None:
        if qualification_record_limit < 8:
            raise ValueError("recursive D1 qualification limit is too small")
        qualification = qualification[:qualification_record_limit]
    result = {
        "fit": sorted(ordered[:fit_count]),
        "stability_probe": sorted(
            ordered[fit_count : fit_count + probe_count]
        ),
        "qualification": sorted(qualification),
    }
    for name, records in result.items():
        expected = roles.get(f"{name}_record_ids_sha256")
        observed = canonical_sha256({"record_ids": records})
        if (
            name != "qualification" or qualification_record_limit is None
        ) and observed != expected:
            raise ValueError(f"recursive D1 {name} role hash differs")
    if any(
        set(result[left]) & set(result[right])
        for left, right in (
            ("fit", "stability_probe"),
            ("fit", "qualification"),
            ("stability_probe", "qualification"),
        )
    ):
        raise ValueError("recursive D1 roles overlap")
    return result


def build_bound_batches(
    corpus: XPFixedEdgeCorpus,
    records: Sequence[int],
    update: XPUpdateWindow,
    *,
    max_seq_len: int,
    batch_size_per_rank: int,
    rank: int,
    world_size: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, object]]:
    if (
        not records
        or len(records) != len(set(records))
        or batch_size_per_rank < 1
        or update.update_end > max_seq_len
    ):
        raise ValueError("recursive D1 bound batch request differs")
    global_batch = batch_size_per_rank * world_size
    steps = math.ceil(len(records) / global_batch)
    materialized = {
        int(record): _materialize_window_record(
            corpus,
            int(record),
            update.history_end,
            update.update_end,
            max_seq_len,
        )
        for record in records
    }
    batches = []
    local_real = 0
    local_targets = 0
    for step in range(steps):
        left = step * global_batch + rank * batch_size_per_rank
        right = min(left + batch_size_per_rank, len(records))
        selected = records[left:right]
        values = [materialized[int(record)] for record in selected]
        local_real += len(values)
        while len(values) < batch_size_per_rank:
            values.append(
                {
                    "item_ids": torch.zeros(
                        update.update_end, dtype=torch.int64
                    ),
                    "behaviors": torch.zeros(
                        update.update_end, dtype=torch.int64
                    ),
                    "time_deltas": torch.zeros(
                        update.update_end, dtype=torch.float32
                    ),
                    "labels": torch.zeros(
                        update.update_end, dtype=torch.int64
                    ),
                    "train_mask": torch.zeros(
                        update.update_end, dtype=torch.bool
                    ),
                    "length": torch.tensor(0, dtype=torch.int64),
                    "record": torch.tensor(-1, dtype=torch.int64),
                    "window_start": torch.tensor(0, dtype=torch.int64),
                }
            )
        batch = {
            name: torch.stack([value[name] for value in values])
            for name in (
                "item_ids",
                "behaviors",
                "time_deltas",
                "labels",
                "train_mask",
            )
        }
        batch["lengths"] = torch.stack(
            [value["length"] for value in values]
        )
        batch["record_indices"] = torch.stack(
            [value["record"] for value in values]
        )
        batch["window_starts"] = torch.stack(
            [value["window_start"] for value in values]
        )
        _, valid = build_next_item_targets(
            batch["item_ids"],
            batch["lengths"],
            batch["labels"],
            batch["train_mask"],
        )
        local_targets += int(valid.sum())
        batches.append(batch)
    return batches, {
        "global_records": len(records),
        "global_record_ids_sha256": canonical_sha256(
            {"record_ids": sorted(int(value) for value in records)}
        ),
        "steps_per_rank": steps,
        "batch_size_per_rank": batch_size_per_rank,
        "local_real_records": local_real,
        "local_padding_records": steps * batch_size_per_rank - local_real,
        "local_positive_targets": local_targets,
        "history_end": update.history_end,
        "evaluation_end": update.update_end,
    }


def prefix_lengths(
    batch: Mapping[str, torch.Tensor], prefix_width: int
) -> torch.Tensor:
    records = batch["record_indices"]
    return torch.where(
        records >= 0,
        torch.full_like(records, prefix_width),
        torch.zeros_like(records),
    )


@torch.no_grad()
def exact_prefix_cache(
    dense,
    embedding,
    batch: Mapping[str, torch.Tensor],
    prefix_width: int,
    device: torch.device,
) -> HSTUKVCache:
    lengths = prefix_lengths(batch, prefix_width).to(device)
    item_ids = batch["item_ids"][:, :prefix_width].to(device)
    behaviors = batch["behaviors"][:, :prefix_width].to(device)
    deltas = batch["time_deltas"][:, :prefix_width].to(device)
    vectors = embedding(item_ids, lengths)
    cache = dense.core.compute_kv_from_item_embeddings(
        vectors,
        behaviors,
        deltas,
        lengths,
    )
    return storage_cache(cache)


def materialize_exact_role(
    dense,
    embedding,
    batches: Sequence[Mapping[str, torch.Tensor]],
    prefix_width: int,
    device: torch.device,
    *,
    phase: str,
) -> list[HSTUKVCache]:
    caches = []
    for index, batch in enumerate(batches):
        caches.append(
            exact_prefix_cache(
                dense, embedding, batch, prefix_width, device
            )
        )
        if index == 0 or (index + 1) % 64 == 0 or index + 1 == len(batches):
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "batch": index + 1,
                        "batches": len(batches),
                    }
                ),
                flush=True,
            )
    return caches


def states_from_exact(
    caches: Sequence[HSTUKVCache],
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    exact_version: int,
) -> list[RecursiveBatchState]:
    return [
        RecursiveBatchState(
            cache=cache,
            record_ids=batch["record_indices"].clone(),
            depths=torch.zeros_like(batch["record_indices"]),
            last_exact_versions=torch.where(
                batch["record_indices"] >= 0,
                torch.full_like(batch["record_indices"], exact_version),
                torch.full_like(batch["record_indices"], -1),
            ),
        )
        for cache, batch in zip(caches, batches, strict=True)
    ]


def recursive_state_binding(
    states: Sequence[RecursiveBatchState],
    *,
    version: int,
    rank_id: int,
    world_size: int,
) -> dict[str, object]:
    digest = hashlib.sha256()
    real_records = 0
    for state in states:
        digest.update(np.asarray([state.cache.seq_len], dtype="<i8").tobytes())
        for tensor in (
            state.record_ids,
            state.depths,
            state.last_exact_versions,
            state.cache.k,
            state.cache.v,
        ):
            array = tensor.detach().contiguous().numpy()
            digest.update(array.dtype.str.encode())
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(memoryview(array).cast("B"))
        real_records += int((state.record_ids >= 0).sum())
    local = {
        "rank": rank_id,
        "sha256": digest.hexdigest(),
        "batches": len(states),
        "real_records": real_records,
    }
    gathered = gather_objects(local, world_size)
    if rank_id == 0:
        binding = {
            "version": version,
            "storage_dtype": "torch.float16",
            "rank_states": gathered,
            "sha256": canonical_sha256(
                {
                    "version": version,
                    "storage_dtype": "torch.float16",
                    "rank_states": gathered,
                }
            ),
        }
    else:
        binding = None
    return broadcast_object(binding, rank_id)


def validate_state_batches(
    states: Sequence[RecursiveBatchState],
    batches: Sequence[Mapping[str, torch.Tensor]],
    prefix_width: int,
) -> None:
    if len(states) != len(batches):
        raise ValueError("recursive D1 state batch count differs")
    for state, batch in zip(states, batches, strict=True):
        if (
            state.cache.seq_len != prefix_width
            or not torch.equal(state.record_ids, batch["record_indices"])
        ):
            raise ValueError("recursive D1 state lineage differs")


def method_requires_fit(method: str) -> bool:
    return method in {
        "rollout_only_exact0",
        "ract_kv_exact0",
        "ract_kv_exact10",
        "ract_kv_exact20",
    }


def method_requires_program(method: str) -> bool:
    return method != "reuse_exact_baselines"


def method_colors(method: str) -> int | None:
    if method == "ract_kv_exact10":
        return 10
    if method == "ract_kv_exact20":
        return 5
    return None


def role_renewal(
    records: Sequence[int],
    *,
    prefix_width: int,
    role: str,
    method: str,
    edge_ordinal: int,
    fallback_all_exact: bool,
    salt: str,
) -> tuple[set[int], dict[str, object]]:
    scheduled, ledger = token_balanced_renewal(
        [(int(record), prefix_width, "homogeneous_quality_prefix") for record in records],
        colors=method_colors(method),
        edge_ordinal=edge_ordinal,
        salt=f"{salt}:{role}",
    )
    if fallback_all_exact:
        exact = set(int(value) for value in records)
        fallback = exact - scheduled
    else:
        exact = scheduled
        fallback = set()
    fallback_tokens = len(fallback) * prefix_width
    ledger = {
        **ledger,
        "fallback_exact_records": len(fallback),
        "fallback_exact_valid_tokens": fallback_tokens,
        "total_d1_exact_records": len(exact),
        "total_d1_exact_valid_tokens": len(exact) * prefix_width,
        "total_d1_exact_valid_token_fraction": len(exact) / len(records),
        "fallback_consumes_budget": True,
        "budget_admitted": (
            not fallback_all_exact
            and bool(ledger["within_integer_token_cap"])
        ),
    }
    return exact, ledger


def _float_cache(cache: HSTUKVCache, device: torch.device) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.to(device=device, dtype=torch.float32),
        v=cache.v.to(device=device, dtype=torch.float32),
        seq_len=cache.seq_len,
    )


@torch.no_grad()
def append_suffix(
    dense,
    embedding,
    batch: Mapping[str, torch.Tensor],
    cache: HSTUKVCache,
    prefix_width: int,
    next_prefix_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, HSTUKVCache]:
    suffix_ids = batch["item_ids"][:, prefix_width:next_prefix_width].to(
        device
    )
    suffix_behaviors = batch["behaviors"][
        :, prefix_width:next_prefix_width
    ].to(device)
    suffix_deltas = batch["time_deltas"][
        :, prefix_width:next_prefix_width
    ].to(device)
    suffix_lengths = torch.where(
        batch["record_indices"].to(device) >= 0,
        torch.full(
            (len(batch["record_indices"]),),
            next_prefix_width - prefix_width,
            dtype=torch.int64,
            device=device,
        ),
        torch.zeros(
            len(batch["record_indices"]),
            dtype=torch.int64,
            device=device,
        ),
    )
    vectors = embedding(suffix_ids, suffix_lengths)
    hidden, updated = dense.core.forward_with_cache_from_item_embeddings(
        _float_cache(cache, device),
        vectors,
        suffix_behaviors,
        suffix_deltas,
    )
    return hidden, updated


def score_candidates(
    hidden: torch.Tensor,
    candidate_vectors: torch.Tensor,
    suffix_valid: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum(
        "nh,nch->nc",
        hidden[suffix_valid],
        candidate_vectors,
    )


def score_contributions(
    scores: torch.Tensor,
) -> tuple[list[int], list[float]]:
    positive = scores[:, :1]
    ranks = 1 + (scores[:, 1:] >= positive).sum(dim=1)
    losses = torch.nn.functional.cross_entropy(
        scores,
        torch.zeros(
            scores.shape[0], dtype=torch.int64, device=scores.device
        ),
        reduction="none",
    )
    return (
        [int(value) for value in ranks.detach().cpu().tolist()],
        [float(value) for value in losses.detach().cpu().double().tolist()],
    )


def add_sums(
    destination: dict[str, float], source: Mapping[str, float | int]
) -> None:
    for key, value in source.items():
        destination[key] += float(value)


def cache_error_rows(
    cache: HSTUKVCache,
    exact: HSTUKVCache,
    lengths: torch.Tensor,
    record_ids: torch.Tensor,
) -> dict[int, float]:
    errors = cache_relative_error(cache, exact, lengths)
    return {
        int(record_id): float(error)
        for record_id, error in zip(
            record_ids.tolist(), errors.detach().cpu().tolist(), strict=True
        )
        if int(record_id) >= 0
    }


def summarize_errors(values: Sequence[float]) -> dict[str, object]:
    if not values:
        raise ValueError("recursive D1 cache errors are empty")
    ordered = sorted(float(value) for value in values)
    return {
        "records": len(ordered),
        "relative_error_mean": float(sum(ordered) / len(ordered)),
        "relative_error_p90": float(
            ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]
        ),
        "relative_error_max": float(ordered[-1]),
    }


def prepare_program(
    *,
    method: str,
    edge_name: str,
    edge_ordinal: int,
    source_version: int,
    target_version: int,
    source_dense,
    target_dense,
    fit_batches: Sequence[Mapping[str, torch.Tensor]],
    fit_states: Sequence[RecursiveBatchState],
    exact_source_fit: Sequence[HSTUKVCache],
    exact_target_fit: Sequence[HSTUKVCache],
    config: Mapping[str, object],
    repository_root: Path,
    output_root: Path,
    rank_id: int,
    world_size: int,
    device: torch.device,
) -> tuple[
    DirectOldKVProgram,
    DirectOldKVProgram | None,
    dict[str, object],
    dict[str, object] | None,
]:
    if method == "incumbent_rank16_recursive":
        descriptor = config["incumbent_programs"][edge_name]
        path = resolve(repository_root, descriptor["path"])
        program, payload = load_direct_oldkv_program(
            path,
            expected_sha256=str(descriptor["sha256"]),
            expected_source_version=f"theta{source_version}",
            expected_target_version=f"theta{target_version}",
        )
        return program, None, {
            "path": str(path),
            "sha256": str(descriptor["sha256"]),
            "bytes": path.stat().st_size,
            "source": "retained_one_edge_rank16_incumbent",
            "payload_provenance": payload.get("provenance"),
        }, None
    source_dense.to(device)
    analytic = compile_migration_program(
        target_dense.core,
        f"theta{source_version}",
        f"theta{target_version}",
    )
    base, base_metrics = compile_direct_oldkv_program(
        source_dense.core, analytic
    )
    source_dense.to("cpu")
    del analytic
    fit_lengths = [
        prefix_lengths(batch, fit_states[index].cache.seq_len)
        for index, batch in enumerate(fit_batches)
    ]
    selected, sampling = select_depth_balanced_tokens(
        [state.record_ids for state in fit_states],
        fit_lengths,
        [state.depths for state in fit_states],
        maximum_global_tokens=int(config["fit"]["maximum_global_tokens_per_layer"]),
        seed=int(config["fit"]["sampling_seed"]) + edge_ordinal * 1_000_003,
    )
    fit_mode = "rollout_only" if method == "rollout_only_exact0" else "ract_kv"
    float_program, program, fit_metrics = fit_rollout_aware_direct_program(
        base,
        exact_source_fit,
        [state.cache for state in fit_states],
        exact_target_fit,
        fit_lengths,
        selected,
        mode=fit_mode,
        rank=int(config["fit"]["rank"]),
        ridge=float(config["fit"]["ridge"]),
        seed=int(config["fit"]["solver_seed"]) + edge_ordinal * 1_000_003,
        device=device,
        maximum_jitter_multiplier=float(
            config["fit"]["maximum_adaptive_jitter_over_requested"]
        ),
    )
    fit_metrics = {
        **fit_metrics,
        "sampling": sampling,
        "fit_records_global": int(config["roles"]["fit_records_global"]),
    }
    identity = direct_program_sha256(program)
    identities = gather_objects(identity, world_size)
    if len(set(str(value) for value in identities)) != 1:
        raise RuntimeError("recursive D1 program differs across ranks")
    path = output_root / "programs" / f"{edge_name}_direct_oldkv_fp16.pt"
    if rank_id == 0:
        descriptor = write_direct_oldkv_program(
            program,
            path,
            provenance={
                "protocol": RECURSIVE_D1_PROTOCOL,
                "method": method,
                "edge": edge_name,
                "program_identity_sha256": identity,
                "fit_metrics": fit_metrics,
            },
            compile_metrics=base_metrics,
        )
        descriptor["identity_sha256"] = identity
        descriptor["source"] = "round_a_rollout_aware_direct_kv_fit"
    else:
        descriptor = None
    descriptor = broadcast_object(descriptor, rank_id)
    return program, float_program, descriptor, fit_metrics


def probe_certificate(
    *,
    method: str,
    program: DirectOldKVProgram,
    float_program: DirectOldKVProgram | None,
    exact_source_probe: Sequence[HSTUKVCache],
    probe_states: Sequence[RecursiveBatchState],
    exact_target_probe: Sequence[HSTUKVCache],
    probe_batches: Sequence[Mapping[str, torch.Tensor]],
    config: Mapping[str, object],
    device: torch.device,
) -> dict[str, object] | None:
    if method not in {
        "rollout_only_exact0",
        "ract_kv_exact0",
        "ract_kv_exact10",
        "ract_kv_exact20",
    }:
        return None
    if float_program is None:
        raise RuntimeError("recursive D1 float program is absent")
    lengths = [
        prefix_lengths(batch, state.cache.seq_len)
        for batch, state in zip(probe_batches, probe_states, strict=True)
    ]
    return rollout_stability_certificate(
        float_program,
        program,
        exact_source_probe,
        [state.cache for state in probe_states],
        exact_target_probe,
        lengths,
        [state.depths for state in probe_states],
        target_ratio=float(config["gates"]["stability_bound_target"]),
        hard_ratio=float(config["gates"]["stability_bound_hard_limit"]),
        device=device,
    )


@torch.no_grad()
def advance_role(
    *,
    dense,
    embedding,
    batches: Sequence[Mapping[str, torch.Tensor]],
    states: Sequence[RecursiveBatchState],
    exact_targets: Sequence[HSTUKVCache],
    program: DirectOldKVProgram | None,
    exact_record_ids: set[int],
    source_version: int,
    target_version: int,
    prefix_width: int,
    next_prefix_width: int,
    method: str,
    fallback_all_exact: bool,
    device: torch.device,
) -> tuple[list[RecursiveBatchState], list[dict[str, object]]]:
    validate_state_batches(states, batches, prefix_width)
    if len(exact_targets) != len(states):
        raise ValueError("recursive D1 role target count differs")
    program_device = None if program is None else program.to(device)
    next_states = []
    action_rows = []
    for index, (batch, state, exact_cpu) in enumerate(
        zip(batches, states, exact_targets, strict=True)
    ):
        deployed = state.cache.to(device)
        exact = exact_cpu.to(device)
        if program_device is None:
            post = deployed
            nonexact_action = "reuse"
        else:
            compiled = apply_direct_oldkv(program_device, deployed)
            exact_mask = torch.tensor(
                [
                    int(record_id) in exact_record_ids
                    for record_id in state.record_ids.tolist()
                ],
                dtype=torch.bool,
                device=device,
            )
            post = mix_exact_cache(compiled, exact, exact_mask)
            nonexact_action = "compiled"
        _, updated = append_suffix(
            dense,
            embedding,
            batch,
            post,
            prefix_width,
            next_prefix_width,
            device,
        )
        depths, last_exact, rows = update_lineage(
            state,
            target_version=target_version,
            exact_record_ids=exact_record_ids,
            action_for_nonexact=nonexact_action,
        )
        for row in rows:
            if row["action"] == "exact":
                row["reason"] = (
                    "fallback_certificate"
                    if fallback_all_exact
                    else "scheduled_renewal"
                )
            elif row["action"] == "reuse":
                row["reason"] = "no_maintenance_baseline"
            else:
                row["reason"] = "compiled_program"
            row["source_version"] = source_version
            row["target_version"] = target_version
            row["retained_prefix_tokens"] = prefix_width
        action_rows.extend(rows)
        next_states.append(
            RecursiveBatchState(
                cache=storage_cache(updated),
                record_ids=state.record_ids,
                depths=depths,
                last_exact_versions=last_exact,
            )
        )
        if index == 0 or (index + 1) % 64 == 0 or index + 1 == len(states):
            print(
                json.dumps(
                    {
                        "phase": "advance_role",
                        "method": method,
                        "batch": index + 1,
                        "batches": len(states),
                    }
                ),
                flush=True,
            )
    return next_states, action_rows


@torch.no_grad()
def evaluate_qualification(
    *,
    dense,
    embedding,
    batches: Sequence[Mapping[str, torch.Tensor]],
    states: Sequence[RecursiveBatchState],
    exact_source_caches: Sequence[HSTUKVCache] | None,
    program: DirectOldKVProgram | None,
    exact_record_ids: set[int],
    source_version: int,
    target_version: int,
    prefix_width: int,
    next_prefix_width: int,
    method: str,
    fallback_all_exact: bool,
    negative_count: int,
    candidate_seed: int,
    rank_id: int,
    world_size: int,
    device: torch.device,
) -> tuple[list[RecursiveBatchState], list[dict[str, object]], dict[str, object]]:
    validate_state_batches(states, batches, prefix_width)
    if exact_source_caches is not None and len(exact_source_caches) != len(states):
        raise ValueError("recursive D1 qualification source count differs")
    program_device = None if program is None else program.to(device)
    metric_names = ("pre", "source", "post", "oracle", "exact")
    local_sums = {name: defaultdict(float) for name in metric_names}
    errors = {name: {} for name in metric_names}
    contributions = []
    action_rows = []
    next_states = []
    candidate_digest = hashlib.sha256()
    for index, (batch, state) in enumerate(zip(batches, states, strict=True)):
        exact_cpu = exact_prefix_cache(
            dense, embedding, batch, prefix_width, device
        )
        deployed = state.cache.to(device)
        exact = exact_cpu.to(device)
        if program_device is None:
            source = deployed
            post = deployed
            oracle = exact
            nonexact_action = "reuse"
        else:
            compiled = apply_direct_oldkv(program_device, deployed)
            exact_mask = torch.tensor(
                [
                    int(record_id) in exact_record_ids
                    for record_id in state.record_ids.tolist()
                ],
                dtype=torch.bool,
                device=device,
            )
            post = mix_exact_cache(compiled, exact, exact_mask)
            if exact_source_caches is None:
                raise RuntimeError("recursive D1 oracle source is absent")
            source = exact_source_caches[index].to(device)
            oracle_compiled = apply_direct_oldkv(
                program_device, source
            )
            oracle = mix_exact_cache(oracle_compiled, exact, exact_mask)
            nonexact_action = "compiled"
        lengths = prefix_lengths(batch, prefix_width).to(device)
        record_ids = batch["record_indices"]
        errors["pre"].update(
            cache_error_rows(deployed, exact, lengths, record_ids)
        )
        errors["source"].update(
            cache_error_rows(source, exact, lengths, record_ids)
        )
        errors["post"].update(cache_error_rows(post, exact, lengths, record_ids))
        errors["oracle"].update(
            cache_error_rows(oracle, exact, lengths, record_ids)
        )
        errors["exact"].update(
            cache_error_rows(exact, exact, lengths, record_ids)
        )
        hidden_pre, _ = append_suffix(
            dense,
            embedding,
            batch,
            deployed,
            prefix_width,
            next_prefix_width,
            device,
        )
        hidden_post, updated_post = append_suffix(
            dense,
            embedding,
            batch,
            post,
            prefix_width,
            next_prefix_width,
            device,
        )
        if program_device is None:
            hidden_source = hidden_pre
            hidden_oracle = None
        else:
            hidden_source, _ = append_suffix(
                dense,
                embedding,
                batch,
                source,
                prefix_width,
                next_prefix_width,
                device,
            )
            hidden_oracle, _ = append_suffix(
                dense,
                embedding,
                batch,
                oracle,
                prefix_width,
                next_prefix_width,
                device,
            )
        hidden_exact, _ = append_suffix(
            dense,
            embedding,
            batch,
            exact,
            prefix_width,
            next_prefix_width,
            device,
        )
        item_ids = batch["item_ids"].to(device)
        labels = batch["labels"].to(device)
        train_mask = batch["train_mask"].to(device)
        full_lengths = batch["lengths"].to(device)
        targets, valid = build_next_item_targets(
            item_ids, full_lengths, labels, train_mask
        )
        suffix_valid = valid[:, prefix_width:next_prefix_width]
        positives = targets[:, prefix_width:next_prefix_width][suffix_valid]
        candidates = fixed_candidate_ids(
            positives,
            int(dense.core.cfg.num_prediction_items),
            negative_count,
            candidate_seed + index * world_size + rank_id,
        )
        candidate_digest.update(
            np.asarray(candidates.shape, dtype="<i8").tobytes()
        )
        candidate_digest.update(
            np.asarray(
                candidates.detach().cpu(), dtype="<i8"
            ).tobytes()
        )
        candidate_lengths = torch.full(
            (len(candidates),),
            candidates.shape[1],
            dtype=torch.int64,
            device=device,
        )
        candidate_vectors = embedding(candidates, candidate_lengths)
        scores = {
            "pre": score_candidates(
                hidden_pre, candidate_vectors, suffix_valid
            ),
            "source": score_candidates(
                hidden_source, candidate_vectors, suffix_valid
            ),
            "post": score_candidates(
                hidden_post, candidate_vectors, suffix_valid
            ),
            "exact": score_candidates(
                hidden_exact, candidate_vectors, suffix_valid
            ),
        }
        scores["oracle"] = (
            scores["exact"]
            if hidden_oracle is None
            else score_candidates(
                hidden_oracle, candidate_vectors, suffix_valid
            )
        )
        for name in scores:
            add_sums(
                local_sums[name],
                recommendation_sums(scores[name], scores["exact"]),
            )
        target_record_ids = (
            record_ids.to(device)[:, None]
            .expand(-1, suffix_valid.shape[1])[suffix_valid]
            .detach()
            .cpu()
            .tolist()
        )
        offsets = (
            torch.arange(
                1,
                suffix_valid.shape[1] + 1,
                dtype=torch.int64,
                device=device,
            )[None]
            .expand_as(suffix_valid)[suffix_valid]
            .detach()
            .cpu()
            .tolist()
        )
        score_rows = {
            name: score_contributions(value) for name, value in scores.items()
        }
        for row_index, (record_id, offset) in enumerate(
            zip(target_record_ids, offsets, strict=True)
        ):
            contributions.append(
                {
                    "record_id": int(record_id),
                    "suffix_offset": int(offset),
                    **{
                        f"{name}_rank": score_rows[name][0][row_index]
                        for name in score_rows
                    },
                    **{
                        f"{name}_sampled_cross_entropy": score_rows[name][1][
                            row_index
                        ]
                        for name in score_rows
                    },
                }
            )
        depths, last_exact, rows = update_lineage(
            state,
            target_version=target_version,
            exact_record_ids=exact_record_ids,
            action_for_nonexact=nonexact_action,
        )
        for row in rows:
            if row["action"] == "exact":
                row["reason"] = (
                    "fallback_certificate"
                    if fallback_all_exact
                    else "scheduled_renewal"
                )
            elif row["action"] == "reuse":
                row["reason"] = "no_maintenance_baseline"
            else:
                row["reason"] = "compiled_program"
            row["source_version"] = source_version
            row["target_version"] = target_version
            row["retained_prefix_tokens"] = prefix_width
        action_rows.extend(rows)
        next_states.append(
            RecursiveBatchState(
                cache=storage_cache(updated_post),
                record_ids=state.record_ids,
                depths=depths,
                last_exact_versions=last_exact,
            )
        )
        if index == 0 or (index + 1) % 64 == 0 or index + 1 == len(states):
            print(
                json.dumps(
                    {
                        "phase": "qualification",
                        "method": method,
                        "edge": f"theta{source_version}_to_theta{target_version}",
                        "batch": index + 1,
                        "batches": len(states),
                    }
                ),
                flush=True,
            )
    gathered_sums = gather_objects(
        {name: dict(value) for name, value in local_sums.items()}, world_size
    )
    gathered_errors = gather_objects(errors, world_size)
    gathered_contributions = gather_objects(contributions, world_size)
    candidate_hashes = gather_objects(candidate_digest.hexdigest(), world_size)
    if rank_id == 0:
        merged_sums = {
            name: defaultdict(float) for name in metric_names
        }
        for rank_sums in gathered_sums:
            for name, values in rank_sums.items():
                add_sums(merged_sums[name], values)
        merged_errors = {name: {} for name in metric_names}
        for rank_errors in gathered_errors:
            for name, values in rank_errors.items():
                overlap = set(merged_errors[name]) & set(values)
                if overlap:
                    raise RuntimeError("recursive D1 cache errors overlap")
                merged_errors[name].update(values)
        merged_contributions = sorted(
            [
                row
                for rank_rows in gathered_contributions
                for row in rank_rows
            ],
            key=lambda value: (
                int(value["record_id"]), int(value["suffix_offset"])
            ),
        )
        if len(merged_contributions) != len(
            {
                (int(value["record_id"]), int(value["suffix_offset"]))
                for value in merged_contributions
            }
        ):
            raise RuntimeError("recursive D1 target contributions overlap")
        recommendation = {
            name: summarize_recommendation_sums(values)
            for name, values in merged_sums.items()
        }
        fidelity = {
            name: summarize_errors(list(values.values()))
            for name, values in merged_errors.items()
        }
        denominator = (
            float(recommendation["pre"]["sampled_cross_entropy"])
            - float(recommendation["exact"]["sampled_cross_entropy"])
        )
        recovery = (
            None
            if denominator <= 0
            else (
                float(recommendation["pre"]["sampled_cross_entropy"])
                - float(recommendation["post"]["sampled_cross_entropy"])
            )
            / denominator
        )
        oracle_recovery = (
            None
            if denominator <= 0
            else (
                float(recommendation["pre"]["sampled_cross_entropy"])
                - float(recommendation["oracle"]["sampled_cross_entropy"])
            )
            / denominator
        )
        pre_error = float(fidelity["pre"]["relative_error_mean"])
        fidelity_recovery = (
            None
            if pre_error <= 0
            else 1.0
            - float(fidelity["post"]["relative_error_mean"]) / pre_error
        )
        quality = {
            "recommendation": recommendation,
            "cache_fidelity": fidelity,
            "edge_ce_recovery": recovery,
            "oracle_reset_ce_recovery": oracle_recovery,
            "oracle_reset_gap_percentage_points": (
                None
                if recovery is None or oracle_recovery is None
                else 100.0 * (oracle_recovery - recovery)
            ),
            "mean_kv_fidelity_recovery": fidelity_recovery,
            "paired_target_contributions": merged_contributions,
            "paired_target_key_sha256": canonical_sha256(
                {
                    "keys": [
                        [value["record_id"], value["suffix_offset"]]
                        for value in merged_contributions
                    ]
                }
            ),
            "candidate_sha256_per_rank": candidate_hashes,
        }
    else:
        quality = None
    quality = broadcast_object(quality, rank_id)
    return next_states, action_rows, quality


def merge_action_rows(
    local_rows: Sequence[dict[str, object]], world_size: int, rank_id: int
) -> list[dict[str, object]] | None:
    gathered = gather_objects(list(local_rows), world_size)
    if rank_id != 0:
        return None
    merged = sorted(
        [row for rank_rows in gathered for row in rank_rows],
        key=lambda value: int(value["record_id"]),
    )
    if len(merged) != len({int(value["record_id"]) for value in merged}):
        raise RuntimeError("recursive D1 action coverage overlaps")
    return merged


def main() -> None:
    args = parse_args()
    rank_id, world_size, device = initialize(args)
    config_path = Path(args.config)
    config, repository_root = load_round_config(
        config_path, method=args.method, world_size=world_size
    )
    output_root = Path(args.output_root)
    if rank_id == 0:
        if (output_root / "method_summary.json").exists():
            raise FileExistsError("recursive D1 method is already complete")
        output_root.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    benchmark_path = resolve(
        repository_root, config["bindings"]["benchmark"]["path"]
    )
    inputs = load_fixed_inputs(
        benchmark_path,
        str(config["capacity_name"]),
        world_size=world_size,
    )
    edge_descriptor = inputs.benchmark["data"]["fixed_edge_inputs"]
    corpus = load_xp_fixed_edge_corpus(
        resolve(repository_root, edge_descriptor["path"]),
        resolve(repository_root, edge_descriptor["summary_path"]),
        num_embeddings=inputs.spec.num_embeddings,
        num_prediction_items=inputs.spec.num_prediction_items,
        num_behaviors=inputs.spec.num_behaviors,
    )
    role_records = selected_role_records(
        corpus,
        config,
        qualification_record_limit=args.qualification_record_limit,
    )
    active_roles = (
        ("qualification",)
        if not method_requires_fit(args.method)
        else ("fit", "stability_probe", "qualification")
    )
    checkpoint_root = resolve(
        repository_root, config["checkpoint_root"]
    )
    edges = [dict(value) for value in config["edges"]]
    if args.edge_limit is not None:
        edges = edges[: args.edge_limit]
    first = edges[0]
    first_update = XPUpdateWindow(
        source_version=int(first["source_version"]),
        target_version=int(first["target_version"]),
        history_end=int(first["history_end"]),
        update_end=int(first["evaluation_end"]),
    )
    batches_by_role = {}
    audits_by_role = {}
    for role in active_roles:
        batches, audit = build_bound_batches(
            corpus,
            role_records[role],
            first_update,
            max_seq_len=inputs.spec.max_seq_len,
            batch_size_per_rank=int(config["batch_size_per_rank"]),
            rank=rank_id,
            world_size=world_size,
        )
        batches_by_role[role] = batches
        audits_by_role[role] = audit
    source_dense, source_embedding, source_checkpoint = load_inference_checkpoint(
        checkpoint_root,
        int(first["source_version"]),
        inputs.spec,
        rank=rank_id,
        world_size=world_size,
        device=device,
    )
    states_by_role = {}
    for role in active_roles:
        initial = materialize_exact_role(
            source_dense,
            source_embedding,
            batches_by_role[role],
            int(first["history_end"]) - 1,
            device,
            phase=f"initial_theta1_exact_{role}",
        )
        states_by_role[role] = states_from_exact(
            initial,
            batches_by_role[role],
            exact_version=int(first["source_version"]),
        )
    state_binding = recursive_state_binding(
        states_by_role["qualification"],
        version=int(first["source_version"]),
        rank_id=rank_id,
        world_size=world_size,
    )
    lineage_sha = canonical_sha256(
        {
            "version": int(first["source_version"]),
            "records": role_records["qualification"],
            "state": "exact_fp16_storage",
            "cache_state_sha256": state_binding["sha256"],
        }
    )
    edge_outputs = []
    for edge_ordinal, edge in enumerate(edges):
        source_version = int(edge["source_version"])
        target_version = int(edge["target_version"])
        history_end = int(edge["history_end"])
        evaluation_end = int(edge["evaluation_end"])
        prefix_width = history_end - 1
        next_prefix_width = evaluation_end - 1
        edge_name = f"theta{source_version}_to_theta{target_version}"
        update = XPUpdateWindow(
            source_version=source_version,
            target_version=target_version,
            history_end=history_end,
            update_end=evaluation_end,
        )
        if edge_ordinal:
            batches_by_role = {}
            audits_by_role = {}
            for role in active_roles:
                batches, audit = build_bound_batches(
                    corpus,
                    role_records[role],
                    update,
                    max_seq_len=inputs.spec.max_seq_len,
                    batch_size_per_rank=int(config["batch_size_per_rank"]),
                    rank=rank_id,
                    world_size=world_size,
                )
                validate_state_batches(
                    states_by_role[role], batches, prefix_width
                )
                batches_by_role[role] = batches
                audits_by_role[role] = audit
        if method_requires_program(args.method):
            exact_source_by_role = {
                role: (
                    [state.cache for state in states_by_role[role]]
                    if edge_ordinal == 0
                    else materialize_exact_role(
                        source_dense,
                        source_embedding,
                        batches_by_role[role],
                        prefix_width,
                        device,
                        phase=f"{edge_name}_exact_source_{role}",
                    )
                )
                for role in active_roles
            }
        else:
            exact_source_by_role = {}
        source_dense.to("cpu")
        source_embedding.to("cpu")
        del source_embedding
        gc.collect()
        torch.cuda.empty_cache()
        target_dense, target_embedding, target_checkpoint = (
            load_inference_checkpoint(
                checkpoint_root,
                target_version,
                inputs.spec,
                rank=rank_id,
                world_size=world_size,
                device=device,
            )
        )
        if method_requires_fit(args.method):
            exact_target_fit = materialize_exact_role(
                target_dense,
                target_embedding,
                batches_by_role["fit"],
                prefix_width,
                device,
                phase=f"{edge_name}_exact_target_fit",
            )
            exact_target_probe = materialize_exact_role(
                target_dense,
                target_embedding,
                batches_by_role["stability_probe"],
                prefix_width,
                device,
                phase=f"{edge_name}_exact_target_stability_probe",
            )
        else:
            exact_target_fit = []
            exact_target_probe = []
        if method_requires_program(args.method):
            program, float_program, program_descriptor, fit_metrics = (
                prepare_program(
                    method=args.method,
                    edge_name=edge_name,
                    edge_ordinal=edge_ordinal,
                    source_version=source_version,
                    target_version=target_version,
                    source_dense=source_dense,
                    target_dense=target_dense,
                    fit_batches=batches_by_role.get("fit", ()),
                    fit_states=states_by_role.get("fit", ()),
                    exact_source_fit=exact_source_by_role.get("fit", ()),
                    exact_target_fit=exact_target_fit,
                    config=config,
                    repository_root=repository_root,
                    output_root=output_root,
                    rank_id=rank_id,
                    world_size=world_size,
                    device=device,
                )
            )
        else:
            program = None
            float_program = None
            program_descriptor = None
            fit_metrics = None
        if method_requires_fit(args.method):
            certificate = probe_certificate(
                method=args.method,
                program=program,
                float_program=float_program,
                exact_source_probe=exact_source_by_role["stability_probe"],
                probe_states=states_by_role["stability_probe"],
                exact_target_probe=exact_target_probe,
                probe_batches=batches_by_role["stability_probe"],
                config=config,
                device=device,
            )
        else:
            certificate = None
        fallback_all_exact = bool(
            args.method in {"ract_kv_exact10", "ract_kv_exact20"}
            and certificate is not None
            and certificate["hard_failure"]
        )
        renewal_by_role = {}
        exact_ids_by_role = {}
        for role in active_roles:
            exact_ids, renewal = role_renewal(
                role_records[role],
                prefix_width=prefix_width,
                role=role,
                method=args.method,
                edge_ordinal=edge_ordinal,
                fallback_all_exact=fallback_all_exact,
                salt=str(config["renewal"]["selection_salt"]),
            )
            exact_ids_by_role[role] = exact_ids
            renewal_by_role[role] = renewal
        if method_requires_fit(args.method):
            states_by_role["fit"], _ = advance_role(
                dense=target_dense,
                embedding=target_embedding,
                batches=batches_by_role["fit"],
                states=states_by_role["fit"],
                exact_targets=exact_target_fit,
                program=program,
                exact_record_ids=exact_ids_by_role["fit"],
                source_version=source_version,
                target_version=target_version,
                prefix_width=prefix_width,
                next_prefix_width=next_prefix_width,
                method=args.method,
                fallback_all_exact=fallback_all_exact,
                device=device,
            )
            states_by_role["stability_probe"], _ = advance_role(
                dense=target_dense,
                embedding=target_embedding,
                batches=batches_by_role["stability_probe"],
                states=states_by_role["stability_probe"],
                exact_targets=exact_target_probe,
                program=program,
                exact_record_ids=exact_ids_by_role["stability_probe"],
                source_version=source_version,
                target_version=target_version,
                prefix_width=prefix_width,
                next_prefix_width=next_prefix_width,
                method=args.method,
                fallback_all_exact=fallback_all_exact,
                device=device,
            )
        qualification_states, local_action_rows, quality = (
            evaluate_qualification(
                dense=target_dense,
                embedding=target_embedding,
                batches=batches_by_role["qualification"],
                states=states_by_role["qualification"],
                exact_source_caches=exact_source_by_role.get("qualification"),
                program=program,
                exact_record_ids=exact_ids_by_role["qualification"],
                source_version=source_version,
                target_version=target_version,
                prefix_width=prefix_width,
                next_prefix_width=next_prefix_width,
                method=args.method,
                fallback_all_exact=fallback_all_exact,
                negative_count=int(config["quality"]["negative_candidates"]),
                candidate_seed=int(config["quality"]["candidate_seed"]),
                rank_id=rank_id,
                world_size=world_size,
                device=device,
            )
        )
        states_by_role["qualification"] = qualification_states
        output_state_binding = recursive_state_binding(
            qualification_states,
            version=target_version,
            rank_id=rank_id,
            world_size=world_size,
        )
        action_rows = merge_action_rows(
            local_action_rows, world_size, rank_id
        )
        role_audits = {
            role: gather_objects(audits_by_role[role], world_size)
            for role in active_roles
        }
        action_path = output_root / "action_plans" / f"{edge_name}.json"
        edge_path = output_root / "edges" / f"{edge_name}.json"
        if rank_id == 0:
            plan = action_plan_document(
                method=args.method,
                source_version=source_version,
                target_version=target_version,
                prefix_tokens=prefix_width,
                program_sha256=(
                    None
                    if program_descriptor is None
                    else str(program_descriptor["sha256"])
                ),
                renewal=renewal_by_role["qualification"],
                fallback_all_exact=fallback_all_exact,
                rows=action_rows,
                input_lineage_sha256=lineage_sha,
                output_cache_state_sha256=str(
                    output_state_binding["sha256"]
                ),
            )
            atomic_json(action_path, plan)
            edge_result = {
                "protocol": RECURSIVE_D1_PROTOCOL,
                "scientific_result": False,
                "formal_result": False,
                "status": "complete",
                "method": args.method,
                "edge": edge_name,
                "edge_ordinal": edge_ordinal,
                "source_version": source_version,
                "target_version": target_version,
                "single_current_serving_model": True,
                "history_end": history_end,
                "evaluation_end": evaluation_end,
                "retained_prefix_tokens": prefix_width,
                "append_tokens": next_prefix_width - prefix_width,
                "bindings": {
                    "source_checkpoint": source_checkpoint,
                    "target_checkpoint": target_checkpoint,
                    "program": program_descriptor,
                    "action_plan": {
                        "path": str(action_path),
                        "sha256": file_sha256(action_path),
                        "records_sha256": plan["records_sha256"],
                        "output_lineage_sha256": plan[
                            "output_lineage_sha256"
                        ],
                    },
                },
                "roles": {
                    role: {
                        "audit_per_rank": role_audits[role],
                        "renewal": renewal_by_role[role],
                    }
                    for role in active_roles
                },
                "fit": fit_metrics,
                "stability_certificate": certificate,
                "fallback_all_exact": fallback_all_exact,
                "quality": quality,
                "logical_work": {
                    "primary_unit": "valid_retained_prefix_tokens",
                    "qualification": renewal_by_role["qualification"],
                    "physical_gpu_time_used_for_selection": False,
                },
                "recursive_handoff": {
                    "input_lineage_sha256": lineage_sha,
                    "output_lineage_sha256": plan[
                        "output_lineage_sha256"
                    ],
                    "input_cache_state": state_binding,
                    "output_cache_state": output_state_binding,
                    "hidden_exact_reset": False,
                    "output_becomes_next_edge_input": edge_ordinal < len(edges) - 1,
                },
                "full_kv_payloads_persisted": 0,
            }
            atomic_json(edge_path, edge_result)
            lineage_sha = str(plan["output_lineage_sha256"])
            edge_descriptor = {
                "edge": edge_name,
                "path": str(edge_path),
                "sha256": file_sha256(edge_path),
                "action_plan_path": str(action_path),
                "action_plan_sha256": file_sha256(action_path),
                "output_lineage_sha256": lineage_sha,
            }
        else:
            edge_descriptor = None
        edge_descriptor = broadcast_object(edge_descriptor, rank_id)
        lineage_sha = str(edge_descriptor["output_lineage_sha256"])
        state_binding = output_state_binding
        edge_outputs.append(edge_descriptor)
        del exact_source_by_role, exact_target_fit, exact_target_probe
        del program, float_program
        gc.collect()
        torch.cuda.empty_cache()
        source_dense = target_dense
        source_embedding = target_embedding
        source_embedding.eval()
        source_checkpoint = target_checkpoint
        dist.barrier()
    if rank_id == 0:
        summary = {
            "protocol": RECURSIVE_D1_PROTOCOL,
            "scientific_result": False,
            "formal_result": False,
            "status": "complete",
            "method": args.method,
            "world_size": world_size,
            "single_current_serving_model": True,
            "true_recursive_handoff": True,
            "hidden_exact_reset": False,
            "edges": edge_outputs,
            "round_config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "role_bindings": {
                name: {
                    "records": len(records),
                    "record_ids_sha256": canonical_sha256(
                        {"record_ids": records}
                    ),
                }
                for name, records in role_records.items()
            },
            "qualification_record_limit": args.qualification_record_limit,
            "edge_limit": args.edge_limit,
            "admissible_full_round": (
                args.qualification_record_limit is None
                and args.edge_limit is None
            ),
            "full_kv_payloads_persisted": 0,
            "args": vars(args),
        }
        atomic_json(output_root / "method_summary.json", summary)
        print(
            json.dumps(
                {
                    "method": args.method,
                    "output": str(output_root / "method_summary.json"),
                    "status": "complete",
                }
            )
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
