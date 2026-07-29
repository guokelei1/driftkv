from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from hstu_kvcache.migration.cohort_jagged import JaggedMigratedKVBatch
from hstu_kvcache.migration.design2_distributed import (
    D2CollectiveGuard,
    broadcast_d2_metadata,
    capture_d2_preflight_failures,
    close_d2_distributed_runtime,
    gather_d2_rank_metadata,
    init_d2_distributed_runtime,
    vote_d2_preflight,
)
from hstu_kvcache.migration.design2_embedding import (
    build_modulo_sharded_hstu_from_cpu,
    sharded_append_padded_cache,
    sharded_exact_jagged_hidden_and_kv,
)
from hstu_kvcache.migration.design2_owner import (
    D2CompiledRetainedPhaseCounters,
    execute_compiled_retained_owner_compute,
)
from hstu_kvcache.migration.design2_plan import (
    D2ActionPlan,
    build_d2_record_owner_map,
    canonical_sha256,
    d2_record_owner_map_sha256,
    file_sha256,
)
from hstu_kvcache.migration.design2_transaction import (
    D2RankCapacity,
    D2RankFragmentMetadata,
    validate_d2_private_fragments,
)
from hstu_kvcache.migration.organic import slice_jagged_token_ranges
from hstu_kvcache.migration.recompute import (
    RawHistoryBatch,
    exact_hidden_and_kv_from_item_embeddings,
)
from hstu_kvcache.migration.stage5_closure import jagged_kv_sha256
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    execute_direct_oldkv_reference,
    load_direct_oldkv_program,
)
from hstu_kvcache.migration.stage46_chain import pack_padded_cache
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_PLAN = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_STAGE_A_SUMMARY = "configs/cohortkv_d2/stage_a_summary.json"
DEFAULT_SAMPLE_INPUTS = "configs/cohortkv_d2/stage_b_sample_inputs.json"
DEFAULT_TRAINING = (
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
DEFAULT_CHECKPOINT_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
)
PROTOCOL = "cohortkv_d2_stage_b_distributed_primitives_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--stage-a-summary", default=DEFAULT_STAGE_A_SUMMARY)
    parser.add_argument("--sample-inputs", default=DEFAULT_SAMPLE_INPUTS)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output")
    parser.add_argument(
        "--case",
        choices=("normal", "hard_failure"),
        default="normal",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _checkpoint_path(
    checkpoint_dir: Path,
    version: str,
) -> Path:
    return checkpoint_dir / (
        f"theta_{int(version.removeprefix('theta'))}.pt"
    )


def _load_cpu_model(
    cfg: HSTUConfig,
    checkpoint: Path,
) -> HSTU:
    model = HSTU(cfg)
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    return model


def _records_by_id(
    sample: dict[str, object],
) -> dict[int, dict[str, object]]:
    return {
        int(value["action"]["record_id"]): value
        for value in sample["records"]
    }


def _history_batch(
    record_ids: tuple[int, ...],
    records: dict[int, dict[str, object]],
    history_key: str,
    starts: tuple[int, ...],
    stops: tuple[int, ...],
    version: str,
    device: torch.device,
) -> RawHistoryBatch:
    if (
        len(record_ids) != len(starts)
        or len(record_ids) != len(stops)
    ):
        raise ValueError("D2 Stage B history batch ranges differ")
    lengths = tuple(
        stop - start
        for start, stop in zip(starts, stops, strict=True)
    )
    if any(value < 0 for value in lengths):
        raise ValueError("D2 Stage B history batch range is invalid")
    width = max(lengths, default=1)
    item_ids = torch.zeros(
        (len(record_ids), width),
        dtype=torch.long,
        device=device,
    )
    behaviors = torch.zeros_like(item_ids)
    time_deltas = torch.zeros(
        (len(record_ids), width),
        dtype=torch.float32,
        device=device,
    )
    for row, (record_id, start, stop) in enumerate(
        zip(record_ids, starts, stops, strict=True)
    ):
        history = records[record_id][history_key]
        if history is None or not 0 <= start <= stop <= len(
            history["item_ids"]
        ):
            raise ValueError("D2 Stage B history range exceeds sample")
        length = stop - start
        if length:
            item_ids[row, :length] = torch.tensor(
                history["item_ids"][start:stop],
                dtype=torch.long,
                device=device,
            )
            behaviors[row, :length] = torch.tensor(
                history["behaviors"][start:stop],
                dtype=torch.long,
                device=device,
            )
            time_deltas[row, :length] = torch.tensor(
                history["time_deltas"][start:stop],
                dtype=torch.float32,
                device=device,
            )
    return RawHistoryBatch(
        record_ids=record_ids,
        migration_anchor_version=version,
        item_ids=item_ids,
        behaviors=behaviors,
        time_deltas=time_deltas,
        lengths=torch.tensor(
            lengths,
            dtype=torch.long,
            device=device,
        ),
    )


def _reference_vectors(
    weight: torch.Tensor,
    batch: RawHistoryBatch,
    device: torch.device,
) -> torch.Tensor:
    vectors = torch.zeros(
        *batch.item_ids.shape,
        weight.shape[1],
        dtype=weight.dtype,
        device=device,
    )
    positions = (
        torch.arange(batch.item_ids.shape[1], device=device).unsqueeze(0)
        < batch.lengths.unsqueeze(1)
    )
    if positions.any():
        vectors[positions] = weight.index_select(
            0,
            batch.item_ids[positions].detach().cpu(),
        ).to(device)
    return vectors


def _expected_lookup_evidence(
    batch: RawHistoryBatch,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    positions = (
        torch.arange(
            batch.item_ids.shape[1],
            device=batch.device,
        ).unsqueeze(0)
        < batch.lengths.unsqueeze(1)
    )
    requested = batch.item_ids[positions].detach().cpu().tolist()
    local = [value for value in requested if value % world_size == rank]
    remote = [
        value for value in requested if value % world_size != rank
    ]
    peer_ids = [
        [
            value
            for value in remote
            if value % world_size == destination
        ]
        for destination in range(world_size)
    ]
    return {
        "requested_tokens": len(requested),
        "unique_tokens": len(set(requested)),
        "local_requested_tokens": len(local),
        "local_unique_tokens": len(set(local)),
        "remote_requested_tokens": len(remote),
        "remote_unique_tokens": len(set(remote)),
        "requested_ids_sha256": canonical_sha256(
            {"item_ids": requested}
        ),
        "requested_unique_ids_sha256": canonical_sha256(
            {"item_ids": sorted(set(requested))}
        ),
        "local_requested_ids_sha256": canonical_sha256(
            {"item_ids": local}
        ),
        "local_unique_ids_sha256": canonical_sha256(
            {"item_ids": sorted(set(local))}
        ),
        "remote_requested_ids_sha256": canonical_sha256(
            {"item_ids": remote}
        ),
        "remote_unique_ids_sha256": canonical_sha256(
            {"item_ids": sorted(set(remote))}
        ),
        "remote_send_counts": tuple(len(value) for value in peer_ids),
        "remote_send_ids_sha256": tuple(
            canonical_sha256({"item_ids": value})
            for value in peer_ids
        ),
    }


def _lookup_input_evidence(
    batch: RawHistoryBatch,
    metrics,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    expected = _expected_lookup_evidence(
        batch,
        rank,
        world_size,
    )
    observed = metrics if isinstance(metrics, dict) else metrics.to_dict()
    passed = all(
        observed[key] == value
        for key, value in expected.items()
    )
    return {
        "passed": passed,
        "expected": expected,
        "observed": {
            key: observed[key]
            for key in expected
        },
    }


def _cache_bitwise(
    left: HSTUKVCache,
    right: HSTUKVCache,
) -> bool:
    return (
        left.seq_len == right.seq_len
        and torch.equal(left.k, right.k)
        and torch.equal(left.v, right.v)
    )


def _relative_l2(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    numerator = (candidate.float() - reference.float()).square().sum()
    denominator = reference.float().square().sum()
    return float(
        torch.sqrt(
            numerator
            / torch.clamp(
                denominator,
                min=torch.finfo(torch.float32).eps,
            )
        ).item()
    )


def _phase_order(
    world_size: int,
    stage_a_parity: bool,
) -> tuple[str, ...]:
    phases = ["preflight_vote"]
    lookup_phases = [
        "source_full",
        "natural_exact",
        "scheduled_exact",
        "one_owner_exact",
        "one_owner_append",
    ]
    if stage_a_parity:
        lookup_phases.append("stage_a_parity")
    lookup_phases.extend(
        (
            "synthetic_routing",
            "padding_only",
            "compiled_delta_append",
            "compiled_latest_append",
        )
    )
    for index, phase in enumerate(lookup_phases):
        if index == 1:
            phases.append("compiled_retained_local")
        phases.append(f"{phase}.invoke")
        if world_size > 1:
            phases.extend(
                (
                    f"{phase}.embedding_counts",
                    f"{phase}.embedding_ids",
                    f"{phase}.embedding_vectors",
                )
            )
    phases.extend(
        (
            "placement_ring_steal",
            "placement_ring_return",
            "cooperative_fault_vote",
        )
    )
    return tuple(phases)


@contextmanager
def _guard_context(
    guard: D2CollectiveGuard,
    phase: str,
):
    guard.enter(phase)
    yield


def _lookup_guard(
    guard: D2CollectiveGuard,
    logical_phase: str,
):
    return lambda operation: _guard_context(
        guard,
        f"{logical_phase}.{operation}",
    )


def _invoke_lookup(
    sharded,
    batch: RawHistoryBatch,
    guard: D2CollectiveGuard,
    phase: str,
):
    guard.enter(f"{phase}.invoke")
    return sharded.item_embedding.lookup(
        batch.item_ids,
        batch.lengths,
        collective_phase_guard=_lookup_guard(guard, phase),
    )


def _invoke_exact(
    sharded,
    batch: RawHistoryBatch,
    target_version: str,
    guard: D2CollectiveGuard,
    phase: str,
):
    guard.enter(f"{phase}.invoke")
    return sharded_exact_jagged_hidden_and_kv(
        sharded,
        batch,
        target_version,
        collective_phase_guard=_lookup_guard(guard, phase),
    )


def _invoke_append(
    sharded,
    cache: HSTUKVCache,
    batch: RawHistoryBatch,
    guard: D2CollectiveGuard,
    phase: str,
):
    guard.enter(f"{phase}.invoke")
    return sharded_append_padded_cache(
        sharded,
        cache,
        batch.item_ids,
        batch.behaviors,
        batch.time_deltas,
        batch.lengths,
        collective_phase_guard=_lookup_guard(guard, phase),
    )


def _synthetic_batch(
    rank: int,
    world_size: int,
    num_embeddings: int,
    device: torch.device,
    padding_only: bool = False,
) -> RawHistoryBatch:
    if padding_only:
        item_ids = torch.full(
            (1, 4),
            num_embeddings + 17,
            dtype=torch.long,
            device=device,
        )
        lengths = torch.zeros(1, dtype=torch.long, device=device)
    else:
        local_id = rank + world_size
        first_remote = (
            (rank + 1) % world_size + world_size
            if world_size > 1
            else local_id
        )
        second_remote = (
            (rank + 2) % world_size + 2 * world_size
            if world_size > 2
            else first_remote
        )
        item_ids = torch.tensor(
            [
                [
                    0,
                    local_id,
                    first_remote,
                    first_remote,
                    second_remote,
                    num_embeddings + 31,
                ]
            ],
            dtype=torch.long,
            device=device,
        )
        lengths = torch.tensor([5], dtype=torch.long, device=device)
    return RawHistoryBatch(
        record_ids=(90_000 + rank,),
        migration_anchor_version="synthetic",
        item_ids=item_ids,
        behaviors=torch.zeros_like(item_ids),
        time_deltas=torch.zeros(
            item_ids.shape,
            dtype=torch.float32,
            device=device,
        ),
        lengths=lengths,
    )


def _program_descriptor(
    stage_a_summary: dict[str, object],
) -> dict[str, object]:
    capacity_path = _path(
        stage_a_summary["artifacts"]["capacity"]["path"]
    )
    capacity = json.loads(capacity_path.read_text())
    return capacity["program"]


def _checkpoint_descriptor(
    action_plan: D2ActionPlan,
    version: str,
) -> dict[str, object]:
    upstream = json.loads(
        _path(action_plan.provenance.artifact).read_text()
    )
    return next(
        value
        for value in upstream["input_provenance"]["checkpoints"]
        if value["version"] == version
    )


def _device_process_memory(
    device: torch.device,
) -> dict[str, object]:
    result = {
        "process_used_bytes": None,
        "context_and_nonallocator_bytes": None,
        "source": "unavailable",
    }
    try:
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
        process_id = os.getpid()
        used_mib = next(
            int(row.split(",")[2].strip())
            for row in rows
            if int(row.split(",")[0].strip()) == process_id
        )
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
        return result
    process_used = used_mib * 1024 * 1024
    reserved = torch.cuda.memory_reserved(device)
    result.update(
        {
            "process_used_bytes": process_used,
            "context_and_nonallocator_bytes": max(
                0,
                process_used - reserved,
            ),
            "source": "nvidia_smi_process_used_minus_torch_reserved",
        }
    )
    return result


def _capacity_report(
    sharded,
    program,
    action_plan: D2ActionPlan,
    owner_map: dict[int, int],
    rank: int,
    device: torch.device,
) -> dict[str, object]:
    torch.cuda.synchronize(device)
    dense_bytes = sum(
        value.numel() * value.element_size()
        for value in (
            *sharded.dense_model.parameters(),
            *sharded.dense_model.buffers(),
        )
    )
    shard_bytes = (
        sharded.item_embedding.local_weight.numel()
        * sharded.item_embedding.local_weight.element_size()
    )
    program_bytes = program.nbytes
    kv_bytes_per_token = (
        2
        * sharded.dense_model.cfg.num_layers
        * sharded.dense_model.cfg.hidden_size
        * 2
    )
    owned = tuple(
        value
        for value in action_plan.records
        if owner_map[value.record_id] == rank
    )
    old_kv_bytes = (
        sum(value.old_tokens for value in owned)
        * kv_bytes_per_token
    )
    new_kv_bytes = (
        sum(value.final_tokens for value in owned)
        * kv_bytes_per_token
    )
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    process_memory = _device_process_memory(device)
    measured_context = process_memory["context_and_nonallocator_bytes"]
    context_budget = max(
        2 * 1024**3,
        0 if measured_context is None else measured_context,
    )
    transient_bytes = 512 * 1024**2
    required = (
        dense_bytes
        + shard_bytes
        + program_bytes
        + old_kv_bytes
        + new_kv_bytes
        + transient_bytes
        + context_budget
    )
    full_embedding_present = any(
        name.startswith("item_emb.")
        for name, _ in sharded.dense_model.named_parameters()
    )
    return {
        "rank": rank,
        "records": len(owned),
        "dense_model_tensor_bytes": dense_bytes,
        "embedding_shard_tensor_bytes": shard_bytes,
        "program_tensor_bytes": program_bytes,
        "projected_old_kv_bytes": old_kv_bytes,
        "projected_new_kv_bytes": new_kv_bytes,
        "projected_strict_cow_kv_bytes": old_kv_bytes + new_kv_bytes,
        "transient_bytes": transient_bytes,
        "context_budget_bytes": context_budget,
        "torch_allocated_bytes": allocated,
        "torch_reserved_bytes": reserved,
        "torch_peak_allocated_bytes": peak_allocated,
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
        "projected_required_bytes": required,
        "projected_admitted": required <= total_bytes,
        "full_embedding_parameter_present": full_embedding_present,
        "process_memory": process_memory,
    }


def _ring_exchange(
    k: torch.Tensor,
    v: torch.Tensor,
    rank: int,
    world_size: int,
    direction: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if direction not in {-1, 1}:
        raise ValueError("D2 ring direction must be -1 or 1")
    if world_size == 1:
        return k.clone(), v.clone(), 0.0
    destination = (rank + direction) % world_size
    source = (rank - direction) % world_size
    received_k = torch.empty_like(k)
    received_v = torch.empty_like(v)
    torch.cuda.synchronize(k.device)
    started = time.perf_counter()
    requests = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, k, destination),
            dist.P2POp(dist.irecv, received_k, source),
            dist.P2POp(dist.isend, v, destination),
            dist.P2POp(dist.irecv, received_v, source),
        ]
    )
    for request in requests:
        request.wait()
    torch.cuda.synchronize(k.device)
    return received_k, received_v, time.perf_counter() - started


def _placement_ring(
    runtime,
    program,
    operator: DirectOldKVFusedOperator,
    guard: D2CollectiveGuard,
) -> dict[str, object]:
    rank = runtime.rank
    world_size = runtime.world_size
    tokens = 16
    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(7813 + rank)
    shape = (program.num_layers, tokens, program.kv_width)
    source = JaggedMigratedKVBatch(
        record_ids=(100_000 + rank,),
        migration_anchor_version=program.source_version,
        served_kv_target=program.source_version,
        k=torch.randn(
            shape,
            dtype=torch.float16,
            device=runtime.device,
            generator=generator,
        ),
        v=torch.randn(
            shape,
            dtype=torch.float16,
            device=runtime.device,
            generator=generator,
        ),
        lengths=torch.tensor(
            [tokens],
            dtype=torch.long,
            device=runtime.device,
        ),
        offsets=torch.tensor(
            [0, tokens],
            dtype=torch.long,
            device=runtime.device,
        ),
    )
    local = execute_compiled_retained_owner_compute(
        program,
        source,
        {100_000 + rank: rank},
        rank,
        operator=operator,
    )
    guard.enter("placement_ring_steal")
    received_k, received_v, steal_seconds = _ring_exchange(
        source.k,
        source.v,
        rank,
        world_size,
    )
    previous = (rank - 1) % world_size
    stolen = JaggedMigratedKVBatch(
        record_ids=(100_000 + previous,),
        migration_anchor_version=program.source_version,
        served_kv_target=program.source_version,
        k=received_k,
        v=received_v,
        lengths=source.lengths.clone(),
        offsets=source.offsets.clone(),
    )
    stolen_output = JaggedMigratedKVBatch(
        record_ids=stolen.record_ids,
        migration_anchor_version=program.source_version,
        served_kv_target=program.target_version,
        k=torch.empty_like(stolen.k),
        v=torch.empty_like(stolen.v),
        lengths=stolen.lengths.clone(),
        offsets=stolen.offsets.clone(),
    )
    prepared = operator.prepare_program(program, runtime.device)
    operator.execute_into(prepared, stolen, stolen_output)
    guard.enter("placement_ring_return")
    returned_k, returned_v, return_seconds = _ring_exchange(
        stolen_output.k,
        stolen_output.v,
        rank,
        world_size,
        direction=-1,
    )
    payload_bytes = source.k.numel() * source.k.element_size() * 2
    return {
        "tokens": tokens,
        "extent_payload_bytes": payload_bytes,
        "p2p_steal_seconds": steal_seconds,
        "target_return_seconds": return_seconds,
        "old_kv_send_bytes": 0 if world_size == 1 else payload_bytes,
        "old_kv_receive_bytes": 0 if world_size == 1 else payload_bytes,
        "target_kv_send_bytes": 0 if world_size == 1 else payload_bytes,
        "target_kv_receive_bytes": (
            0 if world_size == 1 else payload_bytes
        ),
        "returned_output_bitwise": (
            local.output is not None
            and torch.equal(returned_k, local.output.k)
            and torch.equal(returned_v, local.output.v)
        ),
        "cross_island_edge": (
            world_size == 4
            and (
                (rank in {0, 1} and (rank + 1) % world_size in {2, 3})
                or (
                    rank in {2, 3}
                    and (rank + 1) % world_size in {0, 1}
                )
            )
        ),
        "scientific_result": False,
    }


def _exact_reference(
    sharded,
    batch: RawHistoryBatch,
    vectors: torch.Tensor,
    target_version: str,
) -> tuple[JaggedMigratedKVBatch | None, torch.Tensor]:
    if batch.batch_size == 0:
        return (
            None,
            torch.empty(
                (0, sharded.dense_model.cfg.hidden_size),
                dtype=torch.float32,
                device=batch.device,
            ),
        )
    hidden, cache = exact_hidden_and_kv_from_item_embeddings(
        sharded.dense_model,
        vectors,
        batch.behaviors,
        batch.time_deltas,
        lengths=batch.lengths,
    )
    return (
        pack_padded_cache(
            cache,
            batch.lengths,
            batch.record_ids,
            target_version,
            target_version,
            dtype=torch.float16,
        ),
        sharded.dense_model.last_hidden(hidden, batch.lengths),
    )


def _exact_report(
    result,
    reference_fragment: JaggedMigratedKVBatch | None,
    reference_hidden: torch.Tensor,
) -> dict[str, object]:
    empty_equal = result.fragment is None and reference_fragment is None
    fragment_equal = (
        empty_equal
        or (
            result.fragment is not None
            and reference_fragment is not None
            and result.fragment.record_ids
            == reference_fragment.record_ids
            and torch.equal(result.fragment.k, reference_fragment.k)
            and torch.equal(result.fragment.v, reference_fragment.v)
            and torch.equal(
                result.fragment.lengths,
                reference_fragment.lengths,
            )
            and torch.equal(
                result.fragment.offsets,
                reference_fragment.offsets,
            )
        )
    )
    return {
        "fragment_bitwise": fragment_equal,
        "hidden_bitwise": torch.equal(
            result.last_hidden,
            reference_hidden,
        ),
        "empty_fragment": result.fragment is None,
        "record_ids": (
            []
            if result.fragment is None
            else list(result.fragment.record_ids)
        ),
        "lookup": result.lookup_metrics.to_dict(),
    }


def _stage_a_plan_ledger(
    action_plan: D2ActionPlan,
) -> dict[str, int | float]:
    compiled_delta = sum(
        value.delta_tokens
        for value in action_plan.records
        if value.requested_action == "compiled"
    )
    compiled_latest = sum(
        value.latest_tokens
        for value in action_plan.records
        if value.requested_action == "compiled"
    )
    all_exact_retained = sum(
        value.retained_tokens
        for value in action_plan.records
        if value.previous_cache_present
    )
    mixed_retained = sum(
        value.retained_tokens
        for value in action_plan.records
        if value.requested_reason == "scheduled_exact"
    )
    natural = sum(
        value.target_prefix_tokens
        for value in action_plan.records
        if value.requested_reason == "natural_exact"
    )
    delta = sum(
        value.delta_tokens
        for value in action_plan.records
        if value.requested_reason != "natural_exact"
    )
    latest = sum(value.latest_tokens for value in action_plan.records)
    return {
        "compiled_delta_tokens": compiled_delta,
        "compiled_latest_tokens": compiled_latest,
        "compiled_append_tokens": compiled_delta + compiled_latest,
        "all_exact_retained_tokens": all_exact_retained,
        "mixed_scheduled_retained_tokens": mixed_retained,
        "natural_exact_prefix_tokens": natural,
        "delta_tokens": delta,
        "latest_tokens": latest,
        "all_exact_full_wave_tokens": (
            all_exact_retained + natural + delta + latest
        ),
        "mixed_full_wave_tokens": (
            mixed_retained + natural + delta + latest
        ),
    }


def _stage_a_expected_ledger(
    stage_a: dict[str, object],
) -> dict[str, int]:
    request_descriptor = stage_a["artifacts"]["requests"]
    request_path = _path(request_descriptor["path"])
    if file_sha256(request_path) != request_descriptor["sha256"]:
        raise RuntimeError("Stage A request ledger hash differs")
    boundaries = json.loads(request_path.read_text())[
        "phase_ledger_boundaries"
    ]
    retained = boundaries["retained_prefix"]
    append = boundaries["method_independent_append"]
    integrated = boundaries["integrated_post_append"]
    natural = (
        integrated["mixed_lookup_tokens"]
        - retained["mixed_lookup_tokens"]
        - append["lookup_tokens"]
    )
    return {
        "all_exact_retained_tokens": retained[
            "all_exact_lookup_tokens"
        ],
        "mixed_scheduled_retained_tokens": retained[
            "mixed_lookup_tokens"
        ],
        "natural_exact_prefix_tokens": natural,
        "delta_tokens": append["delta_lookup_tokens"],
        "latest_tokens": append["latest_lookup_tokens"],
        "all_exact_full_wave_tokens": integrated[
            "all_exact_lookup_tokens"
        ],
        "mixed_full_wave_tokens": integrated["mixed_lookup_tokens"],
    }


def _normal_rank(
    args: argparse.Namespace,
    runtime,
) -> dict[str, object]:
    action_path = _path(args.action_plan)
    stage_a_path = _path(args.stage_a_summary)
    sample_path = _path(args.sample_inputs)
    training_path = _path(args.training_result)
    checkpoint_dir = _path(args.checkpoint_dir)
    action_plan = D2ActionPlan.load(action_path)
    stage_a = json.loads(stage_a_path.read_text())
    device_properties = torch.cuda.get_device_properties(runtime.device)
    p2p_descriptor = stage_a["artifacts"]["p2p"]
    p2p_path = _path(p2p_descriptor["path"])
    p2p = json.loads(p2p_path.read_text())
    stage_a_expected_ledger = _stage_a_expected_ledger(stage_a)
    sample = json.loads(sample_path.read_text())
    training = json.loads(training_path.read_text())
    cfg = HSTUConfig(**training["model"])
    owner_map = build_d2_record_owner_map(
        action_plan,
        runtime.world_size,
        "strict_cow_lpt",
    )
    selection = sample["selections"][str(runtime.world_size)]["ranks"][
        str(runtime.rank)
    ]
    records = _records_by_id(sample)
    source_checkpoint = _checkpoint_path(
        checkpoint_dir,
        action_plan.source_version,
    )
    target_checkpoint = _checkpoint_path(
        checkpoint_dir,
        action_plan.target_version,
    )
    source_descriptor = _checkpoint_descriptor(
        action_plan,
        action_plan.source_version,
    )
    target_descriptor = _checkpoint_descriptor(
        action_plan,
        action_plan.target_version,
    )
    program_descriptor = _program_descriptor(stage_a)
    stage_a_parity = runtime.world_size == 1
    guard = D2CollectiveGuard(
        runtime,
        _phase_order(runtime.world_size, stage_a_parity),
    )
    preflight_failures = capture_d2_preflight_failures(
        {
            "action_content": (
                lambda: action_plan.content_sha256
                == stage_a["action_plan"]["content_sha256"]
            ),
            "action_file": (
                lambda: file_sha256(action_path)
                == stage_a["action_plan"]["file_sha256"]
            ),
            "stage_a_status": (
                lambda: stage_a["status"] == "complete"
                and stage_a["stage_b_entry"] == "go"
            ),
            "stage_a_phase_ledger": (
                lambda: all(
                    _stage_a_plan_ledger(action_plan)[key] == value
                    for key, value in stage_a_expected_ledger.items()
                )
            ),
            "stage_a_topology": (
                lambda: file_sha256(p2p_path)
                == p2p_descriptor["sha256"]
                and (
                    runtime.world_size != 4
                    or p2p["devices"][runtime.rank]["uuid"]
                    == f"GPU-{device_properties.uuid}"
                )
            ),
            "sample_content": (
                lambda: sample["content_sha256"]
                == canonical_sha256(
                    {
                        key: value
                        for key, value in sample.items()
                        if key != "content_sha256"
                    }
                )
            ),
            "sample_plan": (
                lambda: sample["action_plan"]["content_sha256"]
                == action_plan.content_sha256
                and sample["action_plan"]["file_sha256"]
                == file_sha256(action_path)
            ),
            "sample_stage_a": (
                lambda: sample["stage_a_summary"]["sha256"]
                == file_sha256(stage_a_path)
                and _path(sample["stage_a_summary"]["path"])
                == stage_a_path
            ),
            "source_checkpoint": (
                lambda: file_sha256(source_checkpoint)
                == source_descriptor["sha256"]
            ),
            "target_checkpoint": (
                lambda: file_sha256(target_checkpoint)
                == target_descriptor["sha256"]
            ),
            "program": (
                lambda: file_sha256(
                    _path(program_descriptor["path"])
                )
                == program_descriptor["sha256"]
            ),
            "owner_selection": (
                lambda: all(
                    owner_map[int(record_id)] == runtime.rank
                    for record_id in selection.values()
                )
            ),
            "owner_map": (
                lambda: d2_record_owner_map_sha256(owner_map)
                == sample["selections"][str(runtime.world_size)][
                    "owner_map_sha256"
                ]
            ),
        }
    )
    preflight = vote_d2_preflight(
        runtime,
        preflight_failures,
        guard=guard,
    )
    if not preflight.passed:
        raise RuntimeError(
            f"D2 Stage B preflight failed: {preflight.failure_reasons}"
        )
    compiled_record = action_plan.records[
        int(selection["compiled"])
    ]
    if compiled_record.record_id != int(selection["compiled"]):
        compiled_record = next(
            value
            for value in action_plan.records
            if value.record_id == int(selection["compiled"])
        )
    source_batch = _history_batch(
        (compiled_record.record_id,),
        records,
        "source_history",
        (0,),
        (compiled_record.old_tokens,),
        action_plan.source_version,
        runtime.device,
    )
    source_cpu = _load_cpu_model(cfg, source_checkpoint)
    source_reference_vectors = _reference_vectors(
        source_cpu.item_emb.weight.detach(),
        source_batch,
        runtime.device,
    )
    source_sharded = build_modulo_sharded_hstu_from_cpu(
        source_cpu,
        runtime.rank,
        runtime.world_size,
        runtime.device,
    )
    del source_cpu
    gc.collect()
    source_result = _invoke_exact(
        source_sharded,
        source_batch,
        action_plan.source_version,
        guard,
        "source_full",
    )
    source_reference_fragment, source_reference_hidden = _exact_reference(
        source_sharded,
        source_batch,
        source_reference_vectors,
        action_plan.source_version,
    )
    source_report = _exact_report(
        source_result,
        source_reference_fragment,
        source_reference_hidden,
    )
    if source_result.fragment is None:
        raise RuntimeError("source exact returned no owner fragment")
    retained_slice = slice_jagged_token_ranges(
        source_result.fragment,
        (compiled_record.retained_start,),
        (compiled_record.old_tokens,),
    )
    if retained_slice.cache is None:
        raise RuntimeError("compiled retained slice is empty")
    program_cpu, loaded_program = load_direct_oldkv_program(
        _path(program_descriptor["path"]),
        expected_sha256=program_descriptor["sha256"],
        expected_source_version=action_plan.source_version,
        expected_target_version=action_plan.target_version,
        expected_num_layers=cfg.num_layers,
        expected_kv_width=cfg.hidden_size,
    )
    operator = DirectOldKVFusedOperator()
    guard.enter("compiled_retained_local")
    compiled_fragment = execute_compiled_retained_owner_compute(
        program_cpu,
        retained_slice.cache,
        owner_map,
        runtime.rank,
        operator=operator,
        phase_counters=D2CompiledRetainedPhaseCounters(),
    )
    program_gpu = program_cpu.to(runtime.device)
    reference_compiled = JaggedMigratedKVBatch(
        record_ids=retained_slice.cache.record_ids,
        migration_anchor_version=action_plan.source_version,
        served_kv_target=action_plan.target_version,
        k=torch.empty_like(retained_slice.cache.k),
        v=torch.empty_like(retained_slice.cache.v),
        lengths=retained_slice.cache.lengths.clone(),
        offsets=retained_slice.cache.offsets.clone(),
    )
    execute_direct_oldkv_reference(
        program_gpu,
        retained_slice.cache,
        reference_compiled,
    )
    if compiled_fragment.output is None:
        raise RuntimeError("owner compute returned no private fragment")
    compiled_k_l2 = _relative_l2(
        compiled_fragment.output.k,
        reference_compiled.k,
    )
    compiled_v_l2 = _relative_l2(
        compiled_fragment.output.v,
        reference_compiled.v,
    )
    del source_sharded
    del source_result
    del source_reference_fragment
    gc.collect()
    torch.cuda.empty_cache()
    natural_record = next(
        value
        for value in action_plan.records
        if value.record_id == int(selection["natural_exact"])
    )
    scheduled_record = next(
        value
        for value in action_plan.records
        if value.record_id == int(selection["scheduled_exact"])
    )
    natural_batch = _history_batch(
        (natural_record.record_id,),
        records,
        "target_history",
        (0,),
        (natural_record.target_prefix_tokens,),
        action_plan.target_version,
        runtime.device,
    )
    scheduled_batch = _history_batch(
        (scheduled_record.record_id,),
        records,
        "target_history",
        (0,),
        (scheduled_record.retained_tokens,),
        action_plan.target_version,
        runtime.device,
    )
    if runtime.rank == 0:
        one_owner_batch = natural_batch
    else:
        one_owner_batch = _history_batch(
            (),
            records,
            "target_history",
            (),
            (),
            action_plan.target_version,
            runtime.device,
        )
    delta_batch = _history_batch(
        (compiled_record.record_id,),
        records,
        "target_history",
        (compiled_record.delta_start,),
        (compiled_record.target_prefix_tokens,),
        action_plan.target_version,
        runtime.device,
    )
    latest_batch = _history_batch(
        (compiled_record.record_id,),
        records,
        "target_history",
        (compiled_record.target_prefix_tokens,),
        (compiled_record.final_tokens,),
        action_plan.target_version,
        runtime.device,
    )
    if runtime.rank == 0:
        one_owner_append_batch = latest_batch
        one_owner_append_cache = HSTUKVCache(
            k=compiled_fragment.output.k.unsqueeze(1),
            v=compiled_fragment.output.v.unsqueeze(1),
            seq_len=compiled_record.retained_tokens,
        )
    else:
        one_owner_append_batch = _history_batch(
            (),
            records,
            "target_history",
            (),
            (),
            action_plan.target_version,
            runtime.device,
        )
        empty_shape = (
            cfg.num_layers,
            0,
            0,
            cfg.hidden_size,
        )
        one_owner_append_cache = HSTUKVCache(
            k=torch.empty(
                empty_shape,
                dtype=compiled_fragment.output.k.dtype,
                device=runtime.device,
            ),
            v=torch.empty(
                empty_shape,
                dtype=compiled_fragment.output.v.dtype,
                device=runtime.device,
            ),
            seq_len=0,
        )
    synthetic_batch = _synthetic_batch(
        runtime.rank,
        runtime.world_size,
        cfg.num_items + 1,
        runtime.device,
    )
    padding_batch = _synthetic_batch(
        runtime.rank,
        runtime.world_size,
        cfg.num_items + 1,
        runtime.device,
        padding_only=True,
    )
    parity_batch = None
    if stage_a_parity:
        parity_ids = tuple(
            int(value)
            for value in sample["selections"]["stage_a_parity"][
                "record_ids"
            ]
        )
        parity_actions = tuple(
            next(
                value
                for value in action_plan.records
                if value.record_id == record_id
            )
            for record_id in parity_ids
        )
        parity_batch = _history_batch(
            parity_ids,
            records,
            "target_history",
            (0,) * len(parity_ids),
            tuple(value.final_tokens for value in parity_actions),
            action_plan.target_version,
            runtime.device,
        )
    target_batches = {
        "natural": natural_batch,
        "scheduled": scheduled_batch,
        "one_owner": one_owner_batch,
        "one_owner_append": one_owner_append_batch,
        "delta": delta_batch,
        "latest": latest_batch,
        "synthetic": synthetic_batch,
        "padding": padding_batch,
    }
    if parity_batch is not None:
        target_batches["stage_a"] = parity_batch
    target_cpu = _load_cpu_model(cfg, target_checkpoint)
    target_weight = target_cpu.item_emb.weight.detach()
    reference_vectors = {
        key: _reference_vectors(
            target_weight,
            batch,
            runtime.device,
        )
        for key, batch in target_batches.items()
    }
    target_sharded = build_modulo_sharded_hstu_from_cpu(
        target_cpu,
        runtime.rank,
        runtime.world_size,
        runtime.device,
    )
    del target_cpu
    del target_weight
    gc.collect()
    natural_result = _invoke_exact(
        target_sharded,
        natural_batch,
        action_plan.target_version,
        guard,
        "natural_exact",
    )
    natural_reference, natural_hidden = _exact_reference(
        target_sharded,
        natural_batch,
        reference_vectors["natural"],
        action_plan.target_version,
    )
    natural_report = _exact_report(
        natural_result,
        natural_reference,
        natural_hidden,
    )
    scheduled_result = _invoke_exact(
        target_sharded,
        scheduled_batch,
        action_plan.target_version,
        guard,
        "scheduled_exact",
    )
    scheduled_reference, scheduled_hidden = _exact_reference(
        target_sharded,
        scheduled_batch,
        reference_vectors["scheduled"],
        action_plan.target_version,
    )
    scheduled_report = _exact_report(
        scheduled_result,
        scheduled_reference,
        scheduled_hidden,
    )
    one_owner_result = _invoke_exact(
        target_sharded,
        one_owner_batch,
        action_plan.target_version,
        guard,
        "one_owner_exact",
    )
    one_owner_reference, one_owner_hidden = _exact_reference(
        target_sharded,
        one_owner_batch,
        reference_vectors["one_owner"],
        action_plan.target_version,
    )
    one_owner_report = _exact_report(
        one_owner_result,
        one_owner_reference,
        one_owner_hidden,
    )
    one_owner_append_result = _invoke_append(
        target_sharded,
        one_owner_append_cache,
        one_owner_append_batch,
        guard,
        "one_owner_append",
    )
    if runtime.rank == 0:
        one_owner_append_hidden, one_owner_append_cache_reference = (
            target_sharded.dense_model.forward_with_cache_from_item_embeddings(
                one_owner_append_cache,
                reference_vectors["one_owner_append"],
                one_owner_append_batch.behaviors,
                one_owner_append_batch.time_deltas,
            )
        )
        one_owner_append_bitwise = (
            _cache_bitwise(
                one_owner_append_result.updated_cache,
                one_owner_append_cache_reference,
            )
            and torch.equal(
                one_owner_append_result.last_hidden,
                one_owner_append_hidden[:, -1],
            )
            and one_owner_append_result.lengths.tolist()
            == [
                compiled_record.retained_tokens
                + compiled_record.latest_tokens
            ]
        )
    else:
        one_owner_append_bitwise = (
            one_owner_append_result.updated_cache.k.numel() == 0
            and one_owner_append_result.updated_cache.v.numel() == 0
            and one_owner_append_result.last_hidden.numel() == 0
            and one_owner_append_result.lengths.numel() == 0
        )
    one_owner_append_report = {
        "bitwise": one_owner_append_bitwise,
        "empty_rank": runtime.rank != 0,
        "lengths": one_owner_append_result.lengths.tolist(),
        "lookup": one_owner_append_result.lookup_metrics.to_dict(),
    }
    parity_report = None
    if parity_batch is not None:
        parity_result = _invoke_exact(
            target_sharded,
            parity_batch,
            action_plan.target_version,
            guard,
            "stage_a_parity",
        )
        parity_reference, parity_hidden = _exact_reference(
            target_sharded,
            parity_batch,
            reference_vectors["stage_a"],
            action_plan.target_version,
        )
        parity_report = _exact_report(
            parity_result,
            parity_reference,
            parity_hidden,
        )
    synthetic_lookup = _invoke_lookup(
        target_sharded,
        synthetic_batch,
        guard,
        "synthetic_routing",
    )
    padding_lookup = _invoke_lookup(
        target_sharded,
        padding_batch,
        guard,
        "padding_only",
    )
    compiled_cache = HSTUKVCache(
        k=compiled_fragment.output.k.unsqueeze(1),
        v=compiled_fragment.output.v.unsqueeze(1),
        seq_len=compiled_record.retained_tokens,
    )
    reference_compiled_cache = HSTUKVCache(
        k=compiled_cache.k.clone(),
        v=compiled_cache.v.clone(),
        seq_len=compiled_cache.seq_len,
    )
    delta_result = _invoke_append(
        target_sharded,
        compiled_cache,
        delta_batch,
        guard,
        "compiled_delta_append",
    )
    delta_reference_hidden, delta_reference_cache = (
        target_sharded.dense_model.forward_with_cache_from_item_embeddings(
            reference_compiled_cache,
            reference_vectors["delta"],
            delta_batch.behaviors,
            delta_batch.time_deltas,
        )
    )
    delta_bitwise = (
        _cache_bitwise(
            delta_result.updated_cache,
            delta_reference_cache,
        )
        and torch.equal(
            delta_result.last_hidden,
            delta_reference_hidden[:, -1],
        )
    )
    latest_result = _invoke_append(
        target_sharded,
        delta_result.updated_cache,
        latest_batch,
        guard,
        "compiled_latest_append",
    )
    latest_reference_hidden, latest_reference_cache = (
        target_sharded.dense_model.forward_with_cache_from_item_embeddings(
            delta_reference_cache,
            reference_vectors["latest"],
            latest_batch.behaviors,
            latest_batch.time_deltas,
        )
    )
    latest_bitwise = (
        _cache_bitwise(
            latest_result.updated_cache,
            latest_reference_cache,
        )
        and torch.equal(
            latest_result.last_hidden,
            latest_reference_hidden[:, -1],
        )
    )
    final_compiled_fragment = pack_padded_cache(
        latest_result.updated_cache,
        latest_result.lengths,
        (compiled_record.record_id,),
        action_plan.source_version,
        action_plan.target_version,
        dtype=torch.float16,
    )
    if natural_result.fragment is None or scheduled_result.fragment is None:
        raise RuntimeError("D2 Stage B selected exact fragment is empty")
    private_component_hashes = {
        "compiled_final": jagged_kv_sha256(final_compiled_fragment),
        "natural_exact": jagged_kv_sha256(natural_result.fragment),
        "scheduled_exact": jagged_kv_sha256(
            scheduled_result.fragment
        ),
    }
    synthetic_reference = reference_vectors["synthetic"]
    synthetic_valid = (
        torch.arange(
            synthetic_batch.item_ids.shape[1],
            device=runtime.device,
        ).unsqueeze(0)
        < synthetic_batch.lengths.unsqueeze(1)
    )
    synthetic_bitwise = torch.equal(
        synthetic_lookup.item_vectors[synthetic_valid],
        synthetic_reference[synthetic_valid],
    )
    padding_zero = (
        padding_lookup.metrics.requested_tokens == 0
        and torch.count_nonzero(padding_lookup.item_vectors).item() == 0
    )
    placement = _placement_ring(
        runtime,
        program_cpu,
        operator,
        guard,
    )
    cooperative_fault = vote_d2_preflight(
        runtime,
        (
            ("synthetic capacity rejection",)
            if runtime.rank == runtime.world_size - 1
            else ()
        ),
        guard=guard,
        phase="cooperative_fault_vote",
    )
    phase_trace = guard.require_complete()
    capacity = _capacity_report(
        target_sharded,
        program_gpu,
        action_plan,
        owner_map,
        runtime.rank,
        runtime.device,
    )
    sample_record_ids = tuple(
        sorted(
            (
                compiled_record.record_id,
                natural_record.record_id,
                scheduled_record.record_id,
            )
        )
    )
    sample_payload_bytes = sum(
        value.nbytes
        for value in (
            final_compiled_fragment,
            natural_result.fragment,
            scheduled_result.fragment,
        )
    )
    fragment_sha = canonical_sha256(
        {
            "rank": runtime.rank,
            "record_ids": list(sample_record_ids),
            "component_sha256": private_component_hashes,
            "payload_bytes": sample_payload_bytes,
        }
    )
    transaction_capacity = D2RankCapacity(
        required_bytes=torch.cuda.memory_reserved(runtime.device),
        capacity_bytes=capacity["device_total_bytes"],
        measured_peak_bytes=torch.cuda.max_memory_allocated(
            runtime.device
        ),
    )
    local_metadata = D2RankFragmentMetadata(
        action_plan_sha256=action_plan.content_sha256,
        target_version=action_plan.target_version,
        rank=runtime.rank,
        world_size=runtime.world_size,
        owner_record_ids=sample_record_ids,
        fragment_sha256=fragment_sha,
        payload_bytes=sample_payload_bytes,
        phase_trace=phase_trace,
        capacity=transaction_capacity,
    )
    transaction_guard = D2CollectiveGuard(
        runtime,
        (
            "transaction_ready_gather",
            "transaction_ready_broadcast",
            "transaction_abort_gather",
            "transaction_abort_broadcast",
        ),
    )
    gathered = gather_d2_rank_metadata(
        runtime,
        local_metadata,
        guard=transaction_guard,
        phase="transaction_ready_gather",
    )
    sample_owner_map = {
        record_id: owner
        for owner, rank_values in sample["selections"][
            str(runtime.world_size)
        ]["ranks"].items()
        for record_id in rank_values.values()
    }
    ready_envelope = None
    if runtime.rank == 0:
        try:
            ready_envelope = {
                "ok": True,
                "value": validate_d2_private_fragments(
                    action_plan_sha256=action_plan.content_sha256,
                    target_version=action_plan.target_version,
                    world_size=runtime.world_size,
                    record_owner_map={
                        int(record_id): int(owner)
                        for record_id, owner in sample_owner_map.items()
                    },
                    rank_metadata=gathered,
                    expected_phase_trace=phase_trace,
                ),
            }
        except Exception as error:
            ready_envelope = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    ready_envelope = broadcast_d2_metadata(
        runtime,
        ready_envelope,
        guard=transaction_guard,
        phase="transaction_ready_broadcast",
    )
    if not ready_envelope["ok"]:
        raise RuntimeError(
            f"D2 ready validation failed: {ready_envelope['error']}"
        )
    transaction_ready = ready_envelope["value"]
    failing_rank = runtime.world_size - 1
    abort_metadata = (
        local_metadata.with_synthetic_failure("pre-ready failure")
        if runtime.rank == failing_rank
        else local_metadata
    )
    gathered_abort = gather_d2_rank_metadata(
        runtime,
        abort_metadata,
        guard=transaction_guard,
        phase="transaction_abort_gather",
    )
    abort_envelope = None
    if runtime.rank == 0:
        try:
            abort_envelope = {
                "ok": True,
                "value": validate_d2_private_fragments(
                    action_plan_sha256=action_plan.content_sha256,
                    target_version=action_plan.target_version,
                    world_size=runtime.world_size,
                    record_owner_map={
                        int(record_id): int(owner)
                        for record_id, owner in sample_owner_map.items()
                    },
                    rank_metadata=gathered_abort,
                    expected_phase_trace=phase_trace,
                ),
            }
        except Exception as error:
            abort_envelope = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    abort_envelope = broadcast_d2_metadata(
        runtime,
        abort_envelope,
        guard=transaction_guard,
        phase="transaction_abort_broadcast",
    )
    if not abort_envelope["ok"]:
        raise RuntimeError(
            f"D2 abort validation failed: {abort_envelope['error']}"
        )
    transaction_abort = abort_envelope["value"]
    transaction_phase_trace = transaction_guard.require_complete()
    lookup_input_evidence = {
        "source_full": _lookup_input_evidence(
            source_batch,
            source_report["lookup"],
            runtime.rank,
            runtime.world_size,
        ),
        "natural_exact": _lookup_input_evidence(
            natural_batch,
            natural_result.lookup_metrics,
            runtime.rank,
            runtime.world_size,
        ),
        "scheduled_exact": _lookup_input_evidence(
            scheduled_batch,
            scheduled_result.lookup_metrics,
            runtime.rank,
            runtime.world_size,
        ),
        "one_owner_exact": _lookup_input_evidence(
            one_owner_batch,
            one_owner_result.lookup_metrics,
            runtime.rank,
            runtime.world_size,
        ),
        "one_owner_append": _lookup_input_evidence(
            one_owner_append_batch,
            one_owner_append_result.lookup_metrics,
            runtime.rank,
            runtime.world_size,
        ),
        "synthetic_routing": _lookup_input_evidence(
            synthetic_batch,
            synthetic_lookup.metrics,
            runtime.rank,
            runtime.world_size,
        ),
        "padding_only": _lookup_input_evidence(
            padding_batch,
            padding_lookup.metrics,
            runtime.rank,
            runtime.world_size,
        ),
        "compiled_delta_append": _lookup_input_evidence(
            delta_batch,
            delta_result.lookup_metrics,
            runtime.rank,
            runtime.world_size,
        ),
        "compiled_latest_append": _lookup_input_evidence(
            latest_batch,
            latest_result.lookup_metrics,
            runtime.rank,
            runtime.world_size,
        ),
    }
    if parity_batch is not None and parity_report is not None:
        lookup_input_evidence["stage_a_parity"] = (
            _lookup_input_evidence(
                parity_batch,
                parity_report["lookup"],
                runtime.rank,
                runtime.world_size,
            )
        )
    return {
        "rank": runtime.rank,
        "local_rank": runtime.local_rank,
        "device": str(runtime.device),
        "device_name": torch.cuda.get_device_name(runtime.device),
        "device_uuid": f"GPU-{device_properties.uuid}",
        "device_pci": {
            "domain": device_properties.pci_domain_id,
            "bus": device_properties.pci_bus_id,
            "device": device_properties.pci_device_id,
        },
        "embedding_hidden_size": target_sharded.item_embedding.hidden_size,
        "selection": {
            "compiled": compiled_record.record_id,
            "natural_exact": natural_record.record_id,
            "scheduled_exact": scheduled_record.record_id,
        },
        "preflight": {
            "passed": preflight.passed,
            "failure_reasons": list(preflight.failure_reasons),
        },
        "source_materialization": source_report,
        "compiled_retained": {
            **compiled_fragment.to_dict(),
            "reference_k_relative_l2": compiled_k_l2,
            "reference_v_relative_l2": compiled_v_l2,
        },
        "natural_exact": natural_report,
        "scheduled_exact": scheduled_report,
        "one_owner_exact": one_owner_report,
        "one_owner_append": one_owner_append_report,
        "stage_a_parity": parity_report,
        "synthetic_routing": {
            "item_vectors_bitwise": synthetic_bitwise,
            "valid_item_ids": synthetic_batch.item_ids[
                synthetic_valid
            ].tolist(),
            "lookup": synthetic_lookup.metrics.to_dict(),
        },
        "padding_only": {
            "zero_output": padding_zero,
            "lookup": padding_lookup.metrics.to_dict(),
        },
        "compiled_delta_append": {
            "bitwise": delta_bitwise,
            "tokens": compiled_record.delta_tokens,
            "lengths": delta_result.lengths.tolist(),
            "lookup": delta_result.lookup_metrics.to_dict(),
        },
        "compiled_latest_append": {
            "bitwise": latest_bitwise,
            "tokens": compiled_record.latest_tokens,
            "lengths": latest_result.lengths.tolist(),
            "lookup": latest_result.lookup_metrics.to_dict(),
        },
        "placement_ring": placement,
        "cooperative_failure": {
            "isolated_probe": True,
            "does_not_gate_normal_private_fragment": True,
            "passed": cooperative_fault.passed,
            "failure_reasons": list(
                cooperative_fault.failure_reasons
            ),
        },
        "capacity": capacity,
        "phase_trace": [
            {
                "ordinal": value.ordinal,
                "phase": value.phase,
                "token": value.token,
            }
            for value in phase_trace
        ],
        "transaction_phase_trace": [
            {
                "ordinal": value.ordinal,
                "phase": value.phase,
                "token": value.token,
            }
            for value in transaction_phase_trace
        ],
        "transaction_ready": transaction_ready.to_dict(),
        "transaction_abort": transaction_abort.to_dict(),
        "lookup_input_evidence": lookup_input_evidence,
        "private_fragment": {
            "sha256": fragment_sha,
            "component_sha256": private_component_hashes,
            "payload_bytes": sample_payload_bytes,
            "dtype": "float16",
        },
        "full_embedding_lookup_forbidden": not any(
            name.startswith("item_emb.")
            for name, _ in target_sharded.dense_model.named_parameters()
        ),
        "program": loaded_program,
    }


def _lookup_reports(
    rank_reports: tuple[dict[str, object], ...],
) -> dict[str, list[dict[str, object]]]:
    paths = {
        "source_full": ("source_materialization", "lookup"),
        "natural_exact": ("natural_exact", "lookup"),
        "scheduled_exact": ("scheduled_exact", "lookup"),
        "one_owner_exact": ("one_owner_exact", "lookup"),
        "one_owner_append": ("one_owner_append", "lookup"),
        "synthetic_routing": ("synthetic_routing", "lookup"),
        "padding_only": ("padding_only", "lookup"),
        "compiled_delta_append": (
            "compiled_delta_append",
            "lookup",
        ),
        "compiled_latest_append": (
            "compiled_latest_append",
            "lookup",
        ),
    }
    values = {}
    for name, path in paths.items():
        current = []
        for report in rank_reports:
            value = report
            for key in path:
                value = value[key]
            current.append(value)
        values[name] = current
    return values


def _aggregate_lookup(
    values: list[dict[str, object]],
    hidden_size: int,
) -> dict[str, object]:
    world_size = len(values)
    remote = sum(value["remote_requested_tokens"] for value in values)
    served = sum(
        value["served_remote_requested_tokens"] for value in values
    )
    id_input = sum(value["id_collective_input_bytes"] for value in values)
    id_output = sum(
        value["id_collective_output_bytes"] for value in values
    )
    vector_input = sum(
        value["vector_collective_input_bytes"] for value in values
    )
    vector_output = sum(
        value["vector_collective_output_bytes"] for value in values
    )
    counts_input = sum(
        value["counts_collective_input_bytes"] for value in values
    )
    counts_output = sum(
        value["counts_collective_output_bytes"] for value in values
    )
    peer_matrix = [
        list(value["remote_send_counts"])
        for value in values
    ]
    receive_matrix = [
        list(value["remote_receive_counts"])
        for value in values
    ]
    send_hash_matrix = [
        list(value["remote_send_ids_sha256"])
        for value in values
    ]
    receive_hash_matrix = [
        list(value["remote_receive_ids_sha256"])
        for value in values
    ]
    cross_island_tokens = 0
    if world_size == 4:
        cross_island_tokens = sum(
            peer_matrix[source][destination]
            for source in range(world_size)
            for destination in range(world_size)
            if (
                source in {0, 1}
                and destination in {2, 3}
            )
            or (
                source in {2, 3}
                and destination in {0, 1}
            )
        )
    expected_counts = (
        0 if world_size == 1 else world_size * world_size * 8
    )
    actual_payload = sum(
        value["actual_collective_tensor_payload_bytes"]
        for value in values
    )
    off_diagonal = sum(
        value["off_diagonal_bytes"] for value in values
    )
    expected_off_diagonal = (
        0
        if world_size == 1
        else (
            world_size * (world_size - 1) * 16
            + remote * 2 * (8 + hidden_size * 4)
        )
    )
    return {
        "requested_tokens": sum(
            value["requested_tokens"] for value in values
        ),
        "unique_tokens_rank_local_sum": sum(
            value["unique_tokens"] for value in values
        ),
        "local_requested_tokens": sum(
            value["local_requested_tokens"] for value in values
        ),
        "remote_requested_tokens": remote,
        "served_remote_requested_tokens": served,
        "counts_collective_input_bytes": counts_input,
        "counts_collective_output_bytes": counts_output,
        "id_collective_input_bytes": id_input,
        "id_collective_output_bytes": id_output,
        "vector_collective_input_bytes": vector_input,
        "vector_collective_output_bytes": vector_output,
        "actual_collective_tensor_payload_bytes": actual_payload,
        "off_diagonal_bytes": off_diagonal,
        "collective_seconds_rank_sum": sum(
            value["collective_seconds"] for value in values
        ),
        "remote_send_matrix": peer_matrix,
        "remote_receive_matrix": receive_matrix,
        "remote_send_id_sha256_matrix": send_hash_matrix,
        "remote_receive_id_sha256_matrix": receive_hash_matrix,
        "rank_id_evidence": [
            {
                key: value[key]
                for key in (
                    "requested_ids_sha256",
                    "requested_unique_ids_sha256",
                    "local_requested_ids_sha256",
                    "local_unique_ids_sha256",
                    "remote_requested_ids_sha256",
                    "remote_unique_ids_sha256",
                    "served_remote_ids_sha256",
                    "served_remote_unique_ids_sha256",
                )
            }
            for value in values
        ],
        "cross_island_tokens": cross_island_tokens,
        "checks": {
            "request_service_conservation": remote == served,
            "id_input_formula": id_input == remote * 8,
            "id_output_formula": id_output == remote * 8,
            "vector_input_formula": (
                vector_input == remote * hidden_size * 4
            ),
            "vector_output_formula": (
                vector_output == remote * hidden_size * 4
            ),
            "counts_input_formula": counts_input == expected_counts,
            "counts_output_formula": counts_output == expected_counts,
            "actual_payload_component_sum": (
                actual_payload
                == counts_input
                + counts_output
                + id_input
                + id_output
                + vector_input
                + vector_output
            ),
            "off_diagonal_formula": (
                off_diagonal == expected_off_diagonal
            ),
            "collective_call_formula": all(
                value["collective_calls"]
                == (0 if world_size == 1 else 3)
                and value["off_diagonal_collective_calls"]
                == (0 if world_size == 1 else 3)
                for value in values
            ),
            "peer_rows_match_remote": all(
                sum(peer_matrix[rank])
                == values[rank]["remote_requested_tokens"]
                for rank in range(world_size)
            ),
            "peer_columns_match_served": all(
                sum(peer_matrix[source][destination]
                    for source in range(world_size))
                == values[destination][
                    "served_remote_requested_tokens"
                ]
                for destination in range(world_size)
            ),
            "peer_count_transpose": all(
                peer_matrix[source][destination]
                == receive_matrix[destination][source]
                for source in range(world_size)
                for destination in range(world_size)
            ),
            "peer_id_hash_transpose": all(
                send_hash_matrix[source][destination]
                == receive_hash_matrix[destination][source]
                for source in range(world_size)
                for destination in range(world_size)
            ),
        },
    }


def _aggregate(
    args: argparse.Namespace,
    runtime,
    rank_reports: tuple[dict[str, object], ...],
) -> dict[str, object]:
    action_path = _path(args.action_plan)
    stage_a_path = _path(args.stage_a_summary)
    sample_path = _path(args.sample_inputs)
    action_plan = D2ActionPlan.load(action_path)
    stage_a = json.loads(stage_a_path.read_text())
    sample = json.loads(sample_path.read_text())
    stage_a_ledger = _stage_a_plan_ledger(action_plan)
    expected_stage_a_ledger = _stage_a_expected_ledger(stage_a)
    source_checkpoint = _checkpoint_descriptor(
        action_plan,
        action_plan.source_version,
    )
    target_checkpoint = _checkpoint_descriptor(
        action_plan,
        action_plan.target_version,
    )
    program = _program_descriptor(stage_a)
    lookups = {
        name: _aggregate_lookup(
            values,
            int(rank_reports[0]["embedding_hidden_size"]),
        )
        for name, values in _lookup_reports(rank_reports).items()
    }
    traces = {
        tuple(value["token"] for value in report["phase_trace"])
        for report in rank_reports
    }
    transaction_traces = {
        tuple(
            value["token"]
            for value in report["transaction_phase_trace"]
        )
        for report in rank_reports
    }
    all_lookup_checks = all(
        all(value for value in report["checks"].values())
        for report in lookups.values()
    )
    exact_paths = (
        "source_materialization",
        "natural_exact",
        "scheduled_exact",
        "one_owner_exact",
    )
    exact_bitwise = all(
        report[path]["fragment_bitwise"]
        and report[path]["hidden_bitwise"]
        for report in rank_reports
        for path in exact_paths
    )
    compiled_invariants = all(
        report["compiled_retained"]["metrics"]["phase_counters"]
        == {
            "item_lookup_calls": 0,
            "embedding_collective_count": 0,
            "embedding_collective_bytes": 0,
            "old_kv_p2p_bytes": 0,
        }
        for report in rank_reports
    )
    transaction_ready = all(
        report["transaction_ready"]["status"] == "ready"
        and not report["transaction_ready"]["publishes_target_epoch"]
        for report in rank_reports
    )
    transaction_abort = all(
        report["transaction_abort"]["status"] == "abort"
        and not report["transaction_abort"]["publishes_target_epoch"]
        for report in rank_reports
    )
    capacity_expected = all(
        report["capacity"]["projected_admitted"]
        == (runtime.world_size > 1)
        for report in rank_reports
    )
    cross_island_lookup = (
        runtime.world_size != 4
        or lookups["synthetic_routing"]["cross_island_tokens"] > 0
    )
    cross_island_ring = (
        runtime.world_size != 4
        or any(
            report["placement_ring"]["cross_island_edge"]
            and report["placement_ring"]["old_kv_send_bytes"] > 0
            for report in rank_reports
        )
    )
    stage_a_parity = (
        runtime.world_size != 1
        or all(
            report["stage_a_parity"] is not None
            and report["stage_a_parity"]["fragment_bitwise"]
            and report["stage_a_parity"]["hidden_bitwise"]
            for report in rank_reports
        )
    )
    checks = {
        "stage_a_action_hash": (
            action_plan.content_sha256
            == stage_a["action_plan"]["content_sha256"]
        ),
        "stage_a_file_hash": (
            file_sha256(action_path)
            == stage_a["action_plan"]["file_sha256"]
        ),
        "stage_a_summary_hash_bound": (
            sample["stage_a_summary"]["sha256"]
            == file_sha256(stage_a_path)
        ),
        "stage_a_phase_ledger_parity": all(
            stage_a_ledger[key] == value
            for key, value in expected_stage_a_ledger.items()
        ),
        "all_rank_preflight": all(
            report["preflight"]["passed"] for report in rank_reports
        ),
        "world_size_one_stage_a_parity": stage_a_parity,
        "full_embedding_absent": all(
            report["full_embedding_lookup_forbidden"]
            and not report["capacity"]["full_embedding_parameter_present"]
            for report in rank_reports
        ),
        "sharded_exact_bitwise": exact_bitwise,
        "one_owner_append_bitwise": all(
            report["one_owner_append"]["bitwise"]
            for report in rank_reports
        ),
        "one_owner_append_empty_ranks_participate": all(
            (
                report["rank"] == 0
                and not report["one_owner_append"]["empty_rank"]
            )
            or (
                report["rank"] != 0
                and report["one_owner_append"]["empty_rank"]
                and report["one_owner_append"]["lookup"][
                    "requested_tokens"
                ]
                == 0
                and report["one_owner_append"]["lookup"][
                    "collective_calls"
                ]
                == 3
            )
            for report in rank_reports
        ),
        "synthetic_vectors_bitwise": all(
            report["synthetic_routing"]["item_vectors_bitwise"]
            and 0 in report["synthetic_routing"]["valid_item_ids"]
            for report in rank_reports
        ),
        "padding_only_zero": all(
            report["padding_only"]["zero_output"]
            for report in rank_reports
        ),
        "compiled_owner_local": all(
            report["compiled_retained"]["metadata"]["owner_rank"]
            == report["rank"]
            for report in rank_reports
        ),
        "compiled_reference": all(
            report["compiled_retained"]["reference_k_relative_l2"]
            <= 5e-4
            and report["compiled_retained"][
                "reference_v_relative_l2"
            ]
            <= 5e-4
            for report in rank_reports
        ),
        "compiled_phase_invariants": compiled_invariants,
        "compiled_delta_append_bitwise": all(
            report["compiled_delta_append"]["bitwise"]
            and report["compiled_delta_append"]["lookup"][
                "requested_tokens"
            ]
            == report["compiled_delta_append"]["tokens"]
            and report["compiled_delta_append"]["lengths"]
            == [
                report["compiled_retained"]["metadata"]["lengths"][0]
                + report["compiled_delta_append"]["tokens"]
            ]
            for report in rank_reports
        ),
        "compiled_latest_append_bitwise": all(
            report["compiled_latest_append"]["bitwise"]
            and report["compiled_latest_append"]["lookup"][
                "requested_tokens"
            ]
            == report["compiled_latest_append"]["tokens"]
            and report["compiled_latest_append"]["lengths"]
            == [
                report["compiled_delta_append"]["lengths"][0]
                + report["compiled_latest_append"]["tokens"]
            ]
            for report in rank_reports
        ),
        "lookup_bytes_reconstruct": all_lookup_checks,
        "lookup_id_evidence_reconstruct": all(
            evidence["passed"]
            for report in rank_reports
            for evidence in report["lookup_input_evidence"].values()
        ),
        "collective_trace_equal": len(traces) == 1,
        "transaction_collective_trace_equal": (
            len(transaction_traces) == 1
        ),
        "cooperative_failure_propagated": all(
            not report["cooperative_failure"]["passed"]
            and report["cooperative_failure"]["failure_reasons"]
            for report in rank_reports
        ),
        "transaction_private_ready": transaction_ready,
        "transaction_abort_no_publication": transaction_abort,
        "placement_return_bitwise": all(
            report["placement_ring"]["returned_output_bitwise"]
            for report in rank_reports
        ),
        "w4_cross_island_lookup": cross_island_lookup,
        "w4_cross_island_p2p": cross_island_ring,
        "capacity_expected_boundary": capacity_expected,
    }
    return {
        "protocol": PROTOCOL,
        "status": "complete" if all(checks.values()) else "failed",
        "scientific_result": False,
        "scope": {
            "primitive_correctness_only": True,
            "full_mixed_wave_executed": False,
            "target_epoch_published": False,
            "nccl_wire_bytes_observed": False,
            "actual_collective_tensor_payload_recorded": True,
            "real_theta1_theta2_samples": True,
            "synthetic_adversarial_cases": True,
            "old_kv_source_materialized_as_test_fixture": True,
            "full_plan_ledger_replayed_without_full_wave_execution": True,
        },
        "action_plan": {
            "path": str(action_path.relative_to(ROOT)),
            "content_sha256": action_plan.content_sha256,
            "file_sha256": file_sha256(action_path),
            "counts": action_plan.counts.to_dict(),
        },
        "stage_a_summary": {
            "path": str(stage_a_path.relative_to(ROOT)),
            "sha256": file_sha256(stage_a_path),
            "status": stage_a["status"],
            "stage_b_entry": stage_a["stage_b_entry"],
        },
        "sample_inputs": {
            "path": str(sample_path.relative_to(ROOT)),
            "sha256": file_sha256(sample_path),
        },
        "model_inputs": {
            "source_checkpoint": source_checkpoint,
            "target_checkpoint": target_checkpoint,
            "program": program,
        },
        "configuration": {
            "world_size": runtime.world_size,
            "backend": runtime.backend,
            "embedding_owner": "item_id_mod_world_size",
            "embedding_owner_sha256": canonical_sha256(
                {
                    "rule": "item_id_mod_world_size",
                    "world_size": runtime.world_size,
                }
            ),
            "record_owner": "strict_cow_lpt",
            "record_owner_map_sha256": sample["selections"][
                str(runtime.world_size)
            ]["owner_map_sha256"],
            "embedding_transport_dtype": "float32",
            "publication_dtype": "float16",
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                "default_runtime_order",
            ),
        },
        "stage_a_ledger_recheck": {
            "observed": stage_a_ledger,
            "expected": expected_stage_a_ledger,
        },
        "lookups": lookups,
        "rank_reports": list(rank_reports),
        "checks": checks,
    }


def run(args: argparse.Namespace) -> dict[str, object] | None:
    runtime = init_d2_distributed_runtime(
        backend="nccl",
        timeout_seconds=args.timeout_seconds,
    )
    try:
        if args.case == "hard_failure":
            if runtime.world_size < 2:
                raise ValueError("hard failure case requires at least two ranks")
            if runtime.rank == 1:
                os._exit(23)
            dist.barrier()
            raise RuntimeError("hard failure peer unexpectedly survived")
        if args.output is None:
            raise ValueError("normal Stage B run requires --output")
        torch.cuda.reset_peak_memory_stats(runtime.device)
        local = _normal_rank(args, runtime)
        launcher_guard = D2CollectiveGuard(
            runtime,
            (
                "rank_report_gather",
                "aggregate_status_broadcast",
                "output_vote",
            ),
        )
        launcher_guard.enter("rank_report_gather")
        gathered: list[object] = [None] * runtime.world_size
        dist.all_gather_object(gathered, local)
        output = None
        aggregate_envelope = None
        if runtime.is_primary:
            try:
                if not all(isinstance(value, dict) for value in gathered):
                    raise RuntimeError("D2 Stage B rank report is invalid")
                output = _aggregate(args, runtime, tuple(gathered))
                if output["status"] != "complete":
                    raise RuntimeError(
                        "D2 Stage B distributed checks failed: "
                        f"{output['checks']}"
                    )
                output["launcher_control"] = {
                    "phase_order": [
                        "rank_report_gather",
                        "aggregate_status_broadcast",
                        "output_vote",
                    ],
                    "rank0_errors_broadcast": True,
                }
                aggregate_envelope = {"ok": True}
            except Exception as error:
                aggregate_envelope = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        aggregate_envelope = broadcast_d2_metadata(
            runtime,
            aggregate_envelope,
            guard=launcher_guard,
            phase="aggregate_status_broadcast",
        )
        if not aggregate_envelope["ok"]:
            raise RuntimeError(
                "D2 Stage B aggregate failed: "
                f"{aggregate_envelope['error']}"
            )
        output_failures = ()
        if runtime.is_primary:
            try:
                if output is None:
                    raise RuntimeError("D2 Stage B aggregate is absent")
                output_path = _path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = output_path.with_suffix(
                    f"{output_path.suffix}.tmp"
                )
                temporary_path.write_text(
                    json.dumps(output, indent=2, sort_keys=True) + "\n"
                )
                os.replace(temporary_path, output_path)
            except Exception as error:
                output_failures = (
                    f"output write: {type(error).__name__}: {error}",
                )
        output_vote = vote_d2_preflight(
            runtime,
            output_failures,
            guard=launcher_guard,
            phase="output_vote",
        )
        launcher_guard.require_complete()
        if not output_vote.passed:
            raise RuntimeError(
                f"D2 Stage B output failed: {output_vote.failure_reasons}"
            )
        return output
    finally:
        close_d2_distributed_runtime(runtime)


def main() -> None:
    output = run(parse_args())
    if output is not None:
        print(
            json.dumps(
                {
                    "status": output["status"],
                    "world_size": output["configuration"]["world_size"],
                    "scientific_result": output["scientific_result"],
                    "checks": output["checks"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
