from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

from hstu_kvcache.migration import (
    D2ActionPlan,
    JaggedMigratedKVBatch,
)
from hstu_kvcache.migration.design2_plan import file_sha256
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    DirectOldKVProgram,
    load_direct_oldkv_program,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_PLAN = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_OUTPUT = (
    "configs/cohortkv_d2/stage_a_p2p_topology.json"
)
PROTOCOL = "cohortkv_d2_stage_a_p2p_topology_v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--busy-threshold-mib", type=int, default=4096)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _inventory() -> dict[int, dict[str, object]]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,"
            "memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    output = {}
    for line in query.strip().splitlines():
        index, uuid, bus_id, name, total, used, free = (
            value.strip() for value in line.split(",")
        )
        output[int(index)] = {
            "index": int(index),
            "uuid": uuid,
            "pci_bus_id": bus_id,
            "name": name,
            "total_mib": int(total),
            "used_mib_before": int(used),
            "free_mib_before": int(free),
        }
    return output


def _copy_once(
    source_index: int,
    destination_index: int,
    size_bytes: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    with torch.cuda.device(source_index):
        source = torch.empty(
            size_bytes,
            dtype=torch.uint8,
            device=f"cuda:{source_index}",
        )
        source.fill_(7)
        torch.cuda.synchronize(source_index)
    with torch.cuda.device(destination_index):
        destination = torch.empty(
            size_bytes,
            dtype=torch.uint8,
            device=f"cuda:{destination_index}",
        )
        stream = torch.cuda.Stream(device=destination_index)
        for _ in range(warmup):
            with torch.cuda.stream(stream):
                destination.copy_(source, non_blocking=True)
        stream.synchronize()
        samples_ms = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                destination.copy_(source, non_blocking=True)
                end.record(stream)
            end.synchronize()
            samples_ms.append(float(start.elapsed_time(end)))
        valid = int(destination[0].item()) == 7
    median_ms = float(statistics.median(samples_ms))
    result = {
        "source_index": source_index,
        "destination_index": destination_index,
        "size_bytes": size_bytes,
        "warmup": warmup,
        "repeats": repeats,
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "median_gb_per_second": (
            float(size_bytes / (median_ms / 1000.0) / 1e9)
        ),
        "sample_copy_valid": valid,
    }
    del destination
    del source
    for index in (source_index, destination_index):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
    return result


def _concurrent_copy_once(
    pairs: tuple[tuple[int, int], ...],
    size_bytes: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    states = []
    for value, (source_index, destination_index) in enumerate(pairs):
        with torch.cuda.device(source_index):
            source = torch.full(
                (size_bytes,),
                value + 3,
                dtype=torch.uint8,
                device=f"cuda:{source_index}",
            )
        with torch.cuda.device(destination_index):
            destination = torch.empty(
                size_bytes,
                dtype=torch.uint8,
                device=f"cuda:{destination_index}",
            )
            stream = torch.cuda.Stream(device=destination_index)
        states.append((source, destination, stream))
    for _ in range(warmup):
        for source, destination, stream in states:
            with torch.cuda.stream(stream):
                destination.copy_(source, non_blocking=True)
        for _, _, stream in states:
            stream.synchronize()
    samples_ms = []
    for _ in range(repeats):
        for source_index, destination_index in pairs:
            torch.cuda.synchronize(source_index)
            torch.cuda.synchronize(destination_index)
        started = time.perf_counter()
        for source, destination, stream in states:
            with torch.cuda.stream(stream):
                destination.copy_(source, non_blocking=True)
        for _, _, stream in states:
            stream.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    valid = all(
        int(destination[0].item()) == value + 3
        for value, (_, destination, _) in enumerate(states)
    )
    median_ms = float(statistics.median(samples_ms))
    result = {
        "pairs": [
            {
                "source_index": source,
                "destination_index": destination,
            }
            for source, destination in pairs
        ],
        "size_bytes_per_pair": size_bytes,
        "aggregate_bytes_per_repeat": size_bytes * len(pairs),
        "warmup": warmup,
        "repeats": repeats,
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "aggregate_median_gb_per_second": (
            float(
                size_bytes
                * len(pairs)
                / (median_ms / 1000.0)
                / 1e9
            )
        ),
        "sample_copies_valid": valid,
    }
    for source, destination, _ in states:
        del destination
        del source
    for index in {value for pair in pairs for value in pair}:
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
    return result


def _compiled_overlap_once(
    source_index: int,
    destination_index: int,
    token_count: int,
    program_cpu: DirectOldKVProgram,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    source_device = torch.device("cuda", source_index)
    destination_device = torch.device("cuda", destination_index)
    operator = DirectOldKVFusedOperator()
    program = operator.prepare_program(program_cpu, source_device)
    with torch.cuda.device(source_device):
        lengths = torch.tensor(
            [token_count],
            dtype=torch.long,
            device=source_device,
        )
        offsets = torch.tensor(
            [0, token_count],
            dtype=torch.long,
            device=source_device,
        )
        source = JaggedMigratedKVBatch(
            record_ids=(0,),
            migration_anchor_version=program.source_version,
            served_kv_target=program.source_version,
            k=torch.randn(
                program.num_layers,
                token_count,
                program.kv_width,
                dtype=torch.float16,
                device=source_device,
            ),
            v=torch.randn(
                program.num_layers,
                token_count,
                program.kv_width,
                dtype=torch.float16,
                device=source_device,
            ),
            lengths=lengths,
            offsets=offsets,
        )
        output = JaggedMigratedKVBatch(
            record_ids=(0,),
            migration_anchor_version=program.source_version,
            served_kv_target=program.target_version,
            k=torch.empty_like(source.k),
            v=torch.empty_like(source.v),
            lengths=lengths.clone(),
            offsets=offsets.clone(),
        )
        payload_bytes = (
            source.k.numel() * source.k.element_size()
            + source.v.numel() * source.v.element_size()
        )
        compute_stream = torch.cuda.Stream(device=source_device)
        copy_source = torch.full(
            (payload_bytes,),
            11,
            dtype=torch.uint8,
            device=source_device,
        )
    with torch.cuda.device(destination_device):
        copy_destination = torch.empty(
            payload_bytes,
            dtype=torch.uint8,
            device=destination_device,
        )
        copy_stream = torch.cuda.Stream(device=destination_device)

    def enqueue_compute() -> None:
        with torch.cuda.stream(compute_stream):
            operator.execute_into(program, source, output)

    def enqueue_copy() -> None:
        with torch.cuda.stream(copy_stream):
            copy_destination.copy_(copy_source, non_blocking=True)

    for _ in range(warmup):
        enqueue_compute()
        enqueue_copy()
    compute_stream.synchronize()
    copy_stream.synchronize()
    compute_samples = []
    copy_samples = []
    overlap_samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(compute_stream):
            start.record(compute_stream)
            operator.execute_into(program, source, output)
            end.record(compute_stream)
        end.synchronize()
        compute_samples.append(float(start.elapsed_time(end)))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(copy_stream):
            start.record(copy_stream)
            copy_destination.copy_(copy_source, non_blocking=True)
            end.record(copy_stream)
        end.synchronize()
        copy_samples.append(float(start.elapsed_time(end)))
        torch.cuda.synchronize(source_device)
        torch.cuda.synchronize(destination_device)
        started = time.perf_counter()
        enqueue_compute()
        enqueue_copy()
        compute_stream.synchronize()
        copy_stream.synchronize()
        overlap_samples.append(
            (time.perf_counter() - started) * 1000.0
        )
    compute_median = float(statistics.median(compute_samples))
    copy_median = float(statistics.median(copy_samples))
    overlap_median = float(statistics.median(overlap_samples))
    sequential = compute_median + copy_median
    result = {
        "source_index": source_index,
        "destination_index": destination_index,
        "token_count": token_count,
        "extent_payload_bytes": payload_bytes,
        "extent_metadata_bytes": source.nbytes - payload_bytes,
        "warmup": warmup,
        "repeats": repeats,
        "compute_samples_ms": compute_samples,
        "copy_samples_ms": copy_samples,
        "overlap_wall_samples_ms": overlap_samples,
        "compute_median_ms": compute_median,
        "copy_median_ms": copy_median,
        "sequential_median_ms": sequential,
        "overlap_wall_median_ms": overlap_median,
        "overlap_gain_fraction": (
            float(1.0 - overlap_median / sequential)
            if sequential
            else None
        ),
        "copy_valid": int(copy_destination[0].item()) == 11,
        "compute_finite": (
            bool(torch.isfinite(output.k).all())
            and bool(torch.isfinite(output.v).all())
        ),
    }
    for index in (source_index, destination_index):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
    return result


def _owner_load_characterization(
    plan: D2ActionPlan,
    world_size: int,
    bytes_per_token: int,
) -> dict[str, object]:
    records = tuple(
        value
        for value in plan.records
        if value.requested_action == "compiled"
    )

    def values(owner_map: dict[int, int]) -> dict[str, object]:
        loads = [0] * world_size
        counts = [0] * world_size
        for record in records:
            owner = owner_map[record.record_id]
            loads[owner] += record.retained_tokens * bytes_per_token
            counts[owner] += 1
        mean = sum(loads) / world_size
        return {
            "records_per_rank": counts,
            "retained_bytes_per_rank": loads,
            "maximum_over_mean": (
                float(max(loads) / mean) if mean else 0.0
            ),
        }

    modulo = {
        value.record_id: value.record_id % world_size
        for value in records
    }
    loads = [0] * world_size
    lpt = {}
    for record in sorted(
        records,
        key=lambda value: (
            -value.retained_tokens,
            value.record_id,
        ),
    ):
        owner = min(
            range(world_size),
            key=lambda rank: (loads[rank], rank),
        )
        lpt[record.record_id] = owner
        loads[owner] += record.retained_tokens * bytes_per_token
    injected = {value.record_id: 0 for value in records}
    return {
        "world_size": world_size,
        "modulo": values(modulo),
        "retained_lpt": values(lpt),
        "injected_all_rank0": values(injected),
        "execution_measured": False,
    }


def _owner_execution_microbenchmark(
    plan: D2ActionPlan,
    devices: tuple[int, ...],
    program_cpu: DirectOldKVProgram,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    compiled = sorted(
        (
            value
            for value in plan.records
            if value.requested_action == "compiled"
        ),
        key=lambda value: (
            value.retained_tokens,
            value.record_id,
        ),
    )
    sample_count = min(16, len(compiled))
    sampled = tuple(
        compiled[
            round(
                index
                * (len(compiled) - 1)
                / max(sample_count - 1, 1)
            )
        ]
        for index in range(sample_count)
    )
    balanced_loads = [0] * len(devices)
    balanced = {}
    for record in sorted(
        sampled,
        key=lambda value: (
            -value.retained_tokens,
            value.record_id,
        ),
    ):
        owner = min(
            range(len(devices)),
            key=lambda rank: (balanced_loads[rank], rank),
        )
        balanced[record.record_id] = owner
        balanced_loads[owner] += record.retained_tokens
    scenarios = {
        "retained_lpt": balanced,
        "injected_all_rank0": {
            value.record_id: 0 for value in sampled
        },
    }
    operator = DirectOldKVFusedOperator()
    scenario_results = {}
    for name, owner_map in scenarios.items():
        states = []
        for rank, device_index in enumerate(devices):
            assigned = tuple(
                value
                for value in sampled
                if owner_map[value.record_id] == rank
            )
            if not assigned:
                continue
            device = torch.device("cuda", device_index)
            program = operator.prepare_program(program_cpu, device)
            lengths_tuple = tuple(
                value.retained_tokens for value in assigned
            )
            token_count = sum(lengths_tuple)
            with torch.cuda.device(device):
                lengths = torch.tensor(
                    lengths_tuple,
                    dtype=torch.long,
                    device=device,
                )
                offsets = torch.cat(
                    (
                        torch.zeros(
                            1,
                            dtype=torch.long,
                            device=device,
                        ),
                        lengths.cumsum(0),
                    )
                )
                source = JaggedMigratedKVBatch(
                    record_ids=tuple(
                        value.record_id for value in assigned
                    ),
                    migration_anchor_version=program.source_version,
                    served_kv_target=program.source_version,
                    k=torch.zeros(
                        program.num_layers,
                        token_count,
                        program.kv_width,
                        dtype=torch.float16,
                        device=device,
                    ),
                    v=torch.zeros(
                        program.num_layers,
                        token_count,
                        program.kv_width,
                        dtype=torch.float16,
                        device=device,
                    ),
                    lengths=lengths,
                    offsets=offsets,
                )
                output = JaggedMigratedKVBatch(
                    record_ids=source.record_ids,
                    migration_anchor_version=program.source_version,
                    served_kv_target=program.target_version,
                    k=torch.empty_like(source.k),
                    v=torch.empty_like(source.v),
                    lengths=lengths.clone(),
                    offsets=offsets.clone(),
                )
                stream = torch.cuda.Stream(device=device)
            states.append(
                (
                    rank,
                    device_index,
                    program,
                    source,
                    output,
                    stream,
                )
            )

        for _ in range(min(warmup, 2)):
            for _, _, program, source, output, stream in states:
                with torch.cuda.stream(stream):
                    operator.execute_into(program, source, output)
        for *_, stream in states:
            stream.synchronize()
        samples_ms = []
        for _ in range(min(repeats, 5)):
            for device_index in devices:
                torch.cuda.synchronize(device_index)
            started = time.perf_counter()
            for _, _, program, source, output, stream in states:
                with torch.cuda.stream(stream):
                    operator.execute_into(program, source, output)
            for *_, stream in states:
                stream.synchronize()
            samples_ms.append(
                (time.perf_counter() - started) * 1000.0
            )
        rank_values = []
        for rank, device_index in enumerate(devices):
            assigned = tuple(
                value
                for value in sampled
                if owner_map[value.record_id] == rank
            )
            rank_values.append(
                {
                    "rank": rank,
                    "device_index": device_index,
                    "records": len(assigned),
                    "retained_tokens": sum(
                        value.retained_tokens for value in assigned
                    ),
                }
            )
        scenario_results[name] = {
            "rank_values": rank_values,
            "samples_ms": samples_ms,
            "median_ms": float(statistics.median(samples_ms)),
            "outputs_finite": all(
                bool(torch.isfinite(output.k).all())
                and bool(torch.isfinite(output.v).all())
                for _, _, _, _, output, _ in states
            ),
        }
        states.clear()
        for device_index in devices:
            with torch.cuda.device(device_index):
                torch.cuda.empty_cache()
    balanced_median = scenario_results["retained_lpt"]["median_ms"]
    imbalanced_median = scenario_results[
        "injected_all_rank0"
    ]["median_ms"]
    return {
        "devices": list(devices),
        "sampled_records": [
            {
                "record_id": value.record_id,
                "retained_tokens": value.retained_tokens,
            }
            for value in sampled
        ],
        "sample_selection": (
            "sixteen_length_quantiles_of_compiled_retained_extents"
        ),
        "warmup": min(warmup, 2),
        "repeats": min(repeats, 5),
        "scenarios": scenario_results,
        "imbalanced_over_balanced_makespan": float(
            imbalanced_median / balanced_median
        ),
        "full_wave_execution": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError(
            "Stage A topology requires physical CUDA indices without remapping"
        )
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("D2 P2P repetitions must be positive")
    output_path = _path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            "D2 Stage A P2P output exists; pass --force"
        )
    devices = tuple(
        int(value) for value in args.devices.split(",") if value
    )
    inventory = _inventory()
    if (
        not torch.cuda.is_available()
        or not devices
        or len(set(devices)) != len(devices)
        or any(value not in inventory for value in devices)
        or any(value >= torch.cuda.device_count() for value in devices)
    ):
        raise ValueError("D2 P2P physical device selection is invalid")
    action_plan_path = _path(args.action_plan)
    plan = D2ActionPlan.load(action_plan_path)
    bytes_per_token = 16 * 512 * 2 * 2
    extent_bytes = sorted(
        value.retained_tokens * bytes_per_token
        for value in plan.records
        if value.requested_action == "compiled"
    )

    def quantile(fraction: float) -> int:
        index = round((len(extent_bytes) - 1) * fraction)
        return extent_bytes[index]

    sizes = tuple(
        sorted(
            {
                extent_bytes[0],
                quantile(0.5),
                quantile(0.95),
                extent_bytes[-1],
            }
        )
    )
    peer_matrix = []
    supported_pairs = []
    for source in devices:
        for destination in devices:
            if source == destination:
                supported = True
            else:
                supported = bool(
                    torch.cuda.can_device_access_peer(
                        source,
                        destination,
                    )
                )
                if supported:
                    supported_pairs.append((source, destination))
            peer_matrix.append(
                {
                    "source_index": source,
                    "destination_index": destination,
                    "direct_peer_supported": supported,
                }
            )
    idle = {
        index
        for index in devices
        if inventory[index]["used_mib_before"]
        <= args.busy_threshold_mib
    }
    measured_pairs = [
        pair
        for pair in supported_pairs
        if pair[0] in idle and pair[1] in idle
    ]
    measurements = [
        _copy_once(
            source,
            destination,
            size,
            args.warmup,
            args.repeats,
        )
        for source, destination in measured_pairs
        for size in sizes
    ]
    canonical_pairs = tuple(
        (source, destination)
        for source, destination in supported_pairs
        if source < destination
        and source in idle
        and destination in idle
    )
    disjoint_pairs = []
    used_devices = set()
    for pair in canonical_pairs:
        if not used_devices.intersection(pair):
            disjoint_pairs.append(pair)
            used_devices.update(pair)
    concurrent = (
        None
        if len(disjoint_pairs) < 2
        else _concurrent_copy_once(
            tuple(disjoint_pairs),
            quantile(0.5),
            args.warmup,
            args.repeats,
        )
    )
    upstream = json.loads(
        _path(plan.provenance.artifact).read_text()
    )
    compiler_descriptor = upstream["input_provenance"]["compiler"]
    compiler_path = _path(compiler_descriptor["path"])
    if file_sha256(compiler_path) != compiler_descriptor["sha256"]:
        raise ValueError("D2 P2P compiler artifact hash differs")
    compiler = json.loads(compiler_path.read_text())
    pair = next(
        value
        for value in compiler["pairs"]
        if value["source_version"] == plan.source_version
        and value["target_version"] == plan.target_version
    )
    program_descriptor = pair["direct_program"]
    program_cpu, loaded_program = load_direct_oldkv_program(
        _path(program_descriptor["path"]),
        expected_sha256=program_descriptor["sha256"],
        expected_source_version=plan.source_version,
        expected_target_version=plan.target_version,
        expected_num_layers=16,
        expected_kv_width=512,
    )
    median_tokens = quantile(0.5) // bytes_per_token
    overlap = [
        _compiled_overlap_once(
            source,
            destination,
            median_tokens,
            program_cpu,
            args.warmup,
            args.repeats,
        )
        for source, destination in canonical_pairs
    ]
    owner_execution = (
        _owner_execution_microbenchmark(
            plan,
            devices,
            program_cpu,
            args.warmup,
            args.repeats,
        )
        if len(devices) == 4 and set(devices).issubset(idle)
        else None
    )
    busy_supported_pairs = [
        {
            "source_index": source,
            "destination_index": destination,
            "reason": "device_busy_before_stage_a",
        }
        for source, destination in supported_pairs
        if (source, destination) not in measured_pairs
    ]
    topology_text = subprocess.check_output(
        ["nvidia-smi", "topo", "-m"],
        text=True,
    )
    checks = {
        "physical_indices_not_remapped": True,
        "peer_matrix_complete": (
            len(peer_matrix) == len(devices) ** 2
        ),
        "at_least_one_peer_pair_measured": bool(measured_pairs),
        "both_directions_measured": all(
            (destination, source) in measured_pairs
            for source, destination in measured_pairs
        ),
        "real_extent_sizes": (
            sizes[0] == 393216
            and sizes[-1] == 66912256
        ),
        "sample_copies_valid": all(
            value["sample_copy_valid"] for value in measurements
        ),
        "busy_devices_not_benchmarked": all(
            value["source_index"] in idle
            and value["destination_index"] in idle
            for value in measurements
        ),
        "all_idle_direct_pairs_measured": (
            set(measured_pairs)
            == {
                pair
                for pair in supported_pairs
                if pair[0] in idle and pair[1] in idle
            }
        ),
        "concurrent_measurement_consistent": (
            (
                len(disjoint_pairs) < 2
                and concurrent is None
            )
            or (
                concurrent is not None
                and concurrent["sample_copies_valid"]
            )
        ),
        "copy_compute_overlap_consistent": (
            len(overlap) == len(canonical_pairs)
            and all(
                value["copy_valid"]
                and value["compute_finite"]
                for value in overlap
            )
        ),
        "owner_execution_consistent": (
            owner_execution is None
            or all(
                value["outputs_finite"]
                for value in owner_execution["scenarios"].values()
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"D2 Stage A P2P checks failed: {checks}"
        )
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "action_plan": {
            "path": str(action_plan_path.relative_to(ROOT)),
            "content_sha256": plan.content_sha256,
            "file_sha256": file_sha256(action_plan_path),
        },
        "devices": [inventory[index] for index in devices],
        "topology_text": topology_text,
        "peer_matrix": peer_matrix,
        "extent_size_bytes": {
            "minimum": extent_bytes[0],
            "median": quantile(0.5),
            "p95": quantile(0.95),
            "maximum": extent_bytes[-1],
            "benchmarked": list(sizes),
        },
        "direct_peer_measurements": measurements,
        "concurrent_direct_peer_measurement": concurrent,
        "copy_compute_overlap_measurements": overlap,
        "owner_load_characterization": [
            _owner_load_characterization(
                plan,
                world_size,
                bytes_per_token,
            )
            for world_size in (1, 2, 4)
        ],
        "owner_execution_microbenchmark": owner_execution,
        "program": {
            "path": str(
                _path(program_descriptor["path"]).relative_to(ROOT)
            ),
            "sha256": loaded_program["sha256"],
            "tensor_bytes": program_cpu.nbytes,
        },
        "unmeasured_supported_pairs": busy_supported_pairs,
        "scope": {
            "direct_cuda_peer_copy_measured": True,
            "nccl_send_recv_measured": False,
            "cross_island_route_measured": False,
            "full_four_gpu_direct_peer_topology_measured": (
                len(devices) == 4
                and not busy_supported_pairs
            ),
            "concurrent_copy_measured": concurrent is not None,
            "real_compiled_compute_overlap_measured": bool(overlap),
            "balanced_and_injected_owner_execution_measured": (
                owner_execution is not None
            ),
            "owner_execution_is_sampled_microbenchmark": True,
            "owner_loads_characterized": True,
            "busy_threshold_mib": args.busy_threshold_mib,
            "busy_device_indices": sorted(set(devices) - idle),
            "stage_b_must_remeasure_all_declared_routes": True,
        },
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
