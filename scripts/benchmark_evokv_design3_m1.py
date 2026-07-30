from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.data import load_prepared_exposure_plan
from hstu_kvcache.migration import canonical_json_bytes
from hstu_kvcache.migration.cohort_jagged import (
    JaggedMigratedKVBatch,
)
from hstu_kvcache.migration.design2_distributed import (
    close_d2_distributed_runtime,
    init_d2_distributed_runtime,
)
from hstu_kvcache.migration.design2_integrated import (
    IntegratedAppendOnlyKVBatch,
    integrated_sharded_append_only,
    integrated_sharded_exact,
)
from hstu_kvcache.migration.design2_plan import (
    canonical_sha256,
    file_sha256,
)
from hstu_kvcache.migration.design3_checkpoint import (
    load_runtime_sharded_hstu,
    resolve_version_checkpoint,
    training_model_config,
)
from hstu_kvcache.migration.design3_residency import (
    D3ResidencyPlan,
    d3_stack_revision_sha256,
)
from hstu_kvcache.migration.design3_store import (
    PageableDramExtentStore,
)
from hstu_kvcache.migration.recompute import RawHistoryBatch
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    load_direct_oldkv_program,
)
from hstu_kvcache.models import HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "evokv_design3_m1_qk_pageable_s0_development_v0"
S1_PROTOCOL = "evokv_design3_m1_qk_pageable_s1_development_v0"
D3_PROTOCOL = "evokv_design3_m1_qk_decoupled_io_d3_development_v2"
D3_RESIDENCY_PROTOCOL = (
    "evokv_design3_m1_qk_route_aware_residency_d3_development_v3"
)
D3_RUNNER_PROTOCOL = (
    "evokv_design3_m1_qk_decoupled_residency_runner_development_v2"
)
DEFAULT_PREPARED_DATA = (
    "data/processed/evokv_d3_m1_qk_entity_2560.npz"
)
DEFAULT_ACTION_SNAPSHOT = (
    "configs/evokv_d3/m1/"
    "qk_entity_adjacent_action_snapshot.json"
)
DEFAULT_TRAINING_RESULT = (
    "results/system/evokv_design3_m1/"
    "qk_entity_h1536_sharded_two_version_training_seed0.json"
)
DEFAULT_CHECKPOINT_DIR = (
    "checkpoints/evokv_design3_m1_qk_entity_h1536/seed0"
)
DEFAULT_COMPILER_RESULT = (
    "results/system/evokv_design3_m1/"
    "qk_entity_h1536_adjacent_compiler_seed0.json"
)
DEFAULT_STORE_DIR = "/dev/shm/evokv_d3"
DEFAULT_OUTPUT = (
    "results/system/evokv_design3_m1/"
    "qk_entity_h1536_s0_canary_seed0.json"
)


@dataclass(frozen=True)
class M1Action:
    record_id: int
    prepared_user_id: int
    requested_action: str
    requested_reason: str
    old_tokens: int
    retained_start: int
    retained_tokens: int
    delta_start: int
    delta_tokens: int
    target_prefix_tokens: int
    latest_tokens: int
    final_tokens: int

    def __post_init__(self) -> None:
        if (
            self.record_id < 0
            or self.prepared_user_id < 1
            or self.requested_action not in {"compiled", "exact"}
            or self.requested_reason
            not in {"migrate", "scheduled_exact", "natural_exact"}
            or (
                self.requested_action == "compiled"
                and self.requested_reason != "migrate"
            )
            or (
                self.requested_action == "exact"
                and self.requested_reason
                not in {"scheduled_exact", "natural_exact"}
            )
            or min(
                self.old_tokens,
                self.retained_start,
                self.retained_tokens,
                self.delta_start,
                self.delta_tokens,
                self.target_prefix_tokens,
                self.latest_tokens,
                self.final_tokens,
            )
            < 0
            or self.old_tokens < 1
            or self.retained_tokens < 1
            or self.final_tokens < 1
            or self.retained_start + self.retained_tokens
            != self.old_tokens
            or self.delta_start != self.retained_tokens
            or self.retained_tokens + self.delta_tokens
            != self.target_prefix_tokens
            or self.target_prefix_tokens + self.latest_tokens
            != self.final_tokens
            or self.latest_tokens != 1
        ):
            raise ValueError("M1 action extent is invalid")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> M1Action:
        return cls(
            record_id=int(value["record_id"]),
            prepared_user_id=int(value["prepared_user_id"]),
            requested_action=str(value["requested_action"]),
            requested_reason=str(value["requested_reason"]),
            old_tokens=int(value["old_tokens"]),
            retained_start=int(value["retained_start"]),
            retained_tokens=int(value["retained_tokens"]),
            delta_start=int(value["delta_start"]),
            delta_tokens=int(value["delta_tokens"]),
            target_prefix_tokens=int(value["target_prefix_tokens"]),
            latest_tokens=int(value["latest_tokens"]),
            final_tokens=int(value["final_tokens"]),
        )

    @property
    def route(self) -> str:
        return self.requested_action

    @property
    def suffix_tokens(self) -> int:
        return self.final_tokens - self.delta_start


@dataclass(frozen=True)
class M1Group:
    ordinal: int
    route: str
    record_ids_by_rank: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            self.ordinal < 0
            or self.route not in {"compiled", "exact"}
            or not self.record_ids_by_rank
            or any(
                len(values) != len(set(values))
                for values in self.record_ids_by_rank
            )
        ):
            raise ValueError("M1 group is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "route": self.route,
            "record_ids_by_rank": [
                list(values) for values in self.record_ids_by_rank
            ],
        }


@dataclass
class M1PinnedSlot:
    source_k: torch.Tensor
    source_v: torch.Tensor
    source_lengths: torch.Tensor
    source_offsets: torch.Tensor
    history_item_ids: torch.Tensor
    history_behaviors: torch.Tensor
    history_time_deltas: torch.Tensor
    history_lengths: torch.Tensor
    output_k: torch.Tensor
    output_v: torch.Tensor

    @property
    def nbytes(self) -> int:
        storages = {
            value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
            for value in self.__dict__.values()
        }
        return sum(storages.values())


@dataclass(frozen=True)
class M1Histories:
    old: dict[int, dict[str, np.ndarray]]
    target: dict[int, dict[str, np.ndarray]]


@dataclass(frozen=True)
class M1S1StagedGroup:
    group: M1Group
    actions: tuple[M1Action, ...]
    local_microbatches: tuple[tuple[M1Action, ...], ...]
    micro_steps: int
    source: JaggedMigratedKVBatch | None
    device_histories: tuple[RawHistoryBatch, ...]
    oldkv_read_bytes: int
    h2d_bytes: int
    pageable_to_pinned_seconds: float
    h2d_seconds: float
    staging_started_at: float
    staging_finished_at: float
    slot_index: int
    input_pipeline: str = "serial_microbatch"
    input_segment_records: int = 0
    input_segments: int = 0
    input_slot_reuse_wait_seconds: float = 0.0
    input_final_tail_wait_seconds: float = 0.0

    @property
    def staging_wall_seconds(self) -> float:
        return self.staging_finished_at - self.staging_started_at

    @property
    def estimated_hidden_input_seconds(self) -> float:
        return max(
            self.pageable_to_pinned_seconds
            + self.h2d_seconds
            - self.staging_wall_seconds,
            0.0,
        )


@dataclass(frozen=True)
class M1S1ComputedGroup:
    group: M1Group
    actions: tuple[M1Action, ...]
    target_retained: JaggedMigratedKVBatch | None
    target_suffix: JaggedMigratedKVBatch | None
    target_exact: JaggedMigratedKVBatch | None
    report: dict[str, object]
    lookup_metrics: dict[str, int | float]
    execution_started_at: float
    execution_finished_at: float
    ready_event: torch.cuda.Event
    slot_index: int


@dataclass(frozen=True)
class M1D3OutputTransfer:
    actions: tuple[M1Action, ...]
    segments: tuple[
        tuple[torch.Tensor, torch.Tensor],
        ...,
    ]
    started: torch.cuda.Event
    finished: torch.cuda.Event
    bytes_moved: int


@dataclass(frozen=True)
class M1RouteExecutionGranularity:
    input_segment_records: int
    compute_batch_records: int
    output_segment_records: int

    def __post_init__(self) -> None:
        if min(
            self.input_segment_records,
            self.compute_batch_records,
            self.output_segment_records,
        ) < 1:
            raise ValueError("M1 route execution granularity is invalid")

    def to_dict(self) -> dict[str, int]:
        return {
            "input_segment_records": self.input_segment_records,
            "compute_batch_records": self.compute_batch_records,
            "output_segment_records": self.output_segment_records,
        }


@dataclass(frozen=True)
class M1ExecutionGranularity:
    compiled: M1RouteExecutionGranularity
    exact: M1RouteExecutionGranularity

    def for_route(self, route: str) -> M1RouteExecutionGranularity:
        if route == "compiled":
            return self.compiled
        if route == "exact":
            return self.exact
        raise ValueError("M1 execution route differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "compiled": self.compiled.to_dict(),
            "exact": self.exact.to_dict(),
        }


@dataclass(frozen=True)
class M1PinnedSlotCapacity:
    max_rows: int
    max_source_tokens: int
    max_output_tokens: int
    max_history_tokens: int

    def __post_init__(self) -> None:
        if min(
            self.max_rows,
            self.max_source_tokens,
            self.max_output_tokens,
            self.max_history_tokens,
        ) < 1:
            raise ValueError("M1 pinned slot capacity is invalid")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_rows": self.max_rows,
            "max_source_tokens": self.max_source_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_history_tokens": self.max_history_tokens,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED_DATA)
    parser.add_argument(
        "--action-snapshot",
        default=DEFAULT_ACTION_SNAPSHOT,
    )
    parser.add_argument(
        "--training-result",
        default=DEFAULT_TRAINING_RESULT,
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument(
        "--compiler-result",
        default=DEFAULT_COMPILER_RESULT,
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "materialize", "s0", "s1", "d3"),
        default="dry-run",
    )
    parser.add_argument(
        "--scope",
        choices=("canary", "full"),
        default="canary",
    )
    parser.add_argument(
        "--canary-records-per-route-per-rank",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--group-records-per-rank",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--micro-batch-records",
        type=int,
        default=8,
    )
    parser.add_argument("--input-segment-records", type=int)
    parser.add_argument("--compute-batch-records", type=int)
    parser.add_argument("--output-segment-records", type=int)
    parser.add_argument("--compiled-input-segment-records", type=int)
    parser.add_argument("--compiled-compute-batch-records", type=int)
    parser.add_argument("--compiled-output-segment-records", type=int)
    parser.add_argument("--exact-input-segment-records", type=int)
    parser.add_argument("--exact-compute-batch-records", type=int)
    parser.add_argument("--exact-output-segment-records", type=int)
    parser.add_argument("--residency-plan")
    parser.add_argument(
        "--materialize-records-per-rank",
        type=int,
        default=8,
    )
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR)
    parser.add_argument(
        "--run-id",
        default="qk_entity_h1536_seed0",
    )
    parser.add_argument(
        "--reuse-complete-old-store",
        action="store_true",
    )
    parser.add_argument(
        "--prefault-target-store",
        action="store_true",
    )
    parser.add_argument(
        "--allow-functional-scale-full",
        action="store_true",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--expected-visible-devices", default="0,1")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def resolve_execution_granularity(
    args: argparse.Namespace,
    residency_plan: D3ResidencyPlan | None = None,
) -> M1ExecutionGranularity:
    if residency_plan is not None:
        return M1ExecutionGranularity(
            compiled=M1RouteExecutionGranularity(
                **residency_plan.compiled.to_dict()
            ),
            exact=M1RouteExecutionGranularity(
                **residency_plan.exact.to_dict()
            ),
        )
    legacy = int(args.micro_batch_records)

    def value(route: str, name: str) -> int:
        route_value = getattr(args, f"{route}_{name}")
        global_value = getattr(args, name)
        if route_value is not None:
            return int(route_value)
        if global_value is not None:
            return int(global_value)
        return legacy

    return M1ExecutionGranularity(
        compiled=M1RouteExecutionGranularity(
            input_segment_records=value(
                "compiled",
                "input_segment_records",
            ),
            compute_batch_records=value(
                "compiled",
                "compute_batch_records",
            ),
            output_segment_records=value(
                "compiled",
                "output_segment_records",
            ),
        ),
        exact=M1RouteExecutionGranularity(
            input_segment_records=value(
                "exact",
                "input_segment_records",
            ),
            compute_batch_records=value(
                "exact",
                "compute_batch_records",
            ),
            output_segment_records=value(
                "exact",
                "output_segment_records",
            ),
        ),
    )


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.canary_records_per_route_per_rank < 1
        or args.group_records_per_rank < 1
        or args.micro_batch_records < 1
        or args.materialize_records_per_rank < 1
        or args.timeout_seconds <= 0
        or not args.run_id
        or "/" in args.run_id
    ):
        raise ValueError("M1 runtime arguments are invalid")
    if args.micro_batch_records > args.group_records_per_rank:
        raise ValueError("M1 microbatch exceeds its capacity group")
    granularity = resolve_execution_granularity(args)
    route_granularities = (
        granularity.compiled,
        granularity.exact,
    )
    if max(
        value
        for route in route_granularities
        for value in route.to_dict().values()
    ) > args.group_records_per_rank:
        raise ValueError("M1 execution granularity exceeds its capacity group")
    if args.mode != "d3" and (
        any(
            value != args.micro_batch_records
            for route in route_granularities
            for value in route.to_dict().values()
        )
    ):
        raise ValueError(
            "independent M1 execution granularities require D3 mode"
        )
    explicit_granularity = any(
        getattr(args, name) is not None
        for name in (
            "input_segment_records",
            "compute_batch_records",
            "output_segment_records",
            "compiled_input_segment_records",
            "compiled_compute_batch_records",
            "compiled_output_segment_records",
            "exact_input_segment_records",
            "exact_compute_batch_records",
            "exact_output_segment_records",
        )
    )
    if args.residency_plan is not None and (
        args.mode != "d3" or explicit_granularity
    ):
        raise ValueError(
            "M1 residency plan requires D3 mode and owns its granularities"
        )
    if (
        args.mode == "materialize"
        and args.reuse_complete_old_store
    ):
        raise ValueError("materialize mode cannot reuse the old store")


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_distributed_residency_plan(
    path: str | Path | None,
    runtime,
) -> tuple[D3ResidencyPlan | None, str | None]:
    if path is None:
        return None, None
    envelope: list[object] = [None]
    if runtime.is_primary:
        try:
            resolved = _path(path)
            plan = D3ResidencyPlan.load(resolved)
            envelope[0] = {
                "plan": plan.to_dict(),
                "file_sha256": file_sha256(resolved),
            }
        except Exception as error:
            envelope[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(envelope, src=0)
    value = envelope[0]
    if not isinstance(value, Mapping):
        raise RuntimeError("M1 residency plan broadcast differs")
    if "error" in value:
        raise ValueError(f"M1 residency plan load failed: {value['error']}")
    plan_value = value.get("plan")
    if not isinstance(plan_value, Mapping):
        raise ValueError("M1 residency plan payload differs")
    plan = D3ResidencyPlan.from_dict(plan_value)
    hashes: list[object] = [None] * runtime.world_size
    dist.all_gather_object(hashes, plan.content_sha256)
    if len(set(str(item) for item in hashes)) != 1:
        raise RuntimeError("M1 residency plan differs across ranks")
    return plan, str(value["file_sha256"])


def _load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open() as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _d3_execution_identity(
    args: argparse.Namespace,
    visible_tokens: Sequence[str],
) -> dict[str, object]:
    compiler_path = _path(args.compiler_result)
    compiler = _load_json(compiler_path)
    program = compiler.get("program")
    if (
        compiler.get("status") != "complete"
        or not isinstance(program, Mapping)
        or len(str(program.get("sha256", ""))) != 64
    ):
        raise ValueError("M1 compiler identity differs")
    source_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/plan_evokv_design3_residency.py",
        ROOT / "src/hstu_kvcache/migration/design3_residency.py",
        ROOT / "src/hstu_kvcache/migration/design3_store.py",
        ROOT / "src/hstu_kvcache/migration/design2_integrated.py",
        ROOT / "src/hstu_kvcache/migration/stage45_oldkv.py",
    )
    return {
        "runner_protocol": D3_RUNNER_PROTOCOL,
        "source_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in source_paths
        },
        "compiler_result": str(compiler_path),
        "compiler_result_sha256": file_sha256(compiler_path),
        "program_path": str(_path(str(program["path"]))),
        "program_sha256": str(program["sha256"]),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "source_store_tier": "tmpfs_pageable_dram",
        "source_store_protocol": "complete_theta0_oldkv_fp16",
        "target_store_protocol": "complete_private_theta1_kv_fp16",
        "store_dir": str(_path(args.store_dir)),
        "target_prefaulted": bool(args.prefault_target_store),
        "physical_visible_devices": list(visible_tokens),
        "buffer_depth": 2,
        "input_lookahead_groups": 1,
        "output_drain_credit": 1,
    }


def load_action_snapshot(
    path: str | Path,
) -> tuple[dict[str, object], tuple[M1Action, ...]]:
    snapshot = _load_json(path)
    claimed = str(snapshot.get("owner_independent_plan_sha256", ""))
    payload = {
        key: value
        for key, value in snapshot.items()
        if key not in {"owner_independent_plan_sha256", "bindings"}
    }
    raw_records = snapshot.get("records")
    if (
        snapshot.get("status") != "development_snapshot"
        or snapshot.get("source_version") != "theta0"
        or snapshot.get("target_version") != "theta1"
        or snapshot.get("labels_used") is not False
        or snapshot.get("future_history_used") is not False
        or not isinstance(raw_records, list)
        or claimed != canonical_sha256(payload)
    ):
        raise ValueError("M1 action snapshot binding differs")
    actions = tuple(M1Action.from_dict(value) for value in raw_records)
    record_ids = tuple(value.record_id for value in actions)
    counts = snapshot.get("counts")
    if (
        not actions
        or record_ids != tuple(sorted(record_ids))
        or len(record_ids) != len(set(record_ids))
        or not isinstance(counts, dict)
        or int(counts.get("records", -1)) != len(actions)
        or int(counts.get("compiled", -1))
        != sum(value.route == "compiled" for value in actions)
        or int(counts.get("exact", -1))
        != sum(value.route == "exact" for value in actions)
    ):
        raise ValueError("M1 action snapshot records differ")
    return snapshot, actions


def build_owner_map(
    actions: Sequence[M1Action],
    world_size: int,
) -> dict[int, int]:
    if world_size < 1 or not actions:
        raise ValueError("M1 owner inputs are invalid")
    owners = {
        value.record_id: value.record_id % world_size
        for value in actions
    }
    counts = [
        sum(owner == rank for owner in owners.values())
        for rank in range(world_size)
    ]
    if max(counts) - min(counts) > 1:
        raise RuntimeError("M1 stable owner map is imbalanced")
    return owners


def select_actions(
    actions: Sequence[M1Action],
    owners: Mapping[int, int],
    world_size: int,
    scope: str,
    canary_records_per_route_per_rank: int,
) -> tuple[M1Action, ...]:
    if scope == "full":
        return tuple(actions)
    selected = []
    for route in ("compiled", "exact"):
        for rank in range(world_size):
            candidates = [
                value
                for value in actions
                if value.route == route
                and owners[value.record_id] == rank
            ]
            if len(candidates) < canary_records_per_route_per_rank:
                raise ValueError("M1 canary route stratum is too small")
            selected.extend(
                candidates[:canary_records_per_route_per_rank]
            )
    return tuple(sorted(selected, key=lambda value: value.record_id))


def build_s0_groups(
    actions: Sequence[M1Action],
    owners: Mapping[int, int],
    world_size: int,
    records_per_rank: int,
) -> tuple[M1Group, ...]:
    if (
        not actions
        or world_size < 1
        or records_per_rank < 1
    ):
        raise ValueError("M1 grouping inputs are invalid")
    groups = []
    ordinal = 0
    for route in ("compiled", "exact"):
        by_rank = [
            [
                value.record_id
                for value in actions
                if value.route == route
                and owners[value.record_id] == rank
            ]
            for rank in range(world_size)
        ]
        steps = max(
            (
                math.ceil(len(values) / records_per_rank)
                for values in by_rank
            ),
            default=0,
        )
        for step in range(steps):
            start = step * records_per_rank
            stop = start + records_per_rank
            groups.append(
                M1Group(
                    ordinal=ordinal,
                    route=route,
                    record_ids_by_rank=tuple(
                        tuple(values[start:stop])
                        for values in by_rank
                    ),
                )
            )
            ordinal += 1
    observed = [
        record_id
        for group in groups
        for values in group.record_ids_by_rank
        for record_id in values
    ]
    expected = {value.record_id for value in actions}
    if (
        len(observed) != len(actions)
        or len(observed) != len(set(observed))
        or set(observed) != expected
    ):
        raise RuntimeError("M1 group coverage differs")
    return tuple(groups)


def reorder_groups(
    groups: Sequence[M1Group],
    launch_ordinals: Sequence[int],
) -> tuple[M1Group, ...]:
    original = tuple(groups)
    original_ordinals = tuple(value.ordinal for value in original)
    requested = tuple(launch_ordinals)
    if len(original_ordinals) != len(set(original_ordinals)):
        raise ValueError("M1 original group ordinals are not unique")
    if len(requested) != len(set(requested)):
        raise ValueError("M1 launch group ordinals contain duplicates")
    if (
        len(requested) != len(original_ordinals)
        or set(requested) != set(original_ordinals)
    ):
        raise ValueError("M1 launch group ordinals do not cover the plan")
    by_ordinal = {value.ordinal: value for value in original}
    reordered = tuple(by_ordinal[value] for value in requested)
    for route in ("compiled", "exact"):
        original_route = tuple(
            value.ordinal for value in original if value.route == route
        )
        reordered_route = tuple(
            value.ordinal for value in reordered if value.route == route
        )
        if reordered_route != original_route:
            raise ValueError("M1 launch order changes a route dependency")
    return reordered


def group_plan_sha256(groups: Sequence[M1Group]) -> str:
    return canonical_sha256(
        {"groups": [value.to_dict() for value in groups]}
    )


def owner_map_sha256(owners: Mapping[int, int]) -> str:
    return canonical_sha256(
        {
            "owner_policy": "stable_record_modulo",
            "owners": [
                [record_id, owners[record_id]]
                for record_id in sorted(owners)
            ],
        }
    )


def capacity_projection(
    actions: Sequence[M1Action],
    owners: Mapping[int, int],
    world_size: int,
    cfg: HSTUConfig,
) -> dict[str, object]:
    element_size = torch.empty((), dtype=torch.float16).element_size()

    def kv_bytes(tokens: int) -> int:
        return (
            2
            * cfg.num_layers
            * tokens
            * cfg.hidden_size
            * element_size
        )

    ranks = []
    for rank in range(world_size):
        local = [
            value
            for value in actions
            if owners[value.record_id] == rank
        ]
        ranks.append(
            {
                "rank": rank,
                "records": len(local),
                "compiled": sum(
                    value.route == "compiled" for value in local
                ),
                "exact": sum(value.route == "exact" for value in local),
                "old_store_payload_bytes": sum(
                    kv_bytes(value.old_tokens) for value in local
                ),
                "target_store_payload_bytes": sum(
                    kv_bytes(value.final_tokens) for value in local
                ),
                "s0_oldkv_read_bytes": sum(
                    kv_bytes(value.retained_tokens)
                    for value in local
                    if value.route == "compiled"
                ),
                "s0_target_publish_bytes": sum(
                    kv_bytes(value.final_tokens) for value in local
                ),
            }
        )
    return {
        "dtype": "float16",
        "num_layers": cfg.num_layers,
        "kv_width": cfg.hidden_size,
        "embedding_rows": cfg.num_items + 1,
        "prediction_rows": cfg.num_prediction_items,
        "embedding_global_fp32_bytes": (
            (cfg.num_items + 1) * cfg.hidden_size * 4
        ),
        "old_store_payload_bytes": sum(
            value["old_store_payload_bytes"] for value in ranks
        ),
        "target_store_payload_bytes": sum(
            value["target_store_payload_bytes"] for value in ranks
        ),
        "combined_store_payload_bytes": sum(
            value["old_store_payload_bytes"]
            + value["target_store_payload_bytes"]
            for value in ranks
        ),
        "s0_oldkv_read_bytes": sum(
            value["s0_oldkv_read_bytes"] for value in ranks
        ),
        "s0_target_publish_bytes": sum(
            value["s0_target_publish_bytes"] for value in ranks
        ),
        "ranks": ranks,
    }


def build_dry_run_report(args: argparse.Namespace) -> dict[str, object]:
    action_path = _path(args.action_snapshot)
    training_path = _path(args.training_result)
    prepared_path = _path(args.prepared_data)
    snapshot, all_actions = load_action_snapshot(action_path)
    training = _load_json(training_path)
    if training.get("status") != "complete":
        raise ValueError("M1 training result is incomplete")
    cfg = training_model_config(training)
    if (
        str(snapshot.get("prepared_data_sha256"))
        != file_sha256(prepared_path)
    ):
        raise ValueError("M1 prepared data binding differs")
    owners = build_owner_map(all_actions, 2)
    actions = select_actions(
        all_actions,
        owners,
        2,
        args.scope,
        args.canary_records_per_route_per_rank,
    )
    groups = build_s0_groups(
        actions,
        owners,
        2,
        args.group_records_per_rank,
    )
    projection = capacity_projection(actions, owners, 2, cfg)
    execution_granularity = resolve_execution_granularity(args)
    actions_by_id = {
        value.record_id: value for value in actions
    }
    hbm_bytes = []
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        hbm_bytes = [
            torch.cuda.get_device_properties(rank).total_memory
            for rank in range(2)
        ]
    report = {
        "protocol": PROTOCOL,
        "status": "dry_run_complete",
        "scientific_result": False,
        "formal_design3": False,
        "mode": "dry-run",
        "scope": args.scope,
        "embedding_scale_role": (
            "functional_canary_not_primary_d2_partition_evidence"
            if cfg.num_items == 512144
            else "large_qk_entity_primary_m1_candidate"
        ),
        "records": len(actions),
        "counts": {
            route: sum(value.route == route for value in actions)
            for route in ("compiled", "exact")
        },
        "groups": len(groups),
        "group_records_per_rank": args.group_records_per_rank,
        "micro_batch_records": args.micro_batch_records,
        "execution_granularity": execution_granularity.to_dict(),
        "owner_policy": "stable_record_modulo",
        "owner_map_sha256": owner_map_sha256(
            {
                value.record_id: owners[value.record_id]
                for value in actions
            }
        ),
        "group_plan_sha256": group_plan_sha256(groups),
        "capacity": {
            **projection,
            "capacity_groups": capacity_group_projection(
                groups,
                actions_by_id,
                cfg,
            ),
            "visible_hbm_bytes": hbm_bytes,
            "combined_store_to_visible_hbm_ratio": (
                projection["combined_store_payload_bytes"]
                / sum(hbm_bytes)
                if hbm_bytes
                else None
            ),
        },
        "bindings": {
            "prepared_data": str(prepared_path),
            "prepared_data_sha256": file_sha256(prepared_path),
            "action_snapshot": str(action_path),
            "action_snapshot_sha256": file_sha256(action_path),
            "owner_independent_plan_sha256": snapshot[
                "owner_independent_plan_sha256"
            ],
            "training_result": str(training_path),
            "training_result_sha256": file_sha256(training_path),
        },
        "next": (
            "materialize complete theta0 K/V or run the two-rank S0 "
            "canary without freezing a D3 policy"
        ),
    }
    return report


def _copy_history(
    value: Mapping[str, np.ndarray],
    start: int,
    stop: int,
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value[name][start:stop]).copy()
        for name in (
            "item_ids",
            "behaviors",
            "time_deltas",
            "timestamps",
        )
    }


def load_histories(
    prepared_path: Path,
    actions: Sequence[M1Action],
    snapshot: Mapping[str, object],
) -> M1Histories:
    layout = snapshot.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("M1 adjacent layout is missing")
    history_tokens = int(layout["history_tokens"])
    slide = int(layout["target_filtered_start"])
    plan, metadata = load_prepared_exposure_plan(
        prepared_path,
        max_seq_len=history_tokens,
    )
    if (
        str(metadata.get("dataset", "")).lower().replace("_", "-")
        != "tenrec-qk"
        or plan.base_dates != ["base"]
        or plan.stream_dates[:1] != ["window_0"]
    ):
        raise ValueError("M1 QK prepared stream differs")
    plan.init_base()
    old = {}
    for action in actions:
        history = plan.user_histories.get(action.prepared_user_id)
        if history is None or len(history["item_ids"]) != action.old_tokens:
            raise ValueError("M1 old history extent differs")
        old[action.record_id] = _copy_history(
            history,
            0,
            action.old_tokens,
        )
    plan.ingest_day("window_0")
    target = {}
    for action in actions:
        history = plan.user_histories.get(action.prepared_user_id)
        if (
            history is None
            or len(history["item_ids"]) < slide + action.final_tokens
        ):
            raise ValueError("M1 target history extent differs")
        target[action.record_id] = _copy_history(
            history,
            slide,
            slide + action.final_tokens,
        )
        for name in (
            "item_ids",
            "behaviors",
            "time_deltas",
            "timestamps",
        ):
            if not np.array_equal(
                old[action.record_id][name][
                    action.retained_start : action.old_tokens
                ],
                target[action.record_id][name][
                    : action.retained_tokens
                ],
            ):
                raise ValueError("M1 retained history identity differs")
    return M1Histories(old=old, target=target)


def _execution_slot_capacity(
    granularity: M1ExecutionGranularity,
    materialize_records_per_rank: int,
    max_old_tokens: int,
    max_final_tokens: int,
    max_retained_tokens: int,
) -> M1PinnedSlotCapacity:
    return M1PinnedSlotCapacity(
        max_rows=max(
            granularity.compiled.input_segment_records,
            granularity.exact.input_segment_records,
            materialize_records_per_rank,
        ),
        max_source_tokens=max(
            granularity.compiled.input_segment_records
            * max_retained_tokens,
            1,
        ),
        max_output_tokens=max(
            max(
                granularity.compiled.output_segment_records,
                granularity.exact.output_segment_records,
            )
            * max_final_tokens,
            materialize_records_per_rank * max_old_tokens,
            1,
        ),
        max_history_tokens=max(max_final_tokens, 1),
    )


def _pinned_slot_nbytes(
    cfg: HSTUConfig,
    capacity: M1PinnedSlotCapacity,
) -> int:
    source = (
        2
        * cfg.num_layers
        * capacity.max_source_tokens
        * cfg.hidden_size
        * torch.float16.itemsize
    )
    output = (
        2
        * cfg.num_layers
        * capacity.max_output_tokens
        * cfg.hidden_size
        * torch.float16.itemsize
    )
    rows = capacity.max_rows
    history = capacity.max_history_tokens
    metadata = (
        rows * torch.int64.itemsize
        + (rows + 1) * torch.int64.itemsize
        + rows * history * torch.int64.itemsize
        + rows * history * torch.int64.itemsize
        + rows * history * torch.float32.itemsize
        + rows * torch.int64.itemsize
    )
    return source + output + metadata


def _allocate_slot(
    cfg: HSTUConfig,
    max_rows: int,
    max_source_tokens: int,
    max_output_tokens: int,
    max_history_tokens: int,
) -> M1PinnedSlot:
    return M1PinnedSlot(
        source_k=torch.empty(
            (
                cfg.num_layers,
                max(max_source_tokens, 1),
                cfg.hidden_size,
            ),
            dtype=torch.float16,
            pin_memory=True,
        ),
        source_v=torch.empty(
            (
                cfg.num_layers,
                max(max_source_tokens, 1),
                cfg.hidden_size,
            ),
            dtype=torch.float16,
            pin_memory=True,
        ),
        source_lengths=torch.empty(
            max(max_rows, 1),
            dtype=torch.long,
            pin_memory=True,
        ),
        source_offsets=torch.empty(
            max(max_rows + 1, 1),
            dtype=torch.long,
            pin_memory=True,
        ),
        history_item_ids=torch.empty(
            (max(max_rows, 1), max(max_history_tokens, 1)),
            dtype=torch.long,
            pin_memory=True,
        ),
        history_behaviors=torch.empty(
            (max(max_rows, 1), max(max_history_tokens, 1)),
            dtype=torch.long,
            pin_memory=True,
        ),
        history_time_deltas=torch.empty(
            (max(max_rows, 1), max(max_history_tokens, 1)),
            dtype=torch.float32,
            pin_memory=True,
        ),
        history_lengths=torch.empty(
            max(max_rows, 1),
            dtype=torch.long,
            pin_memory=True,
        ),
        output_k=torch.empty(
            (
                cfg.num_layers,
                max(max_output_tokens, 1),
                cfg.hidden_size,
            ),
            dtype=torch.float16,
            pin_memory=True,
        ),
        output_v=torch.empty(
            (
                cfg.num_layers,
                max(max_output_tokens, 1),
                cfg.hidden_size,
            ),
            dtype=torch.float16,
            pin_memory=True,
        ),
    )


def _history_batch_into_slot(
    slot: M1PinnedSlot,
    actions: Sequence[M1Action],
    histories: Mapping[int, Mapping[str, np.ndarray]],
    starts: Sequence[int],
    stops: Sequence[int],
    version: str,
) -> RawHistoryBatch:
    if (
        len(actions) != len(starts)
        or len(actions) != len(stops)
    ):
        raise ValueError("M1 history batch ranges differ")
    lengths = tuple(
        int(stop) - int(start)
        for start, stop in zip(starts, stops, strict=True)
    )
    width = max(max(lengths, default=0), 1)
    rows = len(actions)
    item_ids = slot.history_item_ids[:rows, :width]
    behaviors = slot.history_behaviors[:rows, :width]
    time_deltas = slot.history_time_deltas[:rows, :width]
    prepared_lengths = slot.history_lengths[:rows]
    if rows:
        item_ids.zero_()
        behaviors.zero_()
        time_deltas.zero_()
    for row, (action, start, stop, length) in enumerate(
        zip(actions, starts, stops, lengths, strict=True)
    ):
        history = histories[action.record_id]
        if start < 0 or stop <= start or stop > len(history["item_ids"]):
            raise ValueError("M1 history range exceeds its record")
        item_ids[row, :length].copy_(
            torch.from_numpy(history["item_ids"][start:stop])
        )
        behaviors[row, :length].copy_(
            torch.from_numpy(history["behaviors"][start:stop])
        )
        time_deltas[row, :length].copy_(
            torch.from_numpy(history["time_deltas"][start:stop])
        )
    prepared_lengths.copy_(torch.tensor(lengths, dtype=torch.long))
    return RawHistoryBatch(
        record_ids=tuple(value.record_id for value in actions),
        migration_anchor_version=version,
        item_ids=item_ids,
        behaviors=behaviors,
        time_deltas=time_deltas,
        lengths=prepared_lengths,
    )


def _source_batch_from_store(
    slot: M1PinnedSlot,
    store: PageableDramExtentStore,
    actions: Sequence[M1Action],
    source_version: str,
) -> tuple[JaggedMigratedKVBatch | None, int]:
    if not actions:
        return None, 0
    tokens = sum(value.retained_tokens for value in actions)
    k = _packed_slot_kv_view(slot.source_k, tokens)
    v = _packed_slot_kv_view(slot.source_v, tokens)
    record_ids = tuple(value.record_id for value in actions)
    starts = tuple(value.retained_start for value in actions)
    stops = tuple(value.old_tokens for value in actions)
    read_bytes = store.read_ranges_into(
        record_ids,
        starts,
        stops,
        k,
        v,
    )
    lengths = slot.source_lengths[: len(actions)]
    offsets = slot.source_offsets[: len(actions) + 1]
    lengths.copy_(
        torch.tensor(
            [value.retained_tokens for value in actions],
            dtype=torch.long,
        )
    )
    offsets[0] = 0
    torch.cumsum(lengths, dim=0, out=offsets[1:])
    return (
        JaggedMigratedKVBatch(
            record_ids=record_ids,
            migration_anchor_version=source_version,
            served_kv_target=source_version,
            k=k,
            v=v,
            lengths=lengths,
            offsets=offsets,
        ),
        read_bytes,
    )


def _device_destination(
    source: JaggedMigratedKVBatch,
    target_version: str,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=source.record_ids,
        migration_anchor_version=source.migration_anchor_version,
        served_kv_target=target_version,
        k=torch.empty_like(source.k),
        v=torch.empty_like(source.v),
        lengths=source.lengths.clone(),
        offsets=source.offsets.clone(),
    )


def _copy_fragment_to_output(
    slot: M1PinnedSlot,
    fragment: JaggedMigratedKVBatch,
    token_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    return _copy_values_to_output(
        slot,
        fragment.k,
        fragment.v,
        token_offset,
    )


def _copy_values_to_output(
    slot: M1PinnedSlot,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    token_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if (
        source_k.ndim != 3
        or source_k.shape != source_v.shape
        or source_k.dtype != torch.float16
        or source_v.dtype != torch.float16
    ):
        raise ValueError("M1 output K/V values differ")
    stop = token_offset + source_k.shape[1]
    k = _packed_slot_kv_view(
        slot.output_k,
        source_k.shape[1],
        token_offset,
    )
    v = _packed_slot_kv_view(
        slot.output_v,
        source_v.shape[1],
        token_offset,
    )
    k.copy_(source_k, non_blocking=True)
    v.copy_(source_v, non_blocking=True)
    return k, v, stop


def _packed_slot_kv_view(
    value: torch.Tensor,
    tokens: int,
    token_offset: int = 0,
) -> torch.Tensor:
    if (
        value.ndim != 3
        or tokens < 1
        or token_offset < 0
    ):
        raise ValueError("M1 pinned K/V view is invalid")
    layers, _, width = value.shape
    element_start = token_offset * layers * width
    element_stop = element_start + tokens * layers * width
    flattened = value.view(-1)
    if element_stop > flattened.numel():
        raise ValueError("M1 pinned K/V view exceeds its slot")
    return flattened[element_start:element_stop].view(
        layers,
        tokens,
        width,
    )


def publish_output_segments(
    store: PageableDramExtentStore,
    actions: Sequence[M1Action],
    route: str,
    segments: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> int:
    if not actions:
        if segments:
            raise ValueError("empty M1 publication has segments")
        return 0
    record_ids = tuple(value.record_id for value in actions)
    if route == "exact" and len(segments) == 1:
        starts = (0,) * len(actions)
        stops = tuple(value.final_tokens for value in actions)
        return store.write_ranges(
            record_ids,
            starts,
            stops,
            segments[0][0],
            segments[0][1],
        )
    if route == "compiled" and len(segments) == 2:
        retained = store.write_ranges(
            record_ids,
            (0,) * len(actions),
            tuple(value.retained_tokens for value in actions),
            segments[0][0],
            segments[0][1],
        )
        suffix = store.write_ranges(
            record_ids,
            tuple(value.retained_tokens for value in actions),
            tuple(value.final_tokens for value in actions),
            segments[1][0],
            segments[1][1],
        )
        return retained + suffix
    raise ValueError("M1 publication route and segments differ")


def _finite_sample(
    segments: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> bool:
    for k, v in segments:
        for value in (k, v):
            flat = value.flatten()
            if flat.numel() and not bool(
                torch.isfinite(
                    torch.cat(
                        (
                            flat[: min(flat.numel(), 256)],
                            flat[max(flat.numel() - 256, 0) :],
                        )
                    )
                ).all()
            ):
                return False
    return True


def _event_seconds(
    started: torch.cuda.Event,
    finished: torch.cuda.Event,
) -> float:
    finished.synchronize()
    return started.elapsed_time(finished) / 1000.0


def _actions_for_ids(
    record_ids: Sequence[int],
    actions_by_id: Mapping[int, M1Action],
) -> tuple[M1Action, ...]:
    return tuple(actions_by_id[value] for value in record_ids)


def _chunks(
    values: Sequence[M1Action],
    size: int,
) -> tuple[tuple[M1Action, ...], ...]:
    return tuple(
        tuple(values[start : start + size])
        for start in range(0, len(values), size)
    )


def _allocate_device_history_group(
    actions: Sequence[M1Action],
    route: str,
    device: torch.device,
    stream: torch.cuda.Stream,
) -> RawHistoryBatch:
    widths = tuple(
        (
            value.final_tokens - value.delta_start
            if route == "compiled"
            else value.final_tokens
        )
        for value in actions
    )
    width = max(max(widths, default=0), 1)
    rows = len(actions)
    with torch.cuda.stream(stream):
        item_ids = torch.zeros(
            (rows, width),
            dtype=torch.long,
            device=device,
        )
        behaviors = torch.zeros_like(item_ids)
        time_deltas = torch.zeros(
            (rows, width),
            dtype=torch.float32,
            device=device,
        )
        lengths = torch.empty(
            rows,
            dtype=torch.long,
            device=device,
        )
    return RawHistoryBatch(
        record_ids=tuple(value.record_id for value in actions),
        migration_anchor_version="theta1",
        item_ids=item_ids,
        behaviors=behaviors,
        time_deltas=time_deltas,
        lengths=lengths,
    )


def _device_history_views(
    history: RawHistoryBatch,
    local_batches: Sequence[Sequence[M1Action]],
    steps: int,
) -> tuple[RawHistoryBatch, ...]:
    flattened_ids = tuple(
        value.record_id
        for batch in local_batches
        for value in batch
    )
    if flattened_ids != history.record_ids or steps < len(local_batches):
        raise ValueError("M1 compute history partition differs")
    output = []
    row_offset = 0
    for step in range(steps):
        actions = (
            local_batches[step]
            if step < len(local_batches)
            else ()
        )
        row_stop = row_offset + len(actions)
        record_ids = tuple(value.record_id for value in actions)
        if history.record_ids[row_offset:row_stop] != record_ids:
            raise RuntimeError("M1 compute history order differs")
        output.append(
            RawHistoryBatch(
                record_ids=record_ids,
                migration_anchor_version=(
                    history.migration_anchor_version
                ),
                item_ids=history.item_ids[row_offset:row_stop],
                behaviors=history.behaviors[row_offset:row_stop],
                time_deltas=history.time_deltas[
                    row_offset:row_stop
                ],
                lengths=history.lengths[row_offset:row_stop],
            )
        )
        row_offset = row_stop
    if row_offset != history.batch_size:
        raise RuntimeError("M1 compute history coverage differs")
    return tuple(output)


def _device_jagged(
    actions: Sequence[M1Action],
    k: torch.Tensor,
    v: torch.Tensor,
    length_name: str,
    migration_anchor_version: str,
    served_kv_target: str,
) -> JaggedMigratedKVBatch:
    lengths = torch.tensor(
        [int(getattr(value, length_name)) for value in actions],
        dtype=torch.long,
        device=k.device,
    )
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=k.device),
            lengths.cumsum(0),
        )
    )
    return JaggedMigratedKVBatch(
        record_ids=tuple(value.record_id for value in actions),
        migration_anchor_version=migration_anchor_version,
        served_kv_target=served_kv_target,
        k=k,
        v=v,
        lengths=lengths,
        offsets=offsets,
    )


def _tensor_pair_nbytes(
    k: torch.Tensor,
    v: torch.Tensor,
) -> int:
    return (
        k.numel() * k.element_size()
        + v.numel() * v.element_size()
    )


def group_resident_extent_bytes(
    actions: Sequence[M1Action],
    cfg: HSTUConfig,
) -> int:
    element_size = torch.empty((), dtype=torch.float16).element_size()
    return (
        2
        * cfg.num_layers
        * cfg.hidden_size
        * element_size
        * sum(
            value.final_tokens
            + (
                value.retained_tokens
                if value.route == "compiled"
                else 0
            )
            for value in actions
        )
    )


def capacity_group_projection(
    groups: Sequence[M1Group],
    actions_by_id: Mapping[int, M1Action],
    cfg: HSTUConfig,
) -> dict[str, object]:
    values = [
        {
            "ordinal": group.ordinal,
            "route": group.route,
            "records_by_rank": [
                len(record_ids)
                for record_ids in group.record_ids_by_rank
            ],
            "resident_extent_bytes_by_rank": [
                group_resident_extent_bytes(
                    _actions_for_ids(record_ids, actions_by_id),
                    cfg,
                )
                for record_ids in group.record_ids_by_rank
            ],
        }
        for group in groups
    ]
    return {
        "groups": values,
        "maximum_resident_extent_bytes": max(
            (
                value
                for group in values
                for value in group["resident_extent_bytes_by_rank"]
            ),
            default=0,
        ),
    }


def _aligned_batches(
    record_ids_by_rank: Sequence[Sequence[int]],
    records_per_rank: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    steps = max(
        math.ceil(len(values) / records_per_rank)
        for values in record_ids_by_rank
    )
    output = []
    for step in range(steps):
        start = step * records_per_rank
        stop = start + records_per_rank
        output.append(
            tuple(
                tuple(values[start:stop])
                for values in record_ids_by_rank
            )
        )
    return tuple(output)


def _sum_lookup_metrics(
    totals: dict[str, int | float],
    metrics,
) -> None:
    values = metrics.to_dict()
    for name in (
        "requested_tokens",
        "local_requested_tokens",
        "remote_requested_tokens",
        "served_remote_requested_tokens",
        "actual_collective_tensor_payload_bytes",
        "off_diagonal_bytes",
        "collective_calls",
    ):
        totals[name] = int(totals.get(name, 0)) + int(values[name])
    totals["collective_seconds"] = float(
        totals.get("collective_seconds", 0.0)
    ) + float(values["collective_seconds"])


def _merge_lookup_metrics(
    totals: dict[str, int | float],
    values: Mapping[str, int | float],
) -> None:
    for name, value in values.items():
        if isinstance(value, float):
            totals[name] = float(totals.get(name, 0.0)) + value
        else:
            totals[name] = int(totals.get(name, 0)) + value


def materialize_old_store(
    store: PageableDramExtentStore,
    batches: Sequence[Sequence[Sequence[int]]],
    rank: int,
    actions_by_id: Mapping[int, M1Action],
    histories: M1Histories,
    model,
    slot: M1PinnedSlot,
    device: torch.device,
) -> dict[str, object]:
    phases = {
        "pageable_to_pinned_seconds": 0.0,
        "h2d_seconds": 0.0,
        "lookup_and_compute_seconds": 0.0,
        "d2h_seconds": 0.0,
        "pinned_to_pageable_seconds": 0.0,
    }
    lookup_totals: dict[str, int | float] = {}
    written_bytes = 0
    observed = []
    started = time.perf_counter()
    for record_ids_by_rank in batches:
        actions = _actions_for_ids(
            record_ids_by_rank[rank],
            actions_by_id,
        )
        stage_started = time.perf_counter()
        history = _history_batch_into_slot(
            slot,
            actions,
            histories.old,
            (0,) * len(actions),
            tuple(value.old_tokens for value in actions),
            "theta0",
        )
        phases["pageable_to_pinned_seconds"] += (
            time.perf_counter() - stage_started
        )
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        h2d_start.record()
        device_history = history.to(device, non_blocking=True)
        h2d_end.record()
        phases["h2d_seconds"] += _event_seconds(h2d_start, h2d_end)
        compute_started = time.perf_counter()
        result = integrated_sharded_exact(
            model,
            device_history,
            "theta0",
        )
        torch.cuda.synchronize(device)
        phases["lookup_and_compute_seconds"] += (
            time.perf_counter() - compute_started
        )
        _sum_lookup_metrics(lookup_totals, result.lookup_metrics)
        segments = []
        if result.fragment is not None:
            d2h_start = torch.cuda.Event(enable_timing=True)
            d2h_end = torch.cuda.Event(enable_timing=True)
            d2h_start.record()
            k, v, _ = _copy_fragment_to_output(
                slot,
                result.fragment,
                0,
            )
            d2h_end.record()
            phases["d2h_seconds"] += _event_seconds(
                d2h_start,
                d2h_end,
            )
            segments = [(k, v)]
            if not _finite_sample(segments):
                raise RuntimeError("M1 materialized old K/V is nonfinite")
            publish_started = time.perf_counter()
            written_bytes += publish_output_segments(
                store,
                actions,
                "exact",
                segments,
            )
            phases["pinned_to_pageable_seconds"] += (
                time.perf_counter() - publish_started
            )
            observed.extend(value.record_id for value in actions)
        elif actions:
            raise RuntimeError("M1 old materialization lost local output")
        del device_history, result, segments
    torch.cuda.synchronize(device)
    return {
        "wall_seconds": time.perf_counter() - started,
        "phase_seconds": phases,
        "lookup_metrics": lookup_totals,
        "records": len(observed),
        "record_ids_sha256": canonical_sha256(
            {"record_ids": sorted(observed)}
        ),
        "written_bytes": written_bytes,
        "exactly_once_pass": (
            len(observed) == len(set(observed))
            and set(observed) == set(store.record_ids)
        ),
    }


def run_s0(
    groups: Sequence[M1Group],
    rank: int,
    actions_by_id: Mapping[int, M1Action],
    histories: M1Histories,
    old_store: PageableDramExtentStore,
    target_store: PageableDramExtentStore,
    model,
    operator: DirectOldKVFusedOperator,
    program,
    slot: M1PinnedSlot,
    device: torch.device,
    micro_batch_records: int,
) -> dict[str, object]:
    phases = {
        "pageable_to_pinned_seconds": 0.0,
        "h2d_seconds": 0.0,
        "d2d_transform_seconds": 0.0,
        "d2d_assemble_seconds": 0.0,
        "lookup_exchange_seconds": 0.0,
        "compute_excluding_lookup_seconds": 0.0,
        "d2h_seconds": 0.0,
        "publish_seconds": 0.0,
    }
    lookup_totals: dict[str, int | float] = {}
    group_reports = []
    observed = []
    h2d_bytes = 0
    d2h_bytes = 0
    published_bytes = 0
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    wall_started = time.perf_counter()
    for group in groups:
        group_actions = _actions_for_ids(
            group.record_ids_by_rank[rank],
            actions_by_id,
        )
        microbatches_by_rank = tuple(
            _chunks(
                _actions_for_ids(
                    group.record_ids_by_rank[value],
                    actions_by_id,
                ),
                micro_batch_records,
            )
            for value in range(len(group.record_ids_by_rank))
        )
        micro_steps = max(
            (len(value) for value in microbatches_by_rank),
            default=0,
        )
        local_microbatches = microbatches_by_rank[rank]
        oldkv_read_bytes = 0
        group_h2d_bytes = 0
        group_d2h_bytes = 0
        group_published_bytes = 0
        group_lookup: dict[str, int | float] = {}
        group_stage_seconds = 0.0
        group_h2d_seconds = 0.0
        group_d2d_seconds = 0.0
        group_assemble_seconds = 0.0
        group_lookup_seconds = 0.0
        group_compute_seconds = 0.0
        group_d2h_seconds = 0.0
        group_publish_seconds = 0.0
        source_group = None
        target_retained = None
        target_suffix = None
        target_exact = None
        if group.route == "compiled" and group_actions:
            retained_tokens = sum(
                value.retained_tokens for value in group_actions
            )
            source_k = torch.empty(
                (
                    program.num_layers,
                    retained_tokens,
                    program.kv_width,
                ),
                dtype=torch.float16,
                device=device,
            )
            source_v = torch.empty_like(source_k)
            target_retained_k = torch.empty_like(source_k)
            target_retained_v = torch.empty_like(source_v)
            suffix_tokens = sum(
                value.suffix_tokens for value in group_actions
            )
            target_suffix_k = torch.empty(
                (
                    program.num_layers,
                    suffix_tokens,
                    program.kv_width,
                ),
                dtype=torch.float16,
                device=device,
            )
            target_suffix_v = torch.empty_like(target_suffix_k)
            source_offset = 0
            for actions in local_microbatches:
                stage_started = time.perf_counter()
                source, read_bytes = _source_batch_from_store(
                    slot,
                    old_store,
                    actions,
                    "theta0",
                )
                stage_seconds = time.perf_counter() - stage_started
                group_stage_seconds += stage_seconds
                phases["pageable_to_pinned_seconds"] += stage_seconds
                if source is None:
                    raise RuntimeError("M1 compiled group source is missing")
                stop = source_offset + source.token_count
                h2d_start = torch.cuda.Event(enable_timing=True)
                h2d_end = torch.cuda.Event(enable_timing=True)
                h2d_start.record()
                source_k[:, source_offset:stop].copy_(
                    source.k,
                    non_blocking=True,
                )
                source_v[:, source_offset:stop].copy_(
                    source.v,
                    non_blocking=True,
                )
                h2d_end.record()
                seconds = _event_seconds(h2d_start, h2d_end)
                group_h2d_seconds += seconds
                phases["h2d_seconds"] += seconds
                bytes_moved = _tensor_pair_nbytes(source.k, source.v)
                group_h2d_bytes += bytes_moved
                h2d_bytes += bytes_moved
                oldkv_read_bytes += read_bytes
                source_offset = stop
            source_group = _device_jagged(
                group_actions,
                source_k,
                source_v,
                "retained_tokens",
                "theta0",
                "theta0",
            )
            target_retained = _device_jagged(
                group_actions,
                target_retained_k,
                target_retained_v,
                "retained_tokens",
                "theta0",
                "theta1",
            )
            target_suffix = _device_jagged(
                group_actions,
                target_suffix_k,
                target_suffix_v,
                "suffix_tokens",
                "theta1",
                "theta1",
            )
            del (
                source_k,
                source_v,
                target_retained_k,
                target_retained_v,
                target_suffix_k,
                target_suffix_v,
            )
            d2d_start = torch.cuda.Event(enable_timing=True)
            d2d_end = torch.cuda.Event(enable_timing=True)
            d2d_start.record()
            operator.execute_into(
                program,
                source_group,
                target_retained,
            )
            d2d_end.record()
            group_d2d_seconds = _event_seconds(d2d_start, d2d_end)
            phases["d2d_transform_seconds"] += group_d2d_seconds
        elif group.route == "exact" and group_actions:
            exact_tokens = sum(
                value.final_tokens for value in group_actions
            )
            exact_k = torch.empty(
                (
                    model.dense_model.cfg.num_layers,
                    exact_tokens,
                    model.dense_model.cfg.hidden_size,
                ),
                dtype=torch.float16,
                device=device,
            )
            exact_v = torch.empty_like(exact_k)
            target_exact = _device_jagged(
                group_actions,
                exact_k,
                exact_v,
                "final_tokens",
                "theta1",
                "theta1",
            )
            del exact_k, exact_v
        resident_allocated = torch.cuda.memory_allocated(device)
        retained_offset = 0
        suffix_offset = 0
        exact_offset = 0
        for step in range(micro_steps):
            actions = (
                local_microbatches[step]
                if step < len(local_microbatches)
                else ()
            )
            if group.route == "compiled":
                starts = tuple(value.delta_start for value in actions)
            else:
                starts = (0,) * len(actions)
            stage_started = time.perf_counter()
            history = _history_batch_into_slot(
                slot,
                actions,
                histories.target,
                starts,
                tuple(value.final_tokens for value in actions),
                "theta1",
            )
            stage_seconds = time.perf_counter() - stage_started
            group_stage_seconds += stage_seconds
            phases["pageable_to_pinned_seconds"] += stage_seconds
            history_bytes = history.nbytes
            group_h2d_bytes += history_bytes
            h2d_bytes += history_bytes
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            h2d_start.record()
            device_history = history.to(device, non_blocking=True)
            h2d_end.record()
            seconds = _event_seconds(h2d_start, h2d_end)
            group_h2d_seconds += seconds
            phases["h2d_seconds"] += seconds
            transformed = None
            if actions and group.route == "compiled":
                retained_stop = retained_offset + sum(
                    value.retained_tokens for value in actions
                )
                assert target_retained is not None
                gather_start = torch.cuda.Event(enable_timing=True)
                gather_end = torch.cuda.Event(enable_timing=True)
                gather_start.record()
                transformed_k = target_retained.k[
                    :, retained_offset:retained_stop
                ].contiguous()
                transformed_v = target_retained.v[
                    :, retained_offset:retained_stop
                ].contiguous()
                transformed = _device_jagged(
                    actions,
                    transformed_k,
                    transformed_v,
                    "retained_tokens",
                    "theta0",
                    "theta1",
                )
                gather_end.record()
                seconds = _event_seconds(gather_start, gather_end)
                group_assemble_seconds += seconds
                phases["d2d_assemble_seconds"] += seconds
                del transformed_k, transformed_v
            compute_started = time.perf_counter()
            if group.route == "compiled":
                result = integrated_sharded_append_only(
                    model,
                    transformed,
                    device_history,
                    "theta1",
                )
            else:
                result = integrated_sharded_exact(
                    model,
                    device_history,
                    "theta1",
                )
            torch.cuda.synchronize(device)
            compute_and_lookup_seconds = (
                time.perf_counter() - compute_started
            )
            lookup_seconds = result.lookup_metrics.collective_seconds
            compute_seconds = max(
                compute_and_lookup_seconds - lookup_seconds,
                0.0,
            )
            group_lookup_seconds += lookup_seconds
            group_compute_seconds += compute_seconds
            phases["lookup_exchange_seconds"] += lookup_seconds
            phases["compute_excluding_lookup_seconds"] += (
                compute_seconds
            )
            _sum_lookup_metrics(group_lookup, result.lookup_metrics)
            _sum_lookup_metrics(lookup_totals, result.lookup_metrics)
            if result.fragment is not None:
                assemble_start = torch.cuda.Event(enable_timing=True)
                assemble_end = torch.cuda.Event(enable_timing=True)
                assemble_start.record()
                if isinstance(
                    result.fragment,
                    IntegratedAppendOnlyKVBatch,
                ):
                    suffix_stop = suffix_offset + (
                        result.fragment.suffix.token_count
                    )
                    assert target_suffix is not None
                    target_suffix.k[:, suffix_offset:suffix_stop].copy_(
                        result.fragment.suffix.k
                    )
                    target_suffix.v[:, suffix_offset:suffix_stop].copy_(
                        result.fragment.suffix.v
                    )
                    suffix_offset = suffix_stop
                    retained_offset += (
                        result.fragment.retained.token_count
                    )
                else:
                    exact_stop = (
                        exact_offset + result.fragment.token_count
                    )
                    assert target_exact is not None
                    target_exact.k[:, exact_offset:exact_stop].copy_(
                        result.fragment.k
                    )
                    target_exact.v[:, exact_offset:exact_stop].copy_(
                        result.fragment.v
                    )
                    exact_offset = exact_stop
                assemble_end.record()
                seconds = _event_seconds(
                    assemble_start,
                    assemble_end,
                )
                group_assemble_seconds += seconds
                phases["d2d_assemble_seconds"] += seconds
            elif actions:
                raise RuntimeError("M1 S0 lost local microbatch output")
            del device_history, result, transformed
        if group.route == "compiled" and group_actions:
            assert target_retained is not None
            assert target_suffix is not None
            if (
                retained_offset != target_retained.token_count
                or suffix_offset != target_suffix.token_count
            ):
                raise RuntimeError("M1 compiled group assembly differs")
        if group.route == "exact" and group_actions:
            assert target_exact is not None
            if exact_offset != target_exact.token_count:
                raise RuntimeError("M1 exact group assembly differs")
        drain_microbatches = _chunks(
            group_actions,
            micro_batch_records,
        )
        retained_offset = 0
        suffix_offset = 0
        exact_offset = 0
        for actions in drain_microbatches:
            segments: list[tuple[torch.Tensor, torch.Tensor]] = []
            d2h_start = torch.cuda.Event(enable_timing=True)
            d2h_end = torch.cuda.Event(enable_timing=True)
            d2h_start.record()
            token_offset = 0
            if group.route == "compiled":
                retained_stop = retained_offset + sum(
                    value.retained_tokens for value in actions
                )
                suffix_stop = suffix_offset + sum(
                    value.suffix_tokens for value in actions
                )
                assert target_retained is not None
                assert target_suffix is not None
                retained_k, retained_v, token_offset = (
                    _copy_values_to_output(
                        slot,
                        target_retained.k[
                            :, retained_offset:retained_stop
                        ],
                        target_retained.v[
                            :, retained_offset:retained_stop
                        ],
                        token_offset,
                    )
                )
                suffix_k, suffix_v, token_offset = (
                    _copy_values_to_output(
                        slot,
                        target_suffix.k[
                            :, suffix_offset:suffix_stop
                        ],
                        target_suffix.v[
                            :, suffix_offset:suffix_stop
                        ],
                        token_offset,
                    )
                )
                segments = [
                    (retained_k, retained_v),
                    (suffix_k, suffix_v),
                ]
                retained_offset = retained_stop
                suffix_offset = suffix_stop
            else:
                exact_stop = exact_offset + sum(
                    value.final_tokens for value in actions
                )
                assert target_exact is not None
                k, v, token_offset = _copy_values_to_output(
                    slot,
                    target_exact.k[:, exact_offset:exact_stop],
                    target_exact.v[:, exact_offset:exact_stop],
                    token_offset,
                )
                segments = [(k, v)]
                exact_offset = exact_stop
            d2h_end.record()
            seconds = _event_seconds(d2h_start, d2h_end)
            group_d2h_seconds += seconds
            phases["d2h_seconds"] += seconds
            bytes_moved = sum(
                _tensor_pair_nbytes(k, v) for k, v in segments
            )
            group_d2h_bytes += bytes_moved
            d2h_bytes += bytes_moved
            if not _finite_sample(segments):
                raise RuntimeError("M1 S0 output K/V is nonfinite")
            publish_started = time.perf_counter()
            bytes_written = publish_output_segments(
                target_store,
                actions,
                group.route,
                segments,
            )
            seconds = time.perf_counter() - publish_started
            group_publish_seconds += seconds
            phases["publish_seconds"] += seconds
            group_published_bytes += bytes_written
            published_bytes += bytes_written
            observed.extend(value.record_id for value in actions)
            del segments
        expected_resident = group_resident_extent_bytes(
            group_actions,
            model.dense_model.cfg,
        )
        measured_resident = max(
            resident_allocated - baseline_allocated,
            0,
        )
        group_reports.append(
            {
                "ordinal": group.ordinal,
                "route": group.route,
                "capacity_group_records": len(group_actions),
                "compute_microbatches": micro_steps,
                "micro_batch_records": micro_batch_records,
                "resident_extent_logical_bytes": expected_resident,
                "resident_extent_allocated_bytes": measured_resident,
                "oldkv_read_bytes": oldkv_read_bytes,
                "h2d_bytes": group_h2d_bytes,
                "d2h_bytes": group_d2h_bytes,
                "published_bytes": group_published_bytes,
                "pageable_to_pinned_seconds": group_stage_seconds,
                "h2d_seconds": group_h2d_seconds,
                "d2d_transform_seconds": group_d2d_seconds,
                "d2d_assemble_seconds": group_assemble_seconds,
                "lookup_exchange_seconds": group_lookup_seconds,
                "compute_excluding_lookup_seconds": (
                    group_compute_seconds
                ),
                "d2h_seconds": group_d2h_seconds,
                "publish_seconds": group_publish_seconds,
                "lookup_metrics": group_lookup,
            }
        )
        del (
            source_group,
            target_retained,
            target_suffix,
            target_exact,
        )
        torch.cuda.synchronize(device)
    torch.cuda.synchronize(device)
    target_ledger = target_store.ledger()
    expected_ids = set(target_store.record_ids)
    return {
        "wall_seconds": time.perf_counter() - wall_started,
        "phase_seconds": phases,
        "primary_phase_sum_seconds": sum(phases.values()),
        "lookup_metrics": lookup_totals,
        "records": len(observed),
        "record_ids_sha256": canonical_sha256(
            {"record_ids": sorted(observed)}
        ),
        "h2d_bytes": h2d_bytes,
        "d2h_bytes": d2h_bytes,
        "published_bytes": published_bytes,
        "baseline_hbm_allocated_bytes": baseline_allocated,
        "peak_hbm_allocated_bytes": torch.cuda.max_memory_allocated(
            device
        ),
        "peak_hbm_reserved_bytes": torch.cuda.max_memory_reserved(
            device
        ),
        "pinned_slot_bytes": slot.nbytes,
        "old_store_ledger": old_store.ledger().to_dict(),
        "target_store_ledger": target_ledger.to_dict(),
        "group_reports": group_reports,
        "exactly_once_pass": (
            len(observed) == len(set(observed))
            and set(observed) == expected_ids
            and target_ledger.complete_records == len(expected_ids)
            and target_ledger.partial_records == 0
            and target_ledger.missing_records == 0
        ),
    }


def _validate_s1_slots(
    slots: Sequence[M1PinnedSlot],
) -> None:
    if len(slots) != 2:
        raise ValueError("M1 S1 requires exactly two pinned slots")
    storage_lists = [
        [
            value.untyped_storage().data_ptr()
            for value in slot.__dict__.values()
        ]
        for slot in slots
    ]
    if any(
        len(values) != len(set(values))
        for values in storage_lists
    ):
        raise ValueError("M1 S1 pinned slot components alias")
    storages = [set(values) for values in storage_lists]
    if storages[0] & storages[1]:
        raise ValueError("M1 S1 pinned slots alias")


def _interval_overlap_seconds(
    left_start: float,
    left_stop: float,
    right_start: float,
    right_stop: float,
) -> float:
    if (
        left_stop < left_start
        or right_stop < right_start
    ):
        raise ValueError("M1 S1 overlap interval is invalid")
    return max(
        min(left_stop, right_stop) - max(left_start, right_start),
        0.0,
    )


def _stage_s1_group_input(
    group: M1Group,
    rank: int,
    actions_by_id: Mapping[int, M1Action],
    histories: M1Histories,
    old_store: PageableDramExtentStore,
    slot: M1PinnedSlot,
    slot_index: int,
    device: torch.device,
    stream: torch.cuda.Stream,
    micro_batch_records: int,
) -> M1S1StagedGroup:
    staging_started_at = time.perf_counter()
    group_actions = _actions_for_ids(
        group.record_ids_by_rank[rank],
        actions_by_id,
    )
    microbatches_by_rank = tuple(
        _chunks(
            _actions_for_ids(
                group.record_ids_by_rank[value],
                actions_by_id,
            ),
            micro_batch_records,
        )
        for value in range(len(group.record_ids_by_rank))
    )
    micro_steps = max(
        (len(value) for value in microbatches_by_rank),
        default=0,
    )
    local_microbatches = microbatches_by_rank[rank]
    pageable_seconds = 0.0
    h2d_seconds = 0.0
    h2d_bytes = 0
    oldkv_read_bytes = 0
    source_group = None
    device_histories = []
    with torch.cuda.device(device):
        if group.route == "compiled" and group_actions:
            retained_tokens = sum(
                value.retained_tokens for value in group_actions
            )
            with torch.cuda.stream(stream):
                source_k = torch.empty(
                    (
                        old_store.num_layers,
                        retained_tokens,
                        old_store.width,
                    ),
                    dtype=torch.float16,
                    device=device,
                )
                source_v = torch.empty_like(source_k)
            source_offset = 0
            for actions in local_microbatches:
                stage_started = time.perf_counter()
                source, read_bytes = _source_batch_from_store(
                    slot,
                    old_store,
                    actions,
                    "theta0",
                )
                pageable_seconds += time.perf_counter() - stage_started
                if source is None:
                    raise RuntimeError(
                        "M1 S1 compiled source is missing"
                    )
                stop = source_offset + source.token_count
                h2d_start = torch.cuda.Event(enable_timing=True)
                h2d_end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    h2d_start.record()
                    source_k[:, source_offset:stop].copy_(
                        source.k,
                        non_blocking=True,
                    )
                    source_v[:, source_offset:stop].copy_(
                        source.v,
                        non_blocking=True,
                    )
                    h2d_end.record()
                h2d_seconds += _event_seconds(
                    h2d_start,
                    h2d_end,
                )
                bytes_moved = _tensor_pair_nbytes(
                    source.k,
                    source.v,
                )
                h2d_bytes += bytes_moved
                oldkv_read_bytes += read_bytes
                source_offset = stop
            if source_offset != retained_tokens:
                raise RuntimeError(
                    "M1 S1 compiled source extent differs"
                )
            with torch.cuda.stream(stream):
                source_group = _device_jagged(
                    group_actions,
                    source_k,
                    source_v,
                    "retained_tokens",
                    "theta0",
                    "theta0",
                )
            del source_k, source_v
        for step in range(micro_steps):
            actions = (
                local_microbatches[step]
                if step < len(local_microbatches)
                else ()
            )
            starts = (
                tuple(value.delta_start for value in actions)
                if group.route == "compiled"
                else (0,) * len(actions)
            )
            stage_started = time.perf_counter()
            history = _history_batch_into_slot(
                slot,
                actions,
                histories.target,
                starts,
                tuple(value.final_tokens for value in actions),
                "theta1",
            )
            pageable_seconds += time.perf_counter() - stage_started
            h2d_bytes += history.nbytes
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                h2d_start.record()
                device_history = history.to(
                    device,
                    non_blocking=True,
                )
                h2d_end.record()
            h2d_seconds += _event_seconds(
                h2d_start,
                h2d_end,
            )
            device_histories.append(device_history)
            del history
        stream.synchronize()
    return M1S1StagedGroup(
        group=group,
        actions=group_actions,
        local_microbatches=local_microbatches,
        micro_steps=micro_steps,
        source=source_group,
        device_histories=tuple(device_histories),
        oldkv_read_bytes=oldkv_read_bytes,
        h2d_bytes=h2d_bytes,
        pageable_to_pinned_seconds=pageable_seconds,
        h2d_seconds=h2d_seconds,
        staging_started_at=staging_started_at,
        staging_finished_at=time.perf_counter(),
        slot_index=slot_index,
        input_segment_records=micro_batch_records,
        input_segments=micro_steps,
    )


def _stage_d3_group_input(
    group: M1Group,
    rank: int,
    actions_by_id: Mapping[int, M1Action],
    histories: M1Histories,
    old_store: PageableDramExtentStore,
    slots: Sequence[M1PinnedSlot],
    slot_index: int,
    device: torch.device,
    stream: torch.cuda.Stream,
    input_segment_records: int,
    compute_batch_records: int,
) -> M1S1StagedGroup:
    _validate_s1_slots(slots)
    staging_started_at = time.perf_counter()
    group_actions = _actions_for_ids(
        group.record_ids_by_rank[rank],
        actions_by_id,
    )
    input_batches_by_rank = tuple(
        _chunks(
            _actions_for_ids(
                group.record_ids_by_rank[value],
                actions_by_id,
            ),
            input_segment_records,
        )
        for value in range(len(group.record_ids_by_rank))
    )
    input_steps = max(
        (len(value) for value in input_batches_by_rank),
        default=0,
    )
    compute_batches_by_rank = tuple(
        _chunks(
            _actions_for_ids(
                group.record_ids_by_rank[value],
                actions_by_id,
            ),
            compute_batch_records,
        )
        for value in range(len(group.record_ids_by_rank))
    )
    compute_steps = max(
        (len(value) for value in compute_batches_by_rank),
        default=0,
    )
    local_input_batches = input_batches_by_rank[rank]
    local_compute_batches = compute_batches_by_rank[rank]
    pageable_seconds = 0.0
    h2d_bytes = 0
    oldkv_read_bytes = 0
    source_group = None
    source_offset = 0
    source_k = None
    source_v = None
    history_row_offset = 0
    h2d_events: list[
        tuple[torch.cuda.Event, torch.cuda.Event]
    ] = []
    component_ready: list[torch.cuda.Event | None] = [
        None,
        None,
    ]
    slot_reuse_wait_seconds = 0.0
    final_tail_wait_seconds = 0.0
    with torch.cuda.device(device):
        device_history_group = _allocate_device_history_group(
            group_actions,
            group.route,
            device,
            stream,
        )
        if group.route == "compiled" and group_actions:
            retained_tokens = sum(
                value.retained_tokens for value in group_actions
            )
            with torch.cuda.stream(stream):
                source_k = torch.empty(
                    (
                        old_store.num_layers,
                        retained_tokens,
                        old_store.width,
                    ),
                    dtype=torch.float16,
                    device=device,
                )
                source_v = torch.empty_like(source_k)
        for step in range(input_steps):
            component = step % 2
            ready = component_ready[component]
            if ready is not None:
                wait_started = time.perf_counter()
                ready.synchronize()
                slot_reuse_wait_seconds += (
                    time.perf_counter() - wait_started
                )
            slot = slots[component]
            actions = (
                local_input_batches[step]
                if step < len(local_input_batches)
                else ()
            )
            if group.route == "compiled" and actions:
                stage_started = time.perf_counter()
                source, read_bytes = _source_batch_from_store(
                    slot,
                    old_store,
                    actions,
                    "theta0",
                )
                pageable_seconds += (
                    time.perf_counter() - stage_started
                )
                if (
                    source is None
                    or source_k is None
                    or source_v is None
                ):
                    raise RuntimeError(
                        "M1 D3 compiled source is missing"
                    )
                stop = source_offset + source.token_count
                source_start = torch.cuda.Event(
                    enable_timing=True
                )
                source_end = torch.cuda.Event(
                    enable_timing=True
                )
                with torch.cuda.stream(stream):
                    source_start.record()
                    source_k[:, source_offset:stop].copy_(
                        source.k,
                        non_blocking=True,
                    )
                    source_v[:, source_offset:stop].copy_(
                        source.v,
                        non_blocking=True,
                    )
                    source_end.record()
                h2d_events.append(
                    (source_start, source_end)
                )
                h2d_bytes += _tensor_pair_nbytes(
                    source.k,
                    source.v,
                )
                oldkv_read_bytes += read_bytes
                source_offset = stop
            starts = (
                tuple(value.delta_start for value in actions)
                if group.route == "compiled"
                else (0,) * len(actions)
            )
            stage_started = time.perf_counter()
            history = _history_batch_into_slot(
                slot,
                actions,
                histories.target,
                starts,
                tuple(value.final_tokens for value in actions),
                "theta1",
            )
            pageable_seconds += (
                time.perf_counter() - stage_started
            )
            h2d_bytes += history.nbytes
            history_row_stop = history_row_offset + len(actions)
            history_start = torch.cuda.Event(
                enable_timing=True
            )
            history_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                history_start.record()
                device_history_group.item_ids[
                    history_row_offset:history_row_stop,
                    : history.seq_len,
                ].copy_(
                    history.item_ids,
                    non_blocking=True,
                )
                device_history_group.behaviors[
                    history_row_offset:history_row_stop,
                    : history.seq_len,
                ].copy_(
                    history.behaviors,
                    non_blocking=True,
                )
                device_history_group.time_deltas[
                    history_row_offset:history_row_stop,
                    : history.seq_len,
                ].copy_(
                    history.time_deltas,
                    non_blocking=True,
                )
                device_history_group.lengths[
                    history_row_offset:history_row_stop
                ].copy_(
                    history.lengths,
                    non_blocking=True,
                )
                history_end.record()
            h2d_events.append(
                (history_start, history_end)
            )
            component_ready[component] = history_end
            history_row_offset = history_row_stop
            del history
        tail_started = time.perf_counter()
        for ready in component_ready:
            if ready is not None:
                ready.synchronize()
        final_tail_wait_seconds = (
            time.perf_counter() - tail_started
        )
        h2d_seconds = sum(
            started.elapsed_time(finished) / 1000.0
            for started, finished in h2d_events
        )
        if history_row_offset != len(group_actions):
            raise RuntimeError("M1 D3 input history coverage differs")
        if group.route == "compiled" and group_actions:
            retained_tokens = sum(
                value.retained_tokens for value in group_actions
            )
            if (
                source_k is None
                or source_v is None
                or source_offset != retained_tokens
            ):
                raise RuntimeError(
                    "M1 D3 compiled source extent differs"
                )
            with torch.cuda.stream(stream):
                source_group = _device_jagged(
                    group_actions,
                    source_k,
                    source_v,
                    "retained_tokens",
                    "theta0",
                    "theta0",
                )
            stream.synchronize()
        else:
            stream.synchronize()
        device_histories = _device_history_views(
            device_history_group,
            local_compute_batches,
            compute_steps,
        )
    return M1S1StagedGroup(
        group=group,
        actions=group_actions,
        local_microbatches=local_compute_batches,
        micro_steps=compute_steps,
        source=source_group,
        device_histories=device_histories,
        oldkv_read_bytes=oldkv_read_bytes,
        h2d_bytes=h2d_bytes,
        pageable_to_pinned_seconds=pageable_seconds,
        h2d_seconds=h2d_seconds,
        staging_started_at=staging_started_at,
        staging_finished_at=time.perf_counter(),
        slot_index=slot_index,
        input_pipeline="segmented_microbatch_pingpong",
        input_segment_records=input_segment_records,
        input_segments=input_steps,
        input_slot_reuse_wait_seconds=(
            slot_reuse_wait_seconds
        ),
        input_final_tail_wait_seconds=(
            final_tail_wait_seconds
        ),
    )


def _compute_s1_staged_group(
    staged: M1S1StagedGroup,
    model,
    operator: DirectOldKVFusedOperator,
    program,
    device: torch.device,
    compute_batch_records: int,
) -> dict[str, object]:
    group = staged.group
    group_actions = staged.actions
    local_microbatches = staged.local_microbatches
    group_lookup: dict[str, int | float] = {}
    group_d2d_seconds = 0.0
    group_assemble_seconds = 0.0
    group_lookup_seconds = 0.0
    group_compute_seconds = 0.0
    target_retained = None
    target_suffix = None
    target_exact = None
    execution_started_at = time.perf_counter()
    if group.route == "compiled" and group_actions:
        if staged.source is None:
            raise RuntimeError("M1 S1 staged source is absent")
        retained_tokens = sum(
            value.retained_tokens for value in group_actions
        )
        target_retained_k = torch.empty(
            (
                program.num_layers,
                retained_tokens,
                program.kv_width,
            ),
            dtype=torch.float16,
            device=device,
        )
        target_retained_v = torch.empty_like(target_retained_k)
        suffix_tokens = sum(
            value.suffix_tokens for value in group_actions
        )
        target_suffix_k = torch.empty(
            (
                program.num_layers,
                suffix_tokens,
                program.kv_width,
            ),
            dtype=torch.float16,
            device=device,
        )
        target_suffix_v = torch.empty_like(target_suffix_k)
        target_retained = _device_jagged(
            group_actions,
            target_retained_k,
            target_retained_v,
            "retained_tokens",
            "theta0",
            "theta1",
        )
        target_suffix = _device_jagged(
            group_actions,
            target_suffix_k,
            target_suffix_v,
            "suffix_tokens",
            "theta1",
            "theta1",
        )
        del (
            target_retained_k,
            target_retained_v,
            target_suffix_k,
            target_suffix_v,
        )
        d2d_start = torch.cuda.Event(enable_timing=True)
        d2d_end = torch.cuda.Event(enable_timing=True)
        d2d_start.record()
        operator.execute_into(
            program,
            staged.source,
            target_retained,
        )
        d2d_end.record()
        group_d2d_seconds = _event_seconds(
            d2d_start,
            d2d_end,
        )
    elif group.route == "exact" and group_actions:
        exact_tokens = sum(
            value.final_tokens for value in group_actions
        )
        exact_k = torch.empty(
            (
                model.dense_model.cfg.num_layers,
                exact_tokens,
                model.dense_model.cfg.hidden_size,
            ),
            dtype=torch.float16,
            device=device,
        )
        exact_v = torch.empty_like(exact_k)
        target_exact = _device_jagged(
            group_actions,
            exact_k,
            exact_v,
            "final_tokens",
            "theta1",
            "theta1",
        )
        del exact_k, exact_v
    retained_offset = 0
    suffix_offset = 0
    exact_offset = 0
    for step, device_history in enumerate(
        staged.device_histories
    ):
        actions = (
            local_microbatches[step]
            if step < len(local_microbatches)
            else ()
        )
        transformed = None
        if actions and group.route == "compiled":
            retained_stop = retained_offset + sum(
                value.retained_tokens for value in actions
            )
            assert target_retained is not None
            gather_start = torch.cuda.Event(enable_timing=True)
            gather_end = torch.cuda.Event(enable_timing=True)
            gather_start.record()
            transformed_k = target_retained.k[
                :, retained_offset:retained_stop
            ].contiguous()
            transformed_v = target_retained.v[
                :, retained_offset:retained_stop
            ].contiguous()
            transformed = _device_jagged(
                actions,
                transformed_k,
                transformed_v,
                "retained_tokens",
                "theta0",
                "theta1",
            )
            gather_end.record()
            group_assemble_seconds += _event_seconds(
                gather_start,
                gather_end,
            )
            del transformed_k, transformed_v
        compute_started = time.perf_counter()
        if group.route == "compiled":
            result = integrated_sharded_append_only(
                model,
                transformed,
                device_history,
                "theta1",
                collective_timing="current_stream",
            )
        else:
            result = integrated_sharded_exact(
                model,
                device_history,
                "theta1",
                collective_timing="current_stream",
            )
        compute_end = torch.cuda.Event(enable_timing=True)
        compute_end.record()
        compute_end.synchronize()
        compute_and_lookup_seconds = (
            time.perf_counter() - compute_started
        )
        lookup_seconds = result.lookup_metrics.collective_seconds
        compute_seconds = max(
            compute_and_lookup_seconds - lookup_seconds,
            0.0,
        )
        group_lookup_seconds += lookup_seconds
        group_compute_seconds += compute_seconds
        _sum_lookup_metrics(group_lookup, result.lookup_metrics)
        if result.fragment is not None:
            assemble_start = torch.cuda.Event(enable_timing=True)
            assemble_end = torch.cuda.Event(enable_timing=True)
            assemble_start.record()
            if isinstance(
                result.fragment,
                IntegratedAppendOnlyKVBatch,
            ):
                suffix_stop = suffix_offset + (
                    result.fragment.suffix.token_count
                )
                assert target_suffix is not None
                target_suffix.k[:, suffix_offset:suffix_stop].copy_(
                    result.fragment.suffix.k
                )
                target_suffix.v[:, suffix_offset:suffix_stop].copy_(
                    result.fragment.suffix.v
                )
                suffix_offset = suffix_stop
                retained_offset += (
                    result.fragment.retained.token_count
                )
            else:
                exact_stop = (
                    exact_offset + result.fragment.token_count
                )
                assert target_exact is not None
                target_exact.k[:, exact_offset:exact_stop].copy_(
                    result.fragment.k
                )
                target_exact.v[:, exact_offset:exact_stop].copy_(
                    result.fragment.v
                )
                exact_offset = exact_stop
            assemble_end.record()
            group_assemble_seconds += _event_seconds(
                assemble_start,
                assemble_end,
            )
        elif actions:
            raise RuntimeError("M1 S1 lost local microbatch output")
        del result, transformed
    if group.route == "compiled" and group_actions:
        assert target_retained is not None
        assert target_suffix is not None
        if (
            retained_offset != target_retained.token_count
            or suffix_offset != target_suffix.token_count
        ):
            raise RuntimeError("M1 S1 compiled assembly differs")
    if group.route == "exact" and group_actions:
        assert target_exact is not None
        if exact_offset != target_exact.token_count:
            raise RuntimeError("M1 S1 exact assembly differs")
    ready_event = torch.cuda.Event()
    ready_event.record()
    execution_finished_at = time.perf_counter()
    report = {
        "ordinal": group.ordinal,
        "route": group.route,
        "capacity_group_records": len(group_actions),
        "compute_microbatches": staged.micro_steps,
        "micro_batch_records": compute_batch_records,
        "compute_batch_records": compute_batch_records,
        "resident_extent_logical_bytes": (
            group_resident_extent_bytes(
                group_actions,
                model.dense_model.cfg,
            )
        ),
        "resident_extent_allocated_bytes": None,
        "oldkv_read_bytes": staged.oldkv_read_bytes,
        "h2d_bytes": staged.h2d_bytes,
        "d2h_bytes": 0,
        "published_bytes": 0,
        "pageable_to_pinned_seconds": (
            staged.pageable_to_pinned_seconds
        ),
        "h2d_seconds": staged.h2d_seconds,
        "input_staging_wall_seconds": staged.staging_wall_seconds,
        "input_pipeline": staged.input_pipeline,
        "input_segment_records": staged.input_segment_records,
        "input_segments": staged.input_segments,
        "input_slot_reuse_wait_seconds": (
            staged.input_slot_reuse_wait_seconds
        ),
        "input_final_tail_wait_seconds": (
            staged.input_final_tail_wait_seconds
        ),
        "estimated_hidden_input_seconds": (
            staged.estimated_hidden_input_seconds
        ),
        "d2d_transform_seconds": group_d2d_seconds,
        "d2d_assemble_seconds": group_assemble_seconds,
        "lookup_exchange_seconds": group_lookup_seconds,
        "compute_excluding_lookup_seconds": group_compute_seconds,
        "execution_wall_seconds": (
            execution_finished_at - execution_started_at
        ),
        "d2h_seconds": 0.0,
        "publish_seconds": 0.0,
        "lookup_metrics": group_lookup,
    }
    return {
        "computed_group": M1S1ComputedGroup(
            group=group,
            actions=group_actions,
            target_retained=target_retained,
            target_suffix=target_suffix,
            target_exact=target_exact,
            report=report,
            lookup_metrics=group_lookup,
            execution_started_at=execution_started_at,
            execution_finished_at=execution_finished_at,
            ready_event=ready_event,
            slot_index=staged.slot_index,
        )
    }


def _drain_s1_computed_group(
    computed: M1S1ComputedGroup,
    target_store: PageableDramExtentStore,
    slot: M1PinnedSlot,
    device: torch.device,
    stream: torch.cuda.Stream,
    output_segment_records: int,
) -> dict[str, object]:
    drain_started_at = time.perf_counter()
    group = computed.group
    group_actions = computed.actions
    target_retained = computed.target_retained
    target_suffix = computed.target_suffix
    target_exact = computed.target_exact
    group_d2h_bytes = 0
    group_published_bytes = 0
    group_d2h_seconds = 0.0
    group_publish_seconds = 0.0
    observed = []
    retained_offset = 0
    suffix_offset = 0
    exact_offset = 0
    with torch.cuda.device(device):
        with torch.cuda.stream(stream):
            stream.wait_event(computed.ready_event)
        for actions in _chunks(
            group_actions,
            output_segment_records,
        ):
            segments: list[
                tuple[torch.Tensor, torch.Tensor]
            ] = []
            d2h_start = torch.cuda.Event(enable_timing=True)
            d2h_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                d2h_start.record()
                token_offset = 0
                if group.route == "compiled":
                    retained_stop = retained_offset + sum(
                        value.retained_tokens
                        for value in actions
                    )
                    suffix_stop = suffix_offset + sum(
                        value.suffix_tokens for value in actions
                    )
                    if (
                        target_retained is None
                        or target_suffix is None
                    ):
                        raise RuntimeError(
                            "M1 S1 compiled drain output is absent"
                        )
                    retained_k, retained_v, token_offset = (
                        _copy_values_to_output(
                            slot,
                            target_retained.k[
                                :,
                                retained_offset:retained_stop,
                            ],
                            target_retained.v[
                                :,
                                retained_offset:retained_stop,
                            ],
                            token_offset,
                        )
                    )
                    suffix_k, suffix_v, token_offset = (
                        _copy_values_to_output(
                            slot,
                            target_suffix.k[
                                :,
                                suffix_offset:suffix_stop,
                            ],
                            target_suffix.v[
                                :,
                                suffix_offset:suffix_stop,
                            ],
                            token_offset,
                        )
                    )
                    segments = [
                        (retained_k, retained_v),
                        (suffix_k, suffix_v),
                    ]
                    retained_offset = retained_stop
                    suffix_offset = suffix_stop
                else:
                    exact_stop = exact_offset + sum(
                        value.final_tokens for value in actions
                    )
                    if target_exact is None:
                        raise RuntimeError(
                            "M1 S1 exact drain output is absent"
                        )
                    k, v, token_offset = _copy_values_to_output(
                        slot,
                        target_exact.k[
                            :,
                            exact_offset:exact_stop,
                        ],
                        target_exact.v[
                            :,
                            exact_offset:exact_stop,
                        ],
                        token_offset,
                    )
                    segments = [(k, v)]
                    exact_offset = exact_stop
                d2h_end.record()
            group_d2h_seconds += _event_seconds(
                d2h_start,
                d2h_end,
            )
            bytes_moved = sum(
                _tensor_pair_nbytes(k, v)
                for k, v in segments
            )
            group_d2h_bytes += bytes_moved
            if not _finite_sample(segments):
                raise RuntimeError(
                    "M1 S1 output K/V is nonfinite"
                )
            publish_started = time.perf_counter()
            group_published_bytes += publish_output_segments(
                target_store,
                actions,
                group.route,
                segments,
            )
            group_publish_seconds += (
                time.perf_counter() - publish_started
            )
            observed.extend(
                value.record_id for value in actions
            )
    if group.route == "compiled" and group_actions:
        if (
            target_retained is None
            or target_suffix is None
            or retained_offset != target_retained.token_count
            or suffix_offset != target_suffix.token_count
        ):
            raise RuntimeError(
                "M1 S1 compiled drain extent differs"
            )
    if group.route == "exact" and group_actions:
        if (
            target_exact is None
            or exact_offset != target_exact.token_count
        ):
            raise RuntimeError("M1 S1 exact drain extent differs")
    drain_finished_at = time.perf_counter()
    report = dict(computed.report)
    report.update(
        {
            "d2h_bytes": group_d2h_bytes,
            "published_bytes": group_published_bytes,
            "d2h_seconds": group_d2h_seconds,
            "publish_seconds": group_publish_seconds,
            "drain_wall_seconds": (
                drain_finished_at - drain_started_at
            ),
        }
    )
    return {
        "group_report": report,
        "observed_record_ids": tuple(observed),
        "d2h_bytes": group_d2h_bytes,
        "published_bytes": group_published_bytes,
        "drain_started_at": drain_started_at,
        "drain_finished_at": drain_finished_at,
    }


def _enqueue_d3_output_transfer(
    computed: M1S1ComputedGroup,
    actions: tuple[M1Action, ...],
    slot: M1PinnedSlot,
    stream: torch.cuda.Stream,
    retained_offset: int,
    suffix_offset: int,
    exact_offset: int,
) -> tuple[M1D3OutputTransfer, int, int, int]:
    group = computed.group
    target_retained = computed.target_retained
    target_suffix = computed.target_suffix
    target_exact = computed.target_exact
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    segments: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.cuda.stream(stream):
        started.record()
        token_offset = 0
        if group.route == "compiled":
            retained_stop = retained_offset + sum(
                value.retained_tokens for value in actions
            )
            suffix_stop = suffix_offset + sum(
                value.suffix_tokens for value in actions
            )
            if (
                target_retained is None
                or target_suffix is None
            ):
                raise RuntimeError(
                    "M1 D3 compiled drain output is absent"
                )
            retained_k, retained_v, token_offset = (
                _copy_values_to_output(
                    slot,
                    target_retained.k[
                        :,
                        retained_offset:retained_stop,
                    ],
                    target_retained.v[
                        :,
                        retained_offset:retained_stop,
                    ],
                    token_offset,
                )
            )
            suffix_k, suffix_v, token_offset = (
                _copy_values_to_output(
                    slot,
                    target_suffix.k[
                        :,
                        suffix_offset:suffix_stop,
                    ],
                    target_suffix.v[
                        :,
                        suffix_offset:suffix_stop,
                    ],
                    token_offset,
                )
            )
            segments = [
                (retained_k, retained_v),
                (suffix_k, suffix_v),
            ]
            retained_offset = retained_stop
            suffix_offset = suffix_stop
        else:
            exact_stop = exact_offset + sum(
                value.final_tokens for value in actions
            )
            if target_exact is None:
                raise RuntimeError(
                    "M1 D3 exact drain output is absent"
                )
            k, v, token_offset = _copy_values_to_output(
                slot,
                target_exact.k[
                    :,
                    exact_offset:exact_stop,
                ],
                target_exact.v[
                    :,
                    exact_offset:exact_stop,
                ],
                token_offset,
            )
            segments = [(k, v)]
            exact_offset = exact_stop
        finished.record()
    values = tuple(segments)
    return (
        M1D3OutputTransfer(
            actions=actions,
            segments=values,
            started=started,
            finished=finished,
            bytes_moved=sum(
                _tensor_pair_nbytes(k, v)
                for k, v in values
            ),
        ),
        retained_offset,
        suffix_offset,
        exact_offset,
    )


def _publish_d3_output_transfer(
    transfer: M1D3OutputTransfer,
    target_store: PageableDramExtentStore,
    route: str,
) -> tuple[float, float, int]:
    wait_started = time.perf_counter()
    transfer.finished.synchronize()
    ready_wait_seconds = time.perf_counter() - wait_started
    if not _finite_sample(transfer.segments):
        raise RuntimeError("M1 D3 output K/V is nonfinite")
    publish_started = time.perf_counter()
    published_bytes = publish_output_segments(
        target_store,
        transfer.actions,
        route,
        transfer.segments,
    )
    publish_seconds = time.perf_counter() - publish_started
    return ready_wait_seconds, publish_seconds, published_bytes


def _drain_d3_computed_group(
    computed: M1S1ComputedGroup,
    target_store: PageableDramExtentStore,
    slots: Sequence[M1PinnedSlot],
    device: torch.device,
    stream: torch.cuda.Stream,
    output_segment_records: int,
) -> dict[str, object]:
    _validate_s1_slots(slots)
    drain_started_at = time.perf_counter()
    group = computed.group
    group_actions = computed.actions
    group_d2h_bytes = 0
    group_published_bytes = 0
    group_d2h_seconds = 0.0
    group_publish_seconds = 0.0
    d2h_ready_wait_seconds = 0.0
    observed = []
    retained_offset = 0
    suffix_offset = 0
    exact_offset = 0
    output_segments = 0
    pending = None
    with torch.cuda.device(device):
        with torch.cuda.stream(stream):
            stream.wait_event(computed.ready_event)
        for index, actions in enumerate(
            _chunks(group_actions, output_segment_records)
        ):
            transfer, retained_offset, suffix_offset, exact_offset = (
                _enqueue_d3_output_transfer(
                    computed,
                    actions,
                    slots[index % 2],
                    stream,
                    retained_offset,
                    suffix_offset,
                    exact_offset,
                )
            )
            output_segments += 1
            if pending is not None:
                (
                    ready_wait,
                    publish_seconds,
                    published_bytes,
                ) = _publish_d3_output_transfer(
                    pending,
                    target_store,
                    group.route,
                )
                d2h_ready_wait_seconds += ready_wait
                group_publish_seconds += publish_seconds
                group_published_bytes += published_bytes
                group_d2h_seconds += (
                    pending.started.elapsed_time(
                        pending.finished
                    )
                    / 1000.0
                )
                group_d2h_bytes += pending.bytes_moved
                observed.extend(
                    value.record_id
                    for value in pending.actions
                )
            pending = transfer
        if pending is not None:
            (
                ready_wait,
                publish_seconds,
                published_bytes,
            ) = _publish_d3_output_transfer(
                pending,
                target_store,
                group.route,
            )
            d2h_ready_wait_seconds += ready_wait
            group_publish_seconds += publish_seconds
            group_published_bytes += published_bytes
            group_d2h_seconds += (
                pending.started.elapsed_time(
                    pending.finished
                )
                / 1000.0
            )
            group_d2h_bytes += pending.bytes_moved
            observed.extend(
                value.record_id for value in pending.actions
            )
    if group.route == "compiled" and group_actions:
        if (
            computed.target_retained is None
            or computed.target_suffix is None
            or retained_offset
            != computed.target_retained.token_count
            or suffix_offset
            != computed.target_suffix.token_count
        ):
            raise RuntimeError(
                "M1 D3 compiled drain extent differs"
            )
    if group.route == "exact" and group_actions:
        if (
            computed.target_exact is None
            or exact_offset != computed.target_exact.token_count
        ):
            raise RuntimeError("M1 D3 exact drain extent differs")
    drain_finished_at = time.perf_counter()
    drain_wall_seconds = (
        drain_finished_at - drain_started_at
    )
    report = dict(computed.report)
    report.update(
        {
            "d2h_bytes": group_d2h_bytes,
            "published_bytes": group_published_bytes,
            "d2h_seconds": group_d2h_seconds,
            "publish_seconds": group_publish_seconds,
            "drain_wall_seconds": drain_wall_seconds,
            "output_pipeline": (
                "segmented_microbatch_pingpong"
            ),
            "output_segments": output_segments,
            "output_segment_records": output_segment_records,
            "output_d2h_ready_wait_seconds": (
                d2h_ready_wait_seconds
            ),
            "estimated_hidden_output_seconds": max(
                group_d2h_seconds
                + group_publish_seconds
                - drain_wall_seconds,
                0.0,
            ),
        }
    )
    return {
        "group_report": report,
        "observed_record_ids": tuple(observed),
        "d2h_bytes": group_d2h_bytes,
        "published_bytes": group_published_bytes,
        "drain_started_at": drain_started_at,
        "drain_finished_at": drain_finished_at,
    }


def _s1_input_edge(
    producer_group_ordinal: int,
    producer_execution_started_at: float,
    producer_execution_finished_at: float,
    prefetched: M1S1StagedGroup,
    measured_wait_seconds: float,
) -> dict[str, int | float]:
    overlap = _interval_overlap_seconds(
        prefetched.staging_started_at,
        prefetched.staging_finished_at,
        producer_execution_started_at,
        producer_execution_finished_at,
    )
    tail = max(
        prefetched.staging_finished_at
        - producer_execution_finished_at,
        0.0,
    )
    return {
        "producer_group_ordinal": producer_group_ordinal,
        "prefetched_group_ordinal": (
            prefetched.group.ordinal
        ),
        "producer_execution_seconds": (
            producer_execution_finished_at
            - producer_execution_started_at
        ),
        "input_staging_wall_seconds": (
            prefetched.staging_wall_seconds
        ),
        "overlap_interval_seconds": overlap,
        "staging_tail_after_producer_seconds": tail,
        "measured_boundary_wait_seconds": (
            measured_wait_seconds
        ),
        "overlap_fraction": (
            overlap / prefetched.staging_wall_seconds
            if prefetched.staging_wall_seconds
            else 0.0
        ),
    }


def _s1_drain_edge(
    drained: Mapping[str, object],
    consumer: M1S1ComputedGroup,
    measured_credit_wait_seconds: float,
) -> dict[str, int | float]:
    started = float(drained["drain_started_at"])
    finished = float(drained["drain_finished_at"])
    overlap = _interval_overlap_seconds(
        started,
        finished,
        consumer.execution_started_at,
        consumer.execution_finished_at,
    )
    wall = finished - started
    report = drained["group_report"]
    if not isinstance(report, Mapping):
        raise RuntimeError("M1 S1 drain report differs")
    return {
        "drained_group_ordinal": int(report["ordinal"]),
        "overlapped_compute_group_ordinal": (
            consumer.group.ordinal
        ),
        "drain_wall_seconds": wall,
        "compute_wall_seconds": (
            consumer.execution_finished_at
            - consumer.execution_started_at
        ),
        "overlap_interval_seconds": overlap,
        "measured_output_credit_wait_seconds": (
            measured_credit_wait_seconds
        ),
        "overlap_fraction": (
            overlap / wall if wall else 0.0
        ),
    }


def run_s1(
    groups: Sequence[M1Group],
    rank: int,
    actions_by_id: Mapping[int, M1Action],
    histories: M1Histories,
    old_store: PageableDramExtentStore,
    target_store: PageableDramExtentStore,
    model,
    operator: DirectOldKVFusedOperator,
    program,
    slots: Sequence[M1PinnedSlot],
    device: torch.device,
    granularity: M1ExecutionGranularity,
    segmented_input: bool = False,
) -> dict[str, object]:
    if not groups:
        raise ValueError("M1 S1 requires at least one group")
    _validate_s1_slots(slots)
    phases = {
        "pageable_to_pinned_seconds": 0.0,
        "h2d_seconds": 0.0,
        "d2d_transform_seconds": 0.0,
        "d2d_assemble_seconds": 0.0,
        "lookup_exchange_seconds": 0.0,
        "compute_excluding_lookup_seconds": 0.0,
        "d2h_seconds": 0.0,
        "publish_seconds": 0.0,
    }
    lookup_totals: dict[str, int | float] = {}
    group_reports = []
    input_edges = []
    drain_edges = []
    input_arrivals: dict[int, dict[str, int | float | str]] = {}
    observed = []
    h2d_bytes = 0
    d2h_bytes = 0
    published_bytes = 0
    input_boundary_wait_seconds = 0.0
    output_credit_wait_seconds = 0.0
    prefetched_input_seconds = 0.0
    overlapped_input_seconds = 0.0
    overlapped_drain_seconds = 0.0
    eligible_drain_seconds = 0.0
    input_segments = 0
    input_slot_reuse_wait_seconds = 0.0
    input_final_tail_wait_seconds = 0.0
    estimated_hidden_input_seconds = 0.0
    output_segments = 0
    output_d2h_ready_wait_seconds = 0.0
    estimated_hidden_output_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    wall_started = time.perf_counter()
    prefetch_stream = torch.cuda.Stream(device=device)
    drain_stream = torch.cuda.Stream(device=device)

    def stage_group_input(
        group: M1Group,
        group_index: int,
    ) -> M1S1StagedGroup:
        slot_index = group_index % 2
        route_granularity = granularity.for_route(group.route)
        if segmented_input:
            return _stage_d3_group_input(
                group,
                rank,
                actions_by_id,
                histories,
                old_store,
                slots,
                slot_index,
                device,
                prefetch_stream,
                route_granularity.input_segment_records,
                route_granularity.compute_batch_records,
            )
        return _stage_s1_group_input(
            group,
            rank,
            actions_by_id,
            histories,
            old_store,
            slots[slot_index],
            slot_index,
            device,
            prefetch_stream,
            route_granularity.compute_batch_records,
        )

    current = stage_group_input(groups[0], 0)
    initial_fill_seconds = current.staging_wall_seconds
    phases["pageable_to_pinned_seconds"] += (
        current.pageable_to_pinned_seconds
    )
    phases["h2d_seconds"] += current.h2d_seconds
    h2d_bytes += current.h2d_bytes
    input_segments += current.input_segments
    input_slot_reuse_wait_seconds += (
        current.input_slot_reuse_wait_seconds
    )
    input_final_tail_wait_seconds += (
        current.input_final_tail_wait_seconds
    )
    estimated_hidden_input_seconds += (
        current.estimated_hidden_input_seconds
    )
    input_arrivals[current.group.ordinal] = {
        "input_fill_class": "initial_unoverlapped_fill",
        "input_overlap_seconds": 0.0,
        "input_staging_tail_seconds": initial_fill_seconds,
        "measured_input_boundary_wait_seconds": (
            initial_fill_seconds
        ),
    }
    drain_future = None
    final_drain_wait_seconds = 0.0

    def consume_drain(
        drained: Mapping[str, object],
    ) -> None:
        nonlocal d2h_bytes, published_bytes
        nonlocal output_segments
        nonlocal output_d2h_ready_wait_seconds
        nonlocal estimated_hidden_output_seconds
        report_value = drained["group_report"]
        if not isinstance(report_value, Mapping):
            raise RuntimeError("M1 S1 group report differs")
        report = dict(report_value)
        report.update(
            input_arrivals[int(report["ordinal"])]
        )
        group_reports.append(report)
        phases["d2h_seconds"] += float(
            report["d2h_seconds"]
        )
        phases["publish_seconds"] += float(
            report["publish_seconds"]
        )
        d2h_bytes += int(drained["d2h_bytes"])
        published_bytes += int(drained["published_bytes"])
        output_segments += int(
            report.get("output_segments", 0)
        )
        output_d2h_ready_wait_seconds += float(
            report.get(
                "output_d2h_ready_wait_seconds",
                0.0,
            )
        )
        estimated_hidden_output_seconds += float(
            report.get(
                "estimated_hidden_output_seconds",
                0.0,
            )
        )
        observed.extend(drained["observed_record_ids"])

    with (
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="evokv-s1-prefetch",
        ) as prefetch_executor,
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="evokv-s1-drain",
        ) as drain_executor,
    ):
        for index, group in enumerate(groups):
            if current.group != group:
                raise RuntimeError("M1 S1 group order changed")
            prefetch_future = None
            if index + 1 < len(groups):
                next_index = index + 1
                prefetch_future = prefetch_executor.submit(
                    stage_group_input,
                    groups[next_index],
                    next_index,
                )
            computed_value = _compute_s1_staged_group(
                current,
                model,
                operator,
                program,
                device,
                granularity.for_route(
                    current.group.route
                ).compute_batch_records,
            )
            computed = computed_value.get("computed_group")
            if not isinstance(computed, M1S1ComputedGroup):
                raise RuntimeError("M1 S1 computed group differs")
            compute_report = computed.report
            for name in (
                "d2d_transform_seconds",
                "d2d_assemble_seconds",
                "lookup_exchange_seconds",
                "compute_excluding_lookup_seconds",
            ):
                phases[name] += float(compute_report[name])
            _merge_lookup_metrics(
                lookup_totals,
                computed.lookup_metrics,
            )
            if drain_future is not None:
                wait_started = time.perf_counter()
                drained = drain_future.result()
                credit_wait = time.perf_counter() - wait_started
                output_credit_wait_seconds += credit_wait
                drain_edge = _s1_drain_edge(
                    drained,
                    computed,
                    credit_wait,
                )
                drain_edges.append(drain_edge)
                overlapped_drain_seconds += float(
                    drain_edge["overlap_interval_seconds"]
                )
                eligible_drain_seconds += float(
                    drain_edge["drain_wall_seconds"]
                )
                consume_drain(drained)
            if segmented_input:
                drain_future = drain_executor.submit(
                    _drain_d3_computed_group,
                    computed,
                    target_store,
                    slots,
                    device,
                    drain_stream,
                    granularity.for_route(
                        computed.group.route
                    ).output_segment_records,
                )
            else:
                drain_future = drain_executor.submit(
                    _drain_s1_computed_group,
                    computed,
                    target_store,
                    slots[computed.slot_index],
                    device,
                    drain_stream,
                    granularity.for_route(
                        computed.group.route
                    ).compute_batch_records,
                )
            producer_group_ordinal = computed.group.ordinal
            producer_execution_started_at = (
                computed.execution_started_at
            )
            producer_execution_finished_at = (
                computed.execution_finished_at
            )
            del computed_value, computed
            if prefetch_future is None:
                continue
            wait_started = time.perf_counter()
            next_group = prefetch_future.result()
            input_wait = time.perf_counter() - wait_started
            input_boundary_wait_seconds += input_wait
            edge = _s1_input_edge(
                producer_group_ordinal,
                producer_execution_started_at,
                producer_execution_finished_at,
                next_group,
                input_wait,
            )
            input_edges.append(edge)
            prefetched_input_seconds += (
                next_group.staging_wall_seconds
            )
            overlapped_input_seconds += float(
                edge["overlap_interval_seconds"]
            )
            phases["pageable_to_pinned_seconds"] += (
                next_group.pageable_to_pinned_seconds
            )
            phases["h2d_seconds"] += next_group.h2d_seconds
            h2d_bytes += next_group.h2d_bytes
            input_segments += next_group.input_segments
            input_slot_reuse_wait_seconds += (
                next_group.input_slot_reuse_wait_seconds
            )
            input_final_tail_wait_seconds += (
                next_group.input_final_tail_wait_seconds
            )
            estimated_hidden_input_seconds += (
                next_group.estimated_hidden_input_seconds
            )
            input_arrivals[next_group.group.ordinal] = {
                "input_fill_class": "prefetched",
                "input_overlap_seconds": edge[
                    "overlap_interval_seconds"
                ],
                "input_staging_tail_seconds": edge[
                    "staging_tail_after_producer_seconds"
                ],
                "measured_input_boundary_wait_seconds": (
                    input_wait
                ),
            }
            previous = current
            current = next_group
            del previous
        if drain_future is None:
            raise RuntimeError("M1 S1 final drain is absent")
        wait_started = time.perf_counter()
        drained = drain_future.result()
        final_drain_wait_seconds = (
            time.perf_counter() - wait_started
        )
        consume_drain(drained)
    torch.cuda.synchronize(device)
    target_ledger = target_store.ledger()
    expected_ids = set(target_store.record_ids)
    wall_seconds = time.perf_counter() - wall_started
    group_reports.sort(key=lambda value: int(value["ordinal"]))
    exposed_wait_seconds = (
        initial_fill_seconds
        + input_boundary_wait_seconds
        + output_credit_wait_seconds
        + final_drain_wait_seconds
    )
    return {
        "wall_seconds": wall_seconds,
        "phase_seconds": phases,
        "raw_phase_sum_seconds": sum(phases.values()),
        "lookup_metrics": lookup_totals,
        "records": len(observed),
        "record_ids_sha256": canonical_sha256(
            {"record_ids": sorted(observed)}
        ),
        "h2d_bytes": h2d_bytes,
        "d2h_bytes": d2h_bytes,
        "published_bytes": published_bytes,
        "baseline_hbm_allocated_bytes": baseline_allocated,
        "peak_hbm_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_hbm_reserved_bytes": (
            torch.cuda.max_memory_reserved(device)
        ),
        "pinned_slot_bytes": sum(
            slot.nbytes for slot in slots
        ),
        "old_store_ledger": old_store.ledger().to_dict(),
        "target_store_ledger": target_ledger.to_dict(),
        "group_reports": group_reports,
        "input_pipeline_edges": input_edges,
        "drain_pipeline_edges": drain_edges,
        "overlap_metrics": {
            "buffer_depth": 2,
            "execution_granularity": granularity.to_dict(),
            "input_pipeline": (
                "segmented_microbatch_pingpong"
                if segmented_input
                else "serial_microbatch"
            ),
            "input_segments": input_segments,
            "input_slot_reuse_wait_seconds": (
                input_slot_reuse_wait_seconds
            ),
            "input_final_tail_wait_seconds": (
                input_final_tail_wait_seconds
            ),
            "estimated_hidden_input_seconds": (
                estimated_hidden_input_seconds
            ),
            "output_pipeline": (
                "segmented_microbatch_pingpong"
                if segmented_input
                else "serial_microbatch"
            ),
            "output_segments": output_segments,
            "output_d2h_ready_wait_seconds": (
                output_d2h_ready_wait_seconds
            ),
            "estimated_hidden_output_seconds": (
                estimated_hidden_output_seconds
            ),
            "initial_fill_seconds": initial_fill_seconds,
            "prefetched_input_staging_seconds": (
                prefetched_input_seconds
            ),
            "overlapped_input_staging_seconds": (
                overlapped_input_seconds
            ),
            "input_boundary_wait_seconds": (
                input_boundary_wait_seconds
            ),
            "output_credit_wait_seconds": (
                output_credit_wait_seconds
            ),
            "final_drain_wait_seconds": (
                final_drain_wait_seconds
            ),
            "eligible_drain_wall_seconds": (
                eligible_drain_seconds
            ),
            "drain_compute_overlap_seconds": (
                overlapped_drain_seconds
            ),
            "prefetch_overlap_ratio": (
                overlapped_input_seconds
                / prefetched_input_seconds
                if prefetched_input_seconds
                else 0.0
            ),
            "drain_compute_overlap_ratio": (
                overlapped_drain_seconds
                / eligible_drain_seconds
                if eligible_drain_seconds
                else 0.0
            ),
            "exposed_pipeline_wait_seconds": (
                exposed_wait_seconds
            ),
            "pipeline_exposure_fraction": (
                exposed_wait_seconds / wall_seconds
                if wall_seconds
                else 0.0
            ),
        },
        "overlap_contract": {
            "overlapped": (
                "prefetch group i+1, execute group i, and "
                "D2H plus CPU publication for group i-1 run "
                "concurrently on disjoint slot components; "
                + (
                    "inside each group, pageable fill for input "
                    "segment j+1 overlaps H2D j, and output D2H "
                    "j+1 overlaps CPU publication j"
                    if segmented_input
                    else "input microbatches remain serial"
                )
            ),
            "collective_order": (
                "only the main thread issues D2 collectives in "
                "unchanged group, microbatch, counts, ids, vectors order"
            ),
            "collective_timing": (
                "pipelined modes use current-compute-stream CUDA "
                "events so "
                "collective accounting does not fence copy streams"
            ),
            "non_overlappable": (
                "initial input fill, input-ready tails, bounded "
                "one-drain output-credit waits, and final drain"
            ),
        },
        "exactly_once_pass": (
            len(observed) == len(set(observed))
            and set(observed) == expected_ids
            and target_ledger.complete_records
            == len(expected_ids)
            and target_ledger.partial_records == 0
            and target_ledger.missing_records == 0
        ),
    }


def _validate_visible_devices(
    expected: str,
) -> tuple[str, ...]:
    visible = tuple(
        value.strip()
        for value in os.environ.get(
            "CUDA_VISIBLE_DEVICES",
            "",
        ).split(",")
        if value.strip()
    )
    expected_tokens = tuple(
        value.strip()
        for value in expected.split(",")
        if value.strip()
    )
    if visible != expected_tokens or len(visible) != 2:
        raise RuntimeError("M1 physical GPU pair differs")
    return visible


def _store_paths(
    args: argparse.Namespace,
    rank: int,
) -> tuple[Path, Path]:
    directory = _path(args.store_dir) / args.run_id / args.scope
    return (
        directory / f"rank{rank}.old.bin",
        directory / f"rank{rank}.target.bin",
    )


def _old_store_binding_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.binding.json")


def _old_store_binding(
    store: PageableDramExtentStore,
    rank: int,
    prepared_path: Path,
    action_path: Path,
    snapshot: Mapping[str, object],
    training_path: Path,
    source_checkpoint: str | Path,
    owner_sha256: str,
) -> dict[str, object]:
    checkpoint_sha256 = (
        file_sha256(source_checkpoint)
        if isinstance(source_checkpoint, Path)
        else source_checkpoint
    )
    if len(checkpoint_sha256) != 64:
        raise ValueError("M1 source checkpoint identity is invalid")
    return {
        "protocol": f"{PROTOCOL}_old_store_binding_v0",
        "rank": rank,
        "source_version": "theta0",
        "dtype": "float16",
        "store_layout_sha256": store.layout_sha256,
        "record_ids_sha256": canonical_sha256(
            {"record_ids": list(store.record_ids)}
        ),
        "prepared_data_sha256": file_sha256(prepared_path),
        "action_snapshot_sha256": file_sha256(action_path),
        "owner_independent_plan_sha256": snapshot[
            "owner_independent_plan_sha256"
        ],
        "owner_map_sha256": owner_sha256,
        "training_result_sha256": file_sha256(training_path),
        "source_checkpoint_sha256": checkpoint_sha256,
    }


def _write_old_store_binding(
    path: Path,
    binding: Mapping[str, object],
) -> None:
    output = _old_store_binding_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target:
        target.write(canonical_json_bytes(dict(binding)))


def _validate_old_store_binding(
    path: Path,
    expected: Mapping[str, object],
) -> None:
    observed = _load_json(_old_store_binding_path(path))
    if observed != dict(expected):
        raise ValueError("M1 reusable old store binding differs")


def _open_or_create_old_store(
    path: Path,
    actions: Sequence[M1Action],
    cfg: HSTUConfig,
    reuse_complete: bool,
) -> tuple[PageableDramExtentStore, bool]:
    ids = tuple(value.record_id for value in actions)
    lengths = tuple(value.old_tokens for value in actions)
    if reuse_complete:
        store = PageableDramExtentStore.open(
            path,
            ids,
            lengths,
            num_layers=cfg.num_layers,
            width=cfg.hidden_size,
        )
        ledger = store.ledger()
        if (
            ledger.complete_records != len(ids)
            or ledger.partial_records
            or ledger.missing_records
        ):
            store.close()
            raise ValueError("M1 reusable old store is incomplete")
        return store, True
    return (
        PageableDramExtentStore.create(
            path,
            ids,
            lengths,
            num_layers=cfg.num_layers,
            width=cfg.hidden_size,
        ),
        False,
    )


def _load_program(
    compiler_path: Path,
    action_path: Path,
    snapshot: Mapping[str, object],
    cfg: HSTUConfig,
):
    compiler = _load_json(compiler_path)
    action_binding = compiler.get("action_snapshot")
    descriptor = compiler.get("program")
    if (
        compiler.get("status") != "complete"
        or not isinstance(action_binding, dict)
        or action_binding.get("sha256") != file_sha256(action_path)
        or action_binding.get("owner_independent_plan_sha256")
        != snapshot["owner_independent_plan_sha256"]
        or not isinstance(descriptor, dict)
    ):
        raise ValueError("M1 compiler binding differs")
    program, loaded = load_direct_oldkv_program(
        _path(str(descriptor["path"])),
        expected_sha256=str(descriptor["sha256"]),
        expected_source_version="theta0",
        expected_target_version="theta1",
        expected_num_layers=cfg.num_layers,
        expected_kv_width=cfg.hidden_size,
    )
    return program, loaded


def _warmup_operator(
    operator: DirectOldKVFusedOperator,
    program,
    device: torch.device,
) -> float:
    lengths = torch.ones(1, dtype=torch.long, device=device)
    offsets = torch.tensor((0, 1), dtype=torch.long, device=device)
    source = JaggedMigratedKVBatch(
        record_ids=(0,),
        migration_anchor_version="theta0",
        served_kv_target="theta0",
        k=torch.zeros(
            (program.num_layers, 1, program.kv_width),
            dtype=torch.float16,
            device=device,
        ),
        v=torch.zeros(
            (program.num_layers, 1, program.kv_width),
            dtype=torch.float16,
            device=device,
        ),
        lengths=lengths,
        offsets=offsets,
    )
    destination = _device_destination(source, "theta1")
    started = time.perf_counter()
    operator.execute_into(program, source, destination)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    del source, destination
    return elapsed


def run_distributed(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    runtime = init_d2_distributed_runtime(
        timeout_seconds=args.timeout_seconds
    )
    old_store = None
    target_store = None
    try:
        if (
            runtime.world_size != 2
            or runtime.backend != "nccl"
            or runtime.device.type != "cuda"
        ):
            raise RuntimeError("M1 requires two torchrun NCCL ranks")
        visible_tokens = _validate_visible_devices(
            args.expected_visible_devices
        )
        d3_execution_identity = (
            _d3_execution_identity(args, visible_tokens)
            if args.mode == "d3"
            else None
        )
        residency_plan, residency_plan_file_sha256 = (
            _load_distributed_residency_plan(
                args.residency_plan,
                runtime,
            )
        )
        prepared_path = _path(args.prepared_data)
        action_path = _path(args.action_snapshot)
        training_path = _path(args.training_result)
        checkpoint_dir = _path(args.checkpoint_dir)
        snapshot, all_actions = load_action_snapshot(action_path)
        training = _load_json(training_path)
        if (
            training.get("status") != "complete"
            or snapshot.get("prepared_data_sha256")
            != file_sha256(prepared_path)
        ):
            raise ValueError("M1 data/training boundary differs")
        cfg = training_model_config(training)
        execution_granularity = resolve_execution_granularity(
            args,
            residency_plan,
        )
        if max(
            value
            for route in (
                execution_granularity.compiled,
                execution_granularity.exact,
            )
            for value in route.to_dict().values()
        ) > args.group_records_per_rank:
            raise ValueError(
                "M1 residency execution granularity exceeds its group"
            )
        layout = snapshot.get("layout")
        if (
            not isinstance(layout, Mapping)
            or cfg.max_seq_len != int(layout["history_tokens"])
            or cfg.head_dim is None
            or cfg.num_heads * cfg.head_dim != cfg.hidden_size
        ):
            raise ValueError("M1 QK model shape differs")
        if (
            args.scope == "full"
            and cfg.num_items == 512144
            and not args.allow_functional_scale_full
        ):
            raise ValueError(
                "512144-row embedding is a functional canary; "
                "full execution requires an explicit override"
            )
        owners = build_owner_map(all_actions, runtime.world_size)
        actions = select_actions(
            all_actions,
            owners,
            runtime.world_size,
            args.scope,
            args.canary_records_per_route_per_rank,
        )
        actions_by_id = {
            value.record_id: value for value in actions
        }
        selected_owner_sha256 = owner_map_sha256(
            {
                value.record_id: owners[value.record_id]
                for value in actions
            }
        )
        original_groups = build_s0_groups(
            actions,
            owners,
            runtime.world_size,
            args.group_records_per_rank,
        )
        original_group_sha256 = group_plan_sha256(original_groups)
        stack_bindings = {
            "prepared_data": str(prepared_path),
            "prepared_data_sha256": file_sha256(prepared_path),
            "action_snapshot": str(action_path),
            "action_snapshot_sha256": file_sha256(action_path),
            "owner_independent_plan_sha256": snapshot[
                "owner_independent_plan_sha256"
            ],
            "owner_map_sha256": selected_owner_sha256,
            "group_plan_sha256": original_group_sha256,
            "training_result": str(training_path),
            "training_result_sha256": file_sha256(training_path),
        }
        if d3_execution_identity is not None:
            stack_bindings.update(
                {
                    "compiler_result": d3_execution_identity[
                        "compiler_result"
                    ],
                    "compiler_result_sha256": d3_execution_identity[
                        "compiler_result_sha256"
                    ],
                    "program_path": d3_execution_identity["program_path"],
                    "program_sha256": d3_execution_identity[
                        "program_sha256"
                    ],
                }
            )
        if residency_plan is not None:
            if (
                residency_plan.group_plan_sha256
                != original_group_sha256
                or residency_plan.original_group_order
                != tuple(value.ordinal for value in original_groups)
            ):
                raise ValueError("M1 residency plan group boundary differs")
            groups = reorder_groups(
                original_groups,
                residency_plan.launch_order,
            )
        else:
            groups = original_groups
        local_actions = tuple(
            value
            for value in actions
            if owners[value.record_id] == runtime.rank
        )
        ids_by_rank = tuple(
            tuple(
                value.record_id
                for value in actions
                if owners[value.record_id] == rank
            )
            for rank in range(runtime.world_size)
        )
        materialize_batches = _aligned_batches(
            ids_by_rank,
            args.materialize_records_per_rank,
        )
        histories = load_histories(
            prepared_path,
            local_actions,
            snapshot,
        )
        max_old_tokens = max(
            value.old_tokens for value in local_actions
        )
        max_final_tokens = max(
            value.final_tokens for value in local_actions
        )
        max_compiled_retained_tokens = max(
            (
                value.retained_tokens
                for value in local_actions
                if value.route == "compiled"
            ),
            default=1,
        )
        slot_capacity = _execution_slot_capacity(
            execution_granularity,
            args.materialize_records_per_rank,
            max_old_tokens,
            max_final_tokens,
            max_compiled_retained_tokens,
        )
        if residency_plan is not None:
            rank = runtime.rank
            total_hbm = torch.cuda.get_device_properties(
                runtime.device
            ).total_memory
            required_pinned = 2 * _pinned_slot_nbytes(
                cfg,
                slot_capacity,
            )
            capacity_errors = []
            if (
                len(
                    residency_plan.projected_peak_hbm_reserved_bytes_by_rank
                )
                != runtime.world_size
                or len(residency_plan.projected_pinned_bytes_by_rank)
                != runtime.world_size
                or len(residency_plan.hbm_total_bytes_by_rank)
                != runtime.world_size
                or len(residency_plan.pinned_limit_bytes_by_rank)
                != runtime.world_size
            ):
                capacity_errors.append("rank_count")
            else:
                if residency_plan.hbm_total_bytes_by_rank[rank] != total_hbm:
                    capacity_errors.append("hbm_total")
                if (
                    residency_plan.projected_peak_hbm_reserved_bytes_by_rank[
                        rank
                    ]
                    + residency_plan.hbm_margin_bytes
                    > total_hbm
                ):
                    capacity_errors.append("hbm_projection")
                if (
                    required_pinned
                    > residency_plan.projected_pinned_bytes_by_rank[rank]
                ):
                    capacity_errors.append("pinned_projection")
                if (
                    residency_plan.projected_pinned_bytes_by_rank[rank]
                    > residency_plan.pinned_limit_bytes_by_rank[rank]
                ):
                    capacity_errors.append("pinned_limit")
            local_capacity = {
                "rank": rank,
                "errors": capacity_errors,
                "total_hbm_bytes": total_hbm,
                "required_pinned_bytes": required_pinned,
            }
            gathered_capacity: list[object] = [
                None
            ] * runtime.world_size
            dist.all_gather_object(gathered_capacity, local_capacity)
            if any(
                bool(dict(value)["errors"])
                for value in gathered_capacity
            ):
                raise ValueError(
                    "M1 residency plan exceeds runtime capacity: "
                    + json.dumps(gathered_capacity, sort_keys=True)
                )
        slot = _allocate_slot(
            cfg,
            slot_capacity.max_rows,
            slot_capacity.max_source_tokens,
            slot_capacity.max_output_tokens,
            slot_capacity.max_history_tokens,
        )
        execution_slots = (
            (
                slot,
                _allocate_slot(
                    cfg,
                    slot_capacity.max_rows,
                    slot_capacity.max_source_tokens,
                    slot_capacity.max_output_tokens,
                    slot_capacity.max_history_tokens,
                ),
            )
            if args.mode in {"s1", "d3"}
            else (slot,)
        )
        if residency_plan is not None and sum(
            value.nbytes for value in execution_slots
        ) != required_pinned:
            raise RuntimeError("M1 pinned-slot capacity accounting differs")
        old_path, target_path = _store_paths(args, runtime.rank)
        source_checkpoint = resolve_version_checkpoint(
            training,
            checkpoint_dir,
            0,
            verify_shard_ranks=(runtime.rank,),
        )
        target_checkpoint = resolve_version_checkpoint(
            training,
            checkpoint_dir,
            1,
            verify_shard_ranks=(runtime.rank,),
        )
        properties = torch.cuda.get_device_properties(runtime.device)
        local_device_identity = {
            "device_name": torch.cuda.get_device_name(runtime.device),
            "device_uuid": str(properties.uuid),
            "pci_domain_id": properties.pci_domain_id,
            "pci_bus_id": properties.pci_bus_id,
            "pci_device_id": properties.pci_device_id,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
            "total_memory": properties.total_memory,
            "physical_visible_token": visible_tokens[runtime.rank],
        }
        if residency_plan is not None:
            if d3_execution_identity is None:
                raise RuntimeError("M1 D3 execution identity is missing")
            local_stack_identity = {
                **local_device_identity,
                "source_checkpoint_sha256": source_checkpoint.descriptor()[
                    "sha256"
                ],
                "target_checkpoint_sha256": target_checkpoint.descriptor()[
                    "sha256"
                ],
            }
            gathered_stack_identities: list[object] = [
                None
            ] * runtime.world_size
            dist.all_gather_object(
                gathered_stack_identities,
                local_stack_identity,
            )
            stack_identities = [
                dict(value) for value in gathered_stack_identities
            ]
            observed_stack_sha256 = d3_stack_revision_sha256(
                bindings=stack_bindings,
                world_size=runtime.world_size,
                group_records_per_rank=args.group_records_per_rank,
                records=len(actions),
                counts={
                    route: sum(
                        value.route == route for value in actions
                    )
                    for route in ("compiled", "exact")
                },
                embedding_scale_role=(
                    "functional_canary_not_primary_d2_partition_evidence"
                    if cfg.num_items == 512144
                    else "large_qk_entity_primary_m1_candidate"
                ),
                device_names=tuple(
                    str(value["device_name"])
                    for value in stack_identities
                ),
                source_checkpoint_sha256=tuple(
                    str(value["source_checkpoint_sha256"])
                    for value in stack_identities
                ),
                target_checkpoint_sha256=tuple(
                    str(value["target_checkpoint_sha256"])
                    for value in stack_identities
                ),
                capacity_groups=capacity_group_projection(
                    original_groups,
                    actions_by_id,
                    cfg,
                ),
                execution_identity={
                    **d3_execution_identity,
                    "devices": stack_identities,
                },
            )
            if (
                observed_stack_sha256
                != residency_plan.stack_revision_sha256
            ):
                raise ValueError("M1 residency stack revision differs")
        old_store, reused_old = _open_or_create_old_store(
            old_path,
            local_actions,
            cfg,
            args.reuse_complete_old_store,
        )
        old_binding = _old_store_binding(
            old_store,
            runtime.rank,
            prepared_path,
            action_path,
            snapshot,
            training_path,
            source_checkpoint.identity_sha256,
            selected_owner_sha256,
        )
        if reused_old:
            _validate_old_store_binding(old_path, old_binding)
        source_materialization = {
            "reused": reused_old,
            "records": len(local_actions),
            "written_bytes": 0,
        }
        if not reused_old:
            source_model = load_runtime_sharded_hstu(
                cfg,
                source_checkpoint,
                runtime.rank,
                runtime.world_size,
                runtime.device,
            )
            dist.barrier()
            source_materialization = {
                "reused": False,
                **materialize_old_store(
                    old_store,
                    materialize_batches,
                    runtime.rank,
                    actions_by_id,
                    histories,
                    source_model,
                    slot,
                    runtime.device,
                ),
            }
            del source_model
            gc.collect()
            torch.cuda.empty_cache()
            _write_old_store_binding(old_path, old_binding)
        old_ledger = old_store.ledger()
        if (
            old_ledger.complete_records != len(local_actions)
            or old_ledger.partial_records
            or old_ledger.missing_records
        ):
            raise RuntimeError("M1 old store coverage differs")
        dist.barrier()
        if args.mode == "materialize":
            local_report = {
                "rank": runtime.rank,
                "records": len(local_actions),
                "source_checkpoint": source_checkpoint.descriptor(),
                "source_materialization": source_materialization,
                "old_store_ledger": old_ledger.to_dict(),
                "old_store_binding": {
                    "path": str(_old_store_binding_path(old_path)),
                    "sha256": file_sha256(
                        _old_store_binding_path(old_path)
                    ),
                },
            }
        else:
            target_store = PageableDramExtentStore.create(
                target_path,
                tuple(value.record_id for value in local_actions),
                tuple(value.final_tokens for value in local_actions),
                num_layers=cfg.num_layers,
                width=cfg.hidden_size,
            )
            prefault_pages = (
                target_store.prefault(write=True)
                if args.prefault_target_store
                else 0
            )
            target_model = load_runtime_sharded_hstu(
                cfg,
                target_checkpoint,
                runtime.rank,
                runtime.world_size,
                runtime.device,
            )
            program_cpu, loaded_program = _load_program(
                _path(args.compiler_result),
                action_path,
                snapshot,
                cfg,
            )
            operator = DirectOldKVFusedOperator()
            program = operator.prepare_program(
                program_cpu,
                runtime.device,
            )
            del program_cpu
            operator_warmup_seconds = _warmup_operator(
                operator,
                program,
                runtime.device,
            )
            dist.barrier()
            execution_result = (
                run_s1(
                    groups,
                    runtime.rank,
                    actions_by_id,
                    histories,
                    old_store,
                    target_store,
                    target_model,
                    operator,
                    program,
                    execution_slots,
                    runtime.device,
                    execution_granularity,
                    segmented_input=args.mode == "d3",
                )
                if args.mode in {"s1", "d3"}
                else run_s0(
                    groups,
                    runtime.rank,
                    actions_by_id,
                    histories,
                    old_store,
                    target_store,
                    target_model,
                    operator,
                    program,
                    slot,
                    runtime.device,
                    args.micro_batch_records,
                )
            )
            local_report = {
                "rank": runtime.rank,
                "local_rank": runtime.local_rank,
                "physical_visible_token": visible_tokens[
                    runtime.local_rank
                ],
                "device_name": torch.cuda.get_device_name(
                    runtime.device
                ),
                "device_identity": local_device_identity,
                "records": len(local_actions),
                "source_checkpoint": source_checkpoint.descriptor(),
                "target_checkpoint": target_checkpoint.descriptor(),
                "compiled": sum(
                    value.route == "compiled"
                    for value in local_actions
                ),
                "exact": sum(
                    value.route == "exact"
                    for value in local_actions
                ),
                "source_materialization": source_materialization,
                "old_store_binding": {
                    "path": str(_old_store_binding_path(old_path)),
                    "sha256": file_sha256(
                        _old_store_binding_path(old_path)
                    ),
                },
                "target_prefault_pages": prefault_pages,
                "loaded_program": loaded_program,
                "operator_warmup_seconds": operator_warmup_seconds,
                "pinned_slot_capacity": slot_capacity.to_dict(),
                args.mode: execution_result,
            }
            del target_model, program
        gathered: list[object] = [None] * runtime.world_size
        dist.all_gather_object(gathered, local_report)
        if not runtime.is_primary:
            return None
        rank_reports = [dict(value) for value in gathered]
        complete = all(
            (
                value["old_store_ledger"]["complete_records"]
                == value["records"]
                if args.mode == "materialize"
                else value[args.mode]["exactly_once_pass"]
            )
            for value in rank_reports
        )
        projection = capacity_projection(
            actions,
            owners,
            runtime.world_size,
            cfg,
        )
        projection["capacity_groups"] = capacity_group_projection(
            original_groups,
            actions_by_id,
            cfg,
        )
        report = {
            "protocol": (
                D3_RESIDENCY_PROTOCOL
                if residency_plan is not None
                else D3_PROTOCOL
                if args.mode == "d3"
                else S1_PROTOCOL if args.mode == "s1" else PROTOCOL
            ),
            "status": "complete" if complete else "failed",
            "scientific_result": False,
            "formal_design3": False,
            "mode": args.mode,
            "scope": args.scope,
            "benchmark_role": (
                "two_gpu_qk_pageable_dram_group_at_a_time_s0_foundation"
                if args.mode == "s0"
                else (
                    (
                        "two_gpu_qk_route_aware_residency_d3_proposed"
                        if residency_plan is not None
                        else "two_gpu_qk_segmented_io_pingpong_d3_proposed"
                    )
                    if args.mode == "d3"
                    else "two_gpu_qk_strong_group_double_buffer_s1_baseline"
                    if args.mode == "s1"
                    else "two_gpu_qk_complete_oldkv_materialization_foundation"
                )
            ),
            "embedding_scale_role": (
                "functional_canary_not_primary_d2_partition_evidence"
                if cfg.num_items == 512144
                else "large_qk_entity_primary_m1_candidate"
            ),
            "execution": {
                "world_size": runtime.world_size,
                "physical_visible_devices": list(visible_tokens),
                "owner_policy": "stable_record_modulo",
                "group_order": (
                    "residency_plan_stable_route_interleave"
                    if residency_plan is not None
                    else "sequential_route_pure"
                ),
                "group_launch_ordinals": [
                    value.ordinal for value in groups
                ],
                "buffer_depth": (
                    2 if args.mode in {"s1", "d3"} else 1
                ),
                "input_pipeline": (
                    "segmented_microbatch_pingpong"
                    if args.mode == "d3"
                    else "serial_microbatch"
                ),
                "output_pipeline": (
                    "segmented_microbatch_pingpong"
                    if args.mode == "d3"
                    else "serial_microbatch"
                ),
                "output_drain_credit": (
                    1 if args.mode in {"s1", "d3"} else 0
                ),
                "bounded_pinned_staging": True,
                "source_store": "complete_theta0_oldkv_fp16",
                "timed_source_read": (
                    "compiled_retained_extent_only"
                    if args.mode in {"s0", "s1", "d3"}
                    else None
                ),
                "target_store": "complete_private_theta1_kv_fp16",
                "source_materialization_in_primary_timer": False,
                "operator_warmup_in_primary_timer": False,
                "target_prefaulted": args.prefault_target_store,
                "functional_scale_full_override": (
                    args.allow_functional_scale_full
                ),
            },
            "records": len(actions),
            "counts": {
                route: sum(
                    value.route == route for value in actions
                )
                for route in ("compiled", "exact")
            },
            "groups": len(groups),
            "group_records_per_rank": args.group_records_per_rank,
            "micro_batch_records": args.micro_batch_records,
            "execution_granularity": (
                execution_granularity.to_dict()
            ),
            "development_identity": d3_execution_identity,
            "residency_plan": (
                None
                if residency_plan is None
                else {
                    "path": str(_path(args.residency_plan)),
                    "file_sha256": residency_plan_file_sha256,
                    "content_sha256": residency_plan.content_sha256,
                    "stack_revision_sha256": (
                        residency_plan.stack_revision_sha256
                    ),
                    "predicted_makespan_seconds": (
                        residency_plan.predicted_makespan_seconds
                    ),
                    "original_group_order": list(
                        residency_plan.original_group_order
                    ),
                    "launch_order": list(
                        residency_plan.launch_order
                    ),
                }
            ),
            "materialize_records_per_rank": (
                args.materialize_records_per_rank
            ),
            "capacity": projection,
            "bindings": stack_bindings,
            "rank_reports": rank_reports,
            "makespan_seconds": (
                max(
                    value[args.mode]["wall_seconds"]
                    for value in rank_reports
                )
                if args.mode in {"s0", "s1", "d3"}
                else max(
                    value["source_materialization"]["wall_seconds"]
                    for value in rank_reports
                )
            ),
            "exactly_once_pass": complete,
            "next": (
                "measure the full same-revision S0 movement fraction"
                if args.scope == "canary"
                else (
                    "compare S1 against the same-revision S0 boundary"
                    if args.mode in {"s1", "d3"}
                    else "test a same-revision double buffer only after "
                    "confirming the movement bottleneck"
                )
            ),
        }
        output = _path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(report))
        return report
    finally:
        if target_store is not None:
            target_store.close()
        if old_store is not None:
            old_store.close()
        close_d2_distributed_runtime(runtime)


def run(args: argparse.Namespace) -> dict[str, object] | None:
    validate_args(args)
    if args.mode == "dry-run":
        report = build_dry_run_report(args)
        output = _path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(report))
        return report
    return run_distributed(args)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    if report is not None:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "mode": report["mode"],
                    "scope": report["scope"],
                    "records": report["records"],
                    "output": str(_path(args.output)),
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
