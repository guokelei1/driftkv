from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import time
import uuid
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration.cohort_jagged import JaggedMigratedKVBatch
from hstu_kvcache.migration.design2_distributed import (
    close_d2_distributed_runtime,
    init_d2_distributed_runtime,
)
from hstu_kvcache.migration.design2_embedding import (
    build_modulo_sharded_hstu_from_cpu,
)
from hstu_kvcache.migration.design2_integrated import (
    INTEGRATED_ROUTES,
    D2IntegratedExtent,
    IntegratedAppendOnlyKVBatch,
    IntegratedLookupMetrics,
    build_integrated_exact_pool_schedule,
    build_integrated_schedule,
    integrated_exact_reason_counts,
    integrated_lookup_token_ledger,
    integrated_route,
    integrated_sharded_append,
    integrated_sharded_append_only,
    integrated_sharded_exact,
    select_integrated_records,
    slice_integrated_jagged_ranges,
)
from hstu_kvcache.migration.design2_plan import (
    D2ActionPlan,
    D2ActionRecord,
    build_d2_record_owner_map,
    canonical_sha256,
    d2_record_owner_map_sha256,
    file_sha256,
)
from hstu_kvcache.migration.recompute import RawHistoryBatch
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    load_direct_oldkv_program,
)
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import (
    reconstruct_organic_windows,
    validate_long_context_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cohortkv_design2_integrated_w3_development_v5"
DEFAULT_ACTION_PLAN = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_STAGE_A_SUMMARY = "configs/cohortkv_d2/stage_a_summary.json"
DEFAULT_TRAINING = (
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
DEFAULT_CHECKPOINT_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--stage-a-summary", default=DEFAULT_STAGE_A_SUMMARY)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--cohort",
        choices=("pilot192", "full682"),
        default="pilot192",
    )
    parser.add_argument("--extent-size", type=int, default=16)
    parser.add_argument(
        "--compiled-order",
        choices=("final_length", "suffix_retained"),
        default="final_length",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--include-two-stage-all-exact",
        action="store_true",
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--expected-visible-devices",
        default="0,1,3",
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _checkpoint_path(checkpoint_dir: Path, version: str) -> Path:
    return checkpoint_dir / f"theta_{int(version.removeprefix('theta'))}.pt"


def _load_cpu_model(cfg: HSTUConfig, checkpoint: Path) -> HSTU:
    model = HSTU(cfg)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def _program_descriptor(
    stage_a: dict[str, object],
) -> dict[str, object]:
    capacity = json.loads(
        _path(stage_a["artifacts"]["capacity"]["path"]).read_text()
    )
    return capacity["program"]


def _checkpoint_descriptor(
    plan: D2ActionPlan,
    version: str,
) -> dict[str, object]:
    upstream = json.loads(_path(plan.provenance.artifact).read_text())
    return next(
        value
        for value in upstream["input_provenance"]["checkpoints"]
        if value["version"] == version
    )


def _actions_for_extent(
    extent: D2IntegratedExtent,
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
) -> tuple[D2ActionRecord, ...]:
    return tuple(
        actions_by_id[value]
        for value in extent.local_record_ids(rank)
    )


def _history_batch(
    actions: tuple[D2ActionRecord, ...],
    window,
    starts: tuple[int, ...],
    stops: tuple[int, ...],
    version: str,
    device: torch.device,
) -> RawHistoryBatch:
    if len(actions) != len(starts) or len(actions) != len(stops):
        raise ValueError("integrated history ranges differ")
    lengths = tuple(
        stop - start
        for start, stop in zip(starts, stops, strict=True)
    )
    if any(value < 0 for value in lengths):
        raise ValueError("integrated history range is negative")
    width = max(max(lengths, default=0), 1)
    item_ids = torch.zeros((len(actions), width), dtype=torch.long)
    behaviors = torch.zeros_like(item_ids)
    time_deltas = torch.zeros(
        (len(actions), width),
        dtype=torch.float32,
    )
    for row, (action, start, stop) in enumerate(
        zip(actions, starts, stops, strict=True)
    ):
        history = window.records[action.prepared_user_id].history
        if history is None or not 0 <= start <= stop <= len(history):
            raise ValueError("integrated history range exceeds its record")
        length = stop - start
        if length:
            item_ids[row, :length].copy_(
                torch.from_numpy(history.item_ids[start:stop].copy())
            )
            behaviors[row, :length].copy_(
                torch.from_numpy(history.behaviors[start:stop].copy())
            )
            time_deltas[row, :length].copy_(
                torch.from_numpy(history.time_deltas[start:stop].copy())
            )
    return RawHistoryBatch(
        record_ids=tuple(value.record_id for value in actions),
        migration_anchor_version=version,
        item_ids=item_ids.to(device),
        behaviors=behaviors.to(device),
        time_deltas=time_deltas.to(device),
        lengths=torch.tensor(lengths, dtype=torch.long, device=device),
    )


def _batch_key(extent: D2IntegratedExtent) -> tuple[str, int]:
    return extent.route, extent.ordinal


def _prepare_source_batches(
    schedule: tuple[D2IntegratedExtent, ...],
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
    source_window,
    source_version: str,
    device: torch.device,
) -> dict[tuple[str, int], RawHistoryBatch]:
    output = {}
    for extent in schedule:
        actions = _actions_for_extent(extent, rank, actions_by_id)
        output[_batch_key(extent)] = _history_batch(
            actions,
            source_window,
            (0,) * len(actions),
            tuple(value.old_tokens for value in actions),
            source_version,
            device,
        )
    return output


def _prepare_target_batches(
    route_schedule: tuple[D2IntegratedExtent, ...],
    all_schedule: tuple[D2IntegratedExtent, ...],
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
    target_window,
    target_version: str,
    device: torch.device,
) -> dict[str, dict[tuple[str, int], RawHistoryBatch]]:
    output: dict[str, dict[tuple[str, int], RawHistoryBatch]] = {
        "retained": {},
        "delta": {},
        "fused_suffix": {},
        "latest_route": {},
        "prefix_route": {},
        "full_route": {},
        "full": {},
        "prefix_all": {},
        "latest_all": {},
    }
    for extent in route_schedule:
        actions = _actions_for_extent(extent, rank, actions_by_id)
        key = _batch_key(extent)
        output["latest_route"][key] = _history_batch(
            actions,
            target_window,
            tuple(value.target_prefix_tokens for value in actions),
            tuple(value.final_tokens for value in actions),
            target_version,
            device,
        )
        output["full_route"][key] = _history_batch(
            actions,
            target_window,
            (0,) * len(actions),
            tuple(value.final_tokens for value in actions),
            target_version,
            device,
        )
        if extent.route in {"compiled", "scheduled_exact"}:
            output["delta"][key] = _history_batch(
                actions,
                target_window,
                tuple(value.delta_start for value in actions),
                tuple(value.target_prefix_tokens for value in actions),
                target_version,
                device,
            )
        if extent.route == "compiled":
            output["fused_suffix"][key] = _history_batch(
                actions,
                target_window,
                tuple(value.delta_start for value in actions),
                tuple(value.final_tokens for value in actions),
                target_version,
                device,
            )
        if extent.route == "scheduled_exact":
            output["retained"][key] = _history_batch(
                actions,
                target_window,
                (0,) * len(actions),
                tuple(value.retained_tokens for value in actions),
                target_version,
                device,
            )
        if extent.route == "natural_exact":
            output["prefix_route"][key] = _history_batch(
                actions,
                target_window,
                (0,) * len(actions),
                tuple(value.target_prefix_tokens for value in actions),
                target_version,
                device,
            )
    for extent in all_schedule:
        actions = _actions_for_extent(extent, rank, actions_by_id)
        key = _batch_key(extent)
        output["full"][key] = _history_batch(
            actions,
            target_window,
            (0,) * len(actions),
            tuple(value.final_tokens for value in actions),
            target_version,
            device,
        )
        output["prefix_all"][key] = _history_batch(
            actions,
            target_window,
            (0,) * len(actions),
            tuple(value.target_prefix_tokens for value in actions),
            target_version,
            device,
        )
        output["latest_all"][key] = _history_batch(
            actions,
            target_window,
            tuple(value.target_prefix_tokens for value in actions),
            tuple(value.final_tokens for value in actions),
            target_version,
            device,
        )
    return output


def _prepare_merged_exact_batches(
    exact_schedule: tuple[D2IntegratedExtent, ...],
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
    target_window,
    target_version: str,
    device: torch.device,
) -> dict[tuple[str, int], RawHistoryBatch]:
    output = {}
    for extent in exact_schedule:
        actions = _actions_for_extent(extent, rank, actions_by_id)
        output[_batch_key(extent)] = _history_batch(
            actions,
            target_window,
            (0,) * len(actions),
            tuple(value.final_tokens for value in actions),
            target_version,
            device,
        )
    return output


def _lookup_batches_for_method(
    method: str,
    route_schedule: tuple[D2IntegratedExtent, ...],
    all_schedule: tuple[D2IntegratedExtent, ...],
    merged_exact_schedule: tuple[D2IntegratedExtent, ...],
    inputs: dict[str, dict[tuple[str, int], RawHistoryBatch]],
) -> tuple[RawHistoryBatch, ...]:
    batches = []
    if method == "owner_mixed":
        for extent in route_schedule:
            key = _batch_key(extent)
            if extent.route == "scheduled_exact":
                batches.append(inputs["retained"][key])
            elif extent.route == "natural_exact":
                batches.append(inputs["prefix_route"][key])
            if extent.route in {"compiled", "scheduled_exact"}:
                batches.append(inputs["delta"][key])
            batches.append(inputs["latest_route"][key])
    elif method in {
        "owner_mixed_fused_finalization",
        "owner_mixed_fused_append_only",
    }:
        for extent in route_schedule:
            key = _batch_key(extent)
            if extent.route == "compiled":
                batches.append(inputs["fused_suffix"][key])
            else:
                batches.append(inputs["full_route"][key])
    elif method == "owner_mixed_fused_append_only_merged_exact":
        batches.extend(
            inputs["fused_suffix"][_batch_key(extent)]
            for extent in route_schedule
            if extent.route == "compiled"
        )
        batches.extend(
            inputs["merged_exact"][_batch_key(extent)]
            for extent in merged_exact_schedule
        )
    elif method == "one_shot_all_exact":
        batches.extend(
            inputs["full"][_batch_key(extent)]
            for extent in all_schedule
        )
    elif method == "two_stage_all_exact":
        for extent in all_schedule:
            key = _batch_key(extent)
            batches.extend(
                (
                    inputs["prefix_all"][key],
                    inputs["latest_all"][key],
                )
            )
    else:
        raise ValueError("integrated lookup ledger method differs")
    return tuple(batches)


def _lookup_multiset(
    batches: tuple[RawHistoryBatch, ...],
) -> dict[str, object]:
    requested = []
    for batch in batches:
        valid = (
            torch.arange(
                batch.item_ids.shape[1],
                device=batch.device,
            ).unsqueeze(0)
            < batch.lengths.long().unsqueeze(1)
        )
        requested.append(batch.item_ids[valid].long())
    joined = torch.cat(requested).cpu()
    ordered = torch.sort(joined).values.contiguous()
    digest = hashlib.sha256()
    digest.update(b"cohortkv-d2-integrated-lookup-multiset-v1")
    digest.update(struct.pack("<Q", ordered.numel()))
    digest.update(ordered.numpy().tobytes())
    return {
        "requested_tokens": ordered.numel(),
        "unique_tokens": torch.unique_consecutive(ordered).numel(),
        "multiset_sha256": digest.hexdigest(),
    }


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _lookup_payload(
    phase: str,
    metrics: IntegratedLookupMetrics,
) -> dict[str, object]:
    return {"phase": phase, **metrics.to_dict()}


def _timed_phase(
    phase: str,
    device: torch.device,
    phase_seconds: dict[str, float],
    action,
):
    _synchronize(device)
    started = time.perf_counter()
    result = action()
    _synchronize(device)
    phase_seconds[phase] += time.perf_counter() - started
    return result


def _direct_destination(
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


def _execute_owner_mixed(
    route_schedule: tuple[D2IntegratedExtent, ...],
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
    source_fragments: dict[
        tuple[str, int],
        JaggedMigratedKVBatch | None,
    ],
    inputs: dict[str, dict[tuple[str, int], RawHistoryBatch]],
    target_model,
    operator: DirectOldKVFusedOperator,
    program,
    target_version: str,
    device: torch.device,
) -> tuple[
    list[JaggedMigratedKVBatch],
    dict[str, float],
    list[dict[str, object]],
]:
    outputs = []
    phase_seconds: dict[str, float] = defaultdict(float)
    lookups = []
    for extent in route_schedule:
        key = _batch_key(extent)
        actions = _actions_for_extent(extent, rank, actions_by_id)
        prefix = None
        if extent.route == "compiled":
            source = source_fragments[key]

            def compile_retained(
                source_fragment=source,
                batch_actions=actions,
            ):
                if source_fragment is None:
                    return None
                retained = slice_integrated_jagged_ranges(
                    source_fragment,
                    tuple(
                        value.retained_start
                        for value in batch_actions
                    ),
                    tuple(value.old_tokens for value in batch_actions),
                )
                return operator.execute_into(
                    program,
                    retained,
                    _direct_destination(retained, target_version),
                )

            prefix = _timed_phase(
                "compiled_retained",
                device,
                phase_seconds,
                compile_retained,
            )
        elif extent.route == "scheduled_exact":
            exact = _timed_phase(
                "scheduled_exact_retained",
                device,
                phase_seconds,
                lambda batch=inputs["retained"][key]: integrated_sharded_exact(
                    target_model,
                    batch,
                    target_version,
                ),
            )
            prefix = exact.fragment
            lookups.append(
                _lookup_payload(
                    "scheduled_exact_retained",
                    exact.lookup_metrics,
                )
            )
            del exact
        elif extent.route == "natural_exact":
            exact = _timed_phase(
                "natural_exact_prefix",
                device,
                phase_seconds,
                lambda batch=inputs["prefix_route"][key]: integrated_sharded_exact(
                    target_model,
                    batch,
                    target_version,
                ),
            )
            prefix = exact.fragment
            lookups.append(
                _lookup_payload(
                    "natural_exact_prefix",
                    exact.lookup_metrics,
                )
            )
            del exact
        else:
            raise ValueError("integrated mixed route differs")
        if extent.route in {"compiled", "scheduled_exact"}:
            delta = _timed_phase(
                "delta_append",
                device,
                phase_seconds,
                lambda retained=prefix, batch=inputs["delta"][key]: integrated_sharded_append(
                    target_model,
                    retained,
                    batch,
                    target_version,
                ),
            )
            prefix = delta.fragment
            lookups.append(
                _lookup_payload("delta_append", delta.lookup_metrics)
            )
            del delta
        latest = _timed_phase(
            "latest_append",
            device,
            phase_seconds,
            lambda retained=prefix, batch=inputs["latest_route"][key]: integrated_sharded_append(
                target_model,
                retained,
                batch,
                target_version,
            ),
        )
        lookups.append(
            _lookup_payload("latest_append", latest.lookup_metrics)
        )
        if latest.fragment is not None:
            outputs.append(latest.fragment)
        prefix = None
        del latest
    return outputs, dict(phase_seconds), lookups


def _execute_owner_mixed_fused_finalization(
    route_schedule: tuple[D2IntegratedExtent, ...],
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
    source_fragments: dict[
        tuple[str, int],
        JaggedMigratedKVBatch | None,
    ],
    inputs: dict[str, dict[tuple[str, int], RawHistoryBatch]],
    target_model,
    operator: DirectOldKVFusedOperator,
    program,
    target_version: str,
    device: torch.device,
) -> tuple[
    list[JaggedMigratedKVBatch],
    dict[str, float],
    list[dict[str, object]],
]:
    outputs = []
    phase_seconds: dict[str, float] = defaultdict(float)
    lookups = []
    for extent in route_schedule:
        key = _batch_key(extent)
        actions = _actions_for_extent(extent, rank, actions_by_id)
        if extent.route == "compiled":
            source = source_fragments[key]

            def compile_retained(
                source_fragment=source,
                batch_actions=actions,
            ):
                if source_fragment is None:
                    return None
                retained = slice_integrated_jagged_ranges(
                    source_fragment,
                    tuple(
                        value.retained_start
                        for value in batch_actions
                    ),
                    tuple(value.old_tokens for value in batch_actions),
                )
                return operator.execute_into(
                    program,
                    retained,
                    _direct_destination(retained, target_version),
                )

            retained = _timed_phase(
                "compiled_retained",
                device,
                phase_seconds,
                compile_retained,
            )
            finalized = _timed_phase(
                "compiled_finalization_append",
                device,
                phase_seconds,
                lambda cache=retained, batch=inputs["fused_suffix"][key]: integrated_sharded_append(
                    target_model,
                    cache,
                    batch,
                    target_version,
                ),
            )
            lookups.append(
                _lookup_payload(
                    "compiled_finalization_append",
                    finalized.lookup_metrics,
                )
            )
            if finalized.fragment is not None:
                outputs.append(finalized.fragment)
            retained = None
            del finalized
        else:
            phase = f"{extent.route.removesuffix('_exact')}_final_exact"
            exact = _timed_phase(
                phase,
                device,
                phase_seconds,
                lambda batch=inputs["full_route"][key]: integrated_sharded_exact(
                    target_model,
                    batch,
                    target_version,
                ),
            )
            lookups.append(
                _lookup_payload(phase, exact.lookup_metrics)
            )
            if exact.fragment is not None:
                outputs.append(exact.fragment)
            del exact
    return outputs, dict(phase_seconds), lookups


def _execute_owner_mixed_fused_append_only(
    route_schedule: tuple[D2IntegratedExtent, ...],
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
    source_fragments: dict[
        tuple[str, int],
        JaggedMigratedKVBatch | None,
    ],
    inputs: dict[str, dict[tuple[str, int], RawHistoryBatch]],
    target_model,
    operator: DirectOldKVFusedOperator,
    program,
    target_version: str,
    device: torch.device,
) -> tuple[
    list[JaggedMigratedKVBatch | IntegratedAppendOnlyKVBatch],
    dict[str, float],
    list[dict[str, object]],
]:
    outputs: list[
        JaggedMigratedKVBatch | IntegratedAppendOnlyKVBatch
    ] = []
    phase_seconds: dict[str, float] = defaultdict(float)
    lookups = []
    for extent in route_schedule:
        key = _batch_key(extent)
        actions = _actions_for_extent(extent, rank, actions_by_id)
        if extent.route == "compiled":
            source = source_fragments[key]

            def compile_retained(
                source_fragment=source,
                batch_actions=actions,
            ):
                if source_fragment is None:
                    return None
                retained = slice_integrated_jagged_ranges(
                    source_fragment,
                    tuple(
                        value.retained_start
                        for value in batch_actions
                    ),
                    tuple(value.old_tokens for value in batch_actions),
                )
                return operator.execute_into(
                    program,
                    retained,
                    _direct_destination(retained, target_version),
                )

            retained = _timed_phase(
                "compiled_retained",
                device,
                phase_seconds,
                compile_retained,
            )
            finalized = _timed_phase(
                "compiled_finalization_append_only",
                device,
                phase_seconds,
                lambda cache=retained, batch=inputs["fused_suffix"][key]: integrated_sharded_append_only(
                    target_model,
                    cache,
                    batch,
                    target_version,
                ),
            )
            lookups.append(
                _lookup_payload(
                    "compiled_finalization_append_only",
                    finalized.lookup_metrics,
                )
            )
            if finalized.fragment is not None:
                outputs.append(finalized.fragment)
            retained = None
            del finalized
        else:
            phase = f"{extent.route.removesuffix('_exact')}_final_exact"
            exact = _timed_phase(
                phase,
                device,
                phase_seconds,
                lambda batch=inputs["full_route"][key]: integrated_sharded_exact(
                    target_model,
                    batch,
                    target_version,
                ),
            )
            lookups.append(
                _lookup_payload(phase, exact.lookup_metrics)
            )
            if exact.fragment is not None:
                outputs.append(exact.fragment)
            del exact
    return outputs, dict(phase_seconds), lookups


def _execute_owner_mixed_fused_append_only_merged_exact(
    route_schedule: tuple[D2IntegratedExtent, ...],
    merged_exact_schedule: tuple[D2IntegratedExtent, ...],
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
    source_fragments: dict[
        tuple[str, int],
        JaggedMigratedKVBatch | None,
    ],
    inputs: dict[str, dict[tuple[str, int], RawHistoryBatch]],
    target_model,
    operator: DirectOldKVFusedOperator,
    program,
    target_version: str,
    device: torch.device,
) -> tuple[
    list[JaggedMigratedKVBatch | IntegratedAppendOnlyKVBatch],
    dict[str, float],
    list[dict[str, object]],
]:
    outputs: list[
        JaggedMigratedKVBatch | IntegratedAppendOnlyKVBatch
    ] = []
    phase_seconds: dict[str, float] = defaultdict(float)
    lookups = []
    for extent in route_schedule:
        if extent.route != "compiled":
            continue
        key = _batch_key(extent)
        actions = _actions_for_extent(extent, rank, actions_by_id)
        source = source_fragments[key]

        def compile_retained(
            source_fragment=source,
            batch_actions=actions,
        ):
            if source_fragment is None:
                return None
            retained = slice_integrated_jagged_ranges(
                source_fragment,
                tuple(
                    value.retained_start
                    for value in batch_actions
                ),
                tuple(value.old_tokens for value in batch_actions),
            )
            return operator.execute_into(
                program,
                retained,
                _direct_destination(retained, target_version),
            )

        retained = _timed_phase(
            "compiled_retained",
            device,
            phase_seconds,
            compile_retained,
        )
        finalized = _timed_phase(
            "compiled_finalization_append_only",
            device,
            phase_seconds,
            lambda cache=retained, batch=inputs["fused_suffix"][key]: integrated_sharded_append_only(
                target_model,
                cache,
                batch,
                target_version,
            ),
        )
        lookups.append(
            _lookup_payload(
                "compiled_finalization_append_only",
                finalized.lookup_metrics,
            )
        )
        if finalized.fragment is not None:
            outputs.append(finalized.fragment)
        retained = None
        del finalized
    for extent in merged_exact_schedule:
        key = _batch_key(extent)
        exact = _timed_phase(
            "merged_exact_pool",
            device,
            phase_seconds,
            lambda batch=inputs["merged_exact"][key]: integrated_sharded_exact(
                target_model,
                batch,
                target_version,
            ),
        )
        payload = _lookup_payload(
            "merged_exact_pool",
            exact.lookup_metrics,
        )
        payload["reason_counts"] = integrated_exact_reason_counts(
            extent,
            rank,
            actions_by_id,
        )
        payload["pool_ordinal"] = extent.ordinal
        lookups.append(payload)
        if exact.fragment is not None:
            outputs.append(exact.fragment)
        del exact
    return outputs, dict(phase_seconds), lookups


def _execute_one_shot_all_exact(
    all_schedule: tuple[D2IntegratedExtent, ...],
    inputs: dict[str, dict[tuple[str, int], RawHistoryBatch]],
    target_model,
    target_version: str,
    device: torch.device,
) -> tuple[
    list[JaggedMigratedKVBatch],
    dict[str, float],
    list[dict[str, object]],
]:
    outputs = []
    phase_seconds: dict[str, float] = defaultdict(float)
    lookups = []
    for extent in all_schedule:
        key = _batch_key(extent)
        exact = _timed_phase(
            "one_shot_exact",
            device,
            phase_seconds,
            lambda batch=inputs["full"][key]: integrated_sharded_exact(
                target_model,
                batch,
                target_version,
            ),
        )
        lookups.append(
            _lookup_payload("one_shot_exact", exact.lookup_metrics)
        )
        if exact.fragment is not None:
            outputs.append(exact.fragment)
        del exact
    return outputs, dict(phase_seconds), lookups


def _execute_two_stage_all_exact(
    all_schedule: tuple[D2IntegratedExtent, ...],
    inputs: dict[str, dict[tuple[str, int], RawHistoryBatch]],
    target_model,
    target_version: str,
    device: torch.device,
) -> tuple[
    list[JaggedMigratedKVBatch],
    dict[str, float],
    list[dict[str, object]],
]:
    outputs = []
    phase_seconds: dict[str, float] = defaultdict(float)
    lookups = []
    for extent in all_schedule:
        key = _batch_key(extent)
        exact = _timed_phase(
            "two_stage_exact_prefix",
            device,
            phase_seconds,
            lambda batch=inputs["prefix_all"][key]: integrated_sharded_exact(
                target_model,
                batch,
                target_version,
            ),
        )
        lookups.append(
            _lookup_payload(
                "two_stage_exact_prefix",
                exact.lookup_metrics,
            )
        )
        latest = _timed_phase(
            "latest_append",
            device,
            phase_seconds,
            lambda retained=exact.fragment, batch=inputs["latest_all"][key]: integrated_sharded_append(
                target_model,
                retained,
                batch,
                target_version,
            ),
        )
        lookups.append(
            _lookup_payload("latest_append", latest.lookup_metrics)
        )
        if latest.fragment is not None:
            outputs.append(latest.fragment)
        del exact
        del latest
    return outputs, dict(phase_seconds), lookups


def _output_closure(
    outputs: list[
        JaggedMigratedKVBatch | IntegratedAppendOnlyKVBatch
    ],
    local_actions: tuple[D2ActionRecord, ...],
) -> dict[str, object]:
    expected = {
        value.record_id: value.final_tokens
        for value in local_actions
    }
    observed: dict[int, int] = {}
    for fragment in outputs:
        lengths = tuple(int(value) for value in fragment.lengths.tolist())
        for record_id, length in zip(
            fragment.record_ids,
            lengths,
            strict=True,
        ):
            if record_id in observed:
                raise RuntimeError("integrated output record is duplicated")
            observed[record_id] = length
    segmented = [
        value
        for value in outputs
        if isinstance(value, IntegratedAppendOnlyKVBatch)
    ]
    retained_reused_bytes = sum(
        value.retained.k.numel() * value.retained.k.element_size()
        + value.retained.v.numel() * value.retained.v.element_size()
        for value in segmented
    )
    appended_suffix_bytes = sum(
        value.suffix.k.numel() * value.suffix.k.element_size()
        + value.suffix.v.numel() * value.suffix.v.element_size()
        for value in segmented
    )
    return {
        "passed": observed == expected,
        "records": len(observed),
        "tokens": sum(observed.values()),
        "logical_kv_bytes": sum(value.nbytes for value in outputs),
        "segmented_extents": len(segmented),
        "segmented_records": sum(
            value.batch_size for value in segmented
        ),
        "retained_reused_bytes": retained_reused_bytes,
        "appended_suffix_bytes": appended_suffix_bytes,
        "record_ids": sorted(observed),
    }


def _sample_ids(
    local_actions: tuple[D2ActionRecord, ...],
) -> tuple[int, ...]:
    output = []
    for route in INTEGRATED_ROUTES:
        candidates = [
            value.record_id
            for value in local_actions
            if integrated_route(value) == route
        ]
        if candidates:
            output.append(candidates[len(candidates) // 2])
    return tuple(output)


def _output_samples(
    outputs: list[
        JaggedMigratedKVBatch | IntegratedAppendOnlyKVBatch
    ],
    sample_ids: tuple[int, ...],
) -> dict[str, dict[str, object]]:
    requested = set(sample_ids)
    samples = {}
    for fragment in outputs:
        lengths = tuple(int(value) for value in fragment.lengths.tolist())
        if isinstance(fragment, IntegratedAppendOnlyKVBatch):
            retained_lengths = tuple(
                int(value)
                for value in fragment.retained.lengths.tolist()
            )
            suffix_lengths = tuple(
                int(value)
                for value in fragment.suffix.lengths.tolist()
            )
            retained_offsets = [0]
            suffix_offsets = [0]
            for retained_length, suffix_length in zip(
                retained_lengths,
                suffix_lengths,
                strict=True,
            ):
                retained_offsets.append(
                    retained_offsets[-1] + retained_length
                )
                suffix_offsets.append(
                    suffix_offsets[-1] + suffix_length
                )
            num_layers = fragment.retained.k.shape[0]
        else:
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            num_layers = fragment.k.shape[0]
        for row, record_id in enumerate(fragment.record_ids):
            if record_id not in requested:
                continue
            length = lengths[row]
            token_rows = sorted({0, length // 2, length - 1})
            layer_rows = sorted({0, num_layers - 1})
            pieces = []
            for layer in layer_rows:
                for token in token_rows:
                    if isinstance(
                        fragment,
                        IntegratedAppendOnlyKVBatch,
                    ):
                        if token < retained_lengths[row]:
                            position = retained_offsets[row] + token
                            source_k = fragment.retained.k
                            source_v = fragment.retained.v
                        else:
                            position = (
                                suffix_offsets[row]
                                + token
                                - retained_lengths[row]
                            )
                            source_k = fragment.suffix.k
                            source_v = fragment.suffix.v
                    else:
                        position = offsets[row] + token
                        source_k = fragment.k
                        source_v = fragment.v
                    pieces.extend(
                        (
                            source_k[layer, position, :8],
                            source_v[layer, position, :8],
                        )
                    )
            values = torch.cat(pieces).float().cpu()
            samples[str(record_id)] = {
                "length": length,
                "values": values.tolist(),
                "finite": bool(torch.isfinite(values).all()),
            }
    if set(samples) != {str(value) for value in sample_ids}:
        raise RuntimeError("integrated output samples are incomplete")
    return samples


def _lookup_totals(
    lookups: list[dict[str, object]],
) -> dict[str, object]:
    fields = (
        "requested_tokens",
        "local_requested_tokens",
        "remote_requested_tokens",
        "served_remote_requested_tokens",
        "actual_collective_tensor_payload_bytes",
        "off_diagonal_send_bytes",
        "off_diagonal_receive_bytes",
        "off_diagonal_bytes",
        "collective_calls",
        "collective_seconds",
    )
    return {
        field: sum(float(value[field]) for value in lookups)
        for field in fields
    }


def _run_iteration(
    method: str,
    rank: int,
    local_actions: tuple[D2ActionRecord, ...],
    route_schedule: tuple[D2IntegratedExtent, ...],
    all_schedule: tuple[D2IntegratedExtent, ...],
    merged_exact_schedule: tuple[D2IntegratedExtent, ...],
    actions_by_id: dict[int, D2ActionRecord],
    source_fragments,
    inputs,
    target_model,
    operator,
    program,
    target_version: str,
    device: torch.device,
) -> dict[str, object]:
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    resident_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    _synchronize(device)
    dist.barrier()
    started = time.perf_counter()
    if method == "owner_mixed":
        outputs, phase_seconds, lookups = _execute_owner_mixed(
            route_schedule,
            rank,
            actions_by_id,
            source_fragments,
            inputs,
            target_model,
            operator,
            program,
            target_version,
            device,
        )
    elif method == "owner_mixed_fused_finalization":
        outputs, phase_seconds, lookups = (
            _execute_owner_mixed_fused_finalization(
                route_schedule,
                rank,
                actions_by_id,
                source_fragments,
                inputs,
                target_model,
                operator,
                program,
                target_version,
                device,
            )
        )
    elif method == "owner_mixed_fused_append_only":
        outputs, phase_seconds, lookups = (
            _execute_owner_mixed_fused_append_only(
                route_schedule,
                rank,
                actions_by_id,
                source_fragments,
                inputs,
                target_model,
                operator,
                program,
                target_version,
                device,
            )
        )
    elif method == "owner_mixed_fused_append_only_merged_exact":
        outputs, phase_seconds, lookups = (
            _execute_owner_mixed_fused_append_only_merged_exact(
                route_schedule,
                merged_exact_schedule,
                rank,
                actions_by_id,
                source_fragments,
                inputs,
                target_model,
                operator,
                program,
                target_version,
                device,
            )
        )
    elif method == "one_shot_all_exact":
        outputs, phase_seconds, lookups = _execute_one_shot_all_exact(
            all_schedule,
            inputs,
            target_model,
            target_version,
            device,
        )
    elif method == "two_stage_all_exact":
        outputs, phase_seconds, lookups = _execute_two_stage_all_exact(
            all_schedule,
            inputs,
            target_model,
            target_version,
            device,
        )
    else:
        raise ValueError("integrated benchmark method differs")
    _synchronize(device)
    dist.barrier()
    wall_seconds = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    closure = _output_closure(outputs, local_actions)
    samples = _output_samples(outputs, _sample_ids(local_actions))
    report = {
        "method": method,
        "wall_seconds": wall_seconds,
        "phase_seconds": phase_seconds,
        "lookup_totals": _lookup_totals(lookups),
        "lookup_calls": lookups,
        "memory": {
            "resident_allocated_before_bytes": resident_before,
            "resident_reserved_before_bytes": reserved_before,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "closure": closure,
        "samples": samples,
    }
    outputs.clear()
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _sample_comparison(
    left: dict[str, dict[str, object]],
    right: dict[str, dict[str, object]],
    actions_by_id: dict[int, D2ActionRecord],
) -> dict[str, object]:
    output = {}
    for record_id in sorted(set(left) & set(right), key=int):
        lhs = torch.tensor(left[record_id]["values"], dtype=torch.float32)
        rhs = torch.tensor(right[record_id]["values"], dtype=torch.float32)
        delta = (lhs - rhs).abs()
        route = integrated_route(actions_by_id[int(record_id)])
        output[record_id] = {
            "route": route,
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "deployment_close": bool(
                torch.allclose(lhs, rhs, atol=2e-2, rtol=2e-2)
            ),
            "exact_route_close": (
                None
                if route == "compiled"
                else bool(torch.allclose(lhs, rhs, atol=2e-2, rtol=2e-2))
            ),
        }
    return output


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object] | None:
    if args.extent_size < 1:
        raise ValueError("integrated W3 extent size must be positive")
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("integrated timing requires warmup and repeats")
    runtime = init_d2_distributed_runtime(
        timeout_seconds=args.timeout_seconds,
    )
    try:
        if (
            runtime.world_size != 3
            or runtime.backend != "nccl"
            or runtime.device.type != "cuda"
        ):
            raise RuntimeError(
                "integrated W3 benchmark requires three torchrun NCCL ranks"
            )
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_tokens = tuple(
            value.strip()
            for value in visible_devices.split(",")
            if value.strip()
        )
        expected_visible_tokens = tuple(
            value.strip()
            for value in args.expected_visible_devices.split(",")
            if value.strip()
        )
        if visible_tokens != expected_visible_tokens or len(visible_tokens) != 3:
            raise RuntimeError(
                "integrated W3 CUDA_VISIBLE_DEVICES differs from the "
                "expected physical mapping"
            )
        action_path = _path(args.action_plan)
        stage_a_path = _path(args.stage_a_summary)
        training_path = _path(args.training_result)
        checkpoint_dir = _path(args.checkpoint_dir)
        action_plan = D2ActionPlan.load(action_path)
        stage_a = json.loads(stage_a_path.read_text())
        training = json.loads(training_path.read_text())
        if (
            stage_a["status"] != "complete"
            or stage_a["action_plan"]["content_sha256"]
            != action_plan.content_sha256
        ):
            raise ValueError("integrated Stage A binding differs")
        owner_map = build_d2_record_owner_map(
            action_plan,
            runtime.world_size,
            "strict_cow_lpt",
        )
        selected = select_integrated_records(
            action_plan,
            owner_map,
            runtime.world_size,
            args.cohort,
        )
        actions_by_id = {value.record_id: value for value in selected}
        local_actions = tuple(
            value
            for value in selected
            if owner_map[value.record_id] == runtime.rank
        )
        route_schedule = build_integrated_schedule(
            selected,
            owner_map,
            runtime.world_size,
            args.extent_size,
            route_major=True,
            compiled_order=args.compiled_order,
        )
        all_schedule = build_integrated_schedule(
            selected,
            owner_map,
            runtime.world_size,
            args.extent_size,
            route_major=False,
        )
        merged_exact_schedule = build_integrated_exact_pool_schedule(
            selected,
            owner_map,
            runtime.world_size,
            args.extent_size,
        )
        reconstruction_started = time.perf_counter()
        prepared_path = _path(action_plan.provenance.prepared_data)
        data_plan, prepared_metadata = load_prepared_kuairand_plan(
            prepared_path
        )
        validate_long_context_plan(data_plan, prepared_metadata, 4)
        windows = reconstruct_organic_windows(
            data_plan,
            (value.prepared_user_id for value in selected),
        )
        source_index = int(
            action_plan.source_version.removeprefix("theta")
        )
        target_index = int(
            action_plan.target_version.removeprefix("theta")
        )
        source_window = windows[source_index]
        target_window = windows[target_index]
        for action in selected:
            source_record = source_window.records[action.prepared_user_id]
            target_record = target_window.records[action.prepared_user_id]
            if (
                source_record.history is None
                or source_record.history_sha256 != action.old_history_sha256
                or len(source_record.history) != action.old_tokens
                or target_record.history is None
                or target_record.history_sha256
                != action.target_history_sha256
                or len(target_record.history) != action.final_tokens
            ):
                raise ValueError("integrated reconstructed history differs")
        reconstruction_seconds = time.perf_counter() - reconstruction_started
        cfg = HSTUConfig(**training["model"])
        source_checkpoint = _checkpoint_path(
            checkpoint_dir,
            action_plan.source_version,
        )
        target_checkpoint = _checkpoint_path(
            checkpoint_dir,
            action_plan.target_version,
        )
        source_batches = _prepare_source_batches(
            route_schedule,
            runtime.rank,
            actions_by_id,
            source_window,
            action_plan.source_version,
            runtime.device,
        )
        source_model_started = time.perf_counter()
        source_cpu = _load_cpu_model(cfg, source_checkpoint)
        source_model = build_modulo_sharded_hstu_from_cpu(
            source_cpu,
            runtime.rank,
            runtime.world_size,
            runtime.device,
        )
        del source_cpu
        source_fragments = {}
        source_lookups = []
        dist.barrier()
        source_materialization_started = time.perf_counter()
        for extent in route_schedule:
            key = _batch_key(extent)
            exact = integrated_sharded_exact(
                source_model,
                source_batches[key],
                action_plan.source_version,
            )
            source_fragments[key] = exact.fragment
            source_lookups.append(
                _lookup_payload(
                    "source_theta1_fixture",
                    exact.lookup_metrics,
                )
            )
        _synchronize(runtime.device)
        dist.barrier()
        source_materialization_seconds = (
            time.perf_counter() - source_materialization_started
        )
        source_model_setup_seconds = (
            time.perf_counter() - source_model_started
        )
        source_resident_bytes = sum(
            0 if value is None else value.nbytes
            for value in source_fragments.values()
        )
        del source_model
        del source_batches
        gc.collect()
        torch.cuda.empty_cache()
        target_input_preparation_started = time.perf_counter()
        target_inputs = _prepare_target_batches(
            route_schedule,
            all_schedule,
            runtime.rank,
            actions_by_id,
            target_window,
            action_plan.target_version,
            runtime.device,
        )
        merged_exact_preparation_started = time.perf_counter()
        target_inputs["merged_exact"] = _prepare_merged_exact_batches(
            merged_exact_schedule,
            runtime.rank,
            actions_by_id,
            target_window,
            action_plan.target_version,
            runtime.device,
        )
        merged_exact_input_preparation_seconds = (
            time.perf_counter() - merged_exact_preparation_started
        )
        target_input_preparation_seconds = (
            time.perf_counter() - target_input_preparation_started
        )
        target_model_started = time.perf_counter()
        target_cpu = _load_cpu_model(cfg, target_checkpoint)
        target_model = build_modulo_sharded_hstu_from_cpu(
            target_cpu,
            runtime.rank,
            runtime.world_size,
            runtime.device,
        )
        del target_cpu
        stage_a_program = _program_descriptor(stage_a)
        program_cpu, loaded_program = load_direct_oldkv_program(
            _path(stage_a_program["path"]),
            expected_sha256=stage_a_program["sha256"],
            expected_source_version=action_plan.source_version,
            expected_target_version=action_plan.target_version,
            expected_num_layers=cfg.num_layers,
            expected_kv_width=cfg.hidden_size,
        )
        operator = DirectOldKVFusedOperator()
        program = operator.prepare_program(program_cpu, runtime.device)
        del program_cpu
        target_model_setup_seconds = (
            time.perf_counter() - target_model_started
        )
        methods = [
            "owner_mixed",
            "owner_mixed_fused_finalization",
            "owner_mixed_fused_append_only",
            "owner_mixed_fused_append_only_merged_exact",
            "one_shot_all_exact",
        ]
        if args.include_two_stage_all_exact:
            methods.append("two_stage_all_exact")
        lookup_multisets = {
            method: _lookup_multiset(
                _lookup_batches_for_method(
                    method,
                    route_schedule,
                    all_schedule,
                    merged_exact_schedule,
                    target_inputs,
                )
            )
            for method in methods
        }
        local_action_lookup_ledgers = {
            organization: integrated_lookup_token_ledger(
                local_actions,
                organization,
            )
            for organization in ("staged", "fused_finalization")
        }
        if (
            lookup_multisets["owner_mixed"]
            != lookup_multisets[
                "owner_mixed_fused_finalization"
            ]
            or lookup_multisets["owner_mixed"]
            != lookup_multisets[
                "owner_mixed_fused_append_only"
            ]
            or lookup_multisets["owner_mixed"]
            != lookup_multisets[
                "owner_mixed_fused_append_only_merged_exact"
            ]
            or local_action_lookup_ledgers["staged"]["total"]
            != local_action_lookup_ledgers[
                "fused_finalization"
            ]["total"]
            or lookup_multisets["owner_mixed"]["requested_tokens"]
            != local_action_lookup_ledgers["staged"]["total"]
        ):
            raise RuntimeError(
                "integrated fused lookup multiset differs from staged mixed"
            )
        method_reports = {}
        method_samples = {}
        for method in methods:
            for _ in range(args.warmup):
                _run_iteration(
                    method,
                    runtime.rank,
                    local_actions,
                    route_schedule,
                    all_schedule,
                    merged_exact_schedule,
                    actions_by_id,
                    source_fragments,
                    target_inputs,
                    target_model,
                    operator,
                    program,
                    action_plan.target_version,
                    runtime.device,
                )
            repeats = []
            for repeat in range(args.repeats):
                report = _run_iteration(
                    method,
                    runtime.rank,
                    local_actions,
                    route_schedule,
                    all_schedule,
                    merged_exact_schedule,
                    actions_by_id,
                    source_fragments,
                    target_inputs,
                    target_model,
                    operator,
                    program,
                    action_plan.target_version,
                    runtime.device,
                )
                report["repeat"] = repeat
                repeats.append(report)
            method_reports[method] = repeats
            method_samples[method] = repeats[0]["samples"]
        sample_comparison = _sample_comparison(
            method_samples["owner_mixed"],
            method_samples["one_shot_all_exact"],
            actions_by_id,
        )
        fused_sample_comparison = _sample_comparison(
            method_samples["owner_mixed"],
            method_samples["owner_mixed_fused_finalization"],
            actions_by_id,
        )
        append_only_sample_comparison = _sample_comparison(
            method_samples["owner_mixed_fused_finalization"],
            method_samples["owner_mixed_fused_append_only"],
            actions_by_id,
        )
        merged_exact_sample_comparison = _sample_comparison(
            method_samples["owner_mixed_fused_append_only"],
            method_samples[
                "owner_mixed_fused_append_only_merged_exact"
            ],
            actions_by_id,
        )
        local_report = {
            "rank": runtime.rank,
            "local_rank": runtime.local_rank,
            "logical_cuda_index": torch.cuda.current_device(),
            "cuda_visible_devices": visible_devices,
            "physical_visible_token": visible_tokens[runtime.local_rank],
            "device_name": torch.cuda.get_device_name(runtime.device),
            "device_uuid": (
                f"GPU-{torch.cuda.get_device_properties(runtime.device).uuid}"
            ),
            "record_ids": [value.record_id for value in local_actions],
            "route_counts": {
                route: sum(
                    integrated_route(value) == route
                    for value in local_actions
                )
                for route in INTEGRATED_ROUTES
            },
            "old_tokens": sum(value.old_tokens for value in local_actions),
            "final_tokens": sum(value.final_tokens for value in local_actions),
            "setup": {
                "reconstruction_seconds": reconstruction_seconds,
                "source_model_and_materialization_seconds": (
                    source_model_setup_seconds
                ),
                "source_materialization_seconds": (
                    source_materialization_seconds
                ),
                "target_model_and_program_seconds": (
                    target_model_setup_seconds
                ),
                "target_input_preparation_seconds": (
                    target_input_preparation_seconds
                ),
                "merged_exact_input_preparation_seconds": (
                    merged_exact_input_preparation_seconds
                ),
                "source_resident_bytes": source_resident_bytes,
                "source_lookup_totals": _lookup_totals(source_lookups),
            },
            "merged_exact_pool": {
                "reason_counts": {
                    reason: sum(
                        integrated_exact_reason_counts(
                            extent,
                            runtime.rank,
                            actions_by_id,
                        )[reason]
                        for extent in merged_exact_schedule
                    )
                    for reason in (
                        "scheduled_exact",
                        "natural_exact",
                    )
                },
                "record_ids": [
                    record_id
                    for extent in merged_exact_schedule
                    for record_id in extent.local_record_ids(runtime.rank)
                ],
                "extents": len(merged_exact_schedule),
                "history_batches_prepared_before_method_timer": True,
            },
            "methods": method_reports,
            "lookup_multisets": lookup_multisets,
            "action_lookup_ledgers": local_action_lookup_ledgers,
            "sample_comparison": sample_comparison,
            "fused_sample_comparison": fused_sample_comparison,
            "append_only_sample_comparison": (
                append_only_sample_comparison
            ),
            "merged_exact_sample_comparison": (
                merged_exact_sample_comparison
            ),
        }
        gathered: list[dict[str, object] | None] = [
            None
            for _ in range(runtime.world_size)
        ]
        dist.all_gather_object(gathered, local_report)
        if not runtime.is_primary:
            return None
        ranks = [value for value in gathered if value is not None]
        checks = {
            "rank_count": len(ranks) == runtime.world_size,
            "owner_coverage": (
                sorted(
                    record_id
                    for rank_report in ranks
                    for record_id in rank_report["record_ids"]
                )
                == [value.record_id for value in selected]
            ),
            "all_method_closures": all(
                repeat["closure"]["passed"]
                for rank_report in ranks
                for method in methods
                for repeat in rank_report["methods"][method]
            ),
            "sample_values_finite": all(
                sample["finite"]
                for rank_report in ranks
                for method in methods
                for sample in rank_report["methods"][method][0][
                    "samples"
                ].values()
            ),
            "sample_comparison_completed": all(
                len(rank_report["sample_comparison"]) > 0
                and len(rank_report["fused_sample_comparison"]) > 0
                and len(
                    rank_report["append_only_sample_comparison"]
                )
                > 0
                and len(
                    rank_report["merged_exact_sample_comparison"]
                )
                > 0
                for rank_report in ranks
            ),
            "fused_lookup_multiset_matches_staged": all(
                rank_report["lookup_multisets"]["owner_mixed"]
                == rank_report["lookup_multisets"][
                    "owner_mixed_fused_finalization"
                ]
                == rank_report["lookup_multisets"][
                    "owner_mixed_fused_append_only"
                ]
                == rank_report["lookup_multisets"][
                    "owner_mixed_fused_append_only_merged_exact"
                ]
                for rank_report in ranks
            ),
            "measured_mixed_lookup_tokens_match": all(
                int(
                    rank_report["methods"]["owner_mixed"][repeat][
                        "lookup_totals"
                    ]["requested_tokens"]
                )
                == int(
                    rank_report["methods"][
                        "owner_mixed_fused_finalization"
                    ][repeat]["lookup_totals"]["requested_tokens"]
                )
                == int(
                    rank_report["methods"][
                        "owner_mixed_fused_append_only"
                    ][repeat]["lookup_totals"]["requested_tokens"]
                )
                == int(
                    rank_report["methods"][
                        "owner_mixed_fused_append_only_merged_exact"
                    ][repeat]["lookup_totals"]["requested_tokens"]
                )
                == rank_report["action_lookup_ledgers"]["staged"][
                    "total"
                ]
                for rank_report in ranks
                for repeat in range(args.repeats)
            ),
            "fused_collective_calls_match_route_extents": all(
                int(
                    rank_report["methods"][
                        "owner_mixed_fused_finalization"
                    ][repeat]["lookup_totals"]["collective_calls"]
                )
                == int(
                    rank_report["methods"][
                        "owner_mixed_fused_append_only"
                    ][repeat]["lookup_totals"]["collective_calls"]
                )
                == 3 * len(route_schedule)
                for rank_report in ranks
                for repeat in range(args.repeats)
            ),
            "merged_exact_collective_calls_match_pool_extents": all(
                int(
                    rank_report["methods"][
                        "owner_mixed_fused_append_only_merged_exact"
                    ][repeat]["lookup_totals"]["collective_calls"]
                )
                == 3
                * (
                    sum(
                        extent.route == "compiled"
                        for extent in route_schedule
                    )
                    + len(merged_exact_schedule)
                )
                for rank_report in ranks
                for repeat in range(args.repeats)
            ),
            "merged_exact_reason_counts_preserved": all(
                rank_report["merged_exact_pool"]["reason_counts"]
                == {
                    reason: rank_report["route_counts"][reason]
                    for reason in (
                        "scheduled_exact",
                        "natural_exact",
                    )
                }
                for rank_report in ranks
            ),
            "merged_exact_owner_coverage_preserved": (
                sorted(
                    record_id
                    for rank_report in ranks
                    for record_id in rank_report[
                        "merged_exact_pool"
                    ]["record_ids"]
                )
                == sorted(
                    value.record_id
                    for value in selected
                    if integrated_route(value) != "compiled"
                )
                and all(
                    owner_map[record_id] == rank_report["rank"]
                    for rank_report in ranks
                    for record_id in rank_report[
                        "merged_exact_pool"
                    ]["record_ids"]
                )
            ),
            "append_only_segments_cover_compiled_records": all(
                all(
                    rank_report["methods"][method][repeat][
                        "closure"
                    ]["segmented_records"]
                    == rank_report["route_counts"]["compiled"]
                    and rank_report["methods"][method][repeat][
                        "closure"
                    ]["retained_reused_bytes"]
                    > 0
                    and rank_report["methods"][method][repeat][
                        "closure"
                    ]["appended_suffix_bytes"]
                    > 0
                    for method in (
                        "owner_mixed_fused_append_only",
                        "owner_mixed_fused_append_only_merged_exact",
                    )
                )
                for rank_report in ranks
                for repeat in range(args.repeats)
            ),
            "merged_exact_outputs_match_append_only": all(
                all(
                    sample["deployment_close"]
                    for sample in rank_report[
                        "merged_exact_sample_comparison"
                    ].values()
                )
                for rank_report in ranks
            ),
        }
        makespans = {
            method: [
                max(
                    rank_report["methods"][method][repeat][
                        "wall_seconds"
                    ]
                    for rank_report in ranks
                )
                for repeat in range(args.repeats)
            ]
            for method in methods
        }
        artifact: dict[str, object] = {
            "protocol": PROTOCOL,
            "status": "complete" if all(checks.values()) else "failed",
            "scientific_result": False,
            "formal_stage_c": False,
            "scope": {
                "development_only": True,
                "physical_w3": True,
                "formal_w4_substitute": False,
                "paper_performance_claim": False,
                "publication_and_commit_timed": False,
                "source_fixture_materialization_timed": False,
                "source_theta1_kv_retained_during_methods": True,
                "strict_cow_target_endpoint_retained_during_timer": True,
                "full_cpu_payload_sha_in_timed_path": False,
                "collective_guard_in_timed_path": False,
                "planned_wave_prefetch": False,
                "planned_phase_fusion": True,
                "append_only_destination_assembly": True,
                "exact_reason_execution_pool": True,
                "retained_prefix_rewritten_during_finalization": False,
                "wave_plan_and_history_batch_preparation_timed": False,
                "method_timer_includes_model_embedding_collectives": True,
                "low_level_kernel_optimization": False,
            },
            "configuration": {
                "cohort": args.cohort,
                "records": len(selected),
                "world_size": runtime.world_size,
                "backend": runtime.backend,
                "cuda_visible_devices": visible_devices,
                "expected_visible_devices": (
                    args.expected_visible_devices
                ),
                "extent_size": args.extent_size,
                "compiled_order": args.compiled_order,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "methods": methods,
                "owner_strategy": "strict_cow_lpt",
                "owner_map_sha256": d2_record_owner_map_sha256(
                    owner_map
                ),
                "route_schedule": [
                    {
                        "route": value.route,
                        "ordinal": value.ordinal,
                        "record_counts": [
                            len(record_ids)
                            for record_ids in value.record_ids_by_rank
                        ],
                    }
                    for value in route_schedule
                ],
                "merged_exact_schedule": [
                    {
                        "route": value.route,
                        "ordinal": value.ordinal,
                        "record_counts": [
                            len(record_ids)
                            for record_ids in value.record_ids_by_rank
                        ],
                        "reason_counts_by_rank": [
                            integrated_exact_reason_counts(
                                value,
                                rank,
                                actions_by_id,
                            )
                            for rank in range(runtime.world_size)
                        ],
                    }
                    for value in merged_exact_schedule
                ],
                "merged_exact_collective_call_expectation_per_rank": (
                    3
                    * (
                        sum(
                            extent.route == "compiled"
                            for extent in route_schedule
                        )
                        + len(merged_exact_schedule)
                    )
                ),
            },
            "inputs": {
                "action_plan": {
                    "path": str(action_path.relative_to(ROOT)),
                    "content_sha256": action_plan.content_sha256,
                    "file_sha256": file_sha256(action_path),
                },
                "prepared_data": {
                    "path": str(prepared_path.relative_to(ROOT)),
                    "expected_sha256": (
                        action_plan.provenance.prepared_data_sha256
                    ),
                },
                "training_result": str(
                    training_path.relative_to(ROOT)
                ),
                "source_checkpoint": _checkpoint_descriptor(
                    action_plan,
                    action_plan.source_version,
                ),
                "target_checkpoint": _checkpoint_descriptor(
                    action_plan,
                    action_plan.target_version,
                ),
                "program": loaded_program,
            },
            "makespans_seconds": makespans,
            "action_lookup_ledgers": {
                organization: integrated_lookup_token_ledger(
                    selected,
                    organization,
                )
                for organization in (
                    "staged",
                    "fused_finalization",
                )
            },
            "checks": checks,
            "rank_reports": ranks,
        }
        artifact["content_sha256"] = canonical_sha256(artifact)
        return artifact
    finally:
        close_d2_distributed_runtime(runtime)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    artifact = run(args)
    if artifact is None:
        return
    if args.output is not None:
        _write_json_atomic(_path(args.output), artifact)
    print(
        json.dumps(
            {
                "protocol": artifact["protocol"],
                "status": artifact["status"],
                "configuration": artifact["configuration"],
                "makespans_seconds": artifact["makespans_seconds"],
                "checks": artifact["checks"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
