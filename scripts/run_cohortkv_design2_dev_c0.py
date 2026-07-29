from __future__ import annotations

import argparse
import gc
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch

from hstu_kvcache.migration.cohort_jagged import JaggedMigratedKVBatch
from hstu_kvcache.migration.design2_dev_epoch import (
    D2DevEpochPointer,
    D2DevEpochRecord,
    D2DevEpochSpec,
    D2DevEpochStateMachine,
    D2DevPrivatePrepare,
    D2DevReadbackAck,
)
from hstu_kvcache.migration.design2_dev_wave import (
    D2_DEV_WAVE_PROTOCOL,
    assemble_d2_dev_jagged,
    build_d2_dev_lineages,
    close_d2_dev_wave,
    d2_dev_record_payload_sha256,
)
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
    D2ActionRecord,
    build_d2_record_owner_map,
    canonical_sha256,
    d2_record_owner_map_sha256,
    file_sha256,
)
from hstu_kvcache.migration.organic import slice_jagged_token_ranges
from hstu_kvcache.migration.recompute import RawHistoryBatch
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    load_direct_oldkv_program,
)
from hstu_kvcache.migration.stage46_chain import (
    pack_padded_cache,
    select_jagged_rows,
    unpack_jagged_cache,
)
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
        choices=("normal", "pre_commit_abort"),
        default="normal",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args(argv)


def _path(value: str) -> Path:
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


def _records_by_id(
    sample: dict[str, object],
) -> dict[int, dict[str, object]]:
    return {
        int(value["action"]["record_id"]): value
        for value in sample["records"]
    }


def _history_batch(
    actions: tuple[D2ActionRecord, ...],
    records: dict[int, dict[str, object]],
    history_key: str,
    starts: tuple[int, ...],
    stops: tuple[int, ...],
    version: str,
    device: torch.device,
) -> RawHistoryBatch:
    if len(actions) != len(starts) or len(actions) != len(stops):
        raise ValueError("D2 dev history batch ranges differ")
    record_ids = tuple(value.record_id for value in actions)
    lengths = tuple(
        stop - start
        for start, stop in zip(starts, stops, strict=True)
    )
    if any(value < 0 for value in lengths):
        raise ValueError("D2 dev history batch range is invalid")
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
            raise ValueError("D2 dev history range exceeds sample")
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
        lengths=torch.tensor(lengths, dtype=torch.long, device=device),
    )


def _phase_order(world_size: int, case: str) -> tuple[str, ...]:
    phases = ["preflight_vote"]
    lookup_phases = (
        "source_old_kv_fixture",
        "natural_exact_prefix",
        "scheduled_exact_retained",
        "compiled_delta_append",
        "scheduled_delta_append",
        "all_records_latest_append",
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
            "epoch_manifest_gather",
            "epoch_spec_broadcast",
            "epoch_prepare_gather",
            "epoch_commit_broadcast",
        )
    )
    if case == "normal":
        phases.extend(
            (
                "epoch_readback_gather",
                "epoch_publication_broadcast",
            )
        )
    phases.append("rank_report_gather")
    return tuple(phases)


@contextmanager
def _guard_context(guard: D2CollectiveGuard, phase: str):
    guard.enter(phase)
    yield


def _lookup_guard(guard: D2CollectiveGuard, phase: str):
    return lambda operation: _guard_context(
        guard,
        f"{phase}.{operation}",
    )


def _invoke_exact(
    model,
    batch: RawHistoryBatch,
    target_version: str,
    guard: D2CollectiveGuard,
    phase: str,
):
    guard.enter(f"{phase}.invoke")
    return sharded_exact_jagged_hidden_and_kv(
        model,
        batch,
        target_version,
        dtype=torch.float16,
        collective_phase_guard=_lookup_guard(guard, phase),
    )


def _empty_cache(
    cfg: HSTUConfig,
    device: torch.device,
) -> HSTUKVCache:
    shape = (cfg.num_layers, 0, 0, cfg.hidden_size)
    return HSTUKVCache(
        k=torch.empty(shape, dtype=torch.float16, device=device),
        v=torch.empty(shape, dtype=torch.float16, device=device),
        seq_len=0,
    )


def _relabel(
    fragment: JaggedMigratedKVBatch,
    version: str,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=fragment.record_ids,
        migration_anchor_version=version,
        served_kv_target=version,
        k=fragment.k,
        v=fragment.v,
        lengths=fragment.lengths,
        offsets=fragment.offsets,
    )


def _select(
    fragment: JaggedMigratedKVBatch,
    record_ids: tuple[int, ...],
) -> JaggedMigratedKVBatch:
    return select_jagged_rows(
        fragment,
        tuple(fragment.record_index(value) for value in record_ids),
    )


def _invoke_append(
    model,
    initial: JaggedMigratedKVBatch | None,
    actions: tuple[D2ActionRecord, ...],
    records: dict[int, dict[str, object]],
    starts: tuple[int, ...],
    stops: tuple[int, ...],
    target_version: str,
    cfg: HSTUConfig,
    guard: D2CollectiveGuard,
    phase: str,
) -> tuple[JaggedMigratedKVBatch | None, dict[str, object]]:
    if len(actions) != len(starts) or len(actions) != len(stops):
        raise ValueError("D2 dev append ranges differ")
    positive_rows = tuple(
        index
        for index, (start, stop) in enumerate(
            zip(starts, stops, strict=True)
        )
        if stop > start
    )
    positive_actions = tuple(actions[index] for index in positive_rows)
    positive_starts = tuple(starts[index] for index in positive_rows)
    positive_stops = tuple(stops[index] for index in positive_rows)
    if positive_actions:
        if initial is None:
            raise ValueError("D2 dev append has no retained fragment")
        positive_ids = tuple(value.record_id for value in positive_actions)
        positive_initial = _select(initial, positive_ids)
        cache = unpack_jagged_cache(
            positive_initial,
            dtype=positive_initial.k.dtype,
        )
        retained_lengths = positive_initial.lengths
    else:
        cache = _empty_cache(cfg, model.item_embedding.local_weight.device)
        retained_lengths = torch.empty(
            0,
            dtype=torch.long,
            device=model.item_embedding.local_weight.device,
        )
    batch = _history_batch(
        positive_actions,
        records,
        "target_history",
        positive_starts,
        positive_stops,
        target_version,
        model.item_embedding.local_weight.device,
    )
    guard.enter(f"{phase}.invoke")
    result = sharded_append_padded_cache(
        model,
        cache,
        batch.item_ids,
        batch.behaviors,
        batch.time_deltas,
        batch.lengths,
        retained_lengths=retained_lengths,
        collective_phase_guard=_lookup_guard(guard, phase),
    )
    updated = None
    if positive_actions:
        updated = pack_padded_cache(
            result.updated_cache,
            result.lengths,
            tuple(value.record_id for value in positive_actions),
            target_version,
            target_version,
            dtype=torch.float16,
        )
    zero_ids = tuple(
        action.record_id
        for index, action in enumerate(actions)
        if index not in positive_rows
    )
    unchanged = None
    if zero_ids:
        if initial is None:
            raise ValueError("D2 dev zero append has no retained fragment")
        unchanged = _relabel(_select(initial, zero_ids), target_version)
    assembled = assemble_d2_dev_jagged(
        tuple(value.record_id for value in actions),
        (updated, unchanged),
        target_version,
        target_version,
    )
    return assembled, _lookup_evidence(result.lookup_metrics)


def _lookup_evidence(metrics) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.to_dict().items()
        if "seconds" not in key
    }


def _program_descriptor(
    stage_a_summary: dict[str, object],
) -> dict[str, object]:
    capacity = json.loads(
        _path(stage_a_summary["artifacts"]["capacity"]["path"]).read_text()
    )
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


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _normal_rank(
    args: argparse.Namespace,
    runtime,
) -> dict[str, object] | None:
    action_path = _path(args.action_plan)
    stage_a_path = _path(args.stage_a_summary)
    sample_path = _path(args.sample_inputs)
    training_path = _path(args.training_result)
    checkpoint_dir = _path(args.checkpoint_dir)
    action_plan = D2ActionPlan.load(action_path)
    stage_a = json.loads(stage_a_path.read_text())
    sample = json.loads(sample_path.read_text())
    training = json.loads(training_path.read_text())
    cfg = HSTUConfig(**training["model"])
    records = _records_by_id(sample)
    sample_actions = tuple(
        D2ActionRecord.from_dict(value["action"])
        for value in sample["records"]
    )
    plan_actions = {value.record_id: value for value in action_plan.records}
    owner_map = build_d2_record_owner_map(
        action_plan,
        runtime.world_size,
        "strict_cow_lpt",
    )
    local_actions = tuple(
        sorted(
            (
                value
                for value in sample_actions
                if owner_map[value.record_id] == runtime.rank
            ),
            key=lambda value: value.record_id,
        )
    )
    compiled_actions = tuple(
        value
        for value in local_actions
        if value.requested_action == "compiled"
    )
    scheduled_actions = tuple(
        value
        for value in local_actions
        if value.requested_reason == "scheduled_exact"
    )
    natural_actions = tuple(
        value
        for value in local_actions
        if value.requested_reason == "natural_exact"
    )
    lineages = build_d2_dev_lineages(
        local_actions,
        owner_map,
        runtime.world_size,
        action_plan.source_version,
        action_plan.target_version,
    )
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
    guard = D2CollectiveGuard(
        runtime,
        _phase_order(runtime.world_size, args.case),
    )
    preflight_failures = capture_d2_preflight_failures(
        {
            "world_size_development_scope": (
                lambda: runtime.world_size in {1, 2, 3}
            ),
            "action_plan_stage_a_binding": (
                lambda: action_plan.content_sha256
                == stage_a["action_plan"]["content_sha256"]
                and file_sha256(action_path)
                == stage_a["action_plan"]["file_sha256"]
            ),
            "stage_a_entry": (
                lambda: stage_a["status"] == "complete"
                and stage_a["stage_b_entry"] == "go"
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
            "sample_action_coverage": (
                lambda: len(sample_actions) == 16
                and len({value.record_id for value in sample_actions}) == 16
                and all(
                    plan_actions[value.record_id] == value
                    for value in sample_actions
                )
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
                lambda: file_sha256(_path(program_descriptor["path"]))
                == program_descriptor["sha256"]
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
            f"D2 dev C0 preflight failed: {preflight.failure_reasons}"
        )
    source_batch = _history_batch(
        local_actions,
        records,
        "source_history",
        (0,) * len(local_actions),
        tuple(value.old_tokens for value in local_actions),
        action_plan.source_version,
        runtime.device,
    )
    source_cpu = _load_cpu_model(cfg, source_checkpoint)
    source_sharded = build_modulo_sharded_hstu_from_cpu(
        source_cpu,
        runtime.rank,
        runtime.world_size,
        runtime.device,
    )
    del source_cpu
    source_result = _invoke_exact(
        source_sharded,
        source_batch,
        action_plan.source_version,
        guard,
        "source_old_kv_fixture",
    )
    source_lookup = _lookup_evidence(source_result.lookup_metrics)
    source_old_fragment = source_result.fragment
    source_payload_before = (
        {}
        if source_old_fragment is None
        else {
            str(record_id): d2_dev_record_payload_sha256(
                source_old_fragment,
                record_id,
            )
            for record_id in source_old_fragment.record_ids
        }
    )
    retained = None
    retained_slice = None
    compiled_source = None
    if source_old_fragment is not None and compiled_actions:
        compiled_source = _select(
            source_old_fragment,
            tuple(value.record_id for value in compiled_actions),
        )
        retained_slice = slice_jagged_token_ranges(
            compiled_source,
            tuple(value.retained_start for value in compiled_actions),
            tuple(value.old_tokens for value in compiled_actions),
        )
        retained = retained_slice.cache
    program_cpu, loaded_program = load_direct_oldkv_program(
        _path(program_descriptor["path"]),
        expected_sha256=program_descriptor["sha256"],
        expected_source_version=action_plan.source_version,
        expected_target_version=action_plan.target_version,
        expected_num_layers=cfg.num_layers,
        expected_kv_width=cfg.hidden_size,
    )
    guard.enter("compiled_retained_local")
    compiled_private = execute_compiled_retained_owner_compute(
        program_cpu,
        retained,
        owner_map,
        runtime.rank,
        operator=DirectOldKVFusedOperator(),
        phase_counters=D2CompiledRetainedPhaseCounters(),
    )
    compiled_private.metrics.phase_counters.assert_normal_path()
    compiled_retained_report = {
        "record_ids": list(compiled_private.metadata.record_ids),
        "token_count": compiled_private.metadata.token_count,
        "checksum_sha256": compiled_private.metadata.checksum_sha256,
        "phase_counters": (
            compiled_private.metrics.phase_counters.to_dict()
        ),
        "private_output": compiled_private.metrics.private_output,
    }
    del source_sharded
    del source_result
    gc.collect()
    torch.cuda.empty_cache()
    target_cpu = _load_cpu_model(cfg, target_checkpoint)
    target_sharded = build_modulo_sharded_hstu_from_cpu(
        target_cpu,
        runtime.rank,
        runtime.world_size,
        runtime.device,
    )
    del target_cpu
    natural_batch = _history_batch(
        natural_actions,
        records,
        "target_history",
        (0,) * len(natural_actions),
        tuple(value.target_prefix_tokens for value in natural_actions),
        action_plan.target_version,
        runtime.device,
    )
    natural_result = _invoke_exact(
        target_sharded,
        natural_batch,
        action_plan.target_version,
        guard,
        "natural_exact_prefix",
    )
    natural_lookup = _lookup_evidence(natural_result.lookup_metrics)
    scheduled_batch = _history_batch(
        scheduled_actions,
        records,
        "target_history",
        (0,) * len(scheduled_actions),
        tuple(value.retained_tokens for value in scheduled_actions),
        action_plan.target_version,
        runtime.device,
    )
    scheduled_result = _invoke_exact(
        target_sharded,
        scheduled_batch,
        action_plan.target_version,
        guard,
        "scheduled_exact_retained",
    )
    scheduled_lookup = _lookup_evidence(
        scheduled_result.lookup_metrics
    )
    compiled_prefix, compiled_delta_lookup = _invoke_append(
        target_sharded,
        compiled_private.output,
        compiled_actions,
        records,
        tuple(value.delta_start for value in compiled_actions),
        tuple(value.target_prefix_tokens for value in compiled_actions),
        action_plan.target_version,
        cfg,
        guard,
        "compiled_delta_append",
    )
    scheduled_prefix, scheduled_delta_lookup = _invoke_append(
        target_sharded,
        scheduled_result.fragment,
        scheduled_actions,
        records,
        tuple(value.delta_start for value in scheduled_actions),
        tuple(value.target_prefix_tokens for value in scheduled_actions),
        action_plan.target_version,
        cfg,
        guard,
        "scheduled_delta_append",
    )
    local_record_ids = tuple(value.record_id for value in local_actions)
    prefix = assemble_d2_dev_jagged(
        local_record_ids,
        (
            compiled_prefix,
            scheduled_prefix,
            natural_result.fragment,
        ),
        action_plan.target_version,
        action_plan.target_version,
    )
    final_fragment, latest_lookup = _invoke_append(
        target_sharded,
        prefix,
        local_actions,
        records,
        tuple(value.target_prefix_tokens for value in local_actions),
        tuple(value.final_tokens for value in local_actions),
        action_plan.target_version,
        cfg,
        guard,
        "all_records_latest_append",
    )
    closure = close_d2_dev_wave(
        final_fragment,
        lineages,
        runtime.rank,
        runtime.world_size,
        action_plan.source_version,
        action_plan.target_version,
    )
    lineage_by_id = {
        value.record_id: value
        for value in lineages
    }
    payload_by_id = dict(
        zip(
            closure.record_ids,
            closure.record_payload_sha256,
            strict=True,
        )
    )
    local_epoch_records = tuple(
        D2DevEpochRecord(
            record_id=record_id,
            owner_rank=runtime.rank,
            token_length=lineage_by_id[record_id].final_tokens,
            action=lineage_by_id[record_id].route,
            lineage_sha256=lineage_by_id[
                record_id
            ].lineage_sha256,
            payload_sha256=payload_by_id[record_id],
        )
        for record_id in closure.record_ids
    )
    gathered_manifests = gather_d2_rank_metadata(
        runtime,
        local_epoch_records,
        guard=guard,
        phase="epoch_manifest_gather",
    )
    source_epoch = int(
        action_plan.source_version.removeprefix("theta")
    )
    target_epoch = int(
        action_plan.target_version.removeprefix("theta")
    )
    epoch_spec = None
    if runtime.is_primary:
        if gathered_manifests is None:
            raise RuntimeError("D2 dev epoch manifests were not gathered")
        epoch_spec = D2DevEpochSpec(
            action_plan_sha256=action_plan.content_sha256,
            source_version=action_plan.source_version,
            target_version=action_plan.target_version,
            source_epoch=source_epoch,
            target_epoch=target_epoch,
            world_size=runtime.world_size,
            records=tuple(
                sorted(
                    (
                        record
                        for rank_records in gathered_manifests
                        for record in rank_records
                    ),
                    key=lambda value: value.record_id,
                )
            ),
        )
    epoch_spec = broadcast_d2_metadata(
        runtime,
        epoch_spec,
        guard=guard,
        phase="epoch_spec_broadcast",
    )
    source_pointer = D2DevEpochPointer(
        version=action_plan.source_version,
        epoch=source_epoch,
        certificate_sha256=canonical_sha256(
            {
                "scope": "D2 dev source pointer fixture",
                "action_plan_sha256": action_plan.content_sha256,
                "version": action_plan.source_version,
                "epoch": source_epoch,
            }
        ),
    )
    prepare_failure = (
        "synthetic pre-commit abort"
        if (
            args.case == "pre_commit_abort"
            and runtime.rank == runtime.world_size - 1
        )
        else None
    )
    local_prepare = D2DevPrivatePrepare.create(
        spec=epoch_spec,
        rank=runtime.rank,
        records=local_epoch_records,
        failure_reason=prepare_failure,
    )
    gathered_prepares = gather_d2_rank_metadata(
        runtime,
        local_prepare,
        guard=guard,
        phase="epoch_prepare_gather",
    )
    machine = None
    commit = None
    if runtime.is_primary:
        if gathered_prepares is None:
            raise RuntimeError("D2 dev prepares were not gathered")
        machine = D2DevEpochStateMachine(epoch_spec, source_pointer)
        commit = machine.decide_prepares(gathered_prepares)
    commit = broadcast_d2_metadata(
        runtime,
        commit,
        guard=guard,
        phase="epoch_commit_broadcast",
    )
    publication = None
    local_readback = None
    if args.case == "normal":
        readback_closure = close_d2_dev_wave(
            final_fragment,
            lineages,
            runtime.rank,
            runtime.world_size,
            action_plan.source_version,
            action_plan.target_version,
        )
        readback_failure = (
            None
            if readback_closure == closure
            else "private fragment readback mismatch"
        )
        local_readback = D2DevReadbackAck.create(
            spec=epoch_spec,
            commit=commit,
            rank=runtime.rank,
            records=local_epoch_records,
            failure_reason=readback_failure,
        )
        gathered_readbacks = gather_d2_rank_metadata(
            runtime,
            local_readback,
            guard=guard,
            phase="epoch_readback_gather",
        )
        if runtime.is_primary:
            if machine is None or gathered_readbacks is None:
                raise RuntimeError("D2 dev readbacks were not gathered")
            publication = machine.validate_readbacks(gathered_readbacks)
        publication = broadcast_d2_metadata(
            runtime,
            publication,
            guard=guard,
            phase="epoch_publication_broadcast",
        )
    visible_pointer = source_pointer
    if publication is not None and publication.publishes_target_epoch:
        visible_pointer = D2DevEpochPointer(
            version=action_plan.target_version,
            epoch=target_epoch,
            certificate_sha256=publication.certificate_sha256,
        )
    target_visible = (
        visible_pointer.version == action_plan.target_version
        and visible_pointer.epoch == target_epoch
    )
    epoch_case_passed = (
        args.case == "normal"
        and commit.committed
        and publication is not None
        and publication.publishes_target_epoch
        and target_visible
    ) or (
        args.case == "pre_commit_abort"
        and not commit.committed
        and publication is None
        and not target_visible
        and visible_pointer == source_pointer
    )
    source_payload_after = (
        {}
        if source_old_fragment is None
        else {
            str(record_id): d2_dev_record_payload_sha256(
                source_old_fragment,
                record_id,
            )
            for record_id in source_old_fragment.record_ids
        }
    )
    source_fixture_closed = (
        source_old_fragment is not None
        and source_old_fragment.record_ids
        == tuple(value.record_id for value in local_actions)
        and tuple(
            int(value)
            for value in source_old_fragment.lengths.detach().cpu()
        )
        == tuple(value.old_tokens for value in local_actions)
        and source_old_fragment.k.dtype == torch.float16
        and source_old_fragment.migration_anchor_version
        == action_plan.source_version
        and source_old_fragment.served_kv_target
        == action_plan.source_version
    ) or (source_old_fragment is None and not local_actions)
    source_unchanged = source_payload_before == source_payload_after
    source_preserved_after_abort = (
        args.case == "pre_commit_abort"
        and not target_visible
        and (source_old_fragment is not None or not local_actions)
    )
    source_fixture_reference_released_after_publication = False
    private_target_references_released_after_abort = False
    if args.case == "normal" and target_visible:
        del retained
        del retained_slice
        del compiled_source
        del source_old_fragment
        gc.collect()
        torch.cuda.empty_cache()
        source_fixture_reference_released_after_publication = True
    if args.case == "pre_commit_abort" and not target_visible:
        del final_fragment
        del prefix
        del compiled_prefix
        del scheduled_prefix
        del natural_result
        del scheduled_result
        del compiled_private
        gc.collect()
        torch.cuda.empty_cache()
        private_target_references_released_after_abort = True
    epoch_report = {
        "spec": epoch_spec.to_dict(),
        "local_records": [
            value.to_dict() for value in local_epoch_records
        ],
        "private_prepare": local_prepare.to_dict(),
        "commit": commit.to_dict(),
        "readback": (
            None
            if local_readback is None
            else local_readback.to_dict()
        ),
        "publication": (
            None if publication is None else publication.to_dict()
        ),
        "source_pointer": source_pointer.to_dict(),
        "visible_pointer": visible_pointer.to_dict(),
        "target_visible": target_visible,
        "case_passed": epoch_case_passed,
        "cow_source_fixture": {
            "fixture_only": True,
            "resident_source_manifest_claim": False,
            "performance_claim": False,
            "record_payload_sha256_before": source_payload_before,
            "record_payload_sha256_after_decision": source_payload_after,
            "source_fixture_closed": source_fixture_closed,
            "source_unchanged_through_decision": source_unchanged,
            "source_preserved_after_abort": (
                source_preserved_after_abort
            ),
            "source_fixture_reference_released_after_publication": (
                source_fixture_reference_released_after_publication
            ),
            "private_target_references_released_after_abort": (
                private_target_references_released_after_abort
            ),
        },
    }
    rank_report = {
        "rank": runtime.rank,
        "local_rank": runtime.local_rank,
        "device": str(runtime.device),
        "device_name": torch.cuda.get_device_name(runtime.device),
        "device_uuid": (
            f"GPU-{torch.cuda.get_device_properties(runtime.device).uuid}"
        ),
        "record_ids": list(local_record_ids),
        "routes": {
            "compiled": [value.record_id for value in compiled_actions],
            "scheduled_exact": [
                value.record_id for value in scheduled_actions
            ],
            "natural_exact": [
                value.record_id for value in natural_actions
            ],
        },
        "lineages": [value.to_dict() for value in lineages],
        "closure": closure.to_dict(),
        "lookup_evidence": {
            "source_old_kv_fixture": source_lookup,
            "natural_exact_prefix": natural_lookup,
            "scheduled_exact_retained": scheduled_lookup,
            "compiled_delta_append": compiled_delta_lookup,
            "scheduled_delta_append": scheduled_delta_lookup,
            "all_records_latest_append": latest_lookup,
        },
        "compiled_retained": compiled_retained_report,
        "epoch": epoch_report,
        "program": loaded_program,
        "scientific_result": False,
        "formal_stage_c": False,
    }
    gathered = gather_d2_rank_metadata(
        runtime,
        rank_report,
        guard=guard,
        phase="rank_report_gather",
    )
    phase_trace = guard.require_complete()
    rank_report["phase_trace"] = [
        {
            "ordinal": value.ordinal,
            "phase": value.phase,
            "token": value.token,
        }
        for value in phase_trace
    ]
    if gathered is None:
        return None
    reports = list(gathered)
    for report in reports:
        report["phase_trace"] = rank_report["phase_trace"]
    expected_record_ids = tuple(
        sorted(value.record_id for value in sample_actions)
    )
    observed_records = {
        int(record_id): int(report["rank"])
        for report in reports
        for record_id in report["record_ids"]
    }
    route_counts = {
        route: sum(
            len(report["routes"][route]) for report in reports
        )
        for route in ("compiled", "scheduled_exact", "natural_exact")
    }
    expected_route_counts = {
        route: sum(
            1
            for value in sample_actions
            if (
                route == "compiled"
                and value.requested_action == "compiled"
            )
            or (
                route != "compiled"
                and value.requested_reason == route
            )
        )
        for route in ("compiled", "scheduled_exact", "natural_exact")
    }
    checks = {
        "all_rank_closures_pass": all(
            report["closure"]["passed"] for report in reports
        ),
        "sample_coverage_exact": (
            tuple(sorted(observed_records)) == expected_record_ids
            and len(observed_records) == len(expected_record_ids)
        ),
        "owner_assignment_exact": all(
            observed_records[record_id] == owner_map[record_id]
            for record_id in expected_record_ids
        ),
        "route_counts_exact": route_counts == expected_route_counts,
        "final_token_count_exact": sum(
            report["closure"]["token_count"] for report in reports
        )
        == sum(value.final_tokens for value in sample_actions),
        "compiled_embedding_free": all(
            all(
                value == 0
                for value in report["compiled_retained"][
                    "phase_counters"
                ].values()
            )
            for report in reports
        ),
        "development_labels": all(
            report["scientific_result"] is False
            and report["formal_stage_c"] is False
            for report in reports
        ),
        "epoch_case_passes": all(
            report["epoch"]["case_passed"] for report in reports
        ),
        "cow_source_fixture_closed": all(
            report["epoch"]["cow_source_fixture"][
                "source_fixture_closed"
            ]
            and report["epoch"]["cow_source_fixture"][
                "source_unchanged_through_decision"
            ]
            for report in reports
        ),
        "dev_visibility_and_reference_release_order": all(
            (
                args.case == "normal"
                and report["epoch"]["target_visible"]
                and report["epoch"]["cow_source_fixture"][
                    "source_fixture_reference_released_after_publication"
                ]
                and not report["epoch"]["cow_source_fixture"][
                    "source_preserved_after_abort"
                ]
                and not report["epoch"]["cow_source_fixture"][
                    "private_target_references_released_after_abort"
                ]
            )
            or (
                args.case == "pre_commit_abort"
                and not report["epoch"]["target_visible"]
                and report["epoch"]["cow_source_fixture"][
                    "source_preserved_after_abort"
                ]
                and not report["epoch"]["cow_source_fixture"][
                    "source_fixture_reference_released_after_publication"
                ]
                and report["epoch"]["cow_source_fixture"][
                    "private_target_references_released_after_abort"
                ]
            )
            for report in reports
        ),
    }
    artifact: dict[str, object] = {
        "protocol": D2_DEV_WAVE_PROTOCOL,
        "status": "complete" if all(checks.values()) else "failed",
        "case": args.case,
        "scientific_result": False,
        "formal_stage_c": False,
        "scope": {
            "development_only": True,
            "sample_records": 16,
            "full_cohort": False,
            "performance_result": False,
            "timing_claim": False,
            "target_epoch_published": False,
            "development_target_pointer_published": all(
                report["epoch"]["target_visible"]
                for report in reports
            ),
            "development_epoch_namespace_only": True,
            "formal_stage_b_substitute": False,
            "formal_w4_substitute": False,
        },
        "configuration": {
            "world_size": runtime.world_size,
            "backend": runtime.backend,
            "owner_strategy": "strict_cow_lpt",
            "owner_map_sha256": d2_record_owner_map_sha256(owner_map),
            "source_version": action_plan.source_version,
            "target_version": action_plan.target_version,
        },
        "inputs": {
            "action_plan": {
                "path": str(action_path.relative_to(ROOT)),
                "content_sha256": action_plan.content_sha256,
                "file_sha256": file_sha256(action_path),
            },
            "stage_a_summary": {
                "path": str(stage_a_path.relative_to(ROOT)),
                "sha256": file_sha256(stage_a_path),
            },
            "sample_inputs": {
                "path": str(sample_path.relative_to(ROOT)),
                "content_sha256": sample["content_sha256"],
                "sha256": file_sha256(sample_path),
            },
            "source_checkpoint": source_descriptor,
            "target_checkpoint": target_descriptor,
            "program": program_descriptor,
        },
        "epoch_integration": {
            "requested_case": args.case,
            "state_machine_interface_reserved": True,
            "connected": True,
            "commit": reports[0]["epoch"]["commit"],
            "publication": reports[0]["epoch"]["publication"],
            "visible_pointer": reports[0]["epoch"]["visible_pointer"],
            "cow_fixture_only": True,
        },
        "route_counts": route_counts,
        "expected_route_counts": expected_route_counts,
        "record_owner_sample": [
            {
                "record_id": record_id,
                "owner_rank": observed_records[record_id],
            }
            for record_id in expected_record_ids
        ],
        "rank_reports": reports,
        "checks": checks,
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    return artifact


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runtime = init_d2_distributed_runtime(
        timeout_seconds=args.timeout_seconds,
    )
    try:
        artifact = _normal_rank(args, runtime)
        if runtime.is_primary:
            if artifact is None:
                raise RuntimeError("D2 dev primary produced no artifact")
            if args.output is not None:
                _write_json_atomic(_path(args.output), artifact)
            print(
                json.dumps(
                    {
                        "protocol": artifact["protocol"],
                        "status": artifact["status"],
                        "case": artifact["case"],
                        "world_size": artifact["configuration"][
                            "world_size"
                        ],
                        "checks": artifact["checks"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        close_d2_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
