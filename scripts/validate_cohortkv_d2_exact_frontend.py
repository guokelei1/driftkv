from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    D2ActionPlan,
    D2EmbeddingLookupCounter,
    JaggedMigratedKVBatch,
    build_d2_phase_ledger,
    build_retained_history_batch,
    build_segment_history_batch,
    exact_hidden_and_kv,
    exact_hidden_and_kv_from_item_embeddings,
    exact_jagged_kv,
)
from hstu_kvcache.migration.design2_plan import file_sha256
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    load_direct_oldkv_program,
)
from hstu_kvcache.models import HSTUConfig, HSTUKVCache
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_organic_windows,
    validate_long_context_plan,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_PLAN = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_TRAINING = (
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
DEFAULT_CHECKPOINT_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
)
DEFAULT_OUTPUT = (
    "configs/cohortkv_d2/stage_a_exact_frontend_validation.json"
)
PROTOCOL = "cohortkv_d2_stage_a_exact_frontend_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _batch(
    records,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    lengths = tuple(len(value.history) for value in records)
    width = max(lengths)
    item_ids = torch.zeros(
        len(records),
        width,
        dtype=torch.long,
        device=device,
    )
    behaviors = torch.zeros_like(item_ids)
    time_deltas = torch.zeros(
        len(records),
        width,
        dtype=torch.float32,
        device=device,
    )
    for row, record in enumerate(records):
        length = lengths[row]
        item_ids[row, :length] = torch.tensor(
            record.history.item_ids.copy(),
            dtype=torch.long,
            device=device,
        )
        behaviors[row, :length] = torch.tensor(
            record.history.behaviors.copy(),
            dtype=torch.long,
            device=device,
        )
        time_deltas[row, :length] = torch.tensor(
            record.history.time_deltas.copy(),
            dtype=torch.float32,
            device=device,
        )
    return {
        "item_ids": item_ids,
        "behaviors": behaviors,
        "time_deltas": time_deltas,
        "lengths": torch.tensor(
            lengths,
            dtype=torch.long,
            device=device,
        ),
    }


def _relative_l2(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    numerator = (
        candidate.float() - reference.float()
    ).square().sum()
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


def _top100_overlap(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    candidate_ids = candidate.topk(100, dim=1).indices.cpu()
    reference_ids = reference.topk(100, dim=1).indices.cpu()
    return float(
        sum(
            len(
                set(candidate_ids[row].tolist()).intersection(
                    reference_ids[row].tolist()
                )
            )
            / 100.0
            for row in range(candidate.shape[0])
        )
        / candidate.shape[0]
    )


def _lookup_observations(
    counter: D2EmbeddingLookupCounter,
) -> dict[str, dict[str, object]]:
    return {
        value.phase: value.to_dict()
        for value in counter.observations()
    }


def _replay_item_lookups(
    model,
    values: list[int],
    device: torch.device,
    chunk_size: int = 65536,
) -> None:
    for start in range(0, len(values), chunk_size):
        model.lookup_item_embeddings(
            torch.tensor(
                values[start : start + chunk_size],
                dtype=torch.long,
                device=device,
            )
        )


def run(args: argparse.Namespace) -> dict[str, object]:
    output_path = _path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            "D2 Stage A exact output exists; pass --force"
        )
    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or device.index is None
        or device.index >= torch.cuda.device_count()
    ):
        raise ValueError("D2 Stage A exact validation requires CUDA")
    action_plan_path = _path(args.action_plan)
    action_plan = D2ActionPlan.load(action_plan_path)
    prepared_path = _path(action_plan.provenance.prepared_data)
    training_path = _path(args.training_result)
    checkpoint_dir = _path(args.checkpoint_dir)
    training = json.loads(training_path.read_text())
    cfg = HSTUConfig(**training["model"])
    target_version = int(
        action_plan.target_version.removeprefix("theta")
    )
    checkpoint_path = (
        checkpoint_dir / f"theta_{target_version}.pt"
    )
    expected_checkpoint = next(
        value
        for value in json.loads(
            _path(action_plan.provenance.artifact).read_text()
        )["input_provenance"]["checkpoints"]
        if value["version"] == action_plan.target_version
    )
    if file_sha256(checkpoint_path) != expected_checkpoint["sha256"]:
        raise ValueError("D2 exact checkpoint hash differs")
    data_plan, metadata = load_prepared_kuairand_plan(
        prepared_path
    )
    validate_long_context_plan(data_plan, metadata, 4)
    user_ids = tuple(
        value.prepared_user_id for value in action_plan.records
    )
    windows = reconstruct_organic_windows(data_plan, user_ids)
    target_window = windows[target_version]
    selected = tuple(
        next(
            value
            for value in action_plan.records
            if value.requested_reason == reason
        )
        for reason in (
            "migrate",
            "scheduled_exact",
            "natural_exact",
        )
    )
    records = tuple(
        target_window.records[value.prepared_user_id]
        for value in selected
    )
    batch = _batch(records, device)
    model = load_checkpoint_model(
        cfg,
        checkpoint_dir,
        target_version,
        device,
    )
    state_keys_before = tuple(model.state_dict())
    model.train()
    lookup_calls = []
    handle = model.item_emb.register_forward_hook(
        lambda module, inputs, output: lookup_calls.append(
            {
                "padded_elements": int(inputs[0].numel()),
                "shape": list(inputs[0].shape),
            }
        )
    )
    started = time.perf_counter()
    direct_hidden, direct_cache = exact_hidden_and_kv(
        model,
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        batch["lengths"],
    )
    direct_calls = list(lookup_calls)
    lookup_calls.clear()
    item_vectors = model.lookup_item_embeddings(
        batch["item_ids"]
    ).detach()
    lookup_calls.clear()
    split_hidden, split_cache = (
        exact_hidden_and_kv_from_item_embeddings(
            model,
            item_vectors,
            batch["behaviors"],
            batch["time_deltas"],
            batch["lengths"],
        )
    )
    split_calls = list(lookup_calls)
    handle.remove()
    candidate_ids = torch.arange(
        1,
        cfg.num_prediction_items + 1,
        device=device,
    ).unsqueeze(0).expand(len(records), -1)
    direct_scores = model.score_candidates(
        direct_hidden,
        candidate_ids,
        batch["lengths"],
    )
    split_scores = model.score_candidates(
        split_hidden,
        candidate_ids,
        batch["lengths"],
    )
    mechanical = {
        "hidden_bitwise": torch.equal(
            direct_hidden,
            split_hidden,
        ),
        "k_bitwise": torch.equal(direct_cache.k, split_cache.k),
        "v_bitwise": torch.equal(direct_cache.v, split_cache.v),
        "scores_bitwise": torch.equal(direct_scores, split_scores),
        "top100_bitwise": torch.equal(
            direct_scores.topk(100, dim=1).indices,
            split_scores.topk(100, dim=1).indices,
        ),
        "public_lookup_calls": len(direct_calls) == 1,
        "embedded_lookup_calls": len(split_calls) == 0,
        "training_mode_restored": model.training,
        "state_dict_keys_unchanged": (
            tuple(model.state_dict()) == state_keys_before
        ),
    }
    transport = {}
    for name, dtype in (
        ("bf16_candidate", torch.bfloat16),
        ("fp16_candidate", torch.float16),
    ):
        transported = item_vectors.to(dtype).to(
            item_vectors.dtype
        )
        hidden, cache = exact_hidden_and_kv_from_item_embeddings(
            model,
            transported,
            batch["behaviors"],
            batch["time_deltas"],
            batch["lengths"],
        )
        scores = model.score_candidates(
            hidden,
            candidate_ids,
            batch["lengths"],
        )
        transport[name] = {
            "wire_element_bytes": dtype.itemsize,
            "hidden_relative_l2": _relative_l2(
                hidden,
                direct_hidden,
            ),
            "k_relative_l2": _relative_l2(
                cache.k,
                direct_cache.k,
            ),
            "v_relative_l2": _relative_l2(
                cache.v,
                direct_cache.v,
            ),
            "score_relative_l2": _relative_l2(
                scores,
                direct_scores,
            ),
            "top100_overlap": _top100_overlap(
                scores,
                direct_scores,
            ),
            "mechanical_equivalence": False,
        }
        del hidden
        del cache
        del scores
    append_record = selected[0]
    window_record = target_window.records[
        append_record.prepared_user_id
    ]
    history = window_record.history
    retained_record = type(window_record)(
        user_id=window_record.user_id,
        as_of_timestamp_ms=window_record.as_of_timestamp_ms,
        history=type(history)(
            events=type(history.events)(
                item_ids=history.item_ids[
                    : append_record.retained_tokens
                ],
                behaviors=history.behaviors[
                    : append_record.retained_tokens
                ],
                time_deltas=history.time_deltas[
                    : append_record.retained_tokens
                ],
                labels=history.labels[
                    : append_record.retained_tokens
                ],
                timestamps=history.timestamps[
                    : append_record.retained_tokens
                ],
            ),
            available_length_before_token_cap=(
                append_record.retained_tokens
            ),
            token_truncated=False,
        ),
        engaged_positive_item_ids=(),
        new_events=type(window_record.new_events).empty(),
    )
    retained_batch = _batch((retained_record,), device)
    retained_cache = model.compute_kv(
        retained_batch["item_ids"],
        retained_batch["behaviors"],
        retained_batch["time_deltas"],
        retained_batch["lengths"],
    )
    start = append_record.delta_start
    stop = append_record.final_tokens
    suffix_item_ids = torch.tensor(
        history.item_ids[start:stop].copy(),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    suffix_behaviors = torch.tensor(
        history.behaviors[start:stop].copy(),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    suffix_time_deltas = torch.tensor(
        history.time_deltas[start:stop].copy(),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    direct_append_hidden, direct_append_cache = (
        model.forward_with_cache(
            retained_cache,
            suffix_item_ids,
            suffix_behaviors,
            suffix_time_deltas,
        )
    )
    suffix_vectors = model.lookup_item_embeddings(
        suffix_item_ids
    )
    split_append_hidden, split_append_cache = (
        model.forward_with_cache_from_item_embeddings(
            retained_cache,
            suffix_vectors,
            suffix_behaviors,
            suffix_time_deltas,
        )
    )
    append_checks = {
        "hidden_bitwise": torch.equal(
            direct_append_hidden,
            split_append_hidden,
        ),
        "k_bitwise": torch.equal(
            direct_append_cache.k,
            split_append_cache.k,
        ),
        "v_bitwise": torch.equal(
            direct_append_cache.v,
            split_append_cache.v,
        ),
        "seq_len": (
            direct_append_cache.seq_len
            == split_append_cache.seq_len
            == append_record.final_tokens
        ),
    }
    upstream = json.loads(
        _path(action_plan.provenance.artifact).read_text()
    )
    compiler_descriptor = upstream["input_provenance"]["compiler"]
    compiler_path = _path(compiler_descriptor["path"])
    if file_sha256(compiler_path) != compiler_descriptor["sha256"]:
        raise ValueError("D2 compiler artifact hash differs")
    compiler = json.loads(compiler_path.read_text())
    pair = next(
        value
        for value in compiler["pairs"]
        if value["source_version"] == action_plan.source_version
        and value["target_version"] == action_plan.target_version
    )
    program_descriptor = pair["direct_program"]
    program_cpu, loaded_program_descriptor = (
        load_direct_oldkv_program(
            _path(program_descriptor["path"]),
            expected_sha256=program_descriptor["sha256"],
            expected_source_version=action_plan.source_version,
            expected_target_version=action_plan.target_version,
            expected_num_layers=cfg.num_layers,
            expected_kv_width=cfg.hidden_size,
        )
    )
    operator = DirectOldKVFusedOperator()
    program = operator.prepare_program(program_cpu, device)
    source_version = int(
        action_plan.source_version.removeprefix("theta")
    )
    source_model = load_checkpoint_model(
        cfg,
        checkpoint_dir,
        source_version,
        device,
    )
    source_batch = build_retained_history_batch(
        record_ids=(append_record.record_id,),
        migration_anchor_version=action_plan.source_version,
        histories=(history,),
        retained_tokens=(append_record.retained_tokens,),
        device=device,
    )
    source_cache = exact_jagged_kv(
        source_model,
        source_batch,
        action_plan.source_version,
        dtype=torch.float16,
    )
    del source_model
    del source_batch
    migrated_cache = JaggedMigratedKVBatch(
        record_ids=source_cache.record_ids,
        migration_anchor_version=action_plan.source_version,
        served_kv_target=action_plan.target_version,
        k=torch.empty_like(source_cache.k),
        v=torch.empty_like(source_cache.v),
        lengths=source_cache.lengths.clone(),
        offsets=source_cache.offsets.clone(),
    )
    scheduled_record = selected[1]
    natural_record = selected[2]
    scheduled_history = target_window.records[
        scheduled_record.prepared_user_id
    ].history
    natural_history = target_window.records[
        natural_record.prepared_user_id
    ].history
    scheduled_raw = build_retained_history_batch(
        record_ids=(scheduled_record.record_id,),
        migration_anchor_version=action_plan.target_version,
        histories=(scheduled_history,),
        retained_tokens=(scheduled_record.retained_tokens,),
        device=device,
    )
    natural_raw = build_retained_history_batch(
        record_ids=(natural_record.record_id,),
        migration_anchor_version=action_plan.target_version,
        histories=(natural_history,),
        retained_tokens=(natural_record.target_prefix_tokens,),
        device=device,
    )
    delta_raw = build_segment_history_batch(
        record_ids=(append_record.record_id,),
        migration_anchor_version=action_plan.target_version,
        histories=(history,),
        starts=(append_record.delta_start,),
        stops=(append_record.target_prefix_tokens,),
        device=device,
    )
    latest_raw = build_segment_history_batch(
        record_ids=(append_record.record_id,),
        migration_anchor_version=action_plan.target_version,
        histories=(history,),
        starts=(append_record.target_prefix_tokens,),
        stops=(append_record.final_tokens,),
        device=device,
    )
    with D2EmbeddingLookupCounter(model.item_emb) as phase_counter:
        with phase_counter.phase("compiled_retained", 0):
            operator.execute_into(
                program,
                source_cache,
                migrated_cache,
            )
        with phase_counter.phase(
            "scheduled_exact_retained",
            scheduled_record.retained_tokens,
        ):
            scheduled_cache = exact_jagged_kv(
                model,
                scheduled_raw,
                action_plan.target_version,
                dtype=torch.float16,
            )
        with phase_counter.phase(
            "natural_exact_target_prefix",
            natural_record.target_prefix_tokens,
        ):
            natural_cache = exact_jagged_kv(
                model,
                natural_raw,
                action_plan.target_version,
                dtype=torch.float16,
            )
        padded_migrated = HSTUKVCache(
            k=migrated_cache.k.unsqueeze(1),
            v=migrated_cache.v.unsqueeze(1),
            seq_len=append_record.retained_tokens,
        )
        with phase_counter.phase(
            "compiled_delta_append",
            append_record.delta_tokens,
        ):
            _, padded_migrated = model.forward_with_cache(
                padded_migrated,
                delta_raw.item_ids,
                delta_raw.behaviors,
                delta_raw.time_deltas,
            )
        with phase_counter.phase(
            "compiled_latest_append",
            append_record.latest_tokens,
        ):
            _, padded_migrated = model.forward_with_cache(
                padded_migrated,
                latest_raw.item_ids,
                latest_raw.behaviors,
                latest_raw.time_deltas,
            )
        representative_lookup = _lookup_observations(
            phase_counter
        )
    phase_ids = {
        "compiled_retained": [],
        "scheduled_exact_retained": [],
        "natural_exact_target_prefix": [],
        "delta_append": [],
        "latest_append": [],
    }
    for record in action_plan.records:
        target_history = target_window.records[
            record.prepared_user_id
        ].history
        if record.requested_reason == "scheduled_exact":
            phase_ids["scheduled_exact_retained"].extend(
                int(value)
                for value in target_history.item_ids[
                    : record.retained_tokens
                ]
            )
        elif record.requested_reason == "natural_exact":
            phase_ids["natural_exact_target_prefix"].extend(
                int(value)
                for value in target_history.item_ids[
                    : record.target_prefix_tokens
                ]
            )
        if record.requested_reason != "natural_exact":
            phase_ids["delta_append"].extend(
                int(value)
                for value in target_history.item_ids[
                    record.delta_start : record.target_prefix_tokens
                ]
            )
        phase_ids["latest_append"].extend(
            int(value)
            for value in target_history.item_ids[
                record.target_prefix_tokens : record.final_tokens
            ]
        )
    with D2EmbeddingLookupCounter(model.item_emb) as replay_counter:
        for phase, values in phase_ids.items():
            with replay_counter.phase(phase, len(values)):
                _replay_item_lookups(model, values, device)
        replay_lookup = _lookup_observations(replay_counter)
    ledger = build_d2_phase_ledger(action_plan, embedding_dim=512)
    expected_replay_tokens = {
        value.phase: value.lookup_tokens for value in ledger.mixed
    }
    representative_expected = {
        "compiled_retained": 0,
        "scheduled_exact_retained": (
            scheduled_record.retained_tokens
        ),
        "natural_exact_target_prefix": (
            natural_record.target_prefix_tokens
        ),
        "compiled_delta_append": append_record.delta_tokens,
        "compiled_latest_append": append_record.latest_tokens,
    }
    representative_checks = {
        phase: (
            representative_lookup[phase]["logical_lookup_tokens"]
            == expected
            and (
                representative_lookup[phase]["lookup_calls"] == 0
                if expected == 0
                else representative_lookup[phase]["lookup_calls"] > 0
            )
        )
        for phase, expected in representative_expected.items()
    }
    replay_checks = {
        phase: (
            replay_lookup[phase]["logical_lookup_tokens"]
            == len(values)
            == expected_replay_tokens[phase]
            and (
                replay_lookup[phase]["lookup_calls"] == 0
                if not values
                else replay_lookup[phase]["lookup_calls"] > 0
            )
        )
        for phase, values in phase_ids.items()
    }
    del scheduled_cache
    del natural_cache
    checks = {
        **mechanical,
        "append_embedded_equivalence": all(
            append_checks.values()
        ),
        "selected_action_coverage": len(selected) == 3,
        "target_window_hash": (
            target_window.content_sha256
            == action_plan.provenance.target_window_content_sha256
        ),
        "representative_phase_lookup": all(
            representative_checks.values()
        ),
        "full_plan_lookup_request_replay": all(
            replay_checks.values()
        ),
        "compiled_output_finite": (
            bool(torch.isfinite(migrated_cache.k).all())
            and bool(torch.isfinite(migrated_cache.v).all())
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"D2 Stage A exact frontend checks failed: {checks}"
        )
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "action_plan": {
            "path": str(action_plan_path.relative_to(ROOT)),
            "content_sha256": action_plan.content_sha256,
            "file_sha256": file_sha256(action_plan_path),
        },
        "configuration": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "target_version": action_plan.target_version,
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": expected_checkpoint["sha256"],
            },
            "record_ids": [
                value.record_id for value in selected
            ],
            "requested_reasons": [
                value.requested_reason for value in selected
            ],
            "lengths": [
                int(value) for value in batch["lengths"].cpu()
            ],
            "model_weight_dtype": str(
                next(model.parameters()).dtype
            ).removeprefix("torch."),
            "exact_compute_dtype": "float32",
            "frontend_item_vector_dtype": "float32",
            "frontend_item_vector_element_bytes": 4,
            "representative_jagged_publication_dtype": "float16",
        },
        "mechanical_equivalence": {
            **mechanical,
            "logical_lookup_tokens": int(
                batch["lengths"].sum().item()
            ),
            "public_lookup_observations": direct_calls,
            "embedded_lookup_observations": split_calls,
            "state_dict_key_sha256": hashlib.sha256(
                json.dumps(state_keys_before).encode()
            ).hexdigest(),
        },
        "transport_dtype_characterization": transport,
        "append_equivalence": append_checks,
        "phase_lookup_instrumentation": {
            "representative_execution": representative_lookup,
            "representative_checks": representative_checks,
            "full_plan_request_replay": replay_lookup,
            "full_plan_replay_checks": replay_checks,
            "program": {
                "path": program_descriptor["path"],
                "sha256": program_descriptor["sha256"],
                "loaded_sha256": loaded_program_descriptor["sha256"],
            },
            "scope": {
                "representative_execution_runs_real_compiled_exact_and_append_paths": True,
                "full_plan_replay_runs_embedding_lookup_requests_without_dense_trunk": True,
                "physical_collective_bytes_measured": False,
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
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
