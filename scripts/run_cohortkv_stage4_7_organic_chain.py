from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from cohortkv_stage4_7_common import (
    CHAIN_OUTPUT,
    CHAIN_PROTOCOL,
    CHECKPOINT_DIR,
    COMPILER_OUTPUT,
    COMPILER_PROTOCOL,
    EXPERIMENT_PROTOCOL,
    PREPARED_PATH,
    RUNTIME_DIR,
    TRAINING_PATH,
    direct_program_path,
    history_view_sha256,
    load_inputs,
    sha256,
)
from evaluate_cohortkv_stage4_6_lifecycle import (
    LAUNCH,
    exact_batch,
    execute_direct,
    timed_cuda,
)
from motivation_validity import move_batch, seed_everything
from run_cohortkv_stage4_6_full_chain import (
    cache_hidden_scores,
    semantic_pair,
    task_metrics,
)
from scipy.stats import spearmanr

from hstu_kvcache.data import collate_batch
from hstu_kvcache.migration import (
    BalancedLifecyclePolicy,
    CacheLifecycleState,
    JaggedMigratedKVBatch,
    JaggedTokenSlice,
    LifecycleDecision,
    absolute_log_norm_ratio_values,
    aggregate_layer_values,
    append_jagged_suffix,
    assemble_jagged_rows,
    pack_padded_cache,
    relative_cache_values,
    select_jagged_rows,
    tail_slice_jagged_cache,
)
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    load_direct_oldkv_program,
)
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_organic_windows,
)
from hstu_kvcache.utils import save_json

EXACT_FRACTION = 0.20
MAX_MIGRATION_DEPTH = 4
SCHEDULER_SEED = 0
BATCH_SIZE = 4
NUM_EDGES = 11


@dataclass(frozen=True)
class TransitionDescriptor:
    status: str
    old_history_hash: str | None
    new_history_hash: str | None
    old_length: int
    new_length: int
    overlap: int
    evicted: int
    appended: int
    previous_actual_consumed: bool
    retained_old_start: int
    appended_new_start: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=PREPARED_PATH)
    parser.add_argument("--training-result", default=TRAINING_PATH)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--compiler-result", default=COMPILER_OUTPUT)
    parser.add_argument("--runtime-dir", default=RUNTIME_DIR)
    parser.add_argument("--output", default=CHAIN_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != BATCH_SIZE or args.seed != 0:
        raise ValueError("organic runner freezes batch size 4 and seed 0")
    if not args.smoke_test and torch.device(args.device).type != "cuda":
        raise ValueError("organic full chain requires CUDA")


def constant_policy() -> BalancedLifecyclePolicy:
    return BalancedLifecyclePolicy(
        max_migration_depth=MAX_MIGRATION_DEPTH,
        exact_fractions=(EXACT_FRACTION,) * NUM_EDGES,
        edge_severities=(0.0,) * NUM_EDGES,
        scheduler_seed=SCHEDULER_SEED,
    )


def _selector_tie(
    record_id: int,
    source_version: int,
    scheduler_seed: int,
) -> int:
    payload = (
        f"{scheduler_seed}:{source_version}:{record_id}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8])


def select_norm_shift_decisions(
    states: tuple[CacheLifecycleState, ...],
    target_version: int,
    norm_shifts: dict[int, float],
    exact_fraction: float = EXACT_FRACTION,
    max_migration_depth: int = MAX_MIGRATION_DEPTH,
    scheduler_seed: int = SCHEDULER_SEED,
) -> tuple[LifecycleDecision, ...]:
    if (
        not states
        or not 0 < exact_fraction < 1
        or max_migration_depth < 1
        or scheduler_seed < 0
        or len({state.record_id for state in states}) != len(states)
        or set(norm_shifts) != {state.record_id for state in states}
        or any(
            target_version != state.served_version + 1
            for state in states
        )
        or any(
            not np.isfinite(norm_shifts[state.record_id])
            or norm_shifts[state.record_id] < 0
            for state in states
        )
    ):
        raise ValueError("organic norm-shift selector inputs are invalid")
    source_version = target_version - 1
    target_exact = int(np.floor(exact_fraction * len(states) + 0.5))
    mandatory = {
        state.record_id
        for state in states
        if state.migration_depth >= max_migration_depth
    }
    optional = sorted(
        (
            state
            for state in states
            if state.record_id not in mandatory
        ),
        key=lambda state: (
            -state.migration_depth,
            -norm_shifts[state.record_id],
            _selector_tie(
                state.record_id,
                source_version,
                scheduler_seed,
            ),
            state.record_id,
        ),
    )
    selected = set(mandatory)
    selected.update(
        state.record_id
        for state in optional[
            : max(0, target_exact - len(mandatory))
        ]
    )
    output = []
    for state in states:
        shift = float(norm_shifts[state.record_id])
        if state.record_id in mandatory:
            action = "exact"
            reason = "depth_deadline_after_probe"
        elif state.record_id in selected:
            action = "exact"
            reason = "norm_shift_exact_quota"
        else:
            action = "migrate"
            reason = "norm_shift_migrate"
        output.append(
            LifecycleDecision(
                record_id=state.record_id,
                source_version=source_version,
                target_version=target_version,
                action=action,
                reason=reason,
                predicted_risk=shift,
                candidate_evaluated=True,
            )
        )
    return tuple(output)


def selector_audit(
    states: tuple[CacheLifecycleState, ...],
    decisions: tuple[LifecycleDecision, ...],
    norm_shifts: dict[int, float],
    exact_fraction: float = EXACT_FRACTION,
    max_migration_depth: int = MAX_MIGRATION_DEPTH,
    scheduler_seed: int = SCHEDULER_SEED,
) -> dict:
    if (
        len(states) != len(decisions)
        or set(norm_shifts) != {state.record_id for state in states}
        or set(norm_shifts) != {
            decision.record_id for decision in decisions
        }
    ):
        raise ValueError("organic selector audit inputs differ")
    mandatory = {
        state.record_id
        for state in states
        if state.migration_depth >= max_migration_depth
    }
    target_exact = int(np.floor(exact_fraction * len(states) + 0.5))
    quota_slots = max(0, target_exact - len(mandatory))
    optional = sorted(
        (
            state
            for state in states
            if state.record_id not in mandatory
        ),
        key=lambda state: (
            -state.migration_depth,
            -norm_shifts[state.record_id],
            _selector_tie(
                state.record_id,
                decisions[0].source_version if decisions else 0,
                scheduler_seed,
            ),
            state.record_id,
        ),
    )
    selected_optional = optional[:quota_slots]
    cutoff = selected_optional[-1] if selected_optional else None
    cutoff_ties = (
        [
            state
            for state in optional
            if state.migration_depth == cutoff.migration_depth
            and norm_shifts[state.record_id]
            == norm_shifts[cutoff.record_id]
        ]
        if cutoff is not None
        else []
    )
    selected_ids = {
        decision.record_id
        for decision in decisions
        if decision.action == "exact"
    }
    selected_cutoff_ties = {
        state.record_id
        for state in cutoff_ties
        if state.record_id in selected_ids
    }
    return {
        "target_exact_records": target_exact,
        "eligible_records": len(states),
        "realized_exact_records": len(selected_ids),
        "realized_exact_fraction": (
            len(selected_ids) / len(states) if states else None
        ),
        "mandatory_depth_exact_records": sum(
            decision.reason == "depth_deadline_after_probe"
            for decision in decisions
        ),
        "norm_shift_quota_exact_records": sum(
            decision.reason == "norm_shift_exact_quota"
            for decision in decisions
        ),
        "norm_shift_unique_count": len(set(norm_shifts.values())),
        "quota_cutoff": (
            {
                "migration_depth": cutoff.migration_depth,
                "norm_shift_q090": norm_shifts[cutoff.record_id],
            }
            if cutoff is not None
            else None
        ),
        "cutoff_tie_records": len(cutoff_ties),
        "cutoff_tie_selected_records": len(selected_cutoff_ties),
        "sha256_boundary_tie_used": bool(
            selected_cutoff_ties
            and len(selected_cutoff_ties) < len(cutoff_ties)
        ),
        "sha256_role": "only exact depth-and-norm-shift ties",
    }


def fixed_record_groups(
    manifest: dict,
    batch_size: int = BATCH_SIZE,
) -> tuple[tuple[dict, ...], ...]:
    records = tuple(
        sorted(
            manifest["records"],
            key=lambda value: int(value["record_id"]),
        )
    )
    if (
        batch_size < 1
        or not records
        or tuple(int(value["record_id"]) for value in records)
        != tuple(range(len(records)))
        or len({int(value["user_id"]) for value in records}) != len(records)
    ):
        raise ValueError("organic fixed record groups are invalid")
    return tuple(
        records[start : start + batch_size]
        for start in range(0, len(records), batch_size)
    )


def _history_length(record) -> int:
    return 0 if record.history is None else len(record.history)


def _suffix_prefix_identity_overlap(
    old_identities: tuple[object, ...],
    new_prefix_identities: tuple[object, ...],
) -> int:
    if not old_identities or not new_prefix_identities:
        return 0
    separator = object()
    combined = [
        *new_prefix_identities,
        separator,
        *old_identities,
    ]
    prefix = [0] * len(combined)
    for index in range(1, len(combined)):
        length = prefix[index - 1]
        while length and combined[index] != combined[length]:
            length = prefix[length - 1]
        if combined[index] == combined[length]:
            length += 1
        prefix[index] = length
    return min(
        prefix[-1],
        len(old_identities),
        len(new_prefix_identities),
    )


def _label_free_history_identities(
    record,
    prefix: bool,
) -> tuple[tuple[int, int, int], ...]:
    history = record.history
    if history is None:
        return ()
    stop = len(history) - 1 if prefix else len(history)
    return tuple(
        (
            int(history.timestamps[index]),
            int(history.item_ids[index]),
            int(history.behaviors[index]),
        )
        for index in range(stop)
    )


def transition_descriptor(
    old_record,
    new_record,
    previous_resident: bool,
) -> TransitionDescriptor:
    old_history = old_record.history
    new_history = new_record.history
    old_hash = old_record.history_sha256
    new_hash = new_record.history_sha256
    if new_history is None:
        return TransitionDescriptor(
            status="expired" if previous_resident else "absent",
            old_history_hash=old_hash,
            new_history_hash=None,
            old_length=_history_length(old_record),
            new_length=0,
            overlap=0,
            evicted=_history_length(old_record),
            appended=0,
            previous_actual_consumed=False,
            retained_old_start=_history_length(old_record),
            appended_new_start=0,
        )
    new_prefix_length = max(0, len(new_history) - 1)
    if new_prefix_length == 0:
        return TransitionDescriptor(
            status="short_no_prefix",
            old_history_hash=old_hash,
            new_history_hash=new_hash,
            old_length=_history_length(old_record),
            new_length=0,
            overlap=0,
            evicted=_history_length(old_record),
            appended=0,
            previous_actual_consumed=False,
            retained_old_start=_history_length(old_record),
            appended_new_start=0,
        )
    if old_history is None or not previous_resident:
        return TransitionDescriptor(
            status="cold",
            old_history_hash=old_hash,
            new_history_hash=new_hash,
            old_length=_history_length(old_record),
            new_length=new_prefix_length,
            overlap=0,
            evicted=_history_length(old_record),
            appended=new_prefix_length,
            previous_actual_consumed=False,
            retained_old_start=_history_length(old_record),
            appended_new_start=0,
        )
    overlap_length = _suffix_prefix_identity_overlap(
        _label_free_history_identities(old_record, prefix=False),
        _label_free_history_identities(new_record, prefix=True),
    )
    status = "continued" if overlap_length > 0 else "zero_overlap"
    return TransitionDescriptor(
        status=status,
        old_history_hash=old_hash,
        new_history_hash=new_hash,
        old_length=len(old_history),
        new_length=new_prefix_length,
        overlap=overlap_length,
        evicted=len(old_history) - overlap_length,
        appended=new_prefix_length - overlap_length,
        previous_actual_consumed=overlap_length > 0,
        retained_old_start=len(old_history) - overlap_length,
        appended_new_start=overlap_length,
    )


def cost_summary(
    costs: dict[str, float],
    all_exact_reference_ms: float,
) -> dict:
    required = {
        "foreground_evict",
        "foreground_incremental_append",
        "candidate_transform",
        "router_probe",
        "exact_refresh",
        "natural_direct_exact",
        "publication",
        "common_latest",
        "common_publication",
    }
    if (
        set(costs) != required
        or any(not np.isfinite(value) or value < 0 for value in costs.values())
        or costs.get("natural_direct_exact", 0.0)
        > costs.get("exact_refresh", 0.0)
        or not np.isfinite(all_exact_reference_ms)
        or all_exact_reference_ms <= 0
    ):
        raise ValueError("organic cost ledger is invalid")
    foreground = (
        costs["foreground_evict"]
        + costs["foreground_incremental_append"]
    )
    update = (
        costs["candidate_transform"]
        + costs["router_probe"]
        + costs["exact_refresh"]
        + costs["publication"]
    )
    common = costs["common_latest"] + costs["common_publication"]
    return {
        **costs,
        "foreground_ms": foreground,
        "update_only_ms": update,
        "all_exact_reference_ms": all_exact_reference_ms,
        "primary_update_only_ratio": update / all_exact_reference_ms,
        "symmetric_lifecycle_numerator_ms": foreground + update,
        "symmetric_lifecycle_denominator_ms": (
            foreground + all_exact_reference_ms
        ),
        "symmetric_lifecycle_ratio": (
            (foreground + update)
            / (foreground + all_exact_reference_ms)
        ),
        "common_shared_ms": common,
        "common_inclusive_numerator_ms": foreground + update + common,
        "common_inclusive_denominator_ms": (
            foreground + all_exact_reference_ms + common
        ),
        "common_inclusive_ratio": (
            (foreground + update + common)
            / (foreground + all_exact_reference_ms + common)
        ),
        "conservative_asymmetric_numerator_ms": foreground + update,
        "conservative_asymmetric_ratio": (
            (foreground + update) / all_exact_reference_ms
        ),
        "update_only_ratio": update / all_exact_reference_ms,
        "migration_ms": costs["candidate_transform"],
        "primary_and_symmetric_exclude_common_latest": True,
        "primary_and_symmetric_exclude_common_publication": True,
    }


def posthoc_selector_diagnostics(
    norm_shifts: dict[int, float],
    candidate_errors: dict[int, float],
    decisions: tuple[LifecycleDecision, ...],
    exact_fraction: float = EXACT_FRACTION,
) -> dict:
    if (
        not norm_shifts
        and not candidate_errors
        and not decisions
        and 0 < exact_fraction < 1
    ):
        return {
            "posthoc_only": True,
            "actions_changed": False,
            "records": 0,
            "spearman_norm_shift_vs_candidate_error": None,
            "selected_exact_candidate_error": {
                "records": 0,
                "mean": None,
                "q90": None,
            },
            "migrated_candidate_error": {
                "records": 0,
                "mean": None,
                "q90": None,
            },
            "oracle_top_error_fraction": exact_fraction,
            "oracle_top_error_records": 0,
            "selected_exact_records": 0,
            "selected_vs_oracle_overlap_records": 0,
            "selected_vs_oracle_overlap_fraction": None,
        }
    if (
        not norm_shifts
        or set(norm_shifts) != set(candidate_errors)
        or set(norm_shifts) != {value.record_id for value in decisions}
        or any(
            not np.isfinite(value) or value < 0
            for value in (
                *norm_shifts.values(),
                *candidate_errors.values(),
            )
        )
        or not 0 < exact_fraction < 1
    ):
        raise ValueError("posthoc selector diagnostic inputs are invalid")
    record_ids = tuple(sorted(norm_shifts))
    shifts = np.asarray([norm_shifts[value] for value in record_ids])
    errors = np.asarray([candidate_errors[value] for value in record_ids])
    correlation = spearmanr(shifts, errors).statistic
    selected = {
        value.record_id for value in decisions if value.action == "exact"
    }
    migrated = {
        value.record_id for value in decisions if value.action == "migrate"
    }
    oracle_count = int(np.floor(exact_fraction * len(record_ids) + 0.5))
    oracle = set(
        sorted(
            record_ids,
            key=lambda record_id: (
                -candidate_errors[record_id],
                record_id,
            ),
        )[:oracle_count]
    )

    def error_summary(selected_ids: set[int]) -> dict:
        values = np.asarray(
            [candidate_errors[value] for value in sorted(selected_ids)]
        )
        return {
            "records": len(values),
            "mean": float(values.mean()) if len(values) else None,
            "q90": (
                float(np.quantile(values, 0.9))
                if len(values)
                else None
            ),
        }

    intersection = selected & oracle
    return {
        "posthoc_only": True,
        "actions_changed": False,
        "records": len(record_ids),
        "spearman_norm_shift_vs_candidate_error": (
            float(correlation) if np.isfinite(correlation) else None
        ),
        "selected_exact_candidate_error": error_summary(selected),
        "migrated_candidate_error": error_summary(migrated),
        "oracle_top_error_fraction": exact_fraction,
        "oracle_top_error_records": len(oracle),
        "selected_exact_records": len(selected),
        "selected_vs_oracle_overlap_records": len(intersection),
        "selected_vs_oracle_overlap_fraction": (
            len(intersection) / len(oracle) if oracle else None
        ),
    }


def validate_windows(
    windows,
    manifest: dict,
) -> dict[str, bool]:
    expected_users = tuple(
        sorted(int(value["user_id"]) for value in manifest["records"])
    )
    expected_dates = tuple(manifest["timeline"]["target_dates"])
    versions = tuple(int(window.version) for window in windows)
    dates = tuple(str(window.target_date) for window in windows)
    user_sets_match = all(
        tuple(window.records) == expected_users for window in windows
    )
    history_identity_lengths_match = all(
        len(record.history_event_identities)
        == (0 if record.history is None else len(record.history))
        for window in windows
        for record in window.records.values()
    )
    target_identity_lengths_match = all(
        len(record.new_event_identities) == len(record.new_events)
        for window in windows
        for record in window.records.values()
    )
    prior_partition_suffixes = True
    if user_sets_match and windows:
        known_by_user = {
            user_id: list(
                windows[0].records[user_id].history_event_identities
            )
            for user_id in expected_users
        }
        for window in windows:
            for user_id in expected_users:
                record = window.records[user_id]
                current = tuple(record.history_event_identities)
                known = known_by_user[user_id]
                if len(current) > len(known) or (
                    current and tuple(known[-len(current) :]) != current
                ):
                    prior_partition_suffixes = False
                known.extend(record.new_event_identities)
    request_events_not_before_request_start = all(
        len(record.new_events) == 0
        or bool(
            np.all(
                record.new_events.timestamps
                >= int(record.as_of_timestamp_ms)
            )
        )
        for window in windows
        for record in window.records.values()
    )
    checks = {
        "twelve_ordered_versions": versions == tuple(range(12)),
        "manifest_target_dates": dates == expected_dates,
        "fixed_base_only_users": user_sets_match,
        "resident_history_identity_lengths_match": (
            history_identity_lengths_match
        ),
        "target_partition_identity_lengths_match": (
            target_identity_lengths_match
        ),
        "history_is_suffix_of_prior_date_partitions": (
            prior_partition_suffixes
        ),
        "request_events_not_before_request_start": (
            request_events_not_before_request_start
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"organic window causality differs: {checks}")
    return checks


def raw_timestamp_overlap_diagnostic(windows) -> dict:
    def empty_counts() -> dict[str, int | None]:
        return {
            "record_windows": 0,
            "active_record_windows": 0,
            "inactive_record_windows": 0,
            "resident_record_windows": 0,
            "active_resident_record_windows": 0,
            "inactive_resident_record_windows": 0,
            "history_tokens": 0,
            "overlap_record_windows": 0,
            "active_overlap_record_windows": 0,
            "inactive_overlap_record_windows": 0,
            "overlap_history_tokens": 0,
            "active_overlap_history_tokens": 0,
            "inactive_overlap_history_tokens": 0,
            "maximum_lead_ms": None,
        }

    def update(counts: dict, record) -> None:
        counts["record_windows"] += 1
        active = len(record.new_events) > 0
        activity_prefix = "active" if active else "inactive"
        counts[f"{activity_prefix}_record_windows"] += 1
        if record.history is None:
            return
        timestamps = np.asarray(record.history.timestamps, dtype=np.int64)
        lead = timestamps - int(record.as_of_timestamp_ms)
        overlap = lead >= 0
        overlap_tokens = int(overlap.sum())
        counts["resident_record_windows"] += 1
        counts["history_tokens"] += int(len(timestamps))
        counts[f"{activity_prefix}_resident_record_windows"] += 1
        if not overlap_tokens:
            return
        counts["overlap_record_windows"] += 1
        counts["overlap_history_tokens"] += overlap_tokens
        counts[f"{activity_prefix}_overlap_record_windows"] += 1
        counts[f"{activity_prefix}_overlap_history_tokens"] += overlap_tokens
        maximum = int(lead[overlap].max())
        previous = counts["maximum_lead_ms"]
        counts["maximum_lead_ms"] = (
            maximum if previous is None else max(previous, maximum)
        )

    def finish(counts: dict) -> dict:
        resident = int(counts["resident_record_windows"])
        tokens = int(counts["history_tokens"])
        maximum = counts["maximum_lead_ms"]
        return {
            **counts,
            "overlap_record_fraction_of_resident": (
                counts["overlap_record_windows"] / resident
                if resident
                else None
            ),
            "overlap_token_fraction": (
                counts["overlap_history_tokens"] / tokens
                if tokens
                else None
            ),
            "maximum_lead_seconds": (
                maximum / 1000.0 if maximum is not None else None
            ),
        }

    total = empty_counts()
    by_version = []
    for window in windows:
        current = empty_counts()
        for record in window.records.values():
            update(total, record)
            update(current, record)
        by_version.append(
            {
                "version": int(window.version),
                "target_date": str(window.target_date),
                **finish(current),
            }
        )
    return {
        "validity_gate": False,
        "interpretation": (
            "raw timestamp overlap across canonical date partitions"
        ),
        **finish(total),
        "by_version": by_version,
    }


def validate_compiler_payload(
    compiler: dict,
    manifest: dict,
    windows,
    checkpoints: list[dict] | None = None,
) -> dict[str, bool]:
    pairs = compiler.get("pairs")
    compiler_windows = compiler.get("windows")
    all_user_ids = tuple(
        int(value["user_id"]) for value in manifest["records"]
    )
    fit_user_ids = tuple(
        int(value["user_id"])
        for value in manifest["records"]
        if value["evaluation_role"] == "fit"
    )
    checkpoint_hashes = (
        {
            value["version"]: value["sha256"]
            for value in checkpoints
        }
        if checkpoints is not None
        else None
    )
    checks = {
        "protocol": compiler.get("protocol") == COMPILER_PROTOCOL,
        "experiment_protocol": (
            compiler.get("experiment_protocol") == EXPERIMENT_PROTOCOL
        ),
        "complete": compiler.get("status") == "complete",
        "manifest": (
            compiler.get("manifest", {}).get("content_sha256")
            == manifest["content_sha256"]
        ),
        "checkpoints": checkpoints is None
        or [
            {
                "version": value.get("version"),
                "sha256": value.get("sha256"),
                "bytes": value.get("bytes"),
            }
            for value in compiler.get("inputs", {}).get(
                "checkpoints",
                [],
            )
        ]
        == [
            {
                "version": value["version"],
                "sha256": value["sha256"],
                "bytes": value["bytes"],
            }
            for value in checkpoints
        ],
        "windows": isinstance(compiler_windows, list)
        and len(compiler_windows) == 12
        and all(
            compiler_windows[index].get("version") == index
            and compiler_windows[index].get("target_date")
            == windows[index].target_date
            and compiler_windows[index].get("history_view_sha256")
            == history_view_sha256(windows[index], all_user_ids)
            for index in range(12)
        ),
        "pairs": isinstance(pairs, list)
        and len(pairs) == NUM_EDGES
        and all(
            pairs[source].get("source_version") == f"theta{source}"
            and pairs[source].get("target_version")
            == f"theta{source + 1}"
            and pairs[source].get("history_target_date")
            == windows[source + 1].target_date
            and pairs[source].get("history_view_sha256")
            == history_view_sha256(
                windows[source + 1],
                fit_user_ids,
            )
            and pairs[source].get("load_validation", {}).get("passed")
            is True
            and pairs[source]
            .get("load_validation", {})
            .get("provenance", {})
            .get("labels_used")
            is False
            and pairs[source]
            .get("load_validation", {})
            .get("provenance", {})
            .get("future_history_used")
            is False
            and pairs[source]
            .get("load_validation", {})
            .get("provenance", {})
            .get("history_view_sha256")
            == pairs[source].get("history_view_sha256")
            and pairs[source]
            .get("load_validation", {})
            .get("provenance", {})
            .get("history_version")
            == f"theta{source + 1}"
            and (
                checkpoint_hashes is None
                or (
                    pairs[source]
                    .get("load_validation", {})
                    .get("provenance", {})
                    .get("source_checkpoint_sha256")
                    == checkpoint_hashes[f"theta{source}"]
                    and pairs[source]
                    .get("load_validation", {})
                    .get("provenance", {})
                    .get("target_checkpoint_sha256")
                    == checkpoint_hashes[f"theta{source + 1}"]
                )
            )
            for source in range(NUM_EDGES)
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"organic compiler payload differs: {checks}")
    return checks


def identity_jagged_slice(
    cache: JaggedMigratedKVBatch,
) -> JaggedTokenSlice:
    lengths = tuple(int(value) for value in cache.lengths.tolist())
    return JaggedTokenSlice(
        record_ids=cache.record_ids,
        migration_anchor_version=cache.migration_anchor_version,
        served_kv_target=cache.served_kv_target,
        starts=(0,) * cache.batch_size,
        stops=lengths,
        retained_rows=tuple(range(cache.batch_size)),
        empty_rows=(),
        num_layers=cache.k.shape[0],
        kv_width=cache.k.shape[2],
        dtype=cache.k.dtype,
        device=cache.k.device,
        cache=cache,
    )


def empty_jagged_slice(
    record_ids: tuple[int, ...],
    version: int,
    num_layers: int,
    kv_width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> JaggedTokenSlice:
    if not record_ids:
        raise ValueError("organic empty prefix batch is empty")
    return JaggedTokenSlice(
        record_ids=record_ids,
        migration_anchor_version=f"theta{version}",
        served_kv_target=f"theta{version}",
        starts=(0,) * len(record_ids),
        stops=(0,) * len(record_ids),
        retained_rows=(),
        empty_rows=tuple(range(len(record_ids))),
        num_layers=num_layers,
        kv_width=kv_width,
        dtype=dtype,
        device=device,
        cache=None,
    )


def _record_cache(
    cache: JaggedMigratedKVBatch,
    row: int,
) -> JaggedMigratedKVBatch:
    return select_jagged_rows(cache, (row,))


def _split_cache(
    cache: JaggedMigratedKVBatch,
) -> dict[int, JaggedMigratedKVBatch]:
    return {
        record_id: _record_cache(cache, row)
        for row, record_id in enumerate(cache.record_ids)
    }


def _assemble_record_caches(
    record_ids: tuple[int, ...],
    cache_by_record: dict[int, JaggedMigratedKVBatch],
) -> JaggedMigratedKVBatch:
    if not record_ids:
        raise ValueError("organic cache assembly is empty")
    selected = [cache_by_record[record_id] for record_id in record_ids]
    first = selected[0]
    if any(
        cache.batch_size != 1
        or cache.record_ids != (record_id,)
        or cache.migration_anchor_version
        != first.migration_anchor_version
        or cache.served_kv_target != first.served_kv_target
        or cache.k.shape[0] != first.k.shape[0]
        or cache.k.shape[2] != first.k.shape[2]
        or cache.k.dtype != first.k.dtype
        or cache.k.device != first.k.device
        for record_id, cache in zip(record_ids, selected, strict=True)
    ):
        raise ValueError("organic record caches cannot be assembled")
    lengths = torch.tensor(
        [int(cache.lengths[0]) for cache in selected],
        dtype=torch.long,
        device=first.k.device,
    )
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=first.k.device),
            lengths.cumsum(0),
        )
    )
    return JaggedMigratedKVBatch(
        record_ids=record_ids,
        migration_anchor_version=first.migration_anchor_version,
        served_kv_target=first.served_kv_target,
        k=torch.cat([cache.k for cache in selected], dim=1).contiguous(),
        v=torch.cat([cache.v for cache in selected], dim=1).contiguous(),
        lengths=lengths,
        offsets=offsets,
    )


def _consume_record_caches(
    record_ids: tuple[int, ...],
    cache_by_record: dict[int, JaggedMigratedKVBatch],
) -> JaggedMigratedKVBatch:
    cache = _assemble_record_caches(record_ids, cache_by_record)
    for record_id in record_ids:
        cache_by_record.pop(record_id)
    return cache


def resident_cache_bytes(
    cache_by_record: dict[int, JaggedMigratedKVBatch],
) -> int:
    return sum(
        cache.k.numel() * cache.k.element_size()
        + cache.v.numel() * cache.v.element_size()
        for cache in cache_by_record.values()
    )


def _history_sequence(record, prefix: bool) -> dict:
    history = record.history
    if history is None:
        raise ValueError("organic history is absent")
    stop = len(history) - 1 if prefix else len(history)
    if stop < 1:
        raise ValueError("organic sequence is empty")
    return {
        "item_ids": history.item_ids[:stop],
        "behaviors": history.behaviors[:stop],
        "time_deltas": history.time_deltas[:stop],
    }


def _history_batch(
    records: list,
    max_seq_len: int,
    device: torch.device,
    prefix: bool,
) -> dict:
    return move_batch(
        collate_batch(
            [_history_sequence(record, prefix) for record in records],
            max_seq_len=max_seq_len - (1 if prefix else 0),
        ),
        device,
    )


def _suffix_batch(
    records: list,
    device: torch.device,
) -> dict:
    return {
        "item_ids": torch.tensor(
            [[int(record.history.item_ids[-1])] for record in records],
            dtype=torch.long,
            device=device,
        ),
        "behaviors": torch.tensor(
            [[int(record.history.behaviors[-1])] for record in records],
            dtype=torch.long,
            device=device,
        ),
        "time_deltas": torch.tensor(
            [[float(record.history.time_deltas[-1])] for record in records],
            dtype=torch.float32,
            device=device,
        ),
        "lengths": torch.ones(
            len(records),
            dtype=torch.long,
            device=device,
        ),
    }


def _appended_batch(
    records: list,
    transitions: list[TransitionDescriptor],
    device: torch.device,
) -> dict:
    lengths = [
        transition.new_length - transition.appended_new_start
        for transition in transitions
    ]
    width = max(lengths, default=0)
    item_ids = torch.zeros(
        (len(records), width),
        dtype=torch.long,
        device=device,
    )
    behaviors = torch.zeros_like(item_ids)
    time_deltas = torch.zeros(
        (len(records), width),
        dtype=torch.float32,
        device=device,
    )
    for row, (record, transition, length) in enumerate(
        zip(records, transitions, lengths, strict=True)
    ):
        start = transition.appended_new_start
        stop = transition.new_length
        if length:
            item_ids[row, :length] = torch.tensor(
                record.history.item_ids[start:stop],
                dtype=torch.long,
                device=device,
            )
            behaviors[row, :length] = torch.tensor(
                record.history.behaviors[start:stop],
                dtype=torch.long,
                device=device,
            )
            time_deltas[row, :length] = torch.tensor(
                record.history.time_deltas[start:stop],
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


@torch.inference_mode()
def _exact_full_batch(
    model,
    batch: dict,
    record_ids: tuple[int, ...],
    version: int,
) -> tuple[JaggedMigratedKVBatch, torch.Tensor]:
    hidden, cache = model(
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        return_kv=True,
        lengths=batch["lengths"],
    )
    if cache is None:
        raise RuntimeError("organic full replay did not return K/V")
    return (
        pack_padded_cache(
            cache,
            batch["lengths"],
            record_ids,
            f"theta{version}",
            f"theta{version}",
        ),
        model.last_hidden(hidden, batch["lengths"]),
    )


def _relabel_exact(
    cache: JaggedMigratedKVBatch,
    version: int,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=cache.record_ids,
        migration_anchor_version=f"theta{version}",
        served_kv_target=f"theta{version}",
        k=cache.k,
        v=cache.v,
        lengths=cache.lengths,
        offsets=cache.offsets,
    )


def _record_metrics(
    actual: JaggedMigratedKVBatch,
    exact: JaggedMigratedKVBatch,
    mixed_hidden: torch.Tensor,
    mixed_scores: torch.Tensor,
    exact_hidden: torch.Tensor,
    exact_scores: torch.Tensor,
) -> list[dict[str, float]]:
    cache_error = aggregate_layer_values(
        relative_cache_values(actual, exact),
        0.9,
    ).cpu().tolist()
    semantic = semantic_pair(
        mixed_hidden,
        mixed_scores,
        exact_hidden,
        exact_scores,
    )
    return [
        {
            "cache_error_q090": float(cache_error[row]),
            "cache_fidelity_q090": max(
                0.0,
                1.0 - float(cache_error[row]),
            ),
            "hidden_cosine": float(semantic["hidden_cosine"][row]),
            "score_cosine": float(semantic["score_cosine"][row]),
            "top100_overlap": float(semantic["top100_overlap"][row]),
        }
        for row in range(actual.batch_size)
    ]


def summarize_organic_tasks(records: list[dict]) -> dict | None:
    selected = [
        value
        for value in records
        if value["evaluation_role"] == "final_test"
        and "task" in value
    ]
    if not selected:
        return None
    reuse_selected = [
        value for value in selected if "reuse" in value["task"]
    ]
    metrics = ("mean_rank", "catalog_auc", "ndcg@100", "hit@100")
    output = {
        "records": len(selected),
        "reuse_coverage_records": len(reuse_selected),
        "reuse_coverage_fraction": len(reuse_selected) / len(selected),
        "mixed": {},
        "all_exact": {},
        "paired_difference_mixed_minus_exact": {},
        "reuse_continued_subset": (
            {
                "reuse": {},
                "mixed": {},
                "all_exact": {},
                "recovery_from_reuse_to_exact": {},
            }
            if reuse_selected
            else None
        ),
    }
    for metric in metrics:
        mixed = np.asarray(
            [value["task"]["mixed"][metric] for value in selected]
        )
        exact = np.asarray(
            [value["task"]["all_exact"][metric] for value in selected]
        )
        output["mixed"][metric] = float(mixed.mean())
        output["all_exact"][metric] = float(exact.mean())
        output["paired_difference_mixed_minus_exact"][metric] = float(
            (mixed - exact).mean()
        )
        if reuse_selected:
            subset = output["reuse_continued_subset"]
            reuse = np.asarray(
                [value["task"]["reuse"][metric] for value in reuse_selected]
            )
            subset_mixed = np.asarray(
                [
                    value["task"]["mixed"][metric]
                    for value in reuse_selected
                ]
            )
            subset_exact = np.asarray(
                [
                    value["task"]["all_exact"][metric]
                    for value in reuse_selected
                ]
            )
            subset["reuse"][metric] = float(reuse.mean())
            subset["mixed"][metric] = float(subset_mixed.mean())
            subset["all_exact"][metric] = float(subset_exact.mean())
            denominator = float(
                (reuse - subset_exact).mean()
                if metric == "mean_rank"
                else (subset_exact - reuse).mean()
            )
            numerator = float(
                (reuse - subset_mixed).mean()
                if metric == "mean_rank"
                else (subset_mixed - reuse).mean()
            )
            subset["recovery_from_reuse_to_exact"][metric] = (
                numerator / denominator
                if abs(denominator) >= 1e-6
                else None
            )
    return output


def _summarize_endpoint(
    version: int,
    target_date: str,
    records: list[dict],
    resident_records: int,
    short_records: int,
) -> dict:
    cached = [value for value in records if value.get("metrics") is not None]
    metrics = (
        "cache_error_q090",
        "cache_fidelity_q090",
        "hidden_cosine",
        "score_cosine",
        "top100_overlap",
    )
    return {
        "version": version,
        "target_date": target_date,
        "resident_records": resident_records,
        "cache_eligible_records": len(cached),
        "short_history_records": short_records,
        "prediction_records": len(records),
        "label_free_metrics": {
            metric: (
                float(np.mean([value["metrics"][metric] for value in cached]))
                if cached
                else None
            )
            for metric in metrics
        },
        "task_metrics": summarize_organic_tasks(records),
    }


def _task_value(
    role: str,
    positives: tuple[int, ...],
    mixed_scores: torch.Tensor,
    exact_scores: torch.Tensor,
    reuse_scores: torch.Tensor | None,
) -> dict | None:
    if role != "final_test" or not positives:
        return None
    output = {
        "mixed": task_metrics(mixed_scores, list(positives)),
        "all_exact": task_metrics(exact_scores, list(positives)),
    }
    if reuse_scores is not None:
        output["reuse"] = task_metrics(reuse_scores, list(positives))
    return output


@torch.inference_mode()
def _initialize_theta0(
    cfg,
    checkpoint_dir: str,
    window,
    groups,
    record_by_user: dict[int, dict],
    device: torch.device,
    all_items: torch.Tensor,
) -> tuple[
    dict[int, JaggedMigratedKVBatch],
    dict[int, CacheLifecycleState],
    list[dict],
    dict,
]:
    model = load_checkpoint_model(cfg, checkpoint_dir, 0, device)
    record_by_id = {
        int(value["record_id"]): value
        for group in groups
        for value in group
    }
    cache_by_record = {}
    states = {}
    endpoint_records = []
    initialization_ms = 0.0
    for group in groups:
        selected = [
            record_by_user[int(value["user_id"])]
            for value in group
            if window.records[int(value["user_id"])].history is not None
        ]
        if not selected:
            continue
        window_records = [
            window.records[int(value["user_id"])] for value in selected
        ]
        ids = tuple(int(value["record_id"]) for value in selected)
        batch = _history_batch(
            window_records,
            cfg.max_seq_len,
            device,
            prefix=False,
        )
        (full, hidden), elapsed = timed_cuda(
            partial(_exact_full_batch, model, batch, ids, 0),
            device,
        )
        initialization_ms += elapsed
        cache_by_record.update(_split_cache(full))
        scores = model.item_emb.score(
            hidden,
            all_items.unsqueeze(0).expand(len(ids), -1),
        )
        for row, (descriptor, record_id) in enumerate(
            zip(selected, ids, strict=True)
        ):
            states[record_id] = CacheLifecycleState.exact(record_id, 0)
            record = window.records[int(descriptor["user_id"])]
            value = {
                "record_id": record_id,
                "user_id": int(descriptor["user_id"]),
                "evaluation_role": descriptor["evaluation_role"],
                "metrics": (
                    {
                        "cache_error_q090": 0.0,
                        "cache_fidelity_q090": 1.0,
                        "hidden_cosine": 1.0,
                        "score_cosine": 1.0,
                        "top100_overlap": 1.0,
                    }
                    if len(record.history) >= 2
                    else None
                ),
            }
            task = _task_value(
                descriptor["evaluation_role"],
                record.engaged_positive_item_ids,
                scores[row],
                scores[row],
                None,
            )
            if task is not None:
                value["task"] = task
            endpoint_records.append(value)
    endpoint = _summarize_endpoint(
        0,
        window.target_date,
        endpoint_records,
        len(cache_by_record),
        sum(
            record.history is not None and len(record.history) == 1
            for record in window.records.values()
        ),
    )
    endpoint["anchor"] = {
        "kind": "theta0_exact_full_history",
        "initialization_gpu_ms": initialization_ms,
        "update_ratio": None,
        "layout_matches_history": all(
            int(cache.lengths[0])
            == len(
                window.records[
                    int(record_by_id[record_id]["user_id"])
                ].history
            )
            for record_id, cache in cache_by_record.items()
        ),
    }
    return cache_by_record, states, endpoint_records, endpoint


def _prepare_source_prefix(
    source_model,
    old_window,
    target_window,
    groups,
    previous_cache: dict[int, JaggedMigratedKVBatch],
    previous_states: dict[int, CacheLifecycleState],
    device: torch.device,
    costs: dict[str, float],
) -> tuple[
    dict[int, JaggedMigratedKVBatch],
    dict[int, CacheLifecycleState],
    set[int],
    dict[int, TransitionDescriptor],
    int,
]:
    source_cache = {}
    source_states = {}
    direct_exact_ids = set()
    transitions = {}
    persistent_peak_bytes = resident_cache_bytes(previous_cache)
    for group in groups:
        continued_descriptors = []
        continued_records = []
        continued_transitions = []
        for descriptor in group:
            record_id = int(descriptor["record_id"])
            user_id = int(descriptor["user_id"])
            transition = transition_descriptor(
                old_window.records[user_id],
                target_window.records[user_id],
                record_id in previous_cache,
            )
            transitions[record_id] = transition
            if target_window.records[user_id].history is None:
                continue
            if transition.status == "continued":
                continued_descriptors.append(descriptor)
                continued_records.append(target_window.records[user_id])
                continued_transitions.append(transition)
                source_states[record_id] = previous_states[record_id]
            else:
                direct_exact_ids.add(record_id)
        if continued_descriptors:
            continued_ids = tuple(
                int(value["record_id"]) for value in continued_descriptors
            )
            old_batch, assembly_ms = timed_cuda(
                partial(
                    _assemble_record_caches,
                    continued_ids,
                    previous_cache,
                ),
                device,
            )
        for descriptor in group:
            previous_cache.pop(int(descriptor["record_id"]), None)
        if continued_descriptors:
            sliced, evict_ms = timed_cuda(
                partial(
                    tail_slice_jagged_cache,
                    old_batch,
                    tuple(
                        value.overlap for value in continued_transitions
                    ),
                ),
                device,
            )
            costs["foreground_evict"] += assembly_ms + evict_ms
            if sliced.cache is None:
                raise RuntimeError("organic continued prefix crop is empty")
            append_rows = tuple(
                row
                for row, transition in enumerate(continued_transitions)
                if transition.appended > 0
            )
            retain_rows = tuple(
                row
                for row, transition in enumerate(continued_transitions)
                if transition.appended == 0
            )
            if retain_rows:
                if len(retain_rows) == len(continued_records):
                    retained = sliced.cache
                    selection_ms = 0.0
                else:
                    retained, selection_ms = timed_cuda(
                        partial(
                            select_jagged_rows,
                            sliced.cache,
                            retain_rows,
                        ),
                        device,
                    )
                retained_split, split_ms = timed_cuda(
                    partial(_split_cache, retained),
                    device,
                )
                costs["foreground_evict"] += selection_ms + split_ms
                source_cache.update(retained_split)
                del retained, retained_split
            if append_rows:
                if len(append_rows) == len(continued_records):
                    append_cache = sliced.cache
                    selection_ms = 0.0
                else:
                    append_cache, selection_ms = timed_cuda(
                        partial(
                            select_jagged_rows,
                            sliced.cache,
                            append_rows,
                        ),
                        device,
                    )
                appended = _appended_batch(
                    [continued_records[row] for row in append_rows],
                    [continued_transitions[row] for row in append_rows],
                    device,
                )
                result, append_ms = timed_cuda(
                    partial(
                        append_jagged_suffix,
                        source_model,
                        identity_jagged_slice(append_cache),
                        appended["item_ids"],
                        appended["behaviors"],
                        appended["time_deltas"],
                        appended["lengths"],
                    ),
                    device,
                )
                appended_split, split_ms = timed_cuda(
                    partial(_split_cache, result.cache),
                    device,
                )
                costs["foreground_incremental_append"] += (
                    selection_ms + append_ms + split_ms
                )
                source_cache.update(appended_split)
                del append_cache, appended, result, appended_split
            del old_batch, sliced
        persistent_peak_bytes = max(
            persistent_peak_bytes,
            resident_cache_bytes(previous_cache)
            + resident_cache_bytes(source_cache),
        )
    residents = {
        int(descriptor["record_id"])
        for group in groups
        for descriptor in group
        if target_window.records[int(descriptor["user_id"])].history
        is not None
    }
    if (
        previous_cache
        or set(source_cache) != set(source_states)
        or set(source_cache) & direct_exact_ids
        or (set(source_cache) | direct_exact_ids) != residents
    ):
        raise RuntimeError("organic foreground does not cover residents")
    return (
        source_cache,
        source_states,
        direct_exact_ids,
        transitions,
        persistent_peak_bytes,
    )


def _load_program(
    args,
    cfg,
    compiler: dict,
    source_version: int,
    device: torch.device,
    operator,
):
    pair = compiler["pairs"][source_version]
    descriptor = pair["direct_program"]
    path = direct_program_path(
        args.runtime_dir,
        source_version,
        source_version + 1,
    )
    program_cpu, loaded = load_direct_oldkv_program(
        path,
        expected_sha256=descriptor["sha256"],
        expected_source_version=f"theta{source_version}",
        expected_target_version=f"theta{source_version + 1}",
        expected_num_layers=cfg.num_layers,
        expected_kv_width=cfg.num_heads * cfg.head_dim,
    )
    if (
        loaded.get("provenance")
        != pair["load_validation"]["provenance"]
        or loaded["provenance"].get("history_view_sha256")
        != pair["history_view_sha256"]
        or loaded["provenance"].get("labels_used") is not False
        or loaded["provenance"].get("future_history_used") is not False
    ):
        raise ValueError("organic loaded program provenance differs")
    return operator.prepare_program(program_cpu, device), loaded, program_cpu


def _probe_candidates(
    source_prefixes: dict[int, JaggedMigratedKVBatch],
    target_window,
    groups,
    record_by_id,
    target_model,
    all_items: torch.Tensor,
    operator,
    program,
    target_version: int,
    device: torch.device,
    costs: dict[str, float],
) -> tuple[
    dict[int, JaggedMigratedKVBatch],
    dict[int, float],
    dict[int, dict[str, float]],
    int,
]:
    candidate_by_record = {}
    norm_shifts = {}
    reuse_task_by_record = {}
    persistent_peak_bytes = resident_cache_bytes(source_prefixes)
    for group in groups:
        eligible = [
            record_by_id[int(value["record_id"])]
            for value in group
            if int(value["record_id"]) in source_prefixes
        ]
        resident_ids = tuple(
            int(value["record_id"])
            for value in group
            if int(value["record_id"]) in source_prefixes
        )
        if not resident_ids:
            continue
        if eligible:
            ids = tuple(int(value["record_id"]) for value in eligible)
            source_prefix, assembly_ms = timed_cuda(
                partial(_assemble_record_caches, ids, source_prefixes),
                device,
            )
        for record_id in resident_ids:
            source_prefixes.pop(record_id)
        if not eligible:
            continue
        candidate, transform_ms = timed_cuda(
            partial(
                execute_direct,
                operator,
                program,
                source_prefix,
                target_version,
            ),
            device,
        )
        candidate_records, candidate_split_ms = timed_cuda(
            partial(_split_cache, candidate),
            device,
        )
        costs["candidate_transform"] += (
            assembly_ms + transform_ms + candidate_split_ms
        )
        shifts, probe_ms = timed_cuda(
            partial(
                _candidate_norm_shifts,
                source_prefix,
                candidate,
            ),
            device,
        )
        costs["router_probe"] += probe_ms
        shift_values = shifts.cpu().tolist()
        candidate_by_record.update(candidate_records)
        norm_shifts.update(
            {
                record_id: float(shift_values[row])
                for row, record_id in enumerate(ids)
            }
        )
        records = [
            target_window.records[int(value["user_id"])]
            for value in eligible
        ]
        suffix = _suffix_batch(records, device)
        _, reuse_scores = cache_hidden_scores(
            target_model,
            source_prefix,
            suffix,
            all_items,
        )
        for row, (descriptor, record_id) in enumerate(
            zip(eligible, ids, strict=True)
        ):
            record = records[row]
            if (
                descriptor["evaluation_role"] == "final_test"
                and record.engaged_positive_item_ids
            ):
                reuse_task_by_record[record_id] = task_metrics(
                    reuse_scores[row],
                    list(record.engaged_positive_item_ids),
                )
        del source_prefix, candidate, candidate_records
        del shifts, suffix, reuse_scores
        persistent_peak_bytes = max(
            persistent_peak_bytes,
            resident_cache_bytes(source_prefixes)
            + resident_cache_bytes(candidate_by_record),
        )
    if source_prefixes or set(candidate_by_record) != set(norm_shifts):
        raise RuntimeError("organic candidate probe coverage differs")
    return (
        candidate_by_record,
        norm_shifts,
        reuse_task_by_record,
        persistent_peak_bytes,
    )


def _candidate_norm_shifts(
    source: JaggedMigratedKVBatch,
    candidate: JaggedMigratedKVBatch,
) -> torch.Tensor:
    return aggregate_layer_values(
        absolute_log_norm_ratio_values(source, candidate),
        0.9,
    )


def _publication_partition(
    group,
    target_window,
    record_by_id,
    source_states: dict[int, CacheLifecycleState],
    direct_exact_ids: set[int],
    planned_by_record: dict[int, LifecycleDecision],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    descriptors = [
        record_by_id[int(value["record_id"])] for value in group
    ]
    eligible = [
        value
        for value in descriptors
        if int(value["record_id"]) in planned_by_record
    ]
    direct_prefix = [
        value
        for value in descriptors
        if int(value["record_id"]) in direct_exact_ids
        and target_window.records[int(value["user_id"])].history is not None
        and len(target_window.records[int(value["user_id"])].history) >= 2
    ]
    short = [
        value
        for value in descriptors
        if int(value["record_id"]) in direct_exact_ids
        and target_window.records[int(value["user_id"])].history is not None
        and len(target_window.records[int(value["user_id"])].history) == 1
    ]
    expected = {
        int(value["record_id"])
        for value in descriptors
        if int(value["record_id"]) in source_states
        or int(value["record_id"]) in direct_exact_ids
    }
    actual = {
        int(value["record_id"])
        for value in (*eligible, *direct_prefix, *short)
    }
    if expected != actual:
        raise RuntimeError("organic publication partition differs")
    return descriptors, eligible, direct_prefix, short


def _exact_reference_descriptors(
    descriptors,
    target_window,
) -> list[dict]:
    return [
        value
        for value in descriptors
        if target_window.records[int(value["user_id"])].history is not None
        and len(target_window.records[int(value["user_id"])].history) >= 2
    ]


@torch.inference_mode()
def _run_edge(
    args,
    cfg,
    compiler,
    old_window,
    target_window,
    groups,
    record_by_id,
    cache_by_record,
    states,
    policy,
    operator,
    all_items,
) -> tuple[
    dict[int, JaggedMigratedKVBatch],
    dict[int, CacheLifecycleState],
    list[dict],
    dict,
    dict,
]:
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    source_version = int(old_window.version)
    target_version = int(target_window.version)
    source_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        source_version,
        device,
    )
    program, program_descriptor, program_cpu = _load_program(
        args,
        cfg,
        compiler,
        source_version,
        device,
        operator,
    )
    costs = {
        "foreground_evict": 0.0,
        "foreground_incremental_append": 0.0,
        "candidate_transform": 0.0,
        "router_probe": 0.0,
        "exact_refresh": 0.0,
        "publication": 0.0,
        "common_latest": 0.0,
        "common_publication": 0.0,
        "natural_direct_exact": 0.0,
    }
    previous_resident_records = len(cache_by_record)
    previous_resident_bytes = resident_cache_bytes(cache_by_record)
    (
        source_prefixes,
        source_states,
        direct_exact_ids,
        transitions,
        foreground_persistent_peak_bytes,
    ) = _prepare_source_prefix(
        source_model,
        old_window,
        target_window,
        groups,
        cache_by_record,
        states,
        device,
        costs,
    )
    previous_cache_consumed = not cache_by_record
    source_resident_records = len(source_prefixes)
    source_resident_bytes = resident_cache_bytes(source_prefixes)
    source_prefix_lengths_match = all(
        int(cache.lengths[0])
        == len(
            target_window.records[
                int(record_by_id[record_id]["user_id"])
            ].history
        )
        - 1
        for record_id, cache in source_prefixes.items()
    )
    del source_model
    gc.collect()
    torch.cuda.empty_cache()
    target_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        target_version,
        device,
    )
    (
        candidate_by_record,
        norm_shifts,
        reuse_task_by_record,
        probe_persistent_peak_bytes,
    ) = _probe_candidates(
        source_prefixes,
        target_window,
        groups,
        record_by_id,
        target_model,
        all_items,
        operator,
        program,
        target_version,
        device,
        costs,
    )
    source_cache_consumed = not source_prefixes
    candidate_resident_bytes = resident_cache_bytes(candidate_by_record)
    del program, program_cpu
    gc.collect()
    torch.cuda.empty_cache()
    eligible_ids = tuple(sorted(candidate_by_record))
    selector_states = tuple(
        source_states[record_id] for record_id in eligible_ids
    )
    planned = (
        select_norm_shift_decisions(
            selector_states,
            target_version,
            norm_shifts,
        )
        if eligible_ids
        else ()
    )
    selector_evidence = selector_audit(
        selector_states,
        planned,
        norm_shifts,
    )
    planned_by_record = {
        value.record_id: value for value in planned
    }
    next_cache = {}
    next_states = {}
    endpoint_records = []
    lineage_by_record = {}
    candidate_errors = {}
    exact_reference_ms = 0.0
    exact_reference_groups = 0
    exact_reference_records = 0
    publication_persistent_peak_bytes = candidate_resident_bytes
    for group in groups:
        descriptors, eligible, direct_prefix, short = _publication_partition(
            group,
            target_window,
            record_by_id,
            source_states,
            direct_exact_ids,
            planned_by_record,
        )
        reference_descriptors = _exact_reference_descriptors(
            descriptors,
            target_window,
        )
        reference_ids = tuple(
            int(value["record_id"]) for value in reference_descriptors
        )
        reference_row_by_id = {
            record_id: row
            for row, record_id in enumerate(reference_ids)
        }
        if set(reference_ids) != {
            int(value["record_id"])
            for value in (*eligible, *direct_prefix)
        }:
            raise RuntimeError("organic exact reference partition differs")
        if reference_descriptors:
            reference_records = [
                target_window.records[int(value["user_id"])]
                for value in reference_descriptors
            ]
            reference_batch = _history_batch(
                reference_records,
                cfg.max_seq_len,
                device,
                prefix=True,
            )
            exact_reference_group, reference_ms = timed_cuda(
                partial(
                    exact_batch,
                    target_model,
                    reference_batch,
                    reference_ids,
                    target_version,
                ),
                device,
            )
            exact_reference_ms += reference_ms
            exact_reference_groups += 1
            exact_reference_records += len(reference_ids)
        if eligible:
            ids = tuple(int(value["record_id"]) for value in eligible)
            records = [
                target_window.records[int(value["user_id"])]
                for value in eligible
            ]
            candidate, candidate_assembly_ms = timed_cuda(
                partial(
                    _consume_record_caches,
                    ids,
                    candidate_by_record,
                ),
                device,
            )
            costs["publication"] += candidate_assembly_ms
            migrate_rows = tuple(
                row
                for row, record_id in enumerate(ids)
                if planned_by_record[record_id].action == "migrate"
            )
            exact_rows = tuple(
                row
                for row, record_id in enumerate(ids)
                if planned_by_record[record_id].action == "exact"
            )
            migrated_selected = None
            if migrate_rows:
                migrated_selected, accepted_selection_ms = timed_cuda(
                    partial(
                        select_jagged_rows,
                        candidate,
                        migrate_rows,
                    ),
                    device,
                )
                costs["publication"] += accepted_selection_ms
            exact_selected = None
            if exact_rows:
                exact_records = [records[row] for row in exact_rows]
                exact_ids = tuple(ids[row] for row in exact_rows)
                exact_prefix_batch = _history_batch(
                    exact_records,
                    cfg.max_seq_len,
                    device,
                    prefix=True,
                )
                exact_selected, exact_ms = timed_cuda(
                    partial(
                        exact_batch,
                        target_model,
                        exact_prefix_batch,
                        exact_ids,
                        target_version,
                    ),
                    device,
                )
                costs["exact_refresh"] += exact_ms
            sources = tuple(
                value
                for value in (migrated_selected, exact_selected)
                if value is not None
            )
            target_prefix, publication_ms = timed_cuda(
                partial(
                    assemble_jagged_rows,
                    candidate,
                    sources,
                    target_version,
                ),
                device,
            )
            costs["publication"] += publication_ms
            exact_reference = select_jagged_rows(
                exact_reference_group,
                tuple(reference_row_by_id[record_id] for record_id in ids),
            )
            candidate_error_values = aggregate_layer_values(
                relative_cache_values(
                    candidate,
                    exact_reference,
                ),
                0.9,
            ).cpu().tolist()
            candidate_errors.update(
                {
                    record_id: float(candidate_error_values[row])
                    for row, record_id in enumerate(ids)
                }
            )
            suffix = _suffix_batch(records, device)
            common, common_ms = timed_cuda(
                partial(
                    append_jagged_suffix,
                    target_model,
                    identity_jagged_slice(target_prefix),
                    suffix["item_ids"],
                    suffix["behaviors"],
                    suffix["time_deltas"],
                    suffix["lengths"],
                ),
                device,
            )
            costs["common_latest"] += common_ms
            published, split_ms = timed_cuda(
                partial(_split_cache, common.cache),
                device,
            )
            costs["common_publication"] += split_ms
            next_cache.update(published)
            mixed_hidden = common.last_appended_hidden
            if mixed_hidden is None:
                raise RuntimeError("organic common latest has no hidden state")
            mixed_scores = target_model.item_emb.score(
                mixed_hidden,
                all_items.unsqueeze(0).expand(len(ids), -1),
            )
            exact_hidden, exact_scores = cache_hidden_scores(
                target_model,
                exact_reference,
                suffix,
                all_items,
            )
            metrics = _record_metrics(
                target_prefix,
                exact_reference,
                mixed_hidden,
                mixed_scores,
                exact_hidden,
                exact_scores,
            )
            for row, (descriptor, record_id) in enumerate(
                zip(eligible, ids, strict=True)
            ):
                decision = planned_by_record[record_id]
                state_before = source_states[record_id]
                state_after = policy.advance(state_before, decision)
                next_states[record_id] = state_after
                record = target_window.records[int(descriptor["user_id"])]
                value = {
                    "record_id": record_id,
                    "user_id": int(descriptor["user_id"]),
                    "evaluation_role": descriptor["evaluation_role"],
                    "metrics": metrics[row],
                }
                task = None
                if (
                    descriptor["evaluation_role"] == "final_test"
                    and record.engaged_positive_item_ids
                ):
                    task = {
                        "mixed": task_metrics(
                            mixed_scores[row],
                            list(record.engaged_positive_item_ids),
                        ),
                        "all_exact": task_metrics(
                            exact_scores[row],
                            list(record.engaged_positive_item_ids),
                        ),
                        "reuse": reuse_task_by_record[record_id],
                    }
                if task is not None:
                    value["task"] = task
                endpoint_records.append(value)
                transition = transitions[record_id]
                lineage_by_record[record_id] = {
                    "record_id": record_id,
                    "user_id": int(descriptor["user_id"]),
                    "old_history_hash": transition.old_history_hash,
                    "new_history_hash": transition.new_history_hash,
                    "overlap_tokens": transition.overlap,
                    "evicted_tokens": transition.evicted,
                    "appended_tokens": transition.appended,
                    "source_prefix_tokens": transition.new_length,
                    "common_latest_tokens": 1,
                    "foreground_status": transition.status,
                    "previous_actual_consumed": (
                        transition.previous_actual_consumed
                    ),
                    "action": decision.action,
                    "decision": decision.to_dict(),
                    "state_before": state_before.to_dict(),
                    "state_after": state_after.to_dict(),
                    "program_sha256": program_descriptor["sha256"],
                    "norm_shift_q090": norm_shifts[record_id],
                    "candidate_evaluated": True,
                }
            del candidate, target_prefix, exact_reference, common
            del migrated_selected, exact_selected, published
            del suffix, mixed_hidden, mixed_scores
            del exact_hidden, exact_scores, candidate_error_values
            del sources
            if exact_rows:
                del exact_prefix_batch
        if direct_prefix:
            ids = tuple(
                int(value["record_id"]) for value in direct_prefix
            )
            records = [
                target_window.records[int(value["user_id"])]
                for value in direct_prefix
            ]
            prefix_batch = _history_batch(
                records,
                cfg.max_seq_len,
                device,
                prefix=True,
            )
            exact_prefix, exact_ms = timed_cuda(
                partial(
                    exact_batch,
                    target_model,
                    prefix_batch,
                    ids,
                    target_version,
                ),
                device,
            )
            costs["exact_refresh"] += exact_ms
            costs["natural_direct_exact"] += exact_ms
            suffix = _suffix_batch(records, device)
            common, common_ms = timed_cuda(
                partial(
                    append_jagged_suffix,
                    target_model,
                    identity_jagged_slice(exact_prefix),
                    suffix["item_ids"],
                    suffix["behaviors"],
                    suffix["time_deltas"],
                    suffix["lengths"],
                ),
                device,
            )
            costs["common_latest"] += common_ms
            published, split_ms = timed_cuda(
                partial(_split_cache, common.cache),
                device,
            )
            costs["common_publication"] += split_ms
            next_cache.update(published)
            hidden = common.last_appended_hidden
            if hidden is None:
                raise RuntimeError("organic direct exact has no hidden state")
            scores = target_model.item_emb.score(
                hidden,
                all_items.unsqueeze(0).expand(len(ids), -1),
            )
            for row, (descriptor, record_id) in enumerate(
                zip(direct_prefix, ids, strict=True)
            ):
                state_before = states.get(record_id)
                state_after = CacheLifecycleState.exact(
                    record_id,
                    target_version,
                )
                next_states[record_id] = state_after
                record = records[row]
                value = {
                    "record_id": record_id,
                    "user_id": int(descriptor["user_id"]),
                    "evaluation_role": descriptor["evaluation_role"],
                    "metrics": {
                        "cache_error_q090": 0.0,
                        "cache_fidelity_q090": 1.0,
                        "hidden_cosine": 1.0,
                        "score_cosine": 1.0,
                        "top100_overlap": 1.0,
                    },
                }
                task = _task_value(
                    descriptor["evaluation_role"],
                    record.engaged_positive_item_ids,
                    scores[row],
                    scores[row],
                    None,
                )
                if task is not None:
                    value["task"] = task
                endpoint_records.append(value)
                transition = transitions[record_id]
                lineage_by_record[record_id] = {
                    "record_id": record_id,
                    "user_id": int(descriptor["user_id"]),
                    "old_history_hash": transition.old_history_hash,
                    "new_history_hash": transition.new_history_hash,
                    "overlap_tokens": transition.overlap,
                    "evicted_tokens": transition.evicted,
                    "appended_tokens": transition.appended,
                    "source_prefix_tokens": transition.new_length,
                    "common_latest_tokens": 1,
                    "foreground_status": transition.status,
                    "previous_actual_consumed": False,
                    "action": "exact",
                    "decision": {
                        "action": "exact",
                        "reason": "no_reusable_prefix_target_exact",
                        "shared_with_exact_reference": False,
                        "reference_execution": "separate_fixed_group",
                    },
                    "state_before": (
                        state_before.to_dict()
                        if state_before is not None
                        else None
                    ),
                    "state_after": state_after.to_dict(),
                    "program_sha256": program_descriptor["sha256"],
                    "norm_shift_q090": None,
                    "candidate_evaluated": False,
                }
            del exact_prefix, common, published, hidden, scores
            del prefix_batch, suffix
        if short:
            ids = tuple(int(value["record_id"]) for value in short)
            records = [
                target_window.records[int(value["user_id"])]
                for value in short
            ]
            suffix = _suffix_batch(records, device)
            common, common_ms = timed_cuda(
                partial(
                    append_jagged_suffix,
                    target_model,
                    empty_jagged_slice(
                        ids,
                        target_version,
                        cfg.num_layers,
                        cfg.num_heads * cfg.head_dim,
                        torch.float16,
                        device,
                    ),
                    suffix["item_ids"],
                    suffix["behaviors"],
                    suffix["time_deltas"],
                    suffix["lengths"],
                ),
                device,
            )
            costs["common_latest"] += common_ms
            published, split_ms = timed_cuda(
                partial(_split_cache, common.cache),
                device,
            )
            costs["common_publication"] += split_ms
            next_cache.update(published)
            hidden = common.last_appended_hidden
            if hidden is None:
                raise RuntimeError("organic short latest has no hidden state")
            scores = target_model.item_emb.score(
                hidden,
                all_items.unsqueeze(0).expand(len(ids), -1),
            )
            for row, (descriptor, record_id) in enumerate(
                zip(short, ids, strict=True)
            ):
                state_before = states.get(record_id)
                state_after = CacheLifecycleState.exact(
                    record_id,
                    target_version,
                )
                next_states[record_id] = state_after
                record = target_window.records[int(descriptor["user_id"])]
                value = {
                    "record_id": record_id,
                    "user_id": int(descriptor["user_id"]),
                    "evaluation_role": descriptor["evaluation_role"],
                    "metrics": None,
                }
                task = _task_value(
                    descriptor["evaluation_role"],
                    record.engaged_positive_item_ids,
                    scores[row],
                    scores[row],
                    None,
                )
                if task is not None:
                    value["task"] = task
                endpoint_records.append(value)
                transition = transitions[record_id]
                lineage_by_record[record_id] = {
                    "record_id": record_id,
                    "user_id": int(descriptor["user_id"]),
                    "old_history_hash": transition.old_history_hash,
                    "new_history_hash": transition.new_history_hash,
                    "overlap_tokens": transition.overlap,
                    "evicted_tokens": transition.evicted,
                    "appended_tokens": transition.appended,
                    "source_prefix_tokens": transition.new_length,
                    "common_latest_tokens": 1,
                    "foreground_status": transition.status,
                    "previous_actual_consumed": (
                        transition.previous_actual_consumed
                    ),
                    "action": "exact",
                    "decision": {
                        "action": "exact",
                        "reason": "short_history_no_prefix",
                    },
                    "state_before": (
                        state_before.to_dict()
                        if state_before is not None
                        else None
                    ),
                    "state_after": state_after.to_dict(),
                    "program_sha256": program_descriptor["sha256"],
                    "norm_shift_q090": None,
                    "candidate_evaluated": False,
                }
            del common, published, hidden, scores, suffix
        if reference_descriptors:
            del exact_reference_group, reference_batch, reference_records
        publication_persistent_peak_bytes = max(
            publication_persistent_peak_bytes,
            resident_cache_bytes(candidate_by_record)
            + resident_cache_bytes(next_cache),
        )
    for descriptor in (
        value
        for group in groups
        for value in group
        if int(value["record_id"]) not in lineage_by_record
    ):
        record_id = int(descriptor["record_id"])
        transition = transitions[record_id]
        lineage_by_record[record_id] = {
            "record_id": record_id,
            "user_id": int(descriptor["user_id"]),
            "old_history_hash": transition.old_history_hash,
            "new_history_hash": transition.new_history_hash,
            "overlap_tokens": transition.overlap,
            "evicted_tokens": transition.evicted,
            "appended_tokens": transition.appended,
            "source_prefix_tokens": transition.new_length,
            "common_latest_tokens": (
                1 if transition.new_history_hash is not None else 0
            ),
            "foreground_status": transition.status,
            "previous_actual_consumed": False,
            "action": "expire" if transition.status == "expired" else "absent",
            "decision": None,
            "state_before": (
                states[record_id].to_dict()
                if record_id in states
                else None
            ),
            "state_after": None,
            "program_sha256": program_descriptor["sha256"],
            "norm_shift_q090": None,
            "candidate_evaluated": False,
        }
    candidate_cache_consumed = not candidate_by_record
    if (
        source_prefixes
        or candidate_by_record
        or set(next_cache) != set(next_states)
    ):
        raise RuntimeError("organic target cache and lifecycle states differ")
    output_resident_bytes = resident_cache_bytes(next_cache)
    cuda_peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    step_cost = cost_summary(costs, exact_reference_ms)
    selector_diagnostic = posthoc_selector_diagnostics(
        norm_shifts,
        candidate_errors,
        planned,
    )
    endpoint = _summarize_endpoint(
        target_version,
        target_window.target_date,
        endpoint_records,
        len(next_cache),
        sum(
            record.history is not None and len(record.history) == 1
            for record in target_window.records.values()
        ),
    )
    step_checks = {
        "target_cache_covers_residents": set(next_cache)
        == {
            int(value["record_id"])
            for group in groups
            for value in group
            if target_window.records[int(value["user_id"])].history
            is not None
        },
        "cache_lengths_match_target_history": all(
            int(cache.lengths[0])
            == len(
                target_window.records[
                    int(record_by_id[record_id]["user_id"])
                ].history
            )
            for record_id, cache in next_cache.items()
        ),
        "cache_versions_match_target": all(
            cache.served_kv_target == f"theta{target_version}"
            and cache.migration_anchor_version == f"theta{target_version}"
            for cache in next_cache.values()
        ),
        "history_overlap_arithmetic": all(
            value.old_length - value.evicted == value.overlap
            and value.overlap + value.appended == value.new_length
            for value in transitions.values()
        ),
        "source_prefix_lengths_match_target_h_minus_latest": (
            source_prefix_lengths_match
        ),
        "prefix_plus_latest_matches_target_history": all(
            transition.new_length
            + (
                1
                if target_window.records[
                    int(record_by_id[record_id]["user_id"])
                ].history
                is not None
                else 0
            )
            == (
                len(
                    target_window.records[
                        int(record_by_id[record_id]["user_id"])
                    ].history
                )
                if target_window.records[
                    int(record_by_id[record_id]["user_id"])
                ].history
                is not None
                else 0
            )
            for record_id, transition in transitions.items()
        ),
        "lineage_covers_fixed_group": len(lineage_by_record)
        == sum(len(group) for group in groups),
        "all_prefix_candidates_evaluated": all(
            lineage_by_record[record_id]["candidate_evaluated"]
            for record_id in eligible_ids
        ),
        "no_reuse_records_bypass_candidate": all(
            not lineage_by_record[record_id]["candidate_evaluated"]
            for record_id in direct_exact_ids
        ),
        "migration_depth_bounded": all(
            value.migration_depth <= MAX_MIGRATION_DEPTH
            for value in next_states.values()
        ),
        "previous_cache_consumed_during_foreground": (
            previous_cache_consumed
        ),
        "source_cache_consumed_during_probe": source_cache_consumed,
        "candidate_cache_consumed_during_publication": (
            candidate_cache_consumed
        ),
        "reuse_diagnostic_coverage": all(
            record_id in reuse_task_by_record
            for record_id in eligible_ids
            if record_by_id[record_id]["evaluation_role"] == "final_test"
            and target_window.records[
                int(record_by_id[record_id]["user_id"])
            ].engaged_positive_item_ids
        ),
        "exact_reference_covers_every_resident_prefix": (
            exact_reference_records
            == sum(
                target_window.records[int(value["user_id"])].history
                is not None
                and len(
                    target_window.records[int(value["user_id"])].history
                )
                >= 2
                for group in groups
                for value in group
            )
        ),
        "one_exact_reference_execution_per_nonempty_fixed_group": (
            exact_reference_groups
            == sum(
                any(
                    target_window.records[int(value["user_id"])].history
                    is not None
                    and len(
                        target_window.records[int(value["user_id"])].history
                    )
                    >= 2
                    for value in group
                )
                for group in groups
            )
        ),
    }
    if not all(step_checks.values()):
        failed = [
            key for key, passed in step_checks.items() if not passed
        ]
        raise RuntimeError(
            f"organic edge {source_version}->{target_version} failed: {failed}"
        )
    shift_values = tuple(norm_shifts.values())
    scheduled_exact_records = sum(
        value["action"] == "exact" and value["candidate_evaluated"]
        for value in lineage_by_record.values()
    )
    natural_exact_records = sum(
        value["decision"] is not None
        and value["decision"].get("reason")
        == "no_reusable_prefix_target_exact"
        for value in lineage_by_record.values()
    )
    short_common_records = sum(
        value["decision"] is not None
        and value["decision"].get("reason")
        == "short_history_no_prefix"
        for value in lineage_by_record.values()
    )
    total_exact_state_records = sum(
        value["action"] == "exact"
        for value in lineage_by_record.values()
    )
    total_prefix_exact_records = (
        scheduled_exact_records + natural_exact_records
    )
    step = {
        "source_version": source_version,
        "target_version": target_version,
        "source_target_date": old_window.target_date,
        "prediction_target_date": target_window.target_date,
        "records": len(lineage_by_record),
        "resident_records": len(next_cache),
        "actions": {
            "migrate": sum(
                value["action"] == "migrate"
                for value in lineage_by_record.values()
            ),
            "exact": sum(
                value["action"] == "exact"
                for value in lineage_by_record.values()
            ),
            "discarded_exact_candidates": sum(
                value["action"] == "exact"
                and value["candidate_evaluated"]
                for value in lineage_by_record.values()
            ),
            "scheduled_selector_exact": scheduled_exact_records,
            "natural_no_reuse_target_exact": natural_exact_records,
            "short_common_only": short_common_records,
            "scheduled_exact_fraction_of_reusable": (
                scheduled_exact_records / len(eligible_ids)
                if eligible_ids
                else None
            ),
            "natural_exact_fraction_of_resident": (
                natural_exact_records / len(next_cache)
                if next_cache
                else None
            ),
            "total_exact_fraction_of_resident": (
                total_prefix_exact_records / len(next_cache)
                if next_cache
                else None
            ),
            "total_exact_state_fraction_of_resident": (
                total_exact_state_records / len(next_cache)
                if next_cache
                else None
            ),
            "configured_selector_fraction_applies_to": (
                "continued reusable prefixes only"
            ),
            "expired": sum(
                value["action"] == "expire"
                for value in lineage_by_record.values()
            ),
            "absent": sum(
                value["action"] == "absent"
                for value in lineage_by_record.values()
            ),
        },
        "cost": step_cost,
        "checks": step_checks,
        "memory_ownership": {
            "strategy": "destructive_groupwise_state_handoff",
            "batch_size": BATCH_SIZE,
            "maximum_simultaneous_full_cohort_dicts": 1,
            "previous_resident_records": previous_resident_records,
            "source_resident_records": source_resident_records,
            "candidate_resident_records": len(eligible_ids),
            "output_resident_records": len(next_cache),
            "logical_kv_bytes": {
                "previous": previous_resident_bytes,
                "source": source_resident_bytes,
                "candidate_prefix": candidate_resident_bytes,
                "output": output_resident_bytes,
                "foreground_previous_plus_source_peak": (
                    foreground_persistent_peak_bytes
                ),
                "probe_source_plus_candidate_peak": (
                    probe_persistent_peak_bytes
                ),
                "publication_candidate_plus_output_peak": (
                    publication_persistent_peak_bytes
                ),
                "measured_persistent_handoff_peak": max(
                    foreground_persistent_peak_bytes,
                    probe_persistent_peak_bytes,
                    publication_persistent_peak_bytes,
                ),
            },
            "cuda_max_memory_allocated_bytes": (
                cuda_peak_allocated_bytes
            ),
            "cuda_peak_includes_models_and_batch_temporaries": True,
        },
        "selector": {
            "name": "age_norm_shift_budget",
            "configured_exact_fraction": EXACT_FRACTION,
            "maximum_migration_depth": MAX_MIGRATION_DEPTH,
            "scheduler_seed": SCHEDULER_SEED,
            "priority": (
                "depth deadline, then migration depth, current norm shift, "
                "stable SHA256 tie"
            ),
            "current_edge_norm_shift": {
                "records": len(norm_shifts),
                "p50": (
                    float(np.median(shift_values))
                    if shift_values
                    else None
                ),
                "p90": (
                    float(np.quantile(shift_values, 0.9))
                    if shift_values
                    else None
                ),
                "maximum": max(shift_values) if shift_values else None,
            },
            "audit": selector_evidence,
            "future_edge_severity_used": False,
        },
        "posthoc_selector_diagnostic": selector_diagnostic,
        "lineage": [
            lineage_by_record[index]
            for index in sorted(lineage_by_record)
        ],
        "program": {
            "sha256": program_descriptor["sha256"],
            "compiler_scoring_excluded": True,
        },
    }
    del target_model
    gc.collect()
    torch.cuda.empty_cache()
    return next_cache, next_states, endpoint_records, endpoint, step


def _global_checks(
    windows,
    endpoints,
    steps,
    manifest,
) -> dict[str, bool]:
    checks = {
        "twelve_endpoints": len(endpoints) == 12,
        "eleven_updates": len(steps) == 11,
        "fixed_lineage_rows": all(
            len(step["lineage"]) == len(manifest["records"])
            for step in steps
        ),
        "adjacent_versions": all(
            step["target_version"] == step["source_version"] + 1
            for step in steps
        ),
        "previous_actual_consumption_disclosed": all(
            value["previous_actual_consumed"]
            == (
                value["foreground_status"] == "continued"
                and value["overlap_tokens"] > 0
            )
            for step in steps
            for value in step["lineage"]
        ),
        "history_hashes_match_windows": all(
            value["new_history_hash"]
            == windows[step["target_version"]]
            .records[value["user_id"]]
            .history_sha256
            for step in steps
            for value in step["lineage"]
        ),
        "labels_not_used_for_routing": True,
        "all_active_candidates_evaluated": all(
            value["candidate_evaluated"]
            for step in steps
            for value in step["lineage"]
            if value["decision"] is not None
            and value["decision"].get("reason")
            in {
                "depth_deadline_after_probe",
                "norm_shift_exact_quota",
                "norm_shift_migrate",
            }
        ),
        "constant_exact_budget": all(
            step["selector"]["configured_exact_fraction"]
            == EXACT_FRACTION
            for step in steps
        ),
        "maximum_depth_four": all(
            value["state_after"] is None
            or (
                0
                <= int(value["state_after"]["migration_depth"])
                <= MAX_MIGRATION_DEPTH
            )
            for step in steps
            for value in step["lineage"]
        ),
        "all_step_checks_pass": all(
            all(step["checks"].values()) for step in steps
        ),
        "theta0_layout_matches": endpoints[0]["anchor"][
            "layout_matches_history"
        ],
    }
    return checks


def smoke_payload() -> dict:
    policy = constant_policy()
    states = tuple(
        CacheLifecycleState.exact(record_id, 0)
        for record_id in range(20)
    )
    decisions = select_norm_shift_decisions(
        states,
        1,
        {record_id: record_id / 100 for record_id in range(20)},
    )
    costs = cost_summary(
        {
            "foreground_evict": 1.0,
            "foreground_incremental_append": 2.0,
            "candidate_transform": 2.0,
            "router_probe": 0.5,
            "exact_refresh": 1.0,
            "natural_direct_exact": 0.25,
            "publication": 1.0,
            "common_latest": 1.0,
            "common_publication": 0.5,
        },
        20.0,
    )
    return {
        "status": "smoke_passed",
        "policy": policy.to_dict(),
        "exact_actions": sum(
            value.action == "exact" for value in decisions
        ),
        "migrate_actions": sum(
            value.action == "migrate" for value in decisions
        ),
        "cost": costs,
        "memory_preflight": {
            "strategy": "destructive_groupwise_state_handoff",
            "maximum_simultaneous_full_cohort_dicts": 1,
            "prepare_consumes_previous_cache": True,
            "probe_consumes_source_cache": True,
            "publication_consumes_candidate_cache": True,
            "batch_temporaries": BATCH_SIZE,
        },
    }


def run_chain(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    plan, metadata, training, cfg, manifest, checkpoints = load_inputs(
        args.prepared_data,
        args.training_result,
        args.checkpoint_dir,
    )
    user_ids = tuple(
        int(value["user_id"]) for value in manifest["records"]
    )
    windows = reconstruct_organic_windows(plan, user_ids)
    window_checks = validate_windows(windows, manifest)
    timestamp_diagnostic = raw_timestamp_overlap_diagnostic(windows)
    compiler = json.loads(Path(args.compiler_result).read_text())
    compiler_checks = validate_compiler_payload(
        compiler,
        manifest,
        windows,
        checkpoints,
    )
    groups = fixed_record_groups(manifest, args.batch_size)
    record_by_user = {
        int(value["user_id"]): value for value in manifest["records"]
    }
    record_by_id = {
        int(value["record_id"]): value for value in manifest["records"]
    }
    all_items = torch.arange(
        1,
        cfg.num_prediction_items + 1,
        device=device,
    )
    (
        cache_by_record,
        states,
        endpoint_records,
        endpoint,
    ) = _initialize_theta0(
        cfg,
        args.checkpoint_dir,
        windows[0],
        groups,
        record_by_user,
        device,
        all_items,
    )
    endpoints = [endpoint]
    steps = []
    policy = constant_policy()
    operator = DirectOldKVFusedOperator(**LAUNCH)
    started = time.perf_counter()
    for source_version in range(NUM_EDGES):
        (
            cache_by_record,
            states,
            endpoint_records,
            endpoint,
            step,
        ) = _run_edge(
            args,
            cfg,
            compiler,
            windows[source_version],
            windows[source_version + 1],
            groups,
            record_by_id,
            cache_by_record,
            states,
            policy,
            operator,
            all_items,
        )
        endpoints.append(endpoint)
        steps.append(step)
        print(
            json.dumps(
                {
                    "source_version": source_version,
                    "target_version": source_version + 1,
                    "target_date": endpoint["target_date"],
                    "actions": step["actions"],
                    "label_free_metrics": endpoint[
                        "label_free_metrics"
                    ],
                    "cost": step["cost"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    global_checks = _global_checks(
        windows,
        endpoints,
        steps,
        manifest,
    )
    failed_checks = {
        "causality": [
            key for key, passed in window_checks.items() if not passed
        ],
        "compiler": [
            key for key, passed in compiler_checks.items() if not passed
        ],
        "chain": [
            key for key, passed in global_checks.items() if not passed
        ],
    }
    failed_checks = {
        family: values
        for family, values in failed_checks.items()
        if values
    }
    if failed_checks:
        raise RuntimeError(
            f"organic chain protocol checks failed: {failed_checks}"
        )
    cumulative_costs = {
        key: sum(step["cost"][key] for step in steps)
        for key in (
            "foreground_evict",
            "foreground_incremental_append",
            "foreground_ms",
            "candidate_transform",
            "router_probe",
            "migration_ms",
            "exact_refresh",
            "natural_direct_exact",
            "publication",
            "common_latest",
            "common_publication",
            "common_shared_ms",
            "update_only_ms",
            "symmetric_lifecycle_numerator_ms",
            "symmetric_lifecycle_denominator_ms",
            "common_inclusive_numerator_ms",
            "common_inclusive_denominator_ms",
            "conservative_asymmetric_numerator_ms",
            "all_exact_reference_ms",
        )
    }
    cumulative_costs["primary_update_only_ratio"] = (
        cumulative_costs["update_only_ms"]
        / cumulative_costs["all_exact_reference_ms"]
    )
    cumulative_costs["update_only_ratio"] = (
        cumulative_costs["update_only_ms"]
        / cumulative_costs["all_exact_reference_ms"]
    )
    cumulative_costs["symmetric_lifecycle_ratio"] = (
        cumulative_costs["symmetric_lifecycle_numerator_ms"]
        / cumulative_costs["symmetric_lifecycle_denominator_ms"]
    )
    cumulative_costs["common_inclusive_ratio"] = (
        cumulative_costs["common_inclusive_numerator_ms"]
        / cumulative_costs["common_inclusive_denominator_ms"]
    )
    cumulative_costs["conservative_asymmetric_ratio"] = (
        cumulative_costs["conservative_asymmetric_numerator_ms"]
        / cumulative_costs["all_exact_reference_ms"]
    )
    cached_endpoints = [
        endpoint
        for endpoint in endpoints
        if endpoint["cache_eligible_records"] > 0
    ]
    return {
        "protocol": CHAIN_PROTOCOL,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "status": "complete",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "configuration": {
            "dataset": "KuaiRand-1K",
            "split": "4+12",
            "seed": 0,
            "batch_size": BATCH_SIZE,
            "device": str(device),
            "records": len(manifest["records"]),
            "policy": policy.to_dict(),
            "model": training["model"],
        },
        "measurement_boundary": {
            "primary_metric": (
                "update_only / all_exact_reference; update_only includes one "
                "candidate per reusable prefix, routing, scheduled exact, "
                "natural no-reuse target exact, and mixed-prefix publication"
            ),
            "symmetric_lifecycle_metric": (
                "(foreground + update_only) / "
                "(foreground + all_exact_reference)"
            ),
            "common_inclusive_metric": (
                "(foreground + update_only + common) / "
                "(foreground + all_exact_reference + common)"
            ),
            "conservative_diagnostic": (
                "(foreground + update_only) / all_exact_reference; this "
                "asymmetric value is not the primary or end-to-end claim"
            ),
            "foreground": (
                "continued-prefix eviction/crop and source-model incremental "
                "append to H[:-1]"
            ),
            "overlap_identity": (
                "label-free exact timestamp/item/behavior event keys; "
                "recommendation labels never affect overlap or routing"
            ),
            "retained_prefix_semantics": (
                "continued rows tail-crop the prior actual mixed K/V without "
                "replaying retained tokens under the source model"
            ),
            "denominator": (
                "exactly one target-model compute_kv plus pack batch per "
                "nonempty fixed group over every resident H[:-1] with "
                "history length at least two; natural target exact numerator "
                "executes separately"
            ),
            "common_latest": (
                "measured and disclosed separately; excluded from both "
                "ratios because both paths process the identical latest token"
            ),
            "common_publication": (
                "per-record full-cache split after common latest is measured "
                "and excluded from both ratios"
            ),
            "excluded": (
                "program compilation, catalog scoring, task metrics, "
                "semantic diagnostics, offline exact-reference scoring, "
                "and host-to-device sequence construction"
            ),
            "source_residency": "prior actual K/V is hot in HBM",
        },
        "inputs": {
            "prepared_data": {
                "path": args.prepared_data,
                "sha256": sha256(args.prepared_data),
                "protocol": metadata["protocol"],
            },
            "training_result": {
                "path": args.training_result,
                "sha256": sha256(args.training_result),
                "protocol": training["protocol"],
            },
            "compiler": {
                "path": args.compiler_result,
                "sha256": sha256(args.compiler_result),
                "protocol": compiler["protocol"],
            },
            "checkpoints": checkpoints,
            "manifest": manifest,
            "windows": [
                {
                    "version": window.version,
                    "target_date": window.target_date,
                    "content_sha256": window.content_sha256,
                }
                for window in windows
            ],
        },
        "checks": {
            "causality": window_checks,
            "compiler": compiler_checks,
            "chain": global_checks,
        },
        "causality_diagnostics": {
            "raw_timestamp_overlap": timestamp_diagnostic,
        },
        "labels_used_for_routing": False,
        "automatic_certificate_action_changes": False,
        "endpoints": endpoints,
        "steps": steps,
        "cumulative_gpu_cost": cumulative_costs,
        "worst_endpoint_label_free_metrics": {
            metric: min(
                endpoint["label_free_metrics"][metric]
                for endpoint in cached_endpoints
                if endpoint["label_free_metrics"][metric] is not None
            )
            for metric in (
                "cache_fidelity_q090",
                "score_cosine",
                "top100_overlap",
            )
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.smoke_test:
        print(json.dumps(smoke_payload(), indent=2))
        return
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    payload = run_chain(args)
    save_json(payload, args.output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": args.output,
                "endpoints": len(payload["endpoints"]),
                "steps": len(payload["steps"]),
                "cumulative_gpu_cost": payload[
                    "cumulative_gpu_cost"
                ],
                "checks": payload["checks"],
            }
        )
    )


if __name__ == "__main__":
    main()
