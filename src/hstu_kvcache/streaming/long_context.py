from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..data import PREPARED_PROTOCOL, StreamingDataPlan
from ..models import HSTU, HSTUConfig

TRAINING_PROTOCOL = "kuairand_long_context_8plus8_training_v2"
MOTIVATION_PROTOCOL = "kuairand_long_context_8plus8_motivation_all_pairs_v3"
METHOD_PROTOCOL = "kuairand_long_context_8plus8_compiled_migration_v2"
SYNC_DESIGN_PROTOCOL = (
    "kuairand_long_context_4plus12_progressive_sync_design_v1"
)
SYNC_SYSTEM_PROTOCOL = (
    "kuairand_long_context_4plus12_progressive_sync_system_v1"
)
TWO_GPU_SYSTEM_PROTOCOL = (
    "kuairand_long_context_4plus12_two_gpu_migration_system_v2"
)
COHORT_JAGGED_SYSTEM_PROTOCOL = (
    "kuairand_long_context_4plus12_cohort_jagged_system_v3"
)
LONG_CONTEXT_DATES = tuple(f"202204{day:02d}" for day in range(8, 24))
LONG_CONTEXT_BASE_DATES = LONG_CONTEXT_DATES[:8]
LONG_CONTEXT_ONLINE_DATES = LONG_CONTEXT_DATES[8:]
SUPPORTED_LONG_CONTEXT_BASE_DAYS = (4, 6, 8)


def long_context_split_name(base_days: int) -> str:
    if base_days not in SUPPORTED_LONG_CONTEXT_BASE_DAYS:
        raise ValueError(
            f"base_days must be selected from {SUPPORTED_LONG_CONTEXT_BASE_DAYS}"
        )
    return f"{base_days}plus{len(LONG_CONTEXT_DATES) - base_days}"


def long_context_dates(base_days: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    long_context_split_name(base_days)
    return LONG_CONTEXT_DATES[:base_days], LONG_CONTEXT_DATES[base_days:]


def prepared_protocol_for_base_days(base_days: int) -> str:
    if base_days == 8:
        return PREPARED_PROTOCOL
    return (
        f"kuairand_long_context_{long_context_split_name(base_days)}"
        "_data_exploration_v1"
    )


def training_protocol_for_base_days(base_days: int) -> str:
    if base_days == 8:
        return TRAINING_PROTOCOL
    return (
        f"kuairand_long_context_{long_context_split_name(base_days)}"
        "_training_exploration_v1"
    )


def motivation_protocol_for_base_days(base_days: int) -> str:
    if base_days == 8:
        return MOTIVATION_PROTOCOL
    return (
        f"kuairand_long_context_{long_context_split_name(base_days)}"
        "_motivation_all_pairs_exploration_v1"
    )


def make_long_context_config(
    num_items: int,
    num_prediction_items: int,
    num_behaviors: int,
) -> HSTUConfig:
    return HSTUConfig(
        num_items=num_items,
        num_prediction_items=num_prediction_items,
        num_behaviors=num_behaviors,
        hidden_size=512,
        num_layers=16,
        num_heads=8,
        head_dim=64,
        max_seq_len=2048,
        activation="relu",
    )


def model_shape_summary(cfg: HSTUConfig) -> dict:
    with torch.device("meta"):
        model = HSTU(cfg)
    return {
        "config": asdict(cfg),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "precision": "float32",
        "attention_execution": "dense_length_bucketed",
    }


def validate_long_context_plan(
    plan: StreamingDataPlan,
    metadata: dict,
    base_days: int = 8,
) -> None:
    base_dates, online_dates = long_context_dates(base_days)
    expected_metadata = {
        "protocol": prepared_protocol_for_base_days(base_days),
        "base_dates": list(base_dates),
        "online_dates": list(online_dates),
        "base_date_count": len(base_dates),
        "online_date_count": len(online_dates),
        "total_dates": 16,
        "max_seq_len": 2048,
        "history_window_days": 8,
        "max_prediction_items": 50000,
        "num_context_items": 312144,
        "num_prediction_items": 50000,
        "context_hash_buckets": 262144,
        "num_behaviors": 9,
    }
    if base_days == 8:
        expected_metadata["num_users"] = 965
    mismatches = {
        name: {"expected": expected, "actual": metadata.get(name)}
        for name, expected in expected_metadata.items()
        if metadata.get(name) != expected
    }
    plan_values = {
        "base_dates": plan.base_dates,
        "online_dates": plan.stream_dates,
        "max_seq_len": plan.max_seq_len,
        "history_window_days": plan.history_window_days,
        "num_context_items": plan.num_items,
        "num_prediction_items": plan.num_prediction_items,
        "num_behaviors": plan.num_behaviors,
        "num_users": plan.num_users,
    }
    expected_plan = {
        "base_dates": expected_metadata["base_dates"],
        "online_dates": expected_metadata["online_dates"],
        "max_seq_len": expected_metadata["max_seq_len"],
        "history_window_days": expected_metadata["history_window_days"],
        "num_context_items": expected_metadata["num_context_items"],
        "num_prediction_items": expected_metadata["num_prediction_items"],
        "num_behaviors": expected_metadata["num_behaviors"],
        "num_users": metadata.get("num_users"),
    }
    mismatches.update(
        {
            f"plan.{name}": {"expected": expected_plan[name], "actual": actual}
            for name, actual in plan_values.items()
            if actual != expected_plan[name]
        }
    )
    if mismatches:
        raise ValueError(f"prepared long-context protocol mismatch: {mismatches}")


def prefix_state_footprint(
    samples: list[dict],
    cfg: HSTUConfig,
    element_size: int = 4,
) -> dict:
    observed_lengths = np.asarray(
        [
            len(sample["history"]["item_ids"])
            for sample in samples
        ],
        dtype=np.int64,
    )
    if np.any(observed_lengths > cfg.max_seq_len):
        raise ValueError("evaluation history exceeds the model context length")
    history_lengths = observed_lengths
    available_lengths = np.asarray(
        [
            int(
                sample["history"].get(
                    "available_length_before_token_cap",
                    len(sample["history"]["item_ids"]),
                )
            )
            for sample in samples
        ],
        dtype=np.int64,
    )
    prefix_lengths = np.maximum(history_lengths - 1, 0)
    prefix_tokens = int(prefix_lengths.sum())
    kv_elements = (
        prefix_tokens
        * cfg.num_layers
        * 2
        * cfg.num_heads
        * cfg.head_dim
    )
    norm_elements = prefix_tokens * cfg.num_layers * cfg.hidden_size
    quantiles = {
        f"p{percentile}": float(np.percentile(history_lengths, percentile))
        for percentile in (50, 90, 99)
    } if len(history_lengths) else {}
    return {
        "definition": "logical unpadded prefix state",
        "dtype_bytes": element_size,
        "users": len(samples),
        "history_length": {
            **quantiles,
            "max": int(history_lengths.max()) if len(history_lengths) else 0,
            "at_max_seq_len_fraction": (
                float(np.mean(history_lengths == cfg.max_seq_len))
                if len(history_lengths)
                else 0.0
            ),
            "token_truncated_users": (
                int(np.sum(available_lengths > cfg.max_seq_len))
                if len(available_lengths)
                else 0
            ),
            "token_truncated_fraction": (
                float(np.mean(available_lengths > cfg.max_seq_len))
                if len(available_lengths)
                else 0.0
            ),
            "tokens_dropped_by_token_cap": (
                int(np.maximum(available_lengths - cfg.max_seq_len, 0).sum())
                if len(available_lengths)
                else 0
            ),
        },
        "context_invariant": (
            "fresh, stale, and migrated paths consume the identical deterministic "
            "tail-cropped resident prefix; full length is at most max_seq_len and "
            "prefix length is at most max_seq_len - 1"
        ),
        "cache_version_semantics": (
            "checkpoint version used to encode the resident prefix, not the date "
            "when a physical cache snapshot was materialized"
        ),
        "prefix_tokens": prefix_tokens,
        "kv_elements": kv_elements,
        "kv_bytes": kv_elements * element_size,
        "normalized_state_elements": norm_elements,
        "normalized_state_bytes": norm_elements * element_size,
        "compiled_method_state_bytes": (kv_elements + norm_elements) * element_size,
    }


def load_checkpoint_model(
    cfg: HSTUConfig,
    checkpoint_dir: str | Path,
    version: int,
    device: torch.device | str,
) -> HSTU:
    model = HSTU(cfg).to(device)
    state = torch.load(
        Path(checkpoint_dir) / f"theta_{version}.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def parameter_distance(current: HSTU, base: HSTU) -> float:
    numerator = torch.zeros((), device=next(current.parameters()).device)
    denominator = torch.zeros_like(numerator)
    with torch.no_grad():
        for current_parameter, base_parameter in zip(
            current.parameters(),
            base.parameters(),
            strict=True,
        ):
            numerator += (current_parameter - base_parameter).float().square().sum()
            denominator += base_parameter.float().square().sum()
    return float((numerator.sqrt() / denominator.sqrt().clamp_min(1e-12)).item())


def reconstruct_online_eval_samples(
    plan: StreamingDataPlan,
    versions: tuple[int, ...],
    max_users: int,
) -> dict[int, tuple[str, list[dict]]]:
    last_evaluable_version = len(plan.stream_dates) - 1
    if (
        not versions
        or min(versions) < 1
        or max(versions) > last_evaluable_version
    ):
        raise ValueError(
            "evaluation versions must be selected from theta1 through "
            f"theta{last_evaluable_version}"
        )
    wanted = set(versions)
    output = {}
    plan.init_base()
    for online_index, date in enumerate(plan.stream_dates):
        current_version = online_index
        if current_version in wanted:
            output[current_version] = (
                date,
                plan.get_eval_set(date, max_users),
            )
        if online_index < len(plan.stream_dates) - 1:
            plan.ingest_day(date)
    if set(output) != wanted:
        raise RuntimeError("prepared stream does not cover all requested model versions")
    return output
