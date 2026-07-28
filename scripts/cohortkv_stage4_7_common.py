from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.models import HSTUConfig
from hstu_kvcache.streaming import (
    training_protocol_for_base_days,
    validate_long_context_plan,
)

EXPERIMENT_PROTOCOL = "cohortkv_single_config_organic_lifecycle_v1"
MANIFEST_PROTOCOL = "cohortkv_single_config_organic_manifest_v1"
COMPILER_PROTOCOL = "cohortkv_single_config_organic_adjacent_compiler_v1"
CHAIN_PROTOCOL = "cohortkv_single_config_organic_recursive_chain_v1"
PREPARED_PATH = (
    "data/processed/kuairand_long_context_4plus12_exploration_v1.npz"
)
TRAINING_PATH = (
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
CHECKPOINT_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
)
RUNTIME_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0/"
    "single_config_v1/stage4_7_organic_runtime"
)
COMPILER_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_7_organic_adjacent_compiler_seed0.json"
)
CHAIN_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_7_organic_full_chain_seed0.json"
)
COHORT_SIZE = 682
COHORT_KEY = "cohortkv_single_config_organic_lifecycle_v1"
ROLE_SPLIT_SEED = 9151
CERTIFICATE_SPLIT_SEED = 27183
EXPECTED_MODEL = {
    "hidden_size": 512,
    "num_layers": 16,
    "num_heads": 8,
    "head_dim": 64,
    "max_seq_len": 2048,
    "num_prediction_items": 50000,
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def organic_user_ids(num_users: int) -> tuple[int, ...]:
    if num_users < COHORT_SIZE:
        raise ValueError("prepared cohort is smaller than the organic cohort")
    ranked = sorted(
        range(1, num_users + 1),
        key=lambda user_id: (
            hashlib.sha256(f"{COHORT_KEY}:{user_id}".encode()).digest(),
            user_id,
        ),
    )
    return tuple(sorted(ranked[:COHORT_SIZE]))


def role_assignment(user_ids: tuple[int, ...]) -> dict[int, str]:
    if len(user_ids) != COHORT_SIZE or len(set(user_ids)) != len(user_ids):
        raise ValueError("organic cohort differs")
    first = np.random.default_rng(ROLE_SPLIT_SEED).permutation(len(user_ids))
    fit = [user_ids[index] for index in first[:40]]
    selection = [user_ids[index] for index in first[40:100]]
    remaining = [user_ids[index] for index in first[100:]]
    second = np.random.default_rng(CERTIFICATE_SPLIT_SEED).permutation(
        len(remaining)
    )
    certificate = [remaining[index] for index in second[:60]]
    final_test = [remaining[index] for index in second[60:]]
    groups = {
        "fit": fit,
        "program_selection": selection,
        "certificate": certificate,
        "final_test": final_test,
    }
    assignment = {
        user_id: role
        for role, values in groups.items()
        for user_id in values
    }
    if len(assignment) != len(user_ids):
        raise RuntimeError("organic roles do not cover the cohort")
    return assignment


def build_manifest(plan, metadata: dict, training: dict) -> dict:
    user_ids = organic_user_ids(plan.num_users)
    roles = role_assignment(user_ids)
    raw_by_user = {
        int(user_index): int(raw_user_id)
        for raw_user_id, user_index in plan.trace.user_map.items()
    }
    records = [
        {
            "record_id": record_id,
            "user_id": user_id,
            "raw_user_id": raw_by_user[user_id],
            "evaluation_role": roles[user_id],
        }
        for record_id, user_id in enumerate(user_ids)
    ]
    value = {
        "protocol": MANIFEST_PROTOCOL,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "selection_boundary": {
            "population": "base-only prepared cohort",
            "prepared_users": plan.num_users,
            "selected_records": len(records),
            "selection": (
                "lowest keyed SHA256 ranks over base-fitted internal user ids"
            ),
            "selection_key": COHORT_KEY,
            "future_activity_used": False,
        },
        "roles": {
            role: sum(record["evaluation_role"] == role for record in records)
            for role in (
                "fit",
                "program_selection",
                "certificate",
                "final_test",
            )
        },
        "timeline": {
            "base_dates": metadata["base_dates"],
            "target_dates": metadata["online_dates"],
            "versions": [f"theta{version}" for version in range(12)],
            "rule": (
                "theta_v uses history available before target_dates[v] and "
                "predicts that unseen date before ingestion"
            ),
        },
        "training_seed": int(training["args"]["seed"]),
        "records": records,
    }
    value["content_sha256"] = content_sha256(value)
    return value


def history_view_sha256(window, user_ids) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "version": window.version,
                "target_date": window.target_date,
                "histories": [
                    {
                        "user_id": user_id,
                        "as_of_timestamp_ms": (
                            window.records[user_id].as_of_timestamp_ms
                        ),
                        "history_sha256": (
                            window.records[user_id].history_sha256
                        ),
                    }
                    for user_id in user_ids
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def sample_from_record(
    record,
    include_positives: bool = True,
) -> dict | None:
    history = record.history
    if history is None:
        return None
    return {
        "history": {
            "item_ids": history.item_ids,
            "behaviors": history.behaviors,
            "time_deltas": history.time_deltas,
            "labels": history.labels,
            "timestamps": history.timestamps,
            "user_id": record.user_id,
            "available_length_before_token_cap": (
                history.available_length_before_token_cap
            ),
            "token_truncated": history.token_truncated,
        },
        "pos_items": (
            list(record.engaged_positive_item_ids)
            if include_positives
            else []
        ),
    }


def samples_for_users(
    window,
    user_ids,
    include_positives: bool = True,
) -> list[dict]:
    output = []
    for user_id in user_ids:
        sample = sample_from_record(
            window.records[user_id],
            include_positives=include_positives,
        )
        if sample is not None:
            output.append(sample)
    return output


def load_inputs(
    prepared_path: str | Path,
    training_path: str | Path,
    checkpoint_dir: str | Path,
) -> tuple[object, dict, dict, HSTUConfig, dict, list[dict]]:
    training = json.loads(Path(training_path).read_text())
    if (
        training.get("protocol") != training_protocol_for_base_days(4)
        or training.get("status") != "complete"
        or int(training["args"]["seed"]) != 0
    ):
        raise ValueError("organic training input differs")
    if sha256(prepared_path) != training["prepared_data"]["sha256"]:
        raise ValueError("organic prepared input differs from training")
    plan, metadata = load_prepared_kuairand_plan(prepared_path)
    validate_long_context_plan(plan, metadata, 4)
    cfg = HSTUConfig(**training["model"])
    mismatches = {
        name: {"expected": expected, "actual": getattr(cfg, name)}
        for name, expected in EXPECTED_MODEL.items()
        if getattr(cfg, name) != expected
    }
    if mismatches:
        raise ValueError(f"organic model shape differs: {mismatches}")
    checkpoints = []
    for version in range(12):
        path = Path(checkpoint_dir) / f"theta_{version}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints.append(
            {
                "version": f"theta{version}",
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = build_manifest(plan, metadata, training)
    return plan, metadata, training, cfg, manifest, checkpoints


def direct_program_path(
    runtime_dir: str | Path,
    source: int,
    target: int,
) -> Path:
    return Path(runtime_dir) / (
        f"theta{source}_to_theta{target}_direct_oldkv_fp16.pt"
    )
