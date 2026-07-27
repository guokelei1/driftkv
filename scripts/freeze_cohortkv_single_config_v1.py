from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.streaming import (
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)

PROTOCOL = "cohortkv_single_config_full_chain_development_v1"
WORKLOAD_PROTOCOL = "cohortkv_single_config_workload_v1"
SCHEMA_VERSION = 1
FROZEN_DATE = "2026-07-27"
PREPARED_PATH = Path(
    "data/processed/kuairand_long_context_4plus12_exploration_v1.npz"
)
TRAINING_RESULT_PATH = Path(
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
VERIFIED_RESULT_PATH = Path(
    "results/motivation_scale/"
    "long_context_4plus12_verified_compiler_seed0.json"
)
CHECKPOINT_DIR = Path(
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
)
PROGRAM_DIR = CHECKPOINT_DIR / "attention_weighted_search"
VERIFIED_PLAN_DIR = CHECKPOINT_DIR / "verified_plans"
OUTPUT_DIR = Path("configs/cohortkv_single_config_v1")
WORKLOAD_NAME = "workload_manifest.json"
BLUEPRINT_NAME = "blueprint.json"
RESULT_SCHEMA_NAME = "result.schema.json"
FREEZE_SCRIPT_PATH = Path("scripts/freeze_cohortkv_single_config_v1.py")
STAGE1_SUMMARY_PATH = OUTPUT_DIR / "stage1_frontier_summary.json"
STAGE2_SUMMARY_PATH = OUTPUT_DIR / "stage2_compiler_summary.json"
STAGE3_SUMMARY_PATH = OUTPUT_DIR / "stage3_operator_summary.json"
STAGE4_SUMMARY_PATH = OUTPUT_DIR / "stage4_system_summary.json"
STAGE45_SUMMARY_PATH = OUTPUT_DIR / "stage4_5_source_plan_summary.json"
STAGE4_SOURCE_MANIFEST_PATH = (
    CHECKPOINT_DIR
    / "single_config_v1"
    / "source_shards"
    / "source_manifest.json"
)
SOURCE_VERSIONS = (0, 4, 10)
SOURCE_WEIGHTS = (0.2, 0.3, 0.5)
TARGET_VERSION = 11
TRAINING_SEED = 0
ROLE_SPLIT_SEED = 9151
CERTIFICATE_SPLIT_SEED = 27183
SOURCE_ASSIGNMENT_SEED = 58211
RUNTIME_CANDIDATE_ORDER_SEED = 73421
EXPECTED_RECORDS = 682
EXPECTED_PREFIX_TOKENS = 1_087_785
EXPECTED_EVAL_DATE = "20220423"
EXPECTED_MODEL = {
    "hidden_size": 512,
    "num_layers": 16,
    "num_heads": 8,
    "head_dim": 64,
    "max_seq_len": 2048,
    "num_prediction_items": 50000,
}
EXPECTED_LOGICAL_TARGET_BYTES_FP16 = (
    EXPECTED_PREFIX_TOKENS
    * EXPECTED_MODEL["num_layers"]
    * 2
    * EXPECTED_MODEL["num_heads"]
    * EXPECTED_MODEL["head_dim"]
    * 2
)
EXPECTED_LOGICAL_CAPSULE_BYTES_FP16 = (
    EXPECTED_PREFIX_TOKENS
    * EXPECTED_MODEL["num_layers"]
    * EXPECTED_MODEL["hidden_size"]
    * 2
)
EXPECTED_LOGICAL_CAPSULE_DATA_BYTES_INT8 = (
    EXPECTED_PREFIX_TOKENS
    * EXPECTED_MODEL["num_layers"]
    * EXPECTED_MODEL["hidden_size"]
)
PRIMARY_METHODS = ("compiled", "selective_contiguous", "exact")
DESTINATIONS = ("hbm", "dram")
GPU_COUNTS = (1, 2, 4)
RESIDUAL_DEPTHS = (4, 8)
SELECTIVE_WIDTHS = (2, 4, 6, 8, 12)
STAGE1_PROFILED_SELECTIVE_ACTION = {
    "m": 12,
    "start_layer": 0,
    "end_layer": 11,
}
EXPECTED_SELECTIVE_INTERVALS = sum(
    EXPECTED_MODEL["num_layers"] - width + 1
    for width in SELECTIVE_WIDTHS
)
EXPECTED_FRONTIER_POINTS = len(SOURCE_VERSIONS) * (
    EXPECTED_SELECTIVE_INTERVALS + len(RESIDUAL_DEPTHS) + 4
)
FAILURE_INJECTIONS = (
    "artifact_hash_mismatch_before_begin",
    "semantic_theta4_program_perturbation",
    "before_first_extent",
    "mid_wave",
    "during_publication",
    "pre_commit_after_complete_coverage",
)
EXPECTED_GPU_NAME = "NVIDIA A40"
EXPECTED_GPU_MEMORY_BYTES = 47_699_722_240
MINIMUM_SOURCE_FREE_BYTES = 128 * 1024**3
TRANSPORT_ATOL = 2e-2
TRANSPORT_RTOL = 2e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(repo_root: Path, path: Path) -> dict[str, Any]:
    resolved = repo_root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"frozen input artifact is missing: {path}")
    return {
        "path": str(path),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def split_user_roles(
    ordered_user_ids: list[int],
    seed: int = TRAINING_SEED,
) -> dict[int, str]:
    if len(ordered_user_ids) != len(set(ordered_user_ids)):
        raise ValueError("role split requires unique users")
    if len(ordered_user_ids) <= 160:
        raise ValueError("role split must leave final-test users")
    first_order = np.random.default_rng(
        ROLE_SPLIT_SEED + seed
    ).permutation(len(ordered_user_ids))
    fit = [ordered_user_ids[index] for index in first_order[:40]]
    selection = [ordered_user_ids[index] for index in first_order[40:100]]
    remaining = [
        ordered_user_ids[index] for index in first_order[100:]
    ]
    second_order = np.random.default_rng(
        CERTIFICATE_SPLIT_SEED + seed
    ).permutation(len(remaining))
    certificate = [remaining[index] for index in second_order[:60]]
    final_test = [remaining[index] for index in second_order[60:]]
    groups = {
        "fit": fit,
        "program_selection": selection,
        "certificate": certificate,
        "final_test": final_test,
    }
    assignment = {
        user_id: role
        for role, user_ids in groups.items()
        for user_id in user_ids
    }
    if len(assignment) != len(ordered_user_ids):
        raise RuntimeError("role split does not cover every user exactly once")
    return assignment


def fixed_count_assignment(
    count: int,
    versions: tuple[int, ...] = SOURCE_VERSIONS,
    weights: tuple[float, ...] = SOURCE_WEIGHTS,
    seed: int = SOURCE_ASSIGNMENT_SEED,
) -> tuple[list[int], dict[int, int]]:
    if count < 1:
        raise ValueError("source assignment requires at least one record")
    if len(versions) != len(weights) or not versions:
        raise ValueError("source versions and weights must align")
    if any(weight < 0 for weight in weights):
        raise ValueError("source weights must be nonnegative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("source weights must have positive sum")
    normalized = np.asarray(weights, dtype=np.float64) / total
    expected = normalized * count
    counts = np.floor(expected).astype(np.int64)
    remainder = count - int(counts.sum())
    order = sorted(
        range(len(versions)),
        key=lambda index: (-(expected[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    assignments = [
        version
        for version, cohort_count in zip(versions, counts, strict=True)
        for _ in range(int(cohort_count))
    ]
    np.random.default_rng(seed).shuffle(assignments)
    return assignments, {
        version: int(value)
        for version, value in zip(versions, counts, strict=True)
    }


def validate_training(training: dict[str, Any]) -> None:
    if training.get("protocol") != training_protocol_for_base_days(4):
        raise ValueError("training protocol does not match frozen 4+12 chain")
    if training.get("status") != "complete":
        raise ValueError("training result is incomplete")
    if int(training["args"]["seed"]) != TRAINING_SEED:
        raise ValueError("training seed differs from the frozen seed")
    mismatches = {
        key: {"expected": expected, "actual": training["model"].get(key)}
        for key, expected in EXPECTED_MODEL.items()
        if training["model"].get(key) != expected
    }
    if mismatches:
        raise ValueError(f"model configuration differs: {mismatches}")


def build_workload(
    repo_root: Path,
    training: dict[str, Any],
) -> dict[str, Any]:
    plan, metadata = load_prepared_kuairand_plan(repo_root / PREPARED_PATH)
    validate_long_context_plan(plan, metadata, 4)
    eval_date, samples = reconstruct_online_eval_samples(
        plan,
        (TARGET_VERSION,),
        1000,
    )[TARGET_VERSION]
    if eval_date != EXPECTED_EVAL_DATE:
        raise ValueError("evaluation date differs from the frozen endpoint")
    if len(samples) != EXPECTED_RECORDS:
        raise ValueError("eligible record count differs from the frozen workload")
    ordered_user_ids = [
        int(sample["history"]["user_id"]) for sample in samples
    ]
    roles = split_user_roles(ordered_user_ids)
    by_user = {
        int(sample["history"]["user_id"]): sample for sample in samples
    }
    canonical_user_ids = sorted(by_user)
    raw_user_id_by_index = {
        int(user_index): int(raw_user_id)
        for raw_user_id, user_index in plan.trace.user_map.items()
    }
    if len(raw_user_id_by_index) != plan.num_users:
        raise ValueError("prepared user map is not one-to-one")
    if any(user_id not in raw_user_id_by_index for user_id in canonical_user_ids):
        raise ValueError("eligible internal user index is absent from the raw user map")
    assignments, counts = fixed_count_assignment(len(canonical_user_ids))
    records = []
    for record_id, (user_id, source_version) in enumerate(
        zip(canonical_user_ids, assignments, strict=True)
    ):
        history = by_user[user_id]["history"]
        history_length = len(history["item_ids"])
        prefix_tokens = history_length - 1
        if prefix_tokens < 1:
            raise ValueError("every update record must have a nonempty prefix")
        records.append(
            {
                "record_id": record_id,
                "user_id": user_id,
                "raw_user_id": raw_user_id_by_index[user_id],
                "evaluation_role": roles[user_id],
                "source_version": f"theta{source_version}",
                "target_version": f"theta{TARGET_VERSION}",
                "history_length": history_length,
                "prefix_tokens": prefix_tokens,
                "available_history_length": int(
                    history.get(
                        "available_length_before_token_cap",
                        history_length,
                    )
                ),
            }
        )
    role_counts = Counter(record["evaluation_role"] for record in records)
    source_counts = Counter(record["source_version"] for record in records)
    expected_source_counts = {
        f"theta{version}": count for version, count in counts.items()
    }
    if dict(source_counts) != expected_source_counts:
        raise RuntimeError("source-version counts differ from frozen assignment")
    prefix_tokens = sum(record["prefix_tokens"] for record in records)
    if prefix_tokens != EXPECTED_PREFIX_TOKENS:
        raise ValueError("valid prefix-token count differs from the frozen workload")
    num_layers = int(training["model"]["num_layers"])
    hidden_size = int(training["model"]["hidden_size"])
    kv_width = (
        int(training["model"]["num_heads"])
        * int(training["model"]["head_dim"])
    )
    workload = {
        "protocol": WORKLOAD_PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "frozen_date": FROZEN_DATE,
        "parent_protocol": PROTOCOL,
        "prepared_data": {
            "path": str(PREPARED_PATH),
            "sha256": training["prepared_data"]["sha256"],
        },
        "training_seed": TRAINING_SEED,
        "evaluation_endpoint": {
            "date": eval_date,
            "target_version": f"theta{TARGET_VERSION}",
            "serving_semantics": (
                "old-version prefix state plus the latest token under "
                "the current model"
            ),
            "update_tensor_scope": (
                "each action and target extent covers history[:-1]; "
                "history[-1] is the current-model latest token and is not "
                "part of the migrated K/V"
            ),
        },
        "record_identity": {
            "rule": (
                "record_id is the zero-based position after sorting the "
                "682 eligible one-based model user indices ascending"
            ),
            "user_id_semantics": (
                "user_id is the prepared model user index used by existing "
                "verified results; raw_user_id is the source-log identifier"
            ),
            "contains_recommendation_labels": False,
        },
        "evaluation_roles": {
            "assignment": (
                "seed-9151 fit/selection permutation followed by the "
                "seed-27183 certificate/final permutation"
            ),
            "counts": dict(sorted(role_counts.items())),
        },
        "source_assignment": {
            "kind": "controlled_label_free_mix",
            "organic": False,
            "rule": (
                "largest-remainder 20/30/50 counts followed by a "
                "seed-58211 shuffle over canonical record order"
            ),
            "seed": SOURCE_ASSIGNMENT_SEED,
            "versions": [f"theta{version}" for version in SOURCE_VERSIONS],
            "weights": list(SOURCE_WEIGHTS),
            "counts": expected_source_counts,
            "uses_recommendation_labels": False,
            "claim_boundary": (
                "the complete record set is real; source versions are "
                "predeclared controlled assignments, not organic cache ages"
            ),
        },
        "summary": {
            "records": len(records),
            "unique_users": len(canonical_user_ids),
            "prefix_tokens": prefix_tokens,
            "history_length": {
                "minimum": min(record["history_length"] for record in records),
                "maximum": max(record["history_length"] for record in records),
            },
            "logical_capsule_bytes_fp16": (
                prefix_tokens * num_layers * hidden_size * 2
            ),
            "logical_transition_hidden_bytes_fp16": (
                prefix_tokens * hidden_size * 2
            ),
            "logical_residual_hidden_suffix_bytes_bf16": {
                str(depth): (
                    prefix_tokens
                    * (num_layers - depth)
                    * hidden_size
                    * 2
                )
                for depth in RESIDUAL_DEPTHS
            },
            "logical_target_kv_bytes_fp16": (
                prefix_tokens * num_layers * 2 * kv_width * 2
            ),
        },
        "records": records,
    }
    if (
        workload["summary"]["logical_target_kv_bytes_fp16"]
        != EXPECTED_LOGICAL_TARGET_BYTES_FP16
    ):
        raise ValueError("logical target K/V bytes differ from the frozen workload")
    if (
        workload["summary"]["logical_capsule_bytes_fp16"]
        != EXPECTED_LOGICAL_CAPSULE_BYTES_FP16
    ):
        raise ValueError("logical capsule bytes differ from the frozen workload")
    workload["content_sha256"] = sha256_bytes(
        canonical_json_bytes(workload)
    )
    return workload


def validate_verified_plans(repo_root: Path) -> list[dict[str, Any]]:
    entries = []
    for source_version in SOURCE_VERSIONS:
        plan_path = VERIFIED_PLAN_DIR / (
            f"theta{source_version}_to_theta{TARGET_VERSION}_verified.json"
        )
        plan = json.loads((repo_root / plan_path).read_text())
        if plan.get("source_version") != f"theta{source_version}":
            raise ValueError("verified plan source version differs")
        if plan.get("target_version") != f"theta{TARGET_VERSION}":
            raise ValueError("verified plan target version differs")
        if plan.get("labels_used") is not False:
            raise ValueError("verified plan must remain label free")
        selected = next(
            action
            for action in plan["actions"]
            if action["name"] == plan["selected_action"]
        )
        program_path = Path(selected["program_path"])
        entries.append(
            {
                "source_version": f"theta{source_version}",
                "target_version": f"theta{TARGET_VERSION}",
                "selected_action": plan["selected_action"],
                "fallback_actions": plan["fallback_actions"],
                "verified_plan": artifact_entry(repo_root, plan_path),
                "selected_program": artifact_entry(repo_root, program_path),
            }
        )
    return entries


def validate_verified_result(
    verified: dict[str, Any],
    workload: dict[str, Any],
) -> None:
    if verified.get("protocol") != (
        "kuairand_long_context_4plus12_verified_compiler_v1"
    ):
        raise ValueError("verified compiler protocol differs")
    if verified.get("status") != "verified_design_complete":
        raise ValueError("verified compiler result is incomplete")
    if int(verified.get("seed", -1)) != TRAINING_SEED:
        raise ValueError("verified compiler seed differs")
    if str(verified.get("eval_date")) != EXPECTED_EVAL_DATE:
        raise ValueError("verified compiler evaluation endpoint differs")
    expected_split = {
        "all_users": EXPECTED_RECORDS,
        "fit_users": 40,
        "program_selection_users": 60,
        "certificate_users": 60,
        "final_test_users": 522,
        "certificate_split_seed": CERTIFICATE_SPLIT_SEED,
    }
    actual_split = verified.get("split", {})
    mismatches = {
        key: {"expected": expected, "actual": actual_split.get(key)}
        for key, expected in expected_split.items()
        if actual_split.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"verified compiler split differs: {mismatches}")
    role_by_user = {
        int(record["user_id"]): record["evaluation_role"]
        for record in workload["records"]
    }
    expected_certificate = {
        user_id
        for user_id, role in role_by_user.items()
        if role == "certificate"
    }
    expected_final = {
        user_id
        for user_id, role in role_by_user.items()
        if role == "final_test"
    }
    seen_pairs = set()
    for pair in verified.get("pairs", []):
        source_version = int(pair["cache_version"])
        target_version = int(pair["current_version"])
        if source_version not in SOURCE_VERSIONS:
            raise ValueError("verified compiler has an unexpected source version")
        if target_version != TARGET_VERSION:
            raise ValueError("verified compiler has an unexpected target version")
        seen_pairs.add(source_version)
        certificate = {
            int(record["user_id"])
            for record in pair["per_user_certificate"]
        }
        final_test = {
            int(record["user_id"])
            for record in pair["per_user_test"]
        }
        if certificate != expected_certificate:
            raise ValueError(
                f"theta{source_version} certificate users differ from the manifest"
            )
        if final_test != expected_final:
            raise ValueError(
                f"theta{source_version} final users differ from the manifest"
            )
    if seen_pairs != set(SOURCE_VERSIONS):
        raise ValueError("verified compiler does not cover every frozen source version")


def build_result_schema(
    workload_content_sha256: str | None = None,
) -> dict[str, Any]:
    nonnegative = {"type": "number", "minimum": 0}
    positive_integer = {"type": "integer", "minimum": 1}
    workload_hash = (
        {
            "const": workload_content_sha256,
        }
        if workload_content_sha256 is not None
        else {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        }
    )
    timing = {
        "type": "object",
        "additionalProperties": True,
        "required": [
            "samples_seconds",
            "median_seconds",
            "breakdown_seconds",
        ],
        "properties": {
            "samples_seconds": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": nonnegative,
            },
            "median_seconds": nonnegative,
            "breakdown_seconds": {
                "type": "object",
                "required": [
                    "source_read",
                    "h2d",
                    "compute",
                    "d2h",
                    "stage",
                    "commit",
                    "elapsed",
                ],
                "properties": {
                    key: nonnegative
                    for key in (
                        "source_read",
                        "h2d",
                        "compute",
                        "d2h",
                        "stage",
                        "commit",
                        "elapsed",
                    )
                },
                "additionalProperties": True,
            },
        },
    }
    fidelity = {
        "type": "object",
        "additionalProperties": True,
        "required": [
            "cache_recovery",
            "score_cosine",
            "top100_overlap",
        ],
        "properties": {
            "cache_recovery": {"type": "number"},
            "score_cosine": {"type": "number"},
            "top100_overlap": {"type": "number"},
        },
    }
    source_representations_by_method = {
        "compiled": ["normalized_capsule_fp16"],
        "selective_contiguous": [
            "old_kv_fp16",
            "raw_history",
        ],
        "residual_p": [
            "raw_history",
            "residual_hidden_suffix_bf16",
        ],
        "exact": ["raw_history"],
        "no_transform": ["old_kv_fp16"],
    }
    system_run = {
        "type": "object",
        "additionalProperties": True,
        "allOf": [
            {
                "if": {
                    "required": ["method"],
                    "properties": {"method": {"const": method}},
                },
                "then": {
                    "properties": {
                        "source_representations": {
                            "const": representations
                        }
                    }
                },
            }
            for method, representations in (
                source_representations_by_method.items()
            )
        ]
        + [
            {
                "if": {
                    "required": ["method"],
                    "properties": {
                        "method": {"const": "selective_contiguous"}
                    },
                },
                "then": {
                    "required": [
                        "action_configuration",
                        "certificate_passed",
                        "publishable_sync_action",
                    ],
                    "properties": {
                        "action_configuration": {
                            "const": STAGE1_PROFILED_SELECTIVE_ACTION
                        },
                        "certificate_passed": {"const": False},
                        "publishable_sync_action": {"const": False},
                    },
                },
            }
        ]
        + [
            {
                "if": {
                    "required": ["gpu_count"],
                    "properties": {"gpu_count": {"const": gpu_count}},
                },
                "then": {
                    "properties": {
                        "per_gpu": {
                            "minItems": gpu_count,
                            "maxItems": gpu_count,
                        }
                    }
                },
            }
            for gpu_count in GPU_COUNTS
        ],
        "required": [
            "method",
            "destination",
            "gpu_count",
            "source_representations",
            "input_components",
            "record_count",
            "prefix_tokens",
            "placement_policy",
            "per_gpu",
            "load_imbalance_ratio",
            "records_per_second",
            "tokens_per_second",
            "selected_runtime_config",
            "capacity_preflight",
            "timing",
            "logical_input_bytes",
            "physical_input_bytes",
            "logical_output_bytes",
            "physical_output_bytes",
            "peak_hbm_bytes",
            "peak_host_bytes",
            "peak_source_resident_bytes",
            "peak_staging_bytes",
            "peak_publication_queue_bytes",
            "manifest",
            "correctness",
        ],
        "properties": {
            "method": {
                "enum": [
                    "compiled",
                    "selective_contiguous",
                    "residual_p",
                    "exact",
                    "no_transform",
                ]
            },
            "destination": {"enum": ["hbm", "dram"]},
            "gpu_count": {"enum": [1, 2, 4]},
            "source_representations": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {
                    "enum": [
                        "normalized_capsule_fp16",
                        "old_kv_fp16",
                        "raw_history",
                        "transition_hidden_fp16",
                        "residual_hidden_suffix_bf16",
                    ]
                },
            },
            "input_components": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": [
                        "representation",
                        "logical_bytes",
                        "physical_bytes",
                    ],
                    "properties": {
                        "representation": {"type": "string"},
                        "logical_bytes": positive_integer,
                        "physical_bytes": positive_integer,
                    },
                },
            },
            "record_count": {"const": EXPECTED_RECORDS},
            "prefix_tokens": {"const": EXPECTED_PREFIX_TOKENS},
            "placement_policy": {"const": "byte_weighted_lpt"},
            "per_gpu": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "required": [
                        "index",
                        "record_count",
                        "prefix_tokens",
                        "logical_input_bytes",
                        "logical_output_bytes",
                        "elapsed_seconds",
                        "peak_hbm_bytes",
                    ],
                    "properties": {
                        "index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 3,
                        },
                        "record_count": positive_integer,
                        "prefix_tokens": positive_integer,
                        "logical_input_bytes": positive_integer,
                        "logical_output_bytes": positive_integer,
                        "elapsed_seconds": nonnegative,
                        "peak_hbm_bytes": nonnegative,
                    },
                    "additionalProperties": True,
                },
            },
            "load_imbalance_ratio": nonnegative,
            "records_per_second": nonnegative,
            "tokens_per_second": nonnegative,
            "selected_runtime_config": {"type": "object"},
            "action_configuration": {"type": "object"},
            "certificate_passed": {"type": "boolean"},
            "publishable_sync_action": {"type": "boolean"},
            "capacity_preflight": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "minimum_observed_free_hbm_bytes",
                    "required_peak_hbm_bytes",
                    "minimum_observed_available_host_bytes",
                    "required_peak_host_bytes",
                    "passed",
                ],
                "properties": {
                    "minimum_observed_free_hbm_bytes": nonnegative,
                    "required_peak_hbm_bytes": nonnegative,
                    "minimum_observed_available_host_bytes": nonnegative,
                    "required_peak_host_bytes": nonnegative,
                    "passed": {"const": True},
                },
            },
            "timing": timing,
            "logical_input_bytes": positive_integer,
            "physical_input_bytes": positive_integer,
            "logical_output_bytes": {
                "const": EXPECTED_LOGICAL_TARGET_BYTES_FP16
            },
            "physical_output_bytes": positive_integer,
            "peak_hbm_bytes": nonnegative,
            "peak_host_bytes": nonnegative,
            "peak_source_resident_bytes": nonnegative,
            "peak_staging_bytes": nonnegative,
            "peak_publication_queue_bytes": nonnegative,
            "manifest": {
                "type": "object",
                "required": [
                    "protocol",
                    "record_count",
                    "prefix_tokens",
                    "workload_content_sha256",
                    "complete",
                    "duplicate_free",
                ],
                "properties": {
                    "protocol": {
                        "const": "streamkv_destination_manifest_v1"
                    },
                    "record_count": {"const": EXPECTED_RECORDS},
                    "prefix_tokens": {"const": EXPECTED_PREFIX_TOKENS},
                    "workload_content_sha256": workload_hash,
                    "complete": {"const": True},
                    "duplicate_free": {"const": True},
                },
                "additionalProperties": True,
            },
            "correctness": {
                "type": "object",
                "required": [
                    "finite",
                    "allclose",
                    "reference_kind",
                    "atol",
                    "rtol",
                    "max_abs_error",
                    "record_order_valid",
                    "lengths_offsets_valid",
                    "valid_element_count",
                ],
                "properties": {
                    "finite": {"const": True},
                    "allclose": {"const": True},
                    "reference_kind": {
                        "const": (
                            "same selected method and numeric path resident "
                            "on the same serialized source representation"
                        )
                    },
                    "atol": {"const": TRANSPORT_ATOL},
                    "rtol": {"const": TRANSPORT_RTOL},
                    "max_abs_error": nonnegative,
                    "record_order_valid": {"const": True},
                    "lengths_offsets_valid": {"const": True},
                    "valid_element_count": {
                        "const": (
                            EXPECTED_PREFIX_TOKENS
                            * EXPECTED_MODEL["num_layers"]
                            * 2
                            * EXPECTED_MODEL["num_heads"]
                            * EXPECTED_MODEL["head_dim"]
                        )
                    },
                },
                "additionalProperties": True,
            },
        },
    }
    primary_coverage = [
        {
            "contains": {
                "type": "object",
                "required": ["method", "destination", "gpu_count"],
                "properties": {
                    "method": {"const": method},
                    "destination": {"const": destination},
                    "gpu_count": {"const": gpu_count},
                },
            },
            "minContains": 1,
            "maxContains": 1,
        }
        for method in PRIMARY_METHODS
        for destination in DESTINATIONS
        for gpu_count in GPU_COUNTS
    ]
    source_version_coverage = [
        {
            "contains": {
                "type": "object",
                "required": ["source_version"],
                "properties": {
                    "source_version": {"const": f"theta{source_version}"}
                },
            },
            "minContains": 1,
            "maxContains": 1,
        }
        for source_version in SOURCE_VERSIONS
    ]
    threshold_coverage = [
        {
            "contains": {
                "type": "object",
                "required": ["recovery_target"],
                "properties": {
                    "recovery_target": {"const": recovery_target}
                },
            },
            "minContains": 1,
            "maxContains": 1,
        }
        for recovery_target in (0.5, 0.6, 0.7, 0.8, 0.9)
    ]
    failure_coverage = [
        {
            "contains": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"const": name}},
            },
            "minContains": 1,
            "maxContains": 1,
        }
        for name in FAILURE_INJECTIONS
    ]
    aborted_failure_names = [
        name
        for name in FAILURE_INJECTIONS
        if name != "semantic_theta4_program_perturbation"
    ]
    failure_outcome_conditions = [
        {
            "if": {
                "required": ["name"],
                "properties": {"name": {"enum": aborted_failure_names}},
            },
            "then": {
                "properties": {
                    "job_outcome": {"const": "aborted"},
                    "previous_current_pointer_preserved": {"const": True},
                    "complete_target_visible": {"const": False},
                    "final_manifest_complete": {"const": False},
                }
            },
        },
        {
            "if": {
                "required": ["name"],
                "properties": {
                    "name": {
                        "const": "semantic_theta4_program_perturbation"
                    }
                },
            },
            "then": {
                "required": [
                    "detection_phase",
                    "integrity_preflight_passed",
                    "perturbed_program_committed",
                    "theta4_final_action",
                ],
                "properties": {
                    "job_outcome": {
                        "const": "committed_after_escalation"
                    },
                    "detection_phase": {
                        "enum": ["semantic_preflight", "runtime_guard"]
                    },
                    "integrity_preflight_passed": {"const": True},
                    "perturbed_program_committed": {"const": False},
                    "theta4_final_action": {"const": "recompute"},
                    "previous_current_pointer_preserved": {
                        "const": False
                    },
                    "complete_target_visible": {"const": True},
                    "final_manifest_complete": {"const": True},
                },
                "allOf": [
                    {
                        "if": {
                            "properties": {
                                "detection_phase": {
                                    "const": "runtime_guard"
                                }
                            }
                        },
                        "then": {
                            "properties": {
                                "reworked_records": {
                                    "type": "integer",
                                    "minimum": 1,
                                }
                            }
                        },
                    }
                ],
            },
        },
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (
            "https://cohortkv.local/schema/"
            "cohortkv_single_config_full_chain_v1.json"
        ),
        "title": "CohortKV single-configuration integrated result",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "protocol",
            "status",
            "study_stage",
            "seed",
            "blueprint_sha256",
            "workload_content_sha256",
            "environment",
            "rq2_compiler",
            "rq3_frontier",
            "rq4_system",
            "rq4_failures",
            "rq5_economics",
            "negative_results",
        ],
        "properties": {
            "protocol": {"const": PROTOCOL},
            "status": {"const": "development_complete"},
            "study_stage": {"const": "adaptive_seed0_development"},
            "seed": {"const": TRAINING_SEED},
            "blueprint_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "workload_content_sha256": workload_hash,
            "environment": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "gpus",
                    "software",
                    "source_storage",
                    "page_cache_condition",
                ],
                "properties": {
                    "gpus": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "required": [
                                "index",
                                "name",
                                "total_memory_bytes",
                            ],
                            "properties": {
                                "index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 3,
                                },
                                "name": {"const": EXPECTED_GPU_NAME},
                                "total_memory_bytes": {
                                    "const": EXPECTED_GPU_MEMORY_BYTES
                                },
                            },
                            "additionalProperties": True,
                        },
                    },
                    "software": {
                        "type": "object",
                        "required": [
                            "python",
                            "torch",
                            "cuda_runtime",
                            "repository_commit",
                            "code_snapshot_sha256",
                        ],
                        "properties": {
                            "python": {"type": "string"},
                            "torch": {"type": "string"},
                            "cuda_runtime": {"type": "string"},
                            "repository_commit": {
                                "type": "string",
                                "pattern": "^[0-9a-f]{40}$",
                            },
                            "code_snapshot_sha256": {
                                "type": "string",
                                "pattern": "^[0-9a-f]{64}$",
                            },
                        },
                        "additionalProperties": True,
                    },
                    "source_storage": {
                        "type": "object",
                        "required": [
                            "mount",
                            "device",
                            "device_model",
                            "filesystem",
                            "free_bytes_before_materialization",
                        ],
                        "properties": {
                            "mount": {"const": "/data"},
                            "device": {"const": "/dev/nvme2n1p1"},
                            "device_model": {
                                "const": "INTEL SSDPF2KX038XZ"
                            },
                            "filesystem": {"const": "ext4"},
                            "free_bytes_before_materialization": {
                                "type": "integer",
                                "minimum": MINIMUM_SOURCE_FREE_BYTES,
                            },
                        },
                        "additionalProperties": True,
                    },
                    "page_cache_condition": {
                        "const": (
                            "one complete untimed warmup, then three measured "
                            "repetitions without explicit page-cache eviction; "
                            "source shards reopen and decode every repetition"
                        )
                    },
                },
            },
            "rq2_compiler": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "cohorts",
                    "threshold_sweep",
                    "compile_seconds",
                    "certificate_seconds",
                    "amortized_seconds_per_record",
                ],
                "properties": {
                    "cohorts": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "allOf": source_version_coverage,
                        "items": {
                            "type": "object",
                            "required": [
                                "source_version",
                                "selected_action",
                                "fallback_actions",
                                "executable_fallback_actions",
                                "program_bytes",
                                "compile_seconds",
                                "certificate_seconds",
                                "selected_cost_ratio_to_exact",
                                "fidelity",
                                "deployed_representation_certificate",
                            ],
                            "properties": {
                                "source_version": {
                                    "enum": [
                                        f"theta{version}"
                                        for version in SOURCE_VERSIONS
                                    ]
                                },
                                "selected_action": {"type": "string"},
                                "fallback_actions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "executable_fallback_actions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "program_bytes": positive_integer,
                                "compile_seconds": nonnegative,
                                "certificate_seconds": nonnegative,
                                "selected_cost_ratio_to_exact": (
                                    nonnegative
                                ),
                                "fidelity": fidelity,
                                "deployed_representation_certificate": {
                                    "type": "object",
                                    "required": [
                                        "source_dtype",
                                        "program_dtype",
                                        "output_dtype",
                                        "residual_hidden_suffix_dtype",
                                        "passed",
                                    ],
                                    "properties": {
                                        "source_dtype": {
                                            "const": "float16"
                                        },
                                        "program_dtype": {
                                            "const": "float16"
                                        },
                                        "output_dtype": {
                                            "const": "float16"
                                        },
                                        "residual_hidden_suffix_dtype": {
                                            "const": "bfloat16"
                                        },
                                        "passed": {"const": True},
                                    },
                                    "additionalProperties": True,
                                },
                            },
                            "additionalProperties": True,
                        },
                    },
                    "threshold_sweep": {
                        "type": "array",
                        "minItems": 5,
                        "maxItems": 5,
                        "allOf": threshold_coverage,
                        "items": {
                            "type": "object",
                            "required": [
                                "recovery_target",
                                "cohort_actions",
                            ],
                            "properties": {
                                "recovery_target": {
                                    "enum": [0.5, 0.6, 0.7, 0.8, 0.9]
                                },
                                "cohort_actions": {
                                    "type": "array",
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "allOf": source_version_coverage,
                                },
                            },
                            "additionalProperties": True,
                        },
                    },
                    "compile_seconds": nonnegative,
                    "certificate_seconds": nonnegative,
                    "amortized_seconds_per_record": nonnegative,
                },
            },
            "rq3_frontier": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "selection_points",
                    "selective_grid_audit",
                    "profiled_selective_actions",
                    "certified_selective_actions",
                ],
                "properties": {
                    "selection_points": {
                        "type": "array",
                        "minItems": EXPECTED_FRONTIER_POINTS,
                        "items": {
                            "type": "object",
                            "required": [
                                "source_version",
                                "target_version",
                                "evaluation_role",
                                "method",
                                "configuration",
                                "cost_ratio_to_exact",
                                "fidelity",
                            ],
                            "properties": {
                                "source_version": {
                                    "enum": [
                                        f"theta{version}"
                                        for version in SOURCE_VERSIONS
                                    ]
                                },
                                "target_version": {
                                    "const": f"theta{TARGET_VERSION}"
                                },
                                "evaluation_role": {
                                    "const": "program_selection"
                                },
                                "method": {
                                    "enum": [
                                        "compiled",
                                        "selective_contiguous",
                                        "residual_p",
                                        "cheap_projection",
                                        "reuse",
                                        "exact",
                                    ]
                                },
                                "configuration": {"type": "object"},
                                "cost_ratio_to_exact": nonnegative,
                                "fidelity": fidelity,
                            },
                            "additionalProperties": True,
                        },
                    },
                    "selective_grid_audit": {
                        "type": "array",
                        "minItems": len(SOURCE_VERSIONS),
                        "maxItems": len(SOURCE_VERSIONS),
                        "allOf": source_version_coverage,
                        "items": {
                            "type": "object",
                            "required": [
                                "source_version",
                                "expected_unique_intervals",
                                "observed_unique_intervals",
                                "complete",
                            ],
                            "properties": {
                                "source_version": {
                                    "enum": [
                                        f"theta{version}"
                                        for version in SOURCE_VERSIONS
                                    ]
                                },
                                "expected_unique_intervals": {
                                    "const": EXPECTED_SELECTIVE_INTERVALS
                                },
                                "observed_unique_intervals": {
                                    "const": EXPECTED_SELECTIVE_INTERVALS
                                },
                                "complete": {"const": True},
                            },
                            "additionalProperties": True,
                        },
                    },
                    "certified_selective_actions": {
                        "type": "array",
                        "minItems": len(SOURCE_VERSIONS),
                        "maxItems": len(SOURCE_VERSIONS),
                        "allOf": source_version_coverage,
                        "items": {
                            "type": "object",
                            "required": [
                                "source_version",
                                "action",
                                "certificate_passed",
                            ],
                            "properties": {
                                "source_version": {
                                    "enum": [
                                        f"theta{version}"
                                        for version in SOURCE_VERSIONS
                                    ]
                                },
                                "action": {
                                    "type": ["object", "null"]
                                },
                                "certificate_passed": {
                                    "type": "boolean"
                                },
                            },
                            "additionalProperties": True,
                        },
                    },
                    "profiled_selective_actions": {
                        "type": "array",
                        "minItems": len(SOURCE_VERSIONS),
                        "maxItems": len(SOURCE_VERSIONS),
                        "allOf": source_version_coverage,
                        "items": {
                            "type": "object",
                            "required": [
                                "source_version",
                                "action",
                                "certificate_passed",
                                "publishable_sync_action",
                                "system_role",
                                "source_representations",
                            ],
                            "properties": {
                                "source_version": {
                                    "enum": [
                                        f"theta{version}"
                                        for version in SOURCE_VERSIONS
                                    ]
                                },
                                "action": {
                                    "const": (
                                        STAGE1_PROFILED_SELECTIVE_ACTION
                                    )
                                },
                                "certificate_passed": {"const": False},
                                "publishable_sync_action": {
                                    "const": False
                                },
                                "system_role": {
                                    "const": (
                                        "frozen_diagnostic_external_baseline"
                                    )
                                },
                                "source_representations": {
                                    "const": [
                                        "old_kv_fp16",
                                        "raw_history",
                                    ]
                                },
                            },
                            "additionalProperties": True,
                        },
                    },
                },
            },
            "rq4_system": {
                "type": "object",
                "additionalProperties": True,
                "required": ["runs"],
                "properties": {
                    "runs": {
                        "type": "array",
                        "minItems": 18,
                        "allOf": primary_coverage,
                        "items": system_run,
                    }
                },
            },
            "rq4_failures": {
                "type": "object",
                "additionalProperties": True,
                "required": ["guard_design", "injections"],
                "properties": {
                    "guard_design": {
                        "type": "object",
                        "required": [
                            "selection_role",
                            "labels_used",
                            "selected_mechanism",
                            "reference_seconds",
                            "reference_bytes",
                            "normal_job_overhead_seconds",
                            "false_escalated_cohorts",
                            "theta4_perturbation_detected",
                            "frozen_before_failure_runs",
                        ],
                        "properties": {
                            "selection_role": {
                                "const": "program_selection"
                            },
                            "labels_used": {"const": False},
                            "selected_mechanism": {"type": "string"},
                            "reference_seconds": nonnegative,
                            "reference_bytes": nonnegative,
                            "normal_job_overhead_seconds": nonnegative,
                            "false_escalated_cohorts": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "theta4_perturbation_detected": {
                                "const": True
                            },
                            "frozen_before_failure_runs": {"const": True},
                        },
                        "additionalProperties": True,
                    },
                    "injections": {
                        "type": "array",
                        "minItems": len(FAILURE_INJECTIONS),
                        "allOf": failure_coverage,
                        "items": {
                            "type": "object",
                            "allOf": failure_outcome_conditions,
                            "required": [
                                "name",
                                "detected",
                                "job_outcome",
                                "partial_target_visible",
                                "previous_current_pointer_preserved",
                                "complete_target_visible",
                                "final_manifest_complete",
                                "staging_reclaimed",
                                "reworked_records",
                                "cleanup_seconds",
                            ],
                            "properties": {
                                "name": {"type": "string"},
                                "detected": {"const": True},
                                "job_outcome": {
                                    "enum": [
                                        "aborted",
                                        "committed_after_escalation",
                                    ]
                                },
                                "partial_target_visible": {
                                    "const": False
                                },
                                "previous_current_pointer_preserved": {
                                    "type": "boolean"
                                },
                                "complete_target_visible": {
                                    "type": "boolean"
                                },
                                "final_manifest_complete": {
                                    "type": "boolean"
                                },
                                "staging_reclaimed": {"const": True},
                                "reworked_records": {
                                    "type": "integer",
                                    "minimum": 0,
                                },
                                "cleanup_seconds": nonnegative,
                            },
                            "additionalProperties": True,
                        },
                    }
                },
            },
            "rq5_economics": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "capture",
                    "fp16_capsule",
                    "int8_capsule",
                    "auxiliary_state",
                    "break_even",
                ],
                "properties": {
                    "capture": {
                        "type": "object",
                        "required": [
                            "evaluation_role",
                            "record_count",
                            "gpu_count",
                            "forward_only_samples_seconds",
                            "forward_plus_device_capture_samples_seconds",
                            "forward_plus_persist_samples_seconds",
                            "device_capture_overhead_ratio",
                            "persist_overhead_ratio",
                        ],
                        "properties": {
                            "evaluation_role": {
                                "const": "program_selection"
                            },
                            "record_count": {"const": 60},
                            "gpu_count": {"const": 1},
                            "forward_only_samples_seconds": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": nonnegative,
                            },
                            "forward_plus_device_capture_samples_seconds": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": nonnegative,
                            },
                            "forward_plus_persist_samples_seconds": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": nonnegative,
                            },
                            "device_capture_overhead_ratio": nonnegative,
                            "persist_overhead_ratio": nonnegative,
                        },
                        "additionalProperties": True,
                    },
                    "fp16_capsule": {
                        "type": "object",
                        "required": [
                            "logical_bytes",
                            "physical_bytes",
                            "logical_ratio_to_fp16_kv",
                        ],
                        "properties": {
                            "logical_bytes": {
                                "const": EXPECTED_LOGICAL_CAPSULE_BYTES_FP16
                            },
                            "physical_bytes": positive_integer,
                            "logical_ratio_to_fp16_kv": {"const": 0.5},
                        },
                        "additionalProperties": True,
                    },
                    "int8_capsule": {
                        "type": "object",
                        "required": [
                            "layout",
                            "logical_data_bytes",
                            "scale_metadata_bytes",
                            "physical_bytes",
                            "logical_data_ratio_to_fp16_kv",
                            "dequantization_seconds",
                            "fidelity",
                            "complete_hbm_run",
                        ],
                        "properties": {
                            "layout": {
                                "const": (
                                    "symmetric signed int8 per record and "
                                    "layer with float32 absmax scale"
                                )
                            },
                            "logical_data_bytes": {
                                "const": (
                                    EXPECTED_LOGICAL_CAPSULE_DATA_BYTES_INT8
                                )
                            },
                            "scale_metadata_bytes": positive_integer,
                            "physical_bytes": positive_integer,
                            "logical_data_ratio_to_fp16_kv": {
                                "const": 0.25
                            },
                            "dequantization_seconds": nonnegative,
                            "fidelity": fidelity,
                            "complete_hbm_run": {
                                "type": "object",
                                "required": [
                                    "record_count",
                                    "destination",
                                    "gpu_count",
                                    "timing",
                                ],
                                "properties": {
                                    "record_count": {
                                        "const": EXPECTED_RECORDS
                                    },
                                    "destination": {"const": "hbm"},
                                    "gpu_count": {"const": 1},
                                    "timing": timing,
                                },
                                "additionalProperties": True,
                            },
                        },
                        "additionalProperties": True,
                    },
                    "auxiliary_state": {
                        "type": "object",
                        "required": [
                            "transition_hidden_fp16_bytes",
                            "residual_p4_hidden_suffix_bf16_bytes",
                            "residual_p8_hidden_suffix_bf16_bytes",
                            "current_verified_p8_fallback_bf16_bytes",
                        ],
                        "properties": {
                            "transition_hidden_fp16_bytes": {
                                "const": (
                                    EXPECTED_PREFIX_TOKENS
                                    * EXPECTED_MODEL["hidden_size"]
                                    * 2
                                )
                            },
                            "residual_p4_hidden_suffix_bf16_bytes": {
                                "const": (
                                    EXPECTED_PREFIX_TOKENS
                                    * (
                                        EXPECTED_MODEL["num_layers"]
                                        - 4
                                    )
                                    * EXPECTED_MODEL["hidden_size"]
                                    * 2
                                )
                            },
                            "residual_p8_hidden_suffix_bf16_bytes": {
                                "const": (
                                    EXPECTED_PREFIX_TOKENS
                                    * (
                                        EXPECTED_MODEL["num_layers"]
                                        - 8
                                    )
                                    * EXPECTED_MODEL["hidden_size"]
                                    * 2
                                )
                            },
                            "current_verified_p8_fallback_bf16_bytes": {
                                "const": 6_255_345_664
                            },
                        },
                        "additionalProperties": True,
                    },
                    "break_even": {
                        "type": "object",
                        "required": [
                            "formula",
                            "capture_seconds_per_record",
                            "exact_seconds_per_record",
                            "compiled_seconds_per_record",
                            "compiler_amortized_seconds_per_record",
                            "denominator_seconds_per_migration",
                            "minimum_migrations",
                            "conclusion",
                        ],
                        "properties": {
                            "formula": {
                                "const": (
                                    "ceil(capture_overhead / "
                                    "(exact - compiled - "
                                    "compiler_amortized))"
                                )
                            },
                            "capture_seconds_per_record": nonnegative,
                            "exact_seconds_per_record": nonnegative,
                            "compiled_seconds_per_record": nonnegative,
                            "compiler_amortized_seconds_per_record": (
                                nonnegative
                            ),
                            "denominator_seconds_per_migration": {
                                "type": "number"
                            },
                            "minimum_migrations": {
                                "type": ["integer", "null"],
                                "minimum": 1,
                            },
                            "conclusion": {
                                "enum": [
                                    "finite_break_even",
                                    "no_time_break_even",
                                ]
                            },
                        },
                        "additionalProperties": True,
                    },
                },
            },
            "negative_results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["slot", "observation", "decision"],
                    "properties": {
                        "slot": {"type": "string"},
                        "observation": {"type": "string"},
                        "decision": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
        },
    }


def build_blueprint(
    repo_root: Path,
    training: dict[str, Any],
    workload: dict[str, Any],
    stage1_summary: dict[str, Any],
    stage2_summary: dict[str, Any],
    stage3_summary: dict[str, Any],
    stage4_summary: dict[str, Any],
    stage45_summary: dict[str, Any],
    workload_file_sha256: str,
    schema_file_sha256: str,
) -> dict[str, Any]:
    checkpoint_artifacts = [
        {
            "version": f"theta{version}",
            **artifact_entry(
                repo_root,
                CHECKPOINT_DIR / f"theta_{version}.pt",
            ),
        }
        for version in (*SOURCE_VERSIONS, TARGET_VERSION)
    ]
    logical = workload["summary"]
    current_p8_fallback_tokens = sum(
        int(record["prefix_tokens"])
        for record in workload["records"]
        if record["source_version"] in {"theta0", "theta10"}
    )
    current_p8_fallback_bytes = (
        current_p8_fallback_tokens
        * (int(training["model"]["num_layers"]) - 8)
        * int(training["model"]["hidden_size"])
        * 2
    )
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "stage4_core_frozen",
        "frozen_date": FROZEN_DATE,
        "study_stage": "adaptive_seed0_development",
        "scope": {
            "objective": (
                "close one complete compiler-to-manifest vertical slice "
                "before new seeds, datasets, or model capacities"
            ),
            "creates_performance_evidence": True,
            "completed_stages": [0, 1, 2, 3, 4],
            "completed_amendments": ["stage4_5_source_plan"],
            "performance_evidence_boundary": (
                "Stages 1-4 contribute adaptive seed-0 resident frontier, "
                "deployed-representation compiler, common-layout resident "
                "operator evidence, and complete 682-record HBM/DRAM "
                "normal-path results; Stage 4.5 freezes the direct existing-"
                "old-K/V HBM source plan at 1/2/4 GPUs with zero extra "
                "per-record source state; automatic guard/fallback and "
                "failure recovery remain open"
            ),
            "confirmation_unit": "future independent training seed",
            "implementation_rule": (
                "preserve each module's logical contract, but pause and "
                "replace a mechanism when measurements falsify it"
            ),
        },
        "data_and_model": {
            "dataset": "KuaiRand-1K standard logs",
            "split": "4+12",
            "base_dates": training["prepared_data"]["metadata"]["base_dates"],
            "update_dates": training["prepared_data"]["metadata"][
                "online_dates"
            ],
            "endpoint_date": EXPECTED_EVAL_DATE,
            "training_seed": TRAINING_SEED,
            "source_versions": [
                f"theta{version}" for version in SOURCE_VERSIONS
            ],
            "target_version": f"theta{TARGET_VERSION}",
            "model": training["model"],
        },
        "frozen_inputs": {
            "freeze_script": artifact_entry(repo_root, FREEZE_SCRIPT_PATH),
            "prepared_data": artifact_entry(repo_root, PREPARED_PATH),
            "training_result": artifact_entry(
                repo_root,
                TRAINING_RESULT_PATH,
            ),
            "verified_compiler_result": artifact_entry(
                repo_root,
                VERIFIED_RESULT_PATH,
            ),
            "stage1_frontier_summary": artifact_entry(
                repo_root,
                STAGE1_SUMMARY_PATH,
            ),
            "stage2_compiler_summary": artifact_entry(
                repo_root,
                STAGE2_SUMMARY_PATH,
            ),
            "stage3_operator_summary": artifact_entry(
                repo_root,
                STAGE3_SUMMARY_PATH,
            ),
            "stage4_system_summary": artifact_entry(
                repo_root,
                STAGE4_SUMMARY_PATH,
            ),
            "stage4_5_source_plan_summary": artifact_entry(
                repo_root,
                STAGE45_SUMMARY_PATH,
            ),
            "stage4_source_manifest": artifact_entry(
                repo_root,
                STAGE4_SOURCE_MANIFEST_PATH,
            ),
            "checkpoints": checkpoint_artifacts,
            "verified_programs": validate_verified_plans(repo_root),
            "workload_manifest": {
                "path": str(OUTPUT_DIR / WORKLOAD_NAME),
                "file_sha256": workload_file_sha256,
                "content_sha256": workload["content_sha256"],
            },
            "result_schema": {
                "path": str(OUTPUT_DIR / RESULT_SCHEMA_NAME),
                "file_sha256": schema_file_sha256,
            },
        },
        "source_plan_contract": {
            "protocol": stage45_summary["protocol"],
            "normal_action": stage45_summary["source_plan"][
                "normal_action"
            ],
            "source_representation": stage45_summary["source_plan"][
                "source_representation"
            ],
            "additional_normx_bytes": stage45_summary["source_plan"][
                "additional_normx_bytes"
            ],
            "fallback_action": stage45_summary["source_plan"][
                "fallback_action"
            ],
            "declared_operating_regime": stage45_summary[
                "declared_operating_regime"
            ],
            "stage5_admitted": stage45_summary["gate"]["stage5_admitted"],
        },
        "role_contract": {
            "fit": {
                "records": 40,
                "allowed": ["fit compiled affine parameters"],
                "forbidden": [
                    "select hyperparameters",
                    "certify actions",
                    "report final quality",
                ],
            },
            "program_selection": {
                "records": 60,
                "allowed": [
                    "select compiled hyperparameters already recorded",
                    "profile selective contiguous intervals",
                    "tune each method's runtime layout",
                ],
                "forbidden": [
                    "fit affine parameters",
                    "change the primary certificate",
                    "report final quality",
                ],
            },
            "certificate": {
                "records": 60,
                "allowed": [
                    "apply the frozen label-free fidelity contract",
                    "select the minimum-cost passing action from a "
                    "predeclared library",
                ],
                "forbidden": [
                    "change candidate grids",
                    "use recommendation labels",
                ],
            },
            "final_test": {
                "records": 522,
                "allowed": [
                    "report final semantic and recommendation behavior"
                ],
                "forbidden": [
                    "fit, tune, select, or change any method or layout"
                ],
            },
            "system_workload": {
                "records": EXPECTED_RECORDS,
                "relationship": (
                    "the destination job processes all four roles after "
                    "every algorithm and layout is frozen; it never reads "
                    "recommendation labels"
                ),
            },
        },
        "workload_contract": {
            "record_count": EXPECTED_RECORDS,
            "prefix_tokens": logical["prefix_tokens"],
            "tensor_scope": (
                "all source state and target K/V cover history[:-1] only; "
                "history[-1] remains the current-model query token"
            ),
            "source_assignment": workload["source_assignment"],
            "logical_capsule_bytes_fp16": logical[
                "logical_capsule_bytes_fp16"
            ],
            "logical_transition_hidden_bytes_fp16": logical[
                "logical_transition_hidden_bytes_fp16"
            ],
            "logical_residual_hidden_suffix_bytes_bf16": logical[
                "logical_residual_hidden_suffix_bytes_bf16"
            ],
            "logical_target_kv_bytes_fp16": logical[
                "logical_target_kv_bytes_fp16"
            ],
        },
        "source_contract": {
            "common_physical_tier": {
                "kind": "buffered_posix_files",
                "mount": "/data",
                "device": "/dev/nvme2n1p1",
                "device_model": "INTEL SSDPF2KX038XZ",
                "filesystem": "ext4",
                "minimum_free_bytes_before_materialization": (
                    MINIMUM_SOURCE_FREE_BYTES
                ),
                "preflight": (
                    "resolve shard_root with findmnt, verify device, model, "
                    "filesystem, and free-space floor, and record observations "
                    "in the result before creating shards"
                ),
                "cache_condition": (
                    "one complete untimed warmup, then three measured "
                    "repetitions without explicit page-cache eviction"
                ),
                "timed_read_included": True,
                "claim_boundary": (
                    "this is a common source tier, not identical source "
                    "bytes and not a cold-device SSD benchmark"
                ),
            },
            "shard_root": str(
                CHECKPOINT_DIR / "single_config_v1/source_shards"
            ),
            "representations": {
                "normalized_capsule_fp16": {
                    "layout": "layer-major unpadded [L,T,H] plus offsets",
                    "logical_bytes": logical[
                        "logical_capsule_bytes_fp16"
                    ],
                },
                "old_kv_fp16": {
                    "layout": (
                        "separate contiguous unpadded "
                        "[L,T,Dkv] K and V plus offsets"
                    ),
                    "logical_bytes": logical[
                        "logical_target_kv_bytes_fp16"
                    ],
                },
                "raw_history": {
                    "fields": [
                        "item_ids",
                        "behaviors",
                        "time_deltas",
                        "lengths",
                    ],
                    "physical_bytes_must_be_measured": True,
                },
                "transition_hidden_fp16": {
                    "layout": (
                        "one selected old-version pre-block hidden state "
                        "[T,H] per record"
                    ),
                    "logical_bytes_for_one_transition_per_record": logical[
                        "logical_transition_hidden_bytes_fp16"
                    ],
                    "physical_bytes_must_be_measured": True,
                },
                "residual_hidden_suffix_bf16": {
                    "layout": (
                        "old-version pre-block hidden states "
                        "[layers p..L-1,T,H] plus offsets"
                    ),
                    "logical_bytes_by_p_for_all_records": logical[
                        "logical_residual_hidden_suffix_bytes_bf16"
                    ],
                    "current_verified_p8_fallback_bytes": (
                        current_p8_fallback_bytes
                    ),
                    "current_verified_p8_fallback_scope": [
                        "theta0",
                        "theta10",
                    ],
                    "physical_bytes_must_be_measured": True,
                    "not_part_of_default_normalized_capsule": True,
                },
            },
            "source_shard_creation": {
                "outside_job_completion": True,
                "reported_separately": True,
                "temporary_and_final_bytes_must_fit_preflight": True,
            },
        },
        "action_contracts": {
            "reuse": {
                "role": "zero-maintenance semantic anchor",
                "publishable_sync_action": False,
                "inputs": ["old_kv_fp16"],
            },
            "cheap_projection": {
                "role": "internal compiler ablation",
                "inputs": ["normalized_capsule_fp16"],
            },
            "compiled": {
                "role": "primary candidate",
                "inputs": ["normalized_capsule_fp16"],
                "program_scope": "one immutable program per source/target pair",
                "primary_contract": {
                    "recovery_target": 0.7,
                    "minimum_coverage": 0.8,
                    "one_sided_confidence": 0.9,
                    "max_cost_ratio_to_exact": 0.3,
                    "views": [
                        "relative_kv_error",
                        "fresh_score_cosine",
                        "fresh_top100_overlap",
                    ],
                },
                "threshold_sweep": [0.5, 0.6, 0.7, 0.8, 0.9],
                "stage2_observation": {
                    "selected_action_by_source": {
                        pair["source_version"]: pair["selected_action"]
                        for pair in stage2_summary["pairs"]
                    },
                    "fallback_actions_by_source": {
                        pair["source_version"]: pair["fallback_actions"]
                        for pair in stage2_summary["pairs"]
                    },
                    "deployed_certificate_passed": True,
                    "final_test_evaluated": False,
                    "summary_protocol": stage2_summary["protocol"],
                },
            },
            "selective_contiguous": {
                "role": "DroidSpeak-adapted external baseline",
                "candidate_inputs": {
                    "start_layer_zero": [
                        "old_kv_fp16",
                        "raw_history",
                    ],
                    "start_layer_positive": [
                        "old_kv_fp16",
                        "transition_hidden_fp16",
                        "raw_history",
                    ],
                },
                "semantics": (
                    "recompute one contiguous current-model interval and "
                    "reuse old K/V outside it"
                ),
                "terminal_semantics": (
                    "execute full current blocks before the terminal layer; "
                    "at the terminal layer execute only current Norm and K/V "
                    "projection because no later current hidden state is used"
                ),
                "m_values": list(SELECTIVE_WIDTHS),
                "candidate_intervals_by_m": {
                    str(width): [
                        [start, start + width - 1]
                        for start in range(
                            int(training["model"]["num_layers"]) - width + 1
                        )
                    ]
                    for width in SELECTIVE_WIDTHS
                },
                "selection_rule": (
                    "on program-selection users, choose the interval for "
                    "each m with the best worst-view label-free recovery; "
                    "break ties by lower measured GPU cost then earlier start"
                ),
                "certificate_rule": (
                    "on certificate users, publish the minimum-cost frozen "
                    "m/interval that passes the same primary contract; exact "
                    "is the terminal fallback"
                ),
                "stage1_observation": {
                    "all_source_pairs_certificate_passed": False,
                    "publishable_fallback": "exact",
                    "profiled_system_action": (
                        STAGE1_PROFILED_SELECTIVE_ACTION
                    ),
                    "profiled_system_action_role": (
                        "frozen_diagnostic_external_baseline"
                    ),
                    "profiled_system_action_inputs": [
                        "old_kv_fp16",
                        "raw_history",
                    ],
                    "transition_hidden_bytes": 0,
                    "claim_boundary": (
                        "the Stage-4 selective_contiguous row must retain "
                        "certificate_passed=false and cannot be called a "
                        "certified or publishable synchronized target"
                    ),
                    "summary_protocol": stage1_summary["protocol"],
                },
                "adaptation_boundary": (
                    "DroidSpeak profiles contiguous recomputation groups "
                    "with task quality and transition E-cache; this HSTU "
                    "adaptation replaces task labels with the common "
                    "label-free semantic contract"
                ),
                "profiling_materialization": (
                    "candidate transition states may be captured only for "
                    "program-selection records outside timed system jobs; "
                    "final source shards retain one frozen transition per "
                    "source-target cohort"
                ),
                "implementation_guard": {
                    "existing_migrate_contiguous_cache_is_compatible": False,
                    "reason": (
                        "the existing helper applies current K/V projections "
                        "to old normalized states outside the interval instead "
                        "of reusing source old K/V"
                    ),
                    "required_reference_tests": [
                        "outside-interval output equals source old K/V",
                        "inside-interval output matches minimal current replay",
                        "full-depth interval matches exact current K/V",
                    ],
                },
            },
            "residual_p": {
                "role": "internal escalation tier",
                "inputs": [
                    "raw_history",
                    "residual_hidden_suffix_bf16",
                ],
                "p_values": [4, 8],
                "state_semantics": (
                    "depth p requires every old pre-block hidden state from "
                    "layer p through L-1; one transition state and the "
                    "normalized capsule cannot reconstruct this action"
                ),
                "storage_policy": (
                    "BF16 auxiliary hidden suffix bytes are counted separately "
                    "from the default capsule; if the shard is absent, remove "
                    "residual-p from the executable fallback chain under a "
                    "revised protocol and fall through to exact"
                ),
                "current_verified_fallback": {
                    "p": 8,
                    "source_versions": ["theta0", "theta10"],
                    "logical_bytes": current_p8_fallback_bytes,
                },
                "implementation_guard": {
                    "existing_operator_semantics_compatible": True,
                    "existing_required_state_label_is_too_coarse": True,
                    "required_reference_tests": [
                        "p-specific hidden suffix is sufficient",
                        "p=L matches exact current K/V",
                        "missing hidden suffix rejects before job begin",
                    ],
                },
            },
            "exact": {
                "role": "current-model K/V semantic reference",
                "inputs": ["raw_history"],
                "compute_candidates": ["bfloat16", "float32"],
                "published_dtype": "float16",
            },
            "no_transform": {
                "role": "pure placement and transaction floor",
                "inputs": ["old_kv_fp16"],
                "semantic_contract_expected_to_pass": False,
            },
        },
        "numeric_contract": {
            "semantic_oracle": (
                "FP32 current-model exact K/V and full-catalog score views"
            ),
            "deployment_recertification": {
                "status": "complete",
                "primary_capsule_dtype": "float16",
                "program_dtype": "float16",
                "output_dtype": "float16",
                "residual_hidden_suffix_dtype": "bfloat16",
                "summary_protocol": stage2_summary["protocol"],
                "final_test_evaluated": False,
            },
            "existing_verified_result_boundary": (
                "the frozen verified compiler evaluated in-memory FP32 "
                "layerwise state; it is algorithm evidence, not a certificate "
                "for the deployed FP16 source representation"
            ),
            "transport_oracle": (
                "the same selected method and numeric path executed resident "
                "on the same serialized source representation"
            ),
            "transport_allclose": {
                "atol": TRANSPORT_ATOL,
                "rtol": TRANSPORT_RTOL,
                "finite_required": True,
            },
            "method_specific_guards": {
                "selective_contiguous": (
                    "outside-interval FP16 values must equal source old K/V "
                    "exactly before the common transport tolerance is applied"
                ),
                "exact_bfloat16": (
                    "selection users must compare BF16-compute published "
                    "FP16 K/V with FP32-compute published FP16 K/V"
                ),
            },
        },
        "frontier_contract": {
            "source_target_pairs": [
                {
                    "source_version": f"theta{source_version}",
                    "target_version": f"theta{TARGET_VERSION}",
                }
                for source_version in SOURCE_VERSIONS
            ],
            "selection_points_per_pair": (
                EXPECTED_SELECTIVE_INTERVALS + len(RESIDUAL_DEPTHS) + 4
            ),
            "total_minimum_selection_points": EXPECTED_FRONTIER_POINTS,
            "families_per_pair": {
                "selective_contiguous": EXPECTED_SELECTIVE_INTERVALS,
                "residual_p": len(RESIDUAL_DEPTHS),
                "compiled": 1,
                "cheap_projection": 1,
                "reuse": 1,
                "exact": 1,
            },
            "grid_audit": (
                "the aggregator verifies every declared interval and anchor "
                "configuration exactly once for each source-target pair"
            ),
            "certification_scope": (
                "one frozen selective candidate per m and source-target pair "
                "may proceed from program selection to certificate; the "
                "completed certificate publishes exact, while the strongest "
                "profiled selective action proceeds only to label-free "
                "diagnostic system timing"
            ),
        },
        "runtime_tuning_contract": {
            "role": "program_selection",
            "labels_used": False,
            "separate_per_method_destination_and_gpu_count": True,
            "freeze_before_complete_workload": True,
            "search_procedure": (
                "open and read the method's complete selection source once "
                "to establish the warm page-cache condition, screen every "
                f"legal candidate once after correctness in "
                f"seed-{RUNTIME_CANDIDATE_ORDER_SEED} order, "
                "then rerun the fastest three with one warmup and three "
                "measured repetitions; publish the winner before any "
                "682-record run"
            ),
            "candidate_order_seed": RUNTIME_CANDIDATE_ORDER_SEED,
            "retain_every_candidate_result": True,
            "grid": {
                "batch_size": [1, 2, 4],
                "length_bucket_width": [16, 32, 64],
                "max_inflight": [2, 3, 4],
                "compiled_operator": ["packed_fp16", "fused_fp16"],
                "exact_compute": ["bfloat16", "float32"],
            },
            "objective": (
                "minimize median complete tuning-job elapsed time after "
                "correctness; ties prefer lower peak memory then less padding"
            ),
            "warmup_runs": 1,
            "measured_repetitions": 3,
            "stage3_observation": {
                "selection_role": "program_selection",
                "default_operator": stage3_summary["selection"]["operator"],
                "default_batch_size": stage3_summary["selection"][
                    "batch_size"
                ],
                "default_bucket_width": stage3_summary["selection"][
                    "bucket_width"
                ],
                "common_output_layout": (
                    "separate contiguous unpadded FP16 [L,T,Dkv] K/V "
                    "plus lengths and offsets"
                ),
                "fused_speedup_over_fastest_packed_resident": (
                    stage3_summary["selection"]["fused_stability_gate"][
                        "fused_speedup_over_packed"
                    ]
                ),
                "stage4_retunes_every_endpoint": True,
                "labels_used": False,
                "final_test_evaluated": False,
                "summary_protocol": stage3_summary["protocol"],
            },
        },
        "destination_contract": {
            "common_output": {
                "dtype": "float16",
                "layout": (
                    "separate contiguous unpadded [L,T,Dkv] K and V "
                    "per extent with lengths and offsets"
                ),
                "allocation": (
                    "fresh target extents are allocated inside each timed "
                    "job for every method"
                ),
                "coverage": (
                    "all 682 record IDs exactly once before manifest commit"
                ),
                "valid_elements": (
                    "only unpadded prefix-token elements exist in the "
                    "published extents; internal dense padding is never "
                    "counted as logical output"
                ),
                "manifest_protocol": "streamkv_destination_manifest_v1",
            },
            "dram": {
                "publication_mode": "host_staged",
                "target_residency": "retained pinned CPU extents",
                "durable": False,
                "includes_d2h": True,
                "commit": "atomic in-memory manifest visibility",
                "capacity_preflight": {
                    "logical_target_bytes": logical[
                        "logical_target_kv_bytes_fp16"
                    ],
                    "required_peak_formula": (
                        "retained pinned target bytes plus bounded source "
                        "wave, decode, and publication-queue host bytes"
                    ),
                    "probe": (
                        "allocate and release one maximum-size pinned extent "
                        "before tuning; verify aggregate available host memory "
                        "before every complete job"
                    ),
                    "failure_policy": (
                        "stop before timing instead of silently publishing "
                        "pageable or non-retained output"
                    ),
                },
            },
            "hbm": {
                "publication_mode": "direct_device",
                "target_residency": (
                    "extent-partitioned retained tensors on the active "
                    "destination GPUs"
                ),
                "durable": False,
                "includes_d2h": False,
                "commit": "atomic in-memory manifest visibility",
                "capacity_preflight": {
                    "expected_gpu_name": EXPECTED_GPU_NAME,
                    "expected_total_bytes_per_gpu": (
                        EXPECTED_GPU_MEMORY_BYTES
                    ),
                    "one_gpu_logical_target_bytes": logical[
                        "logical_target_kv_bytes_fp16"
                    ],
                    "required_peak_formula": (
                        "assigned retained target bytes plus model/program "
                        "residency, measured maximum-batch temporary bytes, "
                        "and allocator margin"
                    ),
                    "oom_policy": (
                        "reduce batch/inflight only within the frozen grid; "
                        "if the one-GPU point remains infeasible, stop and "
                        "revise the protocol instead of dropping or changing "
                        "the destination"
                    ),
                },
            },
            "primary_matrix": {
                "destinations": list(DESTINATIONS),
                "gpu_counts": list(GPU_COUNTS),
                "methods": list(PRIMARY_METHODS),
                "selective_contiguous_semantics": {
                    "action": STAGE1_PROFILED_SELECTIVE_ACTION,
                    "certificate_passed": False,
                    "publishable_sync_action": False,
                    "system_role": (
                        "frozen_diagnostic_external_baseline"
                    ),
                },
                "required_points": [
                    {
                        "method": method,
                        "destination": destination,
                        "gpu_count": gpu_count,
                    }
                    for method in PRIMARY_METHODS
                    for destination in DESTINATIONS
                    for gpu_count in GPU_COUNTS
                ],
                "additional_controls": [
                    "residual_p",
                    "no_transform",
                ],
            },
            "stage4_observation": {
                "summary_protocol": stage4_summary["protocol"],
                "normal_path_complete": True,
                "full_points": stage4_summary["measurement_boundary"][
                    "full_points"
                ],
                "primary_points": stage4_summary["measurement_boundary"][
                    "primary_points"
                ],
                "control_points": stage4_summary["measurement_boundary"][
                    "control_points"
                ],
                "all_capacity_preflights_passed": stage4_summary["derived"][
                    "all_capacity_preflights_passed"
                ],
                "all_manifests_complete_and_duplicate_free": (
                    stage4_summary["derived"][
                        "all_manifests_complete_and_duplicate_free"
                    ]
                ),
                "automatic_guard_and_fallback_open": True,
            },
            "excluded_from_primary_matrix": [
                "filesystem_destination",
                "remote_destination",
            ],
        },
        "placement_contract": {
            "policy": "byte_weighted_lpt",
            "extent_weight": (
                "declared logical source bytes plus logical target bytes for "
                "that method and extent"
            ),
            "program_and_model_residency": (
                "replicated on every active worker and included in peak HBM, "
                "not in per-extent source bytes"
            ),
            "required_reporting": [
                "per-GPU records and prefix tokens",
                "per-GPU logical input and output bytes",
                "per-GPU elapsed and peak HBM",
                "aggregate assigned-byte imbalance",
                "records/s and tokens/s",
                "peak source, staging, and publication-queue bytes",
            ],
            "aggregate_validation": (
                "per-GPU counts and bytes must sum exactly to the aggregate "
                "run and active device indices must be unique"
            ),
        },
        "timing_contract": {
            "job_start": (
                "before opening the frozen source manifest and beginning "
                "the destination transaction"
            ),
            "job_end": "after the complete target manifest is visible",
            "included_phases": [
                "source_manifest_scan",
                "source_read",
                "decode_and_pin",
                "target_allocation",
                "h2d",
                "compute",
                "d2h_when_required",
                "stage",
                "coverage_validation",
                "manifest_commit",
                "coordinator_overhead",
            ],
            "excluded_but_reported": [
                "streaming_training",
                "checkpoint_loading",
                "source_shard_creation",
                "offline_method_and_layout_tuning",
            ],
            "compiler_accounting": (
                "fit and certificate wall time are reported separately and "
                "also amortized over the 682-record job"
            ),
            "warmup_runs": 1,
            "measured_repetitions": 3,
            "repetition_lifecycle": (
                "destroy the prior private target and synchronize before "
                "each warmup or measured job; each job creates fresh target "
                "extents, reopens and decodes source shards, and no repetition "
                "may reuse a committed target or application-decoded input"
            ),
            "statistical_boundary": (
                "timing repetitions describe stability; seed 0 is the only "
                "training replication in this development protocol"
            ),
        },
        "economics_contract": {
            "capture": {
                "role": "program_selection",
                "records": 60,
                "model_version": f"theta{TARGET_VERSION}",
                "gpu_count": 1,
                "shared_input": "the same frozen prefix histories",
                "paths": [
                    "forward materializing fresh K/V only",
                    "same forward plus FP16 normalized-state device capture",
                    "same forward plus device capture, D2H, encode, and "
                    "buffered-POSIX shard persistence",
                ],
                "warmup_runs": 1,
                "measured_repetitions": 3,
            },
            "fp16": {
                "logical_bytes": logical["logical_capsule_bytes_fp16"],
                "logical_ratio_to_fp16_kv": 0.5,
                "physical_bytes_required": True,
            },
            "int8": {
                "layout": (
                    "symmetric signed int8 per record and layer with "
                    "float32 absmax scale"
                ),
                "equation": (
                    "scale=max(abs(z))/127; q=round(clamp(z/scale,-127,127)); "
                    "all-zero tensors use scale 1"
                ),
                "logical_data_bytes": (
                    EXPECTED_LOGICAL_CAPSULE_DATA_BYTES_INT8
                ),
                "logical_data_ratio_to_fp16_kv": 0.25,
                "dequantization": (
                    "dequantize to FP16 during timed host staging without "
                    "changing the prepared compiled program"
                ),
                "semantic_evaluation": (
                    "apply the frozen certificate on certificate users and "
                    "report the frozen representation on final-test users"
                ),
                "complete_job_endpoint": {
                    "destination": "hbm",
                    "gpu_count": 1,
                    "records": EXPECTED_RECORDS,
                },
            },
            "auxiliary_state_reporting": {
                "transition_hidden_fp16": logical[
                    "logical_transition_hidden_bytes_fp16"
                ],
                "residual_hidden_suffix_bf16_by_p": logical[
                    "logical_residual_hidden_suffix_bytes_bf16"
                ],
                "current_verified_p8_fallback_bf16": (
                    current_p8_fallback_bytes
                ),
                "excluded_from_default_capsule_ratio": True,
            },
            "break_even": {
                "formula": (
                    "ceil(capture_overhead / (exact - compiled - "
                    "compiler_amortized))"
                ),
                "unit": "seconds per record",
                "if_denominator_nonpositive": "no_time_break_even",
                "report_fp16_and_int8_separately": True,
                "forbidden": (
                    "convert bytes or update frequency to money or workload "
                    "frequency without an external declared parameter"
                ),
            },
        },
        "guard_selection_contract": {
            "role": "program_selection",
            "labels_used": False,
            "logical_requirement": (
                "detect the frozen integrity-valid theta4 semantic "
                "perturbation before commit while reporting false escalation "
                "on all unperturbed source cohorts"
            ),
            "reference_accounting": [
                "reference source representation and bytes",
                "reference compute seconds",
                "normal no-fault job overhead seconds",
                "perturbed detection phase",
                "false-escalated cohort count",
            ],
            "selection_rule": (
                "choose the lowest normal-job overhead mechanism that detects "
                "the frozen perturbation and preserves unperturbed cohort "
                "certificates; if no runtime sentinel qualifies, use an "
                "executable semantic preflight and narrow the manuscript claim"
            ),
            "freeze_boundary": (
                "freeze the mechanism and parameters before the six complete "
                "failure jobs; final-test recommendation labels are forbidden"
            ),
        },
        "failure_contract": {
            "reader_state": {
                "before_commit": (
                    "the logical current pointer remains on the previously "
                    "committed version; theta11 is not visible"
                ),
                "after_commit": (
                    "one atomic pointer/manifest transition exposes theta11"
                ),
                "after_abort": (
                    "the previous version remains visible and theta11 "
                    "staging is unreachable"
                ),
            },
            "injections": [
                {
                    "name": "artifact_hash_mismatch_before_begin",
                    "expected": "abort before the first extent",
                },
                {
                    "name": "semantic_theta4_program_perturbation",
                    "expected": (
                        "the perturbation remains structurally valid and "
                        "passes integrity checks, then semantic preflight or "
                        "the runtime guard detects it; theta4 escalates to its "
                        "published exact recompute fallback and any earlier "
                        "theta4 extents are replaced before a complete commit"
                    ),
                },
                {
                    "name": "before_first_extent",
                    "expected": "abort with no theta11 visibility",
                },
                {
                    "name": "mid_wave",
                    "expected": "abort with no theta11 visibility",
                },
                {
                    "name": "during_publication",
                    "expected": "abort and reclaim private staging",
                },
                {
                    "name": "pre_commit_after_complete_coverage",
                    "expected": "abort with the previous version still visible",
                },
            ],
            "resume_claim": (
                "optional until an extent journal proves at-most-one-wave "
                "redo; absence of resume does not block atomic abort"
            ),
        },
        "result_contract": {
            "aggregate_output": (
                "results/system/cohortkv_single_config_full_chain_v1/"
                "final_summary_seed0.json"
            ),
            "schema": str(OUTPUT_DIR / RESULT_SCHEMA_NAME),
            "required_artifacts": [
                "raw per-run JSON",
                "aggregate JSON",
                "correctness report",
                "timing and memory breakdown",
                "committed destination manifests",
                "negative-results log",
                "repository commit and code-snapshot hash",
            ],
            "semantic_validation": (
                "JSON Schema enforces one run for every primary "
                "method/destination/GPU point and one record for every "
                "failure injection; the aggregator additionally verifies "
                "component-byte sums and source/action compatibility"
            ),
        },
        "paper_contract": {
            "record": (
                "experiments/system/"
                "COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md"
            ),
            "target_manuscript": (
                "paper/cohortkv/manuscript_v3_target_en.md"
            ),
            "rule": (
                "every target TBD is marked measured, deleted, downgraded, "
                "or still open before any prose is promoted"
            ),
        },
    }


def build_outputs(repo_root: Path) -> dict[Path, bytes]:
    training = json.loads((repo_root / TRAINING_RESULT_PATH).read_text())
    validate_training(training)
    prepared_hash = sha256_file(repo_root / PREPARED_PATH)
    if prepared_hash != training["prepared_data"]["sha256"]:
        raise ValueError("prepared-data hash differs from the training result")
    workload = build_workload(repo_root, training)
    workload_bytes = canonical_json_bytes(workload)
    verified = json.loads((repo_root / VERIFIED_RESULT_PATH).read_text())
    validate_verified_result(verified, workload)
    stage1_summary = json.loads(
        (repo_root / STAGE1_SUMMARY_PATH).read_text()
    )
    if (
        stage1_summary.get("protocol")
        != "cohortkv_single_config_stage1_frozen_v1"
        or stage1_summary.get("status") != "stage1_frozen"
        or len(stage1_summary.get("pairs", [])) != len(SOURCE_VERSIONS)
    ):
        raise ValueError("Stage 1 frozen summary is invalid")
    for pair in stage1_summary["pairs"]:
        action = pair["profiled_selective_action"]
        if (
            action["configuration"]
            != STAGE1_PROFILED_SELECTIVE_ACTION
            or action["source_representations"]
            != ["old_kv_fp16", "raw_history"]
            or action["publishable_sync_action"] is not False
            or pair["certificate"]["passed"] is not False
        ):
            raise ValueError("Stage 1 downstream selective action mismatch")
    stage2_summary = json.loads(
        (repo_root / STAGE2_SUMMARY_PATH).read_text()
    )
    if (
        stage2_summary.get("protocol")
        != "cohortkv_single_config_stage2_frozen_v1"
        or stage2_summary.get("status") != "stage2_frozen"
        or len(stage2_summary.get("pairs", [])) != len(SOURCE_VERSIONS)
        or stage2_summary.get("measurement_boundary", {}).get(
            "final_test_users_evaluated"
        )
        is not False
    ):
        raise ValueError("Stage 2 frozen summary is invalid")
    expected_stage2_fallbacks = {
        "theta0": ["structural_p8", "recompute"],
        "theta4": ["recompute"],
        "theta10": ["structural_p8", "recompute"],
    }
    for pair in stage2_summary["pairs"]:
        certificate = pair["selected_certificate"]
        if (
            pair["selected_action"] != "compiled_full_affine"
            or pair["fallback_actions"]
            != expected_stage2_fallbacks[pair["source_version"]]
            or certificate["certificate_passed"] is not True
            or certificate["worst_recovery_lower_bound"] < 0.7
            or certificate["worst_coverage_lower_bound"] < 0.8
        ):
            raise ValueError("Stage 2 downstream compiler decision mismatch")
    stage3_summary = json.loads(
        (repo_root / STAGE3_SUMMARY_PATH).read_text()
    )
    if (
        stage3_summary.get("protocol")
        != "cohortkv_single_config_stage3_frozen_v1"
        or stage3_summary.get("status") != "stage3_frozen"
        or stage3_summary.get("measurement_boundary", {}).get(
            "source_role"
        )
        != "program_selection"
        or stage3_summary.get("measurement_boundary", {}).get(
            "final_test_evaluated"
        )
        is not False
        or stage3_summary.get("correctness", {}).get("layouts") != 9
        or stage3_summary.get("correctness", {}).get(
            "transport_mismatched_elements"
        )
        != 0
        or stage3_summary.get("selection", {}).get("operator")
        != "fused_fp16"
        or stage3_summary.get("selection", {}).get("batch_size") != 4
        or stage3_summary.get("selection", {}).get("bucket_width") != 32
        or stage3_summary.get("selection", {}).get(
            "fused_stability_gate", {}
        ).get("all_fused_samples_below_all_packed_samples")
        is not True
    ):
        raise ValueError("Stage 3 frozen summary is invalid")
    stage4_summary = json.loads(
        (repo_root / STAGE4_SUMMARY_PATH).read_text()
    )
    stage4_runs = stage4_summary.get("runs", [])
    stage4_tuning = stage4_summary.get("runtime_tuning", {}).get(
        "points",
        [],
    )
    stage4_run_keys = {
        (
            value.get("method"),
            value.get("destination"),
            value.get("gpu_count"),
        )
        for value in stage4_runs
    }
    expected_stage4_keys = {
        (method, destination, gpu_count)
        for method in (
            "compiled",
            "selective_contiguous",
            "exact",
            "residual_p",
            "no_transform",
        )
        for destination in DESTINATIONS
        for gpu_count in GPU_COUNTS
    }
    if (
        stage4_summary.get("protocol")
        != "cohortkv_single_config_stage4_frozen_v1"
        or stage4_summary.get("status") != "stage4_frozen"
        or stage4_summary.get("measurement_boundary", {}).get(
            "full_points"
        )
        != 30
        or stage4_summary.get("measurement_boundary", {}).get(
            "primary_points"
        )
        != 18
        or stage4_summary.get("measurement_boundary", {}).get(
            "control_points"
        )
        != 12
        or len(stage4_tuning) != 30
        or len(stage4_runs) != 30
        or stage4_run_keys != expected_stage4_keys
        or stage4_summary.get("derived", {}).get(
            "all_capacity_preflights_passed"
        )
        is not True
        or stage4_summary.get("derived", {}).get(
            "all_outputs_finite_and_allclose"
        )
        is not True
        or stage4_summary.get("derived", {}).get(
            "all_manifests_complete_and_duplicate_free"
        )
        is not True
        or stage4_summary.get("source_manifest", {}).get("path")
        != str(STAGE4_SOURCE_MANIFEST_PATH)
        or stage4_summary.get("source_manifest", {}).get("sha256")
        != sha256_file(repo_root / STAGE4_SOURCE_MANIFEST_PATH)
    ):
        raise ValueError("Stage 4 frozen summary is invalid")
    stage45_summary = json.loads(
        (repo_root / STAGE45_SUMMARY_PATH).read_text()
    )
    stage45_comparisons = stage45_summary.get("system", {}).get(
        "comparisons",
        [],
    )
    if (
        stage45_summary.get("protocol")
        != "cohortkv_single_config_stage4_5_frozen_v1"
        or stage45_summary.get("status")
        != "stage4_5_source_plan_frozen"
        or stage45_summary.get("upstream", {})
        .get("stage4_summary", {})
        .get("sha256")
        != sha256_file(repo_root / STAGE4_SUMMARY_PATH)
        or stage45_summary.get("source_plan", {}).get(
            "source_representation"
        )
        != "existing_old_kv_fp16"
        or stage45_summary.get("source_plan", {}).get(
            "additional_normx_bytes"
        )
        != 0
        or stage45_summary.get("source_plan", {}).get("fallback_action")
        != "exact"
        or {
            value.get("gpu_count")
            for value in stage45_comparisons
            if value.get("completion_gate_passed") is True
        }
        != {1, 2, 4}
        or stage45_summary.get("gate", {}).get("stage5_admitted")
        is not True
    ):
        raise ValueError("Stage 4.5 frozen source plan is invalid")
    schema = build_result_schema(workload["content_sha256"])
    schema_bytes = canonical_json_bytes(schema)
    blueprint = build_blueprint(
        repo_root,
        training,
        workload,
        stage1_summary,
        stage2_summary,
        stage3_summary,
        stage4_summary,
        stage45_summary,
        sha256_bytes(workload_bytes),
        sha256_bytes(schema_bytes),
    )
    return {
        OUTPUT_DIR / WORKLOAD_NAME: workload_bytes,
        OUTPUT_DIR / RESULT_SCHEMA_NAME: schema_bytes,
        OUTPUT_DIR / BLUEPRINT_NAME: canonical_json_bytes(blueprint),
    }


def check_outputs(repo_root: Path, outputs: dict[Path, bytes]) -> None:
    mismatches = []
    for path, expected in outputs.items():
        resolved = repo_root / path
        if not resolved.is_file():
            mismatches.append(f"missing {path}")
        elif resolved.read_bytes() != expected:
            mismatches.append(f"content differs for {path}")
    if mismatches:
        raise RuntimeError("; ".join(mismatches))


def write_outputs(repo_root: Path, outputs: dict[Path, bytes]) -> None:
    for path, payload in outputs.items():
        resolved = repo_root / path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(payload)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    outputs = build_outputs(repo_root)
    if args.check:
        check_outputs(repo_root, outputs)
        status = "verified"
    else:
        write_outputs(repo_root, outputs)
        status = "frozen"
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": status,
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
