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
STAGE5_FORMAL_CANARY_PATH = (
    OUTPUT_DIR / "stage5_formal_canary.json"
)
STAGE49_SELECTED_CANDIDATE = "staggered_renewal_h12"
STAGE49_COST_ENDPOINT = "token_debt_total10"
STAGE6_PROTOCOL = "cohortkv_single_config_stage6_freeze_v1"
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
    "semantic_theta0_theta1_program_perturbation",
    "mid_job",
    "pre_commit",
)
EXPECTED_GPU_NAME = "NVIDIA A40"
EXPECTED_GPU_MEMORY_BYTES = 47_699_722_240
MINIMUM_SOURCE_FREE_BYTES = 128 * 1024**3
TRANSPORT_ATOL = 2e-2
TRANSPORT_RTOL = 2e-2
STAGE5_ACCOUNTING_FROZEN_INPUTS = {
    "stage2": {
        "protocol": "cohortkv_single_config_stage2_frozen_v1",
        "status": "stage2_frozen",
        "sha256": (
            "09461c81ad7d9a061a6aae2358e478c151befa07614f73afff972bd4b90a8126"
        ),
    },
    "stage4": {
        "protocol": "cohortkv_single_config_stage4_frozen_v1",
        "status": "stage4_frozen",
        "sha256": (
            "2c891e2fb085708bdb83c6c39410f9f7509f25697ff0e8b0a4085906ef7219b6"
        ),
    },
    "stage4_5": {
        "protocol": "cohortkv_single_config_stage4_5_frozen_v1",
        "status": "stage4_5_source_plan_frozen",
        "sha256": (
            "31cb442563152a250705d8ad3405461238810c5a5e81cac09c2ab20ae294e2f8"
        ),
    },
}


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


def load_stage5_formal_canary_contract(
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else repo_root
    )
    source = root / STAGE5_FORMAL_CANARY_PATH
    value = json.loads(source.read_text())
    threshold = value["threshold"]
    canonical = value["canonical_artifacts"]
    injection = value["semantic_injection"]
    canonical_program = canonical["program_memory_sha256"]
    perturbed_program = injection["perturbed_program_memory_sha256"]
    hashes = (
        canonical_program,
        perturbed_program,
        sha256_file(source),
    )
    edge_artifact = sha256_bytes(
        json.dumps(
            {
                "source_checkpoint_sha256": canonical[
                    "source_checkpoint_sha256"
                ],
                "target_checkpoint_sha256": canonical[
                    "target_checkpoint_sha256"
                ],
                "compiler_sha256": canonical["compiler_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if (
        value.get("protocol")
        != "cohortkv_single_config_stage5_formal_canary_v1"
        or value.get("status")
        != "frozen_before_formal_stage5_run"
        or value.get("source_version") != "theta0"
        or value.get("target_version") != "theta1"
        or value.get("selection_role") != "program_selection"
        or value.get("labels_used") is not False
        or value.get("metric") != "kv_relative_l2"
        or threshold.get("recommendation_labels_used") is not False
        or float(threshold["maximum_relative_l2"]) < 0.0
        or injection.get("shape_preserved") is not True
        or injection.get("dtype_preserved") is not True
        or injection.get("finite_required") is not True
        or injection.get("job_expected_hash_is_perturbed_hash")
        is not True
        or canonical_program == perturbed_program
        or any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in hashes
        )
    ):
        raise ValueError("Stage 5 formal canary artifact is invalid")
    return {
        "path": str(STAGE5_FORMAL_CANARY_PATH),
        "sha256": hashes[2],
        "protocol": value["protocol"],
        "source_version": value["source_version"],
        "target_version": value["target_version"],
        "selection_role": value["selection_role"],
        "labels_used": value["labels_used"],
        "metric": value["metric"],
        "maximum_relative_l2": float(
            threshold["maximum_relative_l2"]
        ),
        "canonical_program_sha256": canonical_program,
        "perturbed_program_sha256": perturbed_program,
        "edge_artifact_sha256": edge_artifact,
    }


def build_result_schema(
    workload_content_sha256: str | None = None,
) -> dict[str, Any]:
    formal_canary = load_stage5_formal_canary_contract()
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
    abort_coverage = [
        {
            "contains": {
                "type": "object",
                "required": ["fault"],
                "properties": {"fault": {"const": name}},
            },
            "minContains": 1,
            "maxContains": 1,
        }
        for name in ("mid_job", "pre_commit")
    ]
    direct_point_coverage = [
        {
            "contains": {
                "type": "object",
                "required": ["gpu_count"],
                "properties": {"gpu_count": {"const": gpu_count}},
            },
            "minContains": 1,
            "maxContains": 1,
        }
        for gpu_count in GPU_COUNTS
    ]
    capsule_point_coverage = [
        {
            "contains": {
                "type": "object",
                "required": ["destination", "gpu_count"],
                "properties": {
                    "destination": {"const": destination},
                    "gpu_count": {"const": gpu_count},
                },
            },
            "minContains": 1,
            "maxContains": 1,
        }
        for destination in DESTINATIONS
        for gpu_count in GPU_COUNTS
    ]
    stage5_capacity = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "device",
            "model_and_program_bytes",
            "old_kv_bytes",
            "complete_new_kv_bytes",
            "transient_bytes",
            "allocator_margin_bytes",
            "capacity_bytes",
            "required_bytes",
            "passed",
        ],
        "properties": {
            "device": {"type": "string", "minLength": 1},
            "model_and_program_bytes": positive_integer,
            "old_kv_bytes": positive_integer,
            "complete_new_kv_bytes": positive_integer,
            "transient_bytes": positive_integer,
            "allocator_margin_bytes": positive_integer,
            "capacity_bytes": positive_integer,
            "required_bytes": positive_integer,
            "passed": {"const": True},
        },
    }
    stage5_observed_free_capacity = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "measurement_boundary",
            "all_devices_passed",
            "devices",
        ],
        "properties": {
            "measurement_boundary": {
                "const": (
                    "target models and direct programs resident; before "
                    "per-case old-cache publication"
                )
            },
            "all_devices_passed": {"const": True},
            "devices": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "device",
                        "free_bytes",
                        "required_free_bytes",
                        "passed",
                    ],
                    "properties": {
                        "device": {"type": "string", "minLength": 1},
                        "free_bytes": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "required_free_bytes": positive_integer,
                        "passed": {"const": True},
                    },
                },
            },
        },
    }
    stage5_checks = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "artifact_identity",
            "program_identity",
            "program_shape",
            "old_kv_presence",
            "capacity",
            "semantic_canary",
        ],
        "properties": {
            name: {"type": "boolean"}
            for name in (
                "artifact_identity",
                "program_identity",
                "program_shape",
                "old_kv_presence",
                "capacity",
                "semantic_canary",
            )
        },
    }
    stage5_decision = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "record_id",
            "cohort_id",
            "requested_action",
            "requested_reason",
            "final_action",
            "fallback_reason",
            "source_version",
            "target_version",
            "last_exact_version_before",
            "last_exact_version_after",
            "migration_depth_before",
            "migration_depth_after",
            "state_kind_after",
            "retained_tokens",
            "final_tokens",
        ],
        "properties": {
            "record_id": {"type": "integer", "minimum": 0},
            "cohort_id": {"type": "string", "minLength": 1},
            "requested_action": {"enum": ["migrate", "exact"]},
            "requested_reason": {"type": "string", "minLength": 1},
            "final_action": {"enum": ["migrate", "exact"]},
            "fallback_reason": {
                "type": ["string", "null"],
            },
            "source_version": {
                "const": formal_canary["source_version"]
            },
            "target_version": {
                "const": formal_canary["target_version"]
            },
            "last_exact_version_before": {
                "type": ["string", "null"],
            },
            "last_exact_version_after": {
                "type": "string",
                "minLength": 1,
            },
            "migration_depth_before": {
                "type": "integer",
                "minimum": 0,
            },
            "migration_depth_after": {
                "type": "integer",
                "minimum": 0,
            },
            "state_kind_after": {"enum": ["migrated", "exact"]},
            "retained_tokens": {
                "type": "integer",
                "minimum": 0,
            },
            "final_tokens": positive_integer,
        },
        "allOf": [
            {
                "if": {
                    "properties": {"final_action": {"const": "exact"}},
                    "required": ["final_action"],
                },
                "then": {
                    "properties": {
                        "migration_depth_after": {"const": 0},
                        "state_kind_after": {"const": "exact"},
                    }
                },
            },
            {
                "if": {
                    "properties": {"final_action": {"const": "migrate"}},
                    "required": ["final_action"],
                },
                "then": {
                    "properties": {
                        "migration_depth_after": {"minimum": 1},
                        "state_kind_after": {"const": "migrated"},
                    }
                },
            },
        ],
    }
    stage5_canary = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "cohort_id",
            "record_ids",
            "source_version",
            "target_version",
            "program_sha256",
            "metric",
            "observed_relative_l2",
            "maximum_relative_l2",
            "candidate_sha256",
            "reference_sha256",
            "threshold_artifact_sha256",
            "threshold_source",
            "labels_used",
            "passed",
        ],
        "properties": {
            "cohort_id": {"type": "string", "minLength": 1},
            "record_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "integer", "minimum": 0},
            },
            "source_version": {
                "const": formal_canary["source_version"]
            },
            "target_version": {
                "const": formal_canary["target_version"]
            },
            "program_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "metric": {"const": "kv_relative_l2"},
            "observed_relative_l2": nonnegative,
            "maximum_relative_l2": {
                "const": formal_canary["maximum_relative_l2"]
            },
            "candidate_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "reference_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "threshold_artifact_sha256": {
                "const": formal_canary["sha256"]
            },
            "threshold_source": {"const": "program_selection"},
            "labels_used": {"const": False},
            "passed": {"type": "boolean"},
        },
    }
    stage5_canary_artifact = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "path",
            "sha256",
            "protocol",
            "source_version",
            "target_version",
            "selection_role",
            "labels_used",
            "metric",
            "maximum_relative_l2",
        ],
        "properties": {
            "path": {"const": formal_canary["path"]},
            "sha256": {
                "const": formal_canary["sha256"],
            },
            "protocol": {
                "const": formal_canary["protocol"]
            },
            "source_version": {
                "const": formal_canary["source_version"]
            },
            "target_version": {
                "const": formal_canary["target_version"]
            },
            "selection_role": {
                "const": formal_canary["selection_role"]
            },
            "labels_used": {"const": formal_canary["labels_used"]},
            "metric": {"const": formal_canary["metric"]},
            "maximum_relative_l2": {
                "const": formal_canary["maximum_relative_l2"]
            },
        },
    }
    stage5_preflight = {
        "type": "object",
        "additionalProperties": True,
        "required": [
            "protocol",
            "selection_role",
            "labels_used",
            "guard_hook",
            "elapsed_seconds",
            "input_measurement_seconds",
            "runtime_validation_seconds",
            "decision_seconds",
            "all_cohorts_passed",
            "cohorts",
            "decisions",
        ],
        "properties": {
            "protocol": {"const": "cohortkv_stage5_fixed_preflight_v1"},
            "selection_role": {"const": "program_selection"},
            "labels_used": {"const": False},
            "guard_hook": {
                "const": "post_retained_prefix_pre_append"
            },
            "elapsed_seconds": nonnegative,
            "input_measurement_seconds": nonnegative,
            "runtime_validation_seconds": nonnegative,
            "decision_seconds": nonnegative,
            "all_cohorts_passed": {"type": "boolean"},
            "cohorts": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": [
                        "cohort_id",
                        "source_version",
                        "target_version",
                        "checks",
                        "passed",
                        "fallback_reason",
                        "migration_required",
                        "expected_artifact_sha256",
                        "observed_artifact_sha256",
                        "expected_program_sha256",
                        "observed_program_sha256",
                        "expected_program_shape",
                        "observed_program_shape",
                        "expected_threshold_artifact_sha256",
                        "expected_old_record_ids",
                        "present_old_record_ids",
                        "expected_old_records_source",
                        "present_old_records_source",
                        "canary",
                        "device_capacity",
                        "measurement",
                    ],
                    "properties": {
                        "cohort_id": {"type": "string", "minLength": 1},
                        "source_version": {
                            "const": formal_canary["source_version"],
                        },
                        "target_version": {
                            "const": formal_canary["target_version"],
                        },
                        "checks": stage5_checks,
                        "passed": {"type": "boolean"},
                        "fallback_reason": {
                            "type": ["string", "null"],
                        },
                        "migration_required": {"type": "boolean"},
                        "expected_artifact_sha256": {
                            "const": formal_canary[
                                "edge_artifact_sha256"
                            ],
                        },
                        "observed_artifact_sha256": {
                            "const": formal_canary[
                                "edge_artifact_sha256"
                            ],
                        },
                        "expected_program_sha256": {
                            "type": ["string", "null"],
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "observed_program_sha256": {
                            "type": ["string", "null"],
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "expected_program_shape": {
                            "type": "array",
                            "items": positive_integer,
                        },
                        "observed_program_shape": {
                            "type": "array",
                            "items": positive_integer,
                        },
                        "expected_threshold_artifact_sha256": {
                            "type": ["string", "null"],
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "expected_old_record_ids": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {
                                "type": "integer",
                                "minimum": 0,
                            },
                        },
                        "present_old_record_ids": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {
                                "type": "integer",
                                "minimum": 0,
                            },
                        },
                        "expected_old_records_source": {
                            "const": "prior_committed_manifest"
                        },
                        "present_old_records_source": {
                            "const": "destination_readback"
                        },
                        "canary": {
                            "anyOf": [stage5_canary, {"type": "null"}]
                        },
                        "device_capacity": {
                            "type": "array",
                            "items": stage5_capacity,
                        },
                        "measurement": {
                            "type": "object",
                            "additionalProperties": True,
                            "required": [
                                "artifact_seconds",
                                "old_kv_presence_seconds",
                                "capacity_seconds",
                                "semantic_canary_seconds",
                                "total_seconds",
                            ],
                            "properties": {
                                name: nonnegative
                                for name in (
                                    "artifact_seconds",
                                    "old_kv_presence_seconds",
                                    "capacity_seconds",
                                    "semantic_canary_seconds",
                                    "total_seconds",
                                )
                            },
                        },
                    },
                    "allOf": [
                        {
                            "if": {
                                "properties": {
                                    "migration_required": {
                                        "const": True
                                    }
                                },
                                "required": ["migration_required"],
                            },
                            "then": {
                                "properties": {
                                    "expected_program_sha256": {
                                        "type": "string"
                                    },
                                    "observed_program_sha256": {
                                        "type": "string"
                                    },
                                    "expected_program_shape": {
                                        "minItems": 1
                                    },
                                    "observed_program_shape": {
                                        "minItems": 1
                                    },
                                    "expected_threshold_artifact_sha256": {
                                        "const": formal_canary["sha256"]
                                    },
                                    "expected_old_record_ids": {
                                        "minItems": 1
                                    },
                                    "canary": stage5_canary,
                                }
                            },
                            "else": {
                                "properties": {
                                    "expected_program_sha256": {
                                        "type": "null"
                                    },
                                    "observed_program_sha256": {
                                        "type": "null"
                                    },
                                    "expected_program_shape": {
                                        "maxItems": 0
                                    },
                                    "observed_program_shape": {
                                        "maxItems": 0
                                    },
                                    "expected_threshold_artifact_sha256": {
                                        "type": "null"
                                    },
                                    "expected_old_record_ids": {
                                        "maxItems": 0
                                    },
                                    "present_old_record_ids": {
                                        "maxItems": 0
                                    },
                                    "canary": {"type": "null"},
                                }
                            },
                        }
                    ],
                },
            },
            "decisions": {
                "type": "array",
                "minItems": EXPECTED_RECORDS,
                "maxItems": EXPECTED_RECORDS,
                "items": stage5_decision,
            },
        },
    }
    stage5_extent = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "extent_id",
            "record_ids",
            "migration_anchor_version",
            "served_kv_target",
            "num_layers",
            "token_count",
            "kv_width",
            "dtype",
            "payload_bytes",
            "location",
            "device",
            "checksum_sha256",
        ],
        "properties": {
            "extent_id": {"type": "string", "minLength": 1},
            "record_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "integer", "minimum": 0},
            },
            "migration_anchor_version": {
                "const": formal_canary["target_version"]
            },
            "served_kv_target": {
                "const": formal_canary["target_version"]
            },
            "num_layers": {
                "const": EXPECTED_MODEL["num_layers"]
            },
            "token_count": positive_integer,
            "kv_width": {
                "const": (
                    EXPECTED_MODEL["num_heads"]
                    * EXPECTED_MODEL["head_dim"]
                )
            },
            "dtype": {"const": "float16"},
            "payload_bytes": positive_integer,
            "location": {
                "type": "string",
                "pattern": "^hbm://",
            },
            "device": {"type": "string", "minLength": 1},
            "checksum_sha256": {"type": "null"},
        },
    }
    stage5_committed_manifest = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "protocol",
            "commit_hook",
            "lineage_sha256",
            "destination_manifest",
            "lineage",
        ],
        "properties": {
            "protocol": {
                "const": "cohortkv_single_config_stage5_minimal_closure_v1"
            },
            "commit_hook": {"const": "post_append_full_cache"},
            "lineage_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "destination_manifest": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "protocol",
                    "job_id",
                    "target_version",
                    "destination_id",
                    "destination_kind",
                    "publication_mode",
                    "extents",
                    "record_count",
                    "token_count",
                    "payload_bytes",
                    "metadata_sha256",
                    "metadata",
                ],
                "properties": {
                    "protocol": {
                        "const": "streamkv_destination_manifest_v1"
                    },
                    "job_id": {"type": "string", "minLength": 1},
                    "target_version": {
                        "const": formal_canary["target_version"],
                    },
                    "destination_id": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "destination_kind": {"const": "hbm"},
                    "publication_mode": {"const": "direct_device"},
                    "record_count": {"const": EXPECTED_RECORDS},
                    "token_count": positive_integer,
                    "payload_bytes": positive_integer,
                    "extents": {
                        "type": "array",
                        "minItems": 1,
                        "items": stage5_extent,
                    },
                    "metadata_sha256": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "protocol",
                            "commit_hook",
                            "lineage",
                        ],
                        "properties": {
                            "protocol": {
                                "const": (
                                    "cohortkv_single_config_"
                                    "stage5_minimal_closure_v1"
                                )
                            },
                            "commit_hook": {
                                "const": "post_append_full_cache"
                            },
                            "lineage": {
                                "type": "array",
                                "minItems": EXPECTED_RECORDS,
                                "maxItems": EXPECTED_RECORDS,
                                "items": stage5_decision,
                            },
                        },
                    },
                },
            },
            "lineage": {
                "type": "array",
                "minItems": EXPECTED_RECORDS,
                "maxItems": EXPECTED_RECORDS,
                "items": stage5_decision,
            },
        },
    }
    stage5_readback = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "protocol",
            "target_version",
            "expected_records",
            "read_records",
            "manifest_equal",
            "all_metadata_equal",
            "all_finite",
            "all_checksums_equal",
            "passed",
            "elapsed_seconds",
        ],
        "properties": {
            "protocol": {
                "const": "cohortkv_stage5_manifest_readback_v1"
            },
            "target_version": {
                "type": "string",
                "minLength": 1,
            },
            "expected_records": {"const": EXPECTED_RECORDS},
            "read_records": {"const": EXPECTED_RECORDS},
            "manifest_equal": {"const": True},
            "all_metadata_equal": {"const": True},
            "all_finite": {"const": True},
            "all_checksums_equal": {"const": True},
            "passed": {"const": True},
            "elapsed_seconds": nonnegative,
        },
    }
    stage5_job = {
        "type": "object",
        "additionalProperties": True,
        "required": [
            "protocol",
            "job_id",
            "target_version",
            "outcome",
            "fault",
            "target_manifest",
            "target_visible",
            "partial_target_visible",
            "staging_reclaimed",
            "old_readback",
            "target_readback",
            "guard_invocations",
            "staged_extents",
            "elapsed_seconds",
            "preflight",
        ],
        "properties": {
            "protocol": {
                "const": "cohortkv_single_config_stage5_minimal_closure_v1"
            },
            "job_id": {"type": "string", "minLength": 1},
            "target_version": {"type": "string", "minLength": 1},
            "outcome": {"enum": ["committed", "aborted"]},
            "fault": {
                "type": ["string", "null"],
            },
            "target_manifest": {
                "anyOf": [
                    stage5_committed_manifest,
                    {"type": "null"},
                ]
            },
            "target_visible": {"type": "boolean"},
            "partial_target_visible": {"const": False},
            "staging_reclaimed": {"const": True},
            "old_readback": {
                "anyOf": [stage5_readback, {"type": "null"}]
            },
            "target_readback": {
                "anyOf": [stage5_readback, {"type": "null"}]
            },
            "guard_invocations": positive_integer,
            "staged_extents": positive_integer,
            "elapsed_seconds": nonnegative,
            "preflight": stage5_preflight,
        },
    }
    stage5_closure_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "protocol",
            "canary_artifact",
            "copy_on_write_gpu_count",
            "copy_on_write_capacity",
            "normal_job",
            "semantic_fallback_job",
            "abort_jobs",
        ],
        "properties": {
            "protocol": {
                "const": "cohortkv_single_config_stage5_minimal_closure_v1"
            },
            "canary_artifact": stage5_canary_artifact,
            "copy_on_write_gpu_count": {"enum": [2, 4]},
            "copy_on_write_capacity": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "mode",
                    "old_extents_retained_until_commit",
                    "all_devices_passed",
                    "devices",
                    "observed_free_capacity",
                ],
                "properties": {
                    "mode": {"const": "copy_on_write"},
                    "old_extents_retained_until_commit": {"const": True},
                    "all_devices_passed": {"const": True},
                    "devices": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 4,
                        "items": stage5_capacity,
                    },
                    "observed_free_capacity": (
                        stage5_observed_free_capacity
                    ),
                },
            },
            "normal_job": {
                **stage5_job,
                "allOf": [
                    {
                        "properties": {
                            "outcome": {"const": "committed"},
                            "fault": {"type": "null"},
                            "target_visible": {"const": True},
                            "target_manifest": stage5_committed_manifest,
                            "old_readback": {"type": "null"},
                            "target_readback": stage5_readback,
                            "preflight": {
                                "properties": {
                                    "all_cohorts_passed": {"const": True}
                                }
                            },
                        }
                    }
                ],
            },
            "semantic_fallback_job": {
                **stage5_job,
                "required": [
                    *stage5_job["required"],
                    "semantic_perturbation_detected",
                    "affected_cohort_final_action",
                ],
                "allOf": [
                    {
                        "properties": {
                            "outcome": {"const": "committed"},
                            "fault": {"type": "null"},
                            "target_visible": {"const": True},
                            "target_manifest": stage5_committed_manifest,
                            "old_readback": {"type": "null"},
                            "target_readback": stage5_readback,
                            "semantic_perturbation_detected": {
                                "const": True
                            },
                            "affected_cohort_final_action": {
                                "const": "exact"
                            },
                            "preflight": {
                                "properties": {
                                    "all_cohorts_passed": {"const": False},
                                    "decisions": {
                                        "contains": {
                                            "type": "object",
                                            "required": [
                                                "requested_action",
                                                "final_action",
                                                "fallback_reason",
                                            ],
                                            "properties": {
                                                "requested_action": {
                                                    "const": "migrate"
                                                },
                                                "final_action": {
                                                    "const": "exact"
                                                },
                                                "fallback_reason": {
                                                    "type": "string",
                                                    "pattern": (
                                                        "(^|\\+)"
                                                        "semantic_canary"
                                                        "($|\\+)"
                                                    ),
                                                },
                                            },
                                        },
                                        "minContains": 1,
                                    },
                                }
                            },
                        }
                    }
                ],
            },
            "abort_jobs": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "allOf": abort_coverage,
                "items": {
                    **stage5_job,
                    "required": stage5_job["required"],
                    "properties": {
                        **stage5_job["properties"],
                        "fault": {"enum": ["mid_job", "pre_commit"]},
                        "outcome": {"const": "aborted"},
                        "target_visible": {"const": False},
                        "target_manifest": {"type": "null"},
                        "old_readback": stage5_readback,
                        "target_readback": {"type": "null"},
                    },
                },
            },
        },
        "allOf": [
            {
                "if": {
                    "properties": {
                        "copy_on_write_gpu_count": {"const": gpu_count}
                    },
                    "required": ["copy_on_write_gpu_count"],
                },
                "then": {
                    "properties": {
                        "copy_on_write_capacity": {
                            "properties": {
                                "devices": {
                                    "minItems": gpu_count,
                                    "maxItems": gpu_count,
                                },
                                "observed_free_capacity": {
                                    "properties": {
                                        "devices": {
                                            "minItems": gpu_count,
                                            "maxItems": gpu_count,
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            }
            for gpu_count in (2, 4)
        ],
    }
    accounting_input_properties = {
        name: {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "protocol", "sha256", "status"],
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "protocol": {"const": contract["protocol"]},
                "sha256": {"const": contract["sha256"]},
                "status": {"const": contract["status"]},
            },
        }
        for name, contract in STAGE5_ACCOUNTING_FROZEN_INPUTS.items()
    }
    source_state_accounting_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "protocol",
            "status",
            "scientific_result",
            "inputs",
            "workload",
            "active_direct_oldkv",
            "offline_setup",
            "rejected_fp16_normalized_capsule",
            "dram_resident_backup",
            "claim_boundary",
        ],
        "properties": {
            "protocol": {
                "const": "cohortkv_stage5_source_state_accounting_v1"
            },
            "status": {"const": "artifact_derived"},
            "scientific_result": {"const": True},
            "inputs": {
                "type": "object",
                "required": ["stage2", "stage4", "stage4_5"],
                "additionalProperties": False,
                "properties": accounting_input_properties,
            },
            "workload": {
                "type": "object",
                "required": ["content_sha256", "records", "prefix_tokens"],
                "properties": {
                    "content_sha256": workload_hash,
                    "records": {"const": EXPECTED_RECORDS},
                    "prefix_tokens": {"const": EXPECTED_PREFIX_TOKENS},
                },
                "additionalProperties": True,
            },
            "active_direct_oldkv": {
                "type": "object",
                "required": [
                    "representation",
                    "placement",
                    "additional_per_record_source_state_bytes",
                    "independent_capture_required",
                    "independent_encode_required",
                    "independent_preload_required",
                    "existing_old_kv_logical_bytes",
                    "program_set",
                    "normal_path_points",
                    "copy_on_write_abort_safe_peak_measured",
                ],
                "properties": {
                    "representation": {
                        "const": "existing_old_kv_fp16"
                    },
                    "placement": {
                        "const": "existing serving cache in HBM"
                    },
                    "additional_per_record_source_state_bytes": {
                        "const": 0
                    },
                    "independent_capture_required": {"const": False},
                    "independent_encode_required": {"const": False},
                    "independent_preload_required": {"const": False},
                    "existing_old_kv_logical_bytes": {
                        "const": EXPECTED_LOGICAL_TARGET_BYTES_FP16
                    },
                    "program_set": {
                        "type": "object",
                        "required": [
                            "serialized_file_bytes",
                            "resident_tensor_bytes_per_worker",
                            "composition_seconds",
                            "serialization_timing_available",
                            "programs",
                        ],
                        "properties": {
                            "serialized_file_bytes": positive_integer,
                            "resident_tensor_bytes_per_worker": (
                                positive_integer
                            ),
                            "composition_seconds": nonnegative,
                            "serialization_timing_available": {
                                "const": False
                            },
                            "programs": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "source_version",
                                        "target_version",
                                        "serialized_file_bytes",
                                        "composition_seconds",
                                        "sha256",
                                    ],
                                    "properties": {
                                        "source_version": {
                                            "enum": [
                                                "theta0",
                                                "theta4",
                                                "theta10",
                                            ]
                                        },
                                        "target_version": {
                                            "const": "theta11"
                                        },
                                        "serialized_file_bytes": (
                                            positive_integer
                                        ),
                                        "composition_seconds": nonnegative,
                                        "sha256": {
                                            "type": "string",
                                            "pattern": "^[0-9a-f]{64}$",
                                        },
                                    },
                                },
                                "allOf": source_version_coverage,
                            },
                        },
                        "additionalProperties": True,
                    },
                    "normal_path_points": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "allOf": direct_point_coverage,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "gpu_count",
                                "initial_old_kv_bytes_all_devices",
                                "peak_old_plus_new_kv_bytes_all_devices",
                                "maximum_peak_hbm_bytes_single_device",
                                "median_seconds",
                                "measurement_mode",
                                "abort_safe",
                            ],
                            "properties": {
                                "gpu_count": {"enum": [1, 2, 4]},
                                "initial_old_kv_bytes_all_devices": (
                                    positive_integer
                                ),
                                "peak_old_plus_new_kv_bytes_all_devices": (
                                    positive_integer
                                ),
                                "maximum_peak_hbm_bytes_single_device": (
                                    positive_integer
                                ),
                                "median_seconds": nonnegative,
                                "measurement_mode": {
                                    "const": "extent_reclaim_normal_path"
                                },
                                "abort_safe": {"const": False},
                            },
                        },
                    },
                    "copy_on_write_abort_safe_peak_measured": {
                        "const": False
                    },
                },
                "additionalProperties": True,
            },
            "offline_setup": {
                "type": "object",
                "required": [
                    "historical_fit_seconds",
                    "runtime_prepare_seconds",
                    "certificate_seconds",
                    "stage2_one_time_seconds",
                    "direct_program_composition_seconds",
                    "seconds_per_record_at_682_stage2_floor",
                    "stage2_amortization_curve",
                ],
                "properties": {
                    "historical_fit_seconds": nonnegative,
                    "runtime_prepare_seconds": nonnegative,
                    "certificate_seconds": nonnegative,
                    "stage2_one_time_seconds": nonnegative,
                    "direct_program_composition_seconds": nonnegative,
                    "seconds_per_record_at_682_stage2_floor": nonnegative,
                    "stage2_amortization_curve": {
                        "type": "array",
                        "minItems": 1,
                    },
                },
                "additionalProperties": True,
            },
            "rejected_fp16_normalized_capsule": {
                "type": "object",
                "required": [
                    "logical_bytes",
                    "physical_bytes",
                    "matched_points",
                    "beats_paired_exact_points",
                    "source_read_fraction_min",
                    "source_read_fraction_max",
                    "points",
                ],
                "properties": {
                    "logical_bytes": {
                        "const": EXPECTED_LOGICAL_CAPSULE_BYTES_FP16
                    },
                    "physical_bytes": positive_integer,
                    "matched_points": {"const": 6},
                    "beats_paired_exact_points": {"const": 0},
                    "source_read_fraction_min": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "source_read_fraction_max": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "points": {
                        "type": "array",
                        "minItems": 6,
                        "maxItems": 6,
                        "allOf": capsule_point_coverage,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "destination",
                                "gpu_count",
                                "compiled_median_seconds",
                                "source_read_seconds",
                                "source_read_fraction",
                                "paired_exact_median_seconds",
                                "beats_paired_exact",
                            ],
                            "properties": {
                                "destination": {
                                    "enum": ["hbm", "dram"]
                                },
                                "gpu_count": {"enum": [1, 2, 4]},
                                "compiled_median_seconds": nonnegative,
                                "source_read_seconds": nonnegative,
                                "source_read_fraction": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "paired_exact_median_seconds": nonnegative,
                                "beats_paired_exact": {"const": False},
                            },
                        },
                    },
                },
                "additionalProperties": True,
            },
            "dram_resident_backup": {
                "type": "object",
                "additionalProperties": False,
                "required": ["active_route", "status", "points"],
                "properties": {
                    "active_route": {"const": False},
                    "status": {"type": "string", "minLength": 1},
                    "points": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "allOf": [
                            {
                                "contains": {
                                    "type": "object",
                                    "required": ["gpu_count"],
                                    "properties": {
                                        "gpu_count": {"const": gpu_count}
                                    },
                                },
                                "minContains": 1,
                                "maxContains": 1,
                            }
                            for gpu_count in (1, 4)
                        ],
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "gpu_count",
                                "preload_seconds",
                                "standing_host_source_bytes",
                                "compiled_median_seconds",
                                "paired_exact_median_seconds",
                                "beats_paired_exact_after_preload",
                            ],
                            "properties": {
                                "gpu_count": {"enum": [1, 4]},
                                "preload_seconds": nonnegative,
                                "standing_host_source_bytes": (
                                    positive_integer
                                ),
                                "compiled_median_seconds": nonnegative,
                                "paired_exact_median_seconds": nonnegative,
                                "beats_paired_exact_after_preload": {
                                    "const": True
                                },
                            },
                        },
                    },
                },
            },
            "claim_boundary": {
                "type": "object",
                "required": [
                    "primary_claim",
                    "physical_ssd_performance_claim",
                    "cold_filesystem_speedup_claim",
                    "capsule_capture_claim",
                    "int8_claim",
                    "time_break_even_claim",
                ],
                "properties": {
                    "primary_claim": {
                        "const": "prepublished-program hot-HBM data-plane"
                    },
                    "physical_ssd_performance_claim": {"const": False},
                    "cold_filesystem_speedup_claim": {"const": False},
                    "capsule_capture_claim": {"const": False},
                    "int8_claim": {"const": False},
                    "time_break_even_claim": {"const": False},
                },
                "additionalProperties": True,
            },
        },
    }
    artifact_descriptor = {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "bytes", "sha256", "protocol", "status"],
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "bytes": positive_integer,
            "sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "protocol": {"type": "string", "minLength": 1},
            "status": {"type": "string", "minLength": 1},
        },
    }
    report_descriptor = {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "bytes", "sha256"],
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "bytes": positive_integer,
            "sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
    }
    task_metrics = {
        "type": "object",
        "additionalProperties": False,
        "required": ["catalog_auc", "ndcg_at_100", "hit_at_100"],
        "properties": {
            "catalog_auc": {"type": "number"},
            "ndcg_at_100": {"type": "number"},
            "hit_at_100": {"type": "number"},
        },
    }
    lifecycle_candidate = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "candidate_name",
            "artifact",
            "primary_sum_u_over_sum_e",
            "record_weighted_task_ratio",
            "scheduled_exact_records",
            "reusable_records",
            "maximum_observed_migration_depth",
            "checks_passed",
        ],
        "properties": {
            "candidate_name": {
                "enum": [
                    STAGE49_COST_ENDPOINT,
                    STAGE49_SELECTED_CANDIDATE,
                ]
            },
            "artifact": artifact_descriptor,
            "primary_sum_u_over_sum_e": nonnegative,
            "record_weighted_task_ratio": task_metrics,
            "scheduled_exact_records": {
                "type": "integer",
                "minimum": 0,
            },
            "reusable_records": positive_integer,
            "maximum_observed_migration_depth": {
                "type": "integer",
                "minimum": 0,
            },
            "checks_passed": {"const": True},
        },
    }
    lifecycle_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "fixed_history",
            "corrected_growing_history",
        ],
        "properties": {
            "fixed_history": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "summary_artifact",
                    "policy_artifact",
                    "updates",
                    "records",
                    "maximum_migration_depth",
                    "cumulative_gpu_cost_ratio",
                    "certificate_passed",
                    "scope",
                ],
                "properties": {
                    "summary_artifact": artifact_descriptor,
                    "policy_artifact": artifact_descriptor,
                    "updates": {"const": 11},
                    "records": {"const": EXPECTED_RECORDS},
                    "maximum_migration_depth": {"const": 4},
                    "cumulative_gpu_cost_ratio": nonnegative,
                    "certificate_passed": {"const": True},
                    "scope": {
                        "const": (
                            "fixed_history_hot_hbm_single_seed_development"
                        )
                    },
                },
            },
            "corrected_growing_history": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "summary_artifact",
                    "selected_candidate",
                    "cost_endpoint",
                    "selection_basis",
                    "candidates",
                    "target_append_excluded",
                    "groupwise_host_staging",
                    "state_movement_reported_separately",
                    "full_cohort_hbm_claim",
                    "end_to_end_state_movement_claim",
                    "checks_passed",
                ],
                "properties": {
                    "summary_artifact": artifact_descriptor,
                    "selected_candidate": {
                        "const": STAGE49_SELECTED_CANDIDATE
                    },
                    "cost_endpoint": {
                        "const": STAGE49_COST_ENDPOINT
                    },
                    "selection_basis": {
                        "const": (
                            "freeze the preregistered bounded-renewal "
                            "candidate without using recommendation labels; "
                            "retain token debt only as the cost endpoint"
                        )
                    },
                    "candidates": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": lifecycle_candidate,
                        "allOf": [
                            {
                                "contains": {
                                    "type": "object",
                                    "required": ["candidate_name"],
                                    "properties": {
                                        "candidate_name": {
                                            "const": candidate
                                        }
                                    },
                                },
                                "minContains": 1,
                                "maxContains": 1,
                            }
                            for candidate in (
                                STAGE49_COST_ENDPOINT,
                                STAGE49_SELECTED_CANDIDATE,
                            )
                        ],
                    },
                    "target_append_excluded": {"const": True},
                    "groupwise_host_staging": {"const": True},
                    "state_movement_reported_separately": {
                        "const": True
                    },
                    "full_cohort_hbm_claim": {"const": False},
                    "end_to_end_state_movement_claim": {
                        "const": False
                    },
                    "checks_passed": {"const": True},
                },
            },
        },
    }
    stage6_output_names = (
        "correctness_report",
        "timing_memory_report",
        "paper_tables",
        "paper_figures",
        "artifact_to_claim",
        "negative_results_log",
        "tbd_disposition",
        "code_snapshot_manifest",
    )
    stage6_closure_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "protocol",
            "status",
            "selected_candidate",
            "old_gpu_matrix_rerun",
            "source_artifacts",
            "outputs",
            "checks",
        ],
        "properties": {
            "protocol": {"const": STAGE6_PROTOCOL},
            "status": {"const": "single_configuration_v1_frozen"},
            "selected_candidate": {
                "const": STAGE49_SELECTED_CANDIDATE
            },
            "old_gpu_matrix_rerun": {"const": False},
            "source_artifacts": {
                "type": "array",
                "minItems": 12,
                "items": artifact_descriptor,
            },
            "outputs": {
                "type": "object",
                "additionalProperties": False,
                "required": list(stage6_output_names),
                "properties": {
                    name: report_descriptor
                    for name in stage6_output_names
                },
            },
            "checks": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "all_source_hashes",
                    "whole_aggregate_semantics",
                    "jsonschema",
                    "stage5_semantics",
                    "candidate_binding",
                    "all_tbd_markers_disposed",
                    "all_claims_bound",
                    "all_passed",
                ],
                "properties": {
                    "all_source_hashes": {"const": True},
                    "whole_aggregate_semantics": {"const": True},
                    "jsonschema": {"const": True},
                    "stage5_semantics": {"const": True},
                    "candidate_binding": {"const": True},
                    "all_tbd_markers_disposed": {"const": True},
                    "all_claims_bound": {"const": True},
                    "all_passed": {"const": True},
                },
            },
        },
    }
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
            "lifecycle",
            "stage5_closure",
            "source_state_accounting",
            "stage6_closure",
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
            "lifecycle": lifecycle_schema,
            "stage5_closure": stage5_closure_schema,
            "source_state_accounting": source_state_accounting_schema,
            "stage6_closure": stage6_closure_schema,
            "legacy_rq4_failures": {
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
            "legacy_rq5_economics": {
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


def validate_stage5_closure_semantics(value: dict[str, Any]) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    formal_canary = load_stage5_formal_canary_contract()
    canary_artifact = value["canary_artifact"]
    expected_canary_artifact = {
        name: formal_canary[name]
        for name in (
            "path",
            "sha256",
            "protocol",
            "source_version",
            "target_version",
            "selection_role",
            "labels_used",
            "metric",
            "maximum_relative_l2",
        )
    }
    require(
        canary_artifact == expected_canary_artifact,
        "Stage 5 canary artifact contract differs",
    )
    capacity = value["copy_on_write_capacity"]
    devices = capacity["devices"]
    require(
        len(devices) == int(value["copy_on_write_gpu_count"]),
        "Stage 5 COW device count differs",
    )
    require(
        len({device["device"] for device in devices}) == len(devices),
        "Stage 5 COW devices repeat",
    )
    for device in devices:
        required = sum(
            int(device[name])
            for name in (
                "model_and_program_bytes",
                "old_kv_bytes",
                "complete_new_kv_bytes",
                "transient_bytes",
                "allocator_margin_bytes",
            )
        )
        require(
            int(device["model_and_program_bytes"]) > 0
            and required == int(device["required_bytes"])
            and required <= int(device["capacity_bytes"])
            and device["passed"] is True,
            "Stage 5 COW capacity arithmetic differs",
        )
    capacity_by_device = {
        str(device["device"]): device for device in devices
    }
    observed_free = capacity["observed_free_capacity"]
    observed_devices = observed_free["devices"]
    observed_by_device = {
        str(device["device"]): device for device in observed_devices
    }
    require(
        observed_free["all_devices_passed"] is True
        and len(observed_by_device) == len(observed_devices)
        and set(observed_by_device) == set(capacity_by_device),
        "Stage 5 observed free-capacity coverage differs",
    )
    for device_name, observation in observed_by_device.items():
        declared = capacity_by_device[device_name]
        required_free = sum(
            int(declared[name])
            for name in (
                "old_kv_bytes",
                "complete_new_kv_bytes",
                "transient_bytes",
                "allocator_margin_bytes",
            )
        )
        require(
            int(observation["required_free_bytes"]) == required_free
            and int(observation["free_bytes"]) >= required_free
            and observation["passed"] is True,
            "Stage 5 observed free-capacity arithmetic differs",
        )

    def validate_job(job: dict[str, Any], committed: bool) -> None:
        preflight = job["preflight"]
        decisions = preflight["decisions"]
        cohort_ids = {
            str(cohort["cohort_id"]) for cohort in preflight["cohorts"]
        }
        require(
            len(decisions) == EXPECTED_RECORDS
            and len({int(item["record_id"]) for item in decisions})
            == EXPECTED_RECORDS,
            "Stage 5 preflight decision coverage differs",
        )
        require(
            {str(item["cohort_id"]) for item in decisions} == cohort_ids,
            "Stage 5 preflight cohort coverage differs",
        )
        for item in decisions:
            exact = item["final_action"] == "exact"
            migrated = item["final_action"] == "migrate"
            fallback = item["fallback_reason"]
            require(
                0 <= int(item["retained_tokens"])
                < int(item["final_tokens"])
                and item["target_version"] == job["target_version"]
                and (
                    (
                        exact
                        and item["last_exact_version_after"]
                        == item["target_version"]
                        and int(item["migration_depth_after"]) == 0
                        and item["state_kind_after"] == "exact"
                    )
                    or (
                        migrated
                        and item["last_exact_version_before"] is not None
                        and item["last_exact_version_after"]
                        == item["last_exact_version_before"]
                        and int(item["migration_depth_after"])
                        == int(item["migration_depth_before"]) + 1
                        and item["state_kind_after"] == "migrated"
                    )
                )
                and (
                    (
                        item["requested_action"] == "migrate"
                        and exact
                        and isinstance(fallback, str)
                        and bool(fallback)
                    )
                    or (
                        item["requested_action"] == item["final_action"]
                        and fallback is None
                    )
                ),
                "Stage 5 decision lineage differs",
            )
        require(
            abs(
                float(preflight["elapsed_seconds"])
                - (
                    float(preflight["input_measurement_seconds"])
                    + float(preflight["runtime_validation_seconds"])
                    + float(preflight["decision_seconds"])
                )
            )
            <= 1e-9,
            "Stage 5 preflight timing arithmetic differs",
        )
        require(
            bool(preflight["all_cohorts_passed"])
            == all(bool(item["passed"]) for item in preflight["cohorts"]),
            "Stage 5 preflight cohort aggregate differs",
        )
        capacity_cohorts = 0
        for cohort in preflight["cohorts"]:
            checks = cohort["checks"]
            cohort_id = str(cohort["cohort_id"])
            cohort_decisions = [
                item
                for item in decisions
                if str(item["cohort_id"]) == cohort_id
            ]
            requested_migrant_ids = {
                int(item["record_id"])
                for item in cohort_decisions
                if item["requested_action"] == "migrate"
            }
            migration_required = bool(cohort["migration_required"])
            expected_old_ids = {
                int(record_id)
                for record_id in cohort["expected_old_record_ids"]
            }
            present_old_ids = {
                int(record_id)
                for record_id in cohort["present_old_record_ids"]
            }
            canary = cohort["canary"]
            declared_capacity = cohort["device_capacity"]
            expected_checks = {
                "artifact_identity": (
                    cohort["expected_artifact_sha256"]
                    == cohort["observed_artifact_sha256"]
                ),
                "program_identity": (
                    not migration_required
                    or cohort["expected_program_sha256"]
                    == cohort["observed_program_sha256"]
                ),
                "program_shape": (
                    not migration_required
                    or cohort["expected_program_shape"]
                    == cohort["observed_program_shape"]
                ),
                "old_kv_presence": (
                    not migration_required
                    or expected_old_ids.issubset(present_old_ids)
                ),
                "capacity": all(
                    bool(device["passed"]) for device in declared_capacity
                ),
                "semantic_canary": (
                    not migration_required
                    or (
                        canary is not None
                        and bool(canary["passed"])
                    )
                ),
            }
            require(
                checks == expected_checks,
                "Stage 5 raw preflight evidence differs from checks",
            )
            require(
                cohort["expected_artifact_sha256"]
                == formal_canary["edge_artifact_sha256"],
                "Stage 5 frozen edge artifact differs",
            )
            failed = {
                name
                for name, passed in expected_checks.items()
                if not bool(passed)
            }
            require(
                bool(cohort["passed"]) == (not failed)
                and bool(checks["artifact_identity"])
                and bool(checks["capacity"])
                and (
                    (
                        not failed
                        and cohort["fallback_reason"] is None
                    )
                    or (
                        failed
                        and set(str(cohort["fallback_reason"]).split("+"))
                        == failed
                    )
                ),
                "Stage 5 cohort preflight aggregate differs",
            )
            for item in cohort_decisions:
                if item["requested_action"] == "exact":
                    action_bound = (
                        item["final_action"] == "exact"
                        and item["fallback_reason"] is None
                    )
                elif cohort["passed"]:
                    action_bound = (
                        item["final_action"] == "migrate"
                        and item["fallback_reason"] is None
                    )
                else:
                    action_bound = (
                        item["final_action"] == "exact"
                        and item["fallback_reason"]
                        == cohort["fallback_reason"]
                    )
                require(
                    action_bound,
                    "Stage 5 final action differs from cohort preflight",
                )
            measurement = cohort["measurement"]
            measurement_sum = sum(
                float(measurement[name])
                for name in (
                    "artifact_seconds",
                    "old_kv_presence_seconds",
                    "capacity_seconds",
                    "semantic_canary_seconds",
                )
            )
            require(
                abs(
                    measurement_sum
                    - float(measurement["total_seconds"])
                )
                <= 1e-9,
                "Stage 5 cohort measurement arithmetic differs",
            )
            require(
                migration_required == bool(requested_migrant_ids)
                and expected_old_ids == requested_migrant_ids,
                "Stage 5 migration cohort population differs",
            )
            require(
                all(
                    item["source_version"] == cohort["source_version"]
                    and item["target_version"] == cohort["target_version"]
                    for item in cohort_decisions
                ),
                "Stage 5 cohort and decision versions differ",
            )
            if migration_required:
                require(
                    present_old_ids.issubset(expected_old_ids)
                    and cohort["expected_old_records_source"]
                    == "prior_committed_manifest"
                    and cohort["present_old_records_source"]
                    == "destination_readback"
                    and cohort["expected_program_sha256"] is not None
                    and cohort["observed_program_sha256"] is not None
                    and bool(cohort["expected_program_shape"])
                    and bool(cohort["observed_program_shape"])
                    and cohort["expected_threshold_artifact_sha256"]
                    == canary_artifact["sha256"]
                    and canary is not None
                    and canary["cohort_id"] == cohort_id
                    and set(int(item) for item in canary["record_ids"])
                    .issubset(expected_old_ids)
                    and canary["source_version"]
                    == cohort["source_version"]
                    and canary["target_version"]
                    == cohort["target_version"]
                    and canary["program_sha256"]
                    == cohort["observed_program_sha256"]
                    and canary["metric"] == canary_artifact["metric"]
                    and canary["threshold_artifact_sha256"]
                    == canary_artifact["sha256"]
                    and canary["threshold_source"]
                    == canary_artifact["selection_role"]
                    and canary["labels_used"]
                    == canary_artifact["labels_used"]
                    and float(canary["maximum_relative_l2"])
                    == float(
                        canary_artifact["maximum_relative_l2"]
                    )
                    and bool(canary["passed"])
                    == (
                        float(canary["observed_relative_l2"])
                        <= float(canary["maximum_relative_l2"])
                    )
                    and cohort["source_version"]
                    == canary_artifact["source_version"]
                    and cohort["target_version"]
                    == canary_artifact["target_version"],
                    "Stage 5 migration canary provenance differs",
                )
            else:
                require(
                    cohort["expected_program_sha256"] is None
                    and cohort["observed_program_sha256"] is None
                    and not cohort["expected_program_shape"]
                    and not cohort["observed_program_shape"]
                    and cohort["expected_threshold_artifact_sha256"]
                    is None
                    and not expected_old_ids
                    and not present_old_ids
                    and canary is None,
                    "Stage 5 exact cohort carries migration evidence",
                )
            if declared_capacity:
                capacity_cohorts += 1
                require(
                    {
                        str(device["device"]): device
                        for device in declared_capacity
                    }
                    == capacity_by_device,
                    "Stage 5 job and COW capacity evidence differ",
                )
        require(
            abs(
                float(preflight["input_measurement_seconds"])
                - sum(
                    float(cohort["measurement"]["total_seconds"])
                    for cohort in preflight["cohorts"]
                )
            )
            <= 1e-9,
            "Stage 5 preflight input measurement differs",
        )
        require(
            capacity_cohorts >= 1,
            "Stage 5 job has no COW capacity evidence",
        )
        require(
            int(job["guard_invocations"]) == int(job["staged_extents"]),
            "Stage 5 guarded extent count differs",
        )
        if not committed:
            require(
                job["target_manifest"] is None
                and job["target_visible"] is False,
                "Stage 5 aborted target visibility differs",
            )
            require(
                job["target_readback"] is None,
                "Stage 5 aborted target readback differs",
            )
            readback = job["old_readback"]
            require(
                int(readback["expected_records"]) == EXPECTED_RECORDS
                and int(readback["read_records"]) == EXPECTED_RECORDS
                and readback["passed"] is True,
                "Stage 5 abort readback coverage differs",
            )
            require(
                readback["target_version"]
                in {item["source_version"] for item in decisions},
                "Stage 5 abort source version differs",
            )
            return
        target = job["target_manifest"]
        lineage = target["lineage"]
        destination = target["destination_manifest"]
        metadata = destination["metadata"]
        target_readback = job["target_readback"]
        require(
            len(lineage) == EXPECTED_RECORDS
            and len({int(item["record_id"]) for item in lineage})
            == EXPECTED_RECORDS,
            "Stage 5 committed lineage coverage differs",
        )
        require(
            destination["job_id"] == job["job_id"]
            and destination["target_version"] == job["target_version"]
            and int(destination["record_count"]) == EXPECTED_RECORDS
            and all(
                extent["migration_anchor_version"]
                == job["target_version"]
                and extent["served_kv_target"]
                == job["target_version"]
                for extent in destination["extents"]
            ),
            "Stage 5 committed target version differs",
        )
        require(
            job["old_readback"] is None
            and int(target_readback["expected_records"])
            == EXPECTED_RECORDS
            and int(target_readback["read_records"]) == EXPECTED_RECORDS
            and target_readback["target_version"] == job["target_version"]
            and target_readback["manifest_equal"] is True
            and target_readback["all_metadata_equal"] is True
            and target_readback["all_finite"] is True
            and target_readback["all_checksums_equal"] is True
            and target_readback["passed"] is True,
            "Stage 5 committed target readback differs",
        )
        decision_by_id = {
            int(item["record_id"]): item for item in decisions
        }
        extents = destination["extents"]
        require(
            int(job["guard_invocations"])
            == int(job["staged_extents"])
            == len(extents)
            and len(
                {str(extent["extent_id"]) for extent in extents}
            )
            == len(extents),
            "Stage 5 committed guarded extent count differs",
        )
        expected_kv_width = (
            EXPECTED_MODEL["num_heads"] * EXPECTED_MODEL["head_dim"]
        )
        extent_token_total = 0
        extent_payload_total = 0
        payload_by_device = {
            device_name: 0 for device_name in capacity_by_device
        }
        for extent in extents:
            extent_record_ids = [
                int(record_id) for record_id in extent["record_ids"]
            ]
            require(
                set(extent_record_ids).issubset(decision_by_id),
                "Stage 5 committed extent record IDs differ",
            )
            expected_tokens = sum(
                int(decision_by_id[record_id]["final_tokens"])
                for record_id in extent_record_ids
            )
            expected_payload = (
                2
                * EXPECTED_MODEL["num_layers"]
                * expected_tokens
                * expected_kv_width
                * 2
                + len(extent_record_ids) * 8
                + (len(extent_record_ids) + 1) * 8
            )
            require(
                int(extent["num_layers"])
                == EXPECTED_MODEL["num_layers"]
                and int(extent["kv_width"]) == expected_kv_width
                and extent["dtype"] == "float16"
                and int(extent["token_count"]) == expected_tokens
                and int(extent["payload_bytes"]) == expected_payload
                and extent["migration_anchor_version"]
                == job["target_version"]
                and extent["served_kv_target"]
                == job["target_version"]
                and extent["device"] in capacity_by_device
                and str(extent["location"]).startswith(
                    f"hbm://{destination['destination_id']}/"
                    f"{extent['device']}/"
                ),
                "Stage 5 committed extent ABI differs",
            )
            extent_token_total += expected_tokens
            extent_payload_total += expected_payload
            payload_by_device[str(extent["device"])] += expected_payload
        require(
            destination["destination_kind"] == "hbm"
            and destination["publication_mode"] == "direct_device"
            and int(destination["token_count"]) == extent_token_total
            and int(destination["payload_bytes"])
            == extent_payload_total,
            "Stage 5 committed manifest totals differ",
        )
        require(
            payload_by_device
            == {
                device_name: int(device["complete_new_kv_bytes"])
                for device_name, device in capacity_by_device.items()
            },
            "Stage 5 committed bytes differ from COW capacity evidence",
        )
        require(
            metadata["lineage"] == lineage
            and decision_by_id
            == {int(item["record_id"]): item for item in lineage},
            "Stage 5 atomic lineage payload differs",
        )
        metadata_json = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        )
        lineage_sha256 = hashlib.sha256(metadata_json.encode()).hexdigest()
        require(
            target["lineage_sha256"] == lineage_sha256
            and destination["metadata_sha256"] == lineage_sha256,
            "Stage 5 lineage SHA-256 differs",
        )
        manifest_record_ids = [
            int(record_id)
            for extent in destination["extents"]
            for record_id in extent["record_ids"]
        ]
        require(
            len(manifest_record_ids) == EXPECTED_RECORDS
            and len(set(manifest_record_ids)) == EXPECTED_RECORDS
            and manifest_record_ids
            == [int(item["record_id"]) for item in lineage],
            "Stage 5 manifest and lineage record order differs",
        )

    validate_job(value["normal_job"], True)
    semantic = value["semantic_fallback_job"]
    validate_job(semantic, True)
    require(
        semantic["semantic_perturbation_detected"] is True
        and semantic["affected_cohort_final_action"] == "exact"
        and any(
            item["requested_action"] == "migrate"
            and item["final_action"] == "exact"
            and "semantic_canary" in str(item["fallback_reason"])
            for item in semantic["preflight"]["decisions"]
        ),
        "Stage 5 semantic fallback evidence differs",
    )
    normal_programs = {
        cohort["observed_program_sha256"]
        for cohort in value["normal_job"]["preflight"]["cohorts"]
        if cohort["migration_required"]
    }
    perturbed_programs = {
        cohort["observed_program_sha256"]
        for cohort in semantic["preflight"]["cohorts"]
        if cohort["migration_required"]
        and not cohort["checks"]["semantic_canary"]
    }
    require(
        normal_programs
        == {formal_canary["canonical_program_sha256"]}
        and perturbed_programs
        == {formal_canary["perturbed_program_sha256"]},
        "Stage 5 semantic case did not execute a distinct program",
    )
    for abort in value["abort_jobs"]:
        validate_job(abort, False)
        require(
            {
                cohort["observed_program_sha256"]
                for cohort in abort["preflight"]["cohorts"]
                if cohort["migration_required"]
            }
            == {formal_canary["canonical_program_sha256"]},
            "Stage 5 abort job program identity differs",
        )


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
        "source_state_accounting_contract": {
            "protocol": "cohortkv_stage5_source_state_accounting_v1",
            "inputs": {
                name: {
                    **contract,
                    "path": str(
                        {
                            "stage2": STAGE2_SUMMARY_PATH,
                            "stage4": STAGE4_SUMMARY_PATH,
                            "stage4_5": STAGE45_SUMMARY_PATH,
                        }[name]
                    ),
                }
                for name, contract in (
                    STAGE5_ACCOUNTING_FROZEN_INPUTS.items()
                )
            },
            "validation": [
                "every input protocol, frozen status, and file SHA-256",
                "Stage-4 to Stage-2 and Stage-4.5 to Stage-4 upstream hashes",
                "shared workload and source-manifest hashes",
                "existing-old-K/V representation, HBM placement, and zero added state",
                "all three direct endpoints passed capacity and correctness",
                "all six capsule/exact endpoints passed correctness before timing comparison",
            ],
            "active_route": {
                "representation": "existing_old_kv_fp16",
                "placement": "existing serving cache in HBM",
                "additional_per_record_source_state_bytes": 0,
                "capture_required": False,
                "encode_required": False,
                "preload_required": False,
            },
            "required_existing_evidence": [
                "direct-program file bytes and composition seconds",
                "resident program tensor bytes per worker",
                "one/two/four-GPU normal-path old/new peak bytes",
                "Stage-2 fit, runtime-prepare, certificate, and amortization",
                "rejected FP16 normalized-capsule endpoint source-read cost",
                "retired DRAM-resident capsule preload and standing bytes",
            ],
            "unmeasured_claims_forbidden": [
                "independent program serialization time",
                "capsule-only attribution of joint source materialization",
                "INT8 or FP8 capsule results",
                "physical SSD performance",
                "capture or persistence overhead",
                "time break-even including unmeasured capture",
            ],
        },
        "guard_selection_contract": {
            "role": "program_selection",
            "labels_used": False,
            "logical_requirement": (
                "one fixed label-free semantic preflight detects the frozen "
                "integrity-valid theta0-to-theta1 program perturbation before "
                "any target extent"
            ),
            "reference_accounting": [
                "artifact and program identity",
                "program shape and version",
                "old K/V presence read from the committed source manifest",
                "per-device copy-on-write capacity",
                "semantic-canary GPU time",
                "complete preflight overhead",
            ],
            "selection_rule": (
                "freeze one threshold on program-selection records; "
                "program, shape, old-cache, or semantic failure routes the "
                "affected migration cohort to exact before target execution, "
                "while artifact/version or capacity failure aborts admission"
            ),
            "freeze_boundary": (
                "freeze the canary and threshold before one normal job, one "
                "semantic-fallback job, and two abort jobs"
            ),
            "runtime_sentinel_search_required": False,
            "online_rework_required": False,
            "required_result_evidence": [
                "all per-cohort check outcomes",
                "all 682 final per-record decisions",
                "input, runtime-validation, and decision timing",
                "complete committed target manifest and atomic lineage payload",
            ],
        },
        "failure_contract": {
            "mode": "copy_on_write",
            "representative_gpu_count": [2, 4],
            "capacity_evidence": {
                "required": True,
                "per_device_components": [
                    "model_and_program_bytes",
                    "old_kv_bytes",
                    "complete_new_kv_bytes",
                    "transient_bytes",
                    "allocator_margin_bytes",
                    "capacity_bytes",
                ],
                "arithmetic": (
                    "required_bytes equals the sum of all five required "
                    "components and does not exceed capacity_bytes"
                ),
                "device_count_matches_representative_gpu_count": True,
                "zero_model_or_program_bytes_forbidden": True,
            },
            "reader_state": {
                "before_commit": (
                    "the logical current pointer remains on the previously "
                    "committed version; the private target is not visible"
                ),
                "after_commit": (
                    "one atomic manifest transition exposes the complete "
                    "post-append target and its lineage hash"
                ),
                "after_abort": (
                    "every old expected record remains readback-valid and "
                    "private target staging is reclaimed"
                ),
            },
            "cases": [
                {
                    "name": "semantic_theta0_theta1_program_perturbation",
                    "expected": (
                        "fixed preflight selects exact before execution and "
                        "one complete corrected target commits"
                    ),
                },
                {
                    "name": "mid_job",
                    "expected": (
                        "abort, reclaim staging, hide target, and read back "
                        "every old record with matching metadata and SHA-256"
                    ),
                },
                {
                    "name": "pre_commit",
                    "expected": (
                        "abort after complete private coverage with the same "
                        "full old-manifest readback contract"
                    ),
                },
            ],
            "artifact_mismatch_scope": "unit and smoke",
            "runtime_rework_required": False,
            "journal_or_resume_required": False,
        },
        "stage6_freeze_contract": {
            "protocol": STAGE6_PROTOCOL,
            "selected_candidate": STAGE49_SELECTED_CANDIDATE,
            "cost_endpoint": STAGE49_COST_ENDPOINT,
            "selection_basis": (
                "freeze the preregistered bounded-renewal candidate "
                "without using recommendation labels; retain token debt "
                "only as the cost endpoint"
            ),
            "old_gpu_matrix_rerun": False,
            "required_lifecycle_evidence": [
                "Stage-4.6 fixed-history depth-four policy and chain",
                "Stage-4.9 corrected growing-history same-device candidates",
                "separately reported Stage-4.9 evaluator state movement",
                "no Stage-4.9 full-cohort HBM or end-to-end movement claim",
            ],
            "required_outputs": [
                "final aggregate",
                "correctness report",
                "timing and memory report",
                "paper table data",
                "paper figure data",
                "artifact-to-claim map",
                "negative-results log",
                "target-manuscript TBD disposition",
                "code-snapshot manifest",
            ],
            "validation": (
                "atomic CPU-only assembly validates every upstream path, "
                "protocol, status, and SHA-256; binds the Stage-5 candidate "
                "to Stage 4.9; applies JSON Schema, Stage-5 cross-field "
                "validation, and whole-aggregate semantic checks"
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
                "paper table and figure data",
                "artifact-to-claim map",
                "target-manuscript TBD disposition",
                "committed destination manifests",
                "negative-results log",
                "repository commit and code-snapshot hash",
            ],
            "semantic_validation": (
                "JSON Schema enforces one run for every primary "
                "method/destination/GPU point, fixed and corrected lifecycle "
                "evidence, the minimal Stage-5 closure, Stage-6 output "
                "descriptors, and artifact-derived source-state accounting; "
                "the aggregator additionally verifies every source SHA-256, "
                "candidate binding, COW component-byte sums, exact 682-record "
                "abort readback, manifest/lineage identity, unique record IDs, "
                "source/action compatibility, claim coverage, and TBD "
                "disposition"
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
