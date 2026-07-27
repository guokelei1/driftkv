from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path

import benchmark_cohortkv_stage4_5_resident_ceiling as ceiling
import compile_cohortkv_stage2 as frozen
import numpy as np
import torch

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    FidelityContract,
    MigrationActionSpec,
    capture_layerwise_state,
    compile_verified_plan,
    sample_relative_cache_error,
)
from hstu_kvcache.migration.stage45_oldkv import (
    load_direct_oldkv_program,
)
from hstu_kvcache.models import HSTUConfig, HSTUKVCache
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_online_eval_samples,
    validate_long_context_plan,
)

PROTOCOL = "cohortkv_single_config_stage4_5_oldkv_certificate_v1"
DEFAULT_PREPARED = (
    "data/processed/kuairand_long_context_4plus12_exploration_v1.npz"
)
DEFAULT_COMPILER = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_compiler_seed0.json"
)
DEFAULT_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_5_oldkv_certificate_seed0.json"
)
SOURCE_INDICES = {"theta0": 0, "theta4": 4, "theta10": 10}
ACTION = "compiled_old_kv"
TIMING_REPEATS = 3
BOOTSTRAP_SAMPLES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--blueprint", default=ceiling.DEFAULT_BLUEPRINT)
    parser.add_argument(
        "--workload-manifest",
        default=ceiling.DEFAULT_WORKLOAD,
    )
    parser.add_argument("--stage2-summary", default=ceiling.DEFAULT_STAGE2)
    parser.add_argument("--training-result", default=ceiling.DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=ceiling.DEFAULT_CHECKPOINTS)
    parser.add_argument("--compiler-result", default=DEFAULT_COMPILER)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def implementation_snapshot(root: Path) -> dict[str, object]:
    paths = (
        Path("src/hstu_kvcache/migration/stage45_oldkv.py"),
        Path("scripts/evaluate_cohortkv_stage4_5_oldkv_certificate.py"),
    )
    files = [
        {
            "path": str(path),
            "bytes": (root / path).stat().st_size,
            "sha256": ceiling.sha256_file(root / path),
        }
        for path in paths
    ]
    return {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip(),
        "files": files,
        "code_snapshot_sha256": hashlib.sha256(
            ceiling.canonical_json(files)
        ).hexdigest(),
    }


def direct_cache(
    state,
    program,
    lengths: torch.Tensor,
) -> HSTUKVCache:
    old_k = state.kv.k.to(torch.float16)
    old_v = state.kv.v.to(torch.float16)
    joined = torch.cat((old_k, old_v), dim=-1)
    layers, batch, sequence, _ = joined.shape
    projected = torch.baddbmm(
        program.biases[:, None, :].expand(
            layers,
            batch * sequence,
            2 * program.kv_width,
        ),
        joined.flatten(1, 2),
        program.weights,
    ).unflatten(1, (batch, sequence))
    invalid = (
        torch.arange(sequence, device=lengths.device).unsqueeze(0)
        >= lengths.unsqueeze(1)
    )
    projected.masked_fill_(invalid[None, :, :, None], 0)
    return HSTUKVCache(
        k=projected[..., : program.kv_width],
        v=projected[..., program.kv_width :],
        seq_len=sequence,
    )


def semantic_for_cache(
    current,
    cache: HSTUKVCache,
    latest: dict[str, torch.Tensor],
    candidate_ids: torch.Tensor,
    fresh_hidden: torch.Tensor,
    fresh_scores: torch.Tensor,
    fresh_top: torch.Tensor,
    fresh_cache: HSTUKVCache,
) -> dict[str, list[float]]:
    values, _ = frozen.semantic_values(
        current,
        cache,
        latest,
        candidate_ids,
        fresh_hidden,
        fresh_scores,
        fresh_top,
    )
    values["cache_error_rel"] = (
        sample_relative_cache_error(cache, fresh_cache).cpu().tolist()
    )
    return values


def recompute_cache(
    current,
    prefix: dict[str, torch.Tensor],
) -> HSTUKVCache:
    return frozen.fp16_cache(
        current.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        )
    )


def evaluate_pair(
    current,
    source,
    program,
    certificate_samples: list[dict],
    record_ids_by_user: dict[int, int],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    records = []
    timings = {ACTION: 0.0, "recompute": 0.0}
    batches = math.ceil(len(certificate_samples) / 4)
    for batch_index, (selected, full, prefix, latest) in enumerate(
        frozen.label_free_batches(
            certificate_samples,
            current.cfg.max_seq_len,
            4,
        )
    ):
        prefix_gpu = {
            name: value.to(device) for name, value in prefix.items()
        }
        full_gpu = {
            name: value.to(device) for name, value in full.items()
        }
        latest_gpu = {
            name: value.to(device) for name, value in latest.items()
        }
        state = capture_layerwise_state(
            source,
            prefix_gpu["item_ids"],
            prefix_gpu["behaviors"],
            prefix_gpu["time_deltas"],
            prefix_gpu["lengths"],
        )
        actions = {
            ACTION: partial(
                direct_cache,
                state,
                program,
                prefix_gpu["lengths"],
            ),
            "recompute": partial(
                recompute_cache,
                current,
                prefix_gpu,
            ),
        }
        outputs = {}
        for name, function in actions.items():
            value, elapsed = frozen.timed_call(
                function,
                device,
                TIMING_REPEATS,
            )
            frozen.validate_output_cache(
                value,
                prefix_gpu["lengths"],
            )
            outputs[name] = value
            timings[name] += elapsed
        old_cache = HSTUKVCache(
            k=state.kv.k.to(torch.float16),
            v=state.kv.v.to(torch.float16),
            seq_len=state.kv.seq_len,
        )
        full_hidden, _ = current(
            full_gpu["item_ids"],
            full_gpu["behaviors"],
            full_gpu["time_deltas"],
            lengths=full_gpu["lengths"],
        )
        fresh_hidden = current.last_hidden(
            full_hidden,
            full_gpu["lengths"],
        )
        candidate_ids = torch.arange(
            1,
            current.cfg.num_prediction_items + 1,
            device=device,
        ).unsqueeze(0).expand(len(selected), -1)
        fresh_scores, fresh_top, _ = frozen.timed_catalog_score(
            current,
            fresh_hidden,
            candidate_ids,
        )
        fresh_cache = current.compute_kv(
            prefix_gpu["item_ids"],
            prefix_gpu["behaviors"],
            prefix_gpu["time_deltas"],
            lengths=prefix_gpu["lengths"],
        )
        semantic = {
            "reuse": semantic_for_cache(
                current,
                old_cache,
                latest_gpu,
                candidate_ids,
                fresh_hidden,
                fresh_scores,
                fresh_top,
                fresh_cache,
            ),
            ACTION: semantic_for_cache(
                current,
                outputs[ACTION],
                latest_gpu,
                candidate_ids,
                fresh_hidden,
                fresh_scores,
                fresh_top,
                fresh_cache,
            ),
            "recompute": semantic_for_cache(
                current,
                outputs["recompute"],
                latest_gpu,
                candidate_ids,
                fresh_hidden,
                fresh_scores,
                fresh_top,
                fresh_cache,
            ),
        }
        for row, sample in enumerate(selected):
            user_id = int(sample["history"]["user_id"])
            records.append(
                {
                    "record_id": record_ids_by_user[user_id],
                    "user_id": user_id,
                    "evaluation_role": "certificate",
                    "prefix_tokens": int(prefix_gpu["lengths"][row]),
                    "configs": {
                        name: {
                            metric: float(values[metric][row])
                            for metric in (
                                "cache_error_rel",
                                "hidden_cosine",
                                "score_cosine",
                                "top100_overlap",
                            )
                        }
                        for name, values in semantic.items()
                    },
                }
            )
        print(
            json.dumps(
                {
                    "source_version": program.source_version,
                    "batch": batch_index + 1,
                    "batches_total": batches,
                }
            ),
            flush=True,
        )
        del (
            state,
            outputs,
            semantic,
            old_cache,
            fresh_cache,
            full_hidden,
            fresh_hidden,
            fresh_scores,
            fresh_top,
        )
        torch.cuda.empty_cache()
    return records, timings


def error(record: dict, action: str, metric: str) -> float:
    values = record["configs"][action]
    if metric == "cache":
        return float(values["cache_error_rel"])
    if metric == "score":
        return max(0.0, 1.0 - float(values["score_cosine"]))
    if metric == "top100":
        return max(0.0, 1.0 - float(values["top100_overlap"]))
    raise ValueError("unsupported semantic metric")


def recovery(records: list[dict], action: str, metric: str) -> float:
    reuse = np.asarray(
        [error(value, "reuse", metric) for value in records],
        dtype=np.float64,
    )
    selected = np.asarray(
        [error(value, action, metric) for value in records],
        dtype=np.float64,
    )
    exact = np.asarray(
        [error(value, "recompute", metric) for value in records],
        dtype=np.float64,
    )
    denominator = float(reuse.mean() - exact.mean())
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return float("nan")
    return float((reuse.mean() - selected.mean()) / denominator)


def action_summary(
    records: list[dict],
    timings: dict[str, float],
) -> dict[str, object]:
    users = len(records)
    exact_per_user = timings["recompute"] / users
    return {
        "cache_recovery": recovery(records, ACTION, "cache"),
        "score_recovery": recovery(records, ACTION, "score"),
        "top100_recovery": recovery(records, ACTION, "top100"),
        "cache_error_rel": statistics.mean(
            value["configs"][ACTION]["cache_error_rel"]
            for value in records
        ),
        "score_cosine": statistics.mean(
            value["configs"][ACTION]["score_cosine"]
            for value in records
        ),
        "top100_overlap": statistics.mean(
            value["configs"][ACTION]["top100_overlap"]
            for value in records
        ),
        "migration_ms_per_user": timings[ACTION] / users,
        "exact_ms_per_user": exact_per_user,
        "cost_ratio_to_exact": (
            timings[ACTION] / users / max(exact_per_user, 1e-12)
        ),
    }


def certify(
    source_version: str,
    records: list[dict],
    summary: dict[str, object],
    program_path: str,
) -> dict[str, object]:
    contract = FidelityContract(
        recovery_target=0.7,
        minimum_coverage=0.8,
        confidence_level=0.9,
        max_cost_ratio=0.3,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        minimum_probe_users=50,
    )
    actions = (
        MigrationActionSpec(
            name=ACTION,
            kind="compiled",
            required_state="existing_old_kv_fp16",
            program_path=program_path,
        ),
        MigrationActionSpec(
            name="recompute",
            kind="exact",
            required_state="raw_history",
        ),
    )
    plan = compile_verified_plan(
        protocol=PROTOCOL,
        source_version=source_version,
        target_version="theta11",
        actions=actions,
        records=records,
        cost_ratios={
            ACTION: float(summary["cost_ratio_to_exact"]),
            "recompute": 1.0,
        },
        contract=contract,
        seed=SOURCE_INDICES[source_version] * 10007,
    )
    return plan.to_dict()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    blueprint = json.loads((root / args.blueprint).read_text())
    workload = json.loads((root / args.workload_manifest).read_text())
    stage2 = json.loads((root / args.stage2_summary).read_text())
    training = json.loads((root / args.training_result).read_text())
    compiler = json.loads((root / args.compiler_result).read_text())
    frozen_inputs = blueprint["frozen_inputs"]
    if (
        compiler.get("protocol")
        != "cohortkv_single_config_stage4_5_oldkv_compiler_v1"
        or compiler.get("status") != "oldkv_program_transport_frozen"
        or compiler.get("labels_used") is not False
        or compiler["inputs"]["workload_content_sha256"]
        != workload["content_sha256"]
        or ceiling.sha256_file(root / args.prepared_data)
        != frozen_inputs["prepared_data"]["sha256"]
        or ceiling.sha256_file(root / args.training_result)
        != frozen_inputs["training_result"]["sha256"]
        or ceiling.sha256_file(root / args.stage2_summary)
        != frozen_inputs["stage2_compiler_summary"]["sha256"]
    ):
        raise ValueError("direct old-K/V certificate inputs differ")
    plan_data, metadata = load_prepared_kuairand_plan(
        root / args.prepared_data
    )
    validate_long_context_plan(plan_data, metadata, 4)
    date, samples = reconstruct_online_eval_samples(
        plan_data,
        (11,),
        1000,
    )[11]
    if date != workload["evaluation_endpoint"]["date"]:
        raise ValueError("direct old-K/V evaluation endpoint differs")
    roles = frozen.split_samples(samples)
    certificate_samples = roles["certificate"]
    certificate_users = {
        int(value["history"]["user_id"]) for value in certificate_samples
    }
    expected_users = {
        int(value["user_id"])
        for value in workload["records"]
        if value["evaluation_role"] == "certificate"
    }
    if certificate_users != expected_users:
        raise ValueError("direct old-K/V certificate role differs")
    record_ids_by_user = {
        int(value["user_id"]): int(value["record_id"])
        for value in workload["records"]
    }
    config = HSTUConfig(**training["model"])
    device = torch.device(args.device)
    current = load_checkpoint_model(
        config,
        str(root / args.checkpoint_dir),
        11,
        device,
    ).eval()
    descriptors = {
        value["source_version"]: value
        for value in compiler["representation"]["programs"]
    }
    pairs = []
    for source_version, index in SOURCE_INDICES.items():
        descriptor = descriptors[source_version]
        program, _ = load_direct_oldkv_program(
            descriptor["path"],
            expected_sha256=descriptor["sha256"],
            expected_source_version=source_version,
            expected_target_version="theta11",
            expected_num_layers=config.num_layers,
            expected_kv_width=config.num_heads * config.head_dim,
        )
        program = program.to(device, dtype=torch.float16)
        source = load_checkpoint_model(
            config,
            str(root / args.checkpoint_dir),
            index,
            device,
        ).eval()
        records, timings = evaluate_pair(
            current,
            source,
            program,
            certificate_samples,
            record_ids_by_user,
            device,
        )
        if (
            len(records) != 60
            or {value["user_id"] for value in records}
            != certificate_users
        ):
            raise RuntimeError(
                "direct old-K/V certificate coverage differs"
            )
        summary = action_summary(records, timings)
        certificate = certify(
            source_version,
            records,
            summary,
            descriptor["path"],
        )
        if (
            certificate["selected_action"] != ACTION
            or certificate["fallback_actions"] != ["recompute"]
        ):
            raise RuntimeError("direct old-K/V semantic certificate failed")
        frozen_pair = next(
            value
            for value in stage2["pairs"]
            if value["source_version"] == source_version
        )
        pairs.append(
            {
                "source_version": source_version,
                "target_version": "theta11",
                "records": len(records),
                "prefix_tokens": sum(
                    int(value["prefix_tokens"]) for value in records
                ),
                "summary": summary,
                "certificate": certificate,
                "frozen_normalized_capsule_certificate": frozen_pair[
                    "selected_certificate"
                ],
                "program": {
                    "path": descriptor["path"],
                    "sha256": descriptor["sha256"],
                },
            }
        )
        del source, program, records
        torch.cuda.empty_cache()
    result = {
        "protocol": PROTOCOL,
        "status": "oldkv_semantic_certificate_frozen",
        "study_stage": "stage4_5_b_direct_oldkv_seed0",
        "seed": 0,
        "labels_used": False,
        "measurement_boundary": (
            "reconstructed certificate histories, deployed FP16 old K/V "
            "and direct program, three-view recovery against current exact"
        ),
        "inputs": {
            "compiler_result": {
                "path": args.compiler_result,
                "sha256": ceiling.sha256_file(
                    root / args.compiler_result
                ),
            },
            "workload_content_sha256": workload["content_sha256"],
            "prepared_data_sha256": ceiling.sha256_file(
                root / args.prepared_data
            ),
            "stage2_summary_sha256": ceiling.sha256_file(
                root / args.stage2_summary
            ),
        },
        "contract": {
            "recovery_target": 0.7,
            "minimum_coverage": 0.8,
            "confidence_level": 0.9,
            "max_cost_ratio": 0.3,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "minimum_probe_users": 50,
            "views": ["cache", "score", "top100"],
        },
        "pairs": pairs,
        "aggregate": {
            "all_selected_direct_oldkv": all(
                value["certificate"]["selected_action"] == ACTION
                for value in pairs
            ),
            "all_exact_fallback": all(
                value["certificate"]["fallback_actions"]
                == ["recompute"]
                for value in pairs
            ),
            "minimum_worst_view_recovery": min(
                min(
                    float(value["summary"][name])
                    for name in (
                        "cache_recovery",
                        "score_recovery",
                        "top100_recovery",
                    )
                )
                for value in pairs
            ),
            "maximum_cost_ratio_to_exact": max(
                float(value["summary"]["cost_ratio_to_exact"])
                for value in pairs
            ),
        },
        "implementation": implementation_snapshot(root),
        "last_invocation_seconds": time.perf_counter() - started,
    }
    ceiling.write_json_atomic(root / args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                **result["aggregate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
