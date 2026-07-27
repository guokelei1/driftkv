from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from motivation_validity import move_batch, seed_everything

from hstu_kvcache.data import collate_batch, load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    CompiledCacheAdapter,
    FidelityContract,
    MigrationActionSpec,
    capture_layerwise_state,
    capture_residual_hidden_suffix,
    capture_selective_contiguous_state,
    certify_action,
    migrate_compiled_low_rank_cache,
    migrate_fused_projection_cache,
    migrate_prefix_residual_from_hidden_suffix,
    migrate_selective_contiguous_cache,
    sample_relative_cache_error,
    selective_contiguous_intervals,
)
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.streaming import (
    close_distributed_runtime,
    init_distributed_runtime,
    load_checkpoint_model,
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

PROTOCOL = "cohortkv_single_config_stage1_frontier_v1"
PARENT_PROTOCOL = "cohortkv_single_config_full_chain_development_v1"
PROGRAM_PROTOCOL = "kuairand_long_context_4plus12_attention_weighted_search_v1"
DEFAULT_PREPARED = (
    "data/processed/kuairand_long_context_4plus12_exploration_v1.npz"
)
DEFAULT_TRAINING = (
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
DEFAULT_CHECKPOINTS = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
)
DEFAULT_PROGRAM_RESULT = (
    "results/motivation_scale/"
    "long_context_4plus12_attention_weighted_search_seed0.json"
)
DEFAULT_PROGRAM_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/attention_weighted_search"
)
DEFAULT_BLUEPRINT = "configs/cohortkv_single_config_v1/blueprint.json"
DEFAULT_MANIFEST = "configs/cohortkv_single_config_v1/workload_manifest.json"
DEFAULT_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage1_frontier_seed0.json"
)
WIDTHS = (2, 4, 6, 8, 12)
RESIDUAL_DEPTHS = (4, 8)
CACHE_VERSIONS = (0, 4, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--program-result", default=DEFAULT_PROGRAM_RESULT)
    parser.add_argument("--program-dir", default=DEFAULT_PROGRAM_DIR)
    parser.add_argument("--blueprint", default=DEFAULT_BLUEPRINT)
    parser.add_argument("--workload-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_samples(
    samples: list[dict],
    fit_users: int,
    selection_users: int,
    certificate_users: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    if fit_users + selection_users + certificate_users >= len(samples):
        raise ValueError("frozen roles must leave final-test users")
    order = np.random.default_rng(9151 + seed).permutation(len(samples))
    fit = [samples[index] for index in order[:fit_users]]
    selection = [
        samples[index]
        for index in order[fit_users : fit_users + selection_users]
    ]
    remaining = [
        samples[index]
        for index in order[fit_users + selection_users :]
    ]
    certificate_order = np.random.default_rng(27183 + seed).permutation(
        len(remaining)
    )
    certificate = [
        remaining[index]
        for index in certificate_order[:certificate_users]
    ]
    final_test = [
        remaining[index]
        for index in certificate_order[certificate_users:]
    ]
    return fit, selection, certificate, final_test


def prepare_unlabeled_sequence(history: dict, seq_len: int) -> dict:
    length = min(len(history["item_ids"]), seq_len)
    return {
        "item_ids": history["item_ids"][-length:],
        "behaviors": history["behaviors"][-length:],
        "time_deltas": history["time_deltas"][-length:],
    }


def label_free_eval_batches(
    samples: list[dict],
    seq_len: int,
    batch_size: int,
):
    usable = [
        sample
        for sample in samples
        if len(sample["history"]["item_ids"]) >= 2
    ]
    for start in range(0, len(usable), batch_size):
        selected = usable[start : start + batch_size]
        full_sequences = [
            prepare_unlabeled_sequence(sample["history"], seq_len)
            for sample in selected
        ]
        prefix_sequences = [
            {name: values[:-1] for name, values in sequence.items()}
            for sequence in full_sequences
        ]
        full = collate_batch(full_sequences, max_seq_len=seq_len)
        prefix = collate_batch(prefix_sequences, max_seq_len=seq_len - 1)
        suffix = {
            "item_ids": torch.tensor(
                [[sequence["item_ids"][-1]] for sequence in full_sequences],
                dtype=torch.long,
            ),
            "behaviors": torch.tensor(
                [[sequence["behaviors"][-1]] for sequence in full_sequences],
                dtype=torch.long,
            ),
            "time_deltas": torch.tensor(
                [[sequence["time_deltas"][-1]] for sequence in full_sequences],
                dtype=torch.float32,
            ),
        }
        yield selected, full, prefix, suffix


def validate_frozen_inputs(
    args: argparse.Namespace,
) -> tuple[dict, dict, dict, HSTUConfig, list[dict], dict[str, list[dict]]]:
    if args.batch_size != 4:
        raise ValueError("Stage 1 freezes batch size 4")
    if args.timing_repeats != 3:
        raise ValueError("Stage 1 freezes three timing repetitions")
    if args.bootstrap_samples != 1000:
        raise ValueError("Stage 1 freezes 1000 bootstrap samples")
    blueprint = json.loads(Path(args.blueprint).read_text())
    manifest = json.loads(Path(args.workload_manifest).read_text())
    training = json.loads(Path(args.training_result).read_text())
    if blueprint.get("protocol") != PARENT_PROTOCOL:
        raise ValueError("blueprint protocol mismatch")
    if manifest.get("parent_protocol") != PARENT_PROTOCOL:
        raise ValueError("workload manifest protocol mismatch")
    if training.get("protocol") != training_protocol_for_base_days(4):
        raise ValueError("training protocol mismatch")
    if training.get("status") != "complete":
        raise ValueError("training result is incomplete")
    frozen = blueprint["frozen_inputs"]
    if sha256(args.training_result) != frozen["training_result"]["sha256"]:
        raise ValueError("training result differs from the frozen blueprint")
    prepared_sha = sha256(args.prepared_data)
    if (
        prepared_sha != training["prepared_data"]["sha256"]
        or prepared_sha != frozen["prepared_data"]["sha256"]
    ):
        raise ValueError("prepared data differs from training")
    if training["model"] != blueprint["data_and_model"]["model"]:
        raise ValueError("training model differs from the frozen blueprint")
    if training["args"].get("seed") != blueprint["data_and_model"]["training_seed"]:
        raise ValueError("training seed differs from the frozen blueprint")
    manifest_frozen = frozen["workload_manifest"]
    if (
        sha256(args.workload_manifest) != manifest_frozen["file_sha256"]
        or manifest["content_sha256"] != manifest_frozen["content_sha256"]
    ):
        raise ValueError("workload manifest differs from the frozen blueprint")
    expected_checkpoints = {
        value["version"]: value["sha256"]
        for value in frozen["checkpoints"]
    }
    for version in (*CACHE_VERSIONS, 11):
        path = Path(args.checkpoint_dir) / f"theta_{version}.pt"
        if sha256(path) != expected_checkpoints[f"theta{version}"]:
            raise ValueError(f"theta{version} checkpoint differs from the blueprint")
    expected_programs = {
        value["source_version"]: value["selected_program"]["sha256"]
        for value in frozen["verified_programs"]
    }
    for version in CACHE_VERSIONS:
        if sha256(program_path(args, version)) != expected_programs[f"theta{version}"]:
            raise ValueError(f"theta{version} program differs from the blueprint")
    program_result = json.loads(Path(args.program_result).read_text())
    if (
        program_result.get("protocol") != PROGRAM_PROTOCOL
        or program_result.get("status") != "design_search_complete"
        or program_result["design"]["selection"]["selected_attention_mix"] != 1.0
    ):
        raise ValueError("compiled program selection artifact mismatch")
    plan_data, metadata = load_prepared_kuairand_plan(args.prepared_data)
    validate_long_context_plan(plan_data, metadata, 4)
    cfg = HSTUConfig(**training["model"])
    if cfg.num_layers != 16 or cfg.max_seq_len != 2048:
        raise ValueError("Stage 1 model shape differs from the blueprint")
    date, samples = reconstruct_online_eval_samples(
        plan_data,
        (11,),
        1000,
    )[11]
    if date != manifest["evaluation_endpoint"]["date"]:
        raise ValueError("evaluation date differs from the workload manifest")
    roles = split_samples(samples, 40, 60, 60, 0)
    role_names = ("fit", "program_selection", "certificate", "final_test")
    role_samples = dict(zip(role_names, roles, strict=True))
    expected = {
        role: {
            int(record["user_id"])
            for record in manifest["records"]
            if record["evaluation_role"] == role
        }
        for role in role_names
    }
    actual = {
        role: {
            int(sample["history"]["user_id"])
            for sample in selected
        }
        for role, selected in role_samples.items()
    }
    if actual != expected:
        raise ValueError("reconstructed evaluation roles differ from the manifest")
    records_by_user = {
        int(record["user_id"]): record
        for record in manifest["records"]
    }
    for sample in samples:
        user_id = int(sample["history"]["user_id"])
        record = records_by_user[user_id]
        history_length = min(
            len(sample["history"]["item_ids"]),
            cfg.max_seq_len,
        )
        if history_length != record["history_length"]:
            raise ValueError("history length differs from the workload manifest")
    if selective_contiguous_intervals(cfg.num_layers, WIDTHS) != tuple(
        tuple(interval)
        for width in WIDTHS
        for interval in blueprint["action_contracts"]["selective_contiguous"][
            "candidate_intervals_by_m"
        ][str(width)]
    ):
        raise ValueError("selective interval grid differs from the blueprint")
    return (
        blueprint,
        manifest,
        training,
        cfg,
        samples,
        role_samples,
    )


def program_path(args: argparse.Namespace, cache_version: int) -> Path:
    return Path(args.program_dir) / (
        f"theta{cache_version}_to_theta11_"
        "compiled_attention_mix_1.00.pt"
    )


def load_compiled_program(
    args: argparse.Namespace,
    cache_version: int,
    cfg: HSTUConfig,
    device: torch.device,
) -> tuple[CompiledCacheAdapter, dict]:
    path = program_path(args, cache_version)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROGRAM_PROTOCOL:
        raise ValueError("compiled program protocol mismatch")
    if payload.get("source_version") != f"theta{cache_version}":
        raise ValueError("compiled program source version mismatch")
    if payload.get("target_version") != "theta11":
        raise ValueError("compiled program target version mismatch")
    weights = payload["weights"]
    biases = payload["biases"]
    inner = cfg.num_heads * cfg.head_dim
    if weights.shape != (cfg.num_layers, cfg.hidden_size, 2 * inner):
        raise ValueError("compiled program weight shape mismatch")
    if biases.shape != (cfg.num_layers, 2 * inner):
        raise ValueError("compiled program bias shape mismatch")
    if payload["fit"].get("fit_users") != 40:
        raise ValueError("compiled program fit split mismatch")
    if payload["fit"].get("labels_used") is not False:
        raise ValueError("compiled program must be label-free")
    return (
        CompiledCacheAdapter(
            weights=weights.to(device),
            biases=biases.to(device),
            source_rank=cfg.hidden_size,
            ridge=float(payload["ridge"]),
        ),
        payload,
    )


def timed_call(
    function,
    device: torch.device,
    repeats: int,
) -> tuple[HSTUKVCache, float]:
    function()
    torch.cuda.synchronize(device)
    samples = []
    value = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    if value is None:
        raise RuntimeError("timed action produced no cache")
    return value, float(np.median(samples))


def action_definitions(num_layers: int) -> list[dict]:
    definitions = [
        {
            "name": "reuse",
            "method": "reuse",
            "configuration": {},
        },
        {
            "name": "cheap_projection",
            "method": "cheap_projection",
            "configuration": {},
        },
        {
            "name": "compiled",
            "method": "compiled",
            "configuration": {"program": "attention_mix_1.00"},
        },
        {
            "name": "residual_p4",
            "method": "residual_p",
            "configuration": {"p": 4},
        },
        {
            "name": "residual_p8",
            "method": "residual_p",
            "configuration": {"p": 8},
        },
    ]
    for start, end in selective_contiguous_intervals(num_layers, WIDTHS):
        width = end - start + 1
        definitions.append(
            {
                "name": f"selective_m{width}_s{start}_e{end}",
                "method": "selective_contiguous",
                "configuration": {
                    "m": width,
                    "start_layer": start,
                    "end_layer": end,
                },
            }
        )
    definitions.append(
        {
            "name": "exact",
            "method": "exact",
            "configuration": {"compute_dtype": "float32"},
        }
    )
    if len(definitions) != 59:
        raise RuntimeError("Stage 1 action grid must contain 59 points")
    return definitions


@torch.inference_mode()
def semantic_values(
    model: HSTU,
    cache: HSTUKVCache,
    suffix: dict,
    candidate_ids: torch.Tensor,
    fresh_hidden: torch.Tensor,
    fresh_scores: torch.Tensor,
) -> dict[str, list[float]]:
    hidden, _ = model.forward_with_cache(
        cache,
        suffix["item_ids"],
        suffix["behaviors"],
        suffix["time_deltas"],
    )
    hidden = hidden[:, 0]
    scores = model.item_emb.score(hidden, candidate_ids)
    hidden_cosine = torch.nn.functional.cosine_similarity(
        hidden.float(),
        fresh_hidden.float(),
        dim=-1,
    )
    score_cosine = torch.nn.functional.cosine_similarity(
        scores.float(),
        fresh_scores.float(),
        dim=-1,
    )
    topk = min(100, scores.shape[1])
    fresh_top = torch.topk(fresh_scores, topk, dim=1).indices
    action_top = torch.topk(scores, topk, dim=1).indices
    overlap = (
        (action_top.unsqueeze(2) == fresh_top.unsqueeze(1))
        .any(dim=2)
        .float()
        .mean(dim=1)
    )
    return {
        "hidden_cosine": hidden_cosine.cpu().tolist(),
        "score_cosine": score_cosine.cpu().tolist(),
        "top100_overlap": overlap.cpu().tolist(),
    }


def build_actions(
    definitions: list[dict],
    current: HSTU,
    old_state,
    prefix: dict,
    compiled: CompiledCacheAdapter,
) -> dict[str, object]:
    residual_states = {
        depth: capture_residual_hidden_suffix(old_state, depth)
        for depth in RESIDUAL_DEPTHS
        if any(
            definition["name"] == f"residual_p{depth}"
            for definition in definitions
        )
    }
    selective_starts = {
        definition["configuration"]["start_layer"]
        for definition in definitions
        if definition["method"] == "selective_contiguous"
    }
    selective_states = {
        start: capture_selective_contiguous_state(old_state, start)
        for start in selective_starts
    }
    actions = {
        "reuse": lambda: old_state.kv,
        "cheap_projection": lambda: migrate_fused_projection_cache(
            current,
            old_state,
        ),
        "compiled": lambda: migrate_compiled_low_rank_cache(
            old_state,
            compiled,
        ),
        "exact": lambda: current.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        ),
    }
    for depth, state in residual_states.items():
        actions[f"residual_p{depth}"] = (
            lambda state=state: migrate_prefix_residual_from_hidden_suffix(
                current,
                state,
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
            )
        )
    for definition in definitions:
        if definition["method"] != "selective_contiguous":
            continue
        configuration = definition["configuration"]
        state = selective_states[configuration["start_layer"]]
        end_layer = configuration["end_layer"]
        actions[definition["name"]] = (
            lambda state=state, end_layer=end_layer: (
                migrate_selective_contiguous_cache(
                    current,
                    state,
                    prefix["item_ids"],
                    prefix["behaviors"],
                    prefix["time_deltas"],
                    end_layer,
                )
            )
        )
    missing = {
        definition["name"]
        for definition in definitions
        if definition["name"] not in actions
    }
    if missing:
        raise RuntimeError(f"missing action builders: {sorted(missing)}")
    return actions


@torch.inference_mode()
def evaluate_actions(
    current: HSTU,
    old: HSTU,
    compiled: CompiledCacheAdapter,
    samples: list[dict],
    definitions: list[dict],
    args: argparse.Namespace,
    device: torch.device,
    cache_version: int,
    role: str,
) -> tuple[list[dict], dict]:
    names = [definition["name"] for definition in definitions]
    timing = {name: 0.0 for name in names}
    records = []
    all_items = torch.arange(
        1,
        current.cfg.num_prediction_items + 1,
        device=device,
    )
    ordered = sorted(
        samples,
        key=lambda sample: (
            min(
                len(sample["history"]["item_ids"]),
                current.cfg.max_seq_len,
            ),
            int(sample["history"]["user_id"]),
        ),
    )
    batches = 0
    for selected, full_cpu, prefix_cpu, suffix_cpu in label_free_eval_batches(
        ordered,
        current.cfg.max_seq_len,
        args.batch_size,
    ):
        full = move_batch(full_cpu, device)
        prefix = move_batch(prefix_cpu, device)
        suffix = move_batch(suffix_cpu, device)
        full_output, _ = current(
            full["item_ids"],
            full["behaviors"],
            full["time_deltas"],
            lengths=full["lengths"],
        )
        fresh_hidden = current.last_hidden(full_output, full["lengths"])
        candidate_ids = all_items.unsqueeze(0).expand(len(selected), -1)
        fresh_scores = current.item_emb.score(fresh_hidden, candidate_ids)
        old_state = capture_layerwise_state(
            old,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            prefix["lengths"],
        )
        fresh_cache = current.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        )
        actions = build_actions(
            definitions,
            current,
            old_state,
            prefix,
            compiled,
        )
        batch_values = {}
        for name in names:
            if name == "reuse":
                cache = old_state.kv
                elapsed_ms = 0.0
            else:
                cache, elapsed_ms = timed_call(
                    actions[name],
                    device,
                    args.timing_repeats,
                )
            timing[name] += elapsed_ms
            values = semantic_values(
                current,
                cache,
                suffix,
                candidate_ids,
                fresh_hidden,
                fresh_scores,
            )
            values["cache_error_rel"] = (
                sample_relative_cache_error(cache, fresh_cache).cpu().tolist()
            )
            batch_values[name] = values
            if name not in {"reuse"}:
                del cache
        for row, sample in enumerate(selected):
            records.append(
                {
                    "user_id": int(sample["history"]["user_id"]),
                    "history_length": int(full["lengths"][row].item()),
                    "prefix_tokens": int(prefix["lengths"][row].item()),
                    "evaluation_role": role,
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
                        for name, values in batch_values.items()
                    },
                }
            )
        batches += 1
        print(
            json.dumps(
                {
                    "cache_version": cache_version,
                    "role": role,
                    "batch": batches,
                    "batches_total": math.ceil(len(ordered) / args.batch_size),
                }
            ),
            flush=True,
        )
    return records, {
        "milliseconds": timing,
        "users": len(records),
        "batches": batches,
        "batch_order": "history_length_then_user_id",
        "warmup_per_action_per_batch": 1,
        "measured_repetitions": args.timing_repeats,
        "aggregation": "sum_of_per_batch_median_cuda_event_latency",
    }


def recovery(
    records: list[dict],
    action_name: str,
    metric: str,
) -> float:
    def error(record: dict, name: str) -> float:
        config = record["configs"][name]
        if metric == "cache":
            return config["cache_error_rel"]
        if metric == "score":
            return max(0.0, 1.0 - config["score_cosine"])
        if metric == "top100":
            return max(0.0, 1.0 - config["top100_overlap"])
        raise ValueError("unsupported fidelity metric")

    reuse = np.asarray(
        [error(record, "reuse") for record in records],
        dtype=np.float64,
    )
    action = np.asarray(
        [error(record, action_name) for record in records],
        dtype=np.float64,
    )
    exact = np.asarray(
        [error(record, "exact") for record in records],
        dtype=np.float64,
    )
    denominator = float(reuse.mean() - exact.mean())
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return float("nan")
    return float((reuse.mean() - action.mean()) / denominator)


def summarize_actions(
    records: list[dict],
    timing: dict,
) -> dict[str, dict]:
    users = timing["users"]
    if users != len(records) or users < 1:
        raise ValueError("timing and semantic record counts differ")
    names = list(records[0]["configs"])
    exact_ms = timing["milliseconds"]["exact"] / users
    output = {}
    for name in names:
        cache_recovery = recovery(records, name, "cache")
        score_recovery = recovery(records, name, "score")
        top100_recovery = recovery(records, name, "top100")
        output[name] = {
            "cache_error_rel": float(
                np.mean(
                    [
                        record["configs"][name]["cache_error_rel"]
                        for record in records
                    ]
                )
            ),
            "hidden_cosine": float(
                np.mean(
                    [
                        record["configs"][name]["hidden_cosine"]
                        for record in records
                    ]
                )
            ),
            "score_cosine": float(
                np.mean(
                    [
                        record["configs"][name]["score_cosine"]
                        for record in records
                    ]
                )
            ),
            "top100_overlap": float(
                np.mean(
                    [
                        record["configs"][name]["top100_overlap"]
                        for record in records
                    ]
                )
            ),
            "cache_recovery": cache_recovery,
            "score_recovery": score_recovery,
            "top100_recovery": top100_recovery,
            "worst_view_recovery": min(
                cache_recovery,
                score_recovery,
                top100_recovery,
            ),
            "migration_ms_per_user": timing["milliseconds"][name] / users,
            "cost_ratio_to_exact": (
                timing["milliseconds"][name]
                / users
                / max(exact_ms, 1e-12)
            ),
        }
    return output


def select_width_winners(
    definitions: list[dict],
    summary: dict[str, dict],
) -> list[dict]:
    winners = []
    for width in WIDTHS:
        candidates = [
            definition
            for definition in definitions
            if definition["method"] == "selective_contiguous"
            and definition["configuration"]["m"] == width
        ]
        selected = min(
            candidates,
            key=lambda definition: (
                -summary[definition["name"]]["worst_view_recovery"],
                summary[definition["name"]]["cost_ratio_to_exact"],
                definition["configuration"]["start_layer"],
            ),
        )
        winners.append(selected)
    return winners


def source_components(
    definition: dict,
    prefix_tokens: int,
    records: int,
    cfg: HSTUConfig,
) -> dict[str, int]:
    fp16 = 2
    inner = cfg.num_heads * cfg.head_dim
    raw = prefix_tokens * (8 + 8 + 4) + records * 8
    if definition["method"] in {"compiled", "cheap_projection"}:
        return {
            "normalized_capsule_fp16": (
                prefix_tokens * cfg.num_layers * cfg.hidden_size * fp16
            )
        }
    if definition["method"] in {"reuse"}:
        return {
            "old_kv_fp16": (
                prefix_tokens * cfg.num_layers * 2 * inner * fp16
            )
        }
    if definition["method"] == "selective_contiguous":
        start = definition["configuration"]["start_layer"]
        return {
            "old_kv_fp16": (
                prefix_tokens * cfg.num_layers * 2 * inner * fp16
            ),
            "transition_hidden_fp16": (
                prefix_tokens * cfg.hidden_size * fp16 if start > 0 else 0
            ),
            "raw_history": raw,
        }
    if definition["method"] == "residual_p":
        depth = definition["configuration"]["p"]
        return {
            "residual_hidden_suffix_fp16": (
                prefix_tokens
                * (cfg.num_layers - depth)
                * cfg.hidden_size
                * fp16
            ),
            "raw_history": raw,
        }
    if definition["method"] == "exact":
        return {"raw_history": raw}
    raise ValueError("unsupported frontier method")


def frontier_points(
    cache_version: int,
    definitions: list[dict],
    summary: dict[str, dict],
    records: list[dict],
    cfg: HSTUConfig,
) -> list[dict]:
    prefix_tokens = sum(record["prefix_tokens"] for record in records)
    output = []
    for definition in definitions:
        values = summary[definition["name"]]
        components = source_components(
            definition,
            prefix_tokens,
            len(records),
            cfg,
        )
        output.append(
            {
                "source_version": f"theta{cache_version}",
                "target_version": "theta11",
                "evaluation_role": "program_selection",
                "method": definition["method"],
                "configuration": definition["configuration"],
                "cost_ratio_to_exact": values["cost_ratio_to_exact"],
                "migration_ms_per_user": values["migration_ms_per_user"],
                "fidelity": {
                    "cache_recovery": values["cache_recovery"],
                    "score_cosine": values["score_cosine"],
                    "top100_overlap": values["top100_overlap"],
                    "score_recovery": values["score_recovery"],
                    "top100_recovery": values["top100_recovery"],
                    "worst_view_recovery": values["worst_view_recovery"],
                },
                "logical_source_components": components,
                "logical_source_bytes": sum(components.values()),
                "physical_source_bytes": None,
            }
        )
    return output


def certify_winners(
    cache_version: int,
    winners: list[dict],
    selection_summary: dict[str, dict],
    certificate_records: list[dict],
    contract: FidelityContract,
) -> dict:
    verification_records = [
        {
            **record,
            "configs": {
                **record["configs"],
                "recompute": record["configs"]["exact"],
            },
        }
        for record in certificate_records
    ]
    actions = [
        MigrationActionSpec(
            name=winner["name"],
            kind="selective_contiguous",
            required_state=(
                "old_kv_and_raw_history"
                if winner["configuration"]["start_layer"] == 0
                else "old_kv_transition_hidden_and_raw_history"
            ),
        )
        for winner in winners
    ]
    actions.append(
        MigrationActionSpec(
            name="recompute",
            kind="exact",
            required_state="raw_history",
        )
    )
    certificates = [
        certify_action(
            verification_records,
            action,
            (
                1.0
                if action.name == "recompute"
                else selection_summary[action.name]["cost_ratio_to_exact"]
            ),
            contract,
            seed=cache_version * 10007 + index * 1009,
        )
        for index, action in enumerate(actions)
    ]
    passing = [
        certificate
        for certificate in certificates
        if certificate.action_name != "recompute"
        and certificate.fidelity_passed
        and certificate.budget_passed
    ]
    if passing:
        selected_certificate = min(
            passing,
            key=lambda certificate: (
                certificate.cost_ratio,
                -certificate.worst_recovery_lower_bound,
                certificate.action_name,
            ),
        )
        selected_name = selected_certificate.action_name
        reason = "minimum_cost_frozen_interval_passing_primary_contract"
        selected_action = next(
            winner
            for winner in winners
            if winner["name"] == selected_name
        )
        passed = True
    else:
        selected_name = "recompute"
        reason = "exact_fallback_no_selective_interval_passed_primary_contract"
        selected_action = None
        passed = False
    return {
        "source_version": f"theta{cache_version}",
        "target_version": "theta11",
        "action": (
            None
            if selected_action is None
            else selected_action["configuration"]
        ),
        "action_name": selected_name,
        "certificate_passed": passed,
        "selection_reason": reason,
        "certificates": [
            certificate.to_dict()
            for certificate in certificates
        ],
    }


def evaluate_pair(
    args: argparse.Namespace,
    cfg: HSTUConfig,
    current: HSTU,
    selection_samples: list[dict],
    certificate_samples: list[dict],
    cache_version: int,
    device: torch.device,
) -> dict:
    started = time.perf_counter()
    old = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        cache_version,
        device,
    )
    compiled, program = load_compiled_program(
        args,
        cache_version,
        cfg,
        device,
    )
    definitions = action_definitions(cfg.num_layers)
    selection_records, selection_timing = evaluate_actions(
        current,
        old,
        compiled,
        selection_samples,
        definitions,
        args,
        device,
        cache_version,
        "program_selection",
    )
    selection_summary = summarize_actions(
        selection_records,
        selection_timing,
    )
    winners = select_width_winners(definitions, selection_summary)
    profiled_definition = max(
        (
            definition
            for definition in definitions
            if definition["method"] == "selective_contiguous"
        ),
        key=lambda definition: (
            selection_summary[definition["name"]][
                "worst_view_recovery"
            ],
            -selection_summary[definition["name"]][
                "cost_ratio_to_exact"
            ],
            -definition["configuration"]["start_layer"],
        ),
    )
    profiled_source_representations = ["old_kv_fp16", "raw_history"]
    if profiled_definition["configuration"]["start_layer"] > 0:
        profiled_source_representations.insert(
            1,
            "transition_hidden_fp16",
        )
    certificate_definitions = [
        definitions[0],
        *winners,
        definitions[-1],
    ]
    certificate_records, certificate_timing = evaluate_actions(
        current,
        old,
        compiled,
        certificate_samples,
        certificate_definitions,
        args,
        device,
        cache_version,
        "certificate",
    )
    certificate_summary = summarize_actions(
        certificate_records,
        certificate_timing,
    )
    contract = FidelityContract(
        recovery_target=0.7,
        minimum_coverage=0.8,
        confidence_level=0.9,
        max_cost_ratio=0.3,
        bootstrap_samples=args.bootstrap_samples,
        minimum_probe_users=50,
    )
    certified = certify_winners(
        cache_version,
        winners,
        selection_summary,
        certificate_records,
        contract,
    )
    intervals = {
        (
            definition["configuration"]["start_layer"],
            definition["configuration"]["end_layer"],
        )
        for definition in definitions
        if definition["method"] == "selective_contiguous"
    }
    result = {
        "cache_version": cache_version,
        "source_version": f"theta{cache_version}",
        "target_version": "theta11",
        "program": {
            "path": str(program_path(args, cache_version)),
            "sha256": sha256(program_path(args, cache_version)),
            "protocol": program["protocol"],
        },
        "selection_points": frontier_points(
            cache_version,
            definitions,
            selection_summary,
            selection_records,
            cfg,
        ),
        "selective_grid_audit": {
            "source_version": f"theta{cache_version}",
            "expected_unique_intervals": 53,
            "observed_unique_intervals": len(intervals),
            "complete": len(intervals) == 53,
        },
        "width_winners": [
            {
                "configuration": winner["configuration"],
                **selection_summary[winner["name"]],
            }
            for winner in winners
        ],
        "profiled_selective_action": {
            "source_version": f"theta{cache_version}",
            "target_version": "theta11",
            "action": profiled_definition["configuration"],
            "action_name": profiled_definition["name"],
            "certificate_passed": False,
            "publishable_sync_action": False,
            "system_role": "frozen_diagnostic_external_baseline",
            "source_representations": profiled_source_representations,
            **selection_summary[profiled_definition["name"]],
        },
        "certified_selective_action": certified,
        "selection": {
            "summary": selection_summary,
            "timing": selection_timing,
            "per_user": selection_records,
        },
        "certificate": {
            "summary": certificate_summary,
            "timing": certificate_timing,
            "per_user": certificate_records,
            "contract": contract.to_dict(),
        },
        "resident_measurement": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_bytes": torch.cuda.get_device_properties(
                device
            ).total_memory,
            "compute_dtype": "float32",
            "state_dtype": "float32",
            "boundary": (
                "resident action compute and output assembly only; source reads, "
                "serialization, destination allocation, and publication excluded"
            ),
            "deployed_representation_note": (
                "logical source bytes use the frozen FP16 representations; "
                "Stage 2 must recertify deployed FP16 numeric paths"
            ),
        },
        "elapsed_setup_and_measurement_seconds": time.perf_counter() - started,
    }
    del old, compiled
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    (
        blueprint,
        manifest,
        training,
        cfg,
        samples,
        role_samples,
    ) = validate_frozen_inputs(args)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "records": len(samples),
                    "roles": {
                        name: len(selected)
                        for name, selected in role_samples.items()
                    },
                    "intervals": len(
                        selective_contiguous_intervals(
                            cfg.num_layers,
                            WIDTHS,
                        )
                    ),
                    "status": "validated",
                },
                indent=2,
            ),
            flush=True,
        )
        return
    runtime = init_distributed_runtime(
        args.device,
        args.distributed_backend,
    )
    try:
        if runtime.device.type != "cuda":
            raise ValueError("Stage 1 measurement requires CUDA")
        if runtime.world_size not in {1, len(CACHE_VERSIONS)}:
            raise ValueError("Stage 1 requires one process or three pair workers")
        seed_everything(0)
        current = load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            11,
            runtime.device,
        )
        if runtime.world_size == 1:
            local_results = [
                evaluate_pair(
                    args,
                    cfg,
                    current,
                    role_samples["program_selection"],
                    role_samples["certificate"],
                    cache_version,
                    runtime.device,
                )
                for cache_version in CACHE_VERSIONS
            ]
        else:
            local_results = [
                evaluate_pair(
                    args,
                    cfg,
                    current,
                    role_samples["program_selection"],
                    role_samples["certificate"],
                    CACHE_VERSIONS[runtime.rank],
                    runtime.device,
                )
            ]
        if runtime.initialized:
            gathered = [None] * runtime.world_size if runtime.is_primary else None
            dist.gather_object(local_results, gathered, dst=0)
            pair_results = (
                [
                    pair
                    for shard in gathered
                    if shard is not None
                    for pair in shard
                ]
                if runtime.is_primary
                else []
            )
        else:
            pair_results = local_results
        if runtime.is_primary:
            pair_results.sort(key=lambda pair: pair["cache_version"])
            selection_points = [
                point
                for pair in pair_results
                for point in pair["selection_points"]
            ]
            grid_audit = [
                pair["selective_grid_audit"]
                for pair in pair_results
            ]
            certified_actions = [
                pair["certified_selective_action"]
                for pair in pair_results
            ]
            profiled_actions = [
                pair["profiled_selective_action"]
                for pair in pair_results
            ]
            if len(selection_points) != 177:
                raise RuntimeError("Stage 1 aggregate must contain 177 points")
            if any(not audit["complete"] for audit in grid_audit):
                raise RuntimeError("Stage 1 interval grid is incomplete")
            result = {
                "protocol": PROTOCOL,
                "parent_protocol": PARENT_PROTOCOL,
                "status": "stage1_complete",
                "study_stage": "single_configuration_seed0_development",
                "seed": 0,
                "labels_used": False,
                "blueprint": {
                    "path": args.blueprint,
                    "sha256": sha256(args.blueprint),
                },
                "workload_manifest": {
                    "path": args.workload_manifest,
                    "sha256": sha256(args.workload_manifest),
                    "content_sha256": manifest["content_sha256"],
                },
                "prepared_data": {
                    "path": args.prepared_data,
                    "sha256": sha256(args.prepared_data),
                },
                "training_result": {
                    "path": args.training_result,
                    "sha256": sha256(args.training_result),
                    "protocol": training["protocol"],
                },
                "role_counts": {
                    name: len(selected)
                    for name, selected in role_samples.items()
                },
                "final_test_evaluated": False,
                "selection_rule": (
                    "per width, maximize worst cache/score/top100 recovery on "
                    "program-selection users; break ties by lower measured GPU "
                    "cost and earlier start"
                ),
                "certificate_rule": (
                    "minimum-cost frozen interval passing the primary fidelity "
                    "and cost contract on certificate users; otherwise exact"
                ),
                "rq3_frontier": {
                    "selection_points": selection_points,
                    "selective_grid_audit": grid_audit,
                    "profiled_selective_actions": profiled_actions,
                    "certified_selective_actions": certified_actions,
                },
                "anchor_and_control_closure": {
                    "reuse": "old K/V returned unchanged at zero resident compute",
                    "exact": "current-model full prefix K/V recomputation",
                    "cheap_projection": "current K/V projection over old Norm(x)",
                    "residual_p": (
                        "raw history plus p-specific old hidden suffix; p4/p8 "
                        "measured"
                    ),
                    "no_transform": (
                        "semantically identical to reuse; its placement and "
                        "transaction cost is measured only at the common Stage 4 "
                        "destination boundary"
                    ),
                    "operator_paths": (
                        "existing FP32 reference, packed FP16, and fused FP16 "
                        "correctness remain separate Stage 3 operator evidence"
                    ),
                    "bucketing": (
                        "on/off is a Stage 4 independently tuned runtime control, "
                        "not part of resident RQ3 algorithm cost"
                    ),
                },
                "pairs": pair_results,
                "environment": {
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "world_size": runtime.world_size,
                    "devices": [
                        {
                            "index": index,
                            "name": torch.cuda.get_device_name(index),
                            "total_bytes": torch.cuda.get_device_properties(
                                index
                            ).total_memory,
                        }
                        for index in range(torch.cuda.device_count())
                    ],
                },
            }
            save_json(result, args.output)
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "selection_points": len(selection_points),
                        "certified_selective_actions": certified_actions,
                    },
                    indent=2,
                ),
                flush=True,
            )
        del current
    finally:
        close_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
