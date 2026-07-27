from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.data import collate_batch, load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    EXECUTABLE_PLAN_PROTOCOL,
    CompiledCacheAdapter,
    FidelityContract,
    MigrationActionSpec,
    MigrationCapsuleBatch,
    MigrationProgram,
    PackedMigrationOperator,
    ResidualHiddenSuffixState,
    capture_layerwise_state,
    compile_projection_cache_adapter,
    compile_verified_plan,
    load_executable_plan,
    load_runtime_program,
    migrate_prefix_residual_from_hidden_suffix,
    sample_relative_cache_error,
    sha256_file,
    write_runtime_program,
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

PROTOCOL = "cohortkv_single_config_stage2_compiler_v1"
PARENT_PROTOCOL = "cohortkv_single_config_full_chain_development_v1"
SOURCE_PROGRAM_PROTOCOL = (
    "kuairand_long_context_4plus12_attention_weighted_search_v1"
)
CERTIFICATE_SHARD_PROTOCOL = "cohortkv_stage2_certificate_shard_v1"
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
DEFAULT_RUNTIME_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/single_config_v1/stage2_runtime"
)
DEFAULT_PLAN_DIR = "configs/cohortkv_single_config_v1/stage2_plans"
DEFAULT_BLUEPRINT = "configs/cohortkv_single_config_v1/blueprint.json"
DEFAULT_MANIFEST = "configs/cohortkv_single_config_v1/workload_manifest.json"
DEFAULT_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage2_compiler_seed0.json"
)
DEFAULT_SHARD_ROOT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage2_certificate_shards"
)
SOURCE_VERSIONS = (0, 4, 10)
TARGET_VERSION = 11
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
PRIMARY_THRESHOLD = 0.7
STRUCTURAL_DEPTHS = (4, 8)
ACTION_NAMES = (
    "reuse",
    "projection_only",
    "compiled_full_affine",
    "structural_p4",
    "structural_p8",
    "recompute",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--program-result", default=DEFAULT_PROGRAM_RESULT)
    parser.add_argument("--program-dir", default=DEFAULT_PROGRAM_DIR)
    parser.add_argument("--runtime-program-dir", default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--plan-dir", default=DEFAULT_PLAN_DIR)
    parser.add_argument("--blueprint", default=DEFAULT_BLUEPRINT)
    parser.add_argument("--workload-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--certificate-shard-root", default=DEFAULT_SHARD_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "batch_size": (args.batch_size, 4),
        "timing_repeats": (args.timing_repeats, 3),
        "bootstrap_samples": (args.bootstrap_samples, 1000),
    }
    changed = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in frozen.items()
        if actual != expected
    }
    if changed:
        raise ValueError(f"Stage 2 frozen settings changed: {changed}")


def split_samples(
    samples: list[dict],
) -> dict[str, list[dict]]:
    order = np.random.default_rng(9151).permutation(len(samples))
    fit = [samples[index] for index in order[:40]]
    selection = [samples[index] for index in order[40:100]]
    remaining = [samples[index] for index in order[100:]]
    certificate_order = np.random.default_rng(27183).permutation(
        len(remaining)
    )
    certificate = [
        remaining[index]
        for index in certificate_order[:60]
    ]
    final_test = [
        remaining[index]
        for index in certificate_order[60:]
    ]
    return {
        "fit": fit,
        "program_selection": selection,
        "certificate": certificate,
        "final_test": final_test,
    }


def prepare_unlabeled_sequence(history: dict, seq_len: int) -> dict:
    length = min(len(history["item_ids"]), seq_len)
    return {
        "item_ids": history["item_ids"][-length:],
        "behaviors": history["behaviors"][-length:],
        "time_deltas": history["time_deltas"][-length:],
    }


def label_free_batches(
    samples: list[dict],
    seq_len: int,
    batch_size: int,
):
    ordered = sorted(
        samples,
        key=lambda sample: (
            min(len(sample["history"]["item_ids"]), seq_len),
            int(sample["history"]["user_id"]),
        ),
    )
    for start in range(0, len(ordered), batch_size):
        selected = ordered[start : start + batch_size]
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
        latest = {
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
        yield selected, full, prefix, latest


def source_program_path(args: argparse.Namespace, source_version: int) -> Path:
    return Path(args.program_dir) / (
        f"theta{source_version}_to_theta{TARGET_VERSION}_"
        "compiled_attention_mix_1.00.pt"
    )


def runtime_program_path(
    args: argparse.Namespace,
    source_version: int,
) -> Path:
    return Path(args.runtime_program_dir) / (
        f"theta{source_version}_to_theta{TARGET_VERSION}_runtime_fp16.pt"
    )


def plan_path(args: argparse.Namespace, source_version: int) -> Path:
    return Path(args.plan_dir) / (
        f"theta{source_version}_to_theta{TARGET_VERSION}_executable.json"
    )


def checkpoint_path(args: argparse.Namespace, version: int) -> Path:
    return Path(args.checkpoint_dir) / f"theta_{version}.pt"


def source_storage_preflight(
    root: Path,
    blueprint: dict,
    shard_root: str,
) -> dict:
    target = (root / shard_root).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    observed = json.loads(
        subprocess.check_output(
            [
                "findmnt",
                "-J",
                "-T",
                str(target.parent),
                "-o",
                "TARGET,SOURCE,FSTYPE",
            ],
            text=True,
        )
    )["filesystems"][0]
    expected = blueprint["source_contract"]["common_physical_tier"]
    if (
        observed["target"] != expected["mount"]
        or observed["source"] != expected["device"]
        or observed["fstype"] != expected["filesystem"]
    ):
        raise ValueError("certificate shard storage differs from the blueprint")
    parent_name = subprocess.check_output(
        ["lsblk", "-ndo", "PKNAME", observed["source"]],
        text=True,
    ).strip()
    model = subprocess.check_output(
        ["lsblk", "-ndo", "MODEL", f"/dev/{parent_name}"],
        text=True,
    ).strip()
    if model != expected["device_model"]:
        raise ValueError("certificate shard device model differs from the blueprint")
    free_bytes = shutil.disk_usage(target.parent).free
    minimum = expected["minimum_free_bytes_before_materialization"]
    if free_bytes < minimum:
        raise ValueError("certificate shard storage has insufficient free space")
    return {
        "mount": observed["target"],
        "device": observed["source"],
        "device_model": model,
        "filesystem": observed["fstype"],
        "free_bytes_before_materialization": free_bytes,
        "minimum_required_bytes": minimum,
        "temporary_shards_retained": False,
    }


def validate_frozen_inputs(
    args: argparse.Namespace,
    root: Path,
) -> tuple[dict, dict, dict, HSTUConfig, dict[str, list[dict]], dict]:
    blueprint = json.loads((root / args.blueprint).read_text())
    manifest = json.loads((root / args.workload_manifest).read_text())
    training = json.loads((root / args.training_result).read_text())
    if blueprint.get("protocol") != PARENT_PROTOCOL:
        raise ValueError("blueprint protocol mismatch")
    if manifest.get("parent_protocol") != PARENT_PROTOCOL:
        raise ValueError("workload manifest protocol mismatch")
    frozen = blueprint["frozen_inputs"]
    if (
        training.get("protocol") != training_protocol_for_base_days(4)
        or training.get("status") != "complete"
    ):
        raise ValueError("training result is invalid")
    if sha256_file(root / args.training_result) != frozen["training_result"][
        "sha256"
    ]:
        raise ValueError("training result differs from the blueprint")
    prepared_sha = sha256_file(root / args.prepared_data)
    if (
        prepared_sha != frozen["prepared_data"]["sha256"]
        or prepared_sha != training["prepared_data"]["sha256"]
    ):
        raise ValueError("prepared data differs from the blueprint")
    if training["model"] != blueprint["data_and_model"]["model"]:
        raise ValueError("model differs from the blueprint")
    if training["args"].get("seed") != 0:
        raise ValueError("Stage 2 freezes training seed zero")
    workload_frozen = frozen["workload_manifest"]
    if (
        sha256_file(root / args.workload_manifest)
        != workload_frozen["file_sha256"]
        or manifest["content_sha256"]
        != workload_frozen["content_sha256"]
    ):
        raise ValueError("workload manifest differs from the blueprint")
    checkpoint_hashes = {
        value["version"]: value["sha256"]
        for value in frozen["checkpoints"]
    }
    for version in (*SOURCE_VERSIONS, TARGET_VERSION):
        if (
            sha256_file(root / checkpoint_path(args, version))
            != checkpoint_hashes[f"theta{version}"]
        ):
            raise ValueError(f"theta{version} checkpoint hash mismatch")
    program_hashes = {
        value["source_version"]: value["selected_program"]["sha256"]
        for value in frozen["verified_programs"]
    }
    for version in SOURCE_VERSIONS:
        if (
            sha256_file(root / source_program_path(args, version))
            != program_hashes[f"theta{version}"]
        ):
            raise ValueError(f"theta{version} source program hash mismatch")
    program_result = json.loads((root / args.program_result).read_text())
    if (
        program_result.get("protocol") != SOURCE_PROGRAM_PROTOCOL
        or program_result.get("status") != "design_search_complete"
        or program_result["design"]["ridge"] != 0.001
        or program_result["design"]["selection"]["selected_attention_mix"]
        != 1.0
        or program_result["split"]
        != {
            "fit_users": 40,
            "probe_users": 60,
            "test_users": 582,
            "split_seed": 9151,
        }
    ):
        raise ValueError("source program selection artifact is invalid")
    plan_data, metadata = load_prepared_kuairand_plan(
        root / args.prepared_data
    )
    validate_long_context_plan(plan_data, metadata, 4)
    cfg = HSTUConfig(**training["model"])
    date, samples = reconstruct_online_eval_samples(
        plan_data,
        (TARGET_VERSION,),
        1000,
    )[TARGET_VERSION]
    if date != manifest["evaluation_endpoint"]["date"]:
        raise ValueError("evaluation endpoint differs from the manifest")
    roles = split_samples(samples)
    expected_ids = {
        role: {
            int(record["user_id"])
            for record in manifest["records"]
            if record["evaluation_role"] == role
        }
        for role in roles
    }
    actual_ids = {
        role: {
            int(sample["history"]["user_id"])
            for sample in selected
        }
        for role, selected in roles.items()
    }
    if actual_ids != expected_ids:
        raise ValueError("reconstructed roles differ from the manifest")
    storage = source_storage_preflight(
        root,
        blueprint,
        args.certificate_shard_root,
    )
    return blueprint, manifest, training, cfg, roles, storage


def load_source_program(
    path: Path,
    source_version: int,
    cfg: HSTUConfig,
) -> tuple[MigrationProgram, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_shape = (
        cfg.num_layers,
        cfg.hidden_size,
        2 * cfg.num_heads * cfg.head_dim,
    )
    if (
        payload.get("protocol") != SOURCE_PROGRAM_PROTOCOL
        or payload.get("source_version") != f"theta{source_version}"
        or payload.get("target_version") != f"theta{TARGET_VERSION}"
        or payload.get("ridge") != 0.001
        or payload["weights"].dtype != torch.float32
        or payload["biases"].dtype != torch.float32
        or tuple(payload["weights"].shape) != expected_shape
        or tuple(payload["biases"].shape)
        != (cfg.num_layers, expected_shape[-1])
        or payload["fit"].get("fit_users") != 40
        or payload["fit"].get("sampled_tokens_per_layer")
        != [8192] * cfg.num_layers
        or payload["fit"].get("attention_mixes")
        != [0.0, 0.25, 0.5, 0.75, 1.0]
        or payload["fit"].get("labels_used") is not False
    ):
        raise ValueError("source full-affine program is invalid")
    program = MigrationProgram(
        source_version=payload["source_version"],
        target_version=payload["target_version"],
        adapter=CompiledCacheAdapter(
            weights=payload["weights"],
            biases=payload["biases"],
            source_rank=cfg.hidden_size,
            ridge=float(payload["ridge"]),
        ),
    )
    return program, payload


def tensor_nbytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def shard_tensor_bytes(payload: dict) -> int:
    values = [
        payload["normed"],
        payload["old_k"],
        payload["old_v"],
        payload["hidden_suffix_p4"],
        *payload["prefix"].values(),
        *payload["full"].values(),
        *payload["latest"].values(),
    ]
    return sum(tensor_nbytes(value) for value in values)


def serialize_certificate_shard(
    path: Path,
    old: HSTU,
    selected: list[dict],
    full: dict,
    prefix: dict,
    latest: dict,
    source_version: int,
    record_ids_by_user: dict[int, int],
    device: torch.device,
) -> dict:
    materialize_started = time.perf_counter()
    prefix_gpu = {
        name: value.to(device)
        for name, value in prefix.items()
    }
    state = capture_layerwise_state(
        old,
        prefix_gpu["item_ids"],
        prefix_gpu["behaviors"],
        prefix_gpu["time_deltas"],
        prefix_gpu["lengths"],
    )
    residual_hidden_suffix_absmax = max(
        float(value.abs().max())
        for value in state.hidden_states[4:]
    )
    residual_hidden_suffix_fp16_overflow_values = sum(
        int(
            torch.count_nonzero(
                ~torch.isfinite(value)
                | (value.abs() > torch.finfo(torch.float16).max)
            )
        )
        for value in state.hidden_states[4:]
    )
    payload = {
        "protocol": CERTIFICATE_SHARD_PROTOCOL,
        "source_version": f"theta{source_version}",
        "target_version": f"theta{TARGET_VERSION}",
        "record_ids": [
            record_ids_by_user[int(sample["history"]["user_id"])]
            for sample in selected
        ],
        "user_ids": [
            int(sample["history"]["user_id"])
            for sample in selected
        ],
        "normed": torch.stack(state.normed_states)
        .to(device="cpu", dtype=torch.float16)
        .contiguous(),
        "old_k": state.kv.k.to(device="cpu", dtype=torch.float16).contiguous(),
        "old_v": state.kv.v.to(device="cpu", dtype=torch.float16).contiguous(),
        "hidden_suffix_p4": torch.stack(state.hidden_states[4:])
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous(),
        "prefix": {
            name: value.contiguous()
            for name, value in prefix.items()
        },
        "full": {
            name: value.contiguous()
            for name, value in full.items()
        },
        "latest": {
            name: value.contiguous()
            for name, value in latest.items()
        },
        "labels_used": False,
    }
    materialize_seconds = time.perf_counter() - materialize_started
    serialize_started = time.perf_counter()
    torch.save(payload, path)
    serialize_seconds = time.perf_counter() - serialize_started
    descriptor = {
        "sha256": sha256_file(path),
        "physical_bytes": path.stat().st_size,
        "logical_tensor_bytes": shard_tensor_bytes(payload),
        "records": len(selected),
        "valid_prefix_tokens": int(prefix["lengths"].sum()),
        "residual_hidden_suffix_dtype": "bfloat16",
        "residual_hidden_suffix_absmax": residual_hidden_suffix_absmax,
        "residual_hidden_suffix_fp16_overflow_values": (
            residual_hidden_suffix_fp16_overflow_values
        ),
        "materialize_seconds": materialize_seconds,
        "serialize_seconds": serialize_seconds,
    }
    del state, prefix_gpu, payload
    torch.cuda.empty_cache()
    return descriptor


def validate_zero_padding(
    tensor: torch.Tensor,
    lengths: torch.Tensor,
) -> None:
    positions = torch.arange(tensor.shape[2])
    invalid = positions.unsqueeze(0) >= lengths.unsqueeze(1)
    mask = invalid.unsqueeze(0).unsqueeze(-1).expand_as(tensor)
    if bool(torch.count_nonzero(tensor.masked_select(mask))):
        raise ValueError("serialized certificate tensor has nonzero padding")


def load_certificate_shard(
    path: Path,
    descriptor: dict,
    source_version: int,
    cfg: HSTUConfig,
) -> tuple[dict, float]:
    if sha256_file(path) != descriptor["sha256"]:
        raise ValueError("certificate shard hash mismatch")
    started = time.perf_counter()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    elapsed = time.perf_counter() - started
    prefix = payload["prefix"]
    batch, sequence = prefix["item_ids"].shape
    expected = (
        cfg.num_layers,
        batch,
        sequence,
        cfg.hidden_size,
    )
    if (
        payload.get("protocol") != CERTIFICATE_SHARD_PROTOCOL
        or payload.get("source_version") != f"theta{source_version}"
        or payload.get("target_version") != f"theta{TARGET_VERSION}"
        or payload.get("labels_used") is not False
        or len(payload["record_ids"]) != batch
        or len(set(payload["record_ids"])) != batch
        or tuple(payload["normed"].shape) != expected
        or tuple(payload["old_k"].shape) != expected
        or tuple(payload["old_v"].shape) != expected
        or tuple(payload["hidden_suffix_p4"].shape)
        != (
            cfg.num_layers - 4,
            batch,
            sequence,
            cfg.hidden_size,
        )
    ):
        raise ValueError("certificate shard structure is invalid")
    for name in ("normed", "old_k", "old_v"):
        value = payload[name]
        finite = bool(torch.isfinite(value).all())
        if value.dtype != torch.float16 or not value.is_contiguous() or not finite:
            raise ValueError(
                "certificate shard tensor contract failed: "
                f"name={name}, dtype={value.dtype}, "
                f"contiguous={value.is_contiguous()}, finite={finite}"
            )
        validate_zero_padding(value, prefix["lengths"])
    hidden_suffix = payload["hidden_suffix_p4"]
    hidden_suffix_finite = bool(torch.isfinite(hidden_suffix).all())
    if (
        hidden_suffix.dtype != torch.bfloat16
        or not hidden_suffix.is_contiguous()
        or not hidden_suffix_finite
    ):
        raise ValueError(
            "certificate shard tensor contract failed: "
            f"name=hidden_suffix_p4, dtype={hidden_suffix.dtype}, "
            f"contiguous={hidden_suffix.is_contiguous()}, "
            f"finite={hidden_suffix_finite}"
        )
    validate_zero_padding(hidden_suffix, prefix["lengths"])
    if set(prefix) != {
        "item_ids",
        "behaviors",
        "time_deltas",
        "lengths",
    }:
        raise ValueError("certificate shard raw prefix is invalid")
    if "labels" in payload["full"] or "labels" in payload["latest"]:
        raise ValueError("certificate shard contains recommendation labels")
    return payload, elapsed


def fp16_cache(cache: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.to(torch.float16),
        v=cache.v.to(torch.float16),
        seq_len=cache.seq_len,
    )


def fp32_cache(cache: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.float(),
        v=cache.v.float(),
        seq_len=cache.seq_len,
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
        raise RuntimeError("timed action did not produce K/V")
    return value, float(np.median(samples))


def timed_catalog_score(
    model: HSTU,
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    scores = model.item_emb.score(hidden, candidate_ids)
    top = torch.topk(scores, min(100, scores.shape[1]), dim=1).indices
    end.record()
    end.synchronize()
    return scores, top, start.elapsed_time(end)


@torch.inference_mode()
def semantic_values(
    model: HSTU,
    cache: HSTUKVCache,
    latest: dict,
    candidate_ids: torch.Tensor,
    fresh_hidden: torch.Tensor,
    fresh_scores: torch.Tensor,
    fresh_top: torch.Tensor,
) -> tuple[dict[str, list[float]], float]:
    hidden, _ = model.forward_with_cache(
        fp32_cache(cache),
        latest["item_ids"],
        latest["behaviors"],
        latest["time_deltas"],
    )
    hidden = hidden[:, 0]
    scores, action_top, score_ms = timed_catalog_score(
        model,
        hidden,
        candidate_ids,
    )
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
    }, score_ms


def validate_output_cache(
    cache: HSTUKVCache,
    lengths: torch.Tensor,
) -> None:
    if (
        cache.k.dtype != torch.float16
        or cache.v.dtype != torch.float16
        or not bool(torch.isfinite(cache.k).all())
        or not bool(torch.isfinite(cache.v).all())
    ):
        raise ValueError("deployed output cache is not finite FP16")
    positions = torch.arange(cache.seq_len, device=lengths.device)
    invalid = positions.unsqueeze(0) >= lengths.unsqueeze(1)
    mask = invalid.unsqueeze(0).unsqueeze(-1).expand_as(cache.k)
    if bool(torch.count_nonzero(cache.k.masked_select(mask))) or bool(
        torch.count_nonzero(cache.v.masked_select(mask))
    ):
        raise ValueError("deployed output cache has nonzero padding")


def action_specs(runtime_path: str) -> tuple[MigrationActionSpec, ...]:
    return (
        MigrationActionSpec(
            name="projection_only",
            kind="projection",
            required_state="normalized_capsule_fp16",
        ),
        MigrationActionSpec(
            name="compiled_full_affine",
            kind="compiled",
            required_state="normalized_capsule_fp16",
            program_path=runtime_path,
        ),
        MigrationActionSpec(
            name="structural_p4",
            kind="structural_replay",
            required_state="raw_history_and_residual_hidden_suffix_bf16",
            replay_depth=4,
        ),
        MigrationActionSpec(
            name="structural_p8",
            kind="structural_replay",
            required_state="raw_history_and_residual_hidden_suffix_bf16",
            replay_depth=8,
        ),
        MigrationActionSpec(
            name="recompute",
            kind="exact",
            required_state="raw_history",
        ),
    )


def source_representations() -> dict[str, list[str]]:
    return {
        "projection_only": ["normalized_capsule_fp16"],
        "compiled_full_affine": ["normalized_capsule_fp16"],
        "structural_p4": [
            "raw_history",
            "residual_hidden_suffix_p4_bf16",
        ],
        "structural_p8": [
            "raw_history",
            "residual_hidden_suffix_p8_bf16",
        ],
        "recompute": ["raw_history"],
    }


def evaluate_loaded_shard(
    payload: dict,
    current: HSTU,
    compiled_program: MigrationProgram,
    projection_program: MigrationProgram,
    device: torch.device,
    timing_repeats: int,
) -> tuple[list[dict], dict]:
    prefix = {
        name: value.to(device)
        for name, value in payload["prefix"].items()
    }
    full = {
        name: value.to(device)
        for name, value in payload["full"].items()
    }
    latest = {
        name: value.to(device)
        for name, value in payload["latest"].items()
    }
    normed = payload["normed"].to(device)
    lengths = prefix["lengths"]
    capsule = MigrationCapsuleBatch(
        record_ids=tuple(payload["record_ids"]),
        migration_anchor_version=payload["source_version"],
        normed=normed,
        lengths=lengths,
    )
    old_cache = HSTUKVCache(
        k=payload["old_k"].to(device),
        v=payload["old_v"].to(device),
        seq_len=prefix["item_ids"].shape[1],
    )
    hidden_suffix = payload["hidden_suffix_p4"].to(device)
    residual_states = {
        depth: ResidualHiddenSuffixState(
            hidden_states=tuple(
                hidden_suffix[depth - 4 :].unbind(0)
            ),
            lengths=lengths,
            start_layer=depth,
            num_layers=len(current.blocks),
        )
        for depth in STRUCTURAL_DEPTHS
    }
    packed = PackedMigrationOperator(torch.float16)
    compiled_gpu = packed.prepare_program(compiled_program, device)
    projection_gpu = packed.prepare_program(projection_program, device)
    actions = {
        "reuse": lambda: old_cache,
        "projection_only": lambda: packed.execute(
            projection_gpu,
            capsule,
        ).cache,
        "compiled_full_affine": lambda: packed.execute(
            compiled_gpu,
            capsule,
        ).cache,
        "structural_p4": lambda: fp16_cache(
            migrate_prefix_residual_from_hidden_suffix(
                current,
                residual_states[4],
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
            )
        ),
        "structural_p8": lambda: fp16_cache(
            migrate_prefix_residual_from_hidden_suffix(
                current,
                residual_states[8],
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
            )
        ),
        "recompute": lambda: fp16_cache(
            current.compute_kv(
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
                lengths=lengths,
            )
        ),
    }
    full_hidden, _ = current(
        full["item_ids"],
        full["behaviors"],
        full["time_deltas"],
        lengths=full["lengths"],
    )
    fresh_hidden = current.last_hidden(full_hidden, full["lengths"])
    candidate_ids = torch.arange(
        1,
        current.cfg.num_prediction_items + 1,
        device=device,
    ).unsqueeze(0).expand(len(payload["record_ids"]), -1)
    fresh_scores, fresh_top, fresh_score_ms = timed_catalog_score(
        current,
        fresh_hidden,
        candidate_ids,
    )
    fresh_cache = current.compute_kv(
        prefix["item_ids"],
        prefix["behaviors"],
        prefix["time_deltas"],
        lengths=lengths,
    )
    timing = {name: 0.0 for name in ACTION_NAMES}
    score_timing = {"fresh": fresh_score_ms, **{name: 0.0 for name in ACTION_NAMES}}
    values = {}
    for name in ACTION_NAMES:
        if name == "reuse":
            cache = old_cache
            elapsed_ms = 0.0
        else:
            cache, elapsed_ms = timed_call(
                actions[name],
                device,
                timing_repeats,
            )
        validate_output_cache(cache, lengths)
        timing[name] += elapsed_ms
        semantic, score_ms = semantic_values(
            current,
            cache,
            latest,
            candidate_ids,
            fresh_hidden,
            fresh_scores,
            fresh_top,
        )
        semantic["cache_error_rel"] = (
            sample_relative_cache_error(cache, fresh_cache).cpu().tolist()
        )
        values[name] = semantic
        score_timing[name] += score_ms
    records = [
        {
            "record_id": int(payload["record_ids"][row]),
            "user_id": int(payload["user_ids"][row]),
            "evaluation_role": "certificate",
            "prefix_tokens": int(lengths[row]),
            "configs": {
                name: {
                    metric: float(values[name][metric][row])
                    for metric in (
                        "cache_error_rel",
                        "hidden_cosine",
                        "score_cosine",
                        "top100_overlap",
                    )
                }
                for name in ACTION_NAMES
            },
        }
        for row in range(len(payload["record_ids"]))
    ]
    return records, {
        "migration_milliseconds": timing,
        "full_catalog_score_milliseconds": score_timing,
        "users": len(records),
    }


def error(record: dict, action: str, metric: str) -> float:
    values = record["configs"][action]
    if metric == "cache":
        return values["cache_error_rel"]
    if metric == "score":
        return max(0.0, 1.0 - values["score_cosine"])
    if metric == "top100":
        return max(0.0, 1.0 - values["top100_overlap"])
    raise ValueError("unsupported metric")


def recovery(records: list[dict], action: str, metric: str) -> float:
    reuse = np.asarray(
        [error(record, "reuse", metric) for record in records],
        dtype=np.float64,
    )
    action_values = np.asarray(
        [error(record, action, metric) for record in records],
        dtype=np.float64,
    )
    exact = np.asarray(
        [error(record, "recompute", metric) for record in records],
        dtype=np.float64,
    )
    denominator = float(reuse.mean() - exact.mean())
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return float("nan")
    return float((reuse.mean() - action_values.mean()) / denominator)


def summarize_actions(
    records: list[dict],
    migration_milliseconds: dict[str, float],
) -> dict[str, dict]:
    users = len(records)
    exact_ms = migration_milliseconds["recompute"] / users
    output = {}
    for name in ACTION_NAMES:
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
            "migration_ms_per_user": migration_milliseconds[name] / users,
            "cost_ratio_to_exact": (
                migration_milliseconds[name]
                / users
                / max(exact_ms, 1e-12)
            ),
        }
    return output


def certificate_for_threshold(
    source_version: int,
    actions: tuple[MigrationActionSpec, ...],
    records: list[dict],
    costs: dict[str, float],
    threshold: float,
    bootstrap_samples: int,
) -> dict:
    contract = FidelityContract(
        recovery_target=threshold,
        minimum_coverage=0.8,
        confidence_level=0.9,
        max_cost_ratio=0.3,
        bootstrap_samples=bootstrap_samples,
        minimum_probe_users=50,
    )
    return compile_verified_plan(
        protocol=PROTOCOL,
        source_version=f"theta{source_version}",
        target_version=f"theta{TARGET_VERSION}",
        actions=actions,
        records=records,
        cost_ratios=costs,
        contract=contract,
        seed=source_version * 10007,
    ).to_dict()


def build_executable_plan(
    root: Path,
    args: argparse.Namespace,
    training: dict,
    manifest: dict,
    source_version: int,
    source_program: Path,
    runtime_descriptor: dict,
    primary_plan: dict,
    threshold_plans: list[dict],
    action_summary: dict,
    certificate_cost: dict,
    shard_summary: dict,
    fit_seconds: float,
    compile_seconds: float,
    runtime_load_seconds: float,
) -> tuple[dict, Path]:
    selected = primary_plan["selected_action"]
    selected_certificate = next(
        value
        for value in primary_plan["certificates"]
        if value["action_name"] == selected
    )
    selected_values = action_summary[selected]
    fallback = primary_plan["fallback_actions"]
    frozen_inputs = {
        "source_checkpoint": {
            "path": str(checkpoint_path(args, source_version)),
            "sha256": sha256_file(
                root / checkpoint_path(args, source_version)
            ),
        },
        "target_checkpoint": {
            "path": str(checkpoint_path(args, TARGET_VERSION)),
            "sha256": sha256_file(
                root / checkpoint_path(args, TARGET_VERSION)
            ),
        },
        "source_program": {
            "path": str(source_program),
            "sha256": sha256_file(root / source_program),
        },
        "role_manifest": {
            "path": args.workload_manifest,
            "sha256": sha256_file(root / args.workload_manifest),
        },
        "training_result": {
            "path": args.training_result,
            "sha256": sha256_file(root / args.training_result),
        },
        "prepared_data": {
            "path": args.prepared_data,
            "sha256": sha256_file(root / args.prepared_data),
        },
    }
    actions = primary_plan["actions"]
    payload = {
        "protocol": EXECUTABLE_PLAN_PROTOCOL,
        "status": "executable",
        "parent_protocol": PROTOCOL,
        "source_version": f"theta{source_version}",
        "target_version": f"theta{TARGET_VERSION}",
        "labels_used": False,
        "model": training["model"],
        "frozen_inputs": frozen_inputs,
        "runtime_program": runtime_descriptor,
        "actions": actions,
        "source_representations": source_representations(),
        "selected_action": selected,
        "selection_reason": primary_plan["selection_reason"],
        "fallback_actions": fallback,
        "executable_fallback_actions": fallback,
        "contract": primary_plan["contract"],
        "certificates": primary_plan["certificates"],
        "threshold_sweep": [
            {
                "recovery_target": value["contract"]["recovery_target"],
                "selected_action": value["selected_action"],
                "selection_reason": value["selection_reason"],
                "fallback_actions": value["fallback_actions"],
                "certificates": value["certificates"],
            }
            for value in threshold_plans
        ],
        "deployed_representation_certificate": {
            "source_dtype": "float16",
            "residual_hidden_suffix_dtype": "bfloat16",
            "program_dtype": "float16",
            "output_dtype": "float16",
            "passed": (
                selected_certificate["fidelity_passed"]
                and selected_certificate["budget_passed"]
            ),
            "certificate_users": 60,
            "views": ["cache", "score", "top100"],
            "selected_certificate": selected_certificate,
            "selected_summary": selected_values,
            "serialized_source": shard_summary,
        },
        "compiler_cost": {
            "historical_fit_seconds": fit_seconds,
            "runtime_prepare_seconds": compile_seconds,
            "runtime_load_validation_seconds": runtime_load_seconds,
            **certificate_cost,
        },
        "workload_content_sha256": manifest["content_sha256"],
    }
    output_path = root / plan_path(args, source_version)
    save_json(payload, output_path)
    load_executable_plan(
        output_path,
        repository_root=root,
        verify_input_hashes=True,
    )
    return payload, output_path


def evaluate_pair(
    root: Path,
    args: argparse.Namespace,
    training: dict,
    manifest: dict,
    cfg: HSTUConfig,
    certificate_samples: list[dict],
    current: HSTU,
    source_version: int,
    device: torch.device,
) -> dict:
    pair_started = time.perf_counter()
    source_path = source_program_path(args, source_version)
    source_program, source_payload = load_source_program(
        root / source_path,
        source_version,
        cfg,
    )
    runtime_relative = runtime_program_path(args, source_version)
    compile_started = time.perf_counter()
    descriptor = write_runtime_program(
        source_program,
        root / runtime_relative,
        {
            "source_program": {
                "path": str(source_path),
                "sha256": sha256_file(root / source_path),
                "protocol": source_payload["protocol"],
            },
            "source_checkpoint_sha256": sha256_file(
                root / checkpoint_path(args, source_version)
            ),
            "target_checkpoint_sha256": sha256_file(
                root / checkpoint_path(args, TARGET_VERSION)
            ),
            "role_manifest_sha256": sha256_file(
                root / args.workload_manifest
            ),
            "attention_mix": 1.0,
            "ridge": 0.001,
            "fit_users": 40,
            "sampled_tokens_per_layer": [8192] * cfg.num_layers,
            "labels_used": False,
        },
    )
    descriptor["path"] = str(runtime_relative)
    compile_seconds = time.perf_counter() - compile_started
    load_started = time.perf_counter()
    compiled_program, actual_descriptor = load_runtime_program(
        root / runtime_relative,
        expected_sha256=descriptor["sha256"],
        expected_source_version=f"theta{source_version}",
        expected_target_version=f"theta{TARGET_VERSION}",
        expected_model=training["model"],
    )
    runtime_load_seconds = time.perf_counter() - load_started
    actual_descriptor["path"] = str(runtime_relative)
    projection_program = MigrationProgram(
        source_version=f"theta{source_version}",
        target_version=f"theta{TARGET_VERSION}",
        adapter=compile_projection_cache_adapter(current),
    ).to("cpu", dtype=torch.float16)
    old = load_checkpoint_model(
        cfg,
        str(root / args.checkpoint_dir),
        source_version,
        device,
    )
    record_ids_by_user = {
        int(record["user_id"]): int(record["record_id"])
        for record in manifest["records"]
    }
    records = []
    migration_milliseconds = {name: 0.0 for name in ACTION_NAMES}
    score_milliseconds = {
        "fresh": 0.0,
        **{name: 0.0 for name in ACTION_NAMES},
    }
    shard_descriptors = []
    deserialize_seconds = 0.0
    certificate_started = time.perf_counter()
    shard_parent = root / args.certificate_shard_root
    shard_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"theta{source_version}_",
        dir=shard_parent,
    ) as temporary:
        for batch_index, (selected, full, prefix, latest) in enumerate(
            label_free_batches(
                certificate_samples,
                cfg.max_seq_len,
                args.batch_size,
            )
        ):
            shard_path = Path(temporary) / f"batch_{batch_index:03d}.pt"
            shard_descriptor = serialize_certificate_shard(
                shard_path,
                old,
                selected,
                full,
                prefix,
                latest,
                source_version,
                record_ids_by_user,
                device,
            )
            loaded, elapsed = load_certificate_shard(
                shard_path,
                shard_descriptor,
                source_version,
                cfg,
            )
            deserialize_seconds += elapsed
            batch_records, timing = evaluate_loaded_shard(
                loaded,
                current,
                compiled_program,
                projection_program,
                device,
                args.timing_repeats,
            )
            records.extend(batch_records)
            for name, value in timing["migration_milliseconds"].items():
                migration_milliseconds[name] += value
            for name, value in timing[
                "full_catalog_score_milliseconds"
            ].items():
                score_milliseconds[name] += value
            shard_descriptors.append(shard_descriptor)
            del loaded
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "source_version": source_version,
                        "batch": batch_index + 1,
                        "batches_total": math.ceil(
                            len(certificate_samples) / args.batch_size
                        ),
                    }
                ),
                flush=True,
            )
    certificate_seconds = time.perf_counter() - certificate_started
    if len(records) != 60 or {
        record["user_id"] for record in records
    } != {
        int(sample["history"]["user_id"])
        for sample in certificate_samples
    }:
        raise ValueError("deployed certificate record coverage is incomplete")
    action_summary = summarize_actions(records, migration_milliseconds)
    actions = action_specs(str(runtime_relative))
    costs = {
        action.name: action_summary[action.name]["cost_ratio_to_exact"]
        for action in actions
    }
    threshold_plans = [
        certificate_for_threshold(
            source_version,
            actions,
            records,
            costs,
            threshold,
            args.bootstrap_samples,
        )
        for threshold in THRESHOLDS
    ]
    primary_plan = next(
        value
        for value in threshold_plans
        if value["contract"]["recovery_target"] == PRIMARY_THRESHOLD
    )
    aggregate_shard_hash = hashlib.sha256(
        "".join(
            descriptor["sha256"]
            for descriptor in shard_descriptors
        ).encode()
    ).hexdigest()
    shard_summary = {
        "protocol": CERTIFICATE_SHARD_PROTOCOL,
        "batches": len(shard_descriptors),
        "records": sum(value["records"] for value in shard_descriptors),
        "valid_prefix_tokens": sum(
            value["valid_prefix_tokens"]
            for value in shard_descriptors
        ),
        "logical_tensor_bytes": sum(
            value["logical_tensor_bytes"]
            for value in shard_descriptors
        ),
        "physical_bytes": sum(
            value["physical_bytes"]
            for value in shard_descriptors
        ),
        "aggregate_batch_sha256": aggregate_shard_hash,
        "residual_hidden_suffix_dtype": "bfloat16",
        "residual_hidden_suffix_absmax": max(
            value["residual_hidden_suffix_absmax"]
            for value in shard_descriptors
        ),
        "residual_hidden_suffix_fp16_overflow_values": sum(
            value["residual_hidden_suffix_fp16_overflow_values"]
            for value in shard_descriptors
        ),
        "materialize_seconds": sum(
            value["materialize_seconds"]
            for value in shard_descriptors
        ),
        "serialize_seconds": sum(
            value["serialize_seconds"]
            for value in shard_descriptors
        ),
        "deserialize_seconds": deserialize_seconds,
        "temporary_shards_retained": False,
    }
    certificate_cost = {
        "certificate_seconds": certificate_seconds,
        "full_catalog_score_seconds": (
            sum(score_milliseconds.values()) / 1000.0
        ),
        "resident_migration_seconds": (
            sum(migration_milliseconds.values()) / 1000.0
        ),
    }
    fit_seconds = float(source_payload["fit"]["elapsed_ms"]) / 1000.0
    executable, output_path = build_executable_plan(
        root,
        args,
        training,
        manifest,
        source_version,
        source_path,
        actual_descriptor,
        primary_plan,
        threshold_plans,
        action_summary,
        certificate_cost,
        shard_summary,
        fit_seconds,
        compile_seconds,
        runtime_load_seconds,
    )
    selected = executable["selected_action"]
    exact_ms = action_summary["recompute"]["migration_ms_per_user"]
    selected_ms = action_summary[selected]["migration_ms_per_user"]
    saved_seconds_per_record = max(0.0, exact_ms - selected_ms) / 1000.0
    one_time_seconds = (
        fit_seconds
        + compile_seconds
        + certificate_seconds
    )
    result = {
        "source_version": f"theta{source_version}",
        "target_version": f"theta{TARGET_VERSION}",
        "runtime_program": actual_descriptor,
        "executable_plan": {
            "path": str(plan_path(args, source_version)),
            "sha256": sha256_file(output_path),
            "protocol": executable["protocol"],
        },
        "selected_action": selected,
        "selection_reason": executable["selection_reason"],
        "fallback_actions": executable["fallback_actions"],
        "executable_fallback_actions": executable[
            "executable_fallback_actions"
        ],
        "actions": executable["actions"],
        "action_summary": action_summary,
        "primary_certificate": executable[
            "deployed_representation_certificate"
        ],
        "threshold_sweep": executable["threshold_sweep"],
        "compiler_cost": executable["compiler_cost"],
        "amortization": {
            "one_time_seconds": one_time_seconds,
            "resident_seconds_saved_per_record": saved_seconds_per_record,
            "resident_break_even_records": (
                math.ceil(one_time_seconds / saved_seconds_per_record)
                if saved_seconds_per_record > 0
                else None
            ),
            "boundary": (
                "resident algorithmic floor; Stage 4 must recompute "
                "end-to-end break-even with source reads and destination writes"
            ),
        },
        "certificate_records": records,
        "pair_wall_seconds": time.perf_counter() - pair_started,
    }
    del old, source_program, compiled_program, projection_program
    torch.cuda.empty_cache()
    return result


def aggregate_result(
    root: Path,
    args: argparse.Namespace,
    training: dict,
    manifest: dict,
    storage: dict,
    roles: dict[str, list[dict]],
    pairs: list[dict],
    world_size: int,
) -> dict:
    pairs.sort(
        key=lambda pair: SOURCE_VERSIONS.index(
            int(pair["source_version"].removeprefix("theta"))
        )
    )
    if [pair["source_version"] for pair in pairs] != [
        f"theta{version}" for version in SOURCE_VERSIONS
    ]:
        raise ValueError("Stage 2 source-version coverage is incomplete")
    one_time_seconds = sum(
        pair["amortization"]["one_time_seconds"]
        for pair in pairs
    )
    threshold_sweep = [
        {
            "recovery_target": threshold,
            "cohort_actions": [
                {
                    "source_version": pair["source_version"],
                    "selected_action": next(
                        value["selected_action"]
                        for value in pair["threshold_sweep"]
                        if value["recovery_target"] == threshold
                    ),
                }
                for pair in pairs
            ],
        }
        for threshold in THRESHOLDS
    ]
    cohorts = [
        {
            "source_version": pair["source_version"],
            "selected_action": pair["selected_action"],
            "fallback_actions": pair["fallback_actions"],
            "executable_fallback_actions": pair[
                "executable_fallback_actions"
            ],
            "program_bytes": pair["runtime_program"]["bytes"],
            "compile_seconds": pair["compiler_cost"][
                "runtime_prepare_seconds"
            ],
            "certificate_seconds": pair["compiler_cost"][
                "certificate_seconds"
            ],
            "selected_cost_ratio_to_exact": pair["action_summary"][
                pair["selected_action"]
            ]["cost_ratio_to_exact"],
            "fidelity": {
                "cache_recovery": pair["action_summary"][
                    pair["selected_action"]
                ]["cache_recovery"],
                "score_cosine": pair["action_summary"][
                    pair["selected_action"]
                ]["score_cosine"],
                "top100_overlap": pair["action_summary"][
                    pair["selected_action"]
                ]["top100_overlap"],
            },
            "deployed_representation_certificate": pair[
                "primary_certificate"
            ],
            "runtime_program": pair["runtime_program"],
            "executable_plan": pair["executable_plan"],
        }
        for pair in pairs
    ]
    return {
        "protocol": PROTOCOL,
        "parent_protocol": PARENT_PROTOCOL,
        "status": "stage2_complete",
        "study_stage": "single_configuration_seed0_development",
        "seed": 0,
        "labels_used": False,
        "final_test_evaluated": False,
        "role_counts": {
            role: len(selected)
            for role, selected in roles.items()
        },
        "blueprint": {
            "path": args.blueprint,
            "sha256": sha256_file(root / args.blueprint),
            "protocol": PARENT_PROTOCOL,
        },
        "workload_manifest": {
            "path": args.workload_manifest,
            "sha256": sha256_file(root / args.workload_manifest),
            "content_sha256": manifest["content_sha256"],
            "protocol": manifest["protocol"],
        },
        "workload_content_sha256": manifest["content_sha256"],
        "model": training["model"],
        "source_storage_preflight": storage,
        "frozen_hyperparameters": {
            "attention_mix": 1.0,
            "ridge": 0.001,
            "fit_users": 40,
            "sampled_tokens_per_layer": [8192] * training["model"]["num_layers"],
            "action_library": [
                "projection_only",
                "compiled_full_affine",
                "structural_p4",
                "structural_p8",
                "recompute",
            ],
            "primary_recovery_target": PRIMARY_THRESHOLD,
            "threshold_sweep": list(THRESHOLDS),
        },
        "rq2_compiler": {
            "cohorts": cohorts,
            "threshold_sweep": threshold_sweep,
            "compile_seconds": sum(
                pair["compiler_cost"]["runtime_prepare_seconds"]
                for pair in pairs
            ),
            "certificate_seconds": sum(
                pair["compiler_cost"]["certificate_seconds"]
                for pair in pairs
            ),
            "historical_fit_seconds": sum(
                pair["compiler_cost"]["historical_fit_seconds"]
                for pair in pairs
            ),
            "full_catalog_score_seconds": sum(
                pair["compiler_cost"]["full_catalog_score_seconds"]
                for pair in pairs
            ),
            "amortized_seconds_per_record": one_time_seconds / 682,
            "amortization_curve": [
                {
                    "cohort_records": records,
                    "one_time_seconds_per_record": one_time_seconds / records,
                }
                for records in (64, 682, 1000, 10000)
            ],
            "boundary": (
                "compiler/certificate resident amortization only; Stage 4 "
                "adds source materialization, reads, writes, and commit"
            ),
        },
        "pairs": pairs,
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "world_size": world_size,
            "gpu_name": torch.cuda.get_device_name(0),
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    root = Path(__file__).resolve().parents[1]
    (
        blueprint,
        manifest,
        training,
        cfg,
        roles,
        storage,
    ) = validate_frozen_inputs(args, root)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "roles": {
                        role: len(selected)
                        for role, selected in roles.items()
                    },
                    "source_versions": list(SOURCE_VERSIONS),
                    "target_version": TARGET_VERSION,
                    "storage": storage,
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
            raise ValueError("Stage 2 requires CUDA")
        if runtime.world_size not in {1, len(SOURCE_VERSIONS)}:
            raise ValueError("Stage 2 requires one or three workers")
        torch.manual_seed(0)
        np.random.seed(0)
        current = load_checkpoint_model(
            cfg,
            str(root / args.checkpoint_dir),
            TARGET_VERSION,
            runtime.device,
        )
        if runtime.world_size == 1:
            local_results = [
                evaluate_pair(
                    root,
                    args,
                    training,
                    manifest,
                    cfg,
                    roles["certificate"],
                    current,
                    source_version,
                    runtime.device,
                )
                for source_version in SOURCE_VERSIONS
            ]
        else:
            local_results = [
                evaluate_pair(
                    root,
                    args,
                    training,
                    manifest,
                    cfg,
                    roles["certificate"],
                    current,
                    SOURCE_VERSIONS[runtime.rank],
                    runtime.device,
                )
            ]
        if runtime.initialized:
            gathered = [None] * runtime.world_size if runtime.is_primary else None
            dist.gather_object(local_results, gathered, dst=0)
            pairs = (
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
            pairs = local_results
        if runtime.is_primary:
            result = aggregate_result(
                root,
                args,
                training,
                manifest,
                storage,
                roles,
                pairs,
                runtime.world_size,
            )
            save_json(result, root / args.output)
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "selected_actions": [
                            {
                                "source_version": pair["source_version"],
                                "selected_action": pair["selected_action"],
                                "fallback_actions": pair["fallback_actions"],
                            }
                            for pair in result["pairs"]
                        ],
                        "final_test_evaluated": False,
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
