from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hstu_kvcache.data import collate_batch, load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    SourceRecordDescriptor,
    Stage4SourceManifest,
    capture_layerwise_state,
    sha256_file,
    write_source_shard,
)
from hstu_kvcache.models import HSTUConfig
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)

PROTOCOL = "cohortkv_stage4_source_materialization_v1"
PARENT_PROTOCOL = "cohortkv_single_config_full_chain_development_v1"
DEFAULT_BLUEPRINT = "configs/cohortkv_single_config_v1/blueprint.json"
DEFAULT_WORKLOAD = "configs/cohortkv_single_config_v1/workload_manifest.json"
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
DEFAULT_OUTPUT = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0/"
    "single_config_v1/source_shards/source_manifest.json"
)
SOURCE_VERSIONS = ("theta0", "theta4", "theta10")
TARGET_VERSION = "theta11"
RESIDUAL_SOURCES = ("theta0", "theta10")
RESIDUAL_START_LAYER = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint", default=DEFAULT_BLUEPRINT)
    parser.add_argument("--workload-manifest", default=DEFAULT_WORKLOAD)
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture-batch-size", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def split_samples(samples: list[dict]) -> dict[str, list[dict]]:
    order = np.random.default_rng(9151).permutation(len(samples))
    fit = [samples[index] for index in order[:40]]
    selection = [samples[index] for index in order[40:100]]
    remaining = [samples[index] for index in order[100:]]
    certificate_order = np.random.default_rng(27183).permutation(
        len(remaining)
    )
    return {
        "fit": fit,
        "program_selection": selection,
        "certificate": [
            remaining[index] for index in certificate_order[:60]
        ],
        "final_test": [
            remaining[index] for index in certificate_order[60:]
        ],
    }


def unlabeled_prefix(history: dict, maximum_length: int) -> dict:
    length = min(len(history["item_ids"]), maximum_length)
    return {
        "item_ids": history["item_ids"][-length:-1],
        "behaviors": history["behaviors"][-length:-1],
        "time_deltas": history["time_deltas"][-length:-1],
    }


def checkpoint_path(
    root: Path,
    checkpoint_dir: str,
    version: str,
) -> Path:
    index = int(version.removeprefix("theta"))
    return root / checkpoint_dir / f"theta_{index}.pt"


def validate_inputs(
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict, dict, HSTUConfig, dict[int, dict]]:
    if args.capture_batch_size != 4:
        raise ValueError("Stage 4 source capture batch size is frozen at four")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Stage 4 source materialization requires CUDA")
    blueprint = json.loads((root / args.blueprint).read_text())
    workload = json.loads((root / args.workload_manifest).read_text())
    training = json.loads((root / args.training_result).read_text())
    blueprint_state = (
        blueprint.get("status"),
        tuple(blueprint.get("scope", {}).get("completed_stages", ())),
    )
    if (
        blueprint.get("protocol") != PARENT_PROTOCOL
        or blueprint_state
        not in {
            ("stage3_operator_frozen", (0, 1, 2, 3)),
            ("stage4_core_frozen", (0, 1, 2, 3, 4)),
        }
    ):
        raise ValueError("Stage 4 parent blueprint state is invalid")
    frozen = blueprint["frozen_inputs"]
    if (
        workload.get("protocol") != "cohortkv_single_config_workload_v1"
        or workload.get("status") != "frozen"
        or sha256_file(root / args.workload_manifest)
        != frozen["workload_manifest"]["file_sha256"]
        or workload["content_sha256"]
        != frozen["workload_manifest"]["content_sha256"]
    ):
        raise ValueError("Stage 4 workload manifest is invalid")
    if (
        training.get("protocol") != training_protocol_for_base_days(4)
        or training.get("status") != "complete"
        or sha256_file(root / args.training_result)
        != frozen["training_result"]["sha256"]
    ):
        raise ValueError("Stage 4 training result is invalid")
    if (
        sha256_file(root / args.prepared_data)
        != frozen["prepared_data"]["sha256"]
    ):
        raise ValueError("Stage 4 prepared data hash mismatch")
    checkpoint_hashes = {
        value["version"]: value for value in frozen["checkpoints"]
    }
    for source in SOURCE_VERSIONS:
        path = checkpoint_path(root, args.checkpoint_dir, source)
        descriptor = checkpoint_hashes[source]
        if (
            sha256_file(path) != descriptor["sha256"]
            or path.stat().st_size != descriptor["bytes"]
        ):
            raise ValueError(f"{source} checkpoint differs from Stage 0")
    cfg = HSTUConfig(**training["model"])
    if training["model"] != blueprint["data_and_model"]["model"]:
        raise ValueError("Stage 4 model configuration differs from Stage 0")
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
        raise ValueError("Stage 4 endpoint date differs from the workload")
    roles = split_samples(samples)
    sample_by_user = {
        int(sample["history"]["user_id"]): sample
        for selected in roles.values()
        for sample in selected
    }
    records_by_user = {
        int(record["user_id"]): record
        for record in workload["records"]
    }
    if (
        len(sample_by_user) != 682
        or set(sample_by_user) != set(records_by_user)
    ):
        raise ValueError("Stage 4 reconstructed workload identity is incomplete")
    actual_roles = {
        role: {
            int(sample["history"]["user_id"])
            for sample in selected
        }
        for role, selected in roles.items()
    }
    expected_roles = {
        role: {
            int(record["user_id"])
            for record in workload["records"]
            if record["evaluation_role"] == role
        }
        for role in roles
    }
    if actual_roles != expected_roles:
        raise ValueError("Stage 4 reconstructed workload roles differ")
    return blueprint, workload, cfg, sample_by_user


def source_preflight(
    root: Path,
    blueprint: dict,
    output: Path,
    workload: dict,
) -> dict[str, object]:
    contract = blueprint["source_contract"]["common_physical_tier"]
    shard_root = output.parent.resolve()
    declared_root = (
        root / blueprint["source_contract"]["shard_root"]
    ).resolve()
    if shard_root != declared_root:
        raise ValueError("Stage 4 shard root differs from the frozen contract")
    mount = Path(contract["mount"]).resolve()
    source, filesystem, target = subprocess.check_output(
        [
            "findmnt",
            "-n",
            "-o",
            "SOURCE,FSTYPE,TARGET",
            "--target",
            str(shard_root.parent),
        ],
        text=True,
    ).split()
    if (
        source != contract["device"]
        or filesystem != contract["filesystem"]
        or Path(target).resolve() != mount
    ):
        raise ValueError("Stage 4 source filesystem differs from Stage 0")
    block_device = source.rstrip("0123456789")
    if block_device.endswith("p"):
        block_device = block_device[:-1]
    model = subprocess.check_output(
        ["lsblk", "-dn", "-o", "MODEL", block_device],
        text=True,
    ).strip()
    if model != contract["device_model"]:
        raise ValueError("Stage 4 source device model differs from Stage 0")
    free_bytes = shutil.disk_usage(mount).free
    raw_logical_bytes = workload["summary"]["prefix_tokens"] * (
        torch.tensor([], dtype=torch.long).element_size() * 2
        + torch.tensor([], dtype=torch.float32).element_size()
    )
    source_contract = blueprint["source_contract"]["representations"]
    estimated_final_bytes = int(
        (
            source_contract["normalized_capsule_fp16"]["logical_bytes"]
            + source_contract["old_kv_fp16"]["logical_bytes"]
            + source_contract["residual_hidden_suffix_bf16"][
                "current_verified_p8_fallback_bytes"
            ]
            + raw_logical_bytes
        )
        * 1.05
    )
    minimum_free = int(contract["minimum_free_bytes_before_materialization"])
    if free_bytes < minimum_free or free_bytes < minimum_free + estimated_final_bytes:
        raise OSError("Stage 4 source materialization disk preflight failed")
    return {
        "protocol": PROTOCOL,
        "source": source,
        "filesystem": filesystem,
        "mount": str(mount),
        "device_model": model,
        "observed_free_bytes": free_bytes,
        "minimum_free_bytes": minimum_free,
        "estimated_final_and_serialization_bytes": estimated_final_bytes,
        "temporary_policy": "one atomic record shard at a time",
        "passed": True,
    }


def validate_existing(
    root: Path,
    args: argparse.Namespace,
    blueprint: dict,
    workload: dict,
) -> Stage4SourceManifest:
    path = root / args.output
    manifest = Stage4SourceManifest.load(path)
    if (
        manifest.workload_content_sha256 != workload["content_sha256"]
        or manifest.workload_file_sha256
        != sha256_file(root / args.workload_manifest)
        or manifest.record_count != workload["summary"]["records"]
        or manifest.prefix_tokens != workload["summary"]["prefix_tokens"]
        or manifest.num_layers
        != blueprint["data_and_model"]["model"]["num_layers"]
        or manifest.hidden_size
        != blueprint["data_and_model"]["model"]["hidden_size"]
        or manifest.kv_width
        != (
            blueprint["data_and_model"]["model"]["num_heads"]
            * blueprint["data_and_model"]["model"]["head_dim"]
        )
    ):
        raise ValueError("existing Stage 4 source manifest is invalid")
    for record, expected in zip(
        manifest.records,
        workload["records"],
        strict=True,
    ):
        if (
            record.record_id != expected["record_id"]
            or record.user_id != expected["user_id"]
            or record.evaluation_role != expected["evaluation_role"]
            or record.source_version != expected["source_version"]
            or record.target_version != expected["target_version"]
            or record.prefix_tokens != expected["prefix_tokens"]
        ):
            raise ValueError("existing Stage 4 source record differs")
        for descriptor in record.shards:
            shard = path.parent / descriptor.path
            if (
                not shard.is_file()
                or shard.stat().st_size != descriptor.physical_bytes
                or sha256_file(shard) != descriptor.sha256
            ):
                raise ValueError("existing Stage 4 source shard failed integrity")
    return manifest


@torch.no_grad()
def materialize(
    root: Path,
    args: argparse.Namespace,
    blueprint: dict,
    workload: dict,
    cfg: HSTUConfig,
    sample_by_user: dict[int, dict],
    preflight: dict[str, object],
) -> Stage4SourceManifest:
    started = time.perf_counter()
    output = root / args.output
    shard_root = output.parent
    records_by_source: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for record in workload["records"]:
        records_by_source[record["source_version"]].append(
            (record, sample_by_user[int(record["user_id"])])
        )
    descriptors: dict[int, SourceRecordDescriptor] = {}
    cohort_results = []
    device = torch.device(args.device)
    for source in SOURCE_VERSIONS:
        source_started = time.perf_counter()
        model = load_checkpoint_model(
            cfg,
            str(root / args.checkpoint_dir),
            int(source.removeprefix("theta")),
            device,
        )
        selected = sorted(
            records_by_source[source],
            key=lambda value: (
                int(value[0]["prefix_tokens"]),
                int(value[0]["record_id"]),
            ),
        )
        source_logical = defaultdict(int)
        source_physical = defaultdict(int)
        for start in range(0, len(selected), args.capture_batch_size):
            current = selected[start : start + args.capture_batch_size]
            prefixes = [
                unlabeled_prefix(sample["history"], cfg.max_seq_len)
                for _, sample in current
            ]
            host_batch = collate_batch(
                prefixes,
                max_seq_len=cfg.max_seq_len - 1,
            )
            gpu_batch = {
                name: value.to(device)
                for name, value in host_batch.items()
            }
            state = capture_layerwise_state(
                model,
                gpu_batch["item_ids"],
                gpu_batch["behaviors"],
                gpu_batch["time_deltas"],
                gpu_batch["lengths"],
            )
            normed = torch.stack(state.normed_states)
            residual_hidden = (
                torch.stack(state.hidden_states[RESIDUAL_START_LAYER:])
                if source in RESIDUAL_SOURCES
                else None
            )
            for row, (record, _) in enumerate(current):
                record_id = int(record["record_id"])
                length = int(gpu_batch["lengths"][row])
                if length != int(record["prefix_tokens"]):
                    raise ValueError("Stage 4 source prefix length differs")
                normalized_value = (
                    normed[:, row, :length]
                    .to(device="cpu", dtype=torch.float16)
                    .contiguous()
                )
                old_k = (
                    state.kv.k[:, row, :length]
                    .to(device="cpu", dtype=torch.float16)
                    .contiguous()
                )
                old_v = (
                    state.kv.v[:, row, :length]
                    .to(device="cpu", dtype=torch.float16)
                    .contiguous()
                )
                if not all(
                    bool(torch.isfinite(value).all())
                    for value in (normalized_value, old_k, old_v)
                ):
                    raise ValueError("Stage 4 FP16 source contains nonfinite values")
                normalized = write_source_shard(
                    shard_root,
                    f"normalized_capsule_fp16/{record_id:06d}.pt",
                    "normalized_capsule_fp16",
                    record_id,
                    source,
                    TARGET_VERSION,
                    length,
                    {"normed": normalized_value},
                )
                old_kv = write_source_shard(
                    shard_root,
                    f"old_kv_fp16/{record_id:06d}.pt",
                    "old_kv_fp16",
                    record_id,
                    source,
                    TARGET_VERSION,
                    length,
                    {"k": old_k, "v": old_v},
                )
                raw = write_source_shard(
                    shard_root,
                    f"raw_history/{record_id:06d}.pt",
                    "raw_history",
                    record_id,
                    source,
                    TARGET_VERSION,
                    length,
                    {
                        "item_ids": host_batch["item_ids"][
                            row, :length
                        ].contiguous(),
                        "behaviors": host_batch["behaviors"][
                            row, :length
                        ].contiguous(),
                        "time_deltas": host_batch["time_deltas"][
                            row, :length
                        ].to(torch.float32).contiguous(),
                    },
                )
                shards = [normalized, old_kv, raw]
                if residual_hidden is not None:
                    residual_value = (
                        residual_hidden[:, row, :length]
                        .to(device="cpu", dtype=torch.bfloat16)
                        .contiguous()
                    )
                    if not bool(torch.isfinite(residual_value).all()):
                        raise ValueError(
                            "Stage 4 BF16 residual source contains nonfinite values"
                        )
                    residual = write_source_shard(
                        shard_root,
                        (
                            "residual_hidden_suffix_bf16/"
                            f"{record_id:06d}.pt"
                        ),
                        "residual_hidden_suffix_bf16",
                        record_id,
                        source,
                        TARGET_VERSION,
                        length,
                        {"hidden_states": residual_value},
                        {
                            "start_layer": RESIDUAL_START_LAYER,
                            "num_layers": cfg.num_layers,
                        },
                    )
                    shards.append(residual)
                for shard in shards:
                    source_logical[shard.representation] += shard.logical_bytes
                    source_physical[shard.representation] += shard.physical_bytes
                descriptors[record_id] = SourceRecordDescriptor(
                    record_id=record_id,
                    user_id=int(record["user_id"]),
                    evaluation_role=str(record["evaluation_role"]),
                    source_version=source,
                    target_version=TARGET_VERSION,
                    prefix_tokens=length,
                    shards=tuple(shards),
                )
            del host_batch, gpu_batch, state, normed, residual_hidden
            print(
                json.dumps(
                    {
                        "source_version": source,
                        "completed_records": min(
                            start + args.capture_batch_size,
                            len(selected),
                        ),
                        "total_records": len(selected),
                    }
                ),
                flush=True,
            )
        cohort_results.append(
            {
                "source_version": source,
                "record_count": len(selected),
                "prefix_tokens": sum(
                    int(record["prefix_tokens"])
                    for record, _ in selected
                ),
                "logical_bytes": dict(source_logical),
                "physical_bytes": dict(source_physical),
                "elapsed_seconds": time.perf_counter() - source_started,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    records = tuple(
        descriptors[index]
        for index in sorted(descriptors)
    )
    if len(records) != workload["summary"]["records"]:
        raise ValueError("Stage 4 source record coverage is incomplete")
    logical_totals = {
        representation: sum(
            shard.logical_bytes
            for record in records
            for shard in record.shards
            if shard.representation == representation
        )
        for representation in (
            "normalized_capsule_fp16",
            "old_kv_fp16",
            "raw_history",
            "residual_hidden_suffix_bf16",
        )
    }
    physical_totals = {
        representation: sum(
            shard.physical_bytes
            for record in records
            for shard in record.shards
            if shard.representation == representation
        )
        for representation in logical_totals
    }
    source_contract = blueprint["source_contract"]["representations"]
    expected = {
        "normalized_capsule_fp16": source_contract[
            "normalized_capsule_fp16"
        ]["logical_bytes"],
        "old_kv_fp16": source_contract["old_kv_fp16"]["logical_bytes"],
        "residual_hidden_suffix_bf16": source_contract[
            "residual_hidden_suffix_bf16"
        ]["current_verified_p8_fallback_bytes"],
    }
    if any(logical_totals[name] != value for name, value in expected.items()):
        raise ValueError("Stage 4 source logical bytes differ from Stage 0")
    manifest = Stage4SourceManifest(
        workload_content_sha256=workload["content_sha256"],
        workload_file_sha256=sha256_file(root / args.workload_manifest),
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        kv_width=cfg.num_heads * cfg.head_dim,
        records=records,
        creation={
            "protocol": PROTOCOL,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "capture_batch_size": args.capture_batch_size,
            "residual_storage_dtype": "bfloat16",
            "residual_start_layer": RESIDUAL_START_LAYER,
            "residual_source_versions": list(RESIDUAL_SOURCES),
            "logical_bytes": logical_totals,
            "physical_bytes": physical_totals,
            "cohorts": cohort_results,
            "source_preflight": preflight,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    manifest.write(output)
    return manifest


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    blueprint, workload, cfg, sample_by_user = validate_inputs(root, args)
    output = root / args.output
    preflight = source_preflight(root, blueprint, output, workload)
    if args.validate_only:
        manifest = validate_existing(
            root,
            args,
            blueprint,
            workload,
        )
    else:
        manifest = materialize(
            root,
            args,
            blueprint,
            workload,
            cfg,
            sample_by_user,
            preflight,
        )
        validate_existing(
            root,
            args,
            blueprint,
            workload,
        )
    print(
        json.dumps(
            {
                "status": "valid",
                "path": args.output,
                "sha256": sha256_file(output),
                "record_count": manifest.record_count,
                "prefix_tokens": manifest.prefix_tokens,
                "logical_bytes": manifest.creation["logical_bytes"],
                "physical_bytes": manifest.creation["physical_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
