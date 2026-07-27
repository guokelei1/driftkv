from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hstu_kvcache.data import collate_batch, load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    FusedMigrationOperator,
    JaggedMigratedKVBatch,
    MigrationCapsuleBatch,
    PackedMigrationOperator,
    ReferenceMigrationOperator,
    benchmark_cuda_extent_operator,
    capture_layerwise_state,
    load_runtime_program,
    profile_fused_extent_operator_stages,
    profile_packed_extent_operator_stages,
    profile_reference_extent_operator_stages,
    sha256_file,
    validate_contiguous_output_extent,
)
from hstu_kvcache.models import HSTUConfig
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

PROTOCOL = "cohortkv_single_config_stage3_operator_v1"
PARENT_PROTOCOL = "cohortkv_single_config_full_chain_development_v1"
STAGE2_PROTOCOL = "cohortkv_single_config_stage2_frozen_v1"
DEFAULT_BLUEPRINT = "configs/cohortkv_single_config_v1/blueprint.json"
DEFAULT_WORKLOAD = "configs/cohortkv_single_config_v1/workload_manifest.json"
DEFAULT_STAGE2 = "configs/cohortkv_single_config_v1/stage2_compiler_summary.json"
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
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage3_operator_seed0.json"
)
NEGATIVE_LAYOUT_RESULT = (
    "results/system/"
    "kuairand_long_context_4plus12_cohort_jagged_system_seed0.json"
)
SOURCE_VERSIONS = ("theta0", "theta4", "theta10")
TARGET_VERSION = "theta11"
BATCH_SIZES = (1, 2, 4)
BUCKET_WIDTHS = (16, 32, 64)
OPERATOR_NAMES = ("packed_fp16", "fused_fp16")
TRANSPORT_ATOL = 0.02
TRANSPORT_RTOL = 0.02


@dataclass(frozen=True)
class CapsuleRecord:
    record_id: int
    user_id: int
    source_version: str
    normed: torch.Tensor

    @property
    def length(self) -> int:
        return self.normed.shape[1]

    @property
    def nbytes(self) -> int:
        return self.normed.numel() * self.normed.element_size()


@dataclass(frozen=True)
class ResidentBatch:
    capsule: MigrationCapsuleBatch
    destination: JaggedMigratedKVBatch

    @property
    def allocated_tokens(self) -> int:
        return self.capsule.batch_size * self.capsule.seq_len


@dataclass
class DifferenceAccumulator:
    elements: int = 0
    mismatched: int = 0
    max_abs: float = 0.0
    squared_error: float = 0.0
    squared_reference: float = 0.0

    def update(
        self,
        actual: torch.Tensor,
        reference: torch.Tensor,
        atol: float,
        rtol: float,
    ) -> None:
        delta = actual.float() - reference.float()
        self.elements += actual.numel()
        self.mismatched += int(
            torch.count_nonzero(
                ~torch.isclose(actual, reference, atol=atol, rtol=rtol)
            )
        )
        self.max_abs = max(self.max_abs, float(delta.abs().max()))
        self.squared_error += float(delta.square().sum())
        self.squared_reference += float(reference.float().square().sum())

    def payload(self) -> dict[str, float | int]:
        denominator = max(self.squared_reference, 1e-24)
        return {
            "elements": self.elements,
            "mismatched_elements": self.mismatched,
            "max_abs": self.max_abs,
            "rms": math.sqrt(self.squared_error / max(self.elements, 1)),
            "fro_relative": math.sqrt(self.squared_error / denominator),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint", default=DEFAULT_BLUEPRINT)
    parser.add_argument("--workload-manifest", default=DEFAULT_WORKLOAD)
    parser.add_argument("--stage2-summary", default=DEFAULT_STAGE2)
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture-batch-size", type=int, default=4)
    parser.add_argument("--screen-repeats", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-repeats", type=int, default=3)
    parser.add_argument("--profile-warmup", type=int, default=5)
    parser.add_argument("--profile-repeats", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "capture_batch_size": (args.capture_batch_size, 4),
        "screen_repeats": (args.screen_repeats, 1),
        "warmup_runs": (args.warmup_runs, 1),
        "measured_repeats": (args.measured_repeats, 3),
        "profile_warmup": (args.profile_warmup, 5),
        "profile_repeats": (args.profile_repeats, 20),
    }
    changed = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in frozen.items()
        if actual != expected
    }
    if changed:
        raise ValueError(f"Stage 3 frozen settings changed: {changed}")


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
) -> tuple[dict, dict, dict, dict, HSTUConfig, list[dict]]:
    blueprint = json.loads((root / args.blueprint).read_text())
    workload = json.loads((root / args.workload_manifest).read_text())
    stage2 = json.loads((root / args.stage2_summary).read_text())
    training = json.loads((root / args.training_result).read_text())
    if (
        blueprint.get("protocol") != PARENT_PROTOCOL
        or blueprint.get("status")
        not in {"stage2_compiler_frozen", "stage3_operator_frozen"}
        or blueprint.get("scope", {}).get("completed_stages")[:3]
        != [0, 1, 2]
    ):
        raise ValueError("Stage 3 parent blueprint is invalid")
    stage2_frozen = blueprint["frozen_inputs"]["stage2_compiler_summary"]
    if (
        stage2.get("protocol") != STAGE2_PROTOCOL
        or stage2.get("status") != "stage2_frozen"
        or stage2_frozen["path"] != args.stage2_summary
        or sha256_file(root / args.stage2_summary) != stage2_frozen["sha256"]
    ):
        raise ValueError("Stage 2 frozen input is invalid")
    workload_frozen = blueprint["frozen_inputs"]["workload_manifest"]
    if (
        workload.get("protocol") != "cohortkv_single_config_workload_v1"
        or sha256_file(root / args.workload_manifest)
        != workload_frozen["file_sha256"]
        or workload["content_sha256"]
        != workload_frozen["content_sha256"]
        or stage2["workload"]["content_sha256"]
        != workload["content_sha256"]
    ):
        raise ValueError("Stage 3 workload input is invalid")
    if (
        training.get("protocol") != training_protocol_for_base_days(4)
        or training.get("status") != "complete"
        or training["model"] != blueprint["data_and_model"]["model"]
        or sha256_file(root / args.training_result)
        != blueprint["frozen_inputs"]["training_result"]["sha256"]
    ):
        raise ValueError("Stage 3 training input is invalid")
    if sha256_file(root / args.prepared_data) != blueprint[
        "frozen_inputs"
    ]["prepared_data"]["sha256"]:
        raise ValueError("Stage 3 prepared data hash mismatch")
    grid = blueprint["runtime_tuning_contract"]["grid"]
    if (
        grid["batch_size"] != list(BATCH_SIZES)
        or grid["length_bucket_width"] != list(BUCKET_WIDTHS)
        or grid["compiled_operator"] != list(OPERATOR_NAMES)
        or blueprint["runtime_tuning_contract"]["role"]
        != "program_selection"
        or blueprint["runtime_tuning_contract"]["candidate_order_seed"]
        != 73421
    ):
        raise ValueError("Stage 3 operator grid differs from Stage 0")
    cfg = HSTUConfig(**training["model"])
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
        raise ValueError("Stage 3 endpoint date mismatch")
    roles = split_samples(samples)
    expected_ids = {
        role: {
            int(record["user_id"])
            for record in workload["records"]
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
        raise ValueError("Stage 3 reconstructed roles differ from the manifest")
    if len(roles["program_selection"]) != 60:
        raise ValueError("Stage 3 requires all 60 program-selection records")
    checkpoint_hashes = {
        value["version"]: value["sha256"]
        for value in blueprint["frozen_inputs"]["checkpoints"]
    }
    for version in SOURCE_VERSIONS:
        if (
            sha256_file(checkpoint_path(root, args.checkpoint_dir, version))
            != checkpoint_hashes[version]
        ):
            raise ValueError(f"{version} checkpoint hash mismatch")
    return (
        blueprint,
        workload,
        stage2,
        training,
        cfg,
        roles["program_selection"],
    )


def load_programs(
    root: Path,
    stage2: dict,
    model: dict,
) -> tuple[dict[str, object], list[dict]]:
    programs = {}
    descriptors = []
    for pair in stage2["pairs"]:
        source = pair["source_version"]
        descriptor = pair["runtime_program"]
        program, loaded = load_runtime_program(
            root / descriptor["path"],
            expected_sha256=descriptor["sha256"],
            expected_source_version=source,
            expected_target_version=TARGET_VERSION,
            expected_model=model,
        )
        if (
            loaded["dtype"] != "float16"
            or program.adapter.weights.dtype != torch.float16
            or program.adapter.biases.dtype != torch.float16
        ):
            raise ValueError("Stage 3 runtime program is not FP16")
        programs[source] = program
        descriptors.append(
            {
                "source_version": source,
                "target_version": TARGET_VERSION,
                "path": descriptor["path"],
                "sha256": descriptor["sha256"],
                "bytes": descriptor["bytes"],
                "dtype": descriptor["dtype"],
            }
        )
    if set(programs) != set(SOURCE_VERSIONS):
        raise ValueError("Stage 3 runtime program coverage is incomplete")
    descriptors.sort(key=lambda value: SOURCE_VERSIONS.index(value["source_version"]))
    return programs, descriptors


@torch.no_grad()
def materialize_selection_records(
    root: Path,
    args: argparse.Namespace,
    cfg: HSTUConfig,
    samples: list[dict],
    workload: dict,
    device: torch.device,
) -> tuple[list[CapsuleRecord], dict]:
    started = time.perf_counter()
    manifest_by_user = {
        int(record["user_id"]): record
        for record in workload["records"]
        if record["evaluation_role"] == "program_selection"
    }
    grouped = defaultdict(list)
    for sample in samples:
        user_id = int(sample["history"]["user_id"])
        grouped[manifest_by_user[user_id]["source_version"]].append(sample)
    records = []
    cohort_payload = []
    for source in SOURCE_VERSIONS:
        version = int(source.removeprefix("theta"))
        model = load_checkpoint_model(
            cfg,
            str(root / args.checkpoint_dir),
            version,
            device,
        )
        selected_samples = sorted(
            grouped[source],
            key=lambda sample: (
                manifest_by_user[int(sample["history"]["user_id"])][
                    "prefix_tokens"
                ],
                int(sample["history"]["user_id"]),
            ),
        )
        source_started = time.perf_counter()
        source_tokens = 0
        for start in range(0, len(selected_samples), args.capture_batch_size):
            selected = selected_samples[start : start + args.capture_batch_size]
            prefixes = [
                unlabeled_prefix(sample["history"], cfg.max_seq_len)
                for sample in selected
            ]
            batch = collate_batch(prefixes, max_seq_len=cfg.max_seq_len - 1)
            gpu = {
                name: value.to(device)
                for name, value in batch.items()
            }
            state = capture_layerwise_state(
                model,
                gpu["item_ids"],
                gpu["behaviors"],
                gpu["time_deltas"],
                gpu["lengths"],
            )
            normed = torch.stack(state.normed_states)
            for row, sample in enumerate(selected):
                user_id = int(sample["history"]["user_id"])
                expected = manifest_by_user[user_id]
                length = int(gpu["lengths"][row])
                if length != expected["prefix_tokens"]:
                    raise ValueError("materialized prefix length differs from manifest")
                value = (
                    normed[:, row, :length]
                    .to(device="cpu", dtype=torch.float16)
                    .contiguous()
                )
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("materialized capsule contains nonfinite values")
                records.append(
                    CapsuleRecord(
                        record_id=int(expected["record_id"]),
                        user_id=user_id,
                        source_version=source,
                        normed=value,
                    )
                )
                source_tokens += length
            del batch, gpu, state, normed
        cohort_payload.append(
            {
                "source_version": source,
                "records": len(selected_samples),
                "valid_tokens": source_tokens,
                "capture_seconds": time.perf_counter() - source_started,
            }
        )
        del model
        torch.cuda.empty_cache()
    records.sort(key=lambda value: value.record_id)
    expected_record_ids = {
        int(value["record_id"])
        for value in manifest_by_user.values()
    }
    if (
        len(records) != 60
        or {record.record_id for record in records} != expected_record_ids
        or len({record.record_id for record in records}) != len(records)
    ):
        raise ValueError("Stage 3 materialized record coverage is incomplete")
    identity = [
        {
            "record_id": record.record_id,
            "user_id": record.user_id,
            "source_version": record.source_version,
            "prefix_tokens": record.length,
        }
        for record in records
    ]
    return records, {
        "records": len(records),
        "valid_tokens": sum(record.length for record in records),
        "logical_capsule_bytes": sum(record.nbytes for record in records),
        "dtype": "float16",
        "layout": "layer-major unpadded [L,T,H] per record",
        "labels_used": False,
        "final_test_evaluated": False,
        "record_identity_sha256": hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "cohorts": cohort_payload,
        "elapsed_seconds": time.perf_counter() - started,
    }


def make_destination(
    capsule: MigrationCapsuleBatch,
    program,
) -> JaggedMigratedKVBatch:
    offsets = torch.cat(
        (
            torch.zeros(
                1,
                dtype=torch.long,
                device=capsule.device,
            ),
            capsule.lengths.long().cumsum(0),
        )
    )
    shape = (
        program.num_layers,
        int(offsets[-1]),
        program.kv_width,
    )
    destination = JaggedMigratedKVBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        served_kv_target=program.target_version,
        k=torch.empty(
            shape,
            dtype=torch.float16,
            device=capsule.device,
        ),
        v=torch.empty(
            shape,
            dtype=torch.float16,
            device=capsule.device,
        ),
        lengths=capsule.lengths,
        offsets=offsets,
    )
    validate_contiguous_output_extent(
        program,
        capsule,
        destination,
        check_metadata_values=True,
    )
    return destination


def pack_resident_batches(
    records: list[CapsuleRecord],
    programs: dict[str, object],
    batch_size: int,
    bucket_width: int,
    device: torch.device,
) -> tuple[list[ResidentBatch], dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[
            (
                record.source_version,
                math.ceil(record.length / bucket_width),
            )
        ].append(record)
    batches = []
    allocated_tokens = 0
    for key in sorted(grouped):
        cohort = sorted(
            grouped[key],
            key=lambda value: (value.length, value.record_id),
        )
        for start in range(0, len(cohort), batch_size):
            selected = cohort[start : start + batch_size]
            width = min(
                2047,
                math.ceil(
                    max(record.length for record in selected) / bucket_width
                )
                * bucket_width,
            )
            shape = (
                selected[0].normed.shape[0],
                len(selected),
                width,
                selected[0].normed.shape[2],
            )
            normed = torch.zeros(shape, dtype=torch.float16)
            lengths = torch.empty(len(selected), dtype=torch.long)
            for row, record in enumerate(selected):
                normed[:, row, : record.length].copy_(record.normed)
                lengths[row] = record.length
            capsule = MigrationCapsuleBatch(
                record_ids=tuple(record.record_id for record in selected),
                migration_anchor_version=selected[0].source_version,
                normed=normed.to(device),
                lengths=lengths.to(device),
            )
            destination = make_destination(
                capsule,
                programs[selected[0].source_version],
            )
            batches.append(
                ResidentBatch(
                    capsule=capsule,
                    destination=destination,
                )
            )
            allocated_tokens += capsule.batch_size * capsule.seq_len
    logical_tokens = sum(record.length for record in records)
    if sum(batch.destination.token_count for batch in batches) != logical_tokens:
        raise ValueError("packed output extents do not cover all valid tokens")
    return batches, {
        "batch_size": batch_size,
        "bucket_width": bucket_width,
        "records": len(records),
        "batches": len(batches),
        "logical_tokens": logical_tokens,
        "allocated_tokens": allocated_tokens,
        "padding_tokens": allocated_tokens - logical_tokens,
        "padding_fraction": 1.0
        - logical_tokens / max(allocated_tokens, 1),
        "capsule_bytes": sum(batch.capsule.nbytes for batch in batches),
        "output_extent_bytes": sum(
            batch.destination.nbytes for batch in batches
        ),
        "cohort_batch_counts": {
            source: sum(
                batch.capsule.migration_anchor_version == source
                for batch in batches
            )
            for source in SOURCE_VERSIONS
        },
        "sequence_widths": [
            batch.capsule.seq_len for batch in batches
        ],
        "batch_sizes": [
            batch.capsule.batch_size for batch in batches
        ],
    }


def padding_nonzero(tensor: torch.Tensor, lengths: torch.Tensor) -> int:
    positions = torch.arange(tensor.shape[2], device=tensor.device)
    invalid = positions.unsqueeze(0) >= lengths.unsqueeze(1)
    return int(
        torch.count_nonzero(
            tensor.masked_select(invalid.unsqueeze(0).unsqueeze(-1))
        )
    )


def update_pair(
    accumulator: DifferenceAccumulator,
    actual_k: torch.Tensor,
    actual_v: torch.Tensor,
    reference_k: torch.Tensor,
    reference_v: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    accumulator.update(actual_k, reference_k, atol, rtol)
    accumulator.update(actual_v, reference_v, atol, rtol)


@torch.no_grad()
def validate_layout(
    batches: list[ResidentBatch],
    prepared: dict[str, dict[str, object]],
    operators: dict[str, object],
) -> dict:
    differences = {
        "packed_from_reference": DifferenceAccumulator(),
        "fused_from_reference": DifferenceAccumulator(),
        "fused_from_packed": DifferenceAccumulator(),
        "reference_dense_extent_identity": DifferenceAccumulator(),
        "packed_dense_extent_identity": DifferenceAccumulator(),
        "fused_dense_extent_identity": DifferenceAccumulator(),
    }
    dense_padding_nonzero = {
        "reference_fp32": 0,
        "packed_fp16": 0,
        "fused_fp16": 0,
    }
    source_padding_nonzero = 0
    finite = {
        "reference_fp32": True,
        "packed_fp16": True,
        "fused_fp16": True,
    }
    pointer_preserved = {
        "reference_fp32": True,
        "packed_fp16": True,
        "fused_fp16": True,
    }
    valid_elements = 0
    for resident in batches:
        capsule = resident.capsule
        source = capsule.migration_anchor_version
        source_padding_nonzero += padding_nonzero(
            capsule.normed,
            capsule.lengths,
        )
        dense_outputs = {}
        extent_outputs = {}
        for name, operator in operators.items():
            program = prepared[name][source]
            dense_outputs[name] = operator.execute(program, capsule)
            destination = make_destination(capsule, program)
            pointers = (destination.k.data_ptr(), destination.v.data_ptr())
            extent_outputs[name] = operator.execute_into(
                program,
                capsule,
                destination,
            )
            validate_contiguous_output_extent(
                program,
                capsule,
                extent_outputs[name],
                check_metadata_values=True,
            )
            pointer_preserved[name] = pointer_preserved[name] and pointers == (
                extent_outputs[name].k.data_ptr(),
                extent_outputs[name].v.data_ptr(),
            )
            dense_padding_nonzero[name] += padding_nonzero(
                dense_outputs[name].cache.k,
                capsule.lengths,
            )
            dense_padding_nonzero[name] += padding_nonzero(
                dense_outputs[name].cache.v,
                capsule.lengths,
            )
            finite[name] = finite[name] and bool(
                torch.isfinite(extent_outputs[name].k).all()
                and torch.isfinite(extent_outputs[name].v).all()
            )
            positions = torch.arange(
                capsule.seq_len,
                device=capsule.device,
            )
            valid = positions.unsqueeze(0) < capsule.lengths.unsqueeze(1)
            update_pair(
                differences[f"{name.removesuffix('_fp32').removesuffix('_fp16')}_dense_extent_identity"],
                extent_outputs[name].k,
                extent_outputs[name].v,
                dense_outputs[name].cache.k[:, valid],
                dense_outputs[name].cache.v[:, valid],
                0.0,
                0.0,
            )
        reference = extent_outputs["reference_fp32"]
        packed = extent_outputs["packed_fp16"]
        fused = extent_outputs["fused_fp16"]
        update_pair(
            differences["packed_from_reference"],
            packed.k,
            packed.v,
            reference.k,
            reference.v,
            TRANSPORT_ATOL,
            TRANSPORT_RTOL,
        )
        update_pair(
            differences["fused_from_reference"],
            fused.k,
            fused.v,
            reference.k,
            reference.v,
            TRANSPORT_ATOL,
            TRANSPORT_RTOL,
        )
        update_pair(
            differences["fused_from_packed"],
            fused.k,
            fused.v,
            packed.k,
            packed.v,
            TRANSPORT_ATOL,
            TRANSPORT_RTOL,
        )
        valid_elements += reference.k.numel() + reference.v.numel()
        del dense_outputs, extent_outputs
    payload = {
        "records": sum(batch.capsule.batch_size for batch in batches),
        "valid_tokens": sum(batch.destination.token_count for batch in batches),
        "valid_fp16_kv_elements": valid_elements,
        "source_padding_nonzero": source_padding_nonzero,
        "dense_output_padding_nonzero": dense_padding_nonzero,
        "finite": finite,
        "destination_pointer_preserved": pointer_preserved,
        "differences": {
            name: value.payload()
            for name, value in differences.items()
        },
        "output_contract": (
            "separate contiguous unpadded FP16 [L,T,Dkv] K/V "
            "with lengths and offsets"
        ),
    }
    if (
        source_padding_nonzero
        or any(dense_padding_nonzero.values())
        or not all(finite.values())
        or not all(pointer_preserved.values())
        or any(
            value["mismatched_elements"]
            for value in payload["differences"].values()
        )
    ):
        raise ValueError("Stage 3 operator correctness failed")
    return payload


@torch.no_grad()
def time_operator(
    batches: list[ResidentBatch],
    operator,
    prepared: dict[str, object],
    repeats: int,
    warmup: int,
) -> dict:
    for resident in batches:
        source = resident.capsule.migration_anchor_version
        validate_contiguous_output_extent(
            prepared[source],
            resident.capsule,
            resident.destination,
            check_metadata_values=True,
        )
    for _ in range(warmup):
        for resident in batches:
            source = resident.capsule.migration_anchor_version
            operator.execute_into(
                prepared[source],
                resident.capsule,
                resident.destination,
            )
    torch.cuda.synchronize(batches[0].capsule.device)
    samples = []
    peaks = []
    for _ in range(repeats):
        device = batches[0].capsule.device
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for resident in batches:
            source = resident.capsule.migration_anchor_version
            operator.execute_into(
                prepared[source],
                resident.capsule,
                resident.destination,
            )
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        peaks.append(
            max(0, torch.cuda.max_memory_allocated(device) - baseline)
        )
    return {
        "values_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "cv": (
            statistics.pstdev(samples) / statistics.fmean(samples)
            if len(samples) > 1
            else 0.0
        ),
        "temporary_peak_bytes": peaks,
        "maximum_temporary_peak_bytes": max(peaks),
    }


def release_batches(batches: list[ResidentBatch]) -> None:
    del batches[:]
    gc.collect()
    torch.cuda.empty_cache()


def candidate_id(operator: str, batch_size: int, bucket_width: int) -> str:
    return f"{operator}_b{batch_size}_w{bucket_width}"


def stage_profile_payload(profile) -> dict:
    return {
        "operator": profile.operator,
        "total_ms": list(profile.total.values_ms),
        "total_median_ms": profile.total.median_ms,
        "stages": {
            name: {
                "values_ms": list(samples.values_ms),
                "median_ms": samples.median_ms,
            }
            for name, samples in profile.stages.items()
        },
    }


@torch.no_grad()
def representative_profile(
    records: list[CapsuleRecord],
    programs: dict[str, object],
    selected: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    batches, packing = pack_resident_batches(
        records,
        programs,
        selected["batch_size"],
        selected["bucket_width"],
        device,
    )
    representative = max(
        batches,
        key=lambda value: (
            value.allocated_tokens,
            value.destination.token_count,
        ),
    )
    capsule = representative.capsule
    source = capsule.migration_anchor_version
    program = programs[source]
    operators = (
        ReferenceMigrationOperator(),
        PackedMigrationOperator(torch.float16),
        FusedMigrationOperator(),
    )
    profiles = {}
    for operator in operators:
        destination = make_destination(capsule, program)
        _, profile = benchmark_cuda_extent_operator(
            operator,
            program,
            capsule,
            destination,
            args.profile_warmup,
            args.profile_repeats,
        )
        profiles[operator.name] = {
            "latency_ms": list(profile.latency.values_ms),
            "median_ms": profile.latency.median_ms,
            "temporary_peak_bytes": list(profile.temporary_peak_bytes),
            "maximum_temporary_peak_bytes": max(
                profile.temporary_peak_bytes
            ),
        }
    destination = make_destination(capsule, program)
    stage_profiles = (
        profile_reference_extent_operator_stages(
            program,
            capsule,
            destination,
            args.profile_repeats,
        ),
        profile_packed_extent_operator_stages(
            program,
            capsule,
            destination,
            torch.float16,
            args.profile_repeats,
        ),
        profile_fused_extent_operator_stages(
            FusedMigrationOperator(),
            program,
            capsule,
            destination,
            args.profile_repeats,
        ),
    )
    layers = program.num_layers
    batch = capsule.batch_size
    sequence = capsule.seq_len
    hidden = capsule.hidden_size
    output_width = 2 * program.kv_width
    tokens = representative.destination.token_count
    output_bytes = representative.destination.nbytes
    result = {
        "source_version": source,
        "records": capsule.batch_size,
        "sequence_width": capsule.seq_len,
        "lengths": capsule.lengths.cpu().tolist(),
        "valid_tokens": tokens,
        "capsule_bytes": capsule.nbytes,
        "output_extent_bytes": output_bytes,
        "output_layout": "separate contiguous unpadded FP16 [L,T,Dkv] K/V",
        "profiles": profiles,
        "epilogue_breakdown": {
            value.operator: stage_profile_payload(value)
            for value in stage_profiles
        },
        "logical_temporary_inventory": {
            "reference_input_cast_fp32": layers
            * batch
            * sequence
            * hidden
            * 4,
            "reference_projected_concat_fp32": layers
            * batch
            * sequence
            * output_width
            * 4,
            "packed_projected_concat_fp16": layers
            * batch
            * sequence
            * output_width
            * 2,
            "one_compact_k_or_v_fp16": layers
            * tokens
            * program.kv_width
            * 2,
            "fused_global_temporary_bytes": 0,
            "fused_epilogue_components": [
                "bias",
                "valid_length_resolution",
                "K/V split",
                "direct contiguous extent write",
            ],
            "fused_components_separately_timed": False,
        },
        "packing_context": packing,
    }
    release_batches(batches)
    return result


def select_final_candidate(
    screen: list[dict],
    finalists: dict[str, dict],
    top_three: list[str],
) -> tuple[dict, dict]:
    candidates = {
        value["candidate_id"]: value
        for value in screen
    }
    selected_id = min(
        top_three,
        key=lambda value: (
            finalists[value]["median_ms"],
            finalists[value]["maximum_temporary_peak_bytes"],
            candidates[value]["packing"]["padding_tokens"],
        ),
    )
    selected_candidate = candidates[selected_id]
    selected_timing = finalists[selected_id]
    fastest_packed_id = min(
        (
            value
            for value in finalists
            if candidates[value]["operator"] == "packed_fp16"
        ),
        key=lambda value: finalists[value]["median_ms"],
    )
    fastest_fused_id = min(
        (
            value
            for value in finalists
            if candidates[value]["operator"] == "fused_fp16"
        ),
        key=lambda value: finalists[value]["median_ms"],
    )
    compared_fused_id = (
        selected_id
        if selected_candidate["operator"] == "fused_fp16"
        else fastest_fused_id
    )
    packed_samples = finalists[fastest_packed_id]["values_ms"]
    fused_samples = finalists[compared_fused_id]["values_ms"]
    stable = (
        finalists[compared_fused_id]["median_ms"]
        < finalists[fastest_packed_id]["median_ms"]
        and max(fused_samples) < min(packed_samples)
    )
    fallback_applied = False
    if selected_candidate["operator"] == "fused_fp16" and not stable:
        selected_id = fastest_packed_id
        selected_candidate = candidates[selected_id]
        selected_timing = finalists[selected_id]
        fallback_applied = True
    selection = {
        "candidate_id": selected_id,
        "operator": selected_candidate["operator"],
        "operator_implementation": (
            "FusedMigrationOperator"
            if selected_candidate["operator"] == "fused_fp16"
            else "PackedMigrationOperator(torch.float16)"
        ),
        "batch_size": selected_candidate["batch_size"],
        "bucket_width": selected_candidate["bucket_width"],
        "timing": selected_timing,
        "selection_scope": (
            "resident Stage-3 default; Stage 4 independently tunes every "
            "method/destination/GPU point over the complete frozen grid"
        ),
        "fused_stability_gate": {
            "fastest_packed_candidate": fastest_packed_id,
            "fastest_fused_candidate": fastest_fused_id,
            "tested_fused_candidate": compared_fused_id,
            "packed_median_ms": finalists[fastest_packed_id]["median_ms"],
            "fused_median_ms": finalists[compared_fused_id]["median_ms"],
            "fused_speedup_over_packed": (
                finalists[fastest_packed_id]["median_ms"]
                / finalists[compared_fused_id]["median_ms"]
            ),
            "all_fused_samples_below_all_packed_samples": stable,
            "packed_fallback_applied": fallback_applied,
        },
    }
    return selection, {
        "top_three_screen_candidates": top_three,
        "fully_measured_candidates": sorted(finalists),
        "selection_rule": (
            "minimum median among the fastest three screen candidates; "
            "if every measured run of the selected Triton path is not "
            "faster than every measured run of the fastest packed control, "
            "fall back to packed"
        ),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    root = Path(__file__).resolve().parents[1]
    (
        blueprint,
        workload,
        stage2,
        training,
        cfg,
        selection_samples,
    ) = validate_inputs(root, args)
    programs, program_descriptors = load_programs(
        root,
        stage2,
        training["model"],
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "program_selection_records": len(selection_samples),
                    "programs": program_descriptors,
                    "candidate_count": (
                        len(BATCH_SIZES)
                        * len(BUCKET_WIDTHS)
                        * len(OPERATOR_NAMES)
                    ),
                    "status": "validated",
                },
                indent=2,
            ),
            flush=True,
        )
        return
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Stage 3 requires CUDA")
    torch.manual_seed(0)
    np.random.seed(0)
    records, materialization = materialize_selection_records(
        root,
        args,
        cfg,
        selection_samples,
        workload,
        device,
    )
    operators = {
        "reference_fp32": ReferenceMigrationOperator(),
        "packed_fp16": PackedMigrationOperator(torch.float16),
        "fused_fp16": FusedMigrationOperator(),
    }
    prepared = {
        name: {
            source: operator.prepare_program(program, device)
            for source, program in programs.items()
        }
        for name, operator in operators.items()
    }
    candidates = [
        {
            "candidate_id": candidate_id(operator, batch_size, bucket_width),
            "operator": operator,
            "batch_size": batch_size,
            "bucket_width": bucket_width,
        }
        for batch_size in BATCH_SIZES
        for bucket_width in BUCKET_WIDTHS
        for operator in OPERATOR_NAMES
    ]
    order = np.random.default_rng(73421).permutation(len(candidates))
    correctness_by_layout = {}
    screen = []
    search_started = time.perf_counter()
    for position, index in enumerate(order):
        candidate = candidates[int(index)]
        layout_key = (
            f"b{candidate['batch_size']}_w{candidate['bucket_width']}"
        )
        batches, packing = pack_resident_batches(
            records,
            programs,
            candidate["batch_size"],
            candidate["bucket_width"],
            device,
        )
        if layout_key not in correctness_by_layout:
            correctness_by_layout[layout_key] = validate_layout(
                batches,
                prepared,
                operators,
            )
        timing = time_operator(
            batches,
            operators[candidate["operator"]],
            prepared[candidate["operator"]],
            args.screen_repeats,
            warmup=0,
        )
        screen.append(
            {
                **candidate,
                "screen_position": position,
                "packing": packing,
                "correctness_layout": layout_key,
                "screen_timing": timing,
            }
        )
        print(
            json.dumps(
                {
                    "candidate": candidate["candidate_id"],
                    "position": position + 1,
                    "total": len(candidates),
                    "screen_ms": timing["median_ms"],
                }
            ),
            flush=True,
        )
        release_batches(batches)
    screen.sort(key=lambda value: value["screen_position"])
    ranked = sorted(
        screen,
        key=lambda value: (
            value["screen_timing"]["median_ms"],
            value["screen_timing"]["maximum_temporary_peak_bytes"],
            value["packing"]["padding_tokens"],
        ),
    )
    top_three = [value["candidate_id"] for value in ranked[:3]]
    fastest_per_operator = {
        operator: min(
            (
                value
                for value in screen
                if value["operator"] == operator
            ),
            key=lambda value: value["screen_timing"]["median_ms"],
        )["candidate_id"]
        for operator in OPERATOR_NAMES
    }
    finalist_ids = set(top_three) | set(fastest_per_operator.values())
    candidates_by_id = {
        value["candidate_id"]: value
        for value in screen
    }
    finalists = {}
    for value in sorted(finalist_ids):
        candidate = candidates_by_id[value]
        batches, _ = pack_resident_batches(
            records,
            programs,
            candidate["batch_size"],
            candidate["bucket_width"],
            device,
        )
        finalists[value] = time_operator(
            batches,
            operators[candidate["operator"]],
            prepared[candidate["operator"]],
            args.measured_repeats,
            args.warmup_runs,
        )
        release_batches(batches)
    selection, procedure = select_final_candidate(
        screen,
        finalists,
        top_three,
    )
    profile = representative_profile(
        records,
        programs,
        selection,
        device,
        args,
    )
    negative_result_path = root / NEGATIVE_LAYOUT_RESULT
    result = {
        "protocol": PROTOCOL,
        "parent_protocol": PARENT_PROTOCOL,
        "status": "stage3_complete",
        "study_stage": "single_configuration_seed0_development",
        "seed": 0,
        "labels_used": False,
        "final_test_evaluated": False,
        "role": "program_selection",
        "role_counts": workload["evaluation_roles"]["counts"],
        "blueprint": {
            "path": args.blueprint,
            "sha256": sha256_file(root / args.blueprint),
            "protocol": blueprint["protocol"],
        },
        "workload_manifest": {
            "path": args.workload_manifest,
            "sha256": sha256_file(root / args.workload_manifest),
            "content_sha256": workload["content_sha256"],
        },
        "stage2_summary": {
            "path": args.stage2_summary,
            "sha256": sha256_file(root / args.stage2_summary),
            "protocol": stage2["protocol"],
        },
        "runtime_programs": program_descriptors,
        "contracts": {
            "capsule": {
                "migration_anchor_preserved": True,
                "served_kv_target": TARGET_VERSION,
                "storage_layout": "layer-major unpadded FP16 [L,T,H] plus offsets",
                "execution_layout": "dense length-bucketed FP16 [L,B,S,H]",
                "length_scope": "history[:-1]",
            },
            "output_extent": {
                "layout": (
                    "separate contiguous unpadded FP16 [L,T,Dkv] "
                    "K/V plus lengths and offsets"
                ),
                "allocation": "preallocated outside resident operator timing",
                "write": "all operators use execute_into on the same destination ABI",
                "padding_published": False,
            },
            "transport_allclose": {
                "atol": TRANSPORT_ATOL,
                "rtol": TRANSPORT_RTOL,
                "finite_required": True,
            },
        },
        "materialization": materialization,
        "candidate_grid": {
            "batch_sizes": list(BATCH_SIZES),
            "bucket_widths": list(BUCKET_WIDTHS),
            "operators": list(OPERATOR_NAMES),
            "candidate_count": len(candidates),
            "candidate_order_seed": 73421,
        },
        "correctness_by_layout": correctness_by_layout,
        "candidate_screen": screen,
        "finalist_timings": finalists,
        "selection": selection,
        "selection_procedure": procedure,
        "representative_profile": profile,
        "retained_negative_layout": {
            "protocol": "kuairand_long_context_4plus12_cohort_jagged_system_v3",
            "path": NEGATIVE_LAYOUT_RESULT,
            "sha256": sha256_file(negative_result_path),
            "decision": (
                "retain exact jagged/page correctness and negative performance "
                "boundary; do not reopen layout search in Stage 3"
            ),
        },
        "search_seconds": time.perf_counter() - search_started,
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_bytes": torch.cuda.get_device_properties(
                device
            ).total_memory,
        },
    }
    save_json(result, root / args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "selection": selection,
                "status": result["status"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
