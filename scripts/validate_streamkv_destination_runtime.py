import argparse
import json
from pathlib import Path

import torch

from hstu_kvcache.migration import (
    DESTINATION_OUT_OF_CORE_PROTOCOL,
    DRAMKVUpdateDestination,
    FilesystemKVUpdateDestination,
    HBMKVUpdateDestination,
    InMemoryRemoteObjectStore,
    JaggedMigrationCapsuleBatch,
    MigrationCapsuleBatch,
    OutOfCoreKVUpdateEngine,
    PackedJaggedMigrationOperator,
    RemoteKVUpdateDestination,
    capture_layerwise_state,
    compile_migration_program,
)
from hstu_kvcache.models import HSTU, HSTUConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        choices=("dram", "filesystem", "remote", "hbm"),
        default="dram",
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_inputs(dtype):
    torch.manual_seed(101)
    model = HSTU(
        HSTUConfig(
            num_items=100,
            num_behaviors=8,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=8,
            input_dropout=0.0,
        )
    ).eval()
    lengths = torch.tensor([8, 6, 5, 3])
    valid = torch.arange(8).unsqueeze(0) < lengths.unsqueeze(1)
    state = capture_layerwise_state(
        model,
        torch.randint(1, 101, (4, 8)) * valid,
        torch.randint(1, 9, (4, 8)) * valid,
        torch.rand(4, 8) * 100 * valid,
        lengths,
    )
    dense = MigrationCapsuleBatch.from_layerwise_state(
        state,
        migration_anchor_version="theta-validation-old",
        record_ids=(0, 1, 2, 3),
    )
    dense = MigrationCapsuleBatch(
        record_ids=dense.record_ids,
        migration_anchor_version=dense.migration_anchor_version,
        normed=dense.normed.to(dtype),
        lengths=dense.lengths,
    )
    program = compile_migration_program(
        model,
        source_version=dense.migration_anchor_version,
        target_version="theta-validation-new",
    )
    batches = []
    for part in dense.split(1):
        offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long),
                part.lengths.cumsum(dim=0),
            )
        )
        normed = torch.cat(
            [
                part.normed[:, row, : int(length)]
                for row, length in enumerate(part.lengths)
            ],
            dim=1,
        ).contiguous()
        batches.append(
            JaggedMigrationCapsuleBatch(
                record_ids=part.record_ids,
                migration_anchor_version=part.migration_anchor_version,
                normed=normed,
                lengths=part.lengths,
                offsets=offsets,
            )
        )
    return program, tuple(batches)


def make_destination(args):
    if args.destination == "dram":
        return DRAMKVUpdateDestination(), ("cpu",), torch.float32
    if args.destination == "filesystem":
        if args.root is None:
            raise ValueError("--root is required for filesystem publication")
        return (
            FilesystemKVUpdateDestination(args.root),
            ("cpu",),
            torch.float32,
        )
    if args.destination == "remote":
        return (
            RemoteKVUpdateDestination(InMemoryRemoteObjectStore()),
            ("cpu",),
            torch.float32,
        )
    devices = tuple(
        value.strip()
        for value in args.devices.split(",")
        if value.strip()
    )
    return HBMKVUpdateDestination(devices), devices, torch.float16


def main():
    args = parse_args()
    destination, devices, dtype = make_destination(args)
    program, batches = make_inputs(dtype)
    reference_operator = PackedJaggedMigrationOperator(torch.float32)
    reference_program = reference_operator.prepare_program(program, "cpu")
    expected = {
        record_id: result.record_kv(record_id)
        for batch in batches
        for result in (reference_operator.execute(reference_program, batch),)
        for record_id in batch.record_ids
    }
    engine = OutOfCoreKVUpdateEngine(
        program,
        devices=devices,
        destination=destination,
        wave_batch_limit=1,
    )
    report = engine.run("destination-validation", batches)
    maximum_error = 0.0
    for extent in report.manifest.extents:
        batch = destination.load_extent(
            report.manifest.target_version,
            extent.extent_id,
        )
        for record_id in batch.record_ids:
            actual_k, actual_v = batch.record_kv(record_id)
            expected_k, expected_v = expected[record_id]
            maximum_error = max(
                maximum_error,
                float(
                    (actual_k.float().cpu() - expected_k.float().cpu())
                    .abs()
                    .max()
                ),
                float(
                    (actual_v.float().cpu() - expected_v.float().cpu())
                    .abs()
                    .max()
                ),
            )
    result = {
        "protocol": DESTINATION_OUT_OF_CORE_PROTOCOL,
        "status": "destination_runtime_valid",
        "destination": args.destination,
        "manifest": report.manifest.to_dict(),
        "metrics": {
            "device_count": report.metrics.device_count,
            "wave_count": report.metrics.wave_count,
            "record_count": report.metrics.record_count,
            "token_count": report.metrics.token_count,
            "input_bytes": report.metrics.input_bytes,
            "output_bytes": report.metrics.output_bytes,
            "peak_wave_input_bytes": report.metrics.peak_wave_input_bytes,
            "peak_wave_output_bytes": report.metrics.peak_wave_output_bytes,
            "execution_seconds": report.metrics.execution_seconds,
            "publication_service_seconds": (
                report.metrics.publication_service_seconds
            ),
            "publication_wait_seconds": report.metrics.publication_wait_seconds,
            "commit_seconds": report.metrics.commit_seconds,
            "elapsed_seconds": report.metrics.elapsed_seconds,
        },
        "maximum_absolute_error": maximum_error,
        "correct": maximum_error <= (2e-2 if dtype == torch.float16 else 1e-5),
    }
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
