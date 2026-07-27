import hashlib
from pathlib import Path

import pytest
import torch

from hstu_kvcache.migration import (
    LazyStage4SourceReader,
    SourceRecordDescriptor,
    Stage4SourceManifest,
    build_stage4_extents,
    place_stage4_extents_lpt,
    write_source_shard,
)


def make_source_manifest(root: Path) -> tuple[Path, Stage4SourceManifest]:
    records = []
    lengths = (3, 5, 6, 4)
    sources = ("theta0", "theta0", "theta0", "theta4")
    for record_id, (length, source) in enumerate(zip(lengths, sources, strict=True)):
        base = torch.arange(2 * length * 4).reshape(2, length, 4)
        normalized = write_source_shard(
            root,
            f"normalized/{record_id:06d}.pt",
            "normalized_capsule_fp16",
            record_id,
            source,
            "theta11",
            length,
            {"normed": base.to(torch.float16)},
        )
        old_kv = write_source_shard(
            root,
            f"old_kv/{record_id:06d}.pt",
            "old_kv_fp16",
            record_id,
            source,
            "theta11",
            length,
            {
                "k": base.to(torch.float16),
                "v": (base + 1).to(torch.float16),
            },
        )
        raw = write_source_shard(
            root,
            f"raw/{record_id:06d}.pt",
            "raw_history",
            record_id,
            source,
            "theta11",
            length,
            {
                "item_ids": torch.arange(length, dtype=torch.long),
                "behaviors": torch.ones(length, dtype=torch.long),
                "time_deltas": torch.arange(length, dtype=torch.float32),
            },
        )
        records.append(
            SourceRecordDescriptor(
                record_id=record_id,
                user_id=record_id + 10,
                evaluation_role="program_selection",
                source_version=source,
                target_version="theta11",
                prefix_tokens=length,
                shards=(normalized, old_kv, raw),
            )
        )
    manifest = Stage4SourceManifest(
        workload_content_sha256="a" * 64,
        workload_file_sha256="b" * 64,
        num_layers=2,
        hidden_size=4,
        kv_width=4,
        records=tuple(records),
        creation={"kind": "test"},
    )
    path = root / "source_manifest.json"
    manifest.write(path)
    return path, manifest


def test_stage4_source_manifest_roundtrip_and_lazy_extent_read(tmp_path):
    path, expected = make_source_manifest(tmp_path)
    reader = LazyStage4SourceReader(path, "a" * 64)
    extents = build_stage4_extents(
        reader.manifest,
        (0, 1, 2, 3),
        {
            "theta0": ("normalized_capsule_fp16",),
            "theta4": ("normalized_capsule_fp16",),
        },
        batch_size=2,
        bucket_width=4,
    )

    assert reader.manifest == expected
    assert [value.sequence_width for value in extents] == [4, 8, 4]
    assert sum(value.token_count for value in extents) == 18
    assert sum(value.logical_output_bytes for value in extents) == 576

    damaged = tmp_path / expected.records[0].shard_map["old_kv_fp16"].path
    damaged.write_bytes(b"damaged but not requested")
    batch, metrics = reader.read_extent(extents[0], pin_memory=False)

    assert batch.record_ids == (0,)
    assert batch.normed.shape == (2, 1, 4, 4)
    assert torch.count_nonzero(batch.normed[:, :, 3:]) == 0
    assert metrics.physical_bytes == extents[0].physical_input_bytes
    assert metrics.logical_bytes == extents[0].logical_input_bytes
    assert metrics.peak_source_resident_bytes > batch.nbytes


def test_stage4_source_reader_rejects_damaged_requested_shard(tmp_path):
    path, manifest = make_source_manifest(tmp_path)
    reader = LazyStage4SourceReader(path)
    old_path = tmp_path / manifest.records[0].shard_map["old_kv_fp16"].path
    old_path.write_bytes(b"damaged")
    extents = build_stage4_extents(
        manifest,
        (0,),
        {"theta0": ("old_kv_fp16",)},
        batch_size=1,
        bucket_width=4,
    )

    with pytest.raises(ValueError, match="integrity"):
        reader.read_extent(extents[0], pin_memory=False)


def test_stage4_lpt_is_complete_deterministic_and_balanced(tmp_path):
    path, _ = make_source_manifest(tmp_path)
    manifest = LazyStage4SourceReader(path).manifest
    extents = build_stage4_extents(
        manifest,
        (0, 1, 2, 3),
        {
            "theta0": ("normalized_capsule_fp16", "raw_history"),
            "theta4": ("normalized_capsule_fp16", "raw_history"),
        },
        batch_size=1,
        bucket_width=4,
    )

    first = place_stage4_extents_lpt(extents, 2)
    second = place_stage4_extents_lpt(extents, 2)
    covered = [
        record_id
        for assignment in first
        for extent in assignment
        for record_id in extent.record_ids
    ]
    assigned = [
        sum(extent.placement_weight_bytes for extent in assignment)
        for assignment in first
    ]

    assert first == second
    assert sorted(covered) == [0, 1, 2, 3]
    assert len(covered) == len(set(covered))
    assert max(assigned) - min(assigned) <= max(
        value.placement_weight_bytes for value in extents
    )


def test_stage4_manifest_detects_changed_record_count(tmp_path):
    path, _ = make_source_manifest(tmp_path)
    payload = path.read_bytes().replace(b'"record_count":4', b'"record_count":5')
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="record count"):
        Stage4SourceManifest.load(path)


def test_stage4_shard_hash_is_encoded_file_hash(tmp_path):
    path, manifest = make_source_manifest(tmp_path)
    descriptor = manifest.records[2].shard_map["raw_history"]
    payload = (path.parent / descriptor.path).read_bytes()

    assert hashlib.sha256(payload).hexdigest() == descriptor.sha256
    assert len(payload) == descriptor.physical_bytes
