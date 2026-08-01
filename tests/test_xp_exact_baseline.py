from __future__ import annotations

import json
import socket
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hstu_kvcache.migration.xp_exact_baseline import (
    XPBaselineRecord,
    build_groups,
    file_sha256,
    load_fixed_inputs,
    ordinal_inter_event_time_deltas,
    run_exact_baseline,
    run_partial_exact_baseline,
    select_partial_exact,
)
from hstu_kvcache.streaming.sharded_edge import ExternalEmbeddingHSTU
from hstu_kvcache.streaming.xp_projected_edge import (
    OptimizerActiveRowTracker,
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
    save_xp_projected_checkpoint,
)


def _spec() -> XPProjectedModelSpec:
    return XPProjectedModelSpec(
        num_embeddings=17,
        embedding_width=8,
        hidden_size=4,
        num_prediction_items=8,
        num_behaviors=5,
        num_layers=1,
        num_heads=2,
        head_dim=2,
        max_seq_len=8,
    )


def _write_fixture(root: Path) -> Path:
    roles = root / "configs/roles.json"
    edge = root / "data/edge.npz"
    workload = root / "data/het.npz"
    roles.parent.mkdir(parents=True)
    edge.parent.mkdir(parents=True)
    roles.write_text('{"roles": "tiny"}\n')
    np.savez_compressed(edge, values=np.arange(4))
    lengths = np.asarray([6, 5, 4, 3], dtype=np.int16)
    old_lengths = lengths - 1
    offsets = np.asarray([0, 6, 11, 15, 18], dtype=np.int64)
    items = np.asarray(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
        ],
        dtype=np.int32,
    )
    arrays = {
        "record_user_ids": np.arange(10, 14, dtype=np.int64),
        "history_offsets": offsets,
        "history_item_idx": items,
        "old_start": np.zeros(4, dtype=np.int16),
        "old_length": old_lengths,
        "target_start": np.zeros(4, dtype=np.int16),
        "target_length": lengths,
        "het_old_valid_kv_bytes": old_lengths.astype(np.int64) * 16,
        "het_target_valid_kv_bytes": lengths.astype(np.int64) * 16,
        "owner_rank_1": np.zeros(4, dtype=np.int16),
        "owner_rank_2": np.asarray([0, 1, 0, 1], dtype=np.int16),
        "owner_rank_4": np.asarray([0, 1, 2, 3], dtype=np.int16),
    }
    np.savez_compressed(
        workload,
        **arrays,
        metadata_json=np.asarray(
            json.dumps({"content_sha256": "tiny"})
        ),
    )
    config = {
        "benchmark_id": "tiny-xp-baseline",
        "data": {
            "catalog": {
                "physical_rows": 17,
                "prediction_rows": 8,
            },
            "roles": {
                "path": str(roles.relative_to(root)),
                "sha256": file_sha256(roles),
            },
            "het_workload": {
                "path": str(workload.relative_to(root)),
                "sha256": file_sha256(workload),
            },
            "fixed_edge_inputs": {
                "path": str(edge.relative_to(root)),
                "sha256": file_sha256(edge),
            },
        },
        "model": {
            "layers": 1,
            "hidden_size": 4,
            "heads": 2,
            "head_dim": 2,
            "maximum_context": 8,
            "embedding_width": 8,
            "num_behaviors": 5,
        },
        "hardware": {
            "world_sizes_supported_by_code": [1, 2, 4],
        },
        "capacity_points": {
            "resident_m2": {
                "prefix_records": 4,
                "target_valid_bytes": 288,
            },
            "out_of_core_primary": [],
        },
    }
    path = root / "configs/evokv_baselines/tiny.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(config))
    return path


def _free_port() -> int:
    with socket.socket() as value:
        value.bind(("127.0.0.1", 0))
        return int(value.getsockname()[1])


def _distributed_worker(
    rank: int,
    world_size: int,
    root_value: str,
    port: int,
) -> None:
    root = Path(root_value)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    spec = _spec()
    torch.manual_seed(7)
    dense = ExternalEmbeddingHSTU(spec.hstu_config())
    embedding = TrainableProjectedModuloEmbedding.initialize(
        num_embeddings=spec.num_embeddings,
        embedding_width=spec.embedding_width,
        hidden_size=spec.hidden_size,
        rank=rank,
        world_size=world_size,
        device="cpu",
        embedding_seed=11,
        projection_seed=13,
    )
    tracker = OptimizerActiveRowTracker(
        num_embeddings=spec.num_embeddings,
        rank=rank,
        world_size=world_size,
    )
    checkpoint_root = root / "checkpoints"
    save_xp_projected_checkpoint(
        checkpoint_root,
        1,
        spec,
        dense,
        embedding,
        tracker,
        provenance={"fixture": "tiny"},
    )
    inputs = load_fixed_inputs(
        root / "configs/evokv_baselines/tiny.json",
        "resident_m2",
        world_size=world_size,
    )
    partial = run_partial_exact_baseline(
        inputs,
        checkpoint_root=checkpoint_root,
        checkpoint_version=1,
        fraction=0.5,
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        group_target_bytes=96,
        micro_batch_records=1,
        hash_mode="full",
    )
    exact = run_exact_baseline(
        inputs,
        checkpoint_root=checkpoint_root,
        checkpoint_version=1,
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        method="s1",
        endpoint="target",
        group_target_bytes=96,
        micro_batch_records=1,
        hash_mode="full",
    )
    if rank == 0:
        (root / "result.json").write_text(
            json.dumps({"partial": partial, "exact": exact})
        )
    dist.destroy_process_group()


def test_grouping_and_partial_selection_bind_fixed_records(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    inputs = load_fixed_inputs(
        config,
        "resident_m2",
        world_size=2,
    )
    groups = build_groups(
        inputs.records,
        world_size=2,
        group_target_bytes=144,
    )
    selected = select_partial_exact(
        inputs.records,
        0.5,
        str(inputs.bindings["benchmark_config"]["sha256"]),
    )
    assert len(groups) == 3
    assert sorted(
        record_id
        for group in groups
        for values in group.record_ids_by_rank
        for record_id in values
    ) == [0, 1, 2, 3]
    assert len(selected) == 2
    record = inputs.records[0]
    assert ordinal_inter_event_time_deltas(
        record,
        "old",
    ).tolist() == [0.0] + [1.0] * (record.old_length - 1)
    assert ordinal_inter_event_time_deltas(
        record,
        "target",
    ).tolist() == [0.0] + [1.0] * (
        record.target_length - 1
    )
    cropped = XPBaselineRecord(
        record_id=9,
        user_id=9,
        owner_rank=0,
        item_ids=np.arange(8, dtype=np.int64),
        old_start=2,
        old_length=4,
        target_start=3,
        target_length=5,
        old_valid_bytes=64,
        target_valid_bytes=80,
    )
    assert ordinal_inter_event_time_deltas(
        cropped,
        "old",
    ).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert ordinal_inter_event_time_deltas(
        cropped,
        "target",
    ).tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_two_rank_gloo_partial_and_two_slot_exact(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    assert json.loads(config.read_text())["benchmark_id"] == (
        "tiny-xp-baseline"
    )
    mp.spawn(
        _distributed_worker,
        args=(2, str(tmp_path), _free_port()),
        nprocs=2,
        join=True,
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["partial"]["partial_exact"]["exact_records"] == 2
    assert result["partial"]["partial_exact"]["stale_reuse_records"] == 2
    assert result["partial"]["records"] == 2
    assert result["exact"]["records"] == 4
    assert result["exact"]["records_expected"] == 4
    assert result["exact"]["method"] == "s1"
    assert result["exact"]["rank_reports"][0]["pipeline"][
        "whole_group_host_slots"
    ] == 2
    assert result["exact"]["rank_reports"][0]["pipeline"][
        "validation_and_pageable_publication_overlap"
    ]
    assert result["exact"]["lookup"]["requested_tokens"] == 18
    assert result["exact"]["d2h_bytes"] == 288
    assert result["exact"]["output_hash"]["sha256"]
    assert result["exact"]["checkpoint"]["provenance"] == {
        "fixture": "tiny"
    }
    assert result["exact"]["checkpoint"]["version"] == 1
    assert result["exact"]["world_size"] == 2
    assert result["exact"]["bindings"]["benchmark_config"]["sha256"]


def test_single_rank_same_arena_old_then_target_rolls_forward(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    spec = _spec()
    torch.manual_seed(17)
    dense = ExternalEmbeddingHSTU(spec.hstu_config())
    embedding = TrainableProjectedModuloEmbedding.initialize(
        num_embeddings=spec.num_embeddings,
        embedding_width=spec.embedding_width,
        hidden_size=spec.hidden_size,
        rank=0,
        world_size=1,
        device="cpu",
        embedding_seed=19,
        projection_seed=23,
    )
    tracker = OptimizerActiveRowTracker(
        num_embeddings=spec.num_embeddings,
        rank=0,
        world_size=1,
    )
    checkpoint_root = tmp_path / "checkpoints"
    save_xp_projected_checkpoint(
        checkpoint_root,
        1,
        spec,
        dense,
        embedding,
        tracker,
        provenance={"fixture": "rolling"},
    )
    inputs = load_fixed_inputs(
        config,
        "resident_m2",
        world_size=1,
    )
    store = tmp_path / "rolling"
    source = run_exact_baseline(
        inputs,
        checkpoint_root=checkpoint_root,
        checkpoint_version=1,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        method="s0",
        endpoint="old",
        group_target_bytes=144,
        micro_batch_records=1,
        hash_mode="full",
        store_path=store,
        store_mode="create",
    )
    assert source is not None
    journal_path = Path(
        f"{store}.rank00.dram.ledger.json"
    )
    assert json.loads(journal_path.read_text())["phase"] == (
        "source_complete"
    )
    target = run_exact_baseline(
        inputs,
        checkpoint_root=checkpoint_root,
        checkpoint_version=1,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        method="s1",
        endpoint="target",
        group_target_bytes=144,
        micro_batch_records=1,
        hash_mode="full",
        store_path=store,
        store_mode="open",
        source_manifest=source,
    )
    assert target is not None
    assert target["transaction"][
        "old_extent_reclaimed_by_same-arena overwrite"
    ]
    assert target["rank_reports"][0]["store"]["complete_records"] == 4
    assert target["rank_reports"][0]["store"]["payload_nbytes"] == 288
    assert target["written_bytes"] == 288
    journal = json.loads(journal_path.read_text())
    assert journal["phase"] == "target_complete"
    assert len(journal["target_commits"]) == target["groups"]
    assert target["transaction"]["reclaimed_old_valid_bytes"] == 224
