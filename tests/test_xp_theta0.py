from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import torch

from hstu_kvcache.migration.foundation_workload import (
    array_sha256 as foundation_array_sha256,
)
from hstu_kvcache.streaming.sharded_edge import (
    SHARDED_EDGE_CHECKPOINT_SCHEMA,
    ExternalEmbeddingHSTU,
    ShardedEdgeModelSpec,
    modulo_local_rows,
)
from hstu_kvcache.streaming.xp_theta0 import (
    StructuredSemiOrthogonalExpansion,
    deterministic_cross_user_negatives,
    file_sha256,
)


def test_structured_expansion_is_semiorthogonal_and_preserves_rows() -> None:
    source = torch.arange(
        21,
        dtype=torch.float32,
    ).reshape(7, 3) / 11.0
    expansion = StructuredSemiOrthogonalExpansion(
        source_width=3,
        target_width=8,
    )
    projection = expansion.projection_weight(device="cpu")
    expanded = expansion.expand_rows(source, row_chunk=2)
    oracle = expansion.numeric_oracle(
        source,
        expanded,
        projection,
        maximum_samples=7,
        nullspace_norm_ratio=0.05,
    )
    torch.testing.assert_close(
        projection.matmul(projection.transpose(0, 1)),
        torch.eye(3),
    )
    torch.testing.assert_close(
        torch.nn.functional.linear(expanded, projection),
        source,
    )
    assert oracle["all_target_coordinates_used"] is True
    assert oracle["max_abs_error"] < 1e-6
    assert oracle["nullspace_energy"] > 0
    assert oracle["sampled_projected_nullspace_max_abs"] < 1e-6


def test_cross_user_negatives_are_real_and_distinct() -> None:
    anchor = np.arange(1, 9, dtype=np.int32)
    positive = np.asarray(
        [2, 3, 4, 5, 6, 7, 8, 1],
        dtype=np.int32,
    )
    users = np.arange(100, 108, dtype=np.int64)
    negative = deterministic_cross_user_negatives(
        anchor,
        positive,
        users,
        initial_stride=3,
    )
    assert np.all(negative != anchor)
    assert np.all(negative != positive)
    assert set(negative).issubset(set(positive))


def _artifact_record(path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _write_source_checkpoint(root) -> ShardedEdgeModelSpec:
    spec = ShardedEdgeModelSpec(
        num_embeddings=13,
        num_prediction_items=6,
        num_behaviors=2,
        hidden_size=3,
        num_layers=1,
        num_heads=1,
        head_dim=3,
        max_seq_len=8,
    )
    directory = root / "theta_0"
    directory.mkdir(parents=True)
    torch.manual_seed(19)
    dense = ExternalEmbeddingHSTU(spec.hstu_config())
    dense_path = directory / "dense.pt"
    torch.save(
        {
            "schema": SHARDED_EDGE_CHECKPOINT_SCHEMA,
            "version": 0,
            "config": asdict(spec.hstu_config()),
            "state_dict": dense.state_dict(),
        },
        dense_path,
    )
    full = torch.arange(
        spec.num_embeddings * spec.hidden_size,
        dtype=torch.float32,
    ).reshape(spec.num_embeddings, spec.hidden_size) / 37.0
    full[0].zero_()
    shards = []
    for rank in range(2):
        local = full[rank::2].clone()
        path = directory / f"embedding_rank_{rank:05d}.pt"
        torch.save(
            {
                "schema": SHARDED_EDGE_CHECKPOINT_SCHEMA,
                "version": 0,
                "rank": rank,
                "world_size": 2,
                "num_embeddings": spec.num_embeddings,
                "hidden_size": spec.hidden_size,
                "global_row_start": rank,
                "global_row_stride": 2,
                "local_rows": modulo_local_rows(
                    spec.num_embeddings,
                    rank,
                    2,
                ),
                "local_weight": local,
            },
            path,
        )
        shards.append(
            {
                "rank": rank,
                **_artifact_record(path),
                "local_rows": len(local),
                "global_row_start": rank,
                "global_row_stride": 2,
            }
        )
    manifest = {
        "schema": SHARDED_EDGE_CHECKPOINT_SCHEMA,
        "version": 0,
        "world_size": 2,
        "spec": asdict(spec),
        "dense": _artifact_record(dense_path),
        "embedding_shards": shards,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return spec


def _write_pairs(path, summary_path, spec) -> None:
    anchor = np.arange(1, spec.num_embeddings, dtype=np.int32)
    flags = np.ones(len(anchor), dtype=np.uint8)
    flags[-2:] = 0
    positive = np.asarray(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 11, 12],
        dtype=np.int32,
    )
    users = np.arange(200, 212, dtype=np.int32)
    metadata = {
        "protocol": (
            "evokv_qk_xp_base_row_coverage_development_v0"
        ),
        "scientific_result": False,
        "content_sha256": "small-pair-content",
        "base_only_boundary": {
            "post_base_rows_used": False,
        },
    }
    np.savez_compressed(
        path,
        anchor_row=anchor,
        positive_row=positive,
        occurrence_user_id=users,
        has_same_user_neighbor=flags,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
    )
    summary = {
        **metadata,
        "artifact": {
            "file_sha256": file_sha256(path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def _write_foundation_workload(path) -> None:
    rows = np.asarray([1, 4, 11], dtype=np.int32)
    metadata = {
        "semantic_request_union": {
            "unique_rows": len(rows),
            "unique_rows_sha256": foundation_array_sha256(rows),
        }
    }
    np.savez_compressed(
        path,
        semantic_request_union_item_idx=rows,
        semantic_request_union_eligible_for_update=np.ones(
            len(rows),
            dtype=np.uint8,
        ),
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
    )


def test_two_rank_cpu_theta0_builder_canary(tmp_path) -> None:
    source = tmp_path / "source"
    spec = _write_source_checkpoint(source)
    pairs = tmp_path / "pairs.npz"
    pair_summary = tmp_path / "pairs.json"
    _write_pairs(pairs, pair_summary, spec)
    workload = tmp_path / "foundation.npz"
    _write_foundation_workload(workload)
    target = tmp_path / "target"
    output = tmp_path / "result.json"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "scripts/build_evokv_xp_theta0.py",
            "--source-checkpoint-root",
            str(source),
            "--cooccurrence",
            str(pairs),
            "--cooccurrence-summary",
            str(pair_summary),
            "--foundation-workload",
            str(workload),
            "--checkpoint-root",
            str(target),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--development-canary",
            "--target-embedding-width",
            "8",
            "--batch-size",
            "5",
            "--expected-neighbor-rows",
            "10",
            "--expected-isolated-rows",
            "2",
            "--minimum-active-rows",
            "8",
            "--negative-stride",
            "3",
        ],
        cwd=os.getcwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        completed.stdout + completed.stderr
    )
    result = json.loads(output.read_text())
    assert result["status"] == "complete"
    assert result["scientific_result"] is False
    assert result["training"]["global_pairs"] == 10
    assert result["training"]["isolated_rows_used"] is False
    expansion = result["structured_expansion"]
    assert (
        expansion["oracle_before_training"]["nullspace_energy"]
        > 0
    )
    assert (
        expansion["oracle_before_training"][
            "sampled_projected_nullspace_max_abs"
        ]
        < 1e-6
    )
    assert (
        expansion[
            "projection_response_to_initial_nullspace_after_training"
        ]["responding_basis_directions"]
        > 0
    )
    assert result["optimizer_active_gate"]["passed"] is True
    assert (
        result["optimizer_active_gate"]["observed_active_rows"]
        == 10
    )
    coverage = result["semantic_request_union"][
        "optimizer_active_coverage"
    ]
    assert coverage["passed"] is False
    assert coverage["missing_row_ids"] == [11]
    ledger = result["optimizer_active_gate"]["ledger"]
    assert ledger["global_optimizer_update_events"] >= 10
    assert ledger["active_update_count_minimum"] >= 1
    assert (
        target / "theta_0" / "manifest.json"
    ).is_file()
