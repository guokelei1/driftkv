from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hstu_kvcache.data.qk_base_row_coverage as module
from hstu_kvcache.data.qk_base_row_coverage import (
    CoverageConfig,
    array_sha256,
    run,
)


def write_source(path: Path) -> dict[int, list[int]]:
    sequences = {
        10: [1, 2, 9, 9],
        20: [3],
        30: [3, 4, 10],
        40: [5],
    }
    rows = []
    for position in range(max(map(len, sequences.values()))):
        for user_id, items in sequences.items():
            if position < len(items):
                rows.append(
                    {"user_id": user_id, "item_id": items[position]}
                )
    csv = path.parent / "QK-video.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(csv, "Tenrec/QK-video.csv")
    return sequences


def fingerprint(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("Tenrec/QK-video.csv")
    return {
        "path": str(path.resolve()),
        "archive_size_bytes": path.stat().st_size,
        "member": "Tenrec/QK-video.csv",
        "member_size_bytes": info.file_size,
        "member_compressed_size_bytes": info.compress_size,
        "member_crc32": f"{info.CRC:08x}",
    }


def write_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    source = tmp_path / "Tenrec.zip"
    sequences = write_source(source)
    source_identity = fingerprint(source)
    catalog = tmp_path / "catalog.npz"
    original = np.arange(1, 6, dtype=np.int64)
    frequencies = np.array([1, 1, 2, 1, 1], dtype=np.int64)
    catalog_metadata = {
        "base_prefix_raw_events": 2,
        "base_entity_item_ids_sha256": array_sha256(original),
        "source": source_identity,
    }
    np.savez_compressed(
        catalog,
        base_entity_original_item_ids=original,
        base_item_frequencies=frequencies,
        metadata_json=np.asarray(
            json.dumps(catalog_metadata, sort_keys=True)
        ),
    )
    lengths = tmp_path / "lengths.npz"
    user_ids = np.asarray(sorted(sequences), dtype=np.int64)
    raw_lengths = np.asarray(
        [len(sequences[int(user_id)]) for user_id in user_ids],
        dtype=np.int32,
    )
    length_metadata = {
        "source": source_identity,
        "user_ids_sha256": array_sha256(user_ids),
    }
    np.savez_compressed(
        lengths,
        user_ids=user_ids,
        raw_lengths=raw_lengths,
        metadata_json=np.asarray(
            json.dumps(length_metadata, sort_keys=True)
        ),
    )
    return source, catalog, lengths


def make_config(tmp_path: Path) -> CoverageConfig:
    source, catalog, lengths = write_inputs(tmp_path)
    return CoverageConfig(
        source=source,
        member="Tenrec/QK-video.csv",
        catalog_cache=catalog,
        user_length_cache=lengths,
        cache_dir=tmp_path / "cache",
        output=tmp_path / "coverage.npz",
        summary=tmp_path / "summary.json",
        base_prefix=2,
        chunk_size=3,
        checkpoint_every_chunks=1,
        derive_user_block=8,
    )


def test_recoverable_scan_and_base_only_pair_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)
    original_reader = module.read_qk_chunks

    def interrupted(config_value: CoverageConfig):
        for index, chunk in enumerate(original_reader(config_value)):
            if index == 2:
                raise RuntimeError("injected interruption")
            yield chunk

    monkeypatch.setattr(module, "read_qk_chunks", interrupted)
    with pytest.raises(RuntimeError, match="injected interruption"):
        run(config, audit_only=True)
    state = json.loads(config.state_path.read_text())
    assert state["completed_chunks"] == 2
    assert not state["complete"]

    monkeypatch.setattr(module, "read_qk_chunks", original_reader)
    audit = run(config, audit_only=True)

    assert audit["status"] == "audit_only"
    assert audit["scan"]["complete"]
    assert audit["scan"]["base_rows_retained"] == 6
    assert audit["scan"]["post_base_rows_ignored"] == 3
    assert audit["coverage"]["covered_rows"] == 5
    assert audit["coverage"]["neighbor_rows"] == 4
    assert audit["coverage"]["isolated_fallback_rows"] == 1
    assert not config.output.exists()

    def forbidden_reader(_: CoverageConfig):
        raise AssertionError("complete scan cache was not reused")
        yield

    monkeypatch.setattr(module, "read_qk_chunks", forbidden_reader)
    result = run(config)

    assert result["status"] == "materialized"
    assert result["optimizer_active_gate"] == "pending_training"
    assert result["base_only_boundary"]["post_base_rows_used"] is False
    with np.load(config.output, allow_pickle=False) as artifact:
        assert artifact["anchor_row"].tolist() == [1, 2, 3, 4, 5]
        assert artifact["base_frequency"].tolist() == [1, 1, 2, 1, 1]
        assert artifact["has_same_user_neighbor"].tolist() == [
            1,
            1,
            1,
            1,
            0,
        ]
        assert artifact["positive_row"].tolist() == [2, 1, 4, 3, 5]
        assert artifact["occurrence_user_id"].tolist() == [
            10,
            10,
            30,
            30,
            40,
        ]
        metadata = json.loads(str(artifact["metadata_json"].item()))
        assert metadata["coverage"]["catalog_frequency_exact_match"]
        assert metadata["base_only_boundary"]["d1_actions_used"] is False
