from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from hstu_kvcache.migration.foundation_workload import (
    FoundationConfig,
    allocate_roles,
    array_sha256,
    layout_from_lengths,
    load_upstream_training_users,
    run,
    stable_owner_ranks,
)


def write_fixture(path: Path) -> tuple[np.ndarray, dict[int, list[int]]]:
    sequences = {
        10: [1, 2, 3, 9, 1, 2, 3, 4, 5, 6, 7, 8],
        20: [2, 3, 4, 10, 2, 3, 4, 5, 6, 7],
        30: [3, 4, 5, 11, 3, 4, 5, 6, 7],
        40: [4, 5, 6, 12, 4, 5, 6, 7],
        50: [5, 6, 7, 13, 5, 6, 7],
        60: [6, 7, 8, 14, 6, 7],
        70: [7, 8, 1, 15, 7, 8],
        80: [8, 1, 2, 16, 8],
        90: [1, 3, 5, 17, 1],
    }
    rows = []
    for position in range(max(map(len, sequences.values()))):
        for user_id, items in sequences.items():
            if position < len(items):
                rows.append(
                    {"user_id": user_id, "item_id": items[position]}
                )
    csv_path = path.parent / "QK-video.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(csv_path, "Tenrec/QK-video.csv")
    catalog_items = np.arange(1, 9, dtype=np.int64)
    return catalog_items, sequences


def write_catalog(path: Path, source: Path, items: np.ndarray) -> None:
    with zipfile.ZipFile(source) as archive:
        info = archive.getinfo("Tenrec/QK-video.csv")
    metadata = {
        "num_prediction_items": 3,
        "context_entity_rows": 5,
        "base_entity_item_ids_sha256": array_sha256(items),
        "source": {
            "member": "Tenrec/QK-video.csv",
            "member_size_bytes": info.file_size,
            "member_compressed_size_bytes": info.compress_size,
            "member_crc32": f"{info.CRC:08x}",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        base_entity_original_item_ids=items,
        base_item_frequencies=np.arange(8, 0, -1, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def make_config(tmp_path: Path) -> FoundationConfig:
    source = tmp_path / "Tenrec.zip"
    items, _ = write_fixture(source)
    catalog = tmp_path / "catalog.npz"
    write_catalog(catalog, source, items)
    return FoundationConfig(
        source=source,
        member="Tenrec/QK-video.csv",
        catalog_cache=catalog,
        length_cache=tmp_path / "lengths.npz",
        output=tmp_path / "workload.npz",
        summary=tmp_path / "summary.json",
        roles=tmp_path / "roles.json",
        hash_salt="fixture-foundation-v1",
        theta12_users=1,
        theta01_users=1,
        fit_users=1,
        profile_users=1,
        qualification_users=1,
        final_users=4,
        minimum_events=5,
        theta12_minimum_events=12,
        history_horizon=6,
        target_horizon=4,
        append_events=2,
        model_layers=1,
        model_hidden_size=2,
        kv_element_bytes=2,
        capacity_gib=(4 / (1 << 30), 16 / (1 << 30)),
        chunk_size=7,
    )


def test_role_assignment_is_disjoint_stable_and_long_first(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    user_ids = np.arange(10, 100, 10, dtype=np.int64)
    lengths = np.array([12, 10, 9, 8, 7, 6, 6, 5, 5], dtype=np.int32)

    first, audit = allocate_roles(user_ids, lengths, config)
    second, _ = allocate_roles(user_ids, lengths, config)

    assert first["theta12"].tolist() == [10]
    assert all(
        np.array_equal(first[name], second[name]) for name in first
    )
    selected = np.concatenate(list(first.values()))
    assert len(np.unique(selected)) == len(selected) == 9
    assert audit["post_base_roles_pairwise_disjoint"]
    assert audit["base_builder"]["user_exclusion"] is False
    assert audit["theta12_backfill_selected_users"] == 0


def test_upstream_training_users_are_excluded_before_role_assignment(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    user_ids = np.arange(10, 120, 10, dtype=np.int64)
    lengths = np.array(
        [12, 10, 10, 10, 9, 8, 8, 8, 8, 8, 8],
        dtype=np.int32,
    )
    excluded = np.array([10, 40], dtype=np.int64)

    roles, audit = allocate_roles(
        user_ids,
        lengths,
        config,
        excluded,
    )

    selected = np.concatenate(list(roles.values()))
    assert len(selected) == 9
    assert not np.isin(selected, excluded).any()
    assert roles["theta12"].tolist() != [10]
    assert len(roles["theta12"]) == 1
    assert lengths[np.flatnonzero(user_ids == roles["theta12"][0])[0]] >= 8
    assert audit["eligible_users_before_upstream_exclusion"] == 11
    assert audit["theta12_preferred_selected_users"] == 0
    assert audit["theta12_backfill_selected_users"] == 1
    exclusion = audit["upstream_training_exclusion"]
    assert exclusion["excluded_minimum_event_eligible_users"] == 2
    assert exclusion["selected_role_overlap"] == 0


def test_upstream_prepared_identity_and_hash_are_recorded(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    with zipfile.ZipFile(config.source) as archive:
        info = archive.getinfo(config.member)
    upstream = tmp_path / "upstream.npz"
    users = np.array([10, 20], dtype=np.int64)
    metadata = {
        "protocol": "fixture_upstream_v0",
        "selected_users": len(users),
        "source": {
            "member": config.member,
            "member_size_bytes": info.file_size,
            "member_compressed_size_bytes": info.compress_size,
            "member_crc32": f"{info.CRC:08x}",
        },
    }
    np.savez_compressed(
        upstream,
        original_user_ids=users,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    config = FoundationConfig(
        **{
            **config.__dict__,
            "upstream_prepared": upstream,
        }
    )
    fingerprint = {
        "member": config.member,
        "member_size_bytes": info.file_size,
        "member_compressed_size_bytes": info.compress_size,
        "member_crc32": f"{info.CRC:08x}",
    }

    loaded, audit = load_upstream_training_users(config, fingerprint)

    assert np.array_equal(loaded, users)
    assert audit["enabled"]
    assert audit["excluded_users"] == 2
    assert audit["excluded_user_ids_sha256"] == array_sha256(users)


def test_het_layout_covers_growth_and_sliding_records(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    config = FoundationConfig(
        **{
            **config.__dict__,
            "history_horizon": 544,
            "target_horizon": 512,
            "append_events": 32,
        }
    )

    layout = layout_from_lengths(
        np.array([96, 512, 544, 800], dtype=np.int32),
        config,
    )

    assert layout["old_length"].tolist() == [64, 480, 512, 512]
    assert layout["target_length"].tolist() == [96, 512, 512, 512]
    assert layout["retained_length"].tolist() == [64, 480, 480, 480]
    assert layout["evicted_length"].tolist() == [0, 0, 32, 32]
    assert layout["append_length"].tolist() == [32, 32, 32, 32]
    assert layout["hom_old_left_padding"].tolist() == [448, 32, 0, 0]
    assert layout["hom_target_left_padding"].tolist() == [416, 0, 0, 0]
    assert layout["hom_old_allocated_length"].tolist() == [512] * 4
    assert layout["hom_target_allocated_length"].tolist() == [512] * 4


def test_audit_then_materialize_reuses_lengths_and_emits_union(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)

    audit = run(config, audit_only=True)

    assert audit["status"] == "audit_only"
    assert config.length_cache.exists()
    assert config.roles.exists()
    assert config.summary.exists()
    assert not config.output.exists()
    assert audit["xp_embedding"]["physical_rows"] == 9
    assert audit["xp_embedding"]["global_bytes"] == 9 * 4_096 * 4
    assert (
        audit["xp_embedding"]["dense_parameter_count_including_projection"]
        == config.dense_parameter_count
    )
    assert (
        audit["xp_embedding"]["global_embedding_plus_dense_bytes"]
        == 9 * 4_096 * 4 + config.dense_parameter_count * 4
    )
    assert audit["xp_embedding"]["optimizer_active_gate"] == "pending"
    assert (
        sum(
            audit["byte_and_owner_ledger"]["ranks"]["2"][
                "record_count"
            ]
        )
        == config.final_users
    )
    role_doc = json.loads(config.roles.read_text())
    role_ids = [
        user_id
        for role in role_doc["roles"].values()
        for user_id in role["user_ids"]
    ]
    assert len(role_ids) == len(set(role_ids)) == 9

    import hstu_kvcache.migration.foundation_workload as module

    monkeypatch.setattr(
        module,
        "build_user_lengths",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("length cache was not reused")
        ),
    )
    result = run(config)

    assert result["status"] == "materialized"
    assert result["gates"]["foundation_workload"] == "pass"
    assert result["gates"]["optimizer_active"] == "pending"
    assert result["semantic_request_union"]["model_edges"] == [
        "theta0_to_theta1",
        "theta1_to_theta2",
    ]
    assert (
        result["semantic_request_union"][
            "catalog_frequency_is_optimizer_activity"
        ]
        is False
    )
    with np.load(config.output, allow_pickle=False) as output:
        metadata = json.loads(str(output["metadata_json"].item()))
        offsets = output["history_offsets"]
        boundary = output["boundary_b"]
        target_lengths = output["target_length"]
        assert np.diff(offsets).tolist() == boundary.tolist()
        assert np.all(target_lengths <= config.target_horizon)
        assert output["capacity_prefix_records"].tolist() == [1, 1]
        users = output["record_user_ids"]
        assert output["owner_rank_2"].tolist() == stable_owner_ranks(
            users,
            config.hash_salt,
            2,
        ).tolist()
        assert output["owner_rank_4"].tolist() == stable_owner_ranks(
            users,
            config.hash_salt,
            4,
        ).tolist()
        assert output["het_old_valid_kv_bytes"].tolist() == (
            output["old_length"].astype(np.int64)
            * config.kv_bytes_per_token
        ).tolist()
        assert output["het_target_valid_kv_bytes"].tolist() == (
            target_lengths.astype(np.int64)
            * config.kv_bytes_per_token
        ).tolist()
        assert output["hom_old_allocated_kv_bytes"].tolist() == [
            config.target_horizon * config.kv_bytes_per_token
        ] * config.final_users
        assert output["hom_target_allocated_kv_bytes"].tolist() == [
            config.target_horizon * config.kv_bytes_per_token
        ] * config.final_users
        assert int(output["total_het_target_valid_kv_bytes"]) == int(
            output["het_target_valid_kv_bytes"].sum()
        )
        assert (
            output["capacity_owner2_record_count"].sum(axis=1).tolist()
            == output["capacity_prefix_records"].tolist()
        )
        assert (
            output["capacity_owner4_record_count"].sum(axis=1).tolist()
            == output["capacity_prefix_records"].tolist()
        )
        request_rows = output["semantic_request_union_item_idx"]
        assert len(request_rows) == result["semantic_request_union"]["unique_rows"]
        assert (
            array_sha256(request_rows)
            == result["semantic_request_union"]["unique_rows_sha256"]
        )
        assert metadata["hom_descriptor"]["same_record"]
        assert metadata["optimizer_active_gate"] == "pending"
