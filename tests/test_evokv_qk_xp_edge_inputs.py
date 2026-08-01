from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from hstu_kvcache.data.qk_xp_edge_inputs import (
    EdgeInputConfig,
    array_sha256,
    run,
    splitmix64,
)


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


def write_source(path: Path) -> None:
    sequences = {
        10: [
            (1, 0, 0, 0, 0),
            (2, 1, 0, 1, 0),
            (99, 0, 0, 0, 1),
            (98, 1, 0, 0, 0),
        ],
        20: [
            (2, 0, 0, 0, 0),
            (3, 0, 1, 1, 0),
            (4, 0, 0, 0, 0),
            (5, 0, 0, 0, 0),
            (6, 1, 0, 0, 0),
            (97, 1, 0, 0, 0),
        ],
        30: [
            (1, 0, 0, 0, 0),
            (3, 0, 0, 0, 0),
            (4, 0, 1, 0, 0),
            (96, 1, 0, 0, 0),
        ],
        40: [(1, 1, 0, 0, 0)],
        50: [(2, 1, 0, 0, 0)],
        60: [(3, 1, 0, 0, 0)],
    }
    rows = []
    for position in range(max(map(len, sequences.values()))):
        for user_id, events in sequences.items():
            if position < len(events):
                item, click, follow, like, share = events[position]
                rows.append(
                    {
                        "user_id": user_id,
                        "item_id": item,
                        "click": click,
                        "follow": follow,
                        "like": like,
                        "share": share,
                    }
                )
    csv = path.parent / "QK-video.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(csv, "Tenrec/QK-video.csv")


def write_catalog(path: Path, source: Path) -> None:
    original = np.arange(1, 6, dtype=np.int64)
    metadata = {
        "base_prefix_raw_events": 2,
        "num_prediction_items": 2,
        "context_entity_rows": 3,
        "base_entity_item_ids_sha256": array_sha256(original),
        "source": fingerprint(source),
    }
    np.savez_compressed(
        path,
        base_entity_original_item_ids=original,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def write_roles(path: Path) -> None:
    roles = {
        "theta01": np.asarray([10], dtype=np.int64),
        "theta12": np.asarray([20], dtype=np.int64),
        "qualification": np.asarray([30], dtype=np.int64),
        "fit": np.asarray([40], dtype=np.int64),
        "profile": np.asarray([50], dtype=np.int64),
        "final": np.asarray([60], dtype=np.int64),
    }
    document = {
        "hash_salt": "fixture-successor-v1",
        "protocol": "fixture_roles",
        "roles": {
            name: {
                "count": len(values),
                "user_ids": values.tolist(),
                "user_ids_sha256": array_sha256(values),
            }
            for name, values in roles.items()
        },
    }
    path.write_text(json.dumps(document))


def make_config(tmp_path: Path) -> EdgeInputConfig:
    source = tmp_path / "Tenrec.zip"
    write_source(source)
    catalog = tmp_path / "catalog.npz"
    write_catalog(catalog, source)
    roles = tmp_path / "roles.json"
    write_roles(roles)
    return EdgeInputConfig(
        source=source,
        member="Tenrec/QK-video.csv",
        catalog_cache=catalog,
        roles=roles,
        output=tmp_path / "edges.npz",
        summary=tmp_path / "summary.json",
        hash_salt="fixture-successor-v1",
        prediction_catalog_size=2,
        base_prefix=2,
        theta01_history_end=2,
        theta01_update_end=3,
        theta12_history_end=4,
        theta12_update_end=5,
        qualification_history_end=2,
        qualification_update_end=3,
        theta01_users=1,
        theta12_users=1,
        qualification_users=1,
        chunk_size=4,
    )


def test_fixed_edges_preserve_raw_semantics_and_exclude_other_roles(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    summary = run(config)

    assert summary["status"] == "pass"
    assert summary["records"] == {
        "theta01": 1,
        "theta12": 1,
        "qualification": 1,
    }
    assert summary["rows"]["total"] == 11
    assert summary["scan"]["selected_post_window_rows_ignored"] == 3
    with np.load(config.output, allow_pickle=False) as source:
        arrays = {
            name: source[name]
            for name in source.files
            if name != "metadata_json"
        }
    assert arrays["record_user_ids"].tolist() == [10, 20, 30]
    assert arrays["record_offsets"].tolist() == [0, 3, 8, 11]
    assert arrays["raw_ordinal"].tolist() == [
        0,
        1,
        2,
        0,
        1,
        2,
        3,
        4,
        0,
        1,
        2,
    ]
    assert arrays["behavior"][:3].tolist() == [1, 3, 5]
    assert arrays["action_mask"][:3].tolist() == [0, 5, 8]
    assert arrays["raw_label"][:3].tolist() == [0, 1, 1]
    assert arrays["label"][:3].tolist() == [0, 1, 0]
    expected_fallback = (
        3
        + int(splitmix64(np.asarray([99], dtype=np.int64))[0] % 3)
    )
    assert int(arrays["item_idx"][2]) == expected_fallback
    assert arrays["is_stream_only_fallback"][2] == 1
    assert not np.isin(
        arrays["record_user_ids"],
        np.asarray([40, 50, 60]),
    ).any()
    assert not np.isin(
        arrays["original_item_id"],
        np.asarray([96, 97, 98]),
    ).any()
