from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hstu_kvcache.data import load_prepared_exposure_plan

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "build_evokv_design3_m1_qk_entity_data.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_evokv_design3_m1_qk_entity_data",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_qk_archive(path: Path) -> None:
    sequences = {
        10: [1, 2, 9, 9, 1, 2, 1, 2],
        20: [1, 3, 9, 9, 1, 1, 1, 1],
        30: [1, 4, 9, 9, 1, 1, 1, 1],
        40: [1, 5, 9, 9, 1, 1, 1, 1],
    }
    rows = []
    for position in range(8):
        for user_id, items in sequences.items():
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": items[position],
                    "click": int(position % 2 == 0),
                    "follow": 0,
                    "like": int(position == 3),
                    "share": 0,
                    "watching_times": 1,
                }
            )
    csv_path = path.parent / "QK-video.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(csv_path, "Tenrec/QK-video.csv")


def tiny_config(tmp_path: Path) -> MODULE.BuildConfig:
    return MODULE.BuildConfig(
        source=tmp_path / "Tenrec.zip",
        cache_dir=tmp_path / "entity_cache",
        output=tmp_path / "entity_processed.npz",
        manifest=tmp_path / "entity_manifest.json",
        cohort_ids=tmp_path / "entity_cohorts.json",
        prediction_catalog_size=1,
        base_prefix=2,
        history_length=3,
        slide=1,
        fit_calibration_users=1,
        cohort_sizes=(1, 2),
        primary_cohort_size=2,
        embedding_width=4,
        model_layers=2,
        model_heads=2,
        model_head_dim=2,
        chunk_size=3,
    )


def test_entity_builder_defaults_bind_main_m1_shape() -> None:
    config = MODULE.BuildConfig()
    assert config.output == Path(
        "data/processed/evokv_d3_m1_qk_entity_2560.npz"
    )
    assert config.fit_calibration_users == 512
    assert config.cohort_sizes == (512, 1_024, 2_048)
    assert config.primary_cohort_size == 2_048
    assert config.history_length == 512
    assert config.slide == 32
    assert config.required_events == 576
    assert config.embedding_width == 1_536
    assert config.model_layers == 24
    assert config.model_heads == 24
    assert config.model_head_dim == 64


def test_entity_builder_freezes_base_rows_and_reuses_phase_caches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tiny_config(tmp_path)
    write_qk_archive(config.source)

    audit = MODULE.run(config, audit_only=True)

    assert audit["status"] == "audit_only"
    assert audit["cohorts"]["eligible_users"] == 4
    assert audit["cohorts"]["primary_benchmark_users"] == 2
    assert not config.output.exists()
    catalog, catalog_metadata = MODULE.load_npz(
        config.catalog_cache
    )
    assert catalog["original_item_ids"].tolist() == [1]
    assert catalog["base_entity_original_item_ids"].tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert catalog["base_item_frequencies"].tolist() == [
        4,
        1,
        1,
        1,
        1,
    ]
    assert catalog["is_prediction_item"].tolist() == [
        1,
        0,
        0,
        0,
        0,
    ]
    assert catalog_metadata["base_entity_items"] == 5
    assert catalog_metadata["context_entity_rows"] == 4
    assert catalog_metadata["base_rows"] == 8
    cohort, cohort_metadata = MODULE.load_npz(
        config.cohort_cache
    )
    expected = MODULE.stable_user_order(
        np.array([10, 20, 30, 40]),
        config.hash_salt,
    )
    assert cohort["fit_calibration_user_ids"].tolist() == (
        expected[:1].tolist()
    )
    assert cohort["benchmark_user_ids"].tolist() == (
        expected[1:3].tolist()
    )
    assert set(cohort["fit_calibration_user_ids"]).isdisjoint(
        set(cohort["benchmark_user_ids"])
    )
    assert cohort_metadata["prediction_rows"] == 18
    assert cohort_metadata["exact_context_rows"] == 6
    assert cohort_metadata["stream_only_fallback_rows"] == 8
    assert (
        cohort_metadata["unique_stream_only_original_items_seen"]
        == 1
    )
    assert (
        cohort_metadata["unique_fallback_context_entities_touched"]
        == 1
    )
    assert cohort_metadata["base_entity_direct_coverage"] == 1.0

    monkeypatch.setattr(
        MODULE,
        "build_catalog_cache",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("catalog cache was not reused")
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "build_cohort_cache",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("cohort cache was not reused")
        ),
    )
    result = MODULE.run(config)

    assert result["status"] == "materialized"
    output = result["output"]
    assert output["rows"] == 15
    assert output["prediction_rows"] == 6
    assert output["exact_context_rows"] == 3
    assert output["stream_only_fallback_rows"] == 6
    assert output["unique_mapped_entity_rows_accessed"] == 4
    assert output["split_rows"] == {
        "base": 9,
        "window_0": 3,
        "window_1": 3,
    }
    assert output["split_positive_rows"] == {
        "base": 3,
        "window_0": 0,
        "window_1": 3,
    }
    assert output["window_count"] == 2
    assert output["old_window_filtered_positions"] == [0, 3]
    assert output["target_window_filtered_positions"] == [1, 4]
    assert output["update_window_filtered_positions"] == [3, 4]
    assert output["heldout_window_filtered_positions"] == [4, 5]
    assert (
        audit["catalog"]["base_entity_item_ids_sha256"]
        == output["base_entity_item_ids_sha256"]
    )


def test_entity_builder_direct_context_fallback_and_loader_contract(
    tmp_path: Path,
) -> None:
    config = tiny_config(tmp_path)
    write_qk_archive(config.source)

    MODULE.run(config)

    with np.load(config.output, allow_pickle=False) as source:
        metadata = json.loads(
            str(source["metadata_json"].item())
        )
        users = source["user_idx"]
        items = source["item_idx"]
        labels = source["label"]
        raw = source["raw_ordinal"]
        windows = source["window_index"]
        fallback = source[
            "is_stream_only_fallback"
        ].astype(bool)
        expected_fallback = (
            config.prediction_catalog_size
            + 1
            + int(
                MODULE.splitmix64(
                    np.array([9], dtype=np.int64)
                )[0]
                % np.uint64(4)
            )
        )
        assert expected_fallback == 2
        assert np.all(items[fallback] == expected_fallback)
        assert np.all(items[(raw == 2) | (raw == 3)] == 2)
        assert np.all(labels[items > 1] == 0)
        assert np.all(labels[fallback] == 0)
        assert source["original_item_ids"].tolist() == [1]
        assert source[
            "base_entity_original_item_ids"
        ].tolist() == [1, 2, 3, 4, 5]
        for user_idx in range(1, 4):
            selected = users == user_idx
            assert raw[selected].tolist() == [0, 1, 2, 3, 4]
            assert windows[selected].tolist() == [
                -1,
                -1,
                -1,
                0,
                1,
            ]
        assert metadata["base_entity_items"] == 5
        assert metadata["num_prediction_items"] == 1
        assert metadata["context_entity_rows"] == 4
        assert metadata["context_hash_buckets"] == 4
        assert metadata["fitted_items"] == 5
        assert metadata["window_count"] == 2
        assert (
            metadata["unique_stream_only_original_items_accessed"]
            == 1
        )
        assert (
            metadata[
                "unique_fallback_context_entity_rows_accessed"
            ]
            == 1
        )
    plan, metadata = load_prepared_exposure_plan(
        config.output,
        max_seq_len=3,
    )
    assert plan.num_items == 5
    assert plan.num_prediction_items == 1
    assert plan.stream_dates == ["window_0", "window_1"]
    assert metadata["context_entity_rows"] == 4
    cohort_ids = json.loads(config.cohort_ids.read_text())
    assert cohort_ids["primary_benchmark_users"] == 2
    assert (
        cohort_ids["nested_benchmark_prefixes"]["2"][
            "prefix_length"
        ]
        == 2
    )


def test_stream_fallback_rejects_catalog_without_context_rows() -> None:
    config = MODULE.BuildConfig(
        prediction_catalog_size=2,
        cohort_sizes=(1,),
        primary_cohort_size=1,
    )
    with pytest.raises(
        ValueError,
        match="requires context entity rows",
    ):
        MODULE.map_entity_item_ids(
            np.array([9], dtype=np.int64),
            np.array([0, 1], dtype=np.int32),
            config,
            context_entity_rows=0,
        )
