from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_evokv_design3_m1_qk_data.py"
SPEC = importlib.util.spec_from_file_location(
    "build_evokv_design3_m1_qk_data",
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
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, "Tenrec/QK-video.csv")


def test_qk_builder_uses_filtered_positions_nested_hash_and_phase_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "Tenrec.zip"
    write_qk_archive(source)
    config = MODULE.BuildConfig(
        source=source,
        cache_dir=tmp_path / "cache",
        output=tmp_path / "processed.npz",
        manifest=tmp_path / "manifest.json",
        cohort_ids=tmp_path / "cohorts.json",
        catalog_size=2,
        base_prefix=2,
        history_length=3,
        slide=1,
        fit_calibration_users=1,
        cohort_sizes=(1, 2),
        chunk_size=3,
    )

    audit = MODULE.run(config, audit_only=True)

    assert audit["status"] == "audit_only"
    assert audit["cohorts"]["eligible_users"] == 4
    assert not config.output.exists()
    catalog, _ = MODULE.load_npz(config.catalog_cache)
    assert catalog["original_item_ids"].tolist() == [1, 2]
    cohort, _ = MODULE.load_npz(config.cohort_cache)
    expected = MODULE.stable_user_order(np.array([10, 20, 30, 40]), config.hash_salt)
    assert cohort["fit_calibration_user_ids"].tolist() == expected[:1].tolist()
    assert cohort["benchmark_user_ids"].tolist() == expected[1:3].tolist()
    assert set(cohort["fit_calibration_user_ids"]).isdisjoint(
        set(cohort["benchmark_user_ids"])
    )

    monkeypatch.setattr(
        MODULE,
        "build_catalog_cache",
        lambda *_: (_ for _ in ()).throw(AssertionError("catalog cache was not reused")),
    )
    monkeypatch.setattr(
        MODULE,
        "build_cohort_cache",
        lambda *_: (_ for _ in ()).throw(AssertionError("cohort cache was not reused")),
    )
    result = MODULE.run(config)

    assert result["status"] == "materialized"
    with np.load(config.output, allow_pickle=False) as source_npz:
        metadata = json.loads(str(source_npz["metadata_json"].item()))
        users = source_npz["user_idx"]
        filtered = source_npz["filtered_position"]
        raw = source_npz["raw_ordinal"]
        windows = source_npz["window_index"]
        assert source_npz["original_item_ids"].tolist() == [1, 2]
        assert len(users) == 12
        for user_idx in range(1, 4):
            selected = users == user_idx
            assert filtered[selected].tolist() == [0, 1, 2, 3]
            assert np.all(np.diff(raw[selected]) > 0)
            assert windows[selected].tolist() == [-1, -1, -1, 0]
        assert metadata["old_window_filtered_positions"] == [0, 3]
        assert metadata["target_window_filtered_positions"] == [1, 4]
    cohort_ids = json.loads(config.cohort_ids.read_text())
    assert cohort_ids["benchmark_user_ids"] == expected[1:3].tolist()
    assert cohort_ids["nested_benchmark_prefixes"]["1"]["prefix_length"] == 1
