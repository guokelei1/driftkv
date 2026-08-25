from pathlib import Path
import json

import pyarrow.parquet as pq
import yaml

from hstu_kvcache.data.scale_population import (
    UID_SELECTOR_NAMESPACE,
    select_medium_uids,
    uid_selector_digest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_uid_selector_is_order_independent_and_nested() -> None:
    eligible = [91, 3, 42, 5, 100]
    selected = select_medium_uids(eligible, count=3)
    assert selected == select_medium_uids(reversed(eligible), count=3)
    assert set(selected).issubset(eligible)
    expected = sorted(eligible, key=lambda uid: (uid_selector_digest(uid), uid))[:3]
    assert selected == expected


def test_scale_population_contract_is_label_free_and_preserves_small() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/contracts/yambda500m_scale_population_v1.yaml").read_text()
    )
    assert contract["scope"]["small"] == "frozen_Yambda_50M_reference_not_reselected"
    assert contract["scope"]["training_authorized"] is False
    assert contract["scope"]["theta3_access_authorized"] is False
    assert contract["medium_selector"]["namespace"] == UID_SELECTOR_NAMESPACE
    assert contract["medium_selector"]["target_users"] == 30_000
    assert "like_or_dislike_label" in contract["population_eligibility"]["do_not_filter_on"]
    assert contract["population_eligibility"]["preserve_complete_history_for_selected_uid"] is True


def test_materialized_populations_are_nested_when_present() -> None:
    root = ROOT / "data/manifests/yambda500m_scale_v1"
    medium_path = root / "medium_users.parquet"
    large_path = root / "large_users.parquet"
    if not medium_path.exists() or not large_path.exists():
        return
    medium = set(pq.read_table(medium_path, columns=["uid"])["uid"].to_pylist())
    large = set(pq.read_table(large_path, columns=["uid"])["uid"].to_pylist())
    assert len(medium) == 30_000
    assert medium.issubset(large)


def test_materialized_mapping_and_manifest_contract_when_present() -> None:
    root = ROOT / "data/manifests/yambda500m_scale_v1"
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    assert manifest["training_authorized"] is False
    assert manifest["theta3_access_authorized"] is False
    assert manifest["selector"]["medium_users"] == 30_000
    for scale in ("medium", "large"):
        path = root / f"{scale}_item_mapping.parquet"
        parquet = pq.ParquetFile(path)
        first = parquet.read_row_group(0, columns=["item_idx", "raw_item_id"])
        last = parquet.read_row_group(
            parquet.metadata.num_row_groups - 1, columns=["item_idx", "raw_item_id"]
        )
        assert first["item_idx"][0].as_py() == 1
        assert last["item_idx"][-1].as_py() == parquet.metadata.num_rows
        assert first["raw_item_id"][0].as_py() <= last["raw_item_id"][-1].as_py()
