from pathlib import Path
import json

import pyarrow.parquet as pq
import yaml

from hstu_kvcache.data import YambdaScaleDataset


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_unified_scales_v1.yaml"
PROCESSED = ROOT / "data/processed/yambda500m_unified_v1"


def test_unified_contract_freezes_nested_scales_and_one_builder() -> None:
    value = yaml.safe_load(CONTRACT.read_text())
    scales = value["population"]["scales"]
    assert scales["small"]["rank_limit"] == 10_000
    assert scales["medium"]["rank_limit"] == 30_000
    assert scales["large"]["expected_users"] == 79_681
    assert value["population"]["nesting"] == "small_subset_medium_subset_large"
    assert value["regeneration"]["builds_all_scales_together"] is True
    assert value["physical_store"]["duplicate_full_event_tables_per_scale"] is False
    assert value["authorization"]["training"] is False


def test_materialized_scale_manifests_and_users_when_present() -> None:
    if not (PROCESSED / "manifest.json").exists():
        return
    expected = {"small": 10_000, "medium": 30_000, "large": 79_681}
    uid_sets = {}
    for scale, count in expected.items():
        root = PROCESSED / "scales" / scale
        dataset = json.loads((root / "dataset.json").read_text())
        table = pq.read_table(root / "users.parquet", columns=["uid", "selector_rank"])
        assert table.num_rows == count == dataset["users"]
        assert max(table["selector_rank"].to_pylist()) <= dataset["rank_limit"]
        uid_sets[scale] = set(table["uid"].to_pylist())
    assert uid_sets["small"].issubset(uid_sets["medium"])
    assert uid_sets["medium"].issubset(uid_sets["large"])


def test_loader_enforces_feedback_purpose_lock_when_present() -> None:
    path = PROCESSED / "scales/small/dataset.json"
    if not path.exists():
        return
    dataset = YambdaScaleDataset(path, threads=1)
    # The guard triggers before any Parquet scan.
    try:
        next(dataset.iter_feedback(21_168_000, 21_772_801, purpose="evaluation"))
    except PermissionError:
        pass
    else:
        raise AssertionError("future feedback passed the default lock")


def test_loader_reads_small_canary_windows_when_present() -> None:
    path = PROCESSED / "scales/small/dataset.json"
    if not path.exists():
        return
    dataset = YambdaScaleDataset(path, threads=1)
    listens = next(dataset.iter_listens(18_748_800, 18_749_000, batch_size=1024))
    feedback = next(
        dataset.iter_feedback(
            18_748_800,
            18_835_200,
            purpose="train",
            batch_size=1024,
        )
    )
    assert listens.num_rows > 0
    assert feedback.num_rows > 0
    assert max(listens.column("selector_rank").to_pylist()) <= 10_000
    assert all(feedback.column("target_known").to_pylist())
