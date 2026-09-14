from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import torch

from hstu_kvcache.data.yambda_history import load_yambda_histories
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.training import FoundationHistoryIndex


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d7_d14_full_reuse_v1.yaml"


def load_runner():
    path = ROOT / "scripts/run_yambda500m_medium_full_reuse_matrix.py"
    spec = importlib.util.spec_from_file_location("medium_full_reuse", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_foundation_evaluator():
    path = ROOT / "scripts/evaluate_yambda500m_foundation_raw.py"
    spec = importlib.util.spec_from_file_location("foundation_evaluator_medium_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_foundation_history_index_uses_identical_sorted_uid_groups() -> None:
    uids = np.asarray([7, 2, 7, 2, 9, 7], dtype=np.int64)
    timestamps = np.asarray([30, 20, 10, 10, 5, 20], dtype=np.int64)
    items = np.asarray([3, 4, 6, 8, 1, 5], dtype=np.int64)
    behaviors = np.asarray([1, 2, 3, 4, 1, 2], dtype=np.int64)
    index = FoundationHistoryIndex.from_columns(uids, timestamps, items, behaviors)
    assert list(index.rows) == [2, 7, 9]
    assert index.rows[2][0].tolist() == [10, 20]
    assert index.rows[2][1].tolist() == [8, 4]
    assert index.rows[7][0].tolist() == [10, 20, 30]
    assert index.rows[7][1].tolist() == [6, 5, 3]
    prefix = index.prefix(7, 30, max_history=2)
    assert prefix[0].tolist() == [6, 5]
    assert prefix[1].tolist() == [3, 2]
    assert prefix[2].tolist() == [10, 20]


def test_rejected_full_only_edge_keeps_descendant_reuse_locked(tmp_path: Path) -> None:
    module = load_runner()
    pipeline = module.Pipeline(CONTRACT, threads=1)
    pipeline.output = tmp_path
    pipeline.logs = tmp_path / "logs"
    pipeline.log_jsonl = pipeline.logs / "pipeline.jsonl"

    def write_report(edge: int, *, current_auc: float) -> None:
        directory = pipeline.full_only_dir("D7", edge, 7)
        directory.mkdir(parents=True)
        (directory / "adjudication.json").write_text(json.dumps({
            "parent_absolute": {"hstu_native": {
                "ROC_AUC": 0.70, "log_loss": 0.50, "Brier": 0.20,
            }},
            "candidates": {f"v{edge}": {
                "absolute": {"hstu_native": {
                    "ROC_AUC": current_auc, "log_loss": 0.49, "Brier": 0.19,
                }},
                "paired_release_gain": {"parent_minus_current_log_loss": {
                    "requests": 100,
                    "user_cluster_bootstrap_95CI": {"p2_5": 0.001, "p97_5": 0.02},
                }},
            }},
        }), encoding="utf-8")

    write_report(1, current_auc=0.69)
    write_report(2, current_auc=0.71)
    second = pipeline.seal_admission("D7", 2)
    first = json.loads(pipeline.admission_path("D7", 1).read_text(encoding="utf-8"))
    assert first["reuse_unlocked"] is False
    assert first["reason"] == "full_only_quality_gate_failed"
    assert second["reuse_unlocked"] is False
    assert second["reason"] == "parent_not_in_accepted_diagnostic_lineage"


def test_bounded_history_loader_keeps_last_prefix_and_window_events(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    pq.write_table(pa.table({
        "uid": pa.array([1, 1, 1, 1, 1], type=pa.uint64()),
        "timestamp": pa.array([10, 20, 30, 40, 50], type=pa.uint64()),
        "raw_item_id": pa.array([101, 102, 103, 104, 999], type=pa.uint64()),
        "behavior": pa.array([1, 1, 2, 2, 3], type=pa.uint8()),
        "is_organic": pa.array([0, 0, 0, 0, 0], type=pa.uint8()),
    }), shared / "listens.parquet")
    pq.write_table(pa.table({
        "raw_item_id": pa.array([101, 102, 103, 104], type=pa.uint64()),
        "item_idx": pa.array([1, 2, 3, 4], type=pa.uint64()),
    }), tmp_path / "item_mapping.parquet")
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps({
        "shared_listens_glob": "shared/listens.parquet",
        "item_mapping_path": "item_mapping.parquet",
    }), encoding="utf-8")
    history = load_yambda_histories(
        manifest, [1], known_vocab_size=4, oov_buckets=2,
        start_timestamp=35, end_timestamp=55, max_pre_events=2, threads=1,
    )
    timestamps, items, behaviors = history.rows[1]
    assert timestamps.tolist() == [20, 30, 40, 50]
    assert items[:3].tolist() == [2, 3, 4]
    assert items[3] in (4, 5)
    assert behaviors.tolist() == [1, 2, 2, 3]


def test_three_path_vectorized_evaluator_accepts_contract_context() -> None:
    module = load_foundation_evaluator()
    config = HSTUConfig(
        num_items=16, num_behaviors=4, hidden_size=8, num_layers=1,
        num_heads=1, max_seq_len=4, num_query_types=3, query_type_id=2,
        input_dropout=0.0,
    )
    torch.manual_seed(3)
    parent = HSTU(config).eval()
    current = HSTU(config).eval()
    current.load_state_dict(parent.state_dict())
    history = FoundationHistoryIndex.from_columns(
        uids=torch.tensor([1, 1, 1, 1, 1]).numpy(),
        timestamps=torch.tensor([1, 2, 3, 4, 6]).numpy(),
        item_ids=torch.tensor([1, 2, 3, 4, 5]).numpy(),
        behaviors=torch.tensor([1, 1, 2, 2, 3]).numpy(),
    )
    requests = [{"request_id": "q1", "uid": 1, "query_timestamp": 6, "item_idx": 7}]
    rows = module.evaluate_full_cache_cohort(
        uids=[1], by_user={1: requests}, history=history,
        parent=parent, current=current, parent_name="v0", current_name="v1",
        edge="v0_to_v1", checkpoint_hash="current", parent_hash="parent",
        manifest_hash="manifest", cutover=5, lineage_models=[("v0", parent)],
        event_end_exclusive=8, include_request_local=False,
        include_parent_exact=True, query_chunk_size=2, max_length=4,
    )
    assert {row["path"] for row in rows} == {
        "parent_exact_rolling", "current_exact_rolling", "one_hop_reuse_rolling"
    }
    assert {row["cache_length"] for row in rows} == {4}
    assert all(row["query_timestamp"] == 6 for row in rows)
