from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import torch
import yaml

from hstu_kvcache.data.yambda_history import load_yambda_histories
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.training import FoundationHistoryIndex


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d7_d14_full_reuse_v1.yaml"
EXECUTION = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d7_d14_execution_admission_v1.yaml"
CPU_RUNTIME = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_cpu_runtime_v2.yaml"
REUSE_RUNTIME = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_reuse_4gpu_runtime_v3.yaml"
FORCED_D7_REUSE = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d7_forced_reuse_diagnostic_v1.yaml"
D14_V5 = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_v5_extension_v1.yaml"
D14_V5_EXECUTION = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_v5_execution_v1.yaml"


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


def load_trainer():
    path = ROOT / "scripts/train_yambda500m_foundation_fsdp.py"
    spec = importlib.util.spec_from_file_location("medium_trainer_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_d14_v5_runner():
    path = ROOT / "scripts/run_yambda500m_medium_d14_v5_extension.py"
    spec = importlib.util.spec_from_file_location("medium_d14_v5_extension", path)
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


def test_medium_contract_freezes_complete_symmetric_matrix() -> None:
    value = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert value["scope"]["foundation_days_half_open"] == [0, 217]
    assert value["scope"]["complete_source_days_half_open"] == [0, 300]
    assert value["scope"]["branches"]["D7"]["updates"] == 10
    assert value["scope"]["branches"]["D7"]["evaluation_days"] == [3, 7]
    assert value["scope"]["branches"]["D14"]["updates"] == 4
    assert value["scope"]["branches"]["D14"]["evaluation_days"] == [3, 7, 14]
    assert value["evaluation"]["paths"] == [
        "parent_exact_rolling", "current_exact_rolling", "one_hop_reuse_rolling"
    ]
    assert value["scope"]["recursive_reuse"] == "prohibited"
    assert 217 + 10 * 7 + 7 <= 300
    assert 217 + 4 * 14 + 14 <= 300
    assert 217 + 5 * 14 + 14 > 300


def test_medium_runner_plan_has_checkpoint_first_32_cell_shape() -> None:
    module = load_runner()
    pipeline = module.Pipeline(CONTRACT, threads=1)
    plan = pipeline.plan()
    assert plan["formal_checkpoints"] == 15
    assert plan["formal_full_only_cells"] == 32
    assert plan["formal_reuse_cells_maximum"] == 32
    assert plan["world_size"] == 2
    assert plan["physical_gpus"] == [2, 3]
    assert plan["global_train_batch_size"] == 32
    assert plan["local_batch_sizes_by_rank"] == [16, 16]
    assert plan["D14_cpu_runtime"]["total_physical_history_workers"] == 28
    assert plan["tasks"].index("formal_all_32_full_only_cells") < plan["tasks"].index(
        "formal_reuse_only_for_unlocked_accepted_lineage_edges"
    )
    assert plan["formal_acknowledgement"] == "RUN_MEDIUM_D7_D14"


def test_medium_execution_supplement_freezes_gpu2_gpu3_only() -> None:
    value = yaml.safe_load(EXECUTION.read_text(encoding="utf-8"))
    amendment = value["execution_amendment"]
    assert amendment["world_size"] == 2
    assert amendment["physical_gpus"] == [2, 3]
    assert amendment["global_train_batch_size"] == 32
    assert amendment["local_batch_sizes_by_rank"] == [16, 16]
    assert "candidates" not in amendment
    assert value["release_admission"]["full_only_before_reuse"] == "required"


def test_d14_cpu_runtime_uses_disjoint_numa_local_physical_cores() -> None:
    module = load_runner()
    pipeline = module.Pipeline(CONTRACT, threads=1)
    value = yaml.safe_load(CPU_RUNTIME.read_text(encoding="utf-8"))
    runtime = value["runtime"]
    rank0 = set(runtime["rank0_cpu_affinity"])
    rank1 = set(runtime["rank1_cpu_affinity"])
    assert len(rank0) == len(rank1) == 14
    assert rank0.isdisjoint(rank1)
    assert len(rank0 | rank1) == runtime["total_physical_history_workers"] == 28
    assert pipeline.cpu_runtime_args("D7") == []
    args = pipeline.cpu_runtime_args("D14")
    assert args[args.index("--history-threads") + 1] == "14"
    assert args[args.index("--cpu-affinity-by-rank") + 1] == (
        "28,29,30,31,32,33,34,35,36,37,38,39,40,41;"
        "42,43,44,45,46,47,48,49,50,51,52,53,54,55"
    )


def test_remaining_d14_reuse_runtime_uses_four_gpus_and_larger_batches() -> None:
    module = load_runner()
    pipeline = module.Pipeline(CONTRACT, threads=1)
    value = yaml.safe_load(REUSE_RUNTIME.read_text(encoding="utf-8"))
    runtime = value["runtime"]
    assert value["scope"]["physical_gpus"] == [0, 1, 2, 3]
    assert value["scope"]["world_size"] == pipeline.reuse_world == 4
    assert runtime["cohort_size_per_rank"] == 32
    assert runtime["query_chunk_size_per_rank"] == 256
    cpu_sets = [set(runtime[f"rank{rank}_cpu_affinity"]) for rank in range(4)]
    assert all(len(values) == 14 for values in cpu_sets)
    assert len(set().union(*cpu_sets)) == 56
    assert sum(len(left & right) for i, left in enumerate(cpu_sets) for right in cpu_sets[i + 1:]) == 0
    assert pipeline.reuse_distributed_prefix[-1] == "--nproc_per_node=4"
    assert pipeline.reuse_gpu_env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_forced_d7_reuse_is_complete_four_gpu_diagnostic_only() -> None:
    module = load_runner()
    pipeline = module.Pipeline(CONTRACT, threads=1)
    value = yaml.safe_load(FORCED_D7_REUSE.read_text(encoding="utf-8"))
    assert value["scope"]["branch"] == "D7"
    assert value["scope"]["horizons_days"] == [3, 7]
    assert len(value["scope"]["edges"]) == 10
    assert value["scope"]["expected_cells"] == 20
    assert value["scope"]["formal_summary_mutation"] == "prohibited"
    assert value["scope"]["formal_admission_override"] == "prohibited"
    assert value["runtime"]["physical_gpus"] == [0, 1, 2, 3]
    assert value["runtime"]["world_size"] == 4
    assert value["runtime"]["cohort_size_per_rank"] == 32
    assert value["runtime"]["query_chunk_size_per_rank"] == 256
    assert pipeline.reuse_dir("D7", 1, 3, forced_diagnostic=True) == (
        pipeline.output / "D7" / "forced_reuse_diagnostic_v1" / "E3" / "v0_to_v1"
    )
    assert pipeline.reuse_dir("D7", 1, 3) == (
        pipeline.output / "D7" / "reuse" / "E3" / "v0_to_v1"
    )


def test_d14_v5_extension_separates_complete_and_partial_windows() -> None:
    launch = yaml.safe_load(D14_V5.read_text(encoding="utf-8"))
    execution = yaml.safe_load(D14_V5_EXECUTION.read_text(encoding="utf-8"))
    assert launch["scope"]["training_day_range_half_open"] == [273, 287]
    assert launch["scope"]["complete_evaluation_windows"] == {
        "E3": [287, 290], "E7": [287, 294],
    }
    assert launch["scope"]["partial_tail_diagnostic_window"]["requested_range_half_open"] == [287, 301]
    assert launch["scope"]["partial_tail_diagnostic_window"]["interpretation_as_complete_E14"] == "prohibited"
    assert execution["execution_amendment"]["physical_gpus"] == [0, 1, 2, 3]
    assert execution["execution_amendment"]["global_train_batch_size"] == 32
    assert execution["execution_amendment"]["local_batch_sizes_by_rank"] == [8, 8, 8, 8]
    assert execution["full_only_runtime"]["batch_size_per_rank"] == 128
    assert execution["reuse_runtime"]["cohort_size_per_rank"] == 32


def test_d14_v5_runner_uses_separate_extension_paths() -> None:
    module = load_d14_v5_runner()
    runner = module.Runner(D14_V5, D14_V5_EXECUTION, threads=1)
    assert runner.window("E3") == (287, 290, False)
    assert runner.window("E7") == (287, 294, False)
    assert runner.window("E14_partial") == (287, 301, True)
    assert runner.checkpoint_dir == (
        ROOT / "results/yambda500m_medium_seed17/full_reuse_matrix_v1/D14/v5_extension_v1/checkpoint"
    )
    assert "v5_extension_v1" in runner.full_dir("E3", canary=False).parts


def test_global_batch_partition_is_16_16_for_two_ranks() -> None:
    module = load_trainer()
    assert [module.local_batch_size(32, 2, rank) for rank in range(2)] == [16, 16]


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


def test_medium_parameter_count_matches_resource_contract() -> None:
    value = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    model = dict(value["model"])
    known = model.pop("known_items_from_dataset_manifest")
    oov = model.pop("oov_buckets")
    with torch.device("meta"):
        hstu = HSTU(HSTUConfig(num_items=known + oov, **model))
    assert sum(parameter.numel() for parameter in hstu.parameters()) == value["resource_plan"]["model_parameters"]


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
