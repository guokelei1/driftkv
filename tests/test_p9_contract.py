from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from hstu_kvcache.models import HSTUKVCache

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p9_contract_freezes_p8_inputs_and_gpu_scope() -> None:
    contract_path = ROOT / "configs/contracts/p9_tomography_contract_v1.yaml"
    contract = yaml.safe_load(contract_path.read_text())
    assert contract["scope"]["workload"] == "F"
    assert contract["scope"]["models"] == ["m0_f", "m1"]
    assert contract["scope"]["seeds"] == [17, 37, 71]
    assert contract["scope"]["gpu_allowlist"] == [0, 1]
    assert contract["authorization"]["controller"] is False
    assert contract["authorization"]["base_or_release_recipe_tuning"] is False
    sources = {
        "p8_release_contract": "configs/contracts/f_release_chain_contract_v1.yaml",
        "r0_raw_seal": "results/p8/r0_control/raw_score_seal_v1.json",
        "r0_adjudication": "results/p8/r0_control/adjudication_v1.json",
        "r1_edge1_raw_seal": "results/p8/r1_edge1/raw_score_seal_v1.json",
        "r1_edge1_adjudication": "results/p8/r1_edge1/hs_adjudication_v1.json",
        "r1_edge2_raw_seal": "results/p8/r1_edge2/raw_score_seal_v1.json",
        "r1_edge2_adjudication": "results/p8/r1_edge2/hs_adjudication_v1.json",
        "r2_raw_seal": "results/p8/r2/raw_score_seal_v1.json",
        "r2_adjudication": "results/p8/r2/hs_adjudication_v1.json",
    }
    for name, relative in sources.items():
        assert sha256(ROOT / relative) == contract["input_hashes"][name]


def test_p9_diagnostic_splice_is_not_declared_deployable() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p9_tomography_contract_v1.yaml").read_text())
    coarse = contract["coarse_tomography"]
    assert coarse["status"] == "diagnostic_not_executable"
    assert "diagnostic_exact_KV_splice_as_deployable_action" in coarse["prohibited_interpretation"]
    assert contract["authorization"]["p9_3_representative_2d_tomography"] == "pending_p9_2_review"


def test_p9_diagnostic_cache_replaces_only_frozen_region() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_p9_tomography_raw as tomo
    import torch

    old = HSTUKVCache(k=torch.zeros(4, 1, 8, 2), v=torch.zeros(4, 1, 8, 2), seq_len=8)
    exact = HSTUKVCache(k=torch.ones(4, 1, 8, 2), v=torch.ones(4, 1, 8, 2), seq_len=8)
    layer = tomo.diagnostic_cache(old, exact, "layer_2")
    assert torch.equal(layer.k[2], exact.k[2])
    assert torch.equal(layer.k[1], old.k[1])
    recent = tomo.diagnostic_cache(old, exact, "recent_1")
    assert torch.equal(recent.k[:, :, -1], exact.k[:, :, -1])
    assert torch.equal(recent.k[:, :, :-1], old.k[:, :, :-1])
    assert tomo.segment_slice("middle", 8) == slice(2, 6)


def test_p9_canary_selection_is_label_free_and_deterministic() -> None:
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_p9_tomography_raw as tomo

    requests = [
        SimpleNamespace(request_id=f"r{index}", history_timestamps=[1, 2, 3])
        for index in range(10)
    ]
    first = tomo.selected_requests(requests, 3, 4)
    second = tomo.selected_requests(list(reversed(requests)), 3, 4)
    assert [row.request_id for row in first] == [row.request_id for row in second]


def test_p9_job_ledger_covers_all_frozen_cells() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_p9_tomography as runner

    jobs = runner.jobs()
    assert len(jobs) == 24
    assert jobs[0].release == "r0" and jobs[0].model == "m0_f" and jobs[0].seed == 17
    assert "CUDA_VISIBLE_DEVICES=0" in jobs[0].command(0)
    assert "cuda:0" in jobs[0].command(0)
    assert "CUDA_VISIBLE_DEVICES=2" not in jobs[0].command(0)


def test_p9_queue_partitions_without_seed_or_result_selection() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_p9_tomography as runner

    jobs = runner.jobs()
    even_after_first = [job for index, job in enumerate(jobs) if index % 2 == 0 and index > 0]
    odd_after_first = [job for index, job in enumerate(jobs) if index % 2 == 1 and index > 1]
    assert len(even_after_first) == 11 and len(odd_after_first) == 11
    assert {job.seed for job in even_after_first + odd_after_first} == {17, 37, 71}


def test_p9_tomography_empty_above_floor_population_is_explicit() -> None:
    import sys
    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))
    import adjudicate_p9_tomography as adjudicate

    assert adjudicate.summarize(np.asarray([], dtype=np.float64)) == {
        "users": 0, "mean_equal_user": None, "P50_equal_user": None,
        "P90_equal_user": None, "P95_equal_user": None,
    }


def test_p9_raw_seal_action_order_matches_frozen_contract() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import seal_p9_tomography as seal

    assert seal.EXPECTED_ACTIONS == [
        "layer_0", "layer_1", "layer_2", "layer_3", "oldest_half",
        "middle", "recent_128", "recent_32", "recent_8", "recent_1",
    ]


def test_p9_closure_contract_seals_all_inputs_and_prohibits_selection() -> None:
    contract_path = ROOT / "configs/contracts/p9_2_closure_contract_v1.yaml"
    contract = yaml.safe_load(contract_path.read_text())
    paths = {
        "p9_contract": "configs/contracts/p9_tomography_contract_v1.yaml",
        "p9_1_distribution_v2": "results/p9/p9_1_hs_distribution_v2.json",
        "p9_2_raw_seal": "results/p9/p9_2_tomography_raw_seal_v1.json",
        "p9_2_coarse_result": "results/p9/p9_2_coarse_tomography_v1.json",
        "p8_r0_raw_seal": "results/p8/r0_control/raw_score_seal_v1.json",
        "p8_r1_edge1_raw_seal": "results/p8/r1_edge1/raw_score_seal_v1.json",
        "p8_r1_edge2_raw_seal": "results/p8/r1_edge2/raw_score_seal_v1.json",
        "p8_r2_raw_seal": "results/p8/r2/raw_score_seal_v1.json",
    }
    for name, relative in paths.items():
        assert sha256(ROOT / relative) == contract["input_hashes"][name]
    assert contract["scope"]["all_24_cells_and_all_actions_required"] is True
    assert contract["quality_join"]["join_tolerance"] == 0.0
    assert contract["authorization"]["choose_actions_or_cells_using_quality"] is False
    assert contract["authorization"]["controller"] is False


def test_p9_quality_metric_directions_and_equal_user_weighting() -> None:
    import sys
    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))
    import adjudicate_p9_quality_companions as quality

    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    logits = np.asarray([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
    uids = np.asarray([10, 10, 20, 30], dtype=np.int64)
    metrics = quality.binary_metrics(labels, logits, uids)
    assert set(metrics) == set(quality.METRICS)
    assert all(np.isfinite(value) for value in metrics.values())
    assert quality.quality_gain(0.5, 0.4, "aggregate_logloss") > 0
    assert quality.quality_gain(0.5, 0.6, "ROC_AUC") > 0
    weights = quality.equal_user_weights(uids)
    assert np.isclose(weights[uids == 10].sum(), weights[uids == 20].sum())


def test_p9_risk_concentration_handles_identity_and_ties() -> None:
    import sys
    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_p9_risk_concentration as risk

    assert risk.gini(np.zeros(4)) is None
    assert np.isclose(risk.gini(np.ones(4)), 0.0)
    selected = risk.top_indices({4: 1.0, 2: 1.0, 9: 0.5, 1: 0.0}, 0.5)
    assert selected == [2, 4]


def test_p9_3_contract_uses_semantic_cells_and_sealed_closure() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p9_3_2d_tomography_contract_v1.yaml").read_text())
    assert [(row["release"], row["model"]) for row in contract["scope"]["semantic_cells"]] == [
        ("r0", "m0_f"), ("r1_edge1", "m0_f"), ("r2", "m0_f"), ("r2", "m1"),
    ]
    assert contract["scope"]["cell_selection_uses_release_semantics_not_P9_scores"] is True
    assert contract["scope"]["total_seed_cells"] == 12
    assert contract["actions"]["actions_per_cell"] == 24
    assert contract["authorization"]["select_executable_action_from_2d_splice"] is False
    paths = {
        "p8_evidence_seal": "results/p9/p8_evidence_seal_v1.json",
        "p9_2_raw_seal": "results/p9/p9_2_tomography_raw_seal_v1.json",
        "p9_2_quality_companions": "results/p9/p9_2_quality_companions_v1.json",
        "p9_1_risk_concentration": "results/p9/p9_1_risk_concentration_v1.json",
        "p9_2_closure_result": "configs/contracts/p9_2_closure_result_v1.yaml",
    }
    for name, relative in paths.items():
        assert sha256(ROOT / relative) == contract["input_hashes"][name]


def test_p9_3_2d_splice_replaces_only_one_layer_segment() -> None:
    import sys
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_p9_2d_tomography_raw as two_d

    old = HSTUKVCache(k=torch.zeros(4, 1, 8, 2), v=torch.zeros(4, 1, 8, 2), seq_len=8)
    exact = HSTUKVCache(k=torch.ones(4, 1, 8, 2), v=torch.ones(4, 1, 8, 2), seq_len=8)
    mixed = two_d.diagnostic_cache_2d(old, exact, "layer_2__middle")
    assert torch.equal(mixed.k[2, :, 2:6], exact.k[2, :, 2:6])
    assert torch.equal(mixed.k[2, :, :2], old.k[2, :, :2])
    assert torch.equal(mixed.k[2, :, 6:], old.k[2, :, 6:])
    assert torch.equal(mixed.k[1], old.k[1])
    assert len(two_d.action_names_2d(4)) == 24


def test_p9_3_ledger_covers_all_semantic_seed_cells_on_gpu01() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_p9_2d_tomography as ledger

    jobs = ledger.jobs()
    assert len(jobs) == 12
    assert {(job.release, job.model) for job in jobs} == {
        ("r0", "m0_f"), ("r1_edge1", "m0_f"), ("r2", "m0_f"), ("r2", "m1"),
    }
    assert {job.seed for job in jobs} == {17, 37, 71}
    assert "CUDA_VISIBLE_DEVICES=0" in jobs[0].command(0)
    assert "CUDA_VISIBLE_DEVICES=2" not in jobs[0].command(0)


def test_p9_3_seal_requires_frozen_layer_major_action_order() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import seal_p9_2d_tomography as seal

    assert len(seal.EXPECTED_ACTIONS) == 24
    assert seal.EXPECTED_ACTIONS[:6] == [
        "layer_0__oldest_half", "layer_0__middle", "layer_0__recent_128",
        "layer_0__recent_32", "layer_0__recent_8", "layer_0__recent_1",
    ]
    assert seal.EXPECTED_ACTIONS[-1] == "layer_3__recent_1"
