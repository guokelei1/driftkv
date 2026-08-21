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
