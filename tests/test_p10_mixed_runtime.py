from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml

from hstu_kvcache.models import HSTUKVCache


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_p10_mixed_policy_runtime as runtime


def test_p10_mixed_runtime_contract_uses_only_gpu01_and_sealed_policy() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p10_2_mixed_policy_runtime_contract_v1.yaml").read_text())
    assert contract["scope"]["GPU_allowlist"] == [0, 1]
    assert contract["execution"]["sampled_state_operations"] == "all_five_charged_probe_actions_with_Exact_result_retained"
    assert contract["evidence_boundary"]["quality_labels_read"] is False
    assert contract["adjudication"]["controller_authorized_by_this_step"] is False


def test_p10_cache_subset_preserves_layer_sequence_layout() -> None:
    cache = HSTUKVCache(
        k=torch.arange(4 * 3 * 5 * 2).reshape(4, 3, 5, 2),
        v=torch.arange(4 * 3 * 5 * 2).reshape(4, 3, 5, 2) + 100,
        seq_len=5,
    )
    subset = runtime.subset_cache(cache, torch.tensor([2, 0]))
    assert subset.k.shape == (4, 2, 5, 2)
    assert torch.equal(subset.k[:, 0], cache.k[:, 2])
    assert subset.seq_len == 5


def test_p10_runtime_ledger_matches_frozen_semantic_scope() -> None:
    import run_p10_mixed_policy_runtime_queue as queue

    jobs = queue.jobs()
    assert len(jobs) == 10
    assert {(job.release, job.model) for job in jobs} == {
        ("r2", "m1"), ("r2", "m0_f"), ("r1_edge2", "m1")
    }
    assert {job.sample for job in jobs} == {0.01, 0.02}
    assert {job.budget for job in jobs if job.release == "r2" and job.model == "m1"} == {0.05, 0.10, 0.25}


def test_p10_runtime_exact_baseline_mapping_is_release_semantic() -> None:
    import json
    import adjudicate_p10_mixed_policy_runtime as adjudicate

    runtime = json.loads((ROOT / "results/p9/p9_10_full_population_runtime_v1.json").read_text())
    assert adjudicate.exact_baseline(runtime, "r1_edge2", "m1") > 0
    assert adjudicate.exact_baseline(runtime, "r2", "m0_f") > 0
    assert adjudicate.exact_baseline(runtime, "r2", "m1") > 0


def test_grouped_batch_plan_preserves_non_noop_operation_signatures() -> None:
    from types import SimpleNamespace

    states = [
        {"uid": 1, "effective_prefix_length": 32},
        {"uid": 2, "effective_prefix_length": 32},
        {"uid": 3, "effective_prefix_length": 64},
    ]
    assignments = {
        1: SimpleNamespace(calibration_sample=True, action="exact_all"),
        2: SimpleNamespace(calibration_sample=False, action="hybrid_tail128"),
        3: SimpleNamespace(calibration_sample=False, action="noop"),
    }
    grouped = runtime.build_groups(states, assignments, "grouped")
    assert set(grouped) == {(32, "probe"), (32, "hybrid_tail128")}
    assert sum(len(rows) for rows in grouped.values()) == 2


def test_grouped_runtime_queue_reuses_exact_frozen_ledger() -> None:
    import run_p10_grouped_runtime_queue as grouped
    import run_p10_mixed_policy_runtime_queue as reference

    ledger = reference.jobs()
    assert len(ledger) == 10
    assert len({job.name for job in ledger}) == 10
    assert grouped.OUTPUT.name == "grouped"
