from __future__ import annotations

import numpy as np
import pytest
import torch

from hstu_kvcache.migration.stage45_oldkv import DirectOldKVProgram
from hstu_kvcache.migration.xp_d1_quality import (
    METHODS,
    REUSE_EXACT_METHODS,
    apply_direct_oldkv,
    build_action_plan_v2,
    cache_storage_roundtrip,
    evaluate_quality_batch,
    merge_batch_reports,
    split_identity,
)
from hstu_kvcache.migration.xp_exact_baseline import XPBaselineRecord
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.streaming.sharded_edge import (
    ExternalEmbeddingHSTU,
    fixed_candidate_ids,
)
from hstu_kvcache.streaming.trainer import build_next_item_targets
from hstu_kvcache.streaming.xp_projected_edge import (
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
)


def test_direct_oldkv_application_matches_explicit_affine() -> None:
    generator = torch.Generator().manual_seed(7)
    source = HSTUKVCache(
        k=torch.randn(2, 3, 4, 2, generator=generator),
        v=torch.randn(2, 3, 4, 2, generator=generator),
        seq_len=4,
    )
    program = DirectOldKVProgram(
        source_version="theta0",
        target_version="theta1",
        weights=torch.randn(2, 4, 4, generator=generator).contiguous(),
        biases=torch.randn(2, 4, generator=generator).contiguous(),
    )

    observed = apply_direct_oldkv(program, source)
    joined = torch.cat((source.k, source.v), dim=-1)
    expected = torch.stack(
        [
            joined[layer] @ program.weights[layer]
            + program.biases[layer]
            for layer in range(2)
        ]
    )

    torch.testing.assert_close(observed.k, expected[..., :2])
    torch.testing.assert_close(observed.v, expected[..., 2:])


def _record(record_id: int, retained_tokens: int) -> XPBaselineRecord:
    target_tokens = retained_tokens + 32
    return XPBaselineRecord(
        record_id=record_id,
        user_id=10_000 + record_id,
        owner_rank=0,
        item_ids=np.arange(1, target_tokens + 1, dtype=np.int64),
        old_start=0,
        old_length=retained_tokens,
        target_start=0,
        target_length=target_tokens,
        old_valid_bytes=retained_tokens * 16,
        target_valid_bytes=target_tokens * 16,
    )


def test_action_plan_uses_retained_token_budget_and_het_extents() -> None:
    records = tuple(
        _record(index, 10 if index % 2 == 0 else 90)
        for index in range(20)
    )

    plan = build_action_plan_v2(
        records,
        benchmark_id="tiny-xp",
        source_version=0,
        target_version=1,
        program_sha256="program",
        source_checkpoint_sha256="source",
        target_checkpoint_sha256="target",
        workload_sha256="workload",
        split_sha256="split",
        selection_salt="unit-test",
    )

    selection = plan["selection"]
    exact_tokens = sum(
        value["retained_tokens"]
        for value in plan["records"]
        if value["action"] == "exact"
    )
    assert selection["budget_basis"] == "retained_tokens"
    assert selection["quality_labels_read"] is False
    assert selection["actual_exact_retained_tokens"] == exact_tokens
    assert selection["actual_retained_token_fraction"] == pytest.approx(
        exact_tokens / 1_000
    )
    assert abs(selection["actual_retained_token_fraction"] - 0.2) <= 0.1
    assert {value["action"] for value in plan["records"]} == {
        "compiled",
        "exact",
    }
    assert all(value["append_tokens"] == 32 for value in plan["records"])
    assert plan["extent_contract"]["layout"] == "natural_variable_length_het"


def test_qualification_only_split_records_empty_fit_and_probe() -> None:
    qualification = (
        {"record_indices": torch.tensor([7, 3, -1], dtype=torch.int64)},
    )

    identity = split_identity((), (), qualification)

    assert identity["roles"]["fit"]["records"] == 0
    assert identity["roles"]["probe"]["records"] == 0
    assert identity["roles"]["qualification_test"]["records"] == 2


def test_cpu_quality_canary_emits_reuse_compiled_mixed_and_exact() -> None:
    torch.manual_seed(11)
    spec = XPProjectedModelSpec(
        num_embeddings=17,
        embedding_width=8,
        hidden_size=4,
        num_prediction_items=8,
        num_behaviors=3,
        num_layers=1,
        num_heads=2,
        head_dim=2,
        max_seq_len=8,
    )
    dense = ExternalEmbeddingHSTU(spec.hstu_config()).eval()
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=torch.randn(17, 8),
        projection_weight=torch.randn(4, 8),
        num_embeddings=17,
        rank=0,
        world_size=1,
    ).eval()
    batch = {
        "item_ids": torch.tensor(
            [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]],
            dtype=torch.int64,
        ),
        "behaviors": torch.ones(2, 6, dtype=torch.int64),
        "time_deltas": torch.ones(2, 6),
        "labels": torch.ones(2, 6, dtype=torch.int64),
        "train_mask": torch.ones(2, 6, dtype=torch.bool),
        "lengths": torch.full((2,), 6, dtype=torch.int64),
        "record_indices": torch.tensor([10, 11], dtype=torch.int64),
    }
    prefix_lengths = torch.full((2,), 2, dtype=torch.int64)
    exact = dense.core.compute_kv_from_item_embeddings(
        embedding(batch["item_ids"][:, :2], prefix_lengths),
        batch["behaviors"][:, :2],
        batch["time_deltas"][:, :2],
        prefix_lengths,
    )
    old = HSTUKVCache(
        k=exact.k + 0.25,
        v=exact.v - 0.25,
        seq_len=exact.seq_len,
    )
    old_fp16 = cache_storage_roundtrip(old, torch.float16)
    exact_fp16 = cache_storage_roundtrip(exact, torch.float16)
    assert exact_fp16.k.dtype == torch.float16
    assert exact_fp16.v.dtype == torch.float16
    torch.testing.assert_close(exact_fp16.k.float(), exact.k.half().float())
    assert bool(torch.any(exact_fp16.k.float() != exact.k))
    program = DirectOldKVProgram(
        source_version="theta0",
        target_version="theta1",
        weights=torch.eye(8, dtype=torch.float16).unsqueeze(0).contiguous(),
        biases=torch.zeros(1, 8, dtype=torch.float16),
    )
    targets, valid = build_next_item_targets(
        batch["item_ids"],
        batch["lengths"],
        batch["labels"],
        batch["train_mask"],
    )
    positives = targets[valid]
    negatives = (positives.remainder(7) + 1).unsqueeze(1)
    candidates = torch.cat((positives.unsqueeze(1), negatives), dim=1)

    report = evaluate_quality_batch(
        dense,
        embedding,
        batch,
        candidates,
        old_fp16,
        program,
        history_end=3,
        device=torch.device("cpu"),
        timing_repeats=1,
        mixed_exact_record_ids=frozenset({10}),
        common_cache_storage_dtype=torch.float16,
    )

    assert tuple(report["methods"]) == METHODS
    assert report["methods"]["all_exact"]["cache_error_rel"] == [0.0, 0.0]
    mixed_errors = report["methods"]["mixed_fixed20"]["cache_error_rel"]
    assert mixed_errors[0] == 0.0
    assert mixed_errors[1] > 0.0
    assert report["methods"]["mixed_fixed20"][
        "maintenance_cost_kind"
    ].endswith("not_end_to_end")
    assert all(
        value["recommendation_sums"]["positive_targets"] == 6
        for value in report["methods"].values()
    )
    assert "paired_target_contributions" not in report
    assert report["common_cache_storage"]["storage_dtype"] == "torch.float16"
    assert report["common_cache_storage"]["consumption_dtype"] == "torch.float32"
    assert all(
        "recommendation_sums_by_suffix_offset" not in value
        for value in report["methods"].values()
    )

    diagnostic_candidates = fixed_candidate_ids(
        positives,
        spec.num_prediction_items,
        999,
        37,
    )
    diagnostic = evaluate_quality_batch(
        dense,
        embedding,
        batch,
        diagnostic_candidates,
        old_fp16,
        None,
        history_end=3,
        device=torch.device("cpu"),
        timing_repeats=1,
        mixed_exact_record_ids=frozenset(),
        methods=REUSE_EXACT_METHODS,
        suffix_offset_breakdown=True,
        common_cache_storage_dtype=torch.float16,
    )

    assert diagnostic_candidates.shape == (10, 1_000)
    assert tuple(diagnostic["methods"]) == REUSE_EXACT_METHODS
    assert diagnostic["common_cache_storage"] == {
        "storage_dtype": "torch.float16",
        "consumption_dtype": "torch.float32",
        "methods": list(REUSE_EXACT_METHODS),
    }
    assert tuple(
        diagnostic["methods"]["all_reuse"][
            "recommendation_sums_by_suffix_offset"
        ]
    ) == ("1", "2", "3")
    merged = merge_batch_reports([diagnostic])
    for method in REUSE_EXACT_METHODS:
        offsets = merged["methods"][method][
            "recommendation_by_suffix_offset"
        ]
        assert sum(
            int(value["positive_targets"]) for value in offsets.values()
        ) == 6
        assert all(value["positive_targets"] == 2 for value in offsets.values())
    paired = merged["paired_target_contributions"]
    assert paired["targets"] == 6
    assert len(
        set(
            zip(
                paired["record_ids"],
                paired["suffix_offsets"],
                strict=True,
            )
        )
    ) == 6
    assert len(paired["all_reuse"]["ranks"]) == 6
    assert len(paired["all_exact"]["sampled_cross_entropy"]) == 6

    sparse_batch = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }
    sparse_batch["train_mask"][:, 5] = False
    sparse_targets, sparse_valid = build_next_item_targets(
        sparse_batch["item_ids"],
        sparse_batch["lengths"],
        sparse_batch["labels"],
        sparse_batch["train_mask"],
    )
    sparse_candidates = fixed_candidate_ids(
        sparse_targets[sparse_valid],
        spec.num_prediction_items,
        99,
        41,
    )
    sparse_diagnostic = evaluate_quality_batch(
        dense,
        embedding,
        sparse_batch,
        sparse_candidates,
        old_fp16,
        None,
        history_end=3,
        device=torch.device("cpu"),
        timing_repeats=1,
        mixed_exact_record_ids=frozenset(),
        methods=REUSE_EXACT_METHODS,
        suffix_offset_breakdown=True,
        common_cache_storage_dtype=torch.float16,
    )
    sparse_merged = merge_batch_reports([sparse_diagnostic])
    for method in REUSE_EXACT_METHODS:
        empty = sparse_merged["methods"][method][
            "recommendation_by_suffix_offset"
        ]["3"]
        assert empty["positive_targets"] == 0
        assert empty["sampled_cross_entropy"] is None
