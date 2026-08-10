from __future__ import annotations

import numpy as np
import torch

from hstu_kvcache.streaming.qk_stream_runner import (
    _training_evaluation_readiness,
)
from hstu_kvcache.streaming.qk_stream_version import (
    distributed_full_catalog_metrics,
    distributed_full_catalog_topk,
    distributed_projected_candidate_scores,
    paired_full_catalog_summary,
    paired_quality_summary,
    summarize_full_catalog_record,
    summarize_logged_window_record,
    summarize_rank_sensitivity_record,
    summarize_record_scores,
    summarize_window_topk_record,
)
from hstu_kvcache.streaming.xp_projected_edge import (
    TrainableProjectedModuloEmbedding,
)


def test_training_evaluation_readiness_only_blocks_invalid_training() -> None:
    ready = _training_evaluation_readiness(
        parameters_finite=True,
        epoch_mean_losses=[1.4],
        total_targets=82_546,
    )
    assert all(ready.values())
    assert "dense_parameter_sample_changed" not in ready
    assert "optimizer_active_embedding_rows_positive" not in ready


def test_training_evaluation_readiness_rejects_nonfinite_or_empty() -> None:
    nonfinite = _training_evaluation_readiness(
        parameters_finite=True,
        epoch_mean_losses=[float("nan")],
        total_targets=1,
    )
    empty = _training_evaluation_readiness(
        parameters_finite=True,
        epoch_mean_losses=[1.4],
        total_targets=0,
    )
    assert nonfinite["training_losses_finite"] is False
    assert empty["global_targets_positive"] is False


def test_paired_quality_gate_uses_lower_is_better_ce() -> None:
    records = [
        {
            "targets": 4,
            "reuse_loss_sum": 4.8,
            "exact_loss_sum": 4.0,
            "reuse_hit_sum": 2.0,
            "exact_hit_sum": 3.0,
            "reuse_ndcg_sum": 1.5,
            "exact_ndcg_sum": 2.0,
        }
        for _ in range(16)
    ]
    result = paired_quality_summary(
        records,
        epsilon_ce=0.005,
        bootstrap_samples=128,
        bootstrap_seed=17,
    )
    assert np.isclose(result["reuse_minus_exact_ce"], 0.2)
    assert result["existence_gate_passed"] is True
    assert result["practical_gate_passed"] is True


def test_record_score_summary_is_paired() -> None:
    reuse = torch.tensor([[0.0, 1.0, -1.0], [0.5, 0.0, -0.5]])
    exact = torch.tensor([[2.0, 1.0, -1.0], [1.5, 0.0, -0.5]])
    result = summarize_record_scores(reuse, exact)
    assert result["targets"] == 2
    assert result["reuse_loss_sum"] > result["exact_loss_sum"]
    assert result["exact_hit_sum"] >= result["reuse_hit_sum"]


def test_direct_projected_scores_match_projected_candidate_lookup() -> None:
    generator = torch.Generator().manual_seed(29)
    raw = torch.randn(8, 6, generator=generator)
    raw[0].zero_()
    projection = torch.randn(4, 6, generator=generator)
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=raw,
        projection_weight=projection,
        num_embeddings=8,
        rank=0,
        world_size=1,
    )
    hidden = torch.randn(2, 3, 4, generator=generator)
    candidates = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 1, 4]])
    scores = distributed_projected_candidate_scores(embedding, hidden, candidates, real_targets=3)
    lengths = torch.full((3,), 3, dtype=torch.int64)
    vectors = embedding(candidates, lengths)
    expected = torch.einsum("mnh,nch->mnc", hidden, vectors)
    assert torch.allclose(scores, expected, atol=1e-5, rtol=1e-5)


def test_full_catalog_metrics_match_brute_force_world_one() -> None:
    generator = torch.Generator().manual_seed(43)
    raw = torch.randn(8, 6, generator=generator)
    raw[0].zero_()
    projection = torch.randn(4, 6, generator=generator)
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=raw,
        projection_weight=projection,
        num_embeddings=8,
        rank=0,
        world_size=1,
    )
    hidden = torch.randn(2, 3, 4, generator=generator)
    positives = torch.tensor([1, 4, 6])
    nll, ranks = distributed_full_catalog_metrics(
        embedding,
        hidden,
        positives,
        3,
        num_prediction_items=6,
        item_chunk=2,
    )
    queries = torch.matmul(hidden, projection)
    scores = torch.matmul(queries, raw[1:7].t())
    positive_scores = scores.gather(
        2,
        (positives - 1).view(1, 3, 1).expand(2, -1, -1),
    ).squeeze(-1)
    expected_nll = torch.logsumexp(scores, dim=-1) - positive_scores
    expected_ranks = (scores >= positive_scores.unsqueeze(-1)).sum(dim=-1)
    assert torch.allclose(nll, expected_nll, atol=1e-5, rtol=1e-5)
    assert torch.equal(ranks, expected_ranks)


def test_full_catalog_metrics_support_one_method() -> None:
    generator = torch.Generator().manual_seed(44)
    raw = torch.randn(8, 6, generator=generator)
    raw[0].zero_()
    projection = torch.randn(4, 6, generator=generator)
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=raw,
        projection_weight=projection,
        num_embeddings=8,
        rank=0,
        world_size=1,
    )
    hidden = torch.randn(1, 3, 4, generator=generator)
    positives = torch.tensor([1, 4, 6])
    nll, ranks = distributed_full_catalog_metrics(
        embedding,
        hidden,
        positives,
        3,
        num_prediction_items=6,
        item_chunk=2,
    )
    scores = torch.matmul(torch.matmul(hidden, projection), raw[1:7].t())
    positive_scores = scores.gather(2, (positives - 1).view(1, 3, 1)).squeeze(-1)
    expected_nll = torch.logsumexp(scores, dim=-1) - positive_scores
    expected_ranks = (scores >= positive_scores.unsqueeze(-1)).sum(dim=-1)
    assert torch.allclose(nll, expected_nll, atol=1e-5, rtol=1e-5)
    assert torch.equal(ranks, expected_ranks)


def test_full_catalog_metrics_support_many_methods() -> None:
    generator = torch.Generator().manual_seed(45)
    raw = torch.randn(8, 6, generator=generator)
    raw[0].zero_()
    projection = torch.randn(4, 6, generator=generator)
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=raw,
        projection_weight=projection,
        num_embeddings=8,
        rank=0,
        world_size=1,
    )
    hidden = torch.randn(7, 3, 4, generator=generator)
    positives = torch.tensor([1, 4, 6])
    nll, ranks = distributed_full_catalog_metrics(
        embedding,
        hidden,
        positives,
        3,
        num_prediction_items=6,
        item_chunk=2,
    )
    scores = torch.matmul(torch.matmul(hidden, projection), raw[1:7].t())
    positive_scores = scores.gather(2, (positives - 1).view(1, 3, 1).expand(7, -1, -1)).squeeze(-1)
    expected_nll = torch.logsumexp(scores, dim=-1) - positive_scores
    expected_ranks = (scores >= positive_scores.unsqueeze(-1)).sum(dim=-1)
    assert torch.allclose(nll, expected_nll, atol=1e-5, rtol=1e-5)
    assert torch.equal(ranks, expected_ranks)


def test_full_catalog_topk_matches_brute_force_world_one() -> None:
    generator = torch.Generator().manual_seed(47)
    raw = torch.randn(9, 6, generator=generator)
    raw[0].zero_()
    projection = torch.randn(4, 6, generator=generator)
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=raw,
        projection_weight=projection,
        num_embeddings=9,
        rank=0,
        world_size=1,
    )
    hidden = torch.randn(2, 4, generator=generator)
    scores, ids = distributed_full_catalog_topk(
        embedding,
        hidden,
        num_prediction_items=8,
        maximum_k=4,
        item_chunk=3,
    )
    expected_scores = torch.matmul(
        torch.matmul(hidden, projection),
        raw[1:9].t(),
    )
    brute_scores, positions = torch.topk(expected_scores, 4, dim=1)
    assert torch.allclose(scores, brute_scores, atol=1e-5, rtol=1e-5)
    assert torch.equal(ids, positions + 1)


def test_stream_sensitive_metrics_report_wins_flips_and_window_quality() -> None:
    nll = torch.tensor([[3.0, 1.0, 2.0], [2.0, 1.5, 1.0]])
    ranks = torch.tensor([[20, 5, 100], [8, 7, 50]])
    sensitivity = summarize_rank_sensitivity_record(
        nll,
        ranks,
        cutoffs=(10, 50),
    )
    assert sensitivity["recompute_nll_wins"] == 2
    assert sensitivity["reuse_nll_wins"] == 1
    assert sensitivity["recompute_rank_wins"] == 2
    assert sensitivity["reuse_rank_wins"] == 1
    assert sensitivity["recompute_rescues_at_10"] == 1
    assert sensitivity["recompute_regressions_at_10"] == 0
    topk = torch.tensor([[4, 2, 8, 7], [2, 3, 4, 8]])
    window = summarize_window_topk_record(
        topk,
        torch.tensor([2, 3]),
        cutoffs=(2, 4),
    )
    assert window["reuse_recall_at_2"] == 0.5
    assert window["recompute_recall_at_2"] == 1.0
    assert window["recompute_hit_at_2"] == 1
    logged = summarize_logged_window_record(
        torch.tensor([[0.9, 0.1, 0.8], [0.9, 0.8, 0.7]]),
        torch.tensor([1, 0, 1]),
    )
    assert logged["reuse_auc"] == 1.0
    assert logged["recompute_auc"] == 0.5


def test_full_catalog_summary_reports_oriented_paired_gaps() -> None:
    nll = torch.tensor([[2.0, 3.0], [1.0, 2.0]])
    ranks = torch.tensor([[20, 5], [2, 1]])
    record = summarize_full_catalog_record(nll, ranks)
    records = [{**record, "record": index} for index in range(16)]
    result = paired_full_catalog_summary(
        records,
        bootstrap_samples=128,
        bootstrap_seed=53,
    )
    assert np.isclose(result["gaps"]["cross_entropy"]["absolute"], 1.0)
    assert result["gaps"]["cross_entropy"]["positive_direction_with_ci"] is True
    assert result["gaps"]["ndcg_at_10"]["positive_direction_with_ci"] is True
    assert result["gaps"]["mrr"]["absolute"] > 0
    assert result["gaps"]["perplexity"]["penalty_percent"] > 0
