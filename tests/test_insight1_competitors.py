import numpy as np
import torch
from design.competitor_probe import evaluate_batch, path_records, summarize_scores
from insight_one_locality.adjudicate import metrics_for_path
from insight_one_locality.common import score_cache_chunked

from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.baselines.kv_translate import fit
from hstu_kvcache.baselines.tail_recompute import recompute_tail
from hstu_kvcache.models import HSTU, HSTUConfig


def _model(seed):
    torch.manual_seed(seed)
    return HSTU(HSTUConfig(
        num_items=32, num_behaviors=3, hidden_size=12, num_layers=6,
        num_heads=3, max_seq_len=32, input_dropout=0.2, attn_dropout=0.1,
    )).eval()


def _events(offset=0):
    items = torch.tensor([[1, 4, 2, 8, 3, 6, 7, 9], [7, 3, 9, 4, 8, 2, 1, 5]]) + offset
    behaviors = torch.tensor([[1, 2, 1, 2, 2, 1, 1, 2], [2, 1, 2, 1, 1, 2, 2, 1]])
    deltas = torch.tensor([[0., 7., 11., 2., 19., 3., 5., 13.], [0., 3., 7., 1., 5., 9., 2., 6.]])
    return items, behaviors, deltas


def test_executable_paths_match_direct_calls_and_share_unchanged_parent_snapshot(monkeypatch):
    parent, current = _model(17), _model(23)
    # Fitting uses separate explicit synthetic histories, before evaluation.
    calibration = _events(10)
    translator = fit(parent.compute_kv(*calibration), current.compute_kv(*calibration),
                     num_heads=3, k=1, ridge=0.01)
    translators = {"kt_k1": translator}
    raw = _events()
    candidates = torch.arange(12, 24).repeat(2, 1)
    query_deltas = torch.tensor([4., 7.])
    state = lr.capture_state(parent, *raw)
    intervals, tails = [(1, 2), (4, 5)], [2, 4]
    with torch.no_grad():
        def score(cache):
            return score_cache_chunked(current, cache, candidates, query_deltas, 4)

        expected = {"reuse": score(state.cache), "current_exact": score(current.compute_kv(*raw))}
        for start, end in intervals:
            expected[f"lr_{start}_{end}"] = score(lr.recompute_interval(current, state, *raw, (start, end)).cache)
        for n in tails:
            expected[f"tr_{n}"] = score(recompute_tail(current, state.cache, *raw, n))
        expected["kt_k1"] = score(translator.apply(state.cache))

    real_capture = lr.capture_state
    captures = []

    def capture(*args):
        result = real_capture(*args)
        captures.append((result, result.cache.k.clone(), result.cache.v.clone(),
                         {layer: value.clone() for layer, value in result.boundary_inputs.items()}))
        return result

    monkeypatch.setattr(lr, "capture_state", capture)
    weights_before = translator.k_weight.clone(), translator.v_weight.clone()
    parent.train()
    current.train()
    actual = evaluate_batch(parent, current, *raw, candidates, query_deltas,
                            layer_intervals=intervals, tail_lengths=tails,
                            translators=translators, candidate_chunk=4)
    assert parent.training and current.training
    assert list(actual) == list(expected)
    for config_id in expected:
        torch.testing.assert_close(actual[config_id], expected[config_id])
        assert not actual[config_id].requires_grad
    assert len(captures) == 1
    captured, keys, values, boundaries = captures[0]
    torch.testing.assert_close(captured.cache.k, keys, rtol=0, atol=0)
    torch.testing.assert_close(captured.cache.v, values, rtol=0, atol=0)
    for layer in boundaries:
        torch.testing.assert_close(captured.boundary_inputs[layer], boundaries[layer], rtol=0, atol=0)
    torch.testing.assert_close(translator.k_weight, weights_before[0], rtol=0, atol=0)
    torch.testing.assert_close(translator.v_weight, weights_before[1], rtol=0, atol=0)
    records = path_records(intervals, tails, translators, history_length=8, num_layers=6)
    assert [record["config_id"] for record in records] == list(actual)
    by_id = {record["config_id"]: record for record in records}
    assert by_id["lr_1_2"]["kv_updated_fraction"] == 2 / 6
    assert by_id["tr_2"]["kv_updated_fraction"] == 2 / 8
    assert by_id["kt_k1"]["kv_updated_fraction"] == 1
    assert all("cost" not in record for record in records)


def _logits(probabilities):
    return np.log(probabilities / (1 - probabilities))


def test_probability_recovery_is_ratio_of_means_retains_negative_and_matches_old_metrics():
    exact = np.broadcast_to(np.linspace(0.3, 0.6, 12), (2, 12)).copy()
    reuse = exact + np.array([[0.1], [0.3]])
    recovered = exact + 0.2
    worse = exact + 0.35
    scores = {"reuse": _logits(reuse), "current_exact": _logits(exact),
              "tr_1": _logits(recovered), "tr_2": _logits(worse)}
    records = path_records([], [1, 2], {}, history_length=8, num_layers=6)
    rows = summarize_scores(scores, "synthetic_edge", records)
    by_id = {row["config_id"]: row for row in rows}
    np.testing.assert_allclose(by_id["tr_1"]["probability_gap_recovery"], 0, atol=1e-14)
    np.testing.assert_allclose(by_id["tr_2"]["probability_gap_recovery"], -0.75, atol=1e-14)
    # Mean per-user recovery would instead be -1/3, demonstrating the distinction.
    assert not np.isclose(by_id["tr_1"]["probability_gap_recovery"], np.mean([1 - 0.2 / 0.1, 1 - 0.2 / 0.3]))
    stacked = np.stack(list(scores.values()), axis=1)
    legacy_configs = {record["config_id"]: {"family": record["family"], "budget": "synthetic", "cost": 0}
                      for record in records[2:]}
    for index, config_id in enumerate(scores):
        reference = metrics_for_path("synthetic_edge", config_id, index, stacked, legacy_configs)
        for key in ("mean_abs_probability_gap", "reuse_mean_abs_probability_gap", "probability_gap_recovery",
                    "mean_abs_logit_gap", "mean_Bernoulli_JS", "top1_agreement", "top10_overlap", "rank_correlation"):
            np.testing.assert_allclose(by_id[config_id][key], reference[key], rtol=0, atol=1e-14)
    assert all("cost" not in row for row in rows)


def test_near_zero_reuse_gap_is_undefined_without_dropping_rows():
    exact = np.broadcast_to(np.linspace(-1, 1, 12), (2, 12)).copy()
    scores = {"reuse": exact + 1e-14, "current_exact": exact, "tr_1": exact + 0.5}
    records = path_records([], [1], {}, history_length=8, num_layers=6)
    rows = summarize_scores(scores, "no_gap_edge", records)
    assert len(rows) == len(scores)
    assert all(row["probability_gap_recovery"] is None for row in rows)
    assert all(row["probability_gap_recovery_status"] == "near_zero_reuse_gap" for row in rows)
    assert rows[-1]["mean_abs_probability_gap"] > 0
