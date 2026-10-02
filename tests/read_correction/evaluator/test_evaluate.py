"""Zero corrections preserve original saved controls across causal updates."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from hstu_kvcache.evaluation import append_timestamp_group, materialize_state, timestamp_groups
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.read_correction import QueryCorrection, HistoryCorrection
from hstu_kvcache.training import FoundationHistoryIndex, collate_foundation_batch
from read_correction_2026_09.adjudicate import ordered, validate_scores
from read_correction_2026_09.cost import CostModel
from read_correction_2026_09.evaluate import score_unit


@torch.inference_mode()
def test_zero_control_and_ephemeral_correction(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.set_num_threads(1)
    models = []
    for seed in (17, 19):
        torch.manual_seed(seed)
        models.append(HSTU(HSTUConfig(num_items=30, num_behaviors=3, hidden_size=32,
            num_heads=1, num_layers=2, max_seq_len=4, input_dropout=0)).eval())
    parent, current = models
    events = [(1, 1, 1), (4, 3, 1), (10, 8, 2), (10, 2, 1),
              (20, 7, 1), (20, 4, 2), (23, 9, 1)]
    history = FoundationHistoryIndex({1: tuple(np.asarray(c, dtype=np.int64) for c in zip(*events))})
    requests = [dict(request_id=f"r{i}", uid=1, query_timestamp=t, item_idx=item, label=float(i % 2))
                for i, (t, item) in enumerate(((20, 3), (20, 4), (21, 5), (24, 6)))]
    state = materialize_state(parent, [r for r in events if r[0] < 20],
                              producer_version="parent", max_length=4)
    groups = list(timestamp_groups(r for r in events if r[0] >= 20))
    cursor = 0
    for request in requests:
        t = request["query_timestamp"]
        while cursor < len(groups) and groups[cursor][0] < t:
            state = append_timestamp_group(current, state, groups[cursor][1],
                producer_version="current", max_length=4)
            cursor += 1
        request["reuse_logit"] = float(current.observe_cc_reuse(state.cache,
            torch.tensor([[request["item_idx"]]]), torch.tensor([float(t-state.last_timestamp)]))[0][0, 0])
        batch = collate_foundation_batch([{**request, "weight": 1.0}], history,
            device=torch.device("cpu"), max_history=4)
        request["full_logit"] = float(current.observe_cc_full(batch.item_ids, batch.behaviors,
            batch.time_deltas, batch.candidate_ids, batch.query_time_deltas, lengths=batch.lengths)[0][0, 0])
    zero_q = [QueryCorrection(1, 32) for _ in range(2)]
    nonzero_q = [QueryCorrection(1, 32) for _ in range(2)]
    for module in nonzero_q:
        module.bias.fill_(0.001)
    zero_h = [HistoryCorrection(1, 32, width=4, token_chunk=2) for _ in range(2)]
    rows, stats, controls = score_unit([1], {1: requests}, history, parent, current, 20,
        corrections={"query_only": {1: nonzero_q, 2: zero_q}, "history_conditioned": {1: zero_h, 2: zero_h}},
        cost_model=CostModel(32, 2, 1, "torch"), cohort_size=1, query_batch=2, verify=True,
        append_band_size=2)
    expected = {r["request_id"]: r["reuse_logit"] for r in requests}
    for method in rows:
        for row in rows[method]:
            if method == "history_conditioned" or row["budget"] == 2:
                assert row["hstu_logit"] == pytest.approx(expected[row["request_id"]], abs=2e-6)
            assert row["correction_flops"] > 0
    assert controls["requests"] == 4
    assert max(v for k, v in controls.items() if k.endswith("error")) < 2e-6
    assert sum(stats["full_history_hist"].values()) == 4
    assert sum(int(k.split(":")[1]) * v for k, v in stats["band_append_hist"].items()) == 3


def test_aggregation_rejects_missing_request_or_duplicate():
    reference = ordered(pa.Table.from_pylist([
        {"request_id": "b", "uid": 1, "query_timestamp": 2},
        {"request_id": "a", "uid": 1, "query_timestamp": 1}]))
    scores = pa.Table.from_pylist([{**r, "hstu_logit": 0.5, "correction_flops": 4, "total_flops": 4}
                                  for r in reference.to_pylist()])
    assert validate_scores(scores, reference)["request_id"].to_pylist() == ["a", "b"]
    with pytest.raises(RuntimeError, match="differ"):
        validate_scores(scores.slice(0, 1), reference)
    with pytest.raises(RuntimeError, match="duplicate"):
        ordered(pa.concat_tables([scores, scores]))
