"""Real native replay with maintained summaries and repeated candidate owners."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from design_one.scoring import ViewObserver
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design_one import ProducerSummary, SharedReadAdapter, SummaryProjection
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.training import FoundationHistoryIndex
from selective_recompute_2026_09.evaluate import snapshots


def _stats():
    return {key: Counter() for key in ("initial_history_hist", "full_history_hist")}


@torch.inference_mode()
def test_observer_preserves_native_replay_and_tracks_each_actual_cache(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(1795)
    config = HSTUConfig(num_items=32, num_behaviors=3, hidden_size=8, num_layers=2,
                        num_heads=2, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    parent, current = HSTU(config).eval(), HSTU(config).eval()
    events = {
        1: [(1, 1, 1), (2, 2, 2), (3, 3, 1), (4, 4, 2),
            (20, 5, 1), (20, 6, 2), (22, 7, 1), (25, 8, 2)],
        2: [(1, 9, 1), (2, 10, 2), (3, 11, 1), (4, 12, 2),
            (20, 13, 1), (22, 14, 1), (22, 15, 2), (25, 16, 2)],
        3: [(1, 1, 1), (2, 2, 2), (20, 3, 1), (22, 4, 2), (22, 5, 1)],
    }
    times = {1: [20, 20, 21, 22, 26], 2: [20, 23, 23, 26], 3: [20, 21, 23]}
    requests = {uid: [dict(request_id=f"{uid}:{i}", uid=uid, query_timestamp=t,
                           item_idx=17 + i, label=float(i % 2))
                      for i, t in enumerate(values)] for uid, values in times.items()}
    history = FoundationHistoryIndex({uid: tuple(np.asarray(column, dtype=np.int64)
                                                 for column in zip(*rows))
                                      for uid, rows in events.items()})
    # Identity coordinates intentionally exercise sensitivity to every feature,
    # including fractions and writes after the release snapshot.
    dimension = 2 * (2 * 2 * 8 + 2) + 1
    projection = SummaryProjection(torch.zeros(dimension), torch.ones(dimension),
                                   torch.eye(dimension))
    parameters = [dict(weights=torch.randn(dimension + 1, 2, 5, 4) * .0005,
                       query_center=torch.zeros(1, 2, 1, 4),
                       query_scale=torch.ones(1, 2, 1, 4),
                       read_weights=torch.randn(2, 8, 4) * .0005,
                       read_center=torch.zeros(8), read_scale=torch.ones(8)) for _ in range(2)]
    adapters = {mode: SharedReadAdapter(projection, parameters, max_length=4, summary_mode=mode)
                for mode in ("producer_mean", "producer_mass")}
    for cohort in ([1, 2], [3]):
        plain = list(snapshots(cohort, requests, history, parent, current, 20,
                               max_length=4, query_batch=2, stats=_stats(), append_band_size=4))
        observer = ViewObserver([dict(tag=mode, adapter=adapter) for mode, adapter in adapters.items()])
        expected_refreshes = 0
        seen_state = {}
        for plain_snap, snap in zip(plain, snapshots(
                cohort, requests, history, parent, current, 20, max_length=4,
                query_batch=2, stats=_stats(), append_band_size=4, observer=observer), strict=True):
            torch.testing.assert_close(snap.state.cache.k, plain_snap.state.cache.k, atol=0, rtol=0)
            torch.testing.assert_close(snap.state.cache.v, plain_snap.state.cache.v, atol=0, rtol=0)
            native = current.observe_cc_reuse(snap.state.cache, snap.candidates, snap.query_deltas)[0]
            plain_native = current.observe_cc_reuse(plain_snap.state.cache, plain_snap.candidates,
                                                     plain_snap.query_deltas)[0]
            torch.testing.assert_close(native, plain_native, atol=0, rtol=0)
            writes = [sum(20 <= event[0] < row["query_timestamp"] for event in events[row["uid"]])
                      for row in snap.requests]
            length = snap.state.cache.seq_len
            lineage = [[4] * max(0, length - count) + [5] * min(length, count) for count in writes]
            fresh = ProducerSummary.from_cache(snap.state.cache, lineage, max_length=4)
            fresh.native_writes.copy_(torch.tensor(writes))
            owners = [cohort.index(row["uid"]) for row in snap.requests]
            maintained = observer.summary.select(owners)
            torch.testing.assert_close(maintained.features(), fresh.features(), atol=2e-7, rtol=2e-5)
            for mode, adapter in adapters.items():
                expected_view = adapter.prepare(fresh)
                actual_score = score(current, snap.state.cache, snap.candidates, snap.query_deltas,
                                     history_override=snap.context[mode])[0]
                reference_score = score(current, snap.state.cache, snap.candidates, snap.query_deltas,
                                        history_override=expected_view)[0]
                torch.testing.assert_close(actual_score, reference_score, atol=2e-7, rtol=2e-5)
            for row, count in zip(snap.requests, writes, strict=True):
                uid = row["uid"]
                if seen_state.get(uid) != count:
                    expected_refreshes += 1
                    seen_state[uid] = count
        one_user = observer.summary.select([0])
        for mode, adapter in adapters.items():
            costs = adapter.estimate_flops(batch=expected_refreshes)
            expected = (costs["summary_projection"] + costs["view_generation"]
                        + expected_refreshes * one_user.estimate_flops("features"))
            assert observer.costs[mode]["view_flops"] == expected
        # The short initial state performs a native append without an eviction.
        assert one_user.estimate_flops("evict", length=0) == 0
